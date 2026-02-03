// Infer overlap strategy from apply bounds and a simple roofline/halo model.
#include "Dialect/NeptuneIR/NeptuneIRDialect.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include "Passes/NeptuneIRPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>

using namespace mlir;

namespace mlir::Neptune::NeptuneIR {
#define GEN_PASS_DEF_NEPTUNEIRINFEROVERLAPSTRATEGYPASS
#include "Passes/NeptuneIRPasses.h.inc"
} // namespace mlir::Neptune::NeptuneIR

namespace {
using namespace mlir::Neptune::NeptuneIR;

// Configurable model parameters (env overrides control them).
struct StrategyConfig {
  double bwMem = 50e9;   // bytes/s
  double peak = 100e9;   // flops/s
  double bwNet = 10e9;   // bytes/s
  double netLat = 5e-6;  // seconds
  int64_t minPoints = 4096;
  int64_t callOverheadNs = 200;
  double splitOverhead = 0.0; // seconds per element
};

// Parses an integer env var; returns default on missing/invalid.
static int64_t getEnvInt(llvm::StringRef name, int64_t defaultValue) {
  if (auto val = llvm::sys::Process::GetEnv(name)) {
    int64_t parsed = 0;
    if (llvm::to_integer(*val, parsed, 10))
      return parsed;
  }
  return defaultValue;
}

// Parses a double env var; returns default on missing/invalid.
static double getEnvDouble(llvm::StringRef name, double defaultValue) {
  if (auto val = llvm::sys::Process::GetEnv(name)) {
    if (!val->empty()) {
      const std::string &tmp = *val;
      char *end = nullptr;
      double parsed = std::strtod(tmp.c_str(), &end);
      if (end != tmp.c_str())
        return parsed;
    }
  }
  return defaultValue;
}

// Multiplies extents into a volume.
static int64_t product(llvm::ArrayRef<int64_t> vals) {
  int64_t prod = 1;
  for (int64_t v : vals)
    prod *= v;
  return prod;
}

// Parses a single port_map entry to detect ghosted inputs.
static bool parsePortMapEntry(llvm::StringRef entry, bool &isInput,
                              bool &isGhosted) {
  auto eq = entry.split('=');
  if (eq.second.empty())
    return false;
  llvm::SmallVector<llvm::StringRef, 4> fields;
  eq.second.split(fields, ':');
  if (fields.empty())
    return false;
  llvm::StringRef role = fields[0].trim();
  isInput = role.starts_with("in");
  isGhosted = false;
  if (fields.size() >= 2 && fields[1].trim() == "ghosted")
    isGhosted = true;
  return true;
}

// Counts ghosted inputs from neptunecc.port_map; defaults to 0.
static int64_t countGhostedInputs(func::FuncOp func) {
  auto portMapAttr =
      func->getAttrOfType<ArrayAttr>("neptunecc.port_map");
  if (!portMapAttr)
    return 0;
  int64_t count = 0;
  for (Attribute attr : portMapAttr) {
    auto strAttr = dyn_cast<StringAttr>(attr);
    if (!strAttr)
      continue;
    bool isInput = false;
    bool isGhosted = false;
    if (!parsePortMapEntry(strAttr.getValue(), isInput, isGhosted))
      continue;
    if (isInput && isGhosted)
      ++count;
  }
  return count;
}

// Pulls a static shape from the first memref argument.
static bool pickShape(func::FuncOp func, llvm::SmallVector<int64_t, 4> &shape) {
  for (BlockArgument arg : func.getArguments()) {
    auto memrefTy = dyn_cast<MemRefType>(arg.getType());
    if (!memrefTy)
      continue;
    if (!memrefTy.hasStaticShape())
      return false;
    shape.assign(memrefTy.getShape().begin(), memrefTy.getShape().end());
    return true;
  }
  return false;
}

// Estimates boundary volume for 1D/2D/3D face decomposition.
static int64_t computeBoundaryVolume(llvm::ArrayRef<int64_t> extent,
                                     llvm::ArrayRef<int64_t> radius) {
  size_t rank = extent.size();
  if (rank == 1) {
    int64_t e0 = extent[0];
    int64_t r0 = radius[0];
    return std::min(e0, 2 * r0);
  }
  if (rank == 2) {
    int64_t e0 = extent[0];
    int64_t e1 = extent[1];
    int64_t r0 = radius[0];
    int64_t r1 = radius[1];
    int64_t mid0 = std::max<int64_t>(0, e0 - 2 * r0);
    return 2 * r0 * e1 + 2 * r1 * mid0;
  }
  if (rank == 3) {
    int64_t e0 = extent[0];
    int64_t e1 = extent[1];
    int64_t e2 = extent[2];
    int64_t r0 = radius[0];
    int64_t r1 = radius[1];
    int64_t r2 = radius[2];
    int64_t mid0 = std::max<int64_t>(0, e0 - 2 * r0);
    int64_t mid1 = std::max<int64_t>(0, e1 - 2 * r1);
    int64_t mid2 = std::max<int64_t>(0, e2 - 2 * r2);
    return 2 * r2 * e0 * e1 + 2 * r1 * mid0 * e2 + 2 * r0 * mid1 * mid2;
  }
  return 0;
}

// Emits a compact [a,b,c] list for logs.
static void printArray(llvm::raw_ostream &os,
                       llvm::ArrayRef<int64_t> values) {
  os << '[';
  for (size_t i = 0, e = values.size(); i < e; ++i) {
    if (i)
      os << ',';
    os << values[i];
  }
  os << ']';
}

// Stores the strategy as a func attribute for downstream passes.
static void setStrategyAttr(func::FuncOp func, DictionaryAttr attr) {
  func->setAttr("neptune.strategy.overlap", attr);
}

// Prints a deterministic strategy report for paper/debugging.
static void emitStrategyLog(llvm::StringRef tag, llvm::ArrayRef<int64_t> shape,
                            llvm::ArrayRef<int64_t> outMin,
                            llvm::ArrayRef<int64_t> outExtent,
                            llvm::ArrayRef<int64_t> radius, int64_t elemBytes,
                            int64_t stencilPoints, double peak, double bwMem,
                            double bwNet, double lat, double tInt, double tBnd,
                            double tComm, double tOver, double gain,
                            llvm::StringRef mode, llvm::StringRef reason,
                            bool forced, bool radiusDefaulted) {
  llvm::outs() << "neptune-cc: overlap-strategy kernel='" << tag << "'\n";
  llvm::outs() << "  R=" << outMin.size() << " shape=";
  printArray(llvm::outs(), shape);
  llvm::outs() << " outMin=";
  printArray(llvm::outs(), outMin);
  llvm::outs() << " outExtent=";
  printArray(llvm::outs(), outExtent);
  llvm::outs() << " radius=";
  printArray(llvm::outs(), radius);
  llvm::outs() << " b=" << elemBytes << " S=" << stencilPoints;
  if (radiusDefaulted)
    llvm::outs() << " (radius defaulted)";
  llvm::outs() << "\n";
  llvm::outs() << "  Peak=" << peak << " BW_mem=" << bwMem
               << " BW_net=" << bwNet << " Lat=" << lat << "\n";
  llvm::outs() << "  T_int=" << tInt << " T_bnd=" << tBnd
               << " T_comm=" << tComm << " T_overhead=" << tOver
               << " Gain=" << gain << "\n";
  llvm::outs() << "  strategy=" << mode << " reason=" << reason
               << " forced=" << (forced ? "yes" : "no") << "\n";
}

// Builds the neptune.strategy.overlap attribute payload.
static DictionaryAttr buildStrategyAttr(
    MLIRContext *ctx, llvm::StringRef mode, llvm::StringRef reason, bool forced,
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> outMin,
    llvm::ArrayRef<int64_t> outExtent, llvm::ArrayRef<int64_t> radius,
    int64_t elemBytes, int64_t stencilPoints, int64_t flopsPerPoint,
    int64_t bytesPerPoint, int64_t outVolume, int64_t boundaryVolume,
    int64_t haloFields, int64_t haloBytes, double peak, double bwMem,
    double bwNet, double lat, double tInt, double tBnd, double tComm,
    double tOver, double tNo, double tOv, double gain, int64_t minPoints,
    int64_t callOverheadNs, double splitOverhead, bool radiusDefaulted) {
  NamedAttrList model;
  model.set("rank", IntegerAttr::get(IntegerType::get(ctx, 64),
                                     static_cast<int64_t>(outMin.size())));
  model.set("shape", DenseI64ArrayAttr::get(ctx, shape));
  model.set("out_min", DenseI64ArrayAttr::get(ctx, outMin));
  model.set("out_extent", DenseI64ArrayAttr::get(ctx, outExtent));
  model.set("radius", DenseI64ArrayAttr::get(ctx, radius));
  model.set("radius_defaulted", BoolAttr::get(ctx, radiusDefaulted));
  model.set("elem_bytes",
            IntegerAttr::get(IntegerType::get(ctx, 64), elemBytes));
  model.set("stencil_points",
            IntegerAttr::get(IntegerType::get(ctx, 64), stencilPoints));
  model.set("flops_per_point",
            IntegerAttr::get(IntegerType::get(ctx, 64), flopsPerPoint));
  model.set("bytes_per_point",
            IntegerAttr::get(IntegerType::get(ctx, 64), bytesPerPoint));
  model.set("out_volume",
            IntegerAttr::get(IntegerType::get(ctx, 64), outVolume));
  model.set("boundary_volume",
            IntegerAttr::get(IntegerType::get(ctx, 64), boundaryVolume));
  model.set("halo_fields",
            IntegerAttr::get(IntegerType::get(ctx, 64), haloFields));
  model.set("halo_bytes",
            IntegerAttr::get(IntegerType::get(ctx, 64), haloBytes));
  model.set("peak", FloatAttr::get(Float64Type::get(ctx), peak));
  model.set("bw_mem", FloatAttr::get(Float64Type::get(ctx), bwMem));
  model.set("bw_net", FloatAttr::get(Float64Type::get(ctx), bwNet));
  model.set("net_lat", FloatAttr::get(Float64Type::get(ctx), lat));
  model.set("t_int", FloatAttr::get(Float64Type::get(ctx), tInt));
  model.set("t_bnd", FloatAttr::get(Float64Type::get(ctx), tBnd));
  model.set("t_comm", FloatAttr::get(Float64Type::get(ctx), tComm));
  model.set("t_overhead", FloatAttr::get(Float64Type::get(ctx), tOver));
  model.set("t_no", FloatAttr::get(Float64Type::get(ctx), tNo));
  model.set("t_ov", FloatAttr::get(Float64Type::get(ctx), tOv));
  model.set("gain", FloatAttr::get(Float64Type::get(ctx), gain));
  model.set("min_points",
            IntegerAttr::get(IntegerType::get(ctx, 64), minPoints));
  model.set("call_overhead_ns",
            IntegerAttr::get(IntegerType::get(ctx, 64), callOverheadNs));
  model.set("split_overhead_per_elem",
            FloatAttr::get(Float64Type::get(ctx), splitOverhead));

  NamedAttrList top;
  top.set("mode", StringAttr::get(ctx, mode));
  top.set("reason", StringAttr::get(ctx, reason));
  top.set("forced", BoolAttr::get(ctx, forced));
  top.set("model", DictionaryAttr::get(ctx, model));
  return DictionaryAttr::get(ctx, top);
}

struct NeptuneIRInferOverlapStrategyPass final
    : mlir::Neptune::NeptuneIR::impl::
          NeptuneIRInferOverlapStrategyPassBase<
              NeptuneIRInferOverlapStrategyPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = func.getContext();

    // Strategy only makes sense with a single apply in the kernel.
    llvm::SmallVector<ApplyOp, 2> applies;
    func.walk([&](ApplyOp op) { applies.push_back(op); });
    if (applies.empty()) {
      return;
    }

    llvm::StringRef tag = func.getSymName();
    if (auto tagAttr = func->getAttrOfType<StringAttr>("neptunecc.tag"))
      tag = tagAttr.getValue();

    // Pull model knobs from the environment for reproducibility.
    StrategyConfig cfg;
    cfg.bwMem = getEnvDouble("NEPTUNECC_BW_MEM", cfg.bwMem);
    cfg.peak = getEnvDouble("NEPTUNECC_PEAK_FLOP", cfg.peak);
    cfg.bwNet = getEnvDouble("NEPTUNECC_BW_NET", cfg.bwNet);
    cfg.netLat = getEnvDouble("NEPTUNECC_NET_LAT", cfg.netLat);
    cfg.minPoints = getEnvInt("NEPTUNECC_MIN_POINTS", cfg.minPoints);
    cfg.callOverheadNs =
        getEnvInt("NEPTUNECC_CALL_OVERHEAD_NS", cfg.callOverheadNs);
    cfg.splitOverhead = getEnvDouble("NEPTUNECC_BOUNDARY_OVERHEAD_PER_ELEM",
                                     cfg.splitOverhead);

    // Use a static shape for the model; dynamic shapes are skipped.
    bool shapeOk = true;
    llvm::SmallVector<int64_t, 4> shape;
    shapeOk = pickShape(func, shape);

    if (applies.size() != 1 || !shapeOk) {
      llvm::StringRef reason = applies.size() != 1
                                   ? "multiple apply ops"
                                   : "dynamic shape";
      DictionaryAttr attr = buildStrategyAttr(
          ctx, "no_overlap", reason, false, shape, {}, {}, {}, 0, 0, 0, 0, 0, 0,
          0, 0, cfg.peak, cfg.bwMem, cfg.bwNet, cfg.netLat, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, cfg.minPoints, cfg.callOverheadNs, cfg.splitOverhead,
          false);
      setStrategyAttr(func, attr);
      return;
    }

    ApplyOp apply = applies.front();
    auto bounds = apply.getBounds();
    auto lb = bounds.getLb().asArrayRef();
    auto ub = bounds.getUb().asArrayRef();
    if (lb.empty() || lb.size() != ub.size()) {
      DictionaryAttr attr = buildStrategyAttr(
          ctx, "no_overlap", "invalid bounds", false, shape, {}, {}, {}, 0, 0, 0,
          0, 0, 0, 0, 0, cfg.peak, cfg.bwMem, cfg.bwNet, cfg.netLat, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, cfg.minPoints, cfg.callOverheadNs,
          cfg.splitOverhead, false);
      setStrategyAttr(func, attr);
      return;
    }

    size_t rank = lb.size();
    llvm::SmallVector<int64_t, 4> outMin;
    llvm::SmallVector<int64_t, 4> outExtent;
    outMin.reserve(rank);
    outExtent.reserve(rank);
    for (size_t i = 0; i < rank; ++i) {
      outMin.push_back(lb[i]);
      outExtent.push_back(ub[i] - lb[i]);
    }

    // Radius is required for correctness; default to 1 when missing.
    llvm::SmallVector<int64_t, 4> radius(rank, 1);
    bool radiusDefaulted = true;
    if (auto radiusAttr = apply.getRadius()) {
      if (radiusAttr->size() == rank) {
        radius.assign(radiusAttr->begin(), radiusAttr->end());
        radiusDefaulted = false;
      }
    }

    auto tempTy = dyn_cast<TempType>(apply.getResult().getType());
    int64_t elemBytes = 4;
    if (tempTy) {
      Type elemTy = tempTy.getElementType();
      int64_t elemBits = elemTy.getIntOrFloatBitWidth();
      if (elemBits > 0)
        elemBytes = (elemBits + 7) / 8;
    }

    // Only the Manhattan distance-1 stencil model is used for now.
    int64_t stencilPoints = 1 + static_cast<int64_t>(2 * rank);
    int64_t flopsPerPoint = stencilPoints - 1;
    int64_t bytesPerPoint = elemBytes * (stencilPoints + 1);

    int64_t outVolume = product(outExtent);
    int64_t boundaryVolume = computeBoundaryVolume(outExtent, radius);

    // A non-empty safe interior is required for overlap to make sense.
    bool safeInterior = true;
    for (size_t i = 0; i < rank; ++i) {
      if (outExtent[i] <= 2 * radius[i]) {
        safeInterior = false;
        break;
      }
    }

    // Halo bytes count ghosted inputs; default to 1 if no metadata exists.
    int64_t haloFields = countGhostedInputs(func);
    if (haloFields <= 0)
      haloFields = 1;
    int64_t haloBytes = 0;
    if (shape.size() == rank) {
      for (size_t d = 0; d < rank; ++d) {
        int64_t face = 1;
        for (size_t k = 0; k < rank; ++k) {
          if (k == d)
            continue;
          face *= shape[k];
        }
        haloBytes += 2 * radius[d] * face;
      }
      haloBytes *= elemBytes * haloFields;
    }

    // Roofline-style time model.
    double wInt = static_cast<double>(outVolume) *
                  static_cast<double>(flopsPerPoint);
    double qInt = static_cast<double>(outVolume) *
                  static_cast<double>(bytesPerPoint);
    double tInt = std::max(wInt / cfg.peak, qInt / cfg.bwMem);

    double wBnd = static_cast<double>(boundaryVolume) *
                  static_cast<double>(flopsPerPoint);
    double qBnd = static_cast<double>(boundaryVolume) *
                  static_cast<double>(bytesPerPoint);
    double tBnd = std::max(wBnd / cfg.peak, qBnd / cfg.bwMem);

    double tComm = cfg.netLat + static_cast<double>(haloBytes) / cfg.bwNet;
    double tOver = (cfg.callOverheadNs * 1e-9) * (1 + 2 * rank) +
                   cfg.splitOverhead * static_cast<double>(boundaryVolume);
    double tNo = tComm + tInt + tBnd;
    double tOv = std::max(tComm, tInt) + tBnd + tOver;
    double gain = tNo - tOv;

    // Decision rule: enforce safety, then apply gain/minPoints criteria.
    bool forced = false;
    std::string mode = "no_overlap";
    std::string reason = "Gain<=0";

    if (!safeInterior) {
      reason = "safe interior empty";
    } else if (!shapeOk) {
      reason = "dynamic shape";
    } else {
      bool canOverlap = safeInterior;
      std::string forceEnv;
      if (auto val = llvm::sys::Process::GetEnv("NEPTUNECC_FORCE_STRATEGY")) {
        forceEnv = *val;
      }
      if (canOverlap && !forceEnv.empty()) {
        forced = true;
        if (forceEnv == "overlap") {
          mode = "overlap_split";
          reason = "forced overlap";
        } else if (forceEnv == "none") {
          mode = "no_overlap";
          reason = "forced none";
        }
      } else if (canOverlap && outVolume >= cfg.minPoints && gain > 0.0) {
        mode = "overlap_split";
        reason = "GainPositive";
      } else if (outVolume < cfg.minPoints) {
        reason = "below min points";
      } else if (gain <= 0.0) {
        reason = "Gain<=0";
      }
    }

    DictionaryAttr attr = buildStrategyAttr(
        ctx, mode, reason, forced, shape, outMin, outExtent, radius, elemBytes,
        stencilPoints, flopsPerPoint, bytesPerPoint, outVolume, boundaryVolume,
        haloFields, haloBytes, cfg.peak, cfg.bwMem, cfg.bwNet, cfg.netLat, tInt,
        tBnd, tComm, tOver, tNo, tOv, gain, cfg.minPoints, cfg.callOverheadNs,
        cfg.splitOverhead, radiusDefaulted);
    setStrategyAttr(func, attr);

    emitStrategyLog(tag, shape, outMin, outExtent, radius, elemBytes,
                    stencilPoints, cfg.peak, cfg.bwMem, cfg.bwNet, cfg.netLat,
                    tInt, tBnd, tComm, tOver, gain, mode, reason, forced,
                    radiusDefaulted);
  }
};

} // namespace
