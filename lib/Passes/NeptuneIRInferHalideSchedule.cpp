// Infer deterministic Halide schedule from apply.
#include "Dialect/NeptuneIR/NeptuneIRDialect.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include "Passes/NeptuneIRPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <cstdint>

using namespace mlir;

namespace mlir::Neptune::NeptuneIR {
#define GEN_PASS_DEF_NEPTUNEIRINFERHALIDESCHEDULEPASS
#include "Passes/NeptuneIRPasses.h.inc"
} // namespace mlir::Neptune::NeptuneIR

namespace {
using namespace mlir::Neptune::NeptuneIR;

struct ScheduleConfig {
  int64_t l1Bytes = 32 * 1024;
  double alpha = 0.6;
  int64_t threads = 1;
  int64_t vectorWidth = 1;
};

static int64_t getEnvInt(llvm::StringRef name, int64_t defaultValue) {
  if (auto val = llvm::sys::Process::GetEnv(name)) {
    int64_t parsed = 0;
    if (llvm::to_integer(*val, parsed, 10))
      return parsed;
  }
  return defaultValue;
}

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

static int64_t pickThreads() {
  // Deterministic order: NEPTUNECC_THREADS > HL_NUM_THREADS > OMP_NUM_THREADS.
  int64_t threads = getEnvInt("NEPTUNECC_THREADS", 0);
  if (threads > 0)
    return threads;
  threads = getEnvInt("HL_NUM_THREADS", 0);
  if (threads > 0)
    return threads;
  threads = getEnvInt("OMP_NUM_THREADS", 0);
  if (threads > 0)
    return threads;
  return 1;
}

static int64_t pickVectorWidth(bool &fromEnv) {
  // Vector width is user-controlled; default is scalar if not specified.
  int64_t vl = getEnvInt("NEPTUNECC_VECTOR_WIDTH", 0);
  if (vl > 0) {
    fromEnv = true;
    return vl;
  }
  fromEnv = false;
  return 1;
}

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

static int64_t product(llvm::ArrayRef<int64_t> values) {
  int64_t prod = 1;
  for (int64_t v : values)
    prod *= v;
  return prod;
}

static int64_t clampTile(int64_t cand, int64_t extent, int64_t vec) {
  int64_t t = cand < extent ? cand : extent;
  if (t < 1)
    return 0;
  if (vec > 1) {
    t = (t / vec) * vec;
    if (t < 1)
      return 0;
  }
  return t;
}

static int64_t computeFootprint(int64_t elemBytes,
                                llvm::ArrayRef<int64_t> tile,
                                llvm::ArrayRef<int64_t> radius) {
  // Approximate footprint: input halo + output tile.
  int64_t haloProd = 1;
  int64_t tileProd = 1;
  for (size_t i = 0, e = tile.size(); i < e; ++i) {
    haloProd *= (tile[i] + 2 * radius[i]);
    tileProd *= tile[i];
  }
  return elemBytes * (haloProd + tileProd);
}

static llvm::SmallVector<int64_t, 3>
chooseTile1D(int64_t extentFast, int64_t vec) {
  llvm::SmallVector<int64_t, 3> candidates;
  candidates.push_back(vec * 4);
  candidates.push_back(vec * 8);
  candidates.push_back(vec * 16);

  for (int64_t &c : candidates)
    c = clampTile(c, extentFast, vec);

  return candidates;
}

static llvm::SmallVector<std::pair<int64_t, int64_t>, 8>
chooseTile2D(int64_t extentFast, int64_t extentSlow, int64_t vec) {
  llvm::SmallVector<int64_t, 3> txCandidates = {vec * 4, vec * 8, vec * 16};
  llvm::SmallVector<int64_t, 4> tyCandidates = {8, 16, 32, 64};
  llvm::SmallVector<std::pair<int64_t, int64_t>, 8> out;

  for (int64_t tx : txCandidates) {
    int64_t ttx = clampTile(tx, extentFast, vec);
    if (!ttx)
      continue;
    for (int64_t ty : tyCandidates) {
      int64_t tty = clampTile(ty, extentSlow, 1);
      if (!tty)
        continue;
      out.emplace_back(ttx, tty);
    }
  }
  return out;
}

static llvm::SmallVector<llvm::SmallVector<int64_t, 3>, 8>
chooseTile3D(int64_t extentFast, int64_t extentMid, int64_t extentSlow,
             int64_t vec) {
  llvm::SmallVector<int64_t, 2> txCandidates = {vec * 4, vec * 8};
  llvm::SmallVector<int64_t, 3> tyCandidates = {4, 8, 16};
  llvm::SmallVector<int64_t, 3> tzCandidates = {2, 4, 8};
  llvm::SmallVector<llvm::SmallVector<int64_t, 3>, 8> out;

  for (int64_t tx : txCandidates) {
    int64_t ttx = clampTile(tx, extentFast, vec);
    if (!ttx)
      continue;
    for (int64_t ty : tyCandidates) {
      int64_t tty = clampTile(ty, extentMid, 1);
      if (!tty)
        continue;
      for (int64_t tz : tzCandidates) {
        int64_t ttz = clampTile(tz, extentSlow, 1);
        if (!ttz)
          continue;
        out.push_back({ttx, tty, ttz});
      }
    }
  }
  return out;
}

static bool inferScheduleForApply(func::FuncOp func, ApplyOp apply) {
  MLIRContext *ctx = apply.getContext();
  Location loc = apply.getLoc();

  auto bounds = apply.getBounds();
  auto lb = bounds.getLb().asArrayRef();
  auto ub = bounds.getUb().asArrayRef();
  if (lb.empty() || lb.size() != ub.size()) {
    apply.emitOpError("invalid bounds for schedule inference");
    return false;
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

  llvm::SmallVector<int64_t, 4> radius(rank, 1);
  bool radiusFromAttr = false;
  if (auto radiusAttr = apply.getRadius()) {
    if (radiusAttr->size() == rank) {
      radius.assign(radiusAttr->begin(), radiusAttr->end());
      radiusFromAttr = true;
    }
  }

  // Prefer shape from input temp bounds; fall back to outExtent if missing.
  llvm::SmallVector<int64_t, 4> shape;
  if (!apply.getInputs().empty()) {
    if (auto tempTy = dyn_cast<TempType>(apply.getInputs().front().getType())) {
      auto boundsAttr = tempTy.getBounds();
      auto sb = boundsAttr.getLb().asArrayRef();
      auto su = boundsAttr.getUb().asArrayRef();
      if (sb.size() == su.size() && sb.size() == rank) {
        shape.reserve(rank);
        for (size_t i = 0; i < rank; ++i)
          shape.push_back(su[i] - sb[i]);
      }
    }
  }
  if (shape.empty())
    shape = outExtent;

  auto tempTy = dyn_cast<TempType>(apply.getResult().getType());
  if (!tempTy) {
    emitError(loc) << "apply result is not TempType";
    return false;
  }
  Type elemTy = tempTy.getElementType();
  int64_t elemBits = elemTy.getIntOrFloatBitWidth();
  int64_t elemBytes = elemBits > 0 ? (elemBits + 7) / 8 : 4;

  ScheduleConfig cfg;
  cfg.l1Bytes = getEnvInt("NEPTUNECC_L1_BYTES", cfg.l1Bytes);
  cfg.alpha = getEnvDouble("NEPTUNECC_CACHE_ALPHA", cfg.alpha);
  cfg.threads = pickThreads();
  bool vectorFromEnv = false;
  cfg.vectorWidth = pickVectorWidth(vectorFromEnv);

  int64_t extentFast = outExtent[rank - 1];
  int64_t vec = cfg.vectorWidth;
  bool vectorDisabled = false;
  if (vec < 1)
    vec = 1;
  // Do not vectorize if the fast extent is smaller than the vector width.
  if (extentFast < vec) {
    vec = 1;
    vectorDisabled = true;
  }

  llvm::SmallVector<int64_t, 4> tile(rank, 1);
  bool anyFit = false;
  int64_t bestVolume = -1;
  int64_t bestFootprint = 0;

  double limit = cfg.alpha * static_cast<double>(cfg.l1Bytes);

  if (rank == 1) {
    auto candidates = chooseTile1D(extentFast, vec);
    // Pick the largest candidate that fits the cache footprint (or the best
    // available one if none fit).
    for (int64_t tx : candidates) {
      if (tx < 1)
        continue;
      llvm::SmallVector<int64_t, 1> t = {tx};
      int64_t footprint = computeFootprint(elemBytes, t, radius);
      int64_t volume = product(t);
      if (static_cast<double>(footprint) <= limit) {
        anyFit = true;
        if (volume > bestVolume) {
          bestVolume = volume;
          bestFootprint = footprint;
          tile[0] = tx;
        }
      } else if (!anyFit && volume > bestVolume) {
        bestVolume = volume;
        bestFootprint = footprint;
        tile[0] = tx;
      }
    }
  } else if (rank == 2) {
    int64_t extentSlow = outExtent[rank - 2];
    auto candidates = chooseTile2D(extentFast, extentSlow, vec);
    // Candidates are in (fast, slow) order; store as [y, x] tile.
    for (auto pair : candidates) {
      llvm::SmallVector<int64_t, 2> t = {pair.second, pair.first};
      int64_t footprint = computeFootprint(elemBytes, t, radius);
      int64_t volume = product(t);
      if (static_cast<double>(footprint) <= limit) {
        anyFit = true;
        if (volume > bestVolume) {
          bestVolume = volume;
          bestFootprint = footprint;
          tile[0] = t[0];
          tile[1] = t[1];
        }
      } else if (!anyFit && volume > bestVolume) {
        bestVolume = volume;
        bestFootprint = footprint;
        tile[0] = t[0];
        tile[1] = t[1];
      }
    }
  } else if (rank == 3) {
    int64_t extentMid = outExtent[rank - 2];
    int64_t extentSlow = outExtent[rank - 3];
    auto candidates = chooseTile3D(extentFast, extentMid, extentSlow, vec);
    // Candidates are in (fast, mid, slow) order; store as [z, y, x] tile.
    for (auto &cand : candidates) {
      llvm::SmallVector<int64_t, 3> t = {cand[2], cand[1], cand[0]};
      int64_t footprint = computeFootprint(elemBytes, t, radius);
      int64_t volume = product(t);
      if (static_cast<double>(footprint) <= limit) {
        anyFit = true;
        if (volume > bestVolume) {
          bestVolume = volume;
          bestFootprint = footprint;
          tile[0] = t[0];
          tile[1] = t[1];
          tile[2] = t[2];
        }
      } else if (!anyFit && volume > bestVolume) {
        bestVolume = volume;
        bestFootprint = footprint;
        tile[0] = t[0];
        tile[1] = t[1];
        tile[2] = t[2];
      }
    }
  } else {
    apply.emitOpError("schedule inference supports rank 1/2/3 only");
    return false;
  }

  if (bestVolume < 0)
    tile = outExtent;

  int64_t yIndex = (rank >= 2) ? static_cast<int64_t>(rank - 2) : -1;
  int64_t yExtent = (yIndex >= 0) ? outExtent[static_cast<size_t>(yIndex)] : 0;
  int64_t yTile = (yIndex >= 0) ? tile[static_cast<size_t>(yIndex)] : 0;
  if (yTile > yExtent)
    yTile = yExtent;

  llvm::StringRef parDim = "none";
  if (cfg.threads > 1) {
    // Parallelize the slowest useful outer tile dimension.
    if (rank == 2) {
      int64_t extentY = outExtent[0];
      if (extentY >= tile[0] * 2)
        parDim = "y";
    } else if (rank == 3) {
      int64_t extentZ = outExtent[0];
      int64_t extentY = outExtent[1];
      if (extentZ >= tile[0] * 2)
        parDim = "z";
      else if (extentY >= tile[1] * 2)
        parDim = "y";
    }
  }

  int64_t unroll = 1;
  llvm::StringRef unrollDim = "y";
  llvm::StringRef unrollReason = "";
  if (rank >= 2) {
    // Only unroll the inner-y loop when it divides evenly.
    if (yTile < 2) {
      unrollReason = "yi<2";
    } else if (yTile % 2 != 0) {
      unrollReason = "yi not divisible";
    } else {
      int64_t maxU = (rank == 3) ? 2 : 4;
      int64_t footprint = computeFootprint(elemBytes, tile, radius);
      bool allow4 = (maxU >= 4) && (yTile >= 4) && (vec <= 4) &&
                    (yTile % 4 == 0) &&
                    (static_cast<double>(footprint) <= limit);
      if (allow4) {
        unroll = 4;
      } else {
        unroll = 2;
      }
      if (rank == 3 && unroll > 2) {
        unroll = 2;
        unrollReason = "3D conservative cap";
      }
      if (unroll == 2 && (yTile % 2 != 0)) {
        unroll = 1;
        unrollReason = "yi not divisible";
      }
    }
  } else {
    unrollDim = "none";
  }

  NamedAttrList scheduleAttrs;
  // Persist schedule for the emitter; do not bake logic into EmitC.
  scheduleAttrs.set("split", DenseI64ArrayAttr::get(ctx, tile));
  scheduleAttrs.set("vec", IntegerAttr::get(IntegerType::get(ctx, 64), vec));
  scheduleAttrs.set("par_dim", StringAttr::get(ctx, parDim));
  scheduleAttrs.set("unroll",
                    IntegerAttr::get(IntegerType::get(ctx, 64), unroll));
  scheduleAttrs.set("unroll_dim", StringAttr::get(ctx, unrollDim));
  scheduleAttrs.set(
      "unroll_factor",
      IntegerAttr::get(IntegerType::get(ctx, 64), unroll));
  scheduleAttrs.set(
      "threads",
      IntegerAttr::get(IntegerType::get(ctx, 64), cfg.threads));
  apply->setAttr("neptune.schedule", DictionaryAttr::get(ctx, scheduleAttrs));

  llvm::outs() << "neptune-cc: schedule for '" << func.getSymName()
               << "' rank=" << rank << " shape=";
  printArray(llvm::outs(), shape);
  llvm::outs() << " outMin=";
  printArray(llvm::outs(), outMin);
  llvm::outs() << " outExtent=";
  printArray(llvm::outs(), outExtent);
  llvm::outs() << " radius=";
  printArray(llvm::outs(), radius);
  llvm::outs() << " elem_bytes=" << elemBytes;
  llvm::outs() << " split=";
  printArray(llvm::outs(), tile);
  llvm::outs() << " vec=" << vec << " par=" << parDim
               << " unroll=" << unroll;
  if (!radiusFromAttr)
    llvm::outs() << " (radius defaulted)";
  if (!vectorFromEnv)
    llvm::outs() << " (VL defaulted: set NEPTUNECC_VECTOR_WIDTH for target VL)";
  if (vectorDisabled)
    llvm::outs() << " (vectorize disabled: extent < VL)";
  if (!anyFit)
    llvm::outs() << " (cache fit not found; using largest candidate, footprint="
                 << bestFootprint << ")";
  if (unroll == 1 && !unrollReason.empty())
    llvm::outs() << " (unroll disabled: " << unrollReason << ")";
  llvm::outs() << "\n";

  return true;
}

struct NeptuneIRInferHalideSchedulePass final
    : mlir::Neptune::NeptuneIR::impl::
          NeptuneIRInferHalideSchedulePassBase<
              NeptuneIRInferHalideSchedulePass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    llvm::SmallVector<ApplyOp, 4> applies;
    func.walk([&](ApplyOp op) { applies.push_back(op); });
    if (applies.empty())
      return;

    for (ApplyOp apply : applies) {
      if (!inferScheduleForApply(func, apply)) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
