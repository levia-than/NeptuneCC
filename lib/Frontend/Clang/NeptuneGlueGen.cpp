// Generate Halide glue and rewrite sources.
#include "Frontend/Clang/NeptuneGlueGen.h"

#include "Dialect/NeptuneIR/NeptuneIRAttrs.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "Passes/NeptuneIRPasses.h"
#include "Utils/NeptuneScheduleDebug.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdlib>
#include <cstddef>
#include <optional>
#include <string>

namespace neptune {
namespace {

struct PortInfo {
  std::string name;
  bool isInput = false;
  bool isGhosted = false;
  unsigned roleIndex = 0;
  unsigned argIndex = 0;
  unsigned rank = 0;
  llvm::SmallVector<int64_t, 4> shape;
};

struct CopyPairInfo {
  unsigned inArgIndex = 0;
  unsigned outArgIndex = 0;
  bool fullDomain = false;
};

struct ScheduleInfo {
  bool valid = false;
  int64_t rank = 0;
  llvm::SmallVector<int64_t, 4> tile;
  int64_t vec = 1;
  std::string vecDim;
  std::string vecReason;
  int64_t unroll = 1;
  std::string unrollDim;
  std::string unrollReason;
  std::string parDim;
  std::string parReason;
  int64_t threads = 1;
  int64_t l1Bytes = 0;
  double alpha = 0.0;
  int64_t footprint = 0;
  bool cacheFit = false;
  llvm::SmallVector<std::string, 8> reorder;
};

struct KernelInfo {
  std::string tag;
  llvm::SmallVector<PortInfo, 8> ports;
  unsigned numInputs = 0;
  unsigned numOutputs = 0;
  bool halideCompatible = false;
  bool hasOutputBounds = false;
  bool boundsValid = false;
  bool partialCopyBoundary = false;
  bool hasResidualScf = false;
  std::string residualReason;
  llvm::SmallVector<CopyPairInfo, 4> copyPairs;
  llvm::SmallVector<unsigned, 4> halideArgs;
  std::string boundaryCopyMode;
  llvm::SmallVector<int64_t, 4> outMin;
  llvm::SmallVector<int64_t, 4> outExtent;
  llvm::SmallVector<int64_t, 4> radius;
  std::string radiusSource;
  std::string boundsSource;
  ScheduleInfo schedule;
  bool overlapSupported = false;
  std::string overlapReason;
  llvm::SmallVector<int64_t, 4> overlapInteriorLb;
  llvm::SmallVector<int64_t, 4> overlapInteriorUb;
  struct RegionInfo {
    std::string suffix;
    llvm::SmallVector<int64_t, 4> lb;
    llvm::SmallVector<int64_t, 4> ub;
  };
  llvm::SmallVector<RegionInfo, 8> overlapFaces;
};

struct Replacement {
  size_t start = 0;
  size_t end = 0;
  std::string text;
};

struct KernelDecisionInfo {
  std::string mode;
  std::string reason;
  std::string boundaryCopyMode;
  bool overlapCovered = false;
  bool overlapRewritten = false;
};

struct CodeWriter {
  explicit CodeWriter(llvm::raw_ostream &out) : os(out) {}

  void indent() { ++level; }
  void dedent() {
    if (level > 0)
      --level;
  }

  void line(llvm::StringRef text = "") {
    for (unsigned i = 0; i < level; ++i) {
      os << "  ";
    }
    os << text << "\n";
  }

  void openBlock(llvm::StringRef header) {
    line(header);
    indent();
  }

  void closeBlock(llvm::StringRef trailer = "}") {
    dedent();
    line(trailer);
  }

  llvm::raw_ostream &os;
  unsigned level = 0;
};

struct ApplyStencilInfo {
  llvm::SmallVector<int64_t, 4> lb;
  llvm::SmallVector<int64_t, 4> ub;
  llvm::SmallVector<int64_t, 4> radius;
  std::string radiusSource;
  ScheduleInfo schedule;
  llvm::SmallVector<unsigned, 4> applyInputArgs;
  std::optional<unsigned> applyOutputArg;
  bool hasResidualScf = false;
  bool residualCopyBoundary = false;
  llvm::SmallVector<CopyPairInfo, 4> copyPairs;
  std::string residualReason;
};

static void logResidualDecision(const KernelInfo &info,
                                llvm::raw_ostream &os);
static void logKernelSummary(const KernelInfo &info, const PortInfo &shapePort,
                             llvm::raw_ostream &os);
static void logOverlapInfo(const KernelInfo &info, llvm::raw_ostream &os);

static bool computeOverlapRegions(KernelInfo &info) {
  // Overlap is only enabled when we can prove a non-empty safe-interior.
  size_t rank = info.outMin.size();
  if (rank == 0 || rank != info.outExtent.size() ||
      rank != info.radius.size()) {
    info.overlapReason = "missing bounds/radius";
    return false;
  }
  if (rank > 3) {
    info.overlapReason = "rank>3 overlap not supported yet";
    return false;
  }

  llvm::SmallVector<int64_t, 4> lb;
  llvm::SmallVector<int64_t, 4> ub;
  lb.reserve(rank);
  ub.reserve(rank);
  for (size_t i = 0; i < rank; ++i) {
    lb.push_back(info.outMin[i]);
    ub.push_back(info.outMin[i] + info.outExtent[i]);
  }

  // safe-interior = [lb + r, ub - r)
  llvm::SmallVector<int64_t, 4> safeLb;
  llvm::SmallVector<int64_t, 4> safeUb;
  safeLb.reserve(rank);
  safeUb.reserve(rank);
  for (size_t i = 0; i < rank; ++i) {
    safeLb.push_back(lb[i] + info.radius[i]);
    safeUb.push_back(ub[i] - info.radius[i]);
  }
  for (size_t i = 0; i < rank; ++i) {
    if (safeUb[i] <= safeLb[i]) {
      info.overlapReason = "safe interior empty";
      return false;
    }
  }

  info.overlapInteriorLb = safeLb;
  info.overlapInteriorUb = safeUb;

  info.overlapFaces.clear();
  // Partition boundary into fixed faces; these do not overlap the interior.
  if (rank == 1) {
    KernelInfo::RegionInfo lo;
    lo.suffix = "face_lo";
    lo.lb = {lb[0]};
    lo.ub = {lb[0] + info.radius[0]};
    KernelInfo::RegionInfo hi;
    hi.suffix = "face_hi";
    hi.lb = {ub[0] - info.radius[0]};
    hi.ub = {ub[0]};
    info.overlapFaces.push_back(lo);
    info.overlapFaces.push_back(hi);
  } else if (rank == 2) {
    // 2D faces: top/bottom (full j), left/right (interior i).
    const int64_t i0 = lb[0], i1 = ub[0];
    const int64_t j0 = lb[1], j1 = ub[1];
    const int64_t ri = info.radius[0];
    const int64_t rj = info.radius[1];
    KernelInfo::RegionInfo top;
    top.suffix = "face_top";
    top.lb = {i0, j0};
    top.ub = {i0 + ri, j1};
    KernelInfo::RegionInfo bottom;
    bottom.suffix = "face_bottom";
    bottom.lb = {i1 - ri, j0};
    bottom.ub = {i1, j1};
    KernelInfo::RegionInfo left;
    left.suffix = "face_left";
    left.lb = {i0 + ri, j0};
    left.ub = {i1 - ri, j0 + rj};
    KernelInfo::RegionInfo right;
    right.suffix = "face_right";
    right.lb = {i0 + ri, j1 - rj};
    right.ub = {i1 - ri, j1};
    info.overlapFaces.push_back(top);
    info.overlapFaces.push_back(bottom);
    info.overlapFaces.push_back(left);
    info.overlapFaces.push_back(right);
  } else if (rank == 3) {
    // 3D faces: z/y/x slabs with interior-only ranges on the other axes.
    const int64_t i0 = lb[0], i1 = ub[0];
    const int64_t j0 = lb[1], j1 = ub[1];
    const int64_t k0 = lb[2], k1 = ub[2];
    const int64_t ri = info.radius[0];
    const int64_t rj = info.radius[1];
    const int64_t rk = info.radius[2];

    KernelInfo::RegionInfo zlo;
    zlo.suffix = "face_zlo";
    zlo.lb = {i0, j0, k0};
    zlo.ub = {i1, j1, k0 + rk};
    KernelInfo::RegionInfo zhi;
    zhi.suffix = "face_zhi";
    zhi.lb = {i0, j0, k1 - rk};
    zhi.ub = {i1, j1, k1};
    KernelInfo::RegionInfo ylo;
    ylo.suffix = "face_ylo";
    ylo.lb = {i0, j0, k0 + rk};
    ylo.ub = {i1, j0 + rj, k1 - rk};
    KernelInfo::RegionInfo yhi;
    yhi.suffix = "face_yhi";
    yhi.lb = {i0, j1 - rj, k0 + rk};
    yhi.ub = {i1, j1, k1 - rk};
    KernelInfo::RegionInfo xlo;
    xlo.suffix = "face_xlo";
    xlo.lb = {i0, j0 + rj, k0 + rk};
    xlo.ub = {i0 + ri, j1 - rj, k1 - rk};
    KernelInfo::RegionInfo xhi;
    xhi.suffix = "face_xhi";
    xhi.lb = {i1 - ri, j0 + rj, k0 + rk};
    xhi.ub = {i1, j1 - rj, k1 - rk};
    info.overlapFaces.push_back(zlo);
    info.overlapFaces.push_back(zhi);
    info.overlapFaces.push_back(ylo);
    info.overlapFaces.push_back(yhi);
    info.overlapFaces.push_back(xlo);
    info.overlapFaces.push_back(xhi);
  }

  info.overlapSupported = true;
  return true;
}
static std::string parseBoundaryCopyMode();

static ScheduleInfo parseScheduleAttr(mlir::DictionaryAttr dict, size_t rank) {
  ScheduleInfo info;
  if (!dict)
    return info;

  auto splitAttr = dict.getAs<mlir::DenseI64ArrayAttr>("split");
  if (!splitAttr)
    return info;
  auto splitVals = splitAttr.asArrayRef();
  if (splitVals.size() != rank)
    return info;
  info.tile.assign(splitVals.begin(), splitVals.end());
  info.rank = static_cast<int64_t>(rank);

  if (auto rankAttr = dict.getAs<mlir::IntegerAttr>("rank"))
    info.rank = rankAttr.getInt();
  if (auto vecAttr = dict.getAs<mlir::IntegerAttr>("vec"))
    info.vec = vecAttr.getInt();
  if (auto vecDimAttr = dict.getAs<mlir::StringAttr>("vec_dim"))
    info.vecDim = vecDimAttr.getValue().str();
  if (auto vecReasonAttr = dict.getAs<mlir::StringAttr>("vec_reason"))
    info.vecReason = vecReasonAttr.getValue().str();
  if (auto parDimAttr = dict.getAs<mlir::StringAttr>("par_dim"))
    info.parDim = parDimAttr.getValue().str();
  if (auto parReasonAttr = dict.getAs<mlir::StringAttr>("par_reason"))
    info.parReason = parReasonAttr.getValue().str();
  if (auto unrollAttr = dict.getAs<mlir::IntegerAttr>("unroll_factor"))
    info.unroll = unrollAttr.getInt();
  else if (auto unrollAttr = dict.getAs<mlir::IntegerAttr>("unroll"))
    info.unroll = unrollAttr.getInt();
  if (auto unrollDimAttr = dict.getAs<mlir::StringAttr>("unroll_dim"))
    info.unrollDim = unrollDimAttr.getValue().str();
  if (auto unrollReasonAttr = dict.getAs<mlir::StringAttr>("unroll_reason"))
    info.unrollReason = unrollReasonAttr.getValue().str();
  if (auto threadsAttr = dict.getAs<mlir::IntegerAttr>("threads"))
    info.threads = threadsAttr.getInt();
  if (auto l1Attr = dict.getAs<mlir::IntegerAttr>("l1_bytes"))
    info.l1Bytes = l1Attr.getInt();
  if (auto alphaAttr = dict.getAs<mlir::FloatAttr>("cache_alpha"))
    info.alpha = alphaAttr.getValueAsDouble();
  if (auto footprintAttr = dict.getAs<mlir::IntegerAttr>("footprint_bytes"))
    info.footprint = footprintAttr.getInt();
  if (auto cacheFitAttr = dict.getAs<mlir::BoolAttr>("cache_fit"))
    info.cacheFit = cacheFitAttr.getValue();
  if (auto reorderAttr = dict.getAs<mlir::ArrayAttr>("reorder")) {
    for (mlir::Attribute attr : reorderAttr) {
      auto strAttr = llvm::dyn_cast<mlir::StringAttr>(attr);
      if (!strAttr)
        continue;
      info.reorder.push_back(strAttr.getValue().str());
    }
  }

  info.valid = true;
  return info;
}

static bool matchConstantIndex(mlir::Value v, int64_t &out) {
  auto cst = v.getDefiningOp<mlir::arith::ConstantOp>();
  if (!cst || !cst.getType().isIndex()) {
    return false;
  }
  auto attr = llvm::dyn_cast<mlir::IntegerAttr>(cst.getValue());
  if (!attr) {
    return false;
  }
  out = attr.getInt();
  return true;
}

static std::string parseBoundaryCopyMode() {
  const char *modeEnv = std::getenv("NEPTUNECC_BOUNDARY_COPY_MODE");
  if (!modeEnv || modeEnv[0] == '\0') {
    return "slabs";
  }
  llvm::StringRef mode(modeEnv);
  if (mode == "whole") {
    return "whole";
  }
  if (mode == "slabs") {
    return "slabs";
  }
  llvm::errs() << "neptune-cc: unknown NEPTUNECC_BOUNDARY_COPY_MODE '"
               << mode << "', defaulting to slabs\n";
  return "slabs";
}

static mlir::Value stripMemrefCasts(mlir::Value v) {
  while (auto cast = v.getDefiningOp<mlir::memref::CastOp>()) {
    v = cast.getSource();
  }
  return v;
}

static bool isConstLikeOp(mlir::Operation *op) {
  if (llvm::isa<mlir::arith::ConstantOp, mlir::arith::ConstantIndexOp>(op)) {
    return true;
  }
  if (auto addi = llvm::dyn_cast<mlir::arith::AddIOp>(op)) {
    if (!addi.getType().isIndex()) {
      return false;
    }
    int64_t lhs = 0;
    int64_t rhs = 0;
    return matchConstantIndex(addi.getLhs(), lhs) &&
           matchConstantIndex(addi.getRhs(), rhs);
  }
  if (auto subi = llvm::dyn_cast<mlir::arith::SubIOp>(op)) {
    if (!subi.getType().isIndex()) {
      return false;
    }
    int64_t lhs = 0;
    int64_t rhs = 0;
    return matchConstantIndex(subi.getLhs(), lhs) &&
           matchConstantIndex(subi.getRhs(), rhs);
  }
  return false;
}

static void addCopyPair(llvm::SmallVectorImpl<CopyPairInfo> &pairs,
                        unsigned inArg, unsigned outArg, bool fullDomain) {
  for (auto &p : pairs) {
    if (p.inArgIndex == inArg && p.outArgIndex == outArg) {
      p.fullDomain = p.fullDomain || fullDomain;
      return;
    }
  }
  CopyPairInfo info;
  info.inArgIndex = inArg;
  info.outArgIndex = outArg;
  info.fullDomain = fullDomain;
  pairs.push_back(info);
}

static bool collectCopyPairsFromLoopNest(
    mlir::scf::ForOp outer, llvm::SmallVectorImpl<CopyPairInfo> &pairs,
    std::string &reason) {
  // Recognize residual copy loops: store(load(in[ivs])) -> out[ivs].
  auto func = outer->getParentOfType<mlir::func::FuncOp>();
  if (!func) {
    reason = "residual loop not inside func";
    return false;
  }
  llvm::SmallVector<mlir::scf::ForOp, 4> loops;
  mlir::scf::ForOp current = outer;
  while (true) {
    int64_t step = 0;
    if (!matchConstantIndex(current.getStep(), step) || step != 1) {
      reason = "residual loop step != 1";
      return false;
    }
    loops.push_back(current);
    if (loops.size() > 3) {
      reason = "residual loop rank > 3";
      return false;
    }

    mlir::scf::ForOp inner;
    bool sawLoadStore = false;
    for (mlir::Operation &op : current.getBody()->without_terminator()) {
      if (auto nested = llvm::dyn_cast<mlir::scf::ForOp>(op)) {
        if (inner || sawLoadStore) {
          reason = "non-perfect residual loop nest";
          return false;
        }
        inner = nested;
        continue;
      }
      if (llvm::isa<mlir::memref::LoadOp, mlir::memref::StoreOp>(op)) {
        sawLoadStore = true;
        continue;
      }
      if (llvm::isa<mlir::memref::CastOp>(op) || isConstLikeOp(&op)) {
        continue;
      }
      reason = "unsupported op in residual loop";
      return false;
    }
    if (inner) {
      current = inner;
      continue;
    }

    llvm::DenseMap<mlir::Value, unsigned> loadArgs;
    bool sawStore = false;
    for (mlir::Operation &op : current.getBody()->without_terminator()) {
      if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(op)) {
        // Only allow block-argument memrefs with iv-only indices.
        auto memref = stripMemrefCasts(load.getMemref());
        auto memrefTy = llvm::dyn_cast<mlir::MemRefType>(memref.getType());
        if (!memrefTy || memrefTy.getRank() != static_cast<int>(loops.size())) {
          reason = "residual load rank mismatch";
          return false;
        }
        auto arg = llvm::dyn_cast<mlir::BlockArgument>(memref);
        if (!arg || arg.getOwner() != &func.getBody().front()) {
          reason = "residual load memref not block argument";
          return false;
        }
        if (load.getIndices().size() != loops.size()) {
          reason = "residual load indices mismatch";
          return false;
        }
        for (size_t i = 0; i < loops.size(); ++i) {
          if (load.getIndices()[i] != loops[i].getInductionVar()) {
            reason = "residual load indices not ivs";
            return false;
          }
        }
        loadArgs[load.getResult()] = arg.getArgNumber();
        continue;
      }
      if (auto store = llvm::dyn_cast<mlir::memref::StoreOp>(op)) {
        // Store must directly use a load result and share iv-only indices.
        auto memref = stripMemrefCasts(store.getMemref());
        auto memrefTy = llvm::dyn_cast<mlir::MemRefType>(memref.getType());
        if (!memrefTy || memrefTy.getRank() != static_cast<int>(loops.size())) {
          reason = "residual store rank mismatch";
          return false;
        }
        if (!memrefTy.hasStaticShape()) {
          reason = "residual store dynamic shape";
          return false;
        }
        auto arg = llvm::dyn_cast<mlir::BlockArgument>(memref);
        if (!arg || arg.getOwner() != &func.getBody().front()) {
          reason = "residual store memref not block argument";
          return false;
        }
        if (store.getIndices().size() != loops.size()) {
          reason = "residual store indices mismatch";
          return false;
        }
        for (size_t i = 0; i < loops.size(); ++i) {
          if (store.getIndices()[i] != loops[i].getInductionVar()) {
            reason = "residual store indices not ivs";
            return false;
          }
        }
        auto it = loadArgs.find(store.getValue());
        if (it == loadArgs.end()) {
          reason = "store not from load";
          return false;
        }
        bool fullDomain = true;
        auto shape = memrefTy.getShape();
        if (shape.size() != loops.size()) {
          fullDomain = false;
        }
        for (size_t i = 0; i < loops.size() && i < shape.size(); ++i) {
          int64_t lb = 0;
          int64_t ub = 0;
          if (!matchConstantIndex(loops[i].getLowerBound(), lb) ||
              !matchConstantIndex(loops[i].getUpperBound(), ub)) {
            fullDomain = false;
            continue;
          }
          if (lb != 0 || ub != shape[i]) {
            fullDomain = false;
          }
        }
        addCopyPair(pairs, it->second, arg.getArgNumber(), fullDomain);
        sawStore = true;
        continue;
      }
      if (llvm::isa<mlir::memref::CastOp>(op) || isConstLikeOp(&op)) {
        continue;
      }
      reason = "unsupported op in residual loop body";
      return false;
    }
    if (!sawStore) {
      reason = "no store in residual loop";
      return false;
    }
    return true;
  }
}

static bool isWithinResidualLoop(
    mlir::Operation *op,
    llvm::ArrayRef<mlir::scf::ForOp> residualOuters) {
  for (auto loop : residualOuters) {
    if (loop->isAncestor(op)) {
      return true;
    }
  }
  return false;
}

static void analyzeResidualCopyBoundary(
    mlir::func::FuncOp func, mlir::Neptune::NeptuneIR::ApplyOp apply,
    ApplyStencilInfo &info) {
  // Residual SCF is only accepted when it is pure copy-boundary.
  llvm::SmallVector<mlir::scf::ForOp, 4> residualLoops;
  bool sawResidual = false;
  bool ok = true;
  std::string reason;

  func.walk([&](mlir::scf::ForOp op) {
    if (apply->isAncestor(op))
      return;
    auto parent = op->getParentOfType<mlir::scf::ForOp>();
    if (parent && !apply->isAncestor(parent))
      return;
    residualLoops.push_back(op);
  });

  func.walk([&](mlir::scf::IfOp op) {
    if (!apply->isAncestor(op)) {
      sawResidual = true;
      ok = false;
      if (reason.empty())
        reason = "residual scf.if";
    }
  });

  llvm::SmallVector<CopyPairInfo, 4> pairs;
  for (auto loop : residualLoops) {
    sawResidual = true;
    std::string loopReason;
    if (!collectCopyPairsFromLoopNest(loop, pairs, loopReason)) {
      ok = false;
      if (reason.empty())
        reason = loopReason;
      break;
    }
  }

  if (ok) {
    func.walk([&](mlir::Operation *op) {
      if (op == func.getOperation())
        return;
      if (apply->isAncestor(op))
        return;
      if (isWithinResidualLoop(op, residualLoops))
        return;
      if (op->hasTrait<mlir::OpTrait::IsTerminator>())
        return;
      if (llvm::isa<mlir::Neptune::NeptuneIR::ApplyOp,
                    mlir::Neptune::NeptuneIR::WrapOp,
                    mlir::Neptune::NeptuneIR::LoadOp,
                    mlir::Neptune::NeptuneIR::StoreOp>(op)) {
        return;
      }
      // Any other op outside apply/residual loops invalidates partial rewrite.
      if (llvm::isa<mlir::scf::ForOp, mlir::scf::IfOp>(op)) {
        sawResidual = true;
        ok = false;
        if (reason.empty())
          reason = "unsupported residual scf op";
        return;
      }
      if (isConstLikeOp(op) || llvm::isa<mlir::memref::CastOp>(op)) {
        return;
      }
      sawResidual = true;
      ok = false;
      if (reason.empty())
        reason = "unsupported residual op";
    });
  }

  info.hasResidualScf = sawResidual;
  if (!sawResidual) {
    info.residualCopyBoundary = false;
    return;
  }
  if (ok && !pairs.empty()) {
    info.residualCopyBoundary = true;
    info.copyPairs = std::move(pairs);
    return;
  }
  info.residualCopyBoundary = false;
  if (!reason.empty()) {
    info.residualReason = reason;
  }
}

static std::optional<ApplyStencilInfo>
inferStencilFromApply(mlir::ModuleOp module, mlir::func::FuncOp func) {
  // Clone + run SCF->NeptuneIR on the clone to avoid mutating the source.
  mlir::ModuleOp clone = module.clone();
  auto clonedFunc =
      clone.lookupSymbol<mlir::func::FuncOp>(func.getSymName());
  if (!clonedFunc) {
    return std::nullopt;
  }

  mlir::PassManager pm(clone.getContext());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::Neptune::NeptuneIR::createSCFBoundarySimplifyPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::Neptune::NeptuneIR::createSCFToNeptuneIRPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::Neptune::NeptuneIR::createNeptuneIRInferHalideSchedulePass());
  if (mlir::failed(pm.run(clone))) {
    return std::nullopt;
  }

  mlir::Neptune::NeptuneIR::ApplyOp apply;
  bool multiple = false;
  clonedFunc.walk([&](mlir::Neptune::NeptuneIR::ApplyOp op) {
    if (apply) {
      multiple = true;
      return;
    }
    apply = op;
  });
  if (!apply || multiple) {
    return std::nullopt;
  }

  auto bounds = apply.getBounds();
  auto lb = bounds.getLb().asArrayRef();
  auto ub = bounds.getUb().asArrayRef();
  if (lb.empty() || lb.size() != ub.size()) {
    return std::nullopt;
  }

  ApplyStencilInfo info;
  info.lb.assign(lb.begin(), lb.end());
  info.ub.assign(ub.begin(), ub.end());

  if (auto radius = apply.getRadius()) {
    info.radius.assign(radius->begin(), radius->end());
    info.radiusSource = "attr";
  } else {
    // Derive radius from access offsets if possible; otherwise default to 1.
    info.radius.assign(lb.size(), 0);
    bool ok = true;
    bool sawAccess = false;
    apply.walk([&](mlir::Neptune::NeptuneIR::AccessOp acc) {
      sawAccess = true;
      auto offsets = acc.getOffsets();
      if (offsets.size() != info.radius.size()) {
        ok = false;
        return;
      }
      for (size_t i = 0; i < offsets.size(); ++i) {
        int64_t v = offsets[i];
        if (v < 0)
          v = -v;
        if (v > info.radius[i])
          info.radius[i] = v;
      }
    });
    if (!ok || !sawAccess) {
      info.radius.assign(lb.size(), 1);
      info.radiusSource = "default";
    } else {
      info.radiusSource = "derived";
    }
  }

  if (info.radius.size() != info.lb.size()) {
    return std::nullopt;
  }

  for (mlir::Value input : apply.getInputs()) {
    // Inputs must be Wrap -> Load -> Apply, and refer to func arguments.
    auto load = input.getDefiningOp<mlir::Neptune::NeptuneIR::LoadOp>();
    if (!load)
      return std::nullopt;
    auto wrap =
        load.getVarField().getDefiningOp<mlir::Neptune::NeptuneIR::WrapOp>();
    if (!wrap)
      return std::nullopt;
    auto memref = wrap.getBuffer();
    auto arg = llvm::dyn_cast<mlir::BlockArgument>(memref);
    if (!arg || arg.getOwner() != &clonedFunc.getBody().front())
      return std::nullopt;
    info.applyInputArgs.push_back(arg.getArgNumber());
  }

  mlir::Neptune::NeptuneIR::StoreOp store;
  bool multipleStores = false;
  clonedFunc.walk([&](mlir::Neptune::NeptuneIR::StoreOp op) {
    if (op.getValue() != apply.getResult())
      return;
    if (store) {
      multipleStores = true;
      return;
    }
    store = op;
  });
  if (!store || multipleStores)
    return std::nullopt;
  auto wrap =
      store.getVarField().getDefiningOp<mlir::Neptune::NeptuneIR::WrapOp>();
  if (!wrap)
    return std::nullopt;
  auto memref = wrap.getBuffer();
  auto arg = llvm::dyn_cast<mlir::BlockArgument>(memref);
  if (!arg || arg.getOwner() != &clonedFunc.getBody().front())
    return std::nullopt;
  info.applyOutputArg = arg.getArgNumber();

  if (auto schedAttr =
          apply->getAttrOfType<mlir::DictionaryAttr>("neptune.schedule")) {
    info.schedule = parseScheduleAttr(schedAttr, info.lb.size());
  }

  analyzeResidualCopyBoundary(clonedFunc, apply, info);

  return info;
}

static bool validateStencilBounds(llvm::ArrayRef<int64_t> outMin,
                                  llvm::ArrayRef<int64_t> outExtent,
                                  llvm::ArrayRef<int64_t> radius,
                                  llvm::ArrayRef<int64_t> inShape,
                                  llvm::StringRef tag) {
  // Ensure out box + radius does not exceed the input shape.
  if (outMin.size() != inShape.size() ||
      outExtent.size() != inShape.size() ||
      radius.size() != inShape.size()) {
    llvm::errs() << "neptune-cc: stencil bounds rank mismatch for kernel '"
                 << tag << "'\n";
    return false;
  }

  for (size_t d = 0; d < inShape.size(); ++d) {
    if (outExtent[d] <= 0) {
      llvm::errs() << "neptune-cc: invalid extent for kernel '" << tag
                   << "'\n";
      return false;
    }
    if (outMin[d] < radius[d]) {
      llvm::errs() << "neptune-cc: outMin < radius for kernel '" << tag
                   << "'\n";
      return false;
    }
    int64_t upper = outMin[d] + outExtent[d] + radius[d];
    if (upper > inShape[d]) {
      llvm::errs() << "neptune-cc: bounds exceed input shape for kernel '"
                   << tag << "'\n";
      return false;
    }
  }
  return true;
}

static void printI64Array(llvm::raw_ostream &os,
                          llvm::ArrayRef<int64_t> values) {
  os << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << values[i];
  }
  os << "]";
}

static std::string joinPath(llvm::StringRef base,
                            std::initializer_list<llvm::StringRef> parts) {
  llvm::SmallString<256> path(base);
  for (llvm::StringRef part : parts) {
    llvm::sys::path::append(path, part);
  }
  return path.str().str();
}

static bool ensureDir(llvm::StringRef dir, llvm::StringRef label) {
  std::error_code ec = llvm::sys::fs::create_directories(dir);
  if (ec) {
    llvm::errs() << "neptune-cc: failed to create " << label << " dir '"
                 << dir << "': " << ec.message() << "\n";
    return false;
  }
  return true;
}

static bool parsePortMapEntry(llvm::StringRef entry, PortInfo &out,
                              llvm::raw_ostream &errs) {
  auto parts = entry.split('=');
  if (parts.first.empty() || parts.second.empty()) {
    errs << "neptune-cc: invalid port_map entry '" << entry << "'\n";
    return false;
  }

  out.name = parts.first.trim().str();
  llvm::SmallVector<llvm::StringRef, 4> fields;
  parts.second.split(fields, ':');
  if (fields.size() < 2) {
    errs << "neptune-cc: invalid port_map entry '" << entry << "'\n";
    return false;
  }

  llvm::StringRef role = fields[0].trim();
  if (role.starts_with("in")) {
    out.isInput = true;
    llvm::StringRef indexStr = role.drop_front(2);
    if (indexStr.getAsInteger(10, out.roleIndex)) {
      errs << "neptune-cc: invalid port_map role '" << role << "'\n";
      return false;
    }
  } else if (role.starts_with("out")) {
    out.isInput = false;
    llvm::StringRef indexStr = role.drop_front(3);
    if (indexStr.getAsInteger(10, out.roleIndex)) {
      errs << "neptune-cc: invalid port_map role '" << role << "'\n";
      return false;
    }
  } else {
    errs << "neptune-cc: invalid port_map role '" << role << "'\n";
    return false;
  }

  if (fields.size() >= 3) {
    llvm::StringRef storage = fields[1].trim();
    if (storage == "ghosted") {
      out.isGhosted = true;
    }
  }

  llvm::StringRef argPart = fields.back().trim();
  if (!argPart.starts_with("arg")) {
    errs << "neptune-cc: invalid port_map arg '" << argPart << "'\n";
    return false;
  }
  llvm::StringRef argIndexStr = argPart.drop_front(3);
  if (argIndexStr.getAsInteger(10, out.argIndex)) {
    errs << "neptune-cc: invalid port_map arg '" << argPart << "'\n";
    return false;
  }

  return true;
}

static const PortInfo *findPortByArgIndex(const KernelInfo &info,
                                          unsigned argIndex) {
  for (const auto &port : info.ports) {
    if (port.argIndex == argIndex) {
      return &port;
    }
  }
  return nullptr;
}

static const PortInfo *pickShapePort(const KernelInfo &info) {
  for (const auto &port : info.ports) {
    if (!port.isInput)
      return &port;
  }
  if (!info.ports.empty())
    return &info.ports.front();
  return nullptr;
}

static bool isHalideArg(const KernelInfo &info, unsigned argIndex) {
  for (unsigned arg : info.halideArgs) {
    if (arg == argIndex) {
      return true;
    }
  }
  return false;
}

static bool collectKernelInfos(const EventDB &db,
                               llvm::SmallVectorImpl<KernelInfo> &out) {
  if (!db.kernelModule) {
    return true;
  }

  std::string boundaryCopyMode = parseBoundaryCopyMode();
  llvm::StringSet<> seenTags;
  mlir::ModuleOp module = db.kernelModule.get();
  for (auto func : module.getOps<mlir::func::FuncOp>()) {
    auto tagAttr =
        func->getAttrOfType<mlir::StringAttr>("neptunecc.tag");
    if (!tagAttr) {
      continue;
    }
    llvm::StringRef tag = tagAttr.getValue();
    if (tag.empty()) {
      continue;
    }
    if (!seenTags.insert(tag).second) {
      llvm::errs() << "neptune-cc: duplicate kernel tag '" << tag << "'\n";
      return false;
    }

    auto portMapAttr =
        func->getAttrOfType<mlir::ArrayAttr>("neptunecc.port_map");
    if (!portMapAttr) {
      llvm::errs() << "neptune-cc: missing neptunecc.port_map for kernel '"
                   << tag << "'\n";
      return false;
    }

    KernelInfo info;
    info.tag = tag.str();
    for (auto attr : portMapAttr) {
      auto strAttr = llvm::dyn_cast<mlir::StringAttr>(attr);
      if (!strAttr) {
        llvm::errs() << "neptune-cc: invalid port_map entry for kernel '"
                     << tag << "'\n";
        return false;
      }
      PortInfo port;
      if (!parsePortMapEntry(strAttr.getValue(), port, llvm::errs())) {
        return false;
      }
      if (port.argIndex >= func.getNumArguments()) {
        llvm::errs() << "neptune-cc: port_map arg out of range for kernel '"
                     << tag << "'\n";
        return false;
      }
      auto memrefTy = llvm::dyn_cast<mlir::MemRefType>(
          func.getArgument(port.argIndex).getType());
      if (!memrefTy) {
        llvm::errs() << "neptune-cc: expected memref argument for kernel '"
                     << tag << "'\n";
        return false;
      }
      if (!memrefTy.hasStaticShape()) {
        llvm::errs()
            << "neptune-cc: dynamic memref shape not supported for kernel '"
            << tag << "'\n";
        return false;
      }
      auto elemTy = memrefTy.getElementType();
      auto intTy = llvm::dyn_cast<mlir::IntegerType>(elemTy);
      if (!intTy || intTy.getWidth() != 32) {
        llvm::errs() << "neptune-cc: only i32 memrefs supported for kernel '"
                     << tag << "'\n";
        return false;
      }
      port.rank = memrefTy.getRank();
      if (port.rank == 0) {
        llvm::errs() << "neptune-cc: scalar memrefs not supported for kernel '"
                     << tag << "'\n";
        return false;
      }
      port.shape.assign(memrefTy.getShape().begin(),
                        memrefTy.getShape().end());
      if (port.isInput) {
        ++info.numInputs;
      } else {
        ++info.numOutputs;
      }
      info.ports.push_back(std::move(port));
    }

    info.halideCompatible = false;
    const PortInfo *inPort = nullptr;
    const PortInfo *outPort = nullptr;
    auto applyInfo = inferStencilFromApply(module, func);
    if (applyInfo) {
      info.schedule = applyInfo->schedule;
    }
    if (applyInfo && applyInfo->applyOutputArg &&
        !applyInfo->applyInputArgs.empty()) {
      bool ok = true;
      for (unsigned arg : applyInfo->applyInputArgs) {
        const PortInfo *p = findPortByArgIndex(info, arg);
        if (!p || !p->isInput) {
          ok = false;
          break;
        }
      }
      const PortInfo *outCandidate =
          findPortByArgIndex(info, *applyInfo->applyOutputArg);
      if (!outCandidate || outCandidate->isInput) {
        ok = false;
      }
      if (ok) {
        inPort = findPortByArgIndex(info, applyInfo->applyInputArgs.front());
        outPort = outCandidate;
        if (inPort && outPort && inPort->rank == outPort->rank &&
            inPort->rank > 0) {
          bool rankOk = true;
          for (unsigned arg : applyInfo->applyInputArgs) {
            const PortInfo *p = findPortByArgIndex(info, arg);
            if (!p || p->rank != outPort->rank) {
              rankOk = false;
              break;
            }
          }
          bool shapeOk = rankOk;
          if (shapeOk) {
            for (unsigned arg : applyInfo->applyInputArgs) {
              const PortInfo *p = findPortByArgIndex(info, arg);
              if (!p || p->shape != outPort->shape) {
                shapeOk = false;
                break;
              }
            }
          }
          if (shapeOk) {
            info.halideCompatible = true;
            info.halideArgs.assign(applyInfo->applyInputArgs.begin(),
                                   applyInfo->applyInputArgs.end());
            info.halideArgs.push_back(*applyInfo->applyOutputArg);
          }
        }
      }
    }

    if (info.halideCompatible && inPort && outPort) {
      bool boundsFromApply = false;
      bool applyHasResidualScf = false;
      bool applyResidualCopy = false;
      std::string residualReason;
      llvm::SmallVector<CopyPairInfo, 4> residualPairs;
      llvm::SmallVector<int64_t, 4> radius;
      std::string radiusSource;

      if (applyInfo) {
        applyHasResidualScf = applyInfo->hasResidualScf;
        applyResidualCopy = applyInfo->residualCopyBoundary;
        residualReason = applyInfo->residualReason;
        residualPairs = applyInfo->copyPairs;
        if (applyInfo->lb.size() == outPort->rank &&
            applyInfo->ub.size() == outPort->rank) {
          bool ok = true;
          for (size_t d = 0; d < applyInfo->lb.size(); ++d) {
            if (applyInfo->ub[d] <= applyInfo->lb[d]) {
              ok = false;
              break;
            }
          }
          if (ok) {
            info.hasOutputBounds = true;
            info.outMin.assign(applyInfo->lb.begin(), applyInfo->lb.end());
            info.outExtent.assign(outPort->rank, 0);
            for (size_t d = 0; d < outPort->rank; ++d) {
              info.outExtent[d] = applyInfo->ub[d] - applyInfo->lb[d];
            }
            radius.assign(applyInfo->radius.begin(), applyInfo->radius.end());
            radiusSource = applyInfo->radiusSource;
            boundsFromApply = true;
          }
        }
      }

      if (info.hasOutputBounds) {
        if (radius.size() != outPort->rank) {
          llvm::errs()
              << "neptune-cc: missing radius for kernel '" << tag << "'\n";
          info.hasOutputBounds = false;
        }
      }

      if (info.hasOutputBounds) {
        if (!validateStencilBounds(
                info.outMin, info.outExtent, radius, inPort->shape, tag)) {
          info.hasOutputBounds = false;
        } else {
          info.boundsValid = true;
          info.radius = std::move(radius);
          info.radiusSource = radiusSource;
          info.boundsSource = boundsFromApply ? "neptune.ir.apply" : "";
          info.hasResidualScf = applyHasResidualScf;
          if (boundsFromApply) {
            llvm::outs() << "neptune-cc: bounds from neptune.ir.apply for '"
                         << tag << "'\n";
          }
          if (boundsFromApply && applyHasResidualScf) {
            if (applyResidualCopy && !residualPairs.empty()) {
              bool ok = true;
              for (const auto &pair : residualPairs) {
                const PortInfo *pairIn =
                    findPortByArgIndex(info, pair.inArgIndex);
                const PortInfo *pairOut =
                    findPortByArgIndex(info, pair.outArgIndex);
                if (!pairIn || !pairOut) {
                  ok = false;
                  residualReason = "copy pair not in port_map";
                  break;
                }
                if (pairIn->rank != pairOut->rank) {
                  ok = false;
                  residualReason = "copy pair rank mismatch";
                  break;
                }
                if (pairIn->rank < 1) {
                  ok = false;
                  residualReason = "copy pair rank unsupported";
                  break;
                }
                if (pairIn->rank > 3 && boundaryCopyMode != "whole") {
                  ok = false;
                  residualReason = "rank>3 requires NEPTUNECC_BOUNDARY_COPY_MODE=whole";
                  break;
                }
                if (pairIn->shape != pairOut->shape) {
                  ok = false;
                  residualReason = "copy pair shape mismatch";
                  break;
                }
                if (pair.outArgIndex != outPort->argIndex &&
                    !pair.fullDomain) {
                  ok = false;
                  residualReason = "copy pair not full domain";
                  break;
                }
              }
              if (ok) {
                info.partialCopyBoundary = true;
                info.copyPairs = std::move(residualPairs);
                info.boundaryCopyMode = boundaryCopyMode;
              } else {
                info.boundsValid = false;
                info.partialCopyBoundary = false;
                info.residualReason =
                    residualReason.empty()
                        ? "residual scf not copy-boundary"
                        : residualReason;
                logKernelSummary(info, *inPort, llvm::outs());
                logResidualDecision(info, llvm::outs());
              }
            } else {
              info.boundsValid = false;
              info.residualReason =
                  residualReason.empty() ? "residual scf not copy-boundary"
                                         : residualReason;
              logKernelSummary(info, *inPort, llvm::outs());
              logResidualDecision(info, llvm::outs());
            }
          }
        }
      }

      if (!info.boundsValid) {
        llvm::errs()
            << "neptune-cc: no valid stencil bounds for kernel '" << tag
            << "', skipping glue/rewrite\n";
        if (info.hasResidualScf && !info.partialCopyBoundary) {
          llvm::errs() << "neptune-cc: residual scf not eligible: "
                       << (info.residualReason.empty()
                               ? "unknown reason"
                               : info.residualReason)
                       << "\n";
        }
      } else {
        computeOverlapRegions(info);
      }
    }

    std::stable_sort(info.ports.begin(), info.ports.end(),
                     [](const PortInfo &a, const PortInfo &b) {
                       if (a.isInput != b.isInput) {
                         return a.isInput && !b.isInput;
                       }
                       if (a.roleIndex != b.roleIndex) {
                         return a.roleIndex < b.roleIndex;
                       }
                       return a.name < b.name;
                     });

    out.push_back(std::move(info));
  }

  return true;
}

static void emitArrayArg(llvm::raw_ostream &os, llvm::StringRef name,
                         unsigned rank) {
  if (rank <= 1) {
    os << name;
    return;
  }
  os << "&" << name;
  for (unsigned i = 0; i < rank; ++i) {
    os << "[0]";
  }
}

static void emitKernelCall(llvm::raw_ostream &os, const KernelInfo &info) {
  os << "neptunecc::" << info.tag << "(";
  for (size_t i = 0; i < info.ports.size(); ++i) {
    if (i > 0) {
      os << ", ";
    }
    const auto &port = info.ports[i];
    emitArrayArg(os, port.name, port.rank);
  }
  os << ");";
}

static size_t findLineStart(llvm::StringRef content, size_t offset) {
  size_t pos = std::min(offset, content.size());
  while (pos > 0 && content[pos - 1] != '\n') {
    --pos;
  }
  return pos;
}

static size_t findLineEnd(llvm::StringRef content, size_t offset) {
  size_t pos = std::min(offset, content.size());
  while (pos < content.size() && content[pos] != '\n') {
    ++pos;
  }
  if (pos < content.size()) {
    ++pos;
  }
  return pos;
}

static std::string detectNewline(llvm::StringRef content) {
  if (content.find("\r\n") != llvm::StringRef::npos) {
    return "\r\n";
  }
  return "\n";
}

static void ensureIncludeLine(std::string &content,
                              llvm::StringRef includeLine) {
  llvm::StringRef contentRef(content);
  if (contentRef.find(includeLine) != llvm::StringRef::npos) {
    return;
  }

  std::string newline = detectNewline(contentRef);
  size_t insertPos = 0;
  if (contentRef.starts_with("\xEF\xBB\xBF")) {
    insertPos = 3;
  }

  bool inBlockComment = false;
  size_t pos = insertPos;
  while (pos < content.size()) {
    size_t lineEnd = content.find('\n', pos);
    if (lineEnd == std::string::npos) {
      lineEnd = content.size();
    }
    llvm::StringRef line(content.data() + pos, lineEnd - pos);
    llvm::StringRef trimmed = line.trim();

    if (inBlockComment) {
      if (trimmed.find("*/") != llvm::StringRef::npos) {
        inBlockComment = false;
      }
      insertPos = (lineEnd < content.size()) ? lineEnd + 1 : lineEnd;
      pos = insertPos;
      continue;
    }

    if (trimmed.empty()) {
      insertPos = (lineEnd < content.size()) ? lineEnd + 1 : lineEnd;
      pos = insertPos;
      continue;
    }

    if (trimmed.starts_with("/*")) {
      if (trimmed.find("*/") == llvm::StringRef::npos) {
        inBlockComment = true;
      }
      insertPos = (lineEnd < content.size()) ? lineEnd + 1 : lineEnd;
      pos = insertPos;
      continue;
    }

    if (trimmed.starts_with("//")) {
      insertPos = (lineEnd < content.size()) ? lineEnd + 1 : lineEnd;
      pos = insertPos;
      continue;
    }

    if (trimmed.starts_with("#include")) {
      insertPos = (lineEnd < content.size()) ? lineEnd + 1 : lineEnd;
      pos = insertPos;
      continue;
    }

    break;
  }

  std::string includeText = includeLine.str();
  includeText += newline;
  content.insert(insertPos, includeText);
}

static void ensureKernelInclude(std::string &content) {
  ensureIncludeLine(content, "#include \"neptunecc_kernels.h\"");
}

static void logResidualDecision(const KernelInfo &info,
                                llvm::raw_ostream &os) {
  if (info.hasResidualScf) {
    os << "neptune-cc:   residual_scf=yes";
    if (info.partialCopyBoundary) {
      os << " copy_boundary=yes decision=PartialRewrite";
      if (!info.boundaryCopyMode.empty()) {
        os << " mode=" << info.boundaryCopyMode;
      }
      os << "\n";
      os << "neptune-cc:   copy_pairs=" << info.copyPairs.size() << " [";
      for (size_t i = 0; i < info.copyPairs.size(); ++i) {
        const auto &pair = info.copyPairs[i];
        const PortInfo *inPortPair = findPortByArgIndex(info, pair.inArgIndex);
        const PortInfo *outPortPair =
            findPortByArgIndex(info, pair.outArgIndex);
        if (i > 0) {
          os << ", ";
        }
        os << "arg" << pair.inArgIndex << "->arg" << pair.outArgIndex;
        if (inPortPair && outPortPair) {
          os << "(" << inPortPair->name << "->" << outPortPair->name << ")";
        }
        if (!pair.fullDomain) {
          os << ":partial";
        }
      }
      os << "]\n";
    } else {
      os << " copy_boundary=no decision=SkipRewrite";
      if (!info.residualReason.empty()) {
        os << " reason=" << info.residualReason;
      }
      os << "\n";
    }
  } else {
    os << "neptune-cc:   residual_scf=no decision=PureHalide\n";
  }
}

static void logKernelSummary(const KernelInfo &info, const PortInfo &shapePort,
                             llvm::raw_ostream &os) {
  os << "neptune-cc: glue kernel '" << info.tag << "'\n";
  os << "neptune-cc:   rank=" << shapePort.rank << " shape=";
  printI64Array(os, shapePort.shape);
  if (!info.boundsSource.empty()) {
    os << " bounds=" << info.boundsSource;
  }
  if (!info.radiusSource.empty()) {
    os << " radius_source=" << info.radiusSource;
  }
  os << "\n";
  os << "neptune-cc:   outMin=";
  printI64Array(os, info.outMin);
  os << " outExtent=";
  printI64Array(os, info.outExtent);
  os << " radius=";
  printI64Array(os, info.radius);
  os << "\n";
  if (info.radiusSource == "default") {
    os << "neptune-cc:   radius defaulted to 1\n";
  }
}

static void logOverlapInfo(const KernelInfo &info, llvm::raw_ostream &os) {
  if (!info.overlapSupported) {
    os << "neptune-cc:   overlap=skip reason=" << info.overlapReason << "\n";
    return;
  }
  os << "neptune-cc:   overlap=enabled interior=[";
  printI64Array(os, info.overlapInteriorLb);
  os << " .. ";
  printI64Array(os, info.overlapInteriorUb);
  os << ")\n";
  os << "neptune-cc:   overlap_faces=" << info.overlapFaces.size() << "\n";
  for (const auto &face : info.overlapFaces) {
    os << "neptune-cc:     " << face.suffix << " [";
    printI64Array(os, face.lb);
    os << " .. ";
    printI64Array(os, face.ub);
    os << ")\n";
  }
}

static ScheduleDecision buildScheduleDecision(const ScheduleInfo &info) {
  ScheduleDecision out;
  out.valid = info.valid;
  if (!info.valid)
    return out;
  out.rank = info.rank;
  out.tile.assign(info.tile.begin(), info.tile.end());
  out.vec = info.vec;
  out.vecDim = info.vecDim.empty() ? "last" : info.vecDim;
  out.vecReason = info.vecReason;
  out.unroll = info.unroll;
  out.unrollDim = info.unrollDim.empty() ? "none" : info.unrollDim;
  out.unrollReason = info.unrollReason;
  out.parDim = info.parDim.empty() ? "none" : info.parDim;
  out.parReason = info.parReason;
  out.threads = info.threads;
  out.l1Bytes = info.l1Bytes;
  out.alpha = info.alpha;
  out.footprint = info.footprint;
  out.cacheFit = info.cacheFit;
  out.reorder.assign(info.reorder.begin(), info.reorder.end());
  return out;
}

static OverlapPlan buildOverlapPlan(const KernelInfo &info,
                                    const KernelDecisionInfo &decision) {
  OverlapPlan plan;
  plan.supported = info.overlapSupported;
  plan.enabled = decision.overlapRewritten;
  if (!info.overlapSupported) {
    plan.reason = info.overlapReason;
  } else if (plan.enabled) {
    plan.reason = "";
  } else if (decision.overlapCovered && !decision.reason.empty()) {
    plan.reason = decision.reason;
  } else if (decision.overlapCovered) {
    plan.reason = "overlap skipped";
  } else {
    plan.reason = "no overlap block";
  }
  if (info.overlapSupported) {
    plan.interiorLb.assign(info.overlapInteriorLb.begin(),
                           info.overlapInteriorLb.end());
    plan.interiorUb.assign(info.overlapInteriorUb.begin(),
                           info.overlapInteriorUb.end());
    for (const auto &face : info.overlapFaces) {
      OverlapRegion region;
      region.name = face.suffix;
      region.lb.assign(face.lb.begin(), face.lb.end());
      region.ub.assign(face.ub.begin(), face.ub.end());
      plan.faces.push_back(std::move(region));
    }
  }
  return plan;
}

static KernelScheduleRecord buildScheduleRecord(
    const KernelInfo &info, const KernelDecisionInfo &decision) {
  KernelScheduleRecord rec;
  rec.tag = info.tag;
  rec.name = info.tag;
  if (const PortInfo *shapePort = pickShapePort(info)) {
    rec.rank = static_cast<int64_t>(shapePort->rank);
    rec.shape.assign(shapePort->shape.begin(), shapePort->shape.end());
  }
  rec.outMin.assign(info.outMin.begin(), info.outMin.end());
  rec.outExtent.assign(info.outExtent.begin(), info.outExtent.end());
  rec.radius.assign(info.radius.begin(), info.radius.end());
  for (const auto &port : info.ports) {
    PortRecord pr;
    pr.name = port.name;
    pr.direction = port.isInput ? "in" : "out";
    pr.qualifier = port.isGhosted ? "ghosted" : "owned";
    pr.ghosted = port.isGhosted;
    pr.roleIndex = port.roleIndex;
    pr.argIndex = port.argIndex;
    rec.ports.push_back(std::move(pr));
  }
  rec.schedule = buildScheduleDecision(info.schedule);
  rec.overlap = buildOverlapPlan(info, decision);
  rec.rewrite.mode = decision.mode;
  rec.rewrite.reason = decision.reason;
  rec.rewrite.boundaryCopyMode = decision.boundaryCopyMode;
  return rec;
}

static void emitScheduleDebug(llvm::StringRef outDir,
                              llvm::ArrayRef<KernelInfo> kernels,
                              const llvm::StringMap<KernelDecisionInfo> &decisions) {
  if (!scheduleDumpEnabled())
    return;
  ScheduleEmitter emitter(outDir, /*enableJson=*/true, /*enableTxt=*/true,
                          /*enableDot=*/scheduleDotEnabled());
  for (const auto &kernel : kernels) {
    KernelDecisionInfo decision;
    auto it = decisions.find(kernel.tag);
    if (it != decisions.end()) {
      decision = it->second;
    } else {
      decision.mode = "Unknown";
      decision.reason = "no decision";
    }
    emitter.recordKernel(buildScheduleRecord(kernel, decision));
  }
  (void)emitter.flush();
}

static void emitKernelSignatureWithName(llvm::raw_ostream &os,
                                        llvm::StringRef name,
                                        const KernelInfo &kernel) {
  os << "int " << name << "(";
  for (size_t i = 0; i < kernel.ports.size(); ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << "int32_t* " << kernel.ports[i].name;
  }
  os << ")";
}

static void emitKernelSignature(llvm::raw_ostream &os,
                                const KernelInfo &kernel) {
  emitKernelSignatureWithName(os, kernel.tag, kernel);
}

static bool writeGlueHeader(llvm::StringRef glueDir,
                            llvm::ArrayRef<KernelInfo> kernels) {
  std::string headerPath = joinPath(glueDir, {"neptunecc_kernels.h"});
  std::error_code ec;
  llvm::raw_fd_ostream os(headerPath, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "neptune-cc: failed to open glue header '"
                 << headerPath << "': " << ec.message() << "\n";
    return false;
  }

  CodeWriter w(os);
  w.line("#pragma once");
  w.line("#include <cstdint>");
  w.line("namespace neptunecc {");
  w.indent();
  for (const auto &kernel : kernels) {
    if (!kernel.halideCompatible || !kernel.boundsValid) {
      if (!kernel.halideCompatible) {
        llvm::errs() << "neptune-cc: skipping glue header for kernel '"
                     << kernel.tag
                     << "' (missing apply-mapped input/output)\n";
      } else {
        llvm::errs() << "neptune-cc: skipping glue header for kernel '"
                     << kernel.tag << "' (no valid stencil bounds)\n";
      }
      continue;
    }
    std::string sig;
    llvm::raw_string_ostream sigStream(sig);
    emitKernelSignature(sigStream, kernel);
    sigStream << ";";
    w.line(sigStream.str());

    if (kernel.overlapSupported) {
      std::string interiorName = kernel.tag + "_interior";
      std::string interiorSig;
      llvm::raw_string_ostream interiorStream(interiorSig);
      emitKernelSignatureWithName(interiorStream, interiorName, kernel);
      interiorStream << ";";
      w.line(interiorStream.str());

      for (const auto &face : kernel.overlapFaces) {
        std::string faceName = kernel.tag + "_" + face.suffix;
        std::string faceSig;
        llvm::raw_string_ostream faceStream(faceSig);
        emitKernelSignatureWithName(faceStream, faceName, kernel);
        faceStream << ";";
        w.line(faceStream.str());
      }
    }
  }
  w.dedent();
  w.line("} // namespace neptunecc");
  return true;
}

static void emitWholeCopyND(CodeWriter &w, const std::string &inName,
                            const std::string &outName,
                            llvm::ArrayRef<int64_t> shape) {
  int64_t total = 1;
  for (int64_t dim : shape) {
    total *= dim;
  }
  w.line("// copy-boundary (whole domain)");
  w.line(llvm::formatv("for (int64_t idx = 0; idx < {0}; ++idx) {{", total)
             .str());
  w.indent();
  w.line(llvm::formatv("{0}[idx] = {1}[idx];", outName, inName).str());
  w.dedent();
  w.line("}");
  w.line();
}

static void emitSlabCopy1D(CodeWriter &w, const std::string &inName,
                           const std::string &outName,
                           llvm::ArrayRef<int64_t> shape,
                           llvm::ArrayRef<int64_t> outMin,
                           llvm::ArrayRef<int64_t> outExtent) {
  const int64_t n = shape[0];
  const int64_t lb = outMin[0];
  const int64_t ub = outMin[0] + outExtent[0];
  w.line("// copy-boundary (slabs, 1D)");
  if (lb > 0) {
    w.line(llvm::formatv("for (int64_t i = 0; i < {0}; ++i) {{", lb).str());
    w.indent();
    w.line(llvm::formatv("{0}[i] = {1}[i];", outName, inName).str());
    w.dedent();
    w.line("}");
  }
  if (ub < n) {
    w.line(llvm::formatv("for (int64_t i = {0}; i < {1}; ++i) {{", ub, n)
               .str());
    w.indent();
    w.line(llvm::formatv("{0}[i] = {1}[i];", outName, inName).str());
    w.dedent();
    w.line("}");
  }
  w.line();
}

static void emitSlabCopy2D(CodeWriter &w, const std::string &inName,
                           const std::string &outName,
                           llvm::ArrayRef<int64_t> shape,
                           llvm::ArrayRef<int64_t> outMin,
                           llvm::ArrayRef<int64_t> outExtent) {
  const int64_t h = shape[0];
  const int64_t width = shape[1];
  const int64_t i0 = outMin[0];
  const int64_t i1 = outMin[0] + outExtent[0];
  const int64_t j0 = outMin[1];
  const int64_t j1 = outMin[1] + outExtent[1];
  w.line("// copy-boundary (slabs, 2D)");
  if (i0 > 0) {
    w.line(llvm::formatv("for (int64_t i = 0; i < {0}; ++i) {{", i0).str());
    w.indent();
    w.line(
        llvm::formatv("for (int64_t j = 0; j < {0}; ++j) {{", width).str());
    w.indent();
    w.line(llvm::formatv("{0}[i * {1} + j] = {2}[i * {1} + j];", outName,
                         width, inName)
               .str());
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
  }
  if (i1 < h) {
    w.line(llvm::formatv("for (int64_t i = {0}; i < {1}; ++i) {{", i1, h)
               .str());
    w.indent();
    w.line(
        llvm::formatv("for (int64_t j = 0; j < {0}; ++j) {{", width).str());
    w.indent();
    w.line(llvm::formatv("{0}[i * {1} + j] = {2}[i * {1} + j];", outName,
                         width, inName)
               .str());
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
  }
  if (j0 > 0 && i1 > i0) {
    w.line(llvm::formatv("for (int64_t i = {0}; i < {1}; ++i) {{", i0, i1)
               .str());
    w.indent();
    w.line(llvm::formatv("for (int64_t j = 0; j < {0}; ++j) {{", j0).str());
    w.indent();
    w.line(llvm::formatv("{0}[i * {1} + j] = {2}[i * {1} + j];", outName,
                         width, inName)
               .str());
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
  }
  if (j1 < width && i1 > i0) {
    w.line(llvm::formatv("for (int64_t i = {0}; i < {1}; ++i) {{", i0, i1)
               .str());
    w.indent();
    w.line(
        llvm::formatv("for (int64_t j = {0}; j < {1}; ++j) {{", j1, width)
            .str());
    w.indent();
    w.line(llvm::formatv("{0}[i * {1} + j] = {2}[i * {1} + j];", outName,
                         width, inName)
               .str());
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
  }
  w.line();
}

static void emitSlabCopy3D(CodeWriter &w, const std::string &inName,
                           const std::string &outName,
                           llvm::ArrayRef<int64_t> shape,
                           llvm::ArrayRef<int64_t> outMin,
                           llvm::ArrayRef<int64_t> outExtent) {
  const int64_t d0 = shape[0];
  const int64_t d1 = shape[1];
  const int64_t d2 = shape[2];
  const int64_t i0 = outMin[0];
  const int64_t i1 = outMin[0] + outExtent[0];
  const int64_t j0 = outMin[1];
  const int64_t j1 = outMin[1] + outExtent[1];
  const int64_t k0 = outMin[2];
  const int64_t k1 = outMin[2] + outExtent[2];
  w.line("// copy-boundary (slabs, 3D)");
  if (k0 > 0) {
    w.line(llvm::formatv("for (int64_t i = 0; i < {0}; ++i) {{", d0).str());
    w.indent();
    w.line(llvm::formatv("for (int64_t j = 0; j < {0}; ++j) {{", d1).str());
    w.indent();
    w.line(llvm::formatv("for (int64_t k = 0; k < {0}; ++k) {{", k0).str());
    w.indent();
    w.line(llvm::formatv("{0}[((i * {1}) + j) * {2} + k] = {3}[((i * {1}) + j) * {2} + k];",
                         outName, d1, d2, inName)
               .str());
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
  }
  if (k1 < d2) {
    w.line(llvm::formatv("for (int64_t i = 0; i < {0}; ++i) {{", d0).str());
    w.indent();
    w.line(llvm::formatv("for (int64_t j = 0; j < {0}; ++j) {{", d1).str());
    w.indent();
    w.line(llvm::formatv("for (int64_t k = {0}; k < {1}; ++k) {{", k1, d2)
               .str());
    w.indent();
    w.line(llvm::formatv("{0}[((i * {1}) + j) * {2} + k] = {3}[((i * {1}) + j) * {2} + k];",
                         outName, d1, d2, inName)
               .str());
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
  }
  if (j0 > 0 && k1 > k0) {
    w.line(llvm::formatv("for (int64_t i = 0; i < {0}; ++i) {{", d0).str());
    w.indent();
    w.line(llvm::formatv("for (int64_t j = 0; j < {0}; ++j) {{", j0).str());
    w.indent();
    w.line(llvm::formatv("for (int64_t k = {0}; k < {1}; ++k) {{", k0, k1)
               .str());
    w.indent();
    w.line(llvm::formatv("{0}[((i * {1}) + j) * {2} + k] = {3}[((i * {1}) + j) * {2} + k];",
                         outName, d1, d2, inName)
               .str());
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
  }
  if (j1 < d1 && k1 > k0) {
    w.line(llvm::formatv("for (int64_t i = 0; i < {0}; ++i) {{", d0).str());
    w.indent();
    w.line(llvm::formatv("for (int64_t j = {0}; j < {1}; ++j) {{", j1, d1)
               .str());
    w.indent();
    w.line(llvm::formatv("for (int64_t k = {0}; k < {1}; ++k) {{", k0, k1)
               .str());
    w.indent();
    w.line(llvm::formatv("{0}[((i * {1}) + j) * {2} + k] = {3}[((i * {1}) + j) * {2} + k];",
                         outName, d1, d2, inName)
               .str());
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
  }
  if (i0 > 0 && j1 > j0 && k1 > k0) {
    w.line(llvm::formatv("for (int64_t i = 0; i < {0}; ++i) {{", i0).str());
    w.indent();
    w.line(llvm::formatv("for (int64_t j = {0}; j < {1}; ++j) {{", j0, j1)
               .str());
    w.indent();
    w.line(llvm::formatv("for (int64_t k = {0}; k < {1}; ++k) {{", k0, k1)
               .str());
    w.indent();
    w.line(llvm::formatv("{0}[((i * {1}) + j) * {2} + k] = {3}[((i * {1}) + j) * {2} + k];",
                         outName, d1, d2, inName)
               .str());
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
  }
  if (i1 < d0 && j1 > j0 && k1 > k0) {
    w.line(llvm::formatv("for (int64_t i = {0}; i < {1}; ++i) {{", i1, d0)
               .str());
    w.indent();
    w.line(llvm::formatv("for (int64_t j = {0}; j < {1}; ++j) {{", j0, j1)
               .str());
    w.indent();
    w.line(llvm::formatv("for (int64_t k = {0}; k < {1}; ++k) {{", k0, k1)
               .str());
    w.indent();
    w.line(llvm::formatv("{0}[((i * {1}) + j) * {2} + k] = {3}[((i * {1}) + j) * {2} + k];",
                         outName, d1, d2, inName)
               .str());
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
  }
  w.line();
}

struct DimSpec {
  int64_t min = 0;
  int64_t extent = 0;
  int64_t stride = 0;
};

static void emitDimsTemplateAndInit(CodeWriter &w, llvm::StringRef base,
                                    unsigned rank,
                                    llvm::ArrayRef<DimSpec> dims,
                                    int64_t offsetElems) {
  std::string dimsName = std::string(base) + "_dims";
  std::string templName = std::string(base) + "_templ";

  w.line(llvm::formatv("static const halide_dimension_t {0}[{1}] = {{",
                       dimsName, rank)
             .str());
  w.indent();
  for (size_t i = 0; i < dims.size(); ++i) {
    const DimSpec &d = dims[i];
    llvm::StringRef comma = (i + 1 < dims.size()) ? "," : "";
    std::string line;
    llvm::raw_string_ostream lineStream(line);
    lineStream << "{" << d.min << ", " << d.extent << ", " << d.stride
               << ", 0}" << comma;
    w.line(lineStream.str());
  }
  w.dedent();
  w.line("};");
  w.line(llvm::formatv("static const halide_buffer_t {0} = {{", templName)
             .str());
  w.indent();
  w.line("0,");
  w.line("nullptr,");
  w.line("nullptr,");
  w.line("0,");
  w.line("k_i32_type,");
  w.line(llvm::formatv("{0},", rank).str());
  w.line(llvm::formatv("const_cast<halide_dimension_t*>({0}),", dimsName).str());
  w.line("nullptr");
  w.dedent();
  w.line("};");
  w.line();
  w.line(llvm::formatv("static inline void init_{0}(halide_buffer_t &buf, int32_t *base) {{",
                       base)
             .str());
  w.indent();
  w.line(llvm::formatv("buf = {0};", templName).str());
  w.line(llvm::formatv("buf.host = reinterpret_cast<uint8_t*>(base + {0});",
                       offsetElems)
             .str());
  w.dedent();
  w.line("}");
  w.line();
}

static std::string buildBaseName(const KernelInfo &kernel,
                                 const PortInfo &port,
                                 llvm::StringRef suffix) {
  std::string base = llvm::formatv("{0}_{1}{2}", kernel.tag,
                                   port.isInput ? "in" : "out",
                                   port.roleIndex)
                         .str();
  if (!suffix.empty()) {
    base += "_";
    base += suffix.str();
  }
  return base;
}

static std::string buildInitName(const KernelInfo &kernel,
                                 const PortInfo &port,
                                 llvm::StringRef suffix) {
  return "init_" + buildBaseName(kernel, port, suffix);
}

static void emitPortTemplatesForRegion(CodeWriter &w, const KernelInfo &kernel,
                                       const PortInfo &port,
                                       llvm::ArrayRef<int64_t> regionOutMin,
                                       llvm::ArrayRef<int64_t> regionOutExtent,
                                       bool allowGhostedPolicy,
                                       llvm::StringRef suffix,
                                       bool logPolicy) {
  const char *prefix = port.isInput ? "in" : "out";
  size_t rank = port.rank;
  llvm::SmallVector<int64_t, 4> strides(rank, 1);
  for (size_t i = rank; i-- > 1;) {
    strides[i - 1] = strides[i] * port.shape[i];
  }

  llvm::SmallVector<int64_t, 4> mins(rank, 0);
  llvm::SmallVector<int64_t, 4> extents(rank, 0);
  int64_t offsetElems = 0;

  if (port.isInput) {
    // P0: align to outMin (mins=-outMin, host offset 0).
    // P1: ghosted base (mins=-radius, host offset = outMin - radius).
    bool useGhosted = port.isGhosted && allowGhostedPolicy;
    const char *policy = useGhosted ? "P1" : "P0";
    for (size_t d = 0; d < rank; ++d) {
      if (useGhosted) {
        mins[d] = -kernel.radius[d];
      } else {
        mins[d] = -regionOutMin[d];
      }
      extents[d] = port.shape[d];
    }
    if (useGhosted) {
      for (size_t d = 0; d < rank; ++d) {
        offsetElems += (regionOutMin[d] - kernel.radius[d]) * strides[d];
      }
    }
    if (logPolicy) {
      llvm::outs() << "neptune-cc:   " << prefix << port.roleIndex
                   << " policy=" << policy;
      if (port.isGhosted) {
        if (useGhosted) {
          llvm::outs() << "(ghosted)";
        } else {
          llvm::outs() << "(ghosted->P0)";
        }
      }
      llvm::outs() << " mins=";
      printI64Array(llvm::outs(), mins);
      llvm::outs() << " host_offset_elems=" << offsetElems << "\n";
    }
  } else {
    for (size_t d = 0; d < rank; ++d) {
      mins[d] = 0;
      extents[d] = regionOutExtent[d];
      offsetElems += regionOutMin[d] * strides[d];
    }
    if (logPolicy) {
      llvm::outs() << "neptune-cc:   " << prefix << port.roleIndex
                   << " mins=";
      printI64Array(llvm::outs(), mins);
      llvm::outs() << " host_offset_elems=" << offsetElems << "\n";
    }
  }

  llvm::SmallVector<DimSpec, 4> dims;
  dims.reserve(rank);
  for (size_t i = 0; i < rank; ++i) {
    size_t d = rank - 1 - i;
    dims.push_back(DimSpec{mins[d], extents[d], strides[d]});
  }

  std::string base = buildBaseName(kernel, port, suffix);
  emitDimsTemplateAndInit(w, base, rank, dims, offsetElems);
}

static bool writeGlueCpp(llvm::StringRef glueDir,
                         llvm::ArrayRef<KernelInfo> kernels) {
  std::string cppPath = joinPath(glueDir, {"neptunecc_kernels.cpp"});
  std::error_code ec;
  llvm::raw_fd_ostream os(cppPath, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "neptune-cc: failed to open glue source '"
                 << cppPath << "': " << ec.message() << "\n";
    return false;
  }

  CodeWriter w(os);
  w.line("#include \"neptunecc_kernels.h\"");
  w.line("#include \"HalideRuntime.h\"");
  for (const auto &kernel : kernels) {
    if (!kernel.halideCompatible || !kernel.boundsValid) {
      continue;
    }
    w.line(llvm::formatv("#include \"../halide/{0}.h\"", kernel.tag).str());
  }
  w.line();
  w.line("namespace neptunecc {");
  w.line();

  w.openBlock("static inline halide_type_t i32_type() {");
  w.line("halide_type_t t;");
  w.line("t.code = halide_type_int;");
  w.line("t.bits = 32;");
  w.line("t.lanes = 1;");
  w.line("return t;");
  w.closeBlock("}");
  w.line();
  w.line("static const halide_type_t k_i32_type = i32_type();");
  w.line();
  w.line("// dim0 corresponds to the last IR index (fastest varying),");
  w.line("// dim1 corresponds to the first IR index. offset_elems uses");
  w.line("// row-major: offset = sum(outMin[d] * stride[d]).");
  w.line();
  const char *ghostedEnv = std::getenv("NEPTUNECC_GHOSTED_BASE");
  const bool allowGhostedPolicy =
      ghostedEnv && ghostedEnv[0] != '\0' && ghostedEnv[0] != '0';
  if (allowGhostedPolicy) {
    llvm::outs() << "neptune-cc: ghosted input policy enabled (P1)\n";
  }

  for (const auto &kernel : kernels) {
    if (!kernel.halideCompatible || !kernel.boundsValid) {
      continue;
    }
    const PortInfo *inPort = nullptr;
    const PortInfo *outPort = nullptr;
    for (const auto &port : kernel.ports) {
      if (port.isInput) {
        inPort = &port;
      } else {
        outPort = &port;
      }
    }
    if (!inPort || !outPort) {
      continue;
    }
    size_t rank = inPort->rank;
    if (kernel.outMin.size() != rank || kernel.outExtent.size() != rank ||
        kernel.radius.size() != rank) {
      llvm::errs() << "neptune-cc: mismatched stencil metadata for kernel '"
                   << kernel.tag << "'\n";
      continue;
    }
    logKernelSummary(kernel, *inPort, llvm::outs());
    logResidualDecision(kernel, llvm::outs());
    logOverlapInfo(kernel, llvm::outs());

    for (const auto &port : kernel.ports) {
      if (!isHalideArg(kernel, port.argIndex)) {
        continue;
      }
      emitPortTemplatesForRegion(w, kernel, port, kernel.outMin,
                                 kernel.outExtent, allowGhostedPolicy, "",
                                 true);
    }

    if (kernel.overlapSupported) {
      llvm::SmallVector<int64_t, 4> interiorExtent(rank, 0);
      for (size_t d = 0; d < rank; ++d) {
        interiorExtent[d] =
            kernel.overlapInteriorUb[d] - kernel.overlapInteriorLb[d];
      }
      for (const auto &port : kernel.ports) {
        if (!isHalideArg(kernel, port.argIndex)) {
          continue;
        }
        emitPortTemplatesForRegion(w, kernel, port, kernel.overlapInteriorLb,
                                   interiorExtent, allowGhostedPolicy,
                                   "interior", false);
      }
      for (const auto &face : kernel.overlapFaces) {
        llvm::SmallVector<int64_t, 4> faceExtent(rank, 0);
        for (size_t d = 0; d < rank; ++d) {
          faceExtent[d] = face.ub[d] - face.lb[d];
        }
        for (const auto &port : kernel.ports) {
          if (!isHalideArg(kernel, port.argIndex)) {
            continue;
          }
          emitPortTemplatesForRegion(w, kernel, port, face.lb, faceExtent,
                                     allowGhostedPolicy, face.suffix, false);
        }
      }
    }

    std::string signature;
    llvm::raw_string_ostream sigStream(signature);
    emitKernelSignature(sigStream, kernel);
    sigStream << " {";
    w.line(sigStream.str());
    w.indent();

    for (unsigned arg : kernel.halideArgs) {
      const PortInfo *port = findPortByArgIndex(kernel, arg);
      if (!port)
        continue;
      const char *prefix = port->isInput ? "in" : "out";
      w.line(llvm::formatv("halide_buffer_t {0}{1}_buf;", prefix,
                           port->roleIndex)
                 .str());
      w.line(llvm::formatv("{0}({1}{2}_buf, {3});",
                           buildInitName(kernel, *port, ""),
                           prefix, port->roleIndex, port->name)
                 .str());
    }

    if (kernel.partialCopyBoundary) {
      bool useWholeCopy = (kernel.boundaryCopyMode == "whole");
      if (kernel.boundaryCopyMode.empty()) {
        useWholeCopy = false;
      }
      for (const auto &pair : kernel.copyPairs) {
        const PortInfo *pairIn = findPortByArgIndex(kernel, pair.inArgIndex);
        const PortInfo *pairOut = findPortByArgIndex(kernel, pair.outArgIndex);
        if (!pairIn || !pairOut)
          continue;
        if (useWholeCopy) {
          emitWholeCopyND(w, pairIn->name, pairOut->name, pairIn->shape);
        } else if (pairIn->rank == 1) {
          emitSlabCopy1D(w, pairIn->name, pairOut->name, pairIn->shape,
                         kernel.outMin, kernel.outExtent);
        } else if (pairIn->rank == 2) {
          emitSlabCopy2D(w, pairIn->name, pairOut->name, pairIn->shape,
                         kernel.outMin, kernel.outExtent);
        } else if (pairIn->rank == 3) {
          emitSlabCopy3D(w, pairIn->name, pairOut->name, pairIn->shape,
                         kernel.outMin, kernel.outExtent);
        }
      }
    }

    std::string call;
    llvm::raw_string_ostream callStream(call);
    callStream << "return ::" << kernel.tag << "(";
    bool firstArg = true;
    for (unsigned arg : kernel.halideArgs) {
      const PortInfo *port = findPortByArgIndex(kernel, arg);
      if (!port)
        continue;
      if (!firstArg) {
        callStream << ", ";
      }
      firstArg = false;
      const char *prefix = port->isInput ? "in" : "out";
      callStream << "&" << prefix << port->roleIndex << "_buf";
    }
    callStream << ");";
    w.line(callStream.str());
    w.dedent();
    w.line("}");
    w.line();

    if (kernel.overlapSupported) {
      std::string interiorName = kernel.tag + "_interior";
      std::string interiorSig;
      llvm::raw_string_ostream interiorSigStream(interiorSig);
      emitKernelSignatureWithName(interiorSigStream, interiorName, kernel);
      interiorSigStream << " {";
      w.line(interiorSigStream.str());
      w.indent();
      for (unsigned arg : kernel.halideArgs) {
        const PortInfo *port = findPortByArgIndex(kernel, arg);
        if (!port)
          continue;
        const char *prefix = port->isInput ? "in" : "out";
        w.line(llvm::formatv("halide_buffer_t {0}{1}_buf;", prefix,
                             port->roleIndex)
                   .str());
        w.line(llvm::formatv("{0}({1}{2}_buf, {3});",
                             buildInitName(kernel, *port, "interior"),
                             prefix, port->roleIndex, port->name)
                   .str());
      }
      w.line(callStream.str());
      w.dedent();
      w.line("}");
      w.line();

      for (const auto &face : kernel.overlapFaces) {
        std::string faceName = kernel.tag + "_" + face.suffix;
        std::string faceSig;
        llvm::raw_string_ostream faceSigStream(faceSig);
        emitKernelSignatureWithName(faceSigStream, faceName, kernel);
        faceSigStream << " {";
        w.line(faceSigStream.str());
        w.indent();
        for (unsigned arg : kernel.halideArgs) {
          const PortInfo *port = findPortByArgIndex(kernel, arg);
          if (!port)
            continue;
          const char *prefix = port->isInput ? "in" : "out";
          w.line(llvm::formatv("halide_buffer_t {0}{1}_buf;", prefix,
                               port->roleIndex)
                     .str());
          w.line(llvm::formatv("{0}({1}{2}_buf, {3});",
                               buildInitName(kernel, *port, face.suffix),
                               prefix, port->roleIndex, port->name)
                     .str());
        }
        w.line(callStream.str());
        w.dedent();
        w.line("}");
        w.line();
      }
    }
  }

  w.line("} // namespace neptunecc");
  return true;
}

static bool writeGlueCMake(llvm::StringRef glueDir,
                           llvm::ArrayRef<KernelInfo> kernels) {
  std::string cmakePath = joinPath(glueDir, {"neptunecc_generated.cmake"});
  std::error_code ec;
  llvm::raw_fd_ostream os(cmakePath, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "neptune-cc: failed to open glue cmake '"
                 << cmakePath << "': " << ec.message() << "\n";
    return false;
  }

  os << "set(NEPTUNECC_GENERATED_DIR \"${CMAKE_CURRENT_LIST_DIR}\")\n";
  os << "set(NEPTUNECC_HALIDE_DIR \"${CMAKE_CURRENT_LIST_DIR}/../halide\")\n\n";
  os << "add_library(neptunecc_glue STATIC\n";
  os << "  \"${NEPTUNECC_GENERATED_DIR}/neptunecc_kernels.cpp\"\n";
  os << ")\n\n";
  os << "target_include_directories(neptunecc_glue PUBLIC\n";
  os << "  \"${NEPTUNECC_GENERATED_DIR}\"\n";
  os << "  \"${NEPTUNECC_HALIDE_DIR}\"\n";
  os << ")\n\n";
  os << "# Halide kernel libraries (AOT)\n";
  for (const auto &kernel : kernels) {
    if (!kernel.halideCompatible || !kernel.boundsValid) {
      if (!kernel.halideCompatible) {
        llvm::errs() << "neptune-cc: skipping CMake target for kernel '"
                     << kernel.tag
                     << "' (missing apply-mapped input/output)\n";
      } else {
        llvm::errs() << "neptune-cc: skipping CMake target for kernel '"
                     << kernel.tag << "' (no valid stencil bounds)\n";
      }
      continue;
    }
    os << "add_library(neptunecc_" << kernel.tag
       << " STATIC IMPORTED GLOBAL)\n";
    os << "set(_neptunecc_" << kernel.tag << "_lib \"${NEPTUNECC_HALIDE_DIR}/lib"
       << kernel.tag << ".a\")\n";
    os << "if(NOT EXISTS \"${_neptunecc_" << kernel.tag << "_lib}\")\n";
    os << "  set(_neptunecc_" << kernel.tag
       << "_lib \"${NEPTUNECC_HALIDE_DIR}/" << kernel.tag << ".a\")\n";
    os << "endif()\n";
    os << "set_target_properties(neptunecc_" << kernel.tag
       << " PROPERTIES\n";
    os << "  IMPORTED_LOCATION \"${_neptunecc_" << kernel.tag << "_lib}\"\n";
    os << ")\n";
    os << "target_link_libraries(neptunecc_glue PUBLIC neptunecc_" << kernel.tag
       << ")\n";
  }
  os << "\n# If needed, link Halide runtime here.\n";
  return true;
}

static bool writeHalideHelperFile(llvm::StringRef halideDir) {
  std::string helperPath =
      joinPath(halideDir, {"NeptuneHalideHelpers.h"});
  std::error_code ec;
  llvm::raw_fd_ostream os(helperPath, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "neptune-cc: failed to open Halide helper header '"
                 << helperPath << "': " << ec.message() << "\n";
    return false;
  }

  os << "#pragma once\n";
  os << "#include \"Halide.h\"\n";
  os << "#include <string>\n";
  os << "#include <vector>\n";
  os << "\nnamespace neptune_halide {\n\n";
  os << "inline void push_arg(std::vector<Halide::Argument> &args,\n";
  os << "                     const Halide::Argument &arg) {\n";
  os << "  args.push_back(arg);\n";
  os << "}\n\n";
  os << "inline void assign(Halide::FuncRef &ref, const Halide::Expr &expr) {\n";
  os << "  ref = expr;\n";
  os << "}\n\n";
  os << "inline void schedule_1d(Halide::Func &func, int Tx, int VL,\n";
  os << "                        int enableParallel, int threads, int unroll) {\n";
  os << "  Halide::Var x(\"d0\"), xo(\"xo\"), xi(\"xi\");\n";
  os << "  if (Tx > 0) {\n";
  os << "    func.split(x, xo, xi, Tx);\n";
  os << "  } else {\n";
  os << "    xo = x;\n";
  os << "    xi = x;\n";
  os << "  }\n";
  os << "  if (VL > 1) func.vectorize(xi, VL);\n";
  os << "  if (enableParallel && threads > 1) func.parallel(xo);\n";
  os << "  if (unroll > 1) func.unroll(xi, unroll);\n";
  os << "  func.compute_root();\n";
  os << "}\n\n";
  os << "inline void schedule_2d(Halide::Func &func, int Tx, int Ty, int VL,\n";
  os << "                        int Uy, int enableParallel, int threads) {\n";
  os << "  Halide::Var x(\"d0\"), y(\"d1\"), xo(\"xo\"), yo(\"yo\"), xi(\"xi\"),\n";
  os << "      yi(\"yi\");\n";
  os << "  if (Tx > 0 && Ty > 0) {\n";
  os << "    func.tile(x, y, xo, yo, xi, yi, Tx, Ty);\n";
  os << "  } else {\n";
  os << "    xo = x;\n";
  os << "    yo = y;\n";
  os << "    xi = x;\n";
  os << "    yi = y;\n";
  os << "  }\n";
  os << "  if (VL > 1) func.vectorize(xi, VL);\n";
  os << "  if (enableParallel && threads > 1) func.parallel(yo);\n";
  os << "  if (Uy > 1) func.unroll(yi, Uy);\n";
  os << "  func.compute_root();\n";
  os << "}\n\n";
  os << "inline void schedule_3d(Halide::Func &func, int Tx, int Ty, int Tz,\n";
  os << "                        int VL, int Uy, int par_dim,\n";
  os << "                        int enableParallel, int threads) {\n";
  os << "  Halide::Var x(\"d0\"), y(\"d1\"), z(\"d2\"), xo(\"xo\"), yo(\"yo\"),\n";
  os << "      zo(\"zo\"), xi(\"xi\"), yi(\"yi\"), zi(\"zi\");\n";
  os << "  if (Tx > 0 && Ty > 0 && Tz > 0) {\n";
  os << "    func.split(x, xo, xi, Tx);\n";
  os << "    func.split(y, yo, yi, Ty);\n";
  os << "    func.split(z, zo, zi, Tz);\n";
  os << "    func.reorder(xi, yi, zi, xo, yo, zo);\n";
  os << "  } else {\n";
  os << "    xo = x;\n";
  os << "    yo = y;\n";
  os << "    zo = z;\n";
  os << "    xi = x;\n";
  os << "    yi = y;\n";
  os << "    zi = z;\n";
  os << "  }\n";
  os << "  if (VL > 1) func.vectorize(xi, VL);\n";
  os << "  if (enableParallel && threads > 1) {\n";
  os << "    if (par_dim == 2) func.parallel(zo);\n";
  os << "    else if (par_dim == 1) func.parallel(yo);\n";
  os << "  }\n";
  os << "  if (Uy > 1) func.unroll(yi, Uy);\n";
  os << "  func.compute_root();\n";
  os << "}\n\n";
  os << "inline void compile(Halide::Func &func, const std::string &prefix,\n";
  os << "                    const std::vector<Halide::Argument> &args,\n";
  os << "                    const std::string &fn, const Halide::Target &target) {\n";
  os << "  func.compile_to_static_library(prefix, args, fn, target);\n";
  os << "}\n\n";
  os << "} // namespace neptune_halide\n";
  return true;
}

} // namespace

bool writeGlue(const EventDB &db, llvm::StringRef outDir) {
  llvm::SmallVector<KernelInfo, 8> kernels;
  if (!collectKernelInfos(db, kernels)) {
    return false;
  }

  std::string glueDir = joinPath(outDir, {"glue"});
  if (!ensureDir(glueDir, "glue")) {
    return false;
  }

  if (!writeGlueHeader(glueDir, kernels)) {
    return false;
  }
  if (!writeGlueCpp(glueDir, kernels)) {
    return false;
  }
  if (!writeGlueCMake(glueDir, kernels)) {
    return false;
  }

  return true;
}

bool writeHalideHelper(llvm::StringRef outDir) {
  std::string halideDir = joinPath(outDir, {"halide"});
  if (!ensureDir(halideDir, "halide")) {
    return false;
  }
  return writeHalideHelperFile(halideDir);
}

bool rewriteKernelSources(const EventDB &db, llvm::StringRef outDir) {
  llvm::SmallVector<KernelInfo, 8> kernels;
  if (!collectKernelInfos(db, kernels)) {
    return false;
  }

  llvm::StringMap<KernelDecisionInfo> decisions;
  for (const auto &kernel : kernels) {
    KernelDecisionInfo decision;
    if (!kernel.halideCompatible) {
      decision.mode = "SkipRewrite";
      decision.reason = "missing apply-mapped input/output";
    } else if (!kernel.boundsValid) {
      decision.mode = "SkipRewrite";
      decision.reason = "no valid stencil bounds";
    } else if (kernel.hasResidualScf && !kernel.partialCopyBoundary) {
      decision.mode = "SkipRewrite";
      decision.reason =
          kernel.residualReason.empty()
              ? "residual_scf_not_copy_boundary"
              : ("residual_scf_not_copy_boundary: " + kernel.residualReason);
    } else if (kernel.hasResidualScf && kernel.partialCopyBoundary) {
      decision.mode = "PartialRewrite";
      decision.boundaryCopyMode = kernel.boundaryCopyMode;
    } else {
      decision.mode = "PureHalide";
    }
    decisions[kernel.tag] = std::move(decision);
  }

  llvm::StringMap<const KernelInfo *> kernelByTag;
  for (const auto &kernel : kernels) {
    kernelByTag[kernel.tag] = &kernel;
  }

  llvm::StringMap<llvm::SmallVector<Replacement, 4>> replacementsByFile;
  llvm::StringSet<> filesToRewrite;
  llvm::StringSet<> filesWithKernelReplace;
  auto whitespaceOnly = [](llvm::StringRef text) {
    for (char c : text) {
      if (c != ' ' && c != '\t' && c != '\n' && c != '\r')
        return false;
    }
    return true;
  };

  auto findKernelInterval =
      [&](llvm::StringRef filePath, llvm::StringRef tag,
          unsigned blockBegin, unsigned blockEnd) -> const KernelInterval * {
    const KernelInterval *match = nullptr;
    for (const auto &kernel : db.kernels) {
      if (kernel.begin.tag != tag || kernel.begin.filePath != filePath)
        continue;
      if (kernel.begin.fileOffset < blockBegin ||
          kernel.end.fileOffset > blockEnd)
        continue;
      if (match)
        return nullptr;
      match = &kernel;
    }
    return match;
  };

  auto findHaloInterval =
      [&](llvm::StringRef filePath, llvm::StringRef tag,
          unsigned blockBegin, unsigned blockEnd) -> const HaloInterval * {
    const HaloInterval *match = nullptr;
    for (const auto &halo : db.halos) {
      if (halo.begin.tag != tag || halo.begin.filePath != filePath)
        continue;
      if (halo.begin.fileOffset < blockBegin ||
          halo.end.fileOffset > blockEnd)
        continue;
      if (match)
        return nullptr;
      match = &halo;
    }
    return match;
  };

  llvm::StringMap<llvm::SmallVector<const OverlapInterval *, 4>>
      overlapsByFile;
  llvm::StringMap<bool> overlapCoverage;
  for (const auto &ov : db.overlaps) {
    if (!ov.blockBegin.isValid() || !ov.blockEnd.isValid())
      continue;
    if (ov.begin.filePath.empty())
      continue;
    overlapsByFile[ov.begin.filePath].push_back(&ov);
    if (!ov.kernelTag.empty())
      overlapCoverage[ov.kernelTag] = true;
  }
  for (const auto &kernel : db.kernels) {
    if (!kernel.blockBegin.isValid() || !kernel.blockEnd.isValid()) {
      continue;
    }
    llvm::StringRef tag = kernel.begin.tag;
    if (tag.empty()) {
      continue;
    }
    llvm::StringRef filePath = kernel.begin.filePath;
    if (!filePath.empty()) {
      filesToRewrite.insert(filePath);
    }

    auto it = kernelByTag.find(tag);
    if (it == kernelByTag.end()) {
      llvm::errs() << "neptune-cc: no kernel metadata for tag '" << tag
                   << "'\n";
      continue;
    }
    if (!it->second->halideCompatible || !it->second->boundsValid) {
      if (!it->second->halideCompatible) {
        llvm::errs() << "neptune-cc: skipping rewrite for kernel '" << tag
                     << "' (missing apply-mapped input/output)\n";
      } else {
        llvm::errs() << "neptune-cc: skipping rewrite for kernel '" << tag
                     << "' (no valid stencil bounds)\n";
      }
      continue;
    }
    if (filePath.empty()) {
      llvm::errs() << "neptune-cc: missing source path for kernel '" << tag
                   << "'\n";
      auto decIt = decisions.find(tag);
      if (decIt != decisions.end() && !decIt->second.overlapCovered) {
        decIt->second.mode = "SkipRewrite";
        decIt->second.reason = "missing source path";
      }
      continue;
    }

    bool coveredByOverlap = false;
    auto ovIt = overlapsByFile.find(filePath);
    if (ovIt != overlapsByFile.end()) {
      for (const OverlapInterval *ov : ovIt->second) {
        if (ov->kernelTag != tag)
          continue;
        if (kernel.begin.fileOffset >= ov->blockBeginOffset &&
            kernel.end.fileOffset <= ov->blockEndOffset) {
          coveredByOverlap = true;
          break;
        }
      }
    }
    if (coveredByOverlap) {
      auto decIt = decisions.find(tag);
      if (decIt != decisions.end()) {
        decIt->second.overlapCovered = true;
        if (decIt->second.mode != "SkipRewrite") {
          decIt->second.mode = "OverlapRewrite";
          decIt->second.reason = "pending";
        }
      }
      continue;
    }

    auto bufferOrErr = llvm::MemoryBuffer::getFile(filePath);
    if (!bufferOrErr) {
      llvm::errs() << "neptune-cc: failed to read source '" << filePath
                   << "': " << bufferOrErr.getError().message() << "\n";
      auto decIt = decisions.find(tag);
      if (decIt != decisions.end() && !decIt->second.overlapCovered) {
        decIt->second.mode = "SkipRewrite";
        decIt->second.reason = "failed to read source";
      }
      continue;
    }
    llvm::StringRef contentRef = bufferOrErr.get()->getBuffer();

    size_t beginOffset = static_cast<size_t>(kernel.begin.fileOffset);
    size_t endOffset = static_cast<size_t>(kernel.end.fileOffset);
    if (beginOffset > contentRef.size() || endOffset > contentRef.size()) {
      llvm::errs() << "neptune-cc: kernel offsets out of range for '" << tag
                   << "'\n";
      auto decIt = decisions.find(tag);
      if (decIt != decisions.end() && !decIt->second.overlapCovered) {
        decIt->second.mode = "SkipRewrite";
        decIt->second.reason = "kernel offsets out of range";
      }
      continue;
    }

    size_t start = findLineStart(contentRef, beginOffset);
    size_t end = findLineEnd(contentRef, endOffset);
    if (start >= end) {
      llvm::errs() << "neptune-cc: invalid kernel range for '" << tag << "'\n";
      auto decIt = decisions.find(tag);
      if (decIt != decisions.end() && !decIt->second.overlapCovered) {
        decIt->second.mode = "SkipRewrite";
        decIt->second.reason = "invalid kernel range";
      }
      continue;
    }

    size_t indentEnd = start;
    while (indentEnd < contentRef.size() &&
           (contentRef[indentEnd] == ' ' || contentRef[indentEnd] == '\t')) {
      ++indentEnd;
    }
    llvm::StringRef indent =
        contentRef.slice(start, std::min(indentEnd, contentRef.size()));
    std::string newline = detectNewline(contentRef);

    Replacement repl;
    repl.start = start;
    repl.end = end;
    llvm::raw_string_ostream rso(repl.text);
    rso << indent;
    emitKernelCall(rso, *it->second);
    rso << newline;
    replacementsByFile[filePath].push_back(std::move(repl));
    filesWithKernelReplace.insert(filePath);
  }

  auto markOverlapSkip = [&](llvm::StringRef kernelTag,
                             llvm::StringRef reason) {
    auto it = decisions.find(kernelTag);
    if (it == decisions.end())
      return;
    KernelDecisionInfo &dec = it->second;
    dec.overlapCovered = true;
    if (dec.overlapRewritten)
      return;
    if (dec.mode == "SkipRewrite" && !dec.reason.empty())
      return;
    dec.mode = "SkipRewrite";
    dec.reason = std::string("overlap_skip: ") + reason.str();
  };

  auto markOverlapSuccess = [&](llvm::StringRef kernelTag) {
    auto it = decisions.find(kernelTag);
    if (it == decisions.end())
      return;
    KernelDecisionInfo &dec = it->second;
    dec.overlapCovered = true;
    dec.overlapRewritten = true;
    dec.mode = "OverlapRewrite";
    dec.reason.clear();
  };

  for (auto &entry : overlapsByFile) {
    llvm::StringRef filePath = entry.getKey();
    auto bufferOrErr = llvm::MemoryBuffer::getFile(filePath);
    if (!bufferOrErr) {
      llvm::errs() << "neptune-cc: failed to read source '" << filePath
                   << "': " << bufferOrErr.getError().message() << "\n";
      continue;
    }
    llvm::StringRef contentRef = bufferOrErr.get()->getBuffer();

    for (const OverlapInterval *ov : entry.getValue()) {
      if (ov->kernelTag.empty() || ov->haloTag.empty()) {
        llvm::errs() << "neptune-cc: skip overlap rewrite for tag '"
                     << ov->begin.tag << "' (missing halo/kernel)\n";
        if (!ov->kernelTag.empty()) {
          markOverlapSkip(ov->kernelTag, "missing halo/kernel");
        }
        continue;
      }
      if (ov->blockBeginOffset >= ov->blockEndOffset) {
        llvm::errs() << "neptune-cc: skip overlap rewrite for tag '"
                     << ov->begin.tag << "' (invalid block)\n";
        markOverlapSkip(ov->kernelTag, "invalid block");
        continue;
      }

      auto kInfoIt = kernelByTag.find(ov->kernelTag);
      if (kInfoIt == kernelByTag.end()) {
        llvm::errs() << "neptune-cc: skip overlap rewrite for tag '"
                     << ov->begin.tag << "' (missing kernel info)\n";
        markOverlapSkip(ov->kernelTag, "missing kernel info");
        continue;
      }
      const KernelInfo *kinfo = kInfoIt->second;
      if (!kinfo->boundsValid || !kinfo->overlapSupported) {
        llvm::errs() << "neptune-cc: skip overlap rewrite for tag '"
                     << ov->begin.tag << "' ("
                     << (kinfo->overlapReason.empty()
                             ? "overlap unsupported"
                             : kinfo->overlapReason)
                     << ")\n";
        markOverlapSkip(ov->kernelTag,
                        kinfo->overlapReason.empty()
                            ? "overlap unsupported"
                            : kinfo->overlapReason);
        continue;
      }

      const KernelInterval *kInterval = findKernelInterval(
          filePath, ov->kernelTag, ov->blockBeginOffset, ov->blockEndOffset);
      if (!kInterval) {
        llvm::errs() << "neptune-cc: skip overlap rewrite for tag '"
                     << ov->begin.tag << "' (kernel not uniquely inside)\n";
        markOverlapSkip(ov->kernelTag, "kernel not uniquely inside");
        continue;
      }
      const HaloInterval *hInterval = findHaloInterval(
          filePath, ov->haloTag, ov->blockBeginOffset, ov->blockEndOffset);
      if (!hInterval) {
        llvm::errs() << "neptune-cc: skip overlap rewrite for tag '"
                     << ov->begin.tag << "' (halo not uniquely inside)\n";
        markOverlapSkip(ov->kernelTag, "halo not uniquely inside");
        continue;
      }

      size_t blockBodyStart = findLineEnd(contentRef, ov->blockBeginOffset);
      size_t blockBodyEnd = findLineStart(contentRef, ov->blockEndOffset);
      size_t haloBeginLineEnd =
          findLineEnd(contentRef, hInterval->begin.fileOffset);
      size_t haloEndLineStart =
          findLineStart(contentRef, hInterval->end.fileOffset);
      size_t haloEndLineEnd =
          findLineEnd(contentRef, hInterval->end.fileOffset);
      size_t kernelBeginLineStart =
          findLineStart(contentRef, kInterval->begin.fileOffset);
      size_t kernelEndLineEnd =
          findLineEnd(contentRef, kInterval->end.fileOffset);

      if (!(blockBodyStart <= haloBeginLineEnd &&
            haloBeginLineEnd <= kernelBeginLineStart &&
            kernelBeginLineStart <= kernelEndLineEnd &&
            kernelEndLineEnd <= haloEndLineStart &&
            haloEndLineStart <= blockBodyEnd)) {
        llvm::errs() << "neptune-cc: skip overlap rewrite for tag '"
                     << ov->begin.tag << "' (unexpected ordering)\n";
        markOverlapSkip(ov->kernelTag, "unexpected ordering");
        continue;
      }

      // Only allow: halo-begin statements, kernel block, halo-end statements.
      llvm::StringRef midGap =
          contentRef.slice(kernelEndLineEnd, haloEndLineStart);
      if (!whitespaceOnly(midGap)) {
        llvm::errs() << "neptune-cc: skip overlap rewrite for tag '"
                     << ov->begin.tag << "' (extra statements between kernel "
                     << "and halo end)\n";
        markOverlapSkip(ov->kernelTag, "extra statements between kernel/halo");
        continue;
      }

      llvm::StringRef beginStatements =
          contentRef.slice(haloBeginLineEnd, kernelBeginLineStart);
      llvm::StringRef endStatements =
          contentRef.slice(haloEndLineEnd, blockBodyEnd);

      size_t indentStart = findLineStart(contentRef, ov->blockBeginOffset);
      size_t indentEnd = indentStart;
      while (indentEnd < contentRef.size() &&
             (contentRef[indentEnd] == ' ' ||
              contentRef[indentEnd] == '\t')) {
        ++indentEnd;
      }
      llvm::StringRef indent =
          contentRef.slice(indentStart,
                           std::min(indentEnd, contentRef.size()));
      std::string newline = detectNewline(contentRef);

      Replacement repl;
      repl.start = ov->blockBeginOffset;
      repl.end = ov->blockEndOffset;
      llvm::raw_string_ostream rso(repl.text);
      // Rewrite order: halo-begin stmts -> interior -> halo-end stmts -> faces.
      rso << indent << "{" << newline;
      rso << beginStatements;
      rso << indent << "  neptunecc::" << kinfo->tag << "_interior(";
      for (size_t i = 0; i < kinfo->ports.size(); ++i) {
        if (i)
          rso << ", ";
        emitArrayArg(rso, kinfo->ports[i].name, kinfo->ports[i].rank);
      }
      rso << ");" << newline;
      rso << endStatements;
      for (const auto &face : kinfo->overlapFaces) {
        rso << indent << "  neptunecc::" << kinfo->tag << "_"
            << face.suffix << "(";
        for (size_t i = 0; i < kinfo->ports.size(); ++i) {
          if (i)
            rso << ", ";
          emitArrayArg(rso, kinfo->ports[i].name, kinfo->ports[i].rank);
        }
        rso << ");" << newline;
      }
      rso << indent << "}" << newline;
      replacementsByFile[filePath].push_back(std::move(repl));
      filesToRewrite.insert(filePath);
      filesWithKernelReplace.insert(filePath);
      markOverlapSuccess(ov->kernelTag);
    }
  }

  emitScheduleDebug(outDir, kernels, decisions);
  if (filesToRewrite.empty()) {
    return true;
  }

  std::string rewriteDir = joinPath(outDir, {"rewritten"});
  if (!ensureDir(rewriteDir, "rewritten")) {
    return false;
  }

  for (auto &entry : filesToRewrite) {
    llvm::StringRef filePath = entry.getKey();
    auto bufferOrErr = llvm::MemoryBuffer::getFile(filePath);
    if (!bufferOrErr) {
      llvm::errs() << "neptune-cc: failed to read source '" << filePath
                   << "': " << bufferOrErr.getError().message() << "\n";
      continue;
    }

    std::string content = bufferOrErr.get()->getBuffer().str();
    auto replIt = replacementsByFile.find(filePath);
    if (replIt != replacementsByFile.end()) {
      auto &repls = replIt->second;
      std::stable_sort(repls.begin(), repls.end(),
                       [](const Replacement &a, const Replacement &b) {
                         return a.start > b.start;
                       });

      for (const auto &repl : repls) {
        if (repl.end > content.size()) {
          llvm::errs()
              << "neptune-cc: replacement out of range for '" << filePath
              << "'\n";
          return false;
        }
        content.replace(repl.start, repl.end - repl.start, repl.text);
      }

      if (filesWithKernelReplace.contains(filePath)) {
        ensureKernelInclude(content);
      }
    }

    llvm::SmallString<256> outPath(rewriteDir);
    llvm::sys::path::append(outPath, llvm::sys::path::filename(filePath));
    std::error_code ec;
    llvm::raw_fd_ostream os(outPath, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      llvm::errs() << "neptune-cc: failed to write rewritten source '"
                   << outPath << "': " << ec.message() << "\n";
      return false;
    }
    os << content;
  }

  return true;
}

bool writeHalideGenerators(const EventDB &db, llvm::StringRef outDir,
                           bool emitEmitcMLIR, bool emitHalideCpp) {
  if (!emitEmitcMLIR && !emitHalideCpp) {
    return true;
  }
  if (!db.kernelModule) {
    return true;
  }

  std::string halideDir = joinPath(outDir, {"halide"});
  if (!ensureDir(halideDir, "halide")) {
    return false;
  }

  mlir::ModuleOp module = db.kernelModule.get().clone();
  mlir::MLIRContext *ctx = module.getContext();

  mlir::PassManager pm(ctx);
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::Neptune::NeptuneIR::createSCFBoundarySimplifyPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::Neptune::NeptuneIR::createSCFToNeptuneIRPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::Neptune::NeptuneIR::createNeptuneIRInferHalideSchedulePass());
  mlir::Neptune::NeptuneIR::NeptuneIREmitCHalidePassOptions opts;
  opts.outFile = "halide_kernels";
  pm.addPass(mlir::Neptune::NeptuneIR::createNeptuneIREmitCHalidePass(opts));

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "neptune-cc: halide pass pipeline failed\n";
    return false;
  }

  if (emitEmitcMLIR) {
    std::string emitcPath = joinPath(halideDir, {"emitc.mlir"});
    std::error_code ec;
    llvm::raw_fd_ostream emitcFile(emitcPath, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      llvm::errs() << "neptune-cc: failed to open emitc mlir '"
                   << emitcPath << "': " << ec.message() << "\n";
      return false;
    }
    module.print(emitcFile);
    emitcFile << "\n";
  }

  if (emitHalideCpp) {
    std::string cppPath = joinPath(halideDir, {"halide_kernels.cpp"});
    std::error_code ec;
    llvm::raw_fd_ostream cppFile(cppPath, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      llvm::errs() << "neptune-cc: failed to open halide cpp '" << cppPath
                   << "': " << ec.message() << "\n";
      return false;
    }
    if (mlir::failed(
            mlir::emitc::translateToCpp(module, cppFile, false,
                                        "halide_kernels"))) {
      llvm::errs() << "neptune-cc: failed to translate emitc to C++\n";
      return false;
    }
  }

  return true;
}

} // namespace neptune
