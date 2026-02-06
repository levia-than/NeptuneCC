// Simplify boundary control flow for stencil loops.
#include "Dialect/NeptuneIR/NeptuneIRDialect.h"
#include "Passes/NeptuneIRPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseSet.h"

#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

using namespace mlir;

namespace mlir::Neptune::NeptuneIR {
#define GEN_PASS_DEF_SCFBOUNDARYSIMPLIFYPASS
#include "Passes/NeptuneIRPasses.h.inc"
} // namespace mlir::Neptune::NeptuneIR

namespace {
using namespace mlir::Neptune::NeptuneIR;

// Summary snapshot used for logging the decision and geometry.
struct SummaryInfo {
  int64_t rank = 0;
  std::vector<int64_t> lb;
  std::vector<int64_t> ub;
  int64_t r = 0;
  bool peeled = false;
  bool unswitched = false;
  std::string reason;
};

static bool matchConstantIndex(Value v, int64_t &out) {
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    auto attr = dyn_cast<IntegerAttr>(cst.getValue());
    if (!attr)
      return false;
    out = attr.getInt();
    return true;
  }
  return false;
}

static bool evalIndexExpr(Value v, int64_t &out) {
  // Conservative evaluator for index expressions used in loop bounds.
  if (matchConstantIndex(v, out))
    return true;

  if (auto addi = v.getDefiningOp<arith::AddIOp>()) {
    int64_t lhs = 0;
    int64_t rhs = 0;
    if (!evalIndexExpr(addi.getLhs(), lhs))
      return false;
    if (!evalIndexExpr(addi.getRhs(), rhs))
      return false;
    out = lhs + rhs;
    return true;
  }

  if (auto subi = v.getDefiningOp<arith::SubIOp>()) {
    int64_t lhs = 0;
    int64_t rhs = 0;
    if (!evalIndexExpr(subi.getLhs(), lhs))
      return false;
    if (!evalIndexExpr(subi.getRhs(), rhs))
      return false;
    out = lhs - rhs;
    return true;
  }

  if (auto cast = v.getDefiningOp<arith::IndexCastOp>()) {
    return evalIndexExpr(cast.getIn(), out);
  }

  return false;
}

static bool matchIndexOffset(Value idx, Value iv, int64_t &offset) {
  // Accept iv, iv + c, iv - c only.
  if (idx == iv) {
    offset = 0;
    return true;
  }

  if (auto addi = idx.getDefiningOp<arith::AddIOp>()) {
    int64_t cst = 0;
    if (addi.getLhs() == iv && matchConstantIndex(addi.getRhs(), cst)) {
      offset = cst;
      return true;
    }
    if (addi.getRhs() == iv && matchConstantIndex(addi.getLhs(), cst)) {
      offset = cst;
      return true;
    }
  }

  if (auto subi = idx.getDefiningOp<arith::SubIOp>()) {
    int64_t cst = 0;
    if (subi.getLhs() == iv && matchConstantIndex(subi.getRhs(), cst)) {
      offset = -cst;
      return true;
    }
  }

  return false;
}

static std::string joinI64(const std::vector<int64_t> &vals) {
  std::string out = "[";
  for (size_t i = 0; i < vals.size(); ++i) {
    if (i)
      out += ",";
    out += std::to_string(vals[i]);
  }
  out += "]";
  return out;
}

// Centralized logger for this pass.
static void logLine(const std::string &msg) {
  llvm::outs() << "[neptuneir-scf-boundary-simplify] " << msg << "\n";
}

// Kernel functions are identified by neptunecc.tag.
static bool isKernelFunc(func::FuncOp func) {
  return func->hasAttr("neptunecc.tag");
}

// Checks that a value is invariant across a loop nest.
static bool isLoopInvariantAcross(Value cond,
                                  const std::vector<scf::ForOp> &loops) {
  for (scf::ForOp loop : loops) {
    if (!loop.isDefinedOutsideOfLoop(cond))
      return false;
  }
  return true;
}

// Scans a region and records stores; rejects side effects or unsupported ops.
static bool collectStoresAndCheckRegion(Region &region,
                                        llvm::DenseSet<Value> &stores,
                                        std::string &reason) {
  // Only allow side-effect free ops plus memref.load/store and simple arith.
  for (Block &block : region) {
    for (Operation &op : block) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;

      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        if (!collectStoresAndCheckRegion(forOp.getRegion(), stores, reason))
          return false;
        continue;
      }

      if (isa<scf::IfOp>(op)) {
        reason = "nested scf.if not supported";
        return false;
      }

      if (auto store = dyn_cast<memref::StoreOp>(op)) {
        stores.insert(store.getMemref());
        continue;
      }

      if (isa<memref::LoadOp>(op) || isa<arith::ConstantOp>(op) ||
          isa<arith::AddIOp>(op) || isa<arith::SubIOp>(op) ||
          isa<arith::MulIOp>(op) || isa<arith::DivSIOp>(op) ||
          isa<arith::DivUIOp>(op) || isa<arith::IndexCastOp>(op) ||
          isa<arith::CmpIOp>(op) || isa<arith::SelectOp>(op) ||
          isa<arith::AndIOp>(op) || isa<arith::OrIOp>(op) ||
          isa<arith::XOrIOp>(op)) {
        continue;
      }

      if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
        if (!iface.hasNoEffect()) {
          reason = "side-effect op";
          return false;
        }
        continue;
      }

      reason = "unsupported op in region";
      return false;
    }
  }

  return true;
}

static bool checkBranchAccesses(Region &region, Value ivI, Value ivJ,
                                int64_t r, std::string &reason) {
  // Ensure accesses are within radius and use only iv +/- const.
  for (Block &block : region) {
    for (Operation &op : block) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;

      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        if (!checkBranchAccesses(forOp.getRegion(), ivI, ivJ, r, reason))
          return false;
        continue;
      }

      if (isa<scf::IfOp>(op)) {
        reason = "nested scf.if not supported";
        return false;
      }

      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        auto memrefTy = dyn_cast<MemRefType>(load.getMemref().getType());
        if (!memrefTy || memrefTy.getRank() != 2) {
          reason = "memref rank mismatch";
          return false;
        }
        if (load.getIndices().size() != 2) {
          reason = "memref indices mismatch";
          return false;
        }
        int64_t offI = 0;
        int64_t offJ = 0;
        if (!matchIndexOffset(load.getIndices()[0], ivI, offI) ||
            !matchIndexOffset(load.getIndices()[1], ivJ, offJ)) {
          reason = "unsupported load index";
          return false;
        }
        if (std::llabs(offI) > r || std::llabs(offJ) > r) {
          reason = "load offset exceeds radius";
          return false;
        }
        continue;
      }

      if (auto store = dyn_cast<memref::StoreOp>(op)) {
        auto memrefTy = dyn_cast<MemRefType>(store.getMemref().getType());
        if (!memrefTy || memrefTy.getRank() != 2) {
          reason = "memref rank mismatch";
          return false;
        }
        if (store.getIndices().size() != 2) {
          reason = "memref indices mismatch";
          return false;
        }
        int64_t offI = 0;
        int64_t offJ = 0;
        if (!matchIndexOffset(store.getIndices()[0], ivI, offI) ||
            !matchIndexOffset(store.getIndices()[1], ivJ, offJ)) {
          reason = "unsupported store index";
          return false;
        }
        if (std::llabs(offI) > r || std::llabs(offJ) > r) {
          reason = "store offset exceeds radius";
          return false;
        }
        continue;
      }

      if (isa<arith::ConstantOp>(op) || isa<arith::AddIOp>(op) ||
          isa<arith::SubIOp>(op) || isa<arith::MulIOp>(op) ||
          isa<arith::DivSIOp>(op) || isa<arith::DivUIOp>(op) ||
          isa<arith::IndexCastOp>(op) || isa<arith::CmpIOp>(op) ||
          isa<arith::SelectOp>(op) || isa<arith::AndIOp>(op) ||
          isa<arith::OrIOp>(op) || isa<arith::XOrIOp>(op)) {
        continue;
      }

      if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
        if (!iface.hasNoEffect()) {
          reason = "side-effect op";
          return false;
        }
        continue;
      }

      reason = "unsupported op in branch";
      return false;
    }
  }

  return true;
}

static bool checkBranchAccesses3D(Region &region, Value ivI, Value ivJ,
                                  Value ivK, int64_t r,
                                  std::string &reason) {
  // 3D variant of access/radius checks used by peeling.
  for (Block &block : region) {
    for (Operation &op : block) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;

      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        if (!checkBranchAccesses3D(forOp.getRegion(), ivI, ivJ, ivK, r, reason))
          return false;
        continue;
      }

      if (isa<scf::IfOp>(op)) {
        reason = "nested scf.if not supported";
        return false;
      }

      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        auto memrefTy = dyn_cast<MemRefType>(load.getMemref().getType());
        if (!memrefTy || memrefTy.getRank() != 3) {
          reason = "memref rank mismatch";
          return false;
        }
        if (load.getIndices().size() != 3) {
          reason = "memref indices mismatch";
          return false;
        }
        int64_t offI = 0;
        int64_t offJ = 0;
        int64_t offK = 0;
        if (!matchIndexOffset(load.getIndices()[0], ivI, offI) ||
            !matchIndexOffset(load.getIndices()[1], ivJ, offJ) ||
            !matchIndexOffset(load.getIndices()[2], ivK, offK)) {
          reason = "unsupported load index";
          return false;
        }
        if (std::llabs(offI) > r || std::llabs(offJ) > r ||
            std::llabs(offK) > r) {
          reason = "load offset exceeds radius";
          return false;
        }
        continue;
      }

      if (auto store = dyn_cast<memref::StoreOp>(op)) {
        auto memrefTy = dyn_cast<MemRefType>(store.getMemref().getType());
        if (!memrefTy || memrefTy.getRank() != 3) {
          reason = "memref rank mismatch";
          return false;
        }
        if (store.getIndices().size() != 3) {
          reason = "memref indices mismatch";
          return false;
        }
        int64_t offI = 0;
        int64_t offJ = 0;
        int64_t offK = 0;
        if (!matchIndexOffset(store.getIndices()[0], ivI, offI) ||
            !matchIndexOffset(store.getIndices()[1], ivJ, offJ) ||
            !matchIndexOffset(store.getIndices()[2], ivK, offK)) {
          reason = "unsupported store index";
          return false;
        }
        if (std::llabs(offI) > r || std::llabs(offJ) > r ||
            std::llabs(offK) > r) {
          reason = "store offset exceeds radius";
          return false;
        }
        continue;
      }

      if (isa<arith::ConstantOp>(op) || isa<arith::AddIOp>(op) ||
          isa<arith::SubIOp>(op) || isa<arith::MulIOp>(op) ||
          isa<arith::DivSIOp>(op) || isa<arith::DivUIOp>(op) ||
          isa<arith::IndexCastOp>(op) || isa<arith::CmpIOp>(op) ||
          isa<arith::SelectOp>(op) || isa<arith::AndIOp>(op) ||
          isa<arith::OrIOp>(op) || isa<arith::XOrIOp>(op)) {
        continue;
      }

      if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
        if (!iface.hasNoEffect()) {
          reason = "side-effect op";
          return false;
        }
        continue;
      }

      reason = "unsupported op in branch";
      return false;
    }
  }

  return true;
}

static bool checkBranchAccesses1D(Region &region, Value ivI, int64_t r,
                                  std::string &reason) {
  // 1D variant of access/radius checks used by peeling.
  for (Block &block : region) {
    for (Operation &op : block) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;

      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        if (!checkBranchAccesses1D(forOp.getRegion(), ivI, r, reason))
          return false;
        continue;
      }

      if (isa<scf::IfOp>(op)) {
        reason = "nested scf.if not supported";
        return false;
      }

      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        auto memrefTy = dyn_cast<MemRefType>(load.getMemref().getType());
        if (!memrefTy || memrefTy.getRank() != 1) {
          reason = "memref rank mismatch";
          return false;
        }
        if (load.getIndices().size() != 1) {
          reason = "memref indices mismatch";
          return false;
        }
        int64_t offI = 0;
        if (!matchIndexOffset(load.getIndices()[0], ivI, offI)) {
          reason = "unsupported load index";
          return false;
        }
        if (std::llabs(offI) > r) {
          reason = "load offset exceeds radius";
          return false;
        }
        continue;
      }

      if (auto store = dyn_cast<memref::StoreOp>(op)) {
        auto memrefTy = dyn_cast<MemRefType>(store.getMemref().getType());
        if (!memrefTy || memrefTy.getRank() != 1) {
          reason = "memref rank mismatch";
          return false;
        }
        if (store.getIndices().size() != 1) {
          reason = "memref indices mismatch";
          return false;
        }
        int64_t offI = 0;
        if (!matchIndexOffset(store.getIndices()[0], ivI, offI)) {
          reason = "unsupported store index";
          return false;
        }
        if (std::llabs(offI) > r) {
          reason = "store offset exceeds radius";
          return false;
        }
        continue;
      }

      if (isa<arith::ConstantOp>(op) || isa<arith::AddIOp>(op) ||
          isa<arith::SubIOp>(op) || isa<arith::MulIOp>(op) ||
          isa<arith::DivSIOp>(op) || isa<arith::DivUIOp>(op) ||
          isa<arith::IndexCastOp>(op) || isa<arith::CmpIOp>(op) ||
          isa<arith::SelectOp>(op) || isa<arith::AndIOp>(op) ||
          isa<arith::OrIOp>(op) || isa<arith::XOrIOp>(op)) {
        continue;
      }

      if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
        if (!iface.hasNoEffect()) {
          reason = "side-effect op";
          return false;
        }
        continue;
      }

      reason = "unsupported op in branch";
      return false;
    }
  }

  return true;
}

static arith::CmpIPredicate swapPredicate(arith::CmpIPredicate pred) {
  switch (pred) {
  case arith::CmpIPredicate::eq:
  case arith::CmpIPredicate::ne:
    return pred;
  case arith::CmpIPredicate::slt:
    return arith::CmpIPredicate::sgt;
  case arith::CmpIPredicate::sle:
    return arith::CmpIPredicate::sge;
  case arith::CmpIPredicate::sgt:
    return arith::CmpIPredicate::slt;
  case arith::CmpIPredicate::sge:
    return arith::CmpIPredicate::sle;
  case arith::CmpIPredicate::ult:
    return arith::CmpIPredicate::ugt;
  case arith::CmpIPredicate::ule:
    return arith::CmpIPredicate::uge;
  case arith::CmpIPredicate::ugt:
    return arith::CmpIPredicate::ult;
  case arith::CmpIPredicate::uge:
    return arith::CmpIPredicate::ule;
  }
  return pred;
}

static bool parseBoundaryCmp(arith::CmpIOp cmp, Value ivI, Value ivJ,
                             int64_t lbI, int64_t ubI, int64_t lbJ,
                             int64_t ubJ, int64_t &rOut, int &dimOut,
                             std::string &reason) {
  // Decode boundary predicate into (dim, radius) if it is a supported form.
  Value lhs = cmp.getLhs();
  Value rhs = cmp.getRhs();
  arith::CmpIPredicate pred = cmp.getPredicate();
  int dim = -1;
  int64_t cst = 0;

  if (lhs == ivI || lhs == ivJ) {
    if (!evalIndexExpr(rhs, cst)) {
      reason = "cmp rhs not constant";
      return false;
    }
    dim = (lhs == ivI) ? 0 : 1;
  } else if (rhs == ivI || rhs == ivJ) {
    if (!evalIndexExpr(lhs, cst)) {
      reason = "cmp lhs not constant";
      return false;
    }
    dim = (rhs == ivI) ? 0 : 1;
    pred = swapPredicate(pred);
  } else {
    reason = "cmp not against iv";
    return false;
  }

  if (pred == arith::CmpIPredicate::ne) {
    reason = "ne predicate not supported";
    return false;
  }

  int64_t lb = (dim == 0) ? lbI : lbJ;
  int64_t ub = (dim == 0) ? ubI : ubJ;
  int64_t r = -1;

  switch (pred) {
  case arith::CmpIPredicate::eq:
    if (cst == lb || cst == ub - 1) {
      r = 1;
    } else {
      reason = "eq not at boundary";
      return false;
    }
    break;
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    r = cst - lb;
    if (r < 1) {
      reason = "slt radius < 1";
      return false;
    }
    break;
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    r = cst - lb + 1;
    if (r < 1) {
      reason = "sle radius < 1";
      return false;
    }
    break;
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::uge:
    r = ub - cst;
    if (r < 1) {
      reason = "sge radius < 1";
      return false;
    }
    break;
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
    r = ub - cst - 1;
    if (r < 1) {
      reason = "sgt radius < 1";
      return false;
    }
    break;
  default:
    reason = "unsupported predicate";
    return false;
  }

  rOut = r;
  dimOut = dim;
  return true;
}

static bool parseBoundaryCmp3D(arith::CmpIOp cmp, Value ivI, Value ivJ,
                               Value ivK, int64_t lbI, int64_t ubI,
                               int64_t lbJ, int64_t ubJ, int64_t lbK,
                               int64_t ubK, int64_t &rOut, int &dimOut,
                               std::string &reason) {
  // 3D variant of boundary predicate parsing.
  Value lhs = cmp.getLhs();
  Value rhs = cmp.getRhs();
  arith::CmpIPredicate pred = cmp.getPredicate();
  int dim = -1;
  int64_t cst = 0;
  int64_t lb = 0;
  int64_t ub = 0;

  if (lhs == ivI) {
    if (!evalIndexExpr(rhs, cst)) {
      reason = "cmp rhs not constant";
      return false;
    }
    dim = 0;
    lb = lbI;
    ub = ubI;
  } else if (lhs == ivJ) {
    if (!evalIndexExpr(rhs, cst)) {
      reason = "cmp rhs not constant";
      return false;
    }
    dim = 1;
    lb = lbJ;
    ub = ubJ;
  } else if (lhs == ivK) {
    if (!evalIndexExpr(rhs, cst)) {
      reason = "cmp rhs not constant";
      return false;
    }
    dim = 2;
    lb = lbK;
    ub = ubK;
  } else if (rhs == ivI) {
    if (!evalIndexExpr(lhs, cst)) {
      reason = "cmp lhs not constant";
      return false;
    }
    dim = 0;
    lb = lbI;
    ub = ubI;
    pred = swapPredicate(pred);
  } else if (rhs == ivJ) {
    if (!evalIndexExpr(lhs, cst)) {
      reason = "cmp lhs not constant";
      return false;
    }
    dim = 1;
    lb = lbJ;
    ub = ubJ;
    pred = swapPredicate(pred);
  } else if (rhs == ivK) {
    if (!evalIndexExpr(lhs, cst)) {
      reason = "cmp lhs not constant";
      return false;
    }
    dim = 2;
    lb = lbK;
    ub = ubK;
    pred = swapPredicate(pred);
  } else {
    reason = "cmp not against iv";
    return false;
  }

  if (pred == arith::CmpIPredicate::ne) {
    reason = "ne predicate not supported";
    return false;
  }

  int64_t r = -1;
  switch (pred) {
  case arith::CmpIPredicate::eq:
    if (cst == lb || cst == ub - 1) {
      r = 1;
    } else {
      reason = "eq not at boundary";
      return false;
    }
    break;
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    r = cst - lb;
    if (r < 1) {
      reason = "slt radius < 1";
      return false;
    }
    break;
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    r = cst - lb + 1;
    if (r < 1) {
      reason = "sle radius < 1";
      return false;
    }
    break;
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::uge:
    r = ub - cst;
    if (r < 1) {
      reason = "sge radius < 1";
      return false;
    }
    break;
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
    r = ub - cst - 1;
    if (r < 1) {
      reason = "sgt radius < 1";
      return false;
    }
    break;
  default:
    reason = "unsupported predicate";
    return false;
  }

  rOut = r;
  dimOut = dim;
  return true;
}

static bool collectBoundaryConds(Value cond, Value ivI, Value ivJ, int64_t lbI,
                                 int64_t ubI, int64_t lbJ, int64_t ubJ,
                                 int64_t &radius, bool &seenI, bool &seenJ,
                                 std::string &reason) {
  // Accept only OR of boundary comparisons; reject AND to keep it simple.
  Operation *def = cond.getDefiningOp();
  if (!def) {
    reason = "condition not defined by op";
    return false;
  }

  if (auto orOp = dyn_cast<arith::OrIOp>(def)) {
    if (!collectBoundaryConds(orOp.getLhs(), ivI, ivJ, lbI, ubI, lbJ, ubJ,
                              radius, seenI, seenJ, reason))
      return false;
    if (!collectBoundaryConds(orOp.getRhs(), ivI, ivJ, lbI, ubI, lbJ, ubJ,
                              radius, seenI, seenJ, reason))
      return false;
    return true;
  }

  if (isa<arith::AndIOp>(def)) {
    reason = "and condition not supported";
    return false;
  }

  if (auto cmp = dyn_cast<arith::CmpIOp>(def)) {
    int64_t r = -1;
    int dim = -1;
    if (!parseBoundaryCmp(cmp, ivI, ivJ, lbI, ubI, lbJ, ubJ, r, dim, reason))
      return false;
    if (radius < 0) {
      radius = r;
    } else if (radius != r) {
      reason = "radius mismatch";
      return false;
    }
    if (dim == 0)
      seenI = true;
    if (dim == 1)
      seenJ = true;
    return true;
  }

  reason = "unsupported condition op";
  return false;
}

static bool parseCornerCmp2D(arith::CmpIOp cmp, Value ivI, Value ivJ,
                             int64_t lbI, int64_t ubI, int64_t lbJ,
                             int64_t ubJ, int &dimOut, bool &isMin,
                             std::string &reason) {
  // Decode "iv == lb|ub-1" for a single dimension.
  Value lhs = cmp.getLhs();
  Value rhs = cmp.getRhs();
  arith::CmpIPredicate pred = cmp.getPredicate();
  int64_t cst = 0;

  if (lhs == ivI) {
    if (!evalIndexExpr(rhs, cst)) {
      reason = "corner cmp rhs not constant";
      return false;
    }
    dimOut = 0;
  } else if (lhs == ivJ) {
    if (!evalIndexExpr(rhs, cst)) {
      reason = "corner cmp rhs not constant";
      return false;
    }
    dimOut = 1;
  } else if (rhs == ivI) {
    if (!evalIndexExpr(lhs, cst)) {
      reason = "corner cmp lhs not constant";
      return false;
    }
    dimOut = 0;
    pred = swapPredicate(pred);
  } else if (rhs == ivJ) {
    if (!evalIndexExpr(lhs, cst)) {
      reason = "corner cmp lhs not constant";
      return false;
    }
    dimOut = 1;
    pred = swapPredicate(pred);
  } else {
    reason = "corner cmp not against iv";
    return false;
  }

  if (pred != arith::CmpIPredicate::eq) {
    reason = "corner cmp not eq";
    return false;
  }

  int64_t lb = (dimOut == 0) ? lbI : lbJ;
  int64_t ub = (dimOut == 0) ? ubI : ubJ;
  if (cst == lb) {
    isMin = true;
    return true;
  }
  if (cst == ub - 1) {
    isMin = false;
    return true;
  }
  reason = "corner cmp not at boundary";
  return false;
}

static bool collectCornerConds2D(Value cond, Value ivI, Value ivJ, int64_t lbI,
                                 int64_t ubI, int64_t lbJ, int64_t ubJ,
                                 bool &seenI, bool &iAtMin, bool &seenJ,
                                 bool &jAtMin, std::string &reason) {
  Operation *def = cond.getDefiningOp();
  if (!def) {
    reason = "corner condition not defined by op";
    return false;
  }

  if (auto andOp = dyn_cast<arith::AndIOp>(def)) {
    if (!collectCornerConds2D(andOp.getLhs(), ivI, ivJ, lbI, ubI, lbJ, ubJ,
                              seenI, iAtMin, seenJ, jAtMin, reason))
      return false;
    if (!collectCornerConds2D(andOp.getRhs(), ivI, ivJ, lbI, ubI, lbJ, ubJ,
                              seenI, iAtMin, seenJ, jAtMin, reason))
      return false;
    return true;
  }

  if (isa<arith::OrIOp>(def)) {
    reason = "corner condition with or not supported";
    return false;
  }

  if (auto cmp = dyn_cast<arith::CmpIOp>(def)) {
    int dim = -1;
    bool isMin = false;
    if (!parseCornerCmp2D(cmp, ivI, ivJ, lbI, ubI, lbJ, ubJ, dim, isMin,
                          reason))
      return false;
    if (dim == 0) {
      if (seenI && iAtMin != isMin) {
        reason = "corner dim0 conflict";
        return false;
      }
      seenI = true;
      iAtMin = isMin;
    } else if (dim == 1) {
      if (seenJ && jAtMin != isMin) {
        reason = "corner dim1 conflict";
        return false;
      }
      seenJ = true;
      jAtMin = isMin;
    }
    return true;
  }

  reason = "unsupported corner condition op";
  return false;
}

static bool collectBoundaryConds3D(Value cond, Value ivI, Value ivJ,
                                   Value ivK, int64_t lbI, int64_t ubI,
                                   int64_t lbJ, int64_t ubJ, int64_t lbK,
                                   int64_t ubK, int64_t &radius, bool &seenI,
                                   bool &seenJ, bool &seenK,
                                   std::string &reason) {
  // 3D variant of boundary condition collection.
  Operation *def = cond.getDefiningOp();
  if (!def) {
    reason = "condition not defined by op";
    return false;
  }

  if (auto orOp = dyn_cast<arith::OrIOp>(def)) {
    if (!collectBoundaryConds3D(orOp.getLhs(), ivI, ivJ, ivK, lbI, ubI, lbJ,
                                ubJ, lbK, ubK, radius, seenI, seenJ, seenK,
                                reason))
      return false;
    if (!collectBoundaryConds3D(orOp.getRhs(), ivI, ivJ, ivK, lbI, ubI, lbJ,
                                ubJ, lbK, ubK, radius, seenI, seenJ, seenK,
                                reason))
      return false;
    return true;
  }

  if (isa<arith::AndIOp>(def)) {
    reason = "and condition not supported";
    return false;
  }

  if (auto cmp = dyn_cast<arith::CmpIOp>(def)) {
    int64_t r = -1;
    int dim = -1;
    if (!parseBoundaryCmp3D(cmp, ivI, ivJ, ivK, lbI, ubI, lbJ, ubJ, lbK, ubK,
                            r, dim, reason))
      return false;
    if (radius < 0) {
      radius = r;
    } else if (radius != r) {
      reason = "radius mismatch";
      return false;
    }
    if (dim == 0)
      seenI = true;
    if (dim == 1)
      seenJ = true;
    if (dim == 2)
      seenK = true;
    return true;
  }

  reason = "unsupported condition op";
  return false;
}

static bool parseCornerCmp3D(arith::CmpIOp cmp, Value ivI, Value ivJ,
                             Value ivK, int64_t lbI, int64_t ubI, int64_t lbJ,
                             int64_t ubJ, int64_t lbK, int64_t ubK,
                             int &dimOut, bool &isMin, std::string &reason) {
  Value lhs = cmp.getLhs();
  Value rhs = cmp.getRhs();
  arith::CmpIPredicate pred = cmp.getPredicate();
  int64_t cst = 0;

  if (lhs == ivI) {
    if (!evalIndexExpr(rhs, cst)) {
      reason = "corner cmp rhs not constant";
      return false;
    }
    dimOut = 0;
  } else if (lhs == ivJ) {
    if (!evalIndexExpr(rhs, cst)) {
      reason = "corner cmp rhs not constant";
      return false;
    }
    dimOut = 1;
  } else if (lhs == ivK) {
    if (!evalIndexExpr(rhs, cst)) {
      reason = "corner cmp rhs not constant";
      return false;
    }
    dimOut = 2;
  } else if (rhs == ivI) {
    if (!evalIndexExpr(lhs, cst)) {
      reason = "corner cmp lhs not constant";
      return false;
    }
    dimOut = 0;
    pred = swapPredicate(pred);
  } else if (rhs == ivJ) {
    if (!evalIndexExpr(lhs, cst)) {
      reason = "corner cmp lhs not constant";
      return false;
    }
    dimOut = 1;
    pred = swapPredicate(pred);
  } else if (rhs == ivK) {
    if (!evalIndexExpr(lhs, cst)) {
      reason = "corner cmp lhs not constant";
      return false;
    }
    dimOut = 2;
    pred = swapPredicate(pred);
  } else {
    reason = "corner cmp not against iv";
    return false;
  }

  if (pred != arith::CmpIPredicate::eq) {
    reason = "corner cmp not eq";
    return false;
  }

  int64_t lb = 0;
  int64_t ub = 0;
  if (dimOut == 0) {
    lb = lbI;
    ub = ubI;
  } else if (dimOut == 1) {
    lb = lbJ;
    ub = ubJ;
  } else {
    lb = lbK;
    ub = ubK;
  }

  if (cst == lb) {
    isMin = true;
    return true;
  }
  if (cst == ub - 1) {
    isMin = false;
    return true;
  }
  reason = "corner cmp not at boundary";
  return false;
}

static bool collectCornerConds3D(Value cond, Value ivI, Value ivJ, Value ivK,
                                 int64_t lbI, int64_t ubI, int64_t lbJ,
                                 int64_t ubJ, int64_t lbK, int64_t ubK,
                                 bool &seenI, bool &iAtMin, bool &seenJ,
                                 bool &jAtMin, bool &seenK, bool &kAtMin,
                                 std::string &reason) {
  Operation *def = cond.getDefiningOp();
  if (!def) {
    reason = "corner condition not defined by op";
    return false;
  }

  if (auto andOp = dyn_cast<arith::AndIOp>(def)) {
    if (!collectCornerConds3D(andOp.getLhs(), ivI, ivJ, ivK, lbI, ubI, lbJ,
                              ubJ, lbK, ubK, seenI, iAtMin, seenJ, jAtMin,
                              seenK, kAtMin, reason))
      return false;
    if (!collectCornerConds3D(andOp.getRhs(), ivI, ivJ, ivK, lbI, ubI, lbJ,
                              ubJ, lbK, ubK, seenI, iAtMin, seenJ, jAtMin,
                              seenK, kAtMin, reason))
      return false;
    return true;
  }

  if (isa<arith::OrIOp>(def)) {
    reason = "corner condition with or not supported";
    return false;
  }

  if (auto cmp = dyn_cast<arith::CmpIOp>(def)) {
    int dim = -1;
    bool isMin = false;
    if (!parseCornerCmp3D(cmp, ivI, ivJ, ivK, lbI, ubI, lbJ, ubJ, lbK, ubK,
                          dim, isMin, reason))
      return false;
    if (dim == 0) {
      if (seenI && iAtMin != isMin) {
        reason = "corner dim0 conflict";
        return false;
      }
      seenI = true;
      iAtMin = isMin;
    } else if (dim == 1) {
      if (seenJ && jAtMin != isMin) {
        reason = "corner dim1 conflict";
        return false;
      }
      seenJ = true;
      jAtMin = isMin;
    } else if (dim == 2) {
      if (seenK && kAtMin != isMin) {
        reason = "corner dim2 conflict";
        return false;
      }
      seenK = true;
      kAtMin = isMin;
    }
    return true;
  }

  reason = "unsupported corner condition op";
  return false;
}

static bool parseBoundaryCmp1D(arith::CmpIOp cmp, Value ivI, int64_t lbI,
                               int64_t ubI, int64_t &rOut,
                               std::string &reason) {
  Value lhs = cmp.getLhs();
  Value rhs = cmp.getRhs();
  arith::CmpIPredicate pred = cmp.getPredicate();
  int64_t cst = 0;

  if (lhs == ivI) {
    if (!evalIndexExpr(rhs, cst)) {
      reason = "cmp rhs not constant";
      return false;
    }
  } else if (rhs == ivI) {
    if (!evalIndexExpr(lhs, cst)) {
      reason = "cmp lhs not constant";
      return false;
    }
    pred = swapPredicate(pred);
  } else {
    reason = "cmp not against iv";
    return false;
  }

  if (pred == arith::CmpIPredicate::ne) {
    reason = "ne predicate not supported";
    return false;
  }

  int64_t r = -1;
  switch (pred) {
  case arith::CmpIPredicate::eq:
    if (cst == lbI || cst == ubI - 1) {
      r = 1;
    } else {
      reason = "eq not at boundary";
      return false;
    }
    break;
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    r = cst - lbI;
    if (r < 1) {
      reason = "slt radius < 1";
      return false;
    }
    break;
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    r = cst - lbI + 1;
    if (r < 1) {
      reason = "sle radius < 1";
      return false;
    }
    break;
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::uge:
    r = ubI - cst;
    if (r < 1) {
      reason = "sge radius < 1";
      return false;
    }
    break;
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
    r = ubI - cst - 1;
    if (r < 1) {
      reason = "sgt radius < 1";
      return false;
    }
    break;
  default:
    reason = "unsupported predicate";
    return false;
  }

  rOut = r;
  return true;
}

static bool collectBoundaryConds1D(Value cond, Value ivI, int64_t lbI,
                                   int64_t ubI, int64_t &radius,
                                   std::string &reason) {
  // 1D variant: OR of boundary comparisons.
  Operation *def = cond.getDefiningOp();
  if (!def) {
    reason = "condition not defined by op";
    return false;
  }

  if (auto orOp = dyn_cast<arith::OrIOp>(def)) {
    if (!collectBoundaryConds1D(orOp.getLhs(), ivI, lbI, ubI, radius, reason))
      return false;
    if (!collectBoundaryConds1D(orOp.getRhs(), ivI, lbI, ubI, radius, reason))
      return false;
    return true;
  }

  if (isa<arith::AndIOp>(def)) {
    reason = "and condition not supported";
    return false;
  }

  if (auto cmp = dyn_cast<arith::CmpIOp>(def)) {
    int64_t r = -1;
    if (!parseBoundaryCmp1D(cmp, ivI, lbI, ubI, r, reason))
      return false;
    if (radius < 0) {
      radius = r;
    } else if (radius != r) {
      reason = "radius mismatch";
      return false;
    }
    return true;
  }

  reason = "unsupported condition op";
  return false;
}

[[maybe_unused]] static bool parseCornerCmp1D(arith::CmpIOp cmp, Value ivI,
                                              int64_t lbI, int64_t ubI,
                                              bool &isMin,
                                              std::string &reason) {
  Value lhs = cmp.getLhs();
  Value rhs = cmp.getRhs();
  arith::CmpIPredicate pred = cmp.getPredicate();
  int64_t cst = 0;

  if (lhs == ivI) {
    if (!evalIndexExpr(rhs, cst)) {
      reason = "corner cmp rhs not constant";
      return false;
    }
  } else if (rhs == ivI) {
    if (!evalIndexExpr(lhs, cst)) {
      reason = "corner cmp lhs not constant";
      return false;
    }
    pred = swapPredicate(pred);
  } else {
    reason = "corner cmp not against iv";
    return false;
  }

  if (pred != arith::CmpIPredicate::eq) {
    reason = "corner cmp not eq";
    return false;
  }
  if (cst == lbI) {
    isMin = true;
    return true;
  }
  if (cst == ubI - 1) {
    isMin = false;
    return true;
  }
  reason = "corner cmp not at boundary";
  return false;
}

[[maybe_unused]] static bool collectCornerConds1D(Value cond, Value ivI,
                                                  int64_t lbI, int64_t ubI,
                                                  bool &seenI, bool &iAtMin,
                                                  std::string &reason) {
  Operation *def = cond.getDefiningOp();
  if (!def) {
    reason = "corner condition not defined by op";
    return false;
  }
  if (auto andOp = dyn_cast<arith::AndIOp>(def)) {
    if (!collectCornerConds1D(andOp.getLhs(), ivI, lbI, ubI, seenI, iAtMin,
                              reason))
      return false;
    if (!collectCornerConds1D(andOp.getRhs(), ivI, lbI, ubI, seenI, iAtMin,
                              reason))
      return false;
    return true;
  }
  if (isa<arith::OrIOp>(def)) {
    reason = "corner condition with or not supported";
    return false;
  }
  if (auto cmp = dyn_cast<arith::CmpIOp>(def)) {
    bool isMin = false;
    if (!parseCornerCmp1D(cmp, ivI, lbI, ubI, isMin, reason))
      return false;
    if (seenI && iAtMin != isMin) {
      reason = "corner dim0 conflict";
      return false;
    }
    seenI = true;
    iAtMin = isMin;
    return true;
  }
  reason = "unsupported corner condition op";
  return false;
}

// Forward declaration for region cloning helper.
static void cloneRegionOps(Region &src, OpBuilder &b, IRMapping &mapping);

static scf::ForOp buildLoopWithBranch1D(OpBuilder &b, Location loc, Value lbI,
                                        Value ubI, Value step, Region &branch,
                                        Value oldI) {
  // Clone a 1D loop and replace iv with the new loop iv.
  auto loop = b.create<scf::ForOp>(loc, lbI, ubI, step);
  IRMapping mapping;
  mapping.map(oldI, loop.getInductionVar());
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(loop.getBody());
  cloneRegionOps(branch, bodyBuilder, mapping);
  return loop;
}

// Clones a region body into the current builder using a prebuilt value map.
static void cloneRegionOps(Region &src, OpBuilder &b, IRMapping &mapping) {
  Block &block = src.front();
  for (Operation &op : block.without_terminator()) {
    b.clone(op, mapping);
  }
}

// Whitelist for predicate construction: keep conditions simple and local.
static bool isCondDefOp(Operation *op) {
  return isa<arith::CmpIOp, arith::OrIOp, arith::AndIOp, arith::XOrIOp,
             arith::AddIOp, arith::SubIOp, arith::IndexCastOp,
             arith::ConstantOp, arith::ConstantIndexOp>(op);
}

// Whitelist for bound expressions: integer add/sub/cast/const only.
static bool isBoundDefOp(Operation *op) {
  return isa<arith::AddIOp, arith::SubIOp, arith::IndexCastOp,
             arith::ConstantOp, arith::ConstantIndexOp>(op);
}

// Collects the def-use slice of a bound expression within a single block.
static bool collectValueOps(Value v, Block &block,
                            llvm::DenseSet<Operation *> &ops,
                            std::string &reason) {
  Operation *def = v.getDefiningOp();
  if (!def || def->getBlock() != &block) {
    return true;
  }
  if (!isBoundDefOp(def)) {
    reason = "unsupported bound op";
    return false;
  }
  if (!ops.insert(def).second) {
    return true;
  }
  for (Value operand : def->getOperands()) {
    Operation *opDef = operand.getDefiningOp();
    if (!opDef || opDef->getBlock() != &block) {
      reason = "bound uses value outside block";
      return false;
    }
    if (!collectValueOps(operand, block, ops, reason)) {
      return false;
    }
  }
  return true;
}

// Gathers ops that define loop bounds/step; rejects non-local math.
static bool collectLoopBoundOps(scf::ForOp loop, Block &block,
                                llvm::DenseSet<Operation *> &ops,
                                std::string &reason) {
  if (!collectValueOps(loop.getLowerBound(), block, ops, reason))
    return false;
  if (!collectValueOps(loop.getUpperBound(), block, ops, reason))
    return false;
  if (!collectValueOps(loop.getStep(), block, ops, reason))
    return false;
  return true;
}

// Collects the predicate tree; used to prove the if is the only side op.
static bool collectCondOps(Value cond, Block &block,
                           llvm::DenseSet<Operation *> &condOps,
                           std::string &reason) {
  Operation *def = cond.getDefiningOp();
  if (!def) {
    reason = "condition not defined by op";
    return false;
  }
  if (def->getBlock() != &block) {
    if (!isCondDefOp(def)) {
      reason = "unsupported condition op";
      return false;
    }
    return true;
  }
  if (!isCondDefOp(def)) {
    reason = "unsupported condition op";
    return false;
  }
  if (!condOps.insert(def).second) {
    return true;
  }
  for (Value operand : def->getOperands()) {
    Operation *opDef = operand.getDefiningOp();
    if (!opDef || opDef->getBlock() != &block) {
      continue;
    }
    if (!collectCondOps(operand, block, condOps, reason)) {
      return false;
    }
  }
  return true;
}

static bool checkCornerBranchOps(Region &region, std::string &reason) {
  // Corner branches must be straight-line: no nested if/for.
  for (Block &block : region) {
    for (Operation &op : block) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;
      if (isa<scf::ForOp>(op)) {
        reason = "corner branch contains nested loop";
        return false;
      }
      if (isa<scf::IfOp>(op)) {
        reason = "corner branch contains nested if";
        return false;
      }
      if (isa<memref::LoadOp>(op) || isa<memref::StoreOp>(op) ||
          isa<memref::CastOp>(op) || isa<arith::ConstantOp>(op) ||
          isa<arith::ConstantIndexOp>(op) || isa<arith::AddIOp>(op) ||
          isa<arith::SubIOp>(op) || isa<arith::MulIOp>(op) ||
          isa<arith::DivSIOp>(op) || isa<arith::DivUIOp>(op) ||
          isa<arith::IndexCastOp>(op) || isa<arith::CmpIOp>(op) ||
          isa<arith::SelectOp>(op) || isa<arith::AndIOp>(op) ||
          isa<arith::OrIOp>(op) || isa<arith::XOrIOp>(op)) {
        continue;
      }
      if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
        if (!iface.hasNoEffect()) {
          reason = "corner branch side-effect op";
          return false;
        }
        continue;
      }
      reason = "corner branch unsupported op";
      return false;
    }
  }
  return true;
}

static void tagCornerStores(Operation *op) {
  auto *ctx = op->getContext();
  auto attr = UnitAttr::get(ctx);
  op->walk([&](memref::StoreOp store) { store->setAttr("neptune.corner", attr); });
}

static bool findNestedBoundaryIf(scf::IfOp outerIf, scf::IfOp &nested,
                                 std::string &reason) {
  Block &elseBlock = outerIf.getElseRegion().front();
  for (Operation &op : elseBlock.without_terminator()) {
    if (auto candidate = dyn_cast<scf::IfOp>(op)) {
      if (nested) {
        reason = "multiple ifs in corner else";
        return false;
      }
      nested = candidate;
    }
  }
  if (!nested) {
    reason = "missing boundary if in corner else";
    return false;
  }
  if (nested.getNumResults() != 0 || !nested.getThenRegion().hasOneBlock() ||
      !nested.getElseRegion().hasOneBlock()) {
    reason = "nested boundary if malformed";
    return false;
  }

  llvm::DenseSet<Operation *> condOps;
  if (!collectCondOps(nested.getCondition(), elseBlock, condOps, reason)) {
    return false;
  }
  for (Operation &op : elseBlock.without_terminator()) {
    if (&op == nested.getOperation())
      continue;
    if (!condOps.contains(&op)) {
      reason = "extra ops in corner else";
      return false;
    }
  }
  return true;
}

// Builds a fresh 2D loop nest and splices in a branch with remapped IVs.
static scf::ForOp buildLoopNestWithBranch(OpBuilder &b, Location loc,
                                          Value lbI, Value ubI, Value lbJ,
                                          Value ubJ, Value step,
                                          Region &branch, Value oldI,
                                          Value oldJ) {
  // Clone a 2D loop nest and remap ivs to the new loops.
  auto outer = b.create<scf::ForOp>(loc, lbI, ubI, step);
  OpBuilder innerBuilder = OpBuilder::atBlockBegin(outer.getBody());
  auto inner = innerBuilder.create<scf::ForOp>(loc, lbJ, ubJ, step);

  IRMapping mapping;
  mapping.map(oldI, outer.getInductionVar());
  mapping.map(oldJ, inner.getInductionVar());

  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(inner.getBody());
  cloneRegionOps(branch, bodyBuilder, mapping);

  return outer;
}

// Builds a fresh 3D loop nest and splices in a branch with remapped IVs.
static scf::ForOp buildLoopNestWithBranch3D(OpBuilder &b, Location loc,
                                            Value lbI, Value ubI, Value lbJ,
                                            Value ubJ, Value lbK, Value ubK,
                                            Value step, Region &branch,
                                            Value oldI, Value oldJ,
                                            Value oldK) {
  // Clone a 3D loop nest and remap ivs to the new loops.
  auto outer = b.create<scf::ForOp>(loc, lbI, ubI, step);
  OpBuilder middleBuilder = OpBuilder::atBlockBegin(outer.getBody());
  auto middle = middleBuilder.create<scf::ForOp>(loc, lbJ, ubJ, step);
  OpBuilder innerBuilder = OpBuilder::atBlockBegin(middle.getBody());
  auto inner = innerBuilder.create<scf::ForOp>(loc, lbK, ubK, step);

  IRMapping mapping;
  mapping.map(oldI, outer.getInductionVar());
  mapping.map(oldJ, middle.getInductionVar());
  mapping.map(oldK, inner.getInductionVar());

  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(inner.getBody());
  cloneRegionOps(branch, bodyBuilder, mapping);

  return outer;
}

// Peels a 3D boundary-guarded loop into six faces plus a clean interior.
static bool tryPeel3D(func::FuncOp func, SummaryInfo &summary) {
  for (auto outer : func.getOps<scf::ForOp>()) {
    if (!outer.getResults().empty())
      continue;

    int64_t lbI = 0;
    int64_t ubI = 0;
    int64_t stepI = 0;
    if (!matchConstantIndex(outer.getLowerBound(), lbI))
      continue;
    if (!matchConstantIndex(outer.getUpperBound(), ubI))
      continue;
    if (!matchConstantIndex(outer.getStep(), stepI) || stepI != 1)
      continue;

    scf::ForOp middle;
    for (Operation &op : outer.getBody()->without_terminator()) {
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        if (middle) {
          middle = nullptr;
          break;
        }
        middle = loop;
        continue;
      }
    }
    if (!middle)
      continue;
    if (!middle.getResults().empty())
      continue;

    int64_t lbJ = 0;
    int64_t ubJ = 0;
    int64_t stepJ = 0;
    if (!matchConstantIndex(middle.getLowerBound(), lbJ))
      continue;
    if (!matchConstantIndex(middle.getUpperBound(), ubJ))
      continue;
    if (!matchConstantIndex(middle.getStep(), stepJ) || stepJ != 1)
      continue;

    scf::ForOp inner;
    for (Operation &op : middle.getBody()->without_terminator()) {
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        if (inner) {
          inner = nullptr;
          break;
        }
        inner = loop;
        continue;
      }
    }
    if (!inner)
      continue;
    if (!inner.getResults().empty())
      continue;

    int64_t lbK = 0;
    int64_t ubK = 0;
    int64_t stepK = 0;
    if (!matchConstantIndex(inner.getLowerBound(), lbK))
      continue;
    if (!matchConstantIndex(inner.getUpperBound(), ubK))
      continue;
    if (!matchConstantIndex(inner.getStep(), stepK) || stepK != 1)
      continue;

    scf::IfOp ifOp;
    for (Operation &op : inner.getBody()->without_terminator()) {
      if (auto candidate = dyn_cast<scf::IfOp>(op)) {
        if (ifOp) {
          ifOp = nullptr;
          break;
        }
        ifOp = candidate;
      }
    }
    if (!ifOp)
      continue;
    if (ifOp.getNumResults() != 0)
      continue;
    if (!ifOp.getElseRegion().hasOneBlock())
      continue;
    if (!ifOp.getThenRegion().hasOneBlock())
      continue;

    std::string boundReason;
    llvm::DenseSet<Operation *> boundOpsOuter;
    Block &outerBlock = *outer.getBody();
    if (!collectLoopBoundOps(middle, outerBlock, boundOpsOuter, boundReason)) {
      summary.reason = "skip peel: " + boundReason;
      continue;
    }
    bool extraOuterOps = false;
    for (Operation &op : outerBlock.without_terminator()) {
      if (&op == middle.getOperation())
        continue;
      if (!boundOpsOuter.contains(&op)) {
        extraOuterOps = true;
        break;
      }
    }
    if (extraOuterOps) {
      summary.reason = "skip peel: extra ops in outer loop";
      continue;
    }

    std::string middleReason;
    llvm::DenseSet<Operation *> boundOpsMiddle;
    Block &middleBlock = *middle.getBody();
    if (!collectLoopBoundOps(inner, middleBlock, boundOpsMiddle,
                             middleReason)) {
      summary.reason = "skip peel: " + middleReason;
      continue;
    }
    bool extraMiddleOps = false;
    for (Operation &op : middleBlock.without_terminator()) {
      if (&op == inner.getOperation())
        continue;
      if (!boundOpsMiddle.contains(&op)) {
        extraMiddleOps = true;
        break;
      }
    }
    if (extraMiddleOps) {
      summary.reason = "skip peel: extra ops in middle loop";
      continue;
    }

    std::string condReason;
    llvm::DenseSet<Operation *> condOps;
    Block &innerBlock = *inner.getBody();
    if (!collectCondOps(ifOp.getCondition(), innerBlock, condOps, condReason)) {
      summary.reason = "skip peel: " + condReason;
      continue;
    }
    bool extraInnerOps = false;
    for (Operation &op : innerBlock.without_terminator()) {
      if (&op == ifOp.getOperation())
        continue;
      if (!condOps.contains(&op)) {
        extraInnerOps = true;
        break;
      }
    }
    if (extraInnerOps) {
      summary.reason = "skip peel: extra ops in inner loop";
      continue;
    }

    scf::IfOp boundaryIf = ifOp;
    bool hasCorner = false;
    int64_t cornerI = 0;
    int64_t cornerJ = 0;
    int64_t cornerK = 0;
    {
      bool seenCornerI = false;
      bool seenCornerJ = false;
      bool seenCornerK = false;
      bool cornerIAtMin = false;
      bool cornerJAtMin = false;
      bool cornerKAtMin = false;
      std::string cornerReason;
      if (collectCornerConds3D(ifOp.getCondition(), outer.getInductionVar(),
                               middle.getInductionVar(),
                               inner.getInductionVar(), lbI, ubI, lbJ, ubJ,
                               lbK, ubK, seenCornerI, cornerIAtMin,
                               seenCornerJ, cornerJAtMin, seenCornerK,
                               cornerKAtMin, cornerReason) &&
          seenCornerI && seenCornerJ && seenCornerK) {
        scf::IfOp nested;
        if (!findNestedBoundaryIf(ifOp, nested, cornerReason)) {
          summary.reason = "skip peel: " + cornerReason;
          continue;
        }
        hasCorner = true;
        boundaryIf = nested;
        cornerI = cornerIAtMin ? lbI : (ubI - 1);
        cornerJ = cornerJAtMin ? lbJ : (ubJ - 1);
        cornerK = cornerKAtMin ? lbK : (ubK - 1);
      }
    }

    std::string reason;
    int64_t radius = -1;
    bool seenI = false;
    bool seenJ = false;
    bool seenK = false;
    if (!collectBoundaryConds3D(boundaryIf.getCondition(),
                                outer.getInductionVar(),
                                middle.getInductionVar(),
                                inner.getInductionVar(), lbI, ubI, lbJ, ubJ,
                                lbK, ubK, radius, seenI, seenJ, seenK,
                                reason)) {
      summary.reason = "skip peel: " + reason;
      continue;
    }
    if (radius < 1) {
      summary.reason = "skip peel: radius < 1";
      continue;
    }
    if (!seenI || !seenJ || !seenK) {
      summary.reason = "skip peel: boundary only on one dimension";
      continue;
    }

    int64_t lbIInt = lbI + radius;
    int64_t ubIInt = ubI - radius;
    int64_t lbJInt = lbJ + radius;
    int64_t ubJInt = ubJ - radius;
    int64_t lbKInt = lbK + radius;
    int64_t ubKInt = ubK - radius;
    if (lbIInt >= ubIInt || lbJInt >= ubJInt || lbKInt >= ubKInt) {
      summary.reason = "skip peel: interior empty";
      continue;
    }

    if (hasCorner) {
      if (!checkCornerBranchOps(ifOp.getThenRegion(), reason)) {
        summary.reason = "skip peel: " + reason;
        continue;
      }
      if (!checkBranchAccesses3D(ifOp.getThenRegion(),
                                 outer.getInductionVar(),
                                 middle.getInductionVar(),
                                 inner.getInductionVar(), radius, reason)) {
        summary.reason = "skip peel: " + reason;
        continue;
      }
    }

    if (!checkBranchAccesses3D(boundaryIf.getElseRegion(),
                               outer.getInductionVar(),
                               middle.getInductionVar(),
                               inner.getInductionVar(), radius, reason)) {
      summary.reason = "skip peel: " + reason;
      continue;
    }
    if (!checkBranchAccesses3D(boundaryIf.getThenRegion(),
                               outer.getInductionVar(),
                               middle.getInductionVar(),
                               inner.getInductionVar(), radius, reason)) {
      summary.reason = "skip peel: " + reason;
      continue;
    }

    OpBuilder b(outer);
    Location loc = outer.getLoc();
    Value step = b.create<arith::ConstantIndexOp>(loc, 1);

    Value lbIVal = b.create<arith::ConstantIndexOp>(loc, lbI);
    Value ubIVal = b.create<arith::ConstantIndexOp>(loc, ubI);
    Value lbJVal = b.create<arith::ConstantIndexOp>(loc, lbJ);
    Value ubJVal = b.create<arith::ConstantIndexOp>(loc, ubJ);
    Value lbKVal = b.create<arith::ConstantIndexOp>(loc, lbK);
    Value ubKVal = b.create<arith::ConstantIndexOp>(loc, ubK);

    Value lbIPlusR = b.create<arith::ConstantIndexOp>(loc, lbI + radius);
    Value ubIMinusR = b.create<arith::ConstantIndexOp>(loc, ubI - radius);
    Value lbJPlusR = b.create<arith::ConstantIndexOp>(loc, lbJ + radius);
    Value ubJMinusR = b.create<arith::ConstantIndexOp>(loc, ubJ - radius);
    Value lbKPlusR = b.create<arith::ConstantIndexOp>(loc, lbK + radius);
    Value ubKMinusR = b.create<arith::ConstantIndexOp>(loc, ubK - radius);

    if (hasCorner) {
      Value cornerILb = b.create<arith::ConstantIndexOp>(loc, cornerI);
      Value cornerIUb =
          b.create<arith::ConstantIndexOp>(loc, cornerI + 1);
      Value cornerJLb = b.create<arith::ConstantIndexOp>(loc, cornerJ);
      Value cornerJUb =
          b.create<arith::ConstantIndexOp>(loc, cornerJ + 1);
      Value cornerKLb = b.create<arith::ConstantIndexOp>(loc, cornerK);
      Value cornerKUb =
          b.create<arith::ConstantIndexOp>(loc, cornerK + 1);
      scf::ForOp cornerLoop =
          buildLoopNestWithBranch3D(
              b, loc, cornerILb, cornerIUb, cornerJLb, cornerJUb, cornerKLb,
              cornerKUb, step, ifOp.getThenRegion(),
              outer.getInductionVar(), middle.getInductionVar(),
              inner.getInductionVar());
      tagCornerStores(cornerLoop);
    }

    // Build K faces: full I/J with thin K slabs.
    buildLoopNestWithBranch3D(b, loc, lbIVal, ubIVal, lbJVal, ubJVal, lbKVal,
                              lbKPlusR, step, boundaryIf.getThenRegion(),
                              outer.getInductionVar(),
                              middle.getInductionVar(),
                              inner.getInductionVar());
    buildLoopNestWithBranch3D(b, loc, lbIVal, ubIVal, lbJVal, ubJVal, ubKMinusR,
                              ubKVal, step, boundaryIf.getThenRegion(),
                              outer.getInductionVar(),
                              middle.getInductionVar(),
                              inner.getInductionVar());
    // Build J faces: K interior, thin J slabs.
    buildLoopNestWithBranch3D(b, loc, lbIVal, ubIVal, lbJVal, lbJPlusR,
                              lbKPlusR, ubKMinusR, step,
                              boundaryIf.getThenRegion(),
                              outer.getInductionVar(),
                              middle.getInductionVar(),
                              inner.getInductionVar());
    buildLoopNestWithBranch3D(b, loc, lbIVal, ubIVal, ubJMinusR, ubJVal,
                              lbKPlusR, ubKMinusR, step,
                              boundaryIf.getThenRegion(),
                              outer.getInductionVar(),
                              middle.getInductionVar(),
                              inner.getInductionVar());
    // Build I faces: J/K interior, thin I slabs.
    buildLoopNestWithBranch3D(b, loc, lbIVal, lbIPlusR, lbJPlusR, ubJMinusR,
                              lbKPlusR, ubKMinusR, step,
                              boundaryIf.getThenRegion(),
                              outer.getInductionVar(),
                              middle.getInductionVar(),
                              inner.getInductionVar());
    buildLoopNestWithBranch3D(b, loc, ubIMinusR, ubIVal, lbJPlusR, ubJMinusR,
                              lbKPlusR, ubKMinusR, step,
                              boundaryIf.getThenRegion(),
                              outer.getInductionVar(),
                              middle.getInductionVar(),
                              inner.getInductionVar());
    // Build interior: no boundary guards, radius peeled away.
    buildLoopNestWithBranch3D(b, loc, lbIPlusR, ubIMinusR, lbJPlusR, ubJMinusR,
                              lbKPlusR, ubKMinusR, step,
                              boundaryIf.getElseRegion(),
                              outer.getInductionVar(),
                              middle.getInductionVar(),
                              inner.getInductionVar());

    outer.erase();

    summary.rank = 3;
    summary.lb = {lbI, lbJ, lbK};
    summary.ub = {ubI, ubJ, ubK};
    summary.r = radius;
    summary.peeled = true;
    summary.reason = "peel";

    logLine("interior: i=[" + std::to_string(lbIInt) + "," +
            std::to_string(ubIInt) + ") j=[" + std::to_string(lbJInt) + "," +
            std::to_string(ubJInt) + ") k=[" + std::to_string(lbKInt) + "," +
            std::to_string(ubKInt) + ")");
    return true;
  }

  return false;
}

// Peels a 2D boundary-guarded loop into strips plus a clean interior.
static bool tryPeel(func::FuncOp func, SummaryInfo &summary) {
  for (auto outer : func.getOps<scf::ForOp>()) {
    if (!outer.getResults().empty())
      continue;

    int64_t lbI = 0;
    int64_t ubI = 0;
    int64_t stepI = 0;
    if (!matchConstantIndex(outer.getLowerBound(), lbI))
      continue;
    if (!matchConstantIndex(outer.getUpperBound(), ubI))
      continue;
    if (!matchConstantIndex(outer.getStep(), stepI) || stepI != 1)
      continue;

    scf::ForOp inner;
    for (Operation &op : outer.getBody()->without_terminator()) {
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        if (inner) {
          inner = nullptr;
          break;
        }
        inner = loop;
        continue;
      }
    }
    if (!inner)
      continue;
    if (!inner.getResults().empty())
      continue;

    int64_t lbJ = 0;
    int64_t ubJ = 0;
    int64_t stepJ = 0;
    if (!matchConstantIndex(inner.getLowerBound(), lbJ))
      continue;
    if (!matchConstantIndex(inner.getUpperBound(), ubJ))
      continue;
    if (!matchConstantIndex(inner.getStep(), stepJ) || stepJ != 1)
      continue;

    scf::IfOp ifOp;
    for (Operation &op : inner.getBody()->without_terminator()) {
      if (auto candidate = dyn_cast<scf::IfOp>(op)) {
        if (ifOp) {
          ifOp = nullptr;
          break;
        }
        ifOp = candidate;
      }
    }
    if (!ifOp)
      continue;
    if (ifOp.getNumResults() != 0)
      continue;
    if (!ifOp.getElseRegion().hasOneBlock())
      continue;
    if (!ifOp.getThenRegion().hasOneBlock())
      continue;

    std::string boundReason;
    llvm::DenseSet<Operation *> boundOps;
    Block &outerBlock = *outer.getBody();
    if (!collectLoopBoundOps(inner, outerBlock, boundOps, boundReason)) {
      summary.reason = "skip peel: " + boundReason;
      continue;
    }
    bool extraOuterOps = false;
    for (Operation &op : outerBlock.without_terminator()) {
      if (&op == inner.getOperation())
        continue;
      if (!boundOps.contains(&op)) {
        extraOuterOps = true;
        break;
      }
    }
    if (extraOuterOps) {
      summary.reason = "skip peel: extra ops in outer loop";
      continue;
    }

    std::string condReason;
    llvm::DenseSet<Operation *> condOps;
    Block &innerBlock = *inner.getBody();
    if (!collectCondOps(ifOp.getCondition(), innerBlock, condOps, condReason)) {
      summary.reason = "skip peel: " + condReason;
      continue;
    }
    bool extraOps = false;
    for (Operation &op : innerBlock.without_terminator()) {
      if (&op == ifOp.getOperation())
        continue;
      if (!condOps.contains(&op)) {
        extraOps = true;
        break;
      }
    }
    if (extraOps) {
      summary.reason = "skip peel: extra ops in inner loop";
      continue;
    }

    scf::IfOp boundaryIf = ifOp;
    bool hasCorner = false;
    int64_t cornerI = 0;
    int64_t cornerJ = 0;
    {
      bool seenCornerI = false;
      bool seenCornerJ = false;
      bool cornerIAtMin = false;
      bool cornerJAtMin = false;
      std::string cornerReason;
      if (collectCornerConds2D(ifOp.getCondition(), outer.getInductionVar(),
                               inner.getInductionVar(), lbI, ubI, lbJ, ubJ,
                               seenCornerI, cornerIAtMin, seenCornerJ,
                               cornerJAtMin, cornerReason) &&
          seenCornerI && seenCornerJ) {
        scf::IfOp nested;
        if (!findNestedBoundaryIf(ifOp, nested, cornerReason)) {
          summary.reason = "skip peel: " + cornerReason;
          continue;
        }
        hasCorner = true;
        boundaryIf = nested;
        cornerI = cornerIAtMin ? lbI : (ubI - 1);
        cornerJ = cornerJAtMin ? lbJ : (ubJ - 1);
      }
    }

    std::string reason;
    int64_t radius = -1;
    bool seenI = false;
    bool seenJ = false;
    if (!collectBoundaryConds(boundaryIf.getCondition(),
                              outer.getInductionVar(),
                              inner.getInductionVar(), lbI, ubI, lbJ, ubJ,
                              radius, seenI, seenJ, reason)) {
      summary.reason = "skip peel: " + reason;
      continue;
    }
    if (radius < 1) {
      summary.reason = "skip peel: radius < 1";
      continue;
    }
    if (!seenI || !seenJ) {
      summary.reason = "skip peel: boundary only on one dimension";
      continue;
    }

    int64_t lbIInt = lbI + radius;
    int64_t ubIInt = ubI - radius;
    int64_t lbJInt = lbJ + radius;
    int64_t ubJInt = ubJ - radius;
    if (lbIInt >= ubIInt || lbJInt >= ubJInt) {
      summary.reason = "skip peel: interior empty";
      continue;
    }

    if (hasCorner) {
      if (!checkCornerBranchOps(ifOp.getThenRegion(), reason)) {
        summary.reason = "skip peel: " + reason;
        continue;
      }
      if (!checkBranchAccesses(ifOp.getThenRegion(), outer.getInductionVar(),
                               inner.getInductionVar(), radius, reason)) {
        summary.reason = "skip peel: " + reason;
        continue;
      }
    }

    if (!checkBranchAccesses(boundaryIf.getElseRegion(),
                             outer.getInductionVar(),
                             inner.getInductionVar(), radius, reason)) {
      summary.reason = "skip peel: " + reason;
      continue;
    }
    if (!checkBranchAccesses(boundaryIf.getThenRegion(),
                             outer.getInductionVar(),
                             inner.getInductionVar(), radius, reason)) {
      summary.reason = "skip peel: " + reason;
      continue;
    }

    OpBuilder b(outer);
    Location loc = outer.getLoc();
    Value step = b.create<arith::ConstantIndexOp>(loc, 1);

    Value lbIVal = b.create<arith::ConstantIndexOp>(loc, lbI);
    Value ubIVal = b.create<arith::ConstantIndexOp>(loc, ubI);
    Value lbJVal = b.create<arith::ConstantIndexOp>(loc, lbJ);
    Value ubJVal = b.create<arith::ConstantIndexOp>(loc, ubJ);

    Value lbIPlusR = b.create<arith::ConstantIndexOp>(loc, lbI + radius);
    Value ubIMinusR = b.create<arith::ConstantIndexOp>(loc, ubI - radius);
    Value lbJPlusR = b.create<arith::ConstantIndexOp>(loc, lbJ + radius);
    Value ubJMinusR = b.create<arith::ConstantIndexOp>(loc, ubJ - radius);

    if (hasCorner) {
      Value cornerILb = b.create<arith::ConstantIndexOp>(loc, cornerI);
      Value cornerIUb =
          b.create<arith::ConstantIndexOp>(loc, cornerI + 1);
      Value cornerJLb = b.create<arith::ConstantIndexOp>(loc, cornerJ);
      Value cornerJUb =
          b.create<arith::ConstantIndexOp>(loc, cornerJ + 1);
      scf::ForOp cornerLoop =
          buildLoopNestWithBranch(b, loc, cornerILb, cornerIUb, cornerJLb,
                                  cornerJUb, step, ifOp.getThenRegion(),
                                  outer.getInductionVar(),
                                  inner.getInductionVar());
      tagCornerStores(cornerLoop);
    }

    // Top strip.
    buildLoopNestWithBranch(b, loc, lbIVal, lbIPlusR, lbJVal, ubJVal, step,
                            boundaryIf.getThenRegion(),
                            outer.getInductionVar(), inner.getInductionVar());
    // Bottom strip.
    buildLoopNestWithBranch(b, loc, ubIMinusR, ubIVal, lbJVal, ubJVal, step,
                            boundaryIf.getThenRegion(),
                            outer.getInductionVar(), inner.getInductionVar());
    // Middle-left strip.
    buildLoopNestWithBranch(b, loc, lbIPlusR, ubIMinusR, lbJVal, lbJPlusR, step,
                            boundaryIf.getThenRegion(),
                            outer.getInductionVar(), inner.getInductionVar());
    // Middle-right strip.
    buildLoopNestWithBranch(b, loc, lbIPlusR, ubIMinusR, ubJMinusR, ubJVal, step,
                            boundaryIf.getThenRegion(),
                            outer.getInductionVar(), inner.getInductionVar());
    // Interior stencil (no boundary guards).
    buildLoopNestWithBranch(b, loc, lbIPlusR, ubIMinusR, lbJPlusR, ubJMinusR,
                            step, boundaryIf.getElseRegion(),
                            outer.getInductionVar(), inner.getInductionVar());

    outer.erase();

    summary.rank = 2;
    summary.lb = {lbI, lbJ};
    summary.ub = {ubI, ubJ};
    summary.r = radius;
    summary.peeled = true;
    summary.reason = "peel";

    logLine("interior: i=[" + std::to_string(lbIInt) + "," +
            std::to_string(ubIInt) + ") j=[" + std::to_string(lbJInt) + "," +
            std::to_string(ubJInt) + ")");
    return true;
  }

  return false;
}

// Peels a 1D boundary if into two endpoint loops plus interior.
static bool tryPeel1D(func::FuncOp func, SummaryInfo &summary) {
  for (auto loop : func.getOps<scf::ForOp>()) {
    if (!loop.getResults().empty())
      continue;

    int64_t lbI = 0;
    int64_t ubI = 0;
    int64_t stepI = 0;
    if (!matchConstantIndex(loop.getLowerBound(), lbI))
      continue;
    if (!matchConstantIndex(loop.getUpperBound(), ubI))
      continue;
    if (!matchConstantIndex(loop.getStep(), stepI) || stepI != 1)
      continue;

    scf::IfOp ifOp;
    for (Operation &op : loop.getBody()->without_terminator()) {
      if (auto candidate = dyn_cast<scf::IfOp>(op)) {
        if (ifOp) {
          ifOp = nullptr;
          break;
        }
        ifOp = candidate;
      }
    }
    if (!ifOp)
      continue;
    if (ifOp.getNumResults() != 0)
      continue;
    if (!ifOp.getElseRegion().hasOneBlock())
      continue;
    if (!ifOp.getThenRegion().hasOneBlock())
      continue;

    std::string condReason;
    llvm::DenseSet<Operation *> condOps;
    Block &loopBlock = *loop.getBody();
    if (!collectCondOps(ifOp.getCondition(), loopBlock, condOps, condReason)) {
      summary.reason = "skip peel: " + condReason;
      continue;
    }
    bool extraOps = false;
    for (Operation &op : loopBlock.without_terminator()) {
      if (&op == ifOp.getOperation())
        continue;
      if (!condOps.contains(&op)) {
        extraOps = true;
        break;
      }
    }
    if (extraOps) {
      summary.reason = "skip peel: extra ops in loop";
      continue;
    }

    std::string reason;
    int64_t radius = -1;
    if (!collectBoundaryConds1D(ifOp.getCondition(), loop.getInductionVar(),
                                lbI, ubI, radius, reason)) {
      summary.reason = "skip peel: " + reason;
      continue;
    }
    if (radius < 1) {
      summary.reason = "skip peel: radius < 1";
      continue;
    }

    int64_t lbIInt = lbI + radius;
    int64_t ubIInt = ubI - radius;
    if (lbIInt >= ubIInt) {
      summary.reason = "skip peel: interior empty";
      continue;
    }

    if (!checkBranchAccesses1D(ifOp.getElseRegion(), loop.getInductionVar(),
                               radius, reason)) {
      summary.reason = "skip peel: " + reason;
      continue;
    }
    if (!checkBranchAccesses1D(ifOp.getThenRegion(), loop.getInductionVar(),
                               radius, reason)) {
      summary.reason = "skip peel: " + reason;
      continue;
    }

    OpBuilder b(loop);
    Location loc = loop.getLoc();
    Value step = b.create<arith::ConstantIndexOp>(loc, 1);

    Value lbIVal = b.create<arith::ConstantIndexOp>(loc, lbI);
    Value ubIVal = b.create<arith::ConstantIndexOp>(loc, ubI);
    Value lbIPlusR = b.create<arith::ConstantIndexOp>(loc, lbI + radius);
    Value ubIMinusR = b.create<arith::ConstantIndexOp>(loc, ubI - radius);

    buildLoopWithBranch1D(b, loc, lbIVal, lbIPlusR, step,
                          ifOp.getThenRegion(), loop.getInductionVar());
    buildLoopWithBranch1D(b, loc, ubIMinusR, ubIVal, step,
                          ifOp.getThenRegion(), loop.getInductionVar());
    buildLoopWithBranch1D(b, loc, lbIPlusR, ubIMinusR, step,
                          ifOp.getElseRegion(), loop.getInductionVar());

    loop.erase();

    summary.rank = 1;
    summary.lb = {lbI};
    summary.ub = {ubI};
    summary.r = radius;
    summary.peeled = true;
    summary.reason = "peel";

    logLine("interior: i=[" + std::to_string(lbIInt) + "," +
            std::to_string(ubIInt) + ")");
    return true;
  }

  return false;
}

// Rebuilds a loop body by replacing the if with one selected branch.
static bool cloneLoopWithIfBranch(scf::ForOp loop, scf::IfOp ifOp,
                                  bool takeThen, OpBuilder &b) {
  if (!loop.getResults().empty())
    return false;

  auto newLoop = b.create<scf::ForOp>(loop.getLoc(), loop.getLowerBound(),
                                      loop.getUpperBound(), loop.getStep());
  IRMapping mapping;
  mapping.map(loop.getInductionVar(), newLoop.getInductionVar());

  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(newLoop.getBody());
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (&op == ifOp.getOperation()) {
      Region &branch = takeThen ? ifOp.getThenRegion() : ifOp.getElseRegion();
      cloneRegionOps(branch, bodyBuilder, mapping);
      continue;
    }
    bodyBuilder.clone(op, mapping);
  }

  return true;
}

// Hoists a loop-invariant if outside the loop when side effects are safe.
static bool tryUnswitch(func::FuncOp func, SummaryInfo &summary) {
  bool changed = false;

  std::vector<scf::IfOp> ifOps;
  func.walk([&](scf::IfOp op) { ifOps.push_back(op); });
  for (scf::IfOp ifOp : ifOps) {
    scf::ForOp parentLoop = ifOp->getParentOfType<scf::ForOp>();
    if (!parentLoop) {
      logLine("skip unswitch: reason=if not inside scf.for");
      continue;
    }
    if (ifOp->getParentOp() != parentLoop.getOperation()) {
      logLine("skip unswitch: reason=if not direct child of loop");
      continue;
    }
    if (ifOp.getNumResults() != 0) {
      logLine("skip unswitch: reason=if has results");
      continue;
    }
    if (!ifOp.getElseRegion().hasOneBlock()) {
      logLine("skip unswitch: reason=missing else region");
      continue;
    }

    std::vector<scf::ForOp> loops;
    for (scf::ForOp loop = parentLoop; loop;
         loop = loop->getParentOfType<scf::ForOp>()) {
      loops.push_back(loop);
    }

    if (!isLoopInvariantAcross(ifOp.getCondition(), loops)) {
      logLine("skip unswitch: reason=condition not loop-invariant");
      continue;
    }

    llvm::DenseSet<Value> thenStores;
    llvm::DenseSet<Value> elseStores;
    std::string reason;
    if (!collectStoresAndCheckRegion(ifOp.getThenRegion(), thenStores, reason)) {
      logLine("skip unswitch: reason=" + reason);
      continue;
    }
    if (!collectStoresAndCheckRegion(ifOp.getElseRegion(), elseStores, reason)) {
      logLine("skip unswitch: reason=" + reason);
      continue;
    }

    llvm::DenseSet<Value> allStores = thenStores;
    for (Value v : elseStores)
      allStores.insert(v);

    bool hasOutsideStore = false;
    parentLoop.walk([&](memref::StoreOp store) {
      if (ifOp->isAncestor(store.getOperation()))
        return;
      if (allStores.contains(store.getMemref()))
        hasOutsideStore = true;
    });

    if (hasOutsideStore) {
      logLine("skip unswitch: reason=stores outside if");
      continue;
    }

    OpBuilder b(parentLoop);
    auto newIf = b.create<scf::IfOp>(ifOp.getLoc(), ifOp.getCondition(),
                                     /*withElseRegion=*/true);

    {
      OpBuilder thenBuilder = newIf.getThenBodyBuilder();
      if (!cloneLoopWithIfBranch(parentLoop, ifOp, true, thenBuilder)) {
        newIf.erase();
        logLine("skip unswitch: reason=failed cloning then loop");
        continue;
      }
    }
    {
      OpBuilder elseBuilder = newIf.getElseBodyBuilder();
      if (!cloneLoopWithIfBranch(parentLoop, ifOp, false, elseBuilder)) {
        newIf.erase();
        logLine("skip unswitch: reason=failed cloning else loop");
        continue;
      }
    }

    parentLoop.erase();
    changed = true;
    break;
  }

  summary.unswitched = changed;
  if (changed && !summary.peeled) {
    summary.reason = "unswitch";
  }
  return changed;
}

} // namespace

namespace mlir::Neptune::NeptuneIR {
struct SCFBoundarySimplifyPass final
    : public impl::SCFBoundarySimplifyPassBase<SCFBoundarySimplifyPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SummaryInfo summary;
    summary.reason = "skip";

    if (!isKernelFunc(func)) {
      summary.reason = "not a kernel func";
      std::string msg = "func=" + func.getSymName().str() + " rank=0 lb=[]" +
                        " ub=[] r=0 action=skip reason=" + summary.reason;
      logLine(msg);
      return;
    }

    // Prefer 3D/1D/2D peeling; fall back to unswitch when peeling fails.
    if (!tryPeel3D(func, summary)) {
      if (!tryPeel1D(func, summary)) {
        if (!tryPeel(func, summary)) {
          tryUnswitch(func, summary);
        }
      }
    }

    std::string action = "skip";
    if (summary.peeled)
      action = "peel";
    else if (summary.unswitched)
      action = "unswitch";

    std::string rankStr = std::to_string(summary.rank);
    std::string lbStr = joinI64(summary.lb);
    std::string ubStr = joinI64(summary.ub);
    std::string rStr = std::to_string(summary.r);
    if (!summary.peeled && !summary.unswitched) {
      rankStr = "0";
      lbStr = "[]";
      ubStr = "[]";
      rStr = "0";
      if (summary.reason.empty())
        summary.reason = "skip";
    }

    std::string msg = "func=" + func.getSymName().str() + " rank=" + rankStr +
                      " lb=" + lbStr + " ub=" + ubStr + " r=" + rStr +
                      " action=" + action + " reason=" + summary.reason;
    logLine(msg);
  }
};
} // namespace mlir::Neptune::NeptuneIR
