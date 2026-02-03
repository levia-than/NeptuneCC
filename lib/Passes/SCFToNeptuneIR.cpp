// Lower SCF stencil loops to NeptuneIR.
#include "Dialect/NeptuneIR/NeptuneIRAttrs.h"
#include "Dialect/NeptuneIR/NeptuneIRDialect.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include "Passes/NeptuneIRPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <utility>

using namespace mlir;

namespace mlir::Neptune::NeptuneIR {
#define GEN_PASS_DEF_SCFTONEPTUNEIRPASS
#include "Passes/NeptuneIRPasses.h.inc"
} // namespace mlir::Neptune::NeptuneIR

struct OffsetKey1 {
  int64_t di;
};

struct OffsetKey {
  int64_t di;
  int64_t dj;
};

struct OffsetKey3 {
  int64_t dk;
  int64_t dj;
  int64_t di;
};

namespace llvm {
template <>
struct DenseMapInfo<OffsetKey1> {
  static inline OffsetKey1 getEmptyKey() {
    return OffsetKey1{std::numeric_limits<int64_t>::min()};
  }
  static inline OffsetKey1 getTombstoneKey() {
    return OffsetKey1{std::numeric_limits<int64_t>::min() + 1};
  }
  static unsigned getHashValue(const OffsetKey1 &k) {
    return llvm::hash_combine(k.di);
  }
  static bool isEqual(const OffsetKey1 &a, const OffsetKey1 &b) {
    return a.di == b.di;
  }
};

template <>
struct DenseMapInfo<OffsetKey> {
  static inline OffsetKey getEmptyKey() {
    return OffsetKey{std::numeric_limits<int64_t>::min(),
                     std::numeric_limits<int64_t>::min()};
  }
  static inline OffsetKey getTombstoneKey() {
    return OffsetKey{std::numeric_limits<int64_t>::min() + 1,
                     std::numeric_limits<int64_t>::min() + 1};
  }
  static unsigned getHashValue(const OffsetKey &k) {
    return llvm::hash_combine(k.di, k.dj);
  }
  static bool isEqual(const OffsetKey &a, const OffsetKey &b) {
    return a.di == b.di && a.dj == b.dj;
  }
};

template <>
struct DenseMapInfo<OffsetKey3> {
  static inline OffsetKey3 getEmptyKey() {
    return OffsetKey3{std::numeric_limits<int64_t>::min(),
                      std::numeric_limits<int64_t>::min(),
                      std::numeric_limits<int64_t>::min()};
  }
  static inline OffsetKey3 getTombstoneKey() {
    return OffsetKey3{std::numeric_limits<int64_t>::min() + 1,
                      std::numeric_limits<int64_t>::min() + 1,
                      std::numeric_limits<int64_t>::min() + 1};
  }
  static unsigned getHashValue(const OffsetKey3 &k) {
    return llvm::hash_combine(k.dk, k.dj, k.di);
  }
  static bool isEqual(const OffsetKey3 &a, const OffsetKey3 &b) {
    return a.dk == b.dk && a.dj == b.dj && a.di == b.di;
  }
};
} // namespace llvm

namespace {
using namespace mlir::Neptune::NeptuneIR;

static bool matchConstantIndex(Value v, int64_t &out) {
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (!cst.getType().isIndex())
      return false;
    auto attr = dyn_cast<IntegerAttr>(cst.getValue());
    if (!attr)
      return false;
    out = attr.getInt();
    return true;
  }

  if (auto addi = v.getDefiningOp<arith::AddIOp>()) {
    if (!addi.getType().isIndex())
      return false;
    int64_t lhs = 0;
    int64_t rhs = 0;
    if (!matchConstantIndex(addi.getLhs(), lhs) ||
        !matchConstantIndex(addi.getRhs(), rhs))
      return false;
    out = lhs + rhs;
    return true;
  }

  if (auto subi = v.getDefiningOp<arith::SubIOp>()) {
    if (!subi.getType().isIndex())
      return false;
    int64_t lhs = 0;
    int64_t rhs = 0;
    if (!matchConstantIndex(subi.getLhs(), lhs) ||
        !matchConstantIndex(subi.getRhs(), rhs))
      return false;
    out = lhs - rhs;
    return true;
  }

  return false;
}

static bool matchIndex(Value idx, Value iv, int64_t &offset) {
  if (idx == iv) {
    offset = 0;
    return true;
  }

  if (auto addi = idx.getDefiningOp<arith::AddIOp>()) {
    if (!addi.getType().isIndex())
      return false;
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
    if (!subi.getType().isIndex())
      return false;
    int64_t cst = 0;
    if (subi.getLhs() == iv && matchConstantIndex(subi.getRhs(), cst)) {
      offset = -cst;
      return true;
    }
  }

  return false;
}

static Value stripMemrefCasts(Value v) {
  while (auto cast = v.getDefiningOp<memref::CastOp>()) {
    v = cast.getSource();
  }
  return v;
}

static memref::LoadOp getDirectLoad(Value v) {
  return v.getDefiningOp<memref::LoadOp>();
}

static bool isTrivialCopy1(Value storeValue, Value inputMemref,
                           const DenseMap<Value, OffsetKey1> &loadOffsets) {
  auto load = getDirectLoad(storeValue);
  if (!load)
    return false;
  if (stripMemrefCasts(load.getMemref()) != inputMemref)
    return false;
  auto it = loadOffsets.find(load.getResult());
  if (it == loadOffsets.end())
    return false;
  return it->second.di == 0 && loadOffsets.size() == 1;
}

static bool isTrivialCopy2(Value storeValue, Value inputMemref,
                           const DenseMap<Value, OffsetKey> &loadOffsets) {
  auto load = getDirectLoad(storeValue);
  if (!load)
    return false;
  if (stripMemrefCasts(load.getMemref()) != inputMemref)
    return false;
  auto it = loadOffsets.find(load.getResult());
  if (it == loadOffsets.end())
    return false;
  return it->second.di == 0 && it->second.dj == 0 &&
         loadOffsets.size() == 1;
}

static bool isTrivialCopy3(Value storeValue, Value inputMemref,
                           const DenseMap<Value, OffsetKey3> &loadOffsets) {
  auto load = getDirectLoad(storeValue);
  if (!load)
    return false;
  if (stripMemrefCasts(load.getMemref()) != inputMemref)
    return false;
  auto it = loadOffsets.find(load.getResult());
  if (it == loadOffsets.end())
    return false;
  return it->second.di == 0 && it->second.dj == 0 && it->second.dk == 0 &&
         loadOffsets.size() == 1;
}

static bool matchLoadOffsets1(memref::LoadOp load, Value ivI,
                              OffsetKey1 &out) {
  if (load.getIndices().size() != 1)
    return false;

  int64_t di = 0;
  if (!matchIndex(load.getIndices()[0], ivI, di))
    return false;

  if (std::abs(di) > 1)
    return false;

  out = OffsetKey1{di};
  return true;
}

static bool matchLoadOffsets(memref::LoadOp load, Value ivI, Value ivJ,
                             OffsetKey &out) {
  if (load.getIndices().size() != 2)
    return false;

  int64_t di = 0;
  int64_t dj = 0;
  if (!matchIndex(load.getIndices()[0], ivI, di))
    return false;
  if (!matchIndex(load.getIndices()[1], ivJ, dj))
    return false;

  if (std::abs(di) > 1 || std::abs(dj) > 1)
    return false;
  if (std::abs(di) + std::abs(dj) > 1)
    return false;

  out = OffsetKey{di, dj};
  return true;
}

static bool matchLoadOffsets3(memref::LoadOp load, Value ivK, Value ivJ,
                              Value ivI, OffsetKey3 &out) {
  if (load.getIndices().size() != 3)
    return false;

  int64_t dk = 0;
  int64_t dj = 0;
  int64_t di = 0;
  if (!matchIndex(load.getIndices()[0], ivK, dk))
    return false;
  if (!matchIndex(load.getIndices()[1], ivJ, dj))
    return false;
  if (!matchIndex(load.getIndices()[2], ivI, di))
    return false;

  if (std::abs(dk) > 1 || std::abs(dj) > 1 || std::abs(di) > 1)
    return false;
  if (std::abs(dk) + std::abs(dj) + std::abs(di) > 1)
    return false;

  out = OffsetKey3{dk, dj, di};
  return true;
}

struct StencilNestMatch {
  scf::ForOp outer;
  scf::ForOp inner;
  memref::StoreOp store;
  Value inputMemref;
  Value outputMemref;
  Value storeValue;
  Type elemType;
  int64_t lbI;
  int64_t ubI;
  int64_t lbJ;
  int64_t ubJ;
  DenseMap<Value, OffsetKey> loadOffsets;
};

struct StencilNestMatch1 {
  scf::ForOp loop;
  memref::StoreOp store;
  Value inputMemref;
  Value outputMemref;
  Value storeValue;
  Type elemType;
  int64_t lbI;
  int64_t ubI;
  DenseMap<Value, OffsetKey1> loadOffsets;
};

struct StencilNestMatch3 {
  scf::ForOp outer;
  scf::ForOp middle;
  scf::ForOp inner;
  memref::StoreOp store;
  Value inputMemref;
  Value outputMemref;
  Value storeValue;
  Type elemType;
  int64_t lbK;
  int64_t ubK;
  int64_t lbJ;
  int64_t ubJ;
  int64_t lbI;
  int64_t ubI;
  DenseMap<Value, OffsetKey3> loadOffsets;
};

static LogicalResult collectExpr1(Value v, Type elemTy, Value ivI,
                                  Value outputMemref, Value &inputMemref,
                                  DenseMap<Value, OffsetKey1> &loadOffsets,
                                  DenseSet<Value> &visited) {
  if (visited.contains(v))
    return success();
  visited.insert(v);

  Operation *def = v.getDefiningOp();
  if (!def)
    return failure();

  if (auto load = dyn_cast<memref::LoadOp>(def)) {
    auto memrefTy = dyn_cast<MemRefType>(load.getMemref().getType());
    if (!memrefTy || memrefTy.getRank() != 1)
      return failure();
    if (memrefTy.getElementType() != elemTy)
      return failure();
    if (load.getMemref() == outputMemref)
      return failure();

    if (!inputMemref) {
      inputMemref = load.getMemref();
    } else if (inputMemref != load.getMemref()) {
      return failure();
    }

    OffsetKey1 off;
    if (!matchLoadOffsets1(load, ivI, off))
      return failure();
    loadOffsets[load.getResult()] = off;
    return success();
  }

  if (auto cst = dyn_cast<arith::ConstantOp>(def)) {
    auto attr = dyn_cast<IntegerAttr>(cst.getValue());
    if (!attr || cst.getType() != elemTy)
      return failure();
    return success();
  }

  if (auto addi = dyn_cast<arith::AddIOp>(def)) {
    if (addi.getType() != elemTy)
      return failure();
    if (failed(collectExpr1(addi.getLhs(), elemTy, ivI, outputMemref,
                            inputMemref, loadOffsets, visited)))
      return failure();
    if (failed(collectExpr1(addi.getRhs(), elemTy, ivI, outputMemref,
                            inputMemref, loadOffsets, visited)))
      return failure();
    return success();
  }

  return failure();
}

static LogicalResult collectExpr(Value v, Type elemTy, Value ivI, Value ivJ,
                                 Value outputMemref, Value &inputMemref,
                                 DenseMap<Value, OffsetKey> &loadOffsets,
                                 DenseSet<Value> &visited) {
  if (visited.contains(v))
    return success();
  visited.insert(v);

  Operation *def = v.getDefiningOp();
  if (!def)
    return failure();

  if (auto load = dyn_cast<memref::LoadOp>(def)) {
    auto memrefTy = dyn_cast<MemRefType>(load.getMemref().getType());
    if (!memrefTy || memrefTy.getRank() != 2)
      return failure();
    if (memrefTy.getElementType() != elemTy)
      return failure();
    if (load.getMemref() == outputMemref)
      return failure();

    if (!inputMemref) {
      inputMemref = load.getMemref();
    } else if (inputMemref != load.getMemref()) {
      return failure();
    }

    OffsetKey off;
    if (!matchLoadOffsets(load, ivI, ivJ, off))
      return failure();
    loadOffsets[load.getResult()] = off;
    return success();
  }

  if (auto cst = dyn_cast<arith::ConstantOp>(def)) {
    auto attr = dyn_cast<IntegerAttr>(cst.getValue());
    if (!attr || cst.getType() != elemTy)
      return failure();
    return success();
  }

  if (auto addi = dyn_cast<arith::AddIOp>(def)) {
    if (addi.getType() != elemTy)
      return failure();
    if (failed(collectExpr(addi.getLhs(), elemTy, ivI, ivJ, outputMemref,
                           inputMemref, loadOffsets, visited)))
      return failure();
    if (failed(collectExpr(addi.getRhs(), elemTy, ivI, ivJ, outputMemref,
                           inputMemref, loadOffsets, visited)))
      return failure();
    return success();
  }

  return failure();
}

static LogicalResult collectExpr3(Value v, Type elemTy, Value ivK, Value ivJ,
                                  Value ivI, Value outputMemref,
                                  Value &inputMemref,
                                  DenseMap<Value, OffsetKey3> &loadOffsets,
                                  DenseSet<Value> &visited) {
  if (visited.contains(v))
    return success();
  visited.insert(v);

  Operation *def = v.getDefiningOp();
  if (!def)
    return failure();

  if (auto load = dyn_cast<memref::LoadOp>(def)) {
    auto memrefTy = dyn_cast<MemRefType>(load.getMemref().getType());
    if (!memrefTy || memrefTy.getRank() != 3)
      return failure();
    if (memrefTy.getElementType() != elemTy)
      return failure();
    if (load.getMemref() == outputMemref)
      return failure();

    if (!inputMemref) {
      inputMemref = load.getMemref();
    } else if (inputMemref != load.getMemref()) {
      return failure();
    }

    OffsetKey3 off;
    if (!matchLoadOffsets3(load, ivK, ivJ, ivI, off))
      return failure();
    loadOffsets[load.getResult()] = off;
    return success();
  }

  if (auto cst = dyn_cast<arith::ConstantOp>(def)) {
    auto attr = dyn_cast<IntegerAttr>(cst.getValue());
    if (!attr || cst.getType() != elemTy)
      return failure();
    return success();
  }

  if (auto addi = dyn_cast<arith::AddIOp>(def)) {
    if (addi.getType() != elemTy)
      return failure();
    if (failed(collectExpr3(addi.getLhs(), elemTy, ivK, ivJ, ivI, outputMemref,
                            inputMemref, loadOffsets, visited)))
      return failure();
    if (failed(collectExpr3(addi.getRhs(), elemTy, ivK, ivJ, ivI, outputMemref,
                            inputMemref, loadOffsets, visited)))
      return failure();
    return success();
  }

  return failure();
}

static FailureOr<StencilNestMatch> matchStencilNest(scf::ForOp outer) {
  int64_t lbI = 0;
  int64_t ubI = 0;
  int64_t stepI = 0;
  if (!matchConstantIndex(outer.getLowerBound(), lbI) ||
      !matchConstantIndex(outer.getUpperBound(), ubI) ||
      !matchConstantIndex(outer.getStep(), stepI) || stepI != 1)
    return failure();
  if (!outer.getInitArgs().empty())
    return failure();

  scf::ForOp inner;
  Block &outerBody = *outer.getBody();
  for (Operation &op : outerBody.getOperations()) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (inner)
        return failure();
      inner = forOp;
      continue;
    }
    if (isa<arith::AddIOp, arith::SubIOp, arith::ConstantOp, scf::YieldOp>(op))
      continue;
    return failure();
  }
  if (!inner)
    return failure();

  int64_t lbJ = 0;
  int64_t ubJ = 0;
  int64_t stepJ = 0;
  if (!matchConstantIndex(inner.getLowerBound(), lbJ) ||
      !matchConstantIndex(inner.getUpperBound(), ubJ) ||
      !matchConstantIndex(inner.getStep(), stepJ) || stepJ != 1)
    return failure();
  if (!inner.getInitArgs().empty())
    return failure();

  memref::StoreOp store;
  Block &innerBody = *inner.getBody();
  for (Operation &op : innerBody.getOperations()) {
    if (auto s = dyn_cast<memref::StoreOp>(op)) {
      if (store)
        return failure();
      store = s;
      continue;
    }
    if (isa<memref::LoadOp, arith::AddIOp, arith::SubIOp, arith::ConstantOp,
            scf::YieldOp>(op))
      continue;
    return failure();
  }
  if (!store)
    return failure();

  if (store.getIndices().size() != 2)
    return failure();

  int64_t offI = 0;
  int64_t offJ = 0;
  if (!matchIndex(store.getIndices()[0], outer.getInductionVar(), offI) ||
      !matchIndex(store.getIndices()[1], inner.getInductionVar(), offJ) ||
      offI != 0 || offJ != 0)
    return failure();

  auto outMemrefTy = dyn_cast<MemRefType>(store.getMemref().getType());
  if (!outMemrefTy || outMemrefTy.getRank() != 2)
    return failure();
  auto elemTy = outMemrefTy.getElementType();
  if (!isa<IntegerType>(elemTy))
    return failure();

  DenseMap<Value, OffsetKey> loadOffsets;
  DenseSet<Value> visited;
  Value inputMemref;
  if (failed(collectExpr(store.getValue(), elemTy, outer.getInductionVar(),
                         inner.getInductionVar(), store.getMemref(), inputMemref,
                         loadOffsets, visited)))
    return failure();
  if (!inputMemref)
    return failure();
  if (isTrivialCopy2(store.getValue(), inputMemref, loadOffsets))
    return failure();

  return StencilNestMatch{outer,
                          inner,
                          store,
                          inputMemref,
                          store.getMemref(),
                          store.getValue(),
                          elemTy,
                          lbI,
                          ubI,
                          lbJ,
                          ubJ,
                          std::move(loadOffsets)};
}

static FailureOr<StencilNestMatch1> matchStencilNest1(scf::ForOp loop) {
  int64_t lbI = 0;
  int64_t ubI = 0;
  int64_t stepI = 0;
  if (!matchConstantIndex(loop.getLowerBound(), lbI) ||
      !matchConstantIndex(loop.getUpperBound(), ubI) ||
      !matchConstantIndex(loop.getStep(), stepI) || stepI != 1)
    return failure();
  if (!loop.getInitArgs().empty())
    return failure();

  memref::StoreOp store;
  Block &body = *loop.getBody();
  for (Operation &op : body.getOperations()) {
    if (auto s = dyn_cast<memref::StoreOp>(op)) {
      if (store)
        return failure();
      store = s;
      continue;
    }
    if (isa<memref::LoadOp, arith::AddIOp, arith::SubIOp, arith::ConstantOp,
            scf::YieldOp>(op))
      continue;
    return failure();
  }
  if (!store)
    return failure();

  if (store.getIndices().size() != 1)
    return failure();

  int64_t offI = 0;
  if (!matchIndex(store.getIndices()[0], loop.getInductionVar(), offI) ||
      offI != 0)
    return failure();

  auto outMemrefTy = dyn_cast<MemRefType>(store.getMemref().getType());
  if (!outMemrefTy || outMemrefTy.getRank() != 1)
    return failure();
  auto elemTy = outMemrefTy.getElementType();
  if (!isa<IntegerType>(elemTy))
    return failure();

  DenseMap<Value, OffsetKey1> loadOffsets;
  DenseSet<Value> visited;
  Value inputMemref;
  if (failed(collectExpr1(store.getValue(), elemTy, loop.getInductionVar(),
                          store.getMemref(), inputMemref, loadOffsets,
                          visited)))
    return failure();
  if (!inputMemref)
    return failure();
  if (isTrivialCopy1(store.getValue(), inputMemref, loadOffsets))
    return failure();

  return StencilNestMatch1{loop,
                           store,
                           inputMemref,
                           store.getMemref(),
                           store.getValue(),
                           elemTy,
                           lbI,
                           ubI,
                           std::move(loadOffsets)};
}

static FailureOr<StencilNestMatch3> matchStencilNest3(scf::ForOp outer) {
  int64_t lbK = 0;
  int64_t ubK = 0;
  int64_t stepK = 0;
  if (!matchConstantIndex(outer.getLowerBound(), lbK) ||
      !matchConstantIndex(outer.getUpperBound(), ubK) ||
      !matchConstantIndex(outer.getStep(), stepK) || stepK != 1)
    return failure();
  if (!outer.getInitArgs().empty())
    return failure();

  scf::ForOp middle;
  Block &outerBody = *outer.getBody();
  for (Operation &op : outerBody.getOperations()) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (middle)
        return failure();
      middle = forOp;
      continue;
    }
    if (isa<arith::AddIOp, arith::SubIOp, arith::ConstantOp, scf::YieldOp>(op))
      continue;
    return failure();
  }
  if (!middle)
    return failure();

  int64_t lbJ = 0;
  int64_t ubJ = 0;
  int64_t stepJ = 0;
  if (!matchConstantIndex(middle.getLowerBound(), lbJ) ||
      !matchConstantIndex(middle.getUpperBound(), ubJ) ||
      !matchConstantIndex(middle.getStep(), stepJ) || stepJ != 1)
    return failure();
  if (!middle.getInitArgs().empty())
    return failure();

  scf::ForOp inner;
  Block &middleBody = *middle.getBody();
  for (Operation &op : middleBody.getOperations()) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (inner)
        return failure();
      inner = forOp;
      continue;
    }
    if (isa<arith::AddIOp, arith::SubIOp, arith::ConstantOp, scf::YieldOp>(op))
      continue;
    return failure();
  }
  if (!inner)
    return failure();

  int64_t lbI = 0;
  int64_t ubI = 0;
  int64_t stepI = 0;
  if (!matchConstantIndex(inner.getLowerBound(), lbI) ||
      !matchConstantIndex(inner.getUpperBound(), ubI) ||
      !matchConstantIndex(inner.getStep(), stepI) || stepI != 1)
    return failure();
  if (!inner.getInitArgs().empty())
    return failure();

  memref::StoreOp store;
  Block &innerBody = *inner.getBody();
  for (Operation &op : innerBody.getOperations()) {
    if (auto s = dyn_cast<memref::StoreOp>(op)) {
      if (store)
        return failure();
      store = s;
      continue;
    }
    if (isa<memref::LoadOp, arith::AddIOp, arith::SubIOp, arith::ConstantOp,
            scf::YieldOp>(op))
      continue;
    return failure();
  }
  if (!store)
    return failure();

  if (store.getIndices().size() != 3)
    return failure();

  int64_t offK = 0;
  int64_t offJ = 0;
  int64_t offI = 0;
  if (!matchIndex(store.getIndices()[0], outer.getInductionVar(), offK) ||
      !matchIndex(store.getIndices()[1], middle.getInductionVar(), offJ) ||
      !matchIndex(store.getIndices()[2], inner.getInductionVar(), offI) ||
      offK != 0 || offJ != 0 || offI != 0)
    return failure();

  auto outMemrefTy = dyn_cast<MemRefType>(store.getMemref().getType());
  if (!outMemrefTy || outMemrefTy.getRank() != 3)
    return failure();
  auto elemTy = outMemrefTy.getElementType();
  if (!isa<IntegerType>(elemTy))
    return failure();

  DenseMap<Value, OffsetKey3> loadOffsets;
  DenseSet<Value> visited;
  Value inputMemref;
  if (failed(collectExpr3(store.getValue(), elemTy, outer.getInductionVar(),
                          middle.getInductionVar(), inner.getInductionVar(),
                          store.getMemref(), inputMemref, loadOffsets,
                          visited)))
    return failure();
  if (!inputMemref)
    return failure();
  if (isTrivialCopy3(store.getValue(), inputMemref, loadOffsets))
    return failure();

  return StencilNestMatch3{outer,
                           middle,
                           inner,
                           store,
                           inputMemref,
                           store.getMemref(),
                           store.getValue(),
                           elemTy,
                           lbK,
                           ubK,
                           lbJ,
                           ubJ,
                           lbI,
                           ubI,
                           std::move(loadOffsets)};
}

static FailureOr<Value> buildExpr(
    Value v, OpBuilder &b, Value inputTemp,
    const DenseMap<Value, OffsetKey> &loadOffsets,
    DenseMap<OffsetKey, Value> &accessValues, DenseMap<Value, Value> &cache) {
  if (auto it = cache.find(v); it != cache.end())
    return it->second;

  Operation *def = v.getDefiningOp();
  if (!def)
    return failure();

  Location loc = def->getLoc();
  if (auto load = dyn_cast<memref::LoadOp>(def)) {
    auto it = loadOffsets.find(load.getResult());
    if (it == loadOffsets.end())
      return failure();
    OffsetKey off = it->second;
    auto accIt = accessValues.find(off);
    if (accIt != accessValues.end()) {
      cache[v] = accIt->second;
      return accIt->second;
    }

    SmallVector<int64_t, 2> offsets{off.di, off.dj};
    auto acc = b.create<AccessOp>(loc, load.getType(), inputTemp, offsets);
    accessValues[off] = acc.getResult();
    cache[v] = acc.getResult();
    return acc.getResult();
  }

  if (auto cst = dyn_cast<arith::ConstantOp>(def)) {
    auto clone = b.create<arith::ConstantOp>(loc, cst.getValue());
    cache[v] = clone.getResult();
    return clone.getResult();
  }

  if (auto addi = dyn_cast<arith::AddIOp>(def)) {
    auto lhs = buildExpr(addi.getLhs(), b, inputTemp, loadOffsets, accessValues,
                         cache);
    if (failed(lhs))
      return failure();
    auto rhs = buildExpr(addi.getRhs(), b, inputTemp, loadOffsets, accessValues,
                         cache);
    if (failed(rhs))
      return failure();
    auto res = b.create<arith::AddIOp>(loc, *lhs, *rhs);
    cache[v] = res.getResult();
    return res.getResult();
  }

  return failure();
}

static FailureOr<Value> buildExpr1(
    Value v, OpBuilder &b, Value inputTemp,
    const DenseMap<Value, OffsetKey1> &loadOffsets,
    DenseMap<OffsetKey1, Value> &accessValues,
    DenseMap<Value, Value> &cache) {
  if (auto it = cache.find(v); it != cache.end())
    return it->second;

  Operation *def = v.getDefiningOp();
  if (!def)
    return failure();

  Location loc = def->getLoc();
  if (auto load = dyn_cast<memref::LoadOp>(def)) {
    auto it = loadOffsets.find(load.getResult());
    if (it == loadOffsets.end())
      return failure();
    OffsetKey1 off = it->second;
    auto accIt = accessValues.find(off);
    if (accIt != accessValues.end()) {
      cache[v] = accIt->second;
      return accIt->second;
    }

    SmallVector<int64_t, 1> offsets{off.di};
    auto acc = b.create<AccessOp>(loc, load.getType(), inputTemp, offsets);
    accessValues[off] = acc.getResult();
    cache[v] = acc.getResult();
    return acc.getResult();
  }

  if (auto cst = dyn_cast<arith::ConstantOp>(def)) {
    auto clone = b.create<arith::ConstantOp>(loc, cst.getValue());
    cache[v] = clone.getResult();
    return clone.getResult();
  }

  if (auto addi = dyn_cast<arith::AddIOp>(def)) {
    auto lhs = buildExpr1(addi.getLhs(), b, inputTemp, loadOffsets,
                          accessValues, cache);
    if (failed(lhs))
      return failure();
    auto rhs = buildExpr1(addi.getRhs(), b, inputTemp, loadOffsets,
                          accessValues, cache);
    if (failed(rhs))
      return failure();
    auto res = b.create<arith::AddIOp>(loc, *lhs, *rhs);
    cache[v] = res.getResult();
    return res.getResult();
  }

  return failure();
}

static FailureOr<Value> buildExpr3(
    Value v, OpBuilder &b, Value inputTemp,
    const DenseMap<Value, OffsetKey3> &loadOffsets,
    DenseMap<OffsetKey3, Value> &accessValues, DenseMap<Value, Value> &cache) {
  if (auto it = cache.find(v); it != cache.end())
    return it->second;

  Operation *def = v.getDefiningOp();
  if (!def)
    return failure();

  Location loc = def->getLoc();
  if (auto load = dyn_cast<memref::LoadOp>(def)) {
    auto it = loadOffsets.find(load.getResult());
    if (it == loadOffsets.end())
      return failure();
    OffsetKey3 off = it->second;
    auto accIt = accessValues.find(off);
    if (accIt != accessValues.end()) {
      cache[v] = accIt->second;
      return accIt->second;
    }

    SmallVector<int64_t, 3> offsets{off.dk, off.dj, off.di};
    auto acc = b.create<AccessOp>(loc, load.getType(), inputTemp, offsets);
    accessValues[off] = acc.getResult();
    cache[v] = acc.getResult();
    return acc.getResult();
  }

  if (auto cst = dyn_cast<arith::ConstantOp>(def)) {
    auto clone = b.create<arith::ConstantOp>(loc, cst.getValue());
    cache[v] = clone.getResult();
    return clone.getResult();
  }

  if (auto addi = dyn_cast<arith::AddIOp>(def)) {
    auto lhs = buildExpr3(addi.getLhs(), b, inputTemp, loadOffsets,
                          accessValues, cache);
    if (failed(lhs))
      return failure();
    auto rhs = buildExpr3(addi.getRhs(), b, inputTemp, loadOffsets,
                          accessValues, cache);
    if (failed(rhs))
      return failure();
    auto res = b.create<arith::AddIOp>(loc, *lhs, *rhs);
    cache[v] = res.getResult();
    return res.getResult();
  }

  return failure();
}

static LogicalResult rewriteStencil(StencilNestMatch match) {
  auto inMemrefTy = dyn_cast<MemRefType>(match.inputMemref.getType());
  auto outMemrefTy = dyn_cast<MemRefType>(match.outputMemref.getType());
  if (!inMemrefTy || !outMemrefTy)
    return failure();
  if (!inMemrefTy.hasStaticShape() || !outMemrefTy.hasStaticShape())
    return failure();
  if (inMemrefTy.getRank() != 2 || outMemrefTy.getRank() != 2)
    return failure();
  if (inMemrefTy.getElementType() != outMemrefTy.getElementType())
    return failure();
  if (inMemrefTy.getShape() != outMemrefTy.getShape())
    return failure();

  SmallVector<int64_t, 2> fieldLb{0, 0};
  SmallVector<int64_t, 2> fieldUb{inMemrefTy.getDimSize(0),
                                  inMemrefTy.getDimSize(1)};
  SmallVector<int64_t, 2> radius{1, 1};

  if (match.lbI < fieldLb[0] + radius[0] ||
      match.ubI > fieldUb[0] - radius[0])
    return failure();
  if (match.lbJ < fieldLb[1] + radius[1] ||
      match.ubJ > fieldUb[1] - radius[1])
    return failure();

  MLIRContext *ctx = match.outer->getContext();
  Location loc = match.outer->getLoc();
  OpBuilder b(match.outer);

  auto boundsAttr =
      BoundsAttr::get(ctx, DenseI64ArrayAttr::get(ctx, {match.lbI, match.lbJ}),
                      DenseI64ArrayAttr::get(ctx, {match.ubI, match.ubJ}));
  auto radiusAttr = DenseI64ArrayAttr::get(ctx, radius);
  auto fieldBoundsAttr =
      BoundsAttr::get(ctx, DenseI64ArrayAttr::get(ctx, fieldLb),
                      DenseI64ArrayAttr::get(ctx, fieldUb));
  auto locAttr = mlir::Neptune::NeptuneIR::LocationAttr::get(ctx, "cell");
  auto layoutAttr = LayoutAttr::get(ctx, b.getStringAttr("zyx"),
                                    DenseI64ArrayAttr(),
                                    DenseI64ArrayAttr(),
                                    DenseI64ArrayAttr());

  auto fieldTy =
      FieldType::get(ctx, match.elemType, fieldBoundsAttr, locAttr, layoutAttr);
  auto tempTy = TempType::get(ctx, match.elemType, fieldBoundsAttr, locAttr);

  auto inField = b.create<WrapOp>(loc, fieldTy, match.inputMemref);
  auto outField = b.create<WrapOp>(loc, fieldTy, match.outputMemref);
  auto inTemp = b.create<LoadOp>(loc, tempTy, inField);
  auto apply = b.create<ApplyOp>(loc, tempTy, ValueRange{inTemp}, boundsAttr,
                                 StencilShapeAttr(), radiusAttr);

  Block *body = new Block();
  apply.getBody().push_back(body);
  body->addArgument(inTemp.getType(), loc);
  Value regionTemp = body->getArgument(0);

  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(body);
  DenseMap<OffsetKey, Value> accessValues;
  DenseMap<Value, Value> cache;
  auto newVal = buildExpr(match.storeValue, bodyBuilder, regionTemp,
                          match.loadOffsets,
                          accessValues, cache);
  if (failed(newVal))
    return failure();
  bodyBuilder.create<YieldOp>(loc, *newVal);

  b.create<StoreOp>(loc, apply.getResult(), outField, BoundsAttr());

  match.outer.erase();
  return success();
}

static LogicalResult rewriteStencil1(StencilNestMatch1 match) {
  auto inMemrefTy = dyn_cast<MemRefType>(match.inputMemref.getType());
  auto outMemrefTy = dyn_cast<MemRefType>(match.outputMemref.getType());
  if (!inMemrefTy || !outMemrefTy)
    return failure();
  if (!inMemrefTy.hasStaticShape() || !outMemrefTy.hasStaticShape())
    return failure();
  if (inMemrefTy.getRank() != 1 || outMemrefTy.getRank() != 1)
    return failure();
  if (inMemrefTy.getElementType() != outMemrefTy.getElementType())
    return failure();
  if (inMemrefTy.getShape() != outMemrefTy.getShape())
    return failure();

  SmallVector<int64_t, 1> fieldLb{0};
  SmallVector<int64_t, 1> fieldUb{inMemrefTy.getDimSize(0)};
  SmallVector<int64_t, 1> radius{1};

  if (match.lbI < fieldLb[0] + radius[0] ||
      match.ubI > fieldUb[0] - radius[0])
    return failure();

  MLIRContext *ctx = match.loop->getContext();
  Location loc = match.loop->getLoc();
  OpBuilder b(match.loop);

  auto boundsAttr = BoundsAttr::get(
      ctx, DenseI64ArrayAttr::get(ctx, {match.lbI}),
      DenseI64ArrayAttr::get(ctx, {match.ubI}));
  auto radiusAttr = DenseI64ArrayAttr::get(ctx, radius);
  auto fieldBoundsAttr =
      BoundsAttr::get(ctx, DenseI64ArrayAttr::get(ctx, fieldLb),
                      DenseI64ArrayAttr::get(ctx, fieldUb));
  auto locAttr = mlir::Neptune::NeptuneIR::LocationAttr::get(ctx, "cell");
  auto layoutAttr = LayoutAttr::get(ctx, b.getStringAttr("zyx"),
                                    DenseI64ArrayAttr(),
                                    DenseI64ArrayAttr(),
                                    DenseI64ArrayAttr());

  auto fieldTy =
      FieldType::get(ctx, match.elemType, fieldBoundsAttr, locAttr, layoutAttr);
  auto tempTy = TempType::get(ctx, match.elemType, fieldBoundsAttr, locAttr);

  auto inField = b.create<WrapOp>(loc, fieldTy, match.inputMemref);
  auto outField = b.create<WrapOp>(loc, fieldTy, match.outputMemref);
  auto inTemp = b.create<LoadOp>(loc, tempTy, inField);
  auto apply = b.create<ApplyOp>(loc, tempTy, ValueRange{inTemp}, boundsAttr,
                                 StencilShapeAttr(), radiusAttr);

  Block *body = new Block();
  apply.getBody().push_back(body);
  body->addArgument(inTemp.getType(), loc);
  Value regionTemp = body->getArgument(0);

  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(body);
  DenseMap<OffsetKey1, Value> accessValues;
  DenseMap<Value, Value> cache;
  auto newVal = buildExpr1(match.storeValue, bodyBuilder, regionTemp,
                           match.loadOffsets, accessValues, cache);
  if (failed(newVal))
    return failure();
  bodyBuilder.create<YieldOp>(loc, *newVal);

  b.create<StoreOp>(loc, apply.getResult(), outField, BoundsAttr());

  match.loop.erase();
  return success();
}

static LogicalResult rewriteStencil3(StencilNestMatch3 match) {
  auto inMemrefTy = dyn_cast<MemRefType>(match.inputMemref.getType());
  auto outMemrefTy = dyn_cast<MemRefType>(match.outputMemref.getType());
  if (!inMemrefTy || !outMemrefTy)
    return failure();
  if (!inMemrefTy.hasStaticShape() || !outMemrefTy.hasStaticShape())
    return failure();
  if (inMemrefTy.getRank() != 3 || outMemrefTy.getRank() != 3)
    return failure();
  if (inMemrefTy.getElementType() != outMemrefTy.getElementType())
    return failure();
  if (inMemrefTy.getShape() != outMemrefTy.getShape())
    return failure();

  SmallVector<int64_t, 3> fieldLb{0, 0, 0};
  SmallVector<int64_t, 3> fieldUb{inMemrefTy.getDimSize(0),
                                  inMemrefTy.getDimSize(1),
                                  inMemrefTy.getDimSize(2)};
  SmallVector<int64_t, 3> radius{1, 1, 1};

  if (match.lbK < fieldLb[0] + radius[0] ||
      match.ubK > fieldUb[0] - radius[0])
    return failure();
  if (match.lbJ < fieldLb[1] + radius[1] ||
      match.ubJ > fieldUb[1] - radius[1])
    return failure();
  if (match.lbI < fieldLb[2] + radius[2] ||
      match.ubI > fieldUb[2] - radius[2])
    return failure();

  MLIRContext *ctx = match.outer->getContext();
  Location loc = match.outer->getLoc();
  OpBuilder b(match.outer);

  auto boundsAttr = BoundsAttr::get(
      ctx, DenseI64ArrayAttr::get(ctx, {match.lbK, match.lbJ, match.lbI}),
      DenseI64ArrayAttr::get(ctx, {match.ubK, match.ubJ, match.ubI}));
  auto radiusAttr = DenseI64ArrayAttr::get(ctx, radius);
  auto fieldBoundsAttr =
      BoundsAttr::get(ctx, DenseI64ArrayAttr::get(ctx, fieldLb),
                      DenseI64ArrayAttr::get(ctx, fieldUb));
  auto locAttr = mlir::Neptune::NeptuneIR::LocationAttr::get(ctx, "cell");
  auto layoutAttr = LayoutAttr::get(ctx, b.getStringAttr("zyx"),
                                    DenseI64ArrayAttr(),
                                    DenseI64ArrayAttr(),
                                    DenseI64ArrayAttr());

  auto fieldTy =
      FieldType::get(ctx, match.elemType, fieldBoundsAttr, locAttr, layoutAttr);
  auto tempTy = TempType::get(ctx, match.elemType, fieldBoundsAttr, locAttr);

  auto inField = b.create<WrapOp>(loc, fieldTy, match.inputMemref);
  auto outField = b.create<WrapOp>(loc, fieldTy, match.outputMemref);
  auto inTemp = b.create<LoadOp>(loc, tempTy, inField);
  auto apply = b.create<ApplyOp>(loc, tempTy, ValueRange{inTemp}, boundsAttr,
                                 StencilShapeAttr(), radiusAttr);

  Block *body = new Block();
  apply.getBody().push_back(body);
  body->addArgument(inTemp.getType(), loc);
  Value regionTemp = body->getArgument(0);

  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(body);
  DenseMap<OffsetKey3, Value> accessValues;
  DenseMap<Value, Value> cache;
  auto newVal = buildExpr3(match.storeValue, bodyBuilder, regionTemp,
                           match.loadOffsets, accessValues, cache);
  if (failed(newVal))
    return failure();
  bodyBuilder.create<YieldOp>(loc, *newVal);

  b.create<StoreOp>(loc, apply.getResult(), outField, BoundsAttr());

  match.outer.erase();
  return success();
}

} // namespace

namespace mlir::Neptune::NeptuneIR {
struct SCFToNeptuneIRPass final
    : impl::SCFToNeptuneIRPassBase<SCFToNeptuneIRPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    SmallVector<scf::ForOp, 4> candidates;
    func.walk([&](scf::ForOp forOp) {
      if (!forOp->getParentOfType<scf::ForOp>())
        candidates.push_back(forOp);
    });

    for (scf::ForOp outer : candidates) {
      auto match3 = matchStencilNest3(outer);
      if (succeeded(match3)) {
        (void)rewriteStencil3(*match3);
        continue;
      }
      auto match2 = matchStencilNest(outer);
      if (succeeded(match2)) {
        (void)rewriteStencil(*match2);
        continue;
      }
      auto match1 = matchStencilNest1(outer);
      if (succeeded(match1))
        (void)rewriteStencil1(*match1);
    }
  }
};
} // namespace mlir::Neptune::NeptuneIR
