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

struct OffsetKey {
  int64_t di;
  int64_t dj;
};

namespace llvm {
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
} // namespace llvm

namespace {
using namespace mlir::Neptune::NeptuneIR;

static bool matchConstantIndex(Value v, int64_t &out) {
  auto cst = v.getDefiningOp<arith::ConstantOp>();
  if (!cst || !cst.getType().isIndex())
    return false;
  auto attr = dyn_cast<IntegerAttr>(cst.getValue());
  if (!attr)
    return false;
  out = attr.getInt();
  return true;
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
      auto match = matchStencilNest(outer);
      if (succeeded(match))
        (void)rewriteStencil(*match);
    }
  }
};
} // namespace mlir::Neptune::NeptuneIR
