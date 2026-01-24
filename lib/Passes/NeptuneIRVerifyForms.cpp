#include "Dialect/NeptuneIR/NeptuneIRAttrs.h"
#include "Dialect/NeptuneIR/NeptuneIRDialect.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include "Passes/NeptuneIRPasses.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir::Neptune::NeptuneIR {
#define GEN_PASS_DEF_NEPTUNEIRVERIFYFORMSPASS
#include "Passes/NeptuneIRPasses.h.inc"
} // namespace mlir::Neptune::NeptuneIR

namespace {
using namespace mlir::Neptune::NeptuneIR;

static int64_t getRankFromBounds(BoundsAttr b) {
  if (!b) return -1;
  auto lb = b.getLb();
  auto ub = b.getUb();
  if (lb.size() != ub.size()) return -1;
  return (int64_t)lb.size();
}

static LogicalResult verifyApply(ApplyOp op) {
  auto bounds = op.getBounds();
  auto rank = getRankFromBounds(bounds);
  if (rank <= 0)
    return op.emitOpError("invalid bounds: lb/ub length mismatch or empty");

  // radius 必须存在（你说了现在是 array<i64:...>，也就是 DenseI64ArrayAttr）
  auto radiusAttr = op->getAttrOfType<DenseI64ArrayAttr>("radius");
  if (!radiusAttr)
    return op.emitOpError("missing required `radius` attribute (expected array<i64:...>)");
  if ((int64_t)radiusAttr.size() != rank)
    return op.emitOpError("radius rank mismatch: radius has ")
           << radiusAttr.size() << " but bounds rank is " << rank;

  // 检查 body：access offsets 维度匹配且 |offset|<=radius
  Block &body = op.getBody().front();
  for (Operation &nested : body.getOperations()) {
    if (auto acc = dyn_cast<AccessOp>(nested)) {
      auto off = acc.getOffsets();
      if ((int64_t)off.size() != rank) {
        return acc.emitOpError("offset rank mismatch: offsets has ")
               << off.size() << " but apply rank is " << rank;
      }
      for (int64_t d = 0; d < rank; ++d) {
        int64_t o = off[d];
        int64_t r = radiusAttr[d];
        if (std::llabs(o) > r) {
          return acc.emitOpError("offset out of radius at dim ")
                 << d << ": |" << o << "| > " << r;
        }
      }
    }

    if (auto y = dyn_cast<YieldOp>(nested)) {
      // 轻量检查：yield 至少 1 个结果（你也允许多个，但 MVP 我们先只支持 1 个）
      if (y.getNumOperands() != 1)
        return y.emitOpError("currently expect yield with exactly 1 operand for MVP");
      // 类型一致性：yield 标量类型应等于 result TempType 的 elementType
      auto resTy = dyn_cast<TempType>(op.getResult().getType());
      if (!resTy) return op.emitOpError("result must be !neptune.temp<...>");
      Type elemTy = resTy.getElementType();
      if (y.getOperand(0).getType() != elemTy) {
        return y.emitOpError("yield type mismatch: expected ")
               << elemTy << " but got " << y.getOperand(0).getType();
      }
    }
  }

  return success();
}

static LogicalResult verifyLoad(LoadOp op) {
  auto fTy = dyn_cast<FieldType>(op.getVarField().getType());
  auto tTy = dyn_cast<TempType>(op.getResult().getType());
  if (!fTy || !tTy) return op.emitOpError("load expects field -> temp");

  if (fTy.getElementType() != tTy.getElementType())
    return op.emitOpError("elementType mismatch between field and temp");

  // 不强制 bounds 完全相等，但要求 rank 一致
  auto fr = getRankFromBounds(fTy.getBounds());
  auto tr = getRankFromBounds(tTy.getBounds());
  if (fr != tr)
    return op.emitOpError("bounds rank mismatch: field rank ")
           << fr << " temp rank " << tr;

  if (fTy.getLocation() != tTy.getLocation())
    return op.emitOpError("location mismatch between field and temp");

  return success();
}

static LogicalResult verifyStore(StoreOp op) {
  auto tTy = dyn_cast<TempType>(op.getValue().getType());
  auto fTy = dyn_cast<FieldType>(op.getVarField().getType());
  if (!tTy || !fTy) return op.emitOpError("store expects temp -> field");

  if (tTy.getElementType() != fTy.getElementType())
    return op.emitOpError("elementType mismatch between temp and field");

  auto tr = getRankFromBounds(tTy.getBounds());
  auto fr = getRankFromBounds(fTy.getBounds());
  if (tr != fr)
    return op.emitOpError("bounds rank mismatch: temp rank ")
           << tr << " field rank " << fr;

  if (tTy.getLocation() != fTy.getLocation())
    return op.emitOpError("location mismatch between temp and field");

  return success();
}

struct VerifyFormsPattern : OpRewritePattern<ModuleOp> {
  VerifyFormsPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}
  LogicalResult matchAndRewrite(ModuleOp module, PatternRewriter &rewriter) const override {
    (void)rewriter;
    LogicalResult ok = success();

    module.walk([&](Operation *op) {
      if (failed(ok)) return;

      if (auto a = dyn_cast<ApplyOp>(op)) {
        if (failed(verifyApply(a))) ok = failure();
      } else if (auto l = dyn_cast<LoadOp>(op)) {
        if (failed(verifyLoad(l))) ok = failure();
      } else if (auto s = dyn_cast<StoreOp>(op)) {
        if (failed(verifyStore(s))) ok = failure();
      }
    });

    return ok;
  }
};
} // namespace

namespace mlir::Neptune::NeptuneIR {
struct NeptuneIRVerifyFormsPass final
    : public impl::NeptuneIRVerifyFormsPassBase<NeptuneIRVerifyFormsPass> {
  void runOnOperation() override {
    auto module = dyn_cast<ModuleOp>(getOperation());
    MLIRContext *ctx = module.getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<VerifyFormsPattern>(ctx);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace mlir::Neptune::NeptuneIR
