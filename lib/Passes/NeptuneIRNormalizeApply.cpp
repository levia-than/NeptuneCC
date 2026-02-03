// Normalize neptune.ir.apply patterns.
#include "Dialect/NeptuneIR/NeptuneIRAttrs.h"
#include "Dialect/NeptuneIR/NeptuneIRDialect.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include "Passes/NeptuneIRPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir::Neptune::NeptuneIR {
#define GEN_PASS_DEF_NEPTUNEIRNORMALIZEAPPLYPASS
#include "Passes/NeptuneIRPasses.h.inc"
} // namespace mlir::Neptune::NeptuneIR

namespace {
using namespace mlir::Neptune::NeptuneIR;

// 从 apply.body 里扫描所有 ir.access 的 offsets，推导每个维度的 max(|offset|)
static DenseI64ArrayAttr inferRadiusFromAccess(ApplyOp apply) {
  auto *ctx = apply.getContext();

  // rank: 用 bounds.lb 的长度做 rank（你现在的 BoundsAttr 就是 DenseI64ArrayAttr）
  auto bounds = apply.getBounds();
  auto lb = bounds.getLb();
  int64_t rank = (int64_t)lb.size();

  SmallVector<int64_t, 4> r(rank, 0);
  apply.walk([&](AccessOp acc) {
    auto offs = acc.getOffsets(); // DenseI64ArrayAttr
    if ((int64_t)offs.size() != rank) return;
    for (int64_t d = 0; d < rank; ++d) {
      int64_t v = offs[d];
      int64_t av = v >= 0 ? v : -v;
      if (av > r[d]) r[d] = av;
    }
  });

  return DenseI64ArrayAttr::get(ctx, r);
}

static LogicalResult verifyAccessWithinRadius(ApplyOp apply, DenseI64ArrayAttr radius) {
  auto bounds = apply.getBounds();
  int64_t rank = (int64_t)bounds.getLb().size();
  if ((int64_t)radius.size() != rank)
    return apply.emitOpError("radius rank mismatch with bounds rank");

  SmallVector<int64_t, 4> r(rank);
  for (int64_t d = 0; d < rank; ++d) r[d] = radius[d];

  LogicalResult ok = success();
  apply.walk([&](AccessOp acc) {
    auto offs = acc.getOffsets();
    if ((int64_t)offs.size() != rank) {
      ok = failure();
      acc.emitOpError("offset rank mismatch with bounds rank");
      return;
    }
    for (int64_t d = 0; d < rank; ++d) {
      int64_t v = offs[d];
      int64_t av = v >= 0 ? v : -v;
      if (av > r[d]) {
        ok = failure();
        acc.emitOpError("access offset exceeds radius")
            << " dim=" << d << " |off|=" << av << " radius=" << r[d];
        return;
      }
    }
  });
  return ok;
}

struct NormalizeOneApply : OpRewritePattern<ApplyOp> {
  NormalizeOneApply(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(ApplyOp op, PatternRewriter &rewriter) const override {
    // 1) 如果 radius 缺失：推导并补齐
    DenseI64ArrayAttr radius;
    if (auto rAttr = op->getAttrOfType<DenseI64ArrayAttr>("radius")) {
      radius = rAttr;
    } else {
      radius = inferRadiusFromAccess(op);
      rewriter.modifyOpInPlace(op, [&]() { op->setAttr("radius", radius); });
    }

    // 2) 校验 offsets ⊆ radius
    if (failed(verifyAccessWithinRadius(op, radius)))
      return failure();

    // 你还可以在这里做更多 canonicalize：比如把等价表达式规整、合并常量等（以后再加）
    return success();
  }
};

} // namespace

namespace mlir::Neptune::NeptuneIR {

struct NeptuneIRNormalizeApplyPass final
    : impl::NeptuneIRNormalizeApplyPassBase<NeptuneIRNormalizeApplyPass> {
  void runOnOperation() override {
    auto module = dyn_cast<ModuleOp>(getOperation());
    MLIRContext *ctx = module.getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<NormalizeOneApply>(ctx);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace mlir::Neptune::NeptuneIR
