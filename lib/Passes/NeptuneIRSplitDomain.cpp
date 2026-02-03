// Split stencil domains into interior/boundary.
#include "Dialect/NeptuneIR/NeptuneIRAttrs.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include "Passes/NeptuneIRPasses.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir::Neptune::NeptuneIR {
#define GEN_PASS_DEF_NEPTUNEIRSPLITDOMAINPASS
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

static BoundsAttr shrinkToInterior(MLIRContext *ctx, BoundsAttr b, DenseI64ArrayAttr radius) {
  auto lb = b.getLb();
  auto ub = b.getUb();
  int64_t rank = (int64_t)lb.size();

  // Interior bounds = [lb + r, ub - r); keep exclusivity of ub.
  llvm::SmallVector<int64_t, 4> nlb(rank), nub(rank);
  for (int64_t d = 0; d < rank; ++d) {
    nlb[d] = lb[d] + radius[d];
    nub[d] = ub[d] - radius[d];
  }
  return BoundsAttr::get(ctx,
                         DenseI64ArrayAttr::get(ctx, nlb),
                         DenseI64ArrayAttr::get(ctx, nub));
}

struct InteriorOnlyPattern : OpRewritePattern<ApplyOp> {
  InteriorOnlyPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(ApplyOp op, PatternRewriter &rewriter) const override {
    auto bounds = op.getBounds();
    auto rank = getRankFromBounds(bounds);
    if (rank <= 0) return failure();

    auto radius = op->getAttrOfType<DenseI64ArrayAttr>("radius");
    if (!radius || (int64_t)radius.size() != rank) return failure();

    auto ib = shrinkToInterior(op.getContext(), bounds, radius);

    // 检查缩完是否空
    auto lb = ib.getLb();
    auto ub = ib.getUb();
    for (int64_t d = 0; d < rank; ++d) {
      if (lb[d] >= ub[d]) {
        // interior 为空：不改
        return failure();
      }
    }

    if (ib == bounds) return failure();
    // Only rewrite apply's bounds attribute; keep type bounds stable for now.
    op.setBoundsAttr(ib);
    return success();
  }
};
} // namespace

namespace mlir::Neptune::NeptuneIR {
struct NeptuneIRSplitDomainPass final
    : public impl::NeptuneIRSplitDomainPassBase<NeptuneIRSplitDomainPass> {
  using Base::Base;
  
  void runOnOperation() override {
    auto module = dyn_cast<ModuleOp>(getOperation());
    MLIRContext *ctx = module.getContext();

    if (mode != "interior-only") {
      // MVP: only interior-only mode is supported.
      return;
    }

    RewritePatternSet patterns(ctx);
    patterns.add<InteriorOnlyPattern>(ctx);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace mlir::Neptune::NeptuneIR
