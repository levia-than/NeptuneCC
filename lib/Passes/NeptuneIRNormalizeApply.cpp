// Keeps neptune.ir.apply radius explicit and consistent with access offsets.
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

// Scans apply.body access offsets and derives per-dimension max(|offset|).
static DenseI64ArrayAttr inferRadiusFromAccess(ApplyOp apply) {
  auto *ctx = apply.getContext();

  // Rank comes from bounds.lb length; lb/ub are DenseI64ArrayAttr here.
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

// Verifies that all access offsets are within the declared radius.
static LogicalResult verifyAccessWithinRadius(ApplyOp apply,
                                              DenseI64ArrayAttr radius) {
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

// Ensures apply radius is explicit and consistent with access offsets.
struct NormalizeOneApply : OpRewritePattern<ApplyOp> {
  NormalizeOneApply(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(ApplyOp op, PatternRewriter &rewriter) const override {
    // Fill missing radius from observed access offsets.
    DenseI64ArrayAttr radius;
    if (auto rAttr = op->getAttrOfType<DenseI64ArrayAttr>("radius")) {
      radius = rAttr;
    } else {
      radius = inferRadiusFromAccess(op);
      rewriter.modifyOpInPlace(op, [&]() { op->setAttr("radius", radius); });
    }

    // Reject accesses that claim a radius smaller than actual offsets.
    if (failed(verifyAccessWithinRadius(op, radius)))
      return failure();

    // Keep the pass tight: only radius inference + validation for now.
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

    // Pattern-driven normalization: no control-flow changes here.
    RewritePatternSet patterns(ctx);
    patterns.add<NormalizeOneApply>(ctx);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace mlir::Neptune::NeptuneIR
