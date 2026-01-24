#include "Dialect/NeptuneIR/NeptuneIRDialect.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include "Passes/NeptuneIRPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir::Neptune::NeptuneIR {
#define GEN_PASS_DEF_NEPTUNEIREMITCHALIDEPASS
#include "Passes/NeptuneIRPasses.h.inc"
} // namespace mlir::Neptune::NeptuneIR

namespace {
using namespace mlir::Neptune::NeptuneIR;

static StringRef dimName(int64_t d) {
  static const char *names[] = {"x", "y", "z", "w"};
  return names[d];
}

static Type getOpaque(MLIRContext *ctx, StringRef spelling) {
  return emitc::OpaqueType::get(ctx, spelling);
}

static Type getConstCharPtr(MLIRContext *ctx) {
  // !emitc.ptr<!emitc.opaque<"const char">>
  Type cchar = emitc::OpaqueType::get(ctx, "const char");
  return emitc::PointerType::get(cchar);
}

static Value constI32(OpBuilder &b, Location loc, int32_t v) {
  auto ty = b.getI32Type();
  return b.create<emitc::ConstantOp>(loc, ty, b.getI32IntegerAttr(v));
}

static void emitVerbatim(OpBuilder &b, Location loc, StringRef text,
                         ValueRange args = {}) {
  // emitc.verbatim "..." (args %a, %b, ...)
  b.create<emitc::VerbatimOp>(loc, b.getStringAttr(text), args);
}

static Value buildCStringLiteral(OpBuilder &b, Location loc,
                                 llvm::StringRef s) {
  MLIRContext *ctx = b.getContext();
  auto ccharTy = emitc::OpaqueType::get(ctx, "const char");
  auto cstrTy = emitc::PointerType::get(
      ccharTy); // !emitc.ptr<!emitc.opaque<"const char">>

  llvm::SmallString<128> quoted;
  quoted.push_back('"');
  quoted.append(s);
  quoted.push_back('"');

  return b.create<emitc::LiteralOp>(loc, cstrTy, b.getStringAttr(quoted));
}

struct HalideExprMVP {
  AccessOp access;
  FloatAttr addConst; // null if absent
};

// MVP: yield = access(in0,[0..]) or access + const
static FailureOr<HalideExprMVP> parseHalideExprMVP(ApplyOp apply) {
  Block &blk = apply.getBody().front();
  auto *term = blk.getTerminator();
  auto y = dyn_cast_or_null<YieldOp>(term);
  if (!y || y.getResults().size() != 1)
    return failure();

  Value v = y.getResults().front();
  Operation *def = v.getDefiningOp();
  if (!def)
    return failure();

  // access(...)
  auto acc = dyn_cast<AccessOp>(def);
  auto addf = dyn_cast<arith::AddFOp>(def);

  if (acc)
    return HalideExprMVP{acc, FloatAttr()};

  if (addf) {
    // one side must be access, other side must be constant float
    auto a0 = addf.getLhs().getDefiningOp<AccessOp>();
    auto a1 = addf.getRhs().getDefiningOp<AccessOp>();
    auto c0 = addf.getLhs().getDefiningOp<arith::ConstantOp>();
    auto c1 = addf.getRhs().getDefiningOp<arith::ConstantOp>();
    if (a0 && c1) {
      if (auto fa = dyn_cast<FloatAttr>(c1.getValue()))
        return HalideExprMVP{a0, fa};
    }
    if (a1 && c0) {
      if (auto fa = dyn_cast<FloatAttr>(c0.getValue()))
        return HalideExprMVP{a1, fa};
    }
  }

  return failure();
}

static Value buildOffsetExpr(OpBuilder &b, Location loc, Value base,
                             int64_t offset, Type exprTy) {
  if (offset == 0)
    return base;

  int32_t absOff = static_cast<int32_t>(offset > 0 ? offset : -offset);
  Value offVal = constI32(b, loc, absOff);
  if (offset > 0)
    return b.create<emitc::AddOp>(loc, exprTy, base, offVal);
  return b.create<emitc::SubOp>(loc, exprTy, base, offVal);
}

static void emitMemberCall(OpBuilder &b, Location loc, Value obj,
                           StringRef method, ValueRange args) {
  // EmitC has no direct op for member calls; use verbatim placeholders.
  llvm::SmallString<128> fmt;
  fmt.append("{}.");
  fmt.append(method);
  fmt.append("(");
  for (int64_t i = 0, e = (int64_t)args.size(); i < e; ++i) {
    if (i)
      fmt.append(", ");
    fmt.append("{}");
  }
  fmt.append(");");

  llvm::SmallVector<Value, 8> fmtArgs;
  fmtArgs.reserve(args.size() + 1);
  fmtArgs.push_back(obj);
  fmtArgs.append(args.begin(), args.end());
  emitVerbatim(b, loc, fmt, fmtArgs);
}

static void emitFuncAssignment(OpBuilder &b, Location loc, Value func,
                               ArrayRef<Value> lhsIdx, Value input,
                               ArrayRef<Value> rhsIdx, Value addConst) {
  // EmitC lacks a func-call assignment op; use verbatim placeholders.
  llvm::SmallString<128> fmt;
  fmt.append("{}(");
  for (int64_t i = 0, e = (int64_t)lhsIdx.size(); i < e; ++i) {
    if (i)
      fmt.append(", ");
    fmt.append("{}");
  }
  fmt.append(") = {}(");
  for (int64_t i = 0, e = (int64_t)rhsIdx.size(); i < e; ++i) {
    if (i)
      fmt.append(", ");
    fmt.append("{}");
  }
  fmt.append(")");

  llvm::SmallVector<Value, 12> fmtArgs;
  fmtArgs.reserve(lhsIdx.size() + rhsIdx.size() + 3);
  fmtArgs.push_back(func);
  fmtArgs.append(lhsIdx.begin(), lhsIdx.end());
  fmtArgs.push_back(input);
  fmtArgs.append(rhsIdx.begin(), rhsIdx.end());

  if (addConst) {
    fmt.append(" + {}");
    fmtArgs.push_back(addConst);
  }
  fmt.append(";");
  emitVerbatim(b, loc, fmt, fmtArgs);
}

static LogicalResult emitOneKernel(OpBuilder &b, Location loc, ApplyOp apply,
                                   int kernelIndex) {
  MLIRContext *ctx = b.getContext();

  // rank
  auto lb = apply.getBounds().getLb().asArrayRef();
  int64_t rank = (int64_t)lb.size();
  if (rank < 1 || rank > 4)
    return failure();

  auto mvpOr = parseHalideExprMVP(apply);
  if (failed(mvpOr))
    return failure();
  auto mvp = *mvpOr;

  auto offsAttr = mvp.access.getOffsets();
  ArrayRef<int64_t> offs = offsAttr;
  if ((int64_t)offs.size() != rank)
    return failure();

  // kernel name: "unnamed_<k>_x<i>o1"
  int nIn = (int)apply.getInputs().size();
  llvm::SmallString<64> kname;
  {
    llvm::raw_svector_ostream os(kname);
    os << "unnamed_" << kernelIndex << "_x" << nIn << "o1";
  }

  // Halide::Type t = Halide::Float(32);
  Value bits = constI32(b, loc, 32);
  auto f32 = b.create<emitc::CallOpaqueOp>(loc, getOpaque(ctx, "Halide::Type"),
                                           b.getStringAttr("Halide::Float"),
                                           ValueRange{bits});

  // dim rank
  Value dim = constI32(b, loc, (int32_t)rank);

  // ImageParam in0 (MVP: 只做 1 个输入；你后面按 nIn 循环复制即可)
  Value inName = buildCStringLiteral(b, loc, "in0");
  auto in0 =
      b.create<emitc::CallOpaqueOp>(loc, getOpaque(ctx, "Halide::ImageParam"),
                                    b.getStringAttr("Halide::ImageParam"),
                                    ValueRange{f32->getResult(0), dim, inName});

  // Func f("kernel")
  Value fName = buildCStringLiteral(b, loc, kname);
  auto f = b.create<emitc::CallOpaqueOp>(loc, getOpaque(ctx, "Halide::Func"),
                                         b.getStringAttr("Halide::Func"),
                                         ValueRange{fName});

  // Vars x,y,z...
  llvm::SmallVector<Value, 4> vars;
  for (int64_t d = 0; d < rank; ++d) {
    Value vn = buildCStringLiteral(b, loc, dimName(d));
    auto v = b.create<emitc::CallOpaqueOp>(loc, getOpaque(ctx, "Halide::Var"),
                                           b.getStringAttr("Halide::Var"),
                                           ValueRange{vn});
    vars.push_back(v.getResult(0));
  }

  Type exprTy = getOpaque(ctx, "Halide::Expr");
  llvm::SmallVector<Value, 4> idxExprs;
  idxExprs.reserve(rank);
  for (int64_t d = 0; d < rank; ++d)
    idxExprs.push_back(buildOffsetExpr(b, loc, vars[d], offs[d], exprTy));

  Value addConst;
  if (mvp.addConst)
    addConst =
        b.create<emitc::ConstantOp>(loc, mvp.addConst.getType(), mvp.addConst);

  emitFuncAssignment(b, loc, f->getResult(0), vars, in0->getResult(0), idxExprs,
                     addConst);

  // Target
  auto target = b.create<emitc::CallOpaqueOp>(
      loc, getOpaque(ctx, "Halide::Target"),
      b.getStringAttr("Halide::get_host_target"), ValueRange{});

  // std::vector<Halide::Argument> args;
  auto args = b.create<emitc::CallOpaqueOp>(
      loc, getOpaque(ctx, "std::vector<Halide::Argument>"),
      b.getStringAttr("std::vector<Halide::Argument>"), ValueRange{});

  // args.push_back(in0);
  emitMemberCall(b, loc, args->getResult(0), "push_back",
                 ValueRange{in0->getResult(0)});

  // f.compile_to_static_library("name", args, "name", target);
  emitMemberCall(
      b, loc, f->getResult(0), "compile_to_static_library",
      ValueRange{fName, args->getResult(0), fName, target->getResult(0)});

  return success();
}

} // namespace

namespace mlir::Neptune::NeptuneIR {

struct NeptuneIREmitCHalidePass final
    : impl::NeptuneIREmitCHalidePassBase<NeptuneIREmitCHalidePass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = cast<ModuleOp>(getOperation());
    MLIRContext *ctx = module.getContext();

    SmallVector<ApplyOp, 16> applies;
    module.walk([&](ApplyOp op) { applies.push_back(op); });
    if (applies.empty())
      return;

    OpBuilder b(ctx);
    b.setInsertionPointToEnd(module.getBody());

    // emitc.file "halide_kernels"
    const std::string &out = outFile.getValue();
    StringRef fileId =
        out.empty() ? StringRef("halide_kernels") : StringRef(out);
    auto file =
        b.create<emitc::FileOp>(module.getLoc(), b.getStringAttr(fileId));

    // file body builder
    Block *fb = file.getBody();
    OpBuilder fbld = OpBuilder::atBlockBegin(fb);

    // includes
    fbld.create<emitc::IncludeOp>(module.getLoc(), b.getStringAttr("Halide.h"),
                                  /*isStandardInclude*/ false);
    fbld.create<emitc::IncludeOp>(module.getLoc(), b.getStringAttr("<vector>"),
                                  /*isStandardInclude*/ false);

    // emitc.func @neptune_build_halide_kernels()
    auto buildFn = fbld.create<emitc::FuncOp>(
        module.getLoc(), fbld.getStringAttr("neptune_build_halide_kernels"),
        FunctionType::get(ctx, /*inputs*/ {}, /*results*/ {}));

    // fill buildFn body
    {
      Block &bb = buildFn.getBody().front();
      OpBuilder bbld = OpBuilder::atBlockBegin(&bb);

      int k = 0;
      for (ApplyOp ap : applies) {
        if (failed(emitOneKernel(bbld, ap.getLoc(), ap, k++))) {
          ap.emitOpError("EmitCHalide MVP only supports: yield=access(in0,off) "
                         "or access+const");
          signalPassFailure();
          return;
        }
      }
      bbld.create<emitc::ReturnOp>(module.getLoc(), Value{});
    }

    // emitc.func @main() -> i32
    auto mainFn = fbld.create<emitc::FuncOp>(
        module.getLoc(), fbld.getStringAttr("main"),
        FunctionType::get(ctx, /*inputs*/ {}, /*results*/ {b.getI32Type()}));

    {
      Block &mb = mainFn.getBody().front();
      OpBuilder mbld = OpBuilder::atBlockBegin(&mb);
      auto callee = SymbolRefAttr::get(ctx, "neptune_build_halide_kernels");
      mbld.create<emitc::CallOp>(module.getLoc(), callee, TypeRange{},
                                 ValueRange{});
      Value z = constI32(mbld, module.getLoc(), 0);
      mbld.create<emitc::ReturnOp>(module.getLoc(), z);
    }
  }
};

} // namespace mlir::Neptune::NeptuneIR
