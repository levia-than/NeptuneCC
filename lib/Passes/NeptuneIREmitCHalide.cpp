// Emit Halide C++ via EmitC from NeptuneIR.
#include "Dialect/NeptuneIR/NeptuneIRDialect.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include "Passes/NeptuneIRPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir::Neptune::NeptuneIR {
#define GEN_PASS_DEF_NEPTUNEIREMITCHALIDEPASS
#include "Passes/NeptuneIRPasses.h.inc"
} // namespace mlir::Neptune::NeptuneIR

namespace {
using namespace mlir::Neptune::NeptuneIR;

static Type getOpaque(MLIRContext *ctx, StringRef spelling) {
  return emitc::OpaqueType::get(ctx, spelling);
}

static Value constI32(OpBuilder &b, Location loc, int32_t v) {
  return b.create<emitc::ConstantOp>(loc, b.getI32Type(),
                                     b.getI32IntegerAttr(v));
}

static emitc::CallOpaqueOp buildInvoke(OpBuilder &b, Location loc,
                                       TypeRange resultTypes,
                                       ValueRange operands) {
  return b.create<emitc::CallOpaqueOp>(loc, resultTypes, "std::invoke",
                                       operands);
}

static bool isOpaqueType(Type type, StringRef spelling) {
  if (auto opaque = dyn_cast<emitc::OpaqueType>(type))
    return opaque.getValue() == spelling;
  return false;
}

static bool isHalideExprOrVar(Type type) {
  return isOpaqueType(type, "Halide::Expr") ||
         isOpaqueType(type, "Halide::Var");
}

static void appendInt(llvm::SmallVectorImpl<char> &storage, int64_t value) {
  if (value == 0) {
    storage.push_back('0');
    return;
  }
  if (value < 0) {
    storage.push_back('-');
    value = -value;
  }
  char buf[32];
  int len = 0;
  while (value > 0) {
    buf[len++] = static_cast<char>('0' + (value % 10));
    value /= 10;
  }
  for (int i = len - 1; i >= 0; --i)
    storage.push_back(buf[i]);
}

static Value buildCStringLiteral(OpBuilder &b, Location loc, StringRef text) {
  MLIRContext *ctx = b.getContext();
  Type ccharTy = getOpaque(ctx, "const char");
  Type cstrTy = emitc::PointerType::get(ccharTy);

  llvm::SmallVector<char, 64> storage;
  storage.push_back('"');
  storage.append(text.begin(), text.end());
  storage.push_back('"');

  StringRef quoted(storage.data(), storage.size());
  return b.create<emitc::LiteralOp>(loc, cstrTy, b.getStringAttr(quoted));
}

static Value buildIndexedCStringLiteral(OpBuilder &b, Location loc,
                                        StringRef prefix, int64_t index) {
  MLIRContext *ctx = b.getContext();
  Type ccharTy = getOpaque(ctx, "const char");
  Type cstrTy = emitc::PointerType::get(ccharTy);

  llvm::SmallVector<char, 64> storage;
  storage.push_back('"');
  storage.append(prefix.begin(), prefix.end());
  appendInt(storage, index);
  storage.push_back('"');

  StringRef quoted(storage.data(), storage.size());
  return b.create<emitc::LiteralOp>(loc, cstrTy, b.getStringAttr(quoted));
}

static Value buildIntLiteral(OpBuilder &b, Location loc, int64_t value) {
  MLIRContext *ctx = b.getContext();
  Type intTy = getOpaque(ctx, "int");
  llvm::SmallVector<char, 32> storage;
  appendInt(storage, value);
  StringRef literal(storage.data(), storage.size());
  return b.create<emitc::LiteralOp>(loc, intTy, b.getStringAttr(literal));
}

struct ScheduleAttr {
  llvm::SmallVector<int64_t, 4> split;
  int64_t vec = 1;
  int64_t unroll = 1;
  int64_t threads = 1;
  StringRef unrollDim = "none";
  StringRef parDim = "none";
  bool valid = false;
};

static ScheduleAttr parseScheduleAttr(ApplyOp apply, size_t rank) {
  ScheduleAttr info;
  auto dict = apply->getAttrOfType<DictionaryAttr>("neptune.schedule");
  if (!dict)
    return info;

  auto splitAttr = dict.getAs<DenseI64ArrayAttr>("split");
  if (!splitAttr)
    return info;
  auto splitVals = splitAttr.asArrayRef();
  if (splitVals.size() != rank)
    return info;
  info.split.assign(splitVals.begin(), splitVals.end());

  if (auto vecAttr = dict.getAs<IntegerAttr>("vec"))
    info.vec = vecAttr.getInt();
  if (auto unrollAttr = dict.getAs<IntegerAttr>("unroll_factor"))
    info.unroll = unrollAttr.getInt();
  else if (auto unrollAttr = dict.getAs<IntegerAttr>("unroll"))
    info.unroll = unrollAttr.getInt();
  if (auto unrollDimAttr = dict.getAs<StringAttr>("unroll_dim"))
    info.unrollDim = unrollDimAttr.getValue();
  if (auto parAttr = dict.getAs<StringAttr>("par_dim"))
    info.parDim = parAttr.getValue();
  if (auto threadsAttr = dict.getAs<IntegerAttr>("threads"))
    info.threads = threadsAttr.getInt();

  info.valid = true;
  return info;
}

struct ExprEmitter {
  ExprEmitter(OpBuilder &b, Location loc, Type exprTy,
              DenseMap<Value, Value> &inputToImage,
              ArrayRef<Value> halideVars)
      : b(b), loc(loc), exprTy(exprTy), inputToImage(inputToImage),
        halideVars(halideVars.begin(), halideVars.end()) {}

  FailureOr<Value> emit(Value v) {
    auto it = cache.find(v);
    if (it != cache.end())
      return it->second;

    if (auto acc = v.getDefiningOp<AccessOp>()) {
      auto res = emitAccess(acc);
      if (failed(res))
        return failure();
      cache[v] = *res;
      return *res;
    }
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
      auto res = emitConstant(cst);
      if (failed(res))
        return failure();
      cache[v] = *res;
      return *res;
    }
    if (auto addi = v.getDefiningOp<arith::AddIOp>()) {
      auto res = emitBinary(addi.getLhs(), addi.getRhs(),
                            [&](Value l, Value r) {
                              return b.create<emitc::AddOp>(loc, exprTy, l, r);
                            });
      if (failed(res))
        return failure();
      cache[v] = *res;
      return *res;
    }
    if (auto addf = v.getDefiningOp<arith::AddFOp>()) {
      auto res = emitBinary(addf.getLhs(), addf.getRhs(),
                            [&](Value l, Value r) {
                              return b.create<emitc::AddOp>(loc, exprTy, l, r);
                            });
      if (failed(res))
        return failure();
      cache[v] = *res;
      return *res;
    }
    if (auto subi = v.getDefiningOp<arith::SubIOp>()) {
      auto res = emitBinary(subi.getLhs(), subi.getRhs(),
                            [&](Value l, Value r) {
                              return b.create<emitc::SubOp>(loc, exprTy, l, r);
                            });
      if (failed(res))
        return failure();
      cache[v] = *res;
      return *res;
    }
    if (auto subf = v.getDefiningOp<arith::SubFOp>()) {
      auto res = emitBinary(subf.getLhs(), subf.getRhs(),
                            [&](Value l, Value r) {
                              return b.create<emitc::SubOp>(loc, exprTy, l, r);
                            });
      if (failed(res))
        return failure();
      cache[v] = *res;
      return *res;
    }
    if (auto muli = v.getDefiningOp<arith::MulIOp>()) {
      auto res = emitBinary(muli.getLhs(), muli.getRhs(),
                            [&](Value l, Value r) {
                              return b.create<emitc::MulOp>(loc, exprTy, l, r);
                            });
      if (failed(res))
        return failure();
      cache[v] = *res;
      return *res;
    }
    if (auto mulf = v.getDefiningOp<arith::MulFOp>()) {
      auto res = emitBinary(mulf.getLhs(), mulf.getRhs(),
                            [&](Value l, Value r) {
                              return b.create<emitc::MulOp>(loc, exprTy, l, r);
                            });
      if (failed(res))
        return failure();
      cache[v] = *res;
      return *res;
    }
    if (auto divi = v.getDefiningOp<arith::DivSIOp>()) {
      auto res = emitBinary(divi.getLhs(), divi.getRhs(),
                            [&](Value l, Value r) {
                              return b.create<emitc::DivOp>(loc, exprTy, l, r);
                            });
      if (failed(res))
        return failure();
      cache[v] = *res;
      return *res;
    }
    if (auto divf = v.getDefiningOp<arith::DivFOp>()) {
      auto res = emitBinary(divf.getLhs(), divf.getRhs(),
                            [&](Value l, Value r) {
                              return b.create<emitc::DivOp>(loc, exprTy, l, r);
                            });
      if (failed(res))
        return failure();
      cache[v] = *res;
      return *res;
    }

    if (auto *def = v.getDefiningOp())
      def->emitOpError("unsupported op in Halide EmitC expression");
    else
      emitError(loc) << "unsupported block argument in Halide expression";
    return failure();
  }

  FailureOr<Value> emitAccess(AccessOp acc) {
    auto it = inputToImage.find(acc.getInput());
    if (it == inputToImage.end()) {
      acc.emitOpError("access input is not mapped to an ImageParam");
      return failure();
    }

    auto offsets = acc.getOffsets();
    if (offsets.size() != halideVars.size()) {
      acc.emitOpError("access rank does not match apply rank");
      return failure();
    }

    // Halide expects dim0 to be the fastest-varying dimension (stride == 1),
    // so map dim0 -> last IR index, dimN -> first IR index.
    llvm::SmallVector<Value, 4> args;
    args.reserve(1 + halideVars.size());
    args.push_back(it->second);
    for (size_t i = 0, e = halideVars.size(); i < e; ++i) {
      size_t irIndex = e - 1 - i;
      Value ex = buildOffset(halideVars[i], offsets[irIndex]);
      args.push_back(ex);
    }

    auto call = buildInvoke(b, loc, TypeRange{exprTy}, args);
    return call.getResult(0);
  }

  FailureOr<Value> emitConstant(arith::ConstantOp cst) {
    Type cstTy = cst.getType();
    Attribute attr = cst.getValue();
    if (auto fTy = dyn_cast<FloatType>(cstTy)) {
      if (!fTy.isF32()) {
        cst.emitOpError("only f32 constants supported for Halide EmitC");
        return failure();
      }
      auto fAttr = dyn_cast<FloatAttr>(attr);
      if (!fAttr) {
        cst.emitOpError("expected float constant attribute");
        return failure();
      }
      llvm::SmallVector<char, 32> storage;
      fAttr.getValue().toString(storage, /*FormatPrecision=*/0,
                                /*FormatMaxPadding=*/0,
                                /*TruncateZero=*/false);
      bool hasDotOrExp = false;
      for (char c : storage) {
        if (c == '.' || c == 'e' || c == 'E') {
          hasDotOrExp = true;
          break;
        }
      }
      if (!hasDotOrExp) {
        storage.push_back('.');
        storage.push_back('0');
      }
      storage.push_back('f');
      StringRef literal(storage.data(), storage.size());
      auto lit =
          b.create<emitc::LiteralOp>(loc, exprTy, b.getStringAttr(literal));
      return lit.getResult();
    } else if (auto iTy = dyn_cast<IntegerType>(cstTy)) {
      if (!iTy.isInteger(32)) {
        cst.emitOpError("only i32 constants supported for Halide EmitC");
        return failure();
      }
      auto iAttr = dyn_cast<IntegerAttr>(attr);
      if (!iAttr) {
        cst.emitOpError("expected integer constant attribute");
        return failure();
      }
      llvm::SmallVector<char, 32> storage;
      bool isSigned = !iTy.isUnsigned();
      iAttr.getValue().toString(storage, /*Radix=*/10, isSigned,
                                /*formatAsCLiteral=*/false);
      StringRef literal(storage.data(), storage.size());
      auto lit =
          b.create<emitc::LiteralOp>(loc, exprTy, b.getStringAttr(literal));
      return lit.getResult();
    } else {
      cst.emitOpError("unsupported constant type for Halide EmitC");
      return failure();
    }
  }

  template <typename CreateFn>
  FailureOr<Value> emitBinary(Value lhs, Value rhs, CreateFn createOp) {
    auto l = emit(lhs);
    auto r = emit(rhs);
    if (failed(l) || failed(r))
      return failure();
    return createOp(*l, *r).getResult();
  }

  Value buildOffset(Value base, int64_t offset) {
    if (offset == 0)
      return base;

    int64_t absOff = offset > 0 ? offset : -offset;
    Value offVal = buildIntLiteral(b, loc, absOff);
    if (isHalideExprOrVar(base.getType())) {
      StringRef op = offset > 0 ? "Halide::operator+" : "Halide::operator-";
      auto call =
          b.create<emitc::CallOpaqueOp>(loc, TypeRange{exprTy}, op,
                                        ValueRange{base, offVal});
      return call.getResult(0);
    }

    Type resTy = base.getType();
    if (offset > 0)
      return b.create<emitc::AddOp>(loc, resTy, base, offVal).getResult();
    return b.create<emitc::SubOp>(loc, resTy, base, offVal).getResult();
  }

  OpBuilder &b;
  Location loc;
  Type exprTy;
  DenseMap<Value, Value> &inputToImage;
  llvm::SmallVector<Value, 4> halideVars;
  DenseMap<Value, Value> cache;
};

static LogicalResult emitOneKernel(OpBuilder &b, func::FuncOp func,
                                   ApplyOp apply) {
  MLIRContext *ctx = b.getContext();
  Location loc = apply.getLoc();

  auto bounds = apply.getBounds();
  auto lb = bounds.getLb().asArrayRef();
  auto ub = bounds.getUb().asArrayRef();
  if (lb.empty() || lb.size() != ub.size()) {
    apply.emitOpError("apply bounds are invalid");
    return failure();
  }
  size_t rank = lb.size();

  auto tempTy = dyn_cast<TempType>(apply.getResult().getType());
  if (!tempTy) {
    apply.emitOpError("apply result is not a neptune temp type");
    return failure();
  }
  Type elemTy = tempTy.getElementType();

  Type halideTypeTy = getOpaque(ctx, "Halide::Type");
  Type imageParamTy = getOpaque(ctx, "Halide::ImageParam");
  Type funcTy = getOpaque(ctx, "Halide::Func");
  Type funcRefTy = getOpaque(ctx, "Halide::FuncRef");
  Type varTy = getOpaque(ctx, "Halide::Var");
  Type exprTy = getOpaque(ctx, "Halide::Expr");
  Type argsTy = getOpaque(ctx, "std::vector<Halide::Argument>");
  Type targetTy = getOpaque(ctx, "Halide::Target");

  Value bits = constI32(b, loc, 32);
  Value halideElemTy;
  if (auto fTy = dyn_cast<FloatType>(elemTy)) {
    if (!fTy.isF32()) {
      apply.emitOpError("only f32 element types supported for Halide EmitC");
      return failure();
    }
    auto f32 = b.create<emitc::CallOpaqueOp>(
        loc, TypeRange{halideTypeTy}, "Halide::Float", ValueRange{bits});
    halideElemTy = f32.getResult(0);
  } else if (auto iTy = dyn_cast<IntegerType>(elemTy)) {
    if (!iTy.isInteger(32)) {
      apply.emitOpError("only i32 element types supported for Halide EmitC");
      return failure();
    }
    auto i32 = b.create<emitc::CallOpaqueOp>(
        loc, TypeRange{halideTypeTy}, "Halide::Int", ValueRange{bits});
    halideElemTy = i32.getResult(0);
  } else {
    apply.emitOpError("unsupported element type for Halide EmitC");
    return failure();
  }

  Value dim = constI32(b, loc, static_cast<int32_t>(rank));
  StringRef kernelName = func.getSymName();
  Value kernelNameLit = buildCStringLiteral(b, loc, kernelName);

  llvm::SmallVector<Value, 4> imageParams;
  imageParams.reserve(apply.getInputs().size());
  auto inputs = apply.getInputs();
  for (size_t i = 0, e = inputs.size(); i < e; ++i) {
    Value inName = buildIndexedCStringLiteral(b, loc, "in", i);
    auto ip = b.create<emitc::CallOpaqueOp>(
        loc, TypeRange{imageParamTy}, "Halide::ImageParam",
        ValueRange{halideElemTy, dim, inName});
    imageParams.push_back(ip.getResult(0));
  }

  auto out =
      b.create<emitc::CallOpaqueOp>(loc, TypeRange{funcTy}, "Halide::Func",
                                    ValueRange{kernelNameLit});
  Value outFunc = out.getResult(0);

  llvm::SmallVector<Value, 4> vars;
  vars.reserve(rank);
  for (size_t i = 0; i < rank; ++i) {
    Value vName = buildIndexedCStringLiteral(b, loc, "d", i);
    Value v =
        b.create<emitc::CallOpaqueOp>(loc, TypeRange{varTy}, "Halide::Var",
                                      ValueRange{vName})
            .getResult(0);
    vars.push_back(v);
  }

  auto argVec =
      b.create<emitc::CallOpaqueOp>(loc, TypeRange{argsTy},
                                    "std::vector<Halide::Argument>",
                                    ValueRange{});
  Value argsVal = argVec.getResult(0);
  for (Value ip : imageParams) {
    b.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "neptune_halide::push_arg",
        ValueRange{argsVal, ip});
  }

  auto target =
      b.create<emitc::CallOpaqueOp>(loc, TypeRange{targetTy},
                                    "Halide::get_host_target", ValueRange{});
  Value targetVal = target.getResult(0);

  DenseMap<Value, Value> inputToImage;
  Block &body = apply.getBody().front();
  auto bodyArgs = body.getArguments();
  if (bodyArgs.size() != imageParams.size()) {
    apply.emitOpError("apply region arguments do not match input count");
    return failure();
  }
  for (size_t i = 0, e = bodyArgs.size(); i < e; ++i)
    inputToImage[bodyArgs[i]] = imageParams[i];
  for (size_t i = 0, e = inputs.size(); i < e; ++i)
    inputToImage[inputs[i]] = imageParams[i];

  auto *term = body.getTerminator();
  auto yield = dyn_cast_or_null<YieldOp>(term);
  if (!yield || yield.getResults().size() != 1) {
    apply.emitOpError("apply region must yield exactly one value");
    return failure();
  }

  ExprEmitter emitter(b, loc, exprTy, inputToImage, vars);
  FailureOr<Value> expr = emitter.emit(yield.getResults().front());
  if (failed(expr))
    return failure();

  {
    llvm::SmallVector<Value, 4> funcArgs;
    funcArgs.reserve(1 + vars.size());
    funcArgs.push_back(outFunc);
    funcArgs.append(vars.begin(), vars.end());
    auto funcRefCall = buildInvoke(b, loc, TypeRange{funcRefTy}, funcArgs);
    Value funcRef = funcRefCall.getResult(0);
    b.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "neptune_halide::assign",
        ValueRange{funcRef, *expr});
  }

  {
    ScheduleAttr schedule = parseScheduleAttr(apply, rank);
    if (schedule.valid) {
      int64_t parDim = 0;
      if (schedule.parDim == "y")
        parDim = 1;
      else if (schedule.parDim == "z")
        parDim = 2;

      int64_t unrollY = (schedule.unrollDim == "y") ? schedule.unroll : 1;
      int64_t enableParallel =
          (parDim != 0 && schedule.threads > 1) ? 1 : 0;

      llvm::SmallVector<Value, 6> args;
      args.push_back(outFunc);
      if (rank == 1) {
        args.push_back(buildIntLiteral(b, loc, schedule.split[0]));
        args.push_back(buildIntLiteral(b, loc, schedule.vec));
        args.push_back(buildIntLiteral(b, loc, enableParallel));
        args.push_back(buildIntLiteral(b, loc, schedule.threads));
        args.push_back(buildIntLiteral(b, loc, unrollY));
        b.create<emitc::CallOpaqueOp>(loc, TypeRange{},
                                      "neptune_halide::schedule_1d", args);
      } else if (rank == 2) {
        int64_t tx = schedule.split[1];
        int64_t ty = schedule.split[0];
        args.push_back(buildIntLiteral(b, loc, tx));
        args.push_back(buildIntLiteral(b, loc, ty));
        args.push_back(buildIntLiteral(b, loc, schedule.vec));
        args.push_back(buildIntLiteral(b, loc, unrollY));
        args.push_back(buildIntLiteral(b, loc, enableParallel));
        args.push_back(buildIntLiteral(b, loc, schedule.threads));
        b.create<emitc::CallOpaqueOp>(loc, TypeRange{},
                                      "neptune_halide::schedule_2d", args);
      } else if (rank == 3) {
        int64_t tx = schedule.split[2];
        int64_t ty = schedule.split[1];
        int64_t tz = schedule.split[0];
        args.push_back(buildIntLiteral(b, loc, tx));
        args.push_back(buildIntLiteral(b, loc, ty));
        args.push_back(buildIntLiteral(b, loc, tz));
        args.push_back(buildIntLiteral(b, loc, schedule.vec));
        args.push_back(buildIntLiteral(b, loc, unrollY));
        args.push_back(buildIntLiteral(b, loc, parDim));
        args.push_back(buildIntLiteral(b, loc, enableParallel));
        args.push_back(buildIntLiteral(b, loc, schedule.threads));
        b.create<emitc::CallOpaqueOp>(loc, TypeRange{},
                                      "neptune_halide::schedule_3d", args);
      }
    }

    b.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "neptune_halide::compile",
        ValueRange{outFunc, kernelNameLit, argsVal, kernelNameLit,
                   targetVal});
  }
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

    SmallVector<std::pair<func::FuncOp, ApplyOp>, 8> kernels;
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      SmallVector<ApplyOp, 4> applies;
      func.walk([&](ApplyOp op) { applies.push_back(op); });
      if (applies.empty())
        continue;
      if (applies.size() != 1) {
        func.emitOpError("multiple apply in one function not supported yet");
        signalPassFailure();
        return;
      }
      kernels.emplace_back(func, applies.front());
    }

    if (kernels.empty())
      return;

    OpBuilder b(ctx);
    b.setInsertionPointToEnd(module.getBody());

    StringRef fileId = outFile;
    if (fileId.empty())
      fileId = "halide_kernels";

    auto file = b.create<emitc::FileOp>(module.getLoc(), fileId);

    Block *fb = file.getBody();
    OpBuilder fbld = OpBuilder::atBlockBegin(fb);

    fbld.create<emitc::IncludeOp>(module.getLoc(),
                                  fbld.getStringAttr("Halide.h"),
                                  /*isStandardInclude*/ false);
    fbld.create<emitc::IncludeOp>(
        module.getLoc(), fbld.getStringAttr("NeptuneHalideHelpers.h"),
        /*isStandardInclude*/ false);
    fbld.create<emitc::IncludeOp>(module.getLoc(),
                                  fbld.getStringAttr("vector"),
                                  /*isStandardInclude*/ true);
    fbld.create<emitc::IncludeOp>(module.getLoc(),
                                  fbld.getStringAttr("functional"),
                                  /*isStandardInclude*/ true);
    fbld.create<emitc::IncludeOp>(module.getLoc(),
                                  fbld.getStringAttr("cstdint"),
                                  /*isStandardInclude*/ true);

    auto buildFn = fbld.create<emitc::FuncOp>(
        module.getLoc(), "neptuneir_build_halide_kernels",
        FunctionType::get(ctx, /*inputs*/ {}, /*results*/ {}));

    {
      Block *bb = buildFn.addEntryBlock();
      OpBuilder bbld = OpBuilder::atBlockBegin(bb);

      for (auto &pair : kernels) {
        if (failed(emitOneKernel(bbld, pair.first, pair.second))) {
          signalPassFailure();
          return;
        }
      }
      bbld.create<emitc::ReturnOp>(module.getLoc(), Value());
    }

    // Keep only the emitc.file for downstream translation.
    Operation *fileOp = file.getOperation();
    fileOp->remove();
    module.getBody()->clear();
    module.getBody()->push_back(fileOp);
  }
};

} // namespace mlir::Neptune::NeptuneIR
