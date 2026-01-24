/*
 * @Author: leviathan 670916484@qq.com
 * @Date: 2025-11-29 11:41:01
 * @LastEditors: leviathan 670916484@qq.com
 * @LastEditTime: 2025-11-29 11:48:09
 * @FilePath: /neptune-pde-solver/lib/Dialect/NeptuneIR/NeptuneIRVerifier.cpp
 * @Description:
 *
 * Copyright (c) 2025 by leviathan, All Rights Reserved.
 */
#include "Dialect/NeptuneIR/NeptuneIRAttrs.h"
#include "Dialect/NeptuneIR/NeptuneIRDialect.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include "Utils/NeptuneIRVerifierUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include <optional>

using namespace mlir;
using namespace mlir::Neptune::NeptuneIR;

LogicalResult mlir::Neptune::NeptuneIR::BoundsAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI64ArrayAttr lb,
    DenseI64ArrayAttr ub) {
  auto lbVals = lb.asArrayRef();
  auto ubVals = ub.asArrayRef();
  if (lbVals.size() != ubVals.size())
    return emitError() << "lb/ub must have the same rank";
  if (lbVals.empty())
    return emitError() << "bounds rank must be > 0";
  for (size_t i = 0; i < lbVals.size(); ++i) {
    if (lbVals[i] >= ubVals[i]) {
      return emitError() << "invalid bounds at dim " << i
                         << ": lb=" << lbVals[i] << " ub=" << ubVals[i];
    }
  }
  return success();
}

LogicalResult mlir::Neptune::NeptuneIR::LocationAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, StringRef kind) {
  if (kind == "cell" || kind == "vertex" || kind == "face_x" ||
      kind == "face_y" || kind == "face_z") {
    return success();
  }
  return emitError() << "unknown location kind: " << kind;
}

LogicalResult mlir::Neptune::NeptuneIR::LayoutAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, StringAttr order,
    DenseI64ArrayAttr strides, DenseI64ArrayAttr halo,
    DenseI64ArrayAttr offset) {
  if (!order)
    return emitError() << "layout order must be present";
  StringRef orderStr = order.getValue();
  if (orderStr != "xyz" && orderStr != "zyx")
    return emitError() << "layout order must be \"xyz\" or \"zyx\"";

  std::optional<size_t> refLen;
  auto checkLen = [&](DenseI64ArrayAttr arr, StringRef name) -> LogicalResult {
    if (!arr)
      return success();
    size_t len = arr.asArrayRef().size();
    if (!refLen) {
      refLen = len;
      return success();
    }
    if (*refLen != len)
      return emitError() << name << " length (" << len
                         << ") must match other layout arrays (" << *refLen
                         << ")";
    return success();
  };

  if (failed(checkLen(strides, "strides")) || failed(checkLen(halo, "halo")) ||
      failed(checkLen(offset, "offset")))
    return failure();

  if (strides) {
    for (int64_t v : strides.asArrayRef())
      if (v <= 0)
        return emitError() << "layout strides must be > 0";
  }
  if (halo) {
    for (int64_t v : halo.asArrayRef())
      if (v < 0)
        return emitError() << "layout halo must be >= 0";
  }
  if (offset) {
    for (int64_t v : offset.asArrayRef())
      if (v < 0)
        return emitError() << "layout offset must be >= 0";
  }

  return success();
}

LogicalResult mlir::Neptune::NeptuneIR::FieldType::verify(
    function_ref<InFlightDiagnostic()> emitError, Type elementType,
    mlir::Neptune::NeptuneIR::BoundsAttr bounds,
    mlir::Neptune::NeptuneIR::LocationAttr location,
    mlir::Neptune::NeptuneIR::LayoutAttr layout) {
  (void)location;

  if (!llvm::isa<FloatType>(elementType) &&
      !llvm::isa<IntegerType>(elementType))
    return emitError() << "element type must be integer or float";

  auto lbVals = bounds.getLb().asArrayRef();
  auto ubVals = bounds.getUb().asArrayRef();
  if (lbVals.size() != ubVals.size())
    return emitError() << "bounds lb/ub rank mismatch";
  if (lbVals.empty())
    return emitError() << "bounds rank must be > 0";

  size_t rank = lbVals.size();
  if (layout) {
    auto checkLen = [&](DenseI64ArrayAttr arr,
                        StringRef name) -> LogicalResult {
      if (!arr)
        return success();
      if (arr.asArrayRef().size() != rank)
        return emitError() << name << " length must match bounds rank";
      return success();
    };

    if (failed(checkLen(layout.getStrides(), "layout.strides")) ||
        failed(checkLen(layout.getHalo(), "layout.halo")) ||
        failed(checkLen(layout.getOffset(), "layout.offset")))
      return failure();
  }
  return success();
}

LogicalResult mlir::Neptune::NeptuneIR::TempType::verify(
    function_ref<InFlightDiagnostic()> emitError, Type elementType,
    mlir::Neptune::NeptuneIR::BoundsAttr bounds,
    mlir::Neptune::NeptuneIR::LocationAttr location) {
  (void)location;

  if (!llvm::isa<FloatType>(elementType) &&
      !llvm::isa<IntegerType>(elementType))
    return emitError() << "element type must be integer or float";

  auto lbVals = bounds.getLb().asArrayRef();
  auto ubVals = bounds.getUb().asArrayRef();
  if (lbVals.size() != ubVals.size())
    return emitError() << "bounds lb/ub rank mismatch";
  if (lbVals.empty())
    return emitError() << "bounds rank must be > 0";
  return success();
}


LogicalResult WrapOp::verify() {
  auto memrefTy = dyn_cast<MemRefType>(getBuffer().getType());
  if (!memrefTy)
    return emitOpError("buffer must be a ranked memref");

  auto fieldTy = getVarField().getType();
  if (memrefTy.getElementType() != fieldTy.getElementType())
    return emitOpError("buffer element type must match field element type");

  if (memrefTy.getRank() !=
      static_cast<int64_t>(fieldTy.getBounds().getLb().size()))
    return emitOpError("buffer rank must match field bounds rank");
  return success();
}

LogicalResult UnwrapOp::verify() {
  auto memrefTy = dyn_cast<MemRefType>(getBuffer().getType());
  if (!memrefTy)
    return emitOpError("buffer must be a ranked memref");

  auto fieldTy = getVarField().getType();
  if (memrefTy.getElementType() != fieldTy.getElementType())
    return emitOpError("buffer element type must match field element type");

  if (memrefTy.getRank() !=
      static_cast<int64_t>(fieldTy.getBounds().getLb().size()))
    return emitOpError("buffer rank must match field bounds rank");
  return success();
}

LogicalResult LoadOp::verify() {
  auto fieldTy = getVarField().getType();
  auto tempTy = getResult().getType();
  if (fieldTy.getElementType() != tempTy.getElementType())
    return emitOpError("field/temp element type mismatch");
  if (fieldTy.getBounds() != tempTy.getBounds())
    return emitOpError("field/temp bounds mismatch");
  if (fieldTy.getLocation() != tempTy.getLocation())
    return emitOpError("field/temp location mismatch");
  return success();
}

LogicalResult AccessOp::verify() {
  auto tempTy = getInput().getType();
  auto offsets = getOffsetsAttr().asArrayRef();
  size_t rank = tempTy.getBounds().getLb().size();
  if (offsets.size() != rank)
    return emitOpError("offsets rank must match temp bounds rank");
  if (getResult().getType() != tempTy.getElementType())
    return emitOpError("result type must match temp element type");
  return success();
}

LogicalResult ApplyOp::verify() {
  auto bounds = getBounds();
  auto applyLb = bounds.getLb().asArrayRef();
  auto applyUb = bounds.getUb().asArrayRef();
  unsigned rank = applyLb.size();
  if (rank == 0)
    return emitOpError("0-D apply not supported");
  if (applyUb.size() != rank)
    return emitOpError("bounds lb/ub rank mismatch");

  if (getInputs().empty())
    return emitOpError("apply requires at least one input");

  auto firstInputTy = dyn_cast<TempType>(getInputs()[0].getType());
  if (!firstInputTy)
    return emitOpError("inputs must be TempType");

  auto resTy = dyn_cast<TempType>(getResult().getType());
  if (!resTy)
    return emitOpError("result must be TempType");

  for (Value v : getInputs()) {
    auto ty = dyn_cast<TempType>(v.getType());
    if (!ty)
      return emitOpError("inputs must be TempType");
    if (ty.getElementType() != firstInputTy.getElementType() ||
        ty.getBounds() != firstInputTy.getBounds() ||
        ty.getLocation() != firstInputTy.getLocation())
      return emitOpError(
          "all inputs must have the same element/bounds/location");
  }

  if (resTy.getElementType() != firstInputTy.getElementType() ||
      resTy.getBounds() != firstInputTy.getBounds() ||
      resTy.getLocation() != firstInputTy.getLocation())
    return emitOpError("result must match input element/bounds/location");

  auto inputBounds = firstInputTy.getBounds();
  if (failed(detail::verifySubBounds(bounds, inputBounds,
                                     [&]() { return emitOpError(); })))
    return failure();

  auto radiusOpt = getRadius();
  if (radiusOpt) {
    auto radius = *radiusOpt;
    if (radius.size() != rank)
      return emitOpError("radius rank must match bounds rank");
    for (int64_t r : radius)
      if (r < 0)
        return emitOpError("radius values must be >= 0");

    auto inLb = inputBounds.getLb().asArrayRef();
    auto inUb = inputBounds.getUb().asArrayRef();
    for (unsigned d = 0; d < rank; ++d) {
      if (applyLb[d] < inLb[d] + radius[d] ||
          applyUb[d] > inUb[d] - radius[d]) {
        return emitOpError("apply bounds must be shrunk by radius at dim ")
               << d;
      }
    }
  }

  auto shapeOpt = getShape();
  if (shapeOpt) {
    ArrayAttr offsets = shapeOpt->getOffsets();
    for (Attribute attr : offsets) {
      auto off = dyn_cast<DenseI64ArrayAttr>(attr);
      if (!off)
        return emitOpError("stencil_shape offsets must be dense_i64_array");
      auto vals = off.asArrayRef();
      if (vals.size() != rank)
        return emitOpError("stencil_shape offset rank must match bounds rank");
      if (radiusOpt) {
        if (failed(detail::verifyOffsetsWithinRadius(
                vals, *radiusOpt, [&]() { return emitOpError(); })))
          return failure();
      }
    }
  }

  Block &body = getBody().front();
  if (body.getNumArguments() != rank)
    return emitOpError("apply region must have ") << rank << " index arguments";
  for (unsigned d = 0; d < rank; ++d) {
    if (!body.getArgument(d).getType().isIndex())
      return emitOpError("region arg #") << d << " must be index";
  }

  if (radiusOpt) {
    LogicalResult result = success();
    body.walk([&](AccessOp acc) {
      auto offs = acc.getOffsetsAttr().asArrayRef();
      if (failed(detail::verifyOffsetsWithinRadius(
              offs, *radiusOpt, [&]() { return acc.emitOpError(); })))
        result = failure();
    });
    if (failed(result))
      return failure();
  }

  auto *terminator = body.getTerminator();
  auto yield = dyn_cast<YieldOp>(terminator);
  if (!yield)
    return emitOpError("apply region must terminate with neptune_ir.yield");
  if (yield.getNumOperands() != 1)
    return emitOpError("apply region must yield a single scalar");
  if (yield.getOperand(0).getType() != firstInputTy.getElementType())
    return emitOpError("yield type must match input element type");

  return success();
}

LogicalResult StoreOp::verify() {
  auto tempTy = getValue().getType();
  auto fieldTy = getVarField().getType();
  if (tempTy.getElementType() != fieldTy.getElementType())
    return emitOpError("temp/field element type mismatch");
  if (tempTy.getLocation() != fieldTy.getLocation())
    return emitOpError("temp/field location mismatch");

  auto tempBounds = tempTy.getBounds();
  auto fieldBounds = fieldTy.getBounds();
  if (tempBounds.getLb().size() != fieldBounds.getLb().size())
    return emitOpError("temp/field bounds rank mismatch");

  if (auto bounds = getBounds()) {
    if (failed(detail::verifySubBounds(*bounds, fieldBounds,
                                       [&]() { return emitOpError(); })))
      return failure();
    if (failed(detail::verifySubBounds(*bounds, tempBounds,
                                       [&]() { return emitOpError(); })))
      return failure();
  } else if (tempBounds != fieldBounds) {
    return emitOpError("store without bounds requires temp/field bounds match");
  }
  return success();
}

LogicalResult ReduceOp::verify() { return success(); }

void StoreOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // 最保守：写目标 field（按 pointer 语义看待）
  OpOperand &fieldOperand = getOperation()->getOpOperand(1);
  effects.emplace_back(MemoryEffects::Write::get(), &fieldOperand);
}
