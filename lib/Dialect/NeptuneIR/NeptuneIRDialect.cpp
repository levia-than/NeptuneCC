// NeptuneIR dialect registration.
//===- NeptuneIROps.cpp - NeptuneIR ops verifier implementations -------===//
//
// Verifier implementations for NeptuneIR ops.
//
// Replace or add this file in your NeptuneIR library. It intentionally
// minimizes reliance on ODS-generated accessor names and uses generic
// MLIR APIs so it is robust across MLIR versions / small ODS differences.
//
//===----------------------------------------------------------------------===//

#include "Dialect/NeptuneIR/NeptuneIRAttrs.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
// Defines the NeptuneIR dialect and registers ops/attrs.
#include "Dialect/NeptuneIR/NeptuneIRDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect/NeptuneIR/NeptuneIROpsEnumDefs.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/NeptuneIR/NeptuneIROpsAttrDefs.cpp.inc"

// Types implementation (TypeIDResolver etc.)
#define GET_TYPEDEF_CLASSES
#include "Dialect/NeptuneIR/NeptuneIROpsTypes.cpp.inc"

// Ops implementation (print/parse/verify/getEffects, bytecode support ...)
#define GET_OP_CLASSES
#include "Dialect/NeptuneIR/NeptuneIROps.cpp.inc"

// Dialect-level implementation (parse/print helpers, dialect ctor if generated)
#include "Dialect/NeptuneIR/NeptuneIROpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::Neptune::NeptuneIR;

void NeptuneIRDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/NeptuneIR/NeptuneIROpsAttrDefs.cpp.inc"
      >();

  // Register types (this uses the GET_TYPEDEF_LIST part of the generated file)
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/NeptuneIR/NeptuneIROpsTypes.cpp.inc"
      >();

  // Register ops
  addOperations<
#define GET_OP_LIST
#include "Dialect/NeptuneIR/NeptuneIROps.cpp.inc"
      >();
}
