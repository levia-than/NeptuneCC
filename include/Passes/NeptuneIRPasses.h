// Pass creation declarations for NeptuneIR.
//===- LowerEvaluateToLoops.h - pass declaration ---------------*- C++ -*-===//
//
// LowerEvaluateToLoops pass: skeleton that finds evaluate ops and lowers them.
//
//===----------------------------------------------------------------------===//
#ifndef NEPTUNEIR_LOWEREVALUATETOLOOPS_H
#define NEPTUNEIR_LOWEREVALUATETOLOOPS_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "Dialect/NeptuneIR/NeptuneIRAttrs.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "Dialect/NeptuneIR/NeptuneIRDialect.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include <memory>

namespace mlir::Neptune::NeptuneIR {
    
#define GEN_PASS_DECL
#include "Passes/NeptuneIRPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Passes/NeptuneIRPasses.h.inc"

} // namespace mlir::Neptune::NeptuneIR

#endif // NEPTUNEIR_LOWEREVALUATETOLOOPS_H
