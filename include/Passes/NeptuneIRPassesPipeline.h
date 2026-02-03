// Pass pipeline registration declarations.
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
// 如果你要在同一个二进制里跑 convert passes：
#include "mlir/Dialect/EmitC/Transforms/Passes.h"


#include "Dialect/NeptuneIR/NeptuneIRAttrs.h"
#include "Dialect/NeptuneIR/NeptuneIRDialect.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include "Passes/NeptuneIRPasses.h"
#include "mlir/Transforms/Passes.h"


namespace mlir::Neptune::NeptuneIR {
// 构建完整的 Neptune -> LLVM 降级流水线
void buildNeptuneToLLVMPipeline(mlir::OpPassManager &pm);

// 注册 Pipeline 到全局注册表 (供 neptune-opt 使用)
void registerNeptunePipelines();
} // namespace mlir::Neptune::NeptuneIR
// Declares pass pipeline builders for NeptuneIR.
