/*
 * @Author: leviathan 670916484@qq.com
 * @Date: 2025-09-15 17:24:54
 * @LastEditors: leviathan 670916484@qq.com
 * @LastEditTime: 2025-11-29 12:19:11
 * @FilePath: /NeptuneCC/src/neptuneOpt.cpp
 * @Description: The main opt module.
 *
 * Copyright (c) 2025 by leviathan, All Rights Reserved.
 */
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

#include "Dialect/NeptuneIR/NeptuneIRAttrs.h"
#include "Dialect/NeptuneIR/NeptuneIRDialect.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include "Passes/NeptuneIRPasses.h"
#include "Passes/NeptuneIRPassesPipeline.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::Neptune::NeptuneIR;

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<NeptuneIRDialect>();
  mlir::registerAllPasses();
  Neptune::NeptuneIR::registerPasses();
  mlir::Neptune::NeptuneIR::registerNeptunePipelines(); 
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Neptune optimizer driver.\n", registry));
}
