#include "Passes/NeptuneIRPassesPipeline.h"

using namespace mlir;

void mlir::Neptune::NeptuneIR::buildNeptuneToLLVMPipeline(
    mlir::OpPassManager &pm) {
  using namespace mlir::Neptune::NeptuneIR;

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createCanonicalizerPass());

  // Lower memref ops (e.g. subview) before finalizing descriptors.
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createCanonicalizerPass());

  pm.addPass(createReconcileUnrealizedCastsPass());

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createSCCPPass());
  pm.addPass(createCSEPass());
  pm.addPass(createSymbolDCEPass());
}

void mlir::Neptune::NeptuneIR::registerNeptunePipelines() {
  PassPipelineRegistration<>(
      "neptuneir-to-llvm", "Run passes to lower the NeptuneIR to LLVM IR.",
      mlir::Neptune::NeptuneIR::buildNeptuneToLLVMPipeline);
}