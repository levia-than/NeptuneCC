add_mlir_library(MLIRNeptuneIRTransforms
    NeptuneIRVerifyForms.cpp
    NeptuneIRNormalizeApply.cpp
    NeptuneIRSplitDomain.cpp
    NeptuneIREmitCHalide.cpp
    NeptuneIRInferHalideSchedule.cpp
    SCFToNeptuneIR.cpp
    SCFBoundarySimplify.cpp
    DEPENDS MLIRNeptuneIRTransformsIncGen

    LINK_LIBS MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRMemRefDialect
    MLIRLLVMDialect
)
