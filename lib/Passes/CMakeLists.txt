add_mlir_library(MLIRNeptuneIRTransforms
    NeptuneIRVerifyForms.cpp
    NeptuneIRNormalizeApply.cpp
    NeptuneIRSplitDomain.cpp
    NeptuneIREmitCHalide.cpp
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
