add_mlir_library(MLIRNeptuneIRPassesPipeline
    NeptuneIRPassesPipeline.cpp
    DEPENDS MLIRNeptuneIRTransforms

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
