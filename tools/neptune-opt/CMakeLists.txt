set(LIBS
    MLIROptLib
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRParser
    MLIRSupport
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRMemRefDialect
    MLIRCAPIRegisterEverything
    MLIRNeptuneIR
    MLIRNeptuneIRTransforms
    MLIRNeptuneIRPassesPipeline
)

add_llvm_executable(neptune-opt main.cpp
    PARTIAL_SOURCES_INTENDED)

llvm_update_compile_flags(neptune-opt)
target_link_libraries(neptune-opt PRIVATE ${LIBS})

install(TARGETS neptune-opt
        RUNTIME DESTINATION bin)
