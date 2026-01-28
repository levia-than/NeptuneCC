add_library(NeptuneFrontendClang
  NeptunePragma.cpp
  NeptuneBinder.cpp
  NeptuneGlueGen.cpp
  NeptuneManifest.cpp
  NeptuneMLIRGen.cpp
)

target_include_directories(NeptuneFrontendClang PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(NeptuneFrontendClang PUBLIC
  MLIRIR
  MLIRSupport
  MLIRPass
  MLIRTargetCpp
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRNeptuneIRTransforms
  MLIRNeptuneIR
)

include(AddLLVM)  # 提供 llvm_update_compile_flags

llvm_update_compile_flags(NeptuneFrontendClang)
