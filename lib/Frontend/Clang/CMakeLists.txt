add_library(NeptuneFrontendClang
  NeptunePragma.cpp
  NeptuneBinder.cpp
  NeptuneManifest.cpp
  NeptuneMLIRGen.cpp
)

target_include_directories(NeptuneFrontendClang PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(NeptuneFrontendClang PRIVATE
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRMemRefDialect
)

include(AddLLVM)  # 提供 llvm_update_compile_flags

llvm_update_compile_flags(NeptuneFrontendClang)
