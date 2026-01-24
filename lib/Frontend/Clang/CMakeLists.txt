add_library(NeptuneFrontendClang
  NeptunePragma.cpp
  NeptuneBinder.cpp
  NeptuneManifest.cpp
)

target_include_directories(NeptuneFrontendClang PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

include(AddLLVM)  # 提供 llvm_update_compile_flags

llvm_update_compile_flags(NeptuneFrontendClang)
