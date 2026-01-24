add_executable(neptune-cc main.cpp)

target_link_libraries(neptune-cc PRIVATE
  NeptuneFrontendClang
  LLVMSupport
  LLVMOption  
  
  clangTooling
  clangFrontend
  clangAST
  clangLex
  clangBasic
  clangParse
  clangSema
  clangSerialization
  clangDriver
  clangAnalysis
  clangEdit 
)

include(AddLLVM)  # 提供 llvm_update_compile_flags

llvm_update_compile_flags(neptune-cc)

install(TARGETS neptune-cc
        RUNTIME DESTINATION bin)