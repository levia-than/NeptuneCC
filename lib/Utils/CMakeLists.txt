add_library(NeptuneUtils
  NeptuneScheduleDebug.cpp
)

target_include_directories(NeptuneUtils PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(NeptuneUtils PUBLIC
  LLVMSupport
)

include(AddLLVM)
llvm_update_compile_flags(NeptuneUtils)
