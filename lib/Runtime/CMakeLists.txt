add_library(NeptuneHaloRuntime STATIC
  NeptuneHaloRuntime.cpp
)

target_link_libraries(NeptuneHaloRuntime PRIVATE LLVMSupport)
