set(NEPTUNECC_GENERATED_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(NEPTUNECC_HALIDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../halide")

add_library(neptunecc_glue STATIC
  "${NEPTUNECC_GENERATED_DIR}/neptunecc_kernels.cpp"
)

target_include_directories(neptunecc_glue PUBLIC
  "${NEPTUNECC_GENERATED_DIR}"
  "${NEPTUNECC_HALIDE_DIR}"
)

# Halide kernel libraries (AOT)
add_library(neptunecc_k0 STATIC IMPORTED GLOBAL)
set(_neptunecc_k0_lib "${NEPTUNECC_HALIDE_DIR}/libk0.a")
if(NOT EXISTS "${_neptunecc_k0_lib}")
  set(_neptunecc_k0_lib "${NEPTUNECC_HALIDE_DIR}/k0.a")
endif()
set_target_properties(neptunecc_k0 PROPERTIES
  IMPORTED_LOCATION "${_neptunecc_k0_lib}"
)
target_link_libraries(neptunecc_glue PUBLIC neptunecc_k0)

# If needed, link Halide runtime here.
