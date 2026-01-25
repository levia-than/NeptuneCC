cmake_minimum_required(VERSION 3.20)
project(NeptuneCC LANGUAGES CXX C)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

set(NEPTUNE_ENABLE_PETSC OFF CACHE BOOL "Enable PETSc-based solver integration")

if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()

if(CMAKE_GENERATOR STREQUAL "Ninja")
  set(CMAKE_NINJA_FORCE_RESPONSE_FILE ON CACHE BOOL "" FORCE)
endif()

enable_testing()

find_package(LLVM REQUIRED CONFIG)

# Prefer Clang/MLIR from the same LLVM prefix to avoid mixed installations.
if(DEFINED LLVM_CMAKE_DIR)
  get_filename_component(_llvm_cmake_root "${LLVM_CMAKE_DIR}" DIRECTORY)
  if(NOT Clang_DIR AND EXISTS "${_llvm_cmake_root}/clang/ClangConfig.cmake")
    set(Clang_DIR "${_llvm_cmake_root}/clang")
  endif()
  if(NOT MLIR_DIR AND EXISTS "${_llvm_cmake_root}/mlir/MLIRConfig.cmake")
    set(MLIR_DIR "${_llvm_cmake_root}/mlir")
  endif()
endif()

find_package(Clang REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

if(NEPTUNE_ENABLE_PETSC)
  find_package(PETSc REQUIRED)
  set(NEPTUNE_PETSC_VERSION "${PETSC_VERSION}")
  message(STATUS "PETSc enabled (version: ${PETSC_VERSION})")
  add_executable(petsc_smoke cmake/petsc_smoke_test.c)
  target_link_libraries(petsc_smoke PRIVATE PETSc::PETSc)
else()
  set(NEPTUNE_PETSC_VERSION "OFF")
endif()

set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

# 3. 配置编译选项（与MLIR对齐）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(NEPTUNE_BINARY_DIR "${PROJECT_BINARY_DIR}")
set(NEPTUNE_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
add_compile_options(-Wall -Wextra -Wno-unused-parameter)

# 4. 配置头文件路径
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include  # 自定义头文件
  ${CMAKE_CURRENT_BINARY_DIR}/include  # 要导入诸如inc等文件
  ${LLVM_INCLUDE_DIRS}                 # LLVM头文件
  ${MLIR_INCLUDE_DIRS}                 # MLIR头文件
)

# 5. 链接LLVM/MLIR的CMake模块（生成TableGen代码必备）
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR} ${MLIR_CMAKE_DIR})
include(TableGen)          # MLIR TableGen工具
include(AddLLVM)           # LLVM编译辅助
include(AddMLIR)           # MLIR编译辅助
include(HandleLLVMOptions) # LLVM编译选项

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# 6. 添加自定义源码（Dialect/Pass）
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

# 7. 添加测试（可选，需启用BUILD_TESTING）
add_subdirectory(test)
