#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# 配置区域 (Defaults)
# ==============================================================================
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${REPO_ROOT}/build"

# LLVM 配置
LLVM_TAG="llvmorg-21.0.0"
LLVM_PROJECTS="mlir;clang;clang-tools-extra"
LLVM_TARGETS="X86"
LLVM_BUILD_DIR="${BUILD_ROOT}/llvm-build"
LLVM_INSTALL_DIR="${BUILD_ROOT}/llvm-install"

# 项目配置
PROJECT_BUILD_TYPE="Release"
PROJECT_BUILD_DIR="${BUILD_ROOT}/project-build"
INSTALL_PREFIX="${BUILD_ROOT}/project-build"

# 并行度 & 缓存
NINJA_JOBS="${NINJA_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
CCACHE_DIR="${HOME}/.ccache_neptune"
export CCACHE_DIR
export CCACHE_MAXSIZE="20G"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[Neptune Build]${NC} $1"; }
warn() { echo -e "${YELLOW}[Warning]${NC} $1"; }

# ==============================================================================
# 参数解析
# ==============================================================================
CLEAN=0
REBUILD_LLVM=0
ENABLE_PETSC=OFF
PETSC_DIR=""
PETSC_ARCH=""
MODE="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--clean) CLEAN=1; shift ;;
        --rebuild-llvm) REBUILD_LLVM=1; shift ;;
        --enable-petsc) ENABLE_PETSC=ON; shift ;;
        --petsc-dir) PETSC_DIR="$2"; shift 2 ;;
        --petsc-arch) PETSC_ARCH="$2"; shift 2 ;;
        -m|--mode) MODE="$2"; shift 2 ;;
        --project-debug) PROJECT_BUILD_TYPE="Debug"; shift ;;
        --project-release) PROJECT_BUILD_TYPE="Release"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ "$CLEAN" -eq 1 ]; then
    warn "Cleaning build root: ${BUILD_ROOT}"
    rm -rf "${BUILD_ROOT}"
    exit 0
fi

# ==============================================================================
# 1. 环境准备 (Compiler & Ccache)
# ==============================================================================
mkdir -p "${BUILD_ROOT}"
mkdir -p "${CCACHE_DIR}"

# 探测基础编译器
RAW_CC="gcc"
RAW_CXX="g++"
if command -v clang >/dev/null 2>&1; then
    RAW_CC="clang"
    RAW_CXX="clang++"
fi

# 启用 CCACHE (劫持编译器变量)
# 这是确保所有子进程(包括 PETSc configure)都能用上 ccache 的最强手段
USE_CCACHE=OFF
if command -v ccache >/dev/null 2>&1; then
    USE_CCACHE=ON
    log "Enabling ccache globally..."
    
    # 设置 CMake Launcher (给 CMake 项目用)
    CMAKE_CCACHE_ARGS=(
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache"
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
    )
    
    # 设置 Shell 变量 (给 configure 脚本用)
    export CC="ccache ${RAW_CC}"
    export CXX="ccache ${RAW_CXX}"
else
    CMAKE_CCACHE_ARGS=()
    export CC="${RAW_CC}"
    export CXX="${RAW_CXX}"
fi

# 检测 LLD
USE_LLD=OFF
if command -v lld >/dev/null 2>&1; then USE_LLD=ON; fi

log "Compiler: ${CC} / ${CXX}"
log "Jobs: ${NINJA_JOBS}, Linker: lld=${USE_LLD}"

# ==============================================================================
# 2. 构建 LLVM (Lazy Build)
# ==============================================================================
build_llvm() {
    if [ -f "${LLVM_INSTALL_DIR}/bin/mlir-tblgen" ] && [ "$REBUILD_LLVM" -eq 0 ]; then
        log "LLVM found in ${LLVM_INSTALL_DIR}. Skipping rebuild."
        return
    fi

    log "Building LLVM (${LLVM_TAG})..."
    local LLVM_SRC="${REPO_ROOT}/third_party/llvm-project"
    if [ ! -d "${LLVM_SRC}" ]; then
        mkdir -p "${REPO_ROOT}/third_party"
        git clone --depth 1 --branch "${LLVM_TAG}" https://github.com/llvm/llvm-project.git "${LLVM_SRC}"
    fi

    # 注意：LLVM 自己有 LLVM_CCACHE_BUILD 选项，但也吃 CMAKE_XXX_LAUNCHER
    cmake -G Ninja -S "${LLVM_SRC}/llvm" -B "${LLVM_BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL_DIR}" \
        -DLLVM_ENABLE_PROJECTS="${LLVM_PROJECTS}" \
        -DLLVM_TARGETS_TO_BUILD="${LLVM_TARGETS}" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_LLD="${USE_LLD}" \
        -DLLVM_CCACHE_BUILD=ON \
        -DLLVM_INCLUDE_TESTS=OFF \
        "${CMAKE_CCACHE_ARGS[@]}"

    cmake --build "${LLVM_BUILD_DIR}" --target install -j"${NINJA_JOBS}"
}

# ==============================================================================
# 3. 准备 PETSc (集成 Ccache & MPICH)
# ==============================================================================
setup_petsc() {
    if [ "$ENABLE_PETSC" != "ON" ]; then return; fi

    if [ -z "${PETSC_DIR}" ]; then
        local PETSC_VERSION="v3.20.6"
        local PETSC_SRC="${BUILD_ROOT}/petsc-src"
        local PETSC_INSTALL="${BUILD_ROOT}/petsc-install"

        if [ -f "${PETSC_INSTALL}/lib/libpetsc.so" ]; then
            log "Using bootstrapped PETSc at ${PETSC_INSTALL}"
            PETSC_DIR="${PETSC_INSTALL}"
            return
        fi

        log "Bootstrapping PETSc ${PETSC_VERSION}..."
        if [ ! -d "${PETSC_SRC}" ]; then
            log "Cloning PETSc tag ${PETSC_VERSION}..."
            git clone --depth 1 --branch "${PETSC_VERSION}" https://gitlab.com/petsc/petsc.git "${PETSC_SRC}"
        fi

        pushd "${PETSC_SRC}" > /dev/null
        
        # 因为我们在脚本开头已经 export CC="ccache clang"
        # 所以这里直接传 $CC 就行了，PETSc 会识别带空格的编译器命令
        ./configure \
            --prefix="${PETSC_INSTALL}" \
            --with-cc="${CC}" \
            --with-cxx="${CXX}" \
            --with-fc=0 \
            --download-mpich=1 \
            --with-debugging=0 \
            --with-shared-libraries=1 \
            --download-f2cblaslapack=1
            
        make -j"${NINJA_JOBS}"
        make install
        popd > /dev/null

        PETSC_DIR="${PETSC_INSTALL}"
    fi
}

# ==============================================================================
# 4. 构建 Neptune Project
# ==============================================================================
build_neptune() {
    log "Configuring Neptune..."
    
    local CMAKE_ARGS=(
        "-G" "Ninja"
        "-S" "${REPO_ROOT}"
        "-B" "${PROJECT_BUILD_DIR}"
        "-DCMAKE_BUILD_TYPE=${PROJECT_BUILD_TYPE}"
        "-DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}"
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
        "-DMLIR_DIR=${LLVM_INSTALL_DIR}/lib/cmake/mlir"
        "-DClang_DIR=${LLVM_INSTALL_DIR}/lib/cmake/clang"
        "-DLLVM_DIR=${LLVM_INSTALL_DIR}/lib/cmake/llvm"
        "-DLLVM_EXTERNAL_LIT=${LLVM_BUILD_DIR}/bin/llvm-lit"
        "-DNEPTUNE_ENABLE_PETSC=${ENABLE_PETSC}"
        "${CMAKE_CCACHE_ARGS[@]}" # 自动加上 launcher
    )

    if [ -n "${PETSC_DIR}" ]; then
        CMAKE_ARGS+=("-DPETSC_DIR=${PETSC_DIR}")
    fi
    if [ -n "${PETSC_ARCH}" ]; then
        CMAKE_ARGS+=("-DPETSC_ARCH=${PETSC_ARCH}")
    fi

    cmake "${CMAKE_ARGS[@]}"

    log "Building Neptune..."
    cmake --build "${PROJECT_BUILD_DIR}" --target install -j"${NINJA_JOBS}"

    ln -sf "${PROJECT_BUILD_DIR}/compile_commands.json" "${REPO_ROOT}/compile_commands.json"

    log "Running Tests..."
    cmake --build "${PROJECT_BUILD_DIR}" --target check-neptune || warn "Tests failed!"
}

# ==============================================================================
# Main
# ==============================================================================
build_llvm
setup_petsc
build_neptune

log "Build Complete!"
echo "=================================================================="
echo "=================================================================="
