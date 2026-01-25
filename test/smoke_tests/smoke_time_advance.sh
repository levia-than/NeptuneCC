#!/usr/bin/env bash
set -euo pipefail

# ---- Update these paths for your local build ----
NEPTUNE_OPT=${NEPTUNE_OPT:-/home/wyx/project/NeptuneCC/build/project-build/bin/neptune-opt}
MLIR_TRANSLATE=${MLIR_TRANSLATE:-/home/wyx/project/NeptuneCC/build/llvm-build/bin/mlir-translate}
LLVM_AS=${LLVM_AS:-/home/wyx/project/NeptuneCC/build/llvm-build/bin/llvm-as}
LLC=${LLC:-/home/wyx/project/NeptuneCC/build/llvm-build/bin/llc}
CLANG=${CLANG:-clang++}

# Runtime shared library
RUNTIME_SO=${RUNTIME_SO:-/home/wyx/project/NeptuneCC/build/project-build/lib/Runtime/PETSc/libNeptunePETScRuntime.so}

# Input MLIR
INPUT_MLIR=${1:-/home/wyx/project/NeptuneCC/test/smoke_tests/smoke_time_advance.mlir}

WORKDIR=${WORKDIR:-/tmp/neptune_smoke_time_advance_ac}
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "[1/5] Lower to LLVM dialect MLIR"
"$NEPTUNE_OPT" "$INPUT_MLIR" --neptuneir-to-llvm > out_llvm.mlir

echo "[2/5] Translate to LLVM IR"
"$MLIR_TRANSLATE" out_llvm.mlir --mlir-to-llvmir > out.ll

echo "[3/5] Build kernel object"
"$LLVM_AS" out.ll -o out.bc
"$LLC" out.bc -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o kernel.o

echo "[4/5] Generate tiny driver"
cat > driver.cpp <<'EOF'
// smoke_time_advance_ac_driver.cpp
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct MemRef1D {
  void*   allocated;
  void*   aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
};

// Note: signature must match the lowered LLVM function (10 args)
extern "C" MemRef1D entry(
    void* a_alloc, void* a_aligned, int64_t a_off, int64_t a_s0, int64_t a_st0,
    void* b_alloc, void* b_aligned, int64_t b_off, int64_t b_s0, int64_t b_st0);

static void* aligned_malloc(size_t align, size_t bytes) {
  void* p = nullptr;
#if defined(_ISOC11_SOURCE)
  p = aligned_alloc(align, ((bytes + align - 1) / align) * align);
  return p;
#else
  if (posix_memalign(&p, align, bytes) != 0) return nullptr;
  return p;
#endif
}

int main() {
  const int64_t n = 16;

  auto* out = (double*)aligned_malloc(64, sizeof(double) * (size_t)n);
  auto* rhs = (double*)aligned_malloc(64, sizeof(double) * (size_t)n);
  if (!out || !rhs) { std::puts("alloc failed"); return 1; }

  for (int i = 0; i < n; ++i) { out[i] = 0.0; rhs[i] = (double)(i + 1); }

  MemRef1D r = entry(
      out, out, 0, n, 1,
      rhs, rhs, 0, n, 1);

  std::printf("[driver] ret aligned=%p size=%ld stride=%ld off=%ld\n",
              r.aligned, (long)r.sizes[0], (long)r.strides[0], (long)r.offset);

  double* x = (double*)r.aligned;
  for (int i = 0; i < (int)r.sizes[0]; ++i) std::printf("x[%d]=%.6f\n", i, x[i]);

  return 0;
}
EOF

echo "[5/5] Link test exe"
RUNTIME_DIR=$(dirname "$RUNTIME_SO")
"$CLANG" -O2 -c driver.cpp -o driver.o
"$CLANG" -O2 kernel.o driver.o \
  -L"$RUNTIME_DIR" -lNeptunePETScRuntime \
  -Wl,-rpath,"$RUNTIME_DIR" \
  -rdynamic -ldl -o smoke_time_advance_ac_test

echo "[RUN] ./smoke_time_advance_ac_test"
./smoke_time_advance_ac_test
