#include "Runtime/PETSc/NeptunePETScRuntime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <petsc.h>
#include <petscksp.h>
#include <petscmat.h>
#include <string>
#include <vector>

// =============================================================================
// Helpers
// =============================================================================
static inline void neptunePetscCheck(PetscErrorCode ierr, const char *msg) {
  if (ierr) {
    std::fprintf(stderr, "[NeptuneRT][PETSc] %s (ierr=%d)\n", msg, (int)ierr);
    std::abort();
  }
}

static inline void *neptuneDlsymOrAbort(const char *sym) {
  void *p = dlsym(RTLD_DEFAULT, sym);
  if (!p) {
    std::fprintf(stderr, "[NeptuneRT] dlsym failed for symbol '%s'\n",
                 sym ? sym : "(null)");
    std::abort();
  }
  return p;
}

static inline NeptuneMemRef0D makeMemref0D(void *p) {
  NeptuneMemRef0D r{};
  r.allocated = p;
  r.aligned = p;
  r.offset = 0;
  return r;
}

static inline NeptuneMemRef1D makeMemref1D(void *alloc, void *aligned,
                                           int64_t off, int64_t n, int64_t st) {
  NeptuneMemRef1D r{};
  r.allocated = alloc;
  r.aligned = aligned;
  r.offset = off;
  r.sizes[0] = n;
  r.strides[0] = st;
  return r;
}

static inline NeptuneMemRef2D makeMemref2D(void *alloc, void *aligned,
                                           int64_t off, int64_t s0, int64_t s1,
                                           int64_t st0, int64_t st1) {
  NeptuneMemRef2D r{};
  r.allocated = alloc;
  r.aligned = aligned;
  r.offset = off;
  r.sizes[0] = s0;
  r.sizes[1] = s1;
  r.strides[0] = st0;
  r.strides[1] = st1;
  return r;
}

static inline size_t numel0D() { return 1; }

static inline size_t numel1D(int64_t n) { return (n <= 0) ? 0u : (size_t)n; }

static inline size_t numel2D(int64_t s0, int64_t s1) {
  if (s0 <= 0 || s1 <= 0)
    return 0u;
  return (size_t)s0 * (size_t)s1;
}

// =============================================================================
// MLIR memref ABI structs (match LLVM dialect lowering)
//   memref<1d>: {ptr, ptr, i64, [1], [1]}
//   memref<2d>: {ptr, ptr, i64, [2], [2]}
// =============================================================================
extern "C" {

// MLIR lowered linear operator: MemRef1D A(MemRef1D x)
using NeptuneLinearFnTy = NeptuneMemRef1D (*)(void *alloc, void *aligned,
                                              int64_t offset, int64_t size0,
                                              int64_t stride0);
} // extern "C"

// ============================================================================
// Global init/finalize
// ============================================================================
void neptune_rt_init_noargs() {
  PetscBool inited = PETSC_FALSE;
  neptunePetscCheck(PetscInitialized(&inited), "PetscInitialized failed");
  if (inited)
    return;
  neptunePetscCheck(PetscInitializeNoArguments(),
                    "PetscInitializeNoArguments failed");
}

// Forward decl for cleanup of assembled handles
struct NeptuneAssembledMatHandle;
static std::vector<NeptuneAssembledMatHandle *> &neptuneGlobalHandles() {
  static std::vector<NeptuneAssembledMatHandle *> handles;
  return handles;
}

// ============================================================================
// Linear solver ctx (KSP)  —— 现在支持两种模式：
//   1) DenseRM: 你原来的 MatDensePlaceArray / transpose
//   2) MatFreeShell: MatShell + MatMult 回调里调用 MLIR 的 @A
// ============================================================================
struct LinSolverCtx {
  enum class Mode { DenseRM, MatFreeShell };

  KSP ksp = nullptr;
  Mat A = nullptr;
  Vec b = nullptr;
  Vec x = nullptr;

  PetscOptions opts = nullptr;
  std::string prefix; // must end with '_'

  // dense scratch
  std::vector<double> A_cm;

  PetscInt n = -1;
  bool setup_called = false;
  Mode mode = Mode::DenseRM;

  // matfree state
  NeptuneLinearFnTy matfree_fn = nullptr;

  explicit LinSolverCtx(const char *name) {
    prefix = std::string(name ? name : "neptune") + "_";

    neptunePetscCheck(KSPCreate(PETSC_COMM_SELF, &ksp), "KSPCreate failed");

    // per-object options DB to avoid global pollution
    neptunePetscCheck(PetscOptionsCreate(&opts), "PetscOptionsCreate failed");
    neptunePetscCheck(PetscObjectSetOptions((PetscObject)ksp, opts),
                      "PetscObjectSetOptions(KSP) failed");
    neptunePetscCheck(KSPSetOptionsPrefix(ksp, prefix.c_str()),
                      "KSPSetOptionsPrefix failed");

    // sensible default (MVP)
    neptunePetscCheck(KSPSetType(ksp, KSPGMRES), "KSPSetType failed");
    PC pc = nullptr;
    neptunePetscCheck(KSPGetPC(ksp, &pc), "KSPGetPC failed");
    neptunePetscCheck(PCSetType(pc, PCNONE), "PCSetType failed");
  }

  ~LinSolverCtx() {
    if (x)
      VecDestroy(&x);
    if (b)
      VecDestroy(&b);
    if (A)
      MatDestroy(&A);
    if (ksp)
      KSPDestroy(&ksp);
    if (opts)
      PetscOptionsDestroy(&opts);
  }

  void destroyObjects() {
    if (x) {
      VecDestroy(&x);
      x = nullptr;
    }
    if (b) {
      VecDestroy(&b);
      b = nullptr;
    }
    if (A) {
      MatDestroy(&A);
      A = nullptr;
    }
    setup_called = false;
  }

  // ---- MatShell MatMult callback ----
  static PetscErrorCode MatMultThunk(Mat mat, Vec vx, Vec vy) {
    LinSolverCtx *self = nullptr;
    PetscErrorCode ierr = MatShellGetContext(mat, (void **)&self);
    if (ierr)
      return ierr;
    if (!self || !self->matfree_fn)
      return 0;

    const PetscScalar *x_ptr = nullptr;
    PetscScalar *y_ptr = nullptr;

    ierr = VecGetArrayRead(vx, &x_ptr);
    if (ierr)
      return ierr;
    ierr = VecGetArray(vy, &y_ptr);
    if (ierr) {
      VecRestoreArrayRead(vx, &x_ptr);
      return ierr;
    }

    // Build input memref view
    NeptuneMemRef1D xin;
    xin.allocated = (void *)x_ptr;
    xin.aligned = (void *)x_ptr;
    xin.offset = 0;
    xin.sizes[0] = (int64_t)self->n;
    xin.strides[0] = 1;

    // Call MLIR @A(x)
    NeptuneMemRef1D yout = self->matfree_fn(
        xin.allocated, xin.aligned, xin.offset, xin.sizes[0], xin.strides[0]);

    // Copy yout -> PETSc output vector
    const double *src = (const double *)yout.aligned + yout.offset;
    std::memcpy((void *)y_ptr, (const void *)src,
                sizeof(double) * (size_t)self->n);

    // IMPORTANT: your current MLIR-generated @A uses malloc for result; free it
    // to avoid leak.
    std::free(yout.allocated);

    ierr = VecRestoreArrayRead(vx, &x_ptr);
    if (ierr)
      return ierr;
    ierr = VecRestoreArray(vy, &y_ptr);
    if (ierr)
      return ierr;
    return 0;
  }

  void ensureSize(PetscInt newN, Mode newMode) {
    if (n == newN && mode == newMode && A && b && x)
      return;

    destroyObjects();

    n = newN;
    mode = newMode;

    if (mode == Mode::DenseRM) {
      A_cm.assign((size_t)n * (size_t)n, 0.0);

      neptunePetscCheck(MatCreateSeqDense(PETSC_COMM_SELF, n, n, nullptr, &A),
                        "MatCreateSeqDense failed");

      neptunePetscCheck(VecCreateSeq(PETSC_COMM_SELF, n, &b),
                        "VecCreateSeq failed");
      neptunePetscCheck(VecDuplicate(b, &x), "VecDuplicate failed");

      neptunePetscCheck(KSPSetOperators(ksp, A, A), "KSPSetOperators failed");
      return;
    }

    // MatFreeShell
    neptunePetscCheck(
        MatCreateShell(PETSC_COMM_SELF, n, n, n, n, (void *)this, &A),
        "MatCreateShell failed");
    neptunePetscCheck(
        MatShellSetOperation(A, MATOP_MULT,
                             (void (*)(void))&LinSolverCtx::MatMultThunk),
        "MatShellSetOperation(MATOP_MULT) failed");

    neptunePetscCheck(VecCreateSeq(PETSC_COMM_SELF, n, &b),
                      "VecCreateSeq failed");
    neptunePetscCheck(VecDuplicate(b, &x), "VecDuplicate failed");

    neptunePetscCheck(KSPSetOperators(ksp, A, A), "KSPSetOperators failed");
  }
};

neptune_lin_ctx_t neptune_lin_create(const char *name) {
  try {
    return new LinSolverCtx(name);
  } catch (...) {
    return nullptr;
  }
}

void neptune_lin_destroy(neptune_lin_ctx_t ctx) {
  delete static_cast<LinSolverCtx *>(ctx);
}

void neptune_lin_set_options(neptune_lin_ctx_t ctx, const char *options_str) {
  auto *s = static_cast<LinSolverCtx *>(ctx);
  if (!s || !options_str)
    return;

  neptunePetscCheck(PetscOptionsPrefixPush(s->opts, s->prefix.c_str()),
                    "PetscOptionsPrefixPush failed");
  neptunePetscCheck(PetscOptionsInsertString(s->opts, options_str),
                    "PetscOptionsInsertString failed");
  neptunePetscCheck(PetscOptionsPrefixPop(s->opts),
                    "PetscOptionsPrefixPop failed");

  neptunePetscCheck(KSPSetFromOptions(s->ksp), "KSPSetFromOptions failed");
  s->setup_called = false;
}

// row-major -> column-major copy
static inline void transposeRMtoCM(const double *A_rm, double *A_cm,
                                   PetscInt n) {
  for (PetscInt i = 0; i < n; ++i)
    for (PetscInt j = 0; j < n; ++j)
      A_cm[j * n + i] = A_rm[i * n + j];
}

void neptune_lin_solve_dense_rm(neptune_lin_ctx_t ctx, const double *A_rm,
                                const double *b_arr, double *x_arr,
                                int64_t n64) {
  auto *s = static_cast<LinSolverCtx *>(ctx);
  if (!s)
    return;

  if (n64 <= 0)
    return;
  PetscInt n = (PetscInt)n64;
  s->ensureSize(n, LinSolverCtx::Mode::DenseRM);

  transposeRMtoCM(A_rm, s->A_cm.data(), n);

  neptunePetscCheck(MatDensePlaceArray(s->A, s->A_cm.data()),
                    "MatDensePlaceArray failed");
  neptunePetscCheck(VecPlaceArray(s->b, b_arr), "VecPlaceArray(b) failed");
  neptunePetscCheck(VecPlaceArray(s->x, x_arr), "VecPlaceArray(x) failed");

  if (!s->setup_called) {
    neptunePetscCheck(KSPSetUp(s->ksp), "KSPSetUp failed");
    s->setup_called = true;
  }

  neptunePetscCheck(KSPSolve(s->ksp, s->b, s->x), "KSPSolve failed");

  KSPConvergedReason reason;
  PetscInt its = 0;
  PetscReal rnorm = 0.0;

  neptunePetscCheck(KSPGetConvergedReason(s->ksp, &reason),
                    "KSPGetConvergedReason failed");
  neptunePetscCheck(KSPGetIterationNumber(s->ksp, &its),
                    "KSPGetIterationNumber failed");
  neptunePetscCheck(KSPGetResidualNorm(s->ksp, &rnorm),
                    "KSPGetResidualNorm failed");

  neptunePetscCheck(
      PetscPrintf(PETSC_COMM_SELF, "===== KSP Solver Results =====\n"),
      "PetscPrintf failed");
  neptunePetscCheck(
      PetscPrintf(PETSC_COMM_SELF, "Iteration count: %d\n", (int)its),
      "PetscPrintf failed");
  neptunePetscCheck(PetscPrintf(PETSC_COMM_SELF, "Final residual norm: %.10e\n",
                                (double)rnorm),
                    "PetscPrintf failed");
  neptunePetscCheck(PetscPrintf(PETSC_COMM_SELF,
                                "Convergence reason enum: %d\n", (int)reason),
                    "PetscPrintf failed");

  neptunePetscCheck(KSPConvergedReasonView(s->ksp, PETSC_VIEWER_STDOUT_SELF),
                    "KSPConvergedReasonView failed");

  neptunePetscCheck(MatDenseResetArray(s->A), "MatDenseResetArray failed");
  neptunePetscCheck(VecResetArray(s->b), "VecResetArray(b) failed");
  neptunePetscCheck(VecResetArray(s->x), "VecResetArray(x) failed");
}

// NEW: matrix-free solve (MatShell) calling MLIR-generated @A
static void neptune_lin_solve_matfree_mlir(neptune_lin_ctx_t ctx,
                                           NeptuneLinearFnTy fn,
                                           const double *b_arr, double *x_arr,
                                           int64_t n64) {
  auto *s = static_cast<LinSolverCtx *>(ctx);
  if (!s || !fn)
    return;
  if (n64 <= 0)
    return;

  PetscInt n = (PetscInt)n64;
  s->matfree_fn = fn;
  s->ensureSize(n, LinSolverCtx::Mode::MatFreeShell);

  neptunePetscCheck(VecPlaceArray(s->b, b_arr), "VecPlaceArray(b) failed");
  neptunePetscCheck(VecPlaceArray(s->x, x_arr), "VecPlaceArray(x) failed");

  if (!s->setup_called) {
    neptunePetscCheck(KSPSetUp(s->ksp), "KSPSetUp failed");
    s->setup_called = true;
  }

  neptunePetscCheck(KSPSolve(s->ksp, s->b, s->x), "KSPSolve failed");

  KSPConvergedReason reason;
  PetscInt its = 0;
  PetscReal rnorm = 0.0;

  neptunePetscCheck(KSPGetConvergedReason(s->ksp, &reason),
                    "KSPGetConvergedReason failed");
  neptunePetscCheck(KSPGetIterationNumber(s->ksp, &its),
                    "KSPGetIterationNumber failed");
  neptunePetscCheck(KSPGetResidualNorm(s->ksp, &rnorm),
                    "KSPGetResidualNorm failed");

  neptunePetscCheck(
      PetscPrintf(PETSC_COMM_SELF, "===== KSP Solver Results =====\n"),
      "PetscPrintf failed");
  neptunePetscCheck(
      PetscPrintf(PETSC_COMM_SELF, "Iteration count: %d\n", (int)its),
      "PetscPrintf failed");
  neptunePetscCheck(PetscPrintf(PETSC_COMM_SELF, "Final residual norm: %.10e\n",
                                (double)rnorm),
                    "PetscPrintf failed");
  neptunePetscCheck(PetscPrintf(PETSC_COMM_SELF,
                                "Convergence reason enum: %d\n", (int)reason),
                    "PetscPrintf failed");

  neptunePetscCheck(KSPConvergedReasonView(s->ksp, PETSC_VIEWER_STDOUT_SELF),
                    "KSPConvergedReasonView failed");
  neptunePetscCheck(VecResetArray(s->b), "VecResetArray(b) failed");
  neptunePetscCheck(VecResetArray(s->x), "VecResetArray(x) failed");
}

// ============================================================================
// Nonlinear solver ctx (SNES) - residual-only JFNK style
// (你的原代码保持不动)
// ============================================================================
struct NonLinSolverCtx {
  SNES snes = nullptr;
  Vec x = nullptr;
  Vec f = nullptr;

  PetscOptions opts = nullptr;
  std::string prefix;

  ResidualFuncPtr user_func = nullptr;
  void *user_ctx = nullptr;

  PetscInt n = -1;

  explicit NonLinSolverCtx(const char *name) {
    prefix = std::string(name ? name : "neptune") + "_";

    neptunePetscCheck(SNESCreate(PETSC_COMM_SELF, &snes), "SNESCreate failed");

    neptunePetscCheck(PetscOptionsCreate(&opts), "PetscOptionsCreate failed");
    neptunePetscCheck(PetscObjectSetOptions((PetscObject)snes, opts),
                      "PetscObjectSetOptions(SNES) failed");
    neptunePetscCheck(SNESSetOptionsPrefix(snes, prefix.c_str()),
                      "SNESSetOptionsPrefix failed");

    neptunePetscCheck(SNESSetType(snes, SNESNEWTONLS), "SNESSetType failed");
    neptunePetscCheck(SNESSetUseMatrixFree(snes, PETSC_TRUE, PETSC_TRUE),
                      "SNESSetUseMatrixFree failed");
  }

  ~NonLinSolverCtx() {
    if (f)
      VecDestroy(&f);
    if (x)
      VecDestroy(&x);
    if (snes)
      SNESDestroy(&snes);
    if (opts)
      PetscOptionsDestroy(&opts);
  }

  void ensureSize(PetscInt newN) {
    if (n == newN && x && f)
      return;
    if (f) {
      VecDestroy(&f);
      f = nullptr;
    }
    if (x) {
      VecDestroy(&x);
      x = nullptr;
    }

    n = newN;
    neptunePetscCheck(VecCreateSeq(PETSC_COMM_SELF, n, &x),
                      "VecCreateSeq failed");
    neptunePetscCheck(VecDuplicate(x, &f), "VecDuplicate failed");
  }

  static PetscErrorCode FormFunctionWrapper(SNES, Vec x, Vec f, void *ctx) {
    auto *self = static_cast<NonLinSolverCtx *>(ctx);
    if (!self || !self->user_func)
      return 0;

    const PetscScalar *x_ptr = nullptr;
    PetscScalar *f_ptr = nullptr;

    PetscErrorCode ierr = 0;
    ierr = VecGetArrayRead(x, &x_ptr);
    if (ierr)
      return ierr;
    ierr = VecGetArray(f, &f_ptr);
    if (ierr) {
      VecRestoreArrayRead(x, &x_ptr);
      return ierr;
    }

    self->user_func(const_cast<double *>((const double *)x_ptr),
                    (double *)f_ptr, self->user_ctx);

    ierr = VecRestoreArrayRead(x, &x_ptr);
    if (ierr)
      return ierr;
    ierr = VecRestoreArray(f, &f_ptr);
    if (ierr)
      return ierr;
    return 0;
  }
};

neptune_nonlin_ctx_t neptune_nonlin_create(const char *name) {
  try {
    return new NonLinSolverCtx(name);
  } catch (...) {
    return nullptr;
  }
}

void neptune_nonlin_destroy(neptune_nonlin_ctx_t ctx) {
  delete static_cast<NonLinSolverCtx *>(ctx);
}

void neptune_nonlin_set_options(neptune_nonlin_ctx_t ctx,
                                const char *options_str) {
  auto *s = static_cast<NonLinSolverCtx *>(ctx);
  if (!s || !options_str)
    return;

  neptunePetscCheck(PetscOptionsPrefixPush(s->opts, s->prefix.c_str()),
                    "PetscOptionsPrefixPush failed");
  neptunePetscCheck(PetscOptionsInsertString(s->opts, options_str),
                    "PetscOptionsInsertString failed");
  neptunePetscCheck(PetscOptionsPrefixPop(s->opts),
                    "PetscOptionsPrefixPop failed");

  neptunePetscCheck(SNESSetFromOptions(s->snes), "SNESSetFromOptions failed");
}

void neptune_nonlin_solve(neptune_nonlin_ctx_t ctx,
                          ResidualFuncPtr residual_func, double *x_arr,
                          int64_t n64, void *user_ctx) {
  auto *s = static_cast<NonLinSolverCtx *>(ctx);
  if (!s || !residual_func)
    return;
  if (n64 <= 0)
    return;

  PetscInt n = (PetscInt)n64;
  s->ensureSize(n);

  s->user_func = residual_func;
  s->user_ctx = user_ctx;

  neptunePetscCheck(VecPlaceArray(s->x, x_arr), "VecPlaceArray(x) failed");

  neptunePetscCheck(
      SNESSetFunction(s->snes, s->f, &NonLinSolverCtx::FormFunctionWrapper, s),
      "SNESSetFunction failed");

  neptunePetscCheck(SNESSolve(s->snes, nullptr, s->x), "SNESSolve failed");

  neptunePetscCheck(VecResetArray(s->x), "VecResetArray(x) failed");
}

// ============================================================================
// NEW: ABI adapters for your LLVM output
//   - _neptune_rt_runtime_assemble_matrix(i64 keyHash, i8* symNamePtr) ->
//   memref<?x?xf64>
//   - _neptune_rt_runtime_solve_linear(expanded memrefs...) -> memref<?xf64>
// ============================================================================

struct NeptuneAssembledMatHandle {
  int64_t keyHash = 0;
  const char *symName = nullptr;   // points to global "A\00"
  NeptuneLinearFnTy fn = nullptr;  // resolved by dlsym
  neptune_lin_ctx_t lin = nullptr; // PETSc KSP ctx

  explicit NeptuneAssembledMatHandle(int64_t h, const char *n)
      : keyHash(h), symName(n) {
    // use symbol name as prefix for options (nice for -<prefix>ksp_type etc.)
    lin = neptune_lin_create(symName ? symName : "neptune");
    if (!lin) {
      std::fprintf(stderr, "[NeptuneRT] neptune_lin_create failed\n");
      std::abort();
    }
  }

  ~NeptuneAssembledMatHandle() {
    if (lin)
      neptune_lin_destroy(lin);
    lin = nullptr;
  }
};

void neptune_rt_finalize() {
  // free assembled handles first (so KSP/Mat/Vec destroyed before
  // PetscFinalize)
  for (auto *h : neptuneGlobalHandles()) {
    // handle deletion implemented later; safe even if nullptr
    delete h;
  }
  neptuneGlobalHandles().clear();

  PetscBool inited = PETSC_FALSE;
  neptunePetscCheck(PetscInitialized(&inited), "PetscInitialized failed");
  if (!inited)
    return;
  neptunePetscCheck(PetscFinalize(), "PetscFinalize failed");
}

extern "C" NeptuneMemRef2D
_neptune_rt_runtime_assemble_matrix(int64_t keyHash, const char *symName,
                                    int64_t n) {
  neptune_rt_init_noargs();

  auto *h = new NeptuneAssembledMatHandle(keyHash, symName);
  neptuneGlobalHandles().push_back(h); // cleanup at finalize

  NeptuneMemRef2D out{};
  out.allocated = (void *)h;
  out.aligned = (void *)h;
  out.offset = 0;
  out.sizes[0] = n;
  out.sizes[1] = n;
  out.strides[0] = 1;
  out.strides[1] = n;
  return out;
}

using NeptuneUnaryTempFnTy = NeptuneMemRef1D (*)(void *a_alloc, void *a_aligned,
                                                 int64_t a_off, int64_t a_s0,
                                                 int64_t a_st0);

extern "C" NeptuneMemRef1D
_neptune_rt_runtime_time_advance(void *s_alloc, void *s_aligned, int64_t s_off,
                                 int64_t s_s0, int64_t s_st0, double dt,
                                 int32_t method, const char *rhs_sym) {

  // 基本 sanity
  if (!s_aligned) {
    std::fprintf(stderr, "[NeptuneRT] time_advance got null state\n");
    std::abort();
  }
  if (s_s0 < 0) {
    std::fprintf(stderr, "[NeptuneRT] time_advance bad size s_s0=%ld\n",
                 (long)s_s0);
    std::abort();
  }

  // method=0: marker/no-op（直接返回原 memref）
  if (method == 0) {
    return makeMemref1D(s_alloc, s_aligned, s_off, s_s0, s_st0);
  }

  // 统一：输出都返回 contiguous（stride=1, offset=0）
  auto *out = (double *)std::malloc(sizeof(double) * (size_t)s_s0);
  if (!out) {
    std::fprintf(stderr, "[NeptuneRT] time_advance malloc failed (n=%ld)\n",
                 (long)s_s0);
    std::abort();
  }

  const double *sbase = (const double *)s_aligned;
  const double *s = sbase + s_off;

  // method=1: copy
  if (method == 1) {
    for (int64_t i = 0; i < s_s0; ++i) {
      out[i] = s[i * s_st0];
    }
    return makeMemref1D(out, out, /*off=*/0, s_s0, /*st=*/1);
  }

  // method=2: forward euler: out = s + dt * rhs(s)
  if (method == 2) {
    if (!rhs_sym || !rhs_sym[0]) {
      std::fprintf(stderr,
                   "[NeptuneRT] time_advance(method=2) requires rhs_sym\n");
      std::abort();
    }

    auto fn = (NeptuneUnaryTempFnTy)neptuneDlsymOrAbort(rhs_sym);

    // rhs(s) -> temp memref
    NeptuneMemRef1D k = fn(s_alloc, s_aligned, s_off, s_s0, s_st0);

    if (!k.aligned || k.sizes[0] != s_s0) {
      std::fprintf(stderr,
                   "[NeptuneRT] rhs returned invalid memref (aligned=%p n=%ld "
                   "expected=%ld)\n",
                   k.aligned, (long)k.sizes[0], (long)s_s0);
      std::abort();
    }

    const double *kbase = (const double *)k.aligned;
    const double *kk = kbase + k.offset;

    // 这里也按 stride 读 rhs，尽量别假设 contiguous
    for (int64_t i = 0; i < s_s0; ++i) {
      out[i] = s[i * s_st0] + dt * kk[i * k.strides[0]];
    }

    // 尝试释放 rhs 产生的 buffer（你现在 @A 是 malloc 出来的，free 没问题）
    // 如果你未来 rhs 不是 malloc，而是 runtime-managed，就要换成统一 free API。
    if (k.allocated)
      std::free(k.allocated);

    return makeMemref1D(out, out, /*off=*/0, s_s0, /*st=*/1);
  }

  std::fprintf(stderr, "[NeptuneRT] time_advance unknown method=%d\n",
               (int)method);
  std::abort();
}

extern "C" NeptuneMemRef1D
_neptune_rt_runtime_solve_linear(void *A_alloc, void *A_aligned, int64_t A_off,
                                 int64_t A_s0, int64_t A_s1, int64_t A_st0,
                                 int64_t A_st1, void *b_alloc, void *b_aligned,
                                 int64_t b_off, int64_t b_s0, int64_t b_st0) {
  std::fprintf(stderr,
               "[NeptuneRT] A(handle) A_alloc=%p A_aligned=%p A_off=%ld "
               "A_s=(%ld,%ld) A_st=(%ld,%ld)\n",
               A_alloc, A_aligned, (long)A_off, (long)A_s0, (long)A_s1,
               (long)A_st0, (long)A_st1);

  std::fprintf(stderr,
               "[NeptuneRT] b(memref) b_alloc=%p b_aligned=%p b_off=%ld "
               "b_s0=%ld b_st0=%ld\n",
               b_alloc, b_aligned, (long)b_off, (long)b_s0, (long)b_st0);
  neptune_rt_init_noargs();

  (void)A_alloc;
  (void)A_off;
  (void)A_s0;
  (void)A_s1;
  (void)A_st0;
  (void)A_st1;
  (void)b_alloc;
  (void)b_st0;

  auto *h = reinterpret_cast<NeptuneAssembledMatHandle *>(A_aligned);
  if (!h) {
    std::fprintf(stderr, "[NeptuneRT] solve_linear got null handle\n");
    std::abort();
  }

  // Resolve @A once
  if (!h->fn) {
    h->fn =
        reinterpret_cast<NeptuneLinearFnTy>(neptuneDlsymOrAbort(h->symName));
  }

  // RHS pointer (assume contiguous, stride=1 MVP)
  const double *rhs = (const double *)b_aligned + b_off;
  const int64_t n64 = b_s0;

  // Allocate output buffer (malloc) and solve into it

  if (n64 <= 0 || n64 > (1LL << 26)) { // 上限随便先定个安全值
    std::fprintf(stderr, "[NeptuneRT] bad n64=%ld (b_s0)\n", (long)n64);
    std::abort();
  }

  size_t bytes = (size_t)n64 * sizeof(double);
  std::fprintf(stderr, "[NeptuneRT] allocating solution bytes=%zu\n", bytes);
  double *xbuf = (double *)std::malloc(sizeof(double) * (size_t)n64);
  if (!xbuf) {
    std::fprintf(stderr, "[NeptuneRT] malloc failed for solution\n");
    std::abort();
  }
  std::memset(xbuf, 0, sizeof(double) * (size_t)n64);

  neptune_lin_solve_matfree_mlir(h->lin, h->fn, rhs, xbuf, n64);

  NeptuneMemRef1D out{};
  out.allocated = (void *)xbuf;
  out.aligned = (void *)xbuf;
  out.offset = 0;
  out.sizes[0] = n64;
  out.strides[0] = 1;
  return out;
}

// memref<1d> expanded ABI: (alloc, aligned, off, size0, stride0)
using NeptuneBinaryTempFnTy = NeptuneMemRef1D (*)(
    void *a_alloc, void *a_aligned, int64_t a_off, int64_t a_s0, int64_t a_st0,
    void *b_alloc, void *b_aligned, int64_t b_off, int64_t b_s0, int64_t b_st0);

struct NeptuneNonlinMlirCtx {
  NeptuneBinaryTempFnTy residual = nullptr;
  NeptuneMemRef1D u_prev; // capture
  int64_t n = 0;
};

static void neptune_mlir_residual_thunk(double *x_arr, double *f_arr,
                                        void *ctx) {
  auto *c = static_cast<NeptuneNonlinMlirCtx *>(ctx);
  if (!c || !c->residual)
    return;

  // x view (contiguous)
  NeptuneMemRef1D x{};
  x.allocated = x_arr;
  x.aligned = x_arr;
  x.offset = 0;
  x.sizes[0] = c->n;
  x.strides[0] = 1;

  // call residual(x, u_prev) -> memref
  NeptuneMemRef1D r =
      c->residual(x.allocated, x.aligned, x.offset, x.sizes[0], x.strides[0],
                  c->u_prev.allocated, c->u_prev.aligned, c->u_prev.offset,
                  c->u_prev.sizes[0], c->u_prev.strides[0]);

  if (!r.aligned || r.sizes[0] != c->n) {
    std::fprintf(stderr,
                 "[NeptuneRT] residual returned bad memref (aligned=%p n=%ld "
                 "expected=%ld)\n",
                 r.aligned, (long)r.sizes[0], (long)c->n);
    std::abort();
  }

  const double *src = (const double *)r.aligned + r.offset;
  // 注意 r.strides[0] 可能不是 1，保险起见按 stride 读
  for (int64_t i = 0; i < c->n; ++i)
    f_arr[i] = src[i * r.strides[0]];

  if (r.allocated)
    std::free(r.allocated); // 你现在 MLIR 侧 alloc 用 malloc
}

// Pack strided memref -> contiguous buffer (double)
// NOTE: We assume element type is f64 for nonlinear MVP. Extend if you need.
static inline void pack0D_to_contig(const NeptuneMemRef0D &m, double *out1) {
  const double *base = (const double *)m.aligned;
  out1[0] = base[m.offset];
}

static inline void pack1D_to_contig(const NeptuneMemRef1D &m, double *out) {
  const double *base = (const double *)m.aligned + m.offset;
  const int64_t n = m.sizes[0];
  const int64_t st = m.strides[0];
  for (int64_t i = 0; i < n; ++i)
    out[(size_t)i] = base[(size_t)i * (size_t)st];
}

static inline void pack2D_to_contig_rm(const NeptuneMemRef2D &m, double *out) {
  const double *base = (const double *)m.aligned + m.offset;
  const int64_t s0 = m.sizes[0], s1 = m.sizes[1];
  const int64_t st0 = m.strides[0], st1 = m.strides[1];
  size_t k = 0;
  for (int64_t i = 0; i < s0; ++i)
    for (int64_t j = 0; j < s1; ++j)
      out[k++] = base[(size_t)i * (size_t)st0 + (size_t)j * (size_t)st1];
}

// Build an "x view" memref over PETSc vector (contiguous)
// Rank-2 uses row-major contiguous layout: stride0 = s1, stride1 = 1
static inline NeptuneMemRef0D view_x0D(const double *xptr) {
  NeptuneMemRef0D r{};
  r.allocated = (void *)xptr;
  r.aligned = (void *)xptr;
  r.offset = 0;
  return r;
}

static inline NeptuneMemRef1D view_x1D(const double *xptr, int64_t n) {
  NeptuneMemRef1D r{};
  r.allocated = (void *)xptr;
  r.aligned = (void *)xptr;
  r.offset = 0;
  r.sizes[0] = n;
  r.strides[0] = 1;
  return r;
}

static inline NeptuneMemRef2D view_x2D(const double *xptr, int64_t s0,
                                       int64_t s1) {
  NeptuneMemRef2D r{};
  r.allocated = (void *)xptr;
  r.aligned = (void *)xptr;
  r.offset = 0;
  r.sizes[0] = s0;
  r.sizes[1] = s1;
  r.strides[0] = s1; // row-major
  r.strides[1] = 1;
  return r;
}

// =============================================================================
// SNES context templates
// =============================================================================
template <int Rank, int Caps> struct NL {};

template <> struct NL<0, 0> {
  using ResidualFnTy = NeptuneMemRef0D (*)(void *x_alloc, void *x_aligned,
                                           int64_t x_off);

  struct Ctx {
    static constexpr int kRank = 0;
    static constexpr int kCaps = 0;

    SNES snes = nullptr;
    Vec x = nullptr;
    Vec f = nullptr;
    PetscOptions opts = nullptr;
    std::string prefix = "neptune_nl_";
    ResidualFnTy fn = nullptr;
    bool setup_called = false;
  };

  static PetscErrorCode FormFunction(SNES, Vec vx, Vec vf, void *p) {
    auto *c = (Ctx *)p;
    if (!c || !c->fn)
      return 0;

    const PetscScalar *xptr = nullptr;
    PetscScalar *fptr = nullptr;
    PetscErrorCode ierr = VecGetArrayRead(vx, &xptr);
    if (ierr)
      return ierr;
    ierr = VecGetArray(vf, &fptr);
    if (ierr) {
      VecRestoreArrayRead(vx, &xptr);
      return ierr;
    }

    NeptuneMemRef0D xin = view_x0D((const double *)xptr);
    NeptuneMemRef0D fout = c->fn(xin.allocated, xin.aligned, xin.offset);

    const double *src = (const double *)fout.aligned;
    fptr[0] = src[fout.offset];

    if (fout.allocated)
      std::free(fout.allocated);

    ierr = VecRestoreArrayRead(vx, &xptr);
    if (ierr)
      return ierr;
    ierr = VecRestoreArray(vf, &fptr);
    if (ierr)
      return ierr;
    return 0;
  }
};

template <> struct NL<0, 1> {
  using ResidualFnTy = NeptuneMemRef0D (*)(void *x_alloc, void *x_aligned,
                                           int64_t x_off, void *c0_alloc,
                                           void *c0_aligned, int64_t c0_off);

  struct Ctx {
    static constexpr int kRank = 0;
    static constexpr int kCaps = 1;

    SNES snes = nullptr;
    Vec x = nullptr;
    Vec f = nullptr;
    PetscOptions opts = nullptr;
    std::string prefix = "neptune_nl_";
    ResidualFnTy fn = nullptr;
    NeptuneMemRef0D cap0{};
    bool setup_called = false;
  };

  static PetscErrorCode FormFunction(SNES, Vec vx, Vec vf, void *p) {
    auto *c = (Ctx *)p;
    if (!c || !c->fn)
      return 0;

    const PetscScalar *xptr = nullptr;
    PetscScalar *fptr = nullptr;
    PetscErrorCode ierr = VecGetArrayRead(vx, &xptr);
    if (ierr)
      return ierr;
    ierr = VecGetArray(vf, &fptr);
    if (ierr) {
      VecRestoreArrayRead(vx, &xptr);
      return ierr;
    }

    NeptuneMemRef0D xin = view_x0D((const double *)xptr);
    NeptuneMemRef0D fout =
        c->fn(xin.allocated, xin.aligned, xin.offset, c->cap0.allocated,
              c->cap0.aligned, c->cap0.offset);

    const double *src = (const double *)fout.aligned;
    fptr[0] = src[fout.offset];

    if (fout.allocated)
      std::free(fout.allocated);

    ierr = VecRestoreArrayRead(vx, &xptr);
    if (ierr)
      return ierr;
    ierr = VecRestoreArray(vf, &fptr);
    if (ierr)
      return ierr;
    return 0;
  }
};

template <> struct NL<0, 2> {
  using ResidualFnTy = NeptuneMemRef0D (*)(void *x_alloc, void *x_aligned,
                                           int64_t x_off, void *c0_alloc,
                                           void *c0_aligned, int64_t c0_off,
                                           void *c1_alloc, void *c1_aligned,
                                           int64_t c1_off);

  struct Ctx {
    static constexpr int kRank = 0;
    static constexpr int kCaps = 2;

    SNES snes = nullptr;
    Vec x = nullptr;
    Vec f = nullptr;
    PetscOptions opts = nullptr;
    std::string prefix = "neptune_nl_";
    ResidualFnTy fn = nullptr;
    NeptuneMemRef0D cap0{};
    NeptuneMemRef0D cap1{};
    bool setup_called = false;
  };

  static PetscErrorCode FormFunction(SNES, Vec vx, Vec vf, void *p) {
    auto *c = (Ctx *)p;
    if (!c || !c->fn)
      return 0;

    const PetscScalar *xptr = nullptr;
    PetscScalar *fptr = nullptr;
    PetscErrorCode ierr = VecGetArrayRead(vx, &xptr);
    if (ierr)
      return ierr;
    ierr = VecGetArray(vf, &fptr);
    if (ierr) {
      VecRestoreArrayRead(vx, &xptr);
      return ierr;
    }

    NeptuneMemRef0D xin = view_x0D((const double *)xptr);
    NeptuneMemRef0D fout =
        c->fn(xin.allocated, xin.aligned, xin.offset, c->cap0.allocated,
              c->cap0.aligned, c->cap0.offset, c->cap1.allocated,
              c->cap1.aligned, c->cap1.offset);

    const double *src = (const double *)fout.aligned;
    fptr[0] = src[fout.offset];

    if (fout.allocated)
      std::free(fout.allocated);

    ierr = VecRestoreArrayRead(vx, &xptr);
    if (ierr)
      return ierr;
    ierr = VecRestoreArray(vf, &fptr);
    if (ierr)
      return ierr;
    return 0;
  }
};

// --------------------------- Rank 1 ---------------------------
template <> struct NL<1, 0> {
  using ResidualFnTy = NeptuneMemRef1D (*)(void *x_alloc, void *x_aligned,
                                           int64_t x_off, int64_t x_s0,
                                           int64_t x_st0);

  struct Ctx {
    static constexpr int kRank = 1;
    static constexpr int kCaps = 0;

    SNES snes = nullptr;
    Vec x = nullptr;
    Vec f = nullptr;
    PetscOptions opts = nullptr;
    std::string prefix = "neptune_nl_";
    ResidualFnTy fn = nullptr;
    int64_t n = 0;
    bool setup_called = false;
  };

  static PetscErrorCode FormFunction(SNES, Vec vx, Vec vf, void *p) {
    auto *c = (Ctx *)p;
    if (!c || !c->fn)
      return 0;

    const PetscScalar *xptr = nullptr;
    PetscScalar *fptr = nullptr;
    PetscErrorCode ierr = VecGetArrayRead(vx, &xptr);
    if (ierr)
      return ierr;
    ierr = VecGetArray(vf, &fptr);
    if (ierr) {
      VecRestoreArrayRead(vx, &xptr);
      return ierr;
    }

    NeptuneMemRef1D xin = view_x1D((const double *)xptr, c->n);
    NeptuneMemRef1D fout = c->fn(xin.allocated, xin.aligned, xin.offset,
                                 xin.sizes[0], xin.strides[0]);

    // fout may be strided; pack into fptr
    NeptuneMemRef1D tmp = fout;
    tmp.allocated = tmp.aligned; // irrelevant
    pack1D_to_contig(tmp, (double *)fptr);

    if (fout.allocated)
      std::free(fout.allocated);

    ierr = VecRestoreArrayRead(vx, &xptr);
    if (ierr)
      return ierr;
    ierr = VecRestoreArray(vf, &fptr);
    if (ierr)
      return ierr;
    return 0;
  }
};

template <> struct NL<1, 1> {
  using ResidualFnTy = NeptuneMemRef1D (*)(void *x_alloc, void *x_aligned,
                                           int64_t x_off, int64_t x_s0,
                                           int64_t x_st0, void *c0_alloc,
                                           void *c0_aligned, int64_t c0_off,
                                           int64_t c0_s0, int64_t c0_st0);

  struct Ctx {
    static constexpr int kRank = 1;
    static constexpr int kCaps = 1;

    SNES snes = nullptr;
    Vec x = nullptr;
    Vec f = nullptr;
    PetscOptions opts = nullptr;
    std::string prefix = "neptune_nl_";
    ResidualFnTy fn = nullptr;
    NeptuneMemRef1D cap0{};
    int64_t n = 0;
    bool setup_called = false;
  };

  static PetscErrorCode FormFunction(SNES, Vec vx, Vec vf, void *p) {
    auto *c = (Ctx *)p;
    if (!c || !c->fn)
      return 0;

    const PetscScalar *xptr = nullptr;
    PetscScalar *fptr = nullptr;
    PetscErrorCode ierr = VecGetArrayRead(vx, &xptr);
    if (ierr)
      return ierr;
    ierr = VecGetArray(vf, &fptr);
    if (ierr) {
      VecRestoreArrayRead(vx, &xptr);
      return ierr;
    }

    NeptuneMemRef1D xin = view_x1D((const double *)xptr, c->n);
    NeptuneMemRef1D fout =
        c->fn(xin.allocated, xin.aligned, xin.offset, xin.sizes[0],
              xin.strides[0], c->cap0.allocated, c->cap0.aligned,
              c->cap0.offset, c->cap0.sizes[0], c->cap0.strides[0]);

    pack1D_to_contig(fout, (double *)fptr);

    if (fout.allocated)
      std::free(fout.allocated);

    ierr = VecRestoreArrayRead(vx, &xptr);
    if (ierr)
      return ierr;
    ierr = VecRestoreArray(vf, &fptr);
    if (ierr)
      return ierr;
    return 0;
  }
};

template <> struct NL<1, 2> {
  using ResidualFnTy = NeptuneMemRef1D (*)(
      void *x_alloc, void *x_aligned, int64_t x_off, int64_t x_s0,
      int64_t x_st0, void *c0_alloc, void *c0_aligned, int64_t c0_off,
      int64_t c0_s0, int64_t c0_st0, void *c1_alloc, void *c1_aligned,
      int64_t c1_off, int64_t c1_s0, int64_t c1_st0);

  struct Ctx {
    static constexpr int kRank = 1;
    static constexpr int kCaps = 2;

    SNES snes = nullptr;
    Vec x = nullptr;
    Vec f = nullptr;
    PetscOptions opts = nullptr;
    std::string prefix = "neptune_nl_";
    ResidualFnTy fn = nullptr;
    NeptuneMemRef1D cap0{};
    NeptuneMemRef1D cap1{};
    int64_t n = 0;
    bool setup_called = false;
  };

  static PetscErrorCode FormFunction(SNES, Vec vx, Vec vf, void *p) {
    auto *c = (Ctx *)p;
    if (!c || !c->fn)
      return 0;

    const PetscScalar *xptr = nullptr;
    PetscScalar *fptr = nullptr;
    PetscErrorCode ierr = VecGetArrayRead(vx, &xptr);
    if (ierr)
      return ierr;
    ierr = VecGetArray(vf, &fptr);
    if (ierr) {
      VecRestoreArrayRead(vx, &xptr);
      return ierr;
    }

    NeptuneMemRef1D xin = view_x1D((const double *)xptr, c->n);
    NeptuneMemRef1D fout = c->fn(
        xin.allocated, xin.aligned, xin.offset, xin.sizes[0], xin.strides[0],
        c->cap0.allocated, c->cap0.aligned, c->cap0.offset, c->cap0.sizes[0],
        c->cap0.strides[0], c->cap1.allocated, c->cap1.aligned, c->cap1.offset,
        c->cap1.sizes[0], c->cap1.strides[0]);

    pack1D_to_contig(fout, (double *)fptr);

    if (fout.allocated)
      std::free(fout.allocated);

    ierr = VecRestoreArrayRead(vx, &xptr);
    if (ierr)
      return ierr;
    ierr = VecRestoreArray(vf, &fptr);
    if (ierr)
      return ierr;
    return 0;
  }
};

// --------------------------- Rank 2 ---------------------------
template <> struct NL<2, 0> {
  using ResidualFnTy = NeptuneMemRef2D (*)(void *x_alloc, void *x_aligned,
                                           int64_t x_off, int64_t x_s0,
                                           int64_t x_s1, int64_t x_st0,
                                           int64_t x_st1);

  struct Ctx {
    static constexpr int kRank = 2;
    static constexpr int kCaps = 0;

    SNES snes = nullptr;
    Vec x = nullptr;
    Vec f = nullptr;
    PetscOptions opts = nullptr;
    std::string prefix = "neptune_nl_";
    ResidualFnTy fn = nullptr;
    int64_t s0 = 0, s1 = 0;
    bool setup_called = false;
  };

  static PetscErrorCode FormFunction(SNES, Vec vx, Vec vf, void *p) {
    auto *c = (Ctx *)p;
    if (!c || !c->fn)
      return 0;

    const PetscScalar *xptr = nullptr;
    PetscScalar *fptr = nullptr;
    PetscErrorCode ierr = VecGetArrayRead(vx, &xptr);
    if (ierr)
      return ierr;
    ierr = VecGetArray(vf, &fptr);
    if (ierr) {
      VecRestoreArrayRead(vx, &xptr);
      return ierr;
    }

    NeptuneMemRef2D xin = view_x2D((const double *)xptr, c->s0, c->s1);
    NeptuneMemRef2D fout =
        c->fn(xin.allocated, xin.aligned, xin.offset, xin.sizes[0],
              xin.sizes[1], xin.strides[0], xin.strides[1]);

    pack2D_to_contig_rm(fout, (double *)fptr);

    if (fout.allocated)
      std::free(fout.allocated);

    ierr = VecRestoreArrayRead(vx, &xptr);
    if (ierr)
      return ierr;
    ierr = VecRestoreArray(vf, &fptr);
    if (ierr)
      return ierr;
    return 0;
  }
};

template <> struct NL<2, 1> {
  using ResidualFnTy = NeptuneMemRef2D (*)(void *x_alloc, void *x_aligned,
                                           int64_t x_off, int64_t x_s0,
                                           int64_t x_s1, int64_t x_st0,
                                           int64_t x_st1, void *c0_alloc,
                                           void *c0_aligned, int64_t c0_off,
                                           int64_t c0_s0, int64_t c0_s1,
                                           int64_t c0_st0, int64_t c0_st1);

  struct Ctx {
    static constexpr int kRank = 2;
    static constexpr int kCaps = 1;

    SNES snes = nullptr;
    Vec x = nullptr;
    Vec f = nullptr;
    PetscOptions opts = nullptr;
    std::string prefix = "neptune_nl_";
    ResidualFnTy fn = nullptr;
    NeptuneMemRef2D cap0{};
    int64_t s0 = 0, s1 = 0;
    bool setup_called = false;
  };

  static PetscErrorCode FormFunction(SNES, Vec vx, Vec vf, void *p) {
    auto *c = (Ctx *)p;
    if (!c || !c->fn)
      return 0;

    const PetscScalar *xptr = nullptr;
    PetscScalar *fptr = nullptr;
    PetscErrorCode ierr = VecGetArrayRead(vx, &xptr);
    if (ierr)
      return ierr;
    ierr = VecGetArray(vf, &fptr);
    if (ierr) {
      VecRestoreArrayRead(vx, &xptr);
      return ierr;
    }

    NeptuneMemRef2D xin = view_x2D((const double *)xptr, c->s0, c->s1);
    NeptuneMemRef2D fout =
        c->fn(xin.allocated, xin.aligned, xin.offset, xin.sizes[0],
              xin.sizes[1], xin.strides[0], xin.strides[1], c->cap0.allocated,
              c->cap0.aligned, c->cap0.offset, c->cap0.sizes[0],
              c->cap0.sizes[1], c->cap0.strides[0], c->cap0.strides[1]);

    pack2D_to_contig_rm(fout, (double *)fptr);

    if (fout.allocated)
      std::free(fout.allocated);

    ierr = VecRestoreArrayRead(vx, &xptr);
    if (ierr)
      return ierr;
    ierr = VecRestoreArray(vf, &fptr);
    if (ierr)
      return ierr;
    return 0;
  }
};

template <> struct NL<2, 2> {
  using ResidualFnTy = NeptuneMemRef2D (*)(
      void *x_alloc, void *x_aligned, int64_t x_off, int64_t x_s0, int64_t x_s1,
      int64_t x_st0, int64_t x_st1, void *c0_alloc, void *c0_aligned,
      int64_t c0_off, int64_t c0_s0, int64_t c0_s1, int64_t c0_st0,
      int64_t c0_st1, void *c1_alloc, void *c1_aligned, int64_t c1_off,
      int64_t c1_s0, int64_t c1_s1, int64_t c1_st0, int64_t c1_st1);

  struct Ctx {
    static constexpr int kRank = 2;
    static constexpr int kCaps = 2;

    SNES snes = nullptr;
    Vec x = nullptr;
    Vec f = nullptr;
    PetscOptions opts = nullptr;
    std::string prefix = "neptune_nl_";
    ResidualFnTy fn = nullptr;
    NeptuneMemRef2D cap0{};
    NeptuneMemRef2D cap1{};
    int64_t s0 = 0, s1 = 0;
    bool setup_called = false;
  };

  static PetscErrorCode FormFunction(SNES, Vec vx, Vec vf, void *p) {
    auto *c = (Ctx *)p;
    if (!c || !c->fn)
      return 0;

    const PetscScalar *xptr = nullptr;
    PetscScalar *fptr = nullptr;
    PetscErrorCode ierr = VecGetArrayRead(vx, &xptr);
    if (ierr)
      return ierr;
    ierr = VecGetArray(vf, &fptr);
    if (ierr) {
      VecRestoreArrayRead(vx, &xptr);
      return ierr;
    }

    NeptuneMemRef2D xin = view_x2D((const double *)xptr, c->s0, c->s1);
    NeptuneMemRef2D fout = c->fn(
        xin.allocated, xin.aligned, xin.offset, xin.sizes[0], xin.sizes[1],
        xin.strides[0], xin.strides[1], c->cap0.allocated, c->cap0.aligned,
        c->cap0.offset, c->cap0.sizes[0], c->cap0.sizes[1], c->cap0.strides[0],
        c->cap0.strides[1], c->cap1.allocated, c->cap1.aligned, c->cap1.offset,
        c->cap1.sizes[0], c->cap1.sizes[1], c->cap1.strides[0],
        c->cap1.strides[1]);

    pack2D_to_contig_rm(fout, (double *)fptr);

    if (fout.allocated)
      std::free(fout.allocated);

    ierr = VecRestoreArrayRead(vx, &xptr);
    if (ierr)
      return ierr;
    ierr = VecRestoreArray(vf, &fptr);
    if (ierr)
      return ierr;
    return 0;
  }
};

// =============================================================================
// Common SNES runner
// =============================================================================
template <typename CtxT, typename SetCapsFn, typename BuildOutFn>
static inline auto
run_snes_common(const char *prefixName, const char *residual_sym, double tol,
                int64_t max_iters, const char *petsc_options, size_t nUnknowns,
                void *xbuf_contig, CtxT &ctx, SetCapsFn setCaps,
                BuildOutFn buildOut) -> decltype(buildOut()) {

  (void)prefixName;

  neptune_rt_init_noargs();

  neptunePetscCheck(SNESCreate(PETSC_COMM_SELF, &ctx.snes),
                    "SNESCreate failed");
  neptunePetscCheck(PetscOptionsCreate(&ctx.opts), "PetscOptionsCreate failed");
  neptunePetscCheck(PetscObjectSetOptions((PetscObject)ctx.snes, ctx.opts),
                    "PetscObjectSetOptions(SNES) failed");
  neptunePetscCheck(SNESSetOptionsPrefix(ctx.snes, ctx.prefix.c_str()),
                    "SNESSetOptionsPrefix failed");

  neptunePetscCheck(SNESSetType(ctx.snes, SNESNEWTONLS), "SNESSetType failed");
  neptunePetscCheck(SNESSetUseMatrixFree(ctx.snes, PETSC_TRUE, PETSC_TRUE),
                    "SNESSetUseMatrixFree failed");

  neptunePetscCheck(VecCreateSeq(PETSC_COMM_SELF, (PetscInt)nUnknowns, &ctx.x),
                    "VecCreateSeq failed");
  neptunePetscCheck(VecDuplicate(ctx.x, &ctx.f), "VecDuplicate failed");

  // resolve residual
  ctx.fn = (decltype(ctx.fn))neptuneDlsymOrAbort(residual_sym);

  // set captures into ctx
  setCaps();

  // options string (optional)
  if (petsc_options && petsc_options[0]) {
    neptunePetscCheck(PetscOptionsPrefixPush(ctx.opts, ctx.prefix.c_str()),
                      "PetscOptionsPrefixPush failed");
    neptunePetscCheck(PetscOptionsInsertString(ctx.opts, petsc_options),
                      "PetscOptionsInsertString failed");
    neptunePetscCheck(PetscOptionsPrefixPop(ctx.opts),
                      "PetscOptionsPrefixPop failed");
  }

  // tol / max_iters
  if (tol > 0 || max_iters > 0) {
    PetscInt its = (max_iters > 0) ? (PetscInt)max_iters : PETSC_DEFAULT;
    PetscReal rtol = (tol > 0) ? (PetscReal)tol : PETSC_DEFAULT;
    neptunePetscCheck(SNESSetTolerances(ctx.snes, rtol, PETSC_DEFAULT,
                                        PETSC_DEFAULT, its, PETSC_DEFAULT),
                      "SNESSetTolerances failed");
  }
  using NLSpec = NL<CtxT::kRank, CtxT::kCaps>;
  // Set residual callback
  neptunePetscCheck(
      SNESSetFunction(ctx.snes, ctx.f, &NLSpec::FormFunction, &ctx),
      "SNESSetFunction failed");

  // TODO: 这不是正确实现！应该容许默认的Jacobian，或者用户指定的！
    // --- Jacobian: default to matrix-free finite-difference (MFFD) ---
  Mat J = nullptr;
  neptunePetscCheck(MatCreateSNESMF(ctx.snes, &J), "MatCreateSNESMF failed");
  neptunePetscCheck(MatSetFromOptions(J), "MatSetFromOptions(J) failed");

  // Use the same operator for J and P for now (simplest)
  neptunePetscCheck(
      SNESSetJacobian(ctx.snes, J, J, MatMFFDComputeJacobian, nullptr),
      "SNESSetJacobian(MFFD) failed");

  neptunePetscCheck(SNESSetFromOptions(ctx.snes), "SNESSetFromOptions failed");
  neptunePetscCheck(SNESSetUp(ctx.snes), "SNESSetUp failed");

  // Place x
  neptunePetscCheck(VecPlaceArray(ctx.x, (PetscScalar *)xbuf_contig),
                    "VecPlaceArray(x) failed");

  neptunePetscCheck(SNESSolve(ctx.snes, nullptr, ctx.x), "SNESSolve failed");

  SNESConvergedReason reason;
  PetscInt its = 0;
  PetscReal fnorm = 0.0;

  neptunePetscCheck(SNESGetConvergedReason(ctx.snes, &reason),
                    "SNESGetConvergedReason failed");
  neptunePetscCheck(SNESGetIterationNumber(ctx.snes, &its),
                    "SNESGetIterationNumber failed");
  neptunePetscCheck(SNESGetFunctionNorm(ctx.snes, &fnorm),
                    "SNESGetFunctionNorm failed");

  neptunePetscCheck(
      PetscPrintf(PETSC_COMM_SELF, "===== SNES Solver Results =====\n"),
      "PetscPrintf failed");
  neptunePetscCheck(
      PetscPrintf(PETSC_COMM_SELF, "Iteration count: %d\n", (int)its),
      "PetscPrintf failed");
  neptunePetscCheck(
      PetscPrintf(PETSC_COMM_SELF, "Final residual norm: %.10e\n",
                  (double)fnorm),
      "PetscPrintf failed");
  neptunePetscCheck(PetscPrintf(PETSC_COMM_SELF,
                                "Convergence reason enum: %d\n", (int)reason),
                    "PetscPrintf failed");

  neptunePetscCheck(
      SNESConvergedReasonView(ctx.snes, PETSC_VIEWER_STDOUT_SELF),
      "SNESConvergedReasonView failed");

  neptunePetscCheck(VecResetArray(ctx.x), "VecResetArray(x) failed");

  // cleanup PETSc objects
  if (ctx.f)
    VecDestroy(&ctx.f);
  if (ctx.x)
    VecDestroy(&ctx.x);
  if (ctx.snes)
    SNESDestroy(&ctx.snes);
  if (ctx.opts)
    PetscOptionsDestroy(&ctx.opts);
  if (J)
    MatDestroy(&J);

  return buildOut();
}

// =============================================================================
// Exported entry points
// =============================================================================

// -------------------- 0D, 0/1/2 cap --------------------
extern "C" NeptuneMemRef0D _neptune_rt_runtime_solve_nonlinear_0d_0cap(
    void *x_alloc, void *x_aligned, int64_t x_off, const char *residual_sym,
    double tol, int64_t max_iters, const char *petsc_options) {
  std::fprintf(stderr,
               "[NeptuneRT] x(memref) x_alloc=%p x_aligned=%p x_off=%ld\n",
               x_alloc, x_aligned, (long)x_off);

  if (!x_aligned || !residual_sym || !residual_sym[0]) {
    std::fprintf(stderr, "[NeptuneRT] solve_nonlinear_0d_0cap: bad args\n");
    std::abort();
  }

  // pack x -> contiguous scalar buffer
  double *xbuf = (double *)std::malloc(sizeof(double));
  if (!xbuf)
    std::abort();
  NeptuneMemRef0D xin{x_alloc, x_aligned, x_off};
  pack0D_to_contig(xin, xbuf);

  using Impl = NL<0, 0>;
  typename Impl::Ctx ctx{};
  ctx.prefix = "neptune_nl_";

  return run_snes_common(
      "neptune_nl", residual_sym, tol, max_iters, petsc_options, numel0D(),
      xbuf, ctx, [&]() {}, [&]() { return makeMemref0D(xbuf); });
}

extern "C" NeptuneMemRef0D _neptune_rt_runtime_solve_nonlinear_0d_1cap(
    void *x_alloc, void *x_aligned, int64_t x_off, void *c0_alloc,
    void *c0_aligned, int64_t c0_off, const char *residual_sym, double tol,
    int64_t max_iters, const char *petsc_options) {
  std::fprintf(stderr,
               "[NeptuneRT] x(memref) x_alloc=%p x_aligned=%p x_off=%ld\n",
               x_alloc, x_aligned, (long)x_off);
  std::fprintf(stderr,
               "[NeptuneRT] c0(memref) c0_alloc=%p c0_aligned=%p c0_off=%ld\n",
               c0_alloc, c0_aligned, (long)c0_off);

  if (!x_aligned || !c0_aligned || !residual_sym || !residual_sym[0]) {
    std::fprintf(stderr, "[NeptuneRT] solve_nonlinear_0d_1cap: bad args\n");
    std::abort();
  }

  double *xbuf = (double *)std::malloc(sizeof(double));
  if (!xbuf)
    std::abort();
  NeptuneMemRef0D xin{x_alloc, x_aligned, x_off};
  pack0D_to_contig(xin, xbuf);

  using Impl = NL<0, 1>;
  typename Impl::Ctx ctx{};
  ctx.prefix = "neptune_nl_";

  return run_snes_common(
      "neptune_nl", residual_sym, tol, max_iters, petsc_options, numel0D(),
      xbuf, ctx,
      [&]() { ctx.cap0 = NeptuneMemRef0D{c0_alloc, c0_aligned, c0_off}; },
      [&]() { return makeMemref0D(xbuf); });
}

extern "C" NeptuneMemRef0D _neptune_rt_runtime_solve_nonlinear_0d_2cap(
    void *x_alloc, void *x_aligned, int64_t x_off, void *c0_alloc,
    void *c0_aligned, int64_t c0_off, void *c1_alloc, void *c1_aligned,
    int64_t c1_off, const char *residual_sym, double tol, int64_t max_iters,
    const char *petsc_options) {
  std::fprintf(stderr,
               "[NeptuneRT] x(memref) x_alloc=%p x_aligned=%p x_off=%ld\n",
               x_alloc, x_aligned, (long)x_off);
  std::fprintf(stderr,
               "[NeptuneRT] c0(memref) c0_alloc=%p c0_aligned=%p c0_off=%ld\n",
               c0_alloc, c0_aligned, (long)c0_off);
  std::fprintf(stderr,
               "[NeptuneRT] c1(memref) c1_alloc=%p c1_aligned=%p c1_off=%ld\n",
               c1_alloc, c1_aligned, (long)c1_off);

  if (!x_aligned || !c0_aligned || !c1_aligned || !residual_sym ||
      !residual_sym[0]) {
    std::fprintf(stderr, "[NeptuneRT] solve_nonlinear_0d_2cap: bad args\n");
    std::abort();
  }

  double *xbuf = (double *)std::malloc(sizeof(double));
  if (!xbuf)
    std::abort();
  NeptuneMemRef0D xin{x_alloc, x_aligned, x_off};
  pack0D_to_contig(xin, xbuf);

  using Impl = NL<0, 2>;
  typename Impl::Ctx ctx{};
  ctx.prefix = "neptune_nl_";

  return run_snes_common(
      "neptune_nl", residual_sym, tol, max_iters, petsc_options, numel0D(),
      xbuf, ctx,
      [&]() {
        ctx.cap0 = NeptuneMemRef0D{c0_alloc, c0_aligned, c0_off};
        ctx.cap1 = NeptuneMemRef0D{c1_alloc, c1_aligned, c1_off};
      },
      [&]() { return makeMemref0D(xbuf); });
}

// -------------------- 1D, 0/1/2 cap --------------------
extern "C" NeptuneMemRef1D _neptune_rt_runtime_solve_nonlinear_1d_0cap(
    void *x_alloc, void *x_aligned, int64_t x_off, int64_t x_s0, int64_t x_st0,
    const char *residual_sym, double tol, int64_t max_iters,
    const char *petsc_options) {
  std::fprintf(stderr,
               "[NeptuneRT] x(memref) x_alloc=%p x_aligned=%p x_off=%ld "
               "x_s0=%ld x_st0=%ld\n",
               x_alloc, x_aligned, (long)x_off, (long)x_s0, (long)x_st0);

  if (!x_aligned || x_s0 <= 0 || !residual_sym || !residual_sym[0]) {
    std::fprintf(stderr, "[NeptuneRT] solve_nonlinear_1d_0cap: bad args\n");
    std::abort();
  }

  const size_t n = numel1D(x_s0);
  double *xbuf = (double *)std::malloc(sizeof(double) * n);
  if (!xbuf)
    std::abort();

  NeptuneMemRef1D xin = makeMemref1D(x_alloc, x_aligned, x_off, x_s0, x_st0);
  pack1D_to_contig(xin, xbuf);

  using Impl = NL<1, 0>;
  typename Impl::Ctx ctx{};
  ctx.prefix = "neptune_nl_";
  ctx.n = x_s0;

  return run_snes_common(
      "neptune_nl", residual_sym, tol, max_iters, petsc_options, n, xbuf, ctx,
      [&]() {}, [&]() { return makeMemref1D(xbuf, xbuf, 0, x_s0, 1); });
}

extern "C" NeptuneMemRef1D _neptune_rt_runtime_solve_nonlinear_1d_1cap(
    void *x_alloc, void *x_aligned, int64_t x_off, int64_t x_s0, int64_t x_st0,
    void *c0_alloc, void *c0_aligned, int64_t c0_off, int64_t c0_s0,
    int64_t c0_st0, const char *residual_sym, double tol, int64_t max_iters,
    const char *petsc_options) {

  std::fprintf(stderr,
               "[NeptuneRT] solve_nonlinear_1d_1cap enter:\n"
               "  x_alloc=%p x_aligned=%p x_off=%ld x_s0=%ld x_st0=%ld\n"
               "  c0_alloc=%p c0_aligned=%p c0_off=%ld c0_s0=%ld c0_st0=%ld\n"
               "  residual_sym=%p '%s'\n"
               "  tol=%.17g max_iters=%ld petsc_options=%p '%s'\n",
               x_alloc, x_aligned, (long)x_off, (long)x_s0, (long)x_st0,
               c0_alloc, c0_aligned, (long)c0_off, (long)c0_s0, (long)c0_st0,
               (const void *)residual_sym,
               (residual_sym ? residual_sym : "(null)"), tol, (long)max_iters,
               (const void *)petsc_options,
               (petsc_options ? petsc_options : "(null)"));

  if (!x_aligned || !c0_aligned || x_s0 <= 0 || c0_s0 != x_s0 ||
      !residual_sym || !residual_sym[0]) {
    std::fprintf(stderr, "[NeptuneRT] solve_nonlinear_1d_1cap: bad args\n");
    std::abort();
  }

  const size_t n = numel1D(x_s0);
  double *xbuf = (double *)std::malloc(sizeof(double) * n);
  if (!xbuf)
    std::abort();

  NeptuneMemRef1D xin = makeMemref1D(x_alloc, x_aligned, x_off, x_s0, x_st0);
  pack1D_to_contig(xin, xbuf);

  using Impl = NL<1, 1>;
  typename Impl::Ctx ctx{};
  ctx.prefix = "neptune_nl_";
  ctx.n = x_s0;

  return run_snes_common(
      "neptune_nl", residual_sym, tol, max_iters, petsc_options, n, xbuf, ctx,
      [&]() {
        ctx.cap0 = makeMemref1D(c0_alloc, c0_aligned, c0_off, c0_s0, c0_st0);
      },
      [&]() { return makeMemref1D(xbuf, xbuf, 0, x_s0, 1); });
}

extern "C" NeptuneMemRef1D _neptune_rt_runtime_solve_nonlinear_1d_2cap(
    void *x_alloc, void *x_aligned, int64_t x_off, int64_t x_s0, int64_t x_st0,
    void *c0_alloc, void *c0_aligned, int64_t c0_off, int64_t c0_s0,
    int64_t c0_st0, void *c1_alloc, void *c1_aligned, int64_t c1_off,
    int64_t c1_s0, int64_t c1_st0, const char *residual_sym, double tol,
    int64_t max_iters, const char *petsc_options) {
  std::fprintf(stderr,
               "[NeptuneRT] x(memref) x_alloc=%p x_aligned=%p x_off=%ld "
               "x_s0=%ld x_st0=%ld\n",
               x_alloc, x_aligned, (long)x_off, (long)x_s0, (long)x_st0);
  std::fprintf(stderr,
               "[NeptuneRT] c0(memref) c0_alloc=%p c0_aligned=%p c0_off=%ld "
               "c0_s0=%ld c0_st0=%ld\n",
               c0_alloc, c0_aligned, (long)c0_off, (long)c0_s0,
               (long)c0_st0);
  std::fprintf(stderr,
               "[NeptuneRT] c1(memref) c1_alloc=%p c1_aligned=%p c1_off=%ld "
               "c1_s0=%ld c1_st0=%ld\n",
               c1_alloc, c1_aligned, (long)c1_off, (long)c1_s0,
               (long)c1_st0);

  if (!x_aligned || !c0_aligned || !c1_aligned || x_s0 <= 0 || c0_s0 != x_s0 ||
      c1_s0 != x_s0 || !residual_sym || !residual_sym[0]) {
    std::fprintf(stderr, "[NeptuneRT] solve_nonlinear_1d_2cap: bad args\n");
    std::abort();
  }

  const size_t n = numel1D(x_s0);
  double *xbuf = (double *)std::malloc(sizeof(double) * n);
  if (!xbuf)
    std::abort();

  NeptuneMemRef1D xin = makeMemref1D(x_alloc, x_aligned, x_off, x_s0, x_st0);
  pack1D_to_contig(xin, xbuf);

  using Impl = NL<1, 2>;
  typename Impl::Ctx ctx{};
  ctx.prefix = "neptune_nl_";
  ctx.n = x_s0;

  return run_snes_common(
      "neptune_nl", residual_sym, tol, max_iters, petsc_options, n, xbuf, ctx,
      [&]() {
        ctx.cap0 = makeMemref1D(c0_alloc, c0_aligned, c0_off, c0_s0, c0_st0);
        ctx.cap1 = makeMemref1D(c1_alloc, c1_aligned, c1_off, c1_s0, c1_st0);
      },
      [&]() { return makeMemref1D(xbuf, xbuf, 0, x_s0, 1); });
}

// -------------------- 2D, 0/1/2 cap --------------------
extern "C" NeptuneMemRef2D _neptune_rt_runtime_solve_nonlinear_2d_0cap(
    void *x_alloc, void *x_aligned, int64_t x_off, int64_t x_s0, int64_t x_s1,
    int64_t x_st0, int64_t x_st1, const char *residual_sym, double tol,
    int64_t max_iters, const char *petsc_options) {
  std::fprintf(stderr,
               "[NeptuneRT] x(memref) x_alloc=%p x_aligned=%p x_off=%ld "
               "x_s=(%ld,%ld) x_st=(%ld,%ld)\n",
               x_alloc, x_aligned, (long)x_off, (long)x_s0, (long)x_s1,
               (long)x_st0, (long)x_st1);

  if (!x_aligned || x_s0 <= 0 || x_s1 <= 0 || !residual_sym ||
      !residual_sym[0]) {
    std::fprintf(stderr, "[NeptuneRT] solve_nonlinear_2d_0cap: bad args\n");
    std::abort();
  }

  const size_t n = numel2D(x_s0, x_s1);
  double *xbuf = (double *)std::malloc(sizeof(double) * n);
  if (!xbuf)
    std::abort();

  NeptuneMemRef2D xin =
      makeMemref2D(x_alloc, x_aligned, x_off, x_s0, x_s1, x_st0, x_st1);
  pack2D_to_contig_rm(xin, xbuf);

  using Impl = NL<2, 0>;
  typename Impl::Ctx ctx{};
  ctx.prefix = "neptune_nl_";
  ctx.s0 = x_s0;
  ctx.s1 = x_s1;

  // return contiguous row-major 2D: stride0=s1, stride1=1
  return run_snes_common(
      "neptune_nl", residual_sym, tol, max_iters, petsc_options, n, xbuf, ctx,
      [&]() {},
      [&]() { return makeMemref2D(xbuf, xbuf, 0, x_s0, x_s1, x_s1, 1); });
}

extern "C" NeptuneMemRef2D _neptune_rt_runtime_solve_nonlinear_2d_1cap(
    void *x_alloc, void *x_aligned, int64_t x_off, int64_t x_s0, int64_t x_s1,
    int64_t x_st0, int64_t x_st1, void *c0_alloc, void *c0_aligned,
    int64_t c0_off, int64_t c0_s0, int64_t c0_s1, int64_t c0_st0,
    int64_t c0_st1, const char *residual_sym, double tol, int64_t max_iters,
    const char *petsc_options) {
  std::fprintf(stderr,
               "[NeptuneRT] x(memref) x_alloc=%p x_aligned=%p x_off=%ld "
               "x_s=(%ld,%ld) x_st=(%ld,%ld)\n",
               x_alloc, x_aligned, (long)x_off, (long)x_s0, (long)x_s1,
               (long)x_st0, (long)x_st1);
  std::fprintf(stderr,
               "[NeptuneRT] c0(memref) c0_alloc=%p c0_aligned=%p c0_off=%ld "
               "c0_s=(%ld,%ld) c0_st=(%ld,%ld)\n",
               c0_alloc, c0_aligned, (long)c0_off, (long)c0_s0, (long)c0_s1,
               (long)c0_st0, (long)c0_st1);

  if (!x_aligned || !c0_aligned || x_s0 <= 0 || x_s1 <= 0 || c0_s0 != x_s0 ||
      c0_s1 != x_s1 || !residual_sym || !residual_sym[0]) {
    std::fprintf(stderr, "[NeptuneRT] solve_nonlinear_2d_1cap: bad args\n");
    std::abort();
  }

  const size_t n = numel2D(x_s0, x_s1);
  double *xbuf = (double *)std::malloc(sizeof(double) * n);
  if (!xbuf)
    std::abort();

  NeptuneMemRef2D xin =
      makeMemref2D(x_alloc, x_aligned, x_off, x_s0, x_s1, x_st0, x_st1);
  pack2D_to_contig_rm(xin, xbuf);

  using Impl = NL<2, 1>;
  typename Impl::Ctx ctx{};
  ctx.prefix = "neptune_nl_";
  ctx.s0 = x_s0;
  ctx.s1 = x_s1;

  return run_snes_common(
      "neptune_nl", residual_sym, tol, max_iters, petsc_options, n, xbuf, ctx,
      [&]() {
        ctx.cap0 = makeMemref2D(c0_alloc, c0_aligned, c0_off, c0_s0, c0_s1,
                                c0_st0, c0_st1);
      },
      [&]() { return makeMemref2D(xbuf, xbuf, 0, x_s0, x_s1, x_s1, 1); });
}

extern "C" NeptuneMemRef2D _neptune_rt_runtime_solve_nonlinear_2d_2cap(
    void *x_alloc, void *x_aligned, int64_t x_off, int64_t x_s0, int64_t x_s1,
    int64_t x_st0, int64_t x_st1, void *c0_alloc, void *c0_aligned,
    int64_t c0_off, int64_t c0_s0, int64_t c0_s1, int64_t c0_st0,
    int64_t c0_st1, void *c1_alloc, void *c1_aligned, int64_t c1_off,
    int64_t c1_s0, int64_t c1_s1, int64_t c1_st0, int64_t c1_st1,
    const char *residual_sym, double tol, int64_t max_iters,
    const char *petsc_options) {
  std::fprintf(stderr,
               "[NeptuneRT] x(memref) x_alloc=%p x_aligned=%p x_off=%ld "
               "x_s=(%ld,%ld) x_st=(%ld,%ld)\n",
               x_alloc, x_aligned, (long)x_off, (long)x_s0, (long)x_s1,
               (long)x_st0, (long)x_st1);
  std::fprintf(stderr,
               "[NeptuneRT] c0(memref) c0_alloc=%p c0_aligned=%p c0_off=%ld "
               "c0_s=(%ld,%ld) c0_st=(%ld,%ld)\n",
               c0_alloc, c0_aligned, (long)c0_off, (long)c0_s0, (long)c0_s1,
               (long)c0_st0, (long)c0_st1);
  std::fprintf(stderr,
               "[NeptuneRT] c1(memref) c1_alloc=%p c1_aligned=%p c1_off=%ld "
               "c1_s=(%ld,%ld) c1_st=(%ld,%ld)\n",
               c1_alloc, c1_aligned, (long)c1_off, (long)c1_s0, (long)c1_s1,
               (long)c1_st0, (long)c1_st1);

  if (!x_aligned || !c0_aligned || !c1_aligned || x_s0 <= 0 || x_s1 <= 0 ||
      c0_s0 != x_s0 || c0_s1 != x_s1 || c1_s0 != x_s0 || c1_s1 != x_s1 ||
      !residual_sym || !residual_sym[0]) {
    std::fprintf(stderr, "[NeptuneRT] solve_nonlinear_2d_2cap: bad args\n");
    std::abort();
  }

  const size_t n = numel2D(x_s0, x_s1);
  double *xbuf = (double *)std::malloc(sizeof(double) * n);
  if (!xbuf)
    std::abort();

  NeptuneMemRef2D xin =
      makeMemref2D(x_alloc, x_aligned, x_off, x_s0, x_s1, x_st0, x_st1);
  pack2D_to_contig_rm(xin, xbuf);

  using Impl = NL<2, 2>;
  typename Impl::Ctx ctx{};
  ctx.prefix = "neptune_nl_";
  ctx.s0 = x_s0;
  ctx.s1 = x_s1;

  return run_snes_common(
      "neptune_nl", residual_sym, tol, max_iters, petsc_options, n, xbuf, ctx,
      [&]() {
        ctx.cap0 = makeMemref2D(c0_alloc, c0_aligned, c0_off, c0_s0, c0_s1,
                                c0_st0, c0_st1);
        ctx.cap1 = makeMemref2D(c1_alloc, c1_aligned, c1_off, c1_s0, c1_s1,
                                c1_st0, c1_st1);
      },
      [&]() { return makeMemref2D(xbuf, xbuf, 0, x_s0, x_s1, x_s1, 1); });
}

// Optional: for Python to free returned buffers
extern "C" void neptune_rt_free(void *p) { std::free(p); }
