<!--
 * @Author: leviathan 670916484@qq.com
 * @Date: 2025-09-08 20:34:11
 * @LastEditors: leviathan 670916484@qq.com
 * @LastEditTime: 2025-11-09 10:33:05
 * @FilePath: /neptune-pde-solver/README.md
 * @Description: 
 * 
 * Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
-->
# Neptune pde solver

```bash
git submodule update --init
bash scripts/build.sh

# enable PETSc (optional)
NEPTUNE_ENABLE_PETSC=ON PETSC_DIR=/path/to/petsc PETSC_ARCH=arch-name bash scripts/build.sh
# or use CLI flags:
# ./scripts/build.sh --enable-petsc --petsc-dir /path/to/petsc --petsc-arch arch-name
```

`bash scripts/build.sh -c` (or `--clean`) will remove everything under the
centralized `build/` root. `neptune-opt` is installed to
`build/project-build/bin/neptune-opt`.

## basic arch
`lib` directory would contain all mlir-related library.
`src` directory would export some opt tools for the sake of testing.
`include` would contain all tablegen files.

## NeptuneIR snapshot
- Types: `!neptune_ir.field<element=?, bounds=?, location=?>` for storage-backed
  fields, `!neptune_ir.temp<...>` for value-semantics temporaries.
- Attributes: `#neptune_ir.bounds<lb=[...], ub=[...]>` for iteration domains,
  `#neptune_ir.location<"...">` for where data lives on the mesh, optional
  `#neptune_ir.stencil_shape<[[...], ...]>` to mark neighbor offsets.
- Core ops: `wrap`/`unwrap` bridge buffers, `load`/`store` move between field
  and temp, `apply` hosts the stencil body and yields scalars via `yield`,
  neighbor reads use `access`.

## basic usage
```bash
./build/project-build/bin/neptune-opt docs/test.mlir \
  --pass-pipeline='builtin.module(neptuneir-to-llvm)' \
  -o lower.mlir
```

```bash
./build/llvm-install/bin/mlir-translate --mlir-to-llvmir lower.mlir -o lower.ll
```

```bash
./build/llvm-install/bin/llc -filetype=obj lower.ll -o lower.o
```

To debug individual lowering steps, you can also drive passes manually, e.g.:

```bash
./build/project-build/bin/neptune-opt \
  test/mlir_tests/conversion_tests/apply-2d-5pt.mlir \
  --normalize-neptune-ir-storage --neptune-ir-stencil-to-scf
```

## front-ends and runtime
`lib/Codegen` / `lib/Utils` are currently stubs; only the C++/MLIR pipeline and
`neptune-opt` driver are built. AOT runners are planned but not shipped yet.

## PETSc integration
- Enable at configure time with `-DNEPTUNE_ENABLE_PETSC=ON` (or `--enable-petsc` for `scripts/build.sh`).
- If `PETSC_DIR` points to an install prefix (or use `-DCMAKE_PREFIX_PATH=/path/to/petsc-install`), CMake will find it via `FindPETSc.cmake`. The build script still bootstraps into `build/petsc-install` if nothing is provided.
- Override paths via `PETSC_SRC_DIR`/`PETSC_INSTALL_DIR` env vars if you don't want to use `build/`.
- When enabled, link PETSc consumers against `PETSc::PETSc` (imported target from the find module). A small smoke binary `petsc_smoke` is built to validate the linkage.
