<!--
 * @Author: leviathan 670916484@qq.com
 * @Date: 2025-09-08 20:34:11
 * @LastEditors: leviathan 670916484@qq.com
 * @LastEditTime: 2025-11-09 10:33:05
 * @FilePath: /NeptuneCC/README.md
 * @Description: 
 * 
 * Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
-->
# NeptuneCC PDE solver

NeptuneCC is an MLIR/LLVM toolchain for stencil/PDE kernels. It ships the
NeptuneIR dialect and passes, the `neptune-opt` driver, and the `neptune-cc`
Clang frontend for `#pragma neptune`. Runtime integration is currently out of
tree.

## Build

```bash
# Optionally pin LLVM via submodule (build.sh can also clone it).
git submodule update --init --recursive

# Build LLVM + Neptune
bash scripts/build.sh

# Common options
bash scripts/build.sh --project-debug
bash scripts/build.sh --rebuild-llvm
bash scripts/build.sh --clean

```

Artifacts land under `build/`:
- `build/llvm-install` (LLVM/MLIR tools)
- `build/project-build/bin/neptune-opt`
- `build/project-build/bin/neptune-cc`

The build script symlinks `compile_commands.json` to the repo root and runs
`check-neptune` after installing the tools.

## Project layout
- `include/`: tablegen + public headers (dialect, passes, frontend, utils)
- `lib/`: implementations of the dialect, passes, frontend, and pipeline
- `tools/`: `neptune-opt`, `neptune-cc`
- `test/`: lit tests and smoke scripts
- `third_party/llvm-project`: LLVM/MLIR (submodule or cloned by `build.sh`)

## NeptuneIR snapshot
- Types: `!neptune.field<element=?, bounds=?, location=?, layout=?>` for
  storage-backed fields, `!neptune.temp<...>` for value-semantics temporaries.
- Attributes: `#neptune.bounds<lb=[...], ub=[...]>`, `#neptune.location<"...">`,
  `#neptune.layout<order="...", strides=[...], halo=[...], offset=[...]>`,
  optional `#neptune.stencil_shape<[[...], ...]>`.
- Core ops: `neptune.ir.wrap`/`neptune.ir.unwrap`, `neptune.ir.load`/`store`,
  `neptune.ir.apply` + `neptune.ir.access` + `neptune.ir.yield`, `neptune.ir.reduce`.
- Passes: `neptuneir-verify-forms`, `neptuneir-normalize-apply`,
  `neptuneir-split-domain`, `neptuneir-emitc-halide`.

Note: some legacy inputs still use the `neptune_ir` namespace, while the
newer syntax uses the `neptune` dialect (see `test/mlir_tests/smoke_emitC_for_apply.mlir`).

## Basic usage
```bash
./build/project-build/bin/neptune-opt test/mlir_tests/smoke_emitC_for_apply.mlir \
  --neptuneir-verify-forms --neptuneir-normalize-apply -o normalized.mlir

./build/project-build/bin/neptune-opt test/mlir_tests/smoke_emitC_for_apply.mlir \
  --neptuneir-emitc-halide -o emitc.mlir
```

The `neptuneir-to-llvm` pipeline is registered in `neptune-opt` for LLVM
lowering experiments.

For the Clang frontend:
```bash
./build/project-build/bin/neptune-cc test/case_test/case1/case1.cpp \
  --out-dir /tmp/neptune_out

# If compile_commands.json is not in the repo root:
./build/project-build/bin/neptune-cc -p build/project-build \
  test/case_test/case1/case1.cpp --out-dir /tmp/neptune_out
```

`neptune-cc` writes a `manifest.json` under the chosen output directory.
