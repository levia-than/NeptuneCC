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
# NeptuneCC stencil compiler

NeptuneCC is an MLIR/LLVM toolchain for stencil kernels and structured grid
computations. It provides the NeptuneIR dialect + passes, a `neptune-opt`
driver, and the `neptune-cc` Clang frontend for `#pragma neptune`.

Highlights:
- Frontend extraction from C/C++ stencil loops into NeptuneIR.
- Halide AOT emission with deterministic schedule inference.
- Glue generation + source rewrite for kernel calls.
- Overlap-aware regions (halo begin/end + interior/boundary faces).

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

## Quickstart
```bash
# Build the toolchain
bash scripts/build.sh

# Frontend: extract + emit Halide (outputs in --out-dir)
./build/project-build/bin/neptune-cc test/case_test/case1/case1.cpp \
  --out-dir /tmp/neptune_out

# MLIR-only pipelines
./build/project-build/bin/neptune-opt test/mlir_tests/smoke_emitC_for_apply.mlir \
  --neptuneir-verify-forms --neptuneir-normalize-apply -o normalized.mlir
```

## Pragmas (quick reference)
```cpp
#pragma neptune kernel begin tag(k0) in(x:ghosted) out(y:owned)
  { /* stencil loops */ }
#pragma neptune kernel end tag(k0)

#pragma neptune halo begin tag(h0) dm(da) field(x) kind(global_to_local_begin)
  /* user communication begin */
#pragma neptune halo end tag(h0) kind(global_to_local_end)

#pragma neptune overlap begin tag(o0) halo(h0) kernel(k0) policy(auto)
  { /* halo begin + kernel + halo end */ }
#pragma neptune overlap end tag(o0)
```

## Output layout
`neptune-cc --out-dir <out>` produces:
- `out/manifest.json` (front-end events + offsets)
- `out/kernels.mlir` (kernel module snapshot)
- `out/halide/` (Halide generator + AOT headers/libs)
- `out/glue/` (C++ glue: `neptunecc_kernels.*`)
- `out/rewritten/` (rewritten C++ calling `neptunecc::k*`)

## Testing
- MLIR regression tests (lit): `ninja -C build/project-build check-neptune`
- End-to-end cases: `make -C test/case_test/<case> compare`

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

## Docs
See `docs/README.md` for a short index of pragmas, overlap, glue, and tuning.
