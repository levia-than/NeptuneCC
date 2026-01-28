# case_test/case1

This is a minimal end-to-end demo that runs the full pipeline:

1) `neptune-cc` extracts the pragma kernel and emits:
   - `out/manifest.json`
   - `out/glue/` and `out/rewritten/`
   - `out/halide/NeptuneHalideHelpers.h`
   - `out/halide/halide_kernels.cpp` (via `--emit-halide`)
2) Compile and run the Halide generator to produce `k0.h` / `k0.a`
3) Build baseline and generated binaries, compare outputs, then time both.

The example kernel is a simple 2D 5-point stencil that writes only the interior
region (i=1..6, j=1..6). The pragma kernel and `main()` live in the same file
(`case1.cpp`).

No standalone `halide_build_main.cpp` is stored in the repo; the Makefile
generates a tiny `main()` on the fly when building the Halide generator.

## Requirements
- Built `neptune-cc`.
- Halide release installed (default path below).

## Usage
```bash
cd test/case_test/case1
make \
  HALIDE_ROOT=/home/wyx/project/stencil-solver-playground/halide/Halide-21.0.0-x86-64-linux
```

Outputs are under `test/case_test/case1/out/`.
The Makefile passes `--emit-halide` by default via `NEPTUNE_CC_FLAGS`.

If `NEPTUNE_CC` is not found automatically, pass it explicitly:
```bash
make \
  NEPTUNE_CC=/home/wyx/project/NeptuneCC/build/project-build/tools/neptune-cc/neptune-cc \
  HALIDE_ROOT=/home/wyx/project/stencil-solver-playground/halide/Halide-21.0.0-x86-64-linux
```

If you see `stddef.h` not found, ensure Clang's resource dir is set:
```bash
make CLANG=clang++ \
  CLANG_RESOURCE_DIR=$(clang++ -print-resource-dir)
```

## neptune-cc 参数与工作原理（简要）

常用参数：
- `-p <path>`: build path（包含 `compile_commands.json` 的目录）。
- `--out-dir=<path>`: 输出目录（manifest、MLIR、glue、rewritten、halide helper）。
- `--emit-halide`: 生成 `halide/halide_kernels.cpp`。
- `--emit-kernels-mlir`: 生成 `kernels.mlir`（调试用）。
- `--emit-emitc-mlir`: 生成 `halide/emitc.mlir`（调试用）。
- `--extra-arg=<arg>` / `--extra-arg-before=<arg>`: 透传给 clang 的编译参数。

流程概览：
1) `neptune-cc -p <build> <source> --out-dir=<out> --emit-halide` 解析 pragma 区域，生成：
   - `manifest.json`
   - `glue/`、`rewritten/`
   - `halide/NeptuneHalideHelpers.h`
   - `halide/halide_kernels.cpp`
2) 编译并运行 `halide_kernels.cpp` 生成 `k0.h`/`k0.a`。
3) 用 `rewritten` 源文件 + `glue` + `k0.a` 生成最终可执行程序。

注意：本案例会先在 `out/` 里生成一个临时的 `compile_commands.json`，
因此 `-p` 直接指向 `out/` 即可。

如需调试 MLIR，可额外传：`--emit-kernels-mlir`（生成 `kernels.mlir`）
或 `--emit-emitc-mlir`（生成 `halide/emitc.mlir`）。`neptune-opt`
仅用于调试，不再出现在常规流程中。

## 清理
```bash
make clean
```
