# case_test/case_big2d

Large 2D case for early performance sanity checks (512x512, 5-point stencil).
Timing is printed to stderr; stdout stays deterministic for compare.

## Usage
```bash
cd test/case_test/case_big2d
make \
  HALIDE_ROOT=/home/wyx/project/stencil-solver-playground/halide/Halide-21.0.0-x86-64-linux
```

Outputs are under `test/case_test/case_big2d/out/`.
