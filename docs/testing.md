# Testing

## MLIR regression tests (lit)
```bash
ninja -C build/project-build check-neptune
```

## End-to-end cases
Each case in `test/case_test/<case>` provides a Makefile.
```bash
make -C test/case_test/case1 clean
make -C test/case_test/case1 compare
```

Common cases:
- case1 / case2 / case3
- case_big2d
- case_boundary2d / case_boundary3d
- case_overlap2d

Tip: set `CCACHE_DISABLE=1` if ccache blocks builds.
