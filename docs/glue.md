# Glue generation

`neptune-cc` emits C++ glue code to wrap user pointers into `halide_buffer_t`
structures and call the Halide AOT entry points.

## Files
- `out/glue/neptunecc_kernels.h`
- `out/glue/neptunecc_kernels.cpp`
- `out/glue/neptunecc_generated.cmake`

## Buffer conventions
- Row-major layout; IR dims are reversed in Halide dims.
- `dim0` corresponds to the last IR index (fastest varying).
- `offset_elems = sum(outMin[d] * stride[d])` with row-major strides.

## Input policy
- **P0 (default)**: align input mins to `-outMin`, host offset 0.
- **P1 (ghosted)**: mins = `-radius`, host offset = `outMin - radius`.

P1 is enabled only when the input port is marked `ghosted` and
`NEPTUNECC_GHOSTED_BASE=1` is set.

## Overlap wrappers
When overlap is enabled, glue emits additional wrappers:
- `k0_interior(...)` for the safe-interior box.
- `k0_face_*` for boundary faces (rank<=3).

All wrappers reuse the same Halide AOT kernel, with different output boxes.
