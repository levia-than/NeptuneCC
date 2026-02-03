# Halide schedule inference

NeptuneCC infers a deterministic schedule for `neptune.ir.apply` and records it
as `neptune.schedule` attributes. The EmitC Halide emitter consumes those attrs.

## Inputs
- rank, bounds, radius (from apply)
- element size
- shape (from input temp bounds; fallback to outExtent)

## Heuristics
- Choose a tile that maximizes volume under a cache footprint limit.
- Vectorize the fastest dimension when `extent_fast >= VL`.
- Optional parallelization on outer tile dimensions when threads>1.
- Unroll the inner-y loop when it divides evenly (conservative in 3D).

## Knobs
Environment variables:
- `NEPTUNECC_VECTOR_WIDTH` (VL)
- `NEPTUNECC_L1_BYTES` (default 32KB)
- `NEPTUNECC_CACHE_ALPHA` (default 0.6)
- `NEPTUNECC_THREADS` / `HL_NUM_THREADS` / `OMP_NUM_THREADS`

## EmitC lowering
EmitC calls helper functions in `NeptuneHalideHelpers.h` to apply:
- split / reorder
- vectorize
- unroll
- parallel (when enabled)
