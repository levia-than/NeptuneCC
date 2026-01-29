# Halide Tuning Experiments (case_big2d)

Date: 2026-01-29

## Environment

- Case: test/case_test/case_big2d
- Command (per experiment):
  - make -C test/case_test/case_big2d clean
  - <env> make -C test/case_test/case_big2d compare
  - <env> make -C test/case_test/case_big2d time

## Results (elapsed_ms from program stdout)

| Experiment | Env | Schedule summary | Baseline elapsed_ms | Generated elapsed_ms |
|---|---|---|---:|---:|
| default | `` | `neptune-cc: schedule for 'k0' rank=2 shape=[2048,2048] outMin=[1,1] outExtent=[2046,2046] radius=[1,1] elem_bytes=4 tile=[64,16] vec=1 par=none unroll=4 (VL defaulted: set NEPTUNECC_VECTOR_WIDTH for target VL)` | n/a | n/a |
| VL=8, threads=4 | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=4` | `neptune-cc: schedule for 'k0' rank=2 shape=[2048,2048] outMin=[1,1] outExtent=[2046,2046] radius=[1,1] elem_bytes=4 tile=[64,32] vec=8 par=y unroll=2` | n/a | n/a |
| VL=8, threads=8 | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=8` | `neptune-cc: schedule for 'k0' rank=2 shape=[2048,2048] outMin=[1,1] outExtent=[2046,2046] radius=[1,1] elem_bytes=4 tile=[64,32] vec=8 par=y unroll=2` | n/a | n/a |

## Notes

- `elapsed_ms` is printed by the benchmark itself (stderr).
- `time` target also prints shell `time -p` results; those are not included here.
- When `NEPTUNECC_VECTOR_WIDTH` is unset, schedule uses `vec=1` and parallel is disabled.
