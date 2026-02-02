# Halide Tiling Experiments (case_big2d)

Date: 2026-01-29

Scope: tiling sensitivity (L1 bytes / cache alpha). No git commit.

## Commands

- make -C test/case_test/case_big2d clean
- <env> make -C test/case_test/case_big2d compare
- <env> make -C test/case_test/case_big2d time

## Results

| Experiment | Env | Schedule summary | Baseline elapsed_ms | Generated elapsed_ms |
|---|---|---|---:|---:|
| l1-16384 | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=4 NEPTUNECC_L1_BYTES=16384` | `neptune-cc: schedule for 'k0' rank=2 shape=[2048,2048] outMin=[1,1] outExtent=[2046,2046] radius=[1,1] elem_bytes=4 split=[32,32] vec=8 par=y unroll=2` | 3730 | 575 |
| l1-32768 | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=4 NEPTUNECC_L1_BYTES=32768` | `neptune-cc: schedule for 'k0' rank=2 shape=[2048,2048] outMin=[1,1] outExtent=[2046,2046] radius=[1,1] elem_bytes=4 split=[64,32] vec=8 par=y unroll=2` | 3823 | 774 |
| l1-65536 | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=4 NEPTUNECC_L1_BYTES=65536` | `neptune-cc: schedule for 'k0' rank=2 shape=[2048,2048] outMin=[1,1] outExtent=[2046,2046] radius=[1,1] elem_bytes=4 split=[64,64] vec=8 par=y unroll=2` | 3916 | 594 |
| alpha-0.4 | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=4 NEPTUNECC_L1_BYTES=32768 NEPTUNECC_CACHE_ALPHA=0.4` | `neptune-cc: schedule for 'k0' rank=2 shape=[2048,2048] outMin=[1,1] outExtent=[2046,2046] radius=[1,1] elem_bytes=4 split=[32,32] vec=8 par=y unroll=2` | 3877 | 595 |
| alpha-0.6 | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=4 NEPTUNECC_L1_BYTES=32768 NEPTUNECC_CACHE_ALPHA=0.6` | `neptune-cc: schedule for 'k0' rank=2 shape=[2048,2048] outMin=[1,1] outExtent=[2046,2046] radius=[1,1] elem_bytes=4 split=[64,32] vec=8 par=y unroll=2` | 3837 | 767 |
| alpha-0.8 | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=4 NEPTUNECC_L1_BYTES=32768 NEPTUNECC_CACHE_ALPHA=0.8` | `neptune-cc: schedule for 'k0' rank=2 shape=[2048,2048] outMin=[1,1] outExtent=[2046,2046] radius=[1,1] elem_bytes=4 split=[64,32] vec=8 par=y unroll=2` | 4033 | 840 |
