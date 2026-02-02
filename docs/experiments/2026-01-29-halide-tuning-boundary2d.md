# Halide Tuning Experiments (case_boundary2d)

Date: 2026-01-29

Scope: experiments 0/1/2 on test/case_test/case_boundary2d. No git commit.

## Commands

- make -C test/case_test/case_boundary2d clean
- <env> make -C test/case_test/case_boundary2d compare
- <env> make -C test/case_test/case_boundary2d time

## Results

| Experiment | Env | Schedule summary | Baseline elapsed_ms | Generated elapsed_ms |
|---|---|---|---:|---:|
| 0-default | `` | `neptune-cc: schedule for 'k0' rank=2 shape=[1024,1024] outMin=[1,1] outExtent=[1022,1022] radius=[1,1] elem_bytes=4 split=[64,16] vec=1 par=none unroll=4 (VL defaulted: set NEPTUNECC_VECTOR_WIDTH for target VL)` | 60 | 60 |
| 1-vl4 | `NEPTUNECC_VECTOR_WIDTH=4 NEPTUNECC_THREADS=1` | `neptune-cc: schedule for 'k0' rank=2 shape=[1024,1024] outMin=[1,1] outExtent=[1022,1022] radius=[1,1] elem_bytes=4 split=[64,32] vec=4 par=none unroll=4` | 60 | 30 |
| 1-vl8 | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=1` | `neptune-cc: schedule for 'k0' rank=2 shape=[1024,1024] outMin=[1,1] outExtent=[1022,1022] radius=[1,1] elem_bytes=4 split=[64,32] vec=8 par=none unroll=2` | 60 | 30 |
| 1-vl16 | `NEPTUNECC_VECTOR_WIDTH=16 NEPTUNECC_THREADS=1` | `neptune-cc: schedule for 'k0' rank=2 shape=[1024,1024] outMin=[1,1] outExtent=[1022,1022] radius=[1,1] elem_bytes=4 split=[32,64] vec=16 par=none unroll=2` | 60 | 20 |
| 2-vl8-t2 | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=2` | `neptune-cc: schedule for 'k0' rank=2 shape=[1024,1024] outMin=[1,1] outExtent=[1022,1022] radius=[1,1] elem_bytes=4 split=[64,32] vec=8 par=y unroll=2` | 60 | 30 |
| 2-vl8-t4 | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=4` | `neptune-cc: schedule for 'k0' rank=2 shape=[1024,1024] outMin=[1,1] outExtent=[1022,1022] radius=[1,1] elem_bytes=4 split=[64,32] vec=8 par=y unroll=2` | 60 | 30 |
| 2-vl8-t8 | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=8` | `neptune-cc: schedule for 'k0' rank=2 shape=[1024,1024] outMin=[1,1] outExtent=[1022,1022] radius=[1,1] elem_bytes=4 split=[64,32] vec=8 par=y unroll=2` | 60 | 30 |
