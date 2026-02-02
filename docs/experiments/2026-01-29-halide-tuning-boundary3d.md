# Halide Tuning Experiments (case_boundary3d)

Date: 2026-01-29

Scope: experiments 0/1/2 on test/case_test/case_boundary3d. No git commit.

## Commands

- make -C test/case_test/case_boundary3d clean
- <env> make -C test/case_test/case_boundary3d compare
- <env> make -C test/case_test/case_boundary3d time

## Results

| Experiment | Env | Schedule summary | Baseline elapsed_ms | Generated elapsed_ms |
|---|---|---|---:|---:|
| 0-default | `` | `neptune-cc: schedule for 'k0' rank=3 shape=[192,192,192] outMin=[1,1,1] outExtent=[190,190,190] radius=[1,1,1] elem_bytes=4 split=[8,16,8] vec=1 par=none unroll=2 (VL defaulted: set NEPTUNECC_VECTOR_WIDTH for target VL)` | 6570 | 6340 |
| 1-vl4 | `NEPTUNECC_VECTOR_WIDTH=4 NEPTUNECC_THREADS=1` | `neptune-cc: schedule for 'k0' rank=3 shape=[192,192,192] outMin=[1,1,1] outExtent=[190,190,190] radius=[1,1,1] elem_bytes=4 split=[8,8,16] vec=4 par=none unroll=2` | 6580 | 5010 |
| 1-vl8 | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=1` | `neptune-cc: schedule for 'k0' rank=3 shape=[192,192,192] outMin=[1,1,1] outExtent=[190,190,190] radius=[1,1,1] elem_bytes=4 split=[8,4,32] vec=8 par=none unroll=2` | 6590 | 2840 |
| 1-vl16 | `NEPTUNECC_VECTOR_WIDTH=16 NEPTUNECC_THREADS=1` | `neptune-cc: schedule for 'k0' rank=3 shape=[192,192,192] outMin=[1,1,1] outExtent=[190,190,190] radius=[1,1,1] elem_bytes=4 split=[4,4,64] vec=16 par=none unroll=2` | 6610 | 2410 |
| 2-vl8-t2 | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=2` | `neptune-cc: schedule for 'k0' rank=3 shape=[192,192,192] outMin=[1,1,1] outExtent=[190,190,190] radius=[1,1,1] elem_bytes=4 split=[8,4,32] vec=8 par=z unroll=2` | 6590 | 770 |
| 2-vl8-t4 | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=4` | `neptune-cc: schedule for 'k0' rank=3 shape=[192,192,192] outMin=[1,1,1] outExtent=[190,190,190] radius=[1,1,1] elem_bytes=4 split=[8,4,32] vec=8 par=z unroll=2` | 6570 | 850 |
| 2-vl8-t8 | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=8` | `neptune-cc: schedule for 'k0' rank=3 shape=[192,192,192] outMin=[1,1,1] outExtent=[190,190,190] radius=[1,1,1] elem_bytes=4 split=[8,4,32] vec=8 par=z unroll=2` | 6570 | 860 |
| 3-thread8-defaultVL | `HL_NUM_THREADS=8 OMP_NUM_THREADS=8 NEPTUNECC_THREADS=8` | `neptune-cc: schedule for 'k0' rank=3 shape=[192,192,192] outMin=[1,1,1] outExtent=[190,190,190] radius=[1,1,1] elem_bytes=4 split=[8,16,8] vec=1 par=z unroll=2 (VL defaulted: set NEPTUNECC_VECTOR_WIDTH for target VL)` | 6590 | 1370 |
