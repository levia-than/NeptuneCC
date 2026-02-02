# Halide Tuning Experiments (case_boundary2d)

Date: 2026-01-30

Scope: experiment on test/case_test/case_boundary2d. No git commit.

## Commands

- make -C test/case_test/case_boundary2d clean
- HL_NUM_THREADS=8 OMP_NUM_THREADS=8 NEPTUNECC_THREADS=8 make -C test/case_test/case_boundary2d compare
- (time) baseline and generated:
  - cd test/case_test/case_boundary2d/out && time -p ./baseline
  - cd test/case_test/case_boundary2d/out && time -p ./generated

## Results

| Experiment | Env | Schedule summary | Baseline elapsed_ms | Generated elapsed_ms | Notes |
|---|---|---|---:|---:|---|
| 0-thread8-defaultVL | `HL_NUM_THREADS=8 OMP_NUM_THREADS=8 NEPTUNECC_THREADS=8` | `neptune-cc: schedule for 'k0' rank=2 shape=[1024,1024] outMin=[1,1] outExtent=[1022,1022] radius=[1,1] elem_bytes=4 split=[64,16] vec=1 par=y unroll=4 (VL defaulted: set NEPTUNECC_VECTOR_WIDTH for target VL)` | 60 | 40 | runtime still short; consider increasing H/W or loop iterations to reach 10s+ |
