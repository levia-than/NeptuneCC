# 2026-01-30 Halide tuning (split-based schedule)

Host: moster-epyc
Case: case_big2d
Note: schedule logs now report split=[...] instead of tile=[...]

| id | env | schedule | baseline_ms | generated_ms |
| --- | --- | --- | --- | --- |
| 0-default | `` | `neptune-cc: schedule for 'k0' rank=2 shape=[2048,2048] outMin=[1,1] outExtent=[2046,2046] radius=[1,1] elem_bytes=4 split=[64,16] vec=1 par=none unroll=4 (VL defaulted: set NEPTUNECC_VECTOR_WIDTH for target VL)` | 3050 | 37380 |
| 1-vl4 | `NEPTUNECC_VECTOR_WIDTH=4 NEPTUNECC_THREADS=1` | `neptune-cc: schedule for 'k0' rank=2 shape=[2048,2048] outMin=[1,1] outExtent=[2046,2046] radius=[1,1] elem_bytes=4 split=[64,32] vec=4 par=none unroll=4` | 3060 | 6760 |
| 1-vl8 | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=1` | `neptune-cc: schedule for 'k0' rank=2 shape=[2048,2048] outMin=[1,1] outExtent=[2046,2046] radius=[1,1] elem_bytes=4 split=[64,32] vec=8 par=none unroll=2` | 3100 | 6600 |
| 1-vl16 | `NEPTUNECC_VECTOR_WIDTH=16 NEPTUNECC_THREADS=1` | `neptune-cc: schedule for 'k0' rank=2 shape=[2048,2048] outMin=[1,1] outExtent=[2046,2046] radius=[1,1] elem_bytes=4 split=[32,64] vec=16 par=none unroll=2` | 3070 | 5420 |
| 2-vl8-t2 | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=2` | `neptune-cc: schedule for 'k0' rank=2 shape=[2048,2048] outMin=[1,1] outExtent=[2046,2046] radius=[1,1] elem_bytes=4 split=[64,32] vec=8 par=y unroll=2` | 3950 | 1100 |
| 2-vl8-t4 | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=4` | `neptune-cc: schedule for 'k0' rank=2 shape=[2048,2048] outMin=[1,1] outExtent=[2046,2046] radius=[1,1] elem_bytes=4 split=[64,32] vec=8 par=y unroll=2` | 3650 | 900 |
| 2-vl8-t8 | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=8` | `neptune-cc: schedule for 'k0' rank=2 shape=[2048,2048] outMin=[1,1] outExtent=[2046,2046] radius=[1,1] elem_bytes=4 split=[64,32] vec=8 par=y unroll=2` | 3230 | 950 |
| 3-l1-16k | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=4 NEPTUNECC_L1_BYTES=16384` | `neptune-cc: schedule for 'k0' rank=2 shape=[2048,2048] outMin=[1,1] outExtent=[2046,2046] radius=[1,1] elem_bytes=4 split=[32,32] vec=8 par=y unroll=2` | 3760 | 680 |
| 3-l1-32k | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=4 NEPTUNECC_L1_BYTES=32768` | `neptune-cc: schedule for 'k0' rank=2 shape=[2048,2048] outMin=[1,1] outExtent=[2046,2046] radius=[1,1] elem_bytes=4 split=[64,32] vec=8 par=y unroll=2` | 4000 | 760 |
| 3-l1-64k | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=4 NEPTUNECC_L1_BYTES=65536` | `neptune-cc: schedule for 'k0' rank=2 shape=[2048,2048] outMin=[1,1] outExtent=[2046,2046] radius=[1,1] elem_bytes=4 split=[64,64] vec=8 par=y unroll=2` | 4110 | 650 |
| 4-alpha-0.4 | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=4 NEPTUNECC_L1_BYTES=32768 NEPTUNECC_CACHE_ALPHA=0.4` | `neptune-cc: schedule for 'k0' rank=2 shape=[2048,2048] outMin=[1,1] outExtent=[2046,2046] radius=[1,1] elem_bytes=4 split=[32,32] vec=8 par=y unroll=2` | 3990 | 630 |
| 4-alpha-0.6 | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=4 NEPTUNECC_L1_BYTES=32768 NEPTUNECC_CACHE_ALPHA=0.6` | `neptune-cc: schedule for 'k0' rank=2 shape=[2048,2048] outMin=[1,1] outExtent=[2046,2046] radius=[1,1] elem_bytes=4 split=[64,32] vec=8 par=y unroll=2` | 3700 | 760 |
| 4-alpha-0.8 | `NEPTUNECC_VECTOR_WIDTH=8 NEPTUNECC_THREADS=4 NEPTUNECC_L1_BYTES=32768 NEPTUNECC_CACHE_ALPHA=0.8` | `neptune-cc: schedule for 'k0' rank=2 shape=[2048,2048] outMin=[1,1] outExtent=[2046,2046] radius=[1,1] elem_bytes=4 split=[64,32] vec=8 par=y unroll=2` | 3990 | 640 |
