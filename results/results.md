# Operator Benchmark Results

Generated from JSON emitted by the benchmark programs.

- Hardware mode: `cpu_only`
- Correctness is checked before every measured pair.
- CUDA timing uses CUDA Events; CPU fallback timing uses `perf_counter_ns`.

## operator

- `pytorch_reference` shape `[1, 16, 64, 64]`, modes `[8, 8]`: median `0.855150 ms`, speedup `1.000x`, max error `0.000e+00`.
- `optimized_operator` shape `[1, 16, 64, 64]`, modes `[8, 8]`: median `0.437550 ms`, speedup `1.954x`, max error `0.000e+00`.
- `pytorch_reference` shape `[4, 16, 64, 128]`, modes `[8, 16]`: median `3.381500 ms`, speedup `1.000x`, max error `0.000e+00`.
- `optimized_operator` shape `[4, 16, 64, 128]`, modes `[8, 16]`: median `3.142250 ms`, speedup `1.076x`, max error `0.000e+00`.
- `pytorch_reference` shape `[1, 32, 128, 128]`, modes `[16, 16]`: median `4.767300 ms`, speedup `1.000x`, max error `0.000e+00`.
- `optimized_operator` shape `[1, 32, 128, 128]`, modes `[16, 16]`: median `5.116900 ms`, speedup `0.932x`, max error `0.000e+00`.
- `pytorch_reference` shape `[4, 32, 128, 256]`, modes `[16, 32]`: median `33.377450 ms`, speedup `1.000x`, max error `0.000e+00`.
- `optimized_operator` shape `[4, 32, 128, 256]`, modes `[16, 32]`: median `55.293350 ms`, speedup `0.604x`, max error `0.000e+00`.
- `pytorch_reference` shape `[1, 16, 256, 256]`, modes `[32, 32]`: median `12.459650 ms`, speedup `1.000x`, max error `0.000e+00`.
- `optimized_operator` shape `[1, 16, 256, 256]`, modes `[32, 32]`: median `11.732250 ms`, speedup `1.062x`, max error `0.000e+00`.

## complex_multiply

- shape `[1, 16, 16, 8, 8]`: reference median `0.043350 ms`, optimized median `0.046650 ms`, speedup `0.929x`, max error `0.000e+00`.
- shape `[4, 32, 32, 16, 32]`: reference median `4.611750 ms`, optimized median `4.131100 ms`, speedup `1.116x`, max error `0.000e+00`.

## mini_fno

- shape `[1, 1, 64, 64]`, modes `[8, 8]`: reference median `0.932900 ms`, optimized median `1.033950 ms`, speedup `0.902x`, max error `0.000e+00`.
- shape `[1, 1, 128, 128]`, modes `[16, 16]`: reference median `1.884500 ms`, optimized median `1.899250 ms`, speedup `0.992x`, max error `0.000e+00`.

CUDA status: `NOT BENCHMARKED ON CURRENT HARDWARE`.
All numbers in this report are CPU measurements from the PyTorch fallback.
