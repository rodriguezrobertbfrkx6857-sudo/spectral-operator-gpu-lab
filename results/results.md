# Operator Benchmark Results

Generated from JSON emitted by the benchmark programs; timing values are not maintained by hand.

- Hardware mode: `cpu_only`
- Correctness is checked before every measured pair.
- CUDA timing uses CUDA Events; CPU fallback timing uses `perf_counter_ns`.
- A custom-CUDA speedup is emitted only when the recorded optimized backend is custom CUDA.

## operator

- `pytorch_reference` shape `[1, 16, 64, 64]`, modes `[8, 8]`, backend `torch_reference_cpu`, status `BENCHMARKED_CPU_ONLY`: median `0.357050 ms`, speedup `n/a`, max/mean error `0.000000`/`0.000000`.
- `pytorch_improved` shape `[1, 16, 64, 64]`, modes `[8, 8]`, backend `torch_improved_cpu`, status `BENCHMARKED_CPU_ONLY`: median `0.411000 ms`, speedup `n/a`, max/mean error `0.000000`/`0.000000`.
- `optimized_operator` shape `[1, 16, 64, 64]`, modes `[8, 8]`, backend `torch_improved_cpu`, status `BENCHMARKED_CPU_ONLY`: median `0.355300 ms`, speedup `n/a`, max/mean error `0.000000`/`0.000000`.
- `fused_operator` shape `[1, 16, 64, 64]`, modes `[8, 8]`, backend `SKIPPED_CUDA_UNAVAILABLE`, status `SKIPPED_CUDA_UNAVAILABLE`: median `n/a`, speedup `n/a`, max/mean error `n/a`/`n/a`.
- `pytorch_reference` shape `[4, 16, 64, 128]`, modes `[8, 16]`, backend `torch_reference_cpu`, status `BENCHMARKED_CPU_ONLY`: median `2.745600 ms`, speedup `n/a`, max/mean error `0.000000`/`0.000000`.
- `pytorch_improved` shape `[4, 16, 64, 128]`, modes `[8, 16]`, backend `torch_improved_cpu`, status `BENCHMARKED_CPU_ONLY`: median `3.007000 ms`, speedup `n/a`, max/mean error `0.000000`/`0.000000`.
- `optimized_operator` shape `[4, 16, 64, 128]`, modes `[8, 16]`, backend `torch_improved_cpu`, status `BENCHMARKED_CPU_ONLY`: median `3.153550 ms`, speedup `n/a`, max/mean error `0.000000`/`0.000000`.
- `fused_operator` shape `[4, 16, 64, 128]`, modes `[8, 16]`, backend `SKIPPED_CUDA_UNAVAILABLE`, status `SKIPPED_CUDA_UNAVAILABLE`: median `n/a`, speedup `n/a`, max/mean error `n/a`/`n/a`.
- `pytorch_reference` shape `[1, 32, 128, 128]`, modes `[16, 16]`, backend `torch_reference_cpu`, status `BENCHMARKED_CPU_ONLY`: median `7.180200 ms`, speedup `n/a`, max/mean error `0.000000`/`0.000000`.
- `pytorch_improved` shape `[1, 32, 128, 128]`, modes `[16, 16]`, backend `torch_improved_cpu`, status `BENCHMARKED_CPU_ONLY`: median `6.750950 ms`, speedup `n/a`, max/mean error `0.000000`/`0.000000`.
- `optimized_operator` shape `[1, 32, 128, 128]`, modes `[16, 16]`, backend `torch_improved_cpu`, status `BENCHMARKED_CPU_ONLY`: median `7.358900 ms`, speedup `n/a`, max/mean error `0.000000`/`0.000000`.
- `fused_operator` shape `[1, 32, 128, 128]`, modes `[16, 16]`, backend `SKIPPED_CUDA_UNAVAILABLE`, status `SKIPPED_CUDA_UNAVAILABLE`: median `n/a`, speedup `n/a`, max/mean error `n/a`/`n/a`.
- `pytorch_reference` shape `[4, 32, 128, 256]`, modes `[16, 32]`, backend `torch_reference_cpu`, status `BENCHMARKED_CPU_ONLY`: median `53.634950 ms`, speedup `n/a`, max/mean error `0.000000`/`0.000000`.
- `pytorch_improved` shape `[4, 32, 128, 256]`, modes `[16, 32]`, backend `torch_improved_cpu`, status `BENCHMARKED_CPU_ONLY`: median `50.062650 ms`, speedup `n/a`, max/mean error `0.000000`/`0.000000`.
- `optimized_operator` shape `[4, 32, 128, 256]`, modes `[16, 32]`, backend `torch_improved_cpu`, status `BENCHMARKED_CPU_ONLY`: median `61.809750 ms`, speedup `n/a`, max/mean error `0.000000`/`0.000000`.
- `fused_operator` shape `[4, 32, 128, 256]`, modes `[16, 32]`, backend `SKIPPED_CUDA_UNAVAILABLE`, status `SKIPPED_CUDA_UNAVAILABLE`: median `n/a`, speedup `n/a`, max/mean error `n/a`/`n/a`.
- `pytorch_reference` shape `[1, 16, 256, 256]`, modes `[32, 32]`, backend `torch_reference_cpu`, status `BENCHMARKED_CPU_ONLY`: median `10.716450 ms`, speedup `n/a`, max/mean error `0.000000`/`0.000000`.
- `pytorch_improved` shape `[1, 16, 256, 256]`, modes `[32, 32]`, backend `torch_improved_cpu`, status `BENCHMARKED_CPU_ONLY`: median `10.738550 ms`, speedup `n/a`, max/mean error `0.000000`/`0.000000`.
- `optimized_operator` shape `[1, 16, 256, 256]`, modes `[32, 32]`, backend `torch_improved_cpu`, status `BENCHMARKED_CPU_ONLY`: median `16.726550 ms`, speedup `n/a`, max/mean error `0.000000`/`0.000000`.
- `fused_operator` shape `[1, 16, 256, 256]`, modes `[32, 32]`, backend `SKIPPED_CUDA_UNAVAILABLE`, status `SKIPPED_CUDA_UNAVAILABLE`: median `n/a`, speedup `n/a`, max/mean error `n/a`/`n/a`.

## complex_multiply

- shape `[1, 16, 16, 8, 8]`, optimized backend `torch_improved_cpu`, status `BENCHMARKED_CPU_ONLY`: reference median `0.040900 ms`, optimized median `0.100000 ms`, speedup `n/a`, max/mean error `0.000000`/`0.000000`.
- shape `[4, 32, 32, 16, 32]`, optimized backend `torch_improved_cpu`, status `BENCHMARKED_CPU_ONLY`: reference median `5.132600 ms`, optimized median `5.678950 ms`, speedup `n/a`, max/mean error `0.000000`/`0.000000`.

## mini_fno

- shape `[1, 1, 64, 64]`, modes `[8, 8]`, optimized backend `torch_improved_cpu`, status `BENCHMARKED_CPU_ONLY`: reference median `0.797750 ms`, optimized median `0.835550 ms`, speedup `n/a`, max/mean error `0.000000`/`0.000000`.
- shape `[1, 1, 128, 128]`, modes `[16, 16]`, optimized backend `torch_improved_cpu`, status `BENCHMARKED_CPU_ONLY`: reference median `1.925050 ms`, optimized median `1.979950 ms`, speedup `n/a`, max/mean error `0.000000`/`0.000000`.
- shape `[1, 1, 256, 256]`, modes `[32, 32]`, optimized backend `torch_improved_cpu`, status `BENCHMARKED_CPU_ONLY`: reference median `6.107150 ms`, optimized median `6.302250 ms`, speedup `n/a`, max/mean error `0.000000`/`0.000000`.

## Environment

- OS: Windows 11
- Python: `3.13.11`
- PyTorch: `2.9.1+cpu`
- NVIDIA devices: `0`

CUDA status: `NOT BENCHMARKED ON CURRENT HARDWARE`.
All measured timings in this report are CPU measurements from the PyTorch fallback.
The fused CUDA path is listed as skipped rather than represented by a CPU substitute.
