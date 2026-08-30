# spectral-operator-gpu-lab

GPU optimization experiments for spectral operators and Fourier Neural Operators with custom CUDA kernels.

This project isolates the expensive frequency-domain contraction in a small `SpectralConv2d` and compares a PyTorch reference, a cleaned-up PyTorch path, and an optional PyTorch CUDA extension. It also includes a Mini-FNO inference path and deterministic synthetic smooth fields. FFT transforms remain delegated to `torch.fft`; the custom code focuses on complex multiplication and frequency selection.

The current checked-in run uses `torch 2.9.1+cpu` on a host with no NVIDIA driver or CUDA Toolkit. Its timings are real CPU fallback measurements only. CUDA results are never inferred from them.

## Project overview

- V0 reference: `torch.fft.rfft2` → low-frequency selection → complex `einsum` → `torch.fft.irfft2`.
- V1 improved PyTorch path: explicit contiguous slices and controlled output allocation.
- V2 custom CUDA extension: complex64 channel contraction for the actual rank-4 operator layout.
- V3 opt-in fused kernel: frequency selection plus complex multiplication in one CUDA kernel.
- Mini-FNO: lift, two spectral blocks, and projection over synthetic smooth fields.

## Quick start

Use an environment with PyTorch, NumPy, and pytest:

```powershell
python -m pytest
python scripts/run_benchmarks.py --quick
python scripts/generate_report.py
```

The default test matrix covers batch sizes 1 and 4, channels 16 and 32, spatial sizes through 256×256, and modes 8, 16, and 32. The benchmark runner uses CUDA Events with warm-up and repeated measurements on CUDA, and `perf_counter_ns` with adaptive counts on CPU.

## CUDA extension

The extension is lazy-loaded only when `torch.cuda.is_available()` and `CUDA_HOME` are both present. It can be disabled with `SPECTRAL_GPU_DISABLE_EXTENSION=1`. The default optimized layer uses the custom complex contraction when it loads successfully; the fused frequency path requires `SPECTRAL_GPU_ENABLE_FUSED=1`.

The source is [spectral_gpu/cuda/complex_mul_kernel.cu](spectral_gpu/cuda/complex_mul_kernel.cu), with bindings in [spectral_gpu/cuda/bindings.cpp](spectral_gpu/cuda/bindings.cpp). No API key or external service is required.

## Results

The generated result is [results/results.md](results/results.md), with machine-readable data in [results/operator_results.json](results/operator_results.json). Each record includes the exact shape, modes, median/mean/min/std timing, speedup, and maximum error against the reference. The report distinguishes `cpu` from `cuda`.

## Documentation

- [docs/spectral_conv.md](docs/spectral_conv.md): operator data flow and frequency layout.
- [docs/optimization_notes.md](docs/optimization_notes.md): optimization boundary and fusion policy.
- [docs/limitations.md](docs/limitations.md): supported dtype/layout and honest scope.
- [profiling/README.md](profiling/README.md): external Nsight workflow.

## Reproducibility

```powershell
python scripts/detect_environment.py --output-dir results
python scripts/run_benchmarks.py --quick
python scripts/generate_report.py
```

All model weights and synthetic fields use explicit seeds. GPU benchmarks must be reproduced on CUDA-capable hardware. The current checkout reports `NOT BENCHMARKED ON CURRENT HARDWARE` for CUDA because that hardware was not present.

