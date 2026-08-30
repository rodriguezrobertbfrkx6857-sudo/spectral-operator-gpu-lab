# Limitations

- The custom extension currently targets `complex64` CUDA tensors and the rank-4 layout used by this project.
- FFT execution remains delegated to `torch.fft`; this project is not an FFT implementation.
- The fused kernel is opt-in and requires independent correctness and performance validation on the target GPU.
- FP32 is the required dtype. FP16 is not enabled because FFT support and error behavior vary across backends.
- The current checkout was executed on CPU-only PyTorch. Therefore its checked-in measurements are CPU fallback data, and CUDA status is `NOT BENCHMARKED ON CURRENT HARDWARE`.
- Mini-FNO is an inference microbenchmark with synthetic smooth fields, not a claim of training quality or dataset accuracy.

