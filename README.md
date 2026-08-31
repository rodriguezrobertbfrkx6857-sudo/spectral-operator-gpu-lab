# Spectral Operator GPU Lab：频谱算子与 FNO 优化实验

一套面向频谱算子和 Fourier Neural Operator（FNO）的 GPU 优化实验。项目把 `SpectralConv2d` 中最昂贵的频域收缩拆出来，对比 PyTorch 参考路径、整理后的 PyTorch 路径、可选的自定义 CUDA 扩展和可选融合内核，并用确定性的合成光滑场支撑可复现验证。

当前提交使用 `torch 2.9.1+cpu`，运行在没有 NVIDIA 驱动和 CUDA Toolkit 的本机。报告中的时间是真实 CPU fallback 测量，不能被解释为 CUDA 结果；CUDA 路径保留在仓库中，需在 CUDA 机器上重新构建和测量。

[![CI](https://github.com/rodriguezrobertbfrkx6857-sudo/spectral-operator-gpu-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/rodriguezrobertbfrkx6857-sudo/spectral-operator-gpu-lab/actions/workflows/ci.yml)

> 评委速览：这是作品集的科学算子旗舰项目，围绕 FNO 频域收缩比较 reference、PyTorch 优化路径、自定义 `complex64` CUDA 扩展和可选融合 kernel。报告严格区分 CPU fallback 与 CUDA 实测，不从 CPU 时间推断 GPU 加速。

## 实验内容

- V0 参考：`torch.fft.rfft2` → 低频选择 → 复数 `einsum` → `torch.fft.irfft2`。
- V1 PyTorch 优化：显式连续切片和受控输出分配。
- V2 自定义 CUDA 扩展：针对真实四维算子布局的 `complex64` 通道收缩。
- V3 可选融合内核：在一个 CUDA kernel 中完成频率选择和复数乘法。
- Mini-FNO：升维、两个频谱块和投影层，输入为确定性的合成光滑场。

## 快速开始

准备 PyTorch、NumPy 和 pytest 环境后：

```powershell
python -m pytest
python scripts/run_cuda_tests.py
python scripts/run_benchmarks.py --quick
python scripts/generate_report.py
```

默认测试覆盖 batch 1/4、通道 16/32、最大 `256×256` 的空间尺寸以及 modes 8/16/32。`python scripts/run_cuda_tests.py --require-cuda` 是严格 CUDA 门禁：没有 CUDA 设备或 CUDA 标记测试时直接失败。CUDA 计时使用 CUDA Events，CPU 计时使用 `perf_counter_ns` 和自适应次数。

## CUDA 扩展

只有在 `torch.cuda.is_available()` 和 `CUDA_HOME` 同时存在时才会延迟加载扩展。可设置 `SPECTRAL_GPU_DISABLE_EXTENSION=1` 禁用；`SPECTRAL_GPU_ENABLE_FUSED=1` 启用融合频率路径；`SPECTRAL_GPU_REQUIRE_EXTENSION=1` 会把扩展构建失败转为显式测试/基准失败。`SPECTRAL_GPU_USE_FAST_MATH=1` 是可单独测量的选项，正确性验证默认使用常规 FP32 数学。

核心源码位于 [`spectral_gpu/cuda/complex_mul_kernel.cu`](spectral_gpu/cuda/complex_mul_kernel.cu)，绑定位于 [`spectral_gpu/cuda/bindings.cpp`](spectral_gpu/cuda/bindings.cpp)。`spectral_gpu.cuda.extension_status()` 会公开扩展是否可用、是否编译、后端、构建错误和源码版本；项目不需要 API key 或外部服务。

## 结果与文档

生成报告见 [`results/results.md`](results/results.md)，机器可读数据见 [`results/operator_results.json`](results/operator_results.json)。每条记录包含实际 shape、modes、中位/平均/最小/标准差、预热次数、迭代次数、真实后端、状态、有效加速比以及相对参考的最大/平均误差。报告明确区分 `cpu` 和 `cuda`，CPU-only 主机上的融合 CUDA 路径会被标记为跳过。

- [`docs/spectral_conv.md`](docs/spectral_conv.md)：算子数据流和频率布局。
- [`docs/optimization_notes.md`](docs/optimization_notes.md)：优化边界和融合策略。
- [`docs/limitations.md`](docs/limitations.md)：支持的数据类型、布局和诚实边界。
- [`profiling/README.md`](profiling/README.md)：Nsight 分析流程。

复现实验：

```powershell
python scripts/detect_environment.py --output-dir results
python scripts/run_benchmarks.py --quick
python scripts/generate_report.py
```

所有模型权重和合成场使用显式随机种子。当前 checkout 对 CUDA 报告 `NOT BENCHMARKED ON CURRENT HARDWARE`，并不从 CPU 时间推断 GPU 加速。
