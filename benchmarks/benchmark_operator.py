from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch

from .benchmark_common import (
    custom_cuda_backend,
    device_name,
    measure,
    protocol,
    status_for,
)
from spectral_gpu.operators.spectral_conv_optimized import SpectralConv2dOptimized, backend_name
from spectral_gpu.operators.spectral_conv_reference import SpectralConv2dReference


@contextmanager
def _temporary_env(name: str, value: str | None) -> Iterator[None]:
    original = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


def _error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = (actual - expected).abs()
    return float(difference.max().item()), float(difference.mean().item())


def _record(
    *,
    variant: str,
    shape: list[int],
    modes: list[int],
    device: str,
    backend: str,
    warmup: int,
    iterations: int,
    stats: dict[str, float] | None,
    max_error: float | None,
    mean_error: float | None,
    baseline_variant: str | None,
    baseline_median: float | None,
    note: str,
) -> dict:
    speedup = None
    if (
        baseline_median is not None
        and stats is not None
        and stats["median_ms"] > 0
        and custom_cuda_backend(backend)
    ):
        speedup = baseline_median / stats["median_ms"]
    return {
        "benchmark": "spectral_conv2d",
        "variant": variant,
        "shape": shape,
        "modes": modes,
        "dtype": "float32",
        "device": device_name(device),
        "hardware_mode": device,
        "backend": backend,
        "warmup": warmup,
        "iterations": iterations,
        "median_ms": None if stats is None else stats["median_ms"],
        "mean_ms": None if stats is None else stats["mean_ms"],
        "min_ms": None if stats is None else stats["min_ms"],
        "std_ms": None if stats is None else stats["std_ms"],
        "baseline_variant": baseline_variant,
        "speedup": speedup,
        "max_abs_error_vs_reference": max_error,
        "mean_abs_error_vs_reference": mean_error,
        "status": status_for(device) if stats is not None else "SKIPPED_CUDA_UNAVAILABLE",
        "note": note,
    }


def run(quick: bool = False) -> list[dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    torch.manual_seed(101)
    cases = (
        [(1, 16, 64, 64, 8, 8), (1, 32, 128, 128, 16, 16)]
        if quick
        else [
            (1, 16, 64, 64, 8, 8),
            (4, 16, 64, 128, 8, 16),
            (1, 32, 128, 128, 16, 16),
            (4, 32, 128, 256, 16, 32),
            (1, 16, 256, 256, 32, 32),
        ]
    )
    records: list[dict] = []
    warmup, iterations = protocol(device, quick)
    reference_backend = "torch_reference_cuda" if device == "cuda" else "torch_reference_cpu"
    for batch, channels, height, width, modes1, modes2 in cases:
        shape = [batch, channels, height, width]
        modes = [modes1, modes2]
        reference = SpectralConv2dReference(channels, channels, modes1, modes2, seed=103).to(device)
        optimized = SpectralConv2dOptimized(channels, channels, modes1, modes2, seed=103).to(device)
        optimized.load_state_dict(reference.state_dict())
        x = torch.randn(batch, channels, height, width, device=device)
        with torch.inference_mode():
            expected = reference(x)

        reference_stats, _ = measure(lambda: reference(x), device, warmup, iterations)
        records.append(
            _record(
                variant="pytorch_reference",
                shape=shape,
                modes=modes,
                device=device,
                backend=reference_backend,
                warmup=warmup,
                iterations=iterations,
                stats=reference_stats,
                max_error=0.0,
                mean_error=0.0,
                baseline_variant=None,
                baseline_median=None,
                note="Ground-truth torch.fft reference.",
            )
        )

        with _temporary_env("SPECTRAL_GPU_DISABLE_EXTENSION", "1"), _temporary_env(
            "SPECTRAL_GPU_ENABLE_FUSED", None
        ):
            with torch.inference_mode():
                improved_actual = optimized(x)
                if device == "cuda":
                    torch.cuda.synchronize()
            improved_max_error, improved_mean_error = _error(improved_actual, expected)
            torch.testing.assert_close(improved_actual, expected, rtol=2.0e-5, atol=2.0e-5)
            improved_stats, _ = measure(lambda: optimized(x), device, warmup, iterations)
            records.append(
                _record(
                    variant="pytorch_improved",
                    shape=shape,
                    modes=modes,
                    device=device,
                    backend="torch_improved_cuda" if device == "cuda" else "torch_improved_cpu",
                    warmup=warmup,
                    iterations=iterations,
                    stats=improved_stats,
                    max_error=improved_max_error,
                    mean_error=improved_mean_error,
                    baseline_variant="pytorch_reference",
                    baseline_median=reference_stats["median_ms"],
                    note="Contiguous PyTorch frequency path with extension disabled.",
                )
            )

        with _temporary_env("SPECTRAL_GPU_ENABLE_FUSED", None):
            with torch.inference_mode():
                actual = optimized(x)
                if device == "cuda":
                    torch.cuda.synchronize()
            max_error, mean_error = _error(actual, expected)
            torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)
            optimized_stats, _ = measure(lambda: optimized(x), device, warmup, iterations)
            optimized_backend = backend_name(x)
            records.append(
                _record(
                    variant="optimized_operator",
                    shape=shape,
                    modes=modes,
                    device=device,
                    backend=optimized_backend,
                    warmup=warmup,
                    iterations=iterations,
                    stats=optimized_stats,
                    max_error=max_error,
                    mean_error=mean_error,
                    baseline_variant="pytorch_reference",
                    baseline_median=reference_stats["median_ms"],
                    note=(
                        "Custom complex CUDA contraction when available; otherwise the actual "
                        "PyTorch fallback is reported."
                    ),
                )
            )

        if device == "cuda":
            with _temporary_env("SPECTRAL_GPU_ENABLE_FUSED", "1"):
                with torch.inference_mode():
                    fused_actual = optimized(x)
                    torch.cuda.synchronize()
                fused_max_error, fused_mean_error = _error(fused_actual, expected)
                torch.testing.assert_close(fused_actual, expected, rtol=2.0e-5, atol=2.0e-5)
                fused_stats, _ = measure(lambda: optimized(x), device, warmup, iterations)
                records.append(
                    _record(
                        variant="fused_operator",
                        shape=shape,
                        modes=modes,
                        device=device,
                        backend=backend_name(x),
                        warmup=warmup,
                        iterations=iterations,
                        stats=fused_stats,
                        max_error=fused_max_error,
                        mean_error=fused_mean_error,
                        baseline_variant="pytorch_reference",
                        baseline_median=reference_stats["median_ms"],
                        note="Fused frequency selection and positive/negative contraction.",
                    )
                )
        else:
            records.append(
                _record(
                    variant="fused_operator",
                    shape=shape,
                    modes=modes,
                    device=device,
                    backend="SKIPPED_CUDA_UNAVAILABLE",
                    warmup=warmup,
                    iterations=iterations,
                    stats=None,
                    max_error=None,
                    mean_error=None,
                    baseline_variant=None,
                    baseline_median=None,
                    note="Requires a CUDA-capable NVIDIA device and CUDA-enabled PyTorch.",
                )
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = run(args.quick)
    payload = {"schema_version": 2, "benchmark": "operator", "records": records}
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
