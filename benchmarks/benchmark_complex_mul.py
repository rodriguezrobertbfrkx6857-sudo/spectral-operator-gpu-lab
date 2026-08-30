from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .benchmark_common import custom_cuda_backend, device_name, measure, protocol, status_for
from spectral_gpu.operators.complex_multiply import (
    backend_name,
    complex_multiply_optimized,
    complex_multiply_reference,
)


def run(quick: bool = False) -> list[dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    cases = (
        [(1, 16, 16, 8, 8), (4, 32, 32, 16, 16)]
        if quick
        else [(1, 16, 16, 8, 8), (4, 32, 32, 16, 32)]
    )
    warmup, iterations = protocol(device, quick)
    records: list[dict] = []
    reference_backend = "torch_reference_cuda" if device == "cuda" else "torch_reference_cpu"
    for batch, input_channels, output_channels, modes1, modes2 in cases:
        generator = torch.Generator(device="cpu").manual_seed(107 + batch + modes1)
        input_ft = torch.randn(
            batch,
            input_channels,
            modes1,
            modes2,
            dtype=torch.complex64,
            generator=generator,
        ).to(device)
        weight = torch.randn(
            input_channels,
            output_channels,
            modes1,
            modes2,
            dtype=torch.complex64,
            generator=generator,
        ).to(device)
        with torch.inference_mode():
            expected = complex_multiply_reference(input_ft, weight)
            actual = complex_multiply_optimized(input_ft, weight)
            if device == "cuda":
                torch.cuda.synchronize()
        error = (actual - expected).abs()
        max_error = float(error.max().item())
        mean_error = float(error.mean().item())
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-5)
        reference_stats, _ = measure(
            lambda: complex_multiply_reference(input_ft, weight), device, warmup, iterations
        )
        optimized_stats, _ = measure(
            lambda: complex_multiply_optimized(input_ft, weight), device, warmup, iterations
        )
        optimized_backend = backend_name(input_ft)
        speedup = (
            reference_stats["median_ms"] / optimized_stats["median_ms"]
            if custom_cuda_backend(optimized_backend) and optimized_stats["median_ms"] > 0
            else None
        )
        records.append(
            {
                "benchmark": "complex_multiply",
                "shape": [batch, input_channels, output_channels, modes1, modes2],
                "dtype": "complex64",
                "device": device_name(device),
                "hardware_mode": device,
                "warmup": warmup,
                "iterations": iterations,
                "reference": {
                    **reference_stats,
                    "variant": "pytorch_reference",
                    "backend": reference_backend,
                    "status": status_for(device),
                },
                "optimized": {
                    **optimized_stats,
                    "variant": "optimized_complex_multiply",
                    "backend": optimized_backend,
                    "status": status_for(device),
                },
                "speedup": speedup,
                "max_abs_error": max_error,
                "mean_abs_error": mean_error,
                "note": (
                    "Speedup is emitted only for the custom CUDA backend; fallback timings remain "
                    "labelled as fallback execution."
                ),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {"schema_version": 2, "benchmark": "complex_multiply", "records": run(args.quick)}
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
