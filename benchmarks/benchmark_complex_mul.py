from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .benchmark_common import measure, protocol
from spectral_gpu.operators.complex_multiply import complex_multiply_optimized, complex_multiply_reference


def run(quick: bool = False) -> list[dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    cases = [(1, 16, 16, 8, 8), (4, 32, 32, 16, 16)] if quick else [(1, 16, 16, 8, 8), (4, 32, 32, 16, 32)]
    warmup, iterations = protocol(device, quick)
    records: list[dict] = []
    for batch, input_channels, output_channels, modes1, modes2 in cases:
        generator = torch.Generator(device="cpu").manual_seed(107 + batch + modes1)
        input_ft = torch.randn(batch, input_channels, modes1, modes2, dtype=torch.complex64, generator=generator).to(device)
        weight = torch.randn(input_channels, output_channels, modes1, modes2, dtype=torch.complex64, generator=generator).to(device)
        expected = complex_multiply_reference(input_ft, weight)
        actual = complex_multiply_optimized(input_ft, weight)
        if device == "cuda":
            torch.cuda.synchronize()
        max_error = float((actual - expected).abs().max().item())
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-5)
        reference_stats, _ = measure(lambda: complex_multiply_reference(input_ft, weight), device, warmup, iterations)
        optimized_stats, _ = measure(lambda: complex_multiply_optimized(input_ft, weight), device, warmup, iterations)
        records.append(
            {
                "benchmark": "complex_multiply",
                "shape": [batch, input_channels, output_channels, modes1, modes2],
                "hardware_mode": device,
                "reference": reference_stats,
                "optimized": optimized_stats,
                "speedup": reference_stats["median_ms"] / optimized_stats["median_ms"],
                "max_abs_error": max_error,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {"benchmark": "complex_multiply", "records": run(args.quick)}
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
