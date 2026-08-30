from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .benchmark_common import measure, protocol
from spectral_gpu.operators.spectral_conv_optimized import SpectralConv2dOptimized
from spectral_gpu.operators.spectral_conv_reference import SpectralConv2dReference


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
    for batch, channels, height, width, modes1, modes2 in cases:
        reference = SpectralConv2dReference(channels, channels, modes1, modes2, seed=103).to(device)
        optimized = SpectralConv2dOptimized(channels, channels, modes1, modes2, seed=103).to(device)
        optimized.load_state_dict(reference.state_dict())
        x = torch.randn(batch, channels, height, width, device=device)
        with torch.no_grad():
            expected = reference(x)
            actual = optimized(x)
            if device == "cuda":
                torch.cuda.synchronize()
            max_error = float((actual - expected).abs().max().item())
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)
        reference_stats, _ = measure(lambda: reference(x), device, warmup, iterations)
        optimized_stats, _ = measure(lambda: optimized(x), device, warmup, iterations)
        records.extend(
            [
                {
                    "benchmark": "spectral_conv2d",
                    "variant": "pytorch_reference",
                    "shape": [batch, channels, height, width],
                    "modes": [modes1, modes2],
                    "hardware_mode": device,
                    **reference_stats,
                    "speedup": 1.0,
                    "max_abs_error_vs_reference": 0.0,
                    "backend": "torch.fft + torch.einsum",
                },
                {
                    "benchmark": "spectral_conv2d",
                    "variant": "optimized_operator",
                    "shape": [batch, channels, height, width],
                    "modes": [modes1, modes2],
                    "hardware_mode": device,
                    **optimized_stats,
                    "speedup": reference_stats["median_ms"] / optimized_stats["median_ms"],
                    "max_abs_error_vs_reference": max_error,
                    "backend": "custom CUDA complex multiply when available; PyTorch fallback otherwise",
                },
            ]
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = run(args.quick)
    payload = {"benchmark": "operator", "records": records}
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
