"""Small profiling entry point; use Nsight externally on CUDA hardware."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectral_gpu.models.mini_fno import MiniFNO
from spectral_gpu.models.synthetic_data import make_smooth_fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=128)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniFNO(width=8, modes1=min(16, args.side // 2), modes2=min(16, args.side // 2), optimized=True).to(device)
    inputs = make_smooth_fields(1, args.side, args.side, device=device)
    with torch.no_grad():
        for _ in range(20):
            model(inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        output = model(inputs)
        if device == "cuda":
            torch.cuda.synchronize()
    print({"device": device, "output_shape": list(output.shape), "status": "ready for external profiler"})


if __name__ == "__main__":
    main()
