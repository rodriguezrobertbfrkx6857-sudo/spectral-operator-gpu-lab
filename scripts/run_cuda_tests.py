#!/usr/bin/env python3
"""Run CUDA-marked tests and optionally require an actual CUDA test environment."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail when CUDA is unavailable or no CUDA tests are collected",
    )
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        message = "CUDA tests skipped: torch.cuda.is_available() is false"
        print(message)
        return 1 if args.require_cuda else 0

    collect = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", "cuda"],
        check=False,
        text=True,
        capture_output=True,
    )
    print(collect.stdout, end="")
    print(collect.stderr, end="", file=sys.stderr)
    if collect.returncode != 0:
        return collect.returncode
    if "no tests ran" in (collect.stdout or "").lower():
        print("CUDA test gate failed: no CUDA-marked tests were collected")
        return 1

    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "cuda"], check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
