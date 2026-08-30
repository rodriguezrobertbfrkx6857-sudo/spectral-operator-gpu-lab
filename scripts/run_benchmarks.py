#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _run(module: str, output: Path, quick: bool, root: Path, env: dict[str, str]) -> dict:
    command = [sys.executable, "-m", module, "--output", str(output)]
    if quick:
        command.append("--quick")
    subprocess.run(command, check=True, cwd=root, env=env)
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--require-extension",
        action="store_true",
        help="require a CUDA-capable PyTorch runtime and successful custom extension compilation",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    results.mkdir(exist_ok=True)
    env = os.environ.copy()
    if args.require_extension:
        import torch

        if not torch.cuda.is_available() or torch.version.cuda is None:
            raise RuntimeError("--require-extension needs a CUDA-capable NVIDIA PyTorch runtime")
        env["SPECTRAL_GPU_REQUIRE_EXTENSION"] = "1"

    subprocess.run(
        [sys.executable, str(root / "scripts" / "detect_environment.py"), "--output-dir", str(results)],
        check=True,
        cwd=root,
        env=env,
    )
    payloads = [
        _run("benchmarks.benchmark_operator", results / "operator.json", args.quick, root, env),
        _run("benchmarks.benchmark_complex_mul", results / "complex_mul.json", args.quick, root, env),
        _run("benchmarks.benchmark_fno", results / "fno.json", args.quick, root, env),
    ]
    environment = json.loads((results / "environment.json").read_text(encoding="utf-8"))
    combined = {
        "schema_version": 2,
        "generated_by": "scripts/run_benchmarks.py",
        "hardware_mode": environment["hardware_mode"],
        "benchmarks": payloads,
    }
    (results / "operator_results.json").write_text(
        json.dumps(combined, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
