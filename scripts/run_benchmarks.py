#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from detect_environment import main as detect_environment_main


def _run(module: str, output: Path, quick: bool) -> dict:
    command = [sys.executable, "-m", module, "--output", str(output)]
    if quick:
        command.append("--quick")
    subprocess.run(command, check=True)
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    results.mkdir(exist_ok=True)
    original_argv = sys.argv
    sys.argv = [str(root / "scripts" / "detect_environment.py"), "--output-dir", str(results)]
    detect_environment_main()
    sys.argv = original_argv
    payloads = [
        _run("benchmarks.benchmark_operator", results / "operator.json", args.quick),
        _run("benchmarks.benchmark_complex_mul", results / "complex_mul.json", args.quick),
        _run("benchmarks.benchmark_fno", results / "fno.json", args.quick),
    ]
    combined = {"schema_version": 1, "generated_by": "scripts/run_benchmarks.py", "benchmarks": payloads}
    (results / "operator_results.json").write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()

