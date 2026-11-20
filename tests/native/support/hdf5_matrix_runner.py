#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-2-Clause
"""Run the parallel-HDF5 probe at several MPI rank counts.

The probe writes and reads back one dataset collectively while all ranks but
the first two own no data. Before the launches, the runner checks that the
source and generated logical HDF5 layout descriptions agree, list the
compiled-problem, result and checkpoint layouts and declare the zero-owner
collective discipline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

__all__ = ["main"]


def main() -> int:
    """Check the layout descriptions, run every rank count; return the exit status."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpiexec", required=True)
    parser.add_argument("--numproc-flag", required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--layout-source", type=Path, required=True)
    parser.add_argument("--layout-generated", type=Path, required=True)
    parser.add_argument("--ranks", nargs="+", type=int, required=True)
    args = parser.parse_args()
    source = json.loads(args.layout_source.read_text(encoding="utf-8"))
    generated = json.loads(args.layout_generated.read_text(encoding="utf-8"))
    if source != generated:
        print("logical HDF5 source and generated layout differ", file=sys.stderr)
        return 2
    if [item["id"] for item in source["layouts"]] != [
        "compiled-problem-v1",
        "result-v1",
        "checkpoint-v1",
    ]:
        print("required logical HDF5 layouts are incomplete", file=sys.stderr)
        return 2
    if "zero_owner_select_none" not in source["collective_call_discipline"]:
        print("zero-owner collective discipline is absent", file=sys.stderr)
        return 2
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    for ranks in args.ranks:
        artifact = args.artifact_root / f"rank-{ranks}.h5"
        command = [
            args.mpiexec,
            args.numproc_flag,
            str(ranks),
            args.executable,
            str(artifact),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            print(
                f"MPI matrix row ranks={ranks} failed with exit={completed.returncode}",
                file=sys.stderr,
            )
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
