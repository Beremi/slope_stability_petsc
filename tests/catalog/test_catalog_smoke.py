# SPDX-License-Identifier: BSD-2-Clause
"""E2E-CATALOG-SMOKE-001: compile, resolve and run a small catalog case.

This is a smoke test of the product workflow (``slope-stability compile`` and
``run``) at 1x1 and 2x2 with a capped number of accepted states.  It checks
that the installed engine runs, that every rank is bound to disjoint cores and
that the results verify; it is not numerical validation.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from slope_stability.cli import main as cli_main
from slope_stability.results import open_result

ROOT = Path(__file__).resolve().parents[2]
CASE_ID = "CASE-003"
LAYOUTS = ("1x1", "2x2")


def run_cli(*arguments: str) -> tuple[int, list[dict], str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        status = cli_main(list(arguments))
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    return status, lines, stderr.getvalue()


class CatalogSmoke(unittest.TestCase):
    def test_compile_and_run_small_case(self) -> None:
        build = Path(os.environ["SLOPE_BUILD_DIR"])
        mpiexec = os.environ["SLOPE_MPIEXEC"]
        numproc = os.environ.get("SLOPE_MPI_NUMPROC_FLAG", "-n")
        runtime = os.environ.get("SLOPE_RUNTIME_LIBRARY_PATH", "")
        with tempfile.TemporaryDirectory(prefix="slope-catalog-smoke-") as raw:
            work = Path(raw)
            prefix = work / "prefix"
            install = subprocess.run(
                ["cmake", "--install", str(build), "--prefix", str(prefix)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(install.returncode, 0, install.stderr)
            build_manifest = json.loads(
                (prefix / "share/slope-stability/build-manifest.json").read_bytes()
            )
            threaded_petsc = build_manifest["features"]["petsc_openmp_kernels"]
            status, lines, error = run_cli(
                "compile", "--case-id", CASE_ID, "--destination", str(work / "c")
            )
            self.assertEqual(status, 0, error)
            problem_id = lines[0]["problem_id"]
            result_ids = set()
            for layout in LAYOUTS:
                with self.subTest(layout=layout):
                    destination = work / f"run-{layout}"
                    arguments = [
                        "run",
                        "--prefix",
                        str(prefix),
                        "--compiled",
                        str(work / "c"),
                        "--case-id",
                        CASE_ID,
                        "--layout",
                        layout,
                        "--maximum-accepted-states",
                        "3",
                        f"--mpiexec={mpiexec}",
                        f"--numproc-flag={numproc}",
                        "--verify-binding",
                        "--timeout-seconds",
                        "600",
                        "--destination",
                        str(destination),
                    ]
                    if runtime:
                        arguments += ["--runtime-library-path", runtime]
                    status, lines, error = run_cli(*arguments)
                    report_path = destination / "run-catalog.json"
                    logs = sorted((destination / "logs").glob("*.stderr"))
                    diagnostics = "".join(
                        path.read_text(encoding="utf-8")[-2000:] for path in logs
                    )
                    self.assertEqual(status, 0, (lines, error, diagnostics))
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    self.assertEqual(report["status"], "pass")
                    (row,) = report["cases"]
                    self.assertEqual(row["problem_id"], problem_id)
                    self.assertEqual(row["result_validation"]["status"], "pass")
                    ranks, threads = (int(part) for part in layout.split("x"))
                    resolved = json.loads(
                        (destination / row["resolved_run"]["path"]).read_bytes()
                    )
                    self.assertEqual(
                        resolved["resources"]["third_party_threads"]["petsc"],
                        threads if threaded_petsc else 1,
                    )
                    binding = report["observed_binding"]
                    self.assertEqual(len(binding), ranks)
                    cores = [core for owned in binding.values() for core in owned]
                    self.assertEqual(len(cores), ranks * threads)
                    self.assertEqual(len(set(cores)), len(cores))
                    result = open_result(destination / row["result_directory"])
                    try:
                        self.assertTrue(result.complete)
                        self.assertEqual(result.status, "succeeded")
                        self.assertEqual(result.problem_id, problem_id)
                        result_ids.add(result.result_id)
                    finally:
                        result.close()
            self.assertEqual(len(result_ids), len(LAYOUTS))


if __name__ == "__main__":
    unittest.main()
