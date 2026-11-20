#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-2-Clause
"""Run one native test executable at several MPI rank x OpenMP thread layouts.

CTest launches every multi-layout native test through this runner, for example::

    mpi_layout_runner.py --suite pmg --executable ss_test_pmg_mpi \\
        --test-id PMG-CONTRACT-REUSE-001 --artifact-root DIR \\
        --mpiexec mpiexec --numproc-flag=-n --layout 1x1/1 --layout 2x1/2

A layout is ``R`` (``R`` ranks; the OpenMP environment is inherited), ``RxT``
(``R`` ranks with ``T`` OpenMP threads each) or, for the ``pmg`` suite only,
``RxT/A`` (``A`` of the ``R`` ranks own the coarse PMG levels). The suite
fixes the native command-line protocol, the launch binding and the checks:

========== ======================= ==========================================
Suite      Layouts                 Per-layout report checks
========== ======================= ==========================================
app        ``R``                   exit status only
mesh       ``R``                   identity
linear     ``R``                   identity, error <= 2e-11, reductions > 0
assembly   ``RxT``, bound to cores identity, observed threads
pmg        ``RxT/A``               identity, threads, active owners,
                                   error <= 1e-8, checksum spread <= 2e-8
nonlinear  ``RxT``                 identity, threads, error <= 1e-7,
                                   checksum spread <= 2e-7
hydraulic  ``RxT``                 identity, threads, error <= 1e-10,
                                   checksum spread <= 2e-10
result     ``RxT``                 identity, threads, error <= 1e-12, HDF5
                                   layout and payload digest; cross-layout
                                   identity and checksum only on request
========== ======================= ==========================================

Rank 0 of each launch writes one JSON report. Except for the ``result``
suite, all layouts of one test must report the same ``invariant`` (a
``sha256:`` identity of the partition- and thread-independent result). The
``mesh`` suite repeats every layout for each ``--mesh-shared-point-policy``
and requires one invariant over all of them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

__all__ = ["Layout", "Suite", "SUITES", "main", "parse_layout"]


class LayoutTestError(RuntimeError):
    """A launched layout failed or published an invalid report."""

    def __init__(self, message: str, exit_status: int = 1) -> None:
        super().__init__(message)
        self.exit_status = exit_status


@dataclass(frozen=True)
class Layout:
    """One MPI x OpenMP launch layout."""

    ranks: int
    threads: int | None
    active: int | None
    text: str

    @property
    def label(self) -> str:
        """Return a file-name-safe form of the layout."""
        return self.text.replace("/", "-a")


@dataclass(frozen=True)
class Suite:
    """Launch protocol and acceptance checks of one native test executable."""

    threads: bool
    """Layouts give an explicit OpenMP thread count (``RxT``)."""
    binding: str
    """``default`` (plain mpiexec), ``cores`` or ``none`` (oversubscribed)."""
    report: str
    """``none``, ``output`` (``--output`` file) or ``environment``."""
    active_ranks: bool = False
    maximum_error: float | None = None
    checksum_spread: float | None = None
    reductions: bool = False


SUITES: dict[str, Suite] = {
    "app": Suite(threads=False, binding="default", report="none"),
    "mesh": Suite(threads=False, binding="default", report="output"),
    "linear": Suite(
        threads=False,
        binding="default",
        report="output",
        maximum_error=2.0e-11,
        reductions=True,
    ),
    "assembly": Suite(threads=True, binding="cores", report="environment"),
    "pmg": Suite(
        threads=True,
        binding="none",
        report="output",
        active_ranks=True,
        maximum_error=1.0e-8,
        checksum_spread=2.0e-8,
    ),
    "nonlinear": Suite(
        threads=True,
        binding="none",
        report="output",
        maximum_error=1.0e-7,
        checksum_spread=2.0e-7,
    ),
    "hydraulic": Suite(
        threads=True,
        binding="none",
        report="output",
        maximum_error=1.0e-10,
        checksum_spread=2.0e-10,
    ),
    "result": Suite(
        threads=True,
        binding="none",
        report="output",
        maximum_error=1.0e-12,
        checksum_spread=1.0e-11,
    ),
}

MESH_SHARED_POINT_POLICIES = ("petsc", "stable_hash_p2", "stable_hash_high_order")
RESULT_CROSS_LAYOUT_TESTS = frozenset(
    {"OUT-CONTRACT-REPART-001", "OUT-CONTRACT-DEFL-001"}
)
RESULT_IO_BUDGET_TEST = "OUT-PERF-IO-001"
RESULT_DIGEST_TEST = "OUT-CONTRACT-DIGEST-001"
_LAYOUT = re.compile(
    r"^(?P<ranks>[1-9][0-9]*)(?:x(?P<threads>[1-9][0-9]*))?(?:/(?P<active>[1-9][0-9]*))?$"
)


def parse_layout(text: str, suite: Suite) -> Layout:
    """Parse one ``--layout`` value for ``suite``."""
    match = _LAYOUT.match(text)
    if match is None:
        raise ValueError(f"invalid layout {text!r}")
    threads = None if match["threads"] is None else int(match["threads"])
    active = None if match["active"] is None else int(match["active"])
    if suite.threads != (threads is not None) or suite.active_ranks != (
        active is not None
    ):
        expected = "RxT/A" if suite.active_ranks else "RxT" if suite.threads else "R"
        raise ValueError(f"layout {text!r} must have the form {expected}")
    ranks = int(match["ranks"])
    if active is not None and active > ranks:
        raise ValueError(f"layout {text!r} has more active than total ranks")
    return Layout(ranks=ranks, threads=threads, active=active, text=text)


def _launch_prefix(args: argparse.Namespace, suite: Suite, layout: Layout) -> list[str]:
    if args.direct:
        return []
    command = [args.mpiexec]
    if suite.binding == "cores":
        command += ["--bind-to", "core", "--map-by", f"slot:PE={layout.threads}"]
    elif suite.binding == "none":
        command += ["--oversubscribe", "--bind-to", "none"]
    return [*command, args.numproc_flag, str(layout.ranks)]


def _environment(suite: Suite, layout: Layout) -> dict[str, str]:
    environment = os.environ.copy()
    if layout.threads is not None:
        environment.update(
            {
                "OMP_NUM_THREADS": str(layout.threads),
                "OMP_DYNAMIC": "FALSE",
                "OMP_MAX_ACTIVE_LEVELS": "1",
            }
        )
    if suite.binding == "cores":
        environment.update({"OMP_PROC_BIND": "close", "OMP_PLACES": "cores"})
    elif suite.binding == "none":
        environment.update({"OMP_PROC_BIND": "false", "PYTHONDONTWRITEBYTECODE": "1"})
    return environment


def _run(command: list[str], environment: dict[str, str], description: str) -> None:
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode != 0:
        raise LayoutTestError(
            f"{description} failed with exit={completed.returncode}",
            completed.returncode,
        )


def _read_report(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise LayoutTestError(f"{description} did not publish {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LayoutTestError(f"cannot read {path}: {error}") from error
    if not isinstance(report, dict):
        raise LayoutTestError(f"invalid native result in {path}")
    return report


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _check_report(
    args: argparse.Namespace,
    suite: Suite,
    layout: Layout,
    report: dict[str, Any],
    path: Path,
) -> None:
    """Apply the per-layout checks shared by all suites."""
    if report.get("test_id") != args.test_id or report.get("ranks") != layout.ranks:
        raise LayoutTestError(f"invalid native result in {path}")
    if layout.threads is not None and (
        report.get("requested_threads") != layout.threads
        or report.get("observed_threads") != layout.threads
    ):
        raise LayoutTestError(f"invalid native result in {path}")
    if layout.active is not None and (
        report.get("active_count") != layout.active
        or report.get("active_owners") != layout.active
    ):
        raise LayoutTestError(f"invalid native result in {path}")
    invariant = report.get("invariant")
    if not isinstance(invariant, str) or not invariant.startswith("sha256:"):
        raise LayoutTestError(f"missing invariant in {path}")
    if suite.maximum_error is not None:
        maximum_error = report.get("maximum_error")
        if not _finite_number(maximum_error) or maximum_error > suite.maximum_error:
            raise LayoutTestError(f"invalid numerical error in {path}")
    if suite.checksum_spread is not None and not _finite_number(report.get("checksum")):
        raise LayoutTestError(f"missing checksum in {path}")
    if suite.reductions:
        reductions = report.get("global_reductions")
        if (
            isinstance(reductions, bool)
            or not isinstance(reductions, int)
            or reductions <= 0
        ):
            raise LayoutTestError(f"missing collective accounting in {path}")


def _check_cross_layout(
    test_id: str,
    suite: Suite,
    reports: list[dict[str, Any]],
    *,
    invariant: bool = True,
    checksum: bool = True,
) -> None:
    """Require one invariant and a bounded checksum spread over all layouts."""
    if invariant:
        invariants = {str(report["invariant"]) for report in reports}
        if len(invariants) != 1:
            raise LayoutTestError(
                f"{test_id} is layout dependent: {sorted(invariants)}"
            )
    if checksum and suite.checksum_spread is not None:
        checksums = [float(report["checksum"]) for report in reports]
        if checksums and max(checksums) - min(checksums) > suite.checksum_spread:
            raise LayoutTestError(f"{test_id} is layout dependent: {checksums}")


def _app_fixture(test_id: str, root: Path) -> Path | None:
    """Compile the compact problem that the collective HDF5 input tests read."""
    if not test_id.startswith("IO-H5-"):
        return None
    import h5py

    from slope_stability.authoring import load_document
    from slope_stability.compile import compact_simplex_mesh, compile_problem
    from tests.support.problem_fixtures import asset_document, case_document

    asset = load_document(
        asset_document(2, coordinate_order=1), source_path="asset.json"
    )
    case = load_document(
        case_document(2, field_order=1, coordinate_order=1),
        source_path="case.json",
    )
    package = compile_problem(
        asset=asset,
        case=case,
        mesh=compact_simplex_mesh(2, 1),
        destination=root / "problems",
    )
    fixture = package.path / "problem.h5"
    if test_id == "IO-H5-CORRUPT-001":
        corrupt = root / "corrupt.h5"
        shutil.copyfile(fixture, corrupt)
        with h5py.File(corrupt, "r+") as handle:
            del handle["mesh/topology/cells"]
        fixture = corrupt
    return fixture


def _run_app(args: argparse.Namespace, suite: Suite, layouts: list[Layout]) -> None:
    fixture = _app_fixture(args.test_id, args.artifact_root)
    for layout in layouts:
        command = [
            *_launch_prefix(args, suite, layout),
            args.executable,
            "--test-id",
            args.test_id,
        ]
        if fixture is not None:
            command.append(str(fixture))
        _run(
            command, _environment(suite, layout), f"{args.test_id} layout={layout.text}"
        )


def _run_reported(
    args: argparse.Namespace, suite: Suite, layouts: list[Layout]
) -> None:
    """Run the suites whose rank 0 writes one JSON report per layout."""
    policies: tuple[str | None, ...] = (None,)
    if args.mesh_shared_point_policy:
        policies = tuple(args.mesh_shared_point_policy)
    elif args.suite == "mesh":
        policies = ("petsc",)
    reports: list[dict[str, Any]] = []
    for policy in policies:
        for layout in layouts:
            stem = (
                f"layout-{layout.label}"
                if policy is None
                else f"{policy}-{layout.label}"
            )
            output = args.artifact_root / f"{stem}.json"
            environment = _environment(suite, layout)
            native = [args.executable, "--test-id", args.test_id]
            native += ["--expected-ranks", str(layout.ranks)]
            if layout.active is not None:
                native += ["--active-count", str(layout.active)]
            if layout.threads is not None:
                native += ["--expected-threads", str(layout.threads)]
            if suite.report == "output":
                native += ["--output", str(output)]
            else:
                environment["SLOPE_ASSEMBLY_RESULT"] = str(output)
                native += ["-options_left", "no"]
            if policy is not None:
                native += ["--shared-point-balance", policy]
            description = f"{args.test_id} layout={layout.text}"
            if policy is not None:
                description += f" policy={policy}"
            _run(
                [*_launch_prefix(args, suite, layout), *native],
                environment,
                description,
            )
            report = _read_report(output, description)
            _check_report(args, suite, layout, report, output)
            reports.append(report)
    _check_cross_layout(args.test_id, suite, reports)


def _contained_child(root: Path, leaf: str) -> Path:
    root = root.resolve()
    child = (root / leaf).resolve()
    if child.parent != root:
        raise ValueError(f"{child} is not an immediate child of {root}")
    return child


def _check_result_report(
    args: argparse.Namespace, report: dict[str, Any], output: Path, chunk_rows: int
) -> None:
    """Checks of the result writer reports in addition to :func:`_check_report`."""
    if len(str(report["invariant"])) != len("sha256:") + 64:
        raise LayoutTestError(f"missing invariant in {output}")
    expected_physical = "chunked_default" if chunk_rows == 0 else "chunked_requested"
    if report.get("physical_layout") != expected_physical:
        raise LayoutTestError(f"wrong physical HDF5 layout in {output}")
    payload_sha = report.get("payload_sha256")
    if not isinstance(payload_sha, str) or (
        payload_sha
        and (
            len(payload_sha) != 64
            or any(character not in "0123456789abcdef" for character in payload_sha)
        )
    ):
        raise LayoutTestError(f"invalid payload digest in {output}")
    if args.test_id == RESULT_IO_BUDGET_TEST:
        _check_io_budget(report, output)


def _check_io_budget(report: dict[str, Any], output: Path) -> None:
    """Bound the I/O volume, balance, memory and time of the result writer."""
    for key in ("minimum_rank_bytes", "maximum_rank_bytes", "peak_rss_bytes"):
        value = report.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise LayoutTestError(f"invalid {key} in {output}")
    timings = ("pack_seconds", "hdf5_seconds", "digest_seconds", "publication_seconds")
    for key in (*timings, "io_imbalance"):
        value = report.get(key)
        if not _finite_number(value) or value < 0.0:
            raise LayoutTestError(f"invalid {key} in {output}")
    logical_bytes = int(report["logical_bytes"])
    payload_bytes = int(report["payload_bytes"])
    minimum_rank_bytes = int(report["minimum_rank_bytes"])
    maximum_rank_bytes = int(report["maximum_rank_bytes"])
    if (
        payload_bytes <= logical_bytes
        or payload_bytes >= logical_bytes + 4 * 1024 * 1024
        or maximum_rank_bytes < minimum_rank_bytes
        or maximum_rank_bytes > logical_bytes
        or float(report["io_imbalance"]) > 1.01
        or int(report["peak_rss_bytes"]) >= 2 * 1024 * 1024 * 1024
        or any(float(report[key]) >= 60.0 for key in timings)
    ):
        raise LayoutTestError(f"I/O admission bound failed in {output}")


def _run_result_layout(
    args: argparse.Namespace,
    suite: Suite,
    layout: Layout,
    layout_root: Path,
    *,
    checkpoint: Path | None = None,
    checkpoint_mode: str | None = None,
    chunk_rows: int = 0,
) -> dict[str, Any]:
    if layout_root.exists():
        shutil.rmtree(layout_root)
    layout_root.mkdir()
    output = layout_root / "native-outcome.json"
    environment = _environment(suite, layout)
    environment["SLOPE_RESULT_DATASET_CHUNK_ROWS"] = str(chunk_rows)
    native = [
        args.executable,
        "--test-id",
        args.test_id,
        "--artifact-root",
        str(layout_root),
        "--output",
        str(output),
        "--expected-ranks",
        str(layout.ranks),
        "--expected-threads",
        str(layout.threads),
    ]
    if checkpoint is not None and checkpoint_mode is not None:
        native += [
            "--checkpoint-directory",
            str(checkpoint),
            "--checkpoint-mode",
            checkpoint_mode,
        ]
    phase = checkpoint_mode or "roundtrip"
    description = f"{args.test_id} layout={layout.text} phase={phase}"
    _run([*_launch_prefix(args, suite, layout), *native], environment, description)
    report = _read_report(output, description)
    _check_report(args, suite, layout, report, output)
    _check_result_report(args, report, output, chunk_rows)
    report["phase"] = phase
    return report


def _run_result(args: argparse.Namespace, suite: Suite, layouts: list[Layout]) -> None:
    """Run the result and checkpoint writer contracts.

    The repartitioning and deflation contracts write one checkpoint at the
    first layout and restore it at every layout. The digest contract varies
    the HDF5 chunking after the first layout and requires the physical payload
    to change while the logical identity does not.
    """
    if args.direct and any(layout.ranks != 1 for layout in layouts):
        raise LayoutTestError("--direct accepts only 1-rank layouts", 2)
    writer: dict[str, Any] | None = None
    checkpoint: Path | None = None
    cross_layout = args.test_id in RESULT_CROSS_LAYOUT_TESTS
    if cross_layout:
        checkpoint = _contained_child(args.artifact_root, "shared-checkpoint")
        if checkpoint.exists():
            shutil.rmtree(checkpoint)
        first = layouts[0]
        writer = _run_result_layout(
            args,
            suite,
            first,
            _contained_child(args.artifact_root, f"writer-layout-{first.label}"),
            checkpoint=checkpoint,
            checkpoint_mode="write",
        )
    reports = [
        _run_result_layout(
            args,
            suite,
            layout,
            _contained_child(args.artifact_root, f"layout-{layout.label}"),
            checkpoint=checkpoint,
            checkpoint_mode="restore" if cross_layout else None,
            chunk_rows=3 if args.test_id == RESULT_DIGEST_TEST and index != 0 else 0,
        )
        for index, layout in enumerate(layouts)
    ]
    compared = reports if writer is None else [writer, *reports]
    _check_cross_layout(
        args.test_id,
        suite,
        compared,
        invariant=args.require_invariant,
        checksum=args.require_checksum,
    )
    if args.test_id == RESULT_DIGEST_TEST:
        physical_layouts = {str(report["physical_layout"]) for report in reports}
        payload_digests = {
            str(report["payload_sha256"])
            for report in reports
            if report["payload_sha256"]
        }
        if (
            physical_layouts != {"chunked_default", "chunked_requested"}
            or len(payload_digests) < 2
        ):
            raise LayoutTestError(
                f"{args.test_id} did not vary physical HDF5 layout/payload"
            )
    summary = {
        "test_id": args.test_id,
        "layouts": reports,
        "checkpoint_writer": writer,
        "invariant_required": args.require_invariant,
        "checksum_required": args.require_checksum,
    }
    (args.artifact_root / "matrix-summary.json").write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--suite", required=True, choices=sorted(SUITES))
    parser.add_argument("--mpiexec", required=True)
    parser.add_argument("--numproc-flag", required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--test-id", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--layout", action="append", required=True, help="R, RxT or (pmg) RxT/A"
    )
    parser.add_argument(
        "--mesh-shared-point-policy",
        action="append",
        choices=MESH_SHARED_POINT_POLICIES,
        help="mesh suite: repeat every layout with this shared-point balancing",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="result suite: run 1-rank layouts without mpiexec",
    )
    parser.add_argument(
        "--require-invariant",
        action="store_true",
        help="result suite: require one identity over all layouts",
    )
    parser.add_argument(
        "--require-checksum",
        action="store_true",
        help="result suite: require one checksum over all layouts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the layouts of one test; return the process exit status."""
    parser = _parser()
    args = parser.parse_args(argv)
    suite = SUITES[args.suite]
    if args.mesh_shared_point_policy and args.suite != "mesh":
        parser.error("--mesh-shared-point-policy applies only to the mesh suite")
    if (args.direct or args.require_invariant or args.require_checksum) and (
        args.suite != "result"
    ):
        parser.error("--direct and --require-* apply only to the result suite")
    try:
        layouts = [parse_layout(text, suite) for text in args.layout]
    except ValueError as error:
        parser.error(str(error))
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    try:
        if args.suite == "app":
            _run_app(args, suite, layouts)
        elif args.suite == "result":
            _run_result(args, suite, layouts)
        else:
            _run_reported(args, suite, layouts)
    except LayoutTestError as error:
        print(error, file=sys.stderr)
        return error.exit_status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
