# SPDX-License-Identifier: BSD-2-Clause
"""``slope-stability`` command-line interface.

Subcommands::

    slope-stability verify-meshes            # check meshes/ against all digests
    slope-stability compile  --destination D [--case-id CASE-003 ...]
    slope-stability resolve  --problem-package P --prefix I --destination D
                             (--case-id CASE-003 | --case FILE) --layout 4x8
    slope-stability run      --prefix I --compiled D --destination O
                             --case-id CASE-003 --layout 2x2
    slope-stability run      --prefix I --package P --resolved-run R
                             --destination O
    slope-stability inspect  PATH

``--catalog`` and ``--meshes`` default to ``cases/catalog`` and ``meshes``
of the source tree (see :func:`slope_stability.catalog.find_repository_root`).
Every command prints JSON on standard output and returns 0 on success, 1 when
a verification or run failed and 2 on invalid input.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import platform
import signal
import sys
import tempfile
import threading
from time import perf_counter
from typing import Any, Callable, Sequence

from slope_protocol import CanonicalizationError, content_id, loads_strict

from ._io import dumps_json, file_sha256, fsync_path, read_json_object
from .authoring import AuthoringError, Case, load_document
from .catalog import (
    Catalog,
    CatalogError,
    CatalogVariant,
    compile_catalog,
    load_case_profiles,
    load_catalog,
    load_compilation,
    read_sha256sums,
    verify_meshes,
)
from .compile import CompileError
from .launch import (
    LaunchError,
    LaunchLayout,
    configure_resources,
    engine_run_options,
    launch_environment,
    local_resource_row,
    mpiexec_command,
    run_bounded_process,
    verify_binding,
)
from .resolution import (
    BuildCapabilities,
    CheckpointPolicy,
    ResolutionError,
    ResolvedRun,
    load_build_capabilities,
    load_compiled_problem,
    resolve_run,
    write_resolved_run,
)
from .results import IncompleteResult, ResultValidationError, open_result

RUN_REPORT = "run-catalog.json"
_INPUT_ERRORS = (
    AuthoringError,
    CanonicalizationError,
    CatalogError,
    CompileError,
    LaunchError,
    ResolutionError,
    ResultValidationError,
    OSError,
    ValueError,
)


# --------------------------------------------------------------------------
# Resolution overrides


def _positive(value: float) -> bool:
    return value > 0


def _nonnegative(value: float) -> bool:
    return value >= 0


@dataclass(frozen=True)
class _Override:
    flag: str
    path: str
    kind: type
    check: Callable[[Any], bool] | None = None
    requirement: str = ""
    choices: tuple[Any, ...] | None = None


_OVERRIDES = (
    _Override("--nonlinear-maximum-iterations", "nonlinear.maximum_iterations", int),
    _Override(
        "--nonlinear-absolute-tolerance",
        "nonlinear.absolute_tolerance",
        float,
        _positive,
        "positive",
    ),
    _Override(
        "--nonlinear-relative-tolerance",
        "nonlinear.relative_tolerance",
        float,
        _positive,
        "positive",
    ),
    _Override(
        "--nonlinear-stopping-tolerance",
        "nonlinear.stopping_tolerance",
        float,
        _positive,
        "positive",
    ),
    _Override(
        "--initial-nonlinear-stopping-tolerance",
        "continuation.initial_nonlinear_stopping_tolerance",
        float,
        _positive,
        "positive",
    ),
    _Override(
        "--maximum-accepted-states",
        "continuation.maximum_accepted_states",
        int,
        lambda value: value >= 2,
        "at least two",
    ),
    _Override(
        "--maximum-work", "continuation.maximum_work", float, _positive, "positive"
    ),
    _Override(
        "--factor-increment-stop",
        "continuation.factor_increment_stop",
        float,
        _nonnegative,
        "nonnegative",
    ),
    _Override(
        "--scaled-slope-tolerance",
        "continuation.scaled_slope_tolerance",
        float,
        _nonnegative,
        "nonnegative",
    ),
    _Override("--linear-relative-tolerance", "linear.relative_tolerance", float),
    _Override(
        "--linear-absolute-tolerance",
        "linear.absolute_tolerance",
        float,
        _positive,
        "positive",
    ),
    _Override("--linear-maximum-iterations", "linear.maximum_iterations", int),
    _Override("--linear-restart", "linear.restart", int),
    _Override(
        "--linear-ksp-type",
        "linear.ksp_type",
        str,
        choices=("fgmres", "hpddm_gcrodr", "fcg"),
    ),
    _Override(
        "--linear-recycle-dimension",
        "linear.recycle_dimension",
        int,
        _nonnegative,
        "nonnegative",
    ),
    _Override(
        "--linear-stopping-condition",
        "linear.stopping_condition",
        str,
        choices=(
            "original_rhs_true_residual",
            "projected_rhs_petsc",
            "preconditioned_residual_original_rhs",
            "natural_residual_initial",
        ),
    ),
    _Override(
        "--pmg-smoother",
        "pmg.smoother",
        str,
        choices=(
            "chebyshev",
            "chebyshev_point",
            "fgmres",
            "fgmres_block",
            "richardson",
            "richardson_block",
        ),
    ),
    _Override(
        "--pmg-base-iterations",
        "pmg.base_iterations",
        int,
        lambda value: value >= 1,
        "positive",
    ),
    _Override("--pmg-pre-smoothing-steps", "pmg.pre_smoothing_steps", int),
    _Override("--pmg-post-smoothing-steps", "pmg.post_smoothing_steps", int),
    _Override(
        "--pmg-coarse-solver",
        "pmg.coarse_solver",
        str,
        choices=("gamg", "preonly_lu"),
    ),
    _Override(
        "--pmg-coarse-redundant-group-size",
        "pmg.coarse_redundant_group_size",
        int,
        choices=(0, 1),
    ),
    _Override(
        "--deflation-maximum-dimension",
        "deflation.maximum_dimension",
        int,
        lambda value: value >= 1,
        "positive",
    ),
    _Override(
        "--deflation-candidate-source",
        "deflation.candidate_source",
        str,
        choices=(
            "accepted_newton_and_control_directions",
            "accepted_state_vectors",
            "accepted_states_and_full_krylov",
            "accepted_states_and_system_solutions",
        ),
    ),
    _Override(
        "--deflation-admission-tolerance",
        "deflation.admission_tolerance",
        float,
        _positive,
        "positive",
    ),
    _Override(
        "--deflation-projector",
        "deflation.projector",
        str,
        choices=("general_left_right", "legacy_a_orthogonal"),
    ),
    _Override(
        "--deflation-recycle",
        "deflation.recycle",
        str,
        choices=("accepted_steps", "attempt_snapshot_restore"),
    ),
    _Override(
        "--deflation-reorthogonalization-passes",
        "deflation.reorthogonalization_passes",
        int,
        lambda value: 1 <= value <= 4,
        "in [1, 4]",
    ),
    _Override(
        "--deflation-compression",
        "deflation.compression",
        str,
        choices=(
            "none",
            "snapshot_pod",
            "operator_svd",
            "snapshot_pod_total",
            "system_solution_window",
        ),
    ),
    _Override(
        "--deflation-compression-dimension",
        "deflation.compression_dimension",
        int,
        _nonnegative,
        "nonnegative",
    ),
    _Override(
        "--deflation-compression-history-decay",
        "deflation.compression_history_decay",
        float,
        lambda value: 0.0 <= value <= 1.0,
        "in [0, 1]",
    ),
    _Override(
        "--output-frequency",
        "outputs.frequency",
        str,
        choices=("terminal", "accepted_state"),
    ),
)


def _destination_name(flag: str) -> str:
    return "override_" + flag.removeprefix("--").replace("-", "_")


def _add_override_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "policy overrides",
        "typed overrides of the selected profiles (applied after the case's "
        "own resolution_overrides)",
    )
    for item in _OVERRIDES:
        group.add_argument(
            item.flag,
            dest=_destination_name(item.flag),
            type=item.kind,
            choices=item.choices,
            metavar=None if item.choices else item.kind.__name__.upper(),
            help=f"override {item.path}",
        )
    group.add_argument(
        "--pmg-levels",
        dest="override_pmg_levels",
        nargs="+",
        type=int,
        choices=(1, 2, 4, 8),
        help="complete increasing PMG polynomial hierarchy (pmg.levels)",
    )
    group.add_argument(
        "--deflation-enabled",
        dest="override_deflation_enabled",
        choices=("true", "false"),
        help="override deflation.enabled",
    )
    group.add_argument(
        "--set",
        dest="override_set",
        action="append",
        default=[],
        metavar="PATH=JSON",
        help="expert override of any resolvable policy path with a JSON value",
    )
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument(
        "--checkpoint",
        action="store_true",
        help="publish an accepted-state checkpoint (write_accepted)",
    )
    checkpoint.add_argument(
        "--restart-from",
        type=Path,
        metavar="CHECKPOINT",
        help="restart from a published checkpoint (restore_accepted)",
    )


def collect_overrides(
    arguments: argparse.Namespace, base: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Merge typed command-line overrides over ``base`` and validate them."""
    overrides = dict(base or {})
    for item in _OVERRIDES:
        value = getattr(arguments, _destination_name(item.flag))
        if value is None:
            continue
        if item.check is not None and not item.check(value):
            raise ValueError(f"{item.flag} must be {item.requirement}")
        overrides[item.path] = value
    if arguments.override_pmg_levels is not None:
        levels = tuple(arguments.override_pmg_levels)
        if levels != tuple(sorted(set(levels))):
            raise ValueError("--pmg-levels must be unique and increasing")
        overrides["pmg.levels"] = list(levels)
    if arguments.override_deflation_enabled is not None:
        overrides["deflation.enabled"] = arguments.override_deflation_enabled == "true"
    for text in arguments.override_set:
        path, separator, raw = text.partition("=")
        if not separator or not path:
            raise ValueError(f"--set expects PATH=JSON, got {text!r}")
        overrides[path] = loads_strict(raw)
    return overrides


def checkpoint_policy(arguments: argparse.Namespace) -> CheckpointPolicy | None:
    """Translate ``--checkpoint``/``--restart-from`` into a checkpoint policy."""
    if arguments.restart_from is not None:
        return CheckpointPolicy(
            mode="restore_accepted",
            path=str(arguments.restart_from.resolve(strict=True)),
        )
    if arguments.checkpoint:
        return CheckpointPolicy(mode="write_accepted")
    return None


# --------------------------------------------------------------------------
# Shared helpers


def _emit(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")), flush=True)


def _catalog(arguments: argparse.Namespace) -> Catalog:
    return load_catalog(arguments.catalog, mesh_root=arguments.meshes)


def _layout(arguments: argparse.Namespace) -> LaunchLayout:
    return LaunchLayout.parse(arguments.layout)


def _resolve_case(
    *,
    problem_package: Path,
    case: Case,
    profiles_directory: Path,
    build: BuildCapabilities,
    layout: LaunchLayout,
    arguments: argparse.Namespace,
) -> tuple[ResolvedRun, str]:
    problem = load_compiled_problem(problem_package / "problem.json")
    if problem.analysis != case.analysis.selector:
        raise ValueError("compiled problem analysis does not match the case")
    overrides = collect_overrides(
        arguments, dict(case.case_overrides.get("resolution_overrides", {}))
    )
    petsc_threads = arguments.petsc_threads
    if petsc_threads is None:
        petsc_threads = layout.threads if build.petsc_openmp_kernels else 1
    resolved = resolve_run(
        problem=problem,
        profiles=load_case_profiles(case, profiles_directory),
        resource=local_resource_row(
            layout,
            coarse_active_ranks=arguments.coarse_active_ranks,
            petsc_threads=petsc_threads,
            binding=arguments.binding,
            workspace_mib=arguments.operator_workspace_mib,
            address_space_bytes=arguments.address_space_limit_bytes,
        ),
        build=build,
        case_overrides=overrides or None,
        checkpoint=checkpoint_policy(arguments),
    )
    return resolved, problem.problem_id


def _engine_command(
    arguments: argparse.Namespace,
    layout: LaunchLayout,
    engine: Path,
    package: Path,
    resolved_path: Path,
    result_directory: Path,
) -> list[str]:
    return mpiexec_command(
        (
            engine,
            "run",
            package,
            resolved_path,
            result_directory,
            *engine_run_options(
                progress=not arguments.quiet,
                checkpoint_every=arguments.checkpoint_every,
                checkpoint_seconds=arguments.checkpoint_seconds,
                checkpoint_keep=arguments.checkpoint_keep,
            ),
        ),
        layout,
        mpiexec=arguments.mpiexec,
        numproc_flag=arguments.numproc_flag,
        binding=arguments.binding,
    )


def verify_run_result(
    result_directory: Path, *, run_id: str, problem_id: str, build_id: str
) -> dict[str, Any]:
    """Open a result directory and check that it belongs to the given run."""
    try:
        verified = open_result(result_directory)
    except (ResultValidationError, OSError) as error:
        return {"status": "fail", "error": f"{type(error).__name__}: {error}"}
    try:
        if isinstance(verified, IncompleteResult):
            return {"status": "fail", "error": "result has no terminal envelope"}
        identities_match = (
            verified.complete
            and verified.run_id == run_id
            and verified.problem_id == problem_id
            and verified.build_id == build_id
        )
        return {
            "status": "pass" if identities_match else "fail",
            "result_id": verified.result_id,
            "terminal_status": verified.status,
            "complete": verified.complete,
            "payload_verified": verified.verification.payload_verified,
            "logical_digests_verified": verified.verification.logical_digests_verified,
            "identities_match": identities_match,
        }
    finally:
        verified.close()


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "byte_count": path.stat().st_size,
    }


# --------------------------------------------------------------------------
# Commands


def command_verify_meshes(arguments: argparse.Namespace) -> int:
    """Verify every mesh against SHA256SUMS, the catalog and its assets."""
    catalog = _catalog(arguments)
    checks = verify_meshes(catalog, check_assets=not arguments.no_assets)
    for check in checks:
        _emit(
            {
                "mesh": check.mesh,
                "sha256": check.sha256,
                "byte_count": check.byte_count,
                "status": "pass" if check.passed else "fail",
                **(
                    {}
                    if check.passed
                    else {"expected": dict(sorted(check.expected.items()))}
                ),
            }
        )
    failed = sum(not check.passed for check in checks)
    _emit(
        {
            "mesh_root": str(catalog.mesh_root),
            "mesh_count": len(checks),
            "failed": failed,
            "status": "fail" if failed else "pass",
        }
    )
    return 1 if failed else 0


def command_compile(arguments: argparse.Namespace) -> int:
    """Compile catalog variants into content-addressed problem packages."""
    catalog = _catalog(arguments)
    # Mesh readers may print progress; keep standard output machine-readable.
    with contextlib.redirect_stdout(sys.stderr):
        report = compile_catalog(
            catalog,
            arguments.destination,
            case_ids=arguments.case_id,
            variant_ids=arguments.variant_id,
            uniform_refinement_levels=arguments.uniform_refinement_levels,
        )
    for row in report["cases"]:
        _emit(
            {
                "case_id": row["case_id"],
                "variant_id": row["variant_id"],
                "problem_id": row["problem_id"],
                "package": row["package"],
            }
        )
    _emit(
        {
            "catalog_id": report["catalog_id"],
            "case_variant_count": report["case_variant_count"],
            "unique_mesh_count": report["unique_mesh_count"],
            "destination": str(Path(arguments.destination).resolve()),
            "status": report["status"],
        }
    )
    return 0


def command_resolve(arguments: argparse.Namespace) -> int:
    """Resolve one case against a compiled package and an installed build."""
    if arguments.case is not None:
        case = load_document(arguments.case)
        if not isinstance(case, Case):
            raise ValueError(f"{arguments.case}: expected a case document")
        if arguments.profiles_dir is None:
            raise ValueError("--case requires --profiles-dir")
        profiles_directory = arguments.profiles_dir
    else:
        catalog = _catalog(arguments)
        case = catalog.load_case(
            catalog.variant(arguments.case_id, arguments.variant_id)
        )
        profiles_directory = arguments.profiles_dir or catalog.profiles_directory
    resolved, _ = _resolve_case(
        problem_package=arguments.problem_package,
        case=case,
        profiles_directory=profiles_directory,
        build=load_build_capabilities(arguments.prefix),
        layout=_layout(arguments),
        arguments=arguments,
    )
    path = write_resolved_run(resolved, arguments.destination)
    _emit({"resolved_run_id": resolved.run_id, "resolved_run_path": str(path)})
    return 0


def _run_direct(arguments: argparse.Namespace) -> int:
    if arguments.package is None:
        raise ValueError("--resolved-run requires --package")
    prefix = arguments.prefix.resolve()
    build = load_build_capabilities(prefix)
    manifest = read_json_object(arguments.resolved_run, error=ResolutionError)
    resources = manifest.get("resources")
    if not isinstance(resources, dict):
        raise ResolutionError("resolved_run_invalid", "missing resources section")
    layout = LaunchLayout(resources["world_size"], resources["openmp_threads"])
    workspace_mib = resources.get("operator_workspace_mib", 1024)
    if arguments.operator_workspace_mib not in (None, workspace_mib):
        raise ValueError("set --operator-workspace-mib when resolving the run")
    if arguments.address_space_limit_bytes is not None:
        configure_resources({}, address_space_bytes=arguments.address_space_limit_bytes)
    problem = load_compiled_problem(arguments.package / "problem.json")
    if manifest.get("build", {}).get("build_id") != build.build_id:
        raise ResolutionError(
            "resolved_run_build_mismatch", "the run was resolved for another build"
        )
    if arguments.destination.exists():
        raise ValueError(f"destination already exists: {arguments.destination}")
    arguments.destination.parent.mkdir(parents=True, exist_ok=True)
    if arguments.verify_binding and arguments.binding == "core":
        verify_binding(
            layout, mpiexec=arguments.mpiexec, numproc_flag=arguments.numproc_flag
        )
    command = _engine_command(
        arguments,
        layout,
        prefix / "bin" / "slope-engine",
        arguments.package.resolve(),
        arguments.resolved_run.resolve(),
        arguments.destination.resolve(),
    )
    environment = launch_environment(
        os.environ,
        threads=layout.threads,
        runtime_library_path=arguments.runtime_library_path,
    )
    started = perf_counter()
    completed = run_bounded_process(
        command,
        environment=environment,
        timeout_seconds=arguments.timeout_seconds,
        # A large cancelled run may still need to flush its checkpoint and result.
        termination_grace_seconds=60.0,
        capture_output=False,
    )
    elapsed = perf_counter() - started
    validation = verify_run_result(
        arguments.destination,
        run_id=str(manifest.get("id")),
        problem_id=problem.problem_id,
        build_id=build.build_id,
    )
    passed = (
        completed.exit_code == 0
        and not completed.timed_out
        and validation["status"] == "pass"
    )
    _emit(
        {
            "command": command,
            "exit_code": completed.exit_code,
            "timed_out": completed.timed_out,
            "elapsed_seconds": elapsed,
            "result_directory": str(arguments.destination.resolve()),
            "result_validation": validation,
            "status": "pass" if passed else "fail",
        }
    )
    return 0 if passed else 1


def _run_catalog_variant(
    arguments: argparse.Namespace,
    *,
    catalog: Catalog,
    record: CatalogVariant,
    compiled: dict[str, Any],
    compiled_root: Path,
    staging: Path,
    build: BuildCapabilities,
    engine: Path,
    layout: LaunchLayout,
    cancellation: threading.Event,
) -> dict[str, Any]:
    case = catalog.load_case(record)
    package = compiled_root / compiled["package"]
    resolved, problem_id = _resolve_case(
        problem_package=package,
        case=case,
        profiles_directory=catalog.profiles_directory,
        build=build,
        layout=layout,
        arguments=arguments,
    )
    if problem_id != compiled["problem_id"]:
        raise ValueError(f"{record.key}: compiled problem identity mismatch")
    resolved_path = write_resolved_run(resolved, staging / "resolved" / record.slug)
    result_directory = staging / "results" / record.slug
    result_directory.parent.mkdir(parents=True, exist_ok=True)
    command = _engine_command(
        arguments, layout, engine, package, resolved_path, result_directory
    )
    environment = launch_environment(
        os.environ,
        threads=layout.threads,
        runtime_library_path=arguments.runtime_library_path,
    )
    started = perf_counter()
    completed = run_bounded_process(
        command,
        environment=environment,
        timeout_seconds=arguments.timeout_seconds,
        cancellation_event=cancellation,
    )
    elapsed = perf_counter() - started
    logs = staging / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{record.slug}.stdout"
    stderr_path = logs / f"{record.slug}.stderr"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    ran = completed.exit_code == 0 and not completed.timed_out
    validation = (
        verify_run_result(
            result_directory,
            run_id=resolved.run_id,
            problem_id=problem_id,
            build_id=build.build_id,
        )
        if ran and not completed.cancelled
        else {"status": "not_applicable", "reason": "the engine did not succeed"}
    )
    passed = (
        ran
        and not completed.cancelled
        and validation["status"] == "pass"
        and validation.get("terminal_status") == "succeeded"
    )
    return {
        "case_id": record.case_id,
        "variant_id": record.variant_id,
        "case_document_id": case.id,
        "problem_id": problem_id,
        "run_id": resolved.run_id,
        "layout": {"ranks": layout.ranks, "threads": layout.threads},
        "binding": arguments.binding,
        "command": command,
        "exit_code": completed.exit_code,
        "timed_out": completed.timed_out,
        "cancelled": completed.cancelled,
        "termination_signal": completed.termination_signal,
        "elapsed_seconds": elapsed,
        "resolved_run": _artifact(resolved_path, staging),
        "result_directory": result_directory.relative_to(staging).as_posix(),
        "result_validation": validation,
        "stdout": _artifact(stdout_path, staging),
        "stderr": _artifact(stderr_path, staging),
        "status": "pass" if passed else "fail",
    }


def _run_catalog(arguments: argparse.Namespace) -> int:
    if arguments.compiled is None:
        raise ValueError("run requires --compiled (catalog mode) or --resolved-run")
    if arguments.layout is None:
        raise ValueError("catalog runs require --layout RxT")
    if not math.isfinite(arguments.timeout_seconds) or arguments.timeout_seconds <= 0.0:
        raise ValueError("--timeout-seconds must be finite and positive")
    layout = _layout(arguments)
    catalog = _catalog(arguments)
    compiled_root = arguments.compiled.resolve()
    compilation = load_compilation(compiled_root, catalog)
    compiled_rows = {
        (row["case_id"], row["variant_id"]): row for row in compilation["cases"]
    }
    records = catalog.select(arguments.case_id, arguments.variant_id)
    for record in records:
        row = compiled_rows.get((record.case_id, record.variant_id))
        if row is None or row["case_document_id"] != record.case_document_id:
            raise CatalogError(f"{record.key}: no matching compiled package")
    prefix = arguments.prefix.resolve()
    engine = prefix / "bin" / "slope-engine"
    build = load_build_capabilities(prefix)
    if arguments.destination.exists():
        raise ValueError(f"destination already exists: {arguments.destination}")
    resources = configure_resources(
        os.environ,
        workspace_mib=arguments.operator_workspace_mib,
        address_space_bytes=arguments.address_space_limit_bytes,
    )
    binding_report = None
    if arguments.verify_binding and arguments.binding == "core":
        report = verify_binding(
            layout, mpiexec=arguments.mpiexec, numproc_flag=arguments.numproc_flag
        )
        binding_report = {str(rank): list(c) for rank, c in report.rank_cores.items()}
    arguments.destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{arguments.destination.name}.", dir=arguments.destination.parent
        )
    )
    cancellation = threading.Event()
    previous = {
        number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)
    }

    def cancel(_signum: int, _frame: object) -> None:
        cancellation.set()
        raise KeyboardInterrupt("run cancelled")

    for number in previous:
        signal.signal(number, cancel)
    rows: list[dict[str, Any]] = []
    try:
        for record in records:
            try:
                row = _run_catalog_variant(
                    arguments,
                    catalog=catalog,
                    record=record,
                    compiled=compiled_rows[(record.case_id, record.variant_id)],
                    compiled_root=compiled_root,
                    staging=staging,
                    build=build,
                    engine=engine,
                    layout=layout,
                    cancellation=cancellation,
                )
            except _INPUT_ERRORS as error:
                row = {
                    "case_id": record.case_id,
                    "variant_id": record.variant_id,
                    "exception": f"{type(error).__name__}: {error}",
                    "status": "fail",
                }
            rows.append(row)
            _emit(
                {
                    key: row.get(key)
                    for key in (
                        "case_id",
                        "variant_id",
                        "status",
                        "exit_code",
                        "elapsed_seconds",
                        "exception",
                    )
                    if key in row
                }
            )
        passed = all(row["status"] == "pass" for row in rows)
        report: dict[str, Any] = {
            "kind": "slope_stability.catalog_run_report",
            "schema_version": {"major": 2, "minor": 0, "patch": 0},
            "catalog_id": catalog.catalog_id,
            "compilation": {
                "path": str(compiled_root),
                "sha256": file_sha256(compiled_root / "compile-catalog.json"),
            },
            "build_id": build.build_id,
            "executable_id": build.executable_id,
            "dependency_id": build.dependency_id,
            "layout": {"ranks": layout.ranks, "threads": layout.threads},
            "binding": arguments.binding,
            "observed_binding": binding_report,
            "host": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "logical_cpu_count": os.cpu_count(),
            },
            "execution_resources": resources,
            "timeout_seconds": arguments.timeout_seconds,
            "case_variant_count": len(rows),
            "cases": rows,
            "status": "pass" if passed else "fail",
        }
        report["id"] = content_id(report)
        report_path = staging / RUN_REPORT
        report_path.write_text(dumps_json(report), encoding="utf-8")
        fsync_path(report_path)
        staging.rename(arguments.destination)
    except BaseException:
        _emit({"incomplete_output": str(staging), "status": "interrupted"})
        raise
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    _emit(
        {
            "case_variant_count": len(rows),
            "destination": str(arguments.destination.resolve()),
            "report_id": report["id"],
            "status": report["status"],
        }
    )
    return 0 if passed else 1


def command_run(arguments: argparse.Namespace) -> int:
    """Launch the engine for a resolved run or for compiled catalog variants."""
    if arguments.resolved_run is not None:
        return _run_direct(arguments)
    return _run_catalog(arguments)


def _inspect_result(path: Path) -> dict[str, Any]:
    result = open_result(path)
    try:
        if isinstance(result, IncompleteResult):
            return {"kind": "result", "status": "incomplete", "complete": False}
        return {
            "kind": "result",
            "result_id": result.result_id,
            "run_id": result.run_id,
            "problem_id": result.problem_id,
            "build_id": result.build_id,
            "status": result.status,
            "complete": result.complete,
            "analysis": {
                "method": result.analysis.method,
                "outcome": result.analysis.outcome,
            },
            "quantities": list(result.quantities),
            "restart": None
            if result.restart is None
            else {
                "parent_run_id": result.restart.parent_run_id,
                "parent_checkpoint_id": result.restart.parent_checkpoint_id,
            },
        }
    finally:
        result.close()


def inspect_path(path: Path, *, mesh_root: Path | None = None) -> dict[str, Any]:
    """Summarize a result, package, resolved run, compilation, catalog or mesh."""
    if path.is_dir():
        if (path / "result.json").is_file() or (path / "result.h5").exists():
            return _inspect_result(path)
        if (path / "problem.json").is_file():
            problem = load_compiled_problem(path / "problem.json")
            return {"kind": "problem_package", **problem.model_dump(mode="json")}
        if (path / "compile-catalog.json").is_file():
            report = load_compilation(path)
            return {
                "kind": "catalog_compilation",
                "catalog_id": report["catalog_id"],
                "cases": [
                    {
                        key: row[key]
                        for key in ("case_id", "variant_id", "problem_id", "package")
                    }
                    for row in report["cases"]
                ],
            }
        if (path / RUN_REPORT).is_file():
            report = read_json_object(path / RUN_REPORT)
            return {
                "kind": "catalog_run",
                "status": report.get("status"),
                "cases": [
                    {
                        key: row.get(key)
                        for key in ("case_id", "variant_id", "status", "run_id")
                    }
                    for row in report.get("cases", [])
                ],
            }
        if (path / "index.json").is_file():
            catalog = load_catalog(path, mesh_root=mesh_root)
            return {
                "kind": "catalog",
                "catalog_id": catalog.catalog_id,
                "mesh_root": str(catalog.mesh_root),
                "cases": [
                    {
                        "case_id": record.case_id,
                        "variant_id": record.variant_id,
                        "field_order": record.field_order,
                        "mesh": record.mesh,
                    }
                    for record in catalog.variants
                ],
            }
        raise ValueError(f"{path}: unrecognized directory")
    if path.suffix == ".msh":
        digest = file_sha256(path)
        absolute = path.resolve()
        roots = [mesh_root.resolve()] if mesh_root is not None else []
        roots += [
            parent for parent in absolute.parents if (parent / "SHA256SUMS").is_file()
        ]
        listed: bool | None = None
        for root in roots:
            if absolute.is_relative_to(root) and (root / "SHA256SUMS").is_file():
                recorded = read_sha256sums(root).get(
                    absolute.relative_to(root).as_posix()
                )
                listed = recorded == digest
                break
        return {
            "kind": "mesh",
            "sha256": digest,
            "byte_count": path.stat().st_size,
            "listed": listed,
        }
    document = read_json_object(path)
    if document.get("kind") == "slope_stability.resolved_run":
        resources = document.get("resources", {})
        return {
            "kind": "resolved_run",
            "run_id": document.get("id"),
            "problem_id": document.get("problem", {}).get("id"),
            "build_id": document.get("build", {}).get("build_id"),
            "layout": {
                "ranks": resources.get("world_size"),
                "threads": resources.get("openmp_threads"),
            },
        }
    return {
        "kind": document.get("kind"),
        "id": document.get("id"),
        "identity": document.get("identity"),
    }


def command_inspect(arguments: argparse.Namespace) -> int:
    """Print a JSON summary of an artifact."""
    _emit(inspect_path(arguments.path, mesh_root=arguments.meshes))
    return 0


# --------------------------------------------------------------------------
# Parser


def _add_launch_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("launch")
    group.add_argument(
        "--binding",
        choices=("core", "none"),
        default="core",
        help="per-rank CPU binding (default: disjoint whole cores per rank)",
    )
    group.add_argument("--mpiexec", default="mpiexec", help="MPI launcher")
    group.add_argument("--numproc-flag", default="-n", help="rank-count flag")
    group.add_argument(
        "--runtime-library-path",
        type=Path,
        help="directory prepended to LD_LIBRARY_PATH (for example PETSc's lib)",
    )
    group.add_argument(
        "--verify-binding",
        action="store_true",
        help="probe and validate the per-rank binding before running",
    )


def _add_layout_arguments(parser: argparse.ArgumentParser, required: bool) -> None:
    parser.add_argument(
        "--layout",
        required=required,
        metavar="RxT",
        help="MPI ranks x OpenMP threads per rank, for example 4x8",
    )
    parser.add_argument(
        "--coarse-active-ranks",
        type=int,
        help="ranks active on the coarsest PMG level (default: all)",
    )
    parser.add_argument(
        "--petsc-threads",
        type=int,
        help="override the recorded PETSc kernel team size (default from the build)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the ``slope-stability`` command."""
    parser = argparse.ArgumentParser(
        prog="slope-stability",
        description="Prepare, launch and inspect slope-stability engine runs.",
    )
    parser.add_argument(
        "--catalog", type=Path, help="catalog directory (default: cases/catalog)"
    )
    parser.add_argument(
        "--meshes", type=Path, help="mesh root (default: from the catalog index)"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser(
        "verify-meshes", help="verify meshes against SHA256SUMS and the catalog"
    )
    verify.add_argument(
        "--no-assets",
        action="store_true",
        help="skip the asset mesh_source.sha256 cross-check",
    )
    verify.set_defaults(handler=command_verify_meshes)

    compile_parser = commands.add_parser(
        "compile", help="compile catalog variants into problem packages"
    )
    compile_parser.add_argument("--destination", type=Path, required=True)
    compile_parser.add_argument("--case-id", action="append")
    compile_parser.add_argument("--variant-id", action="append")
    compile_parser.add_argument(
        "--uniform-refinement-levels",
        type=int,
        default=0,
        help="uniformly split every straight simplex this many times",
    )
    compile_parser.set_defaults(handler=command_compile)

    resolve = commands.add_parser(
        "resolve", help="resolve a case into resolved-run.json"
    )
    resolve.add_argument("--problem-package", type=Path, required=True)
    resolve.add_argument("--prefix", type=Path, required=True)
    resolve.add_argument("--destination", type=Path, required=True)
    selection = resolve.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case-id", help="catalog case id")
    selection.add_argument("--case", type=Path, help="case document path")
    resolve.add_argument("--variant-id", default="canonical")
    resolve.add_argument("--profiles-dir", type=Path)
    resolve.add_argument("--binding", choices=("core", "none"), default="core")
    resolve.add_argument("--operator-workspace-mib", type=int)
    resolve.add_argument("--address-space-limit-bytes", type=int)
    _add_layout_arguments(resolve, required=True)
    _add_override_arguments(resolve)
    resolve.set_defaults(handler=command_resolve)

    run = commands.add_parser("run", help="launch the engine and verify results")
    run.add_argument("--prefix", type=Path, required=True)
    run.add_argument("--destination", type=Path, required=True)
    run.add_argument(
        "--compiled", type=Path, help="compilation directory (catalog mode)"
    )
    run.add_argument("--case-id", action="append")
    run.add_argument("--variant-id", action="append")
    run.add_argument("--package", type=Path, help="problem package (direct mode)")
    run.add_argument(
        "--resolved-run", type=Path, help="resolved-run.json (direct mode)"
    )
    run.add_argument("--timeout-seconds", type=float, default=86400.0)
    run.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="write a restartable checkpoint every N accepted states",
    )
    run.add_argument(
        "--checkpoint-seconds",
        type=int,
        default=0,
        help="write a restartable checkpoint when N seconds have passed",
    )
    run.add_argument("--checkpoint-keep", type=int, default=1)
    run.add_argument(
        "--quiet", action="store_true", help="no engine progress lines on stderr"
    )
    run.add_argument("--operator-workspace-mib", type=int)
    run.add_argument("--address-space-limit-bytes", type=int)
    _add_layout_arguments(run, required=False)
    _add_launch_arguments(run)
    _add_override_arguments(run)
    run.set_defaults(handler=command_run)

    inspect = commands.add_parser("inspect", help="summarize an artifact as JSON")
    inspect.add_argument("path", type=Path)
    inspect.set_defaults(handler=command_inspect)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``slope-stability`` console script."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except _INPUT_ERRORS as error:
        print(
            f"slope-stability {arguments.command}: error: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_parser",
    "collect_overrides",
    "inspect_path",
    "main",
    "verify_run_result",
]
