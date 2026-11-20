# SPDX-License-Identifier: BSD-2-Clause
"""Exact, data-only run resolution for compiled problem packages."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from slope_protocol import (
    canonicalize,
    content_id,
    loads_strict,
    normalize_package_path,
)

from ._io import file_sha256, fsync_path, read_json_object
from ._solver_settings import MAXIMUM_ACCEPTED_STATES, setting_adapter
from .authoring import Profile, ResourceRow


class ResolutionError(ValueError):
    """Stable resolver failure."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        validate_default=True,
    )


class CompiledProblemRef(_FrozenModel):
    """Identity and capability summary of a compiled problem package."""

    problem_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    schema_version: tuple[int, int, int]
    dimension: int
    field_order: int
    coordinate_order: int
    analysis: str
    operator_class: str
    required_outputs: tuple[str, ...]

    @model_validator(mode="after")
    def supported_orders(self) -> "CompiledProblemRef":
        if self.dimension not in (2, 3):
            raise ValueError("dimension must be 2 or 3")
        if self.field_order not in (1, 2, 4, 8):
            raise ValueError("field order must be P1, P2, P4, or P8")
        if self.coordinate_order not in (1, 2, 4):
            raise ValueError("coordinate order must be G1, G2, or G4")
        return self


class BuildCapabilities(_FrozenModel):
    """Identity and features of an installed native build.

    Loaded from ``share/slope-stability/build-manifest.json`` and ``install-
    manifest.json``; :func:`resolve_run` rejects runs the build cannot execute.
    """

    build_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    executable_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    dependency_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    schema_version: tuple[int, int, int]
    petsc_int_bits: int
    mpi: bool
    openmp: bool
    parallel_hdf5: bool
    hypre: bool
    parmetis: bool
    hydraulic_pmg_budget: bool = False
    petsc_openmp_kernels: bool = False

    @model_validator(mode="after")
    def supported_index_width(self) -> "BuildCapabilities":
        if self.petsc_int_bits not in (32, 64):
            raise ValueError("PETSc index width must be 32 or 64")
        return self


class CheckpointPolicy(_FrozenModel):
    """Accepted-state checkpoint request of a run.

    ``write_accepted`` publishes a checkpoint; ``restore_accepted`` restarts from the
    checkpoint at the normalized absolute ``path``.
    """

    mode: Literal["write_accepted", "restore_accepted"]
    path: str | None = None

    @model_validator(mode="after")
    def valid_mode_path_pair(self) -> "CheckpointPolicy":
        if self.mode == "write_accepted":
            if self.path is not None:
                raise ValueError("write_accepted does not accept a checkpoint path")
            return self
        if (
            self.path is None
            or self.path == "/"
            or not self.path.startswith("/")
            or "//" in self.path
            or "\x00" in self.path
            or str(Path(self.path)) != self.path
            or any(part in {".", ".."} for part in Path(self.path).parts)
        ):
            raise ValueError(
                "restore_accepted requires a normalized absolute checkpoint path"
            )
        return self


class SweepAxis(_FrozenModel):
    """Named Cartesian sweep over one dotted override path."""

    name: str = Field(min_length=1)
    path: str = Field(min_length=1)
    values: tuple[Any, ...] = Field(min_length=1)


class ResolutionTrace(_FrozenModel):
    """Origin (profile, case or override) of one resolved policy leaf."""

    path: str
    source: str
    value: Any


@dataclass(frozen=True)
class SweepRow:
    """One expanded sweep point: coordinates, overrides and repetition."""

    coordinates: Mapping[str, Any]
    overrides: Mapping[str, Any]
    repetition: int
    variant_id: str


@dataclass(frozen=True)
class ResolvedRun:
    """Resolved run manifest, its content identity, trace and canonical bytes."""

    run_id: str
    manifest: Mapping[str, Any]
    trace: Mapping[str, ResolutionTrace]
    canonical_bytes: bytes


def load_compiled_problem(
    source: Path | str | bytes | Mapping[str, Any],
) -> CompiledProblemRef:
    """Read and identity-check one complete compiled-problem manifest."""
    package_root: Path | None = None
    if isinstance(source, Path):
        package_root = source.parent
        raw = loads_strict(source.read_bytes())
    elif isinstance(source, Mapping):
        raw = loads_strict(canonicalize(dict(source)))
    elif isinstance(source, str) and not source.lstrip().startswith("{"):
        source_path = Path(source)
        package_root = source_path.parent
        raw = loads_strict(source_path.read_bytes())
    else:
        raw = loads_strict(source)
    if not isinstance(raw, dict):
        raise ResolutionError("compiled_problem_not_object", type(raw).__name__)
    allowed = {
        "id",
        "kind",
        "schema_version",
        "created_by",
        "provenance",
        "extensions",
        "identity",
        "units",
        "mesh",
        "spaces",
        "supports",
        "materials",
        "physics",
        "boundary_rules",
        "initial_state",
        "coupling",
        "operators",
        "outputs",
        "payloads",
    }
    extra = set(raw).difference(allowed | {"prescribed_saturation"})
    missing = allowed.difference(raw)
    if extra or missing:
        raise ResolutionError(
            "compiled_problem_shape",
            f"missing={sorted(missing)}, extra={sorted(extra)}",
        )
    if raw["kind"] != "slope_stability.compiled_problem":
        raise ResolutionError("compiled_problem_kind", repr(raw["kind"]))
    identity_input = dict(raw)
    supplied_id = identity_input.pop("id")
    expected_id = content_id(identity_input)
    if supplied_id != expected_id:
        raise ResolutionError(
            "compiled_problem_identity_mismatch",
            f"{supplied_id!r} != {expected_id}",
        )
    for payload in raw["payloads"].values():
        relative = normalize_package_path(payload["path"])
        if package_root is not None:
            payload_path = package_root / relative
            if not payload_path.is_file():
                raise ResolutionError("compiled_problem_payload_missing", relative)
            if not _payload_verified(
                payload_path, payload["byte_count"], payload["sha256"]
            ):
                raise ResolutionError("compiled_problem_payload_digest", relative)
    version = raw["schema_version"]
    return CompiledProblemRef(
        problem_id=supplied_id,
        schema_version=(
            version["major"],
            version["minor"],
            version["patch"],
        ),
        dimension=raw["mesh"]["dimension"],
        field_order=raw["spaces"]["displacement"]["order"],
        coordinate_order=raw["mesh"]["coordinate_order"],
        analysis=raw["operators"]["analysis"],
        operator_class=raw["operators"]["operator_class"],
        required_outputs=tuple(raw["outputs"]["required"]),
    )


_OVERRIDE_TYPES: dict[str, type[Any] | tuple[type[Any], ...]] = {
    "continuation.initial_factor": float,
    "continuation.initial_step": float,
    "continuation.minimum_step": float,
    "continuation.maximum_step": float,
    "continuation.maximum_work": float,
    "continuation.initial_work_scale": float,
    "continuation.scaled_slope_tolerance": float,
    "continuation.factor_increment_stop": float,
    "continuation.maximum_accepted_states": int,
    "continuation.initial_nonlinear_stopping_criterion": str,
    "continuation.initial_nonlinear_stopping_tolerance": float,
    "nonlinear.relative_tolerance": float,
    "nonlinear.absolute_tolerance": float,
    "nonlinear.stopping_criterion": str,
    "nonlinear.stopping_tolerance": float,
    "nonlinear.regularization_minimum": float,
    "nonlinear.maximum_iterations": int,
    "linear.relative_tolerance": float,
    "linear.absolute_tolerance": float,
    "linear.stopping_condition": str,
    "linear.maximum_iterations": int,
    "linear.restart": int,
    "linear.ksp_type": str,
    "linear.recycle_dimension": int,
    "pmg.smoother": str,
    "pmg.base_iterations": int,
    "pmg.hydraulic_base_iterations": int,
    "pmg.pre_smoothing_steps": int,
    "pmg.post_smoothing_steps": int,
    "pmg.coarse_solver": str,
    "pmg.coarse_direct_maximum_dofs": int,
    "pmg.coarse_redundant_group_size": int,
    "pmg.levels": list,
    "deflation.enabled": bool,
    "deflation.projector": str,
    "deflation.maximum_dimension": int,
    "deflation.candidate_source": str,
    "deflation.recycle": str,
    "deflation.admission_tolerance": float,
    "deflation.pivot_tolerance": float,
    "deflation.condition_limit": float,
    "deflation.invariant_tolerance": float,
    "deflation.reorthogonalization_passes": int,
    "deflation.dense_factor": str,
    "deflation.compression": str,
    "deflation.compression_dimension": int,
    "deflation.compression_history_decay": float,
    "outputs.frequency": str,
    "outputs.history": bool,
}

_OVERRIDE_VALUES: dict[str, frozenset[Any]] = {
    "pmg.smoother": frozenset(
        {
            "chebyshev",
            "chebyshev_point",
            "fgmres",
            "fgmres_block",
            "richardson",
            "richardson_block",
        }
    ),
    "pmg.coarse_solver": frozenset({"gamg", "preonly_lu"}),
    "linear.stopping_condition": frozenset(
        {
            "original_rhs_true_residual",
            "projected_rhs_petsc",
            "preconditioned_residual_original_rhs",
            "natural_residual_initial",
        }
    ),
    "deflation.candidate_source": frozenset(
        {
            "accepted_newton_and_control_directions",
            "accepted_state_vectors",
            "accepted_states_and_full_krylov",
            "accepted_states_and_system_solutions",
        }
    ),
    "deflation.recycle": frozenset({"accepted_steps", "attempt_snapshot_restore"}),
    "deflation.projector": frozenset({"general_left_right", "legacy_a_orthogonal"}),
    "deflation.dense_factor": frozenset({"partial_pivot_lu"}),
    "deflation.compression": frozenset(
        {
            "none",
            "snapshot_pod",
            "operator_svd",
            "snapshot_pod_total",
            "system_solution_window",
        }
    ),
    "outputs.frequency": frozenset({"terminal", "accepted_state"}),
}

_POSITIVE_DEFLATION_OVERRIDES = frozenset(
    {
        "deflation.admission_tolerance",
        "deflation.pivot_tolerance",
        "deflation.invariant_tolerance",
    }
)


def _serial_copy(value: Any) -> Any:
    """Round-trip a value through canonical JSON to remove object aliases."""
    import json

    return json.loads(canonicalize(value))


def compose_profiles(
    profiles: tuple[Profile, ...] | list[Profile],
) -> dict[str, Profile]:
    """Select exactly one explicit profile per concern family."""
    result: dict[str, Profile] = {}
    for profile in profiles:
        if profile.family in result:
            raise ResolutionError("duplicate_profile_family", profile.family)
        result[profile.family] = profile
    return result


def _validate_override(path: str, value: Any) -> None:
    section, _, name = path.partition(".")
    adapter = setting_adapter(section, name)
    if adapter is not None:
        try:
            if value is None:
                raise ValueError("a typed setting cannot be null")
            adapter.validate_python(value, strict=True)
        except (ValidationError, ValueError) as error:
            raise ResolutionError("solver_setting_invalid", path) from error
        return
    expected = _OVERRIDE_TYPES.get(path)
    if expected is None:
        raise ResolutionError("override_not_allowlisted", path)
    if isinstance(value, bool) and expected is not bool:
        raise ResolutionError("override_type_mismatch", path)
    if not isinstance(value, expected):
        raise ResolutionError(
            "override_type_mismatch",
            f"{path}: expected {expected}, received {type(value).__name__}",
        )
    allowed_values = _OVERRIDE_VALUES.get(path)
    if allowed_values is not None and value not in allowed_values:
        raise ResolutionError("override_value_not_allowlisted", path)
    if path in _POSITIVE_DEFLATION_OVERRIDES and value <= 0.0:
        raise ResolutionError("override_value_out_of_range", path)
    if path == "pmg.levels" and (
        not value
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or tuple(value) != tuple(sorted(set(value)))
        or any(item not in (1, 2, 4, 8) for item in value)
    ):
        raise ResolutionError("override_value_out_of_range", path)
    if path == "deflation.condition_limit" and value <= 1.0:
        raise ResolutionError("override_value_out_of_range", path)
    if path == "deflation.maximum_dimension" and value <= 0:
        raise ResolutionError("override_value_out_of_range", path)
    if path == "deflation.compression_dimension" and value < 0:
        raise ResolutionError("override_value_out_of_range", path)
    if path == "deflation.compression_history_decay" and not 0.0 <= value <= 1.0:
        raise ResolutionError("override_value_out_of_range", path)
    if path == "deflation.reorthogonalization_passes" and not 1 <= value <= 4:
        raise ResolutionError("override_value_out_of_range", path)
    if path in {"pmg.base_iterations", "pmg.hydraulic_base_iterations"} and value <= 0:
        raise ResolutionError("override_value_out_of_range", path)


def expand_sweep(
    *,
    axes: tuple[SweepAxis, ...],
    base_overrides: Mapping[str, Any] | None = None,
    repetitions: int = 1,
) -> tuple[SweepRow, ...]:
    """Expand named axes as a deterministic Cartesian product."""
    if repetitions < 1:
        raise ResolutionError("invalid_repetition_count", str(repetitions))
    ordered_axes = tuple(sorted(axes, key=lambda axis: axis.name))
    if len({axis.name for axis in ordered_axes}) != len(ordered_axes):
        raise ResolutionError("duplicate_sweep_axis", "axis names")
    if len({axis.path for axis in ordered_axes}) != len(ordered_axes):
        raise ResolutionError("duplicate_sweep_path", "axis paths")
    for axis in ordered_axes:
        for value in axis.values:
            _validate_override(axis.path, value)
    baseline = dict(base_overrides or {})
    for path, value in baseline.items():
        _validate_override(path, value)

    rows: list[SweepRow] = []
    seen: set[str] = set()
    value_rows = product(*(axis.values for axis in ordered_axes))
    for values in value_rows:
        axis_coordinates = {
            axis.name: value for axis, value in zip(ordered_axes, values, strict=True)
        }
        overrides = dict(baseline)
        overrides.update(
            {axis.path: value for axis, value in zip(ordered_axes, values, strict=True)}
        )
        for repetition in range(repetitions):
            coordinates = {
                **axis_coordinates,
                "repetition": repetition,
            }
            variant_id = content_id(
                {
                    "coordinates": coordinates,
                    "overrides": overrides,
                }
            )
            if variant_id in seen:
                raise ResolutionError("duplicate_effective_run", variant_id)
            seen.add(variant_id)
            rows.append(
                SweepRow(
                    coordinates=coordinates,
                    overrides=deepcopy(overrides),
                    repetition=repetition,
                    variant_id=variant_id,
                )
            )
    return tuple(rows)


def resolve_resource(
    resource: ResourceRow,
    *,
    field_order: int,
) -> dict[str, Any]:
    """Resolve a resource row into exact rank and thread intent."""
    if field_order not in (1, 2, 4, 8):
        raise ResolutionError("unsupported_field_order", str(field_order))
    if resource.world_size != resource.nodes * resource.ranks_per_node:
        raise ResolutionError(
            "resource_world_size_mismatch",
            f"{resource.world_size} != {resource.nodes}*{resource.ranks_per_node}",
        )
    if resource.smt == "off" and resource.cores_per_rank < resource.openmp_threads:
        raise ResolutionError(
            "insufficient_cores_per_rank",
            f"{resource.cores_per_rank} cores for {resource.openmp_threads} threads",
        )
    active_ranks = {
        "fine": list(resource.active_ranks.fine),
        "coarse": list(resource.active_ranks.coarse),
    }
    for level, ranks in active_ranks.items():
        if len(set(ranks)) != len(ranks):
            raise ResolutionError("duplicate_active_rank", level)
        if any(rank < 0 or rank >= resource.world_size for rank in ranks):
            raise ResolutionError("active_rank_out_of_range", level)
    if set(active_ranks["coarse"]).difference(active_ranks["fine"]):
        raise ResolutionError(
            "coarse_rank_not_active_on_fine", repr(active_ranks["coarse"])
        )
    resource_values = resource.model_dump(mode="json", exclude_none=True)
    caps = [
        total
        for total in (
            resource.memory_per_node_mib * resource.nodes
            if resource.memory_per_node_mib is not None
            else None,
            resource.address_space_per_rank_mib * resource.world_size
            if resource.address_space_per_rank_mib is not None
            else None,
        )
        if total is not None
    ]
    vector_cap = resource.linear_vector_memory_cap_mib
    if vector_cap is None:
        vector_cap = max(1, min(2**31 - 1, min(caps) // 4)) if caps else 1024
    # Unless set explicitly, the rank-local operator workspace may use a quarter
    # of the rank's share of node memory, and never less than the 1024 MiB
    # engine default.  A fixed cap rejected large high-order problems on few
    # ranks although the node had memory to spare.
    workspace_cap = resource.operator_workspace_mib
    rank_shares = [
        share
        for share in (
            resource.memory_per_node_mib // resource.ranks_per_node
            if resource.memory_per_node_mib is not None
            else None,
            resource.address_space_per_rank_mib,
        )
        if share is not None
    ]
    if workspace_cap is None and rank_shares:
        workspace_cap = max(1024, min(2**31 - 1, min(rank_shares) // 4))
    resources = {
        "name": resource.name,
        "nodes": resource.nodes,
        "world_size": resource.world_size,
        "ranks_per_node": resource.ranks_per_node,
        "local_rank_map": [
            {
                "rank": rank,
                "node": rank // resource.ranks_per_node,
                "local_rank": rank % resource.ranks_per_node,
            }
            for rank in range(resource.world_size)
        ],
        "openmp_threads": resource.openmp_threads,
        "cores_per_rank": resource.cores_per_rank,
        "total_workers": resource.world_size * resource.openmp_threads,
        "smt": resource.smt,
        "cpu_binding": resource.cpu_binding,
        "memory_binding": resource.memory_binding,
        "numa_policy": resource.numa_policy,
        "mpi_thread_level": resource.mpi_thread_level,
        "openmp_environment": {
            "OMP_DYNAMIC": "TRUE" if resource.openmp.dynamic else "FALSE",
            "OMP_MAX_ACTIVE_LEVELS": str(resource.openmp.max_active_levels),
            "OMP_PLACES": resource.openmp.places,
            "OMP_PROC_BIND": resource.openmp.proc_bind,
        },
        "third_party_threads": resource.third_party_threads.model_dump(mode="json"),
        "active_ranks": active_ranks,
    }
    for name in (
        "memory_per_node_mib",
        "address_space_per_rank_mib",
        "operator_workspace_mib",
        "mechanics_pmg_memory_cap_mib",
        "hydraulic_pmg_memory_cap_mib",
    ):
        if name in resource_values:
            resources[name] = resource_values[name]
    if workspace_cap is not None:
        resources["operator_workspace_mib"] = workspace_cap
    resources["linear_vector_memory_cap_mib"] = vector_cap
    return resources


def _set_path(tree: dict[str, Any], path: str, value: Any) -> None:
    family, field = path.split(".", 1)
    if family == "outputs":
        family = "output"
    target = tree.get(family)
    if not isinstance(target, dict) or "." in field:
        raise ResolutionError("override_not_allowlisted", path)
    if field not in target and setting_adapter(path.split(".", 1)[0], field) is None:
        raise ResolutionError("override_not_allowlisted", path)
    target[field] = deepcopy(value)


def _format_option(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return canonicalize(value).decode("utf-8")


def _solver_options(policies: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    linear = policies["linear"]
    pmg = policies["pmg"]
    typed = {
        "-mechanics_ksp_atol": _format_option(linear["absolute_tolerance"]),
        "-mechanics_ksp_max_it": _format_option(linear["maximum_iterations"]),
        "-mechanics_ksp_rtol": _format_option(linear["relative_tolerance"]),
        "-mechanics_ksp_type": str(linear["ksp_type"]),
        "-mechanics_ksp_gmres_restart": _format_option(linear["restart"]),
        "-mechanics_pc_type": "slope_pmg" if pmg["enabled"] else "none",
    }
    allowlisted_expert = {
        "-mechanics_ksp_monitor",
        "-mechanics_ksp_converged_reason",
        "-mechanics_ksp_view",
        "-mechanics_pc_view",
    }
    expert = linear.get("expert_options", {})
    if not isinstance(expert, Mapping):
        raise ResolutionError("expert_options_not_mapping", "linear")
    for option, value in sorted(expert.items()):
        if option in typed:
            raise ResolutionError("expert_option_conflict", option)
        if option not in allowlisted_expert:
            raise ResolutionError("expert_option_not_allowlisted", option)
        if value is not None and not isinstance(value, str):
            raise ResolutionError("expert_option_value_type", option)
        typed[option] = value
    return typed


# Immutable payloads verified in this process, keyed by file identity.
_VERIFIED_PAYLOADS: dict[tuple[int, int, int, int, str], bool] = {}


def _payload_verified(path: Path, byte_count: int, digest: str) -> bool:
    """Stream-hash a package payload once per process and file identity.

    A sweep resolving many runs of one immutable package hashes its HDF5 once;
    any change of inode, size or modification time forces a new hash.
    """
    status = path.stat()
    key = (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, digest)
    if key not in _VERIFIED_PAYLOADS:
        _VERIFIED_PAYLOADS[key] = (
            status.st_size == byte_count and file_sha256(path) == digest
        )
    return _VERIFIED_PAYLOADS[key]


def _numerical_policy_id(
    problem: CompiledProblemRef,
    policies: Mapping[str, Mapping[str, Any]],
    petsc_options: Mapping[str, Any],
) -> str:
    continuation = deepcopy(policies["continuation"])
    continuation.pop("maximum_accepted_states", None)
    pmg = deepcopy(policies["pmg"])
    pmg.pop("active_ranks", None)
    validation = deepcopy(policies["validation"])
    if (
        validation.get("qualification_mode")
        and problem.analysis != "seepage.steady_unconfined"
    ):
        # Earlier builds recorded this flag but used performance assembly.
        validation["mechanics_assembly_mode"] = "qualification"
    return content_id(
        {
            "kind": "slope_stability.numerical_policy",
            "schema_version": 1,
            "problem_id": problem.problem_id,
            "analysis": {"selector": problem.analysis},
            "continuation": continuation,
            "nonlinear": policies["nonlinear"],
            "linear": policies["linear"],
            "pmg": pmg,
            "deflation": policies["deflation"],
            "validation": validation,
            "petsc_options": petsc_options,
        }
    )


def _record_leaf_trace(
    trace: dict[str, ResolutionTrace],
    prefix: str,
    value: Any,
    source: str,
) -> None:
    if isinstance(value, Mapping):
        for key, item in sorted(value.items()):
            child = f"{prefix}.{key}" if prefix else str(key)
            _record_leaf_trace(trace, child, item, source)
    else:
        trace[prefix] = ResolutionTrace(
            path=prefix,
            source=source,
            value=deepcopy(value),
        )


def _validate_build(
    problem: CompiledProblemRef,
    resource: ResourceRow,
    build: BuildCapabilities,
    *,
    pmg_enabled: bool,
) -> None:
    if problem.schema_version[0] != build.schema_version[0]:
        raise ResolutionError("schema_major_incompatible", "problem/build")
    if not build.mpi:
        raise ResolutionError("build_missing_mpi", build.build_id)
    if resource.openmp_threads > 1 and not build.openmp:
        raise ResolutionError("build_missing_openmp", build.build_id)
    if not build.parallel_hdf5:
        raise ResolutionError("build_missing_parallel_hdf5", build.build_id)
    if pmg_enabled and not (build.hypre and build.parmetis):
        raise ResolutionError("build_missing_pmg_dependency", build.build_id)


def resolve_run(
    *,
    problem: CompiledProblemRef,
    profiles: tuple[Profile, ...] | list[Profile],
    resource: ResourceRow,
    build: BuildCapabilities,
    case_overrides: Mapping[str, Any] | None = None,
    suite_overrides: Mapping[str, Any] | None = None,
    checkpoint: CheckpointPolicy | None = None,
) -> ResolvedRun:
    """Resolve all defaults, profiles, overrides, resources, and options."""
    selected = compose_profiles(profiles)
    required_families = {
        "continuation",
        "nonlinear",
        "linear",
        "pmg",
        "deflation",
        "output",
        "validation",
    }
    missing = required_families.difference(selected)
    if missing:
        raise ResolutionError("missing_profile_family", sorted(missing)[0])
    pure_seepage = problem.analysis == "seepage.steady_unconfined"
    if pure_seepage and "seepage" not in selected:
        raise ResolutionError("missing_profile_family", "seepage")
    provenance_families = set(required_families)
    if "seepage" in selected:
        provenance_families.add("seepage")

    policies: dict[str, dict[str, Any]] = {
        family: _serial_copy(selected[family].policy)
        for family in sorted(required_families)
    }
    trace: dict[str, ResolutionTrace] = {}
    for family, policy in policies.items():
        trace_family = "outputs" if family == "output" else family
        _record_leaf_trace(
            trace,
            trace_family,
            policy,
            f"profile:{selected[family].identity.name}",
        )
    for field, value in (
        ("levels", None),
        ("base_iterations", 2),
        ("hydraulic_base_iterations", None),
        ("pre_smoothing_steps", 2),
        ("post_smoothing_steps", 2),
        ("coarse_redundant_group_size", 0),
    ):
        if field not in policies["pmg"]:
            policies["pmg"][field] = value
            path = f"pmg.{field}"
            trace[path] = ResolutionTrace(
                path=path,
                source="schema_default",
                value=value,
            )
    for field, value in (
        ("stopping_condition", "original_rhs_true_residual"),
        ("recycle_dimension", 0),
    ):
        if field not in policies["linear"]:
            policies["linear"][field] = value
            path = f"linear.{field}"
            trace[path] = ResolutionTrace(
                path=path,
                source="schema_default",
                value=value,
            )
    for field, value in (
        (
            "candidate_source",
            "accepted_newton_and_control_directions",
        ),
        ("recycle", "accepted_steps"),
        ("admission_tolerance", 1.0e-3),
        ("pivot_tolerance", 1.0e-13),
        ("condition_limit", 1.0e12),
        ("invariant_tolerance", 1.0e-9),
        ("reorthogonalization_passes", 2),
        ("dense_factor", "partial_pivot_lu"),
        ("compression", "none"),
        ("compression_dimension", 0),
        ("compression_history_decay", 1.0),
    ):
        if field not in policies["deflation"]:
            policies["deflation"][field] = value
            path = f"deflation.{field}"
            trace[path] = ResolutionTrace(
                path=path,
                source="schema_default",
                value=value,
            )

    if "seepage" in selected:
        seepage = _serial_copy(selected["seepage"].policy)
        tolerance = seepage["linear_tolerance"]
        source = f"profile:{selected['seepage'].identity.name}"
        hydraulic = {
            "linear.hydraulic_relative_tolerance": tolerance,
            "linear.hydraulic_absolute_tolerance": tolerance,
            "linear.hydraulic_maximum_iterations": seepage["linear_max_iter"],
            "nonlinear.hydraulic_relative_tolerance": tolerance,
            "nonlinear.hydraulic_absolute_tolerance": tolerance,
            "nonlinear.hydraulic_maximum_iterations": seepage["nonlinear_max_iter"],
        }
        for path, value in hydraulic.items():
            family, field = path.split(".", 1)
            policies[family][field] = deepcopy(value)
            trace[path] = ResolutionTrace(
                path=path,
                source=source,
                value=deepcopy(value),
            )

    if pure_seepage:
        seepage = _serial_copy(selected["seepage"].policy)
        tolerance = seepage["linear_tolerance"]
        source = f"profile:{selected['seepage'].identity.name}"
        derived = {
            "linear.relative_tolerance": tolerance,
            "linear.maximum_iterations": seepage["linear_max_iter"],
            "nonlinear.relative_tolerance": tolerance,
            "nonlinear.absolute_tolerance": tolerance,
            "nonlinear.maximum_iterations": seepage["nonlinear_max_iter"],
        }
        for path, value in derived.items():
            _set_path(policies, path, value)
            trace[path] = ResolutionTrace(
                path=path,
                source=source,
                value=deepcopy(value),
            )

    resources = resolve_resource(resource, field_order=problem.field_order)
    resource_sources: dict[str, str] = {}
    for source_name, overrides in (
        ("case_override", case_overrides or {}),
        ("suite_sweep", suite_overrides or {}),
    ):
        for path, value in sorted(overrides.items()):
            _validate_override(path, value)
            if path.startswith("resources."):
                name = path.removeprefix("resources.")
                resources[name] = value
                resource_sources[path] = source_name
                continue
            _set_path(policies, path, value)
            trace[path] = ResolutionTrace(
                path=path,
                source=source_name,
                value=deepcopy(value),
            )

    linear = policies["linear"]
    pmg = policies["pmg"]
    if (
        pmg.get("hydraulic_base_iterations") is not None
        and not build.hydraulic_pmg_budget
    ):
        raise ResolutionError(
            "build_missing_hydraulic_pmg_budget",
            "installed engine does not support pmg.hydraulic_base_iterations",
        )
    if (
        "seepage" in selected
        and build.hydraulic_pmg_budget
        and pmg.get("hydraulic_base_iterations") is None
    ):
        # Restore the bounded coarse work used before the mechanics-only
        # optimization reduced the shared base action to two iterations.
        # Keep this explicit in the resolved identity and independently
        # configurable; mechanics continues to use base_iterations.
        pmg["hydraulic_base_iterations"] = linear["maximum_iterations"]
        trace["pmg.hydraulic_base_iterations"] = ResolutionTrace(
            path="pmg.hydraulic_base_iterations",
            source="derived:linear.maximum_iterations",
            value=linear["maximum_iterations"],
        )
    elif pmg.get("hydraulic_base_iterations") is None:
        pmg.pop("hydraulic_base_iterations", None)
        trace.pop("pmg.hydraulic_base_iterations", None)
    if "hydraulic_base_iterations" in pmg and not (
        1 <= pmg["hydraulic_base_iterations"] < 2 ** (build.petsc_int_bits - 1)
    ):
        raise ResolutionError(
            "pmg_hydraulic_base_iterations_out_of_range",
            str(pmg["hydraulic_base_iterations"]),
        )
    deflation = policies["deflation"]
    ksp_type = linear.get("ksp_type")
    if linear.get("operator_class") != problem.operator_class:
        raise ResolutionError(
            "profile_operator_class_mismatch",
            f"{linear.get('operator_class')}/{problem.operator_class}",
        )
    if ksp_type == "cg" and problem.operator_class != "symmetric_positive_definite":
        raise ResolutionError(
            "ksp_operator_incompatible",
            f"{ksp_type}/{problem.operator_class}",
        )
    if (
        linear.get("stopping_condition")
        in {
            "preconditioned_residual_original_rhs",
            "natural_residual_initial",
        }
        and ksp_type != "fcg"
    ):
        raise ResolutionError(
            "linear_stopping_condition_incompatible",
            f"{linear.get('stopping_condition')}/{ksp_type}",
        )
    if pmg.get("coarse_redundant_group_size") not in (0, 1):
        raise ResolutionError(
            "pmg_redundant_group_size_invalid",
            str(pmg.get("coarse_redundant_group_size")),
        )
    if (
        pmg.get("coarse_redundant_group_size")
        and pmg.get("coarse_solver") != "preonly_lu"
    ):
        raise ResolutionError(
            "pmg_redundant_solver_incompatible",
            str(pmg.get("coarse_solver")),
        )
    if linear.get("orthogonalization", "petsc") != "petsc" and ksp_type != "fgmres":
        raise ResolutionError("linear_orthogonalization_incompatible", str(ksp_type))
    if pmg.get("coarse_factorization", "petsc_default") != "petsc_default" and (
        pmg.get("coarse_solver") != "preonly_lu"
    ):
        raise ResolutionError(
            "pmg_coarse_factorization_incompatible", pmg["coarse_solver"]
        )
    accepted_states = policies["continuation"]["maximum_accepted_states"]
    accepted_limit = MAXIMUM_ACCEPTED_STATES - (
        problem.analysis == "limit_load.work_controlled"
    )
    if not 1 <= accepted_states <= accepted_limit:
        raise ResolutionError(
            "continuation_accepted_state_limit",
            f"continuation.maximum_accepted_states must be between 1 and "
            f"{accepted_limit} for {problem.analysis}, got {accepted_states}",
        )
    nonlinear = policies["nonlinear"]
    guard = nonlinear.get("indirect_progress_guard", "off")
    ceiling = nonlinear.get("indirect_progress_guard_maximum", 0.0)
    if (guard == "fixed") != (ceiling > 0.0) or (
        guard != "off"
        and nonlinear.get("indirect_progress_guard_iteration", 4)
        >= nonlinear["maximum_iterations"]
    ):
        raise ResolutionError("nonlinear_progress_guard_invalid", guard)
    if nonlinear.get("indirect_adaptive_linear_tolerance", False) and (
        nonlinear.get("indirect_adaptive_linear_minimum_relative_tolerance", 1.0e-4)
        > linear["relative_tolerance"]
    ):
        raise ResolutionError(
            "adaptive_linear_tolerance_invalid", str(linear["relative_tolerance"])
        )
    if problem.analysis != "ssr.indirect_work_controlled" and (
        guard != "off" or nonlinear.get("indirect_adaptive_linear_tolerance", False)
    ):
        raise ResolutionError(
            "indirect_nonlinear_setting_incompatible", problem.analysis
        )
    requested_levels = pmg.get("levels")
    default_levels = tuple(
        order for order in (1, 2, 4, 8) if order <= problem.field_order
    )
    if requested_levels is None:
        levels = default_levels
        level_source = "schema_default"
    else:
        supported_levels = tuple(requested_levels)
        level_source = trace["pmg.levels"].source
        if len(set(supported_levels)) != len(supported_levels):
            raise ResolutionError("duplicate_pmg_level", repr(supported_levels))
        if (
            tuple(sorted(supported_levels)) != supported_levels
            or not supported_levels
            or any(order not in (1, 2, 4, 8) for order in supported_levels)
            or problem.field_order not in supported_levels
        ):
            raise ResolutionError("invalid_pmg_hierarchy", repr(supported_levels))
        levels = tuple(
            order for order in supported_levels if order <= problem.field_order
        )
    if not levels or levels[-1] != problem.field_order:
        raise ResolutionError("invalid_pmg_hierarchy", repr(levels))
    pmg["levels"] = list(levels)
    trace["pmg.levels"] = ResolutionTrace(
        path="pmg.levels", source=level_source, value=list(levels)
    )
    if deflation.get("enabled") and deflation.get("projector") not in {
        "general_left_right",
        "legacy_a_orthogonal",
    }:
        raise ResolutionError(
            "deflation_projector_incompatible",
            str(deflation.get("projector")),
        )
    if deflation.get("enabled") and deflation.get("maximum_dimension", 0) < 1:
        raise ResolutionError(
            "deflation_dimension_invalid",
            str(deflation.get("maximum_dimension")),
        )
    deflation_lifecycle = (
        deflation.get("candidate_source"),
        deflation.get("recycle"),
    )
    if deflation_lifecycle not in {
        (
            "accepted_newton_and_control_directions",
            "accepted_steps",
        ),
        (
            "accepted_state_vectors",
            "attempt_snapshot_restore",
        ),
        (
            "accepted_states_and_full_krylov",
            "attempt_snapshot_restore",
        ),
        (
            "accepted_states_and_system_solutions",
            "attempt_snapshot_restore",
        ),
    }:
        raise ResolutionError(
            "deflation_lifecycle_incompatible",
            repr(deflation_lifecycle),
        )
    compression = deflation.get("compression", "none")
    compression_dimension = deflation.get("compression_dimension", 0)
    if (compression == "none") != (compression_dimension == 0):
        raise ResolutionError(
            "deflation_compression_incompatible",
            repr((compression, compression_dimension)),
        )
    compression_lifecycle = (
        ("accepted_states_and_system_solutions", "attempt_snapshot_restore")
        if compression == "system_solution_window"
        else ("accepted_states_and_full_krylov", "attempt_snapshot_restore")
    )
    if compression != "none" and (
        deflation_lifecycle != compression_lifecycle
        or deflation.get("projector") != "legacy_a_orthogonal"
        or compression_dimension > deflation.get("maximum_dimension", 0)
    ):
        raise ResolutionError(
            "deflation_compression_incompatible",
            repr((compression, compression_dimension)),
        )
    missing_outputs = set(problem.required_outputs).difference(
        policies["output"].get("quantities", [])
    )
    if missing_outputs:
        raise ResolutionError("missing_required_output", sorted(missing_outputs)[0])

    pmg["active_ranks"] = {
        str(level): (
            resources["active_ranks"]["coarse"]
            if len(levels) > 1 and level == levels[0]
            else resources["active_ranks"]["fine"]
        )
        for level in levels
    }
    for level, ranks in pmg["active_ranks"].items():
        path = f"pmg.active_ranks.{level}"
        trace[path] = ResolutionTrace(
            path=path,
            source=f"resource:{resource.name}",
            value=deepcopy(ranks),
        )
    _validate_build(
        problem,
        resource,
        build,
        pmg_enabled=bool(pmg.get("enabled")),
    )
    _record_leaf_trace(
        trace,
        "resources",
        resources,
        f"resource:{resource.name}",
    )
    for path, source_name in resource_sources.items():
        trace[path] = ResolutionTrace(
            path=path,
            source=source_name,
            value=resources[path.removeprefix("resources.")],
        )
    trace["analysis.selector"] = ResolutionTrace(
        path="analysis.selector",
        source="compiled_problem",
        value=problem.analysis,
    )
    trace["result_policy.atomic_publication"] = ResolutionTrace(
        path="result_policy.atomic_publication",
        source="schema_default",
        value=True,
    )

    petsc_options = _solver_options(policies)
    numerical_policy_id = _numerical_policy_id(
        problem,
        policies,
        petsc_options,
    )
    trace["result_policy.numerical_policy_id"] = ResolutionTrace(
        path="result_policy.numerical_policy_id",
        source="resolved_numerical_policy",
        value=numerical_policy_id,
    )
    result_policy: dict[str, Any] = {
        "atomic_publication": True,
        "terminal_manifest_last": True,
        "numerical_policy_id": numerical_policy_id,
    }
    if checkpoint is not None:
        if not isinstance(checkpoint, CheckpointPolicy):
            raise ResolutionError("checkpoint_policy_type", type(checkpoint).__name__)
        checkpoint_value = checkpoint.model_dump(mode="json", exclude_none=True)
        result_policy["checkpoint"] = checkpoint_value
        _record_leaf_trace(
            trace,
            "result_policy.checkpoint",
            checkpoint_value,
            "checkpoint_request",
        )
    manifest_without_id = {
        "kind": "slope_stability.resolved_run",
        "schema_version": {"major": 1, "minor": 0, "patch": 0},
        "created_by": {
            "tool": "slope-stability-resolver",
            "tool_version": "0.1.0",
            "source_id": problem.problem_id,
        },
        "provenance": {
            "problem_id": problem.problem_id,
            "build_id": build.build_id,
            "profile_ids": {
                family: selected[family].id for family in sorted(provenance_families)
            },
        },
        "extensions": {},
        "identity": {
            "name": f"{problem.problem_id}.resolved",
            "title": "Resolved slope-stability run",
        },
        "problem": {
            "id": problem.problem_id,
            "schema_version": list(problem.schema_version),
            "dimension": problem.dimension,
            "field_order": problem.field_order,
            "coordinate_order": problem.coordinate_order,
            "operator_class": problem.operator_class,
        },
        # Capabilities used only during resolution are bound by build_id; keep
        # the runtime build projection compatible with older native readers.
        "build": build.model_dump(
            mode="json", exclude={"hydraulic_pmg_budget", "petsc_openmp_kernels"}
        ),
        "analysis": {"selector": problem.analysis},
        "continuation": policies["continuation"],
        "nonlinear": policies["nonlinear"],
        "linear": policies["linear"],
        "pmg": policies["pmg"],
        "deflation": policies["deflation"],
        "outputs": policies["output"],
        "diagnostics": {
            "structured_events": True,
            "resolution_trace": True,
        },
        "validation": policies["validation"],
        "resources": resources,
        "petsc_options": petsc_options,
        "result_policy": result_policy,
        "resolution_trace": {
            path: record.model_dump(mode="json")
            for path, record in sorted(trace.items())
        },
    }
    run_id = content_id(manifest_without_id)
    manifest = {"id": run_id, **manifest_without_id}
    canonical_bytes = canonicalize(manifest)
    return ResolvedRun(
        run_id=run_id,
        manifest=manifest,
        trace=trace,
        canonical_bytes=canonical_bytes,
    )


def load_build_capabilities(prefix: str | os.PathLike[str]) -> BuildCapabilities:
    """Read the capabilities of an installed native build.

    ``prefix`` is the installation prefix (``cmake --install``).  The engine
    bytes are re-hashed and must match ``install-manifest.json``, whose build
    identity must match ``build-manifest.json``.

    Raises:
        ResolutionError: The installation is incomplete or inconsistent.
    """
    root = Path(prefix)
    data = root / "share" / "slope-stability"
    try:
        build_manifest = read_json_object(data / "build-manifest.json")
        install_manifest = read_json_object(data / "install-manifest.json")
        engine_digest = file_sha256(root / "bin" / "slope-engine")
    except (OSError, ValueError) as error:
        raise ResolutionError("build_install_unreadable", str(error)) from error
    if install_manifest.get("engine_sha256") != engine_digest:
        raise ResolutionError(
            "build_engine_digest_mismatch",
            "installed engine digest does not match install-manifest.json",
        )
    if install_manifest.get("build_id") != build_manifest.get("build_id"):
        raise ResolutionError(
            "build_identity_mismatch",
            "build and install manifests name different build identities",
        )
    try:
        features = build_manifest["features"]
        dependency = build_manifest["dependency"]
        schema = build_manifest["schema_version"]
        return BuildCapabilities(
            build_id=build_manifest["build_id"],
            executable_id=install_manifest["executable_id"],
            dependency_id=build_manifest["dependency_id"],
            schema_version=(schema["major"], schema["minor"], schema["patch"]),
            petsc_int_bits=dependency["petsc_int_bits"],
            mpi=bool(dependency["mpi"]),
            openmp=features["openmp"],
            parallel_hdf5=features["hdf5_parallel"],
            hypre=features["petsc_hypre"],
            parmetis=features["petsc_partitioner"],
            hydraulic_pmg_budget=features.get("hydraulic_pmg_budget", False),
            petsc_openmp_kernels=features.get("petsc_openmp_kernels", False),
        )
    except (KeyError, TypeError) as error:
        raise ResolutionError("build_manifest_incomplete", repr(error)) from error


def write_resolved_run(run: ResolvedRun, destination: Path) -> Path:
    """Atomically publish ``resolved-run.json`` under its content identity."""
    identity_input = dict(run.manifest)
    supplied_id = identity_input.pop("id", None)
    expected_id = content_id(identity_input)
    if (
        supplied_id != run.run_id
        or expected_id != run.run_id
        or canonicalize(run.manifest) != run.canonical_bytes
    ):
        raise ResolutionError("resolved_run_identity_mismatch", run.run_id)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    package = destination / run.run_id.replace(":", "-")
    target = package / "resolved-run.json"
    if package.exists():
        if not target.is_file() or target.read_bytes() != run.canonical_bytes:
            raise ResolutionError("existing_run_package_mismatch", str(package))
        return target
    temporary = Path(tempfile.mkdtemp(prefix=".slope-run-", dir=str(destination)))
    try:
        temporary_target = temporary / "resolved-run.json"
        temporary_target.write_bytes(run.canonical_bytes)
        if temporary_target.read_bytes() != run.canonical_bytes:
            raise ResolutionError("run_write_verification_failed", run.run_id)
        fsync_path(temporary_target)
        fsync_path(temporary)
        os.replace(temporary, package)
        fsync_path(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target


__all__ = [
    "BuildCapabilities",
    "CheckpointPolicy",
    "CompiledProblemRef",
    "ResolutionError",
    "ResolutionTrace",
    "ResolvedRun",
    "SweepAxis",
    "SweepRow",
    "compose_profiles",
    "expand_sweep",
    "load_build_capabilities",
    "load_compiled_problem",
    "resolve_resource",
    "resolve_run",
    "write_resolved_run",
]
