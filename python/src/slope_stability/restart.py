# SPDX-License-Identifier: BSD-2-Clause
"""Scientific comparison of accepted-boundary restart results.

Checkpoint restore reconstructs PETSc objects and transactional history rows,
so process-local row identifiers are not an equivalence surface.  Accepted
ordinals, stable field identities, scientific values, terminal semantics, and
restart provenance are.
"""

from __future__ import annotations

import math
from pathlib import Path
import struct
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import h5py
import numpy as np
import numpy.typing as npt

from .results import Result, open_result


RestartMode = Literal["same_layout", "repartitioned"]

_STATE_COLUMNS = (
    "factor",
    "work",
    "direct_control",
    "displacement_maximum",
)
_HYDRAULIC_QUANTITIES = {
    "conductivity",
    "darcy_velocity",
    "hydraulic_epsilon",
    "hydraulic_ramp",
    "pore_pressure",
    "pressure_flux",
    "pressure_gradient",
}
_SUPPORT_FIELDS = {
    "basis",
    "category_count",
    "complete",
    "component_count",
    "entity_count",
    "group",
    "id",
    "kind",
    "location",
    "quadrature_id",
    "sample_count",
    "state_count",
    "unit",
    "value_type",
    "version",
}


class RestartComparisonError(ValueError):
    """Restart artifacts are incomplete or structurally incomparable."""


def _summary(result: Result) -> Mapping[str, Any]:
    value = result.envelope.get("summary")
    if not isinstance(value, Mapping):
        raise RestartComparisonError("result summary is missing")
    return value


def _quantity_index(result: Result) -> dict[str, Mapping[str, Any]]:
    raw = _summary(result).get("quantities")
    if not isinstance(raw, list):
        raise RestartComparisonError("result quantity index is missing")
    indexed: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            raise RestartComparisonError("result quantity record is invalid")
        quantity_id = str(item["id"])
        if quantity_id in indexed:
            raise RestartComparisonError(f"duplicate quantity: {quantity_id}")
        indexed[quantity_id] = item
    return indexed


def _state_rows(directory: Path) -> npt.NDArray[np.void]:
    with h5py.File(directory / "result.h5", "r") as handle:
        rows = np.asarray(handle["/states/index/rows"][...])
    if (
        rows.ndim != 1
        or rows.dtype.names is None
        or not {
            "state_id",
            "accepted_ordinal",
            *_STATE_COLUMNS,
        }.issubset(rows.dtype.names)
    ):
        raise RestartComparisonError("accepted-state index is invalid")
    return rows


def _states_by_ordinal(
    rows: npt.NDArray[np.void],
) -> dict[int, np.void]:
    result: dict[int, np.void] = {}
    for row in rows:
        ordinal = int(row["accepted_ordinal"])
        if ordinal in result:
            raise RestartComparisonError(f"duplicate accepted ordinal: {ordinal}")
        result[ordinal] = row
    return result


def _state_ids_by_ordinal(
    rows: npt.NDArray[np.void],
) -> dict[int, int]:
    return {int(row["accepted_ordinal"]): int(row["state_id"]) for row in rows}


def _history_rows(directory: Path, table: str) -> npt.NDArray[np.void]:
    with h5py.File(directory / "result.h5", "r") as handle:
        rows = np.asarray(handle[f"/history/{table}/rows"][...])
    if (
        rows.ndim != 1
        or rows.dtype.names is None
        or not {"ordinal", "control", "residual"}.issubset(rows.dtype.names)
    ):
        raise RestartComparisonError(f"{table} history is invalid")
    return rows


def _same_float(left: float, right: float) -> bool:
    return struct.pack("<d", left) == struct.pack("<d", right)


def _float_comparison(
    left: npt.NDArray[np.float64],
    right: npt.NDArray[np.float64],
    *,
    mode: RestartMode,
    relative_tolerance: float,
) -> tuple[bool, bool, float | None, float | None]:
    if (
        left.shape != right.shape
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        return False, False, None, None
    bitwise = left.tobytes() == right.tobytes()
    difference = np.asarray(left - right, dtype=np.float64)
    absolute_maximum = (
        0.0 if difference.size == 0 else float(np.max(np.abs(difference)))
    )
    reference_l2 = float(np.linalg.norm(right.reshape(-1)))
    difference_l2 = float(np.linalg.norm(difference.reshape(-1)))
    if not all(
        math.isfinite(value)
        for value in (absolute_maximum, reference_l2, difference_l2)
    ):
        return False, bitwise, None, None
    relative_l2 = difference_l2 / max(
        reference_l2,
        1.0,
    )
    passed = bitwise if mode == "same_layout" else relative_l2 <= relative_tolerance
    return passed, bitwise, relative_l2, absolute_maximum


def _quantity_values(
    result: Result,
    metadata: Mapping[str, Any],
    state_id: int,
) -> tuple[
    npt.NDArray[np.uint64],
    npt.NDArray[np.float64] | npt.NDArray[np.uint32],
]:
    quantity_id = str(metadata["id"])
    kind = metadata.get("kind")
    if kind == "field":
        selection = result.field(quantity_id, state=state_id).read()
    elif kind == "quadrature":
        selection = result.quadrature(quantity_id, state=state_id).read()
    elif kind == "quadrature_category":
        selection = result.category(quantity_id, state=state_id).read()
    elif kind == "cell_reduction":
        selection = result.cell_reduction(quantity_id, state=state_id).read()
    else:
        raise RestartComparisonError(
            f"{quantity_id}: unsupported quantity kind {kind!r}"
        )
    return selection.stable_ids, selection.values


def _align_values(
    baseline_ids: npt.NDArray[np.uint64],
    baseline_values: npt.NDArray[Any],
    restart_ids: npt.NDArray[np.uint64],
    restart_values: npt.NDArray[Any],
) -> tuple[npt.NDArray[Any], bool]:
    if (
        baseline_ids.ndim != 1
        or restart_ids.ndim != 1
        or baseline_values.shape[0] != len(baseline_ids)
        or restart_values.shape[0] != len(restart_ids)
    ):
        raise RestartComparisonError("quantity support arrays are inconsistent")
    restart_rows = {
        int(stable_id): index for index, stable_id in enumerate(restart_ids)
    }
    if len(restart_rows) != len(restart_ids):
        raise RestartComparisonError("restart quantity has duplicate stable IDs")
    baseline_identity = tuple(int(value) for value in baseline_ids)
    if set(baseline_identity) != restart_rows.keys():
        return restart_values, False
    order = [restart_rows[stable_id] for stable_id in baseline_identity]
    return np.asarray(restart_values[order, ...]), True


def _quantity_tolerance(quantity_id: str) -> float:
    return 1.0e-8 if quantity_id in _HYDRAULIC_QUANTITIES else 1.0e-2


def _categorical_comparison(
    candidate: npt.NDArray[np.uint32],
    reference: npt.NDArray[np.uint32],
    stable_ids: npt.NDArray[np.uint64],
) -> dict[str, Any]:
    if (
        candidate.shape != reference.shape
        or candidate.ndim != 2
        or reference.shape[0] != len(stable_ids)
        or reference.size == 0
    ):
        return {
            "metric_status": "unavailable",
            "eligible_count": None,
            "mismatch_count": None,
            "mismatch_fraction": None,
            "jaccard": None,
            "worst_stable_id": None,
            "worst_sample_index": None,
        }
    mismatches = candidate != reference
    mismatch_count = int(np.count_nonzero(mismatches))
    eligible_count = int(reference.size)
    categories = np.union1d(candidate, reference)
    intersection = 0
    union = 0
    for category in categories:
        candidate_membership = candidate == category
        reference_membership = reference == category
        intersection += int(
            np.count_nonzero(candidate_membership & reference_membership)
        )
        union += int(np.count_nonzero(candidate_membership | reference_membership))
    jaccard = 1.0 if union == 0 else intersection / union
    worst_stable_id: int | None = None
    worst_sample_index: int | None = None
    if mismatch_count:
        row, sample = np.argwhere(mismatches)[0]
        worst_stable_id = int(stable_ids[int(row)])
        worst_sample_index = int(sample)
    return {
        "metric_status": "available",
        "eligible_count": eligible_count,
        "mismatch_count": mismatch_count,
        "mismatch_fraction": mismatch_count / eligible_count,
        "jaccard": jaccard,
        "worst_stable_id": worst_stable_id,
        "worst_sample_index": worst_sample_index,
    }


def _compare_quantities(
    baseline: Result,
    restarted: Result,
    baseline_states: npt.NDArray[np.void],
    restart_states: npt.NDArray[np.void],
    checkpoint_ordinal: int,
    mode: RestartMode,
) -> tuple[list[dict[str, Any]], bool, bool]:
    baseline_index = _quantity_index(baseline)
    restart_index = _quantity_index(restarted)
    supports_match = baseline_index.keys() == restart_index.keys()
    comparisons: list[dict[str, Any]] = []
    all_bitwise = True
    baseline_state_ids = _state_ids_by_ordinal(baseline_states)
    restart_state_ids = _state_ids_by_ordinal(restart_states)
    for quantity_id in sorted(baseline_index.keys() | restart_index.keys()):
        baseline_metadata = baseline_index.get(quantity_id)
        restart_metadata = restart_index.get(quantity_id)
        if baseline_metadata is None or restart_metadata is None:
            comparisons.append(
                {
                    "id": quantity_id,
                    "kind": "missing",
                    "state_ordinals": [],
                    "supports_match": False,
                    "bitwise": False,
                    "required_by_profile": True,
                    "relative_l2_maximum": None,
                    "absolute_maximum": None,
                    "metric_status": "unavailable",
                    "passed": False,
                }
            )
            supports_match = False
            all_bitwise = False
            continue
        metadata_match = all(
            baseline_metadata.get(field) == restart_metadata.get(field)
            for field in _SUPPORT_FIELDS
        )
        supports_match = supports_match and metadata_match
        baseline_available = {
            ordinal
            for ordinal, state_id in baseline_state_ids.items()
            if ordinal >= checkpoint_ordinal
            and state_id in baseline_metadata.get("state_ids", [])
        }
        restart_available = {
            ordinal
            for ordinal, state_id in restart_state_ids.items()
            if ordinal >= checkpoint_ordinal
            and state_id in restart_metadata.get("state_ids", [])
        }
        ordinals = sorted(baseline_available | restart_available)
        state_sets_match = baseline_available == restart_available and bool(
            baseline_available
        )
        quantity_passed = metadata_match and state_sets_match
        quantity_bitwise = metadata_match and state_sets_match
        quantity_supports_match = metadata_match and state_sets_match
        kind = str(baseline_metadata.get("kind"))
        categorical = kind == "quadrature_category"
        metrics_complete = metadata_match and state_sets_match
        relative_l2_maximum = 0.0
        absolute_maximum = 0.0
        categorical_eligible_count = 0
        categorical_mismatch_count = 0
        maximum_state_mismatch_fraction = 0.0
        minimum_state_unweighted_jaccard = 1.0
        worst_state_ordinal: int | None = None
        worst_stable_id: int | None = None
        worst_sample_index: int | None = None
        for ordinal in ordinals:
            if ordinal not in baseline_available or ordinal not in restart_available:
                quantity_passed = False
                quantity_bitwise = False
                metrics_complete = False
                continue
            baseline_ids, baseline_values = _quantity_values(
                baseline,
                baseline_metadata,
                baseline_state_ids[ordinal],
            )
            restart_ids, restart_values = _quantity_values(
                restarted,
                restart_metadata,
                restart_state_ids[ordinal],
            )
            aligned_restart, state_support_match = _align_values(
                baseline_ids,
                baseline_values,
                restart_ids,
                restart_values,
            )
            supports_match = supports_match and state_support_match
            quantity_supports_match = quantity_supports_match and state_support_match
            if (
                not state_support_match
                or baseline_values.shape != aligned_restart.shape
            ):
                quantity_passed = False
                quantity_bitwise = False
                metrics_complete = False
                continue
            if categorical:
                state_bitwise = baseline_values.tobytes() == aligned_restart.tobytes()
                state_passed = state_bitwise
                state_metrics = _categorical_comparison(
                    np.asarray(aligned_restart, dtype=np.uint32),
                    np.asarray(baseline_values, dtype=np.uint32),
                    baseline_ids,
                )
                if state_metrics["metric_status"] != "available":
                    metrics_complete = False
                else:
                    state_fraction = float(state_metrics["mismatch_fraction"])
                    state_jaccard = float(state_metrics["jaccard"])
                    categorical_eligible_count += int(state_metrics["eligible_count"])
                    categorical_mismatch_count += int(state_metrics["mismatch_count"])
                    minimum_state_unweighted_jaccard = min(
                        minimum_state_unweighted_jaccard,
                        state_jaccard,
                    )
                    if state_fraction > maximum_state_mismatch_fraction:
                        maximum_state_mismatch_fraction = state_fraction
                        worst_state_ordinal = ordinal
                        worst_stable_id = state_metrics["worst_stable_id"]
                        worst_sample_index = state_metrics["worst_sample_index"]
            else:
                state_passed, state_bitwise, state_relative_l2, state_absolute = (
                    _float_comparison(
                        np.asarray(aligned_restart, dtype=np.float64),
                        np.asarray(baseline_values, dtype=np.float64),
                        mode=mode,
                        relative_tolerance=_quantity_tolerance(quantity_id),
                    )
                )
                if state_relative_l2 is None or state_absolute is None:
                    metrics_complete = False
                else:
                    relative_l2_maximum = max(
                        relative_l2_maximum,
                        state_relative_l2,
                    )
                    absolute_maximum = max(absolute_maximum, state_absolute)
            quantity_passed = quantity_passed and state_passed
            quantity_bitwise = quantity_bitwise and state_bitwise
        all_bitwise = all_bitwise and quantity_bitwise
        record: dict[str, Any] = {
            "id": quantity_id,
            "kind": kind,
            "state_ordinals": ordinals,
            "supports_match": quantity_supports_match,
            "bitwise": quantity_bitwise,
            "required_by_profile": mode == "same_layout" or not categorical,
            "relative_l2_maximum": (
                None if categorical or not metrics_complete else relative_l2_maximum
            ),
            "absolute_maximum": (
                None if categorical or not metrics_complete else absolute_maximum
            ),
            "metric_status": "available" if metrics_complete else "unavailable",
            "passed": quantity_passed,
        }
        if categorical:
            record["categorical"] = {
                "comparison": "exact",
                "eligible_count": (
                    categorical_eligible_count if metrics_complete else None
                ),
                "mismatch_count": (
                    categorical_mismatch_count if metrics_complete else None
                ),
                "mismatch_fraction": (
                    categorical_mismatch_count / categorical_eligible_count
                    if metrics_complete and categorical_eligible_count
                    else None
                ),
                "maximum_state_mismatch_fraction": (
                    maximum_state_mismatch_fraction if metrics_complete else None
                ),
                "minimum_state_unweighted_jaccard": (
                    minimum_state_unweighted_jaccard if metrics_complete else None
                ),
                "worst_state_ordinal": (
                    worst_state_ordinal if metrics_complete else None
                ),
                "worst_stable_id": worst_stable_id if metrics_complete else None,
                "worst_sample_index": (
                    worst_sample_index if metrics_complete else None
                ),
            }
        comparisons.append(record)
    return comparisons, supports_match, all_bitwise


def _compare_state_index(
    baseline_rows: npt.NDArray[np.void],
    restart_rows: npt.NDArray[np.void],
    checkpoint_ordinal: int,
    mode: RestartMode,
) -> dict[str, Any]:
    baseline = _states_by_ordinal(baseline_rows)
    restarted = _states_by_ordinal(restart_rows)
    baseline_ordinals = tuple(
        ordinal for ordinal in sorted(baseline) if ordinal >= checkpoint_ordinal
    )
    restart_ordinals = tuple(
        ordinal for ordinal in sorted(restarted) if ordinal >= checkpoint_ordinal
    )
    passed = baseline_ordinals == restart_ordinals and bool(baseline_ordinals)
    bitwise = passed
    maximum_error: float | None = 0.0
    for ordinal in sorted(set(baseline_ordinals) | set(restart_ordinals)):
        if ordinal not in baseline or ordinal not in restarted:
            passed = False
            bitwise = False
            maximum_error = None
            continue
        for column in _STATE_COLUMNS:
            baseline_value = float(baseline[ordinal][column])
            restart_value = float(restarted[ordinal][column])
            if not math.isfinite(baseline_value) or not math.isfinite(restart_value):
                passed = False
                bitwise = False
                maximum_error = None
                continue
            value_bitwise = _same_float(baseline_value, restart_value)
            value_passed = value_bitwise or (
                mode == "repartitioned"
                and math.isclose(
                    baseline_value,
                    restart_value,
                    rel_tol=1.0e-3 if column == "factor" else 1.0e-2,
                    abs_tol=2.0e-3 if column == "factor" else 1.0e-12,
                )
            )
            passed = passed and value_passed
            bitwise = bitwise and value_bitwise
            if maximum_error is not None:
                maximum_error = max(
                    maximum_error,
                    abs(baseline_value - restart_value),
                )
    return {
        "accepted_ordinals": list(baseline_ordinals),
        "bitwise": bitwise,
        "maximum_absolute_error": maximum_error,
        "passed": passed,
    }


def _compare_scalars(
    baseline: Result,
    restarted: Result,
    mode: RestartMode,
) -> dict[str, Any]:
    baseline_raw = _summary(baseline).get("scalars")
    restart_raw = _summary(restarted).get("scalars")
    if not isinstance(baseline_raw, list) or not isinstance(restart_raw, list):
        raise RestartComparisonError("scalar indexes are missing")
    baseline_index = {
        str(record["id"]): record
        for record in baseline_raw
        if isinstance(record, Mapping) and isinstance(record.get("id"), str)
    }
    restart_index = {
        str(record["id"]): record
        for record in restart_raw
        if isinstance(record, Mapping) and isinstance(record.get("id"), str)
    }
    passed = baseline_index.keys() == restart_index.keys() and bool(baseline_index)
    bitwise = passed
    rows: list[dict[str, Any]] = []
    for quantity_id in sorted(baseline_index.keys() | restart_index.keys()):
        left = baseline_index.get(quantity_id)
        right = restart_index.get(quantity_id)
        if left is None or right is None:
            rows.append(
                {
                    "id": quantity_id,
                    "bitwise": False,
                    "absolute_error": None,
                    "metric_status": "unavailable",
                    "passed": False,
                }
            )
            passed = False
            bitwise = False
            continue
        baseline_value = float(left["value"])
        restart_value = float(right["value"])
        finite = math.isfinite(baseline_value) and math.isfinite(restart_value)
        value_bitwise = _same_float(baseline_value, restart_value)
        value_passed = finite and (
            value_bitwise
            or (
                mode == "repartitioned"
                and math.isclose(
                    baseline_value,
                    restart_value,
                    rel_tol=1.0e-3 if quantity_id == "load_factor" else 1.0e-2,
                    abs_tol=2.0e-3 if quantity_id == "load_factor" else 1.0e-12,
                )
            )
        )
        metadata_match = all(
            left.get(field) == right.get(field) for field in ("id", "unit", "version")
        )
        value_passed = value_passed and metadata_match
        passed = passed and value_passed
        bitwise = bitwise and value_bitwise and metadata_match
        rows.append(
            {
                "id": quantity_id,
                "bitwise": value_bitwise,
                "absolute_error": (
                    abs(baseline_value - restart_value) if finite else None
                ),
                "metric_status": "available" if finite else "unavailable",
                "passed": value_passed,
            }
        )
    return {
        "quantities": rows,
        "bitwise": bitwise,
        "passed": passed,
    }


def _compare_accepted_history(
    baseline_directory: Path,
    restart_directory: Path,
    checkpoint_ordinal: int,
    mode: RestartMode,
    *,
    include_checkpoint_boundary: bool,
) -> dict[str, Any]:
    first_ordinal = (
        checkpoint_ordinal if include_checkpoint_boundary else checkpoint_ordinal + 1
    )
    baseline_rows = tuple(
        row
        for row in _history_rows(baseline_directory, "accepted")
        if int(row["ordinal"]) >= first_ordinal
    )
    restart_rows = tuple(
        row
        for row in _history_rows(restart_directory, "accepted")
        if int(row["ordinal"]) >= first_ordinal
    )
    baseline_ordinals = tuple(int(row["ordinal"]) for row in baseline_rows)
    restart_ordinals = tuple(int(row["ordinal"]) for row in restart_rows)
    passed = baseline_ordinals == restart_ordinals and bool(baseline_ordinals)
    bitwise = passed
    maximum_error: float | None = 0.0 if passed else None
    for left, right in zip(baseline_rows, restart_rows, strict=False):
        for column in ("control", "residual"):
            baseline_value = float(left[column])
            restart_value = float(right[column])
            if not math.isfinite(baseline_value) or not math.isfinite(restart_value):
                passed = False
                bitwise = False
                maximum_error = None
                continue
            value_bitwise = _same_float(baseline_value, restart_value)
            value_passed = value_bitwise or (
                mode == "repartitioned"
                and math.isclose(
                    baseline_value,
                    restart_value,
                    rel_tol=1.0e-2,
                    abs_tol=1.0e-10,
                )
            )
            passed = passed and value_passed
            bitwise = bitwise and value_bitwise
            if maximum_error is not None:
                maximum_error = max(
                    maximum_error,
                    abs(baseline_value - restart_value),
                )
    return {
        "accepted_ordinals": list(baseline_ordinals),
        "bitwise": bitwise,
        "maximum_absolute_error": maximum_error,
        "passed": passed,
    }


_RESTART_MODES = ("same_layout", "repartitioned")


def _is_content_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


@dataclass(frozen=True)
class RestartObservation:
    """Outcome of one restart comparison, reduced to the acceptance inputs.

    ``case_id`` and ``variant_id`` are free-form labels of the compared run.
    ``quantity_comparisons`` holds the pass/fail value of every check that the
    restart profile requires (see :func:`required_restart_scientific_checks`).
    """

    case_id: str
    variant_id: str
    mode: RestartMode
    source_result_id: str
    resumed_result_id: str
    terminal_status_match: bool
    supports_match: bool
    bitwise_from_restart: bool
    quantity_comparisons: tuple[bool, ...]


@dataclass(frozen=True)
class RestartDecision:
    """Acceptance decision for a :class:`RestartObservation`."""

    passed: bool
    mode: str
    failed_checks: tuple[str, ...]


def assess_restart(observation: RestartObservation) -> RestartDecision:
    """Apply the distinct same-layout and repartitioned restart contracts.

    Both modes require matching terminal status, matching quantity supports
    and every required scientific check.  A same-layout restart must in
    addition be bitwise identical to the uninterrupted run from the restart
    boundary on; a repartitioned restart is accepted within tolerances.
    """
    if (
        not observation.case_id
        or not observation.variant_id
        or observation.mode not in _RESTART_MODES
        or not _is_content_id(observation.source_result_id)
        or not _is_content_id(observation.resumed_result_id)
        or not observation.quantity_comparisons
        or any(
            not isinstance(value, bool) for value in observation.quantity_comparisons
        )
    ):
        raise RestartComparisonError("restart observation is invalid")
    failed: list[str] = []
    if not observation.terminal_status_match:
        failed.append("terminal_status")
    if not observation.supports_match:
        failed.append("supports")
    if not all(observation.quantity_comparisons):
        failed.append("quantities")
    if observation.mode == "same_layout" and not observation.bitwise_from_restart:
        failed.append("bitwise_from_restart")
    return RestartDecision(
        passed=not failed,
        mode=observation.mode,
        failed_checks=tuple(failed),
    )


def required_restart_scientific_checks(
    *,
    mode: RestartMode,
    state_index: bool,
    scalar_index: bool,
    accepted_history: bool,
    quantities: tuple[bool, ...],
) -> tuple[bool, ...]:
    """Select the checks required by the frozen restart tolerance profile."""

    if mode == "same_layout":
        return (
            state_index,
            scalar_index,
            accepted_history,
            *quantities,
        )
    if mode == "repartitioned":
        # VAL-RESTART-REPARTITIONED delegates to VAL-LAYOUT: terminal
        # factor/control state, published fields, status, and supports are
        # required. Reduction-order-sensitive diagnostic scalars and accepted
        # nonlinear residual history remain reported, but are not an
        # additional cross-layout acceptance surface.
        return (state_index, *quantities)
    raise RestartComparisonError(f"unsupported restart mode: {mode}")


def required_restart_quantity_checks(
    *,
    mode: RestartMode,
    quantities: tuple[Mapping[str, Any], ...],
) -> tuple[bool, ...]:
    """Select value comparisons required by the frozen restart profile.

    ``VAL-RESTART-REPARTITIONED`` delegates to ``VAL-LAYOUT``.  That profile
    requires continuous mechanics/hydraulic fields and exact quantity support
    identities, while full-case categorical labels remain reported
    diagnostics.  Same-layout replay retains exact value comparison for every
    published quantity.
    """

    if mode == "same_layout":
        return tuple(bool(record.get("passed")) for record in quantities)
    if mode == "repartitioned":
        return tuple(
            bool(record.get("passed"))
            for record in quantities
            if record.get("kind") != "quadrature_category"
        )
    raise RestartComparisonError(f"unsupported restart mode: {mode}")


def compare_restart_results(
    baseline_directory: str | Path,
    source_directory: str | Path,
    restart_directory: str | Path,
    *,
    mode: RestartMode,
) -> dict[str, Any]:
    """Compare an uninterrupted result with an accepted-boundary restart."""

    if mode not in {"same_layout", "repartitioned"}:
        raise RestartComparisonError(f"unsupported restart mode: {mode}")
    baseline_path = Path(baseline_directory).resolve(strict=True)
    source_path = Path(source_directory).resolve(strict=True)
    restart_path = Path(restart_directory).resolve(strict=True)
    baseline = open_result(baseline_path)
    source = open_result(source_path)
    restarted = open_result(restart_path)
    try:
        if not all(
            isinstance(result, Result) and result.complete
            for result in (baseline, source, restarted)
        ):
            raise RestartComparisonError("restart comparison requires complete results")
        checkpoints = _summary(source).get("checkpoints")
        if not isinstance(checkpoints, list) or len(checkpoints) != 1:
            raise RestartComparisonError(
                "checkpoint source must publish exactly one checkpoint"
            )
        checkpoint = checkpoints[0]
        if not isinstance(checkpoint, Mapping):
            raise RestartComparisonError("checkpoint record is invalid")
        provenance = restarted.restart
        if (
            provenance is None
            or provenance.parent_checkpoint_id != checkpoint.get("checkpoint_id")
            or provenance.parent_run_id != source.run_id
            or provenance.parent_build_id != source.build_id
            or provenance.parent_accepted_state_id != checkpoint.get("state_id")
            or baseline.problem_id != source.problem_id
            or baseline.problem_id != restarted.problem_id
            or baseline.build_id != source.build_id
            or baseline.build_id != restarted.build_id
            or baseline.mesh_id != source.mesh_id
            or baseline.mesh_id != restarted.mesh_id
            or baseline.geometry_id != source.geometry_id
            or baseline.geometry_id != restarted.geometry_id
        ):
            raise RestartComparisonError("restart provenance or lineage is invalid")

        source_states = _state_rows(source_path)
        baseline_states = _state_rows(baseline_path)
        restart_states = _state_rows(restart_path)
        checkpoint_ordinal = max(int(row["accepted_ordinal"]) for row in source_states)
        state_comparison = _compare_state_index(
            baseline_states,
            restart_states,
            checkpoint_ordinal,
            mode,
        )
        quantity_comparisons, supports_match, quantity_bitwise = _compare_quantities(
            baseline,
            restarted,
            baseline_states,
            restart_states,
            checkpoint_ordinal,
            mode,
        )
        scalar_comparison = _compare_scalars(baseline, restarted, mode)
        history_comparison = _compare_accepted_history(
            baseline_path,
            restart_path,
            checkpoint_ordinal,
            mode,
            # Mechanics continuation reconstructs its checkpoint state as a
            # synthetic origin without a source attempt row.  Its exact
            # trajectory therefore starts after the boundary.  A terminal
            # steady-seepage restart republishes the restored solve evidence,
            # so the boundary row remains part of that comparison.
            include_checkpoint_boundary=baseline.envelope.get(
                "analysis_outcome",
                {},
            ).get("method")
            == "steady_seepage",
        )
        scalar_comparison["required_by_profile"] = mode == "same_layout"
        history_comparison["required_by_profile"] = mode == "same_layout"
        terminal_status_match = baseline.envelope.get(
            "terminal_status"
        ) == restarted.envelope.get("terminal_status") and baseline.envelope.get(
            "analysis_outcome"
        ) == restarted.envelope.get("analysis_outcome")
        bitwise_from_restart = (
            state_comparison["bitwise"]
            and quantity_bitwise
            and scalar_comparison["bitwise"]
            and history_comparison["bitwise"]
        )
        scientific_checks = required_restart_scientific_checks(
            mode=mode,
            state_index=bool(state_comparison["passed"]),
            scalar_index=bool(scalar_comparison["passed"]),
            accepted_history=bool(history_comparison["passed"]),
            quantities=required_restart_quantity_checks(
                mode=mode,
                quantities=tuple(quantity_comparisons),
            ),
        )
        decision = assess_restart(
            RestartObservation(
                case_id="restart-comparison",
                variant_id="canonical",
                mode=mode,
                source_result_id=source.result_id,
                resumed_result_id=restarted.result_id,
                terminal_status_match=terminal_status_match,
                supports_match=supports_match,
                bitwise_from_restart=bitwise_from_restart,
                quantity_comparisons=scientific_checks,
            )
        )
        return {
            "mode": mode,
            "comparator_profile": (
                "VAL-RESTART-SAME-LAYOUT"
                if mode == "same_layout"
                else "VAL-RESTART-REPARTITIONED"
            ),
            "baseline_result_id": baseline.result_id,
            "source_result_id": source.result_id,
            "restart_result_id": restarted.result_id,
            "checkpoint_id": provenance.parent_checkpoint_id,
            "checkpoint_ordinal": checkpoint_ordinal,
            "terminal_status_match": terminal_status_match,
            "supports_match": supports_match,
            "bitwise_from_restart": bitwise_from_restart,
            "state_index": state_comparison,
            "scalars": scalar_comparison,
            "accepted_history": history_comparison,
            "quantities": quantity_comparisons,
            "failed_checks": list(decision.failed_checks),
            "status": "pass" if decision.passed else "fail",
        }
    finally:
        baseline.close()
        source.close()
        restarted.close()


__all__ = [
    "RestartComparisonError",
    "RestartDecision",
    "RestartMode",
    "RestartObservation",
    "assess_restart",
    "compare_restart_results",
    "required_restart_quantity_checks",
    "required_restart_scientific_checks",
]
