# SPDX-License-Identifier: BSD-2-Clause
"""Unit-aware scientific comparators for scalars, fields and trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import numpy.typing as npt
import pint

from .errors import ResultValidationError
from .views import QuantityValue


_UNITS = pint.UnitRegistry()


@dataclass(frozen=True)
class ScalarTolerance:
    """Absolute and relative scalar tolerance."""

    absolute: float
    relative: float


@dataclass(frozen=True)
class ScalarComparison:
    """Outcome of :func:`compare_scalar` in reference units."""

    passed: bool
    candidate_in_reference_units: float
    reference: float
    absolute_error: float
    limit: float
    reason: str


def _conversion_factor(source: str, destination: str) -> float:
    try:
        return float((1.0 * _UNITS(source)).to(destination).magnitude)
    except (pint.DimensionalityError, pint.UndefinedUnitError) as error:
        raise ResultValidationError(
            f"incompatible units: {source} and {destination}"
        ) from error


def compare_scalar(
    candidate: QuantityValue,
    reference: QuantityValue,
    tolerance: ScalarTolerance,
) -> ScalarComparison:
    """Compare two scalars after converting the candidate to reference units.

    Passes when ``|candidate - reference| <= absolute + relative * |reference|``;
    non-finite values fail with reason ``nonfinite``.
    """

    if candidate.quantity_id != reference.quantity_id:
        raise ResultValidationError("scalar quantity identities differ")
    if (
        not math.isfinite(tolerance.absolute)
        or not math.isfinite(tolerance.relative)
        or tolerance.absolute < 0.0
        or tolerance.relative < 0.0
    ):
        raise ResultValidationError("scalar tolerance is invalid")
    factor = _conversion_factor(candidate.unit, reference.unit)
    converted = candidate.value * factor
    limit = tolerance.absolute + tolerance.relative * abs(reference.value)
    if not math.isfinite(converted) or not math.isfinite(reference.value):
        return ScalarComparison(
            False, converted, reference.value, math.inf, limit, "nonfinite"
        )
    error = abs(converted - reference.value)
    return ScalarComparison(
        error <= limit,
        converted,
        reference.value,
        error,
        limit,
        "within_tolerance" if error <= limit else "outside_tolerance",
    )


@dataclass(frozen=True)
class FieldData:
    """Field values keyed by stable entity identity.

    ``values`` has shape ``(len(stable_ids), len(component_ids))``; optional
    ``weights`` give per-entity quadrature weights.
    """

    quantity_id: str
    quantity_version: int
    unit: str
    component_ids: tuple[str, ...]
    stable_ids: npt.NDArray[np.uint64]
    values: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64] | None = None


@dataclass(frozen=True)
class AlignedFields:
    """Two fields aligned on a common stable-identity and component order."""

    stable_ids: npt.NDArray[np.uint64]
    component_ids: tuple[str, ...]
    candidate: npt.NDArray[np.float64]
    reference: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]


def _validate_field(field: FieldData, label: str) -> None:
    if field.quantity_version <= 0 or not field.quantity_id:
        raise ResultValidationError(f"{label} has invalid quantity identity")
    if field.values.ndim != 2 or field.stable_ids.ndim != 1:
        raise ResultValidationError(f"{label} field arrays have invalid rank")
    if field.values.shape[0] != len(field.stable_ids) or field.values.shape[1] != len(
        field.component_ids
    ):
        raise ResultValidationError(f"{label} field arrays disagree")
    if len(np.unique(field.stable_ids)) != len(field.stable_ids):
        raise ResultValidationError(f"{label} has duplicate stable identities")
    if len(set(field.component_ids)) != len(field.component_ids):
        raise ResultValidationError(f"{label} has duplicate components")
    if field.weights is not None and (
        field.weights.shape != field.stable_ids.shape
        or not np.isfinite(field.weights).all()
        or np.any(field.weights <= 0.0)
    ):
        raise ResultValidationError(f"{label} has invalid weights")


def align_fields(candidate: FieldData, reference: FieldData) -> AlignedFields:
    """Align a candidate field to a reference by stable identity.

    Units are converted to the reference unit and components reordered.

    Raises: ResultValidationError: The quantities, identity sets or components
    differ, or a field is malformed.
    """

    _validate_field(candidate, "candidate")
    _validate_field(reference, "reference")
    if (
        candidate.quantity_id != reference.quantity_id
        or candidate.quantity_version != reference.quantity_version
    ):
        raise ResultValidationError("field quantity identities differ")
    candidate_ids = {
        int(value): index for index, value in enumerate(candidate.stable_ids)
    }
    reference_ids = {
        int(value): index for index, value in enumerate(reference.stable_ids)
    }
    if candidate_ids.keys() != reference_ids.keys():
        raise ResultValidationError("field stable identity sets differ")
    if set(candidate.component_ids) != set(reference.component_ids):
        raise ResultValidationError("field component identities differ")
    row_order = [candidate_ids[int(value)] for value in reference.stable_ids]
    component_order = [
        candidate.component_ids.index(component)
        for component in reference.component_ids
    ]
    factor = _conversion_factor(candidate.unit, reference.unit)
    candidate_values = (
        np.asarray(candidate.values[np.ix_(row_order, component_order)], dtype=float)
        * factor
    )
    reference_values = np.asarray(reference.values, dtype=float)
    if (
        not np.isfinite(candidate_values).all()
        or not np.isfinite(reference_values).all()
    ):
        raise ResultValidationError("field values contain nonfinite entries")
    weights = (
        np.ones(len(reference.stable_ids), dtype=float)
        if reference.weights is None
        else np.asarray(reference.weights, dtype=float)
    )
    return AlignedFields(
        np.asarray(reference.stable_ids, dtype=np.uint64),
        reference.component_ids,
        candidate_values,
        reference_values,
        weights,
    )


@dataclass(frozen=True)
class FieldTolerance:
    """Absolute/relative field tolerance and the normalization scale."""

    absolute: float
    relative: float
    scale: float


@dataclass(frozen=True)
class FieldComparison:
    """Per-component L1/L2/Linf errors, limits and the worst entity."""

    passed: bool
    component_l1: npt.NDArray[np.float64]
    component_l2: npt.NDArray[np.float64]
    component_linf: npt.NDArray[np.float64]
    component_scaled: npt.NDArray[np.float64]
    component_limits: npt.NDArray[np.float64]
    worst_stable_id: int
    worst_component_id: str


def compare_field(
    candidate: FieldData,
    reference: FieldData,
    tolerance: FieldTolerance,
) -> FieldComparison:
    """Compare two fields component-wise after alignment.

    Passes when every component's maximum error is at most ``absolute + relative *
    max|reference|``.
    """

    if (
        not all(
            math.isfinite(value)
            for value in (tolerance.absolute, tolerance.relative, tolerance.scale)
        )
        or tolerance.absolute < 0.0
        or tolerance.relative < 0.0
        or tolerance.scale <= 0.0
    ):
        raise ResultValidationError("field tolerance is invalid")
    aligned = align_fields(candidate, reference)
    error = aligned.candidate - aligned.reference
    absolute = np.abs(error)
    weight_sum = float(np.sum(aligned.weights))
    l1 = np.sum(aligned.weights[:, None] * absolute, axis=0) / weight_sum
    l2 = np.sqrt(np.sum(aligned.weights[:, None] * error**2, axis=0) / weight_sum)
    linf = np.max(absolute, axis=0)
    scaled = l2 / tolerance.scale
    reference_linf = np.max(np.abs(aligned.reference), axis=0)
    limits = tolerance.absolute + tolerance.relative * reference_linf
    flat = int(np.argmax(absolute))
    row, component = np.unravel_index(flat, absolute.shape)
    return FieldComparison(
        bool(np.all(linf <= limits)),
        l1,
        l2,
        linf,
        scaled,
        limits,
        int(aligned.stable_ids[row]),
        aligned.component_ids[component],
    )


@dataclass(frozen=True)
class CategoricalData:
    """Per-quadrature-point category labels keyed by stable cell identity."""

    quantity_id: str
    quantity_version: int
    unit: str
    quadrature_id: str
    category_ids: tuple[str, ...]
    stable_ids: npt.NDArray[np.uint64]
    values: npt.NDArray[np.uint32]


@dataclass(frozen=True)
class CategoricalTolerance:
    """Maximum mismatch fraction and minimum weighted Jaccard index."""

    maximum_mismatch_fraction: float
    minimum_weighted_jaccard: float


@dataclass(frozen=True)
class CategoricalComparison:
    """Mismatch statistics of two categorical quadrature fields."""

    passed: bool
    eligible_count: int
    excluded_count: int
    mismatch_count: int
    mismatch_fraction: float
    weighted_jaccard: float
    worst_stable_id: int | None
    worst_quadrature_index: int | None


def _validate_categorical(value: CategoricalData, label: str) -> None:
    if (
        not value.quantity_id
        or value.quantity_version <= 0
        or not value.unit
        or not value.quadrature_id
        or not value.category_ids
        or len(set(value.category_ids)) != len(value.category_ids)
        or value.stable_ids.ndim != 1
        or value.values.ndim != 2
        or value.values.shape[0] != len(value.stable_ids)
        or value.values.shape[1] == 0
        or len(np.unique(value.stable_ids)) != len(value.stable_ids)
        or (
            value.values.size != 0
            and int(np.max(value.values)) >= len(value.category_ids)
        )
    ):
        raise ResultValidationError(f"{label} categorical data is invalid")


def compare_categorical(
    candidate: CategoricalData,
    reference: CategoricalData,
    tolerance: CategoricalTolerance,
    *,
    ambiguous: npt.NDArray[np.bool_] | None = None,
    weights: npt.NDArray[np.float64] | None = None,
) -> CategoricalComparison:
    """Compare categorical quadrature labels (for example plastic branches).

    Points flagged in ``ambiguous`` are excluded; ``weights`` weight the Jaccard
    index.
    """

    _validate_categorical(candidate, "candidate")
    _validate_categorical(reference, "reference")
    if (
        candidate.quantity_id != reference.quantity_id
        or candidate.quantity_version != reference.quantity_version
        or candidate.unit != reference.unit
        or candidate.quadrature_id != reference.quadrature_id
        or set(candidate.category_ids) != set(reference.category_ids)
        or candidate.values.shape[1] != reference.values.shape[1]
    ):
        raise ResultValidationError("categorical quantity identities differ")
    if (
        not math.isfinite(tolerance.maximum_mismatch_fraction)
        or not math.isfinite(tolerance.minimum_weighted_jaccard)
        or not 0.0 <= tolerance.maximum_mismatch_fraction <= 1.0
        or not 0.0 <= tolerance.minimum_weighted_jaccard <= 1.0
    ):
        raise ResultValidationError("categorical tolerance is invalid")
    candidate_rows = {
        int(stable_id): index for index, stable_id in enumerate(candidate.stable_ids)
    }
    reference_rows = {
        int(stable_id): index for index, stable_id in enumerate(reference.stable_ids)
    }
    if candidate_rows.keys() != reference_rows.keys():
        raise ResultValidationError("categorical stable identity sets differ")
    row_order = [candidate_rows[int(stable_id)] for stable_id in reference.stable_ids]
    category_remap = np.asarray(
        [reference.category_ids.index(category) for category in candidate.category_ids],
        dtype=np.uint32,
    )
    candidate_codes = category_remap[
        np.asarray(candidate.values[row_order, :], dtype=np.uint32)
    ]
    reference_codes = np.asarray(reference.values, dtype=np.uint32)
    ambiguity = (
        np.zeros(reference_codes.shape, dtype=np.bool_)
        if ambiguous is None
        else np.asarray(ambiguous, dtype=np.bool_)
    )
    entity_weights = (
        np.ones(len(reference.stable_ids), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if (
        ambiguity.shape != reference_codes.shape
        or entity_weights.shape != reference.stable_ids.shape
        or not np.isfinite(entity_weights).all()
        or np.any(entity_weights <= 0.0)
    ):
        raise ResultValidationError("categorical ambiguity or weights are invalid")
    eligible = ~ambiguity
    eligible_count = int(np.count_nonzero(eligible))
    if eligible_count == 0:
        raise ResultValidationError("categorical comparison has no eligible values")
    mismatches = (candidate_codes != reference_codes) & eligible
    mismatch_count = int(np.count_nonzero(mismatches))
    mismatch_fraction = mismatch_count / eligible_count
    point_weights = np.broadcast_to(entity_weights[:, None], reference_codes.shape)
    intersection = 0.0
    union = 0.0
    for category in range(len(reference.category_ids)):
        candidate_membership = (candidate_codes == category) & eligible
        reference_membership = (reference_codes == category) & eligible
        intersection += float(
            np.sum(point_weights[candidate_membership & reference_membership])
        )
        union += float(
            np.sum(point_weights[candidate_membership | reference_membership])
        )
    weighted_jaccard = 1.0 if union == 0.0 else intersection / union
    worst_stable_id: int | None = None
    worst_quadrature_index: int | None = None
    if mismatch_count != 0:
        row, quadrature = np.argwhere(mismatches)[0]
        worst_stable_id = int(reference.stable_ids[int(row)])
        worst_quadrature_index = int(quadrature)
    passed = (
        mismatch_fraction <= tolerance.maximum_mismatch_fraction
        and weighted_jaccard >= tolerance.minimum_weighted_jaccard
    )
    return CategoricalComparison(
        passed,
        eligible_count,
        int(np.count_nonzero(ambiguity)),
        mismatch_count,
        mismatch_fraction,
        weighted_jaccard,
        worst_stable_id,
        worst_quadrature_index,
    )


@dataclass(frozen=True)
class ConservationTolerance:
    """Maximum normalized balance residual."""

    normalized: float


@dataclass(frozen=True)
class ConservationComparison:
    """Terms and normalized residual of a conservation balance."""

    passed: bool
    support_terms: Mapping[str, float]
    source_terms: Mapping[str, float]
    physical_scale: float
    numerator: float
    denominator: float
    normalized_balance: float


def compare_conservation(
    *,
    support_terms: Mapping[str, float],
    source_terms: Mapping[str, float],
    physical_scale: float,
    tolerance: ConservationTolerance,
) -> ConservationComparison:
    """Check that support and source terms balance.

    The residual ``|sum(terms)|`` is normalized by ``max(physical_scale,
    sum|terms|)``.
    """

    values = tuple(support_terms.values()) + tuple(source_terms.values())
    if (
        not values
        or not all(math.isfinite(value) for value in values)
        or not math.isfinite(physical_scale)
        or physical_scale <= 0.0
        or not math.isfinite(tolerance.normalized)
        or tolerance.normalized < 0.0
    ):
        raise ResultValidationError("conservation inputs are invalid")
    residual = math.fsum(values)
    numerator = abs(residual)
    denominator = max(physical_scale, math.fsum(abs(value) for value in values))
    normalized = numerator / denominator
    return ConservationComparison(
        normalized <= tolerance.normalized,
        MappingProxyType(dict(support_terms)),
        MappingProxyType(dict(source_terms)),
        physical_scale,
        numerator,
        denominator,
        normalized,
    )


@dataclass(frozen=True)
class TrajectoryData:
    """Monotone control/value trajectory with equilibrium residuals."""

    control: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]
    equilibrium: npt.NDArray[np.float64]
    terminal_reason: str


@dataclass(frozen=True)
class TrajectoryTolerance:
    """Pointwise, curve-distance, terminal-control and equilibrium tolerances."""

    pointwise: float
    curve_distance: float
    terminal_control: float
    equilibrium: float


@dataclass(frozen=True)
class TrajectoryComparison:
    """Outcome of :func:`compare_trajectory` on the reference control grid."""

    passed: bool
    monotone: bool
    common_control: npt.NDArray[np.float64]
    candidate_interpolated: npt.NDArray[np.float64]
    pointwise_error: float
    curve_distance: float
    terminal_match: bool
    equilibrium_passed: bool


def _validate_trajectory(value: TrajectoryData, label: str) -> None:
    if (
        value.control.ndim != 1
        or value.values.shape != value.control.shape
        or value.equilibrium.shape != value.control.shape
        or len(value.control) < 2
        or not np.isfinite(value.control).all()
        or not np.isfinite(value.values).all()
        or not np.isfinite(value.equilibrium).all()
    ):
        raise ResultValidationError(f"{label} trajectory arrays are invalid")
    if not np.all(np.diff(value.control) > 0.0):
        raise ResultValidationError(f"{label} control is not strictly monotone")


def compare_trajectory(
    candidate: TrajectoryData,
    reference: TrajectoryData,
    tolerance: TrajectoryTolerance,
) -> TrajectoryComparison:
    """Compare a candidate trajectory with a reference on the reference grid.

    The candidate is linearly interpolated; it must cover the reference control
    interval.
    """

    _validate_trajectory(candidate, "candidate")
    _validate_trajectory(reference, "reference")
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (
            tolerance.pointwise,
            tolerance.curve_distance,
            tolerance.terminal_control,
            tolerance.equilibrium,
        )
    ):
        raise ResultValidationError("trajectory tolerance is invalid")
    if (
        reference.control[0] < candidate.control[0]
        or reference.control[-1] > candidate.control[-1]
    ):
        raise ResultValidationError("candidate does not cover reference control")
    interpolated = np.interp(reference.control, candidate.control, candidate.values)
    errors = interpolated - reference.values
    pointwise = float(np.max(np.abs(errors)))
    interval = float(reference.control[-1] - reference.control[0])
    distance = math.sqrt(float(np.trapezoid(errors**2, reference.control)) / interval)
    terminal_match = (
        candidate.terminal_reason == reference.terminal_reason
        and abs(float(candidate.control[-1] - reference.control[-1]))
        <= tolerance.terminal_control
    )
    equilibrium_passed = (
        float(np.max(np.abs(candidate.equilibrium))) <= tolerance.equilibrium
        and float(np.max(np.abs(reference.equilibrium))) <= tolerance.equilibrium
    )
    passed = (
        pointwise <= tolerance.pointwise
        and distance <= tolerance.curve_distance
        and terminal_match
        and equilibrium_passed
    )
    return TrajectoryComparison(
        passed,
        True,
        np.asarray(reference.control, dtype=float),
        interpolated,
        pointwise,
        distance,
        terminal_match,
        equilibrium_passed,
    )


__all__ = [
    "AlignedFields",
    "CategoricalComparison",
    "CategoricalData",
    "CategoricalTolerance",
    "ConservationComparison",
    "ConservationTolerance",
    "FieldComparison",
    "FieldData",
    "FieldTolerance",
    "ScalarComparison",
    "ScalarTolerance",
    "TrajectoryComparison",
    "TrajectoryData",
    "TrajectoryTolerance",
    "align_fields",
    "compare_categorical",
    "compare_conservation",
    "compare_field",
    "compare_scalar",
    "compare_trajectory",
]
