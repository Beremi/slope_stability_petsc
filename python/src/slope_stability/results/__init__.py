# SPDX-License-Identifier: BSD-2-Clause
"""Strict, PETSc-free readers and scientific comparators for engine results.

:func:`open_result` verifies a terminal ``result.json``/``result.h5``
directory (payload SHA-256, logical HDF5 digests and identity agreement) and
returns a lazy :class:`Result`, or an :class:`IncompleteResult` for a
directory without a terminal envelope. Comparators in :mod:`.compare` align
values by stable entity identity and convert units before applying explicit
tolerances.
:func:`promote_reference` publishes a verified result into a write-once
reference store.
"""

from __future__ import annotations

from .compare import (
    AlignedFields,
    CategoricalComparison,
    CategoricalData,
    CategoricalTolerance,
    ConservationComparison,
    ConservationTolerance,
    FieldComparison,
    FieldData,
    FieldTolerance,
    ScalarComparison,
    ScalarTolerance,
    TrajectoryComparison,
    TrajectoryData,
    TrajectoryTolerance,
    align_fields,
    compare_categorical,
    compare_conservation,
    compare_field,
    compare_scalar,
    compare_trajectory,
)
from .errors import PromotionError, QuantityUnavailableError, ResultValidationError
from .promotion import PromotionAuthority, PromotionRecord, promote_reference
from .reader import open_result
from .views import (
    AnalysisOutcome,
    CategorySelection,
    CategoryView,
    CheckpointView,
    EventRecord,
    EventView,
    FieldSelection,
    FieldView,
    HistoryView,
    IncompleteResult,
    QuadratureSelection,
    QuadratureView,
    QuantityValue,
    ReadStatistics,
    RestartProvenance,
    Result,
    ResultVerification,
)

__all__ = [
    "AlignedFields",
    "AnalysisOutcome",
    "CategoricalComparison",
    "CategoricalData",
    "CategoricalTolerance",
    "CategorySelection",
    "CategoryView",
    "CheckpointView",
    "ConservationComparison",
    "ConservationTolerance",
    "EventRecord",
    "EventView",
    "FieldComparison",
    "FieldData",
    "FieldSelection",
    "FieldTolerance",
    "FieldView",
    "HistoryView",
    "IncompleteResult",
    "PromotionAuthority",
    "PromotionError",
    "PromotionRecord",
    "QuadratureSelection",
    "QuadratureView",
    "QuantityUnavailableError",
    "QuantityValue",
    "ReadStatistics",
    "RestartProvenance",
    "Result",
    "ResultValidationError",
    "ResultVerification",
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
    "open_result",
    "promote_reference",
]
