# SPDX-License-Identifier: BSD-2-Clause
"""Lazy, read-only views over a verified result directory."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import numpy.typing as npt

from ._validate import (
    _HISTORY_TABLES,
    _attribute_text,
    _integer,
    _object,
    _relative_reference,
    _require_digest,
    _string,
)
from .errors import QuantityUnavailableError, ResultValidationError


@dataclass(frozen=True)
class ResultVerification:
    """Integrity checks performed by :func:`open_result`."""

    payload_verified: bool
    logical_digests_verified: bool
    layout_id: str
    payload_bytes: int
    payload_sha256: str


@dataclass(frozen=True)
class AnalysisOutcome:
    """Analysis method and its scientific outcome label."""

    method: str
    outcome: str


@dataclass(frozen=True)
class EventRecord:
    """One timed engine event (wall seconds)."""

    name: str
    seconds: float
    unit: str
    complete: bool


@dataclass(frozen=True)
class EventView:
    """Read-only access to the timed engine events of a run."""

    rows: tuple[EventRecord, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(row.name for row in self.rows)


@dataclass(frozen=True)
class CheckpointView:
    """Accepted-state checkpoint published by a run."""

    state_id: int
    checkpoint_id: str
    relative_path: Path
    complete: bool


@dataclass(frozen=True)
class RestartProvenance:
    """Parent checkpoint, run and build of a restarted run."""

    mode: str
    parent_checkpoint_id: str
    parent_run_id: str
    parent_build_id: str
    parent_accepted_state_id: int
    source_index_bits: int


def _restart_provenance(
    envelope: Mapping[str, Any],
) -> RestartProvenance | None:
    provenance = _object(envelope.get("provenance"), "provenance")
    if "restart" not in provenance:
        return None
    restart = _object(provenance["restart"], "restart provenance")
    expected_keys = {
        "mode",
        "parent_checkpoint_id",
        "parent_run_id",
        "parent_build_id",
        "parent_accepted_state_id",
        "source_index_bits",
    }
    if set(restart) != expected_keys:
        raise ResultValidationError("restart provenance has an invalid schema")
    mode = _string(restart.get("mode"), "restart mode")
    parent_checkpoint_id = _require_digest(
        restart.get("parent_checkpoint_id"),
        "parent checkpoint ID",
    )
    parent_run_id = _require_digest(restart.get("parent_run_id"), "parent run ID")
    parent_build_id = _require_digest(
        restart.get("parent_build_id"),
        "parent build ID",
    )
    parent_accepted_state_id = _integer(
        restart.get("parent_accepted_state_id"),
        "parent accepted state ID",
        minimum=1,
    )
    source_index_bits = _integer(
        restart.get("source_index_bits"),
        "restart source index bits",
        minimum=1,
    )
    run = _object(envelope.get("run"), "run")
    if (
        mode != "accepted_boundary"
        or source_index_bits not in {32, 64}
        or parent_run_id == _require_digest(run.get("run_id"), "run id")
    ):
        raise ResultValidationError("restart provenance is invalid")
    return RestartProvenance(
        mode,
        parent_checkpoint_id,
        parent_run_id,
        parent_build_id,
        parent_accepted_state_id,
        source_index_bits,
    )


@dataclass
class ReadStatistics:
    """Counters of HDF5 reads performed by lazy views."""

    region_reads: int = 0
    full_reads: int = 0
    value_bytes: int = 0


@dataclass(frozen=True)
class FieldSelection:
    """Selected nodal field values with their stable identities."""

    stable_ids: npt.NDArray[np.uint64]
    values: npt.NDArray[np.float64]
    component_ids: tuple[str, ...]


@dataclass(frozen=True)
class QuadratureSelection:
    """Selected quadrature values with their stable cell identities."""

    stable_ids: npt.NDArray[np.uint64]
    values: npt.NDArray[np.float64]
    component_ids: tuple[str, ...]


@dataclass(frozen=True)
class CategorySelection:
    """Selected categorical quadrature labels with stable cell identities."""

    stable_ids: npt.NDArray[np.uint64]
    values: npt.NDArray[np.uint32]
    category_ids: tuple[str, ...]


def _quantity_state_ids(quantity: Mapping[str, Any]) -> tuple[int, ...]:
    raw = quantity.get("state_ids")
    if not isinstance(raw, list) or not raw:
        raise ResultValidationError("quantity state_ids must be a nonempty array")
    state_ids = tuple(_integer(value, "quantity state ID", minimum=1) for value in raw)
    if len(set(state_ids)) != len(state_ids):
        raise ResultValidationError("quantity state IDs are not unique")
    if _integer(
        quantity.get("state_count"),
        "quantity state count",
        minimum=1,
    ) != len(state_ids):
        raise ResultValidationError("quantity state count disagrees with IDs")
    return state_ids


def _component_selection(
    quantity_id: str,
    available: tuple[str, ...],
    requested: Sequence[str] | None,
) -> tuple[tuple[str, ...], slice | list[int]]:
    if requested is None:
        return available, slice(None)
    component_ids = tuple(requested)
    try:
        indexes = [available.index(component) for component in component_ids]
    except ValueError as error:
        raise QuantityUnavailableError(
            f"unknown component for {quantity_id}"
        ) from error
    return component_ids, indexes


class FieldView:
    """An immutable metadata view with explicit HDF5-backed selections."""

    def __init__(
        self,
        hdf5_path: Path,
        quantity: Mapping[str, Any],
        *,
        status: str,
        state: int,
    ) -> None:
        self._hdf5_path = hdf5_path
        self.quantity_id = _string(quantity["id"], "quantity id")
        self.quantity_version = _integer(
            quantity["version"], "quantity version", minimum=1
        )
        self.unit = _string(quantity["unit"], "quantity unit")
        self.state_id = state
        self._state_ids = _quantity_state_ids(quantity)
        self._state_index = self._state_ids.index(state)
        self._group = _string(quantity["group"], "quantity group")
        self._status = status
        self._cache: npt.NDArray[np.float64] | None = None
        self.read_statistics = ReadStatistics()
        with h5py.File(self._hdf5_path, "r") as handle:
            values = handle[f"{self._group}/values"]
            ids = handle[f"{self._group}/stable_ids"]
            states = handle[f"{self._group}/state_ids"]
            if (
                values.ndim != 3
                or ids.ndim != 1
                or states.ndim != 1
                or tuple(int(value) for value in states[...]) != self._state_ids
                or len(ids) != values.shape[1]
                or len(states) != values.shape[0]
            ):
                raise ResultValidationError(
                    f"{self.quantity_id} has invalid field shape"
                )
            self.shape = (int(values.shape[1]), int(values.shape[2]))
            components = _attribute_text(values.attrs["component_ids"], "component_ids")
            self.component_ids = tuple(components.split(","))
            if len(self.component_ids) != self.shape[1]:
                raise ResultValidationError(
                    f"{self.quantity_id} component metadata disagrees with shape"
                )

    @property
    def materialized(self) -> bool:
        return self._cache is not None

    def read(
        self,
        *,
        entity: slice | Sequence[int] = slice(None),
        components: Sequence[str] | None = None,
    ) -> FieldSelection:
        component_ids, component_indexes = _component_selection(
            self.quantity_id,
            self.component_ids,
            components,
        )
        with h5py.File(self._hdf5_path, "r") as handle:
            stable_dataset = handle[f"{self._group}/stable_ids"]
            value_dataset = handle[f"{self._group}/values"]
            if isinstance(entity, slice):
                stable_ids = np.asarray(stable_dataset[entity], dtype=np.uint64)
                row_values = np.asarray(
                    value_dataset[self._state_index, entity, :],
                    dtype=np.float64,
                )
            else:
                indexes = tuple(int(index) for index in entity)
                stable_ids = np.asarray(
                    [stable_dataset[index] for index in indexes],
                    dtype=np.uint64,
                )
                row_values = np.asarray(
                    [value_dataset[self._state_index, index, :] for index in indexes],
                    dtype=np.float64,
                )
            values = np.asarray(row_values[:, component_indexes], dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if not np.isfinite(values).all():
            raise ResultValidationError(
                f"{self.quantity_id} selection contains nonfinite values"
            )
        self.read_statistics.region_reads += 1
        self.read_statistics.value_bytes += values.nbytes
        return FieldSelection(stable_ids, values, component_ids)

    def values(self) -> npt.NDArray[np.float64]:
        if self._cache is None:
            with h5py.File(self._hdf5_path, "r") as handle:
                self._cache = np.asarray(
                    handle[f"{self._group}/values"][self._state_index, ...],
                    dtype=np.float64,
                )
            if not np.isfinite(self._cache).all():
                self._cache = None
                raise ResultValidationError(
                    f"{self.quantity_id} contains nonfinite values"
                )
            self.read_statistics.full_reads += 1
            self.read_statistics.value_bytes += self._cache.nbytes
        result = self._cache.view()
        result.flags.writeable = False
        return result


class QuadratureView:
    """A lazy state/cell/quadrature/component selection over authoritative data."""

    def __init__(
        self,
        hdf5_path: Path,
        quantity: Mapping[str, Any],
        *,
        status: str,
        state: int,
    ) -> None:
        self._hdf5_path = hdf5_path
        self.quantity_id = _string(quantity["id"], "quantity id")
        self.quantity_version = _integer(
            quantity["version"],
            "quantity version",
            minimum=1,
        )
        self.unit = _string(quantity["unit"], "quantity unit")
        self.quadrature_id = _string(
            quantity["quadrature_id"],
            "quadrature ID",
        )
        self.state_id = state
        self._state_ids = _quantity_state_ids(quantity)
        self._state_index = self._state_ids.index(state)
        self._group = _string(quantity["group"], "quantity group")
        self._status = status
        self._cache: npt.NDArray[np.float64] | None = None
        self.read_statistics = ReadStatistics()
        with h5py.File(self._hdf5_path, "r") as handle:
            values = handle[f"{self._group}/values"]
            ids = handle[f"{self._group}/stable_ids"]
            states = handle[f"{self._group}/state_ids"]
            if (
                values.ndim != 4
                or ids.ndim != 1
                or states.ndim != 1
                or tuple(int(value) for value in states[...]) != self._state_ids
                or len(ids) != values.shape[1]
                or len(states) != values.shape[0]
            ):
                raise ResultValidationError(
                    f"{self.quantity_id} has invalid quadrature shape"
                )
            self.shape = (
                int(values.shape[1]),
                int(values.shape[2]),
                int(values.shape[3]),
            )
            components = _attribute_text(
                values.attrs["component_ids"],
                "component_ids",
            )
            self.component_ids = tuple(components.split(","))
            if len(self.component_ids) != self.shape[2]:
                raise ResultValidationError(
                    f"{self.quantity_id} component metadata disagrees with shape"
                )

    @property
    def materialized(self) -> bool:
        return self._cache is not None

    def read(
        self,
        *,
        entity: slice | Sequence[int] = slice(None),
        quadrature: slice | Sequence[int] = slice(None),
        components: Sequence[str] | None = None,
    ) -> QuadratureSelection:
        component_ids, component_indexes = _component_selection(
            self.quantity_id,
            self.component_ids,
            components,
        )
        with h5py.File(self._hdf5_path, "r") as handle:
            stable_dataset = handle[f"{self._group}/stable_ids"]
            value_dataset = handle[f"{self._group}/values"]
            if isinstance(entity, slice):
                stable_ids = np.asarray(stable_dataset[entity], dtype=np.uint64)
                entity_values = np.asarray(
                    value_dataset[self._state_index, entity, :, :],
                    dtype=np.float64,
                )
            else:
                indexes = tuple(int(index) for index in entity)
                stable_ids = np.asarray(
                    [stable_dataset[index] for index in indexes],
                    dtype=np.uint64,
                )
                entity_values = np.asarray(
                    [
                        value_dataset[self._state_index, index, :, :]
                        for index in indexes
                    ],
                    dtype=np.float64,
                )
        if isinstance(quadrature, slice):
            sampled = entity_values[:, quadrature, :]
        else:
            quadrature_indexes = tuple(int(index) for index in quadrature)
            sampled = entity_values[:, quadrature_indexes, :]
        values = np.asarray(sampled[:, :, component_indexes], dtype=np.float64)
        if values.ndim == 2:
            values = values[:, :, None]
        if not np.isfinite(values).all():
            raise ResultValidationError(
                f"{self.quantity_id} selection contains nonfinite values"
            )
        self.read_statistics.region_reads += 1
        self.read_statistics.value_bytes += values.nbytes
        return QuadratureSelection(stable_ids, values, component_ids)

    def values(self) -> npt.NDArray[np.float64]:
        if self._cache is None:
            with h5py.File(self._hdf5_path, "r") as handle:
                self._cache = np.asarray(
                    handle[f"{self._group}/values"][self._state_index, ...],
                    dtype=np.float64,
                )
            if not np.isfinite(self._cache).all():
                self._cache = None
                raise ResultValidationError(
                    f"{self.quantity_id} contains nonfinite values"
                )
            self.read_statistics.full_reads += 1
            self.read_statistics.value_bytes += self._cache.nbytes
        result = self._cache.view()
        result.flags.writeable = False
        return result


class CategoryView:
    """A lazy scalar categorical quadrature selection."""

    def __init__(
        self,
        hdf5_path: Path,
        quantity: Mapping[str, Any],
        *,
        status: str,
        state: int,
    ) -> None:
        self._hdf5_path = hdf5_path
        self.quantity_id = _string(quantity["id"], "quantity id")
        self.quantity_version = _integer(
            quantity["version"],
            "quantity version",
            minimum=1,
        )
        self.unit = _string(quantity["unit"], "quantity unit")
        self.quadrature_id = _string(
            quantity["quadrature_id"],
            "quadrature ID",
        )
        self.state_id = state
        self._state_ids = _quantity_state_ids(quantity)
        self._state_index = self._state_ids.index(state)
        self._group = _string(quantity["group"], "quantity group")
        self._status = status
        self._cache: npt.NDArray[np.uint32] | None = None
        self.read_statistics = ReadStatistics()
        with h5py.File(self._hdf5_path, "r") as handle:
            values = handle[f"{self._group}/values"]
            ids = handle[f"{self._group}/stable_ids"]
            states = handle[f"{self._group}/state_ids"]
            if (
                values.ndim != 3
                or values.dtype.kind != "u"
                or values.dtype.itemsize != 4
                or ids.ndim != 1
                or states.ndim != 1
                or tuple(int(value) for value in states[...]) != self._state_ids
                or len(ids) != values.shape[1]
                or len(states) != values.shape[0]
            ):
                raise ResultValidationError(
                    f"{self.quantity_id} has invalid categorical shape"
                )
            self.shape = (int(values.shape[1]), int(values.shape[2]))
            categories = _attribute_text(
                values.attrs["category_ids"],
                "category_ids",
            )
            self.category_ids = tuple(categories.split(","))
            if len(self.category_ids) != _integer(
                quantity["category_count"],
                "category count",
                minimum=1,
            ) or len(set(self.category_ids)) != len(self.category_ids):
                raise ResultValidationError(
                    f"{self.quantity_id} category metadata is invalid"
                )

    @property
    def materialized(self) -> bool:
        return self._cache is not None

    def read(
        self,
        *,
        entity: slice | Sequence[int] = slice(None),
        quadrature: slice | Sequence[int] = slice(None),
    ) -> CategorySelection:
        with h5py.File(self._hdf5_path, "r") as handle:
            stable_dataset = handle[f"{self._group}/stable_ids"]
            value_dataset = handle[f"{self._group}/values"]
            if isinstance(entity, slice):
                stable_ids = np.asarray(stable_dataset[entity], dtype=np.uint64)
                entity_values = np.asarray(
                    value_dataset[self._state_index, entity, :],
                    dtype=np.uint32,
                )
            else:
                indexes = tuple(int(index) for index in entity)
                stable_ids = np.asarray(
                    [stable_dataset[index] for index in indexes],
                    dtype=np.uint64,
                )
                entity_values = np.asarray(
                    [value_dataset[self._state_index, index, :] for index in indexes],
                    dtype=np.uint32,
                )
        if isinstance(quadrature, slice):
            values = np.asarray(entity_values[:, quadrature], dtype=np.uint32)
        else:
            quadrature_indexes = tuple(int(index) for index in quadrature)
            values = np.asarray(
                entity_values[:, quadrature_indexes],
                dtype=np.uint32,
            )
        if values.size != 0 and int(np.max(values)) >= len(self.category_ids):
            raise ResultValidationError(
                f"{self.quantity_id} contains an unknown category code"
            )
        self.read_statistics.region_reads += 1
        self.read_statistics.value_bytes += values.nbytes
        values.flags.writeable = False
        stable_ids.flags.writeable = False
        return CategorySelection(stable_ids, values, self.category_ids)

    def values(self) -> npt.NDArray[np.uint32]:
        if self._cache is None:
            with h5py.File(self._hdf5_path, "r") as handle:
                self._cache = np.asarray(
                    handle[f"{self._group}/values"][self._state_index, ...],
                    dtype=np.uint32,
                )
            if self._cache.size != 0 and int(np.max(self._cache)) >= len(
                self.category_ids
            ):
                self._cache = None
                raise ResultValidationError(
                    f"{self.quantity_id} contains an unknown category code"
                )
            self.read_statistics.full_reads += 1
            self.read_statistics.value_bytes += self._cache.nbytes
        result = self._cache.view()
        result.flags.writeable = False
        return result


class HistoryView:
    """A lazy immutable view over one typed relational history table."""

    def __init__(self, hdf5_path: Path, table: str) -> None:
        if table not in _HISTORY_TABLES:
            raise QuantityUnavailableError(f"unknown history table: {table}")
        self._hdf5_path = hdf5_path
        self.table = table
        self._path = f"/history/{table}/rows"
        self._cache: npt.NDArray[np.void] | None = None
        self.read_statistics = ReadStatistics()
        with h5py.File(self._hdf5_path, "r") as handle:
            if self._path not in handle:
                raise QuantityUnavailableError(f"history table unavailable: {table}")
            dataset = handle[self._path]
            if dataset.ndim != 1 or dataset.dtype.names is None:
                raise ResultValidationError(
                    f"history table {table} has an invalid layout"
                )
            self.shape = (int(dataset.shape[0]),)
            self.columns = tuple(dataset.dtype.names)

    @property
    def materialized(self) -> bool:
        return self._cache is not None

    def read(
        self,
        rows: slice | Sequence[int] = slice(None),
    ) -> npt.NDArray[np.void]:
        with h5py.File(self._hdf5_path, "r") as handle:
            dataset = handle[self._path]
            if isinstance(rows, slice):
                selected = np.asarray(dataset[rows])
            else:
                indexes = tuple(int(index) for index in rows)
                selected = np.asarray([dataset[index] for index in indexes])
        selected.flags.writeable = False
        self.read_statistics.region_reads += 1
        self.read_statistics.value_bytes += selected.nbytes
        return selected

    def rows(self) -> npt.NDArray[np.void]:
        if self._cache is None:
            with h5py.File(self._hdf5_path, "r") as handle:
                self._cache = np.asarray(handle[self._path][...])
            self._cache.flags.writeable = False
            self.read_statistics.full_reads += 1
            self.read_statistics.value_bytes += self._cache.nbytes
        return self._cache


class IncompleteResult:
    """A directory without a terminal envelope; never a synthetic failure."""

    status = "incomplete"
    complete = False
    result_id = None
    build_id = None
    quantities: tuple[str, ...] = ()

    def __init__(self, path: Path) -> None:
        self.path = path

    def field(self, quantity: str, *, state: int | str) -> FieldView:
        del quantity, state
        raise QuantityUnavailableError("quantities unavailable in incomplete result")

    def quadrature(
        self,
        quantity: str,
        *,
        state: int | str,
    ) -> QuadratureView:
        del quantity, state
        raise QuantityUnavailableError("quantities unavailable in incomplete result")

    def category(
        self,
        quantity: str,
        *,
        state: int | str,
    ) -> CategoryView:
        del quantity, state
        raise QuantityUnavailableError("quantities unavailable in incomplete result")

    def cell_reduction(
        self,
        quantity: str,
        *,
        state: int | str,
    ) -> FieldView:
        del quantity, state
        raise QuantityUnavailableError("quantities unavailable in incomplete result")

    def history(self, table: str) -> HistoryView:
        del table
        raise QuantityUnavailableError("histories unavailable in incomplete result")

    def scalar(self, quantity: str) -> QuantityValue:
        del quantity
        raise QuantityUnavailableError("scalars unavailable in incomplete result")

    def events(self) -> EventView:
        raise QuantityUnavailableError("events unavailable in incomplete result")

    def checkpoint(self, state: int | str) -> CheckpointView:
        del state
        raise QuantityUnavailableError("checkpoints unavailable in incomplete result")

    def close(self) -> None:
        return None


class Result:
    """Validated terminal metadata plus lazy HDF5-backed scientific views."""

    def __init__(
        self,
        path: Path,
        envelope: Mapping[str, Any],
        verification: ResultVerification,
    ) -> None:
        self.path = path
        self.envelope = MappingProxyType(dict(envelope))
        self.verification = verification
        self.result_id = _require_digest(envelope["id"], "result id")
        run = _object(envelope["run"], "run")
        terminal = _object(envelope["terminal_status"], "terminal_status")
        analysis = _object(envelope["analysis_outcome"], "analysis_outcome")
        summary = _object(envelope["summary"], "summary")
        self.run_id = _require_digest(run["run_id"], "run id")
        self.problem_id = _require_digest(run["problem_id"], "problem id")
        self.build_id = _require_digest(run["build_id"], "build id")
        self.mesh_id = _require_digest(run["mesh_id"], "mesh id")
        self.geometry_id = _require_digest(run["geometry_id"], "geometry id")
        self.restart = _restart_provenance(envelope)
        self.status = _string(terminal["status"], "terminal status")
        self.complete = terminal.get("complete") is True
        self.analysis = AnalysisOutcome(
            _string(analysis["method"], "analysis method"),
            _string(analysis["outcome"], "analysis outcome"),
        )
        raw_quantities = summary.get("quantities")
        if not isinstance(raw_quantities, list):
            raise ResultValidationError("summary.quantities must be an array")
        quantity_index: dict[str, Mapping[str, Any]] = {}
        for raw in raw_quantities:
            quantity = _object(raw, "quantity")
            quantity_id = _string(quantity.get("id"), "quantity id")
            if quantity_id in quantity_index:
                raise ResultValidationError(f"duplicate quantity: {quantity_id}")
            quantity_index[quantity_id] = MappingProxyType(dict(quantity))
        self._quantity_index = MappingProxyType(quantity_index)
        self.quantities = tuple(quantity_index)
        raw_scalars = summary.get("scalars")
        if not isinstance(raw_scalars, list):
            raise ResultValidationError("summary.scalars must be an array")
        scalar_index: dict[str, QuantityValue] = {}
        for raw in raw_scalars:
            scalar = _object(raw, "scalar")
            scalar_id = _string(scalar.get("id"), "scalar ID")
            _integer(scalar.get("version"), "scalar version", minimum=1)
            _integer(scalar.get("state_id"), "scalar state ID", minimum=1)
            value = scalar.get("value")
            if (
                scalar_id in scalar_index
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or scalar.get("complete") is not True
            ):
                raise ResultValidationError("scalar summary is invalid")
            scalar_index[scalar_id] = QuantityValue(
                scalar_id,
                float(value),
                _string(scalar.get("unit"), "scalar unit"),
            )
        self._scalar_index = MappingProxyType(scalar_index)
        raw_checkpoints = summary.get("checkpoints")
        if not isinstance(raw_checkpoints, list):
            raise ResultValidationError("summary.checkpoints must be an array")
        checkpoints: list[CheckpointView] = []
        checkpoint_states: set[int] = set()
        checkpoint_ids: set[str] = set()
        for raw in raw_checkpoints:
            checkpoint = _object(raw, "checkpoint")
            state_id = _integer(
                checkpoint.get("state_id"),
                "checkpoint state ID",
                minimum=1,
            )
            checkpoint_id = _require_digest(
                checkpoint.get("checkpoint_id"),
                "checkpoint ID",
            )
            if (
                state_id in checkpoint_states
                or checkpoint_id in checkpoint_ids
                or checkpoint.get("complete") is not True
            ):
                raise ResultValidationError("checkpoint index is invalid")
            checkpoint_states.add(state_id)
            checkpoint_ids.add(checkpoint_id)
            checkpoints.append(
                CheckpointView(
                    state_id,
                    checkpoint_id,
                    _relative_reference(
                        checkpoint.get("path"),
                        "checkpoint path",
                    ),
                    True,
                )
            )
        self._checkpoints = tuple(checkpoints)
        raw_events = summary.get("events")
        if not isinstance(raw_events, list):
            raise ResultValidationError("summary.events must be an array")
        event_rows: list[EventRecord] = []
        event_names: set[str] = set()
        for raw in raw_events:
            event = _object(raw, "event")
            name = _string(event.get("name"), "event name")
            seconds = event.get("seconds")
            if (
                name in event_names
                or isinstance(seconds, bool)
                or not isinstance(seconds, (int, float))
                or not math.isfinite(float(seconds))
                or float(seconds) < 0.0
                or event.get("complete") is not True
            ):
                raise ResultValidationError("event summary is invalid")
            event_names.add(name)
            event_rows.append(
                EventRecord(
                    name,
                    float(seconds),
                    _string(event.get("unit"), "event unit"),
                    True,
                )
            )
        self._events = EventView(tuple(event_rows))
        self._hdf5_path = path / "result.h5"
        self._closed = False

    def _select_quantity(
        self,
        quantity: str,
        *,
        state: int | str,
        kind: str,
    ) -> tuple[Mapping[str, Any], int]:
        if self._closed:
            raise ResultValidationError("result is closed")
        if state == "terminal" and self.status != "succeeded":
            raise QuantityUnavailableError(
                "terminal solution unavailable for non-succeeded result"
            )
        quantity_meta = self._quantity_index.get(quantity)
        if quantity_meta is None:
            raise QuantityUnavailableError(f"quantity unavailable: {quantity}")
        if _string(quantity_meta.get("kind"), "quantity kind") != kind:
            raise QuantityUnavailableError(f"{quantity} is not a {kind} quantity")
        state_ids = _quantity_state_ids(quantity_meta)
        if state == "terminal":
            selected_state = state_ids[-1]
        elif isinstance(state, int) and not isinstance(state, bool) and state > 0:
            selected_state = state
            if state not in state_ids:
                raise QuantityUnavailableError(
                    f"{quantity} is unavailable at state {state}"
                )
        else:
            raise QuantityUnavailableError("invalid state selector")
        if quantity_meta.get("complete") is not True:
            raise QuantityUnavailableError(f"incomplete quantity: {quantity}")
        return quantity_meta, selected_state

    def field(self, quantity: str, *, state: int | str) -> FieldView:
        quantity_meta, selected_state = self._select_quantity(
            quantity,
            state=state,
            kind="field",
        )
        return FieldView(
            self._hdf5_path,
            quantity_meta,
            status=self.status,
            state=selected_state,
        )

    def quadrature(
        self,
        quantity: str,
        *,
        state: int | str,
    ) -> QuadratureView:
        quantity_meta, selected_state = self._select_quantity(
            quantity,
            state=state,
            kind="quadrature",
        )
        return QuadratureView(
            self._hdf5_path,
            quantity_meta,
            status=self.status,
            state=selected_state,
        )

    def category(
        self,
        quantity: str,
        *,
        state: int | str,
    ) -> CategoryView:
        quantity_meta, selected_state = self._select_quantity(
            quantity,
            state=state,
            kind="quadrature_category",
        )
        return CategoryView(
            self._hdf5_path,
            quantity_meta,
            status=self.status,
            state=selected_state,
        )

    def cell_reduction(
        self,
        quantity: str,
        *,
        state: int | str,
    ) -> FieldView:
        quantity_meta, selected_state = self._select_quantity(
            quantity,
            state=state,
            kind="cell_reduction",
        )
        return FieldView(
            self._hdf5_path,
            quantity_meta,
            status=self.status,
            state=selected_state,
        )

    def scalar(self, quantity: str) -> QuantityValue:
        if self._closed:
            raise ResultValidationError("result is closed")
        value = self._scalar_index.get(quantity)
        if value is None:
            raise QuantityUnavailableError(f"scalar unavailable: {quantity}")
        return value

    def events(self) -> EventView:
        if self._closed:
            raise ResultValidationError("result is closed")
        return self._events

    def checkpoint(self, state: int | str) -> CheckpointView:
        if self._closed:
            raise ResultValidationError("result is closed")
        if state == "terminal":
            if self.status != "succeeded":
                raise QuantityUnavailableError(
                    "terminal checkpoint unavailable for non-succeeded result"
                )
            if not self._checkpoints:
                raise QuantityUnavailableError("terminal checkpoint unavailable")
            return self._checkpoints[-1]
        if isinstance(state, int) and not isinstance(state, bool) and state > 0:
            for checkpoint in self._checkpoints:
                if checkpoint.state_id == state:
                    return checkpoint
            raise QuantityUnavailableError(f"checkpoint unavailable at state {state}")
        raise QuantityUnavailableError("invalid checkpoint state selector")

    def history(self, table: str) -> HistoryView:
        if self._closed:
            raise ResultValidationError("result is closed")
        return HistoryView(self._hdf5_path, table)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> Result:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True)
class QuantityValue:
    """Scalar quantity value with its unit."""

    quantity_id: str
    value: float
    unit: str


__all__ = [
    "AnalysisOutcome",
    "CategorySelection",
    "CategoryView",
    "CheckpointView",
    "EventRecord",
    "EventView",
    "FieldSelection",
    "FieldView",
    "HistoryView",
    "IncompleteResult",
    "QuadratureSelection",
    "QuadratureView",
    "QuantityValue",
    "ReadStatistics",
    "RestartProvenance",
    "Result",
    "ResultVerification",
]
