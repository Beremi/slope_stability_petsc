# SPDX-License-Identifier: BSD-2-Clause
"""Opening and integrity verification of native result directories."""

from __future__ import annotations

import hashlib
import math
import os
import struct
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from ._validate import (
    _DIGEST_CHUNK_ROWS,
    _HISTORY_TABLES,
    _OPTIONAL_HISTORY_TABLES,
    _attribute_text,
    _contained_artifact,
    _integer,
    _object,
    _read_json,
    _relative_reference,
    _require_digest,
    _sha256_bytes,
    _sha256_file,
    _sortable_integer,
    _string,
)
from .errors import ResultValidationError
from .views import (
    IncompleteResult,
    Result,
    ResultVerification,
    _quantity_state_ids,
    _restart_provenance,
)


def _dataset_logical_digest(
    ids: h5py.Dataset,
    values: h5py.Dataset,
    semantic: str,
) -> str:
    if ids.ndim != 1 or values.ndim not in {3, 4} or ids.shape[0] != values.shape[1]:
        raise ResultValidationError("quantity datasets have incompatible shapes")
    chunk_digests: list[tuple[int, str]] = []
    rows = int(ids.shape[0])
    for chunk_index, start in enumerate(range(0, rows, _DIGEST_CHUNK_ROWS)):
        stop = min(start + _DIGEST_CHUNK_ROWS, rows)
        stable = np.asarray(ids[start:stop], dtype="<u8")
        categorical = values.dtype.kind == "u" and values.dtype.itemsize == 4
        quantity_values = np.asarray(
            values[:, start:stop, ...],
            dtype="<u4" if categorical else "<f8",
        )
        if not categorical and not np.isfinite(quantity_values).all():
            raise ResultValidationError("quantity contains nonfinite values")
        row_bytes = bytearray()
        for row in range(stop - start):
            row_bytes.extend(struct.pack("<Q", int(stable[row])))
            entity_values = np.ascontiguousarray(
                quantity_values[:, row, ...],
                dtype="<u4" if categorical else "<f8",
            )
            row_bytes.extend(entity_values.tobytes(order="C"))
        chunk_digests.append(
            (chunk_index, hashlib.sha256(bytes(row_bytes)).hexdigest())
        )
    combined = bytearray(semantic.encode("utf-8"))
    combined.append(0)
    for chunk_index, digest in chunk_digests:
        combined.extend(struct.pack("<Q", chunk_index))
        combined.extend(digest.encode("ascii"))
    return _sha256_bytes(bytes(combined))


def _quantity_semantic(
    values: h5py.Dataset,
    *,
    quantity_id: str,
    version: int,
    unit: str,
    group: str,
    kind: str,
) -> str:
    location = _attribute_text(values.attrs["location"], "location")
    basis = _attribute_text(values.attrs["basis"], "basis")
    mesh_id = _attribute_text(values.attrs["mesh_id"], "mesh_id")
    geometry_id = _attribute_text(values.attrs["geometry_id"], "geometry_id")
    field_order = int(values.attrs["field_order"])
    geometry_order = int(values.attrs["geometry_order"])
    states = _attribute_text(values.attrs["state_ids"], "state_ids")
    semantic_ids = _attribute_text(
        values.attrs[
            "category_ids" if kind == "quadrature_category" else "component_ids"
        ],
        "category_ids" if kind == "quadrature_category" else "component_ids",
    )
    quadrature_id = _attribute_text(
        values.attrs["quadrature_id"],
        "quadrature_id",
    )
    expected_root = {
        "field": "/fields",
        "quadrature": "/quadrature",
        "cell_reduction": "/cell_reductions",
        "quadrature_category": "/quadrature_categories",
    }.get(kind)
    if expected_root is None or group != f"{expected_root}/{quantity_id}":
        raise ResultValidationError("quantity group does not match identity")
    shape = ",".join(str(value) for value in values.shape)
    identity_label = "categories" if kind == "quadrature_category" else "components"
    return (
        f"quantity={quantity_id}|version={version}|unit={unit}|kind={kind}|"
        f"location={location}|basis={basis}|quadrature={quadrature_id}|mesh={mesh_id}|"
        f"geometry={geometry_id}|field_order={field_order}|"
        f"geometry_order={geometry_order}|states={states}|"
        f"{identity_label}={semantic_ids}|"
        f"shape={shape}"
    )


def _result_logical_digest(
    handle: h5py.File,
    envelope: Mapping[str, Any],
    quantity_digests: Mapping[str, str],
) -> str:
    run = _object(envelope["run"], "run")
    terminal = _object(envelope["terminal_status"], "terminal_status")
    analysis = _object(envelope["analysis_outcome"], "analysis_outcome")
    summary = _object(envelope["summary"], "summary")
    quantities = [_object(item, "quantity") for item in summary["quantities"]]
    raw_scalars = summary.get("scalars")
    raw_checkpoints = summary.get("checkpoints")
    if not isinstance(raw_scalars, list) or not isinstance(raw_checkpoints, list):
        raise ResultValidationError("scalar/checkpoint indexes must be arrays")
    scalars = [_object(item, "scalar") for item in raw_scalars]
    checkpoints = [_object(item, "checkpoint") for item in raw_checkpoints]
    restart = _restart_provenance(envelope)
    states = np.asarray(handle["/states/index/rows"][...])
    required_state_columns = {
        "state_id",
        "parent_state_id",
        "accepted_ordinal",
        "hydraulic_generation",
        "mechanics_generation",
        "control_generation",
        "factor",
        "work",
        "direct_control",
        "displacement_maximum",
        "history_begin",
        "history_end",
        "complete",
        "accepted",
    }
    if (
        states.ndim != 1
        or states.dtype.names is None
        or not required_state_columns.issubset(states.dtype.names)
    ):
        raise ResultValidationError("accepted-state index has an invalid schema")
    histories: list[np.void] = []
    required_history_columns = {
        "kind",
        "row_id",
        "state_id",
        "attempt_id",
        "parent_row_id",
        "ordinal",
        "control",
        "residual",
        "complete",
        "accepted",
    }
    for expected_kind, table in enumerate(_HISTORY_TABLES, start=1):
        path = f"/history/{table}/rows"
        if table in _OPTIONAL_HISTORY_TABLES and path not in handle:
            continue
        table_rows = np.asarray(handle[path][...])
        if (
            table_rows.ndim != 1
            or table_rows.dtype.names is None
            or not required_history_columns.issubset(table_rows.dtype.names)
            or any(int(row["kind"]) != expected_kind for row in table_rows)
        ):
            raise ResultValidationError(f"history table {table} has an invalid schema")
        histories.extend(table_rows)
    histories.sort(key=lambda row: int(row["row_id"]))
    if [int(row["row_id"]) for row in histories] != list(range(len(histories))):
        raise ResultValidationError("history row identities are not contiguous")
    for row in histories:
        row_id = int(row["row_id"])
        parent = int(row["parent_row_id"])
        accepted = int(row["accepted"])
        kind = int(row["kind"])
        if (
            int(row["complete"]) != 1
            or not math.isfinite(float(row["control"]))
            or not math.isfinite(float(row["residual"]))
            or (row_id == 0 and parent != 0)
            or (row_id != 0 and parent >= row_id)
            or ((kind == 1) != (accepted != 0))
            or (accepted != 0 and int(row["state_id"]) == 0)
        ):
            raise ResultValidationError("history relational invariant failed")
    state_ids: set[int] = set()
    for state in states:
        state_id = int(state["state_id"])
        parent_state_id = int(state["parent_state_id"])
        begin = int(state["history_begin"])
        end = int(state["history_end"])
        if (
            state_id <= 0
            or state_id in state_ids
            or (parent_state_id != 0 and parent_state_id not in state_ids)
            or int(state["complete"]) != 1
            or int(state["accepted"]) != 1
            or begin < 0
            or begin >= end
            or end > len(histories)
            or not all(
                math.isfinite(float(state[name]))
                for name in (
                    "factor",
                    "work",
                    "direct_control",
                    "displacement_maximum",
                )
            )
            or float(state["displacement_maximum"]) < 0.0
        ):
            raise ResultValidationError("accepted-state relational invariant failed")
        accepted_rows = [
            row
            for row in histories[begin:end]
            if int(row["kind"]) == 1
            and int(row["accepted"]) != 0
            and int(row["state_id"]) == state_id
        ]
        if len(accepted_rows) != 1 or any(
            int(row["state_id"]) not in {0, state_id} for row in histories[begin:end]
        ):
            raise ResultValidationError("accepted state has an invalid history cursor")
        state_ids.add(state_id)
    if _integer(
        summary.get("accepted_state_count"),
        "accepted_state_count",
    ) != len(states) or _integer(
        summary.get("history_row_count"), "history_row_count"
    ) != len(histories):
        raise ResultValidationError("result summary disagrees with relational tables")
    prefix = (
        f"run={run['run_id']}|status={terminal['status']}|"
        f"analysis={analysis['method']}|states={len(states)}|"
        f"history={len(histories)}|scalars={len(scalars)}|"
        f"checkpoints={len(checkpoints)}|restart="
        f"{restart.parent_checkpoint_id if restart is not None else '-'}|"
        f"{restart.parent_run_id if restart is not None else '-'}|"
        f"{restart.parent_build_id if restart is not None else '-'}|"
        f"{restart.parent_accepted_state_id if restart is not None else 0}|"
        f"{restart.source_index_bits if restart is not None else 0}|"
    )
    payload = bytearray(prefix.encode("utf-8"))
    for quantity in sorted(quantities, key=lambda item: str(item["id"])):
        quantity_id = _string(quantity["id"], "quantity id")
        kind = _string(quantity["kind"], "quantity kind")
        kind_code = {
            "field": 1,
            "quadrature": 2,
            "cell_reduction": 3,
            "quadrature_category": 4,
        }.get(kind)
        if kind_code is None:
            raise ResultValidationError(f"unknown quantity kind: {kind}")
        quantity_states = _quantity_state_ids(quantity)
        if any(state not in state_ids for state in quantity_states):
            raise ResultValidationError(f"{quantity_id} references a nonaccepted state")
        state_text = ",".join(str(state) for state in quantity_states)
        payload.extend(
            (
                f"{quantity_id}|{quantity['version']}|{quantity['unit']}|"
                f"{kind_code}|{quantity['component_count']}|"
                f"{quantity['category_count']}|"
                f"{quantity['sample_count']}|{quantity['entity_count']}|"
                f"{state_text}|{quantity_digests[quantity_id]}|"
            ).encode("utf-8")
        )
    scalar_ids: set[str] = set()
    for scalar in sorted(
        scalars,
        key=lambda item: (
            str(item.get("id")),
            _sortable_integer(item.get("state_id")),
        ),
    ):
        scalar_id = _string(scalar.get("id"), "scalar ID")
        version = _integer(scalar.get("version"), "scalar version", minimum=1)
        unit = _string(scalar.get("unit"), "scalar unit")
        state_id = _integer(scalar.get("state_id"), "scalar state ID", minimum=1)
        value = scalar.get("value")
        if (
            scalar_id in scalar_ids
            or scalar_id in quantity_digests
            or state_id not in state_ids
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or scalar.get("complete") is not True
        ):
            raise ResultValidationError("scalar relational invariant failed")
        scalar_ids.add(scalar_id)
        payload.extend(
            f"scalar={scalar_id}|{version}|{unit}|{state_id}|".encode("utf-8")
        )
        payload.extend(struct.pack("<d", float(value)))
        payload.extend(struct.pack("<Q", 1))
    checkpoint_states: set[int] = set()
    checkpoint_ids: set[str] = set()
    checkpoint_paths: set[str] = set()
    for checkpoint in sorted(
        checkpoints,
        key=lambda item: (
            _sortable_integer(item.get("state_id")),
            str(item.get("checkpoint_id")),
        ),
    ):
        state_id = _integer(
            checkpoint.get("state_id"),
            "checkpoint state ID",
            minimum=1,
        )
        checkpoint_id = _require_digest(
            checkpoint.get("checkpoint_id"),
            "checkpoint ID",
        )
        relative_path = _relative_reference(
            checkpoint.get("path"),
            "checkpoint path",
        ).as_posix()
        if (
            state_id not in state_ids
            or state_id in checkpoint_states
            or checkpoint_id in checkpoint_ids
            or relative_path in checkpoint_paths
            or checkpoint.get("complete") is not True
        ):
            raise ResultValidationError("checkpoint relational invariant failed")
        checkpoint_states.add(state_id)
        checkpoint_ids.add(checkpoint_id)
        checkpoint_paths.add(relative_path)
        payload.extend(
            (f"checkpoint={state_id}|{checkpoint_id}|{relative_path}|1|").encode(
                "utf-8"
            )
        )
    for state in states:
        payload.extend(struct.pack("<Q", int(state["state_id"])))
        payload.extend(struct.pack("<Q", int(state["parent_state_id"])))
        payload.extend(struct.pack("<Q", int(state["accepted_ordinal"])))
        payload.extend(struct.pack("<Q", int(state["hydraulic_generation"])))
        payload.extend(struct.pack("<Q", int(state["mechanics_generation"])))
        payload.extend(struct.pack("<Q", int(state["control_generation"])))
        payload.extend(struct.pack("<d", float(state["factor"])))
        payload.extend(struct.pack("<d", float(state["work"])))
        payload.extend(struct.pack("<d", float(state["direct_control"])))
        payload.extend(struct.pack("<d", float(state["displacement_maximum"])))
        payload.extend(struct.pack("<Q", int(state["history_begin"])))
        payload.extend(struct.pack("<Q", int(state["history_end"])))
        payload.extend(struct.pack("<Q", int(state["complete"] != 0)))
        payload.extend(struct.pack("<Q", int(state["accepted"] != 0)))
    for row in histories:
        payload.extend(struct.pack("<Q", int(row["kind"])))
        payload.extend(struct.pack("<Q", int(row["row_id"])))
        payload.extend(struct.pack("<Q", int(row["state_id"])))
        payload.extend(struct.pack("<Q", int(row["attempt_id"])))
        payload.extend(struct.pack("<Q", int(row["parent_row_id"])))
        payload.extend(struct.pack("<Q", int(row["ordinal"])))
        payload.extend(struct.pack("<d", float(row["control"])))
        payload.extend(struct.pack("<d", float(row["residual"])))
        payload.extend(struct.pack("<Q", int(row["complete"] != 0)))
        payload.extend(struct.pack("<Q", int(row["accepted"] != 0)))
    return _sha256_bytes(bytes(payload))


def _validate_hdf5(
    hdf5_path: Path, envelope: Mapping[str, Any]
) -> tuple[str, dict[str, str]]:
    summary = _object(envelope["summary"], "summary")
    quantities = summary.get("quantities")
    if not isinstance(quantities, list):
        raise ResultValidationError("summary.quantities must be an array")
    quantity_digests: dict[str, str] = {}
    required_groups = (
        "/states",
        "/fields",
        "/quadrature",
        "/quadrature_categories",
        "/cell_reductions",
        "/history",
        "/diagnostics",
    )
    with h5py.File(hdf5_path, "r") as handle:
        layout_id = _attribute_text(handle.attrs.get("layout_id"), "layout_id")
        if layout_id != "result-v1" or int(handle.attrs.get("schema_major", -1)) != 1:
            raise ResultValidationError("unsupported result HDF5 layout")
        plan_digest = _require_digest(
            summary.get("collective_plan_digest"),
            "collective plan digest",
        )
        if (
            _attribute_text(
                handle.attrs.get("collective_plan_version"),
                "collective_plan_version",
            )
            != "output-call-plan-v1"
            or _require_digest(
                _attribute_text(
                    handle.attrs.get("collective_plan_digest"),
                    "collective_plan_digest",
                ),
                "HDF5 collective plan digest",
            )
            != plan_digest
            or int(handle.attrs.get("collective_plan_frozen", 0)) != 1
        ):
            raise ResultValidationError("collective output plan is not frozen")
        run = _object(envelope["run"], "run")
        analysis = _object(envelope["analysis_outcome"], "analysis_outcome")
        root_text_attributes = {
            "problem_id": run["problem_id"],
            "run_id": run["run_id"],
            "build_id": run["build_id"],
            "mesh_id": run["mesh_id"],
            "geometry_id": run["geometry_id"],
            "analysis_method": analysis["method"],
        }
        if any(
            _attribute_text(handle.attrs.get(name), name) != expected
            for name, expected in root_text_attributes.items()
        ):
            raise ResultValidationError("result envelope and HDF5 identities disagree")
        for name in ("dimension", "field_order", "geometry_order"):
            if int(handle.attrs.get(name, -1)) != _integer(
                summary.get(name),
                name,
                minimum=1,
            ):
                raise ResultValidationError(f"result envelope and HDF5 {name} disagree")
        for group in required_groups:
            if group not in handle:
                raise ResultValidationError(f"missing HDF5 group: {group}")
        for raw in quantities:
            quantity = _object(raw, "quantity")
            quantity_id = _string(quantity.get("id"), "quantity id")
            version = _integer(quantity.get("version"), "quantity version", minimum=1)
            unit = _string(quantity.get("unit"), "quantity unit")
            group = _string(quantity.get("group"), "quantity group")
            kind = _string(quantity.get("kind"), "quantity kind")
            location = _string(quantity.get("location"), "quantity location")
            basis = _string(quantity.get("basis"), "quantity basis")
            quadrature_id = _string(
                quantity.get("quadrature_id"),
                "quantity quadrature ID",
            )
            state_ids = _quantity_state_ids(quantity)
            component_count = _integer(
                quantity.get("component_count"),
                "component_count",
                minimum=0,
            )
            category_count = _integer(
                quantity.get("category_count"),
                "category_count",
                minimum=0,
            )
            sample_count = _integer(
                quantity.get("sample_count"),
                "sample_count",
                minimum=1,
            )
            entity_count = _integer(
                quantity.get("entity_count"),
                "entity_count",
                minimum=1,
            )
            expected_root = {
                "field": "/fields",
                "quadrature": "/quadrature",
                "cell_reduction": "/cell_reductions",
                "quadrature_category": "/quadrature_categories",
            }.get(kind)
            if expected_root is None:
                raise ResultValidationError(f"unknown quantity kind: {kind}")
            if group != f"{expected_root}/{quantity_id}":
                raise ResultValidationError(
                    f"{quantity_id} group does not match its kind"
                )
            if (
                (
                    kind not in {"quadrature", "quadrature_category"}
                    and sample_count != 1
                )
                or (
                    kind in {"quadrature", "quadrature_category"}
                    and quadrature_id == "none"
                )
                or (
                    kind not in {"quadrature", "quadrature_category"}
                    and quadrature_id != "none"
                )
                or (
                    kind == "quadrature_category"
                    and (component_count != 0 or category_count == 0)
                )
                or (
                    kind != "quadrature_category"
                    and (component_count == 0 or category_count != 0)
                )
            ):
                raise ResultValidationError(
                    f"{quantity_id} has inconsistent sampling metadata"
                )
            if group not in handle:
                raise ResultValidationError(f"missing quantity group: {group}")
            ids = handle[f"{group}/stable_ids"]
            states = handle[f"{group}/state_ids"]
            values = handle[f"{group}/values"]
            identity_attribute = (
                "category_ids" if kind == "quadrature_category" else "component_ids"
            )
            required_attributes = (
                "quantity_id",
                "quantity_version",
                "unit",
                "mesh_id",
                "geometry_id",
                "state_ids",
                "completeness",
                "value_type",
                "digest_method",
                "digest",
                "location",
                "basis",
                identity_attribute,
                "logical_shape",
                "logical_layout_id",
                "state_count",
                "dataset_kind",
                "quadrature_id",
                "sample_count",
                "field_order",
                "geometry_order",
                "value_precision_bits",
                "stable_id_mapping",
            )
            for attribute in required_attributes:
                if attribute not in values.attrs:
                    raise ResultValidationError(
                        f"{quantity_id} lacks attribute {attribute}"
                    )
            categorical = kind == "quadrature_category"
            expected_value_type = "uint32" if categorical else "float64"
            expected_digest_method = (
                "sha256-semantic-stable-id-state-sample-uint32le-chunks-v1"
                if categorical
                else "sha256-semantic-stable-id-state-sample-float64le-chunks-v2"
            )
            expected_layout = (
                "state-entity-quadrature-v1"
                if categorical
                else (
                    "state-entity-quadrature-component-v1"
                    if kind == "quadrature"
                    else "state-entity-component-v1"
                )
            )
            expected_shape = (
                (len(state_ids), entity_count, sample_count)
                if categorical
                else (
                    (
                        len(state_ids),
                        entity_count,
                        sample_count,
                        component_count,
                    )
                    if kind == "quadrature"
                    else (len(state_ids), entity_count, component_count)
                )
            )
            if (
                _attribute_text(values.attrs["quantity_id"], "quantity_id")
                != quantity_id
                or int(values.attrs["quantity_version"]) != version
                or _attribute_text(values.attrs["unit"], "unit") != unit
                or _attribute_text(values.attrs["mesh_id"], "mesh_id") != run["mesh_id"]
                or _attribute_text(values.attrs["geometry_id"], "geometry_id")
                != run["geometry_id"]
                or _attribute_text(values.attrs["completeness"], "completeness")
                != "complete"
                or _attribute_text(values.attrs["value_type"], "value_type")
                != expected_value_type
                or _string(quantity.get("value_type"), "quantity value type")
                != expected_value_type
                or _attribute_text(values.attrs["digest_method"], "digest_method")
                != expected_digest_method
                or _attribute_text(
                    values.attrs["logical_layout_id"],
                    "logical_layout_id",
                )
                != expected_layout
                or _attribute_text(values.attrs["dataset_kind"], "dataset_kind") != kind
                or _attribute_text(values.attrs["location"], "location") != location
                or _attribute_text(values.attrs["basis"], "basis") != basis
                or _attribute_text(values.attrs["quadrature_id"], "quadrature_id")
                != quadrature_id
                or int(values.attrs["sample_count"]) != sample_count
                or int(values.attrs["state_count"]) != len(state_ids)
                or _attribute_text(values.attrs["state_ids"], "state_ids")
                != ",".join(str(state) for state in state_ids)
                or int(values.attrs["field_order"]) != int(handle.attrs["field_order"])
                or int(values.attrs["geometry_order"])
                != int(handle.attrs["geometry_order"])
                or int(values.attrs["value_precision_bits"])
                != (32 if categorical else 64)
                or _attribute_text(
                    values.attrs["stable_id_mapping"],
                    "stable_id_mapping",
                )
                != "canonical_ascending_v1"
                or values.shape != expected_shape
            ):
                raise ResultValidationError(
                    f"{quantity_id} envelope/HDF5 metadata disagree"
                )
            if (
                categorical and (values.dtype.kind != "u" or values.dtype.itemsize != 4)
            ) or (
                not categorical
                and (values.dtype.kind != "f" or values.dtype.itemsize != 8)
            ):
                raise ResultValidationError(
                    f"{quantity_id} has the wrong value scalar type"
                )
            semantic_ids = tuple(
                _attribute_text(
                    values.attrs[identity_attribute],
                    identity_attribute,
                ).split(",")
            )
            if (
                categorical
                and (
                    len(semantic_ids) != category_count
                    or len(set(semantic_ids)) != len(semantic_ids)
                )
            ) or (
                not categorical
                and (
                    len(semantic_ids) != component_count
                    or len(set(semantic_ids)) != len(semantic_ids)
                )
            ):
                raise ResultValidationError(
                    f"{quantity_id} semantic identities disagree"
                )
            if ids.dtype.kind != "u" or ids.dtype.itemsize != 8:
                raise ResultValidationError(f"{quantity_id} IDs are not uint64")
            if (
                states.ndim != 1
                or states.dtype.kind != "u"
                or states.dtype.itemsize != 8
                or tuple(int(state) for state in states[...]) != state_ids
            ):
                raise ResultValidationError(f"{quantity_id} state identities disagree")
            previous_id: int | None = None
            for start in range(0, len(ids), _DIGEST_CHUNK_ROWS):
                stable_chunk = np.asarray(
                    ids[start : start + _DIGEST_CHUNK_ROWS],
                    dtype=np.uint64,
                )
                if (
                    previous_id is not None
                    and len(stable_chunk) != 0
                    and int(stable_chunk[0]) <= previous_id
                ) or (
                    len(stable_chunk) > 1
                    and not np.all(stable_chunk[1:] > stable_chunk[:-1])
                ):
                    raise ResultValidationError(
                        f"{quantity_id} stable identities are not canonical"
                    )
                if len(stable_chunk) != 0:
                    previous_id = int(stable_chunk[-1])
                if categorical:
                    category_chunk = np.asarray(
                        values[:, start : start + _DIGEST_CHUNK_ROWS, :],
                        dtype=np.uint32,
                    )
                    if (
                        category_chunk.size != 0
                        and int(np.max(category_chunk)) >= category_count
                    ):
                        raise ResultValidationError(
                            f"{quantity_id} contains an unknown category code"
                        )
            expected_shape = "[" + ",".join(str(value) for value in values.shape) + "]"
            if (
                _attribute_text(values.attrs["logical_shape"], "logical_shape")
                != expected_shape
            ):
                raise ResultValidationError(
                    f"{quantity_id} logical shape metadata disagrees"
                )
            for attribute in required_attributes:
                if attribute not in ids.attrs:
                    raise ResultValidationError(
                        f"{quantity_id} identity dataset lacks {attribute}"
                    )
            semantic = _quantity_semantic(
                values,
                quantity_id=quantity_id,
                version=version,
                unit=unit,
                group=group,
                kind=kind,
            )
            digest = _dataset_logical_digest(ids, values, semantic)
            expected = _require_digest(
                quantity.get("logical_digest"),
                f"{quantity_id} logical digest",
            )
            attribute_digest = _attribute_text(values.attrs["digest"], "digest")
            id_attribute_digest = _attribute_text(ids.attrs["digest"], "digest")
            if (
                digest != expected
                or digest[7:] != attribute_digest
                or attribute_digest != id_attribute_digest
            ):
                raise ResultValidationError(
                    f"{quantity_id} logical dataset digest mismatch"
                )
            quantity_digests[quantity_id] = digest
        logical = _result_logical_digest(handle, envelope, quantity_digests)
    return logical, quantity_digests


def open_result(path: str | os.PathLike[str]) -> Result | IncompleteResult:
    """Open and verify a native result directory.

    The result.json envelope is read strictly, result.h5 is checked against its
    recorded SHA-256 and logical digests, and all identities must agree.

    Returns: A lazy :class:`Result` when a terminal envelope exists (also for failed
    runs), otherwise an :class:`IncompleteResult` whose quantity accessors raise
    :class:`QuantityUnavailableError`.

    Raises: ResultValidationError: An artifact is malformed or fails verification.
    """

    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise ResultValidationError(f"result directory does not exist: {root}")
    if not root.is_dir():
        raise ResultValidationError(f"result path is not a directory: {root}")
    envelope_path = root / "result.json"
    if not envelope_path.is_file():
        return IncompleteResult(root)
    envelope = _read_json(envelope_path)
    if envelope.get("kind") != "slope_stability.result":
        raise ResultValidationError("terminal envelope kind is not a result")
    version = _object(envelope.get("schema_version"), "schema_version")
    if _integer(version.get("major"), "schema major", minimum=1) != 1:
        raise ResultValidationError("unsupported result schema major version")
    result_id = _require_digest(envelope.get("id"), "result id")
    terminal = _object(envelope.get("terminal_status"), "terminal_status")
    status = _string(terminal.get("status"), "terminal status")
    if status not in {"succeeded", "failed_numerical", "cancelled"}:
        raise ResultValidationError("unknown terminal result status")
    if terminal.get("complete") is not True:
        raise ResultValidationError("terminal result is not complete")
    run = _object(envelope.get("run"), "run")
    for key in ("run_id", "problem_id", "build_id", "mesh_id", "geometry_id"):
        _require_digest(run.get(key), key)
    artifacts = _object(envelope.get("artifacts"), "artifacts")
    payload = _object(artifacts.get("result.h5"), "result.h5 artifact")
    environment_artifact = _object(
        artifacts.get("environment.json"),
        "environment.json artifact",
    )
    hdf5_path = _contained_artifact(root, payload.get("path"))
    if (
        hdf5_path.name != "result.h5"
        or hdf5_path.is_symlink()
        or not hdf5_path.is_file()
    ):
        raise ResultValidationError("authoritative result.h5 is missing")
    expected_bytes = _integer(payload.get("bytes"), "payload bytes", minimum=1)
    expected_sha = _string(payload.get("sha256"), "payload digest")
    if len(expected_sha) != 64 or any(
        byte not in "0123456789abcdef" for byte in expected_sha
    ):
        raise ResultValidationError("payload digest is not sha256")
    actual_sha, actual_bytes = _sha256_file(hdf5_path)
    if actual_sha != expected_sha or actual_bytes != expected_bytes:
        raise ResultValidationError("payload digest or byte count mismatch")
    environment_path = _contained_artifact(
        root,
        environment_artifact.get("path"),
    )
    if (
        environment_path.name != "environment.json"
        or environment_path.is_symlink()
        or not environment_path.is_file()
    ):
        raise ResultValidationError("environment artifact is missing or unsafe")
    environment_expected_bytes = _integer(
        environment_artifact.get("bytes"),
        "environment bytes",
        minimum=1,
    )
    environment_expected_sha = _string(
        environment_artifact.get("sha256"),
        "environment digest",
    )
    if len(environment_expected_sha) != 64 or any(
        byte not in "0123456789abcdef" for byte in environment_expected_sha
    ):
        raise ResultValidationError("environment digest is not sha256")
    environment_actual_sha, environment_actual_bytes = _sha256_file(environment_path)
    if (
        environment_actual_sha != environment_expected_sha
        or environment_actual_bytes != environment_expected_bytes
    ):
        raise ResultValidationError("environment digest or byte count mismatch")
    environment = _read_json(environment_path)
    if (
        environment.get("kind") != "slope_stability.environment"
        or environment.get("run_id") != run["run_id"]
        or environment.get("build_id") != run["build_id"]
        or environment.get("parallel_hdf5") is not True
        or environment.get("authority") != "native-observation"
    ):
        raise ResultValidationError(
            "environment identities or capabilities disagree with result"
        )
    logical, _ = _validate_hdf5(hdf5_path, envelope)
    identity = _object(envelope.get("identity"), "identity")
    if _require_digest(identity.get("result_id"), "identity.result_id") != result_id:
        raise ResultValidationError("result identity fields disagree")
    expected_logical = _require_digest(
        identity.get("logical_digest"), "result logical digest"
    )
    if (
        _require_digest(
            payload.get("logical_digest"),
            "payload logical digest",
        )
        != expected_logical
    ):
        raise ResultValidationError("payload and result logical identities disagree")
    if logical != expected_logical:
        raise ResultValidationError("result logical digest mismatch")
    stop_reason = _string(terminal.get("stop_reason"), "stop reason")
    recomputed_id = _sha256_bytes(
        (f"{run['run_id']}|{status}|{stop_reason}|{logical}").encode("utf-8")
    )
    if recomputed_id != result_id:
        raise ResultValidationError("result identity does not match content")
    verification = ResultVerification(
        payload_verified=True,
        logical_digests_verified=True,
        layout_id="result-v1",
        payload_bytes=actual_bytes,
        payload_sha256=actual_sha,
    )
    return Result(root, envelope, verification)


__all__ = [
    "open_result",
]
