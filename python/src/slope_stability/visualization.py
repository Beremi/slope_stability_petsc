# SPDX-License-Identifier: BSD-2-Clause
"""Derived-only high-order visualization bundles and qualification reports."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import numpy.typing as npt

from ._fe_layout import CANONICAL_BARYCENTRIC, CANONICAL_TO_VTK, SOURCE_DIGEST
from .results import IncompleteResult, Result


_ORDERS = (1, 2, 4, 8)
_VTK_CELL_TYPES = {
    (2, 1): 5,
    (2, 2): 22,
    (2, 4): 69,
    (2, 8): 69,
    (3, 1): 10,
    (3, 2): 24,
    (3, 4): 71,
    (3, 8): 71,
}
_REPORT_EVIDENCE_STATUSES = {"candidate", "reported", "provisional", "negative"}


class VisualizationError(ValueError):
    """A derived export or transformation request is invalid."""


@dataclass(frozen=True)
class SimplexVisualizationData:
    """Field values and geometry of one quantity state prepared for export.

    Point arrays are in the canonical simplex node order of the field and geometry
    orders; ``values`` has one row per field point.
    """

    dimension: int
    field_order: int
    geometry_order: int
    field_points: npt.NDArray[np.float64]
    geometry_points: npt.NDArray[np.float64]
    field_stable_ids: npt.NDArray[np.uint64]
    geometry_stable_ids: npt.NDArray[np.uint64]
    values: npt.NDArray[np.float64]
    quantity_id: str
    quantity_version: int
    component_ids: tuple[str, ...]
    unit: str
    state_id: int
    source_result_id: str


@dataclass(frozen=True)
class ProjectionResult:
    """Values projected between result locations, with the projection rule and an error
    estimate.
    """

    values: npt.NDArray[np.float64]
    rule: str
    unit: str
    error_estimate: float
    source_location: str
    target_location: str


@dataclass(frozen=True)
class VisualizationBundle:
    """Paths of one derived VTU/XDMF export and its manifest."""

    vtu: Path
    xdmf: Path
    heavy_data: Path
    manifest: Path


def _layout_key(dimension: int, order: int) -> tuple[int, int]:
    if dimension not in (2, 3) or order not in _ORDERS:
        raise VisualizationError(
            f"unsupported simplex layout: dimension={dimension}, order={order}"
        )
    key = (dimension, order)
    if key not in CANONICAL_BARYCENTRIC or key not in CANONICAL_TO_VTK:
        raise VisualizationError(f"generated simplex layout is unavailable: {key}")
    return key


def canonical_simplex_nodes(dimension: int, order: int) -> npt.NDArray[np.float64]:
    """Return canonical equispaced reference nodes as ``(r, s[, t])``."""
    key = _layout_key(dimension, order)
    barycentric = np.asarray(CANONICAL_BARYCENTRIC[key], dtype=np.float64)
    points = np.asarray(barycentric[:, 1:] / float(order), dtype=np.float64)
    points.flags.writeable = False
    return points


def canonical_to_vtk_permutation(dimension: int, order: int) -> tuple[int, ...]:
    """Map each canonical local node index to its VTK local-node index."""
    return CANONICAL_TO_VTK[_layout_key(dimension, order)]


def project_quadrature(
    values: npt.ArrayLike,
    weights: npt.ArrayLike,
    *,
    rule: str,
    unit: str,
) -> ProjectionResult:
    """Apply the named conservative cell-average projection."""
    if rule != "weighted_cell_average_v1":
        raise VisualizationError("a supported named projection rule is required")
    if not unit or any(ord(character) < 0x20 for character in unit):
        raise VisualizationError("projection unit is required")
    value_array = np.asarray(values, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    if (
        value_array.ndim != 2
        or weight_array.ndim != 1
        or value_array.shape[0] != weight_array.shape[0]
        or value_array.shape[0] == 0
        or not np.isfinite(value_array).all()
        or not np.isfinite(weight_array).all()
        or np.any(weight_array < 0.0)
    ):
        raise VisualizationError("projection arrays are invalid")
    weight_sum = float(np.sum(weight_array))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise VisualizationError("projection weights have no positive measure")
    normalized = weight_array / weight_sum
    projected = np.sum(value_array * normalized[:, None], axis=0)
    deviation = value_array - projected[None, :]
    component_error = np.sqrt(np.sum(normalized[:, None] * deviation**2, axis=0))
    output = np.asarray(projected, dtype=np.float64)
    output.flags.writeable = False
    return ProjectionResult(
        output,
        rule,
        unit,
        float(np.max(component_error)),
        "quadrature",
        "cell",
    )


def _require_text(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(ord(character) < 0x20 for character in value)
    ):
        raise VisualizationError(f"{label} must be a nonempty text value")
    return value


def _require_result_id(value: str) -> str:
    if (
        len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise VisualizationError("source result ID is not a canonical sha256 identity")
    return value


def _validated_data(data: SimplexVisualizationData) -> SimplexVisualizationData:
    field_key = _layout_key(data.dimension, data.field_order)
    geometry_key = _layout_key(data.dimension, data.geometry_order)
    field_count = len(CANONICAL_BARYCENTRIC[field_key])
    geometry_count = len(CANONICAL_BARYCENTRIC[geometry_key])
    field_points = np.asarray(data.field_points, dtype=np.float64)
    geometry_points = np.asarray(data.geometry_points, dtype=np.float64)
    field_ids = np.asarray(data.field_stable_ids, dtype=np.uint64)
    geometry_ids = np.asarray(data.geometry_stable_ids, dtype=np.uint64)
    values = np.asarray(data.values, dtype=np.float64)
    if (
        field_points.shape != (field_count, 3)
        or geometry_points.shape != (geometry_count, 3)
        or field_ids.shape != (field_count,)
        or geometry_ids.shape != (geometry_count,)
        or values.ndim != 2
        or values.shape[0] != field_count
        or values.shape[1] != len(data.component_ids)
    ):
        raise VisualizationError("visualization array shapes disagree with FE layouts")
    if (
        not np.isfinite(field_points).all()
        or not np.isfinite(geometry_points).all()
        or not np.isfinite(values).all()
    ):
        raise VisualizationError("visualization arrays contain nonfinite values")
    if (
        len(np.unique(field_ids)) != field_count
        or len(np.unique(geometry_ids)) != geometry_count
    ):
        raise VisualizationError("visualization stable IDs must be unique")
    if (
        data.quantity_version <= 0
        or data.state_id <= 0
        or not data.component_ids
        or len(set(data.component_ids)) != len(data.component_ids)
    ):
        raise VisualizationError("visualization identity metadata is invalid")
    _require_text(data.quantity_id, "quantity ID")
    _require_text(data.unit, "unit")
    for component in data.component_ids:
        _require_text(component, "component ID")
    _require_result_id(data.source_result_id)
    return SimplexVisualizationData(
        data.dimension,
        data.field_order,
        data.geometry_order,
        field_points,
        geometry_points,
        field_ids,
        geometry_ids,
        values,
        data.quantity_id,
        data.quantity_version,
        tuple(data.component_ids),
        data.unit,
        data.state_id,
        data.source_result_id,
    )


def _format_values(values: npt.NDArray[Any], kind: str) -> str:
    flat = np.asarray(values).reshape(-1)
    if kind == "float":
        return " ".join(format(float(value), ".17g") for value in flat)
    return " ".join(str(int(value)) for value in flat)


def _add_array(
    parent: ET.Element,
    *,
    name: str,
    vtk_type: str,
    values: npt.NDArray[Any],
    components: int = 1,
    value_kind: str = "integer",
) -> ET.Element:
    attributes = {
        "type": vtk_type,
        "Name": name,
        "format": "ascii",
    }
    if components != 1:
        attributes["NumberOfComponents"] = str(components)
    element = ET.SubElement(parent, "DataArray", attributes)
    element.text = _format_values(values, value_kind)
    return element


def _metadata_bytes(value: str | Sequence[str]) -> npt.NDArray[np.uint8]:
    text = (
        json.dumps(tuple(value), ensure_ascii=True, separators=(",", ":"))
        if not isinstance(value, str)
        else value
    )
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8)


def _vtu_bytes(data: SimplexVisualizationData) -> bytes:
    permutation = np.asarray(
        canonical_to_vtk_permutation(data.dimension, data.field_order),
        dtype=np.intp,
    )
    count = len(permutation)
    vtk_points = np.empty_like(data.field_points)
    vtk_values = np.empty_like(data.values)
    vtk_ids = np.empty_like(data.field_stable_ids)
    vtk_points[permutation] = data.field_points
    vtk_values[permutation] = data.values
    vtk_ids[permutation] = data.field_stable_ids

    root = ET.Element(
        "VTKFile",
        {
            "type": "UnstructuredGrid",
            "version": "1.0",
            "byte_order": "LittleEndian",
            "header_type": "UInt64",
        },
    )
    grid = ET.SubElement(root, "UnstructuredGrid")
    field_data = ET.SubElement(grid, "FieldData")
    piece = ET.SubElement(
        grid,
        "Piece",
        {"NumberOfPoints": str(count), "NumberOfCells": "1"},
    )
    points = ET.SubElement(piece, "Points")
    _add_array(
        points,
        name="Points",
        vtk_type="Float64",
        values=vtk_points,
        components=3,
        value_kind="float",
    )
    cells = ET.SubElement(piece, "Cells")
    _add_array(
        cells,
        name="connectivity",
        vtk_type="Int64",
        values=np.arange(count, dtype=np.int64),
    )
    _add_array(
        cells,
        name="offsets",
        vtk_type="Int64",
        values=np.asarray([count], dtype=np.int64),
    )
    _add_array(
        cells,
        name="types",
        vtk_type="UInt8",
        values=np.asarray(
            [_VTK_CELL_TYPES[(data.dimension, data.field_order)]], dtype=np.uint8
        ),
    )
    point_data = ET.SubElement(piece, "PointData")
    _add_array(
        point_data,
        name="slope.values",
        vtk_type="Float64",
        values=vtk_values,
        components=len(data.component_ids),
        value_kind="float",
    )
    _add_array(
        point_data,
        name="slope.field_stable_ids",
        vtk_type="UInt64",
        values=vtk_ids,
    )
    for name, value in (
        ("slope.dimension", data.dimension),
        ("slope.field_order", data.field_order),
        ("slope.geometry_order", data.geometry_order),
        ("slope.quantity_version", data.quantity_version),
        ("slope.state_id", data.state_id),
    ):
        _add_array(
            field_data,
            name=name,
            vtk_type="Int64",
            values=np.asarray([value], dtype=np.int64),
        )
    for name, value in (
        ("slope.quantity_id", data.quantity_id),
        ("slope.component_ids", data.component_ids),
        ("slope.unit", data.unit),
        ("slope.source_result_id", data.source_result_id),
        ("slope.fe_source_digest", SOURCE_DIGEST),
    ):
        _add_array(
            field_data,
            name=name,
            vtk_type="UInt8",
            values=_metadata_bytes(value),
        )
    _add_array(
        field_data,
        name="slope.geometry_points",
        vtk_type="Float64",
        values=data.geometry_points,
        components=3,
        value_kind="float",
    )
    _add_array(
        field_data,
        name="slope.geometry_stable_ids",
        vtk_type="UInt64",
        values=data.geometry_stable_ids,
    )
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_heavy_data(path: Path, data: SimplexVisualizationData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        with h5py.File(temporary, "w", libver="latest") as handle:
            handle.attrs["authority"] = "derived"
            handle.attrs["source_result_id"] = data.source_result_id
            handle.attrs["fe_source_digest"] = SOURCE_DIGEST
            handle.create_dataset("FieldPoints", data=data.field_points, dtype="<f8")
            handle.create_dataset(
                "GeometryPoints", data=data.geometry_points, dtype="<f8"
            )
            handle.create_dataset("Values", data=data.values, dtype="<f8")
            handle.create_dataset(
                "FieldStableIds", data=data.field_stable_ids, dtype="<u8"
            )
            handle.create_dataset(
                "GeometryStableIds", data=data.geometry_stable_ids, dtype="<u8"
            )
            handle.create_dataset(
                "Topology",
                data=np.arange(len(data.field_points), dtype=np.int64),
                dtype="<i8",
            )
            handle.flush()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _information(parent: ET.Element, name: str, value: object) -> None:
    ET.SubElement(parent, "Information", {"Name": name, "Value": str(value)})


def _xdmf_bytes(path: Path, data: SimplexVisualizationData) -> bytes:
    heavy_name = path.with_suffix(".h5").name
    count = len(data.field_points)
    components = len(data.component_ids)
    root = ET.Element("Xdmf", {"Version": "3.0"})
    domain = ET.SubElement(root, "Domain")
    grid = ET.SubElement(
        domain, "Grid", {"Name": "SlopeFieldNodes", "GridType": "Uniform"}
    )
    topology = ET.SubElement(
        grid,
        "Topology",
        {"TopologyType": "Polyvertex", "NumberOfElements": str(count)},
    )
    item = ET.SubElement(
        topology,
        "DataItem",
        {
            "Dimensions": str(count),
            "NumberType": "Int",
            "Precision": "8",
            "Format": "HDF",
        },
    )
    item.text = f"{heavy_name}:/Topology"
    geometry = ET.SubElement(grid, "Geometry", {"GeometryType": "XYZ"})
    item = ET.SubElement(
        geometry,
        "DataItem",
        {
            "Dimensions": f"{count} 3",
            "NumberType": "Float",
            "Precision": "8",
            "Format": "HDF",
        },
    )
    item.text = f"{heavy_name}:/FieldPoints"
    attribute = ET.SubElement(
        grid,
        "Attribute",
        {
            "Name": data.quantity_id,
            "AttributeType": "Scalar" if components == 1 else "Vector",
            "Center": "Node",
        },
    )
    item = ET.SubElement(
        attribute,
        "DataItem",
        {
            "Dimensions": f"{count} {components}",
            "NumberType": "Float",
            "Precision": "8",
            "Format": "HDF",
        },
    )
    item.text = f"{heavy_name}:/Values"
    metadata: Mapping[str, object] = {
        "slope.dimension": data.dimension,
        "slope.field_order": data.field_order,
        "slope.geometry_order": data.geometry_order,
        "slope.quantity_id": data.quantity_id,
        "slope.quantity_version": data.quantity_version,
        "slope.component_ids": json.dumps(
            data.component_ids, ensure_ascii=True, separators=(",", ":")
        ),
        "slope.unit": data.unit,
        "slope.state_id": data.state_id,
        "slope.source_result_id": data.source_result_id,
        "slope.fe_source_digest": SOURCE_DIGEST,
        "slope.field_stable_ids": f"{heavy_name}:/FieldStableIds",
        "slope.geometry_points": f"{heavy_name}:/GeometryPoints",
        "slope.geometry_stable_ids": f"{heavy_name}:/GeometryStableIds",
        "slope.high_order_topology": (
            "vtk_lagrange" if data.field_order >= 4 else "vtk_standard"
        ),
        "slope.linearized": "false",
    }
    for name, value in metadata.items():
        _information(grid, name, value)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"


def _sha256_record(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def export_visualization(
    data: SimplexVisualizationData, directory: str | os.PathLike[str]
) -> VisualizationBundle:
    """Write a derived VTU/XDMF bundle without changing authoritative data."""
    checked = _validated_data(data)
    root = Path(directory).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise VisualizationError("visualization destination is not a directory")
    root.mkdir(parents=True, exist_ok=True)
    vtu = root / "visualization.vtu"
    xdmf = root / "visualization.xdmf"
    heavy = xdmf.with_suffix(".h5")
    manifest = root / "visualization.manifest.json"
    _write_bytes_atomic(vtu, _vtu_bytes(checked))
    _write_heavy_data(heavy, checked)
    _write_bytes_atomic(xdmf, _xdmf_bytes(xdmf, checked))
    transform = {
        "dimension": checked.dimension,
        "field_order": checked.field_order,
        "geometry_order": checked.geometry_order,
        "topology": ("vtk_lagrange" if checked.field_order >= 4 else "vtk_standard"),
        "linearized": False,
        "coordinate_representation": (
            "field_nodes_with_canonical_geometry_control_points"
        ),
        "canonical_to_vtk": list(
            canonical_to_vtk_permutation(checked.dimension, checked.field_order)
        ),
        "xdmf_view": "polyvertex_with_high_order_control_metadata",
    }
    record = {
        "schema_version": 1,
        "authority": "derived",
        "source_result_id": checked.source_result_id,
        "quantity": {
            "id": checked.quantity_id,
            "version": checked.quantity_version,
            "components": list(checked.component_ids),
            "unit": checked.unit,
            "state_id": checked.state_id,
        },
        "fe_source_digest": SOURCE_DIGEST,
        "transform": transform,
        "vtk_cell_type": _VTK_CELL_TYPES[(checked.dimension, checked.field_order)],
        "files": {
            vtu.name: _sha256_record(vtu),
            xdmf.name: _sha256_record(xdmf),
            heavy.name: _sha256_record(heavy),
        },
    }
    payload = (
        json.dumps(
            record,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _write_bytes_atomic(manifest, payload)
    return VisualizationBundle(vtu, xdmf, heavy, manifest)


def _named_array(parent: ET.Element, name: str) -> ET.Element:
    matches = [
        element
        for element in parent.findall("DataArray")
        if element.attrib.get("Name") == name
    ]
    if len(matches) != 1:
        raise VisualizationError(f"VTU array {name!r} is missing or duplicated")
    return matches[0]


def _array_values(element: ET.Element, dtype: npt.DTypeLike) -> npt.NDArray[Any]:
    if element.attrib.get("format") != "ascii":
        raise VisualizationError("only strict ASCII VTU arrays are accepted")
    text = element.text or ""
    try:
        return np.fromstring(text, sep=" ", dtype=dtype)
    except ValueError as error:
        raise VisualizationError("VTU array cannot be parsed") from error


def _field_integer(field_data: ET.Element, name: str) -> int:
    values = _array_values(_named_array(field_data, name), np.int64)
    if values.shape != (1,):
        raise VisualizationError(f"VTU scalar metadata {name!r} is invalid")
    return int(values[0])


def _field_text(field_data: ET.Element, name: str) -> str:
    values = _array_values(_named_array(field_data, name), np.uint8)
    try:
        return values.tobytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise VisualizationError(f"VTU text metadata {name!r} is invalid") from error


def read_vtu(path: str | os.PathLike[str]) -> SimplexVisualizationData:
    """Strictly read a bundle VTU and restore canonical node order."""
    source = Path(path).expanduser().resolve()
    try:
        root = ET.parse(source).getroot()
    except (OSError, ET.ParseError) as error:
        raise VisualizationError(f"cannot parse VTU: {error}") from error
    if root.tag != "VTKFile" or root.attrib.get("type") != "UnstructuredGrid":
        raise VisualizationError("VTU is not an unstructured grid")
    pieces = root.findall("./UnstructuredGrid/Piece")
    if len(pieces) != 1:
        raise VisualizationError("VTU must contain exactly one piece")
    piece = pieces[0]
    field_data = root.find("./UnstructuredGrid/FieldData")
    points_parent = piece.find("Points")
    point_data = piece.find("PointData")
    cells = piece.find("Cells")
    if any(item is None for item in (field_data, points_parent, point_data, cells)):
        raise VisualizationError("VTU lacks a required section")
    assert field_data is not None
    assert points_parent is not None
    assert point_data is not None
    assert cells is not None
    dimension = _field_integer(field_data, "slope.dimension")
    field_order = _field_integer(field_data, "slope.field_order")
    geometry_order = _field_integer(field_data, "slope.geometry_order")
    key = _layout_key(dimension, field_order)
    geometry_key = _layout_key(dimension, geometry_order)
    count = len(CANONICAL_BARYCENTRIC[key])
    geometry_count = len(CANONICAL_BARYCENTRIC[geometry_key])
    if (
        piece.attrib.get("NumberOfPoints") != str(count)
        or piece.attrib.get("NumberOfCells") != "1"
    ):
        raise VisualizationError("VTU piece counts disagree with FE identity")
    point_element = _named_array(points_parent, "Points")
    external_points = _array_values(point_element, np.float64)
    if external_points.size != count * 3:
        raise VisualizationError("VTU point shape is invalid")
    external_points = external_points.reshape(count, 3)
    value_element = _named_array(point_data, "slope.values")
    component_count = int(value_element.attrib.get("NumberOfComponents", "1"))
    external_values = _array_values(value_element, np.float64)
    if external_values.size != count * component_count:
        raise VisualizationError("VTU value shape is invalid")
    external_values = external_values.reshape(count, component_count)
    external_ids = _array_values(
        _named_array(point_data, "slope.field_stable_ids"), np.uint64
    )
    connectivity = _array_values(_named_array(cells, "connectivity"), np.int64)
    offsets = _array_values(_named_array(cells, "offsets"), np.int64)
    cell_types = _array_values(_named_array(cells, "types"), np.uint8)
    if (
        external_ids.shape != (count,)
        or not np.array_equal(connectivity, np.arange(count, dtype=np.int64))
        or offsets.tolist() != [count]
        or cell_types.tolist() != [_VTK_CELL_TYPES[key]]
    ):
        raise VisualizationError("VTU topology or stable identity is invalid")
    permutation = np.asarray(CANONICAL_TO_VTK[key], dtype=np.intp)
    field_points = np.asarray(external_points[permutation], dtype=np.float64)
    values = np.asarray(external_values[permutation], dtype=np.float64)
    field_ids = np.asarray(external_ids[permutation], dtype=np.uint64)
    geometry_points = _array_values(
        _named_array(field_data, "slope.geometry_points"), np.float64
    )
    geometry_ids = _array_values(
        _named_array(field_data, "slope.geometry_stable_ids"), np.uint64
    )
    if geometry_points.size != geometry_count * 3 or geometry_ids.shape != (
        geometry_count,
    ):
        raise VisualizationError("VTU geometry control arrays are invalid")
    component_ids_raw = _field_text(field_data, "slope.component_ids")
    try:
        component_ids_value = json.loads(component_ids_raw)
    except json.JSONDecodeError as error:
        raise VisualizationError("VTU component metadata is invalid") from error
    if not isinstance(component_ids_value, list):
        raise VisualizationError("VTU component metadata is not an array")
    result = SimplexVisualizationData(
        dimension,
        field_order,
        geometry_order,
        field_points,
        np.asarray(geometry_points.reshape(geometry_count, 3), dtype=np.float64),
        field_ids,
        np.asarray(geometry_ids, dtype=np.uint64),
        values,
        _field_text(field_data, "slope.quantity_id"),
        _field_integer(field_data, "slope.quantity_version"),
        tuple(str(value) for value in component_ids_value),
        _field_text(field_data, "slope.unit"),
        _field_integer(field_data, "slope.state_id"),
        _field_text(field_data, "slope.source_result_id"),
    )
    if _field_text(field_data, "slope.fe_source_digest") != SOURCE_DIGEST:
        raise VisualizationError("VTU finite-element table authority differs")
    return _validated_data(result)


def _information_index(grid: ET.Element) -> dict[str, str]:
    result: dict[str, str] = {}
    for element in grid.findall("Information"):
        name = element.attrib.get("Name")
        value = element.attrib.get("Value")
        if not name or value is None or name in result:
            raise VisualizationError("XDMF information metadata is invalid")
        result[name] = value
    return result


def _hdf_reference(reference: str, parent: Path) -> tuple[Path, str]:
    if ":/" not in reference:
        raise VisualizationError("XDMF HDF reference is invalid")
    file_name, dataset = reference.split(":", 1)
    relative = Path(file_name)
    if relative.is_absolute() or ".." in relative.parts or relative.name != file_name:
        raise VisualizationError("XDMF HDF reference escapes its bundle")
    heavy = (parent / relative).resolve()
    if heavy.parent != parent.resolve():
        raise VisualizationError("XDMF HDF reference escapes its bundle")
    return heavy, dataset


def read_xdmf(path: str | os.PathLike[str]) -> SimplexVisualizationData:
    """Read the HDF5-backed XDMF derived view."""
    source = Path(path).expanduser().resolve()
    try:
        root = ET.parse(source).getroot()
    except (OSError, ET.ParseError) as error:
        raise VisualizationError(f"cannot parse XDMF: {error}") from error
    grids = root.findall("./Domain/Grid")
    if root.tag != "Xdmf" or root.attrib.get("Version") != "3.0" or len(grids) != 1:
        raise VisualizationError("XDMF must contain one version-3 grid")
    grid = grids[0]
    information = _information_index(grid)
    required = {
        "slope.dimension",
        "slope.field_order",
        "slope.geometry_order",
        "slope.quantity_id",
        "slope.quantity_version",
        "slope.component_ids",
        "slope.unit",
        "slope.state_id",
        "slope.source_result_id",
        "slope.fe_source_digest",
        "slope.field_stable_ids",
        "slope.geometry_points",
        "slope.geometry_stable_ids",
    }
    if not required.issubset(information):
        raise VisualizationError("XDMF lacks required high-order metadata")
    dimension = int(information["slope.dimension"])
    field_order = int(information["slope.field_order"])
    geometry_order = int(information["slope.geometry_order"])
    field_key = _layout_key(dimension, field_order)
    geometry_key = _layout_key(dimension, geometry_order)
    count = len(CANONICAL_BARYCENTRIC[field_key])
    geometry_count = len(CANONICAL_BARYCENTRIC[geometry_key])
    topology = grid.find("Topology")
    geometry = grid.find("Geometry")
    attribute = grid.find("Attribute")
    if topology is None or geometry is None or attribute is None:
        raise VisualizationError("XDMF grid sections are incomplete")
    if topology.attrib.get("TopologyType") != "Polyvertex" or topology.attrib.get(
        "NumberOfElements"
    ) != str(count):
        raise VisualizationError("XDMF derived topology is invalid")
    value_item = attribute.find("DataItem")
    point_item = geometry.find("DataItem")
    if value_item is None or point_item is None:
        raise VisualizationError("XDMF data references are incomplete")
    references = {
        "field_points": (point_item.text or "").strip(),
        "values": (value_item.text or "").strip(),
        "field_ids": information["slope.field_stable_ids"],
        "geometry_points": information["slope.geometry_points"],
        "geometry_ids": information["slope.geometry_stable_ids"],
    }
    resolved = {
        name: _hdf_reference(reference, source.parent)
        for name, reference in references.items()
    }
    heavy_paths = {record[0] for record in resolved.values()}
    if len(heavy_paths) != 1:
        raise VisualizationError("XDMF references more than one heavy-data file")
    heavy = next(iter(heavy_paths))
    try:
        with h5py.File(heavy, "r") as handle:
            arrays = {
                name: np.asarray(handle[dataset][...])
                for name, (_, dataset) in resolved.items()
            }
            if (
                str(handle.attrs.get("authority", "")) != "derived"
                or str(handle.attrs.get("source_result_id", ""))
                != information["slope.source_result_id"]
                or str(handle.attrs.get("fe_source_digest", "")) != SOURCE_DIGEST
            ):
                raise VisualizationError("XDMF heavy-data authority is invalid")
    except (OSError, KeyError) as error:
        raise VisualizationError(f"cannot read XDMF heavy data: {error}") from error
    try:
        component_ids_value = json.loads(information["slope.component_ids"])
    except json.JSONDecodeError as error:
        raise VisualizationError("XDMF component metadata is invalid") from error
    if not isinstance(component_ids_value, list):
        raise VisualizationError("XDMF component metadata is not an array")
    values = np.asarray(arrays["values"], dtype=np.float64)
    result = SimplexVisualizationData(
        dimension,
        field_order,
        geometry_order,
        np.asarray(arrays["field_points"], dtype=np.float64),
        np.asarray(arrays["geometry_points"], dtype=np.float64),
        np.asarray(arrays["field_ids"], dtype=np.uint64),
        np.asarray(arrays["geometry_ids"], dtype=np.uint64),
        values,
        information["slope.quantity_id"],
        int(information["slope.quantity_version"]),
        tuple(str(value) for value in component_ids_value),
        information["slope.unit"],
        int(information["slope.state_id"]),
        information["slope.source_result_id"],
    )
    if result.field_points.shape != (count, 3) or result.geometry_points.shape != (
        geometry_count,
        3,
    ):
        raise VisualizationError("XDMF FE array shapes disagree with metadata")
    if information["slope.fe_source_digest"] != SOURCE_DIGEST:
        raise VisualizationError("XDMF finite-element table authority differs")
    return _validated_data(result)


def _report_target(
    result: Result | IncompleteResult, destination: str | os.PathLike[str]
) -> Path:
    target = Path(destination).expanduser().resolve()
    if target.suffix.lower() != ".html":
        raise VisualizationError("report destination must end in .html")
    source = result.path.resolve()
    try:
        relative = target.relative_to(source)
    except ValueError:
        return target
    if not relative.parts or relative.parts[0] != "derived":
        raise VisualizationError(
            "reports inside a result must be placed under its derived directory"
        )
    return target


def write_report(
    result: Result | IncompleteResult,
    destination: str | os.PathLike[str],
    *,
    evidence_status: str = "candidate",
) -> Path:
    """Write a derived HTML summary that cannot promote scientific evidence."""
    if evidence_status not in _REPORT_EVIDENCE_STATUSES:
        raise VisualizationError(
            "report evidence status must remain candidate/reported/provisional/negative"
        )
    target = _report_target(result, destination)
    status = result.status
    if isinstance(result, IncompleteResult):
        analysis = "unavailable"
        outcome = "unavailable"
        result_id = "unavailable"
        build_id = "unavailable"
        quantity_rows = '<tr><td colspan="2">Quantities unavailable</td></tr>'
        availability = (
            '<p class="unavailable">Quantities unavailable: the result is '
            "incomplete.</p>"
        )
        source_result_id: str | None = None
    else:
        analysis = result.analysis.method
        outcome = result.analysis.outcome
        result_id = result.result_id
        build_id = result.build_id
        summary = result.envelope["summary"]
        raw_quantities = (
            summary.get("quantities", []) if isinstance(summary, Mapping) else []
        )
        quantity_rows = "".join(
            "<tr><td>"
            + escape(str(quantity.get("id", "unknown")))
            + "</td><td>"
            + escape(str(quantity.get("unit", "unknown")))
            + "</td></tr>"
            for quantity in raw_quantities
            if isinstance(quantity, Mapping)
        )
        if result.status == "succeeded":
            availability = "<p>Successful solution quantities are available.</p>"
        else:
            availability = (
                '<p class="unavailable">Last accepted state remains available; '
                "terminal solution unavailable.</p>"
            )
        source_result_id = result.result_id
    html = (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        "<title>Slope-stability result report</title></head><body>"
        "<h1>Slope-stability result report</h1>"
        f"<p>Analysis: {escape(analysis)}</p>"
        f"<p>Outcome: {escape(outcome)}</p>"
        f"<p>Status: {escape(status)}</p>"
        f"<p>Result ID: {escape(result_id)}</p>"
        f"<p>Build ID: {escape(build_id)}</p>"
        f"<p>Evidence status: {escape(evidence_status)}</p>"
        f"{availability}"
        "<h2>Quantities and units</h2><table><thead><tr>"
        "<th>Quantity</th><th>Unit</th></tr></thead><tbody>"
        f"{quantity_rows}</tbody></table>"
        "<p>Authority: derived; this report does not promote evidence.</p>"
        "</body></html>\n"
    ).encode("utf-8")
    _write_bytes_atomic(target, html)
    manifest = {
        "schema_version": 1,
        "authority": "derived",
        "source_result_id": source_result_id,
        "status": status,
        "evidence_status": evidence_status,
        "report": {
            "path": target.name,
            "bytes": len(html),
            "sha256": hashlib.sha256(html).hexdigest(),
        },
    }
    manifest_bytes = (
        json.dumps(
            manifest,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _write_bytes_atomic(target.with_suffix(".manifest.json"), manifest_bytes)
    return target


__all__ = [
    "ProjectionResult",
    "SimplexVisualizationData",
    "VisualizationBundle",
    "VisualizationError",
    "canonical_simplex_nodes",
    "canonical_to_vtk_permutation",
    "export_visualization",
    "project_quadrature",
    "read_vtu",
    "read_xdmf",
    "write_report",
]
