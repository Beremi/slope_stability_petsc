# SPDX-License-Identifier: BSD-2-Clause
"""Deterministic HDF5 problem packages and their atomic publication."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from ..authoring import Asset, Case
from slope_protocol import canonicalize, content_id
from .._io import file_sha256, fsync_path
from ._common import CompileError, _hex_id
from .canonical import CanonicalMesh
from .model import CompiledProblemModel, compile_problem_model


@dataclass(frozen=True)
class CompiledPackage:
    """Location and identities of a published problem package."""

    problem_id: str
    path: Path
    problem_json: Path
    compile_report: Path
    hdf5_logical_digest: str


_DATASET_ATTRIBUTES = {
    "layout_id",
    "logical_shape",
    "value_type",
    "digest_method",
    "digest",
}


# Rows of a dataset compared and hashed together during verification.
_VERIFY_CHUNK_ELEMENTS = 1 << 20


def _digest_prefix(dtype: np.dtype[Any], shape: tuple[int, ...]) -> Any:
    text = f"{dtype.str}:{','.join(str(item) for item in shape)}:"
    return sha256(text.encode("ascii"))


def _dataset_digest(values: NDArray[Any]) -> str:
    """Digest of ``dtype:shape:`` followed by the C-order bytes, without a copy."""
    array = np.ascontiguousarray(values)
    digest = _digest_prefix(array.dtype, array.shape)
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _stream_dataset(
    dataset: h5py.Dataset, expected: NDArray[Any] | None
) -> tuple[str, bool]:
    """Hash a stored dataset in row chunks and compare it with ``expected``.

    Returns the digest of the stored values and whether they equal
    ``expected`` (always true without expectation).  Scratch is bounded by
    ``_VERIFY_CHUNK_ELEMENTS`` elements.
    """
    shape = tuple(int(item) for item in dataset.shape)
    digest = _digest_prefix(dataset.dtype, shape)
    equal = expected is None or (
        tuple(expected.shape) == shape
        and expected.dtype.kind == dataset.dtype.kind
        and expected.dtype.itemsize == dataset.dtype.itemsize
    )
    if not shape:
        value = np.asarray(dataset[()])
        digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))
        return digest.hexdigest(), equal and (
            expected is None or bool(np.array_equal(value, expected))
        )
    row_elements = max(1, int(np.prod(shape[1:], dtype=np.int64)))
    rows = max(1, _VERIFY_CHUNK_ELEMENTS // row_elements)
    for start in range(0, shape[0], rows):
        block = np.ascontiguousarray(dataset[start : start + rows])
        digest.update(memoryview(block).cast("B"))
        if equal and expected is not None:
            equal = bool(np.array_equal(block, expected[start : start + rows]))
    return digest.hexdigest(), equal


def _create_dataset(
    group: h5py.Group,
    name: str,
    values: NDArray[Any],
    *,
    value_type: str,
) -> h5py.Dataset:
    array = np.ascontiguousarray(values)
    dataset = group.create_dataset(
        name,
        data=array,
        dtype=array.dtype,
        track_times=False,
    )
    dataset.attrs["layout_id"] = "compiled-problem-v1"
    dataset.attrs["logical_shape"] = json.dumps(
        list(array.shape), separators=(",", ":")
    )
    dataset.attrs["value_type"] = value_type
    dataset.attrs["digest_method"] = "sha256-dtype-shape-c-order"
    dataset.attrs["digest"] = _dataset_digest(array)
    return dataset


def _write_hdf5(path: Path, compiled: CompiledProblemModel) -> None:
    mesh = compiled.mesh
    with h5py.File(path, "w", track_order=True) as handle:
        handle.attrs["layout_id"] = "compiled-problem-v1"
        handle.attrs["schema_major"] = 1
        handle.attrs["schema_minor"] = 0
        handle.attrs["dimension"] = mesh.dimension
        handle.attrs["cell_type"] = mesh.cell_type
        handle.attrs["coordinate_order"] = mesh.coordinate_order

        mesh_group = handle.create_group("mesh", track_order=True)
        topology = mesh_group.create_group("topology", track_order=True)
        entities = mesh_group.create_group("entities", track_order=True)
        coordinates = mesh_group.create_group("coordinates", track_order=True)
        supports = mesh_group.create_group("supports", track_order=True)
        _create_dataset(
            topology,
            "vertex_ids",
            mesh.topology_vertex_ids,
            value_type="uint64",
        )
        _create_dataset(
            topology,
            "vertex_coordinates",
            mesh.topology_vertex_coordinates,
            value_type="float64",
        )
        _create_dataset(
            topology,
            "cell_ids",
            mesh.topology_cell_ids,
            value_type="uint64",
        )
        _create_dataset(topology, "cells", mesh.topology_cells, value_type="uint64")
        _create_dataset(
            coordinates,
            "values",
            mesh.coordinate_nodes,
            value_type="float64",
        )
        _create_dataset(
            coordinates,
            "connectivity",
            mesh.coordinate_connectivity,
            value_type="uint64",
        )
        _create_dataset(
            coordinates,
            "barycentric_functionals",
            mesh.coordinate_barycentric_functionals,
            value_type="uint64",
        )
        _create_dataset(
            coordinates,
            "functional_ids",
            mesh.coordinate_functional_ids,
            value_type="uint64",
        )
        for dimension, values in sorted(mesh.stable_entity_ids.items()):
            _create_dataset(
                entities,
                f"dimension_{dimension}",
                values,
                value_type="uint64",
            )
            _create_dataset(
                entities,
                f"dimension_{dimension}_vertex_ids",
                mesh.stable_entity_vertex_ids[dimension],
                value_type="uint64",
            )
        for name, support in sorted(mesh.supports.items()):
            dataset = _create_dataset(
                supports,
                name,
                np.asarray(support.entity_ids, dtype="<u8"),
                value_type="uint64",
            )
            dataset.attrs["dimension"] = support.dimension
            dataset.attrs["kind"] = support.kind

        spaces = handle.create_group("spaces", track_order=True)
        for name, description in sorted(compiled.spaces.items()):
            group = spaces.create_group(name, track_order=True)
            for key, value in sorted(description.items()):
                group.attrs[key] = value

        materials = handle.create_group("materials", track_order=True)
        for material in sorted(compiled.materials, key=lambda item: item["name"]):
            group = materials.create_group(material["name"], track_order=True)
            group.attrs["model"] = material["model"]
            group.attrs["support"] = material["support"]
            _create_dataset(
                group,
                "entity_ids",
                np.asarray(
                    [int(item, 16) for item in material["entity_ids"]],
                    dtype="<u8",
                ),
                value_type="uint64",
            )

        fields = handle.create_group("fields", track_order=True)
        initial = fields.create_group("initial", track_order=True)
        initial.attrs["kind"] = "zero"

        provenance = handle.create_group("provenance", track_order=True)
        provenance.attrs["asset_id"] = compiled.asset_id
        provenance.attrs["case_id"] = compiled.case_id
        import_maps = provenance.create_group("import_maps", track_order=True)
        sorted_map = sorted(mesh.import_maps.source_to_canonical_nodes.items())
        _create_dataset(
            import_maps,
            "source_node_ids",
            np.asarray([item[0] for item in sorted_map], dtype="<u8"),
            value_type="uint64",
        )
        _create_dataset(
            import_maps,
            "canonical_node_indices",
            np.asarray([item[1] for item in sorted_map], dtype="<u8"),
            value_type="uint64",
        )
        handle.flush()


def _verify_hdf5(path: Path, compiled: CompiledProblemModel) -> None:
    with h5py.File(path, "r") as handle:
        required_groups = {
            "mesh/topology",
            "mesh/entities",
            "mesh/coordinates",
            "mesh/supports",
            "spaces",
            "materials",
            "fields/initial",
            "provenance/import_maps",
        }
        if any(group not in handle for group in required_groups):
            raise CompileError("payload_verification_failed", "required HDF5 groups")
        expectations = {
            "mesh/topology/vertex_ids": compiled.mesh.topology_vertex_ids,
            "mesh/topology/vertex_coordinates": (
                compiled.mesh.topology_vertex_coordinates
            ),
            "mesh/topology/cell_ids": compiled.mesh.topology_cell_ids,
            "mesh/topology/cells": compiled.mesh.topology_cells,
            "mesh/coordinates/values": compiled.mesh.coordinate_nodes,
            "mesh/coordinates/connectivity": (compiled.mesh.coordinate_connectivity),
            "mesh/coordinates/barycentric_functionals": (
                compiled.mesh.coordinate_barycentric_functionals
            ),
            "mesh/coordinates/functional_ids": (
                compiled.mesh.coordinate_functional_ids
            ),
        }
        for name, dataset_expected in compiled.mesh.stable_entity_ids.items():
            expectations[f"mesh/entities/dimension_{name}"] = dataset_expected
        for name, dataset_expected in compiled.mesh.stable_entity_vertex_ids.items():
            expectations[f"mesh/entities/dimension_{name}_vertex_ids"] = (
                dataset_expected
            )
        for name, support in compiled.mesh.supports.items():
            expectations[f"mesh/supports/{name}"] = np.asarray(
                support.entity_ids, dtype="<u8"
            )
        if set(handle["mesh/entities"]) != {
            name
            for item in compiled.mesh.stable_entity_ids
            for name in (
                f"dimension_{item}",
                f"dimension_{item}_vertex_ids",
            )
        }:
            raise CompileError("payload_verification_failed", "stable_entity_ids")
        if set(handle["mesh/supports"]) != set(compiled.mesh.supports):
            raise CompileError("payload_verification_failed", "supports")
        for name, support in compiled.mesh.supports.items():
            dataset = handle["mesh/supports"][name]
            if (
                int(dataset.attrs["dimension"]) != support.dimension
                or str(dataset.attrs["kind"]) != support.kind
            ):
                raise CompileError("payload_verification_failed", f"supports/{name}")
        invalid_attributes: list[str] = []
        mismatched: list[str] = []

        def inspect(name: str, item: h5py.Group | h5py.Dataset) -> None:
            # One bounded pass per dataset: stored digest and expected values.
            if isinstance(item, h5py.Dataset):
                if not _DATASET_ATTRIBUTES.issubset(item.attrs):
                    invalid_attributes.append(name)
                    return
                digest, equal = _stream_dataset(item, expectations.get(name))
                if not equal:
                    mismatched.append(name)
                if (
                    str(item.attrs["layout_id"]) != "compiled-problem-v1"
                    or str(item.attrs["logical_shape"])
                    != json.dumps(list(item.shape), separators=(",", ":"))
                    or str(item.attrs["digest_method"]) != "sha256-dtype-shape-c-order"
                    or str(item.attrs["digest"]) != digest
                ):
                    invalid_attributes.append(name)

        handle.visititems(inspect)
        missing = sorted(name for name in expectations if name not in handle)
        if mismatched or missing:
            raise CompileError("payload_verification_failed", (mismatched + missing)[0])
        if invalid_attributes:
            raise CompileError(
                "payload_verification_failed",
                f"dataset attributes: {invalid_attributes}",
            )


def compile_problem(
    *,
    asset: Asset,
    case: Case,
    mesh: CanonicalMesh,
    destination: Path,
    failure_at: str | None = None,
) -> CompiledPackage:
    """Write, verify, fsync, and atomically publish one problem package."""
    compiled = compile_problem_model(asset=asset, case=case, mesh=mesh)
    logical_payload = compiled.logical_payload()
    hdf5_logical_digest = content_id(
        {
            "mesh": mesh._logical_payload(),
            "materials": list(compiled.materials),
            "boundary_rules": list(compiled.boundary_rules),
        }
    )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    # Build and verify the payload once, staged on the destination filesystem;
    # its digest names the package and the staging directory is published.
    temporary = Path(tempfile.mkdtemp(prefix=".slope-problem-", dir=str(destination)))
    try:
        payload_path = temporary / "problem.h5"
        _write_hdf5(payload_path, compiled)
        if failure_at == "after_hdf5":
            raise CompileError("injected_write_failure", failure_at)
        _verify_hdf5(payload_path, compiled)
        if failure_at == "after_payload_verification":
            raise CompileError("injected_write_failure", failure_at)
        hdf5_sha256 = file_sha256(payload_path)
        hdf5_byte_count = payload_path.stat().st_size
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    problem_without_id = {
        "kind": "slope_stability.compiled_problem",
        "schema_version": {"major": 1, "minor": 0, "patch": 0},
        "created_by": {
            "tool": "slope-stability-compiler",
            "tool_version": "0.1.0",
            "source_id": case.id,
        },
        "provenance": {
            "asset_id": asset.id,
            "case_id": case.id,
            "importer_id": mesh.provenance.importer_id,
            "source_sha256": mesh.provenance.source_sha256,
        },
        "extensions": {},
        "identity": {
            "name": f"{case.identity.name}.compiled",
            "title": f"Compiled {case.identity.title}",
        },
        "units": {
            "internal": {
                "length": "m",
                "time": "s",
                "force": "kN",
                "stress": "kPa",
                "angle": "radian",
            },
            "conversions": list(compiled.unit_conversions),
        },
        "mesh": {
            "dimension": mesh.dimension,
            "cell_type": mesh.cell_type,
            "coordinate_order": mesh.coordinate_order,
            "logical_digest": mesh.logical_digest(),
            "cell_count": mesh.cells.shape[0],
            "coordinate_node_count": mesh.coordinate_nodes.shape[0],
        },
        "spaces": compiled.spaces,
        "supports": {
            name: {
                "kind": support.kind,
                "dimension": support.dimension,
                "entity_ids": [_hex_id(item) for item in support.entity_ids],
            }
            for name, support in sorted(mesh.supports.items())
        },
        "materials": {material["name"]: material for material in compiled.materials},
        "physics": case.physics.model_dump(mode="json"),
        "boundary_rules": {rule["name"]: rule for rule in compiled.boundary_rules},
        "initial_state": {"kind": "zero"},
        "coupling": {"mode": case.physics.coupling},
        "operators": {
            "analysis": case.analysis.selector,
            "operator_class": "nonsymmetric",
            "quadrature_registry": "generated-v1",
            # Only a non-default volume rule is recorded, so existing problem
            # identities are unchanged.
            **(
                {"volume_quadrature": "paired_higher"}
                if case.discretization.volume_quadrature == "paired_higher"
                else {}
            ),
        },
        "outputs": {"required": list(case.required_outputs)},
        "payloads": {
            "problem_hdf5": {
                "path": "problem.h5",
                "media_type": "application/x-hdf5",
                "layout_id": "compiled-problem-v1",
                "logical_digest": hdf5_logical_digest,
                "byte_count": hdf5_byte_count,
                "sha256": hdf5_sha256,
            }
        },
    }
    if mesh.provenance.uniform_refinement_levels:
        problem_without_id["mesh"]["uniform_refinement_levels"] = (
            mesh.provenance.uniform_refinement_levels
        )
    if compiled.prescribed_saturation is not None:
        problem_without_id["prescribed_saturation"] = dict(
            compiled.prescribed_saturation
        )
    problem_id = content_id(problem_without_id)
    problem_manifest = {"id": problem_id, **problem_without_id}
    compile_report_document = {
        "kind": "slope_stability.compile_report",
        "schema_version": {"major": 1, "minor": 0, "patch": 0},
        "problem_id": problem_id,
        "source": {
            "asset_id": asset.id,
            "case_id": case.id,
            "importer_id": mesh.provenance.importer_id,
            "source_sha256": mesh.provenance.source_sha256,
        },
        "mesh_logical_digest": mesh.logical_digest(),
        "hdf5_logical_digest": hdf5_logical_digest,
        "hdf5_sha256": hdf5_sha256,
        "hdf5_byte_count": hdf5_byte_count,
        "counts": dict(compiled.counts),
        "spaces": compiled.spaces,
        "geometry_audit": logical_payload["geometry_audit"],
        "unit_conversions": list(compiled.unit_conversions),
        "warnings": [],
        "issues": [],
    }
    if mesh.provenance.uniform_refinement_levels:
        compile_report_document["source"]["uniform_refinement_levels"] = (
            mesh.provenance.uniform_refinement_levels
        )
    expected_problem_bytes = canonicalize(problem_manifest)
    expected_report_bytes = canonicalize(compile_report_document)
    package_path = destination / problem_id.replace(":", "-")
    if package_path.exists():
        shutil.rmtree(temporary, ignore_errors=True)
        problem_json = package_path / "problem.json"
        report = package_path / "compile-report.json"
        payload = package_path / "problem.h5"
        if not all(item.is_file() for item in (problem_json, report, payload)):
            raise CompileError("existing_package_incomplete", str(package_path))
        if (
            problem_json.read_bytes() != expected_problem_bytes
            or report.read_bytes() != expected_report_bytes
            or payload.stat().st_size != hdf5_byte_count
            or file_sha256(payload) != hdf5_sha256
        ):
            raise CompileError("existing_package_mismatch", str(package_path))
        return CompiledPackage(
            problem_id=problem_id,
            path=package_path,
            problem_json=problem_json,
            compile_report=report,
            hdf5_logical_digest=hdf5_logical_digest,
        )

    try:
        (temporary / "problem.json").write_bytes(expected_problem_bytes)
        (temporary / "compile-report.json").write_bytes(expected_report_bytes)
        if failure_at == "after_json":
            raise CompileError("injected_write_failure", failure_at)

        for file_path in (
            payload_path,
            temporary / "problem.json",
            temporary / "compile-report.json",
        ):
            fsync_path(file_path)
        fsync_path(temporary)
        if failure_at == "before_rename":
            raise CompileError("injected_write_failure", failure_at)
        os.replace(temporary, package_path)
        fsync_path(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return CompiledPackage(
        problem_id=problem_id,
        path=package_path,
        problem_json=package_path / "problem.json",
        compile_report=package_path / "compile-report.json",
        hdf5_logical_digest=hdf5_logical_digest,
    )


__all__ = [
    "CompiledPackage",
    "compile_problem",
]
