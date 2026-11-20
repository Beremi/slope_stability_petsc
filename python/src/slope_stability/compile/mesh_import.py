# SPDX-License-Identifier: BSD-2-Clause
"""Gmsh MSH 4.x and frozen HDF5 reference mesh importers."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import h5py
import meshio
import numpy as np

from ._common import CompileError, UIntArray, _little_u64
from .canonical import CanonicalMesh, canonicalize_mesh
from .raw import ImportRequest, RawMesh, RawSupport
from .refine import uniformly_refine_raw_mesh


_DOMAIN_CELL_TYPES = {
    (2, 1): "triangle",
    (2, 2): "triangle6",
    (2, 4): "triangle15",
    (3, 1): "tetra",
    (3, 2): "tetra10",
    (3, 4): "tetra35",
}


def _gmsh_barycentric_functionals(
    dimension: int,
    order: int,
) -> UIntArray:
    """Return the Gmsh/meshio complete-simplex node order."""
    if dimension == 2 and order == 1:
        values = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    elif dimension == 2 and order == 2:
        values = [
            (2, 0, 0),
            (0, 2, 0),
            (0, 0, 2),
            (1, 1, 0),
            (0, 1, 1),
            (1, 0, 1),
        ]
    elif dimension == 2 and order == 4:
        values = [
            (4, 0, 0),
            (0, 4, 0),
            (0, 0, 4),
            (3, 1, 0),
            (2, 2, 0),
            (1, 3, 0),
            (0, 3, 1),
            (0, 2, 2),
            (0, 1, 3),
            (1, 0, 3),
            (2, 0, 2),
            (3, 0, 1),
            (2, 1, 1),
            (1, 2, 1),
            (1, 1, 2),
        ]
    elif dimension == 3 and order == 1:
        values = [
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        ]
    elif dimension == 3 and order == 2:
        values = [
            (2, 0, 0, 0),
            (0, 2, 0, 0),
            (0, 0, 2, 0),
            (0, 0, 0, 2),
            (1, 1, 0, 0),
            (0, 1, 1, 0),
            (1, 0, 1, 0),
            (1, 0, 0, 1),
            (0, 1, 0, 1),
            (0, 0, 1, 1),
        ]
    elif dimension == 3 and order == 4:
        values = [
            (4, 0, 0, 0),
            (0, 4, 0, 0),
            (0, 0, 4, 0),
            (0, 0, 0, 4),
            (3, 1, 0, 0),
            (2, 2, 0, 0),
            (1, 3, 0, 0),
            (0, 3, 1, 0),
            (0, 2, 2, 0),
            (0, 1, 3, 0),
            (1, 0, 3, 0),
            (2, 0, 2, 0),
            (3, 0, 1, 0),
            (1, 0, 0, 3),
            (2, 0, 0, 2),
            (3, 0, 0, 1),
            (0, 0, 1, 3),
            (0, 0, 2, 2),
            (0, 0, 3, 1),
            (0, 1, 0, 3),
            (0, 2, 0, 2),
            (0, 3, 0, 1),
            (2, 1, 1, 0),
            (1, 1, 2, 0),
            (1, 2, 1, 0),
            (2, 1, 0, 1),
            (1, 2, 0, 1),
            (1, 1, 0, 2),
            (2, 0, 1, 1),
            (1, 0, 1, 2),
            (1, 0, 2, 1),
            (0, 1, 1, 2),
            (0, 2, 1, 1),
            (0, 1, 2, 1),
            (1, 1, 1, 1),
        ]
    else:
        raise CompileError(
            "unsupported_gmsh_element_order",
            f"dimension={dimension}, order={order}",
        )
    return _little_u64(values)


def _meshio_dimension(cell_type: str) -> int | None:
    if cell_type.startswith("vertex"):
        return 0
    if cell_type.startswith("line"):
        return 1
    if cell_type.startswith("triangle") or cell_type.startswith("quad"):
        return 2
    if cell_type.startswith("tetra") or cell_type.startswith("hexahedron"):
        return 3
    return None


def _gmsh_ascii_node_tags(source: bytes) -> UIntArray:
    """Read MSH 4.1 ASCII node tags in the same order as their coordinates."""
    try:
        lines = source.decode("utf-8", "strict").splitlines()
    except UnicodeDecodeError as error:
        raise CompileError(
            "unsupported_gmsh_binary",
            "the initial MSH 4.x adapter requires an ASCII file",
        ) from error
    try:
        format_index = lines.index("$MeshFormat")
        version, binary, _size = lines[format_index + 1].split()
    except (ValueError, IndexError) as error:
        raise CompileError("invalid_gmsh_header", "$MeshFormat") from error
    if not version.startswith("4.") or binary != "0":
        raise CompileError(
            "unsupported_gmsh_format", f"version={version}, binary={binary}"
        )
    try:
        index = lines.index("$Nodes") + 1
        block_count, node_count, _minimum, _maximum = (
            int(item) for item in lines[index].split()
        )
    except (ValueError, IndexError) as error:
        raise CompileError("invalid_gmsh_nodes", "$Nodes header") from error
    index += 1
    tags: list[int] = []
    for _block in range(block_count):
        try:
            _entity_dimension, _entity_tag, _parametric, block_size = (
                int(item) for item in lines[index].split()
            )
        except (ValueError, IndexError) as error:
            raise CompileError("invalid_gmsh_nodes", "entity block") from error
        index += 1
        block_tags: list[int] = []
        while len(block_tags) < block_size:
            try:
                block_tags.extend(int(item) for item in lines[index].split())
            except (ValueError, IndexError) as error:
                raise CompileError("invalid_gmsh_nodes", "node tags") from error
            index += 1
        if len(block_tags) != block_size:
            raise CompileError("invalid_gmsh_nodes", "node tag count")
        tags.extend(block_tags)
        index += block_size
    if len(tags) != node_count or lines[index] != "$EndNodes":
        raise CompileError("invalid_gmsh_nodes", "node count or terminator")
    return _little_u64(tags)


class GmshImporter:
    """Gmsh MSH 4.x adapter through meshio's data-only reader."""

    importer_id = "gmsh.msh4"

    def import_mesh(
        self,
        source: str | Path,
        requested: ImportRequest,
    ) -> CanonicalMesh:
        path = Path(source)
        source_bytes = path.read_bytes()
        mesh = meshio.read(path)
        domain_type = _DOMAIN_CELL_TYPES[
            (requested.dimension, requested.coordinate_order)
        ]
        cells_dict = mesh.cells_dict
        if domain_type not in cells_dict:
            raise CompileError(
                "unsupported_gmsh_element",
                f"missing {domain_type}; available {sorted(cells_dict)}",
            )
        full_connectivity = np.asarray(cells_dict[domain_type], dtype=np.uint64)
        cells = full_connectivity[:, : requested.dimension + 1]
        cell_sets = mesh.cell_sets_dict
        raw_supports: list[RawSupport] = []
        for name in requested.required_supports:
            members = cell_sets.get(name)
            if not members:
                raise CompileError("missing_physical_group", name)
            entities: list[tuple[int, ...]] = []
            support_dimension: int | None = None
            for cell_type, indices in members.items():
                entity_dimension = _meshio_dimension(cell_type)
                if entity_dimension is None or entity_dimension > requested.dimension:
                    continue
                block = cells_dict.get(cell_type)
                if block is None:
                    continue
                if support_dimension is None:
                    support_dimension = entity_dimension
                if support_dimension != entity_dimension:
                    raise CompileError("mixed_support_dimension", name)
                vertex_count = entity_dimension + 1
                for index in np.asarray(indices, dtype=np.int64):
                    entities.append(
                        tuple(int(item) for item in block[index, :vertex_count])
                    )
            if support_dimension is None or not entities:
                raise CompileError("empty_physical_group", name)
            raw_supports.append(
                RawSupport(
                    name=name,
                    kind=(
                        "node_set"
                        if name.startswith("nodeset:")
                        else (
                            "region"
                            if support_dimension == requested.dimension
                            else "boundary"
                        )
                    ),
                    dimension=support_dimension,
                    entities=tuple(entities),
                )
            )
        coordinates = np.asarray(
            mesh.points[:, : requested.dimension], dtype=np.float64
        )
        node_tags = _gmsh_ascii_node_tags(source_bytes)
        if node_tags.shape != (coordinates.shape[0],):
            raise CompileError(
                "gmsh_node_map_mismatch",
                f"{node_tags.shape[0]} tags for {coordinates.shape[0]} points",
            )
        raw = RawMesh(
            dimension=requested.dimension,
            cell_type="triangle" if requested.dimension == 2 else "tetrahedron",
            coordinate_order=requested.coordinate_order,
            source_node_ids=node_tags,
            coordinate_nodes=coordinates,
            cells=cells,
            coordinate_connectivity=full_connectivity,
            supports=tuple(raw_supports),
            partition_hints=np.zeros(cells.shape[0], dtype=np.int64),
            importer_id=self.importer_id,
            source_sha256=sha256(source_bytes).hexdigest(),
            local_barycentric_functionals=_gmsh_barycentric_functionals(
                requested.dimension,
                requested.coordinate_order,
            ),
        )
        raw = uniformly_refine_raw_mesh(raw, levels=requested.uniform_refinement_levels)
        return canonicalize_mesh(raw)


class Hdf5ReferenceImporter:
    """Explicit adapter for the two frozen reference HDF5 layouts."""

    importer_id = "reference.hdf5"

    def import_mesh(
        self,
        source: str | Path,
        requested: ImportRequest,
    ) -> CanonicalMesh:
        path = Path(source)
        with h5py.File(path, "r") as handle:
            layout = str(handle.attrs["layout"])
            dimension = int(handle.attrs["dimension"])
            cell_type = str(handle.attrs["cell_type"])
            coordinates = np.asarray(handle["coordinates"], dtype=np.float64)
            cells = np.asarray(handle["cells"], dtype=np.int64)
            if layout == "matlab":
                coordinates = coordinates.T
                cells = cells.T - 1
            elif layout != "legacy":
                raise CompileError("unknown_reference_layout", layout)
            supports = []
            for name, dataset in sorted(handle["supports"].items()):
                entities = np.asarray(dataset, dtype=np.int64)
                support_dimension = entities.shape[1] - 1
                supports.append(
                    RawSupport(
                        name=name,
                        kind=(
                            "region" if support_dimension == dimension else "boundary"
                        ),
                        dimension=support_dimension,
                        entities=tuple(
                            tuple(int(item) for item in row) for row in entities
                        ),
                    )
                )
        if dimension != requested.dimension or cell_type != (
            "triangle" if dimension == 2 else "tetrahedron"
        ):
            raise CompileError("reference_mesh_mismatch", path.name)
        if requested.coordinate_order != 1:
            raise CompileError(
                "reference_coordinate_order",
                str(requested.coordinate_order),
            )
        missing = set(requested.required_supports).difference(
            item.name for item in supports
        )
        if missing:
            raise CompileError("missing_support", sorted(missing)[0])
        raw = RawMesh(
            dimension=dimension,
            cell_type=cell_type,
            coordinate_order=1,
            source_node_ids=np.arange(coordinates.shape[0], dtype=np.uint64),
            coordinate_nodes=coordinates[:, :dimension],
            cells=np.asarray(cells, dtype=np.uint64),
            coordinate_connectivity=np.asarray(cells, dtype=np.uint64),
            supports=tuple(supports),
            partition_hints=np.zeros(cells.shape[0], dtype=np.int64),
            importer_id=f"{self.importer_id}.{layout}",
            source_sha256=sha256(path.read_bytes()).hexdigest(),
        )
        raw = uniformly_refine_raw_mesh(raw, levels=requested.uniform_refinement_levels)
        return canonicalize_mesh(raw)


__all__ = [
    "GmshImporter",
    "Hdf5ReferenceImporter",
]
