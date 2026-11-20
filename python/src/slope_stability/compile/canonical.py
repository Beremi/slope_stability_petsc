# SPDX-License-Identifier: BSD-2-Clause
"""Canonical simplex meshes with stable entity identities."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import combinations
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from slope_protocol import canonicalize, content_id
from ._common import (
    CompileError,
    FloatArray,
    UIntArray,
    _barycentric_functionals,
    _hex_id,
    _little_f64,
    _little_u64,
    _stable_u64,
)
from .raw import RawMesh, RawSupport


@dataclass(frozen=True)
class EntitySet:
    """Named support as a sorted tuple of stable entity identities."""

    name: str
    kind: str
    dimension: int
    entity_ids: tuple[int, ...]


@dataclass(frozen=True)
class ImportMaps:
    """Mapping from importer node numbers to canonical node indices."""

    source_to_canonical_nodes: Mapping[int, int]


@dataclass(frozen=True)
class MeshProvenance:
    """Importer identity, source-mesh SHA-256 and applied refinement levels."""

    importer_id: str
    source_sha256: str
    uniform_refinement_levels: int = 0


def _entity_keys(cells: UIntArray, dimension: int) -> dict[int, set[tuple[int, ...]]]:
    result: dict[int, set[tuple[int, ...]]] = {
        item: set() for item in range(dimension + 1)
    }
    for row in cells:
        vertices = tuple(int(item) for item in row[: dimension + 1])
        for entity_dimension in range(dimension + 1):
            width = entity_dimension + 1
            result[entity_dimension].update(
                tuple(sorted(items)) for items in combinations(vertices, width)
            )
    return result


@dataclass(frozen=True)
class CanonicalMesh:
    """Simplex mesh in canonical order with stable entity identities.

    Nodes and cells are ordered by stable identity, every entity of every
    dimension carries a stable 64-bit identity, and every support is an
    :class:`EntitySet`. :meth:`logical_digest` is independent of the importer's
    numbering.
    """

    dimension: Literal[2, 3]
    cell_type: Literal["triangle", "tetrahedron"]
    cells: UIntArray
    coordinate_order: Literal[1, 2, 4]
    coordinate_nodes: FloatArray
    coordinate_connectivity: UIntArray
    coordinate_barycentric_functionals: UIntArray
    coordinate_functional_ids: UIntArray
    stable_entity_ids: Mapping[int, UIntArray]
    stable_entity_vertex_ids: Mapping[int, UIntArray]
    topology_vertex_ids: UIntArray
    topology_vertex_coordinates: FloatArray
    topology_cell_ids: UIntArray
    topology_cells: UIntArray
    supports: Mapping[str, EntitySet]
    import_maps: ImportMaps
    provenance: MeshProvenance
    partition_hints: NDArray[np.int64]

    def _logical_payload(self) -> dict[str, Any]:
        node_order = np.argsort(self.coordinate_functional_ids, kind="stable")
        inverse = np.empty_like(node_order)
        inverse[node_order] = np.arange(node_order.size)
        coordinates = self.coordinate_nodes[node_order]
        cells = inverse[self.cells]
        connectivity = inverse[self.coordinate_connectivity]
        row_order = sorted(
            range(cells.shape[0]),
            key=lambda index: tuple(sorted(int(item) for item in cells[index])),
        )
        return {
            "dimension": self.dimension,
            "cell_type": self.cell_type,
            "coordinate_order": self.coordinate_order,
            "cells": cells[row_order].tolist(),
            "coordinate_nodes": coordinates.tolist(),
            "coordinate_connectivity": connectivity[row_order].tolist(),
            "coordinate_barycentric_functionals": (
                self.coordinate_barycentric_functionals.tolist()
            ),
            "coordinate_functional_ids": sorted(
                _hex_id(int(item)) for item in self.coordinate_functional_ids
            ),
            "stable_entity_ids": {
                str(dimension): sorted(_hex_id(int(item)) for item in values)
                for dimension, values in sorted(self.stable_entity_ids.items())
            },
            "stable_entity_vertex_ids": {
                str(dimension): [
                    [_hex_id(int(item)) for item in row]
                    for row in self.stable_entity_vertex_ids[dimension]
                ]
                for dimension in sorted(self.stable_entity_vertex_ids)
            },
            "topology": {
                "vertex_ids": [_hex_id(int(item)) for item in self.topology_vertex_ids],
                "vertex_coordinates": self.topology_vertex_coordinates.tolist(),
                "cell_ids": [_hex_id(int(item)) for item in self.topology_cell_ids],
                "cells": self.topology_cells.tolist(),
            },
            "supports": {
                name: {
                    "kind": support.kind,
                    "dimension": support.dimension,
                    "entity_ids": sorted(
                        _hex_id(entity_id) for entity_id in support.entity_ids
                    ),
                }
                for name, support in sorted(self.supports.items())
            },
        }

    def logical_digest(self) -> str:
        return content_id(self._logical_payload())

    def reordered(
        self,
        *,
        cell_order: tuple[int, ...],
        coordinate_node_order: tuple[int, ...],
        partition_hints: tuple[int, ...],
    ) -> "CanonicalMesh":
        """Exercise a physical permutation and return canonical order again."""
        cell_permutation = np.asarray(cell_order, dtype=np.int64)
        node_permutation = np.asarray(coordinate_node_order, dtype=np.int64)
        if sorted(cell_permutation.tolist()) != list(range(self.cells.shape[0])):
            raise ValueError("cell_order is not a permutation")
        if sorted(node_permutation.tolist()) != list(
            range(self.coordinate_nodes.shape[0])
        ):
            raise ValueError("coordinate_node_order is not a permutation")
        inverse = np.empty_like(node_permutation)
        inverse[node_permutation] = np.arange(node_permutation.size)
        permuted_nodes = self.coordinate_nodes[node_permutation]
        permuted_functionals = self.coordinate_functional_ids[node_permutation]
        permuted_cells = inverse[self.cells[cell_permutation]]
        permuted_connectivity = inverse[self.coordinate_connectivity[cell_permutation]]
        canonical_position = {
            int(functional): index
            for index, functional in enumerate(self.coordinate_functional_ids)
        }
        canonical_node_order = np.asarray(
            sorted(
                range(permuted_functionals.size),
                key=lambda index: canonical_position[int(permuted_functionals[index])],
            ),
            dtype=np.int64,
        )
        canonical_inverse = np.empty_like(canonical_node_order)
        canonical_inverse[canonical_node_order] = np.arange(canonical_node_order.size)
        canonical_cells = canonical_inverse[permuted_cells]
        canonical_connectivity = canonical_inverse[permuted_connectivity]
        canonical_functionals = permuted_functionals[canonical_node_order]
        canonical_nodes = permuted_nodes[canonical_node_order]
        canonical_cell_order = sorted(
            range(canonical_cells.shape[0]),
            key=lambda index: tuple(
                sorted(int(item) for item in canonical_cells[index])
            ),
        )
        return replace(
            self,
            cells=_little_u64(canonical_cells[canonical_cell_order]),
            coordinate_nodes=_little_f64(canonical_nodes),
            coordinate_connectivity=_little_u64(
                canonical_connectivity[canonical_cell_order]
            ),
            coordinate_functional_ids=_little_u64(canonical_functionals),
            partition_hints=np.asarray(partition_hints, dtype=np.int64),
        )


def canonicalize_mesh(raw: RawMesh) -> CanonicalMesh:
    """Canonicalize importer-local numbering without using partition hints."""
    dimension = raw.dimension
    if raw.cell_type != ("triangle" if dimension == 2 else "tetrahedron"):
        raise CompileError("cell_type_dimension_mismatch", raw.cell_type)
    source_ids = _little_u64(raw.source_node_ids)
    coordinates = _little_f64(raw.coordinate_nodes)
    if coordinates.ndim != 2 or coordinates.shape[1] != dimension:
        raise CompileError("coordinate_shape", repr(coordinates.shape))
    if source_ids.shape != (coordinates.shape[0],):
        raise CompileError("source_node_count", repr(source_ids.shape))
    if len(set(int(item) for item in source_ids)) != source_ids.size:
        raise CompileError("duplicate_source_node_id", "source node IDs")

    node_order = np.argsort(source_ids, kind="stable")
    inverse = np.empty_like(node_order)
    inverse[node_order] = np.arange(node_order.size)
    canonical_source_ids = source_ids[node_order]
    canonical_coordinates = coordinates[node_order]
    cells = _little_u64(inverse[np.asarray(raw.cells, dtype=np.int64)])
    connectivity = _little_u64(
        inverse[np.asarray(raw.coordinate_connectivity, dtype=np.int64)]
    )
    if cells.ndim != 2 or cells.shape[1] != dimension + 1:
        raise CompileError("cell_shape", repr(cells.shape))
    if connectivity.shape[0] != cells.shape[0]:
        raise CompileError("coordinate_connectivity_shape", repr(connectivity.shape))
    cell_keys = [tuple(sorted(int(item) for item in row)) for row in cells]
    if len(set(cell_keys)) != len(cell_keys):
        raise CompileError("duplicate_cell", "canonical vertex closure")
    row_order = sorted(
        range(cells.shape[0]),
        key=lambda index: tuple(sorted(int(item) for item in cells[index])),
    )
    cells = _little_u64(cells[row_order])
    connectivity = _little_u64(connectivity[row_order])
    if raw.local_barycentric_functionals is None:
        local_functionals = _little_u64(
            _barycentric_functionals(dimension, raw.coordinate_order)
        )
    else:
        local_functionals = _little_u64(raw.local_barycentric_functionals)
    if local_functionals.shape != (
        connectivity.shape[1],
        dimension + 1,
    ):
        raise CompileError("coordinate_functional_shape", repr(local_functionals.shape))
    if np.any(np.sum(local_functionals, axis=1) != raw.coordinate_order):
        raise CompileError("coordinate_functional_order", str(raw.coordinate_order))

    topology = _entity_keys(cells, dimension)
    entity_lookup: dict[int, dict[tuple[int, ...], int]] = {}
    stable_entities: dict[int, UIntArray] = {}
    for entity_dimension, keys in topology.items():
        mapping: dict[tuple[int, ...], int] = {}
        for key in keys:
            source_key = tuple(
                sorted(int(canonical_source_ids[index]) for index in key)
            )
            mapping[key] = _stable_u64(
                "entity", dimension, entity_dimension, source_key
            )
        entity_lookup[entity_dimension] = mapping
    stable_entity_vertices: dict[int, UIntArray] = {}
    vertex_id_by_coordinate = {
        key[0]: stable_id for key, stable_id in entity_lookup[0].items()
    }
    ordered_vertex_coordinates: list[int] = []
    for entity_dimension, mapping in entity_lookup.items():
        ordered = sorted(mapping.items(), key=lambda item: item[1])
        stable_entities[entity_dimension] = _little_u64(
            [stable_id for _, stable_id in ordered]
        )
        stable_entity_vertices[entity_dimension] = _little_u64(
            [
                sorted(vertex_id_by_coordinate[coordinate] for coordinate in key)
                for key, _ in ordered
            ]
        )
        if entity_dimension == 0:
            ordered_vertex_coordinates = [key[0] for key, _ in ordered]
    topology_vertex_ids = stable_entities[0]
    topology_vertex_coordinates = _little_f64(
        canonical_coordinates[ordered_vertex_coordinates]
    )
    topology_index_by_coordinate = {
        coordinate: index for index, coordinate in enumerate(ordered_vertex_coordinates)
    }
    topology_cells = _little_u64(
        [
            [topology_index_by_coordinate[int(coordinate)] for coordinate in row]
            for row in cells
        ]
    )
    topology_cell_ids = _little_u64(
        [
            entity_lookup[dimension][tuple(sorted(int(item) for item in row))]
            for row in cells
        ]
    )

    canonical_supports: dict[str, EntitySet] = {}
    for support in sorted(raw.supports, key=lambda item: item.name):
        if support.name in canonical_supports:
            raise CompileError("duplicate_support", support.name)
        mapping = entity_lookup.get(support.dimension, {})
        ids: list[int] = []
        for raw_entity in support.entities:
            try:
                key = tuple(sorted(int(inverse[int(index)]) for index in raw_entity))
            except IndexError as error:
                raise CompileError("support_node_out_of_range", support.name) from error
            if key not in mapping:
                raise CompileError(
                    "support_entity_not_in_mesh", f"{support.name}: {key}"
                )
            ids.append(mapping[key])
        canonical_supports[support.name] = EntitySet(
            name=support.name,
            kind=support.kind,
            dimension=support.dimension,
            entity_ids=tuple(sorted(set(ids))),
        )

    functional_keys: list[tuple[tuple[int, int], ...] | None] = [
        None
    ] * canonical_coordinates.shape[0]
    for cell_index, row in enumerate(connectivity):
        vertices = cells[cell_index]
        for local_index, node_index_value in enumerate(row):
            weights = local_functionals[local_index]
            key = tuple(
                sorted(
                    (
                        int(canonical_source_ids[int(vertices[vertex_index])]),
                        int(weight),
                    )
                    for vertex_index, weight in enumerate(weights)
                    if weight
                )
            )
            node_index = int(node_index_value)
            prior = functional_keys[node_index]
            if prior is not None and prior != key:
                raise CompileError(
                    "shared_coordinate_functional_mismatch",
                    f"node {node_index}: {prior} != {key}",
                )
            functional_keys[node_index] = key
    if any(key is None for key in functional_keys):
        raise CompileError(
            "unused_coordinate_node",
            "every coordinate node must own a topological functional",
        )
    functional_ids = _little_u64(
        [
            _stable_u64(
                "coordinate-functional",
                raw.coordinate_order,
                key,
            )
            for key in functional_keys
        ]
    )
    source_map = {
        int(source_id): int(index)
        for index, source_id in enumerate(canonical_source_ids)
    }
    source_digest = (
        raw.source_sha256
        or sha256(
            canonicalize(
                {
                    "source_ids": canonical_source_ids.tolist(),
                    "coordinates": canonical_coordinates.tolist(),
                    "cells": cells.tolist(),
                }
            )
        ).hexdigest()
    )
    return CanonicalMesh(
        dimension=dimension,
        cell_type=raw.cell_type,
        cells=cells,
        coordinate_order=raw.coordinate_order,
        coordinate_nodes=canonical_coordinates,
        coordinate_connectivity=connectivity,
        coordinate_barycentric_functionals=local_functionals,
        coordinate_functional_ids=functional_ids,
        stable_entity_ids=stable_entities,
        stable_entity_vertex_ids=stable_entity_vertices,
        topology_vertex_ids=topology_vertex_ids,
        topology_vertex_coordinates=topology_vertex_coordinates,
        topology_cell_ids=topology_cell_ids,
        topology_cells=topology_cells,
        supports=canonical_supports,
        import_maps=ImportMaps(source_to_canonical_nodes=source_map),
        provenance=MeshProvenance(
            importer_id=raw.importer_id,
            source_sha256=source_digest,
            uniform_refinement_levels=raw.uniform_refinement_levels,
        ),
        partition_hints=np.asarray(raw.partition_hints, dtype=np.int64),
    )


def compact_simplex_mesh(
    dimension: Literal[2, 3],
    coordinate_order: Literal[1, 2, 4],
    *,
    two_cells: bool = False,
) -> CanonicalMesh:
    """Construct a deterministic straight compact simplex mesh."""
    if dimension == 2:
        vertices = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        cells = [(0, 1, 2)]
        if two_cells:
            vertices.append((1.0, 1.0))
            cells.append((3, 2, 1))
        edge_counts: dict[tuple[int, int], int] = {}
        for cell in cells:
            for edge in combinations(cell, 2):
                key = tuple(sorted(edge))
                edge_counts[key] = edge_counts.get(key, 0) + 1
        exterior = tuple(key for key, count in edge_counts.items() if count == 1)
        top = (2, 3) if two_cells else (1, 2)
        supports = (
            RawSupport("domain", "region", 2, tuple(cells)),
            RawSupport("base", "boundary", 1, ((0, 1),)),
            RawSupport("top", "boundary", 1, (top,)),
            RawSupport("boundary", "boundary", 1, exterior),
        )
    elif dimension == 3:
        if two_cells:
            raise ValueError("the compact 3D fixture has one cell")
        vertices = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        cells = [(0, 1, 2, 3)]
        faces = tuple(tuple(item) for item in combinations(cells[0], 3))
        supports = (
            RawSupport("domain", "region", 3, tuple(cells)),
            RawSupport("base", "boundary", 2, ((0, 1, 2),)),
            RawSupport("top", "boundary", 2, ((1, 2, 3),)),
            RawSupport("boundary", "boundary", 2, faces),
        )
    else:
        raise ValueError("dimension must be 2 or 3")

    functionals = _barycentric_functionals(dimension, coordinate_order)
    node_coordinates: list[tuple[float, ...]] = [tuple(item) for item in vertices]
    node_keys: dict[tuple[tuple[int, int], ...], int] = {
        ((index, coordinate_order),): index for index in range(len(vertices))
    }
    connectivity: list[list[int]] = []
    for cell in cells:
        local: list[int] = []
        for weights in functionals:
            key = tuple(
                sorted(
                    (int(cell[local_vertex]), int(weight))
                    for local_vertex, weight in enumerate(weights)
                    if weight
                )
            )
            index = node_keys.get(key)
            if index is None:
                coordinate = tuple(
                    sum(
                        weights[local_vertex] * vertices[global_vertex][axis]
                        for local_vertex, global_vertex in enumerate(cell)
                    )
                    / coordinate_order
                    for axis in range(dimension)
                )
                index = len(node_coordinates)
                node_keys[key] = index
                node_coordinates.append(coordinate)
            local.append(index)
        connectivity.append(local)
    raw = RawMesh(
        dimension=dimension,
        cell_type="triangle" if dimension == 2 else "tetrahedron",
        coordinate_order=coordinate_order,
        source_node_ids=np.arange(len(node_coordinates), dtype=np.uint64),
        coordinate_nodes=np.asarray(node_coordinates, dtype=np.float64),
        cells=np.asarray(cells, dtype=np.uint64),
        coordinate_connectivity=np.asarray(connectivity, dtype=np.uint64),
        supports=supports,
        partition_hints=np.zeros(len(cells), dtype=np.int64),
        importer_id="compact.simplex",
    )
    return canonicalize_mesh(raw)


__all__ = [
    "CanonicalMesh",
    "EntitySet",
    "ImportMaps",
    "MeshProvenance",
    "canonicalize_mesh",
    "compact_simplex_mesh",
]
