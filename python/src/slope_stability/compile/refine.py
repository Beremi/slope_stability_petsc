# SPDX-License-Identifier: BSD-2-Clause
"""Deterministic uniform refinement of straight raw simplex meshes."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from ._common import (
    CompileError,
    _barycentric_functionals,
    _little_f64,
    _little_u64,
    _simplex_determinant,
    _stable_u64,
)
from .raw import RawMesh, RawSupport


def _orient_child_like_parent(
    child: tuple[int, ...],
    parent_determinant: float,
    coordinates: list[NDArray[np.float64]],
) -> tuple[int, ...]:
    determinant = _simplex_determinant(child, coordinates)
    if parent_determinant == 0.0 or determinant == 0.0:
        raise CompileError("degenerate_refinement_cell", repr(child))
    if determinant * parent_determinant < 0.0:
        values = list(child)
        values[1], values[2] = values[2], values[1]
        child = tuple(values)
    return child


def _uniformly_refine_raw_mesh_once(raw: RawMesh) -> RawMesh:
    if raw.coordinate_order != 1:
        raise CompileError(
            "uniform_refinement_coordinate_order",
            "uniform refinement currently requires straight P1 coordinate geometry",
        )
    dimension = raw.dimension
    expected_width = dimension + 1
    cells = np.asarray(raw.cells, dtype=np.int64)
    connectivity = np.asarray(raw.coordinate_connectivity, dtype=np.int64)
    source_ids = [int(item) for item in np.asarray(raw.source_node_ids)]
    coordinates = [
        np.asarray(row, dtype=np.float64).copy()
        for row in np.asarray(raw.coordinate_nodes, dtype=np.float64)
    ]
    if (
        dimension not in (2, 3)
        or cells.ndim != 2
        or cells.shape[1] != expected_width
        or connectivity.shape != cells.shape
        or not np.array_equal(cells, connectivity)
        or len(source_ids) != len(coordinates)
        or np.asarray(raw.partition_hints).shape != (cells.shape[0],)
    ):
        raise CompileError("uniform_refinement_input", "invalid P1 simplex mesh")
    if len(set(source_ids)) != len(source_ids):
        raise CompileError("duplicate_source_node_id", "uniform refinement input")

    edge_keys = {
        tuple(sorted((int(first), int(second))))
        for cell in cells
        for first, second in combinations(cell, 2)
    }
    edge_keys = set(
        sorted(
            edge_keys,
            key=lambda edge: tuple(sorted((source_ids[edge[0]], source_ids[edge[1]]))),
        )
    )
    midpoint_by_edge: dict[tuple[int, int], int] = {}
    occupied_ids = set(source_ids)
    for edge in sorted(
        edge_keys,
        key=lambda item: tuple(sorted((source_ids[item[0]], source_ids[item[1]]))),
    ):
        first, second = edge
        endpoint_ids = tuple(sorted((source_ids[first], source_ids[second])))
        midpoint_id = _stable_u64(
            "uniform-refinement-midpoint", dimension, endpoint_ids
        ) & ((1 << 63) - 1)
        if midpoint_id in occupied_ids:
            raise CompileError(
                "uniform_refinement_id_collision",
                f"{endpoint_ids[0]}:{endpoint_ids[1]}",
            )
        midpoint_by_edge[edge] = len(source_ids)
        occupied_ids.add(midpoint_id)
        source_ids.append(midpoint_id)
        coordinates.append(0.5 * (coordinates[first] + coordinates[second]))

    def midpoint(first: int, second: int) -> int:
        return midpoint_by_edge[tuple(sorted((int(first), int(second))))]

    def triangle_children(entity: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        first, second, third = (int(item) for item in entity)
        first_second = midpoint(first, second)
        first_third = midpoint(first, third)
        second_third = midpoint(second, third)
        return (
            (first, first_second, first_third),
            (first_second, second, second_third),
            (first_third, second_third, third),
            (first_second, second_third, first_third),
        )

    children_by_cell: dict[tuple[int, ...], tuple[tuple[int, ...], ...]] = {}
    refined_cells: list[tuple[int, ...]] = []
    refined_hints: list[int] = []
    for cell_index, row in enumerate(cells):
        parent = tuple(int(item) for item in row)
        parent_determinant = _simplex_determinant(parent, coordinates)
        if parent_determinant == 0.0:
            raise CompileError("degenerate_refinement_cell", repr(parent))
        if dimension == 2:
            candidates = triangle_children(parent)
        else:
            vertices = tuple(parent)
            corner_children = []
            for vertex in vertices:
                incident = [
                    midpoint(vertex, other) for other in vertices if other != vertex
                ]
                corner_children.append((vertex, *incident))
            opposite_edges = (
                ((vertices[0], vertices[1]), (vertices[2], vertices[3])),
                ((vertices[0], vertices[2]), (vertices[1], vertices[3])),
                ((vertices[0], vertices[3]), (vertices[1], vertices[2])),
            )

            def diagonal_key(
                pair: tuple[tuple[int, int], tuple[int, int]],
            ) -> tuple[int, int]:
                values = tuple(sorted(source_ids[midpoint(*edge)] for edge in pair))
                return values

            first_edge, second_edge = min(opposite_edges, key=diagonal_key)
            if source_ids[midpoint(*second_edge)] < source_ids[midpoint(*first_edge)]:
                first_edge, second_edge = second_edge, first_edge
            first_edge = tuple(sorted(first_edge, key=lambda item: source_ids[item]))
            second_edge = tuple(sorted(second_edge, key=lambda item: source_ids[item]))
            first, second = first_edge
            third, fourth = second_edge
            diagonal_first = midpoint(first, second)
            diagonal_second = midpoint(third, fourth)
            cycle = (
                midpoint(first, third),
                midpoint(first, fourth),
                midpoint(second, fourth),
                midpoint(second, third),
            )
            central_children = [
                (
                    diagonal_first,
                    cycle[index],
                    cycle[(index + 1) % len(cycle)],
                    diagonal_second,
                )
                for index in range(len(cycle))
            ]
            candidates = tuple(corner_children + central_children)
        children = tuple(
            _orient_child_like_parent(child, parent_determinant, coordinates)
            for child in candidates
        )
        key = tuple(sorted(parent))
        if key in children_by_cell:
            raise CompileError("duplicate_cell", "uniform refinement parent")
        children_by_cell[key] = children
        refined_cells.extend(children)
        refined_hints.extend([int(raw.partition_hints[cell_index])] * len(children))

    refined_supports: list[RawSupport] = []
    for support in raw.supports:
        entities: list[tuple[int, ...]] = []
        if support.dimension == 0:
            supported_vertices = {int(entity[0]) for entity in support.entities}
            supported_vertices.update(
                midpoint_by_edge[edge]
                for edge in midpoint_by_edge
                if edge[0] in supported_vertices and edge[1] in supported_vertices
            )
            entities = [(vertex,) for vertex in supported_vertices]
        elif support.dimension == 1:
            for entity in support.entities:
                first, second = (int(item) for item in entity)
                center = midpoint(first, second)
                entities.extend(((first, center), (center, second)))
        elif support.dimension == 2 and dimension == 3:
            for entity in support.entities:
                entities.extend(triangle_children(tuple(int(item) for item in entity)))
        elif support.dimension == dimension:
            for entity in support.entities:
                key = tuple(sorted(int(item) for item in entity))
                children = children_by_cell.get(key)
                if children is None:
                    raise CompileError("support_entity_not_in_mesh", support.name)
                entities.extend(children)
        else:
            raise CompileError(
                "uniform_refinement_support_dimension",
                f"{support.name}:{support.dimension}",
            )
        refined_supports.append(
            RawSupport(
                name=support.name,
                kind=support.kind,
                dimension=support.dimension,
                entities=tuple(
                    sorted(
                        {tuple(int(item) for item in entity) for entity in entities},
                        key=lambda entity: tuple(source_ids[item] for item in entity),
                    )
                ),
            )
        )

    total_levels = raw.uniform_refinement_levels + 1
    importer_base = raw.importer_id.split("+uniform-refinement-l", 1)[0]
    return RawMesh(
        dimension=dimension,
        cell_type=raw.cell_type,
        coordinate_order=1,
        source_node_ids=_little_u64(source_ids),
        coordinate_nodes=_little_f64(coordinates),
        cells=_little_u64(refined_cells),
        coordinate_connectivity=_little_u64(refined_cells),
        supports=tuple(refined_supports),
        partition_hints=np.asarray(refined_hints, dtype=np.int64),
        importer_id=f"{importer_base}+uniform-refinement-l{total_levels}",
        source_sha256=raw.source_sha256,
        local_barycentric_functionals=_little_u64(
            _barycentric_functionals(dimension, 1)
        ),
        uniform_refinement_levels=total_levels,
    )


def uniformly_refine_raw_mesh(raw: RawMesh, *, levels: int = 1) -> RawMesh:
    """Uniformly split every straight simplex while preserving semantic supports."""
    if isinstance(levels, bool) or not isinstance(levels, int) or levels < 0:
        raise ValueError("uniform refinement levels must be a nonnegative integer")
    refined = raw
    for _level in range(levels):
        refined = _uniformly_refine_raw_mesh_once(refined)
    return refined


__all__ = [
    "uniformly_refine_raw_mesh",
]
