# SPDX-License-Identifier: BSD-2-Clause
"""Importer-neutral raw simplex meshes and import requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ._common import FloatArray, UIntArray


@dataclass(frozen=True)
class ImportRequest:
    """Requested dimension, geometry order, required supports and uniform refinement
    levels of a mesh import.
    """

    dimension: Literal[2, 3]
    coordinate_order: Literal[1, 2, 4]
    required_supports: tuple[str, ...]
    uniform_refinement_levels: int = 0

    def __post_init__(self) -> None:
        if self.dimension not in (2, 3):
            raise ValueError("dimension must be 2 or 3")
        if self.coordinate_order not in (1, 2, 4):
            raise ValueError("coordinate order must be P1, P2, or P4")
        if (
            isinstance(self.uniform_refinement_levels, bool)
            or not isinstance(self.uniform_refinement_levels, int)
            or self.uniform_refinement_levels < 0
        ):
            raise ValueError("uniform refinement levels must be a nonnegative integer")


@dataclass(frozen=True)
class RawSupport:
    """Support as imported: entities given by importer node numbers."""

    name: str
    kind: Literal["region", "boundary", "node_set"]
    dimension: int
    entities: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class RawMesh:
    """Importer-neutral simplex mesh before canonicalization."""

    dimension: Literal[2, 3]
    cell_type: Literal["triangle", "tetrahedron"]
    coordinate_order: Literal[1, 2, 4]
    source_node_ids: UIntArray
    coordinate_nodes: FloatArray
    cells: UIntArray
    coordinate_connectivity: UIntArray
    supports: tuple[RawSupport, ...]
    partition_hints: NDArray[np.int64]
    importer_id: str
    source_sha256: str = ""
    local_barycentric_functionals: UIntArray | None = None
    uniform_refinement_levels: int = 0


__all__ = [
    "ImportRequest",
    "RawMesh",
    "RawSupport",
]
