# SPDX-License-Identifier: BSD-2-Clause
"""Shared compiler error, array aliases and deterministic byte encodings."""

from __future__ import annotations

from hashlib import sha256
from itertools import product
from typing import Any

import numpy as np
from numpy.typing import NDArray

from slope_protocol import canonicalize


UIntArray = NDArray[np.uint64]


FloatArray = NDArray[np.float64]


class CompileError(ValueError):
    """Stable compilation failure with an optional stable entity ID."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        entity_id: str | None = None,
    ) -> None:
        self.code = code
        self.entity_id = entity_id
        super().__init__(f"{code}: {detail}")


def _stable_u64(*parts: Any) -> int:
    encoded = canonicalize(list(parts))
    return int.from_bytes(sha256(encoded).digest()[:8], "big", signed=False)


def _hex_id(value: int) -> str:
    return f"{value:016x}"


def _little_u64(value: Any) -> UIntArray:
    return np.ascontiguousarray(value, dtype="<u8")


def _little_f64(value: Any) -> FloatArray:
    return np.ascontiguousarray(value, dtype="<f8")


def _simplex_determinant(
    vertices: tuple[int, ...], coordinates: list[NDArray[np.float64]]
) -> float:
    origin = coordinates[vertices[0]]
    jacobian = np.stack(
        [coordinates[vertex] - origin for vertex in vertices[1:]], axis=1
    )
    return float(np.linalg.det(jacobian))


def _barycentric_functionals(dimension: int, order: int) -> list[tuple[int, ...]]:
    vertices = [
        tuple(order if index == vertex else 0 for index in range(dimension + 1))
        for vertex in range(dimension + 1)
    ]
    if order == 1:
        return vertices
    all_items = [
        item
        for item in product(range(order + 1), repeat=dimension + 1)
        if sum(item) == order and item not in vertices
    ]
    all_items.sort(reverse=True)
    preferred = (0, order - 1, 1) + (0,) * (dimension - 2)
    if preferred in all_items:
        all_items.remove(preferred)
        all_items.append(preferred)
    return vertices + all_items


__all__ = [
    "CompileError",
    "FloatArray",
    "UIntArray",
]
