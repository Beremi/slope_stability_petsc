# SPDX-License-Identifier: BSD-2-Clause
"""Signed-Jacobian geometry audit of canonical simplex meshes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ._common import (
    CompileError,
    FloatArray,
    UIntArray,
    _barycentric_functionals,
    _little_f64,
)
from .canonical import CanonicalMesh


@dataclass(frozen=True)
class GeometryAudit:
    """Signed-Jacobian summary of a mesh (all determinants positive)."""

    minimum_determinant: float
    maximum_determinant: float
    maximum_condition: float
    sign: Literal["positive"]
    audit_points: int


def _entity_hex(mesh: CanonicalMesh, cell_index: int = 0) -> str:
    values = mesh.stable_entity_ids[mesh.dimension]
    value = int(values[min(cell_index, values.size - 1)])
    return f"{value:016x}"


def _lagrange_reference_gradients(
    functionals: UIntArray,
    barycentric_point: NDArray[np.float64],
    order: int,
) -> FloatArray:
    """Evaluate complete simplex nodal-basis gradients in reference axes."""
    derivative_lambda = np.zeros(
        (functionals.shape[0], functionals.shape[1]), dtype=np.float64
    )
    for basis_index, alpha_values in enumerate(functionals):
        alpha = [int(item) for item in alpha_values]
        factors: list[list[float]] = []
        for component, count in enumerate(alpha):
            factors.append(
                [
                    (order * barycentric_point[component] - root) / (count - root)
                    for root in range(count)
                ]
            )
        for component, count in enumerate(alpha):
            derivative = 0.0
            for omitted in range(count):
                term = order / (count - omitted)
                for other_component, other_factors in enumerate(factors):
                    for factor_index, factor in enumerate(other_factors):
                        if other_component == component and factor_index == omitted:
                            continue
                        term *= factor
                derivative += term
            derivative_lambda[basis_index, component] = derivative
    return _little_f64(derivative_lambda[:, 1:] - derivative_lambda[:, [0]])


def audit_geometry(mesh: CanonicalMesh) -> GeometryAudit:
    """Check the sign of the geometry Jacobian at audit points of every cell.

    The Lagrange geometry map is evaluated cell by cell; the maximum condition number
    is recorded.

    Raises: CompileError: A Jacobian changes sign or is below the singularity
    threshold.
    """

    coordinates = mesh.coordinate_nodes
    entity_id = _entity_hex(mesh)
    if not np.isfinite(coordinates).all():
        raise CompileError(
            "nonfinite_geometry",
            "coordinate contains NaN or infinity",
            entity_id=entity_id,
        )
    if np.unique(coordinates, axis=0).shape[0] != coordinates.shape[0]:
        raise CompileError(
            "duplicate_coordinate", "duplicate coordinate node", entity_id=entity_id
        )
    vertices = coordinates[mesh.cells[:, : mesh.dimension + 1]]
    affine_jacobians = np.swapaxes(vertices[:, 1:] - vertices[:, [0]], 1, 2)
    affine_determinants = np.linalg.det(affine_jacobians)
    inverted = np.flatnonzero(affine_determinants < 0.0)
    if inverted.size:
        raise CompileError(
            "inverted_geometry",
            "negative simplex Jacobian",
            entity_id=_entity_hex(mesh, int(inverted[0])),
        )
    singular = np.flatnonzero(affine_determinants < 1.0e-12)
    if singular.size:
        raise CompileError(
            "near_singular_geometry",
            "simplex Jacobian is below the audit threshold",
            entity_id=_entity_hex(mesh, int(singular[0])),
        )
    affine_conditions = np.linalg.cond(affine_jacobians)
    if mesh.coordinate_order == 1:
        return GeometryAudit(
            minimum_determinant=float(np.min(affine_determinants)),
            maximum_determinant=float(np.max(affine_determinants)),
            maximum_condition=float(np.max(affine_conditions)),
            sign="positive",
            audit_points=int(affine_determinants.size),
        )

    audit_order = 12
    audit_functionals = _barycentric_functionals(mesh.dimension, audit_order)
    audit_points = [
        np.asarray(item, dtype=np.float64) / audit_order for item in audit_functionals
    ]
    gradients = np.asarray(
        [
            _lagrange_reference_gradients(
                mesh.coordinate_barycentric_functionals,
                barycentric,
                mesh.coordinate_order,
            )
            for barycentric in audit_points
        ],
        dtype=np.float64,
    )
    minimum = float("inf")
    maximum = float("-inf")
    maximum_condition = float(np.max(affine_conditions))
    chunk_size = 128
    cell_count = mesh.coordinate_connectivity.shape[0]
    expected_nodes = mesh.coordinate_barycentric_functionals.shape[0]
    for start in range(0, cell_count, chunk_size):
        stop = min(start + chunk_size, cell_count)
        connectivity = mesh.coordinate_connectivity[start:stop]
        if connectivity.shape[1] != expected_nodes:
            raise CompileError(
                "coordinate_closure_mismatch",
                str(connectivity.shape),
                entity_id=_entity_hex(mesh, start),
            )
        local_coordinates = coordinates[connectivity]
        jacobians = np.einsum(
            "cni,pnj->cpij",
            local_coordinates,
            gradients,
            optimize=True,
        )
        determinants = np.linalg.det(jacobians)
        negative = np.argwhere(determinants < -1.0e-12)
        if negative.size:
            cell_offset = int(negative[0, 0])
            raise CompileError(
                "sign_changing_geometry",
                "curved coordinate Jacobian changes sign",
                entity_id=_entity_hex(mesh, start + cell_offset),
            )
        singular = np.argwhere(np.abs(determinants) <= 1.0e-12)
        if singular.size:
            cell_offset = int(singular[0, 0])
            raise CompileError(
                "near_singular_geometry",
                "curved coordinate Jacobian is below the audit threshold",
                entity_id=_entity_hex(mesh, start + cell_offset),
            )
        minimum = min(minimum, float(np.min(determinants)))
        maximum = max(maximum, float(np.max(determinants)))
        maximum_condition = max(
            maximum_condition,
            float(np.max(np.linalg.cond(jacobians))),
        )
    return GeometryAudit(
        minimum_determinant=minimum,
        maximum_determinant=maximum,
        maximum_condition=maximum_condition,
        sign="positive",
        audit_points=cell_count * len(audit_points),
    )


__all__ = [
    "GeometryAudit",
    "audit_geometry",
]
