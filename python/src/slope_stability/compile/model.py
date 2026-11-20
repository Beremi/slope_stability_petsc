# SPDX-License-Identifier: BSD-2-Clause
"""Compilation of validated authoring documents into a problem model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..authoring import (
    Asset,
    Case,
    Quantity,
    RegistryError,
    UnitSystem,
    _boundary_rule_semantic_issues,
    default_model_registry,
)
from slope_protocol import canonicalize
from ._common import CompileError, _hex_id
from .canonical import CanonicalMesh
from .geometry import GeometryAudit, audit_geometry


@dataclass(frozen=True)
class CompiledProblemModel:
    """Validated case bound to a canonical mesh in internal SI units."""

    asset_id: str
    case_id: str
    mesh: CanonicalMesh
    spaces: Mapping[str, Mapping[str, Any]]
    geometry_audit: GeometryAudit
    materials: tuple[Mapping[str, Any], ...]
    boundary_rules: tuple[Mapping[str, Any], ...]
    counts: Mapping[str, int]
    unit_conversions: tuple[Mapping[str, Any], ...]
    prescribed_saturation: Mapping[str, Any] | None = None

    def logical_payload(self) -> dict[str, Any]:
        payload = {
            "asset_id": self.asset_id,
            "case_id": self.case_id,
            "mesh": self.mesh._logical_payload(),
            "spaces": self.spaces,
            "geometry_audit": {
                "minimum_determinant": self.geometry_audit.minimum_determinant,
                "maximum_determinant": self.geometry_audit.maximum_determinant,
                "maximum_condition": self.geometry_audit.maximum_condition,
                "sign": self.geometry_audit.sign,
                "audit_points": self.geometry_audit.audit_points,
            },
            "materials": list(self.materials),
            "boundary_rules": list(self.boundary_rules),
            "counts": dict(self.counts),
            "unit_conversions": list(self.unit_conversions),
        }
        if self.prescribed_saturation is not None:
            payload["prescribed_saturation"] = dict(self.prescribed_saturation)
        return payload


def _validate_shared_coordinates(mesh: CanonicalMesh) -> None:
    if mesh.coordinate_functional_ids.shape != (mesh.coordinate_nodes.shape[0],):
        raise CompileError(
            "shared_coordinate_mismatch",
            "coordinate functional identity is incomplete",
        )
    observed: dict[int, tuple[float, ...]] = {}
    for row in mesh.coordinate_connectivity:
        for node_index in row:
            index = int(node_index)
            functional = int(mesh.coordinate_functional_ids[index])
            coordinate = tuple(float(item) for item in mesh.coordinate_nodes[index])
            prior = observed.setdefault(functional, coordinate)
            if prior != coordinate:
                raise CompileError(
                    "shared_coordinate_mismatch",
                    f"functional {functional:016x} has inconsistent coordinates",
                )


def _quantity_conversion(
    units: UnitSystem,
    parameters: Mapping[str, Any],
    name: str,
    dimension: str,
) -> dict[str, Any] | None:
    value = parameters.get(name)
    if not isinstance(value, Mapping) or "value" not in value or "unit" not in value:
        return None
    conversion = units.to_internal(
        Quantity.model_validate_json(canonicalize(dict(value))),
        expected_dimension=dimension,
        compilation_boundary=True,
    )
    return {
        "parameter": name,
        **conversion.model_dump(mode="json"),
    }


def _unit_scale_conversion(
    units: UnitSystem,
    *,
    unit: str,
    dimension: str,
    owner: str,
    parameter: str,
) -> dict[str, Any]:
    conversion = units.to_internal(
        Quantity(value=1.0, unit=unit),
        expected_dimension=dimension,
        compilation_boundary=True,
    )
    return {
        "owner": owner,
        "parameter": parameter,
        **conversion.model_dump(mode="json"),
    }


def _essential_value_signature(
    rule: Any,
    component: str,
    dimension: int,
    *,
    coordinate: tuple[float, ...] | None,
    units: UnitSystem,
) -> bytes:
    parameters = rule.parameters
    if rule.value_model == "constant_vector":
        values = parameters.get("value")
        if isinstance(values, (list, tuple)):
            components = (
                tuple(("x", "y", "z")[:dimension])
                if "all" in rule.components
                else tuple(rule.components)
            )
            index: int | None = None
            if len(values) == len(components) and component in components:
                index = components.index(component)
            elif len(values) == dimension and component in ("x", "y", "z"):
                index = ("x", "y", "z").index(component)
            if index is not None:
                converted = units.to_internal(
                    Quantity(
                        value=float(values[index]),
                        unit=str(parameters.get("unit", "")),
                    ),
                    expected_dimension="length",
                    compilation_boundary=True,
                )
                return canonicalize(
                    {
                        "dimension": "length",
                        "value": converted.converted_value,
                    }
                )
    if rule.value_model == "constant_scalar":
        converted = units.to_internal(
            Quantity(
                value=float(parameters.get("value")),
                unit=str(parameters.get("unit", "")),
            ),
            expected_dimension="pore_pressure",
            compilation_boundary=True,
        )
        value = converted.converted_value
        if abs(value) <= 1.0e-12:
            value = 0.0
        return canonicalize({"dimension": "pore_pressure", "value": value})
    if coordinate is not None and rule.value_model == "affine_coordinate":
        gradient = tuple(float(item) for item in parameters.get("gradient", ()))
        raw_value = float(parameters.get("offset")) + sum(
            gradient[axis] * coordinate[axis] for axis in range(dimension)
        )
        value_dimension = "length" if rule.field == "displacement" else "pore_pressure"
        converted = units.to_internal(
            Quantity(
                value=raw_value,
                unit=str(parameters.get("unit", "")),
            ),
            expected_dimension=value_dimension,
            compilation_boundary=True,
        )
        value = converted.converted_value
        if abs(value) <= 1.0e-12:
            value = 0.0
        return canonicalize({"dimension": value_dimension, "value": value})
    if coordinate is not None and rule.value_model == "hydrostatic":
        length_unit = str(parameters.get("length_unit", ""))
        pressure_unit = str(parameters.get("pressure_unit", ""))
        level = units.to_internal(
            Quantity(
                value=float(parameters.get("reference_level")),
                unit=length_unit,
            ),
            expected_dimension="length",
            compilation_boundary=True,
        ).converted_value
        pressure = units.to_internal(
            Quantity(
                value=float(parameters.get("reference_pressure")),
                unit=pressure_unit,
            ),
            expected_dimension="pore_pressure",
            compilation_boundary=True,
        ).converted_value
        unit_weight = units.to_internal(
            Quantity(
                value=float(parameters.get("unit_weight")),
                unit=f"{pressure_unit}/{length_unit}",
            ),
            expected_dimension="unit_weight",
            compilation_boundary=True,
        ).converted_value
        head = max(level - coordinate[1], 0.0)
        value = pressure + unit_weight * head
        tolerance = 1.0e-12 * max(1.0, abs(pressure), abs(unit_weight * level))
        if abs(value) <= tolerance:
            value = 0.0
        return canonicalize({"dimension": "pore_pressure", "value": value})
    if coordinate is not None and rule.value_model == "piecewise_linear_level":
        coordinate_unit = str(parameters.get("coordinate_unit", ""))
        value_unit = str(parameters.get("value_unit", ""))
        positions = [
            units.to_internal(
                Quantity(value=float(value), unit=coordinate_unit),
                expected_dimension="length",
                compilation_boundary=True,
            ).converted_value
            for value in parameters.get("coordinates", ())
        ]
        levels = [
            units.to_internal(
                Quantity(value=float(value), unit=value_unit),
                expected_dimension="length",
                compilation_boundary=True,
            ).converted_value
            for value in parameters.get("values", ())
        ]
        level = float(np.interp(coordinate[0], positions, levels))
        head = max(level - coordinate[1], 0.0)
        tolerance = 1.0e-12 * max(1.0, abs(level))
        if head <= tolerance:
            return canonicalize({"dimension": "pore_pressure", "value": 0.0})
    return canonicalize(
        {
            "value_model": rule.value_model,
            "parameters": parameters,
        }
    )


def compile_problem_model(
    *,
    asset: Asset,
    case: Case,
    mesh: CanonicalMesh,
) -> CompiledProblemModel:
    """Resolve semantic supports and produce the canonical serial model."""
    if asset.dimension != mesh.dimension or case.discretization.coordinate_order != (
        mesh.coordinate_order
    ):
        raise CompileError("mesh_case_incompatible", case.identity.name)
    if (
        asset.cell_type != mesh.cell_type
        or asset.coordinate_order != mesh.coordinate_order
    ):
        raise CompileError("asset_mesh_incompatible", asset.identity.name)
    if case.asset.name != asset.identity.name:
        raise CompileError("asset_reference_mismatch", case.asset.name)
    if case.asset.variant not in {variant.name for variant in asset.variants}:
        raise CompileError("asset_variant_missing", case.asset.variant)
    if (
        case.analysis.selector == "seepage.steady_unconfined"
        and not case.physics.seepage
    ) or (
        case.analysis.selector != "seepage.steady_unconfined"
        and not case.physics.mechanics
    ):
        raise CompileError("analysis_physics_incompatible", case.analysis.selector)
    if case.physics.coupling != "none" and not (
        case.physics.mechanics and case.physics.seepage
    ):
        raise CompileError(
            "coupling_without_mechanics_and_seepage",
            case.physics.coupling,
        )
    _validate_shared_coordinates(mesh)
    geometry = audit_geometry(mesh)
    declared_supports = {support.name: support for support in asset.supports}
    for support in declared_supports.values():
        mesh_support = mesh.supports.get(support.name)
        if mesh_support is None:
            raise CompileError("missing_support", support.name)
        if mesh_support.dimension != support.dimension:
            raise CompileError("support_dimension_mismatch", support.name)

    cell_ids = set(int(item) for item in mesh.stable_entity_ids[mesh.dimension])
    assignments: dict[int, dict[str, list[str]]] = {
        item: {"mechanics": [], "seepage": []} for item in cell_ids
    }
    materials: list[dict[str, Any]] = []
    unit_conversions: list[dict[str, Any]] = []
    units = UnitSystem()
    registry = default_model_registry()
    material_dimensions = {
        "cohesion": "stress",
        "friction_angle": "angle",
        "dilation_angle": "angle",
        "young_modulus": "stress",
        "poisson_ratio": "dimensionless",
        "unit_weight": "unit_weight",
        "saturated_unit_weight": "unit_weight",
        "unsaturated_unit_weight": "unit_weight",
        "conductivity": "conductivity",
        "water_unit_weight": "unit_weight",
    }
    for material in case.materials:
        try:
            registry.adapt(material.model, material.parameters)
        except RegistryError as error:
            raise CompileError("invalid_material_parameters", str(error)) from error
        support = mesh.supports.get(material.support)
        if support is None:
            raise CompileError("missing_support", material.support)
        if support.dimension != mesh.dimension:
            raise CompileError("material_support_dimension", material.support)
        if material.model == "mohr_coulomb":
            family = "mechanics"
        elif material.model == "steady_unconfined_darcy":
            family = "seepage"
        else:
            raise CompileError("material_model_family", material.model)
        required = (family == "mechanics" and case.physics.mechanics) or (
            family == "seepage" and case.physics.seepage
        )
        if not required:
            raise CompileError("material_physics_incompatible", material.name)
        for entity_id in support.entity_ids:
            if entity_id in assignments:
                assignments[entity_id][family].append(material.name)
        parameters = json.loads(canonicalize(material.parameters))
        materials.append(
            {
                "name": material.name,
                "support": material.support,
                "model": material.model,
                "entity_ids": [_hex_id(item) for item in support.entity_ids],
                "parameters": parameters,
            }
        )
        for name, dimension in material_dimensions.items():
            converted = _quantity_conversion(
                units, material.parameters, name, dimension
            )
            if converted is not None:
                unit_conversions.append(
                    {"owner": f"material:{material.name}", **converted}
                )
    required_families = []
    if case.physics.mechanics:
        required_families.append("mechanics")
    if case.physics.seepage:
        required_families.append("seepage")
    for entity_id, families in assignments.items():
        for family in required_families:
            owners = families[family]
            if not owners:
                raise CompileError(
                    "missing_material_assignment",
                    f"{family}/{entity_id:016x}",
                )
            if len(owners) > 1:
                raise CompileError(
                    "ambiguous_material_assignment",
                    f"{family}/{entity_id:016x}: {owners}",
                )

    boundary_rules: list[dict[str, Any]] = []
    constrained_dofs = 0
    essential_values: dict[tuple[str, str, int], tuple[bytes, str]] = {}
    vertex_coordinates = {
        int(entity_id): tuple(float(item) for item in coordinate)
        for entity_id, coordinate in zip(
            mesh.topology_vertex_ids,
            mesh.topology_vertex_coordinates,
            strict=True,
        )
    }
    for rule in case.boundary_rules:
        try:
            registry.adapt(rule.value_model, rule.parameters)
        except RegistryError as error:
            raise CompileError("invalid_value_parameters", str(error)) from error
        support = mesh.supports.get(rule.support)
        if support is None:
            raise CompileError("missing_support", rule.support)
        facet_support = (
            support.kind == "boundary" and support.dimension == mesh.dimension - 1
        )
        node_support = (
            rule.condition == "essential"
            and support.kind == "node_set"
            and support.dimension < mesh.dimension
        )
        if not (facet_support or node_support):
            raise CompileError("boundary_support_dimension", rule.support)
        semantic_issues = _boundary_rule_semantic_issues(
            rule,
            dimension=mesh.dimension,
            physics=case.physics,
        )
        if semantic_issues:
            semantic = semantic_issues[0]
            raise CompileError(
                semantic.code,
                f"{rule.name}: {semantic.message}",
            )
        component_count = (
            mesh.dimension if "all" in rule.components else max(1, len(rule.components))
        )
        if rule.condition == "essential":
            if rule.field == "pore_pressure":
                components = ("pressure",)
            elif "all" in rule.components:
                components = tuple(("x", "y", "z")[: mesh.dimension])
            else:
                components = rule.components
            for entity_id in support.entity_ids:
                for component in components:
                    signature = _essential_value_signature(
                        rule,
                        component,
                        mesh.dimension,
                        coordinate=vertex_coordinates.get(entity_id),
                        units=units,
                    )
                    key = (rule.field, component, entity_id)
                    prior = essential_values.setdefault(
                        key,
                        (signature, rule.name),
                    )
                    if prior[0] != signature:
                        raise CompileError(
                            "conflicting_essential_boundary",
                            (
                                f"{prior[1]} conflicts with {rule.name}: "
                                f"{component}/{entity_id:016x}"
                            ),
                        )
            constrained_dofs += (
                len(support.entity_ids)
                * component_count
                * case.discretization.field_order
            )
        boundary_rules.append(
            {
                "name": rule.name,
                "field": rule.field,
                "condition": rule.condition,
                "support": rule.support,
                "components": list(rule.components),
                "value_model": rule.value_model,
                "parameters": json.loads(canonicalize(rule.parameters)),
                "geometry_patch": rule.geometry_patch,
                "entity_ids": [_hex_id(item) for item in support.entity_ids],
            }
        )
        owner = f"boundary:{rule.name}"
        unit = rule.parameters.get("unit")
        if isinstance(unit, str):
            if rule.value_model in {
                "global_traction",
                "spatial_global_traction",
                "normal_pressure",
            }:
                dimension = "stress"
            elif rule.value_model == "prescribed_flux":
                dimension = "conductivity"
            elif rule.field == "displacement":
                dimension = "length"
            else:
                dimension = "pore_pressure"
            unit_conversions.append(
                _unit_scale_conversion(
                    units,
                    unit=unit,
                    dimension=dimension,
                    owner=owner,
                    parameter="value",
                )
            )
        if rule.value_model == "hydrostatic":
            length_unit = rule.parameters.get("length_unit")
            pressure_unit = rule.parameters.get("pressure_unit")
            if isinstance(length_unit, str) and isinstance(pressure_unit, str):
                for parameter, unit_name, dimension in (
                    ("reference_level", length_unit, "length"),
                    ("reference_pressure", pressure_unit, "pore_pressure"),
                    (
                        "unit_weight",
                        f"{pressure_unit}/{length_unit}",
                        "unit_weight",
                    ),
                ):
                    unit_conversions.append(
                        _unit_scale_conversion(
                            units,
                            unit=unit_name,
                            dimension=dimension,
                            owner=owner,
                            parameter=parameter,
                        )
                    )

    prescribed_saturation = None
    if case.prescribed_saturation is not None:
        profile = case.prescribed_saturation
        axis = {"x": 0, "z": 2}[profile.axis]
        if axis >= mesh.dimension:
            raise CompileError("saturation_axis_dimension", profile.axis)
        prescribed_saturation = {"axis": axis, "length_unit": "m"}
        for parameter in ("coordinates", "levels"):
            converted = []
            for index, value in enumerate(getattr(profile, parameter)):
                conversion = units.to_internal(
                    Quantity(value=value, unit=profile.length_unit),
                    expected_dimension="length",
                    compilation_boundary=True,
                )
                converted.append(conversion.converted_value)
                unit_conversions.append(
                    {
                        "owner": "prescribed_saturation",
                        "parameter": f"{parameter}[{index}]",
                        **conversion.model_dump(mode="json"),
                    }
                )
            prescribed_saturation[parameter] = converted
        coordinates = prescribed_saturation["coordinates"]
        if any(right <= left for left, right in zip(coordinates, coordinates[1:])):
            raise CompileError(
                "saturation_coordinate_conversion",
                "converted knots must remain ordered",
            )

    spaces = {
        "displacement": {
            "family": "simplex_lagrange",
            "order": case.discretization.field_order,
            "components": mesh.dimension,
        },
        "coordinates": {
            "family": "simplex_lagrange",
            "order": case.discretization.coordinate_order,
            "components": mesh.dimension,
        },
    }
    if case.physics.seepage:
        spaces["pore_pressure"] = {
            "family": "simplex_lagrange",
            "order": case.discretization.field_order,
            "components": 1,
        }
    counts = {
        "cells": mesh.cells.shape[0],
        "coordinate_nodes": mesh.coordinate_nodes.shape[0],
        "material_cells": len(assignments),
        "boundary_rules": len(boundary_rules),
        "constrained_dofs": constrained_dofs,
    }
    return CompiledProblemModel(
        asset_id=asset.id,
        case_id=case.id,
        mesh=mesh,
        spaces=spaces,
        geometry_audit=geometry,
        materials=tuple(materials),
        boundary_rules=tuple(boundary_rules),
        counts=counts,
        unit_conversions=tuple(unit_conversions),
        prescribed_saturation=prescribed_saturation,
    )


__all__ = [
    "CompiledProblemModel",
    "compile_problem_model",
]
