# SPDX-License-Identifier: BSD-2-Clause
"""Independent compact authoring fixtures for the Python contract tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any

from slope_protocol import content_id


VERSION = {"major": 1, "minor": 0, "patch": 0}
CREATED_BY = {
    "tool": "contract-fixture",
    "tool_version": "1",
    "source_id": "fixture-v1",
}
PROVENANCE = {
    "origin": "independent compact oracle",
    "license": "BSD-2-Clause",
}


def stamped(document: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with the canonical authoring content identity."""
    result = deepcopy(document)
    result.pop("id", None)
    result["id"] = content_id(result)
    return result


def envelope(kind: str, name: str) -> dict[str, Any]:
    return {
        "kind": f"slope_stability.{kind}",
        "schema_version": deepcopy(VERSION),
        "created_by": deepcopy(CREATED_BY),
        "provenance": deepcopy(PROVENANCE),
        "extensions": {},
        "identity": {"name": name, "title": name.replace(".", " ")},
    }


def asset_document(
    dimension: int = 2,
    *,
    coordinate_order: int = 1,
    importer: str = "compact",
) -> dict[str, Any]:
    cell = "triangle" if dimension == 2 else "tetrahedron"
    document = envelope("asset", f"asset.compact.{dimension}d.g{coordinate_order}")
    document.update(
        {
            "mesh_source": {
                "importer": importer,
                "path": f"meshes/compact-{dimension}d-g{coordinate_order}.json",
                "sha256": "1" * 64,
            },
            "dimension": dimension,
            "cell_type": cell,
            "coordinate_order": coordinate_order,
            "variants": [{"name": "base"}],
            "supports": [
                {"name": "domain", "kind": "region", "dimension": dimension},
                {
                    "name": "boundary",
                    "kind": "boundary",
                    "dimension": dimension - 1,
                },
                {"name": "base", "kind": "boundary", "dimension": dimension - 1},
                {"name": "top", "kind": "boundary", "dimension": dimension - 1},
            ],
        }
    )
    return stamped(document)


def case_document(
    dimension: int = 2,
    *,
    field_order: int = 1,
    coordinate_order: int = 1,
    analysis: str = "limit_load.work_controlled",
) -> dict[str, Any]:
    document = envelope(
        "case", f"case.compact.{dimension}d.p{field_order}.g{coordinate_order}"
    )
    components = ["x", "y"] if dimension == 2 else ["x", "y", "z"]
    document.update(
        {
            "asset": {
                "name": f"asset.compact.{dimension}d.g{coordinate_order}",
                "variant": "base",
            },
            "discretization": {
                "field_order": field_order,
                "coordinate_order": coordinate_order,
            },
            "materials": [
                {
                    "name": "soil",
                    "support": "domain",
                    "model": "mohr_coulomb",
                    "parameters": {
                        "cohesion": {"value": 6.0, "unit": "kPa"},
                        "friction_angle": {"value": 30.0, "unit": "degree"},
                        "dilation_angle": {"value": 5.0, "unit": "degree"},
                        "young_modulus": {"value": 40000.0, "unit": "kPa"},
                        "poisson_ratio": {"value": 0.3, "unit": "1"},
                        "unit_weight": {"value": 20.0, "unit": "kN/m^3"},
                        "davis": "B",
                    },
                }
            ],
            "physics": {
                "mechanics": True,
                "seepage": False,
                "coupling": "none",
            },
            "boundary_rules": [
                {
                    "name": "fixed-base",
                    "field": "displacement",
                    "condition": "essential",
                    "support": "base",
                    "components": components,
                    "value_model": "constant_vector",
                    "parameters": {
                        "value": [0.0] * dimension,
                        "unit": "m",
                    },
                }
            ],
            "analysis": {"selector": analysis},
            "profiles": {
                "continuation": "continuation.default",
                "nonlinear": "nonlinear.default",
                "linear": "linear.default",
                "pmg": "pmg.default",
                "deflation": "deflation.off",
                "output": "output.default",
                "validation": "validation.default",
            },
            "case_overrides": {},
            "required_outputs": ["displacement", "stress"],
        }
    )
    return stamped(document)


def profile_document(
    family: str,
    policy: dict[str, Any],
    *,
    name: str | None = None,
) -> dict[str, Any]:
    document = envelope("profile", name or f"{family}.default")
    document.update({"family": family, "policy": deepcopy(policy)})
    return stamped(document)


def profile_documents() -> list[dict[str, Any]]:
    return [
        profile_document(
            "continuation",
            {
                "method": "work_controlled",
                "initial_factor": 1.0,
                "initial_step": 0.1,
                "minimum_step": 1.0e-4,
                "maximum_step": 0.2,
                "maximum_accepted_states": 100,
                "maximum_work": 1.0e300,
                "initial_work_scale": 0.2,
                "scaled_slope_tolerance": 1.0e-3,
                "factor_increment_stop": 1.0e-3,
                "initial_nonlinear_stopping_criterion": "relative_correction",
                "initial_nonlinear_stopping_tolerance": 1.0e-3,
            },
        ),
        profile_document(
            "nonlinear",
            {
                "method": "newton_line_search",
                "relative_tolerance": 1.0e-8,
                "absolute_tolerance": 1.0e-12,
                "stopping_criterion": "relative_residual",
                "stopping_tolerance": 1.0e-8,
                "regularization_minimum": 1.0e-6,
                "maximum_iterations": 30,
            },
        ),
        profile_document(
            "linear",
            {
                "ksp_type": "fgmres",
                "operator_class": "nonsymmetric",
                "relative_tolerance": 1.0e-8,
                "absolute_tolerance": 1.0e-12,
                "maximum_iterations": 500,
                "restart": 40,
                "expert_options": {},
            },
        ),
        profile_document(
            "pmg",
            {
                "enabled": True,
                "smoother": "chebyshev",
                "coarse_solver": "preonly_lu",
                "coarse_direct_maximum_dofs": 50000,
                "active_rank_policy": "all",
            },
        ),
        profile_document(
            "deflation",
            {
                "enabled": False,
                "projector": "general_left_right",
                "maximum_dimension": 0,
            },
            name="deflation.off",
        ),
        profile_document(
            "output",
            {
                "quantities": ["displacement", "stress"],
                "history": True,
                "frequency": "accepted_state",
            },
        ),
        profile_document(
            "validation",
            {
                "qualification_mode": True,
                "reject_unused_options": True,
                "conservation_physical_scale": 1.0,
                "conservation_normalized_tolerance": 1.0e-8,
            },
        ),
    ]


def suite_document() -> dict[str, Any]:
    document = envelope("suite", "suite.compact")
    document.update(
        {
            "cases": ["case.compact.2d.p1.g1"],
            "profiles": [item["identity"]["name"] for item in profile_documents()],
            "axes": [
                {
                    "name": "linear-rtol",
                    "path": "linear.relative_tolerance",
                    "values": [1.0e-8, 1.0e-9],
                }
            ],
            "resources": [
                {
                    "name": "local-4x8",
                    "nodes": 1,
                    "world_size": 4,
                    "ranks_per_node": 4,
                    "openmp_threads": 8,
                    "cores_per_rank": 8,
                    "smt": "off",
                    "cpu_binding": "core",
                    "memory_binding": "local",
                    "numa_policy": "local",
                    "mpi_thread_level": "funneled",
                    "openmp": {
                        "dynamic": False,
                        "max_active_levels": 1,
                        "proc_bind": "close",
                        "places": "cores",
                    },
                    "third_party_threads": {
                        "blas": 1,
                        "hypre": 1,
                        "petsc": 1,
                    },
                    "active_ranks": {"fine": [0, 1, 2, 3], "coarse": [0, 2, 3]},
                }
            ],
            "collection": {"repetitions": 1, "warmups": 0},
        }
    )
    return stamped(document)


def source_tree_files(root: Path) -> list[Path]:
    return sorted(
        path
        for top in ("src", "include", "python/src")
        for path in (root / top).rglob("*")
        if path.is_file()
    )


def assert_schema_subset(
    value: Any,
    schema: dict[str, Any],
    *,
    pointer: str = "",
) -> None:
    """Validate the JSON-Schema vocabulary emitted by the v1 generator."""
    if "const" in schema and value != schema["const"]:
        raise AssertionError(f"{pointer}: {value!r} != {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise AssertionError(f"{pointer}: {value!r} is not in {schema['enum']!r}")
    expected_type = schema.get("type")
    predicates = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: (
            isinstance(item, (int, float)) and not isinstance(item, bool)
        ),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    if expected_type is not None and not predicates[expected_type](value):
        raise AssertionError(
            f"{pointer}: expected {expected_type}, got {type(value).__name__}"
        )
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise AssertionError(f"{pointer}: string is too short")
        pattern = schema.get("pattern")
        if pattern is not None and re.search(pattern, value) is None:
            raise AssertionError(f"{pointer}: string does not match {pattern!r}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise AssertionError(f"{pointer}: array is too short")
        if "items" in schema:
            for index, item in enumerate(value):
                assert_schema_subset(
                    item,
                    schema["items"],
                    pointer=f"{pointer}/{index}",
                )
    if isinstance(value, dict):
        required = set(schema.get("required", []))
        missing = required.difference(value)
        if missing:
            raise AssertionError(f"{pointer}: missing {sorted(missing)!r}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = set(value).difference(properties)
            if extra:
                raise AssertionError(f"{pointer}: extra {sorted(extra)!r}")
        property_names = schema.get("propertyNames")
        for key, item in value.items():
            child = f"{pointer}/{key}"
            if property_names is not None:
                assert_schema_subset(key, property_names, pointer=child)
            if key in properties:
                assert_schema_subset(item, properties[key], pointer=child)
