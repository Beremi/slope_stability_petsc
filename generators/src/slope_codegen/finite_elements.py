# SPDX-License-Identifier: BSD-2-Clause
"""Generate immutable simplex descriptor, quadrature, and permutation tables."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from itertools import permutations
from pathlib import Path
from typing import Any, Callable


ORDERS = (1, 2, 4, 8)
TRI_EDGES = ((0, 1), (1, 2), (0, 2))
TET_EDGES = ((0, 1), (1, 2), (0, 2), (1, 3), (2, 3), (0, 3))
TET_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
EXTERNAL_TRI_EDGES = ((0, 1), (1, 2), (2, 0))
GMSH_TET_EDGES = ((0, 1), (1, 2), (2, 0), (3, 0), (3, 2), (3, 1))
GMSH_TET_FACES = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (3, 1, 2))
VTK_TET_EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
VTK_TET_FACES = ((0, 1, 3), (2, 3, 1), (0, 3, 2), (0, 2, 1))
ORDERINGS = (
    ("canonical", "SS_FE_ORDER_CANONICAL"),
    ("petsc", "SS_FE_ORDER_PETSC"),
    ("gmsh", "SS_FE_ORDER_GMSH"),
    ("vtk", "SS_FE_ORDER_VTK"),
    ("reference", "SS_FE_ORDER_REFERENCE"),
)


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _triangle_interior(order: int) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    for a in range(order - 2, 0, -1):
        for b in range(order - a - 1, 0, -1):
            c = order - a - b
            if c > 0:
                result.append((a, b, c))
    return result


def _nodes(
    dimension: int,
    order: int,
    *,
    edges: tuple[tuple[int, int], ...] | None = None,
    faces: tuple[tuple[int, int, int], ...] | None = None,
) -> list[tuple[tuple[int, ...], int, int, int]]:
    edge_order = edges or (TRI_EDGES if dimension == 2 else TET_EDGES)
    face_order = faces or TET_FACES
    result: list[tuple[tuple[int, ...], int, int, int]] = []
    for vertex in range(dimension + 1):
        counts = [0] * (dimension + 1)
        counts[vertex] = order
        result.append((tuple(counts), 0, vertex, 0))
    for edge_index, (first, second) in enumerate(edge_order):
        for slot, position in enumerate(range(1, order), start=0):
            counts = [0] * (dimension + 1)
            counts[first] = order - position
            counts[second] = position
            result.append((tuple(counts), 1, edge_index, slot))
    if dimension == 2:
        for slot, counts in enumerate(_triangle_interior(order)):
            result.append((counts, 2, 0, slot))
    else:
        face_interior = _triangle_interior(order)
        for face_index, face in enumerate(face_order):
            for slot, local_counts in enumerate(face_interior):
                counts = [0, 0, 0, 0]
                for local_index, global_index in enumerate(face):
                    counts[global_index] = local_counts[local_index]
                result.append((tuple(counts), 2, face_index, slot))
        slot = 0
        for a in range(order - 3, 0, -1):
            for b in range(order - a - 2, 0, -1):
                for c in range(order - a - b - 1, 0, -1):
                    d = order - a - b - c
                    if d > 0:
                        result.append(((a, b, c, d), 3, 0, slot))
                        slot += 1
    return result


def _reference_triangle(order: int) -> list[tuple[int, int, int]]:
    if order == 1:
        return [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    if order == 2:
        return [
            (2, 0, 0),
            (0, 2, 0),
            (0, 0, 2),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
        ]
    if order == 4:
        return [
            (4, 0, 0),
            (0, 4, 0),
            (0, 0, 4),
            (2, 2, 0),
            (0, 2, 2),
            (2, 0, 2),
            (3, 1, 0),
            (1, 3, 0),
            (0, 3, 1),
            (0, 1, 3),
            (1, 0, 3),
            (3, 0, 1),
            (2, 1, 1),
            (1, 2, 1),
            (1, 1, 2),
        ]
    # The frozen reference implementation has no P8 external ordering. Keep
    # native P8 reference exchange deterministic with the canonical ordering.
    return [record[0] for record in _nodes(2, order)]


def _external_nodes(dimension: int, order: int, ordering: str) -> list[tuple[int, ...]]:
    canonical = [record[0] for record in _nodes(dimension, order)]
    if ordering in ("canonical", "petsc"):
        return canonical
    if ordering == "reference":
        return _reference_triangle(order) if dimension == 2 else canonical
    # Gmsh and VTK enumerate high-order simplex interiors recursively: after
    # vertices and edges, a triangle interior is a complete triangle of order
    # p - 3 shifted one lattice unit away from every edge; tetrahedron faces
    # follow the same rule and its volume is a complete tetrahedron of order
    # p - 4 shifted away from every face.  A flat lexicographic enumeration is
    # equivalent through P4, but is wrong for P8 face and volume interiors.
    if dimension == 2:
        boundary = [
            record[0]
            for record in _nodes(
                dimension,
                order,
                edges=EXTERNAL_TRI_EDGES,
            )
            if record[1] < 2
        ]
        if order < 3:
            return boundary
        interior = (
            [(0, 0, 0)] if order == 3 else _external_nodes(2, order - 3, ordering)
        )
        return boundary + [tuple(value + 1 for value in node) for node in interior]
    if ordering == "gmsh":
        edges = GMSH_TET_EDGES
        faces = GMSH_TET_FACES
    else:
        edges = VTK_TET_EDGES
        faces = VTK_TET_FACES
    boundary = [
        record[0]
        for record in _nodes(
            dimension,
            order,
            edges=edges,
            faces=faces,
        )
        if record[1] < 2
    ]
    if order >= 3:
        local_face_nodes = (
            [(0, 0, 0)] if order == 3 else _external_nodes(2, order - 3, ordering)
        )
        for face in faces:
            for local_node in local_face_nodes:
                node = [0, 0, 0, 0]
                for local_index, global_index in enumerate(face):
                    node[global_index] = local_node[local_index] + 1
                boundary.append(tuple(node))
    if order >= 4:
        local_volume_nodes = (
            [(0, 0, 0, 0)] if order == 4 else _external_nodes(3, order - 4, ordering)
        )
        boundary.extend(
            tuple(value + 1 for value in node) for node in local_volume_nodes
        )
    return boundary


def _digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _double(value: float) -> str:
    text = format(float(value), ".17g")
    if "." not in text and "e" not in text:
        text += ".0"
    return text


def _polynomial_multiply(
    left: dict[tuple[int, ...], Fraction],
    right: dict[tuple[int, ...], Fraction],
) -> dict[tuple[int, ...], Fraction]:
    result: dict[tuple[int, ...], Fraction] = {}
    for left_power, left_coefficient in left.items():
        for right_power, right_coefficient in right.items():
            power = tuple(
                left_power[index] + right_power[index]
                for index in range(len(left_power))
            )
            result[power] = (
                result.get(power, Fraction(0)) + left_coefficient * right_coefficient
            )
    return {power: coefficient for power, coefficient in result.items() if coefficient}


def _lambda_polynomial(
    dimension: int, barycentric_index: int
) -> dict[tuple[int, ...], Fraction]:
    zero = (0,) * dimension
    if barycentric_index == 0:
        result = {zero: Fraction(1)}
        for coordinate in range(dimension):
            power = [0] * dimension
            power[coordinate] = 1
            result[tuple(power)] = Fraction(-1)
        return result
    power = [0] * dimension
    power[barycentric_index - 1] = 1
    return {tuple(power): Fraction(1)}


def _basis_polynomial(
    dimension: int, order: int, barycentric: tuple[int, ...]
) -> dict[tuple[int, ...], Fraction]:
    result = {(0,) * dimension: Fraction(1)}
    for barycentric_index, degree in enumerate(barycentric):
        lambda_polynomial = _lambda_polynomial(dimension, barycentric_index)
        for factor in range(degree):
            term = {
                power: coefficient * Fraction(order, factor + 1)
                for power, coefficient in lambda_polynomial.items()
            }
            zero = (0,) * dimension
            term[zero] = term.get(zero, Fraction(0)) - Fraction(factor, factor + 1)
            result = _polynomial_multiply(result, term)
    return result


def _differentiate(
    polynomial: dict[tuple[int, ...], Fraction], coordinate: int
) -> dict[tuple[int, ...], Fraction]:
    result: dict[tuple[int, ...], Fraction] = {}
    for power, coefficient in polynomial.items():
        if power[coordinate] == 0:
            continue
        differentiated = list(power)
        differentiated[coordinate] -= 1
        result[tuple(differentiated)] = coefficient * power[coordinate]
    return result


def _polynomial_record(polynomial: dict[tuple[int, ...], Fraction]) -> list[list[Any]]:
    return [
        [list(power), coefficient.numerator, coefficient.denominator]
        for power, coefficient in sorted(polynomial.items())
    ]


def _polynomial_expression(polynomial: dict[tuple[int, ...], Fraction]) -> str:
    terms: list[str] = []
    for power, coefficient in sorted(polynomial.items()):
        factors = [_double(float(coefficient))]
        for coordinate, exponent in enumerate(power):
            if exponent != 0:
                factors.append(
                    "ss_fe_generated_integer_power("
                    f"reference_point[{coordinate}U], {exponent}U)"
                )
        terms.append("(" + " * ".join(factors) + ")")
    return " + ".join(terms) if terms else "0.0"


def _basis_functions(
    prefix: str, dimension: int, tuples: list[tuple[int, ...]]
) -> tuple[list[str], str]:
    order = sum(tuples[0])
    values = [
        _basis_polynomial(dimension, order, barycentric) for barycentric in tuples
    ]
    gradients = [
        [_differentiate(polynomial, coordinate) for coordinate in range(dimension)]
        for polynomial in values
    ]
    digest = _digest(
        {
            "values": [_polynomial_record(polynomial) for polynomial in values],
            "gradients": [
                [_polynomial_record(polynomial) for polynomial in basis_gradients]
                for basis_gradients in gradients
            ],
        }
    )
    lines = [
        f"static void {prefix}_basis_scalar(const double *reference_point,",
        "                                      double *values, double *gradients) {",
    ]
    for basis, polynomial in enumerate(values):
        lines.append(f"  values[{basis}U] = {_polynomial_expression(polynomial)};")
    lines.append("  if (gradients != NULL) {")
    for basis, basis_gradients in enumerate(gradients):
        for coordinate, polynomial in enumerate(basis_gradients):
            offset = basis * dimension + coordinate
            lines.append(
                f"    gradients[{offset}U] = {_polynomial_expression(polynomial)};"
            )
    lines.extend(
        [
            "  }",
            "}",
            "",
            f"static void {prefix}_basis_batch(size_t point_count,",
            "                                     const double *reference_points,",
            "                                     double *values, double *gradients) {",
            "  size_t point;",
            "  for (point = 0U; point < point_count; ++point) {",
            f"    {prefix}_basis_scalar(",
            f"        reference_points + point * {dimension}U,",
            f"        values + point * {len(tuples)}U,",
            "        gradients == NULL",
            "            ? NULL",
            f"            : gradients + point * {len(tuples) * dimension}U);",
            "  }",
            "}",
            "",
        ]
    )
    return lines, digest


def _factored_basis_functions(
    prefix: str, dimension: int, tuples: list[tuple[int, ...]]
) -> tuple[list[str], str]:
    """Emit a stable cardinal-product evaluator for high-order simplices."""
    order = sum(tuples[0])
    node_count = len(tuples)
    digest = _digest(
        {
            "kind": "factored-cardinal-simplex-v1",
            "dimension": dimension,
            "order": order,
            "barycentric": tuples,
        }
    )
    lines = [
        f"static void {prefix}_basis_scalar(const double *reference_point,",
        "                                      double *values, double *gradients) {",
        f"  double lambda[{dimension + 1}U] = {{0.0}};",
        f"  double factors[{dimension + 1}U][{order + 1}U] = {{{{0.0}}}};",
        f"  double derivatives[{dimension + 1}U][{order + 1}U] = {{{{0.0}}}};",
        "  uint32_t barycentric_coordinate;",
        "  uint32_t degree;",
        "  size_t node;",
        "  lambda[0U] = 1.0;",
        f"  for (barycentric_coordinate = 0U; barycentric_coordinate < {dimension}U;",
        "       ++barycentric_coordinate) {",
        "    lambda[barycentric_coordinate + 1U] =",
        "        reference_point[barycentric_coordinate];",
        "    lambda[0U] -= reference_point[barycentric_coordinate];",
        "  }",
        "  for (barycentric_coordinate = 0U; "
        f"barycentric_coordinate < {dimension + 1}U;",
        "       ++barycentric_coordinate) {",
        "    factors[barycentric_coordinate][0U] = 1.0;",
        f"    for (degree = 1U; degree <= {order}U; ++degree) {{",
        "      uint32_t omitted;",
        "      double derivative = 0.0;",
        "      factors[barycentric_coordinate][degree] =",
        "          factors[barycentric_coordinate][degree - 1U] *",
        f"          ({order}.0 * lambda[barycentric_coordinate] -",
        "           (double)(degree - 1U)) / (double)degree;",
        "      for (omitted = 0U; omitted < degree; ++omitted) {",
        f"        double product_value = {order}.0;",
        "        uint32_t factor;",
        "        for (factor = 0U; factor < degree; ++factor) {",
        "          if (factor != omitted) {",
        f"            product_value *= {order}.0 * lambda[barycentric_coordinate] -",
        "                             (double)factor;",
        "          }",
        "        }",
        "        for (factor = 2U; factor <= degree; ++factor) {",
        "          product_value /= (double)factor;",
        "        }",
        "        derivative += product_value;",
        "      }",
        "      derivatives[barycentric_coordinate][degree] = derivative;",
        "    }",
        "  }",
        f"  for (node = 0U; node < {node_count}U; ++node) {{",
        "    double value = 1.0;",
        "    for (barycentric_coordinate = 0U; "
        f"barycentric_coordinate < {dimension + 1}U;",
        "         ++barycentric_coordinate) {",
        "      value *= factors[barycentric_coordinate]",
        f"                      [{prefix}_barycentric[node * {dimension + 1}U +",
        "                                                barycentric_coordinate]];",
        "    }",
        "    values[node] = value;",
        "    if (gradients != NULL) {",
        "      uint32_t coordinate;",
        f"      for (coordinate = 0U; coordinate < {dimension}U; ++coordinate) {{",
        "        double positive = 1.0;",
        "        double negative = 1.0;",
        "        for (barycentric_coordinate = 0U; "
        f"barycentric_coordinate < {dimension + 1}U;",
        "             ++barycentric_coordinate) {",
        "          const uint32_t alpha =",
        f"              {prefix}_barycentric[node * {dimension + 1}U +",
        "                                        barycentric_coordinate];",
        "          positive *= barycentric_coordinate == coordinate + 1U",
        "                          ? derivatives[barycentric_coordinate][alpha]",
        "                          : factors[barycentric_coordinate][alpha];",
        "          negative *= barycentric_coordinate == 0U",
        "                          ? derivatives[barycentric_coordinate][alpha]",
        "                          : factors[barycentric_coordinate][alpha];",
        "        }",
        f"        gradients[node * {dimension}U + coordinate] = positive - negative;",
        "      }",
        "    }",
        "  }",
        "}",
        "",
        f"static void {prefix}_basis_batch(size_t point_count,",
        "                                     const double *reference_points,",
        "                                     double *values, double *gradients) {",
        "  size_t point;",
        "  for (point = 0U; point < point_count; ++point) {",
        f"    {prefix}_basis_scalar(",
        f"        reference_points + point * {dimension}U,",
        f"        values + point * {node_count}U,",
        "        gradients == NULL",
        "            ? NULL",
        f"            : gradients + point * {node_count * dimension}U);",
        "  }",
        "}",
        "",
    ]
    return lines, digest


def _minimal_symmetric_simplex(
    rule: dict[str, Any], source: dict[str, Any]
) -> tuple[list[list[float]], list[float]]:
    dimension = int(rule["dimension"])
    degree = str(int(rule["exact_degree"]))
    compact = source["minimal_symmetric_simplex"]["rules"][str(dimension)][degree]
    partitions_by_dimension = {
        2: ((0, 0, 0), (0, 0, 1), (0, 1, 2)),
        3: (
            (0, 0, 0, 0),
            (0, 0, 0, 1),
            (0, 0, 1, 1),
            (0, 0, 1, 2),
            (0, 1, 2, 3),
        ),
    }
    partitions = partitions_by_dimension[dimension]
    counts = [int(value) for value in compact["counts"]]
    compact_weights = [float(value) for value in compact["weights"]]
    compact_orbits = [float(value) for value in compact["orbits"]]
    if len(counts) != len(partitions):
        raise ValueError(f"{rule['id']} has invalid orbit partition count")
    points: list[list[float]] = []
    weights: list[float] = []
    weight_offset = 0
    orbit_offset = 0
    for partition, orbit_count in zip(partitions, counts, strict=True):
        compact_coordinate_count = max(partition) + 1
        for _ in range(orbit_count):
            coordinates = compact_orbits[
                orbit_offset : orbit_offset + compact_coordinate_count
            ]
            orbit_offset += compact_coordinate_count
            barycentric = tuple(coordinates[index] for index in partition)
            for node in sorted(set(permutations(barycentric))):
                points.append(list(node[1:]))
                weights.append(compact_weights[weight_offset] / (2.0**dimension))
            weight_offset += 1
    if weight_offset != len(compact_weights) or orbit_offset != len(compact_orbits):
        raise ValueError(f"{rule['id']} has unused compact quadrature data")
    return points, weights


def _expand_quadrature(rule: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    if "construction" not in rule:
        return dict(rule)
    construction = rule["construction"]
    if construction.get("kind") == "petsc-minimal-symmetric-simplex":
        points, weights = _minimal_symmetric_simplex(rule, source)
        expanded = dict(rule)
        expanded["points"] = points
        expanded["weights"] = weights
        return expanded
    if construction.get("kind") != "duffy-tensor-gauss-legendre":
        raise ValueError(f"unsupported quadrature construction {construction!r}")
    axis_order = int(construction["axis_order"])
    axis = source["gauss_legendre_unit_interval"][str(axis_order)]
    nodes = [float(value) for value in axis["points"]]
    weights = [float(value) for value in axis["weights"]]
    if len(nodes) != axis_order or len(weights) != axis_order:
        raise ValueError(f"Gauss-Legendre axis {axis_order} has invalid length")
    dimension = int(rule["dimension"])
    expected_degree = 2 * axis_order - (2 if dimension == 2 else 3)
    if dimension not in (2, 3) or int(rule["exact_degree"]) != expected_degree:
        raise ValueError(f"{rule['id']} has inconsistent Duffy exactness metadata")
    points: list[list[float]] = []
    quadrature_weights: list[float] = []
    if dimension == 2:
        for u_index, u in enumerate(nodes):
            for v_index, v in enumerate(nodes):
                points.append([u, (1.0 - u) * v])
                quadrature_weights.append(
                    weights[u_index] * weights[v_index] * (1.0 - u)
                )
    else:
        for u_index, u in enumerate(nodes):
            for v_index, v in enumerate(nodes):
                for w_index, w in enumerate(nodes):
                    points.append(
                        [
                            u,
                            (1.0 - u) * v,
                            (1.0 - u) * (1.0 - v) * w,
                        ]
                    )
                    quadrature_weights.append(
                        weights[u_index]
                        * weights[v_index]
                        * weights[w_index]
                        * (1.0 - u)
                        * (1.0 - u)
                        * (1.0 - v)
                    )
    expanded = dict(rule)
    expanded["points"] = points
    expanded["weights"] = quadrature_weights
    return expanded


def _array(
    type_name: str, name: str, values: list[str], *, columns: int = 12
) -> list[str]:
    lines = [f"static const {type_name} {name}[] = {{"]
    for start in range(0, len(values), columns):
        lines.append("  " + ", ".join(values[start : start + columns]) + ",")
    lines.append("};")
    lines.append("")
    return lines


def _header() -> bytes:
    return (
        "/* Simplex finite-element descriptors, quadrature rules and facet\n"
        " * permutations, generated by slope_codegen.finite_elements from\n"
        " * schemas/source/finite-elements.json. Do not edit.\n"
        " * SPDX-License-Identifier: BSD-2-Clause\n"
        " */\n"
        "#ifndef SS_FE_GENERATED_H\n"
        "#define SS_FE_GENERATED_H\n\n"
        '#include "slope_internal/fe/layout.h"\n\n'
        "#include <stddef.h>\n\n"
        "extern const SsFeDescriptor ss_fe_generated_descriptors[];\n"
        "extern const size_t ss_fe_generated_descriptor_count;\n"
        "extern const SsFeQuadrature ss_fe_generated_quadratures[];\n"
        "extern const size_t ss_fe_generated_quadrature_count;\n"
        "extern const SsFePermutation ss_fe_generated_permutations[];\n"
        "extern const size_t ss_fe_generated_permutation_count;\n\n"
        "#endif\n"
    ).encode("utf-8")


def _python_layout(source_digest: str) -> bytes:
    lines = [
        '"""Generated canonical simplex nodes and external-format maps."""',
        "",
        "# SPDX-License-Identifier: BSD-2-Clause",
        "# Generated by slope_codegen.finite_elements.",
        "",
        "from __future__ import annotations",
        "",
        "SOURCE_DIGEST = (",
        f"    {json.dumps(source_digest)}",
        ")",
        "",
        "CANONICAL_BARYCENTRIC: dict[tuple[int, int], tuple[tuple[int, ...], ...]] = {",
    ]
    for dimension in (2, 3):
        for order in ORDERS:
            lines.append(f"    ({dimension}, {order}): (")
            for record in _nodes(dimension, order):
                lines.append(f"        {record[0]!r},")
            lines.append("    ),")
    lines.extend(
        [
            "}",
            "",
            "CANONICAL_TO_VTK: dict[tuple[int, int], tuple[int, ...]] = {",
        ]
    )
    for dimension in (2, 3):
        for order in ORDERS:
            canonical = [record[0] for record in _nodes(dimension, order)]
            canonical_index = {counts: index for index, counts in enumerate(canonical)}
            external = _external_nodes(dimension, order, "vtk")
            canonical_to_external = [0] * len(external)
            for external_index, counts in enumerate(external):
                canonical_to_external[canonical_index[counts]] = external_index
            entry = f"    ({dimension}, {order}): {tuple(canonical_to_external)!r},"
            if len(entry) <= 88:
                lines.append(entry)
            else:
                lines.append(f"    ({dimension}, {order}): (")
                lines.extend(f"        {value}," for value in canonical_to_external)
                lines.append("    ),")
    lines.extend(["}", ""])
    return "\n".join(lines).encode("utf-8")


def _source(source: dict[str, Any], source_digest: str) -> bytes:
    lines = [
        "/* Simplex finite-element tables, generated by",
        " * slope_codegen.finite_elements from schemas/source/finite-elements.json",
        " * and the PETSc quadrature orbits. Do not edit.",
        " * SPDX-License-Identifier: BSD-2-Clause",
        " */",
        '#include "ss_fe_generated.h"',
        "",
        "static double ss_fe_generated_integer_power("
        "double value, uint32_t exponent) {",
        "  double result = 1.0;",
        "  uint32_t index;",
        "  for (index = 0U; index < exponent; ++index) {",
        "    result *= value;",
        "  }",
        "  return result;",
        "}",
        "",
    ]
    descriptor_names: list[str] = []
    descriptor_digests: dict[tuple[int, int], str] = {}
    basis_digests: dict[tuple[int, int], str] = {}
    permutation_records: list[str] = []

    for dimension in (2, 3):
        cell_name = "triangle" if dimension == 2 else "tetrahedron"
        for order in ORDERS:
            prefix = f"ss_fe_{dimension}d_p{order}"
            records = _nodes(dimension, order)
            tuples = [record[0] for record in records]
            flattened = [str(value) + "U" for item in tuples for value in item]
            entity_dimensions = [str(record[1]) + "U" for record in records]
            entity_indices = [str(record[2]) + "U" for record in records]
            entity_slots = [str(record[3]) + "U" for record in records]
            lines.extend(_array("uint8_t", f"{prefix}_barycentric", flattened))
            lines.extend(
                _array("uint8_t", f"{prefix}_entity_dimensions", entity_dimensions)
            )
            lines.extend(_array("uint8_t", f"{prefix}_entity_indices", entity_indices))
            lines.extend(_array("uint8_t", f"{prefix}_entity_slots", entity_slots))
            if order >= 8:
                basis_lines, basis_digest = _factored_basis_functions(
                    prefix, dimension, tuples
                )
            else:
                basis_lines, basis_digest = _basis_functions(prefix, dimension, tuples)
            lines.extend(basis_lines)
            descriptor_names.append(prefix)
            descriptor_digests[(dimension, order)] = _digest(tuples)
            basis_digests[(dimension, order)] = basis_digest

            canonical_index = {counts: index for index, counts in enumerate(tuples)}
            for ordering, enum_name in ORDERINGS:
                external = _external_nodes(dimension, order, ordering)
                external_to_canonical = [canonical_index[counts] for counts in external]
                canonical_to_external = [0] * len(external)
                for external_index, canonical in enumerate(external_to_canonical):
                    canonical_to_external[canonical] = external_index
                perm_prefix = f"{prefix}_{ordering}"
                lines.extend(
                    _array(
                        "uint8_t",
                        f"{perm_prefix}_canonical_to_external",
                        [str(value) + "U" for value in canonical_to_external],
                    )
                )
                lines.extend(
                    _array(
                        "uint8_t",
                        f"{perm_prefix}_external_to_canonical",
                        [str(value) + "U" for value in external_to_canonical],
                    )
                )
                source_version = (
                    "runtime-owned-reference-v1"
                    if ordering == "petsc"
                    else "generated-v1"
                )
                permutation_records.append(
                    "  {"
                    f'"{cell_name}-p{order}-{ordering}-v1", {enum_name}, '
                    f"{len(tuples)}U, "
                    f"{perm_prefix}_canonical_to_external, "
                    f"{perm_prefix}_external_to_canonical, "
                    f'"{source_version}", '
                    f'"{descriptor_digests[(dimension, order)]}"'
                    "},"
                )

    quadrature_records: list[str] = []
    rules = [_expand_quadrature(rule, source) for rule in source["quadrature"]]
    by_id = {rule["id"]: rule for rule in rules}
    for alias in source["facet_aliases"]:
        base = by_id[alias["source_rule"]]
        rules.append(
            {
                **base,
                "id": alias["id"],
                "domain": "facet",
                "parent_dimension": alias["parent_dimension"],
            }
        )
    for index, rule in enumerate(rules):
        prefix = f"ss_fe_quadrature_{index}"
        points = [
            _double(coordinate) for point in rule["points"] for coordinate in point
        ]
        weights = [_double(weight) for weight in rule["weights"]]
        lines.extend(_array("double", f"{prefix}_points", points, columns=6))
        lines.extend(_array("double", f"{prefix}_weights", weights, columns=6))
        has_negative = any(float(weight) < 0.0 for weight in rule["weights"])
        domain = {
            "volume": "SS_FE_QUADRATURE_VOLUME",
            "facet": "SS_FE_QUADRATURE_FACET",
        }[rule["domain"]]
        quadrature_records.append(
            "  {"
            f"{json.dumps(rule['id'])}, {domain}, {rule['parent_dimension']}U, "
            f"{rule['dimension']}U, "
            f"{rule['exact_degree']}U, {len(rule['points'])}U, "
            f"{1 if has_negative else 0}, {prefix}_points, {prefix}_weights, "
            f"{json.dumps(rule.get('reuse_id', 'REUSE-0005'))}"
            "},"
        )

    lines.append("const SsFeDescriptor ss_fe_generated_descriptors[] = {")
    for dimension in (2, 3):
        for order in ORDERS:
            prefix = f"ss_fe_{dimension}d_p{order}"
            count = len(_nodes(dimension, order))
            facet_nodes = (
                order + 1 if dimension == 2 else (order + 1) * (order + 2) // 2
            )
            lines.append(
                "  {"
                f'"{"triangle" if dimension == 2 else "tetrahedron"}-p{order}-v1", '
                f"{dimension}U, {order}U, {count}U, {facet_nodes}U, "
                f"{3 if dimension == 2 else 6}U, {0 if dimension == 2 else 4}U, "
                f"{prefix}_barycentric, {prefix}_entity_dimensions, "
                f"{prefix}_entity_indices, {prefix}_entity_slots, "
                f"{prefix}_basis_scalar, {prefix}_basis_batch, "
                f'"{basis_digests[(dimension, order)]}", '
                f'"{source_digest}"'
                "},"
            )
    lines.extend(
        [
            "};",
            "",
            "const size_t ss_fe_generated_descriptor_count =",
            "    sizeof(ss_fe_generated_descriptors) /",
            "    sizeof(ss_fe_generated_descriptors[0]);",
            "",
            "const SsFeQuadrature ss_fe_generated_quadratures[] = {",
            *quadrature_records,
            "};",
            "",
            "const size_t ss_fe_generated_quadrature_count =",
            "    sizeof(ss_fe_generated_quadratures) /",
            "    sizeof(ss_fe_generated_quadratures[0]);",
            "",
            "const SsFePermutation ss_fe_generated_permutations[] = {",
            *permutation_records,
            "};",
            "",
            "const size_t ss_fe_generated_permutation_count =",
            "    sizeof(ss_fe_generated_permutations) /",
            "    sizeof(ss_fe_generated_permutations[0]);",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def render_outputs(
    root: Path, format_c: Callable[[bytes, Path], bytes]
) -> tuple[dict[Path, bytes], dict[str, str]]:
    source_path = root / "schemas/source/finite-elements.json"
    minimal_symmetric_path = (
        root / "schemas/source/petsc-minimal-symmetric-simplex.json"
    )
    generator_path = root / "generators/src/slope_codegen/finite_elements.py"
    source = _load(source_path)
    source["minimal_symmetric_simplex"] = _load(minimal_symmetric_path)
    if tuple(source.get("orders", ())) != ORDERS:
        raise ValueError("finite-element source must declare exactly P1/P2/P4/P8")
    source_digest = _digest(source)
    header_path = Path("generated/include/ss_fe_generated.h")
    c_path = Path("generated/src/ss_fe_generated.c")
    python_path = Path("python/src/slope_stability/_fe_layout.py")
    outputs = {
        header_path: format_c(_header(), root / header_path),
        c_path: format_c(_source(source, source_digest), root / c_path),
        python_path: _python_layout(source_digest),
    }
    sources = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (source_path, minimal_symmetric_path, generator_path)
    }
    return outputs, sources
