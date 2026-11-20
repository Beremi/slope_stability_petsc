# SPDX-License-Identifier: BSD-2-Clause
"""Hydraulic units and the CASE-015 hydraulic support in the checked catalog."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from slope_stability.authoring import Asset, SemanticSupport
from slope_stability.compile import compact_simplex_mesh
from slope_stability.compile.adaptations import (
    ADAPTATION,
    SOURCE_SHA256,
    SUPPORT_NAME,
    adapt_mesh,
    derived_support_names,
    select_homogeneous_hydraulic_faces,
)

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "cases" / "catalog" / "cases"
ASSETS = ROOT / "cases" / "catalog" / "assets"


class HydraulicCatalogContract(unittest.TestCase):
    """QAL-CONTRACT-HYD-UNITS-001."""

    def test_checked_catalog_preserves_corrected_physical_units(self) -> None:
        """CASE-006 (Luzec) conductivities are converted from m/day to m/s."""
        case = json.loads((CASES / "CASE-006-canonical.json").read_text())
        expected = [
            1e-8,
            1e-3,
            1e-8,
            0.86 / 86400,
            1e-3,
            0.86 / 86400,
            1e-8,
            0.86 / 86400,
        ]
        hydraulic = [
            row
            for row in case["materials"]
            if row["model"] == "steady_unconfined_darcy"
        ]
        self.assertEqual(len(hydraulic), 8)
        for row, value in zip(hydraulic, expected, strict=True):
            self.assertAlmostEqual(
                row["parameters"]["conductivity"]["value"], value, places=15
            )
            self.assertEqual(row["parameters"]["conductivity"]["unit"], "m/s")
        provenance = case["extensions"]["conductivity_unit_correction"]
        self.assertEqual(provenance["source_unit"], "m/day")
        self.assertEqual(provenance["divisor"], 86400)

    def test_checked_homogeneous_catalog_has_reflected_head(self) -> None:
        """CASE-015 heads use the MATLAB frame mapped by x_M = 10 - x."""
        case = json.loads((CASES / "CASE-015-canonical.json").read_text())
        rules = [
            rule for rule in case["boundary_rules"] if rule["field"] == "pore_pressure"
        ]
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["support"], SUPPORT_NAME)
        self.assertEqual(rules[0]["parameters"]["coordinates"], [-13, 10])
        self.assertEqual(rules[0]["parameters"]["values"], [12, 18])
        self.assertEqual(
            case["extensions"]["head_coordinate_correction"]["reference_x_from_native"],
            {"scale": -1.0, "offset_m": 10.0},
        )


class HydraulicAdaptationContract(unittest.TestCase):
    """Derived hydraulic support of the homogeneous 3D slope mesh."""

    def test_faces_exclude_interior_and_side_wall_triangles(self) -> None:
        vertices: dict[int, tuple[float, float, float]] = {}
        faces: dict[int, tuple[int, int, int]] = {}
        cells = []
        geometries = [
            # A low-bench plane shared by two cells must be excluded.
            [(-20, 10, 0), (-19, 10, 0), (-20, 10, 1)],
            # All vertices lie on pressure planes, but this face is a side wall.
            [(-30, 0, 5), (10, 0, 5), (-5, 20, 5)],
            [(10, 0, 0), (10, 1, 0), (10, 0, 1)],
            [(-14, 11, 0), (-13, 12, 0), (-14, 11, 1)],
            [(-20, 0, 0), (-19, 0, 0), (-20, 0, 1)],
            [(0, 20, 0), (1, 20, 0), (0, 20, 1)],
            [(-25, 10, 0), (-24, 10, 0), (-25, 10, 1)],
            [(0, 10, 0), (1, 10, 0), (0, 10, 1)],
        ]
        for index, points in enumerate(geometries):
            ids = tuple(10 * index + offset for offset in range(3))
            vertices.update(zip(ids, points))
            faces[index] = ids
            cells.append((*ids, 1000 + index))
            vertices[1000 + index] = (0, 5, 2)
        cells.append((*faces[0], 2000))
        vertices[2000] = (-20, 11, 0)
        expected = (2, 3, 5, 6)
        self.assertEqual(
            select_homogeneous_hydraulic_faces(vertices, faces, cells), expected
        )
        self.assertEqual(
            select_homogeneous_hydraulic_faces(
                vertices,
                {key: tuple(reversed(value)) for key, value in faces.items()},
                [tuple(reversed(cell)) for cell in reversed(cells)],
            ),
            expected,
        )

    def test_adaptation_is_bound_to_declared_source(self) -> None:
        raw = json.loads(
            (ASSETS / "asset.3d_homo_slope.adaptive_family_a_l1.g1.json").read_text()
        )
        asset = Asset.model_validate_json(json.dumps(raw))
        supports = tuple(s for s in asset.supports if s.name != SUPPORT_NAME)
        asset = asset.model_copy(
            update={
                "extensions": {"hydraulic_support_adaptation": dict(ADAPTATION)},
                "supports": (
                    *supports,
                    SemanticSupport(name=SUPPORT_NAME, kind="boundary", dimension=2),
                ),
            }
        )
        self.assertEqual(derived_support_names(asset), (SUPPORT_NAME,))
        mesh = compact_simplex_mesh(3, 1)
        with self.assertRaisesRegex(ValueError, "imported-source mismatch"):
            adapt_mesh(asset, mesh)
        wrong = asset.model_copy(
            update={
                "extensions": {
                    "hydraulic_support_adaptation": {**ADAPTATION, "id": "unknown"}
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "unrecognized"):
            derived_support_names(wrong)
        wrong = asset.model_copy(
            update={
                "mesh_source": asset.mesh_source.model_copy(update={"sha256": "0" * 64})
            }
        )
        with self.assertRaisesRegex(ValueError, "unrecognized"):
            derived_support_names(wrong)
        plain = asset.model_copy(update={"extensions": {}})
        self.assertIs(adapt_mesh(plain, mesh), mesh)
        mesh = replace(
            mesh,
            topology_vertex_coordinates=mesh.topology_vertex_coordinates + [10, 0, 0],
            coordinate_nodes=mesh.coordinate_nodes + [10, 0, 0],
            provenance=replace(mesh.provenance, source_sha256=SOURCE_SHA256),
        )
        before = mesh.logical_digest()
        adapted = adapt_mesh(asset, mesh)
        self.assertEqual(len(adapted.supports[SUPPORT_NAME].entity_ids), 1)
        self.assertNotEqual(adapted.logical_digest(), before)
        self.assertEqual(mesh.logical_digest(), before)
        for name, support in mesh.supports.items():
            self.assertEqual(adapted.supports[name], support)
        self.assertEqual(adapted.provenance, mesh.provenance)
        with self.assertRaisesRegex(ValueError, "already exists"):
            adapt_mesh(asset, adapted)
        with self.assertRaisesRegex(ValueError, "requires 3D G1"):
            derived_support_names(asset.model_copy(update={"coordinate_order": 2}))


if __name__ == "__main__":
    unittest.main()
