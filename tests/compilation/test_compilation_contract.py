# SPDX-License-Identifier: BSD-2-Clause
"""Executable P-COMPILE compilation contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/src"))

from slope_stability.authoring import load_document
from slope_stability.compile import (
    CompileError,
    EntitySet,
    GmshImporter,
    Hdf5ReferenceImporter,
    ImportRequest,
    RawMesh,
    RawSupport,
    audit_geometry,
    canonicalize_mesh,
    compact_simplex_mesh,
    compile_problem,
    compile_problem_model,
    uniformly_refine_raw_mesh,
)
from tests.support.problem_fixtures import (
    asset_document,
    assert_schema_subset,
    case_document,
    source_tree_files,
    stamped,
)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def models(
    dimension: int = 2,
    field_order: int = 1,
    coordinate_order: int = 1,
):
    return (
        load_document(
            asset_document(dimension, coordinate_order=coordinate_order),
            source_path="asset.json",
        ),
        load_document(
            case_document(
                dimension,
                field_order=field_order,
                coordinate_order=coordinate_order,
            ),
            source_path="case.json",
        ),
    )


def generate_gmsh(directory: Path, dimension: int, order: int) -> Path:
    if dimension == 2:
        geometry = """
SetFactory("Built-in");
Point(1) = {0, 0, 0, 2};
Point(2) = {1, 0, 0, 2};
Point(3) = {0, 1, 0, 2};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 1};
Curve Loop(1) = {1, 2, 3};
Plane Surface(1) = {1};
Physical Surface("domain", 101) = {1};
Physical Curve("base", 201) = {1};
Physical Curve("boundary", 202) = {1, 2, 3};
"""
        flag = "-2"
    else:
        geometry = """
SetFactory("Built-in");
Point(1) = {0, 0, 0, 2};
Point(2) = {1, 0, 0, 2};
Point(3) = {0, 1, 0, 2};
Point(4) = {0, 0, 1, 2};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 1};
Line(4) = {1, 4};
Line(5) = {2, 4};
Line(6) = {3, 4};
Curve Loop(1) = {1, 2, 3};
Curve Loop(2) = {1, 5, -4};
Curve Loop(3) = {2, 6, -5};
Curve Loop(4) = {3, 4, -6};
Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};
Surface Loop(1) = {1, 2, 3, 4};
Volume(1) = {1};
Physical Volume("domain", 101) = {1};
Physical Surface("base", 201) = {1};
Physical Surface("boundary", 202) = {1, 2, 3, 4};
"""
        flag = "-3"
    source = directory / f"compact-{dimension}d.geo"
    target = directory / f"compact-{dimension}d-p{order}.msh"
    source.write_text(geometry, encoding="utf-8")
    completed = subprocess.run(
        [
            "gmsh",
            str(source),
            flag,
            "-order",
            str(order),
            "-format",
            "msh4",
            "-o",
            str(target),
            "-v",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stdout + completed.stderr)
    return target


class CompilationContract(unittest.TestCase):
    def test_prescribed_saturation(self) -> None:
        document = case_document(2)
        document["prescribed_saturation"] = {
            "axis": "x",
            "coordinates": [0, 100],
            "levels": [50, 150],
            "length_unit": "cm",
        }
        case = load_document(stamped(document))
        schemas = ROOT / "schemas/generated/json"
        schema = json.loads((schemas / "case.schema.json").read_text())
        assert_schema_subset(stamped(document), schema)
        asset, _ = models()
        mesh = compact_simplex_mesh(2, 1)
        with tempfile.TemporaryDirectory() as directory:
            package = compile_problem(
                asset=asset, case=case, mesh=mesh, destination=Path(directory)
            )
            problem = json.loads(package.problem_json.read_text())
            from slope_stability.resolution import load_compiled_problem

            self.assertEqual(
                load_compiled_problem(package.problem_json).problem_id,
                package.problem_id,
            )
            compiled_schema = json.loads(
                (schemas / "compiled-problem.schema.json").read_text()
            )
            assert_schema_subset(problem, compiled_schema)
            self.assertEqual(
                problem["prescribed_saturation"],
                {
                    "axis": 0,
                    "coordinates": [0.0, 1.0],
                    "levels": [0.5, 1.5],
                    "length_unit": "m",
                },
            )
        for malformed in [
            dict(axis="x", coordinates=[1, 0], levels=[0, 1], length_unit="m"),
            dict(axis="x", coordinates=[0, 1], levels=[0], length_unit="m"),
            dict(
                axis="x", coordinates=[0, 1], levels=[0, float("nan")], length_unit="m"
            ),
        ]:
            changed = deepcopy(document)
            changed["prescribed_saturation"] = malformed
            with self.assertRaises(ValueError):
                load_document(stamped(changed))
        for changed_profile in (
            dict(axis="z", coordinates=[0], levels=[1], length_unit="m"),
            dict(axis="x", coordinates=[0], levels=[1], length_unit="kg"),
        ):
            changed = deepcopy(document)
            changed["prescribed_saturation"] = changed_profile
            with self.assertRaises(ValueError):
                compile_problem_model(
                    asset=asset, case=load_document(stamped(changed)), mesh=mesh
                )
        changed = deepcopy(document)
        changed["physics"]["seepage"] = True
        with self.assertRaises(ValueError):
            load_document(stamped(changed))
        base_asset, base_case = models()
        base = compile_problem_model(asset=base_asset, case=base_case, mesh=mesh)
        self.assertNotIn("prescribed_saturation", base.logical_payload())

    def test_uniform_simplex_refinement(self) -> None:
        """CMP-IMPORT-UNIFORM-REFINE-001."""
        fixtures = (
            RawMesh(
                dimension=2,
                cell_type="triangle",
                coordinate_order=1,
                source_node_ids=np.array([10, 20, 30], dtype=np.uint64),
                coordinate_nodes=np.array(
                    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64
                ),
                cells=np.array([[0, 1, 2]], dtype=np.uint64),
                coordinate_connectivity=np.array([[0, 1, 2]], dtype=np.uint64),
                supports=(
                    RawSupport("domain", "region", 2, ((0, 1, 2),)),
                    RawSupport("base", "boundary", 1, ((0, 1),)),
                    RawSupport("locked", "node_set", 0, ((0,), (1,))),
                ),
                partition_hints=np.array([7], dtype=np.int64),
                importer_id="fixture",
                source_sha256="1" * 64,
            ),
            RawMesh(
                dimension=3,
                cell_type="tetrahedron",
                coordinate_order=1,
                source_node_ids=np.array([10, 20, 30, 40], dtype=np.uint64),
                coordinate_nodes=np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                ),
                cells=np.array([[0, 1, 2, 3]], dtype=np.uint64),
                coordinate_connectivity=np.array([[0, 1, 2, 3]], dtype=np.uint64),
                supports=(
                    RawSupport("domain", "region", 3, ((0, 1, 2, 3),)),
                    RawSupport("base", "boundary", 2, ((0, 1, 2),)),
                    RawSupport("locked", "node_set", 0, ((0,), (1,), (2,))),
                ),
                partition_hints=np.array([7], dtype=np.int64),
                importer_id="fixture",
                source_sha256="2" * 64,
            ),
        )
        expected = {
            2: {"vertices": 6, "cells": 4, "base": 2, "locked": 3},
            3: {"vertices": 10, "cells": 8, "base": 4, "locked": 6},
        }
        for fixture in fixtures:
            with self.subTest(dimension=fixture.dimension):
                refined = uniformly_refine_raw_mesh(fixture, levels=1)
                counts = expected[fixture.dimension]
                self.assertEqual(refined.source_node_ids.size, counts["vertices"])
                self.assertEqual(refined.cells.shape[0], counts["cells"])
                self.assertEqual(
                    len(
                        next(
                            item for item in refined.supports if item.name == "domain"
                        ).entities
                    ),
                    counts["cells"],
                )
                self.assertEqual(
                    len(
                        next(
                            item for item in refined.supports if item.name == "base"
                        ).entities
                    ),
                    counts["base"],
                )
                self.assertEqual(
                    len(
                        next(
                            item for item in refined.supports if item.name == "locked"
                        ).entities
                    ),
                    counts["locked"],
                )
                np.testing.assert_array_equal(
                    refined.partition_hints,
                    np.full(counts["cells"], 7, dtype=np.int64),
                )
                self.assertEqual(refined.uniform_refinement_levels, 1)
                self.assertIn("uniform-refinement-l1", refined.importer_id)
                self.assertGreater(
                    audit_geometry(canonicalize_mesh(refined)).minimum_determinant,
                    0.0,
                )

        twice = uniformly_refine_raw_mesh(fixtures[0], levels=2)
        staged = uniformly_refine_raw_mesh(
            uniformly_refine_raw_mesh(fixtures[0], levels=1), levels=1
        )
        self.assertEqual(
            canonicalize_mesh(twice).logical_digest(),
            canonicalize_mesh(staged).logical_digest(),
        )
        self.assertEqual(twice.cells.shape[0], 16)

        curved = replace(fixtures[0], coordinate_order=2)
        with self.assertRaises(CompileError) as caught:
            uniformly_refine_raw_mesh(curved, levels=1)
        self.assertEqual(caught.exception.code, "uniform_refinement_coordinate_order")

    def test_gmsh_import(self) -> None:
        """CMP-IMPORT-GMSH-001."""
        importer = GmshImporter()
        with tempfile.TemporaryDirectory(prefix="cmp-gmsh-") as raw:
            temporary = Path(raw)
            for dimension in (2, 3):
                for order in (1, 2, 4):
                    with self.subTest(dimension=dimension, order=order):
                        source = generate_gmsh(temporary, dimension, order)
                        mesh = importer.import_mesh(
                            source,
                            ImportRequest(
                                dimension=dimension,
                                coordinate_order=order,
                                required_supports=("domain", "base", "boundary"),
                            ),
                        )
                        self.assertEqual(mesh.dimension, dimension)
                        self.assertEqual(
                            mesh.cell_type,
                            "triangle" if dimension == 2 else "tetrahedron",
                        )
                        self.assertEqual(mesh.coordinate_order, order)
                        self.assertGreater(mesh.cells.shape[0], 0)
                        self.assertEqual(
                            set(mesh.supports), {"base", "boundary", "domain"}
                        )
                        self.assertTrue(mesh.import_maps.source_to_canonical_nodes)
                        self.assertGreaterEqual(
                            min(mesh.import_maps.source_to_canonical_nodes), 1
                        )
                        self.assertEqual(mesh.provenance.importer_id, "gmsh.msh4")
                        self.assertRegex(
                            mesh.provenance.source_sha256, r"^[0-9a-f]{64}$"
                        )
                        self.assertGreater(
                            audit_geometry(mesh).minimum_determinant, 0.0
                        )

            source = generate_gmsh(temporary, 3, 1)
            coarse = importer.import_mesh(
                source,
                ImportRequest(
                    dimension=3,
                    coordinate_order=1,
                    required_supports=("domain", "base", "boundary"),
                ),
            )
            refined = importer.import_mesh(
                source,
                ImportRequest(
                    dimension=3,
                    coordinate_order=1,
                    required_supports=("domain", "base", "boundary"),
                    uniform_refinement_levels=1,
                ),
            )
            self.assertEqual(refined.cells.shape[0], 8 * coarse.cells.shape[0])
            self.assertEqual(
                len(refined.supports["base"].entity_ids),
                4 * len(coarse.supports["base"].entity_ids),
            )
            self.assertEqual(
                refined.provenance.source_sha256,
                coarse.provenance.source_sha256,
            )
            self.assertEqual(refined.provenance.uniform_refinement_levels, 1)
            self.assertEqual(
                refined.provenance.importer_id,
                "gmsh.msh4+uniform-refinement-l1",
            )
            self.assertNotEqual(refined.logical_digest(), coarse.logical_digest())
            self.assertGreater(audit_geometry(refined).minimum_determinant, 0.0)

        with self.assertRaises(ValueError):
            ImportRequest(
                dimension=3,
                coordinate_order=1,
                required_supports=("domain",),
                uniform_refinement_levels=-1,
            )

    def test_reference_import(self) -> None:
        """CMP-IMPORT-REFERENCE-001."""
        coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        cells = np.array([[0, 1, 2]], dtype=np.uint64)
        expected = compact_simplex_mesh(2, 1)
        with tempfile.TemporaryDirectory(prefix="cmp-reference-") as raw:
            temporary = Path(raw)
            paths = []
            for layout in ("legacy", "matlab"):
                path = temporary / f"{layout}.h5"
                with h5py.File(path, "w", track_order=True) as handle:
                    handle.attrs["layout"] = layout
                    handle.attrs["dimension"] = 2
                    handle.attrs["cell_type"] = "triangle"
                    if layout == "matlab":
                        handle.create_dataset("coordinates", data=coordinates.T)
                        handle.create_dataset("cells", data=(cells + 1).T)
                    else:
                        handle.create_dataset("coordinates", data=coordinates)
                        handle.create_dataset("cells", data=cells)
                    supports = handle.create_group("supports")
                    supports.create_dataset(
                        "domain", data=np.array([[0, 1, 2]], dtype=np.uint64)
                    )
                    supports.create_dataset(
                        "base", data=np.array([[0, 1]], dtype=np.uint64)
                    )
                    supports.create_dataset(
                        "boundary",
                        data=np.array([[0, 1], [1, 2], [0, 2]], dtype=np.uint64),
                    )
                paths.append(path)
            imported = [
                Hdf5ReferenceImporter().import_mesh(
                    path,
                    ImportRequest(
                        dimension=2,
                        coordinate_order=1,
                        required_supports=("domain", "base", "boundary"),
                    ),
                )
                for path in paths
            ]
        for mesh in imported:
            np.testing.assert_array_equal(mesh.cells, expected.cells)
            np.testing.assert_allclose(mesh.coordinate_nodes, expected.coordinate_nodes)
            self.assertEqual(
                {
                    name: tuple(support.entity_ids)
                    for name, support in mesh.supports.items()
                },
                {
                    name: tuple(support.entity_ids)
                    for name, support in expected.supports.items()
                    if name in {"domain", "base", "boundary"}
                },
            )
        self.assertEqual(imported[0].logical_digest(), imported[1].logical_digest())

    def test_canonical_mesh_reordered_input(self) -> None:
        """CMP-UNIT-CANON-001."""
        raw_a = RawMesh(
            dimension=2,
            cell_type="triangle",
            coordinate_order=1,
            source_node_ids=np.array([10, 20, 30], dtype=np.uint64),
            coordinate_nodes=np.array(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64
            ),
            cells=np.array([[0, 1, 2]], dtype=np.uint64),
            coordinate_connectivity=np.array([[0, 1, 2]], dtype=np.uint64),
            supports=(
                RawSupport("domain", "region", 2, ((0, 1, 2),)),
                RawSupport("base", "boundary", 1, ((0, 1),)),
                RawSupport("boundary", "boundary", 1, ((0, 1), (1, 2), (2, 0))),
            ),
            partition_hints=np.array([7], dtype=np.int64),
            importer_id="independent.raw",
        )
        permutation = np.array([2, 0, 1])
        inverse = np.argsort(permutation)
        raw_b = replace(
            raw_a,
            source_node_ids=raw_a.source_node_ids[permutation],
            coordinate_nodes=raw_a.coordinate_nodes[permutation],
            cells=inverse[raw_a.cells],
            coordinate_connectivity=inverse[raw_a.coordinate_connectivity],
            supports=tuple(
                replace(
                    support,
                    entities=tuple(
                        tuple(int(inverse[index]) for index in entity)
                        for entity in support.entities
                    ),
                )
                for support in reversed(raw_a.supports)
            ),
            partition_hints=np.array([99], dtype=np.int64),
        )
        canonical_a = canonicalize_mesh(raw_a)
        canonical_b = canonicalize_mesh(raw_b)
        self.assertEqual(canonical_a.logical_digest(), canonical_b.logical_digest())
        np.testing.assert_array_equal(canonical_a.cells, canonical_b.cells)
        np.testing.assert_array_equal(
            canonical_a.coordinate_nodes, canonical_b.coordinate_nodes
        )
        self.assertEqual(canonical_a.cells.dtype, np.dtype("<u8"))
        self.assertEqual(canonical_a.coordinate_nodes.dtype, np.dtype("<f8"))

    def test_stable_ids(self) -> None:
        """CMP-MESH-ID-001."""
        baseline = compact_simplex_mesh(2, 4, two_cells=True)
        reordered = baseline.reordered(
            cell_order=tuple(reversed(range(baseline.cells.shape[0]))),
            coordinate_node_order=tuple(
                reversed(range(baseline.coordinate_nodes.shape[0]))
            ),
            partition_hints=(17, 3),
        )
        self.assertEqual(baseline.logical_digest(), reordered.logical_digest())
        for dimension in baseline.stable_entity_ids:
            np.testing.assert_array_equal(
                baseline.stable_entity_ids[dimension],
                reordered.stable_entity_ids[dimension],
            )
            np.testing.assert_array_equal(
                baseline.stable_entity_vertex_ids[dimension],
                reordered.stable_entity_vertex_ids[dimension],
            )
            self.assertEqual(
                baseline.stable_entity_vertex_ids[dimension].shape,
                (
                    baseline.stable_entity_ids[dimension].shape[0],
                    dimension + 1,
                ),
            )
        np.testing.assert_array_equal(
            baseline.topology_vertex_ids,
            reordered.topology_vertex_ids,
        )
        np.testing.assert_array_equal(
            baseline.topology_cell_ids,
            reordered.topology_cell_ids,
        )
        np.testing.assert_array_equal(
            baseline.topology_cells,
            reordered.topology_cells,
        )
        self.assertEqual(baseline.topology_cells.shape, (2, 3))
        self.assertLess(int(np.max(baseline.topology_cells)), 4)
        self.assertEqual(
            set(int(item) for item in baseline.topology_cell_ids),
            set(int(item) for item in baseline.stable_entity_ids[2]),
        )
        np.testing.assert_array_equal(
            baseline.coordinate_functional_ids,
            reordered.coordinate_functional_ids,
        )

    def test_support_and_material_coverage(self) -> None:
        """CMP-MESH-SUPPORT-001."""
        asset, case = models(2, 4, 4)
        mesh = compact_simplex_mesh(2, 4, two_cells=True)
        compiled = compile_problem_model(asset=asset, case=case, mesh=mesh)
        self.assertEqual(compiled.counts["material_cells"], mesh.cells.shape[0])
        self.assertGreater(compiled.counts["constrained_dofs"], 0)

        missing = replace(
            mesh,
            supports={
                name: support
                for name, support in mesh.supports.items()
                if name != "base"
            },
        )
        with self.assertRaises(CompileError) as caught:
            compile_problem_model(asset=asset, case=case, mesh=missing)
        self.assertEqual(caught.exception.code, "missing_support")

        ambiguous_case_raw = case_document(2, field_order=4, coordinate_order=4)
        ambiguous_case_raw["materials"].append(
            deepcopy(ambiguous_case_raw["materials"][0])
        )
        ambiguous_case_raw["materials"][1]["name"] = "duplicate-soil"
        ambiguous_case = load_document(
            stamped(ambiguous_case_raw), source_path="ambiguous-case.json"
        )
        with self.assertRaises(CompileError) as caught:
            compile_problem_model(asset=asset, case=ambiguous_case, mesh=mesh)
        self.assertEqual(caught.exception.code, "ambiguous_material_assignment")

        connectivity = mesh.coordinate_connectivity.copy()
        nodes = np.vstack(
            [mesh.coordinate_nodes, mesh.coordinate_nodes[connectivity[0, -1]]]
        )
        connectivity[1, -1] = nodes.shape[0] - 1
        inconsistent = replace(
            mesh,
            coordinate_nodes=nodes,
            coordinate_connectivity=connectivity,
        )
        with self.assertRaises(CompileError) as caught:
            compile_problem_model(asset=asset, case=case, mesh=inconsistent)
        self.assertEqual(caught.exception.code, "shared_coordinate_mismatch")

    def test_geometry_matrix(self) -> None:
        """CMP-GEO-MATRIX-001."""
        observed = set()
        for dimension in (2, 3):
            for field_order in (1, 2, 4, 8):
                for coordinate_order in (1, 2, 4):
                    with self.subTest(
                        dimension=dimension,
                        field_order=field_order,
                        coordinate_order=coordinate_order,
                    ):
                        asset, case = models(dimension, field_order, coordinate_order)
                        compiled = compile_problem_model(
                            asset=asset,
                            case=case,
                            mesh=compact_simplex_mesh(dimension, coordinate_order),
                        )
                        self.assertEqual(
                            compiled.spaces["displacement"]["order"], field_order
                        )
                        self.assertEqual(
                            compiled.spaces["coordinates"]["order"],
                            coordinate_order,
                        )
                        self.assertGreater(
                            compiled.geometry_audit.minimum_determinant, 0.0
                        )
                        observed.add((dimension, field_order, coordinate_order))
        self.assertEqual(len(observed), 24)

    def test_invalid_geometry(self) -> None:
        """CMP-GEO-INVALID-001."""
        base = compact_simplex_mesh(2, 4)
        variants = {}
        inverted_cells = base.cells.copy()
        inverted_cells[0, [0, 1]] = inverted_cells[0, [1, 0]]
        variants["inverted_geometry"] = replace(base, cells=inverted_cells)
        duplicate = base.coordinate_nodes.copy()
        duplicate[1] = duplicate[0]
        variants["duplicate_coordinate"] = replace(base, coordinate_nodes=duplicate)
        near = base.coordinate_nodes.copy()
        near[:, 1] *= 1.0e-16
        variants["near_singular_geometry"] = replace(base, coordinate_nodes=near)
        nonfinite = base.coordinate_nodes.copy()
        nonfinite[-1, 0] = np.nan
        variants["nonfinite_geometry"] = replace(base, coordinate_nodes=nonfinite)
        curved = base.coordinate_nodes.copy()
        curved[-1] = np.array([2.0, 2.0])
        variants["sign_changing_geometry"] = replace(base, coordinate_nodes=curved)
        for code, mesh in variants.items():
            with self.subTest(code=code):
                with self.assertRaises(CompileError) as caught:
                    audit_geometry(mesh)
                self.assertEqual(caught.exception.code, code)
                self.assertRegex(caught.exception.entity_id, r"^[0-9a-f]{16}$")

    def test_boundary_semantics(self) -> None:
        """CMP-BC-SEMANTIC-001."""
        for dimension in (2, 3):
            raw = case_document(dimension, field_order=4, coordinate_order=4)
            raw["physics"]["seepage"] = True
            raw["materials"].append(
                {
                    "name": "hydraulic-soil",
                    "support": "domain",
                    "model": "steady_unconfined_darcy",
                    "parameters": {
                        "conductivity": {"value": 1.0e-6, "unit": "m/s"},
                        "water_unit_weight": {
                            "value": 9.81,
                            "unit": "kN/m^3",
                        },
                    },
                }
            )
            raw["boundary_rules"].extend(
                [
                    {
                        "name": "nonzero-essential",
                        "field": "displacement",
                        "condition": "essential",
                        "support": "top",
                        "components": ["x"],
                        "value_model": "constant_vector",
                        "parameters": {"value": [0.0], "unit": "m"},
                    },
                    {
                        "name": "traction",
                        "field": "displacement",
                        "condition": "natural",
                        "support": "boundary",
                        "components": ["all"],
                        "value_model": "global_traction",
                        "parameters": {
                            "value": [0.0] * (dimension - 1) + [-1.0],
                            "unit": "kPa",
                        },
                        "geometry_patch": "full-coordinate-field",
                    },
                    {
                        "name": "normal-pressure",
                        "field": "displacement",
                        "condition": "natural",
                        "support": "top",
                        "components": ["normal"],
                        "value_model": "normal_pressure",
                        "parameters": {"value": 2.0, "unit": "kPa"},
                        "geometry_patch": "full-coordinate-field",
                    },
                    {
                        "name": "head",
                        "field": "pore_pressure",
                        "condition": "essential",
                        "support": "top",
                        "components": ["scalar"],
                        "value_model": "hydrostatic",
                        "parameters": {
                            "reference_level": 1.0,
                            "reference_pressure": 0.0,
                            "unit_weight": 9.81,
                            "length_unit": "m",
                            "pressure_unit": "kPa",
                        },
                    },
                    {
                        "name": "flux",
                        "field": "pore_pressure",
                        "condition": "natural",
                        "support": "boundary",
                        "components": ["normal"],
                        "value_model": "prescribed_flux",
                        "parameters": {"value": 1.0e-6, "unit": "m/s"},
                        "geometry_patch": "full-coordinate-field",
                    },
                ]
            )
            asset = load_document(
                asset_document(dimension, coordinate_order=4),
                source_path="asset.json",
            )
            case = load_document(stamped(raw), source_path="case.json")
            compiled = compile_problem_model(
                asset=asset,
                case=case,
                mesh=compact_simplex_mesh(dimension, 4),
            )
            self.assertEqual(len(compiled.boundary_rules), 6)
            for rule in compiled.boundary_rules:
                self.assertTrue(rule["entity_ids"])
                self.assertNotIn("coordinate_predicate", rule)
            curved = [
                rule
                for rule in compiled.boundary_rules
                if rule["condition"] == "natural"
            ]
            self.assertTrue(
                all(
                    rule["geometry_patch"] == "full-coordinate-field" for rule in curved
                )
            )
        self._assert_boundary_semantic_rejections()
        self._assert_equal_head_intersection()

    def _assert_boundary_semantic_rejections(self) -> None:
        asset = load_document(asset_document(), source_path="asset.json")
        mesh = compact_simplex_mesh(2, 1)
        invalid_cases: list[tuple[str, dict[str, object], str]] = []

        incompatible_model = case_document()
        incompatible_model["boundary_rules"][0].update(
            {
                "components": ["x"],
                "value_model": "constant_scalar",
                "parameters": {"value": 0.0, "unit": "kPa"},
            }
        )
        invalid_cases.append(
            (
                "value-model",
                incompatible_model,
                "boundary_value_model_incompatible",
            )
        )

        invalid_components = case_document()
        invalid_components["boundary_rules"][0]["components"] = ["z"]
        invalid_cases.append(
            (
                "components",
                invalid_components,
                "boundary_components_incompatible",
            )
        )

        invalid_dimension = case_document()
        invalid_dimension["boundary_rules"][0]["parameters"]["value"] = [0.0]
        invalid_cases.append(
            (
                "dimension",
                invalid_dimension,
                "boundary_parameter_dimension",
            )
        )

        missing_geometry = case_document()
        missing_geometry["boundary_rules"] = [
            {
                "name": "traction",
                "field": "displacement",
                "condition": "natural",
                "support": "boundary",
                "components": ["all"],
                "value_model": "global_traction",
                "parameters": {"value": [0.0, -1.0], "unit": "kPa"},
            }
        ]
        invalid_cases.append(
            (
                "geometry",
                missing_geometry,
                "missing_geometry_patch",
            )
        )

        disabled_physics = case_document(analysis="seepage.steady_unconfined")
        disabled_physics["physics"] = {
            "mechanics": False,
            "seepage": True,
            "coupling": "none",
        }
        disabled_physics["materials"] = [
            {
                "name": "hydraulic-soil",
                "support": "domain",
                "model": "steady_unconfined_darcy",
                "parameters": {
                    "conductivity": {"value": 1.0e-6, "unit": "m/s"},
                    "water_unit_weight": {"value": 9.81, "unit": "kN/m^3"},
                },
            }
        ]
        invalid_cases.append(
            (
                "physics",
                disabled_physics,
                "boundary_physics_incompatible",
            )
        )

        for name, raw, code in invalid_cases:
            with self.subTest(name=name):
                case = load_document(stamped(raw), source_path="case.json")
                with self.assertRaises(CompileError) as caught:
                    compile_problem_model(asset=asset, case=case, mesh=mesh)
                self.assertEqual(caught.exception.code, code)

    def _assert_equal_head_intersection(self) -> None:
        mesh = compact_simplex_mesh(2, 1)
        shared_id = int(
            mesh.topology_vertex_ids[np.argmax(mesh.topology_vertex_coordinates[:, 1])]
        )
        supports = dict(mesh.supports)
        supports["head-dry"] = EntitySet(
            name="head-dry",
            kind="node_set",
            dimension=0,
            entity_ids=(shared_id,),
        )
        supports["head-level"] = EntitySet(
            name="head-level",
            kind="node_set",
            dimension=0,
            entity_ids=(shared_id,),
        )
        supports["head-affine"] = EntitySet(
            name="head-affine",
            kind="node_set",
            dimension=0,
            entity_ids=(shared_id,),
        )
        mesh = replace(mesh, supports=supports)

        asset_raw = asset_document(2)
        asset_raw["supports"].extend(
            [
                {"name": "head-dry", "kind": "node_set", "dimension": 0},
                {"name": "head-level", "kind": "node_set", "dimension": 0},
                {"name": "head-affine", "kind": "node_set", "dimension": 0},
            ]
        )
        case_raw = case_document(2)
        case_raw["physics"]["seepage"] = True
        case_raw["materials"].append(
            {
                "name": "hydraulic-soil",
                "support": "domain",
                "model": "steady_unconfined_darcy",
                "parameters": {
                    "conductivity": {"value": 1.0e-6, "unit": "m/s"},
                    "water_unit_weight": {"value": 9.81, "unit": "kN/m^3"},
                },
            }
        )
        case_raw["boundary_rules"].extend(
            [
                {
                    "name": "dry",
                    "field": "pore_pressure",
                    "condition": "essential",
                    "support": "head-dry",
                    "components": ["scalar"],
                    "value_model": "constant_scalar",
                    "parameters": {"value": 0.0, "unit": "kPa"},
                },
                {
                    "name": "level",
                    "field": "pore_pressure",
                    "condition": "essential",
                    "support": "head-level",
                    "components": ["scalar"],
                    "value_model": "hydrostatic",
                    "parameters": {
                        "reference_level": 1.0,
                        "reference_pressure": 0.0,
                        "unit_weight": 9.81,
                        "length_unit": "m",
                        "pressure_unit": "kPa",
                    },
                },
                {
                    "name": "affine",
                    "field": "pore_pressure",
                    "condition": "essential",
                    "support": "head-affine",
                    "components": ["pressure"],
                    "value_model": "affine_coordinate",
                    "parameters": {
                        "offset": 9.81,
                        "gradient": [0.0, -9.81],
                        "unit": "kPa",
                    },
                },
            ]
        )
        asset = load_document(stamped(asset_raw), source_path="asset.json")
        case = load_document(stamped(case_raw), source_path="case.json")
        compile_problem_model(asset=asset, case=case, mesh=mesh)

        conflicting = deepcopy(case_raw)
        conflicting["boundary_rules"][-1]["parameters"]["offset"] = 10.81
        conflicting_case = load_document(
            stamped(conflicting),
            source_path="case.json",
        )
        with self.assertRaises(CompileError) as caught:
            compile_problem_model(
                asset=asset,
                case=conflicting_case,
                mesh=mesh,
            )
        self.assertEqual(
            caught.exception.code,
            "conflicting_essential_boundary",
        )
        self.assertIn("dry conflicts with affine", str(caught.exception))

    def test_single_payload_build(self) -> None:
        """CMP-PKG-SINGLE-WRITE-001 (PERF-13).

        A package is built once in its staging directory and verified once;
        a cache hit compares the freshly built bytes without re-verifying, and
        launch resolution hashes an unchanged payload once per process.
        """
        from slope_stability import resolution
        from slope_stability.compile import package as package_module

        asset, case = models(2, 2, 1)
        mesh = compact_simplex_mesh(2, 1)
        writes = unittest.mock.Mock(wraps=package_module._write_hdf5)
        verifications = unittest.mock.Mock(wraps=package_module._verify_hdf5)
        with tempfile.TemporaryDirectory(prefix="cmp-single-write-") as raw:
            with (
                unittest.mock.patch.object(package_module, "_write_hdf5", writes),
                unittest.mock.patch.object(
                    package_module, "_verify_hdf5", verifications
                ),
            ):
                first = compile_problem(
                    asset=asset, case=case, mesh=mesh, destination=Path(raw)
                )
                self.assertEqual((writes.call_count, verifications.call_count), (1, 1))
                again = compile_problem(
                    asset=asset, case=case, mesh=mesh, destination=Path(raw)
                )
                self.assertEqual((writes.call_count, verifications.call_count), (2, 2))
            self.assertEqual(first.path, again.path)
            self.assertEqual(
                sorted(item.name for item in Path(raw).iterdir()), [first.path.name]
            )
            hashes = unittest.mock.Mock(wraps=resolution.file_sha256)
            with unittest.mock.patch.object(resolution, "file_sha256", hashes):
                resolution._VERIFIED_PAYLOADS.clear()
                for _ in range(3):
                    resolution.load_compiled_problem(first.problem_json)
                self.assertEqual(hashes.call_count, 1)

    def test_package_determinism(self) -> None:
        """CMP-PKG-DETERMINISM-001."""
        asset, case = models(3, 4, 2)
        mesh = compact_simplex_mesh(3, 2)
        with tempfile.TemporaryDirectory(prefix="cmp-package-a-") as raw_a:
            with tempfile.TemporaryDirectory(prefix="cmp-package-b-") as raw_b:
                package_a = compile_problem(
                    asset=asset, case=case, mesh=mesh, destination=Path(raw_a)
                )
                package_b = compile_problem(
                    asset=asset, case=case, mesh=mesh, destination=Path(raw_b)
                )
                self.assertEqual(package_a.problem_id, package_b.problem_id)
                self.assertEqual(
                    package_a.hdf5_logical_digest,
                    package_b.hdf5_logical_digest,
                )
                self.assertEqual(
                    package_a.problem_json.read_bytes(),
                    package_b.problem_json.read_bytes(),
                )
                self.assertEqual(
                    package_a.compile_report.read_bytes(),
                    package_b.compile_report.read_bytes(),
                )
                self.assertEqual(
                    (package_a.path / "problem.h5").read_bytes(),
                    (package_b.path / "problem.h5").read_bytes(),
                )
                schema = json.loads(
                    (
                        ROOT / "schemas/generated/json/compiled-problem.schema.json"
                    ).read_text(encoding="utf-8")
                )
                assert_schema_subset(
                    json.loads(package_a.problem_json.read_bytes()), schema
                )
                manifest = json.loads(package_a.problem_json.read_bytes())
                payload_record = manifest["payloads"]["problem_hdf5"]
                self.assertEqual(
                    payload_record["sha256"],
                    file_sha(package_a.path / "problem.h5"),
                )
                self.assertEqual(
                    payload_record["byte_count"],
                    (package_a.path / "problem.h5").stat().st_size,
                )
                layout = json.loads(
                    (ROOT / "schemas/hdf5-layout/generated-layouts.json").read_text(
                        encoding="utf-8"
                    )
                )["layouts"][0]
                with h5py.File(package_a.path / "problem.h5", "r") as handle:
                    for group in layout["groups"]:
                        self.assertIn(group.removeprefix("/"), handle)
                    np.testing.assert_array_equal(
                        handle["mesh/topology/vertex_ids"],
                        mesh.topology_vertex_ids,
                    )
                    np.testing.assert_array_equal(
                        handle["mesh/topology/vertex_coordinates"],
                        mesh.topology_vertex_coordinates,
                    )
                    np.testing.assert_array_equal(
                        handle["mesh/topology/cell_ids"],
                        mesh.topology_cell_ids,
                    )
                    np.testing.assert_array_equal(
                        handle["mesh/topology/cells"],
                        mesh.topology_cells,
                    )
                    for dimension in range(mesh.dimension + 1):
                        np.testing.assert_array_equal(
                            handle[f"mesh/entities/dimension_{dimension}_vertex_ids"],
                            mesh.stable_entity_vertex_ids[dimension],
                        )
                    missing_attributes = []

                    def inspect(name, item):
                        if isinstance(item, h5py.Dataset):
                            missing = set(
                                layout["required_dataset_attributes"]
                            ).difference(item.attrs)
                            if missing:
                                missing_attributes.append((name, missing))

                    handle.visititems(inspect)
                    self.assertEqual(missing_attributes, [])
                mutation = mesh.coordinate_nodes.copy()
                mutation[-1, 0] += 1.0e-6
                package_c = compile_problem(
                    asset=asset,
                    case=case,
                    mesh=replace(mesh, coordinate_nodes=mutation),
                    destination=Path(raw_b) / "mutation",
                )
                self.assertNotEqual(package_a.problem_id, package_c.problem_id)

    def test_atomic_publication(self) -> None:
        """CMP-PKG-ATOMIC-001."""
        asset, case = models(2, 2, 2)
        mesh = compact_simplex_mesh(2, 2)
        for stage in (
            "after_hdf5",
            "after_payload_verification",
            "after_json",
            "before_rename",
        ):
            with self.subTest(stage=stage):
                with tempfile.TemporaryDirectory(prefix="cmp-atomic-") as raw:
                    target = Path(raw)
                    with self.assertRaises(CompileError) as caught:
                        compile_problem(
                            asset=asset,
                            case=case,
                            mesh=mesh,
                            destination=target,
                            failure_at=stage,
                        )
                    self.assertEqual(caught.exception.code, "injected_write_failure")
                    self.assertEqual(list(target.iterdir()), [])

    def test_new_problem_data_only(self) -> None:
        """CMP-E2E-NEWPROBLEM-001."""
        before = {str(path): file_sha(path) for path in source_tree_files(ROOT)}
        with tempfile.TemporaryDirectory(prefix="cmp-new-problem-") as raw:
            destination = Path(raw)
            packages = []
            for dimension in (2, 3):
                asset, case = models(dimension, 2, 4)
                packages.append(
                    compile_problem(
                        asset=asset,
                        case=case,
                        mesh=compact_simplex_mesh(dimension, 4),
                        destination=destination / f"{dimension}d",
                    )
                )
            self.assertEqual(len({item.problem_id for item in packages}), 2)
            self.assertTrue(all(item.path.is_dir() for item in packages))
        after = {str(path): file_sha(path) for path in source_tree_files(ROOT)}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
