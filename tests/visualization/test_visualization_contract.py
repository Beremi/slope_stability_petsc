# SPDX-License-Identifier: BSD-2-Clause
"""Derived-only high-order visualization and report contracts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from slope_stability.results import QuantityUnavailableError, open_result
from slope_stability.visualization import (
    SimplexVisualizationData,
    VisualizationError,
    canonical_simplex_nodes,
    canonical_to_vtk_permutation,
    export_visualization,
    project_quadrature,
    read_vtu,
    read_xdmf,
    write_report,
)


NATIVE = Path(os.environ["SLOPE_RESULT_TEST_EXE"])


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def native_artifacts(root: Path, test_id: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    output = root / "native-outcome.json"
    command = [
        str(NATIVE),
        "--test-id",
        test_id,
        "--artifact-root",
        str(root),
        "--output",
        str(output),
        "--expected-ranks",
        "1",
        "--expected-threads",
        "1",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "OMP_MAX_ACTIVE_LEVELS": "1",
            "OMP_PROC_BIND": "false",
            "SLOPE_RESULT_DATASET_CHUNK_ROWS": "0",
        }
    )
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"native fixture failed: {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return root


def physical_map(reference: np.ndarray, geometry_order: int) -> np.ndarray:
    dimension = reference.shape[1]
    padded = np.zeros((reference.shape[0], 3), dtype=np.float64)
    padded[:, :dimension] = reference
    if geometry_order >= 2:
        r = reference[:, 0]
        s = reference[:, 1]
        padded[:, 0] += 0.08 * r * s
        padded[:, 1] += 0.06 * r * s
        if dimension == 3:
            t = reference[:, 2]
            padded[:, 0] += 0.03 * s * t
            padded[:, 1] += 0.02 * r * t
            padded[:, 2] += 0.05 * r * t + 0.04 * s * t
    if geometry_order == 4:
        r = reference[:, 0]
        s = reference[:, 1]
        first = 1.0 - np.sum(reference, axis=1)
        padded[:, 0] += 0.02 * r * s * first**2
        padded[:, 1] -= 0.015 * r**2 * s * first
        if dimension == 3:
            t = reference[:, 2]
            padded[:, 2] += 0.01 * r * s * t * first
    return padded


def simplex_data(
    dimension: int, field_order: int, geometry_order: int
) -> SimplexVisualizationData:
    field_reference = canonical_simplex_nodes(dimension, field_order)
    geometry_reference = canonical_simplex_nodes(dimension, geometry_order)
    field_points = physical_map(field_reference, geometry_order)
    geometry_points = physical_map(geometry_reference, geometry_order)
    values = np.column_stack(
        (
            1.0 + 2.0 * field_points[:, 0] - 3.0 * field_points[:, 1],
            field_points[:, 0] + field_points[:, 1] + field_points[:, 2],
        )
    )
    stable_ids = np.arange(7000, 7000 + len(field_points), dtype=np.uint64)
    geometry_ids = np.arange(9000, 9000 + len(geometry_points), dtype=np.uint64)
    return SimplexVisualizationData(
        dimension=dimension,
        field_order=field_order,
        geometry_order=geometry_order,
        field_points=field_points,
        geometry_points=geometry_points,
        field_stable_ids=stable_ids,
        geometry_stable_ids=geometry_ids,
        values=values,
        quantity_id="displacement",
        quantity_version=1,
        component_ids=("x", "y"),
        unit="m",
        state_id=17,
        source_result_id="sha256:" + "4d" * 32,
    )


class VisualizationContract(unittest.TestCase):
    maxDiff = None

    def test_derived_artifacts_do_not_mutate_source(self) -> None:
        """VIS-UNIT-DERIVED-001."""
        with tempfile.TemporaryDirectory(prefix="vis-derived-") as raw:
            root = Path(raw)
            result_path = (
                native_artifacts(root / "native", "OUT-UNIT-SCHEMA-001")
                / "schema-result"
            )
            before = tree_digest(result_path)
            result = open_result(result_path)
            report = write_report(result, root / "derived" / "report.html")
            self.assertTrue(report.is_file())
            self.assertFalse(report.is_relative_to(result_path))
            self.assertEqual(tree_digest(result_path), before)
            manifest = json.loads(
                report.with_suffix(".manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["source_result_id"], result.result_id)
            self.assertEqual(manifest["authority"], "derived")
            self.assertNotIn(str(result_path.resolve()), report.read_text("utf-8"))

    def test_vtk_permutations(self) -> None:
        """VIS-UNIT-PERMUTE-001."""
        expected = {
            (2, 1): (0, 1, 2),
            (2, 2): (0, 1, 2, 3, 4, 5),
            (2, 4): (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 10, 9, 12, 13, 14),
            (3, 1): (0, 1, 2, 3),
            (3, 2): (0, 1, 2, 3, 4, 5, 6, 8, 9, 7),
            (3, 4): (
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                12,
                11,
                10,
                16,
                17,
                18,
                19,
                20,
                21,
                13,
                14,
                15,
                31,
                33,
                32,
                22,
                23,
                24,
                28,
                30,
                29,
                27,
                25,
                26,
                34,
            ),
        }
        for key, permutation in expected.items():
            with self.subTest(dimension=key[0], order=key[1]):
                actual = canonical_to_vtk_permutation(*key)
                self.assertEqual(actual, permutation)
                self.assertEqual(sorted(actual), list(range(len(actual))))
                nodes = canonical_simplex_nodes(*key)
                external = np.empty_like(nodes)
                external[list(actual)] = nodes
                for canonical, vtk_index in enumerate(actual):
                    np.testing.assert_allclose(external[vtk_index], nodes[canonical])

    def test_named_projection(self) -> None:
        """VIS-UNIT-PROJECTION-001."""
        values = np.array([[10.0, 2.0], [14.0, 4.0], [20.0, 8.0]], dtype=np.float64)
        weights = np.array([0.2, 0.3, 0.5], dtype=np.float64)
        projected = project_quadrature(
            values,
            weights,
            rule="weighted_cell_average_v1",
            unit="Pa",
        )
        np.testing.assert_allclose(
            projected.values, np.sum(values * weights[:, None], axis=0)
        )
        self.assertEqual(projected.rule, "weighted_cell_average_v1")
        self.assertEqual(projected.unit, "Pa")
        self.assertGreater(projected.error_estimate, 0.0)
        self.assertEqual(projected.source_location, "quadrature")
        self.assertEqual(projected.target_location, "cell")
        with self.assertRaisesRegex(VisualizationError, "named projection"):
            project_quadrature(values, weights, rule="", unit="Pa")

    def assert_roundtrip(self, dimension: int) -> None:
        with tempfile.TemporaryDirectory(prefix=f"vis-{dimension}d-") as raw:
            root = Path(raw)
            for field_order in (1, 2, 4, 8):
                for geometry_order in (1, 2, 4):
                    with self.subTest(
                        dimension=dimension,
                        field_order=field_order,
                        geometry_order=geometry_order,
                    ):
                        data = simplex_data(dimension, field_order, geometry_order)
                        directory = root / f"p{field_order}-g{geometry_order}"
                        bundle = export_visualization(data, directory)
                        vtk = read_vtu(bundle.vtu)
                        xdmf = read_xdmf(bundle.xdmf)
                        for observed in (vtk, xdmf):
                            self.assertEqual(observed.dimension, dimension)
                            self.assertEqual(observed.field_order, field_order)
                            self.assertEqual(observed.geometry_order, geometry_order)
                            self.assertEqual(observed.quantity_id, data.quantity_id)
                            self.assertEqual(observed.unit, "m")
                            np.testing.assert_array_equal(
                                observed.field_stable_ids,
                                data.field_stable_ids,
                            )
                            np.testing.assert_array_equal(
                                observed.geometry_stable_ids,
                                data.geometry_stable_ids,
                            )
                            np.testing.assert_allclose(
                                observed.field_points,
                                data.field_points,
                                rtol=0.0,
                                atol=2.0e-15,
                            )
                            np.testing.assert_allclose(
                                observed.geometry_points,
                                data.geometry_points,
                                rtol=0.0,
                                atol=2.0e-15,
                            )
                            np.testing.assert_allclose(
                                observed.values,
                                data.values,
                                rtol=0.0,
                                atol=2.0e-15,
                            )
                        manifest = json.loads(
                            bundle.manifest.read_text(encoding="utf-8")
                        )
                        self.assertEqual(manifest["authority"], "derived")
                        self.assertEqual(
                            manifest["source_result_id"], data.source_result_id
                        )
                        self.assertEqual(
                            manifest["transform"]["topology"],
                            "vtk_lagrange" if field_order >= 4 else "vtk_standard",
                        )
                        self.assertFalse(manifest["transform"]["linearized"])
                        if dimension == 2:
                            expected_type = {1: 5, 2: 22, 4: 69, 8: 69}[field_order]
                        else:
                            expected_type = {1: 10, 2: 24, 4: 71, 8: 71}[field_order]
                        self.assertEqual(manifest["vtk_cell_type"], expected_type)

    def test_roundtrip_2d(self) -> None:
        """VIS-ROUNDTRIP-2D-001."""
        self.assert_roundtrip(2)

    def test_roundtrip_3d(self) -> None:
        """VIS-ROUNDTRIP-3D-001."""
        self.assert_roundtrip(3)

    def test_report_labels_and_evidence_status(self) -> None:
        """VIS-UNIT-REPORT-001."""
        with tempfile.TemporaryDirectory(prefix="vis-report-") as raw:
            root = Path(raw)
            result_path = (
                native_artifacts(root / "native", "OUT-UNIT-SCHEMA-001")
                / "schema-result"
            )
            result = open_result(result_path)
            report = write_report(
                result,
                root / "derived" / "qualification.html",
                evidence_status="candidate",
            )
            text = report.read_text(encoding="utf-8")
            for expected in (
                "direct_ssr",
                "succeeded",
                result.result_id,
                result.build_id,
                "candidate",
                "m",
                "Pa",
            ):
                self.assertIn(expected, text)
            self.assertNotIn("accepted golden", text.lower())
            manifest = json.loads(
                report.with_suffix(".manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["evidence_status"], "candidate")

    def test_incomplete_and_failed_are_unavailable(self) -> None:
        """VIS-NEG-INCOMPLETE-001."""
        with tempfile.TemporaryDirectory(prefix="vis-incomplete-") as raw:
            root = Path(raw)
            incomplete_path = root / "incomplete"
            incomplete_path.mkdir()
            incomplete = open_result(incomplete_path)
            report = write_report(incomplete, root / "derived" / "incomplete.html")
            text = report.read_text(encoding="utf-8").lower()
            self.assertIn("incomplete", text)
            self.assertIn("quantities unavailable", text)
            self.assertNotIn(">0.0<", text)
            with self.assertRaises(QuantityUnavailableError):
                incomplete.field("displacement", state="terminal")

            terminal_root = native_artifacts(
                root / "terminal", "OUT-CONTRACT-TERMINAL-001"
            )
            failed = open_result(terminal_root / "terminal-failed_numerical")
            self.assertEqual(failed.status, "failed_numerical")
            with self.assertRaisesRegex(
                QuantityUnavailableError, "terminal solution unavailable"
            ):
                failed.field("displacement", state="terminal")
            failed_report = write_report(failed, root / "derived" / "failed.html")
            failed_text = failed_report.read_text(encoding="utf-8").lower()
            self.assertIn("failed", failed_text)
            self.assertIn("last accepted state", failed_text)
            self.assertNotIn("successful solution", failed_text)


if __name__ == "__main__":
    unittest.main()
