# SPDX-License-Identifier: BSD-2-Clause
"""Strict lazy result-reader, comparator, and promotion contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import h5py
import numpy as np

from slope_stability.results import (
    CategoricalData,
    CategoricalTolerance,
    ConservationTolerance,
    FieldData,
    FieldTolerance,
    IncompleteResult,
    PromotionAuthority,
    PromotionError,
    QuantityUnavailableError,
    QuantityValue,
    ResultValidationError,
    ScalarTolerance,
    TrajectoryData,
    TrajectoryTolerance,
    align_fields,
    compare_categorical,
    compare_conservation,
    compare_field,
    compare_scalar,
    compare_trajectory,
    open_result,
    promote_reference,
)


NATIVE = Path(os.environ["SLOPE_RESULT_TEST_EXE"])


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh_payload_index(root: Path) -> None:
    envelope_path = root / "result.json"
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    payload = root / "result.h5"
    envelope["artifacts"]["result.h5"]["bytes"] = payload.stat().st_size
    envelope["artifacts"]["result.h5"]["sha256"] = file_sha(payload)
    envelope_path.write_text(
        json.dumps(
            envelope,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def native_result(root: Path, test_id: str = "OUT-UNIT-SCHEMA-001") -> Path:
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
    leaf = {
        "OUT-UNIT-SCHEMA-001": "schema-result",
        "OUT-UNIT-STATE-001": "state-result",
        "OUT-CONTRACT-TERMINAL-001": "terminal-succeeded",
    }[test_id]
    return root / leaf


class ResultContract(unittest.TestCase):
    maxDiff = None

    def test_open_and_integrity(self) -> None:
        """RDR-UNIT-OPEN-001."""
        with tempfile.TemporaryDirectory(prefix="rdr-open-") as raw:
            root = Path(raw)
            result_path = native_result(root / "native")
            result = open_result(result_path)
            self.assertEqual(result.status, "succeeded")
            self.assertTrue(result.complete)
            self.assertEqual(result.analysis.method, "direct_ssr")
            self.assertIsNone(result.restart)
            self.assertRegex(result.result_id, r"^sha256:[0-9a-f]{64}$")
            self.assertRegex(result.run_id, r"^sha256:[0-9a-f]{64}$")
            self.assertEqual(
                set(result.quantities),
                {
                    "displacement",
                    "plastic_fraction",
                    "pore_pressure",
                    "saturation",
                    "stress",
                },
            )
            self.assertTrue(result.verification.payload_verified)
            self.assertTrue(result.verification.logical_digests_verified)
            self.assertEqual(result.verification.layout_id, "result-v1")
            factor = result.scalar("stability_factor")
            self.assertEqual(factor.quantity_id, "stability_factor")
            self.assertEqual(factor.unit, "dimensionless")
            self.assertAlmostEqual(factor.value, 1.25)
            with self.assertRaises(QuantityUnavailableError):
                result.scalar("not_a_scalar")
            events = result.events()
            self.assertIn("result_pack", events.names)
            self.assertIn("result_hdf5", events.names)
            self.assertIn("result_digest", events.names)
            self.assertTrue(all(event.seconds >= 0.0 for event in events.rows))
            checkpoint = result.checkpoint("terminal")
            self.assertEqual(checkpoint.state_id, 17)
            self.assertRegex(checkpoint.checkpoint_id, r"^sha256:[0-9a-f]{64}$")
            self.assertEqual(checkpoint.relative_path.as_posix(), "checkpoints/17")
            result.close()

            corrupt = root / "corrupt"
            shutil.copytree(result_path, corrupt)
            payload = corrupt / "result.h5"
            with payload.open("r+b") as stream:
                stream.seek(-17, os.SEEK_END)
                original = stream.read(1)
                stream.seek(-1, os.SEEK_CUR)
                stream.write(bytes([original[0] ^ 0x01]))
            with self.assertRaisesRegex(ResultValidationError, "payload digest"):
                open_result(corrupt)

            environment_corrupt = root / "environment-corrupt"
            shutil.copytree(result_path, environment_corrupt)
            with (environment_corrupt / "environment.json").open(
                "a",
                encoding="utf-8",
            ) as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ResultValidationError, "environment digest"):
                open_result(environment_corrupt)

            identity_corrupt = root / "identity-corrupt"
            shutil.copytree(result_path, identity_corrupt)
            identity_envelope_path = identity_corrupt / "result.json"
            identity_envelope = json.loads(
                identity_envelope_path.read_text(encoding="utf-8")
            )
            identity_envelope["identity"]["result_id"] = "sha256:" + "a" * 64
            identity_envelope_path.write_text(
                json.dumps(
                    identity_envelope,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ResultValidationError, "identity fields"):
                open_result(identity_corrupt)

            linked_identity_corrupt = root / "linked-identity-corrupt"
            shutil.copytree(result_path, linked_identity_corrupt)
            with h5py.File(linked_identity_corrupt / "result.h5", "r+") as handle:
                handle.attrs["mesh_id"] = "sha256:" + "a" * 64
            refresh_payload_index(linked_identity_corrupt)
            with self.assertRaisesRegex(ResultValidationError, "identities disagree"):
                open_result(linked_identity_corrupt)

            logical_corrupt = root / "logical-corrupt"
            shutil.copytree(result_path, logical_corrupt)
            with h5py.File(logical_corrupt / "result.h5", "r+") as handle:
                dataset = handle["/history/attempt/rows"]
                row = dataset[0]
                row["parent_row_id"] = np.uint64(99)
                dataset[0] = row
            refresh_payload_index(logical_corrupt)
            with self.assertRaisesRegex(
                ResultValidationError,
                "history relational invariant",
            ):
                open_result(logical_corrupt)

            restart_path = native_result(
                root / "restart-native",
                "OUT-UNIT-STATE-001",
            )
            restart_result = open_result(restart_path)
            self.assertIsNotNone(restart_result.restart)
            self.assertEqual(restart_result.restart.mode, "accepted_boundary")
            self.assertEqual(restart_result.restart.parent_accepted_state_id, 17)
            self.assertEqual(restart_result.restart.source_index_bits, 32)
            restart_result.close()

            restart_corrupt = root / "restart-corrupt"
            shutil.copytree(restart_path, restart_corrupt)
            restart_envelope_path = restart_corrupt / "result.json"
            restart_envelope = json.loads(
                restart_envelope_path.read_text(encoding="utf-8")
            )
            restart_envelope["provenance"]["restart"]["source_index_bits"] = 64
            restart_envelope_path.write_text(
                json.dumps(
                    restart_envelope,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ResultValidationError,
                "logical digest mismatch",
            ):
                open_result(restart_corrupt)

            incomplete = root / "incomplete"
            incomplete.mkdir()
            partial = open_result(incomplete)
            self.assertIsInstance(partial, IncompleteResult)
            self.assertEqual(partial.status, "incomplete")
            with self.assertRaises(QuantityUnavailableError):
                partial.field("displacement", state=17)

    def test_lazy_selections(self) -> None:
        """RDR-UNIT-LAZY-001."""
        with tempfile.TemporaryDirectory(prefix="rdr-lazy-") as raw:
            result = open_result(native_result(Path(raw) / "native"))
            field = result.field("displacement", state=17)
            self.assertFalse(field.materialized)
            self.assertEqual(field.shape, (12, 2))
            selection = field.read(entity=slice(2, 5), components=("y",))
            self.assertEqual(selection.values.shape, (3, 1))
            self.assertEqual(selection.stable_ids.shape, (3,))
            self.assertEqual(selection.component_ids, ("y",))
            self.assertFalse(field.materialized)
            self.assertEqual(field.read_statistics.full_reads, 0)
            self.assertEqual(field.read_statistics.region_reads, 1)
            self.assertLess(field.read_statistics.value_bytes, 12 * 2 * 8)
            values = field.values()
            self.assertEqual(values.shape, (12, 2))
            self.assertTrue(field.materialized)
            self.assertEqual(field.read_statistics.full_reads, 1)
            earlier = result.field("displacement", state=11)
            self.assertEqual(earlier.shape, (12, 2))
            with h5py.File(result.path / "result.h5", "r") as handle:
                states = handle["states/index/rows"][...]
            self.assertIn("displacement_maximum", states.dtype.names)
            np.testing.assert_allclose(
                states["displacement_maximum"],
                [1.0, 1.5],
            )
            stress = result.quadrature("stress", state="terminal")
            self.assertEqual(stress.shape, (12, 3, 4))
            stress_selection = stress.read(
                entity=slice(1, 3),
                quadrature=slice(1, 3),
                components=("zz", "xy"),
            )
            self.assertEqual(stress_selection.values.shape, (2, 2, 2))
            self.assertEqual(stress_selection.component_ids, ("zz", "xy"))
            saturation = result.category("saturation", state="terminal")
            self.assertFalse(saturation.materialized)
            self.assertEqual(saturation.shape, (12, 3))
            self.assertEqual(
                saturation.category_ids,
                ("unsaturated", "saturated"),
            )
            saturation_selection = saturation.read(
                entity=slice(1, 4),
                quadrature=slice(1, 3),
            )
            self.assertEqual(saturation_selection.values.shape, (3, 2))
            self.assertEqual(saturation.read_statistics.full_reads, 0)
            reduction = result.cell_reduction(
                "plastic_fraction",
                state="terminal",
            )
            self.assertEqual(reduction.shape, (12, 1))
            self.assertEqual(
                reduction.read(entity=slice(0, 2)).values.shape,
                (2, 1),
            )
            accepted_history = result.history("accepted")
            self.assertFalse(accepted_history.materialized)
            self.assertEqual(accepted_history.shape, (2,))
            accepted_rows = accepted_history.read(slice(0, 1))
            self.assertEqual(accepted_rows.shape, (1,))
            self.assertEqual(int(accepted_rows[0]["state_id"]), 11)
            self.assertFalse(accepted_history.materialized)
            self.assertEqual(accepted_history.read_statistics.full_reads, 0)
            self.assertEqual(accepted_history.rows().shape, (2,))
            self.assertTrue(accepted_history.materialized)
            result.close()
            self.assertNotIn("petsc4py", os.sys.modules)

    def test_stable_identity_alignment(self) -> None:
        """RDR-UNIT-ALIGN-001."""
        reference = FieldData(
            quantity_id="displacement",
            quantity_version=1,
            unit="m",
            component_ids=("x", "y"),
            stable_ids=np.array([101, 202, 303], dtype=np.uint64),
            values=np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
        )
        candidate = FieldData(
            quantity_id="displacement",
            quantity_version=1,
            unit="m",
            component_ids=("y", "x"),
            stable_ids=np.array([303, 101, 202], dtype=np.uint64),
            values=np.array([[30.0, 3.0], [10.0, 1.0], [20.0, 2.0]]),
        )
        aligned = align_fields(candidate, reference)
        np.testing.assert_array_equal(aligned.stable_ids, reference.stable_ids)
        np.testing.assert_allclose(aligned.candidate, reference.values)
        np.testing.assert_allclose(aligned.reference, reference.values)
        self.assertEqual(aligned.component_ids, ("x", "y"))

        duplicate = FieldData(
            quantity_id="displacement",
            quantity_version=1,
            unit="m",
            component_ids=("x", "y"),
            stable_ids=np.array([101, 101], dtype=np.uint64),
            values=np.zeros((2, 2)),
        )
        with self.assertRaisesRegex(ResultValidationError, "duplicate stable"):
            align_fields(duplicate, reference)
        missing = FieldData(
            quantity_id="displacement",
            quantity_version=1,
            unit="m",
            component_ids=("x", "y"),
            stable_ids=np.array([101, 404], dtype=np.uint64),
            values=np.zeros((2, 2)),
        )
        with self.assertRaisesRegex(ResultValidationError, "stable identity"):
            align_fields(missing, reference)

    def test_scalar_comparator(self) -> None:
        """RDR-CMP-SCALAR-001."""
        tolerance = ScalarTolerance(absolute=2.0e-3, relative=1.0e-2)
        passed = compare_scalar(
            QuantityValue("settlement", 1001.0, "mm"),
            QuantityValue("settlement", 1.0, "m"),
            tolerance,
        )
        self.assertTrue(passed.passed)
        self.assertAlmostEqual(passed.candidate_in_reference_units, 1.001)
        self.assertAlmostEqual(passed.absolute_error, 0.001)
        self.assertAlmostEqual(passed.limit, 0.012)
        zero = compare_scalar(
            QuantityValue("balance", 1.9e-6, "dimensionless"),
            QuantityValue("balance", 0.0, "dimensionless"),
            ScalarTolerance(absolute=2.0e-6, relative=0.5),
        )
        self.assertTrue(zero.passed)
        self.assertAlmostEqual(zero.limit, 2.0e-6)
        failed = compare_scalar(
            QuantityValue("balance", 2.1e-6, "dimensionless"),
            QuantityValue("balance", 0.0, "dimensionless"),
            ScalarTolerance(absolute=2.0e-6, relative=0.5),
        )
        self.assertFalse(failed.passed)
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                report = compare_scalar(
                    QuantityValue("balance", value, "dimensionless"),
                    QuantityValue("balance", 0.0, "dimensionless"),
                    tolerance,
                )
                self.assertFalse(report.passed)
                self.assertEqual(report.reason, "nonfinite")

    def test_field_comparator(self) -> None:
        """RDR-CMP-FIELD-001."""
        ids = np.array([41, 7, 99], dtype=np.uint64)
        reference_values = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        errors = np.array([[0.1, -0.2], [-0.4, 0.3], [0.2, 0.5]])
        weights = np.array([1.0, 2.0, 1.0])
        reference = FieldData(
            "stress",
            1,
            "Pa",
            ("xx", "yy"),
            ids,
            reference_values,
            weights=weights,
        )
        permutation = np.array([2, 0, 1])
        candidate = FieldData(
            "stress",
            1,
            "Pa",
            ("yy", "xx"),
            ids[permutation],
            (reference_values + errors)[permutation][:, ::-1],
            weights=weights[permutation],
        )
        tolerance = FieldTolerance(absolute=0.6, relative=0.0, scale=10.0)
        report = compare_field(candidate, reference, tolerance)
        expected_l1 = np.sum(weights[:, None] * np.abs(errors), axis=0) / np.sum(
            weights
        )
        expected_l2 = np.sqrt(
            np.sum(weights[:, None] * errors**2, axis=0) / np.sum(weights)
        )
        np.testing.assert_allclose(report.component_l1, expected_l1)
        np.testing.assert_allclose(report.component_l2, expected_l2)
        np.testing.assert_allclose(report.component_linf, [0.4, 0.5])
        np.testing.assert_allclose(report.component_scaled, expected_l2 / 10.0)
        self.assertEqual(report.worst_stable_id, 99)
        self.assertEqual(report.worst_component_id, "yy")
        self.assertTrue(report.passed)
        categorical_reference = CategoricalData(
            quantity_id="saturation",
            quantity_version=1,
            unit="boolean",
            quadrature_id="triangle_degree_4_v1",
            category_ids=("unsaturated", "saturated"),
            stable_ids=np.array([101, 202, 303], dtype=np.uint64),
            values=np.array(
                [[0, 1], [1, 1], [0, 0]],
                dtype=np.uint32,
            ),
        )
        categorical_candidate = CategoricalData(
            quantity_id="saturation",
            quantity_version=1,
            unit="boolean",
            quadrature_id="triangle_degree_4_v1",
            category_ids=("saturated", "unsaturated"),
            stable_ids=np.array([303, 101, 202], dtype=np.uint64),
            values=np.array(
                [[1, 1], [1, 0], [0, 0]],
                dtype=np.uint32,
            ),
        )
        categorical = compare_categorical(
            categorical_candidate,
            categorical_reference,
            CategoricalTolerance(
                maximum_mismatch_fraction=0.0,
                minimum_weighted_jaccard=1.0,
            ),
            ambiguous=np.array(
                [[False, False], [False, False], [True, False]],
                dtype=np.bool_,
            ),
            weights=np.array([1.0, 2.0, 3.0]),
        )
        self.assertTrue(categorical.passed)
        self.assertEqual(categorical.excluded_count, 1)
        self.assertEqual(categorical.mismatch_count, 0)
        self.assertAlmostEqual(categorical.mismatch_fraction, 0.0)
        self.assertAlmostEqual(categorical.weighted_jaccard, 1.0)

    def test_conservation_comparator(self) -> None:
        """RDR-CMP-CONSERVATION-001."""
        report = compare_conservation(
            support_terms={"left_flux": -3.0, "right_flux": 1.5, "drain": 0.5},
            source_terms={"rain": 1.0002},
            physical_scale=10.0,
            tolerance=ConservationTolerance(normalized=3.0e-5),
        )
        self.assertEqual(
            report.support_terms,
            {"left_flux": -3.0, "right_flux": 1.5, "drain": 0.5},
        )
        self.assertEqual(report.source_terms, {"rain": 1.0002})
        self.assertAlmostEqual(report.numerator, 2.0e-4)
        self.assertAlmostEqual(report.denominator, 10.0)
        self.assertAlmostEqual(report.normalized_balance, 2.0e-5)
        self.assertTrue(report.passed)

    def test_trajectory_comparator(self) -> None:
        """RDR-CMP-TRAJECTORY-001."""
        reference = TrajectoryData(
            control=np.array([0.0, 0.5, 1.0, 1.5]),
            values=np.array([0.0, 0.4, 1.0, 1.8]),
            equilibrium=np.array([0.0, 1.0e-9, 2.0e-9, 3.0e-9]),
            terminal_reason="bracketed_failure",
        )
        candidate = TrajectoryData(
            control=np.array([0.0, 0.25, 0.75, 1.5]),
            values=np.array([0.0, 0.2, 0.7, 1.80001]),
            equilibrium=np.array([0.0, 1.0e-9, 2.0e-9, 3.0e-9]),
            terminal_reason="bracketed_failure",
        )
        report = compare_trajectory(
            candidate,
            reference,
            TrajectoryTolerance(
                pointwise=0.11,
                curve_distance=0.1,
                terminal_control=1.0e-6,
                equilibrium=1.0e-8,
            ),
        )
        self.assertTrue(report.monotone)
        self.assertEqual(report.common_control.tolist(), [0.0, 0.5, 1.0, 1.5])
        expected = np.interp(reference.control, candidate.control, candidate.values)
        np.testing.assert_allclose(report.candidate_interpolated, expected)
        expected_distance = math.sqrt(
            np.trapezoid((expected - reference.values) ** 2, reference.control)
            / (reference.control[-1] - reference.control[0])
        )
        self.assertAlmostEqual(report.curve_distance, expected_distance)
        self.assertTrue(report.terminal_match)
        self.assertTrue(report.equilibrium_passed)
        self.assertTrue(report.passed)
        nonmonotone = TrajectoryData(
            control=np.array([0.0, 1.0, 0.5]),
            values=np.array([0.0, 1.0, 0.5]),
            equilibrium=np.zeros(3),
            terminal_reason="bracketed_failure",
        )
        with self.assertRaisesRegex(ResultValidationError, "strictly monotone"):
            compare_trajectory(
                nonmonotone,
                reference,
                TrajectoryTolerance(1.0, 1.0, 1.0, 1.0),
            )

    def test_golden_promotion(self) -> None:
        """RDR-GOLD-PROMOTE-001."""
        with tempfile.TemporaryDirectory(prefix="rdr-promote-") as raw:
            root = Path(raw)
            candidate = native_result(root / "candidate")
            destination = root / "goldens" / "compact-v1"
            authority = PromotionAuthority(
                authority_id="independent-matlab-review-2026-07",
                evidence_digest="sha256:" + "a5" * 32,
                reviewer="fixture-reviewer",
                approved=True,
                known_defect_disposition="none",
            )
            record = promote_reference(candidate, destination, authority)
            self.assertEqual(record.destination, destination)
            self.assertRegex(record.dataset_id, r"^sha256:[0-9a-f]{64}$")
            self.assertEqual(
                file_sha(destination / "result.h5"),
                file_sha(candidate / "result.h5"),
            )
            self.assertTrue((destination / "promotion.json").is_file())
            with self.assertRaisesRegex(PromotionError, "already exists"):
                promote_reference(candidate, destination, authority)
            self_authority = PromotionAuthority(
                authority_id=open_result(candidate).result_id,
                evidence_digest=open_result(candidate).result_id,
                reviewer="engine",
                approved=True,
                known_defect_disposition="none",
            )
            with self.assertRaisesRegex(PromotionError, "independent"):
                promote_reference(candidate, root / "goldens" / "self", self_authority)


if __name__ == "__main__":
    unittest.main()
