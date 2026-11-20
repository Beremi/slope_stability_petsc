# SPDX-License-Identifier: BSD-2-Clause
"""Restart acceptance: same-layout replay versus repartitioned restart."""

from __future__ import annotations

import math
import unittest

import numpy as np

from slope_protocol import content_id
from slope_stability.restart import (
    RestartComparisonError,
    RestartObservation,
    _categorical_comparison,
    _float_comparison,
    assess_restart,
    required_restart_quantity_checks,
    required_restart_scientific_checks,
)


class RestartContract(unittest.TestCase):
    def test_same_and_repartitioned_restart_use_distinct_acceptance(self) -> None:
        """QAL-CONTRACT-RESTART-001."""
        categorical = _categorical_comparison(
            np.asarray([[1, 0]], dtype=np.uint32),
            np.asarray([[0, 0]], dtype=np.uint32),
            np.asarray([7], dtype=np.uint64),
        )
        self.assertEqual(categorical["mismatch_count"], 1)
        self.assertTrue(math.isfinite(categorical["jaccard"]))
        self.assertTrue(content_id(categorical).startswith("sha256:"))
        array_comparison = _float_comparison(
            np.asarray([math.inf], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            mode="repartitioned",
            relative_tolerance=1.0e-2,
        )
        self.assertEqual(array_comparison, (False, False, None, None))
        self.assertTrue(
            content_id(
                {
                    "relative_l2": array_comparison[2],
                    "absolute_maximum": array_comparison[3],
                }
            ).startswith("sha256:")
        )
        quantity_rows = (
            {"id": "displacement", "kind": "field", "passed": True},
            {
                "id": "material_branch",
                "kind": "quadrature_category",
                "passed": False,
            },
            {
                "id": "plastic_indicator",
                "kind": "quadrature_category",
                "passed": False,
            },
            {"id": "stress", "kind": "quadrature", "passed": True},
        )
        self.assertEqual(
            required_restart_quantity_checks(
                mode="same_layout",
                quantities=quantity_rows,
            ),
            (True, False, False, True),
        )
        self.assertEqual(
            required_restart_quantity_checks(
                mode="repartitioned",
                quantities=quantity_rows,
            ),
            (True, True),
        )
        self.assertEqual(
            required_restart_scientific_checks(
                mode="same_layout",
                state_index=True,
                scalar_index=False,
                accepted_history=False,
                quantities=(True, True),
            ),
            (True, False, False, True, True),
        )
        self.assertEqual(
            required_restart_scientific_checks(
                mode="repartitioned",
                state_index=True,
                scalar_index=False,
                accepted_history=False,
                quantities=(True, True),
            ),
            (True, True, True),
        )
        self.assertEqual(
            required_restart_scientific_checks(
                mode="repartitioned",
                state_index=True,
                scalar_index=True,
                accepted_history=True,
                quantities=(True, False),
            ),
            (True, True, False),
        )
        same = assess_restart(
            RestartObservation(
                case_id="CASE-003",
                variant_id="canonical",
                mode="same_layout",
                source_result_id="sha256:" + "5" * 64,
                resumed_result_id="sha256:" + "6" * 64,
                terminal_status_match=True,
                supports_match=True,
                bitwise_from_restart=True,
                quantity_comparisons=(True, True, True),
            )
        )
        repartitioned = assess_restart(
            RestartObservation(
                case_id="CASE-003",
                variant_id="canonical",
                mode="repartitioned",
                source_result_id="sha256:" + "5" * 64,
                resumed_result_id="sha256:" + "7" * 64,
                terminal_status_match=True,
                supports_match=True,
                bitwise_from_restart=False,
                quantity_comparisons=(True, True, True),
            )
        )
        self.assertTrue(same.passed)
        self.assertTrue(repartitioned.passed)
        self.assertFalse(
            assess_restart(
                RestartObservation(
                    case_id="CASE-003",
                    variant_id="canonical",
                    mode="same_layout",
                    source_result_id="sha256:" + "5" * 64,
                    resumed_result_id="sha256:" + "8" * 64,
                    terminal_status_match=True,
                    supports_match=True,
                    bitwise_from_restart=False,
                    quantity_comparisons=(True, True, True),
                )
            ).passed
        )
        self.assertFalse(
            assess_restart(
                RestartObservation(
                    case_id="CASE-003",
                    variant_id="canonical",
                    mode="repartitioned",
                    source_result_id="sha256:" + "5" * 64,
                    resumed_result_id="sha256:" + "9" * 64,
                    terminal_status_match=True,
                    supports_match=True,
                    bitwise_from_restart=False,
                    quantity_comparisons=(True, False, True),
                )
            ).passed
        )

    def test_invalid_observations_are_rejected(self) -> None:
        valid = RestartObservation(
            case_id="CASE-003",
            variant_id="canonical",
            mode="same_layout",
            source_result_id="sha256:" + "5" * 64,
            resumed_result_id="sha256:" + "6" * 64,
            terminal_status_match=True,
            supports_match=True,
            bitwise_from_restart=True,
            quantity_comparisons=(True,),
        )
        self.assertTrue(assess_restart(valid).passed)
        for field, value in (
            ("case_id", ""),
            ("mode", "unknown"),
            ("source_result_id", "sha256:short"),
            ("quantity_comparisons", ()),
            ("quantity_comparisons", (1,)),
        ):
            with self.subTest(field=field), self.assertRaises(RestartComparisonError):
                assess_restart(RestartObservation(**{**valid.__dict__, field: value}))
        with self.assertRaises(RestartComparisonError):
            required_restart_quantity_checks(mode="other", quantities=())  # type: ignore
