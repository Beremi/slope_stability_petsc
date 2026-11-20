# SPDX-License-Identifier: BSD-2-Clause
"""Executable P-AUTHOR authoring contracts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/src"))

from slope_stability.authoring import (
    AuthoringError,
    Quantity,
    RegistryError,
    SourceSpan,
    UnitError,
    UnitSystem,
    authoring_identity,
    default_model_registry,
    load_document,
    render_document,
    validate_documents,
)
from tests.support.problem_fixtures import (
    asset_document,
    assert_schema_subset,
    case_document,
    profile_documents,
    stamped,
    suite_document,
)


class AuthoringContract(unittest.TestCase):
    def test_strict_types_and_immutability(self) -> None:
        """AUT-UNIT-TYPES-001."""
        asset = load_document(asset_document(), source_path="asset.json")
        self.assertEqual(asset.dimension, 2)
        with self.assertRaises(ValidationError):
            asset.dimension = 3

        wrong_type = asset_document()
        wrong_type["dimension"] = "2"
        wrong_type = stamped(wrong_type)
        with self.assertRaises(ValidationError):
            load_document(wrong_type, source_path="wrong-type.json")

        extra = asset_document()
        extra["unknown"] = True
        extra = stamped(extra)
        with self.assertRaises(ValidationError):
            load_document(extra, source_path="extra.json")

        p3 = case_document(field_order=3)
        with self.assertRaises(ValidationError):
            load_document(p3, source_path="p3.json")
        p8 = load_document(case_document(field_order=8), source_path="p8.json")
        self.assertEqual(p8.discretization.field_order, 8)

        fcg_profile = profile_documents()[2]
        fcg_profile["policy"]["ksp_type"] = "fcg"
        fcg_profile["policy"]["stopping_condition"] = (
            "preconditioned_residual_original_rhs"
        )
        parsed_fcg = load_document(stamped(fcg_profile), source_path="fcg.json")
        self.assertEqual(
            parsed_fcg.policy["stopping_condition"],
            "preconditioned_residual_original_rhs",
        )
        fcg_profile["policy"]["stopping_condition"] = "natural_residual_initial"
        parsed_natural_fcg = load_document(
            stamped(fcg_profile), source_path="natural-fcg.json"
        )
        self.assertEqual(
            parsed_natural_fcg.policy["stopping_condition"],
            "natural_residual_initial",
        )
        invalid_stopping = profile_documents()[2]
        invalid_stopping["policy"]["stopping_condition"] = "preconditioned_rhs"
        with self.assertRaises(ValidationError):
            load_document(stamped(invalid_stopping), source_path="invalid-fcg.json")

        case_schema = json.loads(
            (ROOT / "schemas/generated/json/case.schema.json").read_text(
                encoding="utf-8"
            )
        )
        assert_schema_subset(json.loads(render_document(p8)), case_schema)

        traversal = asset_document()
        traversal["mesh_source"]["path"] = "../outside.msh"
        with self.assertRaises(ValidationError):
            load_document(stamped(traversal), source_path="traversal.json")

        case = load_document(case_document(), source_path="case.json")
        with self.assertRaises(ValidationError):
            case.discretization.field_order = 2
        self.assertIsInstance(case.boundary_rules, tuple)
        with self.assertRaises(TypeError):
            case.materials[0].parameters["cohesion"] = {
                "value": 7.0,
                "unit": "kPa",
            }
        profile = load_document(profile_documents()[0])
        with self.assertRaises(TypeError):
            profile.policy["initial_step"] = 0.2

        for required_outputs in (
            ["displacement", "unknown"],
            ["displacement", "displacement"],
        ):
            invalid_case = case_document()
            invalid_case["required_outputs"] = required_outputs
            with self.subTest(required_outputs=required_outputs):
                with self.assertRaises(ValidationError):
                    load_document(
                        stamped(invalid_case),
                        source_path="case-invalid-outputs.json",
                    )

        pmg = next(item for item in profile_documents() if item["family"] == "pmg")
        pmg["policy"]["smoother"] = "none"
        with self.assertRaises(ValidationError):
            load_document(stamped(pmg), source_path="pmg-enabled-none.json")
        pmg["policy"]["enabled"] = False
        disabled_pmg = load_document(
            stamped(pmg),
            source_path="pmg-disabled-none.json",
        )
        self.assertFalse(disabled_pmg.policy["enabled"])

        for smoother in (
            "chebyshev_point",
            "fgmres_block",
            "richardson_block",
        ):
            smoother_pmg = next(
                item for item in profile_documents() if item["family"] == "pmg"
            )
            smoother_pmg["policy"]["smoother"] = smoother
            with self.subTest(smoother=smoother):
                parsed_pmg = load_document(
                    stamped(smoother_pmg),
                    source_path=f"pmg-{smoother}.json",
                )
                self.assertEqual(parsed_pmg.policy["smoother"], smoother)

        redundant_gamg = next(
            item for item in profile_documents() if item["family"] == "pmg"
        )
        redundant_gamg["policy"]["coarse_solver"] = "gamg"
        redundant_gamg["policy"]["coarse_redundant_group_size"] = 1
        with self.assertRaises(ValidationError):
            load_document(
                stamped(redundant_gamg), source_path="pmg-redundant-gamg.json"
            )

        legacy = profile_documents()[4]
        parsed = load_document(legacy, source_path="deflation-legacy.json")
        self.assertNotIn("candidate_source", parsed.policy)
        self.assertNotIn("admission_tolerance", parsed.policy)

        explicit = profile_documents()[4]
        explicit["policy"].update(
            {
                "candidate_source": ("accepted_newton_and_control_directions"),
                "recycle": "accepted_steps",
                "admission_tolerance": 1.0e-3,
                "pivot_tolerance": 1.0e-13,
                "condition_limit": 1.0e12,
                "invariant_tolerance": 1.0e-9,
                "reorthogonalization_passes": 2,
                "dense_factor": "partial_pivot_lu",
            }
        )
        parsed = load_document(
            stamped(explicit),
            source_path="deflation-explicit.json",
        )
        self.assertEqual(parsed.policy["reorthogonalization_passes"], 2)

        accepted_states = profile_documents()[4]
        accepted_states["policy"].update(
            {
                "projector": "legacy_a_orthogonal",
                "candidate_source": "accepted_state_vectors",
                "recycle": "attempt_snapshot_restore",
            }
        )
        parsed = load_document(
            stamped(accepted_states),
            source_path="deflation-accepted-states.json",
        )
        self.assertEqual(parsed.policy["projector"], "legacy_a_orthogonal")
        self.assertEqual(parsed.policy["candidate_source"], "accepted_state_vectors")

        full_krylov = profile_documents()[4]
        full_krylov["policy"].update(
            {
                "projector": "legacy_a_orthogonal",
                "candidate_source": "accepted_states_and_full_krylov",
                "recycle": "attempt_snapshot_restore",
                "maximum_dimension": 5000,
                "compression": "snapshot_pod",
                "compression_dimension": 24,
                "compression_history_decay": 0.5,
            }
        )
        parsed = load_document(
            stamped(full_krylov),
            source_path="deflation-full-krylov.json",
        )
        self.assertEqual(
            parsed.policy["candidate_source"],
            "accepted_states_and_full_krylov",
        )
        self.assertEqual(parsed.policy["maximum_dimension"], 5000)
        self.assertEqual(parsed.policy["compression"], "snapshot_pod")
        self.assertEqual(parsed.policy["compression_dimension"], 24)
        self.assertEqual(parsed.policy["compression_history_decay"], 0.5)

        total_pod = profile_documents()[4]
        total_pod["policy"].update(
            {
                "projector": "legacy_a_orthogonal",
                "candidate_source": "accepted_states_and_full_krylov",
                "recycle": "attempt_snapshot_restore",
                "maximum_dimension": 64,
                "compression": "snapshot_pod_total",
                "compression_dimension": 12,
            }
        )
        parsed = load_document(
            stamped(total_pod), source_path="deflation-total-pod.json"
        )
        self.assertEqual(parsed.policy["compression"], "snapshot_pod_total")

        rolling = profile_documents()[4]
        rolling["policy"].update(
            {
                "projector": "legacy_a_orthogonal",
                "candidate_source": "accepted_states_and_system_solutions",
                "recycle": "attempt_snapshot_restore",
                "maximum_dimension": 64,
                "compression": "system_solution_window",
                "compression_dimension": 12,
            }
        )
        parsed = load_document(stamped(rolling), source_path="deflation-rolling.json")
        self.assertEqual(
            parsed.policy["candidate_source"],
            "accepted_states_and_system_solutions",
        )
        self.assertEqual(parsed.policy["compression"], "system_solution_window")

        invalid_compression = profile_documents()[4]
        invalid_compression["policy"].update(
            {
                "projector": "legacy_a_orthogonal",
                "candidate_source": "accepted_states_and_full_krylov",
                "recycle": "attempt_snapshot_restore",
                "maximum_dimension": 16,
                "compression": "operator_svd",
                "compression_dimension": 17,
            }
        )
        with self.assertRaises(ValidationError):
            load_document(
                stamped(invalid_compression),
                source_path="deflation-invalid-compression.json",
            )

        mismatched_lifecycle = profile_documents()[4]
        mismatched_lifecycle["policy"].update(
            {
                "candidate_source": "accepted_state_vectors",
                "recycle": "accepted_steps",
            }
        )
        with self.assertRaises(ValidationError):
            load_document(
                stamped(mismatched_lifecycle),
                source_path="deflation-mismatched-lifecycle.json",
            )

        invalid_fields = {
            "candidate_source": "all_krylov_vectors",
            "recycle": "failed_steps",
            "admission_tolerance": 0.0,
            "pivot_tolerance": -1.0,
            "condition_limit": 1.0,
            "invariant_tolerance": 0.0,
            "reorthogonalization_passes": 5,
            "dense_factor": "inverse",
            "compression": "randomized_guess",
            "compression_dimension": -1,
            "compression_history_decay": 1.1,
        }
        for field, value in invalid_fields.items():
            document = profile_documents()[4]
            document["policy"][field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValidationError):
                    load_document(
                        stamped(document),
                        source_path=f"deflation-invalid-{field}.json",
                    )

        for field, value in (
            ("frequency", "every_iteration"),
            ("frequency", 1),
            ("history", "true"),
            ("quantities", ["displacement", "unknown"]),
            ("quantities", ["displacement", "displacement"]),
        ):
            document = profile_documents()[5]
            document["policy"][field] = value
            with self.subTest(output_field=field, value=value):
                with self.assertRaises(ValidationError):
                    load_document(
                        stamped(document),
                        source_path=f"output-invalid-{field}.json",
                    )

    def test_units(self) -> None:
        """AUT-UNIT-UNITS-001."""
        units = UnitSystem()
        stress = units.to_internal(
            Quantity(value=1.25, unit="MPa"),
            expected_dimension="stress",
            compilation_boundary=True,
        )
        self.assertEqual(stress.internal_unit, "kPa")
        self.assertAlmostEqual(stress.converted_value, 1250.0)
        self.assertEqual(stress.original_unit, "MPa")
        self.assertEqual(stress.scale, 1000.0)

        angle = units.to_internal(
            Quantity(value=180.0, unit="degree"),
            expected_dimension="angle",
            compilation_boundary=True,
        )
        self.assertAlmostEqual(angle.converted_value, 3.141592653589793)
        self.assertEqual(angle.internal_unit, "radian")

        conductivity = units.to_internal(
            Quantity(value=2.5, unit="m/day"),
            expected_dimension="conductivity",
            compilation_boundary=True,
        )
        self.assertAlmostEqual(conductivity.converted_value, 2.5 / 86400.0)
        for quantity, expected, code in (
            (Quantity(value=1.0, unit=""), "stress", "missing_unit"),
            (Quantity(value=1.0, unit="m"), "stress", "incompatible_unit"),
            (Quantity(value=1.0, unit="legacy"), "length", "unknown_unit"),
        ):
            with self.subTest(quantity=quantity, expected=expected):
                with self.assertRaises(UnitError) as caught:
                    units.to_internal(
                        quantity,
                        expected_dimension=expected,
                        compilation_boundary=True,
                    )
                self.assertEqual(caught.exception.code, code)

    def test_spans_and_accumulated_validation(self) -> None:
        """AUT-UNIT-SPANS-001."""
        asset_raw = asset_document()
        asset_raw["supports"] = [
            item for item in asset_raw["supports"] if item["name"] != "base"
        ]
        asset_raw = stamped(asset_raw)
        case_raw = case_document()
        case_raw["materials"][0]["support"] = "missing-region"
        case_raw["boundary_rules"][0]["support"] = "missing-base"
        case_raw = stamped(case_raw)
        spans = {
            "/materials/0/support": SourceSpan(
                path="case.toml", line=21, column=11, end_line=21, end_column=25
            ),
            "/boundary_rules/0/support": SourceSpan(
                path="case.toml", line=34, column=11, end_line=34, end_column=23
            ),
        }
        asset = load_document(asset_raw, source_path="asset.toml")
        case = load_document(case_raw, source_path="case.toml", spans=spans)
        report = validate_documents(
            asset=asset,
            case=case,
            profiles=[
                load_document(item, source_path="profiles.json")
                for item in profile_documents()
            ],
        )
        self.assertFalse(report.valid)
        self.assertEqual(
            [(issue.code, issue.pointer) for issue in report.issues],
            [
                ("unresolved_material_support", "/materials/0/support"),
                ("unresolved_boundary_support", "/boundary_rules/0/support"),
            ],
        )
        self.assertEqual(report.issues[0].span.path, "case.toml")
        self.assertEqual(report.issues[0].span.line, 21)
        self.assertEqual(report.issues[0].object_id, case.identity.name)
        self.assertTrue(report.issues[0].remediation)

    def test_registered_models(self) -> None:
        """AUT-UNIT-REGISTRY-001."""
        registry = default_model_registry()
        valid = {
            "constant_scalar": {"value": 1.0, "unit": "kPa"},
            "constant_vector": {"value": [0.0, 1.0], "unit": "m"},
            "affine_coordinate": {
                "offset": 1.0,
                "gradient": [2.0, 0.0],
                "unit": "kPa",
            },
            "hydrostatic": {
                "reference_level": 10.0,
                "reference_pressure": 0.0,
                "unit_weight": 9.81,
                "length_unit": "m",
                "pressure_unit": "kPa",
            },
            "piecewise_linear_level": {
                "coordinates": [0.0, 1.0],
                "values": [2.0, 3.0],
                "coordinate_unit": "m",
                "value_unit": "m",
            },
            "global_traction": {"value": [0.0, -1.0], "unit": "kPa"},
            "spatial_global_traction": {
                "offset": [0.0, -1.0],
                "gradient": [[0.0, 0.0], [0.0, 0.0]],
                "unit": "kPa",
            },
            "normal_pressure": {"value": 1.0, "unit": "kPa"},
            "prescribed_flux": {"value": 1.0e-6, "unit": "m/s"},
            "mohr_coulomb": {
                "cohesion": {"value": 6.0, "unit": "kPa"},
                "friction_angle": {"value": 30.0, "unit": "degree"},
                "dilation_angle": {"value": 5.0, "unit": "degree"},
                "young_modulus": {"value": 40000.0, "unit": "kPa"},
                "poisson_ratio": 0.3,
                "unit_weight": {"value": 20.0, "unit": "kN/m^3"},
                "davis": "B",
            },
            "steady_unconfined_darcy": {
                "conductivity": {"value": 1.0e-6, "unit": "m/s"},
                "water_unit_weight": {"value": 9.81, "unit": "kN/m^3"},
            },
            "coupling_none": {},
            "coupling_pressure_gradient_and_weight": {},
            "coupling_weight_only": {},
            "limit_load.work_controlled": {},
            "ssr.direct_potential": {},
            "ssr.indirect_work_controlled": {},
            "seepage.steady_unconfined": {},
        }
        self.assertEqual(set(registry.ids()), set(valid))
        for model_id, parameters in valid.items():
            with self.subTest(model_id=model_id):
                adapted = registry.adapt(model_id, parameters)
                self.assertEqual(adapted.model_id, model_id)
        with self.assertRaises(RegistryError) as caught:
            registry.adapt("unknown_model", {})
        self.assertEqual(caught.exception.code, "unknown_model")
        with self.assertRaises(RegistryError) as caught:
            registry.adapt("normal_pressure", {"value": 1.0, "unit": "kPa", "extra": 1})
        self.assertEqual(caught.exception.code, "invalid_model_parameters")
        with self.assertRaises(RegistryError):
            registry.adapt("normal_pressure", {"unit": "kPa"})
        with self.assertRaises(RegistryError):
            registry.adapt(
                "hydrostatic",
                {
                    **valid["hydrostatic"],
                    "unit_weight": 0.0,
                },
            )
        with self.assertRaises(RegistryError):
            registry.adapt(
                "piecewise_linear_level",
                {
                    **valid["piecewise_linear_level"],
                    "coordinates": [1.0, 0.0],
                },
            )

    def _assert_boundary_rule_semantics(self) -> None:
        asset = load_document(asset_document(), source_path="asset.json")
        profiles = [
            load_document(item, source_path="profiles.json")
            for item in profile_documents()
        ]
        base = case_document()
        cases = (
            (
                "value-model",
                {
                    **base["boundary_rules"][0],
                    "components": ["x"],
                    "value_model": "constant_scalar",
                    "parameters": {"value": 0.0, "unit": "kPa"},
                },
                None,
                "boundary_value_model_incompatible",
                "/boundary_rules/0/value_model",
            ),
            (
                "components",
                {
                    **base["boundary_rules"][0],
                    "components": ["z"],
                },
                None,
                "boundary_components_incompatible",
                "/boundary_rules/0/components",
            ),
            (
                "affine-dimension",
                {
                    **base["boundary_rules"][0],
                    "components": ["x"],
                    "value_model": "affine_coordinate",
                    "parameters": {
                        "offset": 0.0,
                        "gradient": [1.0],
                        "unit": "m",
                    },
                },
                None,
                "boundary_parameter_dimension",
                "/boundary_rules/0/parameters/gradient",
            ),
            (
                "natural-geometry",
                {
                    **base["boundary_rules"][0],
                    "condition": "natural",
                    "support": "top",
                    "components": ["all"],
                    "value_model": "global_traction",
                    "parameters": {"value": [0.0, -1.0], "unit": "kPa"},
                },
                None,
                "missing_geometry_patch",
                "/boundary_rules/0/geometry_patch",
            ),
            (
                "spatial-dimension",
                {
                    **base["boundary_rules"][0],
                    "condition": "natural",
                    "support": "top",
                    "components": ["all"],
                    "value_model": "spatial_global_traction",
                    "parameters": {
                        "offset": [0.0, -1.0],
                        "gradient": [[0.0, 0.0]],
                        "unit": "kPa",
                    },
                    "geometry_patch": "full-coordinate-field",
                },
                None,
                "boundary_parameter_dimension",
                "/boundary_rules/0/parameters/gradient",
            ),
            (
                "physics",
                {
                    **base["boundary_rules"][0],
                    "field": "pore_pressure",
                    "components": ["scalar"],
                    "value_model": "constant_scalar",
                    "parameters": {"value": 0.0, "unit": "kPa"},
                },
                None,
                "boundary_physics_incompatible",
                "/boundary_rules/0/field",
            ),
            (
                "flux-components",
                {
                    **base["boundary_rules"][0],
                    "field": "pore_pressure",
                    "condition": "natural",
                    "support": "top",
                    "components": ["scalar"],
                    "value_model": "prescribed_flux",
                    "parameters": {"value": 1.0e-6, "unit": "m/s"},
                    "geometry_patch": "full-coordinate-field",
                },
                {
                    "mechanics": True,
                    "seepage": True,
                    "coupling": "none",
                },
                "boundary_components_incompatible",
                "/boundary_rules/0/components",
            ),
        )
        for name, rule, physics, code, pointer in cases:
            with self.subTest(name=name):
                raw = case_document()
                raw["boundary_rules"] = [rule]
                if physics is not None:
                    raw["physics"] = physics
                case = load_document(stamped(raw), source_path="case.json")
                report = validate_documents(
                    asset=asset,
                    case=case,
                    profiles=profiles,
                )
                observed = [
                    issue
                    for issue in report.issues
                    if issue.code == code and issue.pointer == pointer
                ]
                self.assertEqual(len(observed), 1, report)

    def test_semantic_failures_precede_compilation(self) -> None:
        """AUT-NEG-SEMANTIC-001."""
        asset = load_document(asset_document(), source_path="asset.json")
        raw = case_document()
        raw["boundary_rules"][0]["value_model"] = "python_callable"
        raw["physics"] = {
            "mechanics": False,
            "seepage": False,
            "coupling": "pressure_gradient_and_weight",
        }
        raw["analysis"]["selector"] = "seepage.steady_unconfined"
        raw = stamped(raw)
        case = load_document(raw, source_path="case.json")
        report = validate_documents(
            asset=asset,
            case=case,
            profiles=[
                load_document(item, source_path="profiles.json")
                for item in profile_documents()
            ],
        )
        self.assertFalse(report.valid)
        self.assertEqual(
            [issue.code for issue in report.issues],
            [
                "analysis_physics_incompatible",
                "coupling_without_mechanics_and_seepage",
                "unknown_value_model",
            ],
        )
        with self.assertRaises(AuthoringError) as caught:
            report.require_valid()
        self.assertEqual(caught.exception.code, "authoring_invalid")
        self.assertEqual(caught.exception.phase, "authoring")
        self._assert_boundary_rule_semantics()

    def test_roundtrip_identity(self) -> None:
        """AUT-PROP-ROUNDTRIP-001."""
        raw_documents = (
            asset_document(),
            case_document(),
            *profile_documents(),
            suite_document(),
        )
        for index, raw in enumerate(raw_documents):
            with self.subTest(kind=raw["kind"]):
                parsed = load_document(
                    json.dumps(raw, indent=2).encode("utf-8"),
                    source_path=f"document-{index}.json",
                )
                rendered = render_document(parsed)
                reparsed = load_document(rendered, source_path=f"rendered-{index}.json")
                self.assertEqual(parsed, reparsed)
                self.assertEqual(authoring_identity(parsed), raw["id"])
                self.assertEqual(rendered, render_document(reparsed))
                schema_name = raw["kind"].removeprefix("slope_stability.")
                schema = json.loads(
                    (
                        ROOT / "schemas/generated/json" / f"{schema_name}.schema.json"
                    ).read_text(encoding="utf-8")
                )
                assert_schema_subset(json.loads(rendered), schema)
        raw_profile = profile_documents()[0]
        toml_profile = f"""
kind = "{raw_profile["kind"]}"
id = "{raw_profile["id"]}"
family = "{raw_profile["family"]}"

[schema_version]
major = 1
minor = 0
patch = 0

[created_by]
tool = "contract-fixture"
tool_version = "1"
source_id = "fixture-v1"

[provenance]
origin = "independent compact oracle"
license = "BSD-2-Clause"

[extensions]

[identity]
name = "continuation.default"
title = "continuation default"

[policy]
method = "work_controlled"
initial_factor = 1.0
initial_step = 0.1
minimum_step = 0.0001
maximum_step = 0.2
maximum_accepted_states = 100
maximum_work = 1.0e300
initial_work_scale = 0.2
scaled_slope_tolerance = 0.001
factor_increment_stop = 0.001
initial_nonlinear_stopping_criterion = "relative_correction"
initial_nonlinear_stopping_tolerance = 0.001
"""
        parsed_toml = load_document(
            toml_profile, source_path="continuation.default.toml"
        )
        self.assertEqual(authoring_identity(parsed_toml), raw_profile["id"])
        self.assertEqual(parsed_toml.policy["method"], "work_controlled")


if __name__ == "__main__":
    unittest.main()
