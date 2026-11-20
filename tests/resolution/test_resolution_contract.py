# SPDX-License-Identifier: BSD-2-Clause
"""Executable P-RESOLVE resolution contracts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/src"))

from slope_stability.authoring import ActiveRanks, AuthoringError, load_document
from slope_stability.compile import compact_simplex_mesh, compile_problem
from slope_stability.resolution import (
    BuildCapabilities,
    CheckpointPolicy,
    CompiledProblemRef,
    ResolutionError,
    SweepAxis,
    compose_profiles,
    expand_sweep,
    load_compiled_problem,
    resolve_resource,
    resolve_run,
    write_resolved_run,
)
from tests.support.problem_fixtures import (
    asset_document,
    assert_schema_subset,
    profile_document,
    profile_documents,
    stamped,
    suite_document,
    case_document,
)
from slope_stability.launch import LaunchLayout, local_resource_row


def parsed_profiles(raw_documents=None):
    return tuple(
        load_document(item, source_path=f"{item['identity']['name']}.json")
        for item in (raw_documents or profile_documents())
    )


def problem(
    *,
    dimension: int = 2,
    field_order: int = 4,
    analysis: str = "limit_load.work_controlled",
    operator_class: str = "nonsymmetric",
) -> CompiledProblemRef:
    return CompiledProblemRef(
        problem_id="sha256:" + "a" * 64,
        schema_version=(1, 0, 0),
        dimension=dimension,
        field_order=field_order,
        coordinate_order=4,
        analysis=analysis,
        operator_class=operator_class,
        required_outputs=("displacement", "stress"),
    )


def resource():
    suite = load_document(suite_document(), source_path="suite.json")
    return suite.resources[0]


def build() -> BuildCapabilities:
    return BuildCapabilities(
        build_id="sha256:" + "b" * 64,
        executable_id="sha256:" + "c" * 64,
        dependency_id="sha256:" + "d" * 64,
        schema_version=(1, 0, 0),
        petsc_int_bits=32,
        mpi=True,
        openmp=True,
        parallel_hdf5=True,
        hypre=True,
        parmetis=True,
        hydraulic_pmg_budget=True,
    )


class ResolutionContract(unittest.TestCase):
    def test_precedence(self) -> None:
        """RES-UNIT-PRECEDENCE-001."""
        profiles = list(parsed_profiles())
        baseline = resolve_run(
            problem=problem(),
            profiles=profiles,
            case_overrides={},
            suite_overrides={},
            resource=resource(),
            build=build(),
        )
        self.assertEqual(baseline.manifest["linear"]["relative_tolerance"], 1.0e-8)
        self.assertEqual(
            baseline.trace["linear.relative_tolerance"].source,
            "profile:linear.default",
        )
        self.assertEqual(
            baseline.manifest["nonlinear"]["regularization_minimum"], 1.0e-6
        )
        self.assertEqual(
            baseline.trace["nonlinear.regularization_minimum"].source,
            "profile:nonlinear.default",
        )
        self.assertEqual(
            baseline.manifest["continuation"]["initial_nonlinear_stopping_criterion"],
            "relative_correction",
        )
        self.assertEqual(
            baseline.manifest["continuation"]["initial_nonlinear_stopping_tolerance"],
            1.0e-3,
        )
        self.assertEqual(
            baseline.manifest["nonlinear"]["stopping_criterion"],
            "relative_residual",
        )
        self.assertEqual(
            baseline.manifest["nonlinear"]["stopping_tolerance"],
            1.0e-8,
        )
        self.assertEqual(
            baseline.trace["nonlinear.stopping_criterion"].source,
            "profile:nonlinear.default",
        )
        self.assertEqual(
            baseline.manifest["linear"]["stopping_condition"],
            "original_rhs_true_residual",
        )
        self.assertEqual(
            baseline.trace["linear.stopping_condition"].source,
            "schema_default",
        )
        self.assertEqual(baseline.manifest["linear"]["recycle_dimension"], 0)
        self.assertEqual(
            baseline.trace["linear.recycle_dimension"].source,
            "schema_default",
        )
        self.assertEqual(baseline.manifest["pmg"]["base_iterations"], 2)
        self.assertNotIn("hydraulic_base_iterations", baseline.manifest["pmg"])
        self.assertEqual(baseline.manifest["pmg"]["pre_smoothing_steps"], 2)
        self.assertEqual(baseline.manifest["pmg"]["post_smoothing_steps"], 2)
        self.assertEqual(baseline.manifest["pmg"]["coarse_redundant_group_size"], 0)
        self.assertEqual(
            baseline.trace["pmg.base_iterations"].source,
            "schema_default",
        )
        self.assertEqual(
            baseline.trace["pmg.pre_smoothing_steps"].source,
            "schema_default",
        )
        expected_deflation_defaults = {
            "candidate_source": "accepted_newton_and_control_directions",
            "recycle": "accepted_steps",
            "admission_tolerance": 1.0e-3,
            "pivot_tolerance": 1.0e-13,
            "condition_limit": 1.0e12,
            "invariant_tolerance": 1.0e-9,
            "reorthogonalization_passes": 2,
            "dense_factor": "partial_pivot_lu",
        }
        for field, value in expected_deflation_defaults.items():
            path = f"deflation.{field}"
            with self.subTest(path=path):
                self.assertEqual(
                    baseline.manifest["deflation"][field],
                    value,
                )
                self.assertEqual(
                    baseline.trace[path].source,
                    "schema_default",
                )

        case_value = resolve_run(
            problem=problem(),
            profiles=profiles,
            case_overrides={
                "linear.relative_tolerance": 2.0e-8,
                "linear.ksp_type": "hpddm_gcrodr",
                "linear.recycle_dimension": 12,
                "nonlinear.regularization_minimum": 2.0e-4,
                "continuation.initial_nonlinear_stopping_criterion": (
                    "relative_residual"
                ),
                "continuation.initial_nonlinear_stopping_tolerance": 2.0e-3,
                "nonlinear.stopping_criterion": "relative_correction",
                "nonlinear.stopping_tolerance": 2.0e-6,
                "pmg.smoother": "fgmres",
                "pmg.base_iterations": 1,
                "pmg.pre_smoothing_steps": 3,
                "pmg.post_smoothing_steps": 4,
                "pmg.coarse_solver": "preonly_lu",
                "pmg.coarse_redundant_group_size": 1,
                "deflation.maximum_dimension": 8,
            },
            suite_overrides={},
            resource=resource(),
            build=build(),
        )
        self.assertEqual(case_value.manifest["linear"]["relative_tolerance"], 2.0e-8)
        self.assertEqual(case_value.manifest["linear"]["ksp_type"], "hpddm_gcrodr")
        self.assertEqual(case_value.manifest["linear"]["recycle_dimension"], 12)
        self.assertEqual(
            case_value.trace["linear.recycle_dimension"].source,
            "case_override",
        )
        self.assertEqual(
            case_value.manifest["nonlinear"]["regularization_minimum"], 2.0e-4
        )
        self.assertEqual(
            case_value.trace["nonlinear.regularization_minimum"].source,
            "case_override",
        )
        self.assertEqual(
            case_value.manifest["continuation"]["initial_nonlinear_stopping_criterion"],
            "relative_residual",
        )
        self.assertEqual(
            case_value.manifest["continuation"]["initial_nonlinear_stopping_tolerance"],
            2.0e-3,
        )
        self.assertEqual(
            case_value.manifest["nonlinear"]["stopping_criterion"],
            "relative_correction",
        )
        self.assertEqual(
            case_value.manifest["nonlinear"]["stopping_tolerance"],
            2.0e-6,
        )
        self.assertEqual(
            case_value.trace["nonlinear.stopping_criterion"].source,
            "case_override",
        )
        self.assertEqual(
            case_value.trace["linear.relative_tolerance"].source,
            "case_override",
        )
        self.assertEqual(case_value.manifest["pmg"]["smoother"], "fgmres")
        self.assertEqual(case_value.manifest["pmg"]["base_iterations"], 1)
        self.assertEqual(case_value.manifest["pmg"]["pre_smoothing_steps"], 3)
        self.assertEqual(case_value.manifest["pmg"]["post_smoothing_steps"], 4)
        self.assertEqual(case_value.manifest["pmg"]["coarse_solver"], "preonly_lu")
        self.assertEqual(case_value.manifest["pmg"]["coarse_redundant_group_size"], 1)
        self.assertEqual(
            case_value.trace["pmg.base_iterations"].source,
            "case_override",
        )
        self.assertEqual(
            case_value.trace["pmg.post_smoothing_steps"].source,
            "case_override",
        )
        self.assertEqual(case_value.manifest["deflation"]["maximum_dimension"], 8)
        self.assertEqual(
            case_value.trace["deflation.maximum_dimension"].source,
            "case_override",
        )

        suite_value = resolve_run(
            problem=problem(),
            profiles=profiles,
            case_overrides={"linear.relative_tolerance": 2.0e-8},
            suite_overrides={"linear.relative_tolerance": 3.0e-8},
            resource=resource(),
            build=build(),
        )
        self.assertEqual(suite_value.manifest["linear"]["relative_tolerance"], 3.0e-8)
        self.assertEqual(
            suite_value.trace["linear.relative_tolerance"].source,
            "suite_sweep",
        )
        self.assertNotIn("environment", suite_value.manifest)
        self.assertNotIn("native_defaults", suite_value.manifest)

    def test_continuation_case_overrides_are_typed_and_traced(self) -> None:
        """RES-UNIT-CONTINUATION-OVERRIDE-001."""
        resolved = resolve_run(
            problem=problem(analysis="ssr.indirect_work_controlled"),
            profiles=list(parsed_profiles()),
            case_overrides={
                "continuation.initial_factor": 0.7,
                "continuation.minimum_step": 1.0e-5,
                "continuation.maximum_work": 1.0e10,
            },
            resource=resource(),
            build=build(),
        )
        expected = {
            "initial_factor": 0.7,
            "minimum_step": 1.0e-5,
            "maximum_work": 1.0e10,
        }
        for field, value in expected.items():
            path = f"continuation.{field}"
            with self.subTest(path=path):
                self.assertEqual(resolved.manifest["continuation"][field], value)
                self.assertEqual(resolved.trace[path].source, "case_override")

    def test_pure_seepage_profile_controls_hydraulic_solve(self) -> None:
        """RES-UNIT-SEEPAGE-001."""
        seepage = load_document(
            profile_document(
                "seepage",
                {
                    "linear_tolerance": 2.0e-10,
                    "linear_max_iter": 321,
                    "nonlinear_max_iter": 87,
                },
                name="seepage.test-tight",
            ),
            source_path="seepage.json",
        )
        pure_seepage = problem(
            field_order=1,
            analysis="seepage.steady_unconfined",
        )
        with self.assertRaises(ResolutionError) as caught:
            resolve_run(
                problem=pure_seepage,
                profiles=list(parsed_profiles()),
                resource=resource(),
                build=build(),
            )
        self.assertEqual(caught.exception.code, "missing_profile_family")
        self.assertEqual(str(caught.exception), "missing_profile_family: seepage")

        resolved = resolve_run(
            problem=pure_seepage,
            profiles=[*parsed_profiles(), seepage],
            resource=resource(),
            build=build(),
        )
        self.assertEqual(resolved.manifest["linear"]["relative_tolerance"], 2.0e-10)
        self.assertEqual(resolved.manifest["linear"]["maximum_iterations"], 321)
        self.assertEqual(resolved.manifest["pmg"]["base_iterations"], 2)
        self.assertEqual(resolved.manifest["pmg"]["hydraulic_base_iterations"], 321)
        legacy_build = build().model_copy(update={"hydraulic_pmg_budget": False})
        legacy = resolve_run(
            problem=pure_seepage,
            profiles=[*parsed_profiles(), seepage],
            resource=resource(),
            build=legacy_build,
        )
        self.assertNotIn("hydraulic_base_iterations", legacy.manifest["pmg"])
        self.assertNotIn("hydraulic_pmg_budget", legacy.manifest["build"])
        with self.assertRaises(ResolutionError) as unsupported:
            resolve_run(
                problem=pure_seepage,
                profiles=[*parsed_profiles(), seepage],
                resource=resource(),
                build=legacy_build,
                suite_overrides={"pmg.hydraulic_base_iterations": 7},
            )
        self.assertEqual(
            unsupported.exception.code, "build_missing_hydraulic_pmg_budget"
        )
        self.assertEqual(
            resolved.trace["pmg.hydraulic_base_iterations"].source,
            "derived:linear.maximum_iterations",
        )
        self.assertEqual(resolved.manifest["nonlinear"]["relative_tolerance"], 2.0e-10)
        self.assertEqual(resolved.manifest["nonlinear"]["absolute_tolerance"], 2.0e-10)
        self.assertEqual(resolved.manifest["nonlinear"]["maximum_iterations"], 87)
        self.assertEqual(
            resolved.manifest["provenance"]["profile_ids"]["seepage"],
            seepage.id,
        )
        self.assertEqual(
            resolved.trace["nonlinear.relative_tolerance"].source,
            "profile:seepage.test-tight",
        )
        self.assertEqual(
            resolved.manifest["linear"]["hydraulic_relative_tolerance"],
            2.0e-10,
        )
        self.assertEqual(
            resolved.manifest["linear"]["hydraulic_absolute_tolerance"],
            2.0e-10,
        )
        self.assertEqual(
            resolved.manifest["linear"]["hydraulic_maximum_iterations"],
            321,
        )
        self.assertEqual(
            resolved.manifest["nonlinear"]["hydraulic_relative_tolerance"],
            2.0e-10,
        )
        self.assertEqual(
            resolved.manifest["nonlinear"]["hydraulic_absolute_tolerance"],
            2.0e-10,
        )
        self.assertEqual(
            resolved.manifest["nonlinear"]["hydraulic_maximum_iterations"],
            87,
        )

    def test_coupled_seepage_profile_does_not_replace_mechanics_policy(self) -> None:
        """RES-UNIT-SEEPAGE-COUPLED-001."""
        seepage = load_document(
            profile_document(
                "seepage",
                {
                    "linear_tolerance": 3.0e-11,
                    "linear_max_iter": 456,
                    "nonlinear_max_iter": 98,
                },
                name="seepage.test-coupled",
            ),
            source_path="seepage.json",
        )
        resolved = resolve_run(
            problem=problem(analysis="ssr.indirect_work_controlled"),
            profiles=[*parsed_profiles(), seepage],
            resource=resource(),
            build=build(),
        )
        self.assertEqual(resolved.manifest["linear"]["relative_tolerance"], 1.0e-8)
        self.assertEqual(resolved.manifest["linear"]["maximum_iterations"], 500)
        self.assertEqual(resolved.manifest["pmg"]["base_iterations"], 2)
        self.assertEqual(resolved.manifest["pmg"]["hydraulic_base_iterations"], 500)
        overridden = resolve_run(
            problem=problem(analysis="ssr.indirect_work_controlled"),
            profiles=[*parsed_profiles(), seepage],
            resource=resource(),
            build=build(),
            suite_overrides={
                "pmg.base_iterations": 1,
                "pmg.hydraulic_base_iterations": 7,
            },
        )
        self.assertEqual(overridden.manifest["pmg"]["base_iterations"], 1)
        self.assertEqual(overridden.manifest["pmg"]["hydraulic_base_iterations"], 7)
        self.assertEqual(
            overridden.trace["pmg.hydraulic_base_iterations"].source, "suite_sweep"
        )
        self.assertNotEqual(overridden.run_id, resolved.run_id)
        for value in (0, -1, True, 1.5, "7", 2**31):
            with self.subTest(hydraulic_base_iterations=value):
                with self.assertRaises(ResolutionError):
                    resolve_run(
                        problem=problem(analysis="ssr.indirect_work_controlled"),
                        profiles=[*parsed_profiles(), seepage],
                        resource=resource(),
                        build=build(),
                        suite_overrides={"pmg.hydraulic_base_iterations": value},
                    )
        self.assertEqual(resolved.manifest["nonlinear"]["relative_tolerance"], 1.0e-8)
        self.assertEqual(resolved.manifest["nonlinear"]["maximum_iterations"], 30)
        self.assertEqual(
            resolved.manifest["linear"]["hydraulic_relative_tolerance"],
            3.0e-11,
        )
        self.assertEqual(
            resolved.manifest["linear"]["hydraulic_absolute_tolerance"],
            3.0e-11,
        )
        self.assertEqual(
            resolved.manifest["linear"]["hydraulic_maximum_iterations"],
            456,
        )
        self.assertEqual(
            resolved.manifest["nonlinear"]["hydraulic_relative_tolerance"],
            3.0e-11,
        )
        self.assertEqual(
            resolved.manifest["nonlinear"]["hydraulic_absolute_tolerance"],
            3.0e-11,
        )
        self.assertEqual(
            resolved.manifest["nonlinear"]["hydraulic_maximum_iterations"],
            98,
        )
        self.assertEqual(
            resolved.trace["linear.hydraulic_relative_tolerance"].source,
            "profile:seepage.test-coupled",
        )

    def test_inheritance_is_forbidden(self) -> None:
        """RES-UNIT-INHERIT-001."""
        for forbidden in (
            {"parent": "linear.base"},
            {"extends": ["linear.base"]},
            {"inherit": "linear.base"},
        ):
            with self.subTest(forbidden=forbidden):
                raw = profile_document(
                    "linear",
                    {
                        "ksp_type": "fgmres",
                        "operator_class": "nonsymmetric",
                        "relative_tolerance": 1.0e-8,
                        "absolute_tolerance": 1.0e-12,
                        "maximum_iterations": 100,
                        "restart": 30,
                        "expert_options": {},
                    },
                    name="linear.derived",
                )
                raw.update(forbidden)
                raw = stamped(raw)
                with self.assertRaises(AuthoringError) as caught:
                    load_document(raw, source_path="derived.json")
                self.assertEqual(caught.exception.code, "profile_inheritance_forbidden")
        composed = compose_profiles(parsed_profiles())
        self.assertEqual(
            set(composed),
            {
                "continuation",
                "nonlinear",
                "linear",
                "pmg",
                "deflation",
                "output",
                "validation",
            },
        )
        duplicate = list(parsed_profiles())
        duplicate.append(duplicate[2])
        with self.assertRaises(ResolutionError) as caught:
            compose_profiles(duplicate)
        self.assertEqual(caught.exception.code, "duplicate_profile_family")

    def test_cartesian_sweeps(self) -> None:
        """RES-UNIT-SWEEP-001."""
        axes = (
            SweepAxis(
                name="z-step",
                path="continuation.initial_step",
                values=(0.1, 0.05),
            ),
            SweepAxis(
                name="a-rtol",
                path="linear.relative_tolerance",
                values=(1.0e-8, 1.0e-9),
            ),
        )
        rows = expand_sweep(axes=axes, base_overrides={}, repetitions=2)
        self.assertEqual(len(rows), 8)
        self.assertEqual(
            [row.coordinates for row in rows[:4]],
            [
                {
                    "a-rtol": 1.0e-8,
                    "z-step": 0.1,
                    "repetition": 0,
                },
                {
                    "a-rtol": 1.0e-8,
                    "z-step": 0.1,
                    "repetition": 1,
                },
                {
                    "a-rtol": 1.0e-8,
                    "z-step": 0.05,
                    "repetition": 0,
                },
                {
                    "a-rtol": 1.0e-8,
                    "z-step": 0.05,
                    "repetition": 1,
                },
            ],
        )
        self.assertEqual(
            [row.variant_id for row in rows],
            [row.variant_id for row in expand_sweep(axes=axes, repetitions=2)],
        )
        self.assertEqual(len({row.variant_id for row in rows}), len(rows))
        with self.assertRaises(ResolutionError) as caught:
            expand_sweep(
                axes=(
                    SweepAxis(
                        name="duplicate",
                        path="linear.relative_tolerance",
                        values=(1.0e-8, 1.0e-8),
                    ),
                )
            )
        self.assertEqual(caught.exception.code, "duplicate_effective_run")
        with self.assertRaises(ValidationError):
            SweepAxis(name="zip", path="linear.relative_tolerance", values=())

    def test_resource_resolution(self) -> None:
        """RES-UNIT-RESOURCE-001."""
        measured = local_resource_row(LaunchLayout(2, 16), petsc_threads=16)
        self.assertEqual(measured.openmp_threads, 16)
        self.assertEqual(measured.third_party_threads.petsc, 16)
        self.assertEqual(measured.third_party_threads.blas, 1)
        self.assertEqual(measured.third_party_threads.hypre, 1)
        with self.assertRaises(ValueError):
            local_resource_row(LaunchLayout(2, 8), petsc_threads=16)
        resolved = resolve_resource(resource(), field_order=4)
        self.assertEqual(resolved["world_size"], 4)
        self.assertEqual(resolved["total_workers"], 32)
        self.assertEqual(resolved["openmp_threads"], 8)
        self.assertEqual(resolved["mpi_thread_level"], "funneled")
        self.assertEqual(resolved["active_ranks"]["fine"], [0, 1, 2, 3])
        self.assertEqual(resolved["active_ranks"]["coarse"], [0, 2, 3])
        self.assertEqual(
            resolved["openmp_environment"],
            {
                "OMP_DYNAMIC": "FALSE",
                "OMP_MAX_ACTIVE_LEVELS": "1",
                "OMP_PLACES": "cores",
                "OMP_PROC_BIND": "close",
            },
        )
        self.assertEqual(
            resolved["third_party_threads"],
            {"blas": 1, "hypre": 1, "petsc": 1},
        )

        declared_memory = resource().model_copy(
            update={"memory_per_node_mib": 32_768, "address_space_per_rank_mib": 4_096}
        )
        self.assertEqual(
            resolve_resource(declared_memory, field_order=4)[
                "linear_vector_memory_cap_mib"
            ],
            4_096,
        )
        # The operator workspace defaults to a quarter of the rank's memory
        # share, at least 1024 MiB; an explicit value is kept.
        self.assertNotIn(
            "operator_workspace_mib", resolve_resource(resource(), field_order=4)
        )
        self.assertEqual(
            resolve_resource(declared_memory, field_order=4)["operator_workspace_mib"],
            1_024,
        )
        large_node = resource().model_copy(update={"memory_per_node_mib": 262_144})
        self.assertEqual(
            resolve_resource(large_node, field_order=4)["operator_workspace_mib"],
            262_144 // large_node.ranks_per_node // 4,
        )
        explicit = large_node.model_copy(update={"operator_workspace_mib": 512})
        self.assertEqual(
            resolve_resource(explicit, field_order=4)["operator_workspace_mib"], 512
        )
        overrides = {
            "resources.linear_vector_memory_cap_mib": 2_048,
            "pmg.mechanics_matrix_free": True,
            "pmg.matrix_free_block_nodes": 2,
            "linear.orthogonalization": "cgs_unrefined",
        }
        run = resolve_run(
            problem=problem(),
            profiles=parsed_profiles(),
            build=build(),
            resource=declared_memory,
            case_overrides=overrides,
        )
        for path, value in overrides.items():
            family, field = path.split(".")
            self.assertEqual(run.manifest[family][field], value)
            self.assertEqual(run.trace[path].source, "case_override")
        for path, bad in (
            ("pmg.mechanics_matrix_free", "true"),
            ("pmg.matrix_free_block_nodes", 36),
            ("resources.linear_vector_memory_cap_mib", 0),
            ("linear.orthogonalization", "unknown"),
        ):
            with (
                self.subTest(path=path),
                self.assertRaisesRegex(ResolutionError, "solver_setting_invalid"),
            ):
                resolve_run(
                    problem=problem(),
                    profiles=parsed_profiles(),
                    build=build(),
                    resource=declared_memory,
                    case_overrides={path: bad},
                )

        raw = suite_document()
        raw["resources"][0]["cores_per_rank"] = 4
        invalid = load_document(stamped(raw), source_path="suite-invalid.json")
        with self.assertRaises(ResolutionError) as caught:
            resolve_resource(invalid.resources[0], field_order=4)
        self.assertEqual(caught.exception.code, "insufficient_cores_per_rank")
        raw = suite_document()
        raw["resources"][0]["active_ranks"]["coarse"] = [0, 4]
        invalid = load_document(stamped(raw), source_path="suite-ranks.json")
        with self.assertRaises(ResolutionError) as caught:
            resolve_resource(invalid.resources[0], field_order=4)
        self.assertEqual(caught.exception.code, "active_rank_out_of_range")

    def test_pmg_profile_family_resolves_from_actual_fine_order(self) -> None:
        """RES-PMG-DYNAMIC-HIERARCHY-001."""
        expected = {
            1: [1],
            2: [1, 2],
            4: [1, 2, 4],
            8: [1, 2, 4, 8],
        }
        for field_order, levels in expected.items():
            with self.subTest(field_order=field_order):
                resolved = resolve_run(
                    problem=problem(field_order=field_order),
                    profiles=parsed_profiles(),
                    resource=resource(),
                    build=build(),
                )
                self.assertEqual(resolved.manifest["pmg"]["levels"], levels)
                self.assertEqual(
                    resolved.manifest["pmg"]["coarse_direct_maximum_dofs"],
                    50000,
                )
                self.assertEqual(
                    resolved.trace["pmg.coarse_direct_maximum_dofs"].source,
                    "profile:pmg.default",
                )
        overridden = resolve_run(
            problem=problem(field_order=8),
            profiles=parsed_profiles(),
            case_overrides={"pmg.levels": [1, 2, 4, 8]},
            resource=resource(),
            build=build(),
        )
        self.assertEqual(overridden.manifest["pmg"]["levels"], [1, 2, 4, 8])
        self.assertEqual(overridden.trace["pmg.levels"].source, "case_override")
        swept = resolve_run(
            problem=problem(field_order=4),
            profiles=parsed_profiles(),
            case_overrides={"pmg.levels": [1, 4]},
            suite_overrides={"pmg.levels": [1, 2, 4]},
            resource=resource(),
            build=build(),
        )
        self.assertEqual(swept.manifest["pmg"]["levels"], [1, 2, 4])
        self.assertEqual(swept.trace["pmg.levels"].source, "suite_sweep")
        explicit = profile_documents()
        explicit[3]["policy"]["levels"] = [1, 2, 4]
        profiled = resolve_run(
            problem=problem(field_order=2),
            profiles=parsed_profiles([stamped(item) for item in explicit]),
            resource=resource(),
            build=build(),
        )
        self.assertEqual(profiled.manifest["pmg"]["levels"], [1, 2])
        self.assertEqual(profiled.trace["pmg.levels"].source, "profile:pmg.default")
        for levels in ([], [1, 1, 4], [1, 3, 4], [1, True, 4], [4, 1], [1, 2]):
            with self.subTest(invalid_levels=levels):
                with self.assertRaises(ResolutionError):
                    resolve_run(
                        problem=problem(field_order=4),
                        profiles=parsed_profiles(),
                        case_overrides={"pmg.levels": levels},
                        resource=resource(),
                        build=build(),
                    )

    def test_solver_compatibility(self) -> None:
        """RES-SOLVER-COMPAT-001."""
        scenarios = []
        cg = profile_documents()
        cg[2]["policy"]["ksp_type"] = "cg"
        cg = [stamped(item) for item in cg]
        scenarios.append(
            (
                "ksp_operator_incompatible",
                problem(operator_class="nonsymmetric"),
                parsed_profiles(cg),
            )
        )
        bad_pmg = profile_documents()
        bad_pmg[3]["policy"]["levels"] = [1, 2, 2]
        bad_pmg = [stamped(item) for item in bad_pmg]
        scenarios.append(("duplicate_pmg_level", problem(), parsed_profiles(bad_pmg)))
        bad_deflation = profile_documents()
        bad_deflation[4]["policy"] = {
            "enabled": True,
            "projector": "one_sided",
            "maximum_dimension": 8,
        }
        bad_deflation = [stamped(item) for item in bad_deflation]
        scenarios.append(
            (
                "deflation_projector_incompatible",
                problem(operator_class="nonsymmetric"),
                parsed_profiles(bad_deflation),
            )
        )
        for code, problem_ref, profiles in scenarios:
            with self.subTest(code=code):
                with self.assertRaises(ResolutionError) as caught:
                    resolve_run(
                        problem=problem_ref,
                        profiles=profiles,
                        resource=resource(),
                        build=build(),
                    )
                self.assertEqual(caught.exception.code, code)

    def test_solver_options(self) -> None:
        """RES-SOLVER-OPTIONS-001."""
        run = resolve_run(
            problem=problem(),
            profiles=parsed_profiles(),
            resource=resource(),
            build=build(),
        )
        self.assertEqual(
            run.manifest["petsc_options"],
            {
                "-mechanics_ksp_atol": "1e-12",
                "-mechanics_ksp_max_it": "500",
                "-mechanics_ksp_rtol": "1e-8",
                "-mechanics_ksp_type": "fgmres",
                "-mechanics_ksp_gmres_restart": "40",
                "-mechanics_pc_type": "slope_pmg",
            },
        )
        for option, code in (
            ("-mechanics_ksp_rtol", "expert_option_conflict"),
            ("-pc_type", "expert_option_not_allowlisted"),
            ("-malloc_dump", "expert_option_not_allowlisted"),
        ):
            raw = profile_documents()
            raw[2]["policy"]["expert_options"] = {option: "1"}
            raw = [stamped(item) for item in raw]
            with self.subTest(option=option):
                with self.assertRaises(ResolutionError) as caught:
                    resolve_run(
                        problem=problem(),
                        profiles=parsed_profiles(raw),
                        resource=resource(),
                        build=build(),
                    )
                self.assertEqual(caught.exception.code, code)

        raw = profile_documents()
        raw[2]["policy"]["expert_options"] = {"-mechanics_ksp_monitor": None}
        raw = [stamped(item) for item in raw]
        monitored = resolve_run(
            problem=problem(),
            profiles=parsed_profiles(raw),
            resource=resource(),
            build=build(),
        )
        self.assertIsNone(monitored.manifest["petsc_options"]["-mechanics_ksp_monitor"])

    def test_accepted_state_limit(self) -> None:
        path = "continuation.maximum_accepted_states"
        for analysis, limit in (
            ("limit_load.work_controlled", 127),
            ("ssr.direct_potential", 128),
            ("ssr.indirect_work_controlled", 128),
        ):
            with self.subTest(analysis=analysis):
                inputs = dict(
                    problem=problem(analysis=analysis),
                    profiles=parsed_profiles(),
                    resource=resource(),
                    build=build(),
                )
                resolved = resolve_run(**inputs, case_overrides={path: limit})
                self.assertEqual(
                    resolved.manifest["continuation"]["maximum_accepted_states"],
                    limit,
                )
                for invalid in (limit + 1, 2**32 + 1, 0):
                    with self.assertRaisesRegex(
                        ResolutionError, rf"{path} must be between 1 and {limit}"
                    ) as caught:
                        resolve_run(**inputs, case_overrides={path: invalid})
                    self.assertEqual(
                        caught.exception.code, "continuation_accepted_state_limit"
                    )

    def test_complete_trace_and_identity(self) -> None:
        """RES-UNIT-TRACE-001."""
        first = resolve_run(
            problem=problem(),
            profiles=parsed_profiles(),
            resource=resource(),
            build=build(),
        )
        second = resolve_run(
            problem=problem(),
            profiles=parsed_profiles(),
            resource=resource(),
            build=build(),
        )
        self.assertEqual(first.run_id, second.run_id)
        self.assertEqual(first.canonical_bytes, second.canonical_bytes)
        with tempfile.TemporaryDirectory(prefix="resolved-run-a-") as raw_a:
            with tempfile.TemporaryDirectory(prefix="resolved-run-b-") as raw_b:
                path_a = write_resolved_run(first, Path(raw_a))
                path_b = write_resolved_run(second, Path(raw_b))
                self.assertEqual(path_a.name, "resolved-run.json")
                self.assertEqual(path_a.read_bytes(), path_b.read_bytes())
                self.assertEqual(path_a.read_bytes(), first.canonical_bytes)
                schema = json.loads(
                    (
                        ROOT / "schemas/generated/json/resolved-run.schema.json"
                    ).read_text(encoding="utf-8")
                )
                assert_schema_subset(json.loads(path_a.read_bytes()), schema)
        with tempfile.TemporaryDirectory(prefix="resolved-compiled-pair-") as raw:
            asset = load_document(asset_document(2, coordinate_order=4))
            case = load_document(case_document(2, field_order=4, coordinate_order=4))
            package = compile_problem(
                asset=asset,
                case=case,
                mesh=compact_simplex_mesh(2, 4),
                destination=Path(raw),
            )
            loaded = load_compiled_problem(package.problem_json)
            integrated = resolve_run(
                problem=loaded,
                profiles=parsed_profiles(),
                resource=resource(),
                build=build(),
            )
            self.assertEqual(loaded.problem_id, package.problem_id)
            self.assertEqual(integrated.manifest["problem"]["id"], package.problem_id)
            hdf5_path = package.path / "problem.h5"
            payload = hdf5_path.read_bytes()
            hdf5_path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
            with self.assertRaises(ResolutionError) as caught:
                load_compiled_problem(package.problem_json)
            self.assertEqual(caught.exception.code, "compiled_problem_payload_digest")
        required_traces = {
            "analysis.selector",
            "continuation.initial_step",
            "nonlinear.relative_tolerance",
            "nonlinear.regularization_minimum",
            "linear.ksp_type",
            "linear.relative_tolerance",
            "pmg.enabled",
            "deflation.enabled",
            "outputs.quantities",
            "validation.qualification_mode",
            "resources.world_size",
            "resources.openmp_threads",
            "resources.cpu_binding",
            "resources.active_ranks.coarse",
            "result_policy.atomic_publication",
            "result_policy.numerical_policy_id",
        }
        self.assertTrue(required_traces.issubset(first.trace))
        for path in required_traces:
            record = first.trace[path]
            self.assertTrue(record.source)
            self.assertEqual(record.path, path)
        changed = resolve_run(
            problem=problem(),
            profiles=parsed_profiles(),
            suite_overrides={
                "linear.relative_tolerance": 1.0e-9,
                "deflation.enabled": False,
            },
            resource=resource(),
            build=build(),
        )
        self.assertNotEqual(first.run_id, changed.run_id)
        self.assertNotEqual(
            first.manifest["result_policy"]["numerical_policy_id"],
            changed.manifest["result_policy"]["numerical_policy_id"],
        )
        self.assertFalse(changed.manifest["deflation"]["enabled"])
        self.assertEqual(
            changed.trace["deflation.enabled"].source,
            "suite_sweep",
        )
        repartitioned_resource = resource().model_copy(
            update={
                "name": "repartitioned-2x2",
                "world_size": 2,
                "ranks_per_node": 2,
                "openmp_threads": 2,
                "cores_per_rank": 2,
                "active_ranks": ActiveRanks(
                    fine=(0, 1),
                    coarse=(0, 1),
                ),
            }
        )
        repartitioned = resolve_run(
            problem=problem(),
            profiles=parsed_profiles(),
            resource=repartitioned_resource,
            build=build(),
        )
        self.assertNotEqual(first.run_id, repartitioned.run_id)
        self.assertEqual(
            first.manifest["result_policy"]["numerical_policy_id"],
            repartitioned.manifest["result_policy"]["numerical_policy_id"],
        )
        bounded_source = resolve_run(
            problem=problem(),
            profiles=parsed_profiles(),
            case_overrides={"continuation.maximum_accepted_states": 2},
            resource=resource(),
            build=build(),
        )
        self.assertNotEqual(first.run_id, bounded_source.run_id)
        self.assertEqual(
            first.manifest["result_policy"]["numerical_policy_id"],
            bounded_source.manifest["result_policy"]["numerical_policy_id"],
        )
        checkpoint_write = resolve_run(
            problem=problem(),
            profiles=parsed_profiles(),
            resource=resource(),
            build=build(),
            checkpoint=CheckpointPolicy(mode="write_accepted"),
        )
        checkpoint_restore = resolve_run(
            problem=problem(),
            profiles=parsed_profiles(),
            resource=repartitioned_resource,
            build=build(),
            checkpoint=CheckpointPolicy(
                mode="restore_accepted",
                path="/qualified/checkpoints/state-3",
            ),
        )
        self.assertEqual(
            checkpoint_write.manifest["result_policy"]["checkpoint"],
            {"mode": "write_accepted"},
        )
        self.assertEqual(
            checkpoint_restore.manifest["result_policy"]["checkpoint"],
            {
                "mode": "restore_accepted",
                "path": "/qualified/checkpoints/state-3",
            },
        )
        self.assertEqual(
            checkpoint_write.manifest["result_policy"]["numerical_policy_id"],
            checkpoint_restore.manifest["result_policy"]["numerical_policy_id"],
        )
        self.assertIn("result_policy.checkpoint.mode", checkpoint_write.trace)
        with self.assertRaises(ValidationError):
            CheckpointPolicy(mode="write_accepted", path="/unexpected")
        with self.assertRaises(ValidationError):
            CheckpointPolicy(mode="restore_accepted", path="relative")
        with self.assertRaises(ValidationError):
            CheckpointPolicy(mode="restore_accepted", path="/not/../normalized")
        with self.assertRaises(ValidationError):
            CheckpointPolicy(mode="restore_accepted", path="//not/normalized")
        with self.assertRaises(ValidationError):
            CheckpointPolicy(mode="restore_accepted", path="/not/\x00normalized")
        with self.assertRaises(ResolutionError) as checkpoint_type:
            resolve_run(
                problem=problem(),
                profiles=parsed_profiles(),
                resource=resource(),
                build=build(),
                checkpoint={"mode": "write_accepted"},  # type: ignore[arg-type]
            )
        self.assertEqual(checkpoint_type.exception.code, "checkpoint_policy_type")
        with self.assertRaises(ResolutionError) as caught:
            resolve_run(
                problem=problem(),
                profiles=parsed_profiles(),
                case_overrides={"unknown.path": 1},
                resource=resource(),
                build=build(),
            )
        self.assertEqual(caught.exception.code, "override_not_allowlisted")
        with self.assertRaises(ResolutionError) as caught:
            resolve_run(
                problem=problem(),
                profiles=parsed_profiles(),
                case_overrides={"pmg.smoother": "unbounded-krylov"},
                resource=resource(),
                build=build(),
            )
        self.assertEqual(caught.exception.code, "override_value_not_allowlisted")
        for smoother in (
            "chebyshev_point",
            "fgmres_block",
            "richardson_block",
        ):
            with self.subTest(smoother=smoother):
                resolved = resolve_run(
                    problem=problem(),
                    profiles=parsed_profiles(),
                    case_overrides={"pmg.smoother": smoother},
                    resource=resource(),
                    build=build(),
                )
                self.assertEqual(resolved.manifest["pmg"]["smoother"], smoother)
        invalid_deflation_overrides = {
            "deflation.projector": "one_sided",
            "deflation.candidate_source": "all_krylov_vectors",
            "deflation.recycle": "failed_steps",
            "deflation.admission_tolerance": 0.0,
            "deflation.pivot_tolerance": -1.0,
            "deflation.condition_limit": 1.0,
            "deflation.invariant_tolerance": 0.0,
            "deflation.reorthogonalization_passes": 5,
            "deflation.maximum_dimension": 0,
            "deflation.dense_factor": "inverse",
            "pmg.base_iterations": 0,
            "pmg.hydraulic_base_iterations": 0,
        }
        for path, value in invalid_deflation_overrides.items():
            with self.subTest(path=path):
                with self.assertRaises(ResolutionError) as caught:
                    resolve_run(
                        problem=problem(),
                        profiles=parsed_profiles(),
                        case_overrides={path: value},
                        resource=resource(),
                        build=build(),
                    )
                self.assertIn(
                    caught.exception.code,
                    {
                        "override_value_not_allowlisted",
                        "override_value_out_of_range",
                    },
                )

        legacy_matched = resolve_run(
            problem=problem(),
            profiles=parsed_profiles(),
            case_overrides={
                "continuation.factor_increment_stop": 0.0,
                "linear.absolute_tolerance": 1.0e-30,
                "linear.stopping_condition": "projected_rhs_petsc",
                "deflation.projector": "legacy_a_orthogonal",
                "deflation.candidate_source": "accepted_state_vectors",
                "deflation.recycle": "attempt_snapshot_restore",
            },
            resource=resource(),
            build=build(),
        )
        self.assertEqual(
            legacy_matched.manifest["linear"]["stopping_condition"],
            "projected_rhs_petsc",
        )
        self.assertEqual(
            legacy_matched.manifest["linear"]["absolute_tolerance"],
            1.0e-30,
        )
        self.assertEqual(
            legacy_matched.manifest["continuation"]["factor_increment_stop"],
            0.0,
        )
        self.assertEqual(
            legacy_matched.manifest["deflation"]["projector"],
            "legacy_a_orthogonal",
        )
        self.assertEqual(
            legacy_matched.manifest["deflation"]["candidate_source"],
            "accepted_state_vectors",
        )

        fcg_preconditioned = resolve_run(
            problem=problem(),
            profiles=parsed_profiles(),
            case_overrides={
                "linear.ksp_type": "fcg",
                "linear.stopping_condition": ("preconditioned_residual_original_rhs"),
            },
            resource=resource(),
            build=build(),
        )
        self.assertEqual(fcg_preconditioned.manifest["linear"]["ksp_type"], "fcg")
        self.assertEqual(
            fcg_preconditioned.manifest["linear"]["stopping_condition"],
            "preconditioned_residual_original_rhs",
        )
        fcg_natural = resolve_run(
            problem=problem(),
            profiles=parsed_profiles(),
            case_overrides={
                "linear.ksp_type": "fcg",
                "linear.stopping_condition": "natural_residual_initial",
            },
            resource=resource(),
            build=build(),
        )
        self.assertEqual(fcg_natural.manifest["linear"]["ksp_type"], "fcg")
        self.assertEqual(
            fcg_natural.manifest["linear"]["stopping_condition"],
            "natural_residual_initial",
        )
        with self.assertRaises(ResolutionError) as caught:
            resolve_run(
                problem=problem(),
                profiles=parsed_profiles(),
                case_overrides={
                    "linear.stopping_condition": (
                        "preconditioned_residual_original_rhs"
                    ),
                },
                resource=resource(),
                build=build(),
            )
        self.assertEqual(
            caught.exception.code, "linear_stopping_condition_incompatible"
        )
        with self.assertRaises(ResolutionError) as caught:
            resolve_run(
                problem=problem(),
                profiles=parsed_profiles(),
                case_overrides={
                    "linear.stopping_condition": "natural_residual_initial",
                },
                resource=resource(),
                build=build(),
            )
        self.assertEqual(
            caught.exception.code, "linear_stopping_condition_incompatible"
        )

        full_krylov_bound = resolve_run(
            problem=problem(),
            profiles=parsed_profiles(),
            case_overrides={
                "linear.ksp_type": "fgmres",
                "linear.recycle_dimension": 0,
                "deflation.projector": "legacy_a_orthogonal",
                "deflation.candidate_source": "accepted_states_and_full_krylov",
                "deflation.recycle": "attempt_snapshot_restore",
                "deflation.admission_tolerance": 1.0e-10,
                "deflation.maximum_dimension": 5000,
                "deflation.compression": "operator_svd",
                "deflation.compression_dimension": 24,
                "deflation.compression_history_decay": 0.5,
            },
            resource=resource(),
            build=build(),
        )
        self.assertEqual(
            full_krylov_bound.manifest["deflation"]["candidate_source"],
            "accepted_states_and_full_krylov",
        )
        self.assertEqual(
            full_krylov_bound.manifest["deflation"]["recycle"],
            "attempt_snapshot_restore",
        )
        self.assertEqual(
            full_krylov_bound.manifest["deflation"]["maximum_dimension"], 5000
        )
        self.assertEqual(
            full_krylov_bound.manifest["deflation"]["admission_tolerance"],
            1.0e-10,
        )
        self.assertEqual(
            full_krylov_bound.manifest["deflation"]["compression"],
            "operator_svd",
        )
        self.assertEqual(
            full_krylov_bound.manifest["deflation"]["compression_dimension"],
            24,
        )
        self.assertEqual(
            full_krylov_bound.manifest["deflation"]["compression_history_decay"],
            0.5,
        )

        with self.assertRaises(ResolutionError) as invalid_compression:
            resolve_run(
                problem=problem(),
                profiles=parsed_profiles(),
                case_overrides={
                    "deflation.compression": "snapshot_pod",
                    "deflation.compression_dimension": 6,
                },
                resource=resource(),
                build=build(),
            )
        self.assertEqual(
            invalid_compression.exception.code,
            "deflation_compression_incompatible",
        )

        with self.assertRaises(ResolutionError) as mismatched:
            resolve_run(
                problem=problem(),
                profiles=parsed_profiles(),
                case_overrides={
                    "deflation.candidate_source": "accepted_state_vectors",
                },
                resource=resource(),
                build=build(),
            )
        self.assertEqual(
            mismatched.exception.code,
            "deflation_lifecycle_incompatible",
        )


if __name__ == "__main__":
    unittest.main()
