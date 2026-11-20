# SPDX-License-Identifier: BSD-2-Clause
"""Installed engine, end-to-end and restart contracts."""

from __future__ import annotations

from dataclasses import replace
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import time
import unittest
import unittest.mock

import h5py
import numpy as np

from slope_protocol import CanonicalizationError, canonicalize, content_id
from slope_stability.authoring import ActiveRanks, load_document
from slope_stability.compile import (
    CanonicalMesh,
    RawMesh,
    RawSupport,
    canonicalize_mesh,
    compile_problem,
    compact_simplex_mesh,
)
from slope_stability.resolution import (
    BuildCapabilities,
    CheckpointPolicy,
    load_compiled_problem,
    resolve_run,
    write_resolved_run,
)
from slope_stability.restart import compare_restart_results
from slope_stability.results import open_result
from tests.support.problem_fixtures import (
    assert_schema_subset,
    asset_document,
    case_document,
    profile_document,
    stamped,
)
from tests.resolution.test_resolution_contract import parsed_profiles, resource


ROOT = Path(__file__).resolve().parents[2]
BUILD = Path(os.environ["SLOPE_BUILD_DIR"])


def constraint_identity_fan_mesh() -> CanonicalMesh:
    """Create five cells incident to one nonzero prescribed boundary value."""

    vertices = np.asarray(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.5, 1.5),
            (0.0, 2.0),
            (-0.5, 1.5),
            (-1.0, 1.0),
        ),
        dtype=np.float64,
    )
    cells = np.asarray(
        tuple((0, ordinal, ordinal + 1) for ordinal in range(1, 6)),
        dtype=np.uint64,
    )
    exterior = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (0, 6),
    )
    return canonicalize_mesh(
        RawMesh(
            dimension=2,
            cell_type="triangle",
            coordinate_order=1,
            source_node_ids=np.arange(vertices.shape[0], dtype=np.uint64),
            coordinate_nodes=vertices,
            cells=cells,
            coordinate_connectivity=cells,
            supports=(
                RawSupport(
                    "domain",
                    "region",
                    2,
                    tuple(tuple(int(item) for item in row) for row in cells),
                ),
                RawSupport("base", "boundary", 1, ((0, 1),)),
                RawSupport("top", "boundary", 1, ((3, 4),)),
                RawSupport("boundary", "boundary", 1, exterior),
            ),
            partition_hints=np.zeros(cells.shape[0], dtype=np.int64),
            importer_id="constraint-identity-fan",
        )
    )


def prescribed_saturation_mesh() -> CanonicalMesh:
    """Unit square, 128 triangles: enough P2 work to enter hybrid assembly."""
    n = 8
    vertices = np.asarray([(i / n, j / n) for j in range(n + 1) for i in range(n + 1)])
    cells = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            cells.extend(((a, a + 1, a + n + 1), (a + 1, a + n + 2, a + n + 1)))
    base = tuple((i, i + 1) for i in range(n))
    top = tuple((n * (n + 1) + i, n * (n + 1) + i + 1) for i in range(n))
    sides = tuple((j * (n + 1), (j + 1) * (n + 1)) for j in range(n)) + tuple(
        (j * (n + 1) + n, (j + 1) * (n + 1) + n) for j in range(n)
    )
    connectivity = np.asarray(cells, dtype=np.uint64)
    return canonicalize_mesh(
        RawMesh(
            dimension=2,
            cell_type="triangle",
            coordinate_order=1,
            source_node_ids=np.arange(len(vertices), dtype=np.uint64),
            coordinate_nodes=vertices,
            cells=connectivity,
            coordinate_connectivity=connectivity,
            partition_hints=np.zeros(len(cells), dtype=np.int64),
            importer_id="prescribed-saturation-unit-square-v1",
            supports=(
                RawSupport("domain", "region", 2, tuple(cells)),
                RawSupport("base", "boundary", 1, base),
                RawSupport("top", "boundary", 1, top),
                RawSupport("boundary", "boundary", 1, base + top + sides),
            ),
        )
    )


@contextmanager
def saturation_artifact_directory(prefix: str):
    root = os.environ.get("SLOPE_SATURATION_TEST_ARTIFACTS")
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
        yield tempfile.mkdtemp(prefix=prefix, dir=root)
    else:
        with tempfile.TemporaryDirectory(prefix=prefix) as directory:
            yield directory


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    openmp_threads: int = 1,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    runtime = os.environ.get("SLOPE_RUNTIME_LIBRARY_PATH", "")
    prior = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = ":".join(item for item in (runtime, prior) if item)
    environment.update(
        {
            "OMP_NUM_THREADS": str(openmp_threads),
            "OMP_DYNAMIC": "FALSE",
            "OMP_MAX_ACTIVE_LEVELS": "1",
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    root = os.environ.get("SLOPE_SATURATION_TEST_ARTIFACTS")
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
        recorded = (
            "LD_LIBRARY_PATH",
            "OMP_NUM_THREADS",
            "OMP_DYNAMIC",
            "OMP_MAX_ACTIVE_LEVELS",
            "OMP_PROC_BIND",
            "OMP_PLACES",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="command-", suffix=".json", dir=root, delete=False
        ) as stream:
            json.dump(
                {
                    "argv": command,
                    "cwd": str(cwd or ROOT),
                    "start": started,
                    "end": time.time(),
                    "exit_code": completed.returncode,
                    "environment": {key: environment.get(key) for key in recorded},
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
                stream,
                indent=2,
            )
    return completed


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Result-contract outcome family implied by each successful terminal stop reason
# (SOL-07: a horizon or step-limit stop is not reported as converged).
STOP_REASON_OUTCOMES = {
    "scaled_slope_reached": "converged_criterion",
    "minimum_step_reached": "converged_criterion",
    "work_horizon_reached": "completed_control_horizon",
    "direct_control_horizon_reached": "completed_control_horizon",
    "accepted_step_limit": "completed_requested_step_horizon",
    "converged": "converged_criterion",
}


class InstalledFixture:
    def __init__(
        self,
        temporary: Path,
        ranks: int,
        *,
        hydraulic: bool = False,
        coupled: bool = False,
        coupling_mode: str = "pressure_gradient_and_weight",
        dimension: int = 2,
        field_order: int = 1,
        two_cells: bool = False,
        analysis: str = "limit_load.work_controlled",
        young_modulus: float = 40000.0,
        unit_weight: float = 20.0,
        continuation_initial_step: float | None = None,
        maximum_accepted_states: int = 3,
        deflation_enabled: bool = False,
        full_mechanics_outputs: bool = False,
        openmp_threads: int = 1,
        output_frequency: str | None = None,
        checkpoint_policy: CheckpointPolicy | None = None,
        additional_boundary_rules: tuple[dict[str, object], ...] = (),
        affine_head: bool = False,
        constraint_identity_fan: bool = False,
        prescribed_saturation: dict | None = None,
        saturated_unit_weight: float | None = None,
        mesh_override: CanonicalMesh | None = None,
        problem_mutation=None,
        volume_quadrature: str | None = None,
        qualification_mode: bool | None = None,
    ) -> None:
        self.temporary = temporary
        self.ranks = ranks
        self.openmp_threads = openmp_threads
        self.prefix = temporary / "prefix"
        installed = run(
            ["cmake", "--install", str(BUILD), "--prefix", str(self.prefix)]
        )
        if installed.returncode != 0:
            raise AssertionError(installed.stderr)
        self.engine = self.prefix / "bin/slope-engine"
        self.probe = self.prefix / "bin/slope-probe"
        data = self.prefix / "share/slope-stability"
        build_manifest = json.loads((data / "build-manifest.json").read_text())
        install_manifest = json.loads((data / "install-manifest.json").read_text())
        features = build_manifest["features"]
        dependencies = build_manifest["dependency"]
        build = BuildCapabilities(
            build_id=build_manifest["build_id"],
            executable_id=install_manifest["executable_id"],
            dependency_id=build_manifest["dependency_id"],
            schema_version=(1, 0, 0),
            petsc_int_bits=dependencies["petsc_int_bits"],
            mpi=True,
            openmp=features["openmp"],
            parallel_hdf5=features["hdf5_parallel"],
            hypre=features["petsc_hypre"],
            parmetis=features["petsc_partitioner"],
            hydraulic_pmg_budget=features.get("hydraulic_pmg_budget", False),
        )
        asset = load_document(
            asset_document(dimension, coordinate_order=1), source_path="asset.json"
        )
        case_raw = case_document(
            dimension,
            field_order=field_order,
            coordinate_order=1,
            analysis=analysis,
        )
        if volume_quadrature is not None:
            case_raw["discretization"]["volume_quadrature"] = volume_quadrature
            case_raw = stamped(case_raw)
        if not hydraulic and (young_modulus != 40000.0 or unit_weight != 20.0):
            parameters = case_raw["materials"][0]["parameters"]
            parameters["young_modulus"]["value"] = young_modulus
            parameters["unit_weight"]["value"] = unit_weight
            case_raw = stamped(case_raw)
        profiles = list(parsed_profiles())
        if full_mechanics_outputs and not hydraulic and not coupled:
            case_raw["required_outputs"] = [
                "displacement",
                "strain",
                "stress",
                "material_branch",
                "plastic_indicator",
                "yield_residual",
            ]
            case_raw = stamped(case_raw)
            profiles = [profile for profile in profiles if profile.family != "output"]
            profiles.append(
                load_document(
                    profile_document(
                        "output",
                        {
                            "quantities": case_raw["required_outputs"],
                            "history": True,
                            "frequency": "accepted_state",
                        },
                    ),
                    source_path="output.json",
                )
            )
        if deflation_enabled:
            case_raw["profiles"]["deflation"] = "deflation.app"
            case_raw = stamped(case_raw)
            profiles = [
                profile for profile in profiles if profile.family != "deflation"
            ]
            profiles.append(
                load_document(
                    profile_document(
                        "deflation",
                        {
                            "enabled": True,
                            "projector": "general_left_right",
                            "maximum_dimension": 4,
                        },
                        name="deflation.app",
                    ),
                    source_path="deflation.json",
                )
            )
        if coupled:
            case_raw["materials"].append(
                {
                    "name": "water-bearing-soil",
                    "support": "domain",
                    "model": "steady_unconfined_darcy",
                    "parameters": {
                        "conductivity": {
                            "value": 1.0,
                            "unit": "m/s",
                        },
                        "water_unit_weight": {
                            "value": 9.81,
                            "unit": "kN/m^3",
                        },
                    },
                }
            )
            case_raw["physics"] = {
                "mechanics": True,
                "seepage": True,
                "coupling": coupling_mode,
            }
            case_raw["boundary_rules"].append(
                {
                    "name": "base-head",
                    "field": "pore_pressure",
                    "condition": "essential",
                    "support": "base",
                    "components": ["scalar"],
                    "value_model": "hydrostatic",
                    "parameters": {
                        "reference_level": 1.0,
                        "reference_pressure": 0.0,
                        "unit_weight": 9.81,
                        "length_unit": "m",
                        "pressure_unit": "kPa",
                    },
                }
            )
            case_raw["required_outputs"] = [
                "displacement",
                "strain",
                "stress",
                "material_branch",
                "plastic_indicator",
                "yield_residual",
                "pore_pressure",
                "pressure_gradient",
                "conductivity",
                "hydraulic_ramp",
                "hydraulic_epsilon",
                "saturation",
                "pressure_flux",
                "darcy_velocity",
                "coupled_body_force",
                "global_flux_balance",
            ]
            case_raw = stamped(case_raw)
            profiles = [
                profile
                for profile in profiles
                if profile.family not in {"output", "seepage"}
            ]
            profiles.append(
                load_document(
                    profile_document(
                        "output",
                        {
                            "quantities": case_raw["required_outputs"],
                            "history": True,
                            "frequency": "accepted_state",
                        },
                    ),
                    source_path="output.json",
                )
            )
            profiles.append(
                load_document(
                    profile_document(
                        "seepage",
                        {
                            "linear_tolerance": 1.0e-10,
                            "linear_max_iter": 300,
                            "nonlinear_max_iter": 100,
                        },
                        name="seepage.app-tight",
                    ),
                    source_path="seepage.json",
                )
            )
        elif hydraulic:
            case_raw.update(
                {
                    "materials": [
                        {
                            "name": "water-bearing-soil",
                            "support": "domain",
                            "model": "steady_unconfined_darcy",
                            "parameters": {
                                "conductivity": {
                                    "value": 1.0,
                                    "unit": "m/s",
                                },
                                "water_unit_weight": {
                                    "value": 9.81,
                                    "unit": "kN/m^3",
                                },
                            },
                        }
                    ],
                    "physics": {
                        "mechanics": False,
                        "seepage": True,
                        "coupling": "none",
                    },
                    "boundary_rules": [
                        {
                            "name": "base-head",
                            "field": "pore_pressure",
                            "condition": "essential",
                            "support": "base",
                            "components": ["scalar"],
                            "value_model": (
                                "affine_coordinate" if affine_head else "hydrostatic"
                            ),
                            "parameters": (
                                {
                                    "offset": 9.81,
                                    "gradient": [0.0, -9.81]
                                    + ([0.0] if dimension == 3 else []),
                                    "unit": "kPa",
                                }
                                if affine_head
                                else {
                                    "reference_level": 1.0,
                                    "reference_pressure": 0.0,
                                    "unit_weight": 9.81,
                                    "length_unit": "m",
                                    "pressure_unit": "kPa",
                                }
                            ),
                        }
                    ],
                    "analysis": {"selector": "seepage.steady_unconfined"},
                    "required_outputs": [
                        "pore_pressure",
                        "pressure_gradient",
                        "conductivity",
                        "hydraulic_ramp",
                        "hydraulic_epsilon",
                        "saturation",
                        "pressure_flux",
                        "darcy_velocity",
                        "global_flux_balance",
                    ],
                }
            )
            case_raw = stamped(case_raw)
            profiles = [profile for profile in profiles if profile.family != "output"]
            profiles.append(
                load_document(
                    profile_document(
                        "output",
                        {
                            "quantities": case_raw["required_outputs"],
                            "history": True,
                            "frequency": "accepted_state",
                        },
                    ),
                    source_path="output.json",
                )
            )
            profiles.append(
                load_document(
                    profile_document(
                        "seepage",
                        {
                            "linear_tolerance": 1.0e-10,
                            "linear_max_iter": 300,
                            "nonlinear_max_iter": 100,
                        },
                        name="seepage.app-tight",
                    ),
                    source_path="seepage.json",
                )
            )
        if additional_boundary_rules:
            case_raw["boundary_rules"].extend(additional_boundary_rules)
            case_raw = stamped(case_raw)
        if prescribed_saturation is not None:
            case_raw["prescribed_saturation"] = deepcopy(prescribed_saturation)
        if saturated_unit_weight is not None:
            parameters = case_raw["materials"][0]["parameters"]
            parameters["unsaturated_unit_weight"] = parameters.pop("unit_weight")
            parameters["saturated_unit_weight"] = {
                "value": saturated_unit_weight,
                "unit": "kN/m^3",
            }
        case_raw = stamped(case_raw)
        (temporary / "case.json").write_bytes(canonicalize(case_raw))
        case = load_document(case_raw, source_path="case.json")
        if constraint_identity_fan and dimension != 2:
            raise ValueError("constraint identity fan is a 2D fixture")
        package = compile_problem(
            asset=asset,
            case=case,
            mesh=(
                mesh_override
                if mesh_override is not None
                else constraint_identity_fan_mesh()
                if constraint_identity_fan
                else compact_simplex_mesh(
                    dimension,
                    1,
                    two_cells=two_cells,
                )
            ),
            destination=temporary / "problems",
        )
        if problem_mutation is not None:
            manifest = json.loads(package.problem_json.read_bytes())
            manifest.pop("id")
            problem_mutation(manifest)
            new_id = content_id(manifest)
            manifest["id"] = new_id
            package.problem_json.write_bytes(canonicalize(manifest))
            new_path = package.path.parent / new_id.replace(":", "-")
            package.path.rename(new_path)
            package = replace(
                package,
                problem_id=new_id,
                path=new_path,
                problem_json=new_path / "problem.json",
                compile_report=new_path / "compile-report.json",
            )
        baseline = resource()
        selected_resource = baseline.model_copy(
            update={
                "name": f"local-{ranks}x{openmp_threads}",
                "world_size": ranks,
                "ranks_per_node": ranks,
                "openmp_threads": openmp_threads,
                "cores_per_rank": openmp_threads,
                "active_ranks": ActiveRanks(
                    fine=tuple(range(ranks)),
                    coarse=tuple(range(ranks)),
                ),
            }
        )
        case_overrides = {}
        if not hydraulic:
            case_overrides["continuation.maximum_accepted_states"] = (
                maximum_accepted_states
            )
        if continuation_initial_step is not None:
            case_overrides.update(
                {
                    "continuation.initial_step": continuation_initial_step,
                    "continuation.maximum_step": continuation_initial_step,
                }
            )
        if output_frequency is not None:
            case_overrides["outputs.frequency"] = output_frequency
        if qualification_mode is not None:
            case_overrides["validation.qualification_mode"] = qualification_mode
        resolved = resolve_run(
            problem=load_compiled_problem(package.problem_json),
            profiles=profiles,
            resource=selected_resource,
            build=build,
            case_overrides=case_overrides or None,
            checkpoint=checkpoint_policy,
        )
        self.problem_dir = package.path
        self.problem_id = package.problem_id
        self.run_json = write_resolved_run(resolved, temporary / "runs")
        self.run_id = resolved.run_id
        self.numerical_policy_id = resolved.manifest["result_policy"][
            "numerical_policy_id"
        ]
        self.build_id = build.build_id
        self.executable_id = build.executable_id
        self.dependency_id = build.dependency_id
        self.install_manifest = install_manifest

    def rewrite_run(self, mutation) -> Path:
        document = json.loads(self.run_json.read_bytes())
        document.pop("id")
        mutation(document)
        identity = content_id(document)
        mutated = {"id": identity, **document}
        target = self.temporary / f"mutated-{identity[7:19]}.json"
        target.write_bytes(canonicalize(mutated))
        return target

    def validate(
        self,
        result_dir: Path,
        *,
        launched_ranks: int | None = None,
        run_json: Path | None = None,
        operation: str = "validate",
        engine_options: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess[str]:
        return run(
            [
                os.environ["SLOPE_MPIEXEC"],
                os.environ.get("SLOPE_MPI_NUMPROC_FLAG", "-n"),
                str(self.ranks if launched_ranks is None else launched_ranks),
                str(self.engine),
                operation,
                str(self.problem_dir),
                str(self.run_json if run_json is None else run_json),
                str(result_dir),
                *engine_options,
            ],
            cwd=self.prefix,
            openmp_threads=self.openmp_threads,
        )


class EngineContract(unittest.TestCase):
    def assert_content_identity(self, path: Path) -> dict:
        document = json.loads(path.read_bytes())
        identity_input = dict(document)
        supplied = identity_input.pop("id")
        self.assertEqual(supplied, content_id(identity_input))
        self.assertEqual(path.read_bytes(), canonicalize(document))
        schema_name = {
            "slope_stability.error": "error.schema.json",
            "slope_stability.events": "events.schema.json",
            "slope_stability.result": "result.schema.json",
        }.get(document.get("kind"))
        if schema_name is not None:
            schema = json.loads(
                (ROOT / "schemas/generated/json" / schema_name).read_text()
            )
            assert_schema_subset(document, schema)
        return document

    def assert_collective_failure(
        self,
        completed: subprocess.CompletedProcess[str],
        result_dir: Path,
        *,
        exit_code: int,
        code: str,
    ) -> None:
        self.assertEqual(completed.returncode, exit_code, completed.stderr)
        result = self.assert_content_identity(result_dir / "result.json")
        error = self.assert_content_identity(result_dir / "error.json")
        self.assertEqual(result["errors"]["count"], 1)
        self.assertEqual(result["errors"]["primary_id"], error["id"])
        self.assertEqual(error["code"]["name"], code)
        self.assertEqual(
            sum(f'"code":"{code}"' in line for line in completed.stderr.splitlines()),
            1,
            completed.stderr,
        )

    def assert_mechanics_equilibrium_scalars(
        self, result: dict, dimension: int
    ) -> None:
        force_unit = "kN/m" if dimension == 2 else "kN"
        scalar_records = {
            scalar["id"]: scalar for scalar in result["summary"]["scalars"]
        }
        reaction_ids = tuple(
            f"mechanics_reaction_{component}"
            for component in ("x", "y", "z")[:dimension]
        )
        external_ids = tuple(
            f"mechanics_external_resultant_{component}"
            for component in ("x", "y", "z")[:dimension]
        )
        force_ids = (
            *reaction_ids,
            *external_ids,
            "mechanics_reaction_norm",
            "mechanics_external_resultant_norm",
            "mechanics_equilibrium_imbalance",
            "mechanics_free_residual_l2",
        )
        for quantity_id in force_ids:
            self.assertIn(quantity_id, scalar_records)
            self.assertEqual(scalar_records[quantity_id]["unit"], force_unit)
            self.assertTrue(scalar_records[quantity_id]["complete"])
        normalized = scalar_records["mechanics_normalized_equilibrium"]
        self.assertEqual(normalized["unit"], "dimensionless")
        self.assertTrue(normalized["complete"])
        reactions = tuple(scalar_records[item]["value"] for item in reaction_ids)
        external = tuple(scalar_records[item]["value"] for item in external_ids)
        reaction_norm = math.sqrt(sum(value * value for value in reactions))
        external_norm = math.sqrt(sum(value * value for value in external))
        imbalance = math.sqrt(
            sum(
                (reaction + applied) * (reaction + applied)
                for reaction, applied in zip(reactions, external, strict=True)
            )
        )
        self.assertAlmostEqual(
            scalar_records["mechanics_reaction_norm"]["value"],
            reaction_norm,
            places=12,
        )
        self.assertAlmostEqual(
            scalar_records["mechanics_external_resultant_norm"]["value"],
            external_norm,
            places=12,
        )
        self.assertAlmostEqual(
            scalar_records["mechanics_equilibrium_imbalance"]["value"],
            imbalance,
            places=12,
        )
        scale = max(reaction_norm, external_norm)
        expected_normalized = imbalance / scale if scale > 0.0 else 0.0
        self.assertAlmostEqual(normalized["value"], expected_normalized, places=12)
        self.assertLessEqual(normalized["value"], 1.0e-8)
        self.assertLessEqual(
            scalar_records["mechanics_free_residual_l2"]["value"],
            1.0e-8 * max(1.0, scale),
        )

    def assert_accepted_checkpoint_roundtrip(
        self,
        temporary: Path,
        *,
        source_ranks: int,
        source_threads: int,
        restart_ranks: int,
        restart_threads: int,
        fixture_options: dict[str, object],
        quantities: tuple[tuple[str, str], ...],
        require_deflation: bool,
    ) -> None:
        target_options = dict(fixture_options)
        source_options = dict(fixture_options)
        source_accepted_states = int(
            source_options.pop("checkpoint_source_accepted_states", 2)
        )
        target_options.pop("checkpoint_source_accepted_states", None)
        if not bool(fixture_options.get("hydraulic", False)):
            target_options["maximum_accepted_states"] = max(
                3,
                int(fixture_options.get("maximum_accepted_states", 3)),
            )
            source_options["maximum_accepted_states"] = source_accepted_states
        baseline_fixture = InstalledFixture(
            temporary / "baseline",
            restart_ranks,
            openmp_threads=restart_threads,
            output_frequency="terminal",
            **target_options,
        )
        baseline_result_dir = temporary / "baseline-result"
        completed = baseline_fixture.validate(
            baseline_result_dir,
            operation="run",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        baseline_envelope = json.loads(
            (baseline_result_dir / "result.json").read_bytes()
        )
        source_fixture = InstalledFixture(
            temporary / "source",
            source_ranks,
            openmp_threads=source_threads,
            checkpoint_policy=CheckpointPolicy(mode="write_accepted"),
            **source_options,
        )
        source_result_dir = temporary / "source-result"
        completed = source_fixture.validate(
            source_result_dir,
            operation="run",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        source_envelope = json.loads((source_result_dir / "result.json").read_bytes())
        # SOL-07: a mechanics checkpoint source stops at its requested step
        # horizon and must not be reported as converged.
        source_stop = source_envelope["terminal_status"]["stop_reason"]
        self.assertIn(source_stop, {"accepted_step_limit", "converged"})
        self.assertEqual(
            source_envelope["analysis_outcome"]["outcome"],
            STOP_REASON_OUTCOMES[source_stop],
        )
        source_checkpoints = source_envelope["summary"]["checkpoints"]
        self.assertEqual(len(source_checkpoints), 1)
        checkpoint_record = source_checkpoints[0]
        self.assertEqual(
            checkpoint_record["state_id"],
            source_envelope["summary"]["accepted_state_count"],
        )
        checkpoint_dir = (source_result_dir / checkpoint_record["path"]).resolve(
            strict=True
        )
        checkpoint_envelope = json.loads(
            (checkpoint_dir / "checkpoint.json").read_bytes()
        )
        self.assertEqual(
            checkpoint_envelope["id"],
            checkpoint_record["checkpoint_id"],
        )
        self.assertTrue((checkpoint_dir / "checkpoint.h5").is_file())
        restart_fixture = InstalledFixture(
            temporary / "restart",
            restart_ranks,
            openmp_threads=restart_threads,
            output_frequency="terminal",
            checkpoint_policy=CheckpointPolicy(
                mode="restore_accepted",
                path=str(checkpoint_dir),
            ),
            **target_options,
        )
        self.assertEqual(source_fixture.problem_id, restart_fixture.problem_id)
        self.assertEqual(baseline_fixture.problem_id, restart_fixture.problem_id)
        self.assertEqual(
            source_fixture.numerical_policy_id,
            restart_fixture.numerical_policy_id,
        )
        self.assertEqual(
            baseline_fixture.numerical_policy_id,
            restart_fixture.numerical_policy_id,
        )
        restart_result_dir = temporary / "restart-result"
        completed = restart_fixture.validate(
            restart_result_dir,
            operation="run",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        restart_envelope = json.loads((restart_result_dir / "result.json").read_bytes())
        restart_provenance = restart_envelope["provenance"]["restart"]
        self.assertEqual(
            restart_provenance,
            {
                "mode": "accepted_boundary",
                "parent_checkpoint_id": checkpoint_record["checkpoint_id"],
                "parent_run_id": source_envelope["run"]["run_id"],
                "parent_build_id": source_envelope["run"]["build_id"],
                "parent_accepted_state_id": checkpoint_record["state_id"],
                "source_index_bits": checkpoint_envelope["compatibility"][
                    "source_index_bits"
                ],
            },
        )
        self.assertEqual(
            restart_envelope["terminal_status"]["status"],
            "succeeded",
        )
        self.assertEqual(
            restart_envelope["analysis_outcome"],
            baseline_envelope["analysis_outcome"],
        )
        self.assertGreaterEqual(
            restart_envelope["summary"]["accepted_state_count"],
            1,
        )
        self.assertEqual(restart_envelope["summary"]["checkpoints"], [])
        restart_environment = json.loads(
            (restart_result_dir / "environment.json").read_bytes()
        )
        self.assertEqual(
            restart_environment["mpi"]["world_size"],
            restart_ranks,
        )
        self.assertEqual(
            restart_environment["openmp"]["maximum_threads"],
            restart_threads,
        )
        baseline_result = open_result(baseline_result_dir)
        source_result = open_result(source_result_dir)
        restart_result = open_result(restart_result_dir)
        self.assertIsNone(baseline_result.restart)
        self.assertIsNotNone(restart_result.restart)
        self.assertEqual(
            restart_result.restart.parent_checkpoint_id,
            checkpoint_record["checkpoint_id"],
        )
        same_layout = (
            source_ranks == restart_ranks and source_threads == restart_threads
        )
        for quantity_id, accessor in quantities:
            baseline_values = getattr(baseline_result, accessor)(
                quantity_id,
                state="terminal",
            ).read()
            restart_values = getattr(restart_result, accessor)(
                quantity_id,
                state="terminal",
            ).read()
            self.assertEqual(
                tuple(map(int, baseline_values.stable_ids)),
                tuple(map(int, restart_values.stable_ids)),
                quantity_id,
            )
            if same_layout:
                self.assertEqual(
                    baseline_values.values.tobytes(),
                    restart_values.values.tobytes(),
                    quantity_id,
                )
            elif accessor == "category":
                self.assertEqual(
                    baseline_values.values.tobytes(),
                    restart_values.values.tobytes(),
                    quantity_id,
                )
            else:
                self.assertTrue(
                    np.isfinite(baseline_values.values).all()
                    and np.isfinite(restart_values.values).all(),
                    quantity_id,
                )
        baseline_scalars = {
            scalar["id"]: scalar["value"]
            for scalar in baseline_envelope["summary"]["scalars"]
        }
        restart_scalars = {
            scalar["id"]: scalar["value"]
            for scalar in restart_envelope["summary"]["scalars"]
        }
        self.assertEqual(set(baseline_scalars), set(restart_scalars))
        for quantity_id, baseline_value in baseline_scalars.items():
            restart_value = restart_scalars[quantity_id]
            if same_layout:
                self.assertTrue(
                    math.isclose(
                        baseline_value,
                        restart_value,
                        rel_tol=5.0e-13,
                        abs_tol=5.0e-13,
                    ),
                    (quantity_id, baseline_value, restart_value),
                )
            else:
                self.assertTrue(
                    math.isfinite(baseline_value) and math.isfinite(restart_value),
                    (quantity_id, baseline_value, restart_value),
                )
        with (
            h5py.File(
                source_result_dir / "result.h5",
                "r",
            ) as source_handle,
            h5py.File(
                baseline_result_dir / "result.h5",
                "r",
            ) as baseline_handle,
            h5py.File(
                restart_result_dir / "result.h5",
                "r",
            ) as restart_handle,
        ):
            source_states = source_handle["/states/index/rows"][...]
            baseline_states = baseline_handle["/states/index/rows"][...]
            restart_states = restart_handle["/states/index/rows"][...]
        checkpoint_ordinal = max(int(row["accepted_ordinal"]) for row in source_states)
        baseline_suffix = tuple(
            row
            for row in baseline_states
            if int(row["accepted_ordinal"]) >= checkpoint_ordinal
        )
        restart_suffix = tuple(
            row
            for row in restart_states
            if int(row["accepted_ordinal"]) >= checkpoint_ordinal
        )
        self.assertEqual(
            tuple(int(row["accepted_ordinal"]) for row in baseline_suffix),
            tuple(int(row["accepted_ordinal"]) for row in restart_suffix),
        )
        state_columns = (
            "factor",
            "work",
            "direct_control",
            "displacement_maximum",
        )
        for baseline_state, restart_state in zip(
            baseline_suffix,
            restart_suffix,
            strict=True,
        ):
            for column in state_columns:
                baseline_value = float(baseline_state[column])
                restart_value = float(restart_state[column])
                if same_layout:
                    self.assertEqual(
                        struct.pack("<d", baseline_value),
                        struct.pack("<d", restart_value),
                        column,
                    )
                else:
                    self.assertTrue(
                        math.isfinite(baseline_value) and math.isfinite(restart_value),
                        (column, baseline_value, restart_value),
                    )
        baseline_accepted = baseline_result.history("accepted").read()
        restart_accepted = restart_result.history("accepted").read()
        if bool(fixture_options.get("hydraulic", False)):
            baseline_trajectory = tuple(baseline_accepted)
            restart_trajectory = tuple(restart_accepted)
        else:
            baseline_trajectory = tuple(
                row
                for row in baseline_accepted
                if int(row["ordinal"]) > checkpoint_ordinal
            )
            restart_trajectory = tuple(
                row
                for row in restart_accepted
                if int(row["ordinal"]) > checkpoint_ordinal
            )
            self.assertGreater(len(baseline_trajectory), 0)
        self.assertEqual(
            tuple(int(row["ordinal"]) for row in baseline_trajectory),
            tuple(int(row["ordinal"]) for row in restart_trajectory),
        )
        for baseline_row, restart_row in zip(
            baseline_trajectory,
            restart_trajectory,
            strict=True,
        ):
            for column in ("control", "residual"):
                baseline_value = float(baseline_row[column])
                restart_value = float(restart_row[column])
                if same_layout:
                    self.assertEqual(
                        struct.pack("<d", baseline_value),
                        struct.pack("<d", restart_value),
                        column,
                    )
                else:
                    self.assertTrue(
                        math.isclose(
                            baseline_value,
                            restart_value,
                            rel_tol=5.0e-13,
                            abs_tol=5.0e-13,
                        ),
                        (column, baseline_value, restart_value),
                    )
        if require_deflation:
            for result in (baseline_result, restart_result):
                deflation_rows = result.history("deflation").read()
                self.assertGreaterEqual(deflation_rows.shape[0], 1)
                self.assertGreaterEqual(float(deflation_rows[0]["control"]), 1.0)
                self.assertLessEqual(
                    float(deflation_rows[0]["residual"]),
                    1.0e-9,
                )
        comparison = compare_restart_results(
            baseline_result_dir,
            source_result_dir,
            restart_result_dir,
            mode="same_layout" if same_layout else "repartitioned",
        )
        self.assertEqual(comparison["status"], "pass", comparison)
        self.assertEqual(comparison["supports_match"], True, comparison)
        if same_layout:
            self.assertEqual(
                comparison["bitwise_from_restart"],
                True,
                comparison,
            )
        baseline_result.close()
        source_result.close()
        restart_result.close()

    def test_cli(self) -> None:
        """APP-UNIT-CLI-001."""
        engine = Path(os.environ["SLOPE_ENGINE"])
        probe = Path(os.environ["SLOPE_PROBE"])
        for command in ([str(engine), "--help"], [str(engine), "--version"]):
            completed = run(command)
            self.assertEqual(completed.returncode, 0, completed.stderr)
        invalid = (
            [str(engine)],
            [str(engine), "solve", "a", "b", "c"],
            [str(engine), "validate", "a", "b"],
            [str(engine), "validate", "a", "b", "c", "-ksp_type", "cg"],
            [str(engine), "validate", "--problem", "a", "--run", "b", "--result", "c"],
            [str(probe)],
            [str(probe), "request.json"],
        )
        for command in invalid:
            with self.subTest(command=command):
                completed = run(command)
                self.assertEqual(completed.returncode, 2)
        with tempfile.TemporaryDirectory(prefix="engine-probe-") as raw:
            temporary = Path(raw)
            request = temporary / "request.json"
            request.write_text(
                '{"kind":"slope_stability.probe_request","operation":"identity"}',
                encoding="utf-8",
            )
            completed = run([str(probe), str(request), str(temporary / "result")])
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assert_content_identity(temporary / "result/probe.json")

    def test_prescribed_saturation_native(self) -> None:
        """APP-PRESCRIBED-SATURATION-001: load selection via installed native input."""
        mesh = prescribed_saturation_mesh()
        profile = {
            "axis": "x",
            "coordinates": [0.0, 1.0],
            "levels": [0.75, 0.25],
            "length_unit": "m",
        }
        references = {}
        build_manifest = json.loads((BUILD / "build-manifest.json").read_bytes())
        hybrid_threads = 8 if build_manifest["features"]["openmp"] else 1
        rows = [
            ("uniform", 1, 1, None, 30.0),
            ("saturated", 1, 1, {**profile, "levels": [2.0, 2.0]}, 20.0),
            ("partial", 1, 1, profile, 20.0),
            ("partial", 2, 1, profile, 20.0),
            ("partial", 4, hybrid_threads, profile, 20.0),
        ]
        for name, ranks, threads, prescribed, weight in rows:
            with self.subTest(name=name, ranks=ranks, threads=threads):
                with saturation_artifact_directory(
                    prefix=f"saturation-{name}-{ranks}x{threads}-"
                ) as raw:
                    fixture = InstalledFixture(
                        Path(raw),
                        ranks,
                        openmp_threads=threads,
                        field_order=2,
                        mesh_override=mesh,
                        unit_weight=weight,
                        saturated_unit_weight=30.0,
                        prescribed_saturation=prescribed,
                        full_mechanics_outputs=True,
                    )
                    result_dir = Path(raw) / "result"
                    completed = fixture.validate(result_dir, operation="run")
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    result = json.loads((result_dir / "result.json").read_bytes())
                    environment = json.loads(
                        (result_dir / "environment.json").read_bytes()
                    )
                    openmp = environment["openmp"]
                    self.assertEqual(environment["mpi"]["world_size"], ranks)
                    self.assertEqual(openmp["maximum_threads"], threads)
                    self.assertEqual(
                        openmp["mechanics_assembly_minimum_threads"], threads
                    )
                    self.assertEqual(
                        openmp["mechanics_assembly_maximum_threads"], threads
                    )
                    self.assert_mechanics_equilibrium_scalars(result, 2)
                    with open_result(result_dir) as verified:
                        field = verified.field("displacement", state="terminal").read()
                        observed = (
                            np.asarray(field.stable_ids).copy(),
                            np.asarray(field.values).copy(),
                        )
                    key = "uniform" if name == "saturated" else name
                    if key not in references:
                        references[key] = observed
                    else:
                        np.testing.assert_array_equal(observed[0], references[key][0])
                        np.testing.assert_allclose(
                            observed[1], references[key][1], rtol=1e-8, atol=1e-11
                        )
        self.assertEqual(set(references), {"uniform", "partial"})
        self.assertGreater(
            np.linalg.norm(references["uniform"][1] - references["partial"][1]), 1e-8
        )

    def test_prescribed_saturation_native_rejection(self) -> None:
        """APP-PRESCRIBED-SATURATION-NEG-001: valid identities, invalid profiles."""
        baseline = {
            "axis": 0,
            "coordinates": [0.0, 1.0],
            "levels": [0.75, 0.25],
            "length_unit": "m",
        }
        bad_profiles = [
            None,
            [],
            {**baseline, "axis": 2},
            {**baseline, "coordinates": [1.0, 0.0]},
            {**baseline, "levels": [1.0]},
            {**baseline, "length_unit": "cm"},
            {**baseline, "unknown": True},
        ]
        for index, profile in enumerate(bad_profiles):
            with self.subTest(index=index):
                with saturation_artifact_directory(
                    prefix=f"saturation-negative-{index}-"
                ) as raw:
                    fixture = InstalledFixture(
                        Path(raw),
                        2,
                        problem_mutation=lambda document: document.update(
                            prescribed_saturation=profile
                        ),
                    )
                    result_dir = Path(raw) / "result"
                    completed = fixture.validate(result_dir)
                    self.assert_collective_failure(
                        completed, result_dir, exit_code=2, code="SOLVER_GRAPH_CREATE"
                    )

    def test_validate_layout(self) -> None:
        """APP-VALIDATE-LAYOUT-001."""
        with tempfile.TemporaryDirectory(prefix="engine-hydraulic-coarse-") as raw:
            temporary = Path(raw)
            fixture = InstalledFixture(temporary, 2, hydraulic=True)
            manifest_path = fixture.prefix / "share/slope-stability/build-manifest.json"
            manifest = json.loads(manifest_path.read_text())
            bits = manifest["dependency"]["petsc_int_bits"]
            invalid_values = [0, -1, True, 1.5, "7", 2 ** (bits - 1)]
            if bits == 64:
                # One past the 64-bit PetscInt maximum is outside the strict
                # protocol integer range, so no run document can carry it.
                with self.assertRaisesRegex(CanonicalizationError, "integer_range"):
                    fixture.rewrite_run(
                        lambda document: document["pmg"].update(
                            hydraulic_base_iterations=2**63
                        )
                    )
                invalid_values.pop()
            for ordinal, value in enumerate(invalid_values):
                with self.subTest(hydraulic_base_iterations=value):
                    run_json = fixture.rewrite_run(
                        lambda document: document["pmg"].update(
                            hydraulic_base_iterations=value
                        )
                    )
                    result_dir = temporary / f"invalid-coarse-{ordinal}"
                    completed = fixture.validate(result_dir, run_json=run_json)
                    self.assert_collective_failure(
                        completed, result_dir, exit_code=2, code="SOLVER_GRAPH_CREATE"
                    )
        for ranks in (1, 2, 4):
            with self.subTest(ranks=ranks):
                with tempfile.TemporaryDirectory(
                    prefix=f"engine-layout-{ranks}-"
                ) as raw:
                    temporary = Path(raw)
                    fixture = InstalledFixture(temporary, ranks)
                    result_dir = temporary / "result"
                    completed = fixture.validate(result_dir)
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    result = self.assert_content_identity(result_dir / "result.json")
                    self.assertEqual(result["terminal_status"]["status"], "succeeded")
                    self.assertEqual(
                        result["analysis_outcome"]["operation"], "validate"
                    )
                    self.assertEqual(result["run"]["problem_id"], fixture.problem_id)
                    self.assertEqual(result["run"]["run_id"], fixture.run_id)
                    self.assertEqual(result["provenance"]["build_id"], fixture.build_id)
                    self.assertEqual(
                        result["provenance"]["executable_id"],
                        fixture.executable_id,
                    )
                    self.assertEqual(
                        result["provenance"]["dependency_id"],
                        fixture.dependency_id,
                    )
                    self.assertEqual(
                        result["provenance"]["schema_id"],
                        "sha256:" + fixture.install_manifest["schema_set_sha256"],
                    )
                    for identity_name in (
                        "identity_graph_id",
                        "launcher_id",
                        "machine_id",
                        "thread_id",
                    ):
                        self.assertRegex(
                            result["provenance"][identity_name],
                            r"^sha256:[0-9a-f]{64}$",
                        )
                    placement = result["provenance"]["placement"]
                    self.assertRegex(placement["id"], r"^sha256:[0-9a-f]{64}$")
                    self.assertEqual(len(placement["rank_ids"]), ranks)
                    self.assertTrue(
                        all(
                            re.fullmatch(r"sha256:[0-9a-f]{64}", identity)
                            for identity in placement["rank_ids"]
                        )
                    )
                    self.assertEqual(placement["ranks"], list(range(ranks)))
                    self.assertEqual(result["summary"]["resources"]["mpi_ranks"], ranks)
                    self.assertEqual(
                        result["summary"]["resources"]["openmp_threads"], 1
                    )
                    self.assertTrue((result_dir / "events.json").is_file())
                    self.assert_content_identity(result_dir / "events.json")

        negative_cases = (
            ("payload-corruption", 1, 4, "PAYLOAD_DIGEST"),
            ("schema-major", 1, 2, "SCHEMA_MAJOR_INCOMPATIBLE"),
            ("missing-feature", 1, 3, "BUILD_FEATURE_MISMATCH"),
            ("resource-layout", 2, 3, "RESOURCE_LAYOUT_MISMATCH"),
            ("rank-local-setup", 2, 4, "RANK_LOCAL_SETUP"),
            ("problem-contract", 1, 4, "PROBLEM_RUN_CONTRACT_MISMATCH"),
            ("nonlinear-regularization", 1, 2, "SOLVER_GRAPH_CREATE"),
            ("initial-delta-lambda-stop", 1, 2, "SOLVER_GRAPH_CREATE"),
            ("output-frequency", 1, 2, "SOLVER_GRAPH_CREATE"),
            ("output-history-type", 1, 2, "SOLVER_GRAPH_CREATE"),
            ("output-missing-required", 1, 2, "SOLVER_GRAPH_CREATE"),
            ("output-unknown", 1, 2, "SOLVER_GRAPH_CREATE"),
            ("output-unavailable", 1, 2, "SOLVER_GRAPH_CREATE"),
            ("checkpoint-mode", 1, 2, "SOLVER_GRAPH_CREATE"),
            ("checkpoint-relative-path", 1, 2, "SOLVER_GRAPH_CREATE"),
            ("checkpoint-nonnormalized-path", 1, 2, "SOLVER_GRAPH_CREATE"),
            ("checkpoint-nul-mode", 1, 2, "SOLVER_GRAPH_CREATE"),
            ("checkpoint-nul-path", 1, 2, "SOLVER_GRAPH_CREATE"),
            ("checkpoint-null", 1, 2, "SOLVER_GRAPH_CREATE"),
            ("checkpoint-policy-id", 1, 2, "SOLVER_GRAPH_CREATE"),
            ("noncanonical-manifest", 1, 4, "MANIFEST_CANONICAL_BYTES"),
        )
        for name, ranks, exit_code, code in negative_cases:
            with self.subTest(negative=name):
                with tempfile.TemporaryDirectory(
                    prefix=f"engine-negative-{name}-"
                ) as raw:
                    temporary = Path(raw)
                    fixture = InstalledFixture(temporary, ranks)
                    run_json = fixture.run_json
                    launched_ranks = ranks
                    if name == "payload-corruption":
                        payload = fixture.problem_dir / "problem.h5"
                        data = payload.read_bytes()
                        payload.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
                    elif name == "schema-major":
                        problem_json = fixture.problem_dir / "problem.json"
                        document = json.loads(problem_json.read_bytes())
                        document["schema_version"]["major"] = 2
                        problem_json.write_bytes(canonicalize(document))
                    elif name == "missing-feature":
                        run_json = fixture.rewrite_run(
                            lambda document: document["build"].update(
                                {"parallel_hdf5": False}
                            )
                        )
                    elif name == "resource-layout":
                        launched_ranks = 1
                    elif name == "rank-local-setup":
                        run_json = fixture.rewrite_run(
                            lambda document: document["resources"]["local_rank_map"][
                                1
                            ].update({"rank": 99})
                        )
                    elif name == "problem-contract":
                        run_json = fixture.rewrite_run(
                            lambda document: document["problem"].update(
                                {"dimension": 3}
                            )
                        )
                    elif name == "nonlinear-regularization":
                        run_json = fixture.rewrite_run(
                            lambda document: document["nonlinear"].update(
                                {"regularization_minimum": 0.0}
                            )
                        )
                    elif name == "initial-delta-lambda-stop":
                        run_json = fixture.rewrite_run(
                            lambda document: document["continuation"].update(
                                {
                                    "initial_nonlinear_stopping_criterion": (
                                        "absolute_delta_lambda"
                                    )
                                }
                            )
                        )
                    elif name == "output-frequency":
                        run_json = fixture.rewrite_run(
                            lambda document: document["outputs"].update(
                                {"frequency": "every_iteration"}
                            )
                        )
                    elif name == "output-history-type":
                        run_json = fixture.rewrite_run(
                            lambda document: document["outputs"].update(
                                {"history": "true"}
                            )
                        )
                    elif name == "output-missing-required":
                        run_json = fixture.rewrite_run(
                            lambda document: document["outputs"].update(
                                {"quantities": ["displacement"]}
                            )
                        )
                    elif name == "output-unknown":
                        run_json = fixture.rewrite_run(
                            lambda document: document["outputs"].update(
                                {
                                    "quantities": [
                                        "displacement",
                                        "stress",
                                        "unknown",
                                    ]
                                }
                            )
                        )
                    elif name == "output-unavailable":
                        run_json = fixture.rewrite_run(
                            lambda document: document["outputs"].update(
                                {
                                    "quantities": [
                                        "displacement",
                                        "stress",
                                        "pore_pressure",
                                    ]
                                }
                            )
                        )
                    elif name == "checkpoint-mode":
                        run_json = fixture.rewrite_run(
                            lambda document: document["result_policy"].update(
                                {"checkpoint": {"mode": "unsupported"}}
                            )
                        )
                    elif name == "checkpoint-relative-path":
                        run_json = fixture.rewrite_run(
                            lambda document: document["result_policy"].update(
                                {
                                    "checkpoint": {
                                        "mode": "restore_accepted",
                                        "path": "relative/checkpoint",
                                    }
                                }
                            )
                        )
                    elif name == "checkpoint-nonnormalized-path":
                        run_json = fixture.rewrite_run(
                            lambda document: document["result_policy"].update(
                                {
                                    "checkpoint": {
                                        "mode": "restore_accepted",
                                        "path": "/not/../normalized",
                                    }
                                }
                            )
                        )
                    elif name == "checkpoint-nul-mode":
                        run_json = fixture.rewrite_run(
                            lambda document: document["result_policy"].update(
                                {
                                    "checkpoint": {
                                        "mode": "write_accepted\x00ignored",
                                    }
                                }
                            )
                        )
                    elif name == "checkpoint-nul-path":
                        run_json = fixture.rewrite_run(
                            lambda document: document["result_policy"].update(
                                {
                                    "checkpoint": {
                                        "mode": "restore_accepted",
                                        "path": "/qualified/checkpoint\x00ignored",
                                    }
                                }
                            )
                        )
                    elif name == "checkpoint-null":
                        run_json = fixture.rewrite_run(
                            lambda document: document["result_policy"].update(
                                {"checkpoint": None}
                            )
                        )
                    elif name == "checkpoint-policy-id":
                        run_json = fixture.rewrite_run(
                            lambda document: document["result_policy"].update(
                                {"numerical_policy_id": "sha256:short"}
                            )
                        )
                    elif name == "noncanonical-manifest":
                        document = json.loads(run_json.read_bytes())
                        run_json = temporary / "noncanonical-run.json"
                        run_json.write_text(
                            json.dumps(document, indent=2), encoding="utf-8"
                        )
                    result_dir = temporary / "negative-result"
                    completed = fixture.validate(
                        result_dir,
                        launched_ranks=launched_ranks,
                        run_json=run_json,
                    )
                    self.assert_collective_failure(
                        completed,
                        result_dir,
                        exit_code=exit_code,
                        code=code,
                    )

        with self.subTest(operation="run-numerical-failure-is-truthful"):
            with tempfile.TemporaryDirectory(
                prefix="engine-run-numerical-failure-"
            ) as raw:
                temporary = Path(raw)
                fixture = InstalledFixture(
                    temporary,
                    1,
                    two_cells=True,
                    young_modulus=10.0,
                    unit_weight=1000.0,
                    continuation_initial_step=0.2,
                )

                def fail_numerically(document: dict) -> None:
                    document["nonlinear"]["maximum_iterations"] = 2
                    document["continuation"]["minimum_step"] = 0.2

                run_json = fixture.rewrite_run(fail_numerically)
                result_dir = temporary / "failed-run-result"
                completed = fixture.validate(
                    result_dir,
                    run_json=run_json,
                    operation="run",
                )
                self.assert_collective_failure(
                    completed,
                    result_dir,
                    exit_code=5,
                    code="ANALYSIS_NOT_CONVERGED",
                )
                result = json.loads((result_dir / "result.json").read_bytes())
                error = json.loads((result_dir / "error.json").read_bytes())
                self.assertEqual(
                    result["analysis_outcome"],
                    {
                        "capability": "numerical-analysis",
                        "operation": "run",
                        "solver_executed": True,
                    },
                )
                self.assertEqual(
                    result["terminal_status"]["status"],
                    "failed_numerical",
                )
                self.assertEqual(error["category"]["name"], "numerical")
                self.assertEqual(error["phase"]["name"], "analysis")
                self.assertIn('"category":"numerical"', completed.stderr)
                self.assertIn('"status":"failed_numerical"', completed.stderr)

        with self.subTest(operation="run-publishes-scientific-result"):
            with tempfile.TemporaryDirectory(prefix="engine-run-scientific-") as raw:
                temporary = Path(raw)
                fixture = InstalledFixture(
                    temporary,
                    1,
                    full_mechanics_outputs=True,
                )
                result_dir = temporary / "run-result"
                completed = fixture.validate(result_dir, operation="run")
                self.assertEqual(completed.returncode, 0, completed.stderr)
                verified = open_result(result_dir)
                result = json.loads((result_dir / "result.json").read_bytes())
                self.assertEqual(result["analysis_outcome"]["method"], "limit_load")
                self.assertEqual(
                    result["analysis_outcome"]["outcome"],
                    STOP_REASON_OUTCOMES[result["terminal_status"]["stop_reason"]],
                )
                run_document = json.loads(Path(fixture.run_json).read_bytes())
                self.assertEqual(
                    result["summary"]["validation_policy"],
                    run_document["provenance"]["profile_ids"]["validation"],
                )
                self.assertEqual(
                    result["terminal_status"]["status"],
                    "succeeded",
                )
                self.assertTrue((result_dir / "result.h5").is_file())
                quantity_ids = {
                    quantity["id"] for quantity in result["summary"]["quantities"]
                }
                self.assertEqual(
                    quantity_ids,
                    {
                        "displacement",
                        "material_branch",
                        "plastic_indicator",
                        "strain",
                        "stress",
                        "yield_residual",
                    },
                )
                accepted_state_count = result["summary"]["accepted_state_count"]
                self.assertGreater(accepted_state_count, 1)
                for quantity in result["summary"]["quantities"]:
                    self.assertEqual(
                        quantity["state_ids"],
                        list(range(1, accepted_state_count + 1)),
                    )
                    self.assertEqual(
                        quantity["state_count"],
                        accepted_state_count,
                    )
                initial_displacement = verified.field(
                    "displacement",
                    state=1,
                ).read()
                terminal_displacement = verified.field(
                    "displacement",
                    state="terminal",
                ).read()
                self.assertTrue(
                    all(
                        math.isfinite(float(value))
                        for value in initial_displacement.values.flat
                    )
                )
                self.assertNotEqual(
                    tuple(map(float, initial_displacement.values.flat)),
                    tuple(map(float, terminal_displacement.values.flat)),
                )
                strain = verified.quadrature("strain", state="terminal").read()
                self.assertEqual(strain.component_ids, ("xx", "yy", "xy"))
                branch = verified.category(
                    "material_branch",
                    state="terminal",
                )
                self.assertEqual(
                    branch.category_ids,
                    ("elastic", "smooth", "left_edge", "right_edge", "apex"),
                )
                plastic = verified.category(
                    "plastic_indicator",
                    state="terminal",
                )
                self.assertEqual(plastic.category_ids, ("elastic", "plastic"))
                yield_residual = verified.quadrature(
                    "yield_residual",
                    state="terminal",
                )
                self.assertEqual(yield_residual.component_ids, ("yield",))
                plastic_values = plastic.read().values.reshape(-1)
                yield_values = yield_residual.read().values.reshape(-1)
                self.assertEqual(plastic_values.shape, yield_values.shape)
                for indicator, residual in zip(
                    plastic_values,
                    yield_values,
                    strict=True,
                ):
                    self.assertTrue(math.isfinite(float(residual)))
                    if int(indicator) == 1:
                        self.assertLessEqual(abs(float(residual)), 2.0e-9)
                verified.close()
                scalar_ids = {scalar["id"] for scalar in result["summary"]["scalars"]}
                self.assertIn("load_factor", scalar_ids)
                self.assert_mechanics_equilibrium_scalars(result, 2)

        with self.subTest(operation="run-honors-terminal-output-frequency"):
            with tempfile.TemporaryDirectory(prefix="engine-run-terminal-") as raw:
                temporary = Path(raw)
                fixture = InstalledFixture(temporary, 1)
                run_json = fixture.rewrite_run(
                    lambda document: document["outputs"].update(
                        {"frequency": "terminal"}
                    )
                )
                result_dir = temporary / "run-result"
                completed = fixture.validate(
                    result_dir,
                    run_json=run_json,
                    operation="run",
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                result = json.loads((result_dir / "result.json").read_bytes())
                terminal_state_id = result["summary"]["accepted_state_count"]
                self.assertEqual(
                    {quantity["id"] for quantity in result["summary"]["quantities"]},
                    {"displacement", "stress"},
                )
                for quantity in result["summary"]["quantities"]:
                    self.assertEqual(quantity["state_ids"], [terminal_state_id])
                    self.assertEqual(quantity["state_count"], 1)

        with self.subTest(operation="run-publishes-3d-p2-equilibrium"):
            with tempfile.TemporaryDirectory(prefix="engine-run-3d-p2-") as raw:
                temporary = Path(raw)
                fixture = InstalledFixture(
                    temporary,
                    1,
                    dimension=3,
                    field_order=2,
                )
                result_dir = temporary / "run-result"
                completed = fixture.validate(result_dir, operation="run")
                self.assertEqual(completed.returncode, 0, completed.stderr)
                verified = open_result(result_dir)
                stress = verified.quadrature("stress", state="terminal").read()
                self.assertEqual(
                    stress.component_ids,
                    ("xx", "yy", "zz", "xy", "yz", "xz"),
                )
                verified.close()
                result = json.loads((result_dir / "result.json").read_bytes())
                self.assert_mechanics_equilibrium_scalars(result, 3)

        with self.subTest(operation="run-affine-essential-boundaries"):
            prescribed_displacement = 2.0e-5
            for dimension in (2, 3):
                gradient = [-prescribed_displacement] * (dimension - 1) + [0.0]
                affine_rule = (
                    {
                        "name": "affine-top-x",
                        "field": "displacement",
                        "condition": "essential",
                        "support": "top",
                        "components": ["x"],
                        "value_model": "affine_coordinate",
                        "parameters": {
                            "offset": prescribed_displacement,
                            "gradient": gradient,
                            "unit": "m",
                        },
                    },
                )
                compact_rule = (
                    {
                        "name": "compact-base-x",
                        "field": "displacement",
                        "condition": "essential",
                        "support": "base",
                        "components": ["x"],
                        "value_model": "constant_vector",
                        "parameters": {
                            "value": [0.0],
                            "unit": "m",
                        },
                    },
                )
                rank_reference: tuple[float, ...] | None = None
                for ranks in (1, 2):
                    with self.subTest(
                        dimension=dimension,
                        ranks=ranks,
                        field="displacement",
                    ):
                        with tempfile.TemporaryDirectory(
                            prefix=f"engine-affine-displacement-{dimension}d-"
                        ) as raw:
                            temporary = Path(raw)
                            fixture = InstalledFixture(
                                temporary,
                                ranks,
                                dimension=dimension,
                                maximum_accepted_states=1,
                                additional_boundary_rules=affine_rule + compact_rule,
                            )
                            result_dir = temporary / "run-result"
                            completed = fixture.validate(result_dir, operation="run")
                            self.assertEqual(
                                completed.returncode,
                                0,
                                completed.stderr,
                            )
                            verified = open_result(result_dir)
                            displacement = verified.field(
                                "displacement",
                                state="terminal",
                            ).read()
                            verified.close()
                            observed = tuple(
                                sorted(float(row[0]) for row in displacement.values)
                            )
                            self.assertEqual(len(observed), dimension + 1)
                            for value in observed[:-1]:
                                self.assertAlmostEqual(value, 0.0, places=14)
                            self.assertAlmostEqual(
                                observed[-1],
                                prescribed_displacement,
                                places=14,
                            )
                            if rank_reference is None:
                                rank_reference = observed
                            else:
                                self.assertEqual(observed, rank_reference)

                rank_reference = None
                for ranks in (1, 2):
                    with self.subTest(
                        dimension=dimension,
                        ranks=ranks,
                        field="pore_pressure",
                    ):
                        with tempfile.TemporaryDirectory(
                            prefix=f"engine-affine-head-{dimension}d-"
                        ) as raw:
                            temporary = Path(raw)
                            fixture = InstalledFixture(
                                temporary,
                                ranks,
                                hydraulic=True,
                                dimension=dimension,
                                affine_head=True,
                            )
                            result_dir = temporary / "run-result"
                            completed = fixture.validate(result_dir, operation="run")
                            self.assertEqual(
                                completed.returncode,
                                0,
                                completed.stderr,
                            )
                            verified = open_result(result_dir)
                            pressure = verified.field(
                                "pore_pressure",
                                state="terminal",
                            ).read()
                            verified.close()
                            observed = tuple(
                                sorted(float(row[0]) for row in pressure.values)
                            )
                            self.assertAlmostEqual(observed[-1], 9.81, places=12)
                            if dimension == 3:
                                self.assertAlmostEqual(observed[0], 0.0, places=12)
                            if rank_reference is None:
                                rank_reference = observed
                            else:
                                for actual, expected in zip(
                                    observed,
                                    rank_reference,
                                    strict=True,
                                ):
                                    self.assertAlmostEqual(actual, expected, places=12)

            with tempfile.TemporaryDirectory(prefix="engine-affine-conflict-") as raw:
                temporary = Path(raw)
                conflicting_rule = (
                    {
                        "name": "conflicting-top-x",
                        "field": "displacement",
                        "condition": "essential",
                        "support": "top",
                        "components": ["x"],
                        "value_model": "affine_coordinate",
                        "parameters": {
                            "offset": prescribed_displacement,
                            "gradient": [0.0, 0.0],
                            "unit": "m",
                        },
                    },
                )
                fixture = InstalledFixture(
                    temporary,
                    2,
                    additional_boundary_rules=conflicting_rule,
                )
                result_dir = temporary / "run-result"
                completed = fixture.validate(result_dir, operation="run")
                self.assert_collective_failure(
                    completed,
                    result_dir,
                    exit_code=4,
                    code="SOLVER_GRAPH_CREATE",
                )

        with self.subTest(operation="run-natural-mechanics-boundaries"):
            for dimension in (2, 3):
                load_scale = 0.01
                global_traction = tuple(
                    load_scale * value for value in (0.25, -0.5, 0.125)[:dimension]
                )
                spatial_offset = tuple(
                    load_scale * value for value in (0.1, 0.2, 0.3)[:dimension]
                )
                diagonal = tuple(
                    load_scale * value for value in (0.2, 0.3, 0.4)[:dimension]
                )
                normal_pressure = load_scale * 0.4
                hydrostatic_reference_level = 1.5
                hydrostatic_reference_pressure = load_scale * 0.1
                hydrostatic_unit_weight = load_scale * 0.2
                gradient = [
                    [
                        diagonal[row] if row == column else 0.0
                        for column in range(dimension)
                    ]
                    for row in range(dimension)
                ]
                rules = (
                    {
                        "name": "global-top-traction",
                        "field": "displacement",
                        "condition": "natural",
                        "support": "top",
                        "components": ["all"],
                        "value_model": "global_traction",
                        "parameters": {
                            "value": list(global_traction),
                            "unit": "kPa",
                        },
                        "geometry_patch": "full-coordinate-field",
                    },
                    {
                        "name": "spatial-top-traction",
                        "field": "displacement",
                        "condition": "natural",
                        "support": "top",
                        "components": ["all"],
                        "value_model": "spatial_global_traction",
                        "parameters": {
                            "offset": list(spatial_offset),
                            "gradient": gradient,
                            "unit": "kPa",
                        },
                        "geometry_patch": "full-coordinate-field",
                    },
                    {
                        "name": "top-normal-pressure",
                        "field": "displacement",
                        "condition": "natural",
                        "support": "top",
                        "components": ["normal"],
                        "value_model": "normal_pressure",
                        "parameters": {
                            "value": normal_pressure,
                            "unit": "kPa",
                        },
                        "geometry_patch": "full-coordinate-field",
                    },
                    {
                        "name": "hydrostatic-top-pressure",
                        "field": "displacement",
                        "condition": "natural",
                        "support": "top",
                        "components": ["normal"],
                        "value_model": "hydrostatic",
                        "parameters": {
                            "reference_level": hydrostatic_reference_level,
                            "reference_pressure": hydrostatic_reference_pressure,
                            "unit_weight": hydrostatic_unit_weight,
                            "length_unit": "m",
                            "pressure_unit": "kPa",
                        },
                        "geometry_patch": "full-coordinate-field",
                    },
                )
                measure = math.sqrt(2.0) if dimension == 2 else math.sqrt(3.0) / 2.0
                centroid_coordinate = 0.5 if dimension == 2 else 1.0 / 3.0
                normal_coordinate = (
                    1.0 / math.sqrt(2.0) if dimension == 2 else 1.0 / math.sqrt(3.0)
                )
                hydrostatic_mean_pressure = (
                    hydrostatic_reference_pressure
                    + hydrostatic_unit_weight
                    * (hydrostatic_reference_level - centroid_coordinate)
                )
                unscaled_resultant = tuple(
                    measure
                    * (
                        global_traction[component]
                        + spatial_offset[component]
                        + diagonal[component] * centroid_coordinate
                        + normal_pressure * normal_coordinate
                        + hydrostatic_mean_pressure * normal_coordinate
                    )
                    for component in range(dimension)
                )
                cell_measure = 0.5 if dimension == 2 else 1.0 / 6.0
                unscaled_resultant = tuple(
                    value - (20.0 * cell_measure if component == 1 else 0.0)
                    for component, value in enumerate(unscaled_resultant)
                )
                rank_reference: tuple[float, ...] | None = None
                for ranks in (1, 2):
                    with self.subTest(dimension=dimension, ranks=ranks):
                        with tempfile.TemporaryDirectory(
                            prefix=f"engine-natural-mechanics-{dimension}d-"
                        ) as raw:
                            temporary = Path(raw)
                            fixture = InstalledFixture(
                                temporary,
                                ranks,
                                dimension=dimension,
                                maximum_accepted_states=1,
                                additional_boundary_rules=rules,
                            )
                            result_dir = temporary / "run-result"
                            completed = fixture.validate(
                                result_dir,
                                operation="run",
                            )
                            self.assertEqual(
                                completed.returncode,
                                0,
                                completed.stderr,
                            )
                            result = json.loads(
                                (result_dir / "result.json").read_bytes()
                            )
                            self.assert_mechanics_equilibrium_scalars(
                                result,
                                dimension,
                            )
                            scalars = {
                                scalar["id"]: scalar["value"]
                                for scalar in result["summary"]["scalars"]
                            }
                            factor = scalars["load_factor"]
                            observed = tuple(
                                scalars[f"mechanics_external_resultant_{component}"]
                                for component in ("x", "y", "z")[:dimension]
                            )
                            normalized = tuple(value / factor for value in observed)
                            for actual, expected in zip(
                                observed,
                                unscaled_resultant,
                                strict=True,
                            ):
                                self.assertAlmostEqual(
                                    actual,
                                    factor * expected,
                                    places=10,
                                )
                            if rank_reference is None:
                                rank_reference = normalized
                            else:
                                for actual, expected in zip(
                                    normalized,
                                    rank_reference,
                                    strict=True,
                                ):
                                    self.assertAlmostEqual(
                                        actual,
                                        expected,
                                        places=12,
                                    )

        with self.subTest(operation="run-prescribed-hydraulic-flux"):
            prescribed_flux = 1.0e-3
            flux_rule = (
                {
                    "name": "top-prescribed-flux",
                    "field": "pore_pressure",
                    "condition": "natural",
                    "support": "top",
                    "components": ["normal"],
                    "value_model": "prescribed_flux",
                    "parameters": {
                        "value": prescribed_flux,
                        "unit": "m/s",
                    },
                    "geometry_patch": "full-coordinate-field",
                },
            )
            for dimension in (2, 3):
                measure = 1.0 if dimension == 2 else math.sqrt(3.0) / 2.0
                rank_reference: float | None = None
                for ranks in (1, 2):
                    with self.subTest(dimension=dimension, ranks=ranks):
                        with tempfile.TemporaryDirectory(
                            prefix=f"engine-natural-hydraulic-{dimension}d-"
                        ) as raw:
                            temporary = Path(raw)
                            fixture = InstalledFixture(
                                temporary,
                                ranks,
                                hydraulic=True,
                                dimension=dimension,
                                two_cells=dimension == 2,
                                additional_boundary_rules=flux_rule,
                            )
                            result_dir = temporary / "run-result"
                            completed = fixture.validate(
                                result_dir,
                                operation="run",
                            )
                            self.assertEqual(
                                completed.returncode,
                                0,
                                completed.stderr,
                            )
                            result = json.loads(
                                (result_dir / "result.json").read_bytes()
                            )
                            scalars = {
                                scalar["id"]: scalar["value"]
                                for scalar in result["summary"]["scalars"]
                            }
                            observed = scalars["hydraulic_prescribed_boundary_flux"]
                            self.assertAlmostEqual(
                                observed,
                                prescribed_flux * measure,
                                places=12,
                            )
                            self.assertLessEqual(
                                scalars["hydraulic_normalized_balance"],
                                1.0e-8,
                            )
                            if rank_reference is None:
                                rank_reference = observed
                            else:
                                self.assertEqual(observed, rank_reference)

        with self.subTest(operation="run-all-mechanics-analyses"):
            for selector, expected_method in (
                ("ssr.direct_potential", "ssr_direct"),
                ("ssr.indirect_work_controlled", "ssr_indirect"),
            ):
                with self.subTest(selector=selector):
                    with tempfile.TemporaryDirectory(
                        prefix="engine-run-analysis-"
                    ) as raw:
                        temporary = Path(raw)
                        fixture = InstalledFixture(
                            temporary,
                            1,
                            analysis=selector,
                            young_modulus=1000.0,
                            unit_weight=80.0,
                            continuation_initial_step=1.0,
                            deflation_enabled=selector
                            == "ssr.indirect_work_controlled",
                        )
                        result_dir = temporary / "run-result"
                        completed = fixture.validate(
                            result_dir,
                            operation="run",
                        )
                        self.assertEqual(
                            completed.returncode,
                            0,
                            completed.stderr,
                        )
                        verified = open_result(result_dir)
                        terminal = verified.field(
                            "displacement",
                            state="terminal",
                        ).read()
                        self.assertGreater(len(terminal.stable_ids), 0)
                        if selector == "ssr.indirect_work_controlled":
                            deflation_rows = verified.history("deflation").read()
                            self.assertEqual(deflation_rows.shape, (1,))
                            self.assertGreaterEqual(
                                float(deflation_rows[0]["control"]),
                                1.0,
                            )
                            self.assertLessEqual(
                                float(deflation_rows[0]["residual"]),
                                1.0e-9,
                            )
                        verified.close()
                        result = json.loads((result_dir / "result.json").read_bytes())
                        self.assertEqual(
                            result["analysis_outcome"]["method"],
                            expected_method,
                        )
                        self.assertGreaterEqual(
                            result["summary"]["accepted_state_count"],
                            3,
                        )
                        if selector == "ssr.indirect_work_controlled":
                            resolved = json.loads(fixture.run_json.read_bytes())
                            self.assertTrue(resolved["deflation"]["enabled"])
                            self.assertEqual(
                                resolved["deflation"],
                                {
                                    "enabled": True,
                                    "projector": "general_left_right",
                                    "maximum_dimension": 4,
                                    "candidate_source": (
                                        "accepted_newton_and_control_directions"
                                    ),
                                    "recycle": "accepted_steps",
                                    "compression": "none",
                                    "compression_dimension": 0,
                                    "compression_history_decay": 1.0,
                                    "admission_tolerance": 1.0e-3,
                                    "pivot_tolerance": 1.0e-13,
                                    "condition_limit": 1.0e12,
                                    "invariant_tolerance": 1.0e-9,
                                    "reorthogonalization_passes": 2,
                                    "dense_factor": "partial_pivot_lu",
                                },
                            )

        with self.subTest(operation="run-coupled-analysis"):
            mode_displacements: dict[str, tuple[tuple[float, ...], ...]] = {}
            mode_body_forces: dict[str, tuple[tuple[float, ...], ...]] = {}
            for coupling_mode in (
                "pressure_gradient_and_weight",
                "weight_only",
            ):
                rank_reference: dict[
                    str, tuple[tuple[int, ...], tuple[tuple[float, ...], ...]]
                ] = {}
                for ranks in (1, 2):
                    with self.subTest(
                        coupling_mode=coupling_mode,
                        ranks=ranks,
                    ):
                        with tempfile.TemporaryDirectory(
                            prefix="engine-run-coupled-"
                        ) as raw:
                            temporary = Path(raw)
                            fixture = InstalledFixture(
                                temporary,
                                ranks,
                                coupled=True,
                                coupling_mode=coupling_mode,
                                analysis="ssr.indirect_work_controlled",
                                young_modulus=1000.0,
                                unit_weight=80.0,
                                continuation_initial_step=1.0,
                            )
                            result_dir = temporary / "run-result"
                            completed = fixture.validate(
                                result_dir,
                                operation="run",
                            )
                            self.assertEqual(
                                completed.returncode,
                                0,
                                completed.stderr,
                            )
                            verified = open_result(result_dir)
                            observed = {}
                            for quantity, accessor in (
                                ("displacement", "field"),
                                ("material_branch", "category"),
                                ("plastic_indicator", "category"),
                                ("strain", "quadrature"),
                                ("stress", "quadrature"),
                                ("yield_residual", "quadrature"),
                                ("pore_pressure", "field"),
                                ("pressure_gradient", "quadrature"),
                                ("conductivity", "quadrature"),
                                ("hydraulic_ramp", "quadrature"),
                                ("hydraulic_epsilon", "quadrature"),
                                ("pressure_flux", "quadrature"),
                                ("darcy_velocity", "quadrature"),
                                ("coupled_body_force", "quadrature"),
                                ("saturation", "category"),
                            ):
                                data = getattr(verified, accessor)(
                                    quantity,
                                    state="terminal",
                                ).read()
                                observed[quantity] = (
                                    tuple(map(int, data.stable_ids)),
                                    tuple(
                                        tuple(map(float, row))
                                        for row in data.values.reshape(
                                            data.values.shape[0],
                                            -1,
                                        )
                                    ),
                                )
                            seepage_rows = verified.history("seepage").read()
                            self.assertEqual(seepage_rows.shape, (1,))
                            verified.close()
                            if ranks == 1:
                                rank_reference = observed
                                mode_displacements[coupling_mode] = observed[
                                    "displacement"
                                ][1]
                                mode_body_forces[coupling_mode] = observed[
                                    "coupled_body_force"
                                ][1]
                            else:
                                self.assertEqual(observed, rank_reference)
                            result = json.loads(
                                (result_dir / "result.json").read_bytes()
                            )
                            self.assertEqual(
                                result["analysis_outcome"]["method"],
                                "ssr_indirect",
                            )
                            quantity_ids = {
                                quantity["id"]
                                for quantity in result["summary"]["quantities"]
                            }
                            self.assertEqual(
                                quantity_ids,
                                {
                                    "displacement",
                                    "material_branch",
                                    "plastic_indicator",
                                    "strain",
                                    "stress",
                                    "yield_residual",
                                    "pore_pressure",
                                    "pressure_gradient",
                                    "conductivity",
                                    "hydraulic_ramp",
                                    "hydraulic_epsilon",
                                    "pressure_flux",
                                    "darcy_velocity",
                                    "coupled_body_force",
                                    "saturation",
                                },
                            )
                            scalars = {
                                scalar["id"]: scalar["value"]
                                for scalar in result["summary"]["scalars"]
                            }
                            self.assert_mechanics_equilibrium_scalars(result, 2)
                            self.assertLessEqual(
                                scalars["hydraulic_normalized_balance"],
                                1.0e-8,
                            )
                            self.assertEqual(
                                scalars["coupling_unresolved_points"],
                                0.0,
                            )
                            self.assertEqual(
                                scalars["coupling_ambiguous_points"],
                                0.0,
                            )
                            self.assertGreaterEqual(
                                scalars[
                                    "hydraulic_initialization_boundary_residual_l2"
                                ],
                                0.0,
                            )
                            self.assertGreaterEqual(
                                scalars[
                                    "hydraulic_initialization_saturated_residual_l2"
                                ],
                                0.0,
                            )
                            self.assertIn(
                                scalars["hydraulic_initialization_primary_saturated"],
                                (0.0, 1.0),
                            )
                            self.assertIn(
                                scalars["hydraulic_initialization_selected_saturated"],
                                (0.0, 1.0),
                            )
                            self.assertIn(
                                scalars["hydraulic_initialization_fallback_attempted"],
                                (0.0, 1.0),
                            )
                            self.assertEqual(
                                scalars["hydraulic_initialization_primary_saturated"]
                                != scalars[
                                    "hydraulic_initialization_selected_saturated"
                                ],
                                bool(
                                    scalars[
                                        "hydraulic_initialization_fallback_attempted"
                                    ]
                                ),
                            )
            self.assertNotEqual(
                mode_displacements["pressure_gradient_and_weight"],
                mode_displacements["weight_only"],
            )
            self.assertNotEqual(
                mode_body_forces["pressure_gradient_and_weight"],
                mode_body_forces["weight_only"],
            )

        with self.subTest(operation="run-publishes-hydraulic-result"):
            for dimension, field_order, ranks in (
                (2, 1, 2),
                (2, 2, 1),
                (2, 4, 1),
                (3, 2, 1),
            ):
                with self.subTest(
                    dimension=dimension,
                    field_order=field_order,
                    ranks=ranks,
                ):
                    with tempfile.TemporaryDirectory(
                        prefix=(f"engine-run-hydraulic-{dimension}d-p{field_order}-")
                    ) as raw:
                        temporary = Path(raw)
                        fixture = InstalledFixture(
                            temporary,
                            ranks,
                            hydraulic=True,
                            dimension=dimension,
                            field_order=field_order,
                            two_cells=dimension == 2,
                        )
                        result_dir = temporary / "run-result"
                        completed = fixture.validate(result_dir, operation="run")
                        self.assertEqual(completed.returncode, 0, completed.stderr)
                        verified = open_result(result_dir)
                        pressure = verified.field("pore_pressure", state="terminal")
                        gradient = verified.quadrature(
                            "pressure_gradient", state="terminal"
                        )
                        conductivity = verified.quadrature(
                            "conductivity", state="terminal"
                        )
                        ramp = verified.quadrature("hydraulic_ramp", state="terminal")
                        epsilon = verified.quadrature(
                            "hydraulic_epsilon", state="terminal"
                        )
                        pressure_flux = verified.quadrature(
                            "pressure_flux", state="terminal"
                        )
                        darcy_velocity = verified.quadrature(
                            "darcy_velocity", state="terminal"
                        )
                        saturation = verified.category("saturation", state="terminal")
                        pressure_data = pressure.read()
                        gradient_data = gradient.read()
                        conductivity_data = conductivity.read()
                        ramp_data = ramp.read()
                        epsilon_data = epsilon.read()
                        pressure_flux_data = pressure_flux.read()
                        darcy_velocity_data = darcy_velocity.read()
                        saturation_data = saturation.read()
                        self.assertGreater(pressure.shape[0], 1)
                        self.assertEqual(
                            len(set(map(int, pressure_data.stable_ids))),
                            pressure.shape[0],
                        )
                        if dimension == 2:
                            self.assertGreater(gradient.shape[0], 1)
                            self.assertGreater(saturation.shape[0], 1)
                        else:
                            self.assertEqual(gradient.shape[0], 1)
                            self.assertEqual(saturation.shape[0], 1)
                        self.assertEqual(conductivity.unit, "m/s")
                        self.assertEqual(ramp.unit, "dimensionless")
                        self.assertEqual(epsilon.unit, "kPa")
                        self.assertEqual(pressure_flux.unit, "kPa/s")
                        self.assertEqual(darcy_velocity.unit, "m/s")
                        for value in conductivity_data.values.flat:
                            self.assertGreater(float(value), 0.0)
                        for value in ramp_data.values.flat:
                            self.assertGreaterEqual(float(value), 0.0)
                            self.assertLessEqual(float(value), 1.0)
                        for value in epsilon_data.values.flat:
                            self.assertGreater(float(value), 0.0)
                        for cell_index in range(gradient_data.values.shape[0]):
                            for quadrature_index in range(
                                gradient_data.values.shape[1]
                            ):
                                cell_conductivity = float(
                                    conductivity_data.values[
                                        cell_index,
                                        quadrature_index,
                                        0,
                                    ]
                                )
                                cell_ramp = float(
                                    ramp_data.values[
                                        cell_index,
                                        quadrature_index,
                                        0,
                                    ]
                                )
                                for component in range(dimension):
                                    gravity = (
                                        9.81 * cell_ramp if component == 1 else 0.0
                                    )
                                    self.assertAlmostEqual(
                                        float(
                                            pressure_flux_data.values[
                                                cell_index,
                                                quadrature_index,
                                                component,
                                            ]
                                        ),
                                        -cell_conductivity
                                        * (
                                            float(
                                                gradient_data.values[
                                                    cell_index,
                                                    quadrature_index,
                                                    component,
                                                ]
                                            )
                                            + gravity
                                        ),
                                        places=12,
                                    )
                        for flux, velocity in zip(
                            pressure_flux_data.values.flat,
                            darcy_velocity_data.values.flat,
                            strict=True,
                        ):
                            self.assertAlmostEqual(
                                float(flux),
                                9.81 * float(velocity),
                                places=12,
                            )
                        for row in saturation_data.values:
                            self.assertEqual(
                                len(set(map(int, row))),
                                1,
                                "saturation must be classified once from the "
                                "physical-volume mean pressure and repeated over "
                                "the cell quadrature samples",
                            )
                        verified.close()
                        result = json.loads((result_dir / "result.json").read_bytes())
                        self.assertEqual(
                            result["analysis_outcome"],
                            {
                                "method": "steady_seepage",
                                "outcome": "converged_criterion",
                            },
                        )
                        quantity_ids = {
                            quantity["id"]
                            for quantity in result["summary"]["quantities"]
                        }
                        self.assertEqual(
                            quantity_ids,
                            {
                                "pore_pressure",
                                "pressure_gradient",
                                "conductivity",
                                "hydraulic_ramp",
                                "hydraulic_epsilon",
                                "pressure_flux",
                                "darcy_velocity",
                                "saturation",
                            },
                        )
                        scalars = {
                            scalar["id"]: scalar
                            for scalar in result["summary"]["scalars"]
                        }
                        self.assertIn("global_flux_balance", scalars)
                        flux_unit = "kPa*m/s" if dimension == 2 else "kPa*m^2/s"
                        self.assertEqual(
                            scalars["global_flux_balance"]["unit"], flux_unit
                        )
                        self.assertEqual(
                            scalars["hydraulic_balance_physical_scale"]["unit"],
                            flux_unit,
                        )
                        self.assertEqual(
                            scalars["hydraulic_initialization_boundary_residual_l2"][
                                "unit"
                            ],
                            flux_unit,
                        )
                        self.assertEqual(
                            scalars["hydraulic_initialization_saturated_residual_l2"][
                                "unit"
                            ],
                            flux_unit,
                        )
                        self.assertIn(
                            scalars["hydraulic_initialization_primary_saturated"][
                                "value"
                            ],
                            (0.0, 1.0),
                        )
                        self.assertIn(
                            scalars["hydraulic_initialization_selected_saturated"][
                                "value"
                            ],
                            (0.0, 1.0),
                        )
                        self.assertIn(
                            scalars["hydraulic_initialization_fallback_attempted"][
                                "value"
                            ],
                            (0.0, 1.0),
                        )
                        self.assertEqual(
                            scalars["hydraulic_initialization_primary_saturated"][
                                "value"
                            ]
                            != scalars["hydraulic_initialization_selected_saturated"][
                                "value"
                            ],
                            bool(
                                scalars["hydraulic_initialization_fallback_attempted"][
                                    "value"
                                ]
                            ),
                        )
                        self.assertEqual(
                            scalars["hydraulic_balance_physical_scale"]["value"],
                            1.0,
                        )
                        self.assertGreaterEqual(
                            scalars["hydraulic_balance_normalization"]["value"],
                            1.0,
                        )
                        self.assertLessEqual(
                            scalars["hydraulic_normalized_balance"]["value"],
                            1.0e-8,
                        )
                        self.assertGreaterEqual(
                            scalars["hydraulic_free_residual_l2"]["value"],
                            0.0,
                        )

    def test_restart_matrix(self) -> None:
        """E2E-RESTART-001."""
        mechanics_quantities = (
            ("displacement", "field"),
            ("strain", "quadrature"),
            ("stress", "quadrature"),
            ("yield_residual", "quadrature"),
            ("material_branch", "category"),
            ("plastic_indicator", "category"),
        )
        hydraulic_quantities = (
            ("pore_pressure", "field"),
            ("pressure_gradient", "quadrature"),
            ("conductivity", "quadrature"),
            ("hydraulic_ramp", "quadrature"),
            ("hydraulic_epsilon", "quadrature"),
            ("pressure_flux", "quadrature"),
            ("darcy_velocity", "quadrature"),
            ("saturation", "category"),
        )
        cases = (
            (
                "mechanics-same-layout",
                1,
                1,
                1,
                1,
                {
                    "full_mechanics_outputs": True,
                },
                mechanics_quantities,
                False,
            ),
            (
                "direct-first-seed-same-layout",
                1,
                1,
                1,
                1,
                {
                    "analysis": "ssr.direct_potential",
                    "young_modulus": 1000.0,
                    "unit_weight": 80.0,
                    "continuation_initial_step": 1.0,
                    "checkpoint_source_accepted_states": 1,
                    "full_mechanics_outputs": True,
                },
                mechanics_quantities,
                False,
            ),
            (
                "mechanics-repartitioned-hybrid",
                1,
                1,
                2,
                2,
                {
                    "analysis": "ssr.indirect_work_controlled",
                    "young_modulus": 1000.0,
                    "unit_weight": 80.0,
                    "continuation_initial_step": 1.0,
                    "deflation_enabled": True,
                    "full_mechanics_outputs": True,
                },
                mechanics_quantities,
                True,
            ),
            (
                "hydraulic-constraint-identity-repartitioned-hybrid",
                1,
                1,
                2,
                2,
                {
                    "hydraulic": True,
                    "constraint_identity_fan": True,
                },
                hydraulic_quantities,
                False,
            ),
            (
                "coupled-repartitioned-hybrid",
                1,
                1,
                2,
                2,
                {
                    "coupled": True,
                    "analysis": "ssr.indirect_work_controlled",
                    "young_modulus": 1000.0,
                    "unit_weight": 80.0,
                    "continuation_initial_step": 1.0,
                    "deflation_enabled": True,
                },
                mechanics_quantities
                + hydraulic_quantities
                + (("coupled_body_force", "quadrature"),),
                True,
            ),
        )
        for (
            name,
            source_ranks,
            source_threads,
            restart_ranks,
            restart_threads,
            fixture_options,
            quantities,
            require_deflation,
        ) in cases:
            with self.subTest(restart=name):
                with tempfile.TemporaryDirectory(
                    prefix=f"engine-run-restart-{name}-"
                ) as raw:
                    self.assert_accepted_checkpoint_roundtrip(
                        Path(raw),
                        source_ranks=source_ranks,
                        source_threads=source_threads,
                        restart_ranks=restart_ranks,
                        restart_threads=restart_threads,
                        fixture_options=fixture_options,
                        quantities=quantities,
                        require_deflation=require_deflation,
                    )

    def test_volume_quadrature_rule(self) -> None:
        """APP-VOLUME-RULE-001 (PHY-05).

        A case selecting the paired higher volume rule compiles it into the
        problem, runs end to end, and every quadrature quantity records the
        paired rule ID.  Mechanics operators are created by a module outside
        this package; until they apply the rule the run must refuse it rather
        than mix rules.
        """
        with tempfile.TemporaryDirectory(prefix="engine-volume-rule-") as raw:
            temporary = Path(raw)
            rules: dict[str, dict[str, object]] = {}
            for rule in ("default", "paired_higher"):
                fixture = InstalledFixture(
                    temporary / rule,
                    2,
                    hydraulic=True,
                    two_cells=True,
                    field_order=2,
                    volume_quadrature=None if rule == "default" else rule,
                )
                problem_path = fixture.problem_dir / "problem.json"
                problem = json.loads(problem_path.read_bytes())
                self.assertEqual(
                    problem["operators"].get("volume_quadrature"),
                    None if rule == "default" else rule,
                )
                result_dir = temporary / f"{rule}-result"
                completed = fixture.validate(result_dir, operation="run")
                self.assertEqual(completed.returncode, 0, completed.stderr)
                envelope = json.loads((result_dir / "result.json").read_bytes())
                rules[rule] = {
                    row["id"]: (row["quadrature_id"], row["sample_count"])
                    for row in envelope["summary"]["quantities"]
                    if row["kind"] == "quadrature"
                }
                open_result(result_dir).close()
            self.assertTrue(rules["default"])
            self.assertEqual(set(rules["default"]), set(rules["paired_higher"]))
            paired_ids = {rule for rule, _ in rules["paired_higher"].values()}
            self.assertEqual(len(paired_ids), 1)
            for quantity_id, (rule_id, samples) in rules["paired_higher"].items():
                default_id, default_samples = rules["default"][quantity_id]
                self.assertNotEqual(rule_id, default_id, quantity_id)
                self.assertGreater(samples, default_samples, quantity_id)
            stress_rules = {}
            for rule in ("default", "paired_higher"):
                mechanics = InstalledFixture(
                    temporary / f"mechanics-{rule}",
                    1,
                    field_order=2,
                    full_mechanics_outputs=True,
                    volume_quadrature=None if rule == "default" else rule,
                )
                result_dir = temporary / f"mechanics-{rule}-result"
                completed = mechanics.validate(result_dir, operation="run")
                self.assertEqual(completed.returncode, 0, completed.stderr)
                envelope = json.loads((result_dir / "result.json").read_bytes())
                stress_rules[rule] = next(
                    quantity["quadrature_id"]
                    for quantity in envelope["summary"]["quantities"]
                    if quantity["id"] == "stress"
                )
            self.assertNotEqual(stress_rules["default"], stress_rules["paired_higher"])

    def test_mechanics_assembly_mode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="engine-assembly-mode-") as raw:
            directory = Path(raw)
            policies = {}
            fields = {}
            for qualification, expected in (
                (True, "qualification"),
                (False, "performance"),
            ):
                fixture = InstalledFixture(
                    directory / expected,
                    2,
                    field_order=2,
                    two_cells=True,
                    qualification_mode=qualification,
                )
                destination = directory / f"{expected}-result"
                completed = fixture.validate(destination, operation="run")
                self.assertEqual(completed.returncode, 0, completed.stderr)
                envelope = json.loads((destination / "result.json").read_bytes())
                self.assertEqual(
                    envelope["summary"]["mechanics_assembly_mode"], expected
                )
                policies[expected] = fixture.numerical_policy_id
                with open_result(destination) as result:
                    fields[expected] = (
                        result.field("displacement", state="terminal").read().values
                    )
            self.assertNotEqual(policies["qualification"], policies["performance"])
            np.testing.assert_allclose(
                fields["qualification"], fields["performance"], rtol=1e-8, atol=1e-11
            )

    def test_continuation_state_limit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="engine-continuation-limit-") as raw:
            directory = Path(raw)
            for name, analysis, limit in (
                ("limit-load", "limit_load.work_controlled", 127),
                ("direct", "ssr.direct_potential", 128),
            ):
                fixture = InstalledFixture(directory / name, 2, analysis=analysis)
                boundary = fixture.rewrite_run(
                    lambda doc: doc["continuation"].update(
                        maximum_accepted_states=limit
                    )
                )
                valid = fixture.validate(directory / f"{name}-valid", run_json=boundary)
                self.assertEqual(valid.returncode, 0, valid.stderr)
                for operation, count in (("validate", limit + 1), ("run", 2**32 + 1)):
                    with self.subTest(analysis=analysis, operation=operation):
                        too_many = fixture.rewrite_run(
                            lambda doc: doc["continuation"].update(
                                maximum_accepted_states=count
                            )
                        )
                        destination = directory / f"{name}-{operation}"
                        completed = fixture.validate(
                            destination, run_json=too_many, operation=operation
                        )
                        self.assert_collective_failure(
                            completed,
                            destination,
                            exit_code=2,
                            code="CONTINUATION_STATE_LIMIT",
                        )
                        error = json.loads((destination / "error.json").read_bytes())
                        self.assertIn(
                            "maximum_accepted_states must be an integer from 1 "
                            f"to {limit}",
                            error["message"]["text"],
                        )
                        self.assertFalse(destination.with_suffix(".run").exists())

    def test_live_solver_detail(self) -> None:
        """Bound detail rows join to actual solves and precede their completion."""
        with tempfile.TemporaryDirectory(prefix="engine-solver-detail-") as raw:
            directory = Path(raw)
            fixture = InstalledFixture(
                directory / "fixture",
                2,
                field_order=2,
                analysis="ssr.direct_potential",
                young_modulus=1000.0,
                unit_weight=80.0,
                continuation_initial_step=1.0,
                maximum_accepted_states=2,
            )

            def with_detail(run: dict) -> None:
                run["outputs"].update(krylov_history_limit=5, pmg_apply_history_limit=3)

            destination = directory / "result"
            completed = fixture.validate(
                destination, operation="run", run_json=fixture.rewrite_run(with_detail)
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            journal_rows = [
                json.loads(line)
                for line in (directory / "result.run/journal.jsonl")
                .read_text()
                .splitlines()
            ]
            starts = {
                row["linear_sequence"]: row
                for row in journal_rows
                if row["event"] == "linear_start"
            }
            ends = {
                row["linear_sequence"]: row
                for row in journal_rows
                if row["event"] == "linear_end"
            }
            self.assertEqual(starts.keys(), ends.keys())
            for sequence, start in starts.items():
                end = ends[sequence]
                self.assertLess(start["seq"], end["seq"])
                self.assertLessEqual(
                    end["cancellation_reductions"], end["iterations"] // 10
                )
            newton = [row for row in journal_rows if row["event"] == "newton"]
            self.assertTrue(newton)
            for row in newton:
                self.assertGreater(row["factor"], 0.0)
                end = next(
                    event
                    for event in journal_rows
                    if event["event"] == "solve"
                    and event.get("attempt") == row["attempt"]
                    and event["role"] == row["role"]
                )
                self.assertLess(row["seq"], end["seq"])
            with h5py.File(destination / "result.h5", "r") as handle:
                for table, maximum in (("krylov", 5), ("pmg_apply", 3)):
                    rows = handle[f"history/{table}/rows"][...]
                    self.assertEqual(len(rows), maximum)
                    for row in rows:
                        start = starts[int(row["auxiliary"])]
                        self.assertEqual(int(row["attempt_id"]), start["attempt"])
                        self.assertEqual(
                            int(row["reference"]), start["newton_iteration"]
                        )
                        self.assertEqual(int(row["role"]), start["role"])
            open_result(destination).close()

            seed_only = InstalledFixture(
                directory / "seed-only",
                1,
                field_order=2,
                analysis="ssr.indirect_work_controlled",
                young_modulus=1000.0,
                unit_weight=80.0,
                maximum_accepted_states=1,
            )
            failed = seed_only.validate(directory / "seed-result", operation="run")
            self.assertNotEqual(failed.returncode, 0)
            seed_journal = directory / "seed-result.run/journal.jsonl"
            terminal = json.loads(seed_journal.read_text().splitlines()[-1])
            self.assertEqual(terminal["event"], "terminal")
            self.assertEqual(terminal["accepted_states"], 1)
            self.assertEqual(terminal["outcome"], "failed")
            self.assertNotEqual(terminal["status"], 0)

    def test_cancel_restart(self) -> None:
        """E2E-CANCEL-001.

        A real SIGTERM delivered to rank zero after the third accepted state
        stops all ranks at the next collective safe point.  The run publishes a
        cancelled result with every accepted state, a live journal and a
        restartable checkpoint; restarting from that checkpoint reproduces the
        uninterrupted result bit for bit on the same layout.
        """
        options: dict[str, object] = {
            "analysis": "ssr.indirect_work_controlled",
            "young_modulus": 1000.0,
            "unit_weight": 80.0,
            "continuation_initial_step": 1.0,
            "deflation_enabled": True,
            "full_mechanics_outputs": True,
            "maximum_accepted_states": 6,
        }
        quantities = (
            ("displacement", "field"),
            ("stress", "quadrature"),
            ("material_branch", "category"),
        )
        with tempfile.TemporaryDirectory(prefix="engine-cancel-") as raw:
            temporary = Path(raw)
            baseline = InstalledFixture(
                temporary / "baseline", 2, output_frequency="terminal", **options
            )
            baseline_dir = temporary / "baseline-result"
            completed = baseline.validate(baseline_dir, operation="run")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            source = InstalledFixture(
                temporary / "source", 2, output_frequency="accepted_state", **options
            )
            source_dir = temporary / "source-result"
            with unittest.mock.patch.dict(
                os.environ,
                {"PETSC_OPTIONS": "-native_test_signal_after_accepted 3"},
            ):
                completed = source.validate(
                    source_dir,
                    operation="run",
                    engine_options=("--checkpoint-every", "2"),
                )
            self.assertEqual(completed.returncode, 6, completed.stderr)
            self.assertIn('"status":"cancelled"', completed.stderr)
            self.assertIn("accepted state 3", completed.stderr)
            envelope = json.loads((source_dir / "result.json").read_bytes())
            self.assertEqual(envelope["terminal_status"]["status"], "cancelled")
            self.assertEqual(envelope["terminal_status"]["stop_reason"], "cancelled")
            self.assertEqual(envelope["analysis_outcome"]["outcome"], "cancelled")
            self.assertEqual(envelope["summary"]["accepted_state_count"], 3)
            cancelled = open_result(source_dir)
            self.assertEqual(cancelled.status, "cancelled")
            displacement = cancelled.field("displacement", state=3).read()
            self.assertTrue(np.isfinite(displacement.values).all())
            cancelled.close()
            journal = [
                json.loads(line)
                for line in (temporary / "source-result.run" / "journal.jsonl")
                .read_text()
                .splitlines()
            ]
            events = [row["event"] for row in journal]
            self.assertEqual(events.count("accepted"), 3)
            self.assertEqual(events[-1], "terminal")
            self.assertIn("solve", events)
            self.assertEqual([row["seq"] for row in journal], list(range(len(journal))))
            (checkpoint,) = envelope["summary"]["checkpoints"]
            self.assertEqual(checkpoint["state_id"], 3)
            # The in-run checkpoint of accepted state 2 (ordinal 1) is retained.
            in_run = temporary / "source-result.run" / "checkpoints" / "state-000001"
            self.assertTrue((in_run / "checkpoint.json").is_file())
            self.assertEqual(
                [row["state"] for row in journal if row["event"] == "checkpoint"], [2]
            )
            baseline_result = open_result(baseline_dir)
            baseline_envelope = json.loads((baseline_dir / "result.json").read_bytes())
            for name, path in (
                ("cancelled", (source_dir / checkpoint["path"]).resolve(strict=True)),
                ("in-run", in_run.resolve(strict=True)),
            ):
                restart = InstalledFixture(
                    temporary / f"restart-{name}",
                    2,
                    output_frequency="terminal",
                    checkpoint_policy=CheckpointPolicy(
                        mode="restore_accepted", path=str(path)
                    ),
                    **options,
                )
                restart_dir = temporary / f"restart-{name}-result"
                completed = restart.validate(restart_dir, operation="run")
                self.assertEqual(completed.returncode, 0, (name, completed.stderr))
                restart_result = open_result(restart_dir)
                for quantity_id, accessor in quantities:
                    expected = getattr(baseline_result, accessor)(
                        quantity_id, state="terminal"
                    ).read()
                    actual = getattr(restart_result, accessor)(
                        quantity_id, state="terminal"
                    ).read()
                    self.assertEqual(
                        expected.values.tobytes(),
                        actual.values.tobytes(),
                        (name, quantity_id),
                    )
                restart_envelope = json.loads(
                    (restart_dir / "result.json").read_bytes()
                )
                restart_scalars = restart_envelope["summary"]["scalars"]
                baseline_scalars = baseline_envelope["summary"]["scalars"]
                self.assertEqual(
                    {r["id"]: r["value"] for r in restart_scalars},
                    {r["id"]: r["value"] for r in baseline_scalars},
                    name,
                )
                restart_result.close()
            baseline_result.close()

    def test_no_python(self) -> None:
        """APP-CONTRACT-NOPYTHON-001."""
        with tempfile.TemporaryDirectory(prefix="engine-nopython-") as raw:
            fixture = InstalledFixture(Path(raw), 1)
            forbidden = ("python", "numpy", "petsc4py", "mpi4py", "cython")
            for executable in (fixture.engine, fixture.probe):
                completed = run(["ldd", str(executable)])
                self.assertEqual(completed.returncode, 0, completed.stderr)
                lowered = completed.stdout.lower()
                for marker in forbidden:
                    self.assertNotIn(marker, lowered)
            result_dir = Path(raw) / "result"
            completed = fixture.validate(result_dir)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            events = json.loads((result_dir / "events.json").read_text())
            self.assertFalse(events["provenance"]["runtime"]["python_loaded"])

    def test_foundation(self) -> None:
        """E2E-FOUNDATION-001."""
        observed: list[tuple[str, str]] = []
        for ranks in (1, 2, 4):
            with tempfile.TemporaryDirectory(
                prefix=f"engine-foundation-{ranks}-"
            ) as raw:
                temporary = Path(raw)
                fixture = InstalledFixture(temporary, ranks)
                result_dir = temporary / "result"
                completed = fixture.validate(result_dir)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                result = json.loads((result_dir / "result.json").read_text())
                observed.append((result["run"]["problem_id"], result["run"]["run_id"]))
                self.assertEqual(result["run"]["problem_id"], fixture.problem_id)
                self.assertEqual(result["run"]["run_id"], fixture.run_id)
                self.assertRegex(result["provenance"]["machine_id"], r"^sha256:")
                self.assertRegex(result["provenance"]["launcher_id"], r"^sha256:")
                self.assertEqual(
                    sorted(result["provenance"]["placement"]["ranks"]),
                    list(range(ranks)),
                )
        self.assertEqual(len({problem_id for problem_id, _ in observed}), 1)
        self.assertEqual(len({run_id for _, run_id in observed}), 3)


if __name__ == "__main__":
    unittest.main()
