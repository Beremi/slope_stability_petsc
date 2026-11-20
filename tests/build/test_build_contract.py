# SPDX-License-Identifier: BSD-2-Clause
"""Executable build, install, lock and provenance contracts."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
BUILD = Path(os.environ.get("SLOPE_BUILD_DIR", ROOT / "build"))
sys.path.insert(0, str(ROOT / "generators/src"))

from slope_codegen import protocol as generator
from slope_codegen import solver_options


def canonical_sha(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


class BuildContract(unittest.TestCase):
    maxDiff = None

    def test_build_manifest(self) -> None:
        """BLD-UNIT-MANIFEST-001."""
        manifest = json.loads(
            (BUILD / "build-manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["kind"], "slope_stability.build")
        self.assertEqual(
            manifest["schema_version"], {"major": 1, "minor": 0, "patch": 0}
        )
        self.assertEqual(manifest["source_id"], canonical_sha(manifest["source"]))
        self.assertEqual(
            manifest["dependency_id"], canonical_sha(manifest["dependency"])
        )
        self.assertEqual(manifest["build_id"], canonical_sha(manifest["build"]))
        self.assertEqual(manifest["build"]["source_id"], manifest["source_id"])
        self.assertEqual(manifest["build"]["dependency_id"], manifest["dependency_id"])
        self.assertEqual(manifest["dependency"]["features"], manifest["features"])
        self.assertEqual(
            manifest["features"]["application_openmp"],
            manifest["features"]["openmp"],
        )
        # A dependency may inject compiler flags despite the selected profile.
        # Check the effective compiler mode, not only the configured boolean.
        commands = json.loads((BUILD / "compile_commands.json").read_text())
        coupling_command = next(
            entry
            for entry in commands
            if entry["file"].endswith("/src/coupling/coupling.c")
        )
        arguments = shlex.split(coupling_command["command"])
        probe_arguments = []
        index = 0
        while index < len(arguments):
            argument = arguments[index]
            if argument == "-o":
                index += 2
                continue
            if argument not in {"-c", coupling_command["file"]}:
                probe_arguments.append(argument)
            index += 1
        probe = subprocess.run(
            [*probe_arguments, "-E", "-dM", "-x", "c", "-"],
            input="",
            text=True,
            capture_output=True,
            check=False,
            cwd=coupling_command["directory"],
        )
        self.assertEqual(probe.returncode, 0, probe.stderr)
        self.assertEqual(
            any(
                line.startswith("#define _OPENMP ")
                for line in probe.stdout.splitlines()
            ),
            manifest["features"]["application_openmp"],
            "Effective project OpenMP mode disagrees with the build profile",
        )
        for feature in (
            "petsc_openmp",
            "petsc_openmp_kernels",
            "petsc_hwloc",
            "petsc_kokkos",
            "petsc_mumps",
            "petsc_openblas_thread_control",
            "superlu_dist_openmp",
        ):
            self.assertIsInstance(manifest["features"][feature], bool)
        self.assertTrue(
            not manifest["features"]["petsc_openmp_kernels"]
            or manifest["features"]["petsc_openmp"],
            "PETSc OpenMP kernels require PETSc OpenMP support",
        )
        self.assertEqual(
            set(manifest["dependency"]["linear_algebra"]),
            {"blas_libraries", "lapack_libraries"},
        )
        self.assertTrue(manifest["dependency"]["linear_algebra"]["blas_libraries"])
        self.assertTrue(manifest["dependency"]["linear_algebra"]["lapack_libraries"])
        self.assertTrue(manifest["dependency"]["petsc_configuration"]["cflags_extra"])
        for identity in ("source_id", "dependency_id", "build_id"):
            self.assertRegex(manifest[identity], r"^sha256:[0-9a-f]{64}$")
        self.assertRegex(manifest["source"]["patch_digest"], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(
            manifest["dependency"]["native_lock_sha256"],
            file_sha(ROOT / "environments/spack.lock"),
        )
        self.assertEqual(
            manifest["dependency"]["python_lock_sha256"],
            file_sha(ROOT / "uv.lock"),
        )

        source_mutation = copy.deepcopy(manifest["source"])
        source_mutation["revision"] = "0" * 40
        self.assertNotEqual(manifest["source_id"], canonical_sha(source_mutation))

        for path, value in (
            (("petsc",), "3.24.3"),
            (("features", "hdf5_parallel"), False),
            (
                ("petsc_int_bits",),
                32 if manifest["dependency"]["petsc_int_bits"] == 64 else 64,
            ),
        ):
            mutation = copy.deepcopy(manifest["dependency"])
            target = mutation
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            self.assertNotEqual(
                manifest["dependency_id"],
                canonical_sha(mutation),
                f"dependency mutation {path} did not change identity",
            )

        for path, value in (
            (("compiler", "flags"), "-O0"),
            (("compiler", "version"), "mutated"),
            (("openmp",), not manifest["build"]["openmp"]),
            (("profile",), "mutated"),
        ):
            mutation = copy.deepcopy(manifest["build"])
            target = mutation
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            self.assertNotEqual(
                manifest["build_id"],
                canonical_sha(mutation),
                f"build mutation {path} did not change identity",
            )

    def test_index_contract(self) -> None:
        """BLD-UNIT-INDEX-001."""
        engine = Path(os.environ["SLOPE_ENGINE"])
        probe = Path(os.environ["SLOPE_PROBE"])
        version_result = run([str(engine), "--version"])
        self.assertEqual(version_result.returncode, 0, version_result.stderr)
        version = json.loads(version_result.stdout)
        bits = version["petsc_int_bits"]
        self.assertIn(bits, (32, 64))
        maximum = 2 ** (bits - 1) - 1
        accepted = run([str(probe), "--check-index", str(maximum)])
        rejected = run([str(probe), "--check-index", str(maximum + 1)])
        malformed = run([str(probe), "--check-index", "not-an-integer"])
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertTrue(json.loads(accepted.stdout)["accepted"])
        self.assertEqual(rejected.returncode, 6, rejected.stderr)
        self.assertFalse(json.loads(rejected.stdout)["accepted"])
        self.assertEqual(malformed.returncode, 2, malformed.stderr)
        self.assertFalse(json.loads(malformed.stdout)["accepted"])

    def test_solver_options_reference(self) -> None:
        """The public option reference and C registry must match exactly."""
        options = solver_options.read_registry(ROOT)
        self.assertEqual(
            (ROOT / solver_options.REFERENCE).read_text(encoding="utf-8"),
            solver_options.render_reference(options),
        )

    def test_generation_reproducible(self) -> None:
        """BLD-GEN-REPRO-001."""
        self.assertEqual(generator.check_outputs(ROOT), [])
        self.assertEqual(generator.clean_tree_reproducibility(ROOT), [])
        generated = json.loads(
            (ROOT / "generated/manifest.json").read_text(encoding="utf-8")
        )
        self.assertGreaterEqual(len(generated["outputs"]), 16)
        forbidden = (
            b"/home/",
            b"__DATE__",
            b"__TIME__",
            b"Python.h",
            b"mpi4py",
            b"petsc4py",
        )
        for relative, expected in generated["outputs"].items():
            output = ROOT / relative
            self.assertEqual(file_sha(output), expected)
            if output.suffix in {".c", ".h"}:
                payload = output.read_bytes()
                for marker in forbidden:
                    self.assertNotIn(marker, payload, relative)

    def test_configure_fail_closed(self) -> None:
        """BLD-CONFIG-FAILCLOSED-001."""
        cases = (
            (
                "SLOPE_TEST_FORCE_MISSING_HDF5",
                "injected: parallel HDF5 is unavailable",
            ),
            (
                "SLOPE_TEST_FORCE_MPI_HDF5_MISMATCH",
                "injected: MPI/HDF5 implementation mismatch",
            ),
            (
                "SLOPE_TEST_FORCE_PETSC_INCOMPATIBLE",
                "injected: PETSc family/scalar/index mismatch",
            ),
        )
        with tempfile.TemporaryDirectory(prefix="slope-config-negative-") as raw:
            temporary = Path(raw)
            for index, (variable, expected) in enumerate(cases):
                with self.subTest(case=variable):
                    build = temporary / f"case-{index}"
                    result = run(
                        [
                            "cmake",
                            "-S",
                            str(ROOT),
                            "-B",
                            str(build),
                            "-G",
                            "Ninja",
                            "-DBUILD_TESTING=OFF",
                            "-DSLOPE_ENABLE_FAILURE_INJECTION=ON",
                            f"-D{variable}=ON",
                        ]
                    )
                    combined = result.stdout + result.stderr
                    self.assertNotEqual(result.returncode, 0, combined)
                    self.assertIn(expected, combined)

            current = json.loads(
                (BUILD / "build-manifest.json").read_text(encoding="utf-8")
            )["dependency"]["petsc_int_bits"]
            incompatible = 64 if current == 32 else 32
            result = run(
                [
                    "cmake",
                    "-S",
                    str(ROOT),
                    "-B",
                    str(temporary / "wrong-index"),
                    "-G",
                    "Ninja",
                    "-DBUILD_TESTING=OFF",
                    f"-DSLOPE_PETSC_INDEX_BITS={incompatible}",
                ]
            )
            combined = result.stdout + result.stderr
            self.assertNotEqual(result.returncode, 0, combined)
            self.assertIn("PETSc feature probe failed", combined)

    def test_install_relocation(self) -> None:
        """BLD-INSTALL-RELOCATE-001."""
        with tempfile.TemporaryDirectory(prefix="slope-install-") as raw:
            prefix = Path(raw) / "empty-prefix"
            install = run(["cmake", "--install", str(BUILD), "--prefix", str(prefix)])
            self.assertEqual(install.returncode, 0, install.stderr)
            engine = prefix / "bin/slope-engine"
            probe = prefix / "bin/slope-probe"
            data = prefix / "share/slope-stability"
            self.assertTrue(engine.is_file())
            self.assertTrue(probe.is_file())

            environment = os.environ.copy()
            runtime = os.environ.get("SLOPE_RUNTIME_LIBRARY_PATH", "")
            prior = environment.get("LD_LIBRARY_PATH", "")
            environment["LD_LIBRARY_PATH"] = ":".join(
                item for item in (runtime, prior) if item
            )
            version = run([str(engine), "--version"], cwd=prefix, env=environment)
            self.assertEqual(version.returncode, 0, version.stderr)

            artifact = prefix / "installed-probe.h5"
            matrix = run(
                [
                    os.environ["SLOPE_MPIEXEC"],
                    os.environ.get("SLOPE_MPI_NUMPROC_FLAG", "-n"),
                    "2",
                    str(probe),
                    "--hdf5",
                    str(artifact),
                ],
                cwd=prefix,
                env=environment,
            )
            self.assertEqual(matrix.returncode, 0, matrix.stderr)
            probe_result = json.loads(matrix.stdout)
            self.assertTrue(probe_result["hdf5_collective"])
            self.assertEqual(len(probe_result["openmp_cpu_affinity"]), 2)
            self.assertTrue(
                all(
                    len(row) == probe_result["openmp_team"]
                    for row in probe_result["openmp_cpu_affinity"]
                )
            )

            for executable in (engine, probe):
                dynamic = run(["readelf", "-d", str(executable)])
                self.assertEqual(dynamic.returncode, 0, dynamic.stderr)
                self.assertNotIn(str(ROOT), dynamic.stdout)
                self.assertNotIn(str(BUILD), dynamic.stdout)
                self.assertNotRegex(dynamic.stdout, r"\((?:RPATH|RUNPATH)\)")
                dependencies = run(["ldd", str(executable)], env=environment)
                self.assertEqual(dependencies.returncode, 0, dependencies.stderr)
                self.assertNotRegex(
                    dependencies.stdout.lower(),
                    r"python|numpy|petsc4py|mpi4py|cython",
                )

            manifest = json.loads(
                (data / "install-manifest.json").read_text(encoding="utf-8")
            )
            engine_digest = file_sha(engine)
            probe_digest = file_sha(probe)
            self.assertEqual(manifest["engine_sha256"], engine_digest)
            self.assertEqual(manifest["probe_sha256"], probe_digest)
            schema_files = sorted(
                path for path in (data / "schemas").rglob("*") if path.is_file()
            )
            projection = "".join(
                f"{path.relative_to(data).as_posix()}:{file_sha(path)}\n"
                for path in schema_files
            )
            schema_digest = hashlib.sha256(projection.encode("utf-8")).hexdigest()
            self.assertEqual(manifest["schema_set_sha256"], schema_digest)
            executable_projection = (
                f"engine:{engine_digest}\n"
                f"probe:{probe_digest}\n"
                f"schemas:{schema_digest}\n"
            )
            self.assertEqual(
                manifest["executable_id"],
                "sha256:"
                + hashlib.sha256(executable_projection.encode("utf-8")).hexdigest(),
            )

    def test_lock_consistency(self) -> None:
        """BLD-LOCK-CONSISTENCY-001."""
        spack = json.loads(
            (ROOT / "environments/spack.lock").read_text(encoding="utf-8")
        )
        self.assertEqual(spack["_meta"]["file-type"], "spack-lockfile")
        self.assertEqual(spack["spack"]["version"], "1.2.2")
        specs = list(spack["concrete_specs"].values())
        petsc = [
            spec
            for spec in specs
            if spec["name"] == "petsc" and spec["version"] == "3.24.2"
        ]
        self.assertEqual({spec["parameters"]["int64"] for spec in petsc}, {False, True})
        for spec in petsc:
            for feature in ("double", "hdf5", "hypre", "metis", "mpi", "shared"):
                self.assertTrue(spec["parameters"][feature])
            self.assertFalse(spec["parameters"]["complex"])
            self.assertEqual(spec["arch"]["target"], "x86_64")
        expected_native = {
            "hdf5": "2.2.0",
            "hypre": "3.0.0",
            "llvm": "23.1.5",
            "metis": "5.1.0",
            "openblas": "0.3.34",
            "openmpi": "5.0.11",
            "parmetis": "4.0.3",
            "petsc": "3.24.2",
        }
        observed = {
            spec["name"]: spec["version"]
            for spec in specs
            if spec["name"] in expected_native
        }
        self.assertEqual(observed, expected_native)

        uv = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
        self.assertEqual(uv["requires-python"], ">=3.12")
        contract = json.loads(
            (ROOT / "environments/dependency-contract.json").read_text(encoding="utf-8")
        )
        uv_versions = {package["name"]: package["version"] for package in uv["package"]}
        workspace_members = set(contract["python"]["workspace_members"])
        locked_python = contract["python"]["locked_packages"]
        self.assertEqual(set(uv_versions), workspace_members | set(locked_python))
        self.assertEqual(
            {name: uv_versions[name] for name in locked_python},
            locked_python,
        )
        self.assertEqual(contract["native"]["petsc"]["index_variants"], [32, 64])
        self.assertEqual(
            contract["native"]["petsc"]["performance_profile"],
            {
                "blas_threads": 1,
                "c_optimization_flags": ["-O2", "-g"],
                "cuda": False,
                "openmp": True,
                "openmp_kernels": True,
                "superlu_dist": {
                    "openmp": False,
                    "source_revision": "bce5318a9141cff5840071665b6ffccfc5bbba82",
                    "version": "9.2.0",
                },
            },
        )
        self.assertEqual(
            contract["resolver"]["spack"]["commit"], spack["spack"]["commit"]
        )
        self.assertEqual(
            contract["resolver"]["spack"]["version"], spack["spack"]["version"]
        )

        sbom = json.loads(
            (ROOT / "environments/sbom.cdx.json").read_text(encoding="utf-8")
        )
        self.assertEqual(sbom["bomFormat"], "CycloneDX")
        sbom_versions = {
            component["name"]: component["version"] for component in sbom["components"]
        }
        for name, version in expected_native.items():
            self.assertEqual(sbom_versions[name], version)
        for name, version in locked_python.items():
            self.assertEqual(sbom_versions[name], version)

        manifest = json.loads(
            (BUILD / "build-manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["dependency"]["petsc"], "3.24.2")
        self.assertEqual(manifest["dependency"]["hdf5"], "2.2.0")
        self.assertEqual(
            manifest["dependency"]["native_lock_sha256"],
            file_sha(ROOT / "environments/spack.lock"),
        )
        self.assertEqual(
            manifest["dependency"]["python_lock_sha256"],
            file_sha(ROOT / "uv.lock"),
        )

    def test_reuse_license(self) -> None:
        """BLD-REUSE-LICENSE-001."""
        ledger_path = ROOT / "licenses/reuse-ledger.csv"
        with ledger_path.open(encoding="utf-8", newline="") as stream:
            records = list(csv.DictReader(stream))
        self.assertGreaterEqual(len(records), 2)
        required = {
            "record_id",
            "status",
            "component",
            "classification",
            "origin_repository",
            "origin_revision",
            "origin_path_or_symbol",
            "origin_sha256",
            "license",
            "adaptation",
            "known_defects_excluded",
            "owning_tests",
            "notice_path",
            "tracked_paths",
        }
        self.assertEqual(set(records[0]), required)
        generated_coverage: list[str] = []
        for record in records:
            with self.subTest(record=record["record_id"]):
                self.assertEqual(record["status"], "approved")
                for field in required - {"status"}:
                    self.assertTrue(record[field], field)
                self.assertTrue((ROOT / record["notice_path"]).is_file())
                if record["classification"] == "generated":
                    generated_coverage.extend(record["tracked_paths"].split(";"))

        ancestor = next(
            record for record in records if record["record_id"] == "REUSE-0001"
        )
        ancestor_notice = ROOT / ancestor["notice_path"]
        self.assertEqual(file_sha(ancestor_notice), ancestor["origin_sha256"])
        self.assertIn(
            "licenses/sysala-slope-stability-BSD-2-Clause.txt",
            (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8"),
        )

        generated = json.loads(
            (ROOT / "generated/manifest.json").read_text(encoding="utf-8")
        )
        for relative, expected in {
            **generated["sources"],
            **generated["outputs"],
        }.items():
            self.assertEqual(file_sha(ROOT / relative), expected)
        for relative in generated["outputs"]:
            self.assertTrue(
                any(
                    relative == item
                    or (item.endswith("/") and relative.startswith(item))
                    for item in generated_coverage
                ),
                f"generated output lacks reuse ownership: {relative}",
            )


if __name__ == "__main__":
    unittest.main()
