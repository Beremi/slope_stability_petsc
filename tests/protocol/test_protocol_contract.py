# SPDX-License-Identifier: BSD-2-Clause
"""Executable protocol contract tests."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "python/src"
GENERATOR_SRC = ROOT / "generators/src"
sys.path.insert(0, str(PYTHON_SRC))
sys.path.insert(0, str(GENERATOR_SRC))

import slope_protocol
from slope_protocol import CanonicalizationError
from slope_codegen import protocol as generator


class ProtocolContract(unittest.TestCase):
    def test_canonical_valid_corpus(self) -> None:
        """PRO-UNIT-CANON-001."""
        corpus = json.loads(
            (ROOT / "schemas/source/conformance.json").read_text(encoding="utf-8")
        )
        for case in corpus["canonical_cases"]:
            with self.subTest(case=case["id"]):
                self.assertEqual(
                    slope_protocol.canonicalize(case["input"]).decode("utf-8"),
                    case["canonical"],
                )

    def test_canonical_invalid_corpus(self) -> None:
        """PRO-UNIT-CANON-002."""
        corpus = json.loads(
            (ROOT / "schemas/source/conformance.json").read_text(encoding="utf-8")
        )
        for case in corpus["invalid_cases"]:
            with self.subTest(case=case["id"]):
                payload = (ROOT / "schemas/fixtures" / case["file"]).read_bytes()
                with self.assertRaises(CanonicalizationError) as caught:
                    slope_protocol.canonicalize_json(payload)
                self.assertEqual(caught.exception.code, case["error"])
        with self.assertRaises(CanonicalizationError) as caught:
            slope_protocol.canonicalize("\ud800")
        self.assertEqual(caught.exception.code, "invalid_unicode")

    def test_paths(self) -> None:
        """PRO-UNIT-PATH-001."""
        self.assertEqual(
            slope_protocol.normalize_package_path("payload/mesh.h5"),
            "payload/mesh.h5",
        )
        for invalid in (
            "",
            "/absolute",
            "../escape",
            "a/../b",
            "a//b",
            "a\\b",
            "https://example.invalid/file",
            "C:/device",
            "a/\x00/b",
        ):
            with self.subTest(path=invalid):
                with self.assertRaises(CanonicalizationError):
                    slope_protocol.normalize_package_path(invalid)

    def test_identity_mutation(self) -> None:
        """PRO-UNIT-ID-001."""
        manifest = {
            "kind": "slope_stability.compiled_problem",
            "schema_version": {"major": 1, "minor": 0, "patch": 0},
            "mesh": {"dimension": 2},
        }
        artifacts = [
            {
                "path": "problem.h5",
                "media_type": "application/x-hdf5",
                "layout_id": "compiled-problem-v1",
                "byte_count": 4,
                "sha256": hashlib.sha256(b"mesh").hexdigest(),
            }
        ]
        baseline = slope_protocol.package_id(manifest, artifacts)
        changed_manifest = copy.deepcopy(manifest)
        changed_manifest["mesh"]["dimension"] = 3
        self.assertNotEqual(
            baseline, slope_protocol.package_id(changed_manifest, artifacts)
        )
        changed_artifacts = copy.deepcopy(artifacts)
        changed_artifacts[0]["sha256"] = hashlib.sha256(b"mesH").hexdigest()
        self.assertNotEqual(
            baseline, slope_protocol.package_id(manifest, changed_artifacts)
        )

    def test_cross_language_conformance(self) -> None:
        """PRO-CROSS-CONFORMANCE-001."""
        executable = os.environ.get("SLOPE_SCHEMA_CHECK")
        self.assertIsNotNone(
            executable, "SLOPE_SCHEMA_CHECK must identify the native fixture tool"
        )
        corpus = json.loads(
            (ROOT / "schemas/source/conformance.json").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory(prefix="protocol-cross-") as directory:
            temporary = Path(directory)
            for case in corpus["canonical_cases"]:
                with self.subTest(case=case["id"]):
                    fixture = temporary / f"{case['id']}.json"
                    fixture.write_text(
                        json.dumps(
                            case["input"],
                            ensure_ascii=False,
                            allow_nan=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    completed = subprocess.run(
                        [str(executable), "canonicalize", str(fixture)],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    self.assertEqual(completed.stdout, case["canonical"] + "\n")
            for case in corpus["invalid_cases"]:
                with self.subTest(case=case["id"]):
                    completed = subprocess.run(
                        [
                            str(executable),
                            "canonicalize",
                            str(ROOT / "schemas/fixtures" / case["file"]),
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(completed.returncode, 2, completed.stderr)
                    observation = json.loads(completed.stdout)
                    self.assertEqual(observation["error"], case["error"])

    def test_hdf5_layout_authority(self) -> None:
        """PRO-H5-LAYOUT-001 language-neutral half."""
        source = json.loads(
            (ROOT / "schemas/source/hdf5-layouts.json").read_text(encoding="utf-8")
        )
        generated = json.loads(
            (ROOT / "schemas/hdf5-layout/generated-layouts.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(source, generated)
        self.assertEqual(
            [item["id"] for item in source["layouts"]],
            ["compiled-problem-v1", "result-v1", "checkpoint-v1"],
        )
        self.assertIn("zero_owner_select_none", source["collective_call_discipline"])

    def test_compatibility(self) -> None:
        """PRO-COMPAT-MATRIX-001."""
        self.assertTrue(slope_protocol.compatibility((1, 0, 0), (1, 0, 99)))
        self.assertFalse(slope_protocol.compatibility((1, 0, 0), (2, 0, 0)))
        self.assertFalse(slope_protocol.compatibility((1, 0, 0), (1, 1, 0)))
        self.assertTrue(
            slope_protocol.compatibility((1, 1, 0), (1, 1, 2), maximum_minor=1)
        )

    def test_local_and_executable_controls_rejected(self) -> None:
        """PRO-NEG-LOCALID-001."""
        for key in (
            "petsc_point_id",
            "petsc_dof_id",
            "rank_local_id",
            "python_callable",
            "module_path",
            "shared_library",
            "source_snippet",
        ):
            with self.subTest(key=key):
                with self.assertRaises(CanonicalizationError) as caught:
                    slope_protocol.reject_forbidden_public_keys({"outer": [{key: 1}]})
                self.assertEqual(caught.exception.code, "forbidden_public_key")

    def test_generated_outputs_reproducible(self) -> None:
        """BLD-GEN-REPRO-001 protocol generation half."""
        self.assertEqual(generator.check_outputs(ROOT), [])
        self.assertEqual(generator.clean_tree_reproducibility(ROOT), [])
        forbidden = (b"/home/", b"__DATE__", b"__TIME__", b"python")
        for relative in generator.render(ROOT):
            if relative.suffix in {".c", ".h"}:
                payload = (ROOT / relative).read_bytes()
                for marker in forbidden:
                    self.assertNotIn(marker, payload)


if __name__ == "__main__":
    unittest.main()
