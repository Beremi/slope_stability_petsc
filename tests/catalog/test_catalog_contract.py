# SPDX-License-Identifier: BSD-2-Clause
"""Catalog, in-repository mesh and catalog-compilation contracts."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from slope_stability.catalog import (
    CatalogError,
    compile_catalog,
    load_catalog,
    read_sha256sums,
    verify_meshes,
)
from slope_stability.cli import main as cli_main
from slope_stability.compile.adaptations import SUPPORT_NAME

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "cases" / "catalog"
MESHES = ROOT / "meshes"

# Compiled problem identities of every 2D catalog variant. They were admitted
# for the frozen external mesh sources and are reproduced from meshes/.
EXPECTED_2D_PROBLEM_IDS = {
    ("CASE-001", "canonical"): (
        "sha256:74d6e9bbfc8c6e8f4b8064c5a38d9479f5e06c57ef3db9a761477c5f27ebd631"
    ),
    ("CASE-002", "canonical"): (
        "sha256:8862fdd876304023d4f886a6e4811cfc310f96b2df71927f9779a8d5bf3db485"
    ),
    ("CASE-003", "canonical"): (
        "sha256:1968e49294f67403ea59c912440eec1f7beefc65b634d2bfe9caecf39ec7a467"
    ),
    ("CASE-004", "canonical"): (
        "sha256:ada6f2dc54c2bcb1fa3c97e32e192d970a8f45d90683affd426d640e12f5eebc"
    ),
    ("CASE-005", "p2"): (
        "sha256:7b0783ba2553d9249d50a7a7d4b5e5880bad27c5b8979c659b311e4f3b876d0e"
    ),
    ("CASE-005", "p4"): (
        "sha256:4c3433e00bd166b14b4fcef1abefaa6035c5a49706a308dccf37d580ef3754a1"
    ),
    ("CASE-006", "canonical"): (
        "sha256:cb4a0ca95b2bcca13f7e70c9a2dba2e80eb440378da12a224d76b08f5aa48ec6"
    ),
    ("CASE-007", "canonical"): (
        "sha256:2e7cf274cac31f9693d1a4b6048976e40db3d78fa96f21eea6bd2c44121b03f2"
    ),
    ("CASE-022", "canonical"): (
        "sha256:4ee3cb640e004f00f2fa41d562bdc8b7ceeb231e26e3ee661abf2c0f7fffe3b6"
    ),
}


def run_cli(*arguments: str) -> tuple[int, list[dict]]:
    """Run the console entry point and decode its JSON lines."""
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
        status = cli_main(list(arguments))
    return status, [json.loads(line) for line in stdout.getvalue().splitlines()]


class CatalogMeshContract(unittest.TestCase):
    """CAT-MESH-SHA-001: meshes match SHA256SUMS, the index and their assets."""

    def test_meshes_match_all_recorded_digests(self) -> None:
        catalog = load_catalog(CATALOG)
        self.assertEqual(catalog.mesh_root, MESHES.resolve())
        listed = read_sha256sums(MESHES)
        self.assertEqual(len(listed), 13)
        self.assertEqual(
            set(listed),
            {path.relative_to(MESHES).as_posix() for path in MESHES.rglob("*.msh")},
        )
        checks = verify_meshes(catalog)
        self.assertEqual(len(checks), 13)
        for check in checks:
            with self.subTest(mesh=check.mesh):
                self.assertTrue(check.passed, check)
                self.assertIn("SHA256SUMS", check.expected)
                self.assertTrue(
                    any(key.startswith("asset:") for key in check.expected),
                    "every shipped mesh is used by an asset",
                )
        self.assertEqual(len(catalog.variants), 24)
        self.assertEqual({record.mesh for record in catalog.variants}, set(listed))
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("meshes/**/*.msh -text -diff", attributes.splitlines())

    def test_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            copy = Path(raw) / "meshes"
            shutil.copytree(MESHES, copy)
            target = copy / "2d_homo_slope" / "h1.0.msh"
            payload = bytearray(target.read_bytes())
            payload[-2] ^= 0x01
            target.write_bytes(bytes(payload))
            (copy / "extra.msh").write_bytes(b"$MeshFormat\n")
            catalog = load_catalog(CATALOG, mesh_root=copy)
            failed = {
                check.mesh for check in verify_meshes(catalog) if not check.passed
            }
            self.assertEqual(failed, {"2d_homo_slope/h1.0.msh", "extra.msh"})
            status, lines = run_cli(
                "--meshes", str(copy), "verify-meshes", "--no-assets"
            )
            self.assertEqual(status, 1)
            self.assertEqual(lines[-1]["failed"], 2)
            with self.assertRaisesRegex(CatalogError, "digest mismatch"):
                compile_catalog(catalog, Path(raw) / "out", case_ids=["CASE-003"])
            self.assertFalse((Path(raw) / "out").exists())

    def test_catalog_identity_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            copy = Path(raw) / "catalog"
            shutil.copytree(CATALOG, copy)
            index = json.loads((copy / "index.json").read_text(encoding="utf-8"))
            index["cases"][0]["field_order"] = 4
            (copy / "index.json").write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(CatalogError, "catalog_id"):
                load_catalog(copy, mesh_root=MESHES)
        catalog = load_catalog(CATALOG)
        with self.assertRaisesRegex(CatalogError, "unknown case"):
            catalog.select(["CASE-999"])
        self.assertEqual(
            [record.variant_id for record in catalog.select(["CASE-005"])],
            ["p2", "p4"],
        )

    def test_cli_verify_and_inspect(self) -> None:
        status, lines = run_cli("verify-meshes")
        self.assertEqual(status, 0)
        self.assertEqual(lines[-1]["status"], "pass")
        self.assertEqual(lines[-1]["mesh_count"], 13)
        status, lines = run_cli("inspect", str(CATALOG))
        self.assertEqual(status, 0)
        self.assertEqual(lines[0]["kind"], "catalog")
        self.assertEqual(len(lines[0]["cases"]), 24)
        status, lines = run_cli("inspect", str(MESHES / "2d_homo_slope/h1.0.msh"))
        self.assertEqual(status, 0)
        self.assertTrue(lines[0]["listed"])
        status, _ = run_cli("inspect", str(ROOT / "does-not-exist.json"))
        self.assertEqual(status, 2)


class CatalogCompileContract(unittest.TestCase):
    """CAT-COMPILE-001: compile every 2D catalog variant from meshes/."""

    def test_two_dimensional_variants_reproduce_problem_identities(self) -> None:
        catalog = load_catalog(CATALOG)
        two_dimensional = sorted({key[0] for key in EXPECTED_2D_PROBLEM_IDS})
        with tempfile.TemporaryDirectory() as raw:
            destination = Path(raw) / "compiled"
            status, lines = run_cli(
                "compile",
                "--destination",
                str(destination),
                *(item for case in two_dimensional for item in ("--case-id", case)),
            )
            self.assertEqual(status, 0, lines)
            report = json.loads(
                (destination / "compile-catalog.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["catalog_id"], catalog.catalog_id)
            self.assertEqual(report["status"], "pass")
            observed = {
                (row["case_id"], row["variant_id"]): row["problem_id"]
                for row in report["cases"]
            }
            self.assertEqual(observed, EXPECTED_2D_PROBLEM_IDS)
            for row in report["cases"]:
                package = destination / row["package"]
                self.assertTrue((package / "problem.json").is_file())
                self.assertTrue((package / "problem.h5").is_file())
            # The destination is write-once.
            status, _ = run_cli(
                "compile", "--destination", str(destination), "--case-id", "CASE-003"
            )
            self.assertEqual(status, 2)

    def test_hydraulic_adaptation_is_applied_to_its_declared_asset(self) -> None:
        catalog = load_catalog(CATALOG)
        adapted = [
            record
            for record in catalog.variants
            if catalog.load_asset(record).extensions.get("hydraulic_support_adaptation")
        ]
        # One shared asset (the homogeneous 3D slope) declares the adaptation;
        # only CASE-015 references the derived hydraulic support.
        self.assertEqual(
            {record.mesh for record in adapted},
            {"3d_homo_slope/adaptive_family_a_l1.msh"},
        )
        self.assertIn("CASE-015/canonical", {record.key for record in adapted})
        asset = catalog.load_asset(adapted[0])
        self.assertIn(SUPPORT_NAME, {support.name for support in asset.supports})
        users = {
            record.key
            for record in catalog.variants
            if SUPPORT_NAME
            in (catalog.directory / record.case_document).read_text(encoding="utf-8")
        }
        self.assertEqual(users, {"CASE-015/canonical"})


if __name__ == "__main__":
    unittest.main()
