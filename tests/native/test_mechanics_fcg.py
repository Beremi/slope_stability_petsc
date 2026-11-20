# SPDX-License-Identifier: BSD-2-Clause
"""Installed mechanics must reject an unqualified flexible CG policy."""

import json
from pathlib import Path
import unittest

from tests.app import test_engine_contract as app


class MechanicsFcgTest(unittest.TestCase):
    def test_unqualified_fcg(self) -> None:
        with app.saturation_artifact_directory("mechanics-fcg-") as temporary:
            root = Path(temporary)
            fixture = app.InstalledFixture(root, 2, two_cells=True, field_order=2)
            settings = fixture.rewrite_run(
                lambda run: run["linear"].update(ksp_type="fcg")
            )
            completed = fixture.validate(
                root / "result", operation="run", run_json=settings
            )
            self.assertEqual(
                completed.returncode, 70, completed.stdout + completed.stderr
            )
            error = json.loads((root / "result/error.json").read_bytes())
            self.assertEqual(error["code"]["name"], "MECHANICS_FCG_UNQUALIFIED")
