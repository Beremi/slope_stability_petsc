# SPDX-License-Identifier: BSD-2-Clause
"""Installed PMG regression for shared curved coordinates and empty coarse ownership."""

from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from slope_stability.compile import compact_simplex_mesh
from slope_stability.results import open_result
from tests.app import test_engine_contract as app


class CurvedPmgTest(unittest.TestCase):
    def test_two_rank_p4_g4(self) -> None:
        mesh = compact_simplex_mesh(2, 4, two_cells=True)
        points = np.array(mesh.coordinate_nodes, copy=True)
        x, y = points.T
        x += 0.06 * x * (1.0 - x) * y
        mesh = replace(mesh, coordinate_nodes=points)
        asset_document, case_document = app.asset_document, app.case_document

        def curved_asset(dimension, **options):
            return asset_document(dimension, **(options | {"coordinate_order": 4}))

        def curved_case(dimension, **options):
            return case_document(dimension, **(options | {"coordinate_order": 4}))

        with (
            app.saturation_artifact_directory("pmg-curved-") as temporary,
            patch.object(app, "asset_document", curved_asset),
            patch.object(app, "case_document", curved_case),
        ):
            root = Path(temporary)
            fixture = app.InstalledFixture(root, 2, field_order=4, mesh_override=mesh)
            self.assertTrue(json.loads(fixture.run_json.read_bytes())["pmg"]["enabled"])
            completed = fixture.validate(root / "result", operation="run")
            self.assertEqual(
                completed.returncode, 0, completed.stdout + completed.stderr
            )
            open_result(root / "result")
