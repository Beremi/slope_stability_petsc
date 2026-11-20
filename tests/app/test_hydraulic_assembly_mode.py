# SPDX-License-Identifier: BSD-2-Clause
"""Verify that steady seepage uses the resolved validation assembly mode."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from slope_stability.results import open_result
from tests.app.test_engine_contract import InstalledFixture


class HydraulicAssemblyMode(unittest.TestCase):
    """Check the actual hydraulic operator through the installed engine."""

    def test_effective_mode(self) -> None:
        """APP-HYDRAULIC-ASSEMBLY-MODE-001."""
        with tempfile.TemporaryDirectory(prefix="engine-hydraulic-mode-") as raw:
            directory = Path(raw)
            pressure = {}
            for mode in ("qualification", "performance"):
                with self.subTest(mode=mode):
                    fixture = InstalledFixture(
                        directory / mode,
                        2,
                        hydraulic=True,
                        field_order=2,
                        two_cells=True,
                        qualification_mode=mode == "qualification",
                    )
                    destination = directory / f"{mode}-result"
                    completed = fixture.validate(destination, operation="run")
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    journal = destination.with_name(destination.name + ".run")
                    events = [
                        json.loads(line)
                        for line in (journal / "journal.jsonl").read_text().splitlines()
                    ]
                    self.assertEqual(
                        [
                            event["mode"]
                            for event in events
                            if event["event"] == "assembly_mode"
                            and event["field"] == "hydraulic"
                        ],
                        [mode],
                    )
                    with open_result(destination) as result:
                        pressure[mode] = (
                            result.field("pore_pressure", state="terminal")
                            .read()
                            .values
                        )
            if len(pressure) == 2:
                np.testing.assert_allclose(
                    pressure["qualification"],
                    pressure["performance"],
                    rtol=1e-8,
                    atol=1e-11,
                )
