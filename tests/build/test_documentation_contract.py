# SPDX-License-Identifier: BSD-2-Clause
"""Source headers, interface documentation and Markdown link contracts."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from tests.build import documentation_checks

__all__ = ["DocumentationContract"]

ROOT = documentation_checks.ROOT


class DocumentationContract(unittest.TestCase):
    """Checks that keep the published sources self-describing."""

    maxDiff = None

    def test_spdx_headers(self) -> None:
        """Every owned source file names its license."""
        self.assertEqual(documentation_checks.check_spdx(ROOT), [])

    def test_header_documentation(self) -> None:
        """Every header prototype is documented; include/ fully."""
        self.assertEqual(documentation_checks.check_prototypes(ROOT), [])

    def test_markdown_links(self) -> None:
        """Every relative Markdown link resolves inside the tree."""
        self.assertEqual(documentation_checks.check_links(ROOT), [])

    def test_doxygen_clean(self) -> None:
        """Doxygen reports no warning for include/ when it is available."""
        doxygen = os.environ.get("SLOPE_DOXYGEN", "")
        if not doxygen:
            self.skipTest("SLOPE_DOXYGEN is not set")
        with tempfile.TemporaryDirectory(prefix="slope-doxygen-") as output:
            result = subprocess.run(
                [doxygen, str(Path("docs") / "Doxyfile")],
                cwd=ROOT,
                env={**os.environ, "SLOPE_DOXYGEN_OUTPUT": output},
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
