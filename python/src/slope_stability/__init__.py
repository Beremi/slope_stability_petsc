# SPDX-License-Identifier: BSD-2-Clause
"""Serial preparation, launch and result layer of the slope-stability engine.

The package never runs on numerical MPI ranks. Its subpackages and modules are

- :mod:`slope_stability.authoring`: strict asset, case, profile and suite
  documents with unit and semantic validation;
- :mod:`slope_stability.compile`: mesh import, canonicalization and
  content-addressed problem packages;
- :mod:`slope_stability.resolution`: run resolution against an installed
  build (``resolved-run.json``);
- :mod:`slope_stability.launch`: MPI x OpenMP launch layout, CPU binding and
  bounded execution;
- :mod:`slope_stability.results`: verified result reading and comparators;
- :mod:`slope_stability.restart`: accepted-boundary restart comparison;
- :mod:`slope_stability.visualization`: derived VTU/XDMF exports.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
