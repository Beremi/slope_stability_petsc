# Build-time generators

Generators are serial build-time tools. Checked-in generated C is the native
compile input; no Python interpreter is required to build or run the engine.

| Module | Inputs | Outputs |
|---|---|---|
| `slope_codegen.protocol` | `schemas/source/*.json` and the two modules below | JSON Schemas, HDF5 layouts, `generated/`, `python/src/slope_protocol/generated.py`, `generated/manifest.json` |
| `slope_codegen.finite_elements` | `schemas/source/finite-elements.json`, PETSc quadrature orbits | `generated/*/ss_fe_generated.*`, `python/src/slope_stability/_fe_layout.py` |
| `slope_codegen.solver_options` | `include/slope_internal/core/solver_options.def` | `docs/solver-options.md`, `python/src/slope_stability/_solver_settings.py`, schema fields |

`python -m slope_codegen.protocol` runs all three and rewrites the manifest;
`--check` reports any drift ([schemas/README.md](../schemas/README.md)).
