# Protocol schemas

The protocol shared by the Python layer and the engine is defined once in
`source/` and expanded by the generator in `generators/src/slope_codegen/`.

| Path | Content |
|---|---|
| `source/protocol.json` | Document kinds, envelopes, enumerations and error codes |
| `source/hdf5-layouts.json` | Logical layouts of `problem.h5`, `result.h5` and `checkpoint.h5` |
| `source/conformance.json` | Cross-language conformance cases |
| `source/finite-elements.json` | Finite-element families and quadrature rules |
| `source/petsc-minimal-symmetric-simplex.json` | Degree 14–19 simplex quadrature orbits adapted from PETSc |
| `generated/json/` | JSON Schemas of every document: asset, case, profile, suite, compiled problem, resolved run, result, checkpoint, error, events, build |
| `hdf5-layout/generated-layouts.json` | Expanded HDF5 layouts |
| `fixtures/` | Generated conformance fixtures and invalid JSON inputs that every reader must reject |

The generator also writes the C tables in `generated/` and
`python/src/slope_protocol/generated.py`. All outputs are checked in. After
editing a source file, regenerate them and check that the tree is consistent:

```sh
PYTHONPATH=generators/src python -m slope_codegen.protocol
PYTHONPATH=generators/src python -m slope_codegen.protocol --check
```

The generated C is formatted with clang-format 23. A schema change that alters
the meaning of existing documents needs a new schema version, because document
and result identities are content hashes. The JSON Schemas are installed to
`share/slope-stability/schemas/`.
