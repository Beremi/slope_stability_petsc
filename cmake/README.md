# CMake modules

| File | Purpose |
|---|---|
| `Dependencies.cmake` | Fail-closed discovery of MPI, PETSc, parallel HDF5, BLAS/LAPACK and OpenMP; PETSc feature probes; MPI wrapper runtime-path policy |
| `BuildManifest.cmake` | Source, dependency and build identities written to `build-manifest.json`, including source trees without Git metadata |
| `InstallManifest.cmake.in` | Install step that hashes the installed executables and schemas into `install-manifest.json` |
| `ss_features.h.in` | Compiled-in feature header of the engine |
| `source-revision.txt` | Commit stamp filled in by `git archive` (`export-subst`) |
| `TestRegistration.cmake` | Registration of the native unit and MPI layout tests; the shared test labels |
| `PythonTests.cmake` | Registration of the Python contract and end-to-end tests |

The presets in `CMakePresets.json` select the build type, OpenMP, the PETSc
index width and sanitizers; [docs/building.md](../docs/building.md) explains
them, and [docs/testing.md](../docs/testing.md) the test registration.
