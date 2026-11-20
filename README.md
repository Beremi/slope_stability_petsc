# slope_stability_petsc

A parallel finite-element solver for slope-stability analysis. The numerical
engine is a standalone C11 program built on PETSc, MPI, OpenMP and parallel
HDF5. A serial Python package prepares problems, launches the engine and reads
its results.

## Capabilities

- **Material model.** Elastic–perfectly plastic Mohr–Coulomb with the Davis A,
  B and C strength-reduction variants for non-associated flow.
- **Analyses.** Limit load with work-controlled continuation, and strength
  reduction (SSR) with direct or indirect continuation.
- **Hydraulics.** Steady unconfined seepage, coupled one way into mechanics
  through the pore-pressure gradient and saturated weight (or the weight
  only). Uncoupled mechanics can instead take a prescribed water level.
- **Discretization.** Lagrange P1, P2 and P4 elements on triangles (2D) and
  tetrahedra (3D), with curved geometry of order 1, 2 or 4 read from Gmsh
  high-order meshes. P8 elements are available but experimental.
- **Linear solvers.** Polynomial multigrid (PMG) preconditioning, deflation
  with bases recycled across continuation steps, and PETSc Krylov methods.
  Mechanics operators of order 2 and higher can also be applied matrix-free.
- **Parallelism.** MPI domain decomposition with OpenMP threads inside each
  rank. Each rank reads its share of the mesh with parallel HDF5; no rank
  assembles the global mesh.
- **Runs.** Strict, content-addressed inputs and results; live progress
  journal; accepted-state checkpoints with restart, also on another MPI
  layout; cancellation at a safe point.

## Status

The implementation is complete for the capabilities above, and its numerical
components are covered by unit, regression and MPI/hybrid tests
([docs/testing.md](docs/testing.md)). Scientific validation against
independent reference solutions is not claimed. The 23 catalog cases are
reference inputs, not validated benchmarks
([docs/validation-cases.md](docs/validation-cases.md)). Known limitations
are listed in the [changelog](CHANGELOG.md).

## Quick start

The supported platform is Linux x86-64 with GCC or Clang, Open MPI 5, PETSc
3.24 (real double precision, with parallel HDF5, HYPRE, METIS/ParMETIS and
SuperLU_DIST), CMake 3.31 or newer, Ninja and CPython 3.12 or newer. Configure PETSc as
described in [docs/building.md](docs/building.md); for MPI × OpenMP runs use its
recommended build with threaded sparse kernels (`--with-openmp-kernels=1`).

Choose a work directory outside the source tree, then build and install the
engine and the Python package:

```sh
export WORK=$HOME/slope-stability                 # build, install and run area
export PETSC_DIR=/path/to/petsc-3.24.2 PETSC_ARCH=arch-linux-c-opt-i32
export PKG_CONFIG_PATH=$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig
export SLOPE_BUILD_ROOT=$WORK/build SLOPE_INSTALL_ROOT=$WORK/install

python3.12 -m venv $WORK/venv
$WORK/venv/bin/python -m pip install ./python
export SLOPE_PYTHON=$WORK/venv/bin/python PATH=$WORK/venv/bin:$PATH

cmake --preset release-hybrid-i32
cmake --build --preset release-hybrid-i32
cmake --install $SLOPE_BUILD_ROOT/release-hybrid-i32
```

Run the small 2D strength-reduction case `CASE-003` on 2 MPI ranks with 2
OpenMP threads each, from the source tree:

```sh
slope-stability verify-meshes
slope-stability compile --case-id CASE-003 --destination $WORK/compiled
slope-stability run --prefix $SLOPE_INSTALL_ROOT/release-hybrid-i32 \
    --runtime-library-path $PETSC_DIR/$PETSC_ARCH/lib \
    --compiled $WORK/compiled --case-id CASE-003 --layout 2x2 \
    --destination $WORK/run-2x2
slope-stability inspect $WORK/run-2x2/results/case-003-canonical
```

Each command prints JSON lines. The last one reports the terminal status
(`succeeded`), the analysis outcome (`converged_criterion`) and the published
quantities. [docs/running.md](docs/running.md) explains the commands and
[docs/outputs.md](docs/outputs.md) the result directory.

## Documentation

| Document | Content |
|---|---|
| [docs/building.md](docs/building.md) | Toolchain, PETSc configuration, presets, sanitizers, archives |
| [docs/running.md](docs/running.md) | The `slope-stability` command and the engine command line |
| [docs/parallel-execution.md](docs/parallel-execution.md) | Choosing MPI ranks × OpenMP threads, binding, measured guidance |
| [docs/input-format.md](docs/input-format.md) | Assets, cases, profiles, units, supports, prescribed saturation |
| [docs/solver-options.md](docs/solver-options.md) | Solver policies and their options |
| [docs/outputs.md](docs/outputs.md) | Result directory, outcomes, histories, checkpoints, restart |
| [docs/architecture.md](docs/architecture.md) | Data flow, native modules, a source map, the Python layer and the Doxygen interface documentation |
| [docs/validation-cases.md](docs/validation-cases.md) | The 23-case catalog |
| [docs/testing.md](docs/testing.md) | Test presets, labels, identifiers and documentation checks |
| [meshes/README.md](meshes/README.md) | The 13 shipped meshes and their origin |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development workflow and conventions |
| [CHANGELOG.md](CHANGELOG.md) | Release notes |

## Repository map

| Path | Content |
|---|---|
| `src/`, `include/slope_internal/` | Native engine sources and private headers ([src/README.md](src/README.md)) |
| `generated/`, `generators/` | Checked-in generated C tables and their Python generators |
| `third_party/yyjson/` | Vendored JSON parser |
| `python/` | The `slope-stability` Python package and command ([python/README.md](python/README.md)) |
| `schemas/` | Protocol source, generated JSON Schemas and HDF5 layouts ([schemas/README.md](schemas/README.md)) |
| `cases/catalog/` | Assets, cases and solver profiles of the 23 catalog cases |
| `meshes/` | The 13 Gmsh source meshes with SHA-256 sums |
| `tests/` | Native and Python tests ([docs/testing.md](docs/testing.md)) |
| `cmake/`, `CMakeLists.txt`, `CMakePresets.json` | Build system ([cmake/README.md](cmake/README.md)) |
| `environments/` | Dependency recipes, PETSc configure options and lock files |
| `licenses/`, `THIRD_PARTY_NOTICES.md` | License texts and reuse records |

## License and attribution

The project is distributed under the BSD 2-Clause License ([LICENSE](LICENSE)).

- The formulation, case definitions and meshes originate from the MATLAB
  implementation [sysala/slope_stability](https://github.com/sysala/slope_stability)
  (BSD 2-Clause, © 2023 Stanislav Sysala) and its PETSc/Python successor
  [Beremi/slope_stability_petsc4py](https://github.com/Beremi/slope_stability_petsc4py).
  No MATLAB source is included.
- [yyjson](https://github.com/ibireme/yyjson) 0.12.0 is vendored unmodified
  under the MIT License.
- The degree 14–19 simplex quadrature tables are adapted from
  [PETSc](https://petsc.org) (BSD 2-Clause). PETSc itself, MPI, HDF5 and the
  other numerical libraries are linked, not distributed.

[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[licenses/](licenses/) hold the notices and the per-item reuse records.
