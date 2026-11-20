# Architecture

The software has two runtimes with a file-based boundary. A serial Python
package validates and compiles problems, resolves runs and reads results. A
standalone C11 engine, built on PETSc, MPI, OpenMP and parallel HDF5, does all
distributed numerical work. Python never runs on the MPI ranks, and the engine
never calls Python.

## Data flow

```text
 assets, cases, profiles (JSON)   meshes/*.msh
              │                        │
              └──── slope-stability compile ───────────►  problem package
                                                          problem.json + problem.h5
                                                                   │
 installed build (build-manifest.json), layout RxT, overrides      │
              └──── slope-stability resolve ───────────►  resolved-run.json
                                                                   │
                     mpiexec -n R slope-engine run ◄───────────────┘
                                   │
                                   ├──►  result directory: result.json, result.h5,
                                   │     environment.json, checkpoint-accepted/
                                   └──►  run directory: journal.jsonl, checkpoints/
                                                                   │
                     slope_stability.results ◄─────────────────────┘
```

Every artifact carries a `sha256:` content identity of its canonical JSON
form:

| Identity | Binds |
|---|---|
| Asset, case and profile `id` | The authoring document |
| `problem_id` | Mesh, supports, materials, boundary data, discretization and outputs in SI units |
| `build_id` | Source revision and content, dependencies, compiler and build configuration |
| `run_id` | Problem, every resolved solver policy, the MPI × OpenMP layout and the build |
| `result_id` | The published result envelope and the logical digest of its data |

The engine checks the identities and payload hashes of its inputs before it
starts, and every result names the run, problem and build that produced it.
Changing a policy, the layout or the build therefore gives a new run identity.

## Native engine

The engine is the static library `slope_native` linked into two executables:
`slope-engine` (`validate` and `run`) and `slope-probe` (a runtime probe used
by the launcher). Sources are grouped by responsibility under `src/`, with
private headers under `include/slope_internal/`:

| Module | Responsibility |
|---|---|
| `core/` | Status codes, collective status agreement, SHA-256, sorting, string and partition helpers |
| `protocol/` | Strict JSON over yyjson, generated protocol tables, the `slope-schema-check` tool |
| `io/` | Package verification, streaming reader of `problem.json`, balanced parallel HDF5 reads, resolved-run input |
| `mesh/` | DMPlex mesh from distributed input slabs, partitioning and ownership, stable identities, supports, curved coordinate closures, geometry audit, field spaces, essential constraints, transfer between field orders, field output |
| `fe/` | Lagrange bases P1–P8 on simplices, the quadrature registry, facet and curved-geometry maps (tables in `generated/`) |
| `material/` | Mohr–Coulomb with Davis A/B/C: return mapping, branches and consistent tangent; self-weight and prescribed saturation |
| `assembly/` | Assembly plans, element kernels, assembled operators, matrix-free operators and their block inverses, coarse operators for PMG |
| `hydraulic/` | Steady unconfined seepage elements and operator |
| `coupling/` | Pore pressure and its gradient evaluated at mechanics quadrature points; immutable coupling snapshots and body forces. The engine transfers on the shared mesh; the different-mesh geometric search is a test-only reference with per-point collectives |
| `linear/` | Krylov solves under a stopping policy, explicit residual checks, optional Krylov recycling (PETSc with HPDDM) |
| `pmg/` | Polynomial multigrid: hierarchy, active ranks per level, smoothers, near-nullspace, coarse solver |
| `deflation/` | Deflation bases recycled across continuation steps: admission, coarse factorization, projectors, compression, checkpoint persistence |
| `nonlinear/` | Newton solves with line search and regularization, under fixed or work control, sharing one iteration skeleton; persistent workspaces |
| `analysis/` | Continuation drivers for limit load and for direct and indirect SSR |
| `result/` | Result writer (HDF5 layout, streamed histories, logical digests, atomic publication), checkpoint write and restore, result verification |
| `diagnostics/` | Error records, events and the progress journal |
| `build/` | Compiled-in feature and parallel-HDF5 checks |
| `app/` | Engine entry point and lifecycle, solver graph construction, mechanics and hydraulic solver setup, PMG setup, checkpoints, result publication and the run monitor |

Within `app/`, the solver is split by stage:

| Files | Stage |
|---|---|
| `engine.c`, `probe.c`, `runtime.c`, `lifecycle.c` | Command line, start-up, install identity, cancellation |
| `solver.c` | `validate` and `run` orchestration |
| `solver_graph.c`, `solver_mesh_input.c`, `solver_model_input.c`, `solver_run_policy.c`, `solver_constraints.c`, `stable_id.c` | Solver graph: input slabs, models, policies, constraints |
| `solver_tuning.c` | Typed solver settings ([solver-options.md](solver-options.md)) applied to the solver stack |
| `solver_mechanics.c`, `solver_mechanics_analysis.c`, `solver_pmg_setup.c` | Mechanics operators, PMG hierarchy, continuation callbacks |
| `solver_hydraulic.c`, `solver_hydraulic_output.c`, `solver_coupling.c` | Seepage solve and acceptance, coupling snapshot |
| `solver_publish.c`, `solver_publish_views.c`, `solver_publish_mechanics.c`, `solver_publish_hydraulic.c`, `solver_postprocess.c` | Result publication: shared views, mechanics and seepage publishers, derived outputs |
| `solver_publish_monitor.c`, `solver_publish_history.c`, `solver_publish_spool.c` | Run monitor: journal, history spool, accepted-state stream |
| `solver_checkpoint*.c` | Checkpoint capture, in-run checkpoints and restart |

A `slope-engine run` proceeds as follows:

1. Initialize MPI with `MPI_THREAD_FUNNELED` and PETSc; read and verify the
   resolved run and the package on all ranks.
2. Build the solver graph: read balanced slabs of the mesh, distribute it,
   attach stable identities and supports, create field spaces and essential
   constraints, and free the input buffers.
3. For seepage cases, solve the steady unconfined problem and, for coupled
   cases, freeze the pressure field in an immutable coupling snapshot.
4. Create the mechanics operators, the PMG hierarchy and the deflation space.
5. Run the analysis: an elastic predictor, then continuation steps. Each step
   is a transaction of Newton attempts. An accepted step is journaled, its
   history rows are written to the run directory's history spool, its state
   is appended to the accepted-state stream (`accepted-states.h5`) when the
   output profile asks for every state, and an in-run checkpoint is written
   when due. A rejected attempt is rolled back, including its deflation
   basis.
6. Publish the result directory: `result.h5` is written from the final state
   and the run-directory streams, digested and renamed into place together
   with `result.json`, or a terminal error envelope is published instead.

The engine agrees on the status across all ranks before every collective
operation that a failing rank could skip, including allocations of
distributed work vectors, so a local error ends the run with a published
error. Native tests inject such failures on one rank. OpenMP regions contain
no MPI, PETSc or HDF5 calls; each owned row is written by one thread in a
fixed order.

The different-mesh coupling search is a test-only reference for the
engine's same-mesh transfer. `CPL-CONTRACT-DIFFERENT-001` checks the geometric
search with an analytic pressure field, and `CPL-CONTRACT-LAYOUT-001` also
exercises the distributed same-mesh path.

## Python layer

`python/src/` holds two packages. Neither imports PETSc or MPI.

| Module | Purpose |
|---|---|
| `slope_stability.authoring` | Strict asset, case, profile and suite documents; units; semantic validation |
| `slope_stability.compile` | Gmsh import, uniform refinement, canonical stable identities, geometry audit, binding of documents to the mesh, package writing |
| `slope_stability.catalog` | The in-repository case catalog, mesh verification and catalog compilation |
| `slope_stability.resolution` | Profile composition, overrides and resolution against an installed build |
| `slope_stability.launch` | MPI × OpenMP layouts, CPU binding, OpenMP environment, bounded execution and cancellation |
| `slope_stability.results` | Verified result reading, views and unit-aware comparators |
| `slope_stability.restart` | Comparison of runs restarted from accepted-state checkpoints |
| `slope_stability.visualization` | VTU/XDMF export of high-order results |
| `slope_stability.cli` | The `slope-stability` command |
| `slope_protocol` | Strict JSON, canonicalization and content identities shared by all modules |

## Generated code and schemas

The protocol (document kinds, enumerations, error codes and HDF5 layouts) and
the finite-element tables are defined once in `schemas/source/` and expanded
by the generators in `generators/src/slope_codegen/` into:

- `schemas/generated/json/`: JSON Schemas of every document;
- `schemas/hdf5-layout/`: the HDF5 layouts of packages, results and
  checkpoints;
- `generated/include/`, `generated/src/`: C tables compiled into the engine;
- `python/src/slope_protocol/generated.py`: the Python counterpart.

The generated files are checked in, so building the engine needs no Python
([schemas/README.md](../schemas/README.md)).

## Source documentation

Every declaration under `include/slope_internal/` has a Doxygen comment with
its parameters, return statuses, ownership and whether it is collective or
safe inside OpenMP regions. Module-internal headers next to the sources give
each function a brief. To browse the interface documentation, run Doxygen
from the repository root:

```sh
SLOPE_DOXYGEN_OUTPUT=$WORK/doxygen doxygen docs/Doxyfile
```

The HTML pages are written to `$WORK/doxygen/html`. The run fails on any
undocumented declaration ([testing.md](testing.md#documentation-checks)).
