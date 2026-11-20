# Changelog

All notable changes to this project are recorded here. Versions follow
[Semantic Versioning](https://semver.org).

## [0.1.0] - 2026-11-19

First public release.

### Added

- Standalone C11 engine `slope-engine` on PETSc 3.24, MPI, OpenMP and
  parallel HDF5, with `validate` and `run` operations.
- Elastic–perfectly plastic Mohr–Coulomb material with the Davis A, B and C
  variants.
- Limit-load analysis and strength reduction with direct and indirect
  continuation.
- Steady unconfined seepage with one-way coupling into mechanics, and
  prescribed water levels for uncoupled mechanics.
- Lagrange P1, P2 and P4 elements on triangles and tetrahedra with curved
  geometry of order 1, 2 and 4; experimental P8.
- Polynomial multigrid preconditioning, recycled deflation, assembled and
  matrix-free operators; the distributed matrix-free apply overlaps its halo
  exchange with interior work.
- Typed solver settings in one registry, shared by the engine, the resolver
  and the generated reference [docs/solver-options.md](docs/solver-options.md).
- Hybrid MPI/OpenMP execution with distributed mesh input, explicit per-rank
  core binding and bounded OpenMP waiting; no rank gathers mesh- or
  field-sized data.
- A qualification mode (`validation.qualification_mode`) with a fixed
  summation order that makes results independent of the OpenMP team size.
- Content-addressed problem packages, resolved runs and results; streamed
  result histories; live progress journal; accepted-state checkpoints with
  restart on another layout; cancellation at a safe point.
- Python package and `slope-stability` command: `verify-meshes`, `compile`,
  `resolve`, `run` and `inspect`; result reader, comparators and VTU/XDMF
  export.
- Case catalog of 23 reference cases (24 variants) and the 13 meshes they
  use.
- Unit, contract, oracle, regression, MPI/hybrid layout and end-to-end tests
  with release, debug and sanitizer presets, named after the component they
  test.
- Doxygen documentation of the native interface (`docs/Doxyfile`), SPDX
  license lines on every source file, and tests that check both and the
  Markdown links.

### Removed

- Removed test-only native helpers for staged coupling, restart tokens,
  component teardown, nested events, typed counters, descriptor registries,
  allocation tracking, error scopes and process redaction. The
  [retired test identifiers](docs/testing.md#retired-test-identifiers)
  document the test suite changes.

### Known limitations

- Scientific validation against independent reference solutions is not
  claimed.
- `CASE-019` (3D, P4) is a long run that did not complete with its catalog
  policies: its operator workspace exceeds the default memory admission
  limit, and with a raised limit a step was rejected at the Newton iteration
  limit ([docs/validation-cases.md](docs/validation-cases.md#known-limitations)).
- Most catalog solver profiles were carried over from the earlier
  implementations and are not tuned for this solver.
- Continuation supports up to 128 accepted SSR states or 127 limit-load
  steps plus the elastic reference state; larger requested horizons are
  rejected.
- The optional stable-hash shared-point balancer makes its greedy ownership
  decisions in rank order, with a setup critical path linear in the rank count.
- Numerical results can differ across MPI layouts or solver backends.
  Performance-mode runs (the default) can also differ between repeats at a
  fixed multi-rank layout because PETSc may assemble Galerkin contributions
  in message completion order
  ([docs/parallel-execution.md](docs/parallel-execution.md#reproducibility)).
- Seepage and mechanics are coupled on the shared mesh. A transfer between
  different meshes exists only as a tested reference implementation that
  does not scale.
- Krylov recycling through HPDDM is not covered by the tests.
- Performance has been measured on one single-socket workstation only;
  multi-NUMA and multi-node runs are not characterized.
