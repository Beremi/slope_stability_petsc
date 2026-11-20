# Testing

The tests are registered with CTest. They cover native C units, MPI and
hybrid MPI/OpenMP layouts, the Python layer and end-to-end runs of the
installed engine. The hybrid release presets with SuperLU_DIST register 251
tests, plus one documentation check when Doxygen is available.

## Running the tests

Configure and build a preset as in [building.md](building.md), with
`SLOPE_PYTHON` pointing to a CPython 3.12 or newer that has the Python
package's dependencies installed. Then:

```sh
ctest --preset release-hybrid-i32 -j 6
```

On the 32-core workstation described in
[parallel-execution.md](parallel-execution.md) the release suite takes about
eight minutes. Tests declare how many processors they use, and CTest keeps
the total within `-j`. Some hybrid tests launch the `4x8` layout and need 32
cores. Keep `PKG_CONFIG_PATH` exported as it was during
configuration; one build test reconfigures the project.

Useful selections:

```sh
ctest --preset release-hybrid-i32 -N                 # list the tests
ctest --preset release-hybrid-i32 -R '^PMG-'         # one identifier prefix
ctest --preset release-hybrid-i32 -L '^pmg$'         # one area label
ctest --preset release-hybrid-i32 -L serial -j 8     # single-process tests
ctest --preset release-hybrid-i32 -R E2E-RESTART-001 --output-on-failure
ctest --preset release-hybrid-i32 -R PMG-CONTRACT-REUSE-001 -N -V  # show the command
ctest --test-dir $SLOPE_BUILD_ROOT/release-hybrid-i32 --print-labels
```

The pure Python tests also run without a build:

```sh
PYTHONPATH=python/src:generators/src:. python -m unittest tests.authoring.test_authoring_contract
```

## Test identifiers

Test names have the form `AREA-KIND-NAME-NNN`, for example
`PMG-CONTRACT-ACTIVE-001`.

| Area | Label | Component |
|---|---|---|
| `COR`, `DIA`, `IO` | `core`, `diagnostics`, `io` | Native core, diagnostics, input |
| `MSH`, `MESH`, `FE`, `MAT` | `mesh`, `fe`, `material` | Mesh, finite elements, material |
| `ASM`, `HYD`, `CPL` | `assembly`, `hydraulic`, `coupling` | Assembly, seepage, coupling |
| `LIN`, `DEF`, `PMG`, `NLS`, `SOLVER` | `linear`, `deflation`, `pmg`, `nonlinear`, `solver` | Linear solvers, deflation, PMG, Newton and the solver stack |
| `ANA`, `OUT` | `analysis`, `output` | Continuation analyses; results and checkpoints |
| `APP`, `E2E` | `app`, `e2e` | Installed engine, end-to-end runs |
| `PRO`, `BLD` | `protocol`, `build` | Protocol and generated code; build, install, licenses and documentation |
| `AUT`, `CMP`, `CAT`, `RES`, `LCH`, `RDR`, `VIS` | `authoring`, `compilation`, `catalog`, `resolution`, `launch`, `reader`, `visualization` | Python authoring, compilation, catalog, resolution, launch, result reader, visualization |

A few identifiers keep a prefix outside this table: `MECHANICS-REG-FCG-001`
(label `linear`), `SOLVER-OPTIONS-DOC-001` (label `build`) and the
cross-cutting `QAL-…` contracts (labels `mesh`, `launch`, `restart` and
`catalog`).

| Kind | Meaning |
|---|---|
| `UNIT` | A single function or kernel |
| `CONTRACT` | Behavior across an interface, often at several MPI × OpenMP layouts |
| `ORACLE` | Comparison with an independent analytic or reference computation |
| `REG` | Regression test for a fixed defect |
| `NEG` | Rejection of invalid input or an injected failure |
| `PERF` | Memory or I/O budget |
| `PROP` | Property test over generated inputs |

Other kinds, such as `CANCEL`, `RESTART` or `VALIDATE`, name the feature
under test.

### Retired test identifiers

The following tests covered native helpers that had no engine callers.

| Removed helper | Retired test IDs |
|---|---|
| Staged coupled workflow | `ANA-CONTRACT-STAGING-001` |
| Component teardown model | `APP-LIFE-CLEANUP-001`, `APP-MPI-ERROR-001` |
| Tracking allocator | `COR-UNIT-ALLOC-001` |
| Descriptor registry | `COR-UNIT-REGISTRY-001` |
| Error scope | `COR-UNIT-ERROR-001` |
| Nested event journal | `DIA-UNIT-EVENT-001` |
| Typed counters | `DIA-UNIT-COUNTER-001` |
| Process redaction | `DIA-UNIT-REDACT-001` |

`ANA-CONTRACT-RESTART-001` exercises the continuation controller's resume
path. The empty and singleton sort cases run under `IO-JSON-STRICT-001`.
`DIA-UNIT-ERROR-001`, `DIA-CONTRACT-MPI-001` and `ASM-COLLECTIVE-ERROR-001`
cover the engine's error ordering and collective selection.

## Labels

Each test carries three kinds of labels:

- its area label from the table above;
- `native` for the C test executables or `python` for the Python tests;
- execution labels: `serial` (one process), `mpi` (launched by `mpiexec`),
  `hybrid` (at least one layout with several OpenMP threads per rank),
  `openmp`, `integration` (Python tests that run the engine), `oracle`,
  `restart`, `smoke`, and `sanitizer` and `race` in the `tsan-kernels`
  preset.

`--print-labels` lists them. Label selection with `-L` is a case-sensitive
regular expression, so anchor area labels that occur inside other labels:
`-L '^io$'` selects the input tests, while `-L io` also matches
`integration` and `visualization`.

## Layout tests

The multi-process native tests are executables under `tests/native/` that
`tests/native/support/mpi_layout_runner.py` launches at one or more layouts.
A layout is `R` (`R` ranks with the inherited OpenMP settings), `RxT` (`R`
ranks with `T` OpenMP threads each) or, for PMG, `RxT/A` (`A` of the ranks
active on the coarse levels), for example
`--layout 1x1 --layout 2x1 --layout 4x1 --layout 2x4 --layout 4x8`. The
`--suite` option (`app`, `mesh`, `linear`, `assembly`, `pmg`, `nonlinear`,
`hydraulic`, `result`) selects the executable's command line, the MPI
binding and the checks. Rank 0 of each launch writes a JSON report; the
runner checks the reported layout and the error bounds of the suite, and
requires every layout to report the same partition-independent invariant
and checksums within a tolerance. Artifacts are written under
`test-artifacts/<test id>/` of the build tree.
`tests/native/support/hdf5_matrix_runner.py` runs the parallel-HDF5 probe at
1, 2 and 4 ranks.

## Sanitizers

```sh
ctest --preset asan-ubsan -j 4
ctest --preset tsan-kernels
```

The `asan-ubsan` suite runs the full test set under AddressSanitizer,
UndefinedBehaviorSanitizer and LeakSanitizer. With four jobs it took between
13 and 29 minutes on the workstation above, depending on other load. `tsan-kernels` runs the
OpenMP kernels under ThreadSanitizer and the Archer race detector.
[building.md](building.md#sanitizers) describes both builds.

## Documentation checks

Four `BLD-DOC-…` tests keep the sources and documentation consistent. They
need only Python and Git, and run in a plain checkout:

| Test | Check |
|---|---|
| `BLD-DOC-SPDX-001` | Every owned C, header, Python, CMake and TOML file has the line `SPDX-License-Identifier: BSD-2-Clause` |
| `BLD-DOC-HEADERS-001` | Every function declared in a header has a Doxygen comment; under `include/` it documents every parameter and the return value |
| `BLD-DOC-LINKS-001` | Every relative link in a tracked Markdown file resolves |
| `BLD-DOC-DOXYGEN-001` | `doxygen docs/Doxyfile` reports no warning for `include/`; registered only when CMake finds `doxygen` |

The same checks run without CTest:

```sh
python3 tests/build/documentation_checks.py            # spdx, prototypes, links
SLOPE_DOXYGEN_OUTPUT=$WORK/doxygen doxygen docs/Doxyfile
```

## Writing tests

- Put native tests next to the existing ones in `tests/native/`, named after
  the component they test (`<component>_test.c`, `<component>_mpi_test.c`
  for multi-rank tests), and shared native fixtures in
  `tests/native/fixtures/`. Define the executable with
  `slope_test_executable` in `CMakeLists.txt` and register each test in
  `cmake/TestRegistration.cmake` with `slope_native_test` (one process or a
  fixed `mpiexec` launch) or `slope_native_layout_test` (layout runner).
- Put Python tests into the `tests/<area>/` package (for example
  `test_<area>_contract.py`) and register them with `slope_python_unittest`
  in `cmake/PythonTests.cmake`.
- Give each test a new identifier in the scheme above. The registration
  helpers derive the area label from the prefix; pass `AREA` for any other
  prefix, and `LABELS` for further execution labels.
- When fixing a defect, add a failing `REG` or `ORACLE` test first.
- Prefer small analytic or manufactured problems with explicit tolerances
  over catalog runs.
