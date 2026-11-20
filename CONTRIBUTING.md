# Contributing

Contributions are welcome as pull requests. This page describes the
development workflow and the conventions that reviews check.

## Workflow

1. Build a debug and a release preset ([docs/building.md](docs/building.md))
   and run the test suite once, so that you know the baseline on your
   machine.
2. Work on a branch. Keep each commit to one coherent change, with a message
   that states the outcome ("Keep PMG coarse setup collective"), not the
   activity ("update pmg.c").
3. For a bug fix, first add a test that fails without the fix: a `REG` test
   for the defect, or an `ORACLE` test against an independent computation
   ([docs/testing.md](docs/testing.md#writing-tests)).
4. Run the tests that cover the change, then the full release suite. Changes
   to OpenMP kernels, memory handling or MPI communication also need the
   `asan-ubsan` suite, and kernel changes the `tsan-kernels` suite.
5. Check formatting and lint (below), update the documentation affected by
   the change, and add a line to [CHANGELOG.md](CHANGELOG.md).

Never weaken a tolerance or disable a test to make an unrelated change pass;
report the failure instead. Numerical changes should state, in the pull
request, how results changed on the affected tests or catalog cases.

## C conventions

- **Language.** C11 without compiler extensions. The build uses
  `-Wall -Wextra -Wpedantic -Werror -Wconversion -Wshadow`, so code must
  compile without warnings with GCC and Clang.
- **Formatting.** clang-format 23 with the repository's `.clang-format`
  (LLVM style, two-space indent, 88 columns). Check the files you changed
  against the main branch:

  ```sh
  git diff --name-only --diff-filter=ACMR -z main -- '*.c' '*.h' |
      xargs -0 -r clang-format --dry-run --Werror
  ```

- **Files.** Every source file starts with a short comment that states its
  purpose and the line `SPDX-License-Identifier: BSD-2-Clause` (a `@file`
  Doxygen block in the headers under `include/`). Keep translation units
  cohesive and below about 1,500 lines; split a growing file by
  responsibility.
- **Interfaces.** Every declaration under `include/slope_internal/` has a
  Doxygen comment: a one-sentence brief, `@param[in|out]` for every
  parameter with units and ownership, `@return` with the status values it
  can return, and whether it is collective (on which communicator) and safe
  inside OpenMP regions. Types, members, enumerators and macros are
  documented too, so `doxygen docs/Doxyfile` reports no warning.
  Module-internal declarations go into a `*_internal.h` next to the sources,
  each with at least a brief.
- **Errors.** Functions return `SsStatus`. A failure on one rank must be
  agreed across the communicator (see `core/collective.h`) before the next
  collective operation, so that no rank waits forever.
- **PETSc objects.** Each object has one owner that creates and destroys it;
  error paths release everything they created. Reuse workspaces across
  iterations instead of allocating per step.
- **Threads.** OpenMP regions do not call MPI, PETSc or HDF5, and each thread
  writes disjoint data in a deterministic order. No hidden global state:
  pass context explicitly.
- **Memory.** Distributed data stays distributed. Do not gather mesh-sized
  or vector-sized data onto one rank.

## Python conventions

- Code that runs on CPython 3.12 and newer, type hints on public functions, a
  docstring on every module and public symbol, and an explicit `__all__`.
- Lint and format with Ruff (configuration in `pyproject.toml`; the
  generated modules are excluded from formatting):

  ```sh
  python -m pip install ruff
  ruff check python/src generators/src tests
  ruff format --check python/src generators/src tests
  ```

- The Python layer never imports PETSc or MPI and never runs on MPI ranks.

## Documentation

Update the page in `docs/` that describes a changed behavior, command or
option, and keep `README.md` and the changelog in step. Before a pull
request, run the documentation checks
([docs/testing.md](docs/testing.md#documentation-checks)):

```sh
python3 tests/build/documentation_checks.py
SLOPE_DOXYGEN_OUTPUT=$WORK/doxygen doxygen docs/Doxyfile
```

`docs/solver-options.md` is generated from the solver option registry: edit
`include/slope_internal/core/solver_options.def` and regenerate as described
below.

## Generated code

The protocol, the JSON Schemas, the HDF5 layouts and the finite-element
tables are generated from `schemas/source/`. Edit the source, regenerate and
commit the outputs together ([schemas/README.md](schemas/README.md)). Do not
edit files under `generated/` or `schemas/generated/` by hand.

## Licensing

The project is BSD-2-Clause. New C, Python, CMake and TOML files carry the
SPDX line above. Code or data
adapted from another project needs a compatible license, an entry in
`licenses/reuse-ledger.csv` and, where required, a notice in
`THIRD_PARTY_NOTICES.md` with the license text in `licenses/`.

## Housekeeping

Build trees, install trees, run outputs and virtual environments belong
outside the source tree. `pip install ./python` leaves `python/build/` and
`python/src/slope_stability.egg-info/` behind; do not commit them.
