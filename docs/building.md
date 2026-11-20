# Building

The engine is a C11 program configured with CMake presets. The configure step
is fail-closed: it stops when PETSc, MPI, parallel HDF5, the PETSc index
width, the scalar type or a required PETSc package does not match the
selected preset. Build trees and install trees always live outside the source
tree.

## Requirements

| Component | Requirement | Tested |
|---|---|---|
| Operating system | Linux x86-64 | Arch Linux, kernel 7.3 |
| C compiler | GCC or Clang, C11 | GCC 16.2.1 |
| Build tools | CMake ≥ 3.31, Ninja | CMake 4.5.0 |
| MPI | MPI ≥ 3.1 with `MPI_THREAD_FUNNELED`; Open MPI 5 for the launcher | Open MPI 5.0.11 |
| HDF5 | Parallel HDF5 built with the same MPI | HDF5 2.2.0 |
| PETSc | 3.24.x, real double scalars, shared libraries, HDF5, HYPRE, ParMETIS (or PT-Scotch); SuperLU_DIST recommended | PETSc 3.24.2 |
| BLAS/LAPACK | Any implementation found by CMake | OpenBLAS |
| Python | CPython ≥ 3.12 for the Python package and the tests | CPython 3.12.11 and 3.14.10 |
| Formatting | clang-format 23 (development only) | 23.1.5 |
| Documentation | Doxygen (optional, development only) | 1.18.0 |

SuperLU_DIST provides the distributed direct solve of the coarsest PMG level
(`pmg.coarse_solver = preonly_lu` on more than one active rank). Without it,
that configuration fails at setup; single-rank coarse LU and GAMG coarse
solves still work. The build records whether PETSc provides SuperLU_DIST.

## PETSc

Build PETSc 3.24 once per index width. The options below are the configuration
used for testing (`environments/petsc-native-options.txt`, which also records
the source archive hash and the revisions of the downloaded packages). Point
`--with-hdf5-dir` at the prefix of your parallel HDF5 and use the MPI compiler
wrappers of the same MPI.

```sh
cd $PETSC_DIR    # unpacked petsc-3.24.2 source
./configure PETSC_ARCH=arch-linux-c-opt-i32 --with-64-bit-indices=0 \
    --with-cc=mpicc --with-cxx=mpicxx --with-fc=0 \
    --with-debugging=0 --with-shared-libraries=1 \
    --with-scalar-type=real --with-precision=double \
    --with-hdf5-dir=/usr \
    --download-hypre --download-metis --download-parmetis --download-superlu_dist \
    COPTFLAGS='-O2 -g' CXXOPTFLAGS='-O2 -g'
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=arch-linux-c-opt-i32 all
```

For 64-bit indices repeat both commands with
`PETSC_ARCH=arch-linux-c-opt-i64 --with-64-bit-indices=1`.
`environments/spack.yaml` is an alternative Spack recipe; see
[environments/README.md](../environments/README.md) for its SuperLU_DIST
caveat.

**Recommended for hybrid (MPI × OpenMP) runs:** build a second PETSc
arch whose sparse kernels use the rank's OpenMP team. Without it, PETSc's
sparse matrix-vector products run on one thread per rank, and assembled 3D
cases at the `4x8` layout took 1.3–2.8 times longer in our catalog runs (for
example `CASE-012`: 1,714 s with serial kernels, 668 s with threaded kernels
and the ordered Galerkin plan). Reuse the numerical dependencies from the first
build to keep HYPRE and SuperLU_DIST single-threaded.

```sh
DEPS=$PETSC_DIR/arch-linux-c-opt-i32
./configure PETSC_ARCH=arch-linux-c-opt-i32-omp --with-64-bit-indices=0 \
    --with-cc=mpicc --with-cxx=mpicxx --with-fc=0 \
    --with-debugging=0 --with-shared-libraries=1 \
    --with-scalar-type=real --with-precision=double --with-hdf5-dir=/usr \
    --with-hypre-dir="$DEPS" --with-metis-dir="$DEPS" \
    --with-parmetis-dir="$DEPS" --with-superlu_dist-dir="$DEPS" \
    --with-openmp=1 --with-openmp-kernels=1 \
    COPTFLAGS='-O2 -g' CXXOPTFLAGS='-O2 -g'
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=arch-linux-c-opt-i32-omp all
```

Configure the engine in a separate build directory against the new arch.
At runtime include both PETSc's new `lib` directory and the dependency
`lib` directory in the library search path. The engine's build manifest
records `petsc_openmp_kernels` so two builds using the same PETSc version
with different kernel settings have distinct build identities.

## Python environment

The Python package needs CPython 3.12 or newer. `pip install ./python`
installs its dependencies (h5py, meshio, NumPy, Pint, Pydantic) from PyPI
within the version ranges of `python/pyproject.toml`. The same interpreter
runs the Python tests, so CMake needs its path in `SLOPE_PYTHON`:

```sh
python3.12 -m venv $WORK/venv        # or any newer CPython
$WORK/venv/bin/python -m pip install ./python
export SLOPE_PYTHON=$WORK/venv/bin/python PATH=$WORK/venv/bin:$PATH
```

`uv.lock` at the repository root pins exact versions of the whole Python
workspace for reproducible environments; the lower bounds of the ranges are
these versions. `uv sync --frozen --all-packages` installs them into the
project environment.

The Python tests (`ctest -L python`) pass with these environments:

| Interpreter | Dependencies |
|---|---|
| CPython 3.12.11 | `uv.lock`: h5py 3.16.0, meshio 5.3.5, NumPy 2.4.2, Pint 0.25.2, Pydantic 2.12.5 |
| CPython 3.14.10 | `uv.lock`, as above |
| CPython 3.12.11 and 3.14.10 | Newest releases in range on 2026-11-19: h5py 3.16.0, meshio 5.3.5, NumPy 2.5.5, Pint 0.26.2, Pydantic 2.14.4 |

## Configure, build and install

The presets read four environment variables:

| Variable | Meaning |
|---|---|
| `SLOPE_BUILD_ROOT` | Parent of the build trees; each preset builds in `$SLOPE_BUILD_ROOT/<preset>` |
| `SLOPE_INSTALL_ROOT` | Parent of the install prefixes; each preset installs to `$SLOPE_INSTALL_ROOT/<preset>` |
| `SLOPE_PYTHON` | CPython (3.12 or newer) used by the Python tests and the test launchers |
| `PKG_CONFIG_PATH` | Must contain `$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig` of a PETSc with the preset's index width |

```sh
export PKG_CONFIG_PATH=$PETSC_DIR/arch-linux-c-opt-i32/lib/pkgconfig
export SLOPE_BUILD_ROOT=$WORK/build SLOPE_INSTALL_ROOT=$WORK/install
cmake --preset release-hybrid-i32
cmake --build --preset release-hybrid-i32
cmake --install $SLOPE_BUILD_ROOT/release-hybrid-i32
```

The install prefix contains:

| Path | Content |
|---|---|
| `bin/slope-engine` | The numerical engine (`validate` and `run`) |
| `bin/slope-probe` | Diagnostic probe of the MPI, OpenMP and HDF5 runtime |
| `share/slope-stability/build-manifest.json` | Source, dependency and build identities |
| `share/slope-stability/install-manifest.json` | Hashes of the installed executables and schemas |
| `share/slope-stability/schemas/` | JSON Schemas and HDF5 layouts |
| `share/slope-stability/LICENSE`, `licenses/`, `THIRD_PARTY_NOTICES.md` | Licenses |

The installed executables carry no runtime library path. Put PETSc's library
directory on `LD_LIBRARY_PATH`, or pass it to `slope-stability run
--runtime-library-path`. A quick check of the installation:

```sh
LD_LIBRARY_PATH=$PETSC_DIR/arch-linux-c-opt-i32/lib \
    $SLOPE_INSTALL_ROOT/release-hybrid-i32/bin/slope-engine --version
```

It prints the `build_id`, `dependency_id`, `executable_id` and PETSc index
width. Runs are resolved against one installed build, and every result records
its `build_id` ([outputs.md](outputs.md)).

## Presets

`cmake --list-presets` lists them. Every configure preset has matching build
and test presets.

| Preset | Build type | OpenMP | `PetscInt` | Purpose |
|---|---|---|---|---|
| `release-hybrid-i32` | Release | on | 32-bit | Default production build |
| `release-hybrid-i64` | Release | on | 64-bit | Problems beyond 32-bit indices |
| `release-mpi-i32`, `release-mpi-i64` | Release | off | 32/64-bit | MPI-only builds |
| `relwithdebinfo-hybrid` | RelWithDebInfo | on | 32-bit | Profiling |
| `debug-hybrid`, `debug-mpi` | Debug | on/off | 32-bit | Development |
| `asan-ubsan` | Debug | on | 32-bit | AddressSanitizer and UndefinedBehaviorSanitizer |
| `tsan-kernels` | Debug | on | 32-bit | ThreadSanitizer on the OpenMP kernels (Clang) |
| `coverage` | Debug | off | 32-bit | Compiler coverage instrumentation |

Use a 32-bit PETSc with the `*-i32` presets and a 64-bit PETSc with the
`*-i64` presets. With 32-bit indices the global number of unknowns must stay
below 2³¹; the 64-bit build removes that limit at the cost of larger index
arrays. The two builds have different `build_id` values, so a run resolved
for one cannot be executed by the other.

Every target compiles with `-Wall -Wextra -Wpedantic -Werror -Wconversion
-Wshadow`, so a new warning fails the build.

## Sanitizers

The `asan-ubsan` preset builds the engine and the native tests with
AddressSanitizer and UndefinedBehaviorSanitizer, and its test preset enables
leak detection. Open MPI's memory patcher conflicts with ASan, so the test
preset sets `OMPI_MCA_memory=^patcher`; `tests/native/lsan.supp` suppresses
leaks inside the uninstrumented Open MPI, PMIx and hwloc libraries only.
Test timeouts are multiplied by four in sanitizer builds. Run the suite with
fewer parallel jobs than in a release build:

```sh
cmake --preset asan-ubsan
cmake --build --preset asan-ubsan
ctest --preset asan-ubsan -j 4
```

The `tsan-kernels` preset compiles only the OpenMP worker kernels, without
PETSc or MPI, with Clang and ThreadSanitizer, and runs them under the Archer
OpenMP race detector. It needs `clang` and LLVM's `libarcher.so`:

```sh
cmake --preset tsan-kernels
cmake --build --preset tsan-kernels
ctest --preset tsan-kernels
```

## Building from an archive

The build records the source revision. A Git checkout provides it directly.
`git archive` output, including GitHub source downloads, carries it in
`cmake/source-revision.txt`, which Git fills in through the `export-subst`
attribute:

```sh
git archive --format=tar --prefix=slope_stability_petsc/ HEAD | tar -x -C $WORK
SLOPE_BUILD_ROOT=$WORK/build-archive \
    cmake -S $WORK/slope_stability_petsc --preset release-hybrid-i32
```

A separate `SLOPE_BUILD_ROOT` keeps this build apart from the one of your Git
checkout. For a source tree with neither Git metadata nor a filled-in stamp,
configuration stops with an error; pass the full commit identifier instead:
`cmake --preset release-hybrid-i32 -DSLOPE_SOURCE_REVISION=<40-hex commit>`.
Without Git metadata the source identity is the content digest of every file
in the tree, so a modified archive gets a different identity.

## Troubleshooting

- **PETSc version, index width or package missing.** The configure step names
  the mismatch. Check `PKG_CONFIG_PATH` against the preset's index width.
- **MPI wrapper runtime paths.** The installed engine must not carry an
  absolute RUNPATH. CMake drops runtime paths that the MPI wrapper adds for
  default loader directories and fails on any other.
  `-DSLOPE_MPI_WRAPPER_RPATH=strip` drops all of them;
  `-DSLOPE_MPI_RPATH_ALLOWLIST=<dir>` accepts specific directories.
- **HDF5 and MPI mismatch.** CMake compares the MPI wrapper behind the HDF5
  compiler wrapper with the MPI found for the project and fails when they
  differ.
- **`libpetsc.so` not found at run time.** See the install section above.
