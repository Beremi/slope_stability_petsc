# Locked environments

`spack.yaml` is the reviewed native recipe and `spack.lock` is its concrete
Spack 1.2.2 resolution. It has portable `x86_64` roots for:

- LLVM/clang-format 23.1.5 used by deterministic generation;
- PETSc 3.24.2 with 32-bit indices; and
- PETSc 3.24.2 with 64-bit indices.

Both PETSc roots are real-double, shared, MPI-enabled builds with parallel
HDF5 2.2.0, HYPRE 3.0.0, METIS/ParMETIS, and OpenBLAS. Open MPI 5.0.11,
parallel HDF5, OpenBLAS, GCC 16.2.1, and LLVM 23.1.5 are explicit site
externals in this lock; a different qualified site must regenerate and review
its own lock.

The external versions in `spack.lock` were updated by hand to the toolchain
observed on the reference machine (2026-11-19: HDF5 2.2.0, Open MPI 5.0.11,
GCC 16.2.1, clang-format/LLVM 23.1.5). The concrete spec hashes are still
those of the earlier resolution; re-concretize with Spack 1.2.2 before using
the lock to install.

SuperLU_DIST is required for distributed coarse LU (`preonly_lu` on more than
one active rank) and is part of the qualified manual PETSc configurations in
`petsc-native-options.txt` (`--download-superlu_dist`, SuperLU_DIST 9.2.0).
The Spack `petsc` package offers its `superlu-dist` variant only together
with `+fortran`, so the Fortran-free Spack roots above do not provide it;
use the manual configuration, or add `+fortran +superlu-dist` to both PETSc
roots and re-qualify.

`dependency-contract.json` records the narrow versions, source archive/hash,
downloaded-package revisions, and resolver identities used by the accepted
foundation and serial authoring stack. `sbom.cdx.json` is the matching
CycloneDX dependency projection. `uv.lock` at the repository root freezes the
Python workspace for CPython 3.12 or newer, including NumPy/HDF5/meshio, Pint,
and Pydantic plus their transitive dependencies.

The two manually qualified PETSc configurations are recorded verbatim in
`petsc-native-options.txt`. Their source archive and all build/test logs are
external evidence, not repository content.

Regenerate locks only as a dedicated reviewed change:

```sh
spack -e environments concretize --force
uv lock --python /absolute/python3.12
```

Then regenerate protocol output, run all compact native/Python contracts,
both index builds, install relocation, MPI/hybrid checks, and review numerical
and performance baselines before accepting the lock change.

The ASan/UBSan preset keeps LeakSanitizer enabled for project-only native
tests. Its two MPI-launch rows set Open MPI `memory=^patcher` because the
allocator-patching component conflicts with ASan, and disable leak detection
only for the uninstrumented Open MPI/PMIx/hwloc lifetime allocations. Address
and undefined-behavior detection remain enabled. The TSan preset uses
Clang/LLVM OpenMP and `/usr/lib/libarcher.so` with
`ignore_noninstrumented_modules=1`; it never links a project worker kernel to
PETSc or MPI.
