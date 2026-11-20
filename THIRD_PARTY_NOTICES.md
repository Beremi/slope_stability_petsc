# Third-Party Notices

The project is distributed under the BSD-2-Clause licence in `LICENSE`.
Material adapted from other works is listed below; every reuse record with
its origin revision, adaptation and owning tests is in
`licenses/reuse-ledger.csv`.

## sysala/slope_stability (MATLAB reference)

The scientific reference is `sysala/slope_stability` at revision
`8401cc52d9594b60a2b45df35f15c509ccdc9884`. Its BSD-2-Clause notice
(Copyright (c) 2023, Stanislav Sysala) is preserved at
`licenses/sysala-slope-stability-BSD-2-Clause.txt`. The case definitions of
its scientific drivers informed the case catalog (reuse record REUSE-0008),
and the hydraulic reference planes of its homogeneous 3D seepage driver
define the CASE-015 hydraulic support (REUSE-0010). No MATLAB source is
included.

## slope_stability_petsc4py (earlier implementation)

The earlier petsc4py implementation descends from the MATLAB reference; its
revision `9409b626771171345cd68a2f841a4d24529191bf` carries the same
BSD-2-Clause licence. Its Lagrange node formulas (REUSE-0004) and the
quadrature arrays of its native-engine revision below (REUSE-0005) were
regenerated as data by the code generator.

The 13 meshes in `meshes/` and the case/profile/asset definitions behind
`cases/catalog/` come from the earlier implementation (REUSE-0007). The mesh
files are byte-identical to those published under the BSD-2-Clause licence at
revision `9409b626771171345cd68a2f841a4d24529191bf` of
<https://github.com/Beremi/slope_stability_petsc4py>; see `meshes/README.md`
for their lineage. Several meshes were converted from site data of the MATLAB
reference (Kozinec, Luzec, Franz dam), and
`meshes/3d_hetero_seepage_transition/transition_default.msh` was converted
from a COMSOL-exported mesh.

## yyjson

The native strict-JSON wrapper uses the unmodified upstream `yyjson`
single-file library, version `0.12.0`, revision
`8b4a38dc994a110abaec8a400615567bd996105f`, in `third_party/yyjson/src/`
(REUSE-0003). Its MIT licence is preserved at `licenses/yyjson-MIT.txt`.

## PETSc minimal-symmetric simplex quadrature data

The finite-element registry adapts the degree 14 to 19 compact
minimal-symmetric triangle and tetrahedron orbit tables distributed with
PETSc 3.24.2 (REUSE-0009). The generator expands those orbits
deterministically and maps the biunit-simplex weights to the unit simplex; it
does not copy or call PETSc's runtime quadrature implementation. PETSc's
BSD-2-Clause notice is preserved at `licenses/PETSc-BSD-2-Clause.txt`.

## Linked dependencies

PETSc, MPI, HDF5, HYPRE, METIS/ParMETIS, SuperLU_DIST and BLAS/LAPACK are
linked from the build environment and are not distributed with the source
(`environments/`). The Python dependencies (h5py, meshio, NumPy, Pint,
Pydantic) are installed from their own distributions.
