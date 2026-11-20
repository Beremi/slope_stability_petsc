# Catalog cases

`cases/catalog/` defines 23 reference cases, `CASE-001` to `CASE-023`, in 24
variants: `CASE-005` has a P2 and a P4 variant, every other case one
`canonical` variant. Cases 1–21 correspond to the benchmark cases of the
earlier PETSc implementation
([Beremi/slope_stability_petsc4py](https://github.com/Beremi/slope_stability_petsc4py)),
and cases 22 and 23 to coupled workflows that exist only in the MATLAB
implementation
([sysala/slope_stability](https://github.com/sysala/slope_stability)). Each
case document records its origin in `provenance.origin`.

The cases are reference inputs for regression and performance work. This
repository does not claim that their results are validated against
independent reference solutions.

## Cases

All cases use straight (order 1) geometry, Mohr–Coulomb materials use the
Davis B variant, and the MATLAB driver names refer to
`sysala/slope_stability`. "Coupled" means one-way seepage coupling; "LL" is
limit load and "SSR" strength reduction with indirect continuation.

| Case | Description | Analysis | Physics | Order | Mesh | MATLAB driver |
|---|---|---|---|---|---|---|
| CASE-001 | Franz dam | SSR | Coupled: gradient and weight | P2 | `2d_franz_dam/default` | `slope_stability_2D_Franz_dam_SSR.m` |
| CASE-002 | Homogeneous 2D slope | LL | Mechanics | P2 | `2d_homo_slope/h0.5` | `slope_stability_2D_homo_LL.m` |
| CASE-003 | Homogeneous 2D slope | SSR | Mechanics | P2 | `2d_homo_slope/h1.0` | `slope_stability_2D_homo_SSR.m` |
| CASE-004 | Kozinec site | LL | Mechanics, prescribed saturation | P2 | `2d_kozinec/default` | `slope_stability_2D_Kozinec_LL.m` |
| CASE-005 | Kozinec site | SSR | Mechanics, prescribed saturation | P2, P4 | `2d_kozinec/default` | `slope_stability_2D_Kozinec_SSR.m` |
| CASE-006 | Luzec site | SSR | Coupled: gradient and weight | P2 | `2d_luzec/default` | `slope_stability_2D_Luzec_SSR.m` |
| CASE-007 | Sloan 2013 seepage | Seepage | Seepage | P1 | `2d_sloan2013/default` | `slope_stability_2D_Sloan2013_SSR.m` |
| CASE-008 | Heterogeneous 3D slope | LL | Mechanics | P2 | `3d_hetero_slope/adaptive_family_b_l1` | `slope_stability_3D_hetero_LL.m` |
| CASE-009 | Heterogeneous 3D seepage | Seepage | Seepage | P2 | `3d_hetero_seepage/concave_family_b` | `slope_stability_3D_hetero_seepage_SSR.m` |
| CASE-010 | Heterogeneous 3D slope, COMSOL transition mesh | SSR | Coupled: weight only | P2 | `3d_hetero_seepage_transition/transition_default` | `slope_stability_3D_hetero_seepage_SSR_comsol.m` |
| CASE-011 | Heterogeneous 3D slope | SSR | Mechanics | P2 | `3d_hetero_slope/adaptive_family_a_l1` | `slope_stability_3D_hetero_SSR.m` |
| CASE-012 | Heterogeneous 3D slope | SSR | Mechanics | P4 | `3d_hetero_slope/adaptive_family_a_l1` | `slope_stability_3D_hetero_SSR.m` |
| CASE-013 | As CASE-012 with roller supports | SSR | Mechanics | P4 | `3d_hetero_slope/adaptive_family_a_l1` | `slope_stability_3D_hetero_SSR.m` |
| CASE-014 | Homogeneous 3D slope | LL | Mechanics | P2 | `3d_homo_slope/adaptive_family_b_l1` | `slope_stability_3D_homo_LL.m` |
| CASE-015 | Homogeneous 3D slope with seepage | SSR | Coupled: gradient and weight | P2 | `3d_homo_slope/adaptive_family_a_l1` | `slope_stability_3D_homo_seepage_SSR.m` |
| CASE-016 | Homogeneous 3D slope | SSR | Mechanics | P2 | `3d_homo_slope/adaptive_family_a_l1` | `slope_stability_3D_homo_SSR.m` |
| CASE-017 | Homogeneous 3D slope, fixed base | SSR | Mechanics | P2 | `3d_homo_slope/adaptive_family_a_l1` | `slope_stability_3D_homo_SSR.m` |
| CASE-018 | As CASE-017 with a roller base | SSR | Mechanics | P2 | `3d_homo_slope/adaptive_family_a_l1` | `slope_stability_3D_homo_SSR.m` |
| CASE-019 | Homogeneous 3D slope | SSR | Mechanics | P4 | `3d_homo_slope/adaptive_family_a_l1` | `slope_stability_3D_homo_SSR.m` |
| CASE-020 | SIOPT 3D slope | LL | Mechanics | P2 | `3d_siopt/reference_l0` | `SIOPT_LL.m` |
| CASE-021 | SIOPT 3D slope | SSR | Mechanics | P2 | `3d_siopt/reference_l0` | `SIOPT_SSR.m` |
| CASE-022 | Sloan 2013 slope, coupled | SSR | Coupled: gradient and weight | P1 | `2d_sloan2013/default` | `slope_stability_2D_Sloan2013_SSR.m` |
| CASE-023 | Heterogeneous 3D slope with seepage | SSR | Coupled: gradient and weight | P2 | `3d_hetero_seepage/concave_family_b` | `slope_stability_3D_hetero_seepage_SSR.m` |

Related cases differ as follows:

- `CASE-016` and `CASE-017` have the same boundary conditions but different
  work horizons and Newton iteration limits.
- `CASE-018` replaces the fixed base of `CASE-017` by rollers (vertical
  displacement fixed only) and uses the `matlab-dfgmres` PMG and deflation
  profiles.
- `CASE-013` replaces the supports of `CASE-012` by roller supports and has a
  larger work horizon.
- `CASE-015` uses a hydraulic boundary support that the mesh does not tag;
  compilation derives it ([meshes/README.md](../meshes/README.md)).

[meshes/README.md](../meshes/README.md) lists the size of each mesh, and
`compile-report.json` of each compiled package gives its entity counts.

## Running the catalog

```sh
slope-stability compile --destination $WORK/compiled-all
slope-stability run --prefix $PREFIX --runtime-library-path $PETSC_LIB \
    --compiled $WORK/compiled-all --case-id CASE-001 --case-id CASE-007 \
    --layout 4x8 --destination $WORK/run-selected
```

Without `--case-id`, `run` executes all 24 variants in sequence. Compiling
all variants took about 90 s on the workstation described in
[parallel-execution.md](parallel-execution.md). At `4x8` on the same machine,
`CASE-003` ran in about 3 s, `CASE-007` in 5 s and `CASE-001` in 53 s; the 3D
P4 cases take much longer.

## Known limitations

- During development, `CASE-019` (3D, P4) did not complete with its catalog
  policies at `4x8` on the 32-core workstation described in
  [parallel-execution.md](parallel-execution.md): its mechanics operator
  workspace exceeded the earlier fixed 1024 MiB admission limit
  (`--operator-workspace-mib`), and a run with a raised limit rejected its
  step at load factor 1.2 after reaching the Newton iteration limit.
- Most catalog profiles were carried over from the earlier implementations
  and are not tuned for this solver.
