# Meshes

The 13 source meshes used by the case catalog (`cases/catalog`). Every file is
a Gmsh MSH 4.1 ASCII mesh with straight (G1) simplices: triangles in 2D and
tetrahedra in 3D. Physical names carry the semantic supports used by the
asset documents (`region:*`, `boundary:*`, `nodeset:*`).

The bytes are kept exactly as published by the origin repository, because the
SHA-256 of each file is part of the asset, case and compiled-problem
identities:

- `SHA256SUMS` lists every mesh (`cd meshes && sha256sum -c SHA256SUMS`);
- each asset document records the same digest in `mesh_source.sha256`;
- `cases/catalog/index.json` records it again for every case variant;
- `.gitattributes` marks `meshes/**/*.msh` as `-text -diff`, so Git never
  converts line endings.

`slope-stability verify-meshes` checks all three bindings, and
`slope-stability compile` refuses a mesh whose digest differs.

## Inventory

| Mesh | Bytes | Nodes | Domain cells | Catalog cases |
|---|---:|---:|---:|---|
| `2d_franz_dam/default.msh` | 206,587 | 2,565 | 4,920 triangles | CASE-001 |
| `2d_homo_slope/h0.5.msh` | 141,838 | 2,721 | 5,200 triangles | CASE-002 |
| `2d_homo_slope/h1.0.msh` | 31,937 | 711 | 1,300 triangles | CASE-003 |
| `2d_kozinec/default.msh` | 293,715 | 3,656 | 7,026 triangles | CASE-004, CASE-005 (P2, P4) |
| `2d_luzec/default.msh` | 369,698 | 4,571 | 8,772 triangles | CASE-006 |
| `2d_sloan2013/default.msh` | 321,727 | 4,160 | 7,996 triangles | CASE-007, CASE-022 |
| `3d_hetero_seepage/concave_family_b.msh` | 1,880,288 | 9,365 | 48,205 tetrahedra | CASE-009, CASE-023 |
| `3d_hetero_seepage_transition/transition_default.msh` | 2,279,639 | 10,778 | 55,895 tetrahedra | CASE-010 |
| `3d_hetero_slope/adaptive_family_a_l1.msh` | 789,133 | 3,845 | 18,419 tetrahedra | CASE-011, CASE-012, CASE-013 |
| `3d_hetero_slope/adaptive_family_b_l1.msh` | 1,075,624 | 5,321 | 22,516 tetrahedra | CASE-008 |
| `3d_homo_slope/adaptive_family_a_l1.msh` | 693,950 | 3,625 | 16,615 tetrahedra | CASE-015 to CASE-019 |
| `3d_homo_slope/adaptive_family_b_l1.msh` | 715,069 | 3,769 | 16,813 tetrahedra | CASE-014 |
| `3d_siopt/reference_l0.msh` | 660,412 | 3,431 | 15,356 tetrahedra | CASE-020, CASE-021 |

Total: 9,459,617 bytes. The largest file is
`3d_hetero_seepage_transition/transition_default.msh`.

`3d_homo_slope/adaptive_family_a_l1.msh` carries no tag for the exterior
hydraulic reference planes of the MATLAB homogeneous seepage driver. Its
asset declares the `hydraulic_support_adaptation` extension, and compilation
derives `boundary:homogeneous_hydraulic_reference` from the imported mesh
(`slope_stability.compile.adaptations`; used by CASE-015). The file itself is
not modified.

## Origin and lineage

All 13 files are byte-identical to the files at the same paths under
`meshes/` of the earlier PETSc implementation
[Beremi/slope_stability_petsc4py](https://github.com/Beremi/slope_stability_petsc4py),
at revision `9409b626771171345cd68a2f841a4d24529191bf` (branch `main`) and on
its `release/petsc-ssr-engine` branch. Their Git blob identifiers are:

| Mesh | Git blob |
|---|---|
| `2d_franz_dam/default.msh` | `e304f97031c3835a98557645f71bbb239fbaab2b` |
| `2d_homo_slope/h0.5.msh` | `3620a58543e7f37b76adb3a36436769625a842e3` |
| `2d_homo_slope/h1.0.msh` | `d215f227cbd9e5ea84a77a2f36012af146e1f20c` |
| `2d_kozinec/default.msh` | `bc6c5b98ef072bc9ca546a6527c78eba077ebbff` |
| `2d_luzec/default.msh` | `c8de01cfd03529259e896dde2af1fa6753fbb635` |
| `2d_sloan2013/default.msh` | `f16509a094921eaaae31750ef566bbe55a777223` |
| `3d_hetero_seepage/concave_family_b.msh` | `92c4efb17dbf6af312367d63b0658b54dbec508f` |
| `3d_hetero_seepage_transition/transition_default.msh` | `24c2a20bda9d167288ee07c8524c2f890578e934` |
| `3d_hetero_slope/adaptive_family_a_l1.msh` | `4b55643f81a5a2d5ca9191b912fcafeebb24808b` |
| `3d_hetero_slope/adaptive_family_b_l1.msh` | `ee65b17e0717d5b5b2c8436cdd9ac7922b1cefbb` |
| `3d_homo_slope/adaptive_family_a_l1.msh` | `879c8167472117bf23a9308c9490702b0c38a569` |
| `3d_homo_slope/adaptive_family_b_l1.msh` | `d369811d8ebf87647db814c6408e72c990a34568` |
| `3d_siopt/reference_l0.msh` | `50c8548f3e5e853bb071f7959559e19de04dae58` |

That repository created them with deterministic converters (its
`meshes/converter-index.md`):

- `2d_homo_slope` and `2d_sloan2013` were regenerated from generated
  geometry recipes (the homogeneous 2D slope definition and the Sloan 2013
  seepage mesh).
- `2d_kozinec`, `2d_luzec` and `2d_franz_dam` were converted from MATLAB-style
  text mesh bundles of the original MATLAB implementation.
- `3d_homo_slope`, `3d_hetero_slope`, `3d_hetero_seepage` and `3d_siopt`
  were retagged from earlier tagged Gmsh meshes: numeric physical groups were
  renamed to logical regions and supports (homogeneous regions collapsed to
  one), and the seepage meshes gained canonical node-set supports.
- `3d_hetero_seepage_transition/transition_default.msh` was converted from a
  **COMSOL-exported** transition mesh: the COMSOL `.msh` was retagged and the
  hydraulic node sets were recreated from the earlier transition recipe.

## License

The meshes are distributed under the BSD 2-Clause License of the project.
The identical files are published under the BSD 2-Clause License in
[`Beremi/slope_stability_petsc4py`](https://github.com/Beremi/slope_stability_petsc4py)
at revision `9409b626771171345cd68a2f841a4d24529191bf`, whose licence is that
of the MATLAB origin `sysala/slope_stability`
(`licenses/sysala-slope-stability-BSD-2-Clause.txt`). The site meshes
(Kozinec, Luzec, Franz dam) and the COMSOL-derived transition mesh keep the
lineage described above; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
