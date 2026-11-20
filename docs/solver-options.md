# Solver settings

<!-- Generated from the native solver option registry. -->

Stable settings are typed fields in a resolved run. Profile settings can
also be overridden during resolution with `--set SECTION.FIELD=JSON`.
The `outputs` section is authored in the `output` profile. Expert flags
are read from PETSc's options database by the native engine.

| Setting | Status | Type | Default | Allowed | Effect |
| --- | --- | --- | --- | --- | --- |
| `pmg.mechanics_matrix_free` | stable | boolean | `false` | `false`, `true` | Apply high-order mechanics tangents without assembling the full matrix. |
| `pmg.matrix_free_galerkin` | stable | boolean | `true` | `false`, `true` | Use Galerkin shell actions above P1 when mechanics is matrix-free. |
| `pmg.matrix_free_block_nodes` | stable | integer | `1` | 1 to 35, inclusive | Maximum entity nodes per matrix-free fine-level block. |
| `pmg.matrix_free_batch_cells` | stable | integer | `0` | 0 to 32, inclusive | Matrix-free fine-level action batch size. A value of 0 selects the order-specific kernel default. |
| `pmg.fgmres_full_iterate_post` | stable | boolean | `false` | `false`, `true` | Apply post-FGMRES to the full iterate, starting from the coarse-corrected state. |
| `pmg.coarse_factorization` | stable | string | `petsc_default` | `petsc_default`, `superlu_dist_reuse` | Configure the distributed SuperLU_DIST base factorization when selected. |
| `pmg.coarse_lu_fallback_to_gamg` | stable | boolean | `false` | `false`, `true` | Use FGMRES/GAMG when distributed coarse LU is requested and the PETSc build lacks SuperLU_DIST. |
| `pmg.shared_point_balance` | stable | string | `petsc` | `petsc`, `stable_hash_p2`, `stable_hash_high_order` | Choose how shared vertices, edges and high-order facets are owned after partitioning. |
| `pmg.parmetis_seed` | stable | integer | `-1` | -1 to 2147483647, inclusive | Select ParMETIS with this seed. A value of -1 leaves PETSc partitioner selection in effect. |
| `linear.initial_guess` | stable | string | `coarse_only` | `coarse_only`, `previous_complement`, `previous_complement_if_better` | Choose the mechanics FGMRES initial complementary correction. |
| `linear.orthogonalization` | stable | string | `petsc` | `petsc`, `cgs_refine_always`, `cgs_unrefined`, `mgs` | Choose outer mechanics FGMRES orthogonalization and CGS refinement. |
| `linear.use_mdot_gemv` | stable | boolean | `false` | `false`, `true` | Request PETSc's GEMV implementation of batched vector dot products. |
| `nonlinear.allow_relaxed` | stable | boolean | `false` | `false`, `true` | Allow explicitly labelled relaxed Newton acceptance. |
| `nonlinear.maximum_line_search_trials` | stable | integer | `24` | 1 to 64, inclusive | Line-search trial budget per Newton correction. Directional search allows up to 62 and also evaluates two endpoints. |
| `nonlinear.fixed_line_search` | stable | string | `directional` | `directional`, `residual_monotone` | Choose the line search for fixed-factor mechanics Newton solves. |
| `nonlinear.indirect_legacy_alg5_line_search` | stable | boolean | `false` | `false`, `true` | Use the reference directional ALG5 line search during indirect SSR. |
| `nonlinear.indirect_sensitivity` | stable | string | `reference_forward` | `reference_forward`, `stable_central` | Choose the finite-difference sensitivity in work-controlled SSR. |
| `nonlinear.indirect_sensitivity_step` | stable | number | `0.0` | 0.0 to 0.1, inclusive | Relative central-difference step. A value of 0 selects the cube root of machine epsilon. |
| `nonlinear.indirect_progress_guard` | stable | string | `off` | `off`, `fixed`, `adaptive`, `observe` | Disable, enforce, learn, or observe the indirect Newton progress guard. |
| `nonlinear.indirect_progress_guard_iteration` | stable | integer | `4` | 0 to 255, inclusive | Zero-based Newton correction at which an enabled progress guard samples. |
| `nonlinear.indirect_progress_guard_maximum` | stable | number | `0.0` | 0.0 to 1.7976931348623157e+308, inclusive | Positive relative residual ceiling for the fixed progress guard. |
| `nonlinear.indirect_adaptive_linear_tolerance` | stable | boolean | `false` | `false`, `true` | Tighten subsequent indirect linear solves when Newton damping triggers the forcing policy. |
| `nonlinear.indirect_adaptive_linear_iteration_threshold` | stable | integer | `2` | 0 to 2147483647, inclusive | Maximum previous inner iteration count that permits tolerance tightening. |
| `nonlinear.indirect_adaptive_linear_alpha_threshold` | stable | number | `0.1` | 2.2250738585072014e-308 to 1.0, inclusive | Line-search alpha below which the adaptive linear tolerance may tighten. |
| `nonlinear.indirect_adaptive_linear_tightening_factor` | stable | number | `0.5` | 2.2250738585072014e-308 to 0.9999999999999999, inclusive | Multiplier for tightening a linear relative tolerance. |
| `nonlinear.indirect_adaptive_linear_minimum_relative_tolerance` | stable | number | `0.0001` | 2.2250738585072014e-308 to 1.0, inclusive | Floor for the indirect adaptive linear relative tolerance. |
| `continuation.consecutive_retry_limit` | stable | integer | `5` | 1 to 5, inclusive | Maximum consecutive rejected attempts before continuation stops. |
| `continuation.work_step_controller` | stable | string | `classic` | `classic`, `predictor_error`, `work_efficiency` | Select the indirect work-step growth and retry controller. |
| `continuation.work_growth_iteration_limit` | stable | integer | `0` | 0 to 2147483647, inclusive | Newton effort gate. A value of 0 leaves classic unrestricted and derives five sixths of the Newton limit for adaptive modes. |
| `continuation.work_predictor_error_target` | stable | number | `1.0` | 2.2250738585072014e-308 to 1.7976931348623157e+308, inclusive | Dimensionless predictor error target for adaptive work stepping. |
| `continuation.work_step_scale_minimum` | stable | number | `0.5` | 2.2250738585072014e-308 to 1.0, inclusive | Lower multiplicative bound on adaptive work-step updates. |
| `continuation.work_step_scale_maximum` | stable | number | `2.0` | 1.0 to 1.7976931348623157e+308, inclusive | Upper multiplicative bound on adaptive work-step updates. |
| `resources.operator_workspace_mib` | stable | integer | `1024` | 1 to 2147483647, inclusive | Rank-local mechanics operator workspace cap in MiB. |
| `resources.mechanics_pmg_memory_cap_mib` | stable | integer | `2048` | 1 to 2147483647, inclusive | Aggregate mechanics PMG storage cap in MiB. |
| `resources.hydraulic_pmg_memory_cap_mib` | stable | integer | `2048` | 1 to 2147483647, inclusive | Aggregate seepage PMG storage cap in MiB. |
| `resources.linear_vector_memory_cap_mib` | stable | integer | `1024` | 1 to 2147483647, inclusive | Aggregate Krylov and deflation vector cap in MiB for either field. |
| `outputs.krylov_history_limit` | stable | integer | `0` | 0 to 1000000, inclusive | Maximum detailed Krylov residual rows per run. A value of 0 disables the detail. |
| `outputs.pmg_apply_history_limit` | stable | integer | `0` | 0 to 1000000, inclusive | Maximum detailed PMG application rows per run. A value of 0 disables the detail. |
| `validation.qualification_mode` | stable | boolean | `false` | `false`, `true` | Use fixed-order assembly and operator actions for mechanics and its PMG levels. |
| `-native_memory_view` | expert | boolean | `false` | `false`, `true` | Print rank-local resident and allocator memory snapshots at graph phases. |
| `-native_test_signal_after_accepted` | expert | integer | `0` | 0 to 2147483647, inclusive | Test-only signal injection on rank zero after this accepted-state count. A value of 0 disables it. |

## Memory admission

The local launcher resolves the aggregate linear-vector cap to one
quarter of the lower of available physical memory and the effective
cgroup v2 memory headroom. With a per-rank address-space limit, it also
bounds the cap by one quarter of that limit times the number of ranks.
For an explicit multi-node resource row, `memory_per_node_mib` describes
the smallest admitted node allocation and is multiplied by the node
count. Resolution records these inputs and the selected cap in
`resources`. A caller may explicitly set
`resources.linear_vector_memory_cap_mib` from a known allocation.
If memory information is unavailable the fallback is 1,024 MiB.

The engine estimates the actual configured Krylov basis, optional
correction capture and deflation store from the global number of free
rows before creating assembly plans or running a continuation attempt.
The estimate covers vector payload. PMG storage, operator workspace and
per-rank dense scratch have separate costs.

## Qualification and continuation limits

`validation.qualification_mode` selects fixed-order mechanics and PMG
assembly. Its per-row summation order is independent of OpenMP thread
count at a fixed MPI layout. MPI communication and reductions can
produce different floating-point results. The result records the
effective mode as `summary.mechanics_assembly_mode`.

SSR supports up to 128 accepted states. Limit load
supports 127 continuation steps and reserves
one state for its elastic reference. The resolver and engine reject
values outside these bounds.

## PETSc backends and iteration detail

`superlu_dist_reuse` selects SamePattern_SameRowPerm, NOROWPERM,
METIS_AT_PLUS_A and equilibration for the distributed coarse LU.
It disables tiny-pivot replacement, refinement, parallel symbolic
factorization, the 3D grid, GPU offload and solver statistics.
Single-rank coarse levels use the PETSc sequential LU. PETSc backend
options such as KSP monitors remain available under the outer and
level prefixes.

Krylov and PMG detail limits count rows across the run. These optional
tables use the same relational history columns as the solve summaries.
`reference` is the Newton ordinal and `auxiliary` joins rows from the
same linear solve. The journal's `linear_start` event gives the same
sequence. `linear_end` reports the original-system verification and
the number of additional cancellation reductions. `krylov` records
PETSc's reported norm at each iteration.
`pmg_apply` records the cycle ordinal, level solve count in `control`,
rank-zero inner iterations, final local residual and reason, bridge
bytes in `tolerance` and operator applications in `step`.
