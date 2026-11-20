/** @file
 * @brief Polynomial multigrid (PMG) over the P1/P2/P4(/P8) hierarchy.
 *
 * Levels are ordered coarse to fine. Each level may run on a subset of the
 * ranks (active ranks), and coarse operators are Galerkin products. Functions
 * that take a hierarchy are collective on its communicator unless marked
 * local, and are called from the master thread.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_PMG_PMG_H
#define SS_PMG_PMG_H

#include "slope_internal/core/status.h"

#include <petscksp.h>

#include <stddef.h>
#include <stdint.h>

/** Largest number of PMG levels (orders 1, 2, 4 and 8). */
#define SS_PMG_MAX_LEVELS 4U
/** Capacity of a level options prefix. */
#define SS_PMG_PREFIX_CAPACITY 32U

/** Field of a PMG hierarchy, selecting its near-nullspace. */
typedef enum {
  SS_PMG_FIELD_NONE = 0,      /**< No near-nullspace. */
  SS_PMG_FIELD_MECHANICS = 1, /**< Rigid-body modes of elasticity. */
  SS_PMG_FIELD_HYDRAULIC = 2  /**< Constant mode of a pure-Neumann scalar field. */
} SsPmgFieldKind;

/** Smoother or solver of one PMG level. */
typedef enum {
  SS_PMG_LEVEL_RICHARDSON_NONE = 1,      /**< Unpreconditioned Richardson. */
  SS_PMG_LEVEL_RICHARDSON_JACOBI = 2,    /**< Richardson with Jacobi. */
  SS_PMG_LEVEL_CHEBYSHEV_JACOBI = 3,     /**< Chebyshev with Jacobi. */
  SS_PMG_LEVEL_FGMRES_JACOBI = 4,        /**< FGMRES with Jacobi. */
  SS_PMG_LEVEL_FGMRES_GAMG = 5,          /**< FGMRES with GAMG (coarse level). */
  SS_PMG_LEVEL_PREONLY_LU = 6,           /**< Direct LU (coarse level). */
  SS_PMG_LEVEL_PREONLY_REDUNDANT_LU = 7, /**< Redundant LU on each active rank. */
  SS_PMG_LEVEL_PREONLY_SUPERLU_DIST = 8, /**< Distributed SuperLU_DIST LU. */
  /** Richardson with variable point-block Jacobi; the block inverses come from
   * `MATOP_INVERT_VBLOCK_DIAGONAL` of the operator shell. */
  SS_PMG_LEVEL_RICHARDSON_VPBJACOBI = 9,
  SS_PMG_LEVEL_CHEBYSHEV_POINT_JACOBI = 10, /**< Chebyshev with point Jacobi. */
  SS_PMG_LEVEL_FGMRES_VPBJACOBI = 11 /**< FGMRES with variable point-block Jacobi. */
} SsPmgLevelMethod;

/** Stage of a V-cycle at which a failure occurred or is injected. */
typedef enum {
  SS_PMG_FAIL_NONE = 0,               /**< No failure. */
  SS_PMG_FAIL_PRE_SMOOTH = 1,         /**< Pre-smoothing. */
  SS_PMG_FAIL_RESTRICT = 2,           /**< Restriction. */
  SS_PMG_FAIL_BRIDGE_TO_ACTIVE = 3,   /**< Transfer to the active ranks. */
  SS_PMG_FAIL_LEVEL_SOLVE = 4,        /**< Coarse solve. */
  SS_PMG_FAIL_BRIDGE_FROM_ACTIVE = 5, /**< Transfer back from the active ranks. */
  SS_PMG_FAIL_PROLONG = 6,            /**< Prolongation. */
  SS_PMG_FAIL_POST_SMOOTH = 7         /**< Post-smoothing. */
} SsPmgFailurePoint;

/** Polynomial orders of the levels, coarse to fine. */
typedef struct {
  uint32_t fine_order;                /**< Order of the finest level. */
  size_t level_count;                 /**< Number of levels. */
  uint32_t orders[SS_PMG_MAX_LEVELS]; /**< Order of each level. */
} SsPmgHierarchy;

/** Solver policy of one level. */
typedef struct {
  char options_prefix[SS_PMG_PREFIX_CAPACITY]; /**< PETSc options prefix. */
  SsPmgLevelMethod method;                     /**< Smoother or solver. */
  size_t active_rank_count;        /**< Ranks that own the level; 0 for all. */
  const PetscMPIInt *active_ranks; /**< Active world ranks, borrowed. */
  PetscInt pre_smooth_steps;       /**< Pre-smoothing steps. */
  PetscInt post_smooth_steps;      /**< Post-smoothing steps. */
  PetscInt maximum_iterations;     /**< Iteration limit of a level solve. */
  PetscReal richardson_scale;      /**< Richardson damping. */
  PetscReal rtol;                  /**< Relative tolerance of a level solve. */
  PetscReal atol;                  /**< Absolute tolerance of a level solve. */
  PetscInt redundant_solve_count;  /**< Redundant solve groups. */
} SsPmgLevelPolicy;

/** Data of the near-nullspace of one level. */
typedef struct {
  SsPmgFieldKind field;         /**< Field kind. */
  uint32_t dimension;           /**< Spatial dimension. */
  int hydraulic_pure_neumann;   /**< Nonzero when no pressure is prescribed. */
  size_t owned_node_count;      /**< Owned nodes of this rank. */
  const PetscReal *coordinates; /**< Node-major coordinates of the owned nodes. */
  /** Owned global row of each node component, node-major; negative for a
   * constrained component. */
  const PetscInt *global_dof_indices;
  PetscReal dependence_tolerance; /**< Tolerance for dropping dependent modes. */
} SsPmgNearNullspaceInput;

/** Configuration of a PMG hierarchy; arrays are borrowed until ss_pmg_create()
 * returns. */
typedef struct {
  uint32_t fine_order;    /**< Order of the finest level. */
  size_t level_count;     /**< Number of levels. */
  const uint32_t *orders; /**< Order of each level, coarse to fine. */
  /** `level_count - 1` prolongations, coarse to fine; retained. */
  const Mat *prolongations;
  const SsPmgLevelPolicy *level_policies;         /**< Policy of each level. */
  const SsPmgNearNullspaceInput *near_nullspaces; /**< Data of each level. */
  uint64_t transfer_generation; /**< Nonzero generation of the prolongations. */
  /** Value generations between numeric Galerkin refreshes; 1 refreshes on
   * every new tangent. */
  uint64_t numeric_lag_interval;
  /** Use owner-ordered PetscSF assembly for MPIAIJ Galerkin products. */
  int ordered_galerkin;
  /** Use the full coarse-corrected iterate in post FGMRES. */
  int fgmres_full_iterate_post;
  size_t memory_cap_bytes;            /**< Admission cap of the hierarchy. */
  SsPmgFailurePoint injected_failure; /**< Test hook: failure to inject. */
  size_t injected_failure_level;      /**< Test hook: level of the failure. */
} SsPmgSpecification;

/** Outcome of a near-nullspace construction. */
typedef struct {
  SsPmgFieldKind field;           /**< Field kind. */
  size_t raw_mode_count;          /**< Modes before orthogonalization. */
  size_t retained_mode_count;     /**< Modes retained. */
  size_t dropped_mode_count;      /**< Dependent modes dropped. */
  PetscReal orthogonality_defect; /**< Largest deviation of the Gram matrix from I. */
} SsPmgNullspaceRecord;

/** Setup outcome of one level. */
typedef struct {
  uint32_t order;                 /**< Polynomial order. */
  PetscInt global_dofs;           /**< Global unknowns. */
  SsPmgLevelMethod method;        /**< Smoother or solver. */
  PetscInt redundant_solve_count; /**< Redundant solve groups. */
  size_t active_rank_count;       /**< Ranks that own the level. */
  int local_rank_active;          /**< Nonzero when this rank is active. */
  int local_owns_ksp;             /**< Nonzero when this rank owns the solver. */
  PetscMPIInt local_active_rank;  /**< Rank in the active communicator, or -1. */
  /** Nonzero when a refactorization reuses the row permutation. */
  int repeated_factorization_reuses_row_permutation;
  size_t variable_point_block_count;    /**< Point blocks of VPBJacobi. */
  size_t retained_near_nullspace_modes; /**< Near-nullspace modes kept. */
  size_t dropped_near_nullspace_modes;  /**< Near-nullspace modes dropped. */
  uint64_t solver_setup_count;          /**< Level solver setups so far. */
} SsPmgLevelSetupRecord;

/** Setup outcome of the hierarchy. */
typedef struct {
  SsPmgHierarchy hierarchy;                 /**< Level orders. */
  uint64_t transfer_generation;             /**< Generation of the prolongations. */
  uint64_t structure_generation;            /**< Generation of the operator pattern. */
  uint64_t requested_value_generation;      /**< Requested operator values. */
  uint64_t preconditioner_value_generation; /**< Values the preconditioner uses. */
  uint64_t hierarchy_build_count;           /**< Hierarchy builds so far. */
  uint64_t transfer_setup_count;            /**< Transfer setups so far. */
  uint64_t galerkin_symbolic_count;         /**< Symbolic Galerkin products so far. */
  uint64_t galerkin_numeric_count;          /**< Numeric Galerkin products so far. */
  uint64_t level_solver_setup_count;        /**< Level solver setups so far. */
  int hierarchy_reused;         /**< Nonzero when the hierarchy was reused. */
  int transfer_reused;          /**< Nonzero when transfers were reused. */
  int galerkin_pattern_reused;  /**< Nonzero when the pattern was reused. */
  int numeric_refreshed;        /**< Nonzero when values were refreshed. */
  int numeric_lagged;           /**< Nonzero when the refresh was lagged. */
  size_t matrix_object_count;   /**< Live PETSc matrices. */
  size_t vector_object_count;   /**< Live PETSc vectors. */
  size_t admitted_memory_bytes; /**< Admitted hierarchy memory. */
  SsPmgLevelSetupRecord levels[SS_PMG_MAX_LEVELS]; /**< Per-level outcome. */
} SsPmgSetupRecord;

/** Outcome of one V-cycle. */
typedef struct {
  SsStatus status;                      /**< Status of the cycle. */
  size_t failed_level;                  /**< Level of a failure. */
  SsPmgFailurePoint failed_point;       /**< Stage of a failure. */
  uint64_t sequence;                    /**< Cycle number. */
  uint64_t bridge_calls;                /**< Transfers between active rank sets. */
  uint64_t bridge_bytes;                /**< Bytes moved by those transfers. */
  uint64_t level_solve_calls;           /**< Level solves. */
  uint64_t operator_apply_calls;        /**< Operator applications. */
  KSPConvergedReason last_level_reason; /**< Reason of the last level solve. */
  PetscInt last_level_iterations;       /**< Iterations of the last level solve. */
  /** Total local pre/base/post iterations at each level in this V-cycle. */
  uint64_t level_iterations[SS_PMG_MAX_LEVELS];
  PetscReal last_level_residual_norm; /**< Residual of the last level solve. */
  PetscReal input_norm;               /**< Norm of the input (checked apply). */
  PetscReal output_norm;              /**< Norm of the output (checked apply). */
} SsPmgApplyRecord;

/** Cumulative counters of a hierarchy on this rank. */
typedef struct {
  uint64_t setup_calls;                /**< Setup calls. */
  uint64_t apply_calls;                /**< Apply calls of all kinds. */
  uint64_t contract_apply_calls;       /**< ss_pmg_apply() calls. */
  uint64_t preconditioner_apply_calls; /**< Preconditioner applications. */
  uint64_t norm_validation_reductions; /**< Reductions for norm checks. */
  uint64_t level_status_reductions;    /**< Reductions for status agreement. */
  uint64_t failed_apply_calls;         /**< Failed applications. */
  uint64_t bridge_calls;               /**< Transfers between rank sets. */
  uint64_t bridge_bytes;               /**< Bytes moved by those transfers. */
  uint64_t level_solve_calls;          /**< Level solves. */
  uint64_t operator_apply_calls;       /**< Operator applications. */
  size_t current_matrix_objects;       /**< Live PETSc matrices. */
  size_t current_vector_objects;       /**< Live PETSc vectors. */
  size_t peak_matrix_objects;          /**< Largest number of matrices. */
  size_t peak_vector_objects;          /**< Largest number of vectors. */
  size_t admitted_memory_bytes;        /**< Admitted hierarchy memory. */
} SsPmgDiagnostics;

/** Opaque PMG hierarchy; see ss_pmg_create(). */
typedef struct SsPmg SsPmg;

/** Observe one local V-cycle on the calling rank after all transfers finish.
 * @param context Observer context.
 * @param record Cycle outcome. */
typedef void (*SsPmgApplyObserver)(void *context, const SsPmgApplyRecord *record);

/** Set or clear the optional noncollective apply observer.
 * @param[in,out] pmg Hierarchy.
 * @param[in] observe Observer, or NULL to clear it.
 * @param[in] context Context passed to @p observe. */
void ss_pmg_set_apply_observer(SsPmg *pmg, SsPmgApplyObserver observe, void *context);

/** Level orders of a fine order: 1; 1, 2; 1, 2, 4; or 1, 2, 4, 8.
 * @param[in] fine_order Order of the finest level.
 * @param[out] hierarchy Level orders.
 * @return ::SS_OK, ::SS_ERROR_NOT_FOUND for an unsupported order, or
 *   ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_pmg_hierarchy_resolve(uint32_t fine_order, SsPmgHierarchy *hierarchy);
/** Check requested level orders against the supported hierarchy.
 * @param[in] fine_order Order of the finest level.
 * @param[in] level_count Number of requested levels.
 * @param[in] orders Requested orders, coarse to fine.
 * @param[out] hierarchy Validated hierarchy.
 * @return ::SS_OK, ::SS_ERROR_DUPLICATE, ::SS_ERROR_INTEGRITY for a mismatch,
 *   ::SS_ERROR_NOT_FOUND or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_pmg_hierarchy_validate(uint32_t fine_order, size_t level_count,
                                   const uint32_t *orders, SsPmgHierarchy *hierarchy);

/** Spread a number of active ranks evenly over the world ranks.
 * @param[in] world_size Number of world ranks.
 * @param[in] active_count Number of active ranks, 1 to @p world_size.
 * @param[out] active_ranks `floor(i * world_size / active_count)` for each i.
 * @param[in] capacity Capacity of @p active_ranks.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_pmg_active_ranks_resolve(PetscMPIInt world_size, PetscMPIInt active_count,
                                     PetscMPIInt *active_ranks, size_t capacity);
/** Check a set of active ranks.
 * @param[in] world_size Number of world ranks.
 * @param[in] active_count Number of active ranks.
 * @param[in] active_ranks Active ranks.
 * @return ::SS_OK, ::SS_ERROR_DUPLICATE or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_pmg_active_ranks_validate(PetscMPIInt world_size, size_t active_count,
                                      const PetscMPIInt *active_ranks);

/** Build the orthonormalized near-nullspace of a level operator; collective on
 * the operator's communicator.
 * @param[in] operator_handle Level operator (vector layout).
 * @param[in] input Field kind and node data.
 * @param[out] near_nullspace New PETSc null space, owned by the caller.
 * @param[out] record Mode counts.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_pmg_near_nullspace_create(Mat operator_handle,
                                      const SsPmgNearNullspaceInput *input,
                                      MatNullSpace *near_nullspace,
                                      SsPmgNullspaceRecord *record);

/** Create a hierarchy and its active-rank communicators; collective.
 * @param[in] communicator World communicator of the fine level; duplicated.
 * @param[in] specification Configuration.
 * @param[out] result New hierarchy, released with ss_pmg_destroy().
 * @return ::SS_OK or the agreed failure status (::SS_ERROR_INVALID_ARGUMENT,
 *   ::SS_ERROR_INTEGRITY, ::SS_ERROR_OUT_OF_MEMORY, ::SS_ERROR_INTERNAL). */
SsStatus ss_pmg_create(MPI_Comm communicator, const SsPmgSpecification *specification,
                       SsPmg **result);
/** Retain caller-updated level operators, coarse to fine, before setup;
 * collective.
 * @param[in,out] pmg Hierarchy.
 * @param[in] level_count Number of operators.
 * @param[in] operators Operators; a reference to each is retained.
 * @return ::SS_OK, ::SS_ERROR_TYPE_MISMATCH, ::SS_ERROR_INVALID_ARGUMENT or
 *   ::SS_ERROR_INTERNAL. */
SsStatus ss_pmg_set_level_operators(SsPmg *pmg, size_t level_count,
                                    const Mat *operators);
/** Set up or refresh the hierarchy for a fine operator; collective.
 *
 * A new structure generation rebuilds the Galerkin patterns; a new value
 * generation refreshes the numeric products unless the lag interval allows
 * reuse.
 *
 * @param[in,out] pmg Hierarchy.
 * @param[in] fine_operator Fine-level operator.
 * @param[in] structure_generation Generation of its pattern.
 * @param[in] value_generation Generation of its values.
 * @param[out] record Setup outcome.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_pmg_setup(SsPmg *pmg, Mat fine_operator, uint64_t structure_generation,
                      uint64_t value_generation, SsPmgSetupRecord *record);
/** Apply one V-cycle with collective status agreement and norm checks.
 * @param[in,out] pmg Set-up hierarchy.
 * @param[in] rhs Fine-level residual.
 * @param[out] correction Fine-level correction; flagged on failure.
 * @param[out] record Cycle outcome, identical on all ranks after a failure.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_pmg_apply(SsPmg *pmg, Vec rhs, Vec correction, SsPmgApplyRecord *record);
/** Apply one V-cycle with collective status agreement, without norm checks.
 * @param[in,out] pmg Set-up hierarchy.
 * @param[in] rhs Fine-level residual.
 * @param[out] correction Fine-level correction; flagged on failure.
 * @param[out] record Cycle outcome.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_pmg_apply_preconditioner(SsPmg *pmg, Vec rhs, Vec correction,
                                     SsPmgApplyRecord *record);
/** Apply one V-cycle in a PETSc preconditioner.
 *
 * It always runs the full collective schedule and returns the rank-local
 * status, so no extra reduction is spent. The caller must flag its final
 * output and report the failure to its enclosing KSP.
 *
 * @param[in,out] pmg Set-up hierarchy.
 * @param[in] rhs Fine-level residual.
 * @param[out] correction Fine-level correction.
 * @return ::SS_OK or this rank's failure. */
SsStatus ss_pmg_apply_local(SsPmg *pmg, Vec rhs, Vec correction);
/** Make a PETSc PC apply this hierarchy; collective.
 * @param[in] pmg Set-up hierarchy; it must outlive the PC.
 * @param[in,out] pc Preconditioner to configure as a shell.
 * @return ::SS_OK, ::SS_ERROR_INVALID_STATE, ::SS_ERROR_INVALID_ARGUMENT or
 *   ::SS_ERROR_INTERNAL. */
SsStatus ss_pmg_configure_pc(SsPmg *pmg, PC pc);
/** Borrow the operator of one level; local.
 * @param[in] pmg Set-up hierarchy.
 * @param[in] level Level index, 0 is coarsest.
 * @param[out] operator_handle Borrowed operator.
 * @return ::SS_OK, ::SS_ERROR_INVALID_STATE or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_pmg_level_operator(SsPmg *pmg, size_t level, Mat *operator_handle);
/** Copy the active ranks of one level; local.
 * @param[in] pmg Hierarchy.
 * @param[in] level Level index.
 * @param[out] active_ranks Active world ranks.
 * @param[in] capacity Capacity of @p active_ranks.
 * @param[out] count Number of active ranks.
 * @return ::SS_OK, ::SS_ERROR_RESOURCE when the capacity is too small, or
 *   ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_pmg_active_ranks_get(const SsPmg *pmg, size_t level,
                                 PetscMPIInt *active_ranks, size_t capacity,
                                 size_t *count);
/** Role of this rank on one level; local.
 * @param[in] pmg Set-up hierarchy.
 * @param[in] level Level index.
 * @param[out] active Nonzero when this rank is active.
 * @param[out] owns_ksp Nonzero when this rank owns the level solver.
 * @param[out] active_rank Rank in the active communicator, or -1.
 * @return ::SS_OK, ::SS_ERROR_INVALID_STATE or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_pmg_level_local_ownership(const SsPmg *pmg, size_t level, int *active,
                                      int *owns_ksp, PetscMPIInt *active_rank);
/** Outcome of the most recent V-cycle; local.
 * @param[in] pmg Hierarchy.
 * @param[out] record Cycle outcome.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_pmg_last_apply(const SsPmg *pmg, SsPmgApplyRecord *record);
/** Cumulative counters; local.
 * @param[in] pmg Hierarchy.
 * @param[out] diagnostics Counters.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_pmg_diagnostics(const SsPmg *pmg, SsPmgDiagnostics *diagnostics);
/** Destroy a hierarchy, its operators and communicators; collective.
 * @param[in,out] pmg Hierarchy pointer, set to NULL; `*pmg` may be NULL.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_pmg_destroy(SsPmg **pmg);

#endif
