/* Private PMG level, bridge and hierarchy state and interfaces.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_PMG_PMG_INTERNAL_H
#define SS_PMG_PMG_INTERNAL_H

#include "slope_internal/pmg/pmg.h"

#include <petscsf.h>

typedef struct {
  int *send_counts;
  int *receive_counts;
  int *send_displacements;
  int *receive_displacements;
  int *row_offsets;
  MPI_Comm neighbors;
  PetscScalar *send_values;
  PetscScalar *receive_values;
  int send_total;
  int receive_total;
  PetscInt row_start;
  PetscInt row_end;
  PetscObjectState nonzero_state;
} SsPmgMatrixRoute;

typedef struct {
  SsPmgLevelPolicy policy;
  PetscMPIInt *active_ranks;
  int identity_world_layout;
  int local_active;
  int local_owns_ksp;
  PetscMPIInt local_active_rank;
  MPI_Comm active_communicator;
  int owns_active_communicator;
  Mat active_operator;
  KSP ksp;
  Vec active_rhs;
  Vec active_solution;
  MatNullSpace world_near_nullspace;
  MatNullSpace active_near_nullspace;
  SsPmgNullspaceRecord nullspace_record;
  uint64_t solver_setup_count;
  int repeated_factorization_reuses_row_permutation;
  size_t variable_point_block_count;
  int has_variable_point_blocks;
  PetscSF row_bridge;
  PetscScalar *bridge_world_fallback;
  PetscScalar *bridge_active_fallback;
  PetscInt bridge_global_size;
  SsPmgMatrixRoute matrix_route;
} SsPmgLevel;

typedef struct {
  SsPmgNearNullspaceInput input;
  PetscReal *coordinates;
  PetscInt *global_dof_indices;
} SsPmgNearNullspaceStorage;

struct SsPmg {
  MPI_Comm communicator;
  PetscMPIInt world_rank;
  PetscMPIInt world_size;
  int ordered_galerkin;
  SsPmgHierarchy hierarchy;
  Mat prolongations[SS_PMG_MAX_LEVELS - 1U];
  PetscObjectState prolongation_states[SS_PMG_MAX_LEVELS - 1U];
  SsPmgLevel levels[SS_PMG_MAX_LEVELS];
  SsPmgNearNullspaceStorage nullspace_inputs[SS_PMG_MAX_LEVELS];
  Mat operators[SS_PMG_MAX_LEVELS];
  Mat external_operators[SS_PMG_MAX_LEVELS];
  Vec rhs[SS_PMG_MAX_LEVELS];
  Vec solution[SS_PMG_MAX_LEVELS];
  Vec residual[SS_PMG_MAX_LEVELS];
  Vec delta[SS_PMG_MAX_LEVELS];
  uint64_t transfer_generation;
  uint64_t numeric_lag_interval;
  size_t memory_cap_bytes;
  SsPmgFailurePoint injected_failure;
  size_t injected_failure_level;
  Mat observed_fine_operator;
  PetscObjectState observed_fine_state;
  uint64_t structure_generation;
  uint64_t requested_value_generation;
  uint64_t preconditioner_value_generation;
  uint64_t hierarchy_build_count;
  uint64_t transfer_setup_count;
  uint64_t galerkin_symbolic_count;
  uint64_t galerkin_numeric_count;
  uint64_t level_solver_setup_count;
  uint64_t setup_calls;
  uint64_t apply_sequence;
  int setup_complete;
  int external_operator_mode;
  int fgmres_full_iterate_post;
  SsPmgSetupRecord last_setup;
  SsPmgApplyRecord last_apply;
  SsPmgApplyObserver apply_observer;
  void *apply_observer_context;
  SsPmgDiagnostics diagnostics;
};

/** Whether a level's post FGMRES uses the full coarse-corrected iterate. */
static inline int ss_pmg_fgmres_full_iterate_level(const SsPmg *pmg,
                                                   size_t level_index) {
  const SsPmgLevelMethod method = pmg->levels[level_index].policy.method;
  return pmg->fgmres_full_iterate_post != 0 && level_index > 0U &&
         (method == SS_PMG_LEVEL_FGMRES_JACOBI ||
          method == SS_PMG_LEVEL_FGMRES_VPBJACOBI ||
          method == SS_PMG_LEVEL_FGMRES_GAMG);
}

/* PMG hierarchy, rank and level-solver policy validation */
/** Whether a level method is a smoother rather than a coarse solver. */
int ss_pmg_smoother_method(SsPmgLevelMethod method);
/** Copy a level policy, including its active ranks. */
SsStatus ss_pmg_policy_copy(SsPmg *pmg, size_t level, const SsPmgLevelPolicy *source);

/* Rigid-mode and hydraulic near-nullspace construction */
/** Copy the near-nullspace input of a level. */
SsStatus ss_pmg_nullspace_storage_copy(SsPmgNearNullspaceStorage *storage,
                                       const SsPmgNearNullspaceInput *source);
/** Attach the near-nullspace to a level's active operator; collective. */
SsStatus ss_pmg_active_near_nullspace_create(SsPmg *pmg, size_t level_index);
/** Build the near-nullspaces of every level; collective. */
SsStatus ss_pmg_near_nullspaces_build(SsPmg *pmg);

/* Active-rank communicators, operators and inter-layout bridges */
/** Destroy a level's solver, operators and work vectors. */
void ss_pmg_level_runtime_clear(SsPmgLevel *level);
/** Create a level's active-rank communicator and routes; collective. */
SsStatus ss_pmg_level_communicator_create(SsPmg *pmg, size_t level_index);
/** Redistribute a level matrix onto its active ranks; collective. */
SsStatus ss_pmg_active_matrix_update(SsPmg *pmg, size_t level_index, Mat world_matrix,
                                     int reuse);
/** Build the sparse ownership mapping between world and active rows. */
SsStatus ss_pmg_row_bridge_create(SsPmg *pmg, size_t level_index);
/** Move a vector from all ranks to a level's active ranks; collective. */
SsStatus ss_pmg_bridge_to_active(SsPmg *pmg, size_t level_index, Vec world_vector,
                                 Vec active_vector, int account);
/** Move a vector from a level's active ranks back to all ranks; collective. */
SsStatus ss_pmg_bridge_from_active(SsPmg *pmg, size_t level_index, Vec active_vector,
                                   Vec world_vector, int account);

/* PMG lifecycle, configured operators and diagnostics */
/** Release the setup of every level. */
void ss_pmg_setup_clear(SsPmg *pmg);

/* Level KSPs, world hierarchy and variable point blocks */
/** Build transfers and symbolic Galerkin products on all ranks; collective. */
SsStatus ss_pmg_world_hierarchy_build(SsPmg *pmg, Mat fine_operator);
/** Refresh the numeric Galerkin products for a new fine operator; collective. */
SsStatus ss_pmg_world_hierarchy_refresh(SsPmg *pmg, Mat fine_operator);
/** Set the variable point-block sizes of a level; collective. */
SsStatus ss_pmg_variable_point_blocks_set(SsPmg *pmg, size_t level_index);
/** Create the level solvers on their active ranks; collective. */
SsStatus ss_pmg_active_levels_build(SsPmg *pmg);
/** Refresh the level solvers after new values; collective. */
SsStatus ss_pmg_active_levels_refresh(SsPmg *pmg);

#endif
