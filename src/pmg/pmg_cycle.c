/* PMG cycle, level solves, and PETSc preconditioner application.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "pmg_internal.h"
#include "slope_internal/assembly/assembly.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/core/util.h"
#include "slope_internal/diagnostics/diagnostics.h"

#include <math.h>
#include <string.h>

/* Record a local failure without shortening the cycle. All ranks continue the
 * same transfer, matrix-action and level-solve schedule, even if one of them
 * fails locally. The outer PC reports the sticky result to PETSc. */
static void ss_pmg_record_failure(SsPmg *pmg, size_t level, SsPmgFailurePoint point,
                                  SsStatus status) {
  if (status != SS_OK && pmg->last_apply.status == SS_OK) {
    pmg->last_apply.status = status;
    pmg->last_apply.failed_level = level;
    pmg->last_apply.failed_point = point;
  }
}

static void ss_pmg_inject_failure(SsPmg *pmg, size_t level, SsPmgFailurePoint point) {
  if (pmg->injected_failure == point && pmg->injected_failure_level == level) {
    ss_pmg_record_failure(pmg, level, point, SS_ERROR_INJECTED);
  }
}

static SsStatus ss_pmg_vector_local_finite(Vec vector) {
  const PetscScalar *values = NULL;
  PetscInt local_size = 0;
  if (VecGetLocalSize(vector, &local_size) != PETSC_SUCCESS ||
      VecGetArrayRead(vector, &values) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  SsStatus status = SS_OK;
  for (PetscInt index = 0; index < local_size; ++index) {
    if (PetscIsInfOrNanScalar(values[index])) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
  }
  return VecRestoreArrayRead(vector, &values) == PETSC_SUCCESS ? status
                                                               : SS_ERROR_INTERNAL;
}

static void ss_pmg_level_solve(SsPmg *pmg, size_t level_index, Vec rhs, Vec solution,
                               PetscInt fixed_iterations, int contract_validation) {
  SsPmgLevel *level = &pmg->levels[level_index];
  const int reduced = !level->identity_world_layout;
  ++pmg->last_apply.level_solve_calls;
  ++pmg->diagnostics.level_solve_calls;
  ss_pmg_inject_failure(pmg, level_index, SS_PMG_FAIL_BRIDGE_TO_ACTIVE);
  if (reduced) {
    SsStatus status =
        ss_pmg_bridge_to_active(pmg, level_index, rhs, level->active_rhs, 1);
    ss_pmg_record_failure(pmg, level_index, SS_PMG_FAIL_BRIDGE_TO_ACTIVE, status);
    if (ss_pmg_fgmres_full_iterate_level(pmg, level_index)) {
      status = ss_pmg_bridge_to_active(pmg, level_index, solution,
                                       level->active_solution, 1);
      ss_pmg_record_failure(pmg, level_index, SS_PMG_FAIL_BRIDGE_TO_ACTIVE, status);
    }
  }
  ss_pmg_inject_failure(pmg, level_index, SS_PMG_FAIL_LEVEL_SOLVE);
  if (level->local_active) {
    Vec solve_rhs = reduced ? level->active_rhs : rhs;
    Vec solve_solution = reduced ? level->active_solution : solution;
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    PetscInt iterations = 0;
    PetscReal residual_norm = 0.0;
    PetscErrorCode tolerances = PETSC_SUCCESS;
    if (fixed_iterations > 0) {
      tolerances = KSPSetTolerances(level->ksp, PETSC_CURRENT, PETSC_CURRENT,
                                    PETSC_CURRENT, fixed_iterations);
    }
    const int handler_pushed =
        PetscPushErrorHandler(PetscReturnErrorHandler, NULL) == PETSC_SUCCESS;
    /* Correction solves start at zero. The legacy full-iterate smoother
     * instead uses the current solution as its nonzero initial guess. */
#ifndef NDEBUG
    if (pmg->diagnostics.apply_calls <= 128U) {
      ss_debug_stage_vector("pmg-level-rhs", level_index, solve_rhs);
    }
#endif
    const PetscErrorCode solve_status = KSPSolve(level->ksp, solve_rhs, solve_solution);
#ifndef NDEBUG
    if (pmg->diagnostics.apply_calls <= 128U) {
      ss_debug_stage_vector("pmg-level-solution", level_index, solve_solution);
    }
#endif
    const int handler_ok = handler_pushed && PetscPopErrorHandler() == PETSC_SUCCESS;
    SsStatus status = SS_OK;
    const int action_failed =
        ss_operator_matrix_free_take_action_failure(level->active_operator);
    if (solve_status != PETSC_SUCCESS || tolerances != PETSC_SUCCESS || !handler_ok ||
        action_failed || KSPGetConvergedReason(level->ksp, &reason) != PETSC_SUCCESS ||
        KSPGetIterationNumber(level->ksp, &iterations) != PETSC_SUCCESS ||
        KSPGetResidualNorm(level->ksp, &residual_norm) != PETSC_SUCCESS ||
        (contract_validation && ss_pmg_vector_local_finite(solve_solution) != SS_OK)) {
      status = SS_ERROR_INTERNAL;
    } else {
      const PetscInt budget =
          fixed_iterations > 0 ? fixed_iterations : level->policy.maximum_iterations;
      const int fixed_work =
          reason == KSP_DIVERGED_ITS && iterations > 0 && iterations == budget;
      if ((reason <= 0 && !fixed_work) || !ss_is_finite(residual_norm)) {
        status = SS_ERROR_INTERNAL;
      }
    }
    if (iterations > 0) {
      pmg->last_apply.level_iterations[level_index] += (uint64_t)iterations;
    }
    if (pmg->last_apply.status == SS_OK) {
      pmg->last_apply.last_level_reason = reason;
      pmg->last_apply.last_level_iterations = iterations;
      pmg->last_apply.last_level_residual_norm = residual_norm;
    }
    ss_pmg_record_failure(pmg, level_index, SS_PMG_FAIL_LEVEL_SOLVE, status);
  }
  ss_pmg_inject_failure(pmg, level_index, SS_PMG_FAIL_BRIDGE_FROM_ACTIVE);
  if (reduced) {
    const SsStatus status = ss_pmg_bridge_from_active(
        pmg, level_index, level->active_solution, solution, 1);
    ss_pmg_record_failure(pmg, level_index, SS_PMG_FAIL_BRIDGE_FROM_ACTIVE, status);
  }
}

static void ss_pmg_residual(SsPmg *pmg, size_t level, SsPmgFailurePoint point) {
  ++pmg->last_apply.operator_apply_calls;
  ++pmg->diagnostics.operator_apply_calls;
  const PetscErrorCode mult =
      MatMult(pmg->operators[level], pmg->solution[level], pmg->residual[level]);
  const PetscErrorCode update = VecAYPX(pmg->residual[level], -1.0, pmg->rhs[level]);
  const int action_failed =
      ss_operator_matrix_free_take_action_failure(pmg->operators[level]);
  ss_pmg_record_failure(pmg, level, point,
                        mult == PETSC_SUCCESS && update == PETSC_SUCCESS &&
                                !action_failed
                            ? SS_OK
                            : SS_ERROR_INTERNAL);
}

static void ss_pmg_cycle(SsPmg *pmg, size_t level_index, int contract_validation) {
  SsPmgLevel *level = &pmg->levels[level_index];
  if (level_index == 0U) {
    ss_pmg_level_solve(pmg, level_index, pmg->rhs[level_index],
                       pmg->solution[level_index], 0, contract_validation);
    return;
  }
  /* The cycle starts from zero, so the initial residual is the RHS. */
  ss_pmg_inject_failure(pmg, level_index, SS_PMG_FAIL_PRE_SMOOTH);
  ss_pmg_level_solve(pmg, level_index, pmg->rhs[level_index],
                     level->identity_world_layout ? pmg->solution[level_index]
                                                  : pmg->delta[level_index],
                     level->policy.pre_smooth_steps, contract_validation);
  if (!level->identity_world_layout &&
      VecCopy(pmg->delta[level_index], pmg->solution[level_index]) != PETSC_SUCCESS) {
    ss_pmg_record_failure(pmg, level_index, SS_PMG_FAIL_PRE_SMOOTH, SS_ERROR_INTERNAL);
  }
  ss_pmg_residual(pmg, level_index, SS_PMG_FAIL_RESTRICT);
  ss_pmg_inject_failure(pmg, level_index, SS_PMG_FAIL_RESTRICT);
  if (MatMultTranspose(pmg->prolongations[level_index - 1U], pmg->residual[level_index],
                       pmg->rhs[level_index - 1U]) != PETSC_SUCCESS) {
    ss_pmg_record_failure(pmg, level_index, SS_PMG_FAIL_RESTRICT, SS_ERROR_INTERNAL);
  }
  ss_pmg_cycle(pmg, level_index - 1U, contract_validation);
  ss_pmg_inject_failure(pmg, level_index, SS_PMG_FAIL_PROLONG);
  const PetscErrorCode mult =
      MatMult(pmg->prolongations[level_index - 1U], pmg->solution[level_index - 1U],
              pmg->delta[level_index]);
  const PetscErrorCode update =
      VecAXPY(pmg->solution[level_index], 1.0, pmg->delta[level_index]);
  ss_pmg_record_failure(
      pmg, level_index, SS_PMG_FAIL_PROLONG,
      mult == PETSC_SUCCESS && update == PETSC_SUCCESS ? SS_OK : SS_ERROR_INTERNAL);
  ss_pmg_inject_failure(pmg, level_index, SS_PMG_FAIL_POST_SMOOTH);
  if (ss_pmg_fgmres_full_iterate_level(pmg, level_index)) {
    /* Fixed-work FGMRES with a nonzero full iterate is a different algorithm
     * from solving for a correction with a zero initial guess. */
    ss_pmg_level_solve(pmg, level_index, pmg->rhs[level_index],
                       pmg->solution[level_index], level->policy.post_smooth_steps,
                       contract_validation);
    return;
  }
  ss_pmg_residual(pmg, level_index, SS_PMG_FAIL_POST_SMOOTH);
  ss_pmg_level_solve(pmg, level_index, pmg->residual[level_index],
                     pmg->delta[level_index], level->policy.post_smooth_steps,
                     contract_validation);
  if (VecAXPY(pmg->solution[level_index], 1.0, pmg->delta[level_index]) !=
      PETSC_SUCCESS) {
    ss_pmg_record_failure(pmg, level_index, SS_PMG_FAIL_POST_SMOOTH, SS_ERROR_INTERNAL);
  }
}

static void ss_pmg_apply_cycle(SsPmg *pmg, Vec rhs, Vec correction,
                               int contract_validation) {
  const size_t finest = pmg->hierarchy.level_count - 1U;
  pmg->last_apply = (SsPmgApplyRecord){
      .sequence = ++pmg->apply_sequence,
      .failed_level = SIZE_MAX,
  };
  ++pmg->diagnostics.apply_calls;
  if (contract_validation) {
    ++pmg->diagnostics.contract_apply_calls;
    ++pmg->diagnostics.norm_validation_reductions;
    if (VecNorm(rhs, NORM_2, &pmg->last_apply.input_norm) != PETSC_SUCCESS ||
        !ss_is_finite(pmg->last_apply.input_norm)) {
      ss_pmg_record_failure(pmg, finest, SS_PMG_FAIL_LEVEL_SOLVE, SS_ERROR_INTEGRITY);
    }
  } else {
    ++pmg->diagnostics.preconditioner_apply_calls;
  }
  if (VecCopy(rhs, pmg->rhs[finest]) != PETSC_SUCCESS) {
    ss_pmg_record_failure(pmg, finest, SS_PMG_FAIL_LEVEL_SOLVE, SS_ERROR_INTERNAL);
  }
  if (pmg->fgmres_full_iterate_post) {
    for (size_t level = 0U; level < pmg->hierarchy.level_count; ++level) {
      const PetscErrorCode solution_status = VecSet(pmg->solution[level], 0.0);
      const PetscErrorCode delta_status = VecSet(pmg->delta[level], 0.0);
      if (solution_status != PETSC_SUCCESS || delta_status != PETSC_SUCCESS) {
        ss_pmg_record_failure(pmg, level, SS_PMG_FAIL_LEVEL_SOLVE, SS_ERROR_INTERNAL);
      }
    }
  }
  ss_pmg_cycle(pmg, finest, contract_validation);
  if (VecCopy(pmg->solution[finest], correction) != PETSC_SUCCESS) {
    ss_pmg_record_failure(pmg, finest, SS_PMG_FAIL_LEVEL_SOLVE, SS_ERROR_INTERNAL);
  }
}

SsStatus ss_pmg_apply_local(SsPmg *pmg, Vec rhs, Vec correction) {
  ss_pmg_apply_cycle(pmg, rhs, correction, 0);
  if (pmg->last_apply.status != SS_OK) {
    ++pmg->diagnostics.failed_apply_calls;
  }
  if (pmg->apply_observer != NULL) {
    pmg->apply_observer(pmg->apply_observer_context, &pmg->last_apply);
  }
  return pmg->last_apply.status;
}

void ss_pmg_set_apply_observer(SsPmg *pmg, SsPmgApplyObserver observe, void *context) {
  if (pmg != NULL) {
    pmg->apply_observer = observe;
    pmg->apply_observer_context = context;
  }
}

static SsStatus ss_pmg_apply_checked(SsPmg *pmg, Vec rhs, Vec correction,
                                     SsPmgApplyRecord *record,
                                     int contract_validation) {
  if (pmg == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  SsStatus status = SS_OK;
  PetscInt rhs_size = 0, correction_size = 0, operator_size = 0;
  const size_t finest = pmg->hierarchy.level_count - 1U;
  if (rhs == NULL || correction == NULL || record == NULL) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else if (!pmg->setup_complete || VecGetSize(rhs, &rhs_size) != PETSC_SUCCESS ||
             VecGetSize(correction, &correction_size) != PETSC_SUCCESS ||
             MatGetSize(pmg->operators[finest], &operator_size, NULL) !=
                 PETSC_SUCCESS ||
             rhs_size != operator_size || correction_size != operator_size) {
    status = SS_ERROR_INVALID_STATE;
  }
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  ss_pmg_apply_cycle(pmg, rhs, correction, contract_validation);
  if (contract_validation) {
    ++pmg->diagnostics.norm_validation_reductions;
    if (VecNorm(correction, NORM_2, &pmg->last_apply.output_norm) != PETSC_SUCCESS ||
        !ss_is_finite(pmg->last_apply.output_norm)) {
      ss_pmg_record_failure(pmg, finest, SS_PMG_FAIL_LEVEL_SOLVE, SS_ERROR_INTERNAL);
    }
  }
  status = ss_status_allreduce_max(pmg->communicator, pmg->last_apply.status);
  if (status != SS_OK) {
    int rank = pmg->last_apply.status == status ? pmg->world_rank : pmg->world_size;
    if (MPI_Allreduce(MPI_IN_PLACE, &rank, 1, MPI_INT, MPI_MIN, pmg->communicator) !=
            MPI_SUCCESS ||
        MPI_Bcast(&pmg->last_apply, (int)sizeof(pmg->last_apply), MPI_BYTE, rank,
                  pmg->communicator) != MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
    pmg->last_apply.status = status;
    (void)VecFlag(correction, 1);
    ++pmg->diagnostics.failed_apply_calls;
  }
  *record = pmg->last_apply;
  return status;
}

SsStatus ss_pmg_apply(SsPmg *pmg, Vec rhs, Vec correction, SsPmgApplyRecord *record) {
  return ss_pmg_apply_checked(pmg, rhs, correction, record, 1);
}

SsStatus ss_pmg_apply_preconditioner(SsPmg *pmg, Vec rhs, Vec correction,
                                     SsPmgApplyRecord *record) {
  return ss_pmg_apply_checked(pmg, rhs, correction, record, 0);
}

static PetscErrorCode ss_pmg_pc_apply(PC pc, Vec rhs, Vec correction) {
  SsPmg *pmg = NULL;
  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &pmg));
  const SsStatus status = ss_pmg_apply_local(pmg, rhs, correction);
  if (status != SS_OK) {
    PetscCall(PCSetFailedReason(pc, PC_SUBPC_ERROR));
  }
  /* PETSc reduces the failure reason when a Krylov norm/dot detects Inf. */
  PetscCall(VecFlag(correction, status != SS_OK));
  PetscFunctionReturn(PETSC_SUCCESS);
}

SsStatus ss_pmg_configure_pc(SsPmg *pmg, PC pc) {
  if (pmg == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  SsStatus status = SS_OK;
  if (pc == NULL) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else if (pmg->setup_complete == 0) {
    status = SS_ERROR_INVALID_STATE;
  }
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  status = ss_status_allreduce_max(
      pmg->communicator,
      PCSetType(pc, PCSHELL) == PETSC_SUCCESS ? SS_OK : SS_ERROR_INTERNAL);
  if (status != SS_OK) {
    return status;
  }
  const int configured = PCShellSetContext(pc, pmg) == PETSC_SUCCESS &&
                         PCShellSetName(pc, "slope-pmg-vcycle") == PETSC_SUCCESS &&
                         PCShellSetApply(pc, ss_pmg_pc_apply) == PETSC_SUCCESS;
  return ss_status_allreduce_max(pmg->communicator,
                                 configured ? SS_OK : SS_ERROR_INTERNAL);
}
