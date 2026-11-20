/* Mechanics continuation policy and accepted-attempt callbacks.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/core/collective.h"
#include "solver_internal.h"

#include <petsclog.h>

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static SsStatus ss_solver_mechanics_analysis_fixed(void *opaque, PetscReal factor,
                                                   Vec incoming, Vec solution,
                                                   SsNonlinearSolveRecord *record) {
  SsSolverMechanics *mechanics = (SsSolverMechanics *)opaque;
  SsStatus status;
  if (mechanics == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_status_allreduce_max(
      mechanics->graph->communicator,
      mechanics->next_value_generation == 0U ? SS_ERROR_INVALID_ARGUMENT
      : mechanics->next_value_generation > UINT64_MAX - SS_NONLINEAR_HISTORY_CAPACITY
          ? SS_ERROR_OVERFLOW
          : SS_OK);
  if (status != SS_OK) {
    return status;
  }
  mechanics->nonlinear.first_value_generation = mechanics->next_value_generation;
  mechanics->next_value_generation += SS_NONLINEAR_HISTORY_CAPACITY;
  mechanics->nonlinear.deflation_promotion =
      mechanics->graph->policy.deflation_candidate_policy ==
              SS_DEFLATION_CANDIDATE_ACCEPTED_STATES_AND_FULL_KRYLOV
          ? SS_NONLINEAR_DEFLATION_PROMOTE_FULL_KRYLOV_AND_ACCEPTED_STATE
      : mechanics->graph->policy.deflation_candidate_policy ==
              SS_DEFLATION_CANDIDATE_ACCEPTED_STATES_AND_SYSTEM_SOLUTIONS
          ? SS_NONLINEAR_DEFLATION_PROMOTE_SYSTEM_SOLUTIONS_AND_ACCEPTED_STATE
      : mechanics->graph->policy.deflation_candidate_policy !=
              SS_DEFLATION_CANDIDATE_ACCEPTED_STATE_VECTORS
          ? SS_NONLINEAR_DEFLATION_PROMOTE_DIRECTIONS
      : mechanics->graph->model.analysis_method == SS_ANALYSIS_SSR_INDIRECT &&
              mechanics->accepted_generation != 0U
          ? SS_NONLINEAR_DEFLATION_DISCARD_PROVISIONAL
          : SS_NONLINEAR_DEFLATION_PROMOTE_ACCEPTED_STATE;
  SsNonlinearPolicy policy =
      mechanics->graph->model.analysis_method == SS_ANALYSIS_SSR_INDIRECT
          ? mechanics->initialization_nonlinear_policy
          : mechanics->nonlinear_policy;
  if (mechanics->graph->policy.settings.nonlinear_fixed_line_search == 1) {
    policy.line_method = SS_NONLINEAR_LINE_RESIDUAL_MONOTONE;
  }
  return ss_nonlinear_fixed_solve(&mechanics->nonlinear, &policy, factor, incoming,
                                  solution, record);
}

static SsStatus ss_solver_mechanics_analysis_work(void *opaque, SsAnalysisMethod method,
                                                  PetscReal target_work,
                                                  PetscReal incoming_factor,
                                                  Vec incoming, Vec solution,
                                                  PetscReal *accepted_factor,
                                                  SsNonlinearSolveRecord *record) {
  SsSolverMechanics *mechanics = (SsSolverMechanics *)opaque;
  SsNonlinearPolicy policy;
  SsNonlinearControlMode mode = SS_NONLINEAR_CONTROL_LIMIT_LOAD;
  int guard_enabled = 0;
  SsStatus status = SS_OK;
  if (mechanics == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  policy = mechanics->nonlinear_policy;
  if (method == SS_ANALYSIS_LIMIT_LOAD) {
    mode = SS_NONLINEAR_CONTROL_LIMIT_LOAD;
  } else if (method == SS_ANALYSIS_SSR_INDIRECT) {
    mode = SS_NONLINEAR_CONTROL_SSR_INDIRECT;
    policy.line_method = SS_NONLINEAR_LINE_RESIDUAL_MONOTONE;
  } else {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_status_allreduce_max(mechanics->graph->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  const int learn_guard =
      mode == SS_NONLINEAR_CONTROL_SSR_INDIRECT &&
      ss_solver_progress_guard_learns(mechanics->progress_guard_mode);
  if (learn_guard) {
    PetscReal threshold;
    PetscReal ratio_threshold;
    status = ss_nonlinear_adaptive_progress_guard_threshold(
        &mechanics->adaptive_progress_guard, policy.maximum_iterations, &guard_enabled,
        &threshold, &ratio_threshold);
    if (status == SS_OK && guard_enabled != 0 &&
        mechanics->progress_guard_mode == SS_NONLINEAR_PROGRESS_GUARD_ADAPTIVE) {
      policy.indirect_progress_guard_iteration =
          mechanics->adaptive_progress_guard.checkpoint_iteration;
      policy.indirect_progress_guard_maximum_convergence = threshold;
      policy.indirect_progress_guard_maximum_convergence_ratio = ratio_threshold;
    }
  }
  if (status == SS_OK &&
      (mechanics->next_value_generation == 0U ||
       mechanics->next_value_generation > UINT64_MAX - SS_NONLINEAR_HISTORY_CAPACITY)) {
    status = SS_ERROR_OVERFLOW;
  }
  status = ss_status_allreduce_max(mechanics->graph->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  mechanics->nonlinear.first_value_generation = mechanics->next_value_generation;
  mechanics->next_value_generation += SS_NONLINEAR_HISTORY_CAPACITY;
  mechanics->nonlinear.deflation_promotion =
      mechanics->graph->policy.deflation_candidate_policy ==
              SS_DEFLATION_CANDIDATE_ACCEPTED_STATES_AND_FULL_KRYLOV
          ? SS_NONLINEAR_DEFLATION_PROMOTE_FULL_KRYLOV_AND_ACCEPTED_STATE
      : mechanics->graph->policy.deflation_candidate_policy ==
              SS_DEFLATION_CANDIDATE_ACCEPTED_STATES_AND_SYSTEM_SOLUTIONS
          ? SS_NONLINEAR_DEFLATION_PROMOTE_SYSTEM_SOLUTIONS_AND_ACCEPTED_STATE
      : mechanics->graph->policy.deflation_candidate_policy ==
              SS_DEFLATION_CANDIDATE_ACCEPTED_STATE_VECTORS
          ? SS_NONLINEAR_DEFLATION_PROMOTE_ACCEPTED_STATE
          : SS_NONLINEAR_DEFLATION_PROMOTE_DIRECTIONS;
  status = ss_nonlinear_work_controlled_solve(&mechanics->nonlinear, &policy, mode,
                                              target_work, incoming_factor, incoming,
                                              solution, accepted_factor, record);
  if (status == SS_OK && learn_guard) {
    int sample_observed;
    PetscReal checkpoint_convergence;
    status = ss_nonlinear_adaptive_progress_guard_observe(
        &mechanics->adaptive_progress_guard, record, &sample_observed,
        &checkpoint_convergence);
  }
  return status;
}

static SsStatus ss_solver_mechanics_analysis_potential(void *opaque, PetscReal factor,
                                                       Vec state,
                                                       PetscReal *total_potential) {
  SsSolverMechanics *mechanics = (SsSolverMechanics *)opaque;
  Vec residual = NULL;
  SsStatus status;
  if (mechanics == NULL || state == NULL || total_potential == NULL ||
      VecDuplicate(state, &residual) != PETSC_SUCCESS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_solver_mechanics_evaluate_common(mechanics, state, factor, 1.0, residual,
                                               NULL, total_potential);
  if (VecDestroy(&residual) != PETSC_SUCCESS && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  return status;
}

static SsStatus
ss_solver_mechanics_displacement_maximum(void *opaque, Vec state,
                                         PetscReal *displacement_maximum) {
  SsSolverMechanics *mechanics = (SsSolverMechanics *)opaque;
  if (mechanics == NULL || mechanics->graph == NULL ||
      mechanics->graph->mechanics_space == NULL || state == NULL ||
      displacement_maximum == NULL ||
      (mechanics->graph->model.dimension != 2U &&
       mechanics->graph->model.dimension != 3U)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return ss_field_space_nodal_maximum(mechanics->graph->mechanics_space, state,
                                      displacement_maximum);
}

static SsStatus ss_solver_mechanics_analysis_begin(void *opaque, size_t attempt_ordinal,
                                                   size_t parent_accepted_ordinal) {
  SsSolverMechanics *mechanics = (SsSolverMechanics *)opaque;
  SsStatus status;
  (void)parent_accepted_ordinal;
  if (mechanics == NULL) {
    return SS_ERROR_INVALID_STATE;
  }
  status = ss_status_allreduce_max(
      mechanics->graph->communicator,
      mechanics->transaction_open != 0 ? SS_ERROR_INVALID_STATE : SS_OK);
  if (status != SS_OK) {
    return status;
  }
  mechanics->transaction_generation = (uint64_t)attempt_ordinal + 1U;
  mechanics->nonlinear.transaction_generation = mechanics->transaction_generation;
  mechanics->nonlinear.accepted_generation = mechanics->accepted_generation + 1U;
  mechanics->adaptive_progress_guard_snapshot = mechanics->adaptive_progress_guard;
  if (mechanics->deflation != NULL) {
    status =
        ss_deflation_begin(mechanics->deflation, mechanics->transaction_generation);
    if (status != SS_OK) {
      return status;
    }
  }
  if (mechanics->krylov_recycle != NULL) {
    status = ss_linear_recycle_begin(mechanics->krylov_recycle,
                                     mechanics->transaction_generation);
    if (status != SS_OK) {
      if (mechanics->deflation != NULL) {
        (void)ss_deflation_rollback(mechanics->deflation);
      }
      return status;
    }
  }
  mechanics->transaction_open = 1;
  return SS_OK;
}

static SsStatus ss_solver_mechanics_analysis_commit(void *opaque,
                                                    size_t attempt_ordinal,
                                                    size_t parent_accepted_ordinal) {
  SsSolverMechanics *mechanics = (SsSolverMechanics *)opaque;
  const uint64_t next_accepted_generation =
      mechanics == NULL ? 0U : mechanics->accepted_generation + 1U;
  SsStatus status;
  (void)attempt_ordinal;
  (void)parent_accepted_ordinal;
  if (mechanics == NULL) {
    return SS_ERROR_INVALID_STATE;
  }
  status = ss_status_allreduce_max(
      mechanics->graph->communicator,
      mechanics->transaction_open == 0 ? SS_ERROR_INVALID_STATE : SS_OK);
  if (status != SS_OK) {
    return status;
  }
  if (mechanics->deflation != NULL) {
    (void)ss_deflation_diagnostics(mechanics->deflation,
                                   &mechanics->last_deflation_diagnostics);
    status = ss_deflation_commit(mechanics->deflation, next_accepted_generation);
    mechanics->last_transaction_status = status;
    if (status != SS_OK) {
      return status;
    }
  }
  if (mechanics->krylov_recycle != NULL) {
    status =
        ss_linear_recycle_commit(mechanics->krylov_recycle, next_accepted_generation);
    mechanics->last_transaction_status = status;
    if (status != SS_OK) {
      return status;
    }
  }
  mechanics->accepted_generation = next_accepted_generation;
  mechanics->nonlinear.accepted_generation = mechanics->accepted_generation;
  mechanics->transaction_open = 0;
  return SS_OK;
}

static SsStatus ss_solver_mechanics_analysis_rollback(void *opaque,
                                                      size_t attempt_ordinal,
                                                      size_t parent_accepted_ordinal) {
  SsSolverMechanics *mechanics = (SsSolverMechanics *)opaque;
  SsStatus status;
  (void)attempt_ordinal;
  (void)parent_accepted_ordinal;
  if (mechanics == NULL) {
    return SS_ERROR_INVALID_STATE;
  }
  status = ss_status_allreduce_max(
      mechanics->graph->communicator,
      mechanics->transaction_open == 0 ? SS_ERROR_INVALID_STATE : SS_OK);
  if (status != SS_OK) {
    return status;
  }
  if (mechanics->deflation != NULL) {
    (void)ss_deflation_diagnostics(mechanics->deflation,
                                   &mechanics->last_deflation_diagnostics);
    status = ss_deflation_rollback(mechanics->deflation);
    mechanics->last_transaction_status = status;
    if (status != SS_OK) {
      return status;
    }
  }
  if (mechanics->krylov_recycle != NULL) {
    status = ss_linear_recycle_rollback(mechanics->krylov_recycle);
    mechanics->last_transaction_status = status;
    if (status != SS_OK) {
      return status;
    }
  }
  mechanics->adaptive_progress_guard = mechanics->adaptive_progress_guard_snapshot;
  mechanics->transaction_open = 0;
  return SS_OK;
}

void ss_solver_mechanics_analysis_problem(SsSolverMechanics *mechanics,
                                          SsAnalysisProblem *problem) {
  memset(problem, 0, sizeof(*problem));
  problem->communicator = mechanics->graph->communicator;
  problem->reference_load = mechanics->reference_load;
  problem->operations.solve_fixed = ss_solver_mechanics_analysis_fixed;
  problem->operations.solve_work_controlled = ss_solver_mechanics_analysis_work;
  problem->operations.total_potential = ss_solver_mechanics_analysis_potential;
  problem->operations.displacement_maximum = ss_solver_mechanics_displacement_maximum;
  problem->operations.begin_attempt = ss_solver_mechanics_analysis_begin;
  problem->operations.commit_attempt = ss_solver_mechanics_analysis_commit;
  problem->operations.rollback_attempt = ss_solver_mechanics_analysis_rollback;
  problem->context = mechanics;
}

SsStatus ss_solver_mechanics_analysis_policy(const SsSolverGraph *graph,
                                             PetscReal reference_work,
                                             SsAnalysisPolicy *policy) {
  PetscInt indirect_growth_iteration_limit = 0;
  PetscInt work_efficiency_iteration_maximum = 0;
  PetscReal indirect_sensitivity_relative_step = cbrt(DBL_EPSILON);
  PetscReal predictor_error_target = 0.0;
  PetscReal work_step_scale_minimum = 0.0;
  PetscReal work_step_scale_maximum = 0.0;
  SsNonlinearControlSensitivityMethod indirect_sensitivity_method =
      SS_NONLINEAR_CONTROL_SENSITIVITY_REFERENCE_FORWARD;
  SsAnalysisWorkStepController work_step_controller = SS_ANALYSIS_WORK_STEP_CLASSIC;
  PetscReal factor_horizon;
  PetscReal work_scale;
  if (graph == NULL || policy == NULL || !isfinite((double)reference_work) ||
      reference_work <= 0.0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  ss_solver_indirect_work_step_options(
      &graph->policy, &work_step_controller, &indirect_growth_iteration_limit,
      &work_efficiency_iteration_maximum, &predictor_error_target,
      &work_step_scale_minimum, &work_step_scale_maximum);
  ss_solver_indirect_sensitivity_options(&graph->policy.settings,
                                         &indirect_sensitivity_method,
                                         &indirect_sensitivity_relative_step);
  /*
   * The accepted-state cap is an execution horizon and may be shortened to
   * create a resumable checkpoint.  It must not alter any target proposed
   * before that boundary, so derive the factor guard from the fixed analysis
   * record capacity rather than from the requested execution horizon.
   */
  factor_horizon =
      (PetscReal)graph->policy.initial_factor +
      (PetscReal)SS_ANALYSIS_POINT_CAPACITY * (PetscReal)graph->policy.maximum_step;
  if (!isfinite((double)factor_horizon) ||
      factor_horizon < (PetscReal)graph->policy.initial_factor) {
    factor_horizon = PETSC_MAX_REAL;
  }
  work_scale = fmax(fabs(reference_work), DBL_MIN);
  memset(policy, 0, sizeof(*policy));
  policy->method = graph->model.analysis_method;
  policy->nonlinear_tolerance =
      (PetscReal)fmax(graph->policy.nonlinear_atol, graph->policy.nonlinear_rtol);
  policy->factor_initial = (PetscReal)graph->policy.initial_factor;
  policy->factor_step_initial = (PetscReal)graph->policy.initial_step;
  policy->factor_step_minimum = (PetscReal)graph->policy.minimum_step;
  policy->factor_maximum = factor_horizon;
  policy->work_step_initial = (PetscReal)graph->policy.initial_work_scale * work_scale;
  policy->work_step_minimum = (PetscReal)graph->policy.minimum_step * work_scale;
  policy->work_maximum = (PetscReal)graph->policy.maximum_work;
  policy->direct_control_maximum = 0.0;
  policy->scaled_slope_tolerance = (PetscReal)graph->policy.scaled_slope_tolerance;
  policy->factor_increment_stop = (PetscReal)graph->policy.factor_increment_stop;
  policy->accepted_step_limit = graph->policy.maximum_accepted_states;
  policy->consecutive_retry_limit =
      graph->policy.settings.continuation_consecutive_retry_limit;
  policy->work_growth_iteration_limit =
      policy->method == SS_ANALYSIS_SSR_INDIRECT ? indirect_growth_iteration_limit : 0;
  policy->work_efficiency_iteration_maximum = policy->method == SS_ANALYSIS_SSR_INDIRECT
                                                  ? work_efficiency_iteration_maximum
                                                  : 0;
  policy->work_step_controller = policy->method == SS_ANALYSIS_SSR_INDIRECT
                                     ? work_step_controller
                                     : SS_ANALYSIS_WORK_STEP_CLASSIC;
  policy->indirect_predictor_mode = SS_ANALYSIS_INDIRECT_PREDICTOR_DISPLACEMENT_SECANT;
  policy->work_predictor_error_target =
      policy->method == SS_ANALYSIS_SSR_INDIRECT ? predictor_error_target : 0.0;
  policy->work_step_scale_minimum =
      policy->method == SS_ANALYSIS_SSR_INDIRECT ? work_step_scale_minimum : 0.0;
  policy->work_step_scale_maximum =
      policy->method == SS_ANALYSIS_SSR_INDIRECT ? work_step_scale_maximum : 0.0;
  policy->indirect_sensitivity_method =
      policy->method == SS_ANALYSIS_SSR_INDIRECT
          ? indirect_sensitivity_method
          : SS_NONLINEAR_CONTROL_SENSITIVITY_REFERENCE_FORWARD;
  policy->indirect_sensitivity_relative_step =
      policy->method == SS_ANALYSIS_SSR_INDIRECT ? indirect_sensitivity_relative_step
                                                 : 0.0;
  return ss_analysis_policy_validate(policy);
}
