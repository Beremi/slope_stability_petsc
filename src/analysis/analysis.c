/* Continuation policy, attempt transactions and common driver dispatch.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "analysis_internal.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/core/util.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

SsStatus ss_analysis_collective_nonlinear_outcome(MPI_Comm communicator,
                                                  SsNonlinearOutcome local_outcome,
                                                  SsNonlinearOutcome *global_outcome) {
  int local = (int)local_outcome;
  int global = 0;
  if (MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, communicator) !=
      MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  *global_outcome = (SsNonlinearOutcome)global;
  return SS_OK;
}

SsStatus ss_analysis_collective_real(MPI_Comm communicator, PetscReal local_value,
                                     PetscReal *global_value) {
  PetscReal minimum = 0.0;
  PetscReal maximum = 0.0;
  int local_finite = ss_is_finite(local_value);
  int all_finite = 0;
  if (MPI_Allreduce(&local_finite, &all_finite, 1, MPI_INT, MPI_MIN, communicator) !=
      MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (all_finite == 0) {
    return SS_ERROR_INTEGRITY;
  }
  if (MPI_Allreduce(&local_value, &minimum, 1, MPIU_REAL, MPI_MIN, communicator) !=
          MPI_SUCCESS ||
      MPI_Allreduce(&local_value, &maximum, 1, MPIU_REAL, MPI_MAX, communicator) !=
          MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (minimum != maximum) {
    return SS_ERROR_INTEGRITY;
  }
  *global_value = maximum;
  return SS_OK;
}

SsStatus ss_analysis_collective_cancelled(const SsAnalysisProblem *problem,
                                          int *cancelled) {
  int local = 0;
  int global = 0;
  if (problem->operations.cancelled != NULL) {
    local = problem->operations.cancelled(problem->context) != 0;
  }
  if (MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, problem->communicator) !=
      MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  *cancelled = global != 0;
  return SS_OK;
}

int ss_analysis_same_communicator(MPI_Comm expected, PetscObject object) {
  MPI_Comm actual;
  int comparison = MPI_UNEQUAL;
  if (expected == MPI_COMM_NULL || object == NULL) {
    return 0;
  }
  actual = PetscObjectComm(object);
  return MPI_Comm_compare(expected, actual, &comparison) == MPI_SUCCESS &&
         (comparison == MPI_IDENT || comparison == MPI_CONGRUENT);
}

int ss_analysis_nonlinear_succeeded(SsNonlinearOutcome outcome) {
  return outcome == SS_NONLINEAR_CONVERGED || outcome == SS_NONLINEAR_ACCEPTED_RELAXED;
}

static void ss_analysis_set_identities(SsAnalysisRecord *record,
                                       const SsAnalysisPolicy *policy) {
  const char *method_identity;
  const char *predictor_identity;
  const char *controller_identity;
  switch (policy->method) {
  case SS_ANALYSIS_LIMIT_LOAD:
    method_identity = "limit_load/work_classic";
    predictor_identity = "limit_load/displacement_secant";
    controller_identity = "limit_load/work_classic";
    break;
  case SS_ANALYSIS_SSR_DIRECT:
    method_identity = "ssr_direct/reference_fd";
    predictor_identity = "ssr_direct/accepted_state+paired_beta";
    controller_identity = "ssr_direct/reference";
    break;
  case SS_ANALYSIS_SSR_INDIRECT:
    method_identity = policy->indirect_sensitivity_method ==
                              SS_NONLINEAR_CONTROL_SENSITIVITY_STABLE_CENTRAL
                          ? "ssr_indirect/stable_central_fd"
                          : "ssr_indirect/classic_fd";
    predictor_identity =
        policy->indirect_predictor_mode == SS_ANALYSIS_INDIRECT_PREDICTOR_SCALED_CURRENT
            ? "ssr_indirect/scaled_current+anchored_factor"
            : "ssr_indirect/displacement_secant+anchored_factor";
    controller_identity =
        policy->work_step_controller == SS_ANALYSIS_WORK_STEP_WORK_EFFICIENCY
            ? "ssr_indirect/work_efficiency_adaptive"
        : policy->work_step_controller == SS_ANALYSIS_WORK_STEP_PREDICTOR_ERROR
            ? "ssr_indirect/predictor_error_state_only"
        : policy->work_growth_iteration_limit > 0 ? "ssr_indirect/effort_limited"
                                                  : "ssr_indirect/classic";
    break;
  default:
    record->method_identity[0] = '\0';
    record->predictor_identity[0] = '\0';
    record->controller_identity[0] = '\0';
    return;
  }
  (void)snprintf(record->method_identity, sizeof(record->method_identity), "%s",
                 method_identity);
  (void)snprintf(record->predictor_identity, sizeof(record->predictor_identity), "%s",
                 predictor_identity);
  (void)snprintf(record->controller_identity, sizeof(record->controller_identity), "%s",
                 controller_identity);
}

SsStatus ss_analysis_external_work(const SsAnalysisProblem *problem, Vec state,
                                   PetscReal *work) {
  PetscScalar scalar = 0.0;
  if (VecDot(problem->reference_load, state, &scalar) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  *work = PetscRealPart(scalar);
  return ss_is_finite(*work) ? SS_OK : SS_ERROR_INTEGRITY;
}

static SsStatus ss_analysis_displacement_maximum(const SsAnalysisProblem *problem,
                                                 Vec state, PetscReal *maximum) {
  SsStatus status;
  if (problem->operations.displacement_maximum != NULL) {
    status = problem->operations.displacement_maximum(problem->context, state, maximum);
  } else {
    status = VecNorm(state, NORM_INFINITY, maximum) == PETSC_SUCCESS
                 ? SS_OK
                 : SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(problem->communicator, status);
  return status != SS_OK ? status
                         : (ss_is_finite(*maximum) ? SS_OK : SS_ERROR_INTEGRITY);
}

static SsStatus ss_analysis_problem_validate(const SsAnalysisProblem *problem,
                                             Vec initial_state, Vec accepted_state) {
  PetscInt load_size = 0;
  PetscInt initial_size = 0;
  PetscInt accepted_size = 0;
  if (problem == NULL || initial_state == NULL || accepted_state == NULL ||
      problem->communicator == MPI_COMM_NULL || problem->reference_load == NULL ||
      !ss_analysis_same_communicator(problem->communicator,
                                     (PetscObject)problem->reference_load) ||
      !ss_analysis_same_communicator(problem->communicator,
                                     (PetscObject)initial_state) ||
      !ss_analysis_same_communicator(problem->communicator,
                                     (PetscObject)accepted_state) ||
      VecGetSize(problem->reference_load, &load_size) != PETSC_SUCCESS ||
      VecGetSize(initial_state, &initial_size) != PETSC_SUCCESS ||
      VecGetSize(accepted_state, &accepted_size) != PETSC_SUCCESS || load_size <= 0 ||
      load_size != initial_size || load_size != accepted_size ||
      ((problem->operations.begin_attempt == NULL ||
        problem->operations.commit_attempt == NULL ||
        problem->operations.rollback_attempt == NULL) &&
       (problem->operations.begin_attempt != NULL ||
        problem->operations.commit_attempt != NULL ||
        problem->operations.rollback_attempt != NULL))) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return SS_OK;
}

PetscInt ss_analysis_maximum_accepted_steps(SsAnalysisMethod method) {
  return (PetscInt)SS_ANALYSIS_POINT_CAPACITY - (method == SS_ANALYSIS_LIMIT_LOAD);
}

SsStatus ss_analysis_policy_validate(const SsAnalysisPolicy *policy) {
  if (policy == NULL ||
      (policy->method != SS_ANALYSIS_LIMIT_LOAD &&
       policy->method != SS_ANALYSIS_SSR_DIRECT &&
       policy->method != SS_ANALYSIS_SSR_INDIRECT) ||
      !ss_is_finite(policy->nonlinear_tolerance) ||
      policy->nonlinear_tolerance <= 0.0 || !ss_is_finite(policy->factor_initial) ||
      policy->factor_initial <= 0.0 || !ss_is_finite(policy->factor_step_initial) ||
      policy->factor_step_initial <= 0.0 ||
      !ss_is_finite(policy->factor_step_minimum) ||
      policy->factor_step_minimum <= 0.0 ||
      policy->factor_step_minimum > policy->factor_step_initial ||
      !ss_is_finite(policy->factor_maximum) ||
      policy->factor_maximum < policy->factor_initial ||
      !ss_is_finite(policy->work_step_initial) || policy->work_step_initial <= 0.0 ||
      !ss_is_finite(policy->work_step_minimum) || policy->work_step_minimum <= 0.0 ||
      policy->work_step_minimum > policy->work_step_initial ||
      !ss_is_finite(policy->work_maximum) || policy->work_maximum <= 0.0 ||
      !ss_is_finite(policy->direct_control_maximum) ||
      policy->direct_control_maximum < 0.0 ||
      !ss_is_finite(policy->scaled_slope_tolerance) ||
      policy->scaled_slope_tolerance < 0.0 ||
      !ss_is_finite(policy->factor_increment_stop) ||
      policy->factor_increment_stop < 0.0 || policy->accepted_step_limit <= 0 ||
      policy->accepted_step_limit >
          ss_analysis_maximum_accepted_steps(policy->method) ||
      policy->consecutive_retry_limit <= 0 || policy->consecutive_retry_limit > 5 ||
      policy->work_growth_iteration_limit < 0 ||
      (policy->work_step_controller != SS_ANALYSIS_WORK_STEP_CLASSIC &&
       policy->work_step_controller != SS_ANALYSIS_WORK_STEP_PREDICTOR_ERROR &&
       policy->work_step_controller != SS_ANALYSIS_WORK_STEP_WORK_EFFICIENCY) ||
      (policy->indirect_predictor_mode !=
           SS_ANALYSIS_INDIRECT_PREDICTOR_DISPLACEMENT_SECANT &&
       policy->indirect_predictor_mode !=
           SS_ANALYSIS_INDIRECT_PREDICTOR_SCALED_CURRENT) ||
      (policy->method != SS_ANALYSIS_SSR_INDIRECT &&
       policy->indirect_predictor_mode !=
           SS_ANALYSIS_INDIRECT_PREDICTOR_DISPLACEMENT_SECANT) ||
      (policy->method != SS_ANALYSIS_SSR_INDIRECT &&
       (policy->work_growth_iteration_limit != 0 ||
        policy->work_efficiency_iteration_maximum != 0 ||
        policy->work_step_controller != SS_ANALYSIS_WORK_STEP_CLASSIC)) ||
      (ss_analysis_work_step_uses_predictor_error(policy->work_step_controller) &&
       (policy->method != SS_ANALYSIS_SSR_INDIRECT ||
        policy->work_growth_iteration_limit <= 0 ||
        !ss_is_finite(policy->work_predictor_error_target) ||
        policy->work_predictor_error_target <= 0.0 ||
        !ss_is_finite(policy->work_step_scale_minimum) ||
        policy->work_step_scale_minimum <= 0.0 ||
        policy->work_step_scale_minimum > 1.0 ||
        !ss_is_finite(policy->work_step_scale_maximum) ||
        policy->work_step_scale_maximum < 1.0 ||
        policy->work_step_scale_maximum < policy->work_step_scale_minimum)) ||
      (policy->work_step_controller == SS_ANALYSIS_WORK_STEP_WORK_EFFICIENCY &&
       policy->work_efficiency_iteration_maximum <
           policy->work_growth_iteration_limit) ||
      (policy->work_step_controller != SS_ANALYSIS_WORK_STEP_WORK_EFFICIENCY &&
       policy->work_efficiency_iteration_maximum != 0) ||
      (policy->work_step_controller == SS_ANALYSIS_WORK_STEP_CLASSIC &&
       (policy->work_predictor_error_target != 0.0 ||
        policy->work_step_scale_minimum != 0.0 ||
        policy->work_step_scale_maximum != 0.0)) ||
      (policy->indirect_sensitivity_method !=
           SS_NONLINEAR_CONTROL_SENSITIVITY_REFERENCE_FORWARD &&
       policy->indirect_sensitivity_method !=
           SS_NONLINEAR_CONTROL_SENSITIVITY_STABLE_CENTRAL) ||
      (policy->method != SS_ANALYSIS_SSR_INDIRECT &&
       policy->indirect_sensitivity_method !=
           SS_NONLINEAR_CONTROL_SENSITIVITY_REFERENCE_FORWARD) ||
      (policy->indirect_sensitivity_method ==
           SS_NONLINEAR_CONTROL_SENSITIVITY_STABLE_CENTRAL &&
       (!ss_is_finite(policy->indirect_sensitivity_relative_step) ||
        policy->indirect_sensitivity_relative_step <= 0.0 ||
        policy->indirect_sensitivity_relative_step > 0.1))) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return SS_OK;
}

SsStatus ss_analysis_begin_attempt(SsAnalysisRecord *record,
                                   SsAnalysisPredictor predictor,
                                   PetscInt retry_ordinal, PetscReal target,
                                   PetscReal incoming_factor,
                                   PetscReal incoming_control, PetscReal incoming_work,
                                   PetscReal requested_step, PetscReal actual_step,
                                   SsAnalysisAttemptRecord **attempt) {
  if (record->attempt_count >= SS_ANALYSIS_ATTEMPT_CAPACITY) {
    return SS_ERROR_LIMIT;
  }
  *attempt = &record->attempts[record->attempt_count];
  memset(*attempt, 0, sizeof(**attempt));
  (*attempt)->attempt_ordinal = record->attempt_count;
  (*attempt)->parent_accepted_ordinal =
      record->point_count == 0U
          ? record->accepted_ordinal_offset
          : record->points[record->point_count - 1U].accepted_ordinal;
  (*attempt)->retry_ordinal = retry_ordinal;
  (*attempt)->predictor = predictor;
  (*attempt)->target = target;
  (*attempt)->incoming_factor = incoming_factor;
  (*attempt)->incoming_control = incoming_control;
  (*attempt)->incoming_work = incoming_work;
  (*attempt)->accepted_factor = NAN;
  (*attempt)->accepted_control = NAN;
  (*attempt)->accepted_work = NAN;
  (*attempt)->requested_step = requested_step;
  (*attempt)->actual_step = actual_step;
  (*attempt)->predictor_correction_norm = NAN;
  (*attempt)->accepted_increment_norm = NAN;
  (*attempt)->predictor_error = NAN;
  (*attempt)->work_efficiency_elasticity = NAN;
  (*attempt)->work_efficiency_iteration_budget = NAN;
  (*attempt)->work_step_scale = NAN;
  (*attempt)->next_work_step = NAN;
  (*attempt)->sensitivity_epsilon = NAN;
  (*attempt)->sensitivity_numerator = NAN;
  (*attempt)->sensitivity_denominator = NAN;
  (*attempt)->beta = NAN;
  (*attempt)->signed_scaled_slope = NAN;
  (*attempt)->nonlinear_outcome = SS_NONLINEAR_FAILED_NONFINITE;
  (*attempt)->primary_solve.outcome = SS_NONLINEAR_FAILED_NONFINITE;
  (*attempt)->secondary_solve.outcome = SS_NONLINEAR_FAILED_NONFINITE;
  ++record->attempt_count;
  return SS_OK;
}

static SsStatus
ss_analysis_attempt_transaction(const SsAnalysisProblem *problem,
                                SsAnalysisAttemptTransaction operation,
                                const SsAnalysisAttemptRecord *attempt) {
  SsStatus status;
  if (operation == NULL) {
    return SS_OK;
  }
  status = operation(problem->context, attempt->attempt_ordinal,
                     attempt->parent_accepted_ordinal);
  return ss_status_allreduce_max(problem->communicator, status);
}

SsStatus ss_analysis_open_attempt(const SsAnalysisProblem *problem,
                                  const SsAnalysisAttemptRecord *attempt) {
  return ss_analysis_attempt_transaction(problem, problem->operations.begin_attempt,
                                         attempt);
}

SsStatus ss_analysis_commit_attempt(const SsAnalysisProblem *problem,
                                    SsAnalysisRecord *record,
                                    SsAnalysisAttemptRecord *attempt) {
  SsStatus status = ss_analysis_attempt_transaction(
      problem, problem->operations.commit_attempt, attempt);
  if (status == SS_OK) {
    attempt->committed = 1;
    attempt->restartable = 1;
    ++record->commit_count;
  }
  return status;
}

SsStatus ss_analysis_rollback_attempt(const SsAnalysisProblem *problem,
                                      SsAnalysisRecord *record,
                                      SsAnalysisAttemptRecord *attempt) {
  SsStatus status = ss_analysis_attempt_transaction(
      problem, problem->operations.rollback_attempt, attempt);
  if (status == SS_OK) {
    attempt->rolled_back = 1;
    ++record->rollback_count;
  }
  return status;
}

SsStatus ss_analysis_append_point(const SsAnalysisProblem *problem,
                                  SsAnalysisRecord *record,
                                  size_t source_attempt_ordinal, PetscReal factor,
                                  PetscReal control, PetscReal work,
                                  PetscReal signed_scaled_slope, Vec state,
                                  int count_commit) {
  SsAnalysisPointRecord *point;
  PetscReal maximum = 0.0;
  SsStatus status = ss_analysis_displacement_maximum(problem, state, &maximum);
  if (status != SS_OK) {
    return status;
  }
  if (record->point_count >= SS_ANALYSIS_POINT_CAPACITY) {
    return SS_ERROR_LIMIT;
  }
  if (problem->operations.capture_accepted_state != NULL) {
    status = problem->operations.capture_accepted_state(
        problem->context, record->accepted_ordinal_offset + record->point_count, state);
    status = ss_status_allreduce_max(problem->communicator, status);
    if (status != SS_OK) {
      return status;
    }
  }
  point = &record->points[record->point_count];
  memset(point, 0, sizeof(*point));
  point->accepted_ordinal = record->accepted_ordinal_offset + record->point_count;
  point->source_attempt_ordinal = source_attempt_ordinal;
  point->factor = factor;
  point->control = control;
  point->work = work;
  point->displacement_maximum = maximum;
  point->signed_scaled_slope = signed_scaled_slope;
  point->restartable = 1;
  ++record->point_count;
  record->boundary_pending = 1;
  if (count_commit != 0) {
    ++record->accepted_step_count;
  }
  record->final_factor = factor;
  record->final_control = control;
  record->final_work = work;
  record->final_displacement_maximum = maximum;
  return SS_OK;
}

void ss_analysis_finish(SsAnalysisRecord *record, SsAnalysisStopReason reason,
                        int terminal_success) {
  record->stop_reason = reason;
  record->terminal_success = terminal_success;
}

SsStatus ss_analysis_observe_solve(const SsAnalysisProblem *problem,
                                   const SsAnalysisAttemptRecord *attempt,
                                   SsAnalysisSolveRole role,
                                   const SsNonlinearSolveRecord *solve) {
  SsStatus status;
  if (problem->observer.solve_completed == NULL) {
    return SS_OK;
  }
  status = problem->observer.solve_completed(problem->observer.context,
                                             attempt->attempt_ordinal, role, solve);
  return ss_status_allreduce_max(problem->communicator, status);
}

/*
 * Run the boundary observer for a pending accepted point with the controller
 * state a stop at this point would record; the provisional fields are
 * overwritten by the final values when the analysis ends.
 */
SsStatus ss_analysis_boundary(const SsAnalysisProblem *problem,
                              SsAnalysisRecord *record, PetscReal next_step,
                              PetscReal first_control,
                              SsAnalysisNextOperation next_operation) {
  SsStatus status;
  if (record->boundary_pending == 0) {
    return SS_OK;
  }
  record->boundary_pending = 0;
  if (problem->observer.accepted_boundary == NULL) {
    return SS_OK;
  }
  record->next_step = next_step;
  record->first_control = first_control;
  record->next_operation = next_operation;
  status = problem->observer.accepted_boundary(problem->observer.context, record);
  return ss_status_allreduce_max(problem->communicator, status);
}

void ss_analysis_capture_solve(SsAnalysisRecord *record,
                               SsAnalysisSolveSummary *summary,
                               const SsNonlinearSolveRecord *solve) {
  size_t captured_history_count;
  size_t index;
  memset(summary, 0, sizeof(*summary));
  captured_history_count = solve->history_count;
  /*
   * A failed linear setup or solve leaves its attempted Newton row at
   * history[history_count] because no nonlinear iteration was completed.
   * Preserve that row in the outer-analysis diagnostics even though it does
   * not contribute to the completed-iteration count.
   */
  if (solve->outcome == SS_NONLINEAR_FAILED_LINEAR &&
      captured_history_count < SS_NONLINEAR_HISTORY_CAPACITY) {
    ++captured_history_count;
  }
  summary->outcome = solve->outcome;
  summary->iterations = solve->iterations;
  summary->history_count = captured_history_count;
  summary->initial_convergence = solve->initial_convergence;
  summary->minimum_convergence = solve->initial_convergence;
  summary->final_convergence = solve->final_convergence;
  summary->final_relative_correction = solve->final_relative_correction;
  summary->final_regularization = solve->final_regularization;
  summary->final_alpha = NAN;
  for (index = 0U; index < captured_history_count; ++index) {
    const SsNonlinearIterationRecord *iteration = &solve->history[index];
    if (ss_is_finite(iteration->convergence_before) &&
        (!ss_is_finite(summary->minimum_convergence) ||
         iteration->convergence_before < summary->minimum_convergence)) {
      summary->minimum_convergence = iteration->convergence_before;
    }
    if (ss_is_finite(iteration->convergence_after) &&
        (!ss_is_finite(summary->minimum_convergence) ||
         iteration->convergence_after < summary->minimum_convergence)) {
      summary->minimum_convergence = iteration->convergence_after;
    }
    if (iteration->linear_setup.selected_identity[0] != '\0') {
      ++summary->linear_setup_count;
    }
    if (iteration->control_linear_solve.solve_count != 0U ||
        iteration->residual_linear_solve.solve_count != 0U) {
      if (iteration->control_linear_solve.solve_count != 0U) {
        ++summary->linear_solve_count;
        summary->first_linear_iteration_count +=
            (uint64_t)(iteration->control_linear_solve.iterations < 0
                           ? 0
                           : iteration->control_linear_solve.iterations);
        summary->first_linear_solve = iteration->control_linear_solve;
      }
      if (iteration->residual_linear_solve.solve_count != 0U) {
        ++summary->linear_solve_count;
        summary->second_linear_iteration_count +=
            (uint64_t)(iteration->residual_linear_solve.iterations < 0
                           ? 0
                           : iteration->residual_linear_solve.iterations);
        summary->second_linear_solve = iteration->residual_linear_solve;
      }
    } else if (iteration->linear_solve.solve_count != 0U) {
      ++summary->linear_solve_count;
      summary->first_linear_iteration_count +=
          (uint64_t)(iteration->linear_solve.iterations < 0
                         ? 0
                         : iteration->linear_solve.iterations);
      summary->first_linear_solve = iteration->linear_solve;
    }
    summary->final_alpha = iteration->alpha;
  }
  record->newton_iteration_count += (uint64_t)solve->iterations;
  record->linear_setup_count += summary->linear_setup_count;
  record->linear_solve_count += summary->linear_solve_count;
  record->linear_iteration_count +=
      summary->first_linear_iteration_count + summary->second_linear_iteration_count;
}

void ss_analysis_capture_augmented_diagnostics(SsAnalysisAttemptRecord *attempt,
                                               const SsNonlinearSolveRecord *solve) {
  if (attempt != NULL && solve != NULL &&
      solve->history_count < SS_NONLINEAR_HISTORY_CAPACITY) {
    const size_t history_index =
        (solve->outcome == SS_NONLINEAR_FAILED_LINEAR ||
         solve->outcome == SS_NONLINEAR_FAILED_SINGULAR_CONSTRAINT)
            ? solve->history_count
        : solve->history_count == 0U ? SS_NONLINEAR_HISTORY_CAPACITY
                                     : solve->history_count - 1U;
    if (history_index < SS_NONLINEAR_HISTORY_CAPACITY) {
      const SsNonlinearIterationRecord *iteration = &solve->history[history_index];
      attempt->sensitivity_epsilon = iteration->control_sensitivity_epsilon;
      attempt->sensitivity_numerator = iteration->augmented.numerator;
      attempt->sensitivity_denominator = iteration->augmented.denominator;
    }
  }
}

void ss_analysis_observe_start(const SsAnalysisProblem *problem,
                               SsAnalysisSolveRole role, PetscReal target) {
  if (problem->observer.solve_started != NULL) {
    problem->observer.solve_started(problem->observer.context, role, target);
  }
}

SsStatus ss_analysis_run(const SsAnalysisProblem *problem,
                         const SsAnalysisPolicy *policy, Vec initial_state,
                         Vec accepted_state, SsAnalysisRecord *record) {
  SsStatus status;
  if (record == NULL || ss_analysis_policy_validate(policy) != SS_OK ||
      ss_analysis_problem_validate(problem, initial_state, accepted_state) != SS_OK) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(record, 0, sizeof(*record));
  record->method = policy->method;
  ss_analysis_set_identities(record, policy);
  record->final_factor = NAN;
  record->final_control = NAN;
  record->final_work = NAN;
  record->final_displacement_maximum = NAN;
  switch (policy->method) {
  case SS_ANALYSIS_LIMIT_LOAD:
    status = ss_analysis_run_limit_load(problem, policy, NULL, initial_state,
                                        accepted_state, NULL, record);
    break;
  case SS_ANALYSIS_SSR_DIRECT:
    status = ss_analysis_run_direct(problem, policy, initial_state, accepted_state,
                                    NULL, record);
    break;
  case SS_ANALYSIS_SSR_INDIRECT:
    status = ss_analysis_run_indirect(problem, policy, NULL, initial_state,
                                      accepted_state, NULL, record);
    break;
  default:
    status = SS_ERROR_INVALID_ARGUMENT;
    break;
  }
  if (status == SS_OK) {
    status = ss_analysis_boundary(problem, record, record->next_step,
                                  record->first_control, record->next_operation);
  }
  return status;
}

static int ss_analysis_resume_state_valid(const SsAnalysisPolicy *policy,
                                          const SsAnalysisResumeState *resume) {
  size_t expected_ordinal;
  if (policy == NULL || resume == NULL || resume->method != policy->method ||
      resume->accepted_step_offset == 0U ||
      resume->accepted_step_offset >= (size_t)policy->accepted_step_limit ||
      /*
       * Direct and indirect SSR can accept multiple numerically shifted
       * seeds before finding a distinct pair. SECOND_SEED is therefore valid
       * at every nonzero accepted offset, not only immediately after offset 1.
       */
      (resume->next_operation != SS_ANALYSIS_NEXT_SECOND_SEED &&
       resume->next_operation != SS_ANALYSIS_NEXT_CONTINUATION_ATTEMPT) ||
      !ss_is_finite(resume->previous_factor) ||
      !ss_is_finite(resume->previous_control) || !ss_is_finite(resume->previous_work) ||
      !ss_is_finite(resume->current_factor) || !ss_is_finite(resume->current_control) ||
      !ss_is_finite(resume->current_work) ||
      (!ss_is_finite(resume->current_signed_scaled_slope) &&
       !isnan((double)resume->current_signed_scaled_slope)) ||
      !ss_is_finite(resume->first_control) || !ss_is_finite(resume->next_step) ||
      resume->next_step <= 0.0 || resume->retry_count < 0 ||
      (policy->work_step_controller == SS_ANALYSIS_WORK_STEP_WORK_EFFICIENCY &&
       (resume->work_efficiency_previous_iterations < 0 ||
        resume->work_efficiency_previous_iterations >
            policy->work_efficiency_iteration_maximum ||
        (resume->work_efficiency_elasticity_ready != 0 &&
         resume->work_efficiency_elasticity_ready != 1) ||
        !ss_is_finite(resume->work_efficiency_elasticity) ||
        (resume->work_efficiency_previous_iterations == 0 &&
         resume->work_efficiency_elasticity_ready != 0) ||
        (resume->work_efficiency_elasticity_ready == 0 &&
         resume->work_efficiency_elasticity != 0.0) ||
        (resume->next_operation != SS_ANALYSIS_NEXT_CONTINUATION_ATTEMPT &&
         (resume->work_efficiency_previous_iterations != 0 ||
          resume->work_efficiency_elasticity_ready != 0)))) ||
      (policy->work_step_controller != SS_ANALYSIS_WORK_STEP_WORK_EFFICIENCY &&
       (resume->work_efficiency_previous_iterations != 0 ||
        resume->work_efficiency_elasticity != 0.0 ||
        resume->work_efficiency_elasticity_ready != 0)) ||
      (resume->method == SS_ANALYSIS_LIMIT_LOAD &&
       resume->next_operation != SS_ANALYSIS_NEXT_CONTINUATION_ATTEMPT)) {
    return 0;
  }
  expected_ordinal = resume->method == SS_ANALYSIS_LIMIT_LOAD
                         ? resume->accepted_step_offset
                         : resume->accepted_step_offset - 1U;
  return resume->accepted_ordinal == expected_ordinal;
}

SsStatus ss_analysis_resume(const SsAnalysisProblem *problem,
                            const SsAnalysisPolicy *policy, Vec previous_state,
                            Vec current_state, Vec accepted_state,
                            const SsAnalysisResumeState *resume,
                            SsAnalysisRecord *record) {
  PetscInt previous_size = 0;
  PetscInt current_size = 0;
  SsStatus status;
  if (record == NULL || previous_state == NULL ||
      ss_analysis_policy_validate(policy) != SS_OK ||
      !ss_analysis_resume_state_valid(policy, resume) ||
      ss_analysis_problem_validate(problem, current_state, accepted_state) != SS_OK ||
      !ss_analysis_same_communicator(problem->communicator,
                                     (PetscObject)previous_state) ||
      VecGetSize(previous_state, &previous_size) != PETSC_SUCCESS ||
      VecGetSize(current_state, &current_size) != PETSC_SUCCESS ||
      previous_size != current_size) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(record, 0, sizeof(*record));
  record->method = policy->method;
  record->accepted_ordinal_offset = resume->accepted_ordinal;
  record->resumed_from_accepted_state = 1;
  ss_analysis_set_identities(record, policy);
  record->final_factor = NAN;
  record->final_control = NAN;
  record->final_work = NAN;
  record->final_displacement_maximum = NAN;
  switch (policy->method) {
  case SS_ANALYSIS_LIMIT_LOAD:
    status = ss_analysis_run_limit_load(problem, policy, previous_state, current_state,
                                        accepted_state, resume, record);
    break;
  case SS_ANALYSIS_SSR_DIRECT:
    status = ss_analysis_run_direct(problem, policy, current_state, accepted_state,
                                    resume, record);
    break;
  case SS_ANALYSIS_SSR_INDIRECT:
    status = ss_analysis_run_indirect(problem, policy, previous_state, current_state,
                                      accepted_state, resume, record);
    break;
  default:
    status = SS_ERROR_INVALID_ARGUMENT;
    break;
  }
  if (status == SS_OK) {
    status = ss_analysis_boundary(problem, record, record->next_step,
                                  record->first_control, record->next_operation);
  }
  return status;
}
