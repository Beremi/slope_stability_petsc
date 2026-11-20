/* Load-controlled and direct strength-reduction continuation.
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

SsStatus ss_analysis_direct_point(const SsAnalysisProblem *problem,
                                  const SsAnalysisPolicy *policy, PetscReal factor,
                                  PetscReal factor_step, Vec incoming, Vec solution,
                                  SsAnalysisDirectPointRecord *record) {
  Vec primary_solution = NULL;
  Vec sensitivity_initial = NULL;
  Vec sensitivity_solution = NULL;
  PetscReal epsilon;
  PetscReal beta;
  PetscReal potential = NAN;
  PetscReal perturbed_potential = NAN;
  PetscReal work = NAN;
  PetscInt load_size = 0;
  PetscInt incoming_size = 0;
  PetscInt solution_size = 0;
  SsStatus status;
  if (problem == NULL || policy == NULL || incoming == NULL || solution == NULL ||
      record == NULL || ss_analysis_policy_validate(policy) != SS_OK ||
      policy->method != SS_ANALYSIS_SSR_DIRECT ||
      problem->communicator == MPI_COMM_NULL || problem->reference_load == NULL ||
      problem->operations.solve_fixed == NULL ||
      problem->operations.total_potential == NULL || !ss_is_finite(factor) ||
      factor <= 0.0 || !ss_is_finite(factor_step) || factor_step <= 0.0 ||
      !ss_analysis_same_communicator(problem->communicator, (PetscObject)incoming) ||
      !ss_analysis_same_communicator(problem->communicator, (PetscObject)solution) ||
      VecGetSize(problem->reference_load, &load_size) != PETSC_SUCCESS ||
      VecGetSize(incoming, &incoming_size) != PETSC_SUCCESS ||
      VecGetSize(solution, &solution_size) != PETSC_SUCCESS || load_size <= 0 ||
      load_size != incoming_size || load_size != solution_size) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  epsilon = 100.0 * policy->nonlinear_tolerance;
  if (!ss_is_finite(epsilon) || epsilon <= 0.0 || epsilon >= factor) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  beta = fmin(1.0, epsilon / factor_step);
  memset(record, 0, sizeof(*record));
  record->outcome = SS_NONLINEAR_FAILED_NONFINITE;
  record->primary_solve.outcome = SS_NONLINEAR_FAILED_NONFINITE;
  record->sensitivity_solve.outcome = SS_NONLINEAR_FAILED_NONFINITE;
  record->factor = factor;
  record->epsilon = epsilon;
  record->beta = beta;
  record->potential = NAN;
  record->perturbed_potential = NAN;
  record->direct_control = NAN;
  record->external_work = NAN;
  Vec *primary[] = {&primary_solution};
  status = ss_vec_duplicate_collective(incoming, primary, 1U);
  if (status != SS_OK) {
    goto cleanup;
  }
  ss_analysis_observe_start(problem, SS_ANALYSIS_SOLVE_PRIMARY, factor);
  status = problem->operations.solve_fixed(problem->context, factor, incoming,
                                           primary_solution, &record->primary_solve);
  status = ss_status_allreduce_max(problem->communicator, status);
  if (status == SS_OK) {
    status = ss_analysis_collective_nonlinear_outcome(problem->communicator,
                                                      record->primary_solve.outcome,
                                                      &record->primary_solve.outcome);
  }
  if (status != SS_OK ||
      !ss_analysis_nonlinear_succeeded(record->primary_solve.outcome)) {
    record->outcome = record->primary_solve.outcome;
    goto cleanup;
  }
  status = problem->operations.total_potential(problem->context, factor,
                                               primary_solution, &potential);
  status = ss_status_allreduce_max(problem->communicator, status);
  if (status != SS_OK) {
    status = status == SS_ERROR_INTEGRITY ? SS_OK : status;
    goto cleanup;
  }
  if (!ss_is_finite(potential)) {
    status = SS_OK;
    goto cleanup;
  }
  status = ss_analysis_collective_real(problem->communicator, potential, &potential);
  if (status != SS_OK) {
    goto cleanup;
  }
  status = ss_analysis_external_work(problem, primary_solution, &work);
  if (status != SS_OK || !ss_is_finite(potential)) {
    status = status == SS_OK || status == SS_ERROR_INTEGRITY ? SS_OK : status;
    goto cleanup;
  }
  Vec *sensitivity[] = {&sensitivity_initial, &sensitivity_solution};
  status = ss_vec_duplicate_collective(incoming, sensitivity, 2U);
  if (status == SS_OK &&
      (VecCopy(primary_solution, sensitivity_initial) != PETSC_SUCCESS ||
       VecScale(sensitivity_initial, 1.0 - beta) != PETSC_SUCCESS ||
       VecAXPY(sensitivity_initial, beta, incoming) != PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(problem->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  ss_analysis_observe_start(problem, SS_ANALYSIS_SOLVE_SECONDARY, factor - epsilon);
  status = problem->operations.solve_fixed(problem->context, factor - epsilon,
                                           sensitivity_initial, sensitivity_solution,
                                           &record->sensitivity_solve);
  status = ss_status_allreduce_max(problem->communicator, status);
  if (status == SS_OK) {
    status = ss_analysis_collective_nonlinear_outcome(
        problem->communicator, record->sensitivity_solve.outcome,
        &record->sensitivity_solve.outcome);
  }
  if (status != SS_OK ||
      !ss_analysis_nonlinear_succeeded(record->sensitivity_solve.outcome)) {
    record->outcome = record->sensitivity_solve.outcome;
    goto cleanup;
  }
  status = problem->operations.total_potential(
      problem->context, factor - epsilon, sensitivity_solution, &perturbed_potential);
  status = ss_status_allreduce_max(problem->communicator, status);
  if (status != SS_OK) {
    if (status == SS_ERROR_INTEGRITY) {
      status = SS_OK;
    }
    goto cleanup;
  }
  if (!ss_is_finite(perturbed_potential)) {
    status = SS_OK;
    goto cleanup;
  }
  status = ss_analysis_collective_real(problem->communicator, perturbed_potential,
                                       &perturbed_potential);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (!ss_is_finite(perturbed_potential) ||
      !ss_is_finite((perturbed_potential - potential) / epsilon)) {
    status = SS_OK;
    goto cleanup;
  }
  record->potential = potential;
  record->perturbed_potential = perturbed_potential;
  record->direct_control = (perturbed_potential - potential) / epsilon;
  record->external_work = work;
  record->outcome = record->primary_solve.outcome > record->sensitivity_solve.outcome
                        ? record->primary_solve.outcome
                        : record->sensitivity_solve.outcome;
  if (VecCopy(primary_solution, solution) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }

cleanup:
  (void)VecDestroy(&sensitivity_solution);
  (void)VecDestroy(&sensitivity_initial);
  (void)VecDestroy(&primary_solution);
  return status;
}

SsStatus ss_analysis_run_limit_load(const SsAnalysisProblem *problem,
                                    const SsAnalysisPolicy *policy, Vec previous_state,
                                    Vec initial_state, Vec accepted_state,
                                    const SsAnalysisResumeState *resume,
                                    SsAnalysisRecord *record) {
  Vec previous = NULL;
  Vec current = NULL;
  Vec predictor = NULL;
  Vec candidate = NULL;
  SsNonlinearSolveRecord *solve = NULL;
  PetscReal current_factor = 0.0;
  PetscReal previous_work = 0.0;
  PetscReal current_work = 0.0;
  PetscReal work_step = policy->work_step_initial;
  PetscReal previous_factor_increment = NAN;
  PetscInt retries = 0;
  SsAnalysisAttemptRecord *open_attempt = NULL;
  SsStatus status = SS_OK;
  const size_t accepted_step_offset =
      resume == NULL ? 0U : resume->accepted_step_offset;
  if (problem->operations.solve_work_controlled == NULL ||
      (resume != NULL && previous_state == NULL)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  Vec *work_vectors[] = {&previous, &current, &predictor, &candidate};
  status = ss_vec_duplicate_collective(initial_state, work_vectors, 4U);
  if (status == SS_OK && (VecCopy(resume == NULL ? initial_state : previous_state,
                                  previous) != PETSC_SUCCESS ||
                          VecCopy(initial_state, current) != PETSC_SUCCESS ||
                          VecCopy(initial_state, accepted_state) != PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    solve = calloc(1U, sizeof(*solve));
    if (solve == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(problem->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (resume != NULL) {
    current_factor = resume->current_factor;
    previous_work = resume->previous_work;
    current_work = resume->current_work;
    work_step = resume->next_step;
    previous_factor_increment = resume->current_factor - resume->previous_factor;
    retries = resume->retry_count;
  }
  status = ss_analysis_append_point(
      problem, record, 0U, resume == NULL ? 0.0 : current_factor,
      resume == NULL ? 0.0 : resume->current_control,
      resume == NULL ? 0.0 : current_work, NAN, current, 0);
  if (status != SS_OK) {
    goto cleanup;
  }
  while (accepted_step_offset + record->accepted_step_count <
         (size_t)policy->accepted_step_limit) {
    SsAnalysisAttemptRecord *attempt = NULL;
    PetscReal target = fmin(current_work + work_step, policy->work_maximum);
    PetscReal actual_step = target - current_work;
    PetscReal accepted_factor = NAN;
    PetscReal factor_increment;
    PetscReal slope = NAN;
    Vec incoming = current;
    int cancelled = 0;
    status = ss_analysis_boundary(problem, record, work_step, 0.0,
                                  SS_ANALYSIS_NEXT_CONTINUATION_ATTEMPT);
    if (status != SS_OK) {
      goto cleanup;
    }
    status = ss_analysis_collective_cancelled(problem, &cancelled);
    if (status != SS_OK) {
      goto cleanup;
    }
    if (cancelled != 0) {
      ss_analysis_finish(record, SS_ANALYSIS_STOP_CANCELLED, 0);
      break;
    }
    if (!ss_is_finite(actual_step) || actual_step <= 0.0) {
      ss_analysis_finish(record, SS_ANALYSIS_STOP_WORK_HORIZON_REACHED, 1);
      break;
    }
    status = ss_analysis_begin_attempt(
        record,
        accepted_step_offset + record->accepted_step_count == 0U
            ? SS_ANALYSIS_PREDICTOR_CURRENT_STATE
            : SS_ANALYSIS_PREDICTOR_DISPLACEMENT_SECANT,
        retries, target, current_factor, current_work, current_work, work_step,
        actual_step, &attempt);
    if (status != SS_OK) {
      goto cleanup;
    }
    status = ss_analysis_open_attempt(problem, attempt);
    if (status != SS_OK) {
      goto cleanup;
    }
    open_attempt = attempt;
    if (accepted_step_offset + record->accepted_step_count > 0U) {
      PetscReal denominator = current_work - previous_work;
      PetscReal ratio;
      if (!ss_is_finite(denominator) || fabs(denominator) <= DBL_MIN) {
        status = SS_ERROR_INTEGRITY;
        goto cleanup;
      }
      ratio = actual_step / denominator;
      if (!ss_is_finite(ratio) || VecCopy(current, predictor) != PETSC_SUCCESS ||
          VecAXPY(predictor, ratio, current) != PETSC_SUCCESS ||
          VecAXPY(predictor, -ratio, previous) != PETSC_SUCCESS) {
        status = SS_ERROR_INTEGRITY;
        goto cleanup;
      }
      incoming = predictor;
    }
    memset(solve, 0, sizeof(*solve));
    ss_analysis_observe_start(problem, SS_ANALYSIS_SOLVE_PRIMARY, target);
    status = problem->operations.solve_work_controlled(
        problem->context, SS_ANALYSIS_LIMIT_LOAD, target, current_factor, incoming,
        candidate, &accepted_factor, solve);
    status = ss_status_allreduce_max(problem->communicator, status);
    if (status == SS_OK) {
      status = ss_analysis_collective_nonlinear_outcome(
          problem->communicator, solve->outcome, &solve->outcome);
    }
    if (status != SS_OK) {
      goto cleanup;
    }
    attempt->newton_iterations = solve->iterations;
    attempt->nonlinear_outcome = solve->outcome;
    ss_analysis_capture_solve(record, &attempt->primary_solve, solve);
    ss_analysis_capture_augmented_diagnostics(attempt, solve);
    status =
        ss_analysis_observe_solve(problem, attempt, SS_ANALYSIS_SOLVE_PRIMARY, solve);
    if (status != SS_OK) {
      goto cleanup;
    }
    if (ss_analysis_nonlinear_succeeded(solve->outcome)) {
      status = ss_analysis_collective_real(problem->communicator, accepted_factor,
                                           &accepted_factor);
      if (status != SS_OK) {
        goto cleanup;
      }
    }
    if (!ss_analysis_nonlinear_succeeded(solve->outcome) ||
        !ss_is_finite(accepted_factor)) {
      status = ss_analysis_rollback_attempt(problem, record, attempt);
      if (status != SS_OK) {
        goto cleanup;
      }
      open_attempt = NULL;
      ++retries;
      if (solve->outcome == SS_NONLINEAR_CANCELLED) {
        ss_analysis_finish(record, SS_ANALYSIS_STOP_CANCELLED, 0);
        break;
      }
      work_step *= 0.5;
      if (retries >= policy->consecutive_retry_limit) {
        ss_analysis_finish(record, SS_ANALYSIS_STOP_CONSECUTIVE_RETRY_LIMIT, 0);
        break;
      }
      if (work_step < policy->work_step_minimum) {
        ss_analysis_finish(record, SS_ANALYSIS_STOP_MINIMUM_STEP_REACHED, 0);
        break;
      }
      continue;
    }
    factor_increment = accepted_factor - current_factor;
    if (!ss_is_finite(factor_increment)) {
      status = ss_analysis_rollback_attempt(problem, record, attempt);
      if (status != SS_OK) {
        goto cleanup;
      }
      open_attempt = NULL;
      ss_analysis_finish(record, SS_ANALYSIS_STOP_NONFINITE, 0);
      break;
    }
    if (actual_step != 0.0) {
      slope = factor_increment / actual_step * target;
    }
    attempt->accepted_factor = accepted_factor;
    attempt->accepted_control = target;
    attempt->accepted_work = target;
    attempt->signed_scaled_slope = slope;
    attempt->successful = 1;
    status = ss_analysis_commit_attempt(problem, record, attempt);
    if (status != SS_OK) {
      goto cleanup;
    }
    open_attempt = NULL;
    if (VecCopy(current, previous) != PETSC_SUCCESS ||
        VecCopy(candidate, current) != PETSC_SUCCESS ||
        VecCopy(candidate, accepted_state) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
      goto cleanup;
    }
    current_factor = accepted_factor;
    previous_work = current_work;
    current_work = target;
    status = ss_analysis_append_point(problem, record, attempt->attempt_ordinal,
                                      current_factor, current_work, current_work, slope,
                                      current, 1);
    if (status != SS_OK) {
      goto cleanup;
    }
    retries = 0;
    if (policy->factor_increment_stop > 0.0 &&
        factor_increment < policy->factor_increment_stop) {
      ss_analysis_finish(record, SS_ANALYSIS_STOP_MINIMUM_STEP_REACHED, 1);
      break;
    }
    if (current_work >= policy->work_maximum) {
      ss_analysis_finish(record, SS_ANALYSIS_STOP_WORK_HORIZON_REACHED, 1);
      break;
    }
    if (solve->iterations < 20 && ss_is_finite(previous_factor_increment) &&
        factor_increment < 0.9 * previous_factor_increment) {
      work_step *= 2.0;
    }
    previous_factor_increment = factor_increment;
  }
  if (record->stop_reason == SS_ANALYSIS_STOP_NONE) {
    ss_analysis_finish(record, SS_ANALYSIS_STOP_ACCEPTED_STEP_LIMIT, 1);
  }
  record->next_step = work_step;
  record->first_control = 0.0;
  record->next_operation = record->stop_reason == SS_ANALYSIS_STOP_ACCEPTED_STEP_LIMIT
                               ? SS_ANALYSIS_NEXT_CONTINUATION_ATTEMPT
                               : SS_ANALYSIS_NEXT_TERMINAL;

cleanup:
  if (open_attempt != NULL) {
    SsStatus rollback_status =
        ss_analysis_rollback_attempt(problem, record, open_attempt);
    if (status == SS_OK) {
      status = rollback_status;
    }
  }
  free(solve);
  (void)VecDestroy(&candidate);
  (void)VecDestroy(&predictor);
  (void)VecDestroy(&current);
  (void)VecDestroy(&previous);
  return status;
}

static SsNonlinearOutcome
ss_analysis_direct_outcome(const SsAnalysisDirectPointRecord *direct) {
  if (!ss_analysis_nonlinear_succeeded(direct->outcome)) {
    return direct->outcome;
  }
  if (!ss_analysis_nonlinear_succeeded(direct->primary_solve.outcome)) {
    return direct->primary_solve.outcome;
  }
  return direct->sensitivity_solve.outcome;
}

static PetscInt
ss_analysis_direct_iterations(const SsAnalysisDirectPointRecord *direct) {
  return direct->primary_solve.iterations + direct->sensitivity_solve.iterations;
}

SsStatus ss_analysis_run_direct(const SsAnalysisProblem *problem,
                                const SsAnalysisPolicy *policy, Vec initial_state,
                                Vec accepted_state, const SsAnalysisResumeState *resume,
                                SsAnalysisRecord *record) {
  Vec current = NULL;
  Vec candidate = NULL;
  SsAnalysisDirectPointRecord *direct = NULL;
  PetscReal current_factor = policy->factor_initial;
  PetscReal current_control = NAN;
  PetscReal current_work = NAN;
  PetscReal first_control = NAN;
  PetscReal factor_step = policy->factor_step_initial;
  PetscReal previous_control_increment = NAN;
  PetscInt retries = 0;
  int distinct_seed_ready = 0;
  SsAnalysisAttemptRecord *open_attempt = NULL;
  SsStatus status = SS_OK;
  const size_t accepted_step_offset =
      resume == NULL ? 0U : resume->accepted_step_offset;
  if (problem->operations.solve_fixed == NULL ||
      problem->operations.total_potential == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  Vec *work_vectors[] = {&current, &candidate};
  status = ss_vec_duplicate_collective(initial_state, work_vectors, 2U);
  if (status == SS_OK &&
      (VecCopy(initial_state, current) != PETSC_SUCCESS ||
       (resume != NULL && VecCopy(initial_state, accepted_state) != PETSC_SUCCESS))) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    direct = calloc(1U, sizeof(*direct));
    if (direct == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(problem->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (resume != NULL) {
    current_factor = resume->current_factor;
    current_control = resume->current_control;
    current_work = resume->current_work;
    first_control = resume->first_control;
    factor_step = resume->next_step;
    previous_control_increment =
        resume->next_operation == SS_ANALYSIS_NEXT_CONTINUATION_ATTEMPT
            ? resume->current_control - resume->previous_control
            : NAN;
    retries = resume->retry_count;
    distinct_seed_ready =
        resume->next_operation == SS_ANALYSIS_NEXT_CONTINUATION_ATTEMPT;
    status = ss_analysis_append_point(problem, record, 0U, current_factor,
                                      current_control, current_work,
                                      resume->current_signed_scaled_slope, current, 0);
    if (status != SS_OK) {
      goto cleanup;
    }
  }
  while (accepted_step_offset + record->accepted_step_count <
         (size_t)policy->accepted_step_limit) {
    SsAnalysisAttemptRecord *attempt = NULL;
    const size_t total_point_count = accepted_step_offset + record->accepted_step_count;
    PetscReal target = total_point_count == 0U
                           ? current_factor
                           : fmin(current_factor + factor_step, policy->factor_maximum);
    PetscReal actual_step =
        total_point_count == 0U ? factor_step : target - current_factor;
    PetscReal sensitivity_step = total_point_count == 0U ? factor_step : actual_step;
    PetscReal control_increment = NAN;
    PetscReal slope = NAN;
    SsNonlinearOutcome outcome;
    int cancelled = 0;
    int shifted_seed = 0;
    status = ss_analysis_boundary(problem, record, factor_step, first_control,
                                  distinct_seed_ready != 0
                                      ? SS_ANALYSIS_NEXT_CONTINUATION_ATTEMPT
                                      : SS_ANALYSIS_NEXT_SECOND_SEED);
    if (status != SS_OK) {
      goto cleanup;
    }
    status = ss_analysis_collective_cancelled(problem, &cancelled);
    if (status != SS_OK) {
      goto cleanup;
    }
    if (cancelled != 0) {
      ss_analysis_finish(record, SS_ANALYSIS_STOP_CANCELLED, 0);
      break;
    }
    if (total_point_count != 0U && actual_step <= 0.0) {
      ss_analysis_finish(record, SS_ANALYSIS_STOP_MINIMUM_STEP_REACHED,
                         distinct_seed_ready != 0);
      break;
    }
    status = ss_analysis_begin_attempt(
        record, SS_ANALYSIS_PREDICTOR_PAIRED_BETA, retries, target,
        total_point_count == 0U ? target : current_factor, current_control,
        current_work, factor_step, actual_step, &attempt);
    if (status != SS_OK) {
      goto cleanup;
    }
    status = ss_analysis_open_attempt(problem, attempt);
    if (status != SS_OK) {
      goto cleanup;
    }
    open_attempt = attempt;
    memset(direct, 0, sizeof(*direct));
    status = ss_analysis_direct_point(problem, policy, target, sensitivity_step,
                                      current, candidate, direct);
    if (status != SS_OK) {
      goto cleanup;
    }
    outcome = ss_analysis_direct_outcome(direct);
    attempt->newton_iterations = ss_analysis_direct_iterations(direct);
    attempt->nonlinear_outcome = outcome;
    status = ss_analysis_observe_solve(problem, attempt, SS_ANALYSIS_SOLVE_PRIMARY,
                                       &direct->primary_solve);
    if (status == SS_OK && (direct->sensitivity_solve.history_count != 0U ||
                            direct->sensitivity_solve.iterations != 0)) {
      status = ss_analysis_observe_solve(problem, attempt, SS_ANALYSIS_SOLVE_SECONDARY,
                                         &direct->sensitivity_solve);
    }
    if (status != SS_OK) {
      goto cleanup;
    }
    ss_analysis_capture_solve(record, &attempt->primary_solve, &direct->primary_solve);
    ss_analysis_capture_solve(record, &attempt->secondary_solve,
                              &direct->sensitivity_solve);
    attempt->sensitivity_epsilon = direct->epsilon;
    attempt->sensitivity_numerator = direct->perturbed_potential - direct->potential;
    attempt->sensitivity_denominator = direct->epsilon;
    attempt->beta = direct->beta;
    if (!ss_analysis_nonlinear_succeeded(outcome) ||
        !ss_is_finite(direct->direct_control) || !ss_is_finite(direct->external_work)) {
      status = ss_analysis_rollback_attempt(problem, record, attempt);
      if (status != SS_OK) {
        goto cleanup;
      }
      open_attempt = NULL;
      ++retries;
      if (outcome == SS_NONLINEAR_CANCELLED) {
        ss_analysis_finish(record, SS_ANALYSIS_STOP_CANCELLED, 0);
        break;
      }
      factor_step *= 0.5;
      if (retries >= policy->consecutive_retry_limit) {
        ss_analysis_finish(record, SS_ANALYSIS_STOP_CONSECUTIVE_RETRY_LIMIT, 0);
        break;
      }
      if (factor_step < policy->factor_step_minimum) {
        ss_analysis_finish(record, SS_ANALYSIS_STOP_MINIMUM_STEP_REACHED,
                           distinct_seed_ready != 0);
        break;
      }
      continue;
    }
    if (total_point_count != 0U) {
      control_increment = direct->direct_control - current_control;
      shifted_seed = ss_is_finite(control_increment) && distinct_seed_ready == 0 &&
                     fabs(control_increment) / fmax(1.0, current_control) < 1.0e-5;
      if (!ss_is_finite(control_increment) ||
          (control_increment < 0.0 && shifted_seed == 0)) {
        status = ss_analysis_rollback_attempt(problem, record, attempt);
        if (status != SS_OK) {
          goto cleanup;
        }
        open_attempt = NULL;
        ++retries;
        factor_step *= 0.5;
        if (retries >= policy->consecutive_retry_limit) {
          ss_analysis_finish(record, SS_ANALYSIS_STOP_CONSECUTIVE_RETRY_LIMIT, 0);
          break;
        }
        if (factor_step < policy->factor_step_minimum) {
          ss_analysis_finish(record, SS_ANALYSIS_STOP_MINIMUM_STEP_REACHED,
                             distinct_seed_ready != 0);
          break;
        }
        continue;
      }
      if (shifted_seed == 0 && control_increment > DBL_MIN) {
        slope =
            actual_step / control_increment * (direct->direct_control - first_control);
      }
    }
    attempt->accepted_factor = target;
    attempt->accepted_control = direct->direct_control;
    attempt->accepted_work = direct->external_work;
    attempt->signed_scaled_slope = slope;
    attempt->successful = 1;
    status = ss_analysis_commit_attempt(problem, record, attempt);
    if (status != SS_OK) {
      goto cleanup;
    }
    open_attempt = NULL;
    if (VecCopy(candidate, current) != PETSC_SUCCESS ||
        VecCopy(candidate, accepted_state) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
      goto cleanup;
    }
    current_factor = target;
    current_control = direct->direct_control;
    current_work = direct->external_work;
    if (total_point_count == 0U) {
      first_control = current_control;
    }
    status = ss_analysis_append_point(problem, record, attempt->attempt_ordinal,
                                      current_factor, current_control,
                                      direct->external_work, slope, current, 1);
    if (status != SS_OK) {
      goto cleanup;
    }
    retries = 0;
    if (accepted_step_offset + record->accepted_step_count == 1U) {
      continue;
    }
    if (distinct_seed_ready == 0) {
      if (shifted_seed != 0) {
        first_control = current_control;
        continue;
      }
      distinct_seed_ready = 1;
      previous_control_increment = control_increment;
    } else if (ss_is_finite(control_increment)) {
      if (ss_is_finite(previous_control_increment) &&
          control_increment > 1.5 * previous_control_increment) {
        factor_step *= 0.5;
      }
      previous_control_increment = control_increment;
    }
    if (policy->direct_control_maximum > 0.0 &&
        current_control >= policy->direct_control_maximum) {
      ss_analysis_finish(record, SS_ANALYSIS_STOP_DIRECT_CONTROL_HORIZON_REACHED, 1);
      break;
    }
    if (policy->scaled_slope_tolerance > 0.0 && ss_is_finite(slope) &&
        slope <= policy->scaled_slope_tolerance) {
      ss_analysis_finish(record, SS_ANALYSIS_STOP_SCALED_SLOPE_REACHED, 1);
      break;
    }
  }
  if (record->stop_reason == SS_ANALYSIS_STOP_NONE) {
    ss_analysis_finish(record, SS_ANALYSIS_STOP_ACCEPTED_STEP_LIMIT,
                       distinct_seed_ready != 0);
  }
  record->next_step = factor_step;
  record->first_control = first_control;
  record->next_operation = record->stop_reason != SS_ANALYSIS_STOP_ACCEPTED_STEP_LIMIT
                               ? SS_ANALYSIS_NEXT_TERMINAL
                           : distinct_seed_ready != 0
                               ? SS_ANALYSIS_NEXT_CONTINUATION_ATTEMPT
                               : SS_ANALYSIS_NEXT_SECOND_SEED;

cleanup:
  if (open_attempt != NULL) {
    SsStatus rollback_status =
        ss_analysis_rollback_attempt(problem, record, open_attempt);
    if (status == SS_OK) {
      status = rollback_status;
    }
  }
  free(direct);
  (void)VecDestroy(&candidate);
  (void)VecDestroy(&current);
  return status;
}
