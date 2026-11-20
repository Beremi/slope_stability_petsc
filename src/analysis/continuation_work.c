/* Indirect strength-reduction continuation and work-step controls.
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

static PetscReal ss_analysis_clamp(PetscReal value, PetscReal minimum,
                                   PetscReal maximum) {
  return fmin(maximum, fmax(minimum, value));
}

static PetscReal ss_analysis_adaptive_success_scale(
    const SsAnalysisPolicy *policy, PetscReal predictor_error,
    PetscReal factor_increment, PetscReal previous_factor_increment) {
  PetscReal scale = policy->work_step_scale_maximum;
  if (ss_is_finite(predictor_error) && predictor_error > DBL_MIN) {
    scale = sqrt(policy->work_predictor_error_target / predictor_error);
  }
  scale = ss_analysis_clamp(scale, policy->work_step_scale_minimum,
                            policy->work_step_scale_maximum);
  /* Retain the reference curve-shape trigger for increasing the work step. */
  if (!(factor_increment < 0.9 * previous_factor_increment) && scale > 1.0) {
    scale = 1.0;
  }
  /*
   * Keep predictor-error continuation independent of the selected linear
   * accelerator.  Deflation and preconditioning may change the number of
   * inexact Newton corrections even when they converge to the same accepted
   * state.  Work-efficiency is the explicit effort-aware controller.
   */
  return ss_analysis_clamp(scale, policy->work_step_scale_minimum,
                           policy->work_step_scale_maximum);
}

typedef struct {
  PetscReal previous_step;
  PetscInt previous_iterations;
  PetscReal elasticity;
  int elasticity_ready;
} SsAnalysisWorkEfficiencyState;

int ss_analysis_work_step_uses_predictor_error(
    SsAnalysisWorkStepController controller) {
  return controller == SS_ANALYSIS_WORK_STEP_PREDICTOR_ERROR ||
         controller == SS_ANALYSIS_WORK_STEP_WORK_EFFICIENCY;
}

static void ss_analysis_work_efficiency_reset(SsAnalysisWorkEfficiencyState *state) {
  state->previous_step = NAN;
  state->previous_iterations = 0;
  state->elasticity = 0.0;
  state->elasticity_ready = 0;
}

static PetscReal ss_analysis_work_efficiency_success_scale(
    const SsAnalysisPolicy *policy, PetscReal predictor_error,
    PetscInt nonlinear_iterations, PetscReal factor_increment,
    PetscReal previous_factor_increment, PetscReal actual_step,
    SsAnalysisWorkEfficiencyState *state, PetscReal *elasticity,
    PetscReal *iteration_budget) {
  PetscReal scale = policy->work_step_scale_maximum;
  PetscReal budget = (PetscReal)policy->work_growth_iteration_limit;
  if (ss_is_finite(predictor_error) && predictor_error > DBL_MIN) {
    scale = sqrt(policy->work_predictor_error_target / predictor_error);
  }
  scale = ss_analysis_clamp(scale, policy->work_step_scale_minimum,
                            policy->work_step_scale_maximum);
  if (!(factor_increment < 0.9 * previous_factor_increment) && scale > 1.0) {
    scale = 1.0;
  }
  if (state->previous_iterations > 0 && nonlinear_iterations > 0 &&
      ss_is_finite(state->previous_step) && state->previous_step > 0.0 &&
      actual_step > 0.0) {
    const PetscReal log_step_ratio = log(actual_step / state->previous_step);
    if (ss_is_finite(log_step_ratio) && fabs(log_step_ratio) > 32.0 * DBL_EPSILON) {
      PetscReal pair_elasticity =
          log((PetscReal)nonlinear_iterations / (PetscReal)state->previous_iterations) /
          log_step_ratio;
      if (ss_is_finite(pair_elasticity)) {
        /*
         * Slopes outside this interval have the same control meaning as the
         * endpoint and usually arise from an almost unchanged trial step.
         */
        pair_elasticity = ss_analysis_clamp(pair_elasticity, -1.0, 2.0);
        state->elasticity = state->elasticity_ready != 0
                                ? 0.5 * (state->elasticity + pair_elasticity)
                                : pair_elasticity;
        state->elasticity_ready = 1;
      }
    }
  }
  if (state->elasticity_ready != 0) {
    const PetscReal efficient_growth =
        ss_analysis_clamp(1.0 - state->elasticity, 0.0, 1.0);
    budget += efficient_growth * (PetscReal)(policy->work_efficiency_iteration_maximum -
                                             policy->work_growth_iteration_limit);
  }
  if (nonlinear_iterations > 0) {
    scale = fmin(scale, budget / (PetscReal)nonlinear_iterations);
  }
  state->previous_step = actual_step;
  state->previous_iterations = nonlinear_iterations;
  *elasticity = state->elasticity_ready != 0 ? state->elasticity : NAN;
  *iteration_budget = budget;
  return ss_analysis_clamp(scale, policy->work_step_scale_minimum,
                           policy->work_step_scale_maximum);
}

static PetscReal ss_analysis_adaptive_retry_step(const SsAnalysisPolicy *policy,
                                                 const SsNonlinearSolveRecord *solve,
                                                 PetscReal attempted_step,
                                                 PetscReal previous_accepted_step) {
  PetscReal next_step = 0.5 * attempted_step;
  const int bounded_failure = solve->outcome == SS_NONLINEAR_FAILED_ITERATION_LIMIT ||
                              solve->outcome == SS_NONLINEAR_FAILED_LINE_SEARCH;
  const int late_bounded_failure =
      bounded_failure != 0 &&
      (policy->work_step_controller == SS_ANALYSIS_WORK_STEP_PREDICTOR_ERROR ||
       solve->iterations >= policy->work_growth_iteration_limit);
  /*
   * Predictor-error retries use only the accepted/failed work bracket.  The
   * work-efficiency controller deliberately retains its effort gate.
   */
  if (late_bounded_failure != 0 && ss_is_finite(previous_accepted_step) &&
      previous_accepted_step > 0.0 && previous_accepted_step < attempted_step) {
    /* The accepted and failed steps form a bracket; bisect it geometrically. */
    next_step = sqrt(previous_accepted_step * attempted_step);
    next_step = ss_analysis_clamp(next_step, 0.5 * attempted_step, attempted_step);
  } else if (ss_is_finite(previous_accepted_step) &&
             previous_accepted_step > next_step &&
             previous_accepted_step < attempted_step) {
    /* A cheap failure halves the trial, but never below the last proven step. */
    next_step = previous_accepted_step;
  }
  return next_step;
}

SsStatus ss_analysis_run_indirect(const SsAnalysisProblem *problem,
                                  const SsAnalysisPolicy *policy, Vec previous_state,
                                  Vec initial_state, Vec accepted_state,
                                  const SsAnalysisResumeState *resume,
                                  SsAnalysisRecord *record) {
  Vec previous = NULL;
  Vec current = NULL;
  Vec predictor = NULL;
  Vec predictor_reference = NULL;
  Vec candidate = NULL;
  SsNonlinearSolveRecord *solve = NULL;
  PetscReal previous_factor = NAN;
  PetscReal previous_work = NAN;
  PetscReal current_factor = NAN;
  PetscReal current_work = NAN;
  PetscReal first_work = NAN;
  PetscReal first_factor = policy->factor_initial;
  PetscReal factor_step = policy->factor_step_initial;
  PetscReal work_step = NAN;
  PetscReal previous_factor_increment = NAN;
  SsAnalysisWorkEfficiencyState work_efficiency;
  PetscInt retries = 0;
  int distinct_seed_ready = 0;
  SsAnalysisAttemptRecord *open_attempt = NULL;
  SsStatus status = SS_OK;
  const size_t accepted_step_offset =
      resume == NULL ? 0U : resume->accepted_step_offset;
  ss_analysis_work_efficiency_reset(&work_efficiency);
  if (problem->operations.solve_fixed == NULL ||
      problem->operations.solve_work_controlled == NULL ||
      (resume != NULL && previous_state == NULL)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  Vec *work_vectors[] = {&previous, &current, &predictor, &candidate,
                         &predictor_reference};
  const size_t work_count =
      ss_analysis_work_step_uses_predictor_error(policy->work_step_controller) ? 5U
                                                                               : 4U;
  status = ss_vec_duplicate_collective(initial_state, work_vectors, work_count);
  if (status == SS_OK &&
      (VecCopy(resume == NULL ? initial_state : previous_state, previous) !=
           PETSC_SUCCESS ||
       VecCopy(initial_state, current) != PETSC_SUCCESS ||
       (resume != NULL && VecCopy(initial_state, accepted_state) != PETSC_SUCCESS))) {
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
    previous_factor = resume->previous_factor;
    previous_work = resume->previous_work;
    current_factor = resume->current_factor;
    current_work = resume->current_work;
    first_work = resume->first_control;
    retries = resume->retry_count;
    distinct_seed_ready =
        resume->next_operation == SS_ANALYSIS_NEXT_CONTINUATION_ATTEMPT;
    if (distinct_seed_ready != 0) {
      work_step = resume->next_step;
      previous_factor_increment = current_factor - previous_factor;
      if (policy->work_step_controller == SS_ANALYSIS_WORK_STEP_WORK_EFFICIENCY) {
        work_efficiency.previous_step = current_work - previous_work;
        work_efficiency.previous_iterations =
            resume->work_efficiency_previous_iterations;
        work_efficiency.elasticity = resume->work_efficiency_elasticity;
        work_efficiency.elasticity_ready = resume->work_efficiency_elasticity_ready;
        record->work_efficiency_previous_iterations =
            work_efficiency.previous_iterations;
        record->work_efficiency_elasticity = work_efficiency.elasticity;
        record->work_efficiency_elasticity_ready = work_efficiency.elasticity_ready;
      }
    } else {
      factor_step = resume->next_step;
      first_factor = current_factor;
    }
    status = ss_analysis_append_point(problem, record, 0U, current_factor, current_work,
                                      current_work, resume->current_signed_scaled_slope,
                                      current, 0);
    if (status != SS_OK) {
      goto cleanup;
    }
  }
  while (distinct_seed_ready == 0 &&
         accepted_step_offset + record->accepted_step_count <
             (size_t)policy->accepted_step_limit) {
    SsAnalysisAttemptRecord *attempt = NULL;
    const size_t total_point_count = accepted_step_offset + record->accepted_step_count;
    PetscReal target = total_point_count == 0U
                           ? first_factor
                           : fmin(current_factor + factor_step, policy->factor_maximum);
    PetscReal actual_factor_step =
        total_point_count == 0U ? factor_step : target - current_factor;
    PetscReal work = NAN;
    PetscReal work_increment = NAN;
    int cancelled = 0;
    int shifted_seed = 0;
    status = ss_analysis_boundary(
        problem, record, distinct_seed_ready != 0 ? work_step : factor_step, first_work,
        distinct_seed_ready != 0 ? SS_ANALYSIS_NEXT_CONTINUATION_ATTEMPT
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
    if (!ss_is_finite(target) || target <= 0.0 || !ss_is_finite(actual_factor_step) ||
        (total_point_count != 0U && actual_factor_step <= 0.0)) {
      ss_analysis_finish(record, SS_ANALYSIS_STOP_MINIMUM_STEP_REACHED, 0);
      break;
    }
    status = ss_analysis_begin_attempt(
        record, SS_ANALYSIS_PREDICTOR_CURRENT_STATE, retries, target,
        total_point_count == 0U ? target : current_factor,
        total_point_count == 0U ? NAN : current_work,
        total_point_count == 0U ? NAN : current_work, factor_step, actual_factor_step,
        &attempt);
    if (status != SS_OK) {
      goto cleanup;
    }
    status = ss_analysis_open_attempt(problem, attempt);
    if (status != SS_OK) {
      goto cleanup;
    }
    open_attempt = attempt;
    memset(solve, 0, sizeof(*solve));
    ss_analysis_observe_start(problem, SS_ANALYSIS_SOLVE_PRIMARY, target);
    status = problem->operations.solve_fixed(problem->context, target, current,
                                             candidate, solve);
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
    status =
        ss_analysis_observe_solve(problem, attempt, SS_ANALYSIS_SOLVE_PRIMARY, solve);
    if (status != SS_OK) {
      goto cleanup;
    }
    if (!ss_analysis_nonlinear_succeeded(solve->outcome)) {
      status = ss_analysis_rollback_attempt(problem, record, attempt);
      if (status != SS_OK) {
        goto cleanup;
      }
      open_attempt = NULL;
      ++retries;
      factor_step *= 0.5;
      if (total_point_count == 0U) {
        first_factor *= 0.5;
      }
      if (solve->outcome == SS_NONLINEAR_CANCELLED) {
        ss_analysis_finish(record, SS_ANALYSIS_STOP_CANCELLED, 0);
        break;
      }
      /*
       * The indirect SSR seed search is bounded by the resolved minimum
       * factor increment, not by the continuation-attempt retry budget.
       * Both frozen robust references keep halving the first factor (and
       * increment) until a seed converges or that minimum is crossed. The
       * five-reduction limit applies only after a distinct seed pair enters
       * work-controlled continuation.
       */
      if (factor_step < policy->factor_step_minimum) {
        ss_analysis_finish(record, SS_ANALYSIS_STOP_MINIMUM_STEP_REACHED, 0);
        break;
      }
      continue;
    }
    status = ss_analysis_external_work(problem, candidate, &work);
    if (status != SS_OK) {
      goto cleanup;
    }
    attempt->accepted_factor = target;
    attempt->accepted_control = work;
    attempt->accepted_work = work;
    attempt->successful = 1;
    if (total_point_count == 0U) {
      if (VecCopy(candidate, current) != PETSC_SUCCESS) {
        status = SS_ERROR_INTERNAL;
        goto cleanup;
      }
      current_factor = target;
      current_work = work;
      first_work = work;
    } else {
      work_increment = work - current_work;
      if (!ss_is_finite(work_increment)) {
        attempt->successful = 0;
        status = ss_analysis_rollback_attempt(problem, record, attempt);
        if (status != SS_OK) {
          goto cleanup;
        }
        open_attempt = NULL;
        ss_analysis_finish(record, SS_ANALYSIS_STOP_NONFINITE, 0);
        break;
      }
      shifted_seed = work_increment / fmax(1.0, current_work) < 1.0e-5;
      if (VecCopy(current, previous) != PETSC_SUCCESS ||
          VecCopy(candidate, current) != PETSC_SUCCESS) {
        status = SS_ERROR_INTERNAL;
        goto cleanup;
      }
      previous_factor = current_factor;
      previous_work = current_work;
      current_factor = target;
      current_work = work;
      if (shifted_seed != 0) {
        first_work = current_work;
      } else {
        previous_factor_increment = current_factor - previous_factor;
        work_step = fmax(current_work - previous_work, policy->work_step_minimum);
        distinct_seed_ready = 1;
      }
    }
    status = ss_analysis_commit_attempt(problem, record, attempt);
    if (status != SS_OK) {
      goto cleanup;
    }
    open_attempt = NULL;
    if (VecCopy(current, accepted_state) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
      goto cleanup;
    }
    status = ss_analysis_append_point(problem, record, attempt->attempt_ordinal,
                                      current_factor, current_work, current_work, NAN,
                                      current, 1);
    if (status != SS_OK) {
      goto cleanup;
    }
    retries = 0;
  }
  if (record->stop_reason != SS_ANALYSIS_STOP_NONE) {
    goto cleanup;
  }
  if (distinct_seed_ready == 0) {
    ss_analysis_finish(record, SS_ANALYSIS_STOP_ACCEPTED_STEP_LIMIT, 0);
    goto cleanup;
  }
  while (accepted_step_offset + record->accepted_step_count <
         (size_t)policy->accepted_step_limit) {
    SsAnalysisAttemptRecord *attempt = NULL;
    const PetscReal target = fmin(current_work + work_step, policy->work_maximum);
    const PetscReal actual_step = target - current_work;
    const PetscReal requested_step = work_step;
    PetscReal accepted_factor = NAN;
    PetscReal factor_increment;
    PetscReal slope = NAN;
    PetscReal predictor_ratio;
    int cancelled = 0;
    status = ss_analysis_boundary(problem, record, work_step, first_work,
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
    if (actual_step <= 0.0) {
      ss_analysis_finish(record, SS_ANALYSIS_STOP_WORK_HORIZON_REACHED, 1);
      break;
    }
    predictor_ratio = actual_step / (current_work - previous_work);
    if (!ss_is_finite(predictor_ratio) ||
        VecCopy(current, predictor) != PETSC_SUCCESS ||
        (policy->indirect_predictor_mode ==
                 SS_ANALYSIS_INDIRECT_PREDICTOR_SCALED_CURRENT
             ? VecScale(predictor, target / current_work) != PETSC_SUCCESS
             : (VecAXPY(predictor, predictor_ratio, current) != PETSC_SUCCESS ||
                VecAXPY(predictor, -predictor_ratio, previous) != PETSC_SUCCESS)) ||
        (predictor_reference != NULL &&
         VecCopy(predictor, predictor_reference) != PETSC_SUCCESS)) {
      status = SS_ERROR_INTEGRITY;
      goto cleanup;
    }
    status = ss_analysis_begin_attempt(
        record,
        policy->indirect_predictor_mode == SS_ANALYSIS_INDIRECT_PREDICTOR_SCALED_CURRENT
            ? SS_ANALYSIS_PREDICTOR_CURRENT_STATE
            : SS_ANALYSIS_PREDICTOR_DISPLACEMENT_SECANT_ANCHORED_FACTOR,
        retries, target, current_factor, current_work, current_work, requested_step,
        actual_step, &attempt);
    if (status != SS_OK) {
      goto cleanup;
    }
    status = ss_analysis_open_attempt(problem, attempt);
    if (status != SS_OK) {
      goto cleanup;
    }
    open_attempt = attempt;
    memset(solve, 0, sizeof(*solve));
    ss_analysis_observe_start(problem, SS_ANALYSIS_SOLVE_PRIMARY, target);
    status = problem->operations.solve_work_controlled(
        problem->context, SS_ANALYSIS_SSR_INDIRECT, target, current_factor, predictor,
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
      if (ss_analysis_work_step_uses_predictor_error(policy->work_step_controller)) {
        const PetscReal previous_accepted_step = current_work - previous_work;
        const PetscReal attempted_step = actual_step;
        work_step = ss_analysis_adaptive_retry_step(policy, solve, attempted_step,
                                                    previous_accepted_step);
        work_step = fmax(work_step, policy->work_step_minimum);
        attempt->work_step_scale = work_step / attempted_step;
        attempt->next_work_step = work_step;
        if (policy->work_step_controller == SS_ANALYSIS_WORK_STEP_WORK_EFFICIENCY) {
          ss_analysis_work_efficiency_reset(&work_efficiency);
          record->work_efficiency_previous_iterations = 0;
          record->work_efficiency_elasticity = 0.0;
          record->work_efficiency_elasticity_ready = 0;
        }
      } else {
        work_step *= 0.5;
        attempt->work_step_scale = 0.5;
        attempt->next_work_step = work_step;
      }
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
    if (actual_step > DBL_MIN) {
      slope = factor_increment / actual_step * (target - first_work);
    }
    if (predictor_reference != NULL) {
      PetscReal correction_norm = NAN;
      PetscReal increment_norm = NAN;
      if (VecAYPX(predictor_reference, -1.0, candidate) != PETSC_SUCCESS ||
          VecNorm(predictor_reference, NORM_2, &correction_norm) != PETSC_SUCCESS ||
          VecCopy(candidate, predictor_reference) != PETSC_SUCCESS ||
          VecAXPY(predictor_reference, -1.0, current) != PETSC_SUCCESS ||
          VecNorm(predictor_reference, NORM_2, &increment_norm) != PETSC_SUCCESS ||
          !ss_is_finite(correction_norm) || !ss_is_finite(increment_norm)) {
        status = SS_ERROR_INTEGRITY;
        goto cleanup;
      }
      attempt->predictor_correction_norm = correction_norm;
      attempt->accepted_increment_norm = increment_norm;
      attempt->predictor_error = correction_norm <= DBL_MIN && increment_norm <= DBL_MIN
                                     ? 0.0
                                     : correction_norm / fmax(increment_norm, DBL_MIN);
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
        VecCopy(current, accepted_state) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
      goto cleanup;
    }
    previous_work = current_work;
    current_factor = accepted_factor;
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
    if (policy->scaled_slope_tolerance > 0.0 && ss_is_finite(slope) &&
        slope <= policy->scaled_slope_tolerance) {
      ss_analysis_finish(record, SS_ANALYSIS_STOP_SCALED_SLOPE_REACHED, 1);
      break;
    }
    if (current_work >= policy->work_maximum) {
      ss_analysis_finish(record, SS_ANALYSIS_STOP_WORK_HORIZON_REACHED, 1);
      break;
    }
    if (policy->work_step_controller == SS_ANALYSIS_WORK_STEP_PREDICTOR_ERROR) {
      const PetscReal scale = ss_analysis_adaptive_success_scale(
          policy, attempt->predictor_error, factor_increment,
          previous_factor_increment);
      work_step = fmax(actual_step * scale, policy->work_step_minimum);
      attempt->work_step_scale = work_step / actual_step;
      attempt->next_work_step = work_step;
    } else if (policy->work_step_controller == SS_ANALYSIS_WORK_STEP_WORK_EFFICIENCY) {
      const PetscReal scale = ss_analysis_work_efficiency_success_scale(
          policy, attempt->predictor_error, solve->iterations, factor_increment,
          previous_factor_increment, actual_step, &work_efficiency,
          &attempt->work_efficiency_elasticity,
          &attempt->work_efficiency_iteration_budget);
      work_step = fmax(actual_step * scale, policy->work_step_minimum);
      attempt->work_step_scale = work_step / actual_step;
      attempt->next_work_step = work_step;
      record->work_efficiency_previous_iterations = work_efficiency.previous_iterations;
      record->work_efficiency_elasticity = work_efficiency.elasticity;
      record->work_efficiency_elasticity_ready = work_efficiency.elasticity_ready;
    } else {
      if ((policy->work_growth_iteration_limit == 0 ||
           solve->iterations < policy->work_growth_iteration_limit) &&
          factor_increment < 0.9 * previous_factor_increment) {
        work_step *= 2.0;
      }
      attempt->work_step_scale = work_step / requested_step;
      attempt->next_work_step = work_step;
    }
    previous_factor_increment = factor_increment;
  }
  if (record->stop_reason == SS_ANALYSIS_STOP_NONE) {
    ss_analysis_finish(record, SS_ANALYSIS_STOP_ACCEPTED_STEP_LIMIT, 1);
  }

cleanup:
  record->next_step = distinct_seed_ready != 0 ? work_step : factor_step;
  record->first_control = first_work;
  record->next_operation = record->stop_reason != SS_ANALYSIS_STOP_ACCEPTED_STEP_LIMIT
                               ? SS_ANALYSIS_NEXT_TERMINAL
                           : distinct_seed_ready != 0
                               ? SS_ANALYSIS_NEXT_CONTINUATION_ATTEMPT
                               : SS_ANALYSIS_NEXT_SECOND_SEED;
  if (open_attempt != NULL) {
    SsStatus rollback_status =
        ss_analysis_rollback_attempt(problem, record, open_attempt);
    if (status == SS_OK) {
      status = rollback_status;
    }
  }
  free(solve);
  (void)VecDestroy(&candidate);
  (void)VecDestroy(&predictor_reference);
  (void)VecDestroy(&predictor);
  (void)VecDestroy(&current);
  (void)VecDestroy(&previous);
  return status;
}
