/* Nonlinear policy, line search, regularization and augmented updates.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "nonlinear_internal.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

SsStatus ss_nonlinear_policy_validate(const SsNonlinearPolicy *policy) {
  if (policy == NULL || !ss_is_finite(policy->tolerance) || policy->tolerance <= 0.0 ||
      (policy->stopping_criterion != SS_NONLINEAR_STOP_RELATIVE_RESIDUAL &&
       policy->stopping_criterion != SS_NONLINEAR_STOP_RELATIVE_CORRECTION &&
       policy->stopping_criterion != SS_NONLINEAR_STOP_ABSOLUTE_DELTA_LAMBDA) ||
      !ss_is_finite(policy->stopping_tolerance) || policy->stopping_tolerance <= 0.0 ||
      policy->maximum_iterations <= 0 ||
      policy->maximum_iterations > (PetscInt)SS_NONLINEAR_HISTORY_CAPACITY ||
      policy->indirect_progress_guard_iteration < 0 ||
      !ss_is_finite(policy->indirect_progress_guard_maximum_convergence) ||
      policy->indirect_progress_guard_maximum_convergence < 0.0 ||
      !ss_is_finite(policy->indirect_progress_guard_maximum_convergence_ratio) ||
      policy->indirect_progress_guard_maximum_convergence_ratio < 0.0 ||
      (policy->indirect_progress_guard_maximum_convergence == 0.0 &&
       (policy->indirect_progress_guard_iteration != 0 ||
        policy->indirect_progress_guard_maximum_convergence_ratio != 0.0)) ||
      (policy->indirect_progress_guard_maximum_convergence > 0.0 &&
       policy->indirect_progress_guard_iteration >= policy->maximum_iterations) ||
      policy->maximum_line_search_trials <= 0 ||
      (policy->line_method == SS_NONLINEAR_LINE_DIRECTIONAL &&
       policy->maximum_line_search_trials >
           (PetscInt)SS_NONLINEAR_LINE_TRIAL_CAPACITY - 2) ||
      (policy->line_method == SS_NONLINEAR_LINE_RESIDUAL_MONOTONE &&
       policy->maximum_line_search_trials >
           (PetscInt)SS_NONLINEAR_LINE_TRIAL_CAPACITY) ||
      !ss_is_finite(policy->minimum_alpha) || policy->minimum_alpha <= 0.0 ||
      policy->minimum_alpha > 1.0 || !ss_is_finite(policy->regularization_minimum) ||
      policy->regularization_minimum <= 0.0 || policy->regularization_minimum > 1.0 ||
      !ss_is_finite(policy->singular_relative_tolerance) ||
      policy->singular_relative_tolerance <= 0.0 ||
      policy->singular_relative_tolerance >= 1.0 ||
      !ss_is_finite(policy->relaxed_factor) || policy->relaxed_factor < 1.0 ||
      (policy->allow_relaxed != 0 && policy->allow_relaxed != 1) ||
      (policy->line_search_legacy_alg5 != 0 && policy->line_search_legacy_alg5 != 1) ||
      (policy->line_method != SS_NONLINEAR_LINE_DIRECTIONAL &&
       policy->line_method != SS_NONLINEAR_LINE_RESIDUAL_MONOTONE) ||
      (policy->control_sensitivity_method !=
           SS_NONLINEAR_CONTROL_SENSITIVITY_REFERENCE_FORWARD &&
       policy->control_sensitivity_method !=
           SS_NONLINEAR_CONTROL_SENSITIVITY_STABLE_CENTRAL) ||
      (policy->control_sensitivity_method ==
           SS_NONLINEAR_CONTROL_SENSITIVITY_STABLE_CENTRAL &&
       (!ss_is_finite(policy->control_sensitivity_relative_step) ||
        policy->control_sensitivity_relative_step <= 0.0 ||
        policy->control_sensitivity_relative_step > 0.1)) ||
      (policy->indirect_adaptive_linear_tolerance != 0 &&
       policy->indirect_adaptive_linear_tolerance != 1) ||
      (policy->indirect_adaptive_linear_tolerance != 0 &&
       (policy->indirect_adaptive_linear_iteration_threshold < 0 ||
        !ss_is_finite(policy->indirect_adaptive_linear_alpha_threshold) ||
        policy->indirect_adaptive_linear_alpha_threshold <= 0.0 ||
        policy->indirect_adaptive_linear_alpha_threshold > 1.0 ||
        !ss_is_finite(policy->indirect_adaptive_linear_tightening_factor) ||
        policy->indirect_adaptive_linear_tightening_factor <= 0.0 ||
        policy->indirect_adaptive_linear_tightening_factor >= 1.0 ||
        !ss_is_finite(policy->indirect_adaptive_linear_minimum_relative_tolerance) ||
        policy->indirect_adaptive_linear_minimum_relative_tolerance <= 0.0))) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return SS_OK;
}

static int ss_nonlinear_solve_succeeded(SsNonlinearOutcome outcome) {
  return outcome == SS_NONLINEAR_CONVERGED || outcome == SS_NONLINEAR_ACCEPTED_RELAXED;
}

SsStatus ss_nonlinear_adaptive_progress_guard_initialize(
    PetscInt checkpoint_iteration, SsNonlinearAdaptiveProgressGuard *guard) {
  if (guard == NULL || checkpoint_iteration < 0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(guard, 0, sizeof(*guard));
  guard->checkpoint_iteration = checkpoint_iteration;
  return SS_OK;
}

SsStatus ss_nonlinear_adaptive_progress_guard_restore(
    PetscInt checkpoint_iteration, uint64_t successful_sample_count,
    PetscReal maximum_successful_convergence,
    PetscReal minimum_successful_convergence_ratio,
    PetscReal maximum_successful_convergence_ratio,
    SsNonlinearAdaptiveProgressGuard *guard) {
  SsStatus status =
      ss_nonlinear_adaptive_progress_guard_initialize(checkpoint_iteration, guard);
  if (status != SS_OK || !ss_is_finite(maximum_successful_convergence) ||
      maximum_successful_convergence < 0.0 ||
      !ss_is_finite(minimum_successful_convergence_ratio) ||
      !ss_is_finite(maximum_successful_convergence_ratio) ||
      minimum_successful_convergence_ratio < 0.0 ||
      maximum_successful_convergence_ratio < minimum_successful_convergence_ratio ||
      (successful_sample_count == 0U &&
       (maximum_successful_convergence != 0.0 ||
        minimum_successful_convergence_ratio != 0.0 ||
        maximum_successful_convergence_ratio != 0.0))) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  guard->successful_sample_count = successful_sample_count;
  guard->maximum_successful_convergence = maximum_successful_convergence;
  guard->minimum_successful_convergence_ratio = minimum_successful_convergence_ratio;
  guard->maximum_successful_convergence_ratio = maximum_successful_convergence_ratio;
  return SS_OK;
}

SsStatus ss_nonlinear_adaptive_progress_guard_threshold(
    const SsNonlinearAdaptiveProgressGuard *guard, PetscInt maximum_iterations,
    int *enabled, PetscReal *threshold, PetscReal *convergence_ratio_threshold) {
  if (guard == NULL || enabled == NULL || threshold == NULL ||
      convergence_ratio_threshold == NULL || guard->checkpoint_iteration < 0 ||
      guard->checkpoint_iteration >= maximum_iterations ||
      !ss_is_finite(guard->maximum_successful_convergence) ||
      guard->maximum_successful_convergence < 0.0 ||
      !ss_is_finite(guard->minimum_successful_convergence_ratio) ||
      !ss_is_finite(guard->maximum_successful_convergence_ratio) ||
      guard->minimum_successful_convergence_ratio < 0.0 ||
      guard->maximum_successful_convergence_ratio <
          guard->minimum_successful_convergence_ratio ||
      (guard->successful_sample_count == 0U &&
       (guard->maximum_successful_convergence != 0.0 ||
        guard->minimum_successful_convergence_ratio != 0.0 ||
        guard->maximum_successful_convergence_ratio != 0.0))) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *enabled = 0;
  *threshold = 0.0;
  *convergence_ratio_threshold = 0.0;
  if (guard->successful_sample_count <
          (uint64_t)SS_NONLINEAR_ADAPTIVE_GUARD_MINIMUM_SAMPLES ||
      guard->maximum_successful_convergence <= 0.0 ||
      guard->maximum_successful_convergence_ratio <= 0.0 ||
      guard->maximum_successful_convergence_ratio >= 1.0) {
    return SS_OK;
  }
  /*
   * Geometric interpolation between the worst successful checkpoint
   * contraction and 1 (no contraction).  The same learned margin expands the
   * absolute envelope, so neither cutoff contains a problem-specific residual
   * constant and neither can reject an already observed successful sample.
   * A successful sample at or above 1 provides no safe contraction evidence;
   * the guard remains disabled in that case above.
   */
  *convergence_ratio_threshold = sqrt(guard->maximum_successful_convergence_ratio);
  *threshold = guard->maximum_successful_convergence / *convergence_ratio_threshold;
  if (!ss_is_finite(*threshold)) {
    *threshold = 0.0;
    *convergence_ratio_threshold = 0.0;
    return SS_OK;
  }
  *enabled = 1;
  return SS_OK;
}

SsStatus ss_nonlinear_adaptive_progress_guard_observe(
    SsNonlinearAdaptiveProgressGuard *guard,
    const SsNonlinearSolveRecord *successful_solve, int *sample_observed,
    PetscReal *checkpoint_convergence) {
  PetscReal convergence = NAN;
  PetscReal convergence_ratio = NAN;
  const size_t checkpoint = guard == NULL || guard->checkpoint_iteration < 0
                                ? SIZE_MAX
                                : (size_t)guard->checkpoint_iteration;
  if (guard == NULL || successful_solve == NULL || sample_observed == NULL ||
      checkpoint_convergence == NULL ||
      !ss_is_finite(guard->maximum_successful_convergence) ||
      guard->maximum_successful_convergence < 0.0 ||
      !ss_is_finite(guard->minimum_successful_convergence_ratio) ||
      !ss_is_finite(guard->maximum_successful_convergence_ratio) ||
      guard->minimum_successful_convergence_ratio < 0.0 ||
      guard->maximum_successful_convergence_ratio <
          guard->minimum_successful_convergence_ratio ||
      (guard->successful_sample_count == 0U &&
       (guard->maximum_successful_convergence != 0.0 ||
        guard->minimum_successful_convergence_ratio != 0.0 ||
        guard->maximum_successful_convergence_ratio != 0.0))) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *sample_observed = 0;
  *checkpoint_convergence = NAN;
  if (!ss_nonlinear_solve_succeeded(successful_solve->outcome)) {
    return SS_OK;
  }
  if (successful_solve->history_count > checkpoint) {
    convergence = successful_solve->history[checkpoint].convergence_before;
  } else if (successful_solve->history_count == checkpoint) {
    convergence = successful_solve->final_convergence;
  } else {
    return SS_OK;
  }
  if (!ss_is_finite(convergence) || convergence < 0.0 ||
      !ss_is_finite(successful_solve->initial_convergence) ||
      successful_solve->initial_convergence <= 0.0 ||
      guard->successful_sample_count == UINT64_MAX) {
    return SS_ERROR_INTEGRITY;
  }
  convergence_ratio = convergence / successful_solve->initial_convergence;
  if (!ss_is_finite(convergence_ratio) || convergence_ratio < 0.0) {
    return SS_ERROR_INTEGRITY;
  }
  if (guard->successful_sample_count == 0U) {
    guard->minimum_successful_convergence_ratio = convergence_ratio;
    guard->maximum_successful_convergence_ratio = convergence_ratio;
  } else {
    guard->minimum_successful_convergence_ratio =
        fmin(guard->minimum_successful_convergence_ratio, convergence_ratio);
    guard->maximum_successful_convergence_ratio =
        fmax(guard->maximum_successful_convergence_ratio, convergence_ratio);
  }
  ++guard->successful_sample_count;
  guard->maximum_successful_convergence =
      fmax(guard->maximum_successful_convergence, convergence);
  *sample_observed = 1;
  *checkpoint_convergence = convergence;
  return SS_OK;
}

SsStatus ss_nonlinear_regularization_update(PetscReal current, PetscReal minimum,
                                            PetscReal alpha,
                                            SsNonlinearRegularizationRecord *record) {
  PetscReal next;
  if (record == NULL || !ss_is_finite(current) || !ss_is_finite(minimum) ||
      !ss_is_finite(alpha) || current < minimum || current <= 0.0 || current > 1.0 ||
      minimum <= 0.0 || minimum > 1.0 || alpha < 0.0 || alpha > 1.0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(record, 0, sizeof(*record));
  record->prior_value = current;
  record->alpha = alpha;
  record->outcome = SS_NONLINEAR_CONVERGED;
  if (alpha == 0.0) {
    next = 2.0 * current;
  } else if (alpha < 0.1) {
    next = pow(2.0, 0.25) * current;
  } else if (alpha <= 0.5) {
    next = current;
    record->deflation_candidate_eligible = 1;
  } else {
    next = fmax(current / sqrt(2.0), minimum);
    record->deflation_candidate_eligible = 1;
  }
  /*
   * Both a rejected line search (alpha == 0) and a repeatedly accepted tiny
   * step can push the next blend past the convex interval.  Report the latter
   * through the normal nonlinear-failure path as well; otherwise the next
   * iteration enters with r > 1 and escapes as SS_ERROR_INVALID_ARGUMENT.
   */
  if (next > 1.0) {
    record->outcome = SS_NONLINEAR_FAILED_REGULARIZATION;
    record->deflation_candidate_eligible = 0;
  }
  record->next_value = next;
  record->operator_generation_invalidated = next != current;
  return SS_OK;
}

static SsStatus ss_nonlinear_trial_append(SsNonlinearLineSearchRecord *record,
                                          PetscInt ordinal, PetscReal alpha,
                                          SsNonlinearTrialEvaluation evaluate,
                                          void *context) {
  SsNonlinearLineTrial *trial;
  PetscReal directional = NAN;
  PetscReal residual = NAN;
  int admissible = 0;
  SsStatus status;
  if (record->trial_count >= SS_NONLINEAR_LINE_TRIAL_CAPACITY) {
    return SS_ERROR_LIMIT;
  }
  status = evaluate(context, alpha, &directional, &residual, &admissible);
  if (status != SS_OK) {
    return status;
  }
  trial = &record->trials[record->trial_count++];
  memset(trial, 0, sizeof(*trial));
  trial->ordinal = ordinal;
  trial->alpha = alpha;
  trial->directional_value = directional;
  trial->residual_norm = residual;
  trial->finite = ss_is_finite(directional) && ss_is_finite(residual);
  trial->admissible = admissible != 0;
  return SS_OK;
}

SsStatus ss_nonlinear_line_search(const SsNonlinearPolicy *policy,
                                  PetscReal current_residual_norm,
                                  SsNonlinearTrialEvaluation evaluate, void *context,
                                  SsNonlinearLineSearchRecord *record) {
  return ss_nonlinear_line_search_with_floor(policy, current_residual_norm, 0.0,
                                             evaluate, context, record);
}

SsStatus ss_nonlinear_line_search_with_floor(const SsNonlinearPolicy *policy,
                                             PetscReal current_residual_norm,
                                             PetscReal residual_floor,
                                             SsNonlinearTrialEvaluation evaluate,
                                             void *context,
                                             SsNonlinearLineSearchRecord *record) {
  PetscInt ordinal = 0;
  SsStatus status;
  if (ss_nonlinear_policy_validate(policy) != SS_OK || evaluate == NULL ||
      record == NULL || !ss_is_finite(current_residual_norm) ||
      current_residual_norm < 0.0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(record, 0, sizeof(*record));
  record->method =
      residual_floor > 0.0 ? SS_NONLINEAR_LINE_RESIDUAL_MONOTONE : policy->line_method;
  record->current_residual_norm = current_residual_norm;
  record->accepted_residual_norm = current_residual_norm;
  record->outcome = SS_NONLINEAR_FAILED_LINE_SEARCH;
  if (record->method == SS_NONLINEAR_LINE_DIRECTIONAL) {
    PetscReal lower = 0.0;
    PetscReal upper = 1.0;
    PetscReal lower_residual = current_residual_norm;
    PetscInt evaluation;
    status = ss_nonlinear_trial_append(record, ordinal++, 0.0, evaluate, context);
    if (status != SS_OK) {
      return status;
    }
    if (record->trials[0].finite == 0) {
      record->outcome = SS_NONLINEAR_FAILED_NONFINITE;
      return SS_OK;
    }
    if (record->trials[0].admissible == 0 ||
        record->trials[0].directional_value >= 0.0) {
      return SS_OK;
    }
    lower_residual = record->trials[0].residual_norm;
    status = ss_nonlinear_trial_append(record, ordinal++, 1.0, evaluate, context);
    if (status != SS_OK) {
      return status;
    }
    if (record->trials[1].finite != 0 && record->trials[1].admissible != 0 &&
        record->trials[1].directional_value <= 0.0) {
      record->trials[1].accepted = 1;
      record->accepted_alpha = 1.0;
      record->accepted_residual_norm = record->trials[1].residual_norm;
      record->outcome = SS_NONLINEAR_CONVERGED;
      return SS_OK;
    }
    for (evaluation = 0; evaluation < policy->maximum_line_search_trials;
         ++evaluation) {
      const PetscReal midpoint = 0.5 * (lower + upper);
      SsNonlinearLineTrial *trial;
      status =
          ss_nonlinear_trial_append(record, ordinal++, midpoint, evaluate, context);
      if (status != SS_OK) {
        return status;
      }
      trial = &record->trials[record->trial_count - 1U];
      if (trial->finite != 0 && trial->admissible != 0 &&
          trial->directional_value <= 0.0) {
        lower = midpoint;
        lower_residual = trial->residual_norm;
      } else {
        upper = midpoint;
      }
    }
    if (lower >= policy->minimum_alpha) {
      size_t index;
      for (index = record->trial_count; index > 0U; --index) {
        if (record->trials[index - 1U].alpha == lower) {
          record->trials[index - 1U].accepted = 1;
          break;
        }
      }
      record->accepted_alpha = lower;
      record->accepted_residual_norm = lower_residual;
      record->outcome = SS_NONLINEAR_CONVERGED;
    }
    return SS_OK;
  }
  {
    PetscReal alpha = 1.0;
    PetscInt evaluation;
    size_t last_finite_admissible = SIZE_MAX;
    for (evaluation = 0; evaluation < policy->maximum_line_search_trials;
         ++evaluation) {
      SsNonlinearLineTrial *trial;
      status = ss_nonlinear_trial_append(record, ordinal++, alpha, evaluate, context);
      if (status != SS_OK) {
        return status;
      }
      trial = &record->trials[record->trial_count - 1U];
      if (trial->finite != 0 && trial->admissible != 0) {
        last_finite_admissible = record->trial_count - 1U;
      }
      if (trial->finite != 0 && trial->admissible != 0 &&
          (trial->residual_norm < current_residual_norm ||
           (residual_floor > 0.0 && trial->residual_norm <= residual_floor))) {
        trial->accepted = 1;
        record->accepted_alpha = alpha;
        record->accepted_residual_norm = trial->residual_norm;
        record->outcome = SS_NONLINEAR_CONVERGED;
        return SS_OK;
      }
      alpha *= 0.5;
      if (alpha < policy->minimum_alpha) {
        break;
      }
    }
    if (policy->line_search_legacy_alg5 != 0 && last_finite_admissible != SIZE_MAX) {
      SsNonlinearLineTrial *trial = &record->trials[last_finite_admissible];
      trial->accepted = 1;
      record->accepted_alpha = trial->alpha;
      record->accepted_residual_norm = trial->residual_norm;
      record->outcome = SS_NONLINEAR_CONVERGED;
    }
  }
  return SS_OK;
}

SsStatus ss_nonlinear_augmented_direction(Vec control_gradient, Vec first_direction,
                                          Vec residual_direction,
                                          PetscReal relative_tolerance,
                                          Vec combined_direction,
                                          SsNonlinearAugmentedRecord *record) {
  PetscScalar numerator_scalar = 0.0;
  PetscScalar denominator_scalar = 0.0;
  PetscReal gradient_norm = 0.0;
  PetscReal first_norm = 0.0;
  PetscReal combined_norm = 0.0;
  PetscReal numerator;
  PetscReal denominator;
  PetscReal scale;
  PetscReal threshold;
  PetscReal increment;
  PetscInt gradient_size = 0;
  PetscInt first_size = 0;
  PetscInt residual_size = 0;
  PetscInt combined_size = 0;
  if (control_gradient == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  const MPI_Comm comm = PetscObjectComm((PetscObject)control_gradient);
  SsStatus status = SS_OK;
  if (first_direction == NULL || residual_direction == NULL ||
      combined_direction == NULL || record == NULL ||
      !ss_is_finite(relative_tolerance) || relative_tolerance <= 0.0 ||
      relative_tolerance >= 1.0 ||
      VecGetSize(control_gradient, &gradient_size) != PETSC_SUCCESS ||
      VecGetSize(first_direction, &first_size) != PETSC_SUCCESS ||
      VecGetSize(residual_direction, &residual_size) != PETSC_SUCCESS ||
      VecGetSize(combined_direction, &combined_size) != PETSC_SUCCESS ||
      gradient_size != first_size || gradient_size != residual_size ||
      gradient_size != combined_size) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_status_allreduce_max(comm, status);
  if (status != SS_OK) {
    return status;
  }
  memset(record, 0, sizeof(*record));
  record->control_increment = NAN;
  record->outcome = SS_NONLINEAR_FAILED_SINGULAR_CONSTRAINT;
  if (ss_nonlinear_petsc_status(comm, VecDot(control_gradient, residual_direction,
                                             &numerator_scalar)) != SS_OK ||
      ss_nonlinear_petsc_status(comm, VecDot(control_gradient, first_direction,
                                             &denominator_scalar)) != SS_OK ||
      ss_nonlinear_petsc_status(
          comm, VecNorm(control_gradient, NORM_2, &gradient_norm)) != SS_OK ||
      ss_nonlinear_petsc_status(comm, VecNorm(first_direction, NORM_2, &first_norm)) !=
          SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  numerator = PetscRealPart(numerator_scalar);
  denominator = PetscRealPart(denominator_scalar);
  scale = gradient_norm * first_norm;
  threshold = relative_tolerance * fmax(scale, DBL_MIN);
  record->numerator = numerator;
  record->denominator = denominator;
  record->denominator_scale = scale;
  record->threshold = threshold;
  if (!ss_is_finite(numerator) || !ss_is_finite(denominator) || !ss_is_finite(scale) ||
      fabs(denominator) <= threshold) {
    return SS_OK;
  }
  increment = -numerator / denominator;
  if (!ss_is_finite(increment) ||
      ss_nonlinear_petsc_status(comm, VecWAXPY(combined_direction, increment,
                                               first_direction, residual_direction)) !=
          SS_OK ||
      ss_nonlinear_petsc_status(
          comm, VecNorm(combined_direction, NORM_2, &combined_norm)) != SS_OK ||
      !ss_is_finite(combined_norm)) {
    record->outcome = SS_NONLINEAR_FAILED_NONFINITE;
    return SS_OK;
  }
  record->control_increment = increment;
  record->outcome = SS_NONLINEAR_CONVERGED;
  return SS_OK;
}

SsStatus ss_nonlinear_apply_augmented_update(
    Vec parent, Vec direction, Vec control_gradient, PetscReal target_work,
    PetscReal control, PetscReal control_increment, PetscReal alpha, Vec trial,
    PetscReal *updated_control, PetscReal *work_after_rescale) {
  PetscScalar work_scalar = 0.0;
  PetscReal work;
  PetscReal scale;
  PetscReal final_control;
  if (parent == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  const MPI_Comm comm = PetscObjectComm((PetscObject)parent);
  SsStatus status = SS_OK;
  if (direction == NULL || control_gradient == NULL || trial == NULL ||
      updated_control == NULL || work_after_rescale == NULL ||
      !ss_is_finite(target_work) || !ss_is_finite(control) ||
      !ss_is_finite(control_increment) || !ss_is_finite(alpha) || alpha < 0.0 ||
      alpha > 1.0) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_status_allreduce_max(comm, status);
  if (status != SS_OK) {
    return status;
  }
  final_control = control + alpha * control_increment;
  if (!ss_is_finite(final_control) ||
      ss_nonlinear_petsc_status(comm, VecWAXPY(trial, alpha, direction, parent)) !=
          SS_OK ||
      ss_nonlinear_petsc_status(comm, VecDot(control_gradient, trial, &work_scalar)) !=
          SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  work = PetscRealPart(work_scalar);
  if (!ss_is_finite(work) || fabs(work) <= DBL_EPSILON * fmax(1.0, fabs(target_work))) {
    return SS_ERROR_INTEGRITY;
  }
  scale = target_work / work;
  if (!ss_is_finite(scale) ||
      ss_nonlinear_petsc_status(comm, VecScale(trial, scale)) != SS_OK ||
      ss_nonlinear_petsc_status(comm, VecDot(control_gradient, trial, &work_scalar)) !=
          SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  work = PetscRealPart(work_scalar);
  if (!ss_is_finite(work)) {
    return SS_ERROR_INTEGRITY;
  }
  *updated_control = final_control;
  *work_after_rescale = work;
  return SS_OK;
}
