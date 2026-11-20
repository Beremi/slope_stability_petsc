/* Work-controlled nonlinear equilibrium solve.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "nonlinear_internal.h"

#include <float.h>
#include <math.h>

typedef struct {
  const SsNonlinearNewton *newton;
  PetscReal target_work;
  PetscReal control;
  PetscReal control_increment;
  Vec working;
  Vec direction;
  Vec trial;
  Vec trial_residual;
  int legacy_alg5;
} SsNonlinearControlledTrialContext;

static PetscReal
ss_nonlinear_next_linear_relative_tolerance(const SsNonlinearPolicy *policy,
                                            PetscReal current, PetscInt iterations,
                                            PetscReal alpha, int *tightened) {
  PetscReal next = current;
  if (tightened != NULL) {
    *tightened = 0;
  }
  if (policy == NULL || tightened == NULL ||
      policy->indirect_adaptive_linear_tolerance == 0 || !ss_is_finite(current) ||
      current <= 0.0 || !ss_is_finite(alpha) ||
      alpha >= policy->indirect_adaptive_linear_alpha_threshold ||
      iterations > policy->indirect_adaptive_linear_iteration_threshold) {
    return next;
  }
  next = fmax(policy->indirect_adaptive_linear_minimum_relative_tolerance,
              current * policy->indirect_adaptive_linear_tightening_factor);
  if (next < current) {
    *tightened = 1;
  }
  return next;
}

/* Line-search trial of the augmented direction with the matching control update. */
static SsStatus ss_nonlinear_controlled_trial(void *opaque, PetscReal alpha,
                                              PetscReal *directional_value,
                                              PetscReal *residual_norm,
                                              int *admissible) {
  SsNonlinearControlledTrialContext *context =
      (SsNonlinearControlledTrialContext *)opaque;
  const SsNonlinearNewton *newton = context->newton;
  const MPI_Comm comm = newton->communicator;
  PetscReal trial_control = context->control + alpha * context->control_increment;
  PetscReal work_after_rescale = NAN;
  PetscScalar directional = 0.0;
  SsStatus status;
  if (!ss_is_finite(trial_control) ||
      (newton->mode == SS_NONLINEAR_CONTROL_SSR_INDIRECT && trial_control <= 0.0)) {
    *directional_value = NAN;
    *residual_norm = NAN;
    *admissible = 0;
    return SS_OK;
  }
  if (newton->mode == SS_NONLINEAR_CONTROL_SSR_INDIRECT && context->legacy_alg5 != 0) {
    if (ss_nonlinear_petsc_status(comm,
                                  VecWAXPY(context->trial, alpha, context->direction,
                                           context->working)) != SS_OK) {
      return SS_ERROR_INTERNAL;
    }
  } else {
    status = ss_nonlinear_apply_augmented_update(
        context->working, context->direction, newton->problem->reference_load,
        context->target_work, context->control, context->control_increment, alpha,
        context->trial, &trial_control, &work_after_rescale);
    if (status != SS_OK || !ss_is_finite(work_after_rescale)) {
      return status == SS_OK ? SS_ERROR_INTEGRITY : status;
    }
  }
  status = ss_nonlinear_newton_trial_residual(newton, context->trial, trial_control,
                                              context->trial_residual, admissible);
  if (status != SS_OK) {
    return status;
  }
  if (ss_nonlinear_petsc_status(
          comm, VecNorm(context->trial_residual, NORM_2, residual_norm)) != SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  if (newton->mode == SS_NONLINEAR_CONTROL_LIMIT_LOAD &&
      ss_nonlinear_petsc_status(comm, VecAXPY(context->trial_residual, trial_control,
                                              newton->problem->reference_load)) !=
          SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  if (ss_nonlinear_petsc_status(
          comm, VecDot(context->trial_residual, context->direction, &directional)) !=
      SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  *directional_value = PetscRealPart(directional);
  return SS_OK;
}

static SsStatus ss_nonlinear_initialize_work_state(Vec reference_load,
                                                   PetscReal target_work,
                                                   Vec parent_state, Vec working) {
  const MPI_Comm comm = PetscObjectComm((PetscObject)reference_load);
  PetscScalar work_scalar = 0.0;
  PetscReal load_norm = 0.0;
  PetscReal work;
  PetscReal scale;
  if (ss_nonlinear_petsc_status(comm, VecCopy(parent_state, working)) != SS_OK ||
      ss_nonlinear_petsc_status(comm, VecDot(reference_load, working, &work_scalar)) !=
          SS_OK ||
      ss_nonlinear_petsc_status(comm, VecNorm(reference_load, NORM_2, &load_norm)) !=
          SS_OK ||
      !ss_is_finite(load_norm) || load_norm <= 0.0) {
    return SS_ERROR_INTERNAL;
  }
  work = PetscRealPart(work_scalar);
  if (!ss_is_finite(work)) {
    return SS_ERROR_INTEGRITY;
  }
  if (fabs(work) <= DBL_EPSILON * fmax(1.0, fabs(target_work))) {
    scale = target_work / (load_norm * load_norm);
    if (!ss_is_finite(scale) ||
        ss_nonlinear_petsc_status(comm, VecCopy(reference_load, working)) != SS_OK ||
        ss_nonlinear_petsc_status(comm, VecScale(working, scale)) != SS_OK) {
      return SS_ERROR_INTERNAL;
    }
  } else {
    scale = target_work / work;
    if (!ss_is_finite(scale) ||
        ss_nonlinear_petsc_status(comm, VecScale(working, scale)) != SS_OK) {
      return SS_ERROR_INTERNAL;
    }
  }
  return SS_OK;
}

/* Record the indirect-SSR linear tolerances for the next iteration after a step of
 * length `alpha`. */
static void
ss_nonlinear_update_linear_tolerances(const SsNonlinearPolicy *policy, PetscReal alpha,
                                      SsNonlinearIterationRecord *iteration_record,
                                      PetscReal *control_tolerance,
                                      PetscReal *residual_tolerance) {
  *control_tolerance = ss_nonlinear_next_linear_relative_tolerance(
      policy, *control_tolerance, iteration_record->control_linear_solve.iterations,
      alpha, &iteration_record->control_linear_tolerance_tightened);
  *residual_tolerance = ss_nonlinear_next_linear_relative_tolerance(
      policy, *residual_tolerance, iteration_record->residual_linear_solve.iterations,
      alpha, &iteration_record->residual_linear_tolerance_tightened);
  iteration_record->control_linear_next_relative_tolerance = *control_tolerance;
  iteration_record->residual_linear_next_relative_tolerance = *residual_tolerance;
}

/* Form the control right-hand side: the reference load under limit-load control,
 * or the finite-difference sensitivity of the residual to the strength reduction
 * under indirect SSR control. Follows the phase convention of nonlinear_internal.h. */
static int ss_nonlinear_control_rhs(SsNonlinearNewton *newton,
                                    SsNonlinearNewtonStep *step, Vec control_rhs) {
  const SsNonlinearPolicy *policy = newton->policy;
  SsNonlinearIterationRecord *iteration_record = step->record;
  const MPI_Comm comm = newton->communicator;
  const PetscReal control = newton->control;
  Vec working = newton->work->work[SS_NL_WORKING];
  Vec residual = newton->work->work[SS_NL_RESIDUAL];
  Vec sensitivity_residual = newton->work->work[SS_NL_SENSITIVITY];
  Vec sensitivity_residual_minus = newton->work->work[SS_NL_SENSITIVITY_MINUS];
  PetscReal sensitivity_epsilon = 0.0;
  SsStatus status;
  if (newton->mode == SS_NONLINEAR_CONTROL_LIMIT_LOAD) {
    if (ss_nonlinear_petsc_status(
            comm, VecCopy(newton->problem->reference_load, control_rhs)) != SS_OK) {
      newton->status = SS_ERROR_INTERNAL;
      return 0;
    }
  } else {
    iteration_record->control_sensitivity_method = policy->control_sensitivity_method;
    if (policy->control_sensitivity_method ==
        SS_NONLINEAR_CONTROL_SENSITIVITY_STABLE_CENTRAL) {
      sensitivity_epsilon =
          policy->control_sensitivity_relative_step * fmax(1.0, fabs(control));
      sensitivity_epsilon = fmin(sensitivity_epsilon, 0.5 * control);
      if (!ss_is_finite(sensitivity_epsilon) || sensitivity_epsilon <= 0.0 ||
          sensitivity_residual_minus == NULL ||
          !ss_is_finite(control + sensitivity_epsilon) ||
          !ss_is_finite(control - sensitivity_epsilon) ||
          control - sensitivity_epsilon <= 0.0) {
        newton->record->outcome = SS_NONLINEAR_FAILED_NONFINITE;
        return 0;
      }
      status = ss_nonlinear_newton_evaluate(
          newton, working, control + sensitivity_epsilon, sensitivity_residual, NULL);
      if (status == SS_OK) {
        status =
            ss_nonlinear_newton_evaluate(newton, working, control - sensitivity_epsilon,
                                         sensitivity_residual_minus, NULL);
      }
      if (status != SS_OK) {
        newton->record->outcome = ss_nonlinear_status_outcome(status);
        return 0;
      }
      if (ss_nonlinear_petsc_status(comm,
                                    VecWAXPY(control_rhs, -1.0, sensitivity_residual,
                                             sensitivity_residual_minus)) != SS_OK ||
          ss_nonlinear_petsc_status(
              comm, VecScale(control_rhs, 0.5 / sensitivity_epsilon)) != SS_OK) {
        newton->status = SS_ERROR_INTERNAL;
        return 0;
      }
    } else {
      PetscReal perturbed_control = control + policy->tolerance / 1000.0;
      if (perturbed_control == control) {
        perturbed_control = nextafter(control, INFINITY);
      }
      /* Divide by the increment actually represented in the evaluated control. */
      sensitivity_epsilon = perturbed_control - control;
      if (!ss_is_finite(perturbed_control) || sensitivity_epsilon <= 0.0 ||
          !ss_is_finite(1.0 / sensitivity_epsilon)) {
        newton->record->outcome = SS_NONLINEAR_FAILED_NONFINITE;
        return 0;
      }
      status = ss_nonlinear_newton_evaluate(newton, working, perturbed_control,
                                            sensitivity_residual, NULL);
      if (status != SS_OK) {
        newton->record->outcome = ss_nonlinear_status_outcome(status);
        return 0;
      }
      if (ss_nonlinear_petsc_status(
              comm, VecWAXPY(control_rhs, -1.0, sensitivity_residual, residual)) !=
              SS_OK ||
          ss_nonlinear_petsc_status(
              comm, VecScale(control_rhs, 1.0 / sensitivity_epsilon)) != SS_OK) {
        newton->status = SS_ERROR_INTERNAL;
        return 0;
      }
    }
  }
  iteration_record->control_sensitivity_epsilon = sensitivity_epsilon;
  return 1;
}

/* Newton iterations of the bordered system: one control and one residual
 * correction per step, combined so that the work stays at `target_work`. */
static void ss_nonlinear_controlled_iterate(SsNonlinearNewton *newton,
                                            PetscReal target_work) {
  const SsNonlinearPolicy *policy = newton->policy;
  const SsNonlinearProblem *problem = newton->problem;
  SsNonlinearSolveRecord *record = newton->record;
  const SsNonlinearControlMode mode = newton->mode;
  const MPI_Comm comm = newton->communicator;
  const char *const provenances[2] = {
      mode == SS_NONLINEAR_CONTROL_LIMIT_LOAD ? "nls:limit-load-control"
                                              : "nls:ssr-indirect-control",
      mode == SS_NONLINEAR_CONTROL_LIMIT_LOAD ? "nls:limit-load-residual"
                                              : "nls:ssr-indirect-residual",
  };
  Vec working = newton->work->work[SS_NL_WORKING];
  Vec residual = newton->work->work[SS_NL_RESIDUAL];
  Vec control_rhs = newton->work->work[SS_NL_RHS];
  Vec trial = newton->work->work[SS_NL_TRIAL];
  Vec trial_residual = newton->work->work[SS_NL_TRIAL_RESIDUAL];
  Vec residual_rhs = newton->work->work[SS_NL_RESIDUAL_RHS];
  Vec combined_direction = newton->work->work[SS_NL_COMBINED];
  /* Control and residual corrections, in the order offered to deflation. */
  Vec directions[2] = {newton->work->work[SS_NL_DIRECTION],
                       newton->work->work[SS_NL_RESIDUAL_DIRECTION]};
  PetscReal control_linear_relative_tolerance = NAN;
  PetscReal residual_linear_relative_tolerance = NAN;
  for (newton->iteration = 0; newton->iteration < policy->maximum_iterations;
       ++newton->iteration) {
    SsNonlinearNewtonStep step;
    SsLinearProblem linear_problem;
    SsLinearSetupRecord residual_linear_setup = {0};
    SsNonlinearControlledTrialContext trial_context;
    SsNonlinearIterationRecord *iteration_record;
    SsNonlinearStepVerdict verdict;
    const PetscReal control = newton->control;
    PetscReal updated_control = NAN;
    PetscReal work_after = NAN;
    int proceed;
    if (!ss_nonlinear_newton_start(newton, &step)) {
      return;
    }
    if (mode == SS_NONLINEAR_CONTROL_SSR_INDIRECT &&
        policy->indirect_progress_guard_maximum_convergence > 0.0 &&
        newton->iteration == policy->indirect_progress_guard_iteration &&
        step.convergence > policy->indirect_progress_guard_maximum_convergence &&
        (policy->indirect_progress_guard_maximum_convergence_ratio <= 0.0 ||
         step.convergence / record->initial_convergence >
             policy->indirect_progress_guard_maximum_convergence_ratio)) {
      record->progress_guard_triggered = 1;
      record->outcome = SS_NONLINEAR_FAILED_ITERATION_LIMIT;
      return;
    }
    if (!ss_nonlinear_newton_prepare(newton, &step, &linear_problem)) {
      return;
    }
    iteration_record = step.record;
    if (!ss_is_finite(control_linear_relative_tolerance)) {
      control_linear_relative_tolerance = iteration_record->linear_setup.effective_rtol;
    }
    if (!ss_is_finite(residual_linear_relative_tolerance)) {
      residual_linear_relative_tolerance =
          iteration_record->linear_setup.effective_rtol;
    }
    iteration_record->control_linear_relative_tolerance =
        control_linear_relative_tolerance;
    iteration_record->residual_linear_relative_tolerance =
        residual_linear_relative_tolerance;
    iteration_record->control_linear_next_relative_tolerance =
        control_linear_relative_tolerance;
    iteration_record->residual_linear_next_relative_tolerance =
        residual_linear_relative_tolerance;
    if (!ss_nonlinear_control_rhs(newton, &step, control_rhs)) {
      return;
    }
    if (ss_nonlinear_petsc_status(comm, VecCopy(residual, residual_rhs)) != SS_OK ||
        ss_nonlinear_petsc_status(comm, VecScale(residual_rhs, -1.0)) != SS_OK ||
        (newton->use_previous_linear_guess == 0 &&
         (ss_nonlinear_petsc_status(comm, VecSet(directions[0], 0.0)) != SS_OK ||
          ss_nonlinear_petsc_status(comm, VecSet(directions[1], 0.0)) != SS_OK))) {
      newton->status = SS_ERROR_INTERNAL;
      return;
    }
    {
      const Vec right_hand_sides[2] = {control_rhs, residual_rhs};
      if (!ss_nonlinear_newton_linear_diagnostic(newton, &linear_problem, 2U,
                                                 right_hand_sides)) {
        return;
      }
    }
    if (!ss_nonlinear_newton_linear_solve(newton, SS_NONLINEAR_SYSTEM_CONTROL,
                                          control_rhs, directions[0],
                                          &control_linear_relative_tolerance,
                                          &iteration_record->control_linear_solve) ||
        !ss_nonlinear_newton_promote(newton, directions[0],
                                     "ksp:flexible-control-correction", provenances[0],
                                     &iteration_record->control_linear_solve)) {
      return;
    }
    if (ss_nonlinear_linear_history_enabled(problem)) {
      /* The deflation space is immutable during one flexible-Krylov solve.  Rebuild
       * after the complete control space has been promoted so the residual
       * right-hand side sees the largest mathematically valid projector. */
      if (ss_linear_setup(newton->work->linear, &linear_problem,
                          &residual_linear_setup) != SS_OK) {
        record->outcome = SS_NONLINEAR_FAILED_LINEAR;
        return;
      }
    }
    proceed = ss_nonlinear_newton_linear_solve(
        newton, SS_NONLINEAR_SYSTEM_RESIDUAL, residual_rhs, directions[1],
        &residual_linear_relative_tolerance, &iteration_record->residual_linear_solve);
    /* The iteration's primary linear record is the residual solve before promotion. */
    iteration_record->linear_solve = iteration_record->residual_linear_solve;
    if (!proceed || !ss_nonlinear_newton_promote(
                        newton, directions[1], "ksp:flexible-residual-correction",
                        provenances[1], &iteration_record->residual_linear_solve)) {
      return;
    }
    newton->status = ss_nonlinear_augmented_direction(
        problem->reference_load, directions[0], directions[1],
        policy->singular_relative_tolerance, combined_direction,
        &iteration_record->augmented);
    if (newton->status != SS_OK) {
      return;
    }
    if (iteration_record->augmented.outcome != SS_NONLINEAR_CONVERGED) {
      record->outcome = iteration_record->augmented.outcome;
      return;
    }
    iteration_record->control_increment = iteration_record->augmented.control_increment;
    if (!ss_nonlinear_newton_measure(newton, &step, combined_direction)) {
      return;
    }
    trial_context = (SsNonlinearControlledTrialContext){
        .newton = newton,
        .target_work = target_work,
        .control = control,
        .control_increment = iteration_record->control_increment,
        .working = working,
        .direction = combined_direction,
        .trial = trial,
        .trial_residual = trial_residual,
        .legacy_alg5 = policy->line_search_legacy_alg5,
    };
    const SsNonlinearPolicy *line_policy = policy;
    SsNonlinearPolicy load_step_policy;
    if (mode == SS_NONLINEAR_CONTROL_LIMIT_LOAD &&
        policy->line_method == SS_NONLINEAR_LINE_DIRECTIONAL &&
        step.direction_norm <= 8.0 * PETSC_MACHINE_EPSILON * step.working_norm) {
      /* Work control can make the displacement direction zero while the load changes.
       * The directional work then vanishes, but the equilibrium residual can improve.
       */
      load_step_policy = *policy;
      load_step_policy.line_method = SS_NONLINEAR_LINE_RESIDUAL_MONOTONE;
      load_step_policy.line_search_legacy_alg5 = 0;
      line_policy = &load_step_policy;
    }
    verdict = ss_nonlinear_newton_line_search(
        newton, &step, line_policy, fabs(iteration_record->control_increment),
        ss_nonlinear_controlled_trial, &trial_context);
    if (verdict == SS_NONLINEAR_STEP_STOP) {
      return;
    }
    if (verdict == SS_NONLINEAR_STEP_REJECTED) {
      if (mode == SS_NONLINEAR_CONTROL_SSR_INDIRECT) {
        ss_nonlinear_update_linear_tolerances(policy, 0.0, iteration_record,
                                              &control_linear_relative_tolerance,
                                              &residual_linear_relative_tolerance);
      }
      iteration_record->control_after = control;
      iteration_record->work_after = target_work;
      if (!ss_nonlinear_newton_reject_step(newton, &step)) {
        return;
      }
      continue;
    }
    if (mode == SS_NONLINEAR_CONTROL_SSR_INDIRECT) {
      ss_nonlinear_update_linear_tolerances(
          policy, iteration_record->alpha, iteration_record,
          &control_linear_relative_tolerance, &residual_linear_relative_tolerance);
    }
    if (ss_nonlinear_apply_augmented_update(
            working, combined_direction, problem->reference_load, target_work, control,
            iteration_record->control_increment, iteration_record->alpha, trial,
            &updated_control, &work_after) != SS_OK) {
      record->outcome = SS_NONLINEAR_FAILED_NONFINITE;
      return;
    }
    if (mode == SS_NONLINEAR_CONTROL_SSR_INDIRECT && updated_control <= 0.0) {
      record->outcome = SS_NONLINEAR_FAILED_LINE_SEARCH;
      return;
    }
    iteration_record->control_after = updated_control;
    iteration_record->work_after = work_after;
    step.next_control = updated_control;
    if (!ss_nonlinear_newton_accept_step(newton, &step) ||
        !ss_nonlinear_newton_add_candidates(newton, &step, 2U, directions,
                                            provenances) ||
        !ss_nonlinear_newton_complete_step(newton, &step, trial,
                                           fabs(iteration_record->control_increment),
                                           iteration_record->alpha > 0.0)) {
      return;
    }
  }
}

SsStatus ss_nonlinear_work_controlled_solve(
    const SsNonlinearProblem *problem, const SsNonlinearPolicy *policy,
    SsNonlinearControlMode mode, PetscReal target_work, PetscReal incoming_control,
    Vec parent_state, Vec accepted_state, PetscReal *accepted_control,
    SsNonlinearSolveRecord *record) {
  SsNonlinearNewton newton;
  SsStatus status;
  if (problem == NULL || problem->communicator == MPI_COMM_NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (policy == NULL || accepted_control == NULL || record == NULL ||
      ss_nonlinear_policy_validate(policy) != SS_OK ||
      (mode != SS_NONLINEAR_CONTROL_LIMIT_LOAD &&
       mode != SS_NONLINEAR_CONTROL_SSR_INDIRECT) ||
      (mode == SS_NONLINEAR_CONTROL_LIMIT_LOAD &&
       policy->line_method != SS_NONLINEAR_LINE_DIRECTIONAL) ||
      (mode == SS_NONLINEAR_CONTROL_SSR_INDIRECT &&
       policy->line_method != SS_NONLINEAR_LINE_RESIDUAL_MONOTONE) ||
      !ss_is_finite(target_work) || target_work <= 0.0 ||
      !ss_is_finite(incoming_control) ||
      (mode == SS_NONLINEAR_CONTROL_SSR_INDIRECT && incoming_control <= 0.0) ||
      problem->operations.evaluate_controlled == NULL) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else {
    status = ss_nonlinear_newton_arguments(problem, parent_state, accepted_state);
  }
  status = ss_status_allreduce_max(problem->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  status = ss_nonlinear_newton_begin(
      &newton, problem, policy, &mode, incoming_control, parent_state, accepted_state,
      (mode == SS_NONLINEAR_CONTROL_SSR_INDIRECT &&
               policy->control_sensitivity_method ==
                   SS_NONLINEAR_CONTROL_SENSITIVITY_STABLE_CENTRAL
           ? SS_NL_VECTOR_COUNT
           : SS_NL_SENSITIVITY + 1U),
      record);
  record->progress_guard_iteration = policy->indirect_progress_guard_iteration;
  record->progress_guard_threshold =
      policy->indirect_progress_guard_maximum_convergence;
  record->progress_guard_ratio_threshold =
      policy->indirect_progress_guard_maximum_convergence_ratio;
  record->progress_guard_enabled =
      mode == SS_NONLINEAR_CONTROL_SSR_INDIRECT &&
      policy->indirect_progress_guard_maximum_convergence > 0.0;
  if (status == SS_OK) {
    newton.status = ss_nonlinear_initialize_work_state(
        problem->reference_load, target_work, parent_state,
        newton.work->work[SS_NL_WORKING]);
  }
  if (newton.status == SS_OK &&
      ss_nonlinear_newton_open_transaction(&newton) == SS_OK) {
    ss_nonlinear_controlled_iterate(&newton, target_work);
    if (newton.status == SS_OK) {
      ss_nonlinear_newton_finish(&newton, accepted_control);
    }
  }
  return ss_nonlinear_newton_end(&newton);
}
