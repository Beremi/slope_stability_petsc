/* Fixed-control nonlinear equilibrium solve.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "nonlinear_internal.h"

#include <math.h>

typedef struct {
  const SsNonlinearNewton *newton;
  Vec working;
  Vec direction;
  Vec trial;
  Vec trial_residual;
} SsNonlinearVectorTrialContext;

/* Line-search trial working + alpha * direction at the fixed factor. */
static SsStatus ss_nonlinear_vector_trial(void *opaque, PetscReal alpha,
                                          PetscReal *directional_value,
                                          PetscReal *residual_norm, int *admissible) {
  SsNonlinearVectorTrialContext *context = (SsNonlinearVectorTrialContext *)opaque;
  const MPI_Comm comm = context->newton->communicator;
  PetscScalar directional = 0.0;
  SsStatus status;
  if (ss_nonlinear_petsc_status(comm, VecWAXPY(context->trial, alpha,
                                               context->direction, context->working)) !=
      SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  status = ss_nonlinear_newton_trial_residual(context->newton, context->trial,
                                              context->newton->control,
                                              context->trial_residual, admissible);
  if (status != SS_OK) {
    return status;
  }
  if (ss_nonlinear_petsc_status(
          comm, VecDot(context->trial_residual, context->direction, &directional)) !=
          SS_OK ||
      ss_nonlinear_petsc_status(
          comm, VecNorm(context->trial_residual, NORM_2, residual_norm)) != SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  *directional_value = PetscRealPart(directional);
  return SS_OK;
}

/* Newton iterations with one correction system per step. */
static void ss_nonlinear_fixed_iterate(SsNonlinearNewton *newton) {
  static const char *const candidate_provenances[1] = {"nls:newton-direction"};
  const MPI_Comm comm = newton->communicator;
  Vec working = newton->work->work[SS_NL_WORKING];
  Vec residual = newton->work->work[SS_NL_RESIDUAL];
  Vec rhs = newton->work->work[SS_NL_RHS];
  Vec direction = newton->work->work[SS_NL_DIRECTION];
  Vec trial = newton->work->work[SS_NL_TRIAL];
  Vec trial_residual = newton->work->work[SS_NL_TRIAL_RESIDUAL];
  for (newton->iteration = 0; newton->iteration < newton->policy->maximum_iterations;
       ++newton->iteration) {
    SsNonlinearNewtonStep step;
    SsLinearProblem linear_problem;
    SsNonlinearVectorTrialContext trial_context;
    SsNonlinearIterationRecord *iteration_record;
    SsNonlinearStepVerdict verdict;
    if (!ss_nonlinear_newton_start(newton, &step) ||
        !ss_nonlinear_newton_prepare(newton, &step, &linear_problem)) {
      return;
    }
    iteration_record = step.record;
    if (ss_nonlinear_petsc_status(comm, VecCopy(residual, rhs)) != SS_OK ||
        ss_nonlinear_petsc_status(comm, VecScale(rhs, -1.0)) != SS_OK ||
        (newton->use_previous_linear_guess == 0 &&
         ss_nonlinear_petsc_status(comm, VecSet(direction, 0.0)) != SS_OK)) {
      newton->status = SS_ERROR_INTERNAL;
      return;
    }
    if (!ss_nonlinear_newton_linear_diagnostic(newton, &linear_problem, 1U, &rhs) ||
        !ss_nonlinear_newton_linear_solve(newton, SS_NONLINEAR_SYSTEM_FIXED, rhs,
                                          direction, NULL,
                                          &iteration_record->linear_solve) ||
        !ss_nonlinear_newton_promote(newton, direction, "ksp:flexible-correction",
                                     "nls:newton-direction",
                                     &iteration_record->linear_solve) ||
        !ss_nonlinear_newton_measure(newton, &step, direction)) {
      return;
    }
    trial_context = (SsNonlinearVectorTrialContext){
        .newton = newton,
        .working = working,
        .direction = direction,
        .trial = trial,
        .trial_residual = trial_residual,
    };
    verdict = ss_nonlinear_newton_line_search(
        newton, &step, newton->policy, NAN, ss_nonlinear_vector_trial, &trial_context);
    if (verdict == SS_NONLINEAR_STEP_STOP) {
      return;
    }
    if (verdict == SS_NONLINEAR_STEP_REJECTED) {
      if (!ss_nonlinear_newton_reject_step(newton, &step)) {
        return;
      }
      continue;
    }
    if (!ss_nonlinear_newton_accept_step(newton, &step) ||
        !ss_nonlinear_newton_add_candidates(newton, &step, 1U, &direction,
                                            candidate_provenances)) {
      return;
    }
    if (ss_nonlinear_petsc_status(comm, VecWAXPY(trial, iteration_record->alpha,
                                                 direction, working)) != SS_OK) {
      newton->status = SS_ERROR_INTERNAL;
      return;
    }
    if (!ss_nonlinear_newton_complete_step(newton, &step, trial, NAN,
                                           iteration_record->alpha >= 0.1)) {
      return;
    }
  }
}

SsStatus ss_nonlinear_fixed_solve(const SsNonlinearProblem *problem,
                                  const SsNonlinearPolicy *policy, PetscReal factor,
                                  Vec parent_state, Vec accepted_state,
                                  SsNonlinearSolveRecord *record) {
  SsNonlinearNewton newton;
  SsStatus status;
  if (problem == NULL || problem->communicator == MPI_COMM_NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (policy == NULL || record == NULL ||
      ss_nonlinear_policy_validate(policy) != SS_OK || !ss_is_finite(factor) ||
      factor <= 0.0 || problem->operations.evaluate == NULL) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else {
    status = ss_nonlinear_newton_arguments(problem, parent_state, accepted_state);
  }
  status = ss_status_allreduce_max(problem->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  if (ss_nonlinear_newton_begin(&newton, problem, policy, NULL, factor, parent_state,
                                accepted_state, SS_NL_TRIAL_RESIDUAL + 1U,
                                record) == SS_OK &&
      ss_nonlinear_newton_open_transaction(&newton) == SS_OK) {
    ss_nonlinear_fixed_iterate(&newton);
    if (newton.status == SS_OK) {
      ss_nonlinear_newton_finish(&newton, NULL);
    }
  }
  return ss_nonlinear_newton_end(&newton);
}
