/* Damped Newton iteration phases shared by the fixed and work-controlled solves.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "nonlinear_internal.h"

#include <math.h>
#include <string.h>

SsStatus ss_nonlinear_newton_arguments(const SsNonlinearProblem *problem,
                                       Vec parent_state, Vec accepted_state) {
  PetscInt load_size = 0;
  PetscInt parent_size = 0;
  PetscInt accepted_size = 0;
  PetscInt matrix_rows = 0;
  PetscInt matrix_columns = 0;
  if (problem == NULL || parent_state == NULL || accepted_state == NULL ||
      problem->communicator == MPI_COMM_NULL || problem->reference_load == NULL ||
      problem->elastic_reference == NULL ||
      problem->linear_properties.finite_action == 0 ||
      problem->linear_policies.count == 0U || problem->structure_generation == 0U ||
      problem->first_value_generation == 0U || problem->transaction_generation == 0U ||
      problem->accepted_generation == 0U ||
      problem->first_value_generation > UINT64_MAX - SS_NONLINEAR_HISTORY_CAPACITY ||
      !ss_nonlinear_same_communicator(problem->communicator,
                                      (PetscObject)problem->reference_load) ||
      !ss_nonlinear_same_communicator(problem->communicator,
                                      (PetscObject)problem->elastic_reference) ||
      !ss_nonlinear_same_communicator(problem->communicator,
                                      (PetscObject)parent_state) ||
      !ss_nonlinear_same_communicator(problem->communicator,
                                      (PetscObject)accepted_state)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if ((problem->deflation_mode != SS_NONLINEAR_DEFLATION_NONE &&
       problem->deflation_mode != SS_NONLINEAR_DEFLATION_OWNED_TRANSACTION &&
       problem->deflation_mode != SS_NONLINEAR_DEFLATION_EXTERNAL_TRANSACTION) ||
      !ss_nonlinear_deflation_promotion_valid(problem->deflation_promotion) ||
      (problem->deflation == NULL &&
       problem->deflation_mode != SS_NONLINEAR_DEFLATION_NONE) ||
      VecGetSize(problem->reference_load, &load_size) != PETSC_SUCCESS ||
      VecGetSize(parent_state, &parent_size) != PETSC_SUCCESS ||
      VecGetSize(accepted_state, &accepted_size) != PETSC_SUCCESS ||
      MatGetSize(problem->elastic_reference, &matrix_rows, &matrix_columns) !=
          PETSC_SUCCESS ||
      load_size <= 0 || load_size != parent_size || load_size != accepted_size ||
      load_size != matrix_rows || matrix_rows != matrix_columns) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return SS_OK;
}

SsStatus ss_nonlinear_newton_begin(SsNonlinearNewton *newton,
                                   const SsNonlinearProblem *problem,
                                   const SsNonlinearPolicy *policy,
                                   const SsNonlinearControlMode *mode,
                                   PetscReal control, Vec parent_state,
                                   Vec accepted_state, size_t vector_count,
                                   SsNonlinearSolveRecord *record) {
  const MPI_Comm comm = problem->communicator;
  Vec *work;
  memset(newton, 0, sizeof(*newton));
  newton->problem = problem;
  newton->policy = policy;
  newton->record = record;
  newton->communicator = comm;
  newton->parent_state = parent_state;
  newton->accepted_state = accepted_state;
  newton->control = control;
  newton->final_norm = NAN;
  newton->final_convergence = NAN;
  newton->final_relative_correction = NAN;
  if (mode != NULL) {
    newton->controlled = 1;
    newton->mode = *mode;
    newton->matrix_free = problem->operations.linearize_controlled != NULL;
  } else {
    newton->matrix_free = problem->operations.linearize != NULL;
  }
  newton->use_previous_linear_guess =
      ss_nonlinear_uses_previous_linear_guess(&problem->linear_policies);
  memset(record, 0, sizeof(*record));
  record->outcome = SS_NONLINEAR_FAILED_NONFINITE;
  record->final_relative_correction = NAN;
  record->factor = control;
  record->transaction_generation = problem->transaction_generation;
  record->accepted_generation = problem->accepted_generation;
  newton->regularization = policy->regularization_minimum;
  if (ss_nonlinear_petsc_status(comm, VecNorm(problem->reference_load, NORM_2,
                                              &newton->load_norm)) != SS_OK ||
      !ss_is_finite(newton->load_norm) || newton->load_norm <= 0.0) {
    newton->status = SS_ERROR_INVALID_ARGUMENT;
    return newton->status;
  }
  record->load_norm = newton->load_norm;
  newton->status =
      ss_nonlinear_workspace_prepare(problem, parent_state, newton->matrix_free,
                                     vector_count, &newton->work, &newton->temporary);
  if (newton->status != SS_OK) {
    return newton->status;
  }
  work = newton->work->work;
  newton->status = ss_status_allreduce_max(
      comm, VecCopy(parent_state, work[SS_NL_PARENT]) == PETSC_SUCCESS &&
                    VecCopy(accepted_state, work[SS_NL_ACCEPTED]) == PETSC_SUCCESS &&
                    VecCopy(parent_state, work[SS_NL_WORKING]) == PETSC_SUCCESS &&
                    VecSet(work[SS_NL_DIRECTION], 0.0) == PETSC_SUCCESS &&
                    (vector_count <= SS_NL_RESIDUAL_DIRECTION ||
                     VecSet(work[SS_NL_RESIDUAL_DIRECTION], 0.0) == PETSC_SUCCESS)
                ? SS_OK
                : SS_ERROR_INTERNAL);
  return newton->status;
}

SsStatus ss_nonlinear_newton_open_transaction(SsNonlinearNewton *newton) {
  const SsNonlinearProblem *problem = newton->problem;
  if (problem->deflation != NULL &&
      problem->deflation_mode == SS_NONLINEAR_DEFLATION_OWNED_TRANSACTION) {
    newton->status =
        ss_deflation_begin(problem->deflation, problem->transaction_generation);
    if (newton->status == SS_OK) {
      newton->transaction_open = 1;
    }
  }
  return newton->status;
}

SsStatus ss_nonlinear_newton_evaluate(const SsNonlinearNewton *newton, Vec state,
                                      PetscReal control, Vec residual, Mat tangent) {
  const SsNonlinearProblem *problem = newton->problem;
  SsStatus status;
  if (newton->controlled != 0) {
    status = problem->operations.evaluate_controlled(
        problem->context, newton->mode, state, control, residual, tangent, NULL);
  } else {
    status = problem->operations.evaluate(problem->context, state, control, residual,
                                          tangent, NULL);
  }
  return ss_status_allreduce_max(newton->communicator, status);
}

SsStatus ss_nonlinear_newton_trial_residual(const SsNonlinearNewton *newton, Vec trial,
                                            PetscReal control, Vec trial_residual,
                                            int *admissible) {
  const SsNonlinearProblem *problem = newton->problem;
  SsStatus status;
  if (problem->operations.admissibility != NULL) {
    int global_admissible = 0;
    status =
        problem->operations.admissibility(problem->context, trial, control, admissible);
    status = ss_status_allreduce_max(newton->communicator, status);
    if (status != SS_OK) {
      return status;
    }
    status = ss_nonlinear_collective_boolean(newton->communicator, *admissible, MPI_MIN,
                                             &global_admissible);
    if (status != SS_OK) {
      return status;
    }
    *admissible = global_admissible;
  } else {
    *admissible = 1;
  }
  return ss_nonlinear_newton_evaluate(newton, trial, control, trial_residual, NULL);
}

int ss_nonlinear_newton_start(SsNonlinearNewton *newton, SsNonlinearNewtonStep *step) {
  const SsNonlinearProblem *problem = newton->problem;
  SsNonlinearSolveRecord *record = newton->record;
  const MPI_Comm comm = newton->communicator;
  Vec working = newton->work->work[SS_NL_WORKING];
  Vec residual = newton->work->work[SS_NL_RESIDUAL];
  int cancelled = 0;
  memset(step, 0, sizeof(*step));
  step->direction_norm = NAN;
  step->working_norm = NAN;
  step->regularization_before = newton->regularization;
  step->next_control = newton->control;
  step->tangent_reused = newton->reuse_current_tangent;
  newton->reuse_current_tangent = 0;
  if (problem->operations.cancelled != NULL) {
    newton->status = ss_nonlinear_collective_boolean(
        comm, problem->operations.cancelled(problem->context), MPI_MAX, &cancelled);
    if (newton->status != SS_OK) {
      return 0;
    }
  }
  if (cancelled != 0) {
    record->outcome = SS_NONLINEAR_CANCELLED;
    return 0;
  }
  if (step->tangent_reused == 0) {
    const SsStatus status = ss_nonlinear_newton_evaluate(
        newton, working, newton->control, residual,
        newton->matrix_free != 0 ? NULL : newton->work->tangent);
    if (status != SS_OK) {
      record->outcome = ss_nonlinear_status_outcome(status);
      return 0;
    }
  }
  if (ss_nonlinear_petsc_status(
          comm, VecNorm(residual, NORM_2, &step->residual_norm)) != SS_OK ||
      !ss_is_finite(step->residual_norm)) {
    record->outcome = SS_NONLINEAR_FAILED_NONFINITE;
    return 0;
  }
  step->convergence = step->residual_norm / newton->load_norm;
  if (!ss_is_finite(step->convergence)) {
    record->outcome = SS_NONLINEAR_FAILED_NONFINITE;
    return 0;
  }
  if (newton->iteration == 0) {
    record->initial_residual_norm = step->residual_norm;
    record->initial_convergence = step->convergence;
  }
  newton->final_norm = step->residual_norm;
  newton->final_convergence = step->convergence;
  if (ss_nonlinear_stopping_at_current(newton->policy, step->residual_norm,
                                       step->convergence, newton->controlled,
                                       &newton->final_relative_correction)) {
    record->outcome = SS_NONLINEAR_CONVERGED;
    newton->succeeded = 1;
    return 0;
  }
  return 1;
}

int ss_nonlinear_newton_prepare(SsNonlinearNewton *newton, SsNonlinearNewtonStep *step,
                                SsLinearProblem *linear_problem) {
  const SsNonlinearProblem *problem = newton->problem;
  SsNonlinearSolveRecord *record = newton->record;
  SsNonlinearIterationRecord *iteration_record =
      &record->history[record->history_count];
  const MPI_Comm comm = newton->communicator;
  Vec working = newton->work->work[SS_NL_WORKING];
  Mat regularized = newton->work->regularized;
  memset(iteration_record, 0, sizeof(*iteration_record));
  iteration_record->ordinal = newton->iteration;
  iteration_record->control_before = newton->control;
  iteration_record->residual_before = step->residual_norm;
  iteration_record->convergence_before = step->convergence;
  iteration_record->regularization_before = step->regularization_before;
  iteration_record->tangent_reused = step->tangent_reused;
  step->record = iteration_record;
  if (newton->matrix_free != 0) {
    SsStatus status;
    if (newton->controlled != 0) {
      status = problem->operations.linearize_controlled(problem->context, newton->mode,
                                                        working, newton->control,
                                                        newton->regularization);
    } else {
      status = problem->operations.linearize(problem->context, working, newton->control,
                                             newton->regularization);
    }
    status = ss_status_allreduce_max(comm, status);
    if (status != SS_OK) {
      record->outcome = ss_nonlinear_status_outcome(status);
      return 0;
    }
  } else if (ss_nonlinear_petsc_status(comm, MatCopy(newton->work->tangent, regularized,
                                                     SAME_NONZERO_PATTERN)) != SS_OK ||
             ss_nonlinear_petsc_status(
                 comm, MatScale(regularized, 1.0 - newton->regularization)) != SS_OK ||
             ss_nonlinear_petsc_status(comm,
                                       MatAXPY(regularized, newton->regularization,
                                               problem->elastic_reference,
                                               SAME_NONZERO_PATTERN)) != SS_OK) {
    record->outcome = SS_NONLINEAR_FAILED_NONFINITE;
    return 0;
  }
  ++newton->operator_generation;
  iteration_record->operator_value_generation =
      problem->first_value_generation + newton->operator_generation - 1U;
  memset(linear_problem, 0, sizeof(*linear_problem));
  linear_problem->operator_handle = regularized;
  linear_problem->preconditioner_operator_handle =
      problem->pmg == NULL ? NULL : regularized;
  linear_problem->properties = problem->linear_properties;
  linear_problem->policy_chain = problem->linear_policies;
  linear_problem->deflation = problem->deflation;
  linear_problem->pmg = problem->pmg;
  linear_problem->krylov_recycle = problem->krylov_recycle;
  linear_problem->krylov_capture = ss_nonlinear_full_krylov_enabled(problem)
                                       ? SS_LINEAR_KRYLOV_CAPTURE_FULL_CORRECTION_SPACE
                                       : SS_LINEAR_KRYLOV_CAPTURE_NONE;
  linear_problem->structure_generation = problem->structure_generation;
  linear_problem->value_generation = iteration_record->operator_value_generation;
  if (ss_linear_setup(newton->work->linear, linear_problem,
                      &iteration_record->linear_setup) != SS_OK) {
    record->outcome = SS_NONLINEAR_FAILED_LINEAR;
    return 0;
  }
  return 1;
}

int ss_nonlinear_newton_linear_diagnostic(SsNonlinearNewton *newton,
                                          const SsLinearProblem *linear_problem,
                                          size_t rhs_count,
                                          const Vec *right_hand_sides) {
  const SsNonlinearProblem *problem = newton->problem;
  SsStatus status;
  if (problem->operations.linear_diagnostic == NULL) {
    return 1;
  }
  status = problem->operations.linear_diagnostic(
      problem->context, linear_problem, newton->work->work[SS_NL_WORKING],
      problem->transaction_generation, newton->iteration, newton->control,
      newton->regularization, rhs_count, right_hand_sides);
  newton->status = ss_status_allreduce_max(newton->communicator, status);
  return newton->status == SS_OK;
}

int ss_nonlinear_newton_linear_solve(SsNonlinearNewton *newton,
                                     SsNonlinearSystem system, Vec rhs, Vec solution,
                                     const PetscReal *relative_tolerance,
                                     SsLinearSolveRecord *solve_record) {
  SsLinearSolver *linear = newton->work->linear;
  SsStatus status;
  ss_nonlinear_observe_linear(newton->problem, linear, newton->iteration, system);
  status = relative_tolerance == NULL
               ? ss_linear_solve(linear, rhs, solution, solve_record)
               : ss_linear_solve_with_relative_tolerance(
                     linear, rhs, solution, *relative_tolerance, solve_record);
  ss_nonlinear_observe_linear_result(newton->problem, status, solve_record);
  if (status != SS_OK || solve_record->status != SS_LINEAR_SOLVE_CONVERGED) {
    newton->record->outcome = status == SS_ERROR_CANCELLED ? SS_NONLINEAR_CANCELLED
                                                           : SS_NONLINEAR_FAILED_LINEAR;
    return 0;
  }
  return 1;
}

int ss_nonlinear_newton_promote(SsNonlinearNewton *newton, Vec solution,
                                const char *krylov_provenance,
                                const char *solution_provenance,
                                SsLinearSolveRecord *solve_record) {
  newton->status = ss_nonlinear_promote_linear_history(
      newton->problem, newton->work->linear, solution, krylov_provenance,
      solution_provenance, solve_record);
  return newton->status == SS_OK;
}

int ss_nonlinear_newton_measure(SsNonlinearNewton *newton, SsNonlinearNewtonStep *step,
                                Vec direction) {
  const MPI_Comm comm = newton->communicator;
  if (ss_nonlinear_petsc_status(
          comm, VecNorm(direction, NORM_2, &step->direction_norm)) != SS_OK ||
      ss_nonlinear_petsc_status(comm, VecNorm(newton->work->work[SS_NL_WORKING], NORM_2,
                                              &step->working_norm)) != SS_OK ||
      !ss_is_finite(step->direction_norm) || !ss_is_finite(step->working_norm)) {
    newton->record->outcome = SS_NONLINEAR_FAILED_NONFINITE;
    return 0;
  }
  return 1;
}

SsNonlinearStepVerdict
ss_nonlinear_newton_line_search(SsNonlinearNewton *newton, SsNonlinearNewtonStep *step,
                                const SsNonlinearPolicy *line_policy,
                                PetscReal delta_lambda,
                                SsNonlinearTrialEvaluation trial, void *trial_context) {
  SsNonlinearIterationRecord *iteration_record = step->record;
  const PetscReal residual_floor = ss_nonlinear_line_search_floor(
      newton->policy, newton->load_norm, step->convergence,
      step->direction_norm / fmax(step->working_norm, 1.0e-30), delta_lambda,
      newton->controlled);
  const SsStatus status = ss_nonlinear_line_search_with_floor(
      line_policy, step->residual_norm, residual_floor, trial, trial_context,
      &iteration_record->line_search);
  if (status != SS_OK) {
    newton->record->outcome = ss_nonlinear_status_outcome(status);
    return SS_NONLINEAR_STEP_STOP;
  }
  if (iteration_record->line_search.outcome != SS_NONLINEAR_CONVERGED) {
    if (iteration_record->line_search.outcome != SS_NONLINEAR_FAILED_LINE_SEARCH) {
      newton->record->outcome = iteration_record->line_search.outcome;
      return SS_NONLINEAR_STEP_STOP;
    }
    iteration_record->alpha = 0.0;
    return SS_NONLINEAR_STEP_REJECTED;
  }
  iteration_record->alpha = iteration_record->line_search.accepted_alpha;
  return SS_NONLINEAR_STEP_ACCEPTED;
}

int ss_nonlinear_newton_reject_step(SsNonlinearNewton *newton,
                                    SsNonlinearNewtonStep *step) {
  SsNonlinearSolveRecord *record = newton->record;
  SsNonlinearIterationRecord *iteration_record = step->record;
  iteration_record->residual_after = step->residual_norm;
  iteration_record->convergence_after = step->convergence;
  newton->status = ss_nonlinear_regularization_update(
      step->regularization_before, newton->policy->regularization_minimum, 0.0,
      &step->regularization);
  if (newton->status != SS_OK) {
    return 0;
  }
  newton->regularization = step->regularization.next_value;
  iteration_record->regularization_after = newton->regularization;
  ss_nonlinear_observe_iteration(newton->problem, iteration_record);
  ++record->history_count;
  record->iterations = newton->iteration + 1;
  newton->final_norm = step->residual_norm;
  newton->final_convergence = step->convergence;
  if (step->regularization.outcome == SS_NONLINEAR_FAILED_REGULARIZATION) {
    record->outcome = SS_NONLINEAR_FAILED_REGULARIZATION;
    return 0;
  }
  newton->reuse_current_tangent = 1;
  return 1;
}

int ss_nonlinear_newton_accept_step(SsNonlinearNewton *newton,
                                    SsNonlinearNewtonStep *step) {
  SsNonlinearIterationRecord *iteration_record = step->record;
  iteration_record->residual_after =
      iteration_record->line_search.accepted_residual_norm;
  iteration_record->convergence_after =
      iteration_record->residual_after / newton->load_norm;
  iteration_record->relative_correction = iteration_record->alpha *
                                          step->direction_norm /
                                          fmax(step->working_norm, 1.0e-30);
  if (!ss_is_finite(iteration_record->relative_correction)) {
    newton->record->outcome = SS_NONLINEAR_FAILED_NONFINITE;
    return 0;
  }
  newton->status = ss_nonlinear_regularization_update(
      step->regularization_before, newton->policy->regularization_minimum,
      iteration_record->alpha, &step->regularization);
  if (newton->status != SS_OK) {
    return 0;
  }
  newton->regularization = step->regularization.next_value;
  iteration_record->regularization_after = newton->regularization;
  iteration_record->deflation_candidate =
      step->regularization.deflation_candidate_eligible;
  if (step->regularization.outcome == SS_NONLINEAR_FAILED_REGULARIZATION) {
    newton->record->outcome = SS_NONLINEAR_FAILED_REGULARIZATION;
    return 0;
  }
  return 1;
}

int ss_nonlinear_newton_add_candidates(SsNonlinearNewton *newton,
                                       const SsNonlinearNewtonStep *step, size_t count,
                                       const Vec *directions,
                                       const char *const *provenances) {
  const SsNonlinearProblem *problem = newton->problem;
  size_t index;
  if (problem->deflation == NULL ||
      problem->deflation_mode == SS_NONLINEAR_DEFLATION_NONE ||
      problem->deflation_promotion ==
          SS_NONLINEAR_DEFLATION_PROMOTE_FULL_KRYLOV_AND_ACCEPTED_STATE ||
      problem->deflation_promotion ==
          SS_NONLINEAR_DEFLATION_PROMOTE_SYSTEM_SOLUTIONS_AND_ACCEPTED_STATE ||
      step->regularization.deflation_candidate_eligible == 0) {
    return 1;
  }
  for (index = 0U; index < count; ++index) {
    SsDeflationAdmissionRecord admission;
    newton->status =
        ss_deflation_add_candidate(problem->deflation, directions[index],
                                   directions[index], provenances[index], &admission);
    if (newton->status != SS_OK) {
      return 0;
    }
  }
  return 1;
}

int ss_nonlinear_newton_complete_step(SsNonlinearNewton *newton,
                                      const SsNonlinearNewtonStep *step, Vec trial,
                                      PetscReal delta_lambda, int have_correction) {
  SsNonlinearSolveRecord *record = newton->record;
  const SsNonlinearIterationRecord *iteration_record = step->record;
  if (ss_nonlinear_petsc_status(newton->communicator,
                                VecCopy(trial, newton->work->work[SS_NL_WORKING])) !=
      SS_OK) {
    newton->status = SS_ERROR_INTERNAL;
    return 0;
  }
  newton->control = step->next_control;
  ss_nonlinear_observe_iteration(newton->problem, iteration_record);
  ++record->history_count;
  record->iterations = newton->iteration + 1;
  newton->final_norm = iteration_record->residual_after;
  newton->final_convergence = iteration_record->convergence_after;
  newton->final_relative_correction = iteration_record->relative_correction;
  if (ss_nonlinear_stopping_satisfied(
          newton->policy, iteration_record->convergence_after,
          iteration_record->relative_correction, delta_lambda, have_correction,
          newton->controlled)) {
    record->outcome = SS_NONLINEAR_CONVERGED;
    newton->succeeded = 1;
    return 0;
  }
  return 1;
}

/* Re-evaluate the state left by an exhausted iteration budget and classify it. */
static void ss_nonlinear_newton_iteration_limit(SsNonlinearNewton *newton) {
  const SsNonlinearPolicy *policy = newton->policy;
  SsNonlinearSolveRecord *record = newton->record;
  Vec residual = newton->work->work[SS_NL_RESIDUAL];
  const SsStatus status = ss_nonlinear_newton_evaluate(
      newton, newton->work->work[SS_NL_WORKING], newton->control, residual, NULL);
  if (status != SS_OK ||
      ss_nonlinear_petsc_status(newton->communicator,
                                VecNorm(residual, NORM_2, &newton->final_norm)) !=
          SS_OK ||
      !ss_is_finite(newton->final_norm)) {
    record->outcome = SS_NONLINEAR_FAILED_NONFINITE;
    return;
  }
  newton->final_convergence = newton->final_norm / newton->load_norm;
  if (ss_nonlinear_stopping_at_current(policy, newton->final_norm,
                                       newton->final_convergence, newton->controlled,
                                       &newton->final_relative_correction)) {
    record->outcome = SS_NONLINEAR_CONVERGED;
    newton->succeeded = 1;
  } else if (policy->allow_relaxed != 0 &&
             newton->final_convergence <=
                 ss_nonlinear_relaxed_residual_threshold(policy)) {
    record->outcome = SS_NONLINEAR_ACCEPTED_RELAXED;
    newton->succeeded = 1;
  } else {
    record->outcome = SS_NONLINEAR_FAILED_ITERATION_LIMIT;
  }
}

void ss_nonlinear_newton_finish(SsNonlinearNewton *newton,
                                PetscReal *accepted_control) {
  const SsNonlinearProblem *problem = newton->problem;
  SsNonlinearSolveRecord *record = newton->record;
  Vec working = newton->work->work[SS_NL_WORKING];
  Vec accepted_snapshot = newton->work->work[SS_NL_ACCEPTED];
  if (!newton->succeeded && newton->iteration == newton->policy->maximum_iterations) {
    ss_nonlinear_newton_iteration_limit(newton);
  }
  if (newton->succeeded) {
    newton->status = ss_nonlinear_prepare_deflation_commit(
        problem, working, newton->work->regularized,
        newton->operator_generation == 0U
            ? 0U
            : problem->first_value_generation + newton->operator_generation - 1U);
    if (newton->status != SS_OK) {
      return;
    }
    if (ss_nonlinear_petsc_status(newton->communicator,
                                  VecCopy(working, newton->accepted_state)) != SS_OK) {
      (void)VecCopy(accepted_snapshot, newton->accepted_state);
      newton->status = SS_ERROR_INTERNAL;
      return;
    }
    if (newton->transaction_open != 0) {
      newton->status =
          ss_deflation_commit(problem->deflation, problem->accepted_generation);
      if (newton->status != SS_OK) {
        (void)VecCopy(accepted_snapshot, newton->accepted_state);
        return;
      }
      newton->transaction_open = 0;
    }
    if (accepted_control != NULL) {
      *accepted_control = newton->control;
    }
    record->transaction_committed = 1;
  } else {
    if (newton->transaction_open != 0) {
      newton->status = ss_deflation_rollback(problem->deflation);
      if (newton->status != SS_OK) {
        return;
      }
      newton->transaction_open = 0;
    }
    record->transaction_rolled_back = 1;
    newton->status = ss_nonlinear_parent_equal(newton->parent_state,
                                               newton->work->work[SS_NL_PARENT],
                                               &record->parent_unchanged_on_failure);
    if (newton->status != SS_OK) {
      return;
    }
  }
  record->factor = newton->control;
  record->final_residual_norm = newton->final_norm;
  record->final_convergence = newton->final_convergence;
  record->final_relative_correction = newton->final_relative_correction;
  record->final_regularization = newton->regularization;
  record->regularized_operator_generation = newton->operator_generation;
}

SsStatus ss_nonlinear_newton_end(SsNonlinearNewton *newton) {
  SsStatus status = newton->status;
  SsStatus cleanup_status;
  if (newton->transaction_open != 0) {
    (void)ss_deflation_rollback(newton->problem->deflation);
  }
  if (newton->work != NULL) {
    const SsStatus finish = ss_linear_finish_attempt(newton->work->linear);
    if (status == SS_OK) {
      status = finish;
    }
  }
  cleanup_status = ss_nonlinear_workspace_destroy(&newton->temporary);
  return ss_status_allreduce_max(newton->communicator,
                                 status == SS_OK ? cleanup_status : status);
}
