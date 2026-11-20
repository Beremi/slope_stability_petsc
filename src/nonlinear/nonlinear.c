/* Common nonlinear helpers and elastic prediction.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "nonlinear_internal.h"
#include "slope_internal/diagnostics/diagnostics.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

int ss_nonlinear_uses_previous_linear_guess(const SsLinearPolicyChain *policies) {
  size_t index;
  if (policies == NULL) {
    return 0;
  }
  for (index = 0U; index < policies->count; ++index) {
    if (policies->policies[index].initial_guess_policy !=
        SS_LINEAR_INITIAL_GUESS_COARSE_ONLY) {
      return 1;
    }
  }
  return 0;
}

SsStatus ss_nonlinear_collective_boolean(MPI_Comm communicator, int local_value,
                                         MPI_Op operation, int *global_value) {
  int local = local_value != 0 ? 1 : 0;
  if (MPI_Allreduce(&local, global_value, 1, MPI_INT, operation, communicator) !=
      MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  return SS_OK;
}

int ss_nonlinear_same_communicator(MPI_Comm expected, PetscObject object) {
  MPI_Comm actual;
  int comparison = MPI_UNEQUAL;
  if (expected == MPI_COMM_NULL || object == NULL) {
    return 0;
  }
  actual = PetscObjectComm(object);
  return MPI_Comm_compare(expected, actual, &comparison) == MPI_SUCCESS &&
         (comparison == MPI_IDENT || comparison == MPI_CONGRUENT);
}

int ss_nonlinear_stopping_satisfied(const SsNonlinearPolicy *policy,
                                    PetscReal relative_residual,
                                    PetscReal relative_correction,
                                    PetscReal absolute_delta_lambda,
                                    int have_correction, int have_delta_lambda) {
  if (policy->stopping_criterion == SS_NONLINEAR_STOP_RELATIVE_RESIDUAL) {
    return ss_is_finite(relative_residual) &&
           relative_residual <= policy->stopping_tolerance;
  }
  if (policy->stopping_criterion == SS_NONLINEAR_STOP_RELATIVE_CORRECTION) {
    return have_correction != 0 && ss_is_finite(relative_correction) &&
           relative_correction <= policy->stopping_tolerance;
  }
  return have_delta_lambda != 0 && ss_is_finite(absolute_delta_lambda) &&
         absolute_delta_lambda <= policy->stopping_tolerance;
}

int ss_nonlinear_stopping_at_current(const SsNonlinearPolicy *policy,
                                     PetscReal residual_norm,
                                     PetscReal relative_residual, int have_control,
                                     PetscReal *recorded_correction) {
  const int equilibrium = residual_norm == 0.0;
  if (!ss_nonlinear_stopping_satisfied(policy, relative_residual, 0.0, 0.0, equilibrium,
                                       equilibrium && have_control)) {
    return 0;
  }
  if (equilibrium &&
      policy->stopping_criterion != SS_NONLINEAR_STOP_RELATIVE_RESIDUAL) {
    *recorded_correction = 0.0;
  }
  return 1;
}

PetscReal ss_nonlinear_line_search_floor(
    const SsNonlinearPolicy *policy, PetscReal load_norm, PetscReal relative_residual,
    PetscReal relative_correction, PetscReal absolute_delta_lambda, int have_control) {
  const PetscReal relative_floor = 8.0 * PETSC_MACHINE_EPSILON;
  if (policy->stopping_criterion != SS_NONLINEAR_STOP_RELATIVE_RESIDUAL &&
      relative_residual <= relative_floor &&
      ss_nonlinear_stopping_satisfied(policy, relative_residual, relative_correction,
                                      absolute_delta_lambda, 1, have_control)) {
    return relative_floor * load_norm;
  }
  return 0.0;
}

PetscReal ss_nonlinear_relaxed_residual_threshold(const SsNonlinearPolicy *policy) {
  const PetscReal residual_tolerance =
      policy->stopping_criterion == SS_NONLINEAR_STOP_RELATIVE_RESIDUAL
          ? policy->stopping_tolerance
          : policy->tolerance;
  return policy->relaxed_factor * residual_tolerance;
}

int ss_nonlinear_deflation_promotion_valid(SsNonlinearDeflationPromotion promotion) {
  return promotion == SS_NONLINEAR_DEFLATION_PROMOTE_DIRECTIONS ||
         promotion == SS_NONLINEAR_DEFLATION_PROMOTE_ACCEPTED_STATE ||
         promotion == SS_NONLINEAR_DEFLATION_DISCARD_PROVISIONAL ||
         promotion == SS_NONLINEAR_DEFLATION_PROMOTE_FULL_KRYLOV_AND_ACCEPTED_STATE ||
         promotion ==
             SS_NONLINEAR_DEFLATION_PROMOTE_SYSTEM_SOLUTIONS_AND_ACCEPTED_STATE;
}

int ss_nonlinear_full_krylov_enabled(const SsNonlinearProblem *problem) {
  return problem != NULL && problem->deflation != NULL &&
         problem->deflation_mode != SS_NONLINEAR_DEFLATION_NONE &&
         problem->deflation_promotion ==
             SS_NONLINEAR_DEFLATION_PROMOTE_FULL_KRYLOV_AND_ACCEPTED_STATE;
}

int ss_nonlinear_system_solutions_enabled(const SsNonlinearProblem *problem) {
  return problem != NULL && problem->deflation != NULL &&
         problem->deflation_mode != SS_NONLINEAR_DEFLATION_NONE &&
         problem->deflation_promotion ==
             SS_NONLINEAR_DEFLATION_PROMOTE_SYSTEM_SOLUTIONS_AND_ACCEPTED_STATE;
}

int ss_nonlinear_linear_history_enabled(const SsNonlinearProblem *problem) {
  return ss_nonlinear_full_krylov_enabled(problem) ||
         ss_nonlinear_system_solutions_enabled(problem);
}

SsStatus ss_nonlinear_promote_linear_history(const SsNonlinearProblem *problem,
                                             SsLinearSolver *linear, Vec solution,
                                             const char *krylov_provenance,
                                             const char *solution_provenance,
                                             SsLinearSolveRecord *solve_record) {
  SsLinearKrylovPromotionRecord promotion = {0};
  SsDeflationAdmissionRecord admission = {0};
  SsDeflationDiagnostics diagnostics = {0};
  SsStatus status;
  if (!ss_nonlinear_linear_history_enabled(problem)) {
    return SS_OK;
  }
  if (ss_nonlinear_full_krylov_enabled(problem)) {
    status = ss_linear_promote_captured_krylov(linear, krylov_provenance, solve_record,
                                               &promotion);
    if (status != SS_OK) {
      return status;
    }
  }
  status = ss_deflation_add_weighted_candidate(
      problem->deflation, solution, solution,
      (PetscReal)(ss_nonlinear_full_krylov_enabled(problem) &&
                          promotion.captured_count != 0U
                      ? promotion.captured_count
                      : 1U),
      solution_provenance, &admission);
  if (status != SS_OK) {
    return status;
  }
  status = ss_deflation_diagnostics(problem->deflation, &diagnostics);
  if (status == SS_OK) {
    solve_record->krylov_basis_after_promotion = diagnostics.active_count;
  }
  return status;
}

SsStatus ss_nonlinear_prepare_deflation_commit(const SsNonlinearProblem *problem,
                                               Vec accepted_state, Mat operator_handle,
                                               uint64_t operator_generation) {
  SsDeflationAdmissionRecord admission = {0};
  SsDeflationProjector projector = SS_DEFLATION_PROJECTOR_GENERAL_LEFT_RIGHT;
  SsDeflationCompression compression = SS_DEFLATION_COMPRESSION_NONE;
  SsStatus status;
  if (problem->deflation == NULL ||
      problem->deflation_mode == SS_NONLINEAR_DEFLATION_NONE) {
    return SS_OK;
  }
  status = ss_deflation_projector(problem->deflation, &projector);
  if (status != SS_OK) {
    return status;
  }
  status = ss_deflation_compression(problem->deflation, &compression);
  if (status != SS_OK) {
    return status;
  }
  if (problem->deflation_promotion != SS_NONLINEAR_DEFLATION_PROMOTE_DIRECTIONS &&
      problem->deflation_promotion !=
          SS_NONLINEAR_DEFLATION_PROMOTE_FULL_KRYLOV_AND_ACCEPTED_STATE &&
      problem->deflation_promotion !=
          SS_NONLINEAR_DEFLATION_PROMOTE_SYSTEM_SOLUTIONS_AND_ACCEPTED_STATE) {
    status = ss_deflation_discard_provisional(problem->deflation);
    if (status != SS_OK) {
      return status;
    }
  }
  if (problem->deflation_promotion == SS_NONLINEAR_DEFLATION_PROMOTE_ACCEPTED_STATE ||
      problem->deflation_promotion ==
          SS_NONLINEAR_DEFLATION_PROMOTE_FULL_KRYLOV_AND_ACCEPTED_STATE ||
      problem->deflation_promotion ==
          SS_NONLINEAR_DEFLATION_PROMOTE_SYSTEM_SOLUTIONS_AND_ACCEPTED_STATE) {
    status =
        ss_deflation_add_candidate(problem->deflation, accepted_state, accepted_state,
                                   "nls:accepted-state", &admission);
    if (status != SS_OK) {
      return status;
    }
    if (projector == SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL &&
        (problem->deflation_promotion ==
             SS_NONLINEAR_DEFLATION_PROMOTE_ACCEPTED_STATE ||
         compression == SS_DEFLATION_COMPRESSION_SNAPSHOT_POD_TOTAL ||
         compression == SS_DEFLATION_COMPRESSION_SYSTEM_SOLUTION_WINDOW)) {
      /* The legacy continuation restores its pre-attempt basis and appends
       * the accepted state verbatim.  Its next tangent setup performs the
       * explicit A-orthogonalization; refreshing here would mutate the raw
       * accepted-state history against an obsolete final-attempt tangent. */
      return SS_OK;
    }
  }
  if (operator_generation != 0U) {
    return ss_deflation_refresh(problem->deflation, operator_handle,
                                operator_generation);
  }
  return SS_OK;
}

SsNonlinearOutcome ss_nonlinear_status_outcome(SsStatus status) {
  if (status == SS_ERROR_CANCELLED) {
    return SS_NONLINEAR_CANCELLED;
  }
  if (status == SS_ERROR_RESOURCE || status == SS_ERROR_LIMIT) {
    return SS_NONLINEAR_FAILED_LINEAR;
  }
  return SS_NONLINEAR_FAILED_NONFINITE;
}

SsStatus ss_nonlinear_parent_equal(Vec parent, Vec snapshot, int *equal) {
  PetscBool petsc_equal = PETSC_FALSE;
  if (VecEqual(parent, snapshot, &petsc_equal) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  *equal = petsc_equal == PETSC_TRUE;
  return SS_OK;
}

void ss_nonlinear_observe_linear(const SsNonlinearProblem *problem,
                                 SsLinearSolver *solver, PetscInt ordinal,
                                 SsNonlinearSystem system) {
  const SsNonlinearObserver *observer = &problem->observer;
  if (observer->linear_started != NULL) {
    observer->linear_started(observer->linear.context, ordinal, system);
  }
  ss_linear_set_observer(solver, &observer->linear);
  ss_linear_set_cancellation(solver, problem->operations.cancelled, problem->context);
}

void ss_nonlinear_observe_linear_result(const SsNonlinearProblem *problem,
                                        SsStatus status,
                                        const SsLinearSolveRecord *record) {
  const SsNonlinearObserver *observer = &problem->observer;
  if (observer->linear_completed != NULL) {
    observer->linear_completed(observer->linear.context, status, record);
  }
}

void ss_nonlinear_observe_iteration(const SsNonlinearProblem *problem,
                                    const SsNonlinearIterationRecord *record) {
  if (problem->observer.iteration_completed != NULL) {
    problem->observer.iteration_completed(problem->observer.linear.context, record);
  }
}

SsStatus ss_nonlinear_elastic_predictor(const SsNonlinearProblem *problem,
                                        PetscReal scale, Vec predictor,
                                        PetscReal *unscaled_reference_work,
                                        SsLinearSetupRecord *setup_record,
                                        SsLinearSolveRecord *solve_record) {
  SsLinearSolver *linear = NULL;
  SsLinearProblem linear_problem;
  PetscScalar work_scalar = 0.0;
  PetscInt load_size = 0;
  PetscInt predictor_size = 0;
  PetscInt matrix_rows = 0;
  PetscInt matrix_columns = 0;
  SsStatus status = SS_OK;
  if (problem == NULL || problem->communicator == MPI_COMM_NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (predictor == NULL || unscaled_reference_work == NULL || setup_record == NULL ||
      solve_record == NULL || problem->communicator == MPI_COMM_NULL ||
      problem->reference_load == NULL || problem->elastic_reference == NULL ||
      problem->linear_properties.finite_action == 0 ||
      problem->linear_policies.count == 0U || problem->structure_generation == 0U ||
      problem->first_value_generation == 0U || !ss_is_finite(scale) || scale <= 0.0 ||
      !ss_nonlinear_same_communicator(problem->communicator,
                                      (PetscObject)problem->reference_load) ||
      !ss_nonlinear_same_communicator(problem->communicator,
                                      (PetscObject)problem->elastic_reference) ||
      !ss_nonlinear_same_communicator(problem->communicator, (PetscObject)predictor) ||
      VecGetSize(problem->reference_load, &load_size) != PETSC_SUCCESS ||
      VecGetSize(predictor, &predictor_size) != PETSC_SUCCESS ||
      MatGetSize(problem->elastic_reference, &matrix_rows, &matrix_columns) !=
          PETSC_SUCCESS ||
      load_size <= 0 || load_size != predictor_size || load_size != matrix_rows ||
      matrix_rows != matrix_columns) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_status_allreduce_max(problem->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  memset(setup_record, 0, sizeof(*setup_record));
  memset(solve_record, 0, sizeof(*solve_record));
  *unscaled_reference_work = NAN;
  memset(&linear_problem, 0, sizeof(linear_problem));
  linear_problem.operator_handle = problem->elastic_reference;
  linear_problem.preconditioner_operator_handle =
      problem->pmg == NULL ? NULL : problem->elastic_reference;
  linear_problem.properties = problem->linear_properties;
  linear_problem.policy_chain = problem->linear_policies;
  linear_problem.pmg = problem->pmg;
  linear_problem.krylov_recycle = problem->krylov_recycle;
  linear_problem.structure_generation = problem->structure_generation;
  linear_problem.value_generation = problem->first_value_generation;
  if (ss_nonlinear_petsc_status(problem->communicator, VecSet(predictor, 0.0)) !=
      SS_OK) {
    return SS_ERROR_INTERNAL;
  }
#ifndef NDEBUG
  ss_debug_stage_matrix("elastic-matrix", 0U, problem->elastic_reference);
  ss_debug_stage_vector("elastic-rhs", 0U, problem->reference_load);
#endif
  status = ss_linear_create(problem->communicator, &linear);
  if (status == SS_OK) {
    status = ss_linear_setup(linear, &linear_problem, setup_record);
  }
  if (status == SS_OK) {
    ss_nonlinear_observe_linear(problem, linear, 0, SS_NONLINEAR_SYSTEM_ELASTIC);
    status = ss_linear_solve(linear, problem->reference_load, predictor, solve_record);
    ss_nonlinear_observe_linear_result(problem, status, solve_record);
#ifndef NDEBUG
    ss_debug_stage_vector("elastic-solution", 0U, predictor);
#endif
  }
  if (status == SS_OK && solve_record->status != SS_LINEAR_SOLVE_CONVERGED) {
    status = SS_ERROR_NUMERICAL;
  }
  if (status == SS_OK &&
      VecDot(problem->reference_load, predictor, &work_scalar) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    *unscaled_reference_work = PetscRealPart(work_scalar);
    if (!ss_is_finite(*unscaled_reference_work) ||
        *unscaled_reference_work <= DBL_MIN) {
      status = SS_ERROR_INTEGRITY;
    }
  }
  if (status == SS_OK && VecScale(predictor, scale) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (ss_linear_destroy(&linear) != SS_OK && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (status != SS_OK) {
    (void)VecSet(predictor, 0.0);
  }
  return ss_status_allreduce_max(problem->communicator, status);
}
