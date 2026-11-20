/* Single-process Newton solver tests; nonlinear_analysis_continuation_cases.c
 * holds the limit-load and SSR continuation cases.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "nonlinear_analysis_internal.h"

int create_vector(PetscInt size, PetscReal value, Vec *vector) {
  CHECK(VecCreateSeq(PETSC_COMM_SELF, size, vector) == PETSC_SUCCESS);
  CHECK(VecSet(*vector, value) == PETSC_SUCCESS);
  return 0;
}

static int create_diagonal(PetscInt size, PetscReal value, Mat *matrix) {
  PetscInt index;
  CHECK(MatCreateSeqAIJ(PETSC_COMM_SELF, size, size, 1, NULL, matrix) == PETSC_SUCCESS);
  for (index = 0; index < size; ++index) {
    CHECK(MatSetValue(*matrix, index, index, value, INSERT_VALUES) == PETSC_SUCCESS);
  }
  CHECK(MatAssemblyBegin(*matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(MatAssemblyEnd(*matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  return 0;
}

static PetscReal scalar_equilibrium(PetscReal factor, PetscReal value) {
  return (3.0 - factor) * value + value * value * value - 1.0;
}

static PetscReal scalar_total_potential(PetscReal factor, PetscReal value) {
  return 0.5 * (3.0 - factor) * value * value + 0.25 * value * value * value * value -
         value;
}

SsStatus scalar_evaluate(void *opaque, Vec state, PetscReal factor, Vec residual,
                         Mat tangent, PetscReal *functional) {
  ScalarModel *model = (ScalarModel *)opaque;
  const PetscScalar *state_values = NULL;
  PetscScalar *residual_values = NULL;
  PetscInt local_size = 0;
  PetscInt start = 0;
  PetscInt index;
  PetscReal local_functional = 0.0;
  PetscReal global_functional = 0.0;
  ++model->evaluations;
  if (tangent == NULL) {
    ++model->null_tangent_evaluations;
  } else if (model->require_null_tangent != 0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (VecGetLocalSize(state, &local_size) != PETSC_SUCCESS ||
      VecGetOwnershipRange(state, &start, NULL) != PETSC_SUCCESS ||
      VecGetArrayRead(state, &state_values) != PETSC_SUCCESS ||
      VecGetArray(residual, &residual_values) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (tangent != NULL && MatZeroEntries(tangent) != PETSC_SUCCESS) {
    (void)VecRestoreArray(residual, &residual_values);
    (void)VecRestoreArrayRead(state, &state_values);
    return SS_ERROR_INTERNAL;
  }
  for (index = 0; index < local_size; ++index) {
    const PetscReal value = PetscRealPart(state_values[index]);
    residual_values[index] = scalar_equilibrium(factor, value);
    local_functional += scalar_total_potential(factor, value);
    if (tangent != NULL && MatSetValue(tangent, start + index, start + index,
                                       (3.0 - factor) + 3.0 * value * value,
                                       INSERT_VALUES) != PETSC_SUCCESS) {
      (void)VecRestoreArray(residual, &residual_values);
      (void)VecRestoreArrayRead(state, &state_values);
      return SS_ERROR_INTERNAL;
    }
  }
  if (VecRestoreArray(residual, &residual_values) != PETSC_SUCCESS ||
      VecRestoreArrayRead(state, &state_values) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (tangent != NULL &&
      (MatAssemblyBegin(tangent, MAT_FINAL_ASSEMBLY) != PETSC_SUCCESS ||
       MatAssemblyEnd(tangent, MAT_FINAL_ASSEMBLY) != PETSC_SUCCESS)) {
    return SS_ERROR_INTERNAL;
  }
  if (functional != NULL) {
    if (MPI_Allreduce(&local_functional, &global_functional, 1, MPIU_REAL, MPI_SUM,
                      PetscObjectComm((PetscObject)state)) != MPI_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    *functional = global_functional;
  }
  return SS_OK;
}

static PetscErrorCode scalar_shell_mult(Mat shell, Vec input, Vec output) {
  ScalarModel *model = NULL;
  PetscCall(MatShellGetContext(shell, &model));
  PetscCheck(model != NULL && isfinite((double)model->shell_diagonal), PETSC_COMM_SELF,
             PETSC_ERR_ARG_WRONGSTATE,
             "matrix-free scalar tangent has not been linearized");
  PetscCall(VecCopy(input, output));
  PetscCall(VecScale(output, model->shell_diagonal));
  return PETSC_SUCCESS;
}

static PetscErrorCode scalar_shell_diagonal(Mat shell, Vec diagonal) {
  ScalarModel *model = NULL;
  PetscCall(MatShellGetContext(shell, &model));
  PetscCheck(model != NULL && isfinite((double)model->shell_diagonal), PETSC_COMM_SELF,
             PETSC_ERR_ARG_WRONGSTATE,
             "matrix-free scalar tangent has not been linearized");
  PetscCall(VecSet(diagonal, model->shell_diagonal));
  return PETSC_SUCCESS;
}

static SsStatus scalar_linearize(void *opaque, Vec state, PetscReal factor,
                                 PetscReal elastic_regularization) {
  ScalarModel *model = (ScalarModel *)opaque;
  const PetscScalar *values = NULL;
  PetscInt size = 0;
  PetscReal value;
  PetscReal tangent;
  if (model == NULL || model->shell == NULL || !isfinite((double)factor) ||
      !isfinite((double)elastic_regularization) || elastic_regularization < 0.0 ||
      elastic_regularization > 1.0 || VecGetLocalSize(state, &size) != PETSC_SUCCESS ||
      size != 1 || VecGetArrayRead(state, &values) != PETSC_SUCCESS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  value = PetscRealPart(values[0]);
  if (VecRestoreArrayRead(state, &values) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  tangent = (3.0 - factor) + 3.0 * value * value;
  model->shell_diagonal =
      (1.0 - elastic_regularization) * tangent + elastic_regularization * 2.0;
  ++model->linearizations;
  if (!isfinite((double)model->shell_diagonal) || model->shell_diagonal <= 0.0 ||
      MatAssemblyBegin(model->shell, MAT_FINAL_ASSEMBLY) != PETSC_SUCCESS ||
      MatAssemblyEnd(model->shell, MAT_FINAL_ASSEMBLY) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  return SS_OK;
}

static SsStatus scalar_controlled_evaluate(void *opaque, SsNonlinearControlMode mode,
                                           Vec state, PetscReal control, Vec residual,
                                           Mat tangent, PetscReal *functional) {
  ScalarModel *model = (ScalarModel *)opaque;
  const PetscScalar *state_values = NULL;
  PetscScalar *residual_values = NULL;
  PetscInt local_size = 0;
  PetscReal x;
  PetscReal y;
  if (mode == SS_NONLINEAR_CONTROL_SSR_INDIRECT) {
    return scalar_evaluate(opaque, state, control, residual, tangent, functional);
  }
  if (mode != SS_NONLINEAR_CONTROL_LIMIT_LOAD ||
      VecGetLocalSize(state, &local_size) != PETSC_SUCCESS || local_size != 2 ||
      VecGetArrayRead(state, &state_values) != PETSC_SUCCESS ||
      VecGetArray(residual, &residual_values) != PETSC_SUCCESS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  ++model->evaluations;
  x = PetscRealPart(state_values[0]);
  y = PetscRealPart(state_values[1]);
  residual_values[0] = 2.0 + x - y - control;
  residual_values[1] = y * y * y - x;
  if (VecRestoreArray(residual, &residual_values) != PETSC_SUCCESS ||
      VecRestoreArrayRead(state, &state_values) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (tangent != NULL) {
    const PetscInt rows[2] = {0, 1};
    const PetscScalar values[4] = {1.0, -1.0, -1.0, 3.0 * y * y};
    if (MatZeroEntries(tangent) != PETSC_SUCCESS ||
        MatSetValues(tangent, 2, rows, 2, rows, values, INSERT_VALUES) !=
            PETSC_SUCCESS ||
        MatAssemblyBegin(tangent, MAT_FINAL_ASSEMBLY) != PETSC_SUCCESS ||
        MatAssemblyEnd(tangent, MAT_FINAL_ASSEMBLY) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
  }
  if (functional != NULL) {
    *functional = 2.0 * x + 0.5 * x * x - x * y + 0.25 * y * y * y * y - control * x;
  }
  return SS_OK;
}

static SsStatus scalar_admissibility(void *opaque, Vec state, PetscReal factor,
                                     int *admissible) {
  const ScalarModel *model = (const ScalarModel *)opaque;
  PetscReal norm = 0.0;
  if (admissible == NULL || !isfinite((double)factor) ||
      VecNorm(state, NORM_INFINITY, &norm) != PETSC_SUCCESS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *admissible = isfinite((double)norm) && factor > 0.0 &&
                (model->reject_nonzero_state == 0 || norm == 0.0);
  return SS_OK;
}

static int scalar_cancelled(void *opaque) {
  const ScalarModel *model = (const ScalarModel *)opaque;
  return model->cancel_after_evaluations > 0 &&
         model->evaluations >= model->cancel_after_evaluations;
}

static void fill_nonlinear_policy(SsNonlinearPolicy *policy) {
  memset(policy, 0, sizeof(*policy));
  policy->tolerance = 1.0e-12;
  policy->stopping_criterion = SS_NONLINEAR_STOP_RELATIVE_RESIDUAL;
  policy->stopping_tolerance = policy->tolerance;
  policy->maximum_iterations = 24;
  policy->maximum_line_search_trials = 20;
  policy->minimum_alpha = 1.0 / 1048576.0;
  policy->regularization_minimum = 1.0e-8;
  policy->singular_relative_tolerance = 1.0e-12;
  policy->relaxed_factor = 10.0;
  policy->line_method = SS_NONLINEAR_LINE_DIRECTIONAL;
}

static void fill_direct_policy(SsLinearPolicyChain *chain) {
  SsLinearPolicy *policy;
  memset(chain, 0, sizeof(*chain));
  chain->count = 1U;
  policy = &chain->policies[0];
  (void)snprintf(policy->identity, sizeof(policy->identity), "%s",
                 "nonlinear-direct-validation");
  (void)snprintf(policy->options_prefix, sizeof(policy->options_prefix), "%s",
                 "nonlinear_direct_");
  policy->operator_class = SS_LINEAR_OPERATOR_DIRECT_VALIDATION;
  policy->method = SS_LINEAR_METHOD_PREONLY;
  policy->pc = SS_LINEAR_PC_LU;
  policy->side = SS_LINEAR_SIDE_LEFT;
  policy->rtol = 1.0e-13;
  policy->atol = 1.0e-14;
  policy->dtol = 1.0e8;
  policy->maximum_iterations = 1;
  policy->restart = 2;
  policy->vector_memory_cap_bytes = 8U * 1024U * 1024U;
}

int create_scalar_nonlinear(AnalysisModel *model, PetscInt size, Vec *load,
                            Mat *elastic) {
  memset(model, 0, sizeof(*model));
  CHECK(create_vector(size, 1.0, load) == 0);
  CHECK(create_diagonal(size, 2.0, elastic) == 0);
  fill_nonlinear_policy(&model->nonlinear_policy);
  model->nonlinear.communicator = PETSC_COMM_SELF;
  model->nonlinear.reference_load = *load;
  model->nonlinear.elastic_reference = *elastic;
  model->nonlinear.operations.evaluate = scalar_evaluate;
  model->nonlinear.operations.evaluate_controlled = scalar_controlled_evaluate;
  model->nonlinear.operations.admissibility = scalar_admissibility;
  model->nonlinear.operations.cancelled = scalar_cancelled;
  model->nonlinear.context = &model->model;
  model->nonlinear.linear_properties.finite_action = 1;
  model->nonlinear.linear_properties.measured_symmetric = 1;
  model->nonlinear.linear_properties.positive_energy = 1;
  model->nonlinear.linear_properties.pc_symmetric = 1;
  model->nonlinear.linear_properties.pc_positive = 1;
  fill_direct_policy(&model->nonlinear.linear_policies);
  model->nonlinear.structure_generation = 11U;
  model->nonlinear.first_value_generation = 100U;
  model->nonlinear.transaction_generation = 17U;
  model->nonlinear.accepted_generation = 23U;
  return 0;
}

int check_scalar(Vec vector, PetscReal expected, PetscReal tolerance) {
  const PetscScalar *values = NULL;
  PetscInt size = 0;
  PetscInt index;
  CHECK(VecGetLocalSize(vector, &size) == PETSC_SUCCESS);
  CHECK(VecGetArrayRead(vector, &values) == PETSC_SUCCESS);
  for (index = 0; index < size; ++index) {
    CHECK(fabs(PetscRealPart(values[index]) - expected) <= tolerance);
  }
  CHECK(VecRestoreArrayRead(vector, &values) == PETSC_SUCCESS);
  return 0;
}

static int test_adaptive_progress_guard(void) {
  SsNonlinearAdaptiveProgressGuard guard;
  SsNonlinearAdaptiveProgressGuard snapshot;
  SsNonlinearSolveRecord solve;
  PetscReal threshold = NAN;
  PetscReal ratio_threshold = NAN;
  PetscReal checkpoint = NAN;
  int enabled = -1;
  int observed = -1;
  CHECK(ss_nonlinear_adaptive_progress_guard_initialize(-1, &guard) ==
        SS_ERROR_INVALID_ARGUMENT);
  CHECK(ss_nonlinear_adaptive_progress_guard_initialize(4, &guard) == SS_OK);
  CHECK(guard.checkpoint_iteration == 4 && guard.successful_sample_count == 0U &&
        guard.maximum_successful_convergence == 0.0);
  CHECK(ss_nonlinear_adaptive_progress_guard_threshold(&guard, 4, &enabled, &threshold,
                                                       &ratio_threshold) ==
        SS_ERROR_INVALID_ARGUMENT);
  CHECK(ss_nonlinear_adaptive_progress_guard_threshold(&guard, 60, &enabled, &threshold,
                                                       &ratio_threshold) == SS_OK);
  CHECK(enabled == 0 && threshold == 0.0 && ratio_threshold == 0.0);

  memset(&solve, 0, sizeof(solve));
  solve.outcome = SS_NONLINEAR_CONVERGED;
  solve.initial_convergence = 0.30;
  solve.history_count = 5U;
  solve.history[4].convergence_before = 0.03;
  CHECK(ss_nonlinear_adaptive_progress_guard_observe(&guard, &solve, &observed,
                                                     &checkpoint) == SS_OK);
  CHECK(observed != 0 && fabs(checkpoint - 0.03) < 1.0e-15 &&
        guard.successful_sample_count == 1U &&
        fabs(guard.maximum_successful_convergence - 0.03) < 1.0e-15);
  CHECK(ss_nonlinear_adaptive_progress_guard_threshold(&guard, 60, &enabled, &threshold,
                                                       &ratio_threshold) == SS_OK);
  CHECK(enabled == 0 && threshold == 0.0 && ratio_threshold == 0.0);

  memset(&solve, 0, sizeof(solve));
  solve.outcome = SS_NONLINEAR_ACCEPTED_RELAXED;
  solve.initial_convergence = 0.20;
  solve.history_count = 4U;
  solve.final_convergence = 0.10;
  CHECK(ss_nonlinear_adaptive_progress_guard_observe(&guard, &solve, &observed,
                                                     &checkpoint) == SS_OK);
  CHECK(observed != 0 && fabs(checkpoint - 0.10) < 1.0e-15 &&
        guard.successful_sample_count == 2U &&
        fabs(guard.maximum_successful_convergence - 0.10) < 1.0e-15);
  CHECK(ss_nonlinear_adaptive_progress_guard_threshold(&guard, 60, &enabled, &threshold,
                                                       &ratio_threshold) == SS_OK);
  CHECK(enabled != 0 && fabs(threshold - sqrt(0.02)) < 1.0e-15 &&
        fabs(ratio_threshold - sqrt(0.5)) < 1.0e-14);

  snapshot = guard;
  memset(&solve, 0, sizeof(solve));
  solve.outcome = SS_NONLINEAR_FAILED_ITERATION_LIMIT;
  solve.history_count = 5U;
  solve.history[4].convergence_before = 8.0;
  CHECK(ss_nonlinear_adaptive_progress_guard_observe(&guard, &solve, &observed,
                                                     &checkpoint) == SS_OK);
  CHECK(observed == 0 && !isfinite((double)checkpoint) &&
        guard.successful_sample_count == snapshot.successful_sample_count &&
        guard.maximum_successful_convergence ==
            snapshot.maximum_successful_convergence);

  solve.outcome = SS_NONLINEAR_CONVERGED;
  solve.initial_convergence = 0.50;
  solve.history[4].convergence_before = 0.25;
  CHECK(ss_nonlinear_adaptive_progress_guard_observe(&guard, &solve, &observed,
                                                     &checkpoint) == SS_OK);
  CHECK(guard.successful_sample_count == 3U &&
        fabs(guard.maximum_successful_convergence - 0.25) < 1.0e-15);
  guard = snapshot; /* The continuation rollback contract restores the envelope. */
  CHECK(ss_nonlinear_adaptive_progress_guard_threshold(&guard, 60, &enabled, &threshold,
                                                       &ratio_threshold) == SS_OK);
  CHECK(enabled != 0 && fabs(threshold - sqrt(0.02)) < 1.0e-15 &&
        fabs(ratio_threshold - sqrt(0.5)) < 1.0e-14);

  CHECK(ss_nonlinear_adaptive_progress_guard_restore(4, 0U, 0.1, 0.0, 0.0, &guard) ==
        SS_ERROR_INVALID_ARGUMENT);
  CHECK(ss_nonlinear_adaptive_progress_guard_restore(4, 2U, 0.1, 0.1, 0.5, &guard) ==
        SS_OK);
  CHECK(ss_nonlinear_adaptive_progress_guard_threshold(&guard, 60, &enabled, &threshold,
                                                       &ratio_threshold) == SS_OK);
  CHECK(enabled != 0 && fabs(threshold - sqrt(0.02)) < 1.0e-15 &&
        fabs(ratio_threshold - sqrt(0.5)) < 1.0e-14);
  CHECK(ss_nonlinear_adaptive_progress_guard_restore(4, 2U, 0.1, 0.1, 1.1, &guard) ==
        SS_OK);
  CHECK(ss_nonlinear_adaptive_progress_guard_threshold(&guard, 60, &enabled, &threshold,
                                                       &ratio_threshold) == SS_OK);
  CHECK(enabled == 0 && threshold == 0.0 && ratio_threshold == 0.0);
  return 0;
}

static int test_fixed(void) {
  AnalysisModel model;
  SsNonlinearPolicy correction_policy;
  Vec load = NULL;
  Vec parent = NULL;
  Vec accepted = NULL;
  Mat elastic = NULL;
  SsNonlinearSolveRecord record;
  CHECK(create_scalar_nonlinear(&model, 3, &load, &elastic) == 0);
  model.nonlinear_policy.maximum_iterations = 200;
  CHECK(ss_nonlinear_policy_validate(&model.nonlinear_policy) == SS_OK);
  model.nonlinear_policy.maximum_iterations =
      (PetscInt)SS_NONLINEAR_HISTORY_CAPACITY + 1;
  CHECK(ss_nonlinear_policy_validate(&model.nonlinear_policy) ==
        SS_ERROR_INVALID_ARGUMENT);
  model.nonlinear_policy.maximum_iterations = 24;
  model.nonlinear_policy.indirect_progress_guard_iteration = 24;
  model.nonlinear_policy.indirect_progress_guard_maximum_convergence = 0.1;
  CHECK(ss_nonlinear_policy_validate(&model.nonlinear_policy) ==
        SS_ERROR_INVALID_ARGUMENT);
  model.nonlinear_policy.indirect_progress_guard_iteration = 1;
  model.nonlinear_policy.indirect_progress_guard_maximum_convergence = 0.0;
  CHECK(ss_nonlinear_policy_validate(&model.nonlinear_policy) ==
        SS_ERROR_INVALID_ARGUMENT);
  model.nonlinear_policy.indirect_progress_guard_iteration = 0;
  model.nonlinear_policy.indirect_progress_guard_maximum_convergence = 0.1;
  CHECK(ss_nonlinear_policy_validate(&model.nonlinear_policy) == SS_OK);
  model.nonlinear_policy.indirect_progress_guard_iteration = 0;
  model.nonlinear_policy.indirect_progress_guard_maximum_convergence = 0.0;
  CHECK(create_vector(3, 0.0, &parent) == 0);
  CHECK(VecDuplicate(parent, &accepted) == PETSC_SUCCESS);
  memset(&record, 0, sizeof(record));
  CHECK(ss_nonlinear_fixed_solve(&model.nonlinear, &model.nonlinear_policy, 1.0, parent,
                                 accepted, &record) == SS_OK);
  CHECK(record.outcome == SS_NONLINEAR_CONVERGED && record.iterations > 0);
  CHECK(record.history_count == (size_t)record.iterations);
  CHECK(record.initial_convergence > model.nonlinear_policy.tolerance);
  CHECK(record.final_convergence <= model.nonlinear_policy.tolerance);
  CHECK(record.transaction_committed != 0 && record.transaction_rolled_back == 0);
  CHECK(check_scalar(parent, 0.0, 0.0) == 0);
  CHECK(check_scalar(accepted, 0.45339765151640377, 2.0e-12) == 0);
  correction_policy = model.nonlinear_policy;
  correction_policy.tolerance = 1.0e-30;
  correction_policy.stopping_criterion = SS_NONLINEAR_STOP_RELATIVE_CORRECTION;
  correction_policy.stopping_tolerance = 1.0e-3;
  CHECK(VecSet(parent, 0.0) == PETSC_SUCCESS);
  CHECK(VecSet(accepted, 91.0) == PETSC_SUCCESS);
  CHECK(ss_nonlinear_fixed_solve(&model.nonlinear, &correction_policy, 1.0, parent,
                                 accepted, &record) == SS_OK);
  CHECK(record.outcome == SS_NONLINEAR_CONVERGED && record.iterations > 0);
  CHECK(isfinite((double)record.final_relative_correction));
  CHECK(record.final_relative_correction <= correction_policy.stopping_tolerance);
  CHECK(record.final_convergence > correction_policy.tolerance);
  CHECK(record.history[record.history_count - 1U].alpha >= 0.1);
  CHECK(record.history[record.history_count - 1U].relative_correction ==
        record.final_relative_correction);
  CHECK(check_scalar(parent, 0.0, 0.0) == 0);
  CHECK(check_scalar(accepted, 0.45339765151640377, 1.0e-3) == 0);
  CHECK(VecDestroy(&accepted) == PETSC_SUCCESS);
  CHECK(VecDestroy(&parent) == PETSC_SUCCESS);
  CHECK(VecDestroy(&load) == PETSC_SUCCESS);
  CHECK(MatDestroy(&elastic) == PETSC_SUCCESS);
  return 0;
}

static int test_matrix_free_fixed(void) {
  AnalysisModel model;
  SsNonlinearSolveRecord record = {0};
  SsLinearPolicy *linear_policy;
  Vec load = NULL;
  Vec parent = NULL;
  Vec accepted = NULL;
  Mat assembled_elastic = NULL;
  Mat shell = NULL;
  CHECK(create_scalar_nonlinear(&model, 1, &load, &assembled_elastic) == 0);
  CHECK(MatCreateShell(PETSC_COMM_SELF, 1, 1, 1, 1, &model.model, &shell) ==
        PETSC_SUCCESS);
  CHECK(MatShellSetOperation(shell, MATOP_MULT, (void (*)(void))scalar_shell_mult) ==
        PETSC_SUCCESS);
  CHECK(MatShellSetOperation(shell, MATOP_MULT_TRANSPOSE,
                             (void (*)(void))scalar_shell_mult) == PETSC_SUCCESS);
  CHECK(MatShellSetOperation(shell, MATOP_GET_DIAGONAL,
                             (void (*)(void))scalar_shell_diagonal) == PETSC_SUCCESS);
  model.model.shell = shell;
  model.model.shell_diagonal = 2.0;
  model.model.require_null_tangent = 1;
  model.nonlinear.elastic_reference = shell;
  model.nonlinear.operations.linearize = scalar_linearize;
  linear_policy = &model.nonlinear.linear_policies.policies[0];
  (void)snprintf(linear_policy->identity, sizeof(linear_policy->identity), "%s",
                 "nonlinear-matrix-free-fgmres");
  (void)snprintf(linear_policy->options_prefix, sizeof(linear_policy->options_prefix),
                 "%s", "nonlinear_mf_");
  linear_policy->operator_class = SS_LINEAR_OPERATOR_GENERAL;
  linear_policy->method = SS_LINEAR_METHOD_FGMRES;
  linear_policy->pc = SS_LINEAR_PC_NONE;
  linear_policy->side = SS_LINEAR_SIDE_RIGHT;
  linear_policy->rtol = 1.0e-13;
  linear_policy->atol = 1.0e-14;
  linear_policy->dtol = 1.0e8;
  linear_policy->maximum_iterations = 8;
  linear_policy->restart = 8;
  CHECK(create_vector(1, 0.0, &parent) == 0);
  CHECK(VecDuplicate(parent, &accepted) == PETSC_SUCCESS);
  CHECK(ss_nonlinear_fixed_solve(&model.nonlinear, &model.nonlinear_policy, 1.0, parent,
                                 accepted, &record) == SS_OK);
  CHECK(record.outcome == SS_NONLINEAR_CONVERGED && record.iterations > 0);
  CHECK(model.model.linearizations == record.iterations);
  CHECK(model.model.null_tangent_evaluations == model.model.evaluations);
  CHECK(check_scalar(accepted, 0.45339765151640377, 2.0e-12) == 0);
  CHECK(VecDestroy(&accepted) == PETSC_SUCCESS);
  CHECK(VecDestroy(&parent) == PETSC_SUCCESS);
  CHECK(MatDestroy(&shell) == PETSC_SUCCESS);
  CHECK(MatDestroy(&assembled_elastic) == PETSC_SUCCESS);
  CHECK(VecDestroy(&load) == PETSC_SUCCESS);
  return 0;
}

static SsStatus ll_directional_trial(void *opaque, PetscReal alpha,
                                     PetscReal *directional_value,
                                     PetscReal *residual_norm, int *admissible) {
  const PetscReal y = 0.5 + 0.5 * alpha;
  (void)opaque;
  *directional_value = 0.5 * (y * y * y - 0.5);
  *residual_norm = fabs(*directional_value);
  *admissible = isfinite((double)y);
  return SS_OK;
}

static SsStatus rejected_directional_trial(void *opaque, PetscReal alpha,
                                           PetscReal *directional_value,
                                           PetscReal *residual_norm, int *admissible) {
  const int invalid_origin = *(const int *)opaque;
  *directional_value = alpha == 0.0 ? (invalid_origin != 0 ? 1.0 : -1.0) : 1.0;
  *residual_norm = 1.0;
  *admissible = 1;
  return SS_OK;
}

static int test_bisection(void) {
  SsNonlinearPolicy policy;
  SsNonlinearLineSearchRecord record;
  int invalid_origin = 1;
  fill_nonlinear_policy(&policy);
  policy.maximum_line_search_trials = 12;
  policy.line_method = SS_NONLINEAR_LINE_DIRECTIONAL;
  CHECK(ss_nonlinear_line_search(&policy, 1.0, ll_directional_trial, NULL, &record) ==
        SS_OK);
  CHECK(record.outcome == SS_NONLINEAR_CONVERGED);
  CHECK(record.trial_count == 14U);
  CHECK(record.accepted_alpha == 0.587158203125);
  CHECK(record.trials[0].alpha == 0.0 && record.trials[1].alpha == 1.0);
  CHECK(ss_nonlinear_line_search(&policy, 1.0, rejected_directional_trial,
                                 &invalid_origin, &record) == SS_OK);
  CHECK(record.outcome == SS_NONLINEAR_FAILED_LINE_SEARCH);
  CHECK(record.trial_count == 1U && record.accepted_alpha == 0.0);
  invalid_origin = 0;
  policy.maximum_line_search_trials = 1;
  policy.minimum_alpha = 0.1;
  CHECK(ss_nonlinear_line_search(&policy, 1.0, rejected_directional_trial,
                                 &invalid_origin, &record) == SS_OK);
  CHECK(record.outcome == SS_NONLINEAR_FAILED_LINE_SEARCH);
  CHECK(record.trial_count == 3U && record.accepted_alpha == 0.0);
  CHECK(ss_nonlinear_line_search(&policy, 1.0, NULL, NULL, &record) ==
        SS_ERROR_INVALID_ARGUMENT);
  return 0;
}

static SsStatus nonimproving_trial(void *opaque, PetscReal alpha,
                                   PetscReal *directional_value,
                                   PetscReal *residual_norm, int *admissible) {
  (void)opaque;
  *directional_value = -1.0;
  *residual_norm = 1.0 + alpha;
  *admissible = 1;
  return SS_OK;
}

static int test_monotone(void) {
  SsNonlinearPolicy policy;
  SsNonlinearLineSearchRecord record;
  fill_nonlinear_policy(&policy);
  policy.line_method = SS_NONLINEAR_LINE_RESIDUAL_MONOTONE;
  policy.maximum_line_search_trials = 6;
  CHECK(ss_nonlinear_line_search(&policy, 1.0, nonimproving_trial, NULL, &record) ==
        SS_OK);
  CHECK(record.outcome == SS_NONLINEAR_FAILED_LINE_SEARCH);
  CHECK(record.accepted_alpha == 0.0 && record.trial_count == 6U);
  CHECK(record.trials[record.trial_count - 1U].accepted == 0);
  policy.line_search_legacy_alg5 = 1;
  CHECK(ss_nonlinear_line_search(&policy, 1.0, nonimproving_trial, NULL, &record) ==
        SS_OK);
  CHECK(record.outcome == SS_NONLINEAR_CONVERGED);
  CHECK(record.trial_count == 6U && record.accepted_alpha == 0.03125);
  CHECK(record.trials[record.trial_count - 1U].accepted != 0);
  return 0;
}

static int test_regularization(void) {
  SsNonlinearRegularizationRecord record;
  const PetscReal fourth_root_two = pow(2.0, 0.25);
  CHECK(ss_nonlinear_regularization_update(0.25, 0.01, 0.0, &record) == SS_OK);
  CHECK(record.next_value == 0.5 && record.operator_generation_invalidated != 0);
  CHECK(record.deflation_candidate_eligible == 0);
  CHECK(ss_nonlinear_regularization_update(0.25, 0.01, 0.05, &record) == SS_OK);
  CHECK(fabs(record.next_value - 0.25 * fourth_root_two) < 1.0e-15);
  CHECK(ss_nonlinear_regularization_update(0.25, 0.01, 0.1, &record) == SS_OK);
  CHECK(record.next_value == 0.25 && record.deflation_candidate_eligible != 0);
  CHECK(ss_nonlinear_regularization_update(0.25, 0.01, 0.5, &record) == SS_OK);
  CHECK(record.next_value == 0.25 && record.deflation_candidate_eligible != 0);
  CHECK(ss_nonlinear_regularization_update(0.25, 0.01, 1.0, &record) == SS_OK);
  CHECK(fabs(record.next_value - 0.25 / sqrt(2.0)) < 1.0e-15);
  CHECK(ss_nonlinear_regularization_update(0.75, 0.01, 0.0, &record) == SS_OK);
  CHECK(record.outcome == SS_NONLINEAR_FAILED_REGULARIZATION);
  CHECK(record.next_value > 1.0);
  CHECK(ss_nonlinear_regularization_update(0.9, 0.01, 0.05, &record) == SS_OK);
  CHECK(record.outcome == SS_NONLINEAR_FAILED_REGULARIZATION);
  CHECK(record.next_value > 1.0 && record.deflation_candidate_eligible == 0);
  return 0;
}

static int test_singular(void) {
  Vec gradient = NULL;
  Vec first = NULL;
  Vec residual = NULL;
  Vec combined = NULL;
  SsNonlinearAugmentedRecord record;
  PetscScalar values[2];
  CHECK(create_vector(2, 0.0, &gradient) == 0);
  CHECK(VecDuplicate(gradient, &first) == PETSC_SUCCESS);
  CHECK(VecDuplicate(gradient, &residual) == PETSC_SUCCESS);
  CHECK(VecDuplicate(gradient, &combined) == PETSC_SUCCESS);
  values[0] = 1.0;
  values[1] = 0.0;
  CHECK(VecSetValues(gradient, 2, (const PetscInt[]){0, 1}, values, INSERT_VALUES) ==
        PETSC_SUCCESS);
  CHECK(VecAssemblyBegin(gradient) == PETSC_SUCCESS);
  CHECK(VecAssemblyEnd(gradient) == PETSC_SUCCESS);
  values[0] = 0.0;
  values[1] = 1.0;
  CHECK(VecSetValues(first, 2, (const PetscInt[]){0, 1}, values, INSERT_VALUES) ==
        PETSC_SUCCESS);
  CHECK(VecAssemblyBegin(first) == PETSC_SUCCESS);
  CHECK(VecAssemblyEnd(first) == PETSC_SUCCESS);
  CHECK(VecSet(residual, 1.0) == PETSC_SUCCESS);
  CHECK(VecSet(combined, 91.0) == PETSC_SUCCESS);
  CHECK(ss_nonlinear_augmented_direction(gradient, first, residual, 1.0e-12, combined,
                                         &record) == SS_OK);
  CHECK(record.outcome == SS_NONLINEAR_FAILED_SINGULAR_CONSTRAINT);
  CHECK(!isfinite((double)record.control_increment));
  CHECK(check_scalar(combined, 91.0, 0.0) == 0);
  CHECK(VecSetValue(first, 0, NAN, INSERT_VALUES) == PETSC_SUCCESS);
  CHECK(VecAssemblyBegin(first) == PETSC_SUCCESS);
  CHECK(VecAssemblyEnd(first) == PETSC_SUCCESS);
  CHECK(ss_nonlinear_augmented_direction(gradient, first, residual, 1.0e-12, combined,
                                         &record) == SS_OK);
  CHECK(record.outcome == SS_NONLINEAR_FAILED_SINGULAR_CONSTRAINT);
  CHECK(!isfinite((double)record.control_increment));
  CHECK(check_scalar(combined, 91.0, 0.0) == 0);
  CHECK(VecDestroy(&combined) == PETSC_SUCCESS);
  CHECK(VecDestroy(&residual) == PETSC_SUCCESS);
  CHECK(VecDestroy(&first) == PETSC_SUCCESS);
  CHECK(VecDestroy(&gradient) == PETSC_SUCCESS);
  return 0;
}

static int check_failed_rollback(const SsNonlinearSolveRecord *record,
                                 SsNonlinearOutcome expected, Vec parent, Vec accepted,
                                 SsDeflation *deflation,
                                 const SsDeflationDiagnostics *before) {
  SsDeflationDiagnostics after;
  CHECK(record->outcome == expected);
  CHECK(record->transaction_rolled_back != 0 && record->transaction_committed == 0);
  CHECK(record->parent_unchanged_on_failure != 0);
  CHECK(check_scalar(parent, 0.0, 0.0) == 0);
  CHECK(check_scalar(accepted, 77.0, 0.0) == 0);
  CHECK(ss_deflation_diagnostics(deflation, &after) == SS_OK);
  CHECK(after.persistent_count == before->persistent_count);
  CHECK(after.provisional_count == 0U && after.active_count == before->active_count);
  return 0;
}

static int test_rollback(void) {
  AnalysisModel model;
  SsLinearPolicy *linear_policy;
  Vec load = NULL;
  Vec parent = NULL;
  Vec accepted = NULL;
  Mat elastic = NULL;
  SsDeflation *deflation = NULL;
  SsDeflationPolicy deflation_policy;
  SsDeflationDiagnostics before;
  SsNonlinearSolveRecord record;
  Vec raw_difference = NULL;
  CHECK(create_scalar_nonlinear(&model, 2, &load, &elastic) == 0);
  memset(&deflation_policy, 0, sizeof(deflation_policy));
  deflation_policy.dimension_cap = 4U;
  deflation_policy.norm_tolerance = 1.0e-13;
  deflation_policy.pivot_tolerance = 1.0e-13;
  deflation_policy.condition_limit = 1.0e12;
  deflation_policy.invariant_tolerance = 1.0e-10;
  deflation_policy.reorthogonalization_passes = 2U;
  CHECK(ss_deflation_create(PETSC_COMM_SELF, &deflation_policy, &deflation) == SS_OK);
  model.nonlinear.deflation = deflation;
  model.nonlinear.deflation_mode = SS_NONLINEAR_DEFLATION_OWNED_TRANSACTION;
  linear_policy = &model.nonlinear.linear_policies.policies[0];
  (void)snprintf(linear_policy->identity, sizeof(linear_policy->identity), "%s",
                 "nonlinear-deflated-rollback");
  (void)snprintf(linear_policy->options_prefix, sizeof(linear_policy->options_prefix),
                 "%s", "nonlinear_deflated_rollback_");
  linear_policy->operator_class = SS_LINEAR_OPERATOR_GENERAL;
  linear_policy->method = SS_LINEAR_METHOD_FGMRES;
  linear_policy->pc = SS_LINEAR_PC_JACOBI;
  linear_policy->side = SS_LINEAR_SIDE_RIGHT;
  linear_policy->maximum_iterations = 32;
  linear_policy->restart = 8;
  model.model.cancel_after_evaluations = 2;
  CHECK(create_vector(2, 0.0, &parent) == 0);
  CHECK(VecDuplicate(parent, &accepted) == PETSC_SUCCESS);
  CHECK(VecSet(accepted, 77.0) == PETSC_SUCCESS);
  CHECK(ss_deflation_diagnostics(deflation, &before) == SS_OK);
  CHECK(ss_nonlinear_fixed_solve(&model.nonlinear, &model.nonlinear_policy, 1.0, parent,
                                 accepted, &record) == SS_OK);
  CHECK(check_failed_rollback(&record, SS_NONLINEAR_CANCELLED, parent, accepted,
                              deflation, &before) == 0);

  model.model.evaluations = 0;
  model.model.cancel_after_evaluations = 0;
  model.model.reject_nonzero_state = 1;
  model.nonlinear_policy.maximum_iterations = 32;
  CHECK(ss_nonlinear_fixed_solve(&model.nonlinear, &model.nonlinear_policy, 1.0, parent,
                                 accepted, &record) == SS_OK);
  CHECK(record.history_count > 1U && record.history[0].alpha == 0.0);
  CHECK(record.history[0].tangent_reused == 0 && record.history[1].tangent_reused != 0);
  CHECK(record.final_regularization > 1.0);
  CHECK(check_failed_rollback(&record, SS_NONLINEAR_FAILED_REGULARIZATION, parent,
                              accepted, deflation, &before) == 0);

  model.model.evaluations = 0;
  model.model.reject_nonzero_state = 0;
  linear_policy->vector_memory_cap_bytes = 1U;
  CHECK(ss_nonlinear_fixed_solve(&model.nonlinear, &model.nonlinear_policy, 1.0, parent,
                                 accepted, &record) == SS_OK);
  CHECK(check_failed_rollback(&record, SS_NONLINEAR_FAILED_LINEAR, parent, accepted,
                              deflation, &before) == 0);

  model.model.evaluations = 0;
  linear_policy->vector_memory_cap_bytes = 8U * 1024U * 1024U;
  model.nonlinear_policy.maximum_iterations = 1;
  CHECK(ss_nonlinear_fixed_solve(&model.nonlinear, &model.nonlinear_policy, 1.0, parent,
                                 accepted, &record) == SS_OK);
  CHECK(check_failed_rollback(&record, SS_NONLINEAR_FAILED_ITERATION_LIMIT, parent,
                              accepted, deflation, &before) == 0);
  model.model.evaluations = 0;
  model.nonlinear_policy.maximum_iterations = 24;
  model.nonlinear.deflation_promotion = SS_NONLINEAR_DEFLATION_PROMOTE_ACCEPTED_STATE;
  CHECK(ss_nonlinear_fixed_solve(&model.nonlinear, &model.nonlinear_policy, 1.0, parent,
                                 accepted, &record) == SS_OK);
  CHECK(record.outcome == SS_NONLINEAR_CONVERGED);
  CHECK(record.transaction_committed != 0 && record.transaction_rolled_back == 0);
  CHECK(ss_deflation_diagnostics(deflation, &before) == SS_OK);
  CHECK(before.persistent_count == 1U && before.provisional_count == 0U &&
        before.active_count == before.persistent_count);
  {
    Vec left = NULL;
    Vec right = NULL;
    const char *provenance = NULL;
    CHECK(ss_deflation_persistent_entry(deflation, 0U, &left, &right, &provenance) ==
          SS_OK);
    CHECK(left != NULL && right != NULL && provenance != NULL);
    CHECK(strcmp(provenance, "nls:accepted-state") == 0);
  }

  /* A converged-at-entry legacy transaction must append the accepted state
   * verbatim and defer A-orthogonalization until the next tangent setup. */
  CHECK(ss_deflation_destroy(&deflation) == SS_OK);
  deflation_policy.projector = SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL;
  CHECK(ss_deflation_create(PETSC_COMM_SELF, &deflation_policy, &deflation) == SS_OK);
  model.nonlinear.deflation = deflation;
  CHECK(check_scalar(accepted, 0.45339765151640377, 2.0e-12) == 0);
  CHECK(VecCopy(accepted, parent) == PETSC_SUCCESS);
  CHECK(VecSet(accepted, 77.0) == PETSC_SUCCESS);
  model.model.evaluations = 0;
  CHECK(ss_nonlinear_fixed_solve(&model.nonlinear, &model.nonlinear_policy, 1.0, parent,
                                 accepted, &record) == SS_OK);
  CHECK(record.outcome == SS_NONLINEAR_CONVERGED);
  CHECK(record.transaction_committed != 0 && record.transaction_rolled_back == 0);
  CHECK(ss_deflation_diagnostics(deflation, &before) == SS_OK);
  CHECK(before.persistent_count == 1U);
  CHECK(before.refresh_count == 0U);
  {
    Vec left = NULL;
    Vec right = NULL;
    const char *provenance = NULL;
    PetscReal raw_norm = NAN;
    CHECK(ss_deflation_persistent_entry(deflation, 0U, &left, &right, &provenance) ==
          SS_OK);
    CHECK(VecDuplicate(parent, &raw_difference) == PETSC_SUCCESS);
    CHECK(VecCopy(right, raw_difference) == PETSC_SUCCESS);
    CHECK(VecAXPY(raw_difference, -1.0, parent) == PETSC_SUCCESS);
    CHECK(VecNorm(raw_difference, NORM_2, &raw_norm) == PETSC_SUCCESS);
    CHECK(raw_norm == 0.0);
  }
  CHECK(VecDestroy(&raw_difference) == PETSC_SUCCESS);
  CHECK(VecDestroy(&accepted) == PETSC_SUCCESS);
  CHECK(VecDestroy(&parent) == PETSC_SUCCESS);
  CHECK(ss_deflation_destroy(&deflation) == SS_OK);
  CHECK(VecDestroy(&load) == PETSC_SUCCESS);
  CHECK(MatDestroy(&elastic) == PETSC_SUCCESS);
  return 0;
}

static int test_ll_controlled_solver(void) {
  AnalysisModel model;
  SsNonlinearPolicy delta_policy;
  SsNonlinearSolveRecord record;
  Vec load = NULL;
  Vec parent = NULL;
  Vec accepted = NULL;
  Mat elastic = NULL;
  PetscReal accepted_control = 77.0;
  PetscReal expected_y = cbrt(0.5);
  const PetscInt indices[2] = {0, 1};
  const PetscScalar elastic_values[4] = {2.0, -1.0, -1.0, 2.0};
  memset(&model, 0, sizeof(model));
  CHECK(create_vector(2, 0.0, &load) == 0);
  CHECK(VecSetValues(load, 2, indices, (const PetscScalar[]){1.0, 0.0},
                     INSERT_VALUES) == PETSC_SUCCESS);
  CHECK(VecAssemblyBegin(load) == PETSC_SUCCESS);
  CHECK(VecAssemblyEnd(load) == PETSC_SUCCESS);
  CHECK(MatCreateSeqAIJ(PETSC_COMM_SELF, 2, 2, 2, NULL, &elastic) == PETSC_SUCCESS);
  CHECK(MatSetValues(elastic, 2, indices, 2, indices, elastic_values, INSERT_VALUES) ==
        PETSC_SUCCESS);
  CHECK(MatAssemblyBegin(elastic, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(MatAssemblyEnd(elastic, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  fill_nonlinear_policy(&model.nonlinear_policy);
  model.nonlinear.communicator = PETSC_COMM_SELF;
  model.nonlinear.reference_load = load;
  model.nonlinear.elastic_reference = elastic;
  model.nonlinear.operations.evaluate = scalar_evaluate;
  model.nonlinear.operations.evaluate_controlled = scalar_controlled_evaluate;
  model.nonlinear.operations.admissibility = scalar_admissibility;
  model.nonlinear.context = &model.model;
  model.nonlinear.linear_properties.finite_action = 1;
  model.nonlinear.linear_properties.measured_symmetric = 1;
  model.nonlinear.linear_properties.pc_symmetric = 1;
  fill_direct_policy(&model.nonlinear.linear_policies);
  model.nonlinear.structure_generation = 101U;
  model.nonlinear.first_value_generation = 200U;
  model.nonlinear.transaction_generation = 103U;
  model.nonlinear.accepted_generation = 107U;
  CHECK(create_vector(2, 0.0, &parent) == 0);
  CHECK(VecSetValues(parent, 2, indices, (const PetscScalar[]){0.5, 0.5},
                     INSERT_VALUES) == PETSC_SUCCESS);
  CHECK(VecAssemblyBegin(parent) == PETSC_SUCCESS);
  CHECK(VecAssemblyEnd(parent) == PETSC_SUCCESS);
  CHECK(VecDuplicate(parent, &accepted) == PETSC_SUCCESS);
  CHECK(VecSet(accepted, 77.0) == PETSC_SUCCESS);
  CHECK(ss_nonlinear_work_controlled_solve(
            &model.nonlinear, &model.nonlinear_policy, SS_NONLINEAR_CONTROL_LIMIT_LOAD,
            0.5, 2.0, parent, accepted, &accepted_control, &record) == SS_OK);
  CHECK(record.outcome == SS_NONLINEAR_CONVERGED);
  CHECK(record.history_count >= 2U && record.transaction_committed != 0);
  CHECK(fabs(record.history[0].control_increment + 0.5) < 1.0e-7);
  CHECK(fabs(record.history[0].alpha - 0.5874) < 5.0e-4);
  CHECK(fabs(record.history[0].control_after -
             (2.0 + record.history[0].alpha * record.history[0].control_increment)) <
        1.0e-14);
  CHECK(fabs(record.history[0].control_after -
             (2.0 + record.history[0].control_increment)) > 0.1);
  CHECK(record.history[0].control_linear_solve.solve_count == 1U);
  CHECK(record.history[0].residual_linear_solve.solve_count == 2U);
  CHECK(fabs(accepted_control - (2.5 - expected_y)) < 1.0e-10);
  {
    const PetscScalar *values = NULL;
    CHECK(VecGetArrayRead(accepted, &values) == PETSC_SUCCESS);
    CHECK(fabs(PetscRealPart(values[0]) - 0.5) < 1.0e-13);
    CHECK(fabs(PetscRealPart(values[1]) - expected_y) < 1.0e-10);
    CHECK(VecRestoreArrayRead(accepted, &values) == PETSC_SUCCESS);
  }
  delta_policy = model.nonlinear_policy;
  delta_policy.tolerance = 1.0e-30;
  delta_policy.stopping_criterion = SS_NONLINEAR_STOP_ABSOLUTE_DELTA_LAMBDA;
  delta_policy.stopping_tolerance = 0.49;
  accepted_control = 77.0;
  CHECK(VecSet(accepted, 77.0) == PETSC_SUCCESS);
  CHECK(ss_nonlinear_work_controlled_solve(
            &model.nonlinear, &delta_policy, SS_NONLINEAR_CONTROL_LIMIT_LOAD, 0.5, 2.0,
            parent, accepted, &accepted_control, &record) == SS_OK);
  CHECK(record.outcome == SS_NONLINEAR_CONVERGED && record.iterations > 1);
  delta_policy.stopping_tolerance = 0.51;
  accepted_control = 77.0;
  CHECK(VecSet(accepted, 77.0) == PETSC_SUCCESS);
  CHECK(ss_nonlinear_work_controlled_solve(
            &model.nonlinear, &delta_policy, SS_NONLINEAR_CONTROL_LIMIT_LOAD, 0.5, 2.0,
            parent, accepted, &accepted_control, &record) == SS_OK);
  CHECK(record.outcome == SS_NONLINEAR_CONVERGED && record.iterations == 1);
  CHECK(fabs(record.history[0].control_increment) < delta_policy.stopping_tolerance);
  CHECK(record.final_convergence > delta_policy.tolerance);
  delta_policy = model.nonlinear_policy;
  delta_policy.line_method = SS_NONLINEAR_LINE_RESIDUAL_MONOTONE;
  delta_policy.indirect_adaptive_linear_tolerance = 1;
  delta_policy.indirect_adaptive_linear_iteration_threshold = 2;
  delta_policy.indirect_adaptive_linear_alpha_threshold = 1.0;
  delta_policy.indirect_adaptive_linear_tightening_factor = 0.5;
  delta_policy.indirect_adaptive_linear_minimum_relative_tolerance = 1.0e-16;
  model.model.reject_nonzero_state = 1;
  accepted_control = 77.0;
  CHECK(VecSet(accepted, 77.0) == PETSC_SUCCESS);
  CHECK(ss_nonlinear_work_controlled_solve(
            &model.nonlinear, &delta_policy, SS_NONLINEAR_CONTROL_SSR_INDIRECT, 0.5,
            2.0, parent, accepted, &accepted_control, &record) == SS_OK);
  CHECK(record.outcome == SS_NONLINEAR_FAILED_ITERATION_LIMIT &&
        record.history_count == (size_t)delta_policy.maximum_iterations);
  {
    size_t index;
    int observed_tightening = 0;
    for (index = 0U; index < record.history_count; ++index) {
      const SsNonlinearIterationRecord *step = &record.history[index];
      CHECK(fabs(step->control_linear_relative_tolerance -
                 step->control_linear_solve.effective_rtol) < 1.0e-28);
      CHECK(fabs(step->residual_linear_relative_tolerance -
                 step->residual_linear_solve.effective_rtol) < 1.0e-28);
      if (step->control_linear_tolerance_tightened != 0 ||
          step->residual_linear_tolerance_tightened != 0) {
        observed_tightening = 1;
      }
      if (index + 1U < record.history_count) {
        CHECK(fabs(step->control_linear_next_relative_tolerance -
                   record.history[index + 1U].control_linear_relative_tolerance) <
              1.0e-28);
        CHECK(fabs(step->residual_linear_next_relative_tolerance -
                   record.history[index + 1U].residual_linear_relative_tolerance) <
              1.0e-28);
      }
    }
    CHECK(observed_tightening != 0);
  }
  model.model.reject_nonzero_state = 0;
  delta_policy = model.nonlinear_policy;
  delta_policy.indirect_progress_guard_iteration = 0;
  delta_policy.indirect_progress_guard_maximum_convergence = 1.0e-30;
  delta_policy.line_method = SS_NONLINEAR_LINE_RESIDUAL_MONOTONE;
  accepted_control = 77.0;
  CHECK(VecSet(accepted, 77.0) == PETSC_SUCCESS);
  CHECK(ss_nonlinear_work_controlled_solve(
            &model.nonlinear, &delta_policy, SS_NONLINEAR_CONTROL_SSR_INDIRECT, 0.5,
            2.0, parent, accepted, &accepted_control, &record) == SS_OK);
  CHECK(record.outcome == SS_NONLINEAR_FAILED_ITERATION_LIMIT);
  CHECK(record.iterations == 0 && record.history_count == 0U);
  CHECK(record.transaction_rolled_back != 0 && record.parent_unchanged_on_failure != 0);
  CHECK(accepted_control == 77.0);
  CHECK(check_scalar(accepted, 77.0, 0.0) == 0);
  model.model.reject_nonzero_state = 1;
  model.nonlinear_policy.maximum_iterations = 32;
  accepted_control = 77.0;
  CHECK(VecSet(accepted, 77.0) == PETSC_SUCCESS);
  CHECK(ss_nonlinear_work_controlled_solve(
            &model.nonlinear, &model.nonlinear_policy, SS_NONLINEAR_CONTROL_LIMIT_LOAD,
            0.5, 2.0, parent, accepted, &accepted_control, &record) == SS_OK);
  CHECK(record.outcome == SS_NONLINEAR_FAILED_REGULARIZATION);
  CHECK(record.transaction_rolled_back != 0 && record.parent_unchanged_on_failure != 0);
  CHECK(record.history_count > 1U && record.history[0].tangent_reused == 0 &&
        record.history[1].tangent_reused != 0);
  CHECK(accepted_control == 77.0);
  CHECK(check_scalar(accepted, 77.0, 0.0) == 0);
  CHECK(VecDestroy(&accepted) == PETSC_SUCCESS);
  CHECK(VecDestroy(&parent) == PETSC_SUCCESS);
  CHECK(MatDestroy(&elastic) == PETSC_SUCCESS);
  CHECK(VecDestroy(&load) == PETSC_SUCCESS);
  return 0;
}

static int test_ll_damping(void) {
  Vec parent = NULL;
  Vec direction = NULL;
  Vec load = NULL;
  Vec trial = NULL;
  PetscReal updated = 0.0;
  PetscReal work = 0.0;
  const PetscReal alpha = 0.587158203125;
  const PetscInt indices[2] = {0, 1};
  CHECK(create_vector(2, 0.0, &parent) == 0);
  CHECK(VecDuplicate(parent, &direction) == PETSC_SUCCESS);
  CHECK(VecDuplicate(parent, &load) == PETSC_SUCCESS);
  CHECK(VecDuplicate(parent, &trial) == PETSC_SUCCESS);
  CHECK(VecSetValues(parent, 2, indices, (const PetscScalar[]){0.5, 0.5},
                     INSERT_VALUES) == PETSC_SUCCESS);
  CHECK(VecSetValues(direction, 2, indices, (const PetscScalar[]){0.0, 0.5},
                     INSERT_VALUES) == PETSC_SUCCESS);
  CHECK(VecSetValues(load, 2, indices, (const PetscScalar[]){1.0, 0.0},
                     INSERT_VALUES) == PETSC_SUCCESS);
  CHECK(VecAssemblyBegin(parent) == PETSC_SUCCESS);
  CHECK(VecAssemblyEnd(parent) == PETSC_SUCCESS);
  CHECK(VecAssemblyBegin(direction) == PETSC_SUCCESS);
  CHECK(VecAssemblyEnd(direction) == PETSC_SUCCESS);
  CHECK(VecAssemblyBegin(load) == PETSC_SUCCESS);
  CHECK(VecAssemblyEnd(load) == PETSC_SUCCESS);
  CHECK(ss_nonlinear_apply_augmented_update(parent, direction, load, 0.5, 2.0, -0.5,
                                            alpha, trial, &updated, &work) == SS_OK);
  CHECK(fabs(updated - 1.7064208984375) < 1.0e-14);
  CHECK(fabs(work - 0.5) < 1.0e-14);
  CHECK(fabs((2.0 + 0.5 - 0.7935791015625) - updated) < 1.0e-14);
  CHECK(fabs((2.0 + 0.5 - 0.7935791015625) - 1.5 - 0.2064208984375) < 1.0e-14);
  CHECK(VecDestroy(&trial) == PETSC_SUCCESS);
  CHECK(VecDestroy(&load) == PETSC_SUCCESS);
  CHECK(VecDestroy(&direction) == PETSC_SUCCESS);
  CHECK(VecDestroy(&parent) == PETSC_SUCCESS);
  CHECK(test_ll_controlled_solver() == 0);
  return 0;
}

int main(int argc, char **argv) {
  const char *test_id;
  int result = 1;
  if (PetscInitialize(&argc, &argv, NULL, NULL) != PETSC_SUCCESS) {
    return 1;
  }
  test_id = argc == 2 ? argv[1] : "";
  if (strcmp(test_id, "NLS-UNIT-FIXED-001") == 0) {
    result = test_fixed();
    if (result == 0) {
      result = test_matrix_free_fixed();
    }
  } else if (strcmp(test_id, "NLS-UNIT-ADAPTIVE-GUARD-001") == 0) {
    result = test_adaptive_progress_guard();
  } else if (strcmp(test_id, "NLS-UNIT-BISECT-001") == 0) {
    result = test_bisection();
  } else if (strcmp(test_id, "NLS-REG-MONOTONE-001") == 0) {
    result = test_monotone();
  } else if (strcmp(test_id, "NLS-UNIT-REGULARIZE-001") == 0) {
    result = test_regularization();
  } else if (strcmp(test_id, "NLS-UNIT-SINGULAR-001") == 0) {
    result = test_singular();
  } else if (strcmp(test_id, "NLS-CONTRACT-ROLLBACK-001") == 0) {
    result = test_rollback();
  } else if (strcmp(test_id, "ANA-REG-LL-DAMP-001") == 0) {
    result = test_ll_damping();
  } else if (strcmp(test_id, "ANA-UNIT-LL-001") == 0) {
    result = test_limit_load();
  } else if (strcmp(test_id, "ANA-REG-DIRECT-COORD-001") == 0) {
    result = test_direct_coordinate();
  } else if (strcmp(test_id, "ANA-UNIT-DIRECT-001") == 0) {
    result = test_direct_trajectory();
  } else if (strcmp(test_id, "ANA-UNIT-INDIRECT-001") == 0) {
    result = test_indirect_trajectory();
  } else if (strcmp(test_id, "ANA-UNIT-INDIRECT-ADAPTIVE-001") == 0) {
    result = test_indirect_adaptive_work_controller();
  } else if (strcmp(test_id, "ANA-UNIT-INDIRECT-EFFICIENCY-001") == 0) {
    result = test_indirect_work_efficiency_controller();
  } else if (strcmp(test_id, "ASM-REGION-BODY-FORCE-001") == 0) {
    result = mechanics_fixture_run_region_body_force();
  } else {
    fprintf(stderr, "unknown nonlinear or analysis test: %s\n", test_id);
  }
  if (PetscFinalize() != PETSC_SUCCESS) {
    return 1;
  }
  return result;
}
