/* Linear-solver policies, acceptance, fallback and Krylov recycling;
 * linear_deflation_projection_cases.c holds the deflation projector,
 * compression and rollback cases.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "linear_deflation_internal.h"

static PetscErrorCode shell_mult(Mat matrix, Vec input, Vec output) {
  const PetscScalar *x = NULL;
  PetscScalar *y = NULL;
  Shell2x2 *context = NULL;
  PetscFunctionBegin;
  PetscCall(MatShellGetContext(matrix, &context));
  PetscCall(VecGetArrayRead(input, &x));
  PetscCall(VecGetArray(output, &y));
  y[0] = context->values[0] * x[0] + context->values[1] * x[1];
  y[1] = context->values[2] * x[0] + context->values[3] * x[1];
  PetscCall(VecRestoreArray(output, &y));
  PetscCall(VecRestoreArrayRead(input, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int create_dense(PetscInt size, const double *values, Mat *matrix) {
  PetscInt row;
  CHECK(values != NULL && matrix != NULL);
  CHECK(MatCreateSeqDense(PETSC_COMM_SELF, size, size, NULL, matrix) == PETSC_SUCCESS);
  for (row = 0; row < size; ++row) {
    PetscInt column;
    for (column = 0; column < size; ++column) {
      const PetscScalar value = (PetscScalar)values[row * size + column];
      CHECK(MatSetValue(*matrix, row, column, value, INSERT_VALUES) == PETSC_SUCCESS);
    }
  }
  CHECK(MatAssemblyBegin(*matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(MatAssemblyEnd(*matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  return 0;
}

int create_vector(PetscInt size, const double *values, Vec *vector) {
  PetscInt index;
  CHECK(values != NULL && vector != NULL);
  CHECK(VecCreateSeq(PETSC_COMM_SELF, size, vector) == PETSC_SUCCESS);
  for (index = 0; index < size; ++index) {
    CHECK(VecSetValue(*vector, index, (PetscScalar)values[index], INSERT_VALUES) ==
          PETSC_SUCCESS);
  }
  CHECK(VecAssemblyBegin(*vector) == PETSC_SUCCESS);
  CHECK(VecAssemblyEnd(*vector) == PETSC_SUCCESS);
  return 0;
}

int check_vector(Vec vector, PetscInt size, const double *expected, double tolerance) {
  const PetscScalar *values = NULL;
  PetscInt index;
  CHECK(VecGetArrayRead(vector, &values) == PETSC_SUCCESS);
  for (index = 0; index < size; ++index) {
    CHECK_NEAR(PetscRealPart(values[index]), expected[index], tolerance);
  }
  CHECK(VecRestoreArrayRead(vector, &values) == PETSC_SUCCESS);
  return 0;
}

int check_vector_bits(Vec actual, Vec expected) {
  const PetscScalar *actual_values = NULL;
  const PetscScalar *expected_values = NULL;
  PetscInt actual_size = 0;
  PetscInt expected_size = 0;
  int equal = 0;
  CHECK(VecGetLocalSize(actual, &actual_size) == PETSC_SUCCESS);
  CHECK(VecGetLocalSize(expected, &expected_size) == PETSC_SUCCESS);
  CHECK(actual_size == expected_size);
  CHECK(VecGetArrayRead(actual, &actual_values) == PETSC_SUCCESS);
  CHECK(VecGetArrayRead(expected, &expected_values) == PETSC_SUCCESS);
  equal = memcmp(actual_values, expected_values,
                 (size_t)actual_size * sizeof(*actual_values)) == 0;
  CHECK(VecRestoreArrayRead(expected, &expected_values) == PETSC_SUCCESS);
  CHECK(VecRestoreArrayRead(actual, &actual_values) == PETSC_SUCCESS);
  CHECK(equal != 0);
  return 0;
}

SsLinearOperatorProperties general_properties(void) {
  const SsLinearOperatorProperties properties = {
      1, 0, 0, 0, 0, 1, 0,
  };
  return properties;
}

SsLinearPolicy general_policy(const char *identity, const char *prefix) {
  SsLinearPolicy policy = {0};
  (void)snprintf(policy.identity, sizeof(policy.identity), "%s", identity);
  (void)snprintf(policy.options_prefix, sizeof(policy.options_prefix), "%s", prefix);
  policy.operator_class = SS_LINEAR_OPERATOR_GENERAL;
  policy.method = SS_LINEAR_METHOD_FGMRES;
  policy.pc = SS_LINEAR_PC_NONE;
  policy.side = SS_LINEAR_SIDE_RIGHT;
  policy.rtol = 1.0e-12;
  policy.atol = 1.0e-14;
  policy.dtol = 1.0e6;
  policy.maximum_iterations = 20;
  policy.restart = 4;
  policy.vector_memory_cap_bytes = 1024U * 1024U;
  return policy;
}

SsDeflationPolicy deflation_policy(size_t cap) {
  const SsDeflationPolicy policy = {
      .dimension_cap = cap,
      .norm_tolerance = 1.0e-12,
      .pivot_tolerance = 1.0e-12,
      .condition_limit = 1.0e12,
      .invariant_tolerance = 2.0e-11,
      .reorthogonalization_passes = 2U,
      .projector = SS_DEFLATION_PROJECTOR_GENERAL_LEFT_RIGHT,
      .compression = SS_DEFLATION_COMPRESSION_NONE,
      .compression_dimension = 0U,
      .compression_history_decay = 1.0,
  };
  return policy;
}

static int test_linear_policy(void) {
  SsLinearOperatorProperties properties = general_properties();
  SsLinearPolicy policy = general_policy("general", "linpol_");
  CHECK(ss_linear_policy_validate(&policy, &properties) == SS_OK);
  policy.stopping_condition = (SsLinearStoppingCondition)99;
  CHECK(ss_linear_policy_validate(&policy, &properties) == SS_ERROR_INVALID_ARGUMENT);
  policy.stopping_condition = SS_LINEAR_STOP_PROJECTED_RHS_PETSC;
  CHECK(ss_linear_policy_validate(&policy, &properties) == SS_OK);
  policy.stopping_condition = SS_LINEAR_STOP_PRECONDITIONED_RESIDUAL_ORIGINAL_RHS;
  CHECK(ss_linear_policy_validate(&policy, &properties) == SS_ERROR_INVALID_ARGUMENT);
  policy.stopping_condition = SS_LINEAR_STOP_NATURAL_RESIDUAL_INITIAL;
  CHECK(ss_linear_policy_validate(&policy, &properties) == SS_ERROR_INVALID_ARGUMENT);
  policy.stopping_condition = SS_LINEAR_STOP_ORIGINAL_RHS_TRUE_RESIDUAL;
  policy.initial_guess_policy = (SsLinearInitialGuessPolicy)99;
  CHECK(ss_linear_policy_validate(&policy, &properties) == SS_ERROR_INVALID_ARGUMENT);
  policy.initial_guess_policy = SS_LINEAR_INITIAL_GUESS_COARSE_ONLY;

  policy.method = SS_LINEAR_METHOD_HPDDM_GCRODR;
  policy.recycle_dimension = 2;
#if defined(PETSC_HAVE_HPDDM)
  CHECK(ss_linear_policy_validate(&policy, &properties) == SS_OK);
#else
  CHECK(ss_linear_policy_validate(&policy, &properties) == SS_ERROR_INVALID_ARGUMENT);
#endif
  policy.recycle_dimension = 0;

  policy.method = SS_LINEAR_METHOD_CG;
  CHECK(ss_linear_policy_validate(&policy, &properties) == SS_ERROR_TYPE_MISMATCH);

  policy = general_policy("minres", "linmin_");
  policy.operator_class = SS_LINEAR_OPERATOR_SYMMETRIC_INDEFINITE;
  policy.method = SS_LINEAR_METHOD_MINRES;
  policy.side = SS_LINEAR_SIDE_LEFT;
  properties.measured_symmetric = 1;
  properties.pc_symmetric = 1;
  properties.pc_positive = 1;
  properties.pc_variable = 0;
  CHECK(ss_linear_policy_validate(&policy, &properties) == SS_OK);
  properties.nonlinear_hydraulic = 1;
  CHECK(ss_linear_policy_validate(&policy, &properties) == SS_ERROR_TYPE_MISMATCH);

  properties.nonlinear_hydraulic = 0;
  properties.pc_variable = 1;
  CHECK(ss_linear_policy_validate(&policy, &properties) == SS_ERROR_TYPE_MISMATCH);

  policy = general_policy("cg", "lincg_");
  policy.operator_class = SS_LINEAR_OPERATOR_SPD_VERIFIED;
  policy.method = SS_LINEAR_METHOD_CG;
  policy.side = SS_LINEAR_SIDE_LEFT;
  properties.measured_symmetric = 1;
  properties.positive_energy = 1;
  properties.pc_symmetric = 1;
  properties.pc_positive = 1;
  properties.pc_variable = 0;
  CHECK(ss_linear_policy_validate(&policy, &properties) == SS_OK);
  properties.nonlinear_hydraulic = 1;
  CHECK(ss_linear_policy_validate(&policy, &properties) == SS_ERROR_TYPE_MISMATCH);

  policy.method = SS_LINEAR_METHOD_FCG;
  policy.stopping_condition = SS_LINEAR_STOP_PRECONDITIONED_RESIDUAL_ORIGINAL_RHS;
  properties.nonlinear_hydraulic = 0;
  properties.pc_variable = 1;
  CHECK(ss_linear_policy_validate(&policy, &properties) == SS_OK);
  policy.stopping_condition = SS_LINEAR_STOP_NATURAL_RESIDUAL_INITIAL;
  CHECK(ss_linear_policy_validate(&policy, &properties) == SS_OK);
  policy.side = SS_LINEAR_SIDE_RIGHT;
  CHECK(ss_linear_policy_validate(&policy, &properties) == SS_ERROR_INVALID_ARGUMENT);

  policy = general_policy("direct", "lindir_");
  policy.operator_class = SS_LINEAR_OPERATOR_DIRECT_VALIDATION;
  policy.method = SS_LINEAR_METHOD_PREONLY;
  policy.pc = SS_LINEAR_PC_LU;
  policy.side = SS_LINEAR_SIDE_LEFT;
  properties.nonlinear_hydraulic = 0;
  CHECK(ss_linear_policy_validate(&policy, &properties) == SS_OK);
  return 0;
}

#if defined(PETSC_HAVE_HPDDM)
static int solve_hpddm_recycle_system(Mat matrix, Vec rhs, Vec solution,
                                      SsLinearRecycle *recycle,
                                      uint64_t value_generation) {
  SsLinearSolver *solver = NULL;
  SsLinearProblem problem = {0};
  SsLinearSetupRecord setup = {0};
  SsLinearSolveRecord solve = {0};
  SsLinearPolicy policy = general_policy("hpddm-gcrodr", "linrecycle_");
  policy.method = SS_LINEAR_METHOD_HPDDM_GCRODR;
  policy.restart = 8;
  policy.recycle_dimension = 3;
  policy.maximum_iterations = 80;
  policy.rtol = 1.0e-10;
  policy.atol = 1.0e-14;
  problem.operator_handle = matrix;
  problem.properties = general_properties();
  problem.policy_chain.count = 1U;
  problem.policy_chain.policies[0] = policy;
  problem.krylov_recycle = recycle;
  problem.structure_generation = 1U;
  problem.value_generation = value_generation;
  CHECK(VecSet(solution, 0.0) == PETSC_SUCCESS);
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(strcmp(setup.effective_ksp, KSPHPDDM) == 0);
  CHECK(strcmp(setup.effective_recycle_variant, "flexible") == 0);
  CHECK(setup.effective_recycle_dimension == 3);
  CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
  CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED);
  CHECK(solve.true_residual_norm <= solve.acceptance_threshold);
  CHECK(ss_linear_destroy(&solver) == SS_OK);
  return 0;
}
#endif

static int test_hpddm_recycle(void) {
#if defined(PETSC_HAVE_HPDDM)
  const PetscInt size = 20;
  SsLinearRecycle *recycle = NULL;
  SsLinearRecycleDiagnostics accepted = {0};
  SsLinearRecycleDiagnostics provisional = {0};
  SsLinearRecycleDiagnostics rolled_back = {0};
  SsLinearRecycleDiagnostics committed = {0};
  Mat easy_matrix = NULL;
  Mat matrix = NULL;
  Vec rhs = NULL;
  Vec solution = NULL;
  PetscInt row;
  const char *options[][2] = {
      {"-linrecycle_ksp_gmres_restart", "8"},
      {"-linrecycle_ksp_hpddm_type", "gcrodr"},
      {"-linrecycle_ksp_hpddm_variant", "flexible"},
      {"-linrecycle_ksp_hpddm_recycle", "3"},
      {"-linrecycle_ksp_hpddm_recycle_target", "SM"},
      {"-linrecycle_ksp_hpddm_recycle_strategy", "A"},
      {"-linrecycle_ksp_hpddm_orthogonalization", "mgs"},
  };
  size_t option;
  for (option = 0U; option < sizeof(options) / sizeof(options[0]); ++option) {
    CHECK(PetscOptionsSetValue(NULL, options[option][0], options[option][1]) ==
          PETSC_SUCCESS);
  }
  CHECK(MatCreateSeqAIJ(PETSC_COMM_SELF, size, size, 3, NULL, &matrix) ==
        PETSC_SUCCESS);
  for (row = 0; row < size; ++row) {
    CHECK(MatSetValue(matrix, row, row, 2.01, INSERT_VALUES) == PETSC_SUCCESS);
    if (row > 0) {
      CHECK(MatSetValue(matrix, row, row - 1, -1.0, INSERT_VALUES) == PETSC_SUCCESS);
    }
    if (row + 1 < size) {
      CHECK(MatSetValue(matrix, row, row + 1, -0.93, INSERT_VALUES) == PETSC_SUCCESS);
    }
  }
  CHECK(MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(MatCreateSeqAIJ(PETSC_COMM_SELF, size, size, 1, NULL, &easy_matrix) ==
        PETSC_SUCCESS);
  for (row = 0; row < size; ++row) {
    CHECK(MatSetValue(easy_matrix, row, row, 1.0, INSERT_VALUES) == PETSC_SUCCESS);
  }
  CHECK(MatAssemblyBegin(easy_matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(MatAssemblyEnd(easy_matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(MatCreateVecs(matrix, &solution, &rhs) == PETSC_SUCCESS);
  for (row = 0; row < size; ++row) {
    CHECK(VecSetValue(rhs, row, 1.0 + 0.03 * (PetscScalar)row, INSERT_VALUES) ==
          PETSC_SUCCESS);
  }
  CHECK(VecAssemblyBegin(rhs) == PETSC_SUCCESS);
  CHECK(VecAssemblyEnd(rhs) == PETSC_SUCCESS);
  CHECK(ss_linear_recycle_create(PETSC_COMM_SELF, &recycle) == SS_OK);
  CHECK(solve_hpddm_recycle_system(easy_matrix, rhs, solution, recycle, 1U) == 0);
  CHECK(ss_linear_recycle_diagnostics(recycle, &accepted) == SS_OK);
  CHECK(accepted.dimension == 0);
  CHECK(accepted.update_count == 0U);

  CHECK(solve_hpddm_recycle_system(matrix, rhs, solution, recycle, 2U) == 0);
  CHECK(ss_linear_recycle_diagnostics(recycle, &accepted) == SS_OK);
  CHECK(accepted.dimension == 3);
  CHECK(accepted.update_count == 1U);

  CHECK(ss_linear_recycle_begin(recycle, 7U) == SS_OK);
  CHECK(MatShift(matrix, 0.02) == PETSC_SUCCESS);
  CHECK(solve_hpddm_recycle_system(matrix, rhs, solution, recycle, 3U) == 0);
  CHECK(ss_linear_recycle_diagnostics(recycle, &provisional) == SS_OK);
  CHECK(provisional.update_count == accepted.update_count + 1U);
  CHECK(ss_linear_recycle_rollback(recycle) == SS_OK);
  CHECK(ss_linear_recycle_diagnostics(recycle, &rolled_back) == SS_OK);
  CHECK(rolled_back.dimension == accepted.dimension);
  CHECK(rolled_back.update_count == accepted.update_count);

  CHECK(ss_linear_recycle_begin(recycle, 8U) == SS_OK);
  CHECK(MatShift(matrix, 0.02) == PETSC_SUCCESS);
  CHECK(solve_hpddm_recycle_system(matrix, rhs, solution, recycle, 4U) == 0);
  CHECK(ss_linear_recycle_commit(recycle, 5U) == SS_OK);
  CHECK(ss_linear_recycle_diagnostics(recycle, &committed) == SS_OK);
  CHECK(committed.accepted_generation == 5U);
  CHECK(committed.update_count == accepted.update_count + 1U);
  CHECK(ss_linear_recycle_destroy(&recycle) == SS_OK);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(VecDestroy(&solution) == PETSC_SUCCESS);
  CHECK(MatDestroy(&easy_matrix) == PETSC_SUCCESS);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  for (option = 0U; option < sizeof(options) / sizeof(options[0]); ++option) {
    CHECK(PetscOptionsClearValue(NULL, options[option][0]) == PETSC_SUCCESS);
  }
#endif
  return 0;
}

static int test_linear_acceptance(void) {
  SsLinearPolicy policy = general_policy("accept", "linacc_");
  SsLinearSolveStatus status;
  PetscReal threshold = 0.0;

  policy.rtol = 0.1;
  policy.atol = 1.0e-12;
  CHECK(ss_linear_classify_acceptance(&policy, KSP_DIVERGED_ITS, 0.2, 1.0, 0.2, &status,
                                      &threshold) == SS_OK);
  CHECK(status == SS_LINEAR_SOLVE_DIVERGED_KSP);
  CHECK_NEAR(threshold, 0.1, 1.0e-15);

  CHECK(ss_linear_classify_acceptance(&policy, KSP_CONVERGED_RTOL, 0.09, 1.0, 0.11,
                                      &status, &threshold) == SS_OK);
  CHECK(status == SS_LINEAR_SOLVE_RESIDUAL_VERIFICATION_FAILED);

  CHECK(ss_linear_classify_acceptance(&policy, KSP_CONVERGED_RTOL, 0.09, 1.0, 0.09,
                                      &status, &threshold) == SS_OK);
  CHECK(status == SS_LINEAR_SOLVE_CONVERGED);

  CHECK(ss_linear_classify_acceptance(&policy, KSP_CONVERGED_ATOL, 0.0, 0.0, 5.0e-13,
                                      &status, &threshold) == SS_OK);
  CHECK(status == SS_LINEAR_SOLVE_CONVERGED);
  CHECK_NEAR(threshold, 1.0e-12, 1.0e-24);

  CHECK(ss_linear_classify_acceptance(&policy, KSP_CONVERGED_RTOL, 0.0, 1.0,
                                      (PetscReal)NAN, &status, &threshold) == SS_OK);
  CHECK(status == SS_LINEAR_SOLVE_NONFINITE);
  return 0;
}

static int test_hessenberg(void) {
  const double matrix_values[] = {1.0, 1.0, 0.0, 1.0e-8};
  const double rhs_values[] = {0.0, 1.0e-8};
  const double zero[] = {0.0, 0.0};
  const double expected[] = {-1.0, 1.0};
  SsLinearSolver *solver = NULL;
  SsLinearProblem problem = {0};
  SsLinearSetupRecord setup = {0};
  SsLinearSolveRecord solve = {0};
  Mat matrix = NULL;
  Vec rhs = NULL;
  Vec solution = NULL;
  int repeat;

  CHECK(create_dense(2, matrix_values, &matrix) == 0);
  CHECK(create_vector(2, rhs_values, &rhs) == 0);
  CHECK(create_vector(2, zero, &solution) == 0);
  problem.operator_handle = matrix;
  problem.properties = general_properties();
  problem.policy_chain.count = 1U;
  problem.policy_chain.policies[0] = general_policy("stable-fgmres", "linqr_");
  problem.policy_chain.policies[0].restart = 2;
  problem.policy_chain.policies[0].maximum_iterations = 8;
  problem.policy_chain.policies[0].rtol = 1.0e-13;
  problem.policy_chain.policies[0].atol = 1.0e-15;
  problem.structure_generation = 11U;
  problem.value_generation = 17U;
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(strcmp(setup.effective_ksp, "fgmres") == 0);
  CHECK(setup.effective_restart == 2);
  CHECK(strcmp(setup.effective_orthogonalization, "classical_gram_schmidt_twice") == 0);
  CHECK(setup.admitted_vector_bytes <=
        problem.policy_chain.policies[0].vector_memory_cap_bytes);
  CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
  CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED);
  CHECK(solve.reason > 0);
  CHECK_NEAR(solve.effective_rtol, 1.0e-13, 1.0e-28);
  CHECK(solve.true_residual_norm <= solve.acceptance_threshold);
  CHECK(check_vector(solution, 2, expected, 2.0e-8) == 0);
  CHECK(VecSet(solution, 0.0) == PETSC_SUCCESS);
  CHECK(ss_linear_solve_with_relative_tolerance(solver, rhs, solution, 5.0e-14,
                                                &solve) == SS_OK);
  CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED);
  CHECK_NEAR(solve.effective_rtol, 5.0e-14, 1.0e-28);
  CHECK(solve.true_residual_norm <= solve.acceptance_threshold);
  CHECK(check_vector(solution, 2, expected, 2.0e-8) == 0);
  for (repeat = 0; repeat < 32; ++repeat) {
    CHECK(VecSet(solution, 0.0) == PETSC_SUCCESS);
    CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
    CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED);
    CHECK(check_vector(solution, 2, expected, 2.0e-8) == 0);
  }
  CHECK(solve.solve_count == 34U);
  CHECK(VecSet(solution, (PetscScalar)NAN) == PETSC_SUCCESS);
  CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
  CHECK(solve.status == SS_LINEAR_SOLVE_NONFINITE);
  CHECK(ss_linear_destroy(&solver) == SS_OK);
  CHECK(VecDestroy(&solution) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}

static int test_hydraulic_rhs_normalization(void) {
  const double matrix_values[] = {2.0, 0.0, 0.0, 3.0};
  const double rhs_values[] = {2.0e-24, -3.0e-24};
  const double zero[] = {0.0, 0.0};
  const double expected[] = {1.0e-24, -1.0e-24};
  SsLinearSolver *solver = NULL;
  SsLinearProblem problem = {0};
  SsLinearSetupRecord setup = {0};
  SsLinearSolveRecord solve = {0};
  Mat matrix = NULL;
  Vec rhs = NULL;
  Vec solution = NULL;

  CHECK(create_dense(2, matrix_values, &matrix) == 0);
  CHECK(create_vector(2, rhs_values, &rhs) == 0);
  CHECK(create_vector(2, zero, &solution) == 0);
  problem.operator_handle = matrix;
  problem.properties = general_properties();
  problem.properties.nonlinear_hydraulic = 1;
  problem.policy_chain.count = 1U;
  problem.policy_chain.policies[0] =
      general_policy("hydraulic-normalized-fgmres", "linhydnorm_");
  problem.policy_chain.policies[0].restart = 2;
  problem.policy_chain.policies[0].maximum_iterations = 8;
  problem.policy_chain.policies[0].rtol = 1.0e-12;
  problem.policy_chain.policies[0].atol = 1.0e-40;
  problem.structure_generation = 31U;
  problem.value_generation = 37U;
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
  CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED);
  CHECK(solve.reason > 0);
  CHECK_NEAR(solve.rhs_norm, sqrt(13.0) * 1.0e-24, 5.0e-36);
  CHECK(solve.true_residual_norm <= solve.acceptance_threshold);
  CHECK(check_vector(solution, 2, expected, 5.0e-36) == 0);
  CHECK(ss_linear_destroy(&solver) == SS_OK);
  CHECK(VecDestroy(&solution) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}

static int test_fallback(void) {
  const double dense_values[] = {3.0, 1.0, 1.0, 3.0};
  const double rhs_values[] = {1.0, 2.0};
  const double zero[] = {0.0, 0.0};
  const double expected[] = {0.125, 0.625};
  Shell2x2 context = {{3.0, 1.0, 1.0, 3.0}};
  SsLinearSolver *solver = NULL;
  SsLinearProblem problem = {0};
  SsLinearSetupRecord setup = {0};
  SsLinearSolveRecord solve = {0};
  Mat matrix = NULL;
  Vec rhs = NULL;
  Vec solution = NULL;
  SsLinearPolicy direct = general_policy("direct-primary", "linfb0_");

  CHECK(MatCreateShell(PETSC_COMM_SELF, 2, 2, 2, 2, &context, &matrix) ==
        PETSC_SUCCESS);
  CHECK(MatShellSetOperation(matrix, MATOP_MULT, (void (*)(void))shell_mult) ==
        PETSC_SUCCESS);
  CHECK(create_vector(2, rhs_values, &rhs) == 0);
  CHECK(create_vector(2, zero, &solution) == 0);
  direct.operator_class = SS_LINEAR_OPERATOR_DIRECT_VALIDATION;
  direct.method = SS_LINEAR_METHOD_PREONLY;
  direct.pc = SS_LINEAR_PC_LU;
  direct.side = SS_LINEAR_SIDE_LEFT;
  problem.operator_handle = matrix;
  problem.properties = general_properties();
  problem.policy_chain.count = 2U;
  problem.policy_chain.policies[0] = direct;
  problem.policy_chain.policies[1] =
      general_policy("declared-fgmres-fallback", "linfb1_");
  problem.structure_generation = 21U;
  problem.value_generation = 22U;
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(setup.attempt_count == 2U);
  CHECK(setup.attempts[0].setup_status != SS_OK);
  CHECK(setup.attempts[0].selected == 0);
  CHECK(setup.attempts[1].setup_status == SS_OK);
  CHECK(setup.attempts[1].selected != 0);
  CHECK(strcmp(setup.selected_identity, "declared-fgmres-fallback") == 0);
  CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
  CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED);
  CHECK(check_vector(solution, 2, expected, 2.0e-11) == 0);
  CHECK(ss_linear_destroy(&solver) == SS_OK);
  CHECK(VecDestroy(&solution) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);

  problem = (SsLinearProblem){0};
  setup = (SsLinearSetupRecord){0};
  solve = (SsLinearSolveRecord){0};
  CHECK(create_dense(2, dense_values, &matrix) == 0);
  CHECK(create_vector(2, rhs_values, &rhs) == 0);
  CHECK(create_vector(2, zero, &solution) == 0);
  problem.operator_handle = matrix;
  problem.properties = general_properties();
  problem.policy_chain.count = 1U;
  problem.policy_chain.policies[0] = direct;
  problem.structure_generation = 23U;
  problem.value_generation = 24U;
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(strcmp(setup.effective_ksp, "preonly") == 0);
  CHECK(strcmp(setup.effective_pc, "lu") == 0);
  CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
  CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED);
  CHECK(check_vector(solution, 2, expected, 2.0e-14) == 0);
  CHECK(ss_linear_destroy(&solver) == SS_OK);
  CHECK(VecDestroy(&solution) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}

static int setup_options_case(const char *option, const char *value,
                              SsStatus expected) {
  const double matrix_values[] = {2.0, 0.0, 0.0, 3.0};
  SsLinearSolver *solver = NULL;
  SsLinearProblem problem = {0};
  SsLinearSetupRecord setup = {0};
  Mat matrix = NULL;
  SsStatus status;
  CHECK(create_dense(2, matrix_values, &matrix) == 0);
  problem.operator_handle = matrix;
  problem.properties = general_properties();
  problem.policy_chain.count = 1U;
  problem.policy_chain.policies[0] = general_policy("options", "linopts_");
  problem.structure_generation = 1U;
  problem.value_generation = 1U;
  CHECK(PetscOptionsSetValue(NULL, option, value) == PETSC_SUCCESS);
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  status = ss_linear_setup(solver, &problem, &setup);
  CHECK(status == expected);
  CHECK(ss_linear_destroy(&solver) == SS_OK);
  CHECK(PetscOptionsClearValue(NULL, option) == PETSC_SUCCESS);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}

static int test_options(void) {
  const double matrix_values[] = {2.0, 0.0, 0.0, 3.0};
  SsLinearSolver *solver = NULL;
  SsLinearProblem problem = {0};
  SsLinearSetupRecord setup = {0};
  Mat matrix = NULL;
  CHECK(setup_options_case("-linopts_ksp_type", "cg", SS_ERROR_INTEGRITY) == 0);
  CHECK(setup_options_case("-linopts_ksp_norm_type", "none", SS_ERROR_INTEGRITY) == 0);
  CHECK(setup_options_case("-linopts_ksp_initial_guess_nonzero", "true",
                           SS_ERROR_INTEGRITY) == 0);
  CHECK(setup_options_case("-linopts_ksp_initial_guess_knoll", "true",
                           SS_ERROR_INTEGRITY) == 0);
  CHECK(setup_options_case("-linopts_ksp_error_if_not_converged", "true",
                           SS_ERROR_INTEGRITY) == 0);
  CHECK(setup_options_case("-linopts_ksp_gmres_classicalgramschmidt", "true", SS_OK) ==
        0);
  CHECK(setup_options_case("-linopts_ksp_gmres_modifiedgramschmidt", "true", SS_OK) ==
        0);
  CHECK(setup_options_case("-linopts_ksp_gmres_cgs_refinement_type", "refine_always",
                           SS_OK) == 0);
  CHECK(setup_options_case("-linopts_ksp_gmres_cgs_refinement_type", "refine_never",
                           SS_ERROR_INTEGRITY) == 0);
  CHECK(setup_options_case("-linopts_not_allowed", "1", SS_ERROR_INTEGRITY) == 0);
  CHECK(create_dense(2, matrix_values, &matrix) == 0);
  problem.operator_handle = matrix;
  problem.properties = general_properties();
  problem.policy_chain.count = 1U;
  problem.policy_chain.policies[0] = general_policy("options-clean", "linopts_");
  problem.structure_generation = 2U;
  problem.value_generation = 3U;
  CHECK(PetscOptionsSetValue(NULL, "-linopts_ksp_gmres_classicalgramschmidt", "true") ==
        PETSC_SUCCESS);
  CHECK(PetscOptionsSetValue(NULL, "-linopts_ksp_gmres_cgs_refinement_type",
                             "refine_always") == PETSC_SUCCESS);
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(strcmp(setup.effective_orthogonalization, "classical_gram_schmidt_twice") == 0);
  CHECK(ss_linear_destroy(&solver) == SS_OK);
  CHECK(PetscOptionsClearValue(NULL, "-linopts_ksp_gmres_classicalgramschmidt") ==
        PETSC_SUCCESS);
  CHECK(PetscOptionsClearValue(NULL, "-linopts_ksp_gmres_cgs_refinement_type") ==
        PETSC_SUCCESS);
  CHECK(PetscOptionsSetValue(NULL, "-linopts_ksp_gmres_classicalgramschmidt", "true") ==
        PETSC_SUCCESS);
  CHECK(PetscOptionsSetValue(NULL, "-linopts_ksp_gmres_cgs_refinement_type",
                             "refine_never") == PETSC_SUCCESS);
  problem.policy_chain.policies[0].orthogonalization =
      SS_LINEAR_ORTHOGONALIZATION_CGS_UNREFINED;
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(strcmp(setup.effective_orthogonalization, "classical_gram_schmidt_once") == 0);
  CHECK(ss_linear_destroy(&solver) == SS_OK);
  CHECK(PetscOptionsClearValue(NULL, "-linopts_ksp_gmres_classicalgramschmidt") ==
        PETSC_SUCCESS);
  CHECK(PetscOptionsClearValue(NULL, "-linopts_ksp_gmres_cgs_refinement_type") ==
        PETSC_SUCCESS);
  problem.policy_chain.policies[0].orthogonalization =
      SS_LINEAR_ORTHOGONALIZATION_PETSC;
  problem.policy_chain.policies[0].vector_memory_cap_bytes = 1U;
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_ERROR_RESOURCE);
  CHECK(setup.attempt_count == 1U);
  CHECK(setup.attempts[0].setup_status == SS_ERROR_RESOURCE);
  CHECK(ss_linear_destroy(&solver) == SS_OK);
  problem.policy_chain.policies[0].vector_memory_cap_bytes = 1024U * 1024U;
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(strcmp(setup.effective_ksp, "fgmres") == 0);
  CHECK(strcmp(setup.effective_pc, "none") == 0);
  CHECK(setup.effective_side == SS_LINEAR_SIDE_RIGHT);
  CHECK(setup.effective_norm_type == KSP_NORM_UNPRECONDITIONED);
  CHECK(strcmp(setup.effective_orthogonalization, "classical_gram_schmidt_twice") == 0);
  CHECK(setup.effective_initial_guess_nonzero == PETSC_FALSE);
  CHECK(setup.effective_initial_guess_knoll == PETSC_FALSE);
  CHECK(setup.effective_error_if_not_converged == PETSC_FALSE);
  CHECK(setup.operator_state >= 0);
  CHECK_NEAR(setup.effective_rtol, problem.policy_chain.policies[0].rtol, 0.0);
  CHECK_NEAR(setup.effective_atol, problem.policy_chain.policies[0].atol, 0.0);
  CHECK_NEAR(setup.effective_dtol, problem.policy_chain.policies[0].dtol, 0.0);
  CHECK(setup.reused == 0);
  CHECK(setup.setup_count == 1U);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(setup.reused != 0);
  CHECK(setup.setup_count == 1U);
  problem.policy_chain.policies[0].stopping_condition =
      SS_LINEAR_STOP_PROJECTED_RHS_PETSC;
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(setup.reused == 0);
  CHECK(setup.setup_count == 2U);
  problem.policy_chain.policies[0].stopping_condition =
      SS_LINEAR_STOP_ORIGINAL_RHS_TRUE_RESIDUAL;
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(setup.reused == 0);
  CHECK(setup.setup_count == 3U);
  CHECK(PetscOptionsSetValue(NULL, "-linear_reuse_probe", "true") == PETSC_SUCCESS);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(setup.reused == 0);
  CHECK(setup.setup_count == 4U);
  CHECK(PetscOptionsClearValue(NULL, "-linear_reuse_probe") == PETSC_SUCCESS);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(setup.reused == 0);
  CHECK(setup.setup_count == 5U);
  ++problem.value_generation;
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(setup.reused != 0);
  CHECK(setup.setup_count == 5U);
  CHECK(MatSetValue(matrix, 0, 0, 2.0, INSERT_VALUES) == PETSC_SUCCESS);
  CHECK(MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_ERROR_INTEGRITY);
  ++problem.value_generation;
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(setup.reused != 0);
  CHECK(setup.setup_count == 5U);
  CHECK(ss_linear_invalidate(solver, 1U) == SS_OK);
  {
    const double rhs_values[] = {2.0, 3.0};
    const double zero_values[] = {0.0, 0.0};
    Vec rhs = NULL;
    Vec solution = NULL;
    SsLinearSolveRecord solve = {0};
    CHECK(create_vector(2, rhs_values, &rhs) == 0);
    CHECK(create_vector(2, zero_values, &solution) == 0);
    CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_ERROR_INVALID_STATE);
    CHECK(solve.status == SS_LINEAR_SOLVE_SETUP_FAILED);
    CHECK(VecDestroy(&solution) == PETSC_SUCCESS);
    CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  }
  CHECK(ss_linear_destroy(&solver) == SS_OK);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}

int main(int argc, char **argv) {
  int status;
  if (argc != 2) {
    fputs("usage: ss_test_linear_deflation TEST-ID\n", stderr);
    return 2;
  }
  if (PetscInitialize(&argc, &argv, NULL, NULL) != PETSC_SUCCESS) {
    return 3;
  }
  if (strcmp(argv[1], "LIN-UNIT-POLICY-001") == 0) {
    status = test_linear_policy();
  } else if (strcmp(argv[1], "LIN-UNIT-ACCEPT-001") == 0) {
    status = test_linear_acceptance();
  } else if (strcmp(argv[1], "LIN-REG-HESSENBERG-001") == 0) {
    status = test_hessenberg();
    if (status == 0) {
      status = test_hydraulic_rhs_normalization();
    }
  } else if (strcmp(argv[1], "LIN-UNIT-FALLBACK-001") == 0) {
    status = test_fallback();
  } else if (strcmp(argv[1], "LIN-CONTRACT-OPTIONS-001") == 0) {
    status = test_options();
  } else if (strcmp(argv[1], "LIN-REG-KRYLOV-RECYCLE-001") == 0) {
    status = test_hpddm_recycle();
  } else if (strcmp(argv[1], "DEF-UNIT-PROJECTOR-001") == 0) {
    status = test_projector();
    if (status == 0) {
      status = test_legacy_aorth_projector();
    }
    if (status == 0) {
      status = test_lu_late_row_pivot();
    }
    if (status == 0) {
      status = test_configured_invariant_tolerance();
    }
    if (status == 0) {
      status = test_persistent_restore_bits();
    }
    if (status == 0) {
      status = test_krylov_compression();
    }
  } else if (strcmp(argv[1], "DEF-REG-SIGN-001") == 0) {
    status = test_signed_direction();
  } else if (strcmp(argv[1], "DEF-UNIT-OBLIQUE-001") == 0) {
    status = test_oblique();
    if (status == 0) {
      status = test_deflated_stopping_threshold();
    }
    if (status == 0) {
      status = test_deflated_pmg_projection();
    }
    if (status == 0) {
      status = test_aorth_true_residual_reused_guess();
    }
  } else if (strcmp(argv[1], "DEF-UNIT-DEPENDENT-001") == 0) {
    status = test_dependent();
  } else if (strcmp(argv[1], "DEF-UNIT-ROLLBACK-001") == 0) {
    status = test_rollback();
  } else {
    fprintf(stderr, "unknown test id: %s\n", argv[1]);
    status = 2;
  }
  if (PetscFinalize() != PETSC_SUCCESS) {
    return 4;
  }
  return status;
}
