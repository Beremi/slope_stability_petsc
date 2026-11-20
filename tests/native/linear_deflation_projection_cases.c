/* Deflation, projection and history contract cases.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "linear_deflation_internal.h"

static int build_deflation(const double *matrix_values, PetscInt size,
                           const double *left_values, const double *right_values,
                           size_t count, SsDeflation **deflation, Mat *matrix) {
  SsDeflationPolicy policy = deflation_policy(count + 2U);
  size_t column;
  CHECK(create_dense(size, matrix_values, matrix) == 0);
  CHECK(ss_deflation_create(PETSC_COMM_SELF, &policy, deflation) == SS_OK);
  CHECK(ss_deflation_begin(*deflation, 100U) == SS_OK);
  for (column = 0U; column < count; ++column) {
    Vec left = NULL;
    Vec right = NULL;
    SsDeflationAdmissionRecord record = {0};
    CHECK(create_vector(size, &left_values[column * (size_t)size], &left) == 0);
    CHECK(create_vector(size, &right_values[column * (size_t)size], &right) == 0);
    CHECK(ss_deflation_add_candidate(*deflation, left, right, "SOL-001", &record) ==
          SS_OK);
    CHECK(record.admitted != 0);
    CHECK(VecDestroy(&right) == PETSC_SUCCESS);
    CHECK(VecDestroy(&left) == PETSC_SUCCESS);
  }
  CHECK(ss_deflation_refresh(*deflation, *matrix, 101U) == SS_OK);
  return 0;
}

int test_configured_invariant_tolerance(void) {
  const double epsilon = 1.0e-7;
  const double matrix_values[] = {
      (1.0 / 14.0) + epsilon, 2.0 / 14.0, 3.0 / 14.0, 2.0 / 14.0,
      (4.0 / 14.0) + epsilon, 6.0 / 14.0, 3.0 / 14.0, 6.0 / 14.0,
      (9.0 / 14.0) + epsilon,
  };
  const double basis_values[][3] = {
      {1.0, 2.0, 3.0},
      {4.0, 5.0, 7.0},
  };
  SsDeflationPolicy policy = deflation_policy(4U);
  SsDeflation *deflation = NULL;
  SsDeflationDiagnostics diagnostics = {0};
  SsLinearSolver *solver = NULL;
  SsLinearProblem problem = {0};
  SsLinearSetupRecord setup = {0};
  Mat matrix = NULL;
  PetscReal invariant_error = 0.0;
  size_t column;
  policy.condition_limit = 1.0e14;
  policy.invariant_tolerance = 1.0e-6;
  CHECK(create_dense(3, matrix_values, &matrix) == 0);
  CHECK(ss_deflation_create(PETSC_COMM_SELF, &policy, &deflation) == SS_OK);
  CHECK(ss_deflation_begin(deflation, 1U) == SS_OK);
  for (column = 0U; column < 2U; ++column) {
    SsDeflationAdmissionRecord admission = {0};
    Vec basis = NULL;
    CHECK(create_vector(3, basis_values[column], &basis) == 0);
    CHECK(ss_deflation_add_candidate(deflation, basis, basis, "tolerance-regression",
                                     &admission) == SS_OK);
    CHECK(admission.admitted != 0);
    CHECK(VecDestroy(&basis) == PETSC_SUCCESS);
  }
  CHECK(ss_deflation_refresh(deflation, matrix, 1U) == SS_OK);
  CHECK(ss_deflation_invariant_error(deflation, matrix, &invariant_error) == SS_OK);
  CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
  CHECK(diagnostics.invariants_validated != 0);
  CHECK(invariant_error > 1.0e-10);
  CHECK(invariant_error <= policy.invariant_tolerance);
  problem.operator_handle = matrix;
  problem.properties = general_properties();
  problem.policy_chain.count = 1U;
  problem.policy_chain.policies[0] =
      general_policy("configured-invariant-tolerance", "lintol_");
  problem.deflation = deflation;
  problem.structure_generation = 1U;
  problem.value_generation = 2U;
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(ss_linear_destroy(&solver) == SS_OK);
  CHECK(ss_deflation_rollback(deflation) == SS_OK);
  CHECK(ss_deflation_destroy(&deflation) == SS_OK);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}

int test_projector(void) {
  const double matrix_values[] = {4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0};
  const double bases[] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  const double rhs_values[] = {1.0, 2.0, 3.0};
  const double expected_q[] = {1.0 / 11.0, 7.0 / 11.0, 0.0};
  const double expected_pr[] = {3.0 / 11.0, -12.0 / 11.0, 3.0};
  const double expected_pl[] = {0.0, 0.0, 26.0 / 11.0};
  SsDeflation *deflation = NULL;
  Mat matrix = NULL;
  Vec rhs = NULL;
  Vec output = NULL;
  Vec right = NULL;
  Vec middle = NULL;
  Vec explicit_projected = NULL;
  Vec fused_projected = NULL;
  Vec difference = NULL;
  SsLinearSolver *solver = NULL;
  SsLinearProblem problem = {0};
  SsLinearSetupRecord setup = {0};
  PetscReal error = 0.0;

  CHECK(build_deflation(matrix_values, 3, bases, bases, 2U, &deflation, &matrix) == 0);
  CHECK(create_vector(3, rhs_values, &rhs) == 0);
  CHECK(VecDuplicate(rhs, &output) == PETSC_SUCCESS);
  CHECK(VecDuplicate(rhs, &right) == PETSC_SUCCESS);
  CHECK(VecDuplicate(rhs, &middle) == PETSC_SUCCESS);
  CHECK(VecDuplicate(rhs, &explicit_projected) == PETSC_SUCCESS);
  CHECK(VecDuplicate(rhs, &fused_projected) == PETSC_SUCCESS);
  CHECK(VecDuplicate(rhs, &difference) == PETSC_SUCCESS);
  CHECK(ss_deflation_apply_q(deflation, rhs, output) == SS_OK);
  CHECK(check_vector(output, 3, expected_q, 2.0e-14) == 0);
  CHECK(ss_deflation_apply_right(deflation, matrix, rhs, output) == SS_OK);
  CHECK(check_vector(output, 3, expected_pr, 2.0e-14) == 0);
  CHECK(ss_deflation_apply_left(deflation, matrix, rhs, output) == SS_OK);
  CHECK(check_vector(output, 3, expected_pl, 2.0e-14) == 0);
  CHECK(ss_deflation_apply_right(deflation, matrix, rhs, right) == SS_OK);
  CHECK(MatMult(matrix, right, middle) == PETSC_SUCCESS);
  CHECK(ss_deflation_apply_left(deflation, matrix, middle, explicit_projected) ==
        SS_OK);
  CHECK(ss_deflation_apply_projected_operator(deflation, matrix, rhs,
                                              fused_projected) == SS_OK);
  CHECK(VecWAXPY(difference, -1.0, explicit_projected, fused_projected) ==
        PETSC_SUCCESS);
  CHECK(VecNorm(difference, NORM_2, &error) == PETSC_SUCCESS);
  CHECK(error <= 2.0e-14);
  CHECK(ss_deflation_invariant_error(deflation, matrix, &error) == SS_OK);
  CHECK(error <= 2.0e-14);
  problem.operator_handle = matrix;
  problem.properties = general_properties();
  problem.properties.measured_symmetric = 1;
  problem.properties.positive_energy = 1;
  problem.properties.pc_symmetric = 1;
  problem.properties.pc_positive = 1;
  problem.properties.pc_variable = 0;
  problem.policy_chain.count = 1U;
  problem.policy_chain.policies[0] = general_policy("cg-deflation", "lindefl_");
  problem.policy_chain.policies[0].operator_class = SS_LINEAR_OPERATOR_SPD_VERIFIED;
  problem.policy_chain.policies[0].method = SS_LINEAR_METHOD_CG;
  problem.policy_chain.policies[0].side = SS_LINEAR_SIDE_LEFT;
  problem.deflation = deflation;
  problem.structure_generation = 201U;
  problem.value_generation = 202U;
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_ERROR_TYPE_MISMATCH);
  CHECK(ss_linear_destroy(&solver) == SS_OK);
  CHECK(VecDestroy(&difference) == PETSC_SUCCESS);
  CHECK(VecDestroy(&fused_projected) == PETSC_SUCCESS);
  CHECK(VecDestroy(&explicit_projected) == PETSC_SUCCESS);
  CHECK(VecDestroy(&middle) == PETSC_SUCCESS);
  CHECK(VecDestroy(&right) == PETSC_SUCCESS);
  CHECK(VecDestroy(&output) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(ss_deflation_destroy(&deflation) == SS_OK);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}

int test_legacy_aorth_projector(void) {
  const double matrix_values[] = {
      4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0,
  };
  const double first_values[] = {1.0, 1.0, 0.0};
  const double second_values[] = {0.0, 1.0, 1.0};
  SsDeflationPolicy policy = deflation_policy(4U);
  SsDeflation *deflation = NULL;
  SsDeflationAdmissionRecord admission = {0};
  SsDeflationDiagnostics before = {0};
  SsDeflationDiagnostics after = {0};
  SsDeflationProjector projector = SS_DEFLATION_PROJECTOR_GENERAL_LEFT_RIGHT;
  Mat matrix = NULL;
  Vec first = NULL;
  Vec second = NULL;
  Vec basis[2] = {NULL, NULL};
  Vec image[2] = {NULL, NULL};
  Vec output = NULL;
  Vec projected_rhs = NULL;
  Vec restored = NULL;
  PetscScalar product = 0.0;
  PetscReal norm = 0.0;
  char snapshot_digest[SS_SHA256_HEX_CAPACITY] = {0};
  char rollback_digest[SS_SHA256_HEX_CAPACITY] = {0};
  size_t row;
  size_t column;

  policy.norm_tolerance = 1.0e-3;
  policy.invariant_tolerance = 2.0e-11;
  policy.projector = SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL;
  CHECK(create_dense(3, matrix_values, &matrix) == 0);
  CHECK(create_vector(3, first_values, &first) == 0);
  CHECK(create_vector(3, second_values, &second) == 0);
  CHECK(ss_deflation_create(PETSC_COMM_SELF, &policy, &deflation) == SS_OK);
  CHECK(ss_deflation_projector(deflation, &projector) == SS_OK);
  CHECK(projector == SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL);
  CHECK(ss_deflation_begin(deflation, 1U) == SS_OK);
  CHECK(ss_deflation_add_candidate(deflation, first, first, "first", &admission) ==
        SS_OK);
  CHECK(admission.admitted != 0);
  CHECK(ss_deflation_add_candidate(deflation, second, second, "second", &admission) ==
        SS_OK);
  CHECK(admission.admitted != 0);
  /* Legacy admission is deferred to the tangent-dependent A-energy pass. */
  CHECK(ss_deflation_add_candidate(deflation, second, second, "duplicate",
                                   &admission) == SS_OK);
  CHECK(admission.admitted != 0);
  CHECK(ss_deflation_refresh(deflation, matrix, 1U) == SS_OK);
  CHECK(ss_deflation_diagnostics(deflation, &after) == SS_OK);
  CHECK(after.active_count == 2U);
  CHECK(after.provisional_count == 2U);
  CHECK(after.invariants_validated != 0);
  CHECK(after.maximum_invariant_error <= policy.invariant_tolerance);
  CHECK(ss_deflation_commit(deflation, 1U) == SS_OK);
  for (column = 0U; column < 2U; ++column) {
    Vec left = NULL;
    const char *provenance = NULL;
    CHECK(ss_deflation_persistent_entry(deflation, column, &left, &basis[column],
                                        &provenance) == SS_OK);
    CHECK(VecDuplicate(basis[column], &image[column]) == PETSC_SUCCESS);
    CHECK(MatMult(matrix, basis[column], image[column]) == PETSC_SUCCESS);
  }
  for (row = 0U; row < 2U; ++row) {
    for (column = 0U; column < 2U; ++column) {
      CHECK(VecDot(basis[row], image[column], &product) == PETSC_SUCCESS);
      CHECK_NEAR(PetscRealPart(product), row == column ? 1.0 : 0.0, 2.0e-12);
    }
  }
  CHECK(VecDuplicate(first, &output) == PETSC_SUCCESS);
  CHECK(VecDuplicate(first, &projected_rhs) == PETSC_SUCCESS);
  CHECK(ss_deflation_diagnostics(deflation, &before) == SS_OK);
  CHECK(ss_deflation_apply_cached_right(deflation, basis[0], output) == SS_OK);
  CHECK(VecNorm(output, NORM_2, &norm) == PETSC_SUCCESS);
  CHECK(norm <= 2.0e-12);
  CHECK(ss_deflation_coarse_initial_guess_and_project_rhs(deflation, image[0], output,
                                                          projected_rhs) == SS_OK);
  CHECK(VecAXPY(output, -1.0, basis[0]) == PETSC_SUCCESS);
  CHECK(VecNorm(output, NORM_2, &norm) == PETSC_SUCCESS);
  CHECK(norm <= 2.0e-12);
  CHECK(VecNorm(projected_rhs, NORM_2, &norm) == PETSC_SUCCESS);
  CHECK(norm <= 2.0e-12);
  CHECK(ss_deflation_diagnostics(deflation, &after) == SS_OK);
  CHECK(after.projected_apply_count - before.projected_apply_count == 1U);
  CHECK(after.q_apply_count - before.q_apply_count == 1U);

  /* A new tangent must trigger the same complete explicit reorthogonalization. */
  CHECK(MatSetValue(matrix, 0, 0, 5.0, INSERT_VALUES) == PETSC_SUCCESS);
  CHECK(MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(ss_deflation_refresh(deflation, matrix, 2U) == SS_OK);
  for (column = 0U; column < 2U; ++column) {
    Vec left = NULL;
    const char *provenance = NULL;
    /* Refresh replaces these borrowed A-orthogonalized columns. */
    CHECK(ss_deflation_persistent_entry(deflation, column, &left, &basis[column],
                                        &provenance) == SS_OK);
    CHECK(MatMult(matrix, basis[column], image[column]) == PETSC_SUCCESS);
  }
  for (row = 0U; row < 2U; ++row) {
    for (column = 0U; column < 2U; ++column) {
      CHECK(VecDot(basis[row], image[column], &product) == PETSC_SUCCESS);
      CHECK_NEAR(PetscRealPart(product), row == column ? 1.0 : 0.0, 2.0e-12);
    }
  }

  /* The legacy continuation snapshots raw accepted-state columns around an
   * attempt.  Reorthogonalization may mutate the working basis, but rollback
   * must recover the snapshot byte-for-byte. */
  CHECK(ss_deflation_persistent_digest(deflation, snapshot_digest) == SS_OK);
  CHECK(ss_deflation_begin(deflation, 2U) == SS_OK);
  CHECK(ss_deflation_add_candidate(deflation, first, first, "trial-direction",
                                   &admission) == SS_OK);
  CHECK(ss_deflation_refresh(deflation, matrix, 3U) == SS_OK);
  CHECK(ss_deflation_rollback(deflation) == SS_OK);
  CHECK(ss_deflation_persistent_digest(deflation, rollback_digest) == SS_OK);
  CHECK(strcmp(snapshot_digest, rollback_digest) == 0);

  /* On acceptance, restore the same snapshot, append the raw state, and
   * commit without an extra old-tangent A-orthogonalization. */
  CHECK(ss_deflation_begin(deflation, 3U) == SS_OK);
  CHECK(ss_deflation_add_candidate(deflation, second, second, "trial-direction",
                                   &admission) == SS_OK);
  CHECK(ss_deflation_refresh(deflation, matrix, 4U) == SS_OK);
  CHECK(ss_deflation_discard_provisional(deflation) == SS_OK);
  CHECK(ss_deflation_add_candidate(deflation, first, first, "accepted-state",
                                   &admission) == SS_OK);
  CHECK(ss_deflation_commit(deflation, 2U) == SS_OK);
  CHECK(ss_deflation_persistent_count(deflation, &column) == SS_OK);
  CHECK(column == 3U);
  {
    Vec left = NULL;
    const char *provenance = NULL;
    CHECK(ss_deflation_persistent_entry(deflation, 2U, &left, &restored, &provenance) ==
          SS_OK);
    CHECK(strcmp(provenance, "accepted-state") == 0);
    CHECK(VecCopy(restored, output) == PETSC_SUCCESS);
    CHECK(VecAXPY(output, -1.0, first) == PETSC_SUCCESS);
    CHECK(VecNorm(output, NORM_2, &norm) == PETSC_SUCCESS);
    CHECK(norm == 0.0);
  }
  CHECK(VecDestroy(&projected_rhs) == PETSC_SUCCESS);
  CHECK(VecDestroy(&output) == PETSC_SUCCESS);
  CHECK(VecDestroy(&image[1]) == PETSC_SUCCESS);
  CHECK(VecDestroy(&image[0]) == PETSC_SUCCESS);
  CHECK(VecDestroy(&second) == PETSC_SUCCESS);
  CHECK(VecDestroy(&first) == PETSC_SUCCESS);
  CHECK(ss_deflation_destroy(&deflation) == SS_OK);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}

int test_lu_late_row_pivot(void) {
  /*
   * Column one pivots rows with different column-zero multipliers.  Replaying
   * that pivot during, rather than before, the forward solve corrupts Q.
   */
  const double matrix_values[] = {
      4.0, 1.0, 1.0, 2.0, 2.0, 10.0, 1.0, 10.0, 3.0,
  };
  const double bases[] = {
      1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
  };
  const double rhs_values[] = {4.0, 2.0, 1.0};
  const double expected[] = {1.0, 0.0, 0.0};
  SsDeflation *deflation = NULL;
  SsDeflationDiagnostics diagnostics = {0};
  Mat matrix = NULL;
  Vec rhs = NULL;
  Vec output = NULL;
  PetscReal invariant_error = 0.0;

  CHECK(build_deflation(matrix_values, 3, bases, bases, 3U, &deflation, &matrix) == 0);
  CHECK(create_vector(3, rhs_values, &rhs) == 0);
  CHECK(VecDuplicate(rhs, &output) == PETSC_SUCCESS);
  CHECK(ss_deflation_apply_q(deflation, rhs, output) == SS_OK);
  CHECK(check_vector(output, 3, expected, 2.0e-13) == 0);
  CHECK(ss_deflation_invariant_error(deflation, matrix, &invariant_error) == SS_OK);
  CHECK(invariant_error <= 2.0e-13);
  CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
  CHECK(diagnostics.active_count == 3U);
  CHECK(diagnostics.ill_conditioned_drop_count == 0U);
  CHECK(VecDestroy(&output) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(ss_deflation_destroy(&deflation) == SS_OK);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}

int test_signed_direction(void) {
  const double matrix_values[] = {2.0, 0.0, 0.0, -1.0};
  const double basis[] = {0.0, 1.0};
  const double rhs_values[] = {0.0, 1.0};
  const double expected[] = {0.0, -1.0};
  SsDeflation *deflation = NULL;
  Mat matrix = NULL;
  Vec rhs = NULL;
  Vec output = NULL;
  CHECK(build_deflation(matrix_values, 2, basis, basis, 1U, &deflation, &matrix) == 0);
  CHECK(create_vector(2, rhs_values, &rhs) == 0);
  CHECK(VecDuplicate(rhs, &output) == PETSC_SUCCESS);
  CHECK(ss_deflation_apply_q(deflation, rhs, output) == SS_OK);
  CHECK(check_vector(output, 2, expected, 2.0e-14) == 0);
  CHECK(VecDestroy(&output) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(ss_deflation_destroy(&deflation) == SS_OK);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}

int test_oblique(void) {
  const double matrix_values[] = {3.0, 2.0, 0.0, 1.0};
  const double left[] = {1.0, -2.0};
  const double right[] = {1.0, 0.0};
  const double rhs_values[] = {1.0, 2.0};
  const double expected[] = {-1.0, 0.0};
  SsDeflation *deflation = NULL;
  Mat matrix = NULL;
  Vec rhs = NULL;
  Vec output = NULL;
  PetscReal error = 0.0;
  CHECK(build_deflation(matrix_values, 2, left, right, 1U, &deflation, &matrix) == 0);
  CHECK(create_vector(2, rhs_values, &rhs) == 0);
  CHECK(VecDuplicate(rhs, &output) == PETSC_SUCCESS);
  CHECK(ss_deflation_apply_q(deflation, rhs, output) == SS_OK);
  CHECK(check_vector(output, 2, expected, 2.0e-14) == 0);
  CHECK(ss_deflation_invariant_error(deflation, matrix, &error) == SS_OK);
  CHECK(error <= 2.0e-14);
  CHECK(VecDestroy(&output) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(ss_deflation_destroy(&deflation) == SS_OK);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}

int test_deflated_stopping_threshold(void) {
  const double matrix_values[] = {
      1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 5.0,
  };
  const double left[] = {1.0, 10.0, 0.0};
  const double right[] = {1.0, 0.0, 0.0};
  const double rhs_values[] = {0.0, 1.0, 1.0};
  const double zero[] = {0.0, 0.0, 0.0};
  const double expected[] = {0.0, 0.5, 0.2};
  SsDeflation *deflation = NULL;
  SsLinearSolver *solver = NULL;
  SsLinearProblem problem = {0};
  SsLinearSetupRecord setup = {0};
  SsLinearSolveRecord solve = {0};
  Mat matrix = NULL;
  Vec rhs = NULL;
  Vec solution = NULL;

  CHECK(build_deflation(matrix_values, 3, left, right, 1U, &deflation, &matrix) == 0);
  CHECK(create_vector(3, rhs_values, &rhs) == 0);
  CHECK(create_vector(3, zero, &solution) == 0);
  problem.operator_handle = matrix;
  problem.properties = general_properties();
  problem.policy_chain.count = 1U;
  problem.policy_chain.policies[0] =
      general_policy("deflated-threshold", "lindefltol_");
  problem.policy_chain.policies[0].rtol = 0.1;
  problem.policy_chain.policies[0].atol = 1.0e-14;
  problem.policy_chain.policies[0].maximum_iterations = 4;
  problem.policy_chain.policies[0].restart = 3;
  problem.deflation = deflation;
  problem.structure_generation = 301U;
  problem.value_generation = 302U;
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
  CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED);
  CHECK(solve.reason > 0);
  CHECK(solve.iterations >= 2);
  CHECK(solve.true_residual_norm <= solve.acceptance_threshold);
  CHECK(solve.stopping_condition == SS_LINEAR_STOP_ORIGINAL_RHS_TRUE_RESIDUAL);
  CHECK_NEAR(solve.stopping_rhs_norm, solve.rhs_norm, 1.0e-14);
  CHECK(check_vector(solution, 3, expected, 2.0e-12) == 0);
  CHECK(ss_linear_destroy(&solver) == SS_OK);
  CHECK(VecSet(solution, 0.0) == PETSC_SUCCESS);
  problem.policy_chain.policies[0].stopping_condition =
      SS_LINEAR_STOP_PROJECTED_RHS_PETSC;
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
  CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED && solve.reason > 0);
  CHECK(solve.stopping_condition == SS_LINEAR_STOP_PROJECTED_RHS_PETSC);
  CHECK(solve.stopping_rhs_norm > 5.0 * solve.rhs_norm);
  CHECK_NEAR(solve.acceptance_threshold,
             problem.policy_chain.policies[0].rtol * solve.rhs_norm, 1.0e-12);
  CHECK(solve.true_residual_norm <= solve.acceptance_threshold);
  CHECK(ss_linear_destroy(&solver) == SS_OK);
  CHECK(VecDestroy(&solution) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(ss_deflation_destroy(&deflation) == SS_OK);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}

int test_deflated_pmg_projection(void) {
  static const double matrix_values[] = {
      4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0,
  };
  static const double basis[] = {1.0, 0.0, 0.0};
  static const double rhs_values[] = {1.0, 2.0, 3.0};
  static const double zero[] = {0.0, 0.0, 0.0};
  static const uint32_t orders[] = {1U};
  SsPmgLevelPolicy level = {0};
  SsPmgSpecification specification = {0};
  SsDeflationDiagnostics before = {0};
  SsDeflationDiagnostics after = {0};
  SsDeflation *deflation = NULL;
  SsPmg *pmg = NULL;
  SsPmgDiagnostics pmg_before = {0};
  SsPmgDiagnostics pmg_after = {0};
  SsLinearSolver *solver = NULL;
  SsLinearProblem problem = {0};
  SsLinearSetupRecord setup = {0};
  SsLinearSolveRecord solve = {0};
  Mat matrix = NULL;
  Vec rhs = NULL;
  Vec solution = NULL;
  uint64_t projected_apply_delta;
  uint64_t q_apply_delta;

  CHECK(build_deflation(matrix_values, 3, basis, basis, 1U, &deflation, &matrix) == 0);
  (void)snprintf(level.options_prefix, sizeof(level.options_prefix), "%s",
                 "defpmg_p1_");
  level.method = SS_PMG_LEVEL_PREONLY_LU;
  level.active_rank_count = 1U;
  level.maximum_iterations = 1;
  level.rtol = 1.0e-13;
  level.atol = 1.0e-15;
  specification.fine_order = 1U;
  specification.level_count = 1U;
  specification.orders = orders;
  specification.level_policies = &level;
  specification.transfer_generation = 1U;
  specification.numeric_lag_interval = 1U;
  specification.memory_cap_bytes = 16U * 1024U * 1024U;
  CHECK(ss_pmg_create(PETSC_COMM_SELF, &specification, &pmg) == SS_OK);
  CHECK(create_vector(3, rhs_values, &rhs) == 0);
  CHECK(create_vector(3, zero, &solution) == 0);
  problem.operator_handle = matrix;
  problem.properties = general_properties();
  problem.policy_chain.count = 1U;
  problem.policy_chain.policies[0] =
      general_policy("deflated-pmg-projection", "lindefpmg_");
  problem.policy_chain.policies[0].pc = SS_LINEAR_PC_PMG;
  problem.policy_chain.policies[0].restart = 3;
  problem.policy_chain.policies[0].maximum_iterations = 6;
  problem.deflation = deflation;
  problem.pmg = pmg;
  problem.structure_generation = 401U;
  problem.value_generation = 402U;
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(setup.pmg_selected != 0);
  CHECK(setup.deflation_projected_pc == 0);
  CHECK(ss_deflation_diagnostics(deflation, &before) == SS_OK);
  CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
  CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED);
  CHECK(solve.reason > 0);
  CHECK(solve.true_residual_norm <= solve.acceptance_threshold);
  CHECK(ss_deflation_diagnostics(deflation, &after) == SS_OK);
  projected_apply_delta = after.projected_apply_count - before.projected_apply_count;
  q_apply_delta = after.q_apply_count - before.q_apply_count;
  /*
   * PMG supplies M and the fused outer shell applies A P_R from cached A Z.
   * This is the same right-projected system without a second distributed A
   * action on every PMG correction.
   */
  CHECK(projected_apply_delta > 0U);
  CHECK(q_apply_delta > 3U);
  CHECK(ss_linear_destroy(&solver) == SS_OK);
  CHECK(ss_deflation_destroy(&deflation) == SS_OK);

  /* The requested ||B r||_2/||b||_2 test can already pass at a zero guess
   * when B has inverse-operator scaling.  The solve must still perform one
   * useful FCG correction, without adding a diagnostic PC application. */
  CHECK(VecSet(solution, 0.0) == PETSC_SUCCESS);
  problem.deflation = NULL;
  problem.policy_chain.policies[0] =
      general_policy("zero-guess-fcg-pmg", "linfcgzero_");
  problem.policy_chain.policies[0].operator_class = SS_LINEAR_OPERATOR_SPD_VERIFIED;
  problem.policy_chain.policies[0].method = SS_LINEAR_METHOD_FCG;
  problem.policy_chain.policies[0].pc = SS_LINEAR_PC_PMG;
  problem.policy_chain.policies[0].side = SS_LINEAR_SIDE_LEFT;
  problem.policy_chain.policies[0].stopping_condition =
      SS_LINEAR_STOP_PRECONDITIONED_RESIDUAL_ORIGINAL_RHS;
  problem.policy_chain.policies[0].rtol = 0.9;
  problem.policy_chain.policies[0].restart = 3;
  problem.policy_chain.policies[0].maximum_iterations = 6;
  problem.properties = general_properties();
  problem.properties.measured_symmetric = 1;
  problem.properties.positive_energy = 1;
  problem.properties.pc_symmetric = 1;
  problem.properties.pc_positive = 1;
  problem.properties.pc_variable = 1;
  problem.krylov_capture = SS_LINEAR_KRYLOV_CAPTURE_NONE;
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(ss_pmg_diagnostics(pmg, &pmg_before) == SS_OK);
  CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
  CHECK(ss_pmg_diagnostics(pmg, &pmg_after) == SS_OK);
  CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED);
  CHECK(solve.reason == KSP_CONVERGED_ATOL);
  CHECK(solve.iterations == 1);
  CHECK(pmg_after.preconditioner_apply_calls - pmg_before.preconditioner_apply_calls ==
        2U);
  CHECK(ss_linear_destroy(&solver) == SS_OK);

  {
    SsDeflationPolicy aorth_policy = deflation_policy(3U);
    SsDeflationAdmissionRecord admission = {0};
    Vec aorth_basis = NULL;
    aorth_policy.norm_tolerance = 1.0e-3;
    aorth_policy.projector = SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL;
    CHECK(ss_deflation_create(PETSC_COMM_SELF, &aorth_policy, &deflation) == SS_OK);
    CHECK(ss_deflation_begin(deflation, 403U) == SS_OK);
    CHECK(create_vector(3, basis, &aorth_basis) == 0);
    CHECK(ss_deflation_add_candidate(deflation, aorth_basis, aorth_basis,
                                     "legacy-aorth", &admission) == SS_OK);
    CHECK(admission.admitted != 0);
    CHECK(VecDestroy(&aorth_basis) == PETSC_SUCCESS);
    CHECK(ss_deflation_refresh(deflation, matrix, 403U) == SS_OK);
  }
  CHECK(VecSet(solution, 0.0) == PETSC_SUCCESS);
  problem.policy_chain.policies[0] =
      general_policy("legacy-aorth-projected-pmg", "linaorthpmg_");
  problem.policy_chain.policies[0].pc = SS_LINEAR_PC_PMG;
  problem.policy_chain.policies[0].restart = 3;
  problem.policy_chain.policies[0].maximum_iterations = 6;
  problem.policy_chain.policies[0].stopping_condition =
      SS_LINEAR_STOP_PROJECTED_RHS_PETSC;
  problem.deflation = deflation;
  problem.value_generation = 403U;
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(setup.pmg_selected != 0);
  CHECK(setup.deflation_projected_pc != 0);
  CHECK(setup.effective_initial_guess_nonzero == PETSC_TRUE);
  CHECK(ss_deflation_diagnostics(deflation, &before) == SS_OK);
  CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
  CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED);
  CHECK(solve.reason > 0);
  CHECK(solve.stopping_condition == SS_LINEAR_STOP_PROJECTED_RHS_PETSC);
  CHECK(ss_deflation_diagnostics(deflation, &after) == SS_OK);
  projected_apply_delta = after.projected_apply_count - before.projected_apply_count;
  q_apply_delta = after.q_apply_count - before.q_apply_count;
  /* One coarse initial guess per solve. Each Krylov PC application reuses the
   * cached transposed images and the existing signed coarse factorization. */
  CHECK(q_apply_delta == 1U);
  CHECK(projected_apply_delta > 0U);
  CHECK(ss_linear_destroy(&solver) == SS_OK);

  /* FCG's preconditioned norm is the Z=B_k*r_k vector already formed for the
   * next correction.  Normalize it against the untouched RHS through PETSc's
   * absolute threshold, and retain only the Z vectors actually used by FCG. */
  CHECK(VecSet(solution, 0.0) == PETSC_SUCCESS);
  problem.policy_chain.policies[0] =
      general_policy("legacy-aorth-fcg-pmg", "linaorthfcg_");
  problem.policy_chain.policies[0].operator_class = SS_LINEAR_OPERATOR_SPD_VERIFIED;
  problem.policy_chain.policies[0].method = SS_LINEAR_METHOD_FCG;
  problem.policy_chain.policies[0].pc = SS_LINEAR_PC_PMG;
  problem.policy_chain.policies[0].side = SS_LINEAR_SIDE_LEFT;
  problem.policy_chain.policies[0].stopping_condition =
      SS_LINEAR_STOP_PRECONDITIONED_RESIDUAL_ORIGINAL_RHS;
  problem.policy_chain.policies[0].rtol = 0.1;
  problem.policy_chain.policies[0].restart = 3;
  problem.policy_chain.policies[0].maximum_iterations = 6;
  problem.properties = general_properties();
  problem.properties.measured_symmetric = 1;
  problem.properties.positive_energy = 1;
  problem.properties.pc_symmetric = 1;
  problem.properties.pc_positive = 1;
  problem.properties.pc_variable = 1;
  problem.krylov_capture = SS_LINEAR_KRYLOV_CAPTURE_FULL_CORRECTION_SPACE;
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(strcmp(setup.effective_ksp, KSPFCG) == 0);
  CHECK(setup.effective_side == SS_LINEAR_SIDE_LEFT);
  CHECK(setup.effective_norm_type == KSP_NORM_PRECONDITIONED);
  CHECK(setup.effective_restart == 3);
  CHECK(strcmp(setup.effective_orthogonalization, "fcg_notay_a_orthogonalization") ==
        0);
  CHECK(ss_pmg_diagnostics(pmg, &pmg_before) == SS_OK);
  CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
  CHECK(ss_pmg_diagnostics(pmg, &pmg_after) == SS_OK);
  CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED);
  CHECK(solve.reason == KSP_CONVERGED_ATOL);
  CHECK(solve.iterations > 0);
  CHECK(solve.stopping_condition ==
        SS_LINEAR_STOP_PRECONDITIONED_RESIDUAL_ORIGINAL_RHS);
  CHECK_NEAR(solve.stopping_rhs_norm, solve.rhs_norm, 1.0e-14);
  CHECK_NEAR(solve.acceptance_threshold, 0.1 * solve.rhs_norm, 1.0e-14);
  CHECK(solve.reported_norm <= solve.acceptance_threshold);
  CHECK(pmg_after.preconditioner_apply_calls - pmg_before.preconditioner_apply_calls ==
        (uint64_t)solve.iterations + 1U);
  CHECK(solve.krylov_vectors_captured == (size_t)solve.iterations);
  CHECK(ss_linear_destroy(&solver) == SS_OK);

  /* PETSc FCG maintains the ordinary recursive residual natively.  The same
   * original-RHS true-residual contract used by FGMRES therefore selects the
   * unpreconditioned norm even though flexible CG remains left-preconditioned.
   * The final explicit residual remains the independent acceptance check. */
  CHECK(VecSet(solution, 0.0) == PETSC_SUCCESS);
  problem.policy_chain.policies[0].stopping_condition =
      SS_LINEAR_STOP_ORIGINAL_RHS_TRUE_RESIDUAL;
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(strcmp(setup.effective_ksp, KSPFCG) == 0);
  CHECK(setup.effective_side == SS_LINEAR_SIDE_LEFT);
  CHECK(setup.effective_norm_type == KSP_NORM_UNPRECONDITIONED);
  CHECK(ss_pmg_diagnostics(pmg, &pmg_before) == SS_OK);
  CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
  CHECK(ss_pmg_diagnostics(pmg, &pmg_after) == SS_OK);
  CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED);
  CHECK(solve.reason > 0);
  CHECK(solve.iterations > 0);
  CHECK(solve.stopping_condition == SS_LINEAR_STOP_ORIGINAL_RHS_TRUE_RESIDUAL);
  CHECK_NEAR(solve.stopping_rhs_norm, solve.rhs_norm, 1.0e-14);
  CHECK_NEAR(solve.acceptance_threshold, 0.1 * solve.rhs_norm, 1.0e-14);
  CHECK(solve.reported_norm <= solve.acceptance_threshold);
  CHECK(solve.true_residual_norm <= solve.acceptance_threshold);
  CHECK(pmg_after.preconditioner_apply_calls - pmg_before.preconditioner_apply_calls ==
        (uint64_t)solve.iterations);
  CHECK(solve.krylov_vectors_captured == (size_t)solve.iterations);
  CHECK(ss_linear_destroy(&solver) == SS_OK);

  /* FCG minimizes in the preconditioned energy geometry.  PETSc's natural
   * norm is sqrt(r_k^T B_k r_k); its default relative test scales against the
   * same norm at iteration zero.  This uses the already formed Z=B_k*r_k,
   * incurs no diagnostic PMG application, and retains only correction vectors
   * that actually contributed to the returned iterate. */
  CHECK(VecSet(solution, 0.0) == PETSC_SUCCESS);
  problem.policy_chain.policies[0].stopping_condition =
      SS_LINEAR_STOP_NATURAL_RESIDUAL_INITIAL;
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(strcmp(setup.effective_ksp, KSPFCG) == 0);
  CHECK(setup.effective_side == SS_LINEAR_SIDE_LEFT);
  CHECK(setup.effective_norm_type == KSP_NORM_NATURAL);
  CHECK(ss_pmg_diagnostics(pmg, &pmg_before) == SS_OK);
  CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
  CHECK(ss_pmg_diagnostics(pmg, &pmg_after) == SS_OK);
  CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED);
  CHECK(solve.reason > 0);
  CHECK(solve.iterations > 0);
  CHECK(solve.stopping_condition == SS_LINEAR_STOP_NATURAL_RESIDUAL_INITIAL);
  CHECK(isfinite((double)solve.stopping_rhs_norm));
  CHECK(solve.stopping_rhs_norm > 0.0);
  CHECK_NEAR(solve.acceptance_threshold, 0.1 * solve.rhs_norm, 1.0e-14);
  CHECK(solve.reported_norm <= 0.1 * solve.stopping_rhs_norm);
  CHECK(solve.true_residual_norm <= solve.acceptance_threshold);
  CHECK(pmg_after.preconditioner_apply_calls - pmg_before.preconditioner_apply_calls ==
        (uint64_t)solve.iterations + 1U);
  CHECK(solve.krylov_vectors_captured == (size_t)solve.iterations);
  CHECK(ss_linear_destroy(&solver) == SS_OK);

  problem.properties = general_properties();
  problem.policy_chain.policies[0] =
      general_policy("legacy-aorth-projected-pmg", "linaorthpmg_");
  problem.policy_chain.policies[0].pc = SS_LINEAR_PC_PMG;
  problem.policy_chain.policies[0].restart = 3;
  problem.policy_chain.policies[0].maximum_iterations = 6;
  problem.policy_chain.policies[0].stopping_condition =
      SS_LINEAR_STOP_PROJECTED_RHS_PETSC;
  problem.krylov_capture = SS_LINEAR_KRYLOV_CAPTURE_NONE;

  /* A previous solve is useful only in the A-orthogonal complement.  Preserve
   * an exact prior solution through the coarse-guess refresh and require the
   * screened policy to accept it without Krylov work. */
  problem.policy_chain.policies[0].initial_guess_policy =
      SS_LINEAR_INITIAL_GUESS_PREVIOUS_COMPLEMENT_IF_BETTER;
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
  CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED && solve.reason > 0);
  CHECK(solve.input_guess_tested != 0 && solve.input_guess_accepted != 0);
  CHECK(solve.initial_residual_norm < solve.coarse_initial_residual_norm);
  CHECK(solve.iterations == 0);
  CHECK(ss_linear_destroy(&solver) == SS_OK);

  /* A stale direction can be actively harmful.  The screened policy must
   * reject it and retain the same projected stopping scale as coarse-only. */
  CHECK(VecSet(solution, 100.0) == PETSC_SUCCESS);
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
  CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED && solve.reason > 0);
  CHECK(solve.input_guess_tested != 0 && solve.input_guess_accepted == 0);
  CHECK_NEAR(solve.initial_residual_norm, solve.coarse_initial_residual_norm, 1.0e-12);
  CHECK_NEAR(solve.stopping_rhs_norm, solve.coarse_initial_residual_norm, 1.0e-12);
  CHECK(ss_linear_destroy(&solver) == SS_OK);

  /* The previous-complement screening is shared by both stopping contracts,
   * but only projected-RHS stopping may inherit the coarse projected scale.
   * Original-RHS true-residual stopping must retain ||b|| even when the
   * projected coarse residual is the larger norm. */
  CHECK(VecSet(solution, 100.0) == PETSC_SUCCESS);
  problem.policy_chain.policies[0].stopping_condition =
      SS_LINEAR_STOP_ORIGINAL_RHS_TRUE_RESIDUAL;
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
  CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED && solve.reason > 0);
  CHECK(solve.input_guess_tested != 0 && solve.input_guess_accepted == 0);
  CHECK_NEAR(solve.stopping_rhs_norm, solve.rhs_norm, 1.0e-12);
  CHECK_NEAR(solve.acceptance_threshold,
             problem.policy_chain.policies[0].rtol * solve.rhs_norm, 1.0e-12);
  CHECK(solve.true_residual_norm <= solve.acceptance_threshold);
  CHECK(ss_linear_destroy(&solver) == SS_OK);

  /* Capture the exact flexible-FGMRES correction space at the projected PMG
   * shell.  Promoting that complete space must make a repeated right-hand
   * side a coarse-space solve with no further Krylov iteration. */
  {
    SsLinearKrylovPromotionRecord promotion = {0};
    CHECK(VecSet(solution, 0.0) == PETSC_SUCCESS);
    problem.policy_chain.policies[0].initial_guess_policy =
        SS_LINEAR_INITIAL_GUESS_COARSE_ONLY;
    problem.krylov_capture = SS_LINEAR_KRYLOV_CAPTURE_FULL_CORRECTION_SPACE;
    CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
    CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
    CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
    CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED && solve.iterations > 0);
    CHECK(solve.krylov_vectors_captured == (size_t)solve.iterations);
    CHECK(ss_linear_promote_captured_krylov(solver, "test:fgmres-correction", &solve,
                                            &promotion) == SS_OK);
    CHECK(promotion.captured_count == (size_t)solve.iterations);
    CHECK(promotion.promoted_count == promotion.captured_count);
    CHECK(solve.krylov_vectors_promoted == promotion.promoted_count);
    CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
    CHECK(VecSet(solution, 0.0) == PETSC_SUCCESS);
    CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
    CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED);
    CHECK(solve.iterations == 0);
    CHECK(solve.krylov_vectors_captured == 0U);
    CHECK(ss_linear_promote_captured_krylov(solver, "test:empty-correction", &solve,
                                            &promotion) == SS_OK);
    CHECK(promotion.captured_count == 0U);
    CHECK(ss_linear_destroy(&solver) == SS_OK);
  }

  CHECK(VecDestroy(&solution) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(ss_pmg_destroy(&pmg) == SS_OK);
  CHECK(ss_deflation_destroy(&deflation) == SS_OK);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}

int test_aorth_true_residual_reused_guess(void) {
  static const double matrix_values[] = {1.0, 0.0, 0.0, 10000.0};
  static const double basis_values[] = {1.0, 0.01};
  static const double rhs_values[] = {1.0, 0.0};
  static const double previous_complement[] = {0.495, -0.00495};
  static const double expected[] = {1.0, 0.0};
  static const uint32_t orders[] = {1U};
  SsDeflationPolicy deflation_policy_value = deflation_policy(3U);
  SsDeflationAdmissionRecord admission = {0};
  SsPmgLevelPolicy level = {0};
  SsPmgSpecification specification = {0};
  SsDeflation *deflation = NULL;
  SsPmg *pmg = NULL;
  SsLinearSolver *solver = NULL;
  SsLinearProblem problem = {0};
  SsLinearSetupRecord setup = {0};
  SsLinearSolveRecord solve = {0};
  Mat matrix = NULL;
  Vec basis = NULL;
  Vec rhs = NULL;
  Vec solution = NULL;

  deflation_policy_value.projector = SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL;
  CHECK(create_dense(2, matrix_values, &matrix) == 0);
  CHECK(create_vector(2, basis_values, &basis) == 0);
  CHECK(create_vector(2, rhs_values, &rhs) == 0);
  CHECK(create_vector(2, previous_complement, &solution) == 0);
  CHECK(ss_deflation_create(PETSC_COMM_SELF, &deflation_policy_value, &deflation) ==
        SS_OK);
  CHECK(ss_deflation_begin(deflation, 501U) == SS_OK);
  CHECK(ss_deflation_add_candidate(deflation, basis, basis, "aorth-amplified",
                                   &admission) == SS_OK);
  CHECK(admission.admitted != 0);
  CHECK(ss_deflation_refresh(deflation, matrix, 501U) == SS_OK);

  (void)snprintf(level.options_prefix, sizeof(level.options_prefix), "%s",
                 "aorthtrue_p1_");
  level.method = SS_PMG_LEVEL_PREONLY_LU;
  level.active_rank_count = 1U;
  level.maximum_iterations = 1;
  level.rtol = 1.0e-13;
  level.atol = 1.0e-15;
  specification.fine_order = 1U;
  specification.level_count = 1U;
  specification.orders = orders;
  specification.level_policies = &level;
  specification.transfer_generation = 1U;
  specification.numeric_lag_interval = 1U;
  specification.memory_cap_bytes = 16U * 1024U * 1024U;
  CHECK(ss_pmg_create(PETSC_COMM_SELF, &specification, &pmg) == SS_OK);

  problem.operator_handle = matrix;
  problem.properties = general_properties();
  problem.policy_chain.count = 1U;
  problem.policy_chain.policies[0] =
      general_policy("aorth-true-reused", "linaorthtrue_");
  problem.policy_chain.policies[0].pc = SS_LINEAR_PC_PMG;
  problem.policy_chain.policies[0].stopping_condition =
      SS_LINEAR_STOP_ORIGINAL_RHS_TRUE_RESIDUAL;
  problem.policy_chain.policies[0].initial_guess_policy =
      SS_LINEAR_INITIAL_GUESS_PREVIOUS_COMPLEMENT_IF_BETTER;
  problem.policy_chain.policies[0].rtol = 0.1;
  problem.policy_chain.policies[0].restart = 2;
  problem.policy_chain.policies[0].maximum_iterations = 4;
  problem.deflation = deflation;
  problem.pmg = pmg;
  problem.structure_generation = 501U;
  problem.value_generation = 501U;
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
  CHECK(solve.coarse_initial_residual_norm > 40.0 * solve.rhs_norm);
  CHECK(solve.input_guess_tested != 0 && solve.input_guess_accepted != 0);
  CHECK(solve.candidate_initial_residual_norm < solve.coarse_initial_residual_norm);
  CHECK(solve.candidate_initial_residual_norm > solve.acceptance_threshold);
  CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED && solve.reason > 0);
  CHECK(solve.iterations > 0);
  CHECK_NEAR(solve.stopping_rhs_norm, solve.rhs_norm, 1.0e-14);
  CHECK(solve.true_residual_norm <= solve.acceptance_threshold);
  CHECK(check_vector(solution, 2, expected, 2.0e-12) == 0);

  CHECK(ss_linear_destroy(&solver) == SS_OK);
  CHECK(ss_pmg_destroy(&pmg) == SS_OK);
  CHECK(ss_deflation_destroy(&deflation) == SS_OK);
  CHECK(VecDestroy(&solution) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(VecDestroy(&basis) == PETSC_SUCCESS);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}

int test_dependent(void) {
  const double matrix_values[] = {1.0, 0.0, 0.0, 1.0e-16};
  const double e1[] = {1.0, 0.0};
  const double e2[] = {0.0, 1.0};
  const double zero_values[] = {0.0, 0.0};
  const double nan_values[] = {NAN, 0.0};
  SsDeflationPolicy policy = deflation_policy(4U);
  SsDeflation *deflation = NULL;
  SsDeflationAdmissionRecord record = {0};
  SsDeflationDiagnostics diagnostics = {0};
  Mat matrix = NULL;
  Vec first = NULL;
  Vec second = NULL;
  Vec zero = NULL;
  Vec nonfinite = NULL;

  CHECK(create_dense(2, matrix_values, &matrix) == 0);
  CHECK(create_vector(2, e1, &first) == 0);
  CHECK(create_vector(2, e2, &second) == 0);
  CHECK(create_vector(2, zero_values, &zero) == 0);
  CHECK(create_vector(2, nan_values, &nonfinite) == 0);
  CHECK(ss_deflation_create(PETSC_COMM_SELF, &policy, &deflation) == SS_OK);
  CHECK(ss_deflation_begin(deflation, 1U) == SS_OK);
  CHECK(ss_deflation_add_candidate(deflation, first, first, "first", &record) == SS_OK);
  CHECK(record.admitted != 0);
  CHECK(ss_deflation_add_candidate(deflation, first, first, "duplicate", &record) ==
        SS_OK);
  CHECK(record.admitted == 0);
  CHECK(record.drop_reason == SS_DEFLATION_DROP_DEPENDENT);
  CHECK(ss_deflation_add_candidate(deflation, zero, second, "negligible", &record) ==
        SS_OK);
  CHECK(record.admitted == 0);
  CHECK(record.drop_reason == SS_DEFLATION_DROP_NEGLIGIBLE);
  CHECK(ss_deflation_add_candidate(deflation, nonfinite, second, "nan", &record) ==
        SS_OK);
  CHECK(record.admitted == 0);
  CHECK(record.drop_reason == SS_DEFLATION_DROP_NONFINITE);
  CHECK(ss_deflation_add_candidate(deflation, second, second, "ill-conditioned",
                                   &record) == SS_OK);
  CHECK(record.admitted != 0);
  CHECK(ss_deflation_refresh(deflation, matrix, 2U) == SS_OK);
  CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
  CHECK(diagnostics.provisional_count == 1U);
  CHECK(diagnostics.active_count == 1U);
  CHECK(diagnostics.ill_conditioned_drop_count == 1U);
  CHECK(diagnostics.last_drop_reason == SS_DEFLATION_DROP_ILL_CONDITIONED);
  CHECK(diagnostics.invariants_validated != 0);
  CHECK(VecDestroy(&nonfinite) == PETSC_SUCCESS);
  CHECK(VecDestroy(&zero) == PETSC_SUCCESS);
  CHECK(VecDestroy(&second) == PETSC_SUCCESS);
  CHECK(VecDestroy(&first) == PETSC_SUCCESS);
  CHECK(ss_deflation_destroy(&deflation) == SS_OK);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}

int test_krylov_compression(void) {
  const double matrix_values[] = {
      1.0e-4, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 4.0,
  };
  const double e1_values[] = {1.0, 0.0, 0.0, 0.0};
  const double e2_values[] = {0.0, 1.0, 0.0, 0.0};
  const double e3_values[] = {0.0, 0.0, 1.0, 0.0};
  const double e4_values[] = {0.0, 0.0, 0.0, 1.0};
  SsDeflationPolicy policy = deflation_policy(8U);
  SsDeflation *deflation = NULL;
  SsDeflationAdmissionRecord admission = {0};
  SsDeflationDiagnostics diagnostics = {0};
  Mat matrix = NULL;
  Vec vectors[4] = {NULL, NULL, NULL, NULL};
  Vec retained_left = NULL;
  Vec retained = NULL;
  PetscScalar alignment = 0.0;
  PetscReal retained_norm = 0.0;
  const char *provenance = NULL;
  size_t index;

  CHECK(create_dense(4, matrix_values, &matrix) == 0);
  CHECK(create_vector(4, e1_values, &vectors[0]) == 0);
  CHECK(create_vector(4, e2_values, &vectors[1]) == 0);
  CHECK(create_vector(4, e3_values, &vectors[2]) == 0);
  CHECK(create_vector(4, e4_values, &vectors[3]) == 0);
  policy.projector = SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL;
  policy.compression = SS_DEFLATION_COMPRESSION_SNAPSHOT_POD;
  policy.compression_dimension = 1U;
  CHECK(ss_deflation_create(PETSC_COMM_SELF, &policy, &deflation) == SS_OK);
  CHECK(ss_deflation_begin(deflation, 71U) == SS_OK);
  CHECK(ss_deflation_add_candidate(deflation, vectors[0], vectors[0], "krylov:e1",
                                   &admission) == SS_OK);
  CHECK(ss_deflation_add_candidate(deflation, vectors[1], vectors[1], "krylov:e2a",
                                   &admission) == SS_OK);
  CHECK(ss_deflation_add_candidate(deflation, vectors[1], vectors[1], "krylov:e2b",
                                   &admission) == SS_OK);
  CHECK(ss_deflation_add_candidate(deflation, vectors[2], vectors[2], "krylov:e3",
                                   &admission) == SS_OK);
  CHECK(ss_deflation_add_candidate(deflation, vectors[3], vectors[3],
                                   "nls:accepted-state", &admission) == SS_OK);
  CHECK(ss_deflation_refresh(deflation, matrix, 71U) == SS_OK);
  CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
  CHECK(diagnostics.compression_count == 1U);
  CHECK(diagnostics.last_compression_input_count == 4U);
  CHECK(diagnostics.last_compression_protected_count == 1U);
  CHECK(diagnostics.last_compression_retained_count == 1U);
  CHECK_NEAR(diagnostics.last_compression_information_fraction, 0.5, 1.0e-12);
  CHECK(diagnostics.active_count == 2U);
  CHECK(ss_deflation_active_entry(deflation, 0U, &retained_left, &retained,
                                  &provenance) == SS_OK);
  CHECK(strcmp(provenance, "compress:snapshot-pod") == 0);
  CHECK(VecDot(retained, vectors[1], &alignment) == PETSC_SUCCESS);
  CHECK(VecNorm(retained, NORM_2, &retained_norm) == PETSC_SUCCESS);
  CHECK_NEAR(PetscAbsScalar(alignment) / retained_norm, 1.0, 1.0e-12);
  CHECK(ss_deflation_active_entry(deflation, 1U, &retained_left, &retained,
                                  &provenance) == SS_OK);
  CHECK(strcmp(provenance, "nls:accepted-state") == 0);
  CHECK(ss_deflation_add_candidate(deflation, vectors[0], vectors[0],
                                   "krylov:incremental", &admission) == SS_OK);
  CHECK(ss_deflation_refresh(deflation, matrix, 72U) == SS_OK);
  CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
  CHECK(diagnostics.compression_count == 2U);
  CHECK_NEAR(diagnostics.last_compression_information_fraction, 2.0 / 3.0, 1.0e-12);
  CHECK(ss_deflation_active_entry(deflation, 0U, &retained_left, &retained,
                                  &provenance) == SS_OK);
  CHECK(VecDot(retained, vectors[1], &alignment) == PETSC_SUCCESS);
  CHECK(VecNorm(retained, NORM_2, &retained_norm) == PETSC_SUCCESS);
  CHECK_NEAR(PetscAbsScalar(alignment) / retained_norm, 1.0, 1.0e-12);
  CHECK(ss_deflation_rollback(deflation) == SS_OK);
  CHECK(ss_deflation_destroy(&deflation) == SS_OK);

  /* Total POD admits accepted continuation states into the same rank-k
   * approximation instead of protecting them outside k. */
  policy.compression = SS_DEFLATION_COMPRESSION_SNAPSHOT_POD_TOTAL;
  policy.compression_dimension = 2U;
  CHECK(ss_deflation_create(PETSC_COMM_SELF, &policy, &deflation) == SS_OK);
  CHECK(ss_deflation_begin(deflation, 76U) == SS_OK);
  CHECK(ss_deflation_add_candidate(deflation, vectors[0], vectors[0], "system:e1a",
                                   &admission) == SS_OK);
  CHECK(ss_deflation_add_candidate(deflation, vectors[0], vectors[0], "system:e1b",
                                   &admission) == SS_OK);
  CHECK(ss_deflation_add_candidate(deflation, vectors[1], vectors[1], "system:e2",
                                   &admission) == SS_OK);
  CHECK(ss_deflation_add_candidate(deflation, vectors[3], vectors[3],
                                   "nls:accepted-state", &admission) == SS_OK);
  CHECK(ss_deflation_refresh(deflation, matrix, 76U) == SS_OK);
  CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
  CHECK(diagnostics.last_compression_input_count == 4U);
  CHECK(diagnostics.last_compression_protected_count == 0U);
  CHECK(diagnostics.last_compression_retained_count == 2U);
  CHECK(diagnostics.active_count == 2U);
  for (index = 0U; index < diagnostics.active_count; ++index) {
    CHECK(ss_deflation_active_entry(deflation, index, &retained_left, &retained,
                                    &provenance) == SS_OK);
    CHECK(strcmp(provenance, "compress:snapshot-pod-total") == 0);
  }
  CHECK(ss_deflation_rollback(deflation) == SS_OK);
  CHECK(ss_deflation_destroy(&deflation) == SS_OK);

  /* The rolling policy keeps every accepted state and only the latest k
   * completed-system corrections, in chronological order. */
  policy.compression = SS_DEFLATION_COMPRESSION_SYSTEM_SOLUTION_WINDOW;
  policy.compression_dimension = 2U;
  CHECK(ss_deflation_create(PETSC_COMM_SELF, &policy, &deflation) == SS_OK);
  CHECK(ss_deflation_begin(deflation, 77U) == SS_OK);
  CHECK(ss_deflation_add_candidate(deflation, vectors[3], vectors[3],
                                   "nls:accepted-state", &admission) == SS_OK);
  CHECK(ss_deflation_add_candidate(deflation, vectors[0], vectors[0], "system:e1",
                                   &admission) == SS_OK);
  CHECK(ss_deflation_add_candidate(deflation, vectors[1], vectors[1], "system:e2",
                                   &admission) == SS_OK);
  CHECK(ss_deflation_add_candidate(deflation, vectors[2], vectors[2], "system:e3",
                                   &admission) == SS_OK);
  CHECK(ss_deflation_refresh(deflation, matrix, 77U) == SS_OK);
  CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
  CHECK(diagnostics.last_compression_input_count == 3U);
  CHECK(diagnostics.last_compression_protected_count == 1U);
  CHECK(diagnostics.last_compression_retained_count == 2U);
  CHECK(diagnostics.active_count == 3U);
  CHECK(ss_deflation_active_entry(deflation, 0U, &retained_left, &retained,
                                  &provenance) == SS_OK);
  CHECK(strcmp(provenance, "system:e2") == 0);
  CHECK(ss_deflation_active_entry(deflation, 1U, &retained_left, &retained,
                                  &provenance) == SS_OK);
  CHECK(strcmp(provenance, "system:e3") == 0);
  CHECK(ss_deflation_active_entry(deflation, 2U, &retained_left, &retained,
                                  &provenance) == SS_OK);
  CHECK(strcmp(provenance, "nls:accepted-state") == 0);
  CHECK(ss_deflation_rollback(deflation) == SS_OK);
  CHECK(ss_deflation_destroy(&deflation) == SS_OK);

  /* Restoring a checkpoint is deliberately lossless.  Compression resumes
   * only after the next continuation transaction begins. */
  policy.compression = SS_DEFLATION_COMPRESSION_SNAPSHOT_POD;
  policy.compression_dimension = 1U;
  {
    const Vec restored_left[2] = {vectors[0], vectors[1]};
    const Vec restored_right[2] = {vectors[0], vectors[1]};
    CHECK(ss_deflation_create(PETSC_COMM_SELF, &policy, &deflation) == SS_OK);
    CHECK(ss_deflation_restore_persistent(deflation, restored_left, restored_right, 2U,
                                          "checkpoint-restored-basis-v1", matrix, 74U,
                                          70U) == SS_OK);
    CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
    CHECK(diagnostics.active_count == 2U);
    CHECK(diagnostics.compression_count == 0U);
    CHECK(ss_deflation_begin(deflation, 75U) == SS_OK);
    CHECK(ss_deflation_add_candidate(deflation, vectors[2], vectors[2],
                                     "krylov:after-restore", &admission) == SS_OK);
    CHECK(ss_deflation_refresh(deflation, matrix, 75U) == SS_OK);
    CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
    CHECK(diagnostics.active_count == 1U);
    CHECK(diagnostics.compression_count == 1U);
    CHECK(ss_deflation_rollback(deflation) == SS_OK);
    CHECK(ss_deflation_destroy(&deflation) == SS_OK);
  }

  policy.compression = SS_DEFLATION_COMPRESSION_OPERATOR_SVD;
  CHECK(ss_deflation_create(PETSC_COMM_SELF, &policy, &deflation) == SS_OK);
  CHECK(ss_deflation_begin(deflation, 73U) == SS_OK);
  for (index = 0U; index < 3U; ++index) {
    CHECK(ss_deflation_add_candidate(deflation, vectors[index], vectors[index],
                                     "krylov:operator", &admission) == SS_OK);
  }
  CHECK(ss_deflation_add_candidate(deflation, vectors[3], vectors[3],
                                   "nls:accepted-state", &admission) == SS_OK);
  CHECK(ss_deflation_refresh(deflation, matrix, 73U) == SS_OK);
  CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
  CHECK(diagnostics.compression_count == 1U);
  CHECK(diagnostics.last_compression_input_count == 3U);
  CHECK(diagnostics.last_compression_protected_count == 1U);
  CHECK(diagnostics.last_compression_retained_count == 1U);
  CHECK(diagnostics.last_compression_information_fraction > 0.999);
  CHECK(diagnostics.active_count == 2U);
  CHECK(ss_deflation_active_entry(deflation, 0U, &retained_left, &retained,
                                  &provenance) == SS_OK);
  CHECK(strcmp(provenance, "compress:operator-svd") == 0);
  CHECK(VecDot(retained, vectors[0], &alignment) == PETSC_SUCCESS);
  CHECK(VecNorm(retained, NORM_2, &retained_norm) == PETSC_SUCCESS);
  CHECK_NEAR(PetscAbsScalar(alignment) / retained_norm, 1.0, 1.0e-12);
  CHECK(ss_deflation_rollback(deflation) == SS_OK);
  CHECK(ss_deflation_destroy(&deflation) == SS_OK);

  for (index = 0U; index < 4U; ++index) {
    CHECK(VecDestroy(&vectors[index]) == PETSC_SUCCESS);
  }
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}

int test_persistent_restore_bits(void) {
  const double identity_values[] = {
      1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
  };
  const double first_values[] = {1.1, 2.2, 3.3};
  const double second_values[] = {3.7, 1.3, 4.9};
  SsDeflationPolicy policy = deflation_policy(2U);
  SsDeflation *source = NULL;
  SsDeflation *restored = NULL;
  SsDeflation *empty = NULL;
  SsDeflationAdmissionRecord record = {0};
  Mat matrix = NULL;
  Vec first = NULL;
  Vec second = NULL;
  Vec source_left[2] = {NULL};
  Vec source_right[2] = {NULL};
  PetscReal invariant_error = NAN;
  size_t index;

  CHECK(create_dense(3, identity_values, &matrix) == 0);
  CHECK(create_vector(3, first_values, &first) == 0);
  CHECK(create_vector(3, second_values, &second) == 0);
  CHECK(ss_deflation_create(PETSC_COMM_SELF, &policy, &source) == SS_OK);
  CHECK(ss_deflation_begin(source, 1U) == SS_OK);
  CHECK(ss_deflation_add_candidate(source, first, first, "accepted", &record) == SS_OK);
  CHECK(record.admitted != 0);
  CHECK(ss_deflation_add_candidate(source, second, second, "accepted", &record) ==
        SS_OK);
  CHECK(record.admitted != 0);
  CHECK(ss_deflation_refresh(source, matrix, 1U) == SS_OK);
  CHECK(ss_deflation_commit(source, 1U) == SS_OK);
  for (index = 0U; index < 2U; ++index) {
    const char *provenance = NULL;
    CHECK(ss_deflation_persistent_entry(source, index, &source_left[index],
                                        &source_right[index], &provenance) == SS_OK);
    CHECK(provenance != NULL && strcmp(provenance, "accepted") == 0);
  }

  CHECK(ss_deflation_create(PETSC_COMM_SELF, &policy, &restored) == SS_OK);
  CHECK(ss_deflation_restore_persistent(restored, source_left, source_right, 2U,
                                        "accepted", matrix, 2U, 1U) == SS_OK);
  for (index = 0U; index < 2U; ++index) {
    const char *provenance = NULL;
    Vec restored_left = NULL;
    Vec restored_right = NULL;
    CHECK(ss_deflation_persistent_entry(restored, index, &restored_left,
                                        &restored_right, &provenance) == SS_OK);
    CHECK(provenance != NULL && strcmp(provenance, "accepted") == 0);
    CHECK(check_vector_bits(restored_left, source_left[index]) == 0);
    CHECK(check_vector_bits(restored_right, source_right[index]) == 0);
  }

  CHECK(ss_deflation_create(PETSC_COMM_SELF, &policy, &empty) == SS_OK);
  CHECK(ss_deflation_restore_persistent(empty, NULL, NULL, 0U, "accepted", matrix, 3U,
                                        1U) == SS_OK);
  CHECK(ss_deflation_invariant_error(empty, matrix, &invariant_error) == SS_OK);
  CHECK_NEAR(invariant_error, 0.0, 0.0);

  CHECK(ss_deflation_destroy(&empty) == SS_OK);
  CHECK(ss_deflation_destroy(&restored) == SS_OK);
  CHECK(ss_deflation_destroy(&source) == SS_OK);
  CHECK(VecDestroy(&second) == PETSC_SUCCESS);
  CHECK(VecDestroy(&first) == PETSC_SUCCESS);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}

int test_rollback(void) {
  const double matrix_values[] = {2.0, 0.0, 0.0, 3.0};
  const double refreshed_matrix_values[] = {4.0, 0.0, 0.0, 5.0};
  const double ill_conditioned_matrix_values[] = {1.0e-16, 0.0, 0.0, 3.0};
  const double e1[] = {1.0, 0.0};
  const double e2[] = {0.0, 1.0};
  SsDeflationPolicy policy = deflation_policy(1U);
  SsDeflation *deflation = NULL;
  SsDeflationAdmissionRecord record = {0};
  SsDeflationDiagnostics diagnostics = {0};
  Mat matrix = NULL;
  Mat refreshed_matrix = NULL;
  Mat ill_conditioned_matrix = NULL;
  Vec first = NULL;
  Vec second = NULL;
  Vec output = NULL;
  char before[SS_SHA256_HEX_CAPACITY] = {0};
  char after[SS_SHA256_HEX_CAPACITY] = {0};
  char replaced[SS_SHA256_HEX_CAPACITY] = {0};
  const double expected_q[] = {0.5, 0.0};
  const double expected_refreshed_q[] = {0.25, 0.0};
  const double expected_replacement_q[] = {0.0, 1.0 / 3.0};

  CHECK(create_dense(2, matrix_values, &matrix) == 0);
  CHECK(create_vector(2, e1, &first) == 0);
  CHECK(create_vector(2, e2, &second) == 0);
  CHECK(VecDuplicate(first, &output) == PETSC_SUCCESS);
  CHECK(ss_deflation_create(PETSC_COMM_SELF, &policy, &deflation) == SS_OK);
  CHECK(ss_deflation_begin(deflation, 1U) == SS_OK);
  CHECK(ss_deflation_rollback(deflation) == SS_OK);
  CHECK(ss_deflation_begin(deflation, 2U) == SS_OK);
  CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
  CHECK(diagnostics.active_count == 0U);
  CHECK(diagnostics.invariants_validated != 0);
  CHECK(ss_deflation_commit(deflation, 9U) == SS_OK);
  CHECK(ss_deflation_begin(deflation, 3U) == SS_OK);
  CHECK(ss_deflation_add_candidate(deflation, first, first, "accepted", &record) ==
        SS_OK);
  CHECK(record.admitted != 0);
  CHECK(ss_deflation_refresh(deflation, matrix, 1U) == SS_OK);
  CHECK(ss_deflation_commit(deflation, 10U) == SS_OK);
  CHECK(ss_deflation_persistent_digest(deflation, before) == SS_OK);
  CHECK(ss_deflation_begin(deflation, 20U) == SS_OK);
  CHECK(ss_deflation_add_candidate(deflation, second, second, "discarded-provisional",
                                   &record) == SS_OK);
  CHECK(record.admitted != 0);
  CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
  CHECK(diagnostics.persistent_count == 1U && diagnostics.provisional_count == 1U);
  CHECK(ss_deflation_discard_provisional(deflation) == SS_OK);
  CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
  CHECK(diagnostics.persistent_count == 1U && diagnostics.provisional_count == 0U);
  CHECK(ss_deflation_rollback(deflation) == SS_OK);
  CHECK(ss_deflation_persistent_digest(deflation, after) == SS_OK);
  CHECK(strcmp(before, after) == 0);
  CHECK(create_dense(2, refreshed_matrix_values, &refreshed_matrix) == 0);
  CHECK(ss_deflation_refresh(deflation, refreshed_matrix, 2U) == SS_OK);
  CHECK(ss_deflation_apply_q(deflation, first, output) == SS_OK);
  CHECK(check_vector(output, 2, expected_refreshed_q, 2.0e-14) == 0);
  CHECK(ss_deflation_refresh(deflation, matrix, 3U) == SS_OK);
  CHECK(ss_deflation_apply_q(deflation, first, output) == SS_OK);
  CHECK(check_vector(output, 2, expected_q, 2.0e-14) == 0);
  CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
  CHECK(diagnostics.invariants_validated != 0);
  CHECK(diagnostics.operator_state >= 0);
  CHECK(ss_deflation_apply_right(deflation, refreshed_matrix, first, output) ==
        SS_ERROR_INVALID_STATE);
  {
    PetscReal wrong_operator_error = 0.0;
    CHECK(ss_deflation_invariant_error(deflation, refreshed_matrix,
                                       &wrong_operator_error) ==
          SS_ERROR_INVALID_STATE);
  }
  {
    const uint64_t refresh_count = diagnostics.refresh_count;
    CHECK(ss_deflation_refresh(deflation, matrix, 3U) == SS_OK);
    CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
    CHECK(diagnostics.refresh_count == refresh_count);
  }
  CHECK(MatSetValue(matrix, 0, 0, 6.0, INSERT_VALUES) == PETSC_SUCCESS);
  CHECK(MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(ss_deflation_apply_right(deflation, matrix, first, output) ==
        SS_ERROR_INVALID_STATE);
  CHECK(MatSetValue(matrix, 0, 0, 2.0, INSERT_VALUES) == PETSC_SUCCESS);
  CHECK(MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(ss_deflation_refresh(deflation, matrix, 3U) == SS_ERROR_INTEGRITY);
  CHECK(ss_deflation_refresh(deflation, matrix, 4U) == SS_OK);

  CHECK(create_dense(2, ill_conditioned_matrix_values, &ill_conditioned_matrix) == 0);
  CHECK(ss_deflation_begin(deflation, 19U) == SS_OK);
  CHECK(ss_deflation_refresh(deflation, ill_conditioned_matrix, 5U) == SS_OK);
  CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
  CHECK(diagnostics.persistent_count == 1U);
  CHECK(diagnostics.active_count == 0U);
  CHECK(diagnostics.ill_conditioned_drop_count > 0U);
  CHECK(diagnostics.last_drop_reason == SS_DEFLATION_DROP_ILL_CONDITIONED);
  CHECK(ss_deflation_rollback(deflation) == SS_OK);
  CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
  CHECK(diagnostics.persistent_count == 1U && diagnostics.active_count == 1U);
  CHECK(ss_deflation_apply_q(deflation, first, output) == SS_OK);
  CHECK(check_vector(output, 2, expected_q, 2.0e-14) == 0);
  CHECK(MatDestroy(&ill_conditioned_matrix) == PETSC_SUCCESS);

  CHECK(ss_deflation_begin(deflation, 20U) == SS_OK);
  CHECK(ss_deflation_add_candidate(deflation, second, second, "rejected-trial",
                                   &record) == SS_OK);
  CHECK(record.admitted != 0);
  CHECK(ss_deflation_refresh(deflation, refreshed_matrix, 6U) == SS_OK);
  CHECK(ss_deflation_rollback(deflation) == SS_OK);
  CHECK(ss_deflation_persistent_digest(deflation, after) == SS_OK);
  CHECK(strcmp(before, after) == 0);
  CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
  CHECK(diagnostics.persistent_count == 1U);
  CHECK(diagnostics.provisional_count == 0U);
  CHECK(diagnostics.accepted_generation == 10U);
  CHECK(diagnostics.operator_generation == 4U);
  CHECK(ss_deflation_apply_q(deflation, first, output) == SS_OK);
  CHECK(check_vector(output, 2, expected_q, 2.0e-14) == 0);
  CHECK(MatDestroy(&refreshed_matrix) == PETSC_SUCCESS);

  CHECK(ss_deflation_begin(deflation, 30U) == SS_OK);
  CHECK(ss_deflation_add_candidate(deflation, second, second, "accepted-newest",
                                   &record) == SS_OK);
  CHECK(record.admitted != 0);
  CHECK(ss_deflation_refresh(deflation, matrix, 7U) == SS_OK);
  CHECK(ss_deflation_commit(deflation, 11U) == SS_OK);
  CHECK(ss_deflation_persistent_digest(deflation, replaced) == SS_OK);
  CHECK(strcmp(before, replaced) != 0);
  CHECK(ss_deflation_apply_q(deflation, second, output) == SS_OK);
  CHECK(check_vector(output, 2, expected_replacement_q, 2.0e-14) == 0);
  CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
  CHECK(diagnostics.persistent_count == 1U);
  CHECK(diagnostics.accepted_generation == 11U);
  CHECK(VecDestroy(&output) == PETSC_SUCCESS);
  CHECK(VecDestroy(&second) == PETSC_SUCCESS);
  CHECK(VecDestroy(&first) == PETSC_SUCCESS);
  CHECK(ss_deflation_destroy(&deflation) == SS_OK);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}
