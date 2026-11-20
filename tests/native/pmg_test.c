/* P-multigrid hierarchy, Galerkin coarse operators, V-cycle and near-null
 * space.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/pmg/pmg.h"

#include <petscksp.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(expression)                                                              \
  do {                                                                                 \
    if (!(expression)) {                                                               \
      fprintf(stderr, "%s:%d: assertion failed: %s\n", __FILE__, __LINE__,             \
              #expression);                                                            \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

#define CHECK_NEAR(actual, expected, tolerance)                                        \
  do {                                                                                 \
    const double checked_actual = (double)(actual);                                    \
    const double checked_expected = (double)(expected);                                \
    if (!isfinite(checked_actual) ||                                                   \
        fabs(checked_actual - checked_expected) > (double)(tolerance)) {               \
      fprintf(stderr, "%s:%d: |%.17g - %.17g| > %.3g\n", __FILE__, __LINE__,           \
              checked_actual, checked_expected, (double)(tolerance));                  \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

static int create_dense(PetscInt rows, PetscInt columns, const double *values,
                        Mat *matrix) {
  PetscInt row;
  CHECK(values != NULL && matrix != NULL);
  CHECK(MatCreateSeqAIJ(PETSC_COMM_SELF, rows, columns, columns, NULL, matrix) ==
        PETSC_SUCCESS);
  for (row = 0; row < rows; ++row) {
    PetscInt column;
    for (column = 0; column < columns; ++column) {
      CHECK(MatSetValue(*matrix, row, column,
                        (PetscScalar)values[row * columns + column],
                        INSERT_VALUES) == PETSC_SUCCESS);
    }
  }
  CHECK(MatAssemblyBegin(*matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(MatAssemblyEnd(*matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  return 0;
}

static int create_vector(PetscInt size, const double *values, Vec *vector) {
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

typedef struct {
  Mat assembled;
} TestShellContext;

static PetscErrorCode test_shell_mult(Mat shell, Vec input, Vec output) {
  TestShellContext *context = NULL;
  PetscCall(MatShellGetContext(shell, &context));
  PetscCheck(context != NULL && context->assembled != NULL, PETSC_COMM_SELF,
             PETSC_ERR_ARG_WRONGSTATE, "missing test shell operator");
  PetscCall(MatMult(context->assembled, input, output));
  return PETSC_SUCCESS;
}

static PetscErrorCode test_shell_mult_transpose(Mat shell, Vec input, Vec output) {
  TestShellContext *context = NULL;
  PetscCall(MatShellGetContext(shell, &context));
  PetscCheck(context != NULL && context->assembled != NULL, PETSC_COMM_SELF,
             PETSC_ERR_ARG_WRONGSTATE, "missing test shell operator");
  PetscCall(MatMultTranspose(context->assembled, input, output));
  return PETSC_SUCCESS;
}

static PetscErrorCode test_shell_get_diagonal(Mat shell, Vec diagonal) {
  TestShellContext *context = NULL;
  PetscCall(MatShellGetContext(shell, &context));
  PetscCheck(context != NULL && context->assembled != NULL, PETSC_COMM_SELF,
             PETSC_ERR_ARG_WRONGSTATE, "missing test shell operator");
  PetscCall(MatGetDiagonal(context->assembled, diagonal));
  return PETSC_SUCCESS;
}

static PetscErrorCode test_shell_invert_variable_block_diagonal(
    Mat shell, PetscInt block_count, const PetscInt *block_sizes, PetscScalar *values) {
  TestShellContext *context = NULL;
  PetscCall(MatShellGetContext(shell, &context));
  PetscCheck(context != NULL && context->assembled != NULL, PETSC_COMM_SELF,
             PETSC_ERR_ARG_WRONGSTATE, "missing test shell operator");
  PetscCall(MatInvertVariableBlockDiagonal(context->assembled, block_count, block_sizes,
                                           values));
  return PETSC_SUCCESS;
}

static int create_test_shell(Mat assembled, TestShellContext *context, Mat *shell) {
  PetscInt rows = 0;
  PetscInt columns = 0;
  CHECK(assembled != NULL && context != NULL && shell != NULL);
  CHECK(MatGetSize(assembled, &rows, &columns) == PETSC_SUCCESS);
  CHECK(rows > 0 && rows == columns);
  context->assembled = assembled;
  CHECK(MatCreateShell(PETSC_COMM_SELF, rows, rows, rows, rows, context, shell) ==
        PETSC_SUCCESS);
  CHECK(MatShellSetOperation(*shell, MATOP_MULT, (void (*)(void))test_shell_mult) ==
        PETSC_SUCCESS);
  CHECK(MatShellSetOperation(*shell, MATOP_MULT_TRANSPOSE,
                             (void (*)(void))test_shell_mult_transpose) ==
        PETSC_SUCCESS);
  CHECK(MatShellSetOperation(*shell, MATOP_GET_DIAGONAL,
                             (void (*)(void))test_shell_get_diagonal) == PETSC_SUCCESS);
  CHECK(
      MatShellSetOperation(*shell, MATOP_INVERT_VBLOCK_DIAGONAL,
                           (void (*)(void))test_shell_invert_variable_block_diagonal) ==
      PETSC_SUCCESS);
  return 0;
}

static int check_vector(Vec vector, PetscInt size, const double *expected,
                        double tolerance) {
  const PetscScalar *values = NULL;
  PetscInt index;
  CHECK(VecGetArrayRead(vector, &values) == PETSC_SUCCESS);
  for (index = 0; index < size; ++index) {
    CHECK_NEAR(PetscRealPart(values[index]), expected[index], tolerance);
  }
  CHECK(VecRestoreArrayRead(vector, &values) == PETSC_SUCCESS);
  return 0;
}

static void level_policy(SsPmgLevelPolicy *policy, SsPmgLevelMethod method,
                         const char *prefix) {
  memset(policy, 0, sizeof(*policy));
  (void)snprintf(policy->options_prefix, sizeof(policy->options_prefix), "%s", prefix);
  policy->method = method;
  policy->active_rank_count = 1U;
  policy->pre_smooth_steps = method == SS_PMG_LEVEL_PREONLY_LU ? 0 : 1;
  policy->post_smooth_steps = method == SS_PMG_LEVEL_PREONLY_LU ? 0 : 1;
  policy->maximum_iterations = method == SS_PMG_LEVEL_PREONLY_LU ? 1 : 8;
  policy->richardson_scale = 0.25;
  policy->rtol = 1.0e-13;
  policy->atol = 1.0e-15;
}

static void two_level_spec(Mat prolongation, SsPmgLevelPolicy policies[2],
                           SsPmgSpecification *specification) {
  static const uint32_t orders[] = {1U, 2U};
  static Mat transfers[1];
  transfers[0] = prolongation;
  level_policy(&policies[0], SS_PMG_LEVEL_PREONLY_LU, "pmgc_");
  level_policy(&policies[1], SS_PMG_LEVEL_RICHARDSON_NONE, "pmgf_");
  memset(specification, 0, sizeof(*specification));
  specification->fine_order = 2U;
  specification->level_count = 2U;
  specification->orders = orders;
  specification->prolongations = transfers;
  specification->level_policies = policies;
  specification->transfer_generation = 7U;
  specification->numeric_lag_interval = 1U;
  specification->memory_cap_bytes = 16U * 1024U * 1024U;
}

static int test_hierarchy(void) {
  static const uint32_t expected[4][4] = {
      {1U, 0U, 0U, 0U},
      {1U, 2U, 0U, 0U},
      {1U, 2U, 4U, 0U},
      {1U, 2U, 4U, 8U},
  };
  static const uint32_t fine_orders[] = {1U, 2U, 4U, 8U};
  size_t case_index;
  for (case_index = 0U; case_index < 4U; ++case_index) {
    SsPmgHierarchy hierarchy = {0};
    size_t level;
    CHECK(ss_pmg_hierarchy_resolve(fine_orders[case_index], &hierarchy) == SS_OK);
    CHECK(hierarchy.fine_order == fine_orders[case_index]);
    CHECK(hierarchy.level_count == case_index + 1U);
    for (level = 0U; level < hierarchy.level_count; ++level) {
      CHECK(hierarchy.orders[level] == expected[case_index][level]);
    }
    CHECK(ss_pmg_hierarchy_validate(fine_orders[case_index], hierarchy.level_count,
                                    hierarchy.orders, &hierarchy) == SS_OK);
  }
  {
    const uint32_t duplicate[] = {1U, 2U, 2U};
    const uint32_t missing[] = {1U, 4U};
    const uint32_t reversed[] = {2U, 1U};
    SsPmgHierarchy hierarchy = {0};
    CHECK(ss_pmg_hierarchy_validate(2U, 3U, duplicate, &hierarchy) ==
          SS_ERROR_DUPLICATE);
    CHECK(ss_pmg_hierarchy_validate(4U, 2U, missing, &hierarchy) == SS_ERROR_INTEGRITY);
    CHECK(ss_pmg_hierarchy_validate(2U, 2U, reversed, &hierarchy) ==
          SS_ERROR_INTEGRITY);
    CHECK(ss_pmg_hierarchy_resolve(3U, &hierarchy) == SS_ERROR_NOT_FOUND);
  }
  {
    PetscMPIInt ranks[4] = {-1, -1, -1, -1};
    const PetscMPIInt arbitrary[] = {6, 1, 4};
    CHECK(ss_pmg_active_ranks_resolve(9, 4, ranks, 4U) == SS_OK);
    CHECK(ranks[0] == 0 && ranks[1] == 2 && ranks[2] == 4 && ranks[3] == 6);
    CHECK(ss_pmg_active_ranks_validate(7, 3U, arbitrary) == SS_OK);
    CHECK(ss_pmg_active_ranks_validate(7, 3U, (const PetscMPIInt[]){0, 2, 2}) ==
          SS_ERROR_DUPLICATE);
  }
  return 0;
}

static int create_oracle_objects(Mat *fine, Mat *prolongation) {
  static const double fine_values[] = {
      2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0,
  };
  static const double prolongation_values[] = {
      1.0, 0.0, 0.5, 0.5, 0.0, 1.0,
  };
  CHECK(create_dense(3, 3, fine_values, fine) == 0);
  CHECK(create_dense(3, 2, prolongation_values, prolongation) == 0);
  return 0;
}

static int test_constrained_galerkin(void) {
  static const double fine_values[] = {
      2.0,
      -1.0,
      -1.0,
      2.0,
  };
  static const double constrained_prolongation_values[] = {
      1.0,
      0.5,
  };
  static const double coarse_value[] = {2.0};
  static const double expected_fine[] = {2.0, 1.0};
  SsPmgLevelPolicy policies[2];
  SsPmgSpecification specification;
  SsPmgSetupRecord setup = {0};
  SsPmg *pmg = NULL;
  Mat fine = NULL;
  Mat prolongation = NULL;
  Mat coarse = NULL;
  Vec xc = NULL;
  Vec xf = NULL;
  PetscScalar value = 0.0;
  PetscInt zero = 0;
  CHECK(create_dense(2, 2, fine_values, &fine) == 0);
  CHECK(create_dense(2, 1, constrained_prolongation_values, &prolongation) == 0);
  two_level_spec(prolongation, policies, &specification);
  CHECK(ss_pmg_create(PETSC_COMM_SELF, &specification, &pmg) == SS_OK);
  CHECK(ss_pmg_setup(pmg, fine, 1U, 1U, &setup) == SS_OK);
  CHECK(ss_pmg_level_operator(pmg, 0U, &coarse) == SS_OK);
  CHECK(MatGetValue(coarse, zero, zero, &value) == PETSC_SUCCESS);
  CHECK_NEAR(PetscRealPart(value), 1.5, 2.0e-14);
  CHECK(create_vector(1, coarse_value, &xc) == 0);
  CHECK(MatCreateVecs(prolongation, NULL, &xf) == PETSC_SUCCESS);
  CHECK(MatMult(prolongation, xc, xf) == PETSC_SUCCESS);
  CHECK(check_vector(xf, 2, expected_fine, 2.0e-14) == 0);
  CHECK(ss_pmg_destroy(&pmg) == SS_OK);
  CHECK(VecDestroy(&xf) == PETSC_SUCCESS);
  CHECK(VecDestroy(&xc) == PETSC_SUCCESS);
  CHECK(MatDestroy(&prolongation) == PETSC_SUCCESS);
  CHECK(MatDestroy(&fine) == PETSC_SUCCESS);
  return 0;
}

static int test_galerkin(void) {
  static const double coarse_input[] = {2.0, -1.0};
  static const double expected_prolonged[] = {2.0, 0.5, -1.0};
  static const double expected_coarse[] = {3.5, -2.5};
  SsPmgLevelPolicy policies[2];
  SsPmgSpecification specification;
  SsPmgSetupRecord setup = {0};
  SsPmg *pmg = NULL;
  Mat fine = NULL;
  Mat prolongation = NULL;
  Mat coarse = NULL;
  Vec xc = NULL;
  Vec xf = NULL;
  Vec ac_xc = NULL;
  Vec af_xf = NULL;
  PetscScalar fine_energy = 0.0;
  PetscScalar coarse_energy = 0.0;
  PetscScalar value = 0.0;
  PetscInt row = 0;
  PetscInt column = 0;

  CHECK(create_oracle_objects(&fine, &prolongation) == 0);
  two_level_spec(prolongation, policies, &specification);
  CHECK(ss_pmg_create(PETSC_COMM_SELF, &specification, &pmg) == SS_OK);
  CHECK(ss_pmg_setup(pmg, fine, 11U, 17U, &setup) == SS_OK);
  CHECK(setup.hierarchy.level_count == 2U);
  CHECK(setup.galerkin_symbolic_count == 1U);
  CHECK(setup.galerkin_numeric_count == 1U);
  CHECK(ss_pmg_level_operator(pmg, 0U, &coarse) == SS_OK);
  CHECK(MatGetValue(coarse, row, column, &value) == PETSC_SUCCESS);
  CHECK_NEAR(PetscRealPart(value), 1.5, 2.0e-14);
  row = 0;
  column = 1;
  CHECK(MatGetValue(coarse, row, column, &value) == PETSC_SUCCESS);
  CHECK_NEAR(PetscRealPart(value), -0.5, 2.0e-14);
  CHECK(create_vector(2, coarse_input, &xc) == 0);
  CHECK(MatCreateVecs(prolongation, NULL, &xf) == PETSC_SUCCESS);
  CHECK(MatMult(prolongation, xc, xf) == PETSC_SUCCESS);
  CHECK(check_vector(xf, 3, expected_prolonged, 2.0e-14) == 0);
  CHECK(MatCreateVecs(coarse, &ac_xc, NULL) == PETSC_SUCCESS);
  CHECK(MatMult(coarse, xc, ac_xc) == PETSC_SUCCESS);
  CHECK(check_vector(ac_xc, 2, expected_coarse, 2.0e-14) == 0);
  CHECK(VecDot(xc, ac_xc, &coarse_energy) == PETSC_SUCCESS);
  CHECK(MatCreateVecs(fine, &af_xf, NULL) == PETSC_SUCCESS);
  CHECK(MatMult(fine, xf, af_xf) == PETSC_SUCCESS);
  CHECK(VecDot(xf, af_xf, &fine_energy) == PETSC_SUCCESS);
  CHECK_NEAR(PetscRealPart(coarse_energy), 9.5, 2.0e-14);
  CHECK_NEAR(PetscRealPart(fine_energy), 9.5, 2.0e-14);
  CHECK(ss_pmg_destroy(&pmg) == SS_OK);
  CHECK(VecDestroy(&af_xf) == PETSC_SUCCESS);
  CHECK(VecDestroy(&ac_xc) == PETSC_SUCCESS);
  CHECK(VecDestroy(&xf) == PETSC_SUCCESS);
  CHECK(VecDestroy(&xc) == PETSC_SUCCESS);
  CHECK(MatDestroy(&prolongation) == PETSC_SUCCESS);
  CHECK(MatDestroy(&fine) == PETSC_SUCCESS);
  CHECK(test_constrained_galerkin() == 0);
  return 0;
}

static int test_external_level_operators(void) {
  static const double coarse_values[] = {
      1.5,
      -0.5,
      -0.5,
      1.5,
  };
  static const double rhs_values[] = {1.0, 2.0, 3.0};
  static const double expected[] = {2.25, 3.5, 3.25};
  static const PetscInt variable_blocks[] = {1, 1, 1};
  static const SsPmgLevelMethod smoother_methods[] = {
      SS_PMG_LEVEL_RICHARDSON_JACOBI, SS_PMG_LEVEL_RICHARDSON_VPBJACOBI,
      SS_PMG_LEVEL_CHEBYSHEV_JACOBI,  SS_PMG_LEVEL_CHEBYSHEV_POINT_JACOBI,
      SS_PMG_LEVEL_FGMRES_JACOBI,     SS_PMG_LEVEL_FGMRES_VPBJACOBI,
  };
  SsPmgLevelPolicy policies[2];
  SsPmgSpecification specification;
  SsPmgSetupRecord setup = {0};
  SsPmgApplyRecord apply = {0};
  SsPmgDiagnostics diagnostics = {0};
  SsPmg *pmg = NULL;
  TestShellContext shell_context = {0};
  Mat fine_assembled = NULL;
  Mat fine_shell = NULL;
  Mat prolongation = NULL;
  Mat coarse = NULL;
  Mat installed = NULL;
  Mat operators[2] = {NULL, NULL};
  Vec rhs = NULL;
  Vec correction = NULL;
  size_t method_index;

  CHECK(create_oracle_objects(&fine_assembled, &prolongation) == 0);
  CHECK(create_dense(2, 2, coarse_values, &coarse) == 0);
  CHECK(create_test_shell(fine_assembled, &shell_context, &fine_shell) == 0);
  CHECK(MatSetVariableBlockSizes(fine_shell, 3, variable_blocks) == PETSC_SUCCESS);
  two_level_spec(prolongation, policies, &specification);
  operators[0] = coarse;
  operators[1] = fine_shell;
  CHECK(ss_pmg_create(PETSC_COMM_SELF, &specification, &pmg) == SS_OK);
  CHECK(ss_pmg_set_level_operators(pmg, 2U, operators) == SS_OK);
  CHECK(ss_pmg_setup(pmg, fine_shell, 1U, 1U, &setup) == SS_OK);
  CHECK(setup.galerkin_symbolic_count == 0U);
  CHECK(setup.galerkin_numeric_count == 0U);
  CHECK(setup.matrix_object_count == 3U);
  CHECK(setup.levels[1].variable_point_block_count == 3U);
  CHECK(ss_pmg_level_operator(pmg, 0U, &installed) == SS_OK);
  CHECK(installed == coarse);
  CHECK(ss_pmg_level_operator(pmg, 1U, &installed) == SS_OK);
  CHECK(installed == fine_shell);
  CHECK(create_vector(3, rhs_values, &rhs) == 0);
  CHECK(VecDuplicate(rhs, &correction) == PETSC_SUCCESS);
  CHECK(ss_pmg_apply(pmg, rhs, correction, &apply) == SS_OK);
  CHECK(check_vector(correction, 3, expected, 2.0e-13) == 0);
  CHECK(ss_pmg_diagnostics(pmg, &diagnostics) == SS_OK);
  CHECK(diagnostics.current_matrix_objects == 3U);

  for (method_index = 0U;
       method_index < sizeof(smoother_methods) / sizeof(smoother_methods[0]);
       ++method_index) {
    SsPmg *with_smoother = NULL;
    policies[1].method = smoother_methods[method_index];
    policies[1].pre_smooth_steps = 2;
    policies[1].post_smooth_steps = 3;
    policies[1].maximum_iterations = 3;
    policies[1].richardson_scale = 0.5;
    CHECK(ss_pmg_create(PETSC_COMM_SELF, &specification, &with_smoother) == SS_OK);
    CHECK(ss_pmg_set_level_operators(with_smoother, 2U, operators) == SS_OK);
    CHECK(ss_pmg_setup(with_smoother, fine_shell, 1U, 1U, &setup) == SS_OK);
    CHECK(setup.levels[1].variable_point_block_count == 3U);
    CHECK(ss_pmg_apply(with_smoother, rhs, correction, &apply) == SS_OK);
    CHECK(apply.status == SS_OK);
    CHECK(apply.level_solve_calls == 3U);
    CHECK(ss_pmg_destroy(&with_smoother) == SS_OK);
  }

  CHECK(ss_pmg_setup(pmg, fine_shell, 1U, 1U, &setup) == SS_OK);
  CHECK(setup.numeric_refreshed == 0);

  /* The caller updates all level values in place.  PMG must refresh solver
   * state without constructing a Galerkin product or replacing either Mat. */
  CHECK(MatScale(fine_assembled, 2.0) == PETSC_SUCCESS);
  CHECK(MatScale(coarse, 2.0) == PETSC_SUCCESS);
  CHECK(MatAssemblyBegin(fine_shell, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(MatAssemblyEnd(fine_shell, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(ss_pmg_setup(pmg, fine_shell, 1U, 2U, &setup) == SS_OK);
  CHECK(setup.numeric_refreshed == 1);
  CHECK(setup.galerkin_symbolic_count == 0U);
  CHECK(setup.galerkin_numeric_count == 0U);
  CHECK(ss_pmg_level_operator(pmg, 0U, &installed) == SS_OK);
  CHECK(installed == coarse);
  CHECK(ss_pmg_level_operator(pmg, 1U, &installed) == SS_OK);
  CHECK(installed == fine_shell);

  CHECK(ss_pmg_destroy(&pmg) == SS_OK);
  CHECK(VecDestroy(&correction) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(MatDestroy(&fine_shell) == PETSC_SUCCESS);
  CHECK(MatDestroy(&coarse) == PETSC_SUCCESS);
  CHECK(MatDestroy(&prolongation) == PETSC_SUCCESS);
  CHECK(MatDestroy(&fine_assembled) == PETSC_SUCCESS);
  return 0;
}

static int test_three_level_cycle(void) {
  static const uint32_t orders[] = {1U, 2U, 4U};
  static const double p1_to_p2_values[] = {
      1.0, 0.0, 0.5, 0.5, 0.0, 1.0,
  };
  static const double p2_to_p4_values[] = {
      1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0,
  };
  static const double fine_values[] = {
      4.0,  -1.0, 0.0, 0.0, 0.0,  -1.0, 4.0,  -1.0, 0.0, 0.0, 0.0,  -1.0, 4.0,
      -1.0, 0.0,  0.0, 0.0, -1.0, 4.0,  -1.0, 0.0,  0.0, 0.0, -1.0, 4.0,
  };
  static const double rhs_values[] = {1.0, 2.0, 3.0, 4.0, 5.0};
  SsPmgLevelPolicy policies[3];
  SsPmgSpecification specification = {0};
  SsPmgSetupRecord setup = {0};
  SsPmgApplyRecord apply = {0};
  SsPmg *pmg = NULL;
  Mat fine = NULL;
  Mat transfers[2] = {NULL, NULL};
  Vec rhs = NULL;
  Vec correction = NULL;
  Vec residual = NULL;
  PetscReal rhs_norm = 0.0;
  PetscReal residual_norm = 0.0;

  CHECK(create_dense(5, 5, fine_values, &fine) == 0);
  CHECK(create_dense(3, 2, p1_to_p2_values, &transfers[0]) == 0);
  CHECK(create_dense(5, 3, p2_to_p4_values, &transfers[1]) == 0);
  level_policy(&policies[0], SS_PMG_LEVEL_PREONLY_LU, "pmg3c_");
  level_policy(&policies[1], SS_PMG_LEVEL_RICHARDSON_NONE, "pmg3m_");
  level_policy(&policies[2], SS_PMG_LEVEL_RICHARDSON_NONE, "pmg3f_");
  policies[1].richardson_scale = 0.2;
  policies[2].richardson_scale = 0.2;
  specification.fine_order = 4U;
  specification.level_count = 3U;
  specification.orders = orders;
  specification.prolongations = transfers;
  specification.level_policies = policies;
  specification.transfer_generation = 9U;
  specification.numeric_lag_interval = 1U;
  specification.memory_cap_bytes = 16U * 1024U * 1024U;
  CHECK(ss_pmg_create(PETSC_COMM_SELF, &specification, &pmg) == SS_OK);
  CHECK(ss_pmg_setup(pmg, fine, 1U, 1U, &setup) == SS_OK);
  CHECK(setup.hierarchy.level_count == 3U);
  CHECK(setup.levels[0].global_dofs == 2);
  CHECK(setup.levels[1].global_dofs == 3);
  CHECK(setup.levels[2].global_dofs == 5);
  CHECK(create_vector(5, rhs_values, &rhs) == 0);
  CHECK(VecDuplicate(rhs, &correction) == PETSC_SUCCESS);
  CHECK(VecDuplicate(rhs, &residual) == PETSC_SUCCESS);
  CHECK(ss_pmg_apply(pmg, rhs, correction, &apply) == SS_OK);
  CHECK(apply.level_solve_calls == 5U);
  CHECK(VecNorm(rhs, NORM_2, &rhs_norm) == PETSC_SUCCESS);
  CHECK(MatMult(fine, correction, residual) == PETSC_SUCCESS);
  CHECK(VecAYPX(residual, -1.0, rhs) == PETSC_SUCCESS);
  CHECK(VecNorm(residual, NORM_2, &residual_norm) == PETSC_SUCCESS);
  CHECK(residual_norm < rhs_norm);
  CHECK(ss_pmg_destroy(&pmg) == SS_OK);
  CHECK(VecDestroy(&residual) == PETSC_SUCCESS);
  CHECK(VecDestroy(&correction) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(MatDestroy(&transfers[1]) == PETSC_SUCCESS);
  CHECK(MatDestroy(&transfers[0]) == PETSC_SUCCESS);
  CHECK(MatDestroy(&fine) == PETSC_SUCCESS);
  return 0;
}

static int test_chebyshev_wide_spectrum(void) {
  static const uint32_t orders[] = {1U, 2U, 4U};
  double fine_values[64] = {0.0};
  double prolongation_values[8] = {0.0};
  static const double identity_value[] = {1.0};
  double rhs_values[8];
  SsPmgLevelPolicy policies[3];
  SsPmgSpecification specification = {0};
  SsPmgSetupRecord setup = {0};
  SsPmgApplyRecord apply = {0};
  SsPmg *pmg = NULL;
  Mat fine = NULL;
  Mat coarse_to_intermediate = NULL;
  Mat intermediate_to_fine = NULL;
  Mat transfers[2] = {NULL, NULL};
  Vec rhs = NULL;
  Vec correction = NULL;
  Vec residual = NULL;
  PetscReal rhs_norm = 0.0;
  PetscReal residual_norm = 0.0;
  PetscInt row;

  /*
   * Jacobi leaves this correlation matrix with eigenvalues 0.1 and 7.3.
   * The historic hard-coded Chebyshev interval [0.25, 5.1] amplifies its
   * highest mode instead of acting as a stable fixed-work smoother.  The
   * coarse space deliberately spans only one coordinate so that the fine
   * smoother must handle that wide spectrum itself.
   */
  for (row = 0; row < 8; ++row) {
    PetscInt column;
    rhs_values[row] = 1.0;
    for (column = 0; column < 8; ++column) {
      fine_values[row * 8 + column] = row == column ? 1.0 : 0.9;
    }
  }
  prolongation_values[0] = 1.0;
  CHECK(create_dense(8, 8, fine_values, &fine) == 0);
  CHECK(create_dense(1, 1, identity_value, &coarse_to_intermediate) == 0);
  CHECK(create_dense(8, 1, prolongation_values, &intermediate_to_fine) == 0);
  transfers[0] = coarse_to_intermediate;
  transfers[1] = intermediate_to_fine;
  level_policy(&policies[0], SS_PMG_LEVEL_PREONLY_LU, "pmgwc_");
  level_policy(&policies[1], SS_PMG_LEVEL_RICHARDSON_NONE, "pmgwm_");
  level_policy(&policies[2], SS_PMG_LEVEL_CHEBYSHEV_JACOBI, "pmgwf_");
  policies[2].pre_smooth_steps = 5;
  policies[2].post_smooth_steps = 5;
  policies[2].maximum_iterations = 5;
  specification.fine_order = 4U;
  specification.level_count = 3U;
  specification.orders = orders;
  specification.prolongations = transfers;
  specification.level_policies = policies;
  specification.transfer_generation = 13U;
  specification.numeric_lag_interval = 1U;
  specification.memory_cap_bytes = 16U * 1024U * 1024U;
  CHECK(ss_pmg_create(PETSC_COMM_SELF, &specification, &pmg) == SS_OK);
  CHECK(ss_pmg_setup(pmg, fine, 1U, 1U, &setup) == SS_OK);
  CHECK(create_vector(8, rhs_values, &rhs) == 0);
  CHECK(VecDuplicate(rhs, &correction) == PETSC_SUCCESS);
  CHECK(VecDuplicate(rhs, &residual) == PETSC_SUCCESS);
  CHECK(ss_pmg_apply(pmg, rhs, correction, &apply) == SS_OK);
  CHECK(VecNorm(rhs, NORM_2, &rhs_norm) == PETSC_SUCCESS);
  CHECK(MatMult(fine, correction, residual) == PETSC_SUCCESS);
  CHECK(VecAYPX(residual, -1.0, rhs) == PETSC_SUCCESS);
  CHECK(VecNorm(residual, NORM_2, &residual_norm) == PETSC_SUCCESS);
  CHECK(isfinite((double)residual_norm) && residual_norm < rhs_norm);
  CHECK(ss_pmg_destroy(&pmg) == SS_OK);
  CHECK(VecDestroy(&residual) == PETSC_SUCCESS);
  CHECK(VecDestroy(&correction) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(MatDestroy(&intermediate_to_fine) == PETSC_SUCCESS);
  CHECK(MatDestroy(&coarse_to_intermediate) == PETSC_SUCCESS);
  CHECK(MatDestroy(&fine) == PETSC_SUCCESS);
  return 0;
}

static int test_vcycle(void) {
  static const double rhs_values[] = {1.0, 2.0, 3.0};
  static const double expected[] = {2.25, 3.5, 3.25};
  static const uint32_t one_level_order[] = {1U};
  static const PetscMPIInt active_rank[] = {0};
  SsPmgLevelPolicy policies[2];
  SsPmgLevelPolicy inexact_policy;
  SsPmgSpecification specification;
  SsPmgSpecification inexact_specification;
  SsPmgSetupRecord setup = {0};
  SsPmgApplyRecord apply = {0};
  SsPmgDiagnostics diagnostics = {0};
  SsPmg *pmg = NULL;
  SsPmg *inexact_pmg = NULL;
  Mat fine = NULL;
  Mat prolongation = NULL;
  Vec rhs = NULL;
  Vec correction = NULL;
  Vec residual = NULL;
  PetscReal rhs_norm = NAN;
  PetscReal residual_norm = NAN;

  CHECK(create_oracle_objects(&fine, &prolongation) == 0);
  two_level_spec(prolongation, policies, &specification);
  CHECK(ss_pmg_create(PETSC_COMM_SELF, &specification, &pmg) == SS_OK);
  CHECK(ss_pmg_setup(pmg, fine, 1U, 1U, &setup) == SS_OK);
  CHECK(create_vector(3, rhs_values, &rhs) == 0);
  CHECK(VecDuplicate(rhs, &correction) == PETSC_SUCCESS);
  CHECK(ss_pmg_apply(pmg, rhs, correction, &apply) == SS_OK);
  CHECK(apply.status == SS_OK);
  CHECK(apply.level_solve_calls == 3U);
  CHECK(apply.operator_apply_calls == 2U);
  /* A level spanning the complete world communicator solves directly in the
   * world vectors; no redistribution or identity VecCopy is required. */
  CHECK(apply.bridge_calls == 0U);
  CHECK(apply.bridge_bytes == 0U);
  CHECK(check_vector(correction, 3, expected, 2.0e-13) == 0);
  CHECK(ss_pmg_diagnostics(pmg, &diagnostics) == SS_OK);
  CHECK(diagnostics.apply_calls == 1U);
  CHECK(diagnostics.contract_apply_calls == 1U);
  CHECK(diagnostics.preconditioner_apply_calls == 0U);
  CHECK(diagnostics.norm_validation_reductions == 2U);
  CHECK(diagnostics.level_status_reductions == 0U);
  CHECK(diagnostics.failed_apply_calls == 0U);
  CHECK(ss_pmg_destroy(&pmg) == SS_OK);

  /*
   * A nonsymmetric FGMRES/Jacobi smoother is fixed work: the pre- and
   * post-smoothing calls must use their requested budgets rather than a
   * residual-based convergence test.
   */
  policies[1].method = SS_PMG_LEVEL_FGMRES_JACOBI;
  policies[1].pre_smooth_steps = 2;
  policies[1].post_smooth_steps = 3;
  policies[1].maximum_iterations = 3;
  specification.injected_failure = SS_PMG_FAIL_RESTRICT;
  specification.injected_failure_level = 1U;
  CHECK(ss_pmg_create(PETSC_COMM_SELF, &specification, &pmg) == SS_OK);
  CHECK(ss_pmg_setup(pmg, fine, 1U, 1U, &setup) == SS_OK);
  CHECK(ss_pmg_apply(pmg, rhs, correction, &apply) == SS_ERROR_INJECTED);
  /* The failure preserves the pre-smoother diagnostic, but the rank must
   * execute the complete three-solve cycle. */
  CHECK(apply.level_solve_calls == 3U);
  CHECK(apply.last_level_iterations == 2);
  CHECK(ss_pmg_destroy(&pmg) == SS_OK);

  specification.injected_failure = SS_PMG_FAIL_NONE;
  specification.injected_failure_level = 0U;
  CHECK(ss_pmg_create(PETSC_COMM_SELF, &specification, &pmg) == SS_OK);
  CHECK(ss_pmg_setup(pmg, fine, 1U, 1U, &setup) == SS_OK);
  {
    const SsStatus fgmres_status = ss_pmg_apply(pmg, rhs, correction, &apply);
    if (fgmres_status != SS_OK) {
      fprintf(stderr,
              "FGMRES smoother failed: status=%d level=%zu point=%d reason=%d "
              "iterations=%" PetscInt_FMT " residual=%.17g\n",
              (int)fgmres_status, apply.failed_level, (int)apply.failed_point,
              (int)apply.last_level_reason, apply.last_level_iterations,
              (double)apply.last_level_residual_norm);
    }
    CHECK(fgmres_status == SS_OK);
  }
  CHECK(apply.status == SS_OK);
  CHECK(apply.level_solve_calls == 3U);
  CHECK(apply.last_level_reason == KSP_CONVERGED_ATOL);
  CHECK(apply.last_level_iterations == 0);
  CHECK(ss_pmg_destroy(&pmg) == SS_OK);

  /* The opt-in legacy path post-smooths the complete iterate with FGMRES and
   * therefore performs no second explicit residual matrix action. */
  specification.fgmres_full_iterate_post = 1;
  CHECK(ss_pmg_create(PETSC_COMM_SELF, &specification, &pmg) == SS_OK);
  CHECK(ss_pmg_setup(pmg, fine, 1U, 1U, &setup) == SS_OK);
  CHECK(ss_pmg_apply(pmg, rhs, correction, &apply) == SS_OK);
  CHECK(apply.status == SS_OK);
  CHECK(apply.level_solve_calls == 3U);
  CHECK(apply.operator_apply_calls == 1U);
  CHECK(ss_pmg_destroy(&pmg) == SS_OK);
  specification.fgmres_full_iterate_post = 0;

  /*
   * A bounded inner Krylov solve is an inexact preconditioner application.
   * Reaching its iteration budget with a finite improving iterate must not
   * abort a flexible outer Krylov solve.
   */
  level_policy(&inexact_policy, SS_PMG_LEVEL_FGMRES_JACOBI, "pmgi_");
  inexact_policy.active_ranks = active_rank;
  inexact_policy.maximum_iterations = 1;
  inexact_policy.rtol = 1.0e-14;
  inexact_policy.atol = 1.0e-50;
  memset(&inexact_specification, 0, sizeof(inexact_specification));
  inexact_specification.fine_order = 1U;
  inexact_specification.level_count = 1U;
  inexact_specification.orders = one_level_order;
  inexact_specification.level_policies = &inexact_policy;
  inexact_specification.transfer_generation = 8U;
  inexact_specification.numeric_lag_interval = 1U;
  inexact_specification.memory_cap_bytes = 16U * 1024U * 1024U;
  CHECK(ss_pmg_create(PETSC_COMM_SELF, &inexact_specification, &inexact_pmg) == SS_OK);
  CHECK(ss_pmg_setup(inexact_pmg, fine, 1U, 1U, &setup) == SS_OK);
  CHECK(ss_pmg_apply(inexact_pmg, rhs, correction, &apply) == SS_OK);
  CHECK(apply.status == SS_OK);
  CHECK(apply.last_level_reason == KSP_DIVERGED_ITS);
  CHECK(apply.last_level_iterations == 1);
  CHECK(VecDuplicate(rhs, &residual) == PETSC_SUCCESS);
  CHECK(VecNorm(rhs, NORM_2, &rhs_norm) == PETSC_SUCCESS);
  CHECK(MatMult(fine, correction, residual) == PETSC_SUCCESS);
  CHECK(VecAYPX(residual, -1.0, rhs) == PETSC_SUCCESS);
  CHECK(VecNorm(residual, NORM_2, &residual_norm) == PETSC_SUCCESS);
  CHECK(isfinite((double)residual_norm) && residual_norm < rhs_norm);
  CHECK(ss_pmg_destroy(&inexact_pmg) == SS_OK);
  CHECK(VecDestroy(&residual) == PETSC_SUCCESS);
  CHECK(VecDestroy(&correction) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(MatDestroy(&prolongation) == PETSC_SUCCESS);
  CHECK(MatDestroy(&fine) == PETSC_SUCCESS);
  CHECK(test_three_level_cycle() == 0);
  CHECK(test_chebyshev_wide_spectrum() == 0);
  return 0;
}

static int check_nullspace(const SsPmgNearNullspaceInput *input, PetscInt dofs,
                           size_t expected_retained, size_t expected_dropped) {
  Mat operator_handle = NULL;
  MatNullSpace near_nullspace = NULL;
  SsPmgNullspaceRecord record = {0};
  PetscInt row;
  CHECK(MatCreateSeqAIJ(PETSC_COMM_SELF, dofs, dofs, 1, NULL, &operator_handle) ==
        PETSC_SUCCESS);
  for (row = 0; row < dofs; ++row) {
    CHECK(MatSetValue(operator_handle, row, row, 1.0, INSERT_VALUES) == PETSC_SUCCESS);
  }
  CHECK(MatAssemblyBegin(operator_handle, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(MatAssemblyEnd(operator_handle, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(ss_pmg_near_nullspace_create(operator_handle, input, &near_nullspace,
                                     &record) == SS_OK);
  CHECK(record.retained_mode_count == expected_retained);
  CHECK(record.dropped_mode_count == expected_dropped);
  CHECK(record.orthogonality_defect <= 2.0e-12);
  CHECK((expected_retained == 0U && near_nullspace == NULL) ||
        (expected_retained != 0U && near_nullspace != NULL));
  CHECK(MatNullSpaceDestroy(&near_nullspace) == PETSC_SUCCESS);
  CHECK(MatDestroy(&operator_handle) == PETSC_SUCCESS);
  return 0;
}

static int test_nullspace(void) {
  static const PetscReal coordinates_2d[] = {
      0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
  };
  static const PetscInt dofs_2d[] = {0, 1, 2, 3, 4, 5};
  static const PetscInt constrained_2d[] = {-1, -1, -1, -1, 4, 5};
  static const PetscReal coordinates_3d[] = {
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
  };
  static const PetscInt dofs_3d[] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
  };
  static const PetscReal hydraulic_coordinates[] = {
      0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
  };
  static const PetscInt hydraulic_dofs[] = {0, 1, 2};
  SsPmgNearNullspaceInput input = {
      SS_PMG_FIELD_MECHANICS, 2U, 0, 3U, coordinates_2d, dofs_2d, 1.0e-12,
  };
  CHECK(check_nullspace(&input, 6, 3U, 0U) == 0);
  input.global_dof_indices = constrained_2d;
  CHECK(check_nullspace(&input, 6, 2U, 1U) == 0);
  input = (SsPmgNearNullspaceInput){
      SS_PMG_FIELD_MECHANICS, 3U, 0, 4U, coordinates_3d, dofs_3d, 1.0e-12,
  };
  CHECK(check_nullspace(&input, 12, 6U, 0U) == 0);
  input = (SsPmgNearNullspaceInput){
      SS_PMG_FIELD_HYDRAULIC, 2U, 1, 3U, hydraulic_coordinates, hydraulic_dofs, 1.0e-12,
  };
  CHECK(check_nullspace(&input, 3, 1U, 0U) == 0);
  input.hydraulic_pure_neumann = 0;
  CHECK(check_nullspace(&input, 3, 0U, 0U) == 0);
  {
    static const uint32_t order = 1U;
    static const double identity[] = {
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    };
    SsPmgLevelPolicy policy;
    SsPmgSpecification specification = {0};
    SsPmgSetupRecord setup = {0};
    SsPmg *pmg = NULL;
    Mat matrix = NULL;
    Mat level_operator = NULL;
    MatNullSpace attached = NULL;
    PetscBool has_constant = PETSC_FALSE;
    PetscInt mode_count = 0;
    const Vec *modes = NULL;
    input = (SsPmgNearNullspaceInput){
        SS_PMG_FIELD_MECHANICS, 2U, 0, 3U, coordinates_2d, dofs_2d, 1.0e-12,
    };
    CHECK(create_dense(6, 6, identity, &matrix) == 0);
    level_policy(&policy, SS_PMG_LEVEL_PREONLY_LU, "pmgns_");
    specification.fine_order = 1U;
    specification.level_count = 1U;
    specification.orders = &order;
    specification.level_policies = &policy;
    specification.near_nullspaces = &input;
    specification.transfer_generation = 1U;
    specification.numeric_lag_interval = 1U;
    specification.memory_cap_bytes = 4U * 1024U * 1024U;
    CHECK(ss_pmg_create(PETSC_COMM_SELF, &specification, &pmg) == SS_OK);
    CHECK(ss_pmg_setup(pmg, matrix, 1U, 1U, &setup) == SS_OK);
    CHECK(setup.levels[0].retained_near_nullspace_modes == 3U);
    CHECK(setup.levels[0].variable_point_block_count == 3U);
    CHECK(ss_pmg_level_operator(pmg, 0U, &level_operator) == SS_OK);
    CHECK(MatGetNearNullSpace(level_operator, &attached) == PETSC_SUCCESS);
    CHECK(attached != NULL);
    CHECK(MatNullSpaceGetVecs(attached, &has_constant, &mode_count, &modes) ==
          PETSC_SUCCESS);
    CHECK(has_constant == PETSC_FALSE && mode_count == 3 && modes != NULL);
    CHECK(ss_pmg_destroy(&pmg) == SS_OK);
    {
      static const PetscInt entity_block[] = {6};
      CHECK(MatSetVariableBlockSizes(matrix, 1, entity_block) == PETSC_SUCCESS);
      CHECK(ss_pmg_create(PETSC_COMM_SELF, &specification, &pmg) == SS_OK);
      CHECK(ss_pmg_setup(pmg, matrix, 1U, 1U, &setup) == SS_OK);
      CHECK(setup.levels[0].retained_near_nullspace_modes == 3U);
      CHECK(setup.levels[0].variable_point_block_count == 1U);
      CHECK(ss_pmg_destroy(&pmg) == SS_OK);
    }
    CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  }
  return 0;
}

int main(int argc, char **argv) {
  const char *test_id;
  int result;
  if (argc != 2) {
    fprintf(stderr, "usage: %s TEST-ID\n", argv[0]);
    return 2;
  }
  test_id = argv[1];
  CHECK(PetscInitialize(&argc, &argv, NULL, NULL) == PETSC_SUCCESS);
  if (strcmp(test_id, "PMG-UNIT-HIERARCHY-001") == 0) {
    result = test_hierarchy();
  } else if (strcmp(test_id, "PMG-UNIT-GALERKIN-001") == 0) {
    result = test_galerkin();
    if (result == 0) {
      result = test_external_level_operators();
    }
  } else if (strcmp(test_id, "PMG-UNIT-VCYCLE-001") == 0) {
    result = test_vcycle();
  } else if (strcmp(test_id, "PMG-UNIT-NULLSPACE-001") == 0) {
    result = test_nullspace();
  } else {
    fprintf(stderr, "unknown test id: %s\n", test_id);
    result = 2;
  }
  CHECK(PetscFinalize() == PETSC_SUCCESS);
  return result;
}
