/* Linear-solver, collective Krylov cancellation and deflation contracts
 * across MPI rank counts.
 *
 * Launched by tests/native/support/mpi_layout_runner.py --suite linear.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/core/sha256.h"
#include "slope_internal/deflation/deflation.h"
#include "slope_internal/linear/linear.h"

#include <petscksp.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GLOBAL_SIZE 8

#define CHECK(expression)                                                              \
  do {                                                                                 \
    if (!(expression)) {                                                               \
      int check_rank = -1;                                                             \
      (void)MPI_Comm_rank(PETSC_COMM_WORLD, &check_rank);                              \
      fprintf(stderr, "rank %d %s:%d: assertion failed: %s\n", check_rank, __FILE__,   \
              __LINE__, #expression);                                                  \
      (void)MPI_Abort(PETSC_COMM_WORLD, 1);                                            \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

static int parse_arguments(int argc, char **argv, const char **test_id,
                           int *expected_ranks, const char **output) {
  int index;
  *test_id = NULL;
  *expected_ranks = 0;
  *output = NULL;
  for (index = 1; index + 1 < argc; index += 2) {
    if (strcmp(argv[index], "--test-id") == 0) {
      *test_id = argv[index + 1];
    } else if (strcmp(argv[index], "--expected-ranks") == 0) {
      *expected_ranks = atoi(argv[index + 1]);
    } else if (strcmp(argv[index], "--output") == 0) {
      *output = argv[index + 1];
    } else {
      return 1;
    }
  }
  return *test_id == NULL || *expected_ranks <= 0 || *output == NULL;
}

static int publish_result(const char *path, const char *test_id, int ranks,
                          const char *invariant, double checksum, double maximum_error,
                          uint64_t global_reductions) {
  FILE *stream = fopen(path, "wb");
  int written;
  CHECK(stream != NULL);
  written = fprintf(stream,
                    "{\"checksum\":%.17g,\"global_reductions\":%llu,"
                    "\"invariant\":\"sha256:%s\",\"maximum_error\":%.17g,"
                    "\"ranks\":%d,\"test_id\":\"%s\"}\n",
                    checksum, (unsigned long long)global_reductions, invariant,
                    maximum_error, ranks, test_id);
  if (written <= 0) {
    (void)fclose(stream);
    return 1;
  }
  if (fclose(stream) != 0) {
    return 1;
  }
  return 0;
}

static int create_diagonal_operator(Mat *matrix) {
  PetscInt start = 0;
  PetscInt end = 0;
  PetscInt row;
  CHECK(matrix != NULL);
  CHECK(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, GLOBAL_SIZE,
                     GLOBAL_SIZE, 1, NULL, 0, NULL, matrix) == PETSC_SUCCESS);
  CHECK(MatGetOwnershipRange(*matrix, &start, &end) == PETSC_SUCCESS);
  for (row = start; row < end; ++row) {
    const PetscScalar value = (PetscScalar)((double)row + 2.0);
    CHECK(MatSetValue(*matrix, row, row, value, INSERT_VALUES) == PETSC_SUCCESS);
  }
  CHECK(MatAssemblyBegin(*matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(MatAssemblyEnd(*matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  return 0;
}

static int create_rhs_and_exact(Mat matrix, Vec *rhs, Vec *exact) {
  PetscInt start = 0;
  PetscInt end = 0;
  PetscInt row;
  CHECK(matrix != NULL && rhs != NULL && exact != NULL);
  CHECK(MatCreateVecs(matrix, exact, rhs) == PETSC_SUCCESS);
  CHECK(VecGetOwnershipRange(*exact, &start, &end) == PETSC_SUCCESS);
  for (row = start; row < end; ++row) {
    const PetscScalar value = (PetscScalar)((double)row + 1.0);
    CHECK(VecSetValue(*exact, row, value, INSERT_VALUES) == PETSC_SUCCESS);
  }
  CHECK(VecAssemblyBegin(*exact) == PETSC_SUCCESS);
  CHECK(VecAssemblyEnd(*exact) == PETSC_SUCCESS);
  CHECK(MatMult(matrix, *exact, *rhs) == PETSC_SUCCESS);
  return 0;
}

static SsDeflationPolicy deflation_policy(void) {
  const SsDeflationPolicy policy = {
      .dimension_cap = 4U,
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

static int create_deflation(Mat matrix, SsDeflation **deflation) {
  SsDeflationAdmissionRecord admission = {0};
  SsDeflationPolicy policy = deflation_policy();
  Vec left = NULL;
  Vec right = NULL;
  PetscInt start = 0;
  PetscInt end = 0;
  CHECK(matrix != NULL && deflation != NULL);
  CHECK(MatCreateVecs(matrix, &right, &left) == PETSC_SUCCESS);
  CHECK(VecGetOwnershipRange(right, &start, &end) == PETSC_SUCCESS);
  if (start == 0) {
    CHECK(VecSetValue(right, 0, 1.0, INSERT_VALUES) == PETSC_SUCCESS);
    CHECK(VecSetValue(left, 0, 1.0, INSERT_VALUES) == PETSC_SUCCESS);
  }
  CHECK(VecAssemblyBegin(right) == PETSC_SUCCESS);
  CHECK(VecAssemblyEnd(right) == PETSC_SUCCESS);
  CHECK(VecAssemblyBegin(left) == PETSC_SUCCESS);
  CHECK(VecAssemblyEnd(left) == PETSC_SUCCESS);
  CHECK(ss_deflation_create(PETSC_COMM_WORLD, &policy, deflation) == SS_OK);
  CHECK(ss_deflation_begin(*deflation, 41U) == SS_OK);
  CHECK(ss_deflation_add_candidate(*deflation, left, right, "MPI-e0", &admission) ==
        SS_OK);
  CHECK(admission.admitted != 0);
  CHECK(ss_deflation_refresh(*deflation, matrix, 42U) == SS_OK);
  CHECK(VecDestroy(&left) == PETSC_SUCCESS);
  CHECK(VecDestroy(&right) == PETSC_SUCCESS);
  return 0;
}

static SsLinearPolicy linear_policy(void) {
  SsLinearPolicy policy = {0};
  (void)snprintf(policy.identity, sizeof(policy.identity), "%s", "mpi-fgmres");
  (void)snprintf(policy.options_prefix, sizeof(policy.options_prefix), "%s", "linmpi_");
  policy.operator_class = SS_LINEAR_OPERATOR_GENERAL;
  policy.method = SS_LINEAR_METHOD_FGMRES;
  policy.pc = SS_LINEAR_PC_JACOBI;
  policy.side = SS_LINEAR_SIDE_RIGHT;
  policy.rtol = 1.0e-12;
  policy.atol = 1.0e-14;
  policy.dtol = 1.0e6;
  policy.maximum_iterations = 32;
  policy.restart = 8;
  policy.vector_memory_cap_bytes = 1024U * 1024U;
  return policy;
}

static SsLinearOperatorProperties linear_properties(void) {
  const SsLinearOperatorProperties properties = {
      1, 0, 0, 1, 1, 0, 0,
  };
  return properties;
}

static int test_linear(double *checksum, double *maximum_error,
                       uint64_t *global_reductions,
                       char invariant[SS_SHA256_HEX_CAPACITY]) {
  Mat matrix = NULL;
  Vec rhs = NULL;
  Vec exact = NULL;
  Vec solution = NULL;
  Vec error = NULL;
  SsDeflation *deflation = NULL;
  SsLinearSolver *solver = NULL;
  SsLinearProblem problem = {0};
  SsLinearSetupRecord setup = {0};
  SsLinearSolveRecord solve = {0};
  SsDeflationDiagnostics diagnostics = {0};
  PetscReal error_norm = 0.0;
  PetscReal normal_error_norm = 0.0;
  PetscScalar solution_sum = 0.0;
  PetscScalar normal_sum = 0.0;
  char payload[256];
  int count;

  CHECK(create_diagonal_operator(&matrix) == 0);
  CHECK(create_rhs_and_exact(matrix, &rhs, &exact) == 0);
  CHECK(VecDuplicate(exact, &solution) == PETSC_SUCCESS);
  CHECK(VecDuplicate(exact, &error) == PETSC_SUCCESS);
  problem.operator_handle = matrix;
  problem.properties = linear_properties();
  problem.policy_chain.count = 1U;
  problem.policy_chain.policies[0] = linear_policy();
  problem.structure_generation = 42U;
  problem.value_generation = 43U;

  CHECK(ss_linear_create(PETSC_COMM_WORLD, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
  CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED);
  CHECK(solve.reason > 0);
  CHECK(solve.true_residual_norm <= solve.acceptance_threshold);
  CHECK(VecWAXPY(error, -1.0, exact, solution) == PETSC_SUCCESS);
  CHECK(VecNorm(error, NORM_2, &normal_error_norm) == PETSC_SUCCESS);
  CHECK(VecSum(solution, &normal_sum) == PETSC_SUCCESS);
  CHECK(normal_error_norm <= 2.0e-11);
  CHECK(fabs(PetscRealPart(normal_sum) - 36.0) <= 2.0e-11);
  CHECK(ss_linear_destroy(&solver) == SS_OK);

  CHECK(VecSet(solution, 0.0) == PETSC_SUCCESS);
  CHECK(create_deflation(matrix, &deflation) == 0);
  CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
  {
    const uint64_t validated_reductions = diagnostics.global_reductions;
    problem.value_generation = 42U;
    memset(&setup, 0, sizeof(setup));
    memset(&solve, 0, sizeof(solve));
    problem.deflation = deflation;
    CHECK(ss_linear_create(PETSC_COMM_WORLD, &solver) == SS_OK);
    CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
    CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
    CHECK(diagnostics.global_reductions == validated_reductions);
  }
  CHECK(setup.attempt_count == 1U);
  CHECK(strcmp(setup.effective_ksp, "fgmres") == 0);
  CHECK(setup.effective_side == SS_LINEAR_SIDE_RIGHT);
  CHECK(strcmp(setup.effective_orthogonalization, "classical_gram_schmidt_twice") == 0);
  CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
  CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED);
  CHECK(solve.reason > 0);
  CHECK(solve.true_residual_norm <= solve.acceptance_threshold);
  CHECK(VecWAXPY(error, -1.0, exact, solution) == PETSC_SUCCESS);
  CHECK(VecNorm(error, NORM_2, &error_norm) == PETSC_SUCCESS);
  CHECK(VecSum(solution, &solution_sum) == PETSC_SUCCESS);
  CHECK(error_norm <= 2.0e-11);
  CHECK(fabs(PetscRealPart(solution_sum) - 36.0) <= 2.0e-11);
  CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
  CHECK(diagnostics.invariants_validated != 0);
  CHECK(diagnostics.operator_state >= 0);
  CHECK(diagnostics.global_reductions > 0U);
  count = snprintf(payload, sizeof(payload),
                   "linear|normal=%lld|projected=%lld|converged=%d|"
                   "ksp=fgmres",
                   (long long)llround(PetscRealPart(normal_sum) * 1.0e12),
                   (long long)llround(PetscRealPart(solution_sum) * 1.0e12),
                   solve.status == SS_LINEAR_SOLVE_CONVERGED);
  CHECK(count > 0 && (size_t)count < sizeof(payload));
  CHECK(ss_sha256_bytes((const unsigned char *)payload, (size_t)count, invariant) ==
        SS_OK);
  *checksum = PetscRealPart(normal_sum) + PetscRealPart(solution_sum);
  *maximum_error =
      (double)(normal_error_norm > error_norm ? normal_error_norm : error_norm);
  *global_reductions = diagnostics.global_reductions;
  CHECK(ss_linear_destroy(&solver) == SS_OK);
  CHECK(ss_deflation_destroy(&deflation) == SS_OK);
  CHECK(VecDestroy(&error) == PETSC_SUCCESS);
  CHECK(VecDestroy(&solution) == PETSC_SUCCESS);
  CHECK(VecDestroy(&exact) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}

static int create_spd_oracle_operator(Mat *matrix) {
  static const double values[3][3] = {
      {4.0, 1.0, 0.0},
      {1.0, 3.0, 1.0},
      {0.0, 1.0, 2.0},
  };
  PetscInt start = 0;
  PetscInt end = 0;
  PetscInt row;
  CHECK(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 3, 3, 3, NULL, 3,
                     NULL, matrix) == PETSC_SUCCESS);
  CHECK(MatGetOwnershipRange(*matrix, &start, &end) == PETSC_SUCCESS);
  for (row = start; row < end; ++row) {
    PetscInt column;
    for (column = 0; column < 3; ++column) {
      if (values[row][column] != 0.0) {
        CHECK(MatSetValue(*matrix, row, column, (PetscScalar)values[row][column],
                          INSERT_VALUES) == PETSC_SUCCESS);
      }
    }
  }
  CHECK(MatAssemblyBegin(*matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(MatAssemblyEnd(*matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  return 0;
}

static int test_deflation(double *checksum, double *maximum_error,
                          uint64_t *global_reductions,
                          char invariant[SS_SHA256_HEX_CAPACITY]) {
  Mat matrix = NULL;
  Vec rhs = NULL;
  Vec basis = NULL;
  Vec q = NULL;
  Vec right = NULL;
  Vec left = NULL;
  SsDeflation *deflation = NULL;
  SsDeflationPolicy policy = deflation_policy();
  SsDeflationAdmissionRecord admission = {0};
  SsDeflationDiagnostics diagnostics = {0};
  PetscReal projector_error = 0.0;
  uint64_t reductions_before_refresh = 0U;
  PetscScalar q_sum = 0.0;
  PetscScalar right_sum = 0.0;
  PetscScalar left_sum = 0.0;
  char payload[256];
  int count;
  PetscInt start = 0;
  PetscInt end = 0;
  PetscInt row;

  CHECK(create_spd_oracle_operator(&matrix) == 0);
  CHECK(MatCreateVecs(matrix, &basis, &rhs) == PETSC_SUCCESS);
  CHECK(VecGetOwnershipRange(rhs, &start, &end) == PETSC_SUCCESS);
  for (row = start; row < end; ++row) {
    CHECK(VecSetValue(rhs, row, (PetscScalar)((double)row + 1.0), INSERT_VALUES) ==
          PETSC_SUCCESS);
  }
  CHECK(VecAssemblyBegin(rhs) == PETSC_SUCCESS);
  CHECK(VecAssemblyEnd(rhs) == PETSC_SUCCESS);
  CHECK(VecDuplicate(rhs, &q) == PETSC_SUCCESS);
  CHECK(VecDuplicate(rhs, &right) == PETSC_SUCCESS);
  CHECK(VecDuplicate(rhs, &left) == PETSC_SUCCESS);
  CHECK(ss_deflation_create(PETSC_COMM_WORLD, &policy, &deflation) == SS_OK);
  CHECK(ss_deflation_begin(deflation, 51U) == SS_OK);
  for (row = 0; row < 2; ++row) {
    CHECK(VecSet(basis, 0.0) == PETSC_SUCCESS);
    if (start <= row && row < end) {
      CHECK(VecSetValue(basis, row, 1.0, INSERT_VALUES) == PETSC_SUCCESS);
    }
    CHECK(VecAssemblyBegin(basis) == PETSC_SUCCESS);
    CHECK(VecAssemblyEnd(basis) == PETSC_SUCCESS);
    CHECK(ss_deflation_add_candidate(deflation, basis, basis, "SOL-001-SPD",
                                     &admission) == SS_OK);
    CHECK(admission.admitted != 0);
  }
  CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
  reductions_before_refresh = diagnostics.global_reductions;
  CHECK(ss_deflation_refresh(deflation, matrix, 52U) == SS_OK);
  CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
  /*
   * Refresh validation is a small-coarse-space calculation.  Its collective
   * count must not grow with the number of projector identities or basis
   * probes; the exhaustive distributed oracle below remains explicit.
   */
  CHECK(diagnostics.global_reductions - reductions_before_refresh <= 1U);
  CHECK(ss_deflation_apply_q(deflation, rhs, q) == SS_OK);
  CHECK(ss_deflation_apply_right(deflation, matrix, rhs, right) == SS_OK);
  CHECK(ss_deflation_apply_left(deflation, matrix, rhs, left) == SS_OK);
  CHECK(ss_deflation_invariant_error(deflation, matrix, &projector_error) == SS_OK);
  CHECK(VecSum(q, &q_sum) == PETSC_SUCCESS);
  CHECK(VecSum(right, &right_sum) == PETSC_SUCCESS);
  CHECK(VecSum(left, &left_sum) == PETSC_SUCCESS);
  CHECK(fabs(PetscRealPart(q_sum) - 8.0 / 11.0) <= 2.0e-11);
  CHECK(fabs(PetscRealPart(right_sum) - 24.0 / 11.0) <= 2.0e-11);
  CHECK(fabs(PetscRealPart(left_sum) - 26.0 / 11.0) <= 2.0e-11);
  CHECK(projector_error <= 2.0e-11);
  CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
  CHECK(diagnostics.active_count == 2U);
  CHECK(diagnostics.cached_az_count == 2U);
  CHECK(diagnostics.invariants_validated != 0);
  CHECK(diagnostics.operator_state >= 0);
  CHECK(diagnostics.global_reductions > 0U);
  count = snprintf(
      payload, sizeof(payload), "deflation|q=%lld|pr=%lld|pl=%lld|active=%zu",
      (long long)llround(PetscRealPart(q_sum) * 1.0e12),
      (long long)llround(PetscRealPart(right_sum) * 1.0e12),
      (long long)llround(PetscRealPart(left_sum) * 1.0e12), diagnostics.active_count);
  CHECK(count > 0 && (size_t)count < sizeof(payload));
  CHECK(ss_sha256_bytes((const unsigned char *)payload, (size_t)count, invariant) ==
        SS_OK);
  *checksum = PetscRealPart(q_sum) + PetscRealPart(right_sum) + PetscRealPart(left_sum);
  *maximum_error = (double)projector_error;
  *global_reductions = diagnostics.global_reductions;
  CHECK(ss_deflation_destroy(&deflation) == SS_OK);
  CHECK(VecDestroy(&left) == PETSC_SUCCESS);
  CHECK(VecDestroy(&right) == PETSC_SUCCESS);
  CHECK(VecDestroy(&q) == PETSC_SUCCESS);
  CHECK(VecDestroy(&basis) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}

typedef struct {
  int trigger;
  int requested;
  PetscInt last_iteration;
} CancellationObservation;

static void request_cancel_in_ksp(void *context, PetscInt iteration, PetscReal norm) {
  CancellationObservation *observation = context;
  (void)norm;
  observation->last_iteration = iteration;
  if (observation->trigger && iteration == 3) {
    observation->requested = 1;
  }
}

static int poll_cancel_in_ksp(void *context) {
  return ((CancellationObservation *)context)->requested;
}

static int test_krylov_cancellation(int trigger, uint64_t *reductions) {
  Mat matrix = NULL;
  Vec rhs = NULL;
  Vec solution = NULL;
  SsLinearSolver *solver = NULL;
  SsLinearProblem problem = {
      .properties = {.finite_action = 1},
      .policy_chain =
          {
              .count = 1,
              .policies = {{
                  .identity = "cancel-fgmres",
                  .options_prefix = "cancel_test_",
                  .operator_class = SS_LINEAR_OPERATOR_GENERAL,
                  .method = SS_LINEAR_METHOD_FGMRES,
                  .pc = SS_LINEAR_PC_NONE,
                  .side = SS_LINEAR_SIDE_RIGHT,
                  .rtol = 1.0e-12,
                  .atol = 1.0e-16,
                  .dtol = 1.0e8,
                  .maximum_iterations = 100,
                  .restart = 64,
                  .vector_memory_cap_bytes = 1024U * 1024U,
              }},
          },
      .structure_generation = 1U,
      .value_generation = 1U,
  };
  SsLinearSetupRecord setup = {0};
  SsLinearSolveRecord solve = {0};
  CancellationObservation observation = {.trigger = trigger, .last_iteration = -1};
  const SsLinearObserver observer = {
      .krylov_iteration = request_cancel_in_ksp,
      .context = &observation,
  };
  PetscInt first = 0;
  PetscInt last = 0;
  CHECK(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 64, 64, 1, NULL, 0,
                     NULL, &matrix) == PETSC_SUCCESS);
  CHECK(MatGetOwnershipRange(matrix, &first, &last) == PETSC_SUCCESS);
  for (PetscInt row = first; row < last; ++row) {
    CHECK(MatSetValue(matrix, row, row, (PetscScalar)(row + 1), INSERT_VALUES) ==
          PETSC_SUCCESS);
  }
  CHECK(MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(MatCreateVecs(matrix, &solution, &rhs) == PETSC_SUCCESS);
  CHECK(VecSet(rhs, 1.0) == PETSC_SUCCESS);
  problem.operator_handle = matrix;
  CHECK(ss_linear_create(PETSC_COMM_WORLD, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  ss_linear_set_observer(solver, &observer);
  ss_linear_set_cancellation(solver, poll_cancel_in_ksp, &observation);
  CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_ERROR_CANCELLED);
  CHECK(solve.status == SS_LINEAR_SOLVE_CANCELLED && solve.reason == KSP_DIVERGED_USER);
  CHECK(solve.iterations == 10);
  CHECK(solve.cancellation_reductions == 1U);
  *reductions = solve.cancellation_reductions;
  CHECK(observation.last_iteration >= 9 && observation.last_iteration <= 10);
  ss_linear_set_observer(solver, NULL);
  observation.requested = 0;
  CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
  CHECK(solve.cancellation_reductions <= (uint64_t)solve.iterations / 10U);
  CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED && solve.iterations > 10);
  CHECK(solve.true_residual_norm <= solve.acceptance_threshold);
  CHECK(ss_linear_destroy(&solver) == SS_OK);
  CHECK(VecDestroy(&solution) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}

int main(int argc, char **argv) {
  const char *test_id = NULL;
  const char *output = NULL;
  int expected_ranks = 0;
  int rank = -1;
  int ranks = 0;
  int result;
  double checksum = 0.0;
  double maximum_error = 0.0;
  uint64_t global_reductions = 0U;
  char invariant[SS_SHA256_HEX_CAPACITY] = {0};

  CHECK(PetscInitialize(&argc, &argv, NULL, NULL) == PETSC_SUCCESS);
  CHECK(MPI_Comm_rank(PETSC_COMM_WORLD, &rank) == MPI_SUCCESS);
  CHECK(MPI_Comm_size(PETSC_COMM_WORLD, &ranks) == MPI_SUCCESS);
  CHECK(parse_arguments(argc, argv, &test_id, &expected_ranks, &output) == 0);
  CHECK(ranks == expected_ranks);
  if (strcmp(test_id, "LIN-CONTRACT-MPI-001") == 0) {
    result = test_linear(&checksum, &maximum_error, &global_reductions, invariant);
  } else if (strcmp(test_id, "LIN-CANCEL-001") == 0) {
    result = test_krylov_cancellation(rank == ranks - 1, &global_reductions);
    CHECK(ss_sha256_bytes((const unsigned char *)test_id, strlen(test_id), invariant) ==
          SS_OK);
  } else if (strcmp(test_id, "DEF-CONTRACT-MPI-001") == 0) {
    result = test_deflation(&checksum, &maximum_error, &global_reductions, invariant);
  } else {
    result = 2;
  }
  CHECK(result == 0);
  if (rank == 0) {
    CHECK(publish_result(output, test_id, ranks, invariant, checksum, maximum_error,
                         global_reductions) == 0);
  }
  CHECK(MPI_Barrier(PETSC_COMM_WORLD) == MPI_SUCCESS);
  CHECK(PetscFinalize() == PETSC_SUCCESS);
  return 0;
}
