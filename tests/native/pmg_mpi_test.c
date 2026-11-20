/* P-multigrid contracts across world sizes, active coarse ranks and threads.
 *
 * Launched by tests/native/support/mpi_layout_runner.py --suite pmg.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/fe/fe.h"
#include "slope_internal/linear/linear.h"
#include "slope_internal/pmg/pmg.h"

#include <petscksp.h>

#include <mpi.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

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

typedef struct {
  const char *test_id;
  const char *output;
  int expected_ranks;
  int active_count;
  int expected_threads;
} Arguments;

static int parse_arguments(int argc, char **argv, Arguments *arguments) {
  int index;
  memset(arguments, 0, sizeof(*arguments));
  arguments->expected_threads = 1;
  for (index = 1; index + 1 < argc; index += 2) {
    if (strcmp(argv[index], "--test-id") == 0) {
      arguments->test_id = argv[index + 1];
    } else if (strcmp(argv[index], "--expected-ranks") == 0) {
      arguments->expected_ranks = atoi(argv[index + 1]);
    } else if (strcmp(argv[index], "--active-count") == 0) {
      arguments->active_count = atoi(argv[index + 1]);
    } else if (strcmp(argv[index], "--expected-threads") == 0) {
      arguments->expected_threads = atoi(argv[index + 1]);
    } else if (strcmp(argv[index], "--output") == 0) {
      arguments->output = argv[index + 1];
    } else {
      return 1;
    }
  }
  return arguments->test_id == NULL || arguments->output == NULL ||
         arguments->expected_ranks <= 0 || arguments->active_count <= 0 ||
         arguments->expected_threads <= 0;
}

static int publish_result(const Arguments *arguments, double checksum,
                          double maximum_error, int active_owners,
                          int observed_threads) {
  int rank = -1;
  CHECK(MPI_Comm_rank(PETSC_COMM_WORLD, &rank) == MPI_SUCCESS);
  if (rank == 0) {
    FILE *stream = fopen(arguments->output, "wb");
    int written;
    CHECK(stream != NULL);
    written =
        fprintf(stream,
                "{\"active_count\":%d,\"active_owners\":%d,\"checksum\":%.17g,"
                "\"invariant\":\"sha256:"
                "8b27932f932de636d64bf3e2bfe6dd9a84a46fa12ad1dab2a3c3be56fa69d72e\","
                "\"maximum_error\":%.17g,\"observed_threads\":%d,\"ranks\":%d,"
                "\"requested_threads\":%d,\"test_id\":\"%s\"}\n",
                arguments->active_count, active_owners, checksum, maximum_error,
                observed_threads, arguments->expected_ranks,
                arguments->expected_threads, arguments->test_id);
    if (written <= 0) {
      (void)fclose(stream);
      CHECK(written > 0);
    }
    CHECK(fclose(stream) == 0);
  }
  return 0;
}

static int create_operator(PetscInt size, PetscReal scale, int nonsymmetric,
                           Mat *matrix) {
  PetscInt start = 0;
  PetscInt end = 0;
  PetscInt row;
  CHECK(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, size, size, 3, NULL,
                     3, NULL, matrix) == PETSC_SUCCESS);
  CHECK(MatGetOwnershipRange(*matrix, &start, &end) == PETSC_SUCCESS);
  for (row = start; row < end; ++row) {
    CHECK(MatSetValue(*matrix, row, row, (PetscScalar)(4.0 * scale), INSERT_VALUES) ==
          PETSC_SUCCESS);
    if (row > 0) {
      const PetscReal value = nonsymmetric != 0 ? -1.25 * scale : -1.0 * scale;
      CHECK(MatSetValue(*matrix, row, row - 1, (PetscScalar)value, INSERT_VALUES) ==
            PETSC_SUCCESS);
    }
    if (row + 1 < size) {
      const PetscReal value = nonsymmetric != 0 ? -0.75 * scale : -1.0 * scale;
      CHECK(MatSetValue(*matrix, row, row + 1, (PetscScalar)value, INSERT_VALUES) ==
            PETSC_SUCCESS);
    }
  }
  CHECK(MatAssemblyBegin(*matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(MatAssemblyEnd(*matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  return 0;
}

static int create_diagonal(PetscInt size, Mat *matrix) {
  PetscInt start = 0;
  PetscInt end = 0;
  PetscInt row;
  CHECK(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, size, size, 1, NULL,
                     0, NULL, matrix) == PETSC_SUCCESS);
  CHECK(MatGetOwnershipRange(*matrix, &start, &end) == PETSC_SUCCESS);
  for (row = start; row < end; ++row) {
    CHECK(MatSetValue(*matrix, row, row, (PetscScalar)(2.0 + (double)(row % 7)),
                      INSERT_VALUES) == PETSC_SUCCESS);
  }
  CHECK(MatAssemblyBegin(*matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(MatAssemblyEnd(*matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  return 0;
}

static int create_transfer(PetscInt coarse_size, Mat *prolongation) {
  const PetscInt fine_size = 2 * coarse_size - 1;
  PetscInt start = 0;
  PetscInt end = 0;
  PetscInt row;
  CHECK(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, fine_size,
                     coarse_size, 2, NULL, 2, NULL, prolongation) == PETSC_SUCCESS);
  CHECK(MatGetOwnershipRange(*prolongation, &start, &end) == PETSC_SUCCESS);
  for (row = start; row < end; ++row) {
    if (row % 2 == 0) {
      const PetscInt column = row / 2;
      CHECK(MatSetValue(*prolongation, row, column, 1.0, INSERT_VALUES) ==
            PETSC_SUCCESS);
    } else {
      const PetscInt columns[] = {row / 2, row / 2 + 1};
      const PetscScalar values[] = {0.5, 0.5};
      CHECK(MatSetValues(*prolongation, 1, &row, 2, columns, values, INSERT_VALUES) ==
            PETSC_SUCCESS);
    }
  }
  CHECK(MatAssemblyBegin(*prolongation, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(MatAssemblyEnd(*prolongation, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  return 0;
}

static int create_simplex_transfer(uint32_t dimension, uint32_t coarse_order,
                                   uint32_t fine_order, uint32_t components,
                                   Mat *prolongation) {
  const SsFeDescriptor *coarse = NULL;
  const SsFeDescriptor *fine = NULL;
  double local[SS_FE_MAX_NODES * SS_FE_MAX_NODES] = {0.0};
  size_t local_rows = 0U;
  size_t local_columns = 0U;
  PetscInt global_rows;
  PetscInt global_columns;
  PetscInt row_start = 0;
  PetscInt row_end = 0;
  PetscInt row;
  CHECK(ss_fe_descriptor_find(dimension, coarse_order, &coarse) == SS_OK);
  CHECK(ss_fe_descriptor_find(dimension, fine_order, &fine) == SS_OK);
  CHECK(ss_fe_transfer_build(coarse, fine, local, SS_FE_MAX_NODES * SS_FE_MAX_NODES,
                             &local_rows, &local_columns) == SS_OK);
  CHECK(local_rows == fine->node_count && local_columns == coarse->node_count);
  global_rows = (PetscInt)(local_rows * (size_t)components);
  global_columns = (PetscInt)(local_columns * (size_t)components);
  CHECK(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, global_rows,
                     global_columns, (PetscInt)local_columns, NULL,
                     (PetscInt)local_columns, NULL, prolongation) == PETSC_SUCCESS);
  CHECK(MatGetOwnershipRange(*prolongation, &row_start, &row_end) == PETSC_SUCCESS);
  for (row = row_start; row < row_end; ++row) {
    const size_t fine_node = (size_t)row / (size_t)components;
    const uint32_t component = (uint32_t)((size_t)row % (size_t)components);
    size_t coarse_node;
    for (coarse_node = 0U; coarse_node < local_columns; ++coarse_node) {
      const double value = local[fine_node * local_columns + coarse_node];
      if (value != 0.0) {
        const PetscInt column =
            (PetscInt)(coarse_node * (size_t)components + component);
        CHECK(MatSetValue(*prolongation, row, column, (PetscScalar)value,
                          INSERT_VALUES) == PETSC_SUCCESS);
      }
    }
  }
  CHECK(MatAssemblyBegin(*prolongation, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(MatAssemblyEnd(*prolongation, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  return 0;
}

static void configure_level(SsPmgLevelPolicy *policy, SsPmgLevelMethod method,
                            const char *prefix, size_t active_count) {
  memset(policy, 0, sizeof(*policy));
  (void)snprintf(policy->options_prefix, sizeof(policy->options_prefix), "%s", prefix);
  policy->method = method;
  policy->active_rank_count = active_count;
  policy->pre_smooth_steps = method == SS_PMG_LEVEL_FGMRES_JACOBI ? 0 : 1;
  policy->post_smooth_steps = method == SS_PMG_LEVEL_FGMRES_JACOBI ? 0 : 1;
  policy->maximum_iterations = 64;
  policy->richardson_scale = 0.2;
  policy->rtol = 1.0e-13;
  policy->atol = 1.0e-15;
}

static SsLinearPolicy outer_policy(void) {
  SsLinearPolicy policy = {0};
  (void)snprintf(policy.identity, sizeof(policy.identity), "%s",
                 "production-mechanics-pmg");
  (void)snprintf(policy.options_prefix, sizeof(policy.options_prefix), "%s",
                 "pmgouter_");
  policy.operator_class = SS_LINEAR_OPERATOR_GENERAL;
  policy.method = SS_LINEAR_METHOD_FGMRES;
  policy.pc = SS_LINEAR_PC_PMG;
  policy.side = SS_LINEAR_SIDE_RIGHT;
  policy.rtol = 1.0e-10;
  policy.atol = 1.0e-13;
  policy.dtol = 1.0e8;
  policy.maximum_iterations = 100;
  policy.restart = 20;
  policy.vector_memory_cap_bytes = 64U * 1024U * 1024U;
  return policy;
}

static int exercise_representative_profile(uint32_t dimension, uint32_t fine_order,
                                           uint32_t components, int nonsymmetric,
                                           size_t active_count, double *checksum,
                                           double *maximum_error) {
  const SsFeDescriptor *fine_descriptor = NULL;
  SsPmgHierarchy hierarchy = {0};
  uint32_t orders[SS_PMG_MAX_LEVELS] = {0U, 0U, 0U};
  Mat transfers[SS_PMG_MAX_LEVELS - 1U] = {NULL, NULL};
  SsPmgLevelPolicy policies[SS_PMG_MAX_LEVELS];
  SsPmgSpecification specification = {0};
  SsPmgSetupRecord pmg_setup = {0};
  SsPmgSetupRecord reused = {0};
  SsPmgDiagnostics before = {0};
  SsPmgDiagnostics after = {0};
  SsPmg *pmg = NULL;
  SsLinearSolver *solver = NULL;
  SsLinearProblem problem = {0};
  SsLinearSetupRecord linear_setup = {0};
  SsLinearSolveRecord solve = {0};
  Mat fine = NULL;
  Vec exact = NULL;
  Vec rhs = NULL;
  Vec solution = NULL;
  Vec error = NULL;
  PetscReal error_norm = 0.0;
  PetscScalar solution_sum = 0.0;
  size_t level;
  int repeat;

  CHECK(ss_fe_descriptor_find(dimension, fine_order, &fine_descriptor) == SS_OK);
  CHECK(ss_pmg_hierarchy_resolve(fine_order, &hierarchy) == SS_OK);
  memcpy(orders, hierarchy.orders, sizeof(orders));
  for (level = 0U; level + 1U < hierarchy.level_count; ++level) {
    CHECK(create_simplex_transfer(dimension, hierarchy.orders[level],
                                  hierarchy.orders[level + 1U], components,
                                  &transfers[level]) == 0);
  }
  configure_level(&policies[0], SS_PMG_LEVEL_FGMRES_JACOBI, "pmgrpc_", active_count);
  for (level = 1U; level < hierarchy.level_count; ++level) {
    configure_level(&policies[level], SS_PMG_LEVEL_RICHARDSON_JACOBI,
                    level == 1U ? "pmgrm_" : "pmgrf_", active_count);
    policies[level].richardson_scale = 0.2;
  }
  specification.fine_order = fine_order;
  specification.level_count = hierarchy.level_count;
  specification.orders = orders;
  specification.prolongations = transfers;
  specification.level_policies = policies;
  specification.transfer_generation =
      UINT64_C(100) + (uint64_t)dimension * 10U + fine_order;
  specification.numeric_lag_interval = 1U;
  specification.memory_cap_bytes = 128U * 1024U * 1024U;
  CHECK(create_operator((PetscInt)(fine_descriptor->node_count * (size_t)components),
                        1.0, nonsymmetric, &fine) == 0);
  CHECK(ss_pmg_create(PETSC_COMM_WORLD, &specification, &pmg) == SS_OK);
  CHECK(ss_pmg_setup(pmg, fine, 1U, 1U, &pmg_setup) == SS_OK);
  CHECK(pmg_setup.hierarchy.level_count == hierarchy.level_count);
  CHECK(ss_pmg_diagnostics(pmg, &before) == SS_OK);
  problem.operator_handle = fine;
  problem.properties = (SsLinearOperatorProperties){
      1, 0, 0, 0, 0, 1, 0,
  };
  problem.policy_chain.count = 1U;
  problem.policy_chain.policies[0] = outer_policy();
  problem.pmg = pmg;
  problem.structure_generation = 1U;
  problem.value_generation = 1U;
  CHECK(ss_linear_create(PETSC_COMM_WORLD, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &linear_setup) == SS_OK);
  CHECK(linear_setup.pmg_selected != 0);
  CHECK(MatCreateVecs(fine, &exact, &rhs) == PETSC_SUCCESS);
  CHECK(VecDuplicate(exact, &solution) == PETSC_SUCCESS);
  CHECK(VecDuplicate(exact, &error) == PETSC_SUCCESS);
  CHECK(VecSet(exact, 1.0) == PETSC_SUCCESS);
  CHECK(MatMult(fine, exact, rhs) == PETSC_SUCCESS);
  for (repeat = 0; repeat < (fine_order == 4U ? 16 : 2); ++repeat) {
    CHECK(VecSet(solution, 0.0) == PETSC_SUCCESS);
    CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
    CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED);
    CHECK(ss_pmg_setup(pmg, fine, 1U, 1U, &reused) == SS_OK);
    CHECK(reused.hierarchy_reused != 0);
  }
  CHECK(VecWAXPY(error, -1.0, exact, solution) == PETSC_SUCCESS);
  CHECK(VecNorm(error, NORM_2, &error_norm) == PETSC_SUCCESS);
  CHECK(VecSum(solution, &solution_sum) == PETSC_SUCCESS);
  CHECK(error_norm <= 1.0e-8);
  CHECK(ss_pmg_diagnostics(pmg, &after) == SS_OK);
  CHECK(after.current_matrix_objects == before.current_matrix_objects);
  CHECK(after.current_vector_objects == before.current_vector_objects);
  CHECK(after.peak_matrix_objects == before.peak_matrix_objects);
  CHECK(after.peak_vector_objects == before.peak_vector_objects);
  CHECK(after.preconditioner_apply_calls > before.preconditioner_apply_calls);
  CHECK(after.contract_apply_calls == before.contract_apply_calls);
  CHECK(after.norm_validation_reductions == before.norm_validation_reductions);
  CHECK(after.level_status_reductions == before.level_status_reductions);
  *checksum += PetscRealPart(solution_sum);
  if ((double)error_norm > *maximum_error) {
    *maximum_error = (double)error_norm;
  }
  CHECK(ss_linear_destroy(&solver) == SS_OK);
  CHECK(ss_pmg_destroy(&pmg) == SS_OK);
  CHECK(VecDestroy(&error) == PETSC_SUCCESS);
  CHECK(VecDestroy(&solution) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(VecDestroy(&exact) == PETSC_SUCCESS);
  CHECK(MatDestroy(&fine) == PETSC_SUCCESS);
  for (level = 0U; level + 1U < hierarchy.level_count; ++level) {
    CHECK(MatDestroy(&transfers[level]) == PETSC_SUCCESS);
  }
  return 0;
}

static int test_active(const Arguments *arguments, double *checksum,
                       double *maximum_error, int *active_owners) {
  const PetscInt global_size = 192;
  const uint32_t order = 1U;
  SsPmgLevelPolicy policy;
  SsPmgNearNullspaceInput nullspace = {0};
  SsPmgSpecification specification = {0};
  SsPmgSetupRecord setup = {0};
  SsPmgApplyRecord apply = {0};
  SsPmg *pmg = NULL;
  Mat matrix = NULL;
  Vec exact = NULL;
  Vec rhs = NULL;
  Vec correction = NULL;
  Vec error = NULL;
  PetscReal error_norm = 0.0;
  PetscScalar sum = 0.0;
  int local_active = 0;
  int local_owns = 0;
  PetscMPIInt active_rank = -1;
  PetscMPIInt owner_count = 0;
  PetscMPIInt ranks[64];
  PetscMPIInt configured_ranks[64];
  PetscReal *coordinates = NULL;
  PetscInt *global_dof_indices = NULL;
  PetscInt row_start = 0;
  PetscInt row_end = 0;
  size_t owned_node_count = 0U;
  size_t node;
  size_t configured_count = 0U;

  CHECK(arguments->active_count <= 64);
  CHECK(ss_pmg_active_ranks_resolve(arguments->expected_ranks, arguments->active_count,
                                    ranks, (size_t)arguments->active_count) == SS_OK);
  if (arguments->expected_ranks == 5 && arguments->active_count == 2) {
    CHECK(ranks[0] == 0 && ranks[1] == 2);
  }
  if (arguments->expected_ranks == 7 && arguments->active_count == 3) {
    CHECK(ranks[0] == 0 && ranks[1] == 2 && ranks[2] == 4);
  }
  if (arguments->expected_ranks == 9 && arguments->active_count == 4) {
    CHECK(ranks[0] == 0 && ranks[1] == 2 && ranks[2] == 4 && ranks[3] == 6);
  }
  configure_level(&policy, SS_PMG_LEVEL_FGMRES_JACOBI, "pmga_",
                  (size_t)arguments->active_count);
  if (arguments->expected_ranks == 7 && arguments->active_count == 3) {
    static const PetscMPIInt arbitrary_order[] = {6, 1, 4};
    policy.active_ranks = arbitrary_order;
  }
  specification.fine_order = 1U;
  specification.level_count = 1U;
  specification.orders = &order;
  specification.level_policies = &policy;
  specification.near_nullspaces = &nullspace;
  specification.transfer_generation = 1U;
  specification.numeric_lag_interval = 1U;
  specification.memory_cap_bytes = 128U * 1024U * 1024U;
  CHECK(create_diagonal(global_size, &matrix) == 0);
  CHECK(MatGetOwnershipRange(matrix, &row_start, &row_end) == PETSC_SUCCESS);
  owned_node_count = (size_t)(row_end - row_start + 1) / 2U;
  coordinates = (PetscReal *)calloc(2U * owned_node_count, sizeof(*coordinates));
  global_dof_indices =
      (PetscInt *)calloc(2U * owned_node_count, sizeof(*global_dof_indices));
  CHECK(coordinates != NULL && global_dof_indices != NULL);
  for (node = 0U; node < owned_node_count; ++node) {
    const PetscInt first = row_start + (PetscInt)(2U * node);
    coordinates[2U * node] = (PetscReal)first;
    coordinates[2U * node + 1U] = (PetscReal)(node % 3U);
    global_dof_indices[2U * node] = first;
    global_dof_indices[2U * node + 1U] = first + 1 < row_end ? first + 1 : (PetscInt)-1;
  }
  nullspace.field = SS_PMG_FIELD_MECHANICS;
  nullspace.dimension = 2U;
  nullspace.owned_node_count = owned_node_count;
  nullspace.coordinates = coordinates;
  nullspace.global_dof_indices = global_dof_indices;
  nullspace.dependence_tolerance = 1.0e-12;
  CHECK(ss_pmg_create(PETSC_COMM_WORLD, &specification, &pmg) == SS_OK);
  free(global_dof_indices);
  free(coordinates);
  global_dof_indices = NULL;
  coordinates = NULL;
  CHECK(ss_pmg_active_ranks_get(pmg, 0U, configured_ranks, 64U, &configured_count) ==
        SS_OK);
  CHECK(configured_count == (size_t)arguments->active_count);
  if (arguments->expected_ranks == 7 && arguments->active_count == 3) {
    CHECK(configured_ranks[0] == 6 && configured_ranks[1] == 1 &&
          configured_ranks[2] == 4);
  }
  CHECK(ss_pmg_setup(pmg, matrix, 1U, 1U, &setup) == SS_OK);
  CHECK(ss_pmg_level_local_ownership(pmg, 0U, &local_active, &local_owns,
                                     &active_rank) == SS_OK);
  CHECK(local_active == local_owns);
  CHECK((local_active != 0 && active_rank >= 0) ||
        (local_active == 0 && active_rank == -1));
  CHECK((local_active != 0 && setup.levels[0].variable_point_block_count > 0U) ||
        (local_active == 0 && setup.levels[0].variable_point_block_count == 0U));
  owner_count = (PetscMPIInt)local_owns;
  CHECK(MPI_Allreduce(MPI_IN_PLACE, &owner_count, 1, MPI_INT, MPI_SUM,
                      PETSC_COMM_WORLD) == MPI_SUCCESS);
  CHECK(owner_count == arguments->active_count);
  *active_owners = owner_count;
  CHECK(MatCreateVecs(matrix, &exact, &rhs) == PETSC_SUCCESS);
  CHECK(VecDuplicate(exact, &correction) == PETSC_SUCCESS);
  CHECK(VecDuplicate(exact, &error) == PETSC_SUCCESS);
  CHECK(VecSet(exact, 1.0) == PETSC_SUCCESS);
  CHECK(MatMult(matrix, exact, rhs) == PETSC_SUCCESS);
  CHECK(ss_pmg_apply(pmg, rhs, correction, &apply) == SS_OK);
  CHECK(apply.status == SS_OK);
  CHECK(VecWAXPY(error, -1.0, exact, correction) == PETSC_SUCCESS);
  CHECK(VecNorm(error, NORM_2, &error_norm) == PETSC_SUCCESS);
  CHECK(VecSum(correction, &sum) == PETSC_SUCCESS);
  CHECK(error_norm <= 1.0e-10);
  *checksum = PetscRealPart(sum);
  *maximum_error = (double)error_norm;
  CHECK(ss_pmg_destroy(&pmg) == SS_OK);
  CHECK(VecDestroy(&error) == PETSC_SUCCESS);
  CHECK(VecDestroy(&correction) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(VecDestroy(&exact) == PETSC_SUCCESS);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}

static int test_redundant_lu(const Arguments *arguments, double *checksum,
                             double *maximum_error, int *active_owners) {
  const PetscInt global_size = 192;
  const uint32_t order = 1U;
  SsPmgLevelPolicy policy;
  SsPmgSpecification specification = {0};
  SsPmgSetupRecord setup = {0};
  SsPmgApplyRecord apply = {0};
  SsPmg *pmg = NULL;
  Mat matrix = NULL;
  Vec exact = NULL;
  Vec rhs = NULL;
  Vec correction = NULL;
  Vec error = NULL;
  PetscReal error_norm = 0.0;
  PetscScalar sum = 0.0;

  CHECK(arguments->active_count == arguments->expected_ranks);
  configure_level(&policy, SS_PMG_LEVEL_PREONLY_REDUNDANT_LU, "pmgred_",
                  (size_t)arguments->active_count);
  policy.maximum_iterations = 1;
  policy.redundant_solve_count = arguments->active_count;
  specification.fine_order = 1U;
  specification.level_count = 1U;
  specification.orders = &order;
  specification.level_policies = &policy;
  specification.transfer_generation = 1U;
  specification.numeric_lag_interval = 1U;
  specification.memory_cap_bytes = 128U * 1024U * 1024U;
  CHECK(create_diagonal(global_size, &matrix) == 0);
  CHECK(ss_pmg_create(PETSC_COMM_WORLD, &specification, &pmg) == SS_OK);
  CHECK(ss_pmg_setup(pmg, matrix, 1U, 1U, &setup) == SS_OK);
  CHECK(setup.levels[0].method == SS_PMG_LEVEL_PREONLY_REDUNDANT_LU);
  CHECK(setup.levels[0].redundant_solve_count == arguments->active_count);
  CHECK(setup.levels[0].active_rank_count == (size_t)arguments->active_count);
  CHECK(MatCreateVecs(matrix, &exact, &rhs) == PETSC_SUCCESS);
  CHECK(VecDuplicate(exact, &correction) == PETSC_SUCCESS);
  CHECK(VecDuplicate(exact, &error) == PETSC_SUCCESS);
  CHECK(VecSet(exact, 1.0) == PETSC_SUCCESS);
  CHECK(MatMult(matrix, exact, rhs) == PETSC_SUCCESS);
  CHECK(ss_pmg_apply(pmg, rhs, correction, &apply) == SS_OK);
  CHECK(apply.status == SS_OK);
  CHECK(VecWAXPY(error, -1.0, exact, correction) == PETSC_SUCCESS);
  CHECK(VecNorm(error, NORM_2, &error_norm) == PETSC_SUCCESS);
  CHECK(VecSum(correction, &sum) == PETSC_SUCCESS);
  CHECK(error_norm <= 1.0e-10);
  *checksum = PetscRealPart(sum);
  *maximum_error = (double)error_norm;
  *active_owners = arguments->active_count;
  CHECK(ss_pmg_destroy(&pmg) == SS_OK);
  CHECK(VecDestroy(&error) == PETSC_SUCCESS);
  CHECK(VecDestroy(&correction) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(VecDestroy(&exact) == PETSC_SUCCESS);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}

static int test_parallel_lu(const Arguments *arguments, double *checksum,
                            double *maximum_error, int *active_owners) {
  const PetscInt global_size = 192;
  const uint32_t order = 1U;
  SsPmgLevelPolicy policy;
  SsPmgSpecification specification = {0};
  SsPmgSetupRecord setup = {0};
  SsPmgSetupRecord refreshed = {0};
  SsPmgApplyRecord apply = {0};
  SsPmg *pmg = NULL;
  Mat matrix = NULL;
  Vec exact = NULL;
  Vec rhs = NULL;
  Vec correction = NULL;
  Vec error = NULL;
  PetscReal error_norm = 0.0;
  PetscScalar sum = 0.0;

  CHECK(arguments->active_count == arguments->expected_ranks);
  configure_level(&policy, SS_PMG_LEVEL_PREONLY_SUPERLU_DIST, "pmgdistlu_",
                  (size_t)arguments->active_count);
  policy.maximum_iterations = 1;
  specification.fine_order = 1U;
  specification.level_count = 1U;
  specification.orders = &order;
  specification.level_policies = &policy;
  specification.transfer_generation = 1U;
  specification.numeric_lag_interval = 1U;
  specification.memory_cap_bytes = 128U * 1024U * 1024U;
  CHECK(PetscOptionsSetValue(NULL, "-pmgdistlu_mat_superlu_dist_fact",
                             "SamePattern_SameRowPerm") == PETSC_SUCCESS);
  CHECK(create_diagonal(global_size, &matrix) == 0);
  CHECK(ss_pmg_create(PETSC_COMM_WORLD, &specification, &pmg) == SS_OK);
  CHECK(ss_pmg_setup(pmg, matrix, 1U, 1U, &setup) == SS_OK);
  CHECK(setup.levels[0].method == SS_PMG_LEVEL_PREONLY_SUPERLU_DIST);
  CHECK(setup.levels[0].repeated_factorization_reuses_row_permutation != 0);
  CHECK(setup.levels[0].redundant_solve_count == 0);
  CHECK(setup.levels[0].active_rank_count == (size_t)arguments->active_count);
  CHECK(MatCreateVecs(matrix, &exact, &rhs) == PETSC_SUCCESS);
  CHECK(VecDuplicate(exact, &correction) == PETSC_SUCCESS);
  CHECK(VecDuplicate(exact, &error) == PETSC_SUCCESS);
  CHECK(VecSet(exact, 1.0) == PETSC_SUCCESS);
  CHECK(MatMult(matrix, exact, rhs) == PETSC_SUCCESS);
  CHECK(ss_pmg_apply(pmg, rhs, correction, &apply) == SS_OK);
  CHECK(apply.status == SS_OK);
  CHECK(VecWAXPY(error, -1.0, exact, correction) == PETSC_SUCCESS);
  CHECK(VecNorm(error, NORM_2, &error_norm) == PETSC_SUCCESS);
  CHECK(VecSum(correction, &sum) == PETSC_SUCCESS);
  CHECK(error_norm <= 1.0e-10);
  CHECK(MatScale(matrix, 1.0001) == PETSC_SUCCESS);
  CHECK(ss_pmg_setup(pmg, matrix, 1U, 2U, &refreshed) == SS_OK);
  CHECK(refreshed.numeric_refreshed != 0);
  CHECK(refreshed.hierarchy_reused != 0);
  CHECK(refreshed.levels[0].repeated_factorization_reuses_row_permutation != 0);
  CHECK(MatMult(matrix, exact, rhs) == PETSC_SUCCESS);
  CHECK(ss_pmg_apply(pmg, rhs, correction, &apply) == SS_OK);
  CHECK(apply.status == SS_OK);
  CHECK(VecWAXPY(error, -1.0, exact, correction) == PETSC_SUCCESS);
  CHECK(VecNorm(error, NORM_2, &error_norm) == PETSC_SUCCESS);
  CHECK(error_norm <= 1.0e-10);
  *checksum = PetscRealPart(sum);
  *maximum_error = (double)error_norm;
  *active_owners = arguments->active_count;
  CHECK(ss_pmg_destroy(&pmg) == SS_OK);
  CHECK(PetscOptionsClearValue(NULL, "-pmgdistlu_mat_superlu_dist_fact") ==
        PETSC_SUCCESS);
  CHECK(VecDestroy(&error) == PETSC_SUCCESS);
  CHECK(VecDestroy(&correction) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(VecDestroy(&exact) == PETSC_SUCCESS);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}

static int create_two_level(size_t active_count, uint64_t lag,
                            SsPmgFailurePoint failure, size_t failure_level,
                            SsPmg **pmg, Mat *fine, Mat *prolongation) {
  static const uint32_t orders[] = {1U, 2U};
  SsPmgLevelPolicy policies[2];
  SsPmgSpecification specification = {0};
  CHECK(create_operator(5, 1.0, 0, fine) == 0);
  CHECK(create_transfer(3, prolongation) == 0);
  configure_level(&policies[0], SS_PMG_LEVEL_FGMRES_JACOBI, "pmgc_", active_count);
  configure_level(&policies[1], SS_PMG_LEVEL_RICHARDSON_JACOBI, "pmgf_", active_count);
  policies[1].richardson_scale = 0.25;
  specification.fine_order = 2U;
  specification.level_count = 2U;
  specification.orders = orders;
  specification.prolongations = prolongation;
  specification.level_policies = policies;
  specification.transfer_generation = 3U;
  specification.numeric_lag_interval = lag;
  specification.memory_cap_bytes = 128U * 1024U * 1024U;
  specification.injected_failure = failure;
  specification.injected_failure_level = failure_level;
  CHECK(ss_pmg_create(PETSC_COMM_WORLD, &specification, pmg) == SS_OK);
  return 0;
}

static int create_resource_limited_pmg(size_t active_count, SsPmg **pmg, Mat *fine,
                                       Mat *prolongation) {
  static const uint32_t orders[] = {1U, 2U};
  SsPmgLevelPolicy policies[2];
  SsPmgSpecification specification = {0};
  CHECK(create_operator(5, 1.0, 0, fine) == 0);
  CHECK(create_transfer(3, prolongation) == 0);
  configure_level(&policies[0], SS_PMG_LEVEL_FGMRES_JACOBI, "pmglimc_", active_count);
  configure_level(&policies[1], SS_PMG_LEVEL_RICHARDSON_JACOBI, "pmglimf_",
                  active_count);
  specification.fine_order = 2U;
  specification.level_count = 2U;
  specification.orders = orders;
  specification.prolongations = prolongation;
  specification.level_policies = policies;
  specification.transfer_generation = 1U;
  specification.numeric_lag_interval = 1U;
  specification.memory_cap_bytes = 1U;
  CHECK(ss_pmg_create(PETSC_COMM_WORLD, &specification, pmg) == SS_OK);
  return 0;
}

static int test_reuse(const Arguments *arguments, double *checksum,
                      double *maximum_error, int *active_owners) {
  SsPmg *pmg = NULL;
  SsPmgSetupRecord first = {0};
  SsPmgSetupRecord reused = {0};
  SsPmgSetupRecord lagged = {0};
  SsPmgSetupRecord refreshed = {0};
  SsPmgSetupRecord rebuilt = {0};
  Mat fine = NULL;
  Mat prolongation = NULL;
  Mat coarse = NULL;
  Mat coarse_refreshed = NULL;
  Mat coarse_rebuilt = NULL;
  PetscReal norm_before = 0.0;
  PetscReal norm_after = 0.0;
  PetscMPIInt owners = 0;

  CHECK(create_two_level((size_t)arguments->active_count, 2U, SS_PMG_FAIL_NONE, 0U,
                         &pmg, &fine, &prolongation) == 0);
  CHECK(ss_pmg_setup(pmg, fine, 1U, 1U, &first) == SS_OK);
  CHECK(first.transfer_generation == 3U);
  CHECK(ss_pmg_level_operator(pmg, 0U, &coarse) == SS_OK);
  CHECK(MatNorm(coarse, NORM_FROBENIUS, &norm_before) == PETSC_SUCCESS);
  CHECK(ss_pmg_setup(pmg, fine, 1U, 1U, &reused) == SS_OK);
  CHECK(reused.hierarchy_reused != 0 && reused.transfer_reused != 0);
  CHECK(MatScale(fine, 2.0) == PETSC_SUCCESS);
  CHECK(ss_pmg_setup(pmg, fine, 1U, 2U, &lagged) == SS_OK);
  CHECK(lagged.numeric_lagged != 0 && lagged.numeric_refreshed == 0);
  CHECK(MatNorm(coarse, NORM_FROBENIUS, &norm_after) == PETSC_SUCCESS);
  CHECK(fabs((double)(norm_after - norm_before)) <= 1.0e-12);
  CHECK(ss_pmg_setup(pmg, fine, 1U, 3U, &refreshed) == SS_OK);
  CHECK(refreshed.numeric_refreshed != 0);
  CHECK(refreshed.galerkin_pattern_reused != 0);
  CHECK(ss_pmg_level_operator(pmg, 0U, &coarse_refreshed) == SS_OK);
  CHECK(coarse_refreshed == coarse);
  CHECK(MatNorm(coarse, NORM_FROBENIUS, &norm_after) == PETSC_SUCCESS);
  CHECK(fabs((double)(norm_after - 2.0 * norm_before)) <= 1.0e-10);
  CHECK(ss_pmg_setup(pmg, fine, 2U, 4U, &rebuilt) == SS_OK);
  CHECK(rebuilt.hierarchy_reused == 0);
  CHECK(rebuilt.hierarchy_build_count == first.hierarchy_build_count + 1U);
  CHECK(rebuilt.galerkin_symbolic_count == first.galerkin_symbolic_count + 1U);
  CHECK(ss_pmg_level_operator(pmg, 0U, &coarse_rebuilt) == SS_OK);
  CHECK(coarse_rebuilt != NULL);
  {
    SsLinearSolver *solver = NULL;
    SsLinearProblem problem = {0};
    SsLinearSetupRecord linear_setup = {0};
    problem.operator_handle = fine;
    problem.properties = (SsLinearOperatorProperties){
        1, 0, 0, 0, 0, 1, 0,
    };
    problem.policy_chain.count = 1U;
    problem.policy_chain.policies[0] = outer_policy();
    problem.pmg = pmg;
    problem.structure_generation = 2U;
    problem.value_generation = 4U;
    CHECK(ss_linear_create(PETSC_COMM_WORLD, &solver) == SS_OK);
    CHECK(ss_linear_setup(solver, &problem, &linear_setup) == SS_OK);
    CHECK(linear_setup.pmg_selected != 0);
    CHECK(ss_linear_setup(solver, &problem, &linear_setup) == SS_OK);
    CHECK(linear_setup.reused != 0);
    CHECK(linear_setup.pmg_setup.hierarchy_reused != 0);
    CHECK(MatScale(prolongation, 2.0) == PETSC_SUCCESS);
    CHECK(ss_linear_setup(solver, &problem, &linear_setup) == SS_ERROR_INTEGRITY);
    CHECK(ss_linear_destroy(&solver) == SS_OK);
  }
  CHECK(ss_pmg_setup(pmg, fine, 2U, 4U, &reused) == SS_ERROR_INTEGRITY);
  {
    size_t level;
    for (level = 0U; level < 2U; ++level) {
      int local_owns = 0;
      int active = 0;
      PetscMPIInt active_rank = -1;
      owners = 0;
      CHECK(ss_pmg_level_local_ownership(pmg, level, &active, &local_owns,
                                         &active_rank) == SS_OK);
      CHECK(active == local_owns);
      CHECK((active != 0 && active_rank >= 0) || (active == 0 && active_rank == -1));
      owners = (PetscMPIInt)local_owns;
      CHECK(MPI_Allreduce(MPI_IN_PLACE, &owners, 1, MPI_INT, MPI_SUM,
                          PETSC_COMM_WORLD) == MPI_SUCCESS);
      CHECK(owners == arguments->active_count);
    }
  }
  *active_owners = owners;
  *checksum = (double)(first.galerkin_numeric_count + rebuilt.galerkin_numeric_count);
  *maximum_error = fabs((double)(norm_after - 2.0 * norm_before));
  CHECK(ss_pmg_destroy(&pmg) == SS_OK);
  CHECK(MatDestroy(&prolongation) == PETSC_SUCCESS);
  CHECK(MatDestroy(&fine) == PETSC_SUCCESS);
  return 0;
}

static int test_failure(const Arguments *arguments, double *checksum,
                        double *maximum_error, int *active_owners) {
  static const struct {
    SsPmgFailurePoint point;
    size_t level;
  } cases[] = {
      {SS_PMG_FAIL_PRE_SMOOTH, 1U},       {SS_PMG_FAIL_RESTRICT, 1U},
      {SS_PMG_FAIL_BRIDGE_TO_ACTIVE, 0U}, {SS_PMG_FAIL_LEVEL_SOLVE, 0U},
      {SS_PMG_FAIL_LEVEL_SOLVE, 1U},      {SS_PMG_FAIL_BRIDGE_FROM_ACTIVE, 0U},
      {SS_PMG_FAIL_PROLONG, 1U},          {SS_PMG_FAIL_POST_SMOOTH, 1U},
  };
  size_t case_index;

  for (case_index = 0U; case_index < sizeof(cases) / sizeof(cases[0]); ++case_index) {
    SsPmg *pmg = NULL;
    SsPmgSetupRecord setup = {0};
    SsPmgApplyRecord apply = {0};
    Mat fine = NULL;
    Mat prolongation = NULL;
    Vec rhs = NULL;
    Vec correction = NULL;
    PetscReal norm = 0.0;
    CHECK(create_two_level((size_t)arguments->active_count, 1U, cases[case_index].point,
                           cases[case_index].level, &pmg, &fine, &prolongation) == 0);
    CHECK(ss_pmg_setup(pmg, fine, 1U, 1U, &setup) == SS_OK);
    CHECK(MatCreateVecs(fine, &correction, &rhs) == PETSC_SUCCESS);
    CHECK(VecSet(rhs, 1.0) == PETSC_SUCCESS);
    CHECK(ss_pmg_apply(pmg, rhs, correction, &apply) == SS_ERROR_INJECTED);
    CHECK(apply.status == SS_ERROR_INJECTED);
    CHECK(apply.failed_point == cases[case_index].point);
    CHECK(apply.failed_level == cases[case_index].level);
    CHECK(VecNorm(correction, NORM_2, &norm) == PETSC_SUCCESS);
    CHECK(!isfinite((double)norm));
    CHECK(ss_pmg_destroy(&pmg) == SS_OK);
    CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
    CHECK(VecDestroy(&correction) == PETSC_SUCCESS);
    CHECK(MatDestroy(&prolongation) == PETSC_SUCCESS);
    CHECK(MatDestroy(&fine) == PETSC_SUCCESS);
  }
  {
    static const uint32_t order = 1U;
    SsPmgLevelPolicy policy;
    SsPmgSpecification specification = {0};
    SsPmgSetupRecord setup = {0};
    SsPmgApplyRecord apply = {0};
    SsPmg *pmg = NULL;
    Mat fine = NULL;
    Vec rhs = NULL;
    Vec correction = NULL;
    PetscReal norm = 0.0;
    configure_level(&policy, SS_PMG_LEVEL_FGMRES_JACOBI, "pmgdiv_",
                    (size_t)arguments->active_count);
    policy.maximum_iterations = 4;
    policy.rtol = 1.0e-15;
    specification.fine_order = 1U;
    specification.level_count = 1U;
    specification.orders = &order;
    specification.level_policies = &policy;
    specification.transfer_generation = 1U;
    specification.numeric_lag_interval = 1U;
    specification.memory_cap_bytes = 4U * 1024U * 1024U;
    /*
     * A finite iteration budget is legal fixed work for a PMG level.  Exercise
     * a genuine non-finite level-solve failure instead of treating
     * KSP_DIVERGED_ITS at the requested budget as a failure.
     */
    CHECK(create_operator(5, (PetscReal)NAN, 0, &fine) == 0);
    CHECK(ss_pmg_create(PETSC_COMM_WORLD, &specification, &pmg) == SS_OK);
    CHECK(ss_pmg_setup(pmg, fine, 1U, 1U, &setup) == SS_OK);
    CHECK(MatCreateVecs(fine, &correction, &rhs) == PETSC_SUCCESS);
    CHECK(VecSet(rhs, 1.0) == PETSC_SUCCESS);
    CHECK(ss_pmg_apply(pmg, rhs, correction, &apply) == SS_ERROR_INTERNAL);
    CHECK(apply.status == SS_ERROR_INTERNAL);
    CHECK(apply.failed_point == SS_PMG_FAIL_LEVEL_SOLVE);
    CHECK(apply.failed_level == 0U);
    CHECK(VecNorm(correction, NORM_2, &norm) == PETSC_SUCCESS);
    CHECK(!isfinite((double)norm));
    CHECK(ss_pmg_destroy(&pmg) == SS_OK);
    CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
    CHECK(VecDestroy(&correction) == PETSC_SUCCESS);
    CHECK(MatDestroy(&fine) == PETSC_SUCCESS);
  }
  {
    SsPmg *pmg = NULL;
    SsLinearSolver *solver = NULL;
    SsLinearProblem problem = {0};
    SsLinearSetupRecord setup = {0};
    Mat fine = NULL;
    Mat prolongation = NULL;
    CHECK(create_resource_limited_pmg((size_t)arguments->active_count, &pmg, &fine,
                                      &prolongation) == 0);
    problem.operator_handle = fine;
    problem.properties = (SsLinearOperatorProperties){
        1, 0, 0, 0, 0, 1, 0,
    };
    problem.policy_chain.count = 2U;
    problem.policy_chain.policies[0] = outer_policy();
    problem.policy_chain.policies[1] = outer_policy();
    (void)snprintf(problem.policy_chain.policies[1].identity,
                   sizeof(problem.policy_chain.policies[1].identity), "%s",
                   "declared-jacobi-fallback");
    (void)snprintf(problem.policy_chain.policies[1].options_prefix,
                   sizeof(problem.policy_chain.policies[1].options_prefix), "%s",
                   "pmgfb_");
    problem.policy_chain.policies[1].pc = SS_LINEAR_PC_JACOBI;
    problem.pmg = pmg;
    problem.structure_generation = 1U;
    problem.value_generation = 1U;
    CHECK(ss_linear_create(PETSC_COMM_WORLD, &solver) == SS_OK);
    CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
    CHECK(setup.attempt_count == 2U);
    CHECK(setup.attempts[0].setup_status == SS_ERROR_RESOURCE);
    CHECK(setup.attempts[0].selected == 0);
    CHECK(setup.attempts[1].setup_status == SS_OK);
    CHECK(setup.attempts[1].selected != 0);
    CHECK(setup.pmg_selected == 0);
    CHECK(strcmp(setup.effective_pc, "jacobi") == 0);
    CHECK(ss_linear_destroy(&solver) == SS_OK);
    CHECK(ss_pmg_destroy(&pmg) == SS_OK);
    CHECK(MatDestroy(&prolongation) == PETSC_SUCCESS);
    CHECK(MatDestroy(&fine) == PETSC_SUCCESS);
  }
  *checksum = (double)(sizeof(cases) / sizeof(cases[0]));
  *maximum_error = 0.0;
  *active_owners = arguments->active_count;
  return 0;
}

static int test_pcmg(const Arguments *arguments, double *checksum,
                     double *maximum_error, int *active_owners) {
  SsPmg *pmg = NULL;
  SsPmgSetupRecord setup = {0};
  SsPmgApplyRecord apply = {0};
  Mat fine = NULL;
  Mat prolongation = NULL;
  Vec rhs = NULL;
  Vec project = NULL;
  Vec petsc = NULL;
  Vec difference = NULL;
  PC pc = NULL;
  Mat coarse = NULL;
  KSP coarse_solve = NULL;
  KSP smoother_down = NULL;
  KSP smoother_up = NULL;
  PC level_pc = NULL;
  PetscReal error = 0.0;
  PetscScalar sum = 0.0;

  CHECK(create_two_level((size_t)arguments->expected_ranks, 1U, SS_PMG_FAIL_NONE, 0U,
                         &pmg, &fine, &prolongation) == 0);
  CHECK(ss_pmg_setup(pmg, fine, 1U, 1U, &setup) == SS_OK);
  CHECK(MatCreateVecs(fine, &project, &rhs) == PETSC_SUCCESS);
  CHECK(VecDuplicate(project, &petsc) == PETSC_SUCCESS);
  CHECK(VecDuplicate(project, &difference) == PETSC_SUCCESS);
  CHECK(VecSet(rhs, 1.0) == PETSC_SUCCESS);
  CHECK(ss_pmg_apply(pmg, rhs, project, &apply) == SS_OK);
  CHECK(ss_pmg_level_operator(pmg, 0U, &coarse) == SS_OK);
  CHECK(PCCreate(PETSC_COMM_WORLD, &pc) == PETSC_SUCCESS);
  CHECK(PCSetType(pc, PCMG) == PETSC_SUCCESS);
  CHECK(PCSetOperators(pc, fine, fine) == PETSC_SUCCESS);
  CHECK(PCMGSetLevels(pc, 2, NULL) == PETSC_SUCCESS);
  CHECK(PCMGSetType(pc, PC_MG_MULTIPLICATIVE) == PETSC_SUCCESS);
  CHECK(PCMGSetGalerkin(pc, PC_MG_GALERKIN_NONE) == PETSC_SUCCESS);
  CHECK(PCMGSetDistinctSmoothUp(pc) == PETSC_SUCCESS);
  CHECK(PCMGSetInterpolation(pc, 1, prolongation) == PETSC_SUCCESS);
  CHECK(PCMGSetOperators(pc, 0, coarse, coarse) == PETSC_SUCCESS);
  CHECK(PCMGSetOperators(pc, 1, fine, fine) == PETSC_SUCCESS);
  CHECK(PCMGGetCoarseSolve(pc, &coarse_solve) == PETSC_SUCCESS);
  CHECK(KSPSetType(coarse_solve, KSPFGMRES) == PETSC_SUCCESS);
  CHECK(KSPGetPC(coarse_solve, &level_pc) == PETSC_SUCCESS);
  CHECK(PCSetType(level_pc, PCJACOBI) == PETSC_SUCCESS);
  CHECK(KSPSetNormType(coarse_solve, KSP_NORM_UNPRECONDITIONED) == PETSC_SUCCESS);
  CHECK(KSPSetTolerances(coarse_solve, 1.0e-13, 1.0e-15, PETSC_DEFAULT, 64) ==
        PETSC_SUCCESS);
  CHECK(KSPGMRESSetRestart(coarse_solve, 30) == PETSC_SUCCESS);
  CHECK(PCMGGetSmootherDown(pc, 1, &smoother_down) == PETSC_SUCCESS);
  CHECK(KSPSetType(smoother_down, KSPRICHARDSON) == PETSC_SUCCESS);
  CHECK(KSPGetPC(smoother_down, &level_pc) == PETSC_SUCCESS);
  CHECK(PCSetType(level_pc, PCJACOBI) == PETSC_SUCCESS);
  CHECK(KSPRichardsonSetScale(smoother_down, 0.25) == PETSC_SUCCESS);
  CHECK(KSPSetNormType(smoother_down, KSP_NORM_NONE) == PETSC_SUCCESS);
  CHECK(KSPSetConvergenceTest(smoother_down, KSPConvergedSkip, NULL, NULL) ==
        PETSC_SUCCESS);
  CHECK(KSPSetTolerances(smoother_down, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT,
                         1) == PETSC_SUCCESS);
  CHECK(PCMGGetSmootherUp(pc, 1, &smoother_up) == PETSC_SUCCESS);
  CHECK(KSPSetType(smoother_up, KSPRICHARDSON) == PETSC_SUCCESS);
  CHECK(KSPGetPC(smoother_up, &level_pc) == PETSC_SUCCESS);
  CHECK(PCSetType(level_pc, PCJACOBI) == PETSC_SUCCESS);
  CHECK(KSPRichardsonSetScale(smoother_up, 0.25) == PETSC_SUCCESS);
  CHECK(KSPSetNormType(smoother_up, KSP_NORM_NONE) == PETSC_SUCCESS);
  CHECK(KSPSetConvergenceTest(smoother_up, KSPConvergedSkip, NULL, NULL) ==
        PETSC_SUCCESS);
  CHECK(KSPSetTolerances(smoother_up, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT, 1) ==
        PETSC_SUCCESS);
  CHECK(PCSetUp(pc) == PETSC_SUCCESS);
  CHECK(PCApply(pc, rhs, petsc) == PETSC_SUCCESS);
  CHECK(VecWAXPY(difference, -1.0, project, petsc) == PETSC_SUCCESS);
  CHECK(VecNorm(difference, NORM_2, &error) == PETSC_SUCCESS);
  CHECK(error <= 2.0e-11);
  CHECK(VecSum(project, &sum) == PETSC_SUCCESS);
  *checksum = PetscRealPart(sum);
  *maximum_error = (double)error;
  *active_owners = arguments->expected_ranks;
  CHECK(PCDestroy(&pc) == PETSC_SUCCESS);
  CHECK(ss_pmg_destroy(&pmg) == SS_OK);
  CHECK(VecDestroy(&difference) == PETSC_SUCCESS);
  CHECK(VecDestroy(&petsc) == PETSC_SUCCESS);
  CHECK(VecDestroy(&project) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(MatDestroy(&prolongation) == PETSC_SUCCESS);
  CHECK(MatDestroy(&fine) == PETSC_SUCCESS);
  return 0;
}

static int test_memory(const Arguments *arguments, double *checksum,
                       double *maximum_error, int *active_owners) {
  SsPmg *pmg = NULL;
  SsPmgSetupRecord setup = {0};
  SsPmgSetupRecord repeated = {0};
  SsPmgDiagnostics before = {0};
  SsPmgDiagnostics after = {0};
  SsLinearSolver *solver = NULL;
  SsLinearProblem problem = {0};
  SsLinearSetupRecord linear_setup = {0};
  SsLinearSolveRecord solve = {0};
  Mat fine = NULL;
  Mat prolongation = NULL;
  Vec exact = NULL;
  Vec rhs = NULL;
  Vec solution = NULL;
  Vec error = NULL;
  PetscReal error_norm = 0.0;
  PetscScalar sum = 0.0;
  double representative_checksum = 0.0;
  double representative_error = 0.0;
  int repeat;

  CHECK(create_two_level((size_t)arguments->active_count, 1U, SS_PMG_FAIL_NONE, 0U,
                         &pmg, &fine, &prolongation) == 0);
  CHECK(ss_pmg_setup(pmg, fine, 1U, 1U, &setup) == SS_OK);
  CHECK(ss_pmg_diagnostics(pmg, &before) == SS_OK);
  problem.operator_handle = fine;
  problem.properties = (SsLinearOperatorProperties){
      1, 0, 0, 0, 0, 1, 0,
  };
  problem.policy_chain.count = 1U;
  problem.policy_chain.policies[0] = outer_policy();
  problem.pmg = pmg;
  problem.structure_generation = 1U;
  problem.value_generation = 1U;
  CHECK(ss_linear_create(PETSC_COMM_WORLD, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &linear_setup) == SS_OK);
  CHECK(linear_setup.pmg_selected != 0);
  CHECK(strcmp(linear_setup.effective_pc, "shell") == 0);
  CHECK(MatCreateVecs(fine, &exact, &rhs) == PETSC_SUCCESS);
  CHECK(VecDuplicate(exact, &solution) == PETSC_SUCCESS);
  CHECK(VecDuplicate(exact, &error) == PETSC_SUCCESS);
  CHECK(VecSet(exact, 1.0) == PETSC_SUCCESS);
  CHECK(MatMult(fine, exact, rhs) == PETSC_SUCCESS);
  for (repeat = 0; repeat < 32; ++repeat) {
    CHECK(VecSet(solution, 0.0) == PETSC_SUCCESS);
    CHECK(ss_linear_solve(solver, rhs, solution, &solve) == SS_OK);
    CHECK(solve.status == SS_LINEAR_SOLVE_CONVERGED);
    CHECK(ss_pmg_setup(pmg, fine, 1U, 1U, &repeated) == SS_OK);
    CHECK(repeated.hierarchy_reused != 0);
  }
  CHECK(VecWAXPY(error, -1.0, exact, solution) == PETSC_SUCCESS);
  CHECK(VecNorm(error, NORM_2, &error_norm) == PETSC_SUCCESS);
  CHECK(VecSum(solution, &sum) == PETSC_SUCCESS);
  CHECK(error_norm <= 1.0e-8);
  CHECK(ss_pmg_diagnostics(pmg, &after) == SS_OK);
  CHECK(after.current_matrix_objects == before.current_matrix_objects);
  CHECK(after.current_vector_objects == before.current_vector_objects);
  CHECK(after.peak_matrix_objects == before.peak_matrix_objects);
  CHECK(after.peak_vector_objects == before.peak_vector_objects);
  CHECK(after.admitted_memory_bytes <= 128U * 1024U * 1024U);
  *checksum = PetscRealPart(sum);
  *maximum_error = (double)error_norm;
  *active_owners = arguments->active_count;
  CHECK(ss_linear_destroy(&solver) == SS_OK);
  CHECK(ss_pmg_destroy(&pmg) == SS_OK);
  CHECK(VecDestroy(&error) == PETSC_SUCCESS);
  CHECK(VecDestroy(&solution) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(VecDestroy(&exact) == PETSC_SUCCESS);
  CHECK(MatDestroy(&prolongation) == PETSC_SUCCESS);
  CHECK(MatDestroy(&fine) == PETSC_SUCCESS);
  CHECK(exercise_representative_profile(2U, 1U, 2U, 0, (size_t)arguments->active_count,
                                        &representative_checksum,
                                        &representative_error) == 0);
  CHECK(exercise_representative_profile(2U, 2U, 1U, 1, (size_t)arguments->active_count,
                                        &representative_checksum,
                                        &representative_error) == 0);
  CHECK(exercise_representative_profile(3U, 4U, 3U, 1, (size_t)arguments->active_count,
                                        &representative_checksum,
                                        &representative_error) == 0);
  CHECK(exercise_representative_profile(3U, 8U, 3U, 1, (size_t)arguments->active_count,
                                        &representative_checksum,
                                        &representative_error) == 0);
  *checksum += representative_checksum;
  if (representative_error > *maximum_error) {
    *maximum_error = representative_error;
  }
  return 0;
}

int main(int argc, char **argv) {
  Arguments arguments;
  double checksum = 0.0;
  double maximum_error = 0.0;
  int active_owners = 0;
  int observed_threads = 1;
  int rank = -1;
  int size = 0;
  int result = 0;

  if (parse_arguments(argc, argv, &arguments) != 0) {
    fprintf(stderr,
            "usage: %s --test-id ID --expected-ranks N --active-count A "
            "--expected-threads T --output PATH\n",
            argv[0]);
    return 2;
  }
  argc = 1;
  CHECK(PetscInitialize(&argc, &argv, NULL, NULL) == PETSC_SUCCESS);
  CHECK(MPI_Comm_rank(PETSC_COMM_WORLD, &rank) == MPI_SUCCESS);
  CHECK(MPI_Comm_size(PETSC_COMM_WORLD, &size) == MPI_SUCCESS);
  CHECK(size == arguments.expected_ranks);
  CHECK(arguments.active_count <= size);
#if defined(_OPENMP)
#pragma omp parallel
  {
#pragma omp single
    {
      observed_threads = omp_get_num_threads();
    }
  }
#endif
  CHECK(observed_threads == arguments.expected_threads);

  if (strcmp(arguments.test_id, "PMG-CONTRACT-ACTIVE-001") == 0) {
    result = test_active(&arguments, &checksum, &maximum_error, &active_owners);
  } else if (strcmp(arguments.test_id, "PMG-CONTRACT-REDUNDANT-LU-001") == 0) {
    result = test_redundant_lu(&arguments, &checksum, &maximum_error, &active_owners);
  } else if (strcmp(arguments.test_id, "PMG-CONTRACT-PARALLEL-LU-001") == 0) {
    result = test_parallel_lu(&arguments, &checksum, &maximum_error, &active_owners);
  } else if (strcmp(arguments.test_id, "PMG-CONTRACT-REUSE-001") == 0) {
    result = test_reuse(&arguments, &checksum, &maximum_error, &active_owners);
  } else if (strcmp(arguments.test_id, "PMG-CONTRACT-PCMG-001") == 0) {
    result = test_pcmg(&arguments, &checksum, &maximum_error, &active_owners);
  } else if (strcmp(arguments.test_id, "PMG-NEG-LEVELFAIL-001") == 0) {
    result = test_failure(&arguments, &checksum, &maximum_error, &active_owners);
  } else if (strcmp(arguments.test_id, "PMG-PERF-MEMORY-001") == 0) {
    result = test_memory(&arguments, &checksum, &maximum_error, &active_owners);
  } else {
    fprintf(stderr, "unknown test id: %s\n", arguments.test_id);
    result = 2;
  }
  if (result == 0) {
    result = publish_result(&arguments, checksum, maximum_error, active_owners,
                            observed_threads);
  }
  CHECK(PetscFinalize() == PETSC_SUCCESS);
  return result;
}
