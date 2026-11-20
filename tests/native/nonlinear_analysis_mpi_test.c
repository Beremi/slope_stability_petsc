/* Newton solver and continuation analyses at MPI and hybrid layouts.
 *
 * Launched by tests/native/support/mpi_layout_runner.py --suite nonlinear.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/analysis/analysis.h"
#include "slope_internal/nonlinear/nonlinear.h"
#include "slope_internal/pmg/pmg.h"

#include <petscsys.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expression)                                                              \
  do {                                                                                 \
    if (!(expression)) {                                                               \
      fprintf(stderr, "%s:%d: assertion failed: %s\n", __FILE__, __LINE__,             \
              #expression);                                                            \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

static const char CONTRACT_INVARIANT[] =
    "sha256:159d96baf8213a98965269acd1e8e19b7f137a9583a71856e8be0778be9fd0d3";

typedef struct {
  const char *test_id;
  const char *output;
  int expected_ranks;
  int expected_threads;
} Arguments;

typedef struct {
  MPI_Comm communicator;
  PetscInt evaluations;
} DistributedModel;

typedef struct {
  SsNonlinearProblem nonlinear;
  SsNonlinearPolicy nonlinear_policy;
  DistributedModel model;
  size_t transaction_begin_count;
  size_t transaction_commit_count;
  size_t transaction_rollback_count;
  int transaction_open;
  Vec previous_accepted;
  Vec current_accepted;
  size_t captured_state_count;
  size_t captured_first_ordinal;
  int repeated_seed_mode;
} AnalysisModel;

static int parse_integer(const char *text, int *value) {
  char *end = NULL;
  long parsed;
  if (text == NULL || value == NULL || text[0] == '\0') {
    return 0;
  }
  parsed = strtol(text, &end, 10);
  if (end == text || *end != '\0' || parsed <= 0 || parsed > 1024) {
    return 0;
  }
  *value = (int)parsed;
  return 1;
}

static int parse_arguments(int argc, char **argv, Arguments *arguments) {
  int index;
  memset(arguments, 0, sizeof(*arguments));
  arguments->expected_threads = 1;
  for (index = 1; index < argc; ++index) {
    if (strcmp(argv[index], "--test-id") == 0 && index + 1 < argc) {
      arguments->test_id = argv[++index];
    } else if (strcmp(argv[index], "--output") == 0 && index + 1 < argc) {
      arguments->output = argv[++index];
    } else if (strcmp(argv[index], "--expected-ranks") == 0 && index + 1 < argc) {
      if (!parse_integer(argv[++index], &arguments->expected_ranks)) {
        return 0;
      }
    } else if (strcmp(argv[index], "--expected-threads") == 0 && index + 1 < argc) {
      if (!parse_integer(argv[++index], &arguments->expected_threads)) {
        return 0;
      }
    } else {
      return 0;
    }
  }
  return arguments->test_id != NULL && arguments->output != NULL &&
         arguments->expected_ranks > 0;
}

static int observed_threads(void) {
  int result = 1;
#ifdef _OPENMP
#pragma omp parallel
  {
#pragma omp single
    result = omp_get_num_threads();
  }
#endif
  return result;
}

static int create_vector(MPI_Comm communicator, PetscInt global_size, PetscReal value,
                         Vec *vector) {
  CHECK(VecCreateMPI(communicator, PETSC_DECIDE, global_size, vector) == PETSC_SUCCESS);
  CHECK(VecSet(*vector, value) == PETSC_SUCCESS);
  return 0;
}

static int create_diagonal_like(Vec vector, PetscReal value, Mat *matrix) {
  MPI_Comm communicator = PetscObjectComm((PetscObject)vector);
  PetscInt local_size = 0;
  PetscInt global_size = 0;
  PetscInt start = 0;
  PetscInt end = 0;
  PetscInt row;
  CHECK(VecGetLocalSize(vector, &local_size) == PETSC_SUCCESS);
  CHECK(VecGetSize(vector, &global_size) == PETSC_SUCCESS);
  CHECK(VecGetOwnershipRange(vector, &start, &end) == PETSC_SUCCESS);
  CHECK(MatCreateAIJ(communicator, local_size, local_size, global_size, global_size, 1,
                     NULL, 0, NULL, matrix) == PETSC_SUCCESS);
  for (row = start; row < end; ++row) {
    CHECK(MatSetValue(*matrix, row, row, value, INSERT_VALUES) == PETSC_SUCCESS);
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

static SsStatus distributed_evaluate(void *opaque, Vec state, PetscReal factor,
                                     Vec residual, Mat tangent, PetscReal *functional) {
  DistributedModel *model = (DistributedModel *)opaque;
  const PetscScalar *state_values = NULL;
  PetscScalar *residual_values = NULL;
  PetscInt local_size = 0;
  PetscInt start = 0;
  PetscInt index;
  PetscReal local_functional = 0.0;
  PetscReal global_functional = 0.0;
  ++model->evaluations;
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
                      model->communicator) != MPI_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    *functional = global_functional;
  }
  return SS_OK;
}

static SsStatus distributed_controlled_evaluate(void *opaque,
                                                SsNonlinearControlMode mode, Vec state,
                                                PetscReal control, Vec residual,
                                                Mat tangent, PetscReal *functional) {
  if (mode != SS_NONLINEAR_CONTROL_SSR_INDIRECT) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return distributed_evaluate(opaque, state, control, residual, tangent, functional);
}

static SsStatus distributed_admissibility(void *opaque, Vec state, PetscReal factor,
                                          int *admissible) {
  PetscReal norm = 0.0;
  (void)opaque;
  if (admissible == NULL || VecNorm(state, NORM_INFINITY, &norm) != PETSC_SUCCESS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *admissible = isfinite((double)norm) && isfinite((double)factor) && factor > 0.0;
  return SS_OK;
}

static void fill_nonlinear_policy(SsNonlinearPolicy *policy) {
  memset(policy, 0, sizeof(*policy));
  policy->tolerance = 1.0e-11;
  policy->stopping_criterion = SS_NONLINEAR_STOP_RELATIVE_RESIDUAL;
  policy->stopping_tolerance = policy->tolerance;
  policy->maximum_iterations = 30;
  policy->maximum_line_search_trials = 20;
  policy->minimum_alpha = 1.0 / 1048576.0;
  policy->regularization_minimum = 1.0e-8;
  policy->singular_relative_tolerance = 1.0e-12;
  policy->relaxed_factor = 10.0;
  policy->line_method = SS_NONLINEAR_LINE_DIRECTIONAL;
}

static void fill_iterative_policy(SsLinearPolicyChain *chain,
                                  SsLinearPcKind preconditioner, const char *identity,
                                  const char *prefix) {
  SsLinearPolicy *policy;
  memset(chain, 0, sizeof(*chain));
  chain->count = 1U;
  policy = &chain->policies[0];
  (void)snprintf(policy->identity, sizeof(policy->identity), "%s", identity);
  (void)snprintf(policy->options_prefix, sizeof(policy->options_prefix), "%s", prefix);
  policy->operator_class = SS_LINEAR_OPERATOR_GENERAL;
  policy->method = SS_LINEAR_METHOD_FGMRES;
  policy->pc = preconditioner;
  policy->side = SS_LINEAR_SIDE_RIGHT;
  policy->rtol = 1.0e-12;
  policy->atol = 1.0e-14;
  policy->dtol = 1.0e8;
  policy->maximum_iterations = 80;
  policy->restart = 20;
  policy->vector_memory_cap_bytes = 32U * 1024U * 1024U;
}

static int create_nonlinear(AnalysisModel *model, MPI_Comm communicator,
                            PetscInt global_size, Vec *load, Mat *elastic) {
  memset(model, 0, sizeof(*model));
  model->model.communicator = communicator;
  CHECK(create_vector(communicator, global_size, 1.0, load) == 0);
  CHECK(create_diagonal_like(*load, 2.0, elastic) == 0);
  fill_nonlinear_policy(&model->nonlinear_policy);
  model->nonlinear.communicator = communicator;
  model->nonlinear.reference_load = *load;
  model->nonlinear.elastic_reference = *elastic;
  model->nonlinear.operations.evaluate = distributed_evaluate;
  model->nonlinear.operations.evaluate_controlled = distributed_controlled_evaluate;
  model->nonlinear.operations.admissibility = distributed_admissibility;
  model->nonlinear.context = &model->model;
  model->nonlinear.linear_properties.finite_action = 1;
  model->nonlinear.linear_properties.measured_symmetric = 1;
  model->nonlinear.linear_properties.positive_energy = 1;
  model->nonlinear.linear_properties.pc_symmetric = 1;
  model->nonlinear.linear_properties.pc_positive = 1;
  fill_iterative_policy(&model->nonlinear.linear_policies, SS_LINEAR_PC_JACOBI,
                        "nonlinear-mpi-iterative", "nonlinear_mpi_");
  model->nonlinear.structure_generation = 31U;
  model->nonlinear.first_value_generation = 1000U;
  model->nonlinear.transaction_generation = 37U;
  model->nonlinear.accepted_generation = 41U;
  return 0;
}

static int maximum_value_error(Vec vector, PetscReal expected,
                               PetscReal *maximum_error) {
  const PetscScalar *values = NULL;
  PetscInt local_size = 0;
  PetscInt index;
  PetscReal local_maximum = 0.0;
  CHECK(VecGetLocalSize(vector, &local_size) == PETSC_SUCCESS);
  CHECK(VecGetArrayRead(vector, &values) == PETSC_SUCCESS);
  for (index = 0; index < local_size; ++index) {
    local_maximum = fmax(local_maximum, fabs(PetscRealPart(values[index]) - expected));
  }
  CHECK(VecRestoreArrayRead(vector, &values) == PETSC_SUCCESS);
  CHECK(MPI_Allreduce(&local_maximum, maximum_error, 1, MPIU_REAL, MPI_MAX,
                      PetscObjectComm((PetscObject)vector)) == MPI_SUCCESS);
  return 0;
}

static int run_world_fixed(PetscInt global_size, SsPmg *pmg, PetscReal *maximum_error,
                           PetscReal *checksum, int *pmg_selected) {
  AnalysisModel model;
  Vec load = NULL;
  Vec parent = NULL;
  Vec accepted = NULL;
  Mat elastic = NULL;
  SsNonlinearSolveRecord record;
  PetscScalar sum = 0.0;
  CHECK(create_nonlinear(&model, PETSC_COMM_WORLD, global_size, &load, &elastic) == 0);
  if (pmg != NULL) {
    model.nonlinear.pmg = pmg;
    model.nonlinear.linear_properties.pc_variable = 1;
    fill_iterative_policy(&model.nonlinear.linear_policies, SS_LINEAR_PC_PMG,
                          "nonlinear-production-pmg", "nonlinear_pmg_");
  }
  CHECK(create_vector(PETSC_COMM_WORLD, global_size, 0.0, &parent) == 0);
  CHECK(VecDuplicate(parent, &accepted) == PETSC_SUCCESS);
  CHECK(ss_nonlinear_fixed_solve(&model.nonlinear, &model.nonlinear_policy, 1.0, parent,
                                 accepted, &record) == SS_OK);
  CHECK(record.outcome == SS_NONLINEAR_CONVERGED);
  CHECK(maximum_value_error(accepted, 0.45339765151640377, maximum_error) == 0);
  CHECK(VecSum(accepted, &sum) == PETSC_SUCCESS);
  *checksum = PetscRealPart(sum);
  *pmg_selected =
      record.history_count > 0U && record.history[0].linear_setup.pmg_selected != 0;
  CHECK(VecDestroy(&accepted) == PETSC_SUCCESS);
  CHECK(VecDestroy(&parent) == PETSC_SUCCESS);
  CHECK(VecDestroy(&load) == PETSC_SUCCESS);
  CHECK(MatDestroy(&elastic) == PETSC_SUCCESS);
  return 0;
}

static int run_world_controlled(PetscInt global_size,
                                SsNonlinearControlSensitivityMethod sensitivity_method,
                                PetscReal sensitivity_relative_step,
                                PetscReal *maximum_error, PetscReal *checksum) {
  const PetscReal expected_state = 0.55;
  const PetscReal expected_factor =
      3.0 + expected_state * expected_state - 1.0 / expected_state;
  AnalysisModel model;
  SsNonlinearPolicy policy;
  SsNonlinearSolveRecord record;
  Vec load = NULL;
  Vec parent = NULL;
  Vec accepted = NULL;
  Mat elastic = NULL;
  PetscScalar accepted_sum = 0.0;
  PetscReal accepted_factor = NAN;
  PetscReal state_error = 0.0;
  size_t index;
  CHECK(create_nonlinear(&model, PETSC_COMM_WORLD, global_size, &load, &elastic) == 0);
  policy = model.nonlinear_policy;
  policy.line_method = SS_NONLINEAR_LINE_RESIDUAL_MONOTONE;
  policy.control_sensitivity_method = sensitivity_method;
  policy.control_sensitivity_relative_step = sensitivity_relative_step;
  CHECK(create_vector(PETSC_COMM_WORLD, global_size, expected_state, &parent) == 0);
  CHECK(VecDuplicate(parent, &accepted) == PETSC_SUCCESS);
  CHECK(VecSet(accepted, 77.0) == PETSC_SUCCESS);
  CHECK(ss_nonlinear_work_controlled_solve(
            &model.nonlinear, &policy, SS_NONLINEAR_CONTROL_SSR_INDIRECT,
            expected_state * (PetscReal)global_size, 1.2, parent, accepted,
            &accepted_factor, &record) == SS_OK);
  CHECK(record.outcome == SS_NONLINEAR_CONVERGED);
  CHECK(record.history_count > 0U && record.transaction_committed != 0);
  CHECK(maximum_value_error(accepted, expected_state, &state_error) == 0);
  CHECK(VecSum(accepted, &accepted_sum) == PETSC_SUCCESS);
  *maximum_error = fmax(state_error, fabs(accepted_factor - expected_factor));
  *checksum = PetscRealPart(accepted_sum) + accepted_factor;
  for (index = 0U; index < record.history_count; ++index) {
    const SsNonlinearIterationRecord *iteration = &record.history[index];
    const PetscReal expected_epsilon =
        sensitivity_method == SS_NONLINEAR_CONTROL_SENSITIVITY_STABLE_CENTRAL
            ? sensitivity_relative_step * fmax(1.0, fabs(iteration->control_before))
            : policy.tolerance / 1000.0;
    CHECK(iteration->control_sensitivity_method == sensitivity_method);
    CHECK(fabs(iteration->control_sensitivity_epsilon - expected_epsilon) <=
          4.0 * DBL_EPSILON * fmax(1.0, expected_epsilon));
    CHECK(iteration->control_linear_solve.status == SS_LINEAR_SOLVE_CONVERGED);
    CHECK(iteration->residual_linear_solve.status == SS_LINEAR_SOLVE_CONVERGED);
    CHECK(iteration->control_linear_solve.solve_count + 1U ==
          iteration->residual_linear_solve.solve_count);
    CHECK(iteration->augmented.outcome == SS_NONLINEAR_CONVERGED);
    CHECK(isfinite((double)iteration->augmented.denominator));
    CHECK(iteration->line_search.outcome == SS_NONLINEAR_CONVERGED);
    CHECK(iteration->alpha > 0.0 && iteration->alpha <= 1.0);
    CHECK(fabs(iteration->work_after - expected_state * (PetscReal)global_size) <=
          1.0e-12);
  }
  CHECK(VecDestroy(&accepted) == PETSC_SUCCESS);
  CHECK(VecDestroy(&parent) == PETSC_SUCCESS);
  CHECK(VecDestroy(&load) == PETSC_SUCCESS);
  CHECK(MatDestroy(&elastic) == PETSC_SUCCESS);
  return 0;
}

static int create_one_level_pmg(PetscInt global_size, SsPmg **pmg) {
  int world_size = 0;
  PetscMPIInt active_ranks[128];
  uint32_t order = 1U;
  SsPmgLevelPolicy level;
  SsPmgSpecification specification;
  int index;
  (void)global_size;
  CHECK(MPI_Comm_size(PETSC_COMM_WORLD, &world_size) == MPI_SUCCESS);
  CHECK(world_size > 0 && world_size <= 128);
  for (index = 0; index < world_size; ++index) {
    active_ranks[index] = (PetscMPIInt)index;
  }
  memset(&level, 0, sizeof(level));
  (void)snprintf(level.options_prefix, sizeof(level.options_prefix), "%s",
                 "nonlinear_lvl_");
  level.method = SS_PMG_LEVEL_FGMRES_JACOBI;
  level.active_rank_count = (size_t)world_size;
  level.active_ranks = active_ranks;
  level.pre_smooth_steps = 0;
  level.post_smooth_steps = 0;
  level.maximum_iterations = 80;
  level.rtol = 1.0e-12;
  level.atol = 1.0e-14;
  memset(&specification, 0, sizeof(specification));
  specification.fine_order = 1U;
  specification.level_count = 1U;
  specification.orders = &order;
  specification.level_policies = &level;
  specification.transfer_generation = 51U;
  specification.numeric_lag_interval = 1U;
  specification.memory_cap_bytes = 128U * 1024U * 1024U;
  CHECK(ss_pmg_create(PETSC_COMM_WORLD, &specification, pmg) == SS_OK);
  return 0;
}

static SsStatus analysis_fixed_solve(void *opaque, PetscReal factor, Vec incoming,
                                     Vec solution, SsNonlinearSolveRecord *record) {
  AnalysisModel *model = (AnalysisModel *)opaque;
  if (model->repeated_seed_mode != 0) {
    (void)incoming;
    memset(record, 0, sizeof(*record));
    record->outcome = SS_NONLINEAR_CONVERGED;
    record->iterations = 1;
    return VecSet(solution, factor <= 1.2 + 1.0e-12 ? 0.25 : 0.35) == PETSC_SUCCESS
               ? SS_OK
               : SS_ERROR_INTERNAL;
  }
  return ss_nonlinear_fixed_solve(&model->nonlinear, &model->nonlinear_policy, factor,
                                  incoming, solution, record);
}

static SsStatus analysis_total_potential(void *opaque, PetscReal factor, Vec state,
                                         PetscReal *total_potential) {
  AnalysisModel *model = (AnalysisModel *)opaque;
  Vec residual = NULL;
  SsStatus status;
  if (VecDuplicate(state, &residual) != PETSC_SUCCESS) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  status = distributed_evaluate(&model->model, state, factor, residual, NULL,
                                total_potential);
  if (VecDestroy(&residual) != PETSC_SUCCESS && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  return status;
}

static SsStatus analysis_work_solve(void *opaque, SsAnalysisMethod method,
                                    PetscReal target_work, PetscReal incoming_factor,
                                    Vec incoming, Vec solution,
                                    PetscReal *accepted_factor,
                                    SsNonlinearSolveRecord *record) {
  AnalysisModel *model = (AnalysisModel *)opaque;
  PetscInt global_size = 0;
  PetscReal value;
  if (method == SS_ANALYSIS_SSR_INDIRECT && model->repeated_seed_mode != 0) {
    memset(record, 0, sizeof(*record));
    record->outcome = SS_NONLINEAR_CONVERGED;
    record->iterations = 1;
    *accepted_factor = incoming_factor + 0.2;
    return VecSet(solution, target_work) == PETSC_SUCCESS ? SS_OK : SS_ERROR_INTERNAL;
  }
  if (method == SS_ANALYSIS_SSR_INDIRECT) {
    SsNonlinearPolicy policy = model->nonlinear_policy;
    policy.line_method = SS_NONLINEAR_LINE_RESIDUAL_MONOTONE;
    return ss_nonlinear_work_controlled_solve(
        &model->nonlinear, &policy, SS_NONLINEAR_CONTROL_SSR_INDIRECT, target_work,
        incoming_factor, incoming, solution, accepted_factor, record);
  }
  memset(record, 0, sizeof(*record));
  record->outcome = SS_NONLINEAR_CONVERGED;
  record->iterations = 5;
  if (VecGetSize(solution, &global_size) != PETSC_SUCCESS || global_size <= 0) {
    return SS_ERROR_INTERNAL;
  }
  value = target_work / (PetscReal)global_size;
  if (method != SS_ANALYSIS_LIMIT_LOAD) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *accepted_factor = sqrt(target_work);
  return VecSet(solution, value) == PETSC_SUCCESS ? SS_OK : SS_ERROR_INTERNAL;
}

static SsStatus analysis_begin_attempt(void *opaque, size_t attempt_ordinal,
                                       size_t parent_accepted_ordinal) {
  AnalysisModel *model = (AnalysisModel *)opaque;
  (void)attempt_ordinal;
  (void)parent_accepted_ordinal;
  if (model->transaction_open != 0) {
    return SS_ERROR_INVALID_STATE;
  }
  model->transaction_open = 1;
  ++model->transaction_begin_count;
  return SS_OK;
}

static SsStatus analysis_commit_attempt(void *opaque, size_t attempt_ordinal,
                                        size_t parent_accepted_ordinal) {
  AnalysisModel *model = (AnalysisModel *)opaque;
  (void)attempt_ordinal;
  (void)parent_accepted_ordinal;
  if (model->transaction_open == 0) {
    return SS_ERROR_INVALID_STATE;
  }
  model->transaction_open = 0;
  ++model->transaction_commit_count;
  return SS_OK;
}

static SsStatus analysis_rollback_attempt(void *opaque, size_t attempt_ordinal,
                                          size_t parent_accepted_ordinal) {
  AnalysisModel *model = (AnalysisModel *)opaque;
  (void)attempt_ordinal;
  (void)parent_accepted_ordinal;
  if (model->transaction_open == 0) {
    return SS_ERROR_INVALID_STATE;
  }
  model->transaction_open = 0;
  ++model->transaction_rollback_count;
  return SS_OK;
}

static SsStatus analysis_capture_accepted(void *opaque, size_t accepted_ordinal,
                                          Vec state) {
  AnalysisModel *model = (AnalysisModel *)opaque;
  if (model == NULL || state == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (model->captured_state_count == 0U) {
    model->captured_first_ordinal = accepted_ordinal;
  } else if (accepted_ordinal !=
             model->captured_first_ordinal + model->captured_state_count) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (model->current_accepted == NULL &&
      (VecDuplicate(state, &model->current_accepted) != PETSC_SUCCESS ||
       VecDuplicate(state, &model->previous_accepted) != PETSC_SUCCESS)) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  if ((model->captured_state_count != 0U &&
       VecCopy(model->current_accepted, model->previous_accepted) != PETSC_SUCCESS) ||
      VecCopy(state, model->current_accepted) != PETSC_SUCCESS ||
      (model->captured_state_count == 0U &&
       VecCopy(state, model->previous_accepted) != PETSC_SUCCESS)) {
    return SS_ERROR_INTERNAL;
  }
  ++model->captured_state_count;
  return SS_OK;
}

static int analysis_capture_reset(AnalysisModel *model) {
  CHECK(VecDestroy(&model->current_accepted) == PETSC_SUCCESS);
  CHECK(VecDestroy(&model->previous_accepted) == PETSC_SUCCESS);
  model->captured_state_count = 0U;
  model->captured_first_ordinal = 0U;
  return 0;
}

static void fill_analysis_problem(AnalysisModel *model, Vec load,
                                  SsAnalysisProblem *problem) {
  memset(problem, 0, sizeof(*problem));
  problem->communicator = PETSC_COMM_WORLD;
  problem->reference_load = load;
  problem->operations.solve_fixed = analysis_fixed_solve;
  problem->operations.solve_work_controlled = analysis_work_solve;
  problem->operations.total_potential = analysis_total_potential;
  problem->operations.capture_accepted_state = analysis_capture_accepted;
  problem->operations.begin_attempt = analysis_begin_attempt;
  problem->operations.commit_attempt = analysis_commit_attempt;
  problem->operations.rollback_attempt = analysis_rollback_attempt;
  problem->context = model;
}

static void fill_analysis_policy(SsAnalysisMethod method, SsAnalysisPolicy *policy) {
  memset(policy, 0, sizeof(*policy));
  policy->method = method;
  policy->nonlinear_tolerance = 1.0e-10;
  policy->factor_initial = 1.0;
  policy->factor_step_initial = 0.2;
  policy->factor_step_minimum = 0.0125;
  policy->factor_maximum = 4.0;
  policy->work_step_initial = 0.1;
  policy->work_step_minimum = 0.0125;
  policy->work_maximum = 0.85;
  policy->accepted_step_limit = 8;
  policy->consecutive_retry_limit = 5;
}

static int run_analysis_trajectories(PetscReal *checksum, PetscReal *maximum_error) {
  AnalysisModel model;
  SsAnalysisProblem problem;
  SsAnalysisPolicy policy;
  SsAnalysisRecord record;
  Vec load = NULL;
  Vec initial = NULL;
  Vec accepted = NULL;
  Mat elastic = NULL;
  PetscReal error = 0.0;
  CHECK(create_nonlinear(&model, PETSC_COMM_WORLD, 1, &load, &elastic) == 0);
  fill_analysis_problem(&model, load, &problem);
  CHECK(create_vector(PETSC_COMM_WORLD, 1, 0.0, &initial) == 0);
  CHECK(VecDuplicate(initial, &accepted) == PETSC_SUCCESS);

  fill_analysis_policy(SS_ANALYSIS_SSR_DIRECT, &policy);
  CHECK(ss_analysis_run(&problem, &policy, initial, accepted, &record) == SS_OK);
  CHECK(record.point_count == 8U);
  CHECK(record.terminal_success != 0 &&
        record.stop_reason == SS_ANALYSIS_STOP_ACCEPTED_STEP_LIMIT);
  CHECK(strcmp(record.method_identity, "ssr_direct/reference_fd") == 0);
  CHECK(record.commit_count == record.point_count && record.rollback_count == 0U);
  CHECK(record.newton_iteration_count > 0U && record.linear_setup_count > 0U &&
        record.linear_solve_count > 0U);
  error = fmax(error, fabs(record.final_factor - 2.4));
  error = fmax(error, fabs(record.final_control - 0.32253704883233003));
  *checksum = record.final_factor + record.final_control;
  CHECK(analysis_capture_reset(&model) == 0);

  fill_analysis_policy(SS_ANALYSIS_SSR_INDIRECT, &policy);
  CHECK(VecSet(initial, 0.0) == PETSC_SUCCESS);
  CHECK(ss_analysis_run(&problem, &policy, initial, accepted, &record) == SS_OK);
  CHECK(record.point_count == 7U);
  CHECK(record.terminal_success != 0 &&
        record.stop_reason == SS_ANALYSIS_STOP_WORK_HORIZON_REACHED);
  CHECK(strcmp(record.method_identity, "ssr_indirect/classic_fd") == 0);
  CHECK(record.commit_count == record.point_count && record.rollback_count == 0U);
  CHECK(record.newton_iteration_count > 0U && record.linear_setup_count > 0U &&
        record.linear_solve_count > 0U);
  error = fmax(error, fabs(record.final_factor - 2.546029411765137));
  error = fmax(error, fabs(record.final_work - 0.85));
  *checksum += record.final_factor + record.final_work;
  CHECK(analysis_capture_reset(&model) == 0);

  fill_analysis_policy(SS_ANALYSIS_SSR_INDIRECT, &policy);
  policy.work_growth_iteration_limit = 1;
  policy.indirect_sensitivity_method = SS_NONLINEAR_CONTROL_SENSITIVITY_STABLE_CENTRAL;
  policy.indirect_sensitivity_relative_step = 3.0e-6;
  CHECK(VecSet(initial, 0.0) == PETSC_SUCCESS);
  CHECK(ss_analysis_run(&problem, &policy, initial, accepted, &record) == SS_OK);
  CHECK(strcmp(record.method_identity, "ssr_indirect/stable_central_fd") == 0);
  CHECK(strcmp(record.controller_identity, "ssr_indirect/effort_limited") == 0);
  CHECK(record.point_count == 8U);
  CHECK(record.stop_reason == SS_ANALYSIS_STOP_ACCEPTED_STEP_LIMIT);
  CHECK(record.final_work < 0.85);
  CHECK(fabs(record.next_step - (record.points[1].work - record.points[0].work)) <
        1.0e-14);
  CHECK(analysis_capture_reset(&model) == 0);

  fill_analysis_policy(SS_ANALYSIS_SSR_INDIRECT, &policy);
  policy.work_step_controller = SS_ANALYSIS_WORK_STEP_PREDICTOR_ERROR;
  policy.work_growth_iteration_limit = 6;
  policy.work_predictor_error_target = 1.0;
  policy.work_step_scale_minimum = 0.5;
  policy.work_step_scale_maximum = 2.0;
  CHECK(VecSet(initial, 0.0) == PETSC_SUCCESS);
  CHECK(ss_analysis_run(&problem, &policy, initial, accepted, &record) == SS_OK);
  CHECK(strcmp(record.controller_identity, "ssr_indirect/predictor_error_state_only") ==
        0);
  CHECK(record.terminal_success != 0);
  {
    size_t index;
    int saw_predictor_error = 0;
    for (index = 0U; index < record.attempt_count; ++index) {
      if (record.attempts[index].predictor ==
              SS_ANALYSIS_PREDICTOR_DISPLACEMENT_SECANT_ANCHORED_FACTOR &&
          record.attempts[index].successful != 0 &&
          isfinite((double)record.attempts[index].predictor_error)) {
        saw_predictor_error = 1;
      }
    }
    CHECK(saw_predictor_error != 0);
  }
  CHECK(analysis_capture_reset(&model) == 0);

  fill_analysis_policy(SS_ANALYSIS_LIMIT_LOAD, &policy);
  policy.work_maximum = 0.4;
  CHECK(VecSet(initial, 0.0) == PETSC_SUCCESS);
  CHECK(ss_analysis_run(&problem, &policy, initial, accepted, &record) == SS_OK);
  CHECK(record.terminal_success != 0 &&
        record.stop_reason == SS_ANALYSIS_STOP_WORK_HORIZON_REACHED);
  CHECK(strcmp(record.method_identity, "limit_load/work_classic") == 0);
  error = fmax(error, fabs(record.final_control - 0.4));
  *checksum += record.final_factor + record.final_control;
  CHECK(analysis_capture_reset(&model) == 0);
  *maximum_error = error;
  CHECK(model.transaction_open == 0);
  CHECK(model.transaction_begin_count ==
        model.transaction_commit_count + model.transaction_rollback_count);
  CHECK(model.transaction_rollback_count == 0U);

  CHECK(VecDestroy(&accepted) == PETSC_SUCCESS);
  CHECK(VecDestroy(&initial) == PETSC_SUCCESS);
  CHECK(VecDestroy(&load) == PETSC_SUCCESS);
  CHECK(MatDestroy(&elastic) == PETSC_SUCCESS);
  return 0;
}

static int test_analysis_resume(PetscReal *checksum, PetscReal *maximum_error) {
  static const SsAnalysisMethod methods[] = {
      SS_ANALYSIS_LIMIT_LOAD,
      SS_ANALYSIS_SSR_DIRECT,
      SS_ANALYSIS_SSR_INDIRECT,
  };
  static const PetscInt source_limits[] = {1, 2};
  AnalysisModel model;
  SsAnalysisProblem problem;
  SsAnalysisPolicy target_policy;
  SsAnalysisPolicy source_policy;
  SsAnalysisRecord record;
  SsAnalysisResumeState resume;
  Vec load = NULL;
  Vec initial = NULL;
  Vec baseline = NULL;
  Vec source = NULL;
  Vec resumed = NULL;
  Vec source_previous = NULL;
  Vec source_current = NULL;
  Mat elastic = NULL;
  size_t method_index;
  size_t source_limit_index;
  PetscReal error = 0.0;
  PetscReal result_checksum = 0.0;
  CHECK(create_nonlinear(&model, PETSC_COMM_WORLD, 1, &load, &elastic) == 0);
  fill_analysis_problem(&model, load, &problem);
  CHECK(create_vector(PETSC_COMM_WORLD, 1, 0.0, &initial) == 0);
  CHECK(VecDuplicate(initial, &baseline) == PETSC_SUCCESS);
  CHECK(VecDuplicate(initial, &source) == PETSC_SUCCESS);
  CHECK(VecDuplicate(initial, &resumed) == PETSC_SUCCESS);
  for (method_index = 0U; method_index < sizeof(methods) / sizeof(methods[0]);
       ++method_index) {
    const SsAnalysisMethod method = methods[method_index];
    PetscReal baseline_factor;
    PetscReal baseline_control;
    PetscReal baseline_work;
    PetscReal difference = 0.0;
    PetscBool bitwise_equal = PETSC_FALSE;
    const SsAnalysisPointRecord *current_point;
    const SsAnalysisPointRecord *previous_point;
    fill_analysis_policy(method, &target_policy);
    if (method == SS_ANALYSIS_LIMIT_LOAD) {
      target_policy.work_maximum = 0.4;
    }
    CHECK(VecSet(initial, 0.0) == PETSC_SUCCESS);
    CHECK(ss_analysis_run(&problem, &target_policy, initial, baseline, &record) ==
          SS_OK);
    CHECK(record.terminal_success != 0);
    baseline_factor = record.final_factor;
    baseline_control = record.final_control;
    baseline_work = record.final_work;
    CHECK(analysis_capture_reset(&model) == 0);

    for (source_limit_index = 0U;
         source_limit_index < sizeof(source_limits) / sizeof(source_limits[0]);
         ++source_limit_index) {
      const PetscInt source_limit = source_limits[source_limit_index];
      const SsAnalysisNextOperation expected_next =
          source_limit == 1 && method != SS_ANALYSIS_LIMIT_LOAD
              ? SS_ANALYSIS_NEXT_SECOND_SEED
              : SS_ANALYSIS_NEXT_CONTINUATION_ATTEMPT;
      const int expected_terminal_success =
          expected_next == SS_ANALYSIS_NEXT_CONTINUATION_ATTEMPT;
      source_policy = target_policy;
      source_policy.accepted_step_limit = source_limit;
      CHECK(VecSet(initial, 0.0) == PETSC_SUCCESS);
      CHECK(ss_analysis_run(&problem, &source_policy, initial, source, &record) ==
            SS_OK);
      if (!(record.terminal_success == expected_terminal_success &&
            record.stop_reason == SS_ANALYSIS_STOP_ACCEPTED_STEP_LIMIT &&
            record.accepted_step_count == (size_t)source_limit &&
            record.point_count >= 1U && record.next_step > 0.0 &&
            record.next_operation == expected_next)) {
        fprintf(stderr,
                "resume source mismatch method=%d limit=%d success=%d stop=%d "
                "accepted=%zu points=%zu next_step=%.17g next=%d expected=%d\n",
                (int)method, (int)source_limit, record.terminal_success,
                (int)record.stop_reason, record.accepted_step_count, record.point_count,
                (double)record.next_step, (int)record.next_operation,
                (int)expected_next);
      }
      CHECK(record.terminal_success == expected_terminal_success &&
            record.stop_reason == SS_ANALYSIS_STOP_ACCEPTED_STEP_LIMIT &&
            record.accepted_step_count == (size_t)source_limit &&
            record.point_count >= 1U && record.next_step > 0.0 &&
            record.next_operation == expected_next &&
            record.resumed_from_accepted_state == 0);
      current_point = &record.points[record.point_count - 1U];
      previous_point = record.point_count > 1U ? &record.points[record.point_count - 2U]
                                               : current_point;
      CHECK(VecDuplicate(model.previous_accepted, &source_previous) == PETSC_SUCCESS);
      CHECK(VecDuplicate(model.current_accepted, &source_current) == PETSC_SUCCESS);
      CHECK(VecCopy(model.previous_accepted, source_previous) == PETSC_SUCCESS);
      CHECK(VecCopy(model.current_accepted, source_current) == PETSC_SUCCESS);
      memset(&resume, 0, sizeof(resume));
      resume.method = method;
      resume.accepted_step_offset = record.accepted_step_count;
      resume.accepted_ordinal = current_point->accepted_ordinal;
      resume.next_operation = record.next_operation;
      resume.previous_factor = previous_point->factor;
      resume.previous_control = previous_point->control;
      resume.previous_work = previous_point->work;
      resume.current_factor = current_point->factor;
      resume.current_control = current_point->control;
      resume.current_work = current_point->work;
      resume.current_signed_scaled_slope = current_point->signed_scaled_slope;
      resume.first_control = record.first_control;
      resume.next_step = record.next_step;
      resume.retry_count = 0;
      CHECK(analysis_capture_reset(&model) == 0);
      CHECK(ss_analysis_resume(&problem, &target_policy, source_previous,
                               source_current, resumed, &resume, &record) == SS_OK);
      CHECK(record.terminal_success != 0 && record.resumed_from_accepted_state != 0);
      error = fmax(error, fabs(record.final_factor - baseline_factor));
      error = fmax(error, fabs(record.final_control - baseline_control));
      error = fmax(error, fabs(record.final_work - baseline_work));
      CHECK(VecEqual(baseline, resumed, &bitwise_equal) == PETSC_SUCCESS);
      CHECK(VecAXPY(resumed, -1.0, baseline) == PETSC_SUCCESS);
      CHECK(VecNorm(resumed, NORM_INFINITY, &difference) == PETSC_SUCCESS);
      error = fmax(error, difference);
      CHECK(bitwise_equal == PETSC_TRUE);
      result_checksum += record.final_factor + record.final_control + record.final_work;
      CHECK(analysis_capture_reset(&model) == 0);
      CHECK(VecDestroy(&source_current) == PETSC_SUCCESS);
      CHECK(VecDestroy(&source_previous) == PETSC_SUCCESS);
    }
  }
  {
    PetscReal baseline_factor;
    PetscReal baseline_control;
    PetscReal baseline_work;
    PetscReal difference = 0.0;
    PetscBool bitwise_equal = PETSC_FALSE;
    const SsAnalysisPointRecord *current_point;
    const SsAnalysisPointRecord *previous_point;

    model.repeated_seed_mode = 1;
    fill_analysis_policy(SS_ANALYSIS_SSR_INDIRECT, &target_policy);
    CHECK(VecSet(initial, 0.0) == PETSC_SUCCESS);
    CHECK(ss_analysis_run(&problem, &target_policy, initial, baseline, &record) ==
          SS_OK);
    CHECK(record.terminal_success != 0 &&
          record.stop_reason == SS_ANALYSIS_STOP_ACCEPTED_STEP_LIMIT);
    baseline_factor = record.final_factor;
    baseline_control = record.final_control;
    baseline_work = record.final_work;
    CHECK(analysis_capture_reset(&model) == 0);

    source_policy = target_policy;
    source_policy.accepted_step_limit = 2;
    CHECK(VecSet(initial, 0.0) == PETSC_SUCCESS);
    CHECK(ss_analysis_run(&problem, &source_policy, initial, source, &record) == SS_OK);
    CHECK(record.terminal_success == 0 &&
          record.stop_reason == SS_ANALYSIS_STOP_ACCEPTED_STEP_LIMIT &&
          record.accepted_step_count == 2U && record.point_count == 2U &&
          record.next_operation == SS_ANALYSIS_NEXT_SECOND_SEED);
    previous_point = &record.points[0];
    current_point = &record.points[1];
    CHECK(VecDuplicate(model.previous_accepted, &source_previous) == PETSC_SUCCESS);
    CHECK(VecDuplicate(model.current_accepted, &source_current) == PETSC_SUCCESS);
    CHECK(VecCopy(model.previous_accepted, source_previous) == PETSC_SUCCESS);
    CHECK(VecCopy(model.current_accepted, source_current) == PETSC_SUCCESS);
    memset(&resume, 0, sizeof(resume));
    resume.method = SS_ANALYSIS_SSR_INDIRECT;
    resume.accepted_step_offset = record.accepted_step_count;
    resume.accepted_ordinal = current_point->accepted_ordinal;
    resume.next_operation = record.next_operation;
    resume.previous_factor = previous_point->factor;
    resume.previous_control = previous_point->control;
    resume.previous_work = previous_point->work;
    resume.current_factor = current_point->factor;
    resume.current_control = current_point->control;
    resume.current_work = current_point->work;
    resume.current_signed_scaled_slope = current_point->signed_scaled_slope;
    resume.first_control = record.first_control;
    resume.next_step = record.next_step;
    resume.retry_count = 0;
    CHECK(analysis_capture_reset(&model) == 0);
    CHECK(ss_analysis_resume(&problem, &target_policy, source_previous, source_current,
                             resumed, &resume, &record) == SS_OK);
    CHECK(record.terminal_success != 0 && record.resumed_from_accepted_state != 0);
    error = fmax(error, fabs(record.final_factor - baseline_factor));
    error = fmax(error, fabs(record.final_control - baseline_control));
    error = fmax(error, fabs(record.final_work - baseline_work));
    CHECK(VecEqual(baseline, resumed, &bitwise_equal) == PETSC_SUCCESS);
    CHECK(VecAXPY(resumed, -1.0, baseline) == PETSC_SUCCESS);
    CHECK(VecNorm(resumed, NORM_INFINITY, &difference) == PETSC_SUCCESS);
    error = fmax(error, difference);
    CHECK(bitwise_equal == PETSC_TRUE);
    CHECK(analysis_capture_reset(&model) == 0);
    CHECK(VecDestroy(&source_current) == PETSC_SUCCESS);
    CHECK(VecDestroy(&source_previous) == PETSC_SUCCESS);
    model.repeated_seed_mode = 0;
  }
  CHECK(model.transaction_open == 0);
  CHECK(model.transaction_begin_count ==
        model.transaction_commit_count + model.transaction_rollback_count);
  CHECK(VecDestroy(&resumed) == PETSC_SUCCESS);
  CHECK(VecDestroy(&source) == PETSC_SUCCESS);
  CHECK(VecDestroy(&baseline) == PETSC_SUCCESS);
  CHECK(VecDestroy(&initial) == PETSC_SUCCESS);
  CHECK(VecDestroy(&load) == PETSC_SUCCESS);
  CHECK(MatDestroy(&elastic) == PETSC_SUCCESS);
  *checksum = result_checksum;
  *maximum_error = error;
  return 0;
}

static int test_layout(PetscReal *checksum, PetscReal *maximum_error) {
  SsPmg *pmg = NULL;
  PetscReal pmg_error = 0.0;
  PetscReal pmg_checksum = 0.0;
  PetscReal analysis_error = 0.0;
  PetscReal analysis_checksum = 0.0;
  int pmg_selected = 0;
  CHECK(create_one_level_pmg(18, &pmg) == 0);
  CHECK(run_world_fixed(18, pmg, &pmg_error, &pmg_checksum, &pmg_selected) == 0);
  CHECK(pmg_selected != 0);
  CHECK(ss_pmg_destroy(&pmg) == SS_OK);
  CHECK(run_analysis_trajectories(&analysis_checksum, &analysis_error) == 0);
  *maximum_error = fmax(pmg_error, analysis_error);
  *checksum = pmg_checksum + analysis_checksum;
  return 0;
}

static int publish_result(const Arguments *arguments, int ranks, int threads,
                          PetscReal checksum, PetscReal maximum_error) {
  int rank = 0;
  CHECK(MPI_Comm_rank(PETSC_COMM_WORLD, &rank) == MPI_SUCCESS);
  if (rank == 0) {
    FILE *stream = fopen(arguments->output, "wb");
    int written;
    int close_status;
    if (stream == NULL) {
      return 1;
    }
    written = fprintf(stream,
                      "{\"checksum\":%.17g,\"invariant\":\"%s\","
                      "\"maximum_error\":%.17g,\"observed_threads\":%d,"
                      "\"ranks\":%d,\"requested_threads\":%d,"
                      "\"test_id\":\"%s\"}\n",
                      (double)checksum, CONTRACT_INVARIANT, (double)maximum_error,
                      threads, ranks, arguments->expected_threads, arguments->test_id);
    close_status = fclose(stream);
    CHECK(written > 0 && close_status == 0);
  }
  CHECK(MPI_Barrier(PETSC_COMM_WORLD) == MPI_SUCCESS);
  return 0;
}

int main(int argc, char **argv) {
  Arguments arguments;
  PetscReal checksum = 0.0;
  PetscReal maximum_error = 0.0;
  int ranks = 0;
  int threads;
  int pmg_selected = 0;
  int result = 1;
  if (!parse_arguments(argc, argv, &arguments)) {
    return 2;
  }
  if (PetscInitialize(&argc, &argv, NULL, NULL) != PETSC_SUCCESS) {
    return 1;
  }
  CHECK(MPI_Comm_size(PETSC_COMM_WORLD, &ranks) == MPI_SUCCESS);
  threads = observed_threads();
  CHECK(ranks == arguments.expected_ranks && threads == arguments.expected_threads);
  if (strcmp(arguments.test_id, "NLS-CONTRACT-MPI-001") == 0) {
    PetscReal controlled_checksum = 0.0;
    PetscReal controlled_error = 0.0;
    PetscReal central_checksum = 0.0;
    PetscReal central_error = 0.0;
    CHECK(run_world_fixed(12, NULL, &maximum_error, &checksum, &pmg_selected) == 0);
    CHECK(pmg_selected == 0);
    CHECK(run_world_controlled(12, SS_NONLINEAR_CONTROL_SENSITIVITY_REFERENCE_FORWARD,
                               0.0, &controlled_error, &controlled_checksum) == 0);
    CHECK(run_world_controlled(12, SS_NONLINEAR_CONTROL_SENSITIVITY_STABLE_CENTRAL,
                               6.0e-6, &central_error, &central_checksum) == 0);
    maximum_error = fmax(maximum_error, fmax(controlled_error, central_error));
    checksum += controlled_checksum + central_checksum;
    result = 0;
  } else if (strcmp(arguments.test_id, "ANA-CONTRACT-RESTART-001") == 0) {
    result = test_analysis_resume(&checksum, &maximum_error);
  } else if (strcmp(arguments.test_id, "ANA-CONTRACT-LAYOUT-001") == 0) {
    result = test_layout(&checksum, &maximum_error);
  } else {
    fprintf(stderr, "unknown nonlinear or analysis MPI test: %s\n", arguments.test_id);
  }
  if (result == 0) {
    result = publish_result(&arguments, ranks, threads, checksum, maximum_error);
  }
  if (PetscFinalize() != PETSC_SUCCESS) {
    return 1;
  }
  return result;
}
