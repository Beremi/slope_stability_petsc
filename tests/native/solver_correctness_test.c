/* Solver admission, projection and stopping regressions.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/nonlinear/nonlinear.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr)                                                                    \
  do {                                                                                 \
    if (!(expr)) {                                                                     \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr);                       \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

static int diagonal(MPI_Comm comm, PetscInt n, const PetscScalar *values, Mat *matrix) {
  PetscInt start, end;
  CHECK(MatCreateAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, n, n, 1, NULL, 0, NULL,
                     matrix) == PETSC_SUCCESS);
  CHECK(MatGetOwnershipRange(*matrix, &start, &end) == PETSC_SUCCESS);
  for (PetscInt i = start; i < end; ++i) {
    CHECK(MatSetValue(*matrix, i, i, values[i], INSERT_VALUES) == PETSC_SUCCESS);
  }
  CHECK(MatAssemblyBegin(*matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  CHECK(MatAssemblyEnd(*matrix, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  return 0;
}

static int set_values(Vec vector, const PetscScalar *values) {
  PetscInt start, end;
  PetscScalar *array = NULL;
  CHECK(VecGetOwnershipRange(vector, &start, &end) == PETSC_SUCCESS);
  CHECK(VecGetArray(vector, &array) == PETSC_SUCCESS);
  for (PetscInt i = start; i < end; ++i) {
    array[i - start] = values[i];
  }
  CHECK(VecRestoreArray(vector, &array) == PETSC_SUCCESS);
  return 0;
}

static SsDeflationPolicy deflation_policy(SsDeflationProjector projector) {
  return (SsDeflationPolicy){
      .dimension_cap = 4U,
      .norm_tolerance = 1.0e-14,
      .pivot_tolerance = 1.0e-13,
      .condition_limit = 1.0e12,
      .invariant_tolerance = 1.0e-10,
      .reorthogonalization_passes = 2U,
      .projector = projector,
      .compression_history_decay = 1.0,
  };
}

static SsLinearPolicy linear_policy(void) {
  return (SsLinearPolicy){
      .identity = "review-regression",
      .options_prefix = "review_",
      .operator_class = SS_LINEAR_OPERATOR_GENERAL,
      .method = SS_LINEAR_METHOD_FGMRES,
      .pc = SS_LINEAR_PC_NONE,
      .side = SS_LINEAR_SIDE_RIGHT,
      .rtol = 1.0e-3,
      .atol = 1.0e-12,
      .dtol = 1.0e6,
      .maximum_iterations = 16,
      .restart = 10,
      .vector_memory_cap_bytes = 1024U * 1024U,
  };
}

static int signed_projection(void) {
  const PetscScalar entries[] = {-2.0, 3.0, 5.0};
  const PetscScalar unit[] = {1.0, 1.0, 0.0};
  SsDeflationPolicy policy =
      deflation_policy(SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL);
  SsDeflation *basis = NULL;
  SsDeflationAdmissionRecord admission;
  Mat a = NULL;
  Vec column = NULL, x = NULL, px = NULL, ppx = NULL;
  PetscReal error;
  CHECK(diagonal(PETSC_COMM_WORLD, 3, entries, &a) == 0);
  CHECK(MatCreateVecs(a, &column, &x) == PETSC_SUCCESS);
  CHECK(VecDuplicate(x, &px) == PETSC_SUCCESS);
  CHECK(VecDuplicate(x, &ppx) == PETSC_SUCCESS);
  for (PetscInt n = 1; n <= 2; ++n) {
    CHECK(ss_deflation_create(PETSC_COMM_WORLD, &policy, &basis) == SS_OK);
    CHECK(ss_deflation_begin(basis, 1U) == SS_OK);
    for (PetscInt i = 0; i < n; ++i) {
      PetscScalar values[] = {0.0, 0.0, 0.0};
      values[i] = 1.0;
      CHECK(set_values(column, values) == 0);
      CHECK(ss_deflation_add_candidate(basis, column, column, "signed", &admission) ==
            SS_OK);
      CHECK(admission.admitted);
    }
    CHECK(ss_deflation_refresh(basis, a, 1U) == SS_OK);
    CHECK(ss_deflation_invariant_error(basis, a, &error) == SS_OK && error < 1.0e-12);
    CHECK(set_values(x, unit) == 0);
    CHECK(ss_deflation_apply_right(basis, a, x, px) == SS_OK);
    CHECK(ss_deflation_apply_cached_right(basis, x, ppx) == SS_OK);
    CHECK(VecAXPY(ppx, -1.0, px) == PETSC_SUCCESS);
    CHECK(VecNorm(ppx, NORM_2, &error) == PETSC_SUCCESS && error < 1.0e-12);
    CHECK(ss_deflation_apply_cached_right(basis, px, ppx) == SS_OK);
    CHECK(VecAXPY(ppx, -1.0, px) == PETSC_SUCCESS);
    CHECK(VecNorm(ppx, NORM_2, &error) == PETSC_SUCCESS && error < 1.0e-12);
    CHECK(ss_deflation_coarse_initial_guess_and_project_rhs(basis, x, px, ppx) ==
          SS_OK);
    CHECK(MatMult(a, px, column) == PETSC_SUCCESS);
    CHECK(VecAXPY(column, 1.0, ppx) == PETSC_SUCCESS);
    CHECK(VecAXPY(column, -1.0, x) == PETSC_SUCCESS);
    CHECK(VecNorm(column, NORM_2, &error) == PETSC_SUCCESS && error < 1.0e-12);
    CHECK(ss_deflation_destroy(&basis) == SS_OK);
  }
  CHECK(VecDestroy(&ppx) == PETSC_SUCCESS);
  CHECK(VecDestroy(&px) == PETSC_SUCCESS);
  CHECK(VecDestroy(&x) == PETSC_SUCCESS);
  CHECK(VecDestroy(&column) == PETSC_SUCCESS);
  CHECK(MatDestroy(&a) == PETSC_SUCCESS);
  return 0;
}

static int minres_pc(void) {
  SsLinearPolicy policy = linear_policy();
  SsLinearOperatorProperties properties = {
      .finite_action = 1,
      .measured_symmetric = 1,
      .pc_symmetric = 1,
  };
  policy.method = SS_LINEAR_METHOD_MINRES;
  policy.operator_class = SS_LINEAR_OPERATOR_SYMMETRIC_INDEFINITE;
  policy.side = SS_LINEAR_SIDE_LEFT;
  CHECK(ss_linear_policy_validate(&policy, &properties) == SS_ERROR_TYPE_MISMATCH);
  properties.pc_positive = 1;
  CHECK(ss_linear_policy_validate(&policy, &properties) == SS_OK);
  return 0;
}

static int linear_acceptance(SsLinearStoppingCondition mode) {
  const PetscScalar entries[] = {1.0, 2.0, 2.8};
  const PetscScalar scaled_entries[] = {1.0e16, 2.0e16, 5.0e16};
  const PetscScalar rhs_values[] = {0.0, 1.0, 1.0};
  const int projected = mode == SS_LINEAR_STOP_PROJECTED_RHS_PETSC;
  SsLinearProblem problem = {
      .properties.finite_action = 1,
      .structure_generation = 1U,
      .value_generation = 1U,
      .policy_chain.count = 1U,
  };
  SsLinearPolicy policy = linear_policy();
  SsLinearSolver *solver = NULL;
  SsDeflation *basis = NULL;
  SsLinearSetupRecord setup;
  SsLinearSolveRecord solve;
  Mat a = NULL;
  Vec x = NULL, b = NULL, left = NULL;
  CHECK(diagonal(PETSC_COMM_SELF, 3, projected ? entries : scaled_entries, &a) == 0);
  if (!projected) {
    CHECK(MatSetOption(a, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE) ==
          PETSC_SUCCESS);
    CHECK(MatSetValue(a, 1, 2, 1.5e16, INSERT_VALUES) == PETSC_SUCCESS);
    CHECK(MatSetValue(a, 2, 1, 1.5e16, INSERT_VALUES) == PETSC_SUCCESS);
    CHECK(MatAssemblyBegin(a, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
    CHECK(MatAssemblyEnd(a, MAT_FINAL_ASSEMBLY) == PETSC_SUCCESS);
  }
  CHECK(MatCreateVecs(a, &x, &b) == PETSC_SUCCESS);
  CHECK(set_values(b, rhs_values) == 0);
  if (projected) {
    const PetscScalar left_values[] = {1.0, 10.0, 0.0};
    const PetscScalar right_values[] = {1.0, 0.0, 0.0};
    SsDeflationAdmissionRecord admission;
    SsDeflationPolicy def_policy =
        deflation_policy(SS_DEFLATION_PROJECTOR_GENERAL_LEFT_RIGHT);
    CHECK(ss_deflation_create(PETSC_COMM_SELF, &def_policy, &basis) == SS_OK);
    CHECK(ss_deflation_begin(basis, 1U) == SS_OK);
    CHECK(VecDuplicate(x, &left) == PETSC_SUCCESS);
    CHECK(set_values(left, left_values) == 0);
    CHECK(set_values(x, right_values) == 0);
    CHECK(ss_deflation_add_candidate(basis, left, x, "oblique", &admission) == SS_OK);
    CHECK(admission.admitted);
    policy.rtol = 0.1;
  } else {
    policy.method = SS_LINEAR_METHOD_FCG;
    policy.operator_class = SS_LINEAR_OPERATOR_SPD_VERIFIED;
    policy.pc = SS_LINEAR_PC_JACOBI;
    policy.side = SS_LINEAR_SIDE_LEFT;
    policy.atol = 1.0e-6;
    problem.properties.measured_symmetric = problem.properties.positive_energy = 1;
    problem.properties.pc_symmetric = problem.properties.pc_positive = 1;
  }
  policy.stopping_condition = mode;
  problem.operator_handle = a;
  problem.deflation = basis;
  problem.policy_chain.policies[0] = policy;
  CHECK(VecSet(x, 0.0) == PETSC_SUCCESS);
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_OK);
  CHECK(ss_linear_solve(solver, b, x, &solve) == SS_OK);
  CHECK(solve.reason > 0);
  CHECK(solve.true_residual_norm > fmax(policy.atol, policy.rtol * solve.rhs_norm));
  CHECK(solve.status == SS_LINEAR_SOLVE_RESIDUAL_VERIFICATION_FAILED);
  CHECK(ss_linear_destroy(&solver) == SS_OK);
  CHECK(ss_deflation_destroy(&basis) == SS_OK);
  CHECK(VecDestroy(&left) == PETSC_SUCCESS);
  CHECK(VecDestroy(&x) == PETSC_SUCCESS);
  CHECK(VecDestroy(&b) == PETSC_SUCCESS);
  CHECK(MatDestroy(&a) == PETSC_SUCCESS);
  return 0;
}

static int memory_budget(void) {
  const PetscScalar entries[] = {1.0, 2.0, 5.0};
  SsLinearPolicy policy = linear_policy();
  SsLinearProblem problem = {
      .properties.finite_action = 1,
      .structure_generation = 1U,
      .value_generation = 1U,
      .policy_chain.count = 1U,
  };
  SsLinearSolver *solver = NULL;
  SsLinearSetupRecord setup;
  Mat a = NULL;
  CHECK(diagonal(PETSC_COMM_SELF, 3, entries, &a) == 0);
  problem.operator_handle = a;
  policy.vector_memory_cap_bytes = 3U * 18U * sizeof(PetscScalar);
  problem.policy_chain.policies[0] = policy;
  CHECK(ss_linear_create(PETSC_COMM_SELF, &solver) == SS_OK);
  CHECK(ss_linear_setup(solver, &problem, &setup) == SS_ERROR_RESOURCE);
  CHECK(ss_linear_destroy(&solver) == SS_OK);
  CHECK(MatDestroy(&a) == PETSC_SUCCESS);
  return 0;
}

typedef struct {
  PetscReal state_shift;
  PetscReal control_scale;
} AffineResidual;

static SsStatus affine_evaluate(void *context, Vec state, PetscReal control,
                                Vec residual, Mat tangent, PetscReal *functional) {
  const AffineResidual *model = context;
  (void)functional;
  if (VecCopy(state, residual) != PETSC_SUCCESS ||
      VecShift(residual, model->state_shift + model->control_scale * (control - 1.0)) !=
          PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (tangent != NULL && (MatZeroEntries(tangent) != PETSC_SUCCESS ||
                          MatShift(tangent, 1.0) != PETSC_SUCCESS)) {
    return SS_ERROR_INTERNAL;
  }
  return SS_OK;
}

static SsStatus affine_controlled(void *context, SsNonlinearControlMode mode, Vec state,
                                  PetscReal control, Vec residual, Mat tangent,
                                  PetscReal *functional) {
  (void)mode;
  return affine_evaluate(context, state, control, residual, tangent, functional);
}

typedef enum {
  RESIDUAL_STRICT,
  SENSITIVITY_TIGHT,
  STATIONARY,
  QUANTIZED_CONTROL,
  LOAD_ONLY_UPDATE,
} NonlinearFixture;

static int nonlinear_stopping(int augmented, NonlinearFixture fixture) {
  const PetscScalar entries[] = {1.0};
  SsNonlinearProblem problem = {
      .communicator = PETSC_COMM_SELF,
      .operations.evaluate = affine_evaluate,
      .operations.evaluate_controlled = affine_controlled,
      .linear_properties.finite_action = 1,
      .structure_generation = 1U,
      .first_value_generation = 1U,
      .transaction_generation = 1U,
      .accepted_generation = 1U,
      .linear_policies.count = 1U,
  };
  SsNonlinearPolicy policy = {
      .tolerance = 1.0e-4,
      .stopping_criterion = SS_NONLINEAR_STOP_RELATIVE_RESIDUAL,
      .stopping_tolerance = 1.0e-8,
      .maximum_iterations = 1,
      .maximum_line_search_trials = 4,
      .minimum_alpha = 0.125,
      .regularization_minimum = 0.01,
      .singular_relative_tolerance = 1.0e-12,
      .relaxed_factor = 10.0,
      .line_method = SS_NONLINEAR_LINE_RESIDUAL_MONOTONE,
  };
  AffineResidual model = {.state_shift = -0.5 + 1.0e-5, .control_scale = 1.0};
  static SsNonlinearSolveRecord solve;
  Vec parent = NULL, accepted = NULL;
  PetscReal accepted_control = -1.0;
  SsNonlinearControlMode mode = SS_NONLINEAR_CONTROL_SSR_INDIRECT;
  CHECK(diagonal(PETSC_COMM_SELF, 1, entries, &problem.elastic_reference) == 0);
  CHECK(MatCreateVecs(problem.elastic_reference, &parent, &problem.reference_load) ==
        PETSC_SUCCESS);
  CHECK(VecDuplicate(parent, &accepted) == PETSC_SUCCESS);
  CHECK(VecSet(problem.reference_load, 1.0) == PETSC_SUCCESS);
  CHECK(VecSet(parent, 0.5) == PETSC_SUCCESS);
  problem.context = &model;
  problem.linear_policies.policies[0] = linear_policy();
  if (fixture == SENSITIVITY_TIGHT) {
    policy.tolerance = 1.0e-13;
    policy.stopping_tolerance = 1.0e-13;
  } else if (fixture == STATIONARY) {
    model.state_shift = -0.5;
    policy.stopping_criterion = SS_NONLINEAR_STOP_RELATIVE_CORRECTION;
  } else if (fixture == QUANTIZED_CONTROL) {
    model.state_shift = -0.5 + PETSC_MACHINE_EPSILON / 2.0;
    model.control_scale = 8.0;
    policy.stopping_criterion = SS_NONLINEAR_STOP_RELATIVE_CORRECTION;
  } else if (fixture == LOAD_ONLY_UPDATE) {
    model.state_shift = -0.5 + 0.0125;
    model.control_scale = -1.0;
    mode = SS_NONLINEAR_CONTROL_LIMIT_LOAD;
    policy.stopping_criterion = SS_NONLINEAR_STOP_RELATIVE_CORRECTION;
    policy.line_method = SS_NONLINEAR_LINE_DIRECTIONAL;
  }
  if (augmented) {
    CHECK(ss_nonlinear_work_controlled_solve(&problem, &policy, mode, 0.5, 1.0, parent,
                                             accepted, &accepted_control,
                                             &solve) == SS_OK);
  } else {
    CHECK(ss_nonlinear_fixed_solve(&problem, &policy, 1.0, parent, accepted, &solve) ==
          SS_OK);
  }
  CHECK(solve.outcome == SS_NONLINEAR_CONVERGED);
  CHECK(solve.iterations == (fixture == STATIONARY ? 0 : 1));
  CHECK(solve.final_convergence <= policy.stopping_tolerance);
  if (fixture == SENSITIVITY_TIGHT) {
    CHECK(1.0 + solve.history[0].control_sensitivity_epsilon > 1.0);
    CHECK(fabs(accepted_control - (1.0 - 1.0e-5)) < 1.0e-13);
  }
  if (fixture == STATIONARY || fixture == QUANTIZED_CONTROL ||
      fixture == LOAD_ONLY_UPDATE) {
    CHECK(solve.final_relative_correction <= policy.stopping_tolerance);
    const PetscReal expected_control = fixture == LOAD_ONLY_UPDATE ? 1.0125 : 1.0;
    CHECK(!augmented || fabs(accepted_control - expected_control) < 1.0e-13);
  }
  CHECK(VecDestroy(&accepted) == PETSC_SUCCESS);
  CHECK(VecDestroy(&parent) == PETSC_SUCCESS);
  CHECK(VecDestroy(&problem.reference_load) == PETSC_SUCCESS);
  CHECK(MatDestroy(&problem.elastic_reference) == PETSC_SUCCESS);
  return 0;
}

static int test_coarse_pmg(Mat matrix, size_t active, SsPmgFailurePoint inject,
                           SsPmg **pmg) {
  const uint32_t order = 1U;
  const SsPmgLevelPolicy policy = {
      .options_prefix = "review_p1_",
      .method = SS_PMG_LEVEL_FGMRES_JACOBI,
      .active_rank_count = active,
      .maximum_iterations = 4,
      .rtol = 1.0e-10,
      .atol = 1.0e-14,
  };
  const SsPmgSpecification spec = {
      .fine_order = order,
      .level_count = 1U,
      .orders = &order,
      .level_policies = &policy,
      .transfer_generation = 1U,
      .numeric_lag_interval = 1U,
      .memory_cap_bytes = 1024U * 1024U,
      .injected_failure = inject,
  };
  SsPmgSetupRecord setup;
  CHECK(ss_pmg_create(PetscObjectComm((PetscObject)matrix), &spec, pmg) == SS_OK);
  CHECK(ss_pmg_setup(*pmg, matrix, 1U, 1U, &setup) == SS_OK);
  return 0;
}

static SsStatus reject_diagnostic(void *context, const SsLinearProblem *linear,
                                  Vec state, uint64_t transaction, PetscInt iteration,
                                  PetscReal control, PetscReal regularization,
                                  size_t rhs_count, const Vec *rhs) {
  (void)linear;
  (void)state;
  (void)transaction;
  (void)iteration;
  (void)control;
  (void)regularization;
  (void)rhs_count;
  (void)rhs;
  (void)context;
  PetscMPIInt rank = 0;
  if (MPI_Comm_rank(PETSC_COMM_WORLD, &rank) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  return rank == 1 ? SS_ERROR_INJECTED : SS_OK;
}

typedef struct {
  Mat a;
  int reject;
} ContinuationModel;

static SsStatus continuation_evaluate(void *context, Vec state, PetscReal control,
                                      Vec residual, Mat tangent,
                                      PetscReal *functional) {
  const ContinuationModel *model = context;
  PetscScalar *array = NULL;
  PetscInt first, end;
  (void)functional;
  if (MatMult(model->a, state, residual) != PETSC_SUCCESS ||
      VecGetOwnershipRange(residual, &first, &end) != PETSC_SUCCESS ||
      VecGetArray(residual, &array) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  for (PetscInt i = first; i < end; ++i) {
    array[i - first] -= control + 0.25 * sin((PetscReal)(i + 1) * control);
  }
  if (VecRestoreArray(residual, &array) != PETSC_SUCCESS ||
      (tangent != NULL &&
       MatCopy(model->a, tangent, SAME_NONZERO_PATTERN) != PETSC_SUCCESS)) {
    return SS_ERROR_INTERNAL;
  }
  return SS_OK;
}

static SsStatus continuation_admissible(void *context, Vec state, PetscReal control,
                                        int *admissible) {
  const ContinuationModel *model = context;
  (void)state;
  (void)control;
  *admissible = !model->reject;
  return SS_OK;
}

static SsNonlinearPolicy continuation_policy(void) {
  return (SsNonlinearPolicy){
      .tolerance = 1.0e-9,
      .stopping_criterion = SS_NONLINEAR_STOP_RELATIVE_RESIDUAL,
      .stopping_tolerance = 1.0e-9,
      .maximum_iterations = 2,
      .maximum_line_search_trials = 4,
      .minimum_alpha = 0.125,
      .regularization_minimum = 0.01,
      .singular_relative_tolerance = 1.0e-12,
      .relaxed_factor = 10.0,
      .line_method = SS_NONLINEAR_LINE_RESIDUAL_MONOTONE,
  };
}

static int status_on_all_ranks(SsStatus local, SsStatus expected) {
  int match = local == expected;
  int all_match = 0;
  CHECK(MPI_Allreduce(&match, &all_match, 1, MPI_INT, MPI_MIN, PETSC_COMM_WORLD) ==
        MPI_SUCCESS);
  if (!all_match) {
    fprintf(stderr, "status %d, expected %d\n", (int)local, (int)expected);
  }
  return all_match;
}

static int collective_failures(void) {
  const PetscScalar values[] = {1.0, 2.0, 3.0};
  const PetscReal coordinates[] = {0.0, 0.0};
  PetscMPIInt rank, ranks;
  PetscInt first, end, dof;
  SsPmg *pmg = NULL;
  SsLinearSolver *solver = NULL;
  SsDeflation *basis = NULL;
  SsPmgApplyRecord apply;
  SsLinearSetupRecord setup;
  SsLinearSolveRecord solve;
  SsDeflationAdmissionRecord admission;
  Mat a = NULL;
  Vec b = NULL, x = NULL;
  MatNullSpace space = NULL;
  SsPmgNullspaceRecord modes;
  static SsNonlinearSolveRecord nonlinear_record;
  CHECK(MPI_Comm_rank(PETSC_COMM_WORLD, &rank) == MPI_SUCCESS);
  CHECK(MPI_Comm_size(PETSC_COMM_WORLD, &ranks) == MPI_SUCCESS);
  CHECK(ranks == 2 || ranks == 4);
  CHECK(diagonal(PETSC_COMM_WORLD, 3, values, &a) == 0);
  CHECK(MatCreateVecs(a, &x, &b) == PETSC_SUCCESS);
  CHECK(VecSet(b, 1.0) == PETSC_SUCCESS);
  CHECK(MatGetOwnershipRange(a, &first, &end) == PETSC_SUCCESS);
  dof = rank == 1 ? 6 : first;
  SsPmgNearNullspaceInput input = {
      .field = SS_PMG_FIELD_HYDRAULIC,
      .dimension = 2U,
      .hydraulic_pure_neumann = 1,
      .owned_node_count = end > first ? 1U : 0U,
      .coordinates = coordinates,
      .global_dof_indices = &dof,
      .dependence_tolerance = 1.0e-12,
  };
  CHECK(status_on_all_ranks(ss_pmg_near_nullspace_create(a, &input, &space, &modes),
                            SS_ERROR_INVALID_ARGUMENT));
  dof = first;
  CHECK(status_on_all_ranks(ss_pmg_near_nullspace_create(a, &input, &space, &modes),
                            SS_OK));
  CHECK(modes.retained_mode_count == 1U);
  CHECK(MatNullSpaceDestroy(&space) == PETSC_SUCCESS);
  SsLinearProblem linear = {
      .operator_handle = a,
      .properties.finite_action = 1,
      .policy_chain.count = 1U,
      .structure_generation = 1U,
      .value_generation = 1U,
  };
  for (int reduced = 0; reduced < 2; ++reduced) {
    const int failed_rank = reduced ? ranks - 1 : 0;
    CHECK(test_coarse_pmg(a, reduced ? 1U : (size_t)ranks,
                          rank == failed_rank ? SS_PMG_FAIL_LEVEL_SOLVE
                                              : SS_PMG_FAIL_NONE,
                          &pmg) == 0);
    CHECK(status_on_all_ranks(ss_pmg_apply_preconditioner(pmg, b, x, &apply),
                              SS_ERROR_INJECTED));
    CHECK(apply.failed_point == SS_PMG_FAIL_LEVEL_SOLVE && apply.failed_level == 0U);

    /* Exercise the same local fault through the shell PC. In the four-rank
     * reduced case the failing rank also has no entries to poison. */
    linear.policy_chain.policies[0] = linear_policy();
    linear.policy_chain.policies[0].pc = SS_LINEAR_PC_PMG;
    linear.pmg = pmg;
    CHECK(ss_linear_create(PETSC_COMM_WORLD, &solver) == SS_OK);
    CHECK(status_on_all_ranks(ss_linear_setup(solver, &linear, &setup), SS_OK));
    CHECK(VecSet(x, 0.0) == PETSC_SUCCESS);
    CHECK(status_on_all_ranks(ss_linear_solve(solver, b, x, &solve), SS_OK));
    CHECK(solve.status == SS_LINEAR_SOLVE_DIVERGED_KSP);
    CHECK(solve.reason == KSP_DIVERGED_PC_FAILED);
    CHECK(ss_linear_destroy(&solver) == SS_OK);
    CHECK(ss_pmg_destroy(&pmg) == SS_OK);
  }

  linear.pmg = NULL;
  linear.policy_chain.policies[0] = linear_policy();
  if (rank == 1) {
    linear.policy_chain.policies[0].vector_memory_cap_bytes = 1U;
  }
  CHECK(ss_linear_create(PETSC_COMM_WORLD, &solver) == SS_OK);
  CHECK(
      status_on_all_ranks(ss_linear_setup(solver, &linear, &setup), SS_ERROR_RESOURCE));
  linear.policy_chain.policies[0] = linear_policy();
  CHECK(status_on_all_ranks(ss_linear_setup(solver, &linear, &setup), SS_OK));
  CHECK(status_on_all_ranks(ss_linear_solve(solver, rank == 1 ? NULL : b, x, &solve),
                            SS_ERROR_INVALID_ARGUMENT));
  CHECK(ss_linear_destroy(&solver) == SS_OK);

  const SsDeflationPolicy policy =
      deflation_policy(SS_DEFLATION_PROJECTOR_GENERAL_LEFT_RIGHT);
  CHECK(ss_deflation_create(PETSC_COMM_WORLD, &policy, &basis) == SS_OK);
  CHECK(ss_deflation_begin(basis, 1U) == SS_OK);
  CHECK(status_on_all_ranks(
      ss_deflation_add_candidate(basis, b, b, rank == 1 ? NULL : "injected-invalid",
                                 &admission),
      SS_ERROR_INVALID_ARGUMENT));
  CHECK(status_on_all_ranks(
      ss_deflation_add_candidate(basis, b, b, "valid", &admission), SS_OK));
  CHECK(admission.admitted);
  CHECK(ss_deflation_refresh(basis, a, 1U) == SS_OK);
  CHECK(status_on_all_ranks(
      ss_deflation_apply_projected_operator(basis, a, rank == 1 ? NULL : b, x),
      SS_ERROR_INVALID_ARGUMENT));
  CHECK(status_on_all_ranks(ss_deflation_apply_q(basis, b, rank == 1 ? NULL : x),
                            SS_ERROR_INVALID_ARGUMENT));
  PetscReal invariant_error = 0.0;
  CHECK(status_on_all_ranks(
      ss_deflation_invariant_error(basis, a, rank == 1 ? NULL : &invariant_error),
      SS_ERROR_INVALID_ARGUMENT));
  CHECK(ss_deflation_rollback(basis) == SS_OK);
  CHECK(ss_deflation_destroy(&basis) == SS_OK);

  const SsNonlinearPolicy nls_policy = continuation_policy();
  ContinuationModel model = {.a = a};
  SsNonlinearProblem nonlinear = {
      .communicator = PETSC_COMM_WORLD,
      .reference_load = b,
      .elastic_reference = a,
      .operations.evaluate = continuation_evaluate,
      .context = &model,
      .linear_properties.finite_action = 1,
      .structure_generation = 1U,
      .first_value_generation = 1U,
      .transaction_generation = 1U,
      .accepted_generation = 1U,
      .linear_policies.count = 1U,
  };
  nonlinear.linear_policies.policies[0] = linear_policy();
  CHECK(VecSet(x, 0.0) == PETSC_SUCCESS);
  Vec result = NULL;
  CHECK(VecDuplicate(x, &result) == PETSC_SUCCESS);
  CHECK(status_on_all_ranks(ss_nonlinear_fixed_solve(&nonlinear, &nls_policy,
                                                     rank == 1 ? NAN : 1.0, x, result,
                                                     &nonlinear_record),
                            SS_ERROR_INVALID_ARGUMENT));
  /* The diagnostic receives the agreed phase and injects a local error.
   * Test the PETSc/transaction cleanup with work objects already allocated. */
  nonlinear.operations.linear_diagnostic = reject_diagnostic;
  CHECK(status_on_all_ranks(ss_nonlinear_fixed_solve(&nonlinear, &nls_policy, 1.0, x,
                                                     result, &nonlinear_record),
                            SS_ERROR_INJECTED));
  CHECK(VecDestroy(&result) == PETSC_SUCCESS);
  CHECK(VecDestroy(&x) == PETSC_SUCCESS);
  CHECK(VecDestroy(&b) == PETSC_SUCCESS);
  CHECK(MatDestroy(&a) == PETSC_SUCCESS);
  return 0;
}

static int sparse_pmg_refresh(void) {
  const PetscScalar diagonal_values[] = {1.0, 2.0, 5.0, 3.0, 4.0, 8.0, 6.0};
  const PetscMPIInt owners[] = {1, 3};
  const uint32_t order = 1U;
  const SsPmgLevelPolicy policy = {
      .options_prefix = "review_sparse_",
      .method = SS_PMG_LEVEL_FGMRES_JACOBI,
      .active_rank_count = 2U,
      .active_ranks = owners,
      .maximum_iterations = 6,
      .rtol = 1.0e-12,
      .atol = 1.0e-14,
  };
  const SsPmgSpecification spec = {
      .fine_order = order,
      .level_count = 1U,
      .orders = &order,
      .level_policies = &policy,
      .transfer_generation = 1U,
      .numeric_lag_interval = 1U,
      .memory_cap_bytes = 1024U * 1024U,
  };
  SsPmg *pmg = NULL;
  SsPmgSetupRecord setup;
  SsPmgApplyRecord apply;
  PetscMPIInt rank = -1;
  Mat matrix = NULL;
  Vec exact = NULL, rhs = NULL, correction = NULL;
  CHECK(diagonal(PETSC_COMM_WORLD, 7, diagonal_values, &matrix) == 0);
  CHECK(MatCreateVecs(matrix, &exact, &rhs) == PETSC_SUCCESS);
  CHECK(VecDuplicate(exact, &correction) == PETSC_SUCCESS);
  CHECK(set_values(exact, diagonal_values) == 0);
  CHECK(ss_pmg_create(PETSC_COMM_WORLD, &spec, &pmg) == SS_OK);
  CHECK(MPI_Comm_rank(PETSC_COMM_WORLD, &rank) == MPI_SUCCESS);
  CHECK(status_on_all_ranks(
      ss_pmg_set_level_operators(pmg, 1U, rank == 1 ? NULL : &matrix),
      SS_ERROR_INVALID_ARGUMENT));
  for (uint64_t generation = 1U; generation <= 3U; ++generation) {
    PetscReal error = 0.0;
    CHECK(ss_pmg_setup(pmg, matrix, 1U, generation, &setup) == SS_OK);
    CHECK(setup.numeric_refreshed && setup.hierarchy_build_count == 1U);
    CHECK(MatMult(matrix, exact, rhs) == PETSC_SUCCESS);
    CHECK(ss_pmg_apply(pmg, rhs, correction, &apply) == SS_OK);
    CHECK(VecAXPY(correction, -1.0, exact) == PETSC_SUCCESS);
    CHECK(VecNorm(correction, NORM_2, &error) == PETSC_SUCCESS && error < 5.0e-12);
    CHECK(MatScale(matrix, 2.0) == PETSC_SUCCESS);
  }
  CHECK(ss_pmg_destroy(&pmg) == SS_OK);
  CHECK(VecDestroy(&correction) == PETSC_SUCCESS);
  CHECK(VecDestroy(&rhs) == PETSC_SUCCESS);
  CHECK(VecDestroy(&exact) == PETSC_SUCCESS);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  return 0;
}

static int object_count(size_t *objects) {
  FILE *dump = tmpfile();
  char line[1024];
  CHECK(dump != NULL);
  CHECK(PetscObjectsDump(dump, PETSC_TRUE) == PETSC_SUCCESS);
  rewind(dump);
  *objects = 0U;
  while (fgets(line, (int)sizeof(line), dump) != NULL) {
    if (line[0] == '[') {
      ++*objects;
    }
  }
  CHECK(fclose(dump) == 0);
  return 0;
}

static int workspace_memory(void) {
  const PetscScalar values[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const SsNonlinearPolicy stopping = continuation_policy();
  char capture_option[16] = {0};
  PetscBool capture_set = PETSC_FALSE;
  static SsNonlinearSolveRecord record;
  PetscMPIInt ranks;
  Mat a = NULL;
  Vec parent = NULL, solution = NULL, load = NULL;
  SsPmg *pmg = NULL;
  SsDeflation *basis = NULL;
  PetscLogDouble reference_bytes[2] = {0.0, 0.0};
  size_t reference_objects[2] = {0U, 0U};
  int seen[2] = {0, 0};
  size_t captures = 0U;
  CHECK(PetscOptionsGetString(NULL, NULL, "-review_capture", capture_option,
                              sizeof(capture_option), &capture_set) == PETSC_SUCCESS);
  const int capture = capture_set && strcmp(capture_option, "full") == 0;
  CHECK(MPI_Comm_size(PETSC_COMM_WORLD, &ranks) == MPI_SUCCESS);
  CHECK(diagonal(PETSC_COMM_WORLD, 8, values, &a) == 0);
  CHECK(MatCreateVecs(a, &parent, &load) == PETSC_SUCCESS);
  CHECK(VecDuplicate(parent, &solution) == PETSC_SUCCESS);
  CHECK(VecSet(parent, 0.0) == PETSC_SUCCESS);
  CHECK(VecSet(load, 1.0) == PETSC_SUCCESS);
  CHECK(test_coarse_pmg(a, (size_t)ranks, SS_PMG_FAIL_NONE, &pmg) == 0);
  SsDeflationPolicy def_policy =
      deflation_policy(SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL);
  def_policy.dimension_cap = 2U;
  CHECK(ss_deflation_create(PETSC_COMM_WORLD, &def_policy, &basis) == SS_OK);

  if (capture) {
    SsLinearSolver *linear = NULL;
    SsLinearSetupRecord setup;
    SsLinearProblem budget = {
        .operator_handle = a,
        .pmg = pmg,
        .deflation = basis,
        .properties.finite_action = 1,
        .policy_chain.count = 1U,
        .structure_generation = 1U,
        .value_generation = 1U,
    };
    budget.policy_chain.policies[0] = linear_policy();
    budget.policy_chain.policies[0].pc = SS_LINEAR_PC_PMG;
    CHECK(ss_linear_create(PETSC_COMM_WORLD, &linear) == SS_OK);
    CHECK(ss_linear_setup(linear, &budget, &setup) == SS_OK);
    budget.policy_chain.policies[0].vector_memory_cap_bytes =
        setup.admitted_vector_bytes;
    budget.krylov_capture = SS_LINEAR_KRYLOV_CAPTURE_FULL_CORRECTION_SPACE;
    CHECK(ss_linear_setup(linear, &budget, &setup) == SS_ERROR_RESOURCE);
    CHECK(ss_linear_destroy(&linear) == SS_OK);
  }

  ContinuationModel model = {.a = a};
  SsNonlinearProblem problem = {
      .communicator = PETSC_COMM_WORLD,
      .reference_load = load,
      .elastic_reference = a,
      .operations.evaluate = continuation_evaluate,
      .operations.admissibility = continuation_admissible,
      .context = &model,
      .linear_properties.finite_action = 1,
      .pmg = pmg,
      .deflation = basis,
      .deflation_mode = SS_NONLINEAR_DEFLATION_OWNED_TRANSACTION,
      .deflation_promotion =
          capture ? SS_NONLINEAR_DEFLATION_PROMOTE_FULL_KRYLOV_AND_ACCEPTED_STATE
                  : SS_NONLINEAR_DEFLATION_PROMOTE_DIRECTIONS,
      .structure_generation = 1U,
      .first_value_generation = 2U,
      .transaction_generation = 1U,
      .accepted_generation = 1U,
      .linear_policies.count = 1U,
  };
  problem.linear_policies.policies[0] = linear_policy();
  problem.linear_policies.policies[0].pc = SS_LINEAR_PC_PMG;
  problem.linear_policies.policies[0].rtol = 1.0e-10;
  CHECK(ss_nonlinear_workspace_create(PETSC_COMM_WORLD, &problem.workspace) == SS_OK);
  for (int step = 1; step <= 96; ++step) {
    for (int rejected = step % 3 == 0; rejected >= 0; --rejected) {
      model.reject = rejected;
      CHECK(ss_nonlinear_fixed_solve(&problem, &stopping, 0.8 + 0.014 * (PetscReal)step,
                                     parent, solution, &record) == SS_OK);
      CHECK((record.outcome == SS_NONLINEAR_CONVERGED) == !rejected);
      if (rejected) {
        CHECK(record.transaction_rolled_back && record.parent_unchanged_on_failure);
      } else {
        CHECK(record.transaction_committed);
        CHECK(VecCopy(solution, parent) == PETSC_SUCCESS);
        ++problem.accepted_generation;
      }
      CHECK(record.history[0].linear_setup.setup_count == 1U);
      captures += record.history[0].linear_solve.krylov_vectors_captured;
      problem.first_value_generation += SS_NONLINEAR_HISTORY_CAPACITY;
      ++problem.transaction_generation;

      if (step >= 16) {
        PetscLogDouble bytes = 0.0, max_bytes = 0.0, sum_bytes = 0.0;
        size_t objects = 0U, dimension = 0U;
        CHECK(ss_deflation_persistent_count(basis, &dimension) == SS_OK &&
              dimension == 2U);
        CHECK(object_count(&objects) == 0);
        CHECK(PetscMallocGetCurrentUsage(&bytes) == PETSC_SUCCESS);
        CHECK(bytes > 0.0);
        if (!seen[rejected]) {
          reference_bytes[rejected] = bytes;
          reference_objects[rejected] = objects;
          seen[rejected] = 1;
        }
        CHECK(MPI_Allreduce(&bytes, &max_bytes, 1, MPI_DOUBLE, MPI_MAX,
                            PETSC_COMM_WORLD) == MPI_SUCCESS);
        CHECK(MPI_Allreduce(&bytes, &sum_bytes, 1, MPI_DOUBLE, MPI_SUM,
                            PETSC_COMM_WORLD) == MPI_SUCCESS);
        CHECK(PetscPrintf(PETSC_COMM_WORLD, "memory,%d,%d,%zu,%.0f,%.0f,%zu\n", step,
                          rejected, dimension, max_bytes, sum_bytes,
                          objects) == PETSC_SUCCESS);
        CHECK(status_on_all_ranks(bytes == reference_bytes[rejected] &&
                                          objects == reference_objects[rejected]
                                      ? SS_OK
                                      : SS_ERROR_INTEGRITY,
                                  SS_OK));
      }
    }
  }
  CHECK(!capture || captures > 0U);
  CHECK(ss_nonlinear_workspace_destroy(&problem.workspace) == SS_OK);
  CHECK(ss_pmg_destroy(&pmg) == SS_OK);
  CHECK(ss_deflation_destroy(&basis) == SS_OK);
  CHECK(VecDestroy(&solution) == PETSC_SUCCESS);
  CHECK(VecDestroy(&load) == PETSC_SUCCESS);
  CHECK(VecDestroy(&parent) == PETSC_SUCCESS);
  CHECK(MatDestroy(&a) == PETSC_SUCCESS);
  return 0;
}

int main(int argc, char **argv) {
  int result = 1;
  CHECK(argc >= 2);
  CHECK(PetscInitialize(&argc, &argv, NULL, NULL) == PETSC_SUCCESS);
  if (strcmp(argv[1], "signed") == 0) {
    result = signed_projection();
  } else if (strcmp(argv[1], "minres") == 0) {
    result = minres_pc();
  } else if (strcmp(argv[1], "projected") == 0) {
    result = linear_acceptance(SS_LINEAR_STOP_PROJECTED_RHS_PETSC);
  } else if (strcmp(argv[1], "preconditioned") == 0) {
    result = linear_acceptance(SS_LINEAR_STOP_PRECONDITIONED_RESIDUAL_ORIGINAL_RHS);
  } else if (strcmp(argv[1], "natural") == 0) {
    result = linear_acceptance(SS_LINEAR_STOP_NATURAL_RESIDUAL_INITIAL);
  } else if (strcmp(argv[1], "memory") == 0) {
    result = memory_budget();
  } else if (strcmp(argv[1], "fixed") == 0) {
    result = nonlinear_stopping(0, RESIDUAL_STRICT);
  } else if (strcmp(argv[1], "augmented") == 0) {
    result = nonlinear_stopping(1, RESIDUAL_STRICT);
  } else if (strcmp(argv[1], "stationary-fixed") == 0) {
    result = nonlinear_stopping(0, STATIONARY);
  } else if (strcmp(argv[1], "stationary-augmented") == 0) {
    result = nonlinear_stopping(1, STATIONARY);
  } else if (strcmp(argv[1], "quantized-control") == 0) {
    result = nonlinear_stopping(1, QUANTIZED_CONTROL);
  } else if (strcmp(argv[1], "load-only") == 0) {
    result = nonlinear_stopping(1, LOAD_ONLY_UPDATE);
  } else if (strcmp(argv[1], "collective") == 0) {
    result = collective_failures();
  } else if (strcmp(argv[1], "sparse-pmg") == 0) {
    result = sparse_pmg_refresh();
  } else if (strcmp(argv[1], "workspace") == 0) {
    result = workspace_memory();
  } else if (strcmp(argv[1], "sensitivity") == 0) {
    result = nonlinear_stopping(1, SENSITIVITY_TIGHT);
  }
  CHECK(PetscFinalize() == PETSC_SUCCESS);
  return result;
}
