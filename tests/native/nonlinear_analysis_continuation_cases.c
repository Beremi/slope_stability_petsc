/* Continuation trajectory, predictor and accepted-state tests.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "nonlinear_analysis_internal.h"

static SsStatus analysis_fixed_solve(void *opaque, PetscReal factor, Vec incoming,
                                     Vec solution, SsNonlinearSolveRecord *record) {
  AnalysisModel *model = (AnalysisModel *)opaque;
  SsStatus status;
  ++model->fixed_calls;
  status = ss_nonlinear_fixed_solve(&model->nonlinear, &model->nonlinear_policy, factor,
                                    incoming, solution, record);
  if (status == SS_OK && model->fail_after_fixed_solve_call > 0 &&
      model->fixed_calls == model->fail_after_fixed_solve_call &&
      (record->outcome == SS_NONLINEAR_CONVERGED ||
       record->outcome == SS_NONLINEAR_ACCEPTED_RELAXED)) {
    record->outcome = SS_NONLINEAR_FAILED_LINE_SEARCH;
  }
  return status;
}

static SsStatus analysis_total_potential(void *opaque, PetscReal factor, Vec state,
                                         PetscReal *total_potential) {
  AnalysisModel *model = (AnalysisModel *)opaque;
  Vec residual = NULL;
  SsStatus status;
  if (VecDuplicate(state, &residual) != PETSC_SUCCESS) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  status =
      scalar_evaluate(&model->model, state, factor, residual, NULL, total_potential);
  if (VecDestroy(&residual) != PETSC_SUCCESS && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  return status;
}

static SsStatus analysis_displacement_maximum(void *opaque, Vec state,
                                              PetscReal *maximum) {
  AnalysisModel *model = (AnalysisModel *)opaque;
  PetscReal norm = NAN;
  if (model == NULL || state == NULL || maximum == NULL ||
      VecNorm(state, NORM_2, &norm) != PETSC_SUCCESS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  ++model->displacement_maximum_calls;
  *maximum = 2.0 * norm;
  return isfinite((double)*maximum) ? SS_OK : SS_ERROR_INTEGRITY;
}

static SsStatus analysis_work_solve(void *opaque, SsAnalysisMethod method,
                                    PetscReal target_work, PetscReal incoming_factor,
                                    Vec incoming, Vec solution,
                                    PetscReal *accepted_factor,
                                    SsNonlinearSolveRecord *record) {
  AnalysisModel *model = (AnalysisModel *)opaque;
  PetscInt global_size = 0;
  PetscReal value;
  ++model->work_calls;
  if (method == SS_ANALYSIS_SSR_INDIRECT) {
    SsNonlinearPolicy controlled_policy = model->nonlinear_policy;
    controlled_policy.line_method = SS_NONLINEAR_LINE_RESIDUAL_MONOTONE;
    return ss_nonlinear_work_controlled_solve(
        &model->nonlinear, &controlled_policy, SS_NONLINEAR_CONTROL_SSR_INDIRECT,
        target_work, incoming_factor, incoming, solution, accepted_factor, record);
  }
  memset(record, 0, sizeof(*record));
  record->outcome = SS_NONLINEAR_CONVERGED;
  record->iterations = 5;
  if (method == SS_ANALYSIS_LIMIT_LOAD) {
    PetscScalar incoming_sum = 0.0;
    PetscReal expected = model->work_calls == 1 ? 0.0 : target_work;
    if (VecSum(incoming, &incoming_sum) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    model->maximum_limit_load_predictor_error =
        fmax(model->maximum_limit_load_predictor_error,
             fabs(PetscRealPart(incoming_sum) - expected));
    if (model->fail_one_limit_load_attempt != 0 && model->work_calls == 2) {
      record->outcome = SS_NONLINEAR_FAILED_LINE_SEARCH;
      return SS_OK;
    }
    value = target_work;
    *accepted_factor = sqrt(target_work);
  } else {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (VecGetSize(solution, &global_size) != PETSC_SUCCESS || global_size <= 0 ||
      VecSet(solution, value / (PetscReal)global_size) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  record->final_convergence = 0.0;
  return SS_OK;
}

static SsStatus analysis_begin_attempt(void *opaque, size_t attempt_ordinal,
                                       size_t parent_accepted_ordinal) {
  AnalysisModel *model = (AnalysisModel *)opaque;
  (void)parent_accepted_ordinal;
  if (model->transaction_open != 0) {
    return SS_ERROR_INVALID_STATE;
  }
  if (model->outer_deflation != NULL) {
    SsDeflationDiagnostics diagnostics;
    SsStatus status = ss_deflation_diagnostics(model->outer_deflation, &diagnostics);
    if (status != SS_OK) {
      return status;
    }
    model->deflation_persistent_at_begin = diagnostics.persistent_count;
    status = ss_deflation_begin(model->outer_deflation,
                                UINT64_C(1000) + (uint64_t)attempt_ordinal);
    if (status != SS_OK) {
      return status;
    }
  }
  model->transaction_open = 1;
  ++model->transaction_begin_count;
  return SS_OK;
}

static SsStatus analysis_commit_attempt(void *opaque, size_t attempt_ordinal,
                                        size_t parent_accepted_ordinal) {
  AnalysisModel *model = (AnalysisModel *)opaque;
  (void)parent_accepted_ordinal;
  if (model->transaction_open == 0) {
    return SS_ERROR_INVALID_STATE;
  }
  if (model->outer_deflation != NULL) {
    SsDeflationDiagnostics diagnostics;
    SsStatus status = ss_deflation_commit(model->outer_deflation,
                                          UINT64_C(2000) + (uint64_t)attempt_ordinal);
    if (status != SS_OK) {
      return status;
    }
    status = ss_deflation_diagnostics(model->outer_deflation, &diagnostics);
    if (status != SS_OK) {
      return status;
    }
    model->deflation_commit_closed =
        diagnostics.provisional_count == 0U &&
        diagnostics.persistent_count >= model->deflation_persistent_at_begin;
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
  if (model->outer_deflation != NULL) {
    SsDeflationDiagnostics before;
    SsDeflationDiagnostics after;
    SsStatus status = ss_deflation_diagnostics(model->outer_deflation, &before);
    if (status != SS_OK) {
      return status;
    }
    model->saw_rollback_provisional |= before.provisional_count > 0U;
    status = ss_deflation_rollback(model->outer_deflation);
    if (status != SS_OK) {
      return status;
    }
    status = ss_deflation_diagnostics(model->outer_deflation, &after);
    if (status != SS_OK) {
      return status;
    }
    model->deflation_rollback_preserved =
        after.persistent_count == model->deflation_persistent_at_begin &&
        after.provisional_count == 0U && after.active_count == after.persistent_count;
  }
  model->transaction_open = 0;
  ++model->transaction_rollback_count;
  return SS_OK;
}

static void fill_analysis_problem(AnalysisModel *model, Vec load,
                                  SsAnalysisProblem *problem) {
  memset(problem, 0, sizeof(*problem));
  problem->communicator = PETSC_COMM_SELF;
  problem->reference_load = load;
  problem->operations.solve_fixed = analysis_fixed_solve;
  problem->operations.solve_work_controlled = analysis_work_solve;
  problem->operations.total_potential = analysis_total_potential;
  problem->operations.displacement_maximum = analysis_displacement_maximum;
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
  policy->direct_control_maximum = 0.0;
  policy->scaled_slope_tolerance = 0.0;
  policy->factor_increment_stop = 0.0;
  policy->accepted_step_limit = 8;
  policy->consecutive_retry_limit = 5;
}

static int check_analysis_history(const SsAnalysisRecord *record) {
  uint64_t newton_iterations = 0U;
  uint64_t linear_setups = 0U;
  uint64_t linear_solves = 0U;
  uint64_t linear_iterations = 0U;
  uint64_t commits = 0U;
  uint64_t rollbacks = 0U;
  size_t index;
  CHECK(record->method_identity[0] != '\0');
  CHECK(record->predictor_identity[0] != '\0');
  CHECK(record->controller_identity[0] != '\0');
  for (index = 0U; index < record->attempt_count; ++index) {
    const SsAnalysisAttemptRecord *attempt = &record->attempts[index];
    CHECK(attempt->attempt_ordinal == index);
    CHECK(attempt->predictor >= SS_ANALYSIS_PREDICTOR_CURRENT_STATE &&
          attempt->predictor <=
              SS_ANALYSIS_PREDICTOR_DISPLACEMENT_SECANT_ANCHORED_FACTOR);
    CHECK(attempt->retry_ordinal >= 0);
    CHECK((attempt->committed != 0) + (attempt->rolled_back != 0) == 1);
    CHECK(attempt->successful == 0 || attempt->committed != 0);
    newton_iterations += (uint64_t)attempt->primary_solve.iterations +
                         (uint64_t)attempt->secondary_solve.iterations;
    linear_setups += attempt->primary_solve.linear_setup_count +
                     attempt->secondary_solve.linear_setup_count;
    linear_solves += attempt->primary_solve.linear_solve_count +
                     attempt->secondary_solve.linear_solve_count;
    linear_iterations += attempt->primary_solve.first_linear_iteration_count +
                         attempt->primary_solve.second_linear_iteration_count +
                         attempt->secondary_solve.first_linear_iteration_count +
                         attempt->secondary_solve.second_linear_iteration_count;
    if (attempt->primary_solve.history_count != 0U) {
      CHECK(isfinite((double)attempt->primary_solve.minimum_convergence));
      if (isfinite((double)attempt->primary_solve.initial_convergence)) {
        CHECK(attempt->primary_solve.minimum_convergence <=
              attempt->primary_solve.initial_convergence);
      }
      if (isfinite((double)attempt->primary_solve.final_convergence)) {
        CHECK(attempt->primary_solve.minimum_convergence <=
              attempt->primary_solve.final_convergence);
      }
    }
    commits += attempt->committed != 0 ? 1U : 0U;
    rollbacks += attempt->rolled_back != 0 ? 1U : 0U;
  }
  CHECK(newton_iterations == record->newton_iteration_count);
  CHECK(linear_setups == record->linear_setup_count);
  CHECK(linear_solves == record->linear_solve_count);
  CHECK(linear_iterations == record->linear_iteration_count);
  CHECK(commits == record->commit_count);
  CHECK(rollbacks == record->rollback_count);
  return 0;
}

typedef enum { SEED_SHIFT_DIRECT = 1, SEED_SHIFT_INDIRECT = 2 } SeedShiftMode;

typedef struct {
  SeedShiftMode mode;
  int decreasing_direct_control;
  PetscReal decreasing_control_factor;
  int fail_fixed;
  PetscInt fixed_failures_remaining;
  PetscReal fixed_failure_factor;
  int fixed_work_tracks_factor;
  PetscReal fixed_work_scale;
  int fail_first_work;
  PetscInt fail_work_call;
  PetscInt work_iterations;
  PetscInt work_iteration_sequence[8];
  size_t work_iteration_sequence_count;
  PetscInt failure_iterations;
  PetscReal work_factor_increment;
  PetscReal work_factor_increment_sequence[8];
  size_t work_factor_increment_sequence_count;
  PetscReal work_solution_curvature;
  PetscInt work_calls;
  PetscReal first_work_target;
  size_t accepted_state_count;
  PetscReal accepted_state_sums[8];
} SeedShiftModel;

static SsStatus seed_shift_fixed(void *opaque, PetscReal factor, Vec incoming,
                                 Vec solution, SsNonlinearSolveRecord *record) {
  SeedShiftModel *model = (SeedShiftModel *)opaque;
  PetscReal value = factor;
  (void)incoming;
  if (model->mode == SEED_SHIFT_INDIRECT && model->fixed_work_scale > 0.0) {
    value = model->fixed_work_scale * factor;
  } else if (model->mode == SEED_SHIFT_INDIRECT &&
             model->fixed_work_tracks_factor == 0) {
    value = factor <= 1.2 + 1.0e-12 ? 0.1 : 0.1 + factor - 1.2;
  }
  memset(record, 0, sizeof(*record));
  if (model->fail_fixed != 0 || model->fixed_failures_remaining > 0 ||
      (model->fixed_failure_factor > 0.0 && factor >= model->fixed_failure_factor)) {
    if (model->fixed_failures_remaining > 0) {
      --model->fixed_failures_remaining;
    }
    record->outcome = SS_NONLINEAR_FAILED_LINE_SEARCH;
    return SS_OK;
  }
  record->outcome = SS_NONLINEAR_CONVERGED;
  record->iterations = 1;
  record->final_convergence = 0.0;
  return VecSet(solution, value) == PETSC_SUCCESS ? SS_OK : SS_ERROR_INTERNAL;
}

static SsStatus seed_shift_potential(void *opaque, PetscReal factor, Vec state,
                                     PetscReal *total_potential) {
  const SeedShiftModel *model = (const SeedShiftModel *)opaque;
  PetscReal base_excess;
  PetscReal base_potential;
  PetscReal excess;
  (void)state;
  if (model->mode != SEED_SHIFT_DIRECT) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (model->decreasing_direct_control != 0) {
    *total_potential = 0.5 * factor * factor;
    return SS_OK;
  }
  if (model->decreasing_control_factor > 0.0 &&
      factor > model->decreasing_control_factor) {
    base_excess = fmax(0.0, model->decreasing_control_factor - 1.2);
    if (base_excess <= 0.3) {
      base_potential =
          -0.1 * model->decreasing_control_factor - 0.5 * base_excess * base_excess;
    } else {
      const PetscReal base_tail = base_excess - 0.3;
      base_potential = -0.195 - 0.4 * base_tail + base_tail * base_tail;
    }
    excess = factor - model->decreasing_control_factor;
    *total_potential = base_potential + 0.5 * excess * excess;
    return SS_OK;
  }
  excess = fmax(0.0, factor - 1.2);
  if (excess <= 0.3) {
    *total_potential = -0.1 * factor - 0.5 * excess * excess;
  } else {
    const PetscReal tail = excess - 0.3;
    *total_potential = -0.195 - 0.4 * tail + tail * tail;
  }
  return SS_OK;
}

static SsStatus seed_shift_work(void *opaque, SsAnalysisMethod method,
                                PetscReal target_work, PetscReal incoming_factor,
                                Vec incoming, Vec solution, PetscReal *accepted_factor,
                                SsNonlinearSolveRecord *record) {
  SeedShiftModel *model = (SeedShiftModel *)opaque;
  (void)incoming;
  if (model->mode != SEED_SHIFT_INDIRECT || method != SS_ANALYSIS_SSR_INDIRECT) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(record, 0, sizeof(*record));
  ++model->work_calls;
  if (model->work_calls == 1) {
    model->first_work_target = target_work;
  }
  if ((model->fail_first_work != 0 && model->work_calls == 1) ||
      (model->fail_work_call > 0 && model->work_calls == model->fail_work_call)) {
    record->outcome = model->failure_iterations > 0
                          ? SS_NONLINEAR_FAILED_ITERATION_LIMIT
                          : SS_NONLINEAR_FAILED_LINE_SEARCH;
    record->iterations = model->failure_iterations;
    return SS_OK;
  }
  record->outcome = SS_NONLINEAR_CONVERGED;
  record->iterations = (size_t)model->work_calls <= model->work_iteration_sequence_count
                           ? model->work_iteration_sequence[model->work_calls - 1]
                       : model->work_iterations > 0 ? model->work_iterations
                                                    : 2;
  record->final_convergence = 0.0;
  *accepted_factor =
      incoming_factor +
      ((size_t)model->work_calls <= model->work_factor_increment_sequence_count
           ? model->work_factor_increment_sequence[model->work_calls - 1]
       : model->work_factor_increment > 0.0 ? model->work_factor_increment
                                            : 0.2);
  return VecSet(solution, target_work + model->work_solution_curvature * target_work *
                                            target_work) == PETSC_SUCCESS
             ? SS_OK
             : SS_ERROR_INTERNAL;
}

static int run_adaptive_work_case(SeedShiftModel *model, SsAnalysisPolicy *policy,
                                  SsAnalysisRecord *record) {
  SsAnalysisProblem problem;
  Vec load = NULL;
  Vec initial = NULL;
  Vec accepted = NULL;
  int result = 1;
  if (create_vector(1, 1.0, &load) != 0 || create_vector(1, 0.0, &initial) != 0 ||
      VecDuplicate(initial, &accepted) != PETSC_SUCCESS) {
    goto cleanup;
  }
  memset(&problem, 0, sizeof(problem));
  problem.communicator = PETSC_COMM_SELF;
  problem.reference_load = load;
  problem.operations.solve_fixed = seed_shift_fixed;
  problem.operations.solve_work_controlled = seed_shift_work;
  problem.context = model;
  if (ss_analysis_run(&problem, policy, initial, accepted, record) != SS_OK ||
      check_analysis_history(record) != 0) {
    goto cleanup;
  }
  result = 0;

cleanup:
  (void)VecDestroy(&accepted);
  (void)VecDestroy(&initial);
  (void)VecDestroy(&load);
  return result;
}

int test_indirect_adaptive_work_controller(void) {
  SeedShiftModel model;
  SsAnalysisPolicy policy;
  SsAnalysisPolicy invalid_policy;
  SsAnalysisRecord record;
  const PetscReal initial_work_step = 0.2;

  memset(&model, 0, sizeof(model));
  model.mode = SEED_SHIFT_INDIRECT;
  model.fixed_work_tracks_factor = 1;
  model.work_factor_increment = 0.1;
  model.work_iterations = 10;
  fill_analysis_policy(SS_ANALYSIS_SSR_INDIRECT, &policy);
  policy.accepted_step_limit = 4;
  policy.work_maximum = 2.0;
  policy.work_step_controller = SS_ANALYSIS_WORK_STEP_PREDICTOR_ERROR;
  policy.work_growth_iteration_limit = 50;
  policy.work_predictor_error_target = 0.25;
  policy.work_step_scale_minimum = 0.5;
  policy.work_step_scale_maximum = 2.0;
  invalid_policy = policy;
  invalid_policy.work_growth_iteration_limit = 0;
  CHECK(ss_analysis_policy_validate(&invalid_policy) == SS_ERROR_INVALID_ARGUMENT);
  invalid_policy = policy;
  invalid_policy.work_predictor_error_target = 0.0;
  CHECK(ss_analysis_policy_validate(&invalid_policy) == SS_ERROR_INVALID_ARGUMENT);
  invalid_policy = policy;
  invalid_policy.work_step_scale_minimum = 1.1;
  CHECK(ss_analysis_policy_validate(&invalid_policy) == SS_ERROR_INVALID_ARGUMENT);
  invalid_policy = policy;
  invalid_policy.work_step_scale_maximum = 0.9;
  CHECK(ss_analysis_policy_validate(&invalid_policy) == SS_ERROR_INVALID_ARGUMENT);
  invalid_policy = policy;
  invalid_policy.method = SS_ANALYSIS_SSR_DIRECT;
  CHECK(ss_analysis_policy_validate(&invalid_policy) == SS_ERROR_INVALID_ARGUMENT);
  invalid_policy = policy;
  invalid_policy.indirect_predictor_mode = (SsAnalysisIndirectPredictorMode)99;
  CHECK(ss_analysis_policy_validate(&invalid_policy) == SS_ERROR_INVALID_ARGUMENT);
  CHECK(run_adaptive_work_case(&model, &policy, &record) == 0);
  CHECK(strcmp(record.controller_identity, "ssr_indirect/predictor_error_state_only") ==
        0);
  CHECK(record.point_count == 4U && record.attempt_count == 4U);
  CHECK(fabs(record.attempts[2].predictor_error) < 1.0e-14);
  CHECK(fabs(record.attempts[2].work_step_scale - 2.0) < 1.0e-14);
  CHECK(fabs(record.attempts[2].next_work_step - 2.0 * initial_work_step) < 1.0e-14);
  CHECK(fabs(record.attempts[3].requested_step - 2.0 * initial_work_step) < 1.0e-14);

  memset(&model, 0, sizeof(model));
  model.mode = SEED_SHIFT_INDIRECT;
  model.fixed_work_tracks_factor = 1;
  model.work_factor_increment = 0.1;
  model.work_iterations = 10;
  policy.indirect_predictor_mode = SS_ANALYSIS_INDIRECT_PREDICTOR_SCALED_CURRENT;
  CHECK(run_adaptive_work_case(&model, &policy, &record) == 0);
  CHECK(strcmp(record.predictor_identity,
               "ssr_indirect/scaled_current+anchored_factor") == 0);
  CHECK(record.attempts[2].predictor == SS_ANALYSIS_PREDICTOR_CURRENT_STATE);
  policy.indirect_predictor_mode = SS_ANALYSIS_INDIRECT_PREDICTOR_DISPLACEMENT_SECANT;

  memset(&model, 0, sizeof(model));
  model.mode = SEED_SHIFT_INDIRECT;
  model.fixed_work_tracks_factor = 1;
  model.work_factor_increment = 0.1;
  model.work_iterations = 50;
  CHECK(run_adaptive_work_case(&model, &policy, &record) == 0);
  CHECK(fabs(record.attempts[2].work_step_scale - 2.0) < 1.0e-14);
  CHECK(fabs(record.attempts[3].requested_step - 2.0 * initial_work_step) < 1.0e-14);

  memset(&model, 0, sizeof(model));
  model.mode = SEED_SHIFT_INDIRECT;
  model.fixed_work_tracks_factor = 1;
  model.work_factor_increment = 0.1;
  model.work_iterations = 40;
  CHECK(run_adaptive_work_case(&model, &policy, &record) == 0);
  CHECK(fabs(record.attempts[2].work_step_scale - 2.0) < 1.0e-14);
  CHECK(fabs(record.attempts[3].requested_step - 2.0 * initial_work_step) < 1.0e-14);

  memset(&model, 0, sizeof(model));
  model.mode = SEED_SHIFT_INDIRECT;
  model.fixed_work_tracks_factor = 1;
  model.work_factor_increment = 0.1;
  model.work_iterations = 10;
  model.work_solution_curvature = 1.0;
  CHECK(run_adaptive_work_case(&model, &policy, &record) == 0);
  CHECK(fabs(record.attempts[2].predictor_error - 49.0 / 54.0) < 1.0e-14);
  CHECK(fabs(record.attempts[2].work_step_scale - sqrt(27.0 / 98.0)) < 1.0e-14);
  CHECK(fabs(record.attempts[3].requested_step -
             sqrt(27.0 / 98.0) * initial_work_step) < 1.0e-14);

  memset(&model, 0, sizeof(model));
  model.mode = SEED_SHIFT_INDIRECT;
  model.fixed_work_tracks_factor = 1;
  model.work_factor_increment = 0.1;
  model.work_iterations = 10;
  model.fail_work_call = 2;
  model.failure_iterations = 60;
  policy.accepted_step_limit = 4;
  CHECK(run_adaptive_work_case(&model, &policy, &record) == 0);
  CHECK(record.point_count == 4U && record.attempt_count == 5U);
  CHECK(record.attempts[3].rolled_back != 0);
  CHECK(fabs(record.attempts[3].work_step_scale - sqrt(0.5)) < 1.0e-14);
  CHECK(fabs(record.attempts[3].next_work_step - sqrt(0.08)) < 1.0e-14);
  CHECK(fabs(record.attempts[4].requested_step - sqrt(0.08)) < 1.0e-14);

  memset(&model, 0, sizeof(model));
  model.mode = SEED_SHIFT_INDIRECT;
  model.fixed_work_tracks_factor = 1;
  model.work_factor_increment = 0.1;
  model.work_iterations = 10;
  model.fail_work_call = 2;
  model.failure_iterations = 4;
  CHECK(run_adaptive_work_case(&model, &policy, &record) == 0);
  CHECK(record.attempts[3].rolled_back != 0);
  CHECK(fabs(record.attempts[3].work_step_scale - sqrt(0.5)) < 1.0e-14);
  CHECK(fabs(record.attempts[3].next_work_step - sqrt(0.08)) < 1.0e-14);
  CHECK(fabs(record.attempts[4].requested_step - sqrt(0.08)) < 1.0e-14);

  memset(&model, 0, sizeof(model));
  model.mode = SEED_SHIFT_INDIRECT;
  model.fixed_work_tracks_factor = 1;
  model.work_factor_increment = 0.1;
  model.work_iterations = 40;
  model.fail_work_call = 2;
  model.failure_iterations = 4;
  CHECK(run_adaptive_work_case(&model, &policy, &record) == 0);
  CHECK(record.attempts[3].rolled_back != 0);
  CHECK(fabs(record.attempts[3].work_step_scale - sqrt(0.5)) < 1.0e-14);
  CHECK(fabs(record.attempts[3].next_work_step - sqrt(0.08)) < 1.0e-14);
  CHECK(fabs(record.attempts[4].requested_step - sqrt(0.08)) < 1.0e-14);
  return 0;
}

int test_indirect_work_efficiency_controller(void) {
  SeedShiftModel model;
  SsAnalysisPolicy policy;
  SsAnalysisPolicy invalid_policy;
  SsAnalysisRecord record;
  const PetscReal first_step = 0.2;
  const PetscReal first_scale = 50.0 / 30.0;
  const PetscReal pair_elasticity = log(31.0 / 30.0) / log(first_scale);
  const PetscReal expected_budget = 50.0 + (1.0 - pair_elasticity) * 10.0;
  const PetscReal expected_scale = expected_budget / 31.0;

  memset(&model, 0, sizeof(model));
  model.mode = SEED_SHIFT_INDIRECT;
  model.fixed_work_tracks_factor = 1;
  model.work_factor_increment = 0.1;
  model.work_iteration_sequence[0] = 30;
  model.work_iteration_sequence[1] = 31;
  model.work_iteration_sequence[2] = 32;
  model.work_iteration_sequence_count = 3U;
  model.work_factor_increment_sequence[0] = 0.1;
  model.work_factor_increment_sequence[1] = 0.05;
  model.work_factor_increment_sequence[2] = 0.02;
  model.work_factor_increment_sequence_count = 3U;
  fill_analysis_policy(SS_ANALYSIS_SSR_INDIRECT, &policy);
  policy.accepted_step_limit = 5;
  policy.work_maximum = 3.0;
  policy.work_step_controller = SS_ANALYSIS_WORK_STEP_WORK_EFFICIENCY;
  policy.work_growth_iteration_limit = 50;
  policy.work_efficiency_iteration_maximum = 60;
  policy.work_predictor_error_target = 1.0;
  policy.work_step_scale_minimum = 0.5;
  policy.work_step_scale_maximum = 2.0;

  invalid_policy = policy;
  invalid_policy.work_efficiency_iteration_maximum = 49;
  CHECK(ss_analysis_policy_validate(&invalid_policy) == SS_ERROR_INVALID_ARGUMENT);
  CHECK(run_adaptive_work_case(&model, &policy, &record) == 0);
  CHECK(strcmp(record.controller_identity, "ssr_indirect/work_efficiency_adaptive") ==
        0);
  CHECK(fabs(record.attempts[2].work_step_scale - first_scale) < 1.0e-13);
  CHECK(fabs(record.attempts[3].work_efficiency_elasticity - pair_elasticity) <
        1.0e-13);
  CHECK(fabs(record.attempts[3].work_efficiency_iteration_budget - expected_budget) <
        1.0e-13);
  CHECK(fabs(record.attempts[3].work_step_scale - expected_scale) < 1.0e-13);
  CHECK(record.attempts[3].work_step_scale > 50.0 / 31.0);
  CHECK(fabs(record.attempts[4].requested_step -
             first_step * first_scale * expected_scale) < 1.0e-13);

  memset(&model, 0, sizeof(model));
  model.mode = SEED_SHIFT_INDIRECT;
  model.fixed_work_tracks_factor = 1;
  model.work_factor_increment = 0.1;
  model.work_iteration_sequence[0] = 30;
  model.work_iteration_sequence[1] = 55;
  model.work_iteration_sequence[2] = 40;
  model.work_iteration_sequence_count = 3U;
  model.work_factor_increment_sequence[0] = 0.1;
  model.work_factor_increment_sequence[1] = 0.05;
  model.work_factor_increment_sequence[2] = 0.02;
  model.work_factor_increment_sequence_count = 3U;
  CHECK(run_adaptive_work_case(&model, &policy, &record) == 0);
  CHECK(record.attempts[3].work_efficiency_elasticity > 1.0);
  CHECK(fabs(record.attempts[3].work_efficiency_iteration_budget - 50.0) < 1.0e-13);
  CHECK(fabs(record.attempts[3].work_step_scale - 50.0 / 55.0) < 1.0e-13);
  return 0;
}

static SsStatus seed_shift_capture_accepted_state(void *opaque, size_t accepted_ordinal,
                                                  Vec state) {
  SeedShiftModel *model = (SeedShiftModel *)opaque;
  PetscScalar sum = 0.0;
  if (model == NULL || state == NULL ||
      accepted_ordinal != model->accepted_state_count ||
      accepted_ordinal >=
          sizeof(model->accepted_state_sums) / sizeof(model->accepted_state_sums[0]) ||
      VecSum(state, &sum) != PETSC_SUCCESS) {
    return SS_ERROR_INTEGRITY;
  }
  model->accepted_state_sums[accepted_ordinal] = PetscRealPart(sum);
  ++model->accepted_state_count;
  return SS_OK;
}

static int test_direct_seed_shift(void) {
  SeedShiftModel model = {.mode = SEED_SHIFT_DIRECT};
  SsAnalysisProblem problem;
  SsAnalysisPolicy policy;
  SsAnalysisRecord record;
  Vec load = NULL;
  Vec initial = NULL;
  Vec accepted = NULL;
  CHECK(create_vector(1, 1.0, &load) == 0);
  CHECK(create_vector(1, 0.0, &initial) == 0);
  CHECK(VecDuplicate(initial, &accepted) == PETSC_SUCCESS);
  memset(&problem, 0, sizeof(problem));
  problem.communicator = PETSC_COMM_SELF;
  problem.reference_load = load;
  problem.operations.solve_fixed = seed_shift_fixed;
  problem.operations.total_potential = seed_shift_potential;
  problem.operations.capture_accepted_state = seed_shift_capture_accepted_state;
  problem.context = &model;
  fill_analysis_policy(SS_ANALYSIS_SSR_DIRECT, &policy);
  policy.accepted_step_limit = 4;
  CHECK(ss_analysis_run(&problem, &policy, initial, accepted, &record) == SS_OK);
  CHECK(check_analysis_history(&record) == 0);
  CHECK(strcmp(record.method_identity, "ssr_direct/reference_fd") == 0);
  CHECK(record.point_count == 4U);
  CHECK(model.accepted_state_count == record.point_count);
  CHECK(fabs(model.accepted_state_sums[0] - 1.0) < 1.0e-14);
  CHECK(fabs(model.accepted_state_sums[3] - 1.5) < 1.0e-14);
  CHECK(fabs(record.points[0].factor - 1.0) < 1.0e-14);
  CHECK(fabs(record.points[1].factor - 1.2) < 1.0e-14);
  CHECK(fabs(record.points[2].factor - 1.4) < 1.0e-14);
  CHECK(fabs(record.points[3].factor - 1.5) < 1.0e-14);
  CHECK(!isfinite((double)record.points[1].signed_scaled_slope));
  CHECK(isfinite((double)record.points[2].signed_scaled_slope));
  CHECK(record.rollback_count == 1U && record.attempt_count == 5U);
  CHECK(record.attempts[3].rolled_back != 0);
  CHECK(record.attempts[4].retry_ordinal == 1);
  memset(&model, 0, sizeof(model));
  model.mode = SEED_SHIFT_DIRECT;
  model.fixed_failure_factor = 1.5;
  policy.accepted_step_limit = 16;
  CHECK(VecSet(initial, 0.0) == PETSC_SUCCESS);
  CHECK(VecSet(accepted, 0.0) == PETSC_SUCCESS);
  CHECK(ss_analysis_run(&problem, &policy, initial, accepted, &record) == SS_OK);
  CHECK(check_analysis_history(&record) == 0);
  CHECK(record.point_count == 6U);
  CHECK(model.accepted_state_count == record.point_count);
  CHECK(record.stop_reason == SS_ANALYSIS_STOP_MINIMUM_STEP_REACHED);
  CHECK(record.terminal_success == 1);
  CHECK(record.final_factor >= 1.4 && record.final_factor < 1.5);
  memset(&model, 0, sizeof(model));
  model.mode = SEED_SHIFT_DIRECT;
  model.decreasing_control_factor = 1.5;
  CHECK(VecSet(initial, 0.0) == PETSC_SUCCESS);
  CHECK(VecSet(accepted, 0.0) == PETSC_SUCCESS);
  CHECK(ss_analysis_run(&problem, &policy, initial, accepted, &record) == SS_OK);
  CHECK(check_analysis_history(&record) == 0);
  CHECK(record.point_count == 4U);
  CHECK(model.accepted_state_count == record.point_count);
  CHECK(record.stop_reason == SS_ANALYSIS_STOP_MINIMUM_STEP_REACHED);
  CHECK(record.terminal_success == 1);
  CHECK(fabs(record.final_factor - 1.5) < 1.0e-14);
  memset(&model, 0, sizeof(model));
  model.mode = SEED_SHIFT_DIRECT;
  policy.accepted_step_limit = 4;
  policy.factor_maximum = 1.2;
  CHECK(VecSet(initial, 0.0) == PETSC_SUCCESS);
  CHECK(VecSet(accepted, 0.0) == PETSC_SUCCESS);
  CHECK(ss_analysis_run(&problem, &policy, initial, accepted, &record) == SS_OK);
  CHECK(check_analysis_history(&record) == 0);
  CHECK(record.point_count == 2U);
  CHECK(model.accepted_state_count == record.point_count);
  CHECK(record.stop_reason == SS_ANALYSIS_STOP_MINIMUM_STEP_REACHED);
  CHECK(record.terminal_success == 0);
  memset(&model, 0, sizeof(model));
  model.mode = SEED_SHIFT_DIRECT;
  model.decreasing_direct_control = 1;
  policy.factor_maximum = 4.0;
  CHECK(VecSet(initial, 0.0) == PETSC_SUCCESS);
  CHECK(VecSet(accepted, 77.0) == PETSC_SUCCESS);
  CHECK(ss_analysis_run(&problem, &policy, initial, accepted, &record) == SS_OK);
  CHECK(check_analysis_history(&record) == 0);
  CHECK(record.point_count == 1U);
  CHECK(model.accepted_state_count == record.point_count);
  CHECK(record.rollback_count == 5U && record.attempt_count == 6U);
  CHECK(record.stop_reason == SS_ANALYSIS_STOP_CONSECUTIVE_RETRY_LIMIT);
  CHECK(record.terminal_success == 0);
  CHECK(check_scalar(accepted, 1.0, 1.0e-14) == 0);
  memset(&model, 0, sizeof(model));
  model.mode = SEED_SHIFT_DIRECT;
  model.fail_fixed = 1;
  CHECK(VecSet(initial, 0.0) == PETSC_SUCCESS);
  CHECK(VecSet(accepted, 77.0) == PETSC_SUCCESS);
  CHECK(ss_analysis_run(&problem, &policy, initial, accepted, &record) == SS_OK);
  CHECK(check_analysis_history(&record) == 0);
  CHECK(record.point_count == 0U);
  CHECK(model.accepted_state_count == 0U);
  CHECK(record.rollback_count == 5U && record.attempt_count == 5U);
  CHECK(record.stop_reason == SS_ANALYSIS_STOP_CONSECUTIVE_RETRY_LIMIT);
  CHECK(record.terminal_success == 0);
  CHECK(check_scalar(accepted, 77.0, 0.0) == 0);
  CHECK(VecDestroy(&accepted) == PETSC_SUCCESS);
  CHECK(VecDestroy(&initial) == PETSC_SUCCESS);
  CHECK(VecDestroy(&load) == PETSC_SUCCESS);
  return 0;
}

static int test_indirect_seed_shift(void) {
  SeedShiftModel model = {.mode = SEED_SHIFT_INDIRECT};
  SsAnalysisProblem problem;
  SsAnalysisPolicy policy;
  SsAnalysisRecord record;
  Vec load = NULL;
  Vec initial = NULL;
  Vec accepted = NULL;
  CHECK(create_vector(1, 1.0, &load) == 0);
  CHECK(create_vector(1, 0.0, &initial) == 0);
  CHECK(VecDuplicate(initial, &accepted) == PETSC_SUCCESS);
  memset(&problem, 0, sizeof(problem));
  problem.communicator = PETSC_COMM_SELF;
  problem.reference_load = load;
  problem.operations.solve_fixed = seed_shift_fixed;
  problem.operations.solve_work_controlled = seed_shift_work;
  problem.operations.capture_accepted_state = seed_shift_capture_accepted_state;
  problem.context = &model;
  fill_analysis_policy(SS_ANALYSIS_SSR_INDIRECT, &policy);
  policy.accepted_step_limit = 4;
  model.fail_first_work = 1;
  CHECK(ss_analysis_run(&problem, &policy, initial, accepted, &record) == SS_OK);
  CHECK(check_analysis_history(&record) == 0);
  CHECK(strcmp(record.method_identity, "ssr_indirect/classic_fd") == 0);
  CHECK(record.point_count == 4U);
  CHECK(model.accepted_state_count == record.point_count);
  CHECK(fabs(record.points[0].factor - 1.0) < 1.0e-14);
  CHECK(fabs(record.points[1].factor - 1.2) < 1.0e-14);
  CHECK(fabs(record.points[2].factor - 1.4) < 1.0e-14);
  CHECK(fabs(record.points[3].work - 0.4) < 1.0e-14);
  CHECK(record.rollback_count == 1U && record.attempt_count == 5U);
  CHECK(record.attempts[3].rolled_back != 0);
  CHECK(record.attempts[4].retry_ordinal == 1);
  CHECK(record.stop_reason == SS_ANALYSIS_STOP_ACCEPTED_STEP_LIMIT);
  memset(&model, 0, sizeof(model));
  model.mode = SEED_SHIFT_INDIRECT;
  policy.factor_maximum = 1.2;
  CHECK(VecSet(initial, 0.0) == PETSC_SUCCESS);
  CHECK(VecSet(accepted, 0.0) == PETSC_SUCCESS);
  CHECK(ss_analysis_run(&problem, &policy, initial, accepted, &record) == SS_OK);
  CHECK(check_analysis_history(&record) == 0);
  CHECK(record.point_count == 2U);
  CHECK(model.accepted_state_count == record.point_count);
  CHECK(record.stop_reason == SS_ANALYSIS_STOP_MINIMUM_STEP_REACHED);
  CHECK(record.terminal_success == 0);
  memset(&model, 0, sizeof(model));
  model.mode = SEED_SHIFT_INDIRECT;
  model.fixed_failures_remaining = 5;
  model.fixed_work_tracks_factor = 1;
  policy.factor_maximum = 4.0;
  policy.factor_step_initial = 0.1;
  policy.factor_step_minimum = 0.001;
  policy.accepted_step_limit = 2;
  CHECK(VecSet(initial, 0.0) == PETSC_SUCCESS);
  CHECK(VecSet(accepted, 0.0) == PETSC_SUCCESS);
  CHECK(ss_analysis_run(&problem, &policy, initial, accepted, &record) == SS_OK);
  CHECK(check_analysis_history(&record) == 0);
  CHECK(record.point_count == 2U);
  CHECK(model.accepted_state_count == record.point_count);
  CHECK(record.attempt_count == 7U && record.rollback_count == 5U);
  CHECK(record.attempts[5].retry_ordinal == 5);
  CHECK(fabs(record.points[0].factor - 0.03125) < 1.0e-14);
  CHECK(fabs(record.points[1].factor - 0.034375) < 1.0e-14);
  CHECK(record.stop_reason == SS_ANALYSIS_STOP_ACCEPTED_STEP_LIMIT);
  CHECK(record.terminal_success != 0);
  memset(&model, 0, sizeof(model));
  model.mode = SEED_SHIFT_INDIRECT;
  model.fixed_work_scale = 0.005;
  fill_analysis_policy(SS_ANALYSIS_SSR_INDIRECT, &policy);
  policy.accepted_step_limit = 3;
  CHECK(VecSet(initial, 0.0) == PETSC_SUCCESS);
  CHECK(VecSet(accepted, 0.0) == PETSC_SUCCESS);
  CHECK(ss_analysis_run(&problem, &policy, initial, accepted, &record) == SS_OK);
  CHECK(check_analysis_history(&record) == 0);
  CHECK(record.point_count == 3U && record.attempt_count == 3U);
  CHECK(fabs(record.points[0].work - 0.005) < 1.0e-14);
  CHECK(fabs(record.points[1].work - 0.006) < 1.0e-14);
  CHECK(fabs(record.attempts[2].requested_step - policy.work_step_minimum) < 1.0e-14);
  CHECK(fabs(record.attempts[2].actual_step - policy.work_step_minimum) < 1.0e-14);
  CHECK(fabs(model.first_work_target -
             (record.points[1].work + policy.work_step_minimum)) < 1.0e-14);
  CHECK(record.stop_reason == SS_ANALYSIS_STOP_ACCEPTED_STEP_LIMIT);
  CHECK(record.terminal_success != 0);
  CHECK(VecDestroy(&accepted) == PETSC_SUCCESS);
  CHECK(VecDestroy(&initial) == PETSC_SUCCESS);
  CHECK(VecDestroy(&load) == PETSC_SUCCESS);
  return 0;
}

static int test_outer_deflation_transaction(void) {
  AnalysisModel model;
  SsAnalysisProblem problem;
  SsAnalysisPolicy policy;
  SsAnalysisRecord record;
  SsDeflation *deflation = NULL;
  SsDeflationPolicy deflation_policy;
  SsDeflationDiagnostics diagnostics;
  SsLinearPolicy *linear_policy;
  Vec load = NULL;
  Vec initial = NULL;
  Vec accepted = NULL;
  Mat elastic = NULL;
  CHECK(create_scalar_nonlinear(&model, 2, &load, &elastic) == 0);
  memset(&deflation_policy, 0, sizeof(deflation_policy));
  deflation_policy.dimension_cap = 4U;
  deflation_policy.norm_tolerance = 1.0e-13;
  deflation_policy.pivot_tolerance = 1.0e-13;
  deflation_policy.condition_limit = 1.0e12;
  deflation_policy.invariant_tolerance = 1.0e-10;
  deflation_policy.reorthogonalization_passes = 2U;
  CHECK(ss_deflation_create(PETSC_COMM_SELF, &deflation_policy, &deflation) == SS_OK);
  model.outer_deflation = deflation;
  model.nonlinear.deflation = deflation;
  model.nonlinear.deflation_mode = SS_NONLINEAR_DEFLATION_EXTERNAL_TRANSACTION;
  model.fail_after_fixed_solve_call = 1;
  linear_policy = &model.nonlinear.linear_policies.policies[0];
  (void)snprintf(linear_policy->identity, sizeof(linear_policy->identity), "%s",
                 "nonlinear-analysis-outer-deflation");
  (void)snprintf(linear_policy->options_prefix, sizeof(linear_policy->options_prefix),
                 "%s", "nonlinear_outer_deflation_");
  linear_policy->operator_class = SS_LINEAR_OPERATOR_GENERAL;
  linear_policy->method = SS_LINEAR_METHOD_FGMRES;
  linear_policy->pc = SS_LINEAR_PC_JACOBI;
  linear_policy->side = SS_LINEAR_SIDE_RIGHT;
  linear_policy->maximum_iterations = 32;
  linear_policy->restart = 8;
  fill_analysis_problem(&model, load, &problem);
  fill_analysis_policy(SS_ANALYSIS_SSR_INDIRECT, &policy);
  policy.work_maximum = 1.0;
  policy.accepted_step_limit = 5;
  CHECK(create_vector(2, 0.0, &initial) == 0);
  CHECK(VecDuplicate(initial, &accepted) == PETSC_SUCCESS);
  CHECK(ss_analysis_run(&problem, &policy, initial, accepted, &record) == SS_OK);
  CHECK(check_analysis_history(&record) == 0);
  CHECK(record.rollback_count >= 1U && record.commit_count >= 2U);
  CHECK(record.attempts[0].rolled_back != 0);
  CHECK(model.saw_rollback_provisional != 0);
  CHECK(model.deflation_rollback_preserved != 0);
  CHECK(model.deflation_commit_closed != 0);
  CHECK(model.transaction_open == 0);
  CHECK(ss_deflation_diagnostics(deflation, &diagnostics) == SS_OK);
  CHECK(diagnostics.persistent_count > 0U && diagnostics.provisional_count == 0U &&
        diagnostics.active_count == diagnostics.persistent_count);
  CHECK(VecDestroy(&accepted) == PETSC_SUCCESS);
  CHECK(VecDestroy(&initial) == PETSC_SUCCESS);
  CHECK(ss_deflation_destroy(&deflation) == SS_OK);
  CHECK(VecDestroy(&load) == PETSC_SUCCESS);
  CHECK(MatDestroy(&elastic) == PETSC_SUCCESS);
  return 0;
}

int test_direct_coordinate(void) {
  AnalysisModel model;
  SsAnalysisProblem problem;
  SsAnalysisPolicy policy;
  SsAnalysisDirectPointRecord record;
  Vec load = NULL;
  Vec incoming = NULL;
  Vec solution = NULL;
  Mat elastic = NULL;
  CHECK(create_scalar_nonlinear(&model, 1, &load, &elastic) == 0);
  fill_analysis_problem(&model, load, &problem);
  fill_analysis_policy(SS_ANALYSIS_SSR_DIRECT, &policy);
  CHECK(create_vector(1, 0.0, &incoming) == 0);
  CHECK(VecDuplicate(incoming, &solution) == PETSC_SUCCESS);
  CHECK(ss_analysis_direct_point(&problem, &policy, 1.0, 0.2, incoming, solution,
                                 &record) == SS_OK);
  CHECK(record.primary_solve.outcome == SS_NONLINEAR_CONVERGED);
  CHECK(record.sensitivity_solve.outcome == SS_NONLINEAR_CONVERGED);
  CHECK(fabs(record.direct_control - 0.10278471520016591) < 1.0e-7);
  CHECK(fabs(record.external_work - 0.4533976515161187) < 1.0e-10);
  CHECK(fabs(record.direct_control - record.external_work) > 0.3);
  CHECK(record.epsilon == 1.0e-8 && record.beta == 5.0e-8);
  model.model.evaluations = 0;
  model.model.cancel_after_evaluations = 1;
  CHECK(VecSet(solution, 77.0) == PETSC_SUCCESS);
  CHECK(ss_analysis_direct_point(&problem, &policy, 1.0, 0.2, incoming, solution,
                                 &record) == SS_OK);
  CHECK(record.outcome == SS_NONLINEAR_CANCELLED);
  CHECK(check_scalar(solution, 77.0, 0.0) == 0);
  CHECK(VecDestroy(&solution) == PETSC_SUCCESS);
  CHECK(VecDestroy(&incoming) == PETSC_SUCCESS);
  CHECK(VecDestroy(&load) == PETSC_SUCCESS);
  CHECK(MatDestroy(&elastic) == PETSC_SUCCESS);
  return 0;
}

int test_limit_load(void) {
  AnalysisModel model;
  SsAnalysisProblem problem;
  SsAnalysisPolicy policy;
  SsAnalysisRecord record;
  SsLinearSetupRecord elastic_setup;
  SsLinearSolveRecord elastic_solve;
  Vec load = NULL;
  Vec initial = NULL;
  Vec accepted = NULL;
  Mat elastic = NULL;
  PetscReal elastic_reference_work = NAN;
  size_t index;
  int saw_rollback = 0;
  int saw_growth = 0;
  CHECK(create_scalar_nonlinear(&model, 1, &load, &elastic) == 0);
  fill_analysis_problem(&model, load, &problem);
  fill_analysis_policy(SS_ANALYSIS_LIMIT_LOAD, &policy);
  policy.work_maximum = 0.4;
  policy.accepted_step_limit = 8;
  model.fail_one_limit_load_attempt = 1;
  CHECK(create_vector(1, 0.0, &initial) == 0);
  CHECK(VecDuplicate(initial, &accepted) == PETSC_SUCCESS);
  memset(&elastic_setup, 0, sizeof(elastic_setup));
  memset(&elastic_solve, 0, sizeof(elastic_solve));
  CHECK(ss_nonlinear_elastic_predictor(&model.nonlinear, 0.2, initial,
                                       &elastic_reference_work, &elastic_setup,
                                       &elastic_solve) == SS_OK);
  CHECK(elastic_solve.status == SS_LINEAR_SOLVE_CONVERGED);
  CHECK(strcmp(elastic_setup.selected_identity, "nonlinear-direct-validation") == 0);
  CHECK(fabs(elastic_reference_work - 0.5) < 1.0e-14);
  CHECK(check_scalar(initial, 0.1, 1.0e-14) == 0);
  CHECK(VecSet(initial, 0.0) == PETSC_SUCCESS);
  CHECK(ss_analysis_run(&problem, &policy, initial, accepted, &record) == SS_OK);
  CHECK(check_analysis_history(&record) == 0);
  CHECK(strcmp(record.method_identity, "limit_load/work_classic") == 0);
  CHECK(record.method == SS_ANALYSIS_LIMIT_LOAD);
  CHECK(record.points[0].control == 0.0 && record.points[0].factor == 0.0);
  CHECK(record.point_count >= 3U && record.commit_count == record.accepted_step_count);
  CHECK(record.stop_reason == SS_ANALYSIS_STOP_WORK_HORIZON_REACHED);
  CHECK(fabs(record.final_control - 0.4) < 1.0e-14);
  CHECK(fabs(record.final_displacement_maximum - 0.8) < 1.0e-14);
  CHECK(model.displacement_maximum_calls == record.point_count);
  for (index = 0U; index < record.attempt_count; ++index) {
    if (record.attempts[index].rolled_back != 0) {
      saw_rollback = 1;
    }
    if (index > 0U && record.attempts[index].requested_step >
                          record.attempts[index - 1U].requested_step) {
      saw_growth = 1;
    }
  }
  CHECK(saw_rollback != 0 && saw_growth != 0);
  CHECK(record.rollback_count >= 1U);
  CHECK(model.transaction_begin_count == record.attempt_count);
  CHECK(model.transaction_commit_count == record.commit_count);
  CHECK(model.transaction_rollback_count == record.rollback_count);
  CHECK(model.transaction_open == 0);
  CHECK(model.maximum_limit_load_predictor_error < 1.0e-14);
  CHECK(VecDestroy(&accepted) == PETSC_SUCCESS);
  CHECK(VecDestroy(&initial) == PETSC_SUCCESS);
  CHECK(VecDestroy(&load) == PETSC_SUCCESS);
  CHECK(MatDestroy(&elastic) == PETSC_SUCCESS);
  return 0;
}

int test_direct_trajectory(void) {
  static const PetscReal expected_factors[8] = {1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4};
  static const PetscReal expected_controls[8] = {
      0.10278471520016591, 0.12011824602936996, 0.1411278592526789,
      0.1664885460313212,  0.1968565044236097,  0.2327856159383796,
      0.27464071386942185, 0.32253704883233003};
  AnalysisModel model;
  SsAnalysisProblem problem;
  SsAnalysisPolicy policy;
  SsAnalysisRecord record;
  Vec load = NULL;
  Vec initial = NULL;
  Vec accepted = NULL;
  Mat elastic = NULL;
  size_t index;
  CHECK(create_scalar_nonlinear(&model, 1, &load, &elastic) == 0);
  fill_analysis_problem(&model, load, &problem);
  fill_analysis_policy(SS_ANALYSIS_SSR_DIRECT, &policy);
  CHECK(create_vector(1, 0.0, &initial) == 0);
  CHECK(VecDuplicate(initial, &accepted) == PETSC_SUCCESS);
  CHECK(ss_analysis_run(&problem, &policy, initial, accepted, &record) == SS_OK);
  CHECK(check_analysis_history(&record) == 0);
  CHECK(strcmp(record.method_identity, "ssr_direct/reference_fd") == 0);
  CHECK(record.point_count == 8U);
  CHECK(record.stop_reason == SS_ANALYSIS_STOP_ACCEPTED_STEP_LIMIT);
  CHECK(model.transaction_begin_count == record.attempt_count);
  CHECK(model.transaction_commit_count == record.commit_count);
  CHECK(model.transaction_rollback_count == record.rollback_count);
  CHECK(model.transaction_open == 0);
  for (index = 0U; index < 8U; ++index) {
    CHECK(fabs(record.points[index].factor - expected_factors[index]) < 2.0e-14);
    CHECK(fabs(record.points[index].control - expected_controls[index]) < 1.0e-7);
    CHECK(fabs(record.points[index].work - record.points[index].control) > 0.1);
    CHECK(record.points[index].restartable != 0);
    CHECK(record.attempts[record.points[index].source_attempt_ordinal].predictor ==
          SS_ANALYSIS_PREDICTOR_PAIRED_BETA);
    CHECK(record.attempts[record.points[index].source_attempt_ordinal]
              .primary_solve.outcome == SS_NONLINEAR_CONVERGED);
    CHECK(record.attempts[record.points[index].source_attempt_ordinal]
              .secondary_solve.outcome == SS_NONLINEAR_CONVERGED);
  }
  CHECK(record.linear_setup_count > 0U && record.linear_solve_count > 0U);
  CHECK(VecDestroy(&accepted) == PETSC_SUCCESS);
  CHECK(VecDestroy(&initial) == PETSC_SUCCESS);
  CHECK(VecDestroy(&load) == PETSC_SUCCESS);
  CHECK(MatDestroy(&elastic) == PETSC_SUCCESS);
  CHECK(test_direct_seed_shift() == 0);
  return 0;
}

int test_indirect_trajectory(void) {
  static const PetscReal expected_factors[7] = {1.0,
                                                1.2,
                                                1.3796411766968852,
                                                1.6947810689647629,
                                                1.9700544052662803,
                                                2.455431923679192,
                                                2.546029411765137};
  static const PetscReal expected_work[7] = {0.4533976515161187,
                                             0.49013925782244777,
                                             0.5268808641287768,
                                             0.6003640767414348,
                                             0.6738472893540928,
                                             0.8208137145794088,
                                             0.85};
  AnalysisModel model;
  SsAnalysisProblem problem;
  SsAnalysisPolicy policy;
  SsAnalysisRecord record;
  Vec load = NULL;
  Vec initial = NULL;
  Vec accepted = NULL;
  Mat elastic = NULL;
  size_t index;
  CHECK(create_scalar_nonlinear(&model, 1, &load, &elastic) == 0);
  fill_analysis_problem(&model, load, &problem);
  fill_analysis_policy(SS_ANALYSIS_SSR_INDIRECT, &policy);
  CHECK(create_vector(1, 0.0, &initial) == 0);
  CHECK(VecDuplicate(initial, &accepted) == PETSC_SUCCESS);
  CHECK(ss_analysis_run(&problem, &policy, initial, accepted, &record) == SS_OK);
  CHECK(check_analysis_history(&record) == 0);
  CHECK(strcmp(record.method_identity, "ssr_indirect/classic_fd") == 0);
  CHECK(record.point_count == 7U);
  CHECK(record.stop_reason == SS_ANALYSIS_STOP_WORK_HORIZON_REACHED);
  CHECK(model.transaction_begin_count == record.attempt_count);
  CHECK(model.transaction_commit_count == record.commit_count);
  CHECK(model.transaction_rollback_count == record.rollback_count);
  CHECK(model.transaction_open == 0);
  for (index = 0U; index < 7U; ++index) {
    CHECK(fabs(record.points[index].factor - expected_factors[index]) < 3.0e-10);
    CHECK(fabs(record.points[index].work - expected_work[index]) < 3.0e-13);
    CHECK(record.points[index].restartable != 0);
  }
  {
    int saw_augmented = 0;
    for (index = 0U; index < record.attempt_count; ++index) {
      const SsAnalysisAttemptRecord *attempt = &record.attempts[index];
      if (attempt->predictor ==
              SS_ANALYSIS_PREDICTOR_DISPLACEMENT_SECANT_ANCHORED_FACTOR &&
          attempt->primary_solve.first_linear_solve.solve_count != 0U &&
          attempt->primary_solve.second_linear_solve.solve_count != 0U &&
          attempt->sensitivity_epsilon > 0.0) {
        saw_augmented = 1;
      }
    }
    CHECK(saw_augmented != 0);
  }
  CHECK(record.linear_setup_count > 0U && record.linear_solve_count > 0U);
  CHECK(VecDestroy(&accepted) == PETSC_SUCCESS);
  CHECK(VecDestroy(&initial) == PETSC_SUCCESS);
  CHECK(VecDestroy(&load) == PETSC_SUCCESS);
  CHECK(MatDestroy(&elastic) == PETSC_SUCCESS);
  CHECK(test_indirect_seed_shift() == 0);
  CHECK(test_outer_deflation_transaction() == 0);
  CHECK(mechanics_fixture_run_profiles() == 0);
  return 0;
}
