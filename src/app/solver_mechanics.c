/* Mechanics operators, callbacks, and solver lifecycle.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "solver_internal.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static SsStatus ss_solver_mechanics_natural_kernel(const void *raw_context,
                                                   const SsNaturalPointInput *input,
                                                   SsNaturalPointOutput *output) {
  const SsSolverBoundaryRule *rule = (const SsSolverBoundaryRule *)raw_context;
  uint32_t component;
  if (rule == NULL || input == NULL || output == NULL ||
      rule->condition_kind != SS_SOLVER_BOUNDARY_NATURAL ||
      strcmp(rule->field, "displacement") != 0 ||
      input->components != input->dimension ||
      (input->dimension != 2U && input->dimension != 3U)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(output, 0, sizeof(*output));
  if (rule->value_kind == SS_SOLVER_VALUE_GLOBAL_TRACTION) {
    if (rule->value_count != input->dimension) {
      return SS_ERROR_INTEGRITY;
    }
    memcpy(output->load, rule->values, input->dimension * sizeof(*output->load));
  } else if (rule->value_kind == SS_SOLVER_VALUE_SPATIAL_GLOBAL_TRACTION) {
    if (rule->value_count != input->dimension) {
      return SS_ERROR_INTEGRITY;
    }
    for (component = 0U; component < input->dimension; ++component) {
      uint32_t coordinate;
      output->load[component] = rule->values[component];
      for (coordinate = 0U; coordinate < input->dimension; ++coordinate) {
        output->load[component] +=
            rule->gradient[(size_t)component * input->dimension + coordinate] *
            input->physical[coordinate];
      }
    }
  } else if (rule->value_kind == SS_SOLVER_VALUE_NORMAL_PRESSURE ||
             rule->value_kind == SS_SOLVER_VALUE_HYDROSTATIC) {
    double pressure;
    if (rule->value_kind == SS_SOLVER_VALUE_NORMAL_PRESSURE) {
      if (rule->value_count != 1U) {
        return SS_ERROR_INTEGRITY;
      }
      pressure = rule->values[0];
    } else {
      pressure = rule->reference_pressure +
                 rule->water_unit_weight *
                     fmax(rule->reference_level - input->physical[1], 0.0);
    }
    if (!isfinite(pressure)) {
      return SS_ERROR_INTEGRITY;
    }
    for (component = 0U; component < input->dimension; ++component) {
      output->load[component] = pressure * input->outward_normal[component];
    }
  } else {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (component = 0U; component < input->dimension; ++component) {
    if (!isfinite(output->load[component])) {
      return SS_ERROR_INTEGRITY;
    }
  }
  return SS_OK;
}

static SsStatus ss_solver_mechanics_natural_operations(const SsSolverGraph *graph,
                                                       SsSolverMechanics *mechanics) {
  size_t count = 0U;
  size_t rule_index;
  size_t output_index = 0U;
  SsStatus status = SS_OK;
  if (graph == NULL || mechanics == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (rule_index = 0U; rule_index < graph->model.boundary_count; ++rule_index) {
    const SsSolverBoundaryRule *rule = &graph->model.boundaries[rule_index];
    if (rule->condition_kind == SS_SOLVER_BOUNDARY_NATURAL &&
        strcmp(rule->field, "displacement") == 0) {
      ++count;
    }
  }
  size_t bytes;
  status = ss_checked_mul_size(count, sizeof(*mechanics->natural_operations), &bytes);
  if (status == SS_OK && count != 0U) {
    mechanics->natural_operations = (SsNaturalOperatorSpec *)calloc(1U, bytes);
    if (mechanics->natural_operations == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  for (rule_index = 0U; rule_index < graph->model.boundary_count; ++rule_index) {
    const SsSolverBoundaryRule *rule = &graph->model.boundaries[rule_index];
    if (rule->condition_kind != SS_SOLVER_BOUNDARY_NATURAL ||
        strcmp(rule->field, "displacement") != 0) {
      continue;
    }
    if (output_index >= count) {
      return SS_ERROR_INTEGRITY;
    }
    mechanics->natural_operations[output_index++] = (SsNaturalOperatorSpec){
        rule->support_value, ss_solver_mechanics_natural_kernel, rule, 1};
  }
  mechanics->natural_operation_count = output_index;
  return output_index == count ? SS_OK : SS_ERROR_INTEGRITY;
}

SsStatus ss_solver_mechanics_evaluate_common(SsSolverMechanics *mechanics, Vec state,
                                             PetscReal reduction_factor,
                                             PetscReal external_load_factor,
                                             Vec residual, Mat tangent,
                                             PetscReal *functional) {
  const SsOperatorState operator_state = {
      .field = state,
      .reduction_factor = reduction_factor,
      .external_load_factor = external_load_factor,
      .cancelled = 0U,
  };
  SsErrorRecord error = {0};
  SsStatus status;
  if (mechanics == NULL || mechanics->operator_handle == NULL || state == NULL ||
      residual == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (functional != NULL) {
    double value = NAN;
    status =
        tangent == NULL
            ? ss_operator_residual_functional(mechanics->operator_handle,
                                              &operator_state, residual, &value, &error)
            : ss_operator_residual_jacobian_functional(mechanics->operator_handle,
                                                       &operator_state, residual,
                                                       tangent, &value, &error);
    if (status == SS_OK) {
      *functional = (PetscReal)value;
    }
  } else {
    status = tangent == NULL ? ss_operator_residual(mechanics->operator_handle,
                                                    &operator_state, residual, &error)
                             : ss_operator_residual_jacobian(mechanics->operator_handle,
                                                             &operator_state, residual,
                                                             tangent, &error);
  }
  return status;
}

static SsStatus ss_solver_mechanics_evaluate(void *opaque, Vec state, PetscReal factor,
                                             Vec residual, Mat tangent,
                                             PetscReal *functional) {
  return ss_solver_mechanics_evaluate_common((SsSolverMechanics *)opaque, state, factor,
                                             1.0, residual, tangent, functional);
}

static SsStatus ss_solver_mechanics_controlled(void *opaque,
                                               SsNonlinearControlMode mode, Vec state,
                                               PetscReal control, Vec residual,
                                               Mat tangent, PetscReal *functional) {
  SsSolverMechanics *mechanics = (SsSolverMechanics *)opaque;
  if (mode == SS_NONLINEAR_CONTROL_LIMIT_LOAD) {
    return ss_solver_mechanics_evaluate_common(mechanics, state, 1.0, control, residual,
                                               tangent, functional);
  }
  if (mode == SS_NONLINEAR_CONTROL_SSR_INDIRECT) {
    return ss_solver_mechanics_evaluate_common(mechanics, state, control, 1.0, residual,
                                               tangent, functional);
  }
  return SS_ERROR_INVALID_ARGUMENT;
}

static SsStatus ss_solver_mechanics_admissible(void *opaque, Vec state,
                                               PetscReal factor, int *admissible) {
  PetscReal norm = NAN;
  (void)opaque;
  if (state == NULL || admissible == NULL ||
      VecNorm(state, NORM_INFINITY, &norm) != PETSC_SUCCESS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *admissible = isfinite((double)norm) && isfinite((double)factor);
  return SS_OK;
}

SsStatus ss_solver_linear_policy(const SsSolverGraph *graph, int hydraulic,
                                 SsLinearPolicyChain *chain) {
  const SsSolverRunPolicy *run;
  SsLinearPolicy *policy;
  int use_pmg;
  SsLinearMethod method;
  if (graph == NULL || chain == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  run = &graph->policy;
  use_pmg = run->pmg_enabled != 0;
  method = hydraulic != 0 ? SS_LINEAR_METHOD_FGMRES : run->linear_method;
  memset(chain, 0, sizeof(*chain));
  chain->count = 1U;
  policy = &chain->policies[0];
  (void)snprintf(
      policy->identity, sizeof(policy->identity), "%s",
      method == SS_LINEAR_METHOD_HPDDM_GCRODR
          ? (use_pmg ? "native-hpddm-gcrodr-pmg" : "native-hpddm-gcrodr-jacobi")
          : (use_pmg ? "native-fgmres-pmg" : "native-fgmres-jacobi"));
  (void)snprintf(policy->options_prefix, sizeof(policy->options_prefix), "%s",
                 use_pmg ? "native_pmg_" : "native_jacobi_");
  policy->operator_class = SS_LINEAR_OPERATOR_GENERAL;
  policy->method = method;
  policy->pc = use_pmg ? SS_LINEAR_PC_PMG : SS_LINEAR_PC_JACOBI;
  policy->side = SS_LINEAR_SIDE_RIGHT;
  /*
   * The mechanics projector may intentionally use a projected-RHS stopping
   * test.  A hydraulic tangent is generally nonsymmetric, however, and the
   * mechanics A-orthogonal deflation contract does not certify its projected
   * residual as the residual of the original seepage system.  Requiring the
   * true original-system residual prevents a projected solve from reporting
   * convergence while Newton still sees a large residual.
   */
  policy->stopping_condition = hydraulic != 0
                                   ? SS_LINEAR_STOP_ORIGINAL_RHS_TRUE_RESIDUAL
                                   : run->linear_stopping_condition;
  policy->initial_guess_policy =
      hydraulic != 0 ? SS_LINEAR_INITIAL_GUESS_COARSE_ONLY
                     : (SsLinearInitialGuessPolicy)run->settings.linear_initial_guess;
  policy->orthogonalization =
      hydraulic != 0
          ? SS_LINEAR_ORTHOGONALIZATION_PETSC
          : (SsLinearOrthogonalization)run->settings.linear_orthogonalization;
  policy->rtol =
      (PetscReal)(hydraulic != 0 ? run->hydraulic_linear_rtol : run->linear_rtol);
  policy->atol =
      (PetscReal)(hydraulic != 0 ? run->hydraulic_linear_atol : run->linear_atol);
  policy->dtol = 1.0e8;
  policy->maximum_iterations = hydraulic != 0 ? run->hydraulic_linear_maximum_iterations
                                              : run->linear_maximum_iterations;
  policy->restart = run->linear_restart;
  policy->recycle_dimension =
      method == SS_LINEAR_METHOD_HPDDM_GCRODR ? run->linear_recycle_dimension : 0;
  return ss_solver_cap_bytes(run->settings.resources_linear_vector_memory_cap_mib,
                             &policy->vector_memory_cap_bytes);
}

static void ss_solver_deflation_policy(const SsSolverRunPolicy *run,
                                       SsDeflationPolicy *policy) {
  memset(policy, 0, sizeof(*policy));
  policy->dimension_cap = run->deflation_maximum_dimension;
  policy->norm_tolerance = (PetscReal)run->deflation_admission_tolerance;
  policy->pivot_tolerance = (PetscReal)run->deflation_pivot_tolerance;
  policy->condition_limit = (PetscReal)run->deflation_condition_limit;
  policy->invariant_tolerance = (PetscReal)run->deflation_invariant_tolerance;
  policy->reorthogonalization_passes =
      (uint32_t)run->deflation_reorthogonalization_passes;
  policy->projector = run->deflation_projector;
  policy->compression = run->deflation_compression;
  policy->compression_dimension = run->deflation_compression_dimension;
  policy->compression_history_decay =
      (PetscReal)run->deflation_compression_history_decay;
}

SsStatus ss_solver_deflation_create(const SsSolverGraph *graph,
                                    SsDeflation **deflation) {
  SsDeflationPolicy policy;
  if (graph == NULL || deflation == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *deflation = NULL;
  if (graph->policy.deflation_enabled == 0) {
    return SS_OK;
  }
  ss_solver_deflation_policy(&graph->policy, &policy);
  return ss_deflation_create(graph->communicator, &policy, deflation);
}

SsStatus ss_solver_admit_linear_vectors(const SsSolverGraph *graph, int hydraulic,
                                        SsErrorRecord *error) {
  const SsFieldSpace *space =
      hydraulic ? graph->hydraulic_space : graph->mechanics_space;
  SsFieldSummary summary = {0};
  SsLinearPolicyChain chain;
  size_t required = 0U;
  size_t basis_bytes = 0U;
  size_t dense_bytes = 0U;
  SsLinearKrylovCapture capture = SS_LINEAR_KRYLOV_CAPTURE_NONE;
  SsStatus status = ss_field_space_summary(space, &summary);
  if (status == SS_OK && summary.global_free_dof_count == 0U) {
    return SS_OK;
  }
  if (status == SS_OK && summary.global_free_dof_count > (uint64_t)PETSC_MAX_INT) {
    status = SS_ERROR_OVERFLOW;
  }
  if (status == SS_OK) {
    status = ss_solver_linear_policy(graph, hydraulic, &chain);
  }
  if (status == SS_OK && !hydraulic && graph->policy.deflation_enabled != 0) {
    SsDeflationPolicy policy;
    ss_solver_deflation_policy(&graph->policy, &policy);
    status = ss_deflation_policy_memory_estimate(
        &policy, (PetscInt)summary.global_free_dof_count, &basis_bytes, &dense_bytes);
    if (graph->policy.deflation_candidate_policy ==
        SS_DEFLATION_CANDIDATE_ACCEPTED_STATES_AND_FULL_KRYLOV) {
      capture = SS_LINEAR_KRYLOV_CAPTURE_FULL_CORRECTION_SPACE;
    }
  }
  if (status == SS_OK) {
    status =
        ss_linear_admit_vectors((PetscInt)summary.global_free_dof_count,
                                &chain.policies[0], capture, basis_bytes, &required);
    if (status == SS_ERROR_RESOURCE) {
      char message[SS_ERROR_MESSAGE_CAPACITY];
      (void)snprintf(message, sizeof(message),
                     "%s Krylov and deflation vectors require %zu bytes for %llu free "
                     "rows, exceeding the resolved cap of %zu bytes "
                     "(resources.linear_vector_memory_cap_mib)",
                     hydraulic ? "Seepage" : "Mechanics", required,
                     (unsigned long long)summary.global_free_dof_count,
                     chain.policies[0].vector_memory_cap_bytes);
      ss_solver_error(error, status, "LINEAR_VECTOR_MEMORY_LIMIT", message, 3U);
    }
  }
  return status;
}

SsStatus ss_solver_mechanics_matrix_free_linearize_common(
    SsSolverMechanics *mechanics, Vec fine_state, PetscReal reduction_factor,
    PetscReal external_load_factor, PetscReal elastic_regularization) {
  SsOperatorState state = {0};
  size_t finest;
  size_t level;
  SsStatus status = SS_OK;
  if (mechanics == NULL || mechanics->matrix_free == 0 || fine_state == NULL ||
      mechanics->elastic_reference == NULL || !isfinite((double)reduction_factor) ||
      reduction_factor <= 0.0 || !isfinite((double)external_load_factor) ||
      !isfinite((double)elastic_regularization) || elastic_regularization < 0.0 ||
      elastic_regularization > 1.0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  finest = mechanics->matrix_free_level_count == 0U
               ? 0U
               : mechanics->matrix_free_level_count - 1U;
  /* Exact Galerkin projection consumes the fine cached tangents directly, so
   * coarse nodal states are needed only by the rediscretized fallback. */
  if (mechanics->pmg != NULL && mechanics->matrix_free_galerkin == 0) {
    for (level = finest; level > 0U; --level) {
      Vec source = level == finest ? fine_state : mechanics->matrix_free_states[level];
      if (mechanics->matrix_free_states[level - 1U] == NULL) {
        continue;
      }
      status = ss_status_allreduce_max(
          mechanics->graph->communicator,
          MatMult(mechanics->matrix_free_injections[level - 1U], source,
                  mechanics->matrix_free_states[level - 1U]) == PETSC_SUCCESS
              ? SS_OK
              : SS_ERROR_INTERNAL);
      if (status != SS_OK) {
        return status;
      }
    }
  }
  state.field = fine_state;
  state.reduction_factor = (double)reduction_factor;
  state.external_load_factor = (double)external_load_factor;
  status = ss_operator_matrix_free_linearize(mechanics->operator_handle, &state,
                                             (double)elastic_regularization, NULL);
  status = ss_status_allreduce_max(mechanics->graph->communicator, status);
  for (level = 1U; status == SS_OK && mechanics->pmg != NULL && level < finest;
       ++level) {
    state.field = mechanics->matrix_free_states[level];
    if (mechanics->matrix_free_galerkin != 0) {
      status = ss_operator_matrix_free_project_coarse(
          mechanics->operator_handle, mechanics->matrix_free_operators[level], NULL);
    } else {
      status = ss_operator_matrix_free_linearize(
          mechanics->matrix_free_operators[level], &state,
          (double)elastic_regularization, NULL);
    }
    status = ss_status_allreduce_max(mechanics->graph->communicator, status);
  }
  if (status == SS_OK && mechanics->pmg != NULL && finest != 0U) {
    status = ss_operator_matrix_free_project_p1_jacobian(
        mechanics->operator_handle, mechanics->matrix_free_plans[0],
        mechanics->matrix_free_level_operators[0], NULL);
  }
  return status;
}

SsStatus ss_solver_mechanics_matrix_free_linearize(void *opaque, Vec state,
                                                   PetscReal factor,
                                                   PetscReal elastic_regularization) {
  return ss_solver_mechanics_matrix_free_linearize_common(
      (SsSolverMechanics *)opaque, state, factor, 1.0, elastic_regularization);
}

SsStatus ss_solver_mechanics_matrix_free_controlled_linearize(
    void *opaque, SsNonlinearControlMode mode, Vec state, PetscReal control,
    PetscReal elastic_regularization) {
  SsSolverMechanics *mechanics = (SsSolverMechanics *)opaque;
  if (mode == SS_NONLINEAR_CONTROL_LIMIT_LOAD) {
    return ss_solver_mechanics_matrix_free_linearize_common(
        mechanics, state, 1.0, control, elastic_regularization);
  }
  if (mode == SS_NONLINEAR_CONTROL_SSR_INDIRECT) {
    return ss_solver_mechanics_matrix_free_linearize_common(
        mechanics, state, control, 1.0, elastic_regularization);
  }
  return SS_ERROR_INVALID_ARGUMENT;
}

SsStatus ss_solver_mechanics_create(SsSolverGraph *graph, int requested_threads,
                                    const SsSolverCouplingSnapshot *coupling_snapshot,
                                    SsSolverMechanics *mechanics,
                                    SsErrorRecord *error) {
  Vec zero = NULL;
  Vec residual = NULL;
  SsOperatorState operator_state = {0};
  SsSolverTuning tuning;
  PetscReal load_norm = NAN;
  SsStatus status;
  const char *create_stage = "operator";
  if (graph == NULL || mechanics == NULL || graph->mechanics_plan == NULL ||
      graph->mechanics_region_count == 0U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(mechanics, 0, sizeof(*mechanics));
  if (graph->policy.linear_method == SS_LINEAR_METHOD_FCG) {
    ss_solver_error(error, SS_ERROR_TYPE_MISMATCH, "MECHANICS_FCG_UNQUALIFIED",
                    "mechanics FCG requires verified positive-definite tangents and "
                    "a compatible positive preconditioner",
                    3U);
    return SS_ERROR_TYPE_MISMATCH;
  }
  mechanics->graph = graph;
  mechanics->mechanics_spec.regions = graph->mechanics_regions;
  mechanics->mechanics_spec.region_count = graph->mechanics_region_count;
  if (coupling_snapshot != NULL) {
    mechanics->mechanics_spec.coupling_point_kernel =
        ss_solver_coupling_snapshot_mechanics_force;
    mechanics->mechanics_spec.coupling_point_context = coupling_snapshot;
  } else if (graph->model.prescribed_saturation.count != 0U) {
    mechanics->mechanics_spec.coupling_point_kernel =
        ss_solver_prescribed_mechanics_force;
    mechanics->mechanics_spec.coupling_point_context = &graph->model;
    mechanics->mechanics_spec.point_force_mode = SS_MECHANICS_POINT_FORCE_LOAD;
  }
  mechanics->operator_spec.physics = SS_ASSEMBLY_MECHANICS;
  /* The mechanics operators use the problem-level volume quadrature policy so
   * that outputs, coupling targets and operators share one rule. */
  mechanics->operator_spec.volume_quadrature_mode = graph->model.volume_quadrature_mode;
  status = ss_solver_mechanics_tuning(graph, mechanics, &tuning);
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  mechanics->operator_spec.requested_threads =
      requested_threads > 0 ? requested_threads : 1;
  mechanics->operator_spec.mechanics = &mechanics->mechanics_spec;
  mechanics->operator_spec.matrix_free = mechanics->matrix_free;
  create_stage = "natural_operations";
  status = ss_solver_mechanics_natural_operations(graph, mechanics);
  mechanics->operator_spec.natural_operations = mechanics->natural_operations;
  mechanics->operator_spec.natural_operation_count = mechanics->natural_operation_count;
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    create_stage = "operator";
    status = ss_operator_create(graph->mechanics_plan, &mechanics->operator_spec,
                                &mechanics->operator_handle, error);
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    create_stage = "reference_load";
    status = ss_assembly_plan_create_vector(graph->mechanics_plan,
                                            &mechanics->reference_load);
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    create_stage = "elastic_reference";
    status = mechanics->matrix_free != 0
                 ? ss_operator_matrix_free_create(mechanics->operator_handle,
                                                  &mechanics->elastic_reference, error)
                 : ss_assembly_plan_create_matrix(graph->mechanics_plan,
                                                  &mechanics->elastic_reference);
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    create_stage = "zero_state";
    status = ss_assembly_plan_create_vector(graph->mechanics_plan, &zero);
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK && (VecDuplicate(zero, &residual) != PETSC_SUCCESS ||
                          VecSet(zero, 0.0) != PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  operator_state.field = zero;
  operator_state.reduction_factor = 1.0;
  operator_state.external_load_factor = 1.0;
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    create_stage = "reference_residual";
    status = ss_operator_residual(mechanics->operator_handle, &operator_state, residual,
                                  error);
  }
  if (status == SS_OK &&
      (VecCopy(residual, mechanics->reference_load) != PETSC_SUCCESS ||
       VecScale(mechanics->reference_load, -1.0) != PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  operator_state.external_load_factor = 0.0;
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    create_stage = "elastic_jacobian";
    status = mechanics->matrix_free != 0
                 ? ss_operator_matrix_free_linearize(mechanics->operator_handle,
                                                     &operator_state, 1.0, error)
                 : ss_operator_jacobian(mechanics->operator_handle, &operator_state,
                                        mechanics->elastic_reference, error);
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK &&
      VecNorm(mechanics->reference_load, NORM_2, &load_norm) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK && (!isfinite((double)load_norm) || load_norm <= DBL_MIN)) {
    status = SS_ERROR_INTEGRITY;
    ss_solver_error(error, status, "MECHANICS_ZERO_REFERENCE_LOAD",
                    "mechanics reference load is zero", 3U);
  }
  const PetscErrorCode residual_error = VecDestroy(&residual);
  const PetscErrorCode zero_error = VecDestroy(&zero);
  if (residual_error != PETSC_SUCCESS || zero_error != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK && graph->policy.pmg_enabled != 0) {
    create_stage = "pmg";
    status = ss_solver_pmg_create(
        graph, graph->mechanics_space, graph->model.mechanics_order,
        graph->model.dimension, 0, graph->mechanics_constraints,
        graph->mechanics_constraint_count, "native_mech",
        mechanics->matrix_free != 0 ? mechanics->matrix_free_spaces : NULL,
        mechanics->matrix_free != 0 && mechanics->matrix_free_galerkin == 0
            ? mechanics->matrix_free_injections
            : NULL,
        &mechanics->pmg, error);
  }
  if (status == SS_OK && mechanics->matrix_free != 0 && mechanics->pmg != NULL) {
    create_stage = "matfree_hierarchy";
    status = ss_solver_mechanics_matrix_free_hierarchy_create(mechanics, error);
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    create_stage = "deflation";
    status = ss_solver_deflation_create(graph, &mechanics->deflation);
  }
  if (status == SS_OK && graph->policy.linear_method == SS_LINEAR_METHOD_HPDDM_GCRODR) {
    create_stage = "krylov_recycle";
    status = ss_linear_recycle_create(graph->communicator, &mechanics->krylov_recycle);
  }
  if (status == SS_OK) {
    create_stage = "linear_policy";
    status = ss_solver_linear_policy(graph, 0, &mechanics->nonlinear.linear_policies);
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    create_stage = "nonlinear_workspace";
    status = ss_nonlinear_workspace_create(graph->communicator,
                                           &mechanics->nonlinear.workspace);
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status != SS_OK) {
    if (error->valid == 0) {
      char message[SS_ERROR_MESSAGE_CAPACITY];
      const int written = snprintf(message, sizeof(message),
                                   "mechanics creation failed during %s (status=%d)",
                                   create_stage, (int)status);
      ss_solver_error(error, status, "MECHANICS_CREATE",
                      written >= 0 && (size_t)written < sizeof(message)
                          ? message
                          : "mechanics creation failed",
                      3U);
    }
    (void)ss_solver_mechanics_destroy(mechanics);
    return status;
  }
  mechanics->nonlinear.communicator = graph->communicator;
  mechanics->nonlinear.reference_load = mechanics->reference_load;
  mechanics->nonlinear.elastic_reference = mechanics->elastic_reference;
  mechanics->nonlinear.operations.evaluate = ss_solver_mechanics_evaluate;
  mechanics->nonlinear.operations.evaluate_controlled = ss_solver_mechanics_controlled;
  if (mechanics->matrix_free != 0) {
    mechanics->nonlinear.operations.linearize =
        ss_solver_mechanics_matrix_free_linearize;
    mechanics->nonlinear.operations.linearize_controlled =
        ss_solver_mechanics_matrix_free_controlled_linearize;
  }
  mechanics->nonlinear.operations.admissibility = ss_solver_mechanics_admissible;
  mechanics->nonlinear.context = mechanics;
  mechanics->nonlinear.pmg = mechanics->pmg;
  mechanics->nonlinear.deflation = mechanics->deflation;
  mechanics->nonlinear.krylov_recycle = mechanics->krylov_recycle;
  mechanics->nonlinear.deflation_mode =
      mechanics->deflation == NULL ? SS_NONLINEAR_DEFLATION_NONE
                                   : SS_NONLINEAR_DEFLATION_EXTERNAL_TRANSACTION;
  mechanics->nonlinear.deflation_promotion = SS_NONLINEAR_DEFLATION_PROMOTE_DIRECTIONS;
  mechanics->nonlinear.linear_properties.finite_action = 1;
  mechanics->nonlinear.linear_properties.pc_variable = mechanics->pmg != NULL;
  mechanics->nonlinear.structure_generation = 1U;
  mechanics->next_value_generation = 1U;
  mechanics->nonlinear.first_value_generation = mechanics->next_value_generation;
  mechanics->nonlinear.transaction_generation = 1U;
  mechanics->nonlinear.accepted_generation = 1U;
  mechanics->nonlinear_policy.tolerance =
      (PetscReal)fmax(graph->policy.nonlinear_atol, graph->policy.nonlinear_rtol);
  mechanics->nonlinear_policy.stopping_criterion =
      graph->policy.nonlinear_stopping_criterion;
  mechanics->nonlinear_policy.stopping_tolerance =
      (PetscReal)graph->policy.nonlinear_stopping_tolerance;
  mechanics->nonlinear_policy.maximum_iterations =
      graph->policy.nonlinear_maximum_iterations;
  mechanics->progress_guard_mode = tuning.indirect_progress_guard_mode;
  mechanics->nonlinear_policy.indirect_progress_guard_iteration =
      tuning.indirect_progress_guard_mode == SS_NONLINEAR_PROGRESS_GUARD_FIXED
          ? tuning.indirect_progress_guard_iteration
          : 0;
  mechanics->nonlinear_policy.indirect_progress_guard_maximum_convergence =
      tuning.indirect_progress_guard_mode == SS_NONLINEAR_PROGRESS_GUARD_FIXED
          ? tuning.indirect_progress_guard_maximum_convergence
          : 0.0;
  if (ss_solver_progress_guard_learns(tuning.indirect_progress_guard_mode) &&
      ss_nonlinear_adaptive_progress_guard_initialize(
          tuning.indirect_progress_guard_iteration,
          &mechanics->adaptive_progress_guard) != SS_OK) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  mechanics->nonlinear_policy.maximum_line_search_trials =
      tuning.nonlinear_maximum_line_search_trials;
  mechanics->nonlinear_policy.minimum_alpha = 1.0 / 1048576.0;
  mechanics->nonlinear_policy.regularization_minimum =
      (PetscReal)graph->policy.nonlinear_regularization_minimum;
  mechanics->nonlinear_policy.singular_relative_tolerance = 1.0e-12;
  mechanics->nonlinear_policy.relaxed_factor = 10.0;
  mechanics->nonlinear_policy.allow_relaxed =
      tuning.nonlinear_allow_relaxed != PETSC_FALSE ? 1 : 0;
  mechanics->nonlinear_policy.line_search_legacy_alg5 =
      tuning.indirect_line_search_legacy_alg5 != PETSC_FALSE ? 1 : 0;
  mechanics->nonlinear_policy.line_method = SS_NONLINEAR_LINE_DIRECTIONAL;
  mechanics->nonlinear_policy.control_sensitivity_method =
      tuning.control_sensitivity_method;
  mechanics->nonlinear_policy.control_sensitivity_relative_step =
      tuning.control_sensitivity_relative_step;
  mechanics->nonlinear_policy.indirect_adaptive_linear_tolerance =
      tuning.indirect_adaptive_linear_tolerance != PETSC_FALSE ? 1 : 0;
  mechanics->nonlinear_policy.indirect_adaptive_linear_iteration_threshold =
      tuning.indirect_adaptive_linear_iteration_threshold;
  mechanics->nonlinear_policy.indirect_adaptive_linear_alpha_threshold =
      tuning.indirect_adaptive_linear_alpha_threshold;
  mechanics->nonlinear_policy.indirect_adaptive_linear_tightening_factor =
      tuning.indirect_adaptive_linear_tightening_factor;
  mechanics->nonlinear_policy.indirect_adaptive_linear_minimum_relative_tolerance =
      tuning.indirect_adaptive_linear_minimum_relative_tolerance;
  mechanics->initialization_nonlinear_policy = mechanics->nonlinear_policy;
  mechanics->initialization_nonlinear_policy.stopping_criterion =
      graph->policy.initial_nonlinear_stopping_criterion;
  mechanics->initialization_nonlinear_policy.stopping_tolerance =
      (PetscReal)graph->policy.initial_nonlinear_stopping_tolerance;
  if (mechanics->initialization_nonlinear_policy.stopping_criterion ==
      SS_NONLINEAR_STOP_RELATIVE_RESIDUAL) {
    mechanics->initialization_nonlinear_policy.tolerance =
        fmin(mechanics->initialization_nonlinear_policy.tolerance,
             mechanics->initialization_nonlinear_policy.stopping_tolerance);
  }
  if (ss_nonlinear_policy_validate(&mechanics->nonlinear_policy) != SS_OK) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return ss_nonlinear_policy_validate(&mechanics->initialization_nonlinear_policy);
}

SsStatus ss_solver_mechanics_destroy(SsSolverMechanics *mechanics) {
  SsStatus status = SS_OK;
  size_t level;
  if (mechanics == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (ss_nonlinear_workspace_destroy(&mechanics->nonlinear.workspace) != SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (ss_pmg_destroy(&mechanics->pmg) != SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  for (level = 0U; level + 1U < mechanics->matrix_free_level_count; ++level) {
    const PetscErrorCode matrix_error =
        MatDestroy(&mechanics->matrix_free_level_operators[level]);
    const PetscErrorCode diagonal_error =
        MatDestroy(&mechanics->matrix_free_diagonal_operators[level]);
    const PetscErrorCode state_error =
        VecDestroy(&mechanics->matrix_free_states[level]);
    const SsStatus operator_status =
        ss_operator_destroy(&mechanics->matrix_free_operators[level]);
    const SsStatus plan_status =
        ss_assembly_plan_destroy(&mechanics->matrix_free_plans[level]);
    const SsStatus space_status =
        ss_field_space_destroy(&mechanics->matrix_free_spaces[level]);
    const PetscErrorCode injection_error =
        MatDestroy(&mechanics->matrix_free_injections[level]);
    if (matrix_error != PETSC_SUCCESS || diagonal_error != PETSC_SUCCESS ||
        state_error != PETSC_SUCCESS || operator_status != SS_OK ||
        plan_status != SS_OK || space_status != SS_OK ||
        injection_error != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  const PetscErrorCode elastic_error = MatDestroy(&mechanics->elastic_reference);
  const PetscErrorCode reference_error = VecDestroy(&mechanics->reference_load);
  const PetscErrorCode current_error = VecDestroy(&mechanics->checkpoint_current_state);
  const PetscErrorCode previous_error =
      VecDestroy(&mechanics->checkpoint_previous_state);
  if (elastic_error != PETSC_SUCCESS || reference_error != PETSC_SUCCESS ||
      current_error != PETSC_SUCCESS || previous_error != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (ss_deflation_destroy(&mechanics->deflation) != SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (ss_linear_recycle_destroy(&mechanics->krylov_recycle) != SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (ss_operator_destroy(&mechanics->operator_handle) != SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  free(mechanics->natural_operations);
  memset(mechanics, 0, sizeof(*mechanics));
  return status;
}
