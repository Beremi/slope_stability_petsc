/* Hydraulic boundary data, operator lifecycle, initialization and solve.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/core/sort.h"
#include "slope_internal/mesh/cell_index.h"
#include "solver_internal.h"

#include "slope_internal/fe/fe.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static SsStatus ss_solver_hydraulic_natural_operations(const SsSolverGraph *graph,
                                                       SsSolverHydraulic *hydraulic) {
  size_t count = 0U;
  size_t rule_index;
  size_t output_index = 0U;
  SsStatus status = SS_OK;
  if (graph == NULL || hydraulic == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (rule_index = 0U; rule_index < graph->model.boundary_count; ++rule_index) {
    const SsSolverBoundaryRule *rule = &graph->model.boundaries[rule_index];
    if (rule->condition_kind == SS_SOLVER_BOUNDARY_NATURAL &&
        strcmp(rule->field, "pressure") == 0) {
      ++count;
    }
  }
  if (count > SIZE_MAX / sizeof(*hydraulic->natural_operations) ||
      count > SIZE_MAX / sizeof(*hydraulic->natural_models)) {
    status = SS_ERROR_OVERFLOW;
  }
  if (status == SS_OK && count != 0U) {
    hydraulic->natural_operations =
        (SsNaturalOperatorSpec *)calloc(count, sizeof(*hydraulic->natural_operations));
    hydraulic->natural_models =
        (SsHydraulicFluxModel *)calloc(count, sizeof(*hydraulic->natural_models));
    if (hydraulic->natural_operations == NULL || hydraulic->natural_models == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  for (rule_index = 0U; rule_index < graph->model.boundary_count; ++rule_index) {
    const SsSolverBoundaryRule *rule = &graph->model.boundaries[rule_index];
    SsHydraulicFluxModel *model;
    if (rule->condition_kind != SS_SOLVER_BOUNDARY_NATURAL ||
        strcmp(rule->field, "pressure") != 0) {
      continue;
    }
    if (output_index >= count || rule->value_kind != SS_SOLVER_VALUE_PRESCRIBED_FLUX ||
        rule->value_count != 1U) {
      return SS_ERROR_INTEGRITY;
    }
    model = &hydraulic->natural_models[output_index];
    model->kind = SS_HYDRAULIC_FLUX_CONSTANT;
    model->dimension = graph->model.dimension;
    model->value = rule->values[0];
    (void)snprintf(model->identity, sizeof(model->identity), "%s",
                   "native-prescribed-flux-v1");
    hydraulic->natural_operations[output_index] = (SsNaturalOperatorSpec){
        rule->support_value, ss_hydraulic_flux_point_kernel, model, 0};
    ++output_index;
  }
  hydraulic->natural_operation_count = output_index;
  return output_index == count ? SS_OK : SS_ERROR_INTEGRITY;
}

static int ss_solver_hydraulic_cell_compare(const void *left_raw,
                                            const void *right_raw) {
  const SsSolverHydraulicCell *left = (const SsSolverHydraulicCell *)left_raw;
  const SsSolverHydraulicCell *right = (const SsSolverHydraulicCell *)right_raw;
  return left->stable_cell_id < right->stable_cell_id   ? -1
         : left->stable_cell_id > right->stable_cell_id ? 1
                                                        : 0;
}

const SsSolverHydraulicCell *
ss_solver_hydraulic_cell_find(const SsSolverHydraulic *hydraulic,
                              SsEntityId stable_cell_id) {
  if (hydraulic == NULL) {
    return NULL;
  }
  const size_t index = ss_mesh_cell_index_find(
      hydraulic->cells, hydraulic->cell_count, sizeof(*hydraulic->cells),
      offsetof(SsSolverHydraulicCell, stable_cell_id), stable_cell_id);
  return index < hydraulic->cell_count ? &hydraulic->cells[index] : NULL;
}

static const SsSolverMaterial *ss_solver_hydraulic_material(const SsSolverGraph *graph,
                                                            uint32_t material_value) {
  size_t material;
  if (graph == NULL) {
    return NULL;
  }
  for (material = 0U; material < graph->model.material_count; ++material) {
    const SsSolverMaterial *candidate = &graph->model.materials[material];
    if (candidate->kind == SS_SOLVER_MATERIAL_HYDRAULIC &&
        candidate->support_value == material_value) {
      return candidate;
    }
  }
  return NULL;
}

static SsStatus ss_solver_hydraulic_evaluate(void *opaque, Vec state, PetscReal factor,
                                             Vec residual, Mat tangent,
                                             PetscReal *functional) {
  SsSolverHydraulic *hydraulic = (SsSolverHydraulic *)opaque;
  const SsOperatorState operator_state = {
      .field = state,
      .reduction_factor = 1.0,
      .external_load_factor = 1.0,
      .cancelled = 0U,
  };
  SsErrorRecord error = {0};
  (void)factor;
  if (functional != NULL) {
    *functional = 0.0;
  }
  if (hydraulic == NULL || hydraulic->operator_handle == NULL || state == NULL ||
      residual == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return tangent == NULL ? ss_operator_residual(hydraulic->operator_handle,
                                                &operator_state, residual, &error)
                         : ss_operator_residual_jacobian(hydraulic->operator_handle,
                                                         &operator_state, residual,
                                                         tangent, &error);
}

static SsStatus ss_solver_hydraulic_admissible(void *opaque, Vec state,
                                               PetscReal factor, int *admissible) {
  PetscReal norm = NAN;
  (void)opaque;
  if (state == NULL || admissible == NULL || !isfinite((double)factor) ||
      VecNorm(state, NORM_INFINITY, &norm) != PETSC_SUCCESS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *admissible = isfinite((double)norm);
  return SS_OK;
}

SsStatus ss_solver_hydraulic_create(SsSolverGraph *graph, int requested_threads,
                                    SsSolverHydraulic *hydraulic,
                                    SsErrorRecord *error) {
  const SsFeDescriptor *coordinate_descriptor = NULL;
  SsCellPack *packs = NULL;
  size_t pack_count = 0U;
  size_t pack_index;
  double gamma_water = NAN;
  SsHydraulicOperatorModelSpec model_spec;
  Vec initial = NULL;
  SsOperatorState operator_state = {0};
  SsStatus status;
  if (graph == NULL || hydraulic == NULL || graph->hydraulic_plan == NULL ||
      graph->hydraulic_space == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(hydraulic, 0, sizeof(*hydraulic));
  hydraulic->graph = graph;
  status = ss_fe_descriptor_find(graph->model.dimension, graph->model.coordinate_order,
                                 &coordinate_descriptor);
  for (pack_index = 0U; status == SS_OK && pack_index < graph->model.material_count;
       ++pack_index) {
    const SsSolverMaterial *material = &graph->model.materials[pack_index];
    if (material->kind != SS_SOLVER_MATERIAL_HYDRAULIC) {
      continue;
    }
    if (!isfinite(material->water_unit_weight) || material->water_unit_weight <= 0.0 ||
        (isfinite(gamma_water) &&
         fabs(gamma_water - material->water_unit_weight) >
             1.0e-12 * fmax(gamma_water, material->water_unit_weight))) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    gamma_water = material->water_unit_weight;
  }
  if (status == SS_OK && !isfinite(gamma_water)) {
    status = SS_ERROR_NOT_FOUND;
  }
  if (status == SS_OK) {
    hydraulic->gamma_water = gamma_water;
  }
  if (status == SS_OK) {
    status = ss_field_space_pack_incident_cells(graph->hydraulic_space, NULL, 0U,
                                                &pack_count);
  }
  if (status == SS_OK) {
    packs = (SsCellPack *)calloc(pack_count == 0U ? 1U : pack_count, sizeof(*packs));
    hydraulic->cells = (SsSolverHydraulicCell *)calloc(
        pack_count == 0U ? 1U : pack_count, sizeof(*hydraulic->cells));
    hydraulic->coefficients = (SsHydraulicCellCoefficients *)calloc(
        pack_count == 0U ? 1U : pack_count, sizeof(*hydraulic->coefficients));
    if (packs == NULL || hydraulic->cells == NULL || hydraulic->coefficients == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  if (status == SS_OK) {
    status = ss_field_space_pack_incident_cells(graph->hydraulic_space, packs,
                                                pack_count, &pack_count);
  }
  for (pack_index = 0U; status == SS_OK && pack_index < pack_count; ++pack_index) {
    const SsCellPack *pack = &packs[pack_index];
    const SsSolverMaterial *material =
        ss_solver_hydraulic_material(graph, pack->material_value);
    double vertex_coordinates[(SS_FE_MAX_DIMENSION + 1U) * SS_FE_MAX_DIMENSION] = {0.0};
    double minimum_edge = NAN;
    if (material == NULL || !isfinite(material->conductivity) ||
        material->conductivity <= 0.0 || !isfinite(material->water_unit_weight) ||
        material->water_unit_weight <= 0.0 ||
        fabs(gamma_water - material->water_unit_weight) >
            1.0e-12 * fmax(gamma_water, material->water_unit_weight) ||
        ss_solver_cell_vertex_coordinates(pack, coordinate_descriptor,
                                          vertex_coordinates) != SS_OK) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    hydraulic->cells[pack_index].stable_cell_id = pack->stable_cell_id;
    hydraulic->cells[pack_index].conductivity = material->conductivity;
    status = ss_hydraulic_cell_epsilon(graph->model.dimension, vertex_coordinates,
                                       gamma_water, &minimum_edge,
                                       &hydraulic->cells[pack_index].epsilon);
  }
  free(packs);
  packs = NULL;
  if (status == SS_OK) {
    ss_sort(hydraulic->cells, pack_count, sizeof(*hydraulic->cells),
            ss_solver_hydraulic_cell_compare);
    for (pack_index = 1U; pack_index < pack_count; ++pack_index) {
      if (hydraulic->cells[pack_index - 1U].stable_cell_id ==
          hydraulic->cells[pack_index].stable_cell_id) {
        status = SS_ERROR_DUPLICATE;
        break;
      }
    }
  }
  for (pack_index = 0U; status == SS_OK && pack_index < pack_count; ++pack_index) {
    hydraulic->coefficients[pack_index] = (SsHydraulicCellCoefficients){
        hydraulic->cells[pack_index].stable_cell_id,
        hydraulic->cells[pack_index].epsilon,
        1U,
        &hydraulic->cells[pack_index].conductivity,
    };
  }
  hydraulic->cell_count = status == SS_OK ? pack_count : 0U;
  memset(&model_spec, 0, sizeof(model_spec));
  model_spec.dimension = graph->model.dimension;
  model_spec.upward_axis = 1U;
  model_spec.gamma_water = gamma_water;
  model_spec.cell_count = hydraulic->cell_count;
  model_spec.cells = hydraulic->coefficients;
  (void)snprintf(model_spec.model_identity, sizeof(model_spec.model_identity), "%s",
                 "steady-unconfined-native-v1");
  (void)snprintf(model_spec.pressure_unit, sizeof(model_spec.pressure_unit), "%s",
                 "kPa");
  (void)snprintf(model_spec.coordinate_unit, sizeof(model_spec.coordinate_unit), "%s",
                 "m");
  (void)snprintf(model_spec.conductivity_unit, sizeof(model_spec.conductivity_unit),
                 "%s", "m/s");
  if (status == SS_OK) {
    status = ss_hydraulic_operator_model_create(&model_spec, &hydraulic->model);
  }
  hydraulic->scalar_spec = (SsScalarOperatorSpec){
      ss_hydraulic_operator_point_kernel,
      hydraulic->model,
  };
  hydraulic->operator_spec = (SsOperatorSpec){
      .physics = SS_ASSEMBLY_SCALAR_FLUX,
      .determinism = ss_solver_assembly_determinism(&graph->policy),
      .memory_cap_bytes = 1024U * 1024U * 1024U,
      .openmp_work_threshold = SS_SOLVER_DEFAULT_OPENMP_WORK_THRESHOLD,
      .requested_threads = requested_threads > 0 ? requested_threads : 1,
      .scalar = &hydraulic->scalar_spec,
      .volume_quadrature_mode = graph->model.volume_quadrature_mode,
  };
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    status = ss_solver_hydraulic_natural_operations(graph, hydraulic);
  }
  hydraulic->operator_spec.natural_operations = hydraulic->natural_operations;
  hydraulic->operator_spec.natural_operation_count = hydraulic->natural_operation_count;
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    status = ss_operator_create(graph->hydraulic_plan, &hydraulic->operator_spec,
                                &hydraulic->operator_handle, error);
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    status = ss_assembly_plan_create_vector(graph->hydraulic_plan,
                                            &hydraulic->reference_load);
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    status = ss_assembly_plan_create_matrix(graph->hydraulic_plan,
                                            &hydraulic->elastic_reference);
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    status = ss_assembly_plan_create_vector(graph->hydraulic_plan, &initial);
  }
  if (status == SS_OK && (VecSet(hydraulic->reference_load, 1.0) != PETSC_SUCCESS ||
                          VecSet(initial, 0.0) != PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  operator_state.field = initial;
  operator_state.reduction_factor = 1.0;
  operator_state.external_load_factor = 1.0;
  if (status == SS_OK) {
    status = ss_hydraulic_operator_model_set_gravity_mode(
        hydraulic->model, SS_HYDRAULIC_GRAVITY_DISABLED);
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    status = ss_operator_jacobian(hydraulic->operator_handle, &operator_state,
                                  hydraulic->elastic_reference, error);
  }
  {
    const SsStatus restore_status = ss_hydraulic_operator_model_set_gravity_mode(
        hydraulic->model, SS_HYDRAULIC_GRAVITY_NONLINEAR);
    if (status == SS_OK && restore_status != SS_OK) {
      status = restore_status;
    }
  }
  if (VecDestroy(&initial) != PETSC_SUCCESS && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK && graph->policy.pmg_enabled != 0) {
    status = ss_solver_pmg_create(
        graph, graph->hydraulic_space, graph->model.hydraulic_order, 1U, 1,
        graph->hydraulic_constraints, graph->hydraulic_constraint_count, "native_hyd",
        NULL, NULL, &hydraulic->pmg, error);
  }
  /*
   * Mechanics deflation is deliberately not inherited here.  Its current
   * projector is qualified for the regularized mechanics tangent, not for the
   * nonsymmetric unconfined-seepage Jacobian.  Hydraulic FGMRES retains PMG,
   * but solves the unprojected system to its dedicated tight tolerance.
   */
  hydraulic->deflation = NULL;
  if (status == SS_OK) {
    status = ss_solver_linear_policy(graph, 1, &hydraulic->nonlinear.linear_policies);
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status != SS_OK) {
    (void)ss_solver_hydraulic_destroy(hydraulic);
    return status;
  }
  hydraulic->nonlinear.communicator = graph->communicator;
  hydraulic->nonlinear.reference_load = hydraulic->reference_load;
  hydraulic->nonlinear.elastic_reference = hydraulic->elastic_reference;
  hydraulic->nonlinear.operations.evaluate = ss_solver_hydraulic_evaluate;
  hydraulic->nonlinear.operations.admissibility = ss_solver_hydraulic_admissible;
  hydraulic->nonlinear.context = hydraulic;
  hydraulic->nonlinear.pmg = hydraulic->pmg;
  hydraulic->nonlinear.deflation = hydraulic->deflation;
  hydraulic->nonlinear.deflation_mode = hydraulic->deflation == NULL
                                            ? SS_NONLINEAR_DEFLATION_NONE
                                            : SS_NONLINEAR_DEFLATION_OWNED_TRANSACTION;
  hydraulic->nonlinear.deflation_promotion = SS_NONLINEAR_DEFLATION_PROMOTE_DIRECTIONS;
  hydraulic->nonlinear.linear_properties.finite_action = 1;
  hydraulic->nonlinear.linear_properties.nonlinear_hydraulic = 1;
  hydraulic->nonlinear.structure_generation = 1U;
  hydraulic->nonlinear.first_value_generation = 1U;
  hydraulic->nonlinear.transaction_generation = 1U;
  hydraulic->nonlinear.accepted_generation = 1U;
  hydraulic->nonlinear_policy.tolerance = (PetscReal)fmax(
      graph->policy.hydraulic_nonlinear_atol, graph->policy.hydraulic_nonlinear_rtol);
  hydraulic->nonlinear_policy.stopping_criterion = SS_NONLINEAR_STOP_RELATIVE_RESIDUAL;
  hydraulic->nonlinear_policy.stopping_tolerance =
      hydraulic->nonlinear_policy.tolerance;
  hydraulic->nonlinear_policy.maximum_iterations =
      graph->policy.hydraulic_nonlinear_maximum_iterations;
  hydraulic->nonlinear_policy.maximum_line_search_trials = 24;
  hydraulic->nonlinear_policy.minimum_alpha = 1.0 / 1048576.0;
  hydraulic->nonlinear_policy.regularization_minimum = 1.0e-10;
  hydraulic->nonlinear_policy.singular_relative_tolerance = 1.0e-12;
  hydraulic->nonlinear_policy.relaxed_factor = 10.0;
  hydraulic->nonlinear_policy.allow_relaxed = 0;
  hydraulic->nonlinear_policy.line_method = SS_NONLINEAR_LINE_RESIDUAL_MONOTONE;
  status = ss_status_allreduce_max(
      graph->communicator, ss_nonlinear_policy_validate(&hydraulic->nonlinear_policy));
  if (status == SS_OK) {
    status = ss_nonlinear_workspace_create(graph->communicator,
                                           &hydraulic->nonlinear.workspace);
  }
  if (status != SS_OK) {
    (void)ss_solver_hydraulic_destroy(hydraulic);
  }
  return status;
}

SsStatus ss_solver_hydraulic_destroy(SsSolverHydraulic *hydraulic) {
  SsStatus status = SS_OK;
  if (hydraulic == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (ss_nonlinear_workspace_destroy(&hydraulic->nonlinear.workspace) != SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  const PetscErrorCode elastic_error = MatDestroy(&hydraulic->elastic_reference);
  const PetscErrorCode reference_error = VecDestroy(&hydraulic->reference_load);
  if (elastic_error != PETSC_SUCCESS || reference_error != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (ss_pmg_destroy(&hydraulic->pmg) != SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (ss_deflation_destroy(&hydraulic->deflation) != SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  const SsStatus operator_status = ss_operator_destroy(&hydraulic->operator_handle);
  const SsStatus model_status = ss_hydraulic_operator_model_destroy(&hydraulic->model);
  if (operator_status != SS_OK || model_status != SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  free(hydraulic->natural_models);
  free(hydraulic->natural_operations);
  free(hydraulic->coefficients);
  free(hydraulic->cells);
  memset(hydraulic, 0, sizeof(*hydraulic));
  return status;
}

static SsStatus ss_solver_hydraulic_boundary_value(const SsSolverGraph *graph,
                                                   const double *physical,
                                                   double *value) {
  double maximum = 0.0;
  int matched = 0;
  size_t rule_index;
  if (graph == NULL || physical == NULL || value == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (rule_index = 0U; rule_index < graph->model.boundary_count; ++rule_index) {
    const SsSolverBoundaryRule *rule = &graph->model.boundaries[rule_index];
    double candidate = 0.0;
    if (rule->condition_kind != SS_SOLVER_BOUNDARY_ESSENTIAL ||
        strcmp(rule->field, "pressure") != 0) {
      continue;
    }
    if (rule->value_kind == SS_SOLVER_VALUE_CONSTANT) {
      candidate = rule->value_count == 0U ? 0.0 : rule->values[0];
    } else if (rule->value_kind == SS_SOLVER_VALUE_AFFINE_COORDINATE) {
      size_t axis;
      candidate = rule->values[0];
      for (axis = 0U; axis < graph->model.dimension; ++axis) {
        candidate += rule->gradient[axis] * physical[axis];
      }
    } else {
      SsHydraulicHeadModel head;
      memset(&head, 0, sizeof(head));
      head.dimension = graph->model.dimension;
      head.upward_axis = 1U;
      head.piecewise_axis = 0U;
      head.gamma_water = rule->water_unit_weight;
      if (rule->value_kind == SS_SOLVER_VALUE_HYDROSTATIC) {
        head.kind = SS_HYDRAULIC_HEAD_CONSTANT_LEVEL;
        head.level = rule->reference_level;
      } else if (rule->value_kind == SS_SOLVER_VALUE_PIECEWISE_LEVEL &&
                 rule->piecewise_count <= SS_HYDRAULIC_HEAD_POINT_CAPACITY) {
        head.kind = SS_HYDRAULIC_HEAD_PIECEWISE_LINEAR_LEVEL;
        head.piecewise_point_count = rule->piecewise_count;
        memcpy(head.piecewise_coordinates, rule->piecewise_coordinates,
               rule->piecewise_count * sizeof(*head.piecewise_coordinates));
        memcpy(head.piecewise_levels, rule->piecewise_levels,
               rule->piecewise_count * sizeof(*head.piecewise_levels));
      } else {
        return SS_ERROR_SCHEMA;
      }
      (void)snprintf(head.identity, sizeof(head.identity), "%s",
                     "native-hydraulic-initializer");
      if (ss_hydraulic_head_evaluate(&head, physical, &candidate) != SS_OK) {
        return SS_ERROR_INTEGRITY;
      }
      candidate += rule->reference_pressure;
    }
    if (!isfinite(candidate)) {
      return SS_ERROR_INTEGRITY;
    }
    if (matched == 0 || candidate > maximum) {
      maximum = candidate;
    }
    matched = 1;
  }
  *value = matched != 0 ? maximum : 0.0;
  return SS_OK;
}

static SsStatus ss_solver_hydraulic_boundary_seed(const SsSolverGraph *graph,
                                                  Vec initial) {
  SsCellPack *packs = NULL;
  size_t pack_count = 0U;
  size_t pack_index;
  SsStatus status;
  if (graph == NULL || graph->hydraulic_space == NULL || initial == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status =
      ss_field_space_pack_owned_cells(graph->hydraulic_space, NULL, 0U, &pack_count);
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK && pack_count > SIZE_MAX / sizeof(*packs)) {
    status = SS_ERROR_OVERFLOW;
  }
  if (status == SS_OK) {
    packs = (SsCellPack *)calloc(pack_count == 0U ? 1U : pack_count, sizeof(*packs));
    if (packs == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    status = ss_field_space_pack_owned_cells(graph->hydraulic_space, packs, pack_count,
                                             &pack_count);
  }
  if (status == SS_OK && VecSet(initial, 0.0) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  for (pack_index = 0U; status == SS_OK && pack_index < pack_count; ++pack_index) {
    const SsCellPack *pack = &packs[pack_index];
    size_t slot;
    for (slot = 0U; status == SS_OK && slot < pack->closure_count; ++slot) {
      const size_t node = slot;
      double initial_value = 0.0;
      if (pack->components != 1U || node >= pack->field_node_count) {
        status = SS_ERROR_INTEGRITY;
        break;
      }
      if (pack->constrained[slot] != 0U || pack->owned[slot] == 0U) {
        continue;
      }
      if (pack->global_indices[slot] < 0 ||
          ss_solver_hydraulic_boundary_value(graph,
                                             pack->field_node_coordinates +
                                                 node * (size_t)graph->model.dimension,
                                             &initial_value) != SS_OK ||
          VecSetValue(initial, pack->global_indices[slot], (PetscScalar)initial_value,
                      INSERT_VALUES) != PETSC_SUCCESS) {
        status = SS_ERROR_INTEGRITY;
      }
    }
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK && (VecAssemblyBegin(initial) != PETSC_SUCCESS ||
                          VecAssemblyEnd(initial) != PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  free(packs);
  return ss_status_allreduce_max(graph->communicator, status);
}

static SsStatus ss_solver_hydraulic_initialize(const SsSolverGraph *graph,
                                               SsSolverHydraulic *hydraulic,
                                               Vec initial, Vec alternate,
                                               int *alternate_valid,
                                               SsErrorRecord *error) {
  Vec source_linear = NULL;
  Vec residual = NULL;
  SsLinearSolver *linear = NULL;
  SsLinearProblem problem = {0};
  SsLinearSetupRecord setup = {0};
  SsLinearSolveRecord solve = {0};
  PetscReal boundary_residual_norm = NAN;
  PetscReal source_linear_residual_norm = NAN;
  int source_linear_candidate_valid = 0;
  SsStatus status = SS_OK;
  if (graph == NULL || hydraulic == NULL || initial == NULL || alternate == NULL ||
      alternate_valid == NULL || error == NULL || hydraulic->operator_handle == NULL ||
      hydraulic->elastic_reference == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *alternate_valid = 0;
  hydraulic->primary_outcome_valid = 0;
  if (VecDuplicate(initial, &source_linear) != PETSC_SUCCESS ||
      VecDuplicate(initial, &residual) != PETSC_SUCCESS ||
      VecSet(source_linear, 0.0) != PETSC_SUCCESS) {
    status = SS_ERROR_OUT_OF_MEMORY;
  }
  if (status == SS_OK) {
    status = ss_solver_hydraulic_boundary_seed(graph, initial);
  }
  if (status == SS_OK) {
    status =
        ss_solver_hydraulic_evaluate(hydraulic, initial, 1.0, residual, NULL, NULL);
  }
  if (status == SS_OK &&
      VecNorm(residual, NORM_2, &boundary_residual_norm) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    const SsHydraulicGravityMode initialization_gravity =
        graph->model.dimension == 2U ? SS_HYDRAULIC_GRAVITY_DISABLED
                                     : SS_HYDRAULIC_GRAVITY_SATURATED;
    status = ss_hydraulic_operator_model_set_gravity_mode(hydraulic->model,
                                                          initialization_gravity);
    if (status == SS_OK) {
      status = ss_solver_hydraulic_evaluate(hydraulic, source_linear, 1.0, residual,
                                            NULL, NULL);
    }
    {
      const SsStatus restore_status = ss_hydraulic_operator_model_set_gravity_mode(
          hydraulic->model, SS_HYDRAULIC_GRAVITY_NONLINEAR);
      if (status == SS_OK && restore_status != SS_OK) {
        status = restore_status;
      }
    }
  }
  if (status == SS_OK && VecScale(residual, -1.0) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    problem.operator_handle = hydraulic->elastic_reference;
    problem.preconditioner_operator_handle =
        hydraulic->pmg == NULL ? NULL : hydraulic->elastic_reference;
    problem.properties = hydraulic->nonlinear.linear_properties;
    problem.policy_chain = hydraulic->nonlinear.linear_policies;
    problem.pmg = hydraulic->pmg;
    problem.structure_generation = hydraulic->nonlinear.structure_generation;
    problem.value_generation = hydraulic->nonlinear.first_value_generation;
    status = ss_linear_create(graph->communicator, &linear);
  }
  if (status == SS_OK) {
    status = ss_linear_setup(linear, &problem, &setup);
  }
  if (status == SS_OK) {
    status = ss_linear_solve(linear, residual, source_linear, &solve);
    if (status == SS_OK) {
      source_linear_candidate_valid = solve.status == SS_LINEAR_SOLVE_CONVERGED;
    }
  }
  if (status == SS_OK && source_linear_candidate_valid != 0) {
    status = ss_solver_hydraulic_evaluate(hydraulic, source_linear, 1.0, residual, NULL,
                                          NULL);
  }
  if (status == SS_OK && source_linear_candidate_valid != 0 &&
      VecNorm(residual, NORM_2, &source_linear_residual_norm) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK && source_linear_candidate_valid != 0 &&
      !isfinite((double)source_linear_residual_norm)) {
    source_linear_candidate_valid = 0;
  }
  if (status == SS_OK && !isfinite((double)boundary_residual_norm)) {
    status = SS_ERROR_NUMERICAL;
  }
  if (status == SS_OK) {
    status = ss_hydraulic_initialization_select(
        (double)boundary_residual_norm, (double)source_linear_residual_norm,
        source_linear_candidate_valid, &hydraulic->initialization);
  }
  if (status == SS_OK) {
    if (hydraulic->initialization.choice == SS_HYDRAULIC_INITIAL_SATURATED_LINEAR) {
      if (VecCopy(initial, alternate) != PETSC_SUCCESS ||
          VecCopy(source_linear, initial) != PETSC_SUCCESS) {
        status = SS_ERROR_INTERNAL;
      } else {
        *alternate_valid = 1;
      }
    } else if (source_linear_candidate_valid != 0) {
      if (VecCopy(source_linear, alternate) != PETSC_SUCCESS) {
        status = SS_ERROR_INTERNAL;
      } else {
        *alternate_valid = 1;
      }
    }
  }
  if (status == SS_OK) {
    /*
     * The initializer has consumed PMG value generation one.  The nonlinear
     * tangent is a different matrix object and must advance the generation so
     * PMG cannot reuse the saturated operator by identity alone.
     */
    hydraulic->nonlinear.first_value_generation += 1U;
  }
  if (ss_linear_destroy(&linear) != SS_OK && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (VecDestroy(&residual) != PETSC_SUCCESS ||
      VecDestroy(&source_linear) != PETSC_SUCCESS) {
    if (status == SS_OK) {
      status = SS_ERROR_INTERNAL;
    }
  }
  if (status != SS_OK && error->valid == 0) {
    char message[SS_ERROR_MESSAGE_CAPACITY];
    const int written =
        snprintf(message, sizeof(message),
                 "hydraulic initialization failed: status=%d setup=%zu solve=%d "
                 "reason=%d iterations=%lld true_residual=%.8g boundary_residual=%.8g "
                 "saturated_residual=%.8g",
                 (int)status, setup.attempt_count, (int)solve.status, (int)solve.reason,
                 (long long)solve.iterations, (double)solve.true_residual_norm,
                 (double)boundary_residual_norm, (double)source_linear_residual_norm);
    ss_solver_error(error, status, "HYDRAULIC_INITIALIZATION",
                    written >= 0 && (size_t)written < sizeof(message)
                        ? message
                        : "hydraulic initialization failed",
                    3U);
  }
  return ss_status_allreduce_max(graph->communicator, status);
}

static int ss_solver_nonlinear_outcome_accepted(SsNonlinearOutcome outcome) {
  return outcome == SS_NONLINEAR_CONVERGED || outcome == SS_NONLINEAR_ACCEPTED_RELAXED;
}

SsStatus ss_solver_hydraulic_solve(const SsSolverGraph *graph,
                                   SsSolverHydraulic *hydraulic, Vec initial,
                                   Vec solution, SsNonlinearSolveRecord *record,
                                   SsErrorRecord *error) {
  Vec alternate = NULL;
  int alternate_valid = 0;
  SsStatus status;
  if (graph == NULL || hydraulic == NULL || initial == NULL || solution == NULL ||
      record == NULL || error == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_status_allreduce_max(graph->communicator,
                                   VecDuplicate(initial, &alternate) == PETSC_SUCCESS
                                       ? SS_OK
                                       : SS_ERROR_OUT_OF_MEMORY);
  if (status != SS_OK) {
    (void)VecDestroy(&alternate);
    return status;
  }
  status = ss_solver_hydraulic_initialize(graph, hydraulic, initial, alternate,
                                          &alternate_valid, error);
  if (status == SS_OK) {
    status =
        ss_nonlinear_fixed_solve(&hydraulic->nonlinear, &hydraulic->nonlinear_policy,
                                 1.0, initial, solution, record);
  }
  if (status == SS_OK && alternate_valid != 0 &&
      !ss_solver_nonlinear_outcome_accepted(record->outcome) &&
      record->outcome != SS_NONLINEAR_CANCELLED) {
    const uint64_t generation_advance = record->regularized_operator_generation + 1U;
    hydraulic->primary_outcome = record->outcome;
    hydraulic->primary_outcome_valid = 1;
    if (generation_advance == 0U ||
        hydraulic->nonlinear.first_value_generation > UINT64_MAX - generation_advance ||
        hydraulic->nonlinear.transaction_generation == UINT64_MAX) {
      status = SS_ERROR_OVERFLOW;
    }
    if (status == SS_OK) {
      hydraulic->nonlinear.first_value_generation += generation_advance;
      ++hydraulic->nonlinear.transaction_generation;
      status = ss_hydraulic_initialization_fallback(&hydraulic->initialization);
    }
    if (status == SS_OK) {
      status =
          ss_nonlinear_fixed_solve(&hydraulic->nonlinear, &hydraulic->nonlinear_policy,
                                   1.0, alternate, solution, record);
    }
  }
  if (VecDestroy(&alternate) != PETSC_SUCCESS && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  return ss_status_allreduce_max(graph->communicator, status);
}
