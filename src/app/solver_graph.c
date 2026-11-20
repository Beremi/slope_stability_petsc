/* Construct the solver graph and adapt material loads to its regions.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/core/sort.h"
#include "slope_internal/core/util.h"
#include "solver_input_internal.h"

#include "slope_internal/diagnostics/diagnostics.h"

#include <petsclog.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static PetscLogEvent ss_solver_input_parse_event = -1;
static PetscLogEvent ss_solver_input_mesh_load_event = -1;
static PetscLogEvent ss_solver_input_mesh_create_event = -1;
static PetscLogEvent ss_solver_input_constraints_event = -1;
static PetscLogEvent ss_solver_input_field_preliminary_event = -1;
static PetscLogEvent ss_solver_input_constraint_values_event = -1;
static PetscLogEvent ss_solver_input_field_create_event = -1;
static PetscLogEvent ss_solver_input_assembly_plan_event = -1;

static void ss_solver_input_profile_events_register(void) {
  if (ss_solver_input_parse_event < 0) {
    (void)PetscLogEventRegister("SlopeInputParse", 0, &ss_solver_input_parse_event);
    (void)PetscLogEventRegister("SlopeMeshLoad", 0, &ss_solver_input_mesh_load_event);
    (void)PetscLogEventRegister("SlopeMeshCreate", 0,
                                &ss_solver_input_mesh_create_event);
    (void)PetscLogEventRegister("SlopeConstraints", 0,
                                &ss_solver_input_constraints_event);
    (void)PetscLogEventRegister("SlopeFieldPrelim", 0,
                                &ss_solver_input_field_preliminary_event);
    (void)PetscLogEventRegister("SlopeConstraintVals", 0,
                                &ss_solver_input_constraint_values_event);
    (void)PetscLogEventRegister("SlopeFieldCreate", 0,
                                &ss_solver_input_field_create_event);
    (void)PetscLogEventRegister("SlopeAssemblyPlan", 0,
                                &ss_solver_input_assembly_plan_event);
  }
}

static const char ss_solver_provenance[] =
    "sha256:4a35f3420cdcd04efda216a043f33ae3d85712f30b2d8da31e64080ba7db6f8c";

SsStatus ss_solver_prescribed_mechanics_force(const void *context, uint32_t dimension,
                                              SsEntityId stable_cell_id,
                                              uint32_t quadrature_index,
                                              const double *physical,
                                              double *body_force) {
  const SsSolverProblemModel *model = context;
  size_t index;
  (void)quadrature_index;
  if (model == NULL || body_force == NULL || dimension != model->dimension) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (index = 0U; index < model->material_count; ++index) {
    const SsSolverMaterial *material = &model->materials[index];
    size_t left = 0U, right = material->entity_count;
    if (material->kind != SS_SOLVER_MATERIAL_MECHANICS)
      continue;
    while (left < right) {
      size_t middle = left + (right - left) / 2U;
      if (material->entity_ids[middle] < stable_cell_id)
        left = middle + 1U;
      else
        right = middle;
    }
    if (left < material->entity_count && material->entity_ids[left] == stable_cell_id) {
      double weight;
      SsStatus status = ss_solver_saturation_weight(
          &model->prescribed_saturation, dimension, physical,
          material->unsaturated_unit_weight, material->saturated_unit_weight, &weight);
      if (status != SS_OK)
        return status;
      for (uint32_t component = 0U; component < dimension; ++component)
        body_force[component] = 0.0;
      /* The region force already contains -unsaturated_unit_weight. */
      body_force[1] = material->unsaturated_unit_weight - weight;
      return SS_OK;
    }
  }
  return SS_ERROR_SCHEMA;
}

void ss_solver_error(SsErrorRecord *error, SsStatus status, const char *code,
                     const char *message, uint32_t component) {
  ss_error_set(error, status, (SsErrorKey){component, 4U, 0U, 0U, 0U, 0U}, code,
               message);
}

static void ss_solver_model_destroy(SsSolverProblemModel *model) {
  size_t index;
  if (model == NULL) {
    return;
  }
  for (index = 0U; index < model->material_count; ++index) {
    free(model->materials[index].entity_ids);
  }
  for (index = 0U; index < model->boundary_count; ++index) {
    free(model->boundaries[index].entity_ids);
  }
  free(model->materials);
  free(model->boundaries);
  memset(model, 0, sizeof(*model));
}

static SsStatus ss_solver_create_materials(SsSolverGraph *graph) {
  size_t mechanics_count = 0U;
  size_t material_index;
  size_t region_index = 0U;
  for (material_index = 0U; material_index < graph->model.material_count;
       ++material_index) {
    if (graph->model.materials[material_index].kind == SS_SOLVER_MATERIAL_MECHANICS) {
      ++mechanics_count;
    }
  }
  graph->material_parameters =
      (SsMcParameters **)calloc(mechanics_count == 0U ? 1U : mechanics_count,
                                sizeof(*graph->material_parameters));
  graph->mechanics_regions = (SsMechanicsMaterialRegion *)calloc(
      mechanics_count == 0U ? 1U : mechanics_count, sizeof(*graph->mechanics_regions));
  if (graph->material_parameters == NULL || graph->mechanics_regions == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  graph->mechanics_region_count = mechanics_count;
  for (material_index = 0U; material_index < graph->model.material_count;
       ++material_index) {
    const SsSolverMaterial *material = &graph->model.materials[material_index];
    SsMcParameterInput input;
    SsMcError error = {0};
    if (material->kind != SS_SOLVER_MATERIAL_MECHANICS) {
      continue;
    }
    input = (SsMcParameterInput){
        material->young_modulus,
        material->poisson_ratio,
        material->cohesion,
        material->friction_angle,
        material->dilatancy_angle,
        material->davis,
        (uint64_t)material->support_value,
        ss_solver_provenance,
    };
    if (ss_mc_parameters_create(&input, &graph->material_parameters[region_index],
                                &error) != SS_OK) {
      return error.status == SS_OK ? SS_ERROR_INTEGRITY : error.status;
    }
    graph->mechanics_regions[region_index].material_value = material->support_value;
    graph->mechanics_regions[region_index].parameters =
        graph->material_parameters[region_index];
    graph->mechanics_regions[region_index].has_body_force = 1;
    if (graph->model.coupling_enabled == 0 &&
        ss_solver_self_weight(graph->model.dimension, material->unsaturated_unit_weight,
                              graph->mechanics_regions[region_index].body_force) !=
            SS_OK) {
      return SS_ERROR_INTEGRITY;
    }
    ++region_index;
  }
  return SS_OK;
}

/* Raw memberships have served their purpose once the DM and essential labels
 * have been built. The analytic saturation worker needs only resident cells,
 * obtained later from the mechanics plans in stable-ID order. */
static SsStatus ss_solver_install_essential_supports(SsSolverGraph *graph) {
  SsStatus status = SS_OK;
  for (size_t i = 0U; i < graph->model.material_count; ++i) {
    free(graph->model.materials[i].entity_ids);
    graph->model.materials[i].entity_ids = NULL;
    graph->model.materials[i].entity_count = 0U;
  }
  for (size_t i = 0U; status == SS_OK && i < graph->model.boundary_count; ++i) {
    SsSolverBoundaryRule *rule = &graph->model.boundaries[i];
    if (rule->condition_kind == SS_SOLVER_BOUNDARY_ESSENTIAL) {
      status = ss_mesh_essential_support_set(graph->mesh, rule->support_value,
                                             rule->entity_ids, rule->entity_count);
    }
    free(rule->entity_ids);
    rule->entity_ids = NULL;
    rule->entity_count = 0U;
  }
  return status;
}

static int ss_solver_compare_cell_id(const void *left, const void *right) {
  const SsEntityId a = *(const SsEntityId *)left;
  const SsEntityId b = *(const SsEntityId *)right;
  return (a > b) - (a < b);
}

static SsStatus ss_solver_prescribed_membership(SsSolverGraph *graph) {
  if (graph->model.prescribed_saturation.count == 0U) {
    return SS_OK;
  }
  const SsCellPlan *plans = NULL;
  size_t count = 0U;
  SsStatus status =
      ss_field_space_incident_cell_plan_view(graph->mechanics_space, &plans, &count);
  for (size_t i = 0U; status == SS_OK && i < graph->model.material_count; ++i) {
    SsSolverMaterial *material = &graph->model.materials[i];
    size_t capacity = 0U;
    if (material->kind != SS_SOLVER_MATERIAL_MECHANICS) {
      continue;
    }
    for (size_t j = 0U; j < count; ++j) {
      capacity += plans[j].material_value == material->support_value;
    }
    if (capacity == 0U) {
      continue;
    }
    material->entity_ids = calloc(capacity, sizeof(*material->entity_ids));
    if (material->entity_ids == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
      break;
    }
    for (size_t j = 0U; j < count; ++j) {
      if (plans[j].material_value == material->support_value) {
        material->entity_ids[material->entity_count++] = plans[j].stable_cell_id;
      }
    }
    ss_sort(material->entity_ids, material->entity_count, sizeof(*material->entity_ids),
            ss_solver_compare_cell_id);
  }
  return status;
}

static SsStatus ss_solver_graph_field(SsSolverGraph *graph, const char *field,
                                      uint32_t order, uint32_t components,
                                      SsSolverConstraintValueEvaluator evaluator,
                                      SsStrongConstraint **constraints,
                                      size_t *constraint_count, SsFieldSpace **space) {
  SsFieldSummary summary = {0};
  (void)PetscLogEventBegin(ss_solver_input_constraints_event, 0, 0, 0, 0);
  SsStatus status = ss_solver_build_constraints_for_field(
      graph, field, order, components, constraints, constraint_count);
  (void)PetscLogEventEnd(ss_solver_input_constraints_event, 0, 0, 0, 0);
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  const SsFieldSpec specification = {order, components, *constraint_count,
                                     *constraints};
  (void)PetscLogEventBegin(ss_solver_input_field_preliminary_event, 0, 0, 0, 0);
  status = ss_mesh_field_space_create_local(graph->mesh, &specification, space);
  (void)PetscLogEventEnd(ss_solver_input_field_preliminary_event, 0, 0, 0, 0);
  if (status == SS_OK) {
    status = ss_field_space_summary(*space, &summary);
  }
  if (status == SS_OK && summary.constrained_dof_count != 0U) {
    (void)PetscLogEventBegin(ss_solver_input_constraint_values_event, 0, 0, 0, 0);
    status = ss_solver_refine_constraints(graph, *space, *constraints,
                                          *constraint_count, components, evaluator);
    if (ss_field_space_destroy(space) != SS_OK && status == SS_OK) {
      status = SS_ERROR_INTERNAL;
    }
    (void)PetscLogEventEnd(ss_solver_input_constraint_values_event, 0, 0, 0, 0);
    status = ss_status_allreduce_max(graph->communicator, status);
    if (status == SS_OK) {
      (void)PetscLogEventBegin(ss_solver_input_field_create_event, 0, 0, 0, 0);
      status = ss_mesh_field_space_create_local(graph->mesh, &specification, space);
      (void)PetscLogEventEnd(ss_solver_input_field_create_event, 0, 0, 0, 0);
    }
  }
  return status;
}

SsStatus ss_solver_graph_create(const SsSolverRequest *request, SsSolverGraph *graph,
                                SsErrorRecord *error) {
  char payload_path[SS_RESULT_PATH_CAPACITY];
  SsMeshError mesh_error = {0};
  SsMeshSummary mesh_summary = {0};
  const char *invalid_setting = NULL;
  SsStatus status;
  int written;
  const char *stage = "request";
  if (request == NULL || graph == NULL || request->problem_directory == NULL ||
      request->resolved_run_path == NULL || request->config == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(graph, 0, sizeof(*graph));
  graph->communicator = request->communicator;
  ss_solver_input_profile_events_register();
  if (MPI_Comm_rank(graph->communicator, &graph->rank) != MPI_SUCCESS ||
      MPI_Comm_size(graph->communicator, &graph->size) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  (void)ss_memory_view(graph->communicator, "graph_begin");
  /* Release free allocator pages from preflight before populating the mesh. */
  (void)ss_memory_trim();
  (void)ss_memory_view(graph->communicator, "graph_after_input_trim");
  written = snprintf(payload_path, sizeof(payload_path), "%s/%s",
                     request->problem_directory, request->config->problem.payload_path);
  if (written < 0 || (size_t)written >= sizeof(payload_path)) {
    status = SS_ERROR_LIMIT;
    goto failure;
  }
  stage = "problem_parse";
  (void)PetscLogEventBegin(ss_solver_input_parse_event, 0, 0, 0, 0);
  status = ss_solver_parse_problem(request->config->compiled_manifest, graph->rank,
                                   graph->size, &graph->model);
  if (status == SS_OK) {
    stage = "run_parse";
    graph->policy.settings = request->config->run.settings;
    graph->policy.maximum_accepted_states =
        (PetscInt)request->config->run.maximum_accepted_states;
    status = ss_solver_parse_run(request->resolved_run_path, &graph->policy);
  }
  if (status == SS_OK) {
    stage = "solver_options";
    status = ss_solver_expert_options_validate(&invalid_setting);
  }
  if (status == SS_OK) {
    status = ss_solver_backend_settings(&graph->policy);
  }
  if (status == SS_OK) {
    const uint64_t unavailable = graph->policy.requested_output_mask &
                                 ~ss_solver_available_output_quantities(&graph->model);
    const uint64_t missing =
        graph->model.required_output_mask & ~graph->policy.requested_output_mask;
    stage = "output_contract";
    if (unavailable != 0U || missing != 0U) {
      status = SS_ERROR_SCHEMA;
    }
  }
  (void)PetscLogEventEnd(ss_solver_input_parse_event, 0, 0, 0, 0);
  (void)ss_memory_view(graph->communicator, "graph_after_parse");
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    stage = "mesh_load";
    (void)PetscLogEventBegin(ss_solver_input_mesh_load_event, 0, 0, 0, 0);
    status =
        ss_io_require_input_identity(payload_path, &request->config->payload_identity);
    status = ss_status_allreduce_max(graph->communicator, status);
    if (status == SS_OK) {
      status = ss_solver_load_mesh(graph->communicator, payload_path, &graph->model,
                                   &graph->raw);
    }
    if (status == SS_OK) {
      status = ss_io_require_input_identity(payload_path,
                                            &request->config->payload_identity);
    }
    (void)PetscLogEventEnd(ss_solver_input_mesh_load_event, 0, 0, 0, 0);
    (void)ss_memory_view(graph->communicator, "graph_after_mesh_load");
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status != SS_OK) {
    goto failure;
  }
  graph->raw.input.shared_point_balance =
      (SsMeshSharedPointBalance)graph->policy.settings.pmg_shared_point_balance;
  stage = "mesh_create";
  (void)PetscLogEventBegin(ss_solver_input_mesh_create_event, 0, 0, 0, 0);
  status =
      ss_mesh_create(graph->communicator, &graph->raw.input, &graph->mesh, &mesh_error);
  if (status == SS_OK) {
    status = ss_mesh_summary(graph->mesh, &mesh_summary);
  }
  if (status == SS_OK) {
    written = snprintf(graph->geometry_id, sizeof(graph->geometry_id), "sha256:%s",
                       mesh_summary.coordinate_digest);
    if (written < 0 || (size_t)written >= sizeof(graph->geometry_id)) {
      status = SS_ERROR_LIMIT;
    }
  }
  (void)PetscLogEventEnd(ss_solver_input_mesh_create_event, 0, 0, 0, 0);
  (void)ss_memory_view(graph->communicator, "graph_after_mesh_create");
  ss_solver_raw_release_payload(&graph->raw);
  (void)ss_memory_trim();
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    stage = "essential_supports";
    status = ss_solver_install_essential_supports(graph);
  }
  if (status == SS_OK) {
    stage = "materials";
    status = ss_solver_create_materials(graph);
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  (void)ss_memory_view(graph->communicator, "graph_after_constraints_materials");
  if (status == SS_OK && graph->model.mechanics_enabled != 0) {
    stage = "mechanics_space";
    status = ss_solver_graph_field(
        graph, "displacement", graph->model.mechanics_order, graph->model.dimension,
        ss_solver_displacement_boundary_value, &graph->mechanics_constraints,
        &graph->mechanics_constraint_count, &graph->mechanics_space);
    if (status == SS_OK) {
      stage = "mechanics_linear_admission";
      status = ss_solver_admit_linear_vectors(graph, 0, error);
    }
    if (status == SS_OK) {
      stage = "prescribed_saturation_membership";
      status = ss_solver_prescribed_membership(graph);
    }
    status = ss_status_allreduce_max(graph->communicator, status);
    if (status == SS_OK) {
      stage = "mechanics_plan";
      (void)PetscLogEventBegin(ss_solver_input_assembly_plan_event, 0, 0, 0, 0);
      status =
          graph->policy.settings.pmg_mechanics_matrix_free == PETSC_TRUE &&
                  graph->model.mechanics_order > 1U
              ? ss_assembly_plan_create_matrix_free(graph->communicator,
                                                    graph->mechanics_space,
                                                    &graph->mechanics_plan, error)
              : ss_assembly_plan_create(graph->communicator, graph->mechanics_space,
                                        &graph->mechanics_plan, error);
      (void)PetscLogEventEnd(ss_solver_input_assembly_plan_event, 0, 0, 0, 0);
    }
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK && graph->model.hydraulic_enabled != 0) {
    stage = "hydraulic_space";
    status = ss_solver_graph_field(
        graph, "pressure", graph->model.hydraulic_order, 1U,
        ss_solver_pressure_constraint_value, &graph->hydraulic_constraints,
        &graph->hydraulic_constraint_count, &graph->hydraulic_space);
    if (status == SS_OK) {
      stage = "hydraulic_linear_admission";
      status = ss_solver_admit_linear_vectors(graph, 1, error);
    }
    status = ss_status_allreduce_max(graph->communicator, status);
    if (status == SS_OK) {
      stage = "hydraulic_plan";
      (void)PetscLogEventBegin(ss_solver_input_assembly_plan_event, 0, 0, 0, 0);
      status = ss_assembly_plan_create(graph->communicator, graph->hydraulic_space,
                                       &graph->hydraulic_plan, error);
      (void)PetscLogEventEnd(ss_solver_input_assembly_plan_event, 0, 0, 0, 0);
    }
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status != SS_OK) {
    goto failure;
  }
  (void)ss_memory_view(graph->communicator, "graph_complete");
  (void)ss_memory_trim();
  (void)ss_memory_view(graph->communicator, "graph_after_memory_trim");
  return SS_OK;

failure:
  if (error == NULL || error->valid == 0) {
    char message[SS_ERROR_MESSAGE_CAPACITY];
    written = snprintf(message, sizeof(message),
                       "native numerical graph creation failed during %s",
                       invalid_setting == NULL ? stage : invalid_setting);
    ss_solver_error(error, status, "SOLVER_GRAPH_CREATE",
                    written >= 0 && (size_t)written < sizeof(message)
                        ? message
                        : "native numerical graph creation failed",
                    3U);
  }
  ss_solver_graph_destroy(graph);
  return status;
}

void ss_solver_graph_destroy(SsSolverGraph *graph) {
  size_t index;
  if (graph == NULL) {
    return;
  }
  (void)ss_assembly_plan_destroy(&graph->hydraulic_plan);
  (void)ss_assembly_plan_destroy(&graph->mechanics_plan);
  (void)ss_field_space_destroy(&graph->hydraulic_space);
  (void)ss_field_space_destroy(&graph->mechanics_space);
  (void)ss_mesh_destroy(&graph->mesh);
  for (index = 0U; index < graph->mechanics_region_count; ++index) {
    (void)ss_mc_parameters_destroy(&graph->material_parameters[index]);
  }
  free(graph->material_parameters);
  free(graph->mechanics_regions);
  free(graph->mechanics_constraints);
  free(graph->hydraulic_constraints);
  for (index = 0U; index < SS_PMG_MAX_LEVELS; ++index) {
    free(graph->policy.pmg_active_ranks[index]);
  }
  ss_solver_raw_destroy(&graph->raw);
  ss_solver_model_destroy(&graph->model);
  memset(graph, 0, sizeof(*graph));
}
