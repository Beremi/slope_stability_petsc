/* Coupling snapshot of the engine: freeze the seepage pressure at the
 * mechanics quadrature points and evaluate the coupled body force.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/core/sort.h"
#include "slope_internal/core/util.h"
#include "slope_internal/mesh/cell_index.h"
#include "solver_internal.h"

#include "slope_internal/core/sha256.h"
#include "slope_internal/fe/fe.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  SsSolverGraph *graph;
  int rank;
  SsCellPack *source_cells;
  size_t source_cell_count;
  double *source_values;
  size_t source_stride;
  SsTransferTargetPoint *targets;
  size_t target_count;
  const SsFeDescriptor *source_descriptor;
  const SsFeDescriptor *coordinate_descriptor;
  const SsFeQuadrature *target_quadrature;
  char source_space_identity[SS_TRANSFER_ID_CAPACITY];
  char target_space_identity[SS_TRANSFER_ID_CAPACITY];
  char model_identity[SS_TRANSFER_ID_CAPACITY];
} SsSolverCouplingContext;

struct SsSolverCouplingSnapshot {
  uint32_t dimension;
  size_t quadrature_count;
  size_t cell_count;
  SsEntityId *cell_ids;
  double *body_force;
};

static SsStatus
ss_solver_coupling_copy_identity(char destination[SS_TRANSFER_ID_CAPACITY],
                                 const char *source) {
  size_t length = 0U;
  if (destination == NULL || source == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  while (length < SS_TRANSFER_ID_CAPACITY && source[length] != '\0') {
    ++length;
  }
  if (length == 0U || length == SS_TRANSFER_ID_CAPACITY) {
    return SS_ERROR_LIMIT;
  }
  memcpy(destination, source, length + 1U);
  return SS_OK;
}

static int ss_solver_coupling_cell_compare(const void *left_raw,
                                           const void *right_raw) {
  const SsCellPack *left = (const SsCellPack *)left_raw;
  const SsCellPack *right = (const SsCellPack *)right_raw;
  return left->stable_cell_id < right->stable_cell_id   ? -1
         : left->stable_cell_id > right->stable_cell_id ? 1
                                                        : 0;
}

static size_t ss_solver_coupling_cell_find(const SsSolverCouplingContext *context,
                                           SsEntityId stable_cell_id) {
  if (context == NULL) {
    return 0U;
  }
  return ss_mesh_cell_index_find(context->source_cells, context->source_cell_count,
                                 sizeof(*context->source_cells),
                                 offsetof(SsCellPack, stable_cell_id), stable_cell_id);
}

static size_t ss_solver_coupling_index_find(const PetscInt *indices, size_t count,
                                            PetscInt value) {
  size_t left = 0U;
  size_t right = count;
  while (left < right) {
    const size_t middle = left + (right - left) / 2U;
    if (indices[middle] < value) {
      left = middle + 1U;
    } else {
      right = middle;
    }
  }
  return left < count && indices[left] == value ? left : count;
}

static const SsSolverMaterial *ss_solver_coupling_material(const SsSolverGraph *graph,
                                                           SsSolverMaterialKind kind,
                                                           uint32_t material_value) {
  size_t index;
  if (graph == NULL) {
    return NULL;
  }
  for (index = 0U; index < graph->model.material_count; ++index) {
    const SsSolverMaterial *material = &graph->model.materials[index];
    if (material->kind == kind && material->support_value == material_value) {
      return material;
    }
  }
  return NULL;
}

static SsStatus ss_solver_coupling_identity(const SsAssemblyPlan *plan,
                                            char identity[SS_TRANSFER_ID_CAPACITY]) {
  SsAssemblyPlanSummary summary;
  if (plan == NULL || identity == NULL ||
      ss_assembly_plan_summary(plan, &summary) != SS_OK) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return ss_solver_coupling_copy_identity(identity, summary.pattern_digest);
}

static SsStatus ss_solver_coupling_gather_source(SsSolverCouplingContext *context,
                                                 Vec hydraulic_solution) {
  PetscInt *indices = NULL;
  size_t index_capacity;
  size_t index_count = 0U;
  size_t cell_index;
  IS gather_is = NULL;
  Vec gathered = NULL;
  VecScatter scatter = NULL;
  const PetscScalar *values = NULL;
  int values_open = 0;
  SsStatus status = SS_OK;
  if (context->source_cell_count > SIZE_MAX / context->source_stride) {
    status = SS_ERROR_OVERFLOW;
  } else {
    index_capacity = context->source_cell_count * context->source_stride;
    indices = calloc(index_capacity == 0U ? 1U : index_capacity, sizeof(*indices));
    context->source_values = calloc(index_capacity == 0U ? 1U : index_capacity,
                                    sizeof(*context->source_values));
    if (indices == NULL || context->source_values == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  if (status != SS_OK) {
    goto prepare_scatter;
  }
  for (cell_index = 0U; cell_index < context->source_cell_count; ++cell_index) {
    const SsCellPack *cell = &context->source_cells[cell_index];
    size_t slot;
    if (cell->components != 1U || cell->closure_count > context->source_stride) {
      status = SS_ERROR_INTEGRITY;
      goto prepare_scatter;
    }
    for (slot = 0U; slot < cell->closure_count; ++slot) {
      if (cell->constrained[slot] == 0U) {
        if (cell->global_indices[slot] < 0 || index_count >= index_capacity) {
          status = SS_ERROR_INTEGRITY;
          goto prepare_scatter;
        }
        indices[index_count++] = cell->global_indices[slot];
      }
    }
  }
  ss_sort(indices, index_count, sizeof(*indices), ss_petsc_int_compare);
  {
    size_t source;
    size_t unique = 0U;
    for (source = 0U; source < index_count; ++source) {
      if (unique == 0U || indices[source] != indices[unique - 1U]) {
        indices[unique++] = indices[source];
      }
    }
    index_count = unique;
  }
prepare_scatter:
  status = ss_status_allreduce_max(context->graph->communicator, status);
  if (status == SS_OK &&
      (index_count > (size_t)PETSC_MAX_INT ||
       ISCreateGeneral(context->graph->communicator, (PetscInt)index_count, indices,
                       PETSC_COPY_VALUES, &gather_is) != PETSC_SUCCESS ||
       VecCreateSeq(PETSC_COMM_SELF, (PetscInt)index_count, &gathered) !=
           PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(context->graph->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  const PetscErrorCode created =
      VecScatterCreate(hydraulic_solution, gather_is, gathered, NULL, &scatter);
  status =
      ss_status_allreduce_max(context->graph->communicator,
                              created == PETSC_SUCCESS ? SS_OK : SS_ERROR_INTERNAL);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (VecScatterBegin(scatter, hydraulic_solution, gathered, INSERT_VALUES,
                      SCATTER_FORWARD) != PETSC_SUCCESS ||
      VecScatterEnd(scatter, hydraulic_solution, gathered, INSERT_VALUES,
                    SCATTER_FORWARD) != PETSC_SUCCESS ||
      VecGetArrayRead(gathered, &values) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  values_open = 1;
  for (cell_index = 0U; cell_index < context->source_cell_count; ++cell_index) {
    const SsCellPack *cell = &context->source_cells[cell_index];
    size_t slot;
    for (slot = 0U; slot < cell->closure_count; ++slot) {
      double value;
      if (cell->constrained[slot] != 0U) {
        value = cell->prescribed_values[slot];
      } else {
        const size_t gathered_slot = ss_solver_coupling_index_find(
            indices, index_count, cell->global_indices[slot]);
        if (gathered_slot == index_count) {
          status = SS_ERROR_INTEGRITY;
          break;
        }
        value = (double)PetscRealPart(values[gathered_slot]);
      }
      if (!isfinite(value)) {
        status = SS_ERROR_INTEGRITY;
        break;
      }
      context->source_values[cell_index * context->source_stride + slot] = value;
    }
    if (status != SS_OK) {
      break;
    }
  }

cleanup:
  if (values_open && VecRestoreArrayRead(gathered, &values) != PETSC_SUCCESS &&
      status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (VecScatterDestroy(&scatter) != PETSC_SUCCESS && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (VecDestroy(&gathered) != PETSC_SUCCESS && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (ISDestroy(&gather_is) != PETSC_SUCCESS && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  free(indices);
  return status;
}

static SsStatus ss_solver_coupling_targets(SsSolverCouplingContext *context) {
  const SsFeGeometryPolicy geometry_policy = {
      64.0 * DBL_EPSILON,
      1.0 / (64.0 * DBL_EPSILON),
  };
  SsCellPack *cells = NULL;
  size_t cell_count = 0U;
  size_t target_count;
  size_t cell_index;
  SsStatus status;
  if (context == NULL || context->graph == NULL || context->target_quadrature == NULL ||
      context->coordinate_descriptor == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_field_space_pack_owned_cells(context->graph->mechanics_space, NULL, 0U,
                                           &cell_count);
  if (status != SS_OK) {
    return status;
  }
  if (context->target_quadrature->point_count != 0U &&
      cell_count > SIZE_MAX / context->target_quadrature->point_count) {
    return SS_ERROR_OVERFLOW;
  }
  target_count = cell_count * context->target_quadrature->point_count;
  cells = (SsCellPack *)calloc(cell_count == 0U ? 1U : cell_count, sizeof(*cells));
  context->targets = (SsTransferTargetPoint *)calloc(
      target_count == 0U ? 1U : target_count, sizeof(*context->targets));
  if (cells == NULL || context->targets == NULL) {
    free(cells);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  status = ss_field_space_pack_owned_cells(context->graph->mechanics_space, cells,
                                           cell_count, &cell_count);
  for (cell_index = 0U; status == SS_OK && cell_index < cell_count; ++cell_index) {
    const SsCellPack *cell = &cells[cell_index];
    const SsSolverMaterial *mechanics_material = ss_solver_coupling_material(
        context->graph, SS_SOLVER_MATERIAL_MECHANICS, cell->material_value);
    const SsSolverMaterial *hydraulic_material = ss_solver_coupling_material(
        context->graph, SS_SOLVER_MATERIAL_HYDRAULIC, cell->material_value);
    double vertices[(SS_FE_MAX_DIMENSION + 1U) * SS_FE_MAX_DIMENSION] = {0.0};
    double minimum_edge = NAN;
    double epsilon = NAN;
    size_t quadrature_index;
    if (mechanics_material == NULL || hydraulic_material == NULL ||
        !isfinite(mechanics_material->unsaturated_unit_weight) ||
        mechanics_material->unsaturated_unit_weight <= 0.0 ||
        !isfinite(mechanics_material->saturated_unit_weight) ||
        mechanics_material->saturated_unit_weight <= 0.0 ||
        ss_solver_cell_vertex_coordinates(cell, context->coordinate_descriptor,
                                          vertices) != SS_OK) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    status = ss_hydraulic_cell_epsilon(context->graph->model.dimension, vertices,
                                       hydraulic_material->water_unit_weight,
                                       &minimum_edge, &epsilon);
    for (quadrature_index = 0U;
         status == SS_OK && quadrature_index < context->target_quadrature->point_count;
         ++quadrature_index) {
      const double *reference = context->target_quadrature->points +
                                quadrature_index * context->graph->model.dimension;
      const size_t target_index =
          cell_index * context->target_quadrature->point_count + quadrature_index;
      SsTransferTargetPoint *target = &context->targets[target_index];
      SsFeGeometry geometry;
      if (quadrature_index > UINT32_MAX ||
          ss_fe_geometry_eval(context->coordinate_descriptor,
                              cell->coordinate_coordinates, reference, &geometry_policy,
                              &geometry) != SS_OK ||
          ss_solver_coupling_point_id(context->graph->model.dimension,
                                      cell->stable_cell_id, (uint32_t)quadrature_index,
                                      &target->stable_point_id) != SS_OK) {
        status = SS_ERROR_INTEGRITY;
        break;
      }
      target->stable_target_cell_id = cell->stable_cell_id;
      target->quadrature_index = (uint32_t)quadrature_index;
      target->dimension = context->graph->model.dimension;
      memcpy(target->physical, geometry.physical,
             target->dimension * sizeof(*target->physical));
      target->physical_weight =
          context->target_quadrature->weights[quadrature_index] * geometry.determinant;
      target->cell_epsilon = epsilon;
      target->saturated_unit_weight = mechanics_material->saturated_unit_weight;
      target->unsaturated_unit_weight = mechanics_material->unsaturated_unit_weight;
      if (!isfinite(target->physical_weight) || target->physical_weight <= 0.0) {
        status = SS_ERROR_INTEGRITY;
        break;
      }
    }
  }
  free(cells);
  if (status == SS_OK) {
    context->target_count = target_count;
  }
  return status;
}

static SsStatus ss_solver_coupling_target_count(void *opaque, size_t *count) {
  const SsSolverCouplingContext *context = (const SsSolverCouplingContext *)opaque;
  if (context == NULL || count == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *count = context->target_count;
  return SS_OK;
}

static SsStatus ss_solver_coupling_target_at(void *opaque, size_t index,
                                             SsTransferTargetPoint *target) {
  const SsSolverCouplingContext *context = (const SsSolverCouplingContext *)opaque;
  if (context == NULL || target == NULL || index >= context->target_count) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *target = context->targets[index];
  return SS_OK;
}

static SsStatus
ss_solver_coupling_same_mesh_map(void *opaque, const SsTransferTargetPoint *target,
                                 int *source_rank,
                                 double reference[SS_TRANSFER_MAX_DIMENSION]) {
  const SsSolverCouplingContext *context = opaque;
  if (target->dimension != context->graph->model.dimension ||
      target->quadrature_index >= context->target_quadrature->point_count) {
    return SS_ERROR_INTEGRITY;
  }
  /* Both field spaces are attached to the same distributed mesh. Owned
   * mechanics targets use the matching owned pressure cell and the same
   * reference chart. The resident force halo is imported separately. */
  if (ss_solver_coupling_cell_find(context, target->stable_target_cell_id) ==
      context->source_cell_count) {
    return SS_ERROR_NOT_FOUND;
  }
  memcpy(reference,
         context->target_quadrature->points +
             (size_t)target->quadrature_index * target->dimension,
         target->dimension * sizeof(*reference));
  *source_rank = context->rank;
  return SS_OK;
}

static SsStatus
ss_solver_coupling_evaluate(void *opaque, SsEntityId stable_source_cell_id,
                            const double reference[SS_TRANSFER_MAX_DIMENSION],
                            double *pressure,
                            double gradient[SS_TRANSFER_MAX_DIMENSION]) {
  const SsFeGeometryPolicy geometry_policy = {
      64.0 * DBL_EPSILON,
      1.0 / (64.0 * DBL_EPSILON),
  };
  const SsSolverCouplingContext *context = (const SsSolverCouplingContext *)opaque;
  double basis[SS_FE_MAX_NODES] = {0.0};
  double reference_gradients[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
  double physical_gradients[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
  SsFeGeometry geometry;
  size_t cell_index;
  const SsCellPack *cell;
  size_t node;
  if (context == NULL || reference == NULL || pressure == NULL || gradient == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  cell_index = ss_solver_coupling_cell_find(context, stable_source_cell_id);
  if (cell_index == context->source_cell_count) {
    return SS_ERROR_NOT_FOUND;
  }
  cell = &context->source_cells[cell_index];
  if (cell->field_node_count != context->source_descriptor->node_count ||
      cell->closure_count != cell->field_node_count ||
      ss_fe_basis_eval_scalar(context->source_descriptor, reference, basis,
                              reference_gradients) != SS_OK ||
      ss_fe_geometry_eval(context->coordinate_descriptor, cell->coordinate_coordinates,
                          reference, &geometry_policy, &geometry) != SS_OK ||
      ss_fe_physical_gradients(context->source_descriptor, &geometry,
                               reference_gradients, physical_gradients) != SS_OK) {
    return SS_ERROR_INTEGRITY;
  }
  *pressure = 0.0;
  memset(gradient, 0, SS_TRANSFER_MAX_DIMENSION * sizeof(*gradient));
  for (node = 0U; node < cell->field_node_count; ++node) {
    const double value =
        context->source_values[cell_index * context->source_stride + node];
    uint32_t coordinate;
    *pressure += basis[node] * value;
    for (coordinate = 0U; coordinate < context->graph->model.dimension; ++coordinate) {
      gradient[coordinate] +=
          physical_gradients[node * context->graph->model.dimension + coordinate] *
          value;
    }
  }
  if (!isfinite(*pressure)) {
    return SS_ERROR_INTEGRITY;
  }
  for (node = 0U; node < context->graph->model.dimension; ++node) {
    if (!isfinite(gradient[node])) {
      return SS_ERROR_INTEGRITY;
    }
  }
  return SS_OK;
}

static SsStatus ss_solver_coupling_source_state(void *opaque,
                                                SsTransferSourceState *state) {
  const SsSolverCouplingContext *context = (const SsSolverCouplingContext *)opaque;
  if (context == NULL || state == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(state, 0, sizeof(*state));
  state->accepted_generation = 1U;
  state->mesh_generation = 1U;
  state->coordinate_generation = 1U;
  if (ss_solver_coupling_copy_identity(state->mesh_identity,
                                       context->graph->geometry_id) != SS_OK ||
      ss_solver_coupling_copy_identity(state->field_space_identity,
                                       context->source_space_identity) != SS_OK ||
      ss_solver_coupling_copy_identity(state->model_identity,
                                       context->model_identity) != SS_OK) {
    return SS_ERROR_LIMIT;
  }
  memcpy(state->pressure_unit, "kPa", 4U);
  return SS_OK;
}

static void ss_solver_coupling_context_destroy(SsSolverCouplingContext *context) {
  if (context == NULL) {
    return;
  }
  free(context->targets);
  free(context->source_values);
  free(context->source_cells);
  memset(context, 0, sizeof(*context));
}

static SsStatus ss_solver_coupling_context_create(SsSolverGraph *graph,
                                                  Vec hydraulic_solution,
                                                  SsSolverCouplingContext *context) {
  SsFeQuadratureRequest request;
  SsFeQuadraturePlan quadrature;
  size_t source_count = 0U;
  SsStatus status;
  memset(context, 0, sizeof(*context));
  context->graph = graph;
  context->source_stride = SS_FIELD_MAX_CLOSURE;
  if (graph->mechanics_space == NULL || graph->hydraulic_space == NULL ||
      graph->mechanics_plan == NULL || graph->hydraulic_plan == NULL ||
      MPI_Comm_rank(graph->communicator, &context->rank) != MPI_SUCCESS) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else {
    status = ss_fe_descriptor_find(graph->model.dimension, graph->model.hydraulic_order,
                                   &context->source_descriptor);
  }
  if (status == SS_OK) {
    status =
        ss_fe_descriptor_find(graph->model.dimension, graph->model.coordinate_order,
                              &context->coordinate_descriptor);
  }
  memset(&request, 0, sizeof(request));
  memset(&quadrature, 0, sizeof(quadrature));
  request.parent_dimension = graph->model.dimension;
  request.domain = SS_FE_QUADRATURE_VOLUME;
  request.operation = SS_FE_OPERATION_MECHANICS_STIFFNESS;
  request.field_order = graph->model.mechanics_order;
  request.coordinate_order = graph->model.coordinate_order;
  /* Targets are the mechanics volume points: use the operators' rule. */
  request.mode =
      graph->model.volume_quadrature_mode == SS_FE_QUADRATURE_MODE_PAIRED_HIGHER
          ? SS_FE_QUADRATURE_MODE_PAIRED_HIGHER
          : SS_FE_QUADRATURE_MODE_DEFAULT;
  if (status == SS_OK) {
    status = ss_fe_quadrature_plan(&request, &quadrature);
  }
  if (status == SS_OK && quadrature.selected_rule == NULL) {
    status = SS_ERROR_INTEGRITY;
  }
  if (status == SS_OK) {
    context->target_quadrature = quadrature.selected_rule;
    status = ss_field_space_pack_owned_cells(graph->hydraulic_space, NULL, 0U,
                                             &source_count);
  }
  if (status == SS_OK) {
    context->source_cells = (SsCellPack *)calloc(source_count == 0U ? 1U : source_count,
                                                 sizeof(*context->source_cells));
    if (context->source_cells == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  if (status == SS_OK) {
    status = ss_field_space_pack_owned_cells(
        graph->hydraulic_space, context->source_cells, source_count, &source_count);
  }
  if (status == SS_OK) {
    ss_sort(context->source_cells, source_count, sizeof(*context->source_cells),
            ss_solver_coupling_cell_compare);
    context->source_cell_count = source_count;
    status = ss_solver_coupling_identity(graph->hydraulic_plan,
                                         context->source_space_identity);
  }
  if (status == SS_OK) {
    status = ss_solver_coupling_identity(graph->mechanics_plan,
                                         context->target_space_identity);
  }
  if (status == SS_OK) {
    const int written =
        snprintf(context->model_identity, sizeof(context->model_identity), "%s",
                 "steady-unconfined-native-v1");
    status = written >= 0 && (size_t)written < sizeof(context->model_identity)
                 ? SS_OK
                 : SS_ERROR_LIMIT;
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    status = ss_solver_coupling_gather_source(context, hydraulic_solution);
  }
  if (status == SS_OK) {
    status = ss_solver_coupling_targets(context);
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status != SS_OK) {
    ss_solver_coupling_context_destroy(context);
  }
  return status;
}

static void ss_solver_coupling_snapshot_release(SsSolverCouplingSnapshot *snapshot) {
  if (snapshot == NULL) {
    return;
  }
  free(snapshot->body_force);
  free(snapshot->cell_ids);
  memset(snapshot, 0, sizeof(*snapshot));
  free(snapshot);
}

static SsStatus ss_solver_coupling_resident_snapshot_build(
    SsSolverGraph *graph, const SsFeQuadrature *quadrature,
    const SsCouplingFieldSnapshot *owned_snapshot, SsSolverCouplingSnapshot **snapshot,
    size_t *resident_bytes) {
  SsSolverCouplingSnapshot *created = NULL;
  SsCouplingSnapshotView view;
  SsFieldSummary field_summary;
  double *owned_values = NULL;
  size_t owned_cell_count;
  size_t incident_cell_count;
  size_t values_per_cell;
  size_t owned_value_count;
  size_t incident_value_count;
  size_t cell_index;
  size_t point_index;
  SsStatus status;
  if (snapshot != NULL) {
    *snapshot = NULL;
  }
  SsStatus local_status = SS_OK;
  if (quadrature == NULL || owned_snapshot == NULL || snapshot == NULL ||
      resident_bytes == NULL || graph->mechanics_space == NULL ||
      quadrature->point_count == 0U || quadrature->point_count > UINT32_MAX ||
      ss_coupling_field_snapshot_view(owned_snapshot, &view) != SS_OK ||
      ss_field_space_summary(graph->mechanics_space, &field_summary) != SS_OK ||
      field_summary.owned_cell_plan_count > SIZE_MAX ||
      field_summary.incident_cell_plan_count > SIZE_MAX) {
    local_status = SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_status_allreduce_max(graph->communicator, local_status);
  if (status != SS_OK) {
    return status;
  }
  owned_cell_count = (size_t)field_summary.owned_cell_plan_count;
  incident_cell_count = (size_t)field_summary.incident_cell_plan_count;
  if (view.dimension != graph->model.dimension ||
      view.mode != graph->model.coupling_mode || view.cell_count != owned_cell_count ||
      quadrature->point_count > SIZE_MAX / view.dimension) {
    local_status = SS_ERROR_INTEGRITY;
  }
  status = ss_status_allreduce_max(graph->communicator, local_status);
  if (status != SS_OK) {
    return status;
  }
  values_per_cell = quadrature->point_count * view.dimension;
  if (owned_cell_count > SIZE_MAX / quadrature->point_count ||
      view.point_count != owned_cell_count * quadrature->point_count ||
      owned_cell_count > SIZE_MAX / values_per_cell ||
      incident_cell_count > SIZE_MAX / values_per_cell ||
      (view.point_count != 0U &&
       (view.cell_ids == NULL || view.cell_ids_by_point == NULL ||
        view.quadrature_indices == NULL || view.body_force == NULL))) {
    local_status = SS_ERROR_INTEGRITY;
  }
  if (local_status == SS_OK &&
      (incident_cell_count > SIZE_MAX / sizeof(*created->cell_ids) ||
       incident_cell_count * values_per_cell >
           SIZE_MAX / sizeof(*created->body_force))) {
    local_status = SS_ERROR_OVERFLOW;
  }
  status = ss_status_allreduce_max(graph->communicator, local_status);
  if (status != SS_OK) {
    return status;
  }
  owned_value_count = owned_cell_count * values_per_cell;
  incident_value_count = incident_cell_count * values_per_cell;
  created = (SsSolverCouplingSnapshot *)calloc(1U, sizeof(*created));
  owned_values = (double *)calloc(owned_value_count == 0U ? 1U : owned_value_count,
                                  sizeof(*owned_values));
  if (created != NULL) {
    created->cell_ids =
        (SsEntityId *)calloc(incident_cell_count == 0U ? 1U : incident_cell_count,
                             sizeof(*created->cell_ids));
    created->body_force =
        (double *)calloc(incident_value_count == 0U ? 1U : incident_value_count,
                         sizeof(*created->body_force));
  }
  status = created == NULL || owned_values == NULL || created->cell_ids == NULL ||
                   created->body_force == NULL
               ? SS_ERROR_OUT_OF_MEMORY
               : SS_OK;
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  for (cell_index = 0U; status == SS_OK && cell_index < owned_cell_count;
       ++cell_index) {
    if (view.cell_ids[cell_index] == 0U ||
        (cell_index != 0U &&
         view.cell_ids[cell_index - 1U] >= view.cell_ids[cell_index])) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    for (point_index = 0U; point_index < quadrature->point_count; ++point_index) {
      const size_t point = cell_index * quadrature->point_count + point_index;
      if (view.cell_ids_by_point[point] != view.cell_ids[cell_index] ||
          view.quadrature_indices[point] != (uint32_t)point_index) {
        status = SS_ERROR_INTEGRITY;
        break;
      }
      memcpy(owned_values + cell_index * values_per_cell + point_index * view.dimension,
             view.body_force + point * view.dimension,
             view.dimension * sizeof(*owned_values));
    }
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    status = ss_field_space_broadcast_owned_cell_values(
        graph->mechanics_space, values_per_cell, view.cell_ids, owned_values,
        owned_cell_count, created->cell_ids, created->body_force, incident_cell_count);
  }
  for (cell_index = 0U; status == SS_OK && cell_index < incident_cell_count;
       ++cell_index) {
    size_t value_index;
    if (created->cell_ids[cell_index] == 0U ||
        (cell_index != 0U &&
         created->cell_ids[cell_index - 1U] >= created->cell_ids[cell_index])) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    for (value_index = 0U; value_index < values_per_cell; ++value_index) {
      if (!isfinite(created->body_force[cell_index * values_per_cell + value_index])) {
        status = SS_ERROR_INTEGRITY;
        break;
      }
    }
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (incident_cell_count >
          (SIZE_MAX - sizeof(*created)) / sizeof(*created->cell_ids) ||
      incident_value_count > (SIZE_MAX - sizeof(*created) -
                              incident_cell_count * sizeof(*created->cell_ids)) /
                                 sizeof(*created->body_force)) {
    status = SS_ERROR_OVERFLOW;
    goto cleanup;
  }
  created->dimension = view.dimension;
  created->quadrature_count = quadrature->point_count;
  created->cell_count = incident_cell_count;
  *resident_bytes = sizeof(*created) +
                    incident_cell_count * sizeof(*created->cell_ids) +
                    incident_value_count * sizeof(*created->body_force);
  *snapshot = created;
  created = NULL;

cleanup:
  free(owned_values);
  ss_solver_coupling_snapshot_release(created);
  return status;
}

SsStatus ss_solver_coupling_snapshot_mechanics_force(
    const void *opaque, uint32_t dimension, SsEntityId stable_cell_id,
    uint32_t quadrature_index, const double *physical, double *body_force) {
  const SsSolverCouplingSnapshot *snapshot = (const SsSolverCouplingSnapshot *)opaque;
  size_t left = 0U;
  size_t right;
  uint32_t coordinate;
  if (snapshot == NULL || physical == NULL || body_force == NULL ||
      dimension != snapshot->dimension || stable_cell_id == 0U ||
      quadrature_index >= snapshot->quadrature_count) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (coordinate = 0U; coordinate < dimension; ++coordinate) {
    if (!isfinite(physical[coordinate])) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
  }
  right = snapshot->cell_count;
  while (left < right) {
    const size_t middle = left + (right - left) / 2U;
    if (snapshot->cell_ids[middle] < stable_cell_id) {
      left = middle + 1U;
    } else {
      right = middle;
    }
  }
  if (left >= snapshot->cell_count || snapshot->cell_ids[left] != stable_cell_id) {
    return SS_ERROR_NOT_FOUND;
  }
  memcpy(body_force,
         snapshot->body_force +
             (left * snapshot->quadrature_count + quadrature_index) * dimension,
         dimension * sizeof(*body_force));
  return SS_OK;
}

SsStatus ss_solver_coupling_snapshot_destroy(SsSolverCouplingSnapshot **snapshot) {
  if (snapshot == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  ss_solver_coupling_snapshot_release(*snapshot);
  *snapshot = NULL;
  return SS_OK;
}

SsStatus ss_solver_coupling_snapshot_build(SsSolverGraph *graph, Vec hydraulic_solution,
                                           int requested_threads,
                                           SsSolverCouplingSnapshot **snapshot,
                                           SsCouplingDiagnostics *diagnostics,
                                           SsErrorRecord *error) {
  SsSolverCouplingContext context;
  SsCouplingPlanSpec specification;
  SsCouplingTransferPlan *plan = NULL;
  SsCouplingFieldSnapshot *owned_snapshot = NULL;
  size_t resident_bytes = 0U;
  SsStatus status;
  const char *stage = "context";
  if (snapshot != NULL) {
    *snapshot = NULL;
  }
  if (diagnostics != NULL) {
    memset(diagnostics, 0, sizeof(*diagnostics));
  }
  if (graph == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_status_allreduce_max(
      graph->communicator,
      hydraulic_solution == NULL || snapshot == NULL || diagnostics == NULL ||
              graph->model.coupling_enabled == 0 ||
              (graph->model.coupling_mode != SS_COUPLING_PRESSURE_GRADIENT_AND_WEIGHT &&
               graph->model.coupling_mode != SS_COUPLING_WEIGHT_ONLY)
          ? SS_ERROR_INVALID_ARGUMENT
          : SS_OK);
  if (status != SS_OK) {
    return status;
  }
  memset(&context, 0, sizeof(context));
  status = ss_solver_coupling_context_create(graph, hydraulic_solution, &context);
  memset(&specification, 0, sizeof(specification));
  if (status == SS_OK) {
    stage = "plan";
    specification.dimension = graph->model.dimension;
    specification.upward_axis = 1U;
    specification.transfer_type = SS_COUPLING_TRANSFER_SAME_MESH;
    specification.requested_threads = requested_threads > 0 ? requested_threads : 1;
    specification.openmp_work_threshold = 1U;
    status = ss_solver_coupling_copy_identity(specification.target_mesh_identity,
                                              graph->geometry_id);
    if (status == SS_OK) {
      status = ss_solver_coupling_copy_identity(specification.target_space_identity,
                                                context.target_space_identity);
    }
    if (status == SS_OK) {
      status = ss_solver_coupling_copy_identity(
          specification.target_quadrature_identity, context.target_quadrature->id);
    }
    specification.geometry.context = &context;
    specification.geometry.target_count = ss_solver_coupling_target_count;
    specification.geometry.target_at = ss_solver_coupling_target_at;
    specification.same_mesh_map = ss_solver_coupling_same_mesh_map;
    specification.geometry.evaluate_source = ss_solver_coupling_evaluate;
    specification.geometry.source_state = ss_solver_coupling_source_state;
    status = ss_status_allreduce_max(graph->communicator, status);
    if (status == SS_OK) {
      status = ss_coupling_plan_create(graph->communicator, &specification, &plan);
    }
  }
  if (status == SS_OK) {
    stage = "snapshot";
    status = ss_coupling_field_snapshot_build(plan, graph->model.coupling_mode,
                                              &owned_snapshot);
  }
  if (status == SS_OK) {
    stage = "diagnostics";
    status = ss_coupling_field_snapshot_diagnostics(owned_snapshot, diagnostics);
  }
  if (status == SS_OK) {
    stage = "resident_halo";
    status = ss_solver_coupling_resident_snapshot_build(
        graph, context.target_quadrature, owned_snapshot, snapshot, &resident_bytes);
  }
  if (status == SS_OK) {
    diagnostics->snapshot_bytes = resident_bytes;
  }
  if (ss_coupling_plan_destroy(&plan) != SS_OK && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (ss_coupling_field_snapshot_destroy(&owned_snapshot) != SS_OK && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  ss_solver_coupling_context_destroy(&context);
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status != SS_OK) {
    (void)ss_solver_coupling_snapshot_destroy(snapshot);
    if (error != NULL && error->valid == 0) {
      char message[SS_ERROR_MESSAGE_CAPACITY];
      const int written = snprintf(message, sizeof(message),
                                   "coupling snapshot build failed during %s", stage);
      ss_solver_error(error, status, "COUPLING_SNAPSHOT_BUILD",
                      written >= 0 && (size_t)written < sizeof(message)
                          ? message
                          : "coupling snapshot build failed",
                      3U);
    }
  }
  return status;
}
