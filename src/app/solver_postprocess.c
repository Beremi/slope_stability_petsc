/* Mechanics nodal fields, constitutive output, equilibrium recovery and the
 * natural-boundary facet traversal shared with the hydraulic balance.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "solver_output_internal.h"

#include "../assembly/voigt.h"

#include "slope_internal/core/collective.h"
#include "slope_internal/fe/fe.h"

#include <petsclog.h>

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void ss_solver_output_field_destroy(SsSolverOutputField *field) {
  if (field == NULL) {
    return;
  }
  free(field->ids);
  free(field->slots);
  free(field->values);
  memset(field, 0, sizeof(*field));
}

void ss_solver_output_category_destroy(SsSolverOutputCategory *category) {
  if (category == NULL) {
    return;
  }
  free(category->ids);
  free(category->slots);
  free(category->values);
  memset(category, 0, sizeof(*category));
}

SsStatus ss_solver_nodal_output(const SsSolverGraph *graph, SsFieldSpace *space,
                                uint32_t component_count,
                                const PetscScalar *local_values,
                                SsSolverOutputField *field) {
  const SsFieldOwnedNode *nodes = NULL;
  size_t count = 0U;
  SsStatus status;
  if (graph == NULL || space == NULL || component_count == 0U ||
      component_count > SS_FE_MAX_DIMENSION || local_values == NULL || field == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(field, 0, sizeof(*field));
  field->component_count = component_count;
  field->sample_count = 1U;
  status = ss_field_space_owned_nodes(space, &nodes, &count);
  if (status == SS_OK && count > SIZE_MAX / component_count) {
    status = SS_ERROR_OVERFLOW;
  }
  if (status == SS_OK) {
    const size_t capacity = count == 0U ? 1U : count;
    field->ids = calloc(capacity, sizeof(*field->ids));
    field->slots = calloc(capacity, sizeof(*field->slots));
    field->values = calloc(capacity * component_count, sizeof(*field->values));
    if (field->ids == NULL || field->slots == NULL || field->values == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  for (size_t index = 0U; status == SS_OK && index < count; ++index) {
    const SsFieldOwnedNode *node = &nodes[index];
    status = ss_solver_field_node_id(
        graph->model.dimension,
        space == graph->mechanics_space ? graph->model.mechanics_order
                                        : graph->model.hydraulic_order,
        node->entity_id, node->entity_slot, &field->ids[index]);
    field->slots[index] = index;
    for (uint32_t component = 0U; component < component_count; ++component) {
      field->values[index * component_count + component] =
          node->global_indices[component] < 0
              ? node->prescribed[component]
              : (double)PetscRealPart(
                    local_values[node->local_offset + (PetscInt)component]);
    }
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    const uint64_t local_count = (uint64_t)count;
    field->local_count = count;
    if (MPI_Allreduce(&local_count, &field->global_count, 1, MPI_UINT64_T, MPI_SUM,
                      graph->communicator) != MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    } else if (field->global_count == 0U) {
      status = SS_ERROR_INTEGRITY;
    }
  }
  if (status != SS_OK) {
    ss_solver_output_field_destroy(field);
  }
  return status;
}

static const SsMechanicsMaterialRegion *
ss_solver_cell_material_region(const SsSolverGraph *graph, uint32_t material_value) {
  size_t index;
  for (index = 0U; index < graph->mechanics_region_count; ++index) {
    if (graph->mechanics_regions[index].material_value == material_value) {
      return &graph->mechanics_regions[index];
    }
  }
  return NULL;
}

static const SsMcParameters *ss_solver_cell_material(const SsSolverGraph *graph,
                                                     uint32_t material_value) {
  const SsMechanicsMaterialRegion *region =
      ss_solver_cell_material_region(graph, material_value);
  return region == NULL ? NULL : region->parameters;
}

SsStatus ss_solver_volume_quadrature(const SsSolverGraph *graph,
                                     SsFeOperationKind operation, uint32_t field_order,
                                     const SsFeQuadrature **quadrature) {
  SsFeQuadratureRequest request = {0};
  SsFeQuadraturePlan plan = {0};
  SsStatus status;
  if (graph == NULL || quadrature == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *quadrature = NULL;
  request.parent_dimension = graph->model.dimension;
  request.domain = SS_FE_QUADRATURE_VOLUME;
  request.operation = operation;
  request.field_order = field_order;
  request.coordinate_order = graph->model.coordinate_order;
  request.coefficient_order = 0U;
  request.mode =
      graph->model.volume_quadrature_mode == SS_FE_QUADRATURE_MODE_PAIRED_HIGHER
          ? SS_FE_QUADRATURE_MODE_PAIRED_HIGHER
          : SS_FE_QUADRATURE_MODE_DEFAULT;
  status = ss_fe_quadrature_plan(&request, &plan);
  if (status == SS_OK) {
    *quadrature = plan.selected_rule;
    if (*quadrature == NULL || (*quadrature)->point_count == 0U ||
        (*quadrature)->point_count > UINT32_MAX) {
      status = SS_ERROR_INTEGRITY;
    }
  }
  return status;
}

SsStatus ss_solver_stress_output(
    const SsSolverGraph *graph, const PetscScalar *local_values,
    double reduction_factor, SsSolverOutputField *stress_field,
    SsSolverOutputField *strain_field, SsSolverOutputField *yield_field,
    SsSolverOutputCategory *branch_category, SsSolverOutputCategory *plastic_category) {
  const SsFeDescriptor *field_descriptor = NULL;
  const SsFeDescriptor *coordinate_descriptor = NULL;
  const SsFeQuadrature *quadrature = NULL;
  const SsFeGeometryPolicy geometry_policy = {
      64.0 * DBL_EPSILON,
      1.0 / (64.0 * DBL_EPSILON),
  };
  const SsCellPlan *packs = NULL;
  size_t pack_count = 0U;
  size_t pack_index;
  uint64_t local_count;
  uint64_t global_count = 0U;
  SsStatus status;
  if (graph == NULL || local_values == NULL || stress_field == NULL ||
      strain_field == NULL || yield_field == NULL || branch_category == NULL ||
      plastic_category == NULL || !isfinite(reduction_factor) ||
      reduction_factor <= 0.0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(stress_field, 0, sizeof(*stress_field));
  memset(strain_field, 0, sizeof(*strain_field));
  memset(yield_field, 0, sizeof(*yield_field));
  memset(branch_category, 0, sizeof(*branch_category));
  memset(plastic_category, 0, sizeof(*plastic_category));
  stress_field->component_count = graph->model.dimension == 2U ? 4U : 6U;
  strain_field->component_count = graph->model.dimension == 2U ? 3U : 6U;
  yield_field->component_count = 1U;
  branch_category->category_count = 5U;
  plastic_category->category_count = 2U;
  status = ss_fe_descriptor_find(graph->model.dimension, graph->model.mechanics_order,
                                 &field_descriptor);
  if (status == SS_OK) {
    status = ss_fe_descriptor_find(
        graph->model.dimension, graph->model.coordinate_order, &coordinate_descriptor);
  }
  if (status == SS_OK) {
    status = ss_solver_volume_quadrature(graph, SS_FE_OPERATION_MECHANICS_STIFFNESS,
                                         graph->model.mechanics_order, &quadrature);
  }
  if (status != SS_OK) {
    return status;
  }
  stress_field->sample_count = (uint32_t)quadrature->point_count;
  strain_field->sample_count = (uint32_t)quadrature->point_count;
  yield_field->sample_count = (uint32_t)quadrature->point_count;
  branch_category->sample_count = (uint32_t)quadrature->point_count;
  plastic_category->sample_count = (uint32_t)quadrature->point_count;
  {
    size_t index;
    const size_t length = strlen(quadrature->id);
    if (length == 0U || length >= sizeof(stress_field->quadrature_id) ||
        length >= sizeof(strain_field->quadrature_id) ||
        length >= sizeof(yield_field->quadrature_id) ||
        length >= sizeof(branch_category->quadrature_id) ||
        length >= sizeof(plastic_category->quadrature_id)) {
      return SS_ERROR_LIMIT;
    }
    for (index = 0U; index <= length; ++index) {
      const char value = quadrature->id[index] == '-' ? '_' : quadrature->id[index];
      stress_field->quadrature_id[index] = value;
      strain_field->quadrature_id[index] = value;
      yield_field->quadrature_id[index] = value;
      branch_category->quadrature_id[index] = value;
      plastic_category->quadrature_id[index] = value;
    }
  }
  status =
      ss_field_space_owned_cell_plan_view(graph->mechanics_space, &packs, &pack_count);
  if (status != SS_OK) {
    return status;
  }
  stress_field->ids = (SsEntityId *)calloc(pack_count == 0U ? 1U : pack_count,
                                           sizeof(*stress_field->ids));
  stress_field->slots = (size_t *)calloc(pack_count == 0U ? 1U : pack_count,
                                         sizeof(*stress_field->slots));
  strain_field->ids = (SsEntityId *)calloc(pack_count == 0U ? 1U : pack_count,
                                           sizeof(*strain_field->ids));
  strain_field->slots = (size_t *)calloc(pack_count == 0U ? 1U : pack_count,
                                         sizeof(*strain_field->slots));
  yield_field->ids = (SsEntityId *)calloc(pack_count == 0U ? 1U : pack_count,
                                          sizeof(*yield_field->ids));
  yield_field->slots =
      (size_t *)calloc(pack_count == 0U ? 1U : pack_count, sizeof(*yield_field->slots));
  branch_category->ids = (SsEntityId *)calloc(pack_count == 0U ? 1U : pack_count,
                                              sizeof(*branch_category->ids));
  branch_category->slots = (size_t *)calloc(pack_count == 0U ? 1U : pack_count,
                                            sizeof(*branch_category->slots));
  plastic_category->ids = (SsEntityId *)calloc(pack_count == 0U ? 1U : pack_count,
                                               sizeof(*plastic_category->ids));
  plastic_category->slots = (size_t *)calloc(pack_count == 0U ? 1U : pack_count,
                                             sizeof(*plastic_category->slots));
  if (pack_count > SIZE_MAX / quadrature->point_count ||
      pack_count * quadrature->point_count >
          SIZE_MAX / (size_t)stress_field->component_count ||
      pack_count * quadrature->point_count >
          SIZE_MAX / (size_t)strain_field->component_count) {
    ss_solver_output_category_destroy(plastic_category);
    ss_solver_output_category_destroy(branch_category);
    ss_solver_output_field_destroy(yield_field);
    ss_solver_output_field_destroy(strain_field);
    ss_solver_output_field_destroy(stress_field);
    return SS_ERROR_OVERFLOW;
  }
  stress_field->values =
      (double *)calloc(pack_count == 0U ? 1U
                                        : pack_count * quadrature->point_count *
                                              (size_t)stress_field->component_count,
                       sizeof(*stress_field->values));
  strain_field->values =
      (double *)calloc(pack_count == 0U ? 1U
                                        : pack_count * quadrature->point_count *
                                              (size_t)strain_field->component_count,
                       sizeof(*strain_field->values));
  yield_field->values =
      (double *)calloc(pack_count == 0U ? 1U : pack_count * quadrature->point_count,
                       sizeof(*yield_field->values));
  branch_category->values =
      (uint32_t *)calloc(pack_count == 0U ? 1U : pack_count * quadrature->point_count,
                         sizeof(*branch_category->values));
  plastic_category->values =
      (uint32_t *)calloc(pack_count == 0U ? 1U : pack_count * quadrature->point_count,
                         sizeof(*plastic_category->values));
  if (stress_field->ids == NULL || stress_field->slots == NULL ||
      stress_field->values == NULL || strain_field->ids == NULL ||
      strain_field->slots == NULL || strain_field->values == NULL ||
      yield_field->ids == NULL || yield_field->slots == NULL ||
      yield_field->values == NULL || branch_category->ids == NULL ||
      branch_category->slots == NULL || branch_category->values == NULL ||
      plastic_category->ids == NULL || plastic_category->slots == NULL ||
      plastic_category->values == NULL) {
    ss_solver_output_category_destroy(plastic_category);
    ss_solver_output_category_destroy(branch_category);
    ss_solver_output_field_destroy(yield_field);
    ss_solver_output_field_destroy(strain_field);
    ss_solver_output_field_destroy(stress_field);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (pack_index = 0U; status == SS_OK && pack_index < pack_count; ++pack_index) {
    const SsCellPlan *pack = &packs[pack_index];
    const SsMcParameters *material =
        ss_solver_cell_material(graph, pack->material_value);
    double field_values[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
    size_t closure;
    size_t quadrature_index;
    if (material == NULL || pack->closure_count != field_descriptor->node_count *
                                                       (size_t)graph->model.dimension) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    for (closure = 0U; closure < pack->closure_count; ++closure) {
      if (pack->constrained[closure] != 0U) {
        field_values[closure] = pack->prescribed_values[closure];
      } else if (pack->local_indices[closure] < 0) {
        status = SS_ERROR_INTEGRITY;
        break;
      } else {
        field_values[closure] =
            (double)PetscRealPart(local_values[pack->local_indices[closure]]);
      }
    }
    stress_field->ids[pack_index] = pack->stable_cell_id;
    stress_field->slots[pack_index] = pack_index;
    strain_field->ids[pack_index] = pack->stable_cell_id;
    strain_field->slots[pack_index] = pack_index;
    yield_field->ids[pack_index] = pack->stable_cell_id;
    yield_field->slots[pack_index] = pack_index;
    branch_category->ids[pack_index] = pack->stable_cell_id;
    branch_category->slots[pack_index] = pack_index;
    plastic_category->ids[pack_index] = pack->stable_cell_id;
    plastic_category->slots[pack_index] = pack_index;
    for (quadrature_index = 0U;
         status == SS_OK && quadrature_index < quadrature->point_count;
         ++quadrature_index) {
      const double *reference =
          quadrature->points + quadrature_index * (size_t)graph->model.dimension;
      double basis[SS_FE_MAX_NODES] = {0.0};
      double reference_gradients[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
      double physical_gradients[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
      double strain[6] = {0.0};
      SsFeGeometry geometry = {0};
      SsMcResponse response = {0};
      SsMcError material_error = {0};
      size_t node;
      if (ss_fe_basis_eval_scalar(field_descriptor, reference, basis,
                                  reference_gradients) != SS_OK ||
          ss_fe_geometry_eval(coordinate_descriptor, pack->coordinate_coordinates,
                              reference, &geometry_policy, &geometry) != SS_OK ||
          ss_fe_physical_gradients(field_descriptor, &geometry, reference_gradients,
                                   physical_gradients) != SS_OK) {
        status = SS_ERROR_INTEGRITY;
        break;
      }
      for (node = 0U; node < field_descriptor->node_count; ++node) {
        const double *gradient =
            physical_gradients + node * (size_t)graph->model.dimension;
        uint32_t component;
        for (component = 0U; component < graph->model.dimension; ++component) {
          double column[6] = {0.0};
          size_t strain_component;
          const double value =
              field_values[node * (size_t)graph->model.dimension + component];
          ss_mechanics_b_column(graph->model.dimension, gradient, component, column);
          for (strain_component = 0U;
               strain_component < (graph->model.dimension == 2U ? 3U : 6U);
               ++strain_component) {
            strain[strain_component] += column[strain_component] * value;
          }
        }
      }
      status = ss_mc_evaluate(material,
                              graph->model.dimension == 2U ? SS_MC_DIMENSION_2
                                                           : SS_MC_DIMENSION_3,
                              reduction_factor, strain, &response, &material_error);
      if (status == SS_OK) {
        const size_t sample = pack_index * quadrature->point_count + quadrature_index;
        double *stress_output =
            stress_field->values + sample * (size_t)stress_field->component_count;
        double *strain_output =
            strain_field->values + sample * (size_t)strain_field->component_count;
        uint32_t component;
        if (response.branch < SS_MC_BRANCH_ELASTIC ||
            response.branch > SS_MC_BRANCH_APEX || !isfinite(response.yield_residual)) {
          status = SS_ERROR_INTEGRITY;
          break;
        }
        yield_field->values[sample] = response.yield_residual;
        branch_category->values[sample] =
            (uint32_t)(response.branch - SS_MC_BRANCH_ELASTIC);
        plastic_category->values[sample] =
            response.branch == SS_MC_BRANCH_ELASTIC ? 0U : 1U;
        if (graph->model.dimension == 2U) {
          stress_output[0] = response.stress[0];
          stress_output[1] = response.stress[1];
          stress_output[2] = response.stress[3];
          stress_output[3] = response.stress[2];
          for (component = 0U; component < 3U; ++component) {
            strain_output[component] = strain[component];
          }
        } else {
          for (component = 0U; component < 6U; ++component) {
            stress_output[component] = response.stress[component];
            strain_output[component] = strain[component];
          }
        }
      }
    }
  }
  if (status != SS_OK) {
    ss_solver_output_category_destroy(plastic_category);
    ss_solver_output_category_destroy(branch_category);
    ss_solver_output_field_destroy(yield_field);
    ss_solver_output_field_destroy(strain_field);
    ss_solver_output_field_destroy(stress_field);
    return status;
  }
  stress_field->local_count = pack_count;
  strain_field->local_count = pack_count;
  yield_field->local_count = pack_count;
  branch_category->local_count = pack_count;
  plastic_category->local_count = pack_count;
  local_count = (uint64_t)stress_field->local_count;
  if (MPI_Allreduce(&local_count, &global_count, 1, MPI_UINT64_T, MPI_SUM,
                    graph->communicator) != MPI_SUCCESS) {
    ss_solver_output_category_destroy(plastic_category);
    ss_solver_output_category_destroy(branch_category);
    ss_solver_output_field_destroy(yield_field);
    ss_solver_output_field_destroy(strain_field);
    ss_solver_output_field_destroy(stress_field);
    return SS_ERROR_INTERNAL;
  }
  stress_field->global_count = global_count;
  strain_field->global_count = global_count;
  yield_field->global_count = global_count;
  branch_category->global_count = global_count;
  plastic_category->global_count = global_count;
  return global_count == graph->raw.input.global_cell_count ? SS_OK
                                                            : SS_ERROR_INTEGRITY;
}

SsStatus ss_solver_natural_facet_residuals(
    const SsSolverGraph *graph, SsFieldSpace *space, uint32_t field_order,
    uint32_t components, const SsNaturalOperatorSpec *operations,
    size_t operation_count, const PetscScalar *local_values,
    SsSolverNaturalFacetVisit visit, void *context) {
  const SsFeDescriptor *field_descriptor = NULL;
  const SsFeQuadrature *quadrature = NULL;
  size_t operation_index;
  SsStatus status;
  status =
      ss_fe_descriptor_find(graph->model.dimension, field_order, &field_descriptor);
  if (status == SS_OK) {
    status = ss_fe_facet_quadrature_default(graph->model.dimension, field_order,
                                            &quadrature);
  }
  for (operation_index = 0U; status == SS_OK && operation_index < operation_count;
       ++operation_index) {
    const SsNaturalOperatorSpec *operation = &operations[operation_index];
    SsFieldFacetPack *facets = NULL;
    size_t facet_count = 0U;
    size_t facet_index;
    size_t facet_bytes;
    status = ss_field_space_pack_owned_facets(space, operation->support_value, NULL, 0U,
                                              &facet_count);
    if (status == SS_OK) {
      status = ss_checked_mul_size(facet_count == 0U ? 1U : facet_count,
                                   sizeof(*facets), &facet_bytes);
    }
    if (status == SS_OK) {
      facets = (SsFieldFacetPack *)calloc(1U, facet_bytes);
      if (facets == NULL) {
        status = SS_ERROR_OUT_OF_MEMORY;
      }
    }
    status = ss_status_allreduce_max(graph->communicator, status);
    if (status == SS_OK) {
      status = ss_field_space_pack_owned_facets(space, operation->support_value, facets,
                                                facet_count, &facet_count);
    }
    for (facet_index = 0U; status == SS_OK && facet_index < facet_count;
         ++facet_index) {
      const SsFieldFacetPack *facet = &facets[facet_index];
      double field_values[SS_FIELD_MAX_FACET_CLOSURE] = {0.0};
      double element_residual[SS_FIELD_MAX_FACET_CLOSURE] = {0.0};
      double signed_flux = 0.0;
      SsNaturalElementInput input = {0};
      SsElementOutput output = {0};
      SsErrorRecord element_error = {0};
      size_t slot;
      if (facet->components != components ||
          facet->closure_count > SS_FIELD_MAX_FACET_CLOSURE) {
        status = SS_ERROR_INTEGRITY;
        break;
      }
      for (slot = 0U; slot < facet->closure_count; ++slot) {
        if (facet->constrained[slot] != 0U) {
          field_values[slot] = facet->prescribed_values[slot];
        } else if (facet->local_indices[slot] < 0) {
          status = SS_ERROR_INTEGRITY;
          break;
        } else {
          field_values[slot] =
              (double)PetscRealPart(local_values[facet->local_indices[slot]]);
        }
      }
      input.facet = facet;
      input.field_descriptor = field_descriptor;
      input.quadrature = quadrature;
      input.field_values = field_values;
      input.point_kernel = operation->point_kernel;
      input.point_context = operation->point_context;
      output.closure_capacity = facet->closure_count;
      output.residual = element_residual;
      if (status == SS_OK &&
          ss_assembly_natural_element(&input, SS_ASSEMBLY_RESIDUAL, &output,
                                      &signed_flux, &element_error) != SS_OK) {
        status = element_error.valid != 0 ? element_error.status : SS_ERROR_INTEGRITY;
      }
      if (status == SS_OK) {
        status = visit(context, facet, element_residual);
      }
    }
    free(facets);
    status = ss_status_allreduce_max(graph->communicator, status);
  }
  return status;
}

typedef struct {
  uint32_t dimension;
  double scale;
  double *local;
} SsSolverNaturalEquilibrium;

/* Add the scaled facet residual to the external resultant and, on constrained
 * closure slots, to the reaction. */
static SsStatus ss_solver_natural_equilibrium_visit(void *context,
                                                    const SsFieldFacetPack *facet,
                                                    const double *element_residual) {
  SsSolverNaturalEquilibrium *equilibrium = (SsSolverNaturalEquilibrium *)context;
  size_t slot;
  for (slot = 0U; slot < facet->closure_count; ++slot) {
    const uint32_t component = (uint32_t)(slot % equilibrium->dimension);
    const double residual = element_residual[slot] * equilibrium->scale;
    if (!isfinite(residual)) {
      return SS_ERROR_INTEGRITY;
    }
    equilibrium->local[equilibrium->dimension + component] += residual;
    if (facet->constrained[slot] != 0U) {
      equilibrium->local[component] += residual;
    }
  }
  return SS_OK;
}

static SsStatus
ss_solver_mechanics_natural_equilibrium(const SsSolverGraph *graph,
                                        const SsSolverMechanics *mechanics,
                                        const PetscScalar *local_values, double factor,
                                        double local[2U * SS_FE_MAX_DIMENSION]) {
  SsSolverNaturalEquilibrium equilibrium;
  if (graph == NULL || mechanics == NULL || local_values == NULL || local == NULL ||
      !isfinite(factor) || factor <= 0.0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  equilibrium.dimension = graph->model.dimension;
  equilibrium.scale =
      graph->model.analysis_method == SS_ANALYSIS_LIMIT_LOAD ? factor : 1.0;
  equilibrium.local = local;
  return ss_solver_natural_facet_residuals(
      graph, graph->mechanics_space, graph->model.mechanics_order,
      graph->model.dimension, mechanics->natural_operations,
      mechanics->natural_operation_count, local_values,
      ss_solver_natural_equilibrium_visit, &equilibrium);
}

SsStatus ss_solver_mechanics_equilibrium(const SsSolverGraph *graph,
                                         SsSolverMechanics *mechanics,
                                         const PetscScalar *local_values, Vec solution,
                                         double factor,
                                         SsSolverMechanicsEquilibrium *equilibrium,
                                         SsErrorRecord *error) {
  const SsFeDescriptor *field_descriptor = NULL;
  const SsFeDescriptor *coordinate_descriptor = NULL;
  const SsFeQuadrature *quadrature = NULL;
  const SsCellPlan *packs = NULL;
  Vec residual = NULL;
  size_t pack_count = 0U;
  size_t pack_index;
  double local[2U * SS_FE_MAX_DIMENSION] = {0.0};
  double global[2U * SS_FE_MAX_DIMENSION] = {0.0};
  PetscReal free_residual_l2 = NAN;
  SsStatus status;
  if (graph == NULL || mechanics == NULL || local_values == NULL || solution == NULL ||
      equilibrium == NULL || error == NULL || !isfinite(factor) || factor <= 0.0 ||
      mechanics->operator_handle == NULL || graph->mechanics_space == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(equilibrium, 0, sizeof(*equilibrium));
  status = ss_fe_descriptor_find(graph->model.dimension, graph->model.mechanics_order,
                                 &field_descriptor);
  if (status == SS_OK) {
    status = ss_fe_descriptor_find(
        graph->model.dimension, graph->model.coordinate_order, &coordinate_descriptor);
  }
  if (status == SS_OK) {
    status = ss_solver_volume_quadrature(graph, SS_FE_OPERATION_MECHANICS_STIFFNESS,
                                         graph->model.mechanics_order, &quadrature);
  }
  if (status == SS_OK) {
    status = ss_field_space_owned_cell_plan_view(graph->mechanics_space, &packs,
                                                 &pack_count);
  }
  for (pack_index = 0U; status == SS_OK && pack_index < pack_count; ++pack_index) {
    const SsCellPlan *pack = &packs[pack_index];
    const SsMechanicsMaterialRegion *region =
        ss_solver_cell_material_region(graph, pack->material_value);
    double field_values[SS_FIELD_MAX_CLOSURE] = {0.0};
    double element_residual[SS_FIELD_MAX_CLOSURE] = {0.0};
    SsMechanicsElementInput input = {0};
    SsElementOutput output = {0};
    SsErrorRecord element_error = {0};
    size_t closure;
    if (region == NULL || region->parameters == NULL ||
        pack->dimension != graph->model.dimension ||
        pack->components != graph->model.dimension ||
        pack->field_node_count != field_descriptor->node_count ||
        pack->closure_count !=
            field_descriptor->node_count * (size_t)graph->model.dimension ||
        pack->closure_count > SS_FIELD_MAX_CLOSURE ||
        pack->coordinate_node_count != coordinate_descriptor->node_count) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    for (closure = 0U; closure < pack->closure_count; ++closure) {
      if (pack->constrained[closure] != 0U) {
        field_values[closure] = pack->prescribed_values[closure];
      } else if (pack->local_indices[closure] < 0) {
        status = SS_ERROR_INTEGRITY;
        break;
      } else {
        field_values[closure] =
            (double)PetscRealPart(local_values[pack->local_indices[closure]]);
      }
    }
    input.dimension = graph->model.dimension;
    input.field_descriptor = field_descriptor;
    input.coordinate_descriptor = coordinate_descriptor;
    input.quadrature = quadrature;
    input.coordinate_values = pack->coordinate_coordinates;
    input.field_values = field_values;
    input.body_force = region->has_body_force != 0
                           ? region->body_force
                           : mechanics->mechanics_spec.body_force;
    input.coupling_body_force = mechanics->mechanics_spec.fixed_coupling_force;
    input.material = region->parameters;
    input.reduction_factor =
        graph->model.analysis_method == SS_ANALYSIS_LIMIT_LOAD ? 1.0 : factor;
    input.external_load_factor =
        graph->model.analysis_method == SS_ANALYSIS_LIMIT_LOAD ? factor : 1.0;
    input.stable_cell_id = pack->stable_cell_id;
    input.coupling_point_kernel = mechanics->mechanics_spec.coupling_point_kernel;
    input.coupling_point_context = mechanics->mechanics_spec.coupling_point_context;
    input.point_force_mode = mechanics->mechanics_spec.point_force_mode;
    output.closure_capacity = pack->closure_count;
    output.residual = element_residual;
    if (status == SS_OK &&
        ss_assembly_mechanics_element(&input, SS_ASSEMBLY_RESIDUAL, &output,
                                      &element_error) != SS_OK) {
      status = element_error.valid != 0 ? element_error.status : SS_ERROR_INTEGRITY;
      break;
    }
    for (closure = 0U; status == SS_OK && closure < pack->closure_count; ++closure) {
      const uint32_t component = (uint32_t)(closure % graph->model.dimension);
      if (!isfinite(element_residual[closure])) {
        status = SS_ERROR_INTEGRITY;
        break;
      }
      local[graph->model.dimension + component] += element_residual[closure];
      if (pack->constrained[closure] != 0U) {
        local[component] += element_residual[closure];
      }
    }
  }
  if (status == SS_OK) {
    status = ss_solver_mechanics_natural_equilibrium(graph, mechanics, local_values,
                                                     factor, local);
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    if (MPI_Allreduce(local, global, (int)(2U * graph->model.dimension), MPI_DOUBLE,
                      MPI_SUM, graph->communicator) != MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  if (status == SS_OK) {
    status = ss_assembly_plan_create_vector(graph->mechanics_plan, &residual);
  }
  if (status == SS_OK) {
    const SsOperatorState operator_state = {
        .field = solution,
        .reduction_factor =
            graph->model.analysis_method == SS_ANALYSIS_LIMIT_LOAD ? 1.0 : factor,
        .external_load_factor =
            graph->model.analysis_method == SS_ANALYSIS_LIMIT_LOAD ? factor : 1.0,
        .cancelled = 0U,
    };
    status = ss_operator_residual(mechanics->operator_handle, &operator_state, residual,
                                  error);
  }
  if (status == SS_OK) {
    SsAssemblyDiagnostics diagnostics = {0};
    status = ss_operator_diagnostics(mechanics->operator_handle, &diagnostics);
    status = ss_status_allreduce_max(graph->communicator, status);
    if (status == SS_OK &&
        (MPI_Allreduce(&diagnostics.observed_threads,
                       &equilibrium->assembly_minimum_threads, 1, MPI_INT, MPI_MIN,
                       graph->communicator) != MPI_SUCCESS ||
         MPI_Allreduce(&diagnostics.observed_threads,
                       &equilibrium->assembly_maximum_threads, 1, MPI_INT, MPI_MAX,
                       graph->communicator) != MPI_SUCCESS)) {
      status = SS_ERROR_INTERNAL;
    }
  }
  if (status == SS_OK &&
      VecNorm(residual, NORM_2, &free_residual_l2) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (VecDestroy(&residual) != PETSC_SUCCESS && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    double reaction_squared = 0.0;
    double external_squared = 0.0;
    double imbalance_squared = 0.0;
    uint32_t component;
    for (component = 0U; component < graph->model.dimension; ++component) {
      const double reaction = global[component];
      const double external = -global[graph->model.dimension + (size_t)component];
      const double imbalance = reaction + external;
      equilibrium->reaction[component] = reaction;
      equilibrium->external_resultant[component] = external;
      reaction_squared += reaction * reaction;
      external_squared += external * external;
      imbalance_squared += imbalance * imbalance;
    }
    equilibrium->reaction_norm = sqrt(reaction_squared);
    equilibrium->external_resultant_norm = sqrt(external_squared);
    equilibrium->equilibrium_imbalance_norm = sqrt(imbalance_squared);
    equilibrium->normalized_equilibrium_imbalance =
        equilibrium->equilibrium_imbalance_norm /
        fmax(DBL_MIN,
             fmax(equilibrium->reaction_norm, equilibrium->external_resultant_norm));
    equilibrium->free_residual_l2 = (double)free_residual_l2;
    if (!isfinite(equilibrium->reaction_norm) ||
        !isfinite(equilibrium->external_resultant_norm) ||
        !isfinite(equilibrium->equilibrium_imbalance_norm) ||
        !isfinite(equilibrium->normalized_equilibrium_imbalance) ||
        !isfinite(equilibrium->free_residual_l2) ||
        equilibrium->free_residual_l2 < 0.0) {
      status = SS_ERROR_INTEGRITY;
    } else {
      equilibrium->valid = 1;
    }
  }
  return ss_status_allreduce_max(graph->communicator, status);
}
