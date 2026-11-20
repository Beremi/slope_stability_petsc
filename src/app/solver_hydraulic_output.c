/* Hydraulic balance, quadrature data and result acceptance.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/core/sort.h"
#include "slope_internal/mesh/cell_index.h"
#include "solver_output_internal.h"

#include "slope_internal/fe/fe.h"

#include <petsclog.h>

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  SsEntityId stable_cell_id;
  const SsCellPack *pack;
} SsSolverCellPackReference;

static int ss_solver_cell_pack_reference_compare(const void *left_raw,
                                                 const void *right_raw) {
  const SsSolverCellPackReference *left = (const SsSolverCellPackReference *)left_raw;
  const SsSolverCellPackReference *right = (const SsSolverCellPackReference *)right_raw;
  return left->stable_cell_id < right->stable_cell_id   ? -1
         : left->stable_cell_id > right->stable_cell_id ? 1
                                                        : 0;
}

static const SsCellPack *
ss_solver_cell_pack_find(const SsSolverCellPackReference *references, size_t count,
                         SsEntityId stable_cell_id) {
  const size_t index = ss_mesh_cell_index_find(
      references, count, sizeof(*references),
      offsetof(SsSolverCellPackReference, stable_cell_id), stable_cell_id);
  return index < count ? references[index].pack : NULL;
}

static SsStatus ss_solver_pressure_cell_values(const SsCellPack *pack,
                                               const SsFeDescriptor *field_descriptor,
                                               const PetscScalar *local_values,
                                               double *field_values) {
  size_t node;
  if (pack == NULL || field_descriptor == NULL || local_values == NULL ||
      field_values == NULL || pack->components != 1U ||
      pack->dimension != field_descriptor->dimension ||
      pack->order != field_descriptor->order ||
      pack->field_node_count != field_descriptor->node_count ||
      pack->closure_count != field_descriptor->node_count) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (node = 0U; node < field_descriptor->node_count; ++node) {
    if (pack->constrained[node] != 0U) {
      field_values[node] = pack->prescribed_values[node];
    } else if (pack->local_indices[node] < 0) {
      return SS_ERROR_INTEGRITY;
    } else {
      field_values[node] =
          (double)PetscRealPart(local_values[pack->local_indices[node]]);
    }
    if (!isfinite(field_values[node])) {
      return SS_ERROR_INTEGRITY;
    }
  }
  return SS_OK;
}

static int ss_solver_facet_is_pressure_head(const SsFieldFacetPack *facet) {
  size_t slot;
  if (facet == NULL || facet->components != 1U || facet->closure_count == 0U) {
    return 0;
  }
  for (slot = 0U; slot < facet->closure_count; ++slot) {
    if (facet->constrained[slot] == 0U) {
      return 0;
    }
  }
  return 1;
}

/* Add the residual of constrained closure slots to the constraint reaction. */
static SsStatus ss_solver_constraint_reaction_visit(void *context,
                                                    const SsFieldFacetPack *facet,
                                                    const double *element_residual) {
  long double *reaction = (long double *)context;
  size_t slot;
  for (slot = 0U; slot < facet->closure_count; ++slot) {
    if (facet->constrained[slot] != 0U) {
      if (!isfinite(element_residual[slot])) {
        return SS_ERROR_INTEGRITY;
      }
      *reaction += (long double)element_residual[slot];
    }
  }
  return SS_OK;
}

static SsStatus ss_solver_hydraulic_natural_constraint_reaction(
    const SsSolverGraph *graph, const SsSolverHydraulic *hydraulic,
    const PetscScalar *local_values, long double *reaction) {
  if (graph == NULL || hydraulic == NULL || local_values == NULL || reaction == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *reaction = 0.0L;
  return ss_solver_natural_facet_residuals(
      graph, graph->hydraulic_space, graph->model.hydraulic_order, 1U,
      hydraulic->natural_operations, hydraulic->natural_operation_count, local_values,
      ss_solver_constraint_reaction_visit, reaction);
}

SsStatus ss_solver_hydraulic_boundary_balance(const SsSolverGraph *graph,
                                              const SsSolverHydraulic *hydraulic,
                                              const PetscScalar *local_values,
                                              SsHydraulicBalanceLocal *local_balance,
                                              double *reconstructed_natural_flux,
                                              double *reconstructed_dirichlet_flux) {
  const SsFeDescriptor *field_descriptor = NULL;
  const SsFeDescriptor *coordinate_descriptor = NULL;
  const SsFeQuadrature *facet_quadrature = NULL;
  const SsFeQuadrature *volume_quadrature = NULL;
  const SsFeGeometryPolicy geometry_policy = {
      64.0 * DBL_EPSILON,
      1.0 / (64.0 * DBL_EPSILON),
  };
  SsFeQuadratureRequest facet_quadrature_request = {0};
  SsFeQuadraturePlan facet_quadrature_plan = {0};
  SsFeQuadratureRequest volume_quadrature_request = {0};
  SsFeQuadraturePlan volume_quadrature_plan = {0};
  SsCellPack *cell_packs = NULL;
  SsSolverCellPackReference *cell_references = NULL;
  size_t cell_count = 0U;
  size_t cell_index;
  uint64_t local_facet_count = 0U;
  uint64_t global_facet_count = 0U;
  long double natural_flux = 0.0L;
  long double dirichlet_flux = 0.0L;
  long double dirichlet_reaction = 0.0L;
  long double natural_constraint_reaction = 0.0L;
  SsStatus status;
  if (graph == NULL || hydraulic == NULL || local_values == NULL ||
      local_balance == NULL || reconstructed_natural_flux == NULL ||
      reconstructed_dirichlet_flux == NULL || graph->hydraulic_space == NULL ||
      !isfinite(hydraulic->gamma_water) || hydraulic->gamma_water <= 0.0 ||
      graph->model.boundary_count > (size_t)UINT32_MAX - 1U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_fe_descriptor_find(graph->model.dimension, graph->model.hydraulic_order,
                                 &field_descriptor);
  if (status == SS_OK) {
    status = ss_fe_descriptor_find(
        graph->model.dimension, graph->model.coordinate_order, &coordinate_descriptor);
  }
  facet_quadrature_request.parent_dimension = graph->model.dimension;
  facet_quadrature_request.domain = SS_FE_QUADRATURE_FACET;
  facet_quadrature_request.operation = SS_FE_OPERATION_HYDRAULIC_FLUX;
  facet_quadrature_request.field_order = graph->model.hydraulic_order;
  facet_quadrature_request.coordinate_order = graph->model.coordinate_order;
  facet_quadrature_request.coefficient_order = 0U;
  facet_quadrature_request.mode = SS_FE_QUADRATURE_MODE_PAIRED_HIGHER;
  volume_quadrature_request.parent_dimension = graph->model.dimension;
  volume_quadrature_request.domain = SS_FE_QUADRATURE_VOLUME;
  volume_quadrature_request.operation = SS_FE_OPERATION_HYDRAULIC_DIFFUSION;
  volume_quadrature_request.field_order = graph->model.hydraulic_order;
  volume_quadrature_request.coordinate_order = graph->model.coordinate_order;
  volume_quadrature_request.coefficient_order = 0U;
  volume_quadrature_request.mode =
      graph->model.volume_quadrature_mode == SS_FE_QUADRATURE_MODE_PAIRED_HIGHER
          ? SS_FE_QUADRATURE_MODE_PAIRED_HIGHER
          : SS_FE_QUADRATURE_MODE_DEFAULT;
  if (status == SS_OK) {
    status = ss_fe_quadrature_plan(&facet_quadrature_request, &facet_quadrature_plan);
  }
  if (status == SS_OK) {
    status = ss_fe_quadrature_plan(&volume_quadrature_request, &volume_quadrature_plan);
  }
  if (status == SS_OK) {
    facet_quadrature = facet_quadrature_plan.selected_rule;
    volume_quadrature = volume_quadrature_plan.selected_rule;
    if (facet_quadrature == NULL || volume_quadrature == NULL) {
      status = SS_ERROR_INTEGRITY;
    }
  }
  if (status == SS_OK) {
    status =
        ss_field_space_pack_owned_cells(graph->hydraulic_space, NULL, 0U, &cell_count);
  }
  if (status == SS_OK) {
    cell_packs =
        (SsCellPack *)calloc(cell_count == 0U ? 1U : cell_count, sizeof(*cell_packs));
    cell_references = (SsSolverCellPackReference *)calloc(
        cell_count == 0U ? 1U : cell_count, sizeof(*cell_references));
    if (cell_packs == NULL || cell_references == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    status = ss_solver_hydraulic_natural_constraint_reaction(
        graph, hydraulic, local_values, &natural_constraint_reaction);
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    status = ss_field_space_pack_owned_cells(graph->hydraulic_space, cell_packs,
                                             cell_count, &cell_count);
  }
  for (cell_index = 0U; status == SS_OK && cell_index < cell_count; ++cell_index) {
    cell_references[cell_index].stable_cell_id = cell_packs[cell_index].stable_cell_id;
    cell_references[cell_index].pack = &cell_packs[cell_index];
  }
  if (status == SS_OK) {
    ss_sort(cell_references, cell_count, sizeof(*cell_references),
            ss_solver_cell_pack_reference_compare);
    for (cell_index = 1U; cell_index < cell_count; ++cell_index) {
      if (cell_references[cell_index - 1U].stable_cell_id ==
          cell_references[cell_index].stable_cell_id) {
        status = SS_ERROR_DUPLICATE;
        break;
      }
    }
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  for (cell_index = 0U; status == SS_OK && cell_index < cell_count; ++cell_index) {
    const SsCellPack *pack = &cell_packs[cell_index];
    const SsSolverHydraulicCell *cell =
        ss_solver_hydraulic_cell_find(hydraulic, pack->stable_cell_id);
    double field_values[SS_FE_MAX_NODES] = {0.0};
    double element_residual[SS_FE_MAX_NODES] = {0.0};
    SsScalarElementInput input = {0};
    SsElementOutput output = {0};
    SsErrorRecord element_error = {0};
    size_t node;
    if (cell == NULL ||
        ss_solver_pressure_cell_values(pack, field_descriptor, local_values,
                                       field_values) != SS_OK) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    input.dimension = graph->model.dimension;
    input.field_descriptor = field_descriptor;
    input.coordinate_descriptor = coordinate_descriptor;
    input.quadrature = volume_quadrature;
    input.coordinate_values = pack->coordinate_coordinates;
    input.field_values = field_values;
    input.point_kernel = ss_hydraulic_operator_point_kernel;
    input.point_context = hydraulic->model;
    input.stable_cell_id = pack->stable_cell_id;
    output.closure_capacity = field_descriptor->node_count;
    output.residual = element_residual;
    if (ss_assembly_scalar_element(&input, SS_ASSEMBLY_RESIDUAL, &output,
                                   &element_error) != SS_OK) {
      status = element_error.valid != 0 ? element_error.status : SS_ERROR_INTEGRITY;
      break;
    }
    for (node = 0U; node < field_descriptor->node_count; ++node) {
      if (pack->constrained[node] != 0U) {
        dirichlet_reaction += (long double)element_residual[node];
      }
    }
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    const uint32_t support_value = 1U;
    SsFieldFacetPack *facets = NULL;
    size_t facet_count = 0U;
    size_t facet_index;
    status = ss_field_space_pack_owned_facets(graph->hydraulic_space, support_value,
                                              NULL, 0U, &facet_count);
    if (status == SS_OK) {
      facets = (SsFieldFacetPack *)calloc(facet_count == 0U ? 1U : facet_count,
                                          sizeof(*facets));
      if (facets == NULL) {
        status = SS_ERROR_OUT_OF_MEMORY;
      }
    }
    status = ss_status_allreduce_max(graph->communicator, status);
    if (status == SS_OK) {
      status = ss_field_space_pack_owned_facets(graph->hydraulic_space, support_value,
                                                facets, facet_count, &facet_count);
    }
    for (facet_index = 0U; status == SS_OK && facet_index < facet_count;
         ++facet_index) {
      const SsFieldFacetPack *facet = &facets[facet_index];
      const SsCellPack *parent = ss_solver_cell_pack_find(
          cell_references, cell_count, facet->geometry.stable_parent_cell_id);
      const SsSolverHydraulicCell *cell = ss_solver_hydraulic_cell_find(
          hydraulic, facet->geometry.stable_parent_cell_id);
      double field_values[SS_FE_MAX_NODES] = {0.0};
      long double facet_flux = 0.0L;
      size_t quadrature_index;
      if (parent == NULL || cell == NULL || facet->geometry.facet_owned == 0U ||
          facet->geometry.dimension != graph->model.dimension ||
          ss_solver_pressure_cell_values(parent, field_descriptor, local_values,
                                         field_values) != SS_OK) {
        status = SS_ERROR_INTEGRITY;
        break;
      }
      for (quadrature_index = 0U;
           status == SS_OK && quadrature_index < facet_quadrature->point_count;
           ++quadrature_index) {
        const double *facet_reference =
            facet_quadrature->points +
            quadrature_index * (size_t)(graph->model.dimension - 1U);
        double parent_reference[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
        double basis[SS_FE_MAX_NODES] = {0.0};
        double reference_gradients[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
        double physical_gradients[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
        double pressure = 0.0;
        double pressure_gradient[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
        double signed_flux = 0.0;
        SsFeGeometry geometry = {0};
        SsFacetSample facet_geometry = {0};
        SsHydraulicPointModel point_model = {
            graph->model.dimension, 1U, hydraulic->gamma_water, cell->conductivity,
            cell->epsilon,
        };
        SsHydraulicPointResult point_result = {0};
        size_t node;
        uint32_t coordinate;
        if (ss_fe_facet_parent_reference(graph->model.dimension,
                                         facet->geometry.facet_index, facet_reference,
                                         parent_reference) != SS_OK ||
            ss_fe_basis_eval_scalar(field_descriptor, parent_reference, basis,
                                    reference_gradients) != SS_OK ||
            ss_fe_geometry_eval(
                coordinate_descriptor, facet->geometry.coordinate_coordinates,
                parent_reference, &geometry_policy, &geometry) != SS_OK ||
            ss_fe_physical_gradients(field_descriptor, &geometry, reference_gradients,
                                     physical_gradients) != SS_OK ||
            ss_mesh_facet_geometry_at(&facet->geometry, facet_reference,
                                      &facet_geometry) != SS_OK) {
          status = SS_ERROR_INTEGRITY;
          break;
        }
        for (node = 0U; node < field_descriptor->node_count; ++node) {
          for (coordinate = 0U; coordinate < graph->model.dimension; ++coordinate) {
            pressure_gradient[coordinate] +=
                physical_gradients[node * (size_t)graph->model.dimension + coordinate] *
                field_values[node];
          }
          pressure += basis[node] * field_values[node];
        }
        if (!isfinite(pressure)) {
          status = SS_ERROR_INTEGRITY;
          break;
        }
        if (ss_hydraulic_point_evaluate(&point_model, pressure, pressure_gradient,
                                        &point_result) != SS_OK) {
          status = SS_ERROR_INTEGRITY;
          break;
        }
        for (coordinate = 0U; coordinate < graph->model.dimension; ++coordinate) {
          signed_flux += point_result.pressure_flux[coordinate] *
                         facet_geometry.outward_normal[coordinate];
        }
        if (!isfinite(signed_flux) || !isfinite(facet_geometry.measure) ||
            facet_geometry.measure <= 0.0 ||
            !isfinite(facet_quadrature->weights[quadrature_index])) {
          status = SS_ERROR_INTEGRITY;
          break;
        }
        facet_flux += (long double)facet_quadrature->weights[quadrature_index] *
                      (long double)facet_geometry.measure * (long double)signed_flux;
      }
      if (status != SS_OK || !isfinite((double)facet_flux)) {
        if (status == SS_OK) {
          status = SS_ERROR_INTEGRITY;
        }
        break;
      }
      if (ss_solver_facet_is_pressure_head(facet) != 0) {
        dirichlet_flux += facet_flux;
      } else {
        natural_flux += facet_flux;
      }
      ++local_facet_count;
    }
    free(facets);
    status = ss_status_allreduce_max(graph->communicator, status);
  }
  if (status == SS_OK &&
      MPI_Allreduce(&local_facet_count, &global_facet_count, 1, MPI_UINT64_T, MPI_SUM,
                    graph->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK &&
      global_facet_count != graph->raw.input.global_boundary_facet_count) {
    status = SS_ERROR_INTEGRITY;
  }
  if (status == SS_OK) {
    const double local_reconstructed[2] = {
        (double)natural_flux,
        (double)dirichlet_flux,
    };
    double global_reconstructed[2] = {0.0, 0.0};
    if (MPI_Allreduce(local_reconstructed, global_reconstructed, 2, MPI_DOUBLE, MPI_SUM,
                      graph->communicator) != MPI_SUCCESS ||
        !isfinite(global_reconstructed[0]) || !isfinite(global_reconstructed[1])) {
      status = SS_ERROR_INTERNAL;
    } else {
      *reconstructed_natural_flux = global_reconstructed[0];
      *reconstructed_dirichlet_flux = global_reconstructed[1];
    }
  }
  if (status == SS_OK) {
    local_balance->dirichlet_reaction_flux =
        (double)(dirichlet_reaction + natural_constraint_reaction);
  }
  free(cell_references);
  free(cell_packs);
  return ss_status_allreduce_max(graph->communicator, status);
}

SsStatus ss_solver_hydraulic_quadrature_output(
    const SsSolverGraph *graph, const SsSolverHydraulic *hydraulic,
    const PetscScalar *local_values, SsSolverOutputField *gradient,
    SsSolverOutputField *conductivity, SsSolverOutputField *ramp,
    SsSolverOutputField *epsilon, SsSolverOutputField *pressure_flux,
    SsSolverOutputField *darcy_velocity, SsSolverOutputCategory *saturation) {
  const SsFeDescriptor *field_descriptor = NULL;
  const SsFeDescriptor *coordinate_descriptor = NULL;
  const SsFeQuadrature *quadrature = NULL;
  const SsFeGeometryPolicy geometry_policy = {
      64.0 * DBL_EPSILON,
      1.0 / (64.0 * DBL_EPSILON),
  };
  SsCellPack *packs = NULL;
  SsSolverOutputField *fields[6];
  size_t pack_count = 0U;
  size_t pack_index;
  size_t field_index;
  uint64_t local_count;
  uint64_t global_count = 0U;
  SsStatus status;
  if (graph == NULL || hydraulic == NULL || local_values == NULL || gradient == NULL ||
      conductivity == NULL || ramp == NULL || epsilon == NULL ||
      pressure_flux == NULL || darcy_velocity == NULL || saturation == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  fields[0] = gradient;
  fields[1] = conductivity;
  fields[2] = ramp;
  fields[3] = epsilon;
  fields[4] = pressure_flux;
  fields[5] = darcy_velocity;
  for (field_index = 0U; field_index < sizeof(fields) / sizeof(fields[0]);
       ++field_index) {
    memset(fields[field_index], 0, sizeof(*fields[field_index]));
  }
  memset(saturation, 0, sizeof(*saturation));
  gradient->component_count = graph->model.dimension;
  conductivity->component_count = 1U;
  ramp->component_count = 1U;
  epsilon->component_count = 1U;
  pressure_flux->component_count = graph->model.dimension;
  darcy_velocity->component_count = graph->model.dimension;
  saturation->category_count = 2U;
  status = ss_fe_descriptor_find(graph->model.dimension, graph->model.hydraulic_order,
                                 &field_descriptor);
  if (status == SS_OK) {
    status = ss_fe_descriptor_find(
        graph->model.dimension, graph->model.coordinate_order, &coordinate_descriptor);
  }
  if (status == SS_OK) {
    status = ss_solver_volume_quadrature(graph, SS_FE_OPERATION_HYDRAULIC_DIFFUSION,
                                         graph->model.hydraulic_order, &quadrature);
  }
  if (status != SS_OK) {
    return status;
  }
  for (field_index = 0U; field_index < sizeof(fields) / sizeof(fields[0]);
       ++field_index) {
    fields[field_index]->sample_count = (uint32_t)quadrature->point_count;
  }
  saturation->sample_count = (uint32_t)quadrature->point_count;
  {
    size_t index;
    const size_t length = strlen(quadrature->id);
    if (length == 0U || length >= sizeof(saturation->quadrature_id)) {
      return SS_ERROR_LIMIT;
    }
    for (field_index = 0U; field_index < sizeof(fields) / sizeof(fields[0]);
         ++field_index) {
      if (length >= sizeof(fields[field_index]->quadrature_id)) {
        return SS_ERROR_LIMIT;
      }
    }
    for (index = 0U; index <= length; ++index) {
      const char value = quadrature->id[index] == '-' ? '_' : quadrature->id[index];
      for (field_index = 0U; field_index < sizeof(fields) / sizeof(fields[0]);
           ++field_index) {
        fields[field_index]->quadrature_id[index] = value;
      }
      saturation->quadrature_id[index] = value;
    }
  }
  status =
      ss_field_space_pack_owned_cells(graph->hydraulic_space, NULL, 0U, &pack_count);
  if (status != SS_OK) {
    return status;
  }
  size_t sample_count;
  size_t field_value_counts[sizeof(fields) / sizeof(fields[0])];
  if (ss_checked_mul_size(pack_count, quadrature->point_count, &sample_count) !=
      SS_OK) {
    return SS_ERROR_OVERFLOW;
  }
  for (field_index = 0U; field_index < sizeof(fields) / sizeof(fields[0]);
       ++field_index) {
    if (ss_checked_mul_size(sample_count, fields[field_index]->component_count,
                            &field_value_counts[field_index]) != SS_OK) {
      return SS_ERROR_OVERFLOW;
    }
  }
  packs = (SsCellPack *)calloc(pack_count == 0U ? 1U : pack_count, sizeof(*packs));
  for (field_index = 0U; field_index < sizeof(fields) / sizeof(fields[0]);
       ++field_index) {
    SsSolverOutputField *field = fields[field_index];
    field->ids =
        (SsEntityId *)calloc(pack_count == 0U ? 1U : pack_count, sizeof(*field->ids));
    field->slots =
        (size_t *)calloc(pack_count == 0U ? 1U : pack_count, sizeof(*field->slots));
    field->values =
        (double *)calloc(sample_count == 0U ? 1U : field_value_counts[field_index],
                         sizeof(*field->values));
  }
  saturation->ids = (SsEntityId *)calloc(pack_count == 0U ? 1U : pack_count,
                                         sizeof(*saturation->ids));
  saturation->slots =
      (size_t *)calloc(pack_count == 0U ? 1U : pack_count, sizeof(*saturation->slots));
  saturation->values = (uint32_t *)calloc(sample_count == 0U ? 1U : sample_count,
                                          sizeof(*saturation->values));
  if (packs == NULL || saturation->ids == NULL || saturation->slots == NULL ||
      saturation->values == NULL) {
    status = SS_ERROR_OUT_OF_MEMORY;
  }
  for (field_index = 0U; status != SS_ERROR_OUT_OF_MEMORY &&
                         field_index < sizeof(fields) / sizeof(fields[0]);
       ++field_index) {
    if (fields[field_index]->ids == NULL || fields[field_index]->slots == NULL ||
        fields[field_index]->values == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  if (status == SS_ERROR_OUT_OF_MEMORY) {
    free(packs);
    ss_solver_output_category_destroy(saturation);
    for (field_index = 0U; field_index < sizeof(fields) / sizeof(fields[0]);
         ++field_index) {
      ss_solver_output_field_destroy(fields[field_index]);
    }
    return SS_ERROR_OUT_OF_MEMORY;
  }
  status = ss_field_space_pack_owned_cells(graph->hydraulic_space, packs, pack_count,
                                           &pack_count);
  for (pack_index = 0U; status == SS_OK && pack_index < pack_count; ++pack_index) {
    const SsCellPack *pack = &packs[pack_index];
    const SsSolverHydraulicCell *cell =
        ss_solver_hydraulic_cell_find(hydraulic, pack->stable_cell_id);
    double field_values[SS_FE_MAX_NODES] = {0.0};
    double integrated_pressure = 0.0;
    double physical_measure = 0.0;
    size_t node;
    size_t quadrature_index;
    if (cell == NULL || pack->components != 1U ||
        pack->closure_count != field_descriptor->node_count) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    for (node = 0U; node < field_descriptor->node_count; ++node) {
      if (pack->constrained[node] != 0U) {
        field_values[node] = pack->prescribed_values[node];
      } else if (pack->local_indices[node] < 0) {
        status = SS_ERROR_INTEGRITY;
        break;
      } else {
        field_values[node] =
            (double)PetscRealPart(local_values[pack->local_indices[node]]);
      }
    }
    for (field_index = 0U; field_index < sizeof(fields) / sizeof(fields[0]);
         ++field_index) {
      fields[field_index]->ids[pack_index] = pack->stable_cell_id;
      fields[field_index]->slots[pack_index] = pack_index;
    }
    saturation->ids[pack_index] = pack->stable_cell_id;
    saturation->slots[pack_index] = pack_index;
    for (quadrature_index = 0U;
         status == SS_OK && quadrature_index < quadrature->point_count;
         ++quadrature_index) {
      const double *reference =
          quadrature->points + quadrature_index * (size_t)graph->model.dimension;
      double basis[SS_FE_MAX_NODES] = {0.0};
      double reference_gradients[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
      double physical_gradients[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
      double pressure = 0.0;
      double pressure_gradient[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
      SsHydraulicPointModel point_model = {
          graph->model.dimension, 1U, hydraulic->gamma_water, cell->conductivity,
          cell->epsilon,
      };
      SsHydraulicPointResult point_result = {0};
      SsFeGeometry geometry = {0};
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
        uint32_t coordinate;
        pressure += basis[node] * field_values[node];
        for (coordinate = 0U; coordinate < graph->model.dimension; ++coordinate) {
          pressure_gradient[coordinate] +=
              physical_gradients[node * (size_t)graph->model.dimension + coordinate] *
              field_values[node];
        }
      }
      if (!isfinite(pressure)) {
        status = SS_ERROR_INTEGRITY;
        break;
      }
      if (ss_hydraulic_point_evaluate(&point_model, pressure, pressure_gradient,
                                      &point_result) != SS_OK) {
        status = SS_ERROR_INTEGRITY;
        break;
      }
      {
        const double physical_weight =
            quadrature->weights[quadrature_index] * geometry.determinant;
        /*
         * A valid simplex rule may contain negative weights (the default
         * tetrahedral P2 rule does).  Geometry validity is checked by
         * ss_fe_geometry_eval, and the accumulated physical measure is
         * checked below, so reject only a non-finite weighted sample here.
         */
        if (!isfinite(physical_weight)) {
          status = SS_ERROR_INTEGRITY;
          break;
        }
        integrated_pressure += pressure * physical_weight;
        physical_measure += physical_weight;
      }
      {
        const size_t sample = pack_index * quadrature->point_count + quadrature_index;
        double *gradient_output =
            gradient->values + sample * (size_t)gradient->component_count;
        double *pressure_flux_output =
            pressure_flux->values + sample * (size_t)pressure_flux->component_count;
        double *darcy_velocity_output =
            darcy_velocity->values + sample * (size_t)darcy_velocity->component_count;
        uint32_t coordinate;
        conductivity->values[sample] = cell->conductivity;
        ramp->values[sample] = point_result.ramp.value;
        epsilon->values[sample] = cell->epsilon;
        for (coordinate = 0U; coordinate < graph->model.dimension; ++coordinate) {
          /*
           * The hydraulic kernel stores the positive weak-form vector
           * k(grad(p) + gamma_w*kappa*e_y).  The result contract defines
           * pressure flux and Darcy velocity with the conventional leading
           * minus sign.
           */
          const double conventional_flux = -point_result.pressure_flux[coordinate];
          const double velocity = conventional_flux / hydraulic->gamma_water;
          if (!isfinite(pressure_gradient[coordinate]) ||
              !isfinite(conventional_flux) || !isfinite(velocity)) {
            status = SS_ERROR_INTEGRITY;
            break;
          }
          gradient_output[coordinate] = pressure_gradient[coordinate];
          pressure_flux_output[coordinate] = conventional_flux;
          darcy_velocity_output[coordinate] = velocity;
        }
      }
    }
    if (status == SS_OK) {
      uint32_t saturated;
      if (!isfinite(integrated_pressure) || !isfinite(physical_measure) ||
          physical_measure <= 0.0) {
        status = SS_ERROR_INTEGRITY;
        break;
      }
      saturated =
          integrated_pressure / physical_measure >= 0.1 * cell->epsilon ? 1U : 0U;
      /*
       * Saturation is a cell classification based on the physical-volume
       * mean pressure.  Repeat it over the quadrature samples so the
       * categorical result shares the quadrature layout without changing its
       * cell-level meaning.
       */
      for (quadrature_index = 0U; quadrature_index < quadrature->point_count;
           ++quadrature_index) {
        saturation->values[pack_index * quadrature->point_count + quadrature_index] =
            saturated;
      }
    }
  }
  free(packs);
  if (status != SS_OK) {
    ss_solver_output_category_destroy(saturation);
    for (field_index = 0U; field_index < sizeof(fields) / sizeof(fields[0]);
         ++field_index) {
      ss_solver_output_field_destroy(fields[field_index]);
    }
    return status;
  }
  for (field_index = 0U; field_index < sizeof(fields) / sizeof(fields[0]);
       ++field_index) {
    fields[field_index]->local_count = pack_count;
  }
  saturation->local_count = pack_count;
  local_count = (uint64_t)pack_count;
  if (MPI_Allreduce(&local_count, &global_count, 1, MPI_UINT64_T, MPI_SUM,
                    graph->communicator) != MPI_SUCCESS) {
    ss_solver_output_category_destroy(saturation);
    for (field_index = 0U; field_index < sizeof(fields) / sizeof(fields[0]);
         ++field_index) {
      ss_solver_output_field_destroy(fields[field_index]);
    }
    return SS_ERROR_INTERNAL;
  }
  for (field_index = 0U; field_index < sizeof(fields) / sizeof(fields[0]);
       ++field_index) {
    fields[field_index]->global_count = global_count;
  }
  saturation->global_count = global_count;
  return global_count == graph->raw.input.global_cell_count ? SS_OK
                                                            : SS_ERROR_INTEGRITY;
}

SsStatus ss_solver_coupled_body_force_output(const SsSolverGraph *graph,
                                             const SsSolverCouplingSnapshot *snapshot,
                                             SsSolverOutputField *field) {
  const SsFeDescriptor *coordinate_descriptor = NULL;
  const SsFeQuadrature *quadrature = NULL;
  const SsFeGeometryPolicy geometry_policy = {
      64.0 * DBL_EPSILON,
      1.0 / (64.0 * DBL_EPSILON),
  };
  SsCellPack *packs = NULL;
  size_t pack_count = 0U;
  size_t pack_index;
  uint64_t local_count;
  uint64_t global_count = 0U;
  SsStatus status;
  if (graph == NULL || snapshot == NULL || field == NULL ||
      graph->mechanics_space == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(field, 0, sizeof(*field));
  field->component_count = graph->model.dimension;
  status = ss_fe_descriptor_find(graph->model.dimension, graph->model.coordinate_order,
                                 &coordinate_descriptor);
  if (status == SS_OK) {
    status = ss_solver_volume_quadrature(graph, SS_FE_OPERATION_MECHANICS_STIFFNESS,
                                         graph->model.mechanics_order, &quadrature);
  }
  if (status != SS_OK || quadrature == NULL || quadrature->point_count == 0U ||
      quadrature->point_count > UINT32_MAX) {
    return status == SS_OK ? SS_ERROR_INTEGRITY : status;
  }
  field->sample_count = (uint32_t)quadrature->point_count;
  {
    size_t index;
    const size_t length = strlen(quadrature->id);
    if (length == 0U || length >= sizeof(field->quadrature_id)) {
      return SS_ERROR_LIMIT;
    }
    for (index = 0U; index <= length; ++index) {
      field->quadrature_id[index] =
          quadrature->id[index] == '-' ? '_' : quadrature->id[index];
    }
  }
  status =
      ss_field_space_pack_owned_cells(graph->mechanics_space, NULL, 0U, &pack_count);
  if (status != SS_OK) {
    return status;
  }
  size_t value_count;
  if (ss_checked_product_size(
          (size_t[]){pack_count, quadrature->point_count, field->component_count}, 3U,
          &value_count) != SS_OK) {
    return SS_ERROR_OVERFLOW;
  }
  packs = (SsCellPack *)calloc(pack_count == 0U ? 1U : pack_count, sizeof(*packs));
  field->ids =
      (SsEntityId *)calloc(pack_count == 0U ? 1U : pack_count, sizeof(*field->ids));
  field->slots =
      (size_t *)calloc(pack_count == 0U ? 1U : pack_count, sizeof(*field->slots));
  field->values =
      (double *)calloc(value_count == 0U ? 1U : value_count, sizeof(*field->values));
  if (packs == NULL || field->ids == NULL || field->slots == NULL ||
      field->values == NULL) {
    free(packs);
    ss_solver_output_field_destroy(field);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  status = ss_field_space_pack_owned_cells(graph->mechanics_space, packs, pack_count,
                                           &pack_count);
  for (pack_index = 0U; status == SS_OK && pack_index < pack_count; ++pack_index) {
    const SsCellPack *pack = &packs[pack_index];
    size_t quadrature_index;
    if (pack->dimension != graph->model.dimension ||
        pack->coordinate_node_count != coordinate_descriptor->node_count) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    field->ids[pack_index] = pack->stable_cell_id;
    field->slots[pack_index] = pack_index;
    for (quadrature_index = 0U;
         status == SS_OK && quadrature_index < quadrature->point_count;
         ++quadrature_index) {
      const double *reference =
          quadrature->points + quadrature_index * (size_t)graph->model.dimension;
      double *output =
          field->values + (pack_index * quadrature->point_count + quadrature_index) *
                              (size_t)field->component_count;
      SsFeGeometry geometry = {0};
      if (ss_fe_geometry_eval(coordinate_descriptor, pack->coordinate_coordinates,
                              reference, &geometry_policy, &geometry) != SS_OK ||
          ss_solver_coupling_snapshot_mechanics_force(
              snapshot, graph->model.dimension, pack->stable_cell_id,
              (uint32_t)quadrature_index, geometry.physical, output) != SS_OK) {
        status = SS_ERROR_INTEGRITY;
      }
    }
  }
  free(packs);
  if (status != SS_OK) {
    ss_solver_output_field_destroy(field);
    return status;
  }
  field->local_count = pack_count;
  local_count = (uint64_t)pack_count;
  if (MPI_Allreduce(&local_count, &global_count, 1, MPI_UINT64_T, MPI_SUM,
                    graph->communicator) != MPI_SUCCESS) {
    ss_solver_output_field_destroy(field);
    return SS_ERROR_INTERNAL;
  }
  field->global_count = global_count;
  return global_count == graph->raw.input.global_cell_count ? SS_OK
                                                            : SS_ERROR_INTEGRITY;
}

SsStatus ss_solver_hydraulic_balance_evaluate(
    const SsSolverGraph *graph, SsSolverHydraulic *hydraulic, Vec solution,
    const SsNonlinearSolveRecord *solve_record, SsHydraulicBalanceLocal *local_balance,
    SsHydraulicBalance *balance, SsErrorRecord *error) {
  const SsOperatorState operator_state = {
      .field = solution,
      .reduction_factor = 1.0,
      .external_load_factor = 1.0,
      .cancelled = 0U,
  };
  Vec residual = NULL;
  SsAssemblyDiagnostics diagnostics = {0};
  PetscReal residual_norm = NAN;
  SsStatus status = ss_assembly_plan_create_vector(graph->hydraulic_plan, &residual);
  if (status == SS_OK) {
    status = ss_operator_residual(hydraulic->operator_handle, &operator_state, residual,
                                  error);
  }
  if (status == SS_OK &&
      (VecNorm(residual, NORM_2, &residual_norm) != PETSC_SUCCESS ||
       ss_operator_diagnostics(hydraulic->operator_handle, &diagnostics) != SS_OK)) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    if (graph->rank == 0) {
      local_balance->prescribed_boundary_flux = diagnostics.natural_signed_flux;
      local_balance->free_residual_l2_squared =
          (double)residual_norm * (double)residual_norm;
    }
    status = ss_hydraulic_balance(
        graph->communicator, local_balance, graph->policy.conservation_physical_scale,
        graph->policy.conservation_normalized_tolerance,
        (double)hydraulic->nonlinear_policy.tolerance * (double)solve_record->load_norm,
        balance);
  }
  if (VecDestroy(&residual) != PETSC_SUCCESS && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  return status;
}

SsStatus ss_solver_hydraulic_acceptance(const SsSolverGraph *graph,
                                        SsSolverHydraulic *hydraulic, Vec solution,
                                        const SsNonlinearSolveRecord *solve_record,
                                        SsSolverHydraulicEvidence *evidence,
                                        SsErrorRecord *error) {
  Vec local_vector = NULL;
  const PetscScalar *local_values = NULL;
  SsHydraulicBalanceLocal local_balance = {0};
  SsStatus status;
  if (graph == NULL || hydraulic == NULL || solution == NULL || solve_record == NULL ||
      evidence == NULL || error == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(evidence, 0, sizeof(*evidence));
  status = ss_field_space_extract_local(graph->hydraulic_space, solution, &local_vector,
                                        &local_values);
  if (status == SS_OK) {
    status = ss_solver_hydraulic_boundary_balance(
        graph, hydraulic, local_values, &local_balance,
        &evidence->reconstructed_natural_flux, &evidence->reconstructed_dirichlet_flux);
  }
  if (local_values != NULL &&
      VecRestoreArrayRead(local_vector, &local_values) != PETSC_SUCCESS &&
      status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (VecDestroy(&local_vector) != PETSC_SUCCESS && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    status =
        ss_solver_hydraulic_balance_evaluate(graph, hydraulic, solution, solve_record,
                                             &local_balance, &evidence->balance, error);
  }
  if (status == SS_OK && evidence->balance.accepted == 0) {
    char message[SS_ERROR_MESSAGE_CAPACITY];
    const int written =
        snprintf(message, sizeof(message),
                 "hydraulic balance rejected: normalized=%.8g limit=%.8g "
                 "free_residual=%.8g limit=%.8g",
                 evidence->balance.normalized_imbalance,
                 graph->policy.conservation_normalized_tolerance,
                 evidence->balance.free_residual_l2,
                 (double)hydraulic->nonlinear_policy.tolerance *
                     (double)solve_record->load_norm);
    status = SS_ERROR_NUMERICAL;
    ss_solver_error(error, status, "HYDRAULIC_BALANCE_REJECTED",
                    written >= 0 && (size_t)written < sizeof(message)
                        ? message
                        : "hydraulic balance or free residual was rejected",
                    3U);
  }
  if (status == SS_OK) {
    evidence->initial_residual_norm = solve_record->initial_residual_norm;
    evidence->final_residual_norm = solve_record->final_residual_norm;
    evidence->load_norm = solve_record->load_norm;
    evidence->nonlinear_outcome = solve_record->outcome;
    evidence->initialization = hydraulic->initialization;
    evidence->valid = 1;
  }
  return status;
}
