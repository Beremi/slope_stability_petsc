/* Single-process assembly tests: sparsity pattern, element kernels, lifting
 * and memory budget; assembly_matrix_free_cases.c holds the matrix-free
 * action cases.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "assembly_internal.h"

static const SsEntityId square_vertex_ids[] = {
    UINT64_C(0x310000000001),
    UINT64_C(0x310000000002),
    UINT64_C(0x310000000003),
    UINT64_C(0x310000000004),
};
static const double square_coordinates[] = {
    0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
};
static const uint64_t square_cells[] = {
    0U, 1U, 2U, 1U, 3U, 2U,
};
static const SsEntityId square_cell_ids[] = {
    UINT64_C(0x320000000001),
    UINT64_C(0x320000000002),
};
static const uint64_t square_facets[] = {
    0U, 1U, 1U, 3U, 3U, 2U, 2U, 0U,
};
static const SsEntityId square_facet_ids[] = {
    UINT64_C(0x330000000001),
    UINT64_C(0x330000000002),
    UINT64_C(0x330000000003),
    UINT64_C(0x330000000004),
};
static const SsMeshIntermediateEntity square_intermediate_entities[] = {
    {1U,
     UINT64_C(0x330000000001),
     {UINT64_C(0x310000000001), UINT64_C(0x310000000002), 0U}},
    {1U,
     UINT64_C(0x330000000004),
     {UINT64_C(0x310000000001), UINT64_C(0x310000000003), 0U}},
    {1U,
     UINT64_C(0x340000000001),
     {UINT64_C(0x310000000002), UINT64_C(0x310000000003), 0U}},
    {1U,
     UINT64_C(0x330000000002),
     {UINT64_C(0x310000000002), UINT64_C(0x310000000004), 0U}},
    {1U,
     UINT64_C(0x330000000003),
     {UINT64_C(0x310000000003), UINT64_C(0x310000000004), 0U}},
};
static const SsMeshLabelAssignment square_labels[] = {
    {SS_MESH_LABEL_MATERIAL, UINT64_C(0x320000000001), 1U},
    {SS_MESH_LABEL_MATERIAL, UINT64_C(0x320000000002), 1U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0x330000000001), 1U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0x330000000002), 1U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0x330000000003), 1U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0x330000000004), 1U},
};

static const SsEntityId tetra_vertex_ids[] = {
    UINT64_C(0x410000000001),
    UINT64_C(0x410000000002),
    UINT64_C(0x410000000003),
    UINT64_C(0x410000000004),
};
static const double tetra_coordinates[] = {
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
};
static const uint64_t tetra_cells[] = {0U, 1U, 2U, 3U};
static const SsEntityId tetra_cell_ids[] = {UINT64_C(0x420000000001)};
static const uint64_t tetra_facets[] = {
    0U, 1U, 2U, 0U, 1U, 3U, 0U, 2U, 3U, 1U, 2U, 3U,
};
static const SsEntityId tetra_facet_ids[] = {
    UINT64_C(0x430000000001),
    UINT64_C(0x430000000002),
    UINT64_C(0x430000000003),
    UINT64_C(0x430000000004),
};
static const SsMeshIntermediateEntity tetra_intermediate_entities[] = {
    {1U,
     UINT64_C(0x440000000001),
     {UINT64_C(0x410000000001), UINT64_C(0x410000000002), 0U}},
    {1U,
     UINT64_C(0x440000000002),
     {UINT64_C(0x410000000001), UINT64_C(0x410000000003), 0U}},
    {1U,
     UINT64_C(0x440000000003),
     {UINT64_C(0x410000000001), UINT64_C(0x410000000004), 0U}},
    {1U,
     UINT64_C(0x440000000004),
     {UINT64_C(0x410000000002), UINT64_C(0x410000000003), 0U}},
    {1U,
     UINT64_C(0x440000000005),
     {UINT64_C(0x410000000002), UINT64_C(0x410000000004), 0U}},
    {1U,
     UINT64_C(0x440000000006),
     {UINT64_C(0x410000000003), UINT64_C(0x410000000004), 0U}},
    {2U,
     UINT64_C(0x430000000001),
     {UINT64_C(0x410000000001), UINT64_C(0x410000000002), UINT64_C(0x410000000003)}},
    {2U,
     UINT64_C(0x430000000002),
     {UINT64_C(0x410000000001), UINT64_C(0x410000000002), UINT64_C(0x410000000004)}},
    {2U,
     UINT64_C(0x430000000003),
     {UINT64_C(0x410000000001), UINT64_C(0x410000000003), UINT64_C(0x410000000004)}},
    {2U,
     UINT64_C(0x430000000004),
     {UINT64_C(0x410000000002), UINT64_C(0x410000000003), UINT64_C(0x410000000004)}},
};
static const SsMeshLabelAssignment tetra_labels[] = {
    {SS_MESH_LABEL_MATERIAL, UINT64_C(0x420000000001), 1U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0x430000000001), 1U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0x430000000002), 1U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0x430000000003), 1U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0x430000000004), 1U},
};

SsMeshInput square_input(void) {
  const SsMeshInput input = {
      .dimension = 2U,
      .global_vertex_count = 4U,
      .global_cell_count = 2U,
      .global_boundary_facet_count = 4U,
      .global_intermediate_entity_count = sizeof(square_intermediate_entities) /
                                          sizeof(square_intermediate_entities[0]),
      .global_coordinate_node_count = 4U,
      .vertex_count = 4U,
      .cell_count = 2U,
      .boundary_facet_count = 4U,
      .vertex_ids = square_vertex_ids,
      .vertex_coordinates = square_coordinates,
      .cell_ids = square_cell_ids,
      .cells = square_cells,
      .boundary_facet_ids = square_facet_ids,
      .boundary_facets = square_facets,
      .intermediate_entity_count = sizeof(square_intermediate_entities) /
                                   sizeof(square_intermediate_entities[0]),
      .intermediate_entities = square_intermediate_entities,
      .label_count = sizeof(square_labels) / sizeof(square_labels[0]),
      .labels = square_labels,
      .coordinate_order = 1U,
      .coordinate_node_count = 4U,
      .coordinate_node_ids = square_vertex_ids,
      .coordinate_coordinates = square_coordinates,
      .cell_coordinate_nodes = square_cells,
      .absolute_determinant_tolerance = 1.0e-12,
      .relative_determinant_tolerance = 1.0e-12,
  };
  return input;
}

SsMeshInput tetra_input(void) {
  const SsMeshInput input = {
      .dimension = 3U,
      .global_vertex_count = 4U,
      .global_cell_count = 1U,
      .global_boundary_facet_count = 4U,
      .global_intermediate_entity_count =
          sizeof(tetra_intermediate_entities) / sizeof(tetra_intermediate_entities[0]),
      .global_coordinate_node_count = 4U,
      .vertex_count = 4U,
      .cell_count = 1U,
      .boundary_facet_count = 4U,
      .vertex_ids = tetra_vertex_ids,
      .vertex_coordinates = tetra_coordinates,
      .cell_ids = tetra_cell_ids,
      .cells = tetra_cells,
      .boundary_facet_ids = tetra_facet_ids,
      .boundary_facets = tetra_facets,
      .intermediate_entity_count =
          sizeof(tetra_intermediate_entities) / sizeof(tetra_intermediate_entities[0]),
      .intermediate_entities = tetra_intermediate_entities,
      .label_count = sizeof(tetra_labels) / sizeof(tetra_labels[0]),
      .labels = tetra_labels,
      .coordinate_order = 1U,
      .coordinate_node_count = 4U,
      .coordinate_node_ids = tetra_vertex_ids,
      .coordinate_coordinates = tetra_coordinates,
      .cell_coordinate_nodes = tetra_cells,
      .absolute_determinant_tolerance = 1.0e-12,
      .relative_determinant_tolerance = 1.0e-12,
  };
  return input;
}

static size_t sorted_unique(PetscInt *values, size_t count) {
  size_t index;
  size_t unique = 0U;
  for (index = 0U; index < count; ++index) {
    size_t right;
    for (right = index + 1U; right < count; ++right) {
      if (values[right] < values[index]) {
        const PetscInt swap = values[index];
        values[index] = values[right];
        values[right] = swap;
      }
    }
    if (unique == 0U || values[index] != values[unique - 1U]) {
      values[unique++] = values[index];
    }
  }
  return unique;
}

static void simplex_reference_node(const SsFeDescriptor *descriptor, size_t node,
                                   double reference[SS_FE_MAX_DIMENSION]) {
  uint32_t coordinate;
  for (coordinate = 0U; coordinate < descriptor->dimension; ++coordinate) {
    reference[coordinate] =
        (double)descriptor
            ->barycentric_indices[node * (size_t)(descriptor->dimension + 1U) +
                                  coordinate + 1U] /
        (double)descriptor->order;
  }
}

static void curved_coordinate(uint32_t dimension, const double *reference,
                              double physical[SS_FE_MAX_DIMENSION]) {
  const double r = reference[0];
  const double s = reference[1];
  physical[0] = r + 0.08 * r * s;
  physical[1] = s + 0.06 * r * s;
  if (dimension == 3U) {
    const double t = reference[2];
    physical[0] += 0.03 * s * t;
    physical[1] += 0.02 * r * t;
    physical[2] = t + 0.05 * r * t + 0.04 * s * t;
  }
}

static void mechanics_field(uint32_t dimension, const double *physical,
                            double *values) {
  const double x = physical[0];
  const double y = physical[1];
  values[0] = 7.0e-4 * x + 2.0e-4 * y + 5.0e-5 * x * y;
  values[1] = -3.0e-4 * x + 8.0e-4 * y + 4.0e-5 * y * y;
  if (dimension == 3U) {
    const double z = physical[2];
    values[0] += 3.0e-5 * z;
    values[1] -= 2.0e-5 * z;
    values[2] = 2.0e-4 * x - 1.0e-4 * y + 6.0e-4 * z + 3.0e-5 * x * z;
  }
}

typedef struct {
  double conductivity;
  double water_unit_weight;
  double epsilon;
  uint32_t upward_axis;
} HydraulicPointContext;

typedef struct {
  double conductivity;
  double reaction;
  double source;
} LinearScalarContext;

typedef struct {
  double offset;
  double derivative;
} AffineNaturalContext;

static SsStatus hydraulic_point_kernel(const void *raw_context,
                                       const SsScalarPointInput *input,
                                       SsScalarPointOutput *output) {
  const HydraulicPointContext *context = (const HydraulicPointContext *)raw_context;
  double kappa = 0.0;
  double derivative = 0.0;
  uint32_t coordinate;
  if (context == NULL || input == NULL || output == NULL ||
      (input->dimension != 2U && input->dimension != 3U) ||
      context->upward_axis >= input->dimension || context->conductivity <= 0.0 ||
      context->epsilon <= 0.0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(output, 0, sizeof(*output));
  if (input->value > 0.0 && input->value < context->epsilon) {
    kappa = input->value / context->epsilon;
    derivative = 1.0 / context->epsilon;
  } else if (input->value >= context->epsilon) {
    kappa = 1.0;
  }
  for (coordinate = 0U; coordinate < input->dimension; ++coordinate) {
    output->flux[coordinate] =
        context->conductivity *
        (input->gradient[coordinate] + (coordinate == context->upward_axis
                                            ? context->water_unit_weight * kappa
                                            : 0.0));
    output->flux_derivative_value[coordinate] =
        coordinate == context->upward_axis
            ? context->conductivity * context->water_unit_weight * derivative
            : 0.0;
    output->flux_derivative_gradient[coordinate * input->dimension + coordinate] =
        context->conductivity;
  }
  return SS_OK;
}

static SsStatus constant_flux_kernel(const void *raw_context,
                                     const SsNaturalPointInput *input,
                                     SsNaturalPointOutput *output) {
  const double *flux = (const double *)raw_context;
  if (flux == NULL || input == NULL || output == NULL || input->components != 1U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(output, 0, sizeof(*output));
  output->load[0] = *flux;
  output->signed_flux = *flux;
  return SS_OK;
}

static SsStatus affine_natural_kernel(const void *raw_context,
                                      const SsNaturalPointInput *input,
                                      SsNaturalPointOutput *output) {
  const AffineNaturalContext *context = (const AffineNaturalContext *)raw_context;
  if (context == NULL || input == NULL || output == NULL || input->components != 1U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(output, 0, sizeof(*output));
  output->load[0] = context->offset + context->derivative * input->value[0];
  output->load_derivative[0] = context->derivative;
  output->signed_flux = output->load[0];
  return SS_OK;
}

static SsStatus linear_scalar_kernel(const void *raw_context,
                                     const SsScalarPointInput *input,
                                     SsScalarPointOutput *output) {
  const LinearScalarContext *context = (const LinearScalarContext *)raw_context;
  uint32_t coordinate;
  double gradient_squared = 0.0;
  if (context == NULL || input == NULL || output == NULL ||
      context->conductivity <= 0.0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(output, 0, sizeof(*output));
  for (coordinate = 0U; coordinate < input->dimension; ++coordinate) {
    output->flux[coordinate] = context->conductivity * input->gradient[coordinate];
    output->flux_derivative_gradient[coordinate * input->dimension + coordinate] =
        context->conductivity;
    gradient_squared += input->gradient[coordinate] * input->gradient[coordinate];
  }
  output->source = context->source + context->reaction * input->value;
  output->source_derivative_value = context->reaction;
  output->potential_density = 0.5 * context->conductivity * gradient_squared -
                              context->source * input->value -
                              0.5 * context->reaction * input->value * input->value;
  return SS_OK;
}

static int test_mechanics_element_matrix(SsMcParameters *material) {
  const uint32_t orders[] = {1U, 2U, 4U, 8U};
  const double body[3] = {1.3, -2.1, 0.7};
  const double coupling[3] = {-0.2, 0.4, -0.1};
  uint32_t dimension;
  for (dimension = 2U; dimension <= 3U; ++dimension) {
    size_t field_order_index;
    for (field_order_index = 0U; field_order_index < sizeof(orders) / sizeof(orders[0]);
         ++field_order_index) {
      size_t geometry_order_index;
      for (geometry_order_index = 0U; geometry_order_index < 3U;
           ++geometry_order_index) {
        const SsFeDescriptor *field_descriptor = NULL;
        const SsFeDescriptor *coordinate_descriptor = NULL;
        SsFeQuadratureRequest quadrature_request = {0};
        SsFeQuadraturePlan quadrature_plan = {0};
        SsFeGeometryPolicy geometry_policy = {1.0e-14, 1.0e14};
        double coordinate_values[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
        double values[SS_FIELD_MAX_CLOSURE] = {0.0};
        double plus_values[SS_FIELD_MAX_CLOSURE] = {0.0};
        double minus_values[SS_FIELD_MAX_CLOSURE] = {0.0};
        double direction[SS_FIELD_MAX_CLOSURE] = {0.0};
        double residual[SS_FIELD_MAX_CLOSURE] = {0.0};
        double plus_residual[SS_FIELD_MAX_CLOSURE] = {0.0};
        double minus_residual[SS_FIELD_MAX_CLOSURE] = {0.0};
        double jacobian[SS_FIELD_MAX_CLOSURE * SS_FIELD_MAX_CLOSURE] = {0.0};
        SsElementOutput output = {
            SS_FIELD_MAX_CLOSURE,
            SS_FIELD_MAX_CLOSURE * SS_FIELD_MAX_CLOSURE,
            residual,
            jacobian,
            0.0,
        };
        SsElementOutput plus_output = {
            SS_FIELD_MAX_CLOSURE, 0U, plus_residual, NULL, 0.0,
        };
        SsElementOutput minus_output = {
            SS_FIELD_MAX_CLOSURE, 0U, minus_residual, NULL, 0.0,
        };
        SsErrorRecord error = {0};
        SsMechanicsElementInput element = {0};
        size_t coordinate_node;
        size_t field_node;
        size_t closure_count;
        const double step = 2.0e-7;
        CHECK(ss_fe_descriptor_find(dimension, orders[field_order_index],
                                    &field_descriptor) == SS_OK);
        CHECK(ss_fe_descriptor_find(dimension, orders[geometry_order_index],
                                    &coordinate_descriptor) == SS_OK);
        for (coordinate_node = 0U; coordinate_node < coordinate_descriptor->node_count;
             ++coordinate_node) {
          double reference[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
          double physical[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
          uint32_t coordinate;
          simplex_reference_node(coordinate_descriptor, coordinate_node, reference);
          curved_coordinate(dimension, reference, physical);
          for (coordinate = 0U; coordinate < dimension; ++coordinate) {
            coordinate_values[coordinate_node * dimension + coordinate] =
                physical[coordinate];
          }
        }
        for (field_node = 0U; field_node < field_descriptor->node_count; ++field_node) {
          double reference[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
          SsFeGeometry geometry = {0};
          double field[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
          uint32_t component;
          simplex_reference_node(field_descriptor, field_node, reference);
          CHECK(ss_fe_geometry_eval(coordinate_descriptor, coordinate_values, reference,
                                    &geometry_policy, &geometry) == SS_OK);
          mechanics_field(dimension, geometry.physical, field);
          for (component = 0U; component < dimension; ++component) {
            const size_t slot = field_node * dimension + component;
            values[slot] = field[component];
            direction[slot] = 0.03 * sin((double)(slot + 1U) * 0.71);
            plus_values[slot] = values[slot] + step * direction[slot];
            minus_values[slot] = values[slot] - step * direction[slot];
          }
        }
        closure_count = field_descriptor->node_count * dimension;
        quadrature_request = (SsFeQuadratureRequest){
            dimension,
            SS_FE_QUADRATURE_VOLUME,
            SS_FE_OPERATION_MECHANICS_STIFFNESS,
            orders[field_order_index],
            orders[geometry_order_index],
            0U,
            SS_FE_QUADRATURE_MODE_DEFAULT,
            NULL,
        };
        CHECK(ss_fe_quadrature_plan(&quadrature_request, &quadrature_plan) == SS_OK);
        element = (SsMechanicsElementInput){
            .dimension = dimension,
            .field_descriptor = field_descriptor,
            .coordinate_descriptor = coordinate_descriptor,
            .quadrature = quadrature_plan.selected_rule,
            .coordinate_values = coordinate_values,
            .field_values = values,
            .body_force = body,
            .coupling_body_force = coupling,
            .material = material,
            .reduction_factor = 1.0,
            .external_load_factor = 0.65,
            .stable_cell_id =
                UINT64_C(9001) +
                (SsEntityId)(dimension * 100U + orders[field_order_index] * 10U +
                             orders[geometry_order_index]),
            .coupling_point_kernel = NULL,
            .coupling_point_context = NULL,
        };
        CHECK(ss_assembly_mechanics_element(
                  &element,
                  SS_ASSEMBLY_RESIDUAL | SS_ASSEMBLY_JACOBIAN | SS_ASSEMBLY_POTENTIAL,
                  &output, &error) == SS_OK);
        CHECK(error.valid == 0);
        element.field_values = plus_values;
        CHECK(ss_assembly_mechanics_element(
                  &element, SS_ASSEMBLY_RESIDUAL | SS_ASSEMBLY_POTENTIAL, &plus_output,
                  &error) == SS_OK);
        element.field_values = minus_values;
        CHECK(ss_assembly_mechanics_element(
                  &element, SS_ASSEMBLY_RESIDUAL | SS_ASSEMBLY_POTENTIAL, &minus_output,
                  &error) == SS_OK);
        {
          size_t row;
          double residual_direction = 0.0;
          for (row = 0U; row < closure_count; ++row) {
            size_t column;
            double action = 0.0;
            const double difference =
                (plus_residual[row] - minus_residual[row]) / (2.0 * step);
            residual_direction += residual[row] * direction[row];
            for (column = 0U; column < closure_count; ++column) {
              action += jacobian[row * closure_count + column] * direction[column];
            }
            CHECK_NEAR(difference, action, 2.0e-7 * (1.0 + fabs(action)));
          }
          CHECK_NEAR((plus_output.potential - minus_output.potential) / (2.0 * step),
                     residual_direction, 5.0e-7 * (1.0 + fabs(residual_direction)));
        }
      }
    }
  }
  return 0;
}

static int test_scalar_element_matrix(void) {
  const uint32_t orders[] = {1U, 2U, 4U};
  const HydraulicPointContext context = {2.7, 9.81, 1.0, 1U};
  uint32_t dimension;
  for (dimension = 2U; dimension <= 3U; ++dimension) {
    size_t field_order_index;
    for (field_order_index = 0U; field_order_index < 3U; ++field_order_index) {
      size_t geometry_order_index;
      for (geometry_order_index = 0U; geometry_order_index < 3U;
           ++geometry_order_index) {
        const SsFeDescriptor *field_descriptor = NULL;
        const SsFeDescriptor *coordinate_descriptor = NULL;
        SsFeQuadratureRequest quadrature_request = {0};
        SsFeQuadraturePlan quadrature_plan = {0};
        double coordinate_values[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
        double values[SS_FE_MAX_NODES] = {0.0};
        double plus_values[SS_FE_MAX_NODES] = {0.0};
        double minus_values[SS_FE_MAX_NODES] = {0.0};
        double direction[SS_FE_MAX_NODES] = {0.0};
        double residual[SS_FE_MAX_NODES] = {0.0};
        double plus_residual[SS_FE_MAX_NODES] = {0.0};
        double minus_residual[SS_FE_MAX_NODES] = {0.0};
        double jacobian[SS_FE_MAX_NODES * SS_FE_MAX_NODES] = {0.0};
        SsElementOutput output = {
            SS_FE_MAX_NODES, SS_FE_MAX_NODES * SS_FE_MAX_NODES, residual, jacobian, 0.0,
        };
        SsElementOutput plus_output = {
            SS_FE_MAX_NODES, 0U, plus_residual, NULL, 0.0,
        };
        SsElementOutput minus_output = {
            SS_FE_MAX_NODES, 0U, minus_residual, NULL, 0.0,
        };
        SsScalarElementInput element = {0};
        SsErrorRecord error = {0};
        size_t coordinate_node;
        size_t field_node;
        const double step = 3.0e-7;
        CHECK(ss_fe_descriptor_find(dimension, orders[field_order_index],
                                    &field_descriptor) == SS_OK);
        CHECK(ss_fe_descriptor_find(dimension, orders[geometry_order_index],
                                    &coordinate_descriptor) == SS_OK);
        for (coordinate_node = 0U; coordinate_node < coordinate_descriptor->node_count;
             ++coordinate_node) {
          double reference[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
          double physical[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
          uint32_t coordinate;
          simplex_reference_node(coordinate_descriptor, coordinate_node, reference);
          curved_coordinate(dimension, reference, physical);
          for (coordinate = 0U; coordinate < dimension; ++coordinate) {
            coordinate_values[coordinate_node * dimension + coordinate] =
                physical[coordinate];
          }
        }
        for (field_node = 0U; field_node < field_descriptor->node_count; ++field_node) {
          double reference[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
          simplex_reference_node(field_descriptor, field_node, reference);
          values[field_node] = 0.31 + 0.04 * reference[0] + 0.03 * reference[1] +
                               (dimension == 3U ? 0.02 * reference[2] : 0.0);
          direction[field_node] = 0.07 * cos((double)(field_node + 1U) * 0.43);
          plus_values[field_node] = values[field_node] + step * direction[field_node];
          minus_values[field_node] = values[field_node] - step * direction[field_node];
        }
        quadrature_request = (SsFeQuadratureRequest){
            dimension,
            SS_FE_QUADRATURE_VOLUME,
            SS_FE_OPERATION_HYDRAULIC_DIFFUSION,
            orders[field_order_index],
            orders[geometry_order_index],
            0U,
            SS_FE_QUADRATURE_MODE_DEFAULT,
            NULL,
        };
        CHECK(ss_fe_quadrature_plan(&quadrature_request, &quadrature_plan) == SS_OK);
        element = (SsScalarElementInput){
            dimension,
            field_descriptor,
            coordinate_descriptor,
            quadrature_plan.selected_rule,
            coordinate_values,
            values,
            hydraulic_point_kernel,
            &context,
            UINT64_C(9101) +
                (SsEntityId)(dimension * 100U + orders[field_order_index] * 10U +
                             orders[geometry_order_index]),
        };
        CHECK(ss_assembly_scalar_element(&element, SS_ASSEMBLY_FUSED, &output,
                                         &error) == SS_OK);
        element.field_values = plus_values;
        CHECK(ss_assembly_scalar_element(&element, SS_ASSEMBLY_RESIDUAL, &plus_output,
                                         &error) == SS_OK);
        element.field_values = minus_values;
        CHECK(ss_assembly_scalar_element(&element, SS_ASSEMBLY_RESIDUAL, &minus_output,
                                         &error) == SS_OK);
        {
          size_t row;
          for (row = 0U; row < field_descriptor->node_count; ++row) {
            size_t column;
            double action = 0.0;
            const double difference =
                (plus_residual[row] - minus_residual[row]) / (2.0 * step);
            for (column = 0U; column < field_descriptor->node_count; ++column) {
              action += jacobian[row * field_descriptor->node_count + column] *
                        direction[column];
            }
            CHECK_NEAR(difference, action, 3.0e-7 * (1.0 + fabs(action)));
          }
        }
      }
    }
  }
  return 0;
}

static int test_natural_flux(void) {
  const SsFieldSpec field_spec = {1U, 1U, 0U, NULL};
  const double flux = 2.5;
  SsMeshInput input = square_input();
  SsMesh *mesh = NULL;
  SsMeshError mesh_error = {0};
  SsFieldSpace *space = NULL;
  SsFieldFacetPack facets[4] = {0};
  size_t facet_count = 0U;
  const SsFeDescriptor *field_descriptor = NULL;
  const SsFeQuadrature *quadrature = NULL;
  double field_values[SS_FIELD_MAX_FACET_CLOSURE] = {0.0};
  double residual[SS_FIELD_MAX_FACET_CLOSURE] = {0.0};
  double jacobian[SS_FIELD_MAX_FACET_CLOSURE * SS_FIELD_MAX_FACET_CLOSURE] = {0.0};
  SsElementOutput output = {
      SS_FIELD_MAX_FACET_CLOSURE,
      SS_FIELD_MAX_FACET_CLOSURE * SS_FIELD_MAX_FACET_CLOSURE,
      residual,
      jacobian,
      0.0,
  };
  SsNaturalElementInput natural = {0};
  SsErrorRecord error = {0};
  double signed_flux = 0.0;
  size_t node;
  CHECK(ss_mesh_create(PETSC_COMM_SELF, &input, &mesh, &mesh_error) == SS_OK);
  CHECK(ss_mesh_field_space_create(mesh, &field_spec, &space) == SS_OK);
  CHECK(ss_field_space_pack_owned_facets(space, 1U, facets, 4U, &facet_count) == SS_OK);
  CHECK(facet_count == 4U);
  CHECK(ss_fe_descriptor_find(2U, 1U, &field_descriptor) == SS_OK);
  CHECK(ss_fe_facet_quadrature_default(2U, 1U, &quadrature) == SS_OK);
  natural = (SsNaturalElementInput){
      &facets[0],   field_descriptor,     quadrature,
      field_values, constant_flux_kernel, &flux,
  };
  CHECK(ss_assembly_natural_element(&natural, SS_ASSEMBLY_FUSED, &output, &signed_flux,
                                    &error) == SS_OK);
  CHECK_NEAR(signed_flux, flux, 2.0e-13);
  {
    double residual_sum = 0.0;
    for (node = 0U; node < facets[0].closure_count; ++node) {
      residual_sum += residual[node];
    }
    CHECK_NEAR(residual_sum, -flux, 2.0e-13);
  }
  CHECK(ss_field_space_destroy(&space) == SS_OK);
  CHECK(ss_mesh_destroy(&mesh) == SS_OK);
  return 0;
}

static int test_element(void) {
  const SsMcParameterInput material_input = {
      8.0e4,
      0.27,
      1.0e6,
      0.45,
      0.2,
      SS_MC_DAVIS_B,
      UINT64_C(1),
      "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  };
  SsMcParameters *material = NULL;
  SsMcError error = {0};
  CHECK(ss_mc_parameters_create(&material_input, &material, &error) == SS_OK);
  CHECK(test_mechanics_element_matrix(material) == 0);
  CHECK(test_scalar_element_matrix() == 0);
  CHECK(test_natural_flux() == 0);
  CHECK(ss_mc_parameters_destroy(&material) == SS_OK);
  return 0;
}

static int test_pattern(void) {
  const SsFieldSpec field_spec = {1U, 1U, 0U, NULL};
  SsMeshInput input = square_input();
  SsMesh *mesh = NULL;
  SsMeshError mesh_error = {0};
  SsFieldSpace *space = NULL;
  SsCellPack cells[2] = {{0}};
  size_t cell_count = 0U;
  SsAssemblyPlan *plan = NULL;
  SsErrorRecord error = {0};
  SsAssemblyPlanSummary plan_summary = {0};
  Mat matrix = NULL;
  MatInfo matrix_info = {0};
  size_t local_row;
  CHECK(ss_mesh_create(PETSC_COMM_SELF, &input, &mesh, &mesh_error) == SS_OK);
  CHECK(mesh != NULL && mesh_error.valid == 0);
  CHECK(ss_mesh_field_space_create(mesh, &field_spec, &space) == SS_OK);
  CHECK(ss_field_space_pack_incident_cells(space, cells, 2U, &cell_count) == SS_OK);
  CHECK(cell_count == 2U);
  CHECK(ss_assembly_plan_create(PETSC_COMM_SELF, space, &plan, &error) == SS_OK);
  CHECK(plan != NULL && error.valid == 0);
  CHECK(ss_assembly_plan_summary(plan, &plan_summary) == SS_OK);
  CHECK(plan_summary.local_owned_rows == 4U);
  CHECK(plan_summary.global_rows == 4U);
  CHECK(plan_summary.local_incident_cells == 2U);
  CHECK(plan_summary.local_owned_cells == 2U);
  CHECK(plan_summary.local_halo_cells == 0U);
  CHECK(plan_summary.local_nonzeros == 14U);
  CHECK(plan_summary.local_diagonal_nonzeros == 14U);
  CHECK(plan_summary.local_off_diagonal_nonzeros == 0U);
  CHECK(plan_summary.contribution_count == 6U);
  CHECK(plan_summary.overlap_depth == 0U);
  CHECK(strlen(plan_summary.pattern_digest) == 71U);
  for (local_row = 0U; local_row < 4U; ++local_row) {
    SsAssemblyRowSummary row = {0};
    PetscInt actual_columns[4] = {0};
    SsEntityId actual_cells[2] = {0U, 0U};
    size_t actual_column_count = 0U;
    size_t actual_cell_count = 0U;
    PetscInt expected_columns[6] = {0};
    SsEntityId expected_cells[2] = {0U, 0U};
    size_t expected_column_count = 0U;
    size_t expected_cell_count = 0U;
    size_t cell;
    CHECK(ss_assembly_plan_row(plan, local_row, &row, actual_columns, 4U,
                               &actual_column_count, actual_cells, 2U,
                               &actual_cell_count) == SS_OK);
    for (cell = 0U; cell < cell_count; ++cell) {
      size_t slot;
      int contains = 0;
      for (slot = 0U; slot < cells[cell].closure_count; ++slot) {
        if (cells[cell].global_indices[slot] == row.global_row) {
          contains = 1;
        }
      }
      if (contains != 0) {
        expected_cells[expected_cell_count++] = cells[cell].stable_cell_id;
        for (slot = 0U; slot < cells[cell].closure_count; ++slot) {
          expected_columns[expected_column_count++] = cells[cell].global_indices[slot];
        }
      }
    }
    expected_column_count = sorted_unique(expected_columns, expected_column_count);
    CHECK(actual_column_count == expected_column_count);
    CHECK(actual_cell_count == expected_cell_count);
    CHECK(memcmp(actual_columns, expected_columns,
                 actual_column_count * sizeof(actual_columns[0])) == 0);
    CHECK(memcmp(actual_cells, expected_cells,
                 actual_cell_count * sizeof(actual_cells[0])) == 0);
  }
  CHECK(ss_assembly_plan_create_matrix(plan, &matrix) == SS_OK);
  CHECK(MatGetInfo(matrix, MAT_LOCAL, &matrix_info) == PETSC_SUCCESS);
  CHECK(fabs((double)matrix_info.nz_allocated - 14.0) < 0.5);
  CHECK(MatDestroy(&matrix) == PETSC_SUCCESS);
  CHECK(ss_assembly_plan_destroy(&plan) == SS_OK && plan == NULL);
  CHECK(ss_assembly_plan_create_matrix_free(PETSC_COMM_SELF, space, &plan, &error) ==
        SS_OK);
  CHECK(ss_assembly_plan_summary(plan, &plan_summary) == SS_OK);
  CHECK(plan_summary.local_owned_rows == 4U);
  CHECK(plan_summary.local_nonzeros == 0U);
  CHECK(plan_summary.local_diagonal_nonzeros == 0U);
  CHECK(plan_summary.local_off_diagonal_nonzeros == 0U);
  CHECK(plan_summary.contribution_count == 6U);
  CHECK(strlen(plan_summary.pattern_digest) == 71U);
  for (local_row = 0U; local_row < 4U; ++local_row) {
    SsAssemblyRowSummary row = {0};
    SsEntityId actual_cells[2] = {0U, 0U};
    size_t actual_column_count = SIZE_MAX;
    size_t actual_cell_count = 0U;
    CHECK(ss_assembly_plan_row(plan, local_row, &row, NULL, 0U, &actual_column_count,
                               actual_cells, 2U, &actual_cell_count) == SS_OK);
    CHECK(actual_column_count == 0U);
    CHECK(actual_cell_count >= 1U && actual_cell_count <= 2U);
  }
  CHECK(ss_assembly_plan_create_matrix(plan, &matrix) == SS_ERROR_INVALID_STATE);
  CHECK(matrix == NULL);
  CHECK(ss_assembly_plan_destroy(&plan) == SS_OK && plan == NULL);
  CHECK(ss_field_space_destroy(&space) == SS_OK);
  CHECK(ss_mesh_destroy(&mesh) == SS_OK);
  return 0;
}

static int test_lifting(void) {
  const SsStrongConstraint constraints[] = {
      {UINT64_C(0x310000000001), 0U, 0U, 0.35},
      {UINT64_C(0x310000000004), 0U, 0U, -0.17},
  };
  const SsFieldSpec field_spec = {
      1U,
      1U,
      sizeof(constraints) / sizeof(constraints[0]),
      constraints,
  };
  const LinearScalarContext context = {3.2, 0.6, -0.4};
  const AffineNaturalContext natural_context = {0.8, 0.25};
  const SsScalarOperatorSpec scalar_spec = {
      linear_scalar_kernel,
      &context,
  };
  const SsNaturalOperatorSpec natural_spec = {
      1U,
      affine_natural_kernel,
      &natural_context,
      1,
  };
  const SsOperatorSpec operator_spec = {
      SS_ASSEMBLY_SCALAR_FLUX,
      SS_ASSEMBLY_QUALIFICATION,
      0U,
      0U,
      1,
      NULL,
      &scalar_spec,
      &natural_spec,
      1U,
      0,
      0U,
      0U,
      SS_FE_QUADRATURE_MODE_DEFAULT,
  };
  SsMeshInput input = square_input();
  SsMesh *mesh = NULL;
  SsMeshError mesh_error = {0};
  SsFieldSpace *space = NULL;
  SsAssemblyPlan *plan = NULL;
  SsOperator *operator_handle = NULL;
  SsErrorRecord error = {0};
  Vec field = NULL;
  Vec residual = NULL;
  Mat jacobian = NULL;
  SsOperatorState state = {0};
  SsCellPack cells[2] = {0};
  size_t cell_count = 0U;
  double expected_residual[2] = {0.0, 0.0};
  double expected_jacobian[4] = {0.0, 0.0, 0.0, 0.0};
  PetscInt free_rows[2] = {-1, -1};
  size_t free_row_count = 0U;
  double functional = 0.0;
  double expected_functional = 0.0;
  size_t cell_index;
  CHECK(ss_mesh_create(PETSC_COMM_SELF, &input, &mesh, &mesh_error) == SS_OK);
  CHECK(ss_mesh_field_space_create(mesh, &field_spec, &space) == SS_OK);
  CHECK(ss_assembly_plan_create(PETSC_COMM_SELF, space, &plan, &error) == SS_OK);
  CHECK(ss_operator_create(plan, &operator_spec, &operator_handle, &error) == SS_OK);
  CHECK(ss_assembly_plan_create_vector(plan, &field) == SS_OK);
  CHECK(ss_assembly_plan_create_vector(plan, &residual) == SS_OK);
  CHECK(ss_assembly_plan_create_matrix(plan, &jacobian) == SS_OK);
  CHECK(ss_field_space_pack_incident_cells(space, cells, 2U, &cell_count) == SS_OK);
  CHECK(cell_count == 2U);
  for (cell_index = 0U; cell_index < cell_count; ++cell_index) {
    size_t slot;
    for (slot = 0U; slot < cells[cell_index].closure_count; ++slot) {
      const PetscInt global = cells[cell_index].global_indices[slot];
      size_t previous;
      if (cells[cell_index].constrained[slot] != 0U) {
        continue;
      }
      for (previous = 0U; previous < free_row_count; ++previous) {
        if (free_rows[previous] == global) {
          break;
        }
      }
      if (previous == free_row_count) {
        CHECK(free_row_count < 2U);
        free_rows[free_row_count++] = global;
      }
    }
  }
  CHECK(free_row_count == 2U);
  {
    const PetscScalar values[2] = {0.28, -0.09};
    CHECK(VecSetValues(field, 2, free_rows, values, INSERT_VALUES) == PETSC_SUCCESS);
    CHECK(VecAssemblyBegin(field) == PETSC_SUCCESS);
    CHECK(VecAssemblyEnd(field) == PETSC_SUCCESS);
  }
  state.field = field;
  state.reduction_factor = 1.0;
  state.external_load_factor = 1.25;
  CHECK(ss_operator_residual_jacobian(operator_handle, &state, residual, jacobian,
                                      &error) == SS_OK);
  CHECK(ss_operator_functional(operator_handle, &state, &functional, &error) == SS_OK);
  for (cell_index = 0U; cell_index < cell_count; ++cell_index) {
    const SsCellPack *cell = &cells[cell_index];
    const SsFeDescriptor *field_descriptor = NULL;
    const SsFeDescriptor *coordinate_descriptor = NULL;
    SsFeQuadratureRequest request = {
        cell->dimension,
        SS_FE_QUADRATURE_VOLUME,
        SS_FE_OPERATION_HYDRAULIC_DIFFUSION,
        cell->order,
        cell->coordinate_order,
        0U,
        SS_FE_QUADRATURE_MODE_DEFAULT,
        NULL,
    };
    SsFeQuadraturePlan quadrature = {0};
    double complete[SS_FE_MAX_NODES] = {0.0};
    double element_residual[SS_FE_MAX_NODES] = {0.0};
    double element_jacobian[SS_FE_MAX_NODES * SS_FE_MAX_NODES] = {0.0};
    SsElementOutput element_output = {
        SS_FE_MAX_NODES,
        SS_FE_MAX_NODES * SS_FE_MAX_NODES,
        element_residual,
        element_jacobian,
        0.0,
    };
    SsScalarElementInput element = {0};
    size_t slot;
    CHECK(ss_fe_descriptor_find(cell->dimension, cell->order, &field_descriptor) ==
          SS_OK);
    CHECK(ss_fe_descriptor_find(cell->dimension, cell->coordinate_order,
                                &coordinate_descriptor) == SS_OK);
    CHECK(ss_fe_quadrature_plan(&request, &quadrature) == SS_OK);
    for (slot = 0U; slot < cell->closure_count; ++slot) {
      if (cell->constrained[slot] != 0U) {
        complete[slot] = cell->prescribed_values[slot];
      } else {
        PetscScalar value = 0.0;
        CHECK(VecGetValues(field, 1, &cell->global_indices[slot], &value) ==
              PETSC_SUCCESS);
        complete[slot] = (double)value;
      }
    }
    element = (SsScalarElementInput){
        cell->dimension,
        field_descriptor,
        coordinate_descriptor,
        quadrature.selected_rule,
        cell->coordinate_coordinates,
        complete,
        linear_scalar_kernel,
        &context,
        cell->stable_cell_id,
    };
    CHECK(ss_assembly_scalar_element(&element,
                                     SS_ASSEMBLY_RESIDUAL | SS_ASSEMBLY_JACOBIAN |
                                         SS_ASSEMBLY_POTENTIAL,
                                     &element_output, &error) == SS_OK);
    expected_functional += element_output.potential;
    for (slot = 0U; slot < cell->closure_count; ++slot) {
      size_t row_index;
      size_t column_slot;
      if (cell->constrained[slot] != 0U) {
        continue;
      }
      row_index = cell->global_indices[slot] == free_rows[0] ? 0U : 1U;
      expected_residual[row_index] += element_residual[slot];
      for (column_slot = 0U; column_slot < cell->closure_count; ++column_slot) {
        size_t column_index;
        if (cell->constrained[column_slot] != 0U) {
          continue;
        }
        column_index = cell->global_indices[column_slot] == free_rows[0] ? 0U : 1U;
        expected_jacobian[row_index * 2U + column_index] +=
            element_jacobian[slot * cell->closure_count + column_slot];
      }
    }
  }
  {
    SsFieldFacetPack facets[4] = {0};
    size_t facet_count = 0U;
    const SsFeDescriptor *field_descriptor = NULL;
    const SsFeQuadrature *quadrature = NULL;
    size_t facet_index;
    CHECK(ss_field_space_pack_incident_facets(space, 1U, facets, 4U, &facet_count) ==
          SS_OK);
    CHECK(facet_count == 4U);
    CHECK(ss_fe_descriptor_find(2U, 1U, &field_descriptor) == SS_OK);
    CHECK(ss_fe_facet_quadrature_default(2U, 1U, &quadrature) == SS_OK);
    for (facet_index = 0U; facet_index < facet_count; ++facet_index) {
      double values[SS_FIELD_MAX_FACET_CLOSURE] = {0.0};
      double element_residual[SS_FIELD_MAX_FACET_CLOSURE] = {0.0};
      double element_jacobian[SS_FIELD_MAX_FACET_CLOSURE * SS_FIELD_MAX_FACET_CLOSURE] =
          {0.0};
      double signed_flux = 0.0;
      SsElementOutput element_output = {
          SS_FIELD_MAX_FACET_CLOSURE,
          SS_FIELD_MAX_FACET_CLOSURE * SS_FIELD_MAX_FACET_CLOSURE,
          element_residual,
          element_jacobian,
          0.0,
      };
      SsNaturalElementInput element = {
          &facets[facet_index],  field_descriptor, quadrature, values,
          affine_natural_kernel, &natural_context,
      };
      size_t slot;
      for (slot = 0U; slot < facets[facet_index].closure_count; ++slot) {
        if (facets[facet_index].constrained[slot] != 0U) {
          values[slot] = facets[facet_index].prescribed_values[slot];
        } else {
          PetscScalar value = 0.0;
          CHECK(VecGetValues(field, 1, &facets[facet_index].global_indices[slot],
                             &value) == PETSC_SUCCESS);
          values[slot] = (double)value;
        }
      }
      CHECK(ss_assembly_natural_element(
                &element, SS_ASSEMBLY_FUSED | SS_ASSEMBLY_POTENTIAL, &element_output,
                &signed_flux, &error) == SS_OK);
      expected_functional += state.external_load_factor * element_output.potential;
      for (slot = 0U; slot < facets[facet_index].closure_count; ++slot) {
        size_t row_index;
        size_t column_slot;
        if (facets[facet_index].constrained[slot] != 0U) {
          continue;
        }
        row_index = facets[facet_index].global_indices[slot] == free_rows[0] ? 0U : 1U;
        expected_residual[row_index] +=
            state.external_load_factor * element_residual[slot];
        for (column_slot = 0U; column_slot < facets[facet_index].closure_count;
             ++column_slot) {
          size_t column_index;
          if (facets[facet_index].constrained[column_slot] != 0U) {
            continue;
          }
          column_index =
              facets[facet_index].global_indices[column_slot] == free_rows[0] ? 0U : 1U;
          expected_jacobian[row_index * 2U + column_index] +=
              state.external_load_factor *
              element_jacobian[slot * facets[facet_index].closure_count + column_slot];
        }
      }
    }
  }
  {
    PetscScalar actual_residual[2] = {0.0, 0.0};
    PetscScalar actual_jacobian[4] = {0.0, 0.0, 0.0, 0.0};
    size_t row;
    CHECK(VecGetValues(residual, 2, free_rows, actual_residual) == PETSC_SUCCESS);
    CHECK(MatGetValues(jacobian, 2, free_rows, 2, free_rows, actual_jacobian) ==
          PETSC_SUCCESS);
    for (row = 0U; row < 2U; ++row) {
      size_t column;
      CHECK_NEAR((double)actual_residual[row], expected_residual[row], 2.0e-13);
      for (column = 0U; column < 2U; ++column) {
        CHECK_NEAR((double)actual_jacobian[row * 2U + column],
                   expected_jacobian[row * 2U + column], 2.0e-13);
      }
    }
    CHECK_NEAR(functional, expected_functional, 2.0e-13);
  }
  CHECK(MatDestroy(&jacobian) == PETSC_SUCCESS);
  CHECK(VecDestroy(&residual) == PETSC_SUCCESS);
  CHECK(VecDestroy(&field) == PETSC_SUCCESS);
  CHECK(ss_operator_destroy(&operator_handle) == SS_OK);
  CHECK(ss_assembly_plan_destroy(&plan) == SS_OK);
  CHECK(ss_field_space_destroy(&space) == SS_OK);
  CHECK(ss_mesh_destroy(&mesh) == SS_OK);
  return 0;
}

typedef struct {
  SsMesh *mesh;
  SsFieldSpace *space;
  SsAssemblyPlan *plan;
  SsOperator *operator_handle;
  Vec field;
  Vec residual;
  Mat jacobian;
  LinearScalarContext context;
  SsScalarOperatorSpec scalar_spec;
  SsOperatorSpec operator_spec;
  SsOperatorState state;
} ScalarOperatorFixture;

static int scalar_operator_fixture_create(size_t memory_cap,
                                          ScalarOperatorFixture *fixture) {
  const SsFieldSpec field_spec = {1U, 1U, 0U, NULL};
  SsMeshInput input = square_input();
  SsMeshError mesh_error = {0};
  SsErrorRecord error = {0};
  PetscInt rows[4] = {0, 1, 2, 3};
  PetscScalar values[4] = {0.31, 0.37, 0.42, 0.48};
  memset(fixture, 0, sizeof(*fixture));
  fixture->context = (LinearScalarContext){2.4, 0.3, -0.2};
  fixture->scalar_spec = (SsScalarOperatorSpec){
      linear_scalar_kernel,
      &fixture->context,
  };
  fixture->operator_spec = (SsOperatorSpec){
      SS_ASSEMBLY_SCALAR_FLUX,
      SS_ASSEMBLY_QUALIFICATION,
      memory_cap,
      0U,
      1,
      NULL,
      &fixture->scalar_spec,
      NULL,
      0U,
      0,
      0U,
      0U,
      SS_FE_QUADRATURE_MODE_DEFAULT,
  };
  CHECK(ss_mesh_create(PETSC_COMM_SELF, &input, &fixture->mesh, &mesh_error) == SS_OK);
  CHECK(ss_mesh_field_space_create(fixture->mesh, &field_spec, &fixture->space) ==
        SS_OK);
  CHECK(ss_assembly_plan_create(PETSC_COMM_SELF, fixture->space, &fixture->plan,
                                &error) == SS_OK);
  CHECK(ss_operator_create(fixture->plan, &fixture->operator_spec,
                           &fixture->operator_handle, &error) == SS_OK);
  CHECK(ss_assembly_plan_create_vector(fixture->plan, &fixture->field) == SS_OK);
  CHECK(ss_assembly_plan_create_vector(fixture->plan, &fixture->residual) == SS_OK);
  CHECK(ss_assembly_plan_create_matrix(fixture->plan, &fixture->jacobian) == SS_OK);
  CHECK(VecSetValues(fixture->field, 4, rows, values, INSERT_VALUES) == PETSC_SUCCESS);
  CHECK(VecAssemblyBegin(fixture->field) == PETSC_SUCCESS);
  CHECK(VecAssemblyEnd(fixture->field) == PETSC_SUCCESS);
  fixture->state.field = fixture->field;
  fixture->state.reduction_factor = 1.0;
  fixture->state.external_load_factor = 1.0;
  return 0;
}

static int scalar_operator_fixture_destroy(ScalarOperatorFixture *fixture) {
  CHECK(MatDestroy(&fixture->jacobian) == PETSC_SUCCESS);
  CHECK(VecDestroy(&fixture->residual) == PETSC_SUCCESS);
  CHECK(VecDestroy(&fixture->field) == PETSC_SUCCESS);
  CHECK(ss_operator_destroy(&fixture->operator_handle) == SS_OK);
  CHECK(ss_assembly_plan_destroy(&fixture->plan) == SS_OK);
  CHECK(ss_field_space_destroy(&fixture->space) == SS_OK);
  CHECK(ss_mesh_destroy(&fixture->mesh) == SS_OK);
  return 0;
}

static int test_memory(void) {
  ScalarOperatorFixture baseline;
  ScalarOperatorFixture tiled;
  SsAssemblyDiagnostics baseline_diagnostics = {0};
  SsAssemblyDiagnostics tiled_diagnostics = {0};
  uint64_t per_cell;
  uint64_t fixed;
  size_t admitted_cap;
  size_t iteration;
  CHECK(scalar_operator_fixture_create(0U, &baseline) == 0);
  CHECK(ss_operator_diagnostics(baseline.operator_handle, &baseline_diagnostics) ==
        SS_OK);
  CHECK(baseline_diagnostics.tile_capacity == 2U);
  CHECK(baseline_diagnostics.scratch_peak_bytes > 0U);
  CHECK(baseline_diagnostics.admitted_peak_bytes >
        baseline_diagnostics.scratch_peak_bytes);
  per_cell =
      baseline_diagnostics.scratch_peak_bytes / baseline_diagnostics.tile_capacity;
  fixed = baseline_diagnostics.admitted_peak_bytes -
          baseline_diagnostics.scratch_peak_bytes;
  CHECK(per_cell > 0U && fixed <= SIZE_MAX - per_cell);
  admitted_cap = (size_t)(fixed + per_cell);
  CHECK(scalar_operator_fixture_destroy(&baseline) == 0);
  CHECK(scalar_operator_fixture_create(admitted_cap, &tiled) == 0);
  for (iteration = 0U; iteration < 20U; ++iteration) {
    CHECK(ss_operator_residual_jacobian(tiled.operator_handle, &tiled.state,
                                        tiled.residual, tiled.jacobian, NULL) == SS_OK);
  }
  CHECK(ss_operator_diagnostics(tiled.operator_handle, &tiled_diagnostics) == SS_OK);
  CHECK(tiled_diagnostics.tile_capacity == 1U);
  CHECK(tiled_diagnostics.tile_count == 2U);
  CHECK(tiled_diagnostics.cell_evaluations == 40U);
  CHECK(tiled_diagnostics.completed_rows == 4U);
  CHECK(tiled_diagnostics.row_writer_violations == 0U);
  CHECK(tiled_diagnostics.worker_service_calls == 0U);
  CHECK(tiled_diagnostics.admitted_peak_bytes <= admitted_cap);
  CHECK(tiled_diagnostics.petsc_new_nonzero_allocations == 0U);
  CHECK(scalar_operator_fixture_destroy(&tiled) == 0);
  return 0;
}

int main(int argc, char **argv) {
  int status;
  if (argc != 2) {
    fputs("usage: ss_test_assembly TEST-ID\n", stderr);
    return 2;
  }
  if (PetscInitialize(&argc, &argv, NULL, NULL) != PETSC_SUCCESS) {
    return 3;
  }
  if (strcmp(argv[1], "ASM-UNIT-PATTERN-001") == 0) {
    status = test_pattern();
  } else if (strcmp(argv[1], "ASM-UNIT-ELEMENT-001") == 0) {
    status = test_element();
  } else if (strcmp(argv[1], "ASM-UNIT-LIFTING-001") == 0) {
    status = test_lifting();
  } else if (strcmp(argv[1], "ASM-PERF-MEMORY-001") == 0) {
    status = test_memory();
  } else if (strcmp(argv[1], "ASM-UNIT-MATFREE-001") == 0) {
    status = test_matrix_free(1);
  } else if (strcmp(argv[1], "ASM-CONTRACT-MATFREE-OPENMP-001") == 0) {
    status = test_matrix_free(4);
  } else {
    fprintf(stderr, "unknown test id: %s\n", argv[1]);
    status = 2;
  }
  if (PetscFinalize() != PETSC_SUCCESS) {
    return 4;
  }
  return status;
}
