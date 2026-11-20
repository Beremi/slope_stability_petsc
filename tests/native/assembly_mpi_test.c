/* Distributed and hybrid assembly: overlap, row ownership, COO collectives,
 * matrix-free action and invariance across MPI x OpenMP layouts;
 * assembly_mpi_affinity.c checks the rank and thread placement.
 *
 * Launched by tests/native/support/mpi_layout_runner.py --suite assembly.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "assembly_mpi_internal.h"

int test_rank = -1;
static int local_status = 0;

static void owned_range(size_t global_count, int rank, int size, size_t *start,
                        size_t *count) {
  const size_t quotient = global_count / (size_t)size;
  const size_t remainder = global_count % (size_t)size;
  *count = quotient + ((size_t)rank < remainder ? 1U : 0U);
  *start =
      (size_t)rank * quotient + ((size_t)rank < remainder ? (size_t)rank : remainder);
}

static size_t grid_vertex(size_t column, size_t row) {
  return row * (GRID_NX + 1U) + column;
}

static SsEntityId grid_vertex_id(size_t ordinal) {
  return UINT64_C(0x410000000000) + (SsEntityId)ordinal;
}

static SsEntityId grid_cell_id(size_t ordinal) {
  return UINT64_C(0x420000000000) + (SsEntityId)ordinal;
}

static SsEntityId grid_edge_id(size_t first, size_t second) {
  if (second < first) {
    const size_t swap = first;
    first = second;
    second = swap;
  }
  return UINT64_C(0x430000000000) +
         (SsEntityId)(first * (size_t)GRID_VERTICES + second);
}

static void grid_coordinate(size_t vertex, double *x, double *y) {
  const size_t column = vertex % (GRID_NX + 1U);
  const size_t row = vertex / (GRID_NX + 1U);
  *x = 0.75 * (double)column;
  *y = 0.50 * (double)row;
}

static int grid_append_edge(GridInput *fixture, size_t *count, size_t first,
                            size_t second) {
  const SsEntityId id = grid_edge_id(first, second);
  const SsEntityId first_id = grid_vertex_id(first < second ? first : second);
  const SsEntityId second_id = grid_vertex_id(first < second ? second : first);
  size_t existing;
  for (existing = 0U; existing < *count; ++existing) {
    if (fixture->intermediate_entities[existing].stable_entity_id == id) {
      return 0;
    }
  }
  if (*count >= 3U * (size_t)GRID_CELLS) {
    return 1;
  }
  fixture->intermediate_entities[*count] = (SsMeshIntermediateEntity){
      1U,
      id,
      {first_id, second_id, 0U},
  };
  ++*count;
  return 0;
}

static int grid_allocate(GridInput *fixture) {
  fixture->vertex_ids =
      (SsEntityId *)calloc((size_t)GRID_VERTICES, sizeof(*fixture->vertex_ids));
  fixture->vertex_coordinates = (double *)calloc(2U * (size_t)GRID_VERTICES,
                                                 sizeof(*fixture->vertex_coordinates));
  fixture->cell_ids =
      (SsEntityId *)calloc((size_t)GRID_CELLS, sizeof(*fixture->cell_ids));
  fixture->cells = (uint64_t *)calloc(3U * (size_t)GRID_CELLS, sizeof(*fixture->cells));
  fixture->facet_ids =
      (SsEntityId *)calloc((size_t)GRID_BOUNDARY_FACETS, sizeof(*fixture->facet_ids));
  fixture->facets =
      (uint64_t *)calloc(2U * (size_t)GRID_BOUNDARY_FACETS, sizeof(*fixture->facets));
  fixture->intermediate_entities = (SsMeshIntermediateEntity *)calloc(
      3U * (size_t)GRID_CELLS, sizeof(*fixture->intermediate_entities));
  fixture->labels = (SsMeshLabelAssignment *)calloc(
      (size_t)GRID_CELLS + (size_t)GRID_BOUNDARY_FACETS, sizeof(*fixture->labels));
  fixture->coordinate_node_ids = (SsEntityId *)calloc(
      (size_t)GRID_VERTICES, sizeof(*fixture->coordinate_node_ids));
  fixture->coordinate_coordinates = (double *)calloc(
      2U * (size_t)GRID_VERTICES, sizeof(*fixture->coordinate_coordinates));
  fixture->cell_coordinate_nodes = (uint64_t *)calloc(
      3U * (size_t)GRID_CELLS, sizeof(*fixture->cell_coordinate_nodes));
  return fixture->vertex_ids == NULL || fixture->vertex_coordinates == NULL ||
                 fixture->cell_ids == NULL || fixture->cells == NULL ||
                 fixture->facet_ids == NULL || fixture->facets == NULL ||
                 fixture->intermediate_entities == NULL || fixture->labels == NULL ||
                 fixture->coordinate_node_ids == NULL ||
                 fixture->coordinate_coordinates == NULL ||
                 fixture->cell_coordinate_nodes == NULL
             ? 1
             : 0;
}

static void grid_input_destroy(GridInput *fixture) {
  if (fixture == NULL) {
    return;
  }
  free(fixture->vertex_ids);
  free(fixture->vertex_coordinates);
  free(fixture->cell_ids);
  free(fixture->cells);
  free(fixture->facet_ids);
  free(fixture->facets);
  free(fixture->intermediate_entities);
  free(fixture->labels);
  free(fixture->coordinate_node_ids);
  free(fixture->coordinate_coordinates);
  free(fixture->cell_coordinate_nodes);
  memset(fixture, 0, sizeof(*fixture));
}

static int grid_append_boundary_facet(GridInput *fixture, int rank, int size,
                                      size_t parent_cell, size_t first, size_t second,
                                      size_t *facet_count, size_t *label_count) {
  if ((int)(parent_cell % (size_t)size) != rank) {
    return 0;
  }
  fixture->facet_ids[*facet_count] = grid_edge_id(first, second);
  fixture->facets[2U * *facet_count] = (uint64_t)first;
  fixture->facets[2U * *facet_count + 1U] = (uint64_t)second;
  fixture->labels[*label_count] = (SsMeshLabelAssignment){
      SS_MESH_LABEL_SUPPORT,
      grid_edge_id(first, second),
      7U,
  };
  ++*facet_count;
  ++*label_count;
  return 0;
}

static int grid_input_build(int rank, int size, GridInput *fixture) {
  size_t owned_vertex_start = 0U;
  size_t owned_vertex_count = 0U;
  size_t vertex;
  size_t rectangle_row;
  size_t rectangle_column;
  size_t local_cell = 0U;
  size_t local_facet = 0U;
  size_t label_count = 0U;
  size_t intermediate_count = 0U;
  memset(fixture, 0, sizeof(*fixture));
  if (grid_allocate(fixture) != 0) {
    grid_input_destroy(fixture);
    return 1;
  }
  owned_range((size_t)GRID_VERTICES, rank, size, &owned_vertex_start,
              &owned_vertex_count);
  for (vertex = 0U; vertex < owned_vertex_count; ++vertex) {
    const size_t global_vertex = owned_vertex_start + vertex;
    double x;
    double y;
    grid_coordinate(global_vertex, &x, &y);
    fixture->vertex_ids[vertex] = grid_vertex_id(global_vertex);
    fixture->vertex_coordinates[2U * vertex] = x;
    fixture->vertex_coordinates[2U * vertex + 1U] = y;
    fixture->coordinate_node_ids[vertex] = grid_vertex_id(global_vertex);
    fixture->coordinate_coordinates[2U * vertex] = x;
    fixture->coordinate_coordinates[2U * vertex + 1U] = y;
  }
  for (rectangle_row = 0U; rectangle_row < GRID_NY; ++rectangle_row) {
    for (rectangle_column = 0U; rectangle_column < GRID_NX; ++rectangle_column) {
      const size_t lower_left = grid_vertex(rectangle_column, rectangle_row);
      const size_t lower_right = grid_vertex(rectangle_column + 1U, rectangle_row);
      const size_t upper_left = grid_vertex(rectangle_column, rectangle_row + 1U);
      const size_t upper_right = grid_vertex(rectangle_column + 1U, rectangle_row + 1U);
      const size_t cell_ordinals[2] = {
          2U * (rectangle_row * GRID_NX + rectangle_column),
          2U * (rectangle_row * GRID_NX + rectangle_column) + 1U,
      };
      const size_t corners[2][3] = {
          {lower_left, lower_right, upper_left},
          {lower_right, upper_right, upper_left},
      };
      size_t triangle;
      for (triangle = 0U; triangle < 2U; ++triangle) {
        const size_t global_cell = cell_ordinals[triangle];
        size_t corner;
        if ((int)(global_cell % (size_t)size) != rank) {
          continue;
        }
        fixture->cell_ids[local_cell] = grid_cell_id(global_cell);
        for (corner = 0U; corner < 3U; ++corner) {
          fixture->cells[3U * local_cell + corner] =
              (uint64_t)corners[triangle][corner];
          fixture->cell_coordinate_nodes[3U * local_cell + corner] =
              (uint64_t)corners[triangle][corner];
        }
        if (grid_append_edge(fixture, &intermediate_count, corners[triangle][0],
                             corners[triangle][1]) != 0 ||
            grid_append_edge(fixture, &intermediate_count, corners[triangle][0],
                             corners[triangle][2]) != 0 ||
            grid_append_edge(fixture, &intermediate_count, corners[triangle][1],
                             corners[triangle][2]) != 0) {
          grid_input_destroy(fixture);
          return 1;
        }
        fixture->labels[label_count] = (SsMeshLabelAssignment){
            SS_MESH_LABEL_MATERIAL,
            grid_cell_id(global_cell),
            1U,
        };
        ++label_count;
        ++local_cell;
      }
    }
  }
  for (rectangle_column = 0U; rectangle_column < GRID_NX; ++rectangle_column) {
    const size_t bottom_parent = 2U * rectangle_column;
    const size_t top_parent = 2U * ((GRID_NY - 1U) * GRID_NX + rectangle_column) + 1U;
    (void)grid_append_boundary_facet(
        fixture, rank, size, bottom_parent, grid_vertex(rectangle_column, 0U),
        grid_vertex(rectangle_column + 1U, 0U), &local_facet, &label_count);
    (void)grid_append_boundary_facet(
        fixture, rank, size, top_parent, grid_vertex(rectangle_column + 1U, GRID_NY),
        grid_vertex(rectangle_column, GRID_NY), &local_facet, &label_count);
  }
  for (rectangle_row = 0U; rectangle_row < GRID_NY; ++rectangle_row) {
    const size_t left_parent = 2U * (rectangle_row * GRID_NX);
    const size_t right_parent = 2U * (rectangle_row * GRID_NX + GRID_NX - 1U) + 1U;
    (void)grid_append_boundary_facet(
        fixture, rank, size, left_parent, grid_vertex(0U, rectangle_row + 1U),
        grid_vertex(0U, rectangle_row), &local_facet, &label_count);
    (void)grid_append_boundary_facet(
        fixture, rank, size, right_parent, grid_vertex(GRID_NX, rectangle_row),
        grid_vertex(GRID_NX, rectangle_row + 1U), &local_facet, &label_count);
  }
  fixture->input = (SsMeshInput){
      .dimension = 2U,
      .global_vertex_count = GRID_VERTICES,
      .global_cell_count = GRID_CELLS,
      .global_boundary_facet_count = GRID_BOUNDARY_FACETS,
      .global_intermediate_entity_count = GRID_EDGES,
      .global_coordinate_node_count = GRID_VERTICES,
      .vertex_count = owned_vertex_count,
      .cell_count = local_cell,
      .boundary_facet_count = local_facet,
      .vertex_ids = fixture->vertex_ids,
      .vertex_coordinates = fixture->vertex_coordinates,
      .cell_ids = fixture->cell_ids,
      .cells = fixture->cells,
      .boundary_facet_ids = fixture->facet_ids,
      .boundary_facets = fixture->facets,
      .intermediate_entity_count = intermediate_count,
      .intermediate_entities = fixture->intermediate_entities,
      .label_count = label_count,
      .labels = fixture->labels,
      .coordinate_order = 1U,
      .coordinate_node_count = owned_vertex_count,
      .coordinate_node_ids = fixture->coordinate_node_ids,
      .coordinate_coordinates = fixture->coordinate_coordinates,
      .cell_coordinate_nodes = fixture->cell_coordinate_nodes,
      .absolute_determinant_tolerance = 1.0e-12,
      .relative_determinant_tolerance = 1.0e-12,
  };
  return 0;
}

static SsStatus distributed_scalar_kernel(const void *raw_context,
                                          const SsScalarPointInput *input,
                                          SsScalarPointOutput *output) {
  const DistributedScalarContext *context =
      (const DistributedScalarContext *)raw_context;
  double gradient_squared = 0.0;
  uint32_t coordinate;
  if (context == NULL || input == NULL || output == NULL || input->dimension != 2U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(output, 0, sizeof(*output));
  output->flux[0] = context->conductivity[0] * input->gradient[0] +
                    context->conductivity[1] * input->gradient[1] +
                    context->flux_value[0] * input->value;
  output->flux[1] = context->conductivity[2] * input->gradient[0] +
                    context->conductivity[3] * input->gradient[1] +
                    context->flux_value[1] * input->value;
  output->flux_derivative_value[0] = context->flux_value[0];
  output->flux_derivative_value[1] = context->flux_value[1];
  memcpy(output->flux_derivative_gradient, context->conductivity,
         sizeof(context->conductivity));
  output->source = context->source + context->reaction * input->value;
  output->source_derivative_value = context->reaction;
  for (coordinate = 0U; coordinate < input->dimension; ++coordinate) {
    gradient_squared += input->gradient[coordinate] * input->gradient[coordinate];
  }
  output->potential_density = 0.5 * gradient_squared - context->source * input->value;
  return SS_OK;
}

static double distributed_field_value(double x, double y) {
  return 0.31 + 0.07 * x - 0.04 * y + 0.013 * x * y;
}

static SsStatus distributed_natural_kernel(const void *raw_context,
                                           const SsNaturalPointInput *input,
                                           SsNaturalPointOutput *output) {
  const DistributedNaturalContext *context =
      (const DistributedNaturalContext *)raw_context;
  if (context == NULL || input == NULL || output == NULL || input->components != 1U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(output, 0, sizeof(*output));
  output->load[0] = context->offset + context->derivative * input->value[0];
  output->load_derivative[0] = context->derivative;
  output->signed_flux = output->load[0];
  return SS_OK;
}

static void service_observer(SsAssemblyService service, int in_parallel, int thread,
                             void *raw_context) {
  ServiceObservation *observation = (ServiceObservation *)raw_context;
  (void)thread;
  if (observation == NULL) {
    return;
  }
  if (service == SS_ASSEMBLY_SERVICE_MPI) {
    ++observation->mpi_calls;
  } else if (service == SS_ASSEMBLY_SERVICE_PETSC) {
    ++observation->petsc_calls;
  } else if (service == SS_ASSEMBLY_SERVICE_HDF5) {
    ++observation->hdf5_calls;
  }
  if (in_parallel != 0) {
    ++observation->parallel_calls;
  }
}

static int operator_fixture_initialize_field(OperatorFixture *fixture) {
  SsCellPack *packs = NULL;
  unsigned char *seen = NULL;
  PetscScalar *local_values = NULL;
  PetscInt row_start = 0;
  PetscInt row_end = 0;
  size_t pack_count = 0U;
  size_t pack;
  int result = 1;
  if (ss_field_space_pack_incident_cells(fixture->space, NULL, 0U, &pack_count) !=
          SS_OK ||
      VecGetOwnershipRange(fixture->field, &row_start, &row_end) != PETSC_SUCCESS ||
      row_end < row_start) {
    return 1;
  }
  if (pack_count != 0U) {
    packs = (SsCellPack *)calloc(pack_count, sizeof(*packs));
  }
  if (row_end > row_start) {
    seen = (unsigned char *)calloc((size_t)(row_end - row_start), sizeof(*seen));
  }
  if ((pack_count != 0U && packs == NULL) || (row_end > row_start && seen == NULL) ||
      ss_field_space_pack_incident_cells(fixture->space, packs, pack_count,
                                         &pack_count) != SS_OK ||
      VecSet(fixture->field, 0.0) != PETSC_SUCCESS ||
      VecGetArray(fixture->field, &local_values) != PETSC_SUCCESS) {
    goto cleanup;
  }
  for (pack = 0U; pack < pack_count; ++pack) {
    size_t slot;
    for (slot = 0U; slot < packs[pack].closure_count; ++slot) {
      const PetscInt global = packs[pack].global_indices[slot];
      const size_t node = slot;
      size_t local;
      double value;
      if (packs[pack].constrained[slot] != 0U || global < row_start ||
          global >= row_end || node >= packs[pack].field_node_count) {
        continue;
      }
      local = (size_t)(global - row_start);
      value =
          distributed_field_value(packs[pack].field_node_coordinates[2U * node],
                                  packs[pack].field_node_coordinates[2U * node + 1U]);
      if (seen[local] != 0U && fabs((double)local_values[local] - value) > 2.0e-14) {
        goto restore;
      }
      local_values[local] = (PetscScalar)value;
      seen[local] = 1U;
    }
  }
  {
    size_t local;
    for (local = 0U; local < (size_t)(row_end - row_start); ++local) {
      if (seen[local] == 0U) {
        goto restore;
      }
    }
  }
  result = 0;

restore:
  if (VecRestoreArray(fixture->field, &local_values) != PETSC_SUCCESS) {
    result = 1;
  }

cleanup:
  free(packs);
  free(seen);
  return result;
}

static int operator_fixture_create(MPI_Comm communicator, int rank, int size,
                                   int requested_threads, OperatorFixture *fixture) {
  const SsFieldSpec field_spec = {2U, 1U, 0U, NULL};
  SsMeshError mesh_error = {0};
  SsErrorRecord error = {0};
  SsStatus status;
  memset(fixture, 0, sizeof(*fixture));
  fixture->context = (DistributedScalarContext){
      {2.4, 0.35, -0.12, 1.7},
      {0.17, -0.08},
      -0.23,
      0.41,
  };
  fixture->scalar_spec = (SsScalarOperatorSpec){
      distributed_scalar_kernel,
      &fixture->context,
  };
  fixture->natural_context = (DistributedNaturalContext){0.19, 0.27};
  fixture->natural_spec = (SsNaturalOperatorSpec){
      7U,
      distributed_natural_kernel,
      &fixture->natural_context,
      1,
  };
  fixture->operator_spec = (SsOperatorSpec){
      SS_ASSEMBLY_SCALAR_FLUX,
      SS_ASSEMBLY_QUALIFICATION,
      0U,
      0U,
      requested_threads,
      NULL,
      &fixture->scalar_spec,
      &fixture->natural_spec,
      1U,
      0,
      0U,
      0U,
      SS_FE_QUADRATURE_MODE_DEFAULT,
  };
  if (grid_input_build(rank, size, &fixture->input) != 0) {
    return 1;
  }
  status =
      ss_mesh_create(communicator, &fixture->input.input, &fixture->mesh, &mesh_error);
  if (status != SS_OK ||
      ss_mesh_summary(fixture->mesh, &fixture->mesh_summary) != SS_OK ||
      ss_mesh_field_space_create(fixture->mesh, &field_spec, &fixture->space) !=
          SS_OK ||
      ss_field_space_summary(fixture->space, &fixture->field_summary) != SS_OK ||
      ss_assembly_plan_create(communicator, fixture->space, &fixture->plan, &error) !=
          SS_OK ||
      ss_assembly_plan_summary(fixture->plan, &fixture->plan_summary) != SS_OK ||
      ss_operator_create(fixture->plan, &fixture->operator_spec,
                         &fixture->operator_handle, &error) != SS_OK ||
      ss_assembly_plan_create_vector(fixture->plan, &fixture->field) != SS_OK ||
      ss_assembly_plan_create_vector(fixture->plan, &fixture->residual) != SS_OK ||
      ss_assembly_plan_create_matrix(fixture->plan, &fixture->jacobian) != SS_OK ||
      operator_fixture_initialize_field(fixture) != 0 ||
      ss_operator_set_service_observer(fixture->operator_handle, service_observer,
                                       &fixture->observation) != SS_OK) {
    fprintf(stderr,
            "rank %d: distributed operator fixture failed: status=%d mesh=%s "
            "assembly=%s\n",
            rank, (int)status, mesh_error.code, error.code);
    return 1;
  }
  fixture->state = (SsOperatorState){
      .field = fixture->field,
      .reduction_factor = 1.0,
      .external_load_factor = 1.0,
      .cancelled = 0U,
  };
  return 0;
}

static int operator_fixture_destroy(OperatorFixture *fixture) {
  int result = 0;
  if (MatDestroy(&fixture->jacobian) != PETSC_SUCCESS ||
      VecDestroy(&fixture->residual) != PETSC_SUCCESS ||
      VecDestroy(&fixture->field) != PETSC_SUCCESS ||
      ss_operator_destroy(&fixture->operator_handle) != SS_OK ||
      ss_assembly_plan_destroy(&fixture->plan) != SS_OK ||
      ss_field_space_destroy(&fixture->space) != SS_OK ||
      ss_mesh_destroy(&fixture->mesh) != SS_OK) {
    result = 1;
  }
  grid_input_destroy(&fixture->input);
  return result;
}

static int compare_destinations(Vec vector, Mat matrix, Vec baseline_vector,
                                Mat baseline_matrix) {
  Vec vector_difference = NULL;
  Mat matrix_difference = NULL;
  PetscReal vector_norm = -1.0;
  PetscReal matrix_norm = -1.0;
  int result = 1;
  if (VecDuplicate(vector, &vector_difference) != PETSC_SUCCESS ||
      VecCopy(vector, vector_difference) != PETSC_SUCCESS ||
      VecAXPY(vector_difference, -1.0, baseline_vector) != PETSC_SUCCESS ||
      VecNorm(vector_difference, NORM_2, &vector_norm) != PETSC_SUCCESS ||
      MatDuplicate(matrix, MAT_COPY_VALUES, &matrix_difference) != PETSC_SUCCESS ||
      MatAXPY(matrix_difference, -1.0, baseline_matrix, SAME_NONZERO_PATTERN) !=
          PETSC_SUCCESS ||
      MatNorm(matrix_difference, NORM_FROBENIUS, &matrix_norm) != PETSC_SUCCESS) {
    goto cleanup;
  }
  result = vector_norm == 0.0 && matrix_norm == 0.0 ? 0 : 1;

cleanup:
  (void)MatDestroy(&matrix_difference);
  (void)VecDestroy(&vector_difference);
  return result;
}

static int set_transaction_sentinel(OperatorFixture *fixture, Vec *baseline_vector,
                                    Mat *baseline_matrix) {
  PetscInt row_start = 0;
  PetscInt row_end = 0;
  PetscInt row;
  if (VecSet(fixture->residual, 9.25) != PETSC_SUCCESS ||
      MatZeroEntries(fixture->jacobian) != PETSC_SUCCESS ||
      MatGetOwnershipRange(fixture->jacobian, &row_start, &row_end) != PETSC_SUCCESS) {
    return 1;
  }
  for (row = row_start; row < row_end; ++row) {
    if (MatSetValue(fixture->jacobian, row, row, 7.5, INSERT_VALUES) != PETSC_SUCCESS) {
      return 1;
    }
  }
  if (MatAssemblyBegin(fixture->jacobian, MAT_FINAL_ASSEMBLY) != PETSC_SUCCESS ||
      MatAssemblyEnd(fixture->jacobian, MAT_FINAL_ASSEMBLY) != PETSC_SUCCESS ||
      VecDuplicate(fixture->residual, baseline_vector) != PETSC_SUCCESS ||
      VecCopy(fixture->residual, *baseline_vector) != PETSC_SUCCESS ||
      MatDuplicate(fixture->jacobian, MAT_COPY_VALUES, baseline_matrix) !=
          PETSC_SUCCESS) {
    return 1;
  }
  return 0;
}

static int test_transaction(OperatorFixture *fixture) {
  static const struct {
    SsAssemblyFailurePoint point;
    SsStatus expected;
  } failures[] = {
      {SS_ASSEMBLY_FAIL_PACK, SS_ERROR_INJECTED},
      {SS_ASSEMBLY_FAIL_WORKER, SS_ERROR_INJECTED},
      {SS_ASSEMBLY_FAIL_NONFINITE, SS_ERROR_INTEGRITY},
      {SS_ASSEMBLY_FAIL_CANCEL_AFTER_PACK, SS_ERROR_CANCELLED},
      {SS_ASSEMBLY_FAIL_CANCEL_AFTER_COMPUTE, SS_ERROR_CANCELLED},
      {SS_ASSEMBLY_FAIL_PRECOMMIT, SS_ERROR_INJECTED},
  };
  size_t failure;
  for (failure = 0U; failure < sizeof(failures) / sizeof(failures[0]); ++failure) {
    Vec baseline_vector = NULL;
    Mat baseline_matrix = NULL;
    SsOperatorProperties before = {0};
    SsOperatorProperties after = {0};
    SsErrorRecord error = {0};
    SsStatus status;
    CHECK(set_transaction_sentinel(fixture, &baseline_vector, &baseline_matrix) == 0);
    CHECK(ss_operator_properties(fixture->operator_handle, &before) == SS_OK);
    CHECK(ss_operator_inject_failure(fixture->operator_handle, failures[failure].point,
                                     0, 0U) == SS_OK);
    status =
        ss_operator_residual_jacobian(fixture->operator_handle, &fixture->state,
                                      fixture->residual, fixture->jacobian, &error);
    CHECK(status == failures[failure].expected);
    CHECK(error.valid != 0);
    CHECK(compare_destinations(fixture->residual, fixture->jacobian, baseline_vector,
                               baseline_matrix) == 0);
    CHECK(ss_operator_properties(fixture->operator_handle, &after) == SS_OK);
    CHECK(after.value_generation == before.value_generation);
    CHECK(MatDestroy(&baseline_matrix) == PETSC_SUCCESS);
    CHECK(VecDestroy(&baseline_vector) == PETSC_SUCCESS);
  }
  return local_status;
}

static int compare_performance_destinations(Vec qualification_vector,
                                            Mat qualification_matrix,
                                            Vec performance_vector,
                                            Mat performance_matrix) {
  Vec vector_difference = NULL;
  Mat matrix_difference = NULL;
  PetscReal vector_difference_norm = -1.0;
  PetscReal matrix_difference_norm = -1.0;
  PetscReal vector_reference_norm = -1.0;
  PetscReal matrix_reference_norm = -1.0;
  int result = 1;
  if (VecDuplicate(qualification_vector, &vector_difference) != PETSC_SUCCESS ||
      VecCopy(qualification_vector, vector_difference) != PETSC_SUCCESS ||
      VecAXPY(vector_difference, -1.0, performance_vector) != PETSC_SUCCESS ||
      VecNorm(vector_difference, NORM_2, &vector_difference_norm) != PETSC_SUCCESS ||
      VecNorm(qualification_vector, NORM_2, &vector_reference_norm) != PETSC_SUCCESS ||
      MatDuplicate(qualification_matrix, MAT_COPY_VALUES, &matrix_difference) !=
          PETSC_SUCCESS ||
      MatAXPY(matrix_difference, -1.0, performance_matrix, SAME_NONZERO_PATTERN) !=
          PETSC_SUCCESS ||
      MatNorm(matrix_difference, NORM_FROBENIUS, &matrix_difference_norm) !=
          PETSC_SUCCESS ||
      MatNorm(qualification_matrix, NORM_FROBENIUS, &matrix_reference_norm) !=
          PETSC_SUCCESS) {
    goto cleanup;
  }
  result = vector_difference_norm <= 1.0e-12 * (1.0 + vector_reference_norm) &&
                   matrix_difference_norm <= 1.0e-12 * (1.0 + matrix_reference_norm)
               ? 0
               : 1;

cleanup:
  (void)MatDestroy(&matrix_difference);
  (void)VecDestroy(&vector_difference);
  return result;
}

static int test_performance_mode(MPI_Comm communicator, OperatorFixture *fixture,
                                 int size, int imbalanced_coo_admission) {
  SsOperatorSpec performance_spec = fixture->operator_spec;
  SsOperator *performance_operator = NULL;
  SsOperator *qualification_operator = fixture->operator_handle;
  Vec performance_residual = NULL;
  Mat performance_jacobian = NULL;
  SsAssemblyDiagnostics qualification_diagnostics = {0};
  SsAssemblyDiagnostics performance_diagnostics = {0};
  SsErrorRecord error = {0};
  uint64_t qualification_cell_evaluations = 0U;
  uint64_t qualification_cell_duplicates = 0U;
  uint64_t performance_cell_evaluations = 0U;
  uint64_t performance_cell_duplicates = 0U;
  uint64_t qualification_facet_evaluations = 0U;
  uint64_t qualification_facet_duplicates = 0U;
  uint64_t performance_facet_evaluations = 0U;
  uint64_t performance_facet_duplicates = 0U;
  performance_spec.determinism = SS_ASSEMBLY_PERFORMANCE;
  CHECK(ss_operator_residual_jacobian(qualification_operator, &fixture->state,
                                      fixture->residual, fixture->jacobian,
                                      &error) == SS_OK);
  CHECK(error.valid == 0);
  CHECK(ss_operator_diagnostics(qualification_operator, &qualification_diagnostics) ==
        SS_OK);
  if (imbalanced_coo_admission != 0 && test_rank != 0) {
    /*
     * The base workspace fits exactly, but the additional COO storage does
     * not.  Rank zero retains the default cap and is locally COO-eligible.
     */
    performance_spec.memory_cap_bytes =
        (size_t)qualification_diagnostics.admitted_peak_bytes;
  }
  CHECK(ss_operator_create(fixture->plan, &performance_spec, &performance_operator,
                           &error) == SS_OK);
  CHECK(error.valid == 0);
  CHECK(ss_assembly_plan_create_vector(fixture->plan, &performance_residual) == SS_OK);
  CHECK(ss_assembly_plan_create_matrix(fixture->plan, &performance_jacobian) == SS_OK);
  if (local_status == 0) {
    CHECK(ss_operator_residual_jacobian(performance_operator, &fixture->state,
                                        performance_residual, performance_jacobian,
                                        &error) == SS_OK);
    CHECK(error.valid == 0);
    CHECK(ss_operator_diagnostics(performance_operator, &performance_diagnostics) ==
          SS_OK);
    if (imbalanced_coo_admission != 0) {
      CHECK(performance_diagnostics.admitted_peak_bytes ==
            qualification_diagnostics.admitted_peak_bytes);
    }
    if (performance_spec.requested_threads > 1) {
      CHECK(performance_diagnostics.parallel_pack_calls > 0U);
      CHECK(performance_diagnostics.parallel_stage_calls > 0U);
      CHECK(performance_diagnostics.parallel_stage_entities > 0U);
    } else {
      CHECK(performance_diagnostics.parallel_pack_calls == 0U);
      CHECK(performance_diagnostics.parallel_stage_calls == 0U);
      CHECK(performance_diagnostics.parallel_stage_entities == 0U);
    }
    CHECK(compare_performance_destinations(fixture->residual, fixture->jacobian,
                                           performance_residual,
                                           performance_jacobian) == 0);
  }
  CHECK(MPI_Allreduce(&qualification_diagnostics.cell_evaluations,
                      &qualification_cell_evaluations, 1, MPI_UINT64_T, MPI_SUM,
                      communicator) == MPI_SUCCESS);
  CHECK(MPI_Allreduce(&qualification_diagnostics.duplicate_cell_evaluations,
                      &qualification_cell_duplicates, 1, MPI_UINT64_T, MPI_SUM,
                      communicator) == MPI_SUCCESS);
  CHECK(MPI_Allreduce(&performance_diagnostics.cell_evaluations,
                      &performance_cell_evaluations, 1, MPI_UINT64_T, MPI_SUM,
                      communicator) == MPI_SUCCESS);
  CHECK(MPI_Allreduce(&performance_diagnostics.duplicate_cell_evaluations,
                      &performance_cell_duplicates, 1, MPI_UINT64_T, MPI_SUM,
                      communicator) == MPI_SUCCESS);
  CHECK(MPI_Allreduce(&qualification_diagnostics.natural_facet_evaluations,
                      &qualification_facet_evaluations, 1, MPI_UINT64_T, MPI_SUM,
                      communicator) == MPI_SUCCESS);
  CHECK(MPI_Allreduce(&qualification_diagnostics.duplicate_natural_facet_evaluations,
                      &qualification_facet_duplicates, 1, MPI_UINT64_T, MPI_SUM,
                      communicator) == MPI_SUCCESS);
  CHECK(MPI_Allreduce(&performance_diagnostics.natural_facet_evaluations,
                      &performance_facet_evaluations, 1, MPI_UINT64_T, MPI_SUM,
                      communicator) == MPI_SUCCESS);
  CHECK(MPI_Allreduce(&performance_diagnostics.duplicate_natural_facet_evaluations,
                      &performance_facet_duplicates, 1, MPI_UINT64_T, MPI_SUM,
                      communicator) == MPI_SUCCESS);
  CHECK(qualification_cell_evaluations ==
        (uint64_t)GRID_CELLS + qualification_cell_duplicates);
  CHECK(performance_cell_evaluations == (uint64_t)GRID_CELLS);
  CHECK(performance_cell_duplicates == 0U);
  CHECK(qualification_facet_evaluations ==
        (uint64_t)GRID_BOUNDARY_FACETS + qualification_facet_duplicates);
  CHECK(performance_facet_evaluations == (uint64_t)GRID_BOUNDARY_FACETS);
  CHECK(performance_facet_duplicates == 0U);
  CHECK(size == 1 ? qualification_cell_duplicates == 0U
                  : qualification_cell_duplicates > 0U);
  CHECK(size == 1 ? qualification_facet_duplicates == 0U
                  : qualification_facet_duplicates > 0U);
  if (local_status == 0) {
    fixture->operator_handle = performance_operator;
    CHECK(test_transaction(fixture) == 0);
    fixture->operator_handle = qualification_operator;
  }
  if (fixture->operator_handle != qualification_operator) {
    fixture->operator_handle = qualification_operator;
  }
  CHECK(MatDestroy(&performance_jacobian) == PETSC_SUCCESS);
  CHECK(VecDestroy(&performance_residual) == PETSC_SUCCESS);
  CHECK(ss_operator_destroy(&performance_operator) == SS_OK);
  return local_status;
}

static int test_compatible_source_reuse(OperatorFixture *fixture, int rank) {
  Vec alternate = NULL;
  SsOperatorState state = fixture->state;
  SsErrorRecord error = {0};
  CHECK(VecDuplicate(fixture->field, &alternate) == PETSC_SUCCESS);
  CHECK(VecCopy(fixture->field, alternate) == PETSC_SUCCESS);
  if (local_status == 0 && rank == 0) {
    state.field = alternate;
  }
  if (local_status == 0) {
    CHECK(ss_operator_residual(fixture->operator_handle, &state, fixture->residual,
                               &error) == SS_OK);
    CHECK(error.valid == 0);
  }
  CHECK(VecDestroy(&alternate) == PETSC_SUCCESS);
  return local_status;
}

static int test_matrix_free_p1_projection(MPI_Comm communicator, SsMesh *mesh,
                                          int requested_threads) {
  const SsMcParameterInput material_input = {
      8.0e4,
      0.27,
      120.0,
      0.45,
      0.2,
      SS_MC_DAVIS_B,
      UINT64_C(1),
      "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  };
  const SsFieldSpec fine_spec = {4U, 2U, 0U, NULL};
  const SsFieldSpec p1_spec = {1U, 2U, 0U, NULL};
  SsMcParameters *material = NULL;
  SsMcError material_error = {0};
  SsMechanicsMaterialRegion region = {0};
  SsMechanicsOperatorSpec mechanics = {0};
  SsOperatorSpec matrix_free_spec = {0};
  SsOperatorSpec assembled_spec = {0};
  SsFieldSpace *fine_space = NULL;
  SsFieldSpace *p1_space = NULL;
  SsAssemblyPlan *fine_plan = NULL;
  SsAssemblyPlan *p1_plan = NULL;
  SsOperator *matrix_free_operator = NULL;
  SsOperator *coarse_operator = NULL;
  SsOperator *assembled_operator = NULL;
  Vec field = NULL;
  Vec fine_expected_action = NULL;
  Vec fine_projected_action = NULL;
  Vec fine_expected_transpose = NULL;
  Vec fine_projected_transpose = NULL;
  Vec coarse_field = NULL;
  Vec coarse_direction = NULL;
  Vec expected_action = NULL;
  Vec projected_action = NULL;
  Mat shell = NULL;
  Mat coarse_shell = NULL;
  Mat assembled = NULL;
  Mat transfer = NULL;
  Mat expected = NULL;
  Mat projected = NULL;
  Mat difference = NULL;
  SsOperatorState state = {0};
  SsErrorRecord error = {0};
  PetscInt row_start = 0;
  PetscInt row_end = 0;
  PetscScalar *field_values = NULL;
  PetscReal expected_norm = 0.0;
  PetscReal difference_norm = 0.0;
  PetscReal fine_action_norm = 0.0;
  PetscReal fine_action_difference_norm = 0.0;
  PetscReal action_norm = 0.0;
  PetscReal action_difference_norm = 0.0;
  PetscInt row;
  int result = 1;

  if (ss_mc_parameters_create(&material_input, &material, &material_error) != SS_OK) {
    fprintf(stderr, "rank %d: matrix-free material failed: %s\n", test_rank,
            material_error.code);
    goto cleanup;
  }
  region.material_value = 1U;
  region.parameters = material;
  mechanics.regions = &region;
  mechanics.region_count = 1U;
  matrix_free_spec.physics = SS_ASSEMBLY_MECHANICS;
  matrix_free_spec.determinism = SS_ASSEMBLY_PERFORMANCE;
  matrix_free_spec.openmp_work_threshold = 1U;
  matrix_free_spec.requested_threads = requested_threads;
  matrix_free_spec.mechanics = &mechanics;
  matrix_free_spec.matrix_free = 1;
  assembled_spec = matrix_free_spec;
  assembled_spec.matrix_free = 0;
  if (ss_mesh_field_space_create(mesh, &fine_spec, &fine_space) != SS_OK ||
      ss_mesh_field_space_create(mesh, &p1_spec, &p1_space) != SS_OK ||
      ss_assembly_plan_create(communicator, fine_space, &fine_plan, &error) != SS_OK ||
      ss_assembly_plan_create(communicator, p1_space, &p1_plan, &error) != SS_OK ||
      ss_operator_create(fine_plan, &matrix_free_spec, &matrix_free_operator, &error) !=
          SS_OK ||
      ss_operator_create(fine_plan, &assembled_spec, &assembled_operator, &error) !=
          SS_OK ||
      ss_operator_create(p1_plan, &matrix_free_spec, &coarse_operator, &error) !=
          SS_OK ||
      ss_assembly_plan_create_vector(fine_plan, &field) != SS_OK ||
      ss_assembly_plan_create_vector(p1_plan, &coarse_field) != SS_OK ||
      ss_assembly_plan_create_matrix(fine_plan, &assembled) != SS_OK ||
      ss_assembly_plan_create_matrix(p1_plan, &projected) != SS_OK ||
      ss_field_space_transfer_create(p1_space, fine_space, &transfer) != SS_OK ||
      ss_operator_matrix_free_create(matrix_free_operator, &shell, &error) != SS_OK ||
      ss_operator_matrix_free_create(coarse_operator, &coarse_shell, &error) != SS_OK ||
      VecGetOwnershipRange(field, &row_start, &row_end) != PETSC_SUCCESS ||
      VecGetArray(field, &field_values) != PETSC_SUCCESS) {
    fprintf(stderr, "rank %d: matrix-free projection fixture failed: %s\n", test_rank,
            error.code);
    goto cleanup;
  }
  for (row = row_start; row < row_end; ++row) {
    field_values[row - row_start] =
        (PetscScalar)(0.006 * sin(0.17 * (double)(row + 1)) +
                      0.002 * cos(0.31 * (double)(row + 3)));
  }
  if (VecRestoreArray(field, &field_values) != PETSC_SUCCESS) {
    field_values = NULL;
    goto cleanup;
  }
  field_values = NULL;
  state.field = field;
  state.reduction_factor = 1.3;
  state.external_load_factor = 0.0;
  if (ss_operator_jacobian(assembled_operator, &state, assembled, &error) != SS_OK ||
      ss_operator_matrix_free_linearize(matrix_free_operator, &state, 0.0, &error) !=
          SS_OK ||
      VecDuplicate(field, &fine_expected_action) != PETSC_SUCCESS ||
      VecDuplicate(field, &fine_projected_action) != PETSC_SUCCESS ||
      VecDuplicate(field, &fine_expected_transpose) != PETSC_SUCCESS ||
      VecDuplicate(field, &fine_projected_transpose) != PETSC_SUCCESS ||
      MatMult(assembled, field, fine_expected_action) != PETSC_SUCCESS ||
      MatMult(shell, field, fine_projected_action) != PETSC_SUCCESS ||
      VecNorm(fine_expected_action, NORM_2, &fine_action_norm) != PETSC_SUCCESS ||
      VecAXPY(fine_projected_action, -1.0, fine_expected_action) != PETSC_SUCCESS ||
      VecNorm(fine_projected_action, NORM_2, &fine_action_difference_norm) !=
          PETSC_SUCCESS ||
      !isfinite((double)fine_action_norm) ||
      !isfinite((double)fine_action_difference_norm) ||
      (double)fine_action_difference_norm >
          2.0e-11 * (1.0 + (double)fine_action_norm) ||
      MatMultTranspose(assembled, field, fine_expected_transpose) != PETSC_SUCCESS ||
      MatMultTranspose(shell, field, fine_projected_transpose) != PETSC_SUCCESS ||
      VecNorm(fine_expected_transpose, NORM_2, &fine_action_norm) != PETSC_SUCCESS ||
      VecAXPY(fine_projected_transpose, -1.0, fine_expected_transpose) !=
          PETSC_SUCCESS ||
      VecNorm(fine_projected_transpose, NORM_2, &fine_action_difference_norm) !=
          PETSC_SUCCESS ||
      !isfinite((double)fine_action_norm) ||
      !isfinite((double)fine_action_difference_norm) ||
      (double)fine_action_difference_norm >
          2.0e-11 * (1.0 + (double)fine_action_norm) ||
      ss_operator_matrix_free_project_p1_jacobian(matrix_free_operator, p1_plan,
                                                  projected, &error) != SS_OK ||
      MatPtAP(assembled, transfer, MAT_INITIAL_MATRIX, 2.0, &expected) !=
          PETSC_SUCCESS ||
      MatDuplicate(projected, MAT_COPY_VALUES, &difference) != PETSC_SUCCESS ||
      MatAXPY(difference, -1.0, expected, DIFFERENT_NONZERO_PATTERN) != PETSC_SUCCESS ||
      MatNorm(expected, NORM_FROBENIUS, &expected_norm) != PETSC_SUCCESS ||
      MatNorm(difference, NORM_FROBENIUS, &difference_norm) != PETSC_SUCCESS) {
    fprintf(stderr, "rank %d: matrix-free P1 projection failed: %s\n", test_rank,
            error.code);
    goto cleanup;
  }
  if (!isfinite((double)expected_norm) || !isfinite((double)difference_norm) ||
      (double)difference_norm > 2.0e-11 * (1.0 + (double)expected_norm)) {
    fprintf(stderr,
            "rank %d: matrix-free P1 projection mismatch: |A|=%.17g |dA|=%.17g\n",
            test_rank, (double)expected_norm, (double)difference_norm);
    goto cleanup;
  }
  if (ss_operator_matrix_free_project_coarse(matrix_free_operator, coarse_operator,
                                             &error) != SS_OK ||
      VecDuplicate(coarse_field, &coarse_direction) != PETSC_SUCCESS ||
      VecDuplicate(coarse_field, &expected_action) != PETSC_SUCCESS ||
      VecDuplicate(coarse_field, &projected_action) != PETSC_SUCCESS ||
      VecGetOwnershipRange(coarse_direction, &row_start, &row_end) != PETSC_SUCCESS ||
      VecGetArray(coarse_direction, &field_values) != PETSC_SUCCESS) {
    fprintf(stderr, "rank %d: matrix-free coarse action fixture failed: %s\n",
            test_rank, error.code);
    goto cleanup;
  }
  for (row = row_start; row < row_end; ++row) {
    field_values[row - row_start] = (PetscScalar)(0.19 * sin(0.23 * (double)(row + 2)) -
                                                  0.11 * cos(0.37 * (double)(row + 1)));
  }
  if (VecRestoreArray(coarse_direction, &field_values) != PETSC_SUCCESS) {
    field_values = NULL;
    goto cleanup;
  }
  field_values = NULL;
  if (MatMult(expected, coarse_direction, expected_action) != PETSC_SUCCESS ||
      MatMult(coarse_shell, coarse_direction, projected_action) != PETSC_SUCCESS ||
      VecNorm(expected_action, NORM_2, &action_norm) != PETSC_SUCCESS ||
      VecAXPY(projected_action, -1.0, expected_action) != PETSC_SUCCESS ||
      VecNorm(projected_action, NORM_2, &action_difference_norm) != PETSC_SUCCESS ||
      !isfinite((double)action_norm) || !isfinite((double)action_difference_norm) ||
      (double)action_difference_norm > 2.0e-11 * (1.0 + (double)action_norm)) {
    fprintf(stderr,
            "rank %d: matrix-free coarse action mismatch: |Ax|=%.17g |dAx|=%.17g\n",
            test_rank, (double)action_norm, (double)action_difference_norm);
    goto cleanup;
  }
  /* Inject on the first rank owning action rows. Every peer must still
   * enter the input halo and observe the failure at the next norm. */
  int owns_rows = row_end > row_start ? test_rank : INT_MAX;
  int failed_rank = INT_MAX;
  if (MPI_Allreduce(&owns_rows, &failed_rank, 1, MPI_INT, MPI_MIN, communicator) !=
          MPI_SUCCESS ||
      failed_rank == INT_MAX ||
      (test_rank == failed_rank &&
       ss_operator_inject_failure(coarse_operator, SS_ASSEMBLY_FAIL_ACTION, test_rank,
                                  0U) != SS_OK) ||
      MatMult(coarse_shell, coarse_direction, projected_action) != PETSC_SUCCESS ||
      VecNorm(projected_action, NORM_2, &action_norm) != PETSC_SUCCESS ||
      isfinite((double)action_norm) ||
      ss_operator_matrix_free_take_action_failure(coarse_shell) !=
          (test_rank == failed_rank) ||
      MatMult(coarse_shell, coarse_direction, projected_action) != PETSC_SUCCESS ||
      VecNorm(projected_action, NORM_2, &action_norm) != PETSC_SUCCESS ||
      !isfinite((double)action_norm)) {
    fprintf(stderr, "rank %d: matrix-free action failure did not propagate or clear\n",
            test_rank);
    goto cleanup;
  }
  result = 0;

cleanup:
  if (field_values != NULL) {
    (void)VecRestoreArray(field, &field_values);
  }
  (void)MatDestroy(&difference);
  (void)MatDestroy(&projected);
  (void)MatDestroy(&expected);
  (void)MatDestroy(&transfer);
  (void)MatDestroy(&assembled);
  (void)MatDestroy(&coarse_shell);
  (void)MatDestroy(&shell);
  (void)VecDestroy(&projected_action);
  (void)VecDestroy(&expected_action);
  (void)VecDestroy(&coarse_direction);
  (void)VecDestroy(&coarse_field);
  (void)VecDestroy(&fine_projected_transpose);
  (void)VecDestroy(&fine_expected_transpose);
  (void)VecDestroy(&fine_projected_action);
  (void)VecDestroy(&fine_expected_action);
  (void)VecDestroy(&field);
  (void)ss_operator_destroy(&assembled_operator);
  (void)ss_operator_destroy(&coarse_operator);
  (void)ss_operator_destroy(&matrix_free_operator);
  (void)ss_assembly_plan_destroy(&p1_plan);
  (void)ss_assembly_plan_destroy(&fine_plan);
  (void)ss_field_space_destroy(&p1_space);
  (void)ss_field_space_destroy(&fine_space);
  (void)ss_mc_parameters_destroy(&material);
  return result;
}

static int validate_plan(MPI_Comm communicator, const OperatorFixture *fixture,
                         int size) {
  uint64_t owned_cells = 0U;
  uint64_t incident_cells = 0U;
  uint64_t halo_cells = 0U;
  CHECK(fixture->mesh_summary.global_vertices == GRID_VERTICES);
  CHECK(fixture->mesh_summary.global_cells == GRID_CELLS);
  CHECK(fixture->mesh_summary.global_intermediate_entities == GRID_EDGES);
  CHECK(fixture->field_summary.order == 2U);
  CHECK(fixture->field_summary.components == 1U);
  CHECK(fixture->field_summary.global_free_dof_count == GRID_P2_ROWS);
  CHECK(fixture->field_summary.overlap_depth == (size == 1 ? 0U : 1U));
  CHECK(fixture->plan_summary.global_rows == GRID_P2_ROWS);
  CHECK(fixture->plan_summary.overlap_depth == (size == 1 ? 0U : 1U));
  CHECK(fixture->plan_summary.local_incident_cells ==
        fixture->field_summary.incident_cell_plan_count);
  CHECK(fixture->plan_summary.local_owned_cells ==
        fixture->field_summary.owned_cell_plan_count);
  CHECK(fixture->plan_summary.local_nonzeros > 0U);
  CHECK(MPI_Allreduce(&fixture->plan_summary.local_owned_cells, &owned_cells, 1,
                      MPI_UINT64_T, MPI_SUM, communicator) == MPI_SUCCESS);
  CHECK(MPI_Allreduce(&fixture->plan_summary.local_incident_cells, &incident_cells, 1,
                      MPI_UINT64_T, MPI_SUM, communicator) == MPI_SUCCESS);
  CHECK(MPI_Allreduce(&fixture->plan_summary.local_halo_cells, &halo_cells, 1,
                      MPI_UINT64_T, MPI_SUM, communicator) == MPI_SUCCESS);
  CHECK(owned_cells == GRID_CELLS);
  CHECK(incident_cells >= owned_cells);
  CHECK(size == 1 ? halo_cells == 0U : halo_cells > 0U);
  return local_status;
}

static double distributed_cell_value(SsEntityId stable_cell_id, size_t slot) {
  return 0.25 + (double)(stable_cell_id & UINT64_C(0xffff)) + 0.125 * (double)slot;
}

static int validate_cell_value_halo(const OperatorFixture *fixture) {
  static const size_t values_per_cell = 3U;
  SsCellPack *owned_packs = NULL;
  SsCellPack *incident_packs = NULL;
  SsEntityId *owned_ids = NULL;
  SsEntityId *incident_ids = NULL;
  double *owned_values = NULL;
  double *incident_values = NULL;
  size_t owned_count = 0U;
  size_t incident_count = 0U;
  size_t cell;
  CHECK(ss_field_space_pack_owned_cells(fixture->space, NULL, 0U, &owned_count) ==
        SS_OK);
  CHECK(ss_field_space_pack_incident_cells(fixture->space, NULL, 0U, &incident_count) ==
        SS_OK);
  owned_packs =
      (SsCellPack *)calloc(owned_count == 0U ? 1U : owned_count, sizeof(*owned_packs));
  incident_packs = (SsCellPack *)calloc(incident_count == 0U ? 1U : incident_count,
                                        sizeof(*incident_packs));
  owned_ids =
      (SsEntityId *)calloc(owned_count == 0U ? 1U : owned_count, sizeof(*owned_ids));
  incident_ids = (SsEntityId *)calloc(incident_count == 0U ? 1U : incident_count,
                                      sizeof(*incident_ids));
  owned_values = (double *)calloc(
      owned_count == 0U ? 1U : owned_count * values_per_cell, sizeof(*owned_values));
  incident_values =
      (double *)calloc(incident_count == 0U ? 1U : incident_count * values_per_cell,
                       sizeof(*incident_values));
  CHECK(owned_packs != NULL);
  CHECK(incident_packs != NULL);
  CHECK(owned_ids != NULL);
  CHECK(incident_ids != NULL);
  CHECK(owned_values != NULL);
  CHECK(incident_values != NULL);
  if (local_status != 0) {
    goto cleanup;
  }
  CHECK(ss_field_space_pack_owned_cells(fixture->space, owned_packs, owned_count,
                                        &owned_count) == SS_OK);
  CHECK(ss_field_space_pack_incident_cells(fixture->space, incident_packs,
                                           incident_count, &incident_count) == SS_OK);
  for (cell = 0U; cell < owned_count; ++cell) {
    size_t slot;
    CHECK(owned_packs[cell].cell_owned != 0U);
    owned_ids[cell] = owned_packs[cell].stable_cell_id;
    for (slot = 0U; slot < values_per_cell; ++slot) {
      owned_values[cell * values_per_cell + slot] =
          distributed_cell_value(owned_ids[cell], slot);
    }
  }
  CHECK(ss_field_space_broadcast_owned_cell_values(
            fixture->space, values_per_cell, owned_ids, owned_values, owned_count,
            incident_ids, incident_values, incident_count) == SS_OK);
  for (cell = 0U; cell < incident_count; ++cell) {
    size_t slot;
    CHECK(incident_ids[cell] == incident_packs[cell].stable_cell_id);
    for (slot = 0U; slot < values_per_cell; ++slot) {
      CHECK(incident_values[cell * values_per_cell + slot] ==
            distributed_cell_value(incident_ids[cell], slot));
    }
  }

cleanup:
  free(incident_values);
  free(owned_values);
  free(incident_ids);
  free(owned_ids);
  free(incident_packs);
  free(owned_packs);
  return local_status;
}

static int assemble_and_validate(MPI_Comm communicator, OperatorFixture *fixture,
                                 int size, int requested_threads,
                                 char invariant[SS_ID_TEXT_CAPACITY],
                                 SsAssemblyDiagnostics *diagnostics,
                                 uint64_t *global_cpu_count,
                                 uint64_t *global_core_count) {
  PetscReal residual_norm = 0.0;
  PetscReal jacobian_norm = 0.0;
  PetscScalar residual_sum = 0.0;
  SsErrorRecord error = {0};
  char projection[256];
  int written;
  long long residual_quantized;
  long long jacobian_quantized;
  long long sum_quantized;
  uint64_t global_natural_evaluations = 0U;
  uint64_t global_natural_duplicates = 0U;
  CHECK(ss_operator_residual_jacobian(fixture->operator_handle, &fixture->state,
                                      fixture->residual, fixture->jacobian,
                                      &error) == SS_OK);
  CHECK(error.valid == 0);
  CHECK(ss_operator_diagnostics(fixture->operator_handle, diagnostics) == SS_OK);
  CHECK(diagnostics->requested_threads == requested_threads);
  CHECK(diagnostics->observed_threads == requested_threads);
  CHECK(diagnostics->used_openmp == (requested_threads > 1 ? 1 : 0));
  CHECK(diagnostics->completed_rows == fixture->plan_summary.local_owned_rows);
  CHECK(diagnostics->row_writer_violations == 0U);
  CHECK(diagnostics->worker_service_calls == 0U);
  CHECK(diagnostics->petsc_new_nonzero_allocations == 0U);
  CHECK(diagnostics->unique_worker_cpus == (uint64_t)requested_threads);
  CHECK(diagnostics->unique_physical_cores == (uint64_t)requested_threads);
  CHECK(MPI_Allreduce(&diagnostics->natural_facet_evaluations,
                      &global_natural_evaluations, 1, MPI_UINT64_T, MPI_SUM,
                      communicator) == MPI_SUCCESS);
  CHECK(MPI_Allreduce(&diagnostics->duplicate_natural_facet_evaluations,
                      &global_natural_duplicates, 1, MPI_UINT64_T, MPI_SUM,
                      communicator) == MPI_SUCCESS);
  CHECK(global_natural_evaluations >= GRID_BOUNDARY_FACETS);
  CHECK(size == 1 ? global_natural_duplicates == 0U : global_natural_duplicates > 0U);
  CHECK(validate_global_affinity(communicator, diagnostics, size, requested_threads,
                                 global_cpu_count, global_core_count) == 0);
  CHECK(VecNorm(fixture->residual, NORM_2, &residual_norm) == PETSC_SUCCESS);
  CHECK(MatNorm(fixture->jacobian, NORM_FROBENIUS, &jacobian_norm) == PETSC_SUCCESS);
  CHECK(VecSum(fixture->residual, &residual_sum) == PETSC_SUCCESS);
  CHECK(isfinite((double)residual_norm) && residual_norm > 0.0);
  CHECK(isfinite((double)jacobian_norm) && jacobian_norm > 0.0);
  CHECK(isfinite((double)PetscRealPart(residual_sum)));
  residual_quantized = llround((double)residual_norm * 1.0e10);
  jacobian_quantized = llround((double)jacobian_norm * 1.0e10);
  sum_quantized = llround((double)PetscRealPart(residual_sum) * 1.0e10);
  written = snprintf(projection, sizeof(projection),
                     "rows=%u|residual=%lld|jacobian=%lld|sum=%lld",
                     (unsigned int)GRID_P2_ROWS, residual_quantized, jacobian_quantized,
                     sum_quantized);
  CHECK(written > 0 && (size_t)written < sizeof(projection));
  CHECK(ss_identity_from_text("assembly-layout", projection, invariant) == SS_OK);
  return local_status;
}

static int write_result(const char *path, const char *test_id, int ranks,
                        int requested_threads,
                        const char invariant[SS_ID_TEXT_CAPACITY],
                        const SsAssemblyDiagnostics *diagnostics,
                        const SsAssemblyPlanSummary *plan_summary,
                        uint64_t global_cpu_count, uint64_t global_core_count) {
  FILE *stream;
  int result;
  if (path == NULL || test_id == NULL || invariant == NULL || diagnostics == NULL ||
      plan_summary == NULL) {
    return 1;
  }
  stream = fopen(path, "wb");
  if (stream == NULL) {
    return 1;
  }
  result =
      fprintf(stream,
              "{\"test_id\":\"%s\",\"ranks\":%d,\"requested_threads\":%d,"
              "\"observed_threads\":%d,\"invariant\":\"%s\","
              "\"worker_cpus\":\"%s\",\"global_unique_worker_cpus\":%llu,"
              "\"global_unique_physical_cores\":%llu,"
              "\"diagnostics\":{\"completed_rows\":%llu,\"incident_cells\":%llu,"
              "\"owned_cells\":%llu,\"halo_cells\":%llu,\"local_nonzeros\":%llu,"
              "\"row_writer_violations\":%llu,\"worker_service_calls\":%llu,"
              "\"petsc_new_nonzero_allocations\":%llu,\"tile_count\":%llu}}\n",
              test_id, ranks, requested_threads, diagnostics->observed_threads,
              invariant, diagnostics->worker_cpus, (unsigned long long)global_cpu_count,
              (unsigned long long)global_core_count,
              (unsigned long long)diagnostics->completed_rows,
              (unsigned long long)plan_summary->local_incident_cells,
              (unsigned long long)plan_summary->local_owned_cells,
              (unsigned long long)plan_summary->local_halo_cells,
              (unsigned long long)plan_summary->local_nonzeros,
              (unsigned long long)diagnostics->row_writer_violations,
              (unsigned long long)diagnostics->worker_service_calls,
              (unsigned long long)diagnostics->petsc_new_nonzero_allocations,
              (unsigned long long)diagnostics->tile_count) < 0
          ? 1
          : 0;
  if (fclose(stream) != 0) {
    result = 1;
  }
  return result;
}

int main(int argc, char **argv) {
  const char *test_id = NULL;
  const char *output_path;
  int expected_ranks = 0;
  int expected_threads = 0;
  int ranks = 0;
  int provided = MPI_THREAD_SINGLE;
  int global_status = 0;
  int argument;
  int known_test;
  OperatorFixture fixture;
  SsAssemblyDiagnostics diagnostics = {0};
  char invariant[SS_ID_TEXT_CAPACITY] = {0};
  uint64_t global_cpu_count = 0U;
  uint64_t global_core_count = 0U;
  memset(&fixture, 0, sizeof(fixture));
  for (argument = 1; argument + 1 < argc; argument += 2) {
    if (strcmp(argv[argument], "--test-id") == 0) {
      test_id = argv[argument + 1];
    } else if (strcmp(argv[argument], "--expected-ranks") == 0) {
      expected_ranks = atoi(argv[argument + 1]);
    } else if (strcmp(argv[argument], "--expected-threads") == 0) {
      expected_threads = atoi(argv[argument + 1]);
    }
  }
  output_path = getenv("SLOPE_ASSEMBLY_RESULT");
  if (test_id == NULL || output_path == NULL || expected_ranks < 1 ||
      expected_threads < 1) {
    fputs("usage: ss_test_assembly_mpi --test-id ID --expected-ranks N "
          "--expected-threads N; SLOPE_ASSEMBLY_RESULT must be set\n",
          stderr);
    return 2;
  }
  if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
    return 3;
  }
  if (PetscInitialize(&argc, &argv, NULL, NULL) != PETSC_SUCCESS) {
    MPI_Abort(MPI_COMM_WORLD, 4);
    return 4;
  }
  if (MPI_Comm_rank(PETSC_COMM_WORLD, &test_rank) != MPI_SUCCESS ||
      MPI_Comm_size(PETSC_COMM_WORLD, &ranks) != MPI_SUCCESS) {
    MPI_Abort(PETSC_COMM_WORLD, 5);
    return 5;
  }
  known_test = strcmp(test_id, "ASM-UNIT-OVERLAP-001") == 0 ||
               strcmp(test_id, "ASM-CONTRACT-MATFREE-MPI-001") == 0 ||
               strcmp(test_id, "ASM-CONTRACT-ROWOWNER-001") == 0 ||
               strcmp(test_id, "ASM-CONTRACT-PETSCFUNNEL-001") == 0 ||
               strcmp(test_id, "ASM-CONTRACT-TRANSACTION-001") == 0 ||
               strcmp(test_id, "ASM-CONTRACT-PERFORMANCE-001") == 0 ||
               strcmp(test_id, "ASM-CONTRACT-COO-COLLECTIVE-001") == 0 ||
               strcmp(test_id, "ASM-CONTRACT-LAYOUT-001") == 0 ||
               strcmp(test_id, "ASM-RACE-MUST-001") == 0;
  CHECK(known_test != 0);
  CHECK(ranks == expected_ranks);
  CHECK(provided >= MPI_THREAD_FUNNELED);
  if (local_status == 0 && operator_fixture_create(PETSC_COMM_WORLD, test_rank, ranks,
                                                   expected_threads, &fixture) != 0) {
    local_status = 1;
  }
  if (local_status == 0) {
    CHECK(validate_plan(PETSC_COMM_WORLD, &fixture, ranks) == 0);
  }
  if (local_status == 0) {
    CHECK(validate_cell_value_halo(&fixture) == 0);
  }
  if (local_status == 0 && strcmp(test_id, "ASM-CONTRACT-MATFREE-MPI-001") == 0) {
    CHECK(test_matrix_free_p1_projection(PETSC_COMM_WORLD, fixture.mesh,
                                         expected_threads) == 0);
  }
  if (local_status == 0 && (strcmp(test_id, "ASM-CONTRACT-PERFORMANCE-001") == 0 ||
                            strcmp(test_id, "ASM-CONTRACT-COO-COLLECTIVE-001") == 0)) {
    CHECK(test_performance_mode(PETSC_COMM_WORLD, &fixture, ranks,
                                strcmp(test_id, "ASM-CONTRACT-COO-COLLECTIVE-001") ==
                                    0) == 0);
  }
  if (local_status == 0 && strcmp(test_id, "ASM-CONTRACT-TRANSACTION-001") == 0) {
    CHECK(test_transaction(&fixture) == 0);
  }
  if (local_status == 0) {
    memset(&fixture.observation, 0, sizeof(fixture.observation));
    CHECK(assemble_and_validate(PETSC_COMM_WORLD, &fixture, ranks, expected_threads,
                                invariant, &diagnostics, &global_cpu_count,
                                &global_core_count) == 0);
    CHECK(fixture.observation.mpi_calls > 0U);
    CHECK(fixture.observation.petsc_calls > 0U);
    CHECK(fixture.observation.hdf5_calls == 0U);
    CHECK(fixture.observation.parallel_calls == 0U);
  }
  if (local_status == 0 && strcmp(test_id, "ASM-CONTRACT-LAYOUT-001") == 0) {
    CHECK(test_compatible_source_reuse(&fixture, test_rank) == 0);
  }
  if (MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX,
                    PETSC_COMM_WORLD) != MPI_SUCCESS) {
    global_status = 1;
  }
  if (test_rank == 0 && global_status == 0 &&
      write_result(output_path, test_id, ranks, expected_threads, invariant,
                   &diagnostics, &fixture.plan_summary, global_cpu_count,
                   global_core_count) != 0) {
    global_status = 1;
  }
  if (MPI_Bcast(&global_status, 1, MPI_INT, 0, PETSC_COMM_WORLD) != MPI_SUCCESS) {
    global_status = 1;
  }
  if (operator_fixture_destroy(&fixture) != 0) {
    global_status = 1;
  }
  if (PetscFinalize() != PETSC_SUCCESS) {
    global_status = 1;
  }
  if (MPI_Finalize() != MPI_SUCCESS) {
    global_status = 1;
  }
  return global_status;
}
