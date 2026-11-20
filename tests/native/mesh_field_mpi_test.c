/* Distributed mesh and finite-element field contracts across MPI rank counts
 * and shared-point balancing policies; mesh_field_mpi_fixtures.c builds the
 * distributed meshes.
 *
 * Launched by tests/native/support/mpi_layout_runner.py --suite mesh.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "mesh_field_mpi_internal.h"

static int append_digest(char *buffer, size_t capacity, size_t *length,
                         const SsMeshSummary *summary,
                         const SsFieldSummary field_summaries[3]) {
  int written;
  size_t field;
  written = snprintf(
      buffer + *length, capacity - *length, "%u:%u:%s:%s:%s:%s:%s:", summary->dimension,
      summary->coordinate_order, summary->stable_id_digest, summary->topology_digest,
      summary->label_digest, summary->coordinate_digest, summary->closure_digest);
  if (written < 0 || (size_t)written >= capacity - *length) {
    return 1;
  }
  *length += (size_t)written;
  for (field = 0U; field < 3U; ++field) {
    written = snprintf(buffer + *length, capacity - *length,
                       "%s:", field_summaries[field].identity_digest);
    if (written < 0 || (size_t)written >= capacity - *length) {
      return 1;
    }
    *length += (size_t)written;
  }
  return 0;
}

static int test_shared_distribution(MPI_Comm communicator, int rank, int size,
                                    char *digest_input, size_t digest_capacity,
                                    size_t *digest_length) {
  const SsStrongConstraint constraints[] = {
      {UINT64_C(0x110000000000), 0U, 0U, 0.0}, {UINT64_C(0x110000000002), 0U, 0U, 0.0},
      {UINT64_C(0x130000000003), 0U, 0U, 0.0}, {UINT64_C(0x130000000003), 1U, 0U, 0.0},
      {UINT64_C(0x130000000003), 2U, 0U, 0.0},
  };
  const SsFieldSpec spec = {
      4U,
      1U,
      sizeof(constraints) / sizeof(constraints[0]),
      constraints,
  };
  StripFixture fixture;
  SsMesh *mesh = NULL;
  SsMeshError error = {0};
  SsMeshSummary mesh_summary = {0};
  SsFieldSpace *space = NULL;
  SsFieldSummary field_summary = {0};
  SsCellPack packs[2] = {{0}};
  size_t pack_count = 0U;
  uint64_t local_pack_count = 0U;
  uint64_t global_pack_count = 0U;
  uint64_t local_material_checksum = 0U;
  uint64_t global_material_checksum = 0U;
  uint64_t global_label_count = 0U;
  uint64_t local_facet_count = 0U;
  uint64_t global_facet_count = 0U;
  int local_status = 0;
  int written;
  size_t pack;
  SsStatus create_status;
  LOCAL_CHECK(build_strip_fixture(rank, size, &fixture) == 0);
  create_status = ss_mesh_create(communicator, &fixture.input, &mesh, &error);
  if (create_status != SS_OK) {
    fprintf(stderr, "rank %d: shared mesh creation failed: %s\n", rank,
            ss_status_name(create_status));
  }
  LOCAL_CHECK(create_status == SS_OK);
  LOCAL_CHECK(mesh != NULL && error.valid == 0);
  if (mesh == NULL) {
    return 1;
  }
  LOCAL_CHECK(ss_mesh_summary(mesh, &mesh_summary) == SS_OK);
  LOCAL_CHECK(mesh_summary.global_vertices == 4U);
  LOCAL_CHECK(mesh_summary.global_edges == 5U);
  LOCAL_CHECK(mesh_summary.global_facets == 4U);
  LOCAL_CHECK(mesh_summary.global_intermediate_entities == 5U);
  LOCAL_CHECK(mesh_summary.global_cells == 2U);
  LOCAL_CHECK(mesh_summary.global_coordinate_nodes == 4U);
  LOCAL_CHECK(ss_mesh_label_cardinality_global(mesh, SS_MESH_LABEL_MATERIAL, 101U,
                                               &global_label_count) == SS_OK);
  LOCAL_CHECK(global_label_count == 1U);
  {
    PetscInt point = -1;
    const SsStatus find_status = ss_mesh_point_from_stable_id(
        mesh, SS_MESH_ENTITY_EDGE, strip_intermediate_id(1U, 2U), &point);
    int local_found = find_status == SS_OK ? 1 : 0;
    int global_found = 0;
    LOCAL_CHECK(find_status == SS_OK || find_status == SS_ERROR_NOT_FOUND);
    LOCAL_CHECK(MPI_Allreduce(&local_found, &global_found, 1, MPI_INT, MPI_MAX,
                              communicator) == MPI_SUCCESS);
    LOCAL_CHECK(global_found == 1);
  }
  LOCAL_CHECK(ss_mesh_label_cardinality_global(mesh, SS_MESH_LABEL_SUPPORT, 201U,
                                               &global_label_count) == SS_OK);
  LOCAL_CHECK(global_label_count == 1U);
  LOCAL_CHECK(ss_mesh_field_space_create(mesh, &spec, &space) == SS_OK);
  if (space != NULL) {
    size_t facet_count = 0U;
    LOCAL_CHECK(ss_field_space_summary(space, &field_summary) == SS_OK);
    LOCAL_CHECK(field_summary.global_dof_count == 25U);
    LOCAL_CHECK(field_summary.global_free_dof_count == 20U);
    LOCAL_CHECK(field_summary.constrained_dof_count == 5U);
    LOCAL_CHECK(ss_field_space_pack_owned_facets(space, 201U, NULL, 0U, &facet_count) ==
                SS_OK);
    local_facet_count = (uint64_t)facet_count;
    LOCAL_CHECK(ss_field_space_pack_owned_cells(space, NULL, 0U, &pack_count) == SS_OK);
    LOCAL_CHECK(pack_count == (size_t)mesh_summary.owned_cells);
    LOCAL_CHECK(ss_field_space_pack_owned_cells(space, packs, 2U, &pack_count) ==
                SS_OK);
    for (pack = 0U; pack < pack_count; ++pack) {
      LOCAL_CHECK(packs[pack].field_node_count == 15U);
      LOCAL_CHECK(packs[pack].closure_count == 15U);
      LOCAL_CHECK(packs[pack].material_value == 101U ||
                  packs[pack].material_value == 102U);
      local_material_checksum +=
          packs[pack].stable_cell_id * UINT64_C(257) + packs[pack].material_value;
    }
  }
  local_pack_count = (uint64_t)pack_count;
  LOCAL_CHECK(MPI_Allreduce(&local_pack_count, &global_pack_count, 1, MPI_UINT64_T,
                            MPI_SUM, communicator) == MPI_SUCCESS);
  LOCAL_CHECK(MPI_Allreduce(&local_material_checksum, &global_material_checksum, 1,
                            MPI_UINT64_T, MPI_SUM, communicator) == MPI_SUCCESS);
  LOCAL_CHECK(MPI_Allreduce(&local_facet_count, &global_facet_count, 1, MPI_UINT64_T,
                            MPI_SUM, communicator) == MPI_SUCCESS);
  LOCAL_CHECK(global_pack_count == 2U);
  LOCAL_CHECK(global_facet_count == 1U);
  LOCAL_CHECK(global_material_checksum ==
              strip_cell_id(0U) * UINT64_C(257) + UINT64_C(101) +
                  strip_cell_id(1U) * UINT64_C(257) + UINT64_C(102));
  written = snprintf(digest_input + *digest_length, digest_capacity - *digest_length,
                     "shared:%s:%s:%llu:", mesh_summary.closure_digest,
                     field_summary.identity_digest,
                     (unsigned long long)global_material_checksum);
  LOCAL_CHECK(written > 0 && (size_t)written < digest_capacity - *digest_length);
  if (written > 0 && (size_t)written < digest_capacity - *digest_length) {
    *digest_length += (size_t)written;
  }
  LOCAL_CHECK(ss_field_space_destroy(&space) == SS_OK);
  LOCAL_CHECK(ss_mesh_destroy(&mesh) == SS_OK);
  return local_status;
}

static int check_owned_free_dof_map(MPI_Comm communicator, int rank, int size,
                                    const SsFieldSpace *space,
                                    const SsFieldSummary *summary) {
  SsEntityId *stable_ids = NULL;
  size_t *local_slots = NULL;
  size_t count = 0U;
  size_t mapped_count = 0U;
  uint64_t local_count = 0U;
  uint64_t global_count = 0U;
  int local_empty = 0;
  int global_empty = 0;
  int local_status = 0;
  size_t index;

  LOCAL_CHECK(space != NULL && summary != NULL);
  LOCAL_CHECK(ss_field_space_owned_free_dof_map(space, NULL, NULL, 0U, &count) ==
              SS_OK);
  LOCAL_CHECK(count == (size_t)summary->owned_free_dof_count);
  if (count != 0U) {
    stable_ids = (SsEntityId *)calloc(count, sizeof(*stable_ids));
    local_slots = (size_t *)calloc(count, sizeof(*local_slots));
    LOCAL_CHECK(stable_ids != NULL && local_slots != NULL);
  }
  LOCAL_CHECK(ss_field_space_owned_free_dof_map(space, stable_ids, local_slots, count,
                                                &mapped_count) == SS_OK);
  LOCAL_CHECK(mapped_count == count);
  for (index = 0U; index < mapped_count; ++index) {
    LOCAL_CHECK(stable_ids[index] != 0U);
    LOCAL_CHECK(local_slots[index] < mapped_count);
  }
  local_count = (uint64_t)mapped_count;
  local_empty = mapped_count == 0U ? 1 : 0;
  LOCAL_CHECK(MPI_Allreduce(&local_count, &global_count, 1, MPI_UINT64_T, MPI_SUM,
                            communicator) == MPI_SUCCESS);
  LOCAL_CHECK(MPI_Allreduce(&local_empty, &global_empty, 1, MPI_INT, MPI_MAX,
                            communicator) == MPI_SUCCESS);
  LOCAL_CHECK(global_count == summary->global_free_dof_count);
  if ((uint64_t)size > summary->global_free_dof_count) {
    LOCAL_CHECK(global_empty != 0);
  }
  free(local_slots);
  free(stable_ids);
  return local_status;
}

static int test_distribution(MPI_Comm communicator, int rank, int size,
                             char invariant[SS_SHA256_HEX_CAPACITY]) {
  const uint32_t dimensions[] = {2U, 3U};
  const uint32_t orders[] = {1U, 2U, 4U};
  const uint32_t node_counts_2d[] = {3U, 6U, 15U};
  const uint32_t node_counts_3d[] = {4U, 10U, 35U};
  char digest_input[8192] = {0};
  size_t digest_length = 0U;
  int local_status = 0;
  size_t dimension_index;
  size_t geometry_index;

  for (dimension_index = 0U; dimension_index < 2U; ++dimension_index) {
    const uint32_t dimension = dimensions[dimension_index];
    for (geometry_index = 0U; geometry_index < 3U; ++geometry_index) {
      CompactFixture fixture;
      SsMesh *mesh = NULL;
      SsMeshError error = {0};
      SsMeshSummary summary = {0};
      SsFieldSummary field_summaries[3] = {{0}};
      const SsFeDescriptor *coordinate_descriptor = NULL;
      uint64_t local_input_cells;
      uint64_t local_input_coordinates;
      uint64_t owned_sum = 0U;
      uint64_t owned_vertex_sum = 0U;
      uint64_t owned_intermediate_sum = 0U;
      uint64_t input_cell_sum = 0U;
      uint64_t input_coordinate_sum = 0U;
      uint64_t expected_audit_rules = 0U;
      uint64_t expected_audit_points = 0U;
      int owned = 0;
      size_t field_index;

      LOCAL_CHECK(
          build_fixture(dimension, orders[geometry_index], rank, size, &fixture) == 0);
      LOCAL_CHECK(ss_fe_descriptor_find(dimension, orders[geometry_index],
                                        &coordinate_descriptor) == SS_OK);
      if (local_status != 0) {
        continue;
      }
      local_input_cells = (uint64_t)fixture.input.cell_count;
      local_input_coordinates = (uint64_t)fixture.input.coordinate_node_count;
      LOCAL_CHECK(MPI_Allreduce(&local_input_cells, &input_cell_sum, 1, MPI_UINT64_T,
                                MPI_SUM, communicator) == MPI_SUCCESS);
      LOCAL_CHECK(MPI_Allreduce(&local_input_coordinates, &input_coordinate_sum, 1,
                                MPI_UINT64_T, MPI_SUM, communicator) == MPI_SUCCESS);
      LOCAL_CHECK(input_cell_sum == fixture.input.global_cell_count);
      LOCAL_CHECK(input_coordinate_sum == fixture.input.global_coordinate_node_count);
      if (size > 1) {
        LOCAL_CHECK(fixture.input.coordinate_node_count <
                    fixture.input.global_coordinate_node_count);
      }
      LOCAL_CHECK(ss_mesh_create(communicator, &fixture.input, &mesh, &error) == SS_OK);
      LOCAL_CHECK(mesh != NULL && error.valid == 0);
      if (mesh == NULL) {
        continue;
      }
      LOCAL_CHECK(ss_mesh_summary(mesh, &summary) == SS_OK);
      LOCAL_CHECK(summary.dimension == dimension);
      LOCAL_CHECK(summary.coordinate_order == orders[geometry_index]);
      LOCAL_CHECK(summary.global_vertices == (uint64_t)dimension + 1U);
      LOCAL_CHECK(summary.global_edges == (dimension == 2U ? 3U : 6U));
      LOCAL_CHECK(summary.global_facets == (dimension == 2U ? 3U : 4U));
      LOCAL_CHECK(summary.global_intermediate_entities == (dimension == 2U ? 3U : 10U));
      LOCAL_CHECK(summary.global_cells == 1U);
      LOCAL_CHECK(summary.global_coordinate_nodes ==
                  (uint64_t)(dimension == 2U ? node_counts_2d[geometry_index]
                                             : node_counts_3d[geometry_index]));
      LOCAL_CHECK(summary.minimum_determinant > 0.0);
      LOCAL_CHECK(summary.maximum_determinant >= summary.minimum_determinant);
      LOCAL_CHECK(summary.minimum_condition > 0.0);
      LOCAL_CHECK(summary.maximum_condition >= summary.minimum_condition);
      LOCAL_CHECK(summary.partition_imbalance >= 1.0);
      LOCAL_CHECK(summary.topology_generation == 1U);
      LOCAL_CHECK(summary.coordinate_generation == 1U);
      LOCAL_CHECK(summary.label_generation == 1U);
      LOCAL_CHECK(summary.partition_generation == 1U);
      LOCAL_CHECK(summary.cache_memory_bytes > 0U ||
                  summary.owned_vertices + summary.ghost_vertices +
                          summary.owned_intermediate_entities +
                          summary.ghost_intermediate_entities + summary.owned_cells +
                          summary.ghost_cells ==
                      0U);
      LOCAL_CHECK(summary.owned_facet_plan_count >= summary.owned_facets);
      LOCAL_CHECK(summary.facet_plan_bytes >=
                  summary.owned_facet_plan_count * (uint64_t)sizeof(SsFacetPack));
      LOCAL_CHECK(summary.coordinate_basis_id[0] != '\0');
      LOCAL_CHECK(summary.geometry_quadrature_id[0] != '\0');
      {
        size_t registry_index;
        for (registry_index = 0U; registry_index < ss_fe_quadrature_registry_count();
             ++registry_index) {
          const SsFeQuadrature *registered = NULL;
          LOCAL_CHECK(ss_fe_quadrature_registry_at(registry_index, &registered) ==
                      SS_OK);
          if (registered != NULL && registered->domain == SS_FE_QUADRATURE_VOLUME &&
              registered->parent_dimension == dimension &&
              registered->has_negative_weight == 0) {
            ++expected_audit_rules;
            expected_audit_points += (uint64_t)registered->point_count;
          }
        }
      }
      LOCAL_CHECK(summary.geometry_audit_rule_count == expected_audit_rules);
      LOCAL_CHECK(summary.geometry_audit_point_count == expected_audit_points);
      LOCAL_CHECK(strlen(summary.geometry_audit_rule_digest) == 64U);
      LOCAL_CHECK(strlen(summary.topology_digest) == 64U);
      LOCAL_CHECK(MPI_Allreduce(&summary.owned_cells, &owned_sum, 1, MPI_UINT64_T,
                                MPI_SUM, communicator) == MPI_SUCCESS);
      LOCAL_CHECK(MPI_Allreduce(&summary.owned_vertices, &owned_vertex_sum, 1,
                                MPI_UINT64_T, MPI_SUM, communicator) == MPI_SUCCESS);
      LOCAL_CHECK(MPI_Allreduce(&summary.owned_intermediate_entities,
                                &owned_intermediate_sum, 1, MPI_UINT64_T, MPI_SUM,
                                communicator) == MPI_SUCCESS);
      LOCAL_CHECK(owned_sum == 1U);
      LOCAL_CHECK(owned_vertex_sum == summary.global_vertices);
      LOCAL_CHECK(owned_intermediate_sum == summary.global_intermediate_entities);
      LOCAL_CHECK(ss_mesh_cell_is_owned(mesh, fixture.cell_ids[0], &owned) == SS_OK);
      if (owned != 0) {
        SsEntityId ids[SS_FE_MAX_NODES] = {0};
        double coordinates[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
        size_t closure_count = 0U;
        size_t value;
        LOCAL_CHECK(ss_mesh_cell_coordinate_closure(mesh, fixture.cell_ids[0], ids,
                                                    coordinates, SS_FE_MAX_NODES,
                                                    &closure_count) == SS_OK);
        LOCAL_CHECK(closure_count == fixture.input.global_coordinate_node_count);
        for (value = 0U; value < closure_count; ++value) {
          double expected[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
          size_t coordinate;
          fixture_coordinate(coordinate_descriptor, value, expected);
          LOCAL_CHECK(ids[value] ==
                      fixture_coordinate_id(dimension, orders[geometry_index], value));
          for (coordinate = 0U; coordinate < (size_t)dimension; ++coordinate) {
            LOCAL_CHECK(fabs(coordinates[value * (size_t)dimension + coordinate] -
                             expected[coordinate]) < 2.0e-13);
          }
        }
      }
      for (field_index = 0U; field_index < 3U; ++field_index) {
        const SsFieldSpec spec = {
            orders[field_index],
            dimension,
            0U,
            NULL,
        };
        SsFieldSpace *space = NULL;
        LOCAL_CHECK(ss_mesh_field_space_create(mesh, &spec, &space) == SS_OK);
        if (space != NULL) {
          LOCAL_CHECK(ss_field_space_summary(space, &field_summaries[field_index]) ==
                      SS_OK);
          LOCAL_CHECK(field_summaries[field_index].section_generation == 1U);
          LOCAL_CHECK(field_summaries[field_index].constraint_generation == 1U);
          LOCAL_CHECK(field_summaries[field_index].basis_id[0] != '\0');
          LOCAL_CHECK(field_summaries[field_index].owned_cell_plan_count ==
                      summary.owned_cells);
          LOCAL_CHECK(field_summaries[field_index].closure_plan_bytes ==
                      field_summaries[field_index].owned_cell_plan_count *
                          (uint64_t)sizeof(SsCellPlan));
          LOCAL_CHECK(field_summaries[field_index].incident_closure_plan_bytes >=
                      field_summaries[field_index].incident_cell_plan_count *
                          (uint64_t)sizeof(SsCellPlan));
          LOCAL_CHECK(field_summaries[field_index].incident_cell_plan_count == 0U
                          ? field_summaries[field_index].incident_closure_plan_bytes ==
                                0U
                          : field_summaries[field_index].incident_closure_plan_bytes <
                                field_summaries[field_index].incident_cell_plan_count *
                                    (uint64_t)sizeof(SsCellPack));
          LOCAL_CHECK(field_summaries[field_index].global_dof_count ==
                      (uint64_t)(dimension == 2U ? node_counts_2d[field_index]
                                                 : node_counts_3d[field_index]) *
                          (uint64_t)dimension);
          LOCAL_CHECK(ss_field_space_destroy(&space) == SS_OK);
        }
      }
      LOCAL_CHECK(append_digest(digest_input, sizeof(digest_input), &digest_length,
                                &summary, field_summaries) == 0);
      LOCAL_CHECK(ss_mesh_destroy(&mesh) == SS_OK && mesh == NULL);
    }
  }
  {
    CompactFixture fixture;
    SsMesh *mesh = NULL;
    SsMeshError error = {0};
    SsFieldSpace *space = NULL;
    SsFieldSummary summary = {0};
    const SsFieldSpec spec = {
        1U,
        1U,
        0U,
        NULL,
    };
    LOCAL_CHECK(build_fixture(2U, 1U, rank, size, &fixture) == 0);
    LOCAL_CHECK(ss_mesh_create(communicator, &fixture.input, &mesh, &error) == SS_OK);
    LOCAL_CHECK(mesh != NULL && error.valid == 0);
    if (mesh != NULL) {
      LOCAL_CHECK(ss_mesh_field_space_create(mesh, &spec, &space) == SS_OK);
    }
    if (space != NULL) {
      LOCAL_CHECK(ss_field_space_summary(space, &summary) == SS_OK);
      LOCAL_CHECK(summary.global_free_dof_count == 3U);
      LOCAL_CHECK(check_owned_free_dof_map(communicator, rank, size, space, &summary) ==
                  0);
    }
    LOCAL_CHECK(ss_field_space_destroy(&space) == SS_OK);
    LOCAL_CHECK(ss_mesh_destroy(&mesh) == SS_OK);
  }
  {
    CompactFixture invalid;
    SsMesh *mesh = NULL;
    SsMeshError error = {0};
    SsStatus status;
    LOCAL_CHECK(build_fixture(2U, 1U, rank, size, &invalid) == 0);
    if (rank == size - 1) {
      invalid.input.relative_determinant_tolerance = -1.0;
    }
    status = ss_mesh_create(communicator, &invalid.input, &mesh, &error);
    LOCAL_CHECK(status == SS_ERROR_INVALID_ARGUMENT);
    LOCAL_CHECK(mesh == NULL && error.valid == 0);
  }
  if (size >= 2) {
    StripFixture inconsistent;
    SsMesh *mesh = NULL;
    SsMeshError error = {0};
    SsStatus status;
    size_t entity;
    int changed = rank == 1 ? 0 : 1;
    LOCAL_CHECK(build_strip_fixture(rank, size, &inconsistent) == 0);
    if (rank == 1) {
      for (entity = 0U; entity < inconsistent.input.intermediate_entity_count;
           ++entity) {
        if (inconsistent.intermediate_entities[entity].vertex_ids[0] ==
                strip_vertex_id(1U) &&
            inconsistent.intermediate_entities[entity].vertex_ids[1] ==
                strip_vertex_id(2U)) {
          inconsistent.intermediate_entities[entity].stable_entity_id ^=
              UINT64_C(0x0102030405060708);
          changed = 1;
          break;
        }
      }
    }
    LOCAL_CHECK(changed != 0);
    status = ss_mesh_create(communicator, &inconsistent.input, &mesh, &error);
    LOCAL_CHECK(status == SS_ERROR_DUPLICATE);
    LOCAL_CHECK(mesh == NULL && error.valid == 0);
  }
  LOCAL_CHECK(test_shared_distribution(communicator, rank, size, digest_input,
                                       sizeof(digest_input), &digest_length) == 0);
  LOCAL_CHECK(ss_sha256_bytes((const unsigned char *)digest_input, digest_length,
                              invariant) == SS_OK);
  return local_status;
}

static int configure_bad_fixture(uint32_t variant, int rank, int size,
                                 CompactFixture *fixture, const char **expected_code) {
  if (variant == 0U) {
    if (build_fixture(2U, 1U, rank, size, fixture) != 0) {
      return 1;
    }
    fixture->cells[1] = 2U;
    fixture->cells[2] = 1U;
    fixture->cell_coordinate_nodes[1] = 2U;
    fixture->cell_coordinate_nodes[2] = 1U;
    *expected_code = "MESH_GEOMETRY_INVERTED";
  } else if (variant == 1U) {
    if (build_fixture(2U, 2U, rank, size, fixture) != 0) {
      return 1;
    }
    {
      const SsFeDescriptor *descriptor = NULL;
      size_t node;
      if (ss_fe_descriptor_find(2U, 2U, &descriptor) != SS_OK) {
        return 1;
      }
      for (node = 0U; node < fixture->input.coordinate_node_count; ++node) {
        const size_t global_node = fixture->owned_coordinate_start + node;
        const double r = descriptor_node_coordinate(descriptor, global_node, 0U);
        const double s = descriptor_node_coordinate(descriptor, global_node, 1U);
        fixture->coordinate_coordinates[node * 2U] = r;
        fixture->coordinate_coordinates[node * 2U + 1U] = s * (1.0 - 2.0 * r);
      }
    }
    *expected_code = "MESH_GEOMETRY_SIGN_CHANGE";
  } else if (variant == 2U) {
    if (build_fixture(3U, 1U, rank, size, fixture) != 0) {
      return 1;
    }
    if (fixture->owned_vertex_start <= 3U &&
        3U < fixture->owned_vertex_start + fixture->input.vertex_count) {
      const size_t local_vertex = 3U - fixture->owned_vertex_start;
      fixture->vertex_coordinates[local_vertex * 3U + 2U] = 1.0e-16;
    }
    if (fixture->owned_coordinate_start <= 3U &&
        3U < fixture->owned_coordinate_start + fixture->input.coordinate_node_count) {
      const size_t local_node = 3U - fixture->owned_coordinate_start;
      fixture->coordinate_coordinates[local_node * 3U + 2U] = 1.0e-16;
    }
    *expected_code = "MESH_GEOMETRY_NEAR_SINGULAR";
  } else if (variant == 3U) {
    if (build_fixture(3U, 1U, rank, size, fixture) != 0) {
      return 1;
    }
    fixture->cells[1] = 2U;
    fixture->cells[2] = 1U;
    fixture->cell_coordinate_nodes[1] = 2U;
    fixture->cell_coordinate_nodes[2] = 1U;
    *expected_code = "MESH_GEOMETRY_INVERTED";
  } else if (variant == 4U) {
    const SsFeDescriptor *descriptor = NULL;
    size_t node;
    if (build_fixture(3U, 2U, rank, size, fixture) != 0 ||
        ss_fe_descriptor_find(3U, 2U, &descriptor) != SS_OK) {
      return 1;
    }
    for (node = 0U; node < fixture->input.coordinate_node_count; ++node) {
      const size_t global_node = fixture->owned_coordinate_start + node;
      const double r = descriptor_node_coordinate(descriptor, global_node, 0U);
      const double s = descriptor_node_coordinate(descriptor, global_node, 1U);
      const double t = descriptor_node_coordinate(descriptor, global_node, 2U);
      fixture->coordinate_coordinates[node * 3U] = r;
      fixture->coordinate_coordinates[node * 3U + 1U] = s;
      fixture->coordinate_coordinates[node * 3U + 2U] = t * (1.0 - 3.0 * r);
    }
    *expected_code = "MESH_GEOMETRY_SIGN_CHANGE";
  } else {
    const uint32_t dimension = variant == 5U ? 2U : 3U;
    const double fold_root = 0.92;
    const double fold_scale = 1.0 / (fold_root * fold_root * fold_root);
    const SsFeDescriptor *descriptor = NULL;
    size_t node;
    if (build_fixture(dimension, 4U, rank, size, fixture) != 0 ||
        ss_fe_descriptor_find(dimension, 4U, &descriptor) != SS_OK) {
      return 1;
    }
    for (node = 0U; node < fixture->input.coordinate_node_count; ++node) {
      const size_t global_node = fixture->owned_coordinate_start + node;
      const double r = descriptor_node_coordinate(descriptor, global_node, 0U);
      const double s = descriptor_node_coordinate(descriptor, global_node, 1U);
      fixture->coordinate_coordinates[node * dimension] = r;
      fixture->coordinate_coordinates[node * dimension + 1U] =
          dimension == 2U ? s * (1.0 - fold_scale * r * r * r) : s;
      if (dimension == 3U) {
        const double t = descriptor_node_coordinate(descriptor, global_node, 2U);
        fixture->coordinate_coordinates[node * 3U + 2U] =
            t * (1.0 - fold_scale * r * r * r);
      }
    }
    *expected_code = "MESH_GEOMETRY_SIGN_CHANGE";
  }
  return 0;
}

static int test_negative_geometry(MPI_Comm communicator, int rank, int size,
                                  char invariant[SS_SHA256_HEX_CAPACITY]) {
  char digest_input[512] = {0};
  size_t digest_length = 0U;
  int local_status = 0;
  uint32_t variant;

  for (variant = 0U; variant < 7U; ++variant) {
    CompactFixture fixture;
    const char *expected_code = NULL;
    SsMesh *mesh = NULL;
    SsMeshError error = {0};
    SsStatus status;
    int written;
    LOCAL_CHECK(configure_bad_fixture(variant, rank, size, &fixture, &expected_code) ==
                0);
    if (local_status != 0) {
      continue;
    }
    status = ss_mesh_create(communicator, &fixture.input, &mesh, &error);
    LOCAL_CHECK(status == SS_ERROR_INTEGRITY);
    LOCAL_CHECK(mesh == NULL);
    LOCAL_CHECK(error.valid != 0);
    LOCAL_CHECK(error.status == SS_ERROR_INTEGRITY);
    LOCAL_CHECK(error.stable_entity_id == fixture.cell_ids[0]);
    LOCAL_CHECK(strcmp(error.code, expected_code) == 0);
    LOCAL_CHECK(error.quadrature_rule_id[0] != '\0');
    written =
        snprintf(digest_input + digest_length, sizeof(digest_input) - digest_length,
                 "%s:%llu:", error.code, (unsigned long long)error.stable_entity_id);
    LOCAL_CHECK(written > 0 && (size_t)written < sizeof(digest_input) - digest_length);
    if (written > 0 && (size_t)written < sizeof(digest_input) - digest_length) {
      digest_length += (size_t)written;
    }
  }
  LOCAL_CHECK(ss_sha256_bytes((const unsigned char *)digest_input, digest_length,
                              invariant) == SS_OK);
  return local_status;
}

static double strip_polynomial(const SsCellPack *pack, size_t node, int quadratic) {
  const double x = pack->field_node_coordinates[node * (size_t)pack->dimension];
  const double y = pack->field_node_coordinates[node * (size_t)pack->dimension + 1U];
  return quadratic != 0 ? x * x + 0.5 * x * y : x;
}

static int fill_field_vector(const SsFieldSpace *space, Vec vector, int quadratic,
                             int rank) {
  SsCellPack packs[2] = {{0}};
  unsigned char seen[SS_FE_MAX_NODES * 4U] = {0};
  PetscInt ownership_start = 0;
  PetscInt ownership_end = 0;
  size_t pack_count = 0U;
  size_t pack;
  size_t inserted = 0U;
  int local_status = 0;
  LOCAL_CHECK(VecSet(vector, 0.0) == PETSC_SUCCESS);
  LOCAL_CHECK(VecGetOwnershipRange(vector, &ownership_start, &ownership_end) ==
              PETSC_SUCCESS);
  LOCAL_CHECK(ownership_end >= ownership_start);
  LOCAL_CHECK((size_t)(ownership_end - ownership_start) <= sizeof(seen));
  LOCAL_CHECK(ss_field_space_pack_owned_cells(space, packs, 2U, &pack_count) == SS_OK);
  for (pack = 0U; pack < pack_count; ++pack) {
    size_t node;
    for (node = 0U; node < packs[pack].field_node_count; ++node) {
      const PetscInt index = packs[pack].global_indices[node];
      if (index >= ownership_start && index < ownership_end &&
          seen[index - ownership_start] == 0U) {
        const double value = strip_polynomial(&packs[pack], node, quadratic);
        LOCAL_CHECK(VecSetValue(vector, index, (PetscScalar)value, INSERT_VALUES) ==
                    PETSC_SUCCESS);
        seen[index - ownership_start] = 1U;
        ++inserted;
      }
    }
  }
  LOCAL_CHECK(inserted == (size_t)(ownership_end - ownership_start));
  LOCAL_CHECK(VecAssemblyBegin(vector) == PETSC_SUCCESS);
  LOCAL_CHECK(VecAssemblyEnd(vector) == PETSC_SUCCESS);
  return local_status;
}

static int check_field_vector(const SsFieldSpace *space, Vec vector, int quadratic,
                              int rank, long long *checksum) {
  SsCellPack packs[2] = {{0}};
  unsigned char seen[SS_FE_MAX_NODES * 4U] = {0};
  PetscInt ownership_start = 0;
  PetscInt ownership_end = 0;
  size_t pack_count = 0U;
  size_t pack;
  size_t checked = 0U;
  int local_status = 0;
  LOCAL_CHECK(checksum != NULL);
  LOCAL_CHECK(VecGetOwnershipRange(vector, &ownership_start, &ownership_end) ==
              PETSC_SUCCESS);
  LOCAL_CHECK(ownership_end >= ownership_start);
  LOCAL_CHECK((size_t)(ownership_end - ownership_start) <= sizeof(seen));
  LOCAL_CHECK(ss_field_space_pack_owned_cells(space, packs, 2U, &pack_count) == SS_OK);
  for (pack = 0U; pack < pack_count; ++pack) {
    size_t node;
    for (node = 0U; node < packs[pack].field_node_count; ++node) {
      const PetscInt index = packs[pack].global_indices[node];
      if (index >= ownership_start && index < ownership_end &&
          seen[index - ownership_start] == 0U) {
        PetscScalar actual = 0.0;
        const double expected = strip_polynomial(&packs[pack], node, quadratic);
        const uint64_t weight =
            packs[pack].field_entity_ids[node] % UINT64_C(1009) +
            (uint64_t)packs[pack].field_entity_slots[node] * UINT64_C(17) + UINT64_C(1);
        LOCAL_CHECK(VecGetValues(vector, 1, &index, &actual) == PETSC_SUCCESS);
        LOCAL_CHECK(fabs((double)PetscRealPart(actual) - expected) < 8.0e-12);
        *checksum += (long long)llround((double)PetscRealPart(actual) * 1000000000.0) *
                     (long long)weight;
        seen[index - ownership_start] = 1U;
        ++checked;
      }
    }
  }
  LOCAL_CHECK(checked == (size_t)(ownership_end - ownership_start));
  return local_status;
}

static int test_distributed_transfer(MPI_Comm communicator, int rank, int size,
                                     long long *checksum) {
  const SsStrongConstraint p1_constraints[] = {
      {UINT64_C(0x110000000000), 0U, 0U, 0.0},
      {UINT64_C(0x110000000002), 0U, 0U, 0.0},
  };
  const SsStrongConstraint p2_constraints[] = {
      {UINT64_C(0x110000000000), 0U, 0U, 0.0},
      {UINT64_C(0x110000000002), 0U, 0U, 0.0},
      {UINT64_C(0x130000000003), 0U, 0U, 0.0},
  };
  const SsStrongConstraint p4_constraints[] = {
      {UINT64_C(0x110000000000), 0U, 0U, 0.0}, {UINT64_C(0x110000000002), 0U, 0U, 0.0},
      {UINT64_C(0x130000000003), 0U, 0U, 0.0}, {UINT64_C(0x130000000003), 1U, 0U, 0.0},
      {UINT64_C(0x130000000003), 2U, 0U, 0.0},
  };
  const SsFieldSpec specs[] = {
      {1U, 1U, sizeof(p1_constraints) / sizeof(p1_constraints[0]), p1_constraints},
      {2U, 1U, sizeof(p2_constraints) / sizeof(p2_constraints[0]), p2_constraints},
      {4U, 1U, sizeof(p4_constraints) / sizeof(p4_constraints[0]), p4_constraints},
  };
  const SsFieldSpec zero_row_p2_spec = {
      2U, 1U, sizeof(p1_constraints) / sizeof(p1_constraints[0]), p1_constraints};
  StripFixture fixture;
  SsMesh *mesh = NULL;
  SsMeshError error = {0};
  SsFieldSpace *spaces[3] = {NULL, NULL, NULL};
  SsFieldSpace *zero_row_p2_space = NULL;
  Mat p12 = NULL;
  Mat p24 = NULL;
  Mat p14 = NULL;
  Mat zero_row_p12 = NULL;
  Vec p1_values = NULL;
  Vec p2_values = NULL;
  Vec p2_probe = NULL;
  Vec p1_restricted = NULL;
  Vec p4_composed = NULL;
  Vec p4_direct = NULL;
  PetscInt matrix_rows = 0;
  PetscInt matrix_columns = 0;
  PetscReal difference = 0.0;
  PetscScalar adjoint_left = 0.0;
  PetscScalar adjoint_right = 0.0;
  long long local_zero_rows = 0LL;
  long long global_zero_rows = 0LL;
  int local_status = 0;
  size_t field;
  LOCAL_CHECK(checksum != NULL);
  LOCAL_CHECK(build_strip_fixture(rank, size, &fixture) == 0);
  LOCAL_CHECK(ss_mesh_create(communicator, &fixture.input, &mesh, &error) == SS_OK);
  LOCAL_CHECK(mesh != NULL && error.valid == 0);
  if (mesh == NULL) {
    return 1;
  }
  for (field = 0U; field < 3U; ++field) {
    LOCAL_CHECK(ss_mesh_field_space_create(mesh, &specs[field], &spaces[field]) ==
                SS_OK);
  }
  LOCAL_CHECK(ss_mesh_field_space_create(mesh, &zero_row_p2_spec, &zero_row_p2_space) ==
              SS_OK);
  if (spaces[0] != NULL && spaces[1] != NULL && spaces[2] != NULL) {
    LOCAL_CHECK(ss_field_space_transfer_create(spaces[0], spaces[1], &p12) == SS_OK);
    LOCAL_CHECK(ss_field_space_transfer_create(spaces[1], spaces[2], &p24) == SS_OK);
    LOCAL_CHECK(ss_field_space_transfer_create(spaces[0], spaces[2], &p14) == SS_OK);
    LOCAL_CHECK(ss_field_space_create_global_vector(spaces[0], &p1_values) == SS_OK);
    LOCAL_CHECK(ss_field_space_create_global_vector(spaces[1], &p2_values) == SS_OK);
    LOCAL_CHECK(ss_field_space_create_global_vector(spaces[1], &p2_probe) == SS_OK);
    LOCAL_CHECK(ss_field_space_create_global_vector(spaces[0], &p1_restricted) ==
                SS_OK);
    LOCAL_CHECK(ss_field_space_create_global_vector(spaces[2], &p4_composed) == SS_OK);
    LOCAL_CHECK(ss_field_space_create_global_vector(spaces[2], &p4_direct) == SS_OK);
  }
  if (spaces[0] != NULL && zero_row_p2_space != NULL) {
    LOCAL_CHECK(ss_field_space_transfer_create(spaces[0], zero_row_p2_space,
                                               &zero_row_p12) == SS_OK);
  }
  if (zero_row_p12 != NULL) {
    PetscInt row_start = 0;
    PetscInt row_end = 0;
    PetscInt row;
    if (MatGetOwnershipRange(zero_row_p12, &row_start, &row_end) == PETSC_SUCCESS) {
      for (row = row_start; row < row_end; ++row) {
        PetscInt count = 0;
        const PetscInt *indices = NULL;
        const PetscScalar *values = NULL;
        if (MatGetRow(zero_row_p12, row, &count, &indices, &values) == PETSC_SUCCESS) {
          if (count == 0) {
            ++local_zero_rows;
          }
          LOCAL_CHECK(MatRestoreRow(zero_row_p12, row, &count, &indices, &values) ==
                      PETSC_SUCCESS);
        } else {
          LOCAL_CHECK(0);
        }
      }
    } else {
      LOCAL_CHECK(0);
    }
  }
  LOCAL_CHECK(MPI_Allreduce(&local_zero_rows, &global_zero_rows, 1, MPI_LONG_LONG,
                            MPI_SUM, communicator) == MPI_SUCCESS);
  LOCAL_CHECK(global_zero_rows > 0LL);
  if (p12 != NULL && p24 != NULL && p14 != NULL && p1_values != NULL &&
      p2_values != NULL && p2_probe != NULL && p1_restricted != NULL &&
      p4_composed != NULL && p4_direct != NULL) {
    LOCAL_CHECK(MatGetSize(p12, &matrix_rows, &matrix_columns) == PETSC_SUCCESS);
    LOCAL_CHECK(matrix_rows == 6 && matrix_columns == 2);
    LOCAL_CHECK(MatGetSize(p24, &matrix_rows, &matrix_columns) == PETSC_SUCCESS);
    LOCAL_CHECK(matrix_rows == 20 && matrix_columns == 6);
    LOCAL_CHECK(MatGetSize(p14, &matrix_rows, &matrix_columns) == PETSC_SUCCESS);
    LOCAL_CHECK(matrix_rows == 20 && matrix_columns == 2);
    LOCAL_CHECK(fill_field_vector(spaces[0], p1_values, 0, rank) == 0);
    LOCAL_CHECK(MatMult(p12, p1_values, p2_values) == PETSC_SUCCESS);
    LOCAL_CHECK(fill_field_vector(spaces[1], p2_probe, 1, rank) == 0);
    LOCAL_CHECK(MatMultTranspose(p12, p2_probe, p1_restricted) == PETSC_SUCCESS);
    LOCAL_CHECK(VecDot(p2_values, p2_probe, &adjoint_left) == PETSC_SUCCESS);
    LOCAL_CHECK(VecDot(p1_values, p1_restricted, &adjoint_right) == PETSC_SUCCESS);
    LOCAL_CHECK(PetscAbsScalar(adjoint_left - adjoint_right) < 8.0e-12);
    LOCAL_CHECK(MatMult(p24, p2_values, p4_composed) == PETSC_SUCCESS);
    LOCAL_CHECK(MatMult(p14, p1_values, p4_direct) == PETSC_SUCCESS);
    LOCAL_CHECK(check_field_vector(spaces[1], p2_values, 0, rank, checksum) == 0);
    LOCAL_CHECK(check_field_vector(spaces[2], p4_composed, 0, rank, checksum) == 0);
    LOCAL_CHECK(VecAXPY(p4_direct, -1.0, p4_composed) == PETSC_SUCCESS);
    LOCAL_CHECK(VecNorm(p4_direct, NORM_INFINITY, &difference) == PETSC_SUCCESS);
    LOCAL_CHECK(difference < 8.0e-12);
    LOCAL_CHECK(fill_field_vector(spaces[1], p2_values, 1, rank) == 0);
    LOCAL_CHECK(MatMult(p24, p2_values, p4_composed) == PETSC_SUCCESS);
    LOCAL_CHECK(check_field_vector(spaces[2], p4_composed, 1, rank, checksum) == 0);
  }
  LOCAL_CHECK(VecDestroy(&p1_values) == PETSC_SUCCESS);
  LOCAL_CHECK(VecDestroy(&p2_values) == PETSC_SUCCESS);
  LOCAL_CHECK(VecDestroy(&p2_probe) == PETSC_SUCCESS);
  LOCAL_CHECK(VecDestroy(&p1_restricted) == PETSC_SUCCESS);
  LOCAL_CHECK(VecDestroy(&p4_composed) == PETSC_SUCCESS);
  LOCAL_CHECK(VecDestroy(&p4_direct) == PETSC_SUCCESS);
  LOCAL_CHECK(MatDestroy(&p12) == PETSC_SUCCESS);
  LOCAL_CHECK(MatDestroy(&p24) == PETSC_SUCCESS);
  LOCAL_CHECK(MatDestroy(&p14) == PETSC_SUCCESS);
  LOCAL_CHECK(MatDestroy(&zero_row_p12) == PETSC_SUCCESS);
  for (field = 0U; field < 3U; ++field) {
    LOCAL_CHECK(ss_field_space_destroy(&spaces[field]) == SS_OK);
  }
  LOCAL_CHECK(ss_field_space_destroy(&zero_row_p2_space) == SS_OK);
  LOCAL_CHECK(ss_mesh_destroy(&mesh) == SS_OK);
  return local_status;
}

static int test_distributed_tetra_transfer(MPI_Comm communicator, int rank, int size,
                                           long long *checksum) {
  const SsFieldSpec p2_spec = {2U, 1U, 0U, NULL};
  const SsFieldSpec p4_spec = {4U, 1U, 0U, NULL};
  const SsFieldSpec p8_spec = {8U, 1U, 0U, NULL};
  TetraPairFixture fixture;
  SsMesh *mesh = NULL;
  SsMeshError error = {0};
  SsMeshSummary mesh_summary = {0};
  SsFieldSpace *p2_space = NULL;
  SsFieldSpace *p4_space = NULL;
  SsFieldSpace *p8_space = NULL;
  SsAssemblyPlan *p8_plan = NULL;
  SsAssemblyPlanSummary p8_plan_summary = {0};
  SsErrorRecord p8_plan_error = {0};
  Mat p24 = NULL;
  Mat p48 = NULL;
  Vec p2_values = NULL;
  Vec p4_values = NULL;
  Vec p8_values = NULL;
  Vec p2_restricted = NULL;
  PetscInt matrix_rows = 0;
  PetscInt matrix_columns = 0;
  PetscScalar adjoint_left = 0.0;
  PetscScalar adjoint_right = 0.0;
  int local_status = 0;
  LOCAL_CHECK(checksum != NULL);
  LOCAL_CHECK(build_tetra_pair_fixture(rank, size, &fixture) == 0);
  LOCAL_CHECK(ss_mesh_create(communicator, &fixture.input, &mesh, &error) == SS_OK);
  LOCAL_CHECK(mesh != NULL && error.valid == 0);
  if (mesh == NULL) {
    return 1;
  }
  LOCAL_CHECK(ss_mesh_summary(mesh, &mesh_summary) == SS_OK);
  LOCAL_CHECK(mesh_summary.global_vertices == 5U);
  LOCAL_CHECK(mesh_summary.global_edges == 9U);
  LOCAL_CHECK(mesh_summary.global_facets == 6U);
  LOCAL_CHECK(mesh_summary.global_intermediate_entities == 16U);
  LOCAL_CHECK(mesh_summary.global_cells == 2U);
  LOCAL_CHECK(ss_mesh_field_space_create(mesh, &p2_spec, &p2_space) == SS_OK);
  LOCAL_CHECK(ss_mesh_field_space_create(mesh, &p4_spec, &p4_space) == SS_OK);
  LOCAL_CHECK(ss_mesh_field_space_create(mesh, &p8_spec, &p8_space) == SS_OK);
  if (p2_space != NULL && p4_space != NULL && p8_space != NULL) {
    SsFieldSummary p8_summary = {0};
    size_t facet_count = 0U;
    SsFieldFacetPack *facets = NULL;
    LOCAL_CHECK(ss_field_space_transfer_create(p2_space, p4_space, &p24) == SS_OK);
    LOCAL_CHECK(ss_field_space_transfer_create(p4_space, p8_space, &p48) == SS_OK);
    LOCAL_CHECK(ss_field_space_create_global_vector(p2_space, &p2_values) == SS_OK);
    LOCAL_CHECK(ss_field_space_create_global_vector(p4_space, &p4_values) == SS_OK);
    LOCAL_CHECK(ss_field_space_create_global_vector(p8_space, &p8_values) == SS_OK);
    LOCAL_CHECK(ss_field_space_create_global_vector(p2_space, &p2_restricted) == SS_OK);
    LOCAL_CHECK(ss_field_space_summary(p8_space, &p8_summary) == SS_OK);
    LOCAL_CHECK(p8_summary.global_dof_count == 285U);
    LOCAL_CHECK(
        check_owned_free_dof_map(communicator, rank, size, p8_space, &p8_summary) == 0);
    LOCAL_CHECK(ss_assembly_plan_create(communicator, p8_space, &p8_plan,
                                        &p8_plan_error) == SS_OK);
    LOCAL_CHECK(p8_plan != NULL && p8_plan_error.valid == 0);
    if (p8_plan != NULL) {
      LOCAL_CHECK(ss_assembly_plan_summary(p8_plan, &p8_plan_summary) == SS_OK);
      LOCAL_CHECK(p8_plan_summary.global_rows == 285U);
    }
    LOCAL_CHECK(ss_field_space_pack_owned_facets(p8_space, 401U, NULL, 0U,
                                                 &facet_count) == SS_OK);
    facets = (SsFieldFacetPack *)calloc(facet_count == 0U ? 1U : facet_count,
                                        sizeof(*facets));
    LOCAL_CHECK(facets != NULL);
    LOCAL_CHECK(ss_field_space_pack_owned_facets(p8_space, 401U, facets, facet_count,
                                                 &facet_count) == SS_OK);
    if (facets != NULL) {
      size_t facet;
      for (facet = 0U; facet < facet_count; ++facet) {
        LOCAL_CHECK(facets[facet].field_node_count == 45U);
        LOCAL_CHECK(facets[facet].closure_count == 45U);
      }
    }
    free(facets);
  }
  if (p24 != NULL && p48 != NULL && p2_values != NULL && p4_values != NULL &&
      p8_values != NULL && p2_restricted != NULL) {
    LOCAL_CHECK(MatGetSize(p24, &matrix_rows, &matrix_columns) == PETSC_SUCCESS);
    LOCAL_CHECK(matrix_rows == 55 && matrix_columns == 14);
    LOCAL_CHECK(MatGetSize(p48, &matrix_rows, &matrix_columns) == PETSC_SUCCESS);
    LOCAL_CHECK(matrix_rows == 285 && matrix_columns == 55);
    LOCAL_CHECK(fill_field_vector(p2_space, p2_values, 1, rank) == 0);
    LOCAL_CHECK(MatMult(p24, p2_values, p4_values) == PETSC_SUCCESS);
    LOCAL_CHECK(check_field_vector(p4_space, p4_values, 1, rank, checksum) == 0);
    LOCAL_CHECK(MatMult(p48, p4_values, p8_values) == PETSC_SUCCESS);
    LOCAL_CHECK(check_field_vector(p8_space, p8_values, 1, rank, checksum) == 0);
    LOCAL_CHECK(MatMultTranspose(p24, p4_values, p2_restricted) == PETSC_SUCCESS);
    LOCAL_CHECK(VecDot(p4_values, p4_values, &adjoint_left) == PETSC_SUCCESS);
    LOCAL_CHECK(VecDot(p2_values, p2_restricted, &adjoint_right) == PETSC_SUCCESS);
    LOCAL_CHECK(PetscAbsScalar(adjoint_left - adjoint_right) < 2.0e-11);
  }
  LOCAL_CHECK(VecDestroy(&p2_values) == PETSC_SUCCESS);
  LOCAL_CHECK(VecDestroy(&p4_values) == PETSC_SUCCESS);
  LOCAL_CHECK(VecDestroy(&p8_values) == PETSC_SUCCESS);
  LOCAL_CHECK(VecDestroy(&p2_restricted) == PETSC_SUCCESS);
  LOCAL_CHECK(MatDestroy(&p24) == PETSC_SUCCESS);
  LOCAL_CHECK(MatDestroy(&p48) == PETSC_SUCCESS);
  LOCAL_CHECK(ss_assembly_plan_destroy(&p8_plan) == SS_OK);
  LOCAL_CHECK(ss_field_space_destroy(&p2_space) == SS_OK);
  LOCAL_CHECK(ss_field_space_destroy(&p4_space) == SS_OK);
  LOCAL_CHECK(ss_field_space_destroy(&p8_space) == SS_OK);
  LOCAL_CHECK(ss_mesh_destroy(&mesh) == SS_OK);
  return local_status;
}

static int test_fe_contract(MPI_Comm communicator, int rank, int size,
                            char invariant[SS_SHA256_HEX_CAPACITY]) {
  const uint32_t dimensions[] = {2U, 3U};
  long long local_checksum = 0LL;
  long long global_checksum = 0LL;
  int local_status = 0;
  size_t dimension_index;

  for (dimension_index = 0U; dimension_index < 2U; ++dimension_index) {
    const uint32_t dimension = dimensions[dimension_index];
    const SsFeDescriptor *coarse = NULL;
    const SsFeDescriptor *fine = NULL;
    double transfer[SS_FE_MAX_NODES * SS_FE_MAX_NODES] = {0.0};
    size_t rows = 0U;
    size_t columns = 0U;
    uint64_t global_cell;
    LOCAL_CHECK(ss_fe_descriptor_find(dimension, 2U, &coarse) == SS_OK);
    LOCAL_CHECK(ss_fe_descriptor_find(dimension, 4U, &fine) == SS_OK);
    LOCAL_CHECK(ss_fe_transfer_build(coarse, fine, transfer,
                                     SS_FE_MAX_NODES * SS_FE_MAX_NODES, &rows,
                                     &columns) == SS_OK);
    for (global_cell = 0U; global_cell < 17U; ++global_cell) {
      double coarse_values[SS_FE_MAX_NODES] = {0.0};
      double fine_values[SS_FE_MAX_NODES] = {0.0};
      uint8_t constrained[SS_FE_MAX_NODES] = {0};
      size_t node;
      if ((int)(global_cell % (uint64_t)size) != rank) {
        continue;
      }
      for (node = 0U; node < columns; ++node) {
        const double r = descriptor_node_coordinate(coarse, node, 0U);
        const double s = descriptor_node_coordinate(coarse, node, 1U);
        const double t =
            dimension == 3U ? descriptor_node_coordinate(coarse, node, 2U) : 0.0;
        coarse_values[node] = 1.0 + 0.01 * (double)global_cell + 0.7 * r - 0.2 * s +
                              0.4 * r * r + 0.3 * s * s + 0.15 * t - 0.1 * r * t;
      }
      for (node = 0U; node < rows; ++node) {
        constrained[node] = fine->entity_dimensions[node] == 0U ? 1U : 0U;
      }
      LOCAL_CHECK(ss_fe_transfer_apply_constrained(rows, columns, transfer, 1U,
                                                   coarse_values, constrained,
                                                   fine_values) == SS_OK);
      for (node = 0U; node < rows; ++node) {
        const double r = descriptor_node_coordinate(fine, node, 0U);
        const double s = descriptor_node_coordinate(fine, node, 1U);
        const double t =
            dimension == 3U ? descriptor_node_coordinate(fine, node, 2U) : 0.0;
        const double expected = constrained[node] != 0U
                                    ? 0.0
                                    : 1.0 + 0.01 * (double)global_cell + 0.7 * r -
                                          0.2 * s + 0.4 * r * r + 0.3 * s * s +
                                          0.15 * t - 0.1 * r * t;
        LOCAL_CHECK(fabs(fine_values[node] - expected) < 4.0e-12);
        local_checksum += (long long)llround(fine_values[node] * 1000000000.0) *
                          (long long)(node + 1U);
      }
      {
        const double xi[3] = {0.17, 0.23, 0.11};
        double values[SS_FE_MAX_NODES] = {0.0};
        double gradients[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
        LOCAL_CHECK(ss_fe_basis_eval(fine, 1U, xi, values, gradients) == SS_OK);
        for (node = 0U; node < rows; ++node) {
          local_checksum += (long long)llround(values[node] * 1000000000000.0) *
                            (long long)(global_cell + node + 1U);
        }
      }
    }
  }
  LOCAL_CHECK(test_distributed_transfer(communicator, rank, size, &local_checksum) ==
              0);
  LOCAL_CHECK(
      test_distributed_tetra_transfer(communicator, rank, size, &local_checksum) == 0);
  LOCAL_CHECK(MPI_Allreduce(&local_checksum, &global_checksum, 1, MPI_LONG_LONG,
                            MPI_SUM, communicator) == MPI_SUCCESS);
  LOCAL_CHECK(ss_sha256_bytes((const unsigned char *)&global_checksum,
                              sizeof(global_checksum), invariant) == SS_OK);
  return local_status;
}

static int write_output(const char *path, const char *test_id, int size,
                        const char *invariant) {
  FILE *stream = fopen(path, "wb");
  int result;
  if (stream == NULL) {
    return 1;
  }
  result = fprintf(stream,
                   "{\"test_id\":\"%s\",\"ranks\":%d,"
                   "\"invariant\":\"sha256:%s\"}\n",
                   test_id, size, invariant) < 0
               ? 1
               : 0;
  if (fclose(stream) != 0) {
    result = 1;
  }
  return result;
}

/* Select the shared-point balancing of the fixture meshes by its setting name
 * (pmg.shared_point_balance). */
static int parse_shared_point_balance(const char *name) {
  if (strcmp(name, "petsc") == 0) {
    fixture_shared_point_balance = SS_MESH_BALANCE_PETSC;
  } else if (strcmp(name, "stable_hash_p2") == 0) {
    fixture_shared_point_balance = SS_MESH_BALANCE_STABLE_P2;
  } else if (strcmp(name, "stable_hash_high_order") == 0) {
    fixture_shared_point_balance = SS_MESH_BALANCE_STABLE_HIGH_ORDER;
  } else {
    return 1;
  }
  return 0;
}

int main(int argc, char **argv) {
  const char *test_id;
  const char *output_path;
  int expected_ranks;
  int provided = MPI_THREAD_SINGLE;
  int rank = -1;
  int size = 0;
  int local_status = 0;
  int global_status = 0;
  char invariant[SS_SHA256_HEX_CAPACITY] = {0};

  if ((argc != 7 && argc != 9) || strcmp(argv[1], "--test-id") != 0 ||
      strcmp(argv[3], "--expected-ranks") != 0 || strcmp(argv[5], "--output") != 0 ||
      (argc == 9 && (strcmp(argv[7], "--shared-point-balance") != 0 ||
                     parse_shared_point_balance(argv[8]) != 0))) {
    fputs("usage: ss_test_mesh_field_mpi --test-id ID --expected-ranks N "
          "--output PATH [--shared-point-balance "
          "petsc|stable_hash_p2|stable_hash_high_order]\n",
          stderr);
    return 2;
  }
  test_id = argv[2];
  expected_ranks = atoi(argv[4]);
  output_path = argv[6];
  if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
    return 3;
  }
  if (PetscInitialize(&argc, &argv, NULL, NULL) != PETSC_SUCCESS) {
    MPI_Abort(MPI_COMM_WORLD, 4);
    return 4;
  }
  if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS) {
    MPI_Abort(MPI_COMM_WORLD, 5);
    return 5;
  }
  LOCAL_CHECK(size == expected_ranks);
  LOCAL_CHECK(provided >= MPI_THREAD_FUNNELED);
  if (strcmp(test_id, "MSH-CONTRACT-DISTRIBUTE-001") == 0) {
    local_status = test_distribution(MPI_COMM_WORLD, rank, size, invariant);
  } else if (strcmp(test_id, "MSH-NEG-GEOMETRY-001") == 0) {
    local_status = test_negative_geometry(MPI_COMM_WORLD, rank, size, invariant);
  } else if (strcmp(test_id, "FE-CONTRACT-MPI-001") == 0) {
    local_status = test_fe_contract(MPI_COMM_WORLD, rank, size, invariant);
  } else {
    local_status = 1;
  }
  if (MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS) {
    global_status = 1;
  }
  if (rank == 0 && global_status == 0 &&
      write_output(output_path, test_id, size, invariant) != 0) {
    global_status = 1;
  }
  if (MPI_Bcast(&global_status, 1, MPI_INT, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
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
