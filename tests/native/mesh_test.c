/* Single-process mesh tests: stable identifiers, labels, sections, lifting
 * and (curved) facets.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/fe/fe.h"
#include "slope_internal/mesh/mesh.h"

#include <petscsys.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(expression)                                                              \
  do {                                                                                 \
    if (!(expression)) {                                                               \
      fprintf(stderr, "%s:%d: assertion failed: %s\n", __FILE__, __LINE__,             \
              #expression);                                                            \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

#define CHECK_NEAR(actual, expected, tolerance)                                        \
  do {                                                                                 \
    const double check_actual = (actual);                                              \
    const double check_expected = (expected);                                          \
    if (!isfinite(check_actual) ||                                                     \
        fabs(check_actual - check_expected) > (tolerance)) {                           \
      fprintf(stderr, "%s:%d: |%.17g - %.17g| > %.3g\n", __FILE__, __LINE__,           \
              check_actual, check_expected, (double)(tolerance));                      \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

static const SsEntityId square_vertex_ids[] = {
    UINT64_C(0x10000000001),
    UINT64_C(0x20000000002),
    UINT64_C(0x30000000003),
    UINT64_C(0x40000000004),
};
static const double square_coordinates[] = {
    0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
};
static const uint64_t square_cells[] = {
    0U, 1U, 2U, 1U, 3U, 2U,
};
static const SsEntityId square_cell_ids[] = {
    UINT64_C(0x70000000007),
    UINT64_C(0x80000000008),
};
static const uint64_t square_facets[] = {
    0U, 1U, 1U, 3U, 3U, 2U, 2U, 0U,
};
static const SsEntityId square_facet_ids[] = {
    UINT64_C(0x90000000009),
    UINT64_C(0xa000000000a),
    UINT64_C(0xb000000000b),
    UINT64_C(0xc000000000c),
};
static const SsMeshLabelAssignment square_labels[] = {
    {SS_MESH_LABEL_MATERIAL, UINT64_C(0x70000000007), 41U},
    {SS_MESH_LABEL_MATERIAL, UINT64_C(0x80000000008), 42U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0x90000000009), 11U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0xa000000000a), 12U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0xb000000000b), 13U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(0xc000000000c), 14U},
    {SS_MESH_LABEL_INTERFACE, UINT64_C(0xa000000000a), 71U},
};
static const SsMeshIntermediateEntity square_intermediate_entities[] = {
    {1U,
     UINT64_C(0x90000000009),
     {UINT64_C(0x10000000001), UINT64_C(0x20000000002), 0U}},
    {1U,
     UINT64_C(0xc000000000c),
     {UINT64_C(0x10000000001), UINT64_C(0x30000000003), 0U}},
    {1U,
     UINT64_C(0xd000000000d),
     {UINT64_C(0x20000000002), UINT64_C(0x30000000003), 0U}},
    {1U,
     UINT64_C(0xa000000000a),
     {UINT64_C(0x20000000002), UINT64_C(0x40000000004), 0U}},
    {1U,
     UINT64_C(0xb000000000b),
     {UINT64_C(0x30000000003), UINT64_C(0x40000000004), 0U}},
};

static const SsEntityId tetra_vertex_ids[] = {
    UINT64_C(101),
    UINT64_C(202),
    UINT64_C(303),
    UINT64_C(404),
};
static const double tetra_coordinates[] = {
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
};
static const uint64_t tetra_cells[] = {0U, 1U, 2U, 3U};
static const SsEntityId tetra_cell_ids[] = {UINT64_C(505)};
static const uint64_t tetra_facets[] = {
    0U, 1U, 2U, 0U, 1U, 3U, 0U, 2U, 3U, 1U, 2U, 3U,
};
static const SsEntityId tetra_facet_ids[] = {
    UINT64_C(601),
    UINT64_C(602),
    UINT64_C(603),
    UINT64_C(604),
};
static const SsMeshLabelAssignment tetra_labels[] = {
    {SS_MESH_LABEL_MATERIAL, UINT64_C(505), 51U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(601), 21U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(602), 22U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(603), 23U},
    {SS_MESH_LABEL_SUPPORT, UINT64_C(604), 24U},
};
static const SsMeshIntermediateEntity tetra_intermediate_entities[] = {
    {1U, UINT64_C(701), {UINT64_C(101), UINT64_C(202), 0U}},
    {1U, UINT64_C(702), {UINT64_C(101), UINT64_C(303), 0U}},
    {1U, UINT64_C(703), {UINT64_C(101), UINT64_C(404), 0U}},
    {1U, UINT64_C(704), {UINT64_C(202), UINT64_C(303), 0U}},
    {1U, UINT64_C(705), {UINT64_C(202), UINT64_C(404), 0U}},
    {1U, UINT64_C(706), {UINT64_C(303), UINT64_C(404), 0U}},
    {2U, UINT64_C(601), {UINT64_C(101), UINT64_C(202), UINT64_C(303)}},
    {2U, UINT64_C(602), {UINT64_C(101), UINT64_C(202), UINT64_C(404)}},
    {2U, UINT64_C(603), {UINT64_C(101), UINT64_C(303), UINT64_C(404)}},
    {2U, UINT64_C(604), {UINT64_C(202), UINT64_C(303), UINT64_C(404)}},
};

static SsMeshInput square_input(void) {
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

static SsMeshInput tetra_input(void) {
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

static int create_mesh(const SsMeshInput *input, SsMesh **mesh) {
  SsMeshError error = {0};
  CHECK(ss_mesh_create(PETSC_COMM_SELF, input, mesh, &error) == SS_OK);
  CHECK(*mesh != NULL);
  CHECK(error.valid == 0);
  return 0;
}

static int test_id_map(void) {
  SsMeshInput input = square_input();
  SsMesh *mesh = NULL;
  SsMeshSummary before_summary = {0};
  SsMeshSummary after_summary = {0};
  PetscInt before[10] = {0};
  PetscInt after = 0;
  size_t index;
  size_t changed = 0U;
  CHECK(create_mesh(&input, &mesh) == 0);
  CHECK(ss_mesh_summary(mesh, &before_summary) == SS_OK);
  CHECK(strlen(before_summary.topology_digest) == 64U);
  CHECK(before_summary.topology_generation == 1U);
  CHECK(before_summary.coordinate_generation == 1U);
  CHECK(before_summary.label_generation == 1U);
  CHECK(before_summary.partition_generation == 1U);
  CHECK(before_summary.owned_vertices == 4U);
  CHECK(before_summary.ghost_vertices == 0U);
  CHECK(before_summary.owned_edges == 5U);
  CHECK(before_summary.ghost_edges == 0U);
  CHECK(before_summary.owned_facets == 4U);
  CHECK(before_summary.ghost_facets == 0U);
  CHECK(before_summary.owned_intermediate_entities == 5U);
  CHECK(before_summary.ghost_intermediate_entities == 0U);
  CHECK(before_summary.owned_coordinate_nodes == 4U);
  CHECK(before_summary.ghost_coordinate_nodes == 0U);
  CHECK(before_summary.minimum_condition > 0.0);
  CHECK(before_summary.maximum_condition >= before_summary.minimum_condition);
  CHECK_NEAR(before_summary.partition_imbalance, 1.0, 0.0);
  CHECK(before_summary.cache_memory_bytes > 0U);
  CHECK(before_summary.owned_facet_plan_count == 4U);
  CHECK(before_summary.facet_plan_bytes >=
        before_summary.owned_facet_plan_count * (uint64_t)sizeof(SsFacetPack));
  CHECK(strcmp(before_summary.coordinate_basis_id, "triangle-p1-v1") == 0);
  CHECK(before_summary.geometry_quadrature_id[0] != '\0');
  CHECK(before_summary.geometry_audit_rule_count > 1U);
  CHECK(before_summary.geometry_audit_point_count >
        before_summary.geometry_audit_rule_count);
  CHECK(strlen(before_summary.geometry_audit_rule_digest) == 64U);

  for (index = 0U; index < 4U; ++index) {
    SsEntityId recovered = 0U;
    CHECK(ss_mesh_point_from_stable_id(mesh, SS_MESH_ENTITY_VERTEX,
                                       square_vertex_ids[index],
                                       &before[index]) == SS_OK);
    CHECK(ss_mesh_stable_id_from_point(mesh, before[index], &recovered) == SS_OK);
    CHECK(recovered == square_vertex_ids[index]);
    CHECK((uint64_t)before[index] != square_vertex_ids[index]);
  }
  for (index = 0U; index < 2U; ++index) {
    SsEntityId recovered = 0U;
    CHECK(ss_mesh_point_from_stable_id(mesh, SS_MESH_ENTITY_CELL,
                                       square_cell_ids[index],
                                       &before[4U + index]) == SS_OK);
    CHECK(ss_mesh_stable_id_from_point(mesh, before[4U + index], &recovered) == SS_OK);
    CHECK(recovered == square_cell_ids[index]);
  }
  for (index = 0U; index < 4U; ++index) {
    CHECK(ss_mesh_point_from_stable_id(mesh, SS_MESH_ENTITY_FACET,
                                       square_facet_ids[index],
                                       &before[6U + index]) == SS_OK);
  }
  CHECK(ss_mesh_point_from_stable_id(mesh, SS_MESH_ENTITY_EDGE,
                                     square_intermediate_entities[2U].stable_entity_id,
                                     &after) == SS_OK);
  {
    SsEntityId recovered = 0U;
    CHECK(ss_mesh_stable_id_from_point(mesh, after, &recovered) == SS_OK);
    CHECK(recovered == square_intermediate_entities[2U].stable_entity_id);
  }

  CHECK(ss_mesh_renumber(mesh, UINT64_C(0x5a5aa5a55a5aa5a5)) == SS_OK);
  for (index = 0U; index < 4U; ++index) {
    CHECK(ss_mesh_point_from_stable_id(mesh, SS_MESH_ENTITY_VERTEX,
                                       square_vertex_ids[index], &after) == SS_OK);
    if (after != before[index]) {
      ++changed;
    }
  }
  for (index = 0U; index < 2U; ++index) {
    CHECK(ss_mesh_point_from_stable_id(mesh, SS_MESH_ENTITY_CELL,
                                       square_cell_ids[index], &after) == SS_OK);
    if (after != before[4U + index]) {
      ++changed;
    }
  }
  CHECK(changed > 0U);
  CHECK(ss_mesh_summary(mesh, &after_summary) == SS_OK);
  CHECK(strcmp(after_summary.topology_digest, before_summary.topology_digest) == 0);
  CHECK(after_summary.topology_generation == before_summary.topology_generation);
  CHECK(after_summary.coordinate_generation == before_summary.coordinate_generation);
  CHECK(after_summary.label_generation == before_summary.label_generation);
  CHECK(after_summary.partition_generation == before_summary.partition_generation + 1U);
  CHECK(ss_mesh_point_from_stable_id(mesh, SS_MESH_ENTITY_CELL, UINT64_C(999999),
                                     &after) == SS_ERROR_NOT_FOUND);
  CHECK(ss_mesh_destroy(&mesh) == SS_OK && mesh == NULL);
  {
    SsEntityId duplicate_vertex_ids[4];
    SsMeshError error = {0};
    memcpy(duplicate_vertex_ids, square_vertex_ids, sizeof(duplicate_vertex_ids));
    duplicate_vertex_ids[3] = duplicate_vertex_ids[0];
    input.vertex_ids = duplicate_vertex_ids;
    CHECK(ss_mesh_create(PETSC_COMM_SELF, &input, &mesh, &error) == SS_ERROR_DUPLICATE);
    CHECK(mesh == NULL);
  }
  {
    SsEntityId colliding_cell_ids[2];
    SsMeshLabelAssignment
        colliding_labels[sizeof(square_labels) / sizeof(square_labels[0])];
    SsMeshError error = {0};
    memcpy(colliding_cell_ids, square_cell_ids, sizeof(colliding_cell_ids));
    memcpy(colliding_labels, square_labels, sizeof(colliding_labels));
    colliding_cell_ids[0] = square_intermediate_entities[2U].stable_entity_id;
    colliding_labels[0].stable_entity_id = colliding_cell_ids[0];
    input = square_input();
    input.cell_ids = colliding_cell_ids;
    input.labels = colliding_labels;
    CHECK(ss_mesh_create(PETSC_COMM_SELF, &input, &mesh, &error) == SS_ERROR_DUPLICATE);
    CHECK(mesh == NULL);
  }
  {
    SsMeshError error = {0};
    input = square_input();
    --input.intermediate_entity_count;
    CHECK(ss_mesh_create(PETSC_COMM_SELF, &input, &mesh, &error) == SS_ERROR_INTEGRITY);
    CHECK(mesh == NULL);
  }
  {
    SsMeshIntermediateEntity inconsistent[sizeof(square_intermediate_entities) /
                                          sizeof(square_intermediate_entities[0])];
    SsMeshError error = {0};
    memcpy(inconsistent, square_intermediate_entities, sizeof(inconsistent));
    inconsistent[2U].stable_entity_id = UINT64_C(0xe000000000e);
    inconsistent[3U].stable_entity_id = inconsistent[2U].stable_entity_id;
    input = square_input();
    input.intermediate_entities = inconsistent;
    CHECK(ss_mesh_create(PETSC_COMM_SELF, &input, &mesh, &error) == SS_ERROR_DUPLICATE);
    CHECK(mesh == NULL);
  }
  {
    SsMeshIntermediateEntity mismatched[sizeof(square_intermediate_entities) /
                                        sizeof(square_intermediate_entities[0])];
    SsMeshError error = {0};
    memcpy(mismatched, square_intermediate_entities, sizeof(mismatched));
    mismatched[0U].stable_entity_id = UINT64_C(0xf000000000f);
    input = square_input();
    input.intermediate_entities = mismatched;
    CHECK(ss_mesh_create(PETSC_COMM_SELF, &input, &mesh, &error) == SS_ERROR_INTEGRITY);
    CHECK(mesh == NULL);
  }
  return 0;
}

static int test_labels(void) {
  SsMeshInput input = square_input();
  SsMesh *mesh = NULL;
  uint32_t value = 0U;
  uint64_t global_count = 0U;
  size_t index;
  CHECK(create_mesh(&input, &mesh) == 0);
  CHECK(ss_mesh_label_value(mesh, SS_MESH_LABEL_MATERIAL, square_cell_ids[0], &value) ==
        SS_OK);
  CHECK(value == 41U);
  CHECK(ss_mesh_label_value(mesh, SS_MESH_LABEL_MATERIAL, square_cell_ids[1], &value) ==
        SS_OK);
  CHECK(value == 42U);
  for (index = 0U; index < 4U; ++index) {
    CHECK(ss_mesh_label_value(mesh, SS_MESH_LABEL_SUPPORT, square_facet_ids[index],
                              &value) == SS_OK);
    CHECK(value == 11U + (uint32_t)index);
  }
  CHECK(ss_mesh_label_value(mesh, SS_MESH_LABEL_INTERFACE, square_facet_ids[1],
                            &value) == SS_OK);
  CHECK(value == 71U);
  CHECK(ss_mesh_label_value(mesh, SS_MESH_LABEL_INTERFACE, square_facet_ids[0],
                            &value) == SS_ERROR_NOT_FOUND);
  CHECK(ss_mesh_label_cardinality(mesh, SS_MESH_LABEL_MATERIAL, 41U) == 1U);
  CHECK(ss_mesh_label_cardinality(mesh, SS_MESH_LABEL_SUPPORT, 12U) == 1U);
  CHECK(ss_mesh_label_cardinality_global(mesh, SS_MESH_LABEL_MATERIAL, 41U,
                                         &global_count) == SS_OK);
  CHECK(global_count == 1U);
  CHECK(ss_mesh_label_cardinality_global(mesh, SS_MESH_LABEL_SUPPORT, 12U,
                                         &global_count) == SS_OK);
  CHECK(global_count == 1U);
  CHECK(ss_mesh_destroy(&mesh) == SS_OK);
  {
    SsMeshLabelAssignment incomplete[sizeof(square_labels) / sizeof(square_labels[0])];
    SsMeshError error = {0};
    input = square_input();
    memcpy(incomplete, square_labels + 1U,
           (sizeof(square_labels) / sizeof(square_labels[0]) - 1U) *
               sizeof(incomplete[0]));
    input.labels = incomplete;
    input.label_count = sizeof(square_labels) / sizeof(square_labels[0]) - 1U;
    CHECK(ss_mesh_create(PETSC_COMM_SELF, &input, &mesh, &error) == SS_ERROR_INTEGRITY);
    CHECK(mesh == NULL);
  }
  {
    SsMeshLabelAssignment incomplete[sizeof(square_labels) / sizeof(square_labels[0])];
    SsMeshError error = {0};
    input = square_input();
    memcpy(incomplete, square_labels, 5U * sizeof(incomplete[0]));
    incomplete[5] = square_labels[6];
    input.labels = incomplete;
    input.label_count = 6U;
    CHECK(ss_mesh_create(PETSC_COMM_SELF, &input, &mesh, &error) == SS_ERROR_INTEGRITY);
    CHECK(mesh == NULL);
  }
  {
    SsMeshLabelAssignment
        overlapping[sizeof(square_labels) / sizeof(square_labels[0]) + 1U];
    SsMeshError error = {0};
    int contains = 0;
    size_t facet_count = 0U;
    input = square_input();
    memcpy(overlapping, square_labels, sizeof(square_labels));
    overlapping[sizeof(square_labels) / sizeof(square_labels[0])] =
        (SsMeshLabelAssignment){
            SS_MESH_LABEL_SUPPORT,
            square_facet_ids[0],
            99U,
    };
    input.labels = overlapping;
    input.label_count = sizeof(overlapping) / sizeof(overlapping[0]);
    CHECK(ss_mesh_create(PETSC_COMM_SELF, &input, &mesh, &error) == SS_OK);
    CHECK(mesh != NULL);
    CHECK(ss_mesh_label_contains(mesh, SS_MESH_LABEL_SUPPORT, square_facet_ids[0], 11U,
                                 &contains) == SS_OK);
    CHECK(contains != 0);
    CHECK(ss_mesh_label_contains(mesh, SS_MESH_LABEL_SUPPORT, square_facet_ids[0], 99U,
                                 &contains) == SS_OK);
    CHECK(contains != 0);
    CHECK(ss_mesh_pack_facets(mesh, 11U, NULL, 0U, &facet_count) == SS_OK);
    CHECK(facet_count == 1U);
    CHECK(ss_mesh_pack_facets(mesh, 99U, NULL, 0U, &facet_count) == SS_OK);
    CHECK(facet_count == 1U);
    CHECK(ss_mesh_destroy(&mesh) == SS_OK);
  }
  return 0;
}

static int check_field_sections(const SsMeshInput *input,
                                const uint64_t expected_nodes[3]) {
  const uint32_t orders[] = {1U, 2U, 4U};
  const uint32_t components[] = {1U, 2U, 3U};
  const uint32_t expected_edge_slots[] = {0U, 1U, 3U};
  const uint32_t expected_face_slots_3d[] = {0U, 0U, 3U};
  const uint32_t expected_cell_slots_2d[] = {0U, 0U, 3U};
  const uint32_t expected_cell_slots_3d[] = {0U, 0U, 1U};
  SsMesh *mesh = NULL;
  size_t order_index;
  size_t component_index;
  CHECK(create_mesh(input, &mesh) == 0);

  for (order_index = 0U; order_index < 3U; ++order_index) {
    for (component_index = 0U; component_index < 3U; ++component_index) {
      SsFieldSpace *space = NULL;
      SsFieldSummary summary = {0};
      SsEntityId stable_ids[128] = {0U};
      size_t local_slots[128] = {0U};
      unsigned char seen_slots[128] = {0U};
      const SsFeDescriptor *descriptor = NULL;
      size_t free_dof_count = 0U;
      size_t free_dof;
      int contains_wide_stable_id = 0;
      const SsFieldSpec spec = {
          orders[order_index],
          components[component_index],
          0U,
          NULL,
      };
      CHECK(ss_mesh_field_space_create(mesh, &spec, &space) == SS_OK);
      CHECK(ss_fe_descriptor_find(input->dimension, orders[order_index], &descriptor) ==
            SS_OK);
      CHECK(space != NULL);
      CHECK(ss_field_space_summary(space, &summary) == SS_OK);
      CHECK(summary.order == orders[order_index]);
      CHECK(summary.components == components[component_index]);
      CHECK(summary.vertex_slots == 1U);
      CHECK(summary.edge_interior_slots == expected_edge_slots[order_index]);
      CHECK(summary.face_interior_slots ==
            (input->dimension == 3U ? expected_face_slots_3d[order_index] : 0U));
      CHECK(summary.cell_interior_slots == (input->dimension == 2U
                                                ? expected_cell_slots_2d[order_index]
                                                : expected_cell_slots_3d[order_index]));
      CHECK(summary.global_dof_count ==
            expected_nodes[order_index] * (uint64_t)components[component_index]);
      CHECK(summary.global_free_dof_count == summary.global_dof_count);
      CHECK(summary.owned_free_dof_count == summary.global_dof_count);
      CHECK(summary.constrained_dof_count == 0U);
      CHECK(summary.owned_cell_plan_count == input->global_cell_count);
      CHECK(summary.closure_plan_bytes ==
            summary.owned_cell_plan_count * (uint64_t)sizeof(SsCellPlan));
      CHECK(summary.incident_closure_plan_bytes >=
            summary.owned_cell_plan_count * (uint64_t)sizeof(SsCellPlan));
      CHECK(summary.incident_closure_plan_bytes <
            summary.owned_cell_plan_count * (uint64_t)sizeof(SsCellPack));
      CHECK(summary.section_generation == 1U);
      CHECK(summary.constraint_generation == 1U);
      CHECK(strcmp(summary.basis_id, descriptor->id) == 0);
      CHECK(summary.identity_digest[0] != '\0');
      CHECK(ss_field_space_owned_free_dof_map(space, NULL, NULL, 0U, &free_dof_count) ==
            SS_OK);
      CHECK(free_dof_count == (size_t)summary.owned_free_dof_count);
      CHECK(free_dof_count <= sizeof(stable_ids) / sizeof(stable_ids[0]));
      CHECK(ss_field_space_owned_free_dof_map(
                space, stable_ids, local_slots,
                sizeof(stable_ids) / sizeof(stable_ids[0]), &free_dof_count) == SS_OK);
      for (free_dof = 0U; free_dof < free_dof_count; ++free_dof) {
        size_t earlier;
        CHECK(stable_ids[free_dof] != 0U);
        CHECK(local_slots[free_dof] < free_dof_count);
        CHECK(seen_slots[local_slots[free_dof]] == 0U);
        seen_slots[local_slots[free_dof]] = 1U;
        contains_wide_stable_id =
            contains_wide_stable_id || stable_ids[free_dof] > (SsEntityId)INT32_MAX;
        for (earlier = 0U; earlier < free_dof; ++earlier) {
          CHECK(stable_ids[free_dof] != stable_ids[earlier]);
        }
      }
      CHECK(contains_wide_stable_id != 0);
      CHECK(ss_field_space_destroy(&space) == SS_OK && space == NULL);
    }
  }
  CHECK(ss_mesh_destroy(&mesh) == SS_OK);
  return 0;
}

static int test_sections(void) {
  const uint64_t triangle_nodes[] = {4U, 9U, 25U};
  const uint64_t tetrahedron_nodes[] = {4U, 10U, 35U};
  SsMeshInput triangle = square_input();
  SsMeshInput tetrahedron = tetra_input();
  CHECK(check_field_sections(&triangle, triangle_nodes) == 0);
  CHECK(check_field_sections(&tetrahedron, tetrahedron_nodes) == 0);
  {
    const SsFieldSpec spec = {2U, 1U, 0U, NULL};
    const uint64_t alternate_cells[] = {
        0U, 1U, 3U, 0U, 3U, 2U,
    };
    const SsMeshIntermediateEntity alternate_intermediate_entities[] = {
        {1U,
         UINT64_C(0x90000000009),
         {UINT64_C(0x10000000001), UINT64_C(0x20000000002), 0U}},
        {1U,
         UINT64_C(0xc000000000c),
         {UINT64_C(0x10000000001), UINT64_C(0x30000000003), 0U}},
        {1U,
         UINT64_C(0xe000000000e),
         {UINT64_C(0x10000000001), UINT64_C(0x40000000004), 0U}},
        {1U,
         UINT64_C(0xa000000000a),
         {UINT64_C(0x20000000002), UINT64_C(0x40000000004), 0U}},
        {1U,
         UINT64_C(0xb000000000b),
         {UINT64_C(0x30000000003), UINT64_C(0x40000000004), 0U}},
    };
    SsMesh *first_mesh = NULL;
    SsMesh *second_mesh = NULL;
    SsFieldSpace *first_space = NULL;
    SsFieldSpace *second_space = NULL;
    SsFieldSummary first_summary = {0};
    SsFieldSummary second_summary = {0};
    SsMeshSummary first_mesh_summary = {0};
    SsMeshSummary second_mesh_summary = {0};
    SsMeshInput first_input = square_input();
    SsMeshInput second_input = square_input();
    second_input.cells = alternate_cells;
    second_input.cell_coordinate_nodes = alternate_cells;
    second_input.intermediate_entities = alternate_intermediate_entities;
    CHECK(create_mesh(&first_input, &first_mesh) == 0);
    CHECK(create_mesh(&second_input, &second_mesh) == 0);
    CHECK(ss_mesh_summary(first_mesh, &first_mesh_summary) == SS_OK);
    CHECK(ss_mesh_summary(second_mesh, &second_mesh_summary) == SS_OK);
    CHECK(strcmp(first_mesh_summary.topology_digest,
                 second_mesh_summary.topology_digest) != 0);
    CHECK(ss_mesh_field_space_create(first_mesh, &spec, &first_space) == SS_OK);
    CHECK(ss_mesh_field_space_create(second_mesh, &spec, &second_space) == SS_OK);
    CHECK(ss_field_space_summary(first_space, &first_summary) == SS_OK);
    CHECK(ss_field_space_summary(second_space, &second_summary) == SS_OK);
    CHECK(strcmp(first_summary.identity_digest, second_summary.identity_digest) != 0);
    CHECK(ss_mesh_renumber(first_mesh, UINT64_C(17)) == SS_ERROR_INVALID_STATE);
    CHECK(ss_mesh_destroy(&first_mesh) == SS_ERROR_INVALID_STATE);
    CHECK(first_mesh != NULL);
    CHECK(ss_field_space_destroy(&first_space) == SS_OK);
    CHECK(ss_field_space_destroy(&second_space) == SS_OK);
    CHECK(ss_mesh_renumber(first_mesh, UINT64_C(17)) == SS_OK);
    CHECK(ss_mesh_destroy(&first_mesh) == SS_OK);
    CHECK(ss_mesh_destroy(&second_mesh) == SS_OK);
  }
  return 0;
}

static int test_cell_packs(void) {
  const SsFieldSpec spec = {4U, 1U, 0U, NULL};
  SsMeshInput input = square_input();
  SsMesh *mesh = NULL;
  SsFieldSpace *space = NULL;
  SsCellPack packs[2] = {{0}};
  size_t count = 0U;
  size_t cell;
  size_t shared = 0U;
  CHECK(create_mesh(&input, &mesh) == 0);
  CHECK(ss_mesh_field_space_create(mesh, &spec, &space) == SS_OK);
  CHECK(ss_field_space_pack_owned_cells(space, NULL, 0U, &count) == SS_OK);
  CHECK(count == 2U);
  CHECK(ss_field_space_pack_owned_cells(space, packs, 2U, &count) == SS_OK);
  CHECK(count == 2U);
  CHECK(packs[0].stable_cell_id == square_cell_ids[0]);
  CHECK(packs[1].stable_cell_id == square_cell_ids[1]);
  for (cell = 0U; cell < count; ++cell) {
    size_t node;
    CHECK(packs[cell].dimension == 2U);
    CHECK(packs[cell].order == 4U);
    CHECK(packs[cell].components == 1U);
    CHECK(packs[cell].material_value == 41U + (uint32_t)cell);
    CHECK(packs[cell].coordinate_order == 1U);
    CHECK(packs[cell].coordinate_node_count == 3U);
    CHECK(packs[cell].field_node_count == 15U);
    CHECK(packs[cell].closure_count == 15U);
    for (node = 0U; node < packs[cell].closure_count; ++node) {
      CHECK(packs[cell].field_entity_ids[node] != 0U);
      CHECK(packs[cell].field_components[node] == 0U);
      CHECK(packs[cell].local_indices[node] >= 0);
      CHECK(packs[cell].global_indices[node] >= 0);
      CHECK(packs[cell].constrained[node] == 0U);
      CHECK(isfinite(packs[cell].field_node_coordinates[node * 2U]));
      CHECK(isfinite(packs[cell].field_node_coordinates[node * 2U + 1U]));
    }
  }
  {
    SsCellPack refreshed[2] = {{0}};
    packs[0].stable_cell_id = 0U;
    packs[0].field_entity_ids[0] = 0U;
    CHECK(ss_field_space_pack_owned_cells(space, refreshed, 2U, &count) == SS_OK);
    CHECK(refreshed[0].stable_cell_id == square_cell_ids[0]);
    CHECK(refreshed[0].field_entity_ids[0] != 0U);
  }
  {
    size_t left;
    for (left = 0U; left < packs[0].field_node_count; ++left) {
      size_t right;
      for (right = 0U; right < packs[1].field_node_count; ++right) {
        if (packs[0].field_entity_ids[left] == packs[1].field_entity_ids[right] &&
            packs[0].field_entity_slots[left] == packs[1].field_entity_slots[right]) {
          CHECK(packs[0].global_indices[left] == packs[1].global_indices[right]);
          CHECK_NEAR(packs[0].field_node_coordinates[left * 2U],
                     packs[1].field_node_coordinates[right * 2U], 2.0e-13);
          CHECK_NEAR(packs[0].field_node_coordinates[left * 2U + 1U],
                     packs[1].field_node_coordinates[right * 2U + 1U], 2.0e-13);
          ++shared;
        }
      }
    }
  }
  CHECK(shared == 5U);
  CHECK(ss_field_space_destroy(&space) == SS_OK);
  CHECK(ss_mesh_destroy(&mesh) == SS_OK);
  return 0;
}

static int test_lifting(void) {
  const SsStrongConstraint constraints[] = {
      {square_vertex_ids[0], 0U, 0U, 2.0},
      {square_vertex_ids[2], 0U, 0U, -1.0},
  };
  const SsFieldSpec spec = {
      1U,
      1U,
      sizeof(constraints) / sizeof(constraints[0]),
      constraints,
  };
  const double matrix[] = {
      6.0, -2.0, 0.5, 1.0, 4.0, 2.0, -0.5, 3.0, 5.0,
  };
  const double rhs[] = {7.0, 5.0, -2.0};
  const double trial[] = {19.0, 20.0, 21.0};
  double complete[3] = {0.0, 0.0, 0.0};
  size_t free_slots[3] = {SIZE_MAX, SIZE_MAX, SIZE_MAX};
  double free_rhs[3] = {0.0, 0.0, 0.0};
  size_t free_count = 0U;
  SsMeshInput input = square_input();
  SsMesh *mesh = NULL;
  SsFieldSpace *space = NULL;
  SsFieldSummary summary = {0};

  CHECK(create_mesh(&input, &mesh) == 0);
  CHECK(ss_mesh_field_space_create(mesh, &spec, &space) == SS_OK);
  CHECK(ss_field_space_summary(space, &summary) == SS_OK);
  CHECK(summary.constrained_dof_count == 2U);
  CHECK(summary.global_dof_count == 4U);
  CHECK(summary.global_free_dof_count == 2U);
  CHECK(summary.owned_free_dof_count == 2U);
  CHECK(ss_field_space_apply_cell_lifting(space, square_cell_ids[0], matrix, rhs, trial,
                                          3U, complete, free_slots, free_rhs,
                                          &free_count) == SS_OK);
  CHECK_NEAR(complete[0], 2.0, 0.0);
  CHECK_NEAR(complete[1], 20.0, 0.0);
  CHECK_NEAR(complete[2], -1.0, 0.0);
  CHECK(free_count == 1U && free_slots[0] == 1U);
  CHECK_NEAR(free_rhs[0], 5.0 - matrix[3U] * 2.0 + matrix[5U], 2.0e-14);

  CHECK(ss_field_space_destroy(&space) == SS_OK);
  {
    const SsStrongConstraint reversed_constraints[] = {
        constraints[1],
        constraints[0],
    };
    const SsFieldSpec reversed_spec = {
        1U,
        1U,
        sizeof(reversed_constraints) / sizeof(reversed_constraints[0]),
        reversed_constraints,
    };
    SsFieldSummary reversed_summary = {0};
    CHECK(ss_mesh_field_space_create(mesh, &reversed_spec, &space) == SS_OK);
    CHECK(ss_field_space_summary(space, &reversed_summary) == SS_OK);
    CHECK(strcmp(summary.identity_digest, reversed_summary.identity_digest) == 0);
    CHECK(ss_field_space_destroy(&space) == SS_OK);
  }
  {
    const SsStrongConstraint duplicate_constraints[] = {
        {square_vertex_ids[0], 0U, 0U, 2.0},
        {square_vertex_ids[0], 0U, 0U, 3.0},
    };
    const SsFieldSpec duplicate_spec = {
        1U,
        1U,
        sizeof(duplicate_constraints) / sizeof(duplicate_constraints[0]),
        duplicate_constraints,
    };
    CHECK(ss_mesh_field_space_create(mesh, &duplicate_spec, &space) ==
          SS_ERROR_DUPLICATE);
    CHECK(space == NULL);
  }
  {
    const SsStrongConstraint missing_constraint = {
        UINT64_C(0xfffffffffffffff0),
        0U,
        0U,
        1.0,
    };
    const SsFieldSpec missing_spec = {1U, 1U, 1U, &missing_constraint};
    CHECK(ss_mesh_field_space_create(mesh, &missing_spec, &space) ==
          SS_ERROR_NOT_FOUND);
    CHECK(space == NULL);
  }
  {
    SsStrongConstraint many_constraints[66];
    SsFieldSummary many_summary = {0};
    size_t count = 0U;
    size_t entity;
    uint32_t slot;
    uint32_t component;
    for (entity = 0U; entity < 4U; ++entity) {
      for (component = 0U; component < 3U; ++component) {
        many_constraints[count] = (SsStrongConstraint){
            square_vertex_ids[entity],
            0U,
            component,
            0.01 * (double)(count + 1U),
        };
        ++count;
      }
    }
    for (entity = 0U; entity < 4U; ++entity) {
      for (slot = 0U; slot < 3U; ++slot) {
        for (component = 0U; component < 3U; ++component) {
          many_constraints[count] = (SsStrongConstraint){
              square_facet_ids[entity],
              slot,
              component,
              0.01 * (double)(count + 1U),
          };
          ++count;
        }
      }
    }
    for (entity = 0U; entity < 2U; ++entity) {
      for (slot = 0U; slot < 3U; ++slot) {
        for (component = 0U; component < 3U; ++component) {
          many_constraints[count] = (SsStrongConstraint){
              square_cell_ids[entity],
              slot,
              component,
              0.01 * (double)(count + 1U),
          };
          ++count;
        }
      }
    }
    CHECK(count == sizeof(many_constraints) / sizeof(many_constraints[0]));
    {
      const SsFieldSpec many_spec = {
          4U,
          3U,
          count,
          many_constraints,
      };
      CHECK(ss_mesh_field_space_create(mesh, &many_spec, &space) == SS_OK);
      CHECK(ss_field_space_summary(space, &many_summary) == SS_OK);
      CHECK(many_summary.constrained_dof_count == count);
      CHECK(many_summary.global_dof_count == 75U);
      CHECK(many_summary.global_free_dof_count == 9U);
      CHECK(many_summary.owned_free_dof_count == 9U);
      CHECK(strncmp(many_summary.identity_digest, "sha256:", 7U) == 0);
      CHECK(ss_field_space_destroy(&space) == SS_OK);
    }
    {
      const SsFieldSpec overflowing_spec = {
          4U,
          3U,
          SIZE_MAX / sizeof(many_constraints[0]) + 1U,
          many_constraints,
      };
      CHECK(ss_mesh_field_space_create(mesh, &overflowing_spec, &space) ==
            SS_ERROR_OVERFLOW);
      CHECK(space == NULL);
    }
  }
  CHECK(ss_mesh_destroy(&mesh) == SS_OK);
  return 0;
}

static int test_facets(void) {
  static const SsEntityId expected_parent_ids[] = {
      UINT64_C(0x70000000007),
      UINT64_C(0x80000000008),
      UINT64_C(0x80000000008),
      UINT64_C(0x70000000007),
  };
  SsMeshInput input = square_input();
  SsMesh *mesh = NULL;
  size_t support_index;
  CHECK(create_mesh(&input, &mesh) == 0);
  for (support_index = 0U; support_index < 4U; ++support_index) {
    SsFacetPack pack[2] = {{0}};
    size_t count = 0U;
    CHECK(ss_mesh_pack_facets(mesh, 11U + (uint32_t)support_index, NULL, 0U, &count) ==
          SS_OK);
    CHECK(count == 1U);
    CHECK(ss_mesh_pack_facets(mesh, 11U + (uint32_t)support_index, pack, 2U, &count) ==
          SS_OK);
    CHECK(count == 1U);
    CHECK(pack[0].stable_facet_id == square_facet_ids[support_index]);
    CHECK(pack[0].stable_parent_cell_id == expected_parent_ids[support_index]);
    CHECK(pack[0].support_value == 11U + (uint32_t)support_index);
    CHECK(pack[0].vertex_count == 2U);
    CHECK(pack[0].orientation == 1 || pack[0].orientation == -1);
    CHECK(pack[0].outward_dot > 0.0);
    CHECK_NEAR(hypot(pack[0].outward_normal[0], pack[0].outward_normal[1]), 1.0,
               2.0e-14);
    {
      SsFacetPack refreshed[1] = {{0}};
      pack[0].stable_facet_id = 0U;
      pack[0].outward_normal[0] = NAN;
      CHECK(ss_mesh_pack_facets(mesh, 11U + (uint32_t)support_index, refreshed, 1U,
                                &count) == SS_OK);
      CHECK(refreshed[0].stable_facet_id == square_facet_ids[support_index]);
      CHECK(isfinite(refreshed[0].outward_normal[0]));
    }
  }
  CHECK(ss_mesh_pack_facets(mesh, 999U, NULL, 0U, &support_index) == SS_OK);
  CHECK(support_index == 0U);
  {
    const SsFieldSpec spec = {4U, 2U, 0U, NULL};
    SsFieldSpace *space = NULL;
    SsFieldFacetPack field_pack[1] = {0};
    size_t count = 0U;
    size_t closure;
    CHECK(ss_mesh_field_space_create(mesh, &spec, &space) == SS_OK);
    CHECK(ss_field_space_pack_owned_facets(space, 11U, NULL, 0U, &count) == SS_OK);
    CHECK(count == 1U);
    CHECK(ss_field_space_pack_owned_facets(space, 11U, field_pack, 1U, &count) ==
          SS_OK);
    CHECK(count == 1U);
    CHECK(field_pack[0].geometry.stable_facet_id == square_facet_ids[0]);
    CHECK(field_pack[0].geometry.stable_parent_cell_id == square_cell_ids[0]);
    CHECK(field_pack[0].order == 4U);
    CHECK(field_pack[0].components == 2U);
    CHECK(field_pack[0].field_node_count == 5U);
    CHECK(field_pack[0].closure_count == 10U);
    for (closure = 0U; closure < field_pack[0].closure_count; ++closure) {
      CHECK(field_pack[0].parent_closure_slots[closure] < 30U);
      CHECK(field_pack[0].field_entity_ids[closure] != 0U);
      CHECK(field_pack[0].local_indices[closure] >= 0);
      CHECK(field_pack[0].global_indices[closure] >= 0);
    }
    for (closure = 0U; closure < field_pack[0].field_node_count; ++closure) {
      CHECK_NEAR(field_pack[0].field_node_coordinates[closure * 2U + 1U], 0.0, 3.0e-13);
    }
    CHECK(ss_field_space_destroy(&space) == SS_OK);
  }
  CHECK(ss_mesh_destroy(&mesh) == SS_OK);
  {
    const SsFieldSpec spec = {4U, 1U, 0U, NULL};
    SsFieldSpace *space = NULL;
    SsFieldFacetPack field_pack[1] = {0};
    size_t count = 0U;
    input = tetra_input();
    CHECK(create_mesh(&input, &mesh) == 0);
    CHECK(ss_mesh_field_space_create(mesh, &spec, &space) == SS_OK);
    CHECK(ss_field_space_pack_owned_facets(space, 21U, field_pack, 1U, &count) ==
          SS_OK);
    CHECK(count == 1U);
    CHECK(field_pack[0].field_node_count == 15U);
    CHECK(field_pack[0].closure_count == 15U);
    CHECK(ss_field_space_destroy(&space) == SS_OK);
    CHECK(ss_mesh_destroy(&mesh) == SS_OK);
  }

  {
    const SsFeDescriptor *descriptor = NULL;
    const SsEntityId vertex_ids[] = {UINT64_C(1101), UINT64_C(1102), UINT64_C(1103)};
    const double vertex_coordinates[] = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    const SsEntityId cell_ids[] = {UINT64_C(1201)};
    const uint64_t cells[] = {0U, 1U, 2U};
    const SsEntityId facet_ids[] = {UINT64_C(1301), UINT64_C(1302), UINT64_C(1303)};
    const uint64_t facets[] = {0U, 1U, 1U, 2U, 0U, 2U};
    const SsMeshLabelAssignment labels[] = {
        {SS_MESH_LABEL_MATERIAL, UINT64_C(1201), 91U},
        {SS_MESH_LABEL_SUPPORT, UINT64_C(1301), 81U},
        {SS_MESH_LABEL_SUPPORT, UINT64_C(1302), 82U},
        {SS_MESH_LABEL_SUPPORT, UINT64_C(1303), 83U},
    };
    const SsMeshIntermediateEntity intermediate_entities[] = {
        {1U, UINT64_C(1301), {UINT64_C(1101), UINT64_C(1102), 0U}},
        {1U, UINT64_C(1303), {UINT64_C(1101), UINT64_C(1103), 0U}},
        {1U, UINT64_C(1302), {UINT64_C(1102), UINT64_C(1103), 0U}},
    };
    SsEntityId coordinate_node_ids[SS_FE_MAX_NODES] = {0U};
    double coordinate_coordinates[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
    uint64_t cell_coordinate_nodes[SS_FE_MAX_NODES] = {0U};
    SsMeshInput curved_input = {0};
    SsFacetPack pack[1] = {{0}};
    SsFacetSample sample = {0};
    const double eta[2] = {0.25, 0.0};
    size_t node;
    size_t count = 0U;

    CHECK(ss_fe_descriptor_find(2U, 4U, &descriptor) == SS_OK);
    for (node = 0U; node < descriptor->node_count; ++node) {
      const double r = (double)descriptor->barycentric_indices[node * 3U + 1U] / 4.0;
      const double s = (double)descriptor->barycentric_indices[node * 3U + 2U] / 4.0;
      coordinate_node_ids[node] =
          node < 3U ? vertex_ids[node] : UINT64_C(1400) + (SsEntityId)node;
      coordinate_coordinates[node * 2U] = r;
      coordinate_coordinates[node * 2U + 1U] = s * (1.0 + 0.3 * r);
      cell_coordinate_nodes[node] = (uint64_t)node;
    }
    curved_input = (SsMeshInput){
        .dimension = 2U,
        .global_vertex_count = 3U,
        .global_cell_count = 1U,
        .global_boundary_facet_count = 3U,
        .global_intermediate_entity_count =
            sizeof(intermediate_entities) / sizeof(intermediate_entities[0]),
        .global_coordinate_node_count = descriptor->node_count,
        .vertex_count = 3U,
        .cell_count = 1U,
        .boundary_facet_count = 3U,
        .vertex_ids = vertex_ids,
        .vertex_coordinates = vertex_coordinates,
        .cell_ids = cell_ids,
        .cells = cells,
        .boundary_facet_ids = facet_ids,
        .boundary_facets = facets,
        .intermediate_entity_count =
            sizeof(intermediate_entities) / sizeof(intermediate_entities[0]),
        .intermediate_entities = intermediate_entities,
        .label_count = sizeof(labels) / sizeof(labels[0]),
        .labels = labels,
        .coordinate_order = 4U,
        .coordinate_node_count = descriptor->node_count,
        .coordinate_node_ids = coordinate_node_ids,
        .coordinate_coordinates = coordinate_coordinates,
        .cell_coordinate_nodes = cell_coordinate_nodes,
        .absolute_determinant_tolerance = 1.0e-12,
        .relative_determinant_tolerance = 1.0e-12,
    };
    CHECK(create_mesh(&curved_input, &mesh) == 0);
    CHECK(ss_mesh_pack_facets(mesh, 82U, pack, 1U, &count) == SS_OK);
    CHECK(count == 1U);
    CHECK(pack[0].coordinate_order == 4U);
    CHECK(pack[0].coordinate_node_count == descriptor->node_count);
    CHECK(pack[0].facet_index == 1U);
    CHECK(ss_mesh_facet_geometry_at(&pack[0], eta, &sample) == SS_OK);
    CHECK_NEAR(sample.physical[0], 0.75, 3.0e-12);
    CHECK_NEAR(sample.physical[1], 0.30625, 3.0e-12);
    CHECK_NEAR(sample.measure, sqrt(2.3225), 3.0e-12);
    CHECK_NEAR(sample.outward_normal[0], 1.15 / sqrt(2.3225), 3.0e-12);
    CHECK_NEAR(sample.outward_normal[1], 1.0 / sqrt(2.3225), 3.0e-12);
    CHECK(ss_mesh_destroy(&mesh) == SS_OK);
  }
  return 0;
}

/*
 * MSH-UNIT-CURVED-FACET-001: a valid, strongly curved G4 triangle whose
 * boundary-facet normal at the facet centroid points towards the parent
 * centroid.  The polynomial map x = r, y = s + (1 - r - s) q(r) has
 * det J = 1 - q(r) >= 0.04 everywhere, so the cell is admissible and the
 * topological outward normal must be kept; the centroid alignment is only a
 * diagnostic.
 */
static double curved_facet_edge(double r) {
  const double u = r - 0.5;
  return -3.84 * u + 15.36 * u * u * u;
}

static int test_curved_facet_orientation(void) {
  const SsFeDescriptor *descriptor = NULL;
  const SsEntityId vertex_ids[] = {UINT64_C(2101), UINT64_C(2102), UINT64_C(2103)};
  const double vertex_coordinates[] = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
  const SsEntityId cell_ids[] = {UINT64_C(2201)};
  const uint64_t cells[] = {0U, 1U, 2U};
  const SsEntityId facet_ids[] = {UINT64_C(2301), UINT64_C(2302), UINT64_C(2303)};
  const uint64_t facets[] = {0U, 1U, 1U, 2U, 0U, 2U};
  const SsMeshLabelAssignment labels[] = {
      {SS_MESH_LABEL_MATERIAL, UINT64_C(2201), 91U},
      {SS_MESH_LABEL_SUPPORT, UINT64_C(2301), 81U},
      {SS_MESH_LABEL_SUPPORT, UINT64_C(2302), 82U},
      {SS_MESH_LABEL_SUPPORT, UINT64_C(2303), 83U},
  };
  const SsMeshIntermediateEntity intermediate_entities[] = {
      {1U, UINT64_C(2301), {UINT64_C(2101), UINT64_C(2102), 0U}},
      {1U, UINT64_C(2303), {UINT64_C(2101), UINT64_C(2103), 0U}},
      {1U, UINT64_C(2302), {UINT64_C(2102), UINT64_C(2103), 0U}},
  };
  SsEntityId coordinate_node_ids[SS_FE_MAX_NODES] = {0U};
  double coordinate_coordinates[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
  uint64_t cell_coordinate_nodes[SS_FE_MAX_NODES] = {0U};
  SsMeshInput input = {0};
  SsMesh *mesh = NULL;
  SsFacetPack pack[1] = {{0}};
  const double slope = -3.84;
  const double normal_x = slope / sqrt(slope * slope + 1.0);
  const double normal_y = -1.0 / sqrt(slope * slope + 1.0);
  size_t node;
  size_t count = 0U;
  CHECK(ss_fe_descriptor_find(2U, 4U, &descriptor) == SS_OK);
  for (node = 0U; node < descriptor->node_count; ++node) {
    const double r = (double)descriptor->barycentric_indices[node * 3U + 1U] / 4.0;
    const double s = (double)descriptor->barycentric_indices[node * 3U + 2U] / 4.0;
    const double weight = r < 1.0 ? (1.0 - r - s) / (1.0 - r) : 0.0;
    coordinate_node_ids[node] =
        node < 3U ? vertex_ids[node] : UINT64_C(2400) + (SsEntityId)node;
    coordinate_coordinates[node * 2U] = r;
    coordinate_coordinates[node * 2U + 1U] = s + weight * curved_facet_edge(r);
    cell_coordinate_nodes[node] = (uint64_t)node;
  }
  input = (SsMeshInput){
      .dimension = 2U,
      .global_vertex_count = 3U,
      .global_cell_count = 1U,
      .global_boundary_facet_count = 3U,
      .global_intermediate_entity_count =
          sizeof(intermediate_entities) / sizeof(intermediate_entities[0]),
      .global_coordinate_node_count = descriptor->node_count,
      .vertex_count = 3U,
      .cell_count = 1U,
      .boundary_facet_count = 3U,
      .vertex_ids = vertex_ids,
      .vertex_coordinates = vertex_coordinates,
      .cell_ids = cell_ids,
      .cells = cells,
      .boundary_facet_ids = facet_ids,
      .boundary_facets = facets,
      .intermediate_entity_count =
          sizeof(intermediate_entities) / sizeof(intermediate_entities[0]),
      .intermediate_entities = intermediate_entities,
      .label_count = sizeof(labels) / sizeof(labels[0]),
      .labels = labels,
      .coordinate_order = 4U,
      .coordinate_node_count = descriptor->node_count,
      .coordinate_node_ids = coordinate_node_ids,
      .coordinate_coordinates = coordinate_coordinates,
      .cell_coordinate_nodes = cell_coordinate_nodes,
      .absolute_determinant_tolerance = 1.0e-12,
      .relative_determinant_tolerance = 1.0e-12,
  };
  CHECK(create_mesh(&input, &mesh) == 0);
  CHECK(ss_mesh_pack_facets(mesh, 81U, pack, 1U, &count) == SS_OK);
  CHECK(count == 1U);
  CHECK(pack[0].facet_index == 0U);
  CHECK(pack[0].orientation == 1);
  CHECK_NEAR(pack[0].centroid[0], 0.5, 1.0e-12);
  CHECK_NEAR(pack[0].centroid[1], 0.0, 1.0e-12);
  CHECK_NEAR(pack[0].outward_normal[0], normal_x, 1.0e-12);
  CHECK_NEAR(pack[0].outward_normal[1], normal_y, 1.0e-12);
  /* The diagnostic records the misaligned centroid instead of rejecting it. */
  CHECK(pack[0].outward_dot < 0.0);
  CHECK(ss_mesh_destroy(&mesh) == SS_OK);
  return 0;
}

int main(int argc, char **argv) {
  int result = 0;
  const char *test_id;
  if (argc != 2) {
    fputs("usage: ss_test_mesh TEST-ID\n", stderr);
    return 2;
  }
  test_id = argv[1];
  if (PetscInitialize(&argc, &argv, NULL, NULL) != PETSC_SUCCESS) {
    return 3;
  }
  if (strcmp(test_id, "MSH-UNIT-IDMAP-001") == 0) {
    result = test_id_map();
  } else if (strcmp(test_id, "MSH-UNIT-LABEL-001") == 0) {
    result = test_labels();
  } else if (strcmp(test_id, "MSH-UNIT-SECTION-001") == 0) {
    result = test_sections();
    if (result == 0) {
      result = test_cell_packs();
    }
  } else if (strcmp(test_id, "MSH-UNIT-LIFTING-001") == 0) {
    result = test_lifting();
  } else if (strcmp(test_id, "MSH-UNIT-FACET-001") == 0) {
    result = test_facets();
  } else if (strcmp(test_id, "MSH-UNIT-CURVED-FACET-001") == 0) {
    result = test_curved_facet_orientation();
  } else {
    fprintf(stderr, "unknown mesh test: %s\n", test_id);
    result = 2;
  }
  if (PetscFinalize() != PETSC_SUCCESS && result == 0) {
    result = 4;
  }
  return result;
}
