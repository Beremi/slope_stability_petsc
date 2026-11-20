/* Boundary trace of the strong pore-pressure constraints on a solver graph.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "solver_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check_case(SsSolverGraph *graph, size_t expected, uint32_t order,
                      const SsEntityId *allowed, size_t allowed_count) {
  SsStrongConstraint *constraints = NULL;
  size_t count = 0U, i, j;
  int failed = 0;
  if (ss_solver_build_constraints_for_field(graph, "pore_pressure", order, 1U,
                                            &constraints, &count) != SS_OK)
    return 1;
  if (count != expected) {
    fprintf(stderr, "%uD P%u: expected %zu constraints, got %zu\n",
            graph->model.dimension, order, expected, count);
    failed = 1;
  }
  for (i = 0U; i < count; ++i) {
    for (j = 0U; j < allowed_count; ++j)
      if (allowed[j] == constraints[i].stable_entity_id)
        break;
    if (j == allowed_count || constraints[i].component != 0U)
      failed = 1;
  }
  /* Rule evaluation must use the same support as constraint construction. */
  for (uint32_t d = 0U; d < graph->model.dimension; ++d) {
    const SsSolverEntityTable *table = &graph->raw.entities[d];
    for (i = 0U; i < table->count; ++i) {
      int expected_member = 0;
      for (j = 0U; j < allowed_count; ++j)
        if (allowed[j] == table->ids[i])
          expected_member = 1;
      if (ss_solver_rule_contains_entity(graph, &graph->model.boundaries[0],
                                         table->ids[i]) != expected_member)
        failed = 1;
    }
  }
  free(constraints);
  return failed;
}

static int test_square(void) {
  SsSolverGraph graph = {0};
  SsSolverBoundaryRule rule = {0};
  int failed = 0;
  SsEntityId ids0[] = {1, 2, 3, 4};
  SsEntityId vertices0[] = {1, 2, 3, 4};
  graph.raw.entities[0] = (SsSolverEntityTable){4U, 1U, ids0, vertices0, NULL};
  SsEntityId ids1[] = {101, 102, 103, 104, 105};
  SsEntityId vertices1[] = {1, 2, 1, 3, 1, 4, 2, 3, 3, 4};
  graph.raw.entities[1] = (SsSolverEntityTable){5U, 2U, ids1, vertices1, NULL};
  SsEntityId ids2[] = {201, 202};
  SsEntityId vertices2[] = {1, 2, 3, 1, 3, 4};
  graph.raw.entities[2] = (SsSolverEntityTable){2U, 3U, ids2, vertices2, NULL};
  SsEntityId boundary_ids[] = {101, 103, 104, 105};
  uint64_t boundary_vertices[] = {0, 1, 0, 3, 1, 2, 2, 3};
  graph.model.dimension = 2U;
  graph.raw.vertex_ids = ids0;
  graph.raw.input.global_vertex_count = 4U;
  graph.raw.input.global_boundary_facet_count = 4U;
  graph.raw.boundary_facet_ids = boundary_ids;
  graph.raw.boundary_facets = boundary_vertices;
  graph.model.boundaries = &rule;
  graph.model.boundary_count = 1U;
  if (ss_solver_mark_exterior_entities(&graph.raw, graph.model.dimension) != SS_OK)
    return 1;
  strcpy(rule.field, "pore_pressure");
  rule.condition_kind = SS_SOLVER_BOUNDARY_ESSENTIAL;
  rule.value_kind = SS_SOLVER_VALUE_CONSTANT;
  rule.component_mask = 1U;
  SsEntityId allowed[] = {1, 2, 3, 4, 101, 103, 104, 105};
  rule.entity_ids = ids0;
  rule.entity_count = 4U;
  failed |= check_case(&graph, 4U, 1U, allowed, 8U);
  failed |= check_case(&graph, 8U, 2U, allowed, 8U);
  failed |= check_case(&graph, 16U, 4U, allowed, 8U);
  for (uint32_t d = 1U; d < graph.model.dimension; ++d)
    free(graph.raw.entities[d].exterior);
  return failed;
}

static int test_octahedron(void) {
  SsSolverGraph graph = {0};
  SsSolverBoundaryRule rule = {0};
  int failed = 0;
  SsEntityId ids0[] = {1, 2, 3, 4, 5, 6};
  SsEntityId vertices0[] = {1, 2, 3, 4, 5, 6};
  graph.raw.entities[0] = (SsSolverEntityTable){6U, 1U, ids0, vertices0, NULL};
  SsEntityId ids1[] = {101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113};
  SsEntityId vertices1[] = {1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 2, 3, 2,
                            4, 2, 5, 2, 6, 3, 4, 3, 6, 4, 5, 5, 6};
  graph.raw.entities[1] = (SsSolverEntityTable){13U, 2U, ids1, vertices1, NULL};
  SsEntityId ids2[] = {201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212};
  SsEntityId vertices2[] = {1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2, 6, 1, 3, 4, 1, 3, 6,
                            1, 4, 5, 1, 5, 6, 2, 3, 4, 2, 3, 6, 2, 4, 5, 2, 5, 6};
  graph.raw.entities[2] = (SsSolverEntityTable){12U, 3U, ids2, vertices2, NULL};
  SsEntityId ids3[] = {301, 302, 303, 304};
  SsEntityId vertices3[] = {1, 2, 3, 4, 1, 2, 3, 6, 1, 2, 4, 5, 1, 2, 5, 6};
  graph.raw.entities[3] = (SsSolverEntityTable){4U, 4U, ids3, vertices3, NULL};
  SsEntityId boundary_ids[] = {205, 206, 207, 208, 209, 210, 211, 212};
  uint64_t boundary_vertices[] = {0, 2, 3, 0, 2, 5, 0, 3, 4, 0, 4, 5,
                                  1, 2, 3, 1, 2, 5, 1, 3, 4, 1, 4, 5};
  graph.model.dimension = 3U;
  graph.raw.vertex_ids = ids0;
  graph.raw.input.global_vertex_count = 6U;
  graph.raw.input.global_boundary_facet_count = 8U;
  graph.raw.boundary_facet_ids = boundary_ids;
  graph.raw.boundary_facets = boundary_vertices;
  graph.model.boundaries = &rule;
  graph.model.boundary_count = 1U;
  if (ss_solver_mark_exterior_entities(&graph.raw, graph.model.dimension) != SS_OK)
    return 1;
  strcpy(rule.field, "pore_pressure");
  rule.condition_kind = SS_SOLVER_BOUNDARY_ESSENTIAL;
  rule.value_kind = SS_SOLVER_VALUE_CONSTANT;
  rule.component_mask = 1U;
  SsEntityId allowed[] = {1,   2,   3,   4,   5,   6,   102, 103, 104,
                          105, 106, 107, 108, 109, 110, 111, 112, 113,
                          205, 206, 207, 208, 209, 210, 211, 212};
  rule.entity_ids = ids0;
  rule.entity_count = 6U;
  failed |= check_case(&graph, 6U, 1U, allowed, 26U);
  failed |= check_case(&graph, 18U, 2U, allowed, 26U);
  failed |= check_case(&graph, 66U, 4U, allowed, 26U);
  SsEntityId explicit_face_allowed[] = {1, 2, 3, 101, 102, 106, 201};
  rule.entity_ids = &ids2[0];
  rule.entity_count = 1U;
  failed |= check_case(&graph, 15U, 4U, explicit_face_allowed, 7U);
  for (uint32_t d = 1U; d < graph.model.dimension; ++d)
    free(graph.raw.entities[d].exterior);
  return failed;
}

static int test_interior_point(void) {
  SsSolverGraph graph = {0};
  SsSolverBoundaryRule rule = {0};
  int failed = 0;
  SsEntityId ids0[] = {1, 2, 3, 4, 5};
  SsEntityId vertices0[] = {1, 2, 3, 4, 5};
  graph.raw.entities[0] = (SsSolverEntityTable){5U, 1U, ids0, vertices0, NULL};
  SsEntityId ids1[] = {101, 102, 103, 104, 105, 106, 107, 108};
  SsEntityId vertices1[] = {1, 2, 1, 4, 1, 5, 2, 3, 2, 5, 3, 4, 3, 5, 4, 5};
  graph.raw.entities[1] = (SsSolverEntityTable){8U, 2U, ids1, vertices1, NULL};
  SsEntityId ids2[] = {201, 202, 203, 204};
  SsEntityId vertices2[] = {1, 2, 5, 1, 4, 5, 2, 3, 5, 3, 4, 5};
  graph.raw.entities[2] = (SsSolverEntityTable){4U, 3U, ids2, vertices2, NULL};
  SsEntityId boundary_ids[] = {101, 102, 104, 106};
  uint64_t boundary_vertices[] = {0, 1, 0, 3, 1, 2, 2, 3};
  graph.model.dimension = 2U;
  graph.raw.vertex_ids = ids0;
  graph.raw.input.global_vertex_count = 5U;
  graph.raw.input.global_boundary_facet_count = 4U;
  graph.raw.boundary_facet_ids = boundary_ids;
  graph.raw.boundary_facets = boundary_vertices;
  graph.model.boundaries = &rule;
  graph.model.boundary_count = 1U;
  if (ss_solver_mark_exterior_entities(&graph.raw, graph.model.dimension) != SS_OK)
    return 1;
  strcpy(rule.field, "pore_pressure");
  rule.condition_kind = SS_SOLVER_BOUNDARY_ESSENTIAL;
  rule.value_kind = SS_SOLVER_VALUE_CONSTANT;
  rule.component_mask = 1U;
  rule.entity_ids = &ids0[4];
  rule.entity_count = 1U;
  failed |= check_case(&graph, 1U, 4U, &ids0[4], 1U);
  for (uint32_t d = 1U; d < graph.model.dimension; ++d)
    free(graph.raw.entities[d].exterior);
  return failed;
}

int main(void) { return test_square() | test_octahedron() | test_interior_point(); }
