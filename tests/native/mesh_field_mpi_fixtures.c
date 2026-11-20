/* Distributed simplex and multi-cell mesh fixtures.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "mesh_field_mpi_internal.h"

SsMeshSharedPointBalance fixture_shared_point_balance = SS_MESH_BALANCE_PETSC;

double descriptor_node_coordinate(const SsFeDescriptor *descriptor, size_t node,
                                  uint32_t coordinate) {
  const size_t offset = node * (size_t)(descriptor->dimension + 1U);
  return (double)descriptor->barycentric_indices[offset + (size_t)coordinate + 1U] /
         (double)descriptor->order;
}

static void owned_range(size_t global_count, int rank, int size, size_t *start,
                        size_t *count) {
  const size_t quotient = global_count / (size_t)size;
  const size_t remainder = global_count % (size_t)size;
  *count = quotient + ((size_t)rank < remainder ? 1U : 0U);
  *start =
      (size_t)rank * quotient + ((size_t)rank < remainder ? (size_t)rank : remainder);
}

static SsEntityId fixture_vertex_id(uint32_t dimension, size_t global_node) {
  return UINT64_C(0x10000000000) + (SsEntityId)(dimension * 100U) +
         (SsEntityId)global_node + UINT64_C(1);
}

SsEntityId fixture_coordinate_id(uint32_t dimension, uint32_t coordinate_order,
                                 size_t global_node) {
  if (global_node < (size_t)dimension + 1U) {
    return fixture_vertex_id(dimension, global_node);
  }
  return UINT64_C(0x90000000000) + (SsEntityId)(dimension * 1000U) +
         (SsEntityId)(coordinate_order * 100U) + (SsEntityId)global_node;
}

SsEntityId strip_vertex_id(size_t ordinal) {
  return UINT64_C(0x110000000000) + (SsEntityId)ordinal;
}

SsEntityId strip_cell_id(size_t ordinal) {
  return UINT64_C(0x120000000000) + (SsEntityId)ordinal;
}

static SsEntityId strip_facet_id(size_t ordinal) {
  return UINT64_C(0x130000000000) + (SsEntityId)ordinal;
}

static SsEntityId fixture_intermediate_id(uint32_t dimension, uint32_t entity_dimension,
                                          const SsEntityId *vertex_ids,
                                          size_t vertex_count) {
  uint64_t hash = UINT64_C(0x7a3c6f19d4e2b851);
  size_t vertex;
  hash ^= (uint64_t)dimension << 48U;
  hash ^= (uint64_t)entity_dimension << 40U;
  for (vertex = 0U; vertex < vertex_count; ++vertex) {
    hash ^=
        vertex_ids[vertex] + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6U) + (hash >> 2U);
  }
  return hash == 0U ? UINT64_C(1) : hash;
}

SsEntityId strip_intermediate_id(size_t first, size_t second) {
  static const size_t boundary_vertices[][2] = {
      {0U, 1U},
      {1U, 3U},
      {2U, 3U},
      {0U, 2U},
  };
  size_t boundary;
  if (second < first) {
    const size_t swap = first;
    first = second;
    second = swap;
  }
  for (boundary = 0U;
       boundary < sizeof(boundary_vertices) / sizeof(boundary_vertices[0]);
       ++boundary) {
    if (boundary_vertices[boundary][0] == first &&
        boundary_vertices[boundary][1] == second) {
      return strip_facet_id(boundary);
    }
  }
  return UINT64_C(0x140000000000) + (SsEntityId)(first * 16U + second);
}

static SsEntityId tetra_pair_vertex_id(size_t ordinal) {
  return UINT64_C(0x210000000000) + (SsEntityId)ordinal;
}

static SsEntityId tetra_pair_cell_id(size_t ordinal) {
  return UINT64_C(0x220000000000) + (SsEntityId)ordinal;
}

static SsEntityId tetra_pair_intermediate_id(uint32_t entity_dimension,
                                             const size_t *vertices,
                                             size_t vertex_count) {
  SsEntityId vertex_ids[3] = {0U, 0U, 0U};
  size_t vertex;
  for (vertex = 0U; vertex < vertex_count; ++vertex) {
    vertex_ids[vertex] = tetra_pair_vertex_id(vertices[vertex]);
  }
  return fixture_intermediate_id(3U, entity_dimension, vertex_ids, vertex_count);
}

void fixture_coordinate(const SsFeDescriptor *descriptor, size_t global_node,
                        double result[SS_FE_MAX_DIMENSION]) {
  const double r = descriptor_node_coordinate(descriptor, global_node, 0U);
  const double s = descriptor_node_coordinate(descriptor, global_node, 1U);
  result[0] = r;
  result[1] = descriptor->dimension == 2U ? s * (1.0 + 0.3 * r) : s;
  if (descriptor->dimension == 3U) {
    const double t = descriptor_node_coordinate(descriptor, global_node, 2U);
    result[2] = t * (1.0 + 0.4 * r);
  }
}

int build_fixture(uint32_t dimension, uint32_t coordinate_order, int rank, int size,
                  CompactFixture *fixture) {
  const SsFeDescriptor *descriptor = NULL;
  size_t node;
  size_t coordinate;
  size_t owned_vertex_count = 0U;
  size_t owned_coordinate_count = 0U;
  const size_t global_vertex_count = (size_t)dimension + 1U;
  const size_t facet_count = (size_t)dimension + 1U;
  size_t intermediate_count = 0U;

  memset(fixture, 0, sizeof(*fixture));
  if (ss_fe_descriptor_find(dimension, coordinate_order, &descriptor) != SS_OK) {
    return 1;
  }
  owned_range(global_vertex_count, rank, size, &fixture->owned_vertex_start,
              &owned_vertex_count);
  owned_range(descriptor->node_count, rank, size, &fixture->owned_coordinate_start,
              &owned_coordinate_count);
  for (node = 0U; node < owned_vertex_count; ++node) {
    const size_t global_node = fixture->owned_vertex_start + node;
    fixture->vertex_ids[node] = fixture_vertex_id(dimension, global_node);
    for (coordinate = 0U; coordinate < (size_t)dimension; ++coordinate) {
      fixture->vertex_coordinates[node * (size_t)dimension + coordinate] =
          global_node == coordinate + 1U ? 1.0 : 0.0;
    }
  }
  for (node = 0U; node < global_vertex_count; ++node) {
    fixture->cells[node] = (uint64_t)node;
  }
  fixture->cell_ids[0] = UINT64_C(0x70000000000) + (SsEntityId)(dimension * 100U) +
                         (SsEntityId)coordinate_order;

  if (dimension == 2U) {
    const uint64_t facets[] = {0U, 1U, 1U, 2U, 0U, 2U};
    memcpy(fixture->facets, facets, sizeof(facets));
  } else {
    const uint64_t facets[] = {
        0U, 1U, 2U, 0U, 1U, 3U, 0U, 2U, 3U, 1U, 2U, 3U,
    };
    memcpy(fixture->facets, facets, sizeof(facets));
  }
  if (rank == 0) {
    size_t first;
    for (first = 0U; first < global_vertex_count; ++first) {
      size_t second;
      for (second = first + 1U; second < global_vertex_count; ++second) {
        const SsEntityId vertices[] = {
            fixture_vertex_id(dimension, first),
            fixture_vertex_id(dimension, second),
        };
        fixture->intermediate_entities[intermediate_count++] =
            (SsMeshIntermediateEntity){
                1U,
                fixture_intermediate_id(dimension, 1U, vertices, 2U),
                {vertices[0], vertices[1], 0U},
        };
      }
    }
    if (dimension == 3U) {
      for (first = 0U; first < global_vertex_count; ++first) {
        size_t second;
        for (second = first + 1U; second < global_vertex_count; ++second) {
          size_t third;
          for (third = second + 1U; third < global_vertex_count; ++third) {
            const SsEntityId vertices[] = {
                fixture_vertex_id(dimension, first),
                fixture_vertex_id(dimension, second),
                fixture_vertex_id(dimension, third),
            };
            fixture->intermediate_entities[intermediate_count++] =
                (SsMeshIntermediateEntity){
                    2U,
                    fixture_intermediate_id(dimension, 2U, vertices, 3U),
                    {vertices[0], vertices[1], vertices[2]},
            };
          }
        }
      }
    }
  }
  for (node = 0U; node < facet_count; ++node) {
    SsEntityId vertices[3] = {0U, 0U, 0U};
    size_t vertex;
    for (vertex = 0U; vertex < (size_t)dimension; ++vertex) {
      vertices[vertex] = fixture_vertex_id(
          dimension, (size_t)fixture->facets[node * dimension + vertex]);
    }
    fixture->facet_ids[node] =
        fixture_intermediate_id(dimension, dimension - 1U, vertices, dimension);
  }
  fixture->labels[0] =
      (SsMeshLabelAssignment){SS_MESH_LABEL_MATERIAL, fixture->cell_ids[0], 51U};
  for (node = 0U; node < facet_count; ++node) {
    fixture->labels[node + 1U] = (SsMeshLabelAssignment){
        SS_MESH_LABEL_SUPPORT,
        fixture->facet_ids[node],
        61U + (uint32_t)node,
    };
  }

  for (node = 0U; node < owned_coordinate_count; ++node) {
    const size_t global_node = fixture->owned_coordinate_start + node;
    double values[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
    fixture->coordinate_node_ids[node] =
        fixture_coordinate_id(dimension, coordinate_order, global_node);
    fixture_coordinate(descriptor, global_node, values);
    for (coordinate = 0U; coordinate < (size_t)dimension; ++coordinate) {
      fixture->coordinate_coordinates[node * (size_t)dimension + coordinate] =
          values[coordinate];
    }
  }
  for (node = 0U; node < descriptor->node_count; ++node) {
    fixture->cell_coordinate_nodes[node] = (uint64_t)node;
  }

  fixture->input = (SsMeshInput){
      .dimension = dimension,
      .global_vertex_count = global_vertex_count,
      .global_cell_count = 1U,
      .global_boundary_facet_count = facet_count,
      .global_intermediate_entity_count = dimension == 2U ? 3U : 10U,
      .global_coordinate_node_count = descriptor->node_count,
      .vertex_count = owned_vertex_count,
      .cell_count = rank == 0 ? 1U : 0U,
      .boundary_facet_count = rank == 0 ? facet_count : 0U,
      .vertex_ids = fixture->vertex_ids,
      .vertex_coordinates = fixture->vertex_coordinates,
      .cell_ids = fixture->cell_ids,
      .cells = fixture->cells,
      .boundary_facet_ids = fixture->facet_ids,
      .boundary_facets = fixture->facets,
      .intermediate_entity_count = intermediate_count,
      .intermediate_entities = fixture->intermediate_entities,
      .label_count = rank == 0 ? facet_count + 1U : 0U,
      .labels = fixture->labels,
      .coordinate_order = coordinate_order,
      .coordinate_node_count = owned_coordinate_count,
      .coordinate_node_ids = fixture->coordinate_node_ids,
      .coordinate_coordinates = fixture->coordinate_coordinates,
      .cell_coordinate_nodes = fixture->cell_coordinate_nodes,
      .absolute_determinant_tolerance = 1.0e-12,
      .relative_determinant_tolerance = 1.0e-12,
      .shared_point_balance = fixture_shared_point_balance,
  };
  return 0;
}

int build_strip_fixture(int rank, int size, StripFixture *fixture) {
  static const double global_coordinates[] = {
      0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
  };
  static const uint64_t global_cells[][3] = {
      {0U, 1U, 2U},
      {1U, 3U, 2U},
  };
  static const uint64_t global_facets[][2] = {
      {0U, 1U},
      {1U, 3U},
      {3U, 2U},
      {2U, 0U},
  };
  static const size_t facet_parent[] = {0U, 1U, 1U, 0U};
  size_t owned_vertex_start = 0U;
  size_t owned_vertex_count = 0U;
  size_t vertex;
  size_t global_cell;
  size_t global_facet;
  size_t local_cell = 0U;
  size_t local_facet = 0U;
  size_t label_count = 0U;
  size_t intermediate_count = 0U;
  memset(fixture, 0, sizeof(*fixture));
  owned_range(4U, rank, size, &owned_vertex_start, &owned_vertex_count);
  for (vertex = 0U; vertex < owned_vertex_count; ++vertex) {
    const size_t global_vertex = owned_vertex_start + vertex;
    fixture->vertex_ids[vertex] = strip_vertex_id(global_vertex);
    fixture->coordinate_node_ids[vertex] = strip_vertex_id(global_vertex);
    fixture->vertex_coordinates[vertex * 2U] = global_coordinates[global_vertex * 2U];
    fixture->vertex_coordinates[vertex * 2U + 1U] =
        global_coordinates[global_vertex * 2U + 1U];
    fixture->coordinate_coordinates[vertex * 2U] =
        global_coordinates[global_vertex * 2U];
    fixture->coordinate_coordinates[vertex * 2U + 1U] =
        global_coordinates[global_vertex * 2U + 1U];
  }
  for (global_cell = 0U; global_cell < 2U; ++global_cell) {
    size_t corner;
    if ((int)(global_cell % (size_t)size) != rank) {
      continue;
    }
    fixture->cell_ids[local_cell] = strip_cell_id(global_cell);
    for (corner = 0U; corner < 3U; ++corner) {
      fixture->cells[local_cell * 3U + corner] = global_cells[global_cell][corner];
      fixture->cell_coordinate_nodes[local_cell * 3U + corner] =
          global_cells[global_cell][corner];
    }
    fixture->labels[label_count++] = (SsMeshLabelAssignment){
        SS_MESH_LABEL_MATERIAL,
        strip_cell_id(global_cell),
        101U + (uint32_t)global_cell,
    };
    {
      static const size_t pairs[][2] = {
          {0U, 1U},
          {0U, 2U},
          {1U, 2U},
      };
      size_t pair;
      for (pair = 0U; pair < sizeof(pairs) / sizeof(pairs[0]); ++pair) {
        size_t first = (size_t)global_cells[global_cell][pairs[pair][0]];
        size_t second = (size_t)global_cells[global_cell][pairs[pair][1]];
        SsEntityId first_id;
        SsEntityId second_id;
        size_t existing;
        if (second < first) {
          const size_t swap = first;
          first = second;
          second = swap;
        }
        first_id = strip_vertex_id(first);
        second_id = strip_vertex_id(second);
        for (existing = 0U; existing < intermediate_count; ++existing) {
          if (fixture->intermediate_entities[existing].vertex_ids[0] == first_id &&
              fixture->intermediate_entities[existing].vertex_ids[1] == second_id) {
            break;
          }
        }
        if (existing == intermediate_count) {
          fixture->intermediate_entities[intermediate_count++] =
              (SsMeshIntermediateEntity){
                  1U,
                  strip_intermediate_id(first, second),
                  {first_id, second_id, 0U},
          };
        }
      }
    }
    ++local_cell;
  }
  for (global_facet = 0U; global_facet < 4U; ++global_facet) {
    if ((int)(facet_parent[global_facet] % (size_t)size) != rank) {
      continue;
    }
    fixture->facet_ids[local_facet] = strip_facet_id(global_facet);
    fixture->facets[local_facet * 2U] = global_facets[global_facet][0];
    fixture->facets[local_facet * 2U + 1U] = global_facets[global_facet][1];
    fixture->labels[label_count++] = (SsMeshLabelAssignment){
        SS_MESH_LABEL_SUPPORT,
        strip_facet_id(global_facet),
        201U + (uint32_t)global_facet,
    };
    ++local_facet;
  }
  fixture->input = (SsMeshInput){
      .dimension = 2U,
      .global_vertex_count = 4U,
      .global_cell_count = 2U,
      .global_boundary_facet_count = 4U,
      .global_intermediate_entity_count = 5U,
      .global_coordinate_node_count = 4U,
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
      .shared_point_balance = fixture_shared_point_balance,
  };
  return 0;
}

static int append_tetra_pair_entity(TetraPairFixture *fixture,
                                    uint32_t entity_dimension,
                                    const size_t *unsorted_vertices,
                                    size_t vertex_count, size_t *intermediate_count) {
  size_t vertices[3] = {0U, 0U, 0U};
  SsEntityId vertex_ids[3] = {0U, 0U, 0U};
  size_t vertex;
  size_t existing;
  if (fixture == NULL || unsorted_vertices == NULL || intermediate_count == NULL ||
      vertex_count != (size_t)entity_dimension + 1U || vertex_count > 3U) {
    return 1;
  }
  for (vertex = 0U; vertex < vertex_count; ++vertex) {
    size_t insertion = vertex;
    vertices[vertex] = unsorted_vertices[vertex];
    while (insertion > 0U && vertices[insertion] < vertices[insertion - 1U]) {
      const size_t swap = vertices[insertion];
      vertices[insertion] = vertices[insertion - 1U];
      vertices[insertion - 1U] = swap;
      --insertion;
    }
  }
  for (vertex = 0U; vertex < vertex_count; ++vertex) {
    vertex_ids[vertex] = tetra_pair_vertex_id(vertices[vertex]);
  }
  for (existing = 0U; existing < *intermediate_count; ++existing) {
    const SsMeshIntermediateEntity *entity = &fixture->intermediate_entities[existing];
    if (entity->dimension == entity_dimension &&
        memcmp(entity->vertex_ids, vertex_ids, vertex_count * sizeof(vertex_ids[0])) ==
            0) {
      return 0;
    }
  }
  if (*intermediate_count >= sizeof(fixture->intermediate_entities) /
                                 sizeof(fixture->intermediate_entities[0])) {
    return 1;
  }
  fixture->intermediate_entities[*intermediate_count] = (SsMeshIntermediateEntity){
      entity_dimension,
      fixture_intermediate_id(3U, entity_dimension, vertex_ids, vertex_count),
      {vertex_ids[0], vertex_ids[1], vertex_ids[2]},
  };
  ++*intermediate_count;
  return 0;
}

int build_tetra_pair_fixture(int rank, int size, TetraPairFixture *fixture) {
  static const double global_coordinates[] = {
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
  };
  static const uint64_t global_cells[][4] = {
      {0U, 1U, 2U, 3U},
      {4U, 1U, 3U, 2U},
  };
  static const uint64_t global_facets[][3] = {
      {0U, 1U, 2U}, {0U, 1U, 3U}, {0U, 2U, 3U},
      {1U, 3U, 4U}, {1U, 2U, 4U}, {2U, 3U, 4U},
  };
  static const size_t facet_parent[] = {0U, 0U, 0U, 1U, 1U, 1U};
  static const size_t edge_corners[][2] = {
      {0U, 1U}, {0U, 2U}, {0U, 3U}, {1U, 2U}, {1U, 3U}, {2U, 3U},
  };
  static const size_t face_corners[][3] = {
      {0U, 1U, 2U},
      {0U, 1U, 3U},
      {0U, 2U, 3U},
      {1U, 2U, 3U},
  };
  size_t owned_vertex_start = 0U;
  size_t owned_vertex_count = 0U;
  size_t vertex;
  size_t global_cell;
  size_t global_facet;
  size_t local_cell = 0U;
  size_t local_facet = 0U;
  size_t label_count = 0U;
  size_t intermediate_count = 0U;
  memset(fixture, 0, sizeof(*fixture));
  owned_range(5U, rank, size, &owned_vertex_start, &owned_vertex_count);
  for (vertex = 0U; vertex < owned_vertex_count; ++vertex) {
    const size_t global_vertex = owned_vertex_start + vertex;
    size_t coordinate;
    fixture->vertex_ids[vertex] = tetra_pair_vertex_id(global_vertex);
    fixture->coordinate_node_ids[vertex] = tetra_pair_vertex_id(global_vertex);
    for (coordinate = 0U; coordinate < 3U; ++coordinate) {
      const double value = global_coordinates[global_vertex * 3U + coordinate];
      fixture->vertex_coordinates[vertex * 3U + coordinate] = value;
      fixture->coordinate_coordinates[vertex * 3U + coordinate] = value;
    }
  }
  for (global_cell = 0U; global_cell < 2U; ++global_cell) {
    size_t corner;
    size_t entity;
    if ((int)(global_cell % (size_t)size) != rank) {
      continue;
    }
    fixture->cell_ids[local_cell] = tetra_pair_cell_id(global_cell);
    for (corner = 0U; corner < 4U; ++corner) {
      fixture->cells[local_cell * 4U + corner] = global_cells[global_cell][corner];
      fixture->cell_coordinate_nodes[local_cell * 4U + corner] =
          global_cells[global_cell][corner];
    }
    fixture->labels[label_count++] = (SsMeshLabelAssignment){
        SS_MESH_LABEL_MATERIAL,
        tetra_pair_cell_id(global_cell),
        301U + (uint32_t)global_cell,
    };
    for (entity = 0U; entity < sizeof(edge_corners) / sizeof(edge_corners[0]);
         ++entity) {
      const size_t vertices[] = {
          (size_t)global_cells[global_cell][edge_corners[entity][0]],
          (size_t)global_cells[global_cell][edge_corners[entity][1]],
      };
      if (append_tetra_pair_entity(fixture, 1U, vertices, 2U, &intermediate_count) !=
          0) {
        return 1;
      }
    }
    for (entity = 0U; entity < sizeof(face_corners) / sizeof(face_corners[0]);
         ++entity) {
      const size_t vertices[] = {
          (size_t)global_cells[global_cell][face_corners[entity][0]],
          (size_t)global_cells[global_cell][face_corners[entity][1]],
          (size_t)global_cells[global_cell][face_corners[entity][2]],
      };
      if (append_tetra_pair_entity(fixture, 2U, vertices, 3U, &intermediate_count) !=
          0) {
        return 1;
      }
    }
    ++local_cell;
  }
  for (global_facet = 0U;
       global_facet < sizeof(global_facets) / sizeof(global_facets[0]);
       ++global_facet) {
    size_t corner;
    size_t vertices[3];
    if ((int)(facet_parent[global_facet] % (size_t)size) != rank) {
      continue;
    }
    for (corner = 0U; corner < 3U; ++corner) {
      vertices[corner] = (size_t)global_facets[global_facet][corner];
      fixture->facets[local_facet * 3U + corner] = global_facets[global_facet][corner];
    }
    fixture->facet_ids[local_facet] = tetra_pair_intermediate_id(2U, vertices, 3U);
    fixture->labels[label_count++] = (SsMeshLabelAssignment){
        SS_MESH_LABEL_SUPPORT,
        fixture->facet_ids[local_facet],
        401U + (uint32_t)global_facet,
    };
    ++local_facet;
  }
  fixture->input = (SsMeshInput){
      .dimension = 3U,
      .global_vertex_count = 5U,
      .global_cell_count = 2U,
      .global_boundary_facet_count = 6U,
      .global_intermediate_entity_count = 16U,
      .global_coordinate_node_count = 5U,
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
      .shared_point_balance = fixture_shared_point_balance,
  };
  return 0;
}
