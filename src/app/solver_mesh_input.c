/* Read balanced HDF5 slabs and join the initial cell closures.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/core/sort.h"
#include "slope_internal/core/util.h"
#include "solver_input_internal.h"

#include "slope_internal/core/collective.h"
#include "slope_internal/fe/fe.h"
#include "slope_internal/mesh/distribution.h"

#include <petscsf.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  SsEntityId vertices[3];
  uint32_t width;
  SsEntityId stable_id;
  uint64_t ordinal_vertices[3];
} SsSolverFace;

void ss_solver_raw_release_payload(SsSolverRawMesh *raw) {
  uint32_t dimension;
  if (raw == NULL) {
    return;
  }
  free(raw->vertex_ids);
  free(raw->vertex_coordinates);
  free(raw->cell_ids);
  free(raw->cells);
  free(raw->boundary_facet_ids);
  free(raw->boundary_facets);
  free(raw->intermediate_entities);
  free(raw->labels);
  free(raw->coordinate_node_ids);
  free(raw->coordinate_coordinates);
  free(raw->cell_coordinate_nodes);
  for (dimension = 0U; dimension <= SS_FE_MAX_DIMENSION; ++dimension) {
    free(raw->entities[dimension].ids);
    free(raw->entities[dimension].vertex_ids);
    free(raw->entities[dimension].exterior);
    memset(&raw->entities[dimension], 0, sizeof(raw->entities[dimension]));
  }
  raw->vertex_ids = NULL;
  raw->vertex_coordinates = NULL;
  raw->cell_ids = NULL;
  raw->cells = NULL;
  raw->boundary_facet_ids = NULL;
  raw->boundary_facets = NULL;
  raw->intermediate_entities = NULL;
  raw->labels = NULL;
  raw->label_count = 0U;
  raw->coordinate_node_ids = NULL;
  raw->coordinate_coordinates = NULL;
  raw->cell_coordinate_nodes = NULL;
  raw->input.vertex_count = 0U;
  raw->input.cell_count = 0U;
  raw->input.boundary_facet_count = 0U;
  raw->input.intermediate_entity_count = 0U;
  raw->input.label_count = 0U;
  raw->input.coordinate_node_count = 0U;
  raw->input.vertex_ids = NULL;
  raw->input.vertex_coordinates = NULL;
  raw->input.cell_ids = NULL;
  raw->input.cells = NULL;
  raw->input.boundary_facet_ids = NULL;
  raw->input.boundary_facets = NULL;
  raw->input.intermediate_entities = NULL;
  raw->input.labels = NULL;
  raw->input.coordinate_node_ids = NULL;
  raw->input.coordinate_coordinates = NULL;
  raw->input.cell_coordinate_nodes = NULL;
}

void ss_solver_raw_destroy(SsSolverRawMesh *raw) {
  if (raw == NULL) {
    return;
  }
  ss_solver_raw_release_payload(raw);
  memset(raw, 0, sizeof(*raw));
}

int ss_solver_entity_id_index(const SsEntityId *ids, size_t count, SsEntityId target,
                              size_t *index) {
  size_t left = 0U;
  size_t right = count;
  while (left < right) {
    const size_t middle = left + (right - left) / 2U;
    if (ids[middle] < target) {
      left = middle + 1U;
    } else {
      right = middle;
    }
  }
  if (left < count && ids[left] == target) {
    if (index != NULL) {
      *index = left;
    }
    return 1;
  }
  return 0;
}

static void ss_solver_sort_ids(SsEntityId *values, size_t count) {
  size_t left;
  for (left = 1U; left < count; ++left) {
    SsEntityId value = values[left];
    size_t right = left;
    while (right > 0U && values[right - 1U] > value) {
      values[right] = values[right - 1U];
      --right;
    }
    values[right] = value;
  }
}

static int ss_solver_face_compare(const void *left_pointer, const void *right_pointer) {
  const SsSolverFace *left = (const SsSolverFace *)left_pointer;
  const SsSolverFace *right = (const SsSolverFace *)right_pointer;
  uint32_t index;
  if (left->width != right->width) {
    return left->width < right->width ? -1 : 1;
  }
  for (index = 0U; index < left->width; ++index) {
    if (left->vertices[index] != right->vertices[index]) {
      return left->vertices[index] < right->vertices[index] ? -1 : 1;
    }
  }
  return 0;
}

SsStatus ss_solver_mark_exterior_entities(SsSolverRawMesh *raw, uint32_t dimension) {
  const size_t facets = (size_t)raw->input.global_boundary_facet_count;
  const size_t per_facet = dimension == 3U ? 4U : 1U;
  size_t count = 0U, facet;
  uint32_t d;
  SsSolverFace *trace;
  size_t trace_bytes;
  if ((dimension != 2U && dimension != 3U) ||
      ss_checked_product_size((size_t[]){facets, per_facet, sizeof(*trace)}, 3U,
                              &trace_bytes) != SS_OK) {
    return SS_ERROR_INTEGRITY;
  }
  trace = (SsSolverFace *)calloc(1U, trace_bytes == 0U ? sizeof(*trace) : trace_bytes);
  if (trace == NULL)
    return SS_ERROR_OUT_OF_MEMORY;
  for (facet = 0U; facet < facets; ++facet) {
    SsSolverFace key = {0};
    uint32_t vertex;
    key.width = dimension;
    for (vertex = 0U; vertex < dimension; ++vertex) {
      uint64_t ordinal = raw->boundary_facets[facet * dimension + vertex];
      if (ordinal >= raw->input.global_vertex_count) {
        free(trace);
        return SS_ERROR_INTEGRITY;
      }
      key.vertices[vertex] = raw->vertex_ids[ordinal];
    }
    ss_solver_sort_ids(key.vertices, key.width);
    trace[count++] = key;
    if (dimension == 3U) {
      uint32_t a, b;
      for (a = 0U; a < 3U; ++a) {
        for (b = a + 1U; b < 3U; ++b) {
          SsSolverFace edge = {0};
          edge.width = 2U;
          edge.vertices[0] = key.vertices[a];
          edge.vertices[1] = key.vertices[b];
          trace[count++] = edge;
        }
      }
    }
  }
  ss_sort(trace, count, sizeof(*trace), ss_solver_face_compare);
  for (d = 1U; d < dimension; ++d) {
    SsSolverEntityTable *table = &raw->entities[d];
    size_t entity;
    free(table->exterior);
    table->exterior = (unsigned char *)calloc(table->count == 0U ? 1U : table->count,
                                              sizeof(*table->exterior));
    if (table->exterior == NULL) {
      free(trace);
      return SS_ERROR_OUT_OF_MEMORY;
    }
    for (entity = 0U; entity < table->count; ++entity) {
      SsSolverFace key = {0};
      key.width = d + 1U;
      memcpy(key.vertices, table->vertex_ids + entity * table->width,
             key.width * sizeof(*key.vertices));
      ss_solver_sort_ids(key.vertices, key.width);
      table->exterior[entity] =
          (unsigned char)(bsearch(&key, trace, count, sizeof(*trace),
                                  ss_solver_face_compare) != NULL);
    }
  }
  free(trace);
  return SS_OK;
}

typedef struct {
  SsEntityId vertices[3];
  uint64_t ordinals[3];
  SsEntityId id;
  int requester;
  unsigned int exterior;
} SsInputTuple;

static int ss_solver_tuple_compare(const void *left_raw, const void *right_raw) {
  const SsInputTuple *left = left_raw;
  const SsInputTuple *right = right_raw;
  for (size_t i = 0U; i < 3U; ++i) {
    if (left->vertices[i] != right->vertices[i]) {
      return left->vertices[i] < right->vertices[i] ? -1 : 1;
    }
  }
  return 0;
}

static int ss_solver_tuple_owner(const void *raw, int ranks, const void *context) {
  const SsInputTuple *tuple = raw;
  (void)context;
  uint64_t hash = 0U;
  for (size_t i = 0U; i < 3U; ++i) {
    hash = ss_mesh_distribution_hash(hash ^ tuple->vertices[i]);
  }
  return (int)(hash % (uint64_t)ranks);
}

static int ss_solver_tuple_return(const void *raw, int ranks, const void *context) {
  const SsInputTuple *tuple = raw;
  (void)ranks;
  (void)context;
  return tuple->requester;
}

static void ss_solver_tuple_sort(SsInputTuple *tuple, size_t width) {
  for (size_t i = 1U; i < width; ++i) {
    size_t j = i;
    while (j > 0U && tuple->vertices[j - 1U] > tuple->vertices[j]) {
      const SsEntityId id = tuple->vertices[j - 1U];
      const uint64_t ordinal = tuple->ordinals[j - 1U];
      tuple->vertices[j - 1U] = tuple->vertices[j];
      tuple->ordinals[j - 1U] = tuple->ordinals[j];
      tuple->vertices[j] = id;
      tuple->ordinals[j] = ordinal;
      --j;
    }
  }
}

static SsStatus ss_solver_import_vertices(MPI_Comm communicator,
                                          const SsSolverRawMesh *raw,
                                          SsEntityId **cell_vertices) {
  PetscLayout layout = NULL;
  PetscSF vertices = NULL;
  PetscInt owned = 0, total = 0, leaves = 0;
  PetscInt *ordinals = NULL;
  SsEntityId empty = 0U;
  size_t count = 0U;
  size_t ordinal_bytes;
  size_t vertex_bytes;
  SsStatus status = SS_OK;
  *cell_vertices = NULL;
  if (ss_checked_mul_size(raw->input.cell_count, raw->input.dimension + 1U, &count) !=
          SS_OK ||
      ss_checked_u64_to_petsc(raw->input.vertex_count, &owned) != SS_OK ||
      ss_checked_u64_to_petsc(raw->input.global_vertex_count, &total) != SS_OK ||
      ss_checked_u64_to_petsc(count, &leaves) != SS_OK ||
      ss_checked_mul_size(count, sizeof(**cell_vertices), &vertex_bytes) != SS_OK ||
      ss_checked_mul_size(count, sizeof(*ordinals), &ordinal_bytes) != SS_OK) {
    status = SS_ERROR_OVERFLOW;
  }
  if (status == SS_OK && count != 0U) {
    ordinals = malloc(ordinal_bytes);
    *cell_vertices = malloc(vertex_bytes);
    if (ordinals == NULL || *cell_vertices == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  for (size_t i = 0U; status == SS_OK && i < count; ++i) {
    if (raw->cells[i] >= raw->input.global_vertex_count) {
      status = SS_ERROR_INTEGRITY;
    } else {
      ordinals[i] = (PetscInt)raw->cells[i];
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK &&
      (PetscLayoutCreate(communicator, &layout) != PETSC_SUCCESS ||
       PetscLayoutSetSize(layout, total) != PETSC_SUCCESS ||
       PetscLayoutSetLocalSize(layout, owned) != PETSC_SUCCESS ||
       PetscLayoutSetBlockSize(layout, 1) != PETSC_SUCCESS ||
       PetscLayoutSetUp(layout) != PETSC_SUCCESS ||
       PetscSFCreate(communicator, &vertices) != PETSC_SUCCESS ||
       PetscSFSetGraphLayout(vertices, layout, leaves, NULL, PETSC_COPY_VALUES,
                             ordinals) != PETSC_SUCCESS ||
       PetscSFBcastBegin(vertices, MPI_UINT64_T, owned == 0 ? &empty : raw->vertex_ids,
                         leaves == 0 ? &empty : *cell_vertices,
                         MPI_REPLACE) != PETSC_SUCCESS ||
       PetscSFBcastEnd(vertices, MPI_UINT64_T, owned == 0 ? &empty : raw->vertex_ids,
                       leaves == 0 ? &empty : *cell_vertices,
                       MPI_REPLACE) != PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  free(ordinals);
  (void)PetscSFDestroy(&vertices);
  (void)PetscLayoutDestroy(&layout);
  return ss_status_allreduce_max(communicator, status);
}

static SsStatus ss_solver_entity_rows(MPI_Comm communicator, hid_t file,
                                      uint32_t dimension, int rank, SsH5Rows *ids,
                                      SsH5Rows *vertices) {
  char path[64];
  SsStatus status;
  (void)snprintf(path, sizeof(path), "/mesh/entities/dimension_%u", dimension);
  status =
      ss_io_hdf5_read_rows(communicator, file, path, H5T_NATIVE_UINT64, 1, 1U, ids);
  if (status != SS_OK) {
    return status;
  }
  (void)snprintf(path, sizeof(path), "/mesh/entities/dimension_%u_vertex_ids",
                 dimension);
  status = ss_io_hdf5_read_rows(communicator, file, path, H5T_NATIVE_UINT64, 2,
                                dimension + 1U, vertices);
  if (status == SS_OK && ids->total != vertices->total) {
    status = SS_ERROR_INTEGRITY;
  }
  if (status == SS_OK) {
    const SsEntityId *stable_ids = ids->values;
    const SsEntityId *vertex_ids = vertices->values;
    for (size_t i = 0U; status == SS_OK && i < ids->count; ++i) {
      if (i != 0U && stable_ids[i - 1U] >= stable_ids[i]) {
        status = SS_ERROR_INTEGRITY;
      }
      for (size_t k = 1U; k < vertices->width; ++k) {
        const size_t j = i * vertices->width + k;
        if (vertex_ids[j - 1U] >= vertex_ids[j]) {
          status = SS_ERROR_INTEGRITY;
        }
      }
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    const SsEntityId *stable_ids = ids->values;
    const SsEntityId last = ids->count == 0U ? 0U : stable_ids[ids->count - 1U];
    SsEntityId preceding = 0U;
    if (MPI_Exscan(&last, &preceding, 1, MPI_UINT64_T, MPI_MAX, communicator) !=
        MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    } else if (rank > 0 && ids->count != 0U && stable_ids[0] <= preceding) {
      status = SS_ERROR_INTEGRITY;
    }
  }
  return ss_status_allreduce_max(communicator, status);
}

/* Each source tuple has one compiled entity and at least one incident cell.
 * A facet has one incident cell on the exterior and two in the interior. */
static SsStatus ss_solver_resolve_tuples(SsInputTuple *shard, size_t count, int facets,
                                         size_t *reply_count) {
  *reply_count = 0U;
  ss_sort(shard, count, sizeof(*shard), ss_solver_tuple_compare);
  for (size_t begin = 0U; begin < count;) {
    SsEntityId id = 0U;
    size_t entries = 0U, cells = 0U;
    size_t end = begin;
    while (end < count && ss_solver_tuple_compare(&shard[begin], &shard[end]) == 0) {
      if (shard[end].requester < 0) {
        id = shard[end].id;
        ++entries;
      } else {
        ++cells;
      }
      ++end;
    }
    if (entries != 1U || cells == 0U || (facets != 0 && cells > 2U)) {
      return SS_ERROR_INTEGRITY;
    }
    for (size_t i = begin; i < end; ++i) {
      if (shard[i].requester >= 0) {
        shard[i].id = id;
        shard[i].exterior = (unsigned int)(facets != 0 && cells == 1U);
        shard[(*reply_count)++] = shard[i];
      }
    }
    begin = end;
  }
  return SS_OK;
}

static SsStatus ss_solver_append_tuples(SsSolverRawMesh *raw, SsInputTuple *matches,
                                        size_t matched, size_t width, int facets) {
  size_t unique = 0U;
  size_t exterior = 0U;
  ss_sort(matches, matched, sizeof(*matches), ss_solver_tuple_compare);
  for (size_t i = 0U; i < matched; ++i) {
    if (unique == 0U ||
        ss_solver_tuple_compare(&matches[unique - 1U], &matches[i]) != 0) {
      exterior += matches[i].exterior;
      matches[unique++] = matches[i];
    }
  }
  const size_t first = raw->input.intermediate_entity_count;
  size_t total_entities;
  size_t entity_bytes;
  size_t facet_bytes;
  size_t facet_id_bytes;
  if (ss_checked_add_size(first, unique, &total_entities) != SS_OK ||
      ss_checked_mul_size(total_entities, sizeof(*raw->intermediate_entities),
                          &entity_bytes) != SS_OK ||
      ss_checked_product_size(
          (size_t[]){exterior, width, sizeof(*raw->boundary_facets)}, 3U,
          &facet_bytes) != SS_OK ||
      ss_checked_mul_size(exterior, sizeof(*raw->boundary_facet_ids),
                          &facet_id_bytes) != SS_OK) {
    return SS_ERROR_OVERFLOW;
  }
  if (unique != 0U) {
    SsMeshIntermediateEntity *resized =
        realloc(raw->intermediate_entities, entity_bytes);
    if (resized == NULL) {
      return SS_ERROR_OUT_OF_MEMORY;
    }
    raw->intermediate_entities = resized;
  }
  if (facets != 0 && exterior != 0U) {
    raw->boundary_facet_ids = malloc(facet_id_bytes);
    raw->boundary_facets = malloc(facet_bytes);
    if (raw->boundary_facet_ids == NULL || raw->boundary_facets == NULL) {
      return SS_ERROR_OUT_OF_MEMORY;
    }
  }
  raw->input.intermediate_entity_count = total_entities;
  for (size_t i = 0U; i < unique; ++i) {
    SsMeshIntermediateEntity *entry = &raw->intermediate_entities[first + i];
    entry->dimension = (uint32_t)width - 1U;
    entry->stable_entity_id = matches[i].id;
    memcpy(entry->vertex_ids, matches[i].vertices, sizeof(entry->vertex_ids));
    if (matches[i].exterior != 0U) {
      const size_t j = raw->input.boundary_facet_count++;
      raw->boundary_facet_ids[j] = matches[i].id;
      memcpy(raw->boundary_facets + j * width, matches[i].ordinals,
             width * sizeof(*raw->boundary_facets));
    }
  }
  return SS_OK;
}

static SsStatus ss_solver_join_tuples(MPI_Comm communicator, SsSolverRawMesh *raw,
                                      const SsH5Rows *ids, const SsH5Rows *vertices,
                                      const SsEntityId *cell_vertices, int rank,
                                      int ranks) {
  const size_t corners = raw->input.dimension + 1U;
  const size_t width = vertices->width;
  const size_t per_cell = width == 3U ? 4U : (corners * (corners - 1U)) / 2U;
  const int facets = width == raw->input.dimension;
  const SsEntityId *table_ids = ids->values;
  const SsEntityId *table_vertices = vertices->values;
  SsInputTuple *records = NULL, *shard = NULL, *matches = NULL;
  size_t record_count = 0U, shard_count = 0U, reply_count = 0U, match_count = 0U;
  SsStatus status = SS_OK;
  size_t request_count;
  size_t record_bytes;
  if (ss_checked_mul_size(raw->input.cell_count, per_cell, &request_count) != SS_OK ||
      ss_checked_add_size(ids->count, request_count, &record_count) != SS_OK ||
      ss_checked_mul_size(record_count, sizeof(*records), &record_bytes) != SS_OK) {
    status = SS_ERROR_OVERFLOW;
  } else {
    records = calloc(1U, record_bytes == 0U ? sizeof(*records) : record_bytes);
    if (records == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  for (size_t i = 0U; status == SS_OK && i < ids->count; ++i) {
    records[i].id = table_ids[i];
    records[i].requester = -1;
    memcpy(records[i].vertices, table_vertices + i * width, width * sizeof(SsEntityId));
  }
  size_t next = ids->count;
  for (size_t i = 0U; status == SS_OK && i < raw->input.cell_count; ++i) {
    for (unsigned int mask = 1U; mask < (1U << corners); ++mask) {
      SsInputTuple tuple = {.requester = rank};
      size_t used = 0U;
      for (size_t j = 0U; j < corners && used <= width; ++j) {
        if ((mask & (1U << j)) != 0U) {
          if (used < width) {
            tuple.vertices[used] = cell_vertices[i * corners + j];
            tuple.ordinals[used] = raw->cells[i * corners + j];
          }
          ++used;
        }
      }
      if (used == width) {
        ss_solver_tuple_sort(&tuple, width);
        records[next++] = tuple;
      }
    }
  }
  if (ranks == 1) {
    if (status == SS_OK) {
      status = ss_solver_resolve_tuples(records, record_count, facets, &reply_count);
    }
    if (status == SS_OK) {
      status = ss_solver_append_tuples(raw, records, reply_count, width, facets);
    }
    free(records);
    return status;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    status = ss_mesh_route_records(communicator, records, record_count,
                                   sizeof(*records), ss_solver_tuple_owner, NULL,
                                   (void **)&shard, &shard_count);
  }
  free(records);
  if (status == SS_OK) {
    status = ss_solver_resolve_tuples(shard, shard_count, facets, &reply_count);
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    status = ss_mesh_route_records(communicator, shard, reply_count, sizeof(*shard),
                                   ss_solver_tuple_return, NULL, (void **)&matches,
                                   &match_count);
  }
  free(shard);
  if (status == SS_OK && match_count != record_count - ids->count) {
    status = SS_ERROR_INTEGRITY;
  }
  if (status == SS_OK) {
    status = ss_solver_append_tuples(raw, matches, match_count, width, facets);
  }
  free(matches);
  return ss_status_allreduce_max(communicator, status);
}

static SsStatus ss_solver_read_topological_entities(MPI_Comm communicator, hid_t file,
                                                    SsSolverRawMesh *raw, int rank,
                                                    int ranks) {
  SsEntityId *cell_vertices = NULL;
  SsStatus status = ss_solver_import_vertices(communicator, raw, &cell_vertices);
  for (uint32_t d = 0U; status == SS_OK && d <= raw->input.dimension; ++d) {
    SsH5Rows ids = {0}, vertices = {0};
    status = ss_solver_entity_rows(communicator, file, d, rank, &ids, &vertices);
    if (status == SS_OK && d == 0U) {
      const size_t bytes = ids.count * sizeof(*raw->vertex_ids);
      if (ids.total != raw->input.global_vertex_count ||
          (bytes != 0U && (memcmp(raw->vertex_ids, ids.values, bytes) != 0 ||
                           memcmp(raw->vertex_ids, vertices.values, bytes) != 0))) {
        status = SS_ERROR_INTEGRITY;
      }
    } else if (status == SS_OK && d == raw->input.dimension) {
      if (ids.total != raw->input.global_cell_count) {
        status = SS_ERROR_INTEGRITY;
      }
    } else if (status == SS_OK) {
      status = ss_solver_join_tuples(communicator, raw, &ids, &vertices, cell_vertices,
                                     rank, ranks);
      raw->input.global_intermediate_entity_count += ids.total;
    }
    free(ids.values);
    free(vertices.values);
    status = ss_status_allreduce_max(communicator, status);
  }
  free(cell_vertices);
  const uint64_t local_boundary = raw->input.boundary_facet_count;
  if (status == SS_OK &&
      MPI_Allreduce(&local_boundary, &raw->input.global_boundary_facet_count, 1,
                    MPI_UINT64_T, MPI_SUM, communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  return status;
}

static SsStatus ss_solver_read_connectivity(MPI_Comm communicator, hid_t file,
                                            SsSolverRawMesh *raw,
                                            const SsFeDescriptor *descriptor,
                                            int ranks) {
  SsH5Rows connectivity = {0}, barycentric = {0};
  uint64_t functionals[SS_FE_MAX_NODES * (SS_FE_MAX_DIMENSION + 1U)] = {0};
  uint64_t row[SS_FE_MAX_NODES];
  size_t permutation[SS_FE_MAX_NODES];
  const size_t corners = raw->input.dimension + 1U;
  SsStatus status =
      ss_io_hdf5_read_rows(communicator, file, "/mesh/coordinates/connectivity",
                           H5T_NATIVE_UINT64, 2, descriptor->node_count, &connectivity);
  if (status == SS_OK) {
    status = ss_io_hdf5_read_rows(communicator, file,
                                  "/mesh/coordinates/barycentric_functionals",
                                  H5T_NATIVE_UINT64, 2, corners, &barycentric);
  }
  if (status == SS_OK && (connectivity.total != raw->input.global_cell_count ||
                          barycentric.total != descriptor->node_count)) {
    status = SS_ERROR_INTEGRITY;
  }
  if (status == SS_OK && barycentric.count != 0U) {
    memcpy(functionals + barycentric.start * corners, barycentric.values,
           barycentric.count * corners * sizeof(*functionals));
  }
  free(barycentric.values);
  /* Only the fixed-size reference element metadata is shared. */
  for (int owner = 0;
       status == SS_OK && owner < ranks && (size_t)owner < descriptor->node_count;
       ++owner) {
    size_t start, count;
    ss_partition_range(descriptor->node_count, owner, ranks, &start, &count);
    if (MPI_Bcast(functionals + start * corners, (int)(count * corners), MPI_UINT64_T,
                  owner, communicator) != MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  for (size_t canonical = 0U; status == SS_OK && canonical < descriptor->node_count;
       ++canonical) {
    size_t external;
    for (external = 0U; external < descriptor->node_count; ++external) {
      size_t k = 0U;
      while (k < corners &&
             functionals[external * corners + k] ==
                 descriptor->barycentric_indices[canonical * corners + k]) {
        ++k;
      }
      if (k == corners) {
        break;
      }
    }
    if (external == descriptor->node_count) {
      status = SS_ERROR_INTEGRITY;
    } else {
      permutation[canonical] = external;
    }
  }
  if (status == SS_OK) {
    raw->cell_coordinate_nodes = connectivity.values;
    connectivity.values = NULL;
    for (size_t i = 0U; i < raw->input.cell_count; ++i) {
      uint64_t *target = raw->cell_coordinate_nodes + i * descriptor->node_count;
      memcpy(row, target, descriptor->node_count * sizeof(*row));
      for (size_t j = 0U; j < descriptor->node_count; ++j) {
        target[j] = row[permutation[j]];
      }
    }
  }
  free(connectivity.values);
  return ss_status_allreduce_max(communicator, status);
}

static SsStatus ss_solver_read_payload(MPI_Comm communicator, hid_t file,
                                       SsSolverRawMesh *raw) {
  /* Paired arrays use the same balanced partition by construction. */
  const struct {
    const char *ids_path;
    const char *values_path;
    void **ids;
    void **values;
    size_t *count;
    uint64_t *total;
    hid_t value_type;
    size_t width;
  } tables[] = {
      {"/mesh/topology/vertex_ids", "/mesh/topology/vertex_coordinates",
       (void **)&raw->vertex_ids, (void **)&raw->vertex_coordinates,
       &raw->input.vertex_count, &raw->input.global_vertex_count, H5T_NATIVE_DOUBLE,
       raw->input.dimension},
      {"/mesh/topology/cell_ids", "/mesh/topology/cells", (void **)&raw->cell_ids,
       (void **)&raw->cells, &raw->input.cell_count, &raw->input.global_cell_count,
       H5T_NATIVE_UINT64, raw->input.dimension + 1U},
      {"/mesh/coordinates/functional_ids", "/mesh/coordinates/values",
       (void **)&raw->coordinate_node_ids, (void **)&raw->coordinate_coordinates,
       &raw->input.coordinate_node_count, &raw->input.global_coordinate_node_count,
       H5T_NATIVE_DOUBLE, raw->input.dimension},
  };
  SsStatus status = SS_OK;
  for (size_t i = 0U; status == SS_OK && i < sizeof(tables) / sizeof(tables[0]); ++i) {
    SsH5Rows ids = {0}, values = {0};
    status = ss_io_hdf5_read_rows(communicator, file, tables[i].ids_path,
                                  H5T_NATIVE_UINT64, 1, 1U, &ids);
    if (status == SS_OK) {
      status = ss_io_hdf5_read_rows(communicator, file, tables[i].values_path,
                                    tables[i].value_type, 2, tables[i].width, &values);
    }
    *tables[i].ids = ids.values;
    *tables[i].values = values.values;
    *tables[i].total = ids.total;
    *tables[i].count = ids.count;
    if (status == SS_OK && ids.total != values.total) {
      status = SS_ERROR_INTEGRITY;
    }
    status = ss_status_allreduce_max(communicator, status);
  }
  return status;
}

static SsStatus ss_solver_material_labels(MPI_Comm communicator, SsSolverRawMesh *raw,
                                          const SsSolverProblemModel *model) {
  size_t count = 0U;
  uint32_t *values = NULL;
  SsMeshIdValue *sources = NULL;
  SsStatus status = SS_OK;
  for (size_t i = 0U; status == SS_OK && i < model->material_count; ++i) {
    status = ss_checked_add_size(count, model->materials[i].entity_count, &count);
  }
  if (status == SS_OK) {
    sources = calloc(count == 0U ? 1U : count, sizeof(*sources));
    values = calloc(raw->input.cell_count == 0U ? 1U : raw->input.cell_count,
                    sizeof(*values));
    if (sources == NULL || values == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  size_t source = 0U;
  for (size_t i = 0U; status == SS_OK && i < model->material_count; ++i) {
    const SsSolverMaterial *material = &model->materials[i];
    for (size_t j = 0U; j < material->entity_count; ++j) {
      sources[source++] =
          (SsMeshIdValue){material->entity_ids[j], material->support_value};
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    status = ss_mesh_join_ids(communicator, sources, count, raw->cell_ids,
                              raw->input.cell_count, values);
  }
  for (size_t i = 0U; status == SS_OK && i < raw->input.cell_count; ++i) {
    if (values[i] == 0U) {
      status = SS_ERROR_INTEGRITY;
    } else {
      raw->labels[raw->label_count++] =
          (SsMeshLabelAssignment){SS_MESH_LABEL_MATERIAL, raw->cell_ids[i], values[i]};
    }
  }
  free(sources);
  free(values);
  return ss_status_allreduce_max(communicator, status);
}

static SsStatus ss_solver_natural_labels(MPI_Comm communicator, SsSolverRawMesh *raw,
                                         const SsSolverBoundaryRule *rule) {
  const size_t count = raw->input.boundary_facet_count;
  uint32_t *values = calloc(count == 0U ? 1U : count, sizeof(*values));
  SsMeshIdValue *sources =
      calloc(rule->entity_count == 0U ? 1U : rule->entity_count, sizeof(*sources));
  SsStatus status = sources == NULL || values == NULL ? SS_ERROR_OUT_OF_MEMORY : SS_OK;
  for (size_t i = 0U; status == SS_OK && i < rule->entity_count; ++i) {
    sources[i] = (SsMeshIdValue){rule->entity_ids[i], rule->support_value};
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    status = ss_mesh_join_ids(communicator, sources, rule->entity_count,
                              raw->boundary_facet_ids, count, values);
  }
  size_t additions = 0U;
  for (size_t i = 0U; status == SS_OK && i < count; ++i) {
    additions += values[i] != 0U;
  }
  if (status == SS_OK && additions != 0U) {
    size_t total;
    size_t bytes;
    if (ss_checked_add_size(raw->label_count, additions, &total) != SS_OK ||
        ss_checked_mul_size(total, sizeof(*raw->labels), &bytes) != SS_OK) {
      status = SS_ERROR_OVERFLOW;
    } else {
      SsMeshLabelAssignment *resized = realloc(raw->labels, bytes);
      if (resized == NULL) {
        status = SS_ERROR_OUT_OF_MEMORY;
      } else {
        raw->labels = resized;
      }
    }
  }
  for (size_t i = 0U; status == SS_OK && i < count; ++i) {
    if (values[i] != 0U) {
      raw->labels[raw->label_count++] = (SsMeshLabelAssignment){
          SS_MESH_LABEL_SUPPORT, raw->boundary_facet_ids[i], values[i]};
    }
  }
  free(values);
  free(sources);
  return ss_status_allreduce_max(communicator, status);
}

static SsStatus ss_solver_build_labels(MPI_Comm communicator, SsSolverRawMesh *raw,
                                       const SsSolverProblemModel *model) {
  const size_t facets = raw->input.boundary_facet_count;
  size_t capacity = 0U;
  SsStatus status = ss_checked_add_size(facets, raw->input.cell_count, &capacity);
  if (status == SS_OK) {
    raw->labels = calloc(capacity == 0U ? 1U : capacity, sizeof(*raw->labels));
    if (raw->labels == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    status = ss_solver_material_labels(communicator, raw, model);
  }
  for (size_t i = 0U; status == SS_OK && i < facets; ++i) {
    raw->labels[raw->label_count++] =
        (SsMeshLabelAssignment){SS_MESH_LABEL_SUPPORT, raw->boundary_facet_ids[i], 1U};
  }
  for (size_t i = 0U; status == SS_OK && i < model->boundary_count; ++i) {
    const SsSolverBoundaryRule *rule = &model->boundaries[i];
    if (rule->condition_kind == SS_SOLVER_BOUNDARY_NATURAL) {
      status = ss_solver_natural_labels(communicator, raw, rule);
    }
  }
  return status;
}

SsStatus ss_solver_load_mesh(MPI_Comm communicator, const char *path,
                             const SsSolverProblemModel *model, SsSolverRawMesh *raw) {
  hid_t file = H5I_INVALID_HID;
  int rank = 0, ranks = 0;
  const SsFeDescriptor *descriptor = NULL;
  SsStatus status =
      ss_fe_descriptor_find(model->dimension, model->coordinate_order, &descriptor);
  if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(communicator, &ranks) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  raw->input.dimension = model->dimension;
  raw->input.coordinate_order = model->coordinate_order;
  raw->input.absolute_determinant_tolerance = 1.0e-12;
  raw->input.relative_determinant_tolerance = 1.0e-12;
  if (status == SS_OK) {
    status = ss_io_hdf5_open(communicator, path, &file);
  }
  if (status == SS_OK) {
    status = ss_solver_read_payload(communicator, file, raw);
  }
  if (status == SS_OK) {
    status = ss_solver_read_connectivity(communicator, file, raw, descriptor, ranks);
  }
  if (status == SS_OK) {
    status = ss_solver_read_topological_entities(communicator, file, raw, rank, ranks);
  }
  if (file >= 0 && H5Fclose(file) < 0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    status = ss_solver_build_labels(communicator, raw, model);
  }
  if (status == SS_OK) {
    raw->input.vertex_ids = raw->vertex_ids;
    raw->input.vertex_coordinates = raw->vertex_coordinates;
    raw->input.cell_ids = raw->cell_ids;
    raw->input.cells = raw->cells;
    raw->input.boundary_facet_ids = raw->boundary_facet_ids;
    raw->input.boundary_facets = raw->boundary_facets;
    raw->input.intermediate_entities = raw->intermediate_entities;
    raw->input.labels = raw->labels;
    raw->input.label_count = raw->label_count;
    raw->input.coordinate_node_ids = raw->coordinate_node_ids;
    raw->input.coordinate_coordinates = raw->coordinate_coordinates;
    raw->input.cell_coordinate_nodes = raw->cell_coordinate_nodes;
  }
  return status;
}
