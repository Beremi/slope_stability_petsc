/* Point, vertex tuple and boundary indexes for mesh entities.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "mesh_internal.h"
#include "slope_internal/core/sort.h"

#include <petscdmplex.h>

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int ss_mesh_entity_id_compare(const void *left_raw, const void *right_raw) {
  const SsEntityId left = *(const SsEntityId *)left_raw;
  const SsEntityId right = *(const SsEntityId *)right_raw;
  return left < right ? -1 : left > right ? 1 : 0;
}

int ss_point_entry_compare(const void *left_raw, const void *right_raw) {
  const SsPointEntry *left = (const SsPointEntry *)left_raw;
  const SsPointEntry *right = (const SsPointEntry *)right_raw;
  if (left->kind != right->kind) {
    return left->kind < right->kind ? -1 : 1;
  }
  if (left->stable_id != right->stable_id) {
    return left->stable_id < right->stable_id ? -1 : 1;
  }
  return 0;
}

static int ss_point_id_entry_compare(const void *left_raw, const void *right_raw) {
  const SsPointIdEntry *left = (const SsPointIdEntry *)left_raw;
  const SsPointIdEntry *right = (const SsPointIdEntry *)right_raw;
  if (left->stable_id != right->stable_id) {
    return left->stable_id < right->stable_id ? -1 : 1;
  }
  return 0;
}

static int ss_entity_vertex_entry_compare(const void *left_raw, const void *right_raw) {
  const SsEntityVertexEntry *left = (const SsEntityVertexEntry *)left_raw;
  const SsEntityVertexEntry *right = (const SsEntityVertexEntry *)right_raw;
  size_t vertex;
  if (left->dimension != right->dimension) {
    return left->dimension < right->dimension ? -1 : 1;
  }
  for (vertex = 0U; vertex < 4U; ++vertex) {
    if (left->vertex_ids[vertex] != right->vertex_ids[vertex]) {
      return left->vertex_ids[vertex] < right->vertex_ids[vertex] ? -1 : 1;
    }
  }
  return 0;
}

static int ss_intermediate_index_entry_compare(const void *left_raw,
                                               const void *right_raw) {
  const SsIntermediateIndexEntry *left = (const SsIntermediateIndexEntry *)left_raw;
  const SsIntermediateIndexEntry *right = (const SsIntermediateIndexEntry *)right_raw;
  size_t vertex;
  if (left->dimension != right->dimension) {
    return left->dimension < right->dimension ? -1 : 1;
  }
  for (vertex = 0U; vertex < 3U; ++vertex) {
    if (left->vertex_ids[vertex] != right->vertex_ids[vertex]) {
      return left->vertex_ids[vertex] < right->vertex_ids[vertex] ? -1 : 1;
    }
  }
  return 0;
}

static int ss_boundary_index_entry_compare(const void *left_raw,
                                           const void *right_raw) {
  const SsBoundaryIndexEntry *left = (const SsBoundaryIndexEntry *)left_raw;
  const SsBoundaryIndexEntry *right = (const SsBoundaryIndexEntry *)right_raw;
  size_t vertex;
  for (vertex = 0U; vertex < 3U; ++vertex) {
    if (left->vertex_ids[vertex] != right->vertex_ids[vertex]) {
      return left->vertex_ids[vertex] < right->vertex_ids[vertex] ? -1 : 1;
    }
  }
  return 0;
}

SsStatus ss_mesh_point_vertex_ids(DM dm, const SsMesh *mesh,
                                  const SsChunkedLabels *stable_labels, PetscInt point,
                                  SsEntityId vertex_ids[4], size_t *vertex_count) {
  PetscInt closure_size = 0;
  PetscInt *closure = NULL;
  size_t count = 0U;
  PetscInt index;
  if (dm == NULL || vertex_ids == NULL || vertex_count == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &closure_size, &closure) !=
      PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  for (index = 0; index < closure_size; ++index) {
    const PetscInt closure_point = closure[index * 2];
    PetscInt depth = -1;
    if (DMPlexGetPointDepth(dm, closure_point, &depth) != PETSC_SUCCESS) {
      (void)DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &closure_size,
                                           &closure);
      return SS_ERROR_INTERNAL;
    }
    if (depth == 0) {
      SsEntityId stable_id = 0U;
      const SsStatus identity_status =
          mesh != NULL ? ss_mesh_cached_stable_id(mesh, closure_point, &stable_id)
          : stable_labels != NULL
              ? ss_mesh_chunked_labels_get(stable_labels, closure_point, &stable_id)
              : ss_mesh_get_chunked_id(dm, closure_point, ss_mesh_id_chunk_labels,
                                       &stable_id);
      if (count >= 4U || identity_status != SS_OK) {
        (void)DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &closure_size,
                                             &closure);
        return SS_ERROR_INTERNAL;
      }
      vertex_ids[count++] = stable_id;
    }
  }
  if (DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &closure_size, &closure) !=
      PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  ss_sort(vertex_ids, count, sizeof(vertex_ids[0]), ss_mesh_entity_id_compare);
  *vertex_count = count;
  return SS_OK;
}

SsStatus ss_mesh_build_point_id_index(DM dm, SsPointIdEntry **index, size_t *count) {
  PetscInt start = 0;
  PetscInt end = 0;
  PetscInt point;
  SsPointIdEntry *entries = NULL;
  SsChunkedLabels stable_labels = {0};
  size_t entry_count = 0U;
  if (dm == NULL || index == NULL || count == NULL ||
      DMPlexGetChart(dm, &start, &end) != PETSC_SUCCESS || end < start ||
      ss_mesh_chunked_labels_resolve(dm, ss_mesh_id_chunk_labels, &stable_labels) !=
          SS_OK) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  entries = (SsPointIdEntry *)calloc(end == start ? 1U : (size_t)(end - start),
                                     sizeof(*entries));
  if (entries == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (point = start; point < end; ++point) {
    SsEntityId stable_id = 0U;
    if (ss_mesh_chunked_labels_get(&stable_labels, point, &stable_id) != SS_OK) {
      free(entries);
      return SS_ERROR_INTEGRITY;
    }
    entries[entry_count++] = (SsPointIdEntry){stable_id, point};
  }
  ss_sort(entries, entry_count, sizeof(*entries), ss_point_id_entry_compare);
  for (point = 1; (size_t)point < entry_count; ++point) {
    if (entries[(size_t)point - 1U].stable_id == entries[(size_t)point].stable_id) {
      free(entries);
      return SS_ERROR_DUPLICATE;
    }
  }
  *index = entries;
  *count = entry_count;
  return SS_OK;
}

SsStatus ss_mesh_point_id_index_find(const SsPointIdEntry *index, size_t count,
                                     SsEntityId stable_id, PetscInt *point) {
  size_t left = 0U;
  size_t right = count;
  if ((count != 0U && index == NULL) || point == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  while (left < right) {
    const size_t middle = left + (right - left) / 2U;
    if (index[middle].stable_id < stable_id) {
      left = middle + 1U;
    } else {
      right = middle;
    }
  }
  if (left < count && index[left].stable_id == stable_id) {
    *point = index[left].point;
    return SS_OK;
  }
  return SS_ERROR_NOT_FOUND;
}

SsStatus ss_mesh_build_entity_vertex_index(DM dm, const SsMesh *mesh,
                                           SsEntityVertexEntry **index, size_t *count) {
  PetscInt start = 0;
  PetscInt end = 0;
  PetscInt point;
  SsEntityVertexEntry *entries = NULL;
  SsChunkedLabels stable_labels = {0};
  size_t entry_count = 0U;
  if (dm == NULL || index == NULL || count == NULL ||
      DMPlexGetChart(dm, &start, &end) != PETSC_SUCCESS || end < start ||
      (mesh == NULL && ss_mesh_chunked_labels_resolve(dm, ss_mesh_id_chunk_labels,
                                                      &stable_labels) != SS_OK)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  entries = (SsEntityVertexEntry *)calloc(end == start ? 1U : (size_t)(end - start),
                                          sizeof(*entries));
  if (entries == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (point = start; point < end; ++point) {
    PetscInt depth = -1;
    size_t vertex_count = 0U;
    if (DMPlexGetPointDepth(dm, point, &depth) != PETSC_SUCCESS || depth < 0 ||
        depth > (PetscInt)SS_FE_MAX_DIMENSION) {
      free(entries);
      return SS_ERROR_INTEGRITY;
    }
    entries[entry_count].dimension = (uint32_t)depth;
    entries[entry_count].point = point;
    if (depth == 0) {
      const SsStatus identity_status =
          mesh == NULL ? ss_mesh_chunked_labels_get(&stable_labels, point,
                                                    &entries[entry_count].vertex_ids[0])
                       : ss_mesh_cached_stable_id(mesh, point,
                                                  &entries[entry_count].vertex_ids[0]);
      if (identity_status != SS_OK) {
        free(entries);
        return SS_ERROR_INTEGRITY;
      }
      vertex_count = 1U;
    } else if (ss_mesh_point_vertex_ids(dm, mesh, mesh == NULL ? &stable_labels : NULL,
                                        point, entries[entry_count].vertex_ids,
                                        &vertex_count) != SS_OK) {
      free(entries);
      return SS_ERROR_INTEGRITY;
    }
    if (vertex_count != (size_t)depth + 1U) {
      free(entries);
      return SS_ERROR_INTEGRITY;
    }
    ++entry_count;
  }
  ss_sort(entries, entry_count, sizeof(*entries), ss_entity_vertex_entry_compare);
  for (point = 1; (size_t)point < entry_count; ++point) {
    if (ss_entity_vertex_entry_compare(&entries[(size_t)point - 1U],
                                       &entries[(size_t)point]) == 0) {
      free(entries);
      return SS_ERROR_DUPLICATE;
    }
  }
  *index = entries;
  *count = entry_count;
  return SS_OK;
}

SsStatus ss_mesh_entity_vertex_index_find(const SsEntityVertexEntry *index,
                                          size_t count, uint32_t dimension,
                                          const SsEntityId *vertex_ids,
                                          size_t vertex_count, PetscInt *point) {
  SsEntityVertexEntry key = {0U, {0U, 0U, 0U, 0U}, -1};
  size_t left = 0U;
  size_t right = count;
  if ((count != 0U && index == NULL) || vertex_ids == NULL || point == NULL ||
      dimension > SS_FE_MAX_DIMENSION || vertex_count != (size_t)dimension + 1U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  key.dimension = dimension;
  memcpy(key.vertex_ids, vertex_ids, vertex_count * sizeof(*vertex_ids));
  while (left < right) {
    const size_t middle = left + (right - left) / 2U;
    if (ss_entity_vertex_entry_compare(&index[middle], &key) < 0) {
      left = middle + 1U;
    } else {
      right = middle;
    }
  }
  if (left < count && ss_entity_vertex_entry_compare(&index[left], &key) == 0) {
    *point = index[left].point;
    return SS_OK;
  }
  return SS_ERROR_NOT_FOUND;
}

int ss_mesh_same_vertex_set(const SsEntityId *left, const SsEntityId *right,
                            size_t count) {
  size_t index;
  for (index = 0U; index < count; ++index) {
    if (left[index] != right[index]) {
      return 0;
    }
  }
  return 1;
}

SsStatus ss_mesh_build_intermediate_index(const SsMeshInput *input,
                                          SsIntermediateIndexEntry **index,
                                          size_t *count) {
  SsIntermediateIndexEntry *entries = NULL;
  size_t record;
  if (input == NULL || index == NULL || count == NULL ||
      input->intermediate_entity_count > SIZE_MAX / sizeof(SsIntermediateIndexEntry)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  entries = (SsIntermediateIndexEntry *)calloc(
      input->intermediate_entity_count == 0U ? 1U : input->intermediate_entity_count,
      sizeof(*entries));
  if (entries == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (record = 0U; record < input->intermediate_entity_count; ++record) {
    const SsMeshIntermediateEntity *source = &input->intermediate_entities[record];
    entries[record].dimension = source->dimension;
    memcpy(entries[record].vertex_ids, source->vertex_ids,
           sizeof(entries[record].vertex_ids));
    entries[record].stable_id = source->stable_entity_id;
    entries[record].record_index = record;
  }
  ss_sort(entries, input->intermediate_entity_count, sizeof(*entries),
          ss_intermediate_index_entry_compare);
  for (record = 1U; record < input->intermediate_entity_count; ++record) {
    if (ss_intermediate_index_entry_compare(&entries[record - 1U], &entries[record]) ==
        0) {
      free(entries);
      return SS_ERROR_DUPLICATE;
    }
  }
  *index = entries;
  *count = input->intermediate_entity_count;
  return SS_OK;
}

SsStatus ss_mesh_intermediate_index_find(const SsIntermediateIndexEntry *index,
                                         size_t count, uint32_t entity_dimension,
                                         const SsEntityId *vertex_ids,
                                         size_t vertex_count, SsEntityId *stable_id,
                                         size_t *record_index) {
  SsIntermediateIndexEntry key = {entity_dimension, {0U, 0U, 0U}, 0U, 0U};
  size_t left = 0U;
  size_t right = count;
  if ((count != 0U && index == NULL) || vertex_ids == NULL || stable_id == NULL ||
      record_index == NULL || entity_dimension == 0U ||
      entity_dimension >= SS_FE_MAX_DIMENSION ||
      vertex_count != (size_t)entity_dimension + 1U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memcpy(key.vertex_ids, vertex_ids, vertex_count * sizeof(*vertex_ids));
  while (left < right) {
    const size_t middle = left + (right - left) / 2U;
    if (ss_intermediate_index_entry_compare(&index[middle], &key) < 0) {
      left = middle + 1U;
    } else {
      right = middle;
    }
  }
  if (left < count && ss_intermediate_index_entry_compare(&index[left], &key) == 0) {
    *stable_id = index[left].stable_id;
    *record_index = index[left].record_index;
    return SS_OK;
  }
  return SS_ERROR_NOT_FOUND;
}

SsStatus ss_mesh_adjacent_vertex_id(PetscInt adjacent_count,
                                    const PetscInt *vertices_adjacent,
                                    const SsEntityId *adjacent_ids,
                                    uint64_t global_ordinal, SsEntityId *stable_id) {
  PetscInt ordinal = 0;
  PetscInt left = 0;
  PetscInt right = adjacent_count;
  if (vertices_adjacent == NULL || adjacent_ids == NULL || stable_id == NULL ||
      ss_checked_u64_to_petsc(global_ordinal, &ordinal) != SS_OK) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  while (left < right) {
    const PetscInt middle = left + (right - left) / 2;
    if (vertices_adjacent[middle] < ordinal) {
      left = middle + 1;
    } else {
      right = middle;
    }
  }
  if (left >= adjacent_count || vertices_adjacent[left] != ordinal) {
    return SS_ERROR_NOT_FOUND;
  }
  *stable_id = adjacent_ids[left];
  return SS_OK;
}

SsStatus ss_mesh_build_boundary_index(const SsMeshInput *input, PetscInt adjacent_count,
                                      const PetscInt *vertices_adjacent,
                                      const SsEntityId *adjacent_ids,
                                      SsBoundaryIndexEntry **index, size_t *count) {
  SsBoundaryIndexEntry *entries = NULL;
  size_t boundary;
  if (input == NULL || vertices_adjacent == NULL || adjacent_ids == NULL ||
      index == NULL || count == NULL ||
      input->boundary_facet_count > SIZE_MAX / sizeof(*entries)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  entries = (SsBoundaryIndexEntry *)calloc(
      input->boundary_facet_count == 0U ? 1U : input->boundary_facet_count,
      sizeof(*entries));
  if (entries == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (boundary = 0U; boundary < input->boundary_facet_count; ++boundary) {
    size_t vertex;
    for (vertex = 0U; vertex < (size_t)input->dimension; ++vertex) {
      if (ss_mesh_adjacent_vertex_id(
              adjacent_count, vertices_adjacent, adjacent_ids,
              input->boundary_facets[boundary * (size_t)input->dimension + vertex],
              &entries[boundary].vertex_ids[vertex]) != SS_OK) {
        free(entries);
        return SS_ERROR_INTEGRITY;
      }
    }
    ss_sort(entries[boundary].vertex_ids, (size_t)input->dimension,
            sizeof(entries[boundary].vertex_ids[0]), ss_mesh_entity_id_compare);
    entries[boundary].stable_id = input->boundary_facet_ids[boundary];
  }
  ss_sort(entries, input->boundary_facet_count, sizeof(*entries),
          ss_boundary_index_entry_compare);
  for (boundary = 1U; boundary < input->boundary_facet_count; ++boundary) {
    if (ss_boundary_index_entry_compare(&entries[boundary - 1U], &entries[boundary]) ==
        0) {
      free(entries);
      return SS_ERROR_DUPLICATE;
    }
  }
  *index = entries;
  *count = input->boundary_facet_count;
  return SS_OK;
}

SsStatus ss_mesh_boundary_index_find(const SsBoundaryIndexEntry *index, size_t count,
                                     const SsEntityId *vertex_ids, size_t vertex_count,
                                     SsEntityId *stable_id) {
  SsBoundaryIndexEntry key = {{0U, 0U, 0U}, 0U};
  size_t left = 0U;
  size_t right = count;
  if ((count != 0U && index == NULL) || vertex_ids == NULL || stable_id == NULL ||
      vertex_count < 2U || vertex_count > 3U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memcpy(key.vertex_ids, vertex_ids, vertex_count * sizeof(*vertex_ids));
  while (left < right) {
    const size_t middle = left + (right - left) / 2U;
    if (ss_boundary_index_entry_compare(&index[middle], &key) < 0) {
      left = middle + 1U;
    } else {
      right = middle;
    }
  }
  if (left < count && ss_boundary_index_entry_compare(&index[left], &key) == 0) {
    *stable_id = index[left].stable_id;
    return SS_OK;
  }
  return SS_ERROR_NOT_FOUND;
}
