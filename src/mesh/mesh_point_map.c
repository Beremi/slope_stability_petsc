/* Cache and query stable point identities and ghost metadata.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "mesh_internal.h"
#include "slope_internal/core/sort.h"

#include <petscdm.h>
#include <petscdmplex.h>
#include <petscsection.h>
#include <petscsf.h>

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

void ss_mesh_clear_metadata_cache(SsMesh *mesh) {
  if (mesh == NULL) {
    return;
  }
  free(mesh->cached_stable_ids);
  free(mesh->cached_kinds);
  free(mesh->cached_point_valid);
  free(mesh->cached_point_ghost);
  free(mesh->cached_cell_vertex_ids);
  free(mesh->cached_cell_vertex_counts);
  free(mesh->cached_coordinate_node_ids);
  free(mesh->cached_coordinate_node_counts);
  mesh->cached_stable_ids = NULL;
  mesh->cached_kinds = NULL;
  mesh->cached_point_valid = NULL;
  mesh->cached_point_ghost = NULL;
  mesh->cached_cell_vertex_ids = NULL;
  mesh->cached_cell_vertex_counts = NULL;
  mesh->cached_coordinate_node_ids = NULL;
  mesh->cached_coordinate_node_counts = NULL;
  mesh->cache_chart_start = 0;
  mesh->cache_chart_end = 0;
}

SsStatus ss_mesh_ghost_flags_fill(DM dm, PetscInt chart_start, PetscInt chart_end,
                                  unsigned char *ghost_flags) {
  PetscSF sf = NULL;
  PetscInt leaf_count = 0;
  const PetscInt *local = NULL;
  PetscInt leaf;
  if (dm == NULL || chart_end < chart_start || ghost_flags == NULL ||
      DMGetPointSF(dm, &sf) != PETSC_SUCCESS ||
      PetscSFGetGraph(sf, NULL, &leaf_count, &local, NULL) != PETSC_SUCCESS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (leaf_count < 0) {
    return SS_OK;
  }
  for (leaf = 0; leaf < leaf_count; ++leaf) {
    const PetscInt point = local == NULL ? leaf : local[leaf];
    if (point < chart_start || point >= chart_end) {
      return SS_ERROR_INTEGRITY;
    }
    ghost_flags[(size_t)(point - chart_start)] = 1U;
  }
  return SS_OK;
}

SsStatus ss_mesh_rebuild_point_map(SsMesh *mesh) {
  PetscInt start = 0;
  PetscInt end = 0;
  PetscInt point;
  size_t count = 0U;
  size_t chart_count;
  SsPointEntry *entries = NULL;
  SsEntityId *stable_ids = NULL;
  SsMeshEntityKind *kinds = NULL;
  unsigned char *valid = NULL;
  unsigned char *ghost = NULL;
  SsEntityId *cell_vertex_ids = NULL;
  unsigned char *cell_vertex_counts = NULL;
  SsEntityId *coordinate_node_ids = NULL;
  unsigned char *coordinate_node_counts = NULL;
  PetscSection coordinate_section = NULL;
  SsChunkedLabels stable_labels = {0};
  SsChunkedLabels cell_vertex_labels[4] = {0};
  SsChunkedLabels coordinate_node_labels[SS_COORDINATE_SLOT_COUNT] = {0};
  unsigned char coordinate_labels_resolved[SS_COORDINATE_SLOT_COUNT] = {0U, 0U, 0U};
  DMLabel kind_label = NULL;
  SsStatus status = SS_OK;
  if (mesh == NULL || DMPlexGetChart(mesh->dm, &start, &end) != PETSC_SUCCESS ||
      end < start ||
      DMGetCoordinateSection(mesh->dm, &coordinate_section) != PETSC_SUCCESS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  chart_count = (size_t)(end - start);
  if (chart_count > SIZE_MAX / (4U * sizeof(*cell_vertex_ids)) ||
      chart_count >
          SIZE_MAX / (SS_COORDINATE_SLOT_COUNT * sizeof(*coordinate_node_ids))) {
    return SS_ERROR_OVERFLOW;
  }
  entries =
      (SsPointEntry *)calloc(chart_count == 0U ? 1U : chart_count, sizeof(*entries));
  stable_ids =
      (SsEntityId *)calloc(chart_count == 0U ? 1U : chart_count, sizeof(*stable_ids));
  kinds =
      (SsMeshEntityKind *)calloc(chart_count == 0U ? 1U : chart_count, sizeof(*kinds));
  valid = (unsigned char *)calloc(chart_count == 0U ? 1U : chart_count, sizeof(*valid));
  ghost = (unsigned char *)calloc(chart_count == 0U ? 1U : chart_count, sizeof(*ghost));
  cell_vertex_ids = (SsEntityId *)calloc(chart_count == 0U ? 1U : chart_count * 4U,
                                         sizeof(*cell_vertex_ids));
  cell_vertex_counts = (unsigned char *)calloc(chart_count == 0U ? 1U : chart_count,
                                               sizeof(*cell_vertex_counts));
  coordinate_node_ids = (SsEntityId *)calloc(
      chart_count == 0U ? 1U : chart_count * SS_COORDINATE_SLOT_COUNT,
      sizeof(*coordinate_node_ids));
  coordinate_node_counts = (unsigned char *)calloc(chart_count == 0U ? 1U : chart_count,
                                                   sizeof(*coordinate_node_counts));
  if (entries == NULL || stable_ids == NULL || kinds == NULL || valid == NULL ||
      ghost == NULL || cell_vertex_ids == NULL || cell_vertex_counts == NULL ||
      coordinate_node_ids == NULL || coordinate_node_counts == NULL) {
    free(entries);
    free(stable_ids);
    free(kinds);
    free(valid);
    free(ghost);
    free(cell_vertex_ids);
    free(cell_vertex_counts);
    free(coordinate_node_ids);
    free(coordinate_node_counts);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  status = ss_mesh_ghost_flags_fill(mesh->dm, start, end, ghost);
  if (status == SS_OK) {
    status = ss_mesh_chunked_labels_resolve(mesh->dm, ss_mesh_id_chunk_labels,
                                            &stable_labels);
  }
  if (status == SS_OK &&
      (DMGetLabel(mesh->dm, "ss-entity-kind", &kind_label) != PETSC_SUCCESS ||
       (kind_label == NULL && chart_count != 0U))) {
    status = SS_ERROR_NOT_FOUND;
  }
  for (uint32_t slot = 0U; status == SS_OK && slot <= mesh->dimension; ++slot) {
    char storage[SS_ID_CHUNK_COUNT][48];
    const char *names[SS_ID_CHUNK_COUNT];
    ss_mesh_slot_label_names("ss-cell-vertex", slot, storage, names);
    status = ss_mesh_chunked_labels_resolve(mesh->dm, names, &cell_vertex_labels[slot]);
  }
  for (point = start; status == SS_OK && point < end; ++point) {
    SsEntityId stable_id = 0U;
    PetscInt kind = -1;
    PetscInt coordinate_dof = 0;
    const size_t offset = (size_t)(point - start);
    if (ss_mesh_chunked_labels_get(&stable_labels, point, &stable_id) == SS_OK &&
        DMLabelGetValue(kind_label, point, &kind) == PETSC_SUCCESS &&
        kind >= (PetscInt)SS_MESH_ENTITY_VERTEX &&
        kind <= (PetscInt)SS_MESH_ENTITY_CELL) {
      uint32_t slot;
      entries[count++] = (SsPointEntry){stable_id, (SsMeshEntityKind)kind, point};
      stable_ids[offset] = stable_id;
      kinds[offset] = (SsMeshEntityKind)kind;
      valid[offset] = 1U;
      if ((SsMeshEntityKind)kind == SS_MESH_ENTITY_CELL) {
        for (slot = 0U; slot <= mesh->dimension; ++slot) {
          if (ss_mesh_chunked_labels_get(&cell_vertex_labels[slot], point,
                                         &cell_vertex_ids[offset * 4U + slot]) !=
              SS_OK) {
            status = SS_ERROR_INTEGRITY;
            break;
          }
        }
        cell_vertex_counts[offset] = (unsigned char)(mesh->dimension + 1U);
      }
      if (status == SS_OK &&
          (PetscSectionGetDof(coordinate_section, point, &coordinate_dof) !=
               PETSC_SUCCESS ||
           coordinate_dof < 0 || coordinate_dof % (PetscInt)mesh->dimension != 0 ||
           coordinate_dof / (PetscInt)mesh->dimension >
               (PetscInt)SS_COORDINATE_SLOT_COUNT)) {
        status = SS_ERROR_INTEGRITY;
      }
      if (status == SS_OK) {
        const uint32_t coordinate_slots =
            (uint32_t)(coordinate_dof / (PetscInt)mesh->dimension);
        for (slot = 0U; slot < coordinate_slots; ++slot) {
          if (coordinate_labels_resolved[slot] == 0U) {
            char storage[SS_ID_CHUNK_COUNT][48];
            const char *names[SS_ID_CHUNK_COUNT];
            ss_mesh_slot_label_names("ss-coordinate-node", slot, storage, names);
            status = ss_mesh_chunked_labels_resolve(mesh->dm, names,
                                                    &coordinate_node_labels[slot]);
            if (status == SS_OK) {
              coordinate_labels_resolved[slot] = 1U;
            }
          }
          if (status != SS_OK ||
              ss_mesh_chunked_labels_get(
                  &coordinate_node_labels[slot], point,
                  &coordinate_node_ids[offset * SS_COORDINATE_SLOT_COUNT + slot]) !=
                  SS_OK) {
            status = SS_ERROR_INTEGRITY;
            break;
          }
        }
        coordinate_node_counts[offset] = (unsigned char)coordinate_slots;
      }
    }
  }
  if (status != SS_OK) {
    free(entries);
    free(stable_ids);
    free(kinds);
    free(valid);
    free(ghost);
    free(cell_vertex_ids);
    free(cell_vertex_counts);
    free(coordinate_node_ids);
    free(coordinate_node_counts);
    return status;
  }
  ss_sort(entries, count, sizeof(*entries), ss_point_entry_compare);
  {
    size_t index;
    for (index = 1U; index < count; ++index) {
      if (entries[index - 1U].kind == entries[index].kind &&
          entries[index - 1U].stable_id == entries[index].stable_id) {
        free(entries);
        free(stable_ids);
        free(kinds);
        free(valid);
        free(ghost);
        free(cell_vertex_ids);
        free(cell_vertex_counts);
        free(coordinate_node_ids);
        free(coordinate_node_counts);
        return SS_ERROR_DUPLICATE;
      }
    }
  }
  free(mesh->points);
  ss_mesh_clear_metadata_cache(mesh);
  mesh->points = entries;
  mesh->point_count = count;
  mesh->cache_chart_start = start;
  mesh->cache_chart_end = end;
  mesh->cached_stable_ids = stable_ids;
  mesh->cached_kinds = kinds;
  mesh->cached_point_valid = valid;
  mesh->cached_point_ghost = ghost;
  mesh->cached_cell_vertex_ids = cell_vertex_ids;
  mesh->cached_cell_vertex_counts = cell_vertex_counts;
  mesh->cached_coordinate_node_ids = coordinate_node_ids;
  mesh->cached_coordinate_node_counts = coordinate_node_counts;
  return SS_OK;
}

SsStatus ss_mesh_rebuild_entity_map(SsMesh *mesh) {
  SsEntityVertexEntry *entries = NULL;
  size_t count = 0U;
  SsStatus status;
  if (mesh == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_mesh_build_entity_vertex_index(mesh->dm, mesh, &entries, &count);
  if (status != SS_OK) {
    free(entries);
    return status;
  }
  free(mesh->entities);
  mesh->entities = entries;
  mesh->entity_count = count;
  return SS_OK;
}

int ss_mesh_point_is_ghost(const SsMesh *mesh, DM dm, PetscInt point) {
  if (mesh != NULL && mesh->cached_point_ghost != NULL &&
      point >= mesh->cache_chart_start && point < mesh->cache_chart_end) {
    return mesh->cached_point_ghost[(size_t)(point - mesh->cache_chart_start)] != 0U;
  }
  PetscSF sf = NULL;
  PetscInt leaf_count = 0;
  const PetscInt *local = NULL;
  PetscInt leaf;
  if (DMGetPointSF(dm, &sf) != PETSC_SUCCESS ||
      PetscSFGetGraph(sf, NULL, &leaf_count, &local, NULL) != PETSC_SUCCESS ||
      leaf_count < 0) {
    return 0;
  }
  for (leaf = 0; leaf < leaf_count; ++leaf) {
    const PetscInt leaf_point = local == NULL ? leaf : local[leaf];
    if (leaf_point == point) {
      return 1;
    }
  }
  return 0;
}

SsStatus ss_mesh_point_from_stable_id(const SsMesh *mesh, SsMeshEntityKind kind,
                                      SsEntityId stable_id, PetscInt *point) {
  size_t left = 0U;
  size_t right;
  if (mesh == NULL || point == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  right = mesh->point_count;
  while (left < right) {
    const size_t middle = left + (right - left) / 2U;
    const SsPointEntry *entry = &mesh->points[middle];
    if (entry->kind < kind || (entry->kind == kind && entry->stable_id < stable_id)) {
      left = middle + 1U;
    } else {
      right = middle;
    }
  }
  if (left < mesh->point_count && mesh->points[left].kind == kind &&
      mesh->points[left].stable_id == stable_id) {
    *point = mesh->points[left].point;
    return SS_OK;
  }
  return SS_ERROR_NOT_FOUND;
}

SsStatus ss_mesh_point_from_stable_id_any(const SsMesh *mesh, SsEntityId stable_id,
                                          PetscInt *point) {
  const SsMeshEntityKind kinds[] = {
      SS_MESH_ENTITY_VERTEX,
      SS_MESH_ENTITY_EDGE,
      SS_MESH_ENTITY_FACET,
      SS_MESH_ENTITY_CELL,
  };
  size_t kind;
  if (mesh == NULL || point == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (kind = 0U; kind < sizeof(kinds) / sizeof(kinds[0]); ++kind) {
    if (ss_mesh_point_from_stable_id(mesh, kinds[kind], stable_id, point) == SS_OK) {
      return SS_OK;
    }
  }
  return SS_ERROR_NOT_FOUND;
}

SsStatus ss_mesh_stable_id_from_point(const SsMesh *mesh, PetscInt point,
                                      SsEntityId *stable_id) {
  if (mesh == NULL || stable_id == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return ss_mesh_cached_stable_id(mesh, point, stable_id);
}

SsStatus ss_mesh_cell_point(const SsMesh *mesh, SsEntityId stable_cell_id,
                            PetscInt *point) {
  return ss_mesh_point_from_stable_id(mesh, SS_MESH_ENTITY_CELL, stable_cell_id, point);
}
