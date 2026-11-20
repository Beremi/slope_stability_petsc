/* Attach stable IDs and semantic supports to topology.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "mesh_internal.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/core/sort.h"

#include <petscdm.h>
#include <petscdmplex.h>
#include <petsclog.h>
#include <petscsf.h>

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

SsStatus ss_mesh_attach_vertex_ids(DM dm, PetscSF vertex_sf, const SsMeshInput *input,
                                   SsEntityId **adjacent_ids,
                                   PetscInt *adjacent_count) {
  MPI_Comm communicator = PetscObjectComm((PetscObject)dm);
  PetscInt roots = 0;
  PetscInt leaves = 0;
  PetscInt vertex_start = 0;
  PetscInt vertex_end = 0;
  SsEntityId root_dummy = 0U;
  SsEntityId *ids = NULL;
  PetscInt vertex;
  SsStatus status = SS_OK;
  if (dm == NULL || vertex_sf == NULL || input == NULL || adjacent_ids == NULL ||
      adjacent_count == NULL) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else if (PetscSFGetGraph(vertex_sf, &roots, &leaves, NULL, NULL) != PETSC_SUCCESS ||
             DMPlexGetDepthStratum(dm, 0, &vertex_start, &vertex_end) !=
                 PETSC_SUCCESS ||
             roots < 0 || leaves < 0 ||
             (uint64_t)roots != (uint64_t)input->vertex_count ||
             vertex_end - vertex_start != leaves) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    return status;
  }
  ids = (SsEntityId *)calloc(leaves == 0 ? 1U : (size_t)leaves, sizeof(*ids));
  if (ids == NULL) {
    status = SS_ERROR_OUT_OF_MEMORY;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    free(ids);
    return status;
  }
  if (PetscSFBcastBegin(vertex_sf, MPI_UINT64_T,
                        input->vertex_count == 0U ? &root_dummy : input->vertex_ids,
                        ids, MPI_REPLACE) != PETSC_SUCCESS ||
      PetscSFBcastEnd(vertex_sf, MPI_UINT64_T,
                      input->vertex_count == 0U ? &root_dummy : input->vertex_ids, ids,
                      MPI_REPLACE) != PETSC_SUCCESS) {
    free(ids);
    return SS_ERROR_INTERNAL;
  }
  for (vertex = 0; vertex < leaves; ++vertex) {
    if (ss_mesh_set_point_id(dm, vertex_start + vertex, ids[vertex],
                             SS_MESH_ENTITY_VERTEX) != SS_OK) {
      free(ids);
      return SS_ERROR_INTERNAL;
    }
  }
  *adjacent_ids = ids;
  *adjacent_count = leaves;
  return SS_OK;
}

SsStatus ss_mesh_attach_cells(DM dm, const SsMeshInput *input, PetscInt adjacent_count,
                              const PetscInt *vertices_adjacent,
                              const SsEntityId *adjacent_ids) {
  PetscInt cell_start = 0;
  PetscInt cell_end = 0;
  size_t cell;
  if (DMPlexGetHeightStratum(dm, 0, &cell_start, &cell_end) != PETSC_SUCCESS ||
      (uint64_t)(cell_end - cell_start) != (uint64_t)input->cell_count) {
    return SS_ERROR_INTERNAL;
  }
  for (cell = 0U; cell < input->cell_count; ++cell) {
    const PetscInt point = cell_start + (PetscInt)cell;
    size_t vertex;
    if (ss_mesh_set_point_id(dm, point, input->cell_ids[cell], SS_MESH_ENTITY_CELL) !=
        SS_OK) {
      return SS_ERROR_INTERNAL;
    }
    for (vertex = 0U; vertex <= (size_t)input->dimension; ++vertex) {
      SsEntityId vertex_id = 0U;
      if (ss_mesh_adjacent_vertex_id(
              adjacent_count, vertices_adjacent, adjacent_ids,
              input->cells[cell * ((size_t)input->dimension + 1U) + vertex],
              &vertex_id) != SS_OK ||
          ss_mesh_set_slot_id(dm, point, "ss-cell-vertex", (uint32_t)vertex,
                              vertex_id) != SS_OK) {
        return SS_ERROR_INTEGRITY;
      }
    }
  }
  return SS_OK;
}

SsStatus ss_mesh_attach_intermediate_ids(DM dm, const SsMeshInput *input,
                                         PetscInt adjacent_count,
                                         const PetscInt *vertices_adjacent,
                                         const SsEntityId *adjacent_ids) {
  PetscInt start = 0;
  PetscInt end = 0;
  PetscInt point;
  unsigned char *matched = NULL;
  SsIntermediateIndexEntry *intermediate_index = NULL;
  size_t intermediate_count = 0U;
  SsBoundaryIndexEntry *boundary_index = NULL;
  SsChunkedLabels stable_labels = {0};
  size_t boundary_count = 0U;
  SsStatus status = SS_OK;
  if (DMPlexGetChart(dm, &start, &end) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (ss_mesh_create_label(dm, "ss-explicit-boundary") != SS_OK ||
      ss_mesh_chunked_labels_resolve(dm, ss_mesh_id_chunk_labels, &stable_labels) !=
          SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  status =
      ss_mesh_build_intermediate_index(input, &intermediate_index, &intermediate_count);
  if (status == SS_OK) {
    status =
        ss_mesh_build_boundary_index(input, adjacent_count, vertices_adjacent,
                                     adjacent_ids, &boundary_index, &boundary_count);
  }
  if (status != SS_OK) {
    free(intermediate_index);
    free(boundary_index);
    return status;
  }
  matched = (unsigned char *)calloc(
      input->intermediate_entity_count == 0U ? 1U : input->intermediate_entity_count,
      sizeof(*matched));
  if (matched == NULL) {
    free(intermediate_index);
    free(boundary_index);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (point = start; point < end; ++point) {
    PetscInt depth = -1;
    SsEntityId existing = 0U;
    if (DMPlexGetPointDepth(dm, point, &depth) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
      break;
    }
    if (ss_mesh_chunked_labels_get(&stable_labels, point, &existing) == SS_OK) {
      continue;
    }
    if (depth > 0 && depth < (PetscInt)input->dimension) {
      SsEntityId vertex_ids[4] = {0U, 0U, 0U, 0U};
      size_t vertex_count = 0U;
      SsMeshEntityKind kind = input->dimension == 3U && depth == 2
                                  ? SS_MESH_ENTITY_FACET
                                  : SS_MESH_ENTITY_EDGE;
      SsEntityId stable_id = 0U;
      size_t record_index = 0U;
      SsEntityId boundary_stable_id = 0U;
      if (ss_mesh_point_vertex_ids(dm, NULL, &stable_labels, point, vertex_ids,
                                   &vertex_count) != SS_OK ||
          ss_mesh_intermediate_index_find(intermediate_index, intermediate_count,
                                          (uint32_t)depth, vertex_ids, vertex_count,
                                          &stable_id, &record_index) != SS_OK) {
        status = SS_ERROR_INTEGRITY;
        break;
      }
      matched[record_index] = 1U;
      if ((uint32_t)depth == input->dimension - 1U) {
        const SsStatus boundary_status =
            ss_mesh_boundary_index_find(boundary_index, boundary_count, vertex_ids,
                                        vertex_count, &boundary_stable_id);
        if (boundary_status == SS_OK) {
          if (stable_id != boundary_stable_id) {
            status = SS_ERROR_INTEGRITY;
            break;
          }
          kind = SS_MESH_ENTITY_FACET;
          if (DMSetLabelValue(dm, "ss-explicit-boundary", point, 1) != PETSC_SUCCESS) {
            status = SS_ERROR_INTERNAL;
          }
        } else if (boundary_status != SS_ERROR_NOT_FOUND) {
          status = boundary_status;
        }
      }
      if (status != SS_OK) {
        break;
      }
      if (ss_mesh_set_point_id(dm, point, stable_id, kind) != SS_OK) {
        status = SS_ERROR_INTERNAL;
        break;
      }
    }
  }
  if (status == SS_OK) {
    size_t record;
    for (record = 0U; record < input->intermediate_entity_count; ++record) {
      if (matched[record] == 0U) {
        status = SS_ERROR_INTEGRITY;
        break;
      }
    }
  }
  free(matched);
  free(intermediate_index);
  free(boundary_index);
  return status;
}

static int ss_mesh_label_key_compare(const void *left, const void *right) {
  const uint64_t a = *(const uint64_t *)left;
  const uint64_t b = *(const uint64_t *)right;
  return (a > b) - (a < b);
}

static uint64_t ss_mesh_label_key(const SsMeshLabelAssignment *assignment) {
  const uint32_t value =
      assignment->kind == SS_MESH_LABEL_MATERIAL ? 0U : assignment->value;
  return ((uint64_t)(uint32_t)assignment->kind << 32U) | value;
}

static SsStatus ss_mesh_register_semantic_labels(DM dm, const SsMeshInput *input) {
  const MPI_Comm communicator = PetscObjectComm((PetscObject)dm);
  uint64_t *keys =
      calloc(input->label_count == 0U ? 1U : input->label_count, sizeof(*keys));
  SsStatus status = keys == NULL ? SS_ERROR_OUT_OF_MEMORY : SS_OK;
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    free(keys);
    return status;
  }
  for (size_t index = 0U; index < input->label_count; ++index) {
    keys[index] = ss_mesh_label_key(&input->labels[index]);
  }
  ss_sort(keys, input->label_count, sizeof(*keys), ss_mesh_label_key_compare);
  size_t next = 0U;
  while (status == SS_OK) {
    const uint64_t local = next == input->label_count ? UINT64_MAX : keys[next];
    uint64_t key = 0U;
    if (MPI_Allreduce(&local, &key, 1, MPI_UINT64_T, MPI_MIN, communicator) !=
        MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
      break;
    }
    if (key == UINT64_MAX) {
      break;
    }
    while (next < input->label_count && keys[next] == key) {
      ++next;
    }
    const SsMeshLabelKind kind = (SsMeshLabelKind)(uint32_t)(key >> 32U);
    const char *name = ss_mesh_semantic_label_name(kind);
    char membership_name[64];
    if (kind == SS_MESH_LABEL_SUPPORT || kind == SS_MESH_LABEL_INTERFACE) {
      status = ss_mesh_semantic_membership_name(kind, (uint32_t)key, membership_name);
      name = membership_name;
    }
    if (status == SS_OK && (name == NULL || ss_mesh_create_label(dm, name) != SS_OK)) {
      status = SS_ERROR_INTEGRITY;
    }
    status = ss_status_allreduce_max(communicator, status);
  }
  free(keys);
  return status;
}

SsStatus ss_mesh_attach_semantic_labels(DM dm, const SsMeshInput *input,
                                        const SsPointIdEntry *point_index,
                                        size_t point_count) {
  SsStatus status = ss_mesh_register_semantic_labels(dm, input);
  for (size_t index = 0U; status == SS_OK && index < input->label_count; ++index) {
    const SsMeshLabelAssignment *label = &input->labels[index];
    const char *name = ss_mesh_semantic_label_name(label->kind);
    char membership_name[64];
    PetscInt point = -1;
    PetscInt value = 1;
    if (label->kind == SS_MESH_LABEL_SUPPORT ||
        label->kind == SS_MESH_LABEL_INTERFACE) {
      status =
          ss_mesh_semantic_membership_name(label->kind, label->value, membership_name);
      name = membership_name;
    } else if (name == NULL || ss_checked_u64_to_petsc(label->value, &value) != SS_OK) {
      status = SS_ERROR_INTEGRITY;
    }
    if (status == SS_OK &&
        (ss_mesh_point_id_index_find(point_index, point_count, label->stable_entity_id,
                                     &point) != SS_OK ||
         DMSetLabelValue(dm, name, point, value) != PETSC_SUCCESS)) {
      status = SS_ERROR_INTEGRITY;
    }
  }
  return ss_status_allreduce_max(PetscObjectComm((PetscObject)dm), status);
}
