/* Semantic labels and chunked stable IDs on DMPlex points.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "mesh_internal.h"
#include "slope_internal/core/collective.h"

#include <petscdm.h>
#include <petscdmplex.h>
#include <petsclog.h>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define SS_ID_CHUNK_BITS 8U
#define SS_ID_CHUNK_MASK UINT64_C(0xff)

const char *const ss_mesh_id_chunk_labels[SS_ID_CHUNK_COUNT] = {
    "ss-stable-id-byte0", "ss-stable-id-byte1", "ss-stable-id-byte2",
    "ss-stable-id-byte3", "ss-stable-id-byte4", "ss-stable-id-byte5",
    "ss-stable-id-byte6", "ss-stable-id-byte7",
};

const char *ss_mesh_semantic_label_name(SsMeshLabelKind kind) {
  switch (kind) {
  case SS_MESH_LABEL_MATERIAL:
    return "ss-material";
  default:
    return NULL;
  }
}

static const char *ss_semantic_membership_prefix(SsMeshLabelKind kind) {
  switch (kind) {
  case SS_MESH_LABEL_SUPPORT:
    return "ss-support-value-";
  case SS_MESH_LABEL_INTERFACE:
    return "ss-interface-value-";
  default:
    return NULL;
  }
}

SsStatus ss_mesh_semantic_membership_name(SsMeshLabelKind kind, uint32_t value,
                                          char name[64]) {
  const char *prefix = ss_semantic_membership_prefix(kind);
  int written;
  if (prefix == NULL || name == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  written = snprintf(name, 64U, "%s%u", prefix, value);
  return written < 0 || written >= 64 ? SS_ERROR_LIMIT : SS_OK;
}

int ss_mesh_semantic_membership_parse(const char *name, SsMeshLabelKind *kind,
                                      uint32_t *value) {
  const SsMeshLabelKind kinds[] = {
      SS_MESH_LABEL_SUPPORT,
      SS_MESH_LABEL_INTERFACE,
  };
  size_t kind_index;
  if (name == NULL || kind == NULL || value == NULL) {
    return 0;
  }
  for (kind_index = 0U; kind_index < sizeof(kinds) / sizeof(kinds[0]); ++kind_index) {
    const char *prefix = ss_semantic_membership_prefix(kinds[kind_index]);
    const size_t prefix_length = strlen(prefix);
    const char *cursor;
    uint64_t parsed = 0U;
    if (strncmp(name, prefix, prefix_length) != 0) {
      continue;
    }
    cursor = name + prefix_length;
    if (*cursor == '\0') {
      return 0;
    }
    while (*cursor != '\0') {
      const unsigned char character = (unsigned char)*cursor;
      if (character < (unsigned char)'0' || character > (unsigned char)'9') {
        return 0;
      }
      parsed = parsed * UINT64_C(10) + (uint64_t)(character - (unsigned char)'0');
      if (parsed > UINT32_MAX) {
        return 0;
      }
      ++cursor;
    }
    *kind = kinds[kind_index];
    *value = (uint32_t)parsed;
    return 1;
  }
  return 0;
}

static SsStatus ss_mesh_petsc_status(PetscErrorCode code) {
  return code == PETSC_SUCCESS ? SS_OK : SS_ERROR_INTERNAL;
}

SsStatus ss_mesh_create_label(DM dm, const char *name) {
  if (dm == NULL || name == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return ss_mesh_petsc_status(DMCreateLabel(dm, name));
}

static SsStatus ss_mesh_point_membership(DM dm, SsMeshLabelKind kind, uint32_t value,
                                         PetscInt point, int *contains) {
  char name[64];
  DMLabel label = NULL;
  PetscInt raw_value = -1;
  if (dm == NULL || contains == NULL ||
      ss_mesh_semantic_membership_name(kind, value, name) != SS_OK) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (DMGetLabel(dm, name, &label) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (label == NULL) {
    *contains = 0;
    return SS_OK;
  }
  if (DMLabelGetValue(label, point, &raw_value) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  *contains = raw_value >= 0 ? 1 : 0;
  return SS_OK;
}

SsStatus ss_mesh_point_has_any_membership(DM dm, SsMeshLabelKind kind, PetscInt point,
                                          int *contains) {
  PetscInt label_count = 0;
  PetscInt label_index;
  if (dm == NULL || ss_semantic_membership_prefix(kind) == NULL || contains == NULL ||
      DMGetNumLabels(dm, &label_count) != PETSC_SUCCESS || label_count < 0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *contains = 0;
  for (label_index = 0; label_index < label_count; ++label_index) {
    const char *name = NULL;
    SsMeshLabelKind parsed_kind = SS_MESH_LABEL_MATERIAL;
    uint32_t value = 0U;
    PetscInt raw_value = -1;
    if (DMGetLabelName(dm, label_index, &name) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    if (ss_mesh_semantic_membership_parse(name, &parsed_kind, &value) == 0 ||
        parsed_kind != kind) {
      continue;
    }
    if (DMGetLabelValue(dm, name, point, &raw_value) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    if (raw_value >= 0) {
      *contains = 1;
      return SS_OK;
    }
  }
  return SS_OK;
}

static SsStatus ss_mesh_set_chunked_id(DM dm, PetscInt point,
                                       const char *const names[SS_ID_CHUNK_COUNT],
                                       SsEntityId stable_id) {
  uint32_t chunk;
  for (chunk = 0U; chunk < SS_ID_CHUNK_COUNT; ++chunk) {
    const PetscInt value =
        (PetscInt)((stable_id >> (chunk * SS_ID_CHUNK_BITS)) & SS_ID_CHUNK_MASK);
    if (ss_mesh_create_label(dm, names[chunk]) != SS_OK ||
        DMSetLabelValue(dm, names[chunk], point, value) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
  }
  return SS_OK;
}

SsStatus ss_mesh_get_chunked_id(DM dm, PetscInt point,
                                const char *const names[SS_ID_CHUNK_COUNT],
                                SsEntityId *stable_id) {
  SsEntityId result = 0U;
  uint32_t chunk;
  if (dm == NULL || stable_id == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (chunk = 0U; chunk < SS_ID_CHUNK_COUNT; ++chunk) {
    PetscInt value = -1;
    if (DMGetLabelValue(dm, names[chunk], point, &value) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    if (value < 0) {
      return SS_ERROR_NOT_FOUND;
    }
    result |= (SsEntityId)(uint64_t)value << (chunk * SS_ID_CHUNK_BITS);
  }
  *stable_id = result;
  return SS_OK;
}

SsStatus ss_mesh_chunked_labels_resolve(DM dm,
                                        const char *const names[SS_ID_CHUNK_COUNT],
                                        SsChunkedLabels *labels) {
  uint32_t chunk;
  if (dm == NULL || names == NULL || labels == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (chunk = 0U; chunk < SS_ID_CHUNK_COUNT; ++chunk) {
    if (DMGetLabel(dm, names[chunk], &labels->chunks[chunk]) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
  }
  return SS_OK;
}

SsStatus ss_mesh_chunked_labels_get(const SsChunkedLabels *labels, PetscInt point,
                                    SsEntityId *stable_id) {
  SsEntityId result = 0U;
  uint32_t chunk;
  if (labels == NULL || stable_id == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (chunk = 0U; chunk < SS_ID_CHUNK_COUNT; ++chunk) {
    PetscInt value = -1;
    if (labels->chunks[chunk] == NULL) {
      return SS_ERROR_NOT_FOUND;
    }
    if (DMLabelGetValue(labels->chunks[chunk], point, &value) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    if (value < 0) {
      return SS_ERROR_NOT_FOUND;
    }
    result |= (SsEntityId)(uint64_t)value << (chunk * SS_ID_CHUNK_BITS);
  }
  *stable_id = result;
  return SS_OK;
}

SsStatus ss_mesh_set_point_id(DM dm, PetscInt point, SsEntityId stable_id,
                              SsMeshEntityKind kind) {
  if (ss_mesh_set_chunked_id(dm, point, ss_mesh_id_chunk_labels, stable_id) != SS_OK ||
      ss_mesh_create_label(dm, "ss-entity-kind") != SS_OK ||
      DMSetLabelValue(dm, "ss-entity-kind", point, (PetscInt)kind) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  return SS_OK;
}

void ss_mesh_slot_label_names(const char *prefix, uint32_t slot,
                              char storage[SS_ID_CHUNK_COUNT][48],
                              const char *names[SS_ID_CHUNK_COUNT]) {
  uint32_t chunk;
  for (chunk = 0U; chunk < SS_ID_CHUNK_COUNT; ++chunk) {
    (void)snprintf(storage[chunk], 48U, "%s-%u-%u", prefix, slot, chunk);
    names[chunk] = storage[chunk];
  }
}

SsStatus ss_mesh_set_slot_id(DM dm, PetscInt point, const char *prefix, uint32_t slot,
                             SsEntityId stable_id) {
  char storage[SS_ID_CHUNK_COUNT][48];
  const char *names[SS_ID_CHUNK_COUNT];
  ss_mesh_slot_label_names(prefix, slot, storage, names);
  return ss_mesh_set_chunked_id(dm, point, names, stable_id);
}

SsStatus ss_mesh_get_slot_id(DM dm, PetscInt point, const char *prefix, uint32_t slot,
                             SsEntityId *stable_id) {
  char storage[SS_ID_CHUNK_COUNT][48];
  const char *names[SS_ID_CHUNK_COUNT];
  ss_mesh_slot_label_names(prefix, slot, storage, names);
  return ss_mesh_get_chunked_id(dm, point, names, stable_id);
}

SsStatus ss_mesh_label_value(const SsMesh *mesh, SsMeshLabelKind kind,
                             SsEntityId stable_id, uint32_t *value) {
  PetscInt point = -1;
  if (mesh == NULL || value == NULL ||
      (kind != SS_MESH_LABEL_MATERIAL && ss_semantic_membership_prefix(kind) == NULL)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (ss_mesh_point_from_stable_id_any(mesh, stable_id, &point) != SS_OK) {
    return SS_ERROR_NOT_FOUND;
  }
  if (kind == SS_MESH_LABEL_MATERIAL) {
    PetscInt raw_value = -1;
    if (DMGetLabelValue(mesh->dm, ss_mesh_semantic_label_name(kind), point,
                        &raw_value) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    if (raw_value < 0) {
      return SS_ERROR_NOT_FOUND;
    }
    *value = (uint32_t)raw_value;
    return SS_OK;
  }
  {
    PetscInt label_count = 0;
    PetscInt label_index;
    int found = 0;
    uint32_t selected = UINT32_MAX;
    if (DMGetNumLabels(mesh->dm, &label_count) != PETSC_SUCCESS || label_count < 0) {
      return SS_ERROR_INTERNAL;
    }
    for (label_index = 0; label_index < label_count; ++label_index) {
      const char *name = NULL;
      SsMeshLabelKind parsed_kind = SS_MESH_LABEL_MATERIAL;
      uint32_t parsed_value = 0U;
      PetscInt raw_value = -1;
      if (DMGetLabelName(mesh->dm, label_index, &name) != PETSC_SUCCESS) {
        return SS_ERROR_INTERNAL;
      }
      if (ss_mesh_semantic_membership_parse(name, &parsed_kind, &parsed_value) == 0 ||
          parsed_kind != kind) {
        continue;
      }
      if (DMGetLabelValue(mesh->dm, name, point, &raw_value) != PETSC_SUCCESS) {
        return SS_ERROR_INTERNAL;
      }
      if (raw_value >= 0 && (!found || parsed_value < selected)) {
        selected = parsed_value;
        found = 1;
      }
    }
    if (!found) {
      return SS_ERROR_NOT_FOUND;
    }
    *value = selected;
    return SS_OK;
  }
}

SsStatus ss_mesh_label_contains(const SsMesh *mesh, SsMeshLabelKind kind,
                                SsEntityId stable_id, uint32_t value, int *contains) {
  PetscInt point = -1;
  if (mesh == NULL || contains == NULL ||
      (kind != SS_MESH_LABEL_MATERIAL && ss_semantic_membership_prefix(kind) == NULL)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *contains = 0;
  if (ss_mesh_point_from_stable_id_any(mesh, stable_id, &point) != SS_OK) {
    return SS_ERROR_NOT_FOUND;
  }
  if (kind == SS_MESH_LABEL_MATERIAL) {
    PetscInt raw_value = -1;
    if (DMGetLabelValue(mesh->dm, ss_mesh_semantic_label_name(kind), point,
                        &raw_value) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    *contains = raw_value >= 0 && (uint32_t)raw_value == value ? 1 : 0;
    return SS_OK;
  }
  return ss_mesh_point_membership(mesh->dm, kind, value, point, contains);
}

static SsStatus ss_mesh_label_local_cardinality(const SsMesh *mesh,
                                                SsMeshLabelKind kind, uint32_t value,
                                                int owned_only, uint64_t *count) {
  const char *name = ss_mesh_semantic_label_name(kind);
  char membership_name[64];
  DMLabel label = NULL;
  PetscInt start = 0;
  PetscInt end = 0;
  PetscInt point;
  uint64_t result = 0U;
  if (mesh == NULL || count == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (kind == SS_MESH_LABEL_SUPPORT || kind == SS_MESH_LABEL_INTERFACE) {
    if (ss_mesh_semantic_membership_name(kind, value, membership_name) != SS_OK) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
    name = membership_name;
  } else if (kind != SS_MESH_LABEL_MATERIAL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (DMPlexGetChart(mesh->dm, &start, &end) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (DMGetLabel(mesh->dm, name, &label) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (label == NULL) {
    *count = 0U;
    return SS_OK;
  }
  for (point = start; point < end; ++point) {
    PetscInt point_value = -1;
    if (owned_only != 0 && ss_mesh_point_is_ghost(mesh, mesh->dm, point) != 0) {
      continue;
    }
    if (DMLabelGetValue(label, point, &point_value) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    if (point_value >= 0 &&
        (kind != SS_MESH_LABEL_MATERIAL || (uint32_t)point_value == value)) {
      ++result;
    }
  }
  *count = result;
  return SS_OK;
}

size_t ss_mesh_label_cardinality(const SsMesh *mesh, SsMeshLabelKind kind,
                                 uint32_t value) {
  uint64_t count = 0U;
  size_t converted = 0U;
  if (ss_mesh_label_local_cardinality(mesh, kind, value, 0, &count) != SS_OK ||
      ss_checked_u64_to_size(count, &converted) != SS_OK) {
    return 0U;
  }
  return converted;
}

SsStatus ss_mesh_label_cardinality_global(const SsMesh *mesh, SsMeshLabelKind kind,
                                          uint32_t value, uint64_t *count) {
  uint64_t local = 0U;
  SsStatus status;
  if (mesh == NULL || count == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *count = 0U;
  status = ss_mesh_label_local_cardinality(mesh, kind, value, 1, &local);
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  if (MPI_Allreduce(&local, count, 1, MPI_UINT64_T, MPI_SUM, mesh->communicator) !=
      MPI_SUCCESS) {
    *count = 0U;
    return SS_ERROR_INTERNAL;
  }
  return SS_OK;
}
