/* Collective input validation, ownership, and migration audits.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "mesh_internal.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/core/sort.h"

#include "slope_internal/fe/fe.h"

#include <petscdm.h>
#include <petscdmplex.h>
#include <petsclog.h>
#include <petscsection.h>
#include <petscvec.h>

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static SsStatus ss_mesh_duplicate_ids(MPI_Comm communicator,
                                      const SsEntityId *local_ids, size_t local_count) {
  int size = 0;
  int *send_counts = NULL;
  int *receive_counts = NULL;
  int *send_offsets = NULL;
  int *receive_offsets = NULL;
  int *cursor = NULL;
  SsEntityId *send_ids = NULL;
  SsEntityId *receive_ids = NULL;
  size_t index;
  int total_receive = 0;
  SsStatus status = SS_OK;
  if (MPI_Comm_size(communicator, &size) != MPI_SUCCESS || size <= 0) {
    return SS_ERROR_INTERNAL;
  }
  if ((local_count != 0U && local_ids == NULL) || local_count > (size_t)INT_MAX) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  if (status == SS_OK) {
    send_counts = (int *)calloc((size_t)size, sizeof(*send_counts));
    receive_counts = (int *)calloc((size_t)size, sizeof(*receive_counts));
    send_offsets = (int *)calloc((size_t)size, sizeof(*send_offsets));
    receive_offsets = (int *)calloc((size_t)size, sizeof(*receive_offsets));
    cursor = (int *)calloc((size_t)size, sizeof(*cursor));
    send_ids = (SsEntityId *)malloc(local_count * sizeof(*send_ids));
    if (send_counts == NULL || receive_counts == NULL || send_offsets == NULL ||
        receive_offsets == NULL || cursor == NULL ||
        (local_count != 0U && send_ids == NULL)) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  if (status == SS_OK) {
    for (index = 0U; index < local_count; ++index) {
      const int owner = (int)(local_ids[index] % (SsEntityId)(uint32_t)size);
      if (send_counts[owner] == INT_MAX) {
        status = SS_ERROR_OVERFLOW;
        break;
      }
      ++send_counts[owner];
    }
  }
  if (status == SS_OK) {
    for (index = 1U; index < (size_t)size; ++index) {
      if (send_offsets[index - 1U] > INT_MAX - send_counts[index - 1U]) {
        status = SS_ERROR_OVERFLOW;
        break;
      }
      send_offsets[index] = send_offsets[index - 1U] + send_counts[index - 1U];
    }
  }
  if (status == SS_OK) {
    memcpy(cursor, send_offsets, (size_t)size * sizeof(*cursor));
    for (index = 0U; index < local_count; ++index) {
      const int owner = (int)(local_ids[index] % (SsEntityId)(uint32_t)size);
      send_ids[cursor[owner]++] = local_ids[index];
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (MPI_Alltoall(send_counts, 1, MPI_INT, receive_counts, 1, MPI_INT, communicator) !=
      MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  for (index = 1U; index < (size_t)size; ++index) {
    if (receive_offsets[index - 1U] > INT_MAX - receive_counts[index - 1U]) {
      status = SS_ERROR_OVERFLOW;
      break;
    }
    receive_offsets[index] = receive_offsets[index - 1U] + receive_counts[index - 1U];
  }
  if (status == SS_OK) {
    if (receive_offsets[size - 1] > INT_MAX - receive_counts[size - 1]) {
      status = SS_ERROR_OVERFLOW;
    } else {
      total_receive = receive_offsets[size - 1] + receive_counts[size - 1];
      receive_ids = (SsEntityId *)malloc((size_t)total_receive * sizeof(*receive_ids));
      if (total_receive != 0 && receive_ids == NULL) {
        status = SS_ERROR_OUT_OF_MEMORY;
      }
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (MPI_Alltoallv(send_ids, send_counts, send_offsets, MPI_UINT64_T, receive_ids,
                    receive_counts, receive_offsets, MPI_UINT64_T,
                    communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  ss_sort(receive_ids, (size_t)total_receive, sizeof(*receive_ids),
          ss_mesh_entity_id_compare);
  for (index = 1U; index < (size_t)total_receive; ++index) {
    if (receive_ids[index] == receive_ids[index - 1U]) {
      status = SS_ERROR_DUPLICATE;
      break;
    }
  }

cleanup:
  free(send_counts);
  free(receive_counts);
  free(send_offsets);
  free(receive_offsets);
  free(cursor);
  free(send_ids);
  free(receive_ids);
  return ss_status_allreduce_max(communicator, status);
}

static uint64_t
ss_mesh_intermediate_tuple_hash(const SsMeshIntermediateEntity *record) {
  uint64_t hash = UINT64_C(1469598103934665603);
  size_t field;
  hash ^= (uint64_t)record->dimension;
  hash *= UINT64_C(1099511628211);
  for (field = 0U; field <= (size_t)record->dimension; ++field) {
    uint32_t byte;
    for (byte = 0U; byte < 8U; ++byte) {
      hash ^= (record->vertex_ids[field] >> (byte * 8U)) & UINT64_C(0xff);
      hash *= UINT64_C(1099511628211);
    }
  }
  return hash;
}

static int ss_intermediate_audit_compare(const void *left_raw, const void *right_raw) {
  const SsIntermediateAuditRecord *left = (const SsIntermediateAuditRecord *)left_raw;
  const SsIntermediateAuditRecord *right = (const SsIntermediateAuditRecord *)right_raw;
  size_t vertex;
  if (left->route != right->route) {
    return left->route < right->route ? -1 : 1;
  }
  if (left->route == 0U && left->stable_id != right->stable_id) {
    return left->stable_id < right->stable_id ? -1 : 1;
  }
  if (left->dimension != right->dimension) {
    return left->dimension < right->dimension ? -1 : 1;
  }
  for (vertex = 0U; vertex < 3U; ++vertex) {
    if (left->vertex_ids[vertex] != right->vertex_ids[vertex]) {
      return left->vertex_ids[vertex] < right->vertex_ids[vertex] ? -1 : 1;
    }
  }
  if (left->stable_id != right->stable_id) {
    return left->stable_id < right->stable_id ? -1 : 1;
  }
  return 0;
}

static int ss_intermediate_same_tuple(const SsIntermediateAuditRecord *left,
                                      const SsIntermediateAuditRecord *right) {
  return left->dimension == right->dimension &&
         memcmp(left->vertex_ids, right->vertex_ids, sizeof(left->vertex_ids)) == 0;
}

static SsStatus ss_mesh_audit_intermediate_records(MPI_Comm communicator,
                                                   const SsMeshInput *input) {
  int size = 0;
  int *send_counts = NULL;
  int *receive_counts = NULL;
  int *send_offsets = NULL;
  int *receive_offsets = NULL;
  int *cursor = NULL;
  SsIntermediateAuditRecord *send_records = NULL;
  SsIntermediateAuditRecord *receive_records = NULL;
  MPI_Datatype record_type = MPI_DATATYPE_NULL;
  size_t local_record_count = 0U;
  size_t record;
  int total_receive = 0;
  uint64_t local_unique = 0U;
  uint64_t global_unique = 0U;
  SsStatus status = SS_OK;
  if (input == NULL || MPI_Comm_size(communicator, &size) != MPI_SUCCESS || size <= 0 ||
      input->intermediate_entity_count > (size_t)INT_MAX / 2U) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else {
    local_record_count = input->intermediate_entity_count * 2U;
  }
  if (status == SS_OK) {
    send_counts = (int *)calloc((size_t)size, sizeof(*send_counts));
    receive_counts = (int *)calloc((size_t)size, sizeof(*receive_counts));
    send_offsets = (int *)calloc((size_t)size, sizeof(*send_offsets));
    receive_offsets = (int *)calloc((size_t)size, sizeof(*receive_offsets));
    cursor = (int *)calloc((size_t)size, sizeof(*cursor));
    send_records = (SsIntermediateAuditRecord *)calloc(
        local_record_count == 0U ? 1U : local_record_count, sizeof(*send_records));
    if (send_counts == NULL || receive_counts == NULL || send_offsets == NULL ||
        receive_offsets == NULL || cursor == NULL || send_records == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  if (status == SS_OK) {
    for (record = 0U; record < input->intermediate_entity_count; ++record) {
      const SsMeshIntermediateEntity *source = &input->intermediate_entities[record];
      const int id_owner = (int)(source->stable_entity_id % (SsEntityId)(uint32_t)size);
      const int tuple_owner =
          (int)(ss_mesh_intermediate_tuple_hash(source) % (uint64_t)(uint32_t)size);
      if (send_counts[id_owner] == INT_MAX || send_counts[tuple_owner] == INT_MAX) {
        status = SS_ERROR_OVERFLOW;
        break;
      }
      ++send_counts[id_owner];
      ++send_counts[tuple_owner];
    }
  }
  if (status == SS_OK) {
    size_t rank;
    for (rank = 1U; rank < (size_t)size; ++rank) {
      if (send_offsets[rank - 1U] > INT_MAX - send_counts[rank - 1U]) {
        status = SS_ERROR_OVERFLOW;
        break;
      }
      send_offsets[rank] = send_offsets[rank - 1U] + send_counts[rank - 1U];
    }
  }
  if (status == SS_OK) {
    memcpy(cursor, send_offsets, (size_t)size * sizeof(*cursor));
    for (record = 0U; record < input->intermediate_entity_count; ++record) {
      const SsMeshIntermediateEntity *source = &input->intermediate_entities[record];
      const int owners[2] = {
          (int)(source->stable_entity_id % (SsEntityId)(uint32_t)size),
          (int)(ss_mesh_intermediate_tuple_hash(source) % (uint64_t)(uint32_t)size),
      };
      uint64_t route;
      for (route = 0U; route < 2U; ++route) {
        SsIntermediateAuditRecord *target = &send_records[cursor[owners[route]]++];
        target->route = route;
        target->stable_id = source->stable_entity_id;
        target->dimension = source->dimension;
        memcpy(target->vertex_ids, source->vertex_ids, sizeof(target->vertex_ids));
      }
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (MPI_Type_contiguous(6, MPI_UINT64_T, &record_type) != MPI_SUCCESS ||
      MPI_Type_commit(&record_type) != MPI_SUCCESS ||
      MPI_Alltoall(send_counts, 1, MPI_INT, receive_counts, 1, MPI_INT, communicator) !=
          MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  {
    int rank;
    for (rank = 1; rank < size; ++rank) {
      if (receive_offsets[rank - 1] > INT_MAX - receive_counts[rank - 1]) {
        status = SS_ERROR_OVERFLOW;
        break;
      }
      receive_offsets[rank] = receive_offsets[rank - 1] + receive_counts[rank - 1];
    }
    if (status == SS_OK) {
      if (receive_offsets[size - 1] > INT_MAX - receive_counts[size - 1]) {
        status = SS_ERROR_OVERFLOW;
      } else {
        total_receive = receive_offsets[size - 1] + receive_counts[size - 1];
      }
    }
  }
  if (status == SS_OK) {
    receive_records = (SsIntermediateAuditRecord *)calloc(
        total_receive == 0 ? 1U : (size_t)total_receive, sizeof(*receive_records));
    if (receive_records == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (MPI_Alltoallv(send_records, send_counts, send_offsets, record_type,
                    receive_records, receive_counts, receive_offsets, record_type,
                    communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  ss_sort(receive_records, (size_t)total_receive, sizeof(*receive_records),
          ss_intermediate_audit_compare);
  for (record = 0U; record < (size_t)total_receive;) {
    size_t next = record + 1U;
    if (receive_records[record].route == 0U) {
      while (next < (size_t)total_receive && receive_records[next].route == 0U &&
             receive_records[next].stable_id == receive_records[record].stable_id) {
        if (ss_intermediate_same_tuple(&receive_records[record],
                                       &receive_records[next]) == 0) {
          status = SS_ERROR_DUPLICATE;
          break;
        }
        ++next;
      }
      ++local_unique;
    } else {
      while (next < (size_t)total_receive && receive_records[next].route == 1U &&
             ss_intermediate_same_tuple(&receive_records[record],
                                        &receive_records[next]) != 0) {
        if (receive_records[next].stable_id != receive_records[record].stable_id) {
          status = SS_ERROR_DUPLICATE;
          break;
        }
        ++next;
      }
    }
    if (status != SS_OK) {
      break;
    }
    record = next;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK && MPI_Allreduce(&local_unique, &global_unique, 1, MPI_UINT64_T,
                                       MPI_SUM, communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK && global_unique != input->global_intermediate_entity_count) {
    status = SS_ERROR_INTEGRITY;
  }

cleanup:
  if (record_type != MPI_DATATYPE_NULL && MPI_Type_free(&record_type) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  free(send_counts);
  free(receive_counts);
  free(send_offsets);
  free(receive_offsets);
  free(cursor);
  free(send_records);
  free(receive_records);
  return ss_status_allreduce_max(communicator, status);
}

SsStatus ss_mesh_audit_semantic_coverage(DM dm, uint32_t dimension) {
  PetscInt chart_start = 0;
  PetscInt chart_end = 0;
  PetscInt point;
  if (dm == NULL || (dimension != 2U && dimension != 3U) ||
      DMPlexGetChart(dm, &chart_start, &chart_end) != PETSC_SUCCESS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (point = chart_start; point < chart_end; ++point) {
    PetscInt depth = -1;
    PetscInt material = -1;
    PetscInt explicit_boundary = -1;
    int has_support = 0;
    if (DMPlexGetPointDepth(dm, point, &depth) != PETSC_SUCCESS ||
        DMGetLabelValue(dm, "ss-material", point, &material) != PETSC_SUCCESS ||
        DMGetLabelValue(dm, "ss-explicit-boundary", point, &explicit_boundary) !=
            PETSC_SUCCESS ||
        ss_mesh_point_has_any_membership(dm, SS_MESH_LABEL_SUPPORT, point,
                                         &has_support) != SS_OK) {
      return SS_ERROR_INTERNAL;
    }
    if ((depth == (PetscInt)dimension && material < 0) ||
        (material >= 0 && depth != (PetscInt)dimension) ||
        (explicit_boundary >= 0 && has_support == 0)) {
      return SS_ERROR_INTEGRITY;
    }
  }
  return SS_OK;
}

static SsStatus ss_mesh_validate_input_local(const SsMeshInput *input) {
  const SsFeDescriptor *descriptor = NULL;
  size_t cell;
  size_t facet;
  size_t entity;
  if (input == NULL || (input->dimension != 2U && input->dimension != 3U) ||
      input->global_vertex_count < (uint64_t)input->dimension + 1U ||
      input->global_cell_count == 0U || input->global_boundary_facet_count == 0U ||
      input->global_intermediate_entity_count == 0U ||
      input->global_coordinate_node_count == 0U ||
      (input->vertex_count != 0U &&
       (input->vertex_ids == NULL || input->vertex_coordinates == NULL)) ||
      (input->cell_count != 0U && (input->cell_ids == NULL || input->cells == NULL ||
                                   input->cell_coordinate_nodes == NULL)) ||
      (input->boundary_facet_count != 0U &&
       (input->boundary_facet_ids == NULL || input->boundary_facets == NULL)) ||
      (input->intermediate_entity_count != 0U &&
       input->intermediate_entities == NULL) ||
      (input->label_count != 0U && input->labels == NULL) ||
      (input->coordinate_node_count != 0U &&
       (input->coordinate_node_ids == NULL || input->coordinate_coordinates == NULL)) ||
      !isfinite(input->absolute_determinant_tolerance) ||
      !isfinite(input->relative_determinant_tolerance) ||
      input->absolute_determinant_tolerance < 0.0 ||
      input->relative_determinant_tolerance <= 0.0 ||
      ss_fe_descriptor_find(input->dimension, input->coordinate_order, &descriptor) !=
          SS_OK) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (input->cell_count > SIZE_MAX / ((size_t)input->dimension + 1U) ||
      input->boundary_facet_count > SIZE_MAX / (size_t)input->dimension ||
      input->cell_count > SIZE_MAX / descriptor->node_count ||
      input->intermediate_entity_count > SIZE_MAX / sizeof(SsMeshIntermediateEntity) ||
      input->vertex_count > SIZE_MAX / (size_t)input->dimension ||
      input->coordinate_node_count > SIZE_MAX / (size_t)input->dimension) {
    return SS_ERROR_OVERFLOW;
  }
  for (cell = 0U; cell < input->cell_count * ((size_t)input->dimension + 1U); ++cell) {
    if (input->cells[cell] >= input->global_vertex_count) {
      return SS_ERROR_INTEGRITY;
    }
  }
  for (facet = 0U; facet < input->boundary_facet_count * (size_t)input->dimension;
       ++facet) {
    if (input->boundary_facets[facet] >= input->global_vertex_count) {
      return SS_ERROR_INTEGRITY;
    }
  }
  for (cell = 0U; cell < input->cell_count * descriptor->node_count; ++cell) {
    if (input->cell_coordinate_nodes[cell] >= input->global_coordinate_node_count) {
      return SS_ERROR_INTEGRITY;
    }
  }
  for (entity = 0U; entity < input->intermediate_entity_count; ++entity) {
    const SsMeshIntermediateEntity *record = &input->intermediate_entities[entity];
    size_t vertex;
    if (record->dimension == 0U || record->dimension >= input->dimension) {
      return SS_ERROR_INTEGRITY;
    }
    for (vertex = 0U; vertex <= (size_t)record->dimension; ++vertex) {
      if (vertex != 0U &&
          record->vertex_ids[vertex] <= record->vertex_ids[vertex - 1U]) {
        return SS_ERROR_INTEGRITY;
      }
    }
    for (; vertex < 3U; ++vertex) {
      if (record->vertex_ids[vertex] != 0U) {
        return SS_ERROR_INTEGRITY;
      }
    }
  }
  return SS_OK;
}

SsStatus ss_mesh_validate_collective(MPI_Comm communicator, const SsMeshInput *input,
                                     const SsFeDescriptor **descriptor) {
  uint32_t local_metadata[2] = {0U, 0U};
  uint32_t minimum_metadata[2] = {0U, 0U};
  uint32_t maximum_metadata[2] = {0U, 0U};
  uint64_t local_globals[5] = {0U, 0U, 0U, 0U, 0U};
  uint64_t minimum_globals[5] = {0U, 0U, 0U, 0U, 0U};
  uint64_t maximum_globals[5] = {0U, 0U, 0U, 0U, 0U};
  uint64_t local_counts[4] = {0U, 0U, 0U, 0U};
  uint64_t global_counts[4] = {0U, 0U, 0U, 0U};
  double local_tolerances[2] = {0.0, 0.0};
  double minimum_tolerances[2] = {0.0, 0.0};
  double maximum_tolerances[2] = {0.0, 0.0};
  PetscInt checked = 0;
  SsStatus status = ss_mesh_validate_input_local(input);
  if (descriptor == NULL) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else {
    *descriptor = NULL;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    return status;
  }
  local_metadata[0] = input->dimension;
  local_metadata[1] = input->coordinate_order;
  local_globals[0] = input->global_vertex_count;
  local_globals[1] = input->global_cell_count;
  local_globals[2] = input->global_boundary_facet_count;
  local_globals[3] = input->global_intermediate_entity_count;
  local_globals[4] = input->global_coordinate_node_count;
  local_counts[0] = (uint64_t)input->vertex_count;
  local_counts[1] = (uint64_t)input->cell_count;
  local_counts[2] = (uint64_t)input->boundary_facet_count;
  local_counts[3] = (uint64_t)input->coordinate_node_count;
  local_tolerances[0] = input->absolute_determinant_tolerance;
  local_tolerances[1] = input->relative_determinant_tolerance;
  if (MPI_Allreduce(local_metadata, minimum_metadata, 2, MPI_UINT32_T, MPI_MIN,
                    communicator) != MPI_SUCCESS ||
      MPI_Allreduce(local_metadata, maximum_metadata, 2, MPI_UINT32_T, MPI_MAX,
                    communicator) != MPI_SUCCESS ||
      MPI_Allreduce(local_globals, minimum_globals, 5, MPI_UINT64_T, MPI_MIN,
                    communicator) != MPI_SUCCESS ||
      MPI_Allreduce(local_globals, maximum_globals, 5, MPI_UINT64_T, MPI_MAX,
                    communicator) != MPI_SUCCESS ||
      MPI_Allreduce(local_counts, global_counts, 4, MPI_UINT64_T, MPI_SUM,
                    communicator) != MPI_SUCCESS ||
      MPI_Allreduce(local_tolerances, minimum_tolerances, 2, MPI_DOUBLE, MPI_MIN,
                    communicator) != MPI_SUCCESS ||
      MPI_Allreduce(local_tolerances, maximum_tolerances, 2, MPI_DOUBLE, MPI_MAX,
                    communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (memcmp(minimum_metadata, maximum_metadata, sizeof(minimum_metadata)) != 0 ||
      memcmp(minimum_globals, maximum_globals, sizeof(minimum_globals)) != 0 ||
      memcmp(minimum_tolerances, maximum_tolerances, sizeof(minimum_tolerances)) != 0 ||
      global_counts[0] != local_globals[0] || global_counts[1] != local_globals[1] ||
      global_counts[2] != local_globals[2] || global_counts[3] != local_globals[4]) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  if (status == SS_OK &&
      (ss_checked_u64_to_petsc(input->global_vertex_count, &checked) != SS_OK ||
       ss_checked_u64_to_petsc(input->global_cell_count, &checked) != SS_OK ||
       ss_checked_u64_to_petsc(input->global_boundary_facet_count, &checked) != SS_OK ||
       ss_checked_u64_to_petsc(input->global_intermediate_entity_count, &checked) !=
           SS_OK ||
       ss_checked_u64_to_petsc(input->global_coordinate_node_count, &checked) !=
           SS_OK ||
       ss_fe_descriptor_find(input->dimension, input->coordinate_order, descriptor) !=
           SS_OK)) {
    status = SS_ERROR_OVERFLOW;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    return status;
  }
  {
    size_t topological_count = 0U;
    size_t vertex_and_cell_count = 0U;
    SsEntityId *topological_ids = NULL;
    size_t offset = 0U;
    if (ss_checked_add_size(input->vertex_count, input->cell_count,
                            &vertex_and_cell_count) != SS_OK ||
        ss_checked_add_size(vertex_and_cell_count, input->boundary_facet_count,
                            &topological_count) != SS_OK ||
        topological_count > SIZE_MAX / sizeof(*topological_ids)) {
      status = SS_ERROR_OVERFLOW;
    } else {
      topological_ids = (SsEntityId *)calloc(
          topological_count == 0U ? 1U : topological_count, sizeof(*topological_ids));
    }
    if (status == SS_OK && topological_ids == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
    if (status == SS_OK) {
      if (input->vertex_count != 0U) {
        memcpy(topological_ids + offset, input->vertex_ids,
               input->vertex_count * sizeof(*topological_ids));
        offset += input->vertex_count;
      }
      if (input->cell_count != 0U) {
        memcpy(topological_ids + offset, input->cell_ids,
               input->cell_count * sizeof(*topological_ids));
        offset += input->cell_count;
      }
      if (input->boundary_facet_count != 0U) {
        memcpy(topological_ids + offset, input->boundary_facet_ids,
               input->boundary_facet_count * sizeof(*topological_ids));
      }
    }
    status = ss_status_allreduce_max(communicator, status);
    if (status == SS_OK) {
      status = ss_mesh_duplicate_ids(communicator, topological_ids, topological_count);
    }
    free(topological_ids);
  }
  if (status == SS_OK) {
    status = ss_mesh_duplicate_ids(communicator, input->coordinate_node_ids,
                                   input->coordinate_node_count);
  }
  if (status == SS_OK) {
    status = ss_mesh_audit_intermediate_records(communicator, input);
  }
  return status;
}

SsStatus ss_mesh_audit_point_ownership(SsMesh *mesh, uint64_t expected_global_points) {
  SsEntityId *owned_ids = NULL;
  size_t owned_count = 0U;
  size_t entry;
  uint64_t local_count;
  uint64_t global_count = 0U;
  SsStatus status = SS_OK;
  if (mesh == NULL || mesh->point_count > SIZE_MAX / sizeof(*owned_ids)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  owned_ids = (SsEntityId *)calloc(mesh->point_count == 0U ? 1U : mesh->point_count,
                                   sizeof(*owned_ids));
  if (owned_ids == NULL) {
    status = SS_ERROR_OUT_OF_MEMORY;
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status != SS_OK) {
    free(owned_ids);
    return status;
  }
  for (entry = 0U; entry < mesh->point_count; ++entry) {
    const SsPointEntry *point = &mesh->points[entry];
    if (point->kind < SS_MESH_ENTITY_VERTEX || point->kind > SS_MESH_ENTITY_CELL) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    if (ss_mesh_point_is_ghost(mesh, mesh->dm, point->point) == 0) {
      owned_ids[owned_count++] = point->stable_id;
    }
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status == SS_OK) {
    status = ss_mesh_duplicate_ids(mesh->communicator, owned_ids, owned_count);
  }
  local_count = (uint64_t)owned_count;
  if (status == SS_OK && MPI_Allreduce(&local_count, &global_count, 1, MPI_UINT64_T,
                                       MPI_SUM, mesh->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK && global_count != expected_global_points) {
    status = SS_ERROR_INTEGRITY;
  }
  free(owned_ids);
  return ss_status_allreduce_max(mesh->communicator, status);
}

enum {
  SS_MESH_COUNT_OWNED_VERTEX = 0,
  SS_MESH_COUNT_OWNED_EDGE = 1,
  SS_MESH_COUNT_OWNED_FACET = 2,
  SS_MESH_COUNT_OWNED_CELL = 3,
  SS_MESH_COUNT_GHOST_CELL = 4,
  SS_MESH_COUNT_OWNED_COORDINATE = 5,
  SS_MESH_COUNT_OWNED_INTERMEDIATE = 6,
  SS_MESH_COUNT_GHOST_VERTEX = 7,
  SS_MESH_COUNT_GHOST_EDGE = 8,
  SS_MESH_COUNT_GHOST_FACET = 9,
  SS_MESH_COUNT_GHOST_COORDINATE = 10,
  SS_MESH_COUNT_GHOST_INTERMEDIATE = 11,
  SS_MESH_COUNT_CAPACITY = 12
};

static SsStatus ss_mesh_local_entity_counts(SsMesh *mesh,
                                            uint64_t local[SS_MESH_COUNT_CAPACITY]) {
  PetscInt start = 0;
  PetscInt end = 0;
  PetscInt point;
  PetscSection coordinate_section = NULL;
  memset(local, 0, SS_MESH_COUNT_CAPACITY * sizeof(local[0]));
  if (DMPlexGetChart(mesh->dm, &start, &end) != PETSC_SUCCESS ||
      DMGetCoordinateSection(mesh->dm, &coordinate_section) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  for (point = start; point < end; ++point) {
    PetscInt depth = -1;
    int has_support = 0;
    const int ghost = ss_mesh_point_is_ghost(mesh, mesh->dm, point);
    if (DMPlexGetPointDepth(mesh->dm, point, &depth) != PETSC_SUCCESS ||
        ss_mesh_point_has_any_membership(mesh->dm, SS_MESH_LABEL_SUPPORT, point,
                                         &has_support) != SS_OK) {
      return SS_ERROR_INTERNAL;
    }
    if (depth == 0 && ghost == 0) {
      ++local[SS_MESH_COUNT_OWNED_VERTEX];
    } else if (depth == 0) {
      ++local[SS_MESH_COUNT_GHOST_VERTEX];
    }
    if (depth == 1 && ghost == 0) {
      ++local[SS_MESH_COUNT_OWNED_EDGE];
    } else if (depth == 1) {
      ++local[SS_MESH_COUNT_GHOST_EDGE];
    }
    if (depth > 0 && depth < (PetscInt)mesh->dimension && ghost == 0) {
      ++local[SS_MESH_COUNT_OWNED_INTERMEDIATE];
    } else if (depth > 0 && depth < (PetscInt)mesh->dimension) {
      ++local[SS_MESH_COUNT_GHOST_INTERMEDIATE];
    }
    if (depth == (PetscInt)mesh->dimension - 1 && ghost == 0 && has_support != 0) {
      ++local[SS_MESH_COUNT_OWNED_FACET];
    } else if (depth == (PetscInt)mesh->dimension - 1 && has_support != 0) {
      ++local[SS_MESH_COUNT_GHOST_FACET];
    }
    if (depth == (PetscInt)mesh->dimension) {
      if (ghost == 0) {
        ++local[SS_MESH_COUNT_OWNED_CELL];
      } else {
        ++local[SS_MESH_COUNT_GHOST_CELL];
      }
    }
    {
      PetscInt dof = 0;
      if (PetscSectionGetDof(coordinate_section, point, &dof) != PETSC_SUCCESS) {
        return SS_ERROR_INTERNAL;
      }
      local[ghost == 0 ? SS_MESH_COUNT_OWNED_COORDINATE
                       : SS_MESH_COUNT_GHOST_COORDINATE] +=
          (uint64_t)dof / (uint64_t)mesh->dimension;
    }
  }
  return SS_OK;
}

SsStatus ss_mesh_finish_summary(SsMesh *mesh) {
  uint64_t local[SS_MESH_COUNT_CAPACITY];
  uint64_t global[SS_MESH_COUNT_CAPACITY];
  uint64_t minimum_owned_cells = 0U;
  uint64_t maximum_owned_cells = 0U;
  uint64_t point_bytes;
  uint64_t entity_bytes;
  uint64_t metadata_bytes;
  const uint64_t metadata_bytes_per_point =
      (uint64_t)sizeof(*mesh->cached_stable_ids) +
      (uint64_t)sizeof(*mesh->cached_kinds) +
      (uint64_t)sizeof(*mesh->cached_point_valid) +
      4U * (uint64_t)sizeof(*mesh->cached_cell_vertex_ids) +
      (uint64_t)sizeof(*mesh->cached_cell_vertex_counts) +
      SS_COORDINATE_SLOT_COUNT * (uint64_t)sizeof(*mesh->cached_coordinate_node_ids) +
      (uint64_t)sizeof(*mesh->cached_coordinate_node_counts);
  const uint64_t cached_point_count =
      (uint64_t)(mesh->cache_chart_end - mesh->cache_chart_start);
  SsStatus status = ss_mesh_local_entity_counts(mesh, local);
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  if (MPI_Allreduce(local, global, SS_MESH_COUNT_CAPACITY, MPI_UINT64_T, MPI_SUM,
                    mesh->communicator) != MPI_SUCCESS ||
      MPI_Allreduce(&local[SS_MESH_COUNT_OWNED_CELL], &minimum_owned_cells, 1,
                    MPI_UINT64_T, MPI_MIN, mesh->communicator) != MPI_SUCCESS ||
      MPI_Allreduce(&local[SS_MESH_COUNT_OWNED_CELL], &maximum_owned_cells, 1,
                    MPI_UINT64_T, MPI_MAX, mesh->communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  mesh->summary.dimension = mesh->dimension;
  mesh->summary.coordinate_order = mesh->coordinate_order;
  mesh->summary.global_vertices = global[SS_MESH_COUNT_OWNED_VERTEX];
  mesh->summary.global_edges = global[SS_MESH_COUNT_OWNED_EDGE];
  mesh->summary.global_facets = global[SS_MESH_COUNT_OWNED_FACET];
  mesh->summary.global_intermediate_entities = global[SS_MESH_COUNT_OWNED_INTERMEDIATE];
  mesh->summary.global_cells = global[SS_MESH_COUNT_OWNED_CELL];
  mesh->summary.global_coordinate_nodes = global[SS_MESH_COUNT_OWNED_COORDINATE];
  mesh->summary.owned_vertices = local[SS_MESH_COUNT_OWNED_VERTEX];
  mesh->summary.ghost_vertices = local[SS_MESH_COUNT_GHOST_VERTEX];
  mesh->summary.owned_edges = local[SS_MESH_COUNT_OWNED_EDGE];
  mesh->summary.ghost_edges = local[SS_MESH_COUNT_GHOST_EDGE];
  mesh->summary.owned_facets = local[SS_MESH_COUNT_OWNED_FACET];
  mesh->summary.ghost_facets = local[SS_MESH_COUNT_GHOST_FACET];
  mesh->summary.owned_intermediate_entities = local[SS_MESH_COUNT_OWNED_INTERMEDIATE];
  mesh->summary.ghost_intermediate_entities = local[SS_MESH_COUNT_GHOST_INTERMEDIATE];
  mesh->summary.owned_cells = local[SS_MESH_COUNT_OWNED_CELL];
  mesh->summary.ghost_cells = local[SS_MESH_COUNT_GHOST_CELL];
  mesh->summary.owned_coordinate_nodes = local[SS_MESH_COUNT_OWNED_COORDINATE];
  mesh->summary.ghost_coordinate_nodes = local[SS_MESH_COUNT_GHOST_COORDINATE];
  mesh->summary.minimum_owned_cells = minimum_owned_cells;
  mesh->summary.maximum_owned_cells = maximum_owned_cells;
  mesh->summary.partition_imbalance = mesh->summary.global_cells == 0U
                                          ? 0.0
                                          : (double)maximum_owned_cells *
                                                (double)mesh->size /
                                                (double)mesh->summary.global_cells;
  if (mesh->point_count > UINT64_MAX / sizeof(*mesh->points) ||
      mesh->entity_count > UINT64_MAX / sizeof(*mesh->entities) ||
      cached_point_count > UINT64_MAX / metadata_bytes_per_point) {
    return SS_ERROR_OVERFLOW;
  }
  point_bytes = (uint64_t)mesh->point_count * (uint64_t)sizeof(*mesh->points);
  entity_bytes = (uint64_t)mesh->entity_count * (uint64_t)sizeof(*mesh->entities);
  metadata_bytes = cached_point_count * metadata_bytes_per_point;
  if (point_bytes > UINT64_MAX - entity_bytes ||
      point_bytes + entity_bytes > UINT64_MAX - metadata_bytes) {
    return SS_ERROR_OVERFLOW;
  }
  mesh->summary.cache_memory_bytes = point_bytes + entity_bytes + metadata_bytes;
  return SS_OK;
}

SsStatus ss_mesh_audit_migration(SsMesh *mesh) {
  SsMeshDigestAccumulator stable = {0U, 0U, 0U, 0U, 0U};
  SsMeshDigestAccumulator labels = {0U, 0U, 0U, 0U, 0U};
  SsMeshDigestAccumulator coordinates = {0U, 0U, 0U, 0U, 0U};
  char stable_digest[SS_SHA256_HEX_CAPACITY] = {0};
  char topology_digest[SS_SHA256_HEX_CAPACITY] = {0};
  char label_digest[SS_SHA256_HEX_CAPACITY] = {0};
  char coordinate_digest[SS_SHA256_HEX_CAPACITY] = {0};
  PetscSection coordinate_section = NULL;
  Vec coordinate_vector = NULL;
  const PetscScalar *coordinate_array = NULL;
  PetscInt chart_start = 0;
  PetscInt chart_end = 0;
  PetscInt registered_label_count = 0;
  PetscInt point;
  SsStatus status = SS_OK;
  if (DMPlexGetChart(mesh->dm, &chart_start, &chart_end) != PETSC_SUCCESS ||
      DMGetCoordinateSection(mesh->dm, &coordinate_section) != PETSC_SUCCESS ||
      DMGetCoordinatesLocal(mesh->dm, &coordinate_vector) != PETSC_SUCCESS ||
      coordinate_vector == NULL ||
      VecGetArrayRead(coordinate_vector, &coordinate_array) != PETSC_SUCCESS ||
      DMGetNumLabels(mesh->dm, &registered_label_count) != PETSC_SUCCESS ||
      registered_label_count < 0) {
    status = SS_ERROR_INTERNAL;
  }
  for (point = chart_start; status == SS_OK && point < chart_end; ++point) {
    SsEntityId stable_id = 0U;
    PetscInt raw_kind = -1;
    PetscInt explicit_boundary = -1;
    PetscInt coordinate_dof = 0;
    PetscInt coordinate_offset = 0;
    if (ss_mesh_point_is_ghost(mesh, mesh->dm, point) != 0) {
      continue;
    }
    if (ss_mesh_cached_stable_id(mesh, point, &stable_id) != SS_OK ||
        DMGetLabelValue(mesh->dm, "ss-entity-kind", point, &raw_kind) !=
            PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
      break;
    }
    if (raw_kind == (PetscInt)SS_MESH_ENTITY_VERTEX) {
      ss_mesh_digest_id(&stable, UINT64_C(1), stable_id);
    } else if (raw_kind == (PetscInt)SS_MESH_ENTITY_CELL) {
      ss_mesh_digest_id(&stable, UINT64_C(2), stable_id);
    } else if (raw_kind == (PetscInt)SS_MESH_ENTITY_FACET &&
               DMGetLabelValue(mesh->dm, "ss-explicit-boundary", point,
                               &explicit_boundary) == PETSC_SUCCESS &&
               explicit_boundary >= 0) {
      ss_mesh_digest_id(&stable, UINT64_C(3), stable_id);
    }
    {
      const SsMeshLabelKind label_kind = SS_MESH_LABEL_MATERIAL;
      const char *name = ss_mesh_semantic_label_name(label_kind);
      PetscInt raw_value = -1;
      if (DMGetLabelValue(mesh->dm, name, point, &raw_value) != PETSC_SUCCESS) {
        status = SS_ERROR_INTERNAL;
        break;
      }
      if (raw_value >= 0) {
        const uint32_t value = (uint32_t)raw_value;
        uint64_t first;
        uint64_t second;
        ss_mesh_digest_record_begin(UINT64_C(11), &first, &second);
        ss_mesh_digest_record_u32((uint32_t)label_kind, &first, &second);
        ss_mesh_digest_record_u64(stable_id, &first, &second);
        ss_mesh_digest_record_u32(value, &first, &second);
        ss_mesh_digest_add(&labels, first, second);
      }
    }
    {
      PetscInt label_index;
      for (label_index = 0; label_index < registered_label_count; ++label_index) {
        const char *name = NULL;
        SsMeshLabelKind label_kind = SS_MESH_LABEL_MATERIAL;
        uint32_t value = 0U;
        PetscInt raw_value = -1;
        uint64_t first;
        uint64_t second;
        if (DMGetLabelName(mesh->dm, label_index, &name) != PETSC_SUCCESS) {
          status = SS_ERROR_INTERNAL;
          break;
        }
        if (ss_mesh_semantic_membership_parse(name, &label_kind, &value) == 0) {
          continue;
        }
        if (DMGetLabelValue(mesh->dm, name, point, &raw_value) != PETSC_SUCCESS) {
          status = SS_ERROR_INTERNAL;
          break;
        }
        if (raw_value < 0) {
          continue;
        }
        ss_mesh_digest_record_begin(UINT64_C(11), &first, &second);
        ss_mesh_digest_record_u32((uint32_t)label_kind, &first, &second);
        ss_mesh_digest_record_u64(stable_id, &first, &second);
        ss_mesh_digest_record_u32(value, &first, &second);
        ss_mesh_digest_add(&labels, first, second);
      }
      if (status != SS_OK) {
        break;
      }
    }
    if (PetscSectionGetDof(coordinate_section, point, &coordinate_dof) !=
            PETSC_SUCCESS ||
        PetscSectionGetOffset(coordinate_section, point, &coordinate_offset) !=
            PETSC_SUCCESS ||
        coordinate_dof < 0 || coordinate_dof % (PetscInt)mesh->dimension != 0) {
      status = SS_ERROR_INTERNAL;
      break;
    }
    {
      const PetscInt slot_count = coordinate_dof / (PetscInt)mesh->dimension;
      PetscInt slot;
      for (slot = 0; slot < slot_count; ++slot) {
        SsEntityId coordinate_id = 0U;
        double values[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
        uint64_t first;
        uint64_t second;
        uint32_t component;
        if (ss_mesh_cached_coordinate_node(mesh, point, (uint32_t)slot,
                                           &coordinate_id) != SS_OK) {
          status = SS_ERROR_INTEGRITY;
          break;
        }
        for (component = 0U; component < mesh->dimension; ++component) {
          values[component] = (double)PetscRealPart(
              coordinate_array[coordinate_offset + slot * (PetscInt)mesh->dimension +
                               (PetscInt)component]);
        }
        ss_mesh_digest_id(&stable, UINT64_C(4), coordinate_id);
        ss_mesh_digest_record_begin(UINT64_C(21), &first, &second);
        ss_mesh_digest_record_u64(coordinate_id, &first, &second);
        for (component = 0U; component < mesh->dimension; ++component) {
          ss_mesh_digest_record_f64(values[component], &first, &second);
        }
        ss_mesh_digest_add(&coordinates, first, second);
      }
    }
  }
  if (coordinate_array != NULL &&
      VecRestoreArrayRead(coordinate_vector, &coordinate_array) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  if (ss_mesh_compute_topology_digest(mesh->dm, mesh, mesh->communicator,
                                      topology_digest) != SS_OK ||
      ss_mesh_digest_finalize(mesh->communicator, &stable, stable_digest) != SS_OK ||
      ss_mesh_digest_finalize(mesh->communicator, &labels, label_digest) != SS_OK ||
      ss_mesh_digest_finalize(mesh->communicator, &coordinates, coordinate_digest) !=
          SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  if (strcmp(stable_digest, mesh->summary.stable_id_digest) != 0 ||
      strcmp(topology_digest, mesh->summary.topology_digest) != 0 ||
      strcmp(label_digest, mesh->summary.label_digest) != 0 ||
      strcmp(coordinate_digest, mesh->summary.coordinate_digest) != 0) {
    return SS_ERROR_INTEGRITY;
  }
  return SS_OK;
}
