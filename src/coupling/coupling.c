/* Transfer plan validation, identity, and lifetime.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "coupling_internal.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/core/sha256.h"
#include "slope_internal/core/sort.h"
#include "slope_internal/core/util.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define SS_COUPLING_DIGEST_BYTES 32U

typedef struct {
  SsEntityId point_id;
  SsEntityId target_cell_id;
  SsEntityId source_cell_id;
  uint32_t quadrature_index;
  uint32_t dimension;
  double physical[SS_TRANSFER_MAX_DIMENSION];
  double reference[SS_TRANSFER_MAX_DIMENSION];
  double physical_weight;
  double cell_epsilon;
  double saturated_unit_weight;
  double unsaturated_unit_weight;
} SsCouplingPlanDigestRecord;

typedef struct {
  uint32_t dimension;
  uint32_t upward_axis;
  uint32_t transfer_type;
  uint32_t digest_scheme;
  uint64_t source_generation;
  uint64_t source_mesh_generation;
  uint64_t source_coordinate_generation;
  char source_mesh_identity[SS_COUPLING_ID_CAPACITY];
  char source_space_identity[SS_COUPLING_ID_CAPACITY];
  char source_model_identity[SS_COUPLING_ID_CAPACITY];
  char source_pressure_unit[SS_COUPLING_ID_CAPACITY];
  char target_mesh_identity[SS_COUPLING_ID_CAPACITY];
  char target_space_identity[SS_COUPLING_ID_CAPACITY];
  char target_quadrature_identity[SS_COUPLING_ID_CAPACITY];
} SsCouplingPlanDigestHeader;

typedef struct {
  uint64_t global_record_count;
  unsigned char record_digest_xor[SS_COUPLING_DIGEST_BYTES];
} SsCouplingDigestAggregate;

typedef struct {
  SsEntityId cell_id;
  uint32_t quadrature_index;
} SsCouplingTargetSlot;

SsStatus ss_coupling_checked_product(size_t left, size_t right, size_t *product) {
  if (product == NULL || (right != 0U && left > SIZE_MAX / right)) {
    return SS_ERROR_OVERFLOW;
  }
  *product = left * right;
  return SS_OK;
}

SsStatus ss_coupling_checked_add(size_t left, size_t right, size_t *sum) {
  if (sum == NULL || left > SIZE_MAX - right) {
    return SS_ERROR_OVERFLOW;
  }
  *sum = left + right;
  return SS_OK;
}

static int ss_coupling_valid_identity(const char *identity, size_t capacity) {
  const size_t length = ss_bounded_length(identity, capacity);
  return length != 0U && length < capacity;
}

static int ss_coupling_source_state_valid(const SsTransferSourceState *state) {
  return state != NULL && state->accepted_generation != 0U &&
         state->mesh_generation != 0U && state->coordinate_generation != 0U &&
         ss_coupling_valid_identity(state->mesh_identity,
                                    sizeof(state->mesh_identity)) &&
         ss_coupling_valid_identity(state->field_space_identity,
                                    sizeof(state->field_space_identity)) &&
         ss_coupling_valid_identity(state->model_identity,
                                    sizeof(state->model_identity)) &&
         ss_coupling_valid_identity(state->pressure_unit, sizeof(state->pressure_unit));
}

int ss_coupling_source_state_equal(const SsTransferSourceState *left,
                                   const SsTransferSourceState *right) {
  return left->accepted_generation == right->accepted_generation &&
         left->mesh_generation == right->mesh_generation &&
         left->coordinate_generation == right->coordinate_generation &&
         strcmp(left->mesh_identity, right->mesh_identity) == 0 &&
         strcmp(left->field_space_identity, right->field_space_identity) == 0 &&
         strcmp(left->model_identity, right->model_identity) == 0 &&
         strcmp(left->pressure_unit, right->pressure_unit) == 0;
}

SsStatus ss_coupling_read_collective_source_state(MPI_Comm communicator,
                                                  const SsTransferGeometryOps *geometry,
                                                  SsTransferSourceState *state) {
  SsTransferSourceState local_state = {0};
  SsStatus status = geometry->source_state(geometry->context, &local_state);
  if (status == SS_OK && !ss_coupling_source_state_valid(&local_state)) {
    status = SS_ERROR_INTEGRITY;
  }
  *state = local_state;
  if (MPI_Bcast(state, (int)sizeof(*state), MPI_BYTE, 0, communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (status == SS_OK && (!ss_coupling_source_state_valid(state) ||
                          !ss_coupling_source_state_equal(&local_state, state))) {
    status = SS_ERROR_INTEGRITY;
  }
  return ss_status_allreduce_max(communicator, status);
}

static int ss_coupling_target_valid(const SsTransferTargetPoint *target,
                                    uint32_t dimension) {
  uint32_t coordinate;
  if (target == NULL || target->stable_point_id == 0U ||
      target->stable_target_cell_id == 0U || target->dimension != dimension ||
      !ss_is_finite(target->physical_weight) || target->physical_weight <= 0.0 ||
      !ss_is_finite(target->cell_epsilon) || target->cell_epsilon <= 0.0 ||
      !ss_is_finite(target->saturated_unit_weight) ||
      target->saturated_unit_weight <= 0.0 ||
      !ss_is_finite(target->unsaturated_unit_weight) ||
      target->unsaturated_unit_weight <= 0.0) {
    return 0;
  }
  for (coordinate = 0U; coordinate < dimension; ++coordinate) {
    if (!ss_is_finite(target->physical[coordinate])) {
      return 0;
    }
  }
  return 1;
}

static int ss_coupling_compare_plan_points(const void *left, const void *right) {
  const SsCouplingPlanPoint *first = (const SsCouplingPlanPoint *)left;
  const SsCouplingPlanPoint *second = (const SsCouplingPlanPoint *)right;
  if (first->target.stable_target_cell_id < second->target.stable_target_cell_id) {
    return -1;
  }
  if (first->target.stable_target_cell_id > second->target.stable_target_cell_id) {
    return 1;
  }
  if (first->target.quadrature_index < second->target.quadrature_index) {
    return -1;
  }
  if (first->target.quadrature_index > second->target.quadrature_index) {
    return 1;
  }
  return first->target.stable_point_id < second->target.stable_point_id
             ? -1
             : (first->target.stable_point_id > second->target.stable_point_id ? 1 : 0);
}

static int ss_coupling_compare_entity_ids(const void *left, const void *right) {
  const SsEntityId first = *(const SsEntityId *)left;
  const SsEntityId second = *(const SsEntityId *)right;
  return first < second ? -1 : (first > second ? 1 : 0);
}

static int ss_coupling_compare_target_slots(const void *left, const void *right) {
  const SsCouplingTargetSlot *first = (const SsCouplingTargetSlot *)left;
  const SsCouplingTargetSlot *second = (const SsCouplingTargetSlot *)right;
  if (first->cell_id < second->cell_id) {
    return -1;
  }
  if (first->cell_id > second->cell_id) {
    return 1;
  }
  return first->quadrature_index < second->quadrature_index
             ? -1
             : (first->quadrature_index > second->quadrature_index ? 1 : 0);
}

static size_t ss_coupling_unique_entity_ids(SsEntityId *values, size_t count) {
  size_t index;
  size_t unique = 0U;
  ss_sort(values, count, sizeof(*values), ss_coupling_compare_entity_ids);
  for (index = 0U; index < count; ++index) {
    if (unique == 0U || values[index] != values[unique - 1U]) {
      values[unique++] = values[index];
    }
  }
  return unique;
}

static void ss_coupling_prefixed_digest(const char digest[SS_SHA256_HEX_CAPACITY],
                                        char output[SS_COUPLING_ID_CAPACITY]) {
  memcpy(output, "sha256:", 7U);
  memcpy(output + 7U, digest, SS_SHA256_HEX_CAPACITY);
}

static int ss_coupling_hex_value(char value) {
  if (value >= '0' && value <= '9') {
    return value - '0';
  }
  if (value >= 'a' && value <= 'f') {
    return value - 'a' + 10;
  }
  return -1;
}

SsStatus ss_coupling_collective_digest(MPI_Comm communicator, const void *header,
                                       size_t header_size, const void *local_records,
                                       size_t local_record_count, size_t record_size,
                                       uint64_t global_record_count,
                                       char output[SS_COUPLING_ID_CAPACITY]) {
  unsigned char local_xor[SS_COUPLING_DIGEST_BYTES] = {0U};
  unsigned char global_xor[SS_COUPLING_DIGEST_BYTES] = {0U};
  SsCouplingDigestAggregate aggregate;
  unsigned char *payload = NULL;
  char digest[SS_SHA256_HEX_CAPACITY];
  size_t payload_size = 0U;
  size_t index;
  SsStatus local_status = SS_OK;
  SsStatus status;
  if (header == NULL || header_size == 0U || record_size == 0U || output == NULL ||
      (local_record_count != 0U && local_records == NULL)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (index = 0U; index < local_record_count; ++index) {
    const unsigned char *record =
        (const unsigned char *)local_records + index * record_size;
    char record_digest[SS_SHA256_HEX_CAPACITY];
    size_t byte;
    if (ss_sha256_bytes(record, record_size, record_digest) != SS_OK) {
      local_status = SS_ERROR_INTERNAL;
      break;
    }
    for (byte = 0U; byte < SS_COUPLING_DIGEST_BYTES; ++byte) {
      const int high = ss_coupling_hex_value(record_digest[2U * byte]);
      const int low = ss_coupling_hex_value(record_digest[2U * byte + 1U]);
      if (high < 0 || low < 0) {
        local_status = SS_ERROR_INTERNAL;
        break;
      }
      local_xor[byte] ^= (unsigned char)((unsigned int)high * 16U + (unsigned int)low);
    }
    if (local_status != SS_OK) {
      break;
    }
  }
  status = ss_status_allreduce_max(communicator, local_status);
  if (status != SS_OK) {
    return status;
  }
  if (MPI_Allreduce(local_xor, global_xor, (int)SS_COUPLING_DIGEST_BYTES,
                    MPI_UNSIGNED_CHAR, MPI_BXOR, communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  memset(&aggregate, 0, sizeof(aggregate));
  aggregate.global_record_count = global_record_count;
  memcpy(aggregate.record_digest_xor, global_xor, sizeof(global_xor));
  if (ss_coupling_checked_add(header_size, sizeof(aggregate), &payload_size) != SS_OK) {
    return SS_ERROR_OVERFLOW;
  }
  payload = calloc(payload_size, 1U);
  local_status = payload == NULL ? SS_ERROR_OUT_OF_MEMORY : SS_OK;
  status = ss_status_allreduce_max(communicator, local_status);
  if (status != SS_OK) {
    free(payload);
    return status;
  }
  memcpy(payload, header, header_size);
  memcpy(payload + header_size, &aggregate, sizeof(aggregate));
  local_status = ss_sha256_bytes(payload, payload_size, digest) == SS_OK
                     ? SS_OK
                     : SS_ERROR_INTERNAL;
  free(payload);
  status = ss_status_allreduce_max(communicator, local_status);
  if (status != SS_OK) {
    return status;
  }
  ss_coupling_prefixed_digest(digest, output);
  return SS_OK;
}

static SsStatus ss_coupling_validate_distributed_ids(MPI_Comm communicator,
                                                     int rank_count,
                                                     const SsEntityId *local_ids,
                                                     size_t local_count,
                                                     uint64_t *global_count) {
  int *send_counts = NULL;
  int *receive_counts = NULL;
  int *send_displacements = NULL;
  int *receive_displacements = NULL;
  int *cursors = NULL;
  SsEntityId *send_ids = NULL;
  SsEntityId *receive_ids = NULL;
  size_t send_total = 0U;
  size_t receive_total = 0U;
  size_t index;
  SsStatus local_status = SS_OK;
  SsStatus status;
  uint64_t local_count_u64;
  if (rank_count <= 0 || global_count == NULL ||
      (local_count != 0U && local_ids == NULL) || local_count > (size_t)INT_MAX) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  send_counts = calloc((size_t)rank_count, sizeof(*send_counts));
  receive_counts = calloc((size_t)rank_count, sizeof(*receive_counts));
  send_displacements = calloc((size_t)rank_count, sizeof(*send_displacements));
  receive_displacements = calloc((size_t)rank_count, sizeof(*receive_displacements));
  cursors = calloc((size_t)rank_count, sizeof(*cursors));
  local_status = send_counts == NULL || receive_counts == NULL ||
                         send_displacements == NULL || receive_displacements == NULL ||
                         cursors == NULL
                     ? SS_ERROR_OUT_OF_MEMORY
                     : SS_OK;
  status = ss_status_allreduce_max(communicator, local_status);
  if (status != SS_OK) {
    goto cleanup;
  }
  for (index = 0U; index < local_count; ++index) {
    const int directory_rank =
        (int)(local_ids[index] % (SsEntityId)(uint64_t)rank_count);
    ++send_counts[directory_rank];
  }
  if (MPI_Alltoall(send_counts, 1, MPI_INT, receive_counts, 1, MPI_INT, communicator) !=
      MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  for (index = 0U; index < (size_t)rank_count; ++index) {
    if (send_total > (size_t)INT_MAX - (size_t)send_counts[index] ||
        receive_total > (size_t)INT_MAX - (size_t)receive_counts[index]) {
      local_status = SS_ERROR_RESOURCE;
      break;
    }
    send_displacements[index] = (int)send_total;
    receive_displacements[index] = (int)receive_total;
    cursors[index] = send_displacements[index];
    send_total += (size_t)send_counts[index];
    receive_total += (size_t)receive_counts[index];
  }
  status = ss_status_allreduce_max(communicator, local_status);
  if (status != SS_OK) {
    goto cleanup;
  }
  send_ids = calloc(send_total == 0U ? 1U : send_total, sizeof(*send_ids));
  receive_ids = calloc(receive_total == 0U ? 1U : receive_total, sizeof(*receive_ids));
  local_status =
      send_ids == NULL || receive_ids == NULL ? SS_ERROR_OUT_OF_MEMORY : SS_OK;
  status = ss_status_allreduce_max(communicator, local_status);
  if (status != SS_OK) {
    goto cleanup;
  }
  for (index = 0U; index < local_count; ++index) {
    const int directory_rank =
        (int)(local_ids[index] % (SsEntityId)(uint64_t)rank_count);
    send_ids[(size_t)cursors[directory_rank]++] = local_ids[index];
  }
  if (MPI_Alltoallv(send_ids, send_counts, send_displacements, MPI_UINT64_T,
                    receive_ids, receive_counts, receive_displacements, MPI_UINT64_T,
                    communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  ss_sort(receive_ids, receive_total, sizeof(*receive_ids),
          ss_coupling_compare_entity_ids);
  for (index = 1U; index < receive_total; ++index) {
    if (receive_ids[index - 1U] == receive_ids[index]) {
      local_status = SS_ERROR_DUPLICATE;
      break;
    }
  }
  status = ss_status_allreduce_max(communicator, local_status);
  if (status != SS_OK) {
    goto cleanup;
  }
  local_count_u64 = (uint64_t)local_count;
  if (MPI_Allreduce(&local_count_u64, global_count, 1, MPI_UINT64_T, MPI_SUM,
                    communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  status = SS_OK;

cleanup:
  free(send_counts);
  free(receive_counts);
  free(send_displacements);
  free(receive_displacements);
  free(cursors);
  free(send_ids);
  free(receive_ids);
  return status;
}

static void ss_coupling_plan_release(SsCouplingTransferPlan *plan) {
  if (plan == NULL) {
    return;
  }
  free(plan->points);
  free(plan->local_cell_ids);
  if (plan->communicator != MPI_COMM_NULL) {
    (void)MPI_Comm_free(&plan->communicator);
  }
  free(plan);
}

static SsStatus ss_coupling_build_transfer_identity(SsCouplingTransferPlan *plan) {
  SsCouplingPlanDigestHeader header;
  SsCouplingPlanDigestRecord *records = NULL;
  SsStatus local_status;
  SsStatus status;
  size_t index;
  memset(&header, 0, sizeof(header));
  header.dimension = plan->dimension;
  header.upward_axis = plan->upward_axis;
  header.transfer_type = (uint32_t)plan->transfer_type;
  header.digest_scheme = SS_COUPLING_DIGEST_SCHEME;
  header.source_generation = plan->source_state.accepted_generation;
  header.source_mesh_generation = plan->source_state.mesh_generation;
  header.source_coordinate_generation = plan->source_state.coordinate_generation;
  ss_copy_text(header.source_mesh_identity, sizeof(header.source_mesh_identity),
               plan->source_state.mesh_identity);
  ss_copy_text(header.source_space_identity, sizeof(header.source_space_identity),
               plan->source_state.field_space_identity);
  ss_copy_text(header.source_model_identity, sizeof(header.source_model_identity),
               plan->source_state.model_identity);
  ss_copy_text(header.source_pressure_unit, sizeof(header.source_pressure_unit),
               plan->source_state.pressure_unit);
  ss_copy_text(header.target_mesh_identity, sizeof(header.target_mesh_identity),
               plan->target_mesh_identity);
  ss_copy_text(header.target_space_identity, sizeof(header.target_space_identity),
               plan->target_space_identity);
  ss_copy_text(header.target_quadrature_identity,
               sizeof(header.target_quadrature_identity),
               plan->target_quadrature_identity);
  records = calloc(plan->local_point_count == 0U ? 1U : plan->local_point_count,
                   sizeof(*records));
  local_status = records == NULL ? SS_ERROR_OUT_OF_MEMORY : SS_OK;
  status = ss_status_allreduce_max(plan->communicator, local_status);
  if (status != SS_OK) {
    free(records);
    return status;
  }
  for (index = 0U; index < plan->local_point_count; ++index) {
    const SsCouplingPlanPoint *point = &plan->points[index];
    SsCouplingPlanDigestRecord *record = &records[index];
    uint32_t coordinate;
    record->point_id = point->target.stable_point_id;
    record->target_cell_id = point->target.stable_target_cell_id;
    record->source_cell_id = point->source.stable_source_cell_id;
    record->quadrature_index = point->target.quadrature_index;
    record->dimension = point->target.dimension;
    record->physical_weight = point->target.physical_weight;
    record->cell_epsilon = point->target.cell_epsilon;
    record->saturated_unit_weight = point->target.saturated_unit_weight;
    record->unsaturated_unit_weight = point->target.unsaturated_unit_weight;
    for (coordinate = 0U; coordinate < plan->dimension; ++coordinate) {
      record->physical[coordinate] = point->target.physical[coordinate];
      record->reference[coordinate] = point->source.reference[coordinate];
    }
  }
  status = ss_coupling_collective_digest(
      plan->communicator, &header, sizeof(header), records, plan->local_point_count,
      sizeof(*records), plan->global_point_count, plan->transfer_identity);
  free(records);
  return status;
}

SsStatus ss_coupling_plan_create(MPI_Comm communicator, const SsCouplingPlanSpec *spec,
                                 SsCouplingTransferPlan **plan) {
  SsCouplingTransferPlan *created = NULL;
  SsEntityId *point_ids = NULL;
  SsCouplingTargetSlot *target_slots = NULL;
  size_t local_count = 0U;
  size_t index;
  SsStatus local_status = SS_OK;
  SsStatus status = SS_OK;
  int initialized = 0;
  int rank = 0;
  int size = 0;
  if (plan != NULL) {
    *plan = NULL;
  }
  if (communicator == MPI_COMM_NULL || MPI_Initialized(&initialized) != MPI_SUCCESS ||
      initialized == 0 || MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(communicator, &size) != MPI_SUCCESS || size <= 0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (spec == NULL || plan == NULL ||
      (spec->dimension != 2U && spec->dimension != 3U) ||
      spec->upward_axis >= spec->dimension ||
      (spec->transfer_type != SS_COUPLING_TRANSFER_SAME_MESH &&
       spec->transfer_type != SS_COUPLING_TRANSFER_DIFFERENT_MESH) ||
      spec->requested_threads <= 0 ||
#ifndef _OPENMP
      spec->requested_threads != 1 ||
#endif
      !ss_coupling_valid_identity(spec->target_mesh_identity,
                                  sizeof(spec->target_mesh_identity)) ||
      !ss_coupling_valid_identity(spec->target_space_identity,
                                  sizeof(spec->target_space_identity)) ||
      !ss_coupling_valid_identity(spec->target_quadrature_identity,
                                  sizeof(spec->target_quadrature_identity)) ||
      spec->geometry.target_count == NULL || spec->geometry.target_at == NULL ||
      (spec->transfer_type == SS_COUPLING_TRANSFER_DIFFERENT_MESH
           ? spec->same_mesh_map != NULL || spec->geometry.locate == NULL
           : spec->same_mesh_map == NULL && spec->geometry.locate == NULL) ||
      spec->geometry.evaluate_source == NULL || spec->geometry.source_state == NULL) {
    local_status = SS_ERROR_INVALID_ARGUMENT;
  } else {
    local_status = spec->geometry.target_count(spec->geometry.context, &local_count);
  }
  if (local_status == SS_OK && (local_count > (size_t)INT_MAX ||
                                local_count > SIZE_MAX / sizeof(SsCouplingPlanPoint))) {
    local_status = SS_ERROR_RESOURCE;
  }
  const int transfer = local_status == SS_OK ? (int)spec->transfer_type : 0;
  int configuration[] = {(int)local_status, transfer, -transfer};
  if (MPI_Allreduce(MPI_IN_PLACE, configuration, 3, MPI_INT, MPI_MAX, communicator) !=
      MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (configuration[0] != SS_OK) {
    return (SsStatus)configuration[0];
  }
  if (configuration[1] != -configuration[2]) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  created = calloc(1U, sizeof(*created));
  point_ids = calloc(local_count == 0U ? 1U : local_count, sizeof(*point_ids));
  target_slots = calloc(local_count == 0U ? 1U : local_count, sizeof(*target_slots));
  local_status = created == NULL || point_ids == NULL || target_slots == NULL
                     ? SS_ERROR_OUT_OF_MEMORY
                     : SS_OK;
  if (created != NULL) {
    created->communicator = MPI_COMM_NULL;
    created->points =
        calloc(local_count == 0U ? 1U : local_count, sizeof(*created->points));
    created->local_cell_ids =
        calloc(local_count == 0U ? 1U : local_count, sizeof(*created->local_cell_ids));
    if (created->points == NULL || created->local_cell_ids == NULL) {
      local_status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(communicator, local_status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (MPI_Comm_dup(communicator, &created->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  created->rank = rank;
  created->size = size;
  created->dimension = spec->dimension;
  created->upward_axis = spec->upward_axis;
  created->transfer_type = spec->transfer_type;
  created->requested_threads = spec->requested_threads;
  created->openmp_work_threshold = spec->openmp_work_threshold;
  created->geometry = spec->geometry;
  created->local_point_count = local_count;
  ss_copy_text(created->target_mesh_identity, sizeof(created->target_mesh_identity),
               spec->target_mesh_identity);
  ss_copy_text(created->target_space_identity, sizeof(created->target_space_identity),
               spec->target_space_identity);
  ss_copy_text(created->target_quadrature_identity,
               sizeof(created->target_quadrature_identity),
               spec->target_quadrature_identity);
  for (index = 0U; index < local_count; ++index) {
    local_status = spec->geometry.target_at(spec->geometry.context, index,
                                            &created->points[index].target);
    if (local_status != SS_OK ||
        !ss_coupling_target_valid(&created->points[index].target, spec->dimension)) {
      local_status = local_status == SS_OK ? SS_ERROR_INTEGRITY : local_status;
      break;
    }
  }
  status = ss_status_allreduce_max(communicator, local_status);
  if (status != SS_OK) {
    goto cleanup;
  }
  ss_sort(created->points, local_count, sizeof(*created->points),
          ss_coupling_compare_plan_points);
  for (index = 0U; index < local_count; ++index) {
    const SsTransferTargetPoint *target = &created->points[index].target;
    point_ids[index] = target->stable_point_id;
    created->local_cell_ids[index] = target->stable_target_cell_id;
    target_slots[index].cell_id = target->stable_target_cell_id;
    target_slots[index].quadrature_index = target->quadrature_index;
    if (index != 0U && created->points[index - 1U].target.stable_target_cell_id ==
                           target->stable_target_cell_id) {
      const SsTransferTargetPoint *prior = &created->points[index - 1U].target;
      if (prior->cell_epsilon != target->cell_epsilon ||
          prior->saturated_unit_weight != target->saturated_unit_weight ||
          prior->unsaturated_unit_weight != target->unsaturated_unit_weight) {
        local_status = SS_ERROR_INTEGRITY;
        break;
      }
    }
  }
  ss_sort(target_slots, local_count, sizeof(*target_slots),
          ss_coupling_compare_target_slots);
  for (index = 1U; index < local_count; ++index) {
    if (target_slots[index - 1U].cell_id == target_slots[index].cell_id &&
        target_slots[index - 1U].quadrature_index ==
            target_slots[index].quadrature_index) {
      local_status = SS_ERROR_DUPLICATE;
      break;
    }
  }
  status = ss_status_allreduce_max(communicator, local_status);
  if (status != SS_OK) {
    goto cleanup;
  }
  created->local_cell_count =
      ss_coupling_unique_entity_ids(created->local_cell_ids, local_count);
  status = ss_coupling_validate_distributed_ids(
      communicator, size, point_ids, local_count, &created->global_point_count);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (created->global_point_count == 0U) {
    status = SS_ERROR_NOT_FOUND;
    goto cleanup;
  }
  status = ss_coupling_validate_distributed_ids(
      communicator, size, created->local_cell_ids, created->local_cell_count,
      &created->global_cell_count);
  if (status != SS_OK) {
    goto cleanup;
  }
  status = ss_coupling_read_collective_source_state(communicator, &created->geometry,
                                                    &created->source_state);
  if (status != SS_OK) {
    goto cleanup;
  }
  local_status = created->transfer_type == SS_COUPLING_TRANSFER_SAME_MESH &&
                         strcmp(created->target_mesh_identity,
                                created->source_state.mesh_identity) != 0
                     ? SS_ERROR_TYPE_MISMATCH
                     : SS_OK;
  status = ss_status_allreduce_max(communicator, local_status);
  if (status != SS_OK) {
    goto cleanup;
  }
  status = ss_coupling_locate_points(created, spec->same_mesh_map);
  if (status != SS_OK) {
    goto cleanup;
  }
  {
    SsTransferSourceState current_state;
    status = ss_coupling_read_collective_source_state(communicator, &created->geometry,
                                                      &current_state);
    if (status != SS_OK ||
        !ss_coupling_source_state_equal(&current_state, &created->source_state)) {
      status = status == SS_OK ? SS_ERROR_INVALID_STATE : status;
      goto cleanup;
    }
  }
  status = ss_coupling_build_transfer_identity(created);
  if (status != SS_OK) {
    goto cleanup;
  }
  *plan = created;
  created = NULL;
  status = SS_OK;

cleanup:
  free(point_ids);
  free(target_slots);
  ss_coupling_plan_release(created);
  return status;
}

SsStatus ss_coupling_plan_summary(const SsCouplingTransferPlan *plan,
                                  SsCouplingPlanSummary *summary) {
  size_t point_bytes = 0U;
  size_t cell_bytes = 0U;
  size_t resident_bytes = sizeof(*plan);
  if (plan == NULL || summary == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (ss_coupling_checked_product(plan->local_point_count, sizeof(*plan->points),
                                  &point_bytes) != SS_OK ||
      ss_coupling_checked_product(plan->local_cell_count, sizeof(*plan->local_cell_ids),
                                  &cell_bytes) != SS_OK ||
      ss_coupling_checked_add(resident_bytes, point_bytes, &resident_bytes) != SS_OK ||
      ss_coupling_checked_add(resident_bytes, cell_bytes, &resident_bytes) != SS_OK) {
    return SS_ERROR_OVERFLOW;
  }
  memset(summary, 0, sizeof(*summary));
  summary->dimension = plan->dimension;
  summary->transfer_type = plan->transfer_type;
  summary->local_target_points = (uint64_t)plan->local_point_count;
  summary->global_target_points = plan->global_point_count;
  summary->local_target_cells = (uint64_t)plan->local_cell_count;
  summary->global_target_cells = plan->global_cell_count;
  summary->resident_target_points = (uint64_t)plan->local_point_count;
  summary->source_mesh_generation = plan->source_state.mesh_generation;
  summary->source_coordinate_generation = plan->source_state.coordinate_generation;
  summary->resident_bytes = resident_bytes;
  summary->requested_threads = plan->requested_threads;
  ss_copy_text(summary->transfer_identity, sizeof(summary->transfer_identity),
               plan->transfer_identity);
  return SS_OK;
}

SsStatus ss_coupling_plan_destroy(SsCouplingTransferPlan **plan) {
  if (plan == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  ss_coupling_plan_release(*plan);
  *plan = NULL;
  return SS_OK;
}
