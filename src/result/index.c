/* Distributed stable-ID ordering and value routes for contiguous file blocks.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "result_internal.h"
#include "slope_internal/core/sort.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  uint64_t stable_id;
  uint64_t origin_rank;
  uint64_t origin_index;
  uint64_t origin_exchange_index;
} SsResultRouteRecord;

typedef struct {
  uint64_t origin_index;
  uint64_t file_index;
} SsResultReturnRecord;

static int ss_result_route_compare(const void *left_raw, const void *right_raw) {
  const SsResultRouteRecord *left = (const SsResultRouteRecord *)left_raw;
  const SsResultRouteRecord *right = (const SsResultRouteRecord *)right_raw;
  if (left->stable_id != right->stable_id) {
    return left->stable_id < right->stable_id ? -1 : 1;
  }
  if (left->origin_rank != right->origin_rank) {
    return left->origin_rank < right->origin_rank ? -1 : 1;
  }
  return left->origin_index < right->origin_index   ? -1
         : left->origin_index > right->origin_index ? 1
                                                    : 0;
}

static int ss_result_entry_compare(const void *left_raw, const void *right_raw) {
  const SsResultFileIndexEntry *left = (const SsResultFileIndexEntry *)left_raw;
  const SsResultFileIndexEntry *right = (const SsResultFileIndexEntry *)right_raw;
  return left->file_index < right->file_index   ? -1
         : left->file_index > right->file_index ? 1
                                                : 0;
}

static int ss_result_range_owner(SsEntityId stable_id, int size) {
  uint64_t width;
  uint64_t owner;
  if (size <= 1) {
    return 0;
  }
  width = UINT64_MAX / (uint64_t)(uint32_t)size + UINT64_C(1);
  owner = stable_id / width;
  return owner >= (uint64_t)(uint32_t)size ? size - 1 : (int)owner;
}

static SsStatus ss_result_counts_offsets(const int *counts, int size, int *offsets,
                                         int *total) {
  int rank;
  if (counts == NULL || offsets == NULL || total == NULL || size <= 0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  offsets[0] = 0;
  for (rank = 1; rank < size; ++rank) {
    if (offsets[rank - 1] > INT_MAX - counts[rank - 1]) {
      return SS_ERROR_OVERFLOW;
    }
    offsets[rank] = offsets[rank - 1] + counts[rank - 1];
  }
  if (offsets[size - 1] > INT_MAX - counts[size - 1]) {
    return SS_ERROR_OVERFLOW;
  }
  *total = offsets[size - 1] + counts[size - 1];
  return SS_OK;
}

static SsStatus ss_result_validate_local_spec(const SsResultFileIndexSpec *spec,
                                              int rank) {
  unsigned char *seen = NULL;
  size_t index;
  SsStatus status = SS_OK;
  if (spec == NULL || spec->global_count == 0U || spec->local_count > (size_t)INT_MAX ||
      (spec->local_count != 0U && spec->stable_ids == NULL) || rank < 0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  seen = (unsigned char *)calloc(spec->local_count == 0U ? 1U : spec->local_count,
                                 sizeof(*seen));
  if (seen == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (index = 0U; index < spec->local_count; ++index) {
    const size_t slot = spec->local_slots == NULL ? index : spec->local_slots[index];
    if (slot >= spec->local_count || seen[slot] != 0U) {
      status = SS_ERROR_DUPLICATE;
      break;
    }
    seen[slot] = 1U;
  }
  free(seen);
  return status;
}

SsStatus ss_result_file_index_plan_create(const SsResultFileIndexSpec *spec,
                                          SsResultFileIndexPlan **result) {
  SsResultFileIndexPlan *plan = NULL;
  SsResultRouteRecord *send_records = NULL;
  SsResultRouteRecord *received_records = NULL;
  SsResultReturnRecord *return_send = NULL;
  SsResultReturnRecord *return_received = NULL;
  SsResultFileIndexEntry *ordered_entries = NULL;
  unsigned char *entry_seen = NULL;
  unsigned char *receive_position_seen = NULL;
  size_t *send_local_slots = NULL;
  size_t *receive_sorted_positions = NULL;
  int *send_counts = NULL;
  int *receive_counts = NULL;
  int *send_offsets = NULL;
  int *receive_offsets = NULL;
  int *cursor = NULL;
  int *return_send_counts = NULL;
  int *return_receive_counts = NULL;
  int *return_send_offsets = NULL;
  int *return_receive_offsets = NULL;
  int *return_cursor = NULL;
  MPI_Datatype route_type = MPI_DATATYPE_NULL;
  MPI_Datatype return_type = MPI_DATATYPE_NULL;
  uint64_t local_count_u64;
  uint64_t global_count = 0U;
  uint64_t sorted_offset = 0U;
  uint64_t local_exchange = 0U;
  uint64_t global_exchange = 0U;
  int rank = -1;
  int size = 0;
  int total_receive = 0;
  int total_return_receive = 0;
  size_t index;
  SsStatus status;
  if (result == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *result = NULL;
  if (spec == NULL || MPI_Comm_rank(spec->communicator, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(spec->communicator, &size) != MPI_SUCCESS || size <= 0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_result_validate_local_spec(spec, rank);
  status = ss_status_allreduce_max(spec->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  local_count_u64 = (uint64_t)spec->local_count;
  if (MPI_Allreduce(&local_count_u64, &global_count, 1, MPI_UINT64_T, MPI_SUM,
                    spec->communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (global_count != spec->global_count) {
    return SS_ERROR_INTEGRITY;
  }
  plan = (SsResultFileIndexPlan *)calloc(1U, sizeof(*plan));
  if (plan != NULL) {
    plan->communicator = MPI_COMM_NULL;
  }
  send_counts = (int *)calloc((size_t)size, sizeof(*send_counts));
  receive_counts = (int *)calloc((size_t)size, sizeof(*receive_counts));
  send_offsets = (int *)calloc((size_t)size, sizeof(*send_offsets));
  receive_offsets = (int *)calloc((size_t)size, sizeof(*receive_offsets));
  cursor = (int *)calloc((size_t)size, sizeof(*cursor));
  return_send_counts = (int *)calloc((size_t)size, sizeof(*return_send_counts));
  return_receive_counts = (int *)calloc((size_t)size, sizeof(*return_receive_counts));
  return_send_offsets = (int *)calloc((size_t)size, sizeof(*return_send_offsets));
  return_receive_offsets = (int *)calloc((size_t)size, sizeof(*return_receive_offsets));
  return_cursor = (int *)calloc((size_t)size, sizeof(*return_cursor));
  send_records = (SsResultRouteRecord *)calloc(
      spec->local_count == 0U ? 1U : spec->local_count, sizeof(*send_records));
  send_local_slots = (size_t *)calloc(spec->local_count == 0U ? 1U : spec->local_count,
                                      sizeof(*send_local_slots));
  if (plan == NULL || send_counts == NULL || receive_counts == NULL ||
      send_offsets == NULL || receive_offsets == NULL || cursor == NULL ||
      return_send_counts == NULL || return_receive_counts == NULL ||
      return_send_offsets == NULL || return_receive_offsets == NULL ||
      return_cursor == NULL || send_records == NULL || send_local_slots == NULL) {
    status = SS_ERROR_OUT_OF_MEMORY;
  }
  status = ss_status_allreduce_max(spec->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (plan == NULL || send_counts == NULL || receive_counts == NULL ||
      send_offsets == NULL || receive_offsets == NULL || cursor == NULL ||
      return_send_counts == NULL || return_receive_counts == NULL ||
      return_send_offsets == NULL || return_receive_offsets == NULL ||
      return_cursor == NULL || send_records == NULL || send_local_slots == NULL) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  if (MPI_Comm_dup(spec->communicator, &plan->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(spec->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  plan->rank = rank;
  plan->size = size;
  plan->local_count = spec->local_count;
  plan->entries = (SsResultFileIndexEntry *)calloc(
      spec->local_count == 0U ? 1U : spec->local_count, sizeof(*plan->entries));
  entry_seen = (unsigned char *)calloc(spec->local_count == 0U ? 1U : spec->local_count,
                                       sizeof(*entry_seen));
  if (plan->entries == NULL || entry_seen == NULL) {
    status = SS_ERROR_OUT_OF_MEMORY;
  }
  for (index = 0U; status == SS_OK && index < spec->local_count; ++index) {
    const int owner = ss_result_range_owner(spec->stable_ids[index], size);
    if (send_counts[owner] == INT_MAX) {
      status = SS_ERROR_OVERFLOW;
      break;
    }
    ++send_counts[owner];
  }
  if (status == SS_OK) {
    status = ss_result_counts_offsets(send_counts, size, send_offsets,
                                      &total_return_receive);
  }
  if (status == SS_OK) {
    memcpy(cursor, send_offsets, (size_t)size * sizeof(*cursor));
    for (index = 0U; index < spec->local_count; ++index) {
      const int owner = ss_result_range_owner(spec->stable_ids[index], size);
      const int send_position = cursor[owner]++;
      SsResultRouteRecord *record = &send_records[send_position];
      record->stable_id = spec->stable_ids[index];
      record->origin_rank = (uint64_t)(uint32_t)rank;
      record->origin_index = (uint64_t)index;
      record->origin_exchange_index =
          (uint64_t)(uint32_t)(send_position - send_offsets[owner]);
      send_local_slots[send_position] =
          spec->local_slots == NULL ? index : spec->local_slots[index];
    }
  }
  status = ss_status_allreduce_max(spec->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (plan == NULL || plan->entries == NULL || entry_seen == NULL ||
      send_counts == NULL || receive_counts == NULL || send_offsets == NULL ||
      receive_offsets == NULL || cursor == NULL || return_send_counts == NULL ||
      return_receive_counts == NULL || return_send_offsets == NULL ||
      return_receive_offsets == NULL || return_cursor == NULL || send_records == NULL ||
      send_local_slots == NULL) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  if (MPI_Type_contiguous(4, MPI_UINT64_T, &route_type) != MPI_SUCCESS ||
      MPI_Type_commit(&route_type) != MPI_SUCCESS ||
      MPI_Type_contiguous(2, MPI_UINT64_T, &return_type) != MPI_SUCCESS ||
      MPI_Type_commit(&return_type) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(spec->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (MPI_Alltoall(send_counts, 1, MPI_INT, receive_counts, 1, MPI_INT,
                   spec->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  status =
      ss_result_counts_offsets(receive_counts, size, receive_offsets, &total_receive);
  if (status == SS_OK) {
    received_records = (SsResultRouteRecord *)calloc(
        total_receive == 0 ? 1U : (size_t)total_receive, sizeof(*received_records));
    return_send = (SsResultReturnRecord *)calloc(
        total_receive == 0 ? 1U : (size_t)total_receive, sizeof(*return_send));
    plan->sorted_ids = (SsEntityId *)calloc(
        total_receive == 0 ? 1U : (size_t)total_receive, sizeof(*plan->sorted_ids));
    receive_sorted_positions =
        (size_t *)calloc(total_receive == 0 ? 1U : (size_t)total_receive,
                         sizeof(*receive_sorted_positions));
    plan->sorted_receive_positions =
        calloc(total_receive == 0 ? 1U : (size_t)total_receive,
               sizeof(*plan->sorted_receive_positions));
    receive_position_seen =
        (unsigned char *)calloc(total_receive == 0 ? 1U : (size_t)total_receive,
                                sizeof(*receive_position_seen));
    if (received_records == NULL || return_send == NULL || plan->sorted_ids == NULL ||
        receive_sorted_positions == NULL || plan->sorted_receive_positions == NULL ||
        receive_position_seen == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(spec->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (received_records == NULL || return_send == NULL || plan->sorted_ids == NULL ||
      receive_sorted_positions == NULL || receive_position_seen == NULL) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  if (MPI_Alltoallv(send_records, send_counts, send_offsets, route_type,
                    received_records, receive_counts, receive_offsets, route_type,
                    spec->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  ss_sort(received_records, (size_t)total_receive, sizeof(*received_records),
          ss_result_route_compare);
  for (index = 1U; index < (size_t)total_receive; ++index) {
    if (received_records[index].stable_id == received_records[index - 1U].stable_id) {
      status = SS_ERROR_DUPLICATE;
      break;
    }
  }
  status = ss_status_allreduce_max(spec->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  local_count_u64 = (uint64_t)(uint32_t)total_receive;
  if (MPI_Exscan(&local_count_u64, &sorted_offset, 1, MPI_UINT64_T, MPI_SUM,
                 spec->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  if (rank == 0) {
    sorted_offset = 0U;
  }
  plan->sorted_count = (size_t)total_receive;
  plan->sorted_offset = sorted_offset;
  for (index = 0U; index < (size_t)total_receive; ++index) {
    const int origin = (int)received_records[index].origin_rank;
    const uint64_t exchange_index = received_records[index].origin_exchange_index;
    size_t receive_position;
    plan->sorted_ids[index] = received_records[index].stable_id;
    if (origin < 0 || origin >= size || return_send_counts[origin] == INT_MAX ||
        exchange_index >= (uint64_t)(uint32_t)receive_counts[origin]) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    receive_position = (size_t)receive_offsets[origin] + (size_t)exchange_index;
    if (receive_position >= (size_t)total_receive ||
        receive_position_seen[receive_position] != 0U) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    receive_position_seen[receive_position] = 1U;
    receive_sorted_positions[receive_position] = index;
    plan->sorted_receive_positions[index] = receive_position;
    ++return_send_counts[origin];
  }
  if (status == SS_OK) {
    status = ss_result_counts_offsets(return_send_counts, size, return_send_offsets,
                                      &total_return_receive);
  }
  if (status == SS_OK) {
    memcpy(return_cursor, return_send_offsets, (size_t)size * sizeof(*return_cursor));
    for (index = 0U; index < (size_t)total_receive; ++index) {
      const int origin = (int)received_records[index].origin_rank;
      SsResultReturnRecord *record = &return_send[return_cursor[origin]++];
      record->origin_index = received_records[index].origin_index;
      record->file_index = sorted_offset + (uint64_t)index;
    }
  }
  status = ss_status_allreduce_max(spec->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (MPI_Alltoall(return_send_counts, 1, MPI_INT, return_receive_counts, 1, MPI_INT,
                   spec->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  status = ss_result_counts_offsets(return_receive_counts, size, return_receive_offsets,
                                    &total_return_receive);
  if (status == SS_OK) {
    return_received = (SsResultReturnRecord *)calloc(
        total_return_receive == 0 ? 1U : (size_t)total_return_receive,
        sizeof(*return_received));
    if (return_received == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(spec->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (return_received == NULL) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  if (MPI_Alltoallv(return_send, return_send_counts, return_send_offsets, return_type,
                    return_received, return_receive_counts, return_receive_offsets,
                    return_type, spec->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  if ((size_t)total_return_receive != spec->local_count) {
    status = SS_ERROR_INTEGRITY;
  }
  for (index = 0U; status == SS_OK && index < (size_t)total_return_receive; ++index) {
    const uint64_t origin_index = return_received[index].origin_index;
    SsResultFileIndexEntry *entry;
    if (origin_index >= (uint64_t)spec->local_count) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    entry = &plan->entries[(size_t)origin_index];
    if (entry_seen[(size_t)origin_index] != 0U) {
      status = SS_ERROR_DUPLICATE;
      break;
    }
    entry_seen[(size_t)origin_index] = 1U;
    entry->stable_id = spec->stable_ids[(size_t)origin_index];
    entry->local_slot = spec->local_slots == NULL
                            ? (size_t)origin_index
                            : spec->local_slots[(size_t)origin_index];
    entry->file_index = return_received[index].file_index;
  }
  status = ss_status_allreduce_max(spec->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  ordered_entries = (SsResultFileIndexEntry *)malloc(
      (spec->local_count == 0U ? 1U : spec->local_count) * sizeof(*ordered_entries));
  if (ordered_entries == NULL) {
    status = SS_ERROR_OUT_OF_MEMORY;
  } else {
    memcpy(ordered_entries, plan->entries,
           spec->local_count * sizeof(*ordered_entries));
    ss_sort(ordered_entries, spec->local_count, sizeof(*ordered_entries),
            ss_result_entry_compare);
    plan->summary.contiguous_runs = spec->local_count == 0U ? 0U : 1U;
    for (index = 1U; index < spec->local_count; ++index) {
      if (ordered_entries[index].file_index !=
          ordered_entries[index - 1U].file_index + 1U) {
        ++plan->summary.contiguous_runs;
      }
    }
  }
  status = ss_status_allreduce_max(spec->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  local_exchange = (uint64_t)spec->local_count * (uint64_t)sizeof(SsResultRouteRecord) +
                   (uint64_t)total_receive * (uint64_t)sizeof(SsResultReturnRecord);
  if (MPI_Allreduce(&local_exchange, &global_exchange, 1, MPI_UINT64_T, MPI_SUM,
                    spec->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  plan->summary.global_count = global_count;
  plan->summary.local_count = spec->local_count;
  plan->summary.exchanged_bytes = global_exchange;
  plan->summary.root_gathered_value_bytes = 0U;
  plan->send_counts = send_counts;
  send_counts = NULL;
  plan->send_offsets = send_offsets;
  send_offsets = NULL;
  plan->receive_counts = receive_counts;
  receive_counts = NULL;
  plan->receive_offsets = receive_offsets;
  receive_offsets = NULL;
  plan->send_local_slots = send_local_slots;
  send_local_slots = NULL;
  plan->receive_sorted_positions = receive_sorted_positions;
  receive_sorted_positions = NULL;
  *result = plan;
  plan = NULL;

cleanup:
  if (route_type != MPI_DATATYPE_NULL) {
    (void)MPI_Type_free(&route_type);
  }
  if (return_type != MPI_DATATYPE_NULL) {
    (void)MPI_Type_free(&return_type);
  }
  free(send_records);
  free(received_records);
  free(return_send);
  free(return_received);
  free(ordered_entries);
  free(entry_seen);
  free(receive_position_seen);
  free(send_local_slots);
  free(receive_sorted_positions);
  free(send_counts);
  free(receive_counts);
  free(send_offsets);
  free(receive_offsets);
  free(cursor);
  free(return_send_counts);
  free(return_receive_counts);
  free(return_send_offsets);
  free(return_receive_offsets);
  free(return_cursor);
  if (plan != NULL) {
    (void)ss_result_file_index_plan_destroy(&plan);
  }
  return status;
}

SsStatus ss_result_file_index_plan_entry(const SsResultFileIndexPlan *plan,
                                         size_t local_index,
                                         SsResultFileIndexEntry *entry) {
  if (plan == NULL || entry == NULL || local_index >= plan->local_count) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *entry = plan->entries[local_index];
  return SS_OK;
}

SsStatus ss_result_file_index_plan_summary(const SsResultFileIndexPlan *plan,
                                           SsResultFileIndexSummary *summary) {
  if (plan == NULL || summary == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *summary = plan->summary;
  return SS_OK;
}

SsStatus ss_result_file_index_plan_destroy(SsResultFileIndexPlan **plan) {
  SsStatus status = SS_OK;
  if (plan == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (*plan == NULL) {
    return SS_OK;
  }
  if ((*plan)->communicator != MPI_COMM_NULL &&
      MPI_Comm_free(&(*plan)->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  free((*plan)->entries);
  free((*plan)->sorted_ids);
  free((*plan)->send_counts);
  free((*plan)->send_offsets);
  free((*plan)->receive_counts);
  free((*plan)->receive_offsets);
  free((*plan)->send_local_slots);
  free((*plan)->receive_sorted_positions);
  free((*plan)->sorted_receive_positions);
  free(*plan);
  *plan = NULL;
  return status;
}

/* Pull each row from its source in the same permutation cycle. Both maps
 * describe permutations verified when the file-index plan was created. */
static void ss_result_reorder_rows(size_t count, const size_t *source_row,
                                   size_t row_bytes, unsigned char *values,
                                   unsigned char *visited, unsigned char *first) {
  for (size_t row = 0U; row < count; ++row) {
    if (visited[row] != 0U || source_row[row] == row) {
      continue;
    }
    memcpy(first, values + row * row_bytes, row_bytes);
    size_t current = row;
    while (source_row[current] != row) {
      const size_t next = source_row[current];
      memcpy(values + current * row_bytes, values + next * row_bytes, row_bytes);
      visited[current] = 1U;
      current = next;
    }
    memcpy(values + current * row_bytes, first, row_bytes);
    visited[current] = 1U;
  }
}

static SsStatus ss_result_route_rows(const SsResultFileIndexPlan *plan,
                                     size_t values_per_row, size_t value_size,
                                     MPI_Datatype value_type, const void *local_input,
                                     void *local_output, void **file_values,
                                     int reverse) {
  size_t row_bytes = 0U, local_bytes = 0U, file_bytes = 0U;
  unsigned char *exchange = NULL, *sorted = NULL, *visited = NULL, *first = NULL;
  MPI_Datatype row_type = MPI_DATATYPE_NULL;
  SsStatus status = SS_OK;
  if (plan == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (values_per_row == 0U || value_size == 0U || value_type == MPI_DATATYPE_NULL ||
      file_values == NULL || (reverse && *file_values == NULL) ||
      (plan->local_count != 0U && (reverse ? local_output : local_input) == NULL)) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else if (values_per_row > (size_t)INT_MAX ||
             ss_checked_mul_size(values_per_row, value_size, &row_bytes) != SS_OK ||
             ss_checked_mul_size(plan->local_count, row_bytes, &local_bytes) != SS_OK ||
             ss_checked_mul_size(plan->sorted_count, row_bytes, &file_bytes) != SS_OK) {
    status = SS_ERROR_OVERFLOW;
  }
  if (status == SS_OK) {
    exchange = malloc(local_bytes == 0U ? 1U : local_bytes);
    sorted = reverse ? *file_values : malloc(file_bytes == 0U ? 1U : file_bytes);
    visited = calloc(plan->sorted_count == 0U ? 1U : plan->sorted_count, 1U);
    first = malloc(row_bytes);
    if (exchange == NULL || sorted == NULL || visited == NULL || first == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    } else if (MPI_Type_contiguous((int)values_per_row, value_type, &row_type) !=
                   MPI_SUCCESS ||
               MPI_Type_commit(&row_type) != MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  status = ss_status_allreduce_max(plan->communicator, status);
  if (status == SS_OK && reverse) {
    ss_result_reorder_rows(plan->sorted_count, plan->receive_sorted_positions,
                           row_bytes, sorted, visited, first);
    if (MPI_Alltoallv(sorted, plan->receive_counts, plan->receive_offsets, row_type,
                      exchange, plan->send_counts, plan->send_offsets, row_type,
                      plan->communicator) != MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
    for (size_t i = 0U; status == SS_OK && i < plan->local_count; ++i) {
      memcpy((unsigned char *)local_output + plan->send_local_slots[i] * row_bytes,
             exchange + i * row_bytes, row_bytes);
    }
  } else if (status == SS_OK) {
    for (size_t i = 0U; i < plan->local_count; ++i) {
      memcpy(exchange + i * row_bytes,
             (const unsigned char *)local_input + plan->send_local_slots[i] * row_bytes,
             row_bytes);
    }
    if (MPI_Alltoallv(exchange, plan->send_counts, plan->send_offsets, row_type, sorted,
                      plan->receive_counts, plan->receive_offsets, row_type,
                      plan->communicator) != MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    } else {
      ss_result_reorder_rows(plan->sorted_count, plan->sorted_receive_positions,
                             row_bytes, sorted, visited, first);
    }
  }
  if (row_type != MPI_DATATYPE_NULL) {
    (void)MPI_Type_free(&row_type);
  }
  free(first);
  free(visited);
  free(exchange);
  status = ss_status_allreduce_max(plan->communicator, status);
  if (!reverse) {
    if (status == SS_OK) {
      *file_values = sorted;
    } else {
      free(sorted);
    }
  }
  return status;
}

SsStatus ss_result_gather_indexed_rows(const SsResultFileIndexPlan *plan,
                                       size_t values_per_row, size_t value_size,
                                       MPI_Datatype value_type,
                                       const void *local_values, void **sorted_values) {
  if (sorted_values != NULL) {
    *sorted_values = NULL;
  }
  return ss_result_route_rows(plan, values_per_row, value_size, value_type,
                              local_values, NULL, sorted_values, 0);
}

SsStatus ss_result_scatter_indexed_rows(const SsResultFileIndexPlan *plan,
                                        size_t values_per_row, size_t value_size,
                                        MPI_Datatype value_type, void *sorted_values,
                                        void *local_values) {
  return ss_result_route_rows(plan, values_per_row, value_size, value_type, NULL,
                              local_values, &sorted_values, 1);
}
