/* Batch routing and joins without a global table or a root-owned exchange.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/mesh/distribution.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/core/sort.h"

#include <limits.h>
#include <petscsys.h>
#include <stdlib.h>
#include <string.h>

uint64_t ss_mesh_distribution_hash(uint64_t key) {
  key ^= key >> 30U;
  key *= UINT64_C(0xbf58476d1ce4e5b9);
  key ^= key >> 27U;
  key *= UINT64_C(0x94d049bb133111eb);
  return key ^ (key >> 31U);
}

typedef struct {
  uint64_t *counts;
  size_t *offsets;
  size_t total;
  unsigned char *data;
} SsMeshRouteBuffer;

static SsStatus ss_mesh_route_allocate(SsMeshRouteBuffer *buffer, int ranks,
                                       size_t record_size) {
  for (int rank = 0; rank < ranks; ++rank) {
    buffer->offsets[rank] = buffer->total;
    if (buffer->counts[rank] > SIZE_MAX - buffer->total) {
      return SS_ERROR_OVERFLOW;
    }
    buffer->total += (size_t)buffer->counts[rank];
  }
  if (buffer->total > SIZE_MAX / record_size) {
    return SS_ERROR_OVERFLOW;
  }
  buffer->data = malloc((buffer->total == 0U ? 1U : buffer->total) * record_size);
  return buffer->data == NULL ? SS_ERROR_OUT_OF_MEMORY : SS_OK;
}

static SsStatus ss_mesh_route_exchange(MPI_Comm communicator, int ranks,
                                       const SsMeshRouteBuffer *send,
                                       SsMeshRouteBuffer *receive, size_t record_size) {
  MPI_Comm private_comm = MPI_COMM_NULL;
  MPI_Request *requests = calloc((size_t)ranks * 2U, sizeof(*requests));
  MPI_Datatype record = MPI_DATATYPE_NULL;
  int tag = 0;
  SsStatus status = requests == NULL ? SS_ERROR_OUT_OF_MEMORY : SS_OK;
  if (status == SS_OK &&
      (MPI_Type_contiguous((int)record_size, MPI_BYTE, &record) != MPI_SUCCESS ||
       MPI_Type_commit(&record) != MPI_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK &&
      PetscCommDuplicate(communicator, &private_comm, &tag) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    /* Limit individual messages for MPI implementations with int counts.
     * Partners know their matching lengths and need no global round count. */
    const size_t chunk = (size_t)INT_MAX / record_size;
    size_t maximum = 0U;
    for (int rank = 0; rank < ranks; ++rank) {
      const size_t count = (size_t)PetscMax(send->counts[rank], receive->counts[rank]);
      maximum = PetscMax(count, maximum);
    }
    for (size_t begin = 0U; begin < maximum;
         begin += PetscMin(chunk, maximum - begin)) {
      int pending = 0;
      for (int rank = 0; rank < ranks; ++rank) {
        if (begin < receive->counts[rank]) {
          const int count = (int)PetscMin(chunk, (size_t)receive->counts[rank] - begin);
          if (MPI_Irecv(receive->data + (receive->offsets[rank] + begin) * record_size,
                        count, record, rank, tag, private_comm,
                        &requests[pending++]) != MPI_SUCCESS) {
            status = SS_ERROR_INTERNAL;
          }
        }
      }
      for (int rank = 0; rank < ranks; ++rank) {
        if (begin < send->counts[rank]) {
          const int count = (int)PetscMin(chunk, (size_t)send->counts[rank] - begin);
          if (MPI_Isend(send->data + (send->offsets[rank] + begin) * record_size, count,
                        record, rank, tag, private_comm,
                        &requests[pending++]) != MPI_SUCCESS) {
            status = SS_ERROR_INTERNAL;
          }
        }
      }
      if (MPI_Waitall(pending, requests, MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
        status = SS_ERROR_INTERNAL;
      }
    }
  }
  if (private_comm != MPI_COMM_NULL) {
    (void)PetscCommDestroy(&private_comm);
  }
  if (record != MPI_DATATYPE_NULL) {
    (void)MPI_Type_free(&record);
  }
  free(requests);
  return ss_status_allreduce_max(communicator, status);
}

SsStatus ss_mesh_route_records(MPI_Comm communicator, const void *records,
                               size_t record_count, size_t record_size,
                               SsMeshRouteOwner owner, const void *context,
                               void **received, size_t *received_count) {
  SsMeshRouteBuffer send = {0}, receive = {0};
  size_t *cursor = NULL;
  const unsigned char *bytes = records;
  int ranks = 0;
  SsStatus status = SS_OK;
  if (received == NULL || received_count == NULL || owner == NULL ||
      MPI_Comm_size(communicator, &ranks) != MPI_SUCCESS || ranks <= 0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *received = NULL;
  *received_count = 0U;
  if (record_size == 0U || record_size > (size_t)INT_MAX ||
      record_count > SIZE_MAX / record_size ||
      (record_count != 0U && records == NULL) || ranks > INT_MAX / 2) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  if (status == SS_OK) {
    send.counts = calloc((size_t)ranks, sizeof(*send.counts));
    send.offsets = calloc((size_t)ranks, sizeof(*send.offsets));
    receive.counts = calloc((size_t)ranks, sizeof(*receive.counts));
    receive.offsets = calloc((size_t)ranks, sizeof(*receive.offsets));
    cursor = calloc((size_t)ranks, sizeof(*cursor));
    if (send.counts == NULL || send.offsets == NULL || receive.counts == NULL ||
        receive.offsets == NULL || cursor == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  for (size_t i = 0U; status == SS_OK && i < record_count; ++i) {
    const int target = owner(bytes + i * record_size, ranks, context);
    if (target < 0 || target >= ranks) {
      status = SS_ERROR_INVALID_ARGUMENT;
      break;
    }
    ++send.counts[target];
  }
  if (ranks == 1 && status == SS_OK) {
    receive.counts[0] = record_count;
    status = ss_mesh_route_allocate(&receive, ranks, record_size);
    if (status == SS_OK && record_count != 0U) {
      memcpy(receive.data, records, record_count * record_size);
    }
    goto finish;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK && MPI_Alltoall(send.counts, 1, MPI_UINT64_T, receive.counts, 1,
                                      MPI_UINT64_T, communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    status = ss_mesh_route_allocate(&send, ranks, record_size);
  }
  if (status == SS_OK) {
    status = ss_mesh_route_allocate(&receive, ranks, record_size);
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    memcpy(cursor, send.offsets, (size_t)ranks * sizeof(*cursor));
    for (size_t i = 0U; i < record_count; ++i) {
      const unsigned char *source = bytes + i * record_size;
      const int target = owner(source, ranks, context);
      memcpy(send.data + cursor[target]++ * record_size, source, record_size);
    }
    status = ss_mesh_route_exchange(communicator, ranks, &send, &receive, record_size);
  }
finish:
  if (status == SS_OK) {
    *received = receive.data;
    *received_count = receive.total;
  } else {
    free(receive.data);
  }
  free(send.data);
  free(send.counts);
  free(send.offsets);
  free(receive.counts);
  free(receive.offsets);
  free(cursor);
  return status;
}

typedef struct {
  SsEntityId id;
  size_t query;
  int rank;
  uint32_t value;
} SsMeshJoinRecord;

static int ss_mesh_join_shard(const void *raw, int ranks, const void *context) {
  const SsMeshJoinRecord *record = raw;
  (void)context;
  return (int)(ss_mesh_distribution_hash(record->id) % (uint64_t)ranks);
}

static int ss_mesh_join_return(const void *raw, int ranks, const void *context) {
  const SsMeshJoinRecord *record = raw;
  (void)ranks;
  (void)context;
  return record->rank;
}

static int ss_mesh_join_compare(const void *left_raw, const void *right_raw) {
  const SsMeshJoinRecord *left = left_raw;
  const SsMeshJoinRecord *right = right_raw;
  return (left->id > right->id) - (left->id < right->id);
}

SsStatus ss_mesh_join_ids(MPI_Comm communicator, const SsMeshIdValue *sources,
                          size_t source_count, const SsEntityId *queries,
                          size_t query_count, uint32_t *values) {
  SsMeshJoinRecord *records = NULL;
  SsMeshJoinRecord *shard = NULL;
  SsMeshJoinRecord *matches = NULL;
  size_t shard_count = 0U;
  size_t match_count = 0U;
  size_t reply_count = 0U;
  int rank = 0;
  SsStatus status = SS_OK;
  if ((source_count != 0U && sources == NULL) ||
      (query_count != 0U && (queries == NULL || values == NULL)) ||
      source_count > SIZE_MAX - query_count ||
      MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else {
    const size_t count = source_count + query_count;
    records = calloc(count == 0U ? 1U : count, sizeof(*records));
    if (records == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  for (size_t i = 0U; status == SS_OK && i < source_count; ++i) {
    if (sources[i].value == 0U) {
      status = SS_ERROR_INVALID_ARGUMENT;
    }
    records[i] = (SsMeshJoinRecord){sources[i].id, 0U, -1, sources[i].value};
  }
  for (size_t i = 0U; status == SS_OK && i < query_count; ++i) {
    records[source_count + i] = (SsMeshJoinRecord){queries[i], i, rank, 0U};
    values[i] = 0U;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    status = ss_mesh_route_records(communicator, records, source_count + query_count,
                                   sizeof(*records), ss_mesh_join_shard, NULL,
                                   (void **)&shard, &shard_count);
  }
  free(records);
  if (status != SS_OK) {
    return status;
  }
  ss_sort(shard, shard_count, sizeof(*shard), ss_mesh_join_compare);
  for (size_t begin = 0U; status == SS_OK && begin < shard_count;) {
    uint32_t value = 0U;
    int has_query = 0;
    size_t end = begin;
    while (end < shard_count && shard[end].id == shard[begin].id) {
      if (shard[end].rank >= 0) {
        has_query = 1;
      } else if (value != 0U && value != shard[end].value) {
        status = SS_ERROR_INTEGRITY;
      } else {
        value = shard[end].value;
      }
      ++end;
    }
    if (value != 0U) {
      if (has_query == 0) {
        status = SS_ERROR_NOT_FOUND;
      }
      for (size_t i = begin; i < end; ++i) {
        if (shard[i].rank >= 0) {
          shard[i].value = value;
          shard[reply_count++] = shard[i];
        }
      }
    }
    begin = end;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    status = ss_mesh_route_records(communicator, shard, reply_count, sizeof(*shard),
                                   ss_mesh_join_return, NULL, (void **)&matches,
                                   &match_count);
  }
  free(shard);
  for (size_t i = 0U; status == SS_OK && i < match_count; ++i) {
    if (matches[i].query >= query_count) {
      status = SS_ERROR_INTEGRITY;
    } else {
      values[matches[i].query] = matches[i].value;
    }
  }
  free(matches);
  return ss_status_allreduce_max(communicator, status);
}
