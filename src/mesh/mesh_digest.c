/* Rank-independent logical mesh and topology digests.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "mesh_internal.h"
#include "slope_internal/core/collective.h"

#include <petscdmplex.h>
#include <petsclog.h>

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint64_t ss_mesh_hash_update(uint64_t hash, const void *data, size_t size) {
  const unsigned char *bytes = (const unsigned char *)data;
  size_t index;
  for (index = 0U; index < size; ++index) {
    hash ^= (uint64_t)bytes[index];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

uint64_t ss_mesh_hash_u64(uint64_t hash, uint64_t value) {
  unsigned char bytes[sizeof(value)];
  size_t index;
  for (index = 0U; index < sizeof(value); ++index) {
    bytes[index] = (unsigned char)((value >> (index * 8U)) & UINT64_C(0xff));
  }
  return ss_mesh_hash_update(hash, bytes, sizeof(bytes));
}

void ss_mesh_store_u64(unsigned char output[8], uint64_t value) {
  size_t index;
  for (index = 0U; index < 8U; ++index) {
    output[index] = (unsigned char)((value >> (index * 8U)) & UINT64_C(0xff));
  }
}

void ss_mesh_store_u32(unsigned char output[4], uint32_t value) {
  size_t index;
  for (index = 0U; index < 4U; ++index) {
    output[index] = (unsigned char)((value >> (index * 8U)) & UINT32_C(0xff));
  }
}

void ss_mesh_digest_add(SsMeshDigestAccumulator *accumulator, uint64_t first,
                        uint64_t second) {
  ++accumulator->count;
  accumulator->sum_first += first;
  accumulator->sum_second += second;
  accumulator->xor_first ^= first;
  accumulator->xor_second ^= second;
}

void ss_mesh_digest_record_begin(uint64_t tag, uint64_t *first, uint64_t *second) {
  *first = ss_mesh_hash_u64(UINT64_C(1469598103934665603), tag);
  *second = ss_mesh_hash_u64(UINT64_C(7809847782465536322), tag);
}

void ss_mesh_digest_record_u64(uint64_t value, uint64_t *first, uint64_t *second) {
  *first = ss_mesh_hash_u64(*first, value);
  *second = ss_mesh_hash_u64(*second, value);
}

void ss_mesh_digest_record_u32(uint32_t value, uint64_t *first, uint64_t *second) {
  unsigned char bytes[4];
  ss_mesh_store_u32(bytes, value);
  *first = ss_mesh_hash_update(*first, bytes, sizeof(bytes));
  *second = ss_mesh_hash_update(*second, bytes, sizeof(bytes));
}

void ss_mesh_digest_record_f64(double value, uint64_t *first, uint64_t *second) {
  uint64_t bits = 0U;
  memcpy(&bits, &value, sizeof(bits));
  ss_mesh_digest_record_u64(bits, first, second);
}

SsStatus ss_mesh_digest_finalize(MPI_Comm communicator,
                                 const SsMeshDigestAccumulator *local,
                                 char output[SS_SHA256_HEX_CAPACITY]) {
  uint64_t local_sum[3] = {
      local->count,
      local->sum_first,
      local->sum_second,
  };
  uint64_t local_xor[2] = {
      local->xor_first,
      local->xor_second,
  };
  uint64_t global_sum[3] = {0U, 0U, 0U};
  uint64_t global_xor[2] = {0U, 0U};
  unsigned char canonical[5U * sizeof(uint64_t)];
  size_t field;
  if (MPI_Allreduce(local_sum, global_sum, 3, MPI_UINT64_T, MPI_SUM, communicator) !=
          MPI_SUCCESS ||
      MPI_Allreduce(local_xor, global_xor, 2, MPI_UINT64_T, MPI_BXOR, communicator) !=
          MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  for (field = 0U; field < 3U; ++field) {
    ss_mesh_store_u64(canonical + field * sizeof(uint64_t), global_sum[field]);
  }
  for (field = 0U; field < 2U; ++field) {
    ss_mesh_store_u64(canonical + (field + 3U) * sizeof(uint64_t), global_xor[field]);
  }
  return ss_sha256_bytes(canonical, sizeof(canonical), output);
}

void ss_mesh_digest_id(SsMeshDigestAccumulator *accumulator, uint64_t tag,
                       SsEntityId id) {
  uint64_t first;
  uint64_t second;
  ss_mesh_digest_record_begin(tag, &first, &second);
  ss_mesh_digest_record_u64(id, &first, &second);
  ss_mesh_digest_add(accumulator, first, second);
}

SsStatus ss_mesh_compute_digests(MPI_Comm communicator, const SsMeshInput *input,
                                 const SsFeDescriptor *descriptor,
                                 SsMeshSummary *summary) {
  SsMeshDigestAccumulator stable = {0U, 0U, 0U, 0U, 0U};
  SsMeshDigestAccumulator labels = {0U, 0U, 0U, 0U, 0U};
  SsMeshDigestAccumulator coordinates = {0U, 0U, 0U, 0U, 0U};
  SsMeshDigestAccumulator closures = {0U, 0U, 0U, 0U, 0U};
  size_t index;
  for (index = 0U; index < input->vertex_count; ++index) {
    ss_mesh_digest_id(&stable, UINT64_C(1), input->vertex_ids[index]);
  }
  for (index = 0U; index < input->cell_count; ++index) {
    uint64_t first;
    uint64_t second;
    ss_mesh_digest_id(&stable, UINT64_C(2), input->cell_ids[index]);
    ss_mesh_digest_record_begin(UINT64_C(31), &first, &second);
    ss_mesh_digest_record_u64(input->cell_ids[index], &first, &second);
    {
      size_t value;
      for (value = 0U; value <= (size_t)input->dimension; ++value) {
        ss_mesh_digest_record_u64(
            input->cells[index * ((size_t)input->dimension + 1U) + value], &first,
            &second);
      }
      for (value = 0U; value < descriptor->node_count; ++value) {
        ss_mesh_digest_record_u64(
            input->cell_coordinate_nodes[index * descriptor->node_count + value],
            &first, &second);
      }
    }
    ss_mesh_digest_add(&closures, first, second);
  }
  for (index = 0U; index < input->boundary_facet_count; ++index) {
    uint64_t first;
    uint64_t second;
    ss_mesh_digest_id(&stable, UINT64_C(3), input->boundary_facet_ids[index]);
    ss_mesh_digest_record_begin(UINT64_C(32), &first, &second);
    ss_mesh_digest_record_u64(input->boundary_facet_ids[index], &first, &second);
    {
      size_t value;
      for (value = 0U; value < (size_t)input->dimension; ++value) {
        ss_mesh_digest_record_u64(
            input->boundary_facets[index * (size_t)input->dimension + value], &first,
            &second);
      }
    }
    ss_mesh_digest_add(&closures, first, second);
  }
  for (index = 0U; index < input->coordinate_node_count; ++index) {
    uint64_t first;
    uint64_t second;
    ss_mesh_digest_id(&stable, UINT64_C(4), input->coordinate_node_ids[index]);
    ss_mesh_digest_record_begin(UINT64_C(21), &first, &second);
    ss_mesh_digest_record_u64(input->coordinate_node_ids[index], &first, &second);
    {
      size_t coordinate;
      for (coordinate = 0U; coordinate < (size_t)input->dimension; ++coordinate) {
        ss_mesh_digest_record_f64(
            input
                ->coordinate_coordinates[index * (size_t)input->dimension + coordinate],
            &first, &second);
      }
    }
    ss_mesh_digest_add(&coordinates, first, second);
  }
  for (index = 0U; index < input->label_count; ++index) {
    uint64_t first;
    uint64_t second;
    ss_mesh_digest_record_begin(UINT64_C(11), &first, &second);
    ss_mesh_digest_record_u32((uint32_t)input->labels[index].kind, &first, &second);
    ss_mesh_digest_record_u64(input->labels[index].stable_entity_id, &first, &second);
    ss_mesh_digest_record_u32(input->labels[index].value, &first, &second);
    ss_mesh_digest_add(&labels, first, second);
  }
  if (ss_mesh_digest_finalize(communicator, &stable, summary->stable_id_digest) !=
          SS_OK ||
      ss_mesh_digest_finalize(communicator, &labels, summary->label_digest) != SS_OK ||
      ss_mesh_digest_finalize(communicator, &coordinates, summary->coordinate_digest) !=
          SS_OK ||
      ss_mesh_digest_finalize(communicator, &closures, summary->closure_digest) !=
          SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  return SS_OK;
}

SsStatus ss_mesh_compute_topology_digest(DM dm, const SsMesh *mesh,
                                         MPI_Comm communicator,
                                         char output[SS_SHA256_HEX_CAPACITY]) {
  SsMeshDigestAccumulator topology = {0U, 0U, 0U, 0U, 0U};
  PetscInt chart_start = 0;
  PetscInt chart_end = 0;
  PetscInt point;
  unsigned char *ghost_flags = NULL;
  SsChunkedLabels stable_labels = {0};
  SsStatus status = SS_OK;
  if (dm == NULL || output == NULL ||
      DMPlexGetChart(dm, &chart_start, &chart_end) != PETSC_SUCCESS) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  if (status == SS_OK && mesh == NULL) {
    const size_t chart_count = (size_t)(chart_end - chart_start);
    ghost_flags = (unsigned char *)calloc(chart_count == 0U ? 1U : chart_count,
                                          sizeof(*ghost_flags));
    if (ghost_flags == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    } else {
      status = ss_mesh_ghost_flags_fill(dm, chart_start, chart_end, ghost_flags);
      if (status == SS_OK) {
        status =
            ss_mesh_chunked_labels_resolve(dm, ss_mesh_id_chunk_labels, &stable_labels);
      }
    }
  }
  for (point = chart_start; status == SS_OK && point < chart_end; ++point) {
    PetscInt depth = -1;
    SsEntityId stable_id = 0U;
    SsEntityId vertex_ids[4] = {0U, 0U, 0U, 0U};
    size_t vertex_count = 0U;
    uint64_t first;
    uint64_t second;
    size_t vertex;
    if ((mesh == NULL ? ghost_flags[(size_t)(point - chart_start)] != 0U
                      : ss_mesh_point_is_ghost(mesh, dm, point) != 0)) {
      continue;
    }
    if (DMPlexGetPointDepth(dm, point, &depth) != PETSC_SUCCESS || depth < 0 ||
        depth > (PetscInt)SS_FE_MAX_DIMENSION ||
        (mesh == NULL ? ss_mesh_chunked_labels_get(&stable_labels, point, &stable_id)
                      : ss_mesh_cached_stable_id(mesh, point, &stable_id)) != SS_OK) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    if (depth != 0 &&
        (ss_mesh_point_vertex_ids(dm, mesh, mesh == NULL ? &stable_labels : NULL, point,
                                  vertex_ids, &vertex_count) != SS_OK ||
         vertex_count != (size_t)depth + 1U)) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    ss_mesh_digest_record_begin(UINT64_C(41), &first, &second);
    ss_mesh_digest_record_u32((uint32_t)depth, &first, &second);
    ss_mesh_digest_record_u64(stable_id, &first, &second);
    ss_mesh_digest_record_u32((uint32_t)vertex_count, &first, &second);
    for (vertex = 0U; vertex < vertex_count; ++vertex) {
      ss_mesh_digest_record_u64(vertex_ids[vertex], &first, &second);
    }
    ss_mesh_digest_add(&topology, first, second);
  }
  free(ghost_flags);
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    return status;
  }
  return ss_mesh_digest_finalize(communicator, &topology, output);
}
