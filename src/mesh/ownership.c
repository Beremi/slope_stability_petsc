/* Deterministic shared-point ownership over sparse point-SF candidates.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "ownership.h"
#include "slope_internal/core/sort.h"
#include "slope_internal/core/util.h"
#include "slope_internal/mesh/distribution.h"

#include <petscdmplex.h>
#include <petscsf.h>

#include <stdint.h>
#include <stdlib.h>

#define SS_STABLE_ID_LABEL_COUNT 8U

static const char *const ss_stable_id_labels[SS_STABLE_ID_LABEL_COUNT] = {
    "ss-stable-id-byte0", "ss-stable-id-byte1", "ss-stable-id-byte2",
    "ss-stable-id-byte3", "ss-stable-id-byte4", "ss-stable-id-byte5",
    "ss-stable-id-byte6", "ss-stable-id-byte7",
};

typedef struct {
  PetscInt point;
  PetscInt candidate_count;
  PetscInt weight;
  uint64_t key;
} SsSharedPointRecord;

static PetscErrorCode ss_stable_point_key(DM dm, PetscInt point, uint64_t *key) {
  MPI_Comm communicator;
  uint64_t stable_id = 0U;
  size_t chunk;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &communicator));
  for (chunk = 0U; chunk < SS_STABLE_ID_LABEL_COUNT; ++chunk) {
    PetscInt value = -1;
    PetscCall(DMGetLabelValue(dm, ss_stable_id_labels[chunk], point, &value));
    PetscCheck(value >= 0 && value <= 255, communicator, PETSC_ERR_PLIB,
               "missing stable ID on shared point");
    stable_id |= (uint64_t)(unsigned int)value << (8U * (unsigned int)chunk);
  }
  *key = ss_mesh_distribution_hash(stable_id);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static int ss_record_compare(const void *left_value, const void *right_value) {
  const SsSharedPointRecord *left = (const SsSharedPointRecord *)left_value;
  const SsSharedPointRecord *right = (const SsSharedPointRecord *)right_value;
  if (left->candidate_count < right->candidate_count) {
    return -1;
  }
  if (left->candidate_count > right->candidate_count) {
    return 1;
  }
  if (left->weight > right->weight) {
    return -1;
  }
  if (left->weight < right->weight) {
    return 1;
  }
  if (left->key < right->key) {
    return -1;
  }
  if (left->key > right->key) {
    return 1;
  }
  return 0;
}

typedef struct {
  PetscSF sf;
  PetscInt points;
  PetscInt leaves;
  const PetscInt *local;
  const PetscSFNode *remote;
} SsPointSfLayout;

static PetscErrorCode ss_point_sf_layout(DM dm, SsPointSfLayout *layout) {
  MPI_Comm communicator = PetscObjectComm((PetscObject)dm);
  PetscInt begin, roots;
  PetscFunctionBegin;
  PetscCall(DMPlexGetChart(dm, &begin, &layout->points));
  PetscCheck(begin == 0, communicator, PETSC_ERR_SUP,
             "stable ownership requires a zero-based DMPlex chart");
  PetscCall(DMGetPointSF(dm, &layout->sf));
  PetscCall(PetscSFGetGraph(layout->sf, &roots, &layout->leaves, &layout->local,
                            &layout->remote));
  PetscCheck(roots == layout->points, communicator, PETSC_ERR_SUP,
             "unexpected DMPlex point-SF root layout");
  for (PetscInt i = 0; i < layout->leaves; ++i) {
    const PetscInt point = layout->local == NULL ? i : layout->local[i];
    PetscCheck(point >= 0 && point < layout->points, communicator, PETSC_ERR_PLIB,
               "point-SF leaf lies outside the DMPlex chart");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ss_ownership_agree(MPI_Comm communicator, PetscErrorCode error) {
  int local = (int)error, global = (int)PETSC_SUCCESS;
  return MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, communicator) ==
                 MPI_SUCCESS
             ? (PetscErrorCode)global
             : PETSC_ERR_MPI;
}

static PetscErrorCode ss_point_sf_offsets(MPI_Comm communicator, PetscInt points,
                                          const PetscInt *degrees, PetscInt *offsets) {
  PetscFunctionBegin;
  offsets[0] = 0;
  for (PetscInt point = 0; point < points; ++point) {
    PetscCheck(degrees[point] >= 0 && offsets[point] <= PETSC_MAX_INT - degrees[point],
               communicator, PETSC_ERR_SUP, "point-SF degree storage exceeds PetscInt");
    offsets[point + 1] = offsets[point] + degrees[point];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ss_rewrite_point_sf(DM dm, const SsPointSfLayout *layout,
                                          PetscMPIInt rank, PetscInt count,
                                          const PetscInt *points_to_rewrite,
                                          const PetscInt *target_owners,
                                          const PetscInt *degree_offsets,
                                          const PetscInt *leaf_ranks) {
  const MPI_Comm communicator = PetscObjectComm((PetscObject)dm);
  const PetscSF sf = layout->sf;
  const PetscInt points = layout->points;
  PetscInt *leaf_points = NULL, *root_points = NULL;
  PetscInt *root_owners = NULL, *root_owner_points = NULL;
  PetscInt *leaf_owners = NULL, *leaf_owner_points = NULL;
  PetscInt *new_local_leaves = NULL;
  PetscSFNode *new_remote_leaves = NULL;
  PetscInt *old_leaf_slots = NULL;
  PetscInt new_leaf_count = 0;
  PetscErrorCode error = PETSC_SUCCESS;
  PetscFunctionBegin;
  if (PetscMalloc1((size_t)degree_offsets[points], &leaf_points) != PETSC_SUCCESS ||
      PetscMalloc1((size_t)points, &root_points) != PETSC_SUCCESS ||
      PetscMalloc1((size_t)points, &root_owners) != PETSC_SUCCESS ||
      PetscMalloc1((size_t)points, &root_owner_points) != PETSC_SUCCESS ||
      PetscMalloc1((size_t)points, &leaf_owners) != PETSC_SUCCESS ||
      PetscMalloc1((size_t)points, &leaf_owner_points) != PETSC_SUCCESS ||
      PetscMalloc1((size_t)points, &old_leaf_slots) != PETSC_SUCCESS) {
    error = PETSC_ERR_MEM;
  }
  error = ss_ownership_agree(communicator, error);
  if (error != PETSC_SUCCESS) {
    goto cleanup;
  }
  for (PetscInt point = 0; point < points; ++point) {
    root_points[point] = point;
    root_owners[point] = root_owner_points[point] = -1;
    leaf_owners[point] = leaf_owner_points[point] = -1;
    old_leaf_slots[point] = -1;
  }
  for (PetscInt i = 0; i < layout->leaves; ++i) {
    const PetscInt point = layout->local == NULL ? i : layout->local[i];
    old_leaf_slots[point] = i;
  }
  error = PetscSFGatherBegin(sf, MPIU_INT, root_points, leaf_points);
  if (error == PETSC_SUCCESS) {
    error = PetscSFGatherEnd(sf, MPIU_INT, root_points, leaf_points);
  }
  for (PetscInt i = 0; error == PETSC_SUCCESS && i < count; ++i) {
    const PetscInt old_point = points_to_rewrite[i];
    const PetscInt new_owner = target_owners[i];
    PetscInt new_point = old_point;
    if (new_owner != (PetscInt)rank) {
      new_point = -1;
      for (PetscInt leaf = degree_offsets[old_point];
           leaf < degree_offsets[old_point + 1]; ++leaf) {
        if (leaf_ranks[leaf] == new_owner) {
          new_point = leaf_points[leaf];
          break;
        }
      }
    }
    if (new_point < 0) {
      error = PETSC_ERR_PLIB;
    } else {
      root_owners[old_point] = new_owner;
      root_owner_points[old_point] = new_point;
    }
  }
  error = ss_ownership_agree(communicator, error);
  if (error != PETSC_SUCCESS) {
    goto cleanup;
  }
  error = PetscSFBcastBegin(sf, MPIU_INT, root_owners, leaf_owners, MPI_REPLACE);
  if (error == PETSC_SUCCESS) {
    error = PetscSFBcastEnd(sf, MPIU_INT, root_owners, leaf_owners, MPI_REPLACE);
  }
  error = ss_ownership_agree(communicator, error);
  if (error == PETSC_SUCCESS) {
    error = PetscSFBcastBegin(sf, MPIU_INT, root_owner_points, leaf_owner_points,
                              MPI_REPLACE);
    if (error == PETSC_SUCCESS) {
      error = PetscSFBcastEnd(sf, MPIU_INT, root_owner_points, leaf_owner_points,
                              MPI_REPLACE);
    }
  }
  for (PetscInt point = 0; error == PETSC_SUCCESS && point < points; ++point) {
    const PetscInt owner =
        root_owners[point] >= 0 ? root_owners[point] : leaf_owners[point];
    new_leaf_count += owner >= 0 ? owner != (PetscInt)rank : old_leaf_slots[point] >= 0;
  }
  if (error == PETSC_SUCCESS &&
      (PetscMalloc1((size_t)new_leaf_count, &new_local_leaves) != PETSC_SUCCESS ||
       PetscMalloc1((size_t)new_leaf_count, &new_remote_leaves) != PETSC_SUCCESS)) {
    error = PETSC_ERR_MEM;
  }
  error = ss_ownership_agree(communicator, error);
  if (error != PETSC_SUCCESS) {
    goto cleanup;
  }
  new_leaf_count = 0;
  for (PetscInt point = 0; point < points; ++point) {
    const PetscInt owner =
        root_owners[point] >= 0 ? root_owners[point] : leaf_owners[point];
    const PetscInt owner_point =
        root_owners[point] >= 0 ? root_owner_points[point] : leaf_owner_points[point];
    const PetscInt old_slot = old_leaf_slots[point];
    if (owner >= 0 && owner != (PetscInt)rank) {
      new_local_leaves[new_leaf_count] = point;
      new_remote_leaves[new_leaf_count++] =
          (PetscSFNode){(PetscMPIInt)owner, owner_point};
    } else if (owner < 0 && old_slot >= 0) {
      new_local_leaves[new_leaf_count] = point;
      new_remote_leaves[new_leaf_count++] = layout->remote[old_slot];
    }
  }
  error = PetscSFSetGraph(sf, points, new_leaf_count, new_local_leaves,
                          PETSC_OWN_POINTER, new_remote_leaves, PETSC_OWN_POINTER);
  if (error == PETSC_SUCCESS) {
    new_local_leaves = NULL;
    new_remote_leaves = NULL;
  }

cleanup:
  (void)PetscFree(leaf_points);
  (void)PetscFree(root_points);
  (void)PetscFree(root_owners);
  (void)PetscFree(root_owner_points);
  (void)PetscFree(leaf_owners);
  (void)PetscFree(leaf_owner_points);
  (void)PetscFree(old_leaf_slots);
  (void)PetscFree(new_local_leaves);
  (void)PetscFree(new_remote_leaves);
  PetscFunctionReturn(ss_ownership_agree(communicator, error));
}

static PetscErrorCode ss_shared_point_weights(DM dm, PetscInt shared_depth_end,
                                              PetscInt *weights, PetscInt *capacity) {
  MPI_Comm communicator = PetscObjectComm((PetscObject)dm);
  PetscInt cell_begin = 0, cell_end = 0;
  PetscFunctionBegin;
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cell_begin, &cell_end));
  for (PetscInt depth = 0; depth < shared_depth_end; ++depth) {
    PetscInt begin = 0, end = 0;
    PetscCall(DMPlexGetDepthStratum(dm, depth, &begin, &end));
    PetscCheck(end >= begin && *capacity <= PETSC_MAX_INT - (end - begin), communicator,
               PETSC_ERR_SUP, "shared-point ownership capacity exceeds PetscInt");
    *capacity += end - begin;
    for (PetscInt point = begin; point < end; ++point) {
      PetscInt closure_size = 0, *closure = NULL;
      PetscCall(
          DMPlexGetTransitiveClosure(dm, point, PETSC_FALSE, &closure_size, &closure));
      for (PetscInt item = 0; item < closure_size; ++item) {
        const PetscInt cell = closure[2 * item];
        weights[point] += cell >= cell_begin && cell < cell_end;
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, point, PETSC_FALSE, &closure_size,
                                               &closure));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ss_shared_point_records(
    DM dm, PetscInt shared_depth_end, PetscMPIInt rank, PetscMPIInt size,
    const PetscInt *degrees, const PetscInt *offsets, const PetscInt *leaf_ranks,
    const PetscInt *leaf_weights, const PetscInt *local_weights,
    const PetscBool *is_leaf, PetscInt *candidates, SsSharedPointRecord *records,
    PetscInt *count, uint64_t *exclusive_count) {
  MPI_Comm communicator = PetscObjectComm((PetscObject)dm);
  PetscFunctionBegin;
  for (PetscInt depth = 0; depth < shared_depth_end; ++depth) {
    PetscInt begin = 0, end = 0;
    PetscCall(DMPlexGetDepthStratum(dm, depth, &begin, &end));
    for (PetscInt point = begin; point < end; ++point) {
      if (degrees[point] == 0) {
        if (is_leaf[point] == PETSC_FALSE) {
          *exclusive_count += (uint64_t)local_weights[point];
        }
        continue;
      }
      SsSharedPointRecord *record = &records[(*count)++];
      const PetscInt first = offsets[point], last = offsets[point + 1];
      record->point = point;
      record->weight = local_weights[point];
      for (PetscInt leaf = first; leaf < last; ++leaf) {
        const PetscInt candidate = leaf_ranks[leaf];
        PetscCheck(
            candidate >= 0 && candidate < (PetscInt)size && candidate != (PetscInt)rank,
            communicator, PETSC_ERR_PLIB, "invalid shared-point owner candidate");
        PetscCheck(record->weight <= PETSC_MAX_INT - leaf_weights[leaf], communicator,
                   PETSC_ERR_SUP, "shared point weight exceeds PetscInt");
        record->weight += leaf_weights[leaf];
        candidates[leaf] = candidate;
      }
      ss_sort(candidates + first, (size_t)(last - first), sizeof(*candidates),
              ss_petsc_int_compare);
      record->candidate_count = 1;
      PetscInt preceding = -1;
      for (PetscInt leaf = first; leaf < last; ++leaf) {
        const PetscInt candidate = candidates[leaf];
        if (candidate != preceding) {
          candidates[first + record->candidate_count++ - 1] = candidate;
          preceding = candidate;
        }
      }
      PetscCall(ss_stable_point_key(dm, point, &record->key));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ss_rebalance_shared_points(DM dm, PetscBool include_facets) {
  const MPI_Comm communicator = PetscObjectComm((PetscObject)dm);
  MPI_Comm decision_comm = MPI_COMM_NULL;
  PetscMPIInt rank, size, tag = 0;
  SsPointSfLayout layout = {0};
  PetscInt dimension = 0, capacity = 0, count = 0;
  const PetscInt *degrees = NULL;
  PetscInt *degree_offsets = NULL, *candidates = NULL;
  PetscInt *leaf_ranks = NULL, *root_ranks = NULL;
  PetscInt *local_weights = NULL, *leaf_weights = NULL;
  PetscBool *is_leaf = NULL;
  PetscInt *points = NULL, *owners = NULL;
  uint64_t *loads = NULL;
  uint64_t exclusive_count = 0U;
  SsSharedPointRecord *records = NULL;
  PetscErrorCode error;
  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(communicator, &rank));
  PetscCallMPI(MPI_Comm_size(communicator, &size));
  error = DMGetDimension(dm, &dimension);
  if (error == PETSC_SUCCESS) {
    error = dimension == 2 || dimension == 3 ? ss_point_sf_layout(dm, &layout)
                                             : PETSC_ERR_SUP;
  }
  error = ss_ownership_agree(communicator, error);
  if (error != PETSC_SUCCESS) {
    goto cleanup;
  }
  /* P4 adds facet-interior unknowns in 3D. Both policies preserve cell ownership. */
  const PetscInt shared_depth_end =
      include_facets != PETSC_FALSE ? dimension : PetscMin(2, dimension);
  error = PetscSFComputeDegreeBegin(layout.sf, &degrees);
  if (error == PETSC_SUCCESS) {
    error = PetscSFComputeDegreeEnd(layout.sf, &degrees);
  }
  if (error == PETSC_SUCCESS &&
      (PetscCalloc1((size_t)layout.points, &is_leaf) != PETSC_SUCCESS ||
       PetscMalloc1((size_t)layout.points + 1U, &degree_offsets) != PETSC_SUCCESS ||
       PetscMalloc1((size_t)layout.points, &root_ranks) != PETSC_SUCCESS ||
       PetscCalloc1((size_t)layout.points, &local_weights) != PETSC_SUCCESS)) {
    error = PETSC_ERR_MEM;
  }
  if (error == PETSC_SUCCESS) {
    error = ss_point_sf_offsets(communicator, layout.points, degrees, degree_offsets);
  }
  if (error == PETSC_SUCCESS) {
    const size_t leaves = (size_t)degree_offsets[layout.points];
    if (PetscMalloc1(leaves, &leaf_ranks) != PETSC_SUCCESS ||
        PetscMalloc1(leaves, &leaf_weights) != PETSC_SUCCESS ||
        PetscMalloc1(leaves, &candidates) != PETSC_SUCCESS) {
      error = PETSC_ERR_MEM;
    }
  }
  if (error == PETSC_SUCCESS) {
    for (PetscInt i = 0; i < layout.leaves; ++i) {
      is_leaf[layout.local == NULL ? i : layout.local[i]] = PETSC_TRUE;
    }
    for (PetscInt point = 0; point < layout.points; ++point) {
      root_ranks[point] = (PetscInt)rank;
    }
    error = ss_shared_point_weights(dm, shared_depth_end, local_weights, &capacity);
  }
  error = ss_ownership_agree(communicator, error);
  if (error != PETSC_SUCCESS) {
    goto cleanup;
  }
  error = PetscSFGatherBegin(layout.sf, MPIU_INT, root_ranks, leaf_ranks);
  if (error == PETSC_SUCCESS) {
    error = PetscSFGatherEnd(layout.sf, MPIU_INT, root_ranks, leaf_ranks);
  }
  error = ss_ownership_agree(communicator, error);
  if (error == PETSC_SUCCESS) {
    error = PetscSFGatherBegin(layout.sf, MPIU_INT, local_weights, leaf_weights);
    if (error == PETSC_SUCCESS) {
      error = PetscSFGatherEnd(layout.sf, MPIU_INT, local_weights, leaf_weights);
    }
  }
  if (error == PETSC_SUCCESS &&
      (PetscMalloc1((size_t)capacity, &points) != PETSC_SUCCESS ||
       PetscMalloc1((size_t)capacity, &owners) != PETSC_SUCCESS ||
       PetscMalloc1((size_t)capacity, &records) != PETSC_SUCCESS ||
       PetscMalloc1((size_t)size, &loads) != PETSC_SUCCESS)) {
    error = PETSC_ERR_MEM;
  }
  if (error == PETSC_SUCCESS) {
    error =
        ss_shared_point_records(dm, shared_depth_end, rank, size, degrees,
                                degree_offsets, leaf_ranks, leaf_weights, local_weights,
                                is_leaf, candidates, records, &count, &exclusive_count);
  }
  error = ss_ownership_agree(communicator, error);
  if (error != PETSC_SUCCESS) {
    goto cleanup;
  }

  /* Optional greedy setup with an O(P) critical path. Complete and agree
   * on every allocation and candidate check before the rank-ordered sweep.
   * The chain carries P loads per hop and keeps the points on their SF roots. */
  if (MPI_Allgather(&exclusive_count, 1, MPI_UINT64_T, loads, 1, MPI_UINT64_T,
                    communicator) != MPI_SUCCESS) {
    error = PETSC_ERR_MPI;
  }
  ss_sort(records, (size_t)count, sizeof(*records), ss_record_compare);
  error = ss_ownership_agree(communicator, error);
  if (error == PETSC_SUCCESS) {
    error = PetscCommDuplicate(communicator, &decision_comm, &tag);
  }
  error = ss_ownership_agree(communicator, error);
  if (error != PETSC_SUCCESS) {
    goto cleanup;
  }
  if (rank > 0 && MPI_Recv(loads, size, MPI_UINT64_T, rank - 1, tag, decision_comm,
                           MPI_STATUS_IGNORE) != MPI_SUCCESS) {
    error = PETSC_ERR_MPI;
  }
  for (PetscInt i = 0; i < count; ++i) {
    const SsSharedPointRecord *record = &records[i];
    const PetscInt *shared = candidates + degree_offsets[record->point];
    PetscInt ordinal = (PetscInt)(record->key % (uint64_t)record->candidate_count);
    PetscInt best = -1;
    for (PetscInt j = 0; j < record->candidate_count; ++j) {
      const PetscInt candidate = ordinal == 0 ? (PetscInt)rank : shared[ordinal - 1];
      if (best < 0 || loads[candidate] < loads[best]) {
        best = candidate;
      }
      if (++ordinal == record->candidate_count) {
        ordinal = 0;
      }
    }
    points[i] = record->point;
    owners[i] = best;
    loads[best] += (uint64_t)record->weight;
  }
  if (rank + 1 < size && MPI_Send(loads, size, MPI_UINT64_T, rank + 1, tag,
                                  decision_comm) != MPI_SUCCESS) {
    error = PETSC_ERR_MPI;
  }
  error = ss_ownership_agree(communicator, error);
  if (error == PETSC_SUCCESS) {
    error = ss_rewrite_point_sf(dm, &layout, rank, count, points, owners,
                                degree_offsets, leaf_ranks);
  }

cleanup:
  if (decision_comm != MPI_COMM_NULL) {
    const PetscErrorCode close_error = PetscCommDestroy(&decision_comm);
    if (error == PETSC_SUCCESS) {
      error = close_error;
    }
  }
  (void)PetscFree(degree_offsets);
  (void)PetscFree(candidates);
  (void)PetscFree(leaf_ranks);
  (void)PetscFree(root_ranks);
  (void)PetscFree(local_weights);
  (void)PetscFree(leaf_weights);
  (void)PetscFree(is_leaf);
  (void)PetscFree(points);
  (void)PetscFree(owners);
  (void)PetscFree(loads);
  (void)PetscFree(records);
  PetscFunctionReturn(ss_ownership_agree(communicator, error));
}

static SsStatus ss_shared_point_status(PetscErrorCode error) {
  return error == PETSC_SUCCESS   ? SS_OK
         : error == PETSC_ERR_MEM ? SS_ERROR_OUT_OF_MEMORY
                                  : SS_ERROR_INTERNAL;
}

SsStatus ss_mesh_rebalance_shared_p2_point_ownership(DM dm) {
  if (dm == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return ss_shared_point_status(ss_rebalance_shared_points(dm, PETSC_FALSE));
}

SsStatus ss_mesh_rebalance_shared_p4_point_ownership(DM dm) {
  if (dm == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return ss_shared_point_status(ss_rebalance_shared_points(dm, PETSC_TRUE));
}
