/* Known-cell same-mesh routing and geometric search for unrelated meshes.
 * The engine uses the same-mesh route. The different-mesh search is a
 * test-only reference implementation with world collectives per target point.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "coupling_internal.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/core/util.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
  SsStatus status;
  int source_rank;
  SsTransferSourceCandidate source;
} SsCouplingLocationDecision;

/* A shared mesh partition provides the owned source chart on the target rank.
 * Reuse the locator only as a local reference map in this case. */
static SsStatus ss_coupling_same_mesh_local(const SsCouplingTransferPlan *plan,
                                            SsCouplingPlanPoint *point) {
  SsTransferSourceCandidate candidates[SS_TRANSFER_MAX_CANDIDATES] = {0};
  size_t count = 0;
  const SsStatus status =
      plan->geometry.locate(plan->geometry.context, &point->target, candidates, &count);
  if (status != SS_OK || count > SS_TRANSFER_MAX_CANDIDATES) {
    return status != SS_OK ? status : SS_ERROR_LIMIT;
  }
  point->source_rank = -1;
  for (size_t index = 0; index < count; ++index) {
    const SsTransferSourceCandidate *candidate = &candidates[index];
    if (candidate->stable_source_cell_id == 0U ||
        !ss_is_finite(candidate->interior_margin)) {
      return SS_ERROR_INTEGRITY;
    }
    if (candidate->ambiguous) {
      return SS_ERROR_DUPLICATE;
    }
    if (!candidate->owned) {
      continue;
    }
    if (candidate->stable_source_cell_id != point->target.stable_target_cell_id) {
      return SS_ERROR_TYPE_MISMATCH;
    }
    if (point->source_rank >= 0) {
      return SS_ERROR_DUPLICATE;
    }
    point->source = *candidate;
    point->source_rank = plan->rank;
  }
  return point->source_rank < 0 ? SS_ERROR_NOT_FOUND : SS_OK;
}

static SsStatus ss_coupling_locate_same_mesh(SsCouplingTransferPlan *plan,
                                             SsCouplingSameMeshMap map) {
  SsStatus status = SS_OK;
  for (size_t index = 0; index < plan->local_point_count && status == SS_OK; ++index) {
    SsCouplingPlanPoint *point = &plan->points[index];
    if (map != NULL) {
      point->source.stable_source_cell_id = point->target.stable_target_cell_id;
      point->source_rank = -1;
      status = map(plan->geometry.context, &point->target, &point->source_rank,
                   point->source.reference);
    } else {
      status = ss_coupling_same_mesh_local(plan, point);
    }
    if (status == SS_OK &&
        (point->source_rank < 0 || point->source_rank >= plan->size)) {
      status = SS_ERROR_INTEGRITY;
    }
    for (uint32_t d = 0; d < plan->dimension && status == SS_OK; ++d) {
      if (!ss_is_finite(point->source.reference[d])) {
        status = SS_ERROR_INTEGRITY;
      }
    }
  }
  return status;
}

static SsStatus ss_coupling_locate_geometric(SsCouplingTransferPlan *plan) {
  const MPI_Comm communicator = plan->communicator;
  const int rank = plan->rank;
  const int size = plan->size;
  const size_t local_count = plan->local_point_count;
  uint64_t *rank_point_counts = NULL;
  uint64_t *candidate_counts = NULL;
  SsTransferSourceCandidate *gathered_candidates = NULL;
  size_t candidate_capacity = 0U;
  SsStatus local_status = SS_OK;
  SsStatus status = SS_OK;
  int owner_rank;
  if (ss_coupling_checked_product((size_t)size, SS_TRANSFER_MAX_CANDIDATES,
                                  &candidate_capacity) != SS_OK) {
    status = SS_ERROR_OVERFLOW;
    goto cleanup;
  }
  rank_point_counts = calloc((size_t)size, sizeof(*rank_point_counts));
  candidate_counts = calloc((size_t)size, sizeof(*candidate_counts));
  gathered_candidates = calloc(candidate_capacity, sizeof(*gathered_candidates));
  local_status = rank_point_counts == NULL || candidate_counts == NULL ||
                         gathered_candidates == NULL
                     ? SS_ERROR_OUT_OF_MEMORY
                     : SS_OK;
  status = ss_status_allreduce_max(communicator, local_status);
  if (status != SS_OK) {
    goto cleanup;
  }
  {
    const uint64_t local_count_u64 = (uint64_t)local_count;
    if (MPI_Allgather(&local_count_u64, 1, MPI_UINT64_T, rank_point_counts, 1,
                      MPI_UINT64_T, communicator) != MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
      goto cleanup;
    }
  }
  for (owner_rank = 0; owner_rank < size; ++owner_rank) {
    uint64_t owner_index;
    if (rank_point_counts[owner_rank] > (uint64_t)SIZE_MAX) {
      status = SS_ERROR_RESOURCE;
      goto cleanup;
    }
    for (owner_index = 0U; owner_index < rank_point_counts[owner_rank]; ++owner_index) {
      SsTransferTargetPoint target;
      SsTransferSourceCandidate local_candidates[SS_TRANSFER_MAX_CANDIDATES];
      SsCouplingLocationDecision decision;
      uint64_t local_candidate_count = 0U;
      size_t callback_count = 0U;
      memset(&target, 0, sizeof(target));
      memset(local_candidates, 0, sizeof(local_candidates));
      memset(&decision, 0, sizeof(decision));
      decision.status = SS_OK;
      decision.source_rank = -1;
      if (rank == owner_rank) {
        target = plan->points[(size_t)owner_index].target;
      }
      if (MPI_Bcast(&target, (int)sizeof(target), MPI_BYTE, owner_rank, communicator) !=
          MPI_SUCCESS) {
        status = SS_ERROR_INTERNAL;
        goto cleanup;
      }
      local_status = plan->geometry.locate(plan->geometry.context, &target,
                                           local_candidates, &callback_count);
      if (local_status == SS_OK && callback_count > SS_TRANSFER_MAX_CANDIDATES) {
        local_status = SS_ERROR_LIMIT;
      }
      local_candidate_count = (uint64_t)callback_count;
      status = ss_status_allreduce_max(communicator, local_status);
      if (status != SS_OK) {
        goto cleanup;
      }
      if (MPI_Gather(&local_candidate_count, 1, MPI_UINT64_T, candidate_counts, 1,
                     MPI_UINT64_T, owner_rank, communicator) != MPI_SUCCESS ||
          MPI_Gather(local_candidates, (int)sizeof(local_candidates), MPI_BYTE,
                     gathered_candidates, (int)sizeof(local_candidates), MPI_BYTE,
                     owner_rank, communicator) != MPI_SUCCESS) {
        status = SS_ERROR_INTERNAL;
        goto cleanup;
      }
      if (rank == owner_rank) {
        int source_rank;
        int explicit_ambiguous = 0;
        int selected = 0;
        for (source_rank = 0; source_rank < size; ++source_rank) {
          size_t candidate_index;
          if (candidate_counts[source_rank] > SS_TRANSFER_MAX_CANDIDATES) {
            decision.status = SS_ERROR_LIMIT;
            break;
          }
          for (candidate_index = 0U;
               candidate_index < (size_t)candidate_counts[source_rank];
               ++candidate_index) {
            const SsTransferSourceCandidate *candidate =
                &gathered_candidates[(size_t)source_rank * SS_TRANSFER_MAX_CANDIDATES +
                                     candidate_index];
            uint32_t coordinate;
            int valid = candidate->stable_source_cell_id != 0U &&
                        ss_is_finite(candidate->interior_margin);
            for (coordinate = 0U; coordinate < plan->dimension; ++coordinate) {
              valid = valid && ss_is_finite(candidate->reference[coordinate]);
            }
            if (!valid) {
              decision.status = SS_ERROR_INTEGRITY;
              break;
            }
            if (candidate->ambiguous != 0) {
              explicit_ambiguous = 1;
            }
            if (candidate->owned == 0) {
              continue;
            }
            if (selected != 0 &&
                candidate->stable_source_cell_id ==
                    decision.source.stable_source_cell_id &&
                source_rank != decision.source_rank) {
              decision.status = SS_ERROR_INTEGRITY;
              break;
            }
            if (selected == 0 ||
                candidate->stable_source_cell_id <
                    decision.source.stable_source_cell_id ||
                (candidate->stable_source_cell_id ==
                     decision.source.stable_source_cell_id &&
                 source_rank < decision.source_rank)) {
              decision.source = *candidate;
              decision.source_rank = source_rank;
              selected = 1;
            }
          }
          if (decision.status != SS_OK) {
            break;
          }
        }
        if (decision.status == SS_OK && explicit_ambiguous != 0) {
          decision.status = SS_ERROR_DUPLICATE;
        } else if (decision.status == SS_OK && selected == 0) {
          decision.status = SS_ERROR_NOT_FOUND;
        }
      }
      if (MPI_Bcast(&decision, (int)sizeof(decision), MPI_BYTE, owner_rank,
                    communicator) != MPI_SUCCESS) {
        status = SS_ERROR_INTERNAL;
        goto cleanup;
      }
      if (decision.status != SS_OK) {
        status = decision.status;
        goto cleanup;
      }
      if (rank == owner_rank) {
        plan->points[(size_t)owner_index].source = decision.source;
        plan->points[(size_t)owner_index].source_rank = decision.source_rank;
      }
    }
  }
cleanup:
  free(rank_point_counts);
  free(candidate_counts);
  free(gathered_candidates);
  return status;
}

SsStatus ss_coupling_locate_points(SsCouplingTransferPlan *plan,
                                   SsCouplingSameMeshMap map) {
  const SsStatus status = plan->transfer_type == SS_COUPLING_TRANSFER_SAME_MESH
                              ? ss_coupling_locate_same_mesh(plan, map)
                              : ss_coupling_locate_geometric(plan);
  int state[2] = {(int)status, 0};
  for (size_t index = 0; index < plan->local_point_count && !state[1]; ++index) {
    state[1] = plan->points[index].source_rank != plan->rank;
  }
  if (MPI_Allreduce(MPI_IN_PLACE, state, 2, MPI_INT, MPI_MAX, plan->communicator) !=
      MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  plan->has_remote_points = state[1];
  return (SsStatus)state[0];
}
