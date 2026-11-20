/* Validate, synchronize and hash resident essential constraint tuples.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "mesh_internal.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/core/sort.h"
#include "slope_internal/core/util.h"
#include "slope_internal/mesh/distribution.h"

#include <math.h>
#include <petscdmplex.h>
#include <petscsf.h>
#include <stdlib.h>
#include <string.h>

int ss_field_constraint_compare(const void *left_raw, const void *right_raw) {
  const SsStrongConstraint *left = left_raw;
  const SsStrongConstraint *right = right_raw;
  if (left->stable_entity_id != right->stable_entity_id) {
    return left->stable_entity_id < right->stable_entity_id ? -1 : 1;
  }
  if (left->entity_slot != right->entity_slot) {
    return left->entity_slot < right->entity_slot ? -1 : 1;
  }
  return (left->component > right->component) - (left->component < right->component);
}

static void ss_field_constraint_serialize(const SsStrongConstraint *item,
                                          unsigned char record[24]) {
  uint64_t bits = 0U;
  memcpy(&bits, &item->value, sizeof(bits));
  ss_mesh_store_u64(record, item->stable_entity_id);
  ss_mesh_store_u32(record + 8U, item->entity_slot);
  ss_mesh_store_u32(record + 12U, item->component);
  ss_mesh_store_u64(record + 16U, bits);
}

static SsStatus ss_field_constraint_hash(SsSha256 *hash,
                                         const SsStrongConstraint *items,
                                         size_t count) {
  for (size_t i = 0U; i < count; ++i) {
    unsigned char bytes[24];
    ss_field_constraint_serialize(&items[i], bytes);
    SsStatus status = ss_sha256_update(hash, bytes, sizeof(bytes));
    if (status != SS_OK) {
      return status;
    }
  }
  return SS_OK;
}

enum { SS_CONSTRAINT_SHARD_TARGET = 256, SS_CONSTRAINT_MAX_SHARDS = 65536 };

static size_t ss_field_constraint_shard(SsEntityId id, size_t shards) {
  return (size_t)(ss_mesh_distribution_hash(id) % (uint64_t)shards);
}

static int ss_field_constraint_owner(const void *raw, int ranks, const void *context) {
  const SsStrongConstraint *item = raw;
  const size_t shards = *(const size_t *)context;
  const size_t shard = ss_field_constraint_shard(item->stable_entity_id, shards);
  const size_t even = shards / (size_t)ranks;
  const size_t extra = shards % (size_t)ranks;
  const size_t split = extra * (even + 1U);
  return (int)(shard < split ? shard / (even + 1U) : extra + (shard - split) / even);
}

/* The v2 digest uses numbered shards independent of the MPI layout. A shard
 * hashes its sorted (entity, slot, component, value) tuples. Rank 0 then
 * hashes the version, tuple count and ordered list of raw shard digests. */
static SsStatus ss_field_constraint_digest(const SsMesh *mesh,
                                           const SsStrongConstraint *owned,
                                           size_t count, uint64_t total,
                                           char digest[SS_SHA256_HEX_CAPACITY]) {
  static const char version[] = "essential-constraint-hash-list-v2";
  unsigned char total_bytes[8];
  SsSha256 root_hash;
  ss_mesh_store_u64(total_bytes, total);
  ss_sha256_init(&root_hash);
  (void)ss_sha256_update(&root_hash, version, sizeof(version));
  (void)ss_sha256_update(&root_hash, total_bytes, sizeof(total_bytes));
  if (total == 0U) {
    ss_sha256_final(&root_hash, digest);
    return SS_OK;
  }
  const uint64_t desired_shards = 1U + (total - 1U) / SS_CONSTRAINT_SHARD_TARGET;
  const size_t shards = (size_t)PetscMin(desired_shards, SS_CONSTRAINT_MAX_SHARDS);
  size_t first = 0U, local_shards = 0U, ordered_count = 0U;
  SsStrongConstraint *ordered = NULL;
  ss_partition_range(shards, mesh->rank, mesh->size, &first, &local_shards);
  const size_t allocated = local_shards == 0U ? 1U : local_shards;
  SsSha256 *hashes = calloc(allocated, sizeof(*hashes));
  unsigned char (*local)[SS_SHA256_SIZE] = calloc(allocated, sizeof(*local));
  unsigned char (*all)[SS_SHA256_SIZE] = NULL;
  int *counts = NULL, *offsets = NULL;
  SsStatus status = hashes == NULL || local == NULL ? SS_ERROR_OUT_OF_MEMORY : SS_OK;
  if (mesh->rank == 0) {
    all = calloc(shards, sizeof(*all));
    counts = calloc((size_t)mesh->size, sizeof(*counts));
    offsets = calloc((size_t)mesh->size, sizeof(*offsets));
    if (all == NULL || counts == NULL || offsets == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (mesh->size > 1) {
    status = ss_mesh_route_records(mesh->communicator, owned, count, sizeof(*owned),
                                   ss_field_constraint_owner, &shards,
                                   (void **)&ordered, &ordered_count);
    if (status != SS_OK) {
      goto cleanup;
    }
    ss_sort(ordered, ordered_count, sizeof(*ordered), ss_field_constraint_compare);
    owned = ordered;
    count = ordered_count;
  }
  for (size_t i = 0U; i < local_shards; ++i) {
    ss_sha256_init(&hashes[i]);
  }
  for (size_t i = 0U; status == SS_OK && i < count; ++i) {
    const size_t shard = ss_field_constraint_shard(owned[i].stable_entity_id, shards);
    if (shard < first || shard - first >= local_shards) {
      status = SS_ERROR_INTEGRITY;
    } else {
      status = ss_field_constraint_hash(&hashes[shard - first], &owned[i], 1U);
    }
  }
  for (size_t i = 0U; i < local_shards; ++i) {
    ss_sha256_final_bytes(&hashes[i], local[i]);
  }
  if (mesh->rank == 0) {
    for (int rank = 0; rank < mesh->size; ++rank) {
      size_t start, assigned;
      ss_partition_range(shards, rank, mesh->size, &start, &assigned);
      counts[rank] = (int)(assigned * SS_SHA256_SIZE);
      offsets[rank] = (int)(start * SS_SHA256_SIZE);
    }
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status == SS_OK &&
      MPI_Gatherv(local, (int)(local_shards * SS_SHA256_SIZE), MPI_BYTE, all, counts,
                  offsets, MPI_BYTE, 0, mesh->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status == SS_OK) {
    if (mesh->rank == 0) {
      (void)ss_sha256_update(&root_hash, all, shards * sizeof(*all));
      ss_sha256_final(&root_hash, digest);
    }
    if (MPI_Bcast(digest, SS_SHA256_HEX_CAPACITY, MPI_CHAR, 0, mesh->communicator) !=
        MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }

cleanup:
  free(ordered);
  free(hashes);
  free(local);
  free(all);
  free(counts);
  free(offsets);
  return status;
}

typedef struct {
  PetscInt begin;
  PetscInt count;
} SsPointConstraintRange;

static SsStatus ss_field_constraint_sf_create(SsFieldSpace *space) {
  const SsMesh *mesh = space->mesh;
  PetscSF point_sf = NULL;
  PetscInt nroots = -1, nleaves = -1, nconstraints = 0, leaves = 0;
  const PetscInt *local = NULL;
  const PetscSFNode *remote = NULL;
  SsPointConstraintRange *own_ranges = NULL, *peer_ranges = NULL;
  PetscInt *local_indices = NULL;
  PetscSFNode *remote_indices = NULL;
  MPI_Datatype range_type = MPI_DATATYPE_NULL;
  SsStatus status = SS_OK;
  if (ss_checked_u64_to_petsc(space->constraint_count, &nconstraints) != SS_OK ||
      DMGetPointSF(mesh->dm, &point_sf) != PETSC_SUCCESS ||
      PetscSFGetGraph(point_sf, &nroots, &nleaves, &local, &remote) != PETSC_SUCCESS ||
      (nroots >= 0 && nroots != mesh->cache_chart_end)) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK && nroots >= 0) {
    const size_t points = nroots == 0 ? 1U : (size_t)nroots;
    const size_t count = nconstraints == 0 ? 1U : (size_t)nconstraints;
    own_ranges = calloc(points, sizeof(*own_ranges));
    peer_ranges = calloc(points, sizeof(*peer_ranges));
    local_indices = calloc(count, sizeof(*local_indices));
    remote_indices = calloc(count, sizeof(*remote_indices));
    if (own_ranges == NULL || peer_ranges == NULL || local_indices == NULL ||
        remote_indices == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    } else if (MPI_Type_contiguous(2, MPIU_INT, &range_type) != MPI_SUCCESS ||
               MPI_Type_commit(&range_type) != MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  for (size_t i = 0U;
       status == SS_OK && own_ranges != NULL && i < space->constraint_count;) {
    const SsEntityId id = space->constraints[i].stable_entity_id;
    size_t end = i + 1U;
    PetscInt point = -1;
    while (end < space->constraint_count &&
           space->constraints[end].stable_entity_id == id) {
      ++end;
    }
    status = ss_mesh_point_from_stable_id_any(mesh, id, &point);
    if (status == SS_OK) {
      own_ranges[point] = (SsPointConstraintRange){(PetscInt)i, (PetscInt)(end - i)};
    }
    i = end;
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status == SS_OK && nroots >= 0 &&
      (PetscSFBcastBegin(point_sf, range_type, own_ranges, peer_ranges, MPI_REPLACE) !=
           PETSC_SUCCESS ||
       PetscSFBcastEnd(point_sf, range_type, own_ranges, peer_ranges, MPI_REPLACE) !=
           PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  for (PetscInt i = 0; status == SS_OK && i < nleaves; ++i) {
    const PetscInt point = local == NULL ? i : local[i];
    const SsPointConstraintRange here = own_ranges[point];
    const SsPointConstraintRange peer = peer_ranges[point];
    if (here.count != peer.count) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    for (PetscInt j = 0; j < here.count; ++j) {
      local_indices[leaves] = here.begin + j;
      remote_indices[leaves++] = (PetscSFNode){remote[i].rank, peer.begin + j};
    }
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status == SS_OK &&
      (PetscSFCreate(mesh->communicator, &space->constraint_sf) != PETSC_SUCCESS ||
       PetscSFSetGraph(space->constraint_sf, nconstraints, leaves, local_indices,
                       PETSC_COPY_VALUES, remote_indices,
                       PETSC_COPY_VALUES) != PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  if (range_type != MPI_DATATYPE_NULL) {
    (void)MPI_Type_free(&range_type);
  }
  free(own_ranges);
  free(peer_ranges);
  free(local_indices);
  free(remote_indices);
  return ss_status_allreduce_max(mesh->communicator, status);
}

static SsStatus ss_field_constraint_sf_audit(SsFieldSpace *space) {
  enum { SS_TUPLE_SIZE = 24 };
  MPI_Comm comm = space->mesh->communicator;
  MPI_Datatype tuple = MPI_DATATYPE_NULL;
  const size_t count = space->constraint_count == 0U ? 1U : space->constraint_count;
  unsigned char *local = calloc(count, SS_TUPLE_SIZE);
  unsigned char *peer = calloc(count, SS_TUPLE_SIZE);
  SsStatus status = local == NULL || peer == NULL ? SS_ERROR_OUT_OF_MEMORY : SS_OK;
  if (status == SS_OK &&
      (MPI_Type_contiguous(SS_TUPLE_SIZE, MPI_BYTE, &tuple) != MPI_SUCCESS ||
       MPI_Type_commit(&tuple) != MPI_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(comm, status);
  if (status == SS_OK) {
    for (size_t i = 0U; i < space->constraint_count; ++i) {
      ss_field_constraint_serialize(&space->constraints[i], local + SS_TUPLE_SIZE * i);
    }
    memcpy(peer, local, count * SS_TUPLE_SIZE);
    if (PetscSFBcastBegin(space->constraint_sf, tuple, local, peer, MPI_REPLACE) !=
            PETSC_SUCCESS ||
        PetscSFBcastEnd(space->constraint_sf, tuple, local, peer, MPI_REPLACE) !=
            PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    } else if (memcmp(local, peer, count * SS_TUPLE_SIZE) != 0) {
      status = SS_ERROR_INTEGRITY;
    }
  }
  if (tuple != MPI_DATATYPE_NULL) {
    (void)MPI_Type_free(&tuple);
  }
  free(local);
  free(peer);
  return ss_status_allreduce_max(comm, status);
}

static SsStatus ss_field_replicated_identity(const SsMesh *mesh,
                                             const SsStrongConstraint *constraints,
                                             size_t count,
                                             char digest[SS_SHA256_HEX_CAPACITY]) {
  SsSha256 hash;
  char root[SS_SHA256_HEX_CAPACITY];
  ss_sha256_init(&hash);
  SsStatus status = ss_field_constraint_hash(&hash, constraints, count);
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  ss_sha256_final(&hash, digest);
  if (mesh->rank == 0) {
    memcpy(root, digest, sizeof(root));
  }
  if (MPI_Bcast(root, SS_SHA256_HEX_CAPACITY, MPI_CHAR, 0, mesh->communicator) !=
      MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  return ss_status_allreduce_max(mesh->communicator, strcmp(root, digest) == 0
                                                         ? SS_OK
                                                         : SS_ERROR_INVALID_ARGUMENT);
}

SsStatus ss_field_constraints_prepare(SsFieldSpace *space, const SsFieldSpec *spec,
                                      int distributed, uint64_t *global_count,
                                      char digest[SS_SHA256_HEX_CAPACITY]) {
  SsMesh *mesh = space->mesh;
  SsStrongConstraint *owned = NULL;
  size_t owned_count = 0U, local_count = 0U;
  uint32_t metadata[3] = {spec->order, spec->components, (uint32_t)distributed};
  uint32_t minimum[3], maximum[3];
  SsStatus status = SS_OK;
  if (MPI_Allreduce(metadata, minimum, 3, MPI_UINT32_T, MPI_MIN, mesh->communicator) !=
          MPI_SUCCESS ||
      MPI_Allreduce(metadata, maximum, 3, MPI_UINT32_T, MPI_MAX, mesh->communicator) !=
          MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (memcmp(minimum, maximum, sizeof(minimum)) != 0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (size_t i = 0U; status == SS_OK && i < space->constraint_count; ++i) {
    const SsStrongConstraint *item = &space->constraints[i];
    if (i != 0U && ss_field_constraint_compare(item - 1, item) == 0) {
      status = SS_ERROR_DUPLICATE;
    } else if (!isfinite(item->value) || item->component >= spec->components) {
      status = SS_ERROR_INVALID_ARGUMENT;
    }
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status == SS_OK && distributed == 0) {
    status = ss_field_replicated_identity(mesh, space->constraints,
                                          space->constraint_count, digest);
  }
  for (size_t i = 0U; status == SS_OK && i < space->constraint_count; ++i) {
    const SsStrongConstraint item = space->constraints[i];
    PetscInt point = -1, depth = -1;
    if (ss_mesh_point_from_stable_id_any(mesh, item.stable_entity_id, &point) !=
        SS_OK) {
      if (distributed != 0) {
        status = SS_ERROR_NOT_FOUND;
      }
      continue;
    }
    if (DMPlexGetPointDepth(mesh->dm, point, &depth) != PETSC_SUCCESS || depth < 0 ||
        item.entity_slot >=
            ss_mesh_slots_for_depth(mesh->dimension, spec->order, (uint32_t)depth)) {
      status = SS_ERROR_INVALID_ARGUMENT;
      break;
    }
    owned_count += ss_mesh_point_is_ghost(mesh, mesh->dm, point) == 0;
    space->constraints[local_count++] = item;
  }
  space->constraint_count = local_count;
  status = ss_status_allreduce_max(mesh->communicator, status);
  const uint64_t local_owned = owned_count;
  if (status == SS_OK && MPI_Allreduce(&local_owned, global_count, 1, MPI_UINT64_T,
                                       MPI_SUM, mesh->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK && distributed == 0 && *global_count != spec->constraint_count) {
    status = SS_ERROR_NOT_FOUND;
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  if (distributed != 0 && *global_count != 0U) {
    status = ss_field_constraint_sf_create(space);
    if (status == SS_OK) {
      status = ss_field_constraint_sf_audit(space);
    }
  }
  if (status == SS_OK && owned_count != 0U) {
    owned = calloc(owned_count, sizeof(*owned));
    if (owned == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  size_t next = 0U;
  for (size_t i = 0U; status == SS_OK && i < local_count; ++i) {
    PetscInt point = -1;
    status = ss_mesh_point_from_stable_id_any(
        mesh, space->constraints[i].stable_entity_id, &point);
    if (status == SS_OK && ss_mesh_point_is_ghost(mesh, mesh->dm, point) == 0) {
      owned[next++] = space->constraints[i];
    }
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status == SS_OK) {
    status =
        ss_field_constraint_digest(mesh, owned, owned_count, *global_count, digest);
  }
  free(owned);
  return status;
}

static SsStatus ss_field_constraints_exchange(PetscSF sf, size_t count, MPI_Op op,
                                              uint64_t *values, uint64_t *owners) {
  if (sf == NULL) {
    return SS_OK;
  }
  memcpy(owners, values, count * sizeof(*values));
  if (PetscSFReduceBegin(sf, MPI_UINT64_T, values, owners, op) != PETSC_SUCCESS ||
      PetscSFReduceEnd(sf, MPI_UINT64_T, values, owners, op) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  memcpy(values, owners, count * sizeof(*values));
  if (PetscSFBcastBegin(sf, MPI_UINT64_T, owners, values, MPI_REPLACE) !=
          PETSC_SUCCESS ||
      PetscSFBcastEnd(sf, MPI_UINT64_T, owners, values, MPI_REPLACE) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  return SS_OK;
}

SsStatus ss_field_space_resolve_constraint_values(
    const SsFieldSpace *space, const SsConstraintCandidate *candidates,
    SsStrongConstraint *constraints, size_t count) {
  if (space == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  MPI_Comm comm = space->mesh->communicator;
  const size_t capacity = count == 0U ? 1U : count;
  uint64_t *values = calloc(capacity, sizeof(*values));
  uint64_t *owners = calloc(capacity, sizeof(*owners));
  unsigned char *chosen = calloc(capacity, sizeof(*chosen));
  SsStatus status = values == NULL || owners == NULL || chosen == NULL
                        ? SS_ERROR_OUT_OF_MEMORY
                        : SS_OK;
  if (count != space->constraint_count ||
      (count != 0U && (constraints == NULL || candidates == NULL))) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  for (size_t i = 0U; status == SS_OK && i < count; ++i) {
    if (ss_field_constraint_compare(&space->constraints[i], &constraints[i]) != 0) {
      status = SS_ERROR_INTEGRITY;
    } else {
      values[i] = candidates[i].present == 0 ? UINT64_MAX : candidates[i].cell_id;
    }
  }
  status = ss_status_allreduce_max(comm, status);
  if (status == SS_OK) {
    status = ss_field_constraints_exchange(space->constraint_sf, count, MPI_MIN, values,
                                           owners);
  }
  for (size_t i = 0U; status == SS_OK && i < count; ++i) {
    chosen[i] = (unsigned char)(candidates[i].present != 0 &&
                                values[i] == candidates[i].cell_id);
    values[i] = chosen[i];
  }
  status = ss_status_allreduce_max(comm, status);
  if (status == SS_OK) {
    status = ss_field_constraints_exchange(space->constraint_sf, count, MPI_SUM, values,
                                           owners);
  }
  for (size_t i = 0U; status == SS_OK && i < count; ++i) {
    if (values[i] != 1U) {
      status = SS_ERROR_INTEGRITY;
    } else {
      values[i] = 0U;
      if (chosen[i] != 0U) {
        memcpy(&values[i], &candidates[i].value, sizeof(values[i]));
      }
    }
  }
  status = ss_status_allreduce_max(comm, status);
  if (status == SS_OK) {
    status = ss_field_constraints_exchange(space->constraint_sf, count, MPI_BOR, values,
                                           owners);
  }
  for (size_t i = 0U; status == SS_OK && i < count; ++i) {
    double value;
    memcpy(&value, &values[i], sizeof(value));
    if (!isfinite(value)) {
      status = SS_ERROR_INTEGRITY;
    } else {
      constraints[i].value = value;
    }
  }
  free(values);
  free(owners);
  free(chosen);
  return ss_status_allreduce_max(comm, status);
}
