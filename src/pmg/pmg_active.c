/* Active-rank communicators, assembled operator routes and sparse vector bridges.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "pmg_internal.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/core/util.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static void ss_pmg_matrix_route_clear(SsPmgMatrixRoute *route) {
  if (route->send_counts && route->neighbors != MPI_COMM_NULL) {
    (void)MPI_Comm_free(&route->neighbors);
  }
  free(route->send_counts);
  free(route->receive_counts);
  free(route->send_displacements);
  free(route->receive_displacements);
  free(route->row_offsets);
  free(route->send_values);
  free(route->receive_values);
  memset(route, 0, sizeof(*route));
}

void ss_pmg_level_runtime_clear(SsPmgLevel *level) {
  ss_pmg_matrix_route_clear(&level->matrix_route);
  (void)PetscSFDestroy(&level->row_bridge);
  free(level->bridge_world_fallback);
  free(level->bridge_active_fallback);
  level->bridge_world_fallback = NULL;
  level->bridge_active_fallback = NULL;
  (void)KSPDestroy(&level->ksp);
  (void)VecDestroy(&level->active_rhs);
  (void)VecDestroy(&level->active_solution);
  (void)MatNullSpaceDestroy(&level->active_near_nullspace);
  (void)MatNullSpaceDestroy(&level->world_near_nullspace);
  (void)MatDestroy(&level->active_operator);
  if (level->owns_active_communicator && level->active_communicator != MPI_COMM_NULL) {
    (void)MPI_Comm_free(&level->active_communicator);
  }
  level->active_communicator = MPI_COMM_NULL;
  level->owns_active_communicator = 0;
  level->local_active = 0;
  level->local_owns_ksp = 0;
  level->local_active_rank = -1;
  level->repeated_factorization_reuses_row_permutation = 0;
  level->variable_point_block_count = 0U;
  memset(&level->nullspace_record, 0, sizeof(level->nullspace_record));
}

/* Active matrices use contiguous balanced ownership in active-rank order. */
static PetscMPIInt ss_pmg_partition_owner(PetscInt n, PetscMPIInt parts, PetscInt row,
                                          PetscInt *offset) {
  const PetscInt size = n / (PetscInt)parts;
  const PetscInt larger_parts = n % (PetscInt)parts;
  const PetscInt split = larger_parts != 0 ? (size + 1) * larger_parts : 0;
  if (row < split) {
    *offset = row % (size + 1);
    return (PetscMPIInt)(row / (size + 1));
  }
  *offset = (row - split) % size;
  return (PetscMPIInt)(larger_parts + (row - split) / size);
}

SsStatus ss_pmg_level_communicator_create(SsPmg *pmg, size_t level_index) {
  SsPmgLevel *level = &pmg->levels[level_index];
  level->active_communicator = MPI_COMM_NULL;
  level->local_active_rank = -1;
  for (size_t i = 0U; i < level->policy.active_rank_count; ++i) {
    if (level->active_ranks[i] == pmg->world_rank) {
      level->local_active = 1;
      level->local_active_rank = (PetscMPIInt)i;
      break;
    }
  }
  if (level->identity_world_layout) {
    level->active_communicator = pmg->communicator;
    return SS_OK;
  }
  if (MPI_Comm_split(pmg->communicator, level->local_active ? 0 : MPI_UNDEFINED,
                     level->local_active_rank,
                     &level->active_communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  level->owns_active_communicator = level->local_active;
  return SS_OK;
}

static void *ss_pmg_nonempty_calloc(size_t count, size_t size) {
  return calloc(count == 0U ? 1U : count, size);
}

static SsStatus ss_pmg_matrix_route_counts(SsPmg *pmg, SsPmgLevel *level, Mat matrix,
                                           PetscInt size) {
  SsPmgMatrixRoute *route = &level->matrix_route;
  SsStatus status = SS_OK;
  const size_t ranks = (size_t)pmg->world_size;
  route->neighbors = MPI_COMM_NULL;
  route->send_counts = calloc(ranks, sizeof(*route->send_counts));
  route->receive_counts = calloc(ranks, sizeof(*route->receive_counts));
  route->send_displacements = calloc(ranks, sizeof(*route->send_displacements));
  route->receive_displacements = calloc(ranks, sizeof(*route->receive_displacements));
  if (MatGetOwnershipRange(matrix, &route->row_start, &route->row_end) !=
          PETSC_SUCCESS ||
      MatGetNonzeroState(matrix, &route->nonzero_state) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  } else {
    route->row_offsets = calloc((size_t)(route->row_end - route->row_start) + 1U,
                                sizeof(*route->row_offsets));
    if (!route->send_counts || !route->receive_counts || !route->send_displacements ||
        !route->receive_displacements || !route->row_offsets) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  for (PetscInt row = route->row_start; status == SS_OK && row < route->row_end;
       ++row) {
    PetscInt count = 0;
    PetscInt offset;
    const PetscMPIInt owner = ss_pmg_partition_owner(
        size, (PetscMPIInt)level->policy.active_rank_count, row, &offset);
    const PetscMPIInt target = level->active_ranks[owner];
    if (MatGetRow(matrix, row, &count, NULL, NULL) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
      break;
    }
    if (count < 0 || count > INT_MAX - route->send_counts[target]) {
      status = SS_ERROR_OVERFLOW;
    } else {
      route->row_offsets[row - route->row_start + 1] = (int)count;
      route->send_counts[target] += (int)count;
    }
    if (MatRestoreRow(matrix, row, &count, NULL, NULL) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  if (MPI_Alltoall(route->send_counts, 1, MPI_INT, route->receive_counts, 1, MPI_INT,
                   pmg->communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  for (size_t rank = 0; rank < ranks; ++rank) {
    if (route->send_total > INT_MAX - route->send_counts[rank] ||
        route->receive_total > INT_MAX - route->receive_counts[rank]) {
      return SS_ERROR_OVERFLOW;
    }
    route->send_displacements[rank] = route->send_total;
    route->receive_displacements[rank] = route->receive_total;
    route->send_total += route->send_counts[rank];
    route->receive_total += route->receive_counts[rank];
  }
  route->send_values =
      ss_pmg_nonempty_calloc((size_t)route->send_total, sizeof(*route->send_values));
  route->receive_values = ss_pmg_nonempty_calloc((size_t)route->receive_total,
                                                 sizeof(*route->receive_values));
  return route->send_values && route->receive_values ? SS_OK : SS_ERROR_OUT_OF_MEMORY;
}

/* Compress the setup routing into a persistent sparse neighbor exchange.
 * The order of edges in an adjacent graph follows the input arrays. */
static SsStatus ss_pmg_matrix_route_neighbors(SsPmg *pmg, SsPmgMatrixRoute *route) {
  int *sources = calloc((size_t)pmg->world_size, sizeof(*sources));
  int *targets = calloc((size_t)pmg->world_size, sizeof(*targets));
  int *weights = calloc((size_t)pmg->world_size, sizeof(*weights));
  int incoming = 0, outgoing = 0;
  SsStatus status = ss_status_allreduce_max(
      pmg->communicator,
      sources && targets && weights ? SS_OK : SS_ERROR_OUT_OF_MEMORY);
  if (status == SS_OK) {
    for (int rank = 0; rank < pmg->world_size; ++rank) {
      weights[rank] = 1;
      if (route->receive_counts[rank] != 0) {
        sources[incoming] = rank;
        route->receive_displacements[incoming] = route->receive_displacements[rank];
        route->receive_counts[incoming++] = route->receive_counts[rank];
      }
      if (route->send_counts[rank] != 0) {
        targets[outgoing] = rank;
        route->send_displacements[outgoing] = route->send_displacements[rank];
        route->send_counts[outgoing++] = route->send_counts[rank];
      }
    }
    if (MPI_Dist_graph_create_adjacent(pmg->communicator, incoming, sources, weights,
                                       outgoing, targets, weights, MPI_INFO_NULL, 0,
                                       &route->neighbors) != MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  free(weights);
  free(targets);
  free(sources);
  return ss_status_allreduce_max(pmg->communicator, status);
}

static SsStatus ss_pmg_matrix_route_pattern(SsPmg *pmg, SsPmgLevel *level, Mat matrix,
                                            PetscInt size) {
  SsPmgMatrixRoute *route = &level->matrix_route;
  int *cursor = NULL;
  PetscInt *rows = NULL, *columns = NULL, *active_rows = NULL, *active_columns = NULL;
  size_t first = 0U, local_size = 0U;
  SsStatus status = ss_pmg_matrix_route_counts(pmg, level, matrix, size);
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  cursor = calloc((size_t)pmg->world_size, sizeof(*cursor));
  rows = ss_pmg_nonempty_calloc((size_t)route->send_total, sizeof(*rows));
  columns = ss_pmg_nonempty_calloc((size_t)route->send_total, sizeof(*columns));
  active_rows =
      ss_pmg_nonempty_calloc((size_t)route->receive_total, sizeof(*active_rows));
  active_columns =
      ss_pmg_nonempty_calloc((size_t)route->receive_total, sizeof(*active_columns));
  if (!cursor || !rows || !columns || !active_rows || !active_columns) {
    status = SS_ERROR_OUT_OF_MEMORY;
  }
  if (status == SS_OK) {
    memcpy(cursor, route->send_displacements,
           (size_t)pmg->world_size * sizeof(*cursor));
  }
  for (PetscInt row = route->row_start; status == SS_OK && row < route->row_end;
       ++row) {
    PetscInt count = 0;
    PetscInt offset;
    const PetscInt *indices = NULL;
    const PetscMPIInt owner = ss_pmg_partition_owner(
        size, (PetscMPIInt)level->policy.active_rank_count, row, &offset);
    const PetscMPIInt target = level->active_ranks[owner];
    const int position = cursor[target];
    if (MatGetRow(matrix, row, &count, &indices, NULL) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
      break;
    }
    if (count != route->row_offsets[row - route->row_start + 1]) {
      status = SS_ERROR_INTEGRITY;
    } else {
      route->row_offsets[row - route->row_start] = position;
      for (PetscInt col = 0; col < count; ++col) {
        rows[position + col] = row;
        columns[position + col] = indices[col];
      }
      cursor[target] += (int)count;
    }
    if (MatRestoreRow(matrix, row, &count, &indices, NULL) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  const int rows_status = MPI_Alltoallv(
      rows, route->send_counts, route->send_displacements, MPIU_INT, active_rows,
      route->receive_counts, route->receive_displacements, MPIU_INT, pmg->communicator);
  const int columns_status = MPI_Alltoallv(
      columns, route->send_counts, route->send_displacements, MPIU_INT, active_columns,
      route->receive_counts, route->receive_displacements, MPIU_INT, pmg->communicator);
  if (rows_status != MPI_SUCCESS || columns_status != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (level->local_active) {
    ss_partition_range((size_t)size, level->local_active_rank,
                       (int)level->policy.active_rank_count, &first, &local_size);
    for (int i = 0; i < route->receive_total; ++i) {
      if (active_rows[i] < (PetscInt)first ||
          active_rows[i] >= (PetscInt)(first + local_size) || active_columns[i] < 0 ||
          active_columns[i] >= size) {
        status = SS_ERROR_INTEGRITY;
      }
    }
  } else if (route->receive_total != 0) {
    status = SS_ERROR_INTEGRITY;
  }
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (level->local_active &&
      (MatCreate(level->active_communicator, &level->active_operator) !=
           PETSC_SUCCESS ||
       MatSetSizes(level->active_operator, (PetscInt)local_size, (PetscInt)local_size,
                   size, size) != PETSC_SUCCESS ||
       MatSetType(level->active_operator, MATAIJ) != PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status == SS_OK && level->local_active &&
      MatSetPreallocationCOO(level->active_operator, (PetscCount)route->receive_total,
                             active_rows, active_columns) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status == SS_OK) {
    status = ss_pmg_matrix_route_neighbors(pmg, route);
  }

cleanup:
  free(active_columns);
  free(active_rows);
  free(columns);
  free(rows);
  free(cursor);
  return status;
}

static SsStatus ss_pmg_matrix_route_values(SsPmg *pmg, SsPmgLevel *level, Mat matrix) {
  SsPmgMatrixRoute *route = &level->matrix_route;
  PetscObjectState state;
  SsStatus status = SS_OK;
  if (MatGetNonzeroState(matrix, &state) != PETSC_SUCCESS ||
      state != route->nonzero_state) {
    status = SS_ERROR_INTEGRITY;
  }
  for (PetscInt row = route->row_start; status == SS_OK && row < route->row_end;
       ++row) {
    PetscInt count = 0;
    const PetscScalar *values = NULL;
    const int start = route->row_offsets[row - route->row_start];
    if (MatGetRow(matrix, row, &count, NULL, &values) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
      break;
    }
    if (count != 0) {
      memcpy(route->send_values + start, values, (size_t)count * sizeof(*values));
    }
    if (MatRestoreRow(matrix, row, &count, NULL, &values) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status == SS_OK &&
      MPI_Neighbor_alltoallv(
          route->send_values, route->send_counts, route->send_displacements,
          MPIU_SCALAR, route->receive_values, route->receive_counts,
          route->receive_displacements, MPIU_SCALAR, route->neighbors) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status == SS_OK && level->local_active &&
      MatSetValuesCOO(level->active_operator, route->receive_values, INSERT_VALUES) !=
          PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  return ss_status_allreduce_max(pmg->communicator, status);
}

SsStatus ss_pmg_active_matrix_update(SsPmg *pmg, size_t level_index, Mat world_matrix,
                                     int reuse) {
  SsPmgLevel *level = &pmg->levels[level_index];
  PetscInt size = 0;
  SsStatus status = SS_OK;
  if (level->identity_world_layout) {
    if (!reuse) {
      if (PetscObjectReference((PetscObject)world_matrix) != PETSC_SUCCESS) {
        return SS_ERROR_INTERNAL;
      }
      level->active_operator = world_matrix;
    }
    return level->active_operator == world_matrix ? SS_OK : SS_ERROR_INTEGRITY;
  }
  if (!reuse) {
    if (MatGetSize(world_matrix, &size, NULL) != PETSC_SUCCESS || size <= 0) {
      status = SS_ERROR_TYPE_MISMATCH;
    }
    status = ss_status_allreduce_max(pmg->communicator, status);
    if (status == SS_OK) {
      status = ss_pmg_matrix_route_pattern(pmg, level, world_matrix, size);
    }
  }
  if (status != SS_OK) {
    return status;
  }
  return ss_pmg_matrix_route_values(pmg, level, world_matrix);
}

SsStatus ss_pmg_row_bridge_create(SsPmg *pmg, size_t level_index) {
  SsPmgLevel *level = &pmg->levels[level_index];
  PetscInt first = 0, end = 0, roots = 0, n = 0;
  PetscSFNode *owners = NULL;
  SsStatus status = SS_OK;
  Mat world = pmg->operators[level_index];
  if (MatGetSize(world, &n, NULL) != PETSC_SUCCESS ||
      MatGetOwnershipRange(world, &first, &end) != PETSC_SUCCESS ||
      (level->local_active &&
       MatGetLocalSize(level->active_operator, &roots, NULL) != PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK && end > first) {
    owners = calloc((size_t)(end - first), sizeof(*owners));
    if (owners == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  if (status == SS_OK) {
    /* Keep valid CPU buffers available even if one rank cannot acquire a
     * Vec array. The other ranks can then complete the same SF transfer. */
    level->bridge_world_fallback =
        ss_pmg_nonempty_calloc((size_t)(end - first), sizeof(PetscScalar));
    level->bridge_active_fallback =
        ss_pmg_nonempty_calloc((size_t)roots, sizeof(PetscScalar));
    level->bridge_global_size = n;
    if (!level->bridge_world_fallback || !level->bridge_active_fallback) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  for (PetscInt row = first; status == SS_OK && row < end; ++row) {
    PetscSFNode *leaf = &owners[row - first];
    const PetscMPIInt owner = ss_pmg_partition_owner(
        n, (PetscMPIInt)level->policy.active_rank_count, row, &leaf->index);
    leaf->rank = level->active_ranks[owner];
  }
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status == SS_OK &&
      (PetscSFCreate(pmg->communicator, &level->row_bridge) != PETSC_SUCCESS ||
       PetscSFSetType(level->row_bridge, PETSCSFBASIC) != PETSC_SUCCESS ||
       PetscSFSetGraph(level->row_bridge, roots, end - first, NULL, PETSC_COPY_VALUES,
                       owners, PETSC_COPY_VALUES) != PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  free(owners);
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status == SS_OK && PetscSFSetUp(level->row_bridge) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  return ss_status_allreduce_max(pmg->communicator, status);
}

static SsStatus ss_pmg_bridge(SsPmg *pmg, size_t level_index, Vec source,
                              Vec destination, int to_active, int account) {
  SsPmgLevel *level = &pmg->levels[level_index];
  const PetscScalar *input = NULL;
  PetscScalar *output = NULL;
  int input_open = 0, output_open = 0;
  SsStatus status = SS_OK;
  if (source != NULL) {
    input_open = VecGetArrayRead(source, &input) == PETSC_SUCCESS;
    if (!input_open) {
      input = to_active ? level->bridge_world_fallback : level->bridge_active_fallback;
      status = SS_ERROR_INTERNAL;
    }
  }
  if (destination != NULL) {
    output_open = VecGetArrayWrite(destination, &output) == PETSC_SUCCESS;
    if (!output_open) {
      output = to_active ? level->bridge_active_fallback : level->bridge_world_fallback;
      status = SS_ERROR_INTERNAL;
    }
  }
  PetscErrorCode exchange = to_active
                                ? PetscSFReduceBegin(level->row_bridge, MPIU_SCALAR,
                                                     input, output, MPI_REPLACE)
                                : PetscSFBcastBegin(level->row_bridge, MPIU_SCALAR,
                                                    input, output, MPI_REPLACE);
  if (exchange == PETSC_SUCCESS) {
    exchange = to_active ? PetscSFReduceEnd(level->row_bridge, MPIU_SCALAR, input,
                                            output, MPI_REPLACE)
                         : PetscSFBcastEnd(level->row_bridge, MPIU_SCALAR, input,
                                           output, MPI_REPLACE);
  }
  if (exchange != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (output_open && VecRestoreArrayWrite(destination, &output) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (input_open && VecRestoreArrayRead(source, &input) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (account) {
    const uint64_t bytes = (uint64_t)level->bridge_global_size * sizeof(PetscScalar);
    ++pmg->last_apply.bridge_calls;
    ++pmg->diagnostics.bridge_calls;
    pmg->last_apply.bridge_bytes += bytes;
    pmg->diagnostics.bridge_bytes += bytes;
  }
  return status;
}

SsStatus ss_pmg_bridge_to_active(SsPmg *pmg, size_t level_index, Vec world_vector,
                                 Vec active_vector, int account) {
  return ss_pmg_bridge(pmg, level_index, world_vector, active_vector, 1, account);
}

SsStatus ss_pmg_bridge_from_active(SsPmg *pmg, size_t level_index, Vec active_vector,
                                   Vec world_vector, int account) {
  return ss_pmg_bridge(pmg, level_index, active_vector, world_vector, 0, account);
}
