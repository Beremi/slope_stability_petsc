/* Assembly plans: owned-row layout, sparsity, PETSc matrix and vector
 * creation and collective plan errors.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "assembly_impl.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/core/sort.h"
#include "slope_internal/core/util.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  PetscInt row;
  SsEntityId cell;
} SsRowCellPair;

typedef struct {
  PetscInt row;
  size_t cell_index;
  size_t row_slot;
} SsRowCellOccurrence;

SsStatus ss_assembly_collective_status_private(MPI_Comm communicator,
                                               SsStatus local_status) {
  const int global = (int)ss_status_allreduce_max(communicator, local_status);
  if (global < (int)SS_OK || global > (int)SS_ERROR_INTERNAL) {
    return SS_ERROR_INTERNAL;
  }
  return (SsStatus)global;
}

void ss_assembly_error_private(SsErrorRecord *error, SsStatus status, const char *code,
                               SsEntityId entity, int rank) {
  if (status == SS_OK) {
    if (error != NULL) {
      memset(error, 0, sizeof(*error));
    }
    return;
  }
  ss_error_set(error, status,
               (SsErrorKey){0U, 1U, entity, 0U, 0U, rank < 0 ? 0U : (uint32_t)rank},
               code, code);
}

static int ss_row_cell_pair_compare(const void *left, const void *right) {
  const SsRowCellPair *lhs = (const SsRowCellPair *)left;
  const SsRowCellPair *rhs = (const SsRowCellPair *)right;
  if (lhs->row != rhs->row) {
    return lhs->row < rhs->row ? -1 : 1;
  }
  return lhs->cell < rhs->cell ? -1 : lhs->cell > rhs->cell ? 1 : 0;
}

static int ss_row_cell_occurrence_compare(const void *left, const void *right) {
  const SsRowCellOccurrence *lhs = (const SsRowCellOccurrence *)left;
  const SsRowCellOccurrence *rhs = (const SsRowCellOccurrence *)right;
  if (lhs->row != rhs->row) {
    return lhs->row < rhs->row ? -1 : 1;
  }
  if (lhs->cell_index != rhs->cell_index) {
    return lhs->cell_index < rhs->cell_index ? -1 : 1;
  }
  if (lhs->row_slot != rhs->row_slot) {
    return lhs->row_slot < rhs->row_slot ? -1 : 1;
  }
  return 0;
}

static int ss_row_cell_occurrence_same_cell(const SsRowCellOccurrence *left,
                                            const SsRowCellOccurrence *right) {
  return left->row == right->row && left->cell_index == right->cell_index ? 1 : 0;
}

static SsStatus ss_plan_row_occurrences(const SsAssemblyPlan *plan,
                                        SsRowCellOccurrence **occurrences,
                                        size_t *occurrence_count) {
  size_t upper_bound = 0U;
  size_t count = 0U;
  size_t cell_index;
  SsRowCellOccurrence *items = NULL;
  if (plan == NULL || occurrences == NULL || occurrence_count == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *occurrences = NULL;
  *occurrence_count = 0U;
  for (cell_index = 0U; cell_index < plan->cell_count; ++cell_index) {
    if (upper_bound > SIZE_MAX - plan->cells[cell_index].closure_count) {
      return SS_ERROR_OVERFLOW;
    }
    upper_bound += plan->cells[cell_index].closure_count;
  }
  if (upper_bound > SIZE_MAX / sizeof(*items)) {
    return SS_ERROR_OVERFLOW;
  }
  if (upper_bound != 0U) {
    items = (SsRowCellOccurrence *)malloc(upper_bound * sizeof(*items));
    if (items == NULL) {
      return SS_ERROR_OUT_OF_MEMORY;
    }
  }
  for (cell_index = 0U; cell_index < plan->cell_count; ++cell_index) {
    const SsCellPlan *cell = &plan->cells[cell_index];
    size_t slot;
    for (slot = 0U; slot < cell->closure_count; ++slot) {
      const PetscInt row = cell->global_indices[slot];
      if (cell->constrained[slot] != 0U || row < plan->row_start ||
          row >= plan->row_end) {
        continue;
      }
      if (items == NULL || count >= upper_bound) {
        free(items);
        return SS_ERROR_INTEGRITY;
      }
      items[count++] = (SsRowCellOccurrence){row, cell_index, slot};
    }
  }
  if (count > 1U) {
    size_t source;
    size_t unique = 0U;
    ss_sort(items, count, sizeof(*items), ss_row_cell_occurrence_compare);
    for (source = 0U; source < count; ++source) {
      if (unique == 0U ||
          ss_row_cell_occurrence_same_cell(&items[source], &items[unique - 1U]) == 0) {
        items[unique++] = items[source];
      }
    }
    count = unique;
  }
  *occurrences = items;
  *occurrence_count = count;
  return SS_OK;
}

static size_t ss_cell_unique_free_rows(const SsCellPlan *cell, PetscInt *rows) {
  size_t count = 0U;
  size_t slot;
  for (slot = 0U; slot < cell->closure_count; ++slot) {
    if (cell->constrained[slot] != 0U || cell->global_indices[slot] < 0) {
      continue;
    }
    rows[count++] = cell->global_indices[slot];
  }
  if (count > 1U) {
    size_t source;
    size_t unique = 1U;
    ss_sort(rows, count, sizeof(*rows), ss_petsc_int_compare);
    for (source = 1U; source < count; ++source) {
      if (rows[source] != rows[unique - 1U]) {
        rows[unique++] = rows[source];
      }
    }
    count = unique;
  }
  return count;
}

static int ss_owner_from_ranges(PetscInt row, const PetscInt *starts,
                                const PetscInt *ends, int size) {
  int rank;
  for (rank = 0; rank < size; ++rank) {
    if (row >= starts[rank] && row < ends[rank]) {
      return rank;
    }
  }
  return -1;
}

static SsStatus ss_plan_row_ownership(SsAssemblyPlan *plan) {
  uint64_t local_rows = plan->field_summary.owned_free_dof_count;
  uint64_t prefix = 0U;
  uint64_t global_rows = 0U;
  size_t row;
  if (MPI_Exscan(&local_rows, &prefix, 1, MPI_UINT64_T, MPI_SUM, plan->communicator) !=
          MPI_SUCCESS ||
      MPI_Allreduce(&local_rows, &global_rows, 1, MPI_UINT64_T, MPI_SUM,
                    plan->communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (plan->rank == 0) {
    prefix = 0U;
  }
  if (prefix > (uint64_t)PETSC_MAX_INT ||
      local_rows > (uint64_t)PETSC_MAX_INT - prefix ||
      global_rows != plan->field_summary.global_free_dof_count ||
      local_rows > SIZE_MAX) {
    return SS_ERROR_OVERFLOW;
  }
  plan->row_start = (PetscInt)prefix;
  plan->row_end = (PetscInt)(prefix + local_rows);
  plan->row_count = (size_t)local_rows;
  if (plan->row_count != 0U) {
    if (plan->row_count > SIZE_MAX / sizeof(*plan->rows)) {
      return SS_ERROR_OVERFLOW;
    }
    plan->rows = (SsOwnedRowPlan *)calloc(plan->row_count, sizeof(*plan->rows));
    if (plan->rows == NULL) {
      return SS_ERROR_OUT_OF_MEMORY;
    }
  }
  for (row = 0U; row < plan->row_count; ++row) {
    plan->rows[row].global_row = plan->row_start + (PetscInt)row;
  }
  plan->summary.local_owned_rows = local_rows;
  plan->summary.global_rows = global_rows;
  return SS_OK;
}

static SsStatus ss_plan_row_metadata(SsAssemblyPlan *plan) {
  size_t cell_index;
  unsigned char *found = NULL;
  if (plan->row_count != 0U) {
    found = (unsigned char *)calloc(plan->row_count, sizeof(*found));
    if (found == NULL) {
      return SS_ERROR_OUT_OF_MEMORY;
    }
  }
  for (cell_index = 0U; cell_index < plan->cell_count; ++cell_index) {
    const SsCellPlan *cell = &plan->cells[cell_index];
    size_t slot;
    for (slot = 0U; slot < cell->closure_count; ++slot) {
      PetscInt row;
      size_t local_row;
      SsOwnedRowPlan *record;
      if (cell->constrained[slot] != 0U || cell->owned[slot] == 0U) {
        continue;
      }
      row = cell->global_indices[slot];
      if (row < plan->row_start || row >= plan->row_end) {
        free(found);
        return SS_ERROR_INTEGRITY;
      }
      local_row = (size_t)(row - plan->row_start);
      if (local_row >= plan->row_count || found == NULL || plan->rows == NULL) {
        free(found);
        return SS_ERROR_INTEGRITY;
      }
      record = &plan->rows[local_row];
      if (found[local_row] == 0U) {
        record->component = cell->field_components[slot];
        record->field_entity_id = cell->field_entity_ids[slot];
        record->field_entity_slot = cell->field_entity_slots[slot];
        found[local_row] = 1U;
      } else if (record->component != cell->field_components[slot] ||
                 record->field_entity_id != cell->field_entity_ids[slot] ||
                 record->field_entity_slot != cell->field_entity_slots[slot]) {
        free(found);
        return SS_ERROR_INTEGRITY;
      }
    }
  }
  for (cell_index = 0U; cell_index < plan->row_count; ++cell_index) {
    if (found[cell_index] == 0U) {
      free(found);
      return SS_ERROR_INTEGRITY;
    }
  }
  free(found);
  return SS_OK;
}

static SsStatus ss_plan_build_rows(SsAssemblyPlan *plan, int matrix_pattern) {
  size_t local_row;
  size_t occurrence_index = 0U;
  size_t occurrence_count = 0U;
  SsRowCellOccurrence *occurrences = NULL;
  PetscInt *columns = NULL;
  uint64_t nonzeros = 0U;
  uint64_t diagonal = 0U;
  uint64_t off_diagonal = 0U;
  uint64_t contributions = 0U;
  SsStatus status = ss_plan_row_occurrences(plan, &occurrences, &occurrence_count);
  if (status != SS_OK) {
    return status;
  }
  if (occurrence_count != 0U) {
    if (occurrence_count > SIZE_MAX / sizeof(*plan->row_cell_slots)) {
      status = SS_ERROR_OVERFLOW;
      goto cleanup;
    }
    plan->row_cell_slots =
        (SsRowCellSlot *)malloc(occurrence_count * sizeof(*plan->row_cell_slots));
    if (plan->row_cell_slots == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
      goto cleanup;
    }
  }
  for (local_row = 0U; local_row < plan->row_count; ++local_row) {
    SsOwnedRowPlan *row = &plan->rows[local_row];
    const size_t occurrence_begin = occurrence_index;
    size_t column_upper_bound = 0U;
    size_t column_count = 0U;
    size_t incidence;
    while (occurrence_index < occurrence_count &&
           occurrences[occurrence_index].row == row->global_row) {
      const size_t cell_index = occurrences[occurrence_index].cell_index;
      if (cell_index >= plan->cell_count ||
          (matrix_pattern != 0 &&
           column_upper_bound > SIZE_MAX - plan->cells[cell_index].closure_count)) {
        status =
            cell_index >= plan->cell_count ? SS_ERROR_INTEGRITY : SS_ERROR_OVERFLOW;
        goto cleanup;
      }
      if (matrix_pattern != 0) {
        column_upper_bound += plan->cells[cell_index].closure_count;
      }
      ++occurrence_index;
    }
    row->cell_count = occurrence_index - occurrence_begin;
    if (row->cell_count == 0U ||
        (matrix_pattern != 0 && column_upper_bound > SIZE_MAX / sizeof(*columns))) {
      status = row->cell_count == 0U ? SS_ERROR_INTEGRITY : SS_ERROR_OVERFLOW;
      goto cleanup;
    }
    row->cell_slots = plan->row_cell_slots + occurrence_begin;
    if (matrix_pattern != 0) {
      columns = (PetscInt *)malloc(column_upper_bound * sizeof(*columns));
    }
    if (matrix_pattern != 0 && columns == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
      goto cleanup;
    }
    for (incidence = 0U; incidence < row->cell_count; ++incidence) {
      const SsRowCellOccurrence *occurrence =
          &occurrences[occurrence_begin + incidence];
      const SsCellPlan *cell = &plan->cells[occurrence->cell_index];
      size_t slot;
      plan->row_cell_slots[occurrence_begin + incidence] =
          (SsRowCellSlot){occurrence->cell_index, occurrence->row_slot};
      if (matrix_pattern != 0) {
        for (slot = 0U; slot < cell->closure_count; ++slot) {
          if (cell->constrained[slot] == 0U) {
            if (column_count >= column_upper_bound) {
              status = SS_ERROR_INTEGRITY;
              goto cleanup;
            }
            columns[column_count++] = cell->global_indices[slot];
          }
        }
      }
    }
    if (contributions > UINT64_MAX - (uint64_t)row->cell_count) {
      status = SS_ERROR_OVERFLOW;
      goto cleanup;
    }
    contributions += (uint64_t)row->cell_count;
    if (matrix_pattern == 0) {
      continue;
    }
    ss_sort(columns, column_count, sizeof(*columns), ss_petsc_int_compare);
    {
      size_t source;
      size_t unique = 0U;
      for (source = 0U; source < column_count; ++source) {
        if (columns[source] < 0) {
          status = SS_ERROR_INTEGRITY;
          goto cleanup;
        }
        if (unique == 0U || columns[source] != columns[unique - 1U]) {
          columns[unique++] = columns[source];
        }
      }
      column_count = unique;
    }
    if (column_count == 0U) {
      status = SS_ERROR_INTEGRITY;
      goto cleanup;
    }
    {
      PetscInt *compacted =
          (PetscInt *)realloc(columns, column_count * sizeof(*columns));
      if (compacted == NULL) {
        status = SS_ERROR_OUT_OF_MEMORY;
        goto cleanup;
      }
      row->columns = compacted;
      columns = NULL;
    }
    row->column_count = column_count;
    for (incidence = 0U; incidence < column_count; ++incidence) {
      if (row->columns[incidence] >= plan->row_start &&
          row->columns[incidence] < plan->row_end) {
        ++row->diagonal_count;
      } else {
        ++row->off_diagonal_count;
      }
    }
    if (nonzeros > UINT64_MAX - (uint64_t)row->column_count ||
        diagonal > UINT64_MAX - (uint64_t)row->diagonal_count ||
        off_diagonal > UINT64_MAX - (uint64_t)row->off_diagonal_count) {
      status = SS_ERROR_OVERFLOW;
      goto cleanup;
    }
    nonzeros += (uint64_t)row->column_count;
    diagonal += (uint64_t)row->diagonal_count;
    off_diagonal += (uint64_t)row->off_diagonal_count;
  }
  if (occurrence_index != occurrence_count) {
    status = SS_ERROR_INTEGRITY;
    goto cleanup;
  }
  plan->summary.local_nonzeros = nonzeros;
  plan->summary.local_diagonal_nonzeros = diagonal;
  plan->summary.local_off_diagonal_nonzeros = off_diagonal;
  plan->summary.contribution_count = contributions;
  status = SS_OK;

cleanup:
  free(columns);
  free(occurrences);
  return status;
}

static SsStatus ss_plan_local_overlap_pairs(const SsAssemblyPlan *plan,
                                            SsRowCellPair **pairs, size_t *pair_count) {
  size_t upper_bound = 0U;
  size_t cell_index;
  size_t count = 0U;
  SsRowCellPair *items = NULL;
  if (pairs == NULL || pair_count == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *pairs = NULL;
  *pair_count = 0U;
  for (cell_index = 0U; cell_index < plan->cell_count; ++cell_index) {
    if (upper_bound > SIZE_MAX - plan->cells[cell_index].closure_count) {
      return SS_ERROR_OVERFLOW;
    }
    upper_bound += plan->cells[cell_index].closure_count;
  }
  if (upper_bound > SIZE_MAX / sizeof(*items)) {
    return SS_ERROR_OVERFLOW;
  }
  if (upper_bound != 0U) {
    items = (SsRowCellPair *)malloc(upper_bound * sizeof(*items));
    if (items == NULL) {
      return SS_ERROR_OUT_OF_MEMORY;
    }
  }
  for (cell_index = 0U; cell_index < plan->cell_count; ++cell_index) {
    const SsCellPlan *cell = &plan->cells[cell_index];
    PetscInt cell_rows[SS_FIELD_MAX_CLOSURE];
    size_t cell_row_count = ss_cell_unique_free_rows(cell, cell_rows);
    size_t cell_row;
    for (cell_row = 0U; cell_row < cell_row_count; ++cell_row) {
      if (cell_rows[cell_row] >= plan->row_start &&
          cell_rows[cell_row] < plan->row_end) {
        if (items == NULL || count >= upper_bound) {
          free(items);
          return SS_ERROR_INTEGRITY;
        }
        items[count++] = (SsRowCellPair){cell_rows[cell_row], cell->stable_cell_id};
      }
    }
  }
  ss_sort(items, count, sizeof(*items), ss_row_cell_pair_compare);
  for (cell_index = 1U; cell_index < count; ++cell_index) {
    if (ss_row_cell_pair_compare(&items[cell_index - 1U], &items[cell_index]) == 0) {
      free(items);
      return SS_ERROR_DUPLICATE;
    }
  }
  *pairs = items;
  *pair_count = count;
  return SS_OK;
}

static SsStatus ss_plan_expected_overlap_pairs(const SsAssemblyPlan *plan,
                                               SsRowCellPair **pairs,
                                               size_t *pair_count) {
  PetscInt *starts = NULL;
  PetscInt *ends = NULL;
  int *send_counts = NULL;
  int *receive_counts = NULL;
  int *send_offsets = NULL;
  int *receive_offsets = NULL;
  size_t owned_pair_count = 0U;
  size_t cell_index;
  size_t send_total = 0U;
  size_t receive_total = 0U;
  size_t *cursors = NULL;
  SsRowCellPair *send_pairs = NULL;
  SsRowCellPair *receive_pairs = NULL;
  MPI_Datatype pair_type = MPI_DATATYPE_NULL;
  SsStatus status = SS_OK;
  if (pairs == NULL || pair_count == NULL || sizeof(SsRowCellPair) > (size_t)INT_MAX) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *pairs = NULL;
  *pair_count = 0U;
  starts = (PetscInt *)calloc((size_t)plan->size, sizeof(*starts));
  ends = (PetscInt *)calloc((size_t)plan->size, sizeof(*ends));
  send_counts = (int *)calloc((size_t)plan->size, sizeof(*send_counts));
  receive_counts = (int *)calloc((size_t)plan->size, sizeof(*receive_counts));
  send_offsets = (int *)calloc((size_t)plan->size, sizeof(*send_offsets));
  receive_offsets = (int *)calloc((size_t)plan->size, sizeof(*receive_offsets));
  cursors = (size_t *)calloc((size_t)plan->size, sizeof(*cursors));
  if (starts == NULL || ends == NULL || send_counts == NULL || receive_counts == NULL ||
      send_offsets == NULL || receive_offsets == NULL || cursors == NULL) {
    status = SS_ERROR_OUT_OF_MEMORY;
    goto cleanup;
  }
  if (MPI_Allgather(&plan->row_start, 1, MPIU_INT, starts, 1, MPIU_INT,
                    plan->communicator) != MPI_SUCCESS ||
      MPI_Allgather(&plan->row_end, 1, MPIU_INT, ends, 1, MPIU_INT,
                    plan->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  for (cell_index = 0U; cell_index < plan->cell_count; ++cell_index) {
    if (plan->cells[cell_index].cell_owned != 0U) {
      PetscInt cell_rows[SS_FIELD_MAX_CLOSURE];
      size_t count = ss_cell_unique_free_rows(&plan->cells[cell_index], cell_rows);
      size_t row;
      if (owned_pair_count > SIZE_MAX - count) {
        status = SS_ERROR_OVERFLOW;
        goto cleanup;
      }
      owned_pair_count += count;
      for (row = 0U; row < count; ++row) {
        int owner = ss_owner_from_ranges(cell_rows[row], starts, ends, plan->size);
        if (owner < 0 || send_counts[owner] == INT_MAX) {
          status = owner < 0 ? SS_ERROR_INTEGRITY : SS_ERROR_OVERFLOW;
          goto cleanup;
        }
        ++send_counts[owner];
      }
    }
  }
  for (cell_index = 0U; cell_index < (size_t)plan->size; ++cell_index) {
    if (send_total > (size_t)INT_MAX - (size_t)send_counts[cell_index]) {
      status = SS_ERROR_OVERFLOW;
      goto cleanup;
    }
    send_offsets[cell_index] = (int)send_total;
    cursors[cell_index] = send_total;
    send_total += (size_t)send_counts[cell_index];
  }
  if (send_total != owned_pair_count || send_total > SIZE_MAX / sizeof(*send_pairs)) {
    status = SS_ERROR_INTEGRITY;
    goto cleanup;
  }
  if (send_total != 0U) {
    send_pairs = (SsRowCellPair *)malloc(send_total * sizeof(*send_pairs));
    if (send_pairs == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
      goto cleanup;
    }
  }
  for (cell_index = 0U; cell_index < plan->cell_count; ++cell_index) {
    const SsCellPlan *cell = &plan->cells[cell_index];
    PetscInt cell_rows[SS_FIELD_MAX_CLOSURE];
    size_t count;
    size_t row;
    if (cell->cell_owned == 0U) {
      continue;
    }
    count = ss_cell_unique_free_rows(cell, cell_rows);
    for (row = 0U; row < count; ++row) {
      int owner = ss_owner_from_ranges(cell_rows[row], starts, ends, plan->size);
      size_t destination;
      if (owner < 0 || send_pairs == NULL ||
          cursors[owner] >= (size_t)send_offsets[owner] + (size_t)send_counts[owner]) {
        status = SS_ERROR_INTEGRITY;
        goto cleanup;
      }
      destination = cursors[owner]++;
      if (destination >= send_total) {
        status = SS_ERROR_INTEGRITY;
        goto cleanup;
      }
      send_pairs[destination] = (SsRowCellPair){cell_rows[row], cell->stable_cell_id};
    }
  }
  if (MPI_Alltoall(send_counts, 1, MPI_INT, receive_counts, 1, MPI_INT,
                   plan->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  for (cell_index = 0U; cell_index < (size_t)plan->size; ++cell_index) {
    if (receive_counts[cell_index] < 0 ||
        receive_total > (size_t)INT_MAX - (size_t)receive_counts[cell_index]) {
      status = SS_ERROR_OVERFLOW;
      goto cleanup;
    }
    receive_offsets[cell_index] = (int)receive_total;
    receive_total += (size_t)receive_counts[cell_index];
  }
  if (receive_total > SIZE_MAX / sizeof(*receive_pairs)) {
    status = SS_ERROR_OVERFLOW;
    goto cleanup;
  }
  if (receive_total != 0U) {
    receive_pairs = (SsRowCellPair *)malloc(receive_total * sizeof(*receive_pairs));
    if (receive_pairs == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
      goto cleanup;
    }
  }
  if (MPI_Type_contiguous((int)sizeof(SsRowCellPair), MPI_BYTE, &pair_type) !=
          MPI_SUCCESS ||
      MPI_Type_commit(&pair_type) != MPI_SUCCESS ||
      MPI_Alltoallv(send_pairs, send_counts, send_offsets, pair_type, receive_pairs,
                    receive_counts, receive_offsets, pair_type,
                    plan->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  ss_sort(receive_pairs, receive_total, sizeof(*receive_pairs),
          ss_row_cell_pair_compare);
  for (cell_index = 1U; cell_index < receive_total; ++cell_index) {
    if (ss_row_cell_pair_compare(&receive_pairs[cell_index - 1U],
                                 &receive_pairs[cell_index]) == 0) {
      status = SS_ERROR_DUPLICATE;
      goto cleanup;
    }
  }
  *pairs = receive_pairs;
  *pair_count = receive_total;
  receive_pairs = NULL;

cleanup:
  if (pair_type != MPI_DATATYPE_NULL) {
    (void)MPI_Type_free(&pair_type);
  }
  free(starts);
  free(ends);
  free(send_counts);
  free(receive_counts);
  free(send_offsets);
  free(receive_offsets);
  free(cursors);
  free(send_pairs);
  free(receive_pairs);
  return status;
}

static SsStatus ss_plan_audit_overlap(const SsAssemblyPlan *plan) {
  SsRowCellPair *local = NULL;
  SsRowCellPair *expected = NULL;
  size_t local_count = 0U;
  size_t expected_count = 0U;
  SsStatus status = ss_plan_local_overlap_pairs(plan, &local, &local_count);
  status = ss_assembly_collective_status_private(plan->communicator, status);
  if (status == SS_OK) {
    status = ss_plan_expected_overlap_pairs(plan, &expected, &expected_count);
  }
  status = ss_assembly_collective_status_private(plan->communicator, status);
  if (status == SS_OK && local_count != expected_count) {
    status = SS_ERROR_INTEGRITY;
  }
  if (status == SS_OK) {
    size_t pair;
    for (pair = 0U; pair < local_count; ++pair) {
      if (local[pair].row != expected[pair].row ||
          local[pair].cell != expected[pair].cell) {
        status = SS_ERROR_INTEGRITY;
        break;
      }
    }
  }
  free(local);
  free(expected);
  return ss_assembly_collective_status_private(plan->communicator, status);
}

static unsigned int ss_hex_nibble(char value) {
  if (value >= '0' && value <= '9') {
    return (unsigned int)(value - '0');
  }
  if (value >= 'a' && value <= 'f') {
    return (unsigned int)(value - 'a' + 10);
  }
  return 0U;
}

static uint64_t ss_hex_u64(const char *hex) {
  uint64_t value = 0U;
  size_t index;
  for (index = 0U; index < 16U; ++index) {
    value = (value << 4U) | (uint64_t)ss_hex_nibble(hex[index]);
  }
  return value;
}

static SsStatus ss_plan_digest(SsAssemblyPlan *plan) {
  uint64_t local_xor[4] = {0U, 0U, 0U, 0U};
  uint64_t global_xor[4] = {0U, 0U, 0U, 0U};
  uint64_t local_sum[4] = {0U, 0U, 0U, 0U};
  uint64_t global_sum[4] = {0U, 0U, 0U, 0U};
  size_t local_row;
  char projection[512];
  char final_projection[768];
  char digest[SS_SHA256_HEX_CAPACITY] = {0};
  int written;
  for (local_row = 0U; local_row < plan->row_count; ++local_row) {
    const SsOwnedRowPlan *row = &plan->rows[local_row];
    size_t projection_size;
    size_t column;
    unsigned char *serialized;
    size_t serialized_size;
    if (row->column_count > (SIZE_MAX - 32U) / sizeof(uint64_t) ||
        row->cell_count > (SIZE_MAX - 32U - row->column_count * sizeof(uint64_t)) /
                              sizeof(uint64_t)) {
      return SS_ERROR_OVERFLOW;
    }
    serialized_size = 32U + (row->column_count + row->cell_count) * sizeof(uint64_t);
    serialized = (unsigned char *)calloc(serialized_size, 1U);
    if (serialized == NULL) {
      return SS_ERROR_OUT_OF_MEMORY;
    }
    memcpy(serialized, &row->global_row, sizeof(row->global_row));
    memcpy(serialized + 8U, &row->field_entity_id, sizeof(row->field_entity_id));
    memcpy(serialized + 16U, &row->component, sizeof(row->component));
    memcpy(serialized + 20U, &row->field_entity_slot, sizeof(row->field_entity_slot));
    for (column = 0U; column < row->column_count; ++column) {
      const uint64_t value = (uint64_t)row->columns[column];
      memcpy(serialized + 32U + column * sizeof(value), &value, sizeof(value));
    }
    for (column = 0U; column < row->cell_count; ++column) {
      const uint64_t value =
          plan->cells[row->cell_slots[column].cell_index].stable_cell_id;
      memcpy(serialized + 32U + (row->column_count + column) * sizeof(value), &value,
             sizeof(value));
    }
    if (ss_sha256_bytes(serialized, serialized_size, digest) != SS_OK) {
      free(serialized);
      return SS_ERROR_INTERNAL;
    }
    free(serialized);
    projection_size = strlen(digest);
    if (projection_size != 64U) {
      return SS_ERROR_INTERNAL;
    }
    for (column = 0U; column < 4U; ++column) {
      uint64_t word = ss_hex_u64(digest + column * 16U);
      local_xor[column] ^= word;
      local_sum[column] += word;
    }
  }
  if (MPI_Allreduce(local_xor, global_xor, 4, MPI_UINT64_T, MPI_BXOR,
                    plan->communicator) != MPI_SUCCESS ||
      MPI_Allreduce(local_sum, global_sum, 4, MPI_UINT64_T, MPI_SUM,
                    plan->communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  written =
      snprintf(projection, sizeof(projection),
               "rows=%llu\nxor=%016llx%016llx%016llx%016llx\n"
               "sum=%016llx%016llx%016llx%016llx\n",
               (unsigned long long)plan->summary.global_rows,
               (unsigned long long)global_xor[0], (unsigned long long)global_xor[1],
               (unsigned long long)global_xor[2], (unsigned long long)global_xor[3],
               (unsigned long long)global_sum[0], (unsigned long long)global_sum[1],
               (unsigned long long)global_sum[2], (unsigned long long)global_sum[3]);
  if (written < 0 || (size_t)written >= sizeof(projection) ||
      ss_sha256_bytes((const unsigned char *)projection, (size_t)written, digest) !=
          SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  written = snprintf(final_projection, sizeof(final_projection), "sha256:%s", digest);
  if (written != 71) {
    return SS_ERROR_INTERNAL;
  }
  memcpy(plan->summary.pattern_digest, final_projection,
         sizeof(plan->summary.pattern_digest));
  return SS_OK;
}

static SsStatus ss_plan_finish_summary(SsAssemblyPlan *plan) {
  uint64_t bytes = (uint64_t)sizeof(*plan);
  uint64_t owned = 0U;
  uint64_t halo = 0U;
  size_t cell;
  size_t row;
  if (plan->row_count > UINT64_MAX / sizeof(*plan->rows)) {
    return SS_ERROR_OVERFLOW;
  }
  /* Cell plans are a read-only view owned by the field space. */
  bytes += (uint64_t)plan->row_count * (uint64_t)sizeof(*plan->rows);
  for (cell = 0U; cell < plan->cell_count; ++cell) {
    if (plan->cells[cell].cell_owned != 0U) {
      ++owned;
    } else {
      ++halo;
    }
    if (plan->cells[cell].closure_count > plan->summary.maximum_closure_count) {
      plan->summary.maximum_closure_count = plan->cells[cell].closure_count;
    }
  }
  for (row = 0U; row < plan->row_count; ++row) {
    const SsOwnedRowPlan *item = &plan->rows[row];
    uint64_t row_bytes;
    if (item->column_count > UINT64_MAX / sizeof(*item->columns) ||
        item->cell_count > UINT64_MAX / sizeof(*item->cell_slots)) {
      return SS_ERROR_OVERFLOW;
    }
    row_bytes = (uint64_t)item->column_count * (uint64_t)sizeof(*item->columns);
    if (bytes > UINT64_MAX - row_bytes) {
      return SS_ERROR_OVERFLOW;
    }
    bytes += row_bytes;
    row_bytes = (uint64_t)item->cell_count * (uint64_t)sizeof(*item->cell_slots);
    if (bytes > UINT64_MAX - row_bytes) {
      return SS_ERROR_OVERFLOW;
    }
    bytes += row_bytes;
  }
  plan->summary.local_incident_cells = (uint64_t)plan->cell_count;
  plan->summary.local_owned_cells = owned;
  plan->summary.local_halo_cells = halo;
  plan->summary.plan_bytes = bytes;
  plan->summary.overlap_depth = plan->size > 1 ? 1U : 0U;
  plan->summary.pattern_generation = 1U;
  return ss_plan_digest(plan);
}

static SsStatus ss_assembly_plan_create_private(MPI_Comm communicator,
                                                const SsFieldSpace *field_space,
                                                int matrix_pattern,
                                                SsAssemblyPlan **plan,
                                                SsErrorRecord *error) {
  SsAssemblyPlan *created = NULL;
  size_t cell_count = 0U;
  SsStatus status = SS_OK;
  if (plan != NULL) {
    *plan = NULL;
  }
  if (plan == NULL || field_space == NULL) {
    ss_assembly_error_private(error, SS_ERROR_INVALID_ARGUMENT, "ASM_PLAN_ARGUMENT", 0U,
                              0);
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (error != NULL) {
    memset(error, 0, sizeof(*error));
  }
  created = (SsAssemblyPlan *)calloc(1U, sizeof(*created));
  if (created == NULL) {
    status = SS_ERROR_OUT_OF_MEMORY;
  }
  status = ss_assembly_collective_status_private(communicator, status);
  if (status != SS_OK || created == NULL) {
    if (status == SS_OK) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
    free(created);
    ss_assembly_error_private(error, status, "ASM_PLAN_ALLOCATE", 0U, 0);
    return status;
  }
  created->communicator = communicator;
  created->field_space = field_space;
  created->matrix_pattern = matrix_pattern;
  if (MPI_Comm_rank(communicator, &created->rank) != MPI_SUCCESS ||
      MPI_Comm_size(communicator, &created->size) != MPI_SUCCESS || created->size < 1) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    status = ss_field_space_summary(field_space, &created->field_summary);
  }
  if (status == SS_OK) {
    status = ss_field_space_incident_cell_plan_view(field_space, &created->cells,
                                                    &cell_count);
  }
  status = ss_assembly_collective_status_private(communicator, status);
  if (status != SS_OK || (cell_count != 0U && created->cells == NULL)) {
    if (status == SS_OK) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
    ss_assembly_error_private(error, status, "ASM_PLAN_INPUT", 0U, created->rank);
    (void)ss_assembly_plan_destroy(&created);
    return status;
  }
  created->cell_count = cell_count;
  if (status == SS_OK && created->cell_count > 1U) {
    size_t index;
    for (index = 1U; index < created->cell_count; ++index) {
      if (created->cells[index - 1U].stable_cell_id >=
          created->cells[index].stable_cell_id) {
        status = SS_ERROR_INTEGRITY;
        break;
      }
    }
  }
  if (status == SS_OK) {
    status = ss_plan_row_ownership(created);
  }
  if (status == SS_OK) {
    status = ss_plan_row_metadata(created);
  }
  if (status == SS_OK) {
    status = ss_plan_build_rows(created, matrix_pattern);
  }
  status = ss_assembly_collective_status_private(communicator, status);
  if (status == SS_OK) {
    status = ss_plan_audit_overlap(created);
  }
  if (status == SS_OK) {
    status = ss_plan_finish_summary(created);
  }
  status = ss_assembly_collective_status_private(communicator, status);
  if (status != SS_OK) {
    ss_assembly_error_private(error, status, "ASM_PLAN_INCOMPLETE_OVERLAP", 0U,
                              created->rank);
    (void)ss_assembly_plan_destroy(&created);
    return status;
  }
  *plan = created;
  return SS_OK;
}

SsStatus ss_assembly_plan_create(MPI_Comm communicator, const SsFieldSpace *field_space,
                                 SsAssemblyPlan **plan, SsErrorRecord *error) {
  return ss_assembly_plan_create_private(communicator, field_space, 1, plan, error);
}

SsStatus ss_assembly_plan_create_matrix_free(MPI_Comm communicator,
                                             const SsFieldSpace *field_space,
                                             SsAssemblyPlan **plan,
                                             SsErrorRecord *error) {
  return ss_assembly_plan_create_private(communicator, field_space, 0, plan, error);
}

SsStatus ss_assembly_plan_summary(const SsAssemblyPlan *plan,
                                  SsAssemblyPlanSummary *summary) {
  if (plan == NULL || summary == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *summary = plan->summary;
  return SS_OK;
}

SsStatus ss_assembly_plan_row(const SsAssemblyPlan *plan, size_t local_row,
                              SsAssemblyRowSummary *summary, PetscInt *columns,
                              size_t column_capacity, size_t *column_count,
                              SsEntityId *incident_cells, size_t cell_capacity,
                              size_t *cell_count) {
  const SsOwnedRowPlan *row;
  size_t cell;
  if (plan == NULL || summary == NULL || column_count == NULL || cell_count == NULL ||
      local_row >= plan->row_count || (columns == NULL && column_capacity != 0U) ||
      (incident_cells == NULL && cell_capacity != 0U)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  row = &plan->rows[local_row];
  *summary = (SsAssemblyRowSummary){
      row->global_row,        row->component,    row->field_entity_id,
      row->field_entity_slot, row->column_count, row->cell_count,
  };
  *column_count = row->column_count;
  *cell_count = row->cell_count;
  if (columns != NULL) {
    if (column_capacity < row->column_count) {
      return SS_ERROR_LIMIT;
    }
    memcpy(columns, row->columns, row->column_count * sizeof(*columns));
  }
  if (incident_cells != NULL) {
    if (cell_capacity < row->cell_count) {
      return SS_ERROR_LIMIT;
    }
    for (cell = 0U; cell < row->cell_count; ++cell) {
      incident_cells[cell] =
          plan->cells[row->cell_slots[cell].cell_index].stable_cell_id;
    }
  }
  return SS_OK;
}

SsStatus ss_assembly_plan_create_vector(const SsAssemblyPlan *plan, Vec *vector) {
  PetscInt local_rows;
  PetscInt global_rows;
  SsStatus status = SS_OK;
  if (plan == NULL || vector == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *vector = NULL;
  local_rows = (PetscInt)plan->summary.local_owned_rows;
  global_rows = (PetscInt)plan->summary.global_rows;
  if (VecCreate(plan->communicator, vector) != PETSC_SUCCESS ||
      VecSetSizes(*vector, local_rows, global_rows) != PETSC_SUCCESS ||
      VecSetFromOptions(*vector) != PETSC_SUCCESS ||
      PetscObjectSetName((PetscObject)*vector, "assembly-owned-residual") !=
          PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_assembly_collective_status_private(plan->communicator, status);
  if (status != SS_OK) {
    (void)VecDestroy(vector);
  }
  return status;
}

SsStatus ss_assembly_plan_create_matrix(const SsAssemblyPlan *plan, Mat *matrix) {
  PetscInt *diagonal = NULL;
  PetscInt *off_diagonal = NULL;
  PetscScalar *zeros = NULL;
  size_t row;
  size_t maximum_columns = 0U;
  SsStatus status = SS_OK;
  if (plan == NULL || matrix == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *matrix = NULL;
  if (plan->matrix_pattern == 0) {
    return SS_ERROR_INVALID_STATE;
  }
  if (plan->row_count != 0U) {
    diagonal = (PetscInt *)calloc(plan->row_count, sizeof(*diagonal));
    off_diagonal = (PetscInt *)calloc(plan->row_count, sizeof(*off_diagonal));
    if (diagonal == NULL || off_diagonal == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  for (row = 0U; status == SS_OK && row < plan->row_count; ++row) {
    if (plan->rows[row].diagonal_count > (size_t)PETSC_MAX_INT ||
        plan->rows[row].off_diagonal_count > (size_t)PETSC_MAX_INT) {
      status = SS_ERROR_OVERFLOW;
      break;
    }
    diagonal[row] = (PetscInt)plan->rows[row].diagonal_count;
    off_diagonal[row] = (PetscInt)plan->rows[row].off_diagonal_count;
    if (plan->rows[row].column_count > maximum_columns) {
      maximum_columns = plan->rows[row].column_count;
    }
  }
  if (status == SS_OK && maximum_columns != 0U) {
    zeros = (PetscScalar *)calloc(maximum_columns, sizeof(*zeros));
    if (zeros == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_assembly_collective_status_private(plan->communicator, status);
  if (status == SS_OK &&
      (MatCreate(plan->communicator, matrix) != PETSC_SUCCESS ||
       MatSetSizes(*matrix, (PetscInt)plan->summary.local_owned_rows,
                   (PetscInt)plan->summary.local_owned_rows,
                   (PetscInt)plan->summary.global_rows,
                   (PetscInt)plan->summary.global_rows) != PETSC_SUCCESS ||
       MatSetFromOptions(*matrix) != PETSC_SUCCESS ||
       (plan->size == 1 ? MatSeqAIJSetPreallocation(*matrix, 0, diagonal)
                        : MatMPIAIJSetPreallocation(*matrix, 0, diagonal, 0,
                                                    off_diagonal)) != PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK &&
      (MatSetOption(*matrix, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE) !=
           PETSC_SUCCESS ||
       MatSetOption(*matrix, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE) != PETSC_SUCCESS ||
       PetscObjectSetName((PetscObject)*matrix, "assembly-owned-jacobian") !=
           PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  for (row = 0U; status == SS_OK && row < plan->row_count; ++row) {
    const SsOwnedRowPlan *item = &plan->rows[row];
    if (MatSetValues(*matrix, 1, &item->global_row, (PetscInt)item->column_count,
                     item->columns, zeros, INSERT_VALUES) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  if (status == SS_OK &&
      (MatAssemblyBegin(*matrix, MAT_FINAL_ASSEMBLY) != PETSC_SUCCESS ||
       MatAssemblyEnd(*matrix, MAT_FINAL_ASSEMBLY) != PETSC_SUCCESS ||
       MatZeroEntries(*matrix) != PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_assembly_collective_status_private(plan->communicator, status);
  free(diagonal);
  free(off_diagonal);
  free(zeros);
  if (status != SS_OK) {
    (void)MatDestroy(matrix);
  }
  return status;
}

SsStatus ss_assembly_plan_destroy(SsAssemblyPlan **plan) {
  size_t row;
  if (plan == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (*plan == NULL) {
    return SS_OK;
  }
  for (row = 0U; row < (*plan)->row_count; ++row) {
    free((*plan)->rows[row].columns);
  }
  free((*plan)->row_cell_slots);
  free((*plan)->rows);
  free(*plan);
  *plan = NULL;
  return SS_OK;
}
