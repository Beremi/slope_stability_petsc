/* Plan the distributed matrix-free input and action exchanges once per shell.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "operator_internal.h"

#include <stdlib.h>

static PetscErrorCode ss_matrix_free_exchange_destroy(void **context) {
  SsMatrixFreeExchange *exchange = *context;
  if (exchange == NULL) {
    return PETSC_SUCCESS;
  }
  const PetscErrorCode input_status = PetscSFDestroy(&exchange->input);
  const PetscErrorCode action_status = PetscSFDestroy(&exchange->action);
  free(exchange);
  *context = NULL;
  return input_status == PETSC_SUCCESS ? action_status : input_status;
}

static SsStatus ss_matrix_free_input_graph(const SsOperator *op, PetscLayout layout,
                                           SsMatrixFreeExchange *exchange) {
  const SsAssemblyPlan *plan = op->plan;
  const size_t ghost_count = op->gather_count - plan->row_count;
  const size_t owned_end = exchange->owned_offset + plan->row_count;
  PetscInt *edges = NULL;
  size_t index = 0U;
  SsStatus status = SS_OK;
  if (ghost_count > SIZE_MAX / (2U * sizeof(*edges))) {
    status = SS_ERROR_OVERFLOW;
  } else {
    edges = calloc(ghost_count == 0U ? 1U : 2U * ghost_count, sizeof(*edges));
    if (edges == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_assembly_collective_status_private(plan->communicator, status);
  if (status == SS_OK) {
    /* Only owned halo cells need remote coefficients. Gather slots can
     * have holes, so each requested coefficient carries its leaf index. */
    for (size_t i = exchange->interior_cells; i < exchange->active_cells; ++i) {
      const size_t cell = exchange->cells[i];
      const size_t *packed = op->packed_slots + cell * op->maximum_closure;
      for (size_t slot = 0U; slot < plan->cells[cell].closure_count; ++slot) {
        if (packed[slot] < exchange->owned_offset) {
          edges[packed[slot]] = 1;
        } else if (packed[slot] != SIZE_MAX && packed[slot] >= owned_end) {
          edges[packed[slot] - plan->row_count] = 1;
        }
      }
    }
    for (size_t ghost = 0U; ghost < ghost_count; ++ghost) {
      if (edges[ghost] != 0) {
        const size_t slot =
            ghost < exchange->owned_offset ? ghost : ghost + plan->row_count;
        edges[index] = (PetscInt)slot;
        edges[ghost_count + index++] = op->gather_indices[slot];
      }
    }
    if (PetscSFCreate(plan->communicator, &exchange->input) != PETSC_SUCCESS ||
        PetscSFSetGraphLayout(exchange->input, layout, (PetscInt)index, edges,
                              PETSC_COPY_VALUES,
                              edges + ghost_count) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  free(edges);
  status = ss_assembly_collective_status_private(plan->communicator, status);
  if (status == SS_OK && PetscSFSetUp(exchange->input) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  return ss_assembly_collective_status_private(plan->communicator, status);
}

static void ss_matrix_free_cell_partition(const SsOperator *op,
                                          SsMatrixFreeExchange *exchange) {
  const SsAssemblyPlan *plan = op->plan;
  const size_t owned_end = exchange->owned_offset + plan->row_count;
  size_t halo = plan->cell_count;
  for (size_t index = 0U; index < plan->cell_count; ++index) {
    const SsCellPlan *cell = &plan->cells[index];
    const size_t *packed = op->packed_slots + index * op->maximum_closure;
    int active = 0;
    int needs_ghosts = 0;
    if (cell->cell_owned == 0U) {
      continue;
    }
    for (size_t slot = 0U; slot < cell->closure_count; ++slot) {
      if (packed[slot] == SIZE_MAX) {
        continue;
      }
      active = 1;
      if (packed[slot] < exchange->owned_offset || packed[slot] >= owned_end) {
        needs_ghosts = 1;
        break;
      }
    }
    if (needs_ghosts) {
      exchange->cells[--halo] = index;
    } else if (active) {
      exchange->cells[exchange->interior_cells++] = index;
    }
  }
  /* Halo entries were filled from the end. Put them in ascending cell order. */
  const size_t halo_count = plan->cell_count - halo;
  exchange->active_cells = exchange->interior_cells + halo_count;
  for (size_t i = 0U; i < halo_count / 2U; ++i) {
    const size_t last = plan->cell_count - i - 1U;
    const size_t index = exchange->cells[halo + i];
    exchange->cells[halo + i] = exchange->cells[last];
    exchange->cells[last] = index;
  }
  memmove(exchange->cells + exchange->interior_cells, exchange->cells + halo,
          halo_count * sizeof(*exchange->cells));
}

static size_t ss_matrix_free_action_edges(const SsOperator *op,
                                          const SsMatrixFreeExchange *exchange,
                                          PetscInt *global_rows, PetscInt *send_slots,
                                          SsEntityId *cell_ids) {
  const SsAssemblyPlan *plan = op->plan;
  size_t count = 0U;
  for (size_t index = exchange->interior_cells; index < exchange->active_cells;
       ++index) {
    const size_t cell_index = exchange->cells[index];
    const SsCellPlan *cell = &plan->cells[cell_index];
    for (size_t slot = 0U; slot < cell->closure_count; ++slot) {
      const PetscInt row = cell->global_indices[slot];
      if (cell->constrained[slot] != 0U ||
          (row >= plan->row_start && row < plan->row_end)) {
        continue;
      }
      if (global_rows != NULL) {
        global_rows[count] = row;
        send_slots[count] = (PetscInt)(cell_index * op->maximum_closure + slot);
        cell_ids[count] = cell->stable_cell_id;
      }
      ++count;
    }
  }
  return count;
}

static SsStatus ss_matrix_free_match_actions(const SsOperator *op,
                                             SsMatrixFreeExchange *exchange,
                                             const PetscInt *degree,
                                             const SsEntityId *received_ids,
                                             PetscInt *received_slots) {
  const SsAssemblyPlan *plan = op->plan;
  size_t *rows = exchange->cells + plan->cell_count;
  size_t next = 0U;
  exchange->rows = rows;
  for (size_t index = 0U; index < plan->row_count; ++index) {
    const SsOwnedRowPlan *row = &plan->rows[index];
    size_t ghost_cells = 0U;
    /* Each ghost cell incident on this row must supply its one contribution. */
    for (size_t cell = 0U; cell < row->cell_count; ++cell) {
      ghost_cells += plan->cells[row->cell_slots[cell].cell_index].cell_owned == 0U;
    }
    if ((size_t)degree[index] != ghost_cells) {
      return SS_ERROR_INTEGRITY;
    }
    if (ghost_cells == 0U) {
      rows[exchange->local_rows++] = index;
    }
    for (size_t edge = 0U; edge < ghost_cells; ++edge, ++next) {
      size_t slot = SIZE_MAX;
      for (size_t cell = 0U; cell < row->cell_count; ++cell) {
        const SsRowCellSlot occurrence = row->cell_slots[cell];
        const SsCellPlan *record = &plan->cells[occurrence.cell_index];
        if (record->cell_owned == 0U && record->stable_cell_id == received_ids[next]) {
          slot = occurrence.cell_index * op->maximum_closure + occurrence.row_slot;
          break;
        }
      }
      if (slot == SIZE_MAX || op->matrix_free_element_action[slot] != 0.0) {
        return SS_ERROR_INTEGRITY;
      }
      op->matrix_free_element_action[slot] = 1.0;
      received_slots[next] = (PetscInt)slot;
    }
  }
  size_t remote_row = exchange->local_rows;
  for (size_t index = 0U; index < plan->row_count; ++index) {
    if (degree[index] != 0) {
      rows[remote_row++] = index;
    }
  }
  return SS_OK;
}

static SsStatus ss_matrix_free_action_graph(SsOperator *op, PetscLayout layout,
                                            SsMatrixFreeExchange *exchange) {
  const SsAssemblyPlan *plan = op->plan;
  const size_t send_count = ss_matrix_free_action_edges(op, exchange, NULL, NULL, NULL);
  const size_t send_size = send_count == 0U ? 1U : send_count;
  const size_t action_count = plan->cell_count * op->maximum_closure;
  const PetscInt *degree = NULL;
  const PetscSFNode *row_owners = NULL;
  PetscSF by_row = NULL;
  PetscSF by_contribution = NULL;
  PetscInt receive_count = 0;
  PetscInt *send_slots = malloc(send_size * sizeof(*send_slots));
  PetscInt *global_rows = malloc(send_size * sizeof(*global_rows));
  SsEntityId *send_ids = malloc(send_size * sizeof(*send_ids));
  PetscSFNode *destinations = malloc(send_size * sizeof(*destinations));
  PetscInt *received_slots = NULL;
  SsEntityId *received_ids = NULL;
  SsStatus status = send_slots == NULL || global_rows == NULL || send_ids == NULL ||
                            destinations == NULL
                        ? SS_ERROR_OUT_OF_MEMORY
                        : SS_OK;
  status = ss_assembly_collective_status_private(plan->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  (void)ss_matrix_free_action_edges(op, exchange, global_rows, send_slots, send_ids);
  if (PetscSFCreate(plan->communicator, &by_row) != PETSC_SUCCESS ||
      PetscSFSetRankOrder(by_row, PETSC_FALSE) != PETSC_SUCCESS ||
      PetscSFSetGraphLayout(by_row, layout, (PetscInt)send_count, NULL,
                            PETSC_COPY_VALUES, global_rows) != PETSC_SUCCESS ||
      PetscSFGetGraph(by_row, NULL, NULL, NULL, &row_owners) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_assembly_collective_status_private(plan->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (PetscSFGetMultiSF(by_row, &by_contribution) != PETSC_SUCCESS ||
      PetscSFGetGraph(by_contribution, &receive_count, NULL, NULL, NULL) !=
          PETSC_SUCCESS ||
      PetscSFComputeDegreeBegin(by_row, &degree) != PETSC_SUCCESS ||
      PetscSFComputeDegreeEnd(by_row, &degree) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  } else if (receive_count < 0 || (size_t)receive_count > action_count) {
    status = SS_ERROR_INTEGRITY;
  } else {
    const size_t receive_size = receive_count == 0 ? 1U : (size_t)receive_count;
    received_slots = malloc(receive_size * sizeof(*received_slots));
    received_ids = malloc(receive_size * sizeof(*received_ids));
    if (received_slots == NULL || received_ids == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_assembly_collective_status_private(plan->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  /* Resolve (global row, stable cell ID) to that cell's position in the row
   * owner's action buffer. The final graph gives each receive slot one sender,
   * so the row owner can add contributions in the original plan order. */
  if (PetscSFGatherBegin(by_row, MPI_UINT64_T, send_ids, received_ids) !=
          PETSC_SUCCESS ||
      PetscSFGatherEnd(by_row, MPI_UINT64_T, send_ids, received_ids) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  } else {
    status = ss_matrix_free_match_actions(op, exchange, degree, received_ids,
                                          received_slots);
    memset(op->matrix_free_element_action, 0, action_count * sizeof(double));
  }
  status = ss_assembly_collective_status_private(plan->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (PetscSFScatterBegin(by_row, MPIU_INT, received_slots, global_rows) !=
          PETSC_SUCCESS ||
      PetscSFScatterEnd(by_row, MPIU_INT, received_slots, global_rows) !=
          PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  } else {
    for (size_t edge = 0U; edge < send_count; ++edge) {
      destinations[edge] = (PetscSFNode){row_owners[edge].rank, global_rows[edge]};
    }
    if (PetscSFCreate(plan->communicator, &exchange->action) != PETSC_SUCCESS ||
        PetscSFSetGraph(exchange->action, (PetscInt)action_count, (PetscInt)send_count,
                        send_slots, PETSC_COPY_VALUES, destinations,
                        PETSC_COPY_VALUES) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  status = ss_assembly_collective_status_private(plan->communicator, status);
  if (status == SS_OK && PetscSFSetUp(exchange->action) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }

cleanup:
  if (PetscSFDestroy(&by_row) != PETSC_SUCCESS && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  free(received_ids);
  free(received_slots);
  free(destinations);
  free(send_ids);
  free(global_rows);
  free(send_slots);
  return ss_assembly_collective_status_private(plan->communicator, status);
}

SsStatus ss_operator_matrix_free_prepare_exchange(SsOperator *op) {
  static const char name[] = "SsMatrixFreeExchange";
  const SsAssemblyPlan *plan = op->plan;
  const uint64_t cap = op->spec.memory_cap_bytes == 0U
                           ? SS_ASSEMBLY_DEFAULT_MEMORY_CAP
                           : (uint64_t)op->spec.memory_cap_bytes;
  PetscContainer owner = NULL;
  PetscLayout layout = NULL;
  SsMatrixFreeExchange *exchange = NULL;
  int container_owns_exchange = 0;
  size_t bytes = 0U;
  SsStatus status = SS_OK;
  if (plan->size == 1 || op->matrix_free_exchange != NULL) {
    return SS_OK;
  }
  /* A tight workspace cap or an action buffer beyond the PetscSF index range
   * can still use the row-owned path with the operator's original scatter. */
  if (plan->cell_count > SIZE_MAX - plan->row_count ||
      plan->cell_count + plan->row_count >
          (SIZE_MAX - sizeof(*exchange)) / sizeof(*exchange->cells) ||
      plan->cell_count * op->maximum_closure > (size_t)PETSC_MAX_INT ||
      plan->cell_count * op->maximum_closure > SIZE_MAX / sizeof(PetscSFNode)) {
    status = SS_ERROR_RESOURCE;
  } else {
    bytes = sizeof(*exchange) +
            (plan->cell_count + plan->row_count) * sizeof(*exchange->cells);
    if (op->diagnostics.admitted_peak_bytes > cap ||
        bytes > cap - op->diagnostics.admitted_peak_bytes) {
      status = SS_ERROR_RESOURCE;
    }
  }
  status = ss_assembly_collective_status_private(plan->communicator, status);
  if (status != SS_OK) {
    return status == SS_ERROR_RESOURCE ? SS_OK : status;
  }
  exchange = calloc(1U, bytes);
  if (exchange == NULL) {
    status = SS_ERROR_OUT_OF_MEMORY;
  } else {
    while (exchange->owned_offset < op->gather_count &&
           op->gather_indices[exchange->owned_offset] < plan->row_start) {
      ++exchange->owned_offset;
    }
    if (plan->row_count > op->gather_count - exchange->owned_offset ||
        (plan->row_count != 0U &&
         (op->gather_indices[exchange->owned_offset] != plan->row_start ||
          op->gather_indices[exchange->owned_offset + plan->row_count - 1U] !=
              plan->row_end - 1))) {
      status = SS_ERROR_INTEGRITY;
    } else if (VecGetLayout(op->staging_residual, &layout) != PETSC_SUCCESS ||
               PetscContainerCreate(plan->communicator, &owner) != PETSC_SUCCESS ||
               PetscContainerSetCtxDestroy(owner, ss_matrix_free_exchange_destroy) !=
                   PETSC_SUCCESS ||
               PetscContainerSetPointer(owner, exchange) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    } else {
      container_owns_exchange = 1;
    }
  }
  status = ss_assembly_collective_status_private(plan->communicator, status);
  if (status == SS_OK) {
    ss_matrix_free_cell_partition(op, exchange);
    status = ss_matrix_free_input_graph(op, layout, exchange);
  }
  if (status == SS_OK) {
    status = ss_matrix_free_action_graph(op, layout, exchange);
  }
  if (status == SS_OK && PetscObjectCompose((PetscObject)op->scatter, name,
                                            (PetscObject)owner) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_assembly_collective_status_private(plan->communicator, status);
  if (status == SS_OK) {
    /* The original scatter owns the exchange plan for the operator lifetime. */
    op->matrix_free_exchange = exchange;
    op->diagnostics.matrix_free_bytes += bytes;
    op->diagnostics.admitted_peak_bytes += bytes;
  } else {
    if (!container_owns_exchange) {
      void *context = exchange;
      (void)ss_matrix_free_exchange_destroy(&context);
    }
    (void)PetscObjectCompose((PetscObject)op->scatter, name, NULL);
  }
  (void)PetscContainerDestroy(&owner);
  return status;
}
