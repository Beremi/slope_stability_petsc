/* Complete per-iteration solver histories for the result tables.
 *
 * Every nonlinear solve record already carries one entry per Newton
 * iteration with its Krylov solves, line-search trials and PMG setup.  This
 * file turns those records into relational history rows.  During an analysis
 * the rows are appended to a rank-zero spool file (history rows are replicated,
 * so no rank holds them in memory); publication replays the spool attempt by
 * attempt behind each attempt row in bounded batches.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "solver_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Rows broadcast and appended together during spool replay. */
#define SS_SOLVER_HISTORY_REPLAY_BATCH 1024U

static double ss_solver_history_finite(PetscReal value) {
  return isfinite((double)value) ? (double)value : 0.0;
}

static SsResultHistoryRow ss_solver_history_row(SsResultHistoryKind kind,
                                                uint64_t attempt_id, uint64_t ordinal,
                                                uint32_t role) {
  SsResultHistoryRow row;
  memset(&row, 0, sizeof(row));
  row.kind = kind;
  row.attempt_id = attempt_id;
  row.ordinal = ordinal;
  row.role = role;
  row.complete = 1;
  return row;
}

static SsStatus ss_solver_history_emit(SsSolverHistoryEmitter *emitter,
                                       const SsResultHistoryRow *row,
                                       uint32_t parent_index, uint32_t *index) {
  SsSolverHistoryEntry entry;
  SsStatus status;
  memset(&entry, 0, sizeof(entry));
  entry.attempt_id = row->attempt_id;
  entry.parent_index = parent_index;
  entry.row = *row;
  status = emitter->emit(emitter->context, &entry);
  if (status == SS_OK && index != NULL) {
    *index = emitter->entry_count;
  }
  if (status == SS_OK) {
    ++emitter->entry_count;
  }
  return status;
}

SsStatus ss_solver_history_detail(SsSolverHistoryEmitter *emitter,
                                  const SsResultHistoryRow *row) {
  return ss_solver_history_emit(emitter, row, SS_SOLVER_HISTORY_PARENT_GROUP, NULL);
}

static SsStatus ss_solver_history_linear(SsSolverHistoryEmitter *emitter,
                                         const SsLinearSolveRecord *solve,
                                         uint64_t attempt_id, uint64_t ordinal,
                                         uint32_t role, uint32_t parent) {
  SsResultHistoryRow row;
  if (solve->solve_count == 0U) {
    return SS_OK;
  }
  row = ss_solver_history_row(SS_RESULT_HISTORY_LINEAR, attempt_id, ordinal, role);
  row.control = ss_solver_history_finite(solve->effective_rtol);
  row.residual = ss_solver_history_finite(solve->true_residual_norm);
  row.status = (int32_t)solve->status;
  row.reason = (int32_t)solve->reason;
  row.iterations = solve->iterations < 0 ? 0U : (uint64_t)solve->iterations;
  row.tolerance = ss_solver_history_finite(solve->acceptance_threshold);
  row.reference = ss_solver_history_finite(solve->rhs_norm);
  row.step = ss_solver_history_finite(solve->effective_rtol);
  row.auxiliary = ss_solver_history_finite(solve->reported_norm);
  return ss_solver_history_emit(emitter, &row, parent, NULL);
}

static SsStatus ss_solver_history_iteration(SsSolverHistoryEmitter *emitter,
                                            const SsNonlinearIterationRecord *iteration,
                                            uint64_t attempt_id, uint32_t role,
                                            uint32_t parent) {
  SsResultHistoryRow row = ss_solver_history_row(
      SS_RESULT_HISTORY_NEWTON, attempt_id,
      iteration->ordinal < 0 ? 0U : (uint64_t)iteration->ordinal, role);
  uint32_t newton = 0U;
  size_t trial;
  SsStatus status;
  row.control = ss_solver_history_finite(iteration->control_after);
  row.residual = ss_solver_history_finite(iteration->convergence_after);
  row.status = (int32_t)iteration->line_search.outcome;
  row.reason = iteration->tangent_reused != 0 ? 1 : 0;
  row.iterations = iteration->linear_solve.iterations < 0
                       ? 0U
                       : (uint64_t)iteration->linear_solve.iterations;
  row.tolerance = ss_solver_history_finite(iteration->regularization_after);
  row.reference = ss_solver_history_finite(iteration->residual_before);
  row.step = ss_solver_history_finite(iteration->alpha);
  row.auxiliary = ss_solver_history_finite(iteration->relative_correction);
  status = ss_solver_history_emit(emitter, &row, parent, &newton);
  if (status == SS_OK) {
    status = ss_solver_history_linear(emitter, &iteration->linear_solve, attempt_id,
                                      row.ordinal, role, newton);
  }
  if (status == SS_OK) {
    status =
        ss_solver_history_linear(emitter, &iteration->control_linear_solve, attempt_id,
                                 row.ordinal, SS_RESULT_ROLE_CONTROL, newton);
  }
  if (status == SS_OK) {
    status =
        ss_solver_history_linear(emitter, &iteration->residual_linear_solve, attempt_id,
                                 row.ordinal, SS_RESULT_ROLE_RESIDUAL, newton);
  }
  for (trial = 0U; status == SS_OK && trial < iteration->line_search.trial_count &&
                   trial < SS_NONLINEAR_LINE_TRIAL_CAPACITY;
       ++trial) {
    const SsNonlinearLineTrial *source = &iteration->line_search.trials[trial];
    SsResultHistoryRow trial_row = ss_solver_history_row(
        SS_RESULT_HISTORY_LINE_SEARCH, attempt_id,
        source->ordinal < 0 ? 0U : (uint64_t)source->ordinal, role);
    trial_row.control = ss_solver_history_finite(source->alpha);
    trial_row.residual = ss_solver_history_finite(source->residual_norm);
    trial_row.status = source->accepted != 0 ? 1 : 0;
    trial_row.reason =
        (source->finite != 0 ? 1 : 0) | (source->admissible != 0 ? 2 : 0);
    trial_row.step = ss_solver_history_finite(source->alpha);
    trial_row.reference =
        ss_solver_history_finite(iteration->line_search.current_residual_norm);
    trial_row.auxiliary = ss_solver_history_finite(source->directional_value);
    status = ss_solver_history_emit(emitter, &trial_row, newton, NULL);
  }
  if (status == SS_OK && iteration->linear_setup.pmg_selected != 0 &&
      iteration->linear_setup.reused == 0) {
    const SsPmgSetupRecord *setup = &iteration->linear_setup.pmg_setup;
    SsResultHistoryRow pmg =
        ss_solver_history_row(SS_RESULT_HISTORY_PMG, attempt_id, row.ordinal, role);
    pmg.control = (double)setup->galerkin_numeric_count;
    pmg.status = (setup->numeric_refreshed != 0 ? 1 : 0) |
                 (setup->numeric_lagged != 0 ? 2 : 0) |
                 (setup->hierarchy_reused != 0 ? 4 : 0);
    pmg.iterations = (uint64_t)setup->hierarchy.level_count;
    pmg.reference = (double)setup->admitted_memory_bytes;
    pmg.auxiliary = (double)setup->matrix_object_count;
    status = ss_solver_history_emit(emitter, &pmg, newton, NULL);
  }
  return status;
}

SsStatus ss_solver_history_solve(SsSolverHistoryEmitter *emitter,
                                 const SsNonlinearSolveRecord *solve,
                                 uint64_t attempt_id, uint32_t role,
                                 const SsPmgDiagnostics *pmg_delta) {
  size_t count;
  size_t index;
  SsStatus status = SS_OK;
  if (emitter == NULL || emitter->emit == NULL || solve == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  count = solve->history_count;
  /* A failed linear solve leaves its attempted iteration at history[count]. */
  if (count < SS_NONLINEAR_HISTORY_CAPACITY &&
      (solve->outcome == SS_NONLINEAR_FAILED_LINEAR ||
       (solve->outcome == SS_NONLINEAR_CANCELLED &&
        (solve->history[count].linear_solve.status == SS_LINEAR_SOLVE_CANCELLED ||
         solve->history[count].control_linear_solve.status ==
             SS_LINEAR_SOLVE_CANCELLED)))) {
    ++count;
  }
  if (count > SS_NONLINEAR_HISTORY_CAPACITY) {
    count = SS_NONLINEAR_HISTORY_CAPACITY;
  }
  for (index = 0U; status == SS_OK && index < count; ++index) {
    status = ss_solver_history_iteration(emitter, &solve->history[index], attempt_id,
                                         role, SS_SOLVER_HISTORY_PARENT_GROUP);
  }
  if (status == SS_OK && pmg_delta != NULL && pmg_delta->apply_calls != 0U) {
    SsResultHistoryRow pmg =
        ss_solver_history_row(SS_RESULT_HISTORY_PMG, attempt_id, (uint64_t)count, role);
    pmg.control = (double)pmg_delta->apply_calls;
    pmg.residual = (double)pmg_delta->failed_apply_calls;
    pmg.status = -1;
    pmg.iterations = pmg_delta->level_solve_calls;
    pmg.reference = (double)pmg_delta->bridge_bytes;
    pmg.auxiliary = (double)pmg_delta->operator_apply_calls;
    status =
        ss_solver_history_emit(emitter, &pmg, SS_SOLVER_HISTORY_PARENT_GROUP, NULL);
  }
  return status;
}

SsStatus ss_solver_history_linear_solve(SsSolverHistoryEmitter *emitter,
                                        const SsLinearSolveRecord *solve,
                                        uint64_t attempt_id, uint32_t role) {
  if (emitter == NULL || emitter->emit == NULL || solve == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return ss_solver_history_linear(emitter, solve, attempt_id, 0U, role,
                                  SS_SOLVER_HISTORY_PARENT_GROUP);
}

/* ---- rank-zero spool ---------------------------------------------------- */

static SsStatus ss_solver_history_spool_emit(void *context,
                                             const SsSolverHistoryEntry *entry) {
  SsSolverHistorySpool *spool = (SsSolverHistorySpool *)context;
  if (spool->file != NULL && fwrite(entry, sizeof(*entry), 1U, spool->file) != 1U) {
    return SS_ERROR_IO;
  }
  /* Counted on every rank: the count is replicated (checkpoint cursors). */
  ++spool->written;
  return SS_OK;
}

SsStatus ss_solver_history_spool_open(SsSolverHistorySpool *spool,
                                      MPI_Comm communicator, const char *path) {
  int rank = 0;
  SsStatus status = SS_OK;
  if (spool == NULL || path == NULL ||
      MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(spool, 0, sizeof(*spool));
  spool->communicator = communicator;
  spool->rank = rank;
  if ((size_t)snprintf(spool->path, sizeof(spool->path), "%s", path) >=
      sizeof(spool->path)) {
    status = SS_ERROR_LIMIT;
  }
  if (status == SS_OK && rank == 0) {
    spool->file = fopen(spool->path, "w+b");
    if (spool->file == NULL) {
      status = SS_ERROR_IO;
    }
  }
  return ss_status_allreduce_max(communicator, status);
}

SsSolverHistoryEmitter ss_solver_history_spool_emitter(SsSolverHistorySpool *spool) {
  SsSolverHistoryEmitter emitter;
  memset(&emitter, 0, sizeof(emitter));
  emitter.emit = ss_solver_history_spool_emit;
  emitter.context = spool;
  return emitter;
}

SsStatus ss_solver_history_spool_commit(SsSolverHistorySpool *spool) {
  SsStatus status = SS_OK;
  if (spool == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (spool->file != NULL && fflush(spool->file) != 0) {
    status = SS_ERROR_IO;
  }
  return ss_status_allreduce_max(spool->communicator, status);
}

SsStatus ss_solver_history_spool_rewind(SsSolverHistorySpool *spool) {
  SsStatus status = SS_OK;
  if (spool == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (spool->file != NULL &&
      (fflush(spool->file) != 0 || fseek(spool->file, 0L, SEEK_SET) != 0)) {
    status = SS_ERROR_IO;
  }
  spool->read = 0U;
  spool->has_lookahead = 0;
  return ss_status_allreduce_max(spool->communicator, status);
}

/* Rank zero: peek at the next spooled entry. */
static int ss_solver_history_spool_peek(SsSolverHistorySpool *spool,
                                        SsSolverHistoryEntry **entry) {
  if (spool->has_lookahead == 0) {
    if (spool->file == NULL || spool->read >= spool->written ||
        fread(&spool->lookahead, sizeof(spool->lookahead), 1U, spool->file) != 1U) {
      return 0;
    }
    ++spool->read;
    spool->has_lookahead = 1;
  }
  *entry = &spool->lookahead;
  return 1;
}

SsStatus ss_solver_history_spool_replay(SsSolverHistorySpool *spool,
                                        SsResultWriter *writer, uint64_t attempt_id,
                                        uint64_t group_row, uint64_t *row_id) {
  SsResultHistoryRow *rows = NULL;
  uint64_t group_first = 0U;
  int done = 0;
  SsStatus status = SS_OK;
  if (spool == NULL || writer == NULL || row_id == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  rows = (SsResultHistoryRow *)malloc(SS_SOLVER_HISTORY_REPLAY_BATCH * sizeof(*rows));
  status = rows == NULL ? SS_ERROR_OUT_OF_MEMORY : SS_OK;
  status = ss_status_allreduce_max(spool->communicator, status);
  group_first = *row_id;
  while (status == SS_OK && done == 0) {
    uint64_t count = 0U;
    if (spool->rank == 0) {
      SsSolverHistoryEntry *entry = NULL;
      /* Entries of attempts without an attempt row are not published. */
      while (ss_solver_history_spool_peek(spool, &entry) != 0 &&
             entry->attempt_id < attempt_id) {
        spool->has_lookahead = 0;
      }
      while (count < SS_SOLVER_HISTORY_REPLAY_BATCH &&
             ss_solver_history_spool_peek(spool, &entry) != 0 &&
             entry->attempt_id == attempt_id) {
        SsResultHistoryRow row = entry->row;
        row.row_id = *row_id + count;
        row.parent_row_id = entry->parent_index == SS_SOLVER_HISTORY_PARENT_GROUP
                                ? group_row
                                : group_first + (uint64_t)entry->parent_index;
        if (row.row_id == 0U) {
          row.parent_row_id = 0U;
        }
        rows[count++] = row;
        spool->has_lookahead = 0;
      }
    }
    if (MPI_Bcast(&count, 1, MPI_UINT64_T, 0, spool->communicator) != MPI_SUCCESS ||
        (count != 0U && MPI_Bcast(rows, (int)(count * sizeof(*rows)), MPI_BYTE, 0,
                                  spool->communicator) != MPI_SUCCESS)) {
      status = SS_ERROR_INTERNAL;
      break;
    }
    if (count != 0U) {
      status = ss_result_writer_append_history_rows(writer, rows, (size_t)count);
      *row_id += count;
    }
    done = count < SS_SOLVER_HISTORY_REPLAY_BATCH;
  }
  free(rows);
  return ss_status_allreduce_max(spool->communicator, status);
}

SsStatus ss_solver_history_spool_close(SsSolverHistorySpool *spool, int remove_file) {
  SsStatus status = SS_OK;
  if (spool == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (spool->file != NULL) {
    if (fclose(spool->file) != 0) {
      status = SS_ERROR_IO;
    }
    spool->file = NULL;
    if (remove_file != 0 && unlink(spool->path) != 0) {
      status = SS_ERROR_IO;
    }
  }
  return status;
}

/* ---- direct appends ----------------------------------------------------- */

typedef struct {
  SsResultWriter *writer;
  uint64_t group_row;
  uint64_t group_first;
  uint64_t next_row;
  SsStatus status;
} SsSolverHistoryDirect;

static SsStatus ss_solver_history_direct_emit(void *context,
                                              const SsSolverHistoryEntry *entry) {
  SsSolverHistoryDirect *direct = (SsSolverHistoryDirect *)context;
  SsResultHistoryRow row = entry->row;
  row.row_id = direct->next_row;
  row.parent_row_id = entry->parent_index == SS_SOLVER_HISTORY_PARENT_GROUP
                          ? direct->group_row
                          : direct->group_first + (uint64_t)entry->parent_index;
  ++direct->next_row;
  return ss_result_writer_append_history(direct->writer, &row);
}

SsStatus ss_solver_history_append_solve(SsResultWriter *writer,
                                        const SsNonlinearSolveRecord *solve,
                                        uint64_t attempt_id, uint32_t role,
                                        uint64_t group_row, uint64_t *row_id) {
  SsSolverHistoryDirect direct;
  SsSolverHistoryEmitter emitter;
  SsStatus status;
  if (writer == NULL || solve == NULL || row_id == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(&direct, 0, sizeof(direct));
  direct.writer = writer;
  direct.group_row = group_row;
  direct.group_first = *row_id;
  direct.next_row = *row_id;
  memset(&emitter, 0, sizeof(emitter));
  emitter.emit = ss_solver_history_direct_emit;
  emitter.context = &direct;
  status = ss_solver_history_solve(&emitter, solve, attempt_id, role, NULL);
  *row_id = direct.next_row;
  return status;
}
