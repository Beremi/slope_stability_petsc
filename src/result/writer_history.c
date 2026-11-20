/* Streamed relational history tables of the result writer.
 *
 * History rows are replicated on every rank.  The writer keeps only the rows
 * of the window that is still open (since the last accepted state that
 * flushed); closed windows are appended to the per-kind `/history/<kind>/rows`
 * tables by rank zero inside collective HDF5 calls.  The result logical digest
 * reads the tables back in row-ID order with a fixed per-table batch, so
 * neither streaming nor publication holds the whole history in memory.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "writer_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Rows read per history table while merging tables in row-ID order. */
#define SS_RESULT_HISTORY_READ_BATCH 512U
#define SS_RESULT_HISTORY_KIND_COUNT ((size_t)SS_RESULT_HISTORY_PMG_APPLY)

hid_t ss_result_history_file_type(void) {
  hid_t type = H5Tcreate(H5T_COMPOUND, 120U);
  if (type < 0 || H5Tinsert(type, "kind", 0U, H5T_STD_U32LE) < 0 ||
      H5Tinsert(type, "row_id", 4U, H5T_STD_U64LE) < 0 ||
      H5Tinsert(type, "state_id", 12U, H5T_STD_U64LE) < 0 ||
      H5Tinsert(type, "attempt_id", 20U, H5T_STD_U64LE) < 0 ||
      H5Tinsert(type, "parent_row_id", 28U, H5T_STD_U64LE) < 0 ||
      H5Tinsert(type, "ordinal", 36U, H5T_STD_U64LE) < 0 ||
      H5Tinsert(type, "control", 44U, H5T_IEEE_F64LE) < 0 ||
      H5Tinsert(type, "residual", 52U, H5T_IEEE_F64LE) < 0 ||
      H5Tinsert(type, "complete", 60U, H5T_STD_I32LE) < 0 ||
      H5Tinsert(type, "accepted", 64U, H5T_STD_I32LE) < 0 ||
      H5Tinsert(type, "role", 68U, H5T_STD_U32LE) < 0 ||
      H5Tinsert(type, "status", 72U, H5T_STD_I32LE) < 0 ||
      H5Tinsert(type, "reason", 76U, H5T_STD_I32LE) < 0 ||
      H5Tinsert(type, "iterations", 80U, H5T_STD_U64LE) < 0 ||
      H5Tinsert(type, "tolerance", 88U, H5T_IEEE_F64LE) < 0 ||
      H5Tinsert(type, "reference", 96U, H5T_IEEE_F64LE) < 0 ||
      H5Tinsert(type, "step", 104U, H5T_IEEE_F64LE) < 0 ||
      H5Tinsert(type, "auxiliary", 112U, H5T_IEEE_F64LE) < 0) {
    if (type >= 0) {
      (void)H5Tclose(type);
    }
    return -1;
  }
  return type;
}

hid_t ss_result_history_memory_type(void) {
  hid_t type = H5Tcreate(H5T_COMPOUND, sizeof(SsResultHistoryRow));
#define SS_RESULT_HISTORY_MEMBER(name, native)                                         \
  (H5Tinsert(type, #name, HOFFSET(SsResultHistoryRow, name), native) < 0)
  if (type < 0 || SS_RESULT_HISTORY_MEMBER(kind, H5T_NATIVE_INT) ||
      SS_RESULT_HISTORY_MEMBER(row_id, H5T_NATIVE_UINT64) ||
      SS_RESULT_HISTORY_MEMBER(state_id, H5T_NATIVE_UINT64) ||
      SS_RESULT_HISTORY_MEMBER(attempt_id, H5T_NATIVE_UINT64) ||
      SS_RESULT_HISTORY_MEMBER(parent_row_id, H5T_NATIVE_UINT64) ||
      SS_RESULT_HISTORY_MEMBER(ordinal, H5T_NATIVE_UINT64) ||
      SS_RESULT_HISTORY_MEMBER(control, H5T_NATIVE_DOUBLE) ||
      SS_RESULT_HISTORY_MEMBER(residual, H5T_NATIVE_DOUBLE) ||
      SS_RESULT_HISTORY_MEMBER(complete, H5T_NATIVE_INT) ||
      SS_RESULT_HISTORY_MEMBER(accepted, H5T_NATIVE_INT) ||
      SS_RESULT_HISTORY_MEMBER(role, H5T_NATIVE_UINT32) ||
      SS_RESULT_HISTORY_MEMBER(status, H5T_NATIVE_INT32) ||
      SS_RESULT_HISTORY_MEMBER(reason, H5T_NATIVE_INT32) ||
      SS_RESULT_HISTORY_MEMBER(iterations, H5T_NATIVE_UINT64) ||
      SS_RESULT_HISTORY_MEMBER(tolerance, H5T_NATIVE_DOUBLE) ||
      SS_RESULT_HISTORY_MEMBER(reference, H5T_NATIVE_DOUBLE) ||
      SS_RESULT_HISTORY_MEMBER(step, H5T_NATIVE_DOUBLE) ||
      SS_RESULT_HISTORY_MEMBER(auxiliary, H5T_NATIVE_DOUBLE)) {
    if (type >= 0) {
      (void)H5Tclose(type);
    }
    return -1;
  }
#undef SS_RESULT_HISTORY_MEMBER
  return type;
}

const char *ss_result_history_name(SsResultHistoryKind kind) {
  switch (kind) {
  case SS_RESULT_HISTORY_ACCEPTED:
    return "accepted";
  case SS_RESULT_HISTORY_ATTEMPT:
    return "attempt";
  case SS_RESULT_HISTORY_NEWTON:
    return "newton";
  case SS_RESULT_HISTORY_LINE_SEARCH:
    return "line_search";
  case SS_RESULT_HISTORY_LINEAR:
    return "linear";
  case SS_RESULT_HISTORY_PMG:
    return "pmg";
  case SS_RESULT_HISTORY_DEFLATION:
    return "deflation";
  case SS_RESULT_HISTORY_SEEPAGE:
    return "seepage";
  case SS_RESULT_HISTORY_IO:
    return "io";
  case SS_RESULT_HISTORY_KRYLOV:
    return "krylov";
  case SS_RESULT_HISTORY_PMG_APPLY:
    return "pmg_apply";
  default:
    return NULL;
  }
}

/* Collectively append `count` replicated rows (rank zero supplies them). */
static SsStatus ss_result_history_append_table(SsResultWriter *writer,
                                               SsResultHistoryKind kind,
                                               hid_t memory_type,
                                               const SsResultHistoryRow *rows,
                                               size_t count) {
  char path[SS_RESULT_PATH_CAPACITY];
  const uint64_t offset = writer->history_written[kind];
  hsize_t extent[1] = {(hsize_t)(offset + (uint64_t)count)};
  hsize_t start[1] = {(hsize_t)offset};
  hsize_t selected[1] = {(hsize_t)count};
  hsize_t memory_dimensions[1] = {count == 0U ? 1U : (hsize_t)count};
  hid_t dataset = -1;
  hid_t file_space = -1;
  hid_t memory_space = -1;
  hid_t transfer = -1;
  const int written =
      snprintf(path, sizeof(path), "/history/%s/rows", ss_result_history_name(kind));
  SsStatus status = SS_OK;
  if (written < 0 || (size_t)written >= sizeof(path)) {
    return SS_ERROR_OVERFLOW;
  }
  dataset = H5Dopen2(writer->file, path, H5P_DEFAULT);
  if (dataset < 0 || H5Dset_extent(dataset, extent) < 0) {
    status = SS_ERROR_HDF5;
  }
  file_space = dataset < 0 ? -1 : H5Dget_space(dataset);
  memory_space = H5Screate_simple(1, memory_dimensions, NULL);
  transfer = H5Pcreate(H5P_DATASET_XFER);
  if (status != SS_OK || file_space < 0 || memory_space < 0 || transfer < 0 ||
      H5Pset_dxpl_mpio(transfer, H5FD_MPIO_COLLECTIVE) < 0) {
    status = SS_ERROR_HDF5;
  } else if (writer->rank != 0) {
    if (H5Sselect_none(file_space) < 0 || H5Sselect_none(memory_space) < 0) {
      status = SS_ERROR_HDF5;
    }
  } else if (H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, selected,
                                 NULL) < 0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK && H5Dwrite(dataset, memory_type, memory_space, file_space,
                                  transfer, writer->rank == 0 ? rows : NULL) < 0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_result_hdf5_close(&transfer, H5Pclose, status);
  status = ss_result_hdf5_close(&memory_space, H5Sclose, status);
  status = ss_result_hdf5_close(&file_space, H5Sclose, status);
  status = ss_result_hdf5_close(&dataset, H5Dclose, status);
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK) {
    writer->history_written[kind] = offset + (uint64_t)count;
  }
  return status;
}

SsStatus ss_result_writer_flush_history(SsResultWriter *writer, uint64_t row_end) {
  SsResultHistoryRow *rows = NULL;
  hid_t memory_type = -1;
  size_t flushed;
  size_t buffered;
  size_t kind;
  SsStatus status = SS_OK;
  if (writer == NULL || row_end < writer->history_buffer_begin ||
      row_end > (uint64_t)writer->history_count) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  flushed = (size_t)(row_end - writer->history_buffer_begin);
  buffered = writer->history_count - (size_t)writer->history_buffer_begin;
  if (flushed == 0U) {
    return SS_OK;
  }
  memory_type = ss_result_history_memory_type();
  rows = (SsResultHistoryRow *)malloc(flushed * sizeof(*rows));
  if (memory_type < 0 || rows == NULL) {
    status = memory_type < 0 ? SS_ERROR_HDF5 : SS_ERROR_OUT_OF_MEMORY;
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  /* Rows are identical on every rank, so every rank skips the same empty kinds. */
  for (kind = 1U; status == SS_OK && kind <= SS_RESULT_HISTORY_KIND_COUNT; ++kind) {
    size_t count = 0U;
    size_t index;
    for (index = 0U; index < flushed; ++index) {
      if ((size_t)writer->history[index].kind == kind) {
        rows[count++] = writer->history[index];
      }
    }
    if (count != 0U) {
      status = ss_result_history_append_table(writer, (SsResultHistoryKind)kind,
                                              memory_type, rows, count);
    }
  }
  if (status == SS_OK) {
    memmove(writer->history, writer->history + flushed,
            (buffered - flushed) * sizeof(*writer->history));
    writer->history_buffer_begin = row_end;
  }
  free(rows);
  status = ss_result_hdf5_close(&memory_type, H5Tclose, status);
  return ss_status_allreduce_max(writer->communicator, status);
}

typedef struct {
  hid_t dataset;
  uint64_t total;
  uint64_t next;
  SsResultHistoryRow rows[SS_RESULT_HISTORY_READ_BATCH];
  size_t count;
  size_t cursor;
} SsResultHistoryCursor;

static SsStatus ss_result_history_cursor_fill(SsResultHistoryCursor *cursor,
                                              hid_t memory_type) {
  hsize_t start[1];
  hsize_t count[1];
  hid_t file_space = -1;
  hid_t memory_space = -1;
  SsStatus status = SS_OK;
  if (cursor->cursor < cursor->count || cursor->next >= cursor->total) {
    return SS_OK;
  }
  start[0] = (hsize_t)cursor->next;
  count[0] = (hsize_t)(cursor->total - cursor->next < SS_RESULT_HISTORY_READ_BATCH
                           ? cursor->total - cursor->next
                           : SS_RESULT_HISTORY_READ_BATCH);
  file_space = H5Dget_space(cursor->dataset);
  memory_space = H5Screate_simple(1, count, NULL);
  if (file_space < 0 || memory_space < 0 ||
      H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL) < 0 ||
      H5Dread(cursor->dataset, memory_type, memory_space, file_space, H5P_DEFAULT,
              cursor->rows) < 0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_result_hdf5_close(&memory_space, H5Sclose, status);
  status = ss_result_hdf5_close(&file_space, H5Sclose, status);
  if (status == SS_OK) {
    cursor->count = (size_t)count[0];
    cursor->cursor = 0U;
    cursor->next += (uint64_t)count[0];
  }
  return status;
}

static SsStatus ss_result_history_digest_row(SsSha256 *context,
                                             const SsResultHistoryRow *row) {
  unsigned char encoded[10U * sizeof(uint64_t)];
  ss_result_store_u64_le(encoded, (uint64_t)row->kind);
  ss_result_store_u64_le(encoded + 8U, row->row_id);
  ss_result_store_u64_le(encoded + 16U, row->state_id);
  ss_result_store_u64_le(encoded + 24U, row->attempt_id);
  ss_result_store_u64_le(encoded + 32U, row->parent_row_id);
  ss_result_store_u64_le(encoded + 40U, row->ordinal);
  ss_result_store_double_le(encoded + 48U, row->control);
  ss_result_store_double_le(encoded + 56U, row->residual);
  ss_result_store_u64_le(encoded + 64U, (uint64_t)(row->complete != 0));
  ss_result_store_u64_le(encoded + 72U, (uint64_t)(row->accepted != 0));
  return ss_sha256_update(context, encoded, sizeof(encoded));
}

SsStatus ss_result_writer_history_digest(const SsResultWriter *writer,
                                         const char *hdf5_path, SsSha256 *context) {
  SsResultHistoryCursor *cursors = NULL;
  hid_t file = -1;
  hid_t memory_type = -1;
  uint64_t row;
  size_t kind;
  SsStatus status = SS_OK;
  if (writer == NULL || hdf5_path == NULL || context == NULL ||
      writer->history_buffer_begin != (uint64_t)writer->history_count) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  cursors = (SsResultHistoryCursor *)calloc(SS_RESULT_HISTORY_KIND_COUNT + 1U,
                                            sizeof(*cursors));
  memory_type = ss_result_history_memory_type();
  file = H5Fopen(hdf5_path, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (cursors == NULL || memory_type < 0 || file < 0) {
    status = cursors == NULL ? SS_ERROR_OUT_OF_MEMORY : SS_ERROR_HDF5;
  }
  for (kind = 1U; status == SS_OK && kind <= SS_RESULT_HISTORY_KIND_COUNT; ++kind) {
    char path[SS_RESULT_PATH_CAPACITY];
    const int written = snprintf(path, sizeof(path), "/history/%s/rows",
                                 ss_result_history_name((SsResultHistoryKind)kind));
    cursors[kind].dataset = -1;
    if (written < 0 || (size_t)written >= sizeof(path)) {
      status = SS_ERROR_OVERFLOW;
      break;
    }
    cursors[kind].dataset = H5Dopen2(file, path, H5P_DEFAULT);
    cursors[kind].total = writer->history_written[kind];
    if (cursors[kind].dataset < 0) {
      status = SS_ERROR_HDF5;
    }
  }
  /* Merge the per-kind tables: each is ascending in row ID. */
  for (row = 0U; status == SS_OK && row < (uint64_t)writer->history_count; ++row) {
    int found = 0;
    for (kind = 1U; status == SS_OK && kind <= SS_RESULT_HISTORY_KIND_COUNT; ++kind) {
      SsResultHistoryCursor *cursor = &cursors[kind];
      status = ss_result_history_cursor_fill(cursor, memory_type);
      if (status == SS_OK && cursor->cursor < cursor->count &&
          cursor->rows[cursor->cursor].row_id == row) {
        status = ss_result_history_digest_row(context, &cursor->rows[cursor->cursor]);
        ++cursor->cursor;
        found = 1;
        break;
      }
    }
    if (status == SS_OK && found == 0) {
      status = SS_ERROR_INTEGRITY;
    }
  }
  for (kind = 1U; cursors != NULL && kind <= SS_RESULT_HISTORY_KIND_COUNT; ++kind) {
    if (cursors[kind].dataset >= 0) {
      status = ss_result_hdf5_close(&cursors[kind].dataset, H5Dclose, status);
    }
  }
  if (file >= 0) {
    status = ss_result_hdf5_close(&file, H5Fclose, status);
  }
  if (memory_type >= 0) {
    status = ss_result_hdf5_close(&memory_type, H5Tclose, status);
  }
  free(cursors);
  return status;
}
