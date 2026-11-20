/* OUT-PERF-APPEND-001: result appends read no earlier states.
 *
 * Each state also streams 200 per-iteration history rows, so the writer
 * flushes closed history windows several times before publication.
 *
 * Linked with -Wl,--wrap=H5Dread so every successful read selection of a
 * quantity `values` dataset is counted.  An append must not read stored
 * values; publication must read every stored value a bounded number of times
 * (one finalization pass plus one verification pass).  The test also checks
 * that a different local distribution of the same entities is accepted and a
 * different entity set is rejected without reading the file.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/result/result.h"

#include <hdf5.h>
#include <inttypes.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { cell_count = 1536, samples = 6, components = 4 };

static uint64_t value_read_bytes;

herr_t __real_H5Dread(hid_t dataset, hid_t type, hid_t memory, hid_t file,
                      hid_t transfer, void *buffer);
herr_t __wrap_H5Dread(hid_t dataset, hid_t type, hid_t memory, hid_t file,
                      hid_t transfer, void *buffer);

herr_t __wrap_H5Dread(hid_t dataset, hid_t type, hid_t memory, hid_t file,
                      hid_t transfer, void *buffer) {
  char path[256] = {0};
  const herr_t result = __real_H5Dread(dataset, type, memory, file, transfer, buffer);
  if (result >= 0 && H5Iget_name(dataset, path, sizeof(path)) > 0) {
    const size_t length = strlen(path);
    if (length > 7U && strcmp(path + length - 7U, "/values") == 0) {
      const hid_t space = memory == H5S_ALL ? H5Dget_space(dataset) : memory;
      const hssize_t selected = H5Sget_select_npoints(space);
      if (memory == H5S_ALL) {
        (void)H5Sclose(space);
      }
      if (selected > 0) {
        value_read_bytes += (uint64_t)selected * (uint64_t)H5Tget_size(type);
      }
    }
  }
  return result;
}

static int failures;

#define CHECK(condition)                                                               \
  do {                                                                                 \
    if (!(condition)) {                                                                \
      fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition);    \
      ++failures;                                                                      \
    }                                                                                  \
  } while (0)

static uint64_t global_reads(void) {
  uint64_t total = 0U;
  MPI_Allreduce(&value_read_bytes, &total, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  return total;
}

enum { newton_rows_per_state = 200 };

/* Per-iteration detail rows of one state's attempt (row IDs from `first`). */
static void newton_rows(SsResultHistoryRow *rows, uint64_t first) {
  for (size_t index = 0U; index < newton_rows_per_state; ++index) {
    memset(&rows[index], 0, sizeof(rows[index]));
    rows[index].kind = SS_RESULT_HISTORY_NEWTON;
    rows[index].row_id = first + index;
    rows[index].parent_row_id = first == 0U && index == 0U ? 0U : first + index - 1U;
    rows[index].ordinal = index;
    rows[index].control = 0.5 * (double)index;
    rows[index].residual = 1.0 / (1.0 + (double)index);
    rows[index].complete = 1;
    rows[index].role = SS_RESULT_ROLE_PRIMARY;
    rows[index].iterations = 7U + index;
    rows[index].step = 1.0;
    rows[index].tolerance = 1.0e-8;
  }
}

static SsResultHistoryRow accepted_row(uint64_t row_id, SsStateId state_id) {
  SsResultHistoryRow row;
  memset(&row, 0, sizeof(row));
  row.kind = SS_RESULT_HISTORY_ACCEPTED;
  row.row_id = row_id;
  row.state_id = state_id;
  row.parent_row_id = row_id == 0U ? 0U : row_id - 1U;
  row.ordinal = row_id;
  row.complete = 1;
  row.accepted = 1;
  return row;
}

static int run(const char *directory, uint32_t state_count) {
  static const char *const labels[] = {"xx", "yy", "zz", "xy"};
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const size_t local_count = (size_t)(cell_count / size) +
                             ((size_t)rank < (size_t)(cell_count % size) ? 1U : 0U);
  SsEntityId *ids = calloc(local_count, sizeof(*ids));
  SsEntityId *wrong_ids = calloc(local_count, sizeof(*wrong_ids));
  SsEntityId *reversed_ids = calloc(local_count, sizeof(*reversed_ids));
  double *values = calloc(local_count * samples * components, sizeof(*values));
  double *reversed_values = calloc(local_count * samples * components, sizeof(*values));
  const uint64_t state_bytes = (uint64_t)cell_count * samples * components * 8U;
  SsResultWriter *writer = NULL;
  if (ids == NULL || wrong_ids == NULL || reversed_ids == NULL || values == NULL ||
      reversed_values == NULL) {
    return 1;
  }
  for (size_t index = 0U; index < local_count; ++index) {
    const uint64_t ordinal = (uint64_t)(index * (size_t)size + (size_t)rank);
    ids[index] = UINT64_C(0x9e3779b97f4a7c15) * (ordinal + 1U);
    wrong_ids[index] = ids[index] ^ (rank == 0 && index == 0U ? 1U : 0U);
  }
  for (size_t index = 0U; index < local_count; ++index) {
    reversed_ids[index] = ids[local_count - 1U - index];
  }
  const SsResultWriterSpec spec = {
      .communicator = MPI_COMM_WORLD,
      .final_directory = directory,
      .problem_id =
          "sha256:1111111111111111111111111111111111111111111111111111111111111111",
      .run_id =
          "sha256:2222222222222222222222222222222222222222222222222222222222222222",
      .build_id =
          "sha256:3333333333333333333333333333333333333333333333333333333333333333",
      .mesh_id =
          "sha256:4444444444444444444444444444444444444444444444444444444444444444",
      .geometry_id =
          "sha256:5555555555555555555555555555555555555555555555555555555555555555",
      .analysis_method = "direct_ssr",
      .dimension = 3,
      .field_order = 2,
      .geometry_order = 1,
      .dataset_chunk_rows = 512,
  };
  const SsResultQuantityDeclaration declaration = {
      .quantity_id = "stress",
      .quantity_version = 1,
      .unit = "kPa",
      .location = "mechanics_quadrature",
      .basis = "cell_quadrature",
      .component_ids = labels,
      .component_count = components,
      .stable_ids = ids,
      .local_count = local_count,
      .global_count = cell_count,
      .dataset_kind = SS_RESULT_DATASET_QUADRATURE,
      .quadrature_id = "probe_rule",
      .sample_count = samples,
  };
  SsResultQuantityDeclaration shared = declaration;
  shared.quantity_id = "strain";
  CHECK(ss_result_writer_create(&spec, &writer) == SS_OK);
  CHECK(ss_result_writer_stage(writer) == SS_OK);
  CHECK(ss_result_writer_declare_quantity(writer, &declaration) == SS_OK);
  CHECK(ss_result_writer_declare_quantity(writer, &shared) == SS_OK);
  CHECK(ss_result_writer_begin_stream(writer) == SS_OK);
  for (uint32_t state = 1U; state <= state_count; ++state) {
    const uint64_t window = (uint64_t)newton_rows_per_state + 1U;
    const uint64_t first_row = (uint64_t)(state - 1U) * window;
    SsResultHistoryRow details[newton_rows_per_state];
    const SsResultHistoryRow row =
        accepted_row(first_row + newton_rows_per_state, state);
    const SsResultAcceptedState accepted = {
        .state_id = state,
        .parent_state_id = state - 1U,
        .accepted_ordinal = state - 1U,
        .history_begin = first_row,
        .history_end = first_row + window,
        .complete = 1,
        .accepted = 1,
    };
    newton_rows(details, first_row);
    CHECK(ss_result_writer_append_history_rows(writer, details,
                                               newton_rows_per_state) == SS_OK);
    SsResultFieldView view = {
        .quantity_id = "stress",
        .quantity_version = 1,
        .unit = "kPa",
        .location = "mechanics_quadrature",
        .basis = "cell_quadrature",
        .component_ids = labels,
        .component_count = components,
        .state_id = state,
        .stable_ids = ids,
        .values = values,
        .local_count = local_count,
        .global_count = cell_count,
        .dataset_kind = SS_RESULT_DATASET_QUADRATURE,
        .quadrature_id = "probe_rule",
        .sample_count = samples,
    };
    uint64_t before;
    for (size_t index = 0U; index < local_count * samples * components; ++index) {
      values[index] = (double)state + 1.0e-3 * (double)(index % 977U);
    }
    for (size_t index = 0U; index < local_count; ++index) {
      memcpy(reversed_values + index * samples * components,
             values + (local_count - 1U - index) * samples * components,
             sizeof(double) * samples * components);
    }
    CHECK(ss_result_writer_append_history(writer, &row) == SS_OK);
    CHECK(ss_result_writer_accept_state(writer, &accepted) == SS_OK);
    before = global_reads();
    CHECK(ss_result_writer_write_quantity(writer, &view) == SS_OK);
    view.quantity_id = "strain";
    if (state == 2U) {
      /* Same entities, different local order: accepted through a temporary plan. */
      view.stable_ids = reversed_ids;
      view.local_slots = NULL;
      view.values = reversed_values;
    }
    CHECK(ss_result_writer_write_quantity(writer, &view) == SS_OK);
    {
      const uint64_t appended = global_reads() - before;
      if (rank == 0) {
        printf("append,%" PRIu32 ",%" PRIu64 "\n", state, appended);
      }
      CHECK(appended == 0U);
    }
  }
  {
    SsResultFieldView wrong = {
        .quantity_id = "stress",
        .quantity_version = 1,
        .unit = "kPa",
        .location = "mechanics_quadrature",
        .basis = "cell_quadrature",
        .component_ids = labels,
        .component_count = components,
        .state_id = state_count,
        .stable_ids = wrong_ids,
        .values = values,
        .local_count = local_count,
        .global_count = cell_count,
        .dataset_kind = SS_RESULT_DATASET_QUADRATURE,
        .quadrature_id = "probe_rule",
        .sample_count = samples,
    };
    const uint64_t first_row = (uint64_t)state_count * (newton_rows_per_state + 1U);
    const SsResultHistoryRow row = accepted_row(first_row, state_count + 1U);
    const SsResultAcceptedState accepted = {
        .state_id = state_count + 1U,
        .parent_state_id = state_count,
        .accepted_ordinal = state_count,
        .history_begin = first_row,
        .history_end = first_row + 1U,
        .complete = 1,
        .accepted = 1,
    };
    const uint64_t before = global_reads();
    CHECK(ss_result_writer_append_history(writer, &row) == SS_OK);
    CHECK(ss_result_writer_accept_state(writer, &accepted) == SS_OK);
    wrong.state_id = state_count + 1U;
    CHECK(ss_result_writer_write_quantity(writer, &wrong) == SS_ERROR_INTEGRITY);
    CHECK(global_reads() == before);
  }
  {
    const uint64_t before = global_reads();
    const uint64_t stored = 2U * (uint64_t)state_count * state_bytes;
    uint64_t publication;
    CHECK(ss_result_writer_terminalize(writer, SS_RESULT_SUCCEEDED, "converged",
                                       NULL) == SS_OK);
    CHECK(ss_result_writer_publish(writer) == SS_OK);
    publication = global_reads() - before;
    if (rank == 0) {
      printf("publish,%" PRIu32 ",%" PRIu64 ",stored=%" PRIu64 "\n", state_count,
             publication, stored);
    }
    /* One finalization pass and one verification pass over every stored value. */
    CHECK(publication == 2U * stored);
  }
  {
    SsResultWriterSummary summary;
    CHECK(ss_result_writer_summary(writer, &summary) == SS_OK);
    CHECK(summary.history_row_count ==
          (uint64_t)state_count * (newton_rows_per_state + 1U) + 1U);
  }
  CHECK(ss_result_writer_destroy(&writer) == SS_OK);
  free(reversed_values);
  free(values);
  free(reversed_ids);
  free(wrong_ids);
  free(ids);
  return 0;
}

int main(int argc, char **argv) {
  int provided = 0;
  int rank = 0;
  int total = 0;
  char directory[1024];
  if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
    return 1;
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (argc != 2) {
    if (rank == 0) {
      fprintf(stderr, "usage: %s <result-parent-directory>\n", argv[0]);
    }
    MPI_Finalize();
    return 2;
  }
  for (uint32_t states = 3U; states <= 9U; states += 6U) {
    (void)snprintf(directory, sizeof(directory), "%s/result-%u", argv[1], states);
    if (rank == 0) {
      char command[1200];
      (void)snprintf(command, sizeof(command), "rm -rf '%s'", directory);
      if (system(command) != 0) {
        ++failures;
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (run(directory, states) != 0) {
      ++failures;
    }
  }
  MPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    printf("%s\n",
           total == 0 ? "OUT-PERF-APPEND-001 passed" : "OUT-PERF-APPEND-001 failed");
  }
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}
