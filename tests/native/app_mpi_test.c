/* Diagnostics, collective HDF5 input and application lifecycle under MPI.
 *
 * Launched by tests/native/support/mpi_layout_runner.py --suite app.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/app/lifecycle.h"
#include "slope_internal/build/config.h"
#include "slope_internal/core/core.h"
#include "slope_internal/diagnostics/diagnostics.h"
#include "slope_internal/io/input.h"

#include <mpi.h>

#if SLOPE_ENABLE_OPENMP
#include <omp.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LOCAL_CHECK(expression)                                                        \
  do {                                                                                 \
    if (!(expression)) {                                                               \
      fprintf(stderr, "rank %d: %s:%d: assertion failed: %s\n", rank, __FILE__,        \
              __LINE__, #expression);                                                  \
      local_status = 1;                                                                \
    }                                                                                  \
  } while (0)

static int test_collective_error(MPI_Comm communicator, int rank, int size) {
  SsErrorRecord local = {0};
  SsErrorRecord selected = {0};
  if (size == 1 || rank == size - 1) {
    local.valid = 1;
    local.status = SS_ERROR_SCHEMA;
    local.key = (SsErrorKey){1U, 2U, 3U, 4U, 0U, (uint32_t)rank};
    strcpy(local.code, "RANK_LOCAL_SCHEMA");
    strcpy(local.message, "injected rank-local parse failure");
  }
  if (size > 2 && rank == 1) {
    local.valid = 1;
    local.status = SS_ERROR_INTEGRITY;
    local.key = (SsErrorKey){0U, 9U, 9U, 9U, 0U, (uint32_t)rank};
    strcpy(local.code, "EARLIER_PHASE");
    strcpy(local.message, "deterministically earlier phase");
  }
  if (ss_collective_error_select(communicator, &local, &selected) != SS_OK ||
      selected.valid == 0) {
    return 1;
  }
  if (size > 2) {
    return strcmp(selected.code, "EARLIER_PHASE") == 0 && selected.key.rank == 1U ? 0
                                                                                  : 1;
  }
  return strcmp(selected.code, "RANK_LOCAL_SCHEMA") == 0 &&
                 selected.key.rank == (uint32_t)(size - 1)
             ? 0
             : 1;
}

static int test_hdf5(MPI_Comm communicator, const char *path, int require_zero_owner,
                     int expect_failure) {
  SsH5Validation validation = {0};
  SsErrorRecord error = {0};
  SsStatus status = ss_io_validate_hdf5(communicator, path, &validation, &error);
  if (expect_failure != 0) {
    return status == SS_ERROR_HDF5 && error.valid != 0 &&
                   strcmp(error.code, "HDF5_LAYOUT_INVALID") == 0
               ? 0
               : 1;
  }
  if (status != SS_OK || validation.coordinate_values == 0U ||
      validation.cell_rows == 0U) {
    return 1;
  }
  if (require_zero_owner != 0 && validation.zero_owner_ranks == 0U) {
    return 1;
  }
  return 0;
}

static int test_thread_funneled(int rank, int provided) {
  int local_status = 0;
  LOCAL_CHECK(provided >= MPI_THREAD_FUNNELED);
  LOCAL_CHECK(ss_master_thread_guard() == SS_OK);
#if SLOPE_ENABLE_OPENMP
#pragma omp parallel reduction(max : local_status)
  {
    if (omp_get_thread_num() != 0 && ss_master_thread_guard() != SS_ERROR_THREAD) {
      local_status = 1;
    }
  }
#endif
  return local_status;
}

static int test_collective_cancel(MPI_Comm communicator, int rank, int size) {
  SsCancellation cancellation;
  SsStatus status;
  ss_cancellation_init(&cancellation);
  if (rank == size - 1) {
    ss_cancellation_request(&cancellation);
  }
  status = ss_cancellation_collective_safe_point(communicator, &cancellation,
                                                 SS_SAFEPOINT_SETUP);
  return status == SS_ERROR_CANCELLED ? 0 : 1;
}

int main(int argc, char **argv) {
  int provided = MPI_THREAD_SINGLE;
  int rank = -1;
  int size = 0;
  int local_status = 0;
  int global_status = 0;
  const char *test_id;
  const char *fixture = NULL;
  if (argc < 3 || argc > 4 || strcmp(argv[1], "--test-id") != 0) {
    fputs("usage: ss_test_app_mpi --test-id ID [HDF5]\n", stderr);
    return 2;
  }
  test_id = argv[2];
  if (argc == 4) {
    fixture = argv[3];
  }
  if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
    return 3;
  }
  if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS) {
    MPI_Abort(MPI_COMM_WORLD, 4);
    return 4;
  }
  if (strcmp(test_id, "DIA-CONTRACT-MPI-001") == 0) {
    local_status = test_collective_error(MPI_COMM_WORLD, rank, size);
  } else if (strcmp(test_id, "IO-H5-COLLECTIVE-001") == 0) {
    local_status = fixture == NULL ? 1 : test_hdf5(MPI_COMM_WORLD, fixture, 0, 0);
  } else if (strcmp(test_id, "IO-H5-ZEROOWNER-001") == 0) {
    local_status =
        fixture == NULL ? 1 : test_hdf5(MPI_COMM_WORLD, fixture, size > 1, 0);
  } else if (strcmp(test_id, "IO-H5-CORRUPT-001") == 0) {
    local_status = fixture == NULL ? 1 : test_hdf5(MPI_COMM_WORLD, fixture, 0, 1);
  } else if (strcmp(test_id, "APP-THREAD-FUNNELED-001") == 0) {
    local_status = test_thread_funneled(rank, provided);
  } else if (strcmp(test_id, "APP-CANCEL-SAFEPOINT-001") == 0) {
    local_status = test_collective_cancel(MPI_COMM_WORLD, rank, size);
  } else {
    local_status = 1;
  }
  if (MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS) {
    global_status = 1;
  }
  if (MPI_Finalize() != MPI_SUCCESS) {
    global_status = 1;
  }
  return global_status;
}
