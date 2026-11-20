/* Native identities, collective error selection and runtime diagnostics.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/diagnostics/diagnostics.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/core/solver_options.h"

#include "slope_internal/build/config.h"

#include <petscsys.h>

#if SLOPE_ENABLE_OPENMP
#include <omp.h>
#endif

#include <limits.h>
#if defined(__GLIBC__)
#include <malloc.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

static SsStatus ss_prefixed_digest(const unsigned char *data, size_t size,
                                   char output[SS_ID_TEXT_CAPACITY]) {
  char digest[SS_SHA256_HEX_CAPACITY];
  if (ss_sha256_bytes(data, size, digest) != SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  if (snprintf(output, SS_ID_TEXT_CAPACITY, "sha256:%s", digest) != 71) {
    return SS_ERROR_INTERNAL;
  }
  return SS_OK;
}

SsStatus ss_identity_link(const SsIdentityInputs *inputs,
                          char output[SS_ID_TEXT_CAPACITY]) {
  char projection[1024];
  int written;
  if (inputs == NULL || output == NULL || inputs->source_id == NULL ||
      inputs->build_id == NULL || inputs->executable_id == NULL ||
      inputs->problem_id == NULL || inputs->run_id == NULL ||
      inputs->machine_id == NULL || inputs->launcher_id == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  written = snprintf(
      projection, sizeof(projection),
      "build=%s\nexecutable=%s\nlauncher=%s\nmachine=%s\nproblem=%s\nrun=%s\n"
      "source=%s\n",
      inputs->build_id, inputs->executable_id, inputs->launcher_id, inputs->machine_id,
      inputs->problem_id, inputs->run_id, inputs->source_id);
  if (written < 0 || (size_t)written >= sizeof(projection)) {
    return SS_ERROR_LIMIT;
  }
  return ss_prefixed_digest((const unsigned char *)projection, (size_t)written, output);
}

SsStatus ss_identity_from_text(const char *namespace_name, const char *text,
                               char output[SS_ID_TEXT_CAPACITY]) {
  char projection[1024];
  int written;
  if (namespace_name == NULL || text == NULL || output == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  written = snprintf(projection, sizeof(projection), "%s:%s", namespace_name, text);
  if (written < 0 || (size_t)written >= sizeof(projection)) {
    return SS_ERROR_LIMIT;
  }
  return ss_prefixed_digest((const unsigned char *)projection, (size_t)written, output);
}

typedef struct {
  SsErrorKey key;
  char code[SS_ERROR_CODE_CAPACITY];
  int owner;
} SsErrorCandidate;

/* A full-width key cannot fit in a predefined MPI_MINLOC pair. The final
 * tie-break reproduces the first matching record in communicator order. */
static void ss_error_minloc(void *input, void *inout, int *count,
                            MPI_Datatype *datatype) {
  const SsErrorCandidate *sources = input;
  SsErrorCandidate *targets = inout;
  (void)datatype;
  for (int i = 0; i < *count; ++i) {
    const SsErrorCandidate *source = &sources[i];
    SsErrorCandidate *target = &targets[i];
    if (source->owner == INT_MAX) {
      continue;
    }
    int order = ss_error_key_compare(&source->key, &target->key);
    if (order == 0) {
      order = strcmp(source->code, target->code);
    }
    if (target->owner == INT_MAX || order < 0 ||
        (order == 0 && source->owner < target->owner)) {
      *target = *source;
    }
  }
}

SsStatus ss_collective_error_select(MPI_Comm communicator, const SsErrorRecord *local,
                                    SsErrorRecord *selected) {
  enum { NO_ERROR, HAS_ERROR, INVALID_ARGUMENT };
  const int local_flag = local == NULL || selected == NULL ? INVALID_ARGUMENT
                         : local->valid != 0               ? HAS_ERROR
                                                           : NO_ERROR;
  int flag = NO_ERROR;
  if (communicator == MPI_COMM_NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (MPI_Allreduce(&local_flag, &flag, 1, MPI_INT, MPI_MAX, communicator) !=
      MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (flag == INVALID_ARGUMENT) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (flag == NO_ERROR) {
    memset(selected, 0, sizeof(*selected));
    return SS_OK;
  }

  SsErrorCandidate candidate = {.owner = INT_MAX};
  SsErrorCandidate winner = {.owner = INT_MAX};
  MPI_Datatype datatype = MPI_DATATYPE_NULL;
  MPI_Op operation = MPI_OP_NULL;
  int rank = 0;
  SsStatus status = SS_OK;
  if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS ||
      MPI_Type_contiguous((int)sizeof(candidate), MPI_BYTE, &datatype) != MPI_SUCCESS ||
      MPI_Type_commit(&datatype) != MPI_SUCCESS ||
      MPI_Op_create(ss_error_minloc, 1, &operation) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (local->valid != 0) {
    candidate.key = local->key;
    memcpy(candidate.code, local->code, sizeof(candidate.code));
    candidate.owner = rank;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK && MPI_Allreduce(&candidate, &winner, 1, datatype, operation,
                                       communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (operation != MPI_OP_NULL) {
    (void)MPI_Op_free(&operation);
  }
  if (datatype != MPI_DATATYPE_NULL) {
    (void)MPI_Type_free(&datatype);
  }
  if (status != SS_OK) {
    return status;
  }
  if (rank == winner.owner) {
    *selected = *local;
  }
  return MPI_Bcast(selected, (int)sizeof(*selected), MPI_BYTE, winner.owner,
                   communicator) == MPI_SUCCESS
             ? SS_OK
             : SS_ERROR_INTERNAL;
}

SsStatus ss_master_thread_guard(void) {
#if SLOPE_ENABLE_OPENMP
  if (omp_in_parallel() != 0 && omp_get_thread_num() != 0) {
    return SS_ERROR_THREAD;
  }
#endif
  return SS_OK;
}

SsStatus ss_memory_view(MPI_Comm communicator, const char *stage) {
  SsSolverOptionValue setting;
  PetscLogDouble resident_bytes = 0.0;
  PetscLogDouble petsc_malloc_bytes = 0.0;
  struct rusage usage;
  unsigned long long heap_used_bytes = 0U;
  unsigned long long heap_free_bytes = 0U;
  unsigned long long heap_mmap_bytes = 0U;
  int rank = -1;
  if (stage == NULL || MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS || rank < 0 ||
      ss_solver_expert_option(SS_SOLVER_EXPERT_memory_view, &setting) != SS_OK) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (setting.boolean == PETSC_FALSE) {
    return SS_OK;
  }
  memset(&usage, 0, sizeof(usage));
  if (PetscMemoryGetCurrentUsage(&resident_bytes) != PETSC_SUCCESS ||
      PetscMallocGetCurrentUsage(&petsc_malloc_bytes) != PETSC_SUCCESS ||
      getrusage(RUSAGE_SELF, &usage) != 0) {
    return SS_ERROR_INTERNAL;
  }
#if defined(__GLIBC__)
  {
    const struct mallinfo2 information = mallinfo2();
    heap_used_bytes = (unsigned long long)information.uordblks;
    heap_free_bytes = (unsigned long long)information.fordblks;
    heap_mmap_bytes = (unsigned long long)information.hblkhd;
  }
#endif
  if (PetscSynchronizedPrintf(
          communicator,
          "{\"kind\":\"slope_stability.memory_snapshot\",\"stage\":\"%s\","
          "\"rank\":%d,\"rss_bytes\":%.0f,\"peak_rss_kib\":%llu,"
          "\"petsc_malloc_bytes\":%.0f,\"heap_used_bytes\":%llu,"
          "\"heap_free_bytes\":%llu,\"heap_mmap_bytes\":%llu}\n",
          stage, rank, (double)resident_bytes, (unsigned long long)usage.ru_maxrss,
          (double)petsc_malloc_bytes, heap_used_bytes, heap_free_bytes,
          heap_mmap_bytes) != PETSC_SUCCESS ||
      PetscSynchronizedFlush(communicator, PETSC_STDOUT) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  return SS_OK;
}

SsStatus ss_memory_trim(void) {
#if defined(__GLIBC__)
  (void)malloc_trim(0U);
#endif
  return SS_OK;
}

#ifndef NDEBUG
static int ss_debug_stage_enabled(void) {
  const char *value = getenv("SLOPE_DEBUG_STAGE_DIGEST");
  return value != NULL && strcmp(value, "1") == 0;
}

static void ss_debug_stage_print(const char *stage, size_t ordinal, PetscInt begin,
                                 PetscInt end, SsSha256 *hash) {
  int rank = -1;
  char digest[SS_SHA256_HEX_CAPACITY];
  (void)MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  ss_sha256_final(hash, digest);
  (void)fprintf(stderr, "stage_digest %s %zu rank=%d rows=%lld:%lld sha256=%s\n", stage,
                ordinal, rank, (long long)begin, (long long)end, digest);
}

void ss_debug_stage_matrix(const char *stage, size_t ordinal, Mat matrix) {
  PetscBool has_rows = PETSC_FALSE;
  PetscInt begin = 0, end = 0;
  SsSha256 hash;
  if (!ss_debug_stage_enabled() ||
      MatHasOperation(matrix, MATOP_GET_ROW, &has_rows) != PETSC_SUCCESS || !has_rows ||
      MatGetOwnershipRange(matrix, &begin, &end) != PETSC_SUCCESS) {
    return;
  }
  ss_sha256_init(&hash);
  for (PetscInt row = begin; row < end; ++row) {
    PetscInt columns = 0;
    const PetscInt *indices = NULL;
    const PetscScalar *values = NULL;
    if (MatGetRow(matrix, row, &columns, &indices, &values) != PETSC_SUCCESS) {
      return;
    }
    (void)ss_sha256_update(&hash, &columns, sizeof(columns));
    (void)ss_sha256_update(&hash, indices, (size_t)columns * sizeof(*indices));
    (void)ss_sha256_update(&hash, values, (size_t)columns * sizeof(*values));
    if (MatRestoreRow(matrix, row, &columns, &indices, &values) != PETSC_SUCCESS) {
      return;
    }
  }
  ss_debug_stage_print(stage, ordinal, begin, end, &hash);
}

void ss_debug_stage_vector(const char *stage, size_t ordinal, Vec vector) {
  PetscInt begin = 0, end = 0;
  const PetscScalar *values = NULL;
  SsSha256 hash;
  if (!ss_debug_stage_enabled() ||
      VecGetOwnershipRange(vector, &begin, &end) != PETSC_SUCCESS ||
      VecGetArrayRead(vector, &values) != PETSC_SUCCESS) {
    return;
  }
  ss_sha256_init(&hash);
  (void)ss_sha256_update(&hash, values, (size_t)(end - begin) * sizeof(*values));
  if (VecRestoreArrayRead(vector, &values) == PETSC_SUCCESS) {
    ss_debug_stage_print(stage, ordinal, begin, end, &hash);
  }
}
#endif
