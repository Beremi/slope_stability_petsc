/* Deflation basis access, digest and restoration for checkpointing.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "deflation_internal.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/core/util.h"

#include <stdlib.h>
#include <string.h>

/* Test diagnostic. PETSc owns contiguous vector ranges in rank order;
 * each rank contributes only a digest of its local values. */
static SsStatus ss_digest_append_vector(SsSha256 *hash, Vec vector, int rank, int size,
                                        unsigned char *rank_digests) {
  MPI_Comm communicator = PetscObjectComm((PetscObject)vector);
  const PetscScalar *values = NULL;
  PetscInt global_size = 0, local_size = 0;
  SsSha256 slab;
  unsigned char digest[SS_SHA256_SIZE];
  SsStatus status = SS_OK;
  if (VecGetSize(vector, &global_size) != PETSC_SUCCESS ||
      VecGetLocalSize(vector, &local_size) != PETSC_SUCCESS || local_size < 0) {
    status = SS_ERROR_INTERNAL;
  } else if ((size_t)local_size > SIZE_MAX / sizeof(*values)) {
    status = SS_ERROR_OVERFLOW;
  } else if (VecGetArrayRead(vector, &values) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  } else {
    ss_sha256_init(&slab);
    status = ss_sha256_update(&slab, values, (size_t)local_size * sizeof(*values));
    ss_sha256_final_bytes(&slab, digest);
    if (VecRestoreArrayRead(vector, &values) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK &&
      MPI_Gather(digest, SS_SHA256_SIZE, MPI_BYTE, rank_digests, SS_SHA256_SIZE,
                 MPI_BYTE, 0, communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK && rank == 0) {
    status = ss_sha256_update(hash, &global_size, sizeof(global_size));
    if (status == SS_OK) {
      status = ss_sha256_update(hash, rank_digests, (size_t)size * SS_SHA256_SIZE);
    }
  }
  return ss_status_allreduce_max(communicator, status);
}

SsStatus ss_deflation_persistent_digest(SsDeflation *deflation,
                                        char output[SS_SHA256_HEX_CAPACITY]) {
  static const char version[] = "deflation-rank-hash-list-v2";
  SsSha256 hash;
  unsigned char *rank_digests = NULL;
  int rank = -1, size = 0;
  SsStatus status = SS_OK;
  if (deflation == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  MPI_Comm communicator = deflation->communicator;
  if (output == NULL) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS ||
             MPI_Comm_size(communicator, &size) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  } else if (rank == 0) {
    rank_digests = calloc((size_t)size, SS_SHA256_SIZE);
    if (rank_digests == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  ss_sha256_init(&hash);
  (void)ss_sha256_update(&hash, version, sizeof(version));
  (void)ss_sha256_update(&hash, &deflation->persistent_count,
                         sizeof(deflation->persistent_count));
  status = ss_status_allreduce_max(communicator, status);
  for (size_t index = 0U; status == SS_OK && index < deflation->persistent_count;
       ++index) {
    const SsDeflationEntry *entry = &deflation->persistent[index];
    const size_t provenance_size =
        ss_bounded_length(entry->provenance, sizeof(entry->provenance)) + 1U;
    status = ss_status_allreduce_max(
        communicator, ss_sha256_update(&hash, entry->provenance, provenance_size));
    if (status == SS_OK) {
      status = ss_digest_append_vector(&hash, entry->left, rank, size, rank_digests);
    }
    if (status == SS_OK) {
      status = ss_digest_append_vector(&hash, entry->right, rank, size, rank_digests);
    }
  }
  if (status == SS_OK) {
    if (rank == 0) {
      ss_sha256_final(&hash, output);
    }
    if (MPI_Bcast(output, SS_SHA256_HEX_CAPACITY, MPI_CHAR, 0, communicator) !=
        MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  free(rank_digests);
  return status;
}

SsStatus ss_deflation_persistent_count(const SsDeflation *deflation, size_t *count) {
  if (deflation == NULL || count == NULL || deflation->transaction_open != 0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *count = deflation->persistent_count;
  return SS_OK;
}

SsStatus ss_deflation_persistent_entry(const SsDeflation *deflation, size_t index,
                                       Vec *left, Vec *right, const char **provenance) {
  const SsDeflationEntry *entry;
  if (deflation == NULL || left == NULL || right == NULL || provenance == NULL ||
      deflation->transaction_open != 0 || index >= deflation->persistent_count) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  entry = &deflation->persistent[index];
  if (entry->left == NULL || entry->right == NULL || entry->provenance[0] == '\0') {
    return SS_ERROR_INTEGRITY;
  }
  *left = entry->left;
  *right = entry->right;
  *provenance = entry->provenance;
  return SS_OK;
}

SsStatus ss_deflation_persistent_state_entry(
    const SsDeflation *deflation, size_t index, Vec *left, Vec *right, Vec *source,
    PetscReal *source_weight, SsDeflationPersistentSourceKind *source_kind) {
  const SsDeflationEntry *entry;
  if (deflation == NULL || left == NULL || right == NULL || source == NULL ||
      source_weight == NULL || source_kind == NULL ||
      deflation->transaction_open != 0 || index >= deflation->persistent_count) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  entry = &deflation->persistent[index];
  if (entry->left == NULL || entry->right == NULL || entry->provenance[0] == '\0' ||
      !ss_is_finite(entry->source_weight) || entry->source_weight <= 0.0) {
    return SS_ERROR_INTEGRITY;
  }
  *left = entry->left;
  *right = entry->right;
  *source = entry->source != NULL ? entry->source : entry->right;
  *source_weight = entry->source_weight;
  if (strcmp(entry->provenance, "nls:accepted-state") == 0) {
    *source_kind = SS_DEFLATION_PERSISTENT_SOURCE_ACCEPTED_STATE;
  } else if (strncmp(entry->provenance, "compress:snapshot-pod",
                     sizeof("compress:snapshot-pod") - 1U) == 0) {
    *source_kind = SS_DEFLATION_PERSISTENT_SOURCE_SNAPSHOT_POD;
  } else {
    *source_kind = SS_DEFLATION_PERSISTENT_SOURCE_OTHER;
  }
  return SS_OK;
}

SsStatus ss_deflation_active_entry(const SsDeflation *deflation, size_t index,
                                   Vec *left, Vec *right, const char **provenance) {
  const SsDeflationEntry *entry;
  if (deflation == NULL || left == NULL || right == NULL || provenance == NULL ||
      index >= deflation->active_count) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  entry = ss_deflation_const_entry(deflation, index);
  if (entry->left == NULL || entry->right == NULL || entry->provenance[0] == '\0') {
    return SS_ERROR_INTEGRITY;
  }
  *left = entry->left;
  *right = entry->right;
  *provenance = entry->provenance;
  return SS_OK;
}

static SsStatus ss_deflation_restore_persistent_impl(
    SsDeflation *deflation, const Vec *left, const Vec *right, const Vec *source,
    const PetscReal *source_weights,
    const SsDeflationPersistentSourceKind *source_kinds, size_t count,
    const char *uniform_provenance, Mat operator_handle, uint64_t operator_generation,
    uint64_t accepted_generation) {
  size_t index;
  SsStatus status = SS_OK;
  if (deflation == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if ((count != 0U && (left == NULL || right == NULL)) ||
      (count != 0U && uniform_provenance == NULL &&
       (source == NULL || source_weights == NULL || source_kinds == NULL)) ||
      (uniform_provenance != NULL && uniform_provenance[0] == '\0') ||
      accepted_generation == 0U ||
      (count != 0U && (operator_handle == NULL || operator_generation == 0U)) ||
      (operator_handle != NULL && operator_generation == 0U) ||
      count > deflation->policy.dimension_cap || deflation->transaction_open != 0 ||
      deflation->persistent_count != 0U || deflation->provisional_count != 0U) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else if (uniform_provenance != NULL &&
             ss_bounded_length(uniform_provenance, SS_DEFLATION_PROVENANCE_CAPACITY) ==
                 SS_DEFLATION_PROVENANCE_CAPACITY) {
    status = SS_ERROR_LIMIT;
  }
  status = ss_status_allreduce_max(deflation->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  if (count == 0U) {
    deflation->accepted_generation = accepted_generation;
    return operator_handle == NULL
               ? SS_OK
               : ss_deflation_refresh(deflation, operator_handle, operator_generation);
  }

  for (index = 0U; index < count; ++index) {
    if (left[index] == NULL || right[index] == NULL ||
        (uniform_provenance == NULL &&
         (source[index] == NULL || !ss_is_finite(source_weights[index]) ||
          source_weights[index] <= 0.0 ||
          source_kinds[index] < SS_DEFLATION_PERSISTENT_SOURCE_OTHER ||
          source_kinds[index] > SS_DEFLATION_PERSISTENT_SOURCE_SNAPSHOT_POD))) {
      status = SS_ERROR_INVALID_ARGUMENT;
      break;
    }
  }

  status = ss_status_allreduce_max(deflation->communicator, status);
  for (index = 0U; status == SS_OK && index < count; ++index) {
    SsDeflationEntry *entry = &deflation->persistent[index];
    const Vec compression_source = source != NULL ? source[index] : right[index];
    const char *entry_provenance = uniform_provenance;
    size_t provenance_length;
    if (entry_provenance == NULL) {
      switch (source_kinds[index]) {
      case SS_DEFLATION_PERSISTENT_SOURCE_ACCEPTED_STATE:
        entry_provenance = "nls:accepted-state";
        break;
      case SS_DEFLATION_PERSISTENT_SOURCE_SNAPSHOT_POD:
        entry_provenance = "compress:snapshot-pod";
        break;
      case SS_DEFLATION_PERSISTENT_SOURCE_OTHER:
        entry_provenance = "checkpoint:ordinary-source";
        break;
      default:
        status = SS_ERROR_INVALID_ARGUMENT;
        continue;
      }
    }
    provenance_length =
        ss_bounded_length(entry_provenance, SS_DEFLATION_PROVENANCE_CAPACITY);
    status = ss_deflation_vector_copy(deflation, left[index], &entry->left);
    if (status == SS_OK) {
      status = ss_deflation_vector_copy(deflation, right[index], &entry->right);
    }
    if (status == SS_OK &&
        deflation->policy.compression != SS_DEFLATION_COMPRESSION_NONE) {
      status = ss_deflation_vector_copy(deflation, compression_source, &entry->source);
    }
    if (status == SS_OK) {
      entry->source_weight =
          source_weights != NULL ? source_weights[index] : (PetscReal)1.0;
      memcpy(entry->provenance, entry_provenance, provenance_length + 1U);
    }
  }

  if (status == SS_OK) {
    deflation->persistent_count = count;
    deflation->active_skip_extra = 0U;
    ss_deflation_update_active_count(deflation);
    status = ss_deflation_refresh(deflation, operator_handle, operator_generation);
  }
  if (status == SS_OK && deflation->active_count != count) {
    status = SS_ERROR_INTEGRITY;
  }

  if (status != SS_OK) {
    ss_deflation_clear_factorization(deflation);
    for (index = 0U; index < count; ++index) {
      ss_deflation_entry_clear(&deflation->persistent[index]);
    }
    deflation->persistent_count = 0U;
    deflation->active_count = 0U;
    deflation->active_skip_extra = 0U;
    deflation->operator_generation = 0U;
    deflation->operator_state = 0;
    if (deflation->operator_handle != NULL) {
      (void)MatDestroy(&deflation->operator_handle);
    }
    return status;
  }

  deflation->accepted_generation = accepted_generation;
  return status;
}

SsStatus ss_deflation_restore_persistent(SsDeflation *deflation, const Vec *left,
                                         const Vec *right, size_t count,
                                         const char *provenance, Mat operator_handle,
                                         uint64_t operator_generation,
                                         uint64_t accepted_generation) {
  return ss_deflation_restore_persistent_impl(deflation, left, right, NULL, NULL, NULL,
                                              count, provenance, operator_handle,
                                              operator_generation, accepted_generation);
}

SsStatus ss_deflation_restore_persistent_state(
    SsDeflation *deflation, const Vec *left, const Vec *right, const Vec *source,
    const PetscReal *source_weights,
    const SsDeflationPersistentSourceKind *source_kinds, size_t count,
    Mat operator_handle, uint64_t operator_generation, uint64_t accepted_generation) {
  return ss_deflation_restore_persistent_impl(
      deflation, left, right, source, source_weights, source_kinds, count, NULL,
      operator_handle, operator_generation, accepted_generation);
}
