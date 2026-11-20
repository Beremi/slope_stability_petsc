/* Deflation lifecycle, refresh and transaction commit or rollback.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "deflation_internal.h"
#include "slope_internal/core/core.h"
#include "slope_internal/core/util.h"

#include <petsclog.h>
#include <petscsys.h>

#include <stdlib.h>
#include <string.h>

static SsStatus ss_deflation_refresh_retry(SsDeflation *deflation, Mat operator_handle,
                                           uint64_t operator_generation);

SsStatus ss_deflation_create(MPI_Comm communicator, const SsDeflationPolicy *policy,
                             SsDeflation **result) {
  SsDeflation *deflation = NULL;
  SsStatus status = SS_OK;
  if (communicator == MPI_COMM_NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (policy == NULL || result == NULL || policy->dimension_cap == 0U ||
      !ss_is_finite(policy->norm_tolerance) || !ss_is_finite(policy->pivot_tolerance) ||
      !ss_is_finite(policy->condition_limit) ||
      !ss_is_finite(policy->invariant_tolerance) || policy->norm_tolerance <= 0.0 ||
      policy->pivot_tolerance <= 0.0 || policy->condition_limit <= 1.0 ||
      policy->invariant_tolerance <= 0.0 || policy->reorthogonalization_passes == 0U ||
      policy->reorthogonalization_passes > 4U ||
      (policy->projector != SS_DEFLATION_PROJECTOR_GENERAL_LEFT_RIGHT &&
       policy->projector != SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL) ||
      (policy->compression != SS_DEFLATION_COMPRESSION_NONE &&
       policy->compression != SS_DEFLATION_COMPRESSION_SNAPSHOT_POD &&
       policy->compression != SS_DEFLATION_COMPRESSION_OPERATOR_SVD &&
       policy->compression != SS_DEFLATION_COMPRESSION_SNAPSHOT_POD_TOTAL &&
       policy->compression != SS_DEFLATION_COMPRESSION_SYSTEM_SOLUTION_WINDOW) ||
      (policy->compression == SS_DEFLATION_COMPRESSION_NONE &&
       policy->compression_dimension != 0U) ||
      (policy->compression != SS_DEFLATION_COMPRESSION_NONE &&
       (policy->projector != SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL ||
        policy->compression_dimension == 0U ||
        policy->compression_dimension > policy->dimension_cap ||
        !ss_is_finite(policy->compression_history_decay) ||
        policy->compression_history_decay < 0.0 ||
        policy->compression_history_decay > 1.0))) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    return status;
  }
  *result = NULL;
  deflation = calloc(1U, sizeof(*deflation));
  if (deflation != NULL) {
    deflation->persistent =
        calloc(policy->dimension_cap, sizeof(*deflation->persistent));
    deflation->provisional =
        calloc(policy->dimension_cap, sizeof(*deflation->provisional));
  }
  status = ss_status_allreduce_max(communicator, deflation && deflation->persistent &&
                                                         deflation->provisional
                                                     ? SS_OK
                                                     : SS_ERROR_OUT_OF_MEMORY);
  if (status != SS_OK) {
    if (deflation != NULL) {
      free(deflation->persistent);
      free(deflation->provisional);
      free(deflation);
    }
    return status;
  }
  if (MPI_Comm_dup(communicator, &deflation->communicator) != MPI_SUCCESS) {
    free(deflation->provisional);
    free(deflation->persistent);
    free(deflation);
    return SS_ERROR_INTERNAL;
  }
  deflation->policy = *policy;
  if (policy->compression != SS_DEFLATION_COMPRESSION_NONE &&
      ss_deflation_compression_event < 0 &&
      PetscLogEventRegister("SlopeDeflCompress", 0, &ss_deflation_compression_event) !=
          PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(deflation->communicator, status);
  if (status != SS_OK) {
    (void)MPI_Comm_free(&deflation->communicator);
    free(deflation->provisional);
    free(deflation->persistent);
    free(deflation);
    return status;
  }
  deflation->minimum_relative_pivot = 1.0;
  deflation->condition_estimate = 1.0;
  *result = deflation;
  return SS_OK;
}

SsStatus ss_deflation_policy_memory_estimate(const SsDeflationPolicy *policy,
                                             PetscInt global_rows, size_t *vector_bytes,
                                             size_t *dense_bytes_per_rank) {
  if (policy == NULL || policy->dimension_cap == 0U || global_rows <= 0 ||
      vector_bytes == NULL || dense_bytes_per_rank == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  const size_t cap = policy->dimension_cap;
  const int legacy = policy->projector == SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL;
  const int compression = policy->compression != SS_DEFLATION_COMPRESSION_NONE;
  /* Two capped stores, cached A*Z, legacy transposed images and a snapshot.
   * Include a second set of normalized columns during orthogonalization and
   * POD/SVD modes and images when compression is enabled. */
  const size_t vectors_per_column = compression ? 20U : legacy ? 12U : 6U;
  const size_t dense_scalars_per_entry = compression ? 24U : 6U;
  size_t vectors;
  size_t entries;
  if (ss_checked_mul_size(vectors_per_column, cap, &vectors) != SS_OK ||
      vectors > SIZE_MAX - 8U ||
      ss_checked_mul_size(vectors + 8U, (size_t)global_rows, &entries) != SS_OK ||
      ss_checked_mul_size(entries, sizeof(PetscScalar), vector_bytes) != SS_OK ||
      ss_checked_mul_size(cap, cap, &entries) != SS_OK ||
      ss_checked_mul_size(entries, dense_scalars_per_entry * sizeof(PetscScalar),
                          dense_bytes_per_rank) != SS_OK) {
    return SS_ERROR_OVERFLOW;
  }
  return SS_OK;
}

SsStatus ss_deflation_memory_estimate(const SsDeflation *deflation,
                                      PetscInt global_rows, size_t *vector_bytes,
                                      size_t *dense_bytes_per_rank) {
  return deflation == NULL
             ? SS_ERROR_INVALID_ARGUMENT
             : ss_deflation_policy_memory_estimate(&deflation->policy, global_rows,
                                                   vector_bytes, dense_bytes_per_rank);
}

SsStatus ss_deflation_destroy(SsDeflation **deflation_pointer) {
  SsDeflation *deflation;
  size_t index;
  if (deflation_pointer == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  deflation = *deflation_pointer;
  if (deflation == NULL) {
    return SS_OK;
  }
  ss_deflation_clear_factorization(deflation);
  ss_deflation_destroy_work(deflation);
  for (index = 0U; index < deflation->policy.dimension_cap; ++index) {
    ss_deflation_entry_clear(&deflation->persistent[index]);
    ss_deflation_entry_clear(&deflation->provisional[index]);
  }
  free(deflation->provisional);
  free(deflation->persistent);
  ss_deflation_entry_array_destroy(&deflation->transaction_base_persistent,
                                   deflation->policy.dimension_cap);
  if (deflation->operator_handle != NULL) {
    (void)MatDestroy(&deflation->operator_handle);
  }
  if (deflation->transaction_base_operator_handle != NULL) {
    (void)MatDestroy(&deflation->transaction_base_operator_handle);
  }
  if (deflation->communicator != MPI_COMM_NULL) {
    (void)MPI_Comm_free(&deflation->communicator);
  }
  free(deflation);
  *deflation_pointer = NULL;
  return SS_OK;
}

SsStatus ss_deflation_refresh(SsDeflation *deflation, Mat operator_handle,
                              uint64_t operator_generation) {
  PetscObjectState current_state = 0;
  size_t dimension;
  size_t column;
  size_t row;
  PetscScalar *batched_coarse = NULL;
  SsStatus status;
  if (deflation == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = SS_OK;
  if (operator_handle == NULL || operator_generation == 0U) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else if (MatGetState(operator_handle, &current_state) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  } else if (deflation->operator_handle == operator_handle &&
             deflation->operator_generation == operator_generation &&
             deflation->operator_state != current_state) {
    status = SS_ERROR_INTEGRITY;
  }
  status = ss_status_allreduce_max(deflation->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  const int ready = deflation->operator_handle == operator_handle &&
                    deflation->operator_generation == operator_generation &&
                    deflation->operator_state == current_state &&
                    deflation->factors_ready && deflation->invariants_validated;
  int all_ready = 0;
  if (MPI_Allreduce(&ready, &all_ready, 1, MPI_INT, MPI_MIN, deflation->communicator) !=
      MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (all_ready) {
    return SS_OK;
  }
  if (deflation->operator_handle != operator_handle) {
    if (PetscObjectReference((PetscObject)operator_handle) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    } else if (MatDestroy(&deflation->operator_handle) != PETSC_SUCCESS) {
      (void)MatDestroy(&operator_handle);
      status = SS_ERROR_INTERNAL;
    } else {
      deflation->operator_handle = operator_handle;
    }
  }
  status = ss_status_allreduce_max(deflation->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  if (deflation->policy.projector == SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL) {
    return ss_deflation_refresh_legacy_aorth(deflation, operator_handle, current_state,
                                             operator_generation);
  }
  if (deflation->refresh_retry_depth == 0U) {
    deflation->active_skip_extra = 0U;
  }
  ss_deflation_update_active_count(deflation);
  dimension = deflation->active_count;
  ss_deflation_clear_factorization(deflation);
  if (dimension == 0U) {
    deflation->factors_ready = 1;
    deflation->invariants_validated = 1;
    deflation->minimum_relative_pivot = 1.0;
    deflation->condition_estimate = 1.0;
    deflation->maximum_invariant_error = 0.0;
    deflation->operator_state = current_state;
    deflation->operator_generation = operator_generation;
    ++deflation->refresh_count;
    return SS_OK;
  }
  size_t coarse_bytes;
  if (dimension > (size_t)PETSC_MAX_INT ||
      ss_checked_product_size(
          (size_t[]){dimension, dimension, sizeof(*deflation->coarse)}, 3U,
          &coarse_bytes) != SS_OK) {
    return SS_ERROR_OVERFLOW;
  }
  deflation->coarse = (PetscScalar *)calloc(1U, coarse_bytes);
  deflation->lu = (PetscScalar *)calloc(1U, coarse_bytes);
  deflation->dots = (PetscScalar *)calloc(dimension, sizeof(*deflation->dots));
  deflation->coefficients =
      (PetscScalar *)calloc(dimension, sizeof(*deflation->coefficients));
  deflation->pivots = (PetscInt *)calloc(dimension, sizeof(*deflation->pivots));
  deflation->active_left = (Vec *)calloc(dimension, sizeof(*deflation->active_left));
  deflation->active_right = (Vec *)calloc(dimension, sizeof(*deflation->active_right));
  deflation->active_az = (Vec *)calloc(dimension, sizeof(*deflation->active_az));
  status = ss_status_allreduce_max(
      deflation->communicator,
      (deflation->coarse == NULL || deflation->lu == NULL || deflation->dots == NULL ||
       deflation->coefficients == NULL || deflation->pivots == NULL ||
       deflation->active_left == NULL || deflation->active_right == NULL ||
       deflation->active_az == NULL)
          ? SS_ERROR_OUT_OF_MEMORY
          : SS_OK);
  if (status != SS_OK) {
    ss_deflation_clear_factorization(deflation);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (row = 0U; row < dimension; ++row) {
    SsDeflationEntry *entry = ss_deflation_entry(deflation, row);
    deflation->active_left[row] = entry->left;
    deflation->active_right[row] = entry->right;
    /* The basis vectors and their layouts live for the duration of the
     * recycle entry.  Only A's numeric values change between Newton tangents,
     * so retain the A*Z storage and overwrite it instead of collectively
     * destroying and allocating every cached image on every refresh. */
    if (entry->az == NULL &&
        ss_deflation_petsc_status(deflation, VecDuplicate(entry->right, &entry->az)) !=
            SS_OK) {
      ss_deflation_clear_factorization(deflation);
      return SS_ERROR_OUT_OF_MEMORY;
    }
    if (ss_deflation_petsc_status(
            deflation, MatMult(operator_handle, entry->right, entry->az)) != SS_OK) {
      ss_deflation_clear_factorization(deflation);
      return SS_ERROR_INTERNAL;
    }
    deflation->active_az[row] = entry->az;
  }
  batched_coarse =
      (PetscScalar *)calloc(dimension * dimension, sizeof(*batched_coarse));
  status = ss_status_allreduce_max(
      deflation->communicator, batched_coarse != NULL ? SS_OK : SS_ERROR_OUT_OF_MEMORY);
  if (status != SS_OK) {
    free(batched_coarse);
    ss_deflation_clear_factorization(deflation);
    return status;
  }
  status = ss_deflation_gram(deflation, deflation->active_az, deflation->active_left,
                             dimension, batched_coarse);
  if (status != SS_OK) {
    free(batched_coarse);
    ss_deflation_clear_factorization(deflation);
    return status;
  }
  for (column = 0U; column < dimension; ++column) {
    for (row = 0U; row < dimension; ++row) {
      deflation->coarse[row * dimension + column] =
          batched_coarse[column * dimension + row];
    }
  }
  free(batched_coarse);
  status = ss_status_allreduce_max(deflation->communicator,
                                   ss_deflation_factorize(deflation));
  if (status != SS_OK) {
    if (status == SS_ERROR_INTEGRITY && deflation->provisional_count != 0U) {
      ss_deflation_clear_factorization(deflation);
      (void)ss_deflation_drop_newest_provisional(deflation,
                                                 SS_DEFLATION_DROP_ILL_CONDITIONED);
      return ss_deflation_refresh_retry(deflation, operator_handle,
                                        operator_generation);
    }
    if (status == SS_ERROR_INTEGRITY && deflation->active_count != 0U) {
      ss_deflation_clear_factorization(deflation);
      (void)ss_deflation_drop_oldest_active_view(deflation,
                                                 SS_DEFLATION_DROP_ILL_CONDITIONED);
      return ss_deflation_refresh_retry(deflation, operator_handle,
                                        operator_generation);
    }
    deflation->factors_ready = 0;
    return status;
  }
  for (row = 0U; row < 4U; ++row) {
    if (deflation->work[row] == NULL &&
        ss_deflation_petsc_status(deflation,
                                  VecDuplicate(ss_deflation_entry(deflation, 0U)->right,
                                               &deflation->work[row])) != SS_OK) {
      ss_deflation_clear_factorization(deflation);
      return SS_ERROR_OUT_OF_MEMORY;
    }
  }
  deflation->operator_state = current_state;
  deflation->operator_generation = operator_generation;
  deflation->factors_ready = 1;
  ++deflation->refresh_count;
  {
    PetscReal invariant_error = 0.0;
    status = ss_status_allreduce_max(
        deflation->communicator,
        ss_deflation_coarse_invariant_error(deflation, &invariant_error));
    if (status != SS_OK || invariant_error > deflation->policy.invariant_tolerance) {
      ss_deflation_clear_factorization(deflation);
      if ((status == SS_OK || status == SS_ERROR_INTEGRITY) &&
          deflation->provisional_count != 0U) {
        (void)ss_deflation_drop_newest_provisional(deflation,
                                                   SS_DEFLATION_DROP_ILL_CONDITIONED);
        return ss_deflation_refresh_retry(deflation, operator_handle,
                                          operator_generation);
      }
      if ((status == SS_OK || status == SS_ERROR_INTEGRITY) &&
          deflation->active_count != 0U) {
        (void)ss_deflation_drop_oldest_active_view(deflation,
                                                   SS_DEFLATION_DROP_ILL_CONDITIONED);
        return ss_deflation_refresh_retry(deflation, operator_handle,
                                          operator_generation);
      }
      return status != SS_OK ? status : SS_ERROR_INTEGRITY;
    }
  }
  return SS_OK;
}

static SsStatus ss_deflation_refresh_retry(SsDeflation *deflation, Mat operator_handle,
                                           uint64_t operator_generation) {
  SsStatus status;
  ++deflation->refresh_retry_depth;
  status = ss_deflation_refresh(deflation, operator_handle, operator_generation);
  --deflation->refresh_retry_depth;
  return status;
}

SsStatus ss_deflation_commit(SsDeflation *deflation, uint64_t accepted_generation) {
  SsDeflationEntry *retained;
  SsStatus status = SS_OK;
  const size_t combined =
      deflation == NULL ? 0U
                        : deflation->persistent_count + deflation->provisional_count;
  size_t retained_count;
  size_t retained_start;
  size_t index;
  if (deflation == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (accepted_generation == 0U) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else if (!deflation->transaction_open ||
             ((!deflation->factors_ready || !deflation->invariants_validated) &&
              deflation->policy.projector !=
                  SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL)) {
    status = SS_ERROR_INVALID_STATE;
  }
  status = ss_status_allreduce_max(deflation->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  /* A legacy A-orthogonal transaction may finish by appending raw accepted
   * state/Krylov columns after its last solve.  They are storage-only at this
   * boundary and are explicitly reorthogonalized against the next tangent
   * before any projector action. */
  retained_count = combined < deflation->policy.dimension_cap
                       ? combined
                       : deflation->policy.dimension_cap;
  retained_start = combined - retained_count;
  retained =
      (SsDeflationEntry *)calloc(deflation->policy.dimension_cap, sizeof(*retained));
  status = ss_status_allreduce_max(deflation->communicator,
                                   retained ? SS_OK : SS_ERROR_OUT_OF_MEMORY);
  if (status != SS_OK) {
    free(retained);
    return status;
  }
  for (index = 0U; index < retained_count; ++index) {
    SsDeflationEntry *source =
        ss_deflation_combined_entry(deflation, retained_start + index);
    retained[index] = *source;
    memset(source, 0, sizeof(*source));
  }
  for (index = 0U; index < deflation->policy.dimension_cap; ++index) {
    ss_deflation_entry_clear(&deflation->persistent[index]);
    ss_deflation_entry_clear(&deflation->provisional[index]);
  }
  free(deflation->persistent);
  deflation->persistent = retained;
  deflation->persistent_count = retained_count;
  deflation->provisional_count = 0U;
  ss_deflation_update_active_count(deflation);
  deflation->accepted_generation = accepted_generation;
  status = ss_deflation_petsc_status(
      deflation, MatDestroy(&deflation->transaction_base_operator_handle));
  deflation->transaction_generation = 0U;
  deflation->transaction_base_operator_generation = 0U;
  deflation->transaction_open = 0;
  deflation->transaction_basis_restored = 0;
  ss_deflation_entry_array_destroy(&deflation->transaction_base_persistent,
                                   deflation->policy.dimension_cap);
  deflation->transaction_base_persistent_count = 0U;
  return status;
}

SsStatus ss_deflation_rollback(SsDeflation *deflation) {
  Mat restore_operator = NULL;
  size_t index;
  uint64_t restore_generation;
  SsStatus status = SS_OK;
  if (deflation == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_status_allreduce_max(
      deflation->communicator,
      deflation->transaction_open ? SS_OK : SS_ERROR_INVALID_STATE);
  if (status != SS_OK) {
    return status;
  }
  restore_generation = deflation->transaction_base_operator_generation;
  restore_operator = deflation->transaction_base_operator_handle;
  deflation->transaction_base_operator_handle = NULL;
  if (deflation->policy.projector == SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL) {
    status = ss_deflation_restore_legacy_transaction_basis(deflation);
  } else {
    for (index = 0U; index < deflation->provisional_count; ++index) {
      ss_deflation_entry_clear(&deflation->provisional[index]);
    }
    deflation->provisional_count = 0U;
    deflation->active_skip_extra = 0U;
    deflation->active_count = deflation->persistent_count;
  }
  deflation->transaction_generation = 0U;
  deflation->transaction_base_operator_generation = 0U;
  deflation->transaction_open = 0;
  deflation->transaction_basis_restored = 0;
  status = ss_status_allreduce_max(deflation->communicator, status);
  if (deflation->policy.projector != SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL) {
    PetscObjectState current_state = 0;
    int unchanged = 0, all_unchanged = 0;
    ss_deflation_clear_factorization(deflation);
    /* A reused Newton Mat may now hold the rejected tangent. Keep the
     * accepted basis, and refresh its images when the next tangent is ready. */
    if (status == SS_OK && deflation->persistent_count != 0U &&
        restore_operator != NULL && restore_generation != 0U) {
      if (MatGetState(restore_operator, &current_state) != PETSC_SUCCESS) {
        status = SS_ERROR_INTERNAL;
      } else {
        unchanged = current_state == deflation->transaction_base_operator_state;
      }
      status = ss_status_allreduce_max(deflation->communicator, status);
      if (status == SS_OK &&
          MPI_Allreduce(&unchanged, &all_unchanged, 1, MPI_INT, MPI_MIN,
                        deflation->communicator) != MPI_SUCCESS) {
        status = SS_ERROR_INTERNAL;
      }
      if (status == SS_OK && all_unchanged) {
        status = ss_deflation_refresh(deflation, restore_operator, restore_generation);
      }
    }
  }
  if (MatDestroy(&restore_operator) != PETSC_SUCCESS && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  return ss_status_allreduce_max(deflation->communicator, status);
}

SsStatus ss_deflation_diagnostics(const SsDeflation *deflation,
                                  SsDeflationDiagnostics *diagnostics) {
  size_t cached = 0U;
  size_t index;
  if (deflation == NULL || diagnostics == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (index = 0U; index < deflation->active_count; ++index) {
    if (ss_deflation_const_entry(deflation, index)->az != NULL) {
      ++cached;
    }
  }
  memset(diagnostics, 0, sizeof(*diagnostics));
  diagnostics->persistent_count = deflation->persistent_count;
  diagnostics->provisional_count = deflation->provisional_count;
  diagnostics->active_count = deflation->active_count;
  diagnostics->cached_az_count = cached;
  diagnostics->operator_state = deflation->operator_state;
  diagnostics->operator_generation = deflation->operator_generation;
  diagnostics->accepted_generation = deflation->accepted_generation;
  diagnostics->transaction_generation = deflation->transaction_generation;
  diagnostics->global_reductions = deflation->global_reductions;
  diagnostics->refresh_count = deflation->refresh_count;
  diagnostics->q_apply_count = deflation->q_apply_count;
  diagnostics->projected_apply_count = deflation->projected_apply_count;
  diagnostics->ill_conditioned_drop_count = deflation->ill_conditioned_drop_count;
  diagnostics->last_drop_reason = deflation->last_drop_reason;
  diagnostics->invariants_validated = deflation->invariants_validated;
  diagnostics->minimum_relative_pivot = deflation->minimum_relative_pivot;
  diagnostics->condition_estimate = deflation->condition_estimate;
  diagnostics->maximum_invariant_error = deflation->maximum_invariant_error;
  diagnostics->compression_count = deflation->compression_count;
  diagnostics->last_compression_input_count = deflation->last_compression_input_count;
  diagnostics->last_compression_protected_count =
      deflation->last_compression_protected_count;
  diagnostics->last_compression_retained_count =
      deflation->last_compression_retained_count;
  diagnostics->last_compression_information_fraction =
      deflation->last_compression_information_fraction;
  return SS_OK;
}

SsStatus ss_deflation_projector(const SsDeflation *deflation,
                                SsDeflationProjector *projector) {
  if (deflation == NULL || projector == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *projector = deflation->policy.projector;
  return SS_OK;
}

SsStatus ss_deflation_compression(const SsDeflation *deflation,
                                  SsDeflationCompression *compression) {
  if (deflation == NULL || compression == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *compression = deflation->policy.compression;
  return SS_OK;
}
