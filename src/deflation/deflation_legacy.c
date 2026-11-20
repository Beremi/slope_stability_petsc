/* Legacy A-orthogonal projector, cache, and transaction restore.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "deflation_internal.h"
#include "slope_internal/core/util.h"

#include <petsclog.h>

#include <stdlib.h>
#include <string.h>

SsStatus ss_deflation_restore_legacy_transaction_basis(SsDeflation *deflation) {
  size_t index;
  if (deflation == NULL || !deflation->transaction_open ||
      deflation->policy.projector != SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL) {
    return SS_ERROR_INVALID_STATE;
  }
  if (deflation->transaction_basis_restored != 0) {
    ss_deflation_clear_factorization(deflation);
    for (index = 0U; index < deflation->provisional_count; ++index) {
      ss_deflation_entry_clear(&deflation->provisional[index]);
    }
    deflation->provisional_count = 0U;
    deflation->active_skip_extra = 0U;
    ss_deflation_update_active_count(deflation);
    if (deflation->active_count == 0U) {
      deflation->factors_ready = 1;
      deflation->invariants_validated = 1;
      deflation->minimum_relative_pivot = 1.0;
      deflation->condition_estimate = 1.0;
      deflation->maximum_invariant_error = 0.0;
    }
    return SS_OK;
  }
  if (deflation->transaction_base_persistent == NULL) {
    return SS_ERROR_INVALID_STATE;
  }
  ss_deflation_clear_factorization(deflation);
  for (index = 0U; index < deflation->policy.dimension_cap; ++index) {
    ss_deflation_entry_clear(&deflation->persistent[index]);
    ss_deflation_entry_clear(&deflation->provisional[index]);
  }
  for (index = 0U; index < deflation->transaction_base_persistent_count; ++index) {
    deflation->persistent[index] = deflation->transaction_base_persistent[index];
    memset(&deflation->transaction_base_persistent[index], 0,
           sizeof(deflation->transaction_base_persistent[index]));
  }
  free(deflation->transaction_base_persistent);
  deflation->transaction_base_persistent = NULL;
  deflation->persistent_count = deflation->transaction_base_persistent_count;
  deflation->transaction_base_persistent_count = 0U;
  deflation->provisional_count = 0U;
  deflation->active_skip_extra = 0U;
  ss_deflation_update_active_count(deflation);
  deflation->transaction_basis_restored = 1;
  if (deflation->active_count == 0U) {
    deflation->factors_ready = 1;
    deflation->invariants_validated = 1;
    deflation->minimum_relative_pivot = 1.0;
    deflation->condition_estimate = 1.0;
    deflation->maximum_invariant_error = 0.0;
  }
  return SS_OK;
}

/*
 * Reproduce the legacy petsc4py/MATLAB A-orthogonalization exactly: visit the
 * newest column first, form A*v before subtracting previously accepted
 * columns, use the signed A-energy in the subtraction, and restore the kept
 * columns to chronological order.  One call is one complete legacy pass;
 * refresh invokes it policy.reorthogonalization_passes times.
 */
static SsStatus ss_deflation_legacy_aorth_pass(SsDeflation *deflation,
                                               Mat operator_handle) {
  const size_t dimension = deflation->active_count;
  const size_t active_start = ss_deflation_active_skip(deflation);
  Vec *orthogonal = NULL;
  Vec *replacement = NULL;
  PetscScalar *coefficients = NULL;
  PetscReal *signs = NULL;
  unsigned char *kept_source = NULL;
  size_t kept = 0U;
  size_t processed;
  size_t index;
  SsStatus status = SS_OK;
  if (dimension == 0U) {
    return SS_OK;
  }
  if (dimension > (size_t)PETSC_MAX_INT) {
    return SS_ERROR_OVERFLOW;
  }
  orthogonal = (Vec *)calloc(dimension, sizeof(*orthogonal));
  replacement = (Vec *)calloc(dimension, sizeof(*replacement));
  coefficients = (PetscScalar *)calloc(dimension, sizeof(*coefficients));
  signs = (PetscReal *)calloc(dimension, sizeof(*signs));
  kept_source = (unsigned char *)calloc(dimension, sizeof(*kept_source));
  status = ss_status_allreduce_max(deflation->communicator,
                                   orthogonal && replacement && coefficients && signs &&
                                           kept_source
                                       ? SS_OK
                                       : SS_ERROR_OUT_OF_MEMORY);
  if (status != SS_OK) {
    goto cleanup;
  }
  for (processed = 0U; processed < dimension; ++processed) {
    const size_t source_index = dimension - processed - 1U;
    const SsDeflationEntry *source = ss_deflation_const_entry(deflation, source_index);
    Vec candidate = NULL;
    Vec image = NULL;
    PetscScalar energy = 0.0;
    PetscReal magnitude;
    status = ss_deflation_vector_copy(deflation, source->right, &candidate);
    if (status == SS_OK) {
      status =
          ss_deflation_petsc_status(deflation, VecDuplicate(source->right, &image));
    }
    if (status == SS_OK) {
      status = ss_deflation_petsc_status(
          deflation, MatMult(operator_handle, source->right, image));
    }
    if (status != SS_OK) {
      (void)VecDestroy(&image);
      (void)VecDestroy(&candidate);
      goto cleanup;
    }
    if (kept != 0U) {
      if (ss_deflation_petsc_status(
              deflation, VecMDot(image, (PetscInt)kept, orthogonal, coefficients)) !=
          SS_OK) {
        (void)VecDestroy(&image);
        (void)VecDestroy(&candidate);
        status = SS_ERROR_INTERNAL;
        goto cleanup;
      }
      ++deflation->global_reductions;
      for (index = 0U; index < kept; ++index) {
        coefficients[index] *= -signs[index];
      }
      if (ss_deflation_petsc_status(deflation, VecMAXPY(candidate, (PetscInt)kept,
                                                        coefficients, orthogonal)) !=
          SS_OK) {
        (void)VecDestroy(&image);
        (void)VecDestroy(&candidate);
        status = SS_ERROR_INTERNAL;
        goto cleanup;
      }
    }
    if (ss_deflation_petsc_status(deflation, VecDot(candidate, image, &energy)) !=
        SS_OK) {
      (void)VecDestroy(&image);
      (void)VecDestroy(&candidate);
      status = SS_ERROR_INTERNAL;
      goto cleanup;
    }
    ++deflation->global_reductions;
    magnitude = PetscAbsScalar(energy);
    if (!ss_is_finite(magnitude)) {
      (void)VecDestroy(&image);
      (void)VecDestroy(&candidate);
      status = SS_ERROR_INTEGRITY;
      goto cleanup;
    }
    if (magnitude <= deflation->policy.norm_tolerance) {
      (void)VecDestroy(&image);
      (void)VecDestroy(&candidate);
      continue;
    }
    if (ss_deflation_petsc_status(
            deflation, VecScale(candidate, 1.0 / PetscSqrtReal(magnitude))) != SS_OK) {
      (void)VecDestroy(&image);
      (void)VecDestroy(&candidate);
      status = SS_ERROR_INTERNAL;
      goto cleanup;
    }
    orthogonal[kept] = candidate;
    replacement[source_index] = candidate;
    kept_source[source_index] = 1U;
    signs[kept] = PetscRealPart(energy) < 0.0 ? -1.0 : 1.0;
    ++kept;
    (void)VecDestroy(&image);
  }

  for (index = 0U; index < dimension; ++index) {
    if (kept_source[index] != 0U) {
      SsDeflationEntry *entry = ss_deflation_entry(deflation, index);
      Vec reference = replacement[index];
      Vec left = NULL, right = NULL;
      if (PetscObjectReference((PetscObject)reference) == PETSC_SUCCESS) {
        left = reference;
        if (PetscObjectReference((PetscObject)reference) == PETSC_SUCCESS) {
          right = reference;
        }
      }
      status = ss_status_allreduce_max(deflation->communicator,
                                       right != NULL ? SS_OK : SS_ERROR_INTERNAL);
      if (status != SS_OK) {
        (void)VecDestroy(&left);
        (void)VecDestroy(&right);
        goto cleanup;
      }
      (void)VecDestroy(&entry->right);
      (void)VecDestroy(&entry->left);
      entry->right = right;
      entry->left = left;
    }
  }
  for (index = dimension; index-- > 0U;) {
    if (kept_source[index] == 0U &&
        !ss_deflation_remove_combined_entry(deflation, active_start + index,
                                            SS_DEFLATION_DROP_DEPENDENT)) {
      status = SS_ERROR_INTERNAL;
      goto cleanup;
    }
  }

cleanup:
  for (index = 0U; index < kept; ++index) {
    (void)VecDestroy(&orthogonal[index]);
  }
  free(kept_source);
  free(signs);
  free(coefficients);
  free(replacement);
  free(orthogonal);
  return status;
}

SsStatus ss_deflation_refresh_legacy_aorth(SsDeflation *deflation, Mat operator_handle,
                                           PetscObjectState current_state,
                                           uint64_t operator_generation) {
  size_t dimension;
  size_t column;
  size_t row;
  uint32_t pass;
  PetscScalar *batched_gram = NULL;
  PetscReal maximum_error = 0.0;
  PetscBool known = PETSC_FALSE;
  PetscBool symmetric = PETSC_FALSE;
  SsStatus status = SS_OK;

  /* Checkpoint/snapshot restore refreshes a committed basis outside a
   * continuation transaction.  Preserve that basis exactly at restore time;
   * the first transactional refresh folds it into the incremental POD/SVD
   * state together with newly captured candidates. */
  if (deflation->policy.compression != SS_DEFLATION_COMPRESSION_NONE &&
      deflation->transaction_open) {
    if (PetscLogEventBegin(ss_deflation_compression_event, 0, 0, 0, 0) !=
        PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    status = ss_deflation_compress(deflation, operator_handle);
    if (PetscLogEventEnd(ss_deflation_compression_event, 0, 0, 0, 0) != PETSC_SUCCESS &&
        status == SS_OK) {
      status = SS_ERROR_INTERNAL;
    }
    if (status != SS_OK) {
      return status;
    }
  }

  /* Legacy caps the stored basis itself, rather than retaining an inactive
   * prefix that could re-enter after a dependent column is removed. */
  while (deflation->persistent_count + deflation->provisional_count >
         deflation->policy.dimension_cap) {
    if (!ss_deflation_remove_combined_entry(deflation, 0U,
                                            SS_DEFLATION_DROP_CAPACITY)) {
      return SS_ERROR_INTERNAL;
    }
  }
  deflation->active_skip_extra = 0U;
  ss_deflation_update_active_count(deflation);
  ss_deflation_clear_factorization(deflation);
  for (pass = 0U; pass < deflation->policy.reorthogonalization_passes; ++pass) {
    status = ss_deflation_legacy_aorth_pass(deflation, operator_handle);
    if (status != SS_OK || deflation->active_count == 0U) {
      break;
    }
  }
  if (status != SS_OK) {
    ss_deflation_clear_factorization(deflation);
    return status;
  }
  dimension = deflation->active_count;
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
  if (dimension > (size_t)PETSC_MAX_INT || dimension > SIZE_MAX / dimension ||
      dimension * dimension > SIZE_MAX / sizeof(*deflation->coarse)) {
    return SS_ERROR_OVERFLOW;
  }
  if (ss_deflation_petsc_status(deflation, MatIsSymmetricKnown(operator_handle, &known,
                                                               &symmetric)) != SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  deflation->coarse =
      (PetscScalar *)calloc(dimension * dimension, sizeof(*deflation->coarse));
  deflation->lu = (PetscScalar *)calloc(dimension * dimension, sizeof(*deflation->lu));
  deflation->pivots = (PetscInt *)calloc(dimension, sizeof(*deflation->pivots));
  deflation->dots = (PetscScalar *)calloc(dimension, sizeof(*deflation->dots));
  deflation->coefficients =
      (PetscScalar *)calloc(dimension, sizeof(*deflation->coefficients));
  deflation->active_left = (Vec *)calloc(dimension, sizeof(*deflation->active_left));
  deflation->active_right = (Vec *)calloc(dimension, sizeof(*deflation->active_right));
  deflation->active_az = (Vec *)calloc(dimension, sizeof(*deflation->active_az));
  deflation->active_atz = (Vec *)calloc(dimension, sizeof(*deflation->active_atz));
  batched_gram = (PetscScalar *)calloc(dimension * dimension, sizeof(*batched_gram));
  status = ss_status_allreduce_max(
      deflation->communicator,
      (deflation->coarse == NULL || deflation->dots == NULL ||
       deflation->coefficients == NULL || deflation->active_left == NULL ||
       deflation->active_right == NULL || deflation->active_az == NULL ||
       deflation->active_atz == NULL || deflation->lu == NULL ||
       deflation->pivots == NULL || batched_gram == NULL)
          ? SS_ERROR_OUT_OF_MEMORY
          : SS_OK);
  if (status != SS_OK) {
    free(batched_gram);
    ss_deflation_clear_factorization(deflation);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (row = 0U; row < dimension; ++row) {
    SsDeflationEntry *entry = ss_deflation_entry(deflation, row);
    deflation->active_left[row] = entry->left;
    deflation->active_right[row] = entry->right;
    if ((entry->az == NULL &&
         ss_deflation_petsc_status(deflation, VecDuplicate(entry->right, &entry->az)) !=
             SS_OK) ||
        ss_deflation_petsc_status(
            deflation, MatMult(operator_handle, entry->right, entry->az)) != SS_OK) {
      free(batched_gram);
      ss_deflation_clear_factorization(deflation);
      return SS_ERROR_INTERNAL;
    }
    deflation->active_az[row] = entry->az;
    if (known && symmetric) {
      deflation->active_atz[row] = entry->az;
    } else {
      if ((entry->atz == NULL &&
           ss_deflation_petsc_status(
               deflation, VecDuplicate(entry->left, &entry->atz)) != SS_OK) ||
          ss_deflation_petsc_status(
              deflation, MatMultTranspose(operator_handle, entry->left, entry->atz)) !=
              SS_OK) {
        free(batched_gram);
        ss_deflation_clear_factorization(deflation);
        return SS_ERROR_INTERNAL;
      }
      deflation->active_atz[row] = entry->atz;
    }
  }
  status = ss_deflation_gram(deflation, deflation->active_az, deflation->active_left,
                             dimension, batched_gram);
  if (status != SS_OK) {
    free(batched_gram);
    ss_deflation_clear_factorization(deflation);
    return status;
  }
  for (column = 0U; column < dimension; ++column) {
    for (row = 0U; row < dimension; ++row) {
      deflation->coarse[row * dimension + column] =
          batched_gram[column * dimension + row];
    }
  }
  free(batched_gram);
  /* A-orthogonalization normalizes |z^T A z|. The actual signed coarse
   * matrix defines the inverse, including the finite-precision off-diagonals. */
  status = ss_status_allreduce_max(deflation->communicator,
                                   ss_deflation_factorize(deflation));
  if (status != SS_OK) {
    ss_deflation_clear_factorization(deflation);
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
  status = ss_status_allreduce_max(
      deflation->communicator,
      ss_deflation_coarse_invariant_error(deflation, &maximum_error));
  if (status != SS_OK) {
    ss_deflation_clear_factorization(deflation);
    return status;
  }
  ++deflation->refresh_count;
  return SS_OK;
}

SsStatus ss_deflation_apply_cached_right_local(SsDeflation *deflation, Vec input,
                                               Vec output) {
  const PetscInt count = (PetscInt)deflation->active_count;
  if (count == 0) {
    return VecCopy(input, output) == PETSC_SUCCESS ? SS_OK : SS_ERROR_INTERNAL;
  }
  if (VecMDot(input, count, deflation->active_atz, deflation->coefficients) !=
      PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  ++deflation->global_reductions;
  const SsStatus status = ss_deflation_lu_solve(deflation, deflation->coefficients);
  if (status != SS_OK) {
    return status;
  }
  for (PetscInt index = 0; index < count; ++index) {
    deflation->coefficients[index] = -deflation->coefficients[index];
  }
  if (VecCopy(input, output) != PETSC_SUCCESS ||
      VecMAXPY(output, count, deflation->coefficients, deflation->active_right) !=
          PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  ++deflation->projected_apply_count;
  return SS_OK;
}

SsStatus ss_deflation_apply_cached_right(SsDeflation *deflation, Vec input,
                                         Vec output) {
  if (deflation == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  SsStatus status;
  if (input == NULL || output == NULL ||
      deflation->policy.projector != SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else if (deflation->active_count > (size_t)PETSC_MAX_INT ||
             (deflation->active_count != 0U &&
              (deflation->active_atz == NULL || deflation->active_right == NULL))) {
    status = SS_ERROR_INVALID_STATE;
  } else {
    status = ss_deflation_operator_ready(deflation, deflation->operator_handle);
  }
  status = ss_status_allreduce_max(deflation->communicator, status);
  return status == SS_OK
             ? ss_status_allreduce_max(
                   deflation->communicator,
                   ss_deflation_apply_cached_right_local(deflation, input, output))
             : status;
}

SsStatus ss_deflation_coarse_initial_guess_and_project_rhs(SsDeflation *deflation,
                                                           Vec rhs, Vec solution,
                                                           Vec projected_rhs) {
  PetscInt count;
  size_t index;
  SsStatus status;
  if (deflation == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (rhs == NULL || solution == NULL || projected_rhs == NULL ||
      deflation->policy.projector != SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else if (deflation->active_count > (size_t)PETSC_MAX_INT ||
             (deflation->active_count != 0U &&
              (deflation->active_right == NULL || deflation->active_az == NULL))) {
    status = SS_ERROR_INVALID_STATE;
  } else {
    status = ss_deflation_operator_ready(deflation, deflation->operator_handle);
  }
  status = ss_status_allreduce_max(deflation->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  if (deflation->active_count == 0U) {
    return ss_status_allreduce_max(deflation->communicator,
                                   VecSet(solution, 0.0) == PETSC_SUCCESS &&
                                           VecCopy(rhs, projected_rhs) == PETSC_SUCCESS
                                       ? SS_OK
                                       : SS_ERROR_INTERNAL);
  }
  count = (PetscInt)deflation->active_count;
  status = SS_ERROR_INTERNAL;
  if (VecMDot(rhs, count, deflation->active_left, deflation->coefficients) ==
      PETSC_SUCCESS) {
    ++deflation->global_reductions;
    status = ss_deflation_lu_solve(deflation, deflation->coefficients);
  }
  if (status == SS_OK && (VecSet(solution, 0.0) != PETSC_SUCCESS ||
                          VecMAXPY(solution, count, deflation->coefficients,
                                   deflation->active_right) != PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    for (index = 0U; index < deflation->active_count; ++index) {
      deflation->coefficients[index] = -deflation->coefficients[index];
    }
    if (VecCopy(rhs, projected_rhs) != PETSC_SUCCESS ||
        VecMAXPY(projected_rhs, count, deflation->coefficients, deflation->active_az) !=
            PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  status = ss_status_allreduce_max(deflation->communicator, status);
  if (status == SS_OK) {
    ++deflation->q_apply_count;
  }
  return status;
}
