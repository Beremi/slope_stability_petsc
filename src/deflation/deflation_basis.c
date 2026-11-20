/* Active and candidate basis storage and admission.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "deflation_internal.h"
#include "slope_internal/core/util.h"

#include <petscsys.h>

#include <stdlib.h>
#include <string.h>

void ss_deflation_entry_clear(SsDeflationEntry *entry) {
  if (entry == NULL) {
    return;
  }
  if (entry->az != NULL) {
    (void)VecDestroy(&entry->az);
  }
  (void)VecDestroy(&entry->atz);
  if (entry->source != NULL) {
    (void)VecDestroy(&entry->source);
  }
  if (entry->right != NULL) {
    (void)VecDestroy(&entry->right);
  }
  if (entry->left != NULL) {
    (void)VecDestroy(&entry->left);
  }
  memset(entry, 0, sizeof(*entry));
}

SsStatus ss_deflation_entry_clone(const SsDeflationEntry *source,
                                  SsDeflationEntry *destination) {
  memset(destination, 0, sizeof(*destination));
  /* Published basis columns and compression sources are immutable. Refresh
   * replaces their handles, so a transaction snapshot only needs references. */
  if (source == NULL || source->left == NULL || source->right == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  Vec vectors[] = {source->left, source->right, source->source};
  Vec *copies[] = {&destination->left, &destination->right, &destination->source};
  for (size_t i = 0U; i < sizeof(vectors) / sizeof(*vectors); ++i) {
    if (vectors[i] != NULL) {
      if (PetscObjectReference((PetscObject)vectors[i]) != PETSC_SUCCESS) {
        ss_deflation_entry_clear(destination);
        return SS_ERROR_INTERNAL;
      }
      *copies[i] = vectors[i];
    }
  }
  destination->source_weight = source->source_weight;
  memcpy(destination->provenance, source->provenance, sizeof(destination->provenance));
  return SS_OK;
}

SsStatus ss_deflation_vector_copy(SsDeflation *deflation, Vec source,
                                  Vec *destination) {
  SsStatus status =
      ss_deflation_petsc_status(deflation, VecDuplicate(source, destination));
  if (status == SS_OK) {
    status = ss_deflation_petsc_status(deflation, VecCopy(source, *destination));
  }
  if (status != SS_OK) {
    (void)VecDestroy(destination);
  }
  return status;
}

void ss_deflation_entry_array_destroy(SsDeflationEntry **entries, size_t capacity) {
  size_t index;
  if (entries == NULL || *entries == NULL) {
    return;
  }
  for (index = 0U; index < capacity; ++index) {
    ss_deflation_entry_clear(&(*entries)[index]);
  }
  free(*entries);
  *entries = NULL;
}

SsDeflationEntry *ss_deflation_combined_entry(SsDeflation *deflation, size_t index) {
  if (index < deflation->persistent_count) {
    return &deflation->persistent[index];
  }
  return &deflation->provisional[index - deflation->persistent_count];
}

const SsDeflationEntry *ss_deflation_const_combined_entry(const SsDeflation *deflation,
                                                          size_t index) {
  if (index < deflation->persistent_count) {
    return &deflation->persistent[index];
  }
  return &deflation->provisional[index - deflation->persistent_count];
}

size_t ss_deflation_active_skip(const SsDeflation *deflation) {
  const size_t combined = deflation->persistent_count + deflation->provisional_count;
  const size_t capacity_skip = combined > deflation->policy.dimension_cap
                                   ? combined - deflation->policy.dimension_cap
                                   : 0U;
  return capacity_skip + deflation->active_skip_extra;
}

SsDeflationEntry *ss_deflation_entry(SsDeflation *deflation, size_t index) {
  return ss_deflation_combined_entry(deflation,
                                     ss_deflation_active_skip(deflation) + index);
}

const SsDeflationEntry *ss_deflation_const_entry(const SsDeflation *deflation,
                                                 size_t index) {
  return ss_deflation_const_combined_entry(deflation,
                                           ss_deflation_active_skip(deflation) + index);
}

void ss_deflation_update_active_count(SsDeflation *deflation) {
  const size_t combined = deflation->persistent_count + deflation->provisional_count;
  const size_t admitted = combined < deflation->policy.dimension_cap
                              ? combined
                              : deflation->policy.dimension_cap;
  if (deflation->active_skip_extra > admitted) {
    deflation->active_skip_extra = admitted;
  }
  deflation->active_count = admitted - deflation->active_skip_extra;
}

int ss_deflation_drop_newest_provisional(SsDeflation *deflation,
                                         SsDeflationDropReason reason) {
  if (deflation == NULL || deflation->provisional_count == 0U) {
    return 0;
  }
  --deflation->provisional_count;
  ss_deflation_entry_clear(&deflation->provisional[deflation->provisional_count]);
  ss_deflation_update_active_count(deflation);
  if (reason == SS_DEFLATION_DROP_ILL_CONDITIONED) {
    ++deflation->ill_conditioned_drop_count;
  }
  deflation->last_drop_reason = reason;
  return 1;
}

int ss_deflation_drop_oldest_active_view(SsDeflation *deflation,
                                         SsDeflationDropReason reason) {
  if (deflation == NULL || deflation->active_count == 0U) {
    return 0;
  }
  ++deflation->active_skip_extra;
  ss_deflation_update_active_count(deflation);
  if (reason == SS_DEFLATION_DROP_ILL_CONDITIONED) {
    ++deflation->ill_conditioned_drop_count;
  }
  deflation->last_drop_reason = reason;
  return 1;
}

int ss_deflation_remove_combined_entry(SsDeflation *deflation, size_t index,
                                       SsDeflationDropReason reason) {
  SsDeflationEntry *entries;
  size_t *count;
  size_t local_index;
  size_t move;
  if (deflation == NULL ||
      index >= deflation->persistent_count + deflation->provisional_count) {
    return 0;
  }
  if (index < deflation->persistent_count) {
    entries = deflation->persistent;
    count = &deflation->persistent_count;
    local_index = index;
  } else {
    entries = deflation->provisional;
    count = &deflation->provisional_count;
    local_index = index - deflation->persistent_count;
  }
  ss_deflation_entry_clear(&entries[local_index]);
  for (move = local_index + 1U; move < *count; ++move) {
    entries[move - 1U] = entries[move];
  }
  memset(&entries[*count - 1U], 0, sizeof(*entries));
  --(*count);
  deflation->active_skip_extra = 0U;
  ss_deflation_update_active_count(deflation);
  deflation->last_drop_reason = reason;
  return 1;
}

void ss_deflation_destroy_work(SsDeflation *deflation) {
  size_t index;
  for (index = 0U; index < 4U; ++index) {
    if (deflation->work[index] != NULL) {
      (void)VecDestroy(&deflation->work[index]);
    }
  }
}

static SsStatus ss_deflation_reorthogonalize(SsDeflation *deflation, Vec candidate,
                                             int use_left, PetscReal original_norm,
                                             PetscReal *relative_norm) {
  PetscInt count;
  Vec *basis = NULL;
  PetscScalar *coefficients = NULL;
  PetscReal norm = original_norm;
  uint32_t pass;
  size_t combined;
  size_t basis_count;
  size_t basis_start;
  size_t index;
  combined = deflation->persistent_count + deflation->provisional_count;
  basis_count =
      deflation->policy.dimension_cap > 0U ? deflation->policy.dimension_cap - 1U : 0U;
  if (basis_count > combined) {
    basis_count = combined;
  }
  basis_start = combined - basis_count;
  if (basis_count == 0U) {
    *relative_norm = 1.0;
    return SS_OK;
  }
  if (basis_count > (size_t)PETSC_MAX_INT) {
    return SS_ERROR_OVERFLOW;
  }
  count = (PetscInt)basis_count;
  basis = (Vec *)calloc(basis_count, sizeof(*basis));
  coefficients = (PetscScalar *)calloc(basis_count, sizeof(*coefficients));
  SsStatus status = ss_status_allreduce_max(
      deflation->communicator, basis && coefficients ? SS_OK : SS_ERROR_OUT_OF_MEMORY);
  if (status != SS_OK) {
    free(coefficients);
    free(basis);
    return status;
  }
  for (index = 0U; index < basis_count; ++index) {
    const SsDeflationEntry *entry =
        ss_deflation_const_combined_entry(deflation, basis_start + index);
    basis[index] = use_left != 0 ? entry->left : entry->right;
  }
  for (pass = 0U; pass < deflation->policy.reorthogonalization_passes; ++pass) {
    if (ss_deflation_petsc_status(
            deflation, VecMDot(candidate, count, basis, coefficients)) != SS_OK) {
      free(coefficients);
      free(basis);
      return SS_ERROR_INTERNAL;
    }
    ++deflation->global_reductions;
    for (index = 0U; index < basis_count; ++index) {
      coefficients[index] = -coefficients[index];
    }
    if (ss_deflation_petsc_status(
            deflation, VecMAXPY(candidate, count, coefficients, basis)) != SS_OK) {
      free(coefficients);
      free(basis);
      return SS_ERROR_INTERNAL;
    }
  }
  if (ss_deflation_petsc_status(deflation, VecNorm(candidate, NORM_2, &norm)) !=
      SS_OK) {
    free(coefficients);
    free(basis);
    return SS_ERROR_INTERNAL;
  }
  ++deflation->global_reductions;
  free(coefficients);
  free(basis);
  if (!ss_is_finite(norm)) {
    return SS_ERROR_INTEGRITY;
  }
  *relative_norm = original_norm > 0.0 ? norm / original_norm : 0.0;
  if (*relative_norm <= deflation->policy.norm_tolerance) {
    return SS_ERROR_TYPE_MISMATCH;
  }
  return ss_deflation_petsc_status(deflation, VecScale(candidate, 1.0 / norm));
}

SsStatus ss_deflation_begin(SsDeflation *deflation, uint64_t transaction_generation) {
  size_t index;
  SsStatus status = SS_OK;
  if (deflation == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (transaction_generation == 0U) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else if (deflation->transaction_open ||
             deflation->transaction_base_persistent != NULL) {
    status = SS_ERROR_INVALID_STATE;
  }
  status = ss_status_allreduce_max(deflation->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  if (deflation->policy.projector == SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL) {
    deflation->transaction_base_persistent =
        (SsDeflationEntry *)calloc(deflation->policy.dimension_cap,
                                   sizeof(*deflation->transaction_base_persistent));
    if (deflation->transaction_base_persistent == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
    for (index = 0U; status == SS_OK && index < deflation->persistent_count; ++index) {
      status = ss_deflation_entry_clone(&deflation->persistent[index],
                                        &deflation->transaction_base_persistent[index]);
    }
    deflation->transaction_base_persistent_count = deflation->persistent_count;
  }
  if (status == SS_OK && deflation->operator_handle != NULL) {
    if (MatGetState(deflation->operator_handle,
                    &deflation->transaction_base_operator_state) != PETSC_SUCCESS ||
        PetscObjectReference((PetscObject)deflation->operator_handle) !=
            PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    } else {
      deflation->transaction_base_operator_handle = deflation->operator_handle;
    }
  }
  status = ss_status_allreduce_max(deflation->communicator, status);
  if (status != SS_OK) {
    (void)MatDestroy(&deflation->transaction_base_operator_handle);
    ss_deflation_entry_array_destroy(&deflation->transaction_base_persistent,
                                     deflation->policy.dimension_cap);
    deflation->transaction_base_persistent_count = 0U;
    return status;
  }
  deflation->transaction_open = 1;
  deflation->transaction_generation = transaction_generation;
  deflation->transaction_base_operator_generation = deflation->operator_generation;
  deflation->transaction_basis_restored = 0;
  deflation->provisional_count = 0U;
  ss_deflation_update_active_count(deflation);
  /*
   * A continuation attempt may satisfy its residual at the incoming state and
   * therefore perform no linear setup.  An empty basis needs no operator
   * refresh or invariant calculation, so make that transaction explicitly
   * commit-ready.  Non-empty bases retain the prior validated factorization
   * and are still refreshed by the linear layer whenever an operator is used.
   */
  if (deflation->active_count == 0U) {
    deflation->factors_ready = 1;
    deflation->invariants_validated = 1;
    deflation->minimum_relative_pivot = 1.0;
    deflation->condition_estimate = 1.0;
    deflation->maximum_invariant_error = 0.0;
  }
  return SS_OK;
}

SsStatus ss_deflation_add_weighted_candidate(SsDeflation *deflation, Vec left,
                                             Vec right, PetscReal source_weight,
                                             const char *provenance,
                                             SsDeflationAdmissionRecord *record) {
  SsDeflationEntry candidate = {0};
  PetscReal left_norm = 0.0;
  PetscReal right_norm = 0.0;
  PetscReal left_relative = 1.0;
  PetscReal right_relative = 1.0;
  SsStatus status;
  size_t provenance_length;
  if (deflation == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = SS_OK;
  provenance_length = ss_bounded_length(provenance, SS_DEFLATION_PROVENANCE_CAPACITY);
  if (left == NULL || right == NULL || provenance == NULL || record == NULL ||
      !ss_is_finite(source_weight) || source_weight <= 0.0) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else if (!deflation->transaction_open) {
    status = SS_ERROR_INVALID_STATE;
  } else if (provenance_length == SS_DEFLATION_PROVENANCE_CAPACITY) {
    status = SS_ERROR_LIMIT;
  }
  status = ss_status_allreduce_max(deflation->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  memset(record, 0, sizeof(*record));
  status = ss_deflation_petsc_status(deflation, VecNorm(left, NORM_2, &left_norm));
  if (status == SS_OK) {
    status = ss_deflation_petsc_status(deflation, VecNorm(right, NORM_2, &right_norm));
  }
  if (status != SS_OK) {
    return status;
  }
  deflation->global_reductions += 2U;
  record->left_norm = left_norm;
  record->right_norm = right_norm;
  if (!ss_is_finite(left_norm) || !ss_is_finite(right_norm)) {
    record->drop_reason = SS_DEFLATION_DROP_NONFINITE;
    ss_deflation_fill_record(deflation, record);
    return SS_OK;
  }
  if (left_norm <= deflation->policy.norm_tolerance ||
      right_norm <= deflation->policy.norm_tolerance) {
    record->drop_reason = SS_DEFLATION_DROP_NEGLIGIBLE;
    ss_deflation_fill_record(deflation, record);
    return SS_OK;
  }
  status = ss_deflation_vector_copy(deflation, left, &candidate.left);
  if (status == SS_OK) {
    status = ss_deflation_vector_copy(deflation, right, &candidate.right);
  }
  if (status == SS_OK &&
      deflation->policy.compression != SS_DEFLATION_COMPRESSION_NONE) {
    status = ss_deflation_vector_copy(deflation, right, &candidate.source);
  }
  if (status != SS_OK) {
    ss_deflation_entry_clear(&candidate);
    return status;
  }
  candidate.source_weight = source_weight;
  if (deflation->policy.projector == SS_DEFLATION_PROJECTOR_GENERAL_LEFT_RIGHT) {
    if (ss_status_allreduce_max(
            deflation->communicator,
            VecScale(candidate.left, 1.0 / left_norm) == PETSC_SUCCESS &&
                    VecScale(candidate.right, 1.0 / right_norm) == PETSC_SUCCESS
                ? SS_OK
                : SS_ERROR_INTERNAL) != SS_OK) {
      ss_deflation_entry_clear(&candidate);
      return SS_ERROR_INTERNAL;
    }
    status =
        ss_deflation_reorthogonalize(deflation, candidate.left, 1, 1.0, &left_relative);
    if (status == SS_ERROR_TYPE_MISMATCH) {
      record->drop_reason = SS_DEFLATION_DROP_DEPENDENT;
      ss_deflation_entry_clear(&candidate);
      ss_deflation_fill_record(deflation, record);
      return SS_OK;
    }
    if (status != SS_OK) {
      ss_deflation_entry_clear(&candidate);
      return status;
    }
    status = ss_deflation_reorthogonalize(deflation, candidate.right, 0, 1.0,
                                          &right_relative);
    if (status == SS_ERROR_TYPE_MISMATCH) {
      record->drop_reason = SS_DEFLATION_DROP_DEPENDENT;
      ss_deflation_entry_clear(&candidate);
      ss_deflation_fill_record(deflation, record);
      return SS_OK;
    }
    if (status != SS_OK) {
      ss_deflation_entry_clear(&candidate);
      return status;
    }
  }
  memcpy(candidate.provenance, provenance, provenance_length + 1U);
  if (deflation->provisional_count == deflation->policy.dimension_cap) {
    size_t index;
    ss_deflation_entry_clear(&deflation->provisional[0]);
    for (index = 1U; index < deflation->provisional_count; ++index) {
      deflation->provisional[index - 1U] = deflation->provisional[index];
    }
    memset(&deflation->provisional[deflation->provisional_count - 1U], 0,
           sizeof(deflation->provisional[0]));
    --deflation->provisional_count;
  }
  deflation->provisional[deflation->provisional_count] = candidate;
  ++deflation->provisional_count;
  deflation->active_skip_extra = 0U;
  ss_deflation_update_active_count(deflation);
  deflation->factors_ready = 0;
  deflation->invariants_validated = 0;
  record->admitted = 1;
  record->drop_reason = SS_DEFLATION_DROP_NONE;
  record->relative_pivot =
      left_relative < right_relative ? left_relative : right_relative;
  ss_deflation_fill_record(deflation, record);
  return SS_OK;
}

SsStatus ss_deflation_add_candidate(SsDeflation *deflation, Vec left, Vec right,
                                    const char *provenance,
                                    SsDeflationAdmissionRecord *record) {
  return ss_deflation_add_weighted_candidate(deflation, left, right, 1.0, provenance,
                                             record);
}

SsStatus ss_deflation_discard_provisional(SsDeflation *deflation) {
  size_t index;
  if (deflation == NULL || !deflation->transaction_open) {
    return SS_ERROR_INVALID_STATE;
  }
  /* The petsc4py continuation restores its pre-attempt dense basis before it
   * appends an accepted state.  A-orthogonalization mutates every persistent
   * column, so merely deleting provisional directions is not equivalent. */
  if (deflation->policy.projector == SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL) {
    return ss_deflation_restore_legacy_transaction_basis(deflation);
  }
  if (deflation->provisional_count == 0U) {
    return SS_OK;
  }
  for (index = 0U; index < deflation->provisional_count; ++index) {
    ss_deflation_entry_clear(&deflation->provisional[index]);
  }
  deflation->provisional_count = 0U;
  deflation->active_skip_extra = 0U;
  ss_deflation_update_active_count(deflation);
  ss_deflation_clear_factorization(deflation);
  if (deflation->active_count == 0U) {
    deflation->factors_ready = 1;
    deflation->invariants_validated = 1;
    deflation->minimum_relative_pivot = 1.0;
    deflation->condition_estimate = 1.0;
    deflation->maximum_invariant_error = 0.0;
  }
  return SS_OK;
}
