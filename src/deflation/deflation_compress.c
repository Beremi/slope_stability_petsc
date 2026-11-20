/* Snapshot selection and compression for deflation recycling.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "deflation_internal.h"
#include "slope_internal/core/util.h"

#include <petscblaslapack.h>
#include <petscsys.h>

#include <stdlib.h>
#include <string.h>

PetscLogEvent ss_deflation_compression_event = -1;

static int ss_deflation_entry_is_protected(const SsDeflationEntry *entry) {
  static const char protected_provenance[] = "nls:accepted-state";
  return entry != NULL && strcmp(entry->provenance, protected_provenance) == 0;
}

static void ss_deflation_vec_array_destroy(Vec *vectors, size_t count) {
  size_t index;
  if (vectors == NULL) {
    return;
  }
  for (index = 0U; index < count; ++index) {
    (void)VecDestroy(&vectors[index]);
  }
  free(vectors);
}

static SsStatus ss_deflation_euclidean_orthonormalize(SsDeflation *deflation,
                                                      Vec candidate, Vec *basis,
                                                      size_t basis_count,
                                                      PetscReal *relative_norm) {
  PetscScalar *coefficients = NULL;
  PetscReal original_norm = 0.0;
  PetscReal norm = 0.0;
  uint32_t pass;
  size_t index;
  if (deflation == NULL || candidate == NULL || relative_norm == NULL ||
      (basis_count != 0U && basis == NULL) || basis_count > (size_t)PETSC_MAX_INT) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (ss_deflation_petsc_status(deflation,
                                VecNorm(candidate, NORM_2, &original_norm)) != SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  ++deflation->global_reductions;
  if (!ss_is_finite(original_norm)) {
    return SS_ERROR_INTEGRITY;
  }
  if (original_norm <= deflation->policy.norm_tolerance) {
    *relative_norm = 0.0;
    return SS_ERROR_TYPE_MISMATCH;
  }
  if (basis_count != 0U) {
    coefficients = (PetscScalar *)calloc(basis_count, sizeof(*coefficients));
    if (ss_status_allreduce_max(deflation->communicator, (coefficients == NULL)
                                                             ? SS_ERROR_OUT_OF_MEMORY
                                                             : SS_OK) != SS_OK) {
      return SS_ERROR_OUT_OF_MEMORY;
    }
    for (pass = 0U; pass < deflation->policy.reorthogonalization_passes; ++pass) {
      if (ss_deflation_petsc_status(deflation, VecMDot(candidate, (PetscInt)basis_count,
                                                       basis, coefficients)) != SS_OK) {
        free(coefficients);
        return SS_ERROR_INTERNAL;
      }
      ++deflation->global_reductions;
      for (index = 0U; index < basis_count; ++index) {
        coefficients[index] = -coefficients[index];
      }
      if (ss_deflation_petsc_status(deflation,
                                    VecMAXPY(candidate, (PetscInt)basis_count,
                                             coefficients, basis)) != SS_OK) {
        free(coefficients);
        return SS_ERROR_INTERNAL;
      }
    }
  }
  free(coefficients);
  if (ss_deflation_petsc_status(deflation, VecNorm(candidate, NORM_2, &norm)) !=
      SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  ++deflation->global_reductions;
  if (!ss_is_finite(norm)) {
    return SS_ERROR_INTEGRITY;
  }
  *relative_norm = norm / original_norm;
  if (*relative_norm <= deflation->policy.norm_tolerance) {
    return SS_ERROR_TYPE_MISMATCH;
  }
  return ss_deflation_petsc_status(deflation, VecScale(candidate, 1.0 / norm));
}

static SsStatus ss_deflation_dense_hermitian_eigensolve(size_t dimension,
                                                        PetscScalar *matrix,
                                                        PetscReal *eigenvalues) {
  PetscBLASInt n = 0;
  PetscBLASInt lwork = 0;
  PetscBLASInt info = 0;
  PetscScalar *work = NULL;
#if defined(PETSC_USE_COMPLEX)
  PetscReal *real_work = NULL;
#endif
  if (dimension == 0U || matrix == NULL || eigenvalues == NULL ||
      dimension > (size_t)PETSC_MAX_INT ||
      PetscBLASIntCast((PetscInt)dimension, &n) != PETSC_SUCCESS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (dimension > ((size_t)PETSC_BLAS_INT_MAX - 1U) / 3U) {
    return SS_ERROR_OVERFLOW;
  }
  lwork = (PetscBLASInt)(3U * dimension + 1U);
  work = (PetscScalar *)calloc((size_t)lwork, sizeof(*work));
  if (work == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
#if defined(PETSC_USE_COMPLEX)
  real_work = (PetscReal *)calloc(3U * dimension + 1U, sizeof(*real_work));
  if (real_work == NULL) {
    free(work);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  LAPACKsyev_("V", "U", &n, matrix, &n, eigenvalues, work, &lwork, real_work, &info);
  free(real_work);
#else
  LAPACKsyev_("V", "U", &n, matrix, &n, eigenvalues, work, &lwork, &info);
#endif
  free(work);
  return info == 0 ? SS_OK : SS_ERROR_INTEGRITY;
}

static SsStatus ss_deflation_snapshot_gram(SsDeflation *deflation, Vec *vectors,
                                           size_t count, PetscScalar *gram) {
  size_t column;
  size_t row;
  if (deflation == NULL || vectors == NULL || count == 0U || gram == NULL ||
      count > (size_t)PETSC_MAX_INT) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  const SsStatus status = ss_deflation_gram(deflation, vectors, vectors, count, gram);
  if (status != SS_OK) {
    return status;
  }
  for (column = 0U; column < count; ++column) {
    for (row = 0U; row <= column; ++row) {
      const PetscScalar symmetric =
          0.5 * (gram[row + column * count] + PetscConj(gram[column + row * count]));
      gram[row + column * count] = symmetric;
      gram[column + row * count] = PetscConj(symmetric);
    }
  }
  return SS_OK;
}

static SsStatus
ss_deflation_linear_combinations(SsDeflation *deflation, Vec *basis, size_t basis_count,
                                 const PetscScalar *eigenvectors,
                                 const PetscReal *eigenvalues, const size_t *selected,
                                 size_t selected_count, int scale_by_eigenvalue,
                                 Vec **modes_result) {
  Vec *modes = NULL;
  PetscScalar *coefficients = NULL;
  size_t mode;
  size_t index;
  if (basis == NULL || basis_count == 0U || eigenvectors == NULL ||
      eigenvalues == NULL || selected == NULL || selected_count == 0U ||
      modes_result == NULL || basis_count > (size_t)PETSC_MAX_INT) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *modes_result = NULL;
  modes = (Vec *)calloc(selected_count, sizeof(*modes));
  coefficients = (PetscScalar *)calloc(basis_count, sizeof(*coefficients));
  if (ss_status_allreduce_max(deflation->communicator,
                              (modes == NULL || coefficients == NULL)
                                  ? SS_ERROR_OUT_OF_MEMORY
                                  : SS_OK) != SS_OK) {
    free(coefficients);
    free(modes);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (mode = 0U; mode < selected_count; ++mode) {
    const size_t column = selected[mode];
    const PetscReal scale =
        scale_by_eigenvalue != 0 ? 1.0 / PetscSqrtReal(eigenvalues[column]) : 1.0;
    if (ss_deflation_petsc_status(deflation, VecDuplicate(basis[0], &modes[mode])) !=
            SS_OK ||
        ss_deflation_petsc_status(deflation, VecSet(modes[mode], 0.0)) != SS_OK) {
      ss_deflation_vec_array_destroy(modes, selected_count);
      free(coefficients);
      return SS_ERROR_OUT_OF_MEMORY;
    }
    for (index = 0U; index < basis_count; ++index) {
      coefficients[index] = eigenvectors[index + column * basis_count] * scale;
    }
    if (ss_deflation_petsc_status(deflation,
                                  VecMAXPY(modes[mode], (PetscInt)basis_count,
                                           coefficients, basis)) != SS_OK) {
      ss_deflation_vec_array_destroy(modes, selected_count);
      free(coefficients);
      return SS_ERROR_INTERNAL;
    }
  }
  free(coefficients);
  *modes_result = modes;
  return SS_OK;
}

static SsStatus ss_deflation_build_pod_modes(
    SsDeflation *deflation, Vec *snapshots, size_t snapshot_count,
    size_t target_dimension, Vec **modes, size_t *mode_count, PetscReal **mode_weights,
    PetscReal *information_fraction, PetscReal *boundary_ratio) {
  PetscScalar *gram = NULL;
  PetscReal *eigenvalues = NULL;
  size_t *selected = NULL;
  PetscReal total = 0.0;
  PetscReal retained = 0.0;
  PetscReal cutoff;
  size_t index;
  SsStatus status;
  if (deflation == NULL || snapshots == NULL || snapshot_count == 0U ||
      target_dimension == 0U || modes == NULL || mode_count == NULL ||
      mode_weights == NULL || information_fraction == NULL || boundary_ratio == NULL ||
      snapshot_count > SIZE_MAX / snapshot_count ||
      snapshot_count * snapshot_count > SIZE_MAX / sizeof(*gram)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *modes = NULL;
  *mode_count = 0U;
  *mode_weights = NULL;
  *boundary_ratio = PETSC_MAX_REAL;
  gram = (PetscScalar *)calloc(snapshot_count * snapshot_count, sizeof(*gram));
  eigenvalues = (PetscReal *)calloc(snapshot_count, sizeof(*eigenvalues));
  selected = (size_t *)calloc(snapshot_count, sizeof(*selected));
  if (ss_status_allreduce_max(deflation->communicator,
                              (gram == NULL || eigenvalues == NULL || selected == NULL)
                                  ? SS_ERROR_OUT_OF_MEMORY
                                  : SS_OK) != SS_OK) {
    status = SS_ERROR_OUT_OF_MEMORY;
    goto cleanup;
  }
  status = ss_deflation_snapshot_gram(deflation, snapshots, snapshot_count, gram);
  if (status != SS_OK) {
    goto cleanup;
  }
  status = ss_status_allreduce_max(
      deflation->communicator,
      ss_deflation_dense_hermitian_eigensolve(snapshot_count, gram, eigenvalues));
  if (status != SS_OK) {
    goto cleanup;
  }
  cutoff =
      eigenvalues[snapshot_count - 1U] *
      PetscMax(100.0 * PETSC_MACHINE_EPSILON,
               deflation->policy.norm_tolerance * deflation->policy.norm_tolerance);
  for (index = 0U; index < snapshot_count; ++index) {
    if (eigenvalues[index] > cutoff) {
      total += eigenvalues[index];
    }
  }
  for (index = snapshot_count; index-- > 0U && *mode_count < target_dimension;) {
    if (eigenvalues[index] > cutoff) {
      selected[*mode_count] = index;
      retained += eigenvalues[index];
      ++(*mode_count);
    }
  }
  /* Report whether the fixed-dimensional cutoff splits a clustered singular
   * subspace.  The dimension is a hard storage/refresh-cost bound, so do not
   * silently retain more modes; callers can use this ratio to choose a safe
   * nearby dimension. */
  if (*mode_count != 0U) {
    const size_t weakest_retained = selected[*mode_count - 1U];
    if (weakest_retained > 0U && eigenvalues[weakest_retained - 1U] > cutoff) {
      *boundary_ratio =
          eigenvalues[selected[*mode_count - 1U]] / eigenvalues[weakest_retained - 1U];
    }
  }
  if (*mode_count == 0U) {
    status = SS_OK;
    *information_fraction = 1.0;
    goto cleanup;
  }
  status =
      ss_deflation_linear_combinations(deflation, snapshots, snapshot_count, gram,
                                       eigenvalues, selected, *mode_count, 1, modes);
  if (status == SS_OK) {
    *mode_weights = (PetscReal *)calloc(*mode_count, sizeof(**mode_weights));
    if (ss_status_allreduce_max(deflation->communicator, *mode_weights == NULL
                                                             ? SS_ERROR_OUT_OF_MEMORY
                                                             : SS_OK) != SS_OK) {
      free(*mode_weights);
      *mode_weights = NULL;
      ss_deflation_vec_array_destroy(*modes, *mode_count);
      *modes = NULL;
      status = SS_ERROR_OUT_OF_MEMORY;
      goto cleanup;
    }
    for (index = 0U; index < *mode_count; ++index) {
      (*mode_weights)[index] = eigenvalues[selected[index]];
    }
    *information_fraction = total > 0.0 ? retained / total : 1.0;
  }

cleanup:
  free(selected);
  free(eigenvalues);
  free(gram);
  return status;
}

static SsStatus ss_deflation_build_operator_svd_modes(
    SsDeflation *deflation, Mat operator_handle, Vec *snapshots, size_t snapshot_count,
    size_t target_dimension, Vec **modes, size_t *mode_count,
    PetscReal *information_fraction) {
  Vec *orthogonal = NULL;
  Vec *images = NULL;
  PetscScalar *gram = NULL;
  PetscReal *eigenvalues = NULL;
  size_t *selected = NULL;
  size_t rank = 0U;
  size_t index;
  PetscReal maximum;
  PetscReal floor_value;
  PetscReal total = 0.0;
  PetscReal retained = 0.0;
  SsStatus status = SS_OK;
  if (deflation == NULL || operator_handle == NULL || snapshots == NULL ||
      snapshot_count == 0U || target_dimension == 0U || modes == NULL ||
      mode_count == NULL || information_fraction == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *modes = NULL;
  *mode_count = 0U;
  orthogonal = (Vec *)calloc(snapshot_count, sizeof(*orthogonal));
  if (ss_status_allreduce_max(deflation->communicator,
                              (orthogonal == NULL) ? SS_ERROR_OUT_OF_MEMORY : SS_OK) !=
      SS_OK) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (index = 0U; index < snapshot_count; ++index) {
    PetscReal relative_norm = 0.0;
    if (ss_deflation_vector_copy(deflation, snapshots[index], &orthogonal[rank]) !=
        SS_OK) {
      status = SS_ERROR_OUT_OF_MEMORY;
      goto cleanup;
    }
    status = ss_deflation_euclidean_orthonormalize(deflation, orthogonal[rank],
                                                   orthogonal, rank, &relative_norm);
    if (status == SS_ERROR_TYPE_MISMATCH) {
      (void)VecDestroy(&orthogonal[rank]);
      status = SS_OK;
      continue;
    }
    if (status != SS_OK) {
      goto cleanup;
    }
    ++rank;
  }
  if (rank == 0U) {
    *information_fraction = 1.0;
    goto cleanup;
  }
  images = (Vec *)calloc(rank, sizeof(*images));
  if (rank > SIZE_MAX / rank || rank * rank > SIZE_MAX / sizeof(*gram)) {
    status = SS_ERROR_OVERFLOW;
    goto cleanup;
  }
  gram = (PetscScalar *)calloc(rank * rank, sizeof(*gram));
  eigenvalues = (PetscReal *)calloc(rank, sizeof(*eigenvalues));
  selected = (size_t *)calloc(PetscMin(rank, target_dimension), sizeof(*selected));
  if (ss_status_allreduce_max(
          deflation->communicator,
          (images == NULL || gram == NULL || eigenvalues == NULL || selected == NULL)
              ? SS_ERROR_OUT_OF_MEMORY
              : SS_OK) != SS_OK) {
    status = SS_ERROR_OUT_OF_MEMORY;
    goto cleanup;
  }
  for (index = 0U; index < rank; ++index) {
    if (ss_deflation_petsc_status(
            deflation, VecDuplicate(orthogonal[index], &images[index])) != SS_OK ||
        ss_deflation_petsc_status(deflation, MatMult(operator_handle, orthogonal[index],
                                                     images[index])) != SS_OK) {
      status = SS_ERROR_INTERNAL;
      goto cleanup;
    }
  }
  status = ss_deflation_snapshot_gram(deflation, images, rank, gram);
  if (status != SS_OK) {
    goto cleanup;
  }
  status = ss_status_allreduce_max(
      deflation->communicator,
      ss_deflation_dense_hermitian_eigensolve(rank, gram, eigenvalues));
  if (status != SS_OK) {
    goto cleanup;
  }
  *mode_count = PetscMin(rank, target_dimension);
  for (index = 0U; index < *mode_count; ++index) {
    selected[index] = index;
  }
  maximum = PetscMax(eigenvalues[rank - 1U], 0.0);
  floor_value =
      PetscMax(maximum * 100.0 * PETSC_MACHINE_EPSILON, PETSC_MACHINE_EPSILON);
  for (index = 0U; index < rank; ++index) {
    if (!ss_is_finite(eigenvalues[index]) || eigenvalues[index] < -floor_value) {
      status = SS_ERROR_INTEGRITY;
      goto cleanup;
    }
    total += 1.0 / PetscMax(eigenvalues[index], floor_value);
    if (index < *mode_count) {
      retained += 1.0 / PetscMax(eigenvalues[index], floor_value);
    }
  }
  status = ss_deflation_linear_combinations(
      deflation, orthogonal, rank, gram, eigenvalues, selected, *mode_count, 0, modes);
  if (status == SS_OK) {
    *information_fraction = total > 0.0 ? retained / total : 1.0;
  }

cleanup:
  free(selected);
  free(eigenvalues);
  free(gram);
  ss_deflation_vec_array_destroy(images, rank);
  ss_deflation_vec_array_destroy(orthogonal, rank);
  return status;
}

static SsStatus ss_deflation_install_compressed_basis(
    SsDeflation *deflation, Vec *modes, const PetscReal *mode_weights,
    size_t mode_count, const size_t *protected_indices, size_t protected_count,
    const char *mode_provenance) {
  SsDeflationEntry *replacement = NULL;
  size_t replacement_count = mode_count + protected_count;
  size_t index;
  SsStatus status = SS_OK;
  if (deflation == NULL || (mode_count != 0U && modes == NULL) ||
      (protected_count != 0U && protected_indices == NULL) || mode_provenance == NULL ||
      replacement_count > deflation->policy.dimension_cap) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  replacement =
      (SsDeflationEntry *)calloc(deflation->policy.dimension_cap, sizeof(*replacement));
  if (ss_status_allreduce_max(deflation->communicator,
                              (replacement == NULL) ? SS_ERROR_OUT_OF_MEMORY : SS_OK) !=
      SS_OK) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (index = 0U; index < mode_count; ++index) {
    size_t length =
        ss_bounded_length(mode_provenance, SS_DEFLATION_PROVENANCE_CAPACITY);
    if (length == SS_DEFLATION_PROVENANCE_CAPACITY ||
        ss_deflation_vector_copy(deflation, modes[index], &replacement[index].source) !=
            SS_OK ||
        ss_deflation_vector_copy(deflation, modes[index], &replacement[index].right) !=
            SS_OK ||
        ss_deflation_vector_copy(deflation, modes[index], &replacement[index].left) !=
            SS_OK) {
      status = SS_ERROR_OUT_OF_MEMORY;
      goto cleanup;
    }
    replacement[index].source_weight = mode_weights == NULL ? 1.0 : mode_weights[index];
    memcpy(replacement[index].provenance, mode_provenance, length + 1U);
  }
  /* Protected accepted states are last because the legacy reverse-order
   * A-orthogonalization then preserves them before it processes compressed
   * modes. */
  for (index = 0U; index < protected_count; ++index) {
    const SsDeflationEntry *source =
        ss_deflation_const_combined_entry(deflation, protected_indices[index]);
    status = ss_status_allreduce_max(
        deflation->communicator,
        ss_deflation_entry_clone(source, &replacement[mode_count + index]));
    if (status != SS_OK) {
      goto cleanup;
    }
  }
  ss_deflation_clear_factorization(deflation);
  for (index = 0U; index < deflation->policy.dimension_cap; ++index) {
    ss_deflation_entry_clear(&deflation->persistent[index]);
    ss_deflation_entry_clear(&deflation->provisional[index]);
  }
  deflation->persistent_count = 0U;
  for (index = 0U; index < replacement_count; ++index) {
    deflation->provisional[index] = replacement[index];
    memset(&replacement[index], 0, sizeof(replacement[index]));
  }
  deflation->provisional_count = replacement_count;
  deflation->active_skip_extra = 0U;
  ss_deflation_update_active_count(deflation);

cleanup:
  ss_deflation_entry_array_destroy(&replacement, deflation->policy.dimension_cap);
  return status;
}

static SsStatus ss_deflation_install_selected_basis(SsDeflation *deflation,
                                                    const size_t *selected_indices,
                                                    size_t selected_count,
                                                    const size_t *protected_indices,
                                                    size_t protected_count) {
  SsDeflationEntry *replacement = NULL;
  const size_t replacement_count = selected_count + protected_count;
  size_t index;
  SsStatus status = SS_OK;
  if (deflation == NULL || (selected_count != 0U && selected_indices == NULL) ||
      (protected_count != 0U && protected_indices == NULL) ||
      replacement_count > deflation->policy.dimension_cap) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  replacement =
      (SsDeflationEntry *)calloc(deflation->policy.dimension_cap, sizeof(*replacement));
  if (ss_status_allreduce_max(deflation->communicator,
                              (replacement == NULL) ? SS_ERROR_OUT_OF_MEMORY : SS_OK) !=
      SS_OK) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (index = 0U; index < selected_count; ++index) {
    status = ss_status_allreduce_max(
        deflation->communicator,
        ss_deflation_entry_clone(
            ss_deflation_const_combined_entry(deflation, selected_indices[index]),
            &replacement[index]));
    if (status != SS_OK) {
      goto cleanup;
    }
  }
  /* As in the POD layout, accepted states are visited first by the reverse
   * legacy A-orthogonalization and therefore remain the protected columns. */
  for (index = 0U; index < protected_count; ++index) {
    status = ss_status_allreduce_max(
        deflation->communicator,
        ss_deflation_entry_clone(
            ss_deflation_const_combined_entry(deflation, protected_indices[index]),
            &replacement[selected_count + index]));
    if (status != SS_OK) {
      goto cleanup;
    }
  }
  ss_deflation_clear_factorization(deflation);
  for (index = 0U; index < deflation->policy.dimension_cap; ++index) {
    ss_deflation_entry_clear(&deflation->persistent[index]);
    ss_deflation_entry_clear(&deflation->provisional[index]);
  }
  deflation->persistent_count = 0U;
  for (index = 0U; index < replacement_count; ++index) {
    deflation->provisional[index] = replacement[index];
    memset(&replacement[index], 0, sizeof(replacement[index]));
  }
  deflation->provisional_count = replacement_count;
  deflation->active_skip_extra = 0U;
  ss_deflation_update_active_count(deflation);

cleanup:
  ss_deflation_entry_array_destroy(&replacement, deflation->policy.dimension_cap);
  return status;
}

SsStatus ss_deflation_compress(SsDeflation *deflation, Mat operator_handle) {
  const size_t combined = deflation->persistent_count + deflation->provisional_count;
  size_t *protected_indices = NULL;
  size_t *selected_indices = NULL;
  Vec *protected_basis = NULL;
  Vec *snapshots = NULL;
  Vec *modes = NULL;
  PetscReal *snapshot_weights = NULL;
  PetscReal *mode_weights = NULL;
  size_t protected_count = 0U;
  size_t protected_rank = 0U;
  size_t snapshot_count = 0U;
  size_t mode_count = 0U;
  size_t selected_count = 0U;
  size_t target;
  size_t index;
  PetscReal information_fraction = 1.0;
  PetscReal boundary_ratio = PETSC_MAX_REAL;
  const char *method_name;
  const char *provenance;
  SsStatus status = SS_OK;
  if (deflation == NULL ||
      deflation->policy.compression == SS_DEFLATION_COMPRESSION_NONE) {
    return SS_OK;
  }
  if (operator_handle == NULL) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else if (!deflation->transaction_open ||
             deflation->policy.projector !=
                 SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL ||
             deflation->policy.compression_dimension == 0U) {
    status = SS_ERROR_INVALID_STATE;
  }
  status = ss_status_allreduce_max(deflation->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  protected_indices = (size_t *)calloc(combined, sizeof(*protected_indices));
  selected_indices = (size_t *)calloc(combined, sizeof(*selected_indices));
  protected_basis = (Vec *)calloc(combined, sizeof(*protected_basis));
  snapshots = (Vec *)calloc(combined, sizeof(*snapshots));
  snapshot_weights = (PetscReal *)calloc(combined, sizeof(*snapshot_weights));
  if (ss_status_allreduce_max(deflation->communicator,
                              protected_indices && selected_indices &&
                                      protected_basis && snapshots && snapshot_weights
                                  ? SS_OK
                                  : SS_ERROR_OUT_OF_MEMORY) != SS_OK) {
    status = SS_ERROR_OUT_OF_MEMORY;
    goto cleanup;
  }
  for (index = 0U; index < combined; ++index) {
    const SsDeflationEntry *entry = ss_deflation_const_combined_entry(deflation, index);
    if (deflation->policy.compression != SS_DEFLATION_COMPRESSION_SNAPSHOT_POD_TOTAL &&
        ss_deflation_entry_is_protected(entry)) {
      protected_indices[protected_count++] = index;
    }
  }
  if (protected_count >= deflation->policy.dimension_cap) {
    target = 0U;
    protected_count = deflation->policy.dimension_cap;
  } else {
    target = PetscMin(deflation->policy.compression_dimension,
                      deflation->policy.dimension_cap - protected_count);
  }
  if (combined - protected_count <= target) {
    goto cleanup;
  }
  if (deflation->policy.compression ==
      SS_DEFLATION_COMPRESSION_SYSTEM_SOLUTION_WINDOW) {
    size_t nonprotected_to_skip = combined - protected_count - target;
    for (index = 0U; index < combined; ++index) {
      const SsDeflationEntry *entry =
          ss_deflation_const_combined_entry(deflation, index);
      if (ss_deflation_entry_is_protected(entry)) {
        continue;
      }
      if (nonprotected_to_skip != 0U) {
        --nonprotected_to_skip;
        continue;
      }
      selected_indices[selected_count++] = index;
    }
    status =
        ss_deflation_install_selected_basis(deflation, selected_indices, selected_count,
                                            protected_indices, protected_count);
    if (status != SS_OK) {
      goto cleanup;
    }
    ++deflation->compression_count;
    deflation->last_compression_input_count = combined - protected_count;
    deflation->last_compression_protected_count = protected_count;
    deflation->last_compression_retained_count = selected_count;
    deflation->last_compression_information_fraction =
        combined == protected_count
            ? 1.0
            : (PetscReal)selected_count / (PetscReal)(combined - protected_count);
    if (PetscPrintf(deflation->communicator,
                    "SLOPE_DEFLATION_COMPRESSION "
                    "%llu,system_solution_window,%zu,%zu,%zu,%.17g,1\n",
                    (unsigned long long)deflation->transaction_generation,
                    combined - protected_count, protected_count, selected_count,
                    (double)deflation->last_compression_information_fraction) !=
        PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
    goto cleanup;
  }
  for (index = 0U; index < protected_count; ++index) {
    PetscReal relative_norm = 0.0;
    const SsDeflationEntry *entry =
        ss_deflation_const_combined_entry(deflation, protected_indices[index]);
    Vec source = entry->source != NULL ? entry->source : entry->right;
    if (ss_deflation_vector_copy(deflation, source, &protected_basis[protected_rank]) !=
        SS_OK) {
      status = SS_ERROR_OUT_OF_MEMORY;
      goto cleanup;
    }
    status = ss_deflation_euclidean_orthonormalize(
        deflation, protected_basis[protected_rank], protected_basis, protected_rank,
        &relative_norm);
    if (status == SS_ERROR_TYPE_MISMATCH) {
      (void)VecDestroy(&protected_basis[protected_rank]);
      status = SS_OK;
      continue;
    }
    if (status != SS_OK) {
      goto cleanup;
    }
    ++protected_rank;
  }
  for (index = 0U; index < combined; ++index) {
    const SsDeflationEntry *entry = ss_deflation_const_combined_entry(deflation, index);
    PetscReal relative_norm = 0.0;
    Vec source;
    if (deflation->policy.compression != SS_DEFLATION_COMPRESSION_SNAPSHOT_POD_TOTAL &&
        ss_deflation_entry_is_protected(entry)) {
      continue;
    }
    source = entry->source != NULL ? entry->source : entry->right;
    if (ss_deflation_vector_copy(deflation, source, &snapshots[snapshot_count]) !=
        SS_OK) {
      status = SS_ERROR_OUT_OF_MEMORY;
      goto cleanup;
    }
    status = ss_deflation_euclidean_orthonormalize(deflation, snapshots[snapshot_count],
                                                   protected_basis, protected_rank,
                                                   &relative_norm);
    if (status == SS_ERROR_TYPE_MISMATCH) {
      (void)VecDestroy(&snapshots[snapshot_count]);
      status = SS_OK;
      continue;
    }
    if (status != SS_OK) {
      goto cleanup;
    }
    snapshot_weights[snapshot_count] =
        (ss_is_finite(entry->source_weight) && entry->source_weight > 0.0
             ? entry->source_weight
             : 1.0) *
        relative_norm * relative_norm;
    if (strncmp(entry->provenance, "compress:snapshot-pod",
                sizeof("compress:snapshot-pod") - 1U) == 0) {
      snapshot_weights[snapshot_count] *= deflation->policy.compression_history_decay;
    }
    ++snapshot_count;
  }
  if (target != 0U && snapshot_count != 0U) {
    if (deflation->policy.compression == SS_DEFLATION_COMPRESSION_SNAPSHOT_POD ||
        deflation->policy.compression == SS_DEFLATION_COMPRESSION_SNAPSHOT_POD_TOTAL) {
      for (index = 0U; index < snapshot_count; ++index) {
        if (ss_deflation_petsc_status(
                deflation, VecScale(snapshots[index],
                                    PetscSqrtReal(snapshot_weights[index]))) != SS_OK) {
          status = SS_ERROR_INTERNAL;
          goto cleanup;
        }
      }
      status = ss_deflation_build_pod_modes(deflation, snapshots, snapshot_count,
                                            target, &modes, &mode_count, &mode_weights,
                                            &information_fraction, &boundary_ratio);
    } else if (deflation->policy.compression == SS_DEFLATION_COMPRESSION_OPERATOR_SVD) {
      status = ss_deflation_build_operator_svd_modes(
          deflation, operator_handle, snapshots, snapshot_count, target, &modes,
          &mode_count, &information_fraction);
    } else {
      status = SS_ERROR_INVALID_ARGUMENT;
    }
    if (status != SS_OK) {
      goto cleanup;
    }
  }
  method_name =
      deflation->policy.compression == SS_DEFLATION_COMPRESSION_SNAPSHOT_POD
          ? "snapshot_pod"
      : deflation->policy.compression == SS_DEFLATION_COMPRESSION_SNAPSHOT_POD_TOTAL
          ? "snapshot_pod_total"
          : "operator_svd";
  provenance =
      deflation->policy.compression == SS_DEFLATION_COMPRESSION_SNAPSHOT_POD
          ? "compress:snapshot-pod"
      : deflation->policy.compression == SS_DEFLATION_COMPRESSION_SNAPSHOT_POD_TOTAL
          ? "compress:snapshot-pod-total"
          : "compress:operator-svd";
  status = ss_deflation_install_compressed_basis(deflation, modes, mode_weights,
                                                 mode_count, protected_indices,
                                                 protected_count, provenance);
  if (status != SS_OK) {
    goto cleanup;
  }
  ++deflation->compression_count;
  deflation->last_compression_input_count = snapshot_count;
  deflation->last_compression_protected_count = protected_count;
  deflation->last_compression_retained_count = mode_count;
  deflation->last_compression_information_fraction = information_fraction;
  if (PetscPrintf(deflation->communicator,
                  "SLOPE_DEFLATION_COMPRESSION %llu,%s,%zu,%zu,%zu,%.17g,%.17g\n",
                  (unsigned long long)deflation->transaction_generation, method_name,
                  snapshot_count, protected_count, mode_count,
                  (double)information_fraction,
                  (double)boundary_ratio) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }

cleanup:
  ss_deflation_vec_array_destroy(modes, mode_count);
  free(mode_weights);
  ss_deflation_vec_array_destroy(snapshots, snapshot_count);
  ss_deflation_vec_array_destroy(protected_basis, protected_rank);
  free(snapshot_weights);
  free(selected_indices);
  free(protected_indices);
  return status;
}
