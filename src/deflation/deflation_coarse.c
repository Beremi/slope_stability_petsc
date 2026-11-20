/* Deflation coarse factorization and left/right projector actions.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "deflation_internal.h"
#include "slope_internal/core/util.h"

#include <stdlib.h>
#include <string.h>

SsStatus ss_deflation_gram(SsDeflation *deflation, const Vec *right, const Vec *left,
                           size_t dimension, PetscScalar *gram) {
  PetscMPIInt count = 0;
  SsStatus status = SS_OK;
  if (dimension == 0U || dimension > (size_t)PETSC_MAX_INT / dimension ||
      PetscMPIIntCast((PetscInt)(dimension * dimension), &count) != PETSC_SUCCESS) {
    return SS_ERROR_OVERFLOW;
  }
  /* Complete local dot products before agreeing status. Using a split-reduction
   * queue here would leave colliding collective sequences if local setup failed. */
  for (size_t col = 0U; status == SS_OK && col < dimension; ++col) {
    const PetscScalar *y = NULL;
    PetscInt local = 0;
    if (VecGetLocalSize(right[col], &local) != PETSC_SUCCESS ||
        VecGetArrayRead(right[col], &y) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
      break;
    }
    for (size_t row = 0U; status == SS_OK && row < dimension; ++row) {
      const PetscScalar *x = NULL;
      PetscInt left_size = 0;
      if (VecGetLocalSize(left[row], &left_size) != PETSC_SUCCESS ||
          local != left_size || VecGetArrayRead(left[row], &x) != PETSC_SUCCESS) {
        status = SS_ERROR_TYPE_MISMATCH;
        break;
      }
      PetscScalar dot = 0.0;
      for (PetscInt i = 0; i < local; ++i) {
        dot += PetscConj(x[i]) * y[i];
      }
      gram[col * dimension + row] = dot;
      if (VecRestoreArrayRead(left[row], &x) != PETSC_SUCCESS) {
        status = SS_ERROR_INTERNAL;
      }
    }
    if (VecRestoreArrayRead(right[col], &y) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  status = ss_status_allreduce_max(deflation->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  if (MPI_Allreduce(MPI_IN_PLACE, gram, count, MPIU_SCALAR, MPI_SUM,
                    deflation->communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  ++deflation->global_reductions;
  return SS_OK;
}

void ss_deflation_clear_factorization(SsDeflation *deflation) {
  free(deflation->coarse);
  free(deflation->lu);
  free(deflation->dots);
  free(deflation->coefficients);
  free(deflation->pivots);
  free(deflation->active_left);
  free(deflation->active_right);
  free(deflation->active_az);
  free(deflation->active_atz);
  deflation->coarse = NULL;
  deflation->lu = NULL;
  deflation->dots = NULL;
  deflation->coefficients = NULL;
  deflation->pivots = NULL;
  deflation->active_left = NULL;
  deflation->active_right = NULL;
  deflation->active_az = NULL;
  deflation->active_atz = NULL;
  deflation->factors_ready = 0;
  deflation->invariants_validated = 0;
}

void ss_deflation_fill_record(const SsDeflation *deflation,
                              SsDeflationAdmissionRecord *record) {
  record->persistent_count = deflation->persistent_count;
  record->provisional_count = deflation->provisional_count;
  record->active_count = deflation->active_count;
  record->global_reductions = deflation->global_reductions;
}

static SsStatus ss_deflation_scalar_magnitude(const PetscScalar *values, size_t count,
                                              size_t index, PetscReal *magnitude) {
  if (values == NULL || magnitude == NULL || index >= count) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  /*
   * The explicit index/count guard is the allocation contract. Clang's
   * path-sensitive analyzer loses the symbolic calloc extent when refresh is
   * reached through rollback and otherwise reports the second valid element
   * of a dimension-squared allocation as out of bounds.
   */
  *magnitude =
      PetscAbsScalar(values[index]); /* NOLINT(clang-analyzer-security.ArrayBound) */
  return SS_OK;
}

SsStatus ss_deflation_factorize(SsDeflation *deflation) {
  const size_t dimension = deflation->active_count;
  const size_t element_count = dimension * dimension;
  PetscReal maximum = 0.0;
  PetscReal coarse_norm = 0.0;
  PetscReal minimum_pivot = PETSC_MAX_REAL;
  PetscReal inverse_norm = 0.0;
  PetscReal *inverse_row_sums = NULL;
  PetscScalar *solve_work = NULL;
  size_t column = 0U;
  size_t row;
  memcpy(deflation->lu, deflation->coarse, element_count * sizeof(*deflation->lu));
  for (row = 0U; row < element_count; ++row) {
    PetscReal magnitude = 0.0;
    if (ss_deflation_scalar_magnitude(deflation->lu, element_count, row, &magnitude) !=
        SS_OK) {
      return SS_ERROR_INTERNAL;
    }
    if (!ss_is_finite(magnitude)) {
      return SS_ERROR_INTEGRITY;
    }
    if (magnitude > maximum) {
      maximum = magnitude;
    }
  }
  for (row = 0U; row < dimension; ++row) {
    PetscReal row_sum = 0.0;
    for (column = 0U; column < dimension; ++column) {
      PetscReal magnitude = 0.0;
      if (ss_deflation_scalar_magnitude(deflation->lu, element_count,
                                        row * dimension + column,
                                        &magnitude) != SS_OK) {
        return SS_ERROR_INTERNAL;
      }
      row_sum += magnitude;
    }
    if (row_sum > coarse_norm) {
      coarse_norm = row_sum;
    }
  }
  if (maximum <= deflation->policy.norm_tolerance) {
    return SS_ERROR_INTEGRITY;
  }
  for (column = 0U; column < dimension; ++column) {
    size_t pivot = column;
    PetscReal pivot_magnitude =
        PetscAbsScalar(deflation->lu[column * dimension + column]);
    for (row = column + 1U; row < dimension; ++row) {
      const PetscReal candidate =
          PetscAbsScalar(deflation->lu[row * dimension + column]);
      if (candidate > pivot_magnitude) {
        pivot = row;
        pivot_magnitude = candidate;
      }
    }
    deflation->pivots[column] = (PetscInt)pivot;
    if (!ss_is_finite(pivot_magnitude) ||
        pivot_magnitude / maximum <= deflation->policy.pivot_tolerance) {
      deflation->minimum_relative_pivot = pivot_magnitude / maximum;
      deflation->condition_estimate = PETSC_MAX_REAL;
      return SS_ERROR_INTEGRITY;
    }
    if (pivot != column) {
      size_t swap_column;
      for (swap_column = 0U; swap_column < dimension; ++swap_column) {
        const size_t first = column * dimension + swap_column;
        const size_t second = pivot * dimension + swap_column;
        const PetscScalar temporary = deflation->lu[first];
        deflation->lu[first] = deflation->lu[second];
        deflation->lu[second] = temporary;
      }
    }
    pivot_magnitude = PetscAbsScalar(deflation->lu[column * dimension + column]);
    if (pivot_magnitude < minimum_pivot) {
      minimum_pivot = pivot_magnitude;
    }
    for (row = column + 1U; row < dimension; ++row) {
      size_t update;
      deflation->lu[row * dimension + column] /=
          deflation->lu[column * dimension + column];
      for (update = column + 1U; update < dimension; ++update) {
        deflation->lu[row * dimension + update] -=
            deflation->lu[row * dimension + column] *
            deflation->lu[column * dimension + update];
      }
    }
  }
  deflation->minimum_relative_pivot = minimum_pivot / maximum;
  inverse_row_sums = (PetscReal *)calloc(dimension, sizeof(*inverse_row_sums));
  solve_work = (PetscScalar *)calloc(dimension, sizeof(*solve_work));
  if (inverse_row_sums == NULL || solve_work == NULL) {
    free(solve_work);
    free(inverse_row_sums);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (column = 0U; column < dimension; ++column) {
    memset(solve_work, 0, dimension * sizeof(*solve_work));
    solve_work[column] = 1.0;
    (void)ss_deflation_lu_solve(deflation, solve_work);
    for (row = 0U; row < dimension; ++row) {
      inverse_row_sums[row] += PetscAbsScalar(solve_work[row]);
    }
  }
  for (row = 0U; row < dimension; ++row) {
    if (inverse_row_sums[row] > inverse_norm) {
      inverse_norm = inverse_row_sums[row];
    }
  }
  free(solve_work);
  free(inverse_row_sums);
  deflation->condition_estimate = coarse_norm * inverse_norm;
  if (!ss_is_finite(deflation->condition_estimate) ||
      deflation->condition_estimate > deflation->policy.condition_limit) {
    return SS_ERROR_INTEGRITY;
  }
  return SS_OK;
}

SsStatus ss_deflation_lu_solve(const SsDeflation *deflation, PetscScalar *values) {
  const size_t dimension = deflation->active_count;
  size_t row;
  size_t column;
  /* P A = L U, so solve L U x = P b with the full pivot sequence first. */
  for (column = 0U; column < dimension; ++column) {
    const size_t pivot = (size_t)deflation->pivots[column];
    if (pivot != column) {
      const PetscScalar temporary = values[column];
      values[column] = values[pivot];
      values[pivot] = temporary;
    }
  }
  for (row = 0U; row < dimension; ++row) {
    for (column = 0U; column < row; ++column) {
      values[row] -= deflation->lu[row * dimension + column] * values[column];
    }
  }
  for (row = dimension; row-- > 0U;) {
    for (column = row + 1U; column < dimension; ++column) {
      values[row] -= deflation->lu[row * dimension + column] * values[column];
    }
    values[row] /= deflation->lu[row * dimension + row];
  }
  return SS_OK;
}

static SsStatus ss_deflation_lu_transpose_solve(const SsDeflation *deflation,
                                                PetscScalar *values) {
  const size_t dimension = deflation->active_count;
  size_t row;
  size_t column;
  for (row = 0U; row < dimension; ++row) {
    for (column = 0U; column < row; ++column) {
      values[row] -= deflation->lu[column * dimension + row] * values[column];
    }
    values[row] /= deflation->lu[row * dimension + row];
  }
  for (row = dimension; row-- > 0U;) {
    for (column = row + 1U; column < dimension; ++column) {
      values[row] -= deflation->lu[column * dimension + row] * values[column];
    }
  }
  for (column = dimension; column-- > 0U;) {
    const size_t pivot = (size_t)deflation->pivots[column];
    if (pivot != column) {
      const PetscScalar temporary = values[column];
      values[column] = values[pivot];
      values[pivot] = temporary;
    }
  }
  return SS_OK;
}

/*
 * Validate the projector identities through their coarse-space residuals.
 *
 * Let E = W^T A Z.  If C = E^-1 E and D = E^-T E^T, then
 * Q A Z = Z C, P_R Z = Z(I-C), and P_L^T W = W(I-D).  Idempotence of both
 * projectors and P_L A = A P_R follow from the same two residuals and the
 * shared cached A Z columns.  The basis columns are Euclidean-normalized at
 * admission, so a column one-norm is also a conservative bound on the
 * corresponding distributed-vector error.
 *
 * This is the mandatory validation used on every operator generation.  It
 * catches factor/permutation defects (including late-row LU pivots) without
 * replaying every full distributed projector action for every basis vector.
 * ss_deflation_invariant_error() remains the exhaustive distributed oracle
 * available to contract tests and diagnostics.
 */
SsStatus ss_deflation_coarse_invariant_error(SsDeflation *deflation,
                                             PetscReal *maximum_error) {
  const size_t dimension = deflation == NULL ? 0U : deflation->active_count;
  PetscScalar *work = NULL;
  PetscReal maximum = 0.0;
  size_t column;
  size_t row;
  if (deflation == NULL || maximum_error == NULL || !deflation->factors_ready ||
      dimension == 0U || deflation->coarse == NULL || deflation->lu == NULL ||
      deflation->pivots == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  work = (PetscScalar *)calloc(dimension, sizeof(*work));
  if (work == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (column = 0U; column < dimension; ++column) {
    PetscReal direct_error = 0.0;
    PetscReal transpose_error = 0.0;
    for (row = 0U; row < dimension; ++row) {
      work[row] = deflation->coarse[row * dimension + column];
    }
    (void)ss_deflation_lu_solve(deflation, work);
    for (row = 0U; row < dimension; ++row) {
      const PetscScalar expected = row == column ? 1.0 : 0.0;
      direct_error += PetscAbsScalar(work[row] - expected);
    }
    for (row = 0U; row < dimension; ++row) {
      work[row] = deflation->coarse[column * dimension + row];
    }
    (void)ss_deflation_lu_transpose_solve(deflation, work);
    for (row = 0U; row < dimension; ++row) {
      const PetscScalar expected = row == column ? 1.0 : 0.0;
      transpose_error += PetscAbsScalar(work[row] - expected);
    }
    if (!ss_is_finite(direct_error) || !ss_is_finite(transpose_error)) {
      free(work);
      deflation->maximum_invariant_error = PETSC_MAX_REAL;
      deflation->invariants_validated = 0;
      *maximum_error = PETSC_MAX_REAL;
      return SS_ERROR_INTEGRITY;
    }
    if (direct_error > maximum) {
      maximum = direct_error;
    }
    if (transpose_error > maximum) {
      maximum = transpose_error;
    }
  }
  free(work);
  deflation->maximum_invariant_error = maximum;
  deflation->invariants_validated = maximum <= deflation->policy.invariant_tolerance;
  *maximum_error = maximum;
  return deflation->invariants_validated != 0 ? SS_OK : SS_ERROR_INTEGRITY;
}

static SsStatus ss_deflation_coarse_coefficients(SsDeflation *deflation, Vec input,
                                                 int transpose) {
  Vec *basis = transpose ? deflation->active_right : deflation->active_left;
  const PetscInt count = (PetscInt)deflation->active_count;
  if (VecMDot(input, count, basis, deflation->coefficients) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  ++deflation->global_reductions;
  const SsStatus status =
      transpose ? ss_deflation_lu_transpose_solve(deflation, deflation->coefficients)
                : ss_deflation_lu_solve(deflation, deflation->coefficients);
  if (status == SS_OK) {
    ++deflation->q_apply_count;
  }
  return status;
}

static SsStatus ss_deflation_q_into(SsDeflation *deflation, Vec input, Vec output,
                                    int transpose) {
  if (!deflation->factors_ready) {
    return SS_ERROR_INVALID_STATE;
  }
  if (deflation->active_count == 0U) {
    return VecSet(output, 0.0) == PETSC_SUCCESS ? SS_OK : SS_ERROR_INTERNAL;
  }
  if (deflation->active_count > (size_t)PETSC_MAX_INT) {
    return SS_ERROR_OVERFLOW;
  }
  SsStatus status = ss_deflation_coarse_coefficients(deflation, input, transpose);
  if (status != SS_OK) {
    return status;
  }
  Vec *basis = transpose ? deflation->active_left : deflation->active_right;
  return VecSet(output, 0.0) == PETSC_SUCCESS &&
                 VecMAXPY(output, (PetscInt)deflation->active_count,
                          deflation->coefficients, basis) == PETSC_SUCCESS
             ? SS_OK
             : SS_ERROR_INTERNAL;
}

static SsStatus ss_deflation_cached_az_into(SsDeflation *deflation, Vec output,
                                            PetscScalar scale, int initialize) {
  PetscInt count;
  size_t index;
  if (deflation->active_count == 0U) {
    return initialize != 0 && VecSet(output, 0.0) != PETSC_SUCCESS ? SS_ERROR_INTERNAL
                                                                   : SS_OK;
  }
  if (deflation->active_count > (size_t)PETSC_MAX_INT || deflation->active_az == NULL) {
    return SS_ERROR_INVALID_STATE;
  }
  count = (PetscInt)deflation->active_count;
  for (index = 0U; index < deflation->active_count; ++index) {
    deflation->dots[index] = scale * deflation->coefficients[index];
  }
  if ((initialize != 0 && VecSet(output, 0.0) != PETSC_SUCCESS) ||
      VecMAXPY(output, count, deflation->dots, deflation->active_az) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  return SS_OK;
}

SsStatus ss_deflation_operator_ready(const SsDeflation *deflation,
                                     Mat operator_handle) {
  PetscObjectState current_state = 0;
  if (!deflation->factors_ready || deflation->operator_handle == NULL) {
    return SS_ERROR_INVALID_STATE;
  }
  if (deflation->operator_handle != operator_handle ||
      MatGetState(operator_handle, &current_state) != PETSC_SUCCESS ||
      current_state != deflation->operator_state) {
    return SS_ERROR_INVALID_STATE;
  }
  return SS_OK;
}

static SsStatus ss_deflation_right_into(SsDeflation *deflation, Mat operator_handle,
                                        Vec input, Vec output, Vec ax, Vec qax) {
  SsStatus status;
  if (deflation->active_count == 0U) {
    return VecCopy(input, output) == PETSC_SUCCESS ? SS_OK : SS_ERROR_INTERNAL;
  }
  status = ss_deflation_petsc_status(deflation, MatMult(operator_handle, input, ax));
  if (status != SS_OK) {
    return status;
  }
  status = ss_deflation_q_into(deflation, ax, qax, 0);
  if (status != SS_OK) {
    return status;
  }
  return VecWAXPY(output, -1.0, qax, input) == PETSC_SUCCESS ? SS_OK
                                                             : SS_ERROR_INTERNAL;
}

static SsStatus ss_deflation_left_into(SsDeflation *deflation, Mat operator_handle,
                                       Vec input, Vec output, Vec qinput, Vec aqinput) {
  SsStatus status;
  (void)operator_handle;
  if (deflation->active_count == 0U) {
    return VecCopy(input, output) == PETSC_SUCCESS ? SS_OK : SS_ERROR_INTERNAL;
  }
  status = ss_deflation_q_into(deflation, input, qinput, 0);
  if (status != SS_OK) {
    return status;
  }
  status = ss_deflation_cached_az_into(deflation, aqinput, 1.0, 1);
  if (status != SS_OK) {
    return status;
  }
  return VecWAXPY(output, -1.0, aqinput, input) == PETSC_SUCCESS ? SS_OK
                                                                 : SS_ERROR_INTERNAL;
}

static SsStatus ss_deflation_left_transpose_into(SsDeflation *deflation,
                                                 Mat operator_handle, Vec input,
                                                 Vec output, Vec at_input,
                                                 Vec qt_at_input) {
  SsStatus status;
  if (deflation->active_count == 0U) {
    return VecCopy(input, output) == PETSC_SUCCESS ? SS_OK : SS_ERROR_INTERNAL;
  }
  status = ss_deflation_petsc_status(
      deflation, MatMultTranspose(operator_handle, input, at_input));
  if (status != SS_OK) {
    return status;
  }
  status = ss_deflation_q_into(deflation, at_input, qt_at_input, 1);
  if (status != SS_OK) {
    return status;
  }
  return VecWAXPY(output, -1.0, qt_at_input, input) == PETSC_SUCCESS
             ? SS_OK
             : SS_ERROR_INTERNAL;
}

SsStatus ss_deflation_apply_q(SsDeflation *deflation, Vec input, Vec output) {
  if (deflation == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  const SsStatus status = ss_status_allreduce_max(
      deflation->communicator,
      input == NULL || output == NULL
          ? SS_ERROR_INVALID_ARGUMENT
          : ss_deflation_operator_ready(deflation, deflation->operator_handle));
  return status == SS_OK
             ? ss_status_allreduce_max(deflation->communicator,
                                       ss_deflation_q_into(deflation, input, output, 0))
             : status;
}

SsStatus ss_deflation_apply_right(SsDeflation *deflation, Mat operator_handle,
                                  Vec input, Vec output) {
  SsStatus status;
  if (deflation == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_status_allreduce_max(
      deflation->communicator,
      operator_handle == NULL || input == NULL || output == NULL
          ? SS_ERROR_INVALID_ARGUMENT
          : ss_deflation_operator_ready(deflation, operator_handle));
  if (status != SS_OK) {
    return status;
  }
  return ss_status_allreduce_max(
      deflation->communicator,
      ss_deflation_right_into(deflation, operator_handle, input, output,
                              deflation->work[0], deflation->work[1]));
}

SsStatus ss_deflation_apply_left(SsDeflation *deflation, Mat operator_handle, Vec input,
                                 Vec output) {
  SsStatus status;
  if (deflation == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_status_allreduce_max(
      deflation->communicator,
      operator_handle == NULL || input == NULL || output == NULL
          ? SS_ERROR_INVALID_ARGUMENT
          : ss_deflation_operator_ready(deflation, operator_handle));
  if (status != SS_OK) {
    return status;
  }
  return ss_status_allreduce_max(
      deflation->communicator,
      ss_deflation_left_into(deflation, operator_handle, input, output,
                             deflation->work[0], deflation->work[1]));
}

SsStatus ss_deflation_apply_projected_operator_local(SsDeflation *deflation,
                                                     Mat operator_handle, Vec input,
                                                     Vec output) {
  if (deflation->active_count == 0U) {
    return MatMult(operator_handle, input, output) == PETSC_SUCCESS ? SS_OK
                                                                    : SS_ERROR_INTERNAL;
  }
  Vec action = deflation->work[1];
  const PetscErrorCode mult = MatMult(operator_handle, input, action);
  /* A rank-local MatMult error must not bypass the collective coarse dot. */
  const PetscErrorCode flag = VecFlag(action, mult != PETSC_SUCCESS);
  SsStatus status = ss_deflation_coarse_coefficients(deflation, action, 0);
  if (mult != PETSC_SUCCESS || flag != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (status == SS_OK && VecCopy(action, output) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  /* A P_R = A - A Z (W^T A Z)^-1 W^T A. Use the cached A Z columns to
   * form this action with just the W^T A dot product. */
  if (status == SS_OK) {
    status = ss_deflation_cached_az_into(deflation, output, -1.0, 0);
  }
  if (status == SS_OK) {
    ++deflation->projected_apply_count;
  }
  return status;
}

SsStatus ss_deflation_apply_projected_operator(SsDeflation *deflation,
                                               Mat operator_handle, Vec input,
                                               Vec output) {
  if (deflation == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  const SsStatus status = ss_status_allreduce_max(
      deflation->communicator,
      operator_handle == NULL || input == NULL || output == NULL
          ? SS_ERROR_INVALID_ARGUMENT
          : ss_deflation_operator_ready(deflation, operator_handle));
  return status == SS_OK
             ? ss_status_allreduce_max(deflation->communicator,
                                       ss_deflation_apply_projected_operator_local(
                                           deflation, operator_handle, input, output))
             : status;
}

SsStatus ss_deflation_project_rhs(SsDeflation *deflation, Mat operator_handle, Vec rhs,
                                  Vec projected_rhs) {
  return ss_deflation_apply_left(deflation, operator_handle, rhs, projected_rhs);
}

SsStatus ss_deflation_final_correction(SsDeflation *deflation, Mat operator_handle,
                                       Vec rhs, Vec complement, Vec solution) {
  SsStatus status;
  if (deflation == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_status_allreduce_max(
      deflation->communicator,
      operator_handle == NULL || rhs == NULL || complement == NULL || solution == NULL
          ? SS_ERROR_INVALID_ARGUMENT
          : ss_deflation_operator_ready(deflation, operator_handle));
  if (status != SS_OK) {
    return status;
  }
  if (deflation->active_count == 0U) {
    return ss_deflation_petsc_status(deflation, VecCopy(complement, solution));
  }
  status = ss_status_allreduce_max(
      deflation->communicator,
      ss_deflation_q_into(deflation, rhs, deflation->work[0], 0));
  if (status != SS_OK) {
    return status;
  }
  status = ss_deflation_right_into(deflation, operator_handle, complement,
                                   deflation->work[1], deflation->work[2],
                                   deflation->work[3]);
  if (status == SS_OK && VecWAXPY(solution, 1.0, deflation->work[1],
                                  deflation->work[0]) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  return ss_status_allreduce_max(deflation->communicator, status);
}

SsStatus ss_deflation_invariant_error(SsDeflation *deflation, Mat operator_handle,
                                      PetscReal *maximum_error) {
  Vec vectors[5] = {NULL, NULL, NULL, NULL, NULL};
  PetscReal maximum = 0.0;
  size_t index;
  int probe_kind;
  if (deflation == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  SsStatus status = ss_status_allreduce_max(
      deflation->communicator,
      operator_handle == NULL || maximum_error == NULL
          ? SS_ERROR_INVALID_ARGUMENT
          : ss_deflation_operator_ready(deflation, operator_handle));
  if (status != SS_OK) {
    return status;
  }
  if (deflation->active_count == 0U) {
    *maximum_error = 0.0;
    return SS_OK;
  }
  for (index = 0U; index < 5U; ++index) {
    status = ss_deflation_petsc_status(
        deflation,
        VecDuplicate(ss_deflation_entry(deflation, 0U)->right, &vectors[index]));
    if (status != SS_OK) {
      goto cleanup;
    }
  }
  for (index = 0U; index < deflation->active_count; ++index) {
    const SsDeflationEntry *entry = ss_deflation_const_entry(deflation, index);
    PetscReal error = 0.0;
    status =
        ss_deflation_apply_right(deflation, operator_handle, entry->right, vectors[0]);
    if (status != SS_OK ||
        ss_deflation_petsc_status(deflation, VecNorm(vectors[0], NORM_2, &error)) !=
            SS_OK) {
      maximum = PETSC_MAX_REAL;
      break;
    }
    ++deflation->global_reductions;
    if (error > maximum) {
      maximum = error;
    }
    status = ss_deflation_q_into(deflation, entry->az, vectors[1], 0);
    if (status == SS_OK && VecAXPY(vectors[1], -1.0, entry->right) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
    if (ss_status_allreduce_max(deflation->communicator, status) != SS_OK ||
        ss_deflation_petsc_status(deflation, VecNorm(vectors[1], NORM_2, &error)) !=
            SS_OK) {
      maximum = PETSC_MAX_REAL;
      break;
    }
    ++deflation->global_reductions;
    if (error > maximum) {
      maximum = error;
    }
    status = ss_deflation_left_transpose_into(deflation, operator_handle, entry->left,
                                              vectors[2], vectors[3], vectors[4]);
    if (ss_status_allreduce_max(deflation->communicator, status) != SS_OK ||
        ss_deflation_petsc_status(deflation, VecNorm(vectors[2], NORM_2, &error)) !=
            SS_OK) {
      maximum = PETSC_MAX_REAL;
      break;
    }
    ++deflation->global_reductions;
    if (error > maximum) {
      maximum = error;
    }
    for (probe_kind = 0; probe_kind < 2; ++probe_kind) {
      Vec probe = probe_kind == 0 ? entry->right : entry->left;
      status = ss_deflation_apply_right(deflation, operator_handle, probe, vectors[0]);
      if (status != SS_OK ||
          ss_deflation_petsc_status(
              deflation, MatMult(operator_handle, vectors[0], vectors[1])) != SS_OK ||
          ss_deflation_petsc_status(
              deflation, MatMult(operator_handle, probe, vectors[2])) != SS_OK) {
        maximum = PETSC_MAX_REAL;
        break;
      }
      status =
          ss_deflation_apply_left(deflation, operator_handle, vectors[2], vectors[3]);
      if (status != SS_OK ||
          ss_deflation_petsc_status(
              deflation, VecWAXPY(vectors[4], -1.0, vectors[3], vectors[1])) != SS_OK ||
          ss_deflation_petsc_status(deflation, VecNorm(vectors[4], NORM_2, &error)) !=
              SS_OK) {
        maximum = PETSC_MAX_REAL;
        break;
      }
      ++deflation->global_reductions;
      if (error > maximum) {
        maximum = error;
      }
    }
    if (maximum == PETSC_MAX_REAL) {
      break;
    }
  }
  status = maximum == PETSC_MAX_REAL ? SS_ERROR_INTERNAL : SS_OK;
cleanup:
  for (index = 0U; index < 5U; ++index) {
    (void)VecDestroy(&vectors[index]);
  }
  if (status != SS_OK) {
    maximum = PETSC_MAX_REAL;
  }
  deflation->maximum_invariant_error = maximum;
  deflation->invariants_validated =
      maximum != PETSC_MAX_REAL && maximum <= deflation->policy.invariant_tolerance;
  *maximum_error = maximum;
  return status;
}
