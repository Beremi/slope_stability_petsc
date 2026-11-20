/* Checked size arithmetic and deterministic error records.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/core/core.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>

static int ss_power_of_two(size_t value) {
  return value != 0U && (value & (value - 1U)) == 0U;
}

SsStatus ss_checked_add_size(size_t left, size_t right, size_t *result) {
  if (result == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (right > SIZE_MAX - left) {
    return SS_ERROR_OVERFLOW;
  }
  *result = left + right;
  return SS_OK;
}

SsStatus ss_checked_mul_size(size_t left, size_t right, size_t *result) {
  if (result == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (left != 0U && right > SIZE_MAX / left) {
    return SS_ERROR_OVERFLOW;
  }
  *result = left * right;
  return SS_OK;
}

SsStatus ss_checked_product_size(const size_t *values, size_t count, size_t *result) {
  size_t index;
  size_t product = 1U;
  if (result == NULL || (values == NULL && count != 0U)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (index = 0U; index < count; ++index) {
    SsStatus status = ss_checked_mul_size(product, values[index], &product);
    if (status != SS_OK) {
      return status;
    }
  }
  *result = product;
  return SS_OK;
}

SsStatus ss_checked_align_size(size_t value, size_t alignment, size_t *result) {
  size_t adjusted;
  if (result == NULL || !ss_power_of_two(alignment)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (ss_checked_add_size(value, alignment - 1U, &adjusted) != SS_OK) {
    return SS_ERROR_OVERFLOW;
  }
  *result = adjusted & ~(alignment - 1U);
  return SS_OK;
}

SsStatus ss_checked_u64_to_size(uint64_t value, size_t *result) {
  size_t converted;
  if (result == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  converted = (size_t)value;
  if ((uint64_t)converted != value) {
    return SS_ERROR_OVERFLOW;
  }
  *result = converted;
  return SS_OK;
}

SsStatus ss_checked_u64_to_hsize(uint64_t value, hsize_t *result) {
  hsize_t converted;
  if (result == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  converted = (hsize_t)value;
  if ((uint64_t)converted != value) {
    return SS_ERROR_OVERFLOW;
  }
  *result = converted;
  return SS_OK;
}

SsStatus ss_checked_u64_to_petsc(uint64_t value, PetscInt *result) {
  if (result == NULL || value > (uint64_t)PETSC_MAX_INT) {
    return result == NULL ? SS_ERROR_INVALID_ARGUMENT : SS_ERROR_OVERFLOW;
  }
  *result = (PetscInt)value;
  return SS_OK;
}

SsStatus ss_checked_u64_to_mpi_count(uint64_t value, MPI_Count *result) {
  MPI_Count converted;
  if (result == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  converted = (MPI_Count)value;
  if (converted < (MPI_Count)0 || (uint64_t)converted != value) {
    return SS_ERROR_OVERFLOW;
  }
  *result = converted;
  return SS_OK;
}

void ss_error_set(SsErrorRecord *record, SsStatus status, SsErrorKey key,
                  const char *code, const char *message) {
  if (record == NULL) {
    return;
  }
  memset(record, 0, sizeof(*record));
  record->valid = 1;
  record->status = status;
  record->key = key;
  if (code != NULL) {
    (void)snprintf(record->code, sizeof(record->code), "%s", code);
  }
  if (message != NULL) {
    (void)snprintf(record->message, sizeof(record->message), "%s", message);
  }
}

int ss_error_key_compare(const SsErrorKey *left, const SsErrorKey *right) {
#define SS_COMPARE_FIELD(field)                                                        \
  do {                                                                                 \
    if (left->field < right->field) {                                                  \
      return -1;                                                                       \
    }                                                                                  \
    if (left->field > right->field) {                                                  \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)
  SS_COMPARE_FIELD(phase);
  SS_COMPARE_FIELD(component);
  SS_COMPARE_FIELD(entity);
  SS_COMPARE_FIELD(quadrature);
  SS_COMPARE_FIELD(thread);
  SS_COMPARE_FIELD(rank);
  return 0;
#undef SS_COMPARE_FIELD
}

int ss_error_compare(const SsErrorRecord *left, const SsErrorRecord *right) {
  if (left == NULL || right == NULL) {
    return left == right ? 0 : (left == NULL ? 1 : -1);
  }
  if (left->valid == 0 || right->valid == 0) {
    return left->valid == right->valid ? 0 : (left->valid == 0 ? 1 : -1);
  }
  const int order = ss_error_key_compare(&left->key, &right->key);
  return order != 0 ? order : strcmp(left->code, right->code);
}

const char *ss_status_name(SsStatus status) {
  static const char *const names[] = {
      "ok",           "invalid_argument",
      "overflow",     "out_of_memory",
      "not_found",    "duplicate",
      "version",      "invalid_state",
      "integrity",    "schema",
      "json",         "limit",
      "invalid_path", "hdf5",
      "resource",     "type_mismatch",
      "cancelled",    "thread",
      "injected",     "io",
      "numerical",    "internal",
  };
  size_t index = (size_t)status;
  return index < sizeof(names) / sizeof(names[0]) ? names[index] : "unknown";
}
