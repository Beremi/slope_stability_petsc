/* Shared string, comparison and partition helpers for native modules.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/core/util.h"

#include <string.h>

size_t ss_bounded_length(const char *value, size_t capacity) {
  size_t length = 0U;
  if (value == NULL) {
    return 0U;
  }
  while (length < capacity && value[length] != '\0') {
    ++length;
  }
  return length;
}

int ss_copy_text(char *destination, size_t capacity, const char *source) {
  if (destination == NULL || capacity == 0U || source == NULL) {
    return 0;
  }
  const size_t length = ss_bounded_length(source, capacity);
  const size_t copied = length < capacity ? length : capacity - 1U;
  memcpy(destination, source, copied);
  destination[copied] = '\0';
  return length < capacity;
}

int ss_valid_lower_hex(const char *text, size_t size) {
  size_t index;
  if (text == NULL || strlen(text) != size) {
    return 0;
  }
  for (index = 0U; index < size; ++index) {
    if (!((text[index] >= '0' && text[index] <= '9') ||
          (text[index] >= 'a' && text[index] <= 'f'))) {
      return 0;
    }
  }
  return 1;
}

int ss_valid_content_id(const char *text) {
  return text != NULL && strlen(text) == 71U && strncmp(text, "sha256:", 7U) == 0 &&
         ss_valid_lower_hex(text + 7U, 64U);
}

int ss_petsc_int_compare(const void *left, const void *right) {
  const PetscInt lhs = *(const PetscInt *)left;
  const PetscInt rhs = *(const PetscInt *)right;
  return lhs < rhs ? -1 : lhs > rhs ? 1 : 0;
}

void ss_partition_range(size_t global_count, int rank, int parts, size_t *start,
                        size_t *count) {
  const size_t quotient = global_count / (size_t)parts;
  const size_t remainder = global_count % (size_t)parts;
  *count = quotient + ((size_t)rank < remainder ? 1U : 0U);
  *start =
      (size_t)rank * quotient + ((size_t)rank < remainder ? (size_t)rank : remainder);
}
