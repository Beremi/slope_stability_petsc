/** @file
 * @brief String, scalar, comparison and partition helpers shared by native
 * modules.
 *
 * All helpers are pure functions of their arguments and safe from any thread.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_CORE_UTIL_H
#define SS_CORE_UTIL_H

#include <petscsystypes.h>

#include <math.h>
#include <stddef.h>

/** Test a double for finiteness.
 * @param[in] value Value to test.
 * @return Nonzero when @p value is neither infinite nor NaN. */
static inline int ss_is_finite(double value) { return isfinite(value) != 0; }

/** Length of a possibly unterminated string.
 * @param[in] value String or NULL.
 * @param[in] capacity Maximum number of bytes to inspect.
 * @return The length capped at @p capacity, or zero for NULL. */
size_t ss_bounded_length(const char *value, size_t capacity);

/** Copy and terminate a string in nonoverlapping storage.
 * @param[out] destination Buffer of @p capacity bytes.
 * @param[in] capacity Size of @p destination including the terminator.
 * @param[in] source NUL-terminated source text.
 * @return One if the text fits, zero on truncation or invalid arguments. */
int ss_copy_text(char *destination, size_t capacity, const char *source);

/** Test for a lowercase hexadecimal string of a given length.
 * @param[in] text NUL-terminated text or NULL.
 * @param[in] size Required number of characters.
 * @return Nonzero when @p text has exactly @p size characters in `[0-9a-f]`. */
int ss_valid_lower_hex(const char *text, size_t size);

/** Test for a canonical content identity `sha256:<64 lowercase hex>`.
 * @param[in] text NUL-terminated text or NULL.
 * @return Nonzero when @p text is a canonical identity. */
int ss_valid_content_id(const char *text);

/** Compare two `PetscInt` values for qsort() or bsearch().
 * @param[in] left Pointer to the first `PetscInt`.
 * @param[in] right Pointer to the second `PetscInt`.
 * @return Negative, zero or positive as `*left` is below, equal to or above
 *   `*right`. */
int ss_petsc_int_compare(const void *left, const void *right);

/** Split a count into contiguous parts that differ by at most one item.
 *
 * The first `global_count % parts` parts receive one extra item.
 *
 * @param[in] global_count Number of items to split.
 * @param[in] rank Part index in `[0, parts)`.
 * @param[in] parts Positive number of parts.
 * @param[out] start First item of part @p rank.
 * @param[out] count Number of items of part @p rank. */
void ss_partition_range(size_t global_count, int rank, int parts, size_t *start,
                        size_t *count);

#endif
