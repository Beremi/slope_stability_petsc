/** @file
 * @brief Sorting helper shared by the native and standalone protocol code.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_CORE_SORT_H
#define SS_CORE_SORT_H

#include <stddef.h>

/** Sort an array with qsort() semantics; the order of equal elements is
 * unspecified.
 *
 * A count below two is a no-op and permits a null base and comparator.
 *
 * @param[in,out] base First of @p count elements.
 * @param[in] count Number of elements.
 * @param[in] size Size of one element in bytes.
 * @param[in] compare Comparator with qsort() semantics. */
void ss_sort(void *base, size_t count, size_t size,
             int (*compare)(const void *, const void *));

#endif
