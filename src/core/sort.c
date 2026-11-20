/* Sort only arrays that need ordering.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/core/sort.h"

#include <stdlib.h>

void ss_sort(void *base, size_t count, size_t size,
             int (*compare)(const void *, const void *)) {
  if (count >= 2U) {
    qsort(base, count, size, compare);
  }
}
