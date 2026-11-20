/* Isolated OpenMP reduction probe of the tsan-kernels preset.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/build/config.h"

#include <omp.h>

#include <stdio.h>
#include <stdlib.h>

int main(void) {
  enum { count = 4096 };
  int *values = (int *)calloc((size_t)count, sizeof(*values));
  long long sum = 0;
  int team = 0;
  int index;
  if (values == NULL) {
    return 2;
  }
#pragma omp parallel
  {
#pragma omp single
    team = omp_get_num_threads();
#pragma omp for schedule(static)
    for (index = 0; index < count; ++index) {
      values[index] = index + 1;
    }
#pragma omp for reduction(+ : sum) schedule(static)
    for (index = 0; index < count; ++index) {
      sum += values[index];
    }
  }
  free(values);
  if (team < 2 || sum != ((long long)count * (count + 1)) / 2) {
    fprintf(stderr, "invalid isolated OpenMP probe team=%d sum=%lld\n", team, sum);
    return 3;
  }
  printf("{\"openmp_team\":%d,\"race_free\":true,\"sum\":%lld}\n", team, sum);
  return 0;
}
