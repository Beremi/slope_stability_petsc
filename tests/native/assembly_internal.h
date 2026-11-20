/* Shared fixtures for ss_test_assembly.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_TEST_ASSEMBLY_INTERNAL_H
#define SS_TEST_ASSEMBLY_INTERNAL_H

#include "slope_internal/assembly/assembly.h"

#include <petscpc.h>
#include <petscsys.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expression)                                                              \
  do {                                                                                 \
    if (!(expression)) {                                                               \
      fprintf(stderr, "%s:%d: assertion failed: %s\n", __FILE__, __LINE__,             \
              #expression);                                                            \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

#define CHECK_NEAR(actual, expected, tolerance)                                        \
  do {                                                                                 \
    const double checked_actual = (actual);                                            \
    const double checked_expected = (expected);                                        \
    if (!isfinite(checked_actual) ||                                                   \
        fabs(checked_actual - checked_expected) > (tolerance)) {                       \
      fprintf(stderr, "%s:%d: |%.17g - %.17g| > %.3g\n", __FILE__, __LINE__,           \
              checked_actual, checked_expected, (double)(tolerance));                  \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

/** Two-triangle unit-square mesh input of the assembly tests. */
SsMeshInput square_input(void);
/** Single-tetrahedron mesh input of the assembly tests. */
SsMeshInput tetra_input(void);
/** Run the matrix-free operator cases with a thread count; 0 on success. */
int test_matrix_free(int requested_threads);

#endif
