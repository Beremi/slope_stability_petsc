/* Shared fixtures for ss_test_linear_deflation.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_TEST_LINEAR_DEFLATION_INTERNAL_H
#define SS_TEST_LINEAR_DEFLATION_INTERNAL_H

#include "slope_internal/deflation/deflation.h"
#include "slope_internal/linear/linear.h"

#include <petscksp.h>

#include <math.h>
#include <stdio.h>
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
    const double checked_actual = (double)(actual);                                    \
    const double checked_expected = (double)(expected);                                \
    if (!isfinite(checked_actual) ||                                                   \
        fabs(checked_actual - checked_expected) > (double)(tolerance)) {               \
      fprintf(stderr, "%s:%d: |%.17g - %.17g| > %.3g\n", __FILE__, __LINE__,           \
              checked_actual, checked_expected, (double)(tolerance));                  \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

typedef struct {
  PetscScalar values[4];
} Shell2x2;

/** Create a sequential dense matrix from row-major values. */
int create_dense(PetscInt size, const double *values, Mat *matrix);
/** Create a vector from values. */
int create_vector(PetscInt size, const double *values, Vec *vector);
/** Compare a vector with expected values within a tolerance; 0 when equal. */
int check_vector(Vec vector, PetscInt size, const double *expected, double tolerance);
/** Compare two vectors bit for bit; 0 when identical. */
int check_vector_bits(Vec actual, Vec expected);
/** Operator properties of a general nonsymmetric test operator. */
SsLinearOperatorProperties general_properties(void);
/** FGMRES policy of the deflation tests. */
SsLinearPolicy general_policy(const char *identity, const char *prefix);
/** General left/right deflation policy with a basis cap. */
SsDeflationPolicy deflation_policy(size_t cap);
/** Case: the projector identities respect the configured tolerance. */
int test_configured_invariant_tolerance(void);
/** Case: projector identities of the general projector. */
int test_projector(void);
/** Case: the legacy A-orthogonal projector. */
int test_legacy_aorth_projector(void);
/** Case: coarse LU with a late row pivot. */
int test_lu_late_row_pivot(void);
/** Case: admission keeps the sign of a direction. */
int test_signed_direction(void);
/** Case: oblique left and right bases. */
int test_oblique(void);
/** Case: stopping threshold of a deflated solve. */
int test_deflated_stopping_threshold(void);
/** Case: PMG on the projected system. */
int test_deflated_pmg_projection(void);
/** Case: true residual with a reused guess under A-orthogonal deflation. */
int test_aorth_true_residual_reused_guess(void);
/** Case: dependent candidates are dropped. */
int test_dependent(void);
/** Case: Krylov snapshot compression. */
int test_krylov_compression(void);
/** Case: bitwise restore of a persistent basis. */
int test_persistent_restore_bits(void);
/** Case: rollback restores the previous basis. */
int test_rollback(void);

#endif
