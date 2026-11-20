/* Shared fixtures for ss_test_nonlinear_analysis.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_TEST_NONLINEAR_ANALYSIS_INTERNAL_H
#define SS_TEST_NONLINEAR_ANALYSIS_INTERNAL_H

#include "slope_internal/analysis/analysis.h"
#include "slope_internal/deflation/deflation.h"
#include "slope_internal/nonlinear/nonlinear.h"

#include "fixtures/mechanics_fixture.h"

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

typedef struct {
  PetscInt evaluations;
  PetscInt cancel_after_evaluations;
  PetscInt null_tangent_evaluations;
  PetscInt linearizations;
  PetscReal shell_diagonal;
  Mat shell;
  int reject_nonzero_state;
  int require_null_tangent;
} ScalarModel;

typedef struct {
  SsNonlinearProblem nonlinear;
  SsNonlinearPolicy nonlinear_policy;
  ScalarModel model;
  SsDeflation *outer_deflation;
  PetscInt work_calls;
  PetscInt fixed_calls;
  size_t displacement_maximum_calls;
  int fail_one_limit_load_attempt;
  int fail_after_fixed_solve_call;
  PetscReal maximum_limit_load_predictor_error;
  size_t transaction_begin_count;
  size_t transaction_commit_count;
  size_t transaction_rollback_count;
  size_t deflation_persistent_at_begin;
  int saw_rollback_provisional;
  int deflation_rollback_preserved;
  int deflation_commit_closed;
  int transaction_open;
} AnalysisModel;

/** Create a constant vector. */
int create_vector(PetscInt size, PetscReal value, Vec *vector);
/** Evaluation callback of the scalar nonlinear model. */
SsStatus scalar_evaluate(void *opaque, Vec state, PetscReal factor, Vec residual,
                         Mat tangent, PetscReal *functional);
/** Create the scalar nonlinear model with its load and elastic operator. */
int create_scalar_nonlinear(AnalysisModel *model, PetscInt size, Vec *load,
                            Mat *elastic);
/** Check that every entry equals a value within a tolerance; 0 when equal. */
int check_scalar(Vec vector, PetscReal expected, PetscReal tolerance);
/** Case: adaptive work-step controller of indirect SSR. */
int test_indirect_adaptive_work_controller(void);
/** Case: work-efficiency controller of indirect SSR. */
int test_indirect_work_efficiency_controller(void);
/** Case: direct SSR control coordinate. */
int test_direct_coordinate(void);
/** Case: limit-load continuation. */
int test_limit_load(void);
/** Case: direct SSR trajectory. */
int test_direct_trajectory(void);
/** Case: indirect SSR trajectory. */
int test_indirect_trajectory(void);

#endif
