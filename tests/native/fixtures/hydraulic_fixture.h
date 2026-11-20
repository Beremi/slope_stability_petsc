/* Compact and case-slice seepage problems of the hydraulic tests.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_TEST_HYDRAULIC_FIXTURE_H
#define SS_TEST_HYDRAULIC_FIXTURE_H

#include "slope_internal/nonlinear/nonlinear.h"

#include <stdint.h>

typedef struct {
  uint32_t dimension;
  uint32_t field_order;
  PetscInt free_dof_count;
  double accepted_free_pressure;
  double accepted_pressure_norm;
  double accepted_residual_norm;
  double accepted_jacobian_diagonal;
  double conductivity_minimum;
  double conductivity_maximum;
  double epsilon_minimum;
  double epsilon_maximum;
  PetscInt nonlinear_iterations;
  SsNonlinearOutcome nonlinear_outcome;
  SsNonlinearOutcome injected_failure_outcome;
  int injected_failure_rolled_back;
  int injected_failure_parent_unchanged;
  int accepted_transaction_committed;
  int observed_threads;
} HydraulicFixtureProfile;

/** Compact seepage problem of a dimension; 0 on success. */
int hydraulic_fixture_compact_profile(uint32_t dimension, int requested_threads,
                                      HydraulicFixtureProfile *profile);
/** Seepage problem sliced from a catalog case; 0 on success. */
int hydraulic_fixture_case_slice_profile(const char *case_id, int requested_threads,
                                         HydraulicFixtureProfile *profile);

#endif
