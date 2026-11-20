/* Compact mechanics problems (direct and PMG solver profiles, region body
 * force, coupled SSR) shared by the nonlinear and hydraulic tests.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_TEST_MECHANICS_FIXTURE_H
#define SS_TEST_MECHANICS_FIXTURE_H

#include "slope_internal/assembly/assembly.h"

#include <stdint.h>

/** Solve the compact mechanics problem with each solver profile; 0 on success. */
int mechanics_fixture_run_profiles(void);
/** Check region body forces on the compact mechanics problem; 0 on success. */
int mechanics_fixture_run_region_body_force(void);
/** Run a coupled SSR profile with a coupling point kernel; 0 on success. */
int mechanics_fixture_coupled_ssr_profile(
    uint32_t dimension, SsMechanicsCouplingPointKernel coupling_point_kernel,
    const void *coupling_point_context, int requested_threads,
    double *trajectory_checksum);

#endif
