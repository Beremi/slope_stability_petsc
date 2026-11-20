/* Self-weight and piecewise-linear prescribed saturation for material loads.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/material/saturation.h"

#include <math.h>

SsStatus ss_solver_self_weight(uint32_t dimension, double unit_weight,
                               double body_force[3]) {
  if ((dimension != 2U && dimension != 3U) || body_force == NULL ||
      !isfinite(unit_weight) || unit_weight < 0.0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  /*
   * Every supported 2D/3D asset uses the second coordinate as the vertical
   * axis.  This must agree with the hydraulic and coupling models.
   */
  body_force[0] = 0.0;
  body_force[1] = -unit_weight;
  body_force[2] = 0.0;
  return SS_OK;
}

SsStatus ss_solver_saturation_validate(const SsSolverPrescribedSaturation *profile,
                                       uint32_t dimension) {
  size_t index;
  if (profile == NULL || (dimension != 2U && dimension != 3U) ||
      (profile->axis != 0U && profile->axis != 2U) || profile->axis >= dimension ||
      profile->count == 0U || profile->count > SS_SOLVER_SATURATION_CAPACITY) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (index = 0U; index < profile->count; ++index) {
    if (!isfinite(profile->coordinates[index]) || !isfinite(profile->levels[index]) ||
        (index > 0U &&
         (!(profile->coordinates[index] > profile->coordinates[index - 1U]) ||
          !isfinite(profile->coordinates[index] - profile->coordinates[index - 1U])))) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
  }
  return SS_OK;
}

SsStatus ss_solver_saturation_weight(const SsSolverPrescribedSaturation *profile,
                                     uint32_t dimension, const double *physical,
                                     double unsaturated, double saturated,
                                     double *weight) {
  size_t index;
  double coordinate, level;
  if (ss_solver_saturation_validate(profile, dimension) != SS_OK || physical == NULL ||
      weight == NULL || !isfinite(unsaturated) || unsaturated < 0.0 ||
      !isfinite(saturated) || saturated < 0.0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (index = 0U; index < dimension; ++index) {
    if (!isfinite(physical[index])) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
  }
  coordinate = physical[profile->axis];
  level = profile->levels[0];
  if (coordinate >= profile->coordinates[profile->count - 1U]) {
    level = profile->levels[profile->count - 1U];
  } else if (coordinate > profile->coordinates[0]) {
    for (index = 1U; index < profile->count; ++index) {
      if (coordinate <= profile->coordinates[index]) {
        double fraction =
            (coordinate - profile->coordinates[index - 1U]) /
            (profile->coordinates[index] - profile->coordinates[index - 1U]);
        level = (1.0 - fraction) * profile->levels[index - 1U] +
                fraction * profile->levels[index];
        break;
      }
    }
  }
  *weight = physical[1] <= level ? saturated : unsaturated;
  return SS_OK;
}
