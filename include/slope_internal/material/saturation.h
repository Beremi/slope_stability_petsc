/** @file
 * @brief Self-weight and piecewise-linear prescribed saturation for material
 * loads.
 *
 * The second coordinate is the vertical axis in 2D and 3D. All functions are
 * pure and safe inside OpenMP regions.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_MATERIAL_SATURATION_H
#define SS_MATERIAL_SATURATION_H

#include "slope_internal/core/status.h"

#include <stddef.h>
#include <stdint.h>

/** Maximum number of breakpoints of a prescribed saturation profile. */
#define SS_SOLVER_SATURATION_CAPACITY 32U

/** Water level as a piecewise-linear function of one horizontal coordinate. */
typedef struct {
  uint32_t axis; /**< Horizontal axis of the profile: 0 (x) or, in 3D, 2 (z). */
  size_t count;  /**< Number of breakpoints, 1 to ::SS_SOLVER_SATURATION_CAPACITY. */
  /** Strictly increasing breakpoint coordinates along @ref axis, in m. */
  double coordinates[SS_SOLVER_SATURATION_CAPACITY];
  /** Water level (vertical coordinate) at each breakpoint, in m. */
  double levels[SS_SOLVER_SATURATION_CAPACITY];
} SsSolverPrescribedSaturation;

/** Validate a prescribed saturation profile.
 * @param[in] profile Profile to check.
 * @param[in] dimension Spatial dimension, 2 or 3.
 * @return ::SS_OK, or ::SS_ERROR_INVALID_ARGUMENT for an invalid axis or
 *   count, nonfinite values or coordinates that do not strictly increase. */
SsStatus ss_solver_saturation_validate(const SsSolverPrescribedSaturation *profile,
                                       uint32_t dimension);
/** Unit weight at a point from the interpolated water level.
 *
 * The level is interpolated linearly between breakpoints and held constant
 * beyond the first and last ones. Points at or below the level are saturated.
 *
 * @param[in] profile Valid profile.
 * @param[in] dimension Spatial dimension, 2 or 3.
 * @param[in] physical Point coordinates, @p dimension values in m.
 * @param[in] unsaturated Nonnegative unsaturated unit weight.
 * @param[in] saturated Nonnegative saturated unit weight.
 * @param[out] weight Selected unit weight.
 * @return ::SS_OK, or ::SS_ERROR_INVALID_ARGUMENT for invalid input. */
SsStatus ss_solver_saturation_weight(const SsSolverPrescribedSaturation *profile,
                                     uint32_t dimension, const double *physical,
                                     double unsaturated, double saturated,
                                     double *weight);

/** Body force of self-weight, acting in the negative vertical direction.
 * @param[in] dimension Spatial dimension, 2 or 3.
 * @param[in] unit_weight Nonnegative unit weight.
 * @param[out] body_force Body force; the unused third entry is zero in 2D.
 * @return ::SS_OK, or ::SS_ERROR_INVALID_ARGUMENT for invalid input. */
SsStatus ss_solver_self_weight(uint32_t dimension, double unit_weight,
                               double body_force[3]);

#endif
