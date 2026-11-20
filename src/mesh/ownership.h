/* Balancing of shared high-order point ownership after distribution.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_MESH_OWNERSHIP_H
#define SS_MESH_OWNERSHIP_H

#include "slope_internal/core/core.h"

#include <petscdm.h>

/** Rebalance ownership of shared P2 points by stable hash; collective. */
SsStatus ss_mesh_rebalance_shared_p2_point_ownership(DM dm);
/** Rebalance ownership of shared high-order points by stable hash; collective. */
SsStatus ss_mesh_rebalance_shared_p4_point_ownership(DM dm);

#endif
