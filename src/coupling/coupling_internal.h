/* Private data and helpers shared by coupling plans and snapshots.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_COUPLING_INTERNAL_H
#define SS_COUPLING_INTERNAL_H

#include "slope_internal/coupling/coupling.h"

#define SS_COUPLING_DIGEST_SCHEME 1U

typedef struct {
  SsTransferTargetPoint target;
  SsTransferSourceCandidate source;
  int source_rank;
} SsCouplingPlanPoint;

struct SsCouplingTransferPlan {
  MPI_Comm communicator;
  int rank;
  int size;
  int has_remote_points;
  uint32_t dimension;
  uint32_t upward_axis;
  SsCouplingTransferType transfer_type;
  int requested_threads;
  uint64_t openmp_work_threshold;
  SsTransferGeometryOps geometry;
  SsTransferSourceState source_state;
  SsCouplingPlanPoint *points;
  size_t local_point_count;
  uint64_t global_point_count;
  SsEntityId *local_cell_ids;
  size_t local_cell_count;
  uint64_t global_cell_count;
  char target_mesh_identity[SS_COUPLING_ID_CAPACITY];
  char target_space_identity[SS_COUPLING_ID_CAPACITY];
  char target_quadrature_identity[SS_COUPLING_ID_CAPACITY];
  char transfer_identity[SS_COUPLING_ID_CAPACITY];
};

/** Compute a checked byte count. */
SsStatus ss_coupling_checked_product(size_t left, size_t right, size_t *product);
/** Compute a checked allocation size. */
SsStatus ss_coupling_checked_add(size_t left, size_t right, size_t *sum);
/** Compare source identity and generation across snapshot phases. */
int ss_coupling_source_state_equal(const SsTransferSourceState *left,
                                   const SsTransferSourceState *right);
/** Read and agree source metadata on the plan communicator. */
SsStatus ss_coupling_read_collective_source_state(MPI_Comm communicator,
                                                  const SsTransferGeometryOps *geometry,
                                                  SsTransferSourceState *state);
/** Hash a distributed set of records in a layout-independent projection. */
SsStatus ss_coupling_collective_digest(MPI_Comm communicator, const void *header,
                                       size_t header_size, const void *local_records,
                                       size_t local_record_count, size_t record_size,
                                       uint64_t global_record_count,
                                       char output[SS_COUPLING_ID_CAPACITY]);
/** Locate target points using the known same-mesh routing or geometric search. */
SsStatus ss_coupling_locate_points(SsCouplingTransferPlan *plan,
                                   SsCouplingSameMeshMap map);
/** Evaluate local pressure points and exchange remote requests in one batch. */
SsStatus ss_coupling_source_values(const SsCouplingTransferPlan *plan, double *pressure,
                                   double *gradient);

#endif
