/* Direct and batched remote evaluation of pressure and its gradient.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "coupling_internal.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/core/util.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  size_t owner_index;
  SsEntityId source_cell_id;
  double reference[SS_TRANSFER_MAX_DIMENSION];
} SsCouplingSourceRequest;

typedef struct {
  SsStatus status;
  double pressure;
  double gradient[SS_TRANSFER_MAX_DIMENSION];
} SsCouplingSourceResponse;

static SsStatus ss_coupling_evaluate(const SsCouplingTransferPlan *plan,
                                     SsEntityId cell,
                                     const double reference[SS_TRANSFER_MAX_DIMENSION],
                                     double *pressure,
                                     double gradient[SS_TRANSFER_MAX_DIMENSION]) {
  const SsStatus status = plan->geometry.evaluate_source(plan->geometry.context, cell,
                                                         reference, pressure, gradient);
  if (status != SS_OK) {
    return status;
  }
  if (!ss_is_finite(*pressure)) {
    return SS_ERROR_INTEGRITY;
  }
  for (uint32_t d = 0; d < plan->dimension; ++d) {
    if (!ss_is_finite(gradient[d])) {
      return SS_ERROR_INTEGRITY;
    }
  }
  return SS_OK;
}

/* All requests are grouped by source owner. The response reverses the
 * exchange using the same counts and offsets, so it preserves the request
 * slots without any added point IDs or messages. */
static SsStatus ss_coupling_exchange_remote(const SsCouplingTransferPlan *plan,
                                            double *pressure, double *gradient) {
  SsCouplingSourceRequest *outgoing = NULL;
  SsCouplingSourceRequest *incoming = NULL;
  SsCouplingSourceResponse *responses = NULL;
  SsCouplingSourceResponse *received = NULL;
  MPI_Datatype request_type = MPI_DATATYPE_NULL;
  MPI_Datatype response_type = MPI_DATATYPE_NULL;
  int *send_counts = calloc((size_t)plan->size, sizeof(*send_counts));
  int *receive_counts = calloc((size_t)plan->size, sizeof(*receive_counts));
  int *send_offsets = calloc((size_t)plan->size, sizeof(*send_offsets));
  int *receive_offsets = calloc((size_t)plan->size, sizeof(*receive_offsets));
  int *cursors = calloc((size_t)plan->size, sizeof(*cursors));
  SsStatus status = ss_status_allreduce_max(
      plan->communicator,
      send_counts && receive_counts && send_offsets && receive_offsets && cursors
          ? SS_OK
          : SS_ERROR_OUT_OF_MEMORY);
  if (status != SS_OK) {
    goto cleanup;
  }
  for (size_t i = 0; i < plan->local_point_count; ++i) {
    if (plan->points[i].source_rank != plan->rank) {
      ++send_counts[plan->points[i].source_rank];
    }
  }
  if (MPI_Alltoall(send_counts, 1, MPI_INT, receive_counts, 1, MPI_INT,
                   plan->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  int send_total = 0, receive_total = 0;
  for (int rank = 0; rank < plan->size; ++rank) {
    if (receive_counts[rank] > INT_MAX - receive_total) {
      status = SS_ERROR_RESOURCE;
      break;
    }
    cursors[rank] = send_offsets[rank] = send_total;
    receive_offsets[rank] = receive_total;
    send_total += send_counts[rank];
    receive_total += receive_counts[rank];
  }
  if (status == SS_OK) {
    outgoing = calloc(send_total > 0 ? (size_t)send_total : 1U, sizeof(*outgoing));
    incoming =
        calloc(receive_total > 0 ? (size_t)receive_total : 1U, sizeof(*incoming));
    responses =
        calloc(receive_total > 0 ? (size_t)receive_total : 1U, sizeof(*responses));
    received = calloc(send_total > 0 ? (size_t)send_total : 1U, sizeof(*received));
    if (!outgoing || !incoming || !responses || !received) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  if (status == SS_OK && (MPI_Type_contiguous((int)sizeof(*outgoing), MPI_BYTE,
                                              &request_type) != MPI_SUCCESS ||
                          MPI_Type_commit(&request_type) != MPI_SUCCESS ||
                          MPI_Type_contiguous((int)sizeof(*responses), MPI_BYTE,
                                              &response_type) != MPI_SUCCESS ||
                          MPI_Type_commit(&response_type) != MPI_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(plan->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  for (size_t i = 0; i < plan->local_point_count; ++i) {
    const SsCouplingPlanPoint *point = &plan->points[i];
    if (point->source_rank == plan->rank) {
      continue;
    }
    SsCouplingSourceRequest *request = &outgoing[cursors[point->source_rank]++];
    request->owner_index = i;
    request->source_cell_id = point->source.stable_source_cell_id;
    memcpy(request->reference, point->source.reference, sizeof(request->reference));
  }
  if (MPI_Alltoallv(outgoing, send_counts, send_offsets, request_type, incoming,
                    receive_counts, receive_offsets, request_type,
                    plan->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  for (int i = 0; i < receive_total; ++i) {
    responses[i].status =
        ss_coupling_evaluate(plan, incoming[i].source_cell_id, incoming[i].reference,
                             &responses[i].pressure, responses[i].gradient);
  }
  if (MPI_Alltoallv(responses, receive_counts, receive_offsets, response_type, received,
                    send_counts, send_offsets, response_type,
                    plan->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  for (int i = 0; i < send_total; ++i) {
    const size_t target = outgoing[i].owner_index;
    if (received[i].status != SS_OK) {
      status = received[i].status;
      break;
    }
    pressure[target] = received[i].pressure;
    memcpy(gradient + target * plan->dimension, received[i].gradient,
           plan->dimension * sizeof(*gradient));
  }
  status = ss_status_allreduce_max(plan->communicator, status);

cleanup:
  if (request_type != MPI_DATATYPE_NULL) {
    (void)MPI_Type_free(&request_type);
  }
  if (response_type != MPI_DATATYPE_NULL) {
    (void)MPI_Type_free(&response_type);
  }
  free(send_counts);
  free(receive_counts);
  free(send_offsets);
  free(receive_offsets);
  free(cursors);
  free(outgoing);
  free(incoming);
  free(responses);
  free(received);
  return status;
}

SsStatus ss_coupling_source_values(const SsCouplingTransferPlan *plan, double *pressure,
                                   double *gradient) {
  SsStatus status = SS_OK;
  for (size_t i = 0; i < plan->local_point_count && status == SS_OK; ++i) {
    const SsCouplingPlanPoint *point = &plan->points[i];
    if (point->source_rank == plan->rank) {
      double local_gradient[SS_TRANSFER_MAX_DIMENSION] = {0.0};
      status =
          ss_coupling_evaluate(plan, point->source.stable_source_cell_id,
                               point->source.reference, &pressure[i], local_gradient);
      memcpy(gradient + i * plan->dimension, local_gradient,
             plan->dimension * sizeof(*gradient));
    }
  }
  status = ss_status_allreduce_max(plan->communicator, status);
  return status == SS_OK && plan->has_remote_points
             ? ss_coupling_exchange_remote(plan, pressure, gradient)
             : status;
}
