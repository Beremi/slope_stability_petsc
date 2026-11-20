/* Immutable coupling snapshots and mechanics body forces.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "coupling_internal.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/core/util.h"
#include "slope_internal/mesh/cell_index.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  SsEntityId point_id;
  SsEntityId cell_id;
  uint32_t dimension;
  uint32_t mode;
  uint32_t saturated;
  uint32_t reserved;
  double pressure;
  double gradient[SS_TRANSFER_MAX_DIMENSION];
  double body_force[SS_TRANSFER_MAX_DIMENSION];
  double cell_average_pressure;
  double cell_epsilon;
  double selected_unit_weight;
} SsCouplingSnapshotDigestRecord;

typedef struct {
  uint32_t dimension;
  uint32_t mode;
  uint32_t digest_scheme;
  uint32_t reserved;
  uint64_t source_generation;
  char source_identity[SS_COUPLING_ID_CAPACITY];
  char transfer_identity[SS_COUPLING_ID_CAPACITY];
} SsCouplingSnapshotDigestHeader;

struct SsCouplingFieldSnapshot {
  SsCouplingSnapshotView view;
  SsCouplingDiagnostics diagnostics;
  SsEntityId *point_ids;
  SsEntityId *cell_ids_by_point;
  uint32_t *quadrature_indices;
  double *pressure;
  double *gradient;
  double *body_force;
  SsEntityId *cell_ids;
  double *cell_average_pressure;
  double *cell_epsilon;
  double *selected_unit_weight;
  uint8_t *saturated;
};

static size_t ss_coupling_find_cell(const SsEntityId *cells, size_t count,
                                    SsEntityId cell) {
  return ss_mesh_cell_index_find(cells, count, sizeof(*cells), 0U, cell);
}

SsStatus ss_coupling_body_force(uint32_t dimension, uint32_t upward_axis,
                                SsCouplingMode mode, const double *pressure_gradient,
                                double selected_unit_weight, double *body_force) {
  uint32_t coordinate;
  if ((dimension != 2U && dimension != 3U) || upward_axis >= dimension ||
      pressure_gradient == NULL || body_force == NULL ||
      (mode != SS_COUPLING_PRESSURE_GRADIENT_AND_WEIGHT &&
       mode != SS_COUPLING_WEIGHT_ONLY) ||
      !ss_is_finite(selected_unit_weight) || selected_unit_weight <= 0.0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (coordinate = 0U; coordinate < dimension; ++coordinate) {
    if (!ss_is_finite(pressure_gradient[coordinate])) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
    body_force[coordinate] = (mode == SS_COUPLING_PRESSURE_GRADIENT_AND_WEIGHT
                                  ? -pressure_gradient[coordinate]
                                  : 0.0) -
                             (coordinate == upward_axis ? selected_unit_weight : 0.0);
    if (!ss_is_finite(body_force[coordinate])) {
      return SS_ERROR_INTEGRITY;
    }
  }
  return SS_OK;
}

static void ss_coupling_snapshot_release(SsCouplingFieldSnapshot *snapshot) {
  if (snapshot == NULL) {
    return;
  }
  free(snapshot->point_ids);
  free(snapshot->cell_ids_by_point);
  free(snapshot->quadrature_indices);
  free(snapshot->pressure);
  free(snapshot->gradient);
  free(snapshot->body_force);
  free(snapshot->cell_ids);
  free(snapshot->cell_average_pressure);
  free(snapshot->cell_epsilon);
  free(snapshot->selected_unit_weight);
  free(snapshot->saturated);
  free(snapshot);
}

static SsStatus ss_coupling_snapshot_allocate(size_t point_count, size_t cell_count,
                                              uint32_t dimension, SsCouplingMode mode,
                                              SsCouplingFieldSnapshot **snapshot) {
  SsCouplingFieldSnapshot *created;
  size_t vector_count = 0U;
  if ((dimension != 2U && dimension != 3U) ||
      ss_coupling_checked_product(point_count, (size_t)dimension, &vector_count) !=
          SS_OK) {
    return SS_ERROR_OVERFLOW;
  }
  created = calloc(1U, sizeof(*created));
  if (created == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  if (point_count != 0U) {
    created->point_ids = calloc(point_count, sizeof(*created->point_ids));
    created->cell_ids_by_point =
        calloc(point_count, sizeof(*created->cell_ids_by_point));
    created->quadrature_indices =
        calloc(point_count, sizeof(*created->quadrature_indices));
    created->pressure = calloc(point_count, sizeof(*created->pressure));
    created->gradient = calloc(vector_count, sizeof(*created->gradient));
    created->body_force = calloc(vector_count, sizeof(*created->body_force));
  }
  if (cell_count != 0U) {
    created->cell_ids = calloc(cell_count, sizeof(*created->cell_ids));
    created->cell_average_pressure =
        calloc(cell_count, sizeof(*created->cell_average_pressure));
    created->cell_epsilon = calloc(cell_count, sizeof(*created->cell_epsilon));
    created->selected_unit_weight =
        calloc(cell_count, sizeof(*created->selected_unit_weight));
    created->saturated = calloc(cell_count, sizeof(*created->saturated));
  }
  if ((point_count != 0U &&
       (created->point_ids == NULL || created->cell_ids_by_point == NULL ||
        created->quadrature_indices == NULL || created->pressure == NULL ||
        created->gradient == NULL || created->body_force == NULL)) ||
      (cell_count != 0U &&
       (created->cell_ids == NULL || created->cell_average_pressure == NULL ||
        created->cell_epsilon == NULL || created->selected_unit_weight == NULL ||
        created->saturated == NULL))) {
    ss_coupling_snapshot_release(created);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  created->view.point_count = point_count;
  created->view.cell_count = cell_count;
  created->view.dimension = dimension;
  created->view.mode = mode;
  created->view.point_ids = created->point_ids;
  created->view.cell_ids_by_point = created->cell_ids_by_point;
  created->view.quadrature_indices = created->quadrature_indices;
  created->view.pressure = created->pressure;
  created->view.gradient = created->gradient;
  created->view.body_force = created->body_force;
  created->view.cell_ids = created->cell_ids;
  created->view.cell_average_pressure = created->cell_average_pressure;
  created->view.cell_epsilon = created->cell_epsilon;
  created->view.selected_unit_weight = created->selected_unit_weight;
  created->view.saturated = created->saturated;
  *snapshot = created;
  return SS_OK;
}

SsStatus ss_coupling_field_snapshot_build(const SsCouplingTransferPlan *plan,
                                          SsCouplingMode mode,
                                          SsCouplingFieldSnapshot **snapshot) {
  SsCouplingFieldSnapshot *created = NULL;
  SsCouplingSnapshotDigestRecord *digest_records = NULL;
  SsTransferSourceState current_state;
  double *cell_weight_sum = NULL;
  size_t index;
  SsStatus local_status = SS_OK;
  SsStatus status;
  uint64_t local_saturated = 0U;
  uint64_t local_unsaturated = 0U;
  uint64_t global_saturated = 0U;
  uint64_t global_unsaturated = 0U;
  double local_pressure_minimum = DBL_MAX;
  double local_pressure_maximum = -DBL_MAX;
  double local_gradient_maximum = 0.0;
  double global_pressure_minimum = DBL_MAX;
  double global_pressure_maximum = -DBL_MAX;
  double global_gradient_maximum = 0.0;
  int observed_threads = 1;
  int global_observed_threads = 1;
  int used_openmp = 0;
  int global_used_openmp = 0;
  int worker_error = 0;
  size_t point_count = 0U;
  size_t cell_count = 0U;
  uint32_t dimension = 0U;
  if (snapshot != NULL) {
    *snapshot = NULL;
  }
  if (plan == NULL || snapshot == NULL ||
      (mode != SS_COUPLING_PRESSURE_GRADIENT_AND_WEIGHT &&
       mode != SS_COUPLING_WEIGHT_ONLY)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if ((plan->dimension != 2U && plan->dimension != 3U) ||
      plan->upward_axis >= plan->dimension ||
      (plan->local_point_count != 0U && plan->points == NULL) ||
      (plan->local_cell_count != 0U && plan->local_cell_ids == NULL)) {
    return SS_ERROR_INTEGRITY;
  }
  point_count = plan->local_point_count;
  cell_count = plan->local_cell_count;
  dimension = plan->dimension;
  status = ss_coupling_read_collective_source_state(plan->communicator, &plan->geometry,
                                                    &current_state);
  if (status != SS_OK) {
    return status;
  }
  local_status = ss_coupling_source_state_equal(&current_state, &plan->source_state)
                     ? SS_OK
                     : SS_ERROR_INVALID_STATE;
  status = ss_status_allreduce_max(plan->communicator, local_status);
  if (status != SS_OK) {
    return status;
  }
  status =
      ss_coupling_snapshot_allocate(point_count, cell_count, dimension, mode, &created);
  status = ss_status_allreduce_max(plan->communicator, status);
  if (status != SS_OK) {
    ss_coupling_snapshot_release(created);
    return status;
  }
  if (cell_count != 0U) {
    cell_weight_sum = calloc(cell_count, sizeof(*cell_weight_sum));
  }
  if (point_count != 0U) {
    digest_records = calloc(point_count, sizeof(*digest_records));
  }
  local_status = (cell_count != 0U && cell_weight_sum == NULL) ||
                         (point_count != 0U && digest_records == NULL)
                     ? SS_ERROR_OUT_OF_MEMORY
                     : SS_OK;
  status = ss_status_allreduce_max(plan->communicator, local_status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (cell_count != 0U) {
    memcpy(created->cell_ids, plan->local_cell_ids,
           cell_count * sizeof(*created->cell_ids));
  }
  for (index = 0U; index < point_count; ++index) {
    const SsTransferTargetPoint *target = &plan->points[index].target;
    created->point_ids[index] = target->stable_point_id;
    created->cell_ids_by_point[index] = target->stable_target_cell_id;
    created->quadrature_indices[index] = target->quadrature_index;
  }
  status = ss_coupling_source_values(plan, created->pressure, created->gradient);
  if (status != SS_OK) {
    goto cleanup;
  }
  status = ss_coupling_read_collective_source_state(plan->communicator, &plan->geometry,
                                                    &current_state);
  if (status != SS_OK ||
      !ss_coupling_source_state_equal(&current_state, &plan->source_state)) {
    status = status == SS_OK ? SS_ERROR_INVALID_STATE : status;
    goto cleanup;
  }
  for (index = 0U; index < point_count; ++index) {
    const SsTransferTargetPoint *target = &plan->points[index].target;
    const size_t cell = ss_coupling_find_cell(plan->local_cell_ids, cell_count,
                                              target->stable_target_cell_id);
    if (cell == cell_count) {
      local_status = SS_ERROR_INTEGRITY;
      break;
    }
    created->cell_average_pressure[cell] +=
        target->physical_weight * created->pressure[index];
    cell_weight_sum[cell] += target->physical_weight;
    created->cell_epsilon[cell] = target->cell_epsilon;
  }
  {
    size_t first_point = 0U;
    for (index = 0U; local_status == SS_OK && index < cell_count; ++index) {
      const SsTransferTargetPoint *target;
      while (first_point < point_count &&
             plan->points[first_point].target.stable_target_cell_id <
                 plan->local_cell_ids[index]) {
        ++first_point;
      }
      if (first_point == point_count ||
          plan->points[first_point].target.stable_target_cell_id !=
              plan->local_cell_ids[index] ||
          !ss_is_finite(cell_weight_sum[index]) || cell_weight_sum[index] <= 0.0) {
        local_status = SS_ERROR_INTEGRITY;
        break;
      }
      target = &plan->points[first_point].target;
      created->cell_average_pressure[index] /= cell_weight_sum[index];
      created->saturated[index] =
          created->cell_average_pressure[index] >= 0.1 * created->cell_epsilon[index]
              ? 1U
              : 0U;
      created->selected_unit_weight[index] = created->saturated[index] != 0U
                                                 ? target->saturated_unit_weight
                                                 : target->unsaturated_unit_weight;
      if (!ss_is_finite(created->selected_unit_weight[index]) ||
          created->selected_unit_weight[index] <= 0.0) {
        local_status = SS_ERROR_INTEGRITY;
      }
    }
  }
  status = ss_status_allreduce_max(plan->communicator, local_status);
  if (status != SS_OK) {
    goto cleanup;
  }
#ifdef _OPENMP
  if (plan->requested_threads > 1 && point_count != 0U &&
      (uint64_t)point_count >= plan->openmp_work_threshold) {
    used_openmp = 1;
#pragma omp parallel num_threads(plan->requested_threads) reduction(max : worker_error)
    {
#pragma omp single
      observed_threads = omp_get_num_threads();
#pragma omp for schedule(static)
      for (index = 0U; index < point_count; ++index) {
        const size_t cell =
            ss_coupling_find_cell(plan->local_cell_ids, cell_count,
                                  plan->points[index].target.stable_target_cell_id);
        if (cell == cell_count ||
            ss_coupling_body_force(dimension, plan->upward_axis, mode,
                                   created->gradient + index * dimension,
                                   created->selected_unit_weight[cell],
                                   created->body_force + index * dimension) != SS_OK) {
          worker_error = 1;
        }
      }
    }
  } else
#endif
  {
    for (index = 0U; index < point_count; ++index) {
      const size_t cell =
          ss_coupling_find_cell(plan->local_cell_ids, cell_count,
                                plan->points[index].target.stable_target_cell_id);
      if (cell == cell_count ||
          ss_coupling_body_force(dimension, plan->upward_axis, mode,
                                 created->gradient + index * dimension,
                                 created->selected_unit_weight[cell],
                                 created->body_force + index * dimension) != SS_OK) {
        worker_error = 1;
      }
    }
  }
  local_status = worker_error == 0 ? SS_OK : SS_ERROR_INTEGRITY;
  status = ss_status_allreduce_max(plan->communicator, local_status);
  if (status != SS_OK) {
    goto cleanup;
  }
  for (index = 0U; index < point_count; ++index) {
    const SsTransferTargetPoint *target = &plan->points[index].target;
    const size_t cell = ss_coupling_find_cell(plan->local_cell_ids, cell_count,
                                              target->stable_target_cell_id);
    SsCouplingSnapshotDigestRecord *record = &digest_records[index];
    double norm_squared = 0.0;
    uint32_t coordinate;
    record->point_id = target->stable_point_id;
    record->cell_id = target->stable_target_cell_id;
    record->dimension = dimension;
    record->mode = (uint32_t)mode;
    record->saturated = created->saturated[cell];
    record->pressure = created->pressure[index];
    record->cell_average_pressure = created->cell_average_pressure[cell];
    record->cell_epsilon = created->cell_epsilon[cell];
    record->selected_unit_weight = created->selected_unit_weight[cell];
    for (coordinate = 0U; coordinate < dimension; ++coordinate) {
      const double gradient = created->gradient[index * dimension + coordinate];
      record->gradient[coordinate] = gradient;
      record->body_force[coordinate] =
          created->body_force[index * dimension + coordinate];
      norm_squared += gradient * gradient;
    }
    local_pressure_minimum = fmin(local_pressure_minimum, created->pressure[index]);
    local_pressure_maximum = fmax(local_pressure_maximum, created->pressure[index]);
    local_gradient_maximum = fmax(local_gradient_maximum, sqrt(norm_squared));
  }
  {
    SsCouplingSnapshotDigestHeader header;
    memset(&header, 0, sizeof(header));
    header.dimension = dimension;
    header.mode = (uint32_t)mode;
    header.digest_scheme = SS_COUPLING_DIGEST_SCHEME;
    header.source_generation = plan->source_state.accepted_generation;
    ss_copy_text(header.source_identity, sizeof(header.source_identity),
                 plan->source_state.model_identity);
    ss_copy_text(header.transfer_identity, sizeof(header.transfer_identity),
                 plan->transfer_identity);
    status = ss_coupling_collective_digest(
        plan->communicator, &header, sizeof(header), digest_records, point_count,
        sizeof(*digest_records), plan->global_point_count,
        created->diagnostics.content_digest);
    if (status != SS_OK) {
      goto cleanup;
    }
  }
  for (index = 0U; index < cell_count; ++index) {
    if (created->saturated[index] != 0U) {
      ++local_saturated;
    } else {
      ++local_unsaturated;
    }
  }
  if (MPI_Allreduce(&local_saturated, &global_saturated, 1, MPI_UINT64_T, MPI_SUM,
                    plan->communicator) != MPI_SUCCESS ||
      MPI_Allreduce(&local_unsaturated, &global_unsaturated, 1, MPI_UINT64_T, MPI_SUM,
                    plan->communicator) != MPI_SUCCESS ||
      MPI_Allreduce(&local_pressure_minimum, &global_pressure_minimum, 1, MPI_DOUBLE,
                    MPI_MIN, plan->communicator) != MPI_SUCCESS ||
      MPI_Allreduce(&local_pressure_maximum, &global_pressure_maximum, 1, MPI_DOUBLE,
                    MPI_MAX, plan->communicator) != MPI_SUCCESS ||
      MPI_Allreduce(&local_gradient_maximum, &global_gradient_maximum, 1, MPI_DOUBLE,
                    MPI_MAX, plan->communicator) != MPI_SUCCESS ||
      MPI_Allreduce(&observed_threads, &global_observed_threads, 1, MPI_INT, MPI_MAX,
                    plan->communicator) != MPI_SUCCESS ||
      MPI_Allreduce(&used_openmp, &global_used_openmp, 1, MPI_INT, MPI_MAX,
                    plan->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  created->view.source_generation = plan->source_state.accepted_generation;
  ss_copy_text(created->view.source_identity, sizeof(created->view.source_identity),
               plan->source_state.model_identity);
  ss_copy_text(created->view.source_mesh_identity,
               sizeof(created->view.source_mesh_identity),
               plan->source_state.mesh_identity);
  ss_copy_text(created->view.source_space_identity,
               sizeof(created->view.source_space_identity),
               plan->source_state.field_space_identity);
  ss_copy_text(created->view.source_model_identity,
               sizeof(created->view.source_model_identity),
               plan->source_state.model_identity);
  ss_copy_text(created->view.pressure_unit, sizeof(created->view.pressure_unit),
               plan->source_state.pressure_unit);
  ss_copy_text(created->view.target_mesh_identity,
               sizeof(created->view.target_mesh_identity), plan->target_mesh_identity);
  ss_copy_text(created->view.target_space_identity,
               sizeof(created->view.target_space_identity),
               plan->target_space_identity);
  ss_copy_text(created->view.target_quadrature_identity,
               sizeof(created->view.target_quadrature_identity),
               plan->target_quadrature_identity);
  ss_copy_text(created->view.transfer_identity, sizeof(created->view.transfer_identity),
               plan->transfer_identity);
  ss_copy_text(created->view.snapshot_identity, sizeof(created->view.snapshot_identity),
               created->diagnostics.content_digest);
  created->diagnostics.local_target_points = (uint64_t)point_count;
  created->diagnostics.global_target_points = plan->global_point_count;
  created->diagnostics.local_target_cells = (uint64_t)cell_count;
  created->diagnostics.global_target_cells = plan->global_cell_count;
  created->diagnostics.source_evaluations = plan->global_point_count;
  created->diagnostics.saturated_cells = global_saturated;
  created->diagnostics.unsaturated_cells = global_unsaturated;
  created->diagnostics.pressure_minimum = global_pressure_minimum;
  created->diagnostics.pressure_maximum = global_pressure_maximum;
  created->diagnostics.gradient_norm_maximum = global_gradient_maximum;
  created->diagnostics.requested_threads = plan->requested_threads;
  created->diagnostics.observed_threads = global_observed_threads;
  created->diagnostics.used_openmp = global_used_openmp;
  {
    size_t point_stride = 0U;
    size_t point_bytes = 0U;
    size_t cell_bytes = 0U;
    size_t snapshot_bytes = sizeof(*created);
    const size_t fixed_point_stride =
        2U * sizeof(SsEntityId) + sizeof(uint32_t) + sizeof(double);
    const size_t cell_stride =
        sizeof(SsEntityId) + 3U * sizeof(double) + sizeof(uint8_t);
    local_status =
        ss_coupling_checked_product(2U * (size_t)dimension, sizeof(double),
                                    &point_stride) != SS_OK ||
                ss_coupling_checked_add(point_stride, fixed_point_stride,
                                        &point_stride) != SS_OK ||
                ss_coupling_checked_product(point_count, point_stride, &point_bytes) !=
                    SS_OK ||
                ss_coupling_checked_product(cell_count, cell_stride, &cell_bytes) !=
                    SS_OK ||
                ss_coupling_checked_add(snapshot_bytes, point_bytes, &snapshot_bytes) !=
                    SS_OK ||
                ss_coupling_checked_add(snapshot_bytes, cell_bytes, &snapshot_bytes) !=
                    SS_OK
            ? SS_ERROR_OVERFLOW
            : SS_OK;
    status = ss_status_allreduce_max(plan->communicator, local_status);
    if (status != SS_OK) {
      goto cleanup;
    }
    created->diagnostics.snapshot_bytes = snapshot_bytes;
  }
  *snapshot = created;
  created = NULL;
  status = SS_OK;

cleanup:
  free(cell_weight_sum);
  free(digest_records);
  ss_coupling_snapshot_release(created);
  return status;
}

SsStatus ss_coupling_field_snapshot_view(const SsCouplingFieldSnapshot *snapshot,
                                         SsCouplingSnapshotView *view) {
  if (snapshot == NULL || view == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *view = snapshot->view;
  return SS_OK;
}

SsStatus ss_coupling_field_snapshot_diagnostics(const SsCouplingFieldSnapshot *snapshot,
                                                SsCouplingDiagnostics *diagnostics) {
  if (snapshot == NULL || diagnostics == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *diagnostics = snapshot->diagnostics;
  return SS_OK;
}

SsStatus ss_coupling_field_snapshot_mechanics_force(
    const void *opaque, uint32_t dimension, SsEntityId stable_cell_id,
    uint32_t quadrature_index, const double *physical, double *body_force) {
  const SsCouplingFieldSnapshot *snapshot = (const SsCouplingFieldSnapshot *)opaque;
  size_t left = 0U;
  size_t right;
  uint32_t coordinate;
  if (snapshot == NULL || physical == NULL || body_force == NULL ||
      dimension != snapshot->view.dimension || stable_cell_id == 0U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (coordinate = 0U; coordinate < dimension; ++coordinate) {
    if (!ss_is_finite(physical[coordinate])) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
  }
  right = snapshot->view.point_count;
  while (left < right) {
    const size_t middle = left + (right - left) / 2U;
    const SsEntityId middle_cell = snapshot->cell_ids_by_point[middle];
    const uint32_t middle_quadrature = snapshot->quadrature_indices[middle];
    if (middle_cell < stable_cell_id ||
        (middle_cell == stable_cell_id && middle_quadrature < quadrature_index)) {
      left = middle + 1U;
    } else {
      right = middle;
    }
  }
  if (left >= snapshot->view.point_count ||
      snapshot->cell_ids_by_point[left] != stable_cell_id ||
      snapshot->quadrature_indices[left] != quadrature_index) {
    return SS_ERROR_NOT_FOUND;
  }
  memcpy(body_force, snapshot->body_force + left * dimension,
         dimension * sizeof(*body_force));
  return SS_OK;
}

SsStatus ss_coupling_field_snapshot_destroy(SsCouplingFieldSnapshot **snapshot) {
  if (snapshot == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  ss_coupling_snapshot_release(*snapshot);
  *snapshot = NULL;
  return SS_OK;
}
