/* Pack, evaluate and publish assembled residuals, tangents and potentials.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "operator_internal.h"

#include "slope_internal/build/config.h"

#include <petsclog.h>
#include <petscvec.h>

#if SLOPE_ENABLE_OPENMP
#include <omp.h>
#endif

#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>

static int ss_operator_in_parallel(void) {
#if SLOPE_ENABLE_OPENMP
  return omp_in_parallel() != 0 ? 1 : 0;
#else
  return 0;
#endif
}

int ss_operator_thread(void) {
#if SLOPE_ENABLE_OPENMP
  return omp_in_parallel() != 0 ? omp_get_thread_num() : 0;
#else
  return 0;
#endif
}

static int ss_operator_cpu(void) {
#if defined(__linux__)
  return sched_getcpu();
#else
  return -1;
#endif
}

SsStatus ss_operator_service(SsOperator *operator_handle, SsAssemblyService service) {
  const int in_parallel = ss_operator_in_parallel();
  if (operator_handle == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (in_parallel != 0) {
    ++operator_handle->diagnostics.worker_service_calls;
  }
  if (operator_handle->service_observer != NULL) {
    operator_handle->service_observer(service, in_parallel, ss_operator_thread(),
                                      operator_handle->service_context);
  }
  return in_parallel == 0 ? SS_OK : SS_ERROR_THREAD;
}

static SsStatus ss_operator_validate_destinations(SsOperator *operator_handle,
                                                  uint32_t request, Vec residual,
                                                  Mat jacobian) {
  PetscInt local = 0;
  PetscInt global = 0;
  PetscInt local_rows = 0;
  PetscInt local_columns = 0;
  PetscInt global_rows = 0;
  PetscInt global_columns = 0;
  if ((request & SS_ASSEMBLY_RESIDUAL) != 0U) {
    if (residual == NULL ||
        ss_operator_service(operator_handle, SS_ASSEMBLY_SERVICE_PETSC) != SS_OK ||
        VecGetLocalSize(residual, &local) != PETSC_SUCCESS ||
        VecGetSize(residual, &global) != PETSC_SUCCESS ||
        local != (PetscInt)operator_handle->plan->summary.local_owned_rows ||
        global != (PetscInt)operator_handle->plan->summary.global_rows) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
  }
  if ((request & SS_ASSEMBLY_JACOBIAN) != 0U) {
    if (jacobian == NULL ||
        ss_operator_service(operator_handle, SS_ASSEMBLY_SERVICE_PETSC) != SS_OK ||
        MatGetLocalSize(jacobian, &local_rows, &local_columns) != PETSC_SUCCESS ||
        MatGetSize(jacobian, &global_rows, &global_columns) != PETSC_SUCCESS ||
        local_rows != (PetscInt)operator_handle->plan->summary.local_owned_rows ||
        local_columns != (PetscInt)operator_handle->plan->summary.local_owned_rows ||
        global_rows != (PetscInt)operator_handle->plan->summary.global_rows ||
        global_columns != (PetscInt)operator_handle->plan->summary.global_rows) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
  }
  return SS_OK;
}

SsStatus ss_operator_gather(SsOperator *operator_handle, const SsOperatorState *state,
                            const PetscScalar **gathered) {
  PetscInt local_size = 0;
  PetscInt global_size = 0;
  SsStatus status = SS_OK;
  if (state == NULL || state->field == NULL || !isfinite(state->reduction_factor) ||
      state->reduction_factor <= 0.0 || !isfinite(state->external_load_factor) ||
      ss_operator_service(operator_handle, SS_ASSEMBLY_SERVICE_PETSC) != SS_OK ||
      VecGetLocalSize(state->field, &local_size) != PETSC_SUCCESS ||
      VecGetSize(state->field, &global_size) != PETSC_SUCCESS ||
      local_size != (PetscInt)operator_handle->plan->summary.local_owned_rows ||
      global_size != (PetscInt)operator_handle->plan->summary.global_rows) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                 status);
  if (status != SS_OK) {
    return status;
  }
  const int logging =
      PetscLogEventBegin(ss_operator_scatter_event, 0, 0, 0, 0) == PETSC_SUCCESS;
  const PetscErrorCode begin =
      VecScatterBegin(operator_handle->scatter, state->field,
                      operator_handle->gathered_values, INSERT_VALUES, SCATTER_FORWARD);
  const PetscErrorCode end = begin == PETSC_SUCCESS
                                 ? VecScatterEnd(operator_handle->scatter, state->field,
                                                 operator_handle->gathered_values,
                                                 INSERT_VALUES, SCATTER_FORWARD)
                                 : begin;
  if (logging) {
    if (PetscLogEventEnd(ss_operator_scatter_event, 0, 0, 0, 0) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  return logging && end == PETSC_SUCCESS && status == SS_OK &&
                 VecGetArrayRead(operator_handle->gathered_values, gathered) ==
                     PETSC_SUCCESS
             ? SS_OK
             : SS_ERROR_INTERNAL;
}

void ss_operator_pack_cells(SsOperator *operator_handle, const PetscScalar *gathered) {
  size_t cell_index;
#if SLOPE_ENABLE_OPENMP
#pragma omp for schedule(static)
#endif
  for (cell_index = 0U; cell_index < operator_handle->plan->cell_count; ++cell_index) {
    const SsCellPlan *cell = &operator_handle->plan->cells[cell_index];
    size_t slot;
    if (operator_handle->pack_required[cell_index] == 0U) {
      continue;
    }
    for (slot = 0U; slot < cell->closure_count; ++slot) {
      const size_t gather_slot =
          operator_handle
              ->packed_slots[cell_index * operator_handle->maximum_closure + slot];
      operator_handle
          ->packed_values[cell_index * operator_handle->maximum_closure + slot] =
          gather_slot == SIZE_MAX ? cell->prescribed_values[slot]
                                  : (double)gathered[gather_slot];
    }
  }
}

int ss_operator_use_openmp(const SsOperator *operator_handle) {
#if SLOPE_ENABLE_OPENMP
  uint64_t work = (uint64_t)operator_handle->plan->cell_count *
                  (uint64_t)operator_handle->maximum_closure *
                  (uint64_t)operator_handle->maximum_closure;
  return operator_handle->spec.requested_threads > 1 &&
                 work >= operator_handle->spec.openmp_work_threshold
             ? 1
             : 0;
#else
  (void)operator_handle;
  return 0;
#endif
}

static int ss_operator_failure_matches(const SsOperator *operator_handle,
                                       SsEntityId entity) {
  return (operator_handle->failure_rank < 0 ||
          operator_handle->failure_rank == operator_handle->plan->rank) &&
         (operator_handle->failure_entity == 0U ||
          operator_handle->failure_entity == entity);
}

static void ss_operator_evaluate_cell(SsOperator *operator_handle,
                                      const SsOperatorState *state, uint32_t request,
                                      size_t cell_index, size_t tile_slot) {
  operator_handle->element_errors[tile_slot] = (SsErrorRecord){0};
  const SsCellPlan *cell = &operator_handle->plan->cells[cell_index];
  const SsOperatorCell *metadata = &operator_handle->cells[cell_index];
  SsElementOutput output = {
      operator_handle->maximum_closure,
      operator_handle->matrix_stride,
      operator_handle->element_residual + tile_slot * operator_handle->maximum_closure,
      operator_handle->element_jacobian + tile_slot * operator_handle->matrix_stride,
      0.0,
  };
  SsStatus status;
  if (operator_handle->spec.determinism == SS_ASSEMBLY_PERFORMANCE &&
      cell->cell_owned == 0U) {
    operator_handle->element_potential[tile_slot] = 0.0;
    return;
  }
  if (operator_handle->failure_point == SS_ASSEMBLY_FAIL_WORKER &&
      ss_operator_failure_matches(operator_handle, cell->stable_cell_id) != 0) {
    status = SS_ERROR_INJECTED;
    ss_assembly_error_private(&operator_handle->element_errors[tile_slot], status,
                              "ASM_INJECTED_WORKER", cell->stable_cell_id,
                              operator_handle->plan->rank);
  } else if (operator_handle->spec.physics == SS_ASSEMBLY_MECHANICS) {
    if (operator_handle->spec.matrix_free != 0 && cell->dimension == 3U &&
        metadata->quadrature->point_count <=
            SS_OPERATOR_REFERENCE_ACTION_MAX_QUADRATURE &&
        operator_handle->mechanics.coupling_point_kernel == NULL &&
        (request & SS_ASSEMBLY_JACOBIAN) == 0U) {
      status = ss_operator_matrix_free_reference_residual_cell_3d(
          operator_handle, cell_index, state, request, &output,
          &operator_handle->element_errors[tile_slot]);
    } else {
      const SsMechanicsElementInput input = {
          cell->dimension,
          metadata->field_descriptor,
          metadata->coordinate_descriptor,
          metadata->quadrature,
          cell->coordinate_coordinates,
          operator_handle->packed_values +
              cell_index * operator_handle->maximum_closure,
          metadata->body_force,
          operator_handle->mechanics.fixed_coupling_force,
          metadata->material,
          state->reduction_factor,
          state->external_load_factor,
          cell->stable_cell_id,
          operator_handle->mechanics.coupling_point_kernel,
          operator_handle->mechanics.coupling_point_context,
          operator_handle->mechanics.point_force_mode,
      };
      status = ss_assembly_mechanics_element_prepared(
          &input, &metadata->prepared, request, &output,
          &operator_handle->element_errors[tile_slot]);
    }
  } else {
    const SsScalarElementInput input = {
        cell->dimension,
        metadata->field_descriptor,
        metadata->coordinate_descriptor,
        metadata->quadrature,
        cell->coordinate_coordinates,
        operator_handle->packed_values + cell_index * operator_handle->maximum_closure,
        operator_handle->scalar.point_kernel,
        operator_handle->scalar.point_context,
        cell->stable_cell_id,
    };
    status = ss_assembly_scalar_element(&input, request, &output,
                                        &operator_handle->element_errors[tile_slot]);
  }
  if (status != SS_OK && operator_handle->element_errors[tile_slot].valid == 0) {
    ss_assembly_error_private(&operator_handle->element_errors[tile_slot], status,
                              "ASM_WORKER", cell->stable_cell_id,
                              operator_handle->plan->rank);
  }
  if (operator_handle->element_errors[tile_slot].valid != 0) {
    operator_handle->element_errors[tile_slot].key.rank =
        (uint32_t)operator_handle->plan->rank;
    operator_handle->element_errors[tile_slot].key.thread =
        (uint32_t)ss_operator_thread();
  }
  operator_handle->element_potential[tile_slot] = output.potential;
}

static void ss_operator_form_rows(SsOperator *operator_handle, uint32_t request,
                                  size_t tile_start, size_t tile_end) {
  size_t local_row;
#if SLOPE_ENABLE_OPENMP
#pragma omp for schedule(static)
#endif
  for (local_row = 0U; local_row < operator_handle->plan->row_count; ++local_row) {
    const SsOwnedRowPlan *row = &operator_handle->plan->rows[local_row];
    const int writer = ss_operator_thread();
    size_t contribution;
    if (tile_start == 0U) {
      operator_handle->row_writer[local_row] = writer;
    } else if (operator_handle->row_writer[local_row] != writer) {
      operator_handle->row_writer_violation[local_row] = 1U;
    }
    for (contribution = 0U; contribution < row->cell_count; ++contribution) {
      const size_t cell_index = row->cell_slots[contribution].cell_index;
      const SsCellPlan *cell;
      size_t tile_slot;
      size_t row_slot;
      size_t column_slot;
      if (cell_index < tile_start || cell_index >= tile_end) {
        continue;
      }
      tile_slot = cell_index - tile_start;
      if (operator_handle->element_errors[tile_slot].valid != 0) {
        continue;
      }
      cell = &operator_handle->plan->cells[cell_index];
      row_slot = row->cell_slots[contribution].row_slot;
      if ((request & SS_ASSEMBLY_RESIDUAL) != 0U) {
        operator_handle->shadow_residual[local_row] +=
            operator_handle
                ->element_residual[tile_slot * operator_handle->maximum_closure +
                                   row_slot];
      }
      if ((request & SS_ASSEMBLY_JACOBIAN) != 0U) {
        for (column_slot = 0U; column_slot < cell->closure_count; ++column_slot) {
          size_t shadow_column;
          if (cell->constrained[column_slot] != 0U) {
            continue;
          }
          shadow_column =
              ss_operator_find_row_column(row, cell->global_indices[column_slot]);
          if (shadow_column == SIZE_MAX) {
            operator_handle->row_writer_violation[local_row] = 1U;
            continue;
          }
          operator_handle->shadow_jacobian[operator_handle->row_offsets[local_row] +
                                           shadow_column] +=
              operator_handle
                  ->element_jacobian[tile_slot * operator_handle->matrix_stride +
                                     row_slot * cell->closure_count + column_slot];
        }
      }
    }
    if (tile_end == operator_handle->plan->cell_count) {
      operator_handle->row_complete[local_row] = 1U;
    }
  }
}

static void ss_operator_evaluate_facet(SsOperator *operator_handle,
                                       const SsOperatorState *state, uint32_t request,
                                       size_t facet_index, size_t tile_slot) {
  operator_handle->element_errors[tile_slot] = (SsErrorRecord){0};
  const SsOperatorNaturalFacet *facet = &operator_handle->natural_facets[facet_index];
  const SsNaturalOperatorSpec *operation =
      &operator_handle->natural_operations[facet->operation_index];
  const SsCellPlan *parent = &operator_handle->plan->cells[facet->parent_cell_index];
  double field_values[SS_FIELD_MAX_FACET_CLOSURE] = {0.0};
  double signed_flux = 0.0;
  SsElementOutput output = {
      operator_handle->maximum_closure,
      operator_handle->matrix_stride,
      operator_handle->element_residual + tile_slot * operator_handle->maximum_closure,
      operator_handle->element_jacobian + tile_slot * operator_handle->matrix_stride,
      0.0,
  };
  SsNaturalElementInput input = {
      &facet->pack, facet->field_descriptor, facet->quadrature,
      field_values, operation->point_kernel, operation->point_context,
  };
  const double scale =
      operation->scaled_by_external_load != 0 ? state->external_load_factor : 1.0;
  SsStatus status = SS_OK;
  size_t slot;
  operator_handle->element_natural_work[tile_slot] = 0.0;
  if (operator_handle->spec.determinism == SS_ASSEMBLY_PERFORMANCE &&
      facet->pack.geometry.facet_owned == 0U) {
    operator_handle->element_potential[tile_slot] = 0.0;
    return;
  }
  for (slot = 0U; slot < facet->pack.closure_count; ++slot) {
    const size_t parent_slot = facet->pack.parent_closure_slots[slot];
    if (parent_slot >= parent->closure_count) {
      status = SS_ERROR_INTEGRITY;
      ss_assembly_error_private(&operator_handle->element_errors[tile_slot], status,
                                "ASM_NATURAL_PARENT_SLOT",
                                facet->pack.geometry.stable_facet_id,
                                operator_handle->plan->rank);
      break;
    }
    field_values[slot] =
        operator_handle->packed_values[facet->parent_cell_index *
                                           operator_handle->maximum_closure +
                                       parent_slot];
  }
  if (status == SS_OK) {
    status = ss_assembly_natural_element(&input, request, &output, &signed_flux,
                                         &operator_handle->element_errors[tile_slot]);
  }
  if (status == SS_OK && scale != 1.0) {
    if ((request & SS_ASSEMBLY_RESIDUAL) != 0U) {
      for (slot = 0U; slot < facet->pack.closure_count; ++slot) {
        output.residual[slot] *= scale;
      }
    }
    if ((request & SS_ASSEMBLY_JACOBIAN) != 0U) {
      const size_t entries = facet->pack.closure_count * facet->pack.closure_count;
      for (slot = 0U; slot < entries; ++slot) {
        output.jacobian[slot] *= scale;
      }
    }
  }
  if (status != SS_OK && operator_handle->element_errors[tile_slot].valid == 0) {
    ss_assembly_error_private(
        &operator_handle->element_errors[tile_slot], status, "ASM_NATURAL_WORKER",
        facet->pack.geometry.stable_facet_id, operator_handle->plan->rank);
  }
  if (operator_handle->element_errors[tile_slot].valid != 0) {
    operator_handle->element_errors[tile_slot].key.rank =
        (uint32_t)operator_handle->plan->rank;
    operator_handle->element_errors[tile_slot].key.thread =
        (uint32_t)ss_operator_thread();
  }
  operator_handle->element_potential[tile_slot] =
      status == SS_OK ? signed_flux * scale : 0.0;
  operator_handle->element_natural_work[tile_slot] =
      status == SS_OK ? output.potential * scale : 0.0;
}

static void ss_operator_form_natural_rows(SsOperator *operator_handle, uint32_t request,
                                          size_t tile_start, size_t tile_count) {
  size_t local_row;
#if SLOPE_ENABLE_OPENMP
#pragma omp for schedule(static)
#endif
  for (local_row = 0U; local_row < operator_handle->plan->row_count; ++local_row) {
    const SsOwnedRowPlan *row = &operator_handle->plan->rows[local_row];
    const int writer = ss_operator_thread();
    size_t tile_slot;
    if (operator_handle->row_writer[local_row] != writer) {
      operator_handle->row_writer_violation[local_row] = 1U;
    }
    for (tile_slot = 0U; tile_slot < tile_count; ++tile_slot) {
      const SsOperatorNaturalFacet *facet =
          &operator_handle->natural_facets[tile_start + tile_slot];
      size_t facet_row;
      if (operator_handle->element_errors[tile_slot].valid != 0) {
        continue;
      }
      for (facet_row = 0U; facet_row < facet->pack.closure_count; ++facet_row) {
        size_t facet_column;
        if (facet->pack.constrained[facet_row] != 0U ||
            facet->pack.global_indices[facet_row] != row->global_row) {
          continue;
        }
        if ((request & SS_ASSEMBLY_RESIDUAL) != 0U) {
          operator_handle->shadow_residual[local_row] +=
              operator_handle
                  ->element_residual[tile_slot * operator_handle->maximum_closure +
                                     facet_row];
        }
        if ((request & SS_ASSEMBLY_JACOBIAN) == 0U) {
          continue;
        }
        for (facet_column = 0U; facet_column < facet->pack.closure_count;
             ++facet_column) {
          size_t shadow_column;
          if (facet->pack.constrained[facet_column] != 0U) {
            continue;
          }
          shadow_column = ss_operator_find_row_column(
              row, facet->pack.global_indices[facet_column]);
          if (shadow_column == SIZE_MAX) {
            operator_handle->row_writer_violation[local_row] = 1U;
            continue;
          }
          operator_handle->shadow_jacobian[operator_handle->row_offsets[local_row] +
                                           shadow_column] +=
              operator_handle
                  ->element_jacobian[tile_slot * operator_handle->matrix_stride +
                                     facet_row * facet->pack.closure_count +
                                     facet_column];
        }
      }
    }
  }
}

static int ss_operator_topology_value(int cpu, const char *name, int *value) {
  char path[128];
  FILE *stream;
  int parsed;
  if (cpu < 0 || name == NULL || value == NULL ||
      snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/%s", cpu,
               name) < 0) {
    return 0;
  }
  stream = fopen(path, "r");
  if (stream == NULL) {
    return 0;
  }
  parsed = fscanf(stream, "%d", value);
  (void)fclose(stream);
  return parsed == 1 ? 1 : 0;
}

static void ss_operator_finish_affinity(SsOperator *operator_handle,
                                        int observed_threads) {
  size_t thread;
  size_t cursor = 0U;
  uint64_t unique_cpus = 0U;
  uint64_t unique_cores = 0U;
  if (operator_handle == NULL || observed_threads <= 0) {
    return;
  }
  /* Pinned workers report the same CPUs on every call; reuse the topology
   * summary instead of rereading sysfs O(threads^2) times per assembly. */
  if (operator_handle->affinity_cpus != NULL &&
      operator_handle->affinity_count == observed_threads &&
      memcmp(operator_handle->affinity_cpus, operator_handle->worker_cpus,
             (size_t)observed_threads * sizeof(*operator_handle->worker_cpus)) == 0) {
    return;
  }
  operator_handle->diagnostics.worker_cpus[0] = '\0';
  for (thread = 0U; thread < (size_t)observed_threads; ++thread) {
    const int cpu = operator_handle->worker_cpus[thread];
    size_t prior;
    int cpu_seen = 0;
    int package = 0;
    int core = 0;
    int core_seen = 0;
    int topology_known =
        ss_operator_topology_value(cpu, "physical_package_id", &package) != 0 &&
        ss_operator_topology_value(cpu, "core_id", &core) != 0;
    for (prior = 0U; prior < thread; ++prior) {
      if (operator_handle->worker_cpus[prior] == cpu) {
        cpu_seen = 1;
        break;
      }
    }
    if (cpu_seen == 0 && cpu >= 0) {
      int written;
      ++unique_cpus;
      written = snprintf(operator_handle->diagnostics.worker_cpus + cursor,
                         sizeof(operator_handle->diagnostics.worker_cpus) - cursor,
                         cursor == 0U ? "%d" : ",%d", cpu);
      if (written > 0 &&
          (size_t)written < sizeof(operator_handle->diagnostics.worker_cpus) - cursor) {
        cursor += (size_t)written;
      }
    }
    if (topology_known == 0) {
      continue;
    }
    for (prior = 0U; prior < thread; ++prior) {
      int prior_package = 0;
      int prior_core = 0;
      const int prior_cpu = operator_handle->worker_cpus[prior];
      if (ss_operator_topology_value(prior_cpu, "physical_package_id",
                                     &prior_package) != 0 &&
          ss_operator_topology_value(prior_cpu, "core_id", &prior_core) != 0 &&
          prior_package == package && prior_core == core) {
        core_seen = 1;
        break;
      }
    }
    if (core_seen == 0) {
      ++unique_cores;
    }
  }
  operator_handle->diagnostics.unique_worker_cpus = unique_cpus;
  operator_handle->diagnostics.unique_physical_cores = unique_cores;
  if (operator_handle->affinity_cpus == NULL ||
      operator_handle->affinity_count < observed_threads) {
    int *cached = (int *)realloc(operator_handle->affinity_cpus,
                                 (size_t)observed_threads * sizeof(*cached));
    if (cached == NULL) {
      operator_handle->affinity_count = 0;
      return;
    }
    operator_handle->affinity_cpus = cached;
  }
  memcpy(operator_handle->affinity_cpus, operator_handle->worker_cpus,
         (size_t)observed_threads * sizeof(*operator_handle->affinity_cpus));
  operator_handle->affinity_count = observed_threads;
}

SsStatus ss_operator_collective_validation(SsOperator *operator_handle,
                                           const SsErrorRecord *local_error,
                                           SsErrorRecord *selected) {
  SsErrorRecord empty = {0};
  if (selected == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (ss_operator_service(operator_handle, SS_ASSEMBLY_SERVICE_MPI) != SS_OK ||
      ss_collective_error_select(operator_handle->plan->communicator,
                                 local_error == NULL ? &empty : local_error,
                                 selected) != SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  return selected->valid == 0 ? SS_OK : selected->status;
}

static SsStatus ss_operator_commit_staging(SsOperator *operator_handle,
                                           uint32_t request) {
  size_t row;
  if (ss_operator_service(operator_handle, SS_ASSEMBLY_SERVICE_PETSC) != SS_OK) {
    return SS_ERROR_THREAD;
  }
  if ((request & SS_ASSEMBLY_RESIDUAL) != 0U) {
    if (VecZeroEntries(operator_handle->staging_residual) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    for (row = 0U; row < operator_handle->plan->row_count; ++row) {
      const PetscInt global_row = operator_handle->plan->rows[row].global_row;
      const PetscScalar value = (PetscScalar)operator_handle->shadow_residual[row];
      if (VecSetValues(operator_handle->staging_residual, 1, &global_row, &value,
                       INSERT_VALUES) != PETSC_SUCCESS) {
        return SS_ERROR_INTERNAL;
      }
    }
    if (VecAssemblyBegin(operator_handle->staging_residual) != PETSC_SUCCESS ||
        VecAssemblyEnd(operator_handle->staging_residual) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
  }
  if ((request & SS_ASSEMBLY_JACOBIAN) != 0U) {
    MatInfo info = {0};
    if (MatZeroEntries(operator_handle->staging_jacobian) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    for (row = 0U; row < operator_handle->plan->row_count; ++row) {
      const SsOwnedRowPlan *item = &operator_handle->plan->rows[row];
      if (MatSetValues(operator_handle->staging_jacobian, 1, &item->global_row,
                       (PetscInt)item->column_count, item->columns,
                       operator_handle->shadow_jacobian +
                           operator_handle->row_offsets[row],
                       INSERT_VALUES) != PETSC_SUCCESS) {
        return SS_ERROR_INTERNAL;
      }
    }
    if (MatAssemblyBegin(operator_handle->staging_jacobian, MAT_FINAL_ASSEMBLY) !=
            PETSC_SUCCESS ||
        MatAssemblyEnd(operator_handle->staging_jacobian, MAT_FINAL_ASSEMBLY) !=
            PETSC_SUCCESS ||
        MatGetInfo(operator_handle->staging_jacobian, MAT_LOCAL, &info) !=
            PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    operator_handle->diagnostics.petsc_new_nonzero_allocations =
        info.mallocs < 0.0 ? UINT64_MAX : (uint64_t)info.mallocs;
    if (info.mallocs != 0.0) {
      return SS_ERROR_INTEGRITY;
    }
  }
  return SS_OK;
}

static SsStatus ss_operator_publish(SsOperator *operator_handle, uint32_t request,
                                    Vec residual, Mat jacobian) {
  if (ss_operator_service(operator_handle, SS_ASSEMBLY_SERVICE_PETSC) != SS_OK) {
    return SS_ERROR_THREAD;
  }
  if ((request & SS_ASSEMBLY_RESIDUAL) != 0U &&
      VecCopy(operator_handle->staging_residual, residual) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if ((request & SS_ASSEMBLY_JACOBIAN) != 0U &&
      MatCopy(operator_handle->staging_jacobian, jacobian, SAME_NONZERO_PATTERN) !=
          PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  return SS_OK;
}

/* The same team packs the field and processes all volume and facet tiles.
 * Each COO slot has one owner. Qualification uses the same static row
 * partition for every tile, so each row sums its inputs in the original order. */
static SsErrorRecord ss_operator_compute_tiles(SsOperator *operator_handle,
                                               const SsOperatorState *state,
                                               const PetscScalar *gathered,
                                               uint32_t request, int use_openmp,
                                               double *local_potential,
                                               long double *local_natural_signed_flux) {
  const size_t cell_count = operator_handle->plan->cell_count;
  const size_t facet_count = operator_handle->natural_facet_count;
  const size_t capacity = operator_handle->tile_capacity;
  const int performance = operator_handle->spec.determinism == SS_ASSEMBLY_PERFORMANCE;
  const uint32_t row_request = request & (SS_ASSEMBLY_RESIDUAL | SS_ASSEMBLY_JACOBIAN);
  const uint32_t facet_request = row_request | (request & SS_ASSEMBLY_POTENTIAL);
  const size_t cell_tiles = cell_count / capacity + (cell_count % capacity != 0U);
  const size_t facet_tiles = facet_count / capacity + (facet_count % capacity != 0U);
  int observed_threads = 1;
  uint64_t owned_facets = 0U;
  SsErrorRecord error = {0};
  (void)PetscLogEventBegin(ss_operator_compute_event, 0, 0, 0, 0);
#if SLOPE_ENABLE_OPENMP
#pragma omp parallel num_threads(                                                      \
        operator_handle->spec.requested_threads) if (use_openmp)                       \
    reduction(ss_first_error : error) reduction(+ : owned_facets)
#endif
  {
    operator_handle->worker_cpus[ss_operator_thread()] = ss_operator_cpu();
#if SLOPE_ENABLE_OPENMP
#pragma omp single nowait
    observed_threads = omp_get_num_threads();
#endif
    ss_operator_pack_cells(operator_handle, gathered);
    for (size_t start = 0U; start < cell_count;) {
      const size_t remaining = cell_count - start;
      const size_t count = remaining < capacity ? remaining : capacity;
#if SLOPE_ENABLE_OPENMP
#pragma omp for schedule(static)
#endif
      for (size_t slot = 0U; slot < count; ++slot) {
        ss_operator_evaluate_cell(operator_handle, state, request, start + slot, slot);
        if (performance) {
          ss_operator_stage_cell(operator_handle, request, start + slot, slot);
        }
        ss_operator_keep_first_error(&error, &operator_handle->element_errors[slot]);
      }
      if (!performance) {
        ss_operator_form_rows(operator_handle, request, start, start + count);
      }
      if ((request & SS_ASSEMBLY_POTENTIAL) != 0U) {
        /* The integral retains its original summation order. */
#if SLOPE_ENABLE_OPENMP
#pragma omp single
#endif
        for (size_t slot = 0U; slot < count; ++slot) {
          if (operator_handle->plan->cells[start + slot].cell_owned &&
              !operator_handle->element_errors[slot].valid) {
            *local_potential += operator_handle->element_potential[slot];
          }
        }
      }
      start += count;
    }
    for (size_t start = 0U; facet_request != 0U && start < facet_count;) {
      const size_t remaining = facet_count - start;
      const size_t count = remaining < capacity ? remaining : capacity;
#if SLOPE_ENABLE_OPENMP
#pragma omp for schedule(static)
#endif
      for (size_t slot = 0U; slot < count; ++slot) {
        const size_t facet = start + slot;
        ss_operator_evaluate_facet(operator_handle, state, facet_request, facet, slot);
        if (row_request != 0U && performance) {
          ss_operator_stage_natural(operator_handle, request, facet, slot);
        }
        ss_operator_keep_first_error(&error, &operator_handle->element_errors[slot]);
        owned_facets +=
            operator_handle->natural_facets[facet].pack.geometry.facet_owned != 0U;
      }
      if (row_request != 0U && !performance) {
        ss_operator_form_natural_rows(operator_handle, request, start, count);
      }
#if SLOPE_ENABLE_OPENMP
#pragma omp single
#endif
      for (size_t slot = 0U; slot < count; ++slot) {
        if (operator_handle->natural_facets[start + slot].pack.geometry.facet_owned &&
            !operator_handle->element_errors[slot].valid) {
          *local_natural_signed_flux +=
              (long double)operator_handle->element_potential[slot];
          *local_potential += operator_handle->element_natural_work[slot];
        }
      }
      start += count;
    }
  }
  (void)PetscLogEventEnd(ss_operator_compute_event, 0, 0, 0, 0);
  operator_handle->diagnostics.used_openmp = use_openmp;
  operator_handle->diagnostics.requested_threads =
      operator_handle->spec.requested_threads;
  operator_handle->diagnostics.tile_count = cell_tiles;
  operator_handle->diagnostics.observed_threads = observed_threads;
  if (use_openmp) {
    ++operator_handle->diagnostics.parallel_pack_calls;
    if (performance && operator_handle->performance_coo) {
      operator_handle->diagnostics.parallel_stage_calls +=
          cell_tiles + (row_request != 0U ? facet_tiles : 0U);
      operator_handle->diagnostics.parallel_stage_entities +=
          operator_handle->plan->summary.local_owned_cells +
          (row_request != 0U ? owned_facets : 0U);
    }
  }
  ss_operator_finish_affinity(operator_handle, observed_threads);
  const uint64_t owned_cells = operator_handle->plan->summary.local_owned_cells;
  operator_handle->diagnostics.cell_evaluations +=
      performance ? owned_cells : cell_count;
  if (facet_request != 0U) {
    operator_handle->diagnostics.natural_facet_evaluations +=
        performance ? owned_facets : facet_count;
    if (!performance) {
      operator_handle->diagnostics.duplicate_natural_facet_evaluations +=
          facet_count - owned_facets;
    }
  }
  if (!performance) {
    operator_handle->diagnostics.duplicate_cell_evaluations += cell_count - owned_cells;
  }
  return error;
}

static SsStatus ss_operator_assemble(SsOperator *operator_handle,
                                     const SsOperatorState *state, uint32_t request,
                                     Vec residual, Mat jacobian, double *functional,
                                     SsErrorRecord *error) {
  SsAssemblyFailurePoint failure_point;
  const PetscScalar *gathered = NULL;
  int gathered_open = 0;
  int use_openmp;
  double local_potential = 0.0;
  double global_potential = 0.0;
  long double local_natural_signed_flux = 0.0L;
  double global_natural_signed_flux = 0.0;
  SsErrorRecord local_error = {0};
  SsErrorRecord selected = {0};
  SsStatus status;
  if (operator_handle == NULL) {
    ss_assembly_error_private(error, SS_ERROR_INVALID_ARGUMENT, "ASM_OPERATOR_ARGUMENT",
                              0U, 0);
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = SS_OK;
  if (state == NULL || !ss_operator_request_valid(request) ||
      (operator_handle->spec.matrix_free != 0 &&
       (request & SS_ASSEMBLY_JACOBIAN) != 0U) ||
      ((request & SS_ASSEMBLY_POTENTIAL) != 0U && functional == NULL)) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  if (error != NULL) {
    memset(error, 0, sizeof(*error));
  }
  /* With no COO budget, PETSc insertion stays on the calling thread. */
  use_openmp = ss_operator_use_openmp(operator_handle) &&
               (operator_handle->spec.determinism != SS_ASSEMBLY_PERFORMANCE ||
                operator_handle->performance_coo ||
                (request & (SS_ASSEMBLY_RESIDUAL | SS_ASSEMBLY_JACOBIAN)) == 0U);
  ++operator_handle->diagnostics.assembly_calls;
  failure_point = operator_handle->failure_point;
  if (status == SS_OK) {
    status =
        ss_operator_validate_destinations(operator_handle, request, residual, jacobian);
  }
  if (status == SS_OK && state->cancelled != 0U) {
    status = SS_ERROR_CANCELLED;
  }
  if (status == SS_OK) {
    status =
        ss_operator_prepare_reduced_materials(operator_handle, state->reduction_factor);
  }
  if (status == SS_OK && failure_point == SS_ASSEMBLY_FAIL_PACK &&
      ss_operator_failure_matches(operator_handle, 0U) != 0) {
    status = SS_ERROR_INJECTED;
  }
  status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                 status);
  if (status != SS_OK) {
    ss_assembly_error_private(error, status,
                              status == SS_ERROR_CANCELLED ? "ASM_CANCEL_BEFORE_PACK"
                                                           : "ASM_PREPACK_FAILED",
                              0U, operator_handle->plan->rank);
    operator_handle->failure_point = SS_ASSEMBLY_FAIL_NONE;
    return status;
  }
  if (status == SS_OK) {
    status = ss_operator_gather(operator_handle, state, &gathered);
    gathered_open = status == SS_OK;
  }
  if (status == SS_OK && failure_point == SS_ASSEMBLY_FAIL_CANCEL_AFTER_PACK &&
      ss_operator_failure_matches(operator_handle, 0U) != 0) {
    status = SS_ERROR_CANCELLED;
  }
  status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                 status);
  if (status != SS_OK) {
    if (gathered_open) {
      (void)VecRestoreArrayRead(operator_handle->gathered_values, &gathered);
    }
    ss_assembly_error_private(error, status,
                              status == SS_ERROR_CANCELLED ? "ASM_CANCEL_AFTER_PACK"
                                                           : "ASM_PACK_FAILED",
                              0U, operator_handle->plan->rank);
    operator_handle->failure_point = SS_ASSEMBLY_FAIL_NONE;
    return status;
  }
  if (operator_handle->spec.determinism == SS_ASSEMBLY_PERFORMANCE) {
    (void)PetscLogEventBegin(ss_operator_prepare_event, 0, 0, 0, 0);
    status = ss_operator_prepare_performance_staging(operator_handle, request);
    (void)PetscLogEventEnd(ss_operator_prepare_event, 0, 0, 0, 0);
    status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                   status);
    if (status != SS_OK) {
      (void)VecRestoreArrayRead(operator_handle->gathered_values, &gathered);
      ss_assembly_error_private(error, status, "ASM_PERFORMANCE_STAGE_PREPARE", 0U,
                                operator_handle->plan->rank);
      operator_handle->failure_point = SS_ASSEMBLY_FAIL_NONE;
      return status;
    }
  }
  if (operator_handle->spec.determinism != SS_ASSEMBLY_PERFORMANCE) {
    const size_t nnz = operator_handle->spec.matrix_free == 0
                           ? (size_t)operator_handle->plan->summary.local_nonzeros
                           : 0U;
    memset(operator_handle->shadow_residual, 0,
           operator_handle->plan->row_count *
               sizeof(*operator_handle->shadow_residual));
    memset(operator_handle->shadow_jacobian, 0,
           nnz * sizeof(*operator_handle->shadow_jacobian));
    memset(operator_handle->row_complete, 0,
           operator_handle->plan->row_count * sizeof(*operator_handle->row_complete));
    memset(operator_handle->row_writer_violation, 0,
           operator_handle->plan->row_count *
               sizeof(*operator_handle->row_writer_violation));
  }
  {
    size_t row;
    if (operator_handle->spec.determinism != SS_ASSEMBLY_PERFORMANCE) {
      for (row = 0U; row < operator_handle->plan->row_count; ++row) {
        operator_handle->row_writer[row] = -1;
      }
    }
    for (row = 0U; row < (size_t)operator_handle->spec.requested_threads; ++row) {
      operator_handle->worker_cpus[row] = -1;
    }
  }
  local_error =
      ss_operator_compute_tiles(operator_handle, state, gathered, request, use_openmp,
                                &local_potential, &local_natural_signed_flux);
  if (VecRestoreArrayRead(operator_handle->gathered_values, &gathered) !=
      PETSC_SUCCESS) {
    SsErrorRecord array_error;
    ss_assembly_error_private(&array_error, SS_ERROR_INTERNAL, "ASM_PACK_RESTORE", 0U,
                              operator_handle->plan->rank);
    ss_operator_keep_first_error(&local_error, &array_error);
  }
  if (operator_handle->spec.determinism == SS_ASSEMBLY_PERFORMANCE &&
      !operator_handle->performance_coo) {
    /* Drain off-process Vec/Mat insertion stashes even for a rejected
     * assembly. Incomplete staging must not escape into the next call. */
    (void)PetscLogEventBegin(ss_operator_stage_event, 0, 0, 0, 0);
    const SsStatus staged =
        ss_operator_finish_performance_staging(operator_handle, request);
    (void)PetscLogEventEnd(ss_operator_stage_event, 0, 0, 0, 0);
    if (staged != SS_OK) {
      SsErrorRecord staging_error;
      ss_assembly_error_private(&staging_error, staged, "ASM_PERFORMANCE_STAGE_FINISH",
                                0U, operator_handle->plan->rank);
      ss_operator_keep_first_error(&local_error, &staging_error);
    }
  }
  if (operator_handle->spec.determinism == SS_ASSEMBLY_PERFORMANCE) {
    operator_handle->diagnostics.completed_rows =
        (uint64_t)operator_handle->plan->row_count;
    operator_handle->diagnostics.row_writer_violations = 0U;
  } else {
    size_t row;
    uint64_t completed = 0U;
    uint64_t violations = 0U;
    for (row = 0U; row < operator_handle->plan->row_count; ++row) {
      if (operator_handle->row_complete[row] != 0U) {
        ++completed;
      } else if (local_error.valid == 0) {
        ss_assembly_error_private(
            &local_error, SS_ERROR_INTEGRITY, "ASM_INCOMPLETE_ROW",
            (SsEntityId)operator_handle->plan->rows[row].global_row,
            operator_handle->plan->rank);
      }
      if (operator_handle->row_writer_violation[row] != 0U) {
        ++violations;
      }
    }
    operator_handle->diagnostics.completed_rows = completed;
    operator_handle->diagnostics.row_writer_violations = violations;
    if (violations != 0U && local_error.valid == 0) {
      ss_assembly_error_private(&local_error, SS_ERROR_THREAD,
                                "ASM_ROW_WRITER_VIOLATION", 0U,
                                operator_handle->plan->rank);
    }
  }
  if (failure_point == SS_ASSEMBLY_FAIL_NONFINITE &&
      ss_operator_failure_matches(operator_handle, 0U) != 0 &&
      operator_handle->plan->row_count != 0U) {
    if (operator_handle->spec.determinism == SS_ASSEMBLY_PERFORMANCE) {
      if ((request & SS_ASSEMBLY_RESIDUAL) != 0U && local_error.valid == 0) {
        ss_assembly_error_private(&local_error, SS_ERROR_INTEGRITY, "ASM_NONFINITE_ROW",
                                  (SsEntityId)operator_handle->plan->rows[0].global_row,
                                  operator_handle->plan->rank);
      }
    } else {
      operator_handle->shadow_residual[0] = NAN;
    }
  }
  if (operator_handle->spec.determinism != SS_ASSEMBLY_PERFORMANCE) {
    size_t row;
    for (row = 0U; row < operator_handle->plan->row_count; ++row) {
      size_t column;
      if ((request & SS_ASSEMBLY_RESIDUAL) != 0U &&
          !isfinite(operator_handle->shadow_residual[row]) && local_error.valid == 0) {
        ss_assembly_error_private(
            &local_error, SS_ERROR_INTEGRITY, "ASM_NONFINITE_ROW",
            (SsEntityId)operator_handle->plan->rows[row].global_row,
            operator_handle->plan->rank);
      }
      if ((request & SS_ASSEMBLY_JACOBIAN) == 0U) {
        continue;
      }
      for (column = operator_handle->row_offsets[row];
           column < operator_handle->row_offsets[row + 1U]; ++column) {
        if (!isfinite(operator_handle->shadow_jacobian[column]) &&
            local_error.valid == 0) {
          ss_assembly_error_private(
              &local_error, SS_ERROR_INTEGRITY, "ASM_NONFINITE_ROW",
              (SsEntityId)operator_handle->plan->rows[row].global_row,
              operator_handle->plan->rank);
        }
      }
    }
  }
  if (failure_point == SS_ASSEMBLY_FAIL_CANCEL_AFTER_COMPUTE &&
      ss_operator_failure_matches(operator_handle, 0U) != 0 && local_error.valid == 0) {
    ss_assembly_error_private(&local_error, SS_ERROR_CANCELLED,
                              "ASM_CANCEL_AFTER_COMPUTE", 0U,
                              operator_handle->plan->rank);
  }
  (void)PetscLogEventBegin(ss_operator_validate_event, 0, 0, 0, 0);
  status = ss_operator_collective_validation(operator_handle, &local_error, &selected);
  (void)PetscLogEventEnd(ss_operator_validate_event, 0, 0, 0, 0);
  if (status != SS_OK) {
    if (error != NULL) {
      *error = selected;
    }
    operator_handle->failure_point = SS_ASSEMBLY_FAIL_NONE;
    return status;
  }
  if ((request & SS_ASSEMBLY_POTENTIAL) != 0U) {
    if (ss_operator_service(operator_handle, SS_ASSEMBLY_SERVICE_MPI) != SS_OK ||
        MPI_Allreduce(&local_potential, &global_potential, 1, MPI_DOUBLE, MPI_SUM,
                      operator_handle->plan->communicator) != MPI_SUCCESS) {
      operator_handle->failure_point = SS_ASSEMBLY_FAIL_NONE;
      return SS_ERROR_INTERNAL;
    }
  }
  if ((request & SS_ASSEMBLY_RESIDUAL) != 0U) {
    const double local_flux = (double)local_natural_signed_flux;
    if (ss_operator_service(operator_handle, SS_ASSEMBLY_SERVICE_MPI) != SS_OK ||
        MPI_Allreduce(&local_flux, &global_natural_signed_flux, 1, MPI_DOUBLE, MPI_SUM,
                      operator_handle->plan->communicator) != MPI_SUCCESS ||
        !isfinite(global_natural_signed_flux)) {
      operator_handle->failure_point = SS_ASSEMBLY_FAIL_NONE;
      return SS_ERROR_INTERNAL;
    }
  }
  if (failure_point == SS_ASSEMBLY_FAIL_PRECOMMIT &&
      ss_operator_failure_matches(operator_handle, 0U) != 0) {
    ss_assembly_error_private(&local_error, SS_ERROR_INJECTED, "ASM_INJECTED_PRECOMMIT",
                              0U, operator_handle->plan->rank);
  } else {
    memset(&local_error, 0, sizeof(local_error));
  }
  (void)PetscLogEventBegin(ss_operator_commit_event, 0, 0, 0, 0);
  status = ss_operator_collective_validation(operator_handle, &local_error, &selected);
  if (status == SS_OK) {
    if (operator_handle->spec.determinism == SS_ASSEMBLY_PERFORMANCE) {
      if (operator_handle->performance_coo) {
        (void)PetscLogEventBegin(ss_operator_stage_event, 0, 0, 0, 0);
        status = ss_operator_finish_performance_staging(operator_handle, request);
        (void)PetscLogEventEnd(ss_operator_stage_event, 0, 0, 0, 0);
      }
    } else {
      status = ss_operator_commit_staging(operator_handle, request);
    }
  }
  status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                 status);
  (void)PetscLogEventEnd(ss_operator_commit_event, 0, 0, 0, 0);
  if (status == SS_OK) {
    status = ss_operator_publish(operator_handle, request, residual, jacobian);
  }
  status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                 status);
  operator_handle->failure_point = SS_ASSEMBLY_FAIL_NONE;
  if (status != SS_OK) {
    if (selected.valid != 0 && error != NULL) {
      *error = selected;
    } else {
      ss_assembly_error_private(error, status, "ASM_COMMIT_FAILED", 0U,
                                operator_handle->plan->rank);
    }
    return status;
  }
  if ((request & SS_ASSEMBLY_POTENTIAL) != 0U) {
    *functional = global_potential;
  }
  if ((request & SS_ASSEMBLY_RESIDUAL) != 0U) {
    operator_handle->diagnostics.natural_signed_flux = global_natural_signed_flux;
  }
  ++operator_handle->diagnostics.successful_commits;
  ++operator_handle->diagnostics.value_generation;
  operator_handle->properties.value_generation =
      operator_handle->diagnostics.value_generation;
  return SS_OK;
}

SsStatus ss_operator_assemble_profiled(SsOperator *operator_handle,
                                       const SsOperatorState *state, uint32_t request,
                                       Vec residual, Mat jacobian, double *functional,
                                       SsErrorRecord *error) {
  SsStatus status;
  if (PetscLogEventBegin(ss_operator_assembly_event, 0, 0, 0, 0) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  status = ss_operator_assemble(operator_handle, state, request, residual, jacobian,
                                functional, error);
  if (PetscLogEventEnd(ss_operator_assembly_event, 0, 0, 0, 0) != PETSC_SUCCESS &&
      status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  return status;
}
