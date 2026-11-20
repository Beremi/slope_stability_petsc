/* Operator lifecycle, diagnostics and assembled-operation entry points.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "operator_internal.h"

#include "slope_internal/build/config.h"
#include "slope_internal/core/sort.h"

#include <petscis.h>
#include <petsclog.h>
#include <petscvec.h>

#if SLOPE_ENABLE_OPENMP
#include <omp.h>
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

PetscLogEvent ss_operator_assembly_event = -1;
PetscLogEvent ss_operator_scatter_event = -1;
PetscLogEvent ss_operator_prepare_event = -1;
PetscLogEvent ss_operator_compute_event = -1;
PetscLogEvent ss_operator_stage_event = -1;
PetscLogEvent ss_operator_commit_event = -1;
PetscLogEvent ss_operator_validate_event = -1;
PetscLogEvent ss_operator_p1_project_event = -1;
PetscLogEvent ss_operator_matrix_free_fine_event = -1;
PetscLogEvent ss_operator_matrix_free_projected_event = -1;
PetscLogEvent ss_operator_matrix_free_block_apply_event = -1;
PetscLogEvent ss_operator_matrix_free_block_setup_event = -1;
PetscLogEvent ss_operator_matrix_free_tangent_event = -1;

PetscErrorCode ss_operator_register_events(void) {
  if (ss_operator_assembly_event >= 0) {
    return PETSC_SUCCESS;
  }
  if (PetscLogEventRegister("SlopeAssembly", 0, &ss_operator_assembly_event) !=
          PETSC_SUCCESS ||
      PetscLogEventRegister("SlopeAsmScatter", 0, &ss_operator_scatter_event) !=
          PETSC_SUCCESS ||
      PetscLogEventRegister("SlopeAsmPrepare", 0, &ss_operator_prepare_event) !=
          PETSC_SUCCESS ||
      PetscLogEventRegister("SlopeAsmCompute", 0, &ss_operator_compute_event) !=
          PETSC_SUCCESS ||
      PetscLogEventRegister("SlopeAsmStage", 0, &ss_operator_stage_event) !=
          PETSC_SUCCESS ||
      PetscLogEventRegister("SlopeAsmCommit", 0, &ss_operator_commit_event) !=
          PETSC_SUCCESS ||
      PetscLogEventRegister("SlopeAsmValidate", 0, &ss_operator_validate_event) !=
          PETSC_SUCCESS ||
      PetscLogEventRegister("SlopeP1Project", 0, &ss_operator_p1_project_event) !=
          PETSC_SUCCESS ||
      PetscLogEventRegister("SlopeMatFreeFine", 0,
                            &ss_operator_matrix_free_fine_event) != PETSC_SUCCESS ||
      PetscLogEventRegister("SlopeMatFreeProjected", 0,
                            &ss_operator_matrix_free_projected_event) !=
          PETSC_SUCCESS ||
      PetscLogEventRegister("SlopeMatFreeBlockPC", 0,
                            &ss_operator_matrix_free_block_apply_event) !=
          PETSC_SUCCESS ||
      PetscLogEventRegister("SlopeMatFreeBlockSetup", 0,
                            &ss_operator_matrix_free_block_setup_event) !=
          PETSC_SUCCESS ||
      PetscLogEventRegister("SlopeMatFreeTangent", 0,
                            &ss_operator_matrix_free_tangent_event) != PETSC_SUCCESS) {
    return PETSC_ERR_LIB;
  }
  return PETSC_SUCCESS;
}

int ss_operator_request_valid(uint32_t request) {
  const uint32_t supported =
      SS_ASSEMBLY_RESIDUAL | SS_ASSEMBLY_JACOBIAN | SS_ASSEMBLY_POTENTIAL;
  return request != 0U && (request & ~supported) == 0U;
}

SsStatus ss_operator_create(const SsAssemblyPlan *plan, const SsOperatorSpec *spec,
                            SsOperator **operator_handle, SsErrorRecord *error) {
  SsOperator *created = NULL;
  SsStatus status = SS_OK;
  const char *creation_code = "ASM_OPERATOR_CREATE";
  size_t region;
  if (operator_handle != NULL) {
    *operator_handle = NULL;
  }
  if (plan == NULL || spec == NULL || operator_handle == NULL ||
      (spec->physics != SS_ASSEMBLY_MECHANICS &&
       spec->physics != SS_ASSEMBLY_SCALAR_FLUX) ||
      (spec->determinism != SS_ASSEMBLY_QUALIFICATION &&
       spec->determinism != SS_ASSEMBLY_PERFORMANCE) ||
      spec->requested_threads < 1 ||
      (spec->matrix_free != 0 && spec->matrix_free != 1) ||
      (spec->matrix_free != 0 && spec->physics != SS_ASSEMBLY_MECHANICS) ||
      ((int)spec->volume_quadrature_mode != 0 &&
       spec->volume_quadrature_mode != SS_FE_QUADRATURE_MODE_DEFAULT &&
       spec->volume_quadrature_mode != SS_FE_QUADRATURE_MODE_PAIRED_HIGHER) ||
      (spec->natural_operation_count != 0U && spec->natural_operations == NULL)) {
    ss_assembly_error_private(error, SS_ERROR_INVALID_ARGUMENT, "ASM_OPERATOR_SPEC", 0U,
                              plan == NULL ? 0 : plan->rank);
    return SS_ERROR_INVALID_ARGUMENT;
  }
#if !SLOPE_ENABLE_OPENMP
  if (spec->requested_threads != 1) {
    ss_assembly_error_private(error, SS_ERROR_INVALID_ARGUMENT, "ASM_OPENMP_DISABLED",
                              0U, plan->rank);
    return SS_ERROR_INVALID_ARGUMENT;
  }
#endif
  if (ss_operator_register_events() != PETSC_SUCCESS) {
    ss_assembly_error_private(error, SS_ERROR_INTERNAL, "ASM_LOG_EVENT", 0U,
                              plan->rank);
    return SS_ERROR_INTERNAL;
  }
  if ((spec->physics == SS_ASSEMBLY_MECHANICS &&
       (spec->mechanics == NULL || spec->mechanics->regions == NULL ||
        spec->mechanics->region_count == 0U || plan->field_summary.components == 0U)) ||
      (spec->physics == SS_ASSEMBLY_SCALAR_FLUX &&
       (spec->scalar == NULL || spec->scalar->point_kernel == NULL ||
        plan->field_summary.components != 1U))) {
    ss_assembly_error_private(error, SS_ERROR_INVALID_ARGUMENT, "ASM_OPERATOR_PHYSICS",
                              0U, plan->rank);
    return SS_ERROR_INVALID_ARGUMENT;
  }
  created = (SsOperator *)calloc(1U, sizeof(*created));
  if (created == NULL) {
    status = SS_ERROR_OUT_OF_MEMORY;
  }
  status = ss_assembly_collective_status_private(plan->communicator, status);
  if (status != SS_OK || created == NULL) {
    if (status == SS_OK) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
    free(created);
    return status;
  }
  created->plan = plan;
  created->spec = *spec;
  created->matrix_free_action_batch_cells = plan->field_summary.order >= 8U ? 8U : 1U;
  created->matrix_free_block_node_cap = 1U;
  created->matrix_free_threaded_block_apply = 0;
  created->failure_rank = -1;
  created->maximum_closure = plan->summary.maximum_closure_count;
  created->properties.physics = spec->physics;
  created->properties.determinism = spec->determinism;
  created->properties.dimension =
      plan->cell_count == 0U ? 0U : plan->cells[0].dimension;
  created->properties.components = plan->field_summary.components;
  created->properties.local_rows = plan->summary.local_owned_rows;
  created->properties.global_rows = plan->summary.global_rows;
  created->properties.linear = 0;
  created->properties.state_dependent = 1;
  created->properties.symmetric = 0;
  created->properties.positive_definite = 0;
  created->properties.supported_requests =
      SS_ASSEMBLY_RESIDUAL | (spec->matrix_free == 0 ? SS_ASSEMBLY_JACOBIAN : 0U) |
      (spec->physics == SS_ASSEMBLY_MECHANICS ? SS_ASSEMBLY_POTENTIAL : 0U);
  created->properties.pattern_generation = plan->summary.pattern_generation;
  created->properties.coordinate_generation = 1U;
  created->properties.constraint_generation = plan->field_summary.constraint_generation;
  memcpy(created->properties.pattern_digest, plan->summary.pattern_digest,
         sizeof(created->properties.pattern_digest));
  memcpy(created->diagnostics.pattern_digest, plan->summary.pattern_digest,
         sizeof(created->diagnostics.pattern_digest));
  if (spec->matrix_free != 0) {
    const size_t block_node_cap =
        spec->matrix_free_block_nodes != 0U ? spec->matrix_free_block_nodes : 1U;
    const size_t batch_cells = spec->matrix_free_batch_cells != 0U
                                   ? spec->matrix_free_batch_cells
                                   : created->matrix_free_action_batch_cells;
    if (block_node_cap > SS_OPERATOR_MATRIX_FREE_MAX_BLOCK_NODES ||
        batch_cells > SS_OPERATOR_MATRIX_FREE_MAX_ACTION_BATCH) {
      status = SS_ERROR_INVALID_ARGUMENT;
    } else {
      created->matrix_free_block_node_cap = block_node_cap;
      created->matrix_free_threaded_block_apply =
          spec->requested_threads > 1 && block_node_cap > 1U;
      created->matrix_free_action_batch_cells = batch_cells;
    }
  }
  size_t bytes;
  if (ss_checked_mul_size(spec->natural_operation_count,
                          sizeof(*created->natural_operations), &bytes) != SS_OK) {
    status = SS_ERROR_OVERFLOW;
  } else if (spec->natural_operation_count != 0U) {
    created->natural_operations = (SsNaturalOperatorSpec *)malloc(bytes);
    if (created->natural_operations == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    } else {
      memcpy(created->natural_operations, spec->natural_operations, bytes);
      created->spec.natural_operations = created->natural_operations;
      for (region = 0U; region < spec->natural_operation_count; ++region) {
        if (created->natural_operations[region].point_kernel == NULL ||
            (created->natural_operations[region].scaled_by_external_load != 0 &&
             created->natural_operations[region].scaled_by_external_load != 1)) {
          status = SS_ERROR_INVALID_ARGUMENT;
        }
      }
    }
  }
  if (spec->physics == SS_ASSEMBLY_MECHANICS) {
    created->mechanics = *spec->mechanics;
    if (ss_checked_mul_size(spec->mechanics->region_count,
                            sizeof(*created->material_regions), &bytes) != SS_OK) {
      status = SS_ERROR_OVERFLOW;
    } else {
      created->material_regions = (SsMechanicsMaterialRegion *)malloc(bytes);
      if (created->material_regions == NULL) {
        status = SS_ERROR_OUT_OF_MEMORY;
      } else {
        memcpy(created->material_regions, spec->mechanics->regions, bytes);
        ss_sort(created->material_regions, spec->mechanics->region_count,
                sizeof(*created->material_regions), ss_operator_material_compare);
        created->mechanics.regions = created->material_regions;
        for (region = 0U; region < created->mechanics.region_count; ++region) {
          size_t coordinate;
          if (created->material_regions[region].parameters == NULL ||
              (created->material_regions[region].has_body_force != 0 &&
               created->material_regions[region].has_body_force != 1) ||
              (region != 0U && created->material_regions[region - 1U].material_value ==
                                   created->material_regions[region].material_value)) {
            status = SS_ERROR_INVALID_ARGUMENT;
          }
          for (coordinate = 0U; coordinate < SS_FE_MAX_DIMENSION &&
                                created->material_regions[region].has_body_force != 0;
               ++coordinate) {
            if (!isfinite(created->material_regions[region].body_force[coordinate])) {
              status = SS_ERROR_INVALID_ARGUMENT;
            }
          }
        }
      }
    }
  } else {
    created->scalar = *spec->scalar;
  }
  status = ss_assembly_collective_status_private(plan->communicator, status);
  if (status == SS_OK &&
      ((spec->natural_operation_count != 0U && created->natural_operations == NULL) ||
       (spec->physics == SS_ASSEMBLY_MECHANICS && created->material_regions == NULL))) {
    status = SS_ERROR_OUT_OF_MEMORY;
  }
  if (status == SS_OK) {
    creation_code = "ASM_OPERATOR_CELL_METADATA";
    status = ss_operator_build_cell_metadata(created);
  }
  if (status == SS_OK) {
    creation_code = "ASM_OPERATOR_NATURAL_FACETS";
    status = ss_operator_build_natural_facets(created);
  }
  if (status == SS_OK) {
    creation_code = "ASM_OPERATOR_GATHER";
    status = ss_operator_build_gather(created);
  }
  if (status == SS_OK) {
    creation_code = "ASM_OPERATOR_WORKSPACE";
    status = ss_operator_allocate_work(created);
  }
  status = ss_assembly_collective_status_private(plan->communicator, status);
  if (status == SS_OK) {
    creation_code = "ASM_OPERATOR_RESIDUAL";
    status = ss_assembly_plan_create_vector(plan, &created->staging_residual);
  }
  if (status == SS_OK) {
    /*
     * A VecScatter records layouts and index maps, not the identity of the
     * source Vec used to create it.  Build the context once from the staging
     * vector so later residual/Jacobian calls can use any compatible state
     * vector without a rank-local pointer comparison around a collective
     * VecScatterCreate().
     */
    creation_code = "ASM_OPERATOR_SCATTER";
    if (ss_operator_service(created, SS_ASSEMBLY_SERVICE_PETSC) != SS_OK ||
        VecScatterCreate(created->staging_residual, created->gather_is,
                         created->gathered_values, NULL,
                         &created->scatter) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  status = ss_assembly_collective_status_private(plan->communicator, status);
  if (status == SS_OK && spec->matrix_free == 0) {
    creation_code = "ASM_OPERATOR_JACOBIAN";
    status = ss_assembly_plan_create_matrix(plan, &created->staging_jacobian);
  }
  if (status == SS_OK && spec->determinism == SS_ASSEMBLY_PERFORMANCE) {
    creation_code = "ASM_OPERATOR_PERFORMANCE_COO";
    status = ss_operator_prepare_performance_coo(created);
    if (status == SS_OK && created->performance_coo == 0) {
      if (VecSetOption(created->staging_residual, VEC_SUBSET_OFF_PROC_ENTRIES,
                       PETSC_TRUE) != PETSC_SUCCESS ||
          (spec->matrix_free == 0 &&
           MatSetOption(created->staging_jacobian, MAT_SUBSET_OFF_PROC_ENTRIES,
                        PETSC_TRUE) != PETSC_SUCCESS)) {
        status = SS_ERROR_INTERNAL;
      }
    }
  }
  status = ss_assembly_collective_status_private(plan->communicator, status);
  if (status != SS_OK) {
    ss_assembly_error_private(error, status, creation_code, 0U, plan->rank);
    (void)ss_operator_destroy(&created);
    return status;
  }
  if (error != NULL) {
    memset(error, 0, sizeof(*error));
  }
  *operator_handle = created;
  return SS_OK;
}

/* Attach the mechanics-node block partition owned by operator_handle to a
 * compatible assembled PETSc matrix.  Constrained boundary nodes may produce
 * smaller variable blocks, so a fixed Mat block size is insufficient. */
static SsStatus ss_operator_variable_block_sizes_set(const SsOperator *operator_handle,
                                                     Mat matrix) {
  PetscInt global_rows = 0;
  PetscInt global_columns = 0;
  PetscInt row_start = 0;
  PetscInt row_end = 0;
  PetscInt *block_sizes = NULL;
  size_t block_count = 0U;
  size_t block_entries = 0U;
  size_t block = 0U;
  size_t row = 0U;
  SsStatus status = SS_OK;
  MPI_Comm matrix_communicator;
  int communicator_comparison = MPI_UNEQUAL;
  if (operator_handle == NULL || matrix == NULL ||
      operator_handle->spec.physics != SS_ASSEMBLY_MECHANICS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  matrix_communicator = PetscObjectComm((PetscObject)matrix);
  if (MPI_Comm_compare(operator_handle->plan->communicator, matrix_communicator,
                       &communicator_comparison) != MPI_SUCCESS ||
      (communicator_comparison != MPI_IDENT &&
       communicator_comparison != MPI_CONGRUENT) ||
      MatGetSize(matrix, &global_rows, &global_columns) != PETSC_SUCCESS ||
      MatGetOwnershipRange(matrix, &row_start, &row_end) != PETSC_SUCCESS ||
      global_rows != global_columns ||
      global_rows != (PetscInt)operator_handle->plan->summary.global_rows ||
      row_start != operator_handle->plan->row_start ||
      row_end != operator_handle->plan->row_end) {
    status = SS_ERROR_TYPE_MISMATCH;
  }
  if (status == SS_OK) {
    status = ss_operator_mechanics_block_requirements(operator_handle, &block_count,
                                                      &block_entries);
  }
  (void)block_entries;
  if (status == SS_OK && block_count != 0U) {
    block_sizes = (PetscInt *)calloc(block_count, sizeof(*block_sizes));
    if (block_sizes == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                 status);
  while (status == SS_OK && row < operator_handle->plan->row_count) {
    size_t block_size = 0U;
    if (block >= block_count ||
        ss_operator_mechanics_block_size(operator_handle, row, &block_size) != SS_OK ||
        block_size > (size_t)PETSC_MAX_INT) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    block_sizes[block++] = (PetscInt)block_size;
    row += block_size;
  }
  if (status == SS_OK && block != block_count) {
    status = SS_ERROR_INTEGRITY;
  }
  status = ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                                 status);
  if (status == SS_OK && MatSetVariableBlockSizes(matrix, (PetscInt)block_count,
                                                  block_sizes) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  free(block_sizes);
  return ss_assembly_collective_status_private(operator_handle->plan->communicator,
                                               status);
}

SsStatus ss_operator_properties(const SsOperator *operator_handle,
                                SsOperatorProperties *properties) {
  if (operator_handle == NULL || properties == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *properties = operator_handle->properties;
  return SS_OK;
}

SsStatus ss_operator_diagnostics(const SsOperator *operator_handle,
                                 SsAssemblyDiagnostics *diagnostics) {
  if (operator_handle == NULL || diagnostics == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *diagnostics = operator_handle->diagnostics;
  return SS_OK;
}

SsStatus ss_operator_residual(SsOperator *operator_handle, const SsOperatorState *state,
                              Vec residual, SsErrorRecord *error) {
  return ss_operator_assemble_profiled(operator_handle, state, SS_ASSEMBLY_RESIDUAL,
                                       residual, NULL, NULL, error);
}

SsStatus ss_operator_jacobian(SsOperator *operator_handle, const SsOperatorState *state,
                              Mat jacobian, SsErrorRecord *error) {
  SsStatus status = ss_operator_assemble_profiled(
      operator_handle, state, SS_ASSEMBLY_JACOBIAN, NULL, jacobian, NULL, error);
  if (status == SS_OK && operator_handle->spec.physics == SS_ASSEMBLY_MECHANICS) {
    status = ss_operator_variable_block_sizes_set(operator_handle, jacobian);
  }
  return status;
}

SsStatus ss_operator_residual_jacobian(SsOperator *operator_handle,
                                       const SsOperatorState *state, Vec residual,
                                       Mat jacobian, SsErrorRecord *error) {
  SsStatus status = ss_operator_assemble_profiled(
      operator_handle, state, SS_ASSEMBLY_FUSED, residual, jacobian, NULL, error);
  if (status == SS_OK && operator_handle->spec.physics == SS_ASSEMBLY_MECHANICS) {
    status = ss_operator_variable_block_sizes_set(operator_handle, jacobian);
  }
  return status;
}

SsStatus ss_operator_residual_functional(SsOperator *operator_handle,
                                         const SsOperatorState *state, Vec residual,
                                         double *functional, SsErrorRecord *error) {
  return ss_operator_assemble_profiled(operator_handle, state,
                                       SS_ASSEMBLY_RESIDUAL | SS_ASSEMBLY_POTENTIAL,
                                       residual, NULL, functional, error);
}

SsStatus ss_operator_residual_jacobian_functional(SsOperator *operator_handle,
                                                  const SsOperatorState *state,
                                                  Vec residual, Mat jacobian,
                                                  double *functional,
                                                  SsErrorRecord *error) {
  SsStatus status = ss_operator_assemble_profiled(
      operator_handle, state, SS_ASSEMBLY_FUSED | SS_ASSEMBLY_POTENTIAL, residual,
      jacobian, functional, error);
  if (status == SS_OK && operator_handle->spec.physics == SS_ASSEMBLY_MECHANICS) {
    status = ss_operator_variable_block_sizes_set(operator_handle, jacobian);
  }
  return status;
}

SsStatus ss_operator_functional(SsOperator *operator_handle,
                                const SsOperatorState *state, double *functional,
                                SsErrorRecord *error) {
  return ss_operator_assemble_profiled(operator_handle, state, SS_ASSEMBLY_POTENTIAL,
                                       NULL, NULL, functional, error);
}

SsStatus ss_operator_inject_failure(SsOperator *operator_handle,
                                    SsAssemblyFailurePoint point, int rank,
                                    SsEntityId stable_entity_id) {
  if (operator_handle == NULL || point < SS_ASSEMBLY_FAIL_NONE ||
      point > SS_ASSEMBLY_FAIL_ACTION || rank < -1 ||
      rank >= operator_handle->plan->size) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  operator_handle->failure_point = point;
  operator_handle->failure_rank = rank;
  operator_handle->failure_entity = stable_entity_id;
  return SS_OK;
}

SsStatus ss_operator_set_service_observer(SsOperator *operator_handle,
                                          SsAssemblyServiceObserver observer,
                                          void *context) {
  if (operator_handle == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  operator_handle->service_observer = observer;
  operator_handle->service_context = context;
  return SS_OK;
}

SsStatus ss_operator_destroy(SsOperator **operator_handle) {
  SsStatus status = SS_OK;
  if (operator_handle == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (*operator_handle == NULL) {
    return SS_OK;
  }
  if (VecScatterDestroy(&(*operator_handle)->scatter) != PETSC_SUCCESS ||
      ISDestroy(&(*operator_handle)->gather_is) != PETSC_SUCCESS ||
      VecDestroy(&(*operator_handle)->gathered_values) != PETSC_SUCCESS ||
      VecDestroy(&(*operator_handle)->staging_residual) != PETSC_SUCCESS ||
      MatDestroy(&(*operator_handle)->staging_jacobian) != PETSC_SUCCESS ||
      MatDestroy(&(*operator_handle)->matrix_free_operator) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  free((*operator_handle)->material_regions);
  free((*operator_handle)->natural_operations);
  free((*operator_handle)->natural_facets);
  free((*operator_handle)->cells);
  free((*operator_handle)->cell_basis_cache);
  free((*operator_handle)->basis_blas_cache);
  free((*operator_handle)->cell_inverse_transpose_cache);
  free((*operator_handle)->reference_gradient_cache);
  free((*operator_handle)->reference_gradient_blas_cache);
  free((*operator_handle)->cell_physical_cache);
  free((*operator_handle)->cell_weight_cache);
  free((*operator_handle)->reduced_materials);
  free((*operator_handle)->gather_indices);
  free((*operator_handle)->pack_required);
  free((*operator_handle)->packed_slots);
  free((*operator_handle)->packed_values);
  free((*operator_handle)->element_residual);
  free((*operator_handle)->element_jacobian);
  free((*operator_handle)->element_potential);
  free((*operator_handle)->element_natural_work);
  free((*operator_handle)->stage_matrix_scratch);
  free((*operator_handle)->affinity_cpus);
  free((*operator_handle)->element_errors);
  free((*operator_handle)->row_offsets);
  free((*operator_handle)->shadow_residual);
  free((*operator_handle)->shadow_jacobian);
  free((*operator_handle)->row_complete);
  free((*operator_handle)->row_writer_violation);
  free((*operator_handle)->row_writer);
  free((*operator_handle)->worker_cpus);
  free((*operator_handle)->coo_residual_values);
  free((*operator_handle)->coo_jacobian_values);
  free((*operator_handle)->coo_residual_offsets);
  free((*operator_handle)->coo_jacobian_offsets);
  free((*operator_handle)->matrix_free_tangent_offsets);
  free((*operator_handle)->matrix_free_tangents);
  free((*operator_handle)->matrix_free_element_action);
  free((*operator_handle)->matrix_free_diagonal);
  free((*operator_handle)->matrix_free_block_sizes);
  free((*operator_handle)->matrix_free_block_row_offsets);
  free((*operator_handle)->matrix_free_block_offsets);
  free((*operator_handle)->matrix_free_inverse_blocks);
  free((*operator_handle)->matrix_free_block_workspace);
  free((*operator_handle)->matrix_free_action_workspace);
  free((*operator_handle)->matrix_free_p1_constraint_pattern_indices);
  free((*operator_handle)->matrix_free_p1_reference_gradient_blas_cache);
  free((*operator_handle)->matrix_free_projection_gradient_offsets);
  free((*operator_handle)->matrix_free_projection_gradients);
  free((*operator_handle)->matrix_free_projection_transfer);
  free((*operator_handle)->matrix_free_projection_source_constrained);
  free((*operator_handle)->matrix_free_projection_constraint_pattern_indices);
  free((*operator_handle)->matrix_free_projection_constraint_pattern_representatives);
  free((*operator_handle)->matrix_free_projection_constraint_reference_blas_cache);
  free((*operator_handle)->matrix_free_projection_physical_pattern_indices);
  free((*operator_handle)->matrix_free_projection_physical_pattern_representatives);
  free((*operator_handle)->matrix_free_projection_pattern_physical_gradients);
  free((*operator_handle)->matrix_free_projection_reference_gradients);
  free((*operator_handle)->matrix_free_projection_reference_gradient_blas_cache);
  free(*operator_handle);
  *operator_handle = NULL;
  return status;
}
