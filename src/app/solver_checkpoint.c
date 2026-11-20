/* Checkpoint buffers, controller state and deflation persistence adapters.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "solver_internal.h"

#include "slope_internal/core/sha256.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
ss_solver_checkpoint_field_buffer_destroy(SsSolverCheckpointFieldBuffer *field) {
  if (field == NULL) {
    return;
  }
  free(field->values);
  free(field->local_slots);
  free(field->stable_ids);
  memset(field, 0, sizeof(*field));
}

void ss_solver_checkpoint_buffer_destroy(SsSolverCheckpointBuffer *buffer) {
  if (buffer == NULL) {
    return;
  }
  free(buffer->deflation.right_values);
  free(buffer->deflation.left_values);
  free(buffer->deflation.source_values);
  free(buffer->deflation.source_weights);
  free(buffer->deflation.source_kinds);
  ss_solver_checkpoint_field_buffer_destroy(&buffer->pressure);
  ss_solver_checkpoint_field_buffer_destroy(&buffer->displacement);
  ss_solver_checkpoint_field_buffer_destroy(&buffer->previous_displacement);
  memset(buffer, 0, sizeof(*buffer));
}

SsStatus
ss_solver_checkpoint_field_buffer_create(const SsFieldSpace *space, Vec vector,
                                         int capture,
                                         SsSolverCheckpointFieldBuffer *field) {
  SsFieldSummary summary;
  const PetscScalar *source = NULL;
  PetscInt local_size = 0;
  size_t index;
  size_t count = 0U;
  SsStatus status;
  if (space == NULL || vector == NULL || field == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(field, 0, sizeof(*field));
  status = ss_field_space_summary(space, &summary);
  if (status == SS_OK) {
    status = ss_field_space_owned_free_dof_map(space, NULL, NULL, 0U, &count);
  }
  if (status == SS_OK && (summary.owned_free_dof_count != (uint64_t)count ||
                          summary.global_free_dof_count == 0U ||
                          VecGetLocalSize(vector, &local_size) != PETSC_SUCCESS ||
                          local_size < 0 || (size_t)local_size != count)) {
    status = SS_ERROR_INTEGRITY;
  }
  if (status == SS_OK && count != 0U) {
    field->stable_ids = (SsEntityId *)calloc(count, sizeof(*field->stable_ids));
    field->local_slots = (size_t *)calloc(count, sizeof(*field->local_slots));
    field->values = (double *)calloc(count, sizeof(*field->values));
    if (field->stable_ids == NULL || field->local_slots == NULL ||
        field->values == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(PetscObjectComm((PetscObject)vector), status);
  if (status == SS_OK) {
    status = ss_field_space_owned_free_dof_map(
        space, field->stable_ids, field->local_slots, count, &field->local_count);
  }
  if (status == SS_OK && capture != 0 &&
      VecGetArrayRead(vector, &source) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  for (index = 0U; status == SS_OK && capture != 0 && index < count; ++index) {
    const size_t slot = field->local_slots[index];
    if (slot >= count) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    field->values[index] = (double)PetscRealPart(source[slot]);
    if (!isfinite(field->values[index])) {
      status = SS_ERROR_INTEGRITY;
    }
  }
  if (source != NULL && VecRestoreArrayRead(vector, &source) != PETSC_SUCCESS &&
      status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    field->global_count = summary.global_free_dof_count;
    if (snprintf(field->field_space_id, sizeof(field->field_space_id), "%s",
                 summary.identity_digest) < 0) {
      status = SS_ERROR_LIMIT;
    }
  }
  status = ss_status_allreduce_max(PetscObjectComm((PetscObject)vector), status);
  if (status != SS_OK) {
    ss_solver_checkpoint_field_buffer_destroy(field);
  }
  return status;
}

SsStatus
ss_solver_checkpoint_field_copy_to_vector(const SsSolverCheckpointFieldBuffer *field,
                                          Vec vector) {
  PetscScalar *destination = NULL;
  PetscInt local_size = 0;
  size_t index;
  SsStatus status = SS_OK;
  if (field == NULL || vector == NULL ||
      VecGetLocalSize(vector, &local_size) != PETSC_SUCCESS || local_size < 0 ||
      (size_t)local_size != field->local_count ||
      VecGetArray(vector, &destination) != PETSC_SUCCESS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (index = 0U; index < field->local_count; ++index) {
    const size_t slot = field->local_slots[index];
    if (slot >= field->local_count || !isfinite(field->values[index])) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    destination[slot] = (PetscScalar)field->values[index];
  }
  if (VecRestoreArray(vector, &destination) != PETSC_SUCCESS && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  return ss_status_allreduce_max(PetscObjectComm((PetscObject)vector), status);
}

const char *ss_solver_checkpoint_material_model(const SsSolverGraph *graph) {
  return graph->model.mechanics_enabled != 0 ? "mohr-coulomb-v1" : "none-v1";
}

const char *ss_solver_checkpoint_hydraulic_model(const SsSolverGraph *graph) {
  return graph->model.hydraulic_enabled != 0 ? "steady-unconfined-v1" : "none-v1";
}

const char *ss_solver_checkpoint_coupling_model(const SsSolverGraph *graph) {
  if (graph->model.coupling_enabled == 0) {
    return "none-v1";
  }
  return graph->model.coupling_mode == SS_COUPLING_WEIGHT_ONLY
             ? "one-way-weight-only-v1"
             : "one-way-pressure-gradient-and-weight-v1";
}

const char *ss_solver_checkpoint_controller(const SsSolverGraph *graph) {
  static char indirect_identity[SS_RESULT_TEXT_CAPACITY];
  PetscInt indirect_growth_iteration_limit = 0;
  PetscInt work_efficiency_iteration_maximum = 0;
  PetscInt guard_iteration = 0;
  PetscReal guard_maximum_convergence = 0.0;
  PetscReal indirect_sensitivity_relative_step = cbrt(DBL_EPSILON);
  PetscReal predictor_error_target = 0.0;
  PetscReal work_step_scale_minimum = 0.0;
  PetscReal work_step_scale_maximum = 0.0;
  SsNonlinearControlSensitivityMethod indirect_sensitivity_method =
      SS_NONLINEAR_CONTROL_SENSITIVITY_REFERENCE_FORWARD;
  SsNonlinearProgressGuardMode guard_mode = SS_NONLINEAR_PROGRESS_GUARD_OFF;
  SsAnalysisWorkStepController work_step_controller = SS_ANALYSIS_WORK_STEP_CLASSIC;
  if (strcmp(graph->model.result_analysis_method, "steady_seepage") == 0) {
    return "steady-seepage-controller-v1";
  }
  switch (graph->model.analysis_method) {
  case SS_ANALYSIS_LIMIT_LOAD:
    return "limit-load-controller-v1";
  case SS_ANALYSIS_SSR_DIRECT:
    return "direct-ssr-controller-v1";
  case SS_ANALYSIS_SSR_INDIRECT:
    ss_solver_indirect_work_step_options(
        &graph->policy, &work_step_controller, &indirect_growth_iteration_limit,
        &work_efficiency_iteration_maximum, &predictor_error_target,
        &work_step_scale_minimum, &work_step_scale_maximum);
    ss_solver_indirect_sensitivity_options(&graph->policy.settings,
                                           &indirect_sensitivity_method,
                                           &indirect_sensitivity_relative_step);
    if (ss_solver_indirect_progress_guard_options(
            &graph->policy, &guard_mode, &guard_iteration,
            &guard_maximum_convergence) != SS_OK) {
      return "indirect-ssr-invalid-controller";
    }
    if (work_step_controller == SS_ANALYSIS_WORK_STEP_PREDICTOR_ERROR ||
        work_step_controller == SS_ANALYSIS_WORK_STEP_WORK_EFFICIENCY) {
      char descriptor[384];
      char digest[SS_SHA256_HEX_CAPACITY];
      int written =
          work_step_controller == SS_ANALYSIS_WORK_STEP_WORK_EFFICIENCY
              ? snprintf(
                    descriptor, sizeof(descriptor),
                    "we-v1;n=%lld;nmax=%lld;e=%a;smin=%a;smax=%a;sensitivity=%d;"
                    "h=%a;guard=%d;checkpoint=%lld;guard_max=%a",
                    (long long)indirect_growth_iteration_limit,
                    (long long)work_efficiency_iteration_maximum,
                    (double)predictor_error_target, (double)work_step_scale_minimum,
                    (double)work_step_scale_maximum, (int)indirect_sensitivity_method,
                    (double)indirect_sensitivity_relative_step, (int)guard_mode,
                    (long long)guard_iteration, (double)guard_maximum_convergence)
              : snprintf(
                    descriptor, sizeof(descriptor),
                    "wpe-v2;n=%lld;e=%a;smin=%a;smax=%a;sensitivity=%d;h=%a;"
                    "guard=%d;checkpoint=%lld;guard_max=%a",
                    (long long)indirect_growth_iteration_limit,
                    (double)predictor_error_target, (double)work_step_scale_minimum,
                    (double)work_step_scale_maximum, (int)indirect_sensitivity_method,
                    (double)indirect_sensitivity_relative_step, (int)guard_mode,
                    (long long)guard_iteration, (double)guard_maximum_convergence);
      if (written <= 0 || (size_t)written >= sizeof(descriptor) ||
          ss_sha256_bytes((const unsigned char *)descriptor, (size_t)written, digest) !=
              SS_OK) {
        return "indirect-ssr-invalid-controller";
      }
      written = snprintf(indirect_identity, sizeof(indirect_identity),
                         work_step_controller == SS_ANALYSIS_WORK_STEP_WORK_EFFICIENCY
                             ? "indirect-ssr-we-%s-v1"
                             : "indirect-ssr-wpe-%s-v2",
                         digest);
      return written > 0 && (size_t)written < sizeof(indirect_identity)
                 ? indirect_identity
                 : "indirect-ssr-invalid-controller";
    }
    if (guard_mode != SS_NONLINEAR_PROGRESS_GUARD_OFF) {
      const char growth = indirect_growth_iteration_limit > 0 ? 'e' : 'c';
      int written;
      if (guard_mode == SS_NONLINEAR_PROGRESS_GUARD_FIXED) {
        written = indirect_sensitivity_method ==
                          SS_NONLINEAR_CONTROL_SENSITIVITY_STABLE_CENTRAL
                      ? snprintf(indirect_identity, sizeof(indirect_identity),
                                 "indirect-ssr-%c-sc-%a-gf-k%lld-z%a-v2", growth,
                                 (double)indirect_sensitivity_relative_step,
                                 (long long)guard_iteration,
                                 (double)guard_maximum_convergence)
                      : snprintf(indirect_identity, sizeof(indirect_identity),
                                 "indirect-ssr-%c-rf-gf-k%lld-z%a-v2", growth,
                                 (long long)guard_iteration,
                                 (double)guard_maximum_convergence);
      } else {
        const char *guard_tag =
            guard_mode == SS_NONLINEAR_PROGRESS_GUARD_ADAPTIVE ? "gad" : "go";
        written = indirect_sensitivity_method ==
                          SS_NONLINEAR_CONTROL_SENSITIVITY_STABLE_CENTRAL
                      ? snprintf(indirect_identity, sizeof(indirect_identity),
                                 "indirect-ssr-%c-sc-%a-%s-k%lld-v5", growth,
                                 (double)indirect_sensitivity_relative_step, guard_tag,
                                 (long long)guard_iteration)
                      : snprintf(indirect_identity, sizeof(indirect_identity),
                                 "indirect-ssr-%c-rf-%s-k%lld-v5", growth, guard_tag,
                                 (long long)guard_iteration);
      }
      return written > 0 && (size_t)written < sizeof(indirect_identity)
                 ? indirect_identity
                 : "indirect-ssr-invalid-controller";
    }
    if (indirect_sensitivity_method ==
        SS_NONLINEAR_CONTROL_SENSITIVITY_STABLE_CENTRAL) {
      const int written =
          snprintf(indirect_identity, sizeof(indirect_identity),
                   "indirect-ssr-%s-stable-central-%a-v1",
                   indirect_growth_iteration_limit > 0 ? "effort-limited" : "classic",
                   (double)indirect_sensitivity_relative_step);
      return written > 0 && (size_t)written < sizeof(indirect_identity)
                 ? indirect_identity
                 : "indirect-ssr-invalid-controller";
    }
    return indirect_growth_iteration_limit > 0
               ? "indirect-ssr-effort-limited-controller-v1"
               : "indirect-ssr-controller-v1";
  default:
    return "steady-seepage-controller-v1";
  }
}

const char *ss_solver_checkpoint_operator(const SsSolverGraph *graph) {
  return graph->model.analysis_identity[0] == '\0' ? "native-operator-v1"
                                                   : graph->model.analysis_identity;
}

SsStatus ss_solver_checkpoint_field_space_id(const SsSolverGraph *graph,
                                             const SsSolverCheckpointBuffer *buffer,
                                             char output[SS_SHA256_HEX_CAPACITY + 8U]) {
  char projection[2U * (SS_SHA256_HEX_CAPACITY + 8U) + 2U];
  char digest[SS_SHA256_HEX_CAPACITY];
  int written;
  if (graph == NULL || buffer == NULL || output == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (graph->model.mechanics_enabled == 0) {
    written = snprintf(output, SS_SHA256_HEX_CAPACITY + 8U, "%s",
                       buffer->pressure.field_space_id);
    return written < 0 || (size_t)written >= SS_SHA256_HEX_CAPACITY + 8U
               ? SS_ERROR_OVERFLOW
               : SS_OK;
  }
  if (graph->model.hydraulic_enabled == 0) {
    written = snprintf(output, SS_SHA256_HEX_CAPACITY + 8U, "%s",
                       buffer->displacement.field_space_id);
    return written < 0 || (size_t)written >= SS_SHA256_HEX_CAPACITY + 8U
               ? SS_ERROR_OVERFLOW
               : SS_OK;
  }
  written =
      snprintf(projection, sizeof(projection), "%s|%s",
               buffer->displacement.field_space_id, buffer->pressure.field_space_id);
  if (written < 0 || (size_t)written >= sizeof(projection) ||
      ss_sha256_bytes((const unsigned char *)projection, (size_t)written, digest) !=
          SS_OK) {
    return SS_ERROR_OVERFLOW;
  }
  written = snprintf(output, SS_SHA256_HEX_CAPACITY + 8U, "sha256:%s", digest);
  return written < 0 || (size_t)written >= SS_SHA256_HEX_CAPACITY + 8U
             ? SS_ERROR_OVERFLOW
             : SS_OK;
}

SsStatus ss_solver_checkpoint_deflation_capture(
    const SsSolverMechanics *mechanics,
    const SsSolverCheckpointFieldBuffer *displacement,
    SsSolverCheckpointDeflationBuffer *buffer) {
  size_t basis_count = 0U;
  size_t value_count = 0U;
  size_t value_bytes;
  size_t basis;
  SsStatus status = SS_OK;
  if (mechanics == NULL || displacement == NULL || buffer == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(buffer, 0, sizeof(*buffer));
  if (mechanics->deflation == NULL) {
    return SS_OK;
  }
  status = ss_deflation_persistent_count(mechanics->deflation, &basis_count);
  if (status == SS_OK &&
      (basis_count > SS_RESULT_COMPONENT_CAPACITY || basis_count > UINT32_MAX ||
       ss_checked_mul_size(displacement->local_count, basis_count, &value_count) !=
           SS_OK ||
       ss_checked_mul_size(value_count, sizeof(*buffer->left_values), &value_bytes) !=
           SS_OK)) {
    status = SS_ERROR_OVERFLOW;
  }
  if (status == SS_OK && basis_count != 0U) {
    buffer->left_values = (double *)calloc(value_count == 0U ? 1U : value_count,
                                           sizeof(*buffer->left_values));
    buffer->right_values = (double *)calloc(value_count == 0U ? 1U : value_count,
                                            sizeof(*buffer->right_values));
    buffer->source_values = (double *)calloc(value_count == 0U ? 1U : value_count,
                                             sizeof(*buffer->source_values));
    buffer->source_weights =
        (double *)calloc(basis_count, sizeof(*buffer->source_weights));
    buffer->source_kinds = (double *)calloc(basis_count, sizeof(*buffer->source_kinds));
    if (buffer->left_values == NULL || buffer->right_values == NULL ||
        buffer->source_values == NULL || buffer->source_weights == NULL ||
        buffer->source_kinds == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(mechanics->graph->communicator, status);
  for (basis = 0U; status == SS_OK && basis < basis_count; ++basis) {
    Vec left = NULL;
    Vec right = NULL;
    Vec source = NULL;
    const PetscScalar *left_values = NULL;
    const PetscScalar *right_values = NULL;
    const PetscScalar *source_values = NULL;
    PetscReal source_weight = 0.0;
    SsDeflationPersistentSourceKind source_kind = SS_DEFLATION_PERSISTENT_SOURCE_OTHER;
    PetscInt left_size = 0;
    PetscInt right_size = 0;
    PetscInt source_size = 0;
    size_t local;
    status =
        ss_deflation_persistent_state_entry(mechanics->deflation, basis, &left, &right,
                                            &source, &source_weight, &source_kind);
    if (status == SS_OK &&
        (VecGetLocalSize(left, &left_size) != PETSC_SUCCESS ||
         VecGetLocalSize(right, &right_size) != PETSC_SUCCESS || left_size < 0 ||
         VecGetLocalSize(source, &source_size) != PETSC_SUCCESS ||
         right_size != left_size || source_size != left_size ||
         (size_t)left_size != displacement->local_count ||
         VecGetArrayRead(left, &left_values) != PETSC_SUCCESS ||
         VecGetArrayRead(right, &right_values) != PETSC_SUCCESS ||
         VecGetArrayRead(source, &source_values) != PETSC_SUCCESS ||
         !isfinite((double)source_weight) || source_weight <= 0.0)) {
      status = SS_ERROR_INTEGRITY;
    }
    if (status == SS_OK) {
      buffer->source_weights[basis] = (double)source_weight;
      buffer->source_kinds[basis] = (double)source_kind;
    }
    for (local = 0U; status == SS_OK && local < displacement->local_count; ++local) {
      const size_t offset = local * basis_count + basis;
      buffer->left_values[offset] = (double)PetscRealPart(left_values[local]);
      buffer->right_values[offset] = (double)PetscRealPart(right_values[local]);
      buffer->source_values[offset] = (double)PetscRealPart(source_values[local]);
      if (!isfinite(buffer->left_values[offset]) ||
          !isfinite(buffer->right_values[offset]) ||
          !isfinite(buffer->source_values[offset])) {
        status = SS_ERROR_INTEGRITY;
      }
    }
    if (source_values != NULL &&
        VecRestoreArrayRead(source, &source_values) != PETSC_SUCCESS &&
        status == SS_OK) {
      status = SS_ERROR_INTERNAL;
    }
    if (right_values != NULL &&
        VecRestoreArrayRead(right, &right_values) != PETSC_SUCCESS && status == SS_OK) {
      status = SS_ERROR_INTERNAL;
    }
    if (left_values != NULL &&
        VecRestoreArrayRead(left, &left_values) != PETSC_SUCCESS && status == SS_OK) {
      status = SS_ERROR_INTERNAL;
    }
  }
  status = ss_status_allreduce_max(mechanics->graph->communicator, status);
  if (status == SS_OK) {
    buffer->local_count = basis_count == 0U ? 0U : displacement->local_count;
    buffer->global_count = basis_count == 0U ? 0U : displacement->global_count;
    buffer->basis_count = (uint32_t)basis_count;
    buffer->basis_capacity = (uint32_t)basis_count;
  } else {
    free(buffer->source_kinds);
    free(buffer->source_weights);
    free(buffer->source_values);
    free(buffer->right_values);
    free(buffer->left_values);
    memset(buffer, 0, sizeof(*buffer));
  }
  return status;
}

SsStatus ss_solver_checkpoint_deflation_restore_buffer(
    const SsSolverGraph *graph, const SsSolverCheckpointFieldBuffer *displacement,
    SsSolverCheckpointDeflationBuffer *buffer) {
  size_t capacity;
  size_t value_count;
  size_t value_bytes;
  SsStatus status = SS_OK;
  if (graph == NULL || displacement == NULL || buffer == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(buffer, 0, sizeof(*buffer));
  capacity = graph->policy.deflation_enabled != 0
                 ? graph->policy.deflation_maximum_dimension
                 : 0U;
  if (capacity > SS_RESULT_COMPONENT_CAPACITY || capacity > UINT32_MAX ||
      ss_checked_mul_size(displacement->local_count, capacity, &value_count) != SS_OK ||
      ss_checked_mul_size(value_count, sizeof(*buffer->left_values), &value_bytes) !=
          SS_OK) {
    status = SS_ERROR_OVERFLOW;
  }
  if (status == SS_OK && capacity != 0U) {
    buffer->left_values = (double *)calloc(value_count == 0U ? 1U : value_count,
                                           sizeof(*buffer->left_values));
    buffer->right_values = (double *)calloc(value_count == 0U ? 1U : value_count,
                                            sizeof(*buffer->right_values));
    buffer->source_values = (double *)calloc(value_count == 0U ? 1U : value_count,
                                             sizeof(*buffer->source_values));
    buffer->source_weights =
        (double *)calloc(capacity, sizeof(*buffer->source_weights));
    buffer->source_kinds = (double *)calloc(capacity, sizeof(*buffer->source_kinds));
    if (buffer->left_values == NULL || buffer->right_values == NULL ||
        buffer->source_values == NULL || buffer->source_weights == NULL ||
        buffer->source_kinds == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    buffer->local_count = capacity == 0U ? 0U : displacement->local_count;
    buffer->global_count = capacity == 0U ? 0U : displacement->global_count;
    buffer->basis_capacity = (uint32_t)capacity;
  } else {
    free(buffer->source_kinds);
    free(buffer->source_weights);
    free(buffer->source_values);
    free(buffer->right_values);
    free(buffer->left_values);
    memset(buffer, 0, sizeof(*buffer));
  }
  return status;
}

SsStatus ss_solver_checkpoint_deflation_apply(SsSolverMechanics *mechanics,
                                              Vec template_vector,
                                              SsSolverCheckpointDeflationBuffer *buffer,
                                              uint32_t basis_count, PetscReal factor,
                                              uint64_t accepted_generation) {
  Vec left[SS_RESULT_COMPONENT_CAPACITY] = {NULL};
  Vec right[SS_RESULT_COMPONENT_CAPACITY] = {NULL};
  Vec source[SS_RESULT_COMPONENT_CAPACITY] = {NULL};
  PetscReal source_weights[SS_RESULT_COMPONENT_CAPACITY] = {0.0};
  SsDeflationPersistentSourceKind source_kinds[SS_RESULT_COMPONENT_CAPACITY] = {
      SS_DEFLATION_PERSISTENT_SOURCE_OTHER};
  Vec residual = NULL;
  Mat tangent = NULL;
  int tangent_owned = 0;
  uint32_t basis;
  SsStatus status = SS_OK;
  if (mechanics == NULL || template_vector == NULL || buffer == NULL ||
      basis_count > buffer->basis_capacity ||
      (basis_count != 0U && mechanics->deflation == NULL)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (basis = 0U; status == SS_OK && basis < basis_count; ++basis) {
    PetscScalar *left_values = NULL;
    PetscScalar *right_values = NULL;
    PetscScalar *source_values = NULL;
    size_t local;
    if (VecDuplicate(template_vector, &left[basis]) != PETSC_SUCCESS ||
        VecDuplicate(template_vector, &right[basis]) != PETSC_SUCCESS ||
        VecDuplicate(template_vector, &source[basis]) != PETSC_SUCCESS ||
        VecGetArray(left[basis], &left_values) != PETSC_SUCCESS ||
        VecGetArray(right[basis], &right_values) != PETSC_SUCCESS ||
        VecGetArray(source[basis], &source_values) != PETSC_SUCCESS) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
    if (status == SS_OK &&
        (!isfinite(buffer->source_weights[basis]) ||
         buffer->source_weights[basis] <= 0.0 ||
         floor(buffer->source_kinds[basis]) != buffer->source_kinds[basis] ||
         buffer->source_kinds[basis] < (double)SS_DEFLATION_PERSISTENT_SOURCE_OTHER ||
         buffer->source_kinds[basis] >
             (double)SS_DEFLATION_PERSISTENT_SOURCE_SNAPSHOT_POD)) {
      status = SS_ERROR_INTEGRITY;
    }
    if (status == SS_OK) {
      source_weights[basis] = (PetscReal)buffer->source_weights[basis];
      source_kinds[basis] =
          (SsDeflationPersistentSourceKind)(int)buffer->source_kinds[basis];
    }
    for (local = 0U; status == SS_OK && local < buffer->local_count; ++local) {
      const size_t offset = local * basis_count + basis;
      left_values[local] = (PetscScalar)buffer->left_values[offset];
      right_values[local] = (PetscScalar)buffer->right_values[offset];
      source_values[local] = (PetscScalar)buffer->source_values[offset];
    }
    if (source_values != NULL &&
        VecRestoreArray(source[basis], &source_values) != PETSC_SUCCESS &&
        status == SS_OK) {
      status = SS_ERROR_INTERNAL;
    }
    if (right_values != NULL &&
        VecRestoreArray(right[basis], &right_values) != PETSC_SUCCESS &&
        status == SS_OK) {
      status = SS_ERROR_INTERNAL;
    }
    if (left_values != NULL &&
        VecRestoreArray(left[basis], &left_values) != PETSC_SUCCESS &&
        status == SS_OK) {
      status = SS_ERROR_INTERNAL;
    }
  }
  status = ss_status_allreduce_max(mechanics->graph->communicator, status);
  if (status == SS_OK && basis_count != 0U && mechanics->matrix_free != 0) {
    const size_t finest = mechanics->matrix_free_level_count == 0U
                              ? 0U
                              : mechanics->matrix_free_level_count - 1U;
    status = mechanics->graph->model.analysis_method == SS_ANALYSIS_LIMIT_LOAD
                 ? ss_solver_mechanics_matrix_free_linearize_common(
                       mechanics, template_vector, 1.0, factor, 0.0)
                 : ss_solver_mechanics_matrix_free_linearize_common(
                       mechanics, template_vector, factor, 1.0, 0.0);
    if (status == SS_OK && (mechanics->matrix_free_level_count == 0U ||
                            mechanics->matrix_free_level_operators[finest] == NULL)) {
      status = SS_ERROR_INVALID_STATE;
    }
    if (status == SS_OK) {
      tangent = mechanics->matrix_free_level_operators[finest];
    }
  } else if (status == SS_OK && basis_count != 0U) {
    status =
        ss_assembly_plan_create_vector(mechanics->graph->mechanics_plan, &residual);
    if (status == SS_OK) {
      status =
          ss_assembly_plan_create_matrix(mechanics->graph->mechanics_plan, &tangent);
      tangent_owned = status == SS_OK;
    }
    if (status == SS_OK) {
      status =
          mechanics->graph->model.analysis_method == SS_ANALYSIS_LIMIT_LOAD
              ? ss_solver_mechanics_evaluate_common(mechanics, template_vector, 1.0,
                                                    factor, residual, tangent, NULL)
              : ss_solver_mechanics_evaluate_common(mechanics, template_vector, factor,
                                                    1.0, residual, tangent, NULL);
    }
  }
  if (status == SS_OK) {
    status = ss_deflation_restore_persistent_state(
        mechanics->deflation, left, right, source, source_weights, source_kinds,
        basis_count, tangent, accepted_generation, accepted_generation);
  }
  for (basis = 0U; basis < SS_RESULT_COMPONENT_CAPACITY; ++basis) {
    if (VecDestroy(&source[basis]) != PETSC_SUCCESS && status == SS_OK) {
      status = SS_ERROR_INTERNAL;
    }
    if (VecDestroy(&right[basis]) != PETSC_SUCCESS && status == SS_OK) {
      status = SS_ERROR_INTERNAL;
    }
    if (VecDestroy(&left[basis]) != PETSC_SUCCESS && status == SS_OK) {
      status = SS_ERROR_INTERNAL;
    }
  }
  if (tangent_owned != 0 && MatDestroy(&tangent) != PETSC_SUCCESS && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (VecDestroy(&residual) != PETSC_SUCCESS && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  return ss_status_allreduce_max(mechanics->graph->communicator, status);
}
