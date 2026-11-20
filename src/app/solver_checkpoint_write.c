/* Capture and publish accepted solver checkpoint states.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "solver_internal.h"

#include "slope_internal/core/sha256.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

int ss_solver_checkpoint_boundary_ready(const SsSolverGraph *graph,
                                        const SsAnalysisRecord *analysis_record) {
  return graph != NULL && analysis_record != NULL &&
         graph->policy.checkpoint_mode == SS_SOLVER_CHECKPOINT_WRITE_ACCEPTED &&
         analysis_record->point_count != 0U &&
         analysis_record->stop_reason == SS_ANALYSIS_STOP_ACCEPTED_STEP_LIMIT &&
         (analysis_record->next_operation == SS_ANALYSIS_NEXT_SECOND_SEED ||
          analysis_record->next_operation == SS_ANALYSIS_NEXT_CONTINUATION_ATTEMPT);
}

SsStatus ss_solver_checkpoint_write_state(
    const SsSolverRequest *request, const SsSolverGraph *graph,
    const SsSolverCheckpointTarget *target, Vec mechanics_solution,
    const SsAnalysisRecord *analysis_record, const SsSolverMechanics *mechanics,
    Vec hydraulic_solution, const SsSolverHydraulic *hydraulic,
    const SsNonlinearSolveRecord *hydraulic_solve,
    const SsSolverHydraulicEvidence *hydraulic_evidence, SsStateId state_id,
    SsCheckpointRestoreSummary *checkpoint_summary_out, SsErrorRecord *error) {
  static const char *const scalar_component[] = {"value"};
  SsSolverCheckpointBuffer buffer;
  SsCheckpointFieldView fields[3];
  SsCheckpointSpec spec;
  SsCheckpointRestoreSummary checkpoint_summary;
  const SsSolverBoundary *boundary = target == NULL ? NULL : target->boundary;
  size_t point_count = 0U;
  char field_space_id[SS_SHA256_HEX_CAPACITY + 8U];
  size_t field_count = 0U;
  size_t prior_count = 0U;
  SsStatus status = SS_OK;
  const char *stage = "argument_validation";
  if (request == NULL || graph == NULL || target == NULL || target->directory == NULL ||
      state_id == 0U || error == NULL ||
      (graph->model.mechanics_enabled != 0 &&
       (mechanics_solution == NULL || analysis_record == NULL || mechanics == NULL)) ||
      (graph->model.mechanics_enabled == 0 &&
       (mechanics_solution != NULL || analysis_record != NULL || mechanics != NULL)) ||
      (graph->model.hydraulic_enabled != 0 &&
       (hydraulic_solution == NULL || hydraulic == NULL)) ||
      (graph->model.hydraulic_enabled == 0 &&
       (hydraulic_solution != NULL || hydraulic != NULL || hydraulic_solve != NULL ||
        hydraulic_evidence != NULL))) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(&buffer, 0, sizeof(buffer));
  memset(fields, 0, sizeof(fields));
  memset(&spec, 0, sizeof(spec));
  memset(&checkpoint_summary, 0, sizeof(checkpoint_summary));
  memset(field_space_id, 0, sizeof(field_space_id));
  if (graph->model.mechanics_enabled != 0) {
    point_count =
        boundary != NULL ? boundary->point_count : analysis_record->point_count;
  }
  if (graph->model.mechanics_enabled != 0 &&
      (point_count == 0U || point_count > analysis_record->point_count)) {
    status = SS_ERROR_INTEGRITY;
  }
  if (status == SS_OK && graph->model.mechanics_enabled != 0) {
    PetscBool current_matches = PETSC_FALSE;
    stage = "mechanics_field_capture";
    if (mechanics->checkpoint_previous_state == NULL ||
        mechanics->checkpoint_current_state == NULL ||
        mechanics->checkpoint_state_count != point_count ||
        VecEqual(mechanics_solution, mechanics->checkpoint_current_state,
                 &current_matches) != PETSC_SUCCESS ||
        current_matches != PETSC_TRUE) {
      status = SS_ERROR_INTEGRITY;
    }
    if (status == SS_OK) {
      status = ss_solver_checkpoint_field_buffer_create(
          graph->mechanics_space, mechanics->checkpoint_current_state, 1,
          &buffer.displacement);
    }
    if (status == SS_OK) {
      status = ss_solver_checkpoint_field_buffer_create(
          graph->mechanics_space, mechanics->checkpoint_previous_state, 1,
          &buffer.previous_displacement);
    }
  }
  if (status == SS_OK && graph->model.hydraulic_enabled != 0) {
    stage = "hydraulic_field_capture";
    status = ss_solver_checkpoint_field_buffer_create(
        graph->hydraulic_space, hydraulic_solution, 1, &buffer.pressure);
  }
  if (status == SS_OK && graph->model.mechanics_enabled != 0) {
    stage = "deflation_capture";
    status = ss_solver_checkpoint_deflation_capture(mechanics, &buffer.displacement,
                                                    &buffer.deflation);
  }
  if (status == SS_OK) {
    stage = "field_space_identity";
    status = ss_solver_checkpoint_field_space_id(graph, &buffer, field_space_id);
  }
  if (status == SS_OK && graph->model.mechanics_enabled != 0) {
    stage = "mechanics_controller";
    const SsAnalysisPointRecord *terminal = &analysis_record->points[point_count - 1U];
    const PetscInt efficiency_iterations =
        boundary != NULL ? boundary->work_efficiency_previous_iterations
                         : analysis_record->work_efficiency_previous_iterations;
    const PetscReal efficiency_elasticity =
        boundary != NULL ? boundary->work_efficiency_elasticity
                         : analysis_record->work_efficiency_elasticity;
    const int efficiency_ready =
        boundary != NULL ? boundary->work_efficiency_elasticity_ready
                         : analysis_record->work_efficiency_elasticity_ready;
    size_t available;
    size_t start;
    size_t point;
    if (state_id != (SsStateId)point_count ||
        (boundary == NULL &&
         ss_solver_checkpoint_boundary_ready(graph, analysis_record) == 0) ||
        (boundary != NULL && boundary->next_operation != SS_ANALYSIS_NEXT_SECOND_SEED &&
         boundary->next_operation != SS_ANALYSIS_NEXT_CONTINUATION_ATTEMPT)) {
      status = SS_ERROR_INTEGRITY;
    } else {
      buffer.controller_values[0] =
          isfinite((double)terminal->factor) ? (double)terminal->factor : 0.0;
      buffer.controller_values[1] =
          isfinite((double)terminal->control) ? (double)terminal->control : 0.0;
      buffer.controller_values[2] =
          isfinite((double)terminal->work) ? (double)terminal->work : 0.0;
      buffer.controller_values[3] = isfinite((double)terminal->displacement_maximum)
                                        ? (double)terminal->displacement_maximum
                                        : 0.0;
      buffer.controller_values[4] = isfinite((double)terminal->signed_scaled_slope)
                                        ? (double)terminal->signed_scaled_slope
                                        : 0.0;
      /* An in-run boundary has not stopped: its stop reason is NONE. */
      buffer.controller_values[5] = boundary != NULL
                                        ? (double)SS_ANALYSIS_STOP_NONE
                                        : (double)analysis_record->stop_reason;
      buffer.controller_values[6] = boundary != NULL
                                        ? (double)boundary->next_operation
                                        : (double)analysis_record->next_operation;
      buffer.controller_values[7] = boundary != NULL
                                        ? (double)boundary->first_control
                                        : (double)analysis_record->first_control;
      if (strcmp(analysis_record->controller_identity,
                 "ssr_indirect/work_efficiency_adaptive") == 0) {
        if (efficiency_iterations < 0 ||
            (efficiency_ready != 0 && efficiency_ready != 1) ||
            !isfinite((double)efficiency_elasticity)) {
          status = SS_ERROR_INTEGRITY;
        } else {
          buffer.controller_values[SS_SOLVER_WORK_EFFICIENCY_ITERATIONS_INDEX] =
              efficiency_ready != 0 ? (double)efficiency_iterations
                                    : -(double)efficiency_iterations;
          buffer.controller_values[SS_SOLVER_WORK_EFFICIENCY_ELASTICITY_INDEX] =
              efficiency_ready != 0 ? (double)efficiency_elasticity : 0.0;
        }
      }
      if (ss_solver_progress_guard_learns(mechanics->progress_guard_mode)) {
        const SsNonlinearAdaptiveProgressGuard *guard =
            &mechanics->adaptive_progress_guard;
        if (guard->successful_sample_count > (uint64_t)SS_ANALYSIS_POINT_CAPACITY ||
            !isfinite((double)guard->maximum_successful_convergence) ||
            guard->maximum_successful_convergence < 0.0 ||
            !isfinite((double)guard->minimum_successful_convergence_ratio) ||
            !isfinite((double)guard->maximum_successful_convergence_ratio) ||
            guard->minimum_successful_convergence_ratio < 0.0 ||
            guard->maximum_successful_convergence_ratio <
                guard->minimum_successful_convergence_ratio ||
            (guard->successful_sample_count == 0U &&
             (guard->maximum_successful_convergence != 0.0 ||
              guard->minimum_successful_convergence_ratio != 0.0 ||
              guard->maximum_successful_convergence_ratio != 0.0)) ||
            guard->checkpoint_iteration < 0 ||
            guard->checkpoint_iteration >=
                mechanics->nonlinear_policy.maximum_iterations) {
          status = SS_ERROR_INTEGRITY;
        } else {
          buffer.controller_values[SS_SOLVER_ADAPTIVE_GUARD_SAMPLE_COUNT_INDEX] =
              (double)guard->successful_sample_count;
          buffer.controller_values[SS_SOLVER_ADAPTIVE_GUARD_MAXIMUM_INDEX] =
              (double)guard->maximum_successful_convergence;
          buffer.controller_values[SS_SOLVER_ADAPTIVE_GUARD_VERSION_INDEX] =
              SS_SOLVER_ADAPTIVE_GUARD_CHECKPOINT_VERSION;
          buffer.controller_values[SS_SOLVER_ADAPTIVE_GUARD_CHECKPOINT_INDEX] =
              (double)guard->checkpoint_iteration;
          buffer.controller_values[SS_SOLVER_ADAPTIVE_GUARD_MINIMUM_RATIO_INDEX] =
              (double)guard->minimum_successful_convergence_ratio;
          buffer.controller_values[SS_SOLVER_ADAPTIVE_GUARD_MAXIMUM_RATIO_INDEX] =
              (double)guard->maximum_successful_convergence_ratio;
        }
      }
    }
    available = status == SS_OK && point_count > 1U ? point_count - 1U : 0U;
    start = available < SS_CHECKPOINT_PRIOR_STATE_CAPACITY
                ? 0U
                : available - SS_CHECKPOINT_PRIOR_STATE_CAPACITY;
    prior_count = 0U;
    for (point = start; status == SS_OK && point < available; ++point) {
      const SsAnalysisPointRecord *source = &analysis_record->points[point];
      SsCheckpointPriorState *prior;
      if (!isfinite((double)source->factor) || !isfinite((double)source->work) ||
          !isfinite((double)source->control)) {
        continue;
      }
      prior = &buffer.prior_states[prior_count++];
      prior->state_id = (SsStateId)point + 1U;
      prior->parent_state_id = point == 0U ? 0U : (SsStateId)point;
      prior->accepted_ordinal = (uint64_t)source->accepted_ordinal;
      prior->factor = (double)source->factor;
      prior->work = (double)source->work;
      prior->direct_control = (double)source->control;
    }
  } else if (status == SS_OK) {
    stage = "hydraulic_controller";
    buffer.controller_values[0] = 1.0;
    buffer.controller_values[1] = 1.0;
    buffer.controller_values[2] = 0.0;
    buffer.controller_values[3] = 0.0;
    buffer.controller_values[4] = 0.0;
    buffer.controller_values[5] = 0.0;
    buffer.controller_values[6] = (double)SS_ANALYSIS_NEXT_TERMINAL;
    buffer.controller_values[7] = 1.0;
  }
  if (status == SS_OK && graph->model.hydraulic_enabled != 0) {
    stage = "hydraulic_state";
    if (hydraulic_evidence != NULL && hydraulic_evidence->valid != 0) {
      buffer.coupling_values[0] = (double)hydraulic_evidence->initial_residual_norm;
      buffer.coupling_values[1] = (double)hydraulic_evidence->final_residual_norm;
      buffer.coupling_values[2] = (double)hydraulic_evidence->load_norm;
      buffer.coupling_values[3] =
          hydraulic_evidence->initialization.boundary_residual_norm;
      buffer.coupling_values[4] =
          hydraulic_evidence->initialization.saturated_residual_norm;
      buffer.coupling_values[5] =
          (double)hydraulic_evidence->initialization.primary_choice;
      buffer.coupling_values[6] = (double)hydraulic_evidence->initialization.choice;
      buffer.coupling_values[7] =
          (double)hydraulic_evidence->initialization.fallback_attempted;
      buffer.coupling_values[8] = (double)hydraulic_evidence->nonlinear_outcome;
    } else if (hydraulic_solve != NULL) {
      buffer.coupling_values[0] = (double)hydraulic_solve->initial_residual_norm;
      buffer.coupling_values[1] = (double)hydraulic_solve->final_residual_norm;
      buffer.coupling_values[2] = (double)hydraulic_solve->load_norm;
      buffer.coupling_values[3] = hydraulic->initialization.boundary_residual_norm;
      buffer.coupling_values[4] = hydraulic->initialization.saturated_residual_norm;
      buffer.coupling_values[5] = (double)hydraulic->initialization.primary_choice;
      buffer.coupling_values[6] = (double)hydraulic->initialization.choice;
      buffer.coupling_values[7] = (double)hydraulic->initialization.fallback_attempted;
      buffer.coupling_values[8] = (double)hydraulic_solve->outcome;
    } else {
      status = SS_ERROR_INVALID_ARGUMENT;
    }
  }
  if (status == SS_OK) {
    size_t index;
    stage = "finite_state";
    for (index = 0U; status == SS_OK && index < SS_RESULT_CONTROLLER_CAPACITY;
         ++index) {
      if (!isfinite(buffer.controller_values[index])) {
        status = SS_ERROR_INTEGRITY;
      }
    }
    for (index = 0U; status == SS_OK && graph->model.hydraulic_enabled != 0 &&
                     index < SS_RESULT_COUPLING_CAPACITY;
         ++index) {
      if (!isfinite(buffer.coupling_values[index])) {
        status = SS_ERROR_INTEGRITY;
      }
    }
  }
  if (status == SS_OK && graph->model.mechanics_enabled != 0) {
    stage = "mechanics_field_view";
    fields[field_count++] = (SsCheckpointFieldView){
        .stable_ids = buffer.displacement.stable_ids,
        .local_slots = buffer.displacement.local_slots,
        .values = buffer.displacement.values,
        .local_count = buffer.displacement.local_count,
        .global_count = buffer.displacement.global_count,
        .component_count = 1U,
        .quantity_id = "displacement_free_dofs",
        .unit = "m",
        .space_id = "mechanics_free_dofs_v1",
        .basis = "scalar_dof",
        .component_ids = scalar_component,
    };
    fields[field_count++] = (SsCheckpointFieldView){
        .stable_ids = buffer.previous_displacement.stable_ids,
        .local_slots = buffer.previous_displacement.local_slots,
        .values = buffer.previous_displacement.values,
        .local_count = buffer.previous_displacement.local_count,
        .global_count = buffer.previous_displacement.global_count,
        .component_count = 1U,
        .quantity_id = "displacement_previous_free_dofs",
        .unit = "m",
        .space_id = "mechanics_free_dofs_v1",
        .basis = "scalar_dof",
        .component_ids = scalar_component,
    };
  }
  if (status == SS_OK && graph->model.hydraulic_enabled != 0) {
    stage = "hydraulic_field_view";
    fields[field_count++] = (SsCheckpointFieldView){
        .stable_ids = buffer.pressure.stable_ids,
        .local_slots = buffer.pressure.local_slots,
        .values = buffer.pressure.values,
        .local_count = buffer.pressure.local_count,
        .global_count = buffer.pressure.global_count,
        .component_count = 1U,
        .quantity_id = "pore_pressure_free_dofs",
        .unit = "kPa",
        .space_id = "hydraulic_free_dofs_v1",
        .basis = "scalar_dof",
        .component_ids = scalar_component,
    };
  }
  if (status == SS_OK) {
    stage = "checkpoint_atomic_write";
    spec = (SsCheckpointSpec){
        .final_directory = target->directory,
        .problem_id = request->config->problem.id,
        .run_id = request->config->run.id,
        .build_id = request->build_id,
        .build_compatibility_id = request->build_id,
        .mesh_id = graph->model.mesh_id,
        .geometry_id = graph->geometry_id,
        .field_space_id = field_space_id,
        .numerical_policy_id = graph->policy.numerical_policy_id,
        .material_model_id = ss_solver_checkpoint_material_model(graph),
        .hydraulic_model_id = ss_solver_checkpoint_hydraulic_model(graph),
        .coupling_model_id = ss_solver_checkpoint_coupling_model(graph),
        .analysis_method = graph->model.result_analysis_method,
        .controller_id = ss_solver_checkpoint_controller(graph),
        .field_order = graph->model.mechanics_enabled != 0
                           ? graph->model.mechanics_order
                           : graph->model.hydraulic_order,
        .geometry_order = graph->model.coordinate_order,
        .scalar_bits = 64U,
        .source_index_bits = (uint32_t)(sizeof(PetscInt) * CHAR_BIT),
        .accepted_state_id = state_id,
        .controller_values = buffer.controller_values,
        .controller_value_count = SS_RESULT_CONTROLLER_CAPACITY,
        .coupling_values =
            graph->model.hydraulic_enabled != 0 ? buffer.coupling_values : NULL,
        .coupling_value_count =
            graph->model.hydraulic_enabled != 0 ? SS_RESULT_COUPLING_CAPACITY : 0U,
        .prior_states = prior_count == 0U ? NULL : buffer.prior_states,
        .prior_state_count = prior_count,
        .history_cursor = target->history_cursor,
        .accepted_state_cursor =
            graph->model.mechanics_enabled != 0
                ? (uint64_t)analysis_record->points[point_count - 1U].accepted_ordinal +
                      1U
                : target->accepted_state_count,
        .retry_count = 0U,
        .next_step = graph->model.mechanics_enabled != 0
                         ? (double)(boundary != NULL ? boundary->next_step
                                                     : analysis_record->next_step)
                         : graph->policy.initial_step,
        .dataset_chunk_rows = 0U,
        .fields = fields,
        .field_count = field_count,
        .deflation =
            {
                .stable_free_dof_ids = buffer.displacement.stable_ids,
                .local_slots = buffer.displacement.local_slots,
                .left_values = buffer.deflation.left_values,
                .right_values = buffer.deflation.right_values,
                .source_values = buffer.deflation.source_values,
                .source_weights = buffer.deflation.source_weights,
                .source_kinds = buffer.deflation.source_kinds,
                .local_count = buffer.deflation.local_count,
                .global_count = buffer.deflation.global_count,
                .basis_count = buffer.deflation.basis_count,
                .provenance_id = "native-deflation-basis-v1",
                .operator_identity = ss_solver_checkpoint_operator(graph),
            },
        .communicator = graph->communicator,
    };
    status =
        ss_checkpoint_write_atomic(&spec, SS_RESULT_FAIL_NONE, &checkpoint_summary);
  }
  if (status == SS_OK && checkpoint_summary_out != NULL) {
    *checkpoint_summary_out = checkpoint_summary;
  }
  ss_solver_checkpoint_buffer_destroy(&buffer);
  if (status != SS_OK && error->valid == 0) {
    char message[SS_ERROR_MESSAGE_CAPACITY];
    const int written =
        snprintf(message, sizeof(message),
                 "accepted-boundary checkpoint write failed during %s (status=%d)",
                 stage, (int)status);
    ss_solver_error(error, status, "CHECKPOINT_WRITE",
                    written >= 0 && (size_t)written < sizeof(message)
                        ? message
                        : "accepted-boundary checkpoint write failed",
                    4U);
  }
  return ss_status_allreduce_max(graph->communicator, status);
}

SsStatus ss_solver_checkpoint_write_accepted(
    const SsSolverRequest *request, const SsSolverGraph *graph, SsResultWriter *writer,
    Vec mechanics_solution, const SsAnalysisRecord *analysis_record,
    const SsSolverMechanics *mechanics, Vec hydraulic_solution,
    const SsSolverHydraulic *hydraulic, const SsNonlinearSolveRecord *hydraulic_solve,
    const SsSolverHydraulicEvidence *hydraulic_evidence,
    const SsSolverBoundary *boundary, SsStateId state_id, SsErrorRecord *error) {
  SsResultWriterSummary writer_summary;
  SsCheckpointRestoreSummary checkpoint_summary;
  SsSolverCheckpointTarget target;
  char checkpoint_directory[SS_RESULT_PATH_CAPACITY];
  SsStatus status;
  if (writer == NULL || graph == NULL || error == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(&writer_summary, 0, sizeof(writer_summary));
  memset(&checkpoint_summary, 0, sizeof(checkpoint_summary));
  status = ss_result_writer_summary(writer, &writer_summary);
  if (status == SS_OK) {
    const int written =
        snprintf(checkpoint_directory, sizeof(checkpoint_directory),
                 "%s/checkpoint-accepted", writer_summary.staging_directory);
    if (written < 0 || (size_t)written >= sizeof(checkpoint_directory)) {
      status = SS_ERROR_OVERFLOW;
    }
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  target = (SsSolverCheckpointTarget){
      .directory = checkpoint_directory,
      .history_cursor = writer_summary.history_row_count,
      .accepted_state_count = writer_summary.accepted_state_count,
      .boundary = boundary,
  };
  status = ss_solver_checkpoint_write_state(
      request, graph, &target, mechanics_solution, analysis_record, mechanics,
      hydraulic_solution, hydraulic, hydraulic_solve, hydraulic_evidence, state_id,
      &checkpoint_summary, error);
  if (status == SS_OK) {
    const SsResultCheckpointReference reference = {
        .state_id = state_id,
        .checkpoint_id = checkpoint_summary.checkpoint_id,
        .relative_path = "checkpoint-accepted",
        .complete = 1,
    };
    status = ss_result_writer_index_checkpoint(writer, &reference);
  }
  return status;
}
