/* Restore accepted state into hydraulics, mechanics and continuation.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "solver_internal.h"

#include "slope_internal/core/sha256.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

SsStatus ss_solver_checkpoint_restore_accepted(const SsSolverRequest *request,
                                               SsSolverGraph *graph,
                                               SsSolverMonitor *monitor,
                                               SsSolverSummary *summary,
                                               SsErrorRecord *error) {
  SsSolverCheckpointBuffer buffer;
  SsCheckpointFieldRestoreView fields[3];
  SsCheckpointCompatibility compatibility;
  SsCheckpointRestoreView restore;
  SsSolverMechanics mechanics;
  SsSolverHydraulic hydraulic;
  SsSolverCouplingSnapshot *coupling_snapshot = NULL;
  SsCouplingDiagnostics coupling_diagnostics;
  SsSolverHydraulicEvidence hydraulic_evidence;
  SsNonlinearSolveRecord *hydraulic_solve = NULL;
  SsLinearSetupRecord elastic_setup;
  SsLinearSolveRecord elastic_solve;
  SsAnalysisProblem analysis_problem;
  SsAnalysisPolicy analysis_policy;
  SsAnalysisResumeState resume_state;
  SsAnalysisRecord *analysis_record = NULL;
  Vec mechanics_solution = NULL;
  Vec previous_mechanics_solution = NULL;
  Vec hydraulic_solution = NULL;
  Vec elastic_initial = NULL;
  PetscReal reference_work = NAN;
  char field_space_id[SS_SHA256_HEX_CAPACITY + 8U];
  size_t field_count = 0U;
  const int coupled = graph != NULL && graph->model.mechanics_enabled != 0 &&
                      graph->model.hydraulic_enabled != 0;
  SsStatus status = SS_OK;
  const char *stage = "record_allocation";
  if (request == NULL || graph == NULL || summary == NULL || error == NULL ||
      graph->policy.checkpoint_mode != SS_SOLVER_CHECKPOINT_RESTORE_ACCEPTED ||
      graph->policy.checkpoint_path[0] != '/') {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(&buffer, 0, sizeof(buffer));
  memset(fields, 0, sizeof(fields));
  memset(&compatibility, 0, sizeof(compatibility));
  memset(&restore, 0, sizeof(restore));
  memset(&mechanics, 0, sizeof(mechanics));
  memset(&hydraulic, 0, sizeof(hydraulic));
  memset(&coupling_diagnostics, 0, sizeof(coupling_diagnostics));
  memset(&hydraulic_evidence, 0, sizeof(hydraulic_evidence));
  memset(&elastic_setup, 0, sizeof(elastic_setup));
  memset(&elastic_solve, 0, sizeof(elastic_solve));
  memset(&analysis_problem, 0, sizeof(analysis_problem));
  memset(&analysis_policy, 0, sizeof(analysis_policy));
  memset(&resume_state, 0, sizeof(resume_state));
  memset(field_space_id, 0, sizeof(field_space_id));
  if (graph->model.mechanics_enabled != 0) {
    analysis_record = calloc(1U, sizeof(*analysis_record));
    if (analysis_record == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  if (graph->model.hydraulic_enabled != 0) {
    hydraulic_solve = calloc(1U, sizeof(*hydraulic_solve));
    if (hydraulic_solve == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    stage = "state_create";
  }
  if (status == SS_OK && graph->model.mechanics_enabled != 0) {
    status = ss_assembly_plan_create_vector(graph->mechanics_plan, &mechanics_solution);
    status = ss_status_allreduce_max(graph->communicator, status);
    if (status == SS_OK) {
      Vec *destinations[] = {&previous_mechanics_solution};
      status = ss_vec_duplicate_collective(mechanics_solution, destinations, 1U);
    }
  }
  if (status == SS_OK && graph->model.hydraulic_enabled != 0) {
    status = ss_assembly_plan_create_vector(graph->hydraulic_plan, &hydraulic_solution);
    status = ss_status_allreduce_max(graph->communicator, status);
  }
  if (status == SS_OK && graph->model.mechanics_enabled != 0) {
    stage = "mechanics_field_map";
    status = ss_solver_checkpoint_field_buffer_create(
        graph->mechanics_space, mechanics_solution, 0, &buffer.displacement);
  }
  if (status == SS_OK && graph->model.mechanics_enabled != 0) {
    stage = "mechanics_previous_field_map";
    status = ss_solver_checkpoint_field_buffer_create(graph->mechanics_space,
                                                      previous_mechanics_solution, 0,
                                                      &buffer.previous_displacement);
  }
  if (status == SS_OK && graph->model.hydraulic_enabled != 0) {
    stage = "hydraulic_field_map";
    status = ss_solver_checkpoint_field_buffer_create(
        graph->hydraulic_space, hydraulic_solution, 0, &buffer.pressure);
  }
  if (status == SS_OK && graph->model.mechanics_enabled != 0) {
    stage = "deflation_buffer";
    status = ss_solver_checkpoint_deflation_restore_buffer(graph, &buffer.displacement,
                                                           &buffer.deflation);
  }
  if (status == SS_OK) {
    status = ss_solver_checkpoint_field_space_id(graph, &buffer, field_space_id);
  }
  if (status == SS_OK && graph->model.mechanics_enabled != 0) {
    fields[field_count++] = (SsCheckpointFieldRestoreView){
        .quantity_id = "displacement_free_dofs",
        .stable_ids = buffer.displacement.stable_ids,
        .local_slots = buffer.displacement.local_slots,
        .values = buffer.displacement.values,
        .local_count = buffer.displacement.local_count,
        .component_count = 1U,
    };
    fields[field_count++] = (SsCheckpointFieldRestoreView){
        .quantity_id = "displacement_previous_free_dofs",
        .stable_ids = buffer.previous_displacement.stable_ids,
        .local_slots = buffer.previous_displacement.local_slots,
        .values = buffer.previous_displacement.values,
        .local_count = buffer.previous_displacement.local_count,
        .component_count = 1U,
    };
  }
  if (status == SS_OK && graph->model.hydraulic_enabled != 0) {
    fields[field_count++] = (SsCheckpointFieldRestoreView){
        .quantity_id = "pore_pressure_free_dofs",
        .stable_ids = buffer.pressure.stable_ids,
        .local_slots = buffer.pressure.local_slots,
        .values = buffer.pressure.values,
        .local_count = buffer.pressure.local_count,
        .component_count = 1U,
    };
  }
  if (status == SS_OK) {
    compatibility = (SsCheckpointCompatibility){
        .problem_id = request->config->problem.id,
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
        .operator_identity = ss_solver_checkpoint_operator(graph),
        .field_order = graph->model.mechanics_enabled != 0
                           ? graph->model.mechanics_order
                           : graph->model.hydraulic_order,
        .geometry_order = graph->model.coordinate_order,
        .scalar_bits = 64U,
        .target_index_bits = (uint32_t)(sizeof(PetscInt) * CHAR_BIT),
    };
    restore = (SsCheckpointRestoreView){
        .fields = fields,
        .field_count = field_count,
        .controller_values = buffer.controller_values,
        .controller_capacity = SS_RESULT_CONTROLLER_CAPACITY,
        .coupling_values = buffer.coupling_values,
        .coupling_capacity = SS_RESULT_COUPLING_CAPACITY,
        .prior_states = buffer.prior_states,
        .prior_state_capacity = SS_CHECKPOINT_PRIOR_STATE_CAPACITY,
        .deflation_stable_ids = buffer.displacement.stable_ids,
        .deflation_local_slots = buffer.displacement.local_slots,
        .deflation_local_count = buffer.deflation.local_count,
        .deflation_left_values = buffer.deflation.left_values,
        .deflation_right_values = buffer.deflation.right_values,
        .deflation_source_values = buffer.deflation.source_values,
        .deflation_source_weights = buffer.deflation.source_weights,
        .deflation_source_kinds = buffer.deflation.source_kinds,
        .deflation_basis_capacity = buffer.deflation.basis_capacity,
    };
    stage = "checkpoint_restore";
    status = ss_checkpoint_restore(graph->communicator, graph->policy.checkpoint_path,
                                   &compatibility, &restore, &buffer.summary);
  }
  {
    size_t index;
    if (status == SS_OK) {
      stage = "checkpoint_replicated_values";
    }
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
  if (status == SS_OK) {
    stage = "checkpoint_summary";
  }
  if (status == SS_OK &&
      (buffer.summary.mutation_started == 0 || buffer.summary.accepted_state_id == 0U ||
       buffer.summary.restored_field_count != (uint64_t)field_count ||
       (graph->model.mechanics_enabled != 0 &&
        (buffer.summary.accepted_state_cursor == 0U ||
         !isfinite(buffer.summary.next_step) || buffer.summary.next_step <= 0.0 ||
         (buffer.controller_values[6] != (double)SS_ANALYSIS_NEXT_SECOND_SEED &&
          buffer.controller_values[6] !=
              (double)SS_ANALYSIS_NEXT_CONTINUATION_ATTEMPT) ||
         (buffer.controller_values[6] ==
              (double)SS_ANALYSIS_NEXT_CONTINUATION_ATTEMPT &&
          buffer.summary.prior_state_count == 0U))) ||
       (graph->model.mechanics_enabled == 0 &&
        buffer.controller_values[6] != (double)SS_ANALYSIS_NEXT_TERMINAL))) {
    status = SS_ERROR_INTEGRITY;
  }
  if (status == SS_OK && graph->model.hydraulic_enabled != 0) {
    stage = "checkpoint_hydraulic_evidence";
    const double primary_choice_value = buffer.coupling_values[5];
    const double selected_choice_value = buffer.coupling_values[6];
    const double fallback_attempted_value = buffer.coupling_values[7];
    const double nonlinear_outcome_value = buffer.coupling_values[8];
    if ((primary_choice_value != (double)SS_HYDRAULIC_INITIAL_BOUNDARY_ENVELOPE &&
         primary_choice_value != (double)SS_HYDRAULIC_INITIAL_SATURATED_LINEAR) ||
        (selected_choice_value != (double)SS_HYDRAULIC_INITIAL_BOUNDARY_ENVELOPE &&
         selected_choice_value != (double)SS_HYDRAULIC_INITIAL_SATURATED_LINEAR) ||
        (fallback_attempted_value != 0.0 && fallback_attempted_value != 1.0) ||
        (nonlinear_outcome_value != (double)SS_NONLINEAR_CONVERGED &&
         nonlinear_outcome_value != (double)SS_NONLINEAR_ACCEPTED_RELAXED) ||
        buffer.coupling_values[2] <= 0.0) {
      status = SS_ERROR_INTEGRITY;
    }
  }
  if (status == SS_OK && graph->model.mechanics_enabled != 0) {
    stage = "checkpoint_mechanics_cursor";
    const uint64_t current_ordinal = buffer.summary.accepted_state_cursor - UINT64_C(1);
    const uint64_t accepted_step_offset =
        graph->model.analysis_method == SS_ANALYSIS_LIMIT_LOAD
            ? current_ordinal
            : current_ordinal + UINT64_C(1);
    const SsAnalysisNextOperation next_operation =
        (SsAnalysisNextOperation)(int)buffer.controller_values[6];
    const SsCheckpointPriorState *previous =
        next_operation == SS_ANALYSIS_NEXT_CONTINUATION_ATTEMPT
            ? &buffer.prior_states[buffer.summary.prior_state_count - 1U]
            : NULL;
    if (current_ordinal > (uint64_t)SIZE_MAX ||
        accepted_step_offset > (uint64_t)SIZE_MAX ||
        accepted_step_offset >= (uint64_t)graph->policy.maximum_accepted_states ||
        /* A step-limit stop or an in-run/cancellation boundary (no stop). */
        (buffer.controller_values[5] != (double)SS_ANALYSIS_STOP_ACCEPTED_STEP_LIMIT &&
         buffer.controller_values[5] != (double)SS_ANALYSIS_STOP_NONE) ||
        (previous != NULL &&
         previous->accepted_ordinal + UINT64_C(1) != current_ordinal) ||
        buffer.summary.retry_count > (uint64_t)PETSC_MAX_INT) {
      status = SS_ERROR_INTEGRITY;
    }
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    stage = "checkpoint_provenance";
    const int checkpoint_written =
        snprintf(graph->parent_checkpoint_id, sizeof(graph->parent_checkpoint_id), "%s",
                 buffer.summary.checkpoint_id);
    const int run_written = snprintf(graph->parent_run_id, sizeof(graph->parent_run_id),
                                     "%s", buffer.summary.source_run_id);
    const int build_written =
        snprintf(graph->parent_build_id, sizeof(graph->parent_build_id), "%s",
                 buffer.summary.source_build_id);
    if (checkpoint_written < 0 ||
        (size_t)checkpoint_written >= sizeof(graph->parent_checkpoint_id) ||
        run_written < 0 || (size_t)run_written >= sizeof(graph->parent_run_id) ||
        build_written < 0 || (size_t)build_written >= sizeof(graph->parent_build_id) ||
        (buffer.summary.source_index_bits != 32U &&
         buffer.summary.source_index_bits != 64U)) {
      status = SS_ERROR_INTEGRITY;
    } else {
      graph->parent_accepted_state_id = buffer.summary.accepted_state_id;
      graph->parent_source_index_bits = buffer.summary.source_index_bits;
      graph->restart_provenance_valid = 1;
    }
  }
  if (status == SS_OK && graph->model.mechanics_enabled != 0) {
    stage = "mechanics_field_restore";
    status = ss_solver_checkpoint_field_copy_to_vector(&buffer.displacement,
                                                       mechanics_solution);
  }
  if (status == SS_OK && graph->model.mechanics_enabled != 0) {
    stage = "mechanics_previous_field_restore";
    status = ss_solver_checkpoint_field_copy_to_vector(&buffer.previous_displacement,
                                                       previous_mechanics_solution);
  }
  if (status == SS_OK && graph->model.hydraulic_enabled != 0) {
    stage = "hydraulic_field_restore";
    status =
        ss_solver_checkpoint_field_copy_to_vector(&buffer.pressure, hydraulic_solution);
  }
  if (status == SS_OK && graph->model.hydraulic_enabled != 0) {
    const double primary_choice_value = buffer.coupling_values[5];
    const double selected_choice_value = buffer.coupling_values[6];
    const double fallback_attempted_value = buffer.coupling_values[7];
    const double nonlinear_outcome_value = buffer.coupling_values[8];
    const int primary_choice = (int)primary_choice_value;
    const int selected_choice = (int)selected_choice_value;
    const int fallback_attempted = (int)fallback_attempted_value;
    const SsNonlinearOutcome nonlinear_outcome =
        (SsNonlinearOutcome)(int)nonlinear_outcome_value;
    stage = "hydraulic_create";
    status = ss_solver_hydraulic_create(graph, request->requested_threads, &hydraulic,
                                        error);
    if (status == SS_OK) {
      ss_solver_monitor_watch_hydraulic(monitor, &hydraulic);
      hydraulic.initialization.primary_choice =
          (SsHydraulicInitializationChoice)primary_choice;
      hydraulic.initialization.choice =
          (SsHydraulicInitializationChoice)selected_choice;
      hydraulic.initialization.boundary_residual_norm = buffer.coupling_values[3];
      hydraulic.initialization.saturated_residual_norm = buffer.coupling_values[4];
      hydraulic.initialization.saturated_candidate_valid =
          primary_choice == SS_HYDRAULIC_INITIAL_SATURATED_LINEAR ||
          selected_choice == SS_HYDRAULIC_INITIAL_SATURATED_LINEAR;
      hydraulic.initialization.fallback_attempted = fallback_attempted;
      hydraulic.primary_outcome = SS_NONLINEAR_CONVERGED;
      hydraulic.primary_outcome_valid = 1;
      hydraulic_solve->outcome = nonlinear_outcome;
      hydraulic_solve->initial_residual_norm = (PetscReal)buffer.coupling_values[0];
      hydraulic_solve->final_residual_norm = (PetscReal)buffer.coupling_values[1];
      hydraulic_solve->load_norm = (PetscReal)buffer.coupling_values[2];
      stage = "hydraulic_acceptance";
      status =
          ss_solver_hydraulic_acceptance(graph, &hydraulic, hydraulic_solution,
                                         hydraulic_solve, &hydraulic_evidence, error);
    }
    if (status == SS_OK && coupled != 0) {
      stage = "coupling_snapshot";
      status = ss_solver_coupling_snapshot_build(
          graph, hydraulic_solution, request->requested_threads, &coupling_snapshot,
          &coupling_diagnostics, error);
    }
  }
  if (status == SS_OK && graph->model.mechanics_enabled != 0) {
    const size_t current_ordinal = (size_t)(buffer.summary.accepted_state_cursor - 1U);
    const SsAnalysisNextOperation next_operation =
        (SsAnalysisNextOperation)(int)buffer.controller_values[6];
    const SsCheckpointPriorState *previous =
        next_operation == SS_ANALYSIS_NEXT_CONTINUATION_ATTEMPT
            ? &buffer.prior_states[buffer.summary.prior_state_count - 1U]
            : NULL;
    if (status == SS_OK) {
      stage = "mechanics_create";
      status = ss_solver_mechanics_create(graph, request->requested_threads,
                                          coupling_snapshot, &mechanics, error);
    }
    if (status == SS_OK &&
        ss_solver_progress_guard_learns(mechanics.progress_guard_mode)) {
      const double sample_count =
          buffer.controller_values[SS_SOLVER_ADAPTIVE_GUARD_SAMPLE_COUNT_INDEX];
      const double maximum =
          buffer.controller_values[SS_SOLVER_ADAPTIVE_GUARD_MAXIMUM_INDEX];
      const double version =
          buffer.controller_values[SS_SOLVER_ADAPTIVE_GUARD_VERSION_INDEX];
      const double checkpoint =
          buffer.controller_values[SS_SOLVER_ADAPTIVE_GUARD_CHECKPOINT_INDEX];
      const double minimum_ratio =
          buffer.controller_values[SS_SOLVER_ADAPTIVE_GUARD_MINIMUM_RATIO_INDEX];
      const double maximum_ratio =
          buffer.controller_values[SS_SOLVER_ADAPTIVE_GUARD_MAXIMUM_RATIO_INDEX];
      stage = "adaptive_progress_guard_restore";
      if (sample_count < 0.0 || sample_count > (double)SS_ANALYSIS_POINT_CAPACITY ||
          floor(sample_count) != sample_count ||
          version != SS_SOLVER_ADAPTIVE_GUARD_CHECKPOINT_VERSION ||
          checkpoint !=
              (double)mechanics.adaptive_progress_guard.checkpoint_iteration ||
          !isfinite(maximum) || maximum < 0.0 || !isfinite(minimum_ratio) ||
          !isfinite(maximum_ratio) || minimum_ratio < 0.0 ||
          maximum_ratio < minimum_ratio) {
        status = SS_ERROR_INTEGRITY;
      } else {
        status = ss_nonlinear_adaptive_progress_guard_restore(
            mechanics.adaptive_progress_guard.checkpoint_iteration,
            (uint64_t)sample_count, (PetscReal)maximum, (PetscReal)minimum_ratio,
            (PetscReal)maximum_ratio, &mechanics.adaptive_progress_guard);
      }
    }
    if (status == SS_OK &&
        !ss_solver_progress_guard_learns(mechanics.progress_guard_mode) &&
        (buffer.controller_values[SS_SOLVER_ADAPTIVE_GUARD_SAMPLE_COUNT_INDEX] != 0.0 ||
         buffer.controller_values[SS_SOLVER_ADAPTIVE_GUARD_MAXIMUM_INDEX] != 0.0 ||
         buffer.controller_values[SS_SOLVER_ADAPTIVE_GUARD_VERSION_INDEX] != 0.0 ||
         buffer.controller_values[SS_SOLVER_ADAPTIVE_GUARD_CHECKPOINT_INDEX] != 0.0 ||
         buffer.controller_values[SS_SOLVER_ADAPTIVE_GUARD_MINIMUM_RATIO_INDEX] !=
             0.0 ||
         buffer.controller_values[SS_SOLVER_ADAPTIVE_GUARD_MAXIMUM_RATIO_INDEX] !=
             0.0)) {
      stage = "adaptive_progress_guard_absence";
      status = SS_ERROR_INTEGRITY;
    }
    if (status == SS_OK) {
      const PetscReal predictor_scale =
          graph->model.analysis_method == SS_ANALYSIS_LIMIT_LOAD
              ? (PetscReal)graph->policy.initial_work_scale
              : 1.0;
      stage = "mechanics_elastic_predictor";
      status = ss_assembly_plan_create_vector(graph->mechanics_plan, &elastic_initial);
      if (status == SS_OK && VecSet(elastic_initial, 0.0) != PETSC_SUCCESS) {
        status = SS_ERROR_INTERNAL;
      }
      if (status == SS_OK) {
        mechanics.nonlinear.first_value_generation = mechanics.next_value_generation;
        status = ss_nonlinear_elastic_predictor(&mechanics.nonlinear, predictor_scale,
                                                elastic_initial, &reference_work,
                                                &elastic_setup, &elastic_solve);
      }
      if (mechanics.next_value_generation < UINT64_MAX) {
        ++mechanics.next_value_generation;
      } else if (status == SS_OK) {
        status = SS_ERROR_OVERFLOW;
      }
    }
    if (status == SS_OK) {
      stage = "analysis_policy";
      status =
          ss_solver_mechanics_analysis_policy(graph, reference_work, &analysis_policy);
    }
    if (status == SS_OK) {
      const double encoded_efficiency_iterations =
          buffer.controller_values[SS_SOLVER_WORK_EFFICIENCY_ITERATIONS_INDEX];
      const double efficiency_elasticity =
          buffer.controller_values[SS_SOLVER_WORK_EFFICIENCY_ELASTICITY_INDEX];
      const double absolute_efficiency_iterations = fabs(encoded_efficiency_iterations);
      const int efficiency_controller =
          analysis_policy.work_step_controller == SS_ANALYSIS_WORK_STEP_WORK_EFFICIENCY;
      PetscInt efficiency_previous_iterations = 0;
      int efficiency_elasticity_ready = 0;
      if (efficiency_controller != 0) {
        if (!isfinite(encoded_efficiency_iterations) ||
            absolute_efficiency_iterations >
                (double)analysis_policy.work_efficiency_iteration_maximum ||
            floor(absolute_efficiency_iterations) != absolute_efficiency_iterations ||
            !isfinite(efficiency_elasticity) ||
            (encoded_efficiency_iterations <= 0.0 && efficiency_elasticity != 0.0)) {
          status = SS_ERROR_INTEGRITY;
        } else {
          efficiency_previous_iterations = (PetscInt)absolute_efficiency_iterations;
          efficiency_elasticity_ready = encoded_efficiency_iterations > 0.0;
        }
      } else if (encoded_efficiency_iterations != 0.0 || efficiency_elasticity != 0.0) {
        status = SS_ERROR_INTEGRITY;
      }
      if (status != SS_OK) {
        stage = "work_efficiency_restore";
      } else {
        resume_state = (SsAnalysisResumeState){
            .method = graph->model.analysis_method,
            .accepted_step_offset =
                graph->model.analysis_method == SS_ANALYSIS_LIMIT_LOAD
                    ? current_ordinal
                    : current_ordinal + 1U,
            .accepted_ordinal = current_ordinal,
            .next_operation = next_operation,
            .previous_factor = previous != NULL
                                   ? (PetscReal)previous->factor
                                   : (PetscReal)buffer.controller_values[0],
            .previous_control = previous != NULL
                                    ? (PetscReal)previous->direct_control
                                    : (PetscReal)buffer.controller_values[1],
            .previous_work = previous != NULL ? (PetscReal)previous->work
                                              : (PetscReal)buffer.controller_values[2],
            .current_factor = (PetscReal)buffer.controller_values[0],
            .current_control = (PetscReal)buffer.controller_values[1],
            .current_work = (PetscReal)buffer.controller_values[2],
            .current_signed_scaled_slope = (PetscReal)buffer.controller_values[4],
            .first_control = (PetscReal)buffer.controller_values[7],
            .next_step = (PetscReal)buffer.summary.next_step,
            .retry_count = (PetscInt)buffer.summary.retry_count,
            .work_efficiency_previous_iterations = efficiency_previous_iterations,
            .work_efficiency_elasticity = efficiency_elasticity_ready != 0
                                              ? (PetscReal)efficiency_elasticity
                                              : 0.0,
            .work_efficiency_elasticity_ready = efficiency_elasticity_ready,
        };
        mechanics.accepted_generation = (uint64_t)resume_state.accepted_step_offset;
        mechanics.nonlinear.accepted_generation = mechanics.accepted_generation + 1U;
      }
    }
    if (status == SS_OK && mechanics.deflation != NULL) {
      stage = "deflation_restore";
      status = ss_solver_checkpoint_deflation_apply(
          &mechanics, mechanics_solution, &buffer.deflation,
          buffer.summary.deflation_basis_count, (PetscReal)buffer.controller_values[0],
          (uint64_t)resume_state.accepted_step_offset);
    }
    if (status == SS_OK) {
      ss_solver_mechanics_analysis_problem(&mechanics, &analysis_problem);
      stage = "analysis_monitor";
      status = monitor == NULL
                   ? SS_OK
                   : ss_solver_monitor_attach(
                         monitor, &mechanics, coupled != 0 ? &hydraulic : NULL,
                         coupled != 0 ? hydraulic_solution : NULL,
                         coupled != 0 ? &hydraulic_evidence : NULL, &analysis_problem);
    }
    if (status == SS_OK) {
      if (monitor != NULL) {
        ss_solver_monitor_phase(monitor, "analysis");
      }
      stage = "analysis_resume";
      status = ss_analysis_resume(&analysis_problem, &analysis_policy,
                                  previous_mechanics_solution, mechanics_solution,
                                  mechanics_solution, &resume_state, analysis_record);
    }
    if (status == SS_OK && analysis_record->terminal_success == 0 &&
        analysis_record->stop_reason != SS_ANALYSIS_STOP_CANCELLED) {
      status = SS_ERROR_NUMERICAL;
      ss_solver_error(error, status, "ANALYSIS_RESTART_NOT_CONVERGED",
                      "resumed analysis stopped without scientific success", 3U);
    }
  }
  if (status == SS_OK) {
    stage = "result_publication";
    status =
        graph->model.mechanics_enabled != 0
            ? ss_solver_publish_mechanics(
                  request, graph, mechanics_solution, analysis_record, &mechanics,
                  coupled != 0 ? &hydraulic : NULL,
                  coupled != 0 ? hydraulic_solution : NULL,
                  coupled != 0 ? &hydraulic_evidence : NULL,
                  coupled != 0 ? coupling_snapshot : NULL,
                  coupled != 0 ? &coupling_diagnostics : NULL, monitor, summary, error)
            : ss_solver_publish_hydraulic(request, graph, &hydraulic,
                                          hydraulic_solution, hydraulic_solve, monitor,
                                          summary, error);
  }
  if (status == SS_OK && graph->model.mechanics_enabled != 0 &&
      analysis_record->stop_reason == SS_ANALYSIS_STOP_CANCELLED) {
    status = SS_ERROR_CANCELLED;
    ss_solver_error(error, status, "CANCELLED_IN_ANALYSIS",
                    "cancellation observed at an accepted-state safe point", 1U);
  }
  const PetscErrorCode elastic_error = VecDestroy(&elastic_initial);
  const PetscErrorCode hydraulic_error = VecDestroy(&hydraulic_solution);
  const PetscErrorCode previous_error = VecDestroy(&previous_mechanics_solution);
  const PetscErrorCode mechanics_error = VecDestroy(&mechanics_solution);
  if (elastic_error != PETSC_SUCCESS || hydraulic_error != PETSC_SUCCESS ||
      previous_error != PETSC_SUCCESS || mechanics_error != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  free(hydraulic_solve);
  free(analysis_record);
  if (ss_solver_mechanics_destroy(&mechanics) != SS_OK && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (ss_solver_hydraulic_destroy(&hydraulic) != SS_OK && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (ss_solver_coupling_snapshot_destroy(&coupling_snapshot) != SS_OK &&
      status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  ss_solver_checkpoint_buffer_destroy(&buffer);
  if (status != SS_OK && error->valid == 0) {
    char message[SS_ERROR_MESSAGE_CAPACITY];
    const int written =
        snprintf(message, sizeof(message),
                 "accepted-boundary checkpoint restore failed during %s (status=%d)",
                 stage, (int)status);
    ss_solver_error(error, status, "CHECKPOINT_RESTORE",
                    written >= 0 && (size_t)written < sizeof(message)
                        ? message
                        : "accepted-boundary checkpoint restore failed",
                    4U);
  }
  return ss_status_allreduce_max(graph->communicator, status);
}
