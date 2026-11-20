/* Publish accepted hydraulic fields and diagnostics.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "solver_output_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Append the global flux balance, the free residual and the balance and
 * initialization diagnostics of the accepted seepage state in published order.
 * Values are written unmodified; `stage` names the failing append. */
static SsStatus ss_solver_append_hydraulic_balance(
    SsResultWriter *writer, uint32_t dimension, SsStateId state_id,
    const SsHydraulicBalance *balance,
    const SsHydraulicInitializationRecord *initialization,
    double reconstructed_natural_flux, double reconstructed_dirichlet_flux,
    const char **stage) {
  const char *const flux = dimension == 2U ? "kPa*m/s" : "kPa*m^2/s";
  const struct {
    const char *quantity_id;
    const char *unit;
    double value;
  } scalars[] = {
      {"global_flux_balance", flux, balance->signed_imbalance},
      {"hydraulic_free_residual_l2", flux, balance->free_residual_l2},
      {"hydraulic_domain_source", flux, balance->domain_source},
      {"hydraulic_prescribed_boundary_flux", flux, balance->prescribed_boundary_flux},
      {"hydraulic_natural_boundary_flux", flux, balance->natural_boundary_flux},
      {"hydraulic_dirichlet_reaction_flux", flux, balance->dirichlet_reaction_flux},
      {"hydraulic_balance_physical_scale", flux, balance->physical_scale},
      {"hydraulic_balance_normalization", flux, balance->normalization},
      {"hydraulic_normalized_balance", "dimensionless", balance->normalized_imbalance},
      {"hydraulic_reconstructed_natural_flux", flux, reconstructed_natural_flux},
      {"hydraulic_reconstructed_dirichlet_flux", flux, reconstructed_dirichlet_flux},
      {"hydraulic_initialization_boundary_residual_l2", flux,
       initialization->boundary_residual_norm},
      {"hydraulic_initialization_saturated_residual_l2", flux,
       initialization->saturated_residual_norm},
      {"hydraulic_initialization_primary_saturated", "boolean",
       initialization->primary_choice == SS_HYDRAULIC_INITIAL_SATURATED_LINEAR},
      {"hydraulic_initialization_selected_saturated", "boolean",
       initialization->choice == SS_HYDRAULIC_INITIAL_SATURATED_LINEAR},
      {"hydraulic_initialization_fallback_attempted", "boolean",
       initialization->fallback_attempted},
  };
  size_t index;
  SsStatus status = SS_OK;
  for (index = 0U; status == SS_OK && index < sizeof(scalars) / sizeof(scalars[0]);
       ++index) {
    const SsResultScalarValue scalar = {
        .quantity_id = scalars[index].quantity_id,
        .quantity_version = 1U,
        .unit = scalars[index].unit,
        .state_id = state_id,
        .value = scalars[index].value,
        .complete = 1,
    };
    *stage = index == 0U   ? "append_balance"
             : index == 1U ? "append_residual"
                           : "append_balance_diagnostic";
    status = ss_result_writer_append_scalar(writer, &scalar);
  }
  return status;
}

SsStatus ss_solver_publish_hydraulic(const SsSolverRequest *request,
                                     const SsSolverGraph *graph,
                                     SsSolverHydraulic *hydraulic, Vec solution,
                                     const SsNonlinearSolveRecord *solve_record,
                                     SsSolverMonitor *monitor, SsSolverSummary *summary,
                                     SsErrorRecord *error) {
  Vec local_vector = NULL;
  const PetscScalar *local_values = NULL;
  SsSolverHydraulicOutputs outputs = {0};
  SsSolverPublishedQuantity quantities[SS_SOLVER_HYDRAULIC_OUTPUT_COUNT];
  size_t quantity_count;
  SsResultWriterSpec writer_spec;
  SsResultWriter *writer = NULL;
  SsResultWriterSummary writer_summary = {0};
  SsHydraulicBalanceLocal local_balance = {0};
  SsHydraulicBalance balance = {0};
  const SsStateId state_id = UINT64_C(1);
  SsResultHistoryRow attempt;
  SsResultHistoryRow accepted;
  SsResultAcceptedState accepted_state;
  double reconstructed_natural_flux = 0.0;
  double reconstructed_dirichlet_flux = 0.0;
  SsStatus status;
  const char *publication_stage = "output_extract";
  if (request == NULL || graph == NULL || hydraulic == NULL || solution == NULL ||
      solve_record == NULL || summary == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_field_space_extract_local(graph->hydraulic_space, solution, &local_vector,
                                        &local_values);
  if (status == SS_OK) {
    status = ss_solver_hydraulic_outputs_pack(graph, hydraulic, local_values, &outputs,
                                              &publication_stage);
  }
  if (status == SS_OK) {
    publication_stage = "hydraulic_boundary_balance";
    status = ss_solver_hydraulic_boundary_balance(
        graph, hydraulic, local_values, &local_balance, &reconstructed_natural_flux,
        &reconstructed_dirichlet_flux);
  }
  if (local_values != NULL &&
      VecRestoreArrayRead(local_vector, &local_values) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (VecDestroy(&local_vector) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    status = ss_solver_hydraulic_balance_evaluate(
        graph, hydraulic, solution, solve_record, &local_balance, &balance, error);
  }
  if (status == SS_OK && balance.accepted == 0) {
    char message[SS_ERROR_MESSAGE_CAPACITY];
    const int written = snprintf(
        message, sizeof(message),
        "hydraulic acceptance rejected: signed=%.17g denominator=%.17g "
        "normalized=%.17g limit=%.17g free_residual=%.17g limit=%.17g",
        balance.signed_imbalance, balance.normalization, balance.normalized_imbalance,
        graph->policy.conservation_normalized_tolerance, balance.free_residual_l2,
        (double)hydraulic->nonlinear_policy.tolerance *
            (double)solve_record->load_norm);
    status = SS_ERROR_NUMERICAL;
    ss_solver_error(error, status, "HYDRAULIC_BALANCE_REJECTED",
                    written >= 0 && (size_t)written < sizeof(message)
                        ? message
                        : "hydraulic balance or free residual was rejected",
                    3U);
  }
  if (status != SS_OK) {
    char message[SS_ERROR_MESSAGE_CAPACITY];
    const int written =
        snprintf(message, sizeof(message),
                 "hydraulic result preparation failed during %s (status=%d)",
                 publication_stage, (int)status);
    ss_solver_error(error, status, "RESULT_PUBLICATION",
                    written >= 0 && (size_t)written < sizeof(message)
                        ? message
                        : "hydraulic result preparation failed",
                    4U);
    ss_solver_hydraulic_outputs_destroy(&outputs);
    return status;
  }
  ss_solver_hydraulic_outputs_views(graph, state_id, &outputs);
  quantity_count = ss_solver_hydraulic_outputs_list(&outputs, quantities);
  attempt = (SsResultHistoryRow){
      .kind = SS_RESULT_HISTORY_SEEPAGE,
      .row_id = 0U,
      .state_id = state_id,
      .attempt_id = 1U,
      .parent_row_id = 0U,
      .ordinal = 0U,
      .control = 1.0,
      .residual = (double)solve_record->initial_residual_norm,
      .complete = 1,
      .accepted = 0,
  };
  accepted = (SsResultHistoryRow){
      .kind = SS_RESULT_HISTORY_ACCEPTED,
      .row_id = 1U,
      .state_id = state_id,
      .attempt_id = 1U,
      .parent_row_id = 0U,
      .ordinal = 1U,
      .control = 1.0,
      .residual = (double)solve_record->final_residual_norm,
      .complete = 1,
      .accepted = 1,
  };
  accepted_state = (SsResultAcceptedState){
      .state_id = state_id,
      .parent_state_id = 0U,
      .accepted_ordinal = 1U,
      .hydraulic_generation = 1U,
      .mechanics_generation = 0U,
      .control_generation = 1U,
      .factor = 1.0,
      .work = 0.0,
      .direct_control = 1.0,
      .displacement_maximum = 0.0,
      .history_begin = 0U,
      .history_end = 2U,
      .complete = 1,
      .accepted = 1,
  };
  writer_spec = (SsResultWriterSpec){
      .final_directory = request->result_directory,
      .problem_id = request->config->problem.id,
      .run_id = request->config->run.id,
      .build_id = request->build_id,
      .parent_checkpoint_id =
          graph->restart_provenance_valid != 0 ? graph->parent_checkpoint_id : NULL,
      .parent_run_id =
          graph->restart_provenance_valid != 0 ? graph->parent_run_id : NULL,
      .parent_build_id =
          graph->restart_provenance_valid != 0 ? graph->parent_build_id : NULL,
      .parent_accepted_state_id = graph->parent_accepted_state_id,
      .parent_source_index_bits = graph->parent_source_index_bits,
      .mesh_id = graph->model.mesh_id,
      .geometry_id = graph->geometry_id,
      .analysis_method = graph->model.result_analysis_method,
      .dimension = graph->model.dimension,
      .field_order = graph->model.hydraulic_order,
      .geometry_order = graph->model.coordinate_order,
      .dataset_chunk_rows = 0U,
      .communicator = graph->communicator,
  };
  publication_stage = "writer_create";
  status = ss_result_writer_create(&writer_spec, &writer);
  if (status == SS_OK) {
    publication_stage = "writer_stage";
    status = ss_result_writer_stage(writer);
  }
  if (status == SS_OK) {
    status = ss_solver_publish_declare(writer, graph, quantities, quantity_count,
                                       &publication_stage);
  }
  if (status == SS_OK) {
    publication_stage = "begin_stream";
    status = ss_result_writer_begin_stream(writer);
  }
  if (status == SS_OK) {
    publication_stage = "append_seepage_history";
    status = ss_result_writer_append_history(writer, &attempt);
  }
  if (status == SS_OK) {
    /* Complete per-iteration rows of the seepage solve below its stage row. */
    uint64_t row_id = 1U;
    publication_stage = "append_seepage_iterations";
    if (monitor != NULL && monitor->history_open) {
      status = ss_status_allreduce_max(monitor->communicator, monitor->detail_status);
      if (status == SS_OK) {
        status = ss_solver_history_spool_rewind(&monitor->history);
      }
      if (status == SS_OK) {
        status =
            ss_solver_history_spool_replay(&monitor->history, writer, 1U, 0U, &row_id);
      }
    }
    if (status == SS_OK) {
      status = ss_solver_history_append_solve(writer, solve_record, 1U,
                                              SS_RESULT_ROLE_SEEPAGE, 0U, &row_id);
    }
    accepted.row_id = row_id;
    accepted.parent_row_id = 0U;
    accepted_state.history_end = row_id + 1U;
  }
  if (status == SS_OK) {
    publication_stage = "append_accepted_history";
    status = ss_result_writer_append_history(writer, &accepted);
  }
  if (status == SS_OK) {
    publication_stage = "accept_state";
    status = ss_result_writer_accept_state(writer, &accepted_state);
  }
  if (status == SS_OK) {
    status = ss_solver_publish_write(writer, graph, quantities, quantity_count,
                                     &publication_stage);
  }
  if (status == SS_OK) {
    status = ss_solver_append_hydraulic_balance(
        writer, graph->model.dimension, state_id, &balance, &hydraulic->initialization,
        reconstructed_natural_flux, reconstructed_dirichlet_flux, &publication_stage);
  }
  if (status == SS_OK &&
      graph->policy.checkpoint_mode == SS_SOLVER_CHECKPOINT_WRITE_ACCEPTED) {
    publication_stage = "write_accepted_checkpoint";
    status = ss_solver_checkpoint_write_accepted(
        request, graph, writer, NULL, NULL, NULL, solution, hydraulic, solve_record,
        NULL, NULL, state_id, error);
  }
  if (status == SS_OK) {
    publication_stage = "analysis_outcome";
    status = ss_solver_set_result_outcome(
        writer, request, ss_solver_nonlinear_outcome(solve_record->outcome), NULL);
  }
  if (status == SS_OK) {
    publication_stage = "terminalize";
    status = ss_result_writer_terminalize(
        writer, SS_RESULT_SUCCEEDED,
        solve_record->outcome == SS_NONLINEAR_CONVERGED ? "converged"
                                                        : "accepted_relaxed",
        NULL);
  }
  if (status == SS_OK) {
    publication_stage = "publish";
    status = ss_result_writer_publish(writer);
  }
  if (writer != NULL) {
    (void)ss_result_writer_summary(writer, &writer_summary);
  }
  if (ss_result_writer_destroy(&writer) != SS_OK && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  ss_solver_hydraulic_outputs_destroy(&outputs);
  if (status != SS_OK) {
    if (error->valid == 0) {
      char message[SS_ERROR_MESSAGE_CAPACITY];
      const int written = snprintf(message, sizeof(message),
                                   "hydraulic result failed during %s (status=%d)",
                                   publication_stage, (int)status);
      ss_solver_error(error, status, "RESULT_PUBLICATION",
                      written >= 0 && (size_t)written < sizeof(message)
                          ? message
                          : "hydraulic result publication failed",
                      4U);
    }
    return status;
  }
  (void)snprintf(summary->result_id, sizeof(summary->result_id), "%s",
                 writer_summary.result_id);
  summary->accepted_states = 1U;
  summary->terminal_factor = 1.0;
  summary->result_published = 1;
  return SS_OK;
}
