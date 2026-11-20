/* Shared native result declarations, scalars and history publication.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "solver_internal.h"

#include <petsclog.h>

#include <math.h>
#include <string.h>

void ss_solver_output_field_bind(SsResultFieldView *view,
                                 const SsSolverOutputField *field, SsStateId state_id) {
  view->state_id = state_id;
  view->stable_ids = field->ids;
  view->local_slots = field->slots;
  view->values = field->values;
  view->local_count = field->local_count;
  view->global_count = field->global_count;
  view->quadrature_id =
      view->dataset_kind == SS_RESULT_DATASET_FIELD ? NULL : field->quadrature_id;
  view->sample_count = field->sample_count;
}

void ss_solver_output_category_bind(SsResultCategoryView *view,
                                    const SsSolverOutputCategory *category,
                                    SsStateId state_id) {
  view->state_id = state_id;
  view->stable_ids = category->ids;
  view->local_slots = category->slots;
  view->values = category->values;
  view->local_count = category->local_count;
  view->global_count = category->global_count;
  view->quadrature_id = category->quadrature_id;
  view->sample_count = category->sample_count;
}

int ss_solver_output_requested(const SsSolverGraph *graph, uint64_t quantity) {
  return graph != NULL && (graph->policy.requested_output_mask & quantity) != 0U;
}

SsResultQuantityDeclaration
ss_solver_quantity_declaration(const SsResultFieldView *view) {
  const SsResultQuantityDeclaration declaration = {
      .quantity_id = view->quantity_id,
      .quantity_version = view->quantity_version,
      .unit = view->unit,
      .location = view->location,
      .basis = view->basis,
      .component_ids = view->component_ids,
      .component_count = view->component_count,
      .stable_ids = view->stable_ids,
      .local_slots = view->local_slots,
      .local_count = view->local_count,
      .global_count = view->global_count,
      .dataset_kind = view->dataset_kind,
      .quadrature_id = view->quadrature_id,
      .sample_count = view->sample_count,
  };
  return declaration;
}

SsResultCategoryDeclaration
ss_solver_category_declaration(const SsResultCategoryView *view) {
  const SsResultCategoryDeclaration declaration = {
      .quantity_id = view->quantity_id,
      .quantity_version = view->quantity_version,
      .unit = view->unit,
      .location = view->location,
      .quadrature_id = view->quadrature_id,
      .category_ids = view->category_ids,
      .category_count = view->category_count,
      .stable_ids = view->stable_ids,
      .local_slots = view->local_slots,
      .local_count = view->local_count,
      .global_count = view->global_count,
      .sample_count = view->sample_count,
  };
  return declaration;
}

static double ss_solver_finite_or_zero(PetscReal value) {
  return isfinite((double)value) ? (double)value : 0.0;
}

const char *ss_solver_analysis_stop_reason(SsAnalysisStopReason reason) {
  switch (reason) {
  case SS_ANALYSIS_STOP_SCALED_SLOPE_REACHED:
    return "scaled_slope_reached";
  case SS_ANALYSIS_STOP_MINIMUM_STEP_REACHED:
    return "minimum_step_reached";
  case SS_ANALYSIS_STOP_WORK_HORIZON_REACHED:
    return "work_horizon_reached";
  case SS_ANALYSIS_STOP_DIRECT_CONTROL_HORIZON_REACHED:
    return "direct_control_horizon_reached";
  case SS_ANALYSIS_STOP_ACCEPTED_STEP_LIMIT:
    return "accepted_step_limit";
  case SS_ANALYSIS_STOP_CONSECUTIVE_RETRY_LIMIT:
    return "consecutive_retry_limit";
  case SS_ANALYSIS_STOP_NEWTON_FAILURE:
    return "newton_failure";
  case SS_ANALYSIS_STOP_LINEAR_FAILURE:
    return "linear_failure";
  case SS_ANALYSIS_STOP_SINGULAR_CONSTRAINT:
    return "singular_constraint";
  case SS_ANALYSIS_STOP_NONFINITE:
    return "nonfinite";
  case SS_ANALYSIS_STOP_CANCELLED:
    return "cancelled";
  case SS_ANALYSIS_STOP_NONE:
  default:
    return "none";
  }
}

const char *ss_solver_nonlinear_outcome(SsNonlinearOutcome outcome) {
  switch (outcome) {
  case SS_NONLINEAR_CONVERGED:
    return "converged_criterion";
  case SS_NONLINEAR_ACCEPTED_RELAXED:
    return "accepted_relaxed";
  case SS_NONLINEAR_FAILED_LINE_SEARCH:
    return "failed_line_search";
  case SS_NONLINEAR_FAILED_REGULARIZATION:
    return "failed_regularization";
  case SS_NONLINEAR_FAILED_LINEAR:
    return "failed_linear";
  case SS_NONLINEAR_FAILED_SINGULAR_CONSTRAINT:
    return "failed_singular_constraint";
  case SS_NONLINEAR_FAILED_NONFINITE:
    return "failed_nonfinite";
  case SS_NONLINEAR_CANCELLED:
    return "cancelled";
  case SS_NONLINEAR_FAILED_ITERATION_LIMIT:
  default:
    return "failed_iteration_limit";
  }
}

const char *ss_solver_analysis_outcome(const SsAnalysisRecord *record,
                                       int hydraulic_relaxed) {
  const char *outcome;
  size_t index;
  int relaxed = hydraulic_relaxed != 0;
  if (record == NULL) {
    return "failed_iteration_limit";
  }
  switch (record->stop_reason) {
  case SS_ANALYSIS_STOP_SCALED_SLOPE_REACHED:
    outcome = "converged_criterion";
    break;
  case SS_ANALYSIS_STOP_MINIMUM_STEP_REACHED:
    outcome =
        record->terminal_success != 0 ? "converged_criterion" : "failed_retry_limit";
    break;
  case SS_ANALYSIS_STOP_WORK_HORIZON_REACHED:
  case SS_ANALYSIS_STOP_DIRECT_CONTROL_HORIZON_REACHED:
    outcome = "completed_control_horizon";
    break;
  case SS_ANALYSIS_STOP_ACCEPTED_STEP_LIMIT:
    outcome = "completed_requested_step_horizon";
    break;
  case SS_ANALYSIS_STOP_CONSECUTIVE_RETRY_LIMIT:
    return "failed_retry_limit";
  case SS_ANALYSIS_STOP_LINEAR_FAILURE:
    return "failed_linear";
  case SS_ANALYSIS_STOP_SINGULAR_CONSTRAINT:
    return "failed_singular_constraint";
  case SS_ANALYSIS_STOP_NONFINITE:
    return "failed_nonfinite";
  case SS_ANALYSIS_STOP_CANCELLED:
    return "cancelled";
  case SS_ANALYSIS_STOP_NEWTON_FAILURE:
    return record->attempt_count == 0U
               ? "failed_iteration_limit"
               : ss_solver_nonlinear_outcome(
                     record->attempts[record->attempt_count - 1U].nonlinear_outcome);
  case SS_ANALYSIS_STOP_NONE:
  default:
    return "failed_iteration_limit";
  }
  for (index = 0U;
       index < record->attempt_count && index < SS_ANALYSIS_ATTEMPT_CAPACITY; ++index) {
    const SsAnalysisAttemptRecord *attempt = &record->attempts[index];
    if (attempt->committed != 0 &&
        (attempt->nonlinear_outcome == SS_NONLINEAR_ACCEPTED_RELAXED ||
         attempt->primary_solve.outcome == SS_NONLINEAR_ACCEPTED_RELAXED ||
         attempt->secondary_solve.outcome == SS_NONLINEAR_ACCEPTED_RELAXED)) {
      relaxed = 1;
    }
  }
  return relaxed != 0 ? "accepted_relaxed" : outcome;
}

SsStatus ss_solver_set_result_outcome(SsResultWriter *writer,
                                      const SsSolverRequest *request,
                                      const char *outcome,
                                      const char *mechanics_assembly_mode) {
  const SsResultOutcome result_outcome = {
      .outcome = outcome,
      .validation_policy =
          request != NULL && request->config != NULL && request->config->has_run != 0
              ? request->config->run.validation_profile
              : "unspecified",
      .mechanics_assembly_mode = mechanics_assembly_mode,
  };
  return ss_result_writer_set_outcome(writer, &result_outcome);
}

static SsStatus ss_solver_append_analysis_row(SsResultWriter *writer,
                                              SsResultHistoryKind kind,
                                              uint64_t *row_id, SsStateId state_id,
                                              uint64_t attempt_id, uint64_t ordinal,
                                              double control, double residual,
                                              int accepted) {
  SsResultHistoryRow row;
  if (writer == NULL || row_id == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(&row, 0, sizeof(row));
  row.kind = kind;
  row.row_id = *row_id;
  row.state_id = state_id;
  row.attempt_id = attempt_id;
  row.parent_row_id = *row_id == 0U ? 0U : *row_id - 1U;
  row.ordinal = ordinal;
  row.control = isfinite(control) ? control : 0.0;
  row.residual = isfinite(residual) ? residual : 0.0;
  row.complete = 1;
  row.accepted = accepted;
  if (ss_result_writer_append_history(writer, &row) != SS_OK) {
    return SS_ERROR_INVALID_STATE;
  }
  ++*row_id;
  return SS_OK;
}

static SsStatus ss_solver_append_analysis_solve(SsResultWriter *writer,
                                                const SsAnalysisSolveSummary *solve,
                                                uint64_t attempt_id, double control,
                                                uint64_t *row_id) {
  const SsLinearSolveRecord *linear[2];
  const uint64_t linear_iterations[2] = {
      solve->first_linear_iteration_count,
      solve->second_linear_iteration_count,
  };
  size_t index;
  SsStatus status;
  if (writer == NULL || solve == NULL || row_id == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (solve->history_count != 0U || solve->iterations != 0) {
    status = ss_solver_append_analysis_row(
        writer, SS_RESULT_HISTORY_NEWTON, row_id, 0U, attempt_id,
        (uint64_t)(solve->iterations < 0 ? 0 : solve->iterations), control,
        ss_solver_finite_or_zero(solve->final_convergence), 0);
    if (status != SS_OK) {
      return status;
    }
  }
  linear[0] = &solve->first_linear_solve;
  linear[1] = &solve->second_linear_solve;
  for (index = 0U; index < 2U; ++index) {
    if (linear[index]->solve_count == 0U) {
      continue;
    }
    status = ss_solver_append_analysis_row(
        writer, SS_RESULT_HISTORY_LINEAR, row_id, 0U, attempt_id,
        linear_iterations[index], control,
        ss_solver_finite_or_zero(linear[index]->true_residual_norm), 0);
    if (status != SS_OK) {
      return status;
    }
  }
  return SS_OK;
}

static SsStatus
ss_solver_append_analysis_attempt(SsResultWriter *writer,
                                  const SsAnalysisAttemptRecord *attempt,
                                  SsSolverHistorySpool *spool, uint64_t *row_id) {
  const uint64_t attempt_id = (uint64_t)attempt->attempt_ordinal + 1U;
  const double control = isfinite((double)attempt->target)
                             ? (double)attempt->target
                             : ss_solver_finite_or_zero(attempt->incoming_control);
  SsStatus status = ss_solver_append_analysis_row(
      writer, SS_RESULT_HISTORY_ATTEMPT, row_id, 0U, attempt_id,
      (uint64_t)attempt->retry_ordinal, control,
      ss_solver_finite_or_zero(attempt->primary_solve.final_convergence), 0);
  if (status == SS_OK && spool != NULL) {
    /* Complete per-iteration rows of the attempt's solves (SOL-08). */
    return ss_solver_history_spool_replay(spool, writer, attempt_id, *row_id - 1U,
                                          row_id);
  }
  if (status == SS_OK) {
    status = ss_solver_append_analysis_solve(writer, &attempt->primary_solve,
                                             attempt_id, control, row_id);
  }
  if (status == SS_OK && (attempt->secondary_solve.history_count != 0U ||
                          attempt->secondary_solve.iterations != 0 ||
                          attempt->secondary_solve.linear_solve_count != 0U)) {
    status = ss_solver_append_analysis_solve(writer, &attempt->secondary_solve,
                                             attempt_id, control, row_id);
  }
  return status;
}

SsStatus ss_solver_append_analysis_scalar(SsResultWriter *writer,
                                          const char *quantity_id, const char *unit,
                                          SsStateId state_id, double value) {
  const SsResultScalarValue scalar = {
      .quantity_id = quantity_id,
      .quantity_version = 1U,
      .unit = unit,
      .state_id = state_id,
      .value = !isfinite(value) || value == 0.0 ? 0.0 : value,
      .complete = 1,
  };
  return ss_result_writer_append_scalar(writer, &scalar);
}

SsStatus
ss_solver_append_coupled_diagnostics(SsResultWriter *writer, MPI_Comm communicator,
                                     SsStateId state_id, uint32_t dimension,
                                     const SsSolverHydraulicEvidence *hydraulic,
                                     const SsCouplingDiagnostics *coupling) {
  static const char *const quantity_ids[] = {
      "global_flux_balance",
      "hydraulic_free_residual_l2",
      "hydraulic_domain_source",
      "hydraulic_prescribed_boundary_flux",
      "hydraulic_natural_boundary_flux",
      "hydraulic_dirichlet_reaction_flux",
      "hydraulic_balance_physical_scale",
      "hydraulic_balance_normalization",
      "hydraulic_normalized_balance",
      "hydraulic_reconstructed_natural_flux",
      "hydraulic_reconstructed_dirichlet_flux",
      "coupling_global_target_points",
      "coupling_global_target_cells",
      "coupling_source_evaluations",
      "coupling_saturated_cells",
      "coupling_unsaturated_cells",
      "coupling_unresolved_points",
      "coupling_ambiguous_points",
      "coupling_snapshot_bytes",
      "coupling_pressure_minimum",
      "coupling_pressure_maximum",
      "coupling_gradient_norm_maximum",
      "coupling_observed_threads",
      "coupling_used_openmp",
      "hydraulic_initialization_boundary_residual_l2",
      "hydraulic_initialization_saturated_residual_l2",
      "hydraulic_initialization_primary_saturated",
      "hydraulic_initialization_selected_saturated",
      "hydraulic_initialization_fallback_attempted",
  };
  const char *units[sizeof(quantity_ids) / sizeof(quantity_ids[0])];
  double values[sizeof(quantity_ids) / sizeof(quantity_ids[0])];
  const char *flux_unit;
  uint64_t local_snapshot_bytes;
  uint64_t global_snapshot_bytes = 0U;
  size_t index;
  if (writer == NULL || hydraulic == NULL || hydraulic->valid == 0 ||
      coupling == NULL || communicator == MPI_COMM_NULL ||
      (dimension != 2U && dimension != 3U)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  local_snapshot_bytes = (uint64_t)coupling->snapshot_bytes;
  if (MPI_Allreduce(&local_snapshot_bytes, &global_snapshot_bytes, 1, MPI_UINT64_T,
                    MPI_SUM, communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  flux_unit = dimension == 2U ? "kPa*m/s" : "kPa*m^2/s";
  for (index = 0U; index < 8U; ++index) {
    units[index] = flux_unit;
  }
  units[8] = "dimensionless";
  units[9] = flux_unit;
  units[10] = flux_unit;
  for (index = 11U; index < 18U; ++index) {
    units[index] = "count";
  }
  units[18] = "byte";
  units[19] = "kPa";
  units[20] = "kPa";
  units[21] = "kPa/m";
  units[22] = "count";
  units[23] = "boolean";
  units[24] = flux_unit;
  units[25] = flux_unit;
  units[26] = "boolean";
  units[27] = "boolean";
  units[28] = "boolean";
  values[0] = hydraulic->balance.signed_imbalance;
  values[1] = hydraulic->balance.free_residual_l2;
  values[2] = hydraulic->balance.domain_source;
  values[3] = hydraulic->balance.prescribed_boundary_flux;
  values[4] = hydraulic->balance.natural_boundary_flux;
  values[5] = hydraulic->balance.dirichlet_reaction_flux;
  values[6] = hydraulic->balance.physical_scale;
  values[7] = hydraulic->balance.normalization;
  values[8] = hydraulic->balance.normalized_imbalance;
  values[9] = hydraulic->reconstructed_natural_flux;
  values[10] = hydraulic->reconstructed_dirichlet_flux;
  values[11] = (double)coupling->global_target_points;
  values[12] = (double)coupling->global_target_cells;
  values[13] = (double)coupling->source_evaluations;
  values[14] = (double)coupling->saturated_cells;
  values[15] = (double)coupling->unsaturated_cells;
  values[16] = (double)coupling->unresolved_points;
  values[17] = (double)coupling->ambiguous_points;
  values[18] = (double)global_snapshot_bytes;
  values[19] = coupling->pressure_minimum;
  values[20] = coupling->pressure_maximum;
  values[21] = coupling->gradient_norm_maximum;
  values[22] = (double)coupling->observed_threads;
  values[23] = (double)coupling->used_openmp;
  values[24] = hydraulic->initialization.boundary_residual_norm;
  values[25] = hydraulic->initialization.saturated_residual_norm;
  values[26] =
      hydraulic->initialization.primary_choice == SS_HYDRAULIC_INITIAL_SATURATED_LINEAR;
  values[27] =
      hydraulic->initialization.choice == SS_HYDRAULIC_INITIAL_SATURATED_LINEAR;
  values[28] = hydraulic->initialization.fallback_attempted;
  for (index = 0U; index < sizeof(quantity_ids) / sizeof(quantity_ids[0]); ++index) {
    SsStatus status = ss_solver_append_analysis_scalar(
        writer, quantity_ids[index], units[index], state_id, values[index]);
    if (status != SS_OK) {
      return status;
    }
  }
  return SS_OK;
}

SsStatus ss_solver_append_analysis_history(
    SsResultWriter *writer, const SsAnalysisRecord *record,
    const SsDeflation *deflation, uint64_t hydraulic_generation,
    const SsSolverHydraulicEvidence *hydraulic_evidence,
    const SsCouplingDiagnostics *coupling_diagnostics, SsSolverHistorySpool *spool) {
  uint64_t row_id = 0U;
  size_t attempt_cursor = 0U;
  size_t point_index;
  SsStatus status = SS_OK;
  if (writer == NULL || record == NULL || record->point_count == 0U ||
      record->point_count > SS_ANALYSIS_POINT_CAPACITY ||
      record->attempt_count > SS_ANALYSIS_ATTEMPT_CAPACITY) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (spool != NULL) {
    /* Seepage and elastic-predictor solves precede the first attempt. */
    status = ss_solver_history_spool_rewind(spool);
    if (status == SS_OK) {
      status = ss_solver_history_spool_replay(spool, writer, 0U, 0U, &row_id);
    }
  }
  for (point_index = 0U; status == SS_OK && point_index < record->point_count;
       ++point_index) {
    const SsAnalysisPointRecord *point = &record->points[point_index];
    const SsStateId state_id = (SsStateId)point_index + 1U;
    const SsStateId parent_state_id = point_index == 0U ? 0U : (SsStateId)point_index;
    const int synthetic_limit_load_origin = record->method == SS_ANALYSIS_LIMIT_LOAD &&
                                            point_index == 0U && point->factor == 0.0 &&
                                            point->control == 0.0 && point->work == 0.0;
    const int synthetic_restart_origin =
        point_index == 0U && record->resumed_from_accepted_state != 0;
    const uint64_t history_begin = row_id;
    uint64_t accepted_attempt_id = 0U;
    double accepted_residual = 0.0;
    if (synthetic_limit_load_origin == 0 && synthetic_restart_origin == 0) {
      if (point->source_attempt_ordinal >= record->attempt_count ||
          point->source_attempt_ordinal < attempt_cursor) {
        return SS_ERROR_INTEGRITY;
      }
      while (status == SS_OK && attempt_cursor <= point->source_attempt_ordinal) {
        const SsAnalysisAttemptRecord *attempt = &record->attempts[attempt_cursor];
        status = ss_solver_append_analysis_attempt(writer, attempt, spool, &row_id);
        if (attempt_cursor == point->source_attempt_ordinal) {
          accepted_attempt_id = (uint64_t)attempt->attempt_ordinal + 1U;
          accepted_residual =
              ss_solver_finite_or_zero(attempt->primary_solve.final_convergence);
          if (attempt->committed == 0 || attempt->successful == 0) {
            status = SS_ERROR_INTEGRITY;
          }
        }
        ++attempt_cursor;
      }
    }
    if (status == SS_OK) {
      status = ss_solver_append_analysis_row(
          writer, SS_RESULT_HISTORY_ACCEPTED, &row_id, state_id, accepted_attempt_id,
          (uint64_t)point->accepted_ordinal, ss_solver_finite_or_zero(point->control),
          accepted_residual, 1);
    }
    if (status == SS_OK) {
      const SsResultAcceptedState state = {
          .state_id = state_id,
          .parent_state_id = parent_state_id,
          .accepted_ordinal = (uint64_t)point->accepted_ordinal,
          .hydraulic_generation = hydraulic_generation,
          .mechanics_generation = state_id,
          .control_generation = state_id,
          .factor = ss_solver_finite_or_zero(point->factor),
          .work = ss_solver_finite_or_zero(point->work),
          .direct_control = ss_solver_finite_or_zero(point->control),
          .displacement_maximum = ss_solver_finite_or_zero(point->displacement_maximum),
          .history_begin = history_begin,
          .history_end = row_id,
          .complete = 1,
          .accepted = 1,
      };
      status = ss_result_writer_accept_state(writer, &state);
    }
  }
  while (status == SS_OK && attempt_cursor < record->attempt_count) {
    status = ss_solver_append_analysis_attempt(
        writer, &record->attempts[attempt_cursor], spool, &row_id);
    ++attempt_cursor;
  }
  if (status == SS_OK && hydraulic_evidence != NULL && hydraulic_evidence->valid != 0) {
    status = ss_solver_append_analysis_row(
        writer, SS_RESULT_HISTORY_SEEPAGE, &row_id, (SsStateId)record->point_count, 0U,
        coupling_diagnostics == NULL ? 0U : coupling_diagnostics->source_evaluations,
        hydraulic_evidence->balance.normalized_imbalance,
        hydraulic_evidence->final_residual_norm, 0);
  }
  if (status == SS_OK && deflation != NULL) {
    SsDeflationDiagnostics diagnostics;
    status = ss_deflation_diagnostics(deflation, &diagnostics);
    if (status == SS_OK) {
      status = ss_solver_append_analysis_row(
          writer, SS_RESULT_HISTORY_DEFLATION, &row_id, 0U,
          diagnostics.accepted_generation, diagnostics.refresh_count,
          (double)diagnostics.active_count,
          ss_solver_finite_or_zero(diagnostics.maximum_invariant_error), 0);
    }
  }
  if (status == SS_OK) {
    const SsAnalysisPointRecord *terminal = &record->points[record->point_count - 1U];
    const SsStateId terminal_state_id = (SsStateId)record->point_count;
    status =
        ss_solver_append_analysis_scalar(writer, "load_factor", "dimensionless",
                                         terminal_state_id, (double)terminal->factor);
    if (status == SS_OK) {
      status = ss_solver_append_analysis_scalar(
          writer, "external_work", "kN*m", terminal_state_id, (double)terminal->work);
    }
    if (status == SS_OK) {
      status = ss_solver_append_analysis_scalar(writer, "direct_control", "kN*m",
                                                terminal_state_id,
                                                (double)terminal->control);
    }
    if (status == SS_OK) {
      status = ss_solver_append_analysis_scalar(writer, "displacement_maximum", "m",
                                                terminal_state_id,
                                                (double)terminal->displacement_maximum);
    }
    if (status == SS_OK) {
      status = ss_solver_append_analysis_scalar(writer, "signed_scaled_slope",
                                                "dimensionless", terminal_state_id,
                                                (double)terminal->signed_scaled_slope);
    }
  }
  return status;
}
