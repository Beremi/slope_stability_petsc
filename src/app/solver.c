/* Native validate and run orchestration.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "solver_internal.h"

#include "slope_internal/diagnostics/diagnostics.h"

#include <petsclog.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static PetscLogEvent ss_solver_graph_create_event = -1;
static PetscLogEvent ss_solver_mechanics_create_event = -1;
static PetscLogEvent ss_solver_state_create_event = -1;
static PetscLogEvent ss_solver_elastic_event = -1;
static PetscLogEvent ss_solver_analysis_event = -1;
static PetscLogEvent ss_solver_publish_event = -1;
static PetscLogEvent ss_solver_cleanup_event = -1;

static void ss_solver_profile_events_register(void) {
  if (ss_solver_graph_create_event < 0) {
    (void)PetscLogEventRegister("SlopeGraphCreate", 0, &ss_solver_graph_create_event);
    (void)PetscLogEventRegister("SlopeMechCreate", 0,
                                &ss_solver_mechanics_create_event);
    (void)PetscLogEventRegister("SlopeStateCreate", 0, &ss_solver_state_create_event);
    (void)PetscLogEventRegister("SlopeElastic", 0, &ss_solver_elastic_event);
    (void)PetscLogEventRegister("SlopeAnalysis", 0, &ss_solver_analysis_event);
    (void)PetscLogEventRegister("SlopePublish", 0, &ss_solver_publish_event);
    (void)PetscLogEventRegister("SlopeCleanup", 0, &ss_solver_cleanup_event);
  }
}

static SsStatus ss_solver_fill_summary(const SsSolverGraph *graph,
                                       SsSolverSummary *summary) {
  SsMeshSummary mesh_summary = {0};
  SsFieldSummary field_summary = {0};
  SsStatus status;
  if (graph == NULL || summary == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_mesh_summary(graph->mesh, &mesh_summary);
  if (status == SS_OK && graph->mechanics_space != NULL) {
    status = ss_field_space_summary(graph->mechanics_space, &field_summary);
  } else if (status == SS_OK && graph->hydraulic_space != NULL) {
    status = ss_field_space_summary(graph->hydraulic_space, &field_summary);
  }
  if (status == SS_OK) {
    (void)snprintf(summary->mesh_id, sizeof(summary->mesh_id), "%s",
                   graph->model.mesh_id);
    (void)snprintf(summary->geometry_id, sizeof(summary->geometry_id), "%s",
                   graph->geometry_id);
    summary->global_cells = mesh_summary.global_cells;
    summary->global_free_dofs = field_summary.global_free_dof_count;
  }
  return status;
}

SsStatus ss_solver_validate(const SsSolverRequest *request, SsSolverSummary *summary,
                            SsErrorRecord *error) {
  SsSolverGraph graph;
  SsStatus status;
  if (request == NULL || summary == NULL || error == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(summary, 0, sizeof(*summary));
  memset(error, 0, sizeof(*error));
  status = ss_solver_graph_create(request, &graph, error);
  if (status != SS_OK) {
    return status;
  }
  status = ss_solver_fill_summary(&graph, summary);
  ss_solver_graph_destroy(&graph);
  return status;
}

SsStatus ss_solver_run(const SsSolverRequest *request, SsSolverSummary *summary,
                       SsErrorRecord *error) {
  SsSolverGraph graph;
  SsSolverMechanics mechanics;
  SsSolverHydraulic hydraulic;
  SsNonlinearSolveRecord *solve_record = NULL;
  SsLinearSetupRecord elastic_setup = {0};
  SsLinearSolveRecord elastic_solve = {0};
  SsAnalysisProblem analysis_problem;
  SsAnalysisPolicy analysis_policy;
  SsAnalysisRecord *analysis_record = NULL;
  SsSolverCouplingSnapshot *coupling_snapshot = NULL;
  SsCouplingDiagnostics coupling_diagnostics = {0};
  SsSolverHydraulicEvidence hydraulic_evidence = {0};
  SsSolverMonitor monitor;
  int monitor_open = 0;
  Vec initial = NULL;
  Vec solution = NULL;
  Vec hydraulic_solution = NULL;
  const SsAssemblyPlan *active_plan = NULL;
  PetscReal reference_work = NAN;
  SsDeflationDiagnostics failure_deflation_diagnostics = {0};
  SsStatus failure_transaction_status = SS_OK;
  int failure_transaction_open = 0;
  SsStatus status;
  const char *stage = "graph";
  if (request == NULL || request->communicator == MPI_COMM_NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_status_allreduce_max(request->communicator,
                                   summary == NULL || error == NULL ||
                                           request->result_directory == NULL ||
                                           request->build_id == NULL
                                       ? SS_ERROR_INVALID_ARGUMENT
                                       : SS_OK);
  if (status != SS_OK) {
    return status;
  }
  memset(summary, 0, sizeof(*summary));
  memset(error, 0, sizeof(*error));
  memset(&mechanics, 0, sizeof(mechanics));
  memset(&hydraulic, 0, sizeof(hydraulic));
  memset(&monitor, 0, sizeof(monitor));
  ss_solver_profile_events_register();
  (void)PetscLogEventBegin(ss_solver_graph_create_event, 0, 0, 0, 0);
  status = ss_solver_graph_create(request, &graph, error);
  (void)PetscLogEventEnd(ss_solver_graph_create_event, 0, 0, 0, 0);
  if (status != SS_OK) {
    return status;
  }
  (void)ss_memory_view(graph.communicator, "solver_after_graph");
  stage = "summary";
  status = ss_status_allreduce_max(graph.communicator,
                                   ss_solver_fill_summary(&graph, summary));
  if (status == SS_OK) {
    stage = "run_monitor";
    status = ss_solver_monitor_open(&monitor, request, &graph);
    monitor_open = status == SS_OK;
    ss_solver_monitor_phase(&monitor, "setup");
  }
  if (status == SS_OK && graph.model.mechanics_enabled == 0 &&
      graph.model.hydraulic_enabled == 0) {
    status = SS_ERROR_INVALID_STATE;
    ss_solver_error(error, status, "SOLVER_PHYSICS_NOT_CONNECTED",
                    "problem enables no numerical physics", 3U);
  }
  if (status == SS_OK && graph.model.mechanics_enabled != 0 &&
      graph.model.hydraulic_enabled != 0 && graph.model.coupling_enabled == 0) {
    status = SS_ERROR_INVALID_STATE;
    ss_solver_error(error, status, "SOLVER_PHYSICS_NOT_CONNECTED",
                    "simultaneous mechanics and seepage require one-way coupling", 3U);
  }
  status = ss_status_allreduce_max(graph.communicator, status);
  if (status == SS_OK &&
      graph.policy.checkpoint_mode == SS_SOLVER_CHECKPOINT_RESTORE_ACCEPTED) {
    stage = "accepted_checkpoint_restore";
    ss_solver_monitor_phase(&monitor, "restore");
    status = ss_solver_checkpoint_restore_accepted(request, &graph, &monitor, summary,
                                                   error);
    goto solver_cleanup;
  }
  if (status == SS_OK) {
    stage = "record_allocation";
    if (graph.model.mechanics_enabled != 0) {
      analysis_record = calloc(1U, sizeof(*analysis_record));
      if (analysis_record == NULL) {
        status = SS_ERROR_OUT_OF_MEMORY;
      }
    }
    if (graph.model.hydraulic_enabled != 0) {
      solve_record = calloc(1U, sizeof(*solve_record));
      if (solve_record == NULL) {
        status = SS_ERROR_OUT_OF_MEMORY;
      }
    }
    status = ss_status_allreduce_max(graph.communicator, status);
  }
  if (status == SS_OK && graph.model.mechanics_enabled != 0 &&
      graph.model.hydraulic_enabled != 0) {
    stage = "coupled_hydraulic_create";
    ss_solver_monitor_phase(&monitor, "seepage");
    status = ss_solver_hydraulic_create(&graph, request->requested_threads, &hydraulic,
                                        error);
    if (status == SS_OK) {
      ss_solver_monitor_watch_hydraulic(&monitor, &hydraulic);
      stage = "coupled_hydraulic_state";
      status = ss_assembly_plan_create_vector(graph.hydraulic_plan, &initial);
    }
    if (status == SS_OK &&
        VecDuplicate(initial, &hydraulic_solution) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
    status = ss_status_allreduce_max(graph.communicator, status);
    if (status == SS_OK) {
      stage = "coupled_hydraulic_solve";
      status = ss_solver_hydraulic_solve(&graph, &hydraulic, initial,
                                         hydraulic_solution, solve_record, error);
    }
    if (status == SS_OK) {
      status = ss_solver_monitor_record_solve(&monitor, solve_record,
                                              SS_RESULT_ROLE_SEEPAGE);
    }
    if (status == SS_OK && solve_record->outcome == SS_NONLINEAR_CANCELLED) {
      status = SS_ERROR_CANCELLED;
      ss_solver_error(error, status, "CANCELLED_IN_SOLVE",
                      "cancellation observed at a seepage Newton safe point", 1U);
    }
    if (status == SS_OK && solve_record->outcome != SS_NONLINEAR_CONVERGED &&
        solve_record->outcome != SS_NONLINEAR_ACCEPTED_RELAXED) {
      char message[SS_ERROR_MESSAGE_CAPACITY];
      const SsNonlinearIterationRecord *failed =
          solve_record->history_count < SS_NONLINEAR_HISTORY_CAPACITY
              ? &solve_record->history[solve_record->history_count]
              : NULL;
      SsPmgApplyRecord pmg_record = {0};
      if (hydraulic.pmg != NULL) {
        (void)ss_pmg_last_apply(hydraulic.pmg, &pmg_record);
      }
      const int written = snprintf(
          message, sizeof(message),
          "coupled hydraulic failed: out=%d nit=%lld r=%.5g "
          "setup=%d/%d lin=%d:%d/%lld rn=%.5g pmg=%d/%d/%lld "
          "init=%d primary=%d fallback=%d primary_out=%d boundary=%.5g "
          "saturated=%.5g candidate=%d",
          (int)solve_record->outcome, (long long)solve_record->iterations,
          (double)solve_record->final_residual_norm,
          failed == NULL ? 0 : (int)failed->linear_setup.attempt_count,
          failed == NULL || failed->linear_setup.attempt_count == 0U
              ? 0
              : (int)failed->linear_setup.attempts[0].setup_status,
          failed == NULL ? 0 : (int)failed->linear_solve.status,
          failed == NULL ? 0 : (int)failed->linear_solve.reason,
          failed == NULL ? 0LL : (long long)failed->linear_solve.iterations,
          failed == NULL ? 0.0 : (double)failed->linear_solve.true_residual_norm,
          (int)pmg_record.status, (int)pmg_record.last_level_reason,
          (long long)pmg_record.last_level_iterations,
          (int)hydraulic.initialization.choice,
          (int)hydraulic.initialization.primary_choice,
          hydraulic.initialization.fallback_attempted,
          hydraulic.primary_outcome_valid != 0 ? (int)hydraulic.primary_outcome : -1,
          hydraulic.initialization.boundary_residual_norm,
          hydraulic.initialization.saturated_residual_norm,
          hydraulic.initialization.saturated_candidate_valid);
      status = SS_ERROR_NUMERICAL;
      ss_solver_error(error, status, "COUPLED_HYDRAULIC_NOT_CONVERGED",
                      written >= 0 && (size_t)written < sizeof(message)
                          ? message
                          : "coupled hydraulic solve did not converge",
                      3U);
    }
    if (status == SS_OK) {
      stage = "coupled_hydraulic_acceptance";
      status = ss_solver_hydraulic_acceptance(&graph, &hydraulic, hydraulic_solution,
                                              solve_record, &hydraulic_evidence, error);
    }
    if (status == SS_OK) {
      stage = "coupling_snapshot";
      status = ss_solver_coupling_snapshot_build(
          &graph, hydraulic_solution, request->requested_threads, &coupling_snapshot,
          &coupling_diagnostics, error);
    }
    if (VecDestroy(&initial) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
    memset(solve_record, 0, sizeof(*solve_record));
  }
  if (status == SS_OK && graph.model.mechanics_enabled != 0) {
    stage = "mechanics_create";
    (void)PetscLogEventBegin(ss_solver_mechanics_create_event, 0, 0, 0, 0);
    status = ss_solver_mechanics_create(&graph, request->requested_threads,
                                        coupling_snapshot, &mechanics, error);
    (void)PetscLogEventEnd(ss_solver_mechanics_create_event, 0, 0, 0, 0);
    if (status == SS_OK &&
        (mechanics.operator_spec.volume_quadrature_mode ==
                 SS_FE_QUADRATURE_MODE_PAIRED_HIGHER
             ? SS_FE_QUADRATURE_MODE_PAIRED_HIGHER
             : SS_FE_QUADRATURE_MODE_DEFAULT) != graph.model.volume_quadrature_mode) {
      /* Outputs and coupling targets follow the problem's volume rule; the
       * mechanics operators must use the same rule. */
      status = SS_ERROR_SCHEMA;
      ss_solver_error(error, status, "VOLUME_QUADRATURE_NOT_SUPPORTED",
                      "the mechanics operators do not use the problem volume rule", 3U);
    }
    active_plan = graph.mechanics_plan;
  } else if (status == SS_OK) {
    stage = "hydraulic_create";
    ss_solver_monitor_phase(&monitor, "seepage");
    status = ss_solver_hydraulic_create(&graph, request->requested_threads, &hydraulic,
                                        error);
    if (status == SS_OK) {
      ss_solver_monitor_watch_hydraulic(&monitor, &hydraulic);
    }
    active_plan = graph.hydraulic_plan;
  }
  status = ss_status_allreduce_max(graph.communicator, status);
  (void)ss_memory_view(graph.communicator, "solver_after_physics_create");
  (void)PetscLogEventBegin(ss_solver_state_create_event, 0, 0, 0, 0);
  if (status == SS_OK) {
    stage = "state_create";
    status = ss_assembly_plan_create_vector(active_plan, &initial);
  }
  status = ss_status_allreduce_max(graph.communicator, status);
  if (status == SS_OK && VecDuplicate(initial, &solution) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    status = graph.model.mechanics_enabled != 0
                 ? (VecSet(initial, 0.0) == PETSC_SUCCESS ? SS_OK : SS_ERROR_INTERNAL)
                 : SS_OK;
  }
  status = ss_status_allreduce_max(graph.communicator, status);
  (void)PetscLogEventEnd(ss_solver_state_create_event, 0, 0, 0, 0);
  (void)ss_memory_view(graph.communicator, "solver_after_state_create");
  if (status == SS_OK) {
    if (graph.model.mechanics_enabled != 0) {
      const PetscReal predictor_scale =
          graph.model.analysis_method == SS_ANALYSIS_LIMIT_LOAD
              ? (PetscReal)graph.policy.initial_work_scale
              : 1.0;
      stage = "mechanics_elastic_predictor";
      ss_solver_monitor_phase(&monitor, "elastic_predictor");
      ss_solver_monitor_watch_nonlinear(&monitor, &mechanics.nonlinear,
                                        SS_RESULT_ROLE_ELASTIC);
      mechanics.nonlinear.first_value_generation = mechanics.next_value_generation;
      (void)PetscLogEventBegin(ss_solver_elastic_event, 0, 0, 0, 0);
      status = ss_nonlinear_elastic_predictor(&mechanics.nonlinear, predictor_scale,
                                              initial, &reference_work, &elastic_setup,
                                              &elastic_solve);
      (void)PetscLogEventEnd(ss_solver_elastic_event, 0, 0, 0, 0);
      if (status == SS_OK) {
        status = ss_solver_monitor_record_linear(&monitor, &elastic_solve,
                                                 SS_RESULT_ROLE_ELASTIC);
      }
      if (mechanics.next_value_generation < UINT64_MAX) {
        ++mechanics.next_value_generation;
      } else if (status == SS_OK) {
        status = SS_ERROR_OVERFLOW;
      }
      if (status == SS_OK && graph.model.analysis_method != SS_ANALYSIS_LIMIT_LOAD &&
          VecSet(initial, 0.0) != PETSC_SUCCESS) {
        status = SS_ERROR_INTERNAL;
      }
      if (status != SS_OK && error->valid == 0) {
        char message[SS_ERROR_MESSAGE_CAPACITY];
        const int written = snprintf(
            message, sizeof(message),
            "elastic predictor failed: setup=%d attempts=%zu selected=%s "
            "solve=%d reason=%d it=%lld reported=%.8g true=%.8g rhs=%.8g "
            "threshold=%.8g",
            (int)(elastic_setup.attempt_count == 0U
                      ? status
                      : elastic_setup.attempts[elastic_setup.attempt_count - 1U]
                            .setup_status),
            elastic_setup.attempt_count, elastic_setup.selected_identity,
            (int)elastic_solve.status, (int)elastic_solve.reason,
            (long long)elastic_solve.iterations, (double)elastic_solve.reported_norm,
            (double)elastic_solve.true_residual_norm, (double)elastic_solve.rhs_norm,
            (double)elastic_solve.acceptance_threshold);
        ss_solver_error(error, status, "MECHANICS_ELASTIC_PREDICTOR_FAILED",
                        written >= 0 && (size_t)written < sizeof(message)
                            ? message
                            : "mechanics elastic predictor failed",
                        3U);
      }
    } else {
      stage = "hydraulic_solve";
      status = ss_solver_hydraulic_solve(&graph, &hydraulic, initial, solution,
                                         solve_record, error);
    }
  }
  if (status == SS_OK && graph.model.mechanics_enabled == 0 &&
      solve_record->outcome == SS_NONLINEAR_CANCELLED) {
    status = SS_ERROR_CANCELLED;
    ss_solver_error(error, status, "CANCELLED_IN_SOLVE",
                    "cancellation observed at a seepage Newton safe point", 1U);
  }
  status = ss_status_allreduce_max(graph.communicator, status);
  if (status == SS_OK && graph.model.mechanics_enabled == 0) {
    if (status == SS_OK && solve_record->outcome != SS_NONLINEAR_CONVERGED &&
        solve_record->outcome != SS_NONLINEAR_ACCEPTED_RELAXED) {
      char message[SS_ERROR_MESSAGE_CAPACITY];
      SsPmgApplyRecord pmg_apply = {0};
      const SsNonlinearIterationRecord *failed_iteration =
          solve_record->outcome == SS_NONLINEAR_FAILED_LINEAR &&
                  solve_record->history_count < SS_NONLINEAR_HISTORY_CAPACITY
              ? &solve_record->history[solve_record->history_count]
          : solve_record->history_count == 0U
              ? &solve_record->history[0]
              : &solve_record->history[solve_record->history_count - 1U];
      const int has_pmg_diagnostic =
          hydraulic.nonlinear.pmg != NULL &&
          ss_pmg_last_apply(hydraulic.nonlinear.pmg, &pmg_apply) == SS_OK;
      const int setup_status =
          failed_iteration->linear_setup.attempt_count == 0U
              ? -1
              : (int)failed_iteration->linear_setup.attempts[0].setup_status;
      const int initialization_choice =
          graph.model.mechanics_enabled != 0 ? 0 : (int)hydraulic.initialization.choice;
      const int initialization_primary_choice =
          graph.model.mechanics_enabled != 0
              ? 0
              : (int)hydraulic.initialization.primary_choice;
      const int initialization_fallback =
          graph.model.mechanics_enabled != 0
              ? 0
              : hydraulic.initialization.fallback_attempted;
      const int initialization_primary_outcome =
          graph.model.mechanics_enabled != 0 || hydraulic.primary_outcome_valid == 0
              ? -1
              : (int)hydraulic.primary_outcome;
      const double initialization_boundary_residual =
          graph.model.mechanics_enabled != 0
              ? 0.0
              : hydraulic.initialization.boundary_residual_norm;
      const double initialization_saturated_residual =
          graph.model.mechanics_enabled != 0
              ? 0.0
              : hydraulic.initialization.saturated_residual_norm;
      const int written =
          has_pmg_diagnostic != 0 && pmg_apply.status != SS_OK
              ? snprintf(
                    message, sizeof(message),
                    "%s NLS: out=%d it=%lld last=%lld r=%.8g "
                    "setup=%d solve=%d reason=%d; PMG st=%d pt=%d lev=%zu "
                    "reason=%d it=%lld r=%.8g; init=%d primary=%d "
                    "fallback=%d primary_out=%d boundary=%.8g saturated=%.8g",
                    graph.model.mechanics_enabled != 0 ? "mechanics" : "hydraulic",
                    (int)solve_record->outcome, (long long)solve_record->iterations,
                    (long long)failed_iteration->ordinal,
                    (double)solve_record->final_residual_norm, setup_status,
                    (int)failed_iteration->linear_solve.status,
                    (int)failed_iteration->linear_solve.reason, (int)pmg_apply.status,
                    (int)pmg_apply.failed_point, pmg_apply.failed_level,
                    (int)pmg_apply.last_level_reason,
                    (long long)pmg_apply.last_level_iterations,
                    (double)pmg_apply.last_level_residual_norm, initialization_choice,
                    initialization_primary_choice, initialization_fallback,
                    initialization_primary_outcome, initialization_boundary_residual,
                    initialization_saturated_residual)
              : snprintf(message, sizeof(message),
                         "%s NLS: out=%d it=%lld last=%lld r0=%.8g r=%.8g "
                         "attempts=%zu setup=%d solve=%d reason=%d lit=%lld "
                         "reported=%.8g true=%.8g rhs=%.8g threshold=%.8g "
                         "init=%d primary=%d fallback=%d primary_out=%d "
                         "boundary=%.8g saturated=%.8g",
                         graph.model.mechanics_enabled != 0 ? "mechanics" : "hydraulic",
                         (int)solve_record->outcome,
                         (long long)solve_record->iterations,
                         (long long)failed_iteration->ordinal,
                         (double)solve_record->initial_residual_norm,
                         (double)solve_record->final_residual_norm,
                         failed_iteration->linear_setup.attempt_count, setup_status,
                         (int)failed_iteration->linear_solve.status,
                         (int)failed_iteration->linear_solve.reason,
                         (long long)failed_iteration->linear_solve.iterations,
                         (double)failed_iteration->linear_solve.reported_norm,
                         (double)failed_iteration->linear_solve.true_residual_norm,
                         (double)failed_iteration->linear_solve.rhs_norm,
                         (double)failed_iteration->linear_solve.acceptance_threshold,
                         initialization_choice, initialization_primary_choice,
                         initialization_fallback, initialization_primary_outcome,
                         initialization_boundary_residual,
                         initialization_saturated_residual);
      status = SS_ERROR_NUMERICAL;
      ss_solver_error(error, status, "NONLINEAR_NOT_CONVERGED",
                      written >= 0 && (size_t)written < sizeof(message)
                          ? message
                          : "nonlinear solve did not converge",
                      3U);
    }
  }
  status = ss_status_allreduce_max(graph.communicator, status);
  (void)ss_memory_view(graph.communicator, "solver_after_initial_solve");
  if (status == SS_OK && graph.model.mechanics_enabled != 0) {
    stage = "analysis_policy";
    status =
        ss_solver_mechanics_analysis_policy(&graph, reference_work, &analysis_policy);
    status = ss_status_allreduce_max(graph.communicator, status);
    if (status == SS_OK) {
      ss_solver_mechanics_analysis_problem(&mechanics, &analysis_problem);
      stage = "analysis_monitor";
      status = ss_solver_monitor_attach(
          &monitor, &mechanics, hydraulic_solution == NULL ? NULL : &hydraulic,
          hydraulic_solution, hydraulic_solution == NULL ? NULL : &hydraulic_evidence,
          &analysis_problem);
    }
    if (status == SS_OK) {
      ss_solver_monitor_phase(&monitor, "analysis");
      stage = "analysis";
      (void)PetscLogEventBegin(ss_solver_analysis_event, 0, 0, 0, 0);
      status = ss_analysis_run(&analysis_problem, &analysis_policy, initial, solution,
                               analysis_record);
      (void)PetscLogEventEnd(ss_solver_analysis_event, 0, 0, 0, 0);
    }
    if (status == SS_OK && analysis_record->stop_reason == SS_ANALYSIS_STOP_CANCELLED &&
        analysis_record->point_count == 0U) {
      status = SS_ERROR_CANCELLED;
      ss_solver_error(error, status, "CANCELLED_IN_ANALYSIS",
                      "cancellation observed before the first accepted state", 1U);
    }
    if (status == SS_OK && analysis_record->terminal_success == 0 &&
        analysis_record->stop_reason != SS_ANALYSIS_STOP_CANCELLED &&
        ss_solver_checkpoint_boundary_ready(&graph, analysis_record) == 0) {
      char message[SS_ERROR_MESSAGE_CAPACITY];
      const SsAnalysisAttemptRecord *last_attempt =
          analysis_record->attempt_count == 0U
              ? NULL
              : &analysis_record->attempts[analysis_record->attempt_count - 1U];
      const int written = snprintf(
          message, sizeof(message),
          "analysis failed:r=%s p=%zu a=%zu o=%d n=%lld t=%.4g s=%.3g "
          "l=%d/%d/%lld,%d/%d/%lld "
          "q=%.2e/%.2e/%.2e/%.2e/%.2e/%.2e "
          "c=%d/%.1e,%d/%.1e",
          ss_solver_analysis_stop_reason(analysis_record->stop_reason),
          analysis_record->point_count, analysis_record->attempt_count,
          last_attempt == NULL ? -1 : (int)last_attempt->nonlinear_outcome,
          last_attempt == NULL ? -1LL
                               : (long long)last_attempt->primary_solve.iterations,
          last_attempt == NULL ? NAN : (double)last_attempt->target,
          last_attempt == NULL ? NAN : (double)last_attempt->actual_step,
          last_attempt == NULL
              ? -1
              : (int)last_attempt->primary_solve.first_linear_solve.status,
          last_attempt == NULL
              ? 0
              : (int)last_attempt->primary_solve.first_linear_solve.reason,
          last_attempt == NULL
              ? -1LL
              : (long long)last_attempt->primary_solve.first_linear_solve.iterations,
          last_attempt == NULL
              ? -1
              : (int)last_attempt->primary_solve.second_linear_solve.status,
          last_attempt == NULL
              ? 0
              : (int)last_attempt->primary_solve.second_linear_solve.reason,
          last_attempt == NULL
              ? -1LL
              : (long long)last_attempt->primary_solve.second_linear_solve.iterations,
          last_attempt == NULL
              ? NAN
              : (double)last_attempt->primary_solve.initial_convergence,
          last_attempt == NULL
              ? NAN
              : (double)last_attempt->primary_solve.minimum_convergence,
          last_attempt == NULL ? NAN
                               : (double)last_attempt->primary_solve.final_convergence,
          last_attempt == NULL
              ? NAN
              : (double)last_attempt->primary_solve.final_relative_correction,
          last_attempt == NULL ? NAN : (double)last_attempt->primary_solve.final_alpha,
          last_attempt == NULL
              ? NAN
              : (double)last_attempt->primary_solve.final_regularization,
          (int)mechanics.nonlinear_policy.stopping_criterion,
          (double)mechanics.nonlinear_policy.stopping_tolerance,
          (int)mechanics.initialization_nonlinear_policy.stopping_criterion,
          (double)mechanics.initialization_nonlinear_policy.stopping_tolerance);
      status = SS_ERROR_NUMERICAL;
      ss_solver_error(error, status, "ANALYSIS_NOT_CONVERGED",
                      written >= 0 && (size_t)written < sizeof(message)
                          ? message
                          : "analysis stopped without scientific success",
                      3U);
    }
  }
  (void)ss_memory_view(graph.communicator, "solver_after_analysis");
  if (status == SS_OK) {
    stage = "result_publication";
    ss_solver_monitor_phase(&monitor, "publication");
    (void)PetscLogEventBegin(ss_solver_publish_event, 0, 0, 0, 0);
    status =
        graph.model.mechanics_enabled != 0
            ? ss_solver_publish_mechanics(
                  request, &graph, solution, analysis_record, &mechanics,
                  hydraulic_solution == NULL ? NULL : &hydraulic, hydraulic_solution,
                  hydraulic_solution == NULL ? NULL : &hydraulic_evidence,
                  hydraulic_solution == NULL ? NULL : coupling_snapshot,
                  hydraulic_solution == NULL ? NULL : &coupling_diagnostics, &monitor,
                  summary, error)
            : ss_solver_publish_hydraulic(request, &graph, &hydraulic, solution,
                                          solve_record, &monitor, summary, error);
    (void)PetscLogEventEnd(ss_solver_publish_event, 0, 0, 0, 0);
    if (status == SS_OK && graph.model.mechanics_enabled != 0 &&
        analysis_record->stop_reason == SS_ANALYSIS_STOP_CANCELLED) {
      /* The cancelled result and its checkpoint are published; report it. */
      status = SS_ERROR_CANCELLED;
      ss_solver_error(error, status, "CANCELLED_IN_ANALYSIS",
                      "cancellation observed at an accepted-state safe point; the "
                      "accepted states and a restart checkpoint were published",
                      1U);
    }
  }
  (void)ss_memory_view(graph.communicator, "solver_after_publish");
solver_cleanup:
  (void)PetscLogEventBegin(ss_solver_cleanup_event, 0, 0, 0, 0);
  stage = status == SS_OK ? "cleanup" : stage;
  if (status != SS_OK) {
    failure_deflation_diagnostics = mechanics.last_deflation_diagnostics;
    failure_transaction_status = mechanics.last_transaction_status;
    failure_transaction_open = mechanics.transaction_open;
  }
  const PetscErrorCode solution_error = VecDestroy(&solution);
  const PetscErrorCode hydraulic_error = VecDestroy(&hydraulic_solution);
  const PetscErrorCode initial_error = VecDestroy(&initial);
  if (solution_error != PETSC_SUCCESS || hydraulic_error != PETSC_SUCCESS ||
      initial_error != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
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
  status = ss_status_allreduce_max(graph.communicator, status);
  if (monitor_open != 0) {
    /* An indirect solve can reach the requested count before producing a
     * distinct seed pair. The count alone does not make the run successful. */
    const int failed_seed_prefix =
        status == SS_ERROR_NUMERICAL && analysis_record != NULL &&
        analysis_record->terminal_success == 0 &&
        analysis_record->stop_reason == SS_ANALYSIS_STOP_ACCEPTED_STEP_LIMIT;
    const char *outcome =
        status == SS_ERROR_CANCELLED ? "cancelled"
        : analysis_record != NULL && analysis_record->point_count != 0U &&
                !failed_seed_prefix
            ? ss_solver_analysis_outcome(analysis_record, 0)
        : status == SS_OK
            ? ss_solver_nonlinear_outcome(solve_record == NULL ? SS_NONLINEAR_CONVERGED
                                                               : solve_record->outcome)
            : "failed";
    const SsStatus close_status = ss_solver_monitor_close(&monitor, status, outcome);
    if (status == SS_OK) {
      status = close_status;
    }
    status = ss_status_allreduce_max(graph.communicator, status);
  }
  ss_solver_graph_destroy(&graph);
  if (status != SS_OK && error->valid == 0) {
    char message[SS_ERROR_MESSAGE_CAPACITY];
    const size_t attempts =
        analysis_record == NULL ? 0U : analysis_record->attempt_count;
    const size_t points = analysis_record == NULL ? 0U : analysis_record->point_count;
    const SsAnalysisAttemptRecord *failed_attempt =
        attempts == 0U ? NULL : &analysis_record->attempts[attempts - 1U];
    int written = snprintf(
        message, sizeof(message),
        "native solver failed during %s (%s): attempts=%zu accepted=%zu "
        "transaction_open=%d last_outcome=%d last_successful=%d "
        "last_committed=%d last_rolled_back=%d transaction_status=%d "
        "deflation_columns=%zu deflation_generation=%llu deflation_valid=%d",
        stage, ss_status_name(status), attempts, points, failure_transaction_open,
        failed_attempt == NULL ? -1 : (int)failed_attempt->nonlinear_outcome,
        failed_attempt == NULL ? -1 : failed_attempt->successful,
        failed_attempt == NULL ? -1 : failed_attempt->committed,
        failed_attempt == NULL ? -1 : failed_attempt->rolled_back,
        (int)failure_transaction_status, failure_deflation_diagnostics.active_count,
        (unsigned long long)failure_deflation_diagnostics.transaction_generation,
        failure_deflation_diagnostics.invariants_validated);
    ss_solver_error(error, status, "SOLVER_STAGE_FAILED",
                    written >= 0 && (size_t)written < sizeof(message)
                        ? message
                        : "native solver stage failed",
                    3U);
  }
  free(solve_record);
  free(analysis_record);
  (void)PetscLogEventEnd(ss_solver_cleanup_event, 0, 0, 0, 0);
  return status;
}
