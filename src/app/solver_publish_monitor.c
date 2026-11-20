/* Run monitor: live journal, complete histories, cancellation and recovery.
 *
 * The monitor wraps the mechanics analysis operations without changing the
 * numerical callbacks: accepted states go to the HDF5 accepted-state stream
 * and to the two checkpoint vectors, every completed nonlinear solve is
 * written to the history spool and the journal, and each accepted boundary is
 * journaled (and checkpointed when due).  SIGINT/SIGTERM only set a flag; the
 * analysis and nonlinear solvers poll it collectively at their safe points.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "solver_internal.h"

#include "slope_internal/app/lifecycle.h"

#include <errno.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

static double ss_solver_monitor_finite(PetscReal value) {
  return isfinite((double)value) ? (double)value : 0.0;
}

int ss_solver_cancellation_poll(void *context) {
  (void)context;
  return ss_cancellation_signalled();
}

void ss_solver_monitor_phase(SsSolverMonitor *monitor, const char *phase) {
  char members[160];
  if (monitor == NULL || phase == NULL) {
    return;
  }
  (void)snprintf(members, sizeof(members), "\"phase\":\"%s\"", phase);
  (void)ss_progress_journal_event(monitor->journal, "phase", members);
  ss_progress_journal_say(monitor->journal, "phase %s", phase);
}

static SsStatus ss_solver_monitor_open_history(SsSolverMonitor *monitor) {
  char path[SS_RESULT_PATH_CAPACITY];
  int written;
  SsStatus status;
  if (monitor->history_open != 0) {
    return SS_OK;
  }
  written = snprintf(path, sizeof(path), "%s/history.spool", monitor->run_directory);
  if (written < 0 || (size_t)written >= sizeof(path)) {
    return SS_ERROR_LIMIT;
  }
  status = ss_solver_history_spool_open(&monitor->history, monitor->communicator, path);
  if (status == SS_OK) {
    monitor->emitter = ss_solver_history_spool_emitter(&monitor->history);
    monitor->emitter_attempt = 0U;
    monitor->history_open = 1;
  }
  return status;
}

SsStatus ss_solver_monitor_open(SsSolverMonitor *monitor,
                                const SsSolverRequest *request,
                                const SsSolverGraph *graph) {
  char journal_path[SS_RESULT_PATH_CAPACITY];
  SsProgressJournalSpec journal_spec;
  char members[512];
  int written;
  SsStatus status = SS_OK;
  if (monitor == NULL || request == NULL || graph == NULL ||
      request->result_directory == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(monitor, 0, sizeof(*monitor));
  monitor->request = request;
  monitor->graph = graph;
  monitor->communicator = graph->communicator;
  if (request->options != NULL) {
    monitor->options = *request->options;
  }
  if (monitor->options.checkpoint_keep == 0U) {
    monitor->options.checkpoint_keep = 1U;
  }
  if (MPI_Comm_rank(monitor->communicator, &monitor->rank) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  written = monitor->options.run_directory != NULL
                ? snprintf(monitor->run_directory, sizeof(monitor->run_directory), "%s",
                           monitor->options.run_directory)
                : snprintf(monitor->run_directory, sizeof(monitor->run_directory),
                           "%s.run", request->result_directory);
  if (written < 0 || (size_t)written >= sizeof(monitor->run_directory)) {
    status = SS_ERROR_LIMIT;
  }
  if (status == SS_OK && monitor->rank == 0 &&
      mkdir(monitor->run_directory, 0777) != 0 && errno != EEXIST) {
    status = SS_ERROR_IO;
  }
  status = ss_status_allreduce_max(monitor->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  written = monitor->options.journal_path != NULL
                ? snprintf(journal_path, sizeof(journal_path), "%s",
                           monitor->options.journal_path)
                : snprintf(journal_path, sizeof(journal_path), "%s/journal.jsonl",
                           monitor->run_directory);
  if (written < 0 || (size_t)written >= sizeof(journal_path)) {
    return SS_ERROR_LIMIT;
  }
  memset(&journal_spec, 0, sizeof(journal_spec));
  journal_spec.path = journal_path;
  journal_spec.human_progress = monitor->options.progress;
  journal_spec.communicator = monitor->communicator;
  status = ss_progress_journal_open(&journal_spec, &monitor->journal);
  if (status == SS_OK) {
    int ranks = 1;
    (void)MPI_Comm_size(monitor->communicator, &ranks);
    (void)snprintf(members, sizeof(members),
                   "\"run_id\":\"%s\",\"problem_id\":\"%s\",\"analysis\":\"%s\","
                   "\"ranks\":%d,\"threads\":%d",
                   request->config != NULL ? request->config->run.id : "",
                   request->config != NULL ? request->config->problem.id : "",
                   graph->model.result_analysis_method, ranks,
                   request->requested_threads);
    (void)ss_progress_journal_event(monitor->journal, "start", members);
  }
  status = ss_status_allreduce_max(monitor->communicator, status);
  if (status == SS_OK && (graph->model.mechanics_enabled != 0 ||
                          graph->policy.settings.outputs_krylov_history_limit > 0 ||
                          graph->policy.settings.outputs_pmg_apply_history_limit > 0)) {
    status = ss_solver_monitor_open_history(monitor);
  }
  return status;
}

/* Start a new history entry group when the attempt changes. */
static void ss_solver_monitor_group(SsSolverMonitor *monitor, uint64_t attempt_id) {
  if (monitor->emitter_attempt != attempt_id) {
    monitor->emitter.entry_count = 0U;
    monitor->emitter_attempt = attempt_id;
  }
}

static void ss_solver_monitor_detail(SsSolverMonitor *monitor,
                                     SsResultHistoryRow *row) {
  if (monitor->detail_status == SS_OK) {
    ss_solver_monitor_group(monitor, monitor->detail_attempt);
    row->attempt_id = monitor->detail_attempt;
    row->role = monitor->linear_role;
    row->complete = 1;
    row->reference = (double)monitor->newton_ordinal;
    row->auxiliary = (double)monitor->linear_sequence;
    monitor->detail_status = ss_solver_history_detail(&monitor->emitter, row);
  }
}

static void ss_solver_monitor_krylov(void *context, PetscInt iteration,
                                     PetscReal residual_norm) {
  SsSolverMonitor *monitor = context;
  if (monitor->krylov_details >=
      (uint64_t)monitor->graph->policy.settings.outputs_krylov_history_limit) {
    return;
  }
  ++monitor->krylov_details;
  SsResultHistoryRow row = {
      .kind = SS_RESULT_HISTORY_KRYLOV,
      .ordinal = (uint64_t)iteration,
      .residual = ss_solver_monitor_finite(residual_norm),
      .status = isfinite((double)residual_norm) ? 0 : SS_ERROR_INTEGRITY,
  };
  ss_solver_monitor_detail(monitor, &row);
}

static void ss_solver_monitor_pmg_apply(void *context, const SsPmgApplyRecord *apply) {
  SsSolverMonitor *monitor = context;
  if (monitor->pmg_details >=
      (uint64_t)monitor->graph->policy.settings.outputs_pmg_apply_history_limit) {
    return;
  }
  ++monitor->pmg_details;
  SsResultHistoryRow row = {
      .kind = SS_RESULT_HISTORY_PMG_APPLY,
      .ordinal = apply->sequence,
      .control = (double)apply->level_solve_calls,
      .residual = ss_solver_monitor_finite(apply->last_level_residual_norm),
      .status = (int32_t)apply->status,
      .reason = (int32_t)apply->last_level_reason,
      .tolerance = (double)apply->bridge_bytes,
      .step = (double)apply->operator_apply_calls,
  };
  for (size_t level = 0U; level < SS_PMG_MAX_LEVELS; ++level) {
    row.iterations += apply->level_iterations[level];
  }
  ss_solver_monitor_detail(monitor, &row);
}

static void ss_solver_monitor_linear_started(void *context, PetscInt iteration,
                                             SsNonlinearSystem system) {
  SsSolverMonitor *monitor = context;
  char members[256];
  monitor->newton_ordinal = (uint64_t)iteration;
  monitor->linear_role = system == SS_NONLINEAR_SYSTEM_CONTROL ? SS_RESULT_ROLE_CONTROL
                         : system == SS_NONLINEAR_SYSTEM_RESIDUAL
                             ? SS_RESULT_ROLE_RESIDUAL
                             : monitor->newton_role;
  ++monitor->linear_sequence;
  if (monitor->rank != 0) {
    return;
  }
  (void)snprintf(members, sizeof(members),
                 "\"attempt\":%llu,\"newton_iteration\":%llu,\"role\":%u,"
                 "\"linear_sequence\":%llu",
                 (unsigned long long)monitor->detail_attempt,
                 (unsigned long long)monitor->newton_ordinal, monitor->linear_role,
                 (unsigned long long)monitor->linear_sequence);
  (void)ss_progress_journal_event(monitor->journal, "linear_start", members);
}

static void ss_solver_monitor_linear_completed(void *context, SsStatus status,
                                               const SsLinearSolveRecord *solve) {
  const SsSolverMonitor *monitor = context;
  char members[640];
  if (monitor->rank != 0) {
    return;
  }
  (void)snprintf(members, sizeof(members),
                 "\"attempt\":%llu,\"newton_iteration\":%llu,\"role\":%u,"
                 "\"linear_sequence\":%llu,\"iterations\":%lld,\"status\":%d,"
                 "\"linear_status\":%d,\"reason\":%d,\"reported_residual\":%.17g,"
                 "\"true_residual\":%.17g,\"original_rhs_norm\":%.17g,"
                 "\"stopping_rhs_norm\":%.17g,\"threshold\":%.17g,\"rtol\":%.17g,"
                 "\"cancellation_reductions\":%llu",
                 (unsigned long long)monitor->detail_attempt,
                 (unsigned long long)monitor->newton_ordinal, monitor->linear_role,
                 (unsigned long long)monitor->linear_sequence,
                 (long long)solve->iterations, (int)status, (int)solve->status,
                 (int)solve->reason, ss_solver_monitor_finite(solve->reported_norm),
                 ss_solver_monitor_finite(solve->true_residual_norm),
                 ss_solver_monitor_finite(solve->rhs_norm),
                 ss_solver_monitor_finite(solve->stopping_rhs_norm),
                 ss_solver_monitor_finite(solve->acceptance_threshold),
                 ss_solver_monitor_finite(solve->effective_rtol),
                 (unsigned long long)solve->cancellation_reductions);
  (void)ss_progress_journal_event(monitor->journal, "linear_end", members);
}

static void ss_solver_monitor_newton(void *context,
                                     const SsNonlinearIterationRecord *iteration) {
  const SsSolverMonitor *monitor = context;
  char members[700];
  if (monitor->rank != 0) {
    return;
  }
  const int controlled = iteration->control_linear_solve.solve_count != 0U;
  const SsLinearSolveRecord *solve =
      controlled ? &iteration->residual_linear_solve : &iteration->linear_solve;
  const long long ksp_iterations =
      (long long)solve->iterations +
      (controlled ? (long long)iteration->control_linear_solve.iterations : 0LL);
  (void)snprintf(members, sizeof(members),
                 "\"attempt\":%llu,\"role\":%u,\"iteration\":%lld,"
                 "\"factor\":%.17g,\"residual_before\":%.17g,\"residual_after\":%.17g,"
                 "\"convergence_before\":%.17g,\"convergence_after\":%.17g,"
                 "\"alpha\":%.17g,\"regularization\":%.17g,\"line_trials\":%zu,"
                 "\"direction_norm_relative\":%.17g,\"linear_iterations\":%lld,"
                 "\"linear_residual\":%.17g,\"linear_threshold\":%.17g,"
                 "\"linear_status\":%d,\"linear_reason\":%d",
                 (unsigned long long)monitor->detail_attempt, monitor->newton_role,
                 (long long)iteration->ordinal,
                 ss_solver_monitor_finite(iteration->control_before),
                 ss_solver_monitor_finite(iteration->residual_before),
                 ss_solver_monitor_finite(iteration->residual_after),
                 ss_solver_monitor_finite(iteration->convergence_before),
                 ss_solver_monitor_finite(iteration->convergence_after),
                 ss_solver_monitor_finite(iteration->alpha),
                 ss_solver_monitor_finite(iteration->regularization_after),
                 iteration->line_search.trial_count,
                 ss_solver_monitor_finite(iteration->relative_correction),
                 ksp_iterations, ss_solver_monitor_finite(solve->true_residual_norm),
                 ss_solver_monitor_finite(solve->acceptance_threshold),
                 (int)solve->status, (int)solve->reason);
  (void)ss_progress_journal_event(monitor->journal, "newton", members);
}

void ss_solver_monitor_watch_nonlinear(SsSolverMonitor *monitor,
                                       SsNonlinearProblem *problem, uint32_t role) {
  const SsSolverSettings *settings = &monitor->graph->policy.settings;
  monitor->newton_role = role;
  monitor->detail_attempt =
      role == SS_RESULT_ROLE_SEEPAGE && monitor->graph->model.mechanics_enabled == 0
          ? 1U
          : 0U;
  problem->operations.cancelled = ss_solver_cancellation_poll;
  problem->observer = (SsNonlinearObserver){
      .linear = {.context = monitor},
      .linear_started = ss_solver_monitor_linear_started,
      .linear_completed = ss_solver_monitor_linear_completed,
      .iteration_completed = ss_solver_monitor_newton,
  };
  if (monitor->history_open && settings->outputs_krylov_history_limit > 0) {
    problem->observer.linear.krylov_iteration = ss_solver_monitor_krylov;
  }
  if (monitor->history_open && settings->outputs_pmg_apply_history_limit > 0) {
    problem->observer.linear.pmg_apply = ss_solver_monitor_pmg_apply;
  }
}

void ss_solver_monitor_watch_hydraulic(SsSolverMonitor *monitor,
                                       SsSolverHydraulic *hydraulic) {
  SsOperatorProperties properties;
  ss_solver_monitor_watch_nonlinear(monitor, &hydraulic->nonlinear,
                                    SS_RESULT_ROLE_SEEPAGE);
  if (ss_operator_properties(hydraulic->operator_handle, &properties) == SS_OK) {
    (void)ss_progress_journal_event(
        monitor->journal, "assembly_mode",
        properties.determinism == SS_ASSEMBLY_QUALIFICATION
            ? "\"field\":\"hydraulic\",\"mode\":\"qualification\""
            : "\"field\":\"hydraulic\",\"mode\":\"performance\"");
  }
}

static SsStatus ss_solver_monitor_pmg_delta(SsSolverMonitor *monitor,
                                            SsPmgDiagnostics *delta) {
  SsPmgDiagnostics now;
  memset(delta, 0, sizeof(*delta));
  if (monitor->mechanics == NULL || monitor->mechanics->nonlinear.pmg == NULL ||
      ss_pmg_diagnostics(monitor->mechanics->nonlinear.pmg, &now) != SS_OK) {
    return SS_OK;
  }
  delta->apply_calls = now.apply_calls - monitor->pmg_last.apply_calls;
  delta->failed_apply_calls =
      now.failed_apply_calls - monitor->pmg_last.failed_apply_calls;
  delta->level_solve_calls =
      now.level_solve_calls - monitor->pmg_last.level_solve_calls;
  delta->bridge_bytes = now.bridge_bytes - monitor->pmg_last.bridge_bytes;
  delta->operator_apply_calls =
      now.operator_apply_calls - monitor->pmg_last.operator_apply_calls;
  monitor->pmg_last = now;
  return SS_OK;
}

static uint64_t ss_solver_monitor_ksp_iterations(const SsLinearSolveRecord *solve) {
  return solve->iterations > 0 ? (uint64_t)solve->iterations : 0U;
}

static uint64_t
ss_solver_monitor_linear_iterations(const SsNonlinearSolveRecord *solve) {
  uint64_t total = 0U;
  for (size_t index = 0U;
       index < solve->history_count && index < SS_NONLINEAR_HISTORY_CAPACITY; ++index) {
    const SsNonlinearIterationRecord *iteration = &solve->history[index];
    if (iteration->control_linear_solve.solve_count != 0U) {
      total += ss_solver_monitor_ksp_iterations(&iteration->control_linear_solve) +
               ss_solver_monitor_ksp_iterations(&iteration->residual_linear_solve);
    } else {
      total += ss_solver_monitor_ksp_iterations(&iteration->linear_solve);
    }
  }
  return total;
}

static SsStatus ss_solver_monitor_solve(SsSolverMonitor *monitor,
                                        const SsNonlinearSolveRecord *solve,
                                        uint64_t attempt_id, uint32_t role,
                                        const char *label) {
  SsPmgDiagnostics delta;
  char members[512];
  SsStatus status;
  ss_solver_monitor_group(monitor, attempt_id);
  status = ss_solver_monitor_pmg_delta(monitor, &delta);
  if (status == SS_OK && monitor->history_open != 0) {
    status = monitor->detail_status;
    if (status == SS_OK) {
      status =
          ss_solver_history_solve(&monitor->emitter, solve, attempt_id, role, &delta);
    }
    status = ss_status_allreduce_max(monitor->communicator, status);
    if (status == SS_OK) {
      status = ss_solver_history_spool_commit(&monitor->history);
    }
  }
  (void)snprintf(members, sizeof(members),
                 "\"solve\":\"%s\",\"attempt\":%llu,\"role\":%u,\"outcome\":\"%s\","
                 "\"newton_iterations\":%lld,\"linear_iterations\":%llu,"
                 "\"initial_residual\":%.17g,\"final_residual\":%.17g,"
                 "\"final_convergence\":%.17g",
                 label, (unsigned long long)attempt_id, role,
                 ss_solver_nonlinear_outcome(solve->outcome),
                 (long long)solve->iterations,
                 (unsigned long long)ss_solver_monitor_linear_iterations(solve),
                 ss_solver_monitor_finite(solve->initial_residual_norm),
                 ss_solver_monitor_finite(solve->final_residual_norm),
                 ss_solver_monitor_finite(solve->final_convergence));
  (void)ss_progress_journal_event(monitor->journal, "solve", members);
  return status;
}

SsStatus ss_solver_monitor_record_solve(SsSolverMonitor *monitor,
                                        const SsNonlinearSolveRecord *solve,
                                        uint32_t role) {
  if (monitor == NULL || solve == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return ss_solver_monitor_solve(monitor, solve, 0U, role,
                                 role == SS_RESULT_ROLE_SEEPAGE ? "seepage"
                                                                : "auxiliary");
}

SsStatus ss_solver_monitor_record_linear(SsSolverMonitor *monitor,
                                         const SsLinearSolveRecord *solve,
                                         uint32_t role) {
  char members[256];
  SsStatus status = SS_OK;
  if (monitor == NULL || solve == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  ss_solver_monitor_group(monitor, 0U);
  if (monitor->history_open != 0) {
    status = monitor->detail_status;
    if (status == SS_OK) {
      status = ss_solver_history_linear_solve(&monitor->emitter, solve, 0U, role);
    }
    status = ss_status_allreduce_max(monitor->communicator, status);
  }
  (void)snprintf(members, sizeof(members),
                 "\"solve\":\"elastic_predictor\",\"status\":%d,\"reason\":%d,"
                 "\"linear_iterations\":%lld,\"true_residual\":%.17g",
                 (int)solve->status, (int)solve->reason, (long long)solve->iterations,
                 ss_solver_monitor_finite(solve->true_residual_norm));
  (void)ss_progress_journal_event(monitor->journal, "solve", members);
  return status;
}

/* ---- analysis operation interposition ------------------------------------ */

static SsStatus ss_solver_monitor_fixed(void *context, PetscReal factor, Vec incoming,
                                        Vec solution, SsNonlinearSolveRecord *record) {
  SsSolverMonitor *monitor = (SsSolverMonitor *)context;
  return monitor->base_operations.solve_fixed(monitor->base_context, factor, incoming,
                                              solution, record);
}

static SsStatus ss_solver_monitor_work(void *context, SsAnalysisMethod method,
                                       PetscReal target_work, PetscReal incoming_factor,
                                       Vec incoming, Vec solution,
                                       PetscReal *accepted_factor,
                                       SsNonlinearSolveRecord *record) {
  SsSolverMonitor *monitor = (SsSolverMonitor *)context;
  return monitor->base_operations.solve_work_controlled(
      monitor->base_context, method, target_work, incoming_factor, incoming, solution,
      accepted_factor, record);
}

static SsStatus ss_solver_monitor_potential(void *context, PetscReal factor, Vec state,
                                            PetscReal *potential) {
  SsSolverMonitor *monitor = (SsSolverMonitor *)context;
  return monitor->base_operations.total_potential(monitor->base_context, factor, state,
                                                  potential);
}

static SsStatus ss_solver_monitor_maximum(void *context, Vec state,
                                          PetscReal *maximum) {
  SsSolverMonitor *monitor = (SsSolverMonitor *)context;
  return monitor->base_operations.displacement_maximum(monitor->base_context, state,
                                                       maximum);
}

/*
 * Keep the current and previous accepted states for checkpoints (always, so a
 * cancelled run can publish a restartable checkpoint) and stream the state to
 * the accepted-state file when every accepted state is published.
 */
static SsStatus ss_solver_monitor_capture(void *context, size_t accepted_ordinal,
                                          Vec state) {
  SsSolverMonitor *monitor = (SsSolverMonitor *)context;
  SsSolverMechanics *mechanics = monitor->mechanics;
  SsStatus status = SS_OK;
  if (mechanics == NULL || state == NULL) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else if (mechanics->checkpoint_state_count == 0U) {
    mechanics->checkpoint_first_ordinal = accepted_ordinal;
  } else if (accepted_ordinal !=
             mechanics->checkpoint_first_ordinal + mechanics->checkpoint_state_count) {
    status = SS_ERROR_INTEGRITY;
  }
  status = ss_status_allreduce_max(monitor->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  if (mechanics->checkpoint_current_state == NULL) {
    Vec *destinations[] = {&mechanics->checkpoint_current_state,
                           &mechanics->checkpoint_previous_state};
    status = ss_vec_duplicate_collective(state, destinations, 2U);
    if (status != SS_OK) {
      (void)VecDestroy(&mechanics->checkpoint_current_state);
      (void)VecDestroy(&mechanics->checkpoint_previous_state);
      return status;
    }
  }
  if ((mechanics->checkpoint_state_count != 0U &&
       VecCopy(mechanics->checkpoint_current_state,
               mechanics->checkpoint_previous_state) != PETSC_SUCCESS) ||
      VecCopy(state, mechanics->checkpoint_current_state) != PETSC_SUCCESS ||
      (mechanics->checkpoint_state_count == 0U &&
       VecCopy(state, mechanics->checkpoint_previous_state) != PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(monitor->communicator, status);
  if (status == SS_OK &&
      monitor->graph->policy.output_frequency == SS_SOLVER_OUTPUT_ACCEPTED_STATE) {
    if (monitor->states.open == 0) {
      char path[SS_RESULT_PATH_CAPACITY];
      const int written =
          snprintf(path, sizeof(path), "%s/accepted-states.h5", monitor->run_directory);
      status = ss_status_allreduce_max(
          monitor->communicator,
          written < 0 || (size_t)written >= sizeof(path) ? SS_ERROR_LIMIT : SS_OK);
      if (status == SS_OK) {
        status = ss_solver_state_spool_open(&monitor->states, state, path);
      }
    }
    if (status == SS_OK) {
      status = ss_solver_state_spool_append(&monitor->states, state);
    }
  }
  if (status == SS_OK) {
    ++mechanics->checkpoint_state_count;
  }
  return status;
}

static SsStatus ss_solver_monitor_begin(void *context, size_t attempt_ordinal,
                                        size_t parent_accepted_ordinal) {
  SsSolverMonitor *monitor = (SsSolverMonitor *)context;
  char members[160];
  ++monitor->attempts;
  monitor->detail_attempt = (uint64_t)attempt_ordinal + 1U;
  monitor->newton_role = SS_RESULT_ROLE_PRIMARY;
  (void)snprintf(members, sizeof(members), "\"attempt\":%llu,\"parent_state\":%llu",
                 (unsigned long long)attempt_ordinal + 1U,
                 (unsigned long long)parent_accepted_ordinal);
  (void)ss_progress_journal_event(monitor->journal, "attempt", members);
  return monitor->base_operations.begin_attempt(monitor->base_context, attempt_ordinal,
                                                parent_accepted_ordinal);
}

static SsStatus ss_solver_monitor_commit(void *context, size_t attempt_ordinal,
                                         size_t parent_accepted_ordinal) {
  SsSolverMonitor *monitor = (SsSolverMonitor *)context;
  return monitor->base_operations.commit_attempt(monitor->base_context, attempt_ordinal,
                                                 parent_accepted_ordinal);
}

static SsStatus ss_solver_monitor_rollback(void *context, size_t attempt_ordinal,
                                           size_t parent_accepted_ordinal) {
  SsSolverMonitor *monitor = (SsSolverMonitor *)context;
  char members[96];
  (void)snprintf(members, sizeof(members), "\"attempt\":%llu",
                 (unsigned long long)attempt_ordinal + 1U);
  (void)ss_progress_journal_event(monitor->journal, "attempt_rolled_back", members);
  return monitor->base_operations.rollback_attempt(
      monitor->base_context, attempt_ordinal, parent_accepted_ordinal);
}

static void ss_solver_monitor_solve_started(void *observer, SsAnalysisSolveRole role,
                                            PetscReal target) {
  SsSolverMonitor *monitor = observer;
  char members[192];
  monitor->newton_role = role == SS_ANALYSIS_SOLVE_SECONDARY ? SS_RESULT_ROLE_SECONDARY
                                                             : SS_RESULT_ROLE_PRIMARY;
  if (monitor->rank == 0) {
    (void)snprintf(members, sizeof(members),
                   "\"attempt\":%llu,\"role\":%u,\"target\":%.17g",
                   (unsigned long long)monitor->detail_attempt, monitor->newton_role,
                   ss_solver_monitor_finite(target));
    (void)ss_progress_journal_event(monitor->journal, "newton_start", members);
  }
}

static SsStatus
ss_solver_monitor_solve_completed(void *observer, size_t attempt_ordinal,
                                  SsAnalysisSolveRole role,
                                  const SsNonlinearSolveRecord *record) {
  SsSolverMonitor *monitor = (SsSolverMonitor *)observer;
  const SsStatus status = ss_solver_monitor_solve(
      monitor, record, (uint64_t)attempt_ordinal + 1U,
      role == SS_ANALYSIS_SOLVE_SECONDARY ? SS_RESULT_ROLE_SECONDARY
                                          : SS_RESULT_ROLE_PRIMARY,
      role == SS_ANALYSIS_SOLVE_SECONDARY ? "sensitivity" : "attempt");
  return status;
}

static int ss_solver_monitor_checkpoint_due(SsSolverMonitor *monitor) {
  int due = 0;
  if (monitor->rank == 0) {
    if (monitor->options.checkpoint_every != 0U &&
        monitor->accepted_states % monitor->options.checkpoint_every == 0U) {
      due = 1;
    }
    if (monitor->options.checkpoint_seconds > 0.0 &&
        MPI_Wtime() - monitor->last_checkpoint_time >=
            monitor->options.checkpoint_seconds) {
      due = 1;
    }
  }
  if (MPI_Bcast(&due, 1, MPI_INT, 0, monitor->communicator) != MPI_SUCCESS) {
    return 0;
  }
  return due;
}

static SsStatus ss_solver_monitor_boundary(void *observer,
                                           const SsAnalysisRecord *record) {
  SsSolverMonitor *monitor = (SsSolverMonitor *)observer;
  const SsAnalysisPointRecord *point;
  char members[512];
  SsStatus status = SS_OK;
  if (record == NULL || record->point_count == 0U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  point = &record->points[record->point_count - 1U];
  monitor->boundary = (SsSolverBoundary){
      .point_count = record->point_count,
      .next_step = record->next_step,
      .first_control = record->first_control,
      .next_operation = record->next_operation,
      .work_efficiency_previous_iterations =
          record->work_efficiency_previous_iterations,
      .work_efficiency_elasticity = record->work_efficiency_elasticity,
      .work_efficiency_elasticity_ready = record->work_efficiency_elasticity_ready,
  };
  monitor->boundary_valid = 1;
  ++monitor->accepted_states;
  (void)snprintf(members, sizeof(members),
                 "\"state\":%llu,\"accepted_ordinal\":%llu,\"factor\":%.17g,"
                 "\"work\":%.17g,\"control\":%.17g,\"displacement_maximum\":%.17g,"
                 "\"attempts\":%llu,\"next_step\":%.17g",
                 (unsigned long long)record->point_count,
                 (unsigned long long)point->accepted_ordinal,
                 ss_solver_monitor_finite(point->factor),
                 ss_solver_monitor_finite(point->work),
                 ss_solver_monitor_finite(point->control),
                 ss_solver_monitor_finite(point->displacement_maximum),
                 (unsigned long long)record->attempt_count,
                 ss_solver_monitor_finite(record->next_step));
  (void)ss_progress_journal_event(monitor->journal, "accepted", members);
  ss_progress_journal_say(monitor->journal,
                          "accepted state %zu: factor=%.8g work=%.6g u_max=%.4g "
                          "attempts=%zu",
                          record->point_count, ss_solver_monitor_finite(point->factor),
                          ss_solver_monitor_finite(point->work),
                          ss_solver_monitor_finite(point->displacement_maximum),
                          record->attempt_count);
  if (monitor->signal_after_accepted > 0 &&
      monitor->accepted_states == (uint64_t)monitor->signal_after_accepted &&
      monitor->rank == 0) {
    /* Test hook: deliver a real SIGTERM to rank zero only. */
    (void)raise(SIGTERM);
  }
  if ((record->next_operation == SS_ANALYSIS_NEXT_SECOND_SEED ||
       record->next_operation == SS_ANALYSIS_NEXT_CONTINUATION_ATTEMPT) &&
      ss_solver_monitor_checkpoint_due(monitor) != 0) {
    status = ss_solver_checkpoint_boundary_write(monitor, record);
  }
  return status;
}

SsStatus ss_solver_monitor_attach(SsSolverMonitor *monitor,
                                  SsSolverMechanics *mechanics,
                                  SsSolverHydraulic *hydraulic, Vec hydraulic_solution,
                                  const SsSolverHydraulicEvidence *hydraulic_evidence,
                                  SsAnalysisProblem *problem) {
  if (monitor == NULL || mechanics == NULL || problem == NULL ||
      problem->operations.solve_fixed == NULL ||
      problem->operations.solve_work_controlled == NULL ||
      problem->operations.total_potential == NULL ||
      problem->operations.displacement_maximum == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  monitor->mechanics = mechanics;
  monitor->hydraulic = hydraulic;
  monitor->hydraulic_solution = hydraulic_solution;
  monitor->hydraulic_evidence = hydraulic_evidence;
  monitor->base_operations = problem->operations;
  monitor->base_context = problem->context;
  monitor->last_checkpoint_time = MPI_Wtime();
  SsSolverOptionValue setting;
  if (ss_solver_expert_option(SS_SOLVER_EXPERT_test_signal_after_accepted, &setting) ==
      SS_OK) {
    monitor->signal_after_accepted = setting.integer;
  }
  if (mechanics->nonlinear.pmg != NULL) {
    (void)ss_pmg_diagnostics(mechanics->nonlinear.pmg, &monitor->pmg_last);
  }
  problem->operations.solve_fixed = ss_solver_monitor_fixed;
  problem->operations.solve_work_controlled = ss_solver_monitor_work;
  problem->operations.total_potential = ss_solver_monitor_potential;
  problem->operations.displacement_maximum = ss_solver_monitor_maximum;
  problem->operations.capture_accepted_state = ss_solver_monitor_capture;
  if (monitor->base_operations.begin_attempt != NULL) {
    problem->operations.begin_attempt = ss_solver_monitor_begin;
    problem->operations.commit_attempt = ss_solver_monitor_commit;
    problem->operations.rollback_attempt = ss_solver_monitor_rollback;
  }
  problem->operations.cancelled = ss_solver_cancellation_poll;
  problem->context = monitor;
  problem->observer.solve_started = ss_solver_monitor_solve_started;
  problem->observer.solve_completed = ss_solver_monitor_solve_completed;
  problem->observer.accepted_boundary = ss_solver_monitor_boundary;
  problem->observer.context = monitor;
  /* Newton-iteration safe points of the mechanics solver. */
  ss_solver_monitor_watch_nonlinear(monitor, &mechanics->nonlinear,
                                    SS_RESULT_ROLE_PRIMARY);
  return SS_OK;
}

SsStatus ss_solver_monitor_close(SsSolverMonitor *monitor, SsStatus status,
                                 const char *outcome) {
  char members[256];
  SsStatus close_status = SS_OK;
  if (monitor == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (monitor->journal != NULL) {
    (void)snprintf(members, sizeof(members),
                   "\"status\":%d,\"outcome\":\"%s\",\"accepted_states\":%llu,"
                   "\"attempts\":%llu,\"checkpoints\":%llu",
                   (int)status, outcome == NULL ? "none" : outcome,
                   (unsigned long long)monitor->accepted_states,
                   (unsigned long long)monitor->attempts,
                   (unsigned long long)monitor->checkpoint_count);
    (void)ss_progress_journal_event(monitor->journal, "terminal", members);
    if (monitor->accepted_states != 0U) {
      ss_progress_journal_say(monitor->journal, "finished: %s (%llu accepted states)",
                              outcome == NULL ? "none" : outcome,
                              (unsigned long long)monitor->accepted_states);
    } else {
      ss_progress_journal_say(monitor->journal, "finished: %s",
                              outcome == NULL ? "none" : outcome);
    }
  }
  if (monitor->history_open != 0 &&
      ss_solver_history_spool_close(&monitor->history, 1) != SS_OK) {
    close_status = SS_ERROR_IO;
  }
  monitor->history_open = 0;
  if (monitor->states.open != 0 &&
      ss_solver_state_spool_close(&monitor->states, 1) != SS_OK) {
    close_status = SS_ERROR_IO;
  }
  if (ss_progress_journal_close(&monitor->journal) != SS_OK) {
    close_status = SS_ERROR_IO;
  }
  return close_status;
}
