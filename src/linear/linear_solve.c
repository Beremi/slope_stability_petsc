/* Linear solve, explicit residual checks and Krylov capture.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "linear_internal.h"
#include "slope_internal/assembly/assembly.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/core/util.h"
#include "slope_internal/diagnostics/diagnostics.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void ss_linear_krylov_capture_clear(SsLinearKrylovCaptureBuffer *capture) {
  capture->enabled = 0;
  capture->count = 0U;
}

void ss_linear_krylov_capture_destroy(SsLinearKrylovCaptureBuffer *capture) {
  for (size_t index = 0U; index < capture->allocated; ++index) {
    (void)VecDestroy(&capture->vectors[index]);
  }
  free(capture->vectors);
  memset(capture, 0, sizeof(*capture));
}

PetscErrorCode ss_linear_krylov_capture_append(SsLinearKrylovCaptureBuffer *capture,
                                               Vec direction) {
  PetscFunctionBegin;
  if (capture == NULL || !capture->enabled) {
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(capture->count < capture->allocated, PETSC_COMM_SELF, PETSC_ERR_MEM,
             "Krylov capture exceeded its budget");
  PetscCall(VecCopy(direction, capture->vectors[capture->count]));
  ++capture->count;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static SsStatus ss_linear_true_residual(SsLinearSolver *solver, Vec rhs, Vec solution,
                                        PetscReal *rhs_norm, PetscReal *residual_norm) {
  SsStatus status =
      ss_linear_petsc_status(solver->communicator, VecNorm(rhs, NORM_2, rhs_norm));
  if (status == SS_OK) {
    status = ss_status_allreduce_max(
        solver->communicator,
        MatMult(solver->operator_handle, solution, solver->residual) == PETSC_SUCCESS &&
                VecAYPX(solver->residual, -1.0, rhs) == PETSC_SUCCESS
            ? SS_OK
            : SS_ERROR_INTERNAL);
  }
  if (status == SS_OK) {
    status = ss_linear_petsc_status(solver->communicator,
                                    VecNorm(solver->residual, NORM_2, residual_norm));
  }
  return status;
}

/* Check at a shared iteration number, never branch on a rank-local signal
 * history. Retain the actual convergence test and its owned context. */
typedef struct {
  SsLinearSolver *solver;
  KSPConvergenceTestFn *test;
  PetscCtxDestroyFn *destroy;
  void *context;
  PetscInt last_polled;
  uint64_t reductions;
  int installed;
} SsLinearConvergence;

static PetscErrorCode ss_linear_cancellable_convergence(KSP ksp, PetscInt iteration,
                                                        PetscReal residual,
                                                        KSPConvergedReason *reason,
                                                        void *context) {
  SsLinearConvergence *convergence = context;
  SsLinearSolver *solver = convergence->solver;
  PetscFunctionBegin;
  if (iteration > convergence->last_polled && iteration % 10 == 0) {
    convergence->last_polled = iteration;
    ++convergence->reductions;
    int cancelled = solver->cancelled(solver->cancellation_context) != 0;
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &cancelled, 1, MPI_INT, MPI_MAX,
                               solver->communicator));
    if (cancelled) {
      *reason = KSP_DIVERGED_USER;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscCall(convergence->test(ksp, iteration, residual, reason, convergence->context));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ss_linear_install_cancellation(SsLinearConvergence *convergence) {
  KSP ksp = convergence->solver->ksp;
  PetscFunctionBegin;
  PetscCall(KSPGetAndClearConvergenceTest(
      ksp, &convergence->test, &convergence->context, &convergence->destroy));
  const PetscErrorCode status =
      KSPSetConvergenceTest(ksp, ss_linear_cancellable_convergence, convergence, NULL);
  if (status == PETSC_SUCCESS) {
    convergence->installed = 1;
  } else {
    (void)KSPSetConvergenceTest(ksp, convergence->test, convergence->context,
                                convergence->destroy);
  }
  PetscFunctionReturn(status);
}

static PetscErrorCode ss_linear_restore_convergence(SsLinearConvergence *convergence) {
  return !convergence->installed
             ? PETSC_SUCCESS
             : KSPSetConvergenceTest(convergence->solver->ksp, convergence->test,
                                     convergence->context, convergence->destroy);
}

void ss_linear_set_cancellation(SsLinearSolver *solver, SsLinearCancellation poll,
                                void *context) {
  if (solver != NULL) {
    solver->cancelled = poll;
    solver->cancellation_context = context;
  }
}

void ss_linear_set_observer(SsLinearSolver *solver, const SsLinearObserver *observer) {
  if (solver != NULL) {
    solver->observer = observer == NULL ? (SsLinearObserver){0} : *observer;
  }
}

PetscErrorCode ss_linear_krylov_monitor(KSP ksp, PetscInt iteration,
                                        PetscReal residual_norm, void *context) {
  SsLinearSolver *solver = context;
  (void)ksp;
  if (solver->observer.krylov_iteration != NULL &&
      iteration > solver->last_observed_iteration) {
    solver->last_observed_iteration = iteration;
    solver->observer.krylov_iteration(solver->observer.context, iteration,
                                      residual_norm);
  }
  return PETSC_SUCCESS;
}

static PetscErrorCode ss_linear_ksp_solve(SsLinearSolver *solver, Vec rhs, Vec solution,
                                          PetscBool require_initial_correction,
                                          KSPConvergedReason *apply_failure,
                                          uint64_t *cancellation_reductions) {
  PetscInt minimum_iterations = 0;
  PetscBool enforce_nonzero_update = PETSC_FALSE;
  PC pc = NULL;
  SsLinearConvergence convergence = {.solver = solver};
  PetscFunctionBegin;
  const MPI_Comm comm = solver->communicator;
  PetscErrorCode local = PETSC_SUCCESS;
  solver->projected_context.failed = 0;
  if (solver->selected_policy.pc == SS_LINEAR_PC_PMG) {
    local = KSPGetPC(solver->ksp, &pc);
    if (local == PETSC_SUCCESS) {
      local = PCSetFailedReason(pc, PC_NOERROR);
    }
  }
  solver->captured_krylov.enabled =
      solver->krylov_capture == SS_LINEAR_KRYLOV_CAPTURE_FULL_CORRECTION_SPACE;
  if (local == PETSC_SUCCESS && require_initial_correction == PETSC_TRUE) {
    local = KSPGetMinimumIterations(solver->ksp, &minimum_iterations);
    if (local == PETSC_SUCCESS && minimum_iterations < 1) {
      local = KSPSetMinimumIterations(solver->ksp, 1);
      enforce_nonzero_update = local == PETSC_SUCCESS ? PETSC_TRUE : PETSC_FALSE;
    }
  }
  if (local == PETSC_SUCCESS && solver->cancelled != NULL) {
    local = ss_linear_install_cancellation(&convergence);
  }
  if (ss_linear_petsc_status(comm, local) != SS_OK) {
    (void)ss_linear_restore_convergence(&convergence);
    if (enforce_nonzero_update) {
      (void)KSPSetMinimumIterations(solver->ksp, minimum_iterations);
    }
    solver->captured_krylov.enabled = 0;
    PetscFunctionReturn(PETSC_ERR_LIB);
  }
  solver->last_observed_iteration = -1;
  ss_pmg_set_apply_observer(solver->pmg, solver->observer.pmg_apply,
                            solver->observer.context);
  PetscErrorCode status = KSPSolve(solver->ksp, rhs, solution);
  ss_pmg_set_apply_observer(solver->pmg, NULL, NULL);
  *cancellation_reductions = convergence.reductions;
  if (ss_linear_restore_convergence(&convergence) != PETSC_SUCCESS) {
    status = PETSC_ERR_LIB;
  }
  if (enforce_nonzero_update == PETSC_TRUE) {
    if (KSPSetMinimumIterations(solver->ksp, minimum_iterations) != PETSC_SUCCESS) {
      status = PETSC_ERR_LIB;
    }
  }
  PCFailedReason pc_reason = PC_NOERROR;
  if (pc != NULL && (PCGetFailedReason(pc, &pc_reason) != PETSC_SUCCESS ||
                     PCSetFailedReason(pc, PC_NOERROR) != PETSC_SUCCESS)) {
    status = PETSC_ERR_LIB;
  }
  /* Reuse the post-solve agreement for shell failures, including a failed
   * rank with no owned rows or a KSP type that does not check a norm. */
  const int action_failed =
      ss_operator_matrix_free_take_action_failure(solver->operator_handle);
  int failures[] = {status != PETSC_SUCCESS, pc_reason != PC_NOERROR,
                    solver->projected_context.failed || action_failed,
                    solver->cancelled != NULL &&
                        solver->cancelled(solver->cancellation_context) != 0};
  if (MPI_Allreduce(MPI_IN_PLACE, failures, 4, MPI_INT, MPI_MAX, comm) != MPI_SUCCESS) {
    failures[0] = 1;
  }
  status = failures[0] ? PETSC_ERR_LIB : PETSC_SUCCESS;
  if (failures[3]) {
    *apply_failure = KSP_DIVERGED_USER;
  } else if (failures[1]) {
    *apply_failure = KSP_DIVERGED_PC_FAILED;
  } else if (failures[2]) {
    *apply_failure = KSP_DIVERGED_NANORINF;
  }
  /* With the preconditioned or natural norm, FCG applies B once to form its
   * initial Z and once after every correction.  The final application is used
   * only by the convergence test, so omit it from the retained correction
   * span.  With the unpreconditioned norm PETSc checks ||r|| before applying B;
   * every captured Z then belongs to an actual correction and must be retained. */
  if (solver->captured_krylov.enabled != 0 &&
      solver->selected_policy.method == SS_LINEAR_METHOD_FCG &&
      (ss_linear_norm_type(&solver->selected_policy) == KSP_NORM_PRECONDITIONED ||
       ss_linear_norm_type(&solver->selected_policy) == KSP_NORM_NATURAL) &&
      solver->captured_krylov.count != 0U) {
    --solver->captured_krylov.count;
  }
  solver->captured_krylov.enabled = 0;
  PetscFunctionReturn(status);
}

static SsStatus ss_linear_solve_impl(SsLinearSolver *solver, Vec rhs, Vec solution,
                                     PetscReal relative_tolerance,
                                     SsLinearSolveRecord *record) {
  Vec solve_rhs = rhs;
  KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
  PetscInt iterations = 0;
  PetscReal reported_norm = 0.0;
  PetscReal rhs_norm = 0.0;
  PetscReal true_residual_norm = 0.0;
  PetscReal input_norm = 0.0;
  PetscReal initial_norm = 0.0;
  PetscReal projected_norm = 0.0;
  PetscReal candidate_norm = NAN;
  PetscReal solve_rtol = 0.0;
  PetscReal solve_atol = 0.0;
  PetscReal initial_correction_threshold = 0.0;
  PetscReal initial_stopping_norm = NAN;
  PetscReal rhs_scale = 1.0;
  const PetscReal *residual_history = NULL;
  PetscInt residual_history_count = 0;
  PetscErrorCode petsc_status;
  PetscErrorCode restore_status = PETSC_SUCCESS;
  int projected_stopping;
  int preconditioned_stopping;
  int natural_stopping;
  int aorth_projected_pc = 0;
  SsLinearPolicy acceptance_policy;
  SsStatus status;
  if (solver == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (rhs == NULL || solution == NULL || record == NULL) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else if (!solver->setup_complete || solver->ksp == NULL ||
             solver->captured_krylov.count != 0U) {
    status = SS_ERROR_INVALID_STATE;
  } else if (!ss_is_finite(relative_tolerance) || relative_tolerance <= 0.0) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else {
    PetscInt a_local_rows, a_local_cols, b_local, x_local;
    if (MatGetLocalSize(solver->operator_handle, &a_local_rows, &a_local_cols) !=
            PETSC_SUCCESS ||
        VecGetLocalSize(rhs, &b_local) != PETSC_SUCCESS ||
        VecGetLocalSize(solution, &x_local) != PETSC_SUCCESS ||
        b_local != a_local_rows || x_local != a_local_cols) {
      status = SS_ERROR_TYPE_MISMATCH;
    } else {
      status = SS_OK;
    }
  }
  status = ss_status_allreduce_max(solver->communicator, status);
  if (record != NULL) {
    memset(record, 0, sizeof(*record));
    record->input_guess_norm = NAN;
    record->coarse_initial_residual_norm = NAN;
    record->candidate_initial_residual_norm = NAN;
    record->initial_residual_norm = NAN;
  }
  if (status != SS_OK) {
    if (record != NULL) {
      record->status = SS_LINEAR_SOLVE_SETUP_FAILED;
    }
    return status;
  }
  record->effective_rtol = relative_tolerance;
  record->recycled_dimension_supplied = solver->recycled_dimension_supplied;
  record->deflation_dimension_supplied = solver->deflation_dimension_supplied;
  projected_stopping =
      solver->deflation != NULL &&
      solver->selected_policy.stopping_condition == SS_LINEAR_STOP_PROJECTED_RHS_PETSC;
  preconditioned_stopping = solver->selected_policy.stopping_condition ==
                            SS_LINEAR_STOP_PRECONDITIONED_RESIDUAL_ORIGINAL_RHS;
  natural_stopping = solver->selected_policy.stopping_condition ==
                     SS_LINEAR_STOP_NATURAL_RESIDUAL_INITIAL;
  if (solver->deflation != NULL) {
    SsDeflationProjector projector = SS_DEFLATION_PROJECTOR_GENERAL_LEFT_RIGHT;
    status = ss_deflation_projector(solver->deflation, &projector);
    if (status != SS_OK) {
      record->status = SS_LINEAR_SOLVE_SETUP_FAILED;
      return status;
    }
    aorth_projected_pc =
        projector == SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL ? 1 : 0;
  }
  if (ss_linear_petsc_status(solver->communicator, VecNorm(rhs, NORM_2, &input_norm)) !=
          SS_OK ||
      ss_linear_petsc_status(solver->communicator,
                             VecNorm(solution, NORM_2, &initial_norm)) != SS_OK ||
      !ss_is_finite(input_norm) || !ss_is_finite(initial_norm)) {
    record->status = SS_LINEAR_SOLVE_NONFINITE;
    return SS_OK;
  }
  record->input_guess_norm = initial_norm;
  if (solver->properties.nonlinear_hydraulic != 0 && input_norm > 0.0) {
    rhs_scale = input_norm;
    if (ss_status_allreduce_max(solver->communicator,
                                VecCopy(rhs, solver->residual) == PETSC_SUCCESS &&
                                        VecScale(solver->residual, 1.0 / rhs_scale) ==
                                            PETSC_SUCCESS
                                    ? SS_OK
                                    : SS_ERROR_INTERNAL) != SS_OK) {
      record->status = SS_LINEAR_SOLVE_NONFINITE;
      return SS_ERROR_INTERNAL;
    }
    solve_rhs = solver->residual;
  }
  solve_atol = relative_tolerance * input_norm;
  if (solve_atol < solver->selected_policy.atol) {
    solve_atol = solver->selected_policy.atol;
  }
  solve_atol /= rhs_scale;
  initial_correction_threshold =
      fmax(solver->selected_policy.atol, relative_tolerance * input_norm);
  if (projected_stopping != 0 || natural_stopping != 0) {
    solve_atol = solver->selected_policy.atol / rhs_scale;
  }
  /* PETSc normally scales rtol by the initial preconditioned residual.  The
   * requested FCG contract instead uses the unpreconditioned original RHS, so
   * express that fixed threshold through atol and disable internal rtol. */
  solve_rtol = preconditioned_stopping != 0 ? 0.0 : relative_tolerance;
  if (!ss_is_finite(solve_rtol) || !ss_is_finite(solve_atol) || solve_rtol < 0.0 ||
      solve_atol < 0.0 ||
      ss_linear_petsc_status(
          solver->communicator,
          KSPSetTolerances(solver->ksp, solve_rtol, solve_atol,
                           solver->selected_policy.dtol,
                           solver->selected_policy.maximum_iterations)) != SS_OK) {
    record->status = SS_LINEAR_SOLVE_SETUP_FAILED;
    return SS_ERROR_INTERNAL;
  }
  if (solver->deflation != NULL) {
    if (aorth_projected_pc != 0) {
      const int use_input_guess = solver->selected_policy.initial_guess_policy !=
                                  SS_LINEAR_INITIAL_GUESS_COARSE_ONLY;
      if (use_input_guess != 0) {
        (void)PetscLogEventBegin(ss_linear_initial_guess_event, 0, 0, 0, 0);
        status = ss_deflation_apply_cached_right(solver->deflation, solution,
                                                 solver->complement);
        if (status != SS_OK) {
          (void)PetscLogEventEnd(ss_linear_initial_guess_event, 0, 0, 0, 0);
          record->status = SS_LINEAR_SOLVE_SETUP_FAILED;
          return status;
        }
      }
      /* Legacy path: x0 = B B^T b, then solve the original A operator with
       * the cached projected preconditioner P_R M. */
      status = ss_deflation_coarse_initial_guess_and_project_rhs(
          solver->deflation, solve_rhs, solution, solver->projected_rhs);
      if (status != SS_OK ||
          ss_linear_petsc_status(
              solver->communicator,
              VecNorm(solver->projected_rhs, NORM_2, &projected_norm)) != SS_OK ||
          !ss_is_finite(projected_norm)) {
        if (use_input_guess != 0) {
          (void)PetscLogEventEnd(ss_linear_initial_guess_event, 0, 0, 0, 0);
        }
        record->status = SS_LINEAR_SOLVE_SETUP_FAILED;
        return status != SS_OK ? status : SS_ERROR_INTERNAL;
      }
      record->coarse_initial_residual_norm = projected_norm * rhs_scale;
      record->initial_residual_norm = record->coarse_initial_residual_norm;
      if (use_input_guess != 0) {
        int accept_input = 1;
        record->input_guess_tested = 1;
        if (solver->selected_policy.initial_guess_policy ==
            SS_LINEAR_INITIAL_GUESS_PREVIOUS_COMPLEMENT_IF_BETTER) {
          if (ss_status_allreduce_max(
                  solver->communicator,
                  MatMult(solver->operator_handle, solver->complement,
                          solver->residual) == PETSC_SUCCESS &&
                          VecAYPX(solver->residual, -1.0, solver->projected_rhs) ==
                              PETSC_SUCCESS
                      ? SS_OK
                      : SS_ERROR_INTERNAL) != SS_OK ||
              ss_linear_petsc_status(
                  solver->communicator,
                  VecNorm(solver->residual, NORM_2, &candidate_norm)) != SS_OK ||
              !ss_is_finite(candidate_norm)) {
            (void)PetscLogEventEnd(ss_linear_initial_guess_event, 0, 0, 0, 0);
            record->status = SS_LINEAR_SOLVE_SETUP_FAILED;
            return SS_ERROR_INTERNAL;
          }
          accept_input = candidate_norm < projected_norm;
          record->candidate_initial_residual_norm = candidate_norm * rhs_scale;
        }
        if (accept_input != 0 &&
            ss_linear_petsc_status(solver->communicator,
                                   VecAXPY(solution, 1.0, solver->complement)) !=
                SS_OK) {
          (void)PetscLogEventEnd(ss_linear_initial_guess_event, 0, 0, 0, 0);
          record->status = SS_LINEAR_SOLVE_SETUP_FAILED;
          return SS_ERROR_INTERNAL;
        }
        record->input_guess_accepted = accept_input;
        if (accept_input != 0 && ss_is_finite(candidate_norm)) {
          record->initial_residual_norm = candidate_norm * rhs_scale;
        }
        (void)PetscLogEventEnd(ss_linear_initial_guess_event, 0, 0, 0, 0);
        /* Preserve the coarse-only projected stopping threshold.  Otherwise a
         * better starting direction would inadvertently request a tighter
         * solve merely because PETSc scales rtol from the new initial residual. */
        if (projected_stopping != 0) {
          solve_atol = fmax(solve_atol, relative_tolerance * projected_norm);
          if (KSPSetTolerances(
                  solver->ksp, solve_rtol, solve_atol, solver->selected_policy.dtol,
                  solver->selected_policy.maximum_iterations) != PETSC_SUCCESS) {
            record->status = SS_LINEAR_SOLVE_SETUP_FAILED;
            return SS_ERROR_INTERNAL;
          }
        }
      }
      petsc_status = ss_linear_ksp_solve(solver, solve_rhs, solution,
                                         preconditioned_stopping != 0 &&
                                                 record->initial_residual_norm >
                                                     initial_correction_threshold
                                             ? PETSC_TRUE
                                             : PETSC_FALSE,
                                         &reason, &record->cancellation_reductions);
    } else {
      status = ss_deflation_project_rhs(solver->deflation, solver->operator_handle,
                                        solve_rhs, solver->projected_rhs);
      if (status != SS_OK ||
          ss_linear_petsc_status(
              solver->communicator,
              VecNorm(solver->projected_rhs, NORM_2, &projected_norm)) != SS_OK ||
          !ss_is_finite(projected_norm) ||
          ss_linear_petsc_status(solver->communicator,
                                 VecSet(solver->complement, 0.0)) != SS_OK) {
        record->status = SS_LINEAR_SOLVE_SETUP_FAILED;
        return status != SS_OK ? status : SS_ERROR_INTERNAL;
      }
      /*
       * Preserve the requested tolerance for the original system when P_L
       * magnifies the right-hand-side norm.  This changes only PETSc's internal
       * projected-system rtol; solve_atol and the independent original residual
       * check both retain the user-requested threshold.
       */
      if (projected_stopping == 0 && natural_stopping == 0 && projected_norm > 0.0 &&
          solve_atol / projected_norm < solve_rtol) {
        solve_rtol = solve_atol / projected_norm;
      }
      if (!ss_is_finite(solve_rtol) || !ss_is_finite(solve_atol) || solve_rtol < 0.0 ||
          solve_atol < 0.0 ||
          KSPSetTolerances(
              solver->ksp, solve_rtol, solve_atol, solver->selected_policy.dtol,
              solver->selected_policy.maximum_iterations) != PETSC_SUCCESS) {
        record->status = SS_LINEAR_SOLVE_SETUP_FAILED;
        return SS_ERROR_INTERNAL;
      }
      petsc_status =
          ss_linear_ksp_solve(solver, solver->projected_rhs, solver->complement,
                              PETSC_FALSE, &reason, &record->cancellation_reductions);
      if (petsc_status == PETSC_SUCCESS && reason == KSP_CONVERGED_ITERATING) {
        status =
            ss_deflation_final_correction(solver->deflation, solver->operator_handle,
                                          solve_rhs, solver->complement, solution);
        if (status != SS_OK) {
          return status;
        }
      }
    }
  } else {
    if (solver->selected_policy.initial_guess_policy ==
        SS_LINEAR_INITIAL_GUESS_PREVIOUS_COMPLEMENT_IF_BETTER) {
      record->input_guess_tested = 1;
      if (ss_status_allreduce_max(
              solver->communicator,
              MatMult(solver->operator_handle, solution, solver->residual) ==
                          PETSC_SUCCESS &&
                      VecAYPX(solver->residual, -1.0, solve_rhs) == PETSC_SUCCESS
                  ? SS_OK
                  : SS_ERROR_INTERNAL) != SS_OK ||
          ss_linear_petsc_status(solver->communicator,
                                 VecNorm(solver->residual, NORM_2, &candidate_norm)) !=
              SS_OK ||
          !ss_is_finite(candidate_norm)) {
        record->status = SS_LINEAR_SOLVE_SETUP_FAILED;
        return SS_ERROR_INTERNAL;
      }
      record->coarse_initial_residual_norm = input_norm;
      record->candidate_initial_residual_norm = candidate_norm * rhs_scale;
      record->input_guess_accepted = candidate_norm * rhs_scale < input_norm;
      if (record->input_guess_accepted == 0 &&
          ss_linear_petsc_status(solver->communicator, VecSet(solution, 0.0)) !=
              SS_OK) {
        record->status = SS_LINEAR_SOLVE_SETUP_FAILED;
        return SS_ERROR_INTERNAL;
      }
      record->initial_residual_norm =
          record->input_guess_accepted != 0 ? candidate_norm * rhs_scale : input_norm;
    }
    petsc_status =
        ss_linear_ksp_solve(solver, solve_rhs, solution,
                            preconditioned_stopping != 0 &&
                                    (ss_is_finite(record->initial_residual_norm)
                                         ? record->initial_residual_norm
                                         : input_norm) > initial_correction_threshold
                                ? PETSC_TRUE
                                : PETSC_FALSE,
                            &reason, &record->cancellation_reductions);
  }
  if (reason != KSP_CONVERGED_ITERATING) {
    const int cancelled = reason == KSP_DIVERGED_USER;
    record->status =
        cancelled ? SS_LINEAR_SOLVE_CANCELLED : SS_LINEAR_SOLVE_DIVERGED_KSP;
    record->reason = reason;
    if (cancelled) {
      (void)KSPGetIterationNumber(solver->ksp, &record->iterations);
      (void)KSPGetResidualNorm(solver->ksp, &record->reported_norm);
      record->value_generation = solver->value_generation;
      record->solve_count = ++solver->solve_count;
    }
    const PetscErrorCode restore = KSPSetTolerances(
        solver->ksp, solver->selected_policy.rtol, solver->selected_policy.atol,
        solver->selected_policy.dtol, solver->selected_policy.maximum_iterations);
    const SsStatus agreed = ss_linear_petsc_status(
        solver->communicator, petsc_status != PETSC_SUCCESS ? petsc_status : restore);
    return agreed == SS_OK && cancelled ? SS_ERROR_CANCELLED : agreed;
  }
  if (petsc_status == PETSC_SUCCESS && rhs_scale != 1.0 &&
      ss_linear_petsc_status(solver->communicator, VecScale(solution, rhs_scale)) !=
          SS_OK) {
    record->status = SS_LINEAR_SOLVE_NONFINITE;
    return SS_ERROR_INTERNAL;
  }
  restore_status = KSPSetTolerances(
      solver->ksp, solver->selected_policy.rtol, solver->selected_policy.atol,
      solver->selected_policy.dtol, solver->selected_policy.maximum_iterations);
  if (ss_status_allreduce_max(solver->communicator,
                              petsc_status == PETSC_SUCCESS &&
                                      restore_status == PETSC_SUCCESS
                                  ? SS_OK
                                  : SS_ERROR_INTERNAL) != SS_OK) {
    record->status = SS_LINEAR_SOLVE_DIVERGED_KSP;
    return SS_ERROR_INTERNAL;
  }
  status = SS_OK;
  if (KSPGetConvergedReason(solver->ksp, &reason) != PETSC_SUCCESS ||
      KSPGetIterationNumber(solver->ksp, &iterations) != PETSC_SUCCESS ||
      KSPGetResidualNorm(solver->ksp, &reported_norm) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK && reason == KSP_DIVERGED_USER && solver->cancelled != NULL) {
    record->status = SS_LINEAR_SOLVE_CANCELLED;
    record->reason = reason;
    record->iterations = iterations;
    record->reported_norm = reported_norm;
    record->value_generation = solver->value_generation;
    record->solve_count = ++solver->solve_count;
    return SS_ERROR_CANCELLED;
  }
  if (status == SS_OK && natural_stopping != 0 &&
      (KSPGetResidualHistory(solver->ksp, &residual_history, &residual_history_count) !=
           PETSC_SUCCESS ||
       residual_history == NULL || residual_history_count <= 0 ||
       !ss_is_finite(residual_history[0]))) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(solver->communicator, status);
  if (status != SS_OK) {
    record->status = SS_LINEAR_SOLVE_SETUP_FAILED;
    return status;
  }
  if (natural_stopping != 0) {
    initial_stopping_norm = residual_history[0] * rhs_scale;
  }
  reported_norm *= rhs_scale;
  status =
      ss_linear_true_residual(solver, rhs, solution, &rhs_norm, &true_residual_norm);
  if (status != SS_OK) {
    return status;
  }
  /* The KSP stopping metric controls work. Acceptance always certifies the
   * unprojected, unpreconditioned system against its original right-hand side. */
  acceptance_policy = solver->selected_policy;
  acceptance_policy.rtol = relative_tolerance;
  status = ss_linear_classify_acceptance(&acceptance_policy, reason, reported_norm,
                                         rhs_norm, true_residual_norm, &record->status,
                                         &record->acceptance_threshold);
  if (status == SS_OK && record->status == SS_LINEAR_SOLVE_CONVERGED) {
    const PetscReal metric_threshold =
        natural_stopping != 0
            ? fmax(acceptance_policy.atol, relative_tolerance * initial_stopping_norm)
            : record->acceptance_threshold;
    if ((natural_stopping != 0 &&
         (!ss_is_finite(initial_stopping_norm) || initial_stopping_norm < 0.0)) ||
        reported_norm < 0.0) {
      record->status = SS_LINEAR_SOLVE_NONFINITE;
    } else if ((preconditioned_stopping != 0 || natural_stopping != 0) &&
               reported_norm > metric_threshold) {
      record->status = SS_LINEAR_SOLVE_RESIDUAL_VERIFICATION_FAILED;
    }
  }
  if (status != SS_OK) {
    return status;
  }
  if (solver->selected_policy.method == SS_LINEAR_METHOD_HPDDM_GCRODR) {
    solver->solve_since_setup = 1;
  }
  ++solver->solve_count;
  record->reason = reason;
  record->iterations = iterations;
  record->krylov_vectors_captured = solver->captured_krylov.count;
  record->reported_norm = reported_norm;
  record->true_residual_norm = true_residual_norm;
  record->rhs_norm = rhs_norm;
  record->stopping_rhs_norm = natural_stopping != 0     ? initial_stopping_norm
                              : projected_stopping != 0 ? projected_norm * rhs_scale
                                                        : rhs_norm;
  record->stopping_condition = solver->selected_policy.stopping_condition;
  record->value_generation = solver->value_generation;
  record->solve_count = solver->solve_count;
  return SS_OK;
}

SsStatus ss_linear_solve(SsLinearSolver *solver, Vec rhs, Vec solution,
                         SsLinearSolveRecord *record) {
  if (solver == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return ss_linear_solve_with_relative_tolerance(solver, rhs, solution,
                                                 solver->selected_policy.rtol, record);
}

SsStatus ss_linear_solve_with_relative_tolerance(SsLinearSolver *solver, Vec rhs,
                                                 Vec solution,
                                                 PetscReal relative_tolerance,
                                                 SsLinearSolveRecord *record) {
  if (solver == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
#ifndef NDEBUG
  if (rhs != NULL) {
    ss_debug_stage_vector("linear-rhs", (size_t)solver->value_generation, rhs);
  }
#endif
  (void)PetscLogEventBegin(ss_linear_solve_event, 0, 0, 0, 0);
  const SsStatus status =
      ss_linear_solve_impl(solver, rhs, solution, relative_tolerance, record);
  (void)PetscLogEventEnd(ss_linear_solve_event, 0, 0, 0, 0);
#ifndef NDEBUG
  if (status == SS_OK) {
    ss_debug_stage_vector("linear-solution", (size_t)solver->value_generation,
                          solution);
  }
#endif
  if (status != SS_OK || record->status != SS_LINEAR_SOLVE_CONVERGED) {
    ss_linear_krylov_capture_clear(&solver->captured_krylov);
  }
  return status;
}

SsStatus ss_linear_promote_captured_krylov(SsLinearSolver *solver,
                                           const char *provenance,
                                           SsLinearSolveRecord *solve_record,
                                           SsLinearKrylovPromotionRecord *record) {
  SsDeflationDiagnostics diagnostics = {0};
  SsStatus status = SS_OK;
  size_t index;
  if (solver == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (provenance == NULL || solve_record == NULL || record == NULL ||
      solver->krylov_capture != SS_LINEAR_KRYLOV_CAPTURE_FULL_CORRECTION_SPACE ||
      solver->deflation == NULL || solver->captured_krylov.enabled != 0) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else if (solve_record->krylov_vectors_captured != solver->captured_krylov.count) {
    status = SS_ERROR_INTEGRITY;
  }
  status = ss_status_allreduce_max(solver->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  memset(record, 0, sizeof(*record));
  record->captured_count = solver->captured_krylov.count;
  const PetscErrorCode log_begin =
      PetscLogEventBegin(ss_linear_full_krylov_event, 0, 0, 0, 0);
  status = ss_status_allreduce_max(
      solver->communicator, log_begin == PETSC_SUCCESS ? SS_OK : SS_ERROR_INTERNAL);
  if (status != SS_OK) {
    if (log_begin == PETSC_SUCCESS) {
      (void)PetscLogEventEnd(ss_linear_full_krylov_event, 0, 0, 0, 0);
    }
    return status;
  }
  for (index = 0U; index < solver->captured_krylov.count; ++index) {
    SsDeflationAdmissionRecord admission = {0};
    status = ss_deflation_add_candidate(
        solver->deflation, solver->captured_krylov.vectors[index],
        solver->captured_krylov.vectors[index], provenance, &admission);
    if (status != SS_OK) {
      break;
    }
    if (admission.admitted != 0) {
      ++record->promoted_count;
    } else {
      ++record->rejected_count;
    }
  }
  ss_linear_krylov_capture_clear(&solver->captured_krylov);
  if (status == SS_OK &&
      ss_deflation_diagnostics(solver->deflation, &diagnostics) == SS_OK) {
    record->active_basis_count = diagnostics.active_count;
  } else if (status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  solve_record->krylov_vectors_promoted = record->promoted_count;
  solve_record->krylov_vectors_rejected = record->rejected_count;
  solve_record->krylov_basis_after_promotion = record->active_basis_count;
  if (PetscLogEventEnd(ss_linear_full_krylov_event, 0, 0, 0, 0) != PETSC_SUCCESS &&
      status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  return ss_status_allreduce_max(solver->communicator, status);
}
