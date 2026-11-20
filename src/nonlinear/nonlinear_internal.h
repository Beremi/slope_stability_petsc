/* Shared nonlinear iteration helpers.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_NONLINEAR_INTERNAL_H
#define SS_NONLINEAR_INTERNAL_H

#include "slope_internal/core/collective.h"
#include "slope_internal/core/util.h"
#include "slope_internal/nonlinear/nonlinear.h"

enum {
  SS_NL_PARENT,
  SS_NL_ACCEPTED,
  SS_NL_WORKING,
  SS_NL_RESIDUAL,
  SS_NL_RHS,
  SS_NL_DIRECTION,
  SS_NL_TRIAL,
  SS_NL_TRIAL_RESIDUAL,
  SS_NL_RESIDUAL_RHS,
  SS_NL_RESIDUAL_DIRECTION,
  SS_NL_COMBINED,
  SS_NL_SENSITIVITY,
  SS_NL_SENSITIVITY_MINUS,
  SS_NL_VECTOR_COUNT
};

struct SsNonlinearWorkspace {
  MPI_Comm communicator;
  Vec work[SS_NL_VECTOR_COUNT];
  size_t allocated_vectors;
  Mat elastic_template;
  Mat tangent;
  Mat regularized;
  SsLinearSolver *linear;
  uint64_t structure_generation;
  int matrix_free;
};

/** Obtain compatible work objects, with teardown owned by the caller when temporary is
 * nonnull. */
SsStatus ss_nonlinear_workspace_prepare(const SsNonlinearProblem *problem,
                                        Vec prototype, int matrix_free,
                                        size_t vector_count,
                                        SsNonlinearWorkspace **work,
                                        SsNonlinearWorkspace **temporary);

/** Combine a PETSc failure before entering a new collective solve phase. */
static inline SsStatus ss_nonlinear_petsc_status(MPI_Comm comm, PetscErrorCode ierr) {
  return ss_status_allreduce_max(comm,
                                 ierr == PETSC_SUCCESS ? SS_OK : SS_ERROR_INTERNAL);
}

/** Check whether any candidate linear policy uses a prior correction. */
int ss_nonlinear_uses_previous_linear_guess(const SsLinearPolicyChain *policies);
/** Combine rank-local flags with the requested logical reduction. */
SsStatus ss_nonlinear_collective_boolean(MPI_Comm communicator, int local_value,
                                         MPI_Op operation, int *global_value);
/** Check whether a PETSc object's communicator has the same ordered ranks. */
int ss_nonlinear_same_communicator(MPI_Comm expected, PetscObject object);
/** Evaluate the selected stopping criterion when its metric is available. */
int ss_nonlinear_stopping_satisfied(const SsNonlinearPolicy *policy,
                                    PetscReal relative_residual,
                                    PetscReal relative_correction,
                                    PetscReal absolute_delta_lambda,
                                    int have_correction, int have_delta_lambda);
/** Test stopping before a step, when exactly zero residual proves zero correction. */
int ss_nonlinear_stopping_at_current(const SsNonlinearPolicy *policy,
                                     PetscReal residual_norm,
                                     PetscReal relative_residual, int have_control,
                                     PetscReal *recorded_correction);
/** Allow a residual at the arithmetic noise floor after a verified small correction. */
PetscReal ss_nonlinear_line_search_floor(
    const SsNonlinearPolicy *policy, PetscReal load_norm, PetscReal relative_residual,
    PetscReal relative_correction, PetscReal absolute_delta_lambda, int have_control);
/** Search with residual merit when the caller supplies a positive noise floor. */
SsStatus ss_nonlinear_line_search_with_floor(const SsNonlinearPolicy *policy,
                                             PetscReal current_residual_norm,
                                             PetscReal residual_floor,
                                             SsNonlinearTrialEvaluation evaluate,
                                             void *context,
                                             SsNonlinearLineSearchRecord *record);
/** Return the residual threshold for explicitly relaxed acceptance. */
PetscReal ss_nonlinear_relaxed_residual_threshold(const SsNonlinearPolicy *policy);
/** Validate the deflation promotion selector. */
int ss_nonlinear_deflation_promotion_valid(SsNonlinearDeflationPromotion promotion);
/** Check whether corrections are captured for deflation. */
int ss_nonlinear_full_krylov_enabled(const SsNonlinearProblem *problem);
/** Check whether completed solve directions are added to deflation. */
int ss_nonlinear_system_solutions_enabled(const SsNonlinearProblem *problem);
/** Check whether any Krylov or solved directions are promoted. */
int ss_nonlinear_linear_history_enabled(const SsNonlinearProblem *problem);
/** Promote the configured linear history and update solve diagnostics. */
SsStatus ss_nonlinear_promote_linear_history(const SsNonlinearProblem *problem,
                                             SsLinearSolver *linear, Vec solution,
                                             const char *krylov_provenance,
                                             const char *solution_provenance,
                                             SsLinearSolveRecord *solve_record);
/** Finish preparing the basis for an accepted state. */
SsStatus ss_nonlinear_prepare_deflation_commit(const SsNonlinearProblem *problem,
                                               Vec accepted_state, Mat operator_handle,
                                               uint64_t operator_generation);
/** Translate an evaluation status into a nonlinear outcome. */
SsNonlinearOutcome ss_nonlinear_status_outcome(SsStatus status);
/** Check that a rejected solve preserved its parent state. */
SsStatus ss_nonlinear_parent_equal(Vec parent, Vec snapshot, int *equal);

/** Bind the current observer and cancellation poll to this linear system. */
void ss_nonlinear_observe_linear(const SsNonlinearProblem *problem,
                                 SsLinearSolver *solver, PetscInt ordinal,
                                 SsNonlinearSystem system);

/** Report a linear solve result to the nonlinear observer. */
void ss_nonlinear_observe_linear_result(const SsNonlinearProblem *problem,
                                        SsStatus status,
                                        const SsLinearSolveRecord *record);

/** Notify the run observer without making MPI calls. */
void ss_nonlinear_observe_iteration(const SsNonlinearProblem *problem,
                                    const SsNonlinearIterationRecord *record);

/* ---- Shared Newton iteration skeleton (nonlinear_iteration.c) ----
 *
 * The fixed-control and work-controlled solvers run the same damped Newton
 * iteration: poll cancellation, evaluate and test the residual, assemble or
 * linearize the regularized operator, solve for a direction, run a line search,
 * update the elastic regularization, promote deflation candidates and accept
 * the state. Each solver keeps its own loop and its distinct control logic and
 * calls the phases below in order.
 *
 * A phase that returns an `int` returns nonzero to continue the iteration. Zero
 * means "leave the loop": either `record->outcome` holds the final nonlinear
 * outcome (a regular stop with `newton->status == SS_OK`), or `newton->status`
 * holds the error that aborts the solve. All phases are collective over the
 * problem communicator and agree on their decision.
 */

/** State of one fixed-control or work-controlled Newton solve. */
typedef struct {
  const SsNonlinearProblem *problem;
  const SsNonlinearPolicy *policy;
  SsNonlinearSolveRecord *record;
  MPI_Comm communicator;
  /** Workspace in use; owned by the problem or by `temporary`. */
  SsNonlinearWorkspace *work;
  /** Isolated workspace released by `ss_nonlinear_newton_end()`. */
  SsNonlinearWorkspace *temporary;
  Vec parent_state;
  Vec accepted_state;
  /** Nonzero for the work-controlled variant, which evaluates in `mode`. */
  int controlled;
  SsNonlinearControlMode mode;
  /** Current control: the fixed factor or the work-controlled load factor. */
  PetscReal control;
  PetscReal load_norm;
  /** Elastic regularization for the next operator. */
  PetscReal regularization;
  PetscReal final_norm;
  PetscReal final_convergence;
  PetscReal final_relative_correction;
  /** Loop counter; equals the iteration limit when the loop ran to completion. */
  PetscInt iteration;
  /** Number of regularized operators built by this solve. */
  uint64_t operator_generation;
  /** First error of the solve; SS_OK while the solve can proceed. */
  SsStatus status;
  int matrix_free;
  int use_previous_linear_guess;
  int transaction_open;
  /** Nonzero when the next iteration reuses the current residual and tangent. */
  int reuse_current_tangent;
  int succeeded;
} SsNonlinearNewton;

/** Values of one Newton iteration passed between the phases. */
typedef struct {
  /** Iteration record in `record->history`, set by `ss_nonlinear_newton_prepare()`. */
  SsNonlinearIterationRecord *record;
  PetscReal residual_norm;
  PetscReal convergence;
  PetscReal regularization_before;
  PetscReal direction_norm;
  PetscReal working_norm;
  /** Control adopted with an accepted step; the current control by default. */
  PetscReal next_control;
  SsNonlinearRegularizationRecord regularization;
  int tangent_reused;
} SsNonlinearNewtonStep;

/** Line-search verdict of one Newton iteration. */
typedef enum {
  /** Leave the loop; the outcome or the error is set. */
  SS_NONLINEAR_STEP_STOP = 0,
  /** A step length was accepted; `record->alpha` holds it. */
  SS_NONLINEAR_STEP_ACCEPTED,
  /** No step length was accepted; `record->alpha` is zero. */
  SS_NONLINEAR_STEP_REJECTED
} SsNonlinearStepVerdict;

/**
 * Check the arguments common to both Newton solvers on this rank.
 *
 * Returns SS_ERROR_INVALID_ARGUMENT for missing or inconsistent problem data,
 * states, deflation settings or sizes. Local; the caller agrees on the result.
 */
SsStatus ss_nonlinear_newton_arguments(const SsNonlinearProblem *problem,
                                       Vec parent_state, Vec accepted_state);

/**
 * Initialize the solve record and prepare the workspace.
 *
 * `mode` selects the work-controlled evaluation, or the fixed-control
 * evaluation when NULL; `control` is the initial control. `vector_count` work
 * vectors are prepared. The parent and accepted states are snapshotted, the
 * working state starts from the parent and the direction vectors are zeroed.
 * Returns and stores the first error; `ss_nonlinear_newton_end()` must follow in
 * every case.
 */
SsStatus ss_nonlinear_newton_begin(SsNonlinearNewton *newton,
                                   const SsNonlinearProblem *problem,
                                   const SsNonlinearPolicy *policy,
                                   const SsNonlinearControlMode *mode,
                                   PetscReal control, Vec parent_state,
                                   Vec accepted_state, size_t vector_count,
                                   SsNonlinearSolveRecord *record);

/** Open the deflation transaction when the solve owns it; stores an error. */
SsStatus ss_nonlinear_newton_open_transaction(SsNonlinearNewton *newton);

/**
 * Evaluate the residual, and the tangent when nonnull, of `state` at `control`.
 *
 * Uses the fixed or work-controlled evaluation of the problem and agrees on
 * the status.
 */
SsStatus ss_nonlinear_newton_evaluate(const SsNonlinearNewton *newton, Vec state,
                                      PetscReal control, Vec residual, Mat tangent);

/**
 * Evaluate a line-search trial state at `control`.
 *
 * Applies the optional admissibility test (agreed with a logical AND over the
 * ranks) and then evaluates the trial residual.
 */
SsStatus ss_nonlinear_newton_trial_residual(const SsNonlinearNewton *newton, Vec trial,
                                            PetscReal control, Vec trial_residual,
                                            int *admissible);

/**
 * Start iteration `newton->iteration`: poll cancellation, evaluate the residual
 * and tangent unless the previous step was rejected, and stop when the current
 * state already satisfies the stopping criterion.
 */
int ss_nonlinear_newton_start(SsNonlinearNewton *newton, SsNonlinearNewtonStep *step);

/**
 * Record the iteration and set up the regularized linear operator.
 *
 * Fills the next history entry, linearizes (matrix-free) or forms the
 * regularized assembled operator, and sets up `linear_problem` with a new
 * operator value generation.
 */
int ss_nonlinear_newton_prepare(SsNonlinearNewton *newton, SsNonlinearNewtonStep *step,
                                SsLinearProblem *linear_problem);

/** Run the optional linear diagnostic hook on the formed right-hand sides. */
int ss_nonlinear_newton_linear_diagnostic(SsNonlinearNewton *newton,
                                          const SsLinearProblem *linear_problem,
                                          size_t rhs_count,
                                          const Vec *right_hand_sides);

/**
 * Solve one linear system of the iteration under the run observer.
 *
 * `relative_tolerance` overrides the selected policy's tolerance when nonnull.
 * A cancelled or unconverged solve stops the iteration.
 */
int ss_nonlinear_newton_linear_solve(SsNonlinearNewton *newton,
                                     SsNonlinearSystem system, Vec rhs, Vec solution,
                                     const PetscReal *relative_tolerance,
                                     SsLinearSolveRecord *solve_record);

/** Promote the configured Krylov and solution history of a completed solve. */
int ss_nonlinear_newton_promote(SsNonlinearNewton *newton, Vec solution,
                                const char *krylov_provenance,
                                const char *solution_provenance,
                                SsLinearSolveRecord *solve_record);

/** Measure the direction and the working state; stop when either is not finite. */
int ss_nonlinear_newton_measure(SsNonlinearNewton *newton, SsNonlinearNewtonStep *step,
                                Vec direction);

/**
 * Run the line search with the residual noise floor of this iteration.
 *
 * `delta_lambda` is the control increment for the floor (NaN without control).
 * Sets `record->alpha` to the accepted step length, or to zero on rejection.
 */
SsNonlinearStepVerdict
ss_nonlinear_newton_line_search(SsNonlinearNewton *newton, SsNonlinearNewtonStep *step,
                                const SsNonlinearPolicy *line_policy,
                                PetscReal delta_lambda,
                                SsNonlinearTrialEvaluation trial, void *trial_context);

/**
 * Complete a rejected step: keep the state, stiffen the regularization and
 * reuse the current residual and tangent in the next iteration.
 */
int ss_nonlinear_newton_reject_step(SsNonlinearNewton *newton,
                                    SsNonlinearNewtonStep *step);

/** Record the accepted step and update the regularization from its length. */
int ss_nonlinear_newton_accept_step(SsNonlinearNewton *newton,
                                    SsNonlinearNewtonStep *step);

/**
 * Offer the solved directions as deflation candidates.
 *
 * Applies when an iteration-level deflation policy is active and the
 * regularization update made the step eligible.
 */
int ss_nonlinear_newton_add_candidates(SsNonlinearNewton *newton,
                                       const SsNonlinearNewtonStep *step, size_t count,
                                       const Vec *directions,
                                       const char *const *provenances);

/**
 * Adopt `trial` as the working state and `step->next_control` as the control,
 * notify the observer and stop when the stopping criterion is satisfied.
 *
 * `delta_lambda` and `have_correction` are passed to the stopping test.
 */
int ss_nonlinear_newton_complete_step(SsNonlinearNewton *newton,
                                      const SsNonlinearNewtonStep *step, Vec trial,
                                      PetscReal delta_lambda, int have_correction);

/**
 * Close a solve whose loop ended without an error.
 *
 * Re-evaluates the state after the iteration limit (with optional relaxed
 * acceptance), then commits the accepted state and the deflation transaction
 * or rolls them back and checks that the parent state is unchanged. On
 * success, `accepted_control` (when nonnull) receives the final control.
 */
void ss_nonlinear_newton_finish(SsNonlinearNewton *newton, PetscReal *accepted_control);

/**
 * Release the solve: roll back an open transaction, finish the linear attempt
 * and destroy an isolated workspace. Returns the agreed final status.
 */
SsStatus ss_nonlinear_newton_end(SsNonlinearNewton *newton);

#endif
