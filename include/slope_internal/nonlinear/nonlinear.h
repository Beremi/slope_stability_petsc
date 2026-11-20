/** @file
 * @brief Damped, regularized Newton solves at a fixed factor or under work
 * control, with line search, progress guards and deflation transactions.
 *
 * Solves are collective on the problem's communicator and run on the master
 * thread; the policy helpers are pure functions.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_NONLINEAR_NONLINEAR_H
#define SS_NONLINEAR_NONLINEAR_H

#include "slope_internal/core/status.h"
#include "slope_internal/deflation/deflation.h"
#include "slope_internal/linear/linear.h"
#include "slope_internal/pmg/pmg.h"

#include <mpi.h>
#include <petscmat.h>
#include <petscvec.h>

#include <stddef.h>
#include <stdint.h>

/** Largest number of Newton iterations kept in a solve record. */
#define SS_NONLINEAR_HISTORY_CAPACITY 256U
/** Largest number of line-search trials kept in a record. */
#define SS_NONLINEAR_LINE_TRIAL_CAPACITY 64U
/** Successful samples the adaptive progress guard needs before it acts. */
#define SS_NONLINEAR_ADAPTIVE_GUARD_MINIMUM_SAMPLES 2U

/** Outcome of a nonlinear solve or of one of its steps. */
typedef enum {
  SS_NONLINEAR_CONVERGED = 0,                  /**< Converged. */
  SS_NONLINEAR_ACCEPTED_RELAXED = 1,           /**< Accepted with relaxed tolerance. */
  SS_NONLINEAR_FAILED_LINE_SEARCH = 2,         /**< No acceptable step length. */
  SS_NONLINEAR_FAILED_REGULARIZATION = 3,      /**< Regularization left [0, 1]. */
  SS_NONLINEAR_FAILED_LINEAR = 4,              /**< A linear solve failed. */
  SS_NONLINEAR_FAILED_SINGULAR_CONSTRAINT = 5, /**< Singular control constraint. */
  SS_NONLINEAR_FAILED_NONFINITE = 6,           /**< Nonfinite values. */
  SS_NONLINEAR_FAILED_ITERATION_LIMIT = 7,     /**< Iteration limit reached. */
  SS_NONLINEAR_CANCELLED = 8                   /**< Cancelled. */
} SsNonlinearOutcome;

/** Line-search criterion. */
typedef enum {
  SS_NONLINEAR_LINE_DIRECTIONAL = 1,      /**< Sign of the directional derivative. */
  SS_NONLINEAR_LINE_RESIDUAL_MONOTONE = 2 /**< Monotone decrease of the residual. */
} SsNonlinearLineMethod;

/** Who owns the deflation transaction of a solve. */
typedef enum {
  SS_NONLINEAR_DEFLATION_NONE = 0,              /**< No deflation. */
  SS_NONLINEAR_DEFLATION_OWNED_TRANSACTION = 1, /**< The solve opens and closes it. */
  /** The caller (the continuation analysis) opens and closes it. */
  SS_NONLINEAR_DEFLATION_EXTERNAL_TRANSACTION = 2
} SsNonlinearDeflationMode;

/** Vectors a successful solve offers as deflation candidates. */
typedef enum {
  SS_NONLINEAR_DEFLATION_PROMOTE_DIRECTIONS = 0,     /**< Newton directions. */
  SS_NONLINEAR_DEFLATION_PROMOTE_ACCEPTED_STATE = 1, /**< The accepted state. */
  SS_NONLINEAR_DEFLATION_DISCARD_PROVISIONAL = 2,    /**< Nothing; discard. */
  /** Captured Krylov directions and the accepted state. */
  SS_NONLINEAR_DEFLATION_PROMOTE_FULL_KRYLOV_AND_ACCEPTED_STATE = 3,
  /** Final linear-system corrections and the accepted state. */
  SS_NONLINEAR_DEFLATION_PROMOTE_SYSTEM_SOLUTIONS_AND_ACCEPTED_STATE = 4
} SsNonlinearDeflationPromotion;

/** Control of a work-controlled solve. */
typedef enum {
  SS_NONLINEAR_CONTROL_LIMIT_LOAD = 1,  /**< The control is the load factor. */
  SS_NONLINEAR_CONTROL_SSR_INDIRECT = 2 /**< The control is the reduction factor. */
} SsNonlinearControlMode;

/** Finite difference of the control sensitivity in indirect SSR. */
typedef enum {
  SS_NONLINEAR_CONTROL_SENSITIVITY_REFERENCE_FORWARD = 0, /**< Reference forward. */
  SS_NONLINEAR_CONTROL_SENSITIVITY_STABLE_CENTRAL = 1     /**< Scaled central. */
} SsNonlinearControlSensitivityMethod;

/** Newton stopping criterion. */
typedef enum {
  SS_NONLINEAR_STOP_RELATIVE_RESIDUAL = 1,    /**< Residual relative to the load. */
  SS_NONLINEAR_STOP_RELATIVE_CORRECTION = 2,  /**< Correction relative to the state. */
  SS_NONLINEAR_STOP_ABSOLUTE_DELTA_LAMBDA = 3 /**< Change of the control. */
} SsNonlinearStoppingCriterion;

/** Mode of the indirect-SSR progress guard. */
typedef enum {
  SS_NONLINEAR_PROGRESS_GUARD_OFF = 0,      /**< Disabled. */
  SS_NONLINEAR_PROGRESS_GUARD_FIXED = 1,    /**< Fixed bound from the policy. */
  SS_NONLINEAR_PROGRESS_GUARD_ADAPTIVE = 2, /**< Bound learned from accepted steps. */
  /** Learn and report the adaptive envelope without terminating a solve. */
  SS_NONLINEAR_PROGRESS_GUARD_OBSERVE = 3
} SsNonlinearProgressGuardMode;

/** Online envelope of the adaptive indirect-SSR progress guard.
 *
 * Only checkpoint residuals from continuation attempts that are eventually
 * committed may be retained. The owner of the solver transaction rolls this
 * state back when the continuation controller rejects a provisionally
 * successful solve. */
typedef struct {
  PetscInt checkpoint_iteration;            /**< Newton correction that is sampled. */
  uint64_t successful_sample_count;         /**< Samples from committed solves. */
  PetscReal maximum_successful_convergence; /**< Largest sampled residual. */
  /** Smallest sampled checkpoint/initial residual ratio. */
  PetscReal minimum_successful_convergence_ratio;
  /** Largest sampled checkpoint/initial residual ratio. */
  PetscReal maximum_successful_convergence_ratio;
} SsNonlinearAdaptiveProgressGuard;

/** Newton policy. */
typedef struct {
  /** Residual scale for explicitly relaxed acceptance with non-residual
   * stopping criteria and for the reference indirect finite difference. */
  PetscReal tolerance;
  SsNonlinearStoppingCriterion stopping_criterion; /**< Stopping criterion. */
  PetscReal stopping_tolerance;                    /**< Its tolerance. */
  PetscInt maximum_iterations;                     /**< Newton iteration limit. */
  /** Optional indirect-SSR early-stall guard: after this many completed
   * Newton corrections, reject the attempt when its relative residual still
   * exceeds the bound. Iteration zero screens the incoming predictor before a
   * tangent or linear solve. The checkpoint must precede the iteration
   * limit; fixed-control and limit-load solves ignore the guard. */
  PetscInt indirect_progress_guard_iteration;
  /** Residual bound of the guard; zero disables it. */
  PetscReal indirect_progress_guard_maximum_convergence;
  /** Optional scale-free companion bound: when positive, both the bound above
   * and the checkpoint/initial convergence ratio must be exceeded. */
  PetscReal indirect_progress_guard_maximum_convergence_ratio;
  PetscInt maximum_line_search_trials; /**< Line-search trial budget. */
  PetscReal minimum_alpha;             /**< Smallest accepted step length. */
  /** Smallest blend `r` of the regularized tangent `(1 - r) K + r K_e`. */
  PetscReal regularization_minimum;
  /** Relative tolerance of a singular control constraint. */
  PetscReal singular_relative_tolerance;
  /** Factor (at least 1) applied to the tolerance for relaxed acceptance. */
  PetscReal relaxed_factor;
  int allow_relaxed; /**< Nonzero allows relaxed acceptance at the limit. */
  /** Reproduce the earlier petsc4py indirect-SSR ALG5 trial semantics: test
   * unscaled Newton trial states and keep the final finite trial when no trial
   * decreases the residual. Opt-in; the default is a strict monotone search. */
  int line_search_legacy_alg5;
  SsNonlinearLineMethod line_method; /**< Line-search criterion. */
  /** Finite difference of the control sensitivity. */
  SsNonlinearControlSensitivityMethod control_sensitivity_method;
  /** Relative step of the central difference; 0 selects `cbrt(eps)`. */
  PetscReal control_sensitivity_relative_step;
  /** Optional indirect-SSR forcing update: when a line-search step is below
   * the alpha threshold, tighten the next control or residual solve if that
   * role used no more than the iteration threshold of Krylov steps. */
  int indirect_adaptive_linear_tolerance;
  /** Largest previous Krylov iteration count that permits tightening. */
  PetscInt indirect_adaptive_linear_iteration_threshold;
  /** Step length below which the tolerance may tighten. */
  PetscReal indirect_adaptive_linear_alpha_threshold;
  /** Multiplier applied when tightening. */
  PetscReal indirect_adaptive_linear_tightening_factor;
  /** Floor of the tightened relative tolerance. */
  PetscReal indirect_adaptive_linear_minimum_relative_tolerance;
} SsNonlinearPolicy;

/** Evaluate the residual, and optionally tangent and functional, at a fixed
 * factor. Collective on the problem's communicator.
 * @param context Problem context.
 * @param state Displacement state.
 * @param factor Load or strength-reduction factor.
 * @param residual Residual, or NULL.
 * @param tangent Tangent to assemble, or NULL.
 * @param functional Energy functional, or NULL.
 * @return ::SS_OK or the agreed failure status. */
typedef SsStatus (*SsNonlinearEvaluate)(void *context, Vec state, PetscReal factor,
                                        Vec residual, Mat tangent,
                                        PetscReal *functional);
/** Evaluate the residual, tangent and functional at a control value.
 * @param context Problem context.
 * @param mode Control mode.
 * @param state Displacement state.
 * @param control Control value.
 * @param residual Residual, or NULL.
 * @param tangent Tangent to assemble, or NULL.
 * @param functional Energy functional, or NULL.
 * @return ::SS_OK or the agreed failure status. */
typedef SsStatus (*SsNonlinearControlledEvaluate)(void *context,
                                                  SsNonlinearControlMode mode,
                                                  Vec state, PetscReal control,
                                                  Vec residual, Mat tangent,
                                                  PetscReal *functional);
/** Prepare a matrix-free or regularized tangent at a state.
 * @param context Problem context.
 * @param state Displacement state.
 * @param factor Load or strength-reduction factor.
 * @param elastic_regularization Blend `r` with the elastic operator.
 * @return ::SS_OK or the agreed failure status. */
typedef SsStatus (*SsNonlinearLinearize)(void *context, Vec state, PetscReal factor,
                                         PetscReal elastic_regularization);
/** Prepare the tangent at a state and control value.
 * @param context Problem context.
 * @param mode Control mode.
 * @param state Displacement state.
 * @param control Control value.
 * @param elastic_regularization Blend `r` with the elastic operator.
 * @return ::SS_OK or the agreed failure status. */
typedef SsStatus (*SsNonlinearControlledLinearize)(void *context,
                                                   SsNonlinearControlMode mode,
                                                   Vec state, PetscReal control,
                                                   PetscReal elastic_regularization);
/** Test whether a trial state is admissible.
 * @param context Problem context.
 * @param state Trial state.
 * @param factor Factor of the trial.
 * @param admissible Nonzero when admissible.
 * @return ::SS_OK or the agreed failure status. */
typedef SsStatus (*SsNonlinearAdmissibility)(void *context, Vec state, PetscReal factor,
                                             int *admissible);
/** Process-local cancellation poll.
 * @param context Problem context.
 * @return Nonzero when cancellation was requested on this rank. */
typedef int (*SsNonlinearCancellation)(void *context);
/** Optional collective diagnostic run after the tangent, PMG hierarchy and
 * deflation basis are refreshed and all right-hand sides are formed, before
 * the trajectory-changing solve.
 * @param context Problem context.
 * @param linear_problem Prepared linear problem.
 * @param state Current state.
 * @param transaction_generation Generation of the attempt.
 * @param nonlinear_iteration Newton iteration.
 * @param control Current control.
 * @param elastic_regularization Current regularization blend.
 * @param rhs_count Number of right-hand sides.
 * @param right_hand_sides Right-hand sides.
 * @return ::SS_OK or the agreed failure status. */
typedef SsStatus (*SsNonlinearLinearDiagnostic)(
    void *context, const SsLinearProblem *linear_problem, Vec state,
    uint64_t transaction_generation, PetscInt nonlinear_iteration, PetscReal control,
    PetscReal elastic_regularization, size_t rhs_count, const Vec *right_hand_sides);

/** Record of one Newton iteration; see the struct definition below. */
typedef struct SsNonlinearIterationRecord SsNonlinearIterationRecord;

/** Role of a linear system within a Newton iteration. */
typedef enum {
  SS_NONLINEAR_SYSTEM_FIXED,    /**< Fixed-factor Newton correction. */
  SS_NONLINEAR_SYSTEM_CONTROL,  /**< Control direction of a controlled solve. */
  SS_NONLINEAR_SYSTEM_RESIDUAL, /**< Residual direction of a controlled solve. */
  SS_NONLINEAR_SYSTEM_ELASTIC   /**< Elastic predictor. */
} SsNonlinearSystem;

/** Observe the start of a linear system within the current Newton solve.
 * @param context Observer context.
 * @param iteration Newton iteration.
 * @param system Role of the system. */
typedef void (*SsNonlinearLinearStarted)(void *context, PetscInt iteration,
                                         SsNonlinearSystem system);

/** Observe a linear result before updating the Newton state.
 * @param context Observer context.
 * @param status Agreed status of the solve.
 * @param record Solve outcome. */
typedef void (*SsNonlinearLinearCompleted)(void *context, SsStatus status,
                                           const SsLinearSolveRecord *record);

/** Observe a completed Newton correction before the next solve begins.
 * @param context Observer context.
 * @param record Iteration record. */
typedef void (*SsNonlinearIterationCompleted)(void *context,
                                              const SsNonlinearIterationRecord *record);

/** Optional rank-local observers of a solve. */
typedef struct {
  SsLinearObserver linear;                     /**< Krylov and PMG observers. */
  SsNonlinearLinearStarted linear_started;     /**< Linear system start, or NULL. */
  SsNonlinearLinearCompleted linear_completed; /**< Linear result, or NULL. */
  SsNonlinearIterationCompleted iteration_completed; /**< Iteration, or NULL. */
} SsNonlinearObserver;

/** Callbacks of a nonlinear problem; each is collective. */
typedef struct {
  SsNonlinearEvaluate evaluate;                      /**< Fixed-factor evaluation. */
  SsNonlinearControlledEvaluate evaluate_controlled; /**< Controlled evaluation. */
  SsNonlinearLinearize linearize;                    /**< Fixed-factor linearization. */
  SsNonlinearControlledLinearize linearize_controlled; /**< Controlled one. */
  SsNonlinearAdmissibility admissibility; /**< Trial admissibility, or NULL. */
  SsNonlinearCancellation cancelled;      /**< Cancellation poll, or NULL. */
  /** Optional collective experiment hook; see ::SsNonlinearLinearDiagnostic. */
  SsNonlinearLinearDiagnostic linear_diagnostic;
} SsNonlinearProblemOps;

/** Reusable Newton resources owned by a field solve context. */
typedef struct SsNonlinearWorkspace SsNonlinearWorkspace;

/** Create an empty workspace; collective. Vectors and matrices are allocated
 * on demand and reused across solves.
 * @param[in] communicator Communicator of the field.
 * @param[out] workspace New workspace, released with
 *   ss_nonlinear_workspace_destroy().
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_OUT_OF_MEMORY,
 *   agreed on all ranks. */
SsStatus ss_nonlinear_workspace_create(MPI_Comm communicator,
                                       SsNonlinearWorkspace **workspace);

/** Release the retained linear solver, tangent and work vectors; collective.
 * @param[in,out] workspace Workspace pointer, set to NULL; `*workspace` may be
 *   NULL.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_nonlinear_workspace_destroy(SsNonlinearWorkspace **workspace);

/** Nonlinear problem; every object is borrowed. */
typedef struct {
  MPI_Comm communicator; /**< Communicator of the field. */
  /** Borrowed workspace. A null workspace gives an isolated single solve. */
  SsNonlinearWorkspace *workspace;
  Vec reference_load;    /**< Reference load `f` scaled by the factor. */
  Mat elastic_reference; /**< Elastic operator for the predictor and blending. */
  SsNonlinearProblemOps operations;                  /**< Problem callbacks. */
  SsNonlinearObserver observer;                      /**< Optional observers. */
  void *context;                                     /**< Context of the callbacks. */
  SsLinearOperatorProperties linear_properties;      /**< Tangent properties. */
  SsLinearPolicyChain linear_policies;               /**< Linear policies. */
  SsDeflation *deflation;                            /**< Deflation store, or NULL. */
  SsPmg *pmg;                                        /**< PMG hierarchy, or NULL. */
  SsLinearRecycle *krylov_recycle;                   /**< Recycle space, or NULL. */
  SsNonlinearDeflationMode deflation_mode;           /**< Transaction owner. */
  SsNonlinearDeflationPromotion deflation_promotion; /**< Candidates offered. */
  uint64_t structure_generation;   /**< Pattern generation of the tangent. */
  uint64_t first_value_generation; /**< First value generation to use. */
  uint64_t transaction_generation; /**< Generation of the attempt. */
  uint64_t accepted_generation;    /**< Generation of the parent state. */
} SsNonlinearProblem;

/** One line-search trial. */
typedef struct {
  PetscInt ordinal;            /**< Trial number. */
  PetscReal alpha;             /**< Step length. */
  PetscReal directional_value; /**< Directional derivative at the trial. */
  PetscReal residual_norm;     /**< Residual norm at the trial. */
  int finite;                  /**< Nonzero when the trial was finite. */
  int admissible;              /**< Nonzero when the trial was admissible. */
  int accepted;                /**< Nonzero for the accepted trial. */
} SsNonlinearLineTrial;

/** Outcome of a line search. */
typedef struct {
  SsNonlinearLineMethod method;     /**< Criterion. */
  PetscReal current_residual_norm;  /**< Residual norm before the step. */
  PetscReal accepted_alpha;         /**< Accepted step length. */
  PetscReal accepted_residual_norm; /**< Residual norm after the step. */
  size_t trial_count;               /**< Trials evaluated. */
  SsNonlinearLineTrial trials[SS_NONLINEAR_LINE_TRIAL_CAPACITY]; /**< Trials. */
  SsNonlinearOutcome outcome;                                    /**< Outcome. */
} SsNonlinearLineSearchRecord;

/** Update of the tangent regularization blend. */
typedef struct {
  PetscReal prior_value;               /**< Blend before the update. */
  PetscReal next_value;                /**< Blend after the update. */
  PetscReal alpha;                     /**< Step length that drove the update. */
  int operator_generation_invalidated; /**< Nonzero when the blend changed. */
  int deflation_candidate_eligible;    /**< Nonzero when the step was good. */
  SsNonlinearOutcome outcome;          /**< Outcome. */
} SsNonlinearRegularizationRecord;

/** Work-preserving combination of the two directions of a controlled solve. */
typedef struct {
  PetscReal numerator;         /**< Dot product of `g` and the residual direction. */
  PetscReal denominator;       /**< Dot product of `g` and the control direction. */
  PetscReal denominator_scale; /**< `||g|| ||w_control||`. */
  PetscReal threshold;         /**< Singularity threshold of the denominator. */
  PetscReal control_increment; /**< `-numerator / denominator`. */
  SsNonlinearOutcome outcome;  /**< Outcome. */
} SsNonlinearAugmentedRecord;

/** Record of one Newton iteration. */
struct SsNonlinearIterationRecord {
  PetscInt ordinal;                /**< Iteration number. */
  PetscReal residual_before;       /**< Residual norm before the step. */
  PetscReal residual_after;        /**< Residual norm after the step. */
  PetscReal convergence_before;    /**< Stopping measure before the step. */
  PetscReal convergence_after;     /**< Stopping measure after the step. */
  PetscReal relative_correction;   /**< Correction relative to the state. */
  PetscReal alpha;                 /**< Accepted step length. */
  PetscReal regularization_before; /**< Blend before the step. */
  PetscReal regularization_after;  /**< Blend after the step. */
  PetscReal control_before;        /**< Control before the step. */
  PetscReal control_increment;     /**< Control increment of the direction. */
  PetscReal control_after;         /**< Control after the step. */
  PetscReal work_after;            /**< Work after the step. */
  /** Finite difference of the control sensitivity. */
  SsNonlinearControlSensitivityMethod control_sensitivity_method;
  PetscReal control_sensitivity_epsilon; /**< Its step. */
  uint64_t operator_value_generation;    /**< Value generation of the tangent. */
  int tangent_reused;                    /**< Nonzero when the tangent was reused. */
  int deflation_candidate;               /**< Nonzero when a candidate was offered. */
  SsLinearSetupRecord linear_setup;      /**< Linear setup. */
  SsLinearSolveRecord linear_solve;      /**< Fixed-factor solve. */
  SsLinearSolveRecord control_linear_solve;     /**< Control-direction solve. */
  SsLinearSolveRecord residual_linear_solve;    /**< Residual-direction solve. */
  PetscReal control_linear_relative_tolerance;  /**< Tolerance used (control). */
  PetscReal residual_linear_relative_tolerance; /**< Tolerance used (residual). */
  /** Tolerance for the next control solve. */
  PetscReal control_linear_next_relative_tolerance;
  /** Tolerance for the next residual solve. */
  PetscReal residual_linear_next_relative_tolerance;
  int control_linear_tolerance_tightened;  /**< Nonzero when tightened. */
  int residual_linear_tolerance_tightened; /**< Nonzero when tightened. */
  SsNonlinearAugmentedRecord augmented;    /**< Direction combination. */
  SsNonlinearLineSearchRecord line_search; /**< Line search. */
};

/** Outcome of a Newton solve. */
typedef struct {
  SsNonlinearOutcome outcome;               /**< Outcome. */
  PetscInt iterations;                      /**< Newton iterations. */
  PetscReal factor;                         /**< Final factor or control. */
  PetscReal load_norm;                      /**< Norm of the scaled load. */
  PetscReal initial_residual_norm;          /**< Initial residual norm. */
  PetscReal final_residual_norm;            /**< Final residual norm. */
  PetscReal initial_convergence;            /**< Initial stopping measure. */
  PetscReal final_convergence;              /**< Final stopping measure. */
  PetscReal final_relative_correction;      /**< Last relative correction. */
  PetscReal final_regularization;           /**< Last regularization blend. */
  uint64_t transaction_generation;          /**< Generation of the attempt. */
  uint64_t accepted_generation;             /**< Generation of the parent state. */
  uint64_t regularized_operator_generation; /**< Last tangent generation. */
  PetscInt progress_guard_iteration;        /**< Checkpoint of the progress guard. */
  PetscReal progress_guard_threshold;       /**< Residual bound of the guard. */
  PetscReal progress_guard_ratio_threshold; /**< Ratio bound of the guard. */
  size_t history_count;                     /**< Iterations recorded. */
  /** Per-iteration records, up to ::SS_NONLINEAR_HISTORY_CAPACITY. */
  SsNonlinearIterationRecord history[SS_NONLINEAR_HISTORY_CAPACITY];
  int progress_guard_enabled;      /**< Nonzero when the guard was armed. */
  int progress_guard_triggered;    /**< Nonzero when the guard stopped the solve. */
  int parent_unchanged_on_failure; /**< Nonzero when a failure left the parent intact.
                                    */
  int transaction_committed;       /**< Nonzero when the solve committed deflation. */
  int transaction_rolled_back;     /**< Nonzero when it rolled deflation back. */
} SsNonlinearSolveRecord;

/** Evaluate one line-search trial.
 * @param context Line-search context.
 * @param alpha Step length.
 * @param directional_value Directional derivative at the trial.
 * @param residual_norm Residual norm at the trial.
 * @param admissible Nonzero when the trial is admissible.
 * @return ::SS_OK or the agreed failure status. */
typedef SsStatus (*SsNonlinearTrialEvaluation)(void *context, PetscReal alpha,
                                               PetscReal *directional_value,
                                               PetscReal *residual_norm,
                                               int *admissible);

/** Check a Newton policy; pure.
 * @param[in] policy Policy.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_nonlinear_policy_validate(const SsNonlinearPolicy *policy);
/** Start an empty adaptive guard; pure.
 * @param[in] checkpoint_iteration Newton correction that is sampled.
 * @param[out] guard Empty guard.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_nonlinear_adaptive_progress_guard_initialize(
    PetscInt checkpoint_iteration, SsNonlinearAdaptiveProgressGuard *guard);
/** Restore an adaptive guard from a checkpoint; pure.
 * @param[in] checkpoint_iteration Newton correction that is sampled.
 * @param[in] successful_sample_count Committed samples.
 * @param[in] maximum_successful_convergence Largest sampled residual.
 * @param[in] minimum_successful_convergence_ratio Smallest sampled ratio.
 * @param[in] maximum_successful_convergence_ratio Largest sampled ratio.
 * @param[out] guard Restored guard.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT for inconsistent values. */
SsStatus ss_nonlinear_adaptive_progress_guard_restore(
    PetscInt checkpoint_iteration, uint64_t successful_sample_count,
    PetscReal maximum_successful_convergence,
    PetscReal minimum_successful_convergence_ratio,
    PetscReal maximum_successful_convergence_ratio,
    SsNonlinearAdaptiveProgressGuard *guard);
/** Bounds of the adaptive guard for the next solve; pure.
 *
 * With at least ::SS_NONLINEAR_ADAPTIVE_GUARD_MINIMUM_SAMPLES samples and a
 * worst sampled contraction `q < 1`, the ratio bound is `sqrt(q)` and the
 * residual bound is the largest sample divided by `sqrt(q)`, so no observed
 * successful sample would be rejected.
 *
 * @param[in] guard Guard.
 * @param[in] maximum_iterations Newton iteration limit.
 * @param[out] enabled Nonzero when the guard is armed.
 * @param[out] threshold Residual bound, or 0.
 * @param[out] convergence_ratio_threshold Ratio bound, or 0.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_nonlinear_adaptive_progress_guard_threshold(
    const SsNonlinearAdaptiveProgressGuard *guard, PetscInt maximum_iterations,
    int *enabled, PetscReal *threshold, PetscReal *convergence_ratio_threshold);
/** Add the checkpoint residual of a successful solve to the guard; pure.
 * @param[in,out] guard Guard.
 * @param[in] successful_solve Record of a committed solve.
 * @param[out] sample_observed Nonzero when the solve reached the checkpoint.
 * @param[out] checkpoint_convergence Sampled residual.
 * @return ::SS_OK, ::SS_ERROR_INTEGRITY or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_nonlinear_adaptive_progress_guard_observe(
    SsNonlinearAdaptiveProgressGuard *guard,
    const SsNonlinearSolveRecord *successful_solve, int *sample_observed,
    PetscReal *checkpoint_convergence);
/** Update the regularization blend from the last step length; pure.
 *
 * A rejected step doubles the blend, a step below 0.1 grows it by `2^{1/4}`,
 * a step up to 0.5 keeps it, and a longer step shrinks it by `sqrt(2)` down to
 * the minimum. A blend above 1 fails with ::SS_NONLINEAR_FAILED_REGULARIZATION.
 *
 * @param[in] current Current blend in (0, 1].
 * @param[in] minimum Smallest blend in (0, 1].
 * @param[in] alpha Last step length in [0, 1].
 * @param[out] record Update.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_nonlinear_regularization_update(PetscReal current, PetscReal minimum,
                                            PetscReal alpha,
                                            SsNonlinearRegularizationRecord *record);
/** Search a step length along the Newton direction.
 * @param[in] policy Policy with the trial budget and criterion.
 * @param[in] current_residual_norm Residual norm at alpha zero.
 * @param[in] evaluate Trial evaluation; collective when the trials are.
 * @param[in] context Context of @p evaluate.
 * @param[out] record Trials and outcome.
 * @return ::SS_OK (see `record->outcome`) or the status of @p evaluate. */
SsStatus ss_nonlinear_line_search(const SsNonlinearPolicy *policy,
                                  PetscReal current_residual_norm,
                                  SsNonlinearTrialEvaluation evaluate, void *context,
                                  SsNonlinearLineSearchRecord *record);
/** Combine the control and residual directions so that the work is kept;
 * collective.
 *
 * `combined = residual_direction + c first_direction` with
 * `c = -(g^T residual_direction) / (g^T first_direction)`, where `g` is the
 * control gradient.
 *
 * @param[in] control_gradient Work gradient `g`.
 * @param[in] first_direction Control direction.
 * @param[in] residual_direction Residual direction.
 * @param[in] relative_tolerance Singularity tolerance of the denominator.
 * @param[out] combined_direction Combined direction.
 * @param[out] record Combination and outcome.
 * @return ::SS_OK (see `record->outcome`), ::SS_ERROR_INVALID_ARGUMENT or
 *   ::SS_ERROR_INTERNAL. */
SsStatus ss_nonlinear_augmented_direction(Vec control_gradient, Vec first_direction,
                                          Vec residual_direction,
                                          PetscReal relative_tolerance,
                                          Vec combined_direction,
                                          SsNonlinearAugmentedRecord *record);
/** Take a controlled step and rescale the trial to the target work;
 * collective.
 * @param[in] parent Parent state.
 * @param[in] direction Combined direction.
 * @param[in] control_gradient Work gradient.
 * @param[in] target_work Work to restore.
 * @param[in] control Current control.
 * @param[in] control_increment Control increment of the direction.
 * @param[in] alpha Step length.
 * @param[out] trial Rescaled trial state.
 * @param[out] updated_control `control + alpha control_increment`.
 * @param[out] work_after_rescale Work of @p trial.
 * @return ::SS_OK, ::SS_ERROR_INTEGRITY for a vanishing work,
 *   ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_INTERNAL. */
SsStatus ss_nonlinear_apply_augmented_update(
    Vec parent, Vec direction, Vec control_gradient, PetscReal target_work,
    PetscReal control, PetscReal control_increment, PetscReal alpha, Vec trial,
    PetscReal *updated_control, PetscReal *work_after_rescale);
/** Solve the elastic problem for the predictor and its reference work;
 * collective.
 * @param[in] problem Problem with the elastic operator and reference load.
 * @param[in] scale Load scale of the predictor.
 * @param[out] predictor Scaled elastic displacement.
 * @param[out] unscaled_reference_work Work of the unscaled elastic solution.
 * @param[out] setup_record Linear setup.
 * @param[out] solve_record Linear solve.
 * @return ::SS_OK, ::SS_ERROR_NUMERICAL when the solve fails, or the agreed
 *   failure status. */
SsStatus ss_nonlinear_elastic_predictor(const SsNonlinearProblem *problem,
                                        PetscReal scale, Vec predictor,
                                        PetscReal *unscaled_reference_work,
                                        SsLinearSetupRecord *setup_record,
                                        SsLinearSolveRecord *solve_record);
/** Damped, regularized Newton solve at a fixed factor; collective.
 *
 * Numerical failures are reported in `record->outcome` with ::SS_OK; the
 * accepted state is written only on success.
 *
 * @param[in] problem Problem.
 * @param[in] policy Newton policy.
 * @param[in] factor Positive load or strength-reduction factor.
 * @param[in] parent_state Initial state; unchanged.
 * @param[out] accepted_state Converged state.
 * @param[out] record Solve record.
 * @return ::SS_OK, ::SS_ERROR_CANCELLED, ::SS_ERROR_INVALID_ARGUMENT or the
 *   agreed failure status. */
SsStatus ss_nonlinear_fixed_solve(const SsNonlinearProblem *problem,
                                  const SsNonlinearPolicy *policy, PetscReal factor,
                                  Vec parent_state, Vec accepted_state,
                                  SsNonlinearSolveRecord *record);
/** Newton solve under work control, with the control as an extra unknown;
 * collective.
 * @param[in] problem Problem.
 * @param[in] policy Newton policy.
 * @param[in] mode Control mode.
 * @param[in] target_work Work the solution must have.
 * @param[in] incoming_control Initial control.
 * @param[in] parent_state Initial state; unchanged.
 * @param[out] accepted_state Converged state.
 * @param[out] accepted_control Converged control.
 * @param[out] record Solve record.
 * @return As ss_nonlinear_fixed_solve(). */
SsStatus ss_nonlinear_work_controlled_solve(
    const SsNonlinearProblem *problem, const SsNonlinearPolicy *policy,
    SsNonlinearControlMode mode, PetscReal target_work, PetscReal incoming_control,
    Vec parent_state, Vec accepted_state, PetscReal *accepted_control,
    SsNonlinearSolveRecord *record);

#endif
