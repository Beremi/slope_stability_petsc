/** @file
 * @brief Continuation analyses: limit load and strength reduction with
 * direct or indirect continuation, with attempt transactions and restart.
 *
 * Drivers are collective on the problem's communicator and run on the master
 * thread; records are identical on all ranks.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_ANALYSIS_ANALYSIS_H
#define SS_ANALYSIS_ANALYSIS_H

#include "slope_internal/core/status.h"
#include "slope_internal/nonlinear/nonlinear.h"

#include <mpi.h>
#include <petscvec.h>

#include <stddef.h>
#include <stdint.h>

/** Largest number of accepted points, including the elastic reference of
 * limit load. */
#define SS_ANALYSIS_POINT_CAPACITY 128U
/** Largest number of recorded continuation attempts. */
#define SS_ANALYSIS_ATTEMPT_CAPACITY (SS_ANALYSIS_POINT_CAPACITY * 5U)
/** Capacity of an analysis identity string. */
#define SS_ANALYSIS_ID_CAPACITY 72U

/** Analysis kind. */
typedef enum {
  SS_ANALYSIS_LIMIT_LOAD = 1,  /**< Limit load under work control. */
  SS_ANALYSIS_SSR_DIRECT = 2,  /**< Strength reduction, direct continuation. */
  SS_ANALYSIS_SSR_INDIRECT = 3 /**< Strength reduction, indirect continuation. */
} SsAnalysisMethod;

/** Why a continuation stopped. */
typedef enum {
  SS_ANALYSIS_STOP_NONE = 0,                 /**< Not stopped. */
  SS_ANALYSIS_STOP_SCALED_SLOPE_REACHED = 1, /**< Scaled slope below tolerance. */
  SS_ANALYSIS_STOP_MINIMUM_STEP_REACHED = 2, /**< Step reduced to the minimum. */
  SS_ANALYSIS_STOP_WORK_HORIZON_REACHED = 3, /**< Work horizon reached. */
  SS_ANALYSIS_STOP_DIRECT_CONTROL_HORIZON_REACHED = 4, /**< Direct control horizon. */
  SS_ANALYSIS_STOP_ACCEPTED_STEP_LIMIT = 5, /**< Requested accepted states reached. */
  SS_ANALYSIS_STOP_CONSECUTIVE_RETRY_LIMIT = 6, /**< Too many rejected attempts. */
  SS_ANALYSIS_STOP_NEWTON_FAILURE = 7,          /**< Newton failed. */
  SS_ANALYSIS_STOP_LINEAR_FAILURE = 8,          /**< A linear solve failed. */
  SS_ANALYSIS_STOP_SINGULAR_CONSTRAINT = 9,     /**< Singular control constraint. */
  SS_ANALYSIS_STOP_NONFINITE = 10,              /**< Nonfinite values. */
  SS_ANALYSIS_STOP_CANCELLED = 11               /**< Cancelled. */
} SsAnalysisStopReason;

/** Next operation of a continuation that is resumed. */
typedef enum {
  SS_ANALYSIS_NEXT_FIRST_SEED = 1,           /**< Compute the first seed point. */
  SS_ANALYSIS_NEXT_SECOND_SEED = 2,          /**< Compute the second seed point. */
  SS_ANALYSIS_NEXT_CONTINUATION_ATTEMPT = 3, /**< Attempt the next step. */
  SS_ANALYSIS_NEXT_TERMINAL = 4              /**< Nothing left to do. */
} SsAnalysisNextOperation;

/** Predictor of a continuation attempt. */
typedef enum {
  SS_ANALYSIS_PREDICTOR_CURRENT_STATE = 1,       /**< Last accepted state. */
  SS_ANALYSIS_PREDICTOR_DISPLACEMENT_SECANT = 2, /**< Secant extrapolation. */
  SS_ANALYSIS_PREDICTOR_PAIRED_BETA = 3,         /**< Direct-SSR paired predictor. */
  /** Secant extrapolation with an anchored factor. */
  SS_ANALYSIS_PREDICTOR_DISPLACEMENT_SECANT_ANCHORED_FACTOR = 4
} SsAnalysisPredictor;

/** Step controller of work-controlled continuation. */
typedef enum {
  /** Preserve the frozen reference controller and its exact step topology. */
  SS_ANALYSIS_WORK_STEP_CLASSIC = 0,
  /** Adapt only from the secant predictor's dimensionless error and the
   * accepted curve shape; iteration counts do not enter this controller. */
  SS_ANALYSIS_WORK_STEP_PREDICTOR_ERROR = 1,
  /** Adapt from predictor trust and locally measured work efficiency. */
  SS_ANALYSIS_WORK_STEP_WORK_EFFICIENCY = 2
} SsAnalysisWorkStepController;

/** Predictor of indirect SSR. */
typedef enum {
  /** Extrapolate the last accepted displacement increment to the target work. */
  SS_ANALYSIS_INDIRECT_PREDICTOR_DISPLACEMENT_SECANT = 0,
  /** Scale the last accepted state radially to the target work. */
  SS_ANALYSIS_INDIRECT_PREDICTOR_SCALED_CURRENT = 1
} SsAnalysisIndirectPredictorMode;

/** Continuation policy. */
typedef struct {
  SsAnalysisMethod method;          /**< Analysis kind. */
  PetscReal nonlinear_tolerance;    /**< Newton tolerance; sets the direct-SSR
                                         sensitivity step `100 tol`. */
  PetscReal factor_initial;         /**< First factor of direct continuation. */
  PetscReal factor_step_initial;    /**< First factor step. */
  PetscReal factor_step_minimum;    /**< Smallest factor step. */
  PetscReal factor_maximum;         /**< Largest factor. */
  PetscReal work_step_initial;      /**< First work step. */
  PetscReal work_step_minimum;      /**< Smallest work step. */
  PetscReal work_maximum;           /**< Work horizon. */
  PetscReal direct_control_maximum; /**< Control horizon of direct SSR. */
  PetscReal scaled_slope_tolerance; /**< Stop when the scaled slope falls to it. */
  PetscReal factor_increment_stop;  /**< Stop when the factor increment falls
                                         below it; 0 disables. */
  PetscInt accepted_step_limit;     /**< Requested accepted steps. */
  PetscInt consecutive_retry_limit; /**< Consecutive rejected attempts allowed. */
  /** Newton iterations above which the work step does not grow; zero keeps
   * the reference controller's unrestricted growth. */
  PetscInt work_growth_iteration_limit;
  /** Hard Newton ceiling used only by the work-efficiency controller. */
  PetscInt work_efficiency_iteration_maximum;
  SsAnalysisWorkStepController work_step_controller;       /**< Work step controller. */
  SsAnalysisIndirectPredictorMode indirect_predictor_mode; /**< Indirect predictor. */
  /** Target predictor error of the adaptive controllers. */
  PetscReal work_predictor_error_target;
  PetscReal work_step_scale_minimum; /**< Smallest step scale per attempt. */
  PetscReal work_step_scale_maximum; /**< Largest step scale per attempt. */
  /** Finite difference of the indirect control sensitivity. */
  SsNonlinearControlSensitivityMethod indirect_sensitivity_method;
  PetscReal indirect_sensitivity_relative_step; /**< Its relative step. */
} SsAnalysisPolicy;

/** Solve at a fixed factor.
 * @param context Problem context.
 * @param factor Factor.
 * @param incoming Initial state.
 * @param solution Solution.
 * @param record Solve record.
 * @return ::SS_OK or the agreed failure status. */
typedef SsStatus (*SsAnalysisFixedSolve)(void *context, PetscReal factor, Vec incoming,
                                         Vec solution, SsNonlinearSolveRecord *record);
/** Solve under work control.
 * @param context Problem context.
 * @param method Analysis kind.
 * @param target_work Target work.
 * @param incoming_factor Initial factor.
 * @param incoming Initial state.
 * @param solution Solution.
 * @param accepted_factor Converged factor.
 * @param record Solve record.
 * @return ::SS_OK or the agreed failure status. */
typedef SsStatus (*SsAnalysisWorkSolve)(void *context, SsAnalysisMethod method,
                                        PetscReal target_work,
                                        PetscReal incoming_factor, Vec incoming,
                                        Vec solution, PetscReal *accepted_factor,
                                        SsNonlinearSolveRecord *record);
/** Total potential energy of a state.
 * @param context Problem context.
 * @param factor Factor.
 * @param state State.
 * @param total_potential Potential.
 * @return ::SS_OK or the agreed failure status. */
typedef SsStatus (*SsAnalysisTotalPotential)(void *context, PetscReal factor, Vec state,
                                             PetscReal *total_potential);
/** Largest nodal displacement magnitude of a state.
 * @param context Problem context.
 * @param state State.
 * @param maximum Global maximum.
 * @return ::SS_OK or the agreed failure status. */
typedef SsStatus (*SsAnalysisDisplacementMaximum)(void *context, Vec state,
                                                  PetscReal *maximum);
/** Capture an accepted state, for example for output.
 * @param context Problem context.
 * @param accepted_ordinal Ordinal of the accepted state.
 * @param state State.
 * @return ::SS_OK or the agreed failure status. */
typedef SsStatus (*SsAnalysisAcceptedStateCapture)(void *context,
                                                   size_t accepted_ordinal, Vec state);
/** Begin, commit or roll back the solver transaction of an attempt.
 * @param context Problem context.
 * @param attempt_ordinal Ordinal of the attempt.
 * @param parent_accepted_ordinal Accepted state the attempt starts from.
 * @return ::SS_OK or the agreed failure status. */
typedef SsStatus (*SsAnalysisAttemptTransaction)(void *context, size_t attempt_ordinal,
                                                 size_t parent_accepted_ordinal);
/** Process-local cancellation poll.
 * @param context Problem context.
 * @return Nonzero when cancellation was requested on this rank. */
typedef int (*SsAnalysisCancellation)(void *context);

/** Callbacks of a continuation analysis; each is collective. */
typedef struct {
  SsAnalysisFixedSolve solve_fixed;                      /**< Fixed-factor solve. */
  SsAnalysisWorkSolve solve_work_controlled;             /**< Work-controlled solve. */
  SsAnalysisTotalPotential total_potential;              /**< Potential energy. */
  SsAnalysisDisplacementMaximum displacement_maximum;    /**< Displacement maximum. */
  SsAnalysisAcceptedStateCapture capture_accepted_state; /**< State capture. */
  SsAnalysisAttemptTransaction begin_attempt;            /**< Attempt begin. */
  SsAnalysisAttemptTransaction commit_attempt;           /**< Attempt commit. */
  SsAnalysisAttemptTransaction rollback_attempt;         /**< Attempt rollback. */
  SsAnalysisCancellation cancelled;                      /**< Cancellation poll. */
} SsAnalysisOps;

/** Record of a continuation; see the struct definition below. */
typedef struct SsAnalysisRecord SsAnalysisRecord;

/** Role of a nonlinear solve inside one continuation attempt. */
typedef enum {
  SS_ANALYSIS_SOLVE_PRIMARY = 1, /**< Main solve of the attempt. */
  /** Direct-SSR perturbed (sensitivity) solve of the same attempt. */
  SS_ANALYSIS_SOLVE_SECONDARY = 2
} SsAnalysisSolveRole;

/**
 * Optional collective observer of every nonlinear solve the analysis runs,
 * called after the solve with its complete per-iteration record.
 *
 * @param observer Observer context.
 * @param attempt_ordinal Ordinal of the attempt.
 * @param role Role of the solve.
 * @param record Solve record.
 * @return ::SS_OK, or a status that stops the analysis.
 */
typedef SsStatus (*SsAnalysisSolveObserver)(void *observer, size_t attempt_ordinal,
                                            SsAnalysisSolveRole role,
                                            const SsNonlinearSolveRecord *record);

/**
 * Optional collective observer of an accepted continuation boundary.
 *
 * Called once per accepted point, at the next safe point after the point and
 * the controller state are recorded and before the cancellation check. The
 * record's `next_step`, `first_control` and `next_operation` hold the values a
 * stop at this boundary would publish, so a checkpoint written here resumes
 * exactly like one written after an accepted-step-limit stop.
 *
 * @param observer Observer context.
 * @param record Analysis record at the boundary.
 * @return ::SS_OK, or a status that stops the analysis.
 */
typedef SsStatus (*SsAnalysisBoundaryObserver)(void *observer,
                                               const SsAnalysisRecord *record);

/** Notify a live observer before the primary or perturbed Newton solve.
 * @param observer Observer context.
 * @param role Role of the solve.
 * @param target Target factor or work. */
typedef void (*SsAnalysisSolveStarted)(void *observer, SsAnalysisSolveRole role,
                                       PetscReal target);

/** Optional observers; all fields may be zero. */
typedef struct {
  SsAnalysisSolveStarted solve_started;         /**< Solve start, or NULL. */
  SsAnalysisSolveObserver solve_completed;      /**< Solve end, or NULL. */
  SsAnalysisBoundaryObserver accepted_boundary; /**< Accepted point, or NULL. */
  void *context;                                /**< Observer context. */
} SsAnalysisObserver;

/** Continuation problem; every object is borrowed. */
typedef struct {
  MPI_Comm communicator;       /**< Communicator of the field. */
  Vec reference_load;          /**< Reference load. */
  SsAnalysisOps operations;    /**< Callbacks. */
  void *context;               /**< Context of the callbacks. */
  SsAnalysisObserver observer; /**< Optional observers. */
} SsAnalysisProblem;

/** Summary of one nonlinear solve of an attempt. */
typedef struct {
  SsNonlinearOutcome outcome;              /**< Outcome. */
  PetscInt iterations;                     /**< Newton iterations. */
  size_t history_count;                    /**< Iterations recorded. */
  PetscReal initial_convergence;           /**< Initial stopping measure. */
  PetscReal minimum_convergence;           /**< Smallest stopping measure. */
  PetscReal final_convergence;             /**< Final stopping measure. */
  PetscReal final_relative_correction;     /**< Last relative correction. */
  PetscReal final_regularization;          /**< Last regularization blend. */
  PetscReal final_alpha;                   /**< Last step length. */
  uint64_t linear_setup_count;             /**< Linear setups. */
  uint64_t linear_solve_count;             /**< Linear solves. */
  uint64_t first_linear_iteration_count;   /**< Krylov iterations, first system. */
  uint64_t second_linear_iteration_count;  /**< Krylov iterations, second system. */
  SsLinearSolveRecord first_linear_solve;  /**< Last first-system solve. */
  SsLinearSolveRecord second_linear_solve; /**< Last second-system solve. */
} SsAnalysisSolveSummary;

/** Record of one continuation attempt. */
typedef struct {
  size_t attempt_ordinal;                     /**< Ordinal of the attempt. */
  size_t parent_accepted_ordinal;             /**< Accepted state it starts from. */
  PetscInt retry_ordinal;                     /**< Retry number of the step. */
  SsAnalysisPredictor predictor;              /**< Predictor. */
  PetscReal target;                           /**< Target factor or work. */
  PetscReal incoming_factor;                  /**< Factor of the predictor. */
  PetscReal incoming_control;                 /**< Control of the predictor. */
  PetscReal incoming_work;                    /**< Work of the predictor. */
  PetscReal accepted_factor;                  /**< Converged factor. */
  PetscReal accepted_control;                 /**< Converged control. */
  PetscReal accepted_work;                    /**< Converged work. */
  PetscReal requested_step;                   /**< Requested step. */
  PetscReal actual_step;                      /**< Step taken. */
  PetscReal predictor_correction_norm;        /**< Norm of the Newton correction. */
  PetscReal accepted_increment_norm;          /**< Norm of the accepted increment. */
  PetscReal predictor_error;                  /**< Dimensionless predictor error. */
  PetscReal work_efficiency_elasticity;       /**< Measured work efficiency. */
  PetscReal work_efficiency_iteration_budget; /**< Iteration budget. */
  PetscReal work_step_scale;                  /**< Scale applied to the next step. */
  PetscReal next_work_step;                   /**< Next work step. */
  PetscReal sensitivity_epsilon;              /**< Sensitivity step. */
  PetscReal sensitivity_numerator;            /**< Sensitivity numerator. */
  PetscReal sensitivity_denominator;          /**< Sensitivity denominator. */
  PetscReal beta;                         /**< Direct-SSR blend `min(1, eps/step)`. */
  PetscReal signed_scaled_slope;          /**< Scaled slope of the factor curve. */
  PetscInt newton_iterations;             /**< Newton iterations. */
  SsNonlinearOutcome nonlinear_outcome;   /**< Newton outcome. */
  SsAnalysisSolveSummary primary_solve;   /**< Primary solve. */
  SsAnalysisSolveSummary secondary_solve; /**< Sensitivity solve. */
  int successful;                         /**< Nonzero when the solve succeeded. */
  int committed;                          /**< Nonzero when the attempt was accepted. */
  int rolled_back;                        /**< Nonzero when it was rolled back. */
  int restartable; /**< Nonzero when a restart may start from it. */
} SsAnalysisAttemptRecord;

/** Record of one accepted point. */
typedef struct {
  size_t accepted_ordinal;        /**< Ordinal of the point. */
  size_t source_attempt_ordinal;  /**< Attempt that produced it. */
  PetscReal factor;               /**< Factor. */
  PetscReal control;              /**< Control. */
  PetscReal work;                 /**< Work. */
  PetscReal displacement_maximum; /**< Largest displacement magnitude. */
  PetscReal signed_scaled_slope;  /**< Scaled slope at the point. */
  int restartable;                /**< Nonzero when a restart may use it. */
} SsAnalysisPointRecord;

/** Record of a continuation analysis, identical on all ranks. */
struct SsAnalysisRecord {
  SsAnalysisMethod method;                           /**< Analysis kind. */
  char method_identity[SS_ANALYSIS_ID_CAPACITY];     /**< Method identity. */
  char predictor_identity[SS_ANALYSIS_ID_CAPACITY];  /**< Predictor identity. */
  char controller_identity[SS_ANALYSIS_ID_CAPACITY]; /**< Controller identity. */
  SsAnalysisStopReason stop_reason;     /**< Why the continuation stopped. */
  size_t point_count;                   /**< Accepted points. */
  size_t attempt_count;                 /**< Attempts. */
  size_t accepted_step_count;           /**< Accepted continuation steps. */
  PetscReal final_factor;               /**< Last factor. */
  PetscReal final_control;              /**< Last control. */
  PetscReal final_work;                 /**< Last work. */
  PetscReal final_displacement_maximum; /**< Last displacement maximum. */
  SsAnalysisPointRecord points[SS_ANALYSIS_POINT_CAPACITY];       /**< Points. */
  SsAnalysisAttemptRecord attempts[SS_ANALYSIS_ATTEMPT_CAPACITY]; /**< Attempts. */
  uint64_t commit_count;           /**< Committed attempts. */
  uint64_t rollback_count;         /**< Rolled-back attempts. */
  uint64_t newton_iteration_count; /**< Newton iterations. */
  uint64_t linear_setup_count;     /**< Linear setups. */
  uint64_t linear_solve_count;     /**< Linear solves. */
  uint64_t linear_iteration_count; /**< Krylov iterations. */
  int terminal_success;            /**< Nonzero when the analysis succeeded. */
  size_t accepted_ordinal_offset;  /**< Ordinal of the first point (restart). */
  PetscReal next_step;             /**< Step a resumed run would take next. */
  PetscReal first_control;         /**< Control of the first point. */
  /** Newton iterations of the previous step (work-efficiency controller). */
  PetscInt work_efficiency_previous_iterations;
  PetscReal work_efficiency_elasticity;   /**< Measured work efficiency. */
  int work_efficiency_elasticity_ready;   /**< Nonzero once it is measured. */
  SsAnalysisNextOperation next_operation; /**< Operation a resume would run. */
  int resumed_from_accepted_state;        /**< Nonzero for a restarted analysis. */
  /** Nonzero for an accepted point whose boundary observer has not run yet. */
  int boundary_pending;
};

/** Record of one direct-SSR point with its sensitivity solve. */
typedef struct {
  SsNonlinearOutcome outcome;               /**< Outcome. */
  PetscReal factor;                         /**< Strength-reduction factor. */
  PetscReal direct_control;                 /**< Direct control value. */
  PetscReal external_work;                  /**< External work. */
  PetscReal potential;                      /**< Total potential. */
  PetscReal perturbed_potential;            /**< Potential at the perturbed factor. */
  PetscReal epsilon;                        /**< Factor perturbation `100 tol`. */
  PetscReal beta;                           /**< `min(1, epsilon / factor_step)`. */
  SsNonlinearSolveRecord primary_solve;     /**< Solve at the factor. */
  SsNonlinearSolveRecord sensitivity_solve; /**< Solve at the perturbed factor. */
} SsAnalysisDirectPointRecord;

/** Controller state from which an analysis resumes. */
typedef struct {
  SsAnalysisMethod method;                /**< Analysis kind. */
  size_t accepted_step_offset;            /**< Accepted steps before the restart. */
  size_t accepted_ordinal;                /**< Ordinal of the restart point. */
  SsAnalysisNextOperation next_operation; /**< Operation to run next. */
  PetscReal previous_factor;              /**< Factor of the previous point. */
  PetscReal previous_control;             /**< Control of the previous point. */
  PetscReal previous_work;                /**< Work of the previous point. */
  PetscReal current_factor;               /**< Factor of the restart point. */
  PetscReal current_control;              /**< Control of the restart point. */
  PetscReal current_work;                 /**< Work of the restart point. */
  PetscReal current_signed_scaled_slope;  /**< Scaled slope at the restart point. */
  PetscReal first_control;                /**< Control of the first point. */
  PetscReal next_step;                    /**< Step to take next. */
  PetscInt retry_count;                   /**< Consecutive retries so far. */
  /** Newton iterations of the previous step. */
  PetscInt work_efficiency_previous_iterations;
  PetscReal work_efficiency_elasticity; /**< Measured work efficiency. */
  int work_efficiency_elasticity_ready; /**< Nonzero once it is measured. */
} SsAnalysisResumeState;

/** Maximum accepted continuation steps, excluding the elastic reference.
 * @param[in] method Analysis kind.
 * @return 128 for SSR, 127 for limit load. */
PetscInt ss_analysis_maximum_accepted_steps(SsAnalysisMethod method);

/** Check a continuation policy; pure.
 * @param[in] policy Policy.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_analysis_policy_validate(const SsAnalysisPolicy *policy);
/** Solve one direct-SSR point and its sensitivity solve; collective.
 * @param[in] problem Problem.
 * @param[in] policy Direct-SSR policy.
 * @param[in] factor Positive strength-reduction factor.
 * @param[in] factor_step Positive current factor step.
 * @param[in] incoming Initial state.
 * @param[out] solution Converged state.
 * @param[out] record Point record.
 * @return ::SS_OK (see `record->outcome`), ::SS_ERROR_INVALID_ARGUMENT or the
 *   agreed failure status. */
SsStatus ss_analysis_direct_point(const SsAnalysisProblem *problem,
                                  const SsAnalysisPolicy *policy, PetscReal factor,
                                  PetscReal factor_step, Vec incoming, Vec solution,
                                  SsAnalysisDirectPointRecord *record);
/** Run a continuation analysis from an initial state; collective.
 *
 * Each step is a transaction of attempts; rejected attempts are rolled back
 * through the problem callbacks. Cancellation is honored at safe points.
 *
 * @param[in] problem Problem.
 * @param[in] policy Policy.
 * @param[in] initial_state Initial state (elastic predictor or zero).
 * @param[out] accepted_state Last accepted state.
 * @param[out] record Analysis record.
 * @return ::SS_OK (see `record->stop_reason`), ::SS_ERROR_CANCELLED,
 *   ::SS_ERROR_INVALID_ARGUMENT or the agreed failure status. */
SsStatus ss_analysis_run(const SsAnalysisProblem *problem,
                         const SsAnalysisPolicy *policy, Vec initial_state,
                         Vec accepted_state, SsAnalysisRecord *record);
/** Resume a continuation from a restored accepted state; collective.
 * @param[in] problem Problem.
 * @param[in] policy Policy.
 * @param[in] previous_state State of the previous accepted point.
 * @param[in] current_state State of the restart point.
 * @param[out] accepted_state Last accepted state.
 * @param[in] resume Controller state at the restart point.
 * @param[out] record Analysis record.
 * @return As ss_analysis_run(). */
SsStatus ss_analysis_resume(const SsAnalysisProblem *problem,
                            const SsAnalysisPolicy *policy, Vec previous_state,
                            Vec current_state, Vec accepted_state,
                            const SsAnalysisResumeState *resume,
                            SsAnalysisRecord *record);
#endif
