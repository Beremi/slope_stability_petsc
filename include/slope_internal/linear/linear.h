/** @file
 * @brief Krylov solves under explicit stopping, acceptance and memory
 * policies, with PMG preconditioning, deflation and HPDDM recycling.
 *
 * Functions that take a solver or recycle space are collective on its
 * communicator unless marked local, and are called from the master thread.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_LINEAR_LINEAR_H
#define SS_LINEAR_LINEAR_H

#include "slope_internal/core/status.h"
#include "slope_internal/deflation/deflation.h"
#include "slope_internal/pmg/pmg.h"

#include <petscksp.h>

#include <stddef.h>
#include <stdint.h>

/** Capacity of a policy identity or effective-setting string. */
#define SS_LINEAR_IDENTITY_CAPACITY 64U
/** Capacity of an options prefix. */
#define SS_LINEAR_PREFIX_CAPACITY 32U
/** Largest number of policies in a fallback chain. */
#define SS_LINEAR_MAX_POLICIES 4U

/** Algebraic class that a policy assumes of the operator. */
typedef enum {
  SS_LINEAR_OPERATOR_GENERAL = 0,              /**< Any nonsingular operator. */
  SS_LINEAR_OPERATOR_SYMMETRIC_INDEFINITE = 1, /**< Symmetric, indefinite. */
  SS_LINEAR_OPERATOR_SPD_VERIFIED = 2,         /**< Measured SPD. */
  SS_LINEAR_OPERATOR_DIRECT_VALIDATION = 3     /**< Small direct validation solve. */
} SsLinearOperatorClass;

/** Krylov method. */
typedef enum {
  SS_LINEAR_METHOD_FGMRES = 0,  /**< Flexible GMRES. */
  SS_LINEAR_METHOD_MINRES = 1,  /**< MINRES. */
  SS_LINEAR_METHOD_CG = 2,      /**< Conjugate gradients. */
  SS_LINEAR_METHOD_PREONLY = 3, /**< Preconditioner only (direct solve). */
  /** Flexible GCRO-DR through PETSc KSPHPDDM. Each restart cycle forms the
   * full Krylov basis and compresses it to harmonic Ritz recycle vectors. */
  SS_LINEAR_METHOD_HPDDM_GCRODR = 4,
  /** Flexible conjugate gradients for symmetric positive-definite operators
   * and iteration-dependent positive preconditioners. */
  SS_LINEAR_METHOD_FCG = 5
} SsLinearMethod;

/** Preconditioner. */
typedef enum {
  SS_LINEAR_PC_NONE = 0,   /**< None. */
  SS_LINEAR_PC_JACOBI = 1, /**< Jacobi. */
  SS_LINEAR_PC_LU = 2,     /**< Direct LU. */
  SS_LINEAR_PC_PMG = 3     /**< Polynomial multigrid. */
} SsLinearPcKind;

/** Side of preconditioning. */
typedef enum {
  SS_LINEAR_SIDE_LEFT = 0, /**< Left preconditioning. */
  SS_LINEAR_SIDE_RIGHT = 1 /**< Right preconditioning. */
} SsLinearPcSide;

/** Stopping test of a solve. */
typedef enum {
  /** Verify convergence against the original, unprojected linear system. */
  SS_LINEAR_STOP_ORIGINAL_RHS_TRUE_RESIDUAL = 0,
  /** Match PETSc's stopping test on the deflated projected system. */
  SS_LINEAR_STOP_PROJECTED_RHS_PETSC = 1,
  /** Stop when `||B_k r_k||_2 <= max(atol, rtol ||b||_2)`; FCG already forms
   * `B_k r_k`, so this needs only its existing norm reduction. */
  SS_LINEAR_STOP_PRECONDITIONED_RESIDUAL_ORIGINAL_RHS = 2,
  /** PETSc FCG's natural norm `sqrt(|r_k^T B_k r_k|)`, relative to its value
   * at the initial iterate. */
  SS_LINEAR_STOP_NATURAL_RESIDUAL_INITIAL = 3
} SsLinearStoppingCondition;

/** Initial guess of a solve. */
typedef enum {
  /** Start from zero, or from the deflation coarse component when present. */
  SS_LINEAR_INITIAL_GUESS_COARSE_ONLY = 0,
  /** Add the supplied previous solution in the current A-orthogonal
   * complement. */
  SS_LINEAR_INITIAL_GUESS_PREVIOUS_COMPLEMENT = 1,
  /** Add it only when an explicit residual check proves it improves the
   * start. */
  SS_LINEAR_INITIAL_GUESS_PREVIOUS_COMPLEMENT_IF_BETTER = 2
} SsLinearInitialGuessPolicy;

/** Gram-Schmidt variant of the outer FGMRES. */
typedef enum {
  SS_LINEAR_ORTHOGONALIZATION_PETSC = 0,         /**< PETSc default. */
  SS_LINEAR_ORTHOGONALIZATION_CGS_REFINED = 1,   /**< Classical, always refined. */
  SS_LINEAR_ORTHOGONALIZATION_CGS_UNREFINED = 2, /**< Classical, never refined. */
  SS_LINEAR_ORTHOGONALIZATION_MGS = 3            /**< Modified Gram-Schmidt. */
} SsLinearOrthogonalization;

/** Krylov vectors kept as deflation candidates. */
typedef enum {
  SS_LINEAR_KRYLOV_CAPTURE_NONE = 0, /**< Keep none. */
  /** Retain every projected preconditioned flexible-Krylov direction `Z_j`;
   * their span contains the FGMRES or FCG solution update. */
  SS_LINEAR_KRYLOV_CAPTURE_FULL_CORRECTION_SPACE = 1
} SsLinearKrylovCapture;

/** Classified outcome of a solve. */
typedef enum {
  SS_LINEAR_SOLVE_CONVERGED = 0,                    /**< Accepted. */
  SS_LINEAR_SOLVE_DIVERGED_KSP = 1,                 /**< The KSP diverged. */
  SS_LINEAR_SOLVE_RESIDUAL_VERIFICATION_FAILED = 2, /**< True residual too large. */
  SS_LINEAR_SOLVE_NONFINITE = 3,                    /**< Nonfinite norms. */
  SS_LINEAR_SOLVE_SETUP_FAILED = 4,                 /**< Setup failed. */
  SS_LINEAR_SOLVE_RESOURCE_LIMIT = 5,               /**< Memory admission failed. */
  SS_LINEAR_SOLVE_CANCELLED = 6                     /**< Cancelled. */
} SsLinearSolveStatus;

/** Measured or declared properties of the operator and preconditioner. */
typedef struct {
  int finite_action;       /**< The action is finite. */
  int measured_symmetric;  /**< Symmetry was measured. */
  int positive_energy;     /**< Positive energy was measured. */
  int pc_symmetric;        /**< The preconditioner is symmetric. */
  int pc_positive;         /**< The preconditioner is positive. */
  int pc_variable;         /**< The preconditioner varies between iterations. */
  int nonlinear_hydraulic; /**< The operator is the seepage Jacobian. */
} SsLinearOperatorProperties;

/** One Krylov configuration. */
typedef struct {
  char identity[SS_LINEAR_IDENTITY_CAPACITY];      /**< Policy identity. */
  char options_prefix[SS_LINEAR_PREFIX_CAPACITY];  /**< PETSc options prefix. */
  SsLinearOperatorClass operator_class;            /**< Assumed operator class. */
  SsLinearMethod method;                           /**< Krylov method. */
  SsLinearPcKind pc;                               /**< Preconditioner. */
  SsLinearPcSide side;                             /**< Preconditioning side. */
  SsLinearStoppingCondition stopping_condition;    /**< Stopping test. */
  SsLinearInitialGuessPolicy initial_guess_policy; /**< Initial guess. */
  SsLinearOrthogonalization orthogonalization;     /**< FGMRES orthogonalization. */
  PetscReal rtol;                                  /**< Relative tolerance. */
  PetscReal atol;                                  /**< Absolute tolerance. */
  PetscReal dtol;                                  /**< Divergence tolerance. */
  PetscInt maximum_iterations;                     /**< Iteration limit. */
  PetscInt restart;                                /**< Restart length. */
  PetscInt recycle_dimension;                      /**< HPDDM recycle dimension. */
  size_t vector_memory_cap_bytes; /**< Cap of the Krylov vector payload. */
} SsLinearPolicy;

/** Policies tried in order until one sets up. */
typedef struct {
  size_t count;                                    /**< Number of policies. */
  SsLinearPolicy policies[SS_LINEAR_MAX_POLICIES]; /**< Policies in order. */
} SsLinearPolicyChain;

/** Setup outcome of one policy of a chain. */
typedef struct {
  char identity[SS_LINEAR_IDENTITY_CAPACITY]; /**< Policy identity. */
  SsStatus setup_status;                      /**< Agreed setup status. */
  int selected;                               /**< Nonzero for the selected policy. */
} SsLinearSetupAttempt;

/** Setup outcome and the effective PETSc configuration. */
typedef struct {
  uint64_t structure_generation;   /**< Pattern generation of the operator. */
  uint64_t value_generation;       /**< Value generation of the operator. */
  PetscObjectState operator_state; /**< PETSc state of the operator. */
  size_t attempt_count;            /**< Policies tried. */
  SsLinearSetupAttempt attempts[SS_LINEAR_MAX_POLICIES]; /**< Outcome of each. */
  char selected_identity[SS_LINEAR_IDENTITY_CAPACITY];   /**< Selected policy. */
  char effective_ksp[SS_LINEAR_IDENTITY_CAPACITY];       /**< PETSc KSP type. */
  char effective_pc[SS_LINEAR_IDENTITY_CAPACITY];        /**< PETSc PC type. */
  SsLinearPcSide effective_side;                         /**< Effective side. */
  KSPNormType effective_norm_type;                       /**< KSP norm type. */
  PetscBool effective_initial_guess_nonzero;  /**< KSP nonzero initial guess. */
  PetscBool effective_initial_guess_knoll;    /**< KSP Knoll initial guess. */
  PetscBool effective_error_if_not_converged; /**< KSP error on divergence. */
  PetscReal effective_rtol;                   /**< Effective relative tolerance. */
  PetscReal effective_atol;                   /**< Effective absolute tolerance. */
  PetscReal effective_dtol;                   /**< Effective divergence tolerance. */
  PetscInt effective_restart;                 /**< Effective restart. */
  PetscInt effective_recycle_dimension;       /**< Effective recycle dimension. */
  /** HPDDM recycle variant. */
  char effective_recycle_variant[SS_LINEAR_IDENTITY_CAPACITY];
  /** HPDDM recycle target. */
  char effective_recycle_target[SS_LINEAR_IDENTITY_CAPACITY];
  /** HPDDM recycle strategy. */
  char effective_recycle_strategy[SS_LINEAR_IDENTITY_CAPACITY];
  /** Effective FGMRES orthogonalization. */
  char effective_orthogonalization[SS_LINEAR_IDENTITY_CAPACITY];
  PetscInt effective_maximum_iterations; /**< Effective iteration limit. */
  /** Aggregate Krylov, capture and deflation vector payload at admission. */
  size_t admitted_vector_bytes;
  /** Estimated peak dense deflation scratch on each rank. */
  size_t admitted_deflation_dense_bytes_per_rank;
  uint64_t setup_count;       /**< Setups performed so far. */
  int reused;                 /**< Nonzero when the previous setup was reused. */
  int pmg_selected;           /**< Nonzero when PMG preconditions. */
  int deflation_projected_pc; /**< Nonzero when PMG runs on the projected system. */
  PetscInt recycled_dimension_supplied; /**< Recycle vectors supplied to HPDDM. */
  SsPmgSetupRecord pmg_setup;           /**< PMG setup outcome. */
} SsLinearSetupRecord;

/** Outcome of one solve. */
typedef struct {
  SsLinearSolveStatus status; /**< Classified outcome. */
  KSPConvergedReason reason;  /**< PETSc converged reason. */
  PetscInt iterations;        /**< Krylov iterations. */
  /** Relative tolerance actually requested for this solve; it differs from
   * the policy when a nonlinear forcing policy tightens one right-hand side. */
  PetscReal effective_rtol;
  PetscReal reported_norm;      /**< Final norm reported by PETSc. */
  PetscReal true_residual_norm; /**< `||b - A x||_2` of the original system. */
  PetscReal rhs_norm;           /**< `||b||_2`. */
  PetscReal stopping_rhs_norm;  /**< Right-hand-side norm of the stopping test. */
  /** Original-system residual limit, independent of the KSP stopping norm. */
  PetscReal acceptance_threshold;
  PetscReal input_guess_norm;                   /**< Norm of the supplied guess. */
  PetscReal coarse_initial_residual_norm;       /**< Residual of the coarse start. */
  PetscReal candidate_initial_residual_norm;    /**< Residual with the guess. */
  PetscReal initial_residual_norm;              /**< Residual of the start used. */
  int input_guess_tested;                       /**< The guess was tested. */
  int input_guess_accepted;                     /**< The guess was used. */
  PetscInt recycled_dimension_supplied;         /**< Recycle vectors supplied. */
  size_t deflation_dimension_supplied;          /**< Deflation columns supplied. */
  size_t krylov_vectors_captured;               /**< Krylov directions captured. */
  size_t krylov_vectors_promoted;               /**< Captured directions promoted. */
  size_t krylov_vectors_rejected;               /**< Captured directions rejected. */
  size_t krylov_basis_after_promotion;          /**< Deflation columns afterwards. */
  SsLinearStoppingCondition stopping_condition; /**< Stopping test used. */
  uint64_t value_generation;        /**< Value generation of the operator. */
  uint64_t solve_count;             /**< Solves performed so far. */
  uint64_t cancellation_reductions; /**< Reductions spent on cancellation. */
} SsLinearSolveRecord;

/** Opaque HPDDM recycle space with transactions; see
 * ss_linear_recycle_create(). */
typedef struct SsLinearRecycle SsLinearRecycle;

/** State of a recycle space. */
typedef struct {
  PetscInt dimension;              /**< Current recycle dimension. */
  uint64_t update_count;           /**< Updates so far. */
  uint64_t transaction_generation; /**< Generation of the open transaction. */
  uint64_t accepted_generation;    /**< Generation of the last commit. */
  int transaction_open;            /**< Nonzero while a transaction is open. */
} SsLinearRecycleDiagnostics;

/** Outcome of promoting captured Krylov directions to deflation candidates. */
typedef struct {
  size_t captured_count;     /**< Captured directions. */
  size_t promoted_count;     /**< Directions admitted. */
  size_t rejected_count;     /**< Directions dropped. */
  size_t active_basis_count; /**< Active deflation columns afterwards. */
} SsLinearKrylovPromotionRecord;

/** Linear problem; the solver borrows every object. */
typedef struct {
  Mat operator_handle;                /**< Operator `A`. */
  Mat preconditioner_operator_handle; /**< Matrix the preconditioner is built from. */
  SsLinearOperatorProperties properties; /**< Operator properties. */
  SsLinearPolicyChain policy_chain;      /**< Policies tried in order. */
  SsDeflation *deflation;                /**< Deflation store, or NULL. */
  SsPmg *pmg;                            /**< PMG hierarchy, or NULL. */
  SsLinearRecycle *krylov_recycle;       /**< HPDDM recycle space, or NULL. */
  SsLinearKrylovCapture krylov_capture;  /**< Krylov capture mode. */
  uint64_t structure_generation;         /**< Pattern generation of `A`. */
  uint64_t value_generation;             /**< Value generation of `A`. */
} SsLinearProblem;

/** Opaque Krylov solver; see ss_linear_create(). */
typedef struct SsLinearSolver SsLinearSolver;

/** Process-local, nonblocking cancellation poll.
 * @param context Poll context.
 * @return Nonzero when cancellation was requested on this rank. */
typedef int (*SsLinearCancellation)(void *context);

/** Observe PETSc's reported Krylov norm on the calling rank.
 * @param context Observer context.
 * @param iteration Krylov iteration.
 * @param residual_norm Reported norm. */
typedef void (*SsKrylovIterationObserver)(void *context, PetscInt iteration,
                                          PetscReal residual_norm);

/** Optional observers of a solver. */
typedef struct {
  SsKrylovIterationObserver krylov_iteration; /**< Krylov iterations, or NULL. */
  SsPmgApplyObserver pmg_apply;               /**< PMG cycles, or NULL. */
  void *context;                              /**< Context of both observers. */
} SsLinearObserver;

/** Set or clear optional noncollective observers for subsequent solves.
 * @param[in,out] solver Solver.
 * @param[in] observer Observers; NULL clears them. */
void ss_linear_set_observer(SsLinearSolver *solver, const SsLinearObserver *observer);

/** Configure cooperative cancellation for subsequent collective solves.
 *
 * The poll is evaluated locally and agreed with a reduction at Krylov safe
 * points, so all ranks stop together.
 *
 * @param[in,out] solver Solver.
 * @param[in] poll Local poll, or NULL to disable cancellation.
 * @param[in] context Context of @p poll. */
void ss_linear_set_cancellation(SsLinearSolver *solver, SsLinearCancellation poll,
                                void *context);

/** Check that a policy is consistent and suits the operator properties; local.
 * @param[in] policy Policy.
 * @param[in] properties Operator properties.
 * @return ::SS_OK, ::SS_ERROR_TYPE_MISMATCH when the method does not suit the
 *   operator, or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_linear_policy_validate(const SsLinearPolicy *policy,
                                   const SsLinearOperatorProperties *properties);
/** Classify a finished solve against the original-system residual; pure.
 * @param[in] policy Policy with the tolerances.
 * @param[in] reason PETSc converged reason.
 * @param[in] reported_norm Final norm reported by PETSc.
 * @param[in] rhs_norm `||b||_2`.
 * @param[in] true_residual_norm `||b - A x||_2`.
 * @param[out] status Classified outcome.
 * @param[out] threshold `max(atol, rtol ||b||_2)`.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_linear_classify_acceptance(const SsLinearPolicy *policy,
                                       KSPConvergedReason reason,
                                       PetscReal reported_norm, PetscReal rhs_norm,
                                       PetscReal true_residual_norm,
                                       SsLinearSolveStatus *status,
                                       PetscReal *threshold);

/** Check the global Krylov, capture and base-vector payload against a policy
 * cap; pure.
 * @param[in] global_rows Global vector length.
 * @param[in] policy Policy with the cap.
 * @param[in] capture Krylov capture mode.
 * @param[in] basis_bytes Payload of other bases (deflation).
 * @param[out] required_bytes Required payload, also on a cap rejection.
 * @return ::SS_OK, ::SS_ERROR_RESOURCE above the cap, ::SS_ERROR_OVERFLOW or
 *   ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_linear_admit_vectors(PetscInt global_rows, const SsLinearPolicy *policy,
                                 SsLinearKrylovCapture capture, size_t basis_bytes,
                                 size_t *required_bytes);

/** Create a solver; collective.
 * @param[in] communicator Communicator of the vectors; duplicated.
 * @param[out] result New solver, released with ss_linear_destroy().
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_linear_create(MPI_Comm communicator, SsLinearSolver **result);
/** Create an empty HPDDM recycle space; collective.
 * @param[in] communicator Communicator of the vectors; duplicated.
 * @param[out] result New recycle space, released with
 *   ss_linear_recycle_destroy().
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_linear_recycle_create(MPI_Comm communicator, SsLinearRecycle **result);
/** Open a recycle transaction for an analysis attempt; collective.
 * @param[in,out] recycle Recycle space without an open transaction.
 * @param[in] transaction_generation Generation of the attempt.
 * @return ::SS_OK, ::SS_ERROR_INVALID_STATE or ::SS_ERROR_INTERNAL. */
SsStatus ss_linear_recycle_begin(SsLinearRecycle *recycle,
                                 uint64_t transaction_generation);
/** Keep the recycle space of an accepted attempt; collective.
 * @param[in,out] recycle Recycle space with an open transaction.
 * @param[in] accepted_generation Generation of the accepted state.
 * @return ::SS_OK or ::SS_ERROR_INVALID_STATE. */
SsStatus ss_linear_recycle_commit(SsLinearRecycle *recycle,
                                  uint64_t accepted_generation);
/** Restore the recycle space of the last accepted state; collective.
 * @param[in,out] recycle Recycle space with an open transaction.
 * @return ::SS_OK or ::SS_ERROR_INVALID_STATE. */
SsStatus ss_linear_recycle_rollback(SsLinearRecycle *recycle);
/** State of a recycle space; local.
 * @param[in] recycle Recycle space.
 * @param[out] diagnostics State.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_linear_recycle_diagnostics(const SsLinearRecycle *recycle,
                                       SsLinearRecycleDiagnostics *diagnostics);
/** Destroy a recycle space; collective.
 * @param[in,out] recycle Recycle pointer, set to NULL; `*recycle` may be NULL.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_linear_recycle_destroy(SsLinearRecycle **recycle);
/** Set up the solver for a problem; collective.
 *
 * The policies of the chain are tried in order and the first that sets up on
 * every rank is selected. A setup with the same chain, operator generations
 * and PETSc options is reused.
 *
 * @param[in,out] solver Solver.
 * @param[in] problem Problem; borrowed until the next setup or destroy.
 * @param[out] record Setup outcome.
 * @return ::SS_OK or the agreed status of the last policy tried. */
SsStatus ss_linear_setup(SsLinearSolver *solver, const SsLinearProblem *problem,
                         SsLinearSetupRecord *record);
/** Solve with the selected policy; collective.
 *
 * Convergence is verified against the original system; a solve that ran but
 * did not converge returns ::SS_OK with `record->status` set accordingly.
 *
 * @param[in,out] solver Set-up solver.
 * @param[in] rhs Right-hand side.
 * @param[in,out] solution Initial guess (per the policy) and solution.
 * @param[out] record Solve outcome.
 * @return ::SS_OK, ::SS_ERROR_CANCELLED, or the agreed failure status. */
SsStatus ss_linear_solve(SsLinearSolver *solver, Vec rhs, Vec solution,
                         SsLinearSolveRecord *record);
/** Solve with a relative tolerance that overrides the policy for this solve.
 * @param[in,out] solver Set-up solver.
 * @param[in] rhs Right-hand side.
 * @param[in,out] solution Initial guess and solution.
 * @param[in] relative_tolerance Positive relative tolerance.
 * @param[out] record Solve outcome.
 * @return As ss_linear_solve(). */
SsStatus ss_linear_solve_with_relative_tolerance(SsLinearSolver *solver, Vec rhs,
                                                 Vec solution,
                                                 PetscReal relative_tolerance,
                                                 SsLinearSolveRecord *record);
/** Offer the Krylov directions captured in the last solve to the deflation
 * store as provisional candidates; collective.
 * @param[in,out] solver Solver after a capturing solve.
 * @param[in] provenance Provenance of the candidates.
 * @param[in,out] solve_record Record of that solve; promotion counts added.
 * @param[out] record Promotion outcome.
 * @return ::SS_OK, ::SS_ERROR_INTEGRITY, ::SS_ERROR_INVALID_ARGUMENT or
 *   ::SS_ERROR_INTERNAL. */
SsStatus ss_linear_promote_captured_krylov(SsLinearSolver *solver,
                                           const char *provenance,
                                           SsLinearSolveRecord *solve_record,
                                           SsLinearKrylovPromotionRecord *record);
/** Finish an attempt: update the recycle space and drop captured directions,
 * keeping compatible PETSc work; collective.
 * @param[in,out] solver Solver.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_linear_finish_attempt(SsLinearSolver *solver);

/** Drop the setup so that the next ss_linear_setup() rebuilds it; collective.
 * @param[in,out] solver Solver.
 * @param[in] generation_mask Nonzero mask of the invalidated generations.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT, or the status of the final
 *   recycle update. */
SsStatus ss_linear_invalidate(SsLinearSolver *solver, uint64_t generation_mask);
/** Destroy a solver and its PETSc objects; collective.
 * @param[in,out] solver Solver pointer, set to NULL; `*solver` may be NULL.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_INTERNAL. */
SsStatus ss_linear_destroy(SsLinearSolver **solver);

#endif
