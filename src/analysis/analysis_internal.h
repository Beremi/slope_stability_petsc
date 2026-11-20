/* Private interface shared by continuation drivers.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_ANALYSIS_INTERNAL_H
#define SS_ANALYSIS_INTERNAL_H

#include "slope_internal/analysis/analysis.h"

/** Agree on the worst nonlinear outcome across ranks; collective. */
SsStatus ss_analysis_collective_nonlinear_outcome(MPI_Comm communicator,
                                                  SsNonlinearOutcome local_outcome,
                                                  SsNonlinearOutcome *global_outcome);
/** Check that a real value is finite and identical on all ranks; collective. */
SsStatus ss_analysis_collective_real(MPI_Comm communicator, PetscReal local_value,
                                     PetscReal *global_value);
/** Agree on cancellation at a safe point; collective. */
SsStatus ss_analysis_collective_cancelled(const SsAnalysisProblem *problem,
                                          int *cancelled);
/** Whether a PETSc object lives on the expected communicator. */
int ss_analysis_same_communicator(MPI_Comm expected, PetscObject object);
/** Whether a nonlinear outcome is a success (converged or relaxed). */
int ss_analysis_nonlinear_succeeded(SsNonlinearOutcome outcome);
/** External work of a state against the reference load; collective. */
SsStatus ss_analysis_external_work(const SsAnalysisProblem *problem, Vec state,
                                   PetscReal *work);
/** Append and initialize a new attempt record. */
SsStatus ss_analysis_begin_attempt(SsAnalysisRecord *record,
                                   SsAnalysisPredictor predictor,
                                   PetscInt retry_ordinal, PetscReal target,
                                   PetscReal incoming_factor,
                                   PetscReal incoming_control, PetscReal incoming_work,
                                   PetscReal requested_step, PetscReal actual_step,
                                   SsAnalysisAttemptRecord **attempt);
/** Open the solver transaction of an attempt; collective. */
SsStatus ss_analysis_open_attempt(const SsAnalysisProblem *problem,
                                  const SsAnalysisAttemptRecord *attempt);
/** Commit an attempt's solver transaction and mark it accepted; collective. */
SsStatus ss_analysis_commit_attempt(const SsAnalysisProblem *problem,
                                    SsAnalysisRecord *record,
                                    SsAnalysisAttemptRecord *attempt);
/** Roll back an attempt's solver transaction; collective. */
SsStatus ss_analysis_rollback_attempt(const SsAnalysisProblem *problem,
                                      SsAnalysisRecord *record,
                                      SsAnalysisAttemptRecord *attempt);
/** Record an accepted point and capture its state; collective. */
SsStatus ss_analysis_append_point(const SsAnalysisProblem *problem,
                                  SsAnalysisRecord *record,
                                  size_t source_attempt_ordinal, PetscReal factor,
                                  PetscReal control, PetscReal work,
                                  PetscReal signed_scaled_slope, Vec state,
                                  int count_commit);
/** Set the stop reason and terminal flag of a record. */
void ss_analysis_finish(SsAnalysisRecord *record, SsAnalysisStopReason reason,
                        int terminal_success);
/** Report a finished nonlinear solve to the observer; collective. */
SsStatus ss_analysis_observe_solve(const SsAnalysisProblem *problem,
                                   const SsAnalysisAttemptRecord *attempt,
                                   SsAnalysisSolveRole role,
                                   const SsNonlinearSolveRecord *solve);
/** Record the resume state and run the boundary observer; collective. */
SsStatus ss_analysis_boundary(const SsAnalysisProblem *problem,
                              SsAnalysisRecord *record, PetscReal next_step,
                              PetscReal first_control,
                              SsAnalysisNextOperation next_operation);
/** Summarize a nonlinear solve into an attempt record. */
void ss_analysis_capture_solve(SsAnalysisRecord *record,
                               SsAnalysisSolveSummary *summary,
                               const SsNonlinearSolveRecord *solve);
/** Copy the controlled-solve sensitivity diagnostics into an attempt. */
void ss_analysis_capture_augmented_diagnostics(SsAnalysisAttemptRecord *attempt,
                                               const SsNonlinearSolveRecord *solve);
/** Run or resume limit-load continuation; collective. */
SsStatus ss_analysis_run_limit_load(const SsAnalysisProblem *problem,
                                    const SsAnalysisPolicy *policy, Vec previous_state,
                                    Vec initial_state, Vec accepted_state,
                                    const SsAnalysisResumeState *resume,
                                    SsAnalysisRecord *record);
/** Run or resume direct SSR continuation; collective. */
SsStatus ss_analysis_run_direct(const SsAnalysisProblem *problem,
                                const SsAnalysisPolicy *policy, Vec initial_state,
                                Vec accepted_state, const SsAnalysisResumeState *resume,
                                SsAnalysisRecord *record);
/** Whether a work-step controller adapts from the predictor error. */
int ss_analysis_work_step_uses_predictor_error(SsAnalysisWorkStepController controller);
/** Run or resume indirect SSR continuation; collective. */
SsStatus ss_analysis_run_indirect(const SsAnalysisProblem *problem,
                                  const SsAnalysisPolicy *policy, Vec previous_state,
                                  Vec initial_state, Vec accepted_state,
                                  const SsAnalysisResumeState *resume,
                                  SsAnalysisRecord *record);

/** Notify the live observer before an analysis invokes a numerical solve. */
void ss_analysis_observe_start(const SsAnalysisProblem *problem,
                               SsAnalysisSolveRole role, PetscReal target);

#endif
