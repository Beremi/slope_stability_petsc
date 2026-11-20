/* Inject one rank's VecDuplicate failure before accepted-state HDF5 capture.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "solver_internal.h"

#include <stdio.h>

static int rank;
static unsigned int fail_on_call, duplicate_calls;

PetscErrorCode __real_VecDuplicate(Vec source, Vec *destination);
PetscErrorCode __wrap_VecDuplicate(Vec source, Vec *destination);

PetscErrorCode __wrap_VecDuplicate(Vec source, Vec *destination) {
  ++duplicate_calls;
  const PetscErrorCode status = __real_VecDuplicate(source, destination);
  if (status == PETSC_SUCCESS && rank == 1 && duplicate_calls == fail_on_call) {
    /* Match PETSc's communicator bookkeeping on both ranks, then simulate
     * the local allocation failure by withholding the new vector. */
    (void)VecDestroy(destination);
    return PETSC_ERR_MEM;
  }
  return status;
}

static SsStatus ss_unused_fixed(void *context, PetscReal factor, Vec incoming,
                                Vec solution, SsNonlinearSolveRecord *record) {
  (void)context;
  (void)factor;
  (void)incoming;
  (void)solution;
  (void)record;
  return SS_ERROR_INVALID_STATE;
}

static SsStatus ss_unused_work(void *context, SsAnalysisMethod method, PetscReal work,
                               PetscReal factor, Vec incoming, Vec solution,
                               PetscReal *result_factor,
                               SsNonlinearSolveRecord *record) {
  (void)context;
  (void)method;
  (void)work;
  (void)factor;
  (void)incoming;
  (void)solution;
  (void)result_factor;
  (void)record;
  return SS_ERROR_INVALID_STATE;
}

static SsStatus ss_unused_potential(void *context, PetscReal factor, Vec state,
                                    PetscReal *potential) {
  (void)context;
  (void)factor;
  (void)state;
  (void)potential;
  return SS_ERROR_INVALID_STATE;
}

static SsStatus ss_unused_maximum(void *context, Vec state, PetscReal *maximum) {
  (void)context;
  (void)state;
  (void)maximum;
  return SS_ERROR_INVALID_STATE;
}

int main(int argc, char **argv) {
  if (PetscInitialize(&argc, &argv, NULL, NULL) != PETSC_SUCCESS) {
    return 2;
  }
  int size = 0, failed = 0, total_failed = 0;
  Vec solution = NULL;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  if (size != 2 || VecCreateMPI(PETSC_COMM_WORLD, 2, 4, &solution) != PETSC_SUCCESS) {
    (void)PetscFinalize();
    return 2;
  }
  SsSolverGraph graph = {.communicator = PETSC_COMM_WORLD};
  graph.policy.output_frequency = SS_SOLVER_OUTPUT_ACCEPTED_STATE;
  SsSolverMechanics mechanics = {0};
  SsSolverMonitor monitor = {.communicator = PETSC_COMM_WORLD, .graph = &graph};
  SsAnalysisProblem problem = {
      .communicator = PETSC_COMM_WORLD,
      .operations =
          {
              .solve_fixed = ss_unused_fixed,
              .solve_work_controlled = ss_unused_work,
              .total_potential = ss_unused_potential,
              .displacement_maximum = ss_unused_maximum,
          },
  };
  const SsStatus attached =
      ss_solver_monitor_attach(&monitor, &mechanics, NULL, NULL, NULL, &problem);
  failed |= attached != SS_OK;
  if (attached == SS_OK) {
    /* An allocation on only one rank must stop both ranks before either
     * duplicates the next vector or opens the collective accepted-state file. */
    for (unsigned int call = 1U; call <= 2U; ++call) {
      fail_on_call = call;
      duplicate_calls = 0U;
      const SsStatus status =
          problem.operations.capture_accepted_state(problem.context, 0U, solution);
      (void)printf("rank=%d failed_duplicate=%u status=%d calls=%u spool=%d\n", rank,
                   call, (int)status, duplicate_calls, monitor.states.open);
      failed |= status != SS_ERROR_OUT_OF_MEMORY || duplicate_calls != call ||
                monitor.states.open != 0 || mechanics.checkpoint_state_count != 0U ||
                mechanics.checkpoint_current_state != NULL ||
                mechanics.checkpoint_previous_state != NULL;
    }
  }
  (void)VecDestroy(&solution);
  MPI_Allreduce(&failed, &total_failed, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD);
  if (PetscFinalize() != PETSC_SUCCESS) {
    return 2;
  }
  return total_failed;
}
