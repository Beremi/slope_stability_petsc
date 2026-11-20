/** @file
 * @brief Validate and run entry points of the native solver.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_APP_SOLVER_H
#define SS_APP_SOLVER_H

#include "slope_internal/core/core.h"
#include "slope_internal/io/input.h"
#include "slope_internal/material/saturation.h"

#include <mpi.h>

#include <stdint.h>

/**
 * Operational options of one engine run.
 *
 * These control observation and recovery only; they are not part of the run
 * identity and never change numerical results.  Zero-initialized options give
 * the defaults: run directory `<result>.run`, journal `<run>/journal.jsonl`,
 * no human progress, and no periodic in-run checkpoints.
 */
typedef struct {
  /** Directory for the journal, spools and in-run checkpoints (NULL: default). */
  const char *run_directory;
  /** JSON-lines journal path (NULL: `<run>/journal.jsonl`; "" disables it). */
  const char *journal_path;
  /** Nonzero prints concise progress at accepted states on rank zero's stderr. */
  int progress;
  /** Write an in-run checkpoint every this many accepted states (0: off). */
  uint64_t checkpoint_every;
  /** Write an in-run checkpoint once this many seconds passed (0: off). */
  double checkpoint_seconds;
  /** In-run checkpoints retained (0 is treated as 1). */
  uint32_t checkpoint_keep;
} SsSolverRunOptions;

/** Inputs of one validate or run operation. The request borrows every pointer. */
typedef struct {
  MPI_Comm communicator;         /**< Communicator of all engine ranks. */
  const char *problem_directory; /**< Verified problem package directory. */
  const char *resolved_run_path; /**< Path of `resolved-run.json`. */
  const char *result_directory;  /**< Empty result directory to publish into. */
  const SsNativeConfig *config;  /**< Sealed configuration of problem and run. */
  const char *build_id;          /**< Build identity recorded in the result. */
  int requested_threads;         /**< OpenMP threads per rank from the layout. */
  /** Operational options; NULL selects the defaults. */
  const SsSolverRunOptions *options;
} SsSolverRequest;

/** Outcome of a validate or run operation, identical on all ranks. */
typedef struct {
  char result_id[SS_SHA256_HEX_CAPACITY + 8U];   /**< Published result identity. */
  char mesh_id[SS_SHA256_HEX_CAPACITY + 8U];     /**< Identity of the mesh. */
  char geometry_id[SS_SHA256_HEX_CAPACITY + 8U]; /**< Identity of the geometry. */
  uint64_t global_cells;                         /**< Number of cells. */
  uint64_t global_free_dofs; /**< Unknowns after essential constraints. */
  uint64_t accepted_states;  /**< Accepted continuation states published. */
  double terminal_factor;    /**< Load or strength-reduction factor at the end. */
  int result_published;      /**< Nonzero when `result.json` was published. */
} SsSolverSummary;

/**
 * Build the distributed solver graph without solving, then release it.
 *
 * Reads the mesh, creates field spaces, constraints and assembly plans, and
 * reports their size. Collective on `request->communicator`.
 *
 * @param[in] request Validation inputs.
 * @param[out] summary Mesh and field sizes; zeroed first.
 * @param[out] error Collectively selected error record; zeroed first.
 * @return ::SS_OK or the agreed failure status.
 */
SsStatus ss_solver_validate(const SsSolverRequest *request, SsSolverSummary *summary,
                            SsErrorRecord *error);

/**
 * Run the analysis of a resolved run and publish its result directory.
 *
 * Builds the solver graph, solves seepage and mechanics as configured, runs
 * the continuation analysis, and publishes the result or a terminal error
 * envelope. Collective on `request->communicator`; every rank returns the
 * same status. Cancellation is honored at the analysis safe points.
 *
 * @param[in] request Run inputs.
 * @param[out] summary Result identity and terminal values; zeroed first.
 * @param[out] error Collectively selected error record; zeroed first.
 * @return ::SS_OK, ::SS_ERROR_CANCELLED, or the agreed failure status.
 */
SsStatus ss_solver_run(const SsSolverRequest *request, SsSolverSummary *summary,
                       SsErrorRecord *error);

#endif
