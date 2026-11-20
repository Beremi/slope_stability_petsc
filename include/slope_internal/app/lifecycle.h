/** @file
 * @brief Engine lifecycle states and cancellation at safe points.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_APP_LIFECYCLE_H
#define SS_APP_LIFECYCLE_H

#include "slope_internal/core/status.h"

#include <mpi.h>

#include <signal.h>
#include <stdint.h>

/** State of one engine operation. */
typedef enum {
  SS_ENGINE_CREATED = 1,   /**< Initialized, inputs not yet validated. */
  SS_ENGINE_VALIDATED = 2, /**< Inputs and setup validated. */
  SS_ENGINE_RUNNING = 3,   /**< Analysis in progress. */
  SS_ENGINE_SUCCEEDED = 4, /**< Finished and published (terminal). */
  SS_ENGINE_FAILED = 5,    /**< Failed (terminal). */
  SS_ENGINE_CANCELLED = 6, /**< Cancelled at a safe point (terminal). */
  SS_ENGINE_DESTROYED = 7  /**< Released. */
} SsEngineState;

/** Engine state machine. */
typedef struct {
  SsEngineState state;    /**< Current state. */
  uint32_t destroy_count; /**< Number of completed destroy transitions. */
} SsLifecycle;

/** Kind of safe point at which cancellation is checked. */
typedef enum {
  SS_SAFEPOINT_SETUP = 1, /**< Between setup phases. */
  SS_SAFEPOINT_HDF5 = 2,  /**< Between HDF5 operations. */
  SS_SAFEPOINT_OUTPUT = 3 /**< Before result output. */
} SsSafePoint;

/** Cancellation flag, written by a signal handler. */
typedef struct {
  volatile sig_atomic_t requested; /**< Nonzero once cancellation was requested. */
} SsCancellation;

/** Initialize the state machine in ::SS_ENGINE_CREATED.
 * @param[out] lifecycle State machine; NULL is ignored. */
void ss_lifecycle_init(SsLifecycle *lifecycle);

/** Move to a new state if the transition is allowed.
 *
 * Allowed: created to validated; validated to running or succeeded; created,
 * validated or running to failed or cancelled; running to succeeded; any
 * terminal state to destroyed.
 *
 * @param[in,out] lifecycle State machine.
 * @param[in] next_state Requested state.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_INVALID_STATE. */
SsStatus ss_lifecycle_transition(SsLifecycle *lifecycle, SsEngineState next_state);

/** Move a terminal state machine to ::SS_ENGINE_DESTROYED; idempotent.
 * @param[in,out] lifecycle State machine.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT, or ::SS_ERROR_INVALID_STATE
 *   when the operation has not reached a terminal state. */
SsStatus ss_lifecycle_destroy(SsLifecycle *lifecycle);

/** Clear a cancellation flag.
 * @param[out] cancellation Flag; NULL is ignored. */
void ss_cancellation_init(SsCancellation *cancellation);

/** Request cancellation; async-signal safe.
 * @param[in,out] cancellation Flag; NULL is ignored. */
void ss_cancellation_request(SsCancellation *cancellation);

/** Check the local cancellation flag at a safe point.
 * @param[in] cancellation Flag.
 * @param[in] safe_point Kind of safe point.
 * @return ::SS_OK, ::SS_ERROR_CANCELLED or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_cancellation_safe_point(const SsCancellation *cancellation,
                                    SsSafePoint safe_point);

/** Agree on cancellation at a safe point.
 *
 * Collective on @p communicator: if any rank was cancelled, all ranks return
 * ::SS_ERROR_CANCELLED. An invalid argument returns before the reduction, so
 * every rank must pass valid arguments.
 *
 * @param[in] communicator Communicator of the ranks.
 * @param[in] cancellation Local flag.
 * @param[in] safe_point Kind of safe point.
 * @return ::SS_OK, ::SS_ERROR_CANCELLED, ::SS_ERROR_INVALID_ARGUMENT or
 *   ::SS_ERROR_INTERNAL. */
SsStatus ss_cancellation_collective_safe_point(MPI_Comm communicator,
                                               const SsCancellation *cancellation,
                                               SsSafePoint safe_point);

/** Install SIGINT and SIGTERM handlers that set @p cancellation.
 *
 * The handler only sets the flag. The flag must outlive the process's use of
 * the handlers; installing again replaces the target flag.
 *
 * @param[in,out] cancellation Flag set by the handlers.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_INTERNAL when
 *   `sigaction` fails. */
SsStatus ss_cancellation_install_handlers(SsCancellation *cancellation);

/**
 * Whether SIGINT or SIGTERM reached this process through the installed
 * handlers.
 *
 * Process-local and async-signal safe; callers agree collectively at their
 * safe points.
 *
 * @return Nonzero after a signal; zero when none arrived or no handlers are
 *   installed.
 */
int ss_cancellation_signalled(void);

#endif
