/* Engine lifecycle states and collective cancellation after SIGINT or SIGTERM.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/app/lifecycle.h"

#include <string.h>

static SsCancellation *ss_signal_cancellation = NULL;

static void ss_signal_handler(int signal_number) {
  (void)signal_number;
  if (ss_signal_cancellation != NULL) {
    ss_signal_cancellation->requested = 1;
  }
}

void ss_lifecycle_init(SsLifecycle *lifecycle) {
  if (lifecycle != NULL) {
    lifecycle->state = SS_ENGINE_CREATED;
    lifecycle->destroy_count = 0U;
  }
}

static int ss_transition_allowed(SsEngineState current, SsEngineState next) {
  switch (current) {
  case SS_ENGINE_CREATED:
    return next == SS_ENGINE_VALIDATED || next == SS_ENGINE_FAILED ||
           next == SS_ENGINE_CANCELLED;
  case SS_ENGINE_VALIDATED:
    return next == SS_ENGINE_RUNNING || next == SS_ENGINE_SUCCEEDED ||
           next == SS_ENGINE_FAILED || next == SS_ENGINE_CANCELLED;
  case SS_ENGINE_RUNNING:
    return next == SS_ENGINE_SUCCEEDED || next == SS_ENGINE_FAILED ||
           next == SS_ENGINE_CANCELLED;
  case SS_ENGINE_SUCCEEDED:
  case SS_ENGINE_FAILED:
  case SS_ENGINE_CANCELLED:
    return next == SS_ENGINE_DESTROYED;
  case SS_ENGINE_DESTROYED:
    return next == SS_ENGINE_DESTROYED;
  }
  return 0;
}

SsStatus ss_lifecycle_transition(SsLifecycle *lifecycle, SsEngineState next_state) {
  if (lifecycle == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (!ss_transition_allowed(lifecycle->state, next_state)) {
    return SS_ERROR_INVALID_STATE;
  }
  lifecycle->state = next_state;
  return SS_OK;
}

SsStatus ss_lifecycle_destroy(SsLifecycle *lifecycle) {
  if (lifecycle == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (lifecycle->state == SS_ENGINE_DESTROYED) {
    return SS_OK;
  }
  if (ss_lifecycle_transition(lifecycle, SS_ENGINE_DESTROYED) != SS_OK) {
    return SS_ERROR_INVALID_STATE;
  }
  ++lifecycle->destroy_count;
  return SS_OK;
}

void ss_cancellation_init(SsCancellation *cancellation) {
  if (cancellation != NULL) {
    cancellation->requested = 0;
  }
}

void ss_cancellation_request(SsCancellation *cancellation) {
  if (cancellation != NULL) {
    cancellation->requested = 1;
  }
}

SsStatus ss_cancellation_safe_point(const SsCancellation *cancellation,
                                    SsSafePoint safe_point) {
  if (cancellation == NULL || safe_point < SS_SAFEPOINT_SETUP ||
      safe_point > SS_SAFEPOINT_OUTPUT) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return cancellation->requested != 0 ? SS_ERROR_CANCELLED : SS_OK;
}

SsStatus ss_cancellation_collective_safe_point(MPI_Comm communicator,
                                               const SsCancellation *cancellation,
                                               SsSafePoint safe_point) {
  int local_cancelled;
  int any_cancelled = 0;
  SsStatus local_status = ss_cancellation_safe_point(cancellation, safe_point);
  if (local_status != SS_OK && local_status != SS_ERROR_CANCELLED) {
    return local_status;
  }
  local_cancelled = local_status == SS_ERROR_CANCELLED ? 1 : 0;
  if (MPI_Allreduce(&local_cancelled, &any_cancelled, 1, MPI_INT, MPI_MAX,
                    communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  return any_cancelled != 0 ? SS_ERROR_CANCELLED : SS_OK;
}

SsStatus ss_cancellation_install_handlers(SsCancellation *cancellation) {
  struct sigaction action;
  if (cancellation == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(&action, 0, sizeof(action));
  action.sa_handler = ss_signal_handler;
  if (sigemptyset(&action.sa_mask) != 0) {
    return SS_ERROR_INTERNAL;
  }
  ss_signal_cancellation = cancellation;
  if (sigaction(SIGINT, &action, NULL) != 0 || sigaction(SIGTERM, &action, NULL) != 0) {
    return SS_ERROR_INTERNAL;
  }
  return SS_OK;
}

int ss_cancellation_signalled(void) {
  return ss_signal_cancellation != NULL && ss_signal_cancellation->requested != 0;
}
