/* In-run accepted-boundary checkpoints with bounded retention.
 *
 * At a due accepted boundary the run monitor writes a restartable checkpoint
 * of that boundary into `<run directory>/checkpoints/state-<ordinal>` using the
 * same atomic writer as the published accepted checkpoint.  Only the newest
 * `checkpoint_keep` checkpoints are retained, so disk use does not grow with
 * the horizon; a forced stop therefore leaves the newest complete checkpoint.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/core/sort.h"
#include "solver_internal.h"

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* Checkpoint directories inspected per retention pass. */
#define SS_SOLVER_CHECKPOINT_SCAN 64U

static int ss_solver_checkpoint_descending(const void *left_raw,
                                           const void *right_raw) {
  const uint64_t left = *(const uint64_t *)left_raw;
  const uint64_t right = *(const uint64_t *)right_raw;
  return left > right ? -1 : left < right ? 1 : 0;
}

/* Remove a flat checkpoint directory (rank zero). */
static SsStatus ss_solver_checkpoint_remove(const char *directory) {
  DIR *handle = opendir(directory);
  struct dirent *entry;
  SsStatus status = SS_OK;
  if (handle == NULL) {
    return errno == ENOENT ? SS_OK : SS_ERROR_IO;
  }
  while ((entry = readdir(handle)) != NULL) {
    char path[SS_RESULT_PATH_CAPACITY];
    int written;
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }
    written = snprintf(path, sizeof(path), "%s/%s", directory, entry->d_name);
    if (written < 0 || (size_t)written >= sizeof(path) || unlink(path) != 0) {
      status = SS_ERROR_IO;
    }
  }
  if (closedir(handle) != 0 || rmdir(directory) != 0) {
    status = SS_ERROR_IO;
  }
  return status;
}

static int ss_solver_checkpoint_path(const SsSolverMonitor *monitor, uint64_t ordinal,
                                     char *path, size_t capacity) {
  const int written = snprintf(path, capacity, "%s/checkpoints/state-%06llu",
                               monitor->run_directory, (unsigned long long)ordinal);
  return written >= 0 && (size_t)written < capacity;
}

SsStatus ss_solver_checkpoint_boundary_write(SsSolverMonitor *monitor,
                                             const SsAnalysisRecord *record) {
  char parent[SS_RESULT_PATH_CAPACITY];
  char directory[SS_RESULT_PATH_CAPACITY];
  char members[SS_RESULT_PATH_CAPACITY + 256U];
  SsCheckpointRestoreSummary summary;
  SsErrorRecord error;
  SsSolverCheckpointTarget target;
  const SsAnalysisPointRecord *point;
  uint64_t ordinal;
  int written;
  SsStatus status = SS_OK;
  if (monitor == NULL || record == NULL || monitor->mechanics == NULL ||
      monitor->boundary_valid == 0 || monitor->boundary.point_count == 0U ||
      monitor->boundary.point_count > record->point_count) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  point = &record->points[monitor->boundary.point_count - 1U];
  ordinal = (uint64_t)point->accepted_ordinal;
  written = snprintf(parent, sizeof(parent), "%s/checkpoints", monitor->run_directory);
  if (written < 0 || (size_t)written >= sizeof(parent) ||
      !ss_solver_checkpoint_path(monitor, ordinal, directory, sizeof(directory))) {
    status = SS_ERROR_LIMIT;
  }
  if (status == SS_OK && monitor->rank == 0) {
    if (mkdir(parent, 0777) != 0 && errno != EEXIST) {
      status = SS_ERROR_IO;
    } else {
      /* A stale directory of the same ordinal (an earlier attempt) is replaced. */
      status = ss_solver_checkpoint_remove(directory);
    }
  }
  status = ss_status_allreduce_max(monitor->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  memset(&summary, 0, sizeof(summary));
  memset(&error, 0, sizeof(error));
  target = (SsSolverCheckpointTarget){
      .directory = directory,
      .history_cursor = monitor->history.written,
      .boundary = &monitor->boundary,
  };
  status = ss_solver_checkpoint_write_state(
      monitor->request, monitor->graph, &target,
      monitor->mechanics->checkpoint_current_state, record, monitor->mechanics,
      monitor->hydraulic_solution, monitor->hydraulic, NULL,
      monitor->hydraulic_evidence, (SsStateId)monitor->boundary.point_count, &summary,
      &error);
  if (status != SS_OK) {
    (void)snprintf(members, sizeof(members),
                   "\"state\":%llu,\"status\":%d,\"code\":\"%s\",\"message\":\"%s\"",
                   (unsigned long long)monitor->boundary.point_count, (int)status,
                   error.valid != 0 ? error.code : "",
                   error.valid != 0 ? error.message : "");
    (void)ss_progress_journal_event(monitor->journal, "checkpoint_failed", members);
    ss_progress_journal_say(
        monitor->journal, "checkpoint of accepted state %zu failed: %s",
        monitor->boundary.point_count, error.valid != 0 ? error.message : "unknown");
    return status;
  }
  ++monitor->checkpoint_count;
  monitor->checkpointed_points = monitor->boundary.point_count;
  monitor->last_checkpoint_time = MPI_Wtime();
  (void)snprintf(members, sizeof(members),
                 "\"state\":%llu,\"accepted_ordinal\":%llu,\"checkpoint_id\":\"%s\","
                 "\"path\":\"%s\"",
                 (unsigned long long)monitor->boundary.point_count,
                 (unsigned long long)ordinal, summary.checkpoint_id, directory);
  (void)ss_progress_journal_event(monitor->journal, "checkpoint", members);
  ss_progress_journal_say(monitor->journal, "checkpoint of accepted state %zu: %s",
                          monitor->boundary.point_count, directory);
  /* Retain the newest `checkpoint_keep` checkpoints. */
  if (monitor->rank == 0) {
    uint64_t ordinals[SS_SOLVER_CHECKPOINT_SCAN];
    size_t count = 0U;
    size_t index;
    DIR *handle = opendir(parent);
    struct dirent *entry;
    while (handle != NULL && (entry = readdir(handle)) != NULL &&
           count < SS_SOLVER_CHECKPOINT_SCAN) {
      unsigned long long other = 0ULL;
      if (sscanf(entry->d_name, "state-%llu", &other) == 1) {
        ordinals[count++] = (uint64_t)other;
      }
    }
    if (handle != NULL) {
      (void)closedir(handle);
    }
    ss_sort(ordinals, count, sizeof(ordinals[0]), ss_solver_checkpoint_descending);
    for (index = monitor->options.checkpoint_keep; index < count; ++index) {
      char stale[SS_RESULT_PATH_CAPACITY];
      if (ss_solver_checkpoint_path(monitor, ordinals[index], stale, sizeof(stale))) {
        (void)ss_solver_checkpoint_remove(stale);
      }
    }
  }
  return ss_status_allreduce_max(monitor->communicator, SS_OK);
}
