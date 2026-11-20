/* Live, bounded progress journal of a native run (rank-zero JSON lines).
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/diagnostics/journal.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SS_PROGRESS_JOURNAL_PATH_CAPACITY 1024U
#define SS_PROGRESS_JOURNAL_LINE_CAPACITY 4096U

struct SsProgressJournal {
  int rank;
  int human_progress;
  FILE *file;
  char path[SS_PROGRESS_JOURNAL_PATH_CAPACITY];
  uint64_t maximum_bytes;
  uint64_t bytes;
  uint64_t sequence;
  double started;
};

static void ss_progress_journal_disable(SsProgressJournal *journal,
                                        const char *operation) {
  if (journal->file != NULL) {
    (void)fclose(journal->file);
    journal->file = NULL;
  }
  (void)fprintf(stderr,
                "{\"category\":\"diagnostics\",\"code\":\"JOURNAL_DISABLED\","
                "\"operation\":\"%s\",\"errno\":%d}\n",
                operation, errno);
}

static int ss_progress_journal_reopen(SsProgressJournal *journal, const char *mode) {
  journal->file = fopen(journal->path, mode);
  journal->bytes = 0U;
  if (journal->file != NULL && mode[0] == 'a') {
    const long position =
        fseek(journal->file, 0L, SEEK_END) == 0 ? ftell(journal->file) : -1L;
    journal->bytes = position > 0L ? (uint64_t)position : 0U;
  }
  return journal->file != NULL;
}

SsStatus ss_progress_journal_open(const SsProgressJournalSpec *spec,
                                  SsProgressJournal **result) {
  SsProgressJournal *journal;
  int rank = 0;
  if (spec == NULL || result == NULL ||
      MPI_Comm_rank(spec->communicator, &rank) != MPI_SUCCESS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *result = NULL;
  journal = (SsProgressJournal *)calloc(1U, sizeof(*journal));
  if (journal == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  journal->rank = rank;
  journal->human_progress = spec->human_progress != 0;
  journal->maximum_bytes = spec->maximum_bytes == 0U ? SS_PROGRESS_JOURNAL_DEFAULT_BYTES
                                                     : spec->maximum_bytes;
  journal->started = MPI_Wtime();
  if (rank == 0 && spec->path != NULL && spec->path[0] != '\0') {
    const int written =
        snprintf(journal->path, sizeof(journal->path), "%s", spec->path);
    if (written < 0 || (size_t)written >= sizeof(journal->path)) {
      free(journal);
      return SS_ERROR_LIMIT;
    }
    if (!ss_progress_journal_reopen(journal, "a")) {
      ss_progress_journal_disable(journal, "open");
    }
  }
  *result = journal;
  return SS_OK;
}

double ss_progress_journal_elapsed(const SsProgressJournal *journal) {
  return journal == NULL ? 0.0 : MPI_Wtime() - journal->started;
}

static void ss_progress_journal_rotate(SsProgressJournal *journal) {
  char rotated[SS_PROGRESS_JOURNAL_PATH_CAPACITY + 4U];
  const int written = snprintf(rotated, sizeof(rotated), "%s.1", journal->path);
  (void)fclose(journal->file);
  journal->file = NULL;
  if (written < 0 || (size_t)written >= sizeof(rotated) ||
      rename(journal->path, rotated) != 0 ||
      !ss_progress_journal_reopen(journal, "w")) {
    ss_progress_journal_disable(journal, "rotate");
  }
}

SsStatus ss_progress_journal_event(SsProgressJournal *journal, const char *event,
                                   const char *members) {
  char line[SS_PROGRESS_JOURNAL_LINE_CAPACITY];
  int written;
  if (journal == NULL || event == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (journal->rank != 0 || journal->file == NULL) {
    return SS_OK;
  }
  written = snprintf(
      line, sizeof(line),
      "{\"seq\":%llu,\"elapsed\":%.6f,\"time\":%lld,\"event\":\"%s\"%s%s}\n",
      (unsigned long long)journal->sequence, ss_progress_journal_elapsed(journal),
      (long long)time(NULL), event, members != NULL && members[0] != '\0' ? "," : "",
      members != NULL ? members : "");
  if (written < 0 || (size_t)written >= sizeof(line)) {
    return SS_ERROR_LIMIT;
  }
  if (journal->bytes + (uint64_t)written > journal->maximum_bytes) {
    ss_progress_journal_rotate(journal);
    if (journal->file == NULL) {
      return SS_OK;
    }
  }
  if (fwrite(line, 1U, (size_t)written, journal->file) != (size_t)written ||
      fflush(journal->file) != 0) {
    ss_progress_journal_disable(journal, "write");
    return SS_OK;
  }
  journal->bytes += (uint64_t)written;
  ++journal->sequence;
  return SS_OK;
}

void ss_progress_journal_say(SsProgressJournal *journal, const char *format, ...) {
  va_list arguments;
  if (journal == NULL || journal->rank != 0 || journal->human_progress == 0 ||
      format == NULL) {
    return;
  }
  (void)fprintf(stderr, "slope-engine [%9.1fs] ", ss_progress_journal_elapsed(journal));
  va_start(arguments, format);
  (void)vfprintf(stderr, format, arguments);
  va_end(arguments);
  (void)fputc('\n', stderr);
  (void)fflush(stderr);
}

SsStatus ss_progress_journal_close(SsProgressJournal **journal) {
  SsStatus status = SS_OK;
  if (journal == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (*journal != NULL && (*journal)->file != NULL && fclose((*journal)->file) != 0) {
    status = SS_ERROR_IO;
  }
  free(*journal);
  *journal = NULL;
  return status;
}
