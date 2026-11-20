/** @file
 * @brief Live, bounded progress journal of a native run.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_DIAGNOSTICS_JOURNAL_H
#define SS_DIAGNOSTICS_JOURNAL_H

#include "slope_internal/core/status.h"

#include <mpi.h>

#include <stddef.h>
#include <stdint.h>

/** Default size at which the journal file is rotated to `<path>.1`. */
#define SS_PROGRESS_JOURNAL_DEFAULT_BYTES (UINT64_C(8) << 20U)

/** Opaque journal; created by ss_progress_journal_open(). */
typedef struct SsProgressJournal SsProgressJournal;

/** Journal destination and behaviour. */
typedef struct {
  /** JSON-lines file written by rank zero; NULL or "" disables the file. */
  const char *path;
  /** Rotation threshold in bytes; zero selects the default. */
  uint64_t maximum_bytes;
  /** Nonzero prints concise human-readable progress on rank zero's stderr. */
  int human_progress;
  /** Communicator of the run; rank zero writes. */
  MPI_Comm communicator;
} SsProgressJournalSpec;

/**
 * Open a journal on every rank of the run; no communication takes place.
 *
 * Only rank zero owns a file.  Every line is one JSON object
 * `{"seq":n,"elapsed":s,"time":unix_seconds,"event":"name",...}` written and
 * flushed immediately, so the last line survives a forced stop.  The file is
 * rotated to `<path>.1` once it exceeds the threshold, bounding disk use to
 * about twice the threshold.  A journal I/O failure is reported once on
 * stderr and disables the file without failing the run.
 *
 * @param[in] spec Destination and behaviour; the path is copied.
 * @param[out] journal New journal, released with ss_progress_journal_close().
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT, ::SS_ERROR_OUT_OF_MEMORY, or
 *   ::SS_ERROR_LIMIT on rank zero for an overlong path. The status is local;
 *   the caller agrees on it.
 */
SsStatus ss_progress_journal_open(const SsProgressJournalSpec *spec,
                                  SsProgressJournal **journal);

/**
 * Append one event (rank zero only; other ranks return immediately).
 *
 * `members` is NULL or a comma-separated list of already-encoded JSON members
 * without braces, for example `"\"state\":3,\"factor\":1.25"`.
 *
 * @param[in,out] journal Open journal.
 * @param[in] event Event name, for example `accepted`.
 * @param[in] members Extra JSON members, or NULL.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT, or ::SS_ERROR_LIMIT for a
 *   line longer than the line buffer. Write failures disable the file and are
 *   not returned.
 */
SsStatus ss_progress_journal_event(SsProgressJournal *journal, const char *event,
                                   const char *members);

/** Print one human-readable progress line on rank zero when enabled.
 * @param[in] journal Open journal; NULL is accepted and ignored.
 * @param[in] format printf() format of the line, followed by its arguments. */
void ss_progress_journal_say(SsProgressJournal *journal, const char *format, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

/** Time since the journal was opened.
 * @param[in] journal Open journal.
 * @return Elapsed wall-clock seconds, or 0 for NULL. */
double ss_progress_journal_elapsed(const SsProgressJournal *journal);

/** Close the journal and release it; local, not collective.
 * @param[in,out] journal Journal pointer, set to NULL; `*journal` may be NULL.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT, or ::SS_ERROR_IO when closing
 *   the file fails. */
SsStatus ss_progress_journal_close(SsProgressJournal **journal);

#endif
