/* Result writer state machine, lifecycle, and terminal status.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "writer_internal.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

SsStatus ss_result_writer_fail(SsResultWriter *writer, SsResultFailurePoint point) {
  if (writer == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (writer->failure_point == point) {
    writer->state = SS_RESULT_WRITER_PUBLICATION_FAILED;
    writer->summary.writer_state = writer->state;
    return SS_ERROR_INJECTED;
  }
  return SS_OK;
}

SsStatus ss_result_writer_create(const SsResultWriterSpec *spec,
                                 SsResultWriter **result) {
  SsResultWriter *writer = NULL;
  int rank = -1;
  int size = 0;
  SsStatus status;
  if (result == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *result = NULL;
  if (spec == NULL || MPI_Comm_rank(spec->communicator, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(spec->communicator, &size) != MPI_SUCCESS || size <= 0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_result_writer_spec_validate(spec);
  status = ss_status_allreduce_max(spec->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  status = ss_result_writer_spec_agree(spec, size);
  if (status != SS_OK) {
    return status;
  }
  writer = (SsResultWriter *)calloc(1U, sizeof(*writer));
  if (writer == NULL) {
    status = SS_ERROR_OUT_OF_MEMORY;
  }
  status = ss_status_allreduce_max(spec->communicator, status);
  if (status != SS_OK) {
    free(writer);
    return status;
  }
  if (writer == NULL) {
    return SS_ERROR_INTERNAL;
  }
  writer->communicator = MPI_COMM_NULL;
  if (MPI_Comm_dup(spec->communicator, &writer->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(spec->communicator, status);
  if (status != SS_OK) {
    if (writer->communicator != MPI_COMM_NULL) {
      (void)MPI_Comm_free(&writer->communicator);
    }
    free(writer);
    return status;
  }
  writer->rank = rank;
  writer->size = size;
  writer->state = SS_RESULT_WRITER_UNOPENED;
  writer->access = -1;
  writer->file = -1;
  writer->created_at = MPI_Wtime();
  status = ss_result_writer_copy_spec(writer, spec);
  status = ss_status_allreduce_max(spec->communicator, status);
  if (status != SS_OK) {
    (void)ss_result_writer_destroy(&writer);
    return status;
  }
  writer->summary.writer_state = writer->state;
  (void)ss_result_copy_text(writer->summary.final_directory,
                            sizeof(writer->summary.final_directory),
                            writer->final_directory);
  *result = writer;
  return SS_OK;
}

SsStatus ss_result_writer_inject_failure(SsResultWriter *writer,
                                         SsResultFailurePoint point) {
  int local;
  int minimum;
  int maximum;
  if (writer == NULL || writer->state != SS_RESULT_WRITER_UNOPENED ||
      point < SS_RESULT_FAIL_NONE || point > SS_RESULT_FAIL_DIRECTORY_RENAME) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  local = (int)point;
  if (MPI_Allreduce(&local, &minimum, 1, MPI_INT, MPI_MIN, writer->communicator) !=
          MPI_SUCCESS ||
      MPI_Allreduce(&local, &maximum, 1, MPI_INT, MPI_MAX, writer->communicator) !=
          MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (minimum != maximum) {
    return SS_ERROR_INTEGRITY;
  }
  writer->failure_point = point;
  return SS_OK;
}

static int ss_result_outcome_known(const char *outcome) {
  static const char *const families[] = {
      "converged_criterion",
      "completed_control_horizon",
      "completed_requested_step_horizon",
      "accepted_relaxed",
      "failed_line_search",
      "failed_regularization",
      "failed_linear",
      "failed_singular_constraint",
      "failed_nonfinite",
      "failed_iteration_limit",
      "failed_retry_limit",
      "cancelled",
  };
  size_t index;
  for (index = 0U; index < sizeof(families) / sizeof(families[0]); ++index) {
    if (strcmp(outcome, families[index]) == 0) {
      return 1;
    }
  }
  return 0;
}

SsStatus ss_result_writer_set_outcome(SsResultWriter *writer,
                                      const SsResultOutcome *outcome) {
  char payload[3U * SS_RESULT_TEXT_CAPACITY + 3U];
  int count;
  SsStatus status = SS_OK;
  if (writer == NULL || outcome == NULL ||
      (writer->state != SS_RESULT_WRITER_STAGED &&
       writer->state != SS_RESULT_WRITER_STREAMING &&
       writer->state != SS_RESULT_WRITER_CLOSING)) {
    return SS_ERROR_INVALID_STATE;
  }
  const char *mode = outcome->mechanics_assembly_mode;
  if (outcome->outcome == NULL || !ss_result_outcome_known(outcome->outcome) ||
      !ss_result_text_is_valid(outcome->validation_policy, SS_RESULT_TEXT_CAPACITY) ||
      (mode != NULL && strcmp(mode, "qualification") != 0 &&
       strcmp(mode, "performance") != 0)) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  count = status == SS_OK
              ? snprintf(payload, sizeof(payload), "%s|%s|%s", outcome->outcome,
                         outcome->validation_policy, mode != NULL ? mode : "")
              : 0;
  if (count < 0 || (size_t)count >= sizeof(payload)) {
    status = SS_ERROR_OVERFLOW;
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK) {
    status = ss_result_collective_payload_agree(
        writer->communicator, (const unsigned char *)payload, (size_t)count);
  }
  if (status == SS_OK) {
    (void)ss_result_copy_text(writer->analysis_outcome,
                              sizeof(writer->analysis_outcome), outcome->outcome);
    (void)ss_result_copy_text(writer->validation_policy,
                              sizeof(writer->validation_policy),
                              outcome->validation_policy);
    (void)snprintf(writer->mechanics_assembly_mode,
                   sizeof(writer->mechanics_assembly_mode), "%s",
                   mode != NULL ? mode : "");
  }
  return status;
}

SsStatus ss_result_writer_terminalize(SsResultWriter *writer,
                                      SsResultTerminalStatus status,
                                      const char *stop_reason,
                                      const SsErrorRecord *error) {
  SsStatus validation = SS_OK;
  if (writer == NULL || writer->state != SS_RESULT_WRITER_STREAMING) {
    return SS_ERROR_INVALID_STATE;
  }
  if (status < SS_RESULT_SUCCEEDED || status > SS_RESULT_CANCELLED ||
      !ss_result_name_valid(stop_reason) ||
      (status == SS_RESULT_FAILED &&
       (error == NULL || error->valid == 0 ||
        !ss_result_text_is_valid(error->code, SS_ERROR_CODE_CAPACITY) ||
        !ss_result_text_is_valid(error->message, SS_ERROR_MESSAGE_CAPACITY))) ||
      (status != SS_RESULT_FAILED && error != NULL && error->valid != 0)) {
    validation = SS_ERROR_INVALID_STATE;
  }
  if (status == SS_RESULT_SUCCEEDED &&
      (writer->state_count == 0U || writer->field_count == 0U)) {
    validation = SS_ERROR_INVALID_STATE;
  }
  validation = ss_status_allreduce_max(writer->communicator, validation);
  if (validation != SS_OK) {
    return validation;
  }
  validation = ss_result_writer_terminal_agree(writer, status, stop_reason, error);
  if (validation != SS_OK) {
    return validation;
  }
  writer->terminal_status = status;
  (void)ss_result_copy_text(writer->stop_reason, sizeof(writer->stop_reason),
                            stop_reason);
  if (error != NULL) {
    writer->terminal_error = *error;
  }
  writer->state = SS_RESULT_WRITER_CLOSING;
  writer->summary.writer_state = writer->state;
  writer->summary.terminal_status = status;
  return SS_OK;
}

SsStatus ss_result_writer_summary(const SsResultWriter *writer,
                                  SsResultWriterSummary *summary) {
  if (writer == NULL || summary == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *summary = writer->summary;
  return SS_OK;
}

SsStatus ss_result_writer_destroy(SsResultWriter **writer) {
  SsStatus status = SS_OK;
  int status_value;
  if (writer == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (*writer == NULL) {
    return SS_OK;
  }
  status = ss_result_hdf5_close(&(*writer)->file, H5Fclose, status);
  status = ss_result_hdf5_close(&(*writer)->access, H5Pclose, status);
  status = ss_status_allreduce_max((*writer)->communicator, status);
  if ((*writer)->rank == 0 && (*writer)->staging_directory[0] != '\0') {
    SsStatus cleanup = ss_result_remove_staging((*writer)->staging_directory);
    if (cleanup != SS_OK && status == SS_OK) {
      status = cleanup;
    }
  }
  status_value = (int)status;
  if (MPI_Bcast(&status_value, 1, MPI_INT, 0, (*writer)->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  } else {
    status = (SsStatus)status_value;
  }
  if (ss_result_writer_layouts_destroy(*writer) != SS_OK && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  for (size_t index = 0U; index < (*writer)->field_count; ++index) {
    free((*writer)->fields[index].state_ids);
  }
  free((*writer)->checkpoints);
  free((*writer)->scalars);
  free((*writer)->states);
  free((*writer)->history);
  free((*writer)->fields);
  if (MPI_Comm_free(&(*writer)->communicator) != MPI_SUCCESS && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  free(*writer);
  *writer = NULL;
  return status;
}

const char *ss_result_terminal_status_name(SsResultTerminalStatus status) {
  switch (status) {
  case SS_RESULT_SUCCEEDED:
    return "succeeded";
  case SS_RESULT_FAILED:
    return "failed_numerical";
  case SS_RESULT_CANCELLED:
    return "cancelled";
  default:
    return "invalid";
  }
}
