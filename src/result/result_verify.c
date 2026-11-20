/* Read and verify a committed terminal result envelope.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "result_envelope.h"
#include "result_internal.h"

#include "yyjson.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <errno.h>
#include <stdio.h>
#include <string.h>

SsStatus ss_result_verify_terminal(MPI_Comm communicator, const char *directory,
                                   SsResultWriterSummary *summary) {
  char json_path[SS_RESULT_PATH_CAPACITY];
  char hdf5_path[SS_RESULT_PATH_CAPACITY];
  int rank = -1;
  int status_value = (int)SS_OK;
  if (directory == NULL || summary == NULL ||
      MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(summary, 0, sizeof(*summary));
  if (rank == 0) {
    yyjson_doc *document = NULL;
    yyjson_val *root;
    yyjson_val *identity;
    yyjson_val *run;
    yyjson_val *terminal;
    yyjson_val *result_summary;
    yyjson_val *quantities;
    yyjson_val *artifacts;
    yyjson_val *payload;
    yyjson_val *environment_payload;
    const char *status_text;
    const char *kind_text;
    const char *expected_digest;
    const char *stop_reason;
    uint64_t expected_bytes;
    uint64_t expected_environment_bytes;
    char digest[SS_SHA256_HEX_CAPACITY];
    char environment_digest[SS_SHA256_HEX_CAPACITY];
    char expected_environment_digest[SS_SHA256_HEX_CAPACITY];
    char environment_artifact_path[SS_RESULT_TEXT_CAPACITY];
    char environment_file_path[SS_RESULT_PATH_CAPACITY];
    char identity_result_id[SS_SHA256_HEX_CAPACITY + 8U];
    char payload_logical_digest[SS_SHA256_HEX_CAPACITY + 8U];
    char run_id[SS_SHA256_HEX_CAPACITY + 8U];
    char recomputed_id[SS_SHA256_HEX_CAPACITY + 8U] = "sha256:";
    char result_identity_payload[1024];
    uint64_t bytes = 0U;
    if (snprintf(json_path, sizeof(json_path), "%s/result.json", directory) < 0 ||
        snprintf(hdf5_path, sizeof(hdf5_path), "%s/result.h5", directory) < 0) {
      status_value = (int)SS_ERROR_OVERFLOW;
    } else {
      document = yyjson_read_file(json_path, YYJSON_READ_NOFLAG, NULL, NULL);
      if (document == NULL) {
        status_value = (int)(errno == ENOENT ? SS_ERROR_NOT_FOUND : SS_ERROR_JSON);
      }
    }
    root = document == NULL ? NULL : yyjson_doc_get_root(document);
    kind_text = root == NULL ? NULL : yyjson_get_str(yyjson_obj_get(root, "kind"));
    identity = root == NULL ? NULL : yyjson_obj_get(root, "identity");
    run = root == NULL ? NULL : yyjson_obj_get(root, "run");
    terminal = root == NULL ? NULL : yyjson_obj_get(root, "terminal_status");
    result_summary = root == NULL ? NULL : yyjson_obj_get(root, "summary");
    quantities =
        result_summary == NULL ? NULL : yyjson_obj_get(result_summary, "quantities");
    artifacts = root == NULL ? NULL : yyjson_obj_get(root, "artifacts");
    payload = artifacts == NULL ? NULL : yyjson_obj_get(artifacts, "result.h5");
    environment_payload =
        artifacts == NULL ? NULL : yyjson_obj_get(artifacts, "environment.json");
    status_text =
        terminal == NULL ? NULL : yyjson_get_str(yyjson_obj_get(terminal, "status"));
    stop_reason = terminal == NULL
                      ? NULL
                      : yyjson_get_str(yyjson_obj_get(terminal, "stop_reason"));
    expected_digest =
        payload == NULL ? NULL : yyjson_get_str(yyjson_obj_get(payload, "sha256"));
    expected_bytes =
        payload == NULL ? 0U : yyjson_get_uint(yyjson_obj_get(payload, "bytes"));
    expected_environment_bytes =
        environment_payload == NULL
            ? 0U
            : yyjson_get_uint(yyjson_obj_get(environment_payload, "bytes"));
    if (status_value == (int)SS_OK &&
        (root == NULL || !yyjson_is_obj(root) || kind_text == NULL ||
         strcmp(kind_text, "slope_stability.result") != 0 ||
         !yyjson_is_obj(result_summary) || !yyjson_is_arr(quantities) ||
         !yyjson_get_bool(yyjson_obj_get(terminal, "complete")) ||
         status_text == NULL || stop_reason == NULL || expected_digest == NULL ||
         !ss_result_json_string(identity, "result_id", identity_result_id,
                                sizeof(identity_result_id)) ||
         !ss_result_json_string(payload, "logical_digest", payload_logical_digest,
                                sizeof(payload_logical_digest)) ||
         !ss_result_json_string(run, "run_id", run_id, sizeof(run_id)) ||
         !ss_result_json_string(environment_payload, "path", environment_artifact_path,
                                sizeof(environment_artifact_path)) ||
         strcmp(environment_artifact_path, "environment.json") != 0 ||
         !ss_result_json_string(environment_payload, "sha256",
                                expected_environment_digest,
                                sizeof(expected_environment_digest)) ||
         expected_environment_bytes == 0U ||
         !ss_result_json_string(root, "id", summary->result_id,
                                sizeof(summary->result_id)) ||
         !ss_result_json_string(identity, "logical_digest", summary->logical_digest,
                                sizeof(summary->logical_digest)))) {
      status_value = (int)SS_ERROR_SCHEMA;
    }
    if (status_value == (int)SS_OK &&
        (strcmp(summary->result_id, identity_result_id) != 0 ||
         strcmp(summary->logical_digest, payload_logical_digest) != 0 ||
         !ss_valid_content_id(summary->result_id) ||
         !ss_valid_content_id(summary->logical_digest) ||
         !ss_valid_content_id(run_id))) {
      status_value = (int)SS_ERROR_INTEGRITY;
    }
    if (status_value == (int)SS_OK) {
      if (strcmp(status_text, "succeeded") == 0) {
        summary->terminal_status = SS_RESULT_SUCCEEDED;
      } else if (strcmp(status_text, "failed_numerical") == 0) {
        summary->terminal_status = SS_RESULT_FAILED;
      } else if (strcmp(status_text, "cancelled") == 0) {
        summary->terminal_status = SS_RESULT_CANCELLED;
      } else {
        status_value = (int)SS_ERROR_SCHEMA;
      }
    }
    if (status_value == (int)SS_OK) {
      int identity_size = snprintf(
          result_identity_payload, sizeof(result_identity_payload), "%s|%s|%s|%s",
          run_id, status_text, stop_reason, summary->logical_digest);
      if (identity_size < 0 ||
          (size_t)identity_size >= sizeof(result_identity_payload) ||
          ss_sha256_bytes((const unsigned char *)result_identity_payload,
                          (size_t)identity_size, recomputed_id + 7U) != SS_OK ||
          strcmp(recomputed_id, summary->result_id) != 0) {
        status_value = (int)SS_ERROR_INTEGRITY;
      }
    }
    if (status_value == (int)SS_OK) {
      SsStatus digest_status = ss_result_payload_digest(hdf5_path, digest, &bytes);
      if (digest_status != SS_OK || bytes != expected_bytes ||
          strcmp(digest, expected_digest) != 0) {
        status_value = (int)SS_ERROR_INTEGRITY;
      } else {
        summary->payload_bytes = bytes;
        memcpy(summary->payload_sha256, digest, sizeof(digest));
        summary->field_count = (uint64_t)yyjson_arr_size(quantities);
        summary->accepted_state_count =
            yyjson_get_uint(yyjson_obj_get(result_summary, "accepted_state_count"));
        summary->history_row_count =
            yyjson_get_uint(yyjson_obj_get(result_summary, "history_row_count"));
        summary->writer_state = SS_RESULT_WRITER_PUBLISHED;
        (void)ss_result_copy_text(summary->final_directory,
                                  sizeof(summary->final_directory), directory);
      }
    }
    if (status_value == (int)SS_OK) {
      uint64_t environment_bytes = 0U;
      if (snprintf(environment_file_path, sizeof(environment_file_path),
                   "%s/environment.json", directory) < 0 ||
          ss_result_payload_digest(environment_file_path, environment_digest,
                                   &environment_bytes) != SS_OK ||
          environment_bytes != expected_environment_bytes ||
          strcmp(environment_digest, expected_environment_digest) != 0) {
        status_value = (int)SS_ERROR_INTEGRITY;
      }
    }
    yyjson_doc_free(document);
  }
  if (MPI_Bcast(&status_value, 1, MPI_INT, 0, communicator) != MPI_SUCCESS ||
      MPI_Bcast(summary, (int)sizeof(*summary), MPI_BYTE, 0, communicator) !=
          MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  return (SsStatus)status_value;
}
