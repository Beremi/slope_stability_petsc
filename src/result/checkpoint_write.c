/* Write and publish the accepted-state checkpoint transaction.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "checkpoint_internal.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static SsStatus ss_checkpoint_failure(SsResultFailurePoint configured,
                                      SsResultFailurePoint point) {
  return configured == point ? SS_ERROR_INJECTED : SS_OK;
}

static SsStatus ss_checkpoint_transaction_begin(const SsCheckpointSpec *spec,
                                                SsCheckpointTransaction *transaction,
                                                SsResultFailurePoint failure_point) {
  SsStatus status;
  memset(transaction, 0, sizeof(*transaction));
  transaction->communicator = spec->communicator;
  transaction->access = -1;
  transaction->file = -1;
  if (MPI_Comm_rank(spec->communicator, &transaction->rank) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  status = ss_checkpoint_failure(failure_point, SS_RESULT_FAIL_DIRECTORY_CREATE);
  if (status == SS_OK) {
    status = ss_result_copy_text(transaction->final_directory,
                                 sizeof(transaction->final_directory),
                                 spec->final_directory);
  }
  if (status == SS_OK) {
    status = ss_result_make_staging(spec->communicator, spec->final_directory,
                                    "checkpoint", transaction->staging_directory,
                                    sizeof(transaction->staging_directory));
  }
  if (status == SS_OK) {
    status = ss_checkpoint_path(transaction->hdf5_temporary_path,
                                sizeof(transaction->hdf5_temporary_path),
                                transaction->staging_directory, "checkpoint.h5.tmp");
  }
  if (status == SS_OK) {
    status = ss_checkpoint_path(transaction->hdf5_path, sizeof(transaction->hdf5_path),
                                transaction->staging_directory, "checkpoint.h5");
  }
  if (status == SS_OK) {
    status = ss_checkpoint_path(transaction->envelope_temporary_path,
                                sizeof(transaction->envelope_temporary_path),
                                transaction->staging_directory, "checkpoint.json.tmp");
  }
  if (status == SS_OK) {
    status = ss_checkpoint_path(transaction->envelope_path,
                                sizeof(transaction->envelope_path),
                                transaction->staging_directory, "checkpoint.json");
  }
  return ss_status_allreduce_max(spec->communicator, status);
}

static SsStatus ss_checkpoint_transaction_cleanup(SsCheckpointTransaction *transaction,
                                                  SsStatus status) {
  int value;
  status = ss_result_hdf5_close(&transaction->file, H5Fclose, status);
  status = ss_result_hdf5_close(&transaction->access, H5Pclose, status);
  status = ss_status_allreduce_max(transaction->communicator, status);
  if (transaction->rank == 0 && transaction->staging_directory[0] != '\0') {
    const SsStatus cleanup = ss_result_remove_staging(transaction->staging_directory);
    if (status == SS_OK && cleanup != SS_OK) {
      status = cleanup;
    }
  }
  value = (int)status;
  if (MPI_Bcast(&value, 1, MPI_INT, 0, transaction->communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  return (SsStatus)value;
}

static SsStatus ss_checkpoint_root(const SsCheckpointSpec *spec,
                                   SsCheckpointTransaction *transaction) {
  static const char *const groups[] = {
      "/accepted",  "/accepted/fields", "/analysis",      "/coupling",
      "/deflation", "/history",         "/compatibility",
  };
  char field_ids[SS_CHECKPOINT_FIELD_IDS_CAPACITY];
  SsStatus status = ss_checkpoint_spec_field_ids(spec, field_ids, sizeof(field_ids));
  if (status != SS_OK) {
    return status;
  }
  transaction->access = H5Pcreate(H5P_FILE_ACCESS);
  if (transaction->access < 0 ||
      H5Pset_fapl_mpio(transaction->access, transaction->communicator, MPI_INFO_NULL) <
          0 ||
      H5Pset_all_coll_metadata_ops(transaction->access, 1U) < 0) {
    return SS_ERROR_HDF5;
  }
  transaction->file = H5Fcreate(transaction->hdf5_temporary_path, H5F_ACC_EXCL,
                                H5P_DEFAULT, transaction->access);
  if (transaction->file < 0) {
    return SS_ERROR_HDF5;
  }
  status =
      ss_result_hdf5_string_attribute(transaction->file, "layout_id", "checkpoint-v1");
  if (status == SS_OK) {
    status = ss_result_hdf5_u64_attribute(transaction->file, "schema_major", 1U);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_u64_attribute(transaction->file, "schema_minor", 2U);
  }
  if (status == SS_OK) {
    status =
        ss_result_hdf5_u64_attribute(transaction->file, "deflation_state_version", 1U);
  }
#define SS_CHECKPOINT_ROOT_TEXT(name, value)                                           \
  do {                                                                                 \
    if (status == SS_OK) {                                                             \
      status = ss_result_hdf5_string_attribute(transaction->file, name, value);        \
    }                                                                                  \
  } while (0)
  SS_CHECKPOINT_ROOT_TEXT("problem_id", spec->problem_id);
  SS_CHECKPOINT_ROOT_TEXT("run_id", spec->run_id);
  SS_CHECKPOINT_ROOT_TEXT("build_id", spec->build_id);
  SS_CHECKPOINT_ROOT_TEXT("build_compatibility_id", spec->build_compatibility_id);
  SS_CHECKPOINT_ROOT_TEXT("mesh_id", spec->mesh_id);
  SS_CHECKPOINT_ROOT_TEXT("geometry_id", spec->geometry_id);
  SS_CHECKPOINT_ROOT_TEXT("field_space_id", spec->field_space_id);
  SS_CHECKPOINT_ROOT_TEXT("numerical_policy_id", spec->numerical_policy_id);
  SS_CHECKPOINT_ROOT_TEXT("material_model_id", spec->material_model_id);
  SS_CHECKPOINT_ROOT_TEXT("hydraulic_model_id", spec->hydraulic_model_id);
  SS_CHECKPOINT_ROOT_TEXT("coupling_model_id", spec->coupling_model_id);
  SS_CHECKPOINT_ROOT_TEXT("analysis_method", spec->analysis_method);
  SS_CHECKPOINT_ROOT_TEXT("controller_id", spec->controller_id);
  SS_CHECKPOINT_ROOT_TEXT("deflation_provenance_id", spec->deflation.provenance_id);
  SS_CHECKPOINT_ROOT_TEXT("operator_identity", spec->deflation.operator_identity);
  SS_CHECKPOINT_ROOT_TEXT("field_ids", field_ids);
#undef SS_CHECKPOINT_ROOT_TEXT
#define SS_CHECKPOINT_ROOT_U64(name, value)                                            \
  do {                                                                                 \
    if (status == SS_OK) {                                                             \
      status = ss_result_hdf5_u64_attribute(transaction->file, name, value);           \
    }                                                                                  \
  } while (0)
  SS_CHECKPOINT_ROOT_U64("field_order", spec->field_order);
  SS_CHECKPOINT_ROOT_U64("geometry_order", spec->geometry_order);
  SS_CHECKPOINT_ROOT_U64("scalar_bits", spec->scalar_bits);
  SS_CHECKPOINT_ROOT_U64("source_index_bits", spec->source_index_bits);
  SS_CHECKPOINT_ROOT_U64("accepted_state_id", spec->accepted_state_id);
  SS_CHECKPOINT_ROOT_U64("field_count", spec->field_count);
  SS_CHECKPOINT_ROOT_U64("deflation_basis_count", spec->deflation.basis_count);
  SS_CHECKPOINT_ROOT_U64("dataset_chunk_rows", spec->dataset_chunk_rows);
#undef SS_CHECKPOINT_ROOT_U64
  if (status == SS_OK) {
    status = ss_result_hdf5_create_groups(transaction->file, transaction->communicator,
                                          groups, sizeof(groups) / sizeof(groups[0]));
  }
  return ss_status_allreduce_max(transaction->communicator, status);
}

SsStatus ss_checkpoint_write_atomic(const SsCheckpointSpec *spec,
                                    SsResultFailurePoint failure_point,
                                    SsCheckpointRestoreSummary *summary) {
  SsCheckpointTransaction transaction;
  SsCheckpointFieldDigest *field_digests = NULL;
  char left_digest[SS_SHA256_HEX_CAPACITY + 8U] = "sha256:";
  char right_digest[SS_SHA256_HEX_CAPACITY + 8U] = "sha256:";
  char source_digest[SS_SHA256_HEX_CAPACITY + 8U] = "sha256:";
  char source_weights_digest[SS_SHA256_HEX_CAPACITY + 8U] = "sha256:";
  char source_kinds_digest[SS_SHA256_HEX_CAPACITY + 8U] = "sha256:";
  char controller_digest[SS_SHA256_HEX_CAPACITY + 8U] = {0};
  char coupling_digest[SS_SHA256_HEX_CAPACITY + 8U] = {0};
  char prior_digest[SS_SHA256_HEX_CAPACITY + 8U] = {0};
  char cursor_digest[SS_SHA256_HEX_CAPACITY + 8U] = {0};
  char logical_digest[SS_SHA256_HEX_CAPACITY + 8U] = {0};
  char payload_digest[SS_SHA256_HEX_CAPACITY] = {0};
  char checkpoint_id[SS_SHA256_HEX_CAPACITY + 8U] = {0};
  uint64_t payload_bytes = 0U;
  uint64_t logical_bytes = 0U;
  unsigned char *envelope = NULL;
  size_t envelope_size = 0U;
  int value;
  SsStatus status;
  if (summary == NULL || spec == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(summary, 0, sizeof(*summary));
  status = ss_checkpoint_spec_validate(spec);
  status = ss_status_allreduce_max(spec->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  status = ss_checkpoint_spec_agree(spec);
  if (status != SS_OK) {
    return status;
  }
  status = ss_checkpoint_transaction_begin(spec, &transaction, failure_point);
  if (status != SS_OK) {
    return status;
  }
  status = ss_checkpoint_failure(failure_point, SS_RESULT_FAIL_HDF5_CREATE);
  if (status == SS_OK) {
    status = ss_checkpoint_root(spec, &transaction);
  }
  if (status == SS_OK) {
    status = ss_checkpoint_failure(failure_point, SS_RESULT_FAIL_METADATA_CREATE);
  }
  field_digests =
      (SsCheckpointFieldDigest *)calloc(spec->field_count, sizeof(*field_digests));
  if (status == SS_OK && field_digests == NULL) {
    status = SS_ERROR_OUT_OF_MEMORY;
  }
  status = ss_status_allreduce_max(spec->communicator, status);
  if (status == SS_OK) {
    status = ss_checkpoint_failure(failure_point, SS_RESULT_FAIL_DATA_WRITE);
  }
  if (status == SS_OK) {
    status = ss_checkpoint_write_fields(spec, &transaction, field_digests, left_digest,
                                        right_digest, source_digest, &logical_bytes);
  }
  if (status == SS_OK && spec->deflation.basis_count != 0U) {
    status = ss_checkpoint_write_replicated_double(
        &transaction, "/deflation/source_weights", spec->deflation.source_weights,
        spec->deflation.basis_count, "persistent_compression_source_weights",
        source_weights_digest);
  }
  if (status == SS_OK && spec->deflation.basis_count != 0U) {
    status = ss_checkpoint_write_replicated_double(
        &transaction, "/deflation/source_kinds", spec->deflation.source_kinds,
        spec->deflation.basis_count, "persistent_compression_source_kinds",
        source_kinds_digest);
  }
  if (status == SS_OK) {
    status = ss_checkpoint_write_replicated_double(
        &transaction, "/analysis/controller", spec->controller_values,
        spec->controller_value_count, spec->controller_id, controller_digest);
  }
  if (status == SS_OK) {
    status = ss_checkpoint_write_replicated_double(
        &transaction, "/coupling/accepted", spec->coupling_values,
        spec->coupling_value_count, spec->coupling_model_id, coupling_digest);
  }
  if (status == SS_OK) {
    status = ss_checkpoint_write_prior(spec, &transaction, prior_digest);
  }
  if (status == SS_OK) {
    status = ss_checkpoint_write_cursor(spec, &transaction, cursor_digest);
  }
  if (status == SS_OK && spec->deflation.basis_count == 0U) {
    (void)ss_sha256_bytes((const unsigned char *)"", 0U, left_digest + 7U);
    (void)ss_sha256_bytes((const unsigned char *)"", 0U, right_digest + 7U);
    (void)ss_sha256_bytes((const unsigned char *)"", 0U, source_digest + 7U);
    (void)ss_sha256_bytes((const unsigned char *)"", 0U, source_weights_digest + 7U);
    (void)ss_sha256_bytes((const unsigned char *)"", 0U, source_kinds_digest + 7U);
  }
  if (status == SS_OK) {
    status = ss_checkpoint_failure(failure_point, SS_RESULT_FAIL_FLUSH);
  }
  if (status == SS_OK && H5Fflush(transaction.file, H5F_SCOPE_GLOBAL) < 0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_status_allreduce_max(spec->communicator, status);
  if (status == SS_OK) {
    status = ss_checkpoint_failure(failure_point, SS_RESULT_FAIL_HDF5_CLOSE);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_close(&transaction.file, H5Fclose, status);
    status = ss_result_hdf5_close(&transaction.access, H5Pclose, status);
  }
  if (status == SS_OK) {
    status = ss_checkpoint_failure(failure_point, SS_RESULT_FAIL_LOGICAL_DIGEST);
  }
  if (status == SS_OK) {
    status = ss_checkpoint_logical_digest(
        spec, field_digests, left_digest, right_digest, source_digest,
        source_weights_digest, source_kinds_digest, controller_digest, coupling_digest,
        prior_digest, cursor_digest, logical_digest);
  }
  if (status == SS_OK) {
    status = ss_checkpoint_failure(failure_point, SS_RESULT_FAIL_PAYLOAD_DIGEST);
  }
  if (status == SS_OK && transaction.rank == 0) {
    status = ss_result_payload_digest(transaction.hdf5_temporary_path, payload_digest,
                                      &payload_bytes);
  }
  value = (int)status;
  if (MPI_Bcast(&value, 1, MPI_INT, 0, spec->communicator) != MPI_SUCCESS ||
      MPI_Bcast(payload_digest, SS_SHA256_HEX_CAPACITY, MPI_CHAR, 0,
                spec->communicator) != MPI_SUCCESS ||
      MPI_Bcast(&payload_bytes, 1, MPI_UINT64_T, 0, spec->communicator) !=
          MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  } else {
    status = (SsStatus)value;
  }
  if (status == SS_OK) {
    status = ss_checkpoint_id(spec->communicator, spec->run_id, spec->accepted_state_id,
                              logical_digest, checkpoint_id);
  }
  if (status == SS_OK) {
    status = ss_checkpoint_failure(failure_point, SS_RESULT_FAIL_PAYLOAD_RENAME);
  }
  if (status == SS_OK) {
    status = ss_result_rank_zero_rename(
        spec->communicator, transaction.hdf5_temporary_path, transaction.hdf5_path);
  }
  if (status == SS_OK) {
    status = ss_checkpoint_failure(failure_point, SS_RESULT_FAIL_ENVELOPE_WRITE);
  }
  if (status == SS_OK && transaction.rank == 0) {
    status =
        ss_checkpoint_envelope(spec, checkpoint_id, logical_digest, payload_digest,
                               payload_bytes, field_digests, &envelope, &envelope_size);
    if (status == SS_OK) {
      status = ss_result_write_staged_file(transaction.envelope_temporary_path,
                                           envelope, envelope_size);
    }
  }
  free(envelope);
  value = (int)status;
  if (MPI_Bcast(&value, 1, MPI_INT, 0, spec->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  } else {
    status = (SsStatus)value;
  }
  if (status == SS_OK) {
    status = ss_checkpoint_failure(failure_point, SS_RESULT_FAIL_ENVELOPE_RENAME);
  }
  if (status == SS_OK) {
    status = ss_result_rank_zero_rename(spec->communicator,
                                        transaction.envelope_temporary_path,
                                        transaction.envelope_path);
  }
  if (status == SS_OK) {
    status = ss_checkpoint_verify_written(spec, &transaction, checkpoint_id);
  }
  if (status == SS_OK) {
    status = ss_checkpoint_failure(failure_point, SS_RESULT_FAIL_DIRECTORY_RENAME);
  }
  if (status == SS_OK) {
    status = ss_result_rank_zero_rename(
        spec->communicator, transaction.staging_directory, transaction.final_directory);
  }
  status = ss_status_allreduce_max(spec->communicator, status);
  free(field_digests);
  if (status != SS_OK) {
    return ss_checkpoint_transaction_cleanup(&transaction, status);
  }
  summary->accepted_state_id = spec->accepted_state_id;
  summary->restored_field_count = spec->field_count;
  for (size_t field = 0U; field < spec->field_count; ++field) {
    summary->restored_field_values +=
        spec->fields[field].global_count * spec->fields[field].component_count;
  }
  summary->restored_primary_values =
      spec->fields[0].global_count * spec->fields[0].component_count;
  summary->restored_deflation_values =
      spec->deflation.global_count * spec->deflation.basis_count * 3U +
      (uint64_t)spec->deflation.basis_count * 2U;
  summary->deflation_basis_count = spec->deflation.basis_count;
  summary->prior_state_count = spec->prior_state_count;
  summary->history_cursor = spec->history_cursor;
  summary->accepted_state_cursor = spec->accepted_state_cursor;
  summary->retry_count = spec->retry_count;
  summary->next_step = spec->next_step;
  summary->operator_revalidation_required = spec->deflation.basis_count != 0U;
  (void)ss_result_copy_text(summary->checkpoint_id, sizeof(summary->checkpoint_id),
                            checkpoint_id);
  (void)ss_result_copy_text(summary->operator_identity,
                            sizeof(summary->operator_identity),
                            spec->deflation.operator_identity);
  (void)ss_result_copy_text(summary->source_run_id, sizeof(summary->source_run_id),
                            spec->run_id);
  (void)ss_result_copy_text(summary->source_build_id, sizeof(summary->source_build_id),
                            spec->build_id);
  summary->source_index_bits = spec->source_index_bits;
  (void)logical_bytes;
  return SS_OK;
}
