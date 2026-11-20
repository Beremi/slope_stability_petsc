/* Private checkpoint record layouts and read/write interfaces.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_RESULT_CHECKPOINT_INTERNAL_H
#define SS_RESULT_CHECKPOINT_INTERNAL_H

#include "result_internal.h"

#define SS_CHECKPOINT_FIELD_IDS_CAPACITY                                               \
  (SS_CHECKPOINT_FIELD_CAPACITY * SS_RESULT_TEXT_CAPACITY)

typedef struct {
  MPI_Comm communicator;
  int rank;
  char final_directory[SS_RESULT_PATH_CAPACITY];
  char staging_directory[SS_RESULT_PATH_CAPACITY];
  char hdf5_temporary_path[SS_RESULT_PATH_CAPACITY];
  char hdf5_path[SS_RESULT_PATH_CAPACITY];
  char envelope_temporary_path[SS_RESULT_PATH_CAPACITY];
  char envelope_path[SS_RESULT_PATH_CAPACITY];
  hid_t access;
  hid_t file;
} SsCheckpointTransaction;

typedef struct {
  char quantity_id[SS_RESULT_TEXT_CAPACITY];
  char unit[SS_RESULT_TEXT_CAPACITY];
  char space_id[SS_RESULT_TEXT_CAPACITY];
  char basis[SS_RESULT_TEXT_CAPACITY];
  char components[SS_RESULT_COMPONENT_CAPACITY * SS_RESULT_TEXT_CAPACITY];
  char digest[SS_SHA256_HEX_CAPACITY + 8U];
  uint64_t global_count;
  uint32_t component_count;
} SsCheckpointFieldDigest;

typedef struct {
  uint64_t history_cursor;
  uint64_t accepted_state_cursor;
  uint64_t retry_count;
  double next_step;
} SsCheckpointCursor;

/* Validate, digest and serialize checkpoint identities and envelopes */
/** Join a checkpoint root and a leaf name. */
SsStatus ss_checkpoint_path(char *output, size_t capacity, const char *root,
                            const char *leaf);
/** Comma-separated component list of a field. */
SsStatus ss_checkpoint_components(const char *const *component_ids,
                                  uint32_t component_count, char *output,
                                  size_t capacity);
/** Comma-separated field identifiers of a specification. */
SsStatus ss_checkpoint_spec_field_ids(const SsCheckpointSpec *spec, char *output,
                                      size_t capacity);
/** Comma-separated field identifiers of a restore view. */
SsStatus ss_checkpoint_restore_field_ids(const SsCheckpointRestoreView *restore,
                                         char *output, size_t capacity);
/** Validate a checkpoint specification; local. */
SsStatus ss_checkpoint_spec_validate(const SsCheckpointSpec *spec);
/** Check that the replicated parts of a specification agree; collective. */
SsStatus ss_checkpoint_spec_agree(const SsCheckpointSpec *spec);
/** Logical digest of a checkpoint from its identities and part digests; collective. */
SsStatus ss_checkpoint_logical_digest_values(
    MPI_Comm communicator, const char *problem_id, const char *build_id,
    const char *build_compatibility_id, const char *mesh_id, const char *geometry_id,
    const char *field_space_id, const char *numerical_policy_id,
    const char *material_model_id, const char *hydraulic_model_id,
    const char *coupling_model_id, const char *analysis_method,
    const char *controller_id, const char *deflation_provenance_id,
    const char *operator_identity, uint32_t field_order, uint32_t geometry_order,
    uint32_t scalar_bits, uint32_t source_index_bits, SsStateId accepted_state_id,
    const SsCheckpointFieldDigest *fields, size_t field_count, const char *left_digest,
    const char *right_digest, const char *source_digest,
    const char *source_weights_digest, const char *source_kinds_digest,
    const char *controller_digest, const char *coupling_digest,
    const char *prior_digest, const char *cursor_digest,
    char output[SS_SHA256_HEX_CAPACITY + 8U]);
/** Logical digest of a checkpoint specification; collective. */
SsStatus ss_checkpoint_logical_digest(
    const SsCheckpointSpec *spec, const SsCheckpointFieldDigest *fields,
    const char *left_digest, const char *right_digest, const char *source_digest,
    const char *source_weights_digest, const char *source_kinds_digest,
    const char *controller_digest, const char *coupling_digest,
    const char *prior_digest, const char *cursor_digest,
    char output[SS_SHA256_HEX_CAPACITY + 8U]);
/** Checkpoint identity from run, state and logical digest; collective. */
SsStatus ss_checkpoint_id(MPI_Comm communicator, const char *run_id, SsStateId state_id,
                          const char *logical_digest,
                          char output[SS_SHA256_HEX_CAPACITY + 8U]);
/** Serialize `checkpoint.json`. */
SsStatus ss_checkpoint_envelope(const SsCheckpointSpec *spec, const char *checkpoint_id,
                                const char *logical_digest, const char *payload_digest,
                                uint64_t payload_bytes,
                                const SsCheckpointFieldDigest *fields,
                                unsigned char **output, size_t *output_size);
/** Read and check the identities and digests of `checkpoint.json`; local. */
SsStatus ss_checkpoint_envelope_read(const char *directory,
                                     char checkpoint_id[SS_SHA256_HEX_CAPACITY + 8U],
                                     char logical_digest[SS_SHA256_HEX_CAPACITY + 8U],
                                     char payload_digest[SS_SHA256_HEX_CAPACITY],
                                     uint64_t *payload_bytes, SsStateId *state_id);

/* Read and write checkpoint HDF5 fields and replicated records */
/** Write a small replicated array from rank zero with its digest; collective. */
SsStatus ss_checkpoint_write_replicated_double(
    SsCheckpointTransaction *transaction, const char *path, const double *values,
    size_t count, const char *semantic_id, char digest[SS_SHA256_HEX_CAPACITY + 8U]);
/** Write the prior accepted states; collective. */
SsStatus ss_checkpoint_write_prior(const SsCheckpointSpec *spec,
                                   SsCheckpointTransaction *transaction,
                                   char digest[SS_SHA256_HEX_CAPACITY + 8U]);
/** Write the history and accepted-state cursors; collective. */
SsStatus ss_checkpoint_write_cursor(const SsCheckpointSpec *spec,
                                    SsCheckpointTransaction *transaction,
                                    char digest[SS_SHA256_HEX_CAPACITY + 8U]);
/** Write the fields and the deflation basis in stable order; collective. */
SsStatus ss_checkpoint_write_fields(const SsCheckpointSpec *spec,
                                    SsCheckpointTransaction *transaction,
                                    SsCheckpointFieldDigest *digests,
                                    char left_digest[SS_SHA256_HEX_CAPACITY + 8U],
                                    char right_digest[SS_SHA256_HEX_CAPACITY + 8U],
                                    char source_digest[SS_SHA256_HEX_CAPACITY + 8U],
                                    uint64_t *logical_bytes);
/** Rows and columns of a checkpoint dataset. */
SsStatus ss_checkpoint_dataset_shape(hid_t file, const char *path, uint64_t *rows,
                                     uint32_t *columns);
/** Read a replicated array on rank zero and broadcast it; collective. */
SsStatus ss_checkpoint_read_replicated_double(hid_t file, MPI_Comm communicator,
                                              int rank, const char *path,
                                              double *values, size_t capacity,
                                              size_t *count,
                                              char digest[SS_SHA256_HEX_CAPACITY + 8U]);
/** Read the prior accepted states; collective. */
SsStatus ss_checkpoint_read_prior(hid_t file, MPI_Comm communicator, int rank,
                                  SsCheckpointPriorState *states, size_t capacity,
                                  size_t *count,
                                  char digest[SS_SHA256_HEX_CAPACITY + 8U]);
/** Read the history and accepted-state cursors; collective. */
SsStatus ss_checkpoint_read_cursor(hid_t file, MPI_Comm communicator, int rank,
                                   SsCheckpointCursor *cursor,
                                   char digest[SS_SHA256_HEX_CAPACITY + 8U]);
/** Read the identity attributes of a checkpoint field. */
SsStatus ss_checkpoint_field_metadata(hid_t file, const char *group_path,
                                      SsCheckpointFieldDigest *field);

/* Validate compatibility and recover accepted checkpoint state */
/** Re-read a written checkpoint and check its identity; collective. */
SsStatus ss_checkpoint_verify_written(const SsCheckpointSpec *spec,
                                      const SsCheckpointTransaction *transaction,
                                      const char *expected_checkpoint_id);

#endif
