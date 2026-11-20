/* Private result-file index plans and helpers shared by the result writer,
 * checkpoints and verification.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_RESULT_INTERNAL_H
#define SS_RESULT_INTERNAL_H

#include "slope_internal/core/collective.h"
#include "slope_internal/core/util.h"
#include "slope_internal/result/result.h"

#include <hdf5.h>

#include <stddef.h>
#include <stdint.h>

struct SsResultFileIndexPlan {
  MPI_Comm communicator;
  int rank;
  int size;
  SsResultFileIndexEntry *entries;
  size_t local_count;
  SsEntityId *sorted_ids;
  size_t sorted_count;
  uint64_t sorted_offset;
  int *send_counts;
  int *send_offsets;
  int *receive_counts;
  int *receive_offsets;
  size_t *send_local_slots;
  size_t *receive_sorted_positions;
  size_t *sorted_receive_positions;
  SsResultFileIndexSummary summary;
};

/** Exchange row-major values between local slot order and contiguous file blocks. */
SsStatus ss_result_gather_indexed_rows(const SsResultFileIndexPlan *plan,
                                       size_t values_per_row, size_t value_size,
                                       MPI_Datatype value_type,
                                       const void *local_values, void **sorted_values);
/* The reverse route may reorder its file block in place. */
/** Route row-major values from contiguous file blocks back to local slot order;
 * collective. The reverse route may reorder its file block in place. */
SsStatus ss_result_scatter_indexed_rows(const SsResultFileIndexPlan *plan,
                                        size_t values_per_row, size_t value_size,
                                        MPI_Datatype value_type, void *sorted_values,
                                        void *local_values);

/** Check that every rank holds the same digest; collective. */
SsStatus ss_result_collective_digest_agree(MPI_Comm communicator, int size,
                                           const char digest[SS_SHA256_HEX_CAPACITY],
                                           SsStatus status);
/** Check that every rank holds the same payload bytes; collective. */
SsStatus ss_result_collective_payload_agree(MPI_Comm communicator,
                                            const unsigned char *payload,
                                            size_t payload_size);
/** Whether a quantity or category name is valid. */
int ss_result_name_valid(const char *value);
/** Whether the identities of a writer specification are valid. */
int ss_result_spec_identity_valid(const char *directory, const char *problem_id,
                                  const char *run_id, const char *build_id,
                                  const char *mesh_id, const char *geometry_id);
/** Whether a field and geometry order pair is supported. */
int ss_result_orders_valid(uint32_t field_order, uint32_t geometry_order);
/** Whether a text is nonempty and fits its capacity. */
int ss_result_text_is_valid(const char *value, size_t capacity);
/** Copy a text into a fixed buffer, failing on truncation. */
SsStatus ss_result_copy_text(char *destination, size_t capacity, const char *source);
/** `sha256:` digest of a byte range. */
SsStatus ss_result_digest_bytes(const void *data, size_t size,
                                char output[SS_SHA256_HEX_CAPACITY + 8U]);
/** Broadcast rank zero's digest and agree on the status; collective. */
SsStatus ss_result_broadcast_digest(MPI_Comm communicator, SsStatus status,
                                    char output[SS_SHA256_HEX_CAPACITY + 8U]);
/** Store a 64-bit value in little-endian order. */
void ss_result_store_u64_le(unsigned char output[8], uint64_t value);
/** Store the bits of a double in little-endian order. */
void ss_result_store_double_le(unsigned char output[8], double value);

/** Write a string attribute. */
SsStatus ss_result_hdf5_string_attribute(hid_t object, const char *name,
                                         const char *value);
/** Write a 64-bit unsigned attribute. */
SsStatus ss_result_hdf5_u64_attribute(hid_t object, const char *name, uint64_t value);
/** Read a string attribute. */
SsStatus ss_result_hdf5_read_string_attribute(hid_t object, const char *name,
                                              char *output, size_t capacity);
/** Read a 64-bit unsigned attribute. */
SsStatus ss_result_hdf5_read_u64_attribute(hid_t object, const char *name,
                                           uint64_t *value);
/** Create HDF5 groups in order; collective. */
SsStatus ss_result_hdf5_create_groups(hid_t file, MPI_Comm communicator,
                                      const char *const *paths, size_t count);
/** Close an HDF5 handle if open and fold the outcome into a status. */
SsStatus ss_result_hdf5_close(hid_t *identifier, herr_t (*closer)(hid_t),
                              SsStatus status);

/** SHA-256 and size of a payload file; local. */
SsStatus ss_result_payload_digest(const char *path, char digest[SS_SHA256_HEX_CAPACITY],
                                  uint64_t *bytes);
/** Synchronize a directory entry to disk; local. */
SsStatus ss_result_fsync_directory(const char *path);
/** Write a new file and synchronize it; local. */
SsStatus ss_result_write_staged_file(const char *path, const void *data, size_t size);
/** Write a text file through a temporary file and rename; local. */
SsStatus ss_result_write_text_atomic(const char *path, const char *text, size_t size);
/** Remove a staging directory and its files; local. */
SsStatus ss_result_remove_staging(const char *path);
/** Split a path into its parent directory and final name. */
SsStatus ss_result_path_split(const char *final_path, char *parent,
                              size_t parent_capacity, char *name, size_t name_capacity);
/** Create a hidden staging directory next to the final path on rank zero; collective.
 */
SsStatus ss_result_make_staging(MPI_Comm communicator, const char *final_path,
                                const char *kind, char *staging,
                                size_t staging_capacity);
/** Rename on rank zero and agree on the outcome; collective. */
SsStatus ss_result_rank_zero_rename(MPI_Comm communicator, const char *source,
                                    const char *destination);

/** Write a dataset in stable-identity order with hyperslab I/O and its digest;
 * collective. */
SsStatus ss_result_write_indexed_dataset(
    hid_t file, MPI_Comm communicator, const char *group_path, const char *quantity_id,
    uint32_t quantity_version, const char *unit, const char *location,
    const char *basis, const char *mesh_id, const char *geometry_id, SsStateId state_id,
    uint32_t field_order, uint32_t geometry_order, const char *const *component_ids,
    uint32_t component_count, uint64_t dataset_chunk_rows,
    const SsResultFileIndexPlan *plan, const double *local_values,
    char logical_digest[SS_SHA256_HEX_CAPACITY + 8U], uint64_t *logical_bytes);

/** Read a dataset in stable-identity order into local slots; collective. */
SsStatus ss_result_read_indexed_dataset(hid_t file, MPI_Comm communicator,
                                        const char *group_path,
                                        const SsResultFileIndexPlan *plan,
                                        uint32_t component_count, double *local_values);
/** Recompute and check the logical digest of a dataset; collective. */
SsStatus ss_result_verify_indexed_dataset_digest(
    hid_t file, MPI_Comm communicator, const char *group_path,
    char logical_digest[SS_SHA256_HEX_CAPACITY + 8U]);

/** Number of canonical file rows hashed together in a quantity digest chunk. */
#define SS_RESULT_QUANTITY_DIGEST_CHUNK_ROWS 1024U
/** Capacity of the comma-separated state-ID text of one quantity. */
#define SS_RESULT_STATE_TEXT_CAPACITY 4096U
/** Capacity of the semantic text hashed ahead of the chunk digests. */
#define SS_RESULT_SEMANTIC_CAPACITY 16384U

/** Shape of a stored quantity as recorded by its HDF5 datasets. */
typedef struct {
  SsResultDatasetKind kind;
  uint64_t states;
  uint64_t entities;
  uint32_t samples;
  uint32_t components;
  int value_rank;
} SsResultQuantityShape;

/** Metadata attached to a quantity once, when the writer finalizes it. */
typedef struct {
  SsResultDatasetKind kind;
  const char *quantity_id;
  uint32_t quantity_version;
  const char *unit;
  const char *location;
  const char *basis;
  const char *quadrature_id;
  const char *mesh_id;
  const char *geometry_id;
  uint32_t field_order;
  uint32_t geometry_order;
  /** Comma-joined component IDs, or category IDs for categorical quantities. */
  const char *semantic_ids;
  const SsStateId *state_ids;
  uint64_t state_count;
} SsResultQuantityFinal;

/** Canonical dataset-kind name ("field", "quadrature", ...), or NULL if invalid. */
const char *ss_result_quantity_kind_name(SsResultDatasetKind kind);

/** Read the shape of an open quantity (dataset sizes and `dataset_kind`). */
SsStatus ss_result_quantity_shape(hid_t ids_dataset, hid_t states_dataset,
                                  hid_t values_dataset, SsResultQuantityShape *shape);

/** Format nonzero, distinct state IDs as comma-separated decimal text. */
SsStatus ss_result_quantity_state_text(const SsStateId *state_ids, uint64_t state_count,
                                       char *output, size_t capacity);

/**
 * Collectively compute the canonical quantity digest of open datasets.
 *
 * Reads every stored value exactly once with a fixed scratch cap and gathers
 * chunk digests to rank zero in bounded batches; the result is broadcast.
 */
SsStatus ss_result_quantity_digest(hid_t ids_dataset, hid_t states_dataset,
                                   hid_t values_dataset, MPI_Comm communicator,
                                   char output[SS_SHA256_HEX_CAPACITY + 8U],
                                   SsResultQuantityShape *shape);

/**
 * Collectively append one state of a float64 quantity.
 *
 * Extends the state axis, writes the state ID and the redistributed values.
 * Earlier states are neither read nor hashed; `logical_bytes` receives the
 * logical size of all states so far.
 */
SsStatus ss_result_quantity_append(hid_t file, MPI_Comm communicator,
                                   const char *group_path,
                                   SsResultDatasetKind dataset_kind, SsStateId state_id,
                                   uint32_t component_count, uint32_t sample_count,
                                   const SsResultFileIndexPlan *plan,
                                   const double *local_values, uint64_t state_index,
                                   uint64_t *logical_bytes);

/** Create the datasets of a declared quantity and write its canonical IDs. */
SsStatus ss_result_quantity_declare(hid_t file, MPI_Comm communicator,
                                    const char *group_path,
                                    SsResultDatasetKind dataset_kind,
                                    uint32_t sample_count, uint32_t component_count,
                                    uint64_t dataset_chunk_rows,
                                    const SsResultFileIndexPlan *plan);

/** Collectively append one state of a uint32 categorical quantity. */
SsStatus ss_result_category_append(hid_t file, MPI_Comm communicator,
                                   const char *group_path, SsStateId state_id,
                                   uint32_t category_count, uint32_t sample_count,
                                   const SsResultFileIndexPlan *plan,
                                   const uint32_t *local_values, uint64_t state_index,
                                   uint64_t *logical_bytes);

/**
 * Collectively write the descriptive attributes of a quantity once and store
 * its canonical logical digest, computed with a single bounded pass.
 */
SsStatus ss_result_quantity_finalize(hid_t file, MPI_Comm communicator,
                                     const char *group_path,
                                     const SsResultQuantityFinal *final,
                                     char logical_digest[SS_SHA256_HEX_CAPACITY + 8U]);

/** Recompute a stored quantity digest and compare it with its attribute. */
SsStatus
ss_result_verify_quantity_digest(hid_t file, MPI_Comm communicator,
                                 const char *group_path,
                                 char logical_digest[SS_SHA256_HEX_CAPACITY + 8U]);

#endif
