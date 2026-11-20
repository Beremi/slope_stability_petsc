/* Private writer state, records and publication interfaces.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_RESULT_WRITER_INTERNAL_H
#define SS_RESULT_WRITER_INTERNAL_H

#include "result_internal.h"

#include "slope_internal/core/sha256.h"

typedef struct {
  char quantity_id[SS_RESULT_TEXT_CAPACITY];
  char unit[SS_RESULT_TEXT_CAPACITY];
  char location[SS_RESULT_TEXT_CAPACITY];
  char basis[SS_RESULT_TEXT_CAPACITY];
  char quadrature_id[SS_RESULT_TEXT_CAPACITY];
  char component_ids[SS_RESULT_COMPONENT_CAPACITY * SS_RESULT_TEXT_CAPACITY];
  char category_ids[SS_RESULT_COMPONENT_CAPACITY * SS_RESULT_TEXT_CAPACITY];
  char group_path[SS_RESULT_PATH_CAPACITY];
  char logical_digest[SS_SHA256_HEX_CAPACITY + 8U];
  uint32_t quantity_version;
  uint32_t component_count;
  uint32_t category_count;
  uint32_t sample_count;
  SsResultDatasetKind dataset_kind;
  SsStateId *state_ids;
  size_t state_count;
  size_t state_capacity;
  /** Index of the shared entity layout in SsResultWriter::layouts. */
  size_t layout_index;
  uint64_t global_count;
  uint64_t logical_bytes;
} SsResultFieldRecord;

typedef struct {
  char quantity_id[SS_RESULT_TEXT_CAPACITY];
  char unit[SS_RESULT_TEXT_CAPACITY];
  uint32_t quantity_version;
  SsStateId state_id;
  double value;
  int complete;
} SsResultScalarRecord;

typedef struct {
  SsStateId state_id;
  char checkpoint_id[SS_SHA256_HEX_CAPACITY + 8U];
  char relative_path[SS_RESULT_PATH_CAPACITY];
  int complete;
} SsResultCheckpointRecord;

struct SsResultWriter {
  MPI_Comm communicator;
  int rank;
  int size;
  SsResultWriterState state;
  SsResultTerminalStatus terminal_status;
  SsResultFailurePoint failure_point;
  char final_directory[SS_RESULT_PATH_CAPACITY];
  char staging_directory[SS_RESULT_PATH_CAPACITY];
  char hdf5_temporary_path[SS_RESULT_PATH_CAPACITY];
  char hdf5_path[SS_RESULT_PATH_CAPACITY];
  char environment_path[SS_RESULT_PATH_CAPACITY];
  char result_path[SS_RESULT_PATH_CAPACITY];
  char result_temporary_path[SS_RESULT_PATH_CAPACITY];
  char environment_sha256[SS_SHA256_HEX_CAPACITY];
  char problem_id[SS_RESULT_TEXT_CAPACITY];
  char run_id[SS_RESULT_TEXT_CAPACITY];
  char build_id[SS_RESULT_TEXT_CAPACITY];
  char parent_checkpoint_id[SS_RESULT_TEXT_CAPACITY];
  char parent_run_id[SS_RESULT_TEXT_CAPACITY];
  char parent_build_id[SS_RESULT_TEXT_CAPACITY];
  SsStateId parent_accepted_state_id;
  uint32_t parent_source_index_bits;
  int restart_provenance_valid;
  int mechanics_assembly_minimum_threads;
  int mechanics_assembly_maximum_threads;
  char mesh_id[SS_RESULT_TEXT_CAPACITY];
  char geometry_id[SS_RESULT_TEXT_CAPACITY];
  char analysis_method[SS_RESULT_TEXT_CAPACITY];
  char stop_reason[SS_RESULT_TEXT_CAPACITY];
  char analysis_outcome[SS_RESULT_TEXT_CAPACITY];
  char validation_policy[SS_RESULT_TEXT_CAPACITY];
  char mechanics_assembly_mode[SS_RESULT_TEXT_CAPACITY];
  uint32_t dimension;
  uint32_t field_order;
  uint32_t geometry_order;
  uint64_t dataset_chunk_rows;
  uint64_t environment_bytes;
  hid_t access;
  hid_t file;
  SsResultFieldRecord *fields;
  size_t field_count;
  size_t field_capacity;
  /** One file-index plan per distinct entity layout, shared by quantities. */
  SsResultFileIndexPlan **layouts;
  size_t layout_count;
  size_t layout_capacity;
  /** Rows [history_buffer_begin, history_count) not yet written to HDF5. */
  SsResultHistoryRow *history;
  size_t history_count;
  size_t history_capacity;
  uint64_t history_buffer_begin;
  /** Rows already written to each `/history/<kind>/rows` table. */
  uint64_t history_written[SS_RESULT_HISTORY_PMG_APPLY + 1];
  SsResultAcceptedState *states;
  size_t state_count;
  size_t state_capacity;
  SsResultScalarRecord *scalars;
  size_t scalar_count;
  size_t scalar_capacity;
  SsResultCheckpointRecord *checkpoints;
  size_t checkpoint_count;
  size_t checkpoint_capacity;
  int collective_plan_frozen;
  char collective_plan_digest[SS_SHA256_HEX_CAPACITY + 8U];
  SsErrorRecord terminal_error;
  uint64_t local_io_bytes;
  double created_at;
  SsResultWriterSummary summary;
};

/* Result declarations, specification validation and rank agreement */
/** Whether a path is a safe relative path inside a result. */
int ss_result_relative_path_valid(const char *value);
/** Protocol name of a dataset kind. */
const char *ss_result_writer_dataset_kind_name(SsResultDatasetKind kind);
/** Comma-separated component list of a quantity. */
SsStatus ss_result_writer_component_text(const char *const *component_ids,
                                         uint32_t component_count, char *output,
                                         size_t capacity);
/** Check that history rows are identical on all ranks; collective. */
SsStatus ss_result_writer_history_agree(const SsResultWriter *writer,
                                        const SsResultHistoryRow *rows, size_t count);
/** Check that an accepted-state row is identical on all ranks; collective. */
SsStatus ss_result_writer_state_agree(const SsResultWriter *writer,
                                      const SsResultAcceptedState *state);
/** Check that the replicated metadata of a field view agree; collective. */
SsStatus ss_result_writer_field_agree(const SsResultWriter *writer,
                                      const SsResultFieldView *field,
                                      SsResultDatasetKind dataset_kind,
                                      const char *quadrature_id, uint32_t sample_count);
/** Check that the replicated metadata of a category view agree; collective. */
SsStatus ss_result_writer_category_agree(const SsResultWriter *writer,
                                         const SsResultCategoryView *category);
/** Check that the terminal status is identical on all ranks; collective. */
SsStatus ss_result_writer_terminal_agree(const SsResultWriter *writer,
                                         SsResultTerminalStatus status,
                                         const char *stop_reason,
                                         const SsErrorRecord *error);
/** Validate a writer specification; local. */
SsStatus ss_result_writer_spec_validate(const SsResultWriterSpec *spec);
/** Check that a writer specification agrees on all ranks; collective. */
SsStatus ss_result_writer_spec_agree(const SsResultWriterSpec *spec, int size);
/** Copy the strings of a specification into the writer. */
SsStatus ss_result_writer_copy_spec(SsResultWriter *writer,
                                    const SsResultWriterSpec *spec);

/* Result writer state machine, lifecycle, and terminal status */
/** Return ::SS_ERROR_INJECTED and mark the writer failed at the injected point. */
SsStatus ss_result_writer_fail(SsResultWriter *writer, SsResultFailurePoint point);

/* Shared entity layouts and one-time quantity finalization */
/** Collectively find the layout equal to `spec` or create its plan. */
SsStatus ss_result_writer_layout_acquire(SsResultWriter *writer,
                                         const SsResultFileIndexSpec *spec,
                                         size_t *layout_index);
/**
 * Collectively bind an append's entity arrays to its quantity layout.
 *
 * `plan` receives the shared plan when the arrays equal the declared ones.
 * A different local distribution of the same stable-ID set yields a
 * `temporary` plan that the caller destroys; a different set fails with
 * SS_ERROR_INTEGRITY without reading the file.
 */
SsStatus ss_result_writer_layout_bind(SsResultWriter *writer, size_t layout_index,
                                      const SsResultFileIndexSpec *spec,
                                      SsResultFileIndexPlan **temporary,
                                      const SsResultFileIndexPlan **plan);
/** Destroy every shared layout plan. */
SsStatus ss_result_writer_layouts_destroy(SsResultWriter *writer);
/** Write quantity attributes once and store each logical digest (one pass). */
SsStatus ss_result_writer_finalize_quantities(SsResultWriter *writer);

/* Create, populate and verify the result HDF5 layout */
/** HDF5 file type of a history row (row layout v2). */
hid_t ss_result_history_file_type(void);
/** HDF5 memory type of SsResultHistoryRow. */
hid_t ss_result_history_memory_type(void);
/** Table name of a history kind ("accepted", "attempt", ...), or NULL. */
const char *ss_result_history_name(SsResultHistoryKind kind);
/** Collectively write buffered history rows below `row_end` and drop them. */
SsStatus ss_result_writer_flush_history(SsResultWriter *writer, uint64_t row_end);
/** Feed the ten digest columns of all rows, in row-ID order, to `context`. */
SsStatus ss_result_writer_history_digest(const SsResultWriter *writer,
                                         const char *hdf5_path, SsSha256 *context);
/** Write the accepted-state index and history tables; collective. */
SsStatus ss_result_writer_write_relations(SsResultWriter *writer);
/** Re-read the written file and check its logical digest; collective. */
SsStatus ss_result_writer_verify_hdf5(SsResultWriter *writer);

#endif
