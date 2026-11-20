/** @file
 * @brief Collective result writer, layout-independent file indexing, result
 * verification and accepted-state checkpoints.
 *
 * All data are written by stable identity with parallel HDF5, so files do
 * not depend on the number of ranks. Writer and checkpoint calls are
 * collective on their communicator and run on the master thread.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_RESULT_RESULT_H
#define SS_RESULT_RESULT_H

#include "slope_internal/core/core.h"

#include <mpi.h>

#include <stddef.h>
#include <stdint.h>

/** Capacity of a result identity or short text. */
#define SS_RESULT_TEXT_CAPACITY 96U
/** Capacity of a result path. */
#define SS_RESULT_PATH_CAPACITY 1024U
/** Largest number of components of a quantity. */
#define SS_RESULT_COMPONENT_CAPACITY 64U
/** Largest number of controller values in a checkpoint. */
#define SS_RESULT_CONTROLLER_CAPACITY 16U
/** Largest number of coupling values in a checkpoint. */
#define SS_RESULT_COUPLING_CAPACITY 9U
/** Largest number of fields in a checkpoint. */
#define SS_CHECKPOINT_FIELD_CAPACITY 16U
/** Largest number of prior accepted states in a checkpoint. */
#define SS_CHECKPOINT_PRIOR_STATE_CAPACITY 8U

/** State of a result writer. */
typedef enum {
  SS_RESULT_WRITER_UNOPENED = 0,          /**< Created, nothing written. */
  SS_RESULT_WRITER_STAGED = 1,            /**< Staging files created. */
  SS_RESULT_WRITER_STREAMING = 2,         /**< Accepting data. */
  SS_RESULT_WRITER_CLOSING = 3,           /**< Terminal status recorded. */
  SS_RESULT_WRITER_PUBLISHED = 4,         /**< Directory published. */
  SS_RESULT_WRITER_PUBLICATION_FAILED = 5 /**< Publication failed. */
} SsResultWriterState;

/** Terminal status of a run. */
typedef enum {
  SS_RESULT_SUCCEEDED = 1, /**< The analysis finished. */
  SS_RESULT_FAILED = 2,    /**< The run failed. */
  SS_RESULT_CANCELLED = 3  /**< The run was cancelled. */
} SsResultTerminalStatus;

/** Publication step at which a test injects a failure. */
typedef enum {
  SS_RESULT_FAIL_NONE = 0,              /**< No failure. */
  SS_RESULT_FAIL_DIRECTORY_CREATE = 1,  /**< Creating the staging directory. */
  SS_RESULT_FAIL_ENVIRONMENT_WRITE = 2, /**< Writing `environment.json`. */
  SS_RESULT_FAIL_HDF5_CREATE = 3,       /**< Creating the HDF5 file. */
  SS_RESULT_FAIL_METADATA_CREATE = 4,   /**< Creating HDF5 metadata. */
  SS_RESULT_FAIL_DATA_WRITE = 5,        /**< Writing data. */
  SS_RESULT_FAIL_FLUSH = 6,             /**< Flushing the file. */
  SS_RESULT_FAIL_HDF5_CLOSE = 7,        /**< Closing the file. */
  SS_RESULT_FAIL_LOGICAL_DIGEST = 8,    /**< Computing the logical digest. */
  SS_RESULT_FAIL_PAYLOAD_DIGEST = 9,    /**< Hashing the payload. */
  SS_RESULT_FAIL_PAYLOAD_RENAME = 10,   /**< Renaming the payload. */
  SS_RESULT_FAIL_ENVELOPE_WRITE = 11,   /**< Writing the envelope. */
  SS_RESULT_FAIL_ENVELOPE_RENAME = 12,  /**< Renaming the envelope. */
  SS_RESULT_FAIL_DIRECTORY_RENAME = 13  /**< Renaming the directory. */
} SsResultFailurePoint;

/** History table of a row. */
typedef enum {
  SS_RESULT_HISTORY_ACCEPTED = 1,    /**< Accepted continuation state. */
  SS_RESULT_HISTORY_ATTEMPT = 2,     /**< Continuation attempt. */
  SS_RESULT_HISTORY_NEWTON = 3,      /**< Newton iteration. */
  SS_RESULT_HISTORY_LINE_SEARCH = 4, /**< Line-search trial. */
  SS_RESULT_HISTORY_LINEAR = 5,      /**< Linear solve. */
  SS_RESULT_HISTORY_PMG = 6,         /**< PMG setup or refresh. */
  SS_RESULT_HISTORY_DEFLATION = 7,   /**< Deflation update. */
  SS_RESULT_HISTORY_SEEPAGE = 8,     /**< Seepage iteration. */
  SS_RESULT_HISTORY_IO = 9,          /**< Result or checkpoint I/O. */
  SS_RESULT_HISTORY_KRYLOV = 10,     /**< Krylov iteration (optional detail). */
  SS_RESULT_HISTORY_PMG_APPLY = 11   /**< PMG cycle (optional detail). */
} SsResultHistoryKind;

/** Layout of a quantity dataset. */
typedef enum {
  SS_RESULT_DATASET_FIELD = 1,              /**< Nodal field. */
  SS_RESULT_DATASET_QUADRATURE = 2,         /**< Quadrature-point values. */
  SS_RESULT_DATASET_CELL_REDUCTION = 3,     /**< Per-cell reduction. */
  SS_RESULT_DATASET_QUADRATURE_CATEGORY = 4 /**< Quadrature-point categories. */
} SsResultDatasetKind;

/** File row of one local entity. */
typedef struct {
  SsEntityId stable_id; /**< Stable identity. */
  size_t local_slot;    /**< Local slot of the value. */
  uint64_t file_index;  /**< Row in the file (sorted by identity). */
} SsResultFileIndexEntry;

/** Size of a file index plan. */
typedef struct {
  uint64_t global_count;              /**< Rows in the file. */
  size_t local_count;                 /**< Local entities. */
  size_t contiguous_runs;             /**< Contiguous file runs of the local rows. */
  uint64_t exchanged_bytes;           /**< Bytes exchanged to build the plan. */
  uint64_t root_gathered_value_bytes; /**< Bytes gathered on rank zero (zero). */
} SsResultFileIndexSummary;

/** Opaque plan that orders distributed entities by stable identity in a file;
 * see ss_result_file_index_plan_create(). */
typedef struct SsResultFileIndexPlan SsResultFileIndexPlan;

/** Input of ss_result_file_index_plan_create(). */
typedef struct {
  MPI_Comm communicator;        /**< Communicator of the ranks. */
  const SsEntityId *stable_ids; /**< Local stable identities. */
  const size_t *local_slots;    /**< Local slot of each identity. */
  size_t local_count;           /**< Local entities. */
  uint64_t global_count;        /**< Entities on all ranks. */
} SsResultFileIndexSpec;

/** Identities and layout of a result; strings are copied. */
typedef struct {
  const char *final_directory;        /**< Result directory to publish. */
  const char *problem_id;             /**< Problem identity. */
  const char *run_id;                 /**< Run identity. */
  const char *build_id;               /**< Build identity. */
  const char *parent_checkpoint_id;   /**< Checkpoint of a restart, or NULL. */
  const char *parent_run_id;          /**< Run of the parent checkpoint. */
  const char *parent_build_id;        /**< Build of the parent checkpoint. */
  SsStateId parent_accepted_state_id; /**< State the restart started from. */
  uint32_t parent_source_index_bits;  /**< PetscInt width of the parent run. */
  const char *mesh_id;                /**< Mesh identity. */
  const char *geometry_id;            /**< Geometry identity. */
  const char *analysis_method;        /**< Analysis kind. */
  uint32_t dimension;                 /**< Spatial dimension. */
  uint32_t field_order;               /**< Field order. */
  uint32_t geometry_order;            /**< Geometry order. */
  uint64_t dataset_chunk_rows;        /**< HDF5 chunk rows. */
  MPI_Comm communicator;              /**< Communicator of the writers. */
  /** Smallest observed assembly team; zero (with the maximum) if unknown. */
  int mechanics_assembly_minimum_threads;
  int mechanics_assembly_maximum_threads; /**< Largest observed assembly team. */
} SsResultWriterSpec;

/** Values of one quantity at one state; borrowed for the call. */
typedef struct {
  const char *quantity_id;          /**< Quantity identifier. */
  uint32_t quantity_version;        /**< Quantity version. */
  const char *unit;                 /**< Unit. */
  const char *location;             /**< Location (node, quadrature point, cell). */
  const char *basis;                /**< Basis identity. */
  const char *const *component_ids; /**< Component identifiers. */
  uint32_t component_count;         /**< Number of components. */
  SsStateId state_id;               /**< Accepted state of the values. */
  const SsEntityId *stable_ids;     /**< Local stable identities. */
  const size_t *local_slots;        /**< Local slot of each identity. */
  const double *values;             /**< Values, slot-major. */
  size_t local_count;               /**< Local entities. */
  uint64_t global_count;            /**< Entities on all ranks. */
  SsResultDatasetKind dataset_kind; /**< Dataset layout. */
  const char *quadrature_id;        /**< Quadrature rule, if any. */
  uint32_t sample_count;            /**< Samples per entity. */
} SsResultFieldView;

/** Categories of one quantity at one state; borrowed for the call. */
typedef struct {
  const char *quantity_id;         /**< Quantity identifier. */
  uint32_t quantity_version;       /**< Quantity version. */
  const char *unit;                /**< Unit. */
  const char *location;            /**< Location (node, quadrature point, cell). */
  const char *quadrature_id;       /**< Quadrature rule, if any. */
  const char *const *category_ids; /**< Category names. */
  uint32_t category_count;         /**< Number of categories. */
  SsStateId state_id;              /**< Accepted state of the values. */
  const SsEntityId *stable_ids;    /**< Local stable identities. */
  const size_t *local_slots;       /**< Local slot of each identity. */
  const uint32_t *values;          /**< Category values, slot-major. */
  size_t local_count;              /**< Local entities. */
  uint64_t global_count;           /**< Entities on all ranks. */
  uint32_t sample_count;           /**< Samples per entity. */
} SsResultCategoryView;

/** Declaration of a quantity before streaming. */
typedef struct {
  const char *quantity_id;          /**< Quantity identifier. */
  uint32_t quantity_version;        /**< Quantity version. */
  const char *unit;                 /**< Unit. */
  const char *location;             /**< Location (node, quadrature point, cell). */
  const char *basis;                /**< Basis identity. */
  const char *const *component_ids; /**< Component identifiers. */
  uint32_t component_count;         /**< Number of components. */
  const SsEntityId *stable_ids;     /**< Local stable identities. */
  const size_t *local_slots;        /**< Local slot of each identity. */
  size_t local_count;               /**< Local entities. */
  uint64_t global_count;            /**< Entities on all ranks. */
  SsResultDatasetKind dataset_kind; /**< Dataset layout. */
  const char *quadrature_id;        /**< Quadrature rule, if any. */
  uint32_t sample_count;            /**< Samples per entity. */
} SsResultQuantityDeclaration;

/** Declaration of a category quantity before streaming. */
typedef struct {
  const char *quantity_id;         /**< Quantity identifier. */
  uint32_t quantity_version;       /**< Quantity version. */
  const char *unit;                /**< Unit. */
  const char *location;            /**< Location (node, quadrature point, cell). */
  const char *quadrature_id;       /**< Quadrature rule, if any. */
  const char *const *category_ids; /**< Category names. */
  uint32_t category_count;         /**< Number of categories. */
  const SsEntityId *stable_ids;    /**< Local stable identities. */
  const size_t *local_slots;       /**< Local slot of each identity. */
  size_t local_count;              /**< Local entities. */
  uint64_t global_count;           /**< Entities on all ranks. */
  uint32_t sample_count;           /**< Samples per entity. */
} SsResultCategoryDeclaration;

/** Scalar value of a quantity at a state. */
typedef struct {
  const char *quantity_id;   /**< Quantity identifier. */
  uint32_t quantity_version; /**< Quantity version. */
  const char *unit;          /**< Unit. */
  SsStateId state_id;        /**< State of the value. */
  double value;              /**< Value. */
  int complete;              /**< Nonzero when final. */
} SsResultScalarValue;

/** Reference from a result to a checkpoint. */
typedef struct {
  SsStateId state_id;        /**< Checkpointed state. */
  const char *checkpoint_id; /**< Checkpoint identity. */
  const char *relative_path; /**< Path inside the result directory. */
  int complete;              /**< Nonzero when complete. */
} SsResultCheckpointReference;

/** One row of the accepted-state index. */
typedef struct {
  SsStateId state_id;            /**< State identity. */
  SsStateId parent_state_id;     /**< Parent state. */
  uint64_t accepted_ordinal;     /**< Ordinal of the state. */
  uint64_t hydraulic_generation; /**< Seepage generation. */
  uint64_t mechanics_generation; /**< Mechanics generation. */
  uint64_t control_generation;   /**< Controller generation. */
  double factor;                 /**< Factor. */
  double work;                   /**< Work. */
  double direct_control;         /**< Direct control. */
  double displacement_maximum;   /**< Largest displacement. */
  uint64_t history_begin;        /**< First history row of the state. */
  uint64_t history_end;          /**< One past its last history row. */
  int complete;                  /**< Nonzero when complete. */
  int accepted;                  /**< Nonzero when accepted. */
} SsResultAcceptedState;

/** Solve role of a history detail row (column `role`). */
typedef enum {
  SS_RESULT_ROLE_NONE = 0,      /**< Not applicable. */
  SS_RESULT_ROLE_PRIMARY = 1,   /**< Primary solve of an attempt. */
  SS_RESULT_ROLE_SECONDARY = 2, /**< Direct-SSR sensitivity solve. */
  SS_RESULT_ROLE_CONTROL = 3,   /**< Control-direction system. */
  SS_RESULT_ROLE_RESIDUAL = 4,  /**< Residual-direction system. */
  SS_RESULT_ROLE_ELASTIC = 5,   /**< Elastic predictor. */
  SS_RESULT_ROLE_SEEPAGE = 6    /**< Seepage solve. */
} SsResultHistoryRole;

/**
 * One relational history row.
 *
 * The first ten columns form the logical-digest projection of result-v1.
 * The detail columns (history row layout v2) record the per-iteration
 * evidence of complete Newton, Krylov, line-search and PMG histories; they are
 * zero when a column does not apply and are protected by the payload SHA-256.
 *
 * Detail meaning by kind: NEWTON rows carry the iteration's main Krylov
 * iteration count (`iterations`), the line-search step (`step`), the residual
 * before the update (`reference`), the relative correction (`auxiliary`) and
 * the line-search outcome (`status`). LINEAR rows carry the solve status and
 * KSP reason, iterations, acceptance threshold (`tolerance`), right-hand-side
 * norm (`reference`), effective relative tolerance (`step`) and the reported
 * Krylov norm (`auxiliary`). LINE_SEARCH rows carry the trial alpha
 * (`control`), the trial residual (`residual`), finiteness/admissibility
 * (`reason`), acceptance (`status`) and the directional value (`auxiliary`).
 * PMG rows carry setup status (`status`), level count (`iterations`) and the
 * apply counters of the solve (`reference` apply calls, `auxiliary` level
 * solves).
 */
typedef struct {
  SsResultHistoryKind kind; /**< Table. */
  uint64_t row_id;          /**< Row identity. */
  SsStateId state_id;       /**< State of the row. */
  uint64_t attempt_id;      /**< Attempt of the row. */
  uint64_t parent_row_id;   /**< Parent row. */
  uint64_t ordinal;         /**< Ordinal within the parent. */
  double control;           /**< Control value. */
  double residual;          /**< Residual. */
  int complete;             /**< Nonzero when complete. */
  int accepted;             /**< Nonzero when accepted. */
  uint32_t role;            /**< ::SsResultHistoryRole of the solve. */
  int32_t status;           /**< Status code. */
  int32_t reason;           /**< Reason code. */
  uint64_t iterations;      /**< Iterations. */
  double tolerance;         /**< Tolerance. */
  double reference;         /**< Reference value. */
  double step;              /**< Step. */
  double auxiliary;         /**< Auxiliary value. */
} SsResultHistoryRow;

/** State and size of a writer, identical on all ranks. */
typedef struct {
  SsResultWriterState writer_state;            /**< Writer state. */
  SsResultTerminalStatus terminal_status;      /**< Terminal status. */
  uint64_t field_count;                        /**< Quantity datasets. */
  uint64_t accepted_state_count;               /**< Accepted states. */
  uint64_t history_row_count;                  /**< History rows. */
  uint64_t logical_bytes;                      /**< Bytes of logical data. */
  uint64_t payload_bytes;                      /**< Size of `result.h5`. */
  uint64_t maximum_rank_bytes;                 /**< Most bytes written by a rank. */
  uint64_t minimum_rank_bytes;                 /**< Fewest bytes written by a rank. */
  double pack_seconds;                         /**< Time spent packing. */
  double hdf5_seconds;                         /**< Time spent in HDF5. */
  double digest_seconds;                       /**< Time spent hashing. */
  double publication_seconds;                  /**< Time spent publishing. */
  char result_id[SS_SHA256_HEX_CAPACITY + 8U]; /**< Result identity. */
  char payload_sha256[SS_SHA256_HEX_CAPACITY]; /**< SHA-256 of `result.h5`. */
  char logical_digest[SS_SHA256_HEX_CAPACITY + 8U]; /**< Layout-independent digest. */
  char staging_directory[SS_RESULT_PATH_CAPACITY];  /**< Staging directory. */
  char final_directory[SS_RESULT_PATH_CAPACITY];    /**< Published directory. */
} SsResultWriterSummary;

/** Opaque collective result writer; see ss_result_writer_create(). */
typedef struct SsResultWriter SsResultWriter;

/**
 * Scientific outcome serialized in `result.json`.
 *
 * `outcome` is one analysis outcome family of the result contract, for
 * example "converged_criterion", "completed_control_horizon",
 * "completed_requested_step_horizon", "accepted_relaxed", a "failed_*" family
 * or "cancelled".  `validation_policy` names the resolved validation profile
 * the run used. The optional `mechanics_assembly_mode` gives the effective
 * "qualification" or "performance" mode and may be NULL for seepage.
 * Without a call the envelope reports an unspecified outcome.
 */
typedef struct {
  const char *outcome;                 /**< Analysis outcome family. */
  const char *validation_policy;       /**< Resolved validation profile. */
  const char *mechanics_assembly_mode; /**< Effective assembly mode, or NULL. */
} SsResultOutcome;

/** One field of a checkpoint; borrowed for the call. */
typedef struct {
  const SsEntityId *stable_ids;     /**< Local stable identities. */
  const size_t *local_slots;        /**< Local slot of each identity. */
  const double *values;             /**< Values, slot-major. */
  size_t local_count;               /**< Local entities. */
  uint64_t global_count;            /**< Entities on all ranks. */
  uint32_t component_count;         /**< Number of components. */
  const char *quantity_id;          /**< Quantity identifier. */
  const char *unit;                 /**< Unit. */
  const char *space_id;             /**< Field-space identity. */
  const char *basis;                /**< Basis identity. */
  const char *const *component_ids; /**< Component identifiers. */
} SsCheckpointFieldView;

/** Earlier accepted state kept for the continuation controller. */
typedef struct {
  SsStateId state_id;        /**< State identity. */
  SsStateId parent_state_id; /**< Parent state. */
  uint64_t accepted_ordinal; /**< Ordinal. */
  double factor;             /**< Factor. */
  double work;               /**< Work. */
  double direct_control;     /**< Direct control. */
} SsCheckpointPriorState;

/** Deflation basis of a checkpoint; borrowed for the call. */
typedef struct {
  const SsEntityId *stable_free_dof_ids; /**< Stable free unknowns. */
  const size_t *local_slots;             /**< Local slot of each unknown. */
  const double *left_values;     /**< Left basis, column-major over local slots. */
  const double *right_values;    /**< Right basis. */
  const double *source_values;   /**< Compression sources. */
  const double *source_weights;  /**< Compression weights. */
  const double *source_kinds;    /**< Source kinds as values. */
  size_t local_count;            /**< Local unknowns. */
  uint64_t global_count;         /**< Unknowns on all ranks. */
  uint32_t basis_count;          /**< Basis columns. */
  const char *provenance_id;     /**< Basis provenance. */
  const char *operator_identity; /**< Operator identity. */
} SsCheckpointDeflationView;

/** Contents and identities of an accepted-state checkpoint. */
typedef struct {
  const char *final_directory;                /**< Checkpoint directory to publish. */
  const char *problem_id;                     /**< Problem identity. */
  const char *run_id;                         /**< Run identity. */
  const char *build_id;                       /**< Build identity. */
  const char *build_compatibility_id;         /**< Build compatibility identity. */
  const char *mesh_id;                        /**< Mesh identity. */
  const char *geometry_id;                    /**< Geometry identity. */
  const char *field_space_id;                 /**< Field-space identity. */
  const char *numerical_policy_id;            /**< Numerical policy identity. */
  const char *material_model_id;              /**< Material model identity. */
  const char *hydraulic_model_id;             /**< Seepage model identity. */
  const char *coupling_model_id;              /**< Coupling identity. */
  const char *analysis_method;                /**< Analysis kind. */
  const char *controller_id;                  /**< Controller identity. */
  uint32_t field_order;                       /**< Field order. */
  uint32_t geometry_order;                    /**< Geometry order. */
  uint32_t scalar_bits;                       /**< Scalar width in bits. */
  uint32_t source_index_bits;                 /**< PetscInt width of the writer. */
  SsStateId accepted_state_id;                /**< Checkpointed state. */
  const double *controller_values;            /**< Controller state values. */
  size_t controller_value_count;              /**< Controller values. */
  const double *coupling_values;              /**< Coupling state values. */
  size_t coupling_value_count;                /**< Coupling values. */
  const SsCheckpointPriorState *prior_states; /**< Prior accepted states. */
  size_t prior_state_count;                   /**< Prior states. */
  uint64_t history_cursor;                    /**< Next history row. */
  uint64_t accepted_state_cursor;             /**< Next accepted-state row. */
  uint64_t retry_count;                       /**< Consecutive retries. */
  double next_step;                           /**< Next continuation step. */
  uint64_t dataset_chunk_rows;                /**< HDF5 chunk rows. */
  const SsCheckpointFieldView *fields;        /**< Fields. */
  size_t field_count;                         /**< Number of fields. */
  SsCheckpointDeflationView deflation;        /**< Deflation basis. */
  MPI_Comm communicator;                      /**< Communicator of the writers. */
} SsCheckpointSpec;

/** Identities a checkpoint must match to be restored. */
typedef struct {
  const char *problem_id;             /**< Problem identity. */
  const char *build_compatibility_id; /**< Build compatibility identity. */
  const char *mesh_id;                /**< Mesh identity. */
  const char *geometry_id;            /**< Geometry identity. */
  const char *field_space_id;         /**< Field-space identity. */
  const char *numerical_policy_id;    /**< Numerical policy identity. */
  const char *material_model_id;      /**< Material model identity. */
  const char *hydraulic_model_id;     /**< Seepage model identity. */
  const char *coupling_model_id;      /**< Coupling identity. */
  const char *analysis_method;        /**< Analysis kind. */
  const char *controller_id;          /**< Controller identity. */
  const char *operator_identity;      /**< Operator identity. */
  uint32_t field_order;               /**< Field order. */
  uint32_t geometry_order;            /**< Geometry order. */
  uint32_t scalar_bits;               /**< Scalar width in bits. */
  uint32_t target_index_bits;         /**< PetscInt width of the reader. */
} SsCheckpointCompatibility;

/** Destination of one restored field. */
typedef struct {
  const char *quantity_id;      /**< Field to restore. */
  const SsEntityId *stable_ids; /**< Local stable identities. */
  const size_t *local_slots;    /**< Local slot of each identity. */
  double *values;               /**< Restored values. */
  size_t local_count;           /**< Local entities. */
  uint32_t component_count;     /**< Components. */
} SsCheckpointFieldRestoreView;

/** Destinations of a restore, redistributed by stable identity. */
typedef struct {
  const SsCheckpointFieldRestoreView *fields; /**< Field destinations. */
  size_t field_count;                         /**< Number of fields. */
  double *controller_values;                  /**< Controller values. */
  size_t controller_capacity;                 /**< Their capacity. */
  double *coupling_values;                    /**< Coupling values. */
  size_t coupling_capacity;                   /**< Their capacity. */
  SsCheckpointPriorState *prior_states;       /**< Prior states. */
  size_t prior_state_capacity;                /**< Their capacity. */
  const SsEntityId *deflation_stable_ids;     /**< Local free unknowns. */
  const size_t *deflation_local_slots;        /**< Their local slots. */
  size_t deflation_local_count;               /**< Local unknowns. */
  double *deflation_left_values;              /**< Left basis. */
  double *deflation_right_values;             /**< Right basis. */
  double *deflation_source_values;            /**< Compression sources. */
  double *deflation_source_weights;           /**< Compression weights. */
  double *deflation_source_kinds;             /**< Source kinds. */
  uint32_t deflation_basis_capacity;          /**< Basis capacity in columns. */
} SsCheckpointRestoreView;

/** Outcome of a checkpoint write or restore. */
typedef struct {
  SsStateId accepted_state_id;        /**< Checkpointed state. */
  uint64_t restored_field_count;      /**< Fields restored. */
  uint64_t restored_field_values;     /**< Field values restored. */
  uint64_t restored_primary_values;   /**< Primary-field values restored. */
  uint64_t restored_deflation_values; /**< Deflation values restored. */
  uint32_t deflation_basis_count;     /**< Basis columns restored. */
  uint64_t prior_state_count;         /**< Prior states restored. */
  uint64_t history_cursor;            /**< Next history row. */
  uint64_t accepted_state_cursor;     /**< Next accepted-state row. */
  uint64_t retry_count;               /**< Consecutive retries. */
  double next_step;                   /**< Next continuation step. */
  int mutation_started;               /**< Nonzero once outputs were modified. */
  int operator_revalidation_required; /**< Nonzero when the deflation operator must be
                                         refreshed. */
  char checkpoint_id[SS_SHA256_HEX_CAPACITY + 8U]; /**< Checkpoint identity. */
  char operator_identity[SS_RESULT_TEXT_CAPACITY]; /**< Operator identity. */
  char
      source_run_id[SS_SHA256_HEX_CAPACITY + 8U]; /**< Run that wrote the checkpoint. */
  char source_build_id[SS_SHA256_HEX_CAPACITY + 8U]; /**< Build that wrote it. */
  uint32_t source_index_bits; /**< PetscInt width of that build. */
} SsCheckpointRestoreSummary;

/** Order distributed entities by stable identity and assign file rows;
 * collective. No rank gathers the global index.
 * @param[in] spec Local identities and slots.
 * @param[out] result New plan, released with ss_result_file_index_plan_destroy().
 * @return ::SS_OK, ::SS_ERROR_DUPLICATE for a repeated identity,
 *   ::SS_ERROR_INTEGRITY for a count mismatch, or the agreed failure status. */
SsStatus ss_result_file_index_plan_create(const SsResultFileIndexSpec *spec,
                                          SsResultFileIndexPlan **result);
/** File row of one local entity; local.
 * @param[in] plan Plan.
 * @param[in] local_index Local entity index.
 * @param[out] entry Identity, slot and file row.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_result_file_index_plan_entry(const SsResultFileIndexPlan *plan,
                                         size_t local_index,
                                         SsResultFileIndexEntry *entry);
/** Size of a plan; local.
 * @param[in] plan Plan.
 * @param[out] summary Summary.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_result_file_index_plan_summary(const SsResultFileIndexPlan *plan,
                                           SsResultFileIndexSummary *summary);
/** Destroy a plan; collective (frees its communicator).
 * @param[in,out] plan Plan pointer, set to NULL; `*plan` may be NULL.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_result_file_index_plan_destroy(SsResultFileIndexPlan **plan);

/** Create a result writer; collective.
 *
 * Every writer call below is collective on the writer's communicator and
 * returns a status agreed on all ranks. The writer moves through
 * ::SsResultWriterState in order; a call in the wrong state returns
 * ::SS_ERROR_INVALID_STATE.
 *
 * @param[in] spec Identities and layout; strings are copied.
 * @param[out] result New writer, released with ss_result_writer_destroy().
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_result_writer_create(const SsResultWriterSpec *spec,
                                 SsResultWriter **result);
/** Inject a failure at a publication step (test hook); local.
 * @param[in,out] writer Writer.
 * @param[in] point Step that fails.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_result_writer_inject_failure(SsResultWriter *writer,
                                         SsResultFailurePoint point);
/** Create the hidden staging directory, `environment.json` and the parallel
 * HDF5 file.
 * @param[in,out] writer Unopened writer.
 * @return ::SS_OK, ::SS_ERROR_IO, ::SS_ERROR_HDF5 or another agreed status. */
SsStatus ss_result_writer_stage(SsResultWriter *writer);
/** Declare a quantity and create its dataset before streaming.
 * @param[in,out] writer Staged writer.
 * @param[in] declaration Quantity layout.
 * @return ::SS_OK or the agreed failure status. */
SsStatus
ss_result_writer_declare_quantity(SsResultWriter *writer,
                                  const SsResultQuantityDeclaration *declaration);
/** Declare a category quantity and create its dataset before streaming.
 * @param[in,out] writer Staged writer.
 * @param[in] declaration Quantity layout.
 * @return ::SS_OK or the agreed failure status. */
SsStatus
ss_result_writer_declare_category(SsResultWriter *writer,
                                  const SsResultCategoryDeclaration *declaration);
/** Start accepting data.
 * @param[in,out] writer Staged writer.
 * @return ::SS_OK or ::SS_ERROR_INVALID_STATE. */
SsStatus ss_result_writer_begin_stream(SsResultWriter *writer);
/** Write a nodal field at a state.
 * @param[in,out] writer Streaming writer.
 * @param[in] field Values ordered by the file index.
 * @return As ss_result_writer_write_quantity(). */
SsStatus ss_result_writer_write_field(SsResultWriter *writer,
                                      const SsResultFieldView *field);
/** Write the values of a declared quantity at a state.
 * @param[in,out] writer Streaming writer.
 * @param[in] quantity Values and identities.
 * @return ::SS_OK, ::SS_ERROR_DUPLICATE for a repeated state,
 *   ::SS_ERROR_SCHEMA for an undeclared or mismatched quantity, or another
 *   agreed status. */
SsStatus ss_result_writer_write_quantity(SsResultWriter *writer,
                                         const SsResultFieldView *quantity);
/** Write the categories of a declared quantity at a state.
 * @param[in,out] writer Streaming writer.
 * @param[in] category Categories and identities.
 * @return As ss_result_writer_write_quantity(). */
SsStatus ss_result_writer_write_category(SsResultWriter *writer,
                                         const SsResultCategoryView *category);
/** Append a scalar value; identical on every rank.
 * @param[in,out] writer Streaming writer.
 * @param[in] scalar Value.
 * @return ::SS_OK, ::SS_ERROR_DUPLICATE or another agreed status. */
SsStatus ss_result_writer_append_scalar(SsResultWriter *writer,
                                        const SsResultScalarValue *scalar);
/** Record a checkpoint that belongs to the result.
 * @param[in,out] writer Streaming writer.
 * @param[in] checkpoint Checkpoint reference.
 * @return ::SS_OK or the agreed failure status. */
SsStatus
ss_result_writer_index_checkpoint(SsResultWriter *writer,
                                  const SsResultCheckpointReference *checkpoint);
/** Append one history row; identical on every rank.
 * @param[in,out] writer Streaming writer.
 * @param[in] row Row.
 * @return As ss_result_writer_append_history_rows(). */
SsStatus ss_result_writer_append_history(SsResultWriter *writer,
                                         const SsResultHistoryRow *row);
/**
 * Append `count` consecutive rows, identical on every rank.
 *
 * Equivalent to appending them one by one, with a single agreement exchange.
 * Rows are buffered only until the next accepted state closes their window,
 * so writer memory does not grow with the number of accepted states.
 *
 * @param[in,out] writer Streaming writer.
 * @param[in] rows Rows.
 * @param[in] count Number of rows.
 * @return ::SS_OK, ::SS_ERROR_INVALID_STATE or another agreed status.
 */
SsStatus ss_result_writer_append_history_rows(SsResultWriter *writer,
                                              const SsResultHistoryRow *rows,
                                              size_t count);
/** Close an accepted state: flush its history window and index the state.
 * @param[in,out] writer Streaming writer.
 * @param[in] state Accepted-state row.
 * @return ::SS_OK, ::SS_ERROR_DUPLICATE or another agreed status. */
SsStatus ss_result_writer_accept_state(SsResultWriter *writer,
                                       const SsResultAcceptedState *state);
/** Record the analysis outcome and validation policy before publication.
 * @param[in,out] writer Streaming writer.
 * @param[in] outcome Outcome strings; copied.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_result_writer_set_outcome(SsResultWriter *writer,
                                      const SsResultOutcome *outcome);
/** Record the terminal status and stop accepting data.
 * @param[in,out] writer Streaming writer.
 * @param[in] status Terminal status.
 * @param[in] stop_reason Stop reason, for example `minimum_step_reached`.
 * @param[in] error Error of a failed run, or NULL.
 * @return ::SS_OK or ::SS_ERROR_INVALID_STATE. */
SsStatus ss_result_writer_terminalize(SsResultWriter *writer,
                                      SsResultTerminalStatus status,
                                      const char *stop_reason,
                                      const SsErrorRecord *error);
/** Close the file, compute the digests, write `result.json` and rename the
 * staging directory into place atomically.
 * @param[in,out] writer Terminalized writer.
 * @return ::SS_OK, ::SS_ERROR_HDF5, ::SS_ERROR_INTEGRITY or another agreed
 *   status; on failure no result directory appears. */
SsStatus ss_result_writer_publish(SsResultWriter *writer);
/** State and size of a writer; local.
 * @param[in] writer Writer.
 * @param[out] summary Summary.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_result_writer_summary(const SsResultWriter *writer,
                                  SsResultWriterSummary *summary);
/** Destroy a writer and remove an unpublished staging directory; collective.
 * @param[in,out] writer Writer pointer, set to NULL; `*writer` may be NULL.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_INTERNAL. */
SsStatus ss_result_writer_destroy(SsResultWriter **writer);

/** Verify a published result directory: envelope, hashes and logical digest;
 * collective (rank zero reads, the verdict is broadcast).
 * @param[in] communicator Communicator of the readers.
 * @param[in] directory Result directory.
 * @param[out] summary Identities and digests of the result.
 * @return ::SS_OK, ::SS_ERROR_INTEGRITY, ::SS_ERROR_NOT_FOUND, ::SS_ERROR_JSON,
 *   ::SS_ERROR_SCHEMA or another failure, identical on all ranks. */
SsStatus ss_result_verify_terminal(MPI_Comm communicator, const char *directory,
                                   SsResultWriterSummary *summary);

/** Write an accepted-state checkpoint and publish it atomically; collective.
 * @param[in] spec Contents and identities.
 * @param[in] failure_point Publication step that fails (test hook), or
 *   ::SS_RESULT_FAIL_NONE.
 * @param[out] summary Checkpoint identity and counts.
 * @return ::SS_OK, ::SS_ERROR_HDF5 or the agreed failure status. */
SsStatus ss_checkpoint_write_atomic(const SsCheckpointSpec *spec,
                                    SsResultFailurePoint failure_point,
                                    SsCheckpointRestoreSummary *summary);
/** Verify a checkpoint against the expected identities and restore it,
 * redistributing the values by stable identity; collective. The layout may
 * differ from the one that wrote the checkpoint.
 * @param[in] communicator Communicator of the readers.
 * @param[in] directory Checkpoint directory.
 * @param[in] compatibility Identities the checkpoint must match.
 * @param[in] restore Destinations of the values.
 * @param[out] summary Restored counts and controller state.
 * @return ::SS_OK, ::SS_ERROR_INTEGRITY or ::SS_ERROR_VERSION_MISMATCH for an
 *   incompatible checkpoint, or another agreed failure. */
SsStatus ss_checkpoint_restore(MPI_Comm communicator, const char *directory,
                               const SsCheckpointCompatibility *compatibility,
                               const SsCheckpointRestoreView *restore,
                               SsCheckpointRestoreSummary *summary);

/** Protocol name of a terminal status.
 * @param[in] status Terminal status.
 * @return Static name such as `succeeded`. */
const char *ss_result_terminal_status_name(SsResultTerminalStatus status);

#endif
