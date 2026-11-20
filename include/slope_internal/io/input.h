/** @file
 * @brief Strict input of the problem package and the resolved run: JSON
 * parsing, package verification, the streamed compiled manifest and balanced
 * parallel HDF5 reads.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_IO_INPUT_H
#define SS_IO_INPUT_H

#include "slope_internal/core/core.h"
#include "slope_internal/core/solver_options.h"

#include <mpi.h>

#include <stddef.h>
#include <stdint.h>

/** Capacity of a path. */
#define SS_PATH_CAPACITY 512U
/** Capacity of a short identifier or enumeration text. */
#define SS_SHORT_TEXT_CAPACITY 96U
/** Largest number of ranks in a resolved resource map. */
#define SS_RESOURCE_RANK_CAPACITY 4096U

/** Limits applied while parsing an input JSON document. */
typedef struct {
  size_t maximum_bytes;          /**< Largest document size. */
  size_t maximum_depth;          /**< Deepest nesting. */
  size_t maximum_string_bytes;   /**< Longest string. */
  size_t maximum_array_items;    /**< Largest array. */
  size_t maximum_object_members; /**< Largest object. */
  uint64_t maximum_integer;      /**< Largest integer value. */
} SsJsonLimits;

/** Header of the compiled problem (`problem.json`). */
typedef struct {
  char id[72];                           /**< Problem identity `sha256:<hex>`. */
  uint32_t schema_major;                 /**< Schema major version. */
  uint32_t schema_minor;                 /**< Schema minor version. */
  uint32_t schema_patch;                 /**< Schema patch version. */
  uint32_t dimension;                    /**< Spatial dimension, 2 or 3. */
  uint32_t field_order;                  /**< Mechanics field order. */
  uint32_t coordinate_order;             /**< Geometry order. */
  uint32_t uniform_refinement_levels;    /**< Refinements applied at compilation. */
  char analysis[SS_SHORT_TEXT_CAPACITY]; /**< Analysis kind. */
  char operator_class[SS_SHORT_TEXT_CAPACITY]; /**< Operator class. */
  char payload_path[SS_PATH_CAPACITY];         /**< Relative path of `problem.h5`. */
  char payload_sha256[SS_SHA256_HEX_CAPACITY]; /**< SHA-256 of the payload. */
  uint64_t payload_bytes;                      /**< Size of the payload. */
  char layout_id[SS_SHORT_TEXT_CAPACITY];      /**< HDF5 layout identifier. */
} SsProblemSpec;

/** Resolved run (`resolved-run.json`): identities, layout and settings. */
typedef struct {
  char id[72];                                 /**< Run identity. */
  char problem_id[72];                         /**< Problem the run was resolved for. */
  char build_id[72];                           /**< Build the run was resolved for. */
  char executable_id[72];                      /**< Installed executable identity. */
  char dependency_id[72];                      /**< Dependency identity. */
  char analysis[SS_SHORT_TEXT_CAPACITY];       /**< Analysis kind. */
  char operator_class[SS_SHORT_TEXT_CAPACITY]; /**< Operator class. */
  uint32_t schema_major;                       /**< Schema major version of the run. */
  uint32_t problem_schema_major;    /**< Schema major version of the problem. */
  uint32_t dimension;               /**< Spatial dimension. */
  uint32_t field_order;             /**< Mechanics field order. */
  uint32_t coordinate_order;        /**< Geometry order. */
  uint32_t petsc_int_bits;          /**< Required `PetscInt` width. */
  uint32_t world_size;              /**< Required number of MPI ranks. */
  uint32_t openmp_threads;          /**< OpenMP threads per rank. */
  uint32_t maximum_accepted_states; /**< Requested accepted-state horizon. */
  int mpi;                          /**< Nonzero when MPI is required. */
  int openmp;                       /**< Nonzero when OpenMP is required. */
  int parallel_hdf5;                /**< Nonzero when parallel HDF5 is required. */
  int hypre;                        /**< Nonzero when HYPRE is required. */
  int parmetis;                     /**< Nonzero when ParMETIS is required. */
  /** Required MPI thread level, for example `funneled`. */
  char mpi_thread_level[SS_SHORT_TEXT_CAPACITY];
  /** Resolved validation profile ID (`provenance.profile_ids.validation`), or
   *  "unspecified" when the run does not record one. */
  char validation_profile[SS_SHORT_TEXT_CAPACITY];
  SsSolverSettings settings;     /**< Typed stable solver settings. */
  uint32_t local_rank_map_count; /**< Entries in the rank-to-node map. */
  /** World rank of each map entry. */
  uint32_t local_rank_map_rank[SS_RESOURCE_RANK_CAPACITY];
  /** Node index of each map entry. */
  uint32_t local_rank_map_node[SS_RESOURCE_RANK_CAPACITY];
  /** Node-local rank of each map entry. */
  uint32_t local_rank_map_local[SS_RESOURCE_RANK_CAPACITY];
} SsRunSpec;

/** Shared-file identity recorded immediately around verification. Device
 * numbers are local to a host and are not part of it. */
typedef struct {
  uint64_t inode;             /**< Inode number. */
  uint64_t bytes;             /**< File size. */
  uint64_t mtime_seconds;     /**< Modification time, seconds. */
  uint64_t mtime_nanoseconds; /**< Modification time, nanoseconds. */
  uint64_t ctime_seconds;     /**< Status change time, seconds. */
  uint64_t ctime_nanoseconds; /**< Status change time, nanoseconds. */
} SsInputFileIdentity;

/** Compact compiled manifest; see ss_io_read_compiled_manifest(). */
typedef struct SsCompiledManifest SsCompiledManifest;

/** Sealed configuration of one engine operation. */
typedef struct {
  SsProblemSpec problem; /**< Compiled problem header. */
  SsRunSpec run;         /**< Resolved run; valid when @ref has_run is set. */
  int has_run;           /**< Nonzero when a resolved run was parsed. */
  int sealed;            /**< Nonzero once the configuration is complete. */
  SsInputFileIdentity payload_identity;  /**< Stamp of the verified payload. */
  SsCompiledManifest *compiled_manifest; /**< Owned compact manifest, or NULL. */
} SsNativeConfig;

/** Summary of a collective payload validation. */
typedef struct {
  uint64_t coordinate_values; /**< Coordinate values in the payload. */
  uint64_t cell_rows;         /**< Cell rows in the payload. */
  uint32_t zero_owner_ranks;  /**< Ranks that read no cell rows. */
} SsH5Validation;

/** Observed runtime that the resolved run must match. */
typedef struct {
  const char *build_id;      /**< Build identity of the executable. */
  const char *executable_id; /**< Installed executable identity. */
  const char *dependency_id; /**< Dependency identity. */
  uint32_t petsc_int_bits;   /**< `PetscInt` width. */
  uint32_t world_size;       /**< Number of MPI ranks. */
  uint32_t openmp_threads;   /**< OpenMP threads per rank. */
  uint32_t rank;             /**< World rank of the caller. */
  uint32_t local_rank;       /**< Node-local rank of the caller. */
} SsRuntimeIdentity;

/** Default parsing limits: 64 MiB documents, depth 128, 1 Mi strings, items
 * and members.
 * @return The default limits. */
SsJsonLimits ss_json_default_limits(void);
/** Parse and validate `problem.json`; local.
 * @param[in] data Document bytes.
 * @param[in] size Byte count.
 * @param[in] limits Parsing limits.
 * @param[out] problem Problem header.
 * @param[out] error Error record describing a failure.
 * @return ::SS_OK, ::SS_ERROR_SCHEMA or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_io_parse_problem(const unsigned char *data, size_t size,
                             const SsJsonLimits *limits, SsProblemSpec *problem,
                             SsErrorRecord *error);
/** Parse and validate `resolved-run.json`, including its typed settings; local.
 * @param[in] data Document bytes.
 * @param[in] size Byte count.
 * @param[in] limits Parsing limits.
 * @param[out] run Resolved run.
 * @param[out] error Error record describing a failure.
 * @return ::SS_OK, ::SS_ERROR_SCHEMA, ::SS_ERROR_LIMIT or
 *   ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_io_parse_run(const unsigned char *data, size_t size,
                         const SsJsonLimits *limits, SsRunSpec *run,
                         SsErrorRecord *error);

/** Parse a problem and an optional run, check that they match, and seal the
 * configuration; local.
 * @param[in] problem_data `problem.json` bytes.
 * @param[in] problem_size Byte count.
 * @param[in] run_data `resolved-run.json` bytes, or NULL.
 * @param[in] run_size Byte count, or zero.
 * @param[in] limits Parsing limits.
 * @param[in,out] config Unsealed configuration; cleared on failure.
 * @param[out] error Error record describing a failure.
 * @return ::SS_OK, ::SS_ERROR_INTEGRITY when the run belongs to another
 *   problem or contract, ::SS_ERROR_INVALID_STATE for a sealed configuration,
 *   or a parsing status. */
SsStatus ss_io_configure(const unsigned char *problem_data, size_t problem_size,
                         const unsigned char *run_data, size_t run_size,
                         const SsJsonLimits *limits, SsNativeConfig *config,
                         SsErrorRecord *error);
/** Whether a configuration is sealed.
 * @param[in] config Configuration or NULL.
 * @return Nonzero when sealed. */
int ss_io_config_is_sealed(const SsNativeConfig *config);
/** Problem header of a sealed configuration.
 * @param[in] config Configuration or NULL.
 * @return Borrowed header, or NULL when not sealed. */
const SsProblemSpec *ss_io_problem(const SsNativeConfig *config);
/** Release the owned manifest and clear a configuration.
 * @param[in,out] config Configuration; NULL is ignored. */
void ss_io_config_destroy(SsNativeConfig *config);

/** Check that a package path is relative, stays inside the package and names
 * an existing regular file without symbolic links.
 * @param[in] package_root Package directory.
 * @param[in] relative_path Path inside the package.
 * @param[out] error Error record describing a failure.
 * @return ::SS_OK, ::SS_ERROR_INVALID_PATH, ::SS_ERROR_NOT_FOUND or
 *   ::SS_ERROR_IO. */
SsStatus ss_io_validate_package_path(const char *package_root,
                                     const char *relative_path, SsErrorRecord *error);
/** Verify the size and SHA-256 of the payload named by a problem; local.
 * @param[in] package_root Package directory.
 * @param[in] problem Problem header.
 * @param[out] error Error record describing a failure.
 * @return ::SS_OK or a path, I/O or integrity failure. */
SsStatus ss_io_verify_payload(const char *package_root, const SsProblemSpec *problem,
                              SsErrorRecord *error);

/** Load and verify the problem package and the resolved run on all ranks.
 *
 * Rank zero streams the manifest and hashes the payload; the compact manifest
 * and the verdict are broadcast. The run must match the observed runtime.
 * Collective on @p communicator.
 *
 * @param[in] communicator Communicator of the engine.
 * @param[in] problem_directory Package directory.
 * @param[in] resolved_run_path Path of `resolved-run.json`.
 * @param[in] runtime Observed runtime.
 * @param[out] config Sealed configuration, released with
 *   ss_io_config_destroy().
 * @param[out] selected_error Collectively selected error record.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_io_load_pair_collective(MPI_Comm communicator,
                                    const char *problem_directory,
                                    const char *resolved_run_path,
                                    const SsRuntimeIdentity *runtime,
                                    SsNativeConfig *config,
                                    SsErrorRecord *selected_error);

/** Groups of stable-identity lists in the compiled manifest. */
typedef enum {
  SS_MANIFEST_MATERIAL_IDS = 1, /**< Cells of each material region. */
  SS_MANIFEST_BOUNDARY_IDS = 2, /**< Facets of each boundary condition. */
  SS_MANIFEST_SUPPORT_IDS = 3   /**< Entities of each support. */
} SsManifestIdGroup;

/** Location of one identity list inside the compact manifest. */
typedef struct {
  uint64_t group;  /**< ::SsManifestIdGroup of the list. */
  uint64_t record; /**< Record index within the group. */
  uint64_t offset; /**< Byte offset of the list in the manifest file. */
  uint64_t count;  /**< Number of identities. */
} SsManifestIdSpan;

/** Compact manifest held on every rank. */
struct SsCompiledManifest {
  unsigned char *data;                 /**< Compact JSON without the identity lists. */
  size_t size;                         /**< Size of @ref data. */
  SsManifestIdSpan *spans;             /**< Index of the identity lists. */
  size_t span_count;                   /**< Number of spans. */
  char digest[SS_SHA256_HEX_CAPACITY]; /**< Digest of the canonical manifest. */
  int descriptor; /**< Manifest file opened on every rank for the lists, or -1. */
  SsInputFileIdentity identity; /**< Stamp of the manifest file. */
};

/** Read the immutable regular-file stamp of an open descriptor.
 * @param[in] descriptor Open file descriptor.
 * @param[out] identity File stamp.
 * @return ::SS_OK or ::SS_ERROR_INVALID_PATH for a non-regular file. */
SsStatus ss_io_input_identity(int descriptor, SsInputFileIdentity *identity);

/** Check that a file still has the stamp recorded at verification.
 * @param[in] path File path.
 * @param[in] identity Stamp recorded earlier.
 * @return ::SS_OK, ::SS_ERROR_INTEGRITY when the file changed, or
 *   ::SS_ERROR_IO. */
SsStatus ss_io_require_input_identity(const char *path,
                                      const SsInputFileIdentity *identity);

/** Stream the manifest on rank zero and broadcast its compact JSON and the
 * identity-list index.
 *
 * The digest covers the original canonical bytes without the root `id`.
 * Collective on @p communicator. Close the manifest with
 * ss_io_compiled_manifest_destroy() on all ranks, even after a failed read.
 *
 * @param[in] communicator Communicator of the engine.
 * @param[in] path Path of `problem.json`.
 * @param[out] manifest Compact manifest; the file stays open on every rank
 *   for ss_io_read_manifest_ids().
 * @param[out] error Error record describing a failure.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_io_read_compiled_manifest(MPI_Comm communicator, const char *path,
                                      SsCompiledManifest *manifest,
                                      SsErrorRecord *error);

/** Read this rank's balanced share of one identity list; local file read.
 * @param[in] manifest Compact manifest.
 * @param[in] group Group of the list.
 * @param[in] record Record index within the group.
 * @param[in] rank Share to read.
 * @param[in] ranks Number of shares.
 * @param[out] ids Identities, released by the caller with free(); may be NULL
 *   when the share is empty.
 * @param[out] count Number of identities in the share; zero is valid.
 * @param[out] global_count Length of the whole list.
 * @return ::SS_OK, ::SS_ERROR_SCHEMA or ::SS_ERROR_OUT_OF_MEMORY. */
SsStatus ss_io_read_manifest_ids(const SsCompiledManifest *manifest,
                                 SsManifestIdGroup group, size_t record, int rank,
                                 int ranks, SsEntityId **ids, size_t *count,
                                 uint64_t *global_count);

/** Close the descriptor and free the compact manifest and its index.
 * @param[in,out] manifest Manifest; its storage is not freed. */
void ss_io_compiled_manifest_destroy(SsCompiledManifest *manifest);

/** Balanced contiguous slab of a dataset read by one rank. */
typedef struct {
  uint64_t total; /**< Rows in the dataset. */
  size_t start;   /**< First row of this rank. */
  size_t count;   /**< Rows of this rank. */
  size_t width;   /**< Values per row. */
  void *values;   /**< Row-major values, owned by the caller. */
} SsH5Rows;

/** Open a payload read-only with the MPI-IO driver; collective.
 * @param[in] communicator Communicator of the readers.
 * @param[in] path Payload path.
 * @param[out] file HDF5 file handle, closed by the caller with `H5Fclose`.
 * @return ::SS_OK or ::SS_ERROR_HDF5 on all ranks. */
SsStatus ss_io_hdf5_open(MPI_Comm communicator, const char *path, hid_t *file);

/** Read this rank's balanced contiguous row slab with collective I/O.
 * @param[in] communicator Communicator of the readers.
 * @param[in] file Open payload.
 * @param[in] path Dataset path.
 * @param[in] native_type Expected HDF5 native type.
 * @param[in] dimensions Dataset rank, 1 (width 1) or 2.
 * @param[in] expected_width Required row width.
 * @param[out] rows Slab; the caller releases `rows->values` with free().
 * @return ::SS_OK or ::SS_ERROR_HDF5 on all ranks. */
SsStatus ss_io_hdf5_read_rows(MPI_Comm communicator, hid_t file, const char *path,
                              hid_t native_type, int dimensions, size_t expected_width,
                              SsH5Rows *rows);

/** Read the coordinates and cells of a payload collectively and check their
 * shapes, as done by `slope-engine validate`.
 * @param[in] communicator Communicator of the readers.
 * @param[in] path Payload path.
 * @param[out] validation Counts of the payload.
 * @param[out] error Error record describing a failure.
 * @return ::SS_OK, ::SS_ERROR_HDF5, ::SS_ERROR_INVALID_ARGUMENT or
 *   ::SS_ERROR_INTERNAL. */
SsStatus ss_io_validate_hdf5(MPI_Comm communicator, const char *path,
                             SsH5Validation *validation, SsErrorRecord *error);

#endif
