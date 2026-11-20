/* Validate the native run and verify the compiled package on a shared filesystem.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/io/input.h"

#include "slope_internal/build/features.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/diagnostics/diagnostics.h"
#include "slope_internal/protocol/json.h"
#include "yyjson.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static void ss_package_error(SsErrorRecord *error, SsStatus status, const char *code,
                             const char *message, uint32_t rank) {
  ss_error_set(error, status, (SsErrorKey){1U, 3U, 0U, 0U, 0U, rank}, code, message);
}

static int ss_relative_path_valid(const char *path) {
  size_t index;
  if (path == NULL || path[0] == '\0' || path[0] == '/' || strcmp(path, ".") == 0 ||
      strcmp(path, "..") == 0) {
    return 0;
  }
  for (index = 0U; path[index] != '\0'; ++index) {
    if (path[index] == '/' || path[index] == '\\' || path[index] == ':' ||
        (unsigned char)path[index] < 0x20U) {
      return 0;
    }
  }
  return 1;
}

SsStatus ss_io_validate_package_path(const char *package_root,
                                     const char *relative_path, SsErrorRecord *error) {
  struct stat root_stat;
  struct stat file_stat;
  int root_descriptor;
  int file_descriptor;
  if (package_root == NULL || !ss_relative_path_valid(relative_path)) {
    ss_package_error(error, SS_ERROR_INVALID_PATH, "PACKAGE_PATH_INVALID",
                     "package-relative path is not normalized", 0U);
    return SS_ERROR_INVALID_PATH;
  }
  if (lstat(package_root, &root_stat) != 0) {
    SsStatus status = errno == ENOENT ? SS_ERROR_NOT_FOUND : SS_ERROR_INVALID_PATH;
    ss_package_error(error, status, "PACKAGE_ROOT_INVALID",
                     "package root is unavailable", 0U);
    return status;
  }
  if (!S_ISDIR(root_stat.st_mode) || S_ISLNK(root_stat.st_mode)) {
    ss_package_error(error, SS_ERROR_INVALID_PATH, "PACKAGE_ROOT_TYPE",
                     "package root must be a real directory", 0U);
    return SS_ERROR_INVALID_PATH;
  }
  root_descriptor = open(package_root, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (root_descriptor < 0) {
    ss_package_error(error, SS_ERROR_INVALID_PATH, "PACKAGE_ROOT_OPEN",
                     "package root cannot be opened safely", 0U);
    return SS_ERROR_INVALID_PATH;
  }
  file_descriptor =
      openat(root_descriptor, relative_path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (file_descriptor < 0) {
    SsStatus status = errno == ENOENT ? SS_ERROR_NOT_FOUND : SS_ERROR_INVALID_PATH;
    (void)close(root_descriptor);
    ss_package_error(error, status, "PACKAGE_FILE_OPEN",
                     "package file cannot be opened safely", 0U);
    return status;
  }
  if (fstat(file_descriptor, &file_stat) != 0 || !S_ISREG(file_stat.st_mode)) {
    (void)close(file_descriptor);
    (void)close(root_descriptor);
    ss_package_error(error, SS_ERROR_INVALID_PATH, "PACKAGE_FILE_TYPE",
                     "package payload must be a regular file", 0U);
    return SS_ERROR_INVALID_PATH;
  }
  if (close(file_descriptor) != 0 || close(root_descriptor) != 0) {
    ss_package_error(error, SS_ERROR_IO, "PACKAGE_FILE_CLOSE",
                     "package file close failed", 0U);
    return SS_ERROR_IO;
  }
  return SS_OK;
}

static SsStatus ss_check_package_inventory(int root_descriptor, SsErrorRecord *error) {
  int inventory_descriptor = dup(root_descriptor);
  DIR *directory = inventory_descriptor < 0 ? NULL : fdopendir(inventory_descriptor);
  struct dirent *entry;
  if (directory == NULL) {
    if (inventory_descriptor >= 0) {
      (void)close(inventory_descriptor);
    }
    return SS_ERROR_INVALID_PATH;
  }
  errno = 0;
  while ((entry = readdir(directory)) != NULL) {
    const char *name = entry->d_name;
    if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0) {
      continue;
    }
    if (strcmp(name, "problem.json") != 0 && strcmp(name, "problem.h5") != 0 &&
        strcmp(name, "compile-report.json") != 0) {
      (void)closedir(directory);
      ss_package_error(error, SS_ERROR_INTEGRITY, "PACKAGE_UNINDEXED_FILE",
                       "package contains an unindexed file", 0U);
      return SS_ERROR_INTEGRITY;
    }
  }
  if (errno != 0 || closedir(directory) != 0) {
    return SS_ERROR_IO;
  }
  return SS_OK;
}

static SsStatus ss_io_verify_payload_stamp(const char *package_root,
                                           const SsProblemSpec *problem,
                                           SsInputFileIdentity *stamp,
                                           SsErrorRecord *error) {
  char digest[SS_SHA256_HEX_CAPACITY];
  uint64_t byte_count = 0U;
  SsInputFileIdentity before = {0}, after = {0};
  int root_descriptor = -1;
  int file_descriptor = -1;
  SsStatus status;
  if (package_root == NULL || problem == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (!ss_relative_path_valid(problem->payload_path)) {
    ss_package_error(error, SS_ERROR_INVALID_PATH, "PACKAGE_PATH_INVALID",
                     "package-relative path is not normalized", 0U);
    return SS_ERROR_INVALID_PATH;
  }
  root_descriptor = open(package_root, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (root_descriptor < 0) {
    status = errno == ENOENT ? SS_ERROR_NOT_FOUND : SS_ERROR_INVALID_PATH;
    ss_package_error(error, status, "PACKAGE_ROOT_OPEN",
                     "package root cannot be opened safely", 0U);
    return status;
  }
  file_descriptor =
      openat(root_descriptor, problem->payload_path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (file_descriptor < 0) {
    status = errno == ENOENT ? SS_ERROR_NOT_FOUND : SS_ERROR_INVALID_PATH;
    ss_package_error(error, status, "PACKAGE_FILE_OPEN",
                     "package payload cannot be opened safely", 0U);
    (void)close(root_descriptor);
    return status;
  }
  if (ss_io_input_identity(file_descriptor, &before) != SS_OK) {
    ss_package_error(error, SS_ERROR_INVALID_PATH, "PACKAGE_FILE_TYPE",
                     "package payload must be a regular file", 0U);
    (void)close(file_descriptor);
    (void)close(root_descriptor);
    return SS_ERROR_INVALID_PATH;
  }
  status = ss_sha256_fd(file_descriptor, digest, &byte_count);
  if (status == SS_OK && (ss_io_input_identity(file_descriptor, &after) != SS_OK ||
                          memcmp(&before, &after, sizeof(before)) != 0)) {
    status = SS_ERROR_INTEGRITY;
  }
  if (close(file_descriptor) != 0 && status == SS_OK) {
    status = SS_ERROR_IO;
  }
  file_descriptor = -1;
  if (status != SS_OK) {
    ss_package_error(error, status, "PAYLOAD_READ",
                     "payload changed or digest read failed", 0U);
    (void)close(root_descriptor);
    return status;
  }
  if (byte_count != problem->payload_bytes ||
      strcmp(digest, problem->payload_sha256) != 0) {
    ss_package_error(error, SS_ERROR_INTEGRITY, "PAYLOAD_DIGEST",
                     "payload byte count or SHA-256 differs", 0U);
    (void)close(root_descriptor);
    return SS_ERROR_INTEGRITY;
  }
  *stamp = before;
  status = ss_check_package_inventory(root_descriptor, error);
  if (close(root_descriptor) != 0 && status == SS_OK) {
    status = SS_ERROR_IO;
  }
  return status;
}

SsStatus ss_io_verify_payload(const char *package_root, const SsProblemSpec *problem,
                              SsErrorRecord *error) {
  SsInputFileIdentity stamp;
  return ss_io_verify_payload_stamp(package_root, problem, &stamp, error);
}

static SsStatus ss_io_verify_payload_collective(MPI_Comm communicator,
                                                const char *package_root,
                                                SsNativeConfig *config, int rank,
                                                SsErrorRecord *error) {
  char path[SS_PATH_CAPACITY * 2U];
  SsStatus status = SS_OK;
  if (rank == 0) {
    status = ss_io_verify_payload_stamp(package_root, &config->problem,
                                        &config->payload_identity, error);
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    return status;
  }
  if (MPI_Bcast(&config->payload_identity, (int)sizeof(config->payload_identity),
                MPI_BYTE, 0, communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  const int written =
      snprintf(path, sizeof(path), "%s/%s", package_root, config->problem.payload_path);
  status = written < 0 || (size_t)written >= sizeof(path)
               ? SS_ERROR_LIMIT
               : ss_io_require_input_identity(path, &config->payload_identity);
  if (status != SS_OK) {
    ss_package_error(error, status, "PAYLOAD_READ",
                     "rank cannot identify the verified payload file", (uint32_t)rank);
  }
  return status;
}

static SsStatus ss_read_small_file(const char *path, size_t maximum_bytes,
                                   unsigned char **data, size_t *size,
                                   SsErrorRecord *error, uint32_t rank) {
  struct stat metadata;
  struct stat after;
  FILE *stream;
  unsigned char *buffer;
  int descriptor;
  if (path == NULL || data == NULL || size == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  descriptor = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0 || fstat(descriptor, &metadata) != 0 ||
      !S_ISREG(metadata.st_mode) || metadata.st_size < 0) {
    if (descriptor >= 0) {
      (void)close(descriptor);
    }
    ss_package_error(error, SS_ERROR_INVALID_PATH, "MANIFEST_PATH_INVALID",
                     "manifest must be a regular non-symlink file", rank);
    return SS_ERROR_INVALID_PATH;
  }
  if ((uint64_t)metadata.st_size > (uint64_t)maximum_bytes ||
      (uint64_t)metadata.st_size > (uint64_t)SIZE_MAX) {
    ss_package_error(error, SS_ERROR_LIMIT, "MANIFEST_BYTE_LIMIT",
                     "manifest exceeds configured byte limit", rank);
    (void)close(descriptor);
    return SS_ERROR_LIMIT;
  }
  *size = (size_t)metadata.st_size;
  buffer = (unsigned char *)malloc(*size == 0U ? 1U : *size);
  if (buffer == NULL) {
    (void)close(descriptor);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  stream = fdopen(descriptor, "rb");
  if (stream == NULL) {
    (void)close(descriptor);
    free(buffer);
    ss_package_error(error, SS_ERROR_IO, "MANIFEST_READ", "manifest read failed", rank);
    return SS_ERROR_IO;
  }
  {
    int read_failed = *size != 0U && fread(buffer, 1U, *size, stream) != *size;
    int changed = fstat(fileno(stream), &after) != 0 ||
                  metadata.st_dev != after.st_dev || metadata.st_ino != after.st_ino ||
                  metadata.st_size != after.st_size ||
                  metadata.st_mtim.tv_sec != after.st_mtim.tv_sec ||
                  metadata.st_mtim.tv_nsec != after.st_mtim.tv_nsec;
    int close_failed = fclose(stream) != 0;
    if (read_failed == 0 && changed == 0 && close_failed == 0) {
      *data = buffer;
      return SS_OK;
    }
    free(buffer);
    ss_package_error(error, SS_ERROR_IO, "MANIFEST_READ", "manifest read failed", rank);
    return SS_ERROR_IO;
  }
}

static SsStatus ss_content_id(const unsigned char *data, size_t size,
                              const char *expected, SsErrorRecord *error,
                              uint32_t rank) {
  yyjson_doc *document;
  yyjson_mut_doc *mutable_document;
  yyjson_mut_val *root;
  char *serialized;
  size_t serialized_size = 0U;
  unsigned char *full_canonical = NULL;
  size_t full_canonical_size = 0U;
  unsigned char *canonical = NULL;
  size_t canonical_size = 0U;
  SsJsonError json_error = {0};
  char digest[SS_SHA256_HEX_CAPACITY];
  char actual[72];
  SsStatus status = SS_ERROR_INTERNAL;
  if (data == NULL || size == 0U || expected == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (ss_json_canonicalize(data, size, &full_canonical, &full_canonical_size,
                           &json_error) != SS_JSON_OK) {
    return SS_ERROR_JSON;
  }
  if (full_canonical_size != size || memcmp(full_canonical, data, size) != 0) {
    ss_json_free(full_canonical);
    ss_package_error(error, SS_ERROR_INTEGRITY, "MANIFEST_CANONICAL_BYTES",
                     "manifest bytes are not canonical", rank);
    return SS_ERROR_INTEGRITY;
  }
  ss_json_free(full_canonical);
  document = yyjson_read((const char *)data, size, YYJSON_READ_NOFLAG);
  if (document == NULL) {
    return SS_ERROR_JSON;
  }
  mutable_document = yyjson_doc_mut_copy(document, NULL);
  yyjson_doc_free(document);
  if (mutable_document == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  root = yyjson_mut_doc_get_root(mutable_document);
  if (!yyjson_mut_is_obj(root) || yyjson_mut_obj_remove_str(root, "id") == NULL) {
    yyjson_mut_doc_free(mutable_document);
    return SS_ERROR_SCHEMA;
  }
  serialized =
      yyjson_mut_write(mutable_document, YYJSON_WRITE_NOFLAG, &serialized_size);
  yyjson_mut_doc_free(mutable_document);
  if (serialized == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  if (ss_json_canonicalize((const unsigned char *)serialized, serialized_size,
                           &canonical, &canonical_size, &json_error) != SS_JSON_OK) {
    free(serialized);
    return SS_ERROR_JSON;
  }
  free(serialized);
  if (ss_sha256_bytes(canonical, canonical_size, digest) == SS_OK &&
      snprintf(actual, sizeof(actual), "sha256:%s", digest) == 71) {
    status = strcmp(actual, expected) == 0 ? SS_OK : SS_ERROR_INTEGRITY;
  }
  ss_json_free(canonical);
  if (status != SS_OK) {
    ss_package_error(error, status, "MANIFEST_CONTENT_ID",
                     "manifest content identity differs from canonical bytes", rank);
  }
  return status;
}

static int ss_package_name_matches(const char *directory, const char *id) {
  const char *name = strrchr(directory, '/');
  char expected[72];
  size_t index;
  name = name == NULL ? directory : name + 1;
  if (strlen(id) != 71U) {
    return 0;
  }
  memcpy(expected, id, sizeof(expected));
  expected[6] = '-';
  for (index = 0U; index < sizeof(expected); ++index) {
    if (expected[index] == '\0') {
      break;
    }
  }
  return strcmp(name, expected) == 0;
}

static SsStatus ss_compare_run_digests(MPI_Comm communicator,
                                       const unsigned char *run_data, size_t run_size,
                                       uint32_t rank, SsErrorRecord *error) {
  char digest[SS_SHA256_HEX_CAPACITY] = {0};
  char root_digest[SS_SHA256_HEX_CAPACITY] = {0};
  SsStatus status = ss_sha256_bytes(run_data, run_size, digest);
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    return status;
  }
  if (rank == 0) {
    memcpy(root_digest, digest, sizeof(root_digest));
  }
  if (MPI_Bcast(root_digest, (int)sizeof(root_digest), MPI_CHAR, 0, communicator) !=
      MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (strcmp(root_digest, digest) != 0) {
    ss_package_error(error, SS_ERROR_INTEGRITY, "RANK_MANIFEST_MISMATCH",
                     "ranks observed different run manifest bytes", rank);
    return SS_ERROR_INTEGRITY;
  }
  return SS_OK;
}

static SsStatus ss_runtime_compatible(const SsNativeConfig *config,
                                      const SsRuntimeIdentity *runtime,
                                      SsErrorRecord *error, uint32_t rank) {
  const SsRunSpec *run = &config->run;
  size_t map_index;
  int local_rank_found = 0;
  if (strcmp(run->build_id, runtime->build_id) != 0 ||
      strcmp(run->executable_id, runtime->executable_id) != 0 ||
      strcmp(run->dependency_id, runtime->dependency_id) != 0 ||
      run->petsc_int_bits != runtime->petsc_int_bits) {
    ss_package_error(error, SS_ERROR_RESOURCE, "BUILD_IDENTITY_MISMATCH",
                     "resolved build identity differs from executable", rank);
    return SS_ERROR_RESOURCE;
  }
  if (run->world_size != runtime->world_size ||
      run->openmp_threads != runtime->openmp_threads ||
      strcmp(run->mpi_thread_level, "funneled") != 0) {
    ss_package_error(error, SS_ERROR_RESOURCE, "RESOURCE_LAYOUT_MISMATCH",
                     "launched ranks or threads differ from resolved run", rank);
    return SS_ERROR_RESOURCE;
  }
  if (run->mpi == 0 || run->parallel_hdf5 == 0 || run->hypre == 0 ||
      run->parmetis == 0 || run->openmp != ss_openmp_enabled() ||
      (run->openmp_threads > 1U && run->openmp == 0)) {
    ss_package_error(error, SS_ERROR_RESOURCE, "BUILD_FEATURE_MISMATCH",
                     "resolved run requires unavailable build features", rank);
    return SS_ERROR_RESOURCE;
  }
  for (map_index = 0U; map_index < run->local_rank_map_count; ++map_index) {
    if (run->local_rank_map_rank[map_index] == runtime->rank) {
      local_rank_found =
          run->local_rank_map_local[map_index] == runtime->local_rank ? 1 : 0;
      break;
    }
  }
  if (run->local_rank_map_count != run->world_size || local_rank_found == 0) {
    ss_package_error(error, SS_ERROR_INTEGRITY, "RANK_LOCAL_SETUP",
                     "this rank has no consistent resolved local-rank row", rank);
    return SS_ERROR_INTEGRITY;
  }
  return SS_OK;
}

SsStatus ss_io_load_pair_collective(MPI_Comm communicator,
                                    const char *problem_directory,
                                    const char *resolved_run_path,
                                    const SsRuntimeIdentity *runtime,
                                    SsNativeConfig *config,
                                    SsErrorRecord *selected_error) {
  SsJsonLimits limits = ss_json_default_limits();
  SsCompiledManifest *manifest = NULL;
  unsigned char *run_data = NULL;
  size_t run_size = 0U;
  char problem_path[SS_PATH_CAPACITY * 2U];
  int rank = -1;
  SsStatus status = SS_OK;
  SsErrorRecord local_error = {0};
  if (problem_directory == NULL || resolved_run_path == NULL || runtime == NULL ||
      config == NULL || selected_error == NULL ||
      MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  {
    int written = snprintf(problem_path, sizeof(problem_path), "%s/problem.json",
                           problem_directory);
    if (written < 0 || (size_t)written >= sizeof(problem_path)) {
      return SS_ERROR_LIMIT;
    }
  }
  manifest = calloc(1U, sizeof(*manifest));
  if (manifest == NULL) {
    status = SS_ERROR_OUT_OF_MEMORY;
  } else {
    manifest->descriptor = -1;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    status = ss_io_read_compiled_manifest(communicator, problem_path, manifest,
                                          &local_error);
  }
  if (status == SS_OK) {
    status = ss_read_small_file(resolved_run_path, limits.maximum_bytes, &run_data,
                                &run_size, &local_error, (uint32_t)rank);
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    status = ss_compare_run_digests(communicator, run_data, run_size, (uint32_t)rank,
                                    &local_error);
  }
  if (status == SS_OK) {
    status = ss_io_configure(manifest->data, manifest->size, run_data, run_size,
                             &limits, config, &local_error);
  }
  if (status == SS_OK && strcmp(config->problem.id + 7, manifest->digest) != 0) {
    status = SS_ERROR_INTEGRITY;
    ss_package_error(&local_error, status, "MANIFEST_CONTENT_ID",
                     "manifest content identity differs from canonical bytes",
                     (uint32_t)rank);
  }
  if (status == SS_OK) {
    status =
        ss_content_id(run_data, run_size, config->run.id, &local_error, (uint32_t)rank);
  }
  if (status == SS_OK &&
      !ss_package_name_matches(problem_directory, config->problem.id)) {
    ss_package_error(&local_error, SS_ERROR_INTEGRITY, "PACKAGE_CONTENT_ID",
                     "problem directory name differs from problem identity",
                     (uint32_t)rank);
    status = SS_ERROR_INTEGRITY;
  }
  /* Select preceding errors before entering the collective payload hash. */
  if (status != SS_OK && local_error.valid == 0) {
    ss_package_error(&local_error, status, "NATIVE_PREFLIGHT",
                     "native preflight failed", (uint32_t)rank);
  }
  if (ss_status_allreduce_max(communicator, status) == SS_OK) {
    status = ss_io_verify_payload_collective(communicator, problem_directory, config,
                                             rank, &local_error);
  }
  if (status == SS_OK) {
    status = ss_runtime_compatible(config, runtime, &local_error, (uint32_t)rank);
  }
  if (status != SS_OK && local_error.valid == 0) {
    ss_package_error(&local_error, status, "NATIVE_PREFLIGHT",
                     "native preflight failed", (uint32_t)rank);
  }
  if (ss_collective_error_select(communicator, &local_error, selected_error) != SS_OK) {
    status = SS_ERROR_INTERNAL;
  } else if (selected_error->valid != 0) {
    status = selected_error->status;
  }
  free(run_data);
  if (status == SS_OK) {
    config->compiled_manifest = manifest;
  } else if (manifest != NULL) {
    ss_io_compiled_manifest_destroy(manifest);
    free(manifest);
  }
  return status;
}
