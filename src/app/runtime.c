/* Install identity from the installed build manifest, result directory
 * preparation and atomic publication of content-addressed JSON.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/app/runtime.h"
#include "slope_internal/core/util.h"

#include "slope_internal/build/features.h"
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

static void ss_runtime_error(SsErrorRecord *error, SsStatus status, const char *code,
                             const char *message) {
  ss_error_set(error, status, (SsErrorKey){4U, 1U, 0U, 0U, 0U, 0U}, code, message);
}

static SsStatus ss_read_file(const char *path, unsigned char **data, size_t *size) {
  struct stat metadata;
  FILE *stream;
  unsigned char *buffer;
  int descriptor = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0 || fstat(descriptor, &metadata) != 0 ||
      !S_ISREG(metadata.st_mode) || metadata.st_size < 0 ||
      metadata.st_size > 1024 * 1024) {
    if (descriptor >= 0) {
      (void)close(descriptor);
    }
    return SS_ERROR_NOT_FOUND;
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
    return SS_ERROR_IO;
  }
  {
    int read_failed = *size != 0U && fread(buffer, 1U, *size, stream) != *size;
    int close_failed = fclose(stream) != 0;
    if (read_failed != 0 || close_failed != 0) {
      free(buffer);
      return SS_ERROR_IO;
    }
  }
  *data = buffer;
  return SS_OK;
}

static SsStatus ss_executable_path(const char *argv0, char *path, size_t capacity) {
  char *resolved;
  ssize_t size;
  size_t resolved_size;
  if (path == NULL || capacity < 2U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  size = readlink("/proc/self/exe", path, capacity - 1U);
  if (size >= 0 && (size_t)size < capacity) {
    path[size] = '\0';
    return SS_OK;
  }
  if (argv0 == NULL) {
    return SS_ERROR_NOT_FOUND;
  }
  resolved = realpath(argv0, NULL);
  if (resolved == NULL) {
    return SS_ERROR_NOT_FOUND;
  }
  resolved_size = strlen(resolved);
  if (resolved_size >= capacity) {
    free(resolved);
    return SS_ERROR_LIMIT;
  }
  memcpy(path, resolved, resolved_size + 1U);
  free(resolved);
  return SS_OK;
}

static SsStatus ss_manifest_path(const char *executable, char *path, size_t capacity) {
  char prefix[512];
  char *slash;
  int written;
  if (strlen(executable) >= sizeof(prefix)) {
    return SS_ERROR_LIMIT;
  }
  (void)snprintf(prefix, sizeof(prefix), "%s", executable);
  slash = strrchr(prefix, '/');
  if (slash == NULL) {
    return SS_ERROR_NOT_FOUND;
  }
  *slash = '\0';
  slash = strrchr(prefix, '/');
  if (slash == NULL) {
    return SS_ERROR_NOT_FOUND;
  }
  *slash = '\0';
  written = snprintf(path, capacity, "%s/share/slope-stability/install-manifest.json",
                     prefix);
  return written >= 0 && (size_t)written < capacity ? SS_OK : SS_ERROR_LIMIT;
}

static SsStatus ss_copy_json_string(yyjson_val *object, const char *name,
                                    char *destination, size_t capacity) {
  yyjson_val *value = yyjson_obj_get(object, name);
  size_t size;
  if (!yyjson_is_str(value)) {
    return SS_ERROR_SCHEMA;
  }
  size = yyjson_get_len(value);
  if (size >= capacity) {
    return SS_ERROR_LIMIT;
  }
  memcpy(destination, yyjson_get_str(value), size);
  destination[size] = '\0';
  return SS_OK;
}

static int ss_install_manifest_shape(yyjson_val *root) {
  yyjson_val *version;
  return yyjson_is_obj(root) && yyjson_obj_size(root) == 7U &&
         yyjson_equals_str(yyjson_obj_get(root, "kind"), "slope_stability.install") &&
         yyjson_is_str(yyjson_obj_get(root, "build_id")) &&
         yyjson_is_str(yyjson_obj_get(root, "engine_sha256")) &&
         yyjson_is_str(yyjson_obj_get(root, "executable_id")) &&
         yyjson_is_str(yyjson_obj_get(root, "probe_sha256")) &&
         yyjson_is_str(yyjson_obj_get(root, "schema_set_sha256")) &&
         yyjson_is_obj((version = yyjson_obj_get(root, "schema_version"))) &&
         yyjson_obj_size(version) == 3U &&
         yyjson_is_uint(yyjson_obj_get(version, "major")) &&
         yyjson_is_uint(yyjson_obj_get(version, "minor")) &&
         yyjson_is_uint(yyjson_obj_get(version, "patch")) &&
         yyjson_get_uint(yyjson_obj_get(version, "major")) == 1U;
}

SsStatus ss_install_identity_load(const char *argv0, SsInstallIdentity *identity,
                                  SsErrorRecord *error) {
  unsigned char *manifest_data = NULL;
  size_t manifest_size = 0U;
  yyjson_doc *document = NULL;
  yyjson_val *root;
  char actual_sha256[SS_SHA256_HEX_CAPACITY];
  char actual_probe_sha256[SS_SHA256_HEX_CAPACITY];
  char expected_executable_id[72];
  char executable_projection[320];
  char probe_path[512];
  char probe_sha256[SS_SHA256_HEX_CAPACITY];
  char build_id[72];
  uint64_t executable_bytes = 0U;
  uint64_t probe_bytes = 0U;
  char schema_digest[SS_SHA256_HEX_CAPACITY];
  SsStatus status;
  if (identity == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(identity, 0, sizeof(*identity));
  status = ss_executable_path(argv0, identity->executable_path,
                              sizeof(identity->executable_path));
  if (status == SS_OK) {
    status = ss_manifest_path(identity->executable_path, identity->manifest_path,
                              sizeof(identity->manifest_path));
  }
  if (status == SS_OK) {
    status = ss_read_file(identity->manifest_path, &manifest_data, &manifest_size);
  }
  if (status != SS_OK) {
    ss_runtime_error(error, status, "INSTALL_MANIFEST_MISSING",
                     "installed executable identity is unavailable");
    return status;
  }
  document =
      yyjson_read((const char *)manifest_data, manifest_size, YYJSON_READ_NOFLAG);
  free(manifest_data);
  if (document == NULL || !ss_install_manifest_shape(yyjson_doc_get_root(document))) {
    yyjson_doc_free(document);
    ss_runtime_error(error, SS_ERROR_SCHEMA, "INSTALL_MANIFEST_INVALID",
                     "install manifest is invalid");
    return SS_ERROR_SCHEMA;
  }
  root = yyjson_doc_get_root(document);
  status = ss_copy_json_string(root, "build_id", build_id, sizeof(build_id));
  if (status == SS_OK) {
    status = ss_copy_json_string(root, "executable_id", identity->executable_id,
                                 sizeof(identity->executable_id));
  }
  if (status == SS_OK) {
    status = ss_copy_json_string(root, "engine_sha256", identity->engine_sha256,
                                 sizeof(identity->engine_sha256));
  }
  if (status == SS_OK) {
    status =
        ss_copy_json_string(root, "probe_sha256", probe_sha256, sizeof(probe_sha256));
  }
  if (status == SS_OK) {
    status = ss_copy_json_string(root, "schema_set_sha256", schema_digest,
                                 sizeof(schema_digest));
  }
  yyjson_doc_free(document);
  if (status != SS_OK || strcmp(build_id, ss_build_id()) != 0 ||
      !ss_valid_content_id(build_id) || !ss_valid_content_id(identity->executable_id) ||
      !ss_valid_lower_hex(identity->engine_sha256, 64U) ||
      !ss_valid_lower_hex(probe_sha256, 64U) ||
      !ss_valid_lower_hex(schema_digest, 64U) ||
      snprintf(identity->schema_id, sizeof(identity->schema_id), "sha256:%s",
               schema_digest) != 71) {
    ss_runtime_error(error, SS_ERROR_SCHEMA, "INSTALL_MANIFEST_FIELDS",
                     "install manifest identity fields are invalid");
    return SS_ERROR_SCHEMA;
  }
  status = ss_sha256_file(identity->executable_path, actual_sha256, &executable_bytes);
  (void)executable_bytes;
  if (status != SS_OK || strcmp(actual_sha256, identity->engine_sha256) != 0) {
    ss_runtime_error(error, SS_ERROR_INTEGRITY, "EXECUTABLE_DIGEST_MISMATCH",
                     "running engine differs from install manifest");
    return SS_ERROR_INTEGRITY;
  }
  {
    char *slash;
    int written;
    if (strlen(identity->executable_path) >= sizeof(probe_path)) {
      status = SS_ERROR_LIMIT;
    } else {
      (void)snprintf(probe_path, sizeof(probe_path), "%s", identity->executable_path);
      slash = strrchr(probe_path, '/');
      if (slash == NULL) {
        status = SS_ERROR_NOT_FOUND;
      } else {
        written =
            snprintf(slash + 1U, sizeof(probe_path) - (size_t)(slash + 1U - probe_path),
                     "%s", "slope-probe");
        status = written >= 0 && (size_t)written < sizeof(probe_path) -
                                                       (size_t)(slash + 1U - probe_path)
                     ? SS_OK
                     : SS_ERROR_LIMIT;
      }
    }
  }
  if (status == SS_OK) {
    status = ss_sha256_file(probe_path, actual_probe_sha256, &probe_bytes);
  }
  (void)probe_bytes;
  if (status != SS_OK || strcmp(actual_probe_sha256, probe_sha256) != 0) {
    ss_runtime_error(error, SS_ERROR_INTEGRITY, "PROBE_DIGEST_MISMATCH",
                     "installed probe differs from install manifest");
    return SS_ERROR_INTEGRITY;
  }
  if (snprintf(executable_projection, sizeof(executable_projection),
               "engine:%s\nprobe:%s\nschemas:%s\n", actual_sha256, actual_probe_sha256,
               schema_digest) < 0 ||
      ss_sha256_bytes((const unsigned char *)executable_projection,
                      strlen(executable_projection),
                      expected_executable_id + 7U) != SS_OK) {
    ss_runtime_error(error, SS_ERROR_INTERNAL, "INSTALL_IDENTITY_BUILD",
                     "installed executable identity cannot be reconstructed");
    return SS_ERROR_INTERNAL;
  }
  memcpy(expected_executable_id, "sha256:", 7U);
  if (strcmp(expected_executable_id, identity->executable_id) != 0) {
    ss_runtime_error(error, SS_ERROR_INTEGRITY, "EXECUTABLE_ID_MISMATCH",
                     "installed executable identity differs from its artifacts");
    return SS_ERROR_INTEGRITY;
  }
  return SS_OK;
}

SsStatus ss_prepare_result_directory(const char *path, SsErrorRecord *error) {
  struct stat metadata;
  DIR *directory;
  struct dirent *entry;
  if (path == NULL || path[0] == '\0') {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (lstat(path, &metadata) != 0) {
    if (errno != ENOENT || mkdir(path, 0700) != 0) {
      ss_runtime_error(error, SS_ERROR_IO, "RESULT_DIRECTORY_CREATE",
                       "result directory cannot be created");
      return SS_ERROR_IO;
    }
  } else if (!S_ISDIR(metadata.st_mode) || S_ISLNK(metadata.st_mode)) {
    ss_runtime_error(error, SS_ERROR_INVALID_PATH, "RESULT_DIRECTORY_TYPE",
                     "result location must be a non-symlink directory");
    return SS_ERROR_INVALID_PATH;
  }
  directory = opendir(path);
  if (directory == NULL) {
    return SS_ERROR_IO;
  }
  errno = 0;
  while ((entry = readdir(directory)) != NULL) {
    if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
      (void)closedir(directory);
      ss_runtime_error(error, SS_ERROR_INTEGRITY, "RESULT_DIRECTORY_NOT_EMPTY",
                       "result directory must be empty");
      return SS_ERROR_INTEGRITY;
    }
  }
  if (errno != 0 || closedir(directory) != 0) {
    return SS_ERROR_IO;
  }
  return SS_OK;
}

static SsStatus ss_add_content_id(const unsigned char *canonical, size_t canonical_size,
                                  const char *id, unsigned char **output,
                                  size_t *output_size) {
  yyjson_doc *document;
  yyjson_mut_doc *mutable_document;
  yyjson_mut_val *root;
  char *serialized;
  size_t serialized_size = 0U;
  SsJsonError json_error = {0};
  document = yyjson_read((const char *)canonical, canonical_size, YYJSON_READ_NOFLAG);
  if (document == NULL) {
    return SS_ERROR_JSON;
  }
  mutable_document = yyjson_doc_mut_copy(document, NULL);
  yyjson_doc_free(document);
  if (mutable_document == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  root = yyjson_mut_doc_get_root(mutable_document);
  if (!yyjson_mut_is_obj(root) ||
      !yyjson_mut_obj_add_str(mutable_document, root, "id", id)) {
    yyjson_mut_doc_free(mutable_document);
    return SS_ERROR_SCHEMA;
  }
  serialized =
      yyjson_mut_write(mutable_document, YYJSON_WRITE_NOFLAG, &serialized_size);
  yyjson_mut_doc_free(mutable_document);
  if (serialized == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  if (ss_json_canonicalize((const unsigned char *)serialized, serialized_size, output,
                           output_size, &json_error) != SS_JSON_OK) {
    free(serialized);
    return SS_ERROR_JSON;
  }
  free(serialized);
  return SS_OK;
}

static SsStatus ss_atomic_write(const char *directory, const char *filename,
                                const unsigned char *data, size_t size) {
  char temporary[1024];
  char target[1024];
  int descriptor;
  size_t written_total = 0U;
  int directory_descriptor;
  int written;
  written = snprintf(temporary, sizeof(temporary), "%s/.%s.tmp-%ld", directory,
                     filename, (long)getpid());
  if (written < 0 || (size_t)written >= sizeof(temporary)) {
    return SS_ERROR_LIMIT;
  }
  written = snprintf(target, sizeof(target), "%s/%s", directory, filename);
  if (written < 0 || (size_t)written >= sizeof(target)) {
    return SS_ERROR_LIMIT;
  }
  descriptor = open(temporary, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (descriptor < 0) {
    return SS_ERROR_IO;
  }
  while (written_total < size) {
    ssize_t count = write(descriptor, data + written_total, size - written_total);
    if (count <= 0) {
      (void)close(descriptor);
      (void)unlink(temporary);
      return SS_ERROR_IO;
    }
    written_total += (size_t)count;
  }
  {
    int sync_failed = fsync(descriptor) != 0;
    int close_failed = close(descriptor) != 0;
    if (sync_failed != 0 || close_failed != 0 || link(temporary, target) != 0 ||
        unlink(temporary) != 0) {
      (void)unlink(temporary);
      return SS_ERROR_IO;
    }
  }
  directory_descriptor =
      open(directory, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (directory_descriptor < 0) {
    (void)unlink(temporary);
    return SS_ERROR_IO;
  }
  {
    int sync_failed = fsync(directory_descriptor) != 0;
    int close_failed = close(directory_descriptor) != 0;
    if (sync_failed != 0 || close_failed != 0) {
      return SS_ERROR_IO;
    }
  }
  return SS_OK;
}

SsStatus ss_publish_content_json(const char *directory, const char *filename,
                                 const char *body_without_id, char output_id[72],
                                 SsErrorRecord *error) {
  unsigned char *canonical_body = NULL;
  size_t canonical_body_size = 0U;
  unsigned char *canonical_document = NULL;
  size_t canonical_document_size = 0U;
  SsJsonError json_error = {0};
  char digest[SS_SHA256_HEX_CAPACITY];
  SsStatus status;
  if (directory == NULL || filename == NULL || body_without_id == NULL ||
      output_id == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (ss_json_canonicalize((const unsigned char *)body_without_id,
                           strlen(body_without_id), &canonical_body,
                           &canonical_body_size, &json_error) != SS_JSON_OK ||
      ss_sha256_bytes(canonical_body, canonical_body_size, digest) != SS_OK ||
      snprintf(output_id, 72U, "sha256:%s", digest) != 71) {
    ss_json_free(canonical_body);
    ss_runtime_error(error, SS_ERROR_JSON, "RESULT_JSON_INVALID",
                     "result JSON construction failed");
    return SS_ERROR_JSON;
  }
  status = ss_add_content_id(canonical_body, canonical_body_size, output_id,
                             &canonical_document, &canonical_document_size);
  ss_json_free(canonical_body);
  if (status == SS_OK) {
    status = ss_atomic_write(directory, filename, canonical_document,
                             canonical_document_size);
  }
  ss_json_free(canonical_document);
  if (status != SS_OK) {
    ss_runtime_error(error, status, "RESULT_PUBLICATION_FAILED",
                     "atomic result publication failed");
  }
  return status;
}
