/* Stream the v1 JSON and index the mesh-sized stable-ID arrays.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/io/input.h"

#include "slope_internal/core/collective.h"
#include "slope_internal/core/util.h"
#include "slope_internal/protocol/json.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum { SS_MANIFEST_ID_BYTES = 19, SS_MANIFEST_READ_BYTES = 65536 };

typedef struct {
  FILE *stream;
  const SsJsonLimits *limits;
  SsCompiledManifest *manifest;
  int next;
  uint64_t position;
  uint64_t omit_begin;
  uint64_t omit_end;
  size_t capacity;
  size_t span_capacity;
} SsManifestScanner;

SsStatus ss_io_input_identity(int descriptor, SsInputFileIdentity *identity) {
  struct stat info;
  if (descriptor < 0 || fstat(descriptor, &info) != 0 || !S_ISREG(info.st_mode) ||
      info.st_size < 0) {
    return SS_ERROR_INVALID_PATH;
  }
  *identity = (SsInputFileIdentity){
      (uint64_t)info.st_ino,         (uint64_t)info.st_size,
      (uint64_t)info.st_mtim.tv_sec, (uint64_t)info.st_mtim.tv_nsec,
      (uint64_t)info.st_ctim.tv_sec, (uint64_t)info.st_ctim.tv_nsec};
  return SS_OK;
}

SsStatus ss_io_require_input_identity(const char *path,
                                      const SsInputFileIdentity *identity) {
  const int descriptor = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  SsInputFileIdentity observed = {0};
  SsStatus status = ss_io_input_identity(descriptor, &observed);
  if (status == SS_OK && memcmp(identity, &observed, sizeof(observed)) != 0) {
    status = SS_ERROR_INTEGRITY;
  }
  if (descriptor >= 0 && close(descriptor) != 0 && status == SS_OK) {
    status = SS_ERROR_IO;
  }
  return status;
}

static SsStatus ss_manifest_take(SsManifestScanner *scanner, int keep) {
  SsCompiledManifest *manifest = scanner->manifest;
  if (scanner->next == EOF) {
    return ferror(scanner->stream) != 0 ? SS_ERROR_IO : SS_ERROR_JSON;
  }
  if (scanner->position >= manifest->identity.bytes) {
    return SS_ERROR_INTEGRITY;
  }
  if (keep != 0) {
    if (manifest->size == scanner->capacity) {
      const size_t next_capacity =
          scanner->capacity == 0U ? 2048U : scanner->capacity * 2U;
      unsigned char *grown = realloc(manifest->data, next_capacity);
      if (grown == NULL) {
        return SS_ERROR_OUT_OF_MEMORY;
      }
      manifest->data = grown;
      scanner->capacity = next_capacity;
    }
    manifest->data[manifest->size++] = (unsigned char)scanner->next;
  }
  ++scanner->position;
  scanner->next = getc(scanner->stream);
  return SS_OK;
}

static SsStatus ss_manifest_expect(SsManifestScanner *scanner, int expected, int keep) {
  if (scanner->next != expected) {
    return SS_ERROR_INTEGRITY;
  }
  return ss_manifest_take(scanner, keep);
}

static SsStatus ss_manifest_string(SsManifestScanner *scanner) {
  SsStatus status = ss_manifest_expect(scanner, '"', 1);
  while (status == SS_OK && scanner->next != '"') {
    const int escape = scanner->next == '\\';
    status = ss_manifest_take(scanner, 1);
    if (status == SS_OK && escape != 0) {
      status = ss_manifest_take(scanner, 1);
    }
  }
  return status == SS_OK ? ss_manifest_take(scanner, 1) : status;
}

static int ss_manifest_hex(int byte) {
  if (byte >= '0' && byte <= '9') {
    return byte - '0';
  }
  return byte >= 'a' && byte <= 'f' ? byte - 'a' + 10 : -1;
}

static SsStatus ss_manifest_add_span(SsManifestScanner *scanner,
                                     SsManifestIdSpan span) {
  SsCompiledManifest *manifest = scanner->manifest;
  if (manifest->span_count == scanner->span_capacity) {
    const size_t capacity =
        scanner->span_capacity == 0U ? 16U : 2U * scanner->span_capacity;
    SsManifestIdSpan *grown = realloc(manifest->spans, capacity * sizeof(*grown));
    if (grown == NULL) {
      return SS_ERROR_OUT_OF_MEMORY;
    }
    manifest->spans = grown;
    scanner->span_capacity = capacity;
  }
  manifest->spans[manifest->span_count++] = span;
  return SS_OK;
}

static SsStatus ss_manifest_id_span(SsManifestScanner *scanner, SsManifestIdGroup group,
                                    size_t record) {
  SsManifestIdSpan span = {(uint64_t)group, record, scanner->position + 1U, 0U};
  SsEntityId previous = 0U;
  SsStatus status = ss_manifest_expect(scanner, '[', 1);
  while (status == SS_OK && scanner->next != ']') {
    SsEntityId id = 0U;
    status = ss_manifest_expect(scanner, '"', 0);
    for (size_t i = 0U; status == SS_OK && i < 16U; ++i) {
      const int digit = ss_manifest_hex(scanner->next);
      if (digit < 0) {
        status = SS_ERROR_SCHEMA;
      } else {
        id = (id << 4U) | (uint64_t)digit;
        status = ss_manifest_take(scanner, 0);
      }
    }
    if (status == SS_OK) {
      status = ss_manifest_expect(scanner, '"', 0);
    }
    if (status != SS_OK) {
      return status;
    }
    if (span.count != 0U && id <= previous) {
      return SS_ERROR_SCHEMA;
    }
    previous = id;
    if (++span.count > scanner->limits->maximum_array_items) {
      return SS_ERROR_LIMIT;
    }
    if (scanner->next != ']') {
      status = ss_manifest_expect(scanner, ',', 0);
      if (status == SS_OK && scanner->next != '"') {
        return SS_ERROR_SCHEMA;
      }
    }
  }
  if (status == SS_OK) {
    status = ss_manifest_expect(scanner, ']', 1);
  }
  return status == SS_OK ? ss_manifest_add_span(scanner, span) : status;
}

static int ss_manifest_key_is(const SsCompiledManifest *manifest, size_t start,
                              size_t length, const char *quoted_name) {
  return strlen(quoted_name) == length &&
         memcmp(manifest->data + start, quoted_name, length) == 0;
}

static SsManifestIdGroup ss_manifest_group(const SsCompiledManifest *manifest,
                                           size_t start, size_t length) {
  if (ss_manifest_key_is(manifest, start, length, "\"materials\"")) {
    return SS_MANIFEST_MATERIAL_IDS;
  }
  if (ss_manifest_key_is(manifest, start, length, "\"boundary_rules\"")) {
    return SS_MANIFEST_BOUNDARY_IDS;
  }
  if (ss_manifest_key_is(manifest, start, length, "\"supports\"")) {
    return SS_MANIFEST_SUPPORT_IDS;
  }
  return 0;
}

static SsStatus ss_manifest_value(SsManifestScanner *scanner, size_t depth,
                                  SsManifestIdGroup group, size_t record);

static SsStatus ss_manifest_object(SsManifestScanner *scanner, size_t depth,
                                   SsManifestIdGroup group, size_t record) {
  const SsCompiledManifest *manifest = scanner->manifest;
  size_t member = 0U;
  SsStatus status = ss_manifest_take(scanner, 1);
  while (status == SS_OK && scanner->next != '}') {
    const uint64_t raw_start = scanner->position;
    const size_t start = manifest->size;
    status = ss_manifest_string(scanner);
    if (status != SS_OK) {
      break;
    }
    const size_t length = manifest->size - start;
    const int root_id =
        depth == 0U && ss_manifest_key_is(manifest, start, length, "\"id\"");
    const int entity_ids =
        depth == 2U && group != 0 &&
        ss_manifest_key_is(manifest, start, length, "\"entity_ids\"");
    const SsManifestIdGroup child_group =
        depth == 0U ? ss_manifest_group(manifest, start, length) : group;
    status = ss_manifest_expect(scanner, ':', 1);
    if (status == SS_OK) {
      status = entity_ids != 0 ? ss_manifest_id_span(scanner, group, record)
                               : ss_manifest_value(scanner, depth + 1U, child_group,
                                                   depth == 1U ? member : record);
    }
    if (root_id != 0) {
      scanner->omit_begin =
          scanner->next == '}' && member != 0U ? raw_start - 1U : raw_start;
      scanner->omit_end = scanner->position + (uint64_t)(scanner->next == ',');
    }
    if (status == SS_OK && scanner->next != '}') {
      status = ss_manifest_expect(scanner, ',', 1);
      if (status == SS_OK && scanner->next != '"') {
        status = SS_ERROR_INTEGRITY;
      }
    }
    ++member;
  }
  return status == SS_OK ? ss_manifest_expect(scanner, '}', 1) : status;
}

static SsStatus ss_manifest_value(SsManifestScanner *scanner, size_t depth,
                                  SsManifestIdGroup group, size_t record) {
  if (depth > scanner->limits->maximum_depth) {
    return SS_ERROR_LIMIT;
  }
  if (scanner->next == '{') {
    return ss_manifest_object(scanner, depth, group, record);
  }
  if (scanner->next == '"') {
    return ss_manifest_string(scanner);
  }
  if (scanner->next == '[') {
    SsStatus status = ss_manifest_take(scanner, 1);
    while (status == SS_OK && scanner->next != ']') {
      status = ss_manifest_value(scanner, depth + 1U, 0, 0U);
      if (status == SS_OK && scanner->next != ']') {
        status = ss_manifest_expect(scanner, ',', 1);
        if (status == SS_OK && scanner->next == ']') {
          status = SS_ERROR_JSON;
        }
      }
    }
    return status == SS_OK ? ss_manifest_expect(scanner, ']', 1) : status;
  }
  const uint64_t start = scanner->position;
  SsStatus status = SS_OK;
  while (status == SS_OK && scanner->next != EOF && scanner->next != ',' &&
         scanner->next != '}' && scanner->next != ']') {
    status = ss_manifest_take(scanner, 1);
  }
  return scanner->position == start ? SS_ERROR_JSON : status;
}

static SsStatus ss_manifest_hash_range(int descriptor, uint64_t begin, uint64_t end,
                                       SsSha256 *digest) {
  unsigned char buffer[SS_MANIFEST_READ_BYTES];
  while (begin < end) {
    const size_t count =
        end - begin < sizeof(buffer) ? (size_t)(end - begin) : sizeof(buffer);
    const ssize_t got = pread(descriptor, buffer, count, (off_t)begin);
    if (got < 0 && errno == EINTR) {
      continue;
    }
    if (got <= 0) {
      return SS_ERROR_IO;
    }
    SsStatus status = ss_sha256_update(digest, buffer, (size_t)got);
    if (status != SS_OK) {
      return status;
    }
    begin += (uint64_t)got;
  }
  return SS_OK;
}

static SsStatus ss_manifest_scan(SsCompiledManifest *manifest,
                                 const SsJsonLimits *limits) {
  const int duplicate = dup(manifest->descriptor);
  FILE *stream = duplicate < 0 ? NULL : fdopen(duplicate, "rb");
  unsigned char *canonical = NULL;
  size_t canonical_size = 0U;
  SsJsonError json_error = {0};
  SsInputFileIdentity after = {0};
  if (stream == NULL) {
    if (duplicate >= 0) {
      (void)close(duplicate);
    }
    return SS_ERROR_IO;
  }
  SsManifestScanner scanner = {
      .stream = stream, .limits = limits, .manifest = manifest, .next = getc(stream)};
  SsStatus status = ss_manifest_value(&scanner, 0U, 0, 0U);
  if (status == SS_OK &&
      (scanner.next != EOF || scanner.position != manifest->identity.bytes)) {
    status = SS_ERROR_INTEGRITY;
  }
  if (fclose(stream) != 0 && status == SS_OK) {
    status = SS_ERROR_IO;
  }
  if (status == SS_OK) {
    if (ss_json_canonicalize(manifest->data, manifest->size, &canonical,
                             &canonical_size, &json_error) != SS_JSON_OK) {
      status = SS_ERROR_JSON;
    } else if (canonical_size != manifest->size ||
               memcmp(canonical, manifest->data, manifest->size) != 0) {
      status = SS_ERROR_INTEGRITY;
    }
  }
  ss_json_free(canonical);
  if (status == SS_OK && scanner.omit_end != 0U) {
    SsSha256 digest;
    ss_sha256_init(&digest);
    status =
        ss_manifest_hash_range(manifest->descriptor, 0U, scanner.omit_begin, &digest);
    if (status == SS_OK) {
      status = ss_manifest_hash_range(manifest->descriptor, scanner.omit_end,
                                      manifest->identity.bytes, &digest);
    }
    if (status == SS_OK) {
      ss_sha256_final(&digest, manifest->digest);
    }
  }
  if (status == SS_OK && (ss_io_input_identity(manifest->descriptor, &after) != SS_OK ||
                          memcmp(&after, &manifest->identity, sizeof(after)) != 0)) {
    status = SS_ERROR_INTEGRITY;
  }
  return status;
}

SsStatus ss_io_read_compiled_manifest(MPI_Comm communicator, const char *path,
                                      SsCompiledManifest *manifest,
                                      SsErrorRecord *error) {
  const SsJsonLimits limits = ss_json_default_limits();
  SsInputFileIdentity root_identity = {0};
  uint64_t lengths[2] = {0U, 0U};
  int rank = 0;
  SsStatus status = SS_OK;
  memset(manifest, 0, sizeof(*manifest));
  manifest->descriptor = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  status = ss_io_input_identity(manifest->descriptor, &manifest->identity);
  if (status == SS_OK && manifest->identity.bytes > limits.maximum_bytes) {
    status = SS_ERROR_LIMIT;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    ss_error_set(error, status, (SsErrorKey){1U, 3U, 0U, 0U, 0U, (uint32_t)rank},
                 status == SS_ERROR_LIMIT ? "MANIFEST_BYTE_LIMIT"
                                          : "MANIFEST_PATH_INVALID",
                 "compiled manifest could not be opened within the configured limits");
    return status;
  }
  if (rank == 0) {
    root_identity = manifest->identity;
  }
  if (MPI_Bcast(&root_identity, (int)sizeof(root_identity), MPI_BYTE, 0,
                communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  status = memcmp(&root_identity, &manifest->identity, sizeof(root_identity)) != 0
               ? SS_ERROR_INTEGRITY
               : SS_OK;
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    ss_error_set(error, status, (SsErrorKey){1U, 3U, 0U, 0U, 0U, (uint32_t)rank},
                 "RANK_MANIFEST_MISMATCH",
                 "ranks observed different compiled manifest files");
    return status;
  }
  if (rank == 0) {
    status = ss_manifest_scan(manifest, &limits);
    lengths[0] = manifest->size;
    lengths[1] = manifest->span_count;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    ss_error_set(error, status, (SsErrorKey){1U, 3U, 0U, 0U, 0U, (uint32_t)rank},
                 status == SS_ERROR_INTEGRITY ? "MANIFEST_CANONICAL_BYTES"
                                              : "MANIFEST_READ",
                 "compiled manifest is invalid, has changed, or cannot be read");
    return status;
  }
  if (MPI_Bcast(lengths, 2, MPI_UINT64_T, 0, communicator) != MPI_SUCCESS ||
      lengths[0] > (uint64_t)INT_MAX ||
      lengths[1] > (uint64_t)INT_MAX / sizeof(*manifest->spans)) {
    return SS_ERROR_LIMIT;
  }
  if (rank != 0) {
    manifest->size = (size_t)lengths[0];
    manifest->span_count = (size_t)lengths[1];
    manifest->data = malloc(manifest->size == 0U ? 1U : manifest->size);
    manifest->spans = calloc(manifest->span_count == 0U ? 1U : manifest->span_count,
                             sizeof(*manifest->spans));
    if (manifest->data == NULL || manifest->spans == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK &&
      (MPI_Bcast(manifest->data, (int)manifest->size, MPI_BYTE, 0, communicator) !=
           MPI_SUCCESS ||
       MPI_Bcast(manifest->spans,
                 (int)(manifest->span_count * sizeof(*manifest->spans)), MPI_BYTE, 0,
                 communicator) != MPI_SUCCESS ||
       MPI_Bcast(manifest->digest, (int)sizeof(manifest->digest), MPI_CHAR, 0,
                 communicator) != MPI_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  return status;
}

static SsStatus ss_manifest_read_id_chunk(int descriptor, uint64_t position,
                                          size_t count, int last, SsEntityId *ids) {
  unsigned char
      bytes[(SS_MANIFEST_READ_BYTES / SS_MANIFEST_ID_BYTES) * SS_MANIFEST_ID_BYTES];
  const size_t length = count * SS_MANIFEST_ID_BYTES - (size_t)last;
  size_t read_count = 0U;
  while (read_count < length) {
    const ssize_t got = pread(descriptor, bytes + read_count, length - read_count,
                              (off_t)(position + read_count));
    if (got < 0 && errno == EINTR) {
      continue;
    }
    if (got <= 0) {
      return SS_ERROR_IO;
    }
    read_count += (size_t)got;
  }
  for (size_t i = 0U; i < count; ++i) {
    const unsigned char *row = bytes + i * SS_MANIFEST_ID_BYTES;
    SsEntityId id = 0U;
    if (row[0] != '"' || row[17] != '"' ||
        ((i + 1U < count || last == 0) && row[18] != ',')) {
      return SS_ERROR_INTEGRITY;
    }
    for (size_t j = 1U; j < 17U; ++j) {
      const int digit = ss_manifest_hex(row[j]);
      if (digit < 0) {
        return SS_ERROR_INTEGRITY;
      }
      id = (id << 4U) | (uint64_t)digit;
    }
    ids[i] = id;
  }
  return SS_OK;
}

SsStatus ss_io_read_manifest_ids(const SsCompiledManifest *manifest,
                                 SsManifestIdGroup group, size_t record, int rank,
                                 int ranks, SsEntityId **ids, size_t *count,
                                 uint64_t *global_count) {
  const SsManifestIdSpan *span = NULL;
  size_t start = 0U;
  *ids = NULL;
  *count = 0U;
  *global_count = 0U;
  for (size_t i = 0U; i < manifest->span_count; ++i) {
    if (manifest->spans[i].group == (uint64_t)group &&
        manifest->spans[i].record == record) {
      span = &manifest->spans[i];
      break;
    }
  }
  if (span == NULL) {
    return SS_ERROR_SCHEMA;
  }
  *global_count = span->count;
  ss_partition_range((size_t)span->count, rank, ranks, &start, count);
  if (*count != 0U) {
    *ids = malloc(*count * sizeof(**ids));
    if (*ids == NULL) {
      return SS_ERROR_OUT_OF_MEMORY;
    }
  }
  const size_t block = SS_MANIFEST_READ_BYTES / SS_MANIFEST_ID_BYTES;
  for (size_t offset = 0U; offset < *count;) {
    const size_t chunk = *count - offset < block ? *count - offset : block;
    SsStatus status = ss_manifest_read_id_chunk(
        manifest->descriptor, span->offset + (start + offset) * SS_MANIFEST_ID_BYTES,
        chunk, start + offset + chunk == span->count, *ids + offset);
    if (status != SS_OK) {
      free(*ids);
      *ids = NULL;
      return status;
    }
    offset += chunk;
  }
  return SS_OK;
}

void ss_io_compiled_manifest_destroy(SsCompiledManifest *manifest) {
  if (manifest->descriptor >= 0) {
    (void)close(manifest->descriptor);
  }
  free(manifest->spans);
  free(manifest->data);
  memset(manifest, 0, sizeof(*manifest));
  manifest->descriptor = -1;
}
