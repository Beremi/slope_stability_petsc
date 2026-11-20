/* Helpers shared by the result writer and checkpoints: text validation,
 * envelopes, HDF5 attributes, digests and indexed dataset I/O.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "result_envelope.h"
#include "result_internal.h"
#include "slope_internal/core/sort.h"
#include "slope_internal/protocol/json.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SS_RESULT_DIGEST_CHUNK_ROWS 1024U

typedef struct {
  uint64_t index;
  char digest[SS_SHA256_HEX_CAPACITY];
} SsResultChunkDigest;

SsStatus ss_result_collective_digest_agree(MPI_Comm communicator, int size,
                                           const char digest[SS_SHA256_HEX_CAPACITY],
                                           SsStatus status) {
  char *all_digests = NULL;
  int rank;
  if (status == SS_OK) {
    all_digests = (char *)malloc((size_t)size * (size_t)SS_SHA256_HEX_CAPACITY);
    if (all_digests == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    free(all_digests);
    return status;
  }
  if (all_digests == NULL) {
    return SS_ERROR_INTERNAL;
  }
  if (MPI_Allgather(digest, SS_SHA256_HEX_CAPACITY, MPI_CHAR, all_digests,
                    SS_SHA256_HEX_CAPACITY, MPI_CHAR, communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  for (rank = 1; status == SS_OK && rank < size; ++rank) {
    if (memcmp(all_digests, all_digests + (size_t)rank * (size_t)SS_SHA256_HEX_CAPACITY,
               SS_SHA256_HEX_CAPACITY) != 0) {
      status = SS_ERROR_INTEGRITY;
    }
  }
  free(all_digests);
  return ss_status_allreduce_max(communicator, status);
}

SsStatus ss_result_collective_payload_agree(MPI_Comm communicator,
                                            const unsigned char *payload,
                                            size_t payload_size) {
  char local_digest[SS_SHA256_HEX_CAPACITY];
  const unsigned char empty = 0U;
  int size = 0;
  SsStatus status = SS_OK;
  if ((payload_size != 0U && payload == NULL) ||
      MPI_Comm_size(communicator, &size) != MPI_SUCCESS || size <= 0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_sha256_bytes(payload_size == 0U ? &empty : payload, payload_size,
                           local_digest);
  return ss_result_collective_digest_agree(communicator, size, local_digest, status);
}

int ss_result_text_is_valid(const char *value, size_t capacity) {
  size_t index;
  size_t length;
  if (value == NULL || capacity == 0U) {
    return 0;
  }
  length = ss_bounded_length(value, capacity);
  if (length == 0U || length >= capacity) {
    return 0;
  }
  for (index = 0U; index < length; ++index) {
    const unsigned char byte = (unsigned char)value[index];
    if (byte < 0x20U || byte == 0x7fU) {
      return 0;
    }
  }
  return 1;
}

int ss_result_name_valid(const char *value) {
  size_t index;
  if (!ss_result_text_is_valid(value, SS_RESULT_TEXT_CAPACITY)) {
    return 0;
  }
  for (index = 0U; value[index] != '\0'; ++index) {
    const char byte = value[index];
    if (!((byte >= 'a' && byte <= 'z') || (byte >= '0' && byte <= '9') ||
          byte == '_')) {
      return 0;
    }
  }
  return 1;
}

int ss_result_spec_identity_valid(const char *directory, const char *problem_id,
                                  const char *run_id, const char *build_id,
                                  const char *mesh_id, const char *geometry_id) {
  return ss_result_text_is_valid(directory, SS_RESULT_PATH_CAPACITY) &&
         directory[0] == '/' && ss_valid_content_id(problem_id) &&
         ss_valid_content_id(run_id) && ss_valid_content_id(build_id) &&
         ss_valid_content_id(mesh_id) && ss_valid_content_id(geometry_id);
}

int ss_result_orders_valid(uint32_t field_order, uint32_t geometry_order) {
  return (field_order == 1U || field_order == 2U || field_order == 4U ||
          field_order == 8U) &&
         (geometry_order == 1U || geometry_order == 2U || geometry_order == 4U);
}

SsStatus ss_result_copy_text(char *destination, size_t capacity, const char *source) {
  if (destination == NULL || !ss_result_text_is_valid(source, capacity)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  (void)ss_copy_text(destination, capacity, source);
  return SS_OK;
}

int ss_result_envelope_header(yyjson_mut_doc *document, yyjson_mut_val *root,
                              yyjson_mut_val *version, const char *kind, const char *id,
                              uint32_t minor) {
  return yyjson_mut_obj_add_str(document, root, "kind", kind) &&
         yyjson_mut_obj_add_str(document, root, "id", id) &&
         yyjson_mut_obj_add_uint(document, version, "major", 1U) &&
         yyjson_mut_obj_add_uint(document, version, "minor", minor) &&
         yyjson_mut_obj_add_uint(document, version, "patch", 0U);
}

int ss_result_envelope_identity(yyjson_mut_doc *document, yyjson_mut_val *identity,
                                const char *id_key, const char *id,
                                const char *logical_digest) {
  return yyjson_mut_obj_add_str(document, identity, id_key, id) &&
         yyjson_mut_obj_add_str(document, identity, "logical_digest", logical_digest);
}

int ss_result_envelope_run(yyjson_mut_doc *document, yyjson_mut_val *run,
                           const char *run_id, const char *problem_id,
                           const char *build_id) {
  return yyjson_mut_obj_add_str(document, run, "run_id", run_id) &&
         yyjson_mut_obj_add_str(document, run, "problem_id", problem_id) &&
         yyjson_mut_obj_add_str(document, run, "build_id", build_id);
}

int ss_result_envelope_payload(yyjson_mut_doc *document, yyjson_mut_val *payload,
                               const char *path, uint64_t bytes, const char *sha256,
                               const char *logical_digest) {
  return yyjson_mut_obj_add_str(document, payload, "path", path) &&
         yyjson_mut_obj_add_uint(document, payload, "bytes", bytes) &&
         yyjson_mut_obj_add_str(document, payload, "sha256", sha256) &&
         yyjson_mut_obj_add_str(document, payload, "logical_digest", logical_digest);
}

SsStatus ss_result_envelope_serialize(yyjson_mut_doc *document, unsigned char **output,
                                      size_t *output_size) {
  size_t size = 0U;
  SsJsonError error = {0};
  char *serialized = yyjson_mut_write(document, YYJSON_WRITE_NOFLAG, &size);
  yyjson_mut_doc_free(document);
  if (serialized == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  const SsJsonStatus status = ss_json_canonicalize((const unsigned char *)serialized,
                                                   size, output, output_size, &error);
  free(serialized);
  return status == SS_JSON_OK ? SS_OK : SS_ERROR_JSON;
}

int ss_result_json_string(yyjson_val *object, const char *key, char *output,
                          size_t capacity) {
  yyjson_val *value = object == NULL ? NULL : yyjson_obj_get(object, key);
  const char *text = yyjson_get_str(value);
  if (text == NULL || strlen(text) >= capacity) {
    return 0;
  }
  memcpy(output, text, strlen(text) + 1U);
  return 1;
}

void ss_result_store_u64_le(unsigned char output[8], uint64_t value) {
  size_t index;
  for (index = 0U; index < 8U; ++index) {
    output[index] = (unsigned char)(value >> (index * 8U));
  }
}

void ss_result_store_double_le(unsigned char output[8], double value) {
  uint64_t bits = 0U;
  memcpy(&bits, &value, sizeof(bits));
  ss_result_store_u64_le(output, bits);
}

SsStatus ss_result_hdf5_string_attribute(hid_t object, const char *name,
                                         const char *value) {
  hid_t type = -1;
  hid_t space = -1;
  hid_t attribute = -1;
  SsStatus status = SS_OK;
  if (object < 0 || !ss_result_text_is_valid(name, SS_RESULT_TEXT_CAPACITY) ||
      !ss_result_text_is_valid(value, 4096U)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  type = H5Tcopy(H5T_C_S1);
  space = H5Screate(H5S_SCALAR);
  if (type < 0 || space < 0 || H5Tset_size(type, strlen(value) + 1U) < 0 ||
      H5Tset_cset(type, H5T_CSET_UTF8) < 0 ||
      H5Tset_strpad(type, H5T_STR_NULLTERM) < 0) {
    status = SS_ERROR_HDF5;
    goto cleanup;
  }
  attribute = H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
  if (attribute < 0 || H5Awrite(attribute, type, value) < 0) {
    status = SS_ERROR_HDF5;
  }

cleanup:
  status = ss_result_hdf5_close(&attribute, H5Aclose, status);
  status = ss_result_hdf5_close(&space, H5Sclose, status);
  status = ss_result_hdf5_close(&type, H5Tclose, status);
  return status;
}

SsStatus ss_result_hdf5_u64_attribute(hid_t object, const char *name, uint64_t value) {
  hid_t space = -1;
  hid_t attribute = -1;
  SsStatus status = SS_OK;
  if (object < 0 || !ss_result_text_is_valid(name, SS_RESULT_TEXT_CAPACITY)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  space = H5Screate(H5S_SCALAR);
  attribute = space < 0 ? -1
                        : H5Acreate2(object, name, H5T_STD_U64LE, space, H5P_DEFAULT,
                                     H5P_DEFAULT);
  if (space < 0 || attribute < 0 ||
      H5Awrite(attribute, H5T_NATIVE_UINT64, &value) < 0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_result_hdf5_close(&attribute, H5Aclose, status);
  status = ss_result_hdf5_close(&space, H5Sclose, status);
  return status;
}

SsStatus ss_result_hdf5_read_string_attribute(hid_t object, const char *name,
                                              char *output, size_t capacity) {
  hid_t attribute = -1;
  hid_t type = -1;
  size_t size;
  SsStatus status = SS_ERROR_HDF5;
  if (object < 0 || name == NULL || output == NULL || capacity == 0U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  output[0] = '\0';
  attribute = H5Aopen(object, name, H5P_DEFAULT);
  type = attribute < 0 ? -1 : H5Aget_type(attribute);
  if (type < 0 || H5Tget_class(type) != H5T_STRING || H5Tis_variable_str(type) > 0) {
    goto cleanup;
  }
  size = H5Tget_size(type);
  if (size == 0U || size >= capacity || H5Aread(attribute, type, output) < 0) {
    goto cleanup;
  }
  output[size] = '\0';
  status = SS_OK;

cleanup:
  status = ss_result_hdf5_close(&type, H5Tclose, status);
  status = ss_result_hdf5_close(&attribute, H5Aclose, status);
  return status;
}

SsStatus ss_result_hdf5_read_u64_attribute(hid_t object, const char *name,
                                           uint64_t *value) {
  hid_t attribute;
  if (object < 0 || name == NULL || value == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  attribute = H5Aopen(object, name, H5P_DEFAULT);
  if (attribute < 0) {
    return SS_ERROR_HDF5;
  }
  if (H5Aread(attribute, H5T_NATIVE_UINT64, value) < 0) {
    (void)H5Aclose(attribute);
    return SS_ERROR_HDF5;
  }
  return H5Aclose(attribute) < 0 ? SS_ERROR_HDF5 : SS_OK;
}

SsStatus ss_result_hdf5_create_groups(hid_t file, MPI_Comm communicator,
                                      const char *const *paths, size_t count) {
  hid_t *groups = NULL;
  size_t index;
  SsStatus status = SS_OK;
  if (file < 0 || (count != 0U && paths == NULL)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  size_t bytes;
  status = ss_checked_mul_size(count == 0U ? 1U : count, sizeof(*groups), &bytes);
  if (status == SS_OK) {
    groups = (hid_t *)malloc(bytes);
    if (groups == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    free(groups);
    return status;
  }
  if (groups == NULL) {
    return SS_ERROR_INTERNAL;
  }
  for (index = 0U; index < count; ++index) {
    groups[index] = -1;
  }
  for (index = 0U; index < count; ++index) {
    groups[index] =
        H5Gcreate2(file, paths[index], H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (groups[index] < 0) {
      status = SS_ERROR_HDF5;
    }
    status = ss_status_allreduce_max(communicator, status);
    if (status != SS_OK) {
      break;
    }
  }
  for (index = count; index > 0U; --index) {
    status = ss_result_hdf5_close(&groups[index - 1U], H5Gclose, status);
  }
  free(groups);
  return ss_status_allreduce_max(communicator, status);
}

SsStatus ss_result_hdf5_close(hid_t *identifier, herr_t (*closer)(hid_t),
                              SsStatus status) {
  if (identifier != NULL && *identifier >= 0) {
    if (closer(*identifier) < 0 && status == SS_OK) {
      status = SS_ERROR_HDF5;
    }
    *identifier = -1;
  }
  return status;
}

SsStatus ss_result_digest_bytes(const void *data, size_t size,
                                char output[SS_SHA256_HEX_CAPACITY + 8U]) {
  memcpy(output, "sha256:", 7U);
  return ss_sha256_bytes(data, size, output + 7U);
}

SsStatus ss_result_broadcast_digest(MPI_Comm communicator, SsStatus status,
                                    char output[SS_SHA256_HEX_CAPACITY + 8U]) {
  int value = (int)status;
  if (MPI_Bcast(&value, 1, MPI_INT, 0, communicator) != MPI_SUCCESS ||
      MPI_Bcast(output, SS_SHA256_HEX_CAPACITY + 7, MPI_CHAR, 0, communicator) !=
          MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  return (SsStatus)value;
}

static SsStatus ss_result_select_rows(hid_t file_space, hid_t memory_space,
                                      uint64_t start, uint64_t rows, uint32_t columns) {
  hsize_t file_start[2] = {(hsize_t)start, 0U};
  hsize_t count[2] = {(hsize_t)rows, (hsize_t)columns};
  if (rows == 0U) {
    return H5Sselect_none(file_space) < 0 || H5Sselect_none(memory_space) < 0
               ? SS_ERROR_HDF5
               : SS_OK;
  }
  return H5Sselect_hyperslab(file_space, H5S_SELECT_SET, file_start, NULL, count,
                             NULL) < 0
             ? SS_ERROR_HDF5
             : SS_OK;
}

static int ss_result_chunk_compare(const void *left_raw, const void *right_raw) {
  const SsResultChunkDigest *left = (const SsResultChunkDigest *)left_raw;
  const SsResultChunkDigest *right = (const SsResultChunkDigest *)right_raw;
  return left->index < right->index ? -1 : left->index > right->index ? 1 : 0;
}

static SsStatus
ss_result_dataset_logical_digest(MPI_Comm communicator, hid_t ids_dataset,
                                 hid_t values_dataset, const char *semantic,
                                 uint64_t global_count, uint32_t component_count,
                                 char output[SS_SHA256_HEX_CAPACITY + 8U]) {
  SsResultChunkDigest *local_digests = NULL;
  SsResultChunkDigest *all_digests = NULL;
  int *counts = NULL;
  int *byte_counts = NULL;
  int *byte_offsets = NULL;
  int rank = -1;
  int size = 0;
  uint64_t chunk_count;
  uint64_t chunk;
  size_t local_count = 0U;
  size_t local_cursor = 0U;
  int total_count = 0;
  SsStatus status = SS_OK;
  if (semantic == NULL || output == NULL || component_count == 0U ||
      MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(communicator, &size) != MPI_SUCCESS || size <= 0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  chunk_count = global_count / SS_RESULT_DIGEST_CHUNK_ROWS +
                (global_count % SS_RESULT_DIGEST_CHUNK_ROWS != 0U ? 1U : 0U);
  for (chunk = (uint64_t)(uint32_t)rank; chunk < chunk_count;
       chunk += (uint64_t)(uint32_t)size) {
    ++local_count;
  }
  size_t digest_bytes;
  status = ss_checked_mul_size(local_count == 0U ? 1U : local_count,
                               sizeof(*local_digests), &digest_bytes);
  if (status == SS_OK) {
    local_digests = (SsResultChunkDigest *)calloc(1U, digest_bytes);
    if (local_digests == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    free(local_digests);
    return status;
  }
  for (chunk = (uint64_t)(uint32_t)rank; chunk < chunk_count;
       chunk += (uint64_t)(uint32_t)size) {
    const uint64_t start = chunk * SS_RESULT_DIGEST_CHUNK_ROWS;
    const uint64_t rows = global_count - start < SS_RESULT_DIGEST_CHUNK_ROWS
                              ? global_count - start
                              : SS_RESULT_DIGEST_CHUNK_ROWS;
    SsEntityId *ids = NULL;
    double *values = NULL;
    unsigned char *logical = NULL;
    size_t row;
    size_t row_bytes;
    size_t logical_size = 0U;
    hid_t ids_file_space = -1;
    hid_t ids_memory_space = -1;
    hid_t values_file_space = -1;
    hid_t values_memory_space = -1;
    hid_t transfer = -1;
    hsize_t ids_dimensions[1] = {(hsize_t)rows};
    hsize_t values_dimensions[2] = {(hsize_t)rows, (hsize_t)component_count};
    hsize_t ids_start[1] = {(hsize_t)start};
    hsize_t ids_count[1] = {(hsize_t)rows};
    size_t local_rows;
    size_t id_bytes;
    size_t value_bytes;
    size_t row_components;
    if (status == SS_OK &&
        (ss_checked_u64_to_size(rows, &local_rows) != SS_OK ||
         ss_checked_add_size(component_count, 1U, &row_components) != SS_OK ||
         ss_checked_mul_size(row_components, 8U, &row_bytes) != SS_OK ||
         ss_checked_mul_size(local_rows, sizeof(*ids), &id_bytes) != SS_OK ||
         ss_checked_product_size(
             (size_t[]){local_rows, component_count, sizeof(*values)}, 3U,
             &value_bytes) != SS_OK ||
         ss_checked_mul_size(local_rows, row_bytes, &logical_size) != SS_OK)) {
      status = SS_ERROR_OVERFLOW;
    }
    if (status == SS_OK) {
      ids = (SsEntityId *)malloc(id_bytes);
      values = (double *)malloc(value_bytes);
      logical = (unsigned char *)malloc(logical_size == 0U ? 1U : logical_size);
      if (ids == NULL || values == NULL || logical == NULL) {
        status = SS_ERROR_OUT_OF_MEMORY;
      }
    }
    if (status == SS_OK) {
      ids_file_space = H5Dget_space(ids_dataset);
      values_file_space = H5Dget_space(values_dataset);
      ids_memory_space = H5Screate_simple(1, ids_dimensions, NULL);
      values_memory_space = H5Screate_simple(2, values_dimensions, NULL);
      transfer = H5Pcreate(H5P_DATASET_XFER);
      if (ids_file_space < 0 || values_file_space < 0 || ids_memory_space < 0 ||
          values_memory_space < 0 || transfer < 0 ||
          H5Pset_dxpl_mpio(transfer, H5FD_MPIO_INDEPENDENT) < 0 ||
          H5Sselect_hyperslab(ids_file_space, H5S_SELECT_SET, ids_start, NULL,
                              ids_count, NULL) < 0 ||
          ss_result_select_rows(values_file_space, values_memory_space, start, rows,
                                component_count) != SS_OK ||
          H5Dread(ids_dataset, H5T_NATIVE_UINT64, ids_memory_space, ids_file_space,
                  transfer, ids) < 0 ||
          H5Dread(values_dataset, H5T_NATIVE_DOUBLE, values_memory_space,
                  values_file_space, transfer, values) < 0) {
        status = SS_ERROR_HDF5;
      }
    }
    if (status == SS_OK) {
      for (row = 0U; row < (size_t)rows; ++row) {
        uint32_t component;
        unsigned char *target = logical + row * row_bytes;
        ss_result_store_u64_le(target, ids[row]);
        for (component = 0U; component < component_count; ++component) {
          const double value = values[row * (size_t)component_count + component];
          if (!isfinite(value)) {
            status = SS_ERROR_INTEGRITY;
            break;
          }
          ss_result_store_double_le(target + (size_t)(component + 1U) * 8U, value);
        }
        if (status != SS_OK) {
          break;
        }
      }
    }
    if (status == SS_OK) {
      local_digests[local_cursor].index = chunk;
      status =
          ss_sha256_bytes(logical, logical_size, local_digests[local_cursor].digest);
      ++local_cursor;
    }
    status = ss_result_hdf5_close(&transfer, H5Pclose, status);
    status = ss_result_hdf5_close(&values_memory_space, H5Sclose, status);
    status = ss_result_hdf5_close(&values_file_space, H5Sclose, status);
    status = ss_result_hdf5_close(&ids_memory_space, H5Sclose, status);
    status = ss_result_hdf5_close(&ids_file_space, H5Sclose, status);
    free(logical);
    free(values);
    free(ids);
    if (status != SS_OK) {
      break;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    free(local_digests);
    return status;
  }
  if (local_count > (size_t)INT_MAX) {
    status = SS_ERROR_OVERFLOW;
  }
  if (rank == 0) {
    counts = (int *)calloc((size_t)size, sizeof(*counts));
    byte_counts = (int *)calloc((size_t)size, sizeof(*byte_counts));
    byte_offsets = (int *)calloc((size_t)size, sizeof(*byte_offsets));
    if (counts == NULL || byte_counts == NULL || byte_offsets == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (rank == 0 && (counts == NULL || byte_counts == NULL || byte_offsets == NULL)) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (rank == 0 && (counts == NULL || byte_counts == NULL || byte_offsets == NULL)) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  {
    int local_count_int = (int)local_count;
    if (MPI_Gather(&local_count_int, 1, MPI_INT, counts, 1, MPI_INT, 0, communicator) !=
        MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
      goto cleanup;
    }
  }
  if (rank == 0 && counts != NULL && byte_counts != NULL && byte_offsets != NULL) {
    int process;
    for (process = 0; process < size; ++process) {
      if (counts[process] < 0 ||
          counts[process] > INT_MAX / (int)sizeof(SsResultChunkDigest) ||
          counts[process] > INT_MAX - total_count ||
          (process > 0 &&
           byte_offsets[process - 1] > INT_MAX - byte_counts[process - 1])) {
        status = SS_ERROR_OVERFLOW;
        break;
      }
      byte_counts[process] = counts[process] * (int)sizeof(SsResultChunkDigest);
      if (process > 0) {
        byte_offsets[process] = byte_offsets[process - 1] + byte_counts[process - 1];
      }
      total_count += counts[process];
    }
    if (status == SS_OK && (uint64_t)total_count != chunk_count) {
      status = SS_ERROR_INTEGRITY;
    }
    if (status == SS_OK) {
      all_digests = (SsResultChunkDigest *)calloc(
          total_count == 0 ? 1U : (size_t)total_count, sizeof(*all_digests));
      if (all_digests == NULL) {
        status = SS_ERROR_OUT_OF_MEMORY;
      }
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (rank == 0 && all_digests == NULL) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  if (rank == 0 && all_digests == NULL) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (MPI_Gatherv(local_digests, (int)local_count * (int)sizeof(SsResultChunkDigest),
                  MPI_BYTE, all_digests, byte_counts, byte_offsets, MPI_BYTE, 0,
                  communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (rank == 0 && all_digests != NULL) {
    unsigned char *combined;
    size_t semantic_size = strlen(semantic);
    size_t combined_size;
    size_t prefix_size;
    size_t cursor;
    ss_sort(all_digests, (size_t)total_count, sizeof(*all_digests),
            ss_result_chunk_compare);
    if (semantic_size == SIZE_MAX) {
      status = SS_ERROR_OVERFLOW;
    } else {
      prefix_size = semantic_size + 1U;
    }
    if (status == SS_OK &&
        (size_t)total_count > (SIZE_MAX - prefix_size) /
                                  (sizeof(uint64_t) + SS_SHA256_HEX_CAPACITY - 1U)) {
      status = SS_ERROR_OVERFLOW;
    }
    if (status == SS_OK) {
      combined_size = prefix_size + (size_t)total_count * (sizeof(uint64_t) +
                                                           SS_SHA256_HEX_CAPACITY - 1U);
      if (combined_size < prefix_size || combined_size == 0U) {
        status = SS_ERROR_OVERFLOW;
      }
    }
    if (status == SS_OK) {
      combined = (unsigned char *)malloc(combined_size);
      if (combined == NULL) {
        status = SS_ERROR_OUT_OF_MEMORY;
      } else {
        memcpy(combined, semantic, semantic_size);
        combined[semantic_size] = '\0';
        cursor = semantic_size + 1U;
        for (chunk = 0U; chunk < (uint64_t)total_count; ++chunk) {
          if (all_digests[chunk].index != chunk) {
            status = SS_ERROR_INTEGRITY;
            break;
          }
          ss_result_store_u64_le(combined + cursor, chunk);
          cursor += sizeof(uint64_t);
          memcpy(combined + cursor, all_digests[chunk].digest,
                 SS_SHA256_HEX_CAPACITY - 1U);
          cursor += SS_SHA256_HEX_CAPACITY - 1U;
        }
        if (status == SS_OK) {
          memcpy(output, "sha256:", 7U);
          status = ss_sha256_bytes(combined, combined_size, output + 7U);
        }
        free(combined);
      }
    }
  }
  {
    int status_value = (int)status;
    if (MPI_Bcast(&status_value, 1, MPI_INT, 0, communicator) != MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    } else {
      status = (SsStatus)status_value;
    }
    if (status == SS_OK && MPI_Bcast(output, SS_SHA256_HEX_CAPACITY + 7, MPI_CHAR, 0,
                                     communicator) != MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }

cleanup:
  free(all_digests);
  free(byte_offsets);
  free(byte_counts);
  free(counts);
  free(local_digests);
  return status;
}

static SsStatus ss_result_dataset_attributes(
    hid_t dataset, const char *quantity_id, uint32_t quantity_version, const char *unit,
    const char *location, const char *basis, const char *mesh_id,
    const char *geometry_id, SsStateId state_id, uint32_t field_order,
    uint32_t geometry_order, const char *components, const char *digest, uint64_t rows,
    uint32_t component_count, const char *value_type) {
  char state[32];
  char shape[96];
  SsStatus status;
  if (snprintf(state, sizeof(state), "%llu", (unsigned long long)state_id) < 0 ||
      snprintf(shape, sizeof(shape), "[%llu,%u]", (unsigned long long)rows,
               component_count) < 0) {
    return SS_ERROR_OVERFLOW;
  }
  status = ss_result_hdf5_string_attribute(dataset, "quantity_id", quantity_id);
  if (status == SS_OK) {
    status =
        ss_result_hdf5_u64_attribute(dataset, "quantity_version", quantity_version);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_string_attribute(dataset, "unit", unit);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_string_attribute(dataset, "mesh_id", mesh_id);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_string_attribute(dataset, "geometry_id", geometry_id);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_string_attribute(dataset, "state_ids", state);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_u64_attribute(dataset, "state_id", state_id);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_string_attribute(dataset, "completeness", "complete");
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_string_attribute(dataset, "value_type", value_type);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_string_attribute(
        dataset, "digest_method", "sha256-semantic-stable-id-float64le-chunks-v1");
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_string_attribute(dataset, "digest", digest);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_string_attribute(dataset, "location", location);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_string_attribute(dataset, "basis", basis);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_u64_attribute(dataset, "field_order", field_order);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_u64_attribute(dataset, "geometry_order", geometry_order);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_string_attribute(dataset, "component_ids", components);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_string_attribute(dataset, "logical_shape", shape);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_string_attribute(dataset, "logical_layout_id",
                                             "indexed-entity-component-v1");
  }
  return status;
}

SsStatus ss_result_verify_indexed_dataset_digest(
    hid_t file, MPI_Comm communicator, const char *group_path,
    char logical_digest[SS_SHA256_HEX_CAPACITY + 8U]) {
  char ids_path[SS_RESULT_PATH_CAPACITY];
  char values_path[SS_RESULT_PATH_CAPACITY];
  char quantity_id[SS_RESULT_TEXT_CAPACITY];
  char unit[SS_RESULT_TEXT_CAPACITY];
  char location[SS_RESULT_TEXT_CAPACITY];
  char basis[SS_RESULT_TEXT_CAPACITY];
  char mesh_id[SS_RESULT_TEXT_CAPACITY];
  char geometry_id[SS_RESULT_TEXT_CAPACITY];
  char components[SS_RESULT_COMPONENT_CAPACITY * SS_RESULT_TEXT_CAPACITY];
  char expected[SS_SHA256_HEX_CAPACITY + 1U];
  char semantic[16384];
  uint64_t quantity_version = 0U;
  uint64_t state_id = 0U;
  uint64_t field_order = 0U;
  uint64_t geometry_order = 0U;
  hid_t ids_dataset = -1;
  hid_t values_dataset = -1;
  hid_t ids_space = -1;
  hid_t values_space = -1;
  hsize_t ids_dimensions[1] = {0U};
  hsize_t values_dimensions[2] = {0U, 0U};
  SsStatus status = SS_OK;
  if (file < 0 || group_path == NULL || logical_digest == NULL ||
      snprintf(ids_path, sizeof(ids_path), "%s/stable_ids", group_path) < 0 ||
      snprintf(values_path, sizeof(values_path), "%s/values", group_path) < 0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  ids_dataset = H5Dopen2(file, ids_path, H5P_DEFAULT);
  values_dataset = H5Dopen2(file, values_path, H5P_DEFAULT);
  ids_space = ids_dataset < 0 ? -1 : H5Dget_space(ids_dataset);
  values_space = values_dataset < 0 ? -1 : H5Dget_space(values_dataset);
  if (ids_dataset < 0 || values_dataset < 0 || ids_space < 0 || values_space < 0 ||
      H5Sget_simple_extent_ndims(ids_space) != 1 ||
      H5Sget_simple_extent_ndims(values_space) != 2 ||
      H5Sget_simple_extent_dims(ids_space, ids_dimensions, NULL) != 1 ||
      H5Sget_simple_extent_dims(values_space, values_dimensions, NULL) != 2 ||
      ids_dimensions[0] == 0U || ids_dimensions[0] != values_dimensions[0] ||
      values_dimensions[1] == 0U || values_dimensions[1] > UINT32_MAX) {
    status = SS_ERROR_SCHEMA;
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_string_attribute(values_dataset, "quantity_id",
                                                  quantity_id, sizeof(quantity_id));
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_u64_attribute(values_dataset, "quantity_version",
                                               &quantity_version);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_string_attribute(values_dataset, "unit", unit,
                                                  sizeof(unit));
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_string_attribute(values_dataset, "location", location,
                                                  sizeof(location));
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_string_attribute(values_dataset, "basis", basis,
                                                  sizeof(basis));
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_string_attribute(values_dataset, "mesh_id", mesh_id,
                                                  sizeof(mesh_id));
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_string_attribute(values_dataset, "geometry_id",
                                                  geometry_id, sizeof(geometry_id));
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_u64_attribute(values_dataset, "state_id", &state_id);
  }
  if (status == SS_OK) {
    status =
        ss_result_hdf5_read_u64_attribute(values_dataset, "field_order", &field_order);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_u64_attribute(values_dataset, "geometry_order",
                                               &geometry_order);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_string_attribute(values_dataset, "component_ids",
                                                  components, sizeof(components));
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_string_attribute(values_dataset, "digest", expected,
                                                  sizeof(expected));
  }
  if (status == SS_OK) {
    int count =
        snprintf(semantic, sizeof(semantic),
                 "quantity=%s|version=%llu|unit=%s|location=%s|basis=%s|"
                 "mesh=%s|geometry=%s|field_order=%llu|geometry_order=%llu|"
                 "state=%llu|components=%s|shape=%llu,%llu",
                 quantity_id, (unsigned long long)quantity_version, unit, location,
                 basis, mesh_id, geometry_id, (unsigned long long)field_order,
                 (unsigned long long)geometry_order, (unsigned long long)state_id,
                 components, (unsigned long long)values_dimensions[0],
                 (unsigned long long)values_dimensions[1]);
    if (count < 0 || (size_t)count >= sizeof(semantic)) {
      status = SS_ERROR_OVERFLOW;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    status = ss_result_dataset_logical_digest(
        communicator, ids_dataset, values_dataset, semantic,
        (uint64_t)values_dimensions[0], (uint32_t)values_dimensions[1], logical_digest);
  }
  if (status == SS_OK && (strlen(expected) != SS_SHA256_HEX_CAPACITY - 1U ||
                          strcmp(expected, logical_digest + 7U) != 0)) {
    status = SS_ERROR_INTEGRITY;
  }
  status = ss_result_hdf5_close(&values_space, H5Sclose, status);
  status = ss_result_hdf5_close(&ids_space, H5Sclose, status);
  status = ss_result_hdf5_close(&values_dataset, H5Dclose, status);
  status = ss_result_hdf5_close(&ids_dataset, H5Dclose, status);
  return ss_status_allreduce_max(communicator, status);
}

/* Each rank owns one contiguous, stable-ID-ordered file block. Empty blocks
 * still participate in collective HDF5 I/O with empty selections. */
static SsStatus ss_result_indexed_hyperslab(hid_t dataset, MPI_Comm communicator,
                                            const SsResultFileIndexPlan *plan,
                                            uint32_t columns, hid_t type, void *values,
                                            int read) {
  const int dimensions = columns == 0U ? 1 : 2;
  const hsize_t start[2] = {(hsize_t)plan->sorted_offset, 0U};
  const hsize_t count[2] = {(hsize_t)plan->sorted_count, (hsize_t)columns};
  const hsize_t memory_dimensions[2] = {plan->sorted_count == 0U ? 1U : count[0],
                                        count[1]};
  hsize_t extent[2] = {0U, 0U};
  hid_t file_space = H5Dget_space(dataset);
  hid_t memory_space = H5Screate_simple(dimensions, memory_dimensions, NULL);
  hid_t transfer = H5Pcreate(H5P_DATASET_XFER);
  hid_t creation = H5Dget_create_plist(dataset);
  const H5D_layout_t layout = creation < 0 ? H5D_LAYOUT_ERROR : H5Pget_layout(creation);
  SsStatus status = SS_OK;
  /* For a contiguous dataset the disjoint slabs are already MPI-IO blocks.
   * Keep the collective HDF5 call and bypass MPI-IO aggregator buffers. */
  if (file_space < 0 || memory_space < 0 || transfer < 0 ||
      layout == H5D_LAYOUT_ERROR ||
      (layout == H5D_CONTIGUOUS &&
       H5Pset_dxpl_mpio_collective_opt(transfer, H5FD_MPIO_INDIVIDUAL_IO) < 0) ||
      H5Pset_dxpl_mpio(transfer, H5FD_MPIO_COLLECTIVE) < 0) {
    status = SS_ERROR_HDF5;
  } else if (H5Sget_simple_extent_ndims(file_space) != dimensions ||
             H5Sget_simple_extent_dims(file_space, extent, NULL) != dimensions ||
             extent[0] != plan->summary.global_count || extent[1] != columns) {
    status = SS_ERROR_INTEGRITY;
  } else if (plan->sorted_count == 0U) {
    if (H5Sselect_none(file_space) < 0 || H5Sselect_none(memory_space) < 0) {
      status = SS_ERROR_HDF5;
    }
  } else if (H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL) <
             0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_result_hdf5_close(&creation, H5Pclose, status);
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    const herr_t transferred =
        read ? H5Dread(dataset, type, memory_space, file_space, transfer, values)
             : H5Dwrite(dataset, type, memory_space, file_space, transfer, values);
    if (transferred < 0) {
      status = SS_ERROR_HDF5;
    }
  }
  status = ss_result_hdf5_close(&transfer, H5Pclose, status);
  status = ss_result_hdf5_close(&memory_space, H5Sclose, status);
  status = ss_result_hdf5_close(&file_space, H5Sclose, status);
  return ss_status_allreduce_max(communicator, status);
}

static SsStatus ss_result_indexed_validate_values(size_t rows, uint32_t columns,
                                                  const double *values) {
  size_t count = 0U;
  if (ss_checked_mul_size(rows, columns, &count) != SS_OK) {
    return SS_ERROR_OVERFLOW;
  }
  for (size_t i = 0U; i < count; ++i) {
    if (!isfinite(values[i])) {
      return SS_ERROR_INTEGRITY;
    }
  }
  return SS_OK;
}

SsStatus ss_result_write_indexed_dataset(
    hid_t file, MPI_Comm communicator, const char *group_path, const char *quantity_id,
    uint32_t quantity_version, const char *unit, const char *location,
    const char *basis, const char *mesh_id, const char *geometry_id, SsStateId state_id,
    uint32_t field_order, uint32_t geometry_order, const char *const *component_ids,
    uint32_t component_count, uint64_t dataset_chunk_rows,
    const SsResultFileIndexPlan *plan, const double *local_values,
    char logical_digest[SS_SHA256_HEX_CAPACITY + 8U], uint64_t *logical_bytes) {
  char ids_path[SS_RESULT_PATH_CAPACITY];
  char values_path[SS_RESULT_PATH_CAPACITY];
  char components[SS_RESULT_COMPONENT_CAPACITY * SS_RESULT_TEXT_CAPACITY];
  char semantic[16384];
  char raw_digest[SS_SHA256_HEX_CAPACITY] = {0};
  hid_t group = -1;
  hid_t ids_space = -1;
  hid_t values_space = -1;
  hid_t ids_dataset = -1;
  hid_t values_dataset = -1;
  hid_t ids_creation = -1;
  hid_t values_creation = -1;
  hsize_t ids_dimensions[1];
  hsize_t values_dimensions[2];
  hsize_t ids_chunk[1];
  hsize_t values_chunk[2];
  hsize_t global_rows = 0U;
  void *sorted_values = NULL;
  size_t component_cursor = 0U;
  int ids_path_size;
  int values_path_size;
  int semantic_size;
  SsStatus status = SS_OK;
  if (file < 0 || group_path == NULL || quantity_id == NULL || unit == NULL ||
      location == NULL || basis == NULL || mesh_id == NULL || geometry_id == NULL ||
      component_ids == NULL || component_count == 0U ||
      component_count > SS_RESULT_COMPONENT_CAPACITY || plan == NULL ||
      (plan->local_count != 0U && local_values == NULL) || logical_digest == NULL ||
      logical_bytes == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (ss_checked_u64_to_hsize(plan->summary.global_count, &global_rows) != SS_OK) {
    return SS_ERROR_OVERFLOW;
  }
  components[0] = '\0';
  for (size_t index = 0U; index < component_count; ++index) {
    size_t length;
    if (!ss_result_text_is_valid(component_ids[index], SS_RESULT_TEXT_CAPACITY)) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
    length = strlen(component_ids[index]);
    if (component_cursor + length + (index == 0U ? 0U : 1U) + 1U > sizeof(components)) {
      return SS_ERROR_OVERFLOW;
    }
    if (index != 0U) {
      components[component_cursor++] = ',';
    }
    memcpy(components + component_cursor, component_ids[index], length);
    component_cursor += length;
    components[component_cursor] = '\0';
  }
  ids_path_size = snprintf(ids_path, sizeof(ids_path), "%s/stable_ids", group_path);
  values_path_size =
      snprintf(values_path, sizeof(values_path), "%s/values", group_path);
  semantic_size = snprintf(
      semantic, sizeof(semantic),
      "quantity=%s|version=%u|unit=%s|location=%s|basis=%s|"
      "mesh=%s|geometry=%s|field_order=%u|geometry_order=%u|"
      "state=%llu|components=%s|shape=%llu,%u",
      quantity_id, quantity_version, unit, location, basis, mesh_id, geometry_id,
      field_order, geometry_order, (unsigned long long)state_id, components,
      (unsigned long long)plan->summary.global_count, component_count);
  if (ids_path_size < 0 || (size_t)ids_path_size >= sizeof(ids_path) ||
      values_path_size < 0 || (size_t)values_path_size >= sizeof(values_path) ||
      semantic_size < 0 || (size_t)semantic_size >= sizeof(semantic)) {
    return SS_ERROR_OVERFLOW;
  }
  status = ss_result_indexed_validate_values(plan->local_count, component_count,
                                             local_values);
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    return status;
  }
  ids_dimensions[0] = global_rows;
  values_dimensions[0] = global_rows;
  values_dimensions[1] = (hsize_t)component_count;
  if (dataset_chunk_rows != 0U) {
    ids_chunk[0] = (hsize_t)(dataset_chunk_rows < plan->summary.global_count
                                 ? dataset_chunk_rows
                                 : plan->summary.global_count);
    values_chunk[0] = ids_chunk[0];
    values_chunk[1] = (hsize_t)component_count;
    ids_creation = H5Pcreate(H5P_DATASET_CREATE);
    values_creation = H5Pcreate(H5P_DATASET_CREATE);
    if (ids_creation < 0 || values_creation < 0 ||
        H5Pset_chunk(ids_creation, 1, ids_chunk) < 0 ||
        H5Pset_chunk(values_creation, 2, values_chunk) < 0) {
      status = SS_ERROR_HDF5;
    }
  }
  ids_space = H5Screate_simple(1, ids_dimensions, NULL);
  values_space = H5Screate_simple(2, values_dimensions, NULL);
  if (ids_space < 0 || values_space < 0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    group = H5Gcreate2(file, group_path, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = ss_status_allreduce_max(communicator, group < 0 ? SS_ERROR_HDF5 : SS_OK);
  }
  if (status == SS_OK) {
    ids_dataset =
        H5Dcreate2(file, ids_path, H5T_STD_U64LE, ids_space, H5P_DEFAULT,
                   dataset_chunk_rows == 0U ? H5P_DEFAULT : ids_creation, H5P_DEFAULT);
    values_dataset = H5Dcreate2(
        file, values_path, H5T_IEEE_F64LE, values_space, H5P_DEFAULT,
        dataset_chunk_rows == 0U ? H5P_DEFAULT : values_creation, H5P_DEFAULT);
    status = ss_status_allreduce_max(
        communicator, ids_dataset < 0 || values_dataset < 0 ? SS_ERROR_HDF5 : SS_OK);
  }
  if (status == SS_OK) {
    status = ss_result_indexed_hyperslab(ids_dataset, communicator, plan, 0U,
                                         H5T_NATIVE_UINT64, plan->sorted_ids, 0);
  }
  if (status == SS_OK) {
    status = ss_result_gather_indexed_rows(plan, component_count, sizeof(double),
                                           MPI_DOUBLE, local_values, &sorted_values);
  }
  if (status == SS_OK) {
    status =
        ss_result_indexed_hyperslab(values_dataset, communicator, plan, component_count,
                                    H5T_NATIVE_DOUBLE, sorted_values, 0);
  }
  free(sorted_values);
  if (status != SS_OK) {
    goto cleanup;
  }
  status = ss_result_dataset_logical_digest(communicator, ids_dataset, values_dataset,
                                            semantic, plan->summary.global_count,
                                            component_count, logical_digest);
  if (status != SS_OK) {
    goto cleanup;
  }
  memcpy(raw_digest, logical_digest + 7U, SS_SHA256_HEX_CAPACITY);
  status = ss_result_dataset_attributes(
      values_dataset, quantity_id, quantity_version, unit, location, basis, mesh_id,
      geometry_id, state_id, field_order, geometry_order, components, raw_digest,
      plan->summary.global_count, component_count, "float64");
  if (status == SS_OK) {
    status = ss_result_dataset_attributes(
        ids_dataset, quantity_id, quantity_version, "dimensionless", "stable_identity",
        "none", mesh_id, geometry_id, state_id, field_order, geometry_order,
        "stable_id", raw_digest, plan->summary.global_count, 1U, "uint64");
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    if (plan->summary.global_count >
        UINT64_MAX / ((uint64_t)component_count + UINT64_C(1)) / sizeof(uint64_t)) {
      status = SS_ERROR_OVERFLOW;
    } else {
      *logical_bytes = plan->summary.global_count *
                       ((uint64_t)component_count + UINT64_C(1)) * sizeof(uint64_t);
    }
  }

cleanup:
  status = ss_result_hdf5_close(&values_dataset, H5Dclose, status);
  status = ss_result_hdf5_close(&ids_dataset, H5Dclose, status);
  status = ss_result_hdf5_close(&values_creation, H5Pclose, status);
  status = ss_result_hdf5_close(&ids_creation, H5Pclose, status);
  status = ss_result_hdf5_close(&values_space, H5Sclose, status);
  status = ss_result_hdf5_close(&ids_space, H5Sclose, status);
  status = ss_result_hdf5_close(&group, H5Gclose, status);
  return ss_status_allreduce_max(communicator, status);
}

SsStatus ss_result_read_indexed_dataset(hid_t file, MPI_Comm communicator,
                                        const char *group_path,
                                        const SsResultFileIndexPlan *plan,
                                        uint32_t component_count,
                                        double *local_values) {
  char ids_path[SS_RESULT_PATH_CAPACITY];
  char values_path[SS_RESULT_PATH_CAPACITY];
  hid_t ids_dataset = -1;
  hid_t values_dataset = -1;
  SsEntityId *ids = NULL;
  double *sorted_values = NULL;
  size_t ids_bytes = 0U, values_bytes = 0U;
  SsStatus status = SS_OK;
  if (file < 0 || group_path == NULL || plan == NULL || component_count == 0U ||
      component_count > SS_RESULT_COMPONENT_CAPACITY ||
      (plan->local_count != 0U && local_values == NULL)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  const int ids_path_size =
      snprintf(ids_path, sizeof(ids_path), "%s/stable_ids", group_path);
  const int values_path_size =
      snprintf(values_path, sizeof(values_path), "%s/values", group_path);
  const size_t allocated_rows = plan->sorted_count == 0U ? 1U : plan->sorted_count;
  if (ids_path_size < 0 || (size_t)ids_path_size >= sizeof(ids_path) ||
      values_path_size < 0 || (size_t)values_path_size >= sizeof(values_path) ||
      ss_checked_mul_size(allocated_rows, sizeof(*ids), &ids_bytes) != SS_OK ||
      ss_checked_mul_size(allocated_rows, (size_t)component_count * sizeof(double),
                          &values_bytes) != SS_OK) {
    status = SS_ERROR_OVERFLOW;
  } else {
    ids = malloc(ids_bytes);
    sorted_values = malloc(values_bytes);
    if (ids == NULL || sorted_values == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    ids_dataset = H5Dopen2(file, ids_path, H5P_DEFAULT);
    values_dataset = H5Dopen2(file, values_path, H5P_DEFAULT);
    status = ss_status_allreduce_max(
        communicator, ids_dataset < 0 || values_dataset < 0 ? SS_ERROR_HDF5 : SS_OK);
  }
  if (status == SS_OK) {
    status = ss_result_indexed_hyperslab(ids_dataset, communicator, plan, 0U,
                                         H5T_NATIVE_UINT64, ids, 1);
    if (status == SS_OK && plan->sorted_count != 0U &&
        memcmp(ids, plan->sorted_ids, ids_bytes) != 0) {
      status = SS_ERROR_INTEGRITY;
    }
    status = ss_status_allreduce_max(communicator, status);
  }
  free(ids);
  if (status == SS_OK) {
    status =
        ss_result_indexed_hyperslab(values_dataset, communicator, plan, component_count,
                                    H5T_NATIVE_DOUBLE, sorted_values, 1);
  }
  if (status == SS_OK) {
    status = ss_result_indexed_validate_values(plan->sorted_count, component_count,
                                               sorted_values);
    status = ss_status_allreduce_max(communicator, status);
  }
  if (status == SS_OK) {
    status = ss_result_scatter_indexed_rows(plan, component_count, sizeof(double),
                                            MPI_DOUBLE, sorted_values, local_values);
  }
  free(sorted_values);
  status = ss_result_hdf5_close(&values_dataset, H5Dclose, status);
  status = ss_result_hdf5_close(&ids_dataset, H5Dclose, status);
  return ss_status_allreduce_max(communicator, status);
}
