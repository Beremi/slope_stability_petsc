/* Parallel HDF5 preflight and balanced payload reads.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/io/input.h"

#include "slope_internal/core/collective.h"
#include "slope_internal/core/util.h"

#include "slope_internal/diagnostics/diagnostics.h"

#include <hdf5.h>
#include <mpi.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void ss_hdf5_error(SsErrorRecord *error, int rank, const char *message) {
  ss_error_set(error, SS_ERROR_HDF5, (SsErrorKey){1U, 3U, 0U, 0U, 0U, (uint32_t)rank},
               "HDF5_LAYOUT_INVALID", message);
}

static int ss_string_attribute_read(hid_t object, const char *name, char *output,
                                    size_t capacity) {
  hid_t attribute = -1;
  hid_t type = -1;
  int read_ok = 0;
  if (output == NULL || capacity == 0U) {
    return 0;
  }
  output[0] = '\0';
  attribute = H5Aopen(object, name, H5P_DEFAULT);
  if (attribute < 0) {
    return 0;
  }
  type = H5Aget_type(attribute);
  if (type >= 0 && H5Tget_class(type) == H5T_STRING) {
    if (H5Tis_variable_str(type) > 0) {
      char *text = NULL;
      if (H5Aread(attribute, type, &text) >= 0 && text != NULL) {
        size_t size = strlen(text);
        if (size < capacity) {
          memcpy(output, text, size + 1U);
          read_ok = 1;
        }
        (void)H5free_memory(text);
      }
    } else {
      size_t size = H5Tget_size(type);
      if (size < capacity && H5Aread(attribute, type, output) >= 0) {
        output[size] = '\0';
        read_ok = 1;
      }
    }
  }
  if (type >= 0) {
    (void)H5Tclose(type);
  }
  (void)H5Aclose(attribute);
  return read_ok;
}

static int ss_string_attribute_equals(hid_t object, const char *name,
                                      const char *expected) {
  char text[256];
  return ss_string_attribute_read(object, name, text, sizeof(text)) != 0 &&
                 strcmp(text, expected) == 0
             ? 1
             : 0;
}

static int ss_integer_attribute_equals(hid_t object, const char *name,
                                       int64_t expected) {
  hid_t attribute = H5Aopen(object, name, H5P_DEFAULT);
  int64_t value = 0;
  int matches = 0;
  if (attribute >= 0 && H5Aread(attribute, H5T_NATIVE_INT64, &value) >= 0) {
    matches = value == expected ? 1 : 0;
  }
  if (attribute >= 0) {
    (void)H5Aclose(attribute);
  }
  return matches;
}

static int ss_lower_hex_digest(const char *text) {
  size_t index;
  if (text == NULL || strlen(text) != 64U) {
    return 0;
  }
  for (index = 0U; index < 64U; ++index) {
    if (!((text[index] >= '0' && text[index] <= '9') ||
          (text[index] >= 'a' && text[index] <= 'f'))) {
      return 0;
    }
  }
  return 1;
}

static int ss_collective_failure(MPI_Comm communicator, int local_status) {
  int global_status = 0;
  return MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX,
                       communicator) == MPI_SUCCESS
             ? global_status
             : 1;
}

SsStatus ss_io_hdf5_open(MPI_Comm communicator, const char *path, hid_t *file) {
  hid_t access = H5Pcreate(H5P_FILE_ACCESS);
  SsStatus status = SS_OK;
  *file = H5I_INVALID_HID;
  if (access < 0 || H5Pset_fapl_mpio(access, communicator, MPI_INFO_NULL) < 0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK && (*file = H5Fopen(path, H5F_ACC_RDONLY, access)) < 0) {
    status = SS_ERROR_HDF5;
  }
  if (access >= 0 && H5Pclose(access) < 0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK && *file >= 0) {
    (void)H5Fclose(*file);
    *file = H5I_INVALID_HID;
  }
  return status;
}

static SsStatus ss_hdf5_read_slab(MPI_Comm communicator, hid_t dataset,
                                  hid_t native_type, int dimensions,
                                  const hsize_t shape[2], int rank, int ranks,
                                  SsH5Rows *rows) {
  hid_t file_space = H5Dget_space(dataset);
  hid_t memory_space = H5I_INVALID_HID;
  hid_t transfer = H5I_INVALID_HID;
  size_t total = 0U;
  size_t values = 0U;
  size_t bytes = 0U;
  unsigned char empty = 0;
  SsStatus status = SS_OK;
  if (file_space < 0 || ss_checked_u64_to_size(shape[0], &total) != SS_OK ||
      ss_checked_u64_to_size(shape[1], &rows->width) != SS_OK) {
    status = SS_ERROR_HDF5;
  }
  if (status == SS_OK) {
    ss_partition_range(total, rank, ranks, &rows->start, &rows->count);
    rows->total = shape[0];
    status = ss_checked_mul_size(rows->count, rows->width, &values);
  }
  if (status == SS_OK) {
    status = ss_checked_mul_size(values, H5Tget_size(native_type), &bytes);
  }
  if (status == SS_OK) {
    const hsize_t offset[2] = {rows->start, 0U};
    const hsize_t count[2] = {rows->count, rows->width};
    const hsize_t memory[2] = {rows->count == 0U ? 1U : rows->count, rows->width};
    memory_space = H5Screate_simple(dimensions, memory, NULL);
    transfer = H5Pcreate(H5P_DATASET_XFER);
    if (memory_space < 0 || transfer < 0 ||
        H5Pset_dxpl_mpio(transfer, H5FD_MPIO_COLLECTIVE) < 0 ||
        (rows->count == 0U
             ? (H5Sselect_none(file_space) < 0 || H5Sselect_none(memory_space) < 0)
             : H5Sselect_hyperslab(file_space, H5S_SELECT_SET, offset, NULL, count,
                                   NULL) < 0)) {
      status = SS_ERROR_HDF5;
    }
  }
  if (status == SS_OK && bytes != 0U) {
    rows->values = malloc(bytes);
    if (rows->values == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK &&
      H5Dread(dataset, native_type, memory_space, file_space, transfer,
              rows->values == NULL ? &empty : rows->values) < 0) {
    status = SS_ERROR_HDF5;
  }
  if (file_space >= 0 && H5Sclose(file_space) < 0) {
    status = SS_ERROR_HDF5;
  }
  if (memory_space >= 0 && H5Sclose(memory_space) < 0) {
    status = SS_ERROR_HDF5;
  }
  if (transfer >= 0 && H5Pclose(transfer) < 0) {
    status = SS_ERROR_HDF5;
  }
  return ss_status_allreduce_max(communicator, status);
}

SsStatus ss_io_hdf5_read_rows(MPI_Comm communicator, hid_t file, const char *path,
                              hid_t native_type, int dimensions, size_t expected_width,
                              SsH5Rows *rows) {
  hid_t dataset = H5Dopen2(file, path, H5P_DEFAULT);
  hid_t space = dataset < 0 ? H5I_INVALID_HID : H5Dget_space(dataset);
  hid_t type = dataset < 0 ? H5I_INVALID_HID : H5Dget_type(dataset);
  hsize_t shape[2] = {0U, 1U};
  int rank = 0, ranks = 0;
  SsStatus status = SS_OK;
  memset(rows, 0, sizeof(*rows));
  if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(communicator, &ranks) != MPI_SUCCESS || ranks <= 0 ||
      (dimensions != 1 && dimensions != 2) || space < 0 || type < 0 ||
      H5Sget_simple_extent_ndims(space) != dimensions ||
      H5Sget_simple_extent_dims(space, shape, NULL) != dimensions ||
      H5Tequal(type, native_type) <= 0 || shape[1] != expected_width) {
    status = SS_ERROR_HDF5;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    status = ss_hdf5_read_slab(communicator, dataset, native_type, dimensions, shape,
                               rank, ranks, rows);
  }
  if (type >= 0 && H5Tclose(type) < 0) {
    status = SS_ERROR_HDF5;
  }
  if (space >= 0 && H5Sclose(space) < 0) {
    status = SS_ERROR_HDF5;
  }
  if (dataset >= 0 && H5Dclose(dataset) < 0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    free(rows->values);
    memset(rows, 0, sizeof(*rows));
  }
  return status;
}

static int ss_collective_dataset_read(MPI_Comm communicator, hid_t dataset,
                                      hid_t native_type, const char *value_type,
                                      int finite_values, int rank, int size,
                                      uint64_t *global_rows, uint64_t *global_values,
                                      int *zero_owner) {
  hid_t file_space = -1;
  hid_t file_type = -1;
  hsize_t dimensions[2] = {0U, 0U};
  SsH5Rows rows = {0};
  int status = 0;
  char logical_shape[64];
  char expected_shape[64];
  char digest[65];
  file_space = H5Dget_space(dataset);
  file_type = H5Dget_type(dataset);
  if (file_space < 0 || file_type < 0 || H5Sget_simple_extent_ndims(file_space) != 2 ||
      H5Sget_simple_extent_dims(file_space, dimensions, NULL) != 2 ||
      dimensions[0] == 0U || dimensions[1] == 0U ||
      H5Tequal(file_type, native_type) <= 0) {
    status = 1;
  }
  if (ss_collective_failure(communicator, status) != 0) {
    status = 1;
    goto close_objects;
  }
  if (snprintf(expected_shape, sizeof(expected_shape), "[%llu,%llu]",
               (unsigned long long)dimensions[0],
               (unsigned long long)dimensions[1]) < 0 ||
      !ss_string_attribute_equals(dataset, "layout_id", "compiled-problem-v1") ||
      !ss_string_attribute_equals(dataset, "digest_method",
                                  "sha256-dtype-shape-c-order") ||
      !ss_string_attribute_equals(dataset, "value_type", value_type) ||
      !ss_string_attribute_read(dataset, "logical_shape", logical_shape,
                                sizeof(logical_shape)) ||
      strcmp(logical_shape, expected_shape) != 0 ||
      !ss_string_attribute_read(dataset, "digest", digest, sizeof(digest)) ||
      !ss_lower_hex_digest(digest)) {
    status = 1;
  }
  if (ss_collective_failure(communicator, status) != 0) {
    status = 1;
    goto close_objects;
  }
  if (ss_hdf5_read_slab(communicator, dataset, native_type, 2, dimensions, rank, size,
                        &rows) != SS_OK) {
    status = 1;
    goto close_objects;
  }
  *zero_owner = rows.count == 0U;
  if (finite_values != 0) {
    const double *values = rows.values;
    for (size_t index = 0U; index < rows.count * rows.width; ++index) {
      if (!isfinite(values[index])) {
        status = 1;
        break;
      }
    }
  }
  if (ss_collective_failure(communicator, status) != 0) {
    status = 1;
    goto close_objects;
  }
  *global_rows = (uint64_t)dimensions[0];
  if ((uint64_t)dimensions[0] != 0U &&
      (uint64_t)dimensions[1] > UINT64_MAX / (uint64_t)dimensions[0]) {
    status = 1;
  } else {
    *global_values = (uint64_t)dimensions[0] * (uint64_t)dimensions[1];
  }

close_objects:
  free(rows.values);
  if (file_type >= 0 && H5Tclose(file_type) < 0) {
    status = 1;
  }
  if (file_space >= 0 && H5Sclose(file_space) < 0) {
    status = 1;
  }
  return status;
}

SsStatus ss_io_validate_hdf5(MPI_Comm communicator, const char *path,
                             SsH5Validation *validation, SsErrorRecord *error) {
  static const char *const group_paths[] = {
      "/mesh/topology",    "/mesh/entities",
      "/mesh/coordinates", "/mesh/supports",
      "/spaces",           "/materials",
      "/fields/initial",   "/provenance/import_maps",
  };
  hid_t groups[sizeof(group_paths) / sizeof(group_paths[0])];
  hid_t access = -1;
  hid_t file = -1;
  hid_t coordinates = -1;
  hid_t cells = -1;
  H5E_auto2_t prior_error_handler = NULL;
  void *prior_error_data = NULL;
  int rank = -1;
  int size = 0;
  int local_status = 0;
  int coordinate_zero = 0;
  int cell_zero = 0;
  int zero_owner_total = 0;
  uint64_t coordinate_rows = 0U;
  uint64_t coordinate_values = 0U;
  uint64_t cell_rows = 0U;
  uint64_t cell_values = 0U;
  size_t index;
  SsErrorRecord local_error = {0};
  SsErrorRecord selected = {0};
  if (path == NULL || validation == NULL || error == NULL ||
      ss_master_thread_guard() != SS_OK ||
      MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(communicator, &size) != MPI_SUCCESS || size < 1) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(validation, 0, sizeof(*validation));
  memset(groups, -1, sizeof(groups));
  (void)H5Eget_auto2(H5E_DEFAULT, &prior_error_handler, &prior_error_data);
  (void)H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
  access = H5Pcreate(H5P_FILE_ACCESS);
  if (access < 0 || H5Pset_fapl_mpio(access, communicator, MPI_INFO_NULL) < 0 ||
      H5Pset_all_coll_metadata_ops(access, 1U) < 0) {
    local_status = 1;
  }
  if (ss_collective_failure(communicator, local_status) != 0) {
    local_status = 1;
    goto close_objects;
  }
  file = H5Fopen(path, H5F_ACC_RDONLY, access);
  if (file < 0) {
    local_status = 1;
  }
  if (ss_collective_failure(communicator, local_status) != 0) {
    local_status = 1;
    goto close_objects;
  }
  {
    int layout_valid =
        ss_string_attribute_equals(file, "layout_id", "compiled-problem-v1");
    int schema_valid = ss_integer_attribute_equals(file, "schema_major", 1);
    if (layout_valid == 0 || schema_valid == 0) {
      local_status = 1;
    }
  }
  if (ss_collective_failure(communicator, local_status) != 0) {
    local_status = 1;
    goto close_objects;
  }
  for (index = 0U; index < sizeof(groups) / sizeof(groups[0]); ++index) {
    groups[index] = H5Gopen2(file, group_paths[index], H5P_DEFAULT);
    if (groups[index] < 0) {
      local_status = 1;
    }
    if (ss_collective_failure(communicator, local_status) != 0) {
      local_status = 1;
      goto close_objects;
    }
  }
  coordinates = H5Dopen2(file, "/mesh/coordinates/values", H5P_DEFAULT);
  cells = H5Dopen2(file, "/mesh/topology/cells", H5P_DEFAULT);
  if (coordinates < 0 || cells < 0) {
    local_status = 1;
  }
  if (ss_collective_failure(communicator, local_status) != 0) {
    local_status = 1;
    goto close_objects;
  }
  if (ss_collective_dataset_read(communicator, coordinates, H5T_NATIVE_DOUBLE,
                                 "float64", 1, rank, size, &coordinate_rows,
                                 &coordinate_values, &coordinate_zero) != 0) {
    local_status = 1;
    goto close_objects;
  }
  if (ss_collective_dataset_read(communicator, cells, H5T_NATIVE_UINT64, "uint64", 0,
                                 rank, size, &cell_rows, &cell_values,
                                 &cell_zero) != 0) {
    local_status = 1;
    goto close_objects;
  }

close_objects:
  if (cells >= 0 && H5Dclose(cells) < 0) {
    local_status = 1;
  }
  if (coordinates >= 0 && H5Dclose(coordinates) < 0) {
    local_status = 1;
  }
  for (index = sizeof(groups) / sizeof(groups[0]); index > 0U; --index) {
    if (groups[index - 1U] >= 0 && H5Gclose(groups[index - 1U]) < 0) {
      local_status = 1;
    }
  }
  if (file >= 0 && H5Fclose(file) < 0) {
    local_status = 1;
  }
  if (access >= 0 && H5Pclose(access) < 0) {
    local_status = 1;
  }
  (void)H5Eset_auto2(H5E_DEFAULT, prior_error_handler, prior_error_data);
  if (local_status != 0) {
    ss_hdf5_error(&local_error, rank,
                  "compiled-problem HDF5 layout or collective read failed");
  }
  if (ss_collective_error_select(communicator, &local_error, &selected) != SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  if (selected.valid != 0) {
    *error = selected;
    return SS_ERROR_HDF5;
  }
  if (MPI_Allreduce(&cell_zero, &zero_owner_total, 1, MPI_INT, MPI_SUM, communicator) !=
      MPI_SUCCESS) {
    ss_hdf5_error(error, rank, "collective HDF5 status agreement failed");
    return SS_ERROR_HDF5;
  }
  (void)coordinate_rows;
  (void)coordinate_zero;
  (void)cell_values;
  validation->coordinate_values = coordinate_values;
  validation->cell_rows = cell_rows;
  validation->zero_owner_ranks = (uint32_t)zero_owner_total;
  return SS_OK;
}
