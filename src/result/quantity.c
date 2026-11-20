/* Result quantity datasets: creation, distributed appends and finalization.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "result_internal.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *ss_result_quantity_layout_name(SsResultDatasetKind kind) {
  if (kind == SS_RESULT_DATASET_QUADRATURE) {
    return "state-entity-quadrature-component-v1";
  }
  if (kind == SS_RESULT_DATASET_QUADRATURE_CATEGORY) {
    return "state-entity-quadrature-v1";
  }
  return "state-entity-component-v1";
}

static SsStatus ss_result_quantity_upsert_string_attribute(hid_t object,
                                                           const char *name,
                                                           const char *value) {
  const htri_t exists = H5Aexists(object, name);
  if (exists < 0 || (exists > 0 && H5Adelete(object, name) < 0)) {
    return SS_ERROR_HDF5;
  }
  return ss_result_hdf5_string_attribute(object, name, value);
}

static SsStatus ss_result_quantity_upsert_u64_attribute(hid_t object, const char *name,
                                                        uint64_t value) {
  const htri_t exists = H5Aexists(object, name);
  if (exists < 0 || (exists > 0 && H5Adelete(object, name) < 0)) {
    return SS_ERROR_HDF5;
  }
  return ss_result_hdf5_u64_attribute(object, name, value);
}

static SsStatus ss_result_quantity_write_ids(hid_t ids_dataset, MPI_Comm communicator,
                                             const SsResultFileIndexPlan *plan) {
  hid_t file_space = -1;
  hid_t memory_space = -1;
  hid_t transfer = -1;
  hsize_t memory_dimensions[1] = {
      plan->sorted_count == 0U ? 1U : (hsize_t)plan->sorted_count};
  hsize_t start[1] = {(hsize_t)plan->sorted_offset};
  hsize_t count[1] = {(hsize_t)plan->sorted_count};
  SsStatus status = SS_OK;
  file_space = H5Dget_space(ids_dataset);
  memory_space = H5Screate_simple(1, memory_dimensions, NULL);
  transfer = H5Pcreate(H5P_DATASET_XFER);
  if (file_space < 0 || memory_space < 0 || transfer < 0 ||
      H5Pset_dxpl_mpio(transfer, H5FD_MPIO_COLLECTIVE) < 0) {
    status = SS_ERROR_HDF5;
  } else if (plan->sorted_count == 0U) {
    if (H5Sselect_none(file_space) < 0 || H5Sselect_none(memory_space) < 0) {
      status = SS_ERROR_HDF5;
    }
  } else if (H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL) <
             0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK &&
      H5Dwrite(ids_dataset, H5T_NATIVE_UINT64, memory_space, file_space, transfer,
               plan->sorted_count == 0U ? NULL : plan->sorted_ids) < 0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_result_hdf5_close(&transfer, H5Pclose, status);
  status = ss_result_hdf5_close(&memory_space, H5Sclose, status);
  status = ss_result_hdf5_close(&file_space, H5Sclose, status);
  return ss_status_allreduce_max(communicator, status);
}

static SsStatus ss_result_quantity_write_state_id(hid_t states_dataset,
                                                  MPI_Comm communicator,
                                                  uint64_t state_index,
                                                  SsStateId state_id) {
  int rank = -1;
  hid_t file_space = -1;
  hid_t memory_space = -1;
  hid_t transfer = -1;
  hsize_t memory_dimensions[1] = {1U};
  hsize_t start[1] = {(hsize_t)state_index};
  hsize_t count[1] = {1U};
  SsStatus status = SS_OK;
  if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  file_space = H5Dget_space(states_dataset);
  memory_space = H5Screate_simple(1, memory_dimensions, NULL);
  transfer = H5Pcreate(H5P_DATASET_XFER);
  if (file_space < 0 || memory_space < 0 || transfer < 0 ||
      H5Pset_dxpl_mpio(transfer, H5FD_MPIO_COLLECTIVE) < 0) {
    status = SS_ERROR_HDF5;
  } else if (rank != 0) {
    if (H5Sselect_none(file_space) < 0 || H5Sselect_none(memory_space) < 0) {
      status = SS_ERROR_HDF5;
    }
  } else if (H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL) <
             0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK &&
      H5Dwrite(states_dataset, H5T_NATIVE_UINT64, memory_space, file_space, transfer,
               rank == 0 ? &state_id : NULL) < 0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_result_hdf5_close(&transfer, H5Pclose, status);
  status = ss_result_hdf5_close(&memory_space, H5Sclose, status);
  status = ss_result_hdf5_close(&file_space, H5Sclose, status);
  return ss_status_allreduce_max(communicator, status);
}

static SsStatus
ss_result_quantity_write_values(hid_t values_dataset, MPI_Comm communicator,
                                const SsResultFileIndexPlan *plan, uint64_t state_index,
                                uint32_t sample_count, uint32_t component_count,
                                const double *local_values, int value_rank) {
  void *sorted_values = NULL;
  hid_t file_space = -1;
  hid_t memory_space = -1;
  hid_t transfer = -1;
  hsize_t dimensions[4] = {1U, 1U, 1U, 1U};
  hsize_t start[4] = {(hsize_t)state_index, 0U, 0U, 0U};
  size_t values_per_entity;
  size_t value_count;
  size_t index;
  SsStatus status = SS_OK;
  if (values_dataset < 0 || plan == NULL || sample_count == 0U ||
      component_count == 0U || (value_rank != 3 && value_rank != 4) ||
      (plan->local_count != 0U && local_values == NULL)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (sample_count > SIZE_MAX / component_count) {
    return SS_ERROR_OVERFLOW;
  }
  values_per_entity = (size_t)sample_count * (size_t)component_count;
  if (plan->local_count > SIZE_MAX / values_per_entity) {
    return SS_ERROR_OVERFLOW;
  }
  value_count = plan->local_count * values_per_entity;
  start[1] = (hsize_t)plan->sorted_offset;
  for (index = 0U; index < value_count; ++index) {
    if (!isfinite(local_values[index])) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    status = ss_result_gather_indexed_rows(plan, values_per_entity, sizeof(double),
                                           MPI_DOUBLE, local_values, &sorted_values);
  }
  dimensions[1] = plan->sorted_count == 0U ? 1U : (hsize_t)plan->sorted_count;
  if (value_rank == 4) {
    dimensions[2] = (hsize_t)sample_count;
    dimensions[3] = (hsize_t)component_count;
  } else {
    dimensions[2] = (hsize_t)component_count;
  }
  if (status == SS_OK) {
    file_space = H5Dget_space(values_dataset);
    memory_space = H5Screate_simple(value_rank, dimensions, NULL);
    transfer = H5Pcreate(H5P_DATASET_XFER);
    if (file_space < 0 || memory_space < 0 || transfer < 0 ||
        H5Pset_dxpl_mpio(transfer, H5FD_MPIO_COLLECTIVE) < 0) {
      status = SS_ERROR_HDF5;
    } else if (plan->sorted_count == 0U) {
      if (H5Sselect_none(file_space) < 0 || H5Sselect_none(memory_space) < 0) {
        status = SS_ERROR_HDF5;
      }
    } else if (H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, dimensions,
                                   NULL) < 0) {
      status = SS_ERROR_HDF5;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK &&
      H5Dwrite(values_dataset, H5T_NATIVE_DOUBLE, memory_space, file_space, transfer,
               plan->sorted_count == 0U ? NULL : sorted_values) < 0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_result_hdf5_close(&transfer, H5Pclose, status);
  status = ss_result_hdf5_close(&memory_space, H5Sclose, status);
  status = ss_result_hdf5_close(&file_space, H5Sclose, status);
  free(sorted_values);
  return ss_status_allreduce_max(communicator, status);
}

static SsStatus
ss_result_category_write_values(hid_t values_dataset, MPI_Comm communicator,
                                const SsResultFileIndexPlan *plan, uint64_t state_index,
                                uint32_t sample_count, uint32_t category_count,
                                const uint32_t *local_values) {
  void *sorted_values = NULL;
  hid_t file_space = -1;
  hid_t memory_space = -1;
  hid_t transfer = -1;
  hsize_t dimensions[3] = {1U, 1U, (hsize_t)sample_count};
  hsize_t start[3] = {(hsize_t)state_index, 0U, 0U};
  size_t value_count;
  size_t index;
  SsStatus status = SS_OK;
  if (values_dataset < 0 || plan == NULL || sample_count == 0U ||
      category_count == 0U || (plan->local_count != 0U && local_values == NULL)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (plan->local_count > SIZE_MAX / (size_t)sample_count) {
    return SS_ERROR_OVERFLOW;
  }
  value_count = plan->local_count * (size_t)sample_count;
  start[1] = (hsize_t)plan->sorted_offset;
  for (index = 0U; index < value_count; ++index) {
    if (local_values[index] >= category_count) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    status = ss_result_gather_indexed_rows(plan, (size_t)sample_count, sizeof(uint32_t),
                                           MPI_UINT32_T, local_values, &sorted_values);
  }
  dimensions[1] = plan->sorted_count == 0U ? 1U : (hsize_t)plan->sorted_count;
  if (status == SS_OK) {
    file_space = H5Dget_space(values_dataset);
    memory_space = H5Screate_simple(3, dimensions, NULL);
    transfer = H5Pcreate(H5P_DATASET_XFER);
    if (file_space < 0 || memory_space < 0 || transfer < 0 ||
        H5Pset_dxpl_mpio(transfer, H5FD_MPIO_COLLECTIVE) < 0) {
      status = SS_ERROR_HDF5;
    } else if (plan->sorted_count == 0U) {
      if (H5Sselect_none(file_space) < 0 || H5Sselect_none(memory_space) < 0) {
        status = SS_ERROR_HDF5;
      }
    } else if (H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, dimensions,
                                   NULL) < 0) {
      status = SS_ERROR_HDF5;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK &&
      H5Dwrite(values_dataset, H5T_NATIVE_UINT32, memory_space, file_space, transfer,
               plan->sorted_count == 0U ? NULL : sorted_values) < 0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_result_hdf5_close(&transfer, H5Pclose, status);
  status = ss_result_hdf5_close(&memory_space, H5Sclose, status);
  status = ss_result_hdf5_close(&file_space, H5Sclose, status);
  free(sorted_values);
  return ss_status_allreduce_max(communicator, status);
}

static SsStatus ss_result_quantity_create(
    hid_t file, MPI_Comm communicator, const char *group_path, SsResultDatasetKind kind,
    uint64_t global_count, uint32_t sample_count, uint32_t component_count,
    uint64_t dataset_chunk_rows, const SsResultFileIndexPlan *plan) {
  char ids_path[SS_RESULT_PATH_CAPACITY];
  char states_path[SS_RESULT_PATH_CAPACITY];
  char values_path[SS_RESULT_PATH_CAPACITY];
  hid_t group = -1;
  hid_t ids_space = -1;
  hid_t states_space = -1;
  hid_t values_space = -1;
  hid_t ids_creation = -1;
  hid_t states_creation = -1;
  hid_t values_creation = -1;
  hid_t ids_dataset = -1;
  hid_t states_dataset = -1;
  hid_t values_dataset = -1;
  hsize_t global_rows;
  hsize_t ids_dimensions[1];
  hsize_t state_dimensions[1] = {0U};
  hsize_t state_maximum[1] = {H5S_UNLIMITED};
  hsize_t state_chunk[1] = {1U};
  hsize_t value_dimensions[4] = {0U, 0U, 0U, 0U};
  hsize_t value_maximum[4] = {H5S_UNLIMITED, 0U, 0U, 0U};
  hsize_t value_chunk[4] = {1U, 0U, 0U, 0U};
  hsize_t ids_chunk[1];
  int value_rank = kind == SS_RESULT_DATASET_QUADRATURE ? 4 : 3;
  hid_t value_type =
      kind == SS_RESULT_DATASET_QUADRATURE_CATEGORY ? H5T_STD_U32LE : H5T_IEEE_F64LE;
  SsStatus status = SS_OK;
  if (file < 0 || group_path == NULL || plan == NULL || global_count == 0U ||
      sample_count == 0U || component_count == 0U ||
      ss_checked_u64_to_hsize(global_count, &global_rows) != SS_OK ||
      snprintf(ids_path, sizeof(ids_path), "%s/stable_ids", group_path) < 0 ||
      snprintf(states_path, sizeof(states_path), "%s/state_ids", group_path) < 0 ||
      snprintf(values_path, sizeof(values_path), "%s/values", group_path) < 0) {
    return SS_ERROR_OVERFLOW;
  }
  ids_dimensions[0] = global_rows;
  value_dimensions[1] = global_rows;
  value_dimensions[2] = value_rank == 4 || kind == SS_RESULT_DATASET_QUADRATURE_CATEGORY
                            ? (hsize_t)sample_count
                            : (hsize_t)component_count;
  value_dimensions[3] = (hsize_t)component_count;
  value_maximum[1] = global_rows;
  value_maximum[2] = value_dimensions[2];
  value_maximum[3] = value_dimensions[3];
  value_chunk[1] =
      (hsize_t)(dataset_chunk_rows == 0U
                    ? (global_count < SS_RESULT_QUANTITY_DIGEST_CHUNK_ROWS
                           ? global_count
                           : SS_RESULT_QUANTITY_DIGEST_CHUNK_ROWS)
                    : (dataset_chunk_rows < global_count ? dataset_chunk_rows
                                                         : global_count));
  value_chunk[2] = value_dimensions[2];
  value_chunk[3] = value_dimensions[3];
  ids_chunk[0] = value_chunk[1];
  group = H5Gcreate2(file, group_path, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ids_space = H5Screate_simple(1, ids_dimensions, NULL);
  states_space = H5Screate_simple(1, state_dimensions, state_maximum);
  values_space = H5Screate_simple(value_rank, value_dimensions, value_maximum);
  ids_creation = H5Pcreate(H5P_DATASET_CREATE);
  states_creation = H5Pcreate(H5P_DATASET_CREATE);
  values_creation = H5Pcreate(H5P_DATASET_CREATE);
  if (group < 0 || ids_space < 0 || states_space < 0 || values_space < 0 ||
      ids_creation < 0 || states_creation < 0 || values_creation < 0 ||
      H5Pset_chunk(ids_creation, 1, ids_chunk) < 0 ||
      H5Pset_chunk(states_creation, 1, state_chunk) < 0 ||
      H5Pset_chunk(values_creation, value_rank, value_chunk) < 0) {
    status = SS_ERROR_HDF5;
  }
  if (status == SS_OK) {
    ids_dataset = H5Dcreate2(file, ids_path, H5T_STD_U64LE, ids_space, H5P_DEFAULT,
                             ids_creation, H5P_DEFAULT);
    states_dataset = H5Dcreate2(file, states_path, H5T_STD_U64LE, states_space,
                                H5P_DEFAULT, states_creation, H5P_DEFAULT);
    values_dataset = H5Dcreate2(file, values_path, value_type, values_space,
                                H5P_DEFAULT, values_creation, H5P_DEFAULT);
    if (ids_dataset < 0 || states_dataset < 0 || values_dataset < 0) {
      status = SS_ERROR_HDF5;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    status = ss_result_quantity_write_ids(ids_dataset, communicator, plan);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_string_attribute(values_dataset, "dataset_kind",
                                             ss_result_quantity_kind_name(kind));
  }
  status = ss_status_allreduce_max(communicator, status);
  status = ss_result_hdf5_close(&values_dataset, H5Dclose, status);
  status = ss_result_hdf5_close(&states_dataset, H5Dclose, status);
  status = ss_result_hdf5_close(&ids_dataset, H5Dclose, status);
  status = ss_result_hdf5_close(&values_creation, H5Pclose, status);
  status = ss_result_hdf5_close(&states_creation, H5Pclose, status);
  status = ss_result_hdf5_close(&ids_creation, H5Pclose, status);
  status = ss_result_hdf5_close(&values_space, H5Sclose, status);
  status = ss_result_hdf5_close(&states_space, H5Sclose, status);
  status = ss_result_hdf5_close(&ids_space, H5Sclose, status);
  status = ss_result_hdf5_close(&group, H5Gclose, status);
  return ss_status_allreduce_max(communicator, status);
}

SsStatus ss_result_quantity_declare(hid_t file, MPI_Comm communicator,
                                    const char *group_path,
                                    SsResultDatasetKind dataset_kind,
                                    uint32_t sample_count, uint32_t component_count,
                                    uint64_t dataset_chunk_rows,
                                    const SsResultFileIndexPlan *plan) {
  if (file < 0 || group_path == NULL || plan == NULL || sample_count == 0U ||
      component_count == 0U ||
      (dataset_kind != SS_RESULT_DATASET_FIELD &&
       dataset_kind != SS_RESULT_DATASET_QUADRATURE &&
       dataset_kind != SS_RESULT_DATASET_CELL_REDUCTION &&
       dataset_kind != SS_RESULT_DATASET_QUADRATURE_CATEGORY) ||
      (dataset_kind != SS_RESULT_DATASET_QUADRATURE &&
       dataset_kind != SS_RESULT_DATASET_QUADRATURE_CATEGORY && sample_count != 1U)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return ss_result_quantity_create(file, communicator, group_path, dataset_kind,
                                   plan->summary.global_count, sample_count,
                                   component_count, dataset_chunk_rows, plan);
}

static SsStatus ss_result_quantity_attributes(
    hid_t dataset, SsResultDatasetKind kind, const char *quantity_id,
    uint32_t quantity_version, const char *unit, const char *location,
    const char *basis, const char *quadrature_id, const char *mesh_id,
    const char *geometry_id, uint32_t field_order, uint32_t geometry_order,
    const char *semantic_ids, const char *states, const char *digest,
    const SsResultQuantityShape *shape, const char *value_type) {
  char logical_shape[128];
  const char *kind_name = ss_result_quantity_kind_name(kind);
  int count;
  SsStatus status;
  if (kind_name == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (kind == SS_RESULT_DATASET_QUADRATURE) {
    count =
        snprintf(logical_shape, sizeof(logical_shape), "[%llu,%llu,%u,%u]",
                 (unsigned long long)shape->states, (unsigned long long)shape->entities,
                 shape->samples, shape->components);
  } else if (kind == SS_RESULT_DATASET_QUADRATURE_CATEGORY) {
    count = snprintf(logical_shape, sizeof(logical_shape), "[%llu,%llu,%u]",
                     (unsigned long long)shape->states,
                     (unsigned long long)shape->entities, shape->samples);
  } else {
    count = snprintf(logical_shape, sizeof(logical_shape), "[%llu,%llu,%u]",
                     (unsigned long long)shape->states,
                     (unsigned long long)shape->entities, shape->components);
  }
  if (count < 0 || (size_t)count >= sizeof(logical_shape)) {
    return SS_ERROR_OVERFLOW;
  }
  status =
      ss_result_quantity_upsert_string_attribute(dataset, "quantity_id", quantity_id);
  if (status == SS_OK) {
    status = ss_result_quantity_upsert_u64_attribute(dataset, "quantity_version",
                                                     quantity_version);
  }
  if (status == SS_OK) {
    status = ss_result_quantity_upsert_string_attribute(dataset, "unit", unit);
  }
  if (status == SS_OK) {
    status = ss_result_quantity_upsert_string_attribute(dataset, "mesh_id", mesh_id);
  }
  if (status == SS_OK) {
    status =
        ss_result_quantity_upsert_string_attribute(dataset, "geometry_id", geometry_id);
  }
  if (status == SS_OK) {
    status = ss_result_quantity_upsert_string_attribute(dataset, "state_ids", states);
  }
  if (status == SS_OK) {
    status =
        ss_result_quantity_upsert_u64_attribute(dataset, "state_count", shape->states);
  }
  if (status == SS_OK) {
    status =
        ss_result_quantity_upsert_string_attribute(dataset, "completeness", "complete");
  }
  if (status == SS_OK) {
    status =
        ss_result_quantity_upsert_string_attribute(dataset, "value_type", value_type);
  }
  if (status == SS_OK) {
    status = ss_result_quantity_upsert_string_attribute(
        dataset, "digest_method",
        kind == SS_RESULT_DATASET_QUADRATURE_CATEGORY
            ? "sha256-semantic-stable-id-state-sample-uint32le-chunks-v1"
            : "sha256-semantic-stable-id-state-sample-float64le-chunks-v2");
  }
  if (status == SS_OK) {
    status = ss_result_quantity_upsert_string_attribute(dataset, "digest", digest);
  }
  if (status == SS_OK) {
    status = ss_result_quantity_upsert_string_attribute(dataset, "location", location);
  }
  if (status == SS_OK) {
    status = ss_result_quantity_upsert_string_attribute(dataset, "basis", basis);
  }
  if (status == SS_OK) {
    status = ss_result_quantity_upsert_string_attribute(dataset, "quadrature_id",
                                                        quadrature_id);
  }
  if (status == SS_OK) {
    status =
        ss_result_quantity_upsert_string_attribute(dataset, "dataset_kind", kind_name);
  }
  if (status == SS_OK) {
    status = ss_result_quantity_upsert_u64_attribute(dataset, "sample_count",
                                                     shape->samples);
  }
  if (status == SS_OK) {
    status =
        ss_result_quantity_upsert_u64_attribute(dataset, "field_order", field_order);
  }
  if (status == SS_OK) {
    status = ss_result_quantity_upsert_u64_attribute(dataset, "geometry_order",
                                                     geometry_order);
  }
  if (status == SS_OK) {
    status = ss_result_quantity_upsert_string_attribute(
        dataset,
        kind == SS_RESULT_DATASET_QUADRATURE_CATEGORY ? "category_ids"
                                                      : "component_ids",
        semantic_ids);
  }
  if (status == SS_OK) {
    status = ss_result_quantity_upsert_string_attribute(dataset, "logical_shape",
                                                        logical_shape);
  }
  if (status == SS_OK) {
    status = ss_result_quantity_upsert_string_attribute(
        dataset, "logical_layout_id", ss_result_quantity_layout_name(kind));
  }
  if (status == SS_OK) {
    status = ss_result_quantity_upsert_u64_attribute(
        dataset, "value_precision_bits", strcmp(value_type, "uint32") == 0 ? 32U : 64U);
  }
  if (status == SS_OK) {
    status = ss_result_quantity_upsert_string_attribute(dataset, "stable_id_mapping",
                                                        "canonical_ascending_v1");
  }
  return status;
}

static SsStatus ss_result_quantity_open(hid_t file, const char *group_path,
                                        hid_t *ids_dataset, hid_t *states_dataset,
                                        hid_t *values_dataset) {
  char ids_path[SS_RESULT_PATH_CAPACITY];
  char states_path[SS_RESULT_PATH_CAPACITY];
  char values_path[SS_RESULT_PATH_CAPACITY];
  const int ids_written =
      snprintf(ids_path, sizeof(ids_path), "%s/stable_ids", group_path);
  const int states_written =
      snprintf(states_path, sizeof(states_path), "%s/state_ids", group_path);
  const int values_written =
      snprintf(values_path, sizeof(values_path), "%s/values", group_path);
  if (ids_written < 0 || (size_t)ids_written >= sizeof(ids_path) ||
      states_written < 0 || (size_t)states_written >= sizeof(states_path) ||
      values_written < 0 || (size_t)values_written >= sizeof(values_path)) {
    return SS_ERROR_OVERFLOW;
  }
  *ids_dataset = H5Dopen2(file, ids_path, H5P_DEFAULT);
  *states_dataset = H5Dopen2(file, states_path, H5P_DEFAULT);
  *values_dataset = H5Dopen2(file, values_path, H5P_DEFAULT);
  return *ids_dataset < 0 || *states_dataset < 0 || *values_dataset < 0 ? SS_ERROR_HDF5
                                                                        : SS_OK;
}

/* Extend the state axis by one and write the state ID; no prior state is read. */
static SsStatus ss_result_quantity_extend(hid_t states_dataset, hid_t values_dataset,
                                          MPI_Comm communicator,
                                          const SsResultFileIndexPlan *plan,
                                          uint64_t state_index, SsStateId state_id,
                                          uint32_t axis2, uint32_t axis3) {
  hsize_t state_extent[1] = {(hsize_t)(state_index + UINT64_C(1))};
  hsize_t value_extent[4] = {state_extent[0], (hsize_t)plan->summary.global_count,
                             (hsize_t)axis2, (hsize_t)axis3};
  SsStatus status = SS_OK;
  if (H5Dset_extent(states_dataset, state_extent) < 0 ||
      H5Dset_extent(values_dataset, value_extent) < 0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    status = ss_result_quantity_write_state_id(states_dataset, communicator,
                                               state_index, state_id);
  }
  return status;
}

SsStatus ss_result_quantity_append(hid_t file, MPI_Comm communicator,
                                   const char *group_path,
                                   SsResultDatasetKind dataset_kind, SsStateId state_id,
                                   uint32_t component_count, uint32_t sample_count,
                                   const SsResultFileIndexPlan *plan,
                                   const double *local_values, uint64_t state_index,
                                   uint64_t *logical_bytes) {
  hid_t states_dataset = -1;
  hid_t values_dataset = -1;
  hid_t ids_dataset = -1;
  const int value_rank = dataset_kind == SS_RESULT_DATASET_QUADRATURE ? 4 : 3;
  SsStatus status;
  if (file < 0 || group_path == NULL || state_id == 0U || component_count == 0U ||
      component_count > SS_RESULT_COMPONENT_CAPACITY || sample_count == 0U ||
      plan == NULL || logical_bytes == NULL ||
      (plan->local_count != 0U && local_values == NULL) ||
      ss_result_quantity_kind_name(dataset_kind) == NULL ||
      dataset_kind == SS_RESULT_DATASET_QUADRATURE_CATEGORY ||
      (dataset_kind != SS_RESULT_DATASET_QUADRATURE && sample_count != 1U)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_result_quantity_open(file, group_path, &ids_dataset, &states_dataset,
                                   &values_dataset);
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    status = ss_result_quantity_extend(
        states_dataset, values_dataset, communicator, plan, state_index, state_id,
        value_rank == 4 ? sample_count : component_count, component_count);
  }
  if (status == SS_OK) {
    status = ss_result_quantity_write_values(values_dataset, communicator, plan,
                                             state_index, sample_count, component_count,
                                             local_values, value_rank);
  }
  if (status == SS_OK) {
    const uint64_t states = state_index + UINT64_C(1);
    const uint64_t entities = plan->summary.global_count;
    const uint64_t per_state = entities * (uint64_t)sample_count * component_count;
    if (entities > UINT64_MAX / ((uint64_t)sample_count * component_count) ||
        per_state > (UINT64_MAX / sizeof(uint64_t) - entities - states) / states) {
      status = SS_ERROR_OVERFLOW;
    } else {
      *logical_bytes = (entities + states + states * per_state) * sizeof(uint64_t);
    }
  }
  status = ss_result_hdf5_close(&values_dataset, H5Dclose, status);
  status = ss_result_hdf5_close(&states_dataset, H5Dclose, status);
  status = ss_result_hdf5_close(&ids_dataset, H5Dclose, status);
  return ss_status_allreduce_max(communicator, status);
}

SsStatus ss_result_category_append(hid_t file, MPI_Comm communicator,
                                   const char *group_path, SsStateId state_id,
                                   uint32_t category_count, uint32_t sample_count,
                                   const SsResultFileIndexPlan *plan,
                                   const uint32_t *local_values, uint64_t state_index,
                                   uint64_t *logical_bytes) {
  hid_t ids_dataset = -1;
  hid_t states_dataset = -1;
  hid_t values_dataset = -1;
  SsStatus status;
  if (file < 0 || group_path == NULL || state_id == 0U || category_count == 0U ||
      category_count > SS_RESULT_COMPONENT_CAPACITY || sample_count == 0U ||
      plan == NULL || logical_bytes == NULL ||
      (plan->local_count != 0U && local_values == NULL)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_result_quantity_open(file, group_path, &ids_dataset, &states_dataset,
                                   &values_dataset);
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    status = ss_result_quantity_extend(states_dataset, values_dataset, communicator,
                                       plan, state_index, state_id, sample_count, 1U);
  }
  if (status == SS_OK) {
    status =
        ss_result_category_write_values(values_dataset, communicator, plan, state_index,
                                        sample_count, category_count, local_values);
  }
  if (status == SS_OK) {
    const uint64_t states = state_index + UINT64_C(1);
    const uint64_t entities = plan->summary.global_count;
    const uint64_t per_state = entities * (uint64_t)sample_count;
    if (entities > UINT64_MAX / sample_count ||
        per_state > (UINT64_MAX / sizeof(uint64_t) - entities - states) / states) {
      status = SS_ERROR_OVERFLOW;
    } else {
      *logical_bytes = (entities + states) * sizeof(uint64_t) +
                       states * per_state * sizeof(uint32_t);
    }
  }
  status = ss_result_hdf5_close(&values_dataset, H5Dclose, status);
  status = ss_result_hdf5_close(&states_dataset, H5Dclose, status);
  status = ss_result_hdf5_close(&ids_dataset, H5Dclose, status);
  return ss_status_allreduce_max(communicator, status);
}

SsStatus ss_result_quantity_finalize(hid_t file, MPI_Comm communicator,
                                     const char *group_path,
                                     const SsResultQuantityFinal *final,
                                     char logical_digest[SS_SHA256_HEX_CAPACITY + 8U]) {
  char state_text[SS_RESULT_STATE_TEXT_CAPACITY];
  char placeholder[SS_SHA256_HEX_CAPACITY];
  SsResultQuantityShape shape;
  hid_t ids_dataset = -1;
  hid_t states_dataset = -1;
  hid_t values_dataset = -1;
  const int category =
      final != NULL && final->kind == SS_RESULT_DATASET_QUADRATURE_CATEGORY;
  SsStatus status;
  if (file < 0 || group_path == NULL || final == NULL || logical_digest == NULL ||
      ss_result_quantity_kind_name(final->kind) == NULL ||
      final->semantic_ids == NULL || final->state_ids == NULL ||
      final->state_count == 0U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_result_quantity_state_text(final->state_ids, final->state_count,
                                         state_text, sizeof(state_text));
  if (status == SS_OK) {
    status = ss_result_quantity_open(file, group_path, &ids_dataset, &states_dataset,
                                     &values_dataset);
  }
  if (status == SS_OK) {
    status =
        ss_result_quantity_shape(ids_dataset, states_dataset, values_dataset, &shape);
  }
  if (status == SS_OK && shape.states != final->state_count) {
    status = SS_ERROR_INTEGRITY;
  }
  status = ss_status_allreduce_max(communicator, status);
  memset(placeholder, '0', SS_SHA256_HEX_CAPACITY - 1U);
  placeholder[SS_SHA256_HEX_CAPACITY - 1U] = '\0';
  if (status == SS_OK) {
    status = ss_result_quantity_attributes(
        values_dataset, final->kind, final->quantity_id, final->quantity_version,
        final->unit, final->location, final->basis, final->quadrature_id,
        final->mesh_id, final->geometry_id, final->field_order, final->geometry_order,
        final->semantic_ids, state_text, placeholder, &shape,
        category ? "uint32" : "float64");
  }
  if (status == SS_OK) {
    status = ss_result_quantity_attributes(
        ids_dataset, final->kind, final->quantity_id, final->quantity_version,
        "dimensionless", "stable_identity", "none", final->quadrature_id,
        final->mesh_id, final->geometry_id, final->field_order, final->geometry_order,
        category ? final->semantic_ids : "stable_id", state_text, placeholder, &shape,
        "uint64");
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    status = ss_result_quantity_digest(ids_dataset, states_dataset, values_dataset,
                                       communicator, logical_digest, NULL);
  }
  if (status == SS_OK) {
    status = ss_result_quantity_upsert_string_attribute(values_dataset, "digest",
                                                        logical_digest + 7U);
  }
  if (status == SS_OK) {
    status = ss_result_quantity_upsert_string_attribute(ids_dataset, "digest",
                                                        logical_digest + 7U);
  }
  status = ss_result_hdf5_close(&values_dataset, H5Dclose, status);
  status = ss_result_hdf5_close(&states_dataset, H5Dclose, status);
  status = ss_result_hdf5_close(&ids_dataset, H5Dclose, status);
  return ss_status_allreduce_max(communicator, status);
}
