/* Read and write checkpoint HDF5 fields and replicated records.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "checkpoint_internal.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define SS_CHECKPOINT_REPLICATED_VALUE_CAPACITY SS_RESULT_COMPONENT_CAPACITY

SsStatus ss_checkpoint_write_replicated_double(
    SsCheckpointTransaction *transaction, const char *path, const double *values,
    size_t count, const char *semantic_id, char digest[SS_SHA256_HEX_CAPACITY + 8U]) {
  hid_t space = -1;
  hid_t memory_space = -1;
  hid_t dataset = -1;
  hid_t file_space = -1;
  hid_t transfer = -1;
  hsize_t dimensions[1] = {(hsize_t)count};
  hsize_t memory_dimensions[1] = {count == 0U ? 1U : (hsize_t)count};
  unsigned char encoded[SS_CHECKPOINT_REPLICATED_VALUE_CAPACITY * sizeof(double)];
  char raw[SS_SHA256_HEX_CAPACITY];
  size_t index;
  SsStatus status = SS_OK;
  if (count > SS_CHECKPOINT_REPLICATED_VALUE_CAPACITY ||
      (count != 0U && values == NULL)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (transaction->rank == 0) {
    for (index = 0U; index < count; ++index) {
      if (!isfinite(values[index])) {
        status = SS_ERROR_INTEGRITY;
        break;
      }
      ss_result_store_double_le(encoded + index * sizeof(double), values[index]);
    }
    if (status == SS_OK) {
      memcpy(digest, "sha256:", 7U);
      status = ss_sha256_bytes(encoded, count * sizeof(double), digest + 7U);
    }
  }
  {
    int value = (int)status;
    if (MPI_Bcast(&value, 1, MPI_INT, 0, transaction->communicator) != MPI_SUCCESS ||
        MPI_Bcast(digest, SS_SHA256_HEX_CAPACITY + 7, MPI_CHAR, 0,
                  transaction->communicator) != MPI_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    status = (SsStatus)value;
  }
  if (status != SS_OK) {
    return status;
  }
  space = H5Screate_simple(1, dimensions, NULL);
  memory_space = H5Screate_simple(1, memory_dimensions, NULL);
  dataset = space < 0 ? -1
                      : H5Dcreate2(transaction->file, path, H5T_IEEE_F64LE, space,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  file_space = dataset < 0 ? -1 : H5Dget_space(dataset);
  transfer = H5Pcreate(H5P_DATASET_XFER);
  if (space < 0 || memory_space < 0 || dataset < 0 || file_space < 0 || transfer < 0 ||
      H5Pset_dxpl_mpio(transfer, H5FD_MPIO_COLLECTIVE) < 0) {
    status = SS_ERROR_HDF5;
  } else if (transaction->rank != 0 || count == 0U) {
    if (H5Sselect_none(file_space) < 0 || H5Sselect_none(memory_space) < 0) {
      status = SS_ERROR_HDF5;
    }
  }
  status = ss_status_allreduce_max(transaction->communicator, status);
  if (status == SS_OK &&
      H5Dwrite(dataset, H5T_NATIVE_DOUBLE, memory_space, file_space, transfer,
               transaction->rank == 0 && count != 0U ? values : NULL) < 0) {
    status = SS_ERROR_HDF5;
  }
  memcpy(raw, digest + 7U, sizeof(raw));
  if (status == SS_OK &&
      (ss_result_hdf5_string_attribute(dataset, "semantic_id", semantic_id) != SS_OK ||
       ss_result_hdf5_string_attribute(dataset, "value_type", "float64") != SS_OK ||
       ss_result_hdf5_string_attribute(dataset, "digest_method",
                                       "sha256-float64le-v1") != SS_OK ||
       ss_result_hdf5_string_attribute(dataset, "digest", raw) != SS_OK)) {
    status = SS_ERROR_HDF5;
  }
  status = ss_result_hdf5_close(&transfer, H5Pclose, status);
  status = ss_result_hdf5_close(&file_space, H5Sclose, status);
  status = ss_result_hdf5_close(&dataset, H5Dclose, status);
  status = ss_result_hdf5_close(&memory_space, H5Sclose, status);
  status = ss_result_hdf5_close(&space, H5Sclose, status);
  return ss_status_allreduce_max(transaction->communicator, status);
}

static hid_t ss_checkpoint_prior_file_type(void) {
  hid_t type = H5Tcreate(H5T_COMPOUND, 48U);
  if (type < 0 || H5Tinsert(type, "state_id", 0U, H5T_STD_U64LE) < 0 ||
      H5Tinsert(type, "parent_state_id", 8U, H5T_STD_U64LE) < 0 ||
      H5Tinsert(type, "accepted_ordinal", 16U, H5T_STD_U64LE) < 0 ||
      H5Tinsert(type, "factor", 24U, H5T_IEEE_F64LE) < 0 ||
      H5Tinsert(type, "work", 32U, H5T_IEEE_F64LE) < 0 ||
      H5Tinsert(type, "direct_control", 40U, H5T_IEEE_F64LE) < 0) {
    if (type >= 0) {
      (void)H5Tclose(type);
    }
    return -1;
  }
  return type;
}

static hid_t ss_checkpoint_prior_memory_type(void) {
  hid_t type = H5Tcreate(H5T_COMPOUND, sizeof(SsCheckpointPriorState));
  if (type < 0 ||
      H5Tinsert(type, "state_id", HOFFSET(SsCheckpointPriorState, state_id),
                H5T_NATIVE_UINT64) < 0 ||
      H5Tinsert(type, "parent_state_id",
                HOFFSET(SsCheckpointPriorState, parent_state_id),
                H5T_NATIVE_UINT64) < 0 ||
      H5Tinsert(type, "accepted_ordinal",
                HOFFSET(SsCheckpointPriorState, accepted_ordinal),
                H5T_NATIVE_UINT64) < 0 ||
      H5Tinsert(type, "factor", HOFFSET(SsCheckpointPriorState, factor),
                H5T_NATIVE_DOUBLE) < 0 ||
      H5Tinsert(type, "work", HOFFSET(SsCheckpointPriorState, work),
                H5T_NATIVE_DOUBLE) < 0 ||
      H5Tinsert(type, "direct_control", HOFFSET(SsCheckpointPriorState, direct_control),
                H5T_NATIVE_DOUBLE) < 0) {
    if (type >= 0) {
      (void)H5Tclose(type);
    }
    return -1;
  }
  return type;
}

static SsStatus ss_checkpoint_prior_digest(const SsCheckpointPriorState *states,
                                           size_t count,
                                           char digest[SS_SHA256_HEX_CAPACITY + 8U]) {
  unsigned char encoded[SS_CHECKPOINT_PRIOR_STATE_CAPACITY * 48U];
  size_t index;
  if (count > SS_CHECKPOINT_PRIOR_STATE_CAPACITY || (count != 0U && states == NULL)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (index = 0U; index < count; ++index) {
    unsigned char *row = encoded + index * 48U;
    ss_result_store_u64_le(row, states[index].state_id);
    ss_result_store_u64_le(row + 8U, states[index].parent_state_id);
    ss_result_store_u64_le(row + 16U, states[index].accepted_ordinal);
    ss_result_store_double_le(row + 24U, states[index].factor);
    ss_result_store_double_le(row + 32U, states[index].work);
    ss_result_store_double_le(row + 40U, states[index].direct_control);
  }
  memcpy(digest, "sha256:", 7U);
  return ss_sha256_bytes(encoded, count * 48U, digest + 7U);
}

SsStatus ss_checkpoint_write_prior(const SsCheckpointSpec *spec,
                                   SsCheckpointTransaction *transaction,
                                   char digest[SS_SHA256_HEX_CAPACITY + 8U]) {
  hid_t file_type = -1;
  hid_t memory_type = -1;
  hid_t space = -1;
  hid_t memory_space = -1;
  hid_t dataset = -1;
  hid_t file_space = -1;
  hid_t transfer = -1;
  hsize_t dimensions[1] = {(hsize_t)spec->prior_state_count};
  hsize_t memory_dimensions[1] = {
      spec->prior_state_count == 0U ? 1U : (hsize_t)spec->prior_state_count};
  char raw[SS_SHA256_HEX_CAPACITY];
  SsStatus status =
      ss_checkpoint_prior_digest(spec->prior_states, spec->prior_state_count, digest);
  status = ss_status_allreduce_max(spec->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  file_type = ss_checkpoint_prior_file_type();
  memory_type = ss_checkpoint_prior_memory_type();
  space = H5Screate_simple(1, dimensions, NULL);
  memory_space = H5Screate_simple(1, memory_dimensions, NULL);
  dataset = file_type < 0 || space < 0
                ? -1
                : H5Dcreate2(transaction->file, "/analysis/prior_states", file_type,
                             space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  file_space = dataset < 0 ? -1 : H5Dget_space(dataset);
  transfer = H5Pcreate(H5P_DATASET_XFER);
  if (file_type < 0 || memory_type < 0 || space < 0 || memory_space < 0 ||
      dataset < 0 || file_space < 0 || transfer < 0 ||
      H5Pset_dxpl_mpio(transfer, H5FD_MPIO_COLLECTIVE) < 0) {
    status = SS_ERROR_HDF5;
  } else if (transaction->rank != 0 || spec->prior_state_count == 0U) {
    if (H5Sselect_none(file_space) < 0 || H5Sselect_none(memory_space) < 0) {
      status = SS_ERROR_HDF5;
    }
  }
  status = ss_status_allreduce_max(spec->communicator, status);
  if (status == SS_OK &&
      H5Dwrite(dataset, memory_type, memory_space, file_space, transfer,
               transaction->rank == 0 && spec->prior_state_count != 0U
                   ? spec->prior_states
                   : NULL) < 0) {
    status = SS_ERROR_HDF5;
  }
  memcpy(raw, digest + 7U, sizeof(raw));
  if (status == SS_OK &&
      (ss_result_hdf5_string_attribute(dataset, "digest_method",
                                       "sha256-prior-state-v1") != SS_OK ||
       ss_result_hdf5_string_attribute(dataset, "digest", raw) != SS_OK)) {
    status = SS_ERROR_HDF5;
  }
  status = ss_result_hdf5_close(&transfer, H5Pclose, status);
  status = ss_result_hdf5_close(&file_space, H5Sclose, status);
  status = ss_result_hdf5_close(&dataset, H5Dclose, status);
  status = ss_result_hdf5_close(&memory_space, H5Sclose, status);
  status = ss_result_hdf5_close(&space, H5Sclose, status);
  status = ss_result_hdf5_close(&memory_type, H5Tclose, status);
  status = ss_result_hdf5_close(&file_type, H5Tclose, status);
  return ss_status_allreduce_max(spec->communicator, status);
}

static hid_t ss_checkpoint_cursor_file_type(void) {
  hid_t type = H5Tcreate(H5T_COMPOUND, 32U);
  if (type < 0 || H5Tinsert(type, "history_cursor", 0U, H5T_STD_U64LE) < 0 ||
      H5Tinsert(type, "accepted_state_cursor", 8U, H5T_STD_U64LE) < 0 ||
      H5Tinsert(type, "retry_count", 16U, H5T_STD_U64LE) < 0 ||
      H5Tinsert(type, "next_step", 24U, H5T_IEEE_F64LE) < 0) {
    if (type >= 0) {
      (void)H5Tclose(type);
    }
    return -1;
  }
  return type;
}

static hid_t ss_checkpoint_cursor_memory_type(void) {
  hid_t type = H5Tcreate(H5T_COMPOUND, sizeof(SsCheckpointCursor));
  if (type < 0 ||
      H5Tinsert(type, "history_cursor", HOFFSET(SsCheckpointCursor, history_cursor),
                H5T_NATIVE_UINT64) < 0 ||
      H5Tinsert(type, "accepted_state_cursor",
                HOFFSET(SsCheckpointCursor, accepted_state_cursor),
                H5T_NATIVE_UINT64) < 0 ||
      H5Tinsert(type, "retry_count", HOFFSET(SsCheckpointCursor, retry_count),
                H5T_NATIVE_UINT64) < 0 ||
      H5Tinsert(type, "next_step", HOFFSET(SsCheckpointCursor, next_step),
                H5T_NATIVE_DOUBLE) < 0) {
    if (type >= 0) {
      (void)H5Tclose(type);
    }
    return -1;
  }
  return type;
}

static SsStatus ss_checkpoint_cursor_digest(const SsCheckpointCursor *cursor,
                                            char digest[SS_SHA256_HEX_CAPACITY + 8U]) {
  unsigned char encoded[32U];
  if (cursor == NULL || !isfinite(cursor->next_step) || cursor->next_step <= 0.0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  ss_result_store_u64_le(encoded, cursor->history_cursor);
  ss_result_store_u64_le(encoded + 8U, cursor->accepted_state_cursor);
  ss_result_store_u64_le(encoded + 16U, cursor->retry_count);
  ss_result_store_double_le(encoded + 24U, cursor->next_step);
  memcpy(digest, "sha256:", 7U);
  return ss_sha256_bytes(encoded, sizeof(encoded), digest + 7U);
}

SsStatus ss_checkpoint_write_cursor(const SsCheckpointSpec *spec,
                                    SsCheckpointTransaction *transaction,
                                    char digest[SS_SHA256_HEX_CAPACITY + 8U]) {
  SsCheckpointCursor cursor = {
      spec->history_cursor,
      spec->accepted_state_cursor,
      spec->retry_count,
      spec->next_step,
  };
  hid_t file_type = ss_checkpoint_cursor_file_type();
  hid_t memory_type = ss_checkpoint_cursor_memory_type();
  hid_t space = -1;
  hid_t memory_space = -1;
  hid_t dataset = -1;
  hid_t file_space = -1;
  hid_t transfer = -1;
  hsize_t dimensions[1] = {1U};
  char raw[SS_SHA256_HEX_CAPACITY];
  SsStatus status = ss_checkpoint_cursor_digest(&cursor, digest);
  space = H5Screate_simple(1, dimensions, NULL);
  memory_space = H5Screate_simple(1, dimensions, NULL);
  dataset = file_type < 0 || space < 0
                ? -1
                : H5Dcreate2(transaction->file, "/history/cursor", file_type, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  file_space = dataset < 0 ? -1 : H5Dget_space(dataset);
  transfer = H5Pcreate(H5P_DATASET_XFER);
  if (status != SS_OK || file_type < 0 || memory_type < 0 || space < 0 ||
      memory_space < 0 || dataset < 0 || file_space < 0 || transfer < 0 ||
      H5Pset_dxpl_mpio(transfer, H5FD_MPIO_COLLECTIVE) < 0) {
    status = status == SS_OK ? SS_ERROR_HDF5 : status;
  } else if (transaction->rank != 0) {
    if (H5Sselect_none(file_space) < 0 || H5Sselect_none(memory_space) < 0) {
      status = SS_ERROR_HDF5;
    }
  }
  status = ss_status_allreduce_max(spec->communicator, status);
  if (status == SS_OK &&
      H5Dwrite(dataset, memory_type, memory_space, file_space, transfer,
               transaction->rank == 0 ? &cursor : NULL) < 0) {
    status = SS_ERROR_HDF5;
  }
  memcpy(raw, digest + 7U, sizeof(raw));
  if (status == SS_OK &&
      (ss_result_hdf5_string_attribute(dataset, "digest_method", "sha256-cursor-v1") !=
           SS_OK ||
       ss_result_hdf5_string_attribute(dataset, "digest", raw) != SS_OK)) {
    status = SS_ERROR_HDF5;
  }
  status = ss_result_hdf5_close(&transfer, H5Pclose, status);
  status = ss_result_hdf5_close(&file_space, H5Sclose, status);
  status = ss_result_hdf5_close(&dataset, H5Dclose, status);
  status = ss_result_hdf5_close(&memory_space, H5Sclose, status);
  status = ss_result_hdf5_close(&space, H5Sclose, status);
  status = ss_result_hdf5_close(&memory_type, H5Tclose, status);
  status = ss_result_hdf5_close(&file_type, H5Tclose, status);
  return ss_status_allreduce_max(spec->communicator, status);
}

static SsStatus ss_checkpoint_add_field_attributes(hid_t file, const char *group_path,
                                                   const SsCheckpointFieldView *field) {
  static const char *const leaves[] = {"stable_ids", "values"};
  char path[SS_RESULT_PATH_CAPACITY];
  hid_t dataset = -1;
  size_t leaf;
  SsStatus status = SS_OK;
  for (leaf = 0U; leaf < sizeof(leaves) / sizeof(leaves[0]); ++leaf) {
    if (snprintf(path, sizeof(path), "%s/%s", group_path, leaves[leaf]) < 0) {
      status = SS_ERROR_OVERFLOW;
      break;
    }
    dataset = H5Dopen2(file, path, H5P_DEFAULT);
    if (dataset < 0 ||
        ss_result_hdf5_string_attribute(dataset, "layout_id", "checkpoint-v1") !=
            SS_OK ||
        ss_result_hdf5_string_attribute(dataset, "space_id", field->space_id) !=
            SS_OK) {
      status = SS_ERROR_HDF5;
    }
    status = ss_result_hdf5_close(&dataset, H5Dclose, status);
  }
  return status;
}

SsStatus ss_checkpoint_write_fields(const SsCheckpointSpec *spec,
                                    SsCheckpointTransaction *transaction,
                                    SsCheckpointFieldDigest *digests,
                                    char left_digest[SS_SHA256_HEX_CAPACITY + 8U],
                                    char right_digest[SS_SHA256_HEX_CAPACITY + 8U],
                                    char source_digest[SS_SHA256_HEX_CAPACITY + 8U],
                                    uint64_t *logical_bytes) {
  uint64_t bytes = 0U;
  size_t field;
  SsStatus status = SS_OK;
  for (field = 0U; status == SS_OK && field < spec->field_count; ++field) {
    const SsCheckpointFieldView *view = &spec->fields[field];
    SsResultFileIndexSpec index_spec;
    SsResultFileIndexPlan *plan = NULL;
    char path[SS_RESULT_PATH_CAPACITY];
    uint64_t field_bytes = 0U;
    memset(&index_spec, 0, sizeof(index_spec));
    index_spec.communicator = spec->communicator;
    index_spec.stable_ids = view->stable_ids;
    index_spec.local_slots = view->local_slots;
    index_spec.local_count = view->local_count;
    index_spec.global_count = view->global_count;
    status = ss_result_file_index_plan_create(&index_spec, &plan);
    if (status == SS_OK &&
        snprintf(path, sizeof(path), "/accepted/fields/%s", view->quantity_id) < 0) {
      status = SS_ERROR_OVERFLOW;
    }
    if (status == SS_OK) {
      status = ss_result_write_indexed_dataset(
          transaction->file, spec->communicator, path, view->quantity_id, 1U,
          view->unit, "field_dof", view->basis, spec->mesh_id, spec->geometry_id,
          spec->accepted_state_id, spec->field_order, spec->geometry_order,
          view->component_ids, view->component_count, spec->dataset_chunk_rows, plan,
          view->values, digests[field].digest, &field_bytes);
    }
    if (status == SS_OK) {
      status = ss_checkpoint_add_field_attributes(transaction->file, path, view);
    }
    if (status == SS_OK) {
      (void)ss_result_copy_text(digests[field].quantity_id,
                                sizeof(digests[field].quantity_id), view->quantity_id);
      (void)ss_result_copy_text(digests[field].unit, sizeof(digests[field].unit),
                                view->unit);
      (void)ss_result_copy_text(digests[field].space_id,
                                sizeof(digests[field].space_id), view->space_id);
      (void)ss_result_copy_text(digests[field].basis, sizeof(digests[field].basis),
                                view->basis);
      status = ss_checkpoint_components(view->component_ids, view->component_count,
                                        digests[field].components,
                                        sizeof(digests[field].components));
      digests[field].global_count = view->global_count;
      digests[field].component_count = view->component_count;
      bytes += field_bytes;
    }
    if (ss_result_file_index_plan_destroy(&plan) != SS_OK && status == SS_OK) {
      status = SS_ERROR_INTERNAL;
    }
  }
  if (status == SS_OK && spec->deflation.basis_count != 0U) {
    SsResultFileIndexSpec index_spec;
    SsResultFileIndexPlan *plan = NULL;
    const char *components[SS_RESULT_COMPONENT_CAPACITY];
    char component_names[SS_RESULT_COMPONENT_CAPACITY][24];
    uint32_t basis;
    uint64_t field_bytes = 0U;
    memset(&index_spec, 0, sizeof(index_spec));
    index_spec.communicator = spec->communicator;
    index_spec.stable_ids = spec->deflation.stable_free_dof_ids;
    index_spec.local_slots = spec->deflation.local_slots;
    index_spec.local_count = spec->deflation.local_count;
    index_spec.global_count = spec->deflation.global_count;
    status = ss_result_file_index_plan_create(&index_spec, &plan);
    for (basis = 0U; status == SS_OK && basis < spec->deflation.basis_count; ++basis) {
      const int count = snprintf(component_names[basis], sizeof(component_names[basis]),
                                 "basis_%u", basis);
      if (count < 0 || (size_t)count >= sizeof(component_names[basis])) {
        status = SS_ERROR_OVERFLOW;
      } else {
        components[basis] = component_names[basis];
      }
    }
    if (status == SS_OK) {
      status = ss_result_write_indexed_dataset(
          transaction->file, spec->communicator, "/deflation/left", "deflation_left",
          1U, "dimensionless", "free_dof", "persistent_basis", spec->mesh_id,
          spec->geometry_id, spec->accepted_state_id, spec->field_order,
          spec->geometry_order, components, spec->deflation.basis_count,
          spec->dataset_chunk_rows, plan, spec->deflation.left_values, left_digest,
          &field_bytes);
      bytes += field_bytes;
    }
    if (status == SS_OK) {
      status = ss_result_write_indexed_dataset(
          transaction->file, spec->communicator, "/deflation/right", "deflation_right",
          1U, "dimensionless", "free_dof", "persistent_basis", spec->mesh_id,
          spec->geometry_id, spec->accepted_state_id, spec->field_order,
          spec->geometry_order, components, spec->deflation.basis_count,
          spec->dataset_chunk_rows, plan, spec->deflation.right_values, right_digest,
          &field_bytes);
      bytes += field_bytes;
    }
    if (status == SS_OK) {
      status = ss_result_write_indexed_dataset(
          transaction->file, spec->communicator, "/deflation/source",
          "deflation_source", 1U, "dimensionless", "free_dof",
          "persistent_compression_source", spec->mesh_id, spec->geometry_id,
          spec->accepted_state_id, spec->field_order, spec->geometry_order, components,
          spec->deflation.basis_count, spec->dataset_chunk_rows, plan,
          spec->deflation.source_values, source_digest, &field_bytes);
      bytes += field_bytes;
    }
    if (ss_result_file_index_plan_destroy(&plan) != SS_OK && status == SS_OK) {
      status = SS_ERROR_INTERNAL;
    }
  }
  *logical_bytes = bytes;
  return ss_status_allreduce_max(spec->communicator, status);
}

SsStatus ss_checkpoint_dataset_shape(hid_t file, const char *path, uint64_t *rows,
                                     uint32_t *columns) {
  hid_t dataset = H5Dopen2(file, path, H5P_DEFAULT);
  hid_t space = dataset < 0 ? -1 : H5Dget_space(dataset);
  hsize_t dimensions[2] = {0U, 0U};
  SsStatus status = SS_OK;
  if (dataset < 0 || space < 0 || H5Sget_simple_extent_ndims(space) != 2 ||
      H5Sget_simple_extent_dims(space, dimensions, NULL) != 2 || dimensions[0] == 0U ||
      dimensions[1] == 0U || dimensions[1] > UINT32_MAX) {
    status = SS_ERROR_SCHEMA;
  } else {
    *rows = (uint64_t)dimensions[0];
    *columns = (uint32_t)dimensions[1];
  }
  status = ss_result_hdf5_close(&space, H5Sclose, status);
  status = ss_result_hdf5_close(&dataset, H5Dclose, status);
  return status;
}

SsStatus ss_checkpoint_read_replicated_double(
    hid_t file, MPI_Comm communicator, int rank, const char *path, double *values,
    size_t capacity, size_t *count, char digest[SS_SHA256_HEX_CAPACITY + 8U]) {
  hid_t dataset = H5Dopen2(file, path, H5P_DEFAULT);
  hid_t space = dataset < 0 ? -1 : H5Dget_space(dataset);
  hsize_t dimensions[1] = {0U};
  size_t index;
  SsStatus status = SS_OK;
  if (dataset < 0 || space < 0 || H5Sget_simple_extent_ndims(space) != 1 ||
      H5Sget_simple_extent_dims(space, dimensions, NULL) != 1 ||
      dimensions[0] > (hsize_t)capacity || (dimensions[0] != 0U && values == NULL)) {
    status = SS_ERROR_SCHEMA;
  } else {
    *count = (size_t)dimensions[0];
    if (rank == 0 && *count != 0U &&
        H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values) <
            0) {
      status = SS_ERROR_HDF5;
    }
  }
  status = ss_result_hdf5_close(&space, H5Sclose, status);
  status = ss_result_hdf5_close(&dataset, H5Dclose, status);
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK && *count > (size_t)INT_MAX) {
    status = SS_ERROR_OVERFLOW;
  }
  if (status == SS_OK &&
      MPI_Bcast(values, (int)*count, MPI_DOUBLE, 0, communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  for (index = 0U; status == SS_OK && index < *count; ++index) {
    if (!isfinite(values[index])) {
      status = SS_ERROR_INTEGRITY;
    }
  }
  if (status == SS_OK) {
    /* Stream the little-endian values: datasets may hold up to the
     * replicated-value capacity (64 deflation columns), more than a
     * controller-sized buffer. */
    SsSha256 context;
    ss_sha256_init(&context);
    for (index = 0U; status == SS_OK && index < *count; ++index) {
      unsigned char encoded[sizeof(double)];
      ss_result_store_double_le(encoded, values[index]);
      status = ss_sha256_update(&context, encoded, sizeof(encoded));
    }
    if (status == SS_OK) {
      memcpy(digest, "sha256:", 7U);
      ss_sha256_final(&context, digest + 7U);
    }
  }
  return ss_status_allreduce_max(communicator, status);
}

SsStatus ss_checkpoint_read_prior(hid_t file, MPI_Comm communicator, int rank,
                                  SsCheckpointPriorState *states, size_t capacity,
                                  size_t *count,
                                  char digest[SS_SHA256_HEX_CAPACITY + 8U]) {
  hid_t dataset = H5Dopen2(file, "/analysis/prior_states", H5P_DEFAULT);
  hid_t space = dataset < 0 ? -1 : H5Dget_space(dataset);
  hid_t memory_type = ss_checkpoint_prior_memory_type();
  hsize_t dimensions[1] = {0U};
  SsStatus status = SS_OK;
  if (dataset < 0 || space < 0 || memory_type < 0 ||
      H5Sget_simple_extent_ndims(space) != 1 ||
      H5Sget_simple_extent_dims(space, dimensions, NULL) != 1 ||
      dimensions[0] > (hsize_t)capacity || (dimensions[0] != 0U && states == NULL)) {
    status = SS_ERROR_SCHEMA;
  } else {
    *count = (size_t)dimensions[0];
    if (rank == 0 && *count != 0U &&
        H5Dread(dataset, memory_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, states) < 0) {
      status = SS_ERROR_HDF5;
    }
  }
  status = ss_result_hdf5_close(&memory_type, H5Tclose, status);
  status = ss_result_hdf5_close(&space, H5Sclose, status);
  status = ss_result_hdf5_close(&dataset, H5Dclose, status);
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK && MPI_Bcast(states, (int)(*count * sizeof(*states)), MPI_BYTE, 0,
                                   communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    status = ss_checkpoint_prior_digest(states, *count, digest);
  }
  return ss_status_allreduce_max(communicator, status);
}

SsStatus ss_checkpoint_read_cursor(hid_t file, MPI_Comm communicator, int rank,
                                   SsCheckpointCursor *cursor,
                                   char digest[SS_SHA256_HEX_CAPACITY + 8U]) {
  hid_t dataset = H5Dopen2(file, "/history/cursor", H5P_DEFAULT);
  hid_t memory_type = ss_checkpoint_cursor_memory_type();
  SsStatus status = SS_OK;
  if (dataset < 0 || memory_type < 0 ||
      (rank == 0 &&
       H5Dread(dataset, memory_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, cursor) < 0)) {
    status = SS_ERROR_HDF5;
  }
  status = ss_result_hdf5_close(&memory_type, H5Tclose, status);
  status = ss_result_hdf5_close(&dataset, H5Dclose, status);
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK && MPI_Bcast(cursor, (int)sizeof(*cursor), MPI_BYTE, 0,
                                   communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    status = ss_checkpoint_cursor_digest(cursor, digest);
  }
  return ss_status_allreduce_max(communicator, status);
}

SsStatus ss_checkpoint_field_metadata(hid_t file, const char *group_path,
                                      SsCheckpointFieldDigest *field) {
  char path[SS_RESULT_PATH_CAPACITY];
  hid_t dataset = -1;
  uint64_t version = 0U;
  SsStatus status;
  if (snprintf(path, sizeof(path), "%s/values", group_path) < 0) {
    return SS_ERROR_OVERFLOW;
  }
  dataset = H5Dopen2(file, path, H5P_DEFAULT);
  if (dataset < 0) {
    return SS_ERROR_HDF5;
  }
  status = ss_result_hdf5_read_string_attribute(
      dataset, "quantity_id", field->quantity_id, sizeof(field->quantity_id));
  if (status == SS_OK) {
    status = ss_result_hdf5_read_u64_attribute(dataset, "quantity_version", &version);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_string_attribute(dataset, "unit", field->unit,
                                                  sizeof(field->unit));
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_string_attribute(dataset, "space_id", field->space_id,
                                                  sizeof(field->space_id));
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_string_attribute(dataset, "basis", field->basis,
                                                  sizeof(field->basis));
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_string_attribute(
        dataset, "component_ids", field->components, sizeof(field->components));
  }
  if (status == SS_OK && version != 1U) {
    status = SS_ERROR_VERSION_MISMATCH;
  }
  status = ss_result_hdf5_close(&dataset, H5Dclose, status);
  return status;
}
