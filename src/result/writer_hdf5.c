/* Create, populate and verify the result HDF5 layout.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "writer_internal.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static SsStatus ss_result_writer_precreate_relations(SsResultWriter *writer);

static SsStatus ss_result_writer_environment(SsResultWriter *writer) {
  char environment[2048];
  char assembly[160] = "";
  int threads = 1;
  int count;
  int status_value = (int)SS_OK;
#ifdef _OPENMP
  threads = omp_get_max_threads();
#endif
  if (writer->rank == 0) {
    if (writer->mechanics_assembly_minimum_threads > 0) {
      count = snprintf(assembly, sizeof(assembly),
                       ",\"mechanics_assembly_minimum_threads\":%d,"
                       "\"mechanics_assembly_maximum_threads\":%d",
                       writer->mechanics_assembly_minimum_threads,
                       writer->mechanics_assembly_maximum_threads);
      if (count < 0 || (size_t)count >= sizeof(assembly)) {
        status_value = (int)SS_ERROR_OVERFLOW;
      }
    }
    count = snprintf(environment, sizeof(environment),
                     "{\"kind\":\"slope_stability.environment\","
                     "\"schema_version\":{\"major\":1,\"minor\":0,\"patch\":0},"
                     "\"run_id\":\"%s\",\"build_id\":\"%s\","
                     "\"mpi\":{\"world_size\":%d},"
                     "\"openmp\":{\"maximum_threads\":%d%s},"
                     "\"parallel_hdf5\":true,"
                     "\"authority\":\"native-observation\"}\n",
                     writer->run_id, writer->build_id, writer->size, threads, assembly);
    if (count < 0 || (size_t)count >= sizeof(environment)) {
      status_value = (int)SS_ERROR_OVERFLOW;
    } else if ((SsStatus)status_value == SS_OK) {
      status_value = (int)ss_result_write_text_atomic(writer->environment_path,
                                                      environment, (size_t)count);
      if ((SsStatus)status_value == SS_OK) {
        status_value = (int)ss_result_payload_digest(writer->environment_path,
                                                     writer->environment_sha256,
                                                     &writer->environment_bytes);
      }
    }
  }
  if (MPI_Bcast(&status_value, 1, MPI_INT, 0, writer->communicator) != MPI_SUCCESS ||
      MPI_Bcast(writer->environment_sha256, SS_SHA256_HEX_CAPACITY, MPI_CHAR, 0,
                writer->communicator) != MPI_SUCCESS ||
      MPI_Bcast(&writer->environment_bytes, 1, MPI_UINT64_T, 0, writer->communicator) !=
          MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  return (SsStatus)status_value;
}

static SsStatus ss_result_writer_hdf5_root(SsResultWriter *writer) {
  static const char *const groups[] = {
      "/states",         "/states/index",          "/fields",
      "/quadrature",     "/quadrature_categories", "/cell_reductions",
      "/history",        "/history/accepted",      "/history/attempt",
      "/history/newton", "/history/line_search",   "/history/linear",
      "/history/pmg",    "/history/deflation",     "/history/seepage",
      "/history/io",     "/history/krylov",        "/history/pmg_apply",
      "/diagnostics",
  };
  SsStatus status;
  writer->access = H5Pcreate(H5P_FILE_ACCESS);
  if (writer->access < 0 ||
      H5Pset_fapl_mpio(writer->access, writer->communicator, MPI_INFO_NULL) < 0 ||
      H5Pset_all_coll_metadata_ops(writer->access, 1U) < 0) {
    return SS_ERROR_HDF5;
  }
  writer->file =
      H5Fcreate(writer->hdf5_temporary_path, H5F_ACC_EXCL, H5P_DEFAULT, writer->access);
  if (writer->file < 0) {
    return SS_ERROR_HDF5;
  }
  status = ss_result_hdf5_string_attribute(writer->file, "layout_id", "result-v1");
  if (status == SS_OK) {
    status = ss_result_hdf5_u64_attribute(writer->file, "schema_major", 1U);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_u64_attribute(writer->file, "schema_minor", 0U);
  }
  if (status == SS_OK) {
    status =
        ss_result_hdf5_string_attribute(writer->file, "problem_id", writer->problem_id);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_string_attribute(writer->file, "run_id", writer->run_id);
  }
  if (status == SS_OK) {
    status =
        ss_result_hdf5_string_attribute(writer->file, "build_id", writer->build_id);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_string_attribute(writer->file, "mesh_id", writer->mesh_id);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_string_attribute(writer->file, "geometry_id",
                                             writer->geometry_id);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_string_attribute(writer->file, "analysis_method",
                                             writer->analysis_method);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_u64_attribute(writer->file, "dimension", writer->dimension);
  }
  if (status == SS_OK) {
    status =
        ss_result_hdf5_u64_attribute(writer->file, "field_order", writer->field_order);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_u64_attribute(writer->file, "geometry_order",
                                          writer->geometry_order);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_string_attribute(
        writer->file, "physical_dataset_layout",
        writer->dataset_chunk_rows == 0U ? "chunked_default" : "chunked_requested");
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_u64_attribute(writer->file, "dataset_chunk_rows",
                                          writer->dataset_chunk_rows);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_create_groups(writer->file, writer->communicator, groups,
                                          sizeof(groups) / sizeof(groups[0]));
  }
  return status;
}

SsStatus ss_result_writer_stage(SsResultWriter *writer) {
  SsStatus status;
  if (writer == NULL || writer->state != SS_RESULT_WRITER_UNOPENED) {
    return SS_ERROR_INVALID_STATE;
  }
  status = ss_result_writer_fail(writer, SS_RESULT_FAIL_DIRECTORY_CREATE);
  if (status != SS_OK) {
    return status;
  }
  status = ss_result_make_staging(writer->communicator, writer->final_directory,
                                  "result", writer->staging_directory,
                                  sizeof(writer->staging_directory));
  if (status != SS_OK) {
    writer->state = SS_RESULT_WRITER_PUBLICATION_FAILED;
    writer->summary.writer_state = writer->state;
    return status;
  }
  if (snprintf(writer->hdf5_temporary_path, sizeof(writer->hdf5_temporary_path),
               "%s/result.h5.tmp", writer->staging_directory) < 0 ||
      snprintf(writer->hdf5_path, sizeof(writer->hdf5_path), "%s/result.h5",
               writer->staging_directory) < 0 ||
      snprintf(writer->environment_path, sizeof(writer->environment_path),
               "%s/environment.json", writer->staging_directory) < 0 ||
      snprintf(writer->result_path, sizeof(writer->result_path), "%s/result.json",
               writer->staging_directory) < 0 ||
      snprintf(writer->result_temporary_path, sizeof(writer->result_temporary_path),
               "%s/result.json.tmp", writer->staging_directory) < 0) {
    writer->state = SS_RESULT_WRITER_PUBLICATION_FAILED;
    return SS_ERROR_OVERFLOW;
  }
  (void)ss_result_copy_text(writer->summary.staging_directory,
                            sizeof(writer->summary.staging_directory),
                            writer->staging_directory);
  status = ss_result_writer_fail(writer, SS_RESULT_FAIL_ENVIRONMENT_WRITE);
  if (status == SS_OK) {
    status = ss_result_writer_environment(writer);
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status != SS_OK) {
    writer->state = SS_RESULT_WRITER_PUBLICATION_FAILED;
    writer->summary.writer_state = writer->state;
    return status;
  }
  status = ss_result_writer_fail(writer, SS_RESULT_FAIL_HDF5_CREATE);
  if (status != SS_OK) {
    return status;
  }
  writer->access = H5Pcreate(H5P_FILE_ACCESS);
  if (writer->access < 0 ||
      H5Pset_fapl_mpio(writer->access, writer->communicator, MPI_INFO_NULL) < 0 ||
      H5Pset_all_coll_metadata_ops(writer->access, 1U) < 0) {
    status = SS_ERROR_HDF5;
  } else {
    writer->file = H5Fcreate(writer->hdf5_temporary_path, H5F_ACC_EXCL, H5P_DEFAULT,
                             writer->access);
    if (writer->file < 0) {
      status = SS_ERROR_HDF5;
    }
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status != SS_OK) {
    writer->state = SS_RESULT_WRITER_PUBLICATION_FAILED;
    writer->summary.writer_state = writer->state;
    return status;
  }
  status = ss_result_writer_fail(writer, SS_RESULT_FAIL_METADATA_CREATE);
  if (status == SS_OK) {
    /*
     * Reuse the complete root-plan helper after closing the provisional file
     * created above. The separate create phase makes the injected create and
     * metadata points observable without conflating them.
     */
    status = ss_result_hdf5_close(&writer->file, H5Fclose, status);
    status = ss_result_hdf5_close(&writer->access, H5Pclose, status);
    if (status == SS_OK && writer->rank == 0 &&
        unlink(writer->hdf5_temporary_path) != 0) {
      status = SS_ERROR_IO;
    }
    status = ss_status_allreduce_max(writer->communicator, status);
    if (status == SS_OK && MPI_Barrier(writer->communicator) != MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
    if (status == SS_OK) {
      status = ss_result_writer_hdf5_root(writer);
    }
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status != SS_OK) {
    writer->state = SS_RESULT_WRITER_PUBLICATION_FAILED;
    writer->summary.writer_state = writer->state;
    return status;
  }
  writer->state = SS_RESULT_WRITER_STAGED;
  writer->summary.writer_state = writer->state;
  return SS_OK;
}

static SsStatus ss_result_writer_freeze_plan(SsResultWriter *writer) {
  char *payload = NULL;
  size_t capacity;
  size_t cursor = 0U;
  size_t index;
  SsStatus status = SS_OK;
  if (writer->field_count > (SIZE_MAX - 1024U) / 4096U) {
    status = SS_ERROR_OVERFLOW;
  }
  capacity = status == SS_OK ? 1024U + writer->field_count * 4096U : 0U;
  if (status == SS_OK) {
    payload = (char *)malloc(capacity);
    if (payload == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  if (status == SS_OK) {
    int count = snprintf(
        payload, capacity,
        "output-call-plan-v1|layout=result-v1|quantities=%llu|chunk_rows=%llu|",
        (unsigned long long)writer->field_count,
        (unsigned long long)writer->dataset_chunk_rows);
    if (count < 0 || (size_t)count >= capacity) {
      status = SS_ERROR_OVERFLOW;
    } else {
      cursor = (size_t)count;
    }
  }
  for (index = 0U; status == SS_OK && index < writer->field_count; ++index) {
    const SsResultFieldRecord *record = &writer->fields[index];
    int count =
        snprintf(payload + cursor, capacity - cursor,
                 "%s|%u|%u|%s|%s|%s|%s|%s|%s|%u|%u|%u|%llu|", record->group_path,
                 (unsigned int)record->dataset_kind, record->quantity_version,
                 record->quantity_id, record->unit, record->location, record->basis,
                 record->quadrature_id,
                 record->dataset_kind == SS_RESULT_DATASET_QUADRATURE_CATEGORY
                     ? record->category_ids
                     : record->component_ids,
                 record->component_count, record->category_count, record->sample_count,
                 (unsigned long long)record->global_count);
    if (count < 0 || (size_t)count >= capacity - cursor) {
      status = SS_ERROR_OVERFLOW;
      break;
    }
    cursor += (size_t)count;
  }
  if (status == SS_OK) {
    memcpy(writer->collective_plan_digest, "sha256:", 7U);
    status = ss_sha256_bytes((const unsigned char *)payload, cursor,
                             writer->collective_plan_digest + 7U);
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK) {
    status = ss_result_collective_payload_agree(
        writer->communicator, (const unsigned char *)writer->collective_plan_digest,
        SS_SHA256_HEX_CAPACITY + 7U);
  }
  if (status == SS_OK) {
    status = ss_result_writer_precreate_relations(writer);
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK) {
    status = ss_result_hdf5_string_attribute(writer->file, "collective_plan_version",
                                             "output-call-plan-v1");
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK) {
    status = ss_result_hdf5_string_attribute(writer->file, "collective_plan_digest",
                                             writer->collective_plan_digest);
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK) {
    status = ss_result_hdf5_u64_attribute(writer->file, "collective_plan_frozen", 1U);
  }
  free(payload);
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK) {
    writer->collective_plan_frozen = 1;
  }
  return status;
}

SsStatus ss_result_writer_begin_stream(SsResultWriter *writer) {
  SsStatus status;
  if (writer == NULL || writer->state != SS_RESULT_WRITER_STAGED) {
    return SS_ERROR_INVALID_STATE;
  }
  status = ss_result_writer_freeze_plan(writer);
  if (status != SS_OK) {
    return status;
  }
  writer->state = SS_RESULT_WRITER_STREAMING;
  writer->summary.writer_state = writer->state;
  return SS_OK;
}

static hid_t ss_result_state_file_type(void) {
  hid_t type = H5Tcreate(H5T_COMPOUND, 108U);
  if (type < 0 || H5Tinsert(type, "state_id", 0U, H5T_STD_U64LE) < 0 ||
      H5Tinsert(type, "parent_state_id", 8U, H5T_STD_U64LE) < 0 ||
      H5Tinsert(type, "accepted_ordinal", 16U, H5T_STD_U64LE) < 0 ||
      H5Tinsert(type, "hydraulic_generation", 24U, H5T_STD_U64LE) < 0 ||
      H5Tinsert(type, "mechanics_generation", 32U, H5T_STD_U64LE) < 0 ||
      H5Tinsert(type, "control_generation", 40U, H5T_STD_U64LE) < 0 ||
      H5Tinsert(type, "factor", 48U, H5T_IEEE_F64LE) < 0 ||
      H5Tinsert(type, "work", 56U, H5T_IEEE_F64LE) < 0 ||
      H5Tinsert(type, "direct_control", 64U, H5T_IEEE_F64LE) < 0 ||
      H5Tinsert(type, "displacement_maximum", 72U, H5T_IEEE_F64LE) < 0 ||
      H5Tinsert(type, "history_begin", 80U, H5T_STD_U64LE) < 0 ||
      H5Tinsert(type, "history_end", 88U, H5T_STD_U64LE) < 0 ||
      H5Tinsert(type, "complete", 96U, H5T_STD_I32LE) < 0 ||
      H5Tinsert(type, "accepted", 100U, H5T_STD_I32LE) < 0) {
    if (type >= 0) {
      (void)H5Tclose(type);
    }
    return -1;
  }
  return type;
}

static hid_t ss_result_state_memory_type(void) {
  hid_t type = H5Tcreate(H5T_COMPOUND, sizeof(SsResultAcceptedState));
  if (type < 0 ||
      H5Tinsert(type, "state_id", HOFFSET(SsResultAcceptedState, state_id),
                H5T_NATIVE_UINT64) < 0 ||
      H5Tinsert(type, "parent_state_id",
                HOFFSET(SsResultAcceptedState, parent_state_id),
                H5T_NATIVE_UINT64) < 0 ||
      H5Tinsert(type, "accepted_ordinal",
                HOFFSET(SsResultAcceptedState, accepted_ordinal),
                H5T_NATIVE_UINT64) < 0 ||
      H5Tinsert(type, "hydraulic_generation",
                HOFFSET(SsResultAcceptedState, hydraulic_generation),
                H5T_NATIVE_UINT64) < 0 ||
      H5Tinsert(type, "mechanics_generation",
                HOFFSET(SsResultAcceptedState, mechanics_generation),
                H5T_NATIVE_UINT64) < 0 ||
      H5Tinsert(type, "control_generation",
                HOFFSET(SsResultAcceptedState, control_generation),
                H5T_NATIVE_UINT64) < 0 ||
      H5Tinsert(type, "factor", HOFFSET(SsResultAcceptedState, factor),
                H5T_NATIVE_DOUBLE) < 0 ||
      H5Tinsert(type, "work", HOFFSET(SsResultAcceptedState, work), H5T_NATIVE_DOUBLE) <
          0 ||
      H5Tinsert(type, "direct_control", HOFFSET(SsResultAcceptedState, direct_control),
                H5T_NATIVE_DOUBLE) < 0 ||
      H5Tinsert(type, "displacement_maximum",
                HOFFSET(SsResultAcceptedState, displacement_maximum),
                H5T_NATIVE_DOUBLE) < 0 ||
      H5Tinsert(type, "history_begin", HOFFSET(SsResultAcceptedState, history_begin),
                H5T_NATIVE_UINT64) < 0 ||
      H5Tinsert(type, "history_end", HOFFSET(SsResultAcceptedState, history_end),
                H5T_NATIVE_UINT64) < 0 ||
      H5Tinsert(type, "complete", HOFFSET(SsResultAcceptedState, complete),
                H5T_NATIVE_INT) < 0 ||
      H5Tinsert(type, "accepted", HOFFSET(SsResultAcceptedState, accepted),
                H5T_NATIVE_INT) < 0) {
    if (type >= 0) {
      (void)H5Tclose(type);
    }
    return -1;
  }
  return type;
}

static SsStatus ss_result_write_replicated(SsResultWriter *writer, const char *path,
                                           hid_t file_type, hid_t memory_type,
                                           const void *values, size_t count) {
  hid_t memory_space = -1;
  hid_t dataset = -1;
  hid_t file_space = -1;
  hid_t transfer = -1;
  hsize_t dimensions[1] = {(hsize_t)count};
  hsize_t memory_dimensions[1] = {count == 0U ? 1U : (hsize_t)count};
  hsize_t start[1] = {0U};
  hsize_t selected[1] = {(hsize_t)count};
  SsStatus status = SS_OK;
  (void)file_type;
  dataset = H5Dopen2(writer->file, path, H5P_DEFAULT);
  if (dataset < 0 || H5Dset_extent(dataset, dimensions) < 0) {
    status = SS_ERROR_HDF5;
  }
  memory_space = H5Screate_simple(1, memory_dimensions, NULL);
  file_space = dataset < 0 ? -1 : H5Dget_space(dataset);
  transfer = H5Pcreate(H5P_DATASET_XFER);
  if (status != SS_OK || memory_space < 0 || file_space < 0 || transfer < 0 ||
      H5Pset_dxpl_mpio(transfer, H5FD_MPIO_COLLECTIVE) < 0) {
    status = SS_ERROR_HDF5;
  } else if (writer->rank != 0 || count == 0U) {
    if (H5Sselect_none(file_space) < 0 || H5Sselect_none(memory_space) < 0) {
      status = SS_ERROR_HDF5;
    }
  } else if (H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, selected,
                                 NULL) < 0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK &&
      H5Dwrite(dataset, memory_type, memory_space, file_space, transfer,
               writer->rank == 0 && count != 0U ? values : NULL) < 0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_result_hdf5_close(&transfer, H5Pclose, status);
  status = ss_result_hdf5_close(&file_space, H5Sclose, status);
  status = ss_result_hdf5_close(&dataset, H5Dclose, status);
  status = ss_result_hdf5_close(&memory_space, H5Sclose, status);
  return ss_status_allreduce_max(writer->communicator, status);
}

static SsStatus ss_result_precreate_replicated(SsResultWriter *writer, const char *path,
                                               hid_t file_type) {
  hid_t space = -1;
  hid_t creation = -1;
  hid_t dataset = -1;
  hsize_t dimensions[1] = {0U};
  hsize_t maximum[1] = {H5S_UNLIMITED};
  hsize_t chunk[1] = {256U};
  SsStatus status = SS_OK;
  space = H5Screate_simple(1, dimensions, maximum);
  creation = H5Pcreate(H5P_DATASET_CREATE);
  if (space < 0 || creation < 0 || H5Pset_chunk(creation, 1, chunk) < 0) {
    status = SS_ERROR_HDF5;
  } else {
    dataset = H5Dcreate2(writer->file, path, file_type, space, H5P_DEFAULT, creation,
                         H5P_DEFAULT);
    if (dataset < 0) {
      status = SS_ERROR_HDF5;
    }
  }
  status = ss_result_hdf5_close(&dataset, H5Dclose, status);
  status = ss_result_hdf5_close(&creation, H5Pclose, status);
  status = ss_result_hdf5_close(&space, H5Sclose, status);
  return ss_status_allreduce_max(writer->communicator, status);
}

static SsStatus ss_result_writer_precreate_relations(SsResultWriter *writer) {
  hid_t state_file_type = -1;
  hid_t history_file_type = -1;
  SsResultHistoryKind kind;
  SsStatus status = SS_OK;
  state_file_type = ss_result_state_file_type();
  history_file_type = ss_result_history_file_type();
  if (state_file_type < 0 || history_file_type < 0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK) {
    status =
        ss_result_precreate_replicated(writer, "/states/index/rows", state_file_type);
  }
  for (kind = SS_RESULT_HISTORY_ACCEPTED;
       status == SS_OK && kind <= SS_RESULT_HISTORY_PMG_APPLY;
       kind = (SsResultHistoryKind)((int)kind + 1)) {
    char path[SS_RESULT_PATH_CAPACITY];
    if (snprintf(path, sizeof(path), "/history/%s/rows", ss_result_history_name(kind)) <
        0) {
      status = SS_ERROR_OVERFLOW;
    } else {
      status = ss_result_precreate_replicated(writer, path, history_file_type);
    }
  }
  status = ss_result_hdf5_close(&history_file_type, H5Tclose, status);
  status = ss_result_hdf5_close(&state_file_type, H5Tclose, status);
  return ss_status_allreduce_max(writer->communicator, status);
}

SsStatus ss_result_writer_write_relations(SsResultWriter *writer) {
  hid_t state_file_type = -1;
  hid_t state_memory_type = -1;
  SsStatus status = SS_OK;
  state_file_type = ss_result_state_file_type();
  state_memory_type = ss_result_state_memory_type();
  if (state_file_type < 0 || state_memory_type < 0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK) {
    status = ss_result_writer_flush_history(writer, (uint64_t)writer->history_count);
  }
  if (status == SS_OK) {
    status = ss_result_write_replicated(writer, "/states/index/rows", state_file_type,
                                        state_memory_type, writer->states,
                                        writer->state_count);
  }
  status = ss_result_hdf5_close(&state_memory_type, H5Tclose, status);
  status = ss_result_hdf5_close(&state_file_type, H5Tclose, status);
  return status;
}

SsStatus ss_result_writer_verify_hdf5(SsResultWriter *writer) {
  hid_t access = -1;
  hid_t file = -1;
  char value[SS_RESULT_TEXT_CAPACITY];
  uint64_t field_order = 0U;
  uint64_t geometry_order = 0U;
  uint64_t collective_plan_frozen = 0U;
  size_t index;
  SsStatus status = SS_OK;
  access = H5Pcreate(H5P_FILE_ACCESS);
  if (access < 0 || H5Pset_fapl_mpio(access, writer->communicator, MPI_INFO_NULL) < 0 ||
      H5Pset_all_coll_metadata_ops(access, 1U) < 0) {
    status = SS_ERROR_HDF5;
  } else {
    file = H5Fopen(writer->hdf5_path, H5F_ACC_RDONLY, access);
    if (file < 0) {
      status = SS_ERROR_HDF5;
    }
  }
  status = ss_status_allreduce_max(writer->communicator, status);
#define SS_RESULT_VERIFY_ROOT_TEXT(name, expected)                                     \
  do {                                                                                 \
    if (status == SS_OK) {                                                             \
      status = ss_result_hdf5_read_string_attribute(file, name, value, sizeof(value)); \
      if (status == SS_OK && strcmp(value, expected) != 0) {                           \
        status = SS_ERROR_INTEGRITY;                                                   \
      }                                                                                \
    }                                                                                  \
  } while (0)
  SS_RESULT_VERIFY_ROOT_TEXT("layout_id", "result-v1");
  SS_RESULT_VERIFY_ROOT_TEXT("problem_id", writer->problem_id);
  SS_RESULT_VERIFY_ROOT_TEXT("run_id", writer->run_id);
  SS_RESULT_VERIFY_ROOT_TEXT("build_id", writer->build_id);
  SS_RESULT_VERIFY_ROOT_TEXT("mesh_id", writer->mesh_id);
  SS_RESULT_VERIFY_ROOT_TEXT("geometry_id", writer->geometry_id);
  SS_RESULT_VERIFY_ROOT_TEXT("analysis_method", writer->analysis_method);
  SS_RESULT_VERIFY_ROOT_TEXT("collective_plan_version", "output-call-plan-v1");
  SS_RESULT_VERIFY_ROOT_TEXT("collective_plan_digest", writer->collective_plan_digest);
#undef SS_RESULT_VERIFY_ROOT_TEXT
  if (status == SS_OK) {
    status = ss_result_hdf5_read_u64_attribute(file, "field_order", &field_order);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_u64_attribute(file, "geometry_order", &geometry_order);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_u64_attribute(file, "collective_plan_frozen",
                                               &collective_plan_frozen);
  }
  if (status == SS_OK &&
      (field_order != writer->field_order || geometry_order != writer->geometry_order ||
       collective_plan_frozen != 1U)) {
    status = SS_ERROR_INTEGRITY;
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  for (index = 0U; status == SS_OK && index < writer->field_count; ++index) {
    char digest[SS_SHA256_HEX_CAPACITY + 8U] = {0};
    status = ss_result_verify_quantity_digest(file, writer->communicator,
                                              writer->fields[index].group_path, digest);
    if (status == SS_OK && strcmp(digest, writer->fields[index].logical_digest) != 0) {
      status = SS_ERROR_INTEGRITY;
    }
  }
  status = ss_result_hdf5_close(&file, H5Fclose, status);
  status = ss_result_hdf5_close(&access, H5Pclose, status);
  return ss_status_allreduce_max(writer->communicator, status);
}
