/* Validate compatibility and recover accepted checkpoint state.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "checkpoint_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static SsStatus ss_checkpoint_read_root_text(hid_t file, const char *name,
                                             const char *expected) {
  char value[SS_RESULT_TEXT_CAPACITY];
  const SsStatus status =
      ss_result_hdf5_read_string_attribute(file, name, value, sizeof(value));
  return status == SS_OK && expected != NULL && strcmp(value, expected) == 0
             ? SS_OK
             : SS_ERROR_VERSION_MISMATCH;
}

static SsStatus
ss_checkpoint_compatibility(hid_t file, const SsCheckpointCompatibility *compatibility,
                            SsStateId *accepted_state_id, uint32_t *source_index_bits,
                            uint64_t *field_count, uint32_t *basis_count,
                            char *build_id, char *run_id, char *deflation_provenance,
                            char *operator_identity, char *field_ids) {
  uint64_t schema_major = 0U;
  uint64_t scalar_bits = 0U;
  uint64_t field_order = 0U;
  uint64_t geometry_order = 0U;
  uint64_t source_bits = 0U;
  uint64_t basis = 0U;
  uint64_t deflation_state_version = 0U;
  char layout[SS_RESULT_TEXT_CAPACITY];
  SsStatus status;
  if (compatibility == NULL || !ss_valid_content_id(compatibility->problem_id) ||
      !ss_valid_content_id(compatibility->build_compatibility_id) ||
      !ss_valid_content_id(compatibility->mesh_id) ||
      !ss_valid_content_id(compatibility->geometry_id) ||
      !ss_valid_content_id(compatibility->numerical_policy_id) ||
      (compatibility->target_index_bits != 32U &&
       compatibility->target_index_bits != 64U)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status =
      ss_result_hdf5_read_string_attribute(file, "layout_id", layout, sizeof(layout));
  if (status != SS_OK || strcmp(layout, "checkpoint-v1") != 0 ||
      ss_result_hdf5_read_u64_attribute(file, "schema_major", &schema_major) != SS_OK ||
      schema_major != 1U) {
    return SS_ERROR_VERSION_MISMATCH;
  }
#define SS_CHECKPOINT_MATCH(name, expected)                                            \
  do {                                                                                 \
    if (status == SS_OK) {                                                             \
      status = ss_checkpoint_read_root_text(file, name, expected);                     \
    }                                                                                  \
  } while (0)
  SS_CHECKPOINT_MATCH("problem_id", compatibility->problem_id);
  SS_CHECKPOINT_MATCH("build_compatibility_id", compatibility->build_compatibility_id);
  SS_CHECKPOINT_MATCH("mesh_id", compatibility->mesh_id);
  SS_CHECKPOINT_MATCH("geometry_id", compatibility->geometry_id);
  SS_CHECKPOINT_MATCH("field_space_id", compatibility->field_space_id);
  SS_CHECKPOINT_MATCH("numerical_policy_id", compatibility->numerical_policy_id);
  SS_CHECKPOINT_MATCH("material_model_id", compatibility->material_model_id);
  SS_CHECKPOINT_MATCH("hydraulic_model_id", compatibility->hydraulic_model_id);
  SS_CHECKPOINT_MATCH("coupling_model_id", compatibility->coupling_model_id);
  SS_CHECKPOINT_MATCH("analysis_method", compatibility->analysis_method);
  SS_CHECKPOINT_MATCH("controller_id", compatibility->controller_id);
  SS_CHECKPOINT_MATCH("operator_identity", compatibility->operator_identity);
#undef SS_CHECKPOINT_MATCH
  if (status != SS_OK) {
    return status;
  }
  if (ss_result_hdf5_read_u64_attribute(file, "scalar_bits", &scalar_bits) != SS_OK ||
      scalar_bits != compatibility->scalar_bits) {
    return SS_ERROR_TYPE_MISMATCH;
  }
  if (ss_result_hdf5_read_u64_attribute(file, "field_order", &field_order) != SS_OK ||
      ss_result_hdf5_read_u64_attribute(file, "geometry_order", &geometry_order) !=
          SS_OK ||
      field_order != compatibility->field_order ||
      geometry_order != compatibility->geometry_order) {
    return SS_ERROR_VERSION_MISMATCH;
  }
  if (ss_result_hdf5_read_u64_attribute(file, "source_index_bits", &source_bits) !=
          SS_OK ||
      ss_result_hdf5_read_u64_attribute(file, "accepted_state_id", accepted_state_id) !=
          SS_OK ||
      ss_result_hdf5_read_u64_attribute(file, "field_count", field_count) != SS_OK ||
      ss_result_hdf5_read_u64_attribute(file, "deflation_basis_count", &basis) !=
          SS_OK ||
      ss_result_hdf5_read_u64_attribute(file, "deflation_state_version",
                                        &deflation_state_version) != SS_OK ||
      source_bits > UINT32_MAX || basis > UINT32_MAX ||
      ss_result_hdf5_read_string_attribute(file, "build_id", build_id,
                                           SS_RESULT_TEXT_CAPACITY) != SS_OK ||
      ss_result_hdf5_read_string_attribute(file, "run_id", run_id,
                                           SS_RESULT_TEXT_CAPACITY) != SS_OK ||
      ss_result_hdf5_read_string_attribute(file, "deflation_provenance_id",
                                           deflation_provenance,
                                           SS_RESULT_TEXT_CAPACITY) != SS_OK ||
      ss_result_hdf5_read_string_attribute(file, "operator_identity", operator_identity,
                                           SS_RESULT_TEXT_CAPACITY) != SS_OK ||
      ss_result_hdf5_read_string_attribute(file, "field_ids", field_ids,
                                           SS_CHECKPOINT_FIELD_IDS_CAPACITY) != SS_OK) {
    return SS_ERROR_SCHEMA;
  }
  if ((source_bits != 32U && source_bits != 64U) || deflation_state_version != 1U ||
      !ss_valid_content_id(build_id) || !ss_valid_content_id(run_id)) {
    return SS_ERROR_SCHEMA;
  }
  *source_index_bits = (uint32_t)source_bits;
  *basis_count = (uint32_t)basis;
  return SS_OK;
}

/* Read back one dataset at a time during publication. A restart keeps all
 * recovered values staged until the complete checkpoint has been verified. */
static SsStatus ss_checkpoint_read_values(hid_t file, MPI_Comm communicator,
                                          const char *path,
                                          const SsResultFileIndexPlan *plan,
                                          uint32_t columns, size_t count,
                                          const double *expected, double **staged) {
  double *values = malloc((count == 0U ? 1U : count) * sizeof(*values));
  SsStatus status = ss_status_allreduce_max(
      communicator, values == NULL ? SS_ERROR_OUT_OF_MEMORY : SS_OK);
  if (status == SS_OK) {
    status =
        ss_result_read_indexed_dataset(file, communicator, path, plan, columns, values);
  }
  if (status == SS_OK && staged == NULL && count != 0U &&
      memcmp(values, expected, count * sizeof(*values)) != 0) {
    status = SS_ERROR_INTEGRITY;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK && staged != NULL) {
    *staged = values;
  } else {
    free(values);
  }
  return status;
}

static SsStatus
ss_checkpoint_restore_internal(MPI_Comm communicator, const char *directory,
                               const SsCheckpointCompatibility *compatibility,
                               const SsCheckpointRestoreView *restore,
                               SsCheckpointRestoreSummary *summary,
                               const SsCheckpointSpec *written) {
  char envelope_id[SS_SHA256_HEX_CAPACITY + 8U] = {0};
  char envelope_logical[SS_SHA256_HEX_CAPACITY + 8U] = {0};
  char expected_payload[SS_SHA256_HEX_CAPACITY] = {0};
  char actual_payload[SS_SHA256_HEX_CAPACITY] = {0};
  char hdf5_path[SS_RESULT_PATH_CAPACITY] = {0};
  char build_id[SS_RESULT_TEXT_CAPACITY] = {0};
  char run_id[SS_RESULT_TEXT_CAPACITY] = {0};
  char deflation_provenance[SS_RESULT_TEXT_CAPACITY] = {0};
  char operator_identity[SS_RESULT_TEXT_CAPACITY] = {0};
  char checkpoint_field_ids[SS_CHECKPOINT_FIELD_IDS_CAPACITY] = {0};
  char requested_field_ids[SS_CHECKPOINT_FIELD_IDS_CAPACITY] = {0};
  char controller_digest[SS_SHA256_HEX_CAPACITY + 8U] = {0};
  char coupling_digest[SS_SHA256_HEX_CAPACITY + 8U] = {0};
  char prior_digest[SS_SHA256_HEX_CAPACITY + 8U] = {0};
  char cursor_digest[SS_SHA256_HEX_CAPACITY + 8U] = {0};
  char left_digest[SS_SHA256_HEX_CAPACITY + 8U] = "sha256:";
  char right_digest[SS_SHA256_HEX_CAPACITY + 8U] = "sha256:";
  char source_digest[SS_SHA256_HEX_CAPACITY + 8U] = "sha256:";
  char source_weights_digest[SS_SHA256_HEX_CAPACITY + 8U] = "sha256:";
  char source_kinds_digest[SS_SHA256_HEX_CAPACITY + 8U] = "sha256:";
  char logical[SS_SHA256_HEX_CAPACITY + 8U] = {0};
  char checkpoint_id[SS_SHA256_HEX_CAPACITY + 8U] = {0};
  uint64_t expected_bytes = 0U;
  uint64_t actual_bytes = 0U;
  SsStateId envelope_state = 0U;
  SsStateId accepted_state = 0U;
  uint32_t source_index_bits = 0U;
  uint32_t basis_count = 0U;
  uint64_t field_count = 0U;
  int rank = -1;
  int value;
  hid_t access = -1;
  hid_t file = -1;
  SsCheckpointFieldDigest *field_digests = NULL;
  SsResultFileIndexPlan **field_plans = NULL;
  double **field_values = NULL;
  size_t *field_value_counts = NULL;
  SsResultFileIndexPlan *deflation_plan = NULL;
  double *left = NULL;
  double *right = NULL;
  double *source = NULL;
  double source_weights[SS_RESULT_COMPONENT_CAPACITY] = {0.0};
  double source_kinds[SS_RESULT_COMPONENT_CAPACITY] = {0.0};
  size_t deflation_values = 0U;
  uint64_t deflation_global_count = 0U;
  uint32_t deflation_file_basis = 0U;
  double controller[SS_RESULT_CONTROLLER_CAPACITY] = {0.0};
  double coupling[SS_RESULT_COUPLING_CAPACITY] = {0.0};
  SsCheckpointPriorState prior_states[SS_CHECKPOINT_PRIOR_STATE_CAPACITY];
  SsCheckpointCursor cursor = {0U, 0U, 0U, 0.0};
  size_t controller_count = 0U;
  size_t coupling_count = 0U;
  size_t source_weight_count = 0U;
  size_t source_kind_count = 0U;
  size_t prior_count = 0U;
  size_t field;
  SsStatus status = SS_OK;
  if (directory == NULL || compatibility == NULL || restore == NULL ||
      summary == NULL || restore->fields == NULL || restore->field_count == 0U ||
      restore->field_count > SS_CHECKPOINT_FIELD_CAPACITY ||
      (restore->controller_capacity != 0U && restore->controller_values == NULL) ||
      (restore->coupling_capacity != 0U && restore->coupling_values == NULL) ||
      (restore->prior_state_capacity != 0U && restore->prior_states == NULL) ||
      MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(summary, 0, sizeof(*summary));
  for (field = 0U; field < restore->field_count; ++field) {
    const SsCheckpointFieldRestoreView *view = &restore->fields[field];
    if (!ss_result_name_valid(view->quantity_id) || view->component_count == 0U ||
        (view->local_count != 0U &&
         (view->stable_ids == NULL || (written == NULL && view->values == NULL))) ||
        (field != 0U &&
         strcmp(restore->fields[field - 1U].quantity_id, view->quantity_id) >= 0)) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
  }
  if (rank == 0) {
    status =
        ss_checkpoint_envelope_read(directory, envelope_id, envelope_logical,
                                    expected_payload, &expected_bytes, &envelope_state);
    if (status == SS_OK &&
        snprintf(hdf5_path, sizeof(hdf5_path), "%s/checkpoint.h5", directory) < 0) {
      status = SS_ERROR_OVERFLOW;
    }
    if (status == SS_OK) {
      status = ss_result_payload_digest(hdf5_path, actual_payload, &actual_bytes);
    }
    if (status == SS_OK && (expected_bytes != actual_bytes ||
                            strcmp(expected_payload, actual_payload) != 0)) {
      status = SS_ERROR_INTEGRITY;
    }
  }
  value = (int)status;
  if (MPI_Bcast(&value, 1, MPI_INT, 0, communicator) != MPI_SUCCESS ||
      MPI_Bcast(hdf5_path, SS_RESULT_PATH_CAPACITY, MPI_CHAR, 0, communicator) !=
          MPI_SUCCESS ||
      MPI_Bcast(envelope_id, SS_SHA256_HEX_CAPACITY + 7, MPI_CHAR, 0, communicator) !=
          MPI_SUCCESS ||
      MPI_Bcast(envelope_logical, SS_SHA256_HEX_CAPACITY + 7, MPI_CHAR, 0,
                communicator) != MPI_SUCCESS ||
      MPI_Bcast(&envelope_state, 1, MPI_UINT64_T, 0, communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  status = (SsStatus)value;
  if (status != SS_OK) {
    return status;
  }
  access = H5Pcreate(H5P_FILE_ACCESS);
  if (access < 0 || H5Pset_fapl_mpio(access, communicator, MPI_INFO_NULL) < 0 ||
      H5Pset_all_coll_metadata_ops(access, 1U) < 0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    file = H5Fopen(hdf5_path, H5F_ACC_RDONLY, access);
    if (file < 0) {
      status = SS_ERROR_HDF5;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    status = ss_checkpoint_compatibility(file, compatibility, &accepted_state,
                                         &source_index_bits, &field_count, &basis_count,
                                         build_id, run_id, deflation_provenance,
                                         operator_identity, checkpoint_field_ids);
  }
  if (status == SS_OK) {
    status = ss_checkpoint_restore_field_ids(restore, requested_field_ids,
                                             sizeof(requested_field_ids));
  }
  if (status == SS_OK &&
      (accepted_state != envelope_state || field_count != restore->field_count)) {
    status = SS_ERROR_TYPE_MISMATCH;
  }
  if (status == SS_OK && strcmp(checkpoint_field_ids, requested_field_ids) != 0) {
    status = SS_ERROR_VERSION_MISMATCH;
  }
  if (status == SS_OK &&
      ((written != NULL && basis_count != written->deflation.basis_count) ||
       basis_count > restore->deflation_basis_capacity ||
       restore->controller_capacity > SS_RESULT_CONTROLLER_CAPACITY ||
       restore->coupling_capacity > SS_RESULT_COUPLING_CAPACITY ||
       (basis_count != 0U && restore->deflation_local_count != 0U &&
        (restore->deflation_stable_ids == NULL ||
         (written == NULL && (restore->deflation_left_values == NULL ||
                              restore->deflation_right_values == NULL ||
                              restore->deflation_source_values == NULL)))) ||
       (basis_count != 0U && (restore->deflation_source_weights == NULL ||
                              restore->deflation_source_kinds == NULL)))) {
    status = SS_ERROR_TYPE_MISMATCH;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  field_digests =
      (SsCheckpointFieldDigest *)calloc(restore->field_count, sizeof(*field_digests));
  field_plans =
      (SsResultFileIndexPlan **)calloc(restore->field_count, sizeof(*field_plans));
  field_values = (double **)calloc(restore->field_count, sizeof(*field_values));
  field_value_counts =
      (size_t *)calloc(restore->field_count, sizeof(*field_value_counts));
  if (field_digests == NULL || field_plans == NULL || field_values == NULL ||
      field_value_counts == NULL) {
    status = SS_ERROR_OUT_OF_MEMORY;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (field_digests == NULL || field_plans == NULL || field_values == NULL ||
      field_value_counts == NULL) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  for (field = 0U; status == SS_OK && field < restore->field_count; ++field) {
    const SsCheckpointFieldRestoreView *view = &restore->fields[field];
    SsResultFileIndexSpec index_spec;
    char group_path[SS_RESULT_PATH_CAPACITY];
    char values_path[SS_RESULT_PATH_CAPACITY];
    uint64_t rows = 0U;
    uint32_t columns = 0U;
    if (snprintf(group_path, sizeof(group_path), "/accepted/fields/%s",
                 view->quantity_id) < 0 ||
        snprintf(values_path, sizeof(values_path), "%s/values", group_path) < 0) {
      status = SS_ERROR_OVERFLOW;
      break;
    }
    status = ss_checkpoint_dataset_shape(file, values_path, &rows, &columns);
    if (status == SS_OK &&
        (columns != view->component_count ||
         view->local_count > SIZE_MAX / (size_t)columns ||
         view->local_count * (size_t)columns > SIZE_MAX / sizeof(double))) {
      status = SS_ERROR_TYPE_MISMATCH;
    }
    if (status == SS_OK && compatibility->target_index_bits == 32U &&
        rows > (uint64_t)INT32_MAX / (uint64_t)columns) {
      status = SS_ERROR_OVERFLOW;
    }
    status = ss_status_allreduce_max(communicator, status);
    if (status != SS_OK) {
      break;
    }
    if (status == SS_OK) {
      memset(&index_spec, 0, sizeof(index_spec));
      index_spec.communicator = communicator;
      index_spec.stable_ids = view->stable_ids;
      index_spec.local_slots = view->local_slots;
      index_spec.local_count = view->local_count;
      index_spec.global_count = rows;
      status = ss_result_file_index_plan_create(&index_spec, &field_plans[field]);
    }
    if (status == SS_OK) {
      field_value_counts[field] = view->local_count * (size_t)columns;
      status = ss_checkpoint_read_values(
          file, communicator, group_path, field_plans[field], columns,
          field_value_counts[field],
          written == NULL ? NULL : written->fields[field].values,
          written == NULL ? &field_values[field] : NULL);
    }
    if (status == SS_OK) {
      status = ss_result_verify_indexed_dataset_digest(file, communicator, group_path,
                                                       field_digests[field].digest);
    }
    if (status == SS_OK) {
      status = ss_checkpoint_field_metadata(file, group_path, &field_digests[field]);
      field_digests[field].global_count = rows;
      field_digests[field].component_count = columns;
    }
    if (status == SS_OK &&
        strcmp(field_digests[field].quantity_id, view->quantity_id) != 0) {
      status = SS_ERROR_SCHEMA;
    }
    status = ss_status_allreduce_max(communicator, status);
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK && basis_count != 0U) {
    SsResultFileIndexSpec index_spec;
    memset(&index_spec, 0, sizeof(index_spec));
    status = ss_checkpoint_dataset_shape(
        file, "/deflation/left/values", &deflation_global_count, &deflation_file_basis);
    if (status == SS_OK &&
        (deflation_file_basis != basis_count ||
         restore->deflation_local_count > SIZE_MAX / (size_t)basis_count ||
         restore->deflation_local_count * (size_t)basis_count >
             SIZE_MAX / sizeof(double))) {
      status = SS_ERROR_TYPE_MISMATCH;
    }
    if (status == SS_OK && compatibility->target_index_bits == 32U &&
        deflation_global_count > (uint64_t)INT32_MAX) {
      status = SS_ERROR_OVERFLOW;
    }
    status = ss_status_allreduce_max(communicator, status);
    if (status == SS_OK) {
      index_spec.communicator = communicator;
      index_spec.stable_ids = restore->deflation_stable_ids;
      index_spec.local_slots = restore->deflation_local_slots;
      index_spec.local_count = restore->deflation_local_count;
      index_spec.global_count = deflation_global_count;
      status = ss_result_file_index_plan_create(&index_spec, &deflation_plan);
    }
    if (status == SS_OK) {
      const char *const paths[] = {"/deflation/left", "/deflation/right",
                                   "/deflation/source"};
      double **staged[] = {&left, &right, &source};
      const double *expected[] = {
          written == NULL ? NULL : written->deflation.left_values,
          written == NULL ? NULL : written->deflation.right_values,
          written == NULL ? NULL : written->deflation.source_values};
      deflation_values = restore->deflation_local_count * (size_t)basis_count;
      for (size_t i = 0U; status == SS_OK && i < sizeof(paths) / sizeof(*paths); ++i) {
        status = ss_checkpoint_read_values(file, communicator, paths[i], deflation_plan,
                                           basis_count, deflation_values, expected[i],
                                           written == NULL ? staged[i] : NULL);
      }
    }
    if (status == SS_OK) {
      status = ss_result_verify_indexed_dataset_digest(file, communicator,
                                                       "/deflation/left", left_digest);
    }
    if (status == SS_OK) {
      status = ss_result_verify_indexed_dataset_digest(
          file, communicator, "/deflation/right", right_digest);
    }
    if (status == SS_OK) {
      status = ss_result_verify_indexed_dataset_digest(
          file, communicator, "/deflation/source", source_digest);
    }
    if (status == SS_OK) {
      status = ss_checkpoint_read_replicated_double(
          file, communicator, rank, "/deflation/source_weights", source_weights,
          SS_RESULT_COMPONENT_CAPACITY, &source_weight_count, source_weights_digest);
    }
    if (status == SS_OK) {
      status = ss_checkpoint_read_replicated_double(
          file, communicator, rank, "/deflation/source_kinds", source_kinds,
          SS_RESULT_COMPONENT_CAPACITY, &source_kind_count, source_kinds_digest);
    }
    if (status == SS_OK &&
        (source_weight_count != basis_count || source_kind_count != basis_count)) {
      status = SS_ERROR_SCHEMA;
    }
    for (size_t source_index = 0U;
         status == SS_OK && source_index < (size_t)basis_count; ++source_index) {
      if (source_weights[source_index] <= 0.0 ||
          floor(source_kinds[source_index]) != source_kinds[source_index] ||
          source_kinds[source_index] < 0.0 || source_kinds[source_index] > 2.0) {
        status = SS_ERROR_INTEGRITY;
      }
    }
  } else if (status == SS_OK) {
    (void)ss_sha256_bytes((const unsigned char *)"", 0U, left_digest + 7U);
    (void)ss_sha256_bytes((const unsigned char *)"", 0U, right_digest + 7U);
    (void)ss_sha256_bytes((const unsigned char *)"", 0U, source_digest + 7U);
    (void)ss_sha256_bytes((const unsigned char *)"", 0U, source_weights_digest + 7U);
    (void)ss_sha256_bytes((const unsigned char *)"", 0U, source_kinds_digest + 7U);
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    status = ss_checkpoint_read_replicated_double(
        file, communicator, rank, "/analysis/controller", controller,
        restore->controller_capacity, &controller_count, controller_digest);
  }
  if (status == SS_OK) {
    status = ss_checkpoint_read_replicated_double(
        file, communicator, rank, "/coupling/accepted", coupling,
        restore->coupling_capacity, &coupling_count, coupling_digest);
  }
  if (status == SS_OK) {
    status = ss_checkpoint_read_prior(file, communicator, rank, prior_states,
                                      restore->prior_state_capacity, &prior_count,
                                      prior_digest);
  }
  if (status == SS_OK) {
    status =
        ss_checkpoint_read_cursor(file, communicator, rank, &cursor, cursor_digest);
  }
  if (status == SS_OK) {
    status = ss_checkpoint_logical_digest_values(
        communicator, compatibility->problem_id, build_id,
        compatibility->build_compatibility_id, compatibility->mesh_id,
        compatibility->geometry_id, compatibility->field_space_id,
        compatibility->numerical_policy_id, compatibility->material_model_id,
        compatibility->hydraulic_model_id, compatibility->coupling_model_id,
        compatibility->analysis_method, compatibility->controller_id,
        deflation_provenance, operator_identity, compatibility->field_order,
        compatibility->geometry_order, compatibility->scalar_bits, source_index_bits,
        accepted_state, field_digests, restore->field_count, left_digest, right_digest,
        source_digest, source_weights_digest, source_kinds_digest, controller_digest,
        coupling_digest, prior_digest, cursor_digest, logical);
  }
  if (status == SS_OK && strcmp(logical, envelope_logical) != 0) {
    status = SS_ERROR_INTEGRITY;
  }
  if (status == SS_OK) {
    status =
        ss_checkpoint_id(communicator, run_id, accepted_state, logical, checkpoint_id);
  }
  if (status == SS_OK && strcmp(checkpoint_id, envelope_id) != 0) {
    status = SS_ERROR_INTEGRITY;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    uint64_t restored_values = 0U;
    for (field = 0U; field < restore->field_count; ++field) {
      if (written == NULL && field_value_counts[field] != 0U) {
        memcpy(restore->fields[field].values, field_values[field],
               field_value_counts[field] * sizeof(double));
      }
      restored_values +=
          field_digests[field].global_count * field_digests[field].component_count;
    }
    if (written == NULL && deflation_values != 0U) {
      memcpy(restore->deflation_left_values, left, deflation_values * sizeof(double));
      memcpy(restore->deflation_right_values, right, deflation_values * sizeof(double));
      memcpy(restore->deflation_source_values, source,
             deflation_values * sizeof(double));
    }
    if (basis_count != 0U) {
      memcpy(restore->deflation_source_weights, source_weights,
             (size_t)basis_count * sizeof(double));
      memcpy(restore->deflation_source_kinds, source_kinds,
             (size_t)basis_count * sizeof(double));
    }
    if (controller_count != 0U) {
      memcpy(restore->controller_values, controller, controller_count * sizeof(double));
    }
    if (coupling_count != 0U) {
      memcpy(restore->coupling_values, coupling, coupling_count * sizeof(double));
    }
    if (prior_count != 0U) {
      memcpy(restore->prior_states, prior_states, prior_count * sizeof(*prior_states));
    }
    summary->mutation_started = written == NULL;
    summary->accepted_state_id = accepted_state;
    summary->restored_field_count = restore->field_count;
    summary->restored_field_values = restored_values;
    summary->restored_primary_values =
        field_digests[0].global_count * field_digests[0].component_count;
    summary->restored_deflation_values =
        deflation_global_count * (uint64_t)basis_count * 3U +
        (uint64_t)basis_count * 2U;
    summary->deflation_basis_count = basis_count;
    summary->prior_state_count = prior_count;
    summary->history_cursor = cursor.history_cursor;
    summary->accepted_state_cursor = cursor.accepted_state_cursor;
    summary->retry_count = cursor.retry_count;
    summary->next_step = cursor.next_step;
    summary->operator_revalidation_required = basis_count != 0U;
    (void)ss_result_copy_text(summary->checkpoint_id, sizeof(summary->checkpoint_id),
                              envelope_id);
    (void)ss_result_copy_text(summary->operator_identity,
                              sizeof(summary->operator_identity), operator_identity);
    (void)ss_result_copy_text(summary->source_run_id, sizeof(summary->source_run_id),
                              run_id);
    (void)ss_result_copy_text(summary->source_build_id,
                              sizeof(summary->source_build_id), build_id);
    summary->source_index_bits = source_index_bits;
  }

cleanup:
  if (ss_result_file_index_plan_destroy(&deflation_plan) != SS_OK && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (field_plans != NULL) {
    for (field = 0U; field < restore->field_count; ++field) {
      if (ss_result_file_index_plan_destroy(&field_plans[field]) != SS_OK &&
          status == SS_OK) {
        status = SS_ERROR_INTERNAL;
      }
    }
  }
  if (field_values != NULL) {
    for (field = 0U; field < restore->field_count; ++field) {
      free(field_values[field]);
    }
  }
  free(source);
  free(right);
  free(left);
  free(field_value_counts);
  free(field_values);
  free(field_plans);
  free(field_digests);
  status = ss_result_hdf5_close(&file, H5Fclose, status);
  status = ss_result_hdf5_close(&access, H5Pclose, status);
  return ss_status_allreduce_max(communicator, status);
}

SsStatus ss_checkpoint_restore(MPI_Comm communicator, const char *directory,
                               const SsCheckpointCompatibility *compatibility,
                               const SsCheckpointRestoreView *restore,
                               SsCheckpointRestoreSummary *summary) {
  return ss_checkpoint_restore_internal(communicator, directory, compatibility, restore,
                                        summary, NULL);
}

SsStatus ss_checkpoint_verify_written(const SsCheckpointSpec *spec,
                                      const SsCheckpointTransaction *transaction,
                                      const char *expected_checkpoint_id) {
  SsCheckpointCompatibility compatibility;
  SsCheckpointRestoreView restore;
  SsCheckpointRestoreSummary summary;
  SsCheckpointFieldRestoreView fields[SS_CHECKPOINT_FIELD_CAPACITY] = {0};
  double source_weights[SS_RESULT_COMPONENT_CAPACITY] = {0.0};
  double source_kinds[SS_RESULT_COMPONENT_CAPACITY] = {0.0};
  double controller[SS_RESULT_CONTROLLER_CAPACITY] = {0.0};
  double coupling[SS_RESULT_COUPLING_CAPACITY] = {0.0};
  SsCheckpointPriorState prior_states[SS_CHECKPOINT_PRIOR_STATE_CAPACITY];
  for (size_t field = 0U; field < spec->field_count; ++field) {
    const SsCheckpointFieldView *source = &spec->fields[field];
    fields[field] = (SsCheckpointFieldRestoreView){
        .quantity_id = source->quantity_id,
        .stable_ids = source->stable_ids,
        .local_slots = source->local_slots,
        .local_count = source->local_count,
        .component_count = source->component_count,
    };
  }
  memset(&compatibility, 0, sizeof(compatibility));
  compatibility.problem_id = spec->problem_id;
  compatibility.build_compatibility_id = spec->build_compatibility_id;
  compatibility.mesh_id = spec->mesh_id;
  compatibility.geometry_id = spec->geometry_id;
  compatibility.field_space_id = spec->field_space_id;
  compatibility.numerical_policy_id = spec->numerical_policy_id;
  compatibility.material_model_id = spec->material_model_id;
  compatibility.hydraulic_model_id = spec->hydraulic_model_id;
  compatibility.coupling_model_id = spec->coupling_model_id;
  compatibility.analysis_method = spec->analysis_method;
  compatibility.controller_id = spec->controller_id;
  compatibility.operator_identity = spec->deflation.operator_identity;
  compatibility.field_order = spec->field_order;
  compatibility.geometry_order = spec->geometry_order;
  compatibility.scalar_bits = spec->scalar_bits;
  compatibility.target_index_bits = spec->source_index_bits;
  memset(&restore, 0, sizeof(restore));
  restore.fields = fields;
  restore.field_count = spec->field_count;
  restore.controller_values = controller;
  restore.controller_capacity = spec->controller_value_count;
  restore.coupling_values = coupling;
  restore.coupling_capacity = spec->coupling_value_count;
  restore.prior_states = prior_states;
  restore.prior_state_capacity = spec->prior_state_count;
  restore.deflation_stable_ids = spec->deflation.stable_free_dof_ids;
  restore.deflation_local_slots = spec->deflation.local_slots;
  restore.deflation_local_count = spec->deflation.local_count;
  restore.deflation_source_weights = source_weights;
  restore.deflation_source_kinds = source_kinds;
  restore.deflation_basis_capacity = spec->deflation.basis_count;
  memset(&summary, 0, sizeof(summary));
  SsStatus status =
      ss_checkpoint_restore_internal(spec->communicator, transaction->staging_directory,
                                     &compatibility, &restore, &summary, spec);
  if (status == SS_OK &&
      (summary.mutation_started != 0 ||
       summary.accepted_state_id != spec->accepted_state_id ||
       summary.restored_field_count != spec->field_count ||
       summary.deflation_basis_count != spec->deflation.basis_count ||
       summary.prior_state_count != spec->prior_state_count ||
       summary.history_cursor != spec->history_cursor ||
       summary.accepted_state_cursor != spec->accepted_state_cursor ||
       summary.retry_count != spec->retry_count ||
       summary.next_step != spec->next_step ||
       strcmp(summary.checkpoint_id, expected_checkpoint_id) != 0)) {
    status = SS_ERROR_INTEGRITY;
  }
  if (status == SS_OK && spec->deflation.basis_count != 0U &&
      (memcmp(source_weights, spec->deflation.source_weights,
              (size_t)spec->deflation.basis_count * sizeof(double)) != 0 ||
       memcmp(source_kinds, spec->deflation.source_kinds,
              (size_t)spec->deflation.basis_count * sizeof(double)) != 0)) {
    status = SS_ERROR_INTEGRITY;
  }
  if (status == SS_OK && spec->controller_value_count != 0U &&
      memcmp(controller, spec->controller_values,
             spec->controller_value_count * sizeof(double)) != 0) {
    status = SS_ERROR_INTEGRITY;
  }
  if (status == SS_OK && spec->coupling_value_count != 0U &&
      memcmp(coupling, spec->coupling_values,
             spec->coupling_value_count * sizeof(double)) != 0) {
    status = SS_ERROR_INTEGRITY;
  }
  if (status == SS_OK && spec->prior_state_count != 0U &&
      memcmp(prior_states, spec->prior_states,
             spec->prior_state_count * sizeof(*prior_states)) != 0) {
    status = SS_ERROR_INTEGRITY;
  }

  return ss_status_allreduce_max(spec->communicator, status);
}
