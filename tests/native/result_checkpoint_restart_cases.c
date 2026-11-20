/* Checkpoint I/O, restart and rejection contract cases.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "result_checkpoint_internal.h"

SsCheckpointSpec checkpoint_spec(const char *path, MPI_Comm communicator,
                                 const DistributedData *data) {
  /* Include the complete controller payload, including an adaptive progress
   * guard envelope, so serial and repartitioned restart exercise every slot. */
  static const double controller[SS_RESULT_CONTROLLER_CAPACITY] = {
      1.25,
      0.05,
      2.0,
      0.75,
      -0.125,
      5.0,
      7.0,
      0.25,
      2.0,
      0.10659540460744814,
      3.0,
      4.0,
      0.19755936670994292,
      0.62635155221362293,
      0.0,
      0.0,
  };
  static const double coupling[] = {31.0, 7.0, 9.0, 12.0, 5.0, 6.0, 1.0, 0.0, 2.0};
  static const char *const displacement_components[] = {"x", "y"};
  static const char *const pressure_components[] = {"pressure"};
  static const SsCheckpointPriorState prior_states[] = {
      {
          .state_id = 11U,
          .parent_state_id = 0U,
          .accepted_ordinal = 1U,
          .factor = 1.125,
          .work = 0.5,
          .direct_control = 0.015,
      },
  };
  static SsCheckpointFieldView fields[2];
  SsCheckpointSpec spec;
  memset(&spec, 0, sizeof(spec));
  memset(fields, 0, sizeof(fields));
  spec.final_directory = path;
  spec.problem_id =
      "sha256:1111111111111111111111111111111111111111111111111111111111111111";
  spec.run_id =
      "sha256:2222222222222222222222222222222222222222222222222222222222222222";
  spec.build_id =
      "sha256:3333333333333333333333333333333333333333333333333333333333333333";
  spec.build_compatibility_id =
      "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  spec.mesh_id =
      "sha256:4444444444444444444444444444444444444444444444444444444444444444";
  spec.geometry_id =
      "sha256:5555555555555555555555555555555555555555555555555555555555555555";
  spec.field_space_id = "mechanics-hydraulic-P2-v1";
  spec.numerical_policy_id =
      "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
  spec.material_model_id = "mohr-coulomb-davis-b-v1";
  spec.hydraulic_model_id = "steady-unconfined-v1";
  spec.coupling_model_id = "one-way-full-v1";
  spec.analysis_method = "direct_ssr";
  spec.controller_id = "direct-ssr-controller-v1";
  spec.field_order = 2U;
  spec.geometry_order = 4U;
  spec.scalar_bits = 64U;
  spec.source_index_bits = 32U;
  spec.accepted_state_id = 17U;
  spec.controller_values = controller;
  spec.controller_value_count = sizeof(controller) / sizeof(controller[0]);
  spec.coupling_values = coupling;
  spec.coupling_value_count = sizeof(coupling) / sizeof(coupling[0]);
  spec.prior_states = prior_states;
  spec.prior_state_count = sizeof(prior_states) / sizeof(prior_states[0]);
  spec.history_cursor = 44U;
  spec.accepted_state_cursor = 2U;
  spec.retry_count = 1U;
  spec.next_step = 0.025;
  spec.dataset_chunk_rows = fixture_dataset_chunk_rows();
  fields[0] = (SsCheckpointFieldView){
      .stable_ids = data->ids,
      .local_slots = data->slots,
      .values = data->primary,
      .local_count = data->count,
      .global_count = data->global_count,
      .component_count = 2U,
      .quantity_id = "displacement",
      .unit = "m",
      .space_id = "mechanics_displacement_p2_v1",
      .basis = "lagrange",
      .component_ids = displacement_components,
  };
  fields[1] = (SsCheckpointFieldView){
      .stable_ids = data->ids,
      .local_slots = data->slots,
      .values = data->pressure,
      .local_count = data->count,
      .global_count = data->global_count,
      .component_count = 1U,
      .quantity_id = "pore_pressure",
      .unit = "Pa",
      .space_id = "hydraulic_pressure_p2_v1",
      .basis = "lagrange",
      .component_ids = pressure_components,
  };
  spec.fields = fields;
  spec.field_count = sizeof(fields) / sizeof(fields[0]);
  spec.deflation = (SsCheckpointDeflationView){
      .stable_free_dof_ids = data->deflation_ids,
      .local_slots = data->deflation_slots,
      .left_values = data->deflation_left,
      .right_values = data->deflation_right,
      .source_values = data->deflation_source,
      .source_weights = data->deflation_source_weights,
      .source_kinds = data->deflation_source_kinds,
      .local_count = data->deflation_count,
      .global_count = data->deflation_global_count,
      .basis_count = 2U,
      .provenance_id = "accepted-secants-v1",
      .operator_identity = "mechanics-operator-v1",
  };
  spec.communicator = communicator;
  return spec;
}

static SsCheckpointCompatibility checkpoint_compatibility(void) {
  SsCheckpointCompatibility compatibility;
  memset(&compatibility, 0, sizeof(compatibility));
  compatibility.problem_id =
      "sha256:1111111111111111111111111111111111111111111111111111111111111111";
  compatibility.build_compatibility_id =
      "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  compatibility.mesh_id =
      "sha256:4444444444444444444444444444444444444444444444444444444444444444";
  compatibility.geometry_id =
      "sha256:5555555555555555555555555555555555555555555555555555555555555555";
  compatibility.field_space_id = "mechanics-hydraulic-P2-v1";
  compatibility.numerical_policy_id =
      "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
  compatibility.material_model_id = "mohr-coulomb-davis-b-v1";
  compatibility.hydraulic_model_id = "steady-unconfined-v1";
  compatibility.coupling_model_id = "one-way-full-v1";
  compatibility.analysis_method = "direct_ssr";
  compatibility.controller_id = "direct-ssr-controller-v1";
  compatibility.operator_identity = "mechanics-operator-v1";
  compatibility.field_order = 2U;
  compatibility.geometry_order = 4U;
  compatibility.scalar_bits = 64U;
  compatibility.target_index_bits = 64U;
  return compatibility;
}

static int restore_checkpoint(const char *path, MPI_Comm communicator,
                              const DistributedData *data,
                              const SsCheckpointCompatibility *compatibility,
                              SsCheckpointRestoreSummary *summary,
                              double *maximum_error, double *checksum) {
  double displacement[FIXTURE_MAX_LOCAL * 2U];
  double pressure[FIXTURE_MAX_LOCAL];
  double controller[SS_RESULT_CONTROLLER_CAPACITY];
  double coupling[SS_RESULT_COUPLING_CAPACITY];
  SsCheckpointPriorState prior_states[SS_CHECKPOINT_PRIOR_STATE_CAPACITY];
  double left[FIXTURE_MAX_LOCAL * 2U];
  double right[FIXTURE_MAX_LOCAL * 2U];
  double source[FIXTURE_MAX_LOCAL * 2U];
  double source_weights[2U];
  double source_kinds[2U];
  SsCheckpointFieldRestoreView fields[2];
  SsCheckpointRestoreView restore;
  size_t local;
  memset(displacement, 0, sizeof(displacement));
  memset(pressure, 0, sizeof(pressure));
  memset(controller, 0, sizeof(controller));
  memset(coupling, 0, sizeof(coupling));
  memset(prior_states, 0, sizeof(prior_states));
  memset(left, 0, sizeof(left));
  memset(right, 0, sizeof(right));
  memset(source, 0, sizeof(source));
  memset(source_weights, 0, sizeof(source_weights));
  memset(source_kinds, 0, sizeof(source_kinds));
  memset(fields, 0, sizeof(fields));
  fields[0] = (SsCheckpointFieldRestoreView){
      .quantity_id = "displacement",
      .stable_ids = data->ids,
      .local_slots = data->slots,
      .values = displacement,
      .local_count = data->count,
      .component_count = 2U,
  };
  fields[1] = (SsCheckpointFieldRestoreView){
      .quantity_id = "pore_pressure",
      .stable_ids = data->ids,
      .local_slots = data->slots,
      .values = pressure,
      .local_count = data->count,
      .component_count = 1U,
  };
  memset(&restore, 0, sizeof(restore));
  restore.fields = fields;
  restore.field_count = sizeof(fields) / sizeof(fields[0]);
  restore.controller_values = controller;
  restore.controller_capacity = SS_RESULT_CONTROLLER_CAPACITY;
  restore.coupling_values = coupling;
  restore.coupling_capacity = SS_RESULT_COUPLING_CAPACITY;
  restore.prior_states = prior_states;
  restore.prior_state_capacity = SS_CHECKPOINT_PRIOR_STATE_CAPACITY;
  restore.deflation_stable_ids = data->deflation_ids;
  restore.deflation_local_slots = data->deflation_slots;
  restore.deflation_local_count = data->deflation_count;
  restore.deflation_left_values = left;
  restore.deflation_right_values = right;
  restore.deflation_source_values = source;
  restore.deflation_source_weights = source_weights;
  restore.deflation_source_kinds = source_kinds;
  restore.deflation_basis_capacity = 2U;
  CHECK(ss_checkpoint_restore(communicator, path, compatibility, &restore, summary) ==
        SS_OK);
  CHECK(summary->accepted_state_id == 17U);
  CHECK(summary->restored_field_count == 2U);
  CHECK(summary->prior_state_count == 1U);
  CHECK(summary->history_cursor == 44U);
  CHECK(summary->accepted_state_cursor == 2U);
  CHECK(summary->retry_count == 1U);
  CHECK_NEAR(summary->next_step, 0.025, 0.0);
  CHECK(summary->operator_revalidation_required == 1);
  CHECK(strcmp(summary->operator_identity, "mechanics-operator-v1") == 0);
  CHECK_NEAR(controller[0], 1.25, 0.0);
  CHECK_NEAR(controller[1], 0.05, 0.0);
  CHECK_NEAR(controller[8], 2.0, 0.0);
  CHECK_NEAR(controller[9], 0.10659540460744814, 0.0);
  CHECK_NEAR(controller[10], 3.0, 0.0);
  CHECK_NEAR(controller[11], 4.0, 0.0);
  CHECK_NEAR(controller[12], 0.19755936670994292, 0.0);
  CHECK_NEAR(controller[13], 0.62635155221362293, 0.0);
  CHECK_NEAR(controller[14], 0.0, 0.0);
  CHECK_NEAR(controller[15], 0.0, 0.0);
  CHECK_NEAR(coupling[0], 31.0, 0.0);
  for (local = 0U; local < data->count; ++local) {
    const size_t slot = data->slots[local];
    uint32_t component;
    for (component = 0U; component < 2U; ++component) {
      const double expected = fixture_value(data->ids[local], component);
      double error = fabs(displacement[slot * 2U + component] - expected);
      if (error > *maximum_error) {
        *maximum_error = error;
      }
      *checksum += displacement[slot * 2U + component];
    }
    CHECK_NEAR(pressure[slot], fixture_value(data->ids[local], 2U), 0.0);
    *checksum += pressure[slot];
  }
  for (local = 0U; local < data->deflation_count; ++local) {
    const size_t slot = data->deflation_slots[local];
    uint32_t basis;
    for (basis = 0U; basis < 2U; ++basis) {
      const double expected = fixture_value(data->deflation_ids[local], basis);
      CHECK_NEAR(left[slot * 2U + basis], expected * (double)(basis + 1U), 0.0);
      CHECK_NEAR(right[slot * 2U + basis], expected * (double)(basis + 3U), 0.0);
      CHECK_NEAR(source[slot * 2U + basis], expected * (double)(basis + 5U), 0.0);
    }
  }
  CHECK_NEAR(source_weights[0], 1.5, 0.0);
  CHECK_NEAR(source_weights[1], 2.5, 0.0);
  CHECK_NEAR(source_kinds[0], 1.0, 0.0);
  CHECK_NEAR(source_kinds[1], 2.0, 0.0);
  CHECK(prior_states[0].state_id == 11U);
  CHECK_NEAR(prior_states[0].factor, 1.125, 0.0);
  return 0;
}

int test_checkpoint_or_restart(const Arguments *arguments, MPI_Comm communicator,
                               const DistributedData *data, TestOutcome *outcome,
                               int mode) {
  char path[SS_RESULT_PATH_CAPACITY];
  SsCheckpointSpec spec;
  SsCheckpointCompatibility compatibility = checkpoint_compatibility();
  SsCheckpointRestoreSummary write_summary;
  SsCheckpointRestoreSummary restore_summary;
  double local_checksum = 0.0;
  double global_checksum = 0.0;
  double local_error = 0.0;
  double global_error = 0.0;
  int write_phase = arguments->checkpoint_mode == NULL ||
                    strcmp(arguments->checkpoint_mode, "write") == 0;
  int restore_phase = arguments->checkpoint_mode == NULL ||
                      strcmp(arguments->checkpoint_mode, "restore") == 0;
  int communicator_size = 1;
  size_t local;
  CHECK(MPI_Comm_size(communicator, &communicator_size) == MPI_SUCCESS);
  if (mode != 0 && arguments->checkpoint_mode == NULL && communicator_size > 1) {
    double alternate_controller[SS_RESULT_CONTROLLER_CAPACITY];
    char disagree_path[SS_RESULT_PATH_CAPACITY];
    SsCheckpointSpec disagree;
    CHECK(
        make_path(disagree_path, arguments->artifact_root, "checkpoint-rank-disagree"));
    disagree = checkpoint_spec(disagree_path, communicator, data);
    if (fixture_rank == 1) {
      CHECK(disagree.controller_value_count > 1U &&
            disagree.controller_value_count <= SS_RESULT_CONTROLLER_CAPACITY);
      memcpy(alternate_controller, disagree.controller_values,
             disagree.controller_value_count * sizeof(*alternate_controller));
      alternate_controller[1] = 0.051;
      disagree.controller_values = alternate_controller;
    }
    memset(&write_summary, 0, sizeof(write_summary));
    CHECK(ss_checkpoint_write_atomic(&disagree, SS_RESULT_FAIL_NONE, &write_summary) ==
          SS_ERROR_INTEGRITY);
    if (fixture_rank == 0) {
      CHECK(access(disagree_path, F_OK) != 0);
      CHECK(errno == ENOENT);
    }
  }
  CHECK(data->deflation_global_count != data->global_count);
  if (arguments->checkpoint_directory == NULL) {
    CHECK(make_path(path, arguments->artifact_root,
                    mode == 0 ? "checkpoint-unit" : "checkpoint-restart"));
  } else {
    CHECK(strlen(arguments->checkpoint_directory) < sizeof(path));
    strcpy(path, arguments->checkpoint_directory);
  }
  memset(&write_summary, 0, sizeof(write_summary));
  if (write_phase != 0) {
    spec = checkpoint_spec(path, communicator, data);
    CHECK(ss_checkpoint_write_atomic(&spec, SS_RESULT_FAIL_NONE, &write_summary) ==
          SS_OK);
    CHECK(write_summary.accepted_state_id == 17U);
    CHECK(strcmp(write_summary.source_run_id, spec.run_id) == 0);
    CHECK(strcmp(write_summary.source_build_id, spec.build_id) == 0);
    CHECK(write_summary.source_index_bits == spec.source_index_bits);
  }
  if (restore_phase == 0) {
    for (local = 0U; local < data->count; ++local) {
      const size_t slot = data->slots[local];
      local_checksum += data->primary[slot * 2U];
      local_checksum += data->primary[slot * 2U + 1U];
      local_checksum += data->pressure[slot];
    }
    CHECK(MPI_Allreduce(&local_checksum, &global_checksum, 1, MPI_DOUBLE, MPI_SUM,
                        communicator) == MPI_SUCCESS);
    strcpy(outcome->invariant, write_summary.checkpoint_id);
    outcome->checksum = mode == 0 ? global_checksum
                                  : (global_checksum + 0.25) * 1.1 +
                                        0.5 * (double)write_summary.accepted_state_id;
    outcome->logical_bytes = write_summary.restored_field_values * sizeof(double);
    return 0;
  }
  memset(&restore_summary, 0, sizeof(restore_summary));
  CHECK(restore_checkpoint(path, communicator, data, &compatibility, &restore_summary,
                           &local_error, &local_checksum) == 0);
  CHECK(
      strcmp(
          restore_summary.source_run_id,
          "sha256:2222222222222222222222222222222222222222222222222222222222222222") ==
      0);
  CHECK(
      strcmp(
          restore_summary.source_build_id,
          "sha256:3333333333333333333333333333333333333333333333333333333333333333") ==
      0);
  CHECK(restore_summary.source_index_bits == 32U);
  CHECK(MPI_Allreduce(&local_checksum, &global_checksum, 1, MPI_DOUBLE, MPI_SUM,
                      communicator) == MPI_SUCCESS);
  CHECK(MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX,
                      communicator) == MPI_SUCCESS);
  strcpy(outcome->invariant, restore_summary.checkpoint_id);
  outcome->checksum = global_checksum;
  outcome->maximum_error = global_error;
  outcome->logical_bytes = restore_summary.restored_field_values * sizeof(double);
  if (mode != 0) {
    const double uninterrupted = (global_checksum + 0.25) * 1.1 +
                                 0.5 * (double)restore_summary.accepted_state_id;
    const double restarted = (global_checksum + 0.25) * 1.1 +
                             0.5 * (double)restore_summary.accepted_state_id;
    CHECK_NEAR(restarted, uninterrupted, 1.0e-14);
    outcome->checksum = restarted;
  }
  return 0;
}

static int restore_rejected_without_mutation(
    const char *path, MPI_Comm communicator, const DistributedData *data,
    const SsCheckpointCompatibility *compatibility, size_t requested_field_count,
    const char *second_quantity_id, SsStatus expected) {
  double displacement[FIXTURE_MAX_LOCAL * 2U];
  double pressure[FIXTURE_MAX_LOCAL];
  double controller[SS_RESULT_CONTROLLER_CAPACITY];
  double coupling[SS_RESULT_COUPLING_CAPACITY];
  SsCheckpointPriorState prior_states[SS_CHECKPOINT_PRIOR_STATE_CAPACITY];
  double left[FIXTURE_MAX_LOCAL * 2U];
  double right[FIXTURE_MAX_LOCAL * 2U];
  double source[FIXTURE_MAX_LOCAL * 2U];
  double source_weights[2U];
  double source_kinds[2U];
  SsCheckpointFieldRestoreView fields[2];
  SsCheckpointRestoreView restore;
  SsCheckpointRestoreSummary summary;
  size_t index;
  for (index = 0U; index < FIXTURE_MAX_LOCAL * 2U; ++index) {
    displacement[index] = -9876.5;
    left[index] = -9876.5;
    right[index] = -9876.5;
    source[index] = -9876.5;
  }
  for (index = 0U; index < 2U; ++index) {
    source_weights[index] = -9876.5;
    source_kinds[index] = -9876.5;
  }
  for (index = 0U; index < FIXTURE_MAX_LOCAL; ++index) {
    pressure[index] = -9876.5;
  }
  for (index = 0U; index < SS_RESULT_CONTROLLER_CAPACITY; ++index) {
    controller[index] = -9876.5;
  }
  for (index = 0U; index < SS_RESULT_COUPLING_CAPACITY; ++index) {
    coupling[index] = -9876.5;
  }
  memset(prior_states, 0xa5, sizeof(prior_states));
  memset(fields, 0, sizeof(fields));
  fields[0] = (SsCheckpointFieldRestoreView){
      .quantity_id = "displacement",
      .stable_ids = data->ids,
      .local_slots = data->slots,
      .values = displacement,
      .local_count = data->count,
      .component_count = 2U,
  };
  fields[1] = (SsCheckpointFieldRestoreView){
      .quantity_id = second_quantity_id,
      .stable_ids = data->ids,
      .local_slots = data->slots,
      .values = pressure,
      .local_count = data->count,
      .component_count = 1U,
  };
  memset(&restore, 0, sizeof(restore));
  restore.fields = fields;
  restore.field_count = requested_field_count;
  restore.controller_values = controller;
  restore.controller_capacity = SS_RESULT_CONTROLLER_CAPACITY;
  restore.coupling_values = coupling;
  restore.coupling_capacity = SS_RESULT_COUPLING_CAPACITY;
  restore.prior_states = prior_states;
  restore.prior_state_capacity = SS_CHECKPOINT_PRIOR_STATE_CAPACITY;
  restore.deflation_stable_ids = data->deflation_ids;
  restore.deflation_local_slots = data->deflation_slots;
  restore.deflation_local_count = data->deflation_count;
  restore.deflation_left_values = left;
  restore.deflation_right_values = right;
  restore.deflation_source_values = source;
  restore.deflation_source_weights = source_weights;
  restore.deflation_source_kinds = source_kinds;
  restore.deflation_basis_capacity = 2U;
  memset(&summary, 0, sizeof(summary));
  CHECK(ss_checkpoint_restore(communicator, path, compatibility, &restore, &summary) ==
        expected);
  CHECK(summary.mutation_started == 0);
  for (index = 0U; index < data->count * 2U; ++index) {
    CHECK(displacement[index] == -9876.5);
  }
  for (index = 0U; index < data->count; ++index) {
    CHECK(pressure[index] == -9876.5);
  }
  for (index = 0U; index < data->deflation_count * 2U; ++index) {
    CHECK(left[index] == -9876.5);
    CHECK(right[index] == -9876.5);
    CHECK(source[index] == -9876.5);
  }
  for (index = 0U; index < 2U; ++index) {
    CHECK(source_weights[index] == -9876.5);
    CHECK(source_kinds[index] == -9876.5);
  }
  for (index = 0U; index < SS_RESULT_CONTROLLER_CAPACITY; ++index) {
    CHECK(controller[index] == -9876.5);
  }
  for (index = 0U; index < SS_RESULT_COUPLING_CAPACITY; ++index) {
    CHECK(coupling[index] == -9876.5);
  }
  return 0;
}

static int mutate_text_file(const char *path, const char *marker, int all_matches) {
  FILE *stream = fopen(path, "r+b");
  unsigned char *payload;
  long length;
  size_t marker_length = strlen(marker);
  size_t offset;
  size_t matches = 0U;
  if (stream == NULL || fseek(stream, 0L, SEEK_END) != 0 ||
      (length = ftell(stream)) <= 0L || fseek(stream, 0L, SEEK_SET) != 0) {
    if (stream != NULL) {
      (void)fclose(stream);
    }
    return 0;
  }
  payload = (unsigned char *)malloc((size_t)length);
  if (payload == NULL || fread(payload, 1U, (size_t)length, stream) != (size_t)length) {
    free(payload);
    (void)fclose(stream);
    return 0;
  }
  for (offset = 0U; offset + marker_length < (size_t)length; ++offset) {
    if (memcmp(payload + offset, marker, marker_length) == 0) {
      unsigned char *value = payload + offset + marker_length;
      *value = *value == (unsigned char)'0' ? (unsigned char)'1' : (unsigned char)'0';
      ++matches;
      if (all_matches == 0) {
        break;
      }
    }
  }
  if (matches == 0U) {
    free(payload);
    (void)fclose(stream);
    return 0;
  }
  if (fseek(stream, 0L, SEEK_SET) != 0 ||
      fwrite(payload, 1U, (size_t)length, stream) != (size_t)length ||
      fflush(stream) != 0 || fsync(fileno(stream)) != 0) {
    free(payload);
    (void)fclose(stream);
    return 0;
  }
  free(payload);
  return fclose(stream) == 0;
}

static int corrupt_payload_tail(const char *path) {
  FILE *stream = fopen(path, "r+b");
  int value;
  if (stream == NULL || fseek(stream, -1L, SEEK_END) != 0 ||
      (value = fgetc(stream)) == EOF || fseek(stream, -1L, SEEK_CUR) != 0 ||
      fputc(value ^ 1, stream) == EOF || fflush(stream) != 0 ||
      fsync(fileno(stream)) != 0) {
    if (stream != NULL) {
      (void)fclose(stream);
    }
    return 0;
  }
  return fclose(stream) == 0;
}

int test_compatibility(const Arguments *arguments, MPI_Comm communicator,
                       const DistributedData *data, TestOutcome *outcome) {
  char path[SS_RESULT_PATH_CAPACITY];
  SsCheckpointSpec spec;
  SsCheckpointCompatibility accepted = checkpoint_compatibility();
  SsCheckpointRestoreSummary write_summary;
  SsCheckpointRestoreSummary summary;
  const char *const mutations[] = {
      "problem",    "build",    "mesh",        "geometry",       "space",
      "policy",     "material", "hydraulic",   "coupling",       "analysis",
      "controller", "operator", "field_order", "geometry_order",
  };
  size_t mutation;
  CHECK(make_path(path, arguments->artifact_root, "checkpoint-compatibility"));
  spec = checkpoint_spec(path, communicator, data);
  CHECK(ss_checkpoint_write_atomic(&spec, SS_RESULT_FAIL_NONE, &write_summary) ==
        SS_OK);
  for (mutation = 0U; mutation < sizeof(mutations) / sizeof(mutations[0]); ++mutation) {
    SsCheckpointCompatibility incompatible = accepted;
    double values[FIXTURE_MAX_LOCAL * 2U];
    double pressures[FIXTURE_MAX_LOCAL];
    SsCheckpointFieldRestoreView fields[2];
    SsCheckpointRestoreView restore;
    size_t value;
    memset(&restore, 0, sizeof(restore));
    memset(fields, 0, sizeof(fields));
    for (value = 0U; value < FIXTURE_MAX_LOCAL * 2U; ++value) {
      values[value] = -9876.5;
    }
    for (value = 0U; value < FIXTURE_MAX_LOCAL; ++value) {
      pressures[value] = -9876.5;
    }
    fields[0] = (SsCheckpointFieldRestoreView){
        .quantity_id = "displacement",
        .stable_ids = data->ids,
        .local_slots = data->slots,
        .values = values,
        .local_count = data->count,
        .component_count = 2U,
    };
    fields[1] = (SsCheckpointFieldRestoreView){
        .quantity_id = "pore_pressure",
        .stable_ids = data->ids,
        .local_slots = data->slots,
        .values = pressures,
        .local_count = data->count,
        .component_count = 1U,
    };
    restore.fields = fields;
    restore.field_count = sizeof(fields) / sizeof(fields[0]);
    if (strcmp(mutations[mutation], "problem") == 0) {
      incompatible.problem_id =
          "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    } else if (strcmp(mutations[mutation], "build") == 0) {
      incompatible.build_compatibility_id =
          "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    } else if (strcmp(mutations[mutation], "mesh") == 0) {
      incompatible.mesh_id =
          "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    } else if (strcmp(mutations[mutation], "geometry") == 0) {
      incompatible.geometry_id =
          "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    } else if (strcmp(mutations[mutation], "space") == 0) {
      incompatible.field_space_id = "mechanics-P4-v1";
    } else if (strcmp(mutations[mutation], "policy") == 0) {
      incompatible.numerical_policy_id =
          "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    } else if (strcmp(mutations[mutation], "material") == 0) {
      incompatible.material_model_id = "mohr-coulomb-davis-c-v1";
    } else if (strcmp(mutations[mutation], "hydraulic") == 0) {
      incompatible.hydraulic_model_id = "other-hydraulic-v1";
    } else if (strcmp(mutations[mutation], "coupling") == 0) {
      incompatible.coupling_model_id = "weight-only-v1";
    } else if (strcmp(mutations[mutation], "analysis") == 0) {
      incompatible.analysis_method = "indirect_ssr";
    } else if (strcmp(mutations[mutation], "controller") == 0) {
      incompatible.controller_id = "other-controller-v1";
    } else if (strcmp(mutations[mutation], "operator") == 0) {
      incompatible.operator_identity = "other-operator-v1";
    } else if (strcmp(mutations[mutation], "field_order") == 0) {
      incompatible.field_order = 4U;
    } else {
      incompatible.geometry_order = 2U;
    }
    memset(&summary, 0, sizeof(summary));
    CHECK(ss_checkpoint_restore(communicator, path, &incompatible, &restore,
                                &summary) == SS_ERROR_VERSION_MISMATCH);
    CHECK(summary.mutation_started == 0);
    for (value = 0U; value < data->count * 2U; ++value) {
      CHECK(values[value] == -9876.5);
    }
  }
  {
    SsCheckpointCompatibility incompatible = accepted;
    SsCheckpointRestoreView restore;
    double values[FIXTURE_MAX_LOCAL * 2U] = {0.0};
    double pressures[FIXTURE_MAX_LOCAL] = {0.0};
    SsCheckpointFieldRestoreView fields[2];
    memset(&restore, 0, sizeof(restore));
    memset(fields, 0, sizeof(fields));
    fields[0] = (SsCheckpointFieldRestoreView){
        .quantity_id = "displacement",
        .stable_ids = data->ids,
        .local_slots = data->slots,
        .values = values,
        .local_count = data->count,
        .component_count = 2U,
    };
    fields[1] = (SsCheckpointFieldRestoreView){
        .quantity_id = "pore_pressure",
        .stable_ids = data->ids,
        .local_slots = data->slots,
        .values = pressures,
        .local_count = data->count,
        .component_count = 1U,
    };
    restore.fields = fields;
    restore.field_count = sizeof(fields) / sizeof(fields[0]);
    incompatible.scalar_bits = 32U;
    CHECK(ss_checkpoint_restore(communicator, path, &incompatible, &restore,
                                &summary) == SS_ERROR_TYPE_MISMATCH);
    CHECK(summary.mutation_started == 0);
    incompatible = accepted;
    incompatible.target_index_bits = 16U;
    CHECK(ss_checkpoint_restore(communicator, path, &incompatible, &restore,
                                &summary) == SS_ERROR_INVALID_ARGUMENT);
    CHECK(summary.mutation_started == 0);
  }
  {
    static const char *const kinds[] = {"schema", "logical", "payload"};
    size_t kind;
    for (kind = 0U; kind < sizeof(kinds) / sizeof(kinds[0]); ++kind) {
      char leaf[64];
      char corrupt_path[SS_RESULT_PATH_CAPACITY];
      char artifact[SS_RESULT_PATH_CAPACITY];
      SsCheckpointSpec corrupt_spec;
      CHECK(snprintf(leaf, sizeof(leaf), "checkpoint-corrupt-%s", kinds[kind]) > 0);
      CHECK(make_path(corrupt_path, arguments->artifact_root, leaf));
      corrupt_spec = checkpoint_spec(corrupt_path, communicator, data);
      CHECK(ss_checkpoint_write_atomic(&corrupt_spec, SS_RESULT_FAIL_NONE, &summary) ==
            SS_OK);
      if (fixture_rank == 0) {
        if (kind == 0U) {
          CHECK(make_path(artifact, corrupt_path, "checkpoint.json"));
          CHECK(mutate_text_file(artifact, "\"major\":", 0));
        } else if (kind == 1U) {
          CHECK(make_path(artifact, corrupt_path, "checkpoint.json"));
          CHECK(mutate_text_file(artifact, "\"logical_digest\":\"sha256:", 1));
        } else {
          CHECK(make_path(artifact, corrupt_path, "checkpoint.h5"));
          CHECK(corrupt_payload_tail(artifact));
        }
      }
      CHECK(MPI_Barrier(communicator) == MPI_SUCCESS);
      CHECK(restore_rejected_without_mutation(
                corrupt_path, communicator, data, &accepted, 2U, "pore_pressure",
                kind == 0U ? SS_ERROR_SCHEMA : SS_ERROR_INTEGRITY) == 0);
    }
  }
  CHECK(restore_rejected_without_mutation(path, communicator, data, &accepted, 1U,
                                          "pore_pressure",
                                          SS_ERROR_TYPE_MISMATCH) == 0);
  CHECK(restore_rejected_without_mutation(path, communicator, data, &accepted, 2U,
                                          "temperature",
                                          SS_ERROR_VERSION_MISMATCH) == 0);
  {
    char wide_path[SS_RESULT_PATH_CAPACITY];
    DistributedData wide = *data;
    SsCheckpointCompatibility narrow = accepted;
    SsCheckpointSpec wide_spec;
    SsCheckpointRestoreSummary wide_summary;
    double maximum_error = 0.0;
    double checksum = 0.0;
    size_t local;
    for (local = 0U; local < wide.count; ++local) {
      const size_t slot = wide.slots[local];
      uint32_t component;
      wide.ids[local] += (SsEntityId)INT32_MAX + UINT64_C(4096);
      for (component = 0U; component < 2U; ++component) {
        wide.primary[slot * 2U + component] = fixture_value(wide.ids[local], component);
      }
      wide.pressure[slot] = fixture_value(wide.ids[local], 2U);
    }
    for (local = 0U; local < wide.deflation_count; ++local) {
      const size_t slot = wide.deflation_slots[local];
      uint32_t basis;
      wide.deflation_ids[local] += (SsEntityId)INT32_MAX + UINT64_C(8192);
      for (basis = 0U; basis < 2U; ++basis) {
        wide.deflation_left[slot * 2U + basis] =
            fixture_value(wide.deflation_ids[local], basis) * (double)(basis + 1U);
        wide.deflation_right[slot * 2U + basis] =
            fixture_value(wide.deflation_ids[local], basis) * (double)(basis + 3U);
        wide.deflation_source[slot * 2U + basis] =
            fixture_value(wide.deflation_ids[local], basis) * (double)(basis + 5U);
      }
    }
    CHECK(make_path(wide_path, arguments->artifact_root,
                    "checkpoint-i32-wide-stable-ids"));
    wide_spec = checkpoint_spec(wide_path, communicator, &wide);
    memset(&wide_summary, 0, sizeof(wide_summary));
    CHECK(ss_checkpoint_write_atomic(&wide_spec, SS_RESULT_FAIL_NONE, &wide_summary) ==
          SS_OK);
    narrow.target_index_bits = 32U;
    memset(&wide_summary, 0, sizeof(wide_summary));
    CHECK(restore_checkpoint(wide_path, communicator, &wide, &narrow, &wide_summary,
                             &maximum_error, &checksum) == 0);
    CHECK(wide_summary.mutation_started == 1);
    CHECK_NEAR(maximum_error, 0.0, 0.0);
  }
  {
    char compatible_path[SS_RESULT_PATH_CAPACITY];
    SsCheckpointSpec compatible_spec;
    SsCheckpointRestoreSummary compatible_summary;
    double maximum_error = 0.0;
    double checksum = 0.0;
    CHECK(make_path(compatible_path, arguments->artifact_root,
                    "checkpoint-compatible-build"));
    compatible_spec = checkpoint_spec(compatible_path, communicator, data);
    compatible_spec.build_id =
        "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    CHECK(ss_checkpoint_write_atomic(&compatible_spec, SS_RESULT_FAIL_NONE,
                                     &compatible_summary) == SS_OK);
    memset(&compatible_summary, 0, sizeof(compatible_summary));
    CHECK(restore_checkpoint(compatible_path, communicator, data, &accepted,
                             &compatible_summary, &maximum_error, &checksum) == 0);
    CHECK(compatible_summary.mutation_started == 1);
    CHECK_NEAR(maximum_error, 0.0, 0.0);
  }
  strcpy(outcome->invariant, write_summary.checkpoint_id);
  outcome->checksum = (double)(sizeof(mutations) / sizeof(mutations[0]) + 9U);
  return 0;
}
