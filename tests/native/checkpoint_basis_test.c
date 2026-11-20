/* OUT-CHECKPOINT-WIDE-BASIS-001: checkpoints with more than 16 deflation columns.
 *
 * Deflation source weights and kinds are replicated datasets with one value per
 * basis column (up to 64).  The checkpoint write/restore round trip must accept
 * a 24-column basis; the restored values and the logical digest must match.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/result/result.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { columns = 24, field_count = 12, dof_count = 9 };

static int failures;

#define CHECK(condition)                                                               \
  do {                                                                                 \
    if (!(condition)) {                                                                \
      fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition);    \
      ++failures;                                                                      \
    }                                                                                  \
  } while (0)

int main(int argc, char **argv) {
  static const char *const scalar[] = {"value"};
  SsEntityId field_ids[field_count];
  SsEntityId dof_ids[dof_count];
  double field_values[field_count];
  double left[dof_count * columns];
  double right[dof_count * columns];
  double source[dof_count * columns];
  double weights[columns];
  double kinds[columns];
  double controller[SS_RESULT_CONTROLLER_CAPACITY];
  double restored_field[field_count];
  double restored_left[dof_count * columns];
  double restored_right[dof_count * columns];
  double restored_source[dof_count * columns];
  double restored_weights[64];
  double restored_kinds[64];
  double restored_controller[SS_RESULT_CONTROLLER_CAPACITY];
  SsCheckpointPriorState restored_prior[SS_CHECKPOINT_PRIOR_STATE_CAPACITY];
  SsCheckpointFieldView field;
  SsCheckpointFieldRestoreView field_restore;
  SsCheckpointSpec spec;
  SsCheckpointCompatibility compatibility;
  SsCheckpointRestoreView restore;
  SsCheckpointRestoreSummary written;
  SsCheckpointRestoreSummary summary;
  char path[1024];
  int rank = 0;
  int total = 0;
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS || argc != 2) {
    return 2;
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  (void)snprintf(path, sizeof(path), "%s/wide-basis-checkpoint", argv[1]);
  if (rank == 0) {
    char command[1200];
    /* A published checkpoint directory is never replaced: start clean. */
    (void)snprintf(command, sizeof(command), "rm -rf '%s'", path);
    if (system(command) != 0) {
      ++failures;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  for (size_t index = 0U; index < field_count; ++index) {
    field_ids[index] = UINT64_C(0x100) + index;
    field_values[index] = 0.5 + (double)index;
  }
  for (size_t index = 0U; index < dof_count; ++index) {
    dof_ids[index] = UINT64_C(0x200) + index;
  }
  for (size_t index = 0U; index < dof_count * columns; ++index) {
    left[index] = 1.0 + 0.01 * (double)index;
    right[index] = -1.0 + 0.02 * (double)index;
    source[index] = 0.25 * (double)(index % 7U);
  }
  for (size_t index = 0U; index < columns; ++index) {
    weights[index] = 1.0 + (double)index;
    kinds[index] = (double)(index % 3U);
  }
  for (size_t index = 0U; index < SS_RESULT_CONTROLLER_CAPACITY; ++index) {
    controller[index] = 0.125 * (double)index;
  }
  field = (SsCheckpointFieldView){
      .stable_ids = field_ids,
      .values = field_values,
      .local_count = rank == 0 ? field_count : 0U,
      .global_count = field_count,
      .component_count = 1U,
      .quantity_id = "displacement_free_dofs",
      .unit = "m",
      .space_id = "mechanics_free_dofs_v1",
      .basis = "scalar_dof",
      .component_ids = scalar,
  };
  memset(&spec, 0, sizeof(spec));
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
  spec.field_space_id = "mechanics-P2-v1";
  spec.numerical_policy_id =
      "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
  spec.material_model_id = "mohr-coulomb-davis-b-v1";
  spec.hydraulic_model_id = "none";
  spec.coupling_model_id = "none";
  spec.analysis_method = "ssr_indirect";
  spec.controller_id = "ssr_indirect/classic";
  spec.field_order = 2U;
  spec.geometry_order = 1U;
  spec.scalar_bits = 64U;
  spec.source_index_bits = 32U;
  spec.accepted_state_id = 5U;
  spec.controller_values = controller;
  spec.controller_value_count = SS_RESULT_CONTROLLER_CAPACITY;
  spec.history_cursor = 3U;
  spec.accepted_state_cursor = 5U;
  spec.next_step = 0.5;
  spec.fields = &field;
  spec.field_count = 1U;
  spec.deflation = (SsCheckpointDeflationView){
      .stable_free_dof_ids = dof_ids,
      .left_values = left,
      .right_values = right,
      .source_values = source,
      .source_weights = weights,
      .source_kinds = kinds,
      .local_count = rank == 0 ? dof_count : 0U,
      .global_count = dof_count,
      .basis_count = columns,
      .provenance_id = "native-deflation-basis-v1",
      .operator_identity = "mechanics-operator-v1",
  };
  spec.communicator = MPI_COMM_WORLD;
  memset(&written, 0, sizeof(written));
  CHECK(ss_checkpoint_write_atomic(&spec, SS_RESULT_FAIL_NONE, &written) == SS_OK);
  memset(&compatibility, 0, sizeof(compatibility));
  compatibility.problem_id = spec.problem_id;
  compatibility.build_compatibility_id = spec.build_compatibility_id;
  compatibility.mesh_id = spec.mesh_id;
  compatibility.geometry_id = spec.geometry_id;
  compatibility.field_space_id = spec.field_space_id;
  compatibility.numerical_policy_id = spec.numerical_policy_id;
  compatibility.material_model_id = spec.material_model_id;
  compatibility.hydraulic_model_id = spec.hydraulic_model_id;
  compatibility.coupling_model_id = spec.coupling_model_id;
  compatibility.analysis_method = spec.analysis_method;
  compatibility.controller_id = spec.controller_id;
  compatibility.operator_identity = "mechanics-operator-v1";
  compatibility.field_order = 2U;
  compatibility.geometry_order = 1U;
  compatibility.scalar_bits = 64U;
  compatibility.target_index_bits = 32U;
  field_restore = (SsCheckpointFieldRestoreView){
      .quantity_id = "displacement_free_dofs",
      .stable_ids = field_ids,
      .values = restored_field,
      .local_count = rank == 0 ? field_count : 0U,
      .component_count = 1U,
  };
  memset(&restore, 0, sizeof(restore));
  restore.fields = &field_restore;
  restore.field_count = 1U;
  restore.controller_values = restored_controller;
  restore.controller_capacity = SS_RESULT_CONTROLLER_CAPACITY;
  restore.prior_states = restored_prior;
  restore.prior_state_capacity = SS_CHECKPOINT_PRIOR_STATE_CAPACITY;
  restore.deflation_stable_ids = dof_ids;
  restore.deflation_local_count = rank == 0 ? dof_count : 0U;
  restore.deflation_left_values = restored_left;
  restore.deflation_right_values = restored_right;
  restore.deflation_source_values = restored_source;
  restore.deflation_source_weights = restored_weights;
  restore.deflation_source_kinds = restored_kinds;
  restore.deflation_basis_capacity = 64U;
  memset(&summary, 0, sizeof(summary));
  CHECK(ss_checkpoint_restore(MPI_COMM_WORLD, path, &compatibility, &restore,
                              &summary) == SS_OK);
  CHECK(summary.deflation_basis_count == columns);
  CHECK(strcmp(summary.checkpoint_id, written.checkpoint_id) == 0);
  CHECK(memcmp(restored_weights, weights, sizeof(weights)) == 0);
  CHECK(memcmp(restored_kinds, kinds, sizeof(kinds)) == 0);
  if (rank == 0) {
    CHECK(memcmp(restored_left, left, sizeof(left)) == 0);
    CHECK(memcmp(restored_field, field_values, sizeof(field_values)) == 0);
  }
  MPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    printf("%s\n", total == 0 ? "OUT-CHECKPOINT-WIDE-BASIS-001 passed"
                              : "OUT-CHECKPOINT-WIDE-BASIS-001 failed");
  }
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}
