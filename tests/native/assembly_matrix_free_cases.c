/* Serial and hybrid matrix-free operator contract cases.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "assembly_internal.h"

static int check_vector_relative(Vec actual, Vec expected, double tolerance) {
  Vec difference = NULL;
  PetscReal expected_norm = 0.0;
  PetscReal difference_norm = 0.0;
  CHECK(VecDuplicate(expected, &difference) == PETSC_SUCCESS);
  CHECK(VecCopy(actual, difference) == PETSC_SUCCESS);
  CHECK(VecAXPY(difference, -1.0, expected) == PETSC_SUCCESS);
  CHECK(VecNorm(expected, NORM_2, &expected_norm) == PETSC_SUCCESS);
  CHECK(VecNorm(difference, NORM_2, &difference_norm) == PETSC_SUCCESS);
  CHECK(isfinite((double)expected_norm) && isfinite((double)difference_norm));
  if ((double)difference_norm > tolerance * (1.0 + (double)expected_norm)) {
    fprintf(stderr, "vector mismatch: difference=%.17g expected=%.17g tolerance=%.3g\n",
            (double)difference_norm, (double)expected_norm, tolerance);
  }
  CHECK((double)difference_norm <= tolerance * (1.0 + (double)expected_norm));
  CHECK(VecDestroy(&difference) == PETSC_SUCCESS);
  return 0;
}

int test_matrix_free(int requested_threads) {
  const SsMcParameterInput material_input = {
      8.0e4,
      0.27,
      120.0,
      0.45,
      0.2,
      SS_MC_DAVIS_B,
      UINT64_C(1),
      "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  };
  const uint32_t dimensions[] = {2U, 3U};
  const uint32_t orders[] = {1U, 2U, 4U, 8U};
  SsMcParameters *material = NULL;
  SsMcError material_error = {0};
  size_t dimension_index;
  size_t order_index;
  CHECK(ss_mc_parameters_create(&material_input, &material, &material_error) == SS_OK);
  for (dimension_index = 0U;
       dimension_index < sizeof(dimensions) / sizeof(dimensions[0]);
       ++dimension_index) {
    uint64_t previous_matrix_free_bytes = 0U;
    for (order_index = 0U; order_index < sizeof(orders) / sizeof(orders[0]);
         ++order_index) {
      const SsFieldSpec field_spec = {orders[order_index], dimensions[dimension_index],
                                      0U, NULL};
      const SsMechanicsMaterialRegion region = {
          .material_value = 1U,
          .parameters = material,
      };
      const SsMechanicsOperatorSpec mechanics_spec = {
          .regions = &region,
          .region_count = 1U,
      };
      const SsOperatorSpec operator_spec = {
          .physics = SS_ASSEMBLY_MECHANICS,
          .determinism = requested_threads > 1 ? SS_ASSEMBLY_PERFORMANCE
                                               : SS_ASSEMBLY_QUALIFICATION,
          .openmp_work_threshold = 1U,
          .requested_threads = requested_threads,
          .mechanics = &mechanics_spec,
          .matrix_free = 1,
          .matrix_free_block_nodes =
              requested_threads > 1 && orders[order_index] == 4U ? 2U : 0U,
      };
      SsOperatorSpec assembled_operator_spec = operator_spec;
      SsMeshInput input =
          dimensions[dimension_index] == 2U ? square_input() : tetra_input();
      SsMesh *mesh = NULL;
      SsMeshError mesh_error = {0};
      SsFieldSpace *space = NULL;
      SsAssemblyPlan *plan = NULL;
      SsOperator *operator_handle = NULL;
      SsOperator *assembled_operator_handle = NULL;
      SsErrorRecord error = {0};
      SsOperatorState state = {0};
      SsOperatorState elastic_state = {0};
      Vec field = NULL;
      Vec zero = NULL;
      Vec direction = NULL;
      Vec assembled_action = NULL;
      Vec shell_action = NULL;
      Vec assembled_transpose = NULL;
      Vec shell_transpose = NULL;
      Vec assembled_diagonal = NULL;
      Vec shell_diagonal = NULL;
      Vec matrix_free_residual = NULL;
      Vec assembled_residual = NULL;
      Mat assembled = NULL;
      Mat elastic = NULL;
      Mat shell = NULL;
      Mat p1_transfer = NULL;
      Mat p1_expected = NULL;
      Mat p1_projected = NULL;
      Mat p1_diagonal_shell = NULL;
      SsFieldSpace *p1_space = NULL;
      SsAssemblyPlan *p1_plan = NULL;
      SsOperator *p1_diagonal_operator = NULL;
      Vec p1_field = NULL;
      Vec p1_direction = NULL;
      Vec p1_expected_action = NULL;
      Vec p1_projected_action = NULL;
      Vec p1_expected_transpose = NULL;
      Vec p1_projected_transpose = NULL;
      Vec p1_expected_diagonal = NULL;
      Vec p1_projected_diagonal = NULL;
      PetscScalar *values = NULL;
      PetscInt local_size = 0;
      PetscInt block_count = 0;
      const PetscInt *block_sizes = NULL;
      PetscInt assembled_block_count = 0;
      const PetscInt *assembled_block_sizes = NULL;
      PetscScalar *assembled_inverse_blocks = NULL;
      PetscScalar *shell_inverse_blocks = NULL;
      PetscErrorCode (*threaded_block_apply)(Mat, Vec, Vec) = NULL;
      PC reference_block_pc = NULL;
      PetscInt slot;
      PetscReal projected_norm = 0.0;
      PetscReal expected_norm = 0.0;
      SsAssemblyDiagnostics matrix_free_diagnostics = {0};
      SsAssemblyDiagnostics assembled_diagnostics = {0};
      SsOperatorProperties matrix_free_properties = {0};
      const double regularization = 0.13;
      const char *matrix_type = NULL;
      const size_t entity_node_cap =
          requested_threads > 1 && orders[order_index] == 4U ? 2U : 1U;
      const int entity_blocks = entity_node_cap > 1U;
      assembled_operator_spec.matrix_free = 0;
      CHECK(ss_mesh_create(PETSC_COMM_SELF, &input, &mesh, &mesh_error) == SS_OK);
      CHECK(ss_mesh_field_space_create(mesh, &field_spec, &space) == SS_OK);
      CHECK(ss_assembly_plan_create(PETSC_COMM_SELF, space, &plan, &error) == SS_OK);
      CHECK(ss_operator_create(plan, &operator_spec, &operator_handle, &error) ==
            SS_OK);
      CHECK(ss_operator_create(plan, &assembled_operator_spec,
                               &assembled_operator_handle, &error) == SS_OK);
      CHECK(ss_operator_diagnostics(operator_handle, &matrix_free_diagnostics) ==
            SS_OK);
      CHECK(ss_operator_diagnostics(assembled_operator_handle,
                                    &assembled_diagnostics) == SS_OK);
      CHECK(ss_operator_properties(operator_handle, &matrix_free_properties) == SS_OK);
      CHECK((matrix_free_properties.supported_requests & SS_ASSEMBLY_RESIDUAL) != 0U);
      CHECK((matrix_free_properties.supported_requests & SS_ASSEMBLY_POTENTIAL) != 0U);
      CHECK((matrix_free_properties.supported_requests & SS_ASSEMBLY_JACOBIAN) == 0U);
      CHECK(entity_blocks != 0 ||
            matrix_free_diagnostics.matrix_free_bytes > previous_matrix_free_bytes);
      CHECK(matrix_free_diagnostics.admitted_peak_bytes >=
            matrix_free_diagnostics.matrix_free_bytes);
      CHECK(matrix_free_diagnostics.shadow_bytes < assembled_diagnostics.shadow_bytes);
      if (entity_blocks == 0) {
        previous_matrix_free_bytes = matrix_free_diagnostics.matrix_free_bytes;
      }
      CHECK(ss_assembly_plan_create_vector(plan, &field) == SS_OK);
      CHECK(VecDuplicate(field, &zero) == PETSC_SUCCESS);
      CHECK(VecDuplicate(field, &direction) == PETSC_SUCCESS);
      CHECK(VecDuplicate(field, &assembled_action) == PETSC_SUCCESS);
      CHECK(VecDuplicate(field, &shell_action) == PETSC_SUCCESS);
      CHECK(VecDuplicate(field, &assembled_transpose) == PETSC_SUCCESS);
      CHECK(VecDuplicate(field, &shell_transpose) == PETSC_SUCCESS);
      CHECK(VecDuplicate(field, &assembled_diagonal) == PETSC_SUCCESS);
      CHECK(VecDuplicate(field, &shell_diagonal) == PETSC_SUCCESS);
      CHECK(VecDuplicate(field, &matrix_free_residual) == PETSC_SUCCESS);
      CHECK(VecDuplicate(field, &assembled_residual) == PETSC_SUCCESS);
      CHECK(ss_assembly_plan_create_matrix(plan, &assembled) == SS_OK);
      CHECK(ss_assembly_plan_create_matrix(plan, &elastic) == SS_OK);
      CHECK(ss_operator_jacobian(operator_handle, &state, assembled, &error) ==
            SS_ERROR_INVALID_ARGUMENT);
      CHECK(VecGetLocalSize(field, &local_size) == PETSC_SUCCESS);
      CHECK(VecGetArray(field, &values) == PETSC_SUCCESS);
      for (slot = 0; slot < local_size; ++slot) {
        values[slot] = (PetscScalar)(0.012 * sin(0.37 * (double)(slot + 1)) +
                                     0.004 * cos(0.19 * (double)(slot + 2)));
      }
      CHECK(VecRestoreArray(field, &values) == PETSC_SUCCESS);
      CHECK(VecGetArray(direction, &values) == PETSC_SUCCESS);
      for (slot = 0; slot < local_size; ++slot) {
        values[slot] = (PetscScalar)(0.31 * sin(0.53 * (double)(slot + 3)) -
                                     0.17 * cos(0.29 * (double)(slot + 1)));
      }
      CHECK(VecRestoreArray(direction, &values) == PETSC_SUCCESS);
      CHECK(VecSet(zero, 0.0) == PETSC_SUCCESS);
      state.field = field;
      state.reduction_factor = 1.35;
      state.external_load_factor = 0.7;
      elastic_state.field = zero;
      elastic_state.reduction_factor = 1.0;
      elastic_state.external_load_factor = 0.0;
      CHECK(ss_operator_residual(operator_handle, &state, matrix_free_residual,
                                 &error) == SS_OK);
      CHECK(ss_operator_residual(assembled_operator_handle, &state, assembled_residual,
                                 &error) == SS_OK);
      CHECK(check_vector_relative(matrix_free_residual, assembled_residual, 2.0e-13) ==
            0);
      CHECK(ss_operator_jacobian(assembled_operator_handle, &state, assembled,
                                 &error) == SS_OK);
      CHECK(ss_operator_jacobian(assembled_operator_handle, &elastic_state, elastic,
                                 &error) == SS_OK);
      CHECK(MatGetVariableBlockSizes(assembled, &assembled_block_count,
                                     &assembled_block_sizes) == PETSC_SUCCESS);
      CHECK(assembled_block_count > 0 && assembled_block_sizes != NULL);
      CHECK(MatScale(assembled, 1.0 - regularization) == PETSC_SUCCESS);
      CHECK(MatAXPY(assembled, regularization, elastic, SAME_NONZERO_PATTERN) ==
            PETSC_SUCCESS);
      CHECK(ss_operator_matrix_free_create(operator_handle, &shell, &error) == SS_OK);
      CHECK(MatGetType(shell, &matrix_type) == PETSC_SUCCESS);
      CHECK(strcmp(matrix_type, MATSHELL) == 0);
      CHECK(ss_operator_matrix_free_linearize(operator_handle, &state, regularization,
                                              &error) == SS_OK);
      if (orders[order_index] > 1U) {
        const SsFieldSpec p1_spec = {1U, dimensions[dimension_index], 0U, NULL};
        CHECK(ss_mesh_field_space_create(mesh, &p1_spec, &p1_space) == SS_OK);
        CHECK(ss_assembly_plan_create(PETSC_COMM_SELF, p1_space, &p1_plan, &error) ==
              SS_OK);
        CHECK(ss_assembly_plan_create_matrix(p1_plan, &p1_projected) == SS_OK);
        CHECK(ss_assembly_plan_create_vector(p1_plan, &p1_field) == SS_OK);
        CHECK(VecSet(p1_field, 0.0) == PETSC_SUCCESS);
        CHECK(ss_operator_create(p1_plan, &operator_spec, &p1_diagonal_operator,
                                 &error) == SS_OK);
        CHECK(ss_operator_matrix_free_create(p1_diagonal_operator, &p1_diagonal_shell,
                                             &error) == SS_OK);
        CHECK(ss_field_space_transfer_create(p1_space, space, &p1_transfer) == SS_OK);
        CHECK(MatPtAP(assembled, p1_transfer, MAT_INITIAL_MATRIX, 2.0, &p1_expected) ==
              PETSC_SUCCESS);
        CHECK(ss_operator_matrix_free_project_p1_jacobian(
                  operator_handle, p1_plan, p1_projected, &error) == SS_OK);
        CHECK(MatNorm(p1_expected, NORM_FROBENIUS, &expected_norm) == PETSC_SUCCESS);
        CHECK(MatAXPY(p1_projected, -1.0, p1_expected, SAME_NONZERO_PATTERN) ==
              PETSC_SUCCESS);
        CHECK(MatNorm(p1_projected, NORM_FROBENIUS, &projected_norm) == PETSC_SUCCESS);
        CHECK((double)projected_norm <= 2.0e-11 * (1.0 + (double)expected_norm));
        CHECK(ss_operator_matrix_free_project_coarse(
                  operator_handle, p1_diagonal_operator, &error) == SS_OK);
        CHECK(VecDuplicate(p1_field, &p1_expected_diagonal) == PETSC_SUCCESS);
        CHECK(VecDuplicate(p1_field, &p1_projected_diagonal) == PETSC_SUCCESS);
        CHECK(MatGetDiagonal(p1_expected, p1_expected_diagonal) == PETSC_SUCCESS);
        CHECK(MatGetDiagonal(p1_diagonal_shell, p1_projected_diagonal) ==
              PETSC_SUCCESS);
        CHECK(check_vector_relative(p1_projected_diagonal, p1_expected_diagonal,
                                    2.0e-11) == 0);
        CHECK(VecDuplicate(p1_field, &p1_direction) == PETSC_SUCCESS);
        CHECK(VecDuplicate(p1_field, &p1_expected_action) == PETSC_SUCCESS);
        CHECK(VecDuplicate(p1_field, &p1_projected_action) == PETSC_SUCCESS);
        CHECK(VecDuplicate(p1_field, &p1_expected_transpose) == PETSC_SUCCESS);
        CHECK(VecDuplicate(p1_field, &p1_projected_transpose) == PETSC_SUCCESS);
        CHECK(VecGetLocalSize(p1_direction, &local_size) == PETSC_SUCCESS);
        CHECK(VecGetArray(p1_direction, &values) == PETSC_SUCCESS);
        for (slot = 0; slot < local_size; ++slot) {
          values[slot] = (PetscScalar)(0.23 * sin(0.41 * (double)(slot + 1)) -
                                       0.09 * cos(0.27 * (double)(slot + 4)));
        }
        CHECK(VecRestoreArray(p1_direction, &values) == PETSC_SUCCESS);
        CHECK(MatMult(p1_expected, p1_direction, p1_expected_action) == PETSC_SUCCESS);
        CHECK(MatMult(p1_diagonal_shell, p1_direction, p1_projected_action) ==
              PETSC_SUCCESS);
        CHECK(check_vector_relative(p1_projected_action, p1_expected_action, 2.0e-11) ==
              0);
        CHECK(MatMultTranspose(p1_expected, p1_direction, p1_expected_transpose) ==
              PETSC_SUCCESS);
        CHECK(MatMultTranspose(p1_diagonal_shell, p1_direction,
                               p1_projected_transpose) == PETSC_SUCCESS);
        CHECK(check_vector_relative(p1_projected_transpose, p1_expected_transpose,
                                    2.0e-11) == 0);
      }
      if (orders[order_index] >= 4U) {
        const SsFieldSpec p2_spec = {2U, dimensions[dimension_index], 0U, NULL};
        SsFieldSpace *p2_space = NULL;
        SsAssemblyPlan *p2_plan = NULL;
        SsOperator *p2_operator = NULL;
        Vec p2_direction = NULL;
        Vec p2_expected_action = NULL;
        Vec p2_projected_action = NULL;
        Vec p2_expected_transpose = NULL;
        Vec p2_projected_transpose = NULL;
        Vec p2_expected_diagonal = NULL;
        Vec p2_projected_diagonal = NULL;
        Mat p2_transfer = NULL;
        Mat p2_expected = NULL;
        Mat p2_shell = NULL;
        CHECK(ss_mesh_field_space_create(mesh, &p2_spec, &p2_space) == SS_OK);
        CHECK(ss_assembly_plan_create(PETSC_COMM_SELF, p2_space, &p2_plan, &error) ==
              SS_OK);
        CHECK(ss_operator_create(p2_plan, &operator_spec, &p2_operator, &error) ==
              SS_OK);
        CHECK(ss_operator_matrix_free_create(p2_operator, &p2_shell, &error) == SS_OK);
        CHECK(ss_field_space_transfer_create(p2_space, space, &p2_transfer) == SS_OK);
        CHECK(MatPtAP(assembled, p2_transfer, MAT_INITIAL_MATRIX, 2.0, &p2_expected) ==
              PETSC_SUCCESS);
        CHECK(ss_operator_matrix_free_project_coarse(operator_handle, p2_operator,
                                                     &error) == SS_OK);
        CHECK(MatCreateVecs(p2_shell, &p2_direction, NULL) == PETSC_SUCCESS);
        CHECK(VecDuplicate(p2_direction, &p2_expected_action) == PETSC_SUCCESS);
        CHECK(VecDuplicate(p2_direction, &p2_projected_action) == PETSC_SUCCESS);
        CHECK(VecDuplicate(p2_direction, &p2_expected_transpose) == PETSC_SUCCESS);
        CHECK(VecDuplicate(p2_direction, &p2_projected_transpose) == PETSC_SUCCESS);
        CHECK(VecDuplicate(p2_direction, &p2_expected_diagonal) == PETSC_SUCCESS);
        CHECK(VecDuplicate(p2_direction, &p2_projected_diagonal) == PETSC_SUCCESS);
        CHECK(VecGetLocalSize(p2_direction, &local_size) == PETSC_SUCCESS);
        CHECK(VecGetArray(p2_direction, &values) == PETSC_SUCCESS);
        for (slot = 0; slot < local_size; ++slot) {
          values[slot] = (PetscScalar)(0.17 * sin(0.43 * (double)(slot + 2)) -
                                       0.08 * cos(0.31 * (double)(slot + 5)));
        }
        CHECK(VecRestoreArray(p2_direction, &values) == PETSC_SUCCESS);
        CHECK(MatMult(p2_expected, p2_direction, p2_expected_action) == PETSC_SUCCESS);
        CHECK(MatMult(p2_shell, p2_direction, p2_projected_action) == PETSC_SUCCESS);
        CHECK(check_vector_relative(p2_projected_action, p2_expected_action, 2.0e-11) ==
              0);
        CHECK(MatMultTranspose(p2_expected, p2_direction, p2_expected_transpose) ==
              PETSC_SUCCESS);
        CHECK(MatMultTranspose(p2_shell, p2_direction, p2_projected_transpose) ==
              PETSC_SUCCESS);
        CHECK(check_vector_relative(p2_projected_transpose, p2_expected_transpose,
                                    2.0e-11) == 0);
        CHECK(MatGetDiagonal(p2_expected, p2_expected_diagonal) == PETSC_SUCCESS);
        CHECK(MatGetDiagonal(p2_shell, p2_projected_diagonal) == PETSC_SUCCESS);
        CHECK(check_vector_relative(p2_projected_diagonal, p2_expected_diagonal,
                                    2.0e-11) == 0);
        CHECK(VecDestroy(&p2_projected_diagonal) == PETSC_SUCCESS);
        CHECK(VecDestroy(&p2_expected_diagonal) == PETSC_SUCCESS);
        CHECK(VecDestroy(&p2_projected_transpose) == PETSC_SUCCESS);
        CHECK(VecDestroy(&p2_expected_transpose) == PETSC_SUCCESS);
        CHECK(VecDestroy(&p2_projected_action) == PETSC_SUCCESS);
        CHECK(VecDestroy(&p2_expected_action) == PETSC_SUCCESS);
        CHECK(VecDestroy(&p2_direction) == PETSC_SUCCESS);
        CHECK(MatDestroy(&p2_shell) == PETSC_SUCCESS);
        CHECK(MatDestroy(&p2_expected) == PETSC_SUCCESS);
        CHECK(MatDestroy(&p2_transfer) == PETSC_SUCCESS);
        CHECK(ss_operator_destroy(&p2_operator) == SS_OK);
        CHECK(ss_assembly_plan_destroy(&p2_plan) == SS_OK);
        CHECK(ss_field_space_destroy(&p2_space) == SS_OK);
      }
      CHECK(MatMult(assembled, direction, assembled_action) == PETSC_SUCCESS);
      CHECK(MatMult(shell, direction, shell_action) == PETSC_SUCCESS);
      if (check_vector_relative(shell_action, assembled_action, 2.0e-11) != 0) {
        fprintf(stderr, "matrix-free mismatch: dimension=%u order=%u threads=%d\n",
                dimensions[dimension_index], orders[order_index], requested_threads);
        return 1;
      }
      CHECK(MatMultTranspose(assembled, direction, assembled_transpose) ==
            PETSC_SUCCESS);
      CHECK(MatMultTranspose(shell, direction, shell_transpose) == PETSC_SUCCESS);
      CHECK(check_vector_relative(shell_transpose, assembled_transpose, 2.0e-11) == 0);
      CHECK(MatGetDiagonal(assembled, assembled_diagonal) == PETSC_SUCCESS);
      CHECK(MatGetDiagonal(shell, shell_diagonal) == PETSC_SUCCESS);
      CHECK(check_vector_relative(shell_diagonal, assembled_diagonal, 2.0e-11) == 0);
      CHECK(MatGetVariableBlockSizes(shell, &block_count, &block_sizes) ==
            PETSC_SUCCESS);
      CHECK(block_count > 0 && block_sizes != NULL);
      if (entity_blocks != 0 ? block_count > assembled_block_count
                             : block_count != assembled_block_count) {
        fprintf(stderr,
                "variable-block mismatch: dimension=%u order=%u requested_cap=%zu "
                "shell=%lld assembled=%lld\n",
                dimensions[dimension_index], orders[order_index], entity_node_cap,
                (long long)block_count, (long long)assembled_block_count);
        return 1;
      }
      {
        size_t block_entries = 0U;
        PetscInt block;
        int observed_entity_block = 0;
        for (block = 0; block < block_count; ++block) {
          CHECK(block_sizes[block] > 0 &&
                block_sizes[block] <=
                    (PetscInt)entity_node_cap * (PetscInt)dimensions[dimension_index]);
          if (entity_blocks == 0) {
            CHECK(block_sizes[block] == assembled_block_sizes[block]);
          } else if (block_sizes[block] > (PetscInt)dimensions[dimension_index]) {
            observed_entity_block = 1;
          }
          block_entries += (size_t)block_sizes[block] * (size_t)block_sizes[block];
        }
        CHECK(observed_entity_block != 0 ? block_count < assembled_block_count
                                         : block_count == assembled_block_count);
        assembled_inverse_blocks =
            (PetscScalar *)calloc(block_entries, sizeof(*assembled_inverse_blocks));
        shell_inverse_blocks =
            (PetscScalar *)calloc(block_entries, sizeof(*shell_inverse_blocks));
        CHECK(assembled_inverse_blocks != NULL && shell_inverse_blocks != NULL);
        if (entity_blocks != 0) {
          CHECK(MatSetVariableBlockSizes(assembled, block_count, block_sizes) ==
                PETSC_SUCCESS);
        }
        CHECK(MatInvertVariableBlockDiagonal(assembled, block_count, block_sizes,
                                             assembled_inverse_blocks) ==
              PETSC_SUCCESS);
        CHECK(MatInvertVariableBlockDiagonal(shell, block_count, block_sizes,
                                             shell_inverse_blocks) == PETSC_SUCCESS);
        for (size_t entry = 0U; entry < block_entries; ++entry) {
          CHECK_NEAR(PetscRealPart(shell_inverse_blocks[entry]),
                     PetscRealPart(assembled_inverse_blocks[entry]), 2.0e-11);
        }
        CHECK(PetscObjectQueryFunction((PetscObject)shell,
                                       "SsMatApplyInverseVariableBlocks_C",
                                       &threaded_block_apply) == PETSC_SUCCESS);
        if (entity_blocks != 0) {
          CHECK(threaded_block_apply != NULL);
          CHECK(PCCreate(PETSC_COMM_SELF, &reference_block_pc) == PETSC_SUCCESS);
          CHECK(PCSetOperators(reference_block_pc, assembled, assembled) ==
                PETSC_SUCCESS);
          CHECK(PCSetType(reference_block_pc, PCVPBJACOBI) == PETSC_SUCCESS);
          CHECK(PCSetUp(reference_block_pc) == PETSC_SUCCESS);
          CHECK(PCApply(reference_block_pc, direction, assembled_action) ==
                PETSC_SUCCESS);
          CHECK((*threaded_block_apply)(shell, direction, shell_action) ==
                PETSC_SUCCESS);
          CHECK(check_vector_relative(shell_action, assembled_action, 2.0e-13) == 0);
          CHECK(PCDestroy(&reference_block_pc) == PETSC_SUCCESS);
        } else {
          CHECK(threaded_block_apply == NULL);
        }
        free(shell_inverse_blocks);
        free(assembled_inverse_blocks);
        shell_inverse_blocks = NULL;
        assembled_inverse_blocks = NULL;
      }
      CHECK(MatDestroy(&shell) == PETSC_SUCCESS);
      CHECK(MatDestroy(&p1_projected) == PETSC_SUCCESS);
      CHECK(MatDestroy(&p1_expected) == PETSC_SUCCESS);
      CHECK(MatDestroy(&p1_transfer) == PETSC_SUCCESS);
      CHECK(MatDestroy(&p1_diagonal_shell) == PETSC_SUCCESS);
      CHECK(MatDestroy(&elastic) == PETSC_SUCCESS);
      CHECK(MatDestroy(&assembled) == PETSC_SUCCESS);
      CHECK(VecDestroy(&assembled_residual) == PETSC_SUCCESS);
      CHECK(VecDestroy(&matrix_free_residual) == PETSC_SUCCESS);
      CHECK(VecDestroy(&shell_diagonal) == PETSC_SUCCESS);
      CHECK(VecDestroy(&assembled_diagonal) == PETSC_SUCCESS);
      CHECK(VecDestroy(&shell_transpose) == PETSC_SUCCESS);
      CHECK(VecDestroy(&assembled_transpose) == PETSC_SUCCESS);
      CHECK(VecDestroy(&shell_action) == PETSC_SUCCESS);
      CHECK(VecDestroy(&assembled_action) == PETSC_SUCCESS);
      CHECK(VecDestroy(&direction) == PETSC_SUCCESS);
      CHECK(VecDestroy(&zero) == PETSC_SUCCESS);
      CHECK(VecDestroy(&field) == PETSC_SUCCESS);
      CHECK(VecDestroy(&p1_projected_diagonal) == PETSC_SUCCESS);
      CHECK(VecDestroy(&p1_expected_diagonal) == PETSC_SUCCESS);
      CHECK(VecDestroy(&p1_projected_transpose) == PETSC_SUCCESS);
      CHECK(VecDestroy(&p1_expected_transpose) == PETSC_SUCCESS);
      CHECK(VecDestroy(&p1_projected_action) == PETSC_SUCCESS);
      CHECK(VecDestroy(&p1_expected_action) == PETSC_SUCCESS);
      CHECK(VecDestroy(&p1_direction) == PETSC_SUCCESS);
      CHECK(VecDestroy(&p1_field) == PETSC_SUCCESS);
      CHECK(ss_operator_destroy(&assembled_operator_handle) == SS_OK);
      CHECK(ss_operator_destroy(&operator_handle) == SS_OK);
      CHECK(ss_operator_destroy(&p1_diagonal_operator) == SS_OK);
      CHECK(ss_assembly_plan_destroy(&p1_plan) == SS_OK);
      CHECK(ss_field_space_destroy(&p1_space) == SS_OK);
      CHECK(ss_assembly_plan_destroy(&plan) == SS_OK);
      CHECK(ss_field_space_destroy(&space) == SS_OK);
      CHECK(ss_mesh_destroy(&mesh) == SS_OK);
    }
  }
  CHECK(ss_mc_parameters_destroy(&material) == SS_OK);
  return 0;
}
