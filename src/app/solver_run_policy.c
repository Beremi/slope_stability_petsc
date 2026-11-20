/* Read resolved solver policies, controls and result options.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/core/util.h"
#include "solver_input_internal.h"

#include "yyjson.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int ss_solver_normalized_absolute_path(const char *value,
                                              size_t encoded_length) {
  const char *segment;
  if (value == NULL || strlen(value) != encoded_length || value[0] != '/' ||
      value[1] == '\0') {
    return 0;
  }
  segment = value + 1;
  while (*segment != '\0') {
    const char *separator = strchr(segment, '/');
    const size_t length =
        separator == NULL ? strlen(segment) : (size_t)(separator - segment);
    if (length == 0U || (length == 1U && segment[0] == '.') ||
        (length == 2U && segment[0] == '.' && segment[1] == '.')) {
      return 0;
    }
    if (separator == NULL) {
      return 1;
    }
    segment = separator + 1;
  }
  return 0;
}

static int ss_solver_stopping_criterion(const char *value,
                                        SsNonlinearStoppingCriterion *criterion) {
  if (value == NULL || criterion == NULL) {
    return 0;
  }
  if (strcmp(value, "relative_residual") == 0) {
    *criterion = SS_NONLINEAR_STOP_RELATIVE_RESIDUAL;
    return 1;
  }
  if (strcmp(value, "relative_correction") == 0) {
    *criterion = SS_NONLINEAR_STOP_RELATIVE_CORRECTION;
    return 1;
  }
  if (strcmp(value, "absolute_delta_lambda") == 0) {
    *criterion = SS_NONLINEAR_STOP_ABSOLUTE_DELTA_LAMBDA;
    return 1;
  }
  return 0;
}

static SsStatus ss_solver_parse_pmg_active_ranks(yyjson_val *pmg,
                                                 SsSolverRunPolicy *policy) {
  yyjson_val *levels = pmg == NULL ? NULL : yyjson_obj_get(pmg, "levels");
  yyjson_val *active_ranks = pmg == NULL ? NULL : yyjson_obj_get(pmg, "active_ranks");
  size_t level_count;
  size_t level_index;
  if (!yyjson_is_arr(levels) || !yyjson_is_obj(active_ranks) || policy == NULL) {
    return SS_ERROR_SCHEMA;
  }
  level_count = yyjson_arr_size(levels);
  if (level_count == 0U || level_count > SS_PMG_MAX_LEVELS ||
      yyjson_obj_size(active_ranks) != level_count) {
    return SS_ERROR_SCHEMA;
  }
  policy->pmg_level_count = level_count;
  for (level_index = 0U; level_index < level_count; ++level_index) {
    yyjson_val *level_value = yyjson_arr_get(levels, level_index);
    yyjson_val *ranks;
    uint64_t order;
    size_t rank_count;
    size_t rank_index;
    char key[2];
    if (!yyjson_is_uint(level_value)) {
      return SS_ERROR_SCHEMA;
    }
    order = yyjson_get_uint(level_value);
    if ((order != 1U && order != 2U && order != 4U && order != 8U) ||
        (level_index != 0U &&
         order <= (uint64_t)policy->pmg_level_orders[level_index - 1U]) ||
        snprintf(key, sizeof(key), "%llu", (unsigned long long)order) != 1) {
      return SS_ERROR_SCHEMA;
    }
    ranks = yyjson_obj_get(active_ranks, key);
    if (!yyjson_is_arr(ranks)) {
      return SS_ERROR_SCHEMA;
    }
    rank_count = yyjson_arr_size(ranks);
    if (rank_count == 0U || rank_count > (size_t)PETSC_MPI_INT_MAX) {
      return SS_ERROR_SCHEMA;
    }
    policy->pmg_active_ranks[level_index] =
        (PetscMPIInt *)calloc(rank_count, sizeof(PetscMPIInt));
    if (policy->pmg_active_ranks[level_index] == NULL) {
      return SS_ERROR_OUT_OF_MEMORY;
    }
    policy->pmg_level_orders[level_index] = (uint32_t)order;
    policy->pmg_active_rank_counts[level_index] = rank_count;
    for (rank_index = 0U; rank_index < rank_count; ++rank_index) {
      yyjson_val *rank_value = yyjson_arr_get(ranks, rank_index);
      uint64_t rank;
      if (!yyjson_is_uint(rank_value)) {
        return SS_ERROR_SCHEMA;
      }
      rank = yyjson_get_uint(rank_value);
      if (rank > (uint64_t)PETSC_MPI_INT_MAX ||
          (rank_index != 0U &&
           rank <= (uint64_t)policy->pmg_active_ranks[level_index][rank_index - 1U])) {
        return SS_ERROR_SCHEMA;
      }
      policy->pmg_active_ranks[level_index][rank_index] = (PetscMPIInt)rank;
    }
  }
  return SS_OK;
}

SsStatus ss_solver_parse_run(const char *path, SsSolverRunPolicy *policy) {
  unsigned char *data = NULL;
  size_t size = 0U;
  yyjson_doc *document = NULL;
  yyjson_val *root;
  yyjson_val *continuation;
  yyjson_val *nonlinear;
  yyjson_val *linear;
  yyjson_val *pmg;
  yyjson_val *deflation;
  yyjson_val *outputs;
  yyjson_val *output_history;
  yyjson_val *result_policy;
  yyjson_val *checkpoint;
  yyjson_val *checkpoint_mode_value;
  yyjson_val *checkpoint_path_value;
  yyjson_val *validation;
  const char *pmg_smoother;
  const char *pmg_coarse_solver;
  const char *linear_ksp_type;
  const char *linear_stopping_condition;
  const char *deflation_projector;
  const char *deflation_candidate_source;
  const char *deflation_recycle;
  const char *deflation_dense_factor;
  const char *deflation_compression;
  const char *initial_nonlinear_stopping_criterion;
  const char *nonlinear_stopping_criterion;
  const char *output_frequency;
  const char *checkpoint_path;
  const char *numerical_policy_id;
  const char *run_id;
  SsStatus status = ss_solver_read_file(path, &data, &size);
  if (status != SS_OK) {
    return status;
  }
  document = yyjson_read((char *)(void *)data, size, YYJSON_READ_NOFLAG);
  if (document == NULL) {
    free(data);
    return SS_ERROR_JSON;
  }
  root = yyjson_doc_get_root(document);
  continuation = ss_solver_object(root, "continuation");
  nonlinear = ss_solver_object(root, "nonlinear");
  linear = ss_solver_object(root, "linear");
  pmg = ss_solver_object(root, "pmg");
  deflation = ss_solver_object(root, "deflation");
  outputs = ss_solver_object(root, "outputs");
  output_history = outputs == NULL ? NULL : yyjson_obj_get(outputs, "history");
  result_policy = ss_solver_object(root, "result_policy");
  checkpoint =
      result_policy == NULL ? NULL : ss_solver_object(result_policy, "checkpoint");
  validation = ss_solver_object(root, "validation");
  if (continuation == NULL || nonlinear == NULL || linear == NULL || pmg == NULL ||
      deflation == NULL || outputs == NULL || result_policy == NULL ||
      validation == NULL) {
    status = SS_ERROR_SCHEMA;
    goto cleanup;
  }
  policy->initial_factor = ss_solver_number(continuation, "initial_factor", NAN);
  policy->initial_step = ss_solver_number(continuation, "initial_step", NAN);
  policy->minimum_step = ss_solver_number(continuation, "minimum_step", NAN);
  policy->maximum_step = ss_solver_number(continuation, "maximum_step", NAN);
  policy->maximum_work = ss_solver_number(continuation, "maximum_work", NAN);
  policy->initial_work_scale =
      ss_solver_number(continuation, "initial_work_scale", NAN);
  policy->scaled_slope_tolerance =
      ss_solver_number(continuation, "scaled_slope_tolerance", NAN);
  policy->factor_increment_stop =
      ss_solver_number(continuation, "factor_increment_stop", NAN);
  initial_nonlinear_stopping_criterion =
      ss_solver_string(continuation, "initial_nonlinear_stopping_criterion");
  policy->initial_nonlinear_stopping_tolerance =
      ss_solver_number(continuation, "initial_nonlinear_stopping_tolerance", NAN);
  nonlinear_stopping_criterion = ss_solver_string(nonlinear, "stopping_criterion");
  policy->nonlinear_rtol = ss_solver_number(nonlinear, "relative_tolerance", NAN);
  policy->nonlinear_atol = ss_solver_number(nonlinear, "absolute_tolerance", NAN);
  policy->nonlinear_stopping_tolerance =
      ss_solver_number(nonlinear, "stopping_tolerance", NAN);
  policy->nonlinear_regularization_minimum =
      ss_solver_number(nonlinear, "regularization_minimum", NAN);
  policy->nonlinear_maximum_iterations =
      (PetscInt)ss_solver_uint(nonlinear, "maximum_iterations", UINT64_C(0));
  policy->hydraulic_nonlinear_rtol = ss_solver_number(
      nonlinear, "hydraulic_relative_tolerance", policy->nonlinear_rtol);
  policy->hydraulic_nonlinear_atol = ss_solver_number(
      nonlinear, "hydraulic_absolute_tolerance", policy->nonlinear_atol);
  policy->hydraulic_nonlinear_maximum_iterations =
      (PetscInt)ss_solver_uint(nonlinear, "hydraulic_maximum_iterations",
                               (uint64_t)policy->nonlinear_maximum_iterations);
  linear_ksp_type = ss_solver_string(linear, "ksp_type");
  if (linear_ksp_type == NULL || strcmp(linear_ksp_type, "fgmres") == 0) {
    policy->linear_method = SS_LINEAR_METHOD_FGMRES;
  } else if (strcmp(linear_ksp_type, "hpddm_gcrodr") == 0) {
    policy->linear_method = SS_LINEAR_METHOD_HPDDM_GCRODR;
  } else if (strcmp(linear_ksp_type, "fcg") == 0) {
    policy->linear_method = SS_LINEAR_METHOD_FCG;
  }
  policy->linear_rtol = ss_solver_number(linear, "relative_tolerance", NAN);
  policy->linear_atol = ss_solver_number(linear, "absolute_tolerance", NAN);
  linear_stopping_condition = ss_solver_string(linear, "stopping_condition");
  if (linear_stopping_condition == NULL ||
      strcmp(linear_stopping_condition, "original_rhs_true_residual") == 0) {
    policy->linear_stopping_condition = SS_LINEAR_STOP_ORIGINAL_RHS_TRUE_RESIDUAL;
  } else if (strcmp(linear_stopping_condition, "projected_rhs_petsc") == 0) {
    policy->linear_stopping_condition = SS_LINEAR_STOP_PROJECTED_RHS_PETSC;
  } else if (strcmp(linear_stopping_condition,
                    "preconditioned_residual_original_rhs") == 0) {
    policy->linear_stopping_condition =
        SS_LINEAR_STOP_PRECONDITIONED_RESIDUAL_ORIGINAL_RHS;
  } else if (strcmp(linear_stopping_condition, "natural_residual_initial") == 0) {
    policy->linear_stopping_condition = SS_LINEAR_STOP_NATURAL_RESIDUAL_INITIAL;
  }
  policy->linear_maximum_iterations =
      (PetscInt)ss_solver_uint(linear, "maximum_iterations", UINT64_C(0));
  policy->hydraulic_linear_rtol =
      ss_solver_number(linear, "hydraulic_relative_tolerance", policy->linear_rtol);
  policy->hydraulic_linear_atol =
      ss_solver_number(linear, "hydraulic_absolute_tolerance", policy->linear_atol);
  policy->hydraulic_linear_maximum_iterations =
      (PetscInt)ss_solver_uint(linear, "hydraulic_maximum_iterations",
                               (uint64_t)policy->linear_maximum_iterations);
  policy->linear_restart = (PetscInt)ss_solver_uint(linear, "restart", UINT64_C(0));
  policy->linear_recycle_dimension =
      (PetscInt)ss_solver_uint(linear, "recycle_dimension", UINT64_C(0));
  policy->pmg_enabled = ss_solver_boolean(pmg, "enabled", 0);
  pmg_smoother = ss_solver_string(pmg, "smoother");
  if (pmg_smoother != NULL && strcmp(pmg_smoother, "chebyshev") == 0) {
    policy->pmg_smoother_method = SS_PMG_LEVEL_CHEBYSHEV_JACOBI;
  } else if (pmg_smoother != NULL && strcmp(pmg_smoother, "chebyshev_point") == 0) {
    policy->pmg_smoother_method = SS_PMG_LEVEL_CHEBYSHEV_POINT_JACOBI;
  } else if (pmg_smoother != NULL && strcmp(pmg_smoother, "fgmres") == 0) {
    policy->pmg_smoother_method = SS_PMG_LEVEL_FGMRES_JACOBI;
  } else if (pmg_smoother != NULL && strcmp(pmg_smoother, "fgmres_block") == 0) {
    policy->pmg_smoother_method = SS_PMG_LEVEL_FGMRES_VPBJACOBI;
  } else if (pmg_smoother != NULL && strcmp(pmg_smoother, "richardson") == 0) {
    policy->pmg_smoother_method = SS_PMG_LEVEL_RICHARDSON_JACOBI;
  } else if (pmg_smoother != NULL && strcmp(pmg_smoother, "richardson_block") == 0) {
    policy->pmg_smoother_method = SS_PMG_LEVEL_RICHARDSON_VPBJACOBI;
  } else if (policy->pmg_enabled != 0) {
    status = SS_ERROR_SCHEMA;
    goto cleanup;
  } else {
    policy->pmg_smoother_method = SS_PMG_LEVEL_RICHARDSON_NONE;
  }
  policy->pmg_pre_smoothing_steps =
      (PetscInt)ss_solver_uint(pmg, "pre_smoothing_steps", UINT64_C(2));
  policy->pmg_post_smoothing_steps =
      (PetscInt)ss_solver_uint(pmg, "post_smoothing_steps", UINT64_C(2));
  policy->pmg_base_iterations =
      (PetscInt)ss_solver_uint(pmg, "base_iterations", UINT64_C(2));
  /* Older resolved inputs retain their explicit shared base policy. New
   * hydraulic runs resolve a separate cap and bind it into the run identity. */
  policy->pmg_hydraulic_base_iterations = policy->pmg_base_iterations;
  {
    yyjson_val *hydraulic_base = yyjson_obj_get(pmg, "hydraulic_base_iterations");
    if (hydraulic_base != NULL) {
      policy->pmg_hydraulic_base_iterations =
          yyjson_is_uint(hydraulic_base) && yyjson_get_uint(hydraulic_base) > 0U &&
                  yyjson_get_uint(hydraulic_base) <= (uint64_t)PETSC_MAX_INT
              ? (PetscInt)yyjson_get_uint(hydraulic_base)
              : 0;
    }
  }
  pmg_coarse_solver = ss_solver_string(pmg, "coarse_solver");
  if (pmg_coarse_solver != NULL && strcmp(pmg_coarse_solver, "gamg") == 0) {
    policy->pmg_coarse_method = SS_PMG_LEVEL_FGMRES_GAMG;
  } else if (pmg_coarse_solver != NULL &&
             strcmp(pmg_coarse_solver, "preonly_lu") == 0) {
    policy->pmg_coarse_method = SS_PMG_LEVEL_PREONLY_LU;
  } else if (policy->pmg_enabled != 0) {
    status = SS_ERROR_SCHEMA;
    goto cleanup;
  }
  policy->pmg_coarse_direct_maximum_dofs =
      ss_solver_uint(pmg, "coarse_direct_maximum_dofs", UINT64_C(0));
  policy->pmg_coarse_redundant_group_size =
      (size_t)ss_solver_uint(pmg, "coarse_redundant_group_size", UINT64_C(0));
  status = ss_solver_parse_pmg_active_ranks(pmg, policy);
  if (status != SS_OK) {
    goto cleanup;
  }
  policy->deflation_enabled = ss_solver_boolean(deflation, "enabled", 0);
  policy->deflation_maximum_dimension =
      (size_t)ss_solver_uint(deflation, "maximum_dimension", UINT64_C(0));
  deflation_projector = ss_solver_string(deflation, "projector");
  deflation_candidate_source = ss_solver_string(deflation, "candidate_source");
  deflation_recycle = ss_solver_string(deflation, "recycle");
  deflation_dense_factor = ss_solver_string(deflation, "dense_factor");
  deflation_compression = ss_solver_string(deflation, "compression");
  if (deflation_projector == NULL) {
    deflation_projector = "general_left_right";
  }
  if (strcmp(deflation_projector, "general_left_right") == 0) {
    policy->deflation_projector = SS_DEFLATION_PROJECTOR_GENERAL_LEFT_RIGHT;
  } else if (strcmp(deflation_projector, "legacy_a_orthogonal") == 0) {
    policy->deflation_projector = SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL;
  }
  if (deflation_candidate_source == NULL) {
    deflation_candidate_source = "accepted_newton_and_control_directions";
  }
  if (deflation_recycle == NULL) {
    deflation_recycle = "accepted_steps";
  }
  if (strcmp(deflation_candidate_source, "accepted_newton_and_control_directions") ==
          0 &&
      strcmp(deflation_recycle, "accepted_steps") == 0) {
    policy->deflation_candidate_policy = SS_DEFLATION_CANDIDATE_ACCEPTED_DIRECTIONS;
  } else if (strcmp(deflation_candidate_source, "accepted_state_vectors") == 0 &&
             strcmp(deflation_recycle, "attempt_snapshot_restore") == 0) {
    policy->deflation_candidate_policy = SS_DEFLATION_CANDIDATE_ACCEPTED_STATE_VECTORS;
  } else if (strcmp(deflation_candidate_source, "accepted_states_and_full_krylov") ==
                 0 &&
             strcmp(deflation_recycle, "attempt_snapshot_restore") == 0) {
    policy->deflation_candidate_policy =
        SS_DEFLATION_CANDIDATE_ACCEPTED_STATES_AND_FULL_KRYLOV;
  } else if (strcmp(deflation_candidate_source,
                    "accepted_states_and_system_solutions") == 0 &&
             strcmp(deflation_recycle, "attempt_snapshot_restore") == 0) {
    policy->deflation_candidate_policy =
        SS_DEFLATION_CANDIDATE_ACCEPTED_STATES_AND_SYSTEM_SOLUTIONS;
  }
  if (deflation_dense_factor == NULL) {
    deflation_dense_factor = "partial_pivot_lu";
  }
  if (deflation_compression == NULL) {
    deflation_compression = "none";
  }
  if (strcmp(deflation_compression, "none") == 0) {
    policy->deflation_compression = SS_DEFLATION_COMPRESSION_NONE;
  } else if (strcmp(deflation_compression, "snapshot_pod") == 0) {
    policy->deflation_compression = SS_DEFLATION_COMPRESSION_SNAPSHOT_POD;
  } else if (strcmp(deflation_compression, "operator_svd") == 0) {
    policy->deflation_compression = SS_DEFLATION_COMPRESSION_OPERATOR_SVD;
  } else if (strcmp(deflation_compression, "snapshot_pod_total") == 0) {
    policy->deflation_compression = SS_DEFLATION_COMPRESSION_SNAPSHOT_POD_TOTAL;
  } else if (strcmp(deflation_compression, "system_solution_window") == 0) {
    policy->deflation_compression = SS_DEFLATION_COMPRESSION_SYSTEM_SOLUTION_WINDOW;
  }
  policy->deflation_compression_dimension =
      (size_t)ss_solver_uint(deflation, "compression_dimension", UINT64_C(0));
  policy->deflation_compression_history_decay =
      ss_solver_number(deflation, "compression_history_decay", 1.0);
  policy->deflation_admission_tolerance =
      ss_solver_number(deflation, "admission_tolerance", 1.0e-3);
  policy->deflation_pivot_tolerance =
      ss_solver_number(deflation, "pivot_tolerance", 1.0e-13);
  policy->deflation_condition_limit =
      ss_solver_number(deflation, "condition_limit", 1.0e12);
  policy->deflation_invariant_tolerance =
      ss_solver_number(deflation, "invariant_tolerance", 1.0e-9);
  policy->deflation_reorthogonalization_passes =
      (size_t)ss_solver_uint(deflation, "reorthogonalization_passes", UINT64_C(2));
  status = ss_solver_parse_output_quantities(yyjson_obj_get(outputs, "quantities"),
                                             &policy->requested_output_mask);
  if (status != SS_OK) {
    goto cleanup;
  }
  output_frequency = ss_solver_string(outputs, "frequency");
  if (output_frequency != NULL && strcmp(output_frequency, "terminal") == 0) {
    policy->output_frequency = SS_SOLVER_OUTPUT_TERMINAL;
  } else if (output_frequency != NULL &&
             strcmp(output_frequency, "accepted_state") == 0) {
    policy->output_frequency = SS_SOLVER_OUTPUT_ACCEPTED_STATE;
  } else {
    status = SS_ERROR_SCHEMA;
    goto cleanup;
  }
  if (!yyjson_is_bool(output_history)) {
    status = SS_ERROR_SCHEMA;
    goto cleanup;
  }
  policy->output_history = yyjson_get_bool(output_history);
  policy->checkpoint_mode = SS_SOLVER_CHECKPOINT_NONE;
  run_id = ss_solver_string(root, "id");
  numerical_policy_id = ss_solver_string(result_policy, "numerical_policy_id");
  if (numerical_policy_id == NULL) {
    numerical_policy_id = run_id;
  }
  if (!ss_valid_content_id(numerical_policy_id) ||
      !ss_copy_text(policy->numerical_policy_id, sizeof(policy->numerical_policy_id),
                    numerical_policy_id)) {
    status = SS_ERROR_SCHEMA;
    goto cleanup;
  }
  if (checkpoint != NULL) {
    checkpoint_mode_value = yyjson_obj_get(checkpoint, "mode");
    checkpoint_path_value = yyjson_obj_get(checkpoint, "path");
    checkpoint_path = ss_solver_string(checkpoint, "path");
    if (ss_solver_json_string_equal(checkpoint_mode_value, "write_accepted") &&
        checkpoint_path == NULL && yyjson_obj_size(checkpoint) == 1U) {
      policy->checkpoint_mode = SS_SOLVER_CHECKPOINT_WRITE_ACCEPTED;
    } else if (ss_solver_json_string_equal(checkpoint_mode_value, "restore_accepted") &&
               yyjson_is_str(checkpoint_path_value) &&
               ss_solver_normalized_absolute_path(
                   checkpoint_path, yyjson_get_len(checkpoint_path_value)) != 0 &&
               yyjson_obj_size(checkpoint) == 2U &&
               ss_copy_text(policy->checkpoint_path, sizeof(policy->checkpoint_path),
                            checkpoint_path)) {
      policy->checkpoint_mode = SS_SOLVER_CHECKPOINT_RESTORE_ACCEPTED;
    } else {
      status = SS_ERROR_SCHEMA;
      goto cleanup;
    }
  } else if (yyjson_obj_get(result_policy, "checkpoint") != NULL) {
    status = SS_ERROR_SCHEMA;
    goto cleanup;
  }
  policy->conservation_physical_scale =
      ss_solver_number(validation, "conservation_physical_scale", NAN);
  policy->conservation_normalized_tolerance =
      ss_solver_number(validation, "conservation_normalized_tolerance", NAN);
  if (!ss_solver_stopping_criterion(initial_nonlinear_stopping_criterion,
                                    &policy->initial_nonlinear_stopping_criterion) ||
      !ss_solver_stopping_criterion(nonlinear_stopping_criterion,
                                    &policy->nonlinear_stopping_criterion) ||
      policy->initial_nonlinear_stopping_criterion ==
          SS_NONLINEAR_STOP_ABSOLUTE_DELTA_LAMBDA ||
      !isfinite(policy->initial_factor) || policy->initial_factor <= 0.0 ||
      !isfinite(policy->initial_step) || policy->initial_step <= 0.0 ||
      !isfinite(policy->minimum_step) || policy->minimum_step <= 0.0 ||
      !isfinite(policy->maximum_step) || policy->maximum_step < policy->minimum_step ||
      !isfinite(policy->maximum_work) || policy->maximum_work <= 0.0 ||
      !isfinite(policy->initial_work_scale) || policy->initial_work_scale <= 0.0 ||
      !isfinite(policy->scaled_slope_tolerance) ||
      policy->scaled_slope_tolerance < 0.0 ||
      !isfinite(policy->factor_increment_stop) || policy->factor_increment_stop < 0.0 ||
      policy->maximum_accepted_states <= 0 ||
      !isfinite(policy->initial_nonlinear_stopping_tolerance) ||
      policy->initial_nonlinear_stopping_tolerance <= 0.0 ||
      !isfinite(policy->nonlinear_rtol) || policy->nonlinear_rtol <= 0.0 ||
      !isfinite(policy->nonlinear_atol) || policy->nonlinear_atol < 0.0 ||
      !isfinite(policy->nonlinear_stopping_tolerance) ||
      policy->nonlinear_stopping_tolerance <= 0.0 ||
      !isfinite(policy->nonlinear_regularization_minimum) ||
      policy->nonlinear_regularization_minimum <= 0.0 ||
      policy->nonlinear_regularization_minimum > 1.0 ||
      policy->nonlinear_maximum_iterations <= 0 ||
      !isfinite(policy->hydraulic_nonlinear_rtol) ||
      policy->hydraulic_nonlinear_rtol <= 0.0 ||
      !isfinite(policy->hydraulic_nonlinear_atol) ||
      policy->hydraulic_nonlinear_atol < 0.0 ||
      policy->hydraulic_nonlinear_maximum_iterations <= 0 ||
      !isfinite(policy->linear_rtol) || policy->linear_rtol <= 0.0 ||
      !isfinite(policy->linear_atol) || policy->linear_atol < 0.0 ||
      (linear_ksp_type != NULL && strcmp(linear_ksp_type, "fgmres") != 0 &&
       strcmp(linear_ksp_type, "hpddm_gcrodr") != 0 &&
       strcmp(linear_ksp_type, "fcg") != 0) ||
      (linear_stopping_condition != NULL &&
       strcmp(linear_stopping_condition, "original_rhs_true_residual") != 0 &&
       strcmp(linear_stopping_condition, "projected_rhs_petsc") != 0 &&
       strcmp(linear_stopping_condition, "preconditioned_residual_original_rhs") != 0 &&
       strcmp(linear_stopping_condition, "natural_residual_initial") != 0) ||
      ((policy->linear_stopping_condition ==
            SS_LINEAR_STOP_PRECONDITIONED_RESIDUAL_ORIGINAL_RHS ||
        policy->linear_stopping_condition == SS_LINEAR_STOP_NATURAL_RESIDUAL_INITIAL) &&
       policy->linear_method != SS_LINEAR_METHOD_FCG) ||
      policy->linear_maximum_iterations <= 0 || policy->linear_restart <= 0 ||
      (policy->linear_method == SS_LINEAR_METHOD_HPDDM_GCRODR &&
       (policy->linear_recycle_dimension <= 0 ||
        policy->linear_recycle_dimension >= policy->linear_restart)) ||
      (policy->linear_method != SS_LINEAR_METHOD_HPDDM_GCRODR &&
       policy->linear_recycle_dimension != 0) ||
      !isfinite(policy->hydraulic_linear_rtol) ||
      policy->hydraulic_linear_rtol <= 0.0 ||
      !isfinite(policy->hydraulic_linear_atol) || policy->hydraulic_linear_atol < 0.0 ||
      policy->hydraulic_linear_maximum_iterations <= 0 ||
      (yyjson_obj_get(pmg, "hydraulic_base_iterations") != NULL &&
       policy->pmg_hydraulic_base_iterations <= 0) ||
      (policy->pmg_enabled != 0 &&
       (policy->pmg_base_iterations <= 0 ||
        policy->pmg_hydraulic_base_iterations <= 0 ||
        policy->pmg_pre_smoothing_steps <= 0 || policy->pmg_post_smoothing_steps <= 0 ||
        (policy->pmg_coarse_method != SS_PMG_LEVEL_FGMRES_GAMG &&
         policy->pmg_coarse_method != SS_PMG_LEVEL_PREONLY_LU) ||
        (policy->pmg_coarse_redundant_group_size != 0U &&
         policy->pmg_coarse_method != SS_PMG_LEVEL_PREONLY_LU))) ||
      deflation_projector == NULL ||
      (strcmp(deflation_projector, "general_left_right") != 0 &&
       strcmp(deflation_projector, "legacy_a_orthogonal") != 0) ||
      deflation_candidate_source == NULL || deflation_recycle == NULL ||
      !((strcmp(deflation_candidate_source, "accepted_newton_and_control_directions") ==
             0 &&
         strcmp(deflation_recycle, "accepted_steps") == 0) ||
        (strcmp(deflation_candidate_source, "accepted_state_vectors") == 0 &&
         strcmp(deflation_recycle, "attempt_snapshot_restore") == 0) ||
        (strcmp(deflation_candidate_source, "accepted_states_and_full_krylov") == 0 &&
         strcmp(deflation_recycle, "attempt_snapshot_restore") == 0) ||
        (strcmp(deflation_candidate_source, "accepted_states_and_system_solutions") ==
             0 &&
         strcmp(deflation_recycle, "attempt_snapshot_restore") == 0)) ||
      deflation_dense_factor == NULL ||
      strcmp(deflation_dense_factor, "partial_pivot_lu") != 0 ||
      deflation_compression == NULL ||
      (strcmp(deflation_compression, "none") != 0 &&
       strcmp(deflation_compression, "snapshot_pod") != 0 &&
       strcmp(deflation_compression, "operator_svd") != 0 &&
       strcmp(deflation_compression, "snapshot_pod_total") != 0 &&
       strcmp(deflation_compression, "system_solution_window") != 0) ||
      ((strcmp(deflation_compression, "none") == 0) !=
       (policy->deflation_compression_dimension == 0U)) ||
      !isfinite(policy->deflation_compression_history_decay) ||
      policy->deflation_compression_history_decay < 0.0 ||
      policy->deflation_compression_history_decay > 1.0 ||
      (policy->deflation_compression != SS_DEFLATION_COMPRESSION_NONE &&
       (strcmp(deflation_projector, "legacy_a_orthogonal") != 0 ||
        !(((policy->deflation_compression == SS_DEFLATION_COMPRESSION_SNAPSHOT_POD ||
            policy->deflation_compression == SS_DEFLATION_COMPRESSION_OPERATOR_SVD ||
            policy->deflation_compression ==
                SS_DEFLATION_COMPRESSION_SNAPSHOT_POD_TOTAL) &&
           strcmp(deflation_candidate_source, "accepted_states_and_full_krylov") ==
               0) ||
          (policy->deflation_compression ==
               SS_DEFLATION_COMPRESSION_SYSTEM_SOLUTION_WINDOW &&
           strcmp(deflation_candidate_source, "accepted_states_and_system_solutions") ==
               0)) ||
        policy->deflation_compression_dimension >
            policy->deflation_maximum_dimension)) ||
      (policy->deflation_enabled != 0 && policy->deflation_maximum_dimension == 0U) ||
      !isfinite(policy->deflation_admission_tolerance) ||
      policy->deflation_admission_tolerance <= 0.0 ||
      !isfinite(policy->deflation_pivot_tolerance) ||
      policy->deflation_pivot_tolerance <= 0.0 ||
      !isfinite(policy->deflation_condition_limit) ||
      policy->deflation_condition_limit <= 1.0 ||
      !isfinite(policy->deflation_invariant_tolerance) ||
      policy->deflation_invariant_tolerance <= 0.0 ||
      policy->deflation_reorthogonalization_passes == 0U ||
      policy->deflation_reorthogonalization_passes > 4U ||
      (policy->output_frequency != SS_SOLVER_OUTPUT_TERMINAL &&
       policy->output_frequency != SS_SOLVER_OUTPUT_ACCEPTED_STATE) ||
      !isfinite(policy->conservation_physical_scale) ||
      policy->conservation_physical_scale <= 0.0 ||
      !isfinite(policy->conservation_normalized_tolerance) ||
      policy->conservation_normalized_tolerance < 0.0) {
    status = SS_ERROR_SCHEMA;
  }

cleanup:
  yyjson_doc_free(document);
  free(data);
  return status;
}
