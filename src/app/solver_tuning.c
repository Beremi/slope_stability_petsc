/* Apply typed solver settings and derive continuation controls.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "solver_internal.h"

#include <float.h>
#include <math.h>
#include <string.h>

static SsStatus ss_solver_backend_option(const char *option, const char *value) {
  char current[64];
  PetscBool set = PETSC_FALSE;
  if (PetscOptionsGetString(NULL, NULL, option, current, sizeof(current), &set) !=
      PETSC_SUCCESS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (set == PETSC_TRUE) {
    return strcmp(current, value) == 0 ? SS_OK : SS_ERROR_INTEGRITY;
  }
  return PetscOptionsSetValue(NULL, option, value) == PETSC_SUCCESS ? SS_OK
                                                                    : SS_ERROR_INTERNAL;
}

SsAssemblyDeterminism ss_solver_assembly_determinism(const SsSolverRunPolicy *run) {
  return run->settings.validation_qualification_mode ? SS_ASSEMBLY_QUALIFICATION
                                                     : SS_ASSEMBLY_PERFORMANCE;
}

SsStatus ss_solver_backend_settings(const SsSolverRunPolicy *run) {
  const SsSolverSettings *settings = &run->settings;
  SsStatus status = SS_OK;
  if (settings->pmg_parmetis_seed >= 0) {
    char seed[32];
    (void)snprintf(seed, sizeof(seed), "%lld", (long long)settings->pmg_parmetis_seed);
    status = ss_solver_backend_option("-petscpartitioner_type", "parmetis");
    if (status == SS_OK) {
      status = ss_solver_backend_option("-petscpartitioner_parmetis_seed", seed);
    }
  }
  if (status == SS_OK && settings->linear_use_mdot_gemv == PETSC_TRUE) {
    status = ss_solver_backend_option("-vec_mdot_use_gemv", "true");
  }
  return status;
}

SsStatus ss_solver_superlu_dist_settings(void) {
  static const struct {
    const char *option;
    const char *value;
  } options[] = {
      {"-native_mech_p1_mat_superlu_dist_fact", "SamePattern_SameRowPerm"},
      {"-native_mech_p1_mat_superlu_dist_rowperm", "NOROWPERM"},
      {"-native_mech_p1_mat_superlu_dist_colperm", "METIS_AT_PLUS_A"},
      {"-native_mech_p1_mat_superlu_dist_equil", "true"},
      {"-native_mech_p1_mat_superlu_dist_replacetinypivot", "false"},
      {"-native_mech_p1_mat_superlu_dist_iterrefine", "false"},
      {"-native_mech_p1_mat_superlu_dist_parsymbfact", "false"},
      {"-native_mech_p1_mat_superlu_dist_3d", "false"},
      {"-native_mech_p1_mat_superlu_dist_gpuoffload", "false"},
      {"-native_mech_p1_mat_superlu_dist_printstat", "false"},
  };
  for (size_t index = 0; index < sizeof(options) / sizeof(options[0]); ++index) {
    const SsStatus status =
        ss_solver_backend_option(options[index].option, options[index].value);
    if (status != SS_OK) {
      return status;
    }
  }
  return SS_OK;
}

void ss_solver_indirect_sensitivity_options(const SsSolverSettings *settings,
                                            SsNonlinearControlSensitivityMethod *method,
                                            PetscReal *relative_step) {
  *method =
      (SsNonlinearControlSensitivityMethod)settings->nonlinear_indirect_sensitivity;
  *relative_step = settings->nonlinear_indirect_sensitivity_step > 0.0
                       ? settings->nonlinear_indirect_sensitivity_step
                       : cbrt(DBL_EPSILON);
}

int ss_solver_progress_guard_learns(SsNonlinearProgressGuardMode mode) {
  return mode == SS_NONLINEAR_PROGRESS_GUARD_ADAPTIVE ||
         mode == SS_NONLINEAR_PROGRESS_GUARD_OBSERVE;
}

SsStatus ss_solver_indirect_progress_guard_options(const SsSolverRunPolicy *run,
                                                   SsNonlinearProgressGuardMode *mode,
                                                   PetscInt *checkpoint_iteration,
                                                   PetscReal *maximum_convergence) {
  const SsSolverSettings *settings = &run->settings;
  *mode = (SsNonlinearProgressGuardMode)settings->nonlinear_indirect_progress_guard;
  *checkpoint_iteration = *mode == SS_NONLINEAR_PROGRESS_GUARD_OFF
                              ? 0
                              : settings->nonlinear_indirect_progress_guard_iteration;
  *maximum_convergence = settings->nonlinear_indirect_progress_guard_maximum;
  if (*checkpoint_iteration >= run->nonlinear_maximum_iterations ||
      ((*mode == SS_NONLINEAR_PROGRESS_GUARD_FIXED) != (*maximum_convergence > 0.0))) {
    return SS_ERROR_SCHEMA;
  }
  return SS_OK;
}

void ss_solver_indirect_work_step_options(const SsSolverRunPolicy *run,
                                          SsAnalysisWorkStepController *controller,
                                          PetscInt *growth_iteration_limit,
                                          PetscInt *efficiency_iteration_maximum,
                                          PetscReal *predictor_error_target,
                                          PetscReal *scale_minimum,
                                          PetscReal *scale_maximum) {
  const SsSolverSettings *settings = &run->settings;
  *controller =
      (SsAnalysisWorkStepController)settings->continuation_work_step_controller;
  *growth_iteration_limit = settings->continuation_work_growth_iteration_limit;
  *efficiency_iteration_maximum = 0;
  *predictor_error_target = 0.0;
  *scale_minimum = 0.0;
  *scale_maximum = 0.0;
  if (*controller == SS_ANALYSIS_WORK_STEP_CLASSIC) {
    return;
  }
  if (*growth_iteration_limit == 0) {
    /* ceil(5/6 * n), computed without overflowing PetscInt. */
    *growth_iteration_limit =
        run->nonlinear_maximum_iterations - run->nonlinear_maximum_iterations / 6;
  }
  if (*controller == SS_ANALYSIS_WORK_STEP_WORK_EFFICIENCY) {
    *efficiency_iteration_maximum = run->nonlinear_maximum_iterations;
  }
  *predictor_error_target = settings->continuation_work_predictor_error_target;
  *scale_minimum = settings->continuation_work_step_scale_minimum;
  *scale_maximum = settings->continuation_work_step_scale_maximum;
}

SsStatus ss_solver_mechanics_tuning(const SsSolverGraph *graph,
                                    SsSolverMechanics *mechanics,
                                    SsSolverTuning *tuning) {
  const SsSolverSettings *settings = &graph->policy.settings;
  tuning->nonlinear_allow_relaxed = settings->nonlinear_allow_relaxed;
  tuning->indirect_line_search_legacy_alg5 =
      settings->nonlinear_indirect_legacy_alg5_line_search;
  tuning->indirect_adaptive_linear_tolerance =
      settings->nonlinear_indirect_adaptive_linear_tolerance;
  tuning->nonlinear_maximum_line_search_trials =
      settings->nonlinear_maximum_line_search_trials;
  tuning->indirect_adaptive_linear_iteration_threshold =
      settings->nonlinear_indirect_adaptive_linear_iteration_threshold;
  tuning->indirect_adaptive_linear_alpha_threshold =
      settings->nonlinear_indirect_adaptive_linear_alpha_threshold;
  tuning->indirect_adaptive_linear_tightening_factor =
      settings->nonlinear_indirect_adaptive_linear_tightening_factor;
  tuning->indirect_adaptive_linear_minimum_relative_tolerance =
      settings->nonlinear_indirect_adaptive_linear_minimum_relative_tolerance;
  ss_solver_indirect_sensitivity_options(settings, &tuning->control_sensitivity_method,
                                         &tuning->control_sensitivity_relative_step);
  if (ss_solver_indirect_progress_guard_options(
          &graph->policy, &tuning->indirect_progress_guard_mode,
          &tuning->indirect_progress_guard_iteration,
          &tuning->indirect_progress_guard_maximum_convergence) != SS_OK ||
      (tuning->indirect_adaptive_linear_tolerance == PETSC_TRUE &&
       tuning->indirect_adaptive_linear_minimum_relative_tolerance >
           graph->policy.linear_rtol) ||
      (graph->model.analysis_method != SS_ANALYSIS_SSR_INDIRECT &&
       (tuning->indirect_progress_guard_mode != SS_NONLINEAR_PROGRESS_GUARD_OFF ||
        tuning->indirect_adaptive_linear_tolerance == PETSC_TRUE))) {
    return SS_ERROR_SCHEMA;
  }
  mechanics->matrix_free = settings->pmg_mechanics_matrix_free == PETSC_TRUE &&
                           graph->model.mechanics_order > 1U;
  mechanics->matrix_free_galerkin = settings->pmg_matrix_free_galerkin == PETSC_TRUE;
  mechanics->operator_spec.determinism = ss_solver_assembly_determinism(&graph->policy);
  mechanics->operator_spec.matrix_free_block_nodes =
      (size_t)settings->pmg_matrix_free_block_nodes;
  mechanics->operator_spec.matrix_free_batch_cells =
      (size_t)settings->pmg_matrix_free_batch_cells;
  mechanics->operator_spec.openmp_work_threshold =
      SS_SOLVER_DEFAULT_OPENMP_WORK_THRESHOLD;
  return ss_solver_cap_bytes(settings->resources_operator_workspace_mib,
                             &mechanics->operator_spec.memory_cap_bytes);
}
