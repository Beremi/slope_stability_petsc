/* Linear solver lifecycle, setup and policy admission.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "linear_internal.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/core/core.h"
#include "slope_internal/core/util.h"
#include "slope_internal/diagnostics/diagnostics.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

PetscLogEvent ss_linear_setup_event = -1;
PetscLogEvent ss_linear_solve_event = -1;
PetscLogEvent ss_linear_deflation_refresh_event = -1;
PetscLogEvent ss_linear_pmg_setup_event = -1;
PetscLogEvent ss_linear_initial_guess_event = -1;
PetscLogEvent ss_linear_krylov_recycle_event = -1;
PetscLogEvent ss_linear_full_krylov_event = -1;

static const char *ss_linear_method_name(SsLinearMethod method) {
  switch (method) {
  case SS_LINEAR_METHOD_FGMRES:
    return KSPFGMRES;
  case SS_LINEAR_METHOD_MINRES:
    return KSPMINRES;
  case SS_LINEAR_METHOD_CG:
    return KSPCG;
  case SS_LINEAR_METHOD_PREONLY:
    return KSPPREONLY;
  case SS_LINEAR_METHOD_HPDDM_GCRODR:
#if defined(PETSC_HAVE_HPDDM)
    return KSPHPDDM;
#else
    return NULL;
#endif
  case SS_LINEAR_METHOD_FCG:
    return KSPFCG;
  }
  return NULL;
}

static const char *ss_linear_pc_name(SsLinearPcKind pc) {
  switch (pc) {
  case SS_LINEAR_PC_NONE:
    return PCNONE;
  case SS_LINEAR_PC_JACOBI:
    return PCJACOBI;
  case SS_LINEAR_PC_LU:
    return PCLU;
  case SS_LINEAR_PC_PMG:
    return PCSHELL;
  }
  return NULL;
}

static PCSide ss_linear_petsc_side(SsLinearPcSide side) {
  return side == SS_LINEAR_SIDE_RIGHT ? PC_RIGHT : PC_LEFT;
}

static SsLinearPcSide ss_linear_project_side(PCSide side) {
  return side == PC_RIGHT ? SS_LINEAR_SIDE_RIGHT : SS_LINEAR_SIDE_LEFT;
}

KSPNormType ss_linear_norm_type(const SsLinearPolicy *policy) {
  if (policy->method == SS_LINEAR_METHOD_PREONLY) {
    return KSP_NORM_NONE;
  }
  if (policy->stopping_condition == SS_LINEAR_STOP_NATURAL_RESIDUAL_INITIAL) {
    return KSP_NORM_NATURAL;
  }
  /* PETSc FCG supports an unpreconditioned norm with a left, possibly varying
   * preconditioner and obtains it directly from FCG's recursive residual.
   * Select the norm from the stopping contract before falling back to the PC
   * side: original-RHS true-residual stopping must not silently become a
   * preconditioned-residual test merely because FCG is left-sided. */
  if (policy->stopping_condition == SS_LINEAR_STOP_ORIGINAL_RHS_TRUE_RESIDUAL) {
    return KSP_NORM_UNPRECONDITIONED;
  }
  return policy->side == SS_LINEAR_SIDE_RIGHT ? KSP_NORM_UNPRECONDITIONED
                                              : KSP_NORM_PRECONDITIONED;
}

static int ss_linear_properties_equal(const SsLinearOperatorProperties *left,
                                      const SsLinearOperatorProperties *right) {
  return left->finite_action == right->finite_action &&
         left->measured_symmetric == right->measured_symmetric &&
         left->positive_energy == right->positive_energy &&
         left->pc_symmetric == right->pc_symmetric &&
         left->pc_positive == right->pc_positive &&
         left->pc_variable == right->pc_variable &&
         left->nonlinear_hydraulic == right->nonlinear_hydraulic;
}

static int ss_linear_policy_equal(const SsLinearPolicy *left,
                                  const SsLinearPolicy *right) {
  const size_t left_identity_length =
      ss_bounded_length(left->identity, sizeof(left->identity));
  const size_t right_identity_length =
      ss_bounded_length(right->identity, sizeof(right->identity));
  const size_t left_prefix_length =
      ss_bounded_length(left->options_prefix, sizeof(left->options_prefix));
  const size_t right_prefix_length =
      ss_bounded_length(right->options_prefix, sizeof(right->options_prefix));
  return left_identity_length < sizeof(left->identity) &&
         right_identity_length == left_identity_length &&
         memcmp(left->identity, right->identity, left_identity_length + 1U) == 0 &&
         left_prefix_length < sizeof(left->options_prefix) &&
         right_prefix_length == left_prefix_length &&
         memcmp(left->options_prefix, right->options_prefix, left_prefix_length + 1U) ==
             0 &&
         left->operator_class == right->operator_class &&
         left->method == right->method && left->pc == right->pc &&
         left->side == right->side &&
         left->stopping_condition == right->stopping_condition &&
         left->initial_guess_policy == right->initial_guess_policy &&
         left->orthogonalization == right->orthogonalization &&
         left->rtol == right->rtol && left->atol == right->atol &&
         left->dtol == right->dtol &&
         left->maximum_iterations == right->maximum_iterations &&
         left->restart == right->restart &&
         left->recycle_dimension == right->recycle_dimension &&
         left->vector_memory_cap_bytes == right->vector_memory_cap_bytes;
}

static int ss_linear_chain_equal(const SsLinearPolicyChain *left,
                                 const SsLinearPolicyChain *right) {
  size_t index;
  if (left->count != right->count) {
    return 0;
  }
  for (index = 0U; index < left->count; ++index) {
    if (!ss_linear_policy_equal(&left->policies[index], &right->policies[index])) {
      return 0;
    }
  }
  return 1;
}

static int ss_linear_valid_prefix(const char *prefix) {
  const size_t length = ss_bounded_length(prefix, SS_LINEAR_PREFIX_CAPACITY);
  size_t index;
  if (length == 0U || length == SS_LINEAR_PREFIX_CAPACITY ||
      prefix[length - 1U] != '_') {
    return 0;
  }
  for (index = 0U; index < length; ++index) {
    const char character = prefix[index];
    if (!((character >= 'a' && character <= 'z') ||
          (character >= 'A' && character <= 'Z') ||
          (character >= '0' && character <= '9') || character == '_')) {
      return 0;
    }
  }
  return 1;
}

SsStatus ss_linear_policy_validate(const SsLinearPolicy *policy,
                                   const SsLinearOperatorProperties *properties) {
  const size_t identity_length =
      policy == NULL ? 0U
                     : ss_bounded_length(policy->identity, SS_LINEAR_IDENTITY_CAPACITY);
  if (policy == NULL || properties == NULL || identity_length == 0U ||
      identity_length == SS_LINEAR_IDENTITY_CAPACITY ||
      !ss_linear_valid_prefix(policy->options_prefix) ||
      ss_linear_method_name(policy->method) == NULL ||
      ss_linear_pc_name(policy->pc) == NULL || !ss_is_finite(policy->rtol) ||
      (policy->stopping_condition != SS_LINEAR_STOP_ORIGINAL_RHS_TRUE_RESIDUAL &&
       policy->stopping_condition != SS_LINEAR_STOP_PROJECTED_RHS_PETSC &&
       policy->stopping_condition !=
           SS_LINEAR_STOP_PRECONDITIONED_RESIDUAL_ORIGINAL_RHS &&
       policy->stopping_condition != SS_LINEAR_STOP_NATURAL_RESIDUAL_INITIAL) ||
      (policy->initial_guess_policy != SS_LINEAR_INITIAL_GUESS_COARSE_ONLY &&
       policy->initial_guess_policy != SS_LINEAR_INITIAL_GUESS_PREVIOUS_COMPLEMENT &&
       policy->initial_guess_policy !=
           SS_LINEAR_INITIAL_GUESS_PREVIOUS_COMPLEMENT_IF_BETTER) ||
      (policy->orthogonalization != SS_LINEAR_ORTHOGONALIZATION_PETSC &&
       (policy->method != SS_LINEAR_METHOD_FGMRES ||
        policy->orthogonalization < SS_LINEAR_ORTHOGONALIZATION_CGS_REFINED ||
        policy->orthogonalization > SS_LINEAR_ORTHOGONALIZATION_MGS)) ||
      !ss_is_finite(policy->atol) || !ss_is_finite(policy->dtol) ||
      policy->rtol <= 0.0 || policy->atol < 0.0 || policy->dtol <= 1.0 ||
      policy->maximum_iterations <= 0 || policy->restart <= 0 ||
      (policy->method == SS_LINEAR_METHOD_HPDDM_GCRODR &&
       (policy->recycle_dimension <= 0 ||
        policy->recycle_dimension >= policy->restart)) ||
      (policy->method != SS_LINEAR_METHOD_HPDDM_GCRODR &&
       policy->recycle_dimension != 0) ||
      ((policy->stopping_condition ==
            SS_LINEAR_STOP_PRECONDITIONED_RESIDUAL_ORIGINAL_RHS ||
        policy->stopping_condition == SS_LINEAR_STOP_NATURAL_RESIDUAL_INITIAL) &&
       (policy->method != SS_LINEAR_METHOD_FCG ||
        policy->side != SS_LINEAR_SIDE_LEFT)) ||
      policy->vector_memory_cap_bytes == 0U || !properties->finite_action) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  switch (policy->operator_class) {
  case SS_LINEAR_OPERATOR_GENERAL:
    if ((policy->method != SS_LINEAR_METHOD_FGMRES &&
         policy->method != SS_LINEAR_METHOD_HPDDM_GCRODR) ||
        policy->side != SS_LINEAR_SIDE_RIGHT) {
      return SS_ERROR_TYPE_MISMATCH;
    }
    break;
  case SS_LINEAR_OPERATOR_SYMMETRIC_INDEFINITE:
    if (policy->method != SS_LINEAR_METHOD_MINRES ||
        policy->side != SS_LINEAR_SIDE_LEFT || !properties->measured_symmetric ||
        !properties->pc_symmetric || !properties->pc_positive ||
        properties->pc_variable || properties->nonlinear_hydraulic) {
      return SS_ERROR_TYPE_MISMATCH;
    }
    break;
  case SS_LINEAR_OPERATOR_SPD_VERIFIED:
    if ((policy->method != SS_LINEAR_METHOD_CG &&
         policy->method != SS_LINEAR_METHOD_FCG) ||
        policy->side != SS_LINEAR_SIDE_LEFT || !properties->measured_symmetric ||
        !properties->positive_energy || !properties->pc_symmetric ||
        !properties->pc_positive ||
        (policy->method == SS_LINEAR_METHOD_CG && properties->pc_variable) ||
        properties->nonlinear_hydraulic) {
      return SS_ERROR_TYPE_MISMATCH;
    }
    break;
  case SS_LINEAR_OPERATOR_DIRECT_VALIDATION:
    if (policy->method != SS_LINEAR_METHOD_PREONLY || policy->pc != SS_LINEAR_PC_LU ||
        policy->side != SS_LINEAR_SIDE_LEFT) {
      return SS_ERROR_TYPE_MISMATCH;
    }
    break;
  default:
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return SS_OK;
}

SsStatus ss_linear_classify_acceptance(const SsLinearPolicy *policy,
                                       KSPConvergedReason reason,
                                       PetscReal reported_norm, PetscReal rhs_norm,
                                       PetscReal true_residual_norm,
                                       SsLinearSolveStatus *status,
                                       PetscReal *threshold) {
  PetscReal relative_threshold;
  if (policy == NULL || status == NULL || threshold == NULL ||
      !ss_is_finite(policy->rtol) || !ss_is_finite(policy->atol) ||
      policy->rtol <= 0.0 || policy->atol < 0.0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (!ss_is_finite(reported_norm) || !ss_is_finite(rhs_norm) ||
      !ss_is_finite(true_residual_norm) || rhs_norm < 0.0 || true_residual_norm < 0.0) {
    *threshold = policy->atol;
    *status = SS_LINEAR_SOLVE_NONFINITE;
    return SS_OK;
  }
  relative_threshold = policy->rtol * rhs_norm;
  *threshold = relative_threshold > policy->atol ? relative_threshold : policy->atol;
  if (reason <= 0) {
    *status = SS_LINEAR_SOLVE_DIVERGED_KSP;
  } else if (true_residual_norm > *threshold) {
    *status = SS_LINEAR_SOLVE_RESIDUAL_VERIFICATION_FAILED;
  } else {
    *status = SS_LINEAR_SOLVE_CONVERGED;
  }
  return SS_OK;
}

static PetscErrorCode ss_projected_operator_mult(Mat shell, Vec input, Vec output) {
  SsProjectedOperatorContext *context = NULL;
  PetscFunctionBegin;
  PetscCall(MatShellGetContext(shell, &context));
  const SsStatus status = ss_deflation_apply_projected_operator_local(
      context->deflation, context->operator_handle, input, output);
  context->failed |= status != SS_OK;
  PetscCall(VecFlag(output, status != SS_OK));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ss_projected_operator_create_vecs(Mat shell, Vec *right,
                                                        Vec *left) {
  SsProjectedOperatorContext *context = NULL;
  PetscFunctionBegin;
  PetscCall(MatShellGetContext(shell, &context));
  PetscCheck(context != NULL && context->operator_handle != NULL,
             PetscObjectComm((PetscObject)shell), PETSC_ERR_ARG_WRONGSTATE,
             "projected operator context is incomplete");
  PetscCall(MatCreateVecs(context->operator_handle, right, left));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ss_projected_pmg_pc_apply(PC pc, Vec input, Vec output) {
  SsProjectedPmgPcContext *context = NULL;
  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &context));
  const SsStatus pmg_status = ss_pmg_apply_local(context->pmg, input, context->work);
  /* The projection still enters its dot product if PMG failed locally. */
  const SsStatus projection_status =
      ss_deflation_apply_cached_right_local(context->deflation, context->work, output);
  const PetscErrorCode capture_status =
      ss_linear_krylov_capture_append(context->capture, output);
  const int failed = pmg_status != SS_OK || projection_status != SS_OK ||
                     capture_status != PETSC_SUCCESS;
  if (failed) {
    PetscCall(PCSetFailedReason(pc, PC_SUBPC_ERROR));
  }
  PetscCall(VecFlag(output, failed));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void ss_linear_reset_setup(SsLinearSolver *solver) {
  ss_linear_krylov_capture_destroy(&solver->captured_krylov);
  if (solver->ksp != NULL) {
    (void)KSPDestroy(&solver->ksp);
  }
  if (solver->projected_operator != NULL) {
    (void)MatDestroy(&solver->projected_operator);
  }
  if (solver->projected_rhs != NULL) {
    (void)VecDestroy(&solver->projected_rhs);
  }
  if (solver->complement != NULL) {
    (void)VecDestroy(&solver->complement);
  }
  if (solver->residual != NULL) {
    (void)VecDestroy(&solver->residual);
  }
  if (solver->projected_pc_context.work != NULL) {
    (void)VecDestroy(&solver->projected_pc_context.work);
  }
  if (solver->operator_handle != NULL) {
    (void)MatDestroy(&solver->operator_handle);
  }
  if (solver->preconditioner_operator_handle != NULL) {
    (void)MatDestroy(&solver->preconditioner_operator_handle);
  }
  (void)PetscFree(solver->options_snapshot);
  solver->options_snapshot = NULL;
  solver->deflation = NULL;
  solver->pmg = NULL;
  solver->krylov_recycle = NULL;
  solver->krylov_capture = SS_LINEAR_KRYLOV_CAPTURE_NONE;
  solver->recycled_dimension_supplied = 0;
  solver->deflation_dimension_supplied = 0U;
  solver->deflation_refresh_count = 0U;
  solver->solve_since_setup = 0;
  solver->projected_context.deflation = NULL;
  solver->projected_context.operator_handle = NULL;
  solver->projected_pc_context.pmg = NULL;
  solver->projected_pc_context.deflation = NULL;
  solver->projected_pc_context.capture = NULL;
  solver->structure_generation = 0U;
  solver->value_generation = 0U;
  solver->operator_state = 0;
  solver->preconditioner_operator_state = 0;
  solver->operator_nonzero_state = 0;
  solver->preconditioner_nonzero_state = 0;
  solver->setup_complete = 0;
  memset(&solver->selected_policy, 0, sizeof(solver->selected_policy));
  memset(&solver->properties, 0, sizeof(solver->properties));
  memset(&solver->policy_chain, 0, sizeof(solver->policy_chain));
  memset(&solver->last_setup_record, 0, sizeof(solver->last_setup_record));
}

static int ss_linear_setup_reusable(const SsLinearSolver *solver,
                                    const SsLinearProblem *problem) {
  char *current_options = NULL;
  PetscObjectState current_operator_state = 0;
  PetscObjectState current_preconditioner_state = 0;
  Mat requested_preconditioner = problem->preconditioner_operator_handle == NULL
                                     ? problem->operator_handle
                                     : problem->preconditioner_operator_handle;
  int reusable;
  if (!solver->setup_complete || solver->deflation != problem->deflation ||
      solver->krylov_capture != problem->krylov_capture ||
      solver->pmg != problem->pmg ||
      (solver->selected_policy.method == SS_LINEAR_METHOD_HPDDM_GCRODR &&
       solver->value_generation != problem->value_generation) ||
      solver->krylov_recycle != problem->krylov_recycle ||
      solver->operator_handle != problem->operator_handle ||
      solver->preconditioner_operator_handle != requested_preconditioner ||
      solver->structure_generation != problem->structure_generation ||
      !ss_linear_properties_equal(&solver->properties, &problem->properties) ||
      !ss_linear_chain_equal(&solver->policy_chain, &problem->policy_chain) ||
      solver->options_snapshot == NULL ||
      MatGetState(problem->operator_handle, &current_operator_state) != PETSC_SUCCESS ||
      MatGetState(requested_preconditioner, &current_preconditioner_state) !=
          PETSC_SUCCESS ||
      (problem->value_generation == solver->value_generation &&
       (current_operator_state != solver->operator_state ||
        current_preconditioner_state != solver->preconditioner_operator_state)) ||
      PetscOptionsGetAll(NULL, &current_options) != PETSC_SUCCESS) {
    return 0;
  }
  reusable =
      current_options != NULL && strcmp(solver->options_snapshot, current_options) == 0;
  (void)PetscFree(current_options);
  return reusable;
}

SsStatus ss_linear_create(MPI_Comm communicator, SsLinearSolver **result) {
  SsLinearSolver *solver = NULL;
  SsStatus status = SS_OK;
  if (communicator == MPI_COMM_NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_status_allreduce_max(communicator,
                                   result != NULL ? SS_OK : SS_ERROR_INVALID_ARGUMENT);
  if (status != SS_OK) {
    return status;
  }
  *result = NULL;
  if ((ss_linear_setup_event < 0 &&
       PetscLogEventRegister("SlopeLinearSetup", 0, &ss_linear_setup_event) !=
           PETSC_SUCCESS) ||
      (ss_linear_solve_event < 0 &&
       PetscLogEventRegister("SlopeLinearSolve", 0, &ss_linear_solve_event) !=
           PETSC_SUCCESS) ||
      (ss_linear_deflation_refresh_event < 0 &&
       PetscLogEventRegister("SlopeDeflRefresh", 0,
                             &ss_linear_deflation_refresh_event) != PETSC_SUCCESS) ||
      (ss_linear_pmg_setup_event < 0 &&
       PetscLogEventRegister("SlopePmgSetup", 0, &ss_linear_pmg_setup_event) !=
           PETSC_SUCCESS) ||
      (ss_linear_initial_guess_event < 0 &&
       PetscLogEventRegister("SlopeLinGuess", 0, &ss_linear_initial_guess_event) !=
           PETSC_SUCCESS) ||
      (ss_linear_krylov_recycle_event < 0 &&
       PetscLogEventRegister("SlopeKrylovRecycle", 0,
                             &ss_linear_krylov_recycle_event) != PETSC_SUCCESS) ||
      (ss_linear_full_krylov_event < 0 &&
       PetscLogEventRegister("SlopeFullKrylov", 0, &ss_linear_full_krylov_event) !=
           PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  } else {
    solver = calloc(1U, sizeof(*solver));
    if (solver == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    free(solver);
    return status;
  }
  if (MPI_Comm_dup(communicator, &solver->communicator) != MPI_SUCCESS) {
    free(solver);
    return SS_ERROR_INTERNAL;
  }
  *result = solver;
  return SS_OK;
}

SsStatus ss_linear_admit_vectors(PetscInt global_rows, const SsLinearPolicy *policy,
                                 SsLinearKrylovCapture capture, size_t basis_bytes,
                                 size_t *required_bytes) {
  size_t vectors = 8U;
  size_t entries;
  size_t bytes;
  if (global_rows <= 0 || policy == NULL || policy->restart <= 0 ||
      policy->maximum_iterations <= 0 || policy->recycle_dimension < 0 ||
      required_bytes == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  size_t restart;
  size_t rows;
  size_t basis_vectors;
  if (ss_checked_u64_to_size((uint64_t)policy->restart, &restart) != SS_OK ||
      ss_checked_u64_to_size((uint64_t)global_rows, &rows) != SS_OK ||
      ss_checked_mul_size(restart, 2U, &basis_vectors) != SS_OK) {
    return SS_ERROR_OVERFLOW;
  }
  if (policy->method == SS_LINEAR_METHOD_FGMRES ||
      policy->method == SS_LINEAR_METHOD_FCG) {
    /* Both Arnoldi V/Z and FCG P/A*P keep two restart-sized bases. */
    if (ss_checked_add_size(basis_vectors, 10U, &vectors) != SS_OK) {
      return SS_ERROR_OVERFLOW;
    }
  } else if (policy->method == SS_LINEAR_METHOD_HPDDM_GCRODR) {
    size_t recycle;
    if (ss_checked_u64_to_size((uint64_t)policy->recycle_dimension, &recycle) !=
            SS_OK ||
        ss_checked_add_size(basis_vectors, recycle, &vectors) != SS_OK ||
        ss_checked_add_size(vectors, 16U, &vectors) != SS_OK) {
      return SS_ERROR_OVERFLOW;
    }
  }
  if (capture == SS_LINEAR_KRYLOV_CAPTURE_FULL_CORRECTION_SPACE) {
    /* FCG can apply the PC once more to evaluate the terminal norm. */
    size_t captures;
    if (ss_checked_u64_to_size((uint64_t)policy->maximum_iterations, &captures) !=
            SS_OK ||
        ss_checked_add_size(captures, 1U, &captures) != SS_OK ||
        ss_checked_add_size(vectors, captures, &vectors) != SS_OK) {
      return SS_ERROR_OVERFLOW;
    }
  }
  if (ss_checked_mul_size(vectors, rows, &entries) != SS_OK ||
      ss_checked_mul_size(entries, sizeof(PetscScalar), &bytes) != SS_OK ||
      ss_checked_add_size(bytes, basis_bytes, required_bytes) != SS_OK) {
    return SS_ERROR_OVERFLOW;
  }
  return *required_bytes <= policy->vector_memory_cap_bytes ? SS_OK : SS_ERROR_RESOURCE;
}

static SsStatus
ss_linear_vector_budget(Mat operator_handle, const SsLinearPolicy *policy,
                        SsLinearKrylovCapture capture, const SsDeflation *deflation,
                        size_t *admitted_bytes, size_t *dense_bytes_per_rank) {
  PetscInt rows = 0;
  PetscInt columns = 0;
  size_t basis_bytes = 0U;
  *dense_bytes_per_rank = 0U;
  if (MatGetSize(operator_handle, &rows, &columns) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (rows <= 0 || rows != columns) {
    return SS_ERROR_TYPE_MISMATCH;
  }
  if (deflation != NULL) {
    const SsStatus status = ss_deflation_memory_estimate(deflation, rows, &basis_bytes,
                                                         dense_bytes_per_rank);
    if (status != SS_OK) {
      return status;
    }
  }
  return ss_linear_admit_vectors(rows, policy, capture, basis_bytes, admitted_bytes);
}

static SsStatus ss_linear_unused_prefixed_options(const char *prefix) {
  PetscInt count = 0;
  char **names = NULL;
  char **values = NULL;
  PetscInt index;
  int found = 0;
  size_t prefix_length = strlen(prefix);
  if (PetscOptionsLeftGet(NULL, &count, &names, &values) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  for (index = 0; index < count; ++index) {
    const char *name = names[index];
    if (name != NULL && name[0] == '-') {
      ++name;
    }
    if (name != NULL && strncmp(name, prefix, prefix_length) == 0) {
      found = 1;
      break;
    }
  }
  if (PetscOptionsLeftRestore(NULL, &count, &names, &values) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  return found != 0 ? SS_ERROR_INTEGRITY : SS_OK;
}

#if defined(PETSC_HAVE_HPDDM)
static SsStatus ss_linear_hpddm_option_name(const SsLinearPolicy *policy,
                                            const char *suffix, char *name,
                                            size_t capacity) {
  const int written = snprintf(name, capacity, "-%s%s", policy->options_prefix, suffix);
  return written >= 0 && (size_t)written < capacity ? SS_OK : SS_ERROR_OVERFLOW;
}

static SsStatus ss_linear_audit_hpddm_options(KSP ksp, const SsLinearPolicy *policy,
                                              PetscInt *restart,
                                              SsLinearSetupRecord *record) {
  char option[SS_LINEAR_PREFIX_CAPACITY + 48U];
  char value[SS_LINEAR_IDENTITY_CAPACITY];
  PetscBool set = PETSC_FALSE;
  PetscInt integer_value = 0;
  KSPHPDDMType type = KSP_HPDDM_TYPE_GMRES;
  SsStatus status;
#define SS_HPDDM_STRING_OPTION(suffix, destination)                                    \
  do {                                                                                 \
    status = ss_linear_hpddm_option_name(policy, suffix, option, sizeof(option));      \
    if (status != SS_OK ||                                                             \
        PetscOptionsGetString(NULL, NULL, option, value, sizeof(value), &set) !=       \
            PETSC_SUCCESS ||                                                           \
        set != PETSC_TRUE) {                                                           \
      return status != SS_OK ? status : SS_ERROR_INTEGRITY;                            \
    }                                                                                  \
    (void)snprintf(destination, sizeof(destination), "%s", value);                     \
  } while (0)
  if (KSPHPDDMGetType(ksp, &type) != PETSC_SUCCESS || type != KSP_HPDDM_TYPE_GCRODR) {
    return SS_ERROR_INTEGRITY;
  }
  status =
      ss_linear_hpddm_option_name(policy, "ksp_gmres_restart", option, sizeof(option));
  if (status != SS_OK ||
      PetscOptionsGetInt(NULL, NULL, option, &integer_value, &set) != PETSC_SUCCESS ||
      set != PETSC_TRUE || integer_value != policy->restart) {
    return status != SS_OK ? status : SS_ERROR_INTEGRITY;
  }
  *restart = integer_value;
  status =
      ss_linear_hpddm_option_name(policy, "ksp_hpddm_recycle", option, sizeof(option));
  if (status != SS_OK ||
      PetscOptionsGetInt(NULL, NULL, option, &integer_value, &set) != PETSC_SUCCESS ||
      set != PETSC_TRUE || integer_value != policy->recycle_dimension) {
    return status != SS_OK ? status : SS_ERROR_INTEGRITY;
  }
  record->effective_recycle_dimension = integer_value;
  status =
      ss_linear_hpddm_option_name(policy, "ksp_hpddm_type", option, sizeof(option));
  if (status != SS_OK ||
      PetscOptionsGetString(NULL, NULL, option, value, sizeof(value), &set) !=
          PETSC_SUCCESS ||
      set != PETSC_TRUE || strcmp(value, "gcrodr") != 0) {
    return status != SS_OK ? status : SS_ERROR_INTEGRITY;
  }
  SS_HPDDM_STRING_OPTION("ksp_hpddm_variant", record->effective_recycle_variant);
  if (strcmp(record->effective_recycle_variant, "flexible") != 0) {
    return SS_ERROR_INTEGRITY;
  }
  SS_HPDDM_STRING_OPTION("ksp_hpddm_recycle_target", record->effective_recycle_target);
  if (strcmp(record->effective_recycle_target, "SM") != 0 &&
      strcmp(record->effective_recycle_target, "LM") != 0 &&
      strcmp(record->effective_recycle_target, "SR") != 0 &&
      strcmp(record->effective_recycle_target, "LR") != 0 &&
      strcmp(record->effective_recycle_target, "SI") != 0 &&
      strcmp(record->effective_recycle_target, "LI") != 0) {
    return SS_ERROR_INTEGRITY;
  }
  SS_HPDDM_STRING_OPTION("ksp_hpddm_recycle_strategy",
                         record->effective_recycle_strategy);
  if (strcmp(record->effective_recycle_strategy, "A") != 0 &&
      strcmp(record->effective_recycle_strategy, "B") != 0) {
    return SS_ERROR_INTEGRITY;
  }
  SS_HPDDM_STRING_OPTION("ksp_hpddm_orthogonalization",
                         record->effective_orthogonalization);
  if (strcmp(record->effective_orthogonalization, "mgs") != 0 &&
      strcmp(record->effective_orthogonalization, "cgs") != 0) {
    return SS_ERROR_INTEGRITY;
  }
#undef SS_HPDDM_STRING_OPTION
  return SS_OK;
}
#endif

static SsStatus ss_linear_audit_effective(KSP ksp, const SsLinearPolicy *policy,
                                          PetscBool expected_initial_guess_nonzero,
                                          SsLinearSetupRecord *record) {
  const char *ksp_type = NULL;
  const char *pc_type = NULL;
  PC pc = NULL;
  PCSide side = PC_LEFT;
  KSPNormType norm_type = KSP_NORM_DEFAULT;
  PetscBool initial_guess_nonzero = PETSC_TRUE;
  PetscBool initial_guess_knoll = PETSC_TRUE;
  PetscBool error_if_not_converged = PETSC_TRUE;
  PetscReal rtol = 0.0;
  PetscReal atol = 0.0;
  PetscReal dtol = 0.0;
  PetscInt maximum_iterations = 0;
  PetscInt restart = policy->restart;
  PetscInt fcg_nprealloc = 0;
  KSPFCDTruncationType fcg_truncation = KSP_FCD_TRUNC_TYPE_NOTAY;
  PetscErrorCode (*orthogonalization)(KSP, PetscInt) = NULL;
  KSPGMRESCGSRefinementType cgs_refinement = KSP_GMRES_CGS_REFINE_NEVER;
  const char *expected_ksp = ss_linear_method_name(policy->method);
  const char *expected_pc = ss_linear_pc_name(policy->pc);
  SsStatus status;
  if (KSPGetType(ksp, &ksp_type) != PETSC_SUCCESS || ksp_type == NULL) {
    return SS_ERROR_INTERNAL;
  }
  if (strcmp(ksp_type, expected_ksp) != 0) {
    return SS_ERROR_INTEGRITY;
  }
  if (KSPGetPC(ksp, &pc) != PETSC_SUCCESS || PCGetType(pc, &pc_type) != PETSC_SUCCESS ||
      pc_type == NULL) {
    return SS_ERROR_INTERNAL;
  }
  if (strcmp(pc_type, expected_pc) != 0) {
    return SS_ERROR_INTEGRITY;
  }
  if (KSPGetPCSide(ksp, &side) != PETSC_SUCCESS ||
      KSPGetNormType(ksp, &norm_type) != PETSC_SUCCESS ||
      KSPGetInitialGuessNonzero(ksp, &initial_guess_nonzero) != PETSC_SUCCESS ||
      KSPGetInitialGuessKnoll(ksp, &initial_guess_knoll) != PETSC_SUCCESS ||
      KSPGetErrorIfNotConverged(ksp, &error_if_not_converged) != PETSC_SUCCESS ||
      KSPGetTolerances(ksp, &rtol, &atol, &dtol, &maximum_iterations) !=
          PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (policy->method == SS_LINEAR_METHOD_FGMRES) {
    if (KSPGMRESGetRestart(ksp, &restart) != PETSC_SUCCESS ||
        KSPGMRESGetOrthogonalization(ksp, &orthogonalization) != PETSC_SUCCESS ||
        KSPGMRESGetCGSRefinementType(ksp, &cgs_refinement) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    if (orthogonalization == KSPGMRESModifiedGramSchmidtOrthogonalization) {
      PetscBool explicit_refinement = PETSC_FALSE;
      if (PetscOptionsHasName(NULL, policy->options_prefix,
                              "-ksp_gmres_cgs_refinement_type",
                              &explicit_refinement) != PETSC_SUCCESS) {
        return SS_ERROR_INTERNAL;
      }
      if (!explicit_refinement) {
        cgs_refinement = KSP_GMRES_CGS_REFINE_NEVER;
        if (KSPGMRESSetCGSRefinementType(ksp, cgs_refinement) != PETSC_SUCCESS) {
          return SS_ERROR_INTERNAL;
        }
      }
    }
  }
  if (policy->method == SS_LINEAR_METHOD_FCG &&
      (KSPFCGGetMmax(ksp, &restart) != PETSC_SUCCESS ||
       KSPFCGGetNprealloc(ksp, &fcg_nprealloc) != PETSC_SUCCESS ||
       KSPFCGGetTruncationType(ksp, &fcg_truncation) != PETSC_SUCCESS)) {
    return SS_ERROR_INTERNAL;
  }
#if defined(PETSC_HAVE_HPDDM)
  if (policy->method == SS_LINEAR_METHOD_HPDDM_GCRODR) {
    status = ss_linear_audit_hpddm_options(ksp, policy, &restart, record);
    if (status != SS_OK) {
      return status;
    }
  }
#endif
  if (policy->method == SS_LINEAR_METHOD_FGMRES) {
    const int mgs = orthogonalization == KSPGMRESModifiedGramSchmidtOrthogonalization;
    const int cgs = orthogonalization == KSPGMRESClassicalGramSchmidtOrthogonalization;
    const int refined = cgs && cgs_refinement == KSP_GMRES_CGS_REFINE_ALWAYS;
    const int unrefined = cgs && cgs_refinement == KSP_GMRES_CGS_REFINE_NEVER;
    const int modified = mgs && cgs_refinement == KSP_GMRES_CGS_REFINE_NEVER;
    const int valid =
        policy->orthogonalization == SS_LINEAR_ORTHOGONALIZATION_PETSC
            ? refined || modified
        : policy->orthogonalization == SS_LINEAR_ORTHOGONALIZATION_CGS_REFINED ? refined
        : policy->orthogonalization == SS_LINEAR_ORTHOGONALIZATION_CGS_UNREFINED
            ? unrefined
            : modified;
    if (!valid) {
      return SS_ERROR_INTEGRITY;
    }
  }
  if (side != ss_linear_petsc_side(policy->side) ||
      norm_type != ss_linear_norm_type(policy) ||
      initial_guess_nonzero != expected_initial_guess_nonzero ||
      initial_guess_knoll != PETSC_FALSE || error_if_not_converged != PETSC_FALSE ||
      maximum_iterations != policy->maximum_iterations ||
      ((policy->method == SS_LINEAR_METHOD_FGMRES ||
        policy->method == SS_LINEAR_METHOD_HPDDM_GCRODR ||
        policy->method == SS_LINEAR_METHOD_FCG) &&
       restart != policy->restart) ||
      (policy->method == SS_LINEAR_METHOD_FCG &&
       fcg_nprealloc != policy->restart + 1) ||
      rtol != policy->rtol || atol != policy->atol || dtol != policy->dtol) {
    return SS_ERROR_INTEGRITY;
  }
  status = ss_linear_unused_prefixed_options(policy->options_prefix);
  if (status != SS_OK) {
    return status;
  }
  (void)snprintf(record->effective_ksp, sizeof(record->effective_ksp), "%s", ksp_type);
  (void)snprintf(record->effective_pc, sizeof(record->effective_pc), "%s", pc_type);
  record->effective_side = ss_linear_project_side(side);
  record->effective_norm_type = norm_type;
  record->effective_initial_guess_nonzero = initial_guess_nonzero;
  record->effective_initial_guess_knoll = initial_guess_knoll;
  record->effective_error_if_not_converged = error_if_not_converged;
  record->effective_rtol = rtol;
  record->effective_atol = atol;
  record->effective_dtol = dtol;
  record->effective_restart = restart;
  if (policy->method == SS_LINEAR_METHOD_FGMRES) {
    (void)snprintf(record->effective_orthogonalization,
                   sizeof(record->effective_orthogonalization), "%s",
                   orthogonalization == KSPGMRESClassicalGramSchmidtOrthogonalization
                       ? (cgs_refinement == KSP_GMRES_CGS_REFINE_NEVER
                              ? "classical_gram_schmidt_once"
                              : "classical_gram_schmidt_twice")
                       : "modified_gram_schmidt");
  } else if (policy->method == SS_LINEAR_METHOD_FCG) {
    (void)snprintf(record->effective_orthogonalization,
                   sizeof(record->effective_orthogonalization), "%s",
                   fcg_truncation == KSP_FCD_TRUNC_TYPE_STANDARD
                       ? "fcg_standard_a_orthogonalization"
                       : "fcg_notay_a_orthogonalization");
  }
  record->effective_maximum_iterations = maximum_iterations;
  return SS_OK;
}

static SsStatus
ss_linear_admit_policy(SsLinearSolver *solver, const SsLinearProblem *problem,
                       const SsLinearPolicy *policy, PetscBool *aorth_projected_pc,
                       size_t *admitted_bytes, size_t *dense_bytes_per_rank) {
  SsDeflationProjector deflation_projector = SS_DEFLATION_PROJECTOR_GENERAL_LEFT_RIGHT;
  Mat preconditioner_operator = solver->preconditioner_operator_handle;
  PetscBool preconditioner_is_shell = PETSC_FALSE;
  SsStatus status;
  if (solver->deflation != NULL) {
    status = ss_deflation_projector(solver->deflation, &deflation_projector);
    if (status != SS_OK) {
      return status;
    }
    *aorth_projected_pc =
        deflation_projector == SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL ? PETSC_TRUE
                                                                          : PETSC_FALSE;
  }
  status = ss_linear_policy_validate(policy, &problem->properties);
  if (status != SS_OK) {
    return status;
  }
  if (solver->deflation != NULL) {
    const int flexible_gmres = policy->operator_class == SS_LINEAR_OPERATOR_GENERAL &&
                               (policy->method == SS_LINEAR_METHOD_FGMRES ||
                                policy->method == SS_LINEAR_METHOD_HPDDM_GCRODR) &&
                               policy->side == SS_LINEAR_SIDE_RIGHT;
    const int flexible_cg = *aorth_projected_pc == PETSC_TRUE &&
                            policy->operator_class == SS_LINEAR_OPERATOR_SPD_VERIFIED &&
                            policy->method == SS_LINEAR_METHOD_FCG &&
                            policy->side == SS_LINEAR_SIDE_LEFT;
    if (flexible_gmres == 0 && flexible_cg == 0) {
      return SS_ERROR_TYPE_MISMATCH;
    }
  }
  if (*aorth_projected_pc == PETSC_TRUE && policy->pc != SS_LINEAR_PC_PMG) {
    return SS_ERROR_TYPE_MISMATCH;
  }
  if (solver->deflation != NULL && *aorth_projected_pc == PETSC_FALSE &&
      policy->initial_guess_policy != SS_LINEAR_INITIAL_GUESS_COARSE_ONLY) {
    return SS_ERROR_TYPE_MISMATCH;
  }
  if (solver->krylov_capture == SS_LINEAR_KRYLOV_CAPTURE_FULL_CORRECTION_SPACE &&
      (*aorth_projected_pc == PETSC_FALSE ||
       (policy->method != SS_LINEAR_METHOD_FGMRES &&
        policy->method != SS_LINEAR_METHOD_FCG) ||
       policy->pc != SS_LINEAR_PC_PMG)) {
    return SS_ERROR_TYPE_MISMATCH;
  }
  status =
      ss_linear_vector_budget(problem->operator_handle, policy, solver->krylov_capture,
                              solver->deflation, admitted_bytes, dense_bytes_per_rank);
  if (status != SS_OK) {
    return status;
  }
  if (PetscObjectTypeCompare((PetscObject)preconditioner_operator, MATSHELL,
                             &preconditioner_is_shell) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (policy->pc == SS_LINEAR_PC_LU && preconditioner_is_shell) {
    return SS_ERROR_TYPE_MISMATCH;
  }
  if (policy->pc == SS_LINEAR_PC_PMG && problem->pmg == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return SS_OK;
}

static SsStatus ss_linear_pmg_setup(SsLinearSolver *solver,
                                    const SsLinearProblem *problem,
                                    SsLinearSetupRecord *record) {
  (void)PetscLogEventBegin(ss_linear_pmg_setup_event, 0, 0, 0, 0);
  const SsStatus status = ss_pmg_setup(
      problem->pmg, solver->preconditioner_operator_handle,
      problem->structure_generation, problem->value_generation, &record->pmg_setup);
  (void)PetscLogEventEnd(ss_linear_pmg_setup_event, 0, 0, 0, 0);
  return status;
}

static SsStatus ss_linear_try_policy(SsLinearSolver *solver,
                                     const SsLinearProblem *problem,
                                     const SsLinearPolicy *policy,
                                     SsLinearSetupRecord *record) {
  const char *method_name = ss_linear_method_name(policy->method);
  const char *pc_name = ss_linear_pc_name(policy->pc);
  PetscBool aorth_projected_pc = PETSC_FALSE;
  /* The general projector uses the fused A*P_R operator shell.  The legacy
   * A-orthogonal projector leaves A untouched and wraps PMG as P_R*M. */
  Mat ksp_operator = solver->projected_operator != NULL ? solver->projected_operator
                                                        : solver->operator_handle;
  Mat preconditioner_operator = solver->preconditioner_operator_handle;
  KSP candidate = NULL;
  PC pc = NULL;
  PetscErrorCode petsc_status;
  SsStatus status;
  size_t admitted_bytes = 0U, dense_bytes_per_rank = 0U;
  status = ss_status_allreduce_max(
      solver->communicator,
      ss_linear_admit_policy(solver, problem, policy, &aorth_projected_pc,
                             &admitted_bytes, &dense_bytes_per_rank));
  if (status != SS_OK) {
    return status;
  }
  if (policy->pc == SS_LINEAR_PC_PMG) {
    status = ss_linear_pmg_setup(solver, problem, record);
    if (status != SS_OK) {
      return status;
    }
  }
  status = ss_linear_petsc_status(solver->communicator,
                                  KSPCreate(solver->communicator, &candidate));
  if (status != SS_OK) {
    goto failed;
  }
  status = ss_linear_petsc_status(
      solver->communicator, KSPSetOptionsPrefix(candidate, policy->options_prefix));
  if (status == SS_OK) {
    status = ss_linear_petsc_status(
        solver->communicator,
        KSPSetOperators(candidate, ksp_operator, preconditioner_operator));
  }
  if (status == SS_OK) {
    status = ss_linear_petsc_status(solver->communicator,
                                    KSPSetType(candidate, method_name));
  }
  if (status == SS_OK) {
    status = ss_linear_petsc_status(solver->communicator, KSPGetPC(candidate, &pc));
  }
#if defined(PETSC_HAVE_HPDDM)
  if (status == SS_OK && policy->method == SS_LINEAR_METHOD_HPDDM_GCRODR) {
    status = ss_linear_petsc_status(solver->communicator,
                                    KSPHPDDMSetType(candidate, KSP_HPDDM_TYPE_GCRODR));
  }
#endif
  if (status == SS_OK &&
      (policy->pc == SS_LINEAR_PC_PMG && aorth_projected_pc == PETSC_TRUE
           ? (PCSetType(pc, PCSHELL) == PETSC_SUCCESS &&
                      PCShellSetContext(pc, &solver->projected_pc_context) ==
                          PETSC_SUCCESS &&
                      PCShellSetName(pc, "slope-pmg-vcycle-aorth-deflated") ==
                          PETSC_SUCCESS &&
                      PCShellSetApply(pc, ss_projected_pmg_pc_apply) == PETSC_SUCCESS
                  ? SS_OK
                  : SS_ERROR_INTERNAL)
       : policy->pc == SS_LINEAR_PC_PMG
           ? ss_pmg_configure_pc(problem->pmg, pc)
           : (PCSetType(pc, pc_name) == PETSC_SUCCESS ? SS_OK : SS_ERROR_INTERNAL)) !=
          SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK &&
      ((policy->method != SS_LINEAR_METHOD_HPDDM_GCRODR &&
        KSPSetPCSide(candidate, ss_linear_petsc_side(policy->side)) != PETSC_SUCCESS) ||
       KSPSetNormType(candidate, ss_linear_norm_type(policy)) != PETSC_SUCCESS ||
       KSPSetTolerances(candidate, policy->rtol, policy->atol, policy->dtol,
                        policy->maximum_iterations) != PETSC_SUCCESS ||
       ((policy->stopping_condition ==
             SS_LINEAR_STOP_PRECONDITIONED_RESIDUAL_ORIGINAL_RHS ||
         policy->stopping_condition == SS_LINEAR_STOP_NATURAL_RESIDUAL_INITIAL) &&
        /* A nonzero deflation coarse guess otherwise makes PETSc compute a
         * preconditioned norm of b solely to initialize its default rtol scale.
         * Preconditioned-original-RHS stopping uses rtol=0 per solve, while the
         * natural contract explicitly scales from the actual initial residual;
         * in both cases this also avoids an unused extra PMG application. */
        KSPConvergedDefaultSetUIRNorm(candidate) != PETSC_SUCCESS) ||
       KSPSetInitialGuessNonzero(candidate,
                                 aorth_projected_pc == PETSC_TRUE ||
                                         policy->initial_guess_policy !=
                                             SS_LINEAR_INITIAL_GUESS_COARSE_ONLY
                                     ? PETSC_TRUE
                                     : PETSC_FALSE) != PETSC_SUCCESS ||
       KSPSetInitialGuessKnoll(candidate, PETSC_FALSE) != PETSC_SUCCESS ||
       KSPSetErrorIfNotConverged(candidate, PETSC_FALSE) != PETSC_SUCCESS ||
       (policy->method == SS_LINEAR_METHOD_FGMRES &&
        (KSPGMRESSetRestart(candidate, policy->restart) != PETSC_SUCCESS ||
         KSPGMRESSetOrthogonalization(
             candidate, policy->orthogonalization == SS_LINEAR_ORTHOGONALIZATION_MGS
                            ? KSPGMRESModifiedGramSchmidtOrthogonalization
                            : KSPGMRESClassicalGramSchmidtOrthogonalization) !=
             PETSC_SUCCESS ||
         KSPGMRESSetCGSRefinementType(
             candidate, policy->orthogonalization == SS_LINEAR_ORTHOGONALIZATION_MGS ||
                                policy->orthogonalization ==
                                    SS_LINEAR_ORTHOGONALIZATION_CGS_UNREFINED
                            ? KSP_GMRES_CGS_REFINE_NEVER
                            : KSP_GMRES_CGS_REFINE_ALWAYS) != PETSC_SUCCESS)) ||
       (policy->method == SS_LINEAR_METHOD_FCG &&
        (KSPFCGSetMmax(candidate, policy->restart) != PETSC_SUCCESS ||
         KSPFCGSetNprealloc(candidate, policy->restart + 1) != PETSC_SUCCESS ||
         KSPFCGSetTruncationType(candidate, KSP_FCD_TRUNC_TYPE_NOTAY) !=
             PETSC_SUCCESS)))) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(solver->communicator, status);
  if (status != SS_OK) {
    goto failed;
  }
  status = ss_linear_petsc_status(solver->communicator, KSPSetFromOptions(candidate));
  if (status != SS_OK) {
    goto failed;
  }
  /* PETSc's default convergence test retains only the current residual norm.
   * The native FCG natural-residual contract is relative to the initial
   * sqrt(r^T B r), so retain the per-solve history in order to report and
   * independently verify that exact denominator after KSPSolve returns. */
  if (policy->stopping_condition == SS_LINEAR_STOP_NATURAL_RESIDUAL_INITIAL &&
      KSPSetResidualHistory(candidate, NULL, (PetscCount)policy->maximum_iterations + 1,
                            PETSC_TRUE) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    status = ss_linear_audit_effective(candidate, policy,
                                       aorth_projected_pc == PETSC_TRUE ||
                                               policy->initial_guess_policy !=
                                                   SS_LINEAR_INITIAL_GUESS_COARSE_ONLY
                                           ? PETSC_TRUE
                                           : PETSC_FALSE,
                                       record);
  }
  if (status == SS_OK && KSPMonitorSet(candidate, ss_linear_krylov_monitor, solver,
                                       NULL) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(solver->communicator, status);
  if (status != SS_OK) {
    goto failed;
  }
  const int handler_pushed =
      PetscPushErrorHandler(PetscReturnErrorHandler, NULL) == PETSC_SUCCESS;
  status = ss_status_allreduce_max(solver->communicator,
                                   handler_pushed ? SS_OK : SS_ERROR_INTERNAL);
  if (status == SS_OK) {
    petsc_status = KSPSetUp(candidate);
    status = petsc_status == PETSC_SUCCESS ? SS_OK : SS_ERROR_INTERNAL;
  }
  if (handler_pushed && PetscPopErrorHandler() != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(solver->communicator, status);
  if (status != SS_OK) {
    goto failed;
  }
#if defined(PETSC_HAVE_HPDDM)
  if (policy->method == SS_LINEAR_METHOD_HPDDM_GCRODR &&
      solver->krylov_recycle != NULL && solver->krylov_recycle->space != NULL) {
    PetscInt supplied_dimension = 0;
    const int valid = MatGetSize(solver->krylov_recycle->space, NULL,
                                 &supplied_dimension) == PETSC_SUCCESS &&
                      supplied_dimension > 0;
    status = ss_status_allreduce_max(solver->communicator,
                                     valid ? SS_OK : SS_ERROR_INTERNAL);
    if (status == SS_OK) {
      status = ss_linear_petsc_status(
          solver->communicator,
          KSPHPDDMSetDeflationMat(candidate, solver->krylov_recycle->space));
    }
    if (status != SS_OK) {
      goto failed;
    }
    solver->recycled_dimension_supplied = supplied_dimension;
    record->recycled_dimension_supplied = supplied_dimension;
  }
#endif
  if (solver->krylov_capture == SS_LINEAR_KRYLOV_CAPTURE_FULL_CORRECTION_SPACE) {
    const size_t slots = (size_t)policy->maximum_iterations + 1U;
    SsLinearKrylovCaptureBuffer *capture = &solver->captured_krylov;
    capture->vectors = calloc(slots, sizeof(*capture->vectors));
    status = ss_status_allreduce_max(solver->communicator,
                                     capture->vectors ? SS_OK : SS_ERROR_OUT_OF_MEMORY);
    if (status != SS_OK) {
      goto failed;
    }
    /* Admission covers all capture vectors. Allocate at setup so no rank can
     * leave the iteration schedule early due to capture growth. */
    while (capture->allocated < slots) {
      Vec *next = &capture->vectors[capture->allocated];
      status = ss_linear_petsc_status(solver->communicator,
                                      VecDuplicate(solver->projected_rhs, next));
      if (status != SS_OK) {
        (void)VecDestroy(next);
        goto failed;
      }
      ++capture->allocated;
    }
  }
  solver->ksp = candidate;
  record->admitted_vector_bytes = admitted_bytes;
  record->admitted_deflation_dense_bytes_per_rank = dense_bytes_per_rank;
  record->pmg_selected = policy->pc == SS_LINEAR_PC_PMG;
  record->deflation_projected_pc = aorth_projected_pc == PETSC_TRUE ? 1 : 0;
  return SS_OK;

failed:
  ss_linear_krylov_capture_destroy(&solver->captured_krylov);
  (void)KSPDestroy(&candidate);
  return status;
}

static SsStatus ss_linear_refresh_deflation(SsLinearSolver *solver,
                                            const SsLinearProblem *problem) {
  if (problem->deflation == NULL) {
    return SS_OK;
  }
  SsDeflationDiagnostics diagnostics = {0};
  (void)PetscLogEventBegin(ss_linear_deflation_refresh_event, 0, 0, 0, 0);
  SsStatus status = ss_deflation_refresh(problem->deflation, problem->operator_handle,
                                         problem->value_generation);
  (void)PetscLogEventEnd(ss_linear_deflation_refresh_event, 0, 0, 0, 0);
  if (status == SS_OK &&
      (ss_deflation_diagnostics(problem->deflation, &diagnostics) != SS_OK ||
       !diagnostics.invariants_validated ||
       !ss_is_finite(diagnostics.maximum_invariant_error))) {
    status = SS_ERROR_INTEGRITY;
  }
  status = ss_status_allreduce_max(solver->communicator, status);
  if (status == SS_OK) {
    solver->deflation_dimension_supplied = diagnostics.active_count;
    solver->deflation_refresh_count = diagnostics.refresh_count;
  }
  return status;
}

static SsStatus ss_linear_prepare_operator(SsLinearSolver *solver,
                                           const SsLinearProblem *problem) {
  Mat preconditioner = problem->preconditioner_operator_handle == NULL
                           ? problem->operator_handle
                           : problem->preconditioner_operator_handle;
  PetscInt local_rows = 0, local_columns = 0, global_rows = 0, global_columns = 0;
  SsDeflationProjector projector = SS_DEFLATION_PROJECTOR_GENERAL_LEFT_RIGHT;
  SsStatus status = SS_OK;
  if (PetscObjectReference((PetscObject)problem->operator_handle) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  } else {
    solver->operator_handle = problem->operator_handle;
    if (PetscObjectReference((PetscObject)preconditioner) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    } else {
      solver->preconditioner_operator_handle = preconditioner;
    }
  }
  solver->deflation = problem->deflation;
  solver->pmg = problem->pmg;
  solver->krylov_recycle = problem->krylov_recycle;
  solver->krylov_capture = problem->krylov_capture;
  if (status == SS_OK && solver->deflation != NULL) {
    if (ss_deflation_projector(solver->deflation, &projector) != SS_OK ||
        MatGetLocalSize(problem->operator_handle, &local_rows, &local_columns) !=
            PETSC_SUCCESS ||
        MatGetSize(problem->operator_handle, &global_rows, &global_columns) !=
            PETSC_SUCCESS ||
        local_rows != local_columns || global_rows != global_columns) {
      status = SS_ERROR_TYPE_MISMATCH;
    }
  }
  status = ss_status_allreduce_max(solver->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  status = ss_linear_refresh_deflation(solver, problem);
  if (status != SS_OK) {
    return status;
  }
  if (solver->deflation != NULL &&
      projector == SS_DEFLATION_PROJECTOR_GENERAL_LEFT_RIGHT) {
    solver->projected_context.deflation = solver->deflation;
    solver->projected_context.operator_handle = solver->operator_handle;
    if (MatCreateShell(solver->communicator, local_rows, local_columns, global_rows,
                       global_columns, &solver->projected_context,
                       &solver->projected_operator) != PETSC_SUCCESS ||
        MatShellSetOperation(solver->projected_operator, MATOP_MULT,
                             (void (*)(void))ss_projected_operator_mult) !=
            PETSC_SUCCESS ||
        MatShellSetOperation(solver->projected_operator, MATOP_CREATE_VECS,
                             (void (*)(void))ss_projected_operator_create_vecs) !=
            PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  status = ss_status_allreduce_max(solver->communicator, status);
  if (status == SS_OK && MatCreateVecs(problem->operator_handle, &solver->complement,
                                       &solver->projected_rhs) != PETSC_SUCCESS) {
    status = SS_ERROR_OUT_OF_MEMORY;
  }
  status = ss_status_allreduce_max(solver->communicator, status);
  if (status == SS_OK &&
      VecDuplicate(solver->projected_rhs, &solver->residual) != PETSC_SUCCESS) {
    status = SS_ERROR_OUT_OF_MEMORY;
  }
  status = ss_status_allreduce_max(solver->communicator, status);
  if (status == SS_OK && solver->deflation != NULL &&
      projector == SS_DEFLATION_PROJECTOR_LEGACY_A_ORTHOGONAL) {
    solver->projected_pc_context.pmg = solver->pmg;
    solver->projected_pc_context.deflation = solver->deflation;
    solver->projected_pc_context.capture = &solver->captured_krylov;
    if (VecDuplicate(solver->projected_rhs, &solver->projected_pc_context.work) !=
        PETSC_SUCCESS) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  return ss_status_allreduce_max(solver->communicator, status);
}

static SsStatus ss_linear_reuse_setup(SsLinearSolver *solver,
                                      const SsLinearProblem *problem,
                                      SsLinearSetupRecord *record) {
  const uint64_t previous_refresh = solver->deflation_refresh_count;
  SsStatus status = ss_linear_refresh_deflation(solver, problem);
  *record = solver->last_setup_record;
  if (status == SS_OK && solver->selected_policy.pc == SS_LINEAR_PC_PMG) {
    status = ss_linear_pmg_setup(solver, problem, record);
  }
  if (status != SS_OK) {
    return status;
  }
  if (problem->value_generation != solver->value_generation ||
      previous_refresh != solver->deflation_refresh_count) {
    Mat ksp_operator = solver->projected_operator != NULL ? solver->projected_operator
                                                          : solver->operator_handle;
    if (solver->projected_operator != NULL) {
      status = ss_linear_petsc_status(
          solver->communicator, MatAssemblyBegin(ksp_operator, MAT_FINAL_ASSEMBLY));
      if (status == SS_OK) {
        status = ss_linear_petsc_status(
            solver->communicator, MatAssemblyEnd(ksp_operator, MAT_FINAL_ASSEMBLY));
      }
    }
    if (status == SS_OK) {
      status = ss_linear_petsc_status(
          solver->communicator,
          KSPSetOperators(solver->ksp, ksp_operator,
                          solver->preconditioner_operator_handle));
    }
    if (status == SS_OK) {
      status = ss_linear_petsc_status(solver->communicator, KSPSetUp(solver->ksp));
    }
    if (status != SS_OK) {
      return status;
    }
  }
  status = ss_status_allreduce_max(
      solver->communicator,
      MatGetState(solver->operator_handle, &solver->operator_state) == PETSC_SUCCESS &&
              MatGetState(solver->preconditioner_operator_handle,
                          &solver->preconditioner_operator_state) == PETSC_SUCCESS
          ? SS_OK
          : SS_ERROR_INTERNAL);
  if (status == SS_OK) {
    solver->value_generation = problem->value_generation;
    record->value_generation = problem->value_generation;
    record->operator_state = solver->operator_state;
    record->reused = 1;
    solver->last_setup_record = *record;
  }
  return status;
}

static SsStatus ss_linear_setup_impl(SsLinearSolver *solver,
                                     const SsLinearProblem *problem,
                                     SsLinearSetupRecord *record) {
  SsStatus status;
  SsStatus last_status = SS_ERROR_INVALID_ARGUMENT;
  size_t index;
  if (solver == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = SS_OK;
  if (problem == NULL || record == NULL || problem->operator_handle == NULL ||
      problem->structure_generation == 0U || problem->value_generation == 0U ||
      problem->policy_chain.count == 0U ||
      problem->policy_chain.count > SS_LINEAR_MAX_POLICIES ||
      (problem->krylov_capture != SS_LINEAR_KRYLOV_CAPTURE_NONE &&
       problem->krylov_capture != SS_LINEAR_KRYLOV_CAPTURE_FULL_CORRECTION_SPACE)) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else if (solver->captured_krylov.count != 0U) {
    status = SS_ERROR_INVALID_STATE;
  } else if (solver->setup_complete &&
             solver->structure_generation == problem->structure_generation) {
    Mat requested_pc = problem->preconditioner_operator_handle != NULL
                           ? problem->preconditioner_operator_handle
                           : problem->operator_handle;
    Mat matrices[] = {problem->operator_handle, requested_pc};
    Mat observed[] = {solver->operator_handle, solver->preconditioner_operator_handle};
    PetscObjectState states[] = {solver->operator_state,
                                 solver->preconditioner_operator_state};
    PetscObjectState patterns[] = {solver->operator_nonzero_state,
                                   solver->preconditioner_nonzero_state};
    for (size_t i = 0U; status == SS_OK && i < sizeof(matrices) / sizeof(*matrices);
         ++i) {
      if (matrices[i] == observed[i]) {
        PetscObjectState state = 0, pattern = 0;
        if (MatGetState(matrices[i], &state) != PETSC_SUCCESS ||
            MatGetNonzeroState(matrices[i], &pattern) != PETSC_SUCCESS) {
          status = SS_ERROR_INTERNAL;
        } else if (pattern != patterns[i] ||
                   (solver->value_generation == problem->value_generation &&
                    state != states[i])) {
          status = SS_ERROR_INTEGRITY;
        }
      }
    }
  }
  status = ss_status_allreduce_max(solver->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  const int local_reusable = ss_linear_setup_reusable(solver, problem);
  int reusable = 0;
  if (MPI_Allreduce(&local_reusable, &reusable, 1, MPI_INT, MPI_MIN,
                    solver->communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (reusable) {
    status = ss_linear_reuse_setup(solver, problem, record);
    if (status == SS_OK) {
      return SS_OK;
    }
    ss_linear_reset_setup(solver);
    return status;
  }
  status = ss_status_allreduce_max(solver->communicator,
                                   ss_linear_capture_krylov_recycle(solver));
  if (status != SS_OK) {
    return status;
  }
  ss_linear_reset_setup(solver);
  memset(record, 0, sizeof(*record));
  record->structure_generation = problem->structure_generation;
  record->value_generation = problem->value_generation;
  status = ss_linear_prepare_operator(solver, problem);
  if (status != SS_OK) {
    ss_linear_reset_setup(solver);
    return status;
  }
  for (index = 0U; index < problem->policy_chain.count; ++index) {
    SsLinearSetupAttempt *attempt = &record->attempts[index];
    const SsLinearPolicy *policy = &problem->policy_chain.policies[index];
    (void)snprintf(attempt->identity, sizeof(attempt->identity), "%s",
                   policy->identity);
    status = ss_status_allreduce_max(
        solver->communicator, ss_linear_try_policy(solver, problem, policy, record));
    attempt->setup_status = status;
    ++record->attempt_count;
    last_status = status;
    if (status == SS_OK) {
      PetscErrorCode options_status;
      attempt->selected = 1;
      solver->selected_policy = *policy;
      solver->properties = problem->properties;
      solver->policy_chain = problem->policy_chain;
      solver->pmg = problem->pmg;
      solver->structure_generation = problem->structure_generation;
      solver->value_generation = problem->value_generation;
      if (MatGetState(problem->operator_handle, &solver->operator_state) !=
              PETSC_SUCCESS ||
          MatGetState(solver->preconditioner_operator_handle,
                      &solver->preconditioner_operator_state) != PETSC_SUCCESS ||
          MatGetNonzeroState(problem->operator_handle,
                             &solver->operator_nonzero_state) != PETSC_SUCCESS ||
          MatGetNonzeroState(solver->preconditioner_operator_handle,
                             &solver->preconditioner_nonzero_state) != PETSC_SUCCESS) {
        status = SS_ERROR_INTERNAL;
      }
      options_status = PetscOptionsGetAll(NULL, &solver->options_snapshot);
      if (options_status != PETSC_SUCCESS || solver->options_snapshot == NULL) {
        status = options_status == PETSC_SUCCESS ? SS_ERROR_OUT_OF_MEMORY
                                                 : SS_ERROR_INTERNAL;
      }
      status = ss_status_allreduce_max(solver->communicator, status);
      if (status != SS_OK) {
        ss_linear_reset_setup(solver);
        return status;
      }
      record->operator_state = solver->operator_state;
      solver->setup_complete = 1;
      ++solver->setup_count;
      record->setup_count = solver->setup_count;
      (void)snprintf(record->selected_identity, sizeof(record->selected_identity), "%s",
                     policy->identity);
      solver->last_setup_record = *record;
      return SS_OK;
    }
  }
  ss_linear_reset_setup(solver);
  return last_status;
}

SsStatus ss_linear_setup(SsLinearSolver *solver, const SsLinearProblem *problem,
                         SsLinearSetupRecord *record) {
#ifndef NDEBUG
  if (problem != NULL && problem->operator_handle != NULL) {
    ss_debug_stage_matrix("linear-matrix", (size_t)problem->value_generation,
                          problem->operator_handle);
  }
#endif
  (void)PetscLogEventBegin(ss_linear_setup_event, 0, 0, 0, 0);
  const SsStatus status = ss_linear_setup_impl(solver, problem, record);
  (void)PetscLogEventEnd(ss_linear_setup_event, 0, 0, 0, 0);
  return status;
}

SsStatus ss_linear_finish_attempt(SsLinearSolver *solver) {
  if (solver == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  const SsStatus status = ss_status_allreduce_max(
      solver->communicator, ss_linear_capture_krylov_recycle(solver));
  ss_linear_krylov_capture_clear(&solver->captured_krylov);
  return status;
}

SsStatus ss_linear_invalidate(SsLinearSolver *solver, uint64_t generation_mask) {
  SsStatus status;
  if (solver == NULL || generation_mask == 0U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_linear_capture_krylov_recycle(solver);
  ss_linear_krylov_capture_destroy(&solver->captured_krylov);
  ss_linear_reset_setup(solver);
  return status;
}

SsStatus ss_linear_destroy(SsLinearSolver **solver_pointer) {
  SsLinearSolver *solver;
  SsStatus status;
  if (solver_pointer == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  solver = *solver_pointer;
  if (solver == NULL) {
    return SS_OK;
  }
  status = ss_linear_capture_krylov_recycle(solver);
  ss_linear_krylov_capture_destroy(&solver->captured_krylov);
  ss_linear_reset_setup(solver);
  if (solver->communicator != MPI_COMM_NULL) {
    if (MPI_Comm_free(&solver->communicator) != MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  free(solver);
  *solver_pointer = NULL;
  return status;
}
