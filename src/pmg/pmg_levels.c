/* Level KSPs, world hierarchy and variable point blocks.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "pmg_internal.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/core/util.h"
#include "slope_internal/diagnostics/diagnostics.h"

#include <petscconf.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscsys.h>

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef PetscErrorCode (*SsPmgInverseVariableBlockApply)(Mat, Vec, Vec);

static PetscErrorCode ss_pmg_threaded_vblock_pc_apply(PC pc, Vec rhs, Vec correction) {
  Mat operator_handle = NULL;
  SsPmgInverseVariableBlockApply apply = NULL;
  PetscFunctionBegin;
  PetscCall(PCGetOperators(pc, NULL, &operator_handle));
  PetscCheck(operator_handle != NULL, PetscObjectComm((PetscObject)pc),
             PETSC_ERR_ARG_WRONGSTATE, "threaded variable-block PC has no operator");
  PetscCall(PetscObjectQueryFunction((PetscObject)operator_handle,
                                     "SsMatApplyInverseVariableBlocks_C", &apply));
  PetscCheck(apply != NULL, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP,
             "operator has no threaded variable-block inverse apply");
  PetscCall((*apply)(operator_handle, rhs, correction));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ss_pmg_vblock_pc_configure(PC pc, Mat operator_handle) {
  SsPmgInverseVariableBlockApply apply = NULL;
  PetscFunctionBegin;
  PetscCall(PetscObjectQueryFunction((PetscObject)operator_handle,
                                     "SsMatApplyInverseVariableBlocks_C", &apply));
  if (apply != NULL) {
    PetscCall(PCSetType(pc, PCSHELL));
    PetscCall(PCShellSetName(pc, "slope-threaded-variable-block-jacobi"));
    PetscCall(PCShellSetApply(pc, ss_pmg_threaded_vblock_pc_apply));
  } else {
    PetscCall(PCSetType(pc, PCVPBJACOBI));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * FGMRES always normalizes its initial residual, even for KSP_NORM_NONE.
 * Preserve fixed work for nonzero corrections while terminating the exact
 * zero-right-hand-side case before PETSc attempts a 1 / 0 normalization.
 */
static PetscErrorCode ss_pmg_converged_fixed_work(KSP ksp, PetscInt iteration,
                                                  PetscReal residual_norm,
                                                  KSPConvergedReason *reason,
                                                  void *context) {
  PetscFunctionBegin;
  if (PetscIsInfOrNanReal(residual_norm)) {
    *reason = KSP_DIVERGED_NANORINF;
  } else if (residual_norm == 0.0) {
    *reason = KSP_CONVERGED_ATOL;
  } else {
    PetscCall(KSPConvergedSkip(ksp, iteration, residual_norm, reason, context));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static SsStatus ss_pmg_level_ksp_parameters(SsPmg *pmg, size_t level_index, PC pc) {
  SsPmgLevel *level = &pmg->levels[level_index];
  const PetscInt smoother_iterations =
      PetscMax(level->policy.pre_smooth_steps, level->policy.post_smooth_steps);
  switch (level->policy.method) {
  case SS_PMG_LEVEL_RICHARDSON_NONE:
  case SS_PMG_LEVEL_RICHARDSON_JACOBI:
  case SS_PMG_LEVEL_RICHARDSON_VPBJACOBI:
    if (KSPSetType(level->ksp, KSPRICHARDSON) != PETSC_SUCCESS ||
        (level->policy.method == SS_PMG_LEVEL_RICHARDSON_VPBJACOBI &&
                 level->has_variable_point_blocks
             ? ss_pmg_vblock_pc_configure(pc, level->active_operator)
             : PCSetType(pc, level->policy.method == SS_PMG_LEVEL_RICHARDSON_NONE
                                 ? PCNONE
                                 : PCJACOBI)) != PETSC_SUCCESS ||
        KSPRichardsonSetScale(level->ksp, level->policy.richardson_scale) !=
            PETSC_SUCCESS ||
        KSPSetNormType(level->ksp, KSP_NORM_NONE) != PETSC_SUCCESS ||
        KSPSetConvergenceTest(level->ksp, KSPConvergedSkip, NULL, NULL) !=
            PETSC_SUCCESS ||
        KSPSetTolerances(level->ksp, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT,
                         smoother_iterations) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    break;
  case SS_PMG_LEVEL_CHEBYSHEV_JACOBI:
  case SS_PMG_LEVEL_CHEBYSHEV_POINT_JACOBI:
    if (KSPSetType(level->ksp, KSPCHEBYSHEV) != PETSC_SUCCESS ||
        (level->policy.method != SS_PMG_LEVEL_CHEBYSHEV_POINT_JACOBI &&
                 level->has_variable_point_blocks
             ? ss_pmg_vblock_pc_configure(pc, level->active_operator)
             : PCSetType(pc, PCJACOBI)) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    if (pmg->hierarchy.fine_order >= 4U) {
      /*
       * A fixed [0.25, 5.1] interval is too narrow for Jacobi-scaled P4
       * tangents and can amplify high modes. Fourth-kind Chebyshev only needs
       * a reliable upper spectral estimate and damps small modes without an
       * ad-hoc positive lower bound. PETSc re-estimates when the matrix state
       * changes; prefixed options can still override this default for
       * controlled experiments. Apply this consistently across a P4
       * hierarchy while keeping the measured standalone P2 policy unchanged.
       */
      if (KSPChebyshevSetKind(level->ksp, KSP_CHEBYSHEV_OPT_FOURTH) != PETSC_SUCCESS ||
          KSPChebyshevEstEigSet(level->ksp, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
                                PETSC_DECIDE) != PETSC_SUCCESS) {
        return SS_ERROR_INTERNAL;
      }
    } else if (KSPChebyshevSetEigenvalues(level->ksp, 5.1, 0.25) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    if (KSPSetNormType(level->ksp, KSP_NORM_NONE) != PETSC_SUCCESS ||
        KSPSetConvergenceTest(level->ksp, KSPConvergedSkip, NULL, NULL) !=
            PETSC_SUCCESS ||
        KSPSetTolerances(level->ksp, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT,
                         smoother_iterations) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    break;
  case SS_PMG_LEVEL_FGMRES_JACOBI:
  case SS_PMG_LEVEL_FGMRES_VPBJACOBI:
  case SS_PMG_LEVEL_FGMRES_GAMG: {
    PetscInt restart =
        level_index == 0U ? level->policy.maximum_iterations : smoother_iterations;
    if (restart > 30) {
      restart = 30;
    }
    if (KSPSetType(level->ksp, KSPFGMRES) != PETSC_SUCCESS ||
        (level->policy.method == SS_PMG_LEVEL_FGMRES_VPBJACOBI &&
                 level->has_variable_point_blocks
             ? ss_pmg_vblock_pc_configure(pc, level->active_operator)
             : PCSetType(pc, level->policy.method == SS_PMG_LEVEL_FGMRES_GAMG
                                 ? PCGAMG
                                 : PCJACOBI)) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    if (level_index == 0U) {
      if (KSPSetNormType(level->ksp, KSP_NORM_UNPRECONDITIONED) != PETSC_SUCCESS ||
          KSPSetTolerances(level->ksp, level->policy.rtol, level->policy.atol,
                           PETSC_DEFAULT,
                           level->policy.maximum_iterations) != PETSC_SUCCESS) {
        return SS_ERROR_INTERNAL;
      }
    } else if (ss_pmg_fgmres_full_iterate_level(pmg, level_index) != 0) {
      /*
       * Match petsc4py's KSP defaults exactly for the legacy manual-PMG
       * smoother.  Its builder sets only rtol=0, atol=0 and max_it, leaving
       * FGMRES on the default unpreconditioned norm and PETSc's default
       * convergence test.  KSP_NORM_NONE plus a skip callback produces the
       * same fixed iteration count for ordinary right-hand sides, but changes
       * the residual bookkeeping and the exact finite-precision Krylov path.
       */
      if (KSPSetNormType(level->ksp, KSP_NORM_UNPRECONDITIONED) != PETSC_SUCCESS ||
          KSPSetTolerances(level->ksp, 0.0, 0.0, PETSC_CURRENT, smoother_iterations) !=
              PETSC_SUCCESS) {
        return SS_ERROR_INTERNAL;
      }
    } else if (KSPSetNormType(level->ksp, KSP_NORM_NONE) != PETSC_SUCCESS ||
               KSPSetConvergenceTest(level->ksp, ss_pmg_converged_fixed_work, NULL,
                                     NULL) != PETSC_SUCCESS ||
               KSPSetTolerances(level->ksp, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT,
                                smoother_iterations) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    /* Fixed-work smoothers need the low-reduction one-pass CGS path; the
     * outer FGMRES uses its own orthogonalization policy.  Pin these
     * defaults before KSPSetFromOptions so a recorded prefixed option can
     * still override them. */
    if (KSPGMRESSetRestart(level->ksp, restart) != PETSC_SUCCESS ||
        KSPGMRESSetOrthogonalization(level->ksp,
                                     KSPGMRESClassicalGramSchmidtOrthogonalization) !=
            PETSC_SUCCESS ||
        KSPGMRESSetCGSRefinementType(level->ksp, KSP_GMRES_CGS_REFINE_NEVER) !=
            PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    break;
  }
  case SS_PMG_LEVEL_PREONLY_LU:
    if (KSPSetType(level->ksp, KSPPREONLY) != PETSC_SUCCESS ||
        PCSetType(pc, PCLU) != PETSC_SUCCESS ||
        KSPSetNormType(level->ksp, KSP_NORM_NONE) != PETSC_SUCCESS ||
        KSPSetTolerances(level->ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1) !=
            PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    break;
  case SS_PMG_LEVEL_PREONLY_SUPERLU_DIST:
#if defined(PETSC_HAVE_SUPERLU_DIST)
  {
    char factor_option[SS_PMG_PREFIX_CAPACITY + 32U];
    char factor_policy[64] = "";
    PetscBool factor_policy_set = PETSC_FALSE;
    const int written =
        snprintf(factor_option, sizeof(factor_option), "-%smat_superlu_dist_fact",
                 level->policy.options_prefix);
    if (written < 0 || (size_t)written >= sizeof(factor_option) ||
        PetscOptionsGetString(NULL, NULL, factor_option, factor_policy,
                              sizeof(factor_policy),
                              &factor_policy_set) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    level->repeated_factorization_reuses_row_permutation =
        factor_policy_set != PETSC_FALSE &&
        strcmp(factor_policy, "SamePattern_SameRowPerm") == 0;
  }
    if (KSPSetType(level->ksp, KSPPREONLY) != PETSC_SUCCESS ||
        PCSetType(pc, PCLU) != PETSC_SUCCESS ||
        PCFactorSetMatSolverType(pc, MATSOLVERSUPERLU_DIST) != PETSC_SUCCESS ||
        KSPSetNormType(level->ksp, KSP_NORM_NONE) != PETSC_SUCCESS ||
        KSPSetTolerances(level->ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1) !=
            PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    break;
#else
    return SS_ERROR_RESOURCE;
#endif
  case SS_PMG_LEVEL_PREONLY_REDUNDANT_LU: {
    KSP redundant_ksp = NULL;
    PC redundant_pc = NULL;
    if (KSPSetType(level->ksp, KSPPREONLY) != PETSC_SUCCESS ||
        PCSetType(pc, PCREDUNDANT) != PETSC_SUCCESS ||
        PCRedundantSetNumber(pc, level->policy.redundant_solve_count) !=
            PETSC_SUCCESS ||
        PCRedundantGetKSP(pc, &redundant_ksp) != PETSC_SUCCESS ||
        KSPSetType(redundant_ksp, KSPPREONLY) != PETSC_SUCCESS ||
        KSPSetNormType(redundant_ksp, KSP_NORM_NONE) != PETSC_SUCCESS ||
        KSPSetTolerances(redundant_ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT,
                         1) != PETSC_SUCCESS ||
        KSPSetInitialGuessNonzero(redundant_ksp, PETSC_FALSE) != PETSC_SUCCESS ||
        KSPSetErrorIfNotConverged(redundant_ksp, PETSC_FALSE) != PETSC_SUCCESS ||
        KSPGetPC(redundant_ksp, &redundant_pc) != PETSC_SUCCESS ||
        PCSetType(redundant_pc, PCLU) != PETSC_SUCCESS ||
        KSPSetNormType(level->ksp, KSP_NORM_NONE) != PETSC_SUCCESS ||
        KSPSetTolerances(level->ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1) !=
            PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    break;
  }
  default:
    return SS_ERROR_INVALID_ARGUMENT;
  }
  /* SuperLU_DIST's SamePattern default recomputes its numeric row
   * permutation.  Reusing that permutation is faster for a fixed matrix but
   * was unstable across the evolving elastoplastic tangent. */
  return SS_OK;
}

static SsStatus ss_pmg_level_ksp_configure(SsPmg *pmg, size_t level_index) {
  SsPmgLevel *level = &pmg->levels[level_index];
  PC pc = NULL;
  SsStatus status = SS_OK;
  if (!level->local_active) {
    return level->active_operator == NULL && level->ksp == NULL ? SS_OK
                                                                : SS_ERROR_INTEGRITY;
  }
  const MPI_Comm comm = level->active_communicator;
#ifndef NDEBUG
  if (level->active_operator != NULL && level_index == 0U) {
    ss_debug_stage_matrix("pmg-active-coarse-matrix", level_index,
                          level->active_operator);
  }
#endif
  status = ss_status_allreduce_max(
      comm, level->active_operator == NULL ? SS_ERROR_INVALID_STATE : SS_OK);
  if (status == SS_OK && MatCreateVecs(level->active_operator, &level->active_solution,
                                       &level->active_rhs) != PETSC_SUCCESS) {
    status = SS_ERROR_OUT_OF_MEMORY;
  }
  status = ss_status_allreduce_max(comm, status);
  if (status == SS_OK &&
      (KSPCreate(comm, &level->ksp) != PETSC_SUCCESS ||
       KSPSetOptionsPrefix(level->ksp, level->policy.options_prefix) != PETSC_SUCCESS ||
       KSPSetOperators(level->ksp, level->active_operator, level->active_operator) !=
           PETSC_SUCCESS ||
       KSPGetPC(level->ksp, &pc) != PETSC_SUCCESS ||
       KSPSetInitialGuessNonzero(level->ksp,
                                 ss_pmg_fgmres_full_iterate_level(pmg, level_index)
                                     ? PETSC_TRUE
                                     : PETSC_FALSE) != PETSC_SUCCESS ||
       KSPSetErrorIfNotConverged(level->ksp, PETSC_FALSE) != PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(comm, status);
  if (status == SS_OK) {
    status = ss_pmg_level_ksp_parameters(pmg, level_index, pc);
  }
  status = ss_status_allreduce_max(comm, status);
  if (status == SS_OK && KSPSetFromOptions(level->ksp) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(comm, status);
  if (status == SS_OK && KSPSetUp(level->ksp) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(comm, status);
  if (status != SS_OK) {
    return status;
  }
  level->local_owns_ksp = 1;
  ++level->solver_setup_count;
  ++pmg->level_solver_setup_count;
  return SS_OK;
}

static SsStatus ss_pmg_level_ksp_refresh(SsPmg *pmg, size_t level_index) {
  SsPmgLevel *level = &pmg->levels[level_index];
  PetscBool reuse_preconditioner = PETSC_FALSE;
  SsStatus status = SS_OK;
  if (level->local_active == 0) {
    return level->ksp == NULL ? SS_OK : SS_ERROR_INTEGRITY;
  }
  if (level->ksp == NULL || level->active_operator == NULL ||
      KSPGetReusePreconditioner(level->ksp, &reuse_preconditioner) != PETSC_SUCCESS ||
      KSPSetOperators(level->ksp, level->active_operator, level->active_operator) !=
          PETSC_SUCCESS ||
      KSPSetReusePreconditioner(level->ksp, reuse_preconditioner) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(level->active_communicator, status);
  if (status != SS_OK || KSPSetUp(level->ksp) != PETSC_SUCCESS) {
    return status != SS_OK ? status : SS_ERROR_INTERNAL;
  }
  ++level->solver_setup_count;
  ++pmg->level_solver_setup_count;
  return SS_OK;
}

static SsStatus ss_pmg_transfer_validate(SsPmg *pmg, size_t coarse_level,
                                         PetscInt fine_size) {
  Mat prolongation = pmg->prolongations[coarse_level];
  PetscInt rows = 0;
  PetscInt columns = 0;
  PetscObjectState state = 0;
  PetscReal norm = 0.0;
  PetscInt row_start = 0;
  PetscInt row_end = 0;
  PetscInt row;
  SsStatus status = SS_OK;
  if (MatGetSize(prolongation, &rows, &columns) != PETSC_SUCCESS ||
      MatGetState(prolongation, &state) != PETSC_SUCCESS ||
      MatGetOwnershipRange(prolongation, &row_start, &row_end) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  if (MatNorm(prolongation, NORM_INFINITY, &norm) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  } else if (rows != fine_size || columns <= 0 ||
             state != pmg->prolongation_states[coarse_level] || !ss_is_finite(norm) ||
             norm <= 0.0) {
    status = SS_ERROR_INTEGRITY;
  }
  for (row = row_start; status == SS_OK && row < row_end; ++row) {
    PetscInt count = 0;
    const PetscInt *indices = NULL;
    const PetscScalar *values = NULL;
    PetscInt entry;
    if (MatGetRow(prolongation, row, &count, &indices, &values) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
      break;
    }
    /*
     * A fine free DOF can have no support in the constrained coarse
     * free-space basis.  Such a row is intentionally zero: the fine
     * smoother owns that mode and coarse correction contributes nothing.
     */
    if (count < 0) {
      status = SS_ERROR_INTEGRITY;
    }
    for (entry = 0; entry < count; ++entry) {
      if (indices[entry] < 0 || indices[entry] >= columns ||
          PetscIsInfOrNanScalar(values[entry])) {
        status = SS_ERROR_INTEGRITY;
      }
    }
    if (MatRestoreRow(prolongation, row, &count, &indices, &values) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
      break;
    }
  }
  return ss_status_allreduce_max(pmg->communicator, status);
}

static SsStatus ss_pmg_variable_blocks_copy(Mat source, Mat destination) {
  PetscInt block_count = 0;
  const PetscInt *block_sizes = NULL;
  PetscInt source_start = 0;
  PetscInt source_end = 0;
  PetscInt destination_start = 0;
  PetscInt destination_end = 0;
  PetscInt covered = 0;
  PetscInt block;
  if (source == NULL || destination == NULL ||
      MatGetVariableBlockSizes(source, &block_count, &block_sizes) != PETSC_SUCCESS ||
      MatGetOwnershipRange(source, &source_start, &source_end) != PETSC_SUCCESS ||
      MatGetOwnershipRange(destination, &destination_start, &destination_end) !=
          PETSC_SUCCESS ||
      block_count < 0 || (block_count != 0 && block_sizes == NULL) ||
      source_start != destination_start || source_end != destination_end) {
    return SS_ERROR_INTEGRITY;
  }
  if (block_count == 0) {
    return SS_OK;
  }
  for (block = 0; block < block_count; ++block) {
    if (block_sizes[block] <= 0 || covered > PETSC_MAX_INT - block_sizes[block]) {
      return SS_ERROR_INTEGRITY;
    }
    covered += block_sizes[block];
  }
  if (covered != source_end - source_start ||
      MatSetVariableBlockSizes(destination, block_count, block_sizes) !=
          PETSC_SUCCESS) {
    return SS_ERROR_INTEGRITY;
  }
  return SS_OK;
}

static SsStatus ss_pmg_galerkin_build(SsPmg *pmg, size_t fine_level) {
  Mat fine = pmg->operators[fine_level];
  Mat transfer = pmg->prolongations[fine_level - 1U];
  Mat *coarse = &pmg->operators[fine_level - 1U];
  if (!pmg->ordered_galerkin || pmg->world_size == 1) {
    return MatPtAP(fine, transfer, MAT_INITIAL_MATRIX, 2.0, coarse) == PETSC_SUCCESS
               ? SS_OK
               : SS_ERROR_INTERNAL;
  }
  /* The MPIAIJ backend gathers remote contributions into rank-ordered
   * PetscSF slots and sums them on the owner using a preallocated COO
   * insertion plan. Each numeric refresh reuses the local product layouts
   * and the communication plan. */
  SsStatus status =
      MatProductCreate(fine, transfer, NULL, coarse) == PETSC_SUCCESS &&
              MatProductSetType(*coarse, MATPRODUCT_PtAP) == PETSC_SUCCESS &&
              MatProductSetAlgorithm(*coarse, "backend") == PETSC_SUCCESS &&
              MatProductSetFill(*coarse, 2.0) == PETSC_SUCCESS
          ? SS_OK
          : SS_ERROR_INTERNAL;
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status == SS_OK && MatProductSetFromOptions(*coarse) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status == SS_OK && MatProductSymbolic(*coarse) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status == SS_OK && MatProductNumeric(*coarse) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  return status;
}

SsStatus ss_pmg_world_hierarchy_build(SsPmg *pmg, Mat fine_operator) {
  const size_t finest = pmg->hierarchy.level_count - 1U;
  SsStatus status = SS_OK;
  if (pmg->external_operator_mode) {
    for (size_t level = 0U; level <= finest; ++level) {
      PetscInt rows = 0, columns = 0;
      Mat external = pmg->external_operators[level];
      if (external == NULL || MatGetSize(external, &rows, &columns) != PETSC_SUCCESS ||
          rows <= 0 || rows != columns ||
          PetscObjectReference((PetscObject)external) != PETSC_SUCCESS) {
        status = SS_ERROR_TYPE_MISMATCH;
      } else {
        pmg->operators[level] = external;
      }
      status = ss_status_allreduce_max(pmg->communicator, status);
      if (status != SS_OK) {
        return status;
      }
      if (level != 0U) {
        PetscInt coarse_rows = 0;
        status = ss_pmg_transfer_validate(pmg, level - 1U, rows);
        if (status == SS_OK && (MatGetSize(pmg->operators[level - 1U], &coarse_rows,
                                           NULL) != PETSC_SUCCESS ||
                                MatGetSize(pmg->prolongations[level - 1U], NULL,
                                           &columns) != PETSC_SUCCESS ||
                                columns != coarse_rows)) {
          status = SS_ERROR_INTEGRITY;
        }
        status = ss_status_allreduce_max(pmg->communicator, status);
        if (status != SS_OK) {
          return status;
        }
        ++pmg->transfer_setup_count;
      }
    }
    status = ss_status_allreduce_max(
        pmg->communicator,
        pmg->operators[finest] == fine_operator ? SS_OK : SS_ERROR_INTEGRITY);
  } else {
    PetscInt rows = 0, columns = 0;
    if (MatGetSize(fine_operator, &rows, &columns) != PETSC_SUCCESS || rows <= 0 ||
        rows != columns) {
      status = SS_ERROR_TYPE_MISMATCH;
    }
    status = ss_status_allreduce_max(pmg->communicator, status);
    if (status == SS_OK && MatDuplicate(fine_operator, MAT_COPY_VALUES,
                                        &pmg->operators[finest]) != PETSC_SUCCESS) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
    if (status == SS_OK) {
      status = ss_pmg_variable_blocks_copy(fine_operator, pmg->operators[finest]);
    }
    status = ss_status_allreduce_max(pmg->communicator, status);
    for (size_t level = finest; status == SS_OK && level > 0U; --level) {
      if (MatGetSize(pmg->operators[level], &rows, NULL) != PETSC_SUCCESS) {
        status = SS_ERROR_INTERNAL;
      }
      status = ss_status_allreduce_max(pmg->communicator, status);
      if (status == SS_OK) {
        status = ss_pmg_transfer_validate(pmg, level - 1U, rows);
      }
      if (status == SS_OK) {
        status = ss_pmg_galerkin_build(pmg, level);
      }
      status = ss_status_allreduce_max(pmg->communicator, status);
      if (status == SS_OK) {
        ++pmg->galerkin_symbolic_count;
        ++pmg->galerkin_numeric_count;
        ++pmg->transfer_setup_count;
      }
    }
  }
  for (size_t level = 0U; status == SS_OK && level <= finest; ++level) {
    if (MatCreateVecs(pmg->operators[level], &pmg->rhs[level], NULL) != PETSC_SUCCESS) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
    status = ss_status_allreduce_max(pmg->communicator, status);
    Vec *work[] = {&pmg->solution[level], &pmg->residual[level], &pmg->delta[level]};
    for (size_t i = 0U; status == SS_OK && i < sizeof(work) / sizeof(*work); ++i) {
      status = ss_status_allreduce_max(
          pmg->communicator, VecDuplicate(pmg->rhs[level], work[i]) == PETSC_SUCCESS
                                 ? SS_OK
                                 : SS_ERROR_OUT_OF_MEMORY);
    }
  }
  return status;
}

SsStatus ss_pmg_world_hierarchy_refresh(SsPmg *pmg, Mat fine_operator) {
  const size_t finest = pmg->hierarchy.level_count - 1U;
  SsStatus status = SS_OK;
  if (pmg->external_operator_mode) {
    for (size_t level = 0U; level <= finest; ++level) {
      if (pmg->operators[level] != pmg->external_operators[level]) {
        status = SS_ERROR_INTEGRITY;
      }
    }
    if (pmg->operators[finest] != fine_operator) {
      status = SS_ERROR_INTEGRITY;
    }
    return ss_status_allreduce_max(pmg->communicator, status);
  }
  status = ss_status_allreduce_max(pmg->communicator,
                                   MatCopy(fine_operator, pmg->operators[finest],
                                           SAME_NONZERO_PATTERN) == PETSC_SUCCESS
                                       ? SS_OK
                                       : SS_ERROR_INTEGRITY);
  for (size_t level = finest; status == SS_OK && level > 0U; --level) {
    const PetscErrorCode refreshed =
        pmg->ordered_galerkin && pmg->world_size > 1
            ? MatProductNumeric(pmg->operators[level - 1U])
            : MatPtAP(pmg->operators[level], pmg->prolongations[level - 1U],
                      MAT_REUSE_MATRIX, 2.0, &pmg->operators[level - 1U]);
    status = ss_status_allreduce_max(
        pmg->communicator, refreshed == PETSC_SUCCESS ? SS_OK : SS_ERROR_INTEGRITY);
    if (status == SS_OK) {
      ++pmg->galerkin_numeric_count;
    }
  }
  return status;
}

SsStatus ss_pmg_variable_point_blocks_set(SsPmg *pmg, size_t level_index) {
  SsPmgLevel *level = &pmg->levels[level_index];
  const SsPmgNearNullspaceInput *input = &pmg->nullspace_inputs[level_index].input;
  Mat operator_handle = pmg->operators[level_index];
  PetscInt row_start = 0;
  PetscInt row_end = 0;
  PetscInt attached_block_count = 0;
  const PetscInt *attached_block_sizes = NULL;
  PetscInt cursor;
  PetscInt *block_sizes = NULL;
  size_t block_count = 0U;
  size_t node;

  if (input->field != SS_PMG_FIELD_MECHANICS) {
    return SS_OK;
  }
  if (operator_handle == NULL || input->dimension < 2U || input->dimension > 3U ||
      input->owned_node_count > (size_t)PETSC_MAX_INT ||
      MatGetOwnershipRange(operator_handle, &row_start, &row_end) != PETSC_SUCCESS ||
      MatGetVariableBlockSizes(operator_handle, &attached_block_count,
                               &attached_block_sizes) != PETSC_SUCCESS ||
      attached_block_count < 0 ||
      (attached_block_count != 0 && attached_block_sizes == NULL)) {
    return SS_ERROR_INTEGRITY;
  }
  /* Matrix-free high-order operators may attach a stronger nonoverlapping
   * entity partition (several nodal slots per block).  Near-nullspace setup
   * must not silently replace that operator-owned partition with 3-by-3
   * point blocks.  Assembled operators already attach the ordinary nodal
   * partition, so preserving any complete existing partition is also the
   * least surprising behavior there. */
  if (attached_block_count != 0) {
    PetscInt attached_rows = 0;
    PetscInt attached_block;
    for (attached_block = 0; attached_block < attached_block_count; ++attached_block) {
      if (attached_block_sizes[attached_block] <= 0 ||
          attached_rows > PETSC_MAX_INT - attached_block_sizes[attached_block]) {
        return SS_ERROR_INTEGRITY;
      }
      attached_rows += attached_block_sizes[attached_block];
    }
    if (attached_rows != row_end - row_start) {
      return SS_ERROR_INTEGRITY;
    }
    level->variable_point_block_count = (size_t)attached_block_count;
    return SS_OK;
  }
  if (input->owned_node_count != 0U) {
    block_sizes = (PetscInt *)calloc(input->owned_node_count, sizeof(*block_sizes));
    if (block_sizes == NULL) {
      return SS_ERROR_OUT_OF_MEMORY;
    }
  }
  cursor = row_start;
  for (node = 0U; node < input->owned_node_count; ++node) {
    PetscInt block_size = 0;
    uint32_t component;
    for (component = 0U; component < input->dimension; ++component) {
      const PetscInt global_dof =
          input->global_dof_indices[node * (size_t)input->dimension + component];
      if (global_dof < 0) {
        continue;
      }
      if (global_dof != cursor + block_size || global_dof < row_start ||
          global_dof >= row_end) {
        free(block_sizes);
        return SS_ERROR_INTEGRITY;
      }
      ++block_size;
    }
    if (block_size <= 0) {
      free(block_sizes);
      return SS_ERROR_INTEGRITY;
    }
    block_sizes[block_count++] = block_size;
    cursor += block_size;
  }
  if (cursor != row_end ||
      MatSetVariableBlockSizes(operator_handle, (PetscInt)block_count, block_sizes) !=
          PETSC_SUCCESS) {
    free(block_sizes);
    return SS_ERROR_INTEGRITY;
  }
  free(block_sizes);
  level->variable_point_block_count = block_count;
  return SS_OK;
}

static SsStatus ss_pmg_active_variable_point_blocks_set(SsPmg *pmg,
                                                        size_t level_index) {
  SsPmgLevel *level = &pmg->levels[level_index];
  Mat world = pmg->operators[level_index];
  const PetscInt *blocks = NULL;
  PetscInt count = 0, first = 0, end = 0, active_first = 0, active_end = 0;
  PetscInt active_count = 0;
  PetscInt *leaf_ends = NULL, *root_ends = NULL;
  int has_blocks = 0, global_has_blocks = 0;
  SsStatus status = SS_OK;
  if (MatGetVariableBlockSizes(world, &count, &blocks) != PETSC_SUCCESS ||
      MatGetOwnershipRange(world, &first, &end) != PETSC_SUCCESS ||
      (level->local_active &&
       MatGetOwnershipRange(level->active_operator, &active_first, &active_end) !=
           PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  has_blocks = count > 0;
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  if (MPI_Allreduce(&has_blocks, &global_has_blocks, 1, MPI_INT, MPI_MAX,
                    pmg->communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  level->variable_point_block_count = 0U;
  if (!global_has_blocks) {
    return SS_OK;
  }
  if (end > first) {
    leaf_ends = calloc((size_t)(end - first), sizeof(*leaf_ends));
    if (leaf_ends == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  if (active_end > active_first) {
    root_ends = calloc((size_t)(active_end - active_first), sizeof(*root_ends));
    if (root_ends == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  PetscInt row = first;
  for (PetscInt block = 0; status == SS_OK && block < count; ++block) {
    if (blocks == NULL || blocks[block] <= 0 || blocks[block] > end - row) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    const PetscInt stop = row + blocks[block];
    /* Each resident row carries its block's end. When an active-rank boundary
     * cuts the block, each destination retains the intersection it owns. */
    for (; row < stop; ++row) {
      leaf_ends[row - first] = stop;
    }
  }
  if (status == SS_OK && row != end) {
    status = SS_ERROR_INTEGRITY;
  }
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status == SS_OK && PetscSFReduceBegin(level->row_bridge, MPIU_INT, leaf_ends,
                                            root_ends, MPI_REPLACE) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(pmg->communicator, status);
  if (status == SS_OK && PetscSFReduceEnd(level->row_bridge, MPIU_INT, leaf_ends,
                                          root_ends, MPI_REPLACE) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  for (row = active_first; status == SS_OK && row < active_end;) {
    const PetscInt stop = PetscMin(active_end, root_ends[row - active_first]);
    if (stop <= row) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    root_ends[active_count++] = stop - row;
    row = stop;
  }
  if (status == SS_OK && level->local_active &&
      MatSetVariableBlockSizes(level->active_operator, active_count, root_ends) !=
          PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  level->variable_point_block_count = (size_t)active_count;
  free(leaf_ends);
  free(root_ends);
  return ss_status_allreduce_max(pmg->communicator, status);
}

SsStatus ss_pmg_active_levels_build(SsPmg *pmg) {
  size_t level;
  for (level = 0U; level < pmg->hierarchy.level_count; ++level) {
    SsStatus status = ss_pmg_level_communicator_create(pmg, level);
    status = ss_status_allreduce_max(pmg->communicator, status);
    if (status == SS_OK) {
      status = ss_pmg_active_matrix_update(pmg, level, pmg->operators[level], 0);
    }
    status = ss_status_allreduce_max(pmg->communicator, status);
    if (status == SS_OK && !pmg->levels[level].identity_world_layout) {
      status = ss_pmg_row_bridge_create(pmg, level);
      if (status == SS_OK) {
        status = ss_pmg_active_variable_point_blocks_set(pmg, level);
      }
    }
    if (status == SS_OK) {
      status = ss_pmg_active_near_nullspace_create(pmg, level);
    }
    if (status == SS_OK) {
      SsPmgLevel *configured = &pmg->levels[level];
      const int local_has_blocks = configured->variable_point_block_count != 0U;
      if (MPI_Allreduce(&local_has_blocks, &configured->has_variable_point_blocks, 1,
                        MPI_INT, MPI_MAX, pmg->communicator) != MPI_SUCCESS) {
        status = SS_ERROR_INTERNAL;
      }
    }
    if (status == SS_OK) {
      status = ss_pmg_level_ksp_configure(pmg, level);
    }
    status = ss_status_allreduce_max(pmg->communicator, status);
    if (status != SS_OK) {
      return status;
    }
  }
  return SS_OK;
}

SsStatus ss_pmg_active_levels_refresh(SsPmg *pmg) {
  size_t level;
  for (level = 0U; level < pmg->hierarchy.level_count; ++level) {
    SsStatus status = ss_pmg_active_matrix_update(pmg, level, pmg->operators[level], 1);
    status = ss_status_allreduce_max(pmg->communicator, status);
    if (status == SS_OK) {
      status = ss_pmg_level_ksp_refresh(pmg, level);
    }
    status = ss_status_allreduce_max(pmg->communicator, status);
    if (status != SS_OK) {
      return status;
    }
  }
  return SS_OK;
}
