/* PMG hierarchy, rank and level-solver policy validation.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "pmg_internal.h"
#include "slope_internal/core/util.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int ss_pmg_valid_prefix(const char *prefix) {
  const size_t length = ss_bounded_length(prefix, SS_PMG_PREFIX_CAPACITY);
  size_t index;
  if (length == 0U || length == SS_PMG_PREFIX_CAPACITY || prefix[length - 1U] != '_') {
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

SsStatus ss_pmg_hierarchy_resolve(uint32_t fine_order, SsPmgHierarchy *hierarchy) {
  if (hierarchy == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(hierarchy, 0, sizeof(*hierarchy));
  hierarchy->fine_order = fine_order;
  switch (fine_order) {
  case 1U:
    hierarchy->level_count = 1U;
    hierarchy->orders[0] = 1U;
    return SS_OK;
  case 2U:
    hierarchy->level_count = 2U;
    hierarchy->orders[0] = 1U;
    hierarchy->orders[1] = 2U;
    return SS_OK;
  case 4U:
    hierarchy->level_count = 3U;
    hierarchy->orders[0] = 1U;
    hierarchy->orders[1] = 2U;
    hierarchy->orders[2] = 4U;
    return SS_OK;
  case 8U:
    hierarchy->level_count = 4U;
    hierarchy->orders[0] = 1U;
    hierarchy->orders[1] = 2U;
    hierarchy->orders[2] = 4U;
    hierarchy->orders[3] = 8U;
    return SS_OK;
  default:
    return SS_ERROR_NOT_FOUND;
  }
}

SsStatus ss_pmg_hierarchy_validate(uint32_t fine_order, size_t level_count,
                                   const uint32_t *orders, SsPmgHierarchy *hierarchy) {
  SsPmgHierarchy expected = {0};
  size_t level;
  SsStatus status;
  if (orders == NULL || hierarchy == NULL || level_count == 0U ||
      level_count > SS_PMG_MAX_LEVELS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (level = 1U; level < level_count; ++level) {
    if (orders[level] == orders[level - 1U]) {
      return SS_ERROR_DUPLICATE;
    }
  }
  status = ss_pmg_hierarchy_resolve(fine_order, &expected);
  if (status != SS_OK) {
    return status;
  }
  if (expected.level_count != level_count) {
    return SS_ERROR_INTEGRITY;
  }
  for (level = 0U; level < level_count; ++level) {
    if (orders[level] != expected.orders[level]) {
      return SS_ERROR_INTEGRITY;
    }
  }
  *hierarchy = expected;
  return SS_OK;
}

SsStatus ss_pmg_active_ranks_resolve(PetscMPIInt world_size, PetscMPIInt active_count,
                                     PetscMPIInt *active_ranks, size_t capacity) {
  PetscMPIInt index;
  if (world_size <= 0 || active_count <= 0 || active_count > world_size ||
      active_ranks == NULL || capacity < (size_t)active_count) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (index = 0; index < active_count; ++index) {
    const int64_t numerator = (int64_t)index * (int64_t)world_size;
    active_ranks[index] = (PetscMPIInt)(numerator / (int64_t)active_count);
  }
  return SS_OK;
}

SsStatus ss_pmg_active_ranks_validate(PetscMPIInt world_size, size_t active_count,
                                      const PetscMPIInt *active_ranks) {
  size_t index;
  if (world_size <= 0 || active_count == 0U || active_count > (size_t)world_size ||
      active_ranks == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (index = 0U; index < active_count; ++index) {
    size_t previous;
    if (active_ranks[index] < 0 || active_ranks[index] >= world_size) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
    for (previous = 0U; previous < index; ++previous) {
      if (active_ranks[previous] == active_ranks[index]) {
        return SS_ERROR_DUPLICATE;
      }
    }
  }
  return SS_OK;
}

static int ss_pmg_base_method(SsPmgLevelMethod method) {
  return method == SS_PMG_LEVEL_FGMRES_JACOBI || method == SS_PMG_LEVEL_FGMRES_GAMG ||
         method == SS_PMG_LEVEL_PREONLY_LU ||
         method == SS_PMG_LEVEL_PREONLY_REDUNDANT_LU ||
         method == SS_PMG_LEVEL_PREONLY_SUPERLU_DIST;
}

int ss_pmg_smoother_method(SsPmgLevelMethod method) {
  return method == SS_PMG_LEVEL_RICHARDSON_NONE ||
         method == SS_PMG_LEVEL_RICHARDSON_JACOBI ||
         method == SS_PMG_LEVEL_RICHARDSON_VPBJACOBI ||
         method == SS_PMG_LEVEL_FGMRES_JACOBI ||
         method == SS_PMG_LEVEL_FGMRES_VPBJACOBI ||
         method == SS_PMG_LEVEL_CHEBYSHEV_JACOBI ||
         method == SS_PMG_LEVEL_CHEBYSHEV_POINT_JACOBI;
}

SsStatus ss_pmg_policy_copy(SsPmg *pmg, size_t level, const SsPmgLevelPolicy *source) {
  SsPmgLevel *target = &pmg->levels[level];
  SsStatus status;
  if (source == NULL || !ss_pmg_valid_prefix(source->options_prefix) ||
      source->active_rank_count == 0U ||
      source->active_rank_count > (size_t)pmg->world_size ||
      source->maximum_iterations <= 0 || !ss_is_finite(source->rtol) ||
      !ss_is_finite(source->atol) || source->rtol <= 0.0 || source->atol < 0.0 ||
      (source->method == SS_PMG_LEVEL_PREONLY_REDUNDANT_LU &&
       (source->redundant_solve_count <= 0 ||
        (size_t)source->redundant_solve_count > source->active_rank_count)) ||
      (source->method != SS_PMG_LEVEL_PREONLY_REDUNDANT_LU &&
       source->redundant_solve_count != 0) ||
      (level == 0U && !ss_pmg_base_method(source->method)) ||
      (level != 0U &&
       (!ss_pmg_smoother_method(source->method) || source->pre_smooth_steps <= 0 ||
        source->post_smooth_steps <= 0 || !ss_is_finite(source->richardson_scale) ||
        source->richardson_scale <= 0.0))) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  target->active_ranks =
      (PetscMPIInt *)malloc(source->active_rank_count * sizeof(*target->active_ranks));
  if (target->active_ranks == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  target->policy = *source;
  target->policy.active_ranks = target->active_ranks;
  if (source->active_ranks == NULL) {
    status = ss_pmg_active_ranks_resolve(
        pmg->world_size, (PetscMPIInt)source->active_rank_count, target->active_ranks,
        source->active_rank_count);
  } else {
    status = ss_pmg_active_ranks_validate(pmg->world_size, source->active_rank_count,
                                          source->active_ranks);
    if (status == SS_OK) {
      memcpy(target->active_ranks, source->active_ranks,
             source->active_rank_count * sizeof(*target->active_ranks));
    }
  }
  if (status != SS_OK) {
    return status;
  }
  target->identity_world_layout = source->active_rank_count == (size_t)pmg->world_size;
  if (target->identity_world_layout != 0) {
    size_t index;
    for (index = 0U; index < source->active_rank_count; ++index) {
      if (target->active_ranks[index] != (PetscMPIInt)index) {
        target->identity_world_layout = 0;
        break;
      }
    }
  }
  return SS_OK;
}
