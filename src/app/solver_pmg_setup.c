/* Field constraints, near-nullspace and PMG hierarchy setup.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/core/collective.h"
#include "slope_internal/core/sort.h"
#include "solver_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t ss_solver_pmg_entity_slots(uint32_t entity_dimension, uint32_t order) {
  if (entity_dimension == 0U) {
    return 1U;
  }
  if (entity_dimension == 1U) {
    return (size_t)order - 1U;
  }
  if (entity_dimension == 2U) {
    return ((size_t)order - 1U) * ((size_t)order - 2U) / 2U;
  }
  return 0U;
}

static SsStatus ss_solver_pmg_constraints(const SsSolverGraph *graph, uint32_t order,
                                          const SsStrongConstraint *fine_constraints,
                                          size_t fine_constraint_count,
                                          SsStrongConstraint **constraints,
                                          size_t *constraint_count) {
  SsStrongConstraint *created = NULL;
  size_t count = 0U;
  size_t index;
  if (graph == NULL || constraints == NULL || constraint_count == NULL ||
      (fine_constraint_count != 0U && fine_constraints == NULL)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *constraints = NULL;
  *constraint_count = 0U;
  created = (SsStrongConstraint *)calloc(
      fine_constraint_count == 0U ? 1U : fine_constraint_count, sizeof(*created));
  if (created == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (index = 0U; index < fine_constraint_count; ++index) {
    const SsStrongConstraint *source = &fine_constraints[index];
    uint32_t entity_dimension = 0U;
    if (ss_mesh_entity_dimension(graph->mesh, source->stable_entity_id,
                                 &entity_dimension) != SS_OK) {
      free(created);
      return SS_ERROR_NOT_FOUND;
    }
    if ((size_t)source->entity_slot >=
        ss_solver_pmg_entity_slots(entity_dimension, order)) {
      continue;
    }
    created[count] = *source;
    created[count].value = 0.0;
    ++count;
  }
  *constraints = created;
  *constraint_count = count;
  return SS_OK;
}

typedef struct {
  PetscInt key;
  SsEntityId cell;
  PetscReal coordinates[SS_FE_MAX_DIMENSION];
  PetscInt global_dofs[SS_FE_MAX_DIMENSION];
} SsSolverPmgNode;

static int ss_solver_pmg_node_compare(const void *left, const void *right) {
  const SsSolverPmgNode *a = (const SsSolverPmgNode *)left;
  const SsSolverPmgNode *b = (const SsSolverPmgNode *)right;
  if (a->key != b->key) {
    return a->key < b->key ? -1 : 1;
  }
  return a->cell < b->cell ? -1 : (a->cell > b->cell ? 1 : 0);
}

static SsStatus ss_solver_pmg_nullspace_input(
    const SsFieldSpace *space, uint32_t dimension, uint32_t components,
    SsPmgFieldKind field, int hydraulic_pure_neumann, SsPmgNearNullspaceInput *input,
    PetscReal **coordinates, PetscInt **global_dofs) {
  const SsCellPlan *packs = NULL;
  SsSolverPmgNode *nodes = NULL;
  size_t pack_count = 0U;
  size_t candidate_capacity;
  size_t candidate_count = 0U;
  size_t unique_count = 0U;
  size_t pack_index;
  SsStatus status;
  if (space == NULL || input == NULL || coordinates == NULL || global_dofs == NULL ||
      (dimension != 2U && dimension != 3U) || components == 0U ||
      components > SS_FE_MAX_DIMENSION ||
      (field != SS_PMG_FIELD_MECHANICS && field != SS_PMG_FIELD_HYDRAULIC) ||
      (field == SS_PMG_FIELD_MECHANICS && components != dimension) ||
      (field == SS_PMG_FIELD_HYDRAULIC && components != 1U)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(input, 0, sizeof(*input));
  *coordinates = NULL;
  *global_dofs = NULL;
  status = ss_field_space_incident_cell_plan_view(space, &packs, &pack_count);
  if (status != SS_OK) {
    return status;
  }
  candidate_capacity = 0U;
  for (pack_index = 0U; pack_index < pack_count; ++pack_index) {
    if (packs[pack_index].field_node_count > SIZE_MAX - candidate_capacity) {
      return SS_ERROR_OVERFLOW;
    }
    candidate_capacity += packs[pack_index].field_node_count;
  }
  nodes = (SsSolverPmgNode *)calloc(candidate_capacity == 0U ? 1U : candidate_capacity,
                                    sizeof(*nodes));
  if (nodes == NULL) {
    free(nodes);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (pack_index = 0U; status == SS_OK && pack_index < pack_count; ++pack_index) {
    const SsCellPlan *pack = &packs[pack_index];
    size_t node;
    size_t closure_count;
    if (pack->dimension != dimension || pack->components != components ||
        pack->field_node_count > SS_FE_MAX_NODES ||
        ss_checked_mul_size(pack->field_node_count, components, &closure_count) !=
            SS_OK ||
        closure_count != pack->closure_count) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    for (node = 0U; node < pack->field_node_count; ++node) {
      SsSolverPmgNode candidate;
      PetscInt key = PETSC_MAX_INT;
      uint32_t coordinate;
      uint32_t component;
      memset(&candidate, 0, sizeof(candidate));
      for (component = 0U; component < SS_FE_MAX_DIMENSION; ++component) {
        candidate.global_dofs[component] = -1;
      }
      for (coordinate = 0U; coordinate < dimension; ++coordinate) {
        const double value =
            pack->field_node_coordinates[node * (size_t)dimension + coordinate];
        if (!isfinite(value)) {
          status = SS_ERROR_INTEGRITY;
          break;
        }
        candidate.coordinates[coordinate] = (PetscReal)value;
      }
      for (component = 0U; status == SS_OK && component < components; ++component) {
        const size_t closure = node * (size_t)components + component;
        const PetscInt global_index = pack->global_indices[closure];
        if (pack->field_components[closure] != component) {
          status = SS_ERROR_INTEGRITY;
          break;
        }
        if (pack->owned[closure] != 0U && pack->constrained[closure] == 0U) {
          if (global_index < 0) {
            status = SS_ERROR_INTEGRITY;
            break;
          }
          candidate.global_dofs[component] = global_index;
          if (global_index < key) {
            key = global_index;
          }
        }
      }
      if (status != SS_OK) {
        break;
      }
      if (key == PETSC_MAX_INT) {
        continue;
      }
      if (candidate_count >= candidate_capacity) {
        status = SS_ERROR_OVERFLOW;
        break;
      }
      candidate.key = key;
      candidate.cell = pack->stable_cell_id;
      nodes[candidate_count++] = candidate;
    }
  }
  if (status == SS_OK) {
    ss_sort(nodes, candidate_count, sizeof(*nodes), ss_solver_pmg_node_compare);
    for (pack_index = 0U; pack_index < candidate_count; ++pack_index) {
      if (unique_count != 0U && nodes[pack_index].key == nodes[unique_count - 1U].key) {
        uint32_t coordinate;
        uint32_t component;
        for (coordinate = 0U; coordinate < dimension; ++coordinate) {
          const PetscReal a = nodes[pack_index].coordinates[coordinate];
          const PetscReal b = nodes[unique_count - 1U].coordinates[coordinate];
          const PetscReal scale =
              PetscMax(1.0, PetscMax(PetscAbsReal(a), PetscAbsReal(b)));
          /* The same shared point is evaluated in two different cell reference
           * frames. Curved interpolation can round differently in each frame.
           * Keep the value from the lowest stable incident cell. */
          if (PetscAbsReal(a - b) > 128.0 * PETSC_MACHINE_EPSILON * scale) {
            status = SS_ERROR_INTEGRITY;
          }
        }
        for (component = 0U; component < components; ++component) {
          if (nodes[pack_index].global_dofs[component] !=
              nodes[unique_count - 1U].global_dofs[component]) {
            status = SS_ERROR_INTEGRITY;
          }
        }
        if (status != SS_OK) {
          break;
        }
        continue;
      }
      nodes[unique_count++] = nodes[pack_index];
    }
  }
  if (status == SS_OK && unique_count != 0U) {
    size_t coordinate_bytes;
    size_t dof_bytes;
    if (ss_checked_product_size(
            (size_t[]){unique_count, dimension, sizeof(**coordinates)}, 3U,
            &coordinate_bytes) != SS_OK ||
        ss_checked_product_size(
            (size_t[]){unique_count, components, sizeof(**global_dofs)}, 3U,
            &dof_bytes) != SS_OK) {
      status = SS_ERROR_OVERFLOW;
    } else {
      *coordinates = (PetscReal *)calloc(1U, coordinate_bytes);
      *global_dofs = (PetscInt *)calloc(1U, dof_bytes);
      if (*coordinates == NULL || *global_dofs == NULL) {
        status = SS_ERROR_OUT_OF_MEMORY;
      }
    }
  }
  for (pack_index = 0U; status == SS_OK && pack_index < unique_count; ++pack_index) {
    uint32_t coordinate;
    uint32_t component;
    for (coordinate = 0U; coordinate < dimension; ++coordinate) {
      (*coordinates)[pack_index * (size_t)dimension + coordinate] =
          nodes[pack_index].coordinates[coordinate];
    }
    for (component = 0U; component < components; ++component) {
      (*global_dofs)[pack_index * (size_t)components + component] =
          nodes[pack_index].global_dofs[component];
    }
  }
  free(nodes);
  if (status != SS_OK) {
    free(*global_dofs);
    free(*coordinates);
    *global_dofs = NULL;
    *coordinates = NULL;
    return status;
  }
  *input = (SsPmgNearNullspaceInput){
      .field = field,
      .dimension = dimension,
      .hydraulic_pure_neumann = hydraulic_pure_neumann,
      .owned_node_count = unique_count,
      .coordinates = *coordinates,
      .global_dof_indices = *global_dofs,
      .dependence_tolerance = 1.0e-12,
  };
  return SS_OK;
}

SsStatus ss_solver_pmg_create(const SsSolverGraph *graph, SsFieldSpace *fine_space,
                              uint32_t fine_order, uint32_t components, int hydraulic,
                              const SsStrongConstraint *fine_constraints,
                              size_t fine_constraint_count, const char *options_stem,
                              SsFieldSpace **retained_coarse_spaces, Mat *injections,
                              SsPmg **pmg, SsErrorRecord *error) {
  SsPmgHierarchy hierarchy = {0};
  SsFieldSpace *spaces[SS_PMG_MAX_LEVELS] = {NULL, NULL, NULL};
  SsStrongConstraint *constraints[SS_PMG_MAX_LEVELS - 1U] = {NULL, NULL};
  Mat prolongations[SS_PMG_MAX_LEVELS - 1U] = {NULL, NULL};
  SsPmgLevelPolicy levels[SS_PMG_MAX_LEVELS];
  SsPmgNearNullspaceInput near_nullspaces[SS_PMG_MAX_LEVELS];
  PetscReal *near_coordinates[SS_PMG_MAX_LEVELS] = {NULL, NULL, NULL};
  PetscInt *near_global_dofs[SS_PMG_MAX_LEVELS] = {NULL, NULL, NULL};
  SsPmgSpecification specification = {0};
  size_t level;
  SsStatus status;
  const char *create_stage = "hierarchy";
  if (graph == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = SS_OK;
  if (fine_space == NULL || options_stem == NULL || pmg == NULL || error == NULL ||
      graph->size <= 0 || components == 0U ||
      (injections != NULL && retained_coarse_spaces == NULL) ||
      (fine_constraint_count != 0U && fine_constraints == NULL)) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  *pmg = NULL;
  if (retained_coarse_spaces != NULL) {
    for (level = 0U; level < SS_PMG_MAX_LEVELS - 1U; ++level) {
      retained_coarse_spaces[level] = NULL;
    }
  }
  if (injections != NULL) {
    for (level = 0U; level < SS_PMG_MAX_LEVELS - 1U; ++level) {
      injections[level] = NULL;
    }
  }
  status = ss_pmg_hierarchy_resolve(fine_order, &hierarchy);
  if (status == SS_OK && graph->policy.pmg_level_count != hierarchy.level_count) {
    status = SS_ERROR_SCHEMA;
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  memset(levels, 0, sizeof(levels));
  memset(near_nullspaces, 0, sizeof(near_nullspaces));
  spaces[hierarchy.level_count - 1U] = fine_space;
  for (level = 0U; level < hierarchy.level_count; ++level) {
    const int written =
        snprintf(levels[level].options_prefix, sizeof(levels[level].options_prefix),
                 "%s_p%u_", options_stem, hierarchy.orders[level]);
    if (written < 0 || (size_t)written >= sizeof(levels[level].options_prefix)) {
      status = SS_ERROR_LIMIT;
      break;
    }
    if (graph->policy.pmg_level_orders[level] != hierarchy.orders[level] ||
        ss_pmg_active_ranks_validate((PetscMPIInt)graph->size,
                                     graph->policy.pmg_active_rank_counts[level],
                                     graph->policy.pmg_active_ranks[level]) != SS_OK) {
      status = SS_ERROR_SCHEMA;
      break;
    }
    levels[level].method = level == 0U ? graph->policy.pmg_coarse_method
                                       : graph->policy.pmg_smoother_method;
    levels[level].active_rank_count = graph->policy.pmg_active_rank_counts[level];
    levels[level].active_ranks = graph->policy.pmg_active_ranks[level];
    levels[level].pre_smooth_steps =
        level == 0U ? 0 : graph->policy.pmg_pre_smoothing_steps;
    levels[level].post_smooth_steps =
        level == 0U ? 0 : graph->policy.pmg_post_smoothing_steps;
    /*
     * The P1 solve is the base action of one PMG application, not a second
     * independently converged linear solve.  Driving its nested FGMRES/GAMG
     * solve to the outer iteration limit repeats tens of GAMG cycles for
     * every outer preconditioner application.  Keep the base action fixed
     * work; flexible outer GMRES and its independent true-residual check own
     * convergence of the original system. Hydraulics has its own resolved
     * cap: the inexpensive mechanics base action can stall a nonsymmetric
     * seepage solve at its much tighter outer tolerance.
     */
    levels[level].maximum_iterations =
        level == 0U ? (hydraulic != 0 ? graph->policy.pmg_hydraulic_base_iterations
                                      : graph->policy.pmg_base_iterations)
                    : (graph->policy.pmg_pre_smoothing_steps >
                               graph->policy.pmg_post_smoothing_steps
                           ? graph->policy.pmg_pre_smoothing_steps
                           : graph->policy.pmg_post_smoothing_steps);
    levels[level].richardson_scale = 1.0;
    levels[level].rtol = 1.0e-3;
    levels[level].atol = 1.0e-50;
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  for (level = 0U; level + 1U < hierarchy.level_count; ++level) {
    SsFieldSpec field_specification;
    size_t coarse_constraint_count = 0U;
    create_stage = "coarse_constraints";
    status = ss_solver_pmg_constraints(graph, hierarchy.orders[level], fine_constraints,
                                       fine_constraint_count, &constraints[level],
                                       &coarse_constraint_count);
    status = ss_status_allreduce_max(graph->communicator, status);
    if (status != SS_OK) {
      goto cleanup;
    }
    field_specification = (SsFieldSpec){
        hierarchy.orders[level],
        components,
        coarse_constraint_count,
        constraints[level],
    };
    create_stage = "coarse_space";
    status = ss_mesh_field_space_create_local(graph->mesh, &field_specification,
                                              &spaces[level]);
    status = ss_status_allreduce_max(graph->communicator, status);
    if (status != SS_OK) {
      goto cleanup;
    }
  }
  if (injections != NULL) {
    for (level = 0U; level + 1U < hierarchy.level_count; ++level) {
      create_stage = "injection";
      status = ss_field_space_injection_create(spaces[level], spaces[level + 1U],
                                               &injections[level]);
      status = ss_status_allreduce_max(graph->communicator, status);
      if (status != SS_OK) {
        goto cleanup;
      }
    }
  }
  if (graph->policy.pmg_coarse_method == SS_PMG_LEVEL_PREONLY_LU) {
    SsFieldSummary coarse_summary = {0};
    const size_t active_count = levels[0].active_rank_count;
    const size_t group_size = graph->policy.pmg_coarse_redundant_group_size;
#if defined(PETSC_HAVE_SUPERLU_DIST)
    const int use_gamg_fallback = 0;
#else
    const int use_gamg_fallback =
        graph->policy.settings.pmg_coarse_lu_fallback_to_gamg == PETSC_TRUE &&
        active_count > 1U && group_size == 0U;
#endif
    create_stage = "coarse_policy";
    status = ss_field_space_summary(spaces[0], &coarse_summary);
    status = ss_status_allreduce_max(graph->communicator, status);
    if (status != SS_OK) {
      goto cleanup;
    }
    if (!use_gamg_fallback && graph->policy.pmg_coarse_direct_maximum_dofs > 0U &&
        coarse_summary.global_free_dof_count <=
            graph->policy.pmg_coarse_direct_maximum_dofs) {
      if (active_count == 1U) {
        levels[0].method = SS_PMG_LEVEL_PREONLY_LU;
        levels[0].maximum_iterations = 1;
      } else if (group_size == 1U) {
        if (active_count > (size_t)PETSC_MAX_INT) {
          status = SS_ERROR_LIMIT;
          goto cleanup;
        }
        levels[0].method = SS_PMG_LEVEL_PREONLY_REDUNDANT_LU;
        levels[0].redundant_solve_count = (PetscInt)active_count;
        levels[0].maximum_iterations = 1;
      } else if (group_size != 0U) {
        status = SS_ERROR_SCHEMA;
        goto cleanup;
      } else {
        levels[0].method = SS_PMG_LEVEL_PREONLY_SUPERLU_DIST;
        levels[0].maximum_iterations = 1;
      }
    } else {
      levels[0].method = SS_PMG_LEVEL_FGMRES_GAMG;
    }
  }
  for (level = 0U; level + 1U < hierarchy.level_count; ++level) {
    create_stage = "transfer";
    status = ss_field_space_transfer_create(spaces[level], spaces[level + 1U],
                                            &prolongations[level]);
    status = ss_status_allreduce_max(graph->communicator, status);
    if (status != SS_OK) {
      goto cleanup;
    }
  }
  for (level = 0U; status == SS_OK && level < hierarchy.level_count; ++level) {
    const SsPmgFieldKind field = components == graph->model.dimension
                                     ? SS_PMG_FIELD_MECHANICS
                                     : SS_PMG_FIELD_HYDRAULIC;
    create_stage = "near_nullspace";
    SsFieldSummary field_summary = {0};
    status = ss_field_space_summary(spaces[level], &field_summary);
    if (status == SS_OK) {
      status = ss_solver_pmg_nullspace_input(
          spaces[level], graph->model.dimension, components, field,
          field == SS_PMG_FIELD_HYDRAULIC && field_summary.constrained_dof_count == 0U,
          &near_nullspaces[level], &near_coordinates[level], &near_global_dofs[level]);
    }
    status = ss_status_allreduce_max(graph->communicator, status);
  }
  if (status != SS_OK) {
    goto cleanup;
  }
  specification.fine_order = hierarchy.fine_order;
  specification.level_count = hierarchy.level_count;
  specification.orders = hierarchy.orders;
  specification.prolongations = hierarchy.level_count == 1U ? NULL : prolongations;
  specification.level_policies = levels;
  specification.near_nullspaces = near_nullspaces;
  specification.transfer_generation = 1U;
  specification.numeric_lag_interval = 1U;
  specification.ordered_galerkin =
      ss_solver_assembly_determinism(&graph->policy) == SS_ASSEMBLY_QUALIFICATION;
  specification.fgmres_full_iterate_post =
      hydraulic == 0 &&
      graph->policy.settings.pmg_fgmres_full_iterate_post == PETSC_TRUE;
  create_stage = "memory_cap";
  status = ss_solver_cap_bytes(
      hydraulic != 0 ? graph->policy.settings.resources_hydraulic_pmg_memory_cap_mib
                     : graph->policy.settings.resources_mechanics_pmg_memory_cap_mib,
      &specification.memory_cap_bytes);
  if (status == SS_OK && hydraulic == 0 &&
      levels[0].method == SS_PMG_LEVEL_PREONLY_SUPERLU_DIST &&
      graph->policy.settings.pmg_coarse_factorization == 1) {
    create_stage = "superlu_dist_policy";
    status = ss_solver_superlu_dist_settings();
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  create_stage = "core";
  status = ss_pmg_create(graph->communicator, &specification, pmg);

cleanup:
  for (level = 0U; level < SS_PMG_MAX_LEVELS; ++level) {
    free(near_global_dofs[level]);
    free(near_coordinates[level]);
  }
  for (level = 0U; level + 1U < hierarchy.level_count; ++level) {
    (void)MatDestroy(&prolongations[level]);
    if (status == SS_OK && retained_coarse_spaces != NULL) {
      retained_coarse_spaces[level] = spaces[level];
      spaces[level] = NULL;
    }
    (void)ss_field_space_destroy(&spaces[level]);
    free(constraints[level]);
  }
  if (status != SS_OK && injections != NULL) {
    for (level = 0U; level < SS_PMG_MAX_LEVELS - 1U; ++level) {
      (void)MatDestroy(&injections[level]);
    }
  }
  if (status != SS_OK && error->valid == 0) {
    char message[SS_ERROR_MESSAGE_CAPACITY];
    const int written =
        snprintf(message, sizeof(message), "PMG creation failed during %s (status=%d)",
                 create_stage, (int)status);
    ss_solver_error(error, status, "PMG_CREATE",
                    written >= 0 && (size_t)written < sizeof(message)
                        ? message
                        : "PMG creation failed",
                    3U);
  }
  return status;
}

SsStatus ss_solver_mechanics_matrix_free_hierarchy_create(SsSolverMechanics *mechanics,
                                                          SsErrorRecord *error) {
  SsPmgHierarchy hierarchy = {0};
  SsStatus status;
  size_t level;
  MPI_Comm comm;
  if (mechanics == NULL || mechanics->graph == NULL || mechanics->pmg == NULL ||
      mechanics->elastic_reference == NULL || error == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  comm = mechanics->graph->communicator;
  status =
      ss_pmg_hierarchy_resolve(mechanics->graph->model.mechanics_order, &hierarchy);
  if (status != SS_OK) {
    return status;
  }
  mechanics->matrix_free_level_count = hierarchy.level_count;
  mechanics->matrix_free_level_operators[hierarchy.level_count - 1U] =
      mechanics->elastic_reference;
  for (level = 0U; level + 1U < hierarchy.level_count; ++level) {
    SsOperatorSpec operator_spec = mechanics->operator_spec;
    SsOperatorState state = {0};
    status =
        level == 0U
            ? ss_assembly_plan_create(mechanics->graph->communicator,
                                      mechanics->matrix_free_spaces[level],
                                      &mechanics->matrix_free_plans[level], error)
            : ss_assembly_plan_create_matrix_free(
                  mechanics->graph->communicator, mechanics->matrix_free_spaces[level],
                  &mechanics->matrix_free_plans[level], error);
    status = ss_status_allreduce_max(comm, status);
    if (status == SS_OK && level == 0U) {
      status = ss_assembly_plan_create_matrix(
          mechanics->matrix_free_plans[level],
          &mechanics->matrix_free_level_operators[level]);
      status = ss_status_allreduce_max(comm, status);
      if (status == SS_OK) {
        status = ss_operator_matrix_free_project_p1_jacobian(
            mechanics->operator_handle, mechanics->matrix_free_plans[level],
            mechanics->matrix_free_level_operators[level], error);
      }
    } else if (status == SS_OK) {
      operator_spec.matrix_free = 1;
      operator_spec.matrix_free_block_nodes = 0U;
      operator_spec.matrix_free_batch_cells = 0U;
      status = ss_operator_create(mechanics->matrix_free_plans[level], &operator_spec,
                                  &mechanics->matrix_free_operators[level], error);
      status = ss_status_allreduce_max(comm, status);
      if (status == SS_OK && mechanics->matrix_free_galerkin == 0) {
        status = ss_assembly_plan_create_vector(mechanics->matrix_free_plans[level],
                                                &mechanics->matrix_free_states[level]);
      }
      if (status == SS_OK && mechanics->matrix_free_galerkin == 0 &&
          VecSet(mechanics->matrix_free_states[level], 0.0) != PETSC_SUCCESS) {
        status = SS_ERROR_INTERNAL;
      }
      status = ss_status_allreduce_max(comm, status);
      state.field = mechanics->matrix_free_states[level];
      state.reduction_factor = 1.0;
      state.external_load_factor = 0.0;
      if (status == SS_OK) {
        status = ss_operator_matrix_free_create(
            mechanics->matrix_free_operators[level],
            &mechanics->matrix_free_diagonal_operators[level], error);
        status = ss_status_allreduce_max(comm, status);
        if (status == SS_OK && mechanics->matrix_free_galerkin == 0) {
          status = ss_operator_matrix_free_linearize(
              mechanics->matrix_free_operators[level], &state, 1.0, error);
        }
        if (status == SS_OK && mechanics->matrix_free_galerkin != 0) {
          status = ss_operator_matrix_free_project_coarse(
              mechanics->operator_handle, mechanics->matrix_free_operators[level],
              error);
        }
      }
    }
    status = ss_status_allreduce_max(comm, status);
    if (status != SS_OK) {
      return status;
    }
  }
  /*
   * The projected coarse shell evaluates the exact P^T A P action directly at
   * cell quadrature points.  This preserves the algebraic P4-to-P2 operator
   * without either assembling P2 or nesting three MatMult operations per P2
   * action.  P1 remains the sole assembled tangent level.
   */
  for (level = 1U; level < hierarchy.level_count - 1U; ++level) {
    if (PetscObjectReference(
            (PetscObject)mechanics->matrix_free_diagonal_operators[level]) !=
        PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    mechanics->matrix_free_level_operators[level] =
        mechanics->matrix_free_diagonal_operators[level];
  }
  return ss_pmg_set_level_operators(mechanics->pmg, hierarchy.level_count,
                                    mechanics->matrix_free_level_operators);
}
