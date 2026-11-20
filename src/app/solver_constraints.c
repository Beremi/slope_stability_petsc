/* Build and evaluate essential mechanics and hydraulic constraints.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/core/sort.h"
#include "slope_internal/core/util.h"
#include "solver_input_internal.h"

#include <petsclog.h>

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int ss_solver_entity_subset(const SsEntityId *candidate, size_t candidate_count,
                                   const SsEntityId *container,
                                   size_t container_count) {
  size_t candidate_index;
  size_t container_index = 0U;
  for (candidate_index = 0U; candidate_index < candidate_count; ++candidate_index) {
    while (container_index < container_count &&
           container[container_index] < candidate[candidate_index]) {
      ++container_index;
    }
    if (container_index == container_count ||
        container[container_index] != candidate[candidate_index]) {
      return 0;
    }
    ++container_index;
  }
  return 1;
}

static size_t ss_solver_entity_slots(uint32_t entity_dimension, uint32_t order) {
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

static double ss_solver_boundary_value(const SsSolverBoundaryRule *rule,
                                       uint32_t component) {
  if (rule->value_kind == SS_SOLVER_VALUE_CONSTANT) {
    if (rule->value_count == 0U) {
      return 0.0;
    }
    return component < rule->value_count ? rule->values[component] : rule->values[0];
  }
  if (rule->value_kind == SS_SOLVER_VALUE_AFFINE_COORDINATE) {
    return rule->values[0];
  }
  /*
   * Nonconstant pressure constraints are populated with the reference value
   * here. The hydraulic graph refines them at physical field-node coordinates.
   */
  return rule->reference_pressure;
}

static int ss_solver_constraint_compare(const void *left_pointer,
                                        const void *right_pointer) {
  const SsStrongConstraint *left = (const SsStrongConstraint *)left_pointer;
  const SsStrongConstraint *right = (const SsStrongConstraint *)right_pointer;
  if (left->stable_entity_id != right->stable_entity_id) {
    return left->stable_entity_id < right->stable_entity_id ? -1 : 1;
  }
  if (left->entity_slot != right->entity_slot) {
    return left->entity_slot < right->entity_slot ? -1 : 1;
  }
  if (left->component != right->component) {
    return left->component < right->component ? -1 : 1;
  }
  return 0;
}

static SsStatus ss_solver_constraint_append(SsStrongConstraint **records, size_t *count,
                                            size_t *capacity,
                                            SsEntityId stable_entity_id, size_t slot,
                                            uint32_t component, double value) {
  SsStrongConstraint *next;
  size_t next_capacity;
  if (records == NULL || count == NULL || capacity == NULL || slot > UINT32_MAX ||
      !isfinite(value)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (*count == *capacity) {
    next_capacity = *capacity == 0U ? 32U : *capacity * 2U;
    if (next_capacity < *capacity || next_capacity > SIZE_MAX / sizeof(**records)) {
      return SS_ERROR_OVERFLOW;
    }
    next = (SsStrongConstraint *)realloc(*records, next_capacity * sizeof(**records));
    if (next == NULL) {
      return SS_ERROR_OUT_OF_MEMORY;
    }
    *records = next;
    *capacity = next_capacity;
  }
  (*records)[(*count)++] =
      (SsStrongConstraint){stable_entity_id, (uint32_t)slot, component, value};
  return SS_OK;
}

static SsStatus ss_solver_append_resident_constraints(const SsSolverGraph *graph,
                                                      const SsSolverBoundaryRule *rule,
                                                      uint32_t order,
                                                      uint32_t components,
                                                      SsStrongConstraint **records,
                                                      size_t *count, size_t *capacity) {
  SsMeshEntityRef *members = NULL;
  size_t member_count = 0U;
  SsStatus status = ss_mesh_essential_members(graph->mesh, rule->support_value,
                                              &members, &member_count);
  for (size_t i = 0U; status == SS_OK && i < member_count; ++i) {
    const size_t slots = ss_solver_entity_slots(members[i].dimension, order);
    for (size_t slot = 0U; status == SS_OK && slot < slots; ++slot) {
      for (uint32_t component = 0U; status == SS_OK && component < components;
           ++component) {
        if ((rule->component_mask & (UINT32_C(1) << component)) != 0U) {
          status = ss_solver_constraint_append(
              records, count, capacity, members[i].stable_id, slot, component, 0.0);
        }
      }
    }
  }
  free(members);
  return status;
}

SsStatus ss_solver_build_constraints_for_field(const SsSolverGraph *graph,
                                               const char *field, uint32_t order,
                                               uint32_t components,
                                               SsStrongConstraint **constraints,
                                               size_t *constraint_count) {
  size_t capacity = 0U;
  size_t count = 0U;
  size_t rule_index;
  SsStrongConstraint *records = NULL;
  for (rule_index = 0U; rule_index < graph->model.boundary_count; ++rule_index) {
    const SsSolverBoundaryRule *rule = &graph->model.boundaries[rule_index];
    size_t target_index;
    int vertex_support = 1;
    if (rule->condition_kind != SS_SOLVER_BOUNDARY_ESSENTIAL ||
        strcmp(rule->field, field) != 0) {
      continue;
    }
    if (graph->mesh != NULL) {
      const SsStatus status = ss_solver_append_resident_constraints(
          graph, rule, order, components, &records, &count, &capacity);
      if (status != SS_OK) {
        free(records);
        return status;
      }
      continue;
    }
    /* The in-memory boundary-trace fixtures provide tables instead of a DM. */
    for (target_index = 0U; target_index < rule->entity_count; ++target_index) {
      if (!ss_solver_entity_id_index(graph->raw.entities[0].ids,
                                     graph->raw.entities[0].count,
                                     rule->entity_ids[target_index], NULL)) {
        vertex_support = 0;
        break;
      }
    }
    /*
     * Compiled node-set supports contain their stable vertices.  Essential
     * high-order traces also own edge and face slots, so close the complete
     * vertex support only through exterior entities.  Boundary vertices can
     * also span interior edges/faces; those are not inferred trace supports.
     * Explicitly selected points and entity supports remain valid.
     */
    if (vertex_support != 0) {
      uint32_t closure_dimension;
      for (closure_dimension = 0U; closure_dimension < graph->model.dimension;
           ++closure_dimension) {
        const SsSolverEntityTable *closure = &graph->raw.entities[closure_dimension];
        const size_t slots = ss_solver_entity_slots(closure_dimension, order);
        size_t entity;
        if (slots == 0U) {
          continue;
        }
        for (entity = 0U; entity < closure->count; ++entity) {
          const SsEntityId *vertices = closure->vertex_ids + entity * closure->width;
          size_t slot;
          uint32_t component;
          if ((closure_dimension > 0U &&
               (closure->exterior == NULL || closure->exterior[entity] == 0U)) ||
              !ss_solver_entity_subset(vertices, closure->width, rule->entity_ids,
                                       rule->entity_count)) {
            continue;
          }
          for (slot = 0U; slot < slots; ++slot) {
            for (component = 0U; component < components; ++component) {
              SsStatus status;
              if ((rule->component_mask & (UINT32_C(1) << component)) == 0U) {
                continue;
              }
              status = ss_solver_constraint_append(
                  &records, &count, &capacity, closure->ids[entity], slot, component,
                  ss_solver_boundary_value(rule, component));
              if (status != SS_OK) {
                free(records);
                return status;
              }
            }
          }
        }
      }
      continue;
    }
    for (target_index = 0U; target_index < rule->entity_count; ++target_index) {
      uint32_t target_dimension;
      for (target_dimension = 0U; target_dimension < graph->model.dimension;
           ++target_dimension) {
        const SsSolverEntityTable *target_table =
            &graph->raw.entities[target_dimension];
        size_t target_ordinal;
        if (!ss_solver_entity_id_index(target_table->ids, target_table->count,
                                       rule->entity_ids[target_index],
                                       &target_ordinal)) {
          continue;
        }
        {
          const SsEntityId *target_vertices =
              target_table->vertex_ids + target_ordinal * target_table->width;
          uint32_t closure_dimension;
          for (closure_dimension = 0U; closure_dimension <= target_dimension;
               ++closure_dimension) {
            const SsSolverEntityTable *closure =
                &graph->raw.entities[closure_dimension];
            const size_t slots = ss_solver_entity_slots(closure_dimension, order);
            size_t entity;
            if (slots == 0U) {
              continue;
            }
            for (entity = 0U; entity < closure->count; ++entity) {
              const SsEntityId *vertices =
                  closure->vertex_ids + entity * closure->width;
              size_t slot;
              uint32_t component;
              if (!ss_solver_entity_subset(vertices, closure->width, target_vertices,
                                           target_table->width)) {
                continue;
              }
              for (slot = 0U; slot < slots; ++slot) {
                for (component = 0U; component < components; ++component) {
                  SsStatus status;
                  if ((rule->component_mask & (UINT32_C(1) << component)) == 0U) {
                    continue;
                  }
                  status = ss_solver_constraint_append(
                      &records, &count, &capacity, closure->ids[entity], slot,
                      component, ss_solver_boundary_value(rule, component));
                  if (status != SS_OK) {
                    free(records);
                    return status;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  ss_sort(records, count, sizeof(*records), ss_solver_constraint_compare);
  {
    size_t source;
    size_t output = 0U;
    for (source = 0U; source < count; ++source) {
      if (output != 0U &&
          ss_solver_constraint_compare(&records[output - 1U], &records[source]) == 0) {
        /*
         * The preliminary field space needs one topological constraint per
         * functional.  Actual values, including intersections between unlike
         * analytic models, are compared at physical field-node coordinates by
         * ss_solver_refine_constraints.
         */
        continue;
      }
      records[output++] = records[source];
    }
    count = output;
  }
  *constraints = records;
  *constraint_count = count;
  return SS_OK;
}

int ss_solver_rule_contains_entity(const SsSolverGraph *graph,
                                   const SsSolverBoundaryRule *rule,
                                   SsEntityId stable_entity_id) {
  if (graph->mesh != NULL) {
    return ss_mesh_essential_contains(graph->mesh, rule->support_value,
                                      stable_entity_id);
  }
  const SsSolverEntityTable *candidate = NULL;
  size_t candidate_ordinal = 0U;
  uint32_t candidate_dimension;
  size_t target_index;
  int vertex_support = 1;
  for (candidate_dimension = 0U; candidate_dimension < graph->model.dimension;
       ++candidate_dimension) {
    const SsSolverEntityTable *table = &graph->raw.entities[candidate_dimension];
    if (ss_solver_entity_id_index(table->ids, table->count, stable_entity_id,
                                  &candidate_ordinal)) {
      candidate = table;
      break;
    }
  }
  if (candidate == NULL) {
    return 0;
  }
  for (target_index = 0U; target_index < rule->entity_count; ++target_index) {
    if (!ss_solver_entity_id_index(graph->raw.entities[0].ids,
                                   graph->raw.entities[0].count,
                                   rule->entity_ids[target_index], NULL)) {
      vertex_support = 0;
      break;
    }
  }
  if (vertex_support != 0) {
    if (candidate_dimension > 0U &&
        (candidate->exterior == NULL || candidate->exterior[candidate_ordinal] == 0U)) {
      return 0;
    }
    return ss_solver_entity_subset(
        candidate->vertex_ids + candidate_ordinal * candidate->width, candidate->width,
        rule->entity_ids, rule->entity_count);
  }
  for (target_index = 0U; target_index < rule->entity_count; ++target_index) {
    uint32_t target_dimension;
    for (target_dimension = candidate_dimension;
         target_dimension < graph->model.dimension; ++target_dimension) {
      const SsSolverEntityTable *target = &graph->raw.entities[target_dimension];
      size_t target_ordinal;
      if (ss_solver_entity_id_index(target->ids, target->count,
                                    rule->entity_ids[target_index], &target_ordinal) &&
          ss_solver_entity_subset(
              candidate->vertex_ids + candidate_ordinal * candidate->width,
              candidate->width, target->vertex_ids + target_ordinal * target->width,
              target->width)) {
        return 1;
      }
    }
  }
  return 0;
}

SsStatus ss_solver_displacement_boundary_value(const SsSolverGraph *graph,
                                               SsEntityId stable_entity_id,
                                               const double *physical,
                                               uint32_t component, double *value) {
  size_t rule_index;
  int matched = 0;
  double selected = NAN;
  if (graph == NULL || physical == NULL || value == NULL ||
      component >= graph->model.dimension) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (rule_index = 0U; rule_index < graph->model.boundary_count; ++rule_index) {
    const SsSolverBoundaryRule *rule = &graph->model.boundaries[rule_index];
    double candidate = NAN;
    uint32_t coordinate;
    if (rule->condition_kind != SS_SOLVER_BOUNDARY_ESSENTIAL ||
        strcmp(rule->field, "displacement") != 0 ||
        (rule->component_mask & (UINT32_C(1) << component)) == 0U ||
        !ss_solver_rule_contains_entity(graph, rule, stable_entity_id)) {
      continue;
    }
    if (rule->value_kind == SS_SOLVER_VALUE_CONSTANT) {
      candidate = ss_solver_boundary_value(rule, component);
    } else if (rule->value_kind == SS_SOLVER_VALUE_AFFINE_COORDINATE) {
      candidate = rule->values[0];
      for (coordinate = 0U; coordinate < graph->model.dimension; ++coordinate) {
        candidate += rule->gradient[coordinate] * physical[coordinate];
      }
    } else {
      return SS_ERROR_SCHEMA;
    }
    if (!isfinite(candidate)) {
      return SS_ERROR_INTEGRITY;
    }
    if (matched != 0 &&
        fabs(candidate - selected) >
            1.0e-12 * fmax(1.0, fmax(fabs(candidate), fabs(selected)))) {
      return SS_ERROR_INTEGRITY;
    }
    selected = candidate;
    matched = 1;
  }
  if (matched == 0) {
    return SS_ERROR_NOT_FOUND;
  }
  *value = selected;
  return SS_OK;
}

static SsStatus ss_solver_pressure_boundary_value(const SsSolverGraph *graph,
                                                  SsEntityId stable_entity_id,
                                                  const double *physical,
                                                  double *value) {
  size_t rule_index;
  int matched = 0;
  double selected = NAN;
  if (graph == NULL || physical == NULL || value == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (rule_index = 0U; rule_index < graph->model.boundary_count; ++rule_index) {
    const SsSolverBoundaryRule *rule = &graph->model.boundaries[rule_index];
    double candidate = NAN;
    if (rule->condition_kind != SS_SOLVER_BOUNDARY_ESSENTIAL ||
        strcmp(rule->field, "pressure") != 0 ||
        !ss_solver_rule_contains_entity(graph, rule, stable_entity_id)) {
      continue;
    }
    if (rule->value_kind == SS_SOLVER_VALUE_CONSTANT) {
      candidate = ss_solver_boundary_value(rule, 0U);
    } else if (rule->value_kind == SS_SOLVER_VALUE_AFFINE_COORDINATE) {
      uint32_t coordinate;
      candidate = rule->values[0];
      for (coordinate = 0U; coordinate < graph->model.dimension; ++coordinate) {
        candidate += rule->gradient[coordinate] * physical[coordinate];
      }
    } else {
      SsHydraulicHeadModel head;
      memset(&head, 0, sizeof(head));
      head.dimension = graph->model.dimension;
      head.upward_axis = 1U;
      head.piecewise_axis = 0U;
      head.gamma_water = rule->water_unit_weight;
      if (rule->value_kind == SS_SOLVER_VALUE_HYDROSTATIC) {
        head.kind = SS_HYDRAULIC_HEAD_CONSTANT_LEVEL;
        head.level = rule->reference_level;
      } else if (rule->value_kind == SS_SOLVER_VALUE_PIECEWISE_LEVEL) {
        head.kind = SS_HYDRAULIC_HEAD_PIECEWISE_LINEAR_LEVEL;
        head.piecewise_point_count = rule->piecewise_count;
        memcpy(head.piecewise_coordinates, rule->piecewise_coordinates,
               rule->piecewise_count * sizeof(*head.piecewise_coordinates));
        memcpy(head.piecewise_levels, rule->piecewise_levels,
               rule->piecewise_count * sizeof(*head.piecewise_levels));
      } else {
        return SS_ERROR_SCHEMA;
      }
      (void)snprintf(head.identity, sizeof(head.identity), "%s",
                     "native-boundary-head");
      if (ss_hydraulic_head_evaluate(&head, physical, &candidate) != SS_OK) {
        return SS_ERROR_INTEGRITY;
      }
      candidate += rule->reference_pressure;
    }
    if (!isfinite(candidate)) {
      return SS_ERROR_INTEGRITY;
    }
    if (matched != 0 &&
        fabs(candidate - selected) >
            1.0e-10 * fmax(1.0, fmax(fabs(candidate), fabs(selected)))) {
      return SS_ERROR_INTEGRITY;
    }
    selected = candidate;
    matched = 1;
  }
  if (matched == 0) {
    return SS_ERROR_NOT_FOUND;
  }
  *value = selected;
  return SS_OK;
}

static size_t ss_solver_constraint_find(const SsStrongConstraint *constraints,
                                        size_t count, SsEntityId stable_entity_id,
                                        uint32_t entity_slot, uint32_t component) {
  const SsStrongConstraint key = {stable_entity_id, entity_slot, component, 0.0};
  size_t left = 0U;
  size_t right = count;
  while (left < right) {
    const size_t middle = left + (right - left) / 2U;
    if (ss_solver_constraint_compare(&constraints[middle], &key) < 0) {
      left = middle + 1U;
    } else {
      right = middle;
    }
  }
  return left < count && ss_solver_constraint_compare(&constraints[left], &key) == 0
             ? left
             : count;
}

SsStatus ss_solver_pressure_constraint_value(const SsSolverGraph *graph,
                                             SsEntityId stable_entity_id,
                                             const double *physical, uint32_t component,
                                             double *value) {
  if (component != 0U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return ss_solver_pressure_boundary_value(graph, stable_entity_id, physical, value);
}

SsStatus ss_solver_refine_constraints(SsSolverGraph *graph, SsFieldSpace *space,
                                      SsStrongConstraint *constraints,
                                      size_t constraint_count, uint32_t components,
                                      SsSolverConstraintValueEvaluator evaluator) {
  const SsCellPlan *packs = NULL;
  size_t pack_count = 0U;
  SsConstraintCandidate *candidates = NULL;
  SsStatus status;
  if (graph == NULL || space == NULL ||
      (constraint_count != 0U && constraints == NULL) || components == 0U ||
      components > graph->model.dimension || evaluator == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_field_space_owned_cell_plan_view(space, &packs, &pack_count);
  if (status == SS_OK) {
    candidates =
        calloc(constraint_count == 0U ? 1U : constraint_count, sizeof(*candidates));
    if (candidates == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  for (size_t pack_index = 0U; status == SS_OK && pack_index < pack_count;
       ++pack_index) {
    const SsCellPlan *pack = &packs[pack_index];
    if (pack->components != components) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    for (size_t node = 0U; status == SS_OK && node < pack->field_node_count; ++node) {
      const size_t base = node * (size_t)pack->components;
      const SsEntityId entity_id = pack->field_entity_ids[base];
      const uint32_t entity_slot = pack->field_entity_slots[base];
      for (uint32_t component = 0U; status == SS_OK && component < components;
           ++component) {
        const size_t index = ss_solver_constraint_find(
            constraints, constraint_count, entity_id, entity_slot, component);
        if (index == constraint_count) {
          continue;
        }
        double value = NAN;
        status = evaluator(graph, entity_id,
                           pack->field_node_coordinates +
                               node * (size_t)graph->model.dimension,
                           component, &value);
        if (status == SS_OK && !isfinite(value)) {
          status = SS_ERROR_INTEGRITY;
        }
        if (status == SS_OK && (candidates[index].present == 0 ||
                                pack->stable_cell_id < candidates[index].cell_id)) {
          candidates[index] = (SsConstraintCandidate){pack->stable_cell_id, value, 1};
        }
      }
    }
  }
  status = ss_status_allreduce_max(graph->communicator, status);
  if (status == SS_OK) {
    status = ss_field_space_resolve_constraint_values(space, candidates, constraints,
                                                      constraint_count);
  }
  free(candidates);
  return status;
}
