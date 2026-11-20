/* Build and query packed owned and incident field-cell plans.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "mesh_internal.h"

#include "slope_internal/fe/fe.h"

#include <petscsection.h>

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static const SsStrongConstraint *ss_field_constraint_find(const SsFieldSpace *space,
                                                          SsEntityId entity_id,
                                                          uint32_t entity_slot,
                                                          uint32_t component) {
  const SsStrongConstraint key = {
      entity_id,
      entity_slot,
      component,
      0.0,
  };
  size_t left = 0U;
  size_t right;
  if (space == NULL) {
    return NULL;
  }
  right = space->constraint_count;
  while (left < right) {
    const size_t middle = left + (right - left) / 2U;
    if (ss_field_constraint_compare(&space->constraints[middle], &key) < 0) {
      left = middle + 1U;
    } else {
      right = middle;
    }
  }
  if (left < space->constraint_count &&
      ss_field_constraint_compare(&space->constraints[left], &key) == 0) {
    return &space->constraints[left];
  }
  return NULL;
}

static SsStatus ss_field_cell_dof(const SsFieldSpace *space, PetscInt cell,
                                  const SsEntityId *cell_vertex_ids, size_t node,
                                  uint32_t component, SsEntityId *entity_id,
                                  uint32_t *entity_slot, PetscInt *local_index,
                                  PetscInt *global_index, uint8_t *constrained,
                                  uint8_t *owned, double *prescribed_value) {
  PetscInt entity_point = -1;
  PetscInt local_offset = 0;
  PetscInt global_offset = 0;
  PetscInt dof = 0;
  PetscInt constraint_dof = 0;
  const PetscInt *constraint_indices = NULL;
  PetscInt raw_index;
  PetscInt preceding = 0;
  PetscInt constraint;
  const SsStrongConstraint *prescribed = NULL;
  if (space == NULL || cell_vertex_ids == NULL || entity_id == NULL ||
      entity_slot == NULL || local_index == NULL || global_index == NULL ||
      constrained == NULL || owned == NULL || prescribed_value == NULL ||
      component >= space->summary.components || node >= space->descriptor->node_count ||
      ss_mesh_node_entity_point(space->mesh->entities, space->mesh->entity_count,
                                space->descriptor, cell, cell_vertex_ids, node,
                                &entity_point, entity_slot) != SS_OK ||
      ss_mesh_cached_stable_id(space->mesh, entity_point, entity_id) != SS_OK ||
      PetscSectionGetDof(space->local_section, entity_point, &dof) != PETSC_SUCCESS ||
      PetscSectionGetOffset(space->local_section, entity_point, &local_offset) !=
          PETSC_SUCCESS ||
      PetscSectionGetOffset(space->global_section, entity_point, &global_offset) !=
          PETSC_SUCCESS ||
      PetscSectionGetConstraintDof(space->local_section, entity_point,
                                   &constraint_dof) != PETSC_SUCCESS ||
      (constraint_dof != 0 &&
       PetscSectionGetConstraintIndices(space->local_section, entity_point,
                                        &constraint_indices) != PETSC_SUCCESS)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  raw_index = (PetscInt)(*entity_slot * space->summary.components + component);
  if (raw_index < 0 || raw_index >= dof) {
    return SS_ERROR_INTEGRITY;
  }
  *constrained = 0U;
  for (constraint = 0; constraint < constraint_dof; ++constraint) {
    if (constraint_indices[constraint] < raw_index) {
      ++preceding;
    } else if (constraint_indices[constraint] == raw_index) {
      *constrained = 1U;
    }
  }
  *local_index = local_offset + raw_index;
  if (*constrained != 0U) {
    *global_index = -1;
  } else {
    const PetscInt base = global_offset < 0 ? -(global_offset + 1) : global_offset;
    *global_index = base + raw_index - preceding;
  }
  *owned =
      ss_mesh_point_is_ghost(space->mesh, space->mesh->dm, entity_point) == 0 ? 1U : 0U;
  prescribed = ss_field_constraint_find(space, *entity_id, *entity_slot, component);
  if ((*constrained != 0U) != (prescribed != NULL)) {
    return SS_ERROR_INTEGRITY;
  }
  *prescribed_value = prescribed == NULL ? 0.0 : prescribed->value;
  return SS_OK;
}

static int ss_size_align_up(size_t value, size_t alignment, size_t *aligned) {
  const size_t remainder = alignment == 0U ? 0U : value % alignment;
  const size_t padding = remainder == 0U ? 0U : alignment - remainder;
  if (aligned == NULL || alignment == 0U || value > SIZE_MAX - padding) {
    return 0;
  }
  *aligned = value + padding;
  return 1;
}

static SsStatus ss_cell_plan_segment(void *storage, size_t element_count,
                                     size_t element_size, size_t alignment,
                                     size_t *offset, void **segment) {
  size_t begin;
  size_t bytes;
  if (offset == NULL || segment == NULL ||
      !ss_size_align_up(*offset, alignment, &begin) ||
      (element_count != 0U && element_size > SIZE_MAX / element_count)) {
    return SS_ERROR_OVERFLOW;
  }
  bytes = element_count * element_size;
  if (begin > SIZE_MAX - bytes) {
    return SS_ERROR_OVERFLOW;
  }
  *segment = storage == NULL ? NULL : (void *)((unsigned char *)storage + begin);
  *offset = begin + bytes;
  return SS_OK;
}

static SsStatus ss_cell_plan_bind(const SsFieldSpace *space, void *storage,
                                  SsCellPlan *plan, size_t *stride) {
  size_t coordinate_nodes;
  size_t field_nodes;
  size_t closure;
  size_t offset = 0U;
  void *segment = NULL;
  SsStatus status;
  if (space == NULL || plan == NULL || stride == NULL || space->mesh == NULL ||
      space->mesh->coordinate_descriptor == NULL || space->descriptor == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  coordinate_nodes = space->mesh->coordinate_descriptor->node_count;
  field_nodes = space->descriptor->node_count;
  if (space->summary.components == 0U ||
      field_nodes > SIZE_MAX / (size_t)space->summary.components) {
    return SS_ERROR_OVERFLOW;
  }
  closure = field_nodes * (size_t)space->summary.components;
  if (coordinate_nodes == 0U || coordinate_nodes > SS_FE_MAX_NODES ||
      field_nodes == 0U || field_nodes > SS_FE_MAX_NODES || closure == 0U ||
      closure > SS_FIELD_MAX_CLOSURE) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
#define SS_CELL_PLAN_SEGMENT(member, count, type)                                      \
  do {                                                                                 \
    status = ss_cell_plan_segment(storage, (count), sizeof(type), _Alignof(type),      \
                                  &offset, &segment);                                  \
    if (status != SS_OK) {                                                             \
      return status;                                                                   \
    }                                                                                  \
    plan->member = (type *)segment;                                                    \
  } while (0)
  SS_CELL_PLAN_SEGMENT(coordinate_node_ids, coordinate_nodes, SsEntityId);
  SS_CELL_PLAN_SEGMENT(coordinate_coordinates,
                       coordinate_nodes * (size_t)space->mesh->dimension, double);
  SS_CELL_PLAN_SEGMENT(field_entity_ids, closure, SsEntityId);
  SS_CELL_PLAN_SEGMENT(field_entity_slots, closure, uint32_t);
  SS_CELL_PLAN_SEGMENT(field_components, closure, uint32_t);
  SS_CELL_PLAN_SEGMENT(local_indices, closure, PetscInt);
  SS_CELL_PLAN_SEGMENT(global_indices, closure, PetscInt);
  SS_CELL_PLAN_SEGMENT(constrained, closure, uint8_t);
  SS_CELL_PLAN_SEGMENT(owned, closure, uint8_t);
  SS_CELL_PLAN_SEGMENT(prescribed_values, closure, double);
  SS_CELL_PLAN_SEGMENT(field_node_coordinates,
                       field_nodes * (size_t)space->mesh->dimension, double);
#undef SS_CELL_PLAN_SEGMENT
  if (!ss_size_align_up(offset, _Alignof(double), stride)) {
    return SS_ERROR_OVERFLOW;
  }
  return SS_OK;
}

static SsStatus ss_field_cell_plans_allocate(const SsFieldSpace *space, size_t count,
                                             SsCellPlan **plans, void **storage,
                                             size_t *storage_bytes) {
  SsCellPlan layout = {0};
  size_t stride = 0U;
  size_t index;
  SsStatus status;
  if (space == NULL || plans == NULL || storage == NULL || storage_bytes == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *plans = NULL;
  *storage = NULL;
  *storage_bytes = 0U;
  status = ss_cell_plan_bind(space, NULL, &layout, &stride);
  if (status != SS_OK || count > SIZE_MAX / sizeof(**plans) ||
      (count != 0U && stride > SIZE_MAX / count)) {
    return status != SS_OK ? status : SS_ERROR_OVERFLOW;
  }
  if (count == 0U) {
    return SS_OK;
  }
  *plans = (SsCellPlan *)calloc(count, sizeof(**plans));
  *storage = calloc(count, stride);
  if (*plans == NULL || *storage == NULL) {
    free(*plans);
    free(*storage);
    *plans = NULL;
    *storage = NULL;
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (index = 0U; index < count; ++index) {
    size_t actual_stride = 0U;
    status = ss_cell_plan_bind(space, (unsigned char *)*storage + index * stride,
                               &(*plans)[index], &actual_stride);
    if (status != SS_OK || actual_stride != stride) {
      free(*plans);
      free(*storage);
      *plans = NULL;
      *storage = NULL;
      return status != SS_OK ? status : SS_ERROR_INTEGRITY;
    }
  }
  *storage_bytes = count * stride;
  return SS_OK;
}

static SsStatus ss_field_pack_cell(const SsFieldSpace *space, const SsPointEntry *entry,
                                   SsCellPlan *pack) {
  SsEntityId cell_vertex_ids[4] = {0U, 0U, 0U, 0U};
  double reference[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
  size_t node;
  uint32_t raw_material = 0U;
  if (space == NULL || entry == NULL || pack == NULL ||
      entry->kind != SS_MESH_ENTITY_CELL ||
      ss_mesh_cell_vertex_ids(space->mesh->dm, space->mesh, entry->point,
                              space->mesh->dimension, cell_vertex_ids) != SS_OK ||
      ss_mesh_label_value(space->mesh, SS_MESH_LABEL_MATERIAL, entry->stable_id,
                          &raw_material) != SS_OK) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  pack->dimension = space->mesh->dimension;
  pack->stable_cell_id = entry->stable_id;
  pack->material_value = raw_material;
  pack->cell_owned =
      ss_mesh_point_is_ghost(space->mesh, space->mesh->dm, entry->point) == 0 ? 1U : 0U;
  pack->coordinate_order = space->mesh->coordinate_order;
  if (ss_mesh_cell_coordinate_closure(
          space->mesh, entry->stable_id, pack->coordinate_node_ids,
          pack->coordinate_coordinates, space->mesh->coordinate_descriptor->node_count,
          &pack->coordinate_node_count) != SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  pack->order = space->summary.order;
  pack->components = space->summary.components;
  pack->field_node_count = space->descriptor->node_count;
  pack->closure_count =
      space->descriptor->node_count * (size_t)space->summary.components;
  if (pack->closure_count > SS_FIELD_MAX_CLOSURE) {
    return SS_ERROR_OVERFLOW;
  }
  for (node = 0U; node < space->descriptor->node_count; ++node) {
    SsFeGeometry geometry = {0};
    uint32_t coordinate;
    uint32_t component;
    for (coordinate = 0U; coordinate < space->mesh->dimension; ++coordinate) {
      const size_t barycentric_offset =
          node * (size_t)(space->mesh->dimension + 1U) + coordinate + 1U;
      reference[coordinate] =
          (double)space->descriptor->barycentric_indices[barycentric_offset] /
          (double)space->descriptor->order;
    }
    if (ss_fe_geometry_eval(space->mesh->coordinate_descriptor,
                            pack->coordinate_coordinates, reference,
                            &space->mesh->geometry_policy, &geometry) != SS_OK) {
      return SS_ERROR_INTEGRITY;
    }
    for (coordinate = 0U; coordinate < space->mesh->dimension; ++coordinate) {
      pack->field_node_coordinates[node * (size_t)space->mesh->dimension + coordinate] =
          geometry.physical[coordinate];
    }
    for (component = 0U; component < space->summary.components; ++component) {
      const size_t closure = node * (size_t)space->summary.components + component;
      if (ss_field_cell_dof(
              space, entry->point, cell_vertex_ids, node, component,
              &pack->field_entity_ids[closure], &pack->field_entity_slots[closure],
              &pack->local_indices[closure], &pack->global_indices[closure],
              &pack->constrained[closure], &pack->owned[closure],
              &pack->prescribed_values[closure]) != SS_OK) {
        return SS_ERROR_INTERNAL;
      }
      pack->field_components[closure] = component;
    }
  }
  return SS_OK;
}

SsStatus ss_field_owned_cell_plans_build(SsFieldSpace *space) {
  size_t incident_index;
  size_t required = 0U;
  size_t packed = 0U;
  SsCellPlan *plans = NULL;
  if (space == NULL || space->owned_cell_plans != NULL ||
      space->owned_cell_plan_count != 0U ||
      (space->incident_cell_plan_count != 0U && space->incident_cell_plans == NULL)) {
    return SS_ERROR_INVALID_STATE;
  }
  for (incident_index = 0U; incident_index < space->incident_cell_plan_count;
       ++incident_index) {
    if (space->incident_cell_plans[incident_index].cell_owned != 0U) {
      ++required;
    }
  }
  if (required > SIZE_MAX / sizeof(*plans) ||
      required > (size_t)(UINT64_MAX / sizeof(*plans))) {
    return SS_ERROR_OVERFLOW;
  }
  if (required != 0U) {
    plans = (SsCellPlan *)calloc(required, sizeof(*plans));
    if (plans == NULL) {
      return SS_ERROR_OUT_OF_MEMORY;
    }
  }
  for (incident_index = 0U; incident_index < space->incident_cell_plan_count;
       ++incident_index) {
    if (space->incident_cell_plans[incident_index].cell_owned == 0U) {
      continue;
    }
    if (packed >= required) {
      free(plans);
      return SS_ERROR_INTEGRITY;
    }
    plans[packed] = space->incident_cell_plans[incident_index];
    ++packed;
  }
  if (packed != required) {
    free(plans);
    return SS_ERROR_INTEGRITY;
  }
  space->owned_cell_plans = plans;
  space->owned_cell_plan_count = required;
  space->summary.owned_cell_plan_count = (uint64_t)required;
  space->summary.closure_plan_bytes = (uint64_t)required * (uint64_t)sizeof(*plans);
  return SS_OK;
}

SsStatus ss_field_incident_cell_plans_build(SsFieldSpace *space) {
  size_t entry_index;
  size_t required = 0U;
  size_t packed = 0U;
  SsCellPlan *plans = NULL;
  void *storage = NULL;
  size_t storage_bytes = 0U;
  SsStatus status = SS_OK;
  if (space == NULL || space->incident_cell_plans != NULL ||
      space->incident_cell_plan_count != 0U) {
    return SS_ERROR_INVALID_STATE;
  }
  for (entry_index = 0U; entry_index < space->mesh->point_count; ++entry_index) {
    if (space->mesh->points[entry_index].kind == SS_MESH_ENTITY_CELL) {
      ++required;
    }
  }
  status =
      ss_field_cell_plans_allocate(space, required, &plans, &storage, &storage_bytes);
  if (status != SS_OK) {
    return status;
  }
  for (entry_index = 0U; entry_index < space->mesh->point_count; ++entry_index) {
    const SsPointEntry *entry = &space->mesh->points[entry_index];
    if (entry->kind != SS_MESH_ENTITY_CELL) {
      continue;
    }
    if (packed >= required ||
        ss_field_pack_cell(space, entry, &plans[packed]) != SS_OK) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    ++packed;
  }
  if (status == SS_OK && packed != required) {
    status = SS_ERROR_INTEGRITY;
  }
  if (status != SS_OK) {
    free(plans);
    free(storage);
    return status;
  }
  space->incident_cell_plans = plans;
  space->incident_cell_plan_count = required;
  space->incident_cell_plan_storage = storage;
  space->incident_cell_plan_storage_bytes = storage_bytes;
  space->summary.incident_cell_plan_count = (uint64_t)required;
  space->summary.incident_closure_plan_bytes =
      (uint64_t)required * (uint64_t)sizeof(*plans) + (uint64_t)storage_bytes;
  space->summary.overlap_depth = space->mesh->size > 1 ? 1U : 0U;
  return SS_OK;
}

const SsCellPlan *ss_field_owned_cell_plan_find(const SsFieldSpace *space,
                                                SsEntityId stable_cell_id) {
  size_t left = 0U;
  size_t right;
  if (space == NULL) {
    return NULL;
  }
  right = space->owned_cell_plan_count;
  while (left < right) {
    const size_t middle = left + (right - left) / 2U;
    if (space->owned_cell_plans[middle].stable_cell_id < stable_cell_id) {
      left = middle + 1U;
    } else {
      right = middle;
    }
  }
  if (left < space->owned_cell_plan_count &&
      space->owned_cell_plans[left].stable_cell_id == stable_cell_id) {
    return &space->owned_cell_plans[left];
  }
  return NULL;
}

const SsCellPlan *ss_field_incident_cell_plan_find(const SsFieldSpace *space,
                                                   SsEntityId stable_cell_id) {
  size_t left = 0U;
  size_t right = space->incident_cell_plan_count;
  while (left < right) {
    const size_t middle = left + (right - left) / 2U;
    if (space->incident_cell_plans[middle].stable_cell_id < stable_cell_id) {
      left = middle + 1U;
    } else {
      right = middle;
    }
  }
  if (left < space->incident_cell_plan_count &&
      space->incident_cell_plans[left].stable_cell_id == stable_cell_id) {
    return &space->incident_cell_plans[left];
  }
  return NULL;
}

static SsStatus ss_cell_plan_export(const SsCellPlan *source, SsCellPack *pack) {
  size_t coordinate_value_count;
  size_t field_coordinate_count;
  if (source == NULL || pack == NULL ||
      source->coordinate_node_count > SS_FE_MAX_NODES ||
      source->field_node_count > SS_FE_MAX_NODES ||
      source->closure_count > SS_FIELD_MAX_CLOSURE || source->dimension == 0U ||
      source->dimension > SS_FE_MAX_DIMENSION) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  coordinate_value_count = source->coordinate_node_count * (size_t)source->dimension;
  field_coordinate_count = source->field_node_count * (size_t)source->dimension;
  memset(pack, 0, sizeof(*pack));
  pack->dimension = source->dimension;
  pack->stable_cell_id = source->stable_cell_id;
  pack->material_value = source->material_value;
  pack->coordinate_order = source->coordinate_order;
  pack->coordinate_node_count = source->coordinate_node_count;
  pack->order = source->order;
  pack->components = source->components;
  pack->field_node_count = source->field_node_count;
  pack->closure_count = source->closure_count;
  pack->cell_owned = source->cell_owned;
  memcpy(pack->coordinate_node_ids, source->coordinate_node_ids,
         source->coordinate_node_count * sizeof(*pack->coordinate_node_ids));
  memcpy(pack->coordinate_coordinates, source->coordinate_coordinates,
         coordinate_value_count * sizeof(*pack->coordinate_coordinates));
  memcpy(pack->field_entity_ids, source->field_entity_ids,
         source->closure_count * sizeof(*pack->field_entity_ids));
  memcpy(pack->field_entity_slots, source->field_entity_slots,
         source->closure_count * sizeof(*pack->field_entity_slots));
  memcpy(pack->field_components, source->field_components,
         source->closure_count * sizeof(*pack->field_components));
  memcpy(pack->local_indices, source->local_indices,
         source->closure_count * sizeof(*pack->local_indices));
  memcpy(pack->global_indices, source->global_indices,
         source->closure_count * sizeof(*pack->global_indices));
  memcpy(pack->constrained, source->constrained,
         source->closure_count * sizeof(*pack->constrained));
  memcpy(pack->owned, source->owned, source->closure_count * sizeof(*pack->owned));
  memcpy(pack->prescribed_values, source->prescribed_values,
         source->closure_count * sizeof(*pack->prescribed_values));
  memcpy(pack->field_node_coordinates, source->field_node_coordinates,
         field_coordinate_count * sizeof(*pack->field_node_coordinates));
  return SS_OK;
}

SsStatus ss_field_space_pack_owned_cells(const SsFieldSpace *space, SsCellPack *packs,
                                         size_t capacity, size_t *count) {
  size_t index;
  if (space == NULL || count == NULL || (packs == NULL && capacity != 0U)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *count = space->owned_cell_plan_count;
  if (packs == NULL) {
    return SS_OK;
  }
  if (capacity < space->owned_cell_plan_count) {
    return SS_ERROR_LIMIT;
  }
  for (index = 0U; index < space->owned_cell_plan_count; ++index) {
    const SsStatus status =
        ss_cell_plan_export(&space->owned_cell_plans[index], &packs[index]);
    if (status != SS_OK) {
      return status;
    }
  }
  return SS_OK;
}

SsStatus ss_field_space_pack_incident_cells(const SsFieldSpace *space,
                                            SsCellPack *packs, size_t capacity,
                                            size_t *count) {
  size_t index;
  if (space == NULL || count == NULL || (packs == NULL && capacity != 0U)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *count = space->incident_cell_plan_count;
  if (packs == NULL) {
    return SS_OK;
  }
  if (capacity < space->incident_cell_plan_count) {
    return SS_ERROR_LIMIT;
  }
  for (index = 0U; index < space->incident_cell_plan_count; ++index) {
    const SsStatus status =
        ss_cell_plan_export(&space->incident_cell_plans[index], &packs[index]);
    if (status != SS_OK) {
      return status;
    }
  }
  return SS_OK;
}

SsStatus ss_field_space_owned_cell_plan_view(const SsFieldSpace *space,
                                             const SsCellPlan **plans, size_t *count) {
  if (space == NULL || plans == NULL || count == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *plans = space->owned_cell_plans;
  *count = space->owned_cell_plan_count;
  return SS_OK;
}

SsStatus ss_field_space_incident_cell_plan_view(const SsFieldSpace *space,
                                                const SsCellPlan **plans,
                                                size_t *count) {
  if (space == NULL || plans == NULL || count == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *plans = space->incident_cell_plans;
  *count = space->incident_cell_plan_count;
  return SS_OK;
}
