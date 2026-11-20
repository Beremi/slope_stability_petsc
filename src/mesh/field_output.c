/* Reusable field extraction from local PETSc sections and owned vector rows.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "mesh_internal.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/core/sort.h"

#include <petscsection.h>

#include <math.h>
#include <stdlib.h>

static SsStatus ss_field_point_layout(const SsFieldSpace *space, PetscInt point,
                                      PetscInt *dof, PetscInt *local_offset,
                                      PetscInt *global_offset, PetscInt *constrained,
                                      const PetscInt **indices) {
  *indices = NULL;
  if (PetscSectionGetDof(space->local_section, point, dof) != PETSC_SUCCESS ||
      PetscSectionGetOffset(space->local_section, point, local_offset) !=
          PETSC_SUCCESS ||
      PetscSectionGetOffset(space->global_section, point, global_offset) !=
          PETSC_SUCCESS ||
      PetscSectionGetConstraintDof(space->local_section, point, constrained) !=
          PETSC_SUCCESS ||
      (*constrained != 0 &&
       PetscSectionGetConstraintIndices(space->local_section, point, indices) !=
           PETSC_SUCCESS)) {
    return SS_ERROR_INTERNAL;
  }
  return SS_OK;
}

static SsStatus ss_field_extraction_create(SsFieldSpace *space, Vec source) {
  MPI_Comm communicator = space->mesh->communicator;
  PetscInt storage = 0;
  PetscInt *global = NULL;
  PetscInt *local = NULL;
  PetscInt count = 0;
  IS source_indices = NULL;
  IS target_indices = NULL;
  SsStatus status = SS_OK;
  if (PetscSectionGetStorageSize(space->local_section, &storage) != PETSC_SUCCESS ||
      storage < 0 || (size_t)storage > SIZE_MAX / sizeof(*global)) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK && storage != 0) {
    global = malloc((size_t)storage * sizeof(*global));
    local = malloc((size_t)storage * sizeof(*local));
    if (global == NULL || local == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  for (size_t index = 0U; status == SS_OK && index < space->mesh->point_count;
       ++index) {
    const PetscInt point = space->mesh->points[index].point;
    PetscInt dof, local_offset, global_offset, constrained;
    const PetscInt *indices;
    status = ss_field_point_layout(space, point, &dof, &local_offset, &global_offset,
                                   &constrained, &indices);
    if (status != SS_OK) {
      break;
    }
    const PetscInt base = global_offset < 0 ? -(global_offset + 1) : global_offset;
    PetscInt omitted = 0;
    for (PetscInt slot = 0; slot < dof; ++slot) {
      if (omitted < constrained && indices[omitted] == slot) {
        ++omitted;
      } else {
        global[count] = base + slot - omitted;
        local[count] = local_offset + slot;
        ++count;
      }
    }
  }
  if (status == SS_OK &&
      (VecCreateSeq(PETSC_COMM_SELF, PetscMax(storage, 1), &space->extraction_values) !=
           PETSC_SUCCESS ||
       ISCreateGeneral(PETSC_COMM_SELF, count, global, PETSC_COPY_VALUES,
                       &source_indices) != PETSC_SUCCESS ||
       ISCreateGeneral(PETSC_COMM_SELF, count, local, PETSC_COPY_VALUES,
                       &target_indices) != PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK &&
      VecScatterCreate(source, source_indices, space->extraction_values, target_indices,
                       &space->extraction_scatter) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  (void)ISDestroy(&source_indices);
  (void)ISDestroy(&target_indices);
  free(global);
  free(local);
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    (void)VecScatterDestroy(&space->extraction_scatter);
    (void)VecDestroy(&space->extraction_values);
  }
  return status;
}

SsStatus ss_field_space_extract_local(SsFieldSpace *space, Vec source, Vec *local,
                                      const PetscScalar **values) {
  PetscInt local_size = 0;
  PetscInt global_size = 0;
  SsStatus status = SS_OK;
  if (space == NULL || source == NULL || local == NULL || values == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *local = NULL;
  *values = NULL;
  if (VecGetLocalSize(source, &local_size) != PETSC_SUCCESS ||
      VecGetSize(source, &global_size) != PETSC_SUCCESS ||
      (uint64_t)local_size != space->summary.owned_free_dof_count ||
      (uint64_t)global_size != space->summary.global_free_dof_count) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_status_allreduce_max(space->mesh->communicator, status);
  if (status == SS_OK && space->extraction_scatter == NULL) {
    status = ss_field_extraction_create(space, source);
  }
  if (status == SS_OK &&
      (VecScatterBegin(space->extraction_scatter, source, space->extraction_values,
                       INSERT_VALUES, SCATTER_FORWARD) != PETSC_SUCCESS ||
       VecScatterEnd(space->extraction_scatter, source, space->extraction_values,
                     INSERT_VALUES, SCATTER_FORWARD) != PETSC_SUCCESS ||
       PetscObjectReference((PetscObject)space->extraction_values) != PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    *local = space->extraction_values;
    if (VecGetArrayRead(*local, values) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  return ss_status_allreduce_max(space->mesh->communicator, status);
}

static int ss_field_owned_node_compare(const void *left_raw, const void *right_raw) {
  const SsFieldOwnedNode *left = left_raw;
  const SsFieldOwnedNode *right = right_raw;
  if (left->entity_id != right->entity_id) {
    return left->entity_id < right->entity_id ? -1 : 1;
  }
  return (left->entity_slot > right->entity_slot) -
         (left->entity_slot < right->entity_slot);
}

static SsStatus ss_field_owned_nodes_build(SsFieldSpace *space) {
  const uint32_t components = space->summary.components;
  const SsMesh *mesh = space->mesh;
  size_t count = 0U;
  SsFieldOwnedNode *nodes;
  for (size_t index = 0U; index < mesh->point_count; ++index) {
    PetscInt point = mesh->points[index].point;
    PetscInt dof = 0;
    if (ss_mesh_point_is_ghost(mesh, mesh->dm, point)) {
      continue;
    }
    if (PetscSectionGetDof(space->local_section, point, &dof) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    count += (size_t)dof / components;
  }
  nodes = calloc(count == 0U ? 1U : count, sizeof(*nodes));
  if (nodes == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  count = 0U;
  for (size_t index = 0U; index < mesh->point_count; ++index) {
    const SsPointEntry *entry = &mesh->points[index];
    PetscInt dof, local_offset, global_offset, constrained;
    const PetscInt *indices;
    PetscInt omitted = 0;
    if (ss_mesh_point_is_ghost(mesh, mesh->dm, entry->point)) {
      continue;
    }
    SsStatus status = ss_field_point_layout(space, entry->point, &dof, &local_offset,
                                            &global_offset, &constrained, &indices);
    if (status != SS_OK) {
      free(nodes);
      return status;
    }
    for (PetscInt slot = 0; slot < dof; slot += (PetscInt)components) {
      SsFieldOwnedNode *node = &nodes[count++];
      node->entity_id = entry->stable_id;
      node->entity_slot = (uint32_t)(slot / (PetscInt)components);
      node->local_offset = local_offset + slot;
      for (uint32_t component = 0U; component < components; ++component) {
        const PetscInt raw = slot + (PetscInt)component;
        if (omitted < constrained && indices[omitted] == raw) {
          const SsStrongConstraint key = {entry->stable_id, node->entity_slot,
                                          component, 0.0};
          const SsStrongConstraint *prescribed =
              bsearch(&key, space->constraints, space->constraint_count,
                      sizeof(*space->constraints), ss_field_constraint_compare);
          if (prescribed == NULL) {
            free(nodes);
            return SS_ERROR_INTEGRITY;
          }
          node->global_indices[component] = -1;
          node->prescribed[component] = prescribed->value;
          ++omitted;
        } else {
          node->global_indices[component] = global_offset + raw - omitted;
        }
      }
    }
  }
  ss_sort(nodes, count, sizeof(*nodes), ss_field_owned_node_compare);
  space->owned_nodes = nodes;
  space->owned_node_count = count;
  return SS_OK;
}

SsStatus ss_field_space_owned_nodes(SsFieldSpace *space, const SsFieldOwnedNode **nodes,
                                    size_t *count) {
  if (space == NULL || nodes == NULL || count == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (space->owned_nodes == NULL) {
    SsStatus status = ss_field_owned_nodes_build(space);
    if (status != SS_OK) {
      return status;
    }
  }
  *nodes = space->owned_nodes;
  *count = space->owned_node_count;
  return SS_OK;
}

SsStatus ss_field_space_nodal_maximum(SsFieldSpace *space, Vec state,
                                      PetscReal *maximum) {
  const SsFieldOwnedNode *nodes = NULL;
  const PetscScalar *values = NULL;
  PetscInt start = 0, end = 0;
  int acquired = 0;
  PetscReal local_maximum = 0.0;
  size_t count = 0U;
  SsStatus status;
  if (space == NULL || state == NULL || maximum == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_field_space_owned_nodes(space, &nodes, &count);
  if (status == SS_OK && (VecGetOwnershipRange(state, &start, &end) != PETSC_SUCCESS ||
                          VecGetArrayRead(state, &values) != PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  acquired = status == SS_OK;
  for (size_t index = 0U; status == SS_OK && index < count; ++index) {
    const SsFieldOwnedNode *node = &nodes[index];
    PetscReal squared_norm = 0.0;
    for (uint32_t component = 0U; component < space->summary.components; ++component) {
      const PetscInt global = node->global_indices[component];
      PetscReal value;
      if (global < 0) {
        value = (PetscReal)node->prescribed[component];
      } else if (global >= start && global < end) {
        value = PetscRealPart(values[global - start]);
      } else {
        status = SS_ERROR_INTEGRITY;
        break;
      }
      squared_norm += value * value;
    }
    if (!isfinite((double)squared_norm)) {
      status = SS_ERROR_INTEGRITY;
    } else {
      local_maximum = fmax(local_maximum, sqrt(squared_norm));
    }
  }
  if (acquired != 0 && VecRestoreArrayRead(state, &values) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(space->mesh->communicator, status);
  if (status == SS_OK && MPI_Allreduce(&local_maximum, maximum, 1, MPIU_REAL, MPI_MAX,
                                       space->mesh->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  return status;
}
