/* Construct and validate discretized fields and essential constraints.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "mesh_internal.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/core/sort.h"

#include "slope_internal/diagnostics/diagnostics.h"
#include "slope_internal/fe/fe.h"

#include <petscdm.h>
#include <petscdmplex.h>
#include <petsclog.h>
#include <petscsection.h>
#include <petscsf.h>

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void ss_field_constraint_range(const SsFieldSpec *spec, SsEntityId stable_id,
                                      size_t *begin, size_t *end) {
  size_t left = 0U;
  size_t right = spec->constraint_count;
  while (left < right) {
    const size_t middle = left + (right - left) / 2U;
    if (spec->constraints[middle].stable_entity_id < stable_id) {
      left = middle + 1U;
    } else {
      right = middle;
    }
  }
  *begin = left;
  right = spec->constraint_count;
  while (left < right) {
    const size_t middle = left + (right - left) / 2U;
    if (spec->constraints[middle].stable_entity_id <= stable_id) {
      left = middle + 1U;
    } else {
      right = middle;
    }
  }
  *end = left;
}

static SsStatus ss_field_section_constraints(SsFieldSpace *space,
                                             const SsFieldSpec *spec,
                                             PetscInt chart_start, PetscInt chart_end) {
  PetscInt point;
  for (point = chart_start; point < chart_end; ++point) {
    SsEntityId stable_id = 0U;
    size_t begin = 0U;
    size_t end = 0U;
    size_t count;
    if (ss_mesh_cached_stable_id(space->mesh, point, &stable_id) != SS_OK) {
      continue;
    }
    ss_field_constraint_range(spec, stable_id, &begin, &end);
    count = end - begin;
    if (count != 0U &&
        (PetscSectionSetConstraintDof(space->local_section, point, (PetscInt)count) !=
             PETSC_SUCCESS ||
         PetscSectionSetFieldConstraintDof(space->local_section, point, 0,
                                           (PetscInt)count) != PETSC_SUCCESS)) {
      return SS_ERROR_INTERNAL;
    }
  }
  if (PetscSectionSetUp(space->local_section) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  for (point = chart_start; point < chart_end; ++point) {
    SsEntityId stable_id = 0U;
    PetscInt dof = 0;
    PetscInt constraint_indices[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION];
    size_t begin = 0U;
    size_t end = 0U;
    size_t count = 0U;
    size_t constraint;
    if (ss_mesh_cached_stable_id(space->mesh, point, &stable_id) != SS_OK) {
      continue;
    }
    if (PetscSectionGetDof(space->local_section, point, &dof) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    ss_field_constraint_range(spec, stable_id, &begin, &end);
    for (constraint = begin; constraint < end; ++constraint) {
      const SsStrongConstraint *item = &spec->constraints[constraint];
      const uint64_t local_index =
          (uint64_t)item->entity_slot * (uint64_t)spec->components +
          (uint64_t)item->component;
      if (item->component >= spec->components || local_index >= (uint64_t)dof ||
          count >= sizeof(constraint_indices) / sizeof(constraint_indices[0])) {
        return SS_ERROR_INVALID_ARGUMENT;
      }
      constraint_indices[count++] = (PetscInt)local_index;
    }
    if (count != 0U &&
        (PetscSectionSetConstraintIndices(space->local_section, point,
                                          constraint_indices) != PETSC_SUCCESS ||
         PetscSectionSetFieldConstraintIndices(space->local_section, point, 0,
                                               constraint_indices) != PETSC_SUCCESS)) {
      return SS_ERROR_INTERNAL;
    }
  }
  return SS_OK;
}

static SsStatus ss_field_identity(const SsMesh *mesh, const SsFieldSpec *spec,
                                  const SsFeDescriptor *descriptor,
                                  uint64_t constraint_count,
                                  const char *constraint_digest,
                                  char identity[SS_ID_TEXT_CAPACITY]) {
  char projection[768];
  int written;
  written =
      snprintf(projection, sizeof(projection),
               "component=N-FE-field-space\ndimension=%u\norder=%u\ncomponents=%u\n"
               "basis-id=%s\ntopology-generation=%llu\n"
               "coordinate-generation=%llu\nlabel-generation=%llu\n"
               "partition-generation=%llu\n"
               "stable-entities=%s\ntopology=%s\ncompiled-closures=%s\n"
               "coordinates=%s\nlabels=%s\n"
               "constraints=%llu\nconstraint-digest=%s\n",
               mesh->dimension, spec->order, spec->components, descriptor->id,
               (unsigned long long)mesh->summary.topology_generation,
               (unsigned long long)mesh->summary.coordinate_generation,
               (unsigned long long)mesh->summary.label_generation,
               (unsigned long long)mesh->summary.partition_generation,
               mesh->summary.stable_id_digest, mesh->summary.topology_digest,
               mesh->summary.closure_digest, mesh->summary.coordinate_digest,
               mesh->summary.label_digest, (unsigned long long)constraint_count,
               constraint_digest);
  if (written < 0 || (size_t)written >= sizeof(projection)) {
    return SS_ERROR_LIMIT;
  }
  return ss_identity_from_text("field-space", projection, identity);
}

static SsStatus ss_mesh_field_space_create_mode(SsMesh *mesh, const SsFieldSpec *spec,
                                                int distributed, SsFieldSpace **space) {
  SsFieldSpace *created = NULL;
  SsFieldSpec normalized = {0U, 0U, 0U, NULL};
  const SsFeDescriptor *descriptor = NULL;
  PetscInt chart_start = 0;
  PetscInt chart_end = 0;
  PetscInt point;
  PetscSF point_sf = NULL;
  uint64_t local_dofs = 0U;
  uint64_t global_dofs = 0U;
  uint64_t global_constraints = 0U;
  char constraint_digest[SS_SHA256_HEX_CAPACITY] = {0};
  uint64_t local_free_dofs = 0U;
  uint64_t global_free_dofs = 0U;
  SsStatus status = SS_OK;
  if (mesh == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (space != NULL) {
    *space = NULL;
  }
  if (spec == NULL || space == NULL || spec->components == 0U ||
      spec->components > SS_FE_MAX_DIMENSION ||
      (spec->constraint_count != 0U && spec->constraints == NULL) ||
      ss_fe_descriptor_find(mesh->dimension, spec->order, &descriptor) != SS_OK) {
    status = SS_ERROR_INVALID_ARGUMENT;
  } else if (spec->constraint_count > SIZE_MAX / sizeof(*created->constraints)) {
    status = SS_ERROR_OVERFLOW;
  } else if (mesh->active_field_spaces == SIZE_MAX) {
    status = SS_ERROR_OVERFLOW;
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  created = (SsFieldSpace *)calloc(1U, sizeof(*created));
  if (created == NULL) {
    status = SS_ERROR_OUT_OF_MEMORY;
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  created->mesh = mesh;
  created->descriptor = descriptor;
  created->constraint_count = spec->constraint_count;
  if (spec->constraint_count != 0U) {
    created->constraints = (SsStrongConstraint *)malloc(spec->constraint_count *
                                                        sizeof(*created->constraints));
    if (created->constraints == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    } else {
      memcpy(created->constraints, spec->constraints,
             spec->constraint_count * sizeof(*created->constraints));
      ss_sort(created->constraints, spec->constraint_count,
              sizeof(*created->constraints), ss_field_constraint_compare);
    }
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  status = ss_field_constraints_prepare(created, spec, distributed, &global_constraints,
                                        constraint_digest);
  if (status != SS_OK) {
    goto cleanup;
  }
  normalized = *spec;
  normalized.constraints = created->constraints;
  normalized.constraint_count = created->constraint_count;
  if (DMPlexGetChart(mesh->dm, &chart_start, &chart_end) != PETSC_SUCCESS ||
      PetscSectionCreate(mesh->communicator, &created->local_section) !=
          PETSC_SUCCESS ||
      PetscSectionSetNumFields(created->local_section, 1) != PETSC_SUCCESS ||
      PetscSectionSetFieldComponents(created->local_section, 0,
                                     (PetscInt)spec->components) != PETSC_SUCCESS ||
      PetscSectionSetChart(created->local_section, chart_start, chart_end) !=
          PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  for (point = chart_start; status == SS_OK && point < chart_end; ++point) {
    PetscInt depth = -1;
    PetscInt dof;
    if (DMPlexGetPointDepth(mesh->dm, point, &depth) != PETSC_SUCCESS || depth < 0) {
      status = SS_ERROR_INTERNAL;
      break;
    }
    dof = (PetscInt)(ss_mesh_slots_for_depth(mesh->dimension, spec->order,
                                             (uint32_t)depth) *
                     spec->components);
    if (PetscSectionSetDof(created->local_section, point, dof) != PETSC_SUCCESS ||
        PetscSectionSetFieldDof(created->local_section, point, 0, dof) !=
            PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
      break;
    }
  }
  if (status == SS_OK) {
    status = ss_field_section_constraints(created, &normalized, chart_start, chart_end);
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (DMGetPointSF(mesh->dm, &point_sf) != PETSC_SUCCESS ||
      PetscSectionCreateGlobalSection(created->local_section, point_sf, PETSC_TRUE,
                                      PETSC_FALSE, PETSC_FALSE,
                                      &created->global_section) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  for (point = chart_start; status == SS_OK && point < chart_end; ++point) {
    PetscInt dof = 0;
    PetscInt constrained_dof = 0;
    if (ss_mesh_point_is_ghost(mesh, mesh->dm, point) == 0) {
      if (PetscSectionGetDof(created->local_section, point, &dof) != PETSC_SUCCESS ||
          PetscSectionGetConstraintDof(created->local_section, point,
                                       &constrained_dof) != PETSC_SUCCESS ||
          constrained_dof < 0 || constrained_dof > dof) {
        status = SS_ERROR_INTERNAL;
        break;
      }
      local_dofs += (uint64_t)dof;
      local_free_dofs += (uint64_t)(dof - constrained_dof);
    }
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (MPI_Allreduce(&local_dofs, &global_dofs, 1, MPI_UINT64_T, MPI_SUM,
                    mesh->communicator) != MPI_SUCCESS ||
      MPI_Allreduce(&local_free_dofs, &global_free_dofs, 1, MPI_UINT64_T, MPI_SUM,
                    mesh->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  created->summary.order = spec->order;
  created->summary.components = spec->components;
  created->summary.vertex_slots = 1U;
  created->summary.edge_interior_slots = spec->order - 1U;
  created->summary.face_interior_slots =
      mesh->dimension == 3U ? (spec->order - 1U) * (spec->order - 2U) / 2U : 0U;
  created->summary.cell_interior_slots =
      mesh->dimension == 2U
          ? (spec->order - 1U) * (spec->order - 2U) / 2U
          : (spec->order - 1U) * (spec->order - 2U) * (spec->order - 3U) / 6U;
  created->summary.global_dof_count = global_dofs;
  created->summary.global_free_dof_count = global_free_dofs;
  created->summary.owned_free_dof_count = local_free_dofs;
  created->summary.constrained_dof_count = global_constraints;
  created->summary.section_generation = 1U;
  created->summary.constraint_generation = 1U;
  {
    const int written =
        snprintf(created->summary.basis_id, sizeof(created->summary.basis_id), "%s",
                 descriptor->id);
    if (written < 0 || (size_t)written >= sizeof(created->summary.basis_id)) {
      status = SS_ERROR_LIMIT;
    }
  }
  if (status != SS_OK) {
    goto cleanup;
  }
  status = ss_field_identity(mesh, &normalized, descriptor, global_constraints,
                             constraint_digest, created->summary.identity_digest);
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status == SS_OK) {
    status = ss_field_incident_cell_plans_build(created);
  }
  if (status == SS_OK) {
    status = ss_field_owned_cell_plans_build(created);
  }
  status = ss_status_allreduce_max(mesh->communicator, status);

cleanup:
  if (status != SS_OK) {
    (void)ss_field_space_destroy(&created);
    return status;
  }
  ++mesh->active_field_spaces;
  created->registered_with_mesh = 1;
  *space = created;
  return SS_OK;
}

SsStatus ss_mesh_field_space_create(SsMesh *mesh, const SsFieldSpec *spec,
                                    SsFieldSpace **space) {
  return ss_mesh_field_space_create_mode(mesh, spec, 0, space);
}

SsStatus ss_mesh_field_space_create_local(SsMesh *mesh, const SsFieldSpec *spec,
                                          SsFieldSpace **space) {
  return ss_mesh_field_space_create_mode(mesh, spec, 1, space);
}

SsStatus ss_field_space_summary(const SsFieldSpace *space, SsFieldSummary *summary) {
  if (space == NULL || summary == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *summary = space->summary;
  return SS_OK;
}

SsStatus ss_field_space_owned_free_dof_map(const SsFieldSpace *space,
                                           SsEntityId *stable_ids, size_t *local_slots,
                                           size_t capacity, size_t *count) {
  uint64_t local_count;
  uint64_t ownership_begin = 0U;
  unsigned char *seen = NULL;
  int local_query;
  int global_query = 0;
  size_t cell;
  size_t index;
  SsStatus status = SS_OK;
  if (space == NULL || count == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  local_count = space->summary.owned_free_dof_count;
  if (local_count > (uint64_t)SIZE_MAX) {
    status = SS_ERROR_OVERFLOW;
  }
  status = ss_status_allreduce_max(space->mesh->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  *count = (size_t)local_count;
  local_query = stable_ids == NULL && local_slots == NULL && capacity == 0U;
  if (MPI_Allreduce(&local_query, &global_query, 1, MPI_INT, MPI_MIN,
                    space->mesh->communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (global_query != 0) {
    return SS_OK;
  }
  if ((stable_ids == NULL) != (local_slots == NULL) ||
      (stable_ids == NULL && capacity != 0U) ||
      (local_count != 0U && stable_ids == NULL) || capacity < (size_t)local_count) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_status_allreduce_max(space->mesh->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  if (MPI_Exscan(&local_count, &ownership_begin, 1, MPI_UINT64_T, MPI_SUM,
                 space->mesh->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (space->mesh->rank == 0) {
    ownership_begin = 0U;
  }
  if (status == SS_OK && local_count != 0U) {
    seen = (unsigned char *)calloc((size_t)local_count, sizeof(*seen));
    if (seen == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(space->mesh->communicator, status);
  if (status != SS_OK) {
    free(seen);
    return status;
  }
  for (cell = 0U; status == SS_OK && cell < space->incident_cell_plan_count; ++cell) {
    const SsCellPlan *pack = &space->incident_cell_plans[cell];
    size_t closure;
    for (closure = 0U; closure < pack->closure_count; ++closure) {
      const SsEntityId entity_id = pack->field_entity_ids[closure];
      const uint32_t entity_slot = pack->field_entity_slots[closure];
      const uint32_t component = pack->field_components[closure];
      const PetscInt global_index = pack->global_indices[closure];
      size_t local_slot;
      SsEntityId stable_id;
      if (pack->owned[closure] == 0U || pack->constrained[closure] != 0U) {
        continue;
      }
      if (global_index < 0 || (uint64_t)global_index < ownership_begin ||
          (uint64_t)global_index - ownership_begin >= local_count ||
          component >= space->summary.components) {
        status = SS_ERROR_INTEGRITY;
        break;
      }
      local_slot = (size_t)((uint64_t)global_index - ownership_begin);
      stable_id =
          ss_mesh_hash_u64(UINT64_C(1469598103934665603), UINT64_C(0x73732d646f662d31));
      stable_id = ss_mesh_hash_u64(stable_id, entity_id);
      stable_id = ss_mesh_hash_u64(stable_id, (uint64_t)entity_slot);
      stable_id = ss_mesh_hash_u64(stable_id, (uint64_t)component);
      if (stable_id == 0U) {
        status = SS_ERROR_INTEGRITY;
        break;
      }
      if (seen[local_slot] != 0U) {
        if (stable_ids[local_slot] != stable_id ||
            local_slots[local_slot] != local_slot) {
          status = SS_ERROR_INTEGRITY;
          break;
        }
      } else {
        stable_ids[local_slot] = stable_id;
        local_slots[local_slot] = local_slot;
        seen[local_slot] = 1U;
      }
    }
  }
  for (index = 0U; status == SS_OK && index < (size_t)local_count; ++index) {
    if (seen[index] == 0U) {
      status = SS_ERROR_INTEGRITY;
    }
  }
  free(seen);
  return ss_status_allreduce_max(space->mesh->communicator, status);
}

SsStatus ss_field_space_destroy(SsFieldSpace **space) {
  SsStatus status = SS_OK;
  if (space == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (*space == NULL) {
    return SS_OK;
  }
  if (PetscSectionDestroy(&(*space)->global_section) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (PetscSectionDestroy(&(*space)->local_section) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if ((*space)->registered_with_mesh != 0) {
    if ((*space)->mesh == NULL || (*space)->mesh->active_field_spaces == 0U) {
      status = SS_ERROR_INTERNAL;
    } else {
      --(*space)->mesh->active_field_spaces;
    }
    (*space)->registered_with_mesh = 0;
  }
  if (PetscSFDestroy(&(*space)->constraint_sf) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (VecScatterDestroy(&(*space)->extraction_scatter) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (VecDestroy(&(*space)->extraction_values) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  free((*space)->owned_nodes);
  free((*space)->constraints);
  free((*space)->owned_cell_plans);
  free((*space)->incident_cell_plans);
  free((*space)->incident_cell_plan_storage);
  free(*space);
  *space = NULL;
  return status;
}
