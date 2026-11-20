/* Match compiled memberships to resident points and form essential closures.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "mesh_internal.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/mesh/distribution.h"

#include <petscdmplex.h>
#include <petscsf.h>

#include <stdio.h>
#include <stdlib.h>

static void ss_mesh_essential_name(uint32_t rule, char name[40]) {
  (void)snprintf(name, 40U, "ss-essential-%u", rule);
}

static uint32_t ss_mesh_kind_dimension(const SsMesh *mesh, SsMeshEntityKind kind) {
  switch (kind) {
  case SS_MESH_ENTITY_VERTEX:
    return 0U;
  case SS_MESH_ENTITY_EDGE:
    return 1U;
  case SS_MESH_ENTITY_FACET:
    return mesh->dimension - 1U;
  case SS_MESH_ENTITY_CELL:
    return mesh->dimension;
  }
  return UINT32_MAX;
}

SsStatus ss_mesh_entity_dimension(const SsMesh *mesh, SsEntityId id,
                                  uint32_t *dimension) {
  PetscInt point = -1;
  size_t offset = 0U;
  if (mesh == NULL || dimension == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (ss_mesh_point_from_stable_id_any(mesh, id, &point) != SS_OK ||
      ss_mesh_cached_point_offset(mesh, point, &offset) != SS_OK) {
    return SS_ERROR_NOT_FOUND;
  }
  *dimension = ss_mesh_kind_dimension(mesh, mesh->cached_kinds[offset]);
  return *dimension == UINT32_MAX ? SS_ERROR_INTEGRITY : SS_OK;
}

static SsStatus ss_mesh_mark_closure(DM dm, PetscInt point, PetscInt *flags) {
  PetscInt count = 0;
  PetscInt *closure = NULL;
  if (DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &count, &closure) !=
      PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  for (PetscInt i = 0; i < count; ++i) {
    flags[closure[2 * i]] = 1;
  }
  return DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &count, &closure) ==
                 PETSC_SUCCESS
             ? SS_OK
             : SS_ERROR_INTERNAL;
}

/* An explicit target or an exterior facet may only be resident on some of
 * the ranks that have a point in its closure. Take the union on the point SF. */
static SsStatus ss_mesh_sync_flags(const SsMesh *mesh, PetscInt *flags) {
  PetscSF sf = NULL;
  PetscInt roots = -1;
  PetscInt *owners = NULL;
  SsStatus status = SS_OK;
  if (DMGetPointSF(mesh->dm, &sf) != PETSC_SUCCESS ||
      PetscSFGetGraph(sf, &roots, NULL, NULL, NULL) != PETSC_SUCCESS ||
      (roots >= 0 && roots != mesh->cache_chart_end)) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK && roots >= 0) {
    const size_t count = roots == 0 ? 1U : (size_t)roots;
    owners = calloc(count, sizeof(*owners));
    if (owners == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status == SS_OK && roots >= 0) {
    if (PetscSFReduceBegin(sf, MPIU_INT, flags, owners, MPI_MAX) != PETSC_SUCCESS ||
        PetscSFReduceEnd(sf, MPIU_INT, flags, owners, MPI_MAX) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
    for (PetscInt point = 0; point < roots; ++point) {
      flags[point] = owners[point] = PetscMax(flags[point], owners[point]);
    }
    if (PetscSFBcastBegin(sf, MPIU_INT, owners, flags, MPI_REPLACE) != PETSC_SUCCESS ||
        PetscSFBcastEnd(sf, MPIU_INT, owners, flags, MPI_REPLACE) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  free(owners);
  return ss_status_allreduce_max(mesh->communicator, status);
}

static SsStatus ss_mesh_exterior_flags(const SsMesh *mesh, PetscInt *exterior) {
  DMLabel boundary = NULL;
  SsStatus status = SS_OK;
  if (DMGetLabel(mesh->dm, "ss-explicit-boundary", &boundary) != PETSC_SUCCESS ||
      boundary == NULL) {
    status = SS_ERROR_INTEGRITY;
  }
  for (size_t i = 0U; status == SS_OK && i < mesh->point_count; ++i) {
    const SsPointEntry *entry = &mesh->points[i];
    if (entry->kind == SS_MESH_ENTITY_FACET) {
      PetscInt marked = -1;
      if (DMLabelGetValue(boundary, entry->point, &marked) != PETSC_SUCCESS) {
        status = SS_ERROR_INTERNAL;
      } else if (marked == 1) {
        status = ss_mesh_mark_closure(mesh->dm, entry->point, exterior);
      }
    }
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  return status == SS_OK ? ss_mesh_sync_flags(mesh, exterior) : status;
}

static SsStatus ss_mesh_infer_intermediate(const SsMesh *mesh, const PetscInt *selected,
                                           PetscInt *exterior) {
  SsStatus status = ss_mesh_exterior_flags(mesh, exterior);
  for (size_t i = 0U; status == SS_OK && i < mesh->point_count; ++i) {
    const SsPointEntry *entry = &mesh->points[i];
    const PetscInt point = entry->point;
    if (entry->kind == SS_MESH_ENTITY_VERTEX) {
      exterior[point] = selected[point];
    } else if (exterior[point] != 0) {
      PetscInt count = 0;
      PetscInt *closure = NULL;
      if (DMPlexGetTransitiveClosure(mesh->dm, point, PETSC_TRUE, &count, &closure) !=
          PETSC_SUCCESS) {
        status = SS_ERROR_INTERNAL;
        break;
      }
      for (PetscInt j = 0; j < count; ++j) {
        const PetscInt inner = closure[2 * j];
        if (mesh->cached_kinds[inner - mesh->cache_chart_start] ==
                SS_MESH_ENTITY_VERTEX &&
            selected[inner] == 0) {
          exterior[point] = 0;
          break;
        }
      }
      if (DMPlexRestoreTransitiveClosure(mesh->dm, point, PETSC_TRUE, &count,
                                         &closure) != PETSC_SUCCESS) {
        status = SS_ERROR_INTERNAL;
      }
    }
  }
  return status;
}

static SsStatus ss_mesh_support_match(const SsMesh *mesh, const SsEntityId *ids,
                                      size_t count, PetscInt *selected,
                                      int *vertex_only) {
  const size_t capacity = mesh->point_count == 0U ? 1U : mesh->point_count;
  SsMeshIdValue *source = calloc(count == 0U ? 1U : count, sizeof(*source));
  SsEntityId *query = calloc(capacity, sizeof(*query));
  uint32_t *values = calloc(capacity, sizeof(*values));
  size_t query_count = 0U;
  int nonvertex = 0, any_nonvertex = 0;
  SsStatus status = source == NULL || query == NULL || values == NULL
                        ? SS_ERROR_OUT_OF_MEMORY
                        : SS_OK;
  if (status == SS_OK) {
    for (size_t i = 0U; i < count; ++i) {
      source[i] = (SsMeshIdValue){ids[i], 1U};
    }
    for (; query_count < mesh->point_count &&
           mesh->points[query_count].kind != SS_MESH_ENTITY_CELL;
         ++query_count) {
      query[query_count] = mesh->points[query_count].stable_id;
    }
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status == SS_OK) {
    status =
        ss_mesh_join_ids(mesh->communicator, source, count, query, query_count, values);
  }
  for (size_t i = 0U; status == SS_OK && i < query_count; ++i) {
    if (values[i] != 0U) {
      selected[mesh->points[i].point] = 1;
      nonvertex |= mesh->points[i].kind != SS_MESH_ENTITY_VERTEX;
    }
  }
  if (status == SS_OK && MPI_Allreduce(&nonvertex, &any_nonvertex, 1, MPI_INT, MPI_MAX,
                                       mesh->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  *vertex_only = any_nonvertex == 0;
  free(source);
  free(query);
  free(values);
  return ss_status_allreduce_max(mesh->communicator, status);
}

SsStatus ss_mesh_essential_support_set(SsMesh *mesh, uint32_t rule,
                                       const SsEntityId *ids, size_t count) {
  if (mesh == NULL || (count != 0U && ids == NULL)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  const size_t points = mesh->cache_chart_end == 0 ? 1U : (size_t)mesh->cache_chart_end;
  PetscInt *selected = calloc(points, sizeof(*selected));
  PetscInt *closure = calloc(points, sizeof(*closure));
  int vertex_only = 0;
  char name[40];
  SsStatus status =
      selected == NULL || closure == NULL ? SS_ERROR_OUT_OF_MEMORY : SS_OK;
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status == SS_OK) {
    status = ss_mesh_support_match(mesh, ids, count, selected, &vertex_only);
  }
  if (status == SS_OK && vertex_only != 0) {
    status = ss_mesh_infer_intermediate(mesh, selected, closure);
  } else if (status == SS_OK) {
    for (size_t i = 0U; status == SS_OK && i < mesh->point_count; ++i) {
      const PetscInt point = mesh->points[i].point;
      if (selected[point] != 0) {
        status = ss_mesh_mark_closure(mesh->dm, point, closure);
      }
    }
    status = ss_status_allreduce_max(mesh->communicator, status);
    if (status == SS_OK) {
      status = ss_mesh_sync_flags(mesh, closure);
    }
  }
  ss_mesh_essential_name(rule, name);
  if (status == SS_OK) {
    status = ss_mesh_create_label(mesh->dm, name);
  }
  for (size_t i = 0U; status == SS_OK && i < mesh->point_count; ++i) {
    const PetscInt point = mesh->points[i].point;
    if (closure[point] != 0 &&
        DMSetLabelValue(mesh->dm, name, point, 1) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  free(selected);
  free(closure);
  return ss_status_allreduce_max(mesh->communicator, status);
}

SsStatus ss_mesh_essential_members(const SsMesh *mesh, uint32_t rule,
                                   SsMeshEntityRef **members, size_t *count) {
  char name[40];
  DMLabel label = NULL;
  IS points = NULL;
  PetscInt size = 0;
  const PetscInt *indices = NULL;
  SsStatus status = SS_OK;
  *members = NULL;
  *count = 0U;
  ss_mesh_essential_name(rule, name);
  if (DMGetLabel(mesh->dm, name, &label) != PETSC_SUCCESS || label == NULL ||
      DMLabelGetStratumIS(label, 1, &points) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (points != NULL) {
    if (ISGetLocalSize(points, &size) != PETSC_SUCCESS ||
        ISGetIndices(points, &indices) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    } else if (size != 0) {
      *members = calloc((size_t)size, sizeof(**members));
      if (*members == NULL) {
        status = SS_ERROR_OUT_OF_MEMORY;
      }
    }
  }
  for (PetscInt i = 0; status == SS_OK && i < size; ++i) {
    size_t offset = 0U;
    status = ss_mesh_cached_point_offset(mesh, indices[i], &offset);
    if (status == SS_OK) {
      (*members)[i] =
          (SsMeshEntityRef){mesh->cached_stable_ids[offset],
                            ss_mesh_kind_dimension(mesh, mesh->cached_kinds[offset])};
    }
  }
  if (indices != NULL) {
    (void)ISRestoreIndices(points, &indices);
  }
  (void)ISDestroy(&points);
  if (status == SS_OK) {
    *count = (size_t)size;
  } else {
    free(*members);
    *members = NULL;
  }
  return status;
}

int ss_mesh_essential_contains(const SsMesh *mesh, uint32_t rule, SsEntityId id) {
  char name[40];
  PetscInt point = -1, marked = -1;
  ss_mesh_essential_name(rule, name);
  return ss_mesh_point_from_stable_id_any(mesh, id, &point) == SS_OK &&
         DMGetLabelValue(mesh->dm, name, point, &marked) == PETSC_SUCCESS &&
         marked == 1;
}
