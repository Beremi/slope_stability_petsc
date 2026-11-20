/* Mesh creation, distribution, renumbering, and lifetime.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "mesh_internal.h"
#include "slope_internal/core/collective.h"

#include "ownership.h"
#include "slope_internal/diagnostics/diagnostics.h"
#include "slope_internal/fe/fe.h"

#include <petscdm.h>
#include <petscdmplex.h>
#include <petscis.h>
#include <petsclog.h>
#include <petscpartitioner.h>
#include <petscsf.h>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static PetscLogEvent ss_mesh_validate_event = -1;
static PetscLogEvent ss_mesh_plex_create_event = -1;
static PetscLogEvent ss_mesh_metadata_event = -1;
static PetscLogEvent ss_mesh_coordinate_event = -1;
static PetscLogEvent ss_mesh_digest_event = -1;
static PetscLogEvent ss_mesh_distribute_event = -1;
static PetscLogEvent ss_mesh_maps_event = -1;
static PetscLogEvent ss_mesh_ownership_audit_event = -1;
static PetscLogEvent ss_mesh_geometry_audit_event = -1;
static PetscLogEvent ss_mesh_summary_event = -1;
static PetscLogEvent ss_mesh_migration_audit_event = -1;
static PetscLogEvent ss_mesh_facet_plans_event = -1;
static PetscLogEvent ss_mesh_attach_vertices_event = -1;
static PetscLogEvent ss_mesh_attach_cells_event = -1;
static PetscLogEvent ss_mesh_attach_entities_event = -1;
static PetscLogEvent ss_mesh_point_index_event = -1;
static PetscLogEvent ss_mesh_entity_index_event = -1;
static PetscLogEvent ss_mesh_semantic_labels_event = -1;
static PetscLogEvent ss_mesh_semantic_audit_event = -1;

static void ss_mesh_profile_events_register(void) {
  if (ss_mesh_validate_event < 0) {
    (void)PetscLogEventRegister("SlopeMeshValidate", 0, &ss_mesh_validate_event);
    (void)PetscLogEventRegister("SlopePlexCreate", 0, &ss_mesh_plex_create_event);
    (void)PetscLogEventRegister("SlopeMeshMetadata", 0, &ss_mesh_metadata_event);
    (void)PetscLogEventRegister("SlopeMeshCoordinate", 0, &ss_mesh_coordinate_event);
    (void)PetscLogEventRegister("SlopeMeshDigest", 0, &ss_mesh_digest_event);
    (void)PetscLogEventRegister("SlopeMeshDistribute", 0, &ss_mesh_distribute_event);
    (void)PetscLogEventRegister("SlopeMeshMaps", 0, &ss_mesh_maps_event);
    (void)PetscLogEventRegister("SlopeOwnershipAudit", 0,
                                &ss_mesh_ownership_audit_event);
    (void)PetscLogEventRegister("SlopeGeometryAudit", 0, &ss_mesh_geometry_audit_event);
    (void)PetscLogEventRegister("SlopeMeshSummary", 0, &ss_mesh_summary_event);
    (void)PetscLogEventRegister("SlopeMigrationAudit", 0,
                                &ss_mesh_migration_audit_event);
    (void)PetscLogEventRegister("SlopeFacetPlans", 0, &ss_mesh_facet_plans_event);
    (void)PetscLogEventRegister("SlopeAttachVertices", 0,
                                &ss_mesh_attach_vertices_event);
    (void)PetscLogEventRegister("SlopeAttachCells", 0, &ss_mesh_attach_cells_event);
    (void)PetscLogEventRegister("SlopeAttachEntities", 0,
                                &ss_mesh_attach_entities_event);
    (void)PetscLogEventRegister("SlopePointIndex", 0, &ss_mesh_point_index_event);
    (void)PetscLogEventRegister("SlopeEntityIndex", 0, &ss_mesh_entity_index_event);
    (void)PetscLogEventRegister("SlopeSemanticLabels", 0,
                                &ss_mesh_semantic_labels_event);
    (void)PetscLogEventRegister("SlopeSemanticAudit", 0, &ss_mesh_semantic_audit_event);
  }
}

_Static_assert(sizeof(double) == sizeof(uint64_t),
               "mesh identities require binary64-sized doubles");

_Static_assert(sizeof(SsIntermediateAuditRecord) == 6U * sizeof(uint64_t),
               "intermediate audit records must be packed uint64 fields");

SsStatus ss_mesh_create(MPI_Comm communicator, const SsMeshInput *input, SsMesh **mesh,
                        SsMeshError *error) {
  int rank = 0;
  int size = 0;
  SsStatus status = SS_OK;
  PetscInt local_cell_count = 0;
  PetscInt local_vertex_count = 0;
  PetscInt global_vertex_count = 0;
  PetscInt corners;
  PetscInt *cells = NULL;
  PetscReal *coordinates = NULL;
  PetscSF vertex_sf = NULL;
  PetscInt *vertices_adjacent = NULL;
  SsEntityId *adjacent_ids = NULL;
  PetscInt adjacent_count = 0;
  SsPointIdEntry *point_index = NULL;
  size_t point_index_count = 0U;
  SsEntityVertexEntry *entity_index = NULL;
  size_t entity_index_count = 0U;
  DM initial = NULL;
  DM distributed = NULL;
  SsMesh *created = NULL;
  const SsFeDescriptor *descriptor = NULL;
  SsMeshSummary digest_summary = {0};
  size_t index;
  if (mesh == NULL || error == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *mesh = NULL;
  memset(error, 0, sizeof(*error));
  if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(communicator, &size) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  ss_mesh_profile_events_register();
  (void)ss_memory_view(communicator, "mesh_begin");
  (void)PetscLogEventBegin(ss_mesh_validate_event, 0, 0, 0, 0);
  status = ss_mesh_validate_collective(communicator, input, &descriptor);
  (void)PetscLogEventEnd(ss_mesh_validate_event, 0, 0, 0, 0);
  if (status != SS_OK) {
    return status;
  }
  if (ss_checked_u64_to_petsc((uint64_t)input->cell_count, &local_cell_count) !=
          SS_OK ||
      ss_checked_u64_to_petsc((uint64_t)input->vertex_count, &local_vertex_count) !=
          SS_OK ||
      ss_checked_u64_to_petsc(input->global_vertex_count, &global_vertex_count) !=
          SS_OK) {
    status = SS_ERROR_OVERFLOW;
  }
  corners = (PetscInt)input->dimension + 1;
  if (status == SS_OK) {
    const size_t cell_values = input->cell_count * (size_t)corners;
    const size_t coordinate_values = input->vertex_count * (size_t)input->dimension;
    cells = (PetscInt *)malloc((cell_values == 0U ? 1U : cell_values) * sizeof(*cells));
    vertices_adjacent = (PetscInt *)malloc((cell_values == 0U ? 1U : cell_values) *
                                           sizeof(*vertices_adjacent));
    coordinates = (PetscReal *)malloc(
        (coordinate_values == 0U ? 1U : coordinate_values) * sizeof(*coordinates));
    if (cells == NULL || vertices_adjacent == NULL || coordinates == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
    for (index = 0U; status == SS_OK && index < cell_values; ++index) {
      if (ss_checked_u64_to_petsc(input->cells[index], &cells[index]) != SS_OK) {
        status = SS_ERROR_OVERFLOW;
      }
    }
    for (index = 0U; status == SS_OK && index < coordinate_values; ++index) {
      coordinates[index] = (PetscReal)input->vertex_coordinates[index];
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    goto collective_cleanup;
  }
  (void)PetscLogEventBegin(ss_mesh_plex_create_event, 0, 0, 0, 0);
  if (DMPlexCreateFromCellListParallelPetsc(
          communicator, (PetscInt)input->dimension, local_cell_count,
          local_vertex_count, global_vertex_count, corners, PETSC_TRUE, cells,
          (PetscInt)input->dimension, coordinates, &vertex_sf, &vertices_adjacent,
          &initial) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(communicator, status);
  (void)PetscLogEventEnd(ss_mesh_plex_create_event, 0, 0, 0, 0);
  if (status != SS_OK) {
    goto collective_cleanup;
  }
  (void)ss_memory_view(communicator, "mesh_after_plex_create");
  (void)PetscLogEventBegin(ss_mesh_metadata_event, 0, 0, 0, 0);
  (void)PetscLogEventBegin(ss_mesh_attach_vertices_event, 0, 0, 0, 0);
  status = ss_mesh_attach_vertex_ids(initial, vertex_sf, input, &adjacent_ids,
                                     &adjacent_count);
  (void)PetscLogEventEnd(ss_mesh_attach_vertices_event, 0, 0, 0, 0);
  if (status == SS_OK) {
    (void)PetscLogEventBegin(ss_mesh_attach_cells_event, 0, 0, 0, 0);
    status = ss_mesh_attach_cells(initial, input, adjacent_count, vertices_adjacent,
                                  adjacent_ids);
    (void)PetscLogEventEnd(ss_mesh_attach_cells_event, 0, 0, 0, 0);
  }
  if (status == SS_OK) {
    (void)PetscLogEventBegin(ss_mesh_attach_entities_event, 0, 0, 0, 0);
    status = ss_mesh_attach_intermediate_ids(initial, input, adjacent_count,
                                             vertices_adjacent, adjacent_ids);
    (void)PetscLogEventEnd(ss_mesh_attach_entities_event, 0, 0, 0, 0);
  }
  if (status == SS_OK) {
    (void)PetscLogEventBegin(ss_mesh_point_index_event, 0, 0, 0, 0);
    status = ss_mesh_build_point_id_index(initial, &point_index, &point_index_count);
    (void)PetscLogEventEnd(ss_mesh_point_index_event, 0, 0, 0, 0);
  }
  if (status == SS_OK) {
    (void)PetscLogEventBegin(ss_mesh_entity_index_event, 0, 0, 0, 0);
    status = ss_mesh_build_entity_vertex_index(initial, NULL, &entity_index,
                                               &entity_index_count);
    (void)PetscLogEventEnd(ss_mesh_entity_index_event, 0, 0, 0, 0);
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    (void)PetscLogEventBegin(ss_mesh_semantic_labels_event, 0, 0, 0, 0);
    status =
        ss_mesh_attach_semantic_labels(initial, input, point_index, point_index_count);
    (void)PetscLogEventEnd(ss_mesh_semantic_labels_event, 0, 0, 0, 0);
  }
  if (status == SS_OK) {
    (void)PetscLogEventBegin(ss_mesh_semantic_audit_event, 0, 0, 0, 0);
    status = ss_mesh_audit_semantic_coverage(initial, input->dimension);
    (void)PetscLogEventEnd(ss_mesh_semantic_audit_event, 0, 0, 0, 0);
  }
  status = ss_status_allreduce_max(communicator, status);
  (void)PetscLogEventEnd(ss_mesh_metadata_event, 0, 0, 0, 0);
  if (status != SS_OK) {
    goto collective_cleanup;
  }
  (void)ss_memory_view(communicator, "mesh_after_metadata");
  (void)PetscLogEventBegin(ss_mesh_coordinate_event, 0, 0, 0, 0);
  status = ss_mesh_build_coordinate_space(initial, input, descriptor, point_index,
                                          point_index_count, entity_index,
                                          entity_index_count);
  (void)PetscLogEventEnd(ss_mesh_coordinate_event, 0, 0, 0, 0);
  if (status != SS_OK) {
    goto collective_cleanup;
  }
  (void)ss_memory_view(communicator, "mesh_after_coordinates");
  (void)PetscLogEventBegin(ss_mesh_digest_event, 0, 0, 0, 0);
  status = ss_mesh_compute_digests(communicator, input, descriptor, &digest_summary);
  if (status == SS_OK) {
    status = ss_mesh_compute_topology_digest(initial, NULL, communicator,
                                             digest_summary.topology_digest);
  }
  (void)PetscLogEventEnd(ss_mesh_digest_event, 0, 0, 0, 0);
  if (status != SS_OK) {
    goto collective_cleanup;
  }
  (void)ss_memory_view(communicator, "mesh_after_digest");
  /* Shared-entity ownership is assigned before constructing the required halo. */
  (void)PetscLogEventBegin(ss_mesh_distribute_event, 0, 0, 0, 0);
  {
    const SsMeshSharedPointBalance balance = input->shared_point_balance;
    PetscPartitioner partitioner = NULL;
    if (balance < SS_MESH_BALANCE_PETSC ||
        balance > SS_MESH_BALANCE_STABLE_HIGH_ORDER) {
      status = SS_ERROR_INVALID_ARGUMENT;
    }
    if (status == SS_OK &&
        (DMPlexGetPartitioner(initial, &partitioner) != PETSC_SUCCESS ||
         PetscPartitionerSetFromOptions(partitioner) != PETSC_SUCCESS ||
         DMPlexSetPartitionBalance(initial, PETSC_TRUE) != PETSC_SUCCESS)) {
      status = SS_ERROR_INTERNAL;
    }
    if (status == SS_OK && balance != SS_MESH_BALANCE_PETSC) {
      if (DMPlexDistribute(initial, 0, NULL, &distributed) != PETSC_SUCCESS) {
        status = SS_ERROR_INTERNAL;
      }
      status = ss_status_allreduce_max(communicator, status);
      if (status == SS_OK && distributed != NULL) {
        DM overlapped = NULL;
        status = balance == SS_MESH_BALANCE_STABLE_HIGH_ORDER
                     ? ss_mesh_rebalance_shared_p4_point_ownership(distributed)
                     : ss_mesh_rebalance_shared_p2_point_ownership(distributed);
        status = ss_status_allreduce_max(communicator, status);
        if (status == SS_OK && (DMPlexDistributeOverlap(distributed, 1, NULL,
                                                        &overlapped) != PETSC_SUCCESS ||
                                overlapped == NULL)) {
          status = SS_ERROR_INTERNAL;
        }
        if (status == SS_OK) {
          if (DMDestroy(&distributed) != PETSC_SUCCESS) {
            status = SS_ERROR_INTERNAL;
          } else {
            distributed = overlapped;
            overlapped = NULL;
          }
        }
        (void)DMDestroy(&overlapped);
      }
    } else if (status == SS_OK &&
               DMPlexDistribute(initial, 1, NULL, &distributed) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  (void)PetscLogEventEnd(ss_mesh_distribute_event, 0, 0, 0, 0);
  (void)ss_memory_view(communicator, "mesh_after_distribute_before_initial_destroy");
  if (status != SS_OK) {
    goto collective_cleanup;
  }
  if (distributed != NULL) {
    if (DMDestroy(&initial) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
      goto collective_cleanup;
    }
    initial = distributed;
    distributed = NULL;
  }
  (void)ss_memory_view(communicator, "mesh_after_initial_destroy");
  /* The undistributed DMPlex is a large, setup-only object.  Do not carry its
   * allocator high-water mark into the distributed maps and field plans. */
  (void)ss_memory_trim();
  (void)ss_memory_view(communicator, "mesh_after_initial_trim");
  created = (SsMesh *)calloc(1U, sizeof(*created));
  if (created == NULL) {
    status = SS_ERROR_OUT_OF_MEMORY;
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    goto collective_cleanup;
  }
  created->communicator = communicator;
  created->rank = rank;
  created->size = size;
  created->dm = initial;
  initial = NULL;
  created->dimension = input->dimension;
  created->coordinate_order = input->coordinate_order;
  created->coordinate_descriptor = descriptor;
  created->summary = digest_summary;
  created->summary.topology_generation = 1U;
  created->summary.coordinate_generation = 1U;
  created->summary.label_generation = 1U;
  created->summary.partition_generation = 1U;
  {
    const int written =
        snprintf(created->summary.coordinate_basis_id,
                 sizeof(created->summary.coordinate_basis_id), "%s", descriptor->id);
    if (written < 0 ||
        (size_t)written >= sizeof(created->summary.coordinate_basis_id)) {
      status = SS_ERROR_LIMIT;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    goto collective_cleanup;
  }
  created->geometry_policy.absolute_determinant_tolerance =
      input->absolute_determinant_tolerance;
  created->geometry_policy.maximum_condition =
      1.0 / input->relative_determinant_tolerance;
  (void)PetscLogEventBegin(ss_mesh_maps_event, 0, 0, 0, 0);
  status = ss_mesh_rebuild_point_map(created);
  if (status == SS_OK) {
    status = ss_mesh_rebuild_entity_map(created);
  }
  status = ss_status_allreduce_max(communicator, status);
  (void)PetscLogEventEnd(ss_mesh_maps_event, 0, 0, 0, 0);
  if (status != SS_OK) {
    goto collective_cleanup;
  }
  (void)ss_memory_view(communicator, "mesh_after_maps");
  (void)PetscLogEventBegin(ss_mesh_ownership_audit_event, 0, 0, 0, 0);
  {
    uint64_t expected_global_points = 0U;
    if (input->global_vertex_count >
            UINT64_MAX - input->global_intermediate_entity_count ||
        input->global_vertex_count + input->global_intermediate_entity_count >
            UINT64_MAX - input->global_cell_count) {
      status = SS_ERROR_OVERFLOW;
    } else {
      expected_global_points = input->global_vertex_count +
                               input->global_intermediate_entity_count +
                               input->global_cell_count;
      status = ss_mesh_audit_point_ownership(created, expected_global_points);
    }
  }
  (void)PetscLogEventEnd(ss_mesh_ownership_audit_event, 0, 0, 0, 0);
  if (status != SS_OK) {
    goto collective_cleanup;
  }
  (void)PetscLogEventBegin(ss_mesh_geometry_audit_event, 0, 0, 0, 0);
  status = ss_mesh_collective_geometry_audit(created, error);
  (void)PetscLogEventEnd(ss_mesh_geometry_audit_event, 0, 0, 0, 0);
  if (status != SS_OK) {
    goto collective_cleanup;
  }
  (void)PetscLogEventBegin(ss_mesh_summary_event, 0, 0, 0, 0);
  if (ss_mesh_finish_summary(created) != SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK &&
      (created->summary.global_vertices != input->global_vertex_count ||
       created->summary.global_cells != input->global_cell_count ||
       created->summary.global_facets != input->global_boundary_facet_count ||
       created->summary.global_intermediate_entities !=
           input->global_intermediate_entity_count ||
       created->summary.global_coordinate_nodes !=
           input->global_coordinate_node_count)) {
    status = SS_ERROR_INTEGRITY;
  }
  (void)PetscLogEventEnd(ss_mesh_summary_event, 0, 0, 0, 0);
  if (status != SS_OK) {
    goto collective_cleanup;
  }
  (void)PetscLogEventBegin(ss_mesh_migration_audit_event, 0, 0, 0, 0);
  status = ss_mesh_audit_migration(created);
  (void)PetscLogEventEnd(ss_mesh_migration_audit_event, 0, 0, 0, 0);
  if (status == SS_OK) {
    (void)PetscLogEventBegin(ss_mesh_facet_plans_event, 0, 0, 0, 0);
    status = ss_mesh_owned_facet_plans_build(created);
    (void)PetscLogEventEnd(ss_mesh_facet_plans_event, 0, 0, 0, 0);
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    goto collective_cleanup;
  }
  (void)ss_memory_view(communicator, "mesh_after_audits");

collective_cleanup:
  free(cells);
  free(coordinates);
  free(adjacent_ids);
  free(vertices_adjacent);
  free(point_index);
  free(entity_index);
  (void)PetscSFDestroy(&vertex_sf);
  (void)DMDestroy(&initial);
  (void)DMDestroy(&distributed);
  if (status != SS_OK) {
    if (created != NULL) {
      (void)ss_mesh_destroy(&created);
    }
    return status;
  }
  (void)ss_memory_view(communicator, "mesh_complete");
  *mesh = created;
  return SS_OK;
}

SsStatus ss_mesh_destroy(SsMesh **mesh) {
  SsStatus status = SS_OK;
  if (mesh == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (*mesh == NULL) {
    return SS_OK;
  }
  if ((*mesh)->active_field_spaces != 0U) {
    status = SS_ERROR_INVALID_STATE;
  }
  status = ss_status_allreduce_max((*mesh)->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  free((*mesh)->points);
  free((*mesh)->entities);
  ss_mesh_clear_metadata_cache(*mesh);
  free((*mesh)->facet_plan_support_values);
  free((*mesh)->facet_plans);
  if (DMDestroy(&(*mesh)->dm) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  free(*mesh);
  *mesh = NULL;
  return status;
}

SsStatus ss_mesh_summary(const SsMesh *mesh, SsMeshSummary *summary) {
  if (mesh == NULL || summary == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *summary = mesh->summary;
  return SS_OK;
}

SsStatus ss_mesh_cell_is_owned(const SsMesh *mesh, SsEntityId stable_cell_id,
                               int *owned) {
  PetscInt cell = -1;
  if (mesh == NULL || owned == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (ss_mesh_cell_point(mesh, stable_cell_id, &cell) != SS_OK) {
    *owned = 0;
    return SS_OK;
  }
  *owned = ss_mesh_point_is_ghost(mesh, mesh->dm, cell) == 0 ? 1 : 0;
  return SS_OK;
}

SsStatus ss_mesh_renumber(SsMesh *mesh, uint64_t seed) {
  PetscInt start = 0;
  PetscInt end = 0;
  PetscInt *indices = NULL;
  IS permutation = NULL;
  DM permuted = NULL;
  PetscInt point;
  SsStatus status = SS_OK;
  if (mesh == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (mesh->active_field_spaces != 0U) {
    status = SS_ERROR_INVALID_STATE;
  } else if (mesh->summary.partition_generation == UINT64_MAX) {
    status = SS_ERROR_OVERFLOW;
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  if (DMPlexGetChart(mesh->dm, &start, &end) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  indices = (PetscInt *)malloc((size_t)(end - start) * sizeof(*indices));
  if (end != start && indices == NULL) {
    status = SS_ERROR_OUT_OF_MEMORY;
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status != SS_OK) {
    free(indices);
    return status;
  }
  for (point = start; point < end; ++point) {
    const PetscInt local = point - start;
    if ((seed & UINT64_C(1)) != 0U) {
      indices[local] = start + (end - start - 1 - local);
    } else {
      indices[local] = start + (local + 1) % (end == start ? 1 : end - start);
    }
  }
  if (ISCreateGeneral(PETSC_COMM_SELF, end - start, indices, PETSC_COPY_VALUES,
                      &permutation) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  free(indices);
  indices = NULL;
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (DMPlexPermute(mesh->dm, permutation, &permuted) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (ISDestroy(&permutation) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (DMDestroy(&mesh->dm) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  mesh->dm = permuted;
  permuted = NULL;
  status = ss_mesh_rebuild_point_map(mesh);
  if (status == SS_OK) {
    status = ss_mesh_rebuild_entity_map(mesh);
  }
  status = ss_status_allreduce_max(mesh->communicator, status);
  if (status == SS_OK) {
    char topology_digest[SS_SHA256_HEX_CAPACITY] = {0};
    status = ss_mesh_compute_topology_digest(mesh->dm, mesh, mesh->communicator,
                                             topology_digest);
    if (status == SS_OK &&
        strcmp(topology_digest, mesh->summary.topology_digest) != 0) {
      status = SS_ERROR_INTEGRITY;
    }
  }
  if (status == SS_OK) {
    ++mesh->summary.partition_generation;
  }

cleanup:
  free(indices);
  (void)ISDestroy(&permutation);
  (void)DMDestroy(&permuted);
  return status;
}
