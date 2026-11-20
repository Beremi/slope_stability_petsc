/* Construct and query oriented coordinate closures.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "mesh_internal.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/core/sort.h"
#include "slope_internal/core/util.h"

#include <petscdm.h>
#include <petscdmplex.h>
#include <petscis.h>
#include <petscsection.h>
#include <petscsf.h>
#include <petscvec.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

uint32_t ss_mesh_slots_for_depth(uint32_t dimension, uint32_t order, uint32_t depth) {
  if (depth == 0U) {
    return 1U;
  }
  if (depth == 1U) {
    return order - 1U;
  }
  if (depth == 2U) {
    if (dimension == 2U) {
      return (order - 1U) * (order - 2U) / 2U;
    }
    return (order - 1U) * (order - 2U) / 2U;
  }
  if (depth == 3U && dimension == 3U) {
    return (order - 1U) * (order - 2U) * (order - 3U) / 6U;
  }
  return 0U;
}

static uint32_t ss_mesh_oriented_slot(const SsFeDescriptor *descriptor,
                                      const SsEntityId *cell_vertex_ids, size_t node) {
  SsEntityId ids[4] = {0U, 0U, 0U, 0U};
  uint8_t alpha[4] = {0U, 0U, 0U, 0U};
  size_t count = 0U;
  uint32_t barycentric;
  size_t left;
  for (barycentric = 0U; barycentric <= descriptor->dimension; ++barycentric) {
    const uint8_t value =
        descriptor->barycentric_indices[node * (size_t)(descriptor->dimension + 1U) +
                                        barycentric];
    if (value != 0U) {
      ids[count] = cell_vertex_ids[barycentric];
      alpha[count] = value;
      ++count;
    }
  }
  for (left = 0U; left < count; ++left) {
    size_t right;
    for (right = left + 1U; right < count; ++right) {
      if (ids[right] < ids[left]) {
        const SsEntityId id = ids[left];
        const uint8_t value = alpha[left];
        ids[left] = ids[right];
        alpha[left] = alpha[right];
        ids[right] = id;
        alpha[right] = value;
      }
    }
  }
  if (count <= 1U) {
    return 0U;
  }
  if (count == 2U) {
    return (uint32_t)alpha[1] - 1U;
  }
  if (count == 3U) {
    uint32_t slot = 0U;
    uint32_t a;
    for (a = descriptor->order - 2U; a >= 1U; --a) {
      uint32_t b;
      for (b = descriptor->order - a - 1U; b >= 1U; --b) {
        const uint32_t c = descriptor->order - a - b;
        if ((uint32_t)alpha[0] == a && (uint32_t)alpha[1] == b &&
            (uint32_t)alpha[2] == c) {
          return slot;
        }
        ++slot;
        if (b == 1U) {
          break;
        }
      }
      if (a == 1U) {
        break;
      }
    }
    return UINT32_MAX;
  }
  if (count == 4U) {
    uint32_t slot = 0U;
    uint32_t a;
    for (a = descriptor->order - 3U; a >= 1U; --a) {
      uint32_t b;
      for (b = descriptor->order - a - 2U; b >= 1U; --b) {
        uint32_t c;
        for (c = descriptor->order - a - b - 1U; c >= 1U; --c) {
          const uint32_t d = descriptor->order - a - b - c;
          if ((uint32_t)alpha[0] == a && (uint32_t)alpha[1] == b &&
              (uint32_t)alpha[2] == c && (uint32_t)alpha[3] == d) {
            return slot;
          }
          ++slot;
          if (c == 1U) {
            break;
          }
        }
        if (b == 1U) {
          break;
        }
      }
      if (a == 1U) {
        break;
      }
    }
    return UINT32_MAX;
  }
  return 0U;
}

SsStatus ss_mesh_cell_vertex_ids(DM dm, const SsMesh *mesh, PetscInt cell,
                                 uint32_t dimension, SsEntityId vertex_ids[4]) {
  uint32_t vertex;
  if (mesh != NULL) {
    return ss_mesh_cached_cell_vertices(mesh, cell, dimension, vertex_ids);
  }
  for (vertex = 0U; vertex <= dimension; ++vertex) {
    if (ss_mesh_get_slot_id(dm, cell, "ss-cell-vertex", vertex, &vertex_ids[vertex]) !=
        SS_OK) {
      return SS_ERROR_INTERNAL;
    }
  }
  return SS_OK;
}

SsStatus ss_mesh_node_entity_point(const SsEntityVertexEntry *entity_index,
                                   size_t entity_count,
                                   const SsFeDescriptor *descriptor, PetscInt cell,
                                   const SsEntityId *cell_vertex_ids, size_t node,
                                   PetscInt *entity_point, uint32_t *entity_slot) {
  SsEntityId entity_vertex_ids[4] = {0U, 0U, 0U, 0U};
  size_t vertex_count = 0U;
  uint32_t barycentric;
  for (barycentric = 0U; barycentric <= descriptor->dimension; ++barycentric) {
    if (descriptor->barycentric_indices[node * (size_t)(descriptor->dimension + 1U) +
                                        barycentric] != 0U) {
      entity_vertex_ids[vertex_count++] = cell_vertex_ids[barycentric];
    }
  }
  ss_sort(entity_vertex_ids, vertex_count, sizeof(entity_vertex_ids[0]),
          ss_mesh_entity_id_compare);
  *entity_slot = ss_mesh_oriented_slot(descriptor, cell_vertex_ids, node);
  if (vertex_count == (size_t)descriptor->dimension + 1U) {
    *entity_point = cell;
    return SS_OK;
  }
  return ss_mesh_entity_vertex_index_find(
      entity_index, entity_count, (uint32_t)vertex_count - 1U, entity_vertex_ids,
      vertex_count, entity_point);
}

SsStatus ss_mesh_build_coordinate_space(DM dm, const SsMeshInput *input,
                                        const SsFeDescriptor *descriptor,
                                        const SsPointIdEntry *point_index,
                                        size_t point_count,
                                        const SsEntityVertexEntry *entity_index,
                                        size_t entity_count) {
  MPI_Comm communicator = PetscObjectComm((PetscObject)dm);
  PetscLayout layout = NULL;
  PetscSF coordinate_sf = NULL;
  PetscSection section = NULL;
  DM coordinate_dm = NULL;
  Vec vector = NULL;
  PetscScalar *array = NULL;
  unsigned char *assigned = NULL;
  PetscInt *root_indices = NULL;
  PetscInt *adjacent_indices = NULL;
  SsEntityId *adjacent_ids = NULL;
  double *adjacent_coordinates = NULL;
  PetscInt local_coordinate_count = 0;
  PetscInt global_coordinate_count = 0;
  PetscInt adjacent_count = 0;
  PetscInt owned_start = 0;
  PetscInt owned_end = 0;
  PetscInt chart_start = 0;
  PetscInt chart_end = 0;
  PetscInt storage_size = 0;
  PetscInt point;
  SsEntityId id_dummy = 0U;
  double coordinate_dummy[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
  MPI_Datatype coordinate_type = MPI_DATATYPE_NULL;
  SsStatus status = SS_OK;
  size_t raw_adjacent_count;
  size_t index;
  if (ss_checked_u64_to_petsc((uint64_t)input->coordinate_node_count,
                              &local_coordinate_count) != SS_OK ||
      ss_checked_u64_to_petsc(input->global_coordinate_node_count,
                              &global_coordinate_count) != SS_OK ||
      input->cell_count > SIZE_MAX / descriptor->node_count) {
    status = SS_ERROR_OVERFLOW;
  }
  raw_adjacent_count =
      status == SS_OK ? input->cell_count * descriptor->node_count : 0U;
  if (status == SS_OK && raw_adjacent_count > (size_t)PETSC_INT_MAX) {
    status = SS_ERROR_OVERFLOW;
  }
  if (status == SS_OK) {
    root_indices = (PetscInt *)malloc(
        (input->coordinate_node_count == 0U ? 1U : input->coordinate_node_count) *
        sizeof(*root_indices));
    adjacent_indices =
        (PetscInt *)malloc((raw_adjacent_count == 0U ? 1U : raw_adjacent_count) *
                           sizeof(*adjacent_indices));
    if (root_indices == NULL || adjacent_indices == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (PetscLayoutCreate(communicator, &layout) != PETSC_SUCCESS ||
      PetscLayoutSetSize(layout, global_coordinate_count) != PETSC_SUCCESS ||
      PetscLayoutSetLocalSize(layout, local_coordinate_count) != PETSC_SUCCESS ||
      PetscLayoutSetBlockSize(layout, 1) != PETSC_SUCCESS ||
      PetscLayoutSetUp(layout) != PETSC_SUCCESS ||
      PetscLayoutGetRange(layout, &owned_start, &owned_end) != PETSC_SUCCESS ||
      owned_end - owned_start != local_coordinate_count) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    for (index = 0U; index < input->coordinate_node_count; ++index) {
      root_indices[index] = owned_start + (PetscInt)index;
    }
    for (index = 0U; index < raw_adjacent_count; ++index) {
      if (ss_checked_u64_to_petsc(input->cell_coordinate_nodes[index],
                                  &adjacent_indices[index]) != SS_OK) {
        status = SS_ERROR_OVERFLOW;
        break;
      }
    }
  }
  if (status == SS_OK) {
    ss_sort(adjacent_indices, raw_adjacent_count, sizeof(*adjacent_indices),
            ss_petsc_int_compare);
    for (index = 0U; index < raw_adjacent_count; ++index) {
      if (adjacent_count == 0 ||
          adjacent_indices[index] != adjacent_indices[adjacent_count - 1]) {
        adjacent_indices[adjacent_count++] = adjacent_indices[index];
      }
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (PetscSFCreateByMatchingIndices(layout, local_coordinate_count, root_indices, NULL,
                                     0, adjacent_count, adjacent_indices, NULL, 0, NULL,
                                     &coordinate_sf) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    adjacent_ids = (SsEntityId *)calloc(
        adjacent_count == 0 ? 1U : (size_t)adjacent_count, sizeof(*adjacent_ids));
    adjacent_coordinates = (double *)calloc(
        adjacent_count == 0 ? SS_FE_MAX_DIMENSION
                            : (size_t)adjacent_count * input->dimension,
        sizeof(*adjacent_coordinates));
    if (adjacent_ids == NULL || adjacent_coordinates == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    goto cleanup;
  }
  if (MPI_Type_contiguous((int)input->dimension, MPI_DOUBLE, &coordinate_type) !=
          MPI_SUCCESS ||
      MPI_Type_commit(&coordinate_type) != MPI_SUCCESS ||
      PetscSFBcastBegin(coordinate_sf, MPI_UINT64_T,
                        input->coordinate_node_count == 0U ? &id_dummy
                                                           : input->coordinate_node_ids,
                        adjacent_ids, MPI_REPLACE) != PETSC_SUCCESS ||
      PetscSFBcastEnd(coordinate_sf, MPI_UINT64_T,
                      input->coordinate_node_count == 0U ? &id_dummy
                                                         : input->coordinate_node_ids,
                      adjacent_ids, MPI_REPLACE) != PETSC_SUCCESS ||
      PetscSFBcastBegin(coordinate_sf, coordinate_type,
                        input->coordinate_node_count == 0U
                            ? coordinate_dummy
                            : input->coordinate_coordinates,
                        adjacent_coordinates, MPI_REPLACE) != PETSC_SUCCESS ||
      PetscSFBcastEnd(coordinate_sf, coordinate_type,
                      input->coordinate_node_count == 0U
                          ? coordinate_dummy
                          : input->coordinate_coordinates,
                      adjacent_coordinates, MPI_REPLACE) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  if (DMPlexGetChart(dm, &chart_start, &chart_end) != PETSC_SUCCESS ||
      PetscSectionCreate(PetscObjectComm((PetscObject)dm), &section) != PETSC_SUCCESS ||
      PetscSectionSetNumFields(section, 1) != PETSC_SUCCESS ||
      PetscSectionSetFieldComponents(section, 0, (PetscInt)input->dimension) !=
          PETSC_SUCCESS ||
      PetscSectionSetChart(section, chart_start, chart_end) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  for (point = chart_start; point < chart_end; ++point) {
    PetscInt depth = -1;
    uint32_t slots;
    PetscInt dof;
    if (DMPlexGetPointDepth(dm, point, &depth) != PETSC_SUCCESS || depth < 0) {
      status = SS_ERROR_INTERNAL;
      goto cleanup;
    }
    slots = ss_mesh_slots_for_depth(input->dimension, input->coordinate_order,
                                    (uint32_t)depth);
    dof = (PetscInt)(slots * input->dimension);
    if (PetscSectionSetDof(section, point, dof) != PETSC_SUCCESS ||
        PetscSectionSetFieldDof(section, point, 0, dof) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
      goto cleanup;
    }
  }
  if (PetscSectionSetUp(section) != PETSC_SUCCESS ||
      DMSetCoordinateSection(dm, (PetscInt)input->dimension, section) !=
          PETSC_SUCCESS ||
      DMGetCoordinateDM(dm, &coordinate_dm) != PETSC_SUCCESS ||
      DMCreateLocalVector(coordinate_dm, &vector) != PETSC_SUCCESS ||
      PetscSectionGetStorageSize(section, &storage_size) != PETSC_SUCCESS ||
      VecGetArray(vector, &array) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto cleanup;
  }
  assigned = (unsigned char *)calloc((size_t)storage_size, sizeof(*assigned));
  if (storage_size != 0 && assigned == NULL) {
    status = SS_ERROR_OUT_OF_MEMORY;
    goto cleanup;
  }
  for (point = 0; point < storage_size; ++point) {
    array[point] = (PetscScalar)NAN;
  }
  {
    size_t cell;
    for (cell = 0U; cell < input->cell_count; ++cell) {
      PetscInt cell_point = -1;
      SsEntityId cell_vertex_ids[4] = {0U, 0U, 0U, 0U};
      size_t node;
      if (ss_mesh_point_id_index_find(point_index, point_count, input->cell_ids[cell],
                                      &cell_point) != SS_OK ||
          ss_mesh_cell_vertex_ids(dm, NULL, cell_point, input->dimension,
                                  cell_vertex_ids) != SS_OK) {
        status = SS_ERROR_INTERNAL;
        goto cleanup;
      }
      for (node = 0U; node < descriptor->node_count; ++node) {
        PetscInt global_coordinate = 0;
        PetscInt left = 0;
        PetscInt right = adjacent_count;
        PetscInt entity_point = -1;
        uint32_t entity_slot = 0U;
        PetscInt offset = 0;
        uint32_t coordinate;
        if (ss_checked_u64_to_petsc(
                input->cell_coordinate_nodes[cell * descriptor->node_count + node],
                &global_coordinate) != SS_OK) {
          status = SS_ERROR_OVERFLOW;
          goto cleanup;
        }
        while (left < right) {
          const PetscInt middle = left + (right - left) / 2;
          if (adjacent_indices[middle] < global_coordinate) {
            left = middle + 1;
          } else {
            right = middle;
          }
        }
        if (left >= adjacent_count || adjacent_indices[left] != global_coordinate ||
            ss_mesh_node_entity_point(entity_index, entity_count, descriptor,
                                      cell_point, cell_vertex_ids, node, &entity_point,
                                      &entity_slot) != SS_OK ||
            PetscSectionGetOffset(section, entity_point, &offset) != PETSC_SUCCESS ||
            ss_mesh_set_slot_id(dm, entity_point, "ss-coordinate-node", entity_slot,
                                adjacent_ids[left]) != SS_OK) {
          status = SS_ERROR_INTEGRITY;
          goto cleanup;
        }
        for (coordinate = 0U; coordinate < input->dimension; ++coordinate) {
          const PetscInt value_index =
              offset + (PetscInt)(entity_slot * input->dimension + coordinate);
          const double value =
              adjacent_coordinates[(size_t)left * input->dimension + coordinate];
          if (value_index < 0 || value_index >= storage_size) {
            status = SS_ERROR_INTEGRITY;
            goto cleanup;
          }
          if (assigned[value_index] != 0U &&
              fabs((double)array[value_index] - value) > 4.0e-13) {
            status = SS_ERROR_INTEGRITY;
            goto cleanup;
          }
          array[value_index] = (PetscScalar)value;
          assigned[value_index] = 1U;
        }
      }
    }
    for (point = 0; point < storage_size; ++point) {
      if (assigned[point] == 0U) {
        status = SS_ERROR_INTEGRITY;
        goto cleanup;
      }
    }
  }

cleanup:
  if (array != NULL && VecRestoreArray(vector, &array) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  free(assigned);
  if (status == SS_OK &&
      (DMSetCoordinatesLocal(dm, vector) != PETSC_SUCCESS ||
       PetscObjectSetName((PetscObject)vector, "coordinates") != PETSC_SUCCESS)) {
    status = SS_ERROR_INTERNAL;
  }
  if (VecDestroy(&vector) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (PetscSectionDestroy(&section) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (coordinate_type != MPI_DATATYPE_NULL &&
      MPI_Type_free(&coordinate_type) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  (void)PetscSFDestroy(&coordinate_sf);
  (void)PetscLayoutDestroy(&layout);
  free(root_indices);
  free(adjacent_indices);
  free(adjacent_ids);
  free(adjacent_coordinates);
  return ss_status_allreduce_max(communicator, status);
}

SsStatus ss_mesh_cell_coordinate_closure(const SsMesh *mesh, SsEntityId stable_cell_id,
                                         SsEntityId *node_ids, double *coordinates,
                                         size_t capacity, size_t *node_count) {
  PetscInt cell = -1;
  SsEntityId cell_vertex_ids[4] = {0U, 0U, 0U, 0U};
  PetscSection section = NULL;
  Vec vector = NULL;
  const PetscScalar *array = NULL;
  size_t node;
  SsStatus status = SS_OK;
  if (mesh == NULL || node_ids == NULL || coordinates == NULL || node_count == NULL ||
      capacity < mesh->coordinate_descriptor->node_count ||
      ss_mesh_cell_point(mesh, stable_cell_id, &cell) != SS_OK ||
      ss_mesh_cell_vertex_ids(mesh->dm, mesh, cell, mesh->dimension, cell_vertex_ids) !=
          SS_OK ||
      DMGetCoordinateSection(mesh->dm, &section) != PETSC_SUCCESS ||
      DMGetCoordinatesLocal(mesh->dm, &vector) != PETSC_SUCCESS || vector == NULL ||
      VecGetArrayRead(vector, &array) != PETSC_SUCCESS) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (node = 0U; node < mesh->coordinate_descriptor->node_count; ++node) {
    PetscInt entity_point = -1;
    uint32_t entity_slot = 0U;
    PetscInt offset = 0;
    uint32_t coordinate;
    if (ss_mesh_node_entity_point(mesh->entities, mesh->entity_count,
                                  mesh->coordinate_descriptor, cell, cell_vertex_ids,
                                  node, &entity_point, &entity_slot) != SS_OK ||
        ss_mesh_cached_coordinate_node(mesh, entity_point, entity_slot,
                                       &node_ids[node]) != SS_OK ||
        PetscSectionGetOffset(section, entity_point, &offset) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
      break;
    }
    for (coordinate = 0U; coordinate < mesh->dimension; ++coordinate) {
      coordinates[node * (size_t)mesh->dimension + coordinate] = (double)PetscRealPart(
          array[offset + (PetscInt)(entity_slot * mesh->dimension + coordinate)]);
    }
  }
  if (VecRestoreArrayRead(vector, &array) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    *node_count = mesh->coordinate_descriptor->node_count;
  }
  return status;
}
