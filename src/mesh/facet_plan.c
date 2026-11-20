/* Build mesh and field boundary-facet plans.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "mesh_internal.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/core/sort.h"

#include "slope_internal/fe/fe.h"

#include <petscdm.h>
#include <petscdmplex.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static SsStatus ss_mesh_pack_facets_uncached(const SsMesh *mesh, uint32_t support_value,
                                             int owned_only, SsFacetPack *packs,
                                             size_t capacity, size_t *count);

static SsStatus ss_mesh_parent_facet_index(const SsMesh *mesh, PetscInt parent_cell,
                                           PetscInt facet_point,
                                           uint32_t *facet_index) {
  static const uint32_t triangle_facets[3][2] = {
      {0U, 1U},
      {1U, 2U},
      {0U, 2U},
  };
  static const uint32_t tetrahedron_facets[4][3] = {
      {0U, 1U, 2U},
      {0U, 1U, 3U},
      {0U, 2U, 3U},
      {1U, 2U, 3U},
  };
  SsEntityId cell_vertex_ids[4] = {0U, 0U, 0U, 0U};
  SsEntityId point_vertex_ids[4] = {0U, 0U, 0U, 0U};
  size_t point_vertex_count = 0U;
  uint32_t candidate;
  if (mesh == NULL || facet_index == NULL ||
      ss_mesh_cell_vertex_ids(mesh->dm, mesh, parent_cell, mesh->dimension,
                              cell_vertex_ids) != SS_OK ||
      ss_mesh_point_vertex_ids(mesh->dm, mesh, NULL, facet_point, point_vertex_ids,
                               &point_vertex_count) != SS_OK ||
      point_vertex_count != (size_t)mesh->dimension) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (candidate = 0U; candidate <= mesh->dimension; ++candidate) {
    SsEntityId expected[3] = {0U, 0U, 0U};
    uint32_t vertex;
    for (vertex = 0U; vertex < mesh->dimension; ++vertex) {
      const uint32_t local_vertex = mesh->dimension == 2U
                                        ? triangle_facets[candidate][vertex]
                                        : tetrahedron_facets[candidate][vertex];
      expected[vertex] = cell_vertex_ids[local_vertex];
    }
    ss_sort(expected, (size_t)mesh->dimension, sizeof(expected[0]),
            ss_mesh_entity_id_compare);
    if (ss_mesh_same_vertex_set(expected, point_vertex_ids, (size_t)mesh->dimension) !=
        0) {
      *facet_index = candidate;
      return SS_OK;
    }
  }
  return SS_ERROR_NOT_FOUND;
}

SsStatus ss_mesh_facet_geometry_at(const SsFacetPack *pack,
                                   const double *facet_reference_point,
                                   SsFacetSample *sample) {
  const SsFeDescriptor *descriptor = NULL;
  SsFeFacetGeometry geometry = {0};
  if (pack == NULL || facet_reference_point == NULL || sample == NULL ||
      ss_fe_descriptor_find(pack->dimension, pack->coordinate_order, &descriptor) !=
          SS_OK ||
      pack->coordinate_node_count != descriptor->node_count ||
      ss_fe_facet_geometry_eval(descriptor, pack->facet_index,
                                pack->coordinate_coordinates, facet_reference_point,
                                pack->parent_centroid, &geometry) != SS_OK) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(sample, 0, sizeof(*sample));
  sample->dimension = geometry.dimension;
  memcpy(sample->physical, geometry.physical, sizeof(sample->physical));
  memcpy(sample->tangents, geometry.tangents, sizeof(sample->tangents));
  memcpy(sample->outward_normal, geometry.outward_normal,
         sizeof(sample->outward_normal));
  sample->measure = geometry.measure;
  return SS_OK;
}

static SsStatus ss_mesh_pack_facets_uncached(const SsMesh *mesh, uint32_t support_value,
                                             int owned_only, SsFacetPack *packs,
                                             size_t capacity, size_t *count) {
  char support_name[64];
  DMLabel support_label = NULL;
  size_t entry_index;
  size_t required = 0U;
  size_t packed = 0U;
  if (mesh == NULL || count == NULL ||
      (mesh->dimension != 2U && mesh->dimension != 3U) ||
      (packs == NULL && capacity != 0U) ||
      ss_mesh_semantic_membership_name(SS_MESH_LABEL_SUPPORT, support_value,
                                       support_name) != SS_OK) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (DMGetLabel(mesh->dm, support_name, &support_label) != PETSC_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (support_label == NULL) {
    *count = 0U;
    return SS_OK;
  }
  for (entry_index = 0U; entry_index < mesh->point_count; ++entry_index) {
    const SsPointEntry *entry = &mesh->points[entry_index];
    PetscInt membership = -1;
    if (entry->kind != SS_MESH_ENTITY_FACET ||
        (owned_only != 0 &&
         ss_mesh_point_is_ghost(mesh, mesh->dm, entry->point) != 0)) {
      continue;
    }
    if (DMLabelGetValue(support_label, entry->point, &membership) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    if (membership >= 0) {
      ++required;
    }
  }
  *count = required;
  if (packs == NULL) {
    return SS_OK;
  }
  if (capacity < required) {
    return SS_ERROR_LIMIT;
  }
  for (entry_index = 0U; entry_index < mesh->point_count; ++entry_index) {
    const SsPointEntry *entry = &mesh->points[entry_index];
    PetscInt membership = -1;
    if (entry->kind != SS_MESH_ENTITY_FACET ||
        (owned_only != 0 &&
         ss_mesh_point_is_ghost(mesh, mesh->dm, entry->point) != 0) ||
        DMLabelGetValue(support_label, entry->point, &membership) != PETSC_SUCCESS ||
        membership < 0) {
      continue;
    }
    {
      if (packed >= capacity) {
        return SS_ERROR_INTEGRITY;
      }
      SsFacetPack *pack = &packs[packed];
      PetscInt support_size = 0;
      const PetscInt *support = NULL;
      SsEntityId vertex_ids[4] = {0U, 0U, 0U, 0U};
      size_t vertex_count = 0U;
      double reference_centroid[SS_FE_MAX_DIMENSION] = {0.0, 0.0, 0.0};
      double facet_reference_centroid[SS_FE_MAX_DIMENSION - 1U] = {0.0, 0.0};
      SsFeGeometry parent_geometry = {0};
      SsFacetSample sample = {0};
      size_t coordinate_node_count = 0U;
      uint32_t component;
      memset(pack, 0, sizeof(*pack));
      if (DMPlexGetSupportSize(mesh->dm, entry->point, &support_size) !=
              PETSC_SUCCESS ||
          support_size != 1 ||
          DMPlexGetSupport(mesh->dm, entry->point, &support) != PETSC_SUCCESS ||
          ss_mesh_cached_stable_id(mesh, support[0], &pack->stable_parent_cell_id) !=
              SS_OK ||
          ss_mesh_point_vertex_ids(mesh->dm, mesh, NULL, entry->point, vertex_ids,
                                   &vertex_count) != SS_OK ||
          vertex_count != (size_t)mesh->dimension ||
          ss_mesh_parent_facet_index(mesh, support[0], entry->point,
                                     &pack->facet_index) != SS_OK ||
          ss_mesh_cell_coordinate_closure(mesh, pack->stable_parent_cell_id,
                                          pack->coordinate_node_ids,
                                          pack->coordinate_coordinates, SS_FE_MAX_NODES,
                                          &coordinate_node_count) != SS_OK) {
        return SS_ERROR_INTERNAL;
      }
      pack->dimension = mesh->dimension;
      pack->stable_facet_id = entry->stable_id;
      pack->facet_owned =
          ss_mesh_point_is_ghost(mesh, mesh->dm, entry->point) == 0 ? 1U : 0U;
      pack->support_value = support_value;
      pack->vertex_count = vertex_count;
      memcpy(pack->vertex_ids, vertex_ids, vertex_count * sizeof(pack->vertex_ids[0]));
      pack->coordinate_order = mesh->coordinate_order;
      pack->coordinate_node_count = coordinate_node_count;
      for (component = 0U; component < mesh->dimension; ++component) {
        reference_centroid[component] = 1.0 / ((double)mesh->dimension + 1.0);
      }
      if (ss_fe_geometry_eval(mesh->coordinate_descriptor, pack->coordinate_coordinates,
                              reference_centroid, &mesh->geometry_policy,
                              &parent_geometry) != SS_OK) {
        return SS_ERROR_INTEGRITY;
      }
      memcpy(pack->parent_centroid, parent_geometry.physical,
             sizeof(pack->parent_centroid));
      facet_reference_centroid[0] = mesh->dimension == 2U ? 0.5 : 1.0 / 3.0;
      if (mesh->dimension == 3U) {
        facet_reference_centroid[1] = 1.0 / 3.0;
      }
      if (ss_mesh_facet_geometry_at(pack, facet_reference_centroid, &sample) != SS_OK) {
        return SS_ERROR_INTEGRITY;
      }
      memcpy(pack->centroid, sample.physical, sizeof(pack->centroid));
      memcpy(pack->outward_normal, sample.outward_normal, sizeof(pack->outward_normal));
      for (component = 0U; component < mesh->dimension; ++component) {
        pack->outward_dot +=
            pack->outward_normal[component] *
            (pack->centroid[component] - pack->parent_centroid[component]);
      }
      /*
       * The FE facet normal is oriented topologically: the cofactor of the
       * admissible (positive-determinant) parent map carries the reference
       * outward normal.  On strongly curved cells the facet-centroid offset
       * from the parent centroid can point inwards, so `outward_dot` is kept
       * only as a diagnostic; a nonfinite value still means broken geometry.
       */
      pack->orientation = 1;
      if (!isfinite(pack->outward_dot)) {
        return SS_ERROR_INTEGRITY;
      }
      ++packed;
    }
  }
  if (packed != required) {
    return SS_ERROR_INTEGRITY;
  }
  *count = packed;
  return SS_OK;
}

SsStatus ss_mesh_owned_facet_plans_build(SsMesh *mesh) {
  PetscInt label_count = 0;
  PetscInt label_index;
  size_t required = 0U;
  size_t packed = 0U;
  uint64_t plan_bytes;
  SsStatus status = SS_OK;
  if (mesh == NULL || mesh->facet_plans != NULL ||
      mesh->facet_plan_support_values != NULL || mesh->facet_plan_count != 0U) {
    return SS_ERROR_INVALID_STATE;
  }
  if (DMGetNumLabels(mesh->dm, &label_count) != PETSC_SUCCESS || label_count < 0) {
    return SS_ERROR_INTERNAL;
  }
  for (label_index = 0; label_index < label_count; ++label_index) {
    const char *name = NULL;
    SsMeshLabelKind kind = SS_MESH_LABEL_MATERIAL;
    uint32_t support_value = 0U;
    size_t support_count = 0U;
    if (DMGetLabelName(mesh->dm, label_index, &name) != PETSC_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    if (ss_mesh_semantic_membership_parse(name, &kind, &support_value) == 0 ||
        kind != SS_MESH_LABEL_SUPPORT) {
      continue;
    }
    status =
        ss_mesh_pack_facets_uncached(mesh, support_value, 1, NULL, 0U, &support_count);
    if (status != SS_OK) {
      return status;
    }
    if (required > SIZE_MAX - support_count) {
      return SS_ERROR_OVERFLOW;
    }
    required += support_count;
  }
  size_t facet_bytes;
  size_t support_bytes;
  size_t total_bytes;
  if (ss_checked_mul_size(required, sizeof(*mesh->facet_plans), &facet_bytes) !=
          SS_OK ||
      ss_checked_mul_size(required, sizeof(*mesh->facet_plan_support_values),
                          &support_bytes) != SS_OK ||
      ss_checked_add_size(facet_bytes, support_bytes, &total_bytes) != SS_OK) {
    return SS_ERROR_OVERFLOW;
  }
  if (required != 0U) {
    mesh->facet_plans = (SsFacetPack *)calloc(1U, facet_bytes);
    mesh->facet_plan_support_values = (uint32_t *)calloc(1U, support_bytes);
    if (mesh->facet_plans == NULL || mesh->facet_plan_support_values == NULL) {
      free(mesh->facet_plans);
      free(mesh->facet_plan_support_values);
      mesh->facet_plans = NULL;
      mesh->facet_plan_support_values = NULL;
      return SS_ERROR_OUT_OF_MEMORY;
    }
  }
  for (label_index = 0; label_index < label_count; ++label_index) {
    const char *name = NULL;
    SsMeshLabelKind kind = SS_MESH_LABEL_MATERIAL;
    uint32_t support_value = 0U;
    size_t support_count = 0U;
    size_t plan;
    if (DMGetLabelName(mesh->dm, label_index, &name) != PETSC_SUCCESS) {
      status = SS_ERROR_INTERNAL;
      break;
    }
    if (ss_mesh_semantic_membership_parse(name, &kind, &support_value) == 0 ||
        kind != SS_MESH_LABEL_SUPPORT) {
      continue;
    }
    if (packed > required) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    status = ss_mesh_pack_facets_uncached(
        mesh, support_value, 1, required == 0U ? NULL : mesh->facet_plans + packed,
        required - packed, &support_count);
    if (status != SS_OK || support_count > required - packed) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    for (plan = 0U; plan < support_count; ++plan) {
      mesh->facet_plan_support_values[packed + plan] = support_value;
    }
    packed += support_count;
  }
  if (status == SS_OK && packed != required) {
    status = SS_ERROR_INTEGRITY;
  }
  if (status != SS_OK) {
    free(mesh->facet_plans);
    free(mesh->facet_plan_support_values);
    mesh->facet_plans = NULL;
    mesh->facet_plan_support_values = NULL;
    return status;
  }
  mesh->facet_plan_count = required;
  plan_bytes = (uint64_t)total_bytes;
  if (mesh->summary.cache_memory_bytes > UINT64_MAX - plan_bytes) {
    free(mesh->facet_plans);
    free(mesh->facet_plan_support_values);
    mesh->facet_plans = NULL;
    mesh->facet_plan_support_values = NULL;
    mesh->facet_plan_count = 0U;
    return SS_ERROR_OVERFLOW;
  }
  mesh->summary.owned_facet_plan_count = (uint64_t)required;
  mesh->summary.facet_plan_bytes = plan_bytes;
  mesh->summary.cache_memory_bytes += plan_bytes;
  return SS_OK;
}

SsStatus ss_mesh_pack_facets(const SsMesh *mesh, uint32_t support_value,
                             SsFacetPack *packs, size_t capacity, size_t *count) {
  size_t plan;
  size_t required = 0U;
  if (mesh == NULL || count == NULL || (packs == NULL && capacity != 0U)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (plan = 0U; plan < mesh->facet_plan_count; ++plan) {
    if (mesh->facet_plan_support_values[plan] == support_value) {
      ++required;
    }
  }
  *count = required;
  if (packs == NULL) {
    return SS_OK;
  }
  if (capacity < required) {
    return SS_ERROR_LIMIT;
  }
  required = 0U;
  for (plan = 0U; plan < mesh->facet_plan_count; ++plan) {
    if (mesh->facet_plan_support_values[plan] == support_value) {
      packs[required++] = mesh->facet_plans[plan];
    }
  }
  *count = required;
  return SS_OK;
}

static uint32_t ss_mesh_facet_opposite_vertex(uint32_t dimension,
                                              uint32_t facet_index) {
  static const uint32_t triangle_opposite[3] = {2U, 0U, 1U};
  static const uint32_t tetrahedron_opposite[4] = {3U, 2U, 1U, 0U};
  return dimension == 2U ? triangle_opposite[facet_index]
                         : tetrahedron_opposite[facet_index];
}

static SsStatus ss_field_space_pack_facets(const SsFieldSpace *space,
                                           uint32_t support_value, int owned_only,
                                           SsFieldFacetPack *packs, size_t capacity,
                                           size_t *count) {
  SsFacetPack *geometry_packs = NULL;
  size_t required = 0U;
  size_t geometry_count = 0U;
  size_t pack_index;
  SsStatus status = SS_OK;
  if (space == NULL || count == NULL || (packs == NULL && capacity != 0U)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = owned_only != 0
               ? ss_mesh_pack_facets(space->mesh, support_value, NULL, 0U, &required)
               : ss_mesh_pack_facets_uncached(space->mesh, support_value, 0, NULL, 0U,
                                              &required);
  status = ss_status_allreduce_max(space->mesh->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  *count = required;
  if (packs == NULL) {
    return SS_OK;
  }
  if (capacity < required) {
    status = SS_ERROR_LIMIT;
  }
  status = ss_status_allreduce_max(space->mesh->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  geometry_packs =
      (SsFacetPack *)calloc(required == 0U ? 1U : required, sizeof(*geometry_packs));
  if (geometry_packs == NULL) {
    status = SS_ERROR_OUT_OF_MEMORY;
  }
  status = ss_status_allreduce_max(space->mesh->communicator, status);
  if (status != SS_OK) {
    free(geometry_packs);
    return status;
  }
  status =
      owned_only != 0
          ? ss_mesh_pack_facets(space->mesh, support_value, geometry_packs, required,
                                &geometry_count)
          : ss_mesh_pack_facets_uncached(space->mesh, support_value, 0, geometry_packs,
                                         required, &geometry_count);
  if (status == SS_OK && geometry_count != required) {
    status = SS_ERROR_INTEGRITY;
  }
  status = ss_status_allreduce_max(space->mesh->communicator, status);
  if (status != SS_OK) {
    free(geometry_packs);
    return status;
  }
  for (pack_index = 0U; pack_index < required; ++pack_index) {
    SsFieldFacetPack *pack = &packs[pack_index];
    const SsCellPlan *parent =
        owned_only != 0 ? ss_field_owned_cell_plan_find(
                              space, geometry_packs[pack_index].stable_parent_cell_id)
                        : ss_field_incident_cell_plan_find(
                              space, geometry_packs[pack_index].stable_parent_cell_id);
    uint32_t opposite = 0U;
    size_t parent_node;
    memset(pack, 0, sizeof(*pack));
    if (geometry_packs[pack_index].facet_index > space->mesh->dimension) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    opposite = ss_mesh_facet_opposite_vertex(space->mesh->dimension,
                                             geometry_packs[pack_index].facet_index);
    if (parent == NULL) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
    pack->geometry = geometry_packs[pack_index];
    pack->order = space->summary.order;
    pack->components = space->summary.components;
    for (parent_node = 0U; parent_node < space->descriptor->node_count; ++parent_node) {
      const size_t barycentric_offset =
          parent_node * (size_t)(space->mesh->dimension + 1U) + opposite;
      size_t facet_node;
      uint32_t component;
      if (space->descriptor->barycentric_indices[barycentric_offset] != 0U) {
        continue;
      }
      facet_node = pack->field_node_count;
      if (facet_node >= SS_FIELD_MAX_FACET_NODES) {
        status = SS_ERROR_LIMIT;
        break;
      }
      memcpy(pack->field_node_coordinates + facet_node * (size_t)space->mesh->dimension,
             parent->field_node_coordinates +
                 parent_node * (size_t)space->mesh->dimension,
             (size_t)space->mesh->dimension * sizeof(pack->field_node_coordinates[0]));
      for (component = 0U; component < space->summary.components; ++component) {
        const size_t source =
            parent_node * (size_t)space->summary.components + component;
        const size_t destination =
            facet_node * (size_t)space->summary.components + component;
        if (source >= parent->closure_count ||
            destination >= SS_FIELD_MAX_FACET_CLOSURE) {
          status = SS_ERROR_LIMIT;
          break;
        }
        pack->parent_closure_slots[destination] = source;
        pack->field_entity_ids[destination] = parent->field_entity_ids[source];
        pack->field_entity_slots[destination] = parent->field_entity_slots[source];
        pack->field_components[destination] = parent->field_components[source];
        pack->local_indices[destination] = parent->local_indices[source];
        pack->global_indices[destination] = parent->global_indices[source];
        pack->constrained[destination] = parent->constrained[source];
        pack->owned[destination] = parent->owned[source];
        pack->prescribed_values[destination] = parent->prescribed_values[source];
      }
      if (status != SS_OK) {
        break;
      }
      ++pack->field_node_count;
    }
    pack->closure_count = pack->field_node_count * (size_t)space->summary.components;
    if (status == SS_OK &&
        pack->field_node_count != space->descriptor->facet_node_count) {
      status = SS_ERROR_INTEGRITY;
      break;
    }
  }
  free(geometry_packs);
  status = ss_status_allreduce_max(space->mesh->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  *count = required;
  return SS_OK;
}

SsStatus ss_field_space_pack_owned_facets(const SsFieldSpace *space,
                                          uint32_t support_value,
                                          SsFieldFacetPack *packs, size_t capacity,
                                          size_t *count) {
  return ss_field_space_pack_facets(space, support_value, 1, packs, capacity, count);
}

SsStatus ss_field_space_pack_incident_facets(const SsFieldSpace *space,
                                             uint32_t support_value,
                                             SsFieldFacetPack *packs, size_t capacity,
                                             size_t *count) {
  return ss_field_space_pack_facets(space, support_value, 0, packs, capacity, count);
}
