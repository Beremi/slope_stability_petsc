/* Audit element geometry through the quadrature registry.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "mesh_internal.h"
#include "slope_internal/core/collective.h"

#include "slope_internal/fe/fe.h"

#include <petscdmplex.h>
#include <petsclog.h>

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int ss_mesh_geometry_audit_rule_eligible(const SsFeQuadrature *quadrature,
                                                uint32_t dimension) {
  return quadrature != NULL && quadrature->domain == SS_FE_QUADRATURE_VOLUME &&
                 quadrature->parent_dimension == dimension &&
                 quadrature->dimension == dimension &&
                 quadrature->has_negative_weight == 0
             ? 1
             : 0;
}

static SsStatus ss_mesh_geometry_audit_registry_summary(SsMesh *mesh) {
  unsigned char *projection = NULL;
  size_t capacity = 0U;
  size_t used = 0U;
  size_t registry_index;
  uint64_t rule_count = 0U;
  uint64_t point_count = 0U;
  SsStatus status = SS_OK;
  if (mesh == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (registry_index = 0U; registry_index < ss_fe_quadrature_registry_count();
       ++registry_index) {
    const SsFeQuadrature *quadrature = NULL;
    size_t record_capacity;
    if (ss_fe_quadrature_registry_at(registry_index, &quadrature) != SS_OK ||
        quadrature == NULL) {
      return SS_ERROR_INTERNAL;
    }
    if (ss_mesh_geometry_audit_rule_eligible(quadrature, mesh->dimension) == 0) {
      continue;
    }
    if (strlen(quadrature->id) > SIZE_MAX - 64U) {
      return SS_ERROR_OVERFLOW;
    }
    record_capacity = strlen(quadrature->id) + 64U;
    if (capacity > SIZE_MAX - record_capacity ||
        point_count > UINT64_MAX - (uint64_t)quadrature->point_count ||
        rule_count == UINT64_MAX) {
      return SS_ERROR_OVERFLOW;
    }
    capacity += record_capacity;
    point_count += (uint64_t)quadrature->point_count;
    ++rule_count;
  }
  projection = (unsigned char *)malloc(capacity == 0U ? 1U : capacity);
  if (projection == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (registry_index = 0U; registry_index < ss_fe_quadrature_registry_count();
       ++registry_index) {
    const SsFeQuadrature *quadrature = NULL;
    int written;
    if (ss_fe_quadrature_registry_at(registry_index, &quadrature) != SS_OK ||
        quadrature == NULL) {
      status = SS_ERROR_INTERNAL;
      break;
    }
    if (ss_mesh_geometry_audit_rule_eligible(quadrature, mesh->dimension) == 0) {
      continue;
    }
    written =
        snprintf((char *)projection + used, capacity - used, "%s|%u|%zu\n",
                 quadrature->id, quadrature->exact_degree, quadrature->point_count);
    if (written < 0 || (size_t)written >= capacity - used) {
      status = SS_ERROR_LIMIT;
      break;
    }
    used += (size_t)written;
  }
  if (status == SS_OK) {
    status =
        ss_sha256_bytes(projection, used, mesh->summary.geometry_audit_rule_digest);
  }
  free(projection);
  if (status != SS_OK) {
    return status;
  }
  if (rule_count == 0U || point_count == 0U) {
    return SS_ERROR_NOT_FOUND;
  }
  mesh->summary.geometry_audit_rule_count = rule_count;
  mesh->summary.geometry_audit_point_count = point_count;
  return SS_OK;
}

SsStatus ss_mesh_collective_geometry_audit(SsMesh *mesh, SsMeshError *selected) {
  const SsFeQuadratureRequest quadrature_request = {
      mesh->dimension,
      SS_FE_QUADRATURE_VOLUME,
      SS_FE_OPERATION_GEOMETRY_QUALITY,
      1U,
      mesh->coordinate_order,
      0U,
      SS_FE_QUADRATURE_MODE_DEFAULT,
      NULL,
  };
  SsFeQuadraturePlan quadrature_plan = {0};
  const SsFeQuadrature *quality_quadrature = NULL;
  size_t registry_index;
  SsMeshError local = {0};
  PetscInt start = 0;
  PetscInt end = 0;
  PetscInt cell;
  double local_minimum = DBL_MAX;
  double local_maximum = -DBL_MAX;
  double global_minimum = 0.0;
  double global_maximum = 0.0;
  double local_minimum_condition = DBL_MAX;
  double local_maximum_condition = -DBL_MAX;
  double global_minimum_condition = 0.0;
  double global_maximum_condition = 0.0;
  uint64_t local_bad_id = UINT64_MAX;
  uint64_t global_bad_id = UINT64_MAX;
  int owner_rank;
  int local_owner;
  int raw[2];
  SsStatus scan_status = SS_OK;
  if (ss_fe_quadrature_plan(&quadrature_request, &quadrature_plan) != SS_OK ||
      DMPlexGetHeightStratum(mesh->dm, 0, &start, &end) != PETSC_SUCCESS) {
    scan_status = SS_ERROR_INTERNAL;
  } else {
    quality_quadrature = quadrature_plan.selected_rule;
    scan_status = ss_mesh_geometry_audit_registry_summary(mesh);
  }
  scan_status = ss_status_allreduce_max(mesh->communicator, scan_status);
  if (scan_status != SS_OK) {
    return scan_status;
  }
  if (quality_quadrature == NULL) {
    return SS_ERROR_INTERNAL;
  }
  {
    const int written = snprintf(mesh->summary.geometry_quadrature_id,
                                 sizeof(mesh->summary.geometry_quadrature_id), "%s",
                                 quality_quadrature->id);
    if (written < 0 ||
        (size_t)written >= sizeof(mesh->summary.geometry_quadrature_id)) {
      return SS_ERROR_LIMIT;
    }
  }
  for (cell = start; cell < end; ++cell) {
    SsEntityId cell_id = 0U;
    SsEntityId node_ids[SS_FE_MAX_NODES] = {0U};
    double coordinates[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
    size_t node_count = 0U;
    int has_positive = 0;
    int has_negative = 0;
    int has_near_singular = 0;
    int has_nonfinite = 0;
    size_t bad_quadrature = SIZE_MAX;
    double bad_determinant = 0.0;
    double bad_condition = 0.0;
    const char *bad_rule_id = quality_quadrature->id;
    size_t audit_point = 0U;
    size_t point;
    if (ss_mesh_point_is_ghost(mesh, mesh->dm, cell) != 0) {
      continue;
    }
    if (ss_mesh_cached_stable_id(mesh, cell, &cell_id) != SS_OK ||
        ss_mesh_cell_coordinate_closure(mesh, cell_id, node_ids, coordinates,
                                        SS_FE_MAX_NODES, &node_count) != SS_OK ||
        node_count != mesh->coordinate_descriptor->node_count) {
      scan_status = SS_ERROR_INTERNAL;
      break;
    }
    for (point = 0U; point < node_count * (size_t)mesh->dimension; ++point) {
      if (!isfinite(coordinates[point])) {
        has_nonfinite = 1;
        bad_quadrature = 0U;
        bad_determinant = NAN;
        bad_condition = NAN;
        break;
      }
    }
    for (registry_index = 0U; registry_index < ss_fe_quadrature_registry_count();
         ++registry_index) {
      const SsFeQuadrature *quadrature = NULL;
      if (ss_fe_quadrature_registry_at(registry_index, &quadrature) != SS_OK ||
          quadrature == NULL) {
        scan_status = SS_ERROR_INTERNAL;
        break;
      }
      if (ss_mesh_geometry_audit_rule_eligible(quadrature, mesh->dimension) == 0) {
        continue;
      }
      for (point = 0U; point < quadrature->point_count; ++point) {
        SsFeGeometry geometry = {0};
        const double *reference = quadrature->points + point * (size_t)mesh->dimension;
        const SsStatus geometry_status =
            ss_fe_geometry_eval(mesh->coordinate_descriptor, coordinates, reference,
                                &mesh->geometry_policy, &geometry);
        if (!isfinite(geometry.determinant) || !isfinite(geometry.condition_estimate)) {
          has_nonfinite = 1;
        } else if (geometry_status != SS_OK &&
                   geometry.determinant >
                       mesh->geometry_policy.absolute_determinant_tolerance &&
                   geometry.condition_estimate <=
                       mesh->geometry_policy.maximum_condition) {
          has_nonfinite = 1;
        } else {
          if (geometry.condition_estimate < local_minimum_condition) {
            local_minimum_condition = geometry.condition_estimate;
          }
          if (geometry.condition_estimate > local_maximum_condition) {
            local_maximum_condition = geometry.condition_estimate;
          }
          if (geometry.determinant >
              mesh->geometry_policy.absolute_determinant_tolerance) {
            has_positive = 1;
            if (geometry.determinant < local_minimum) {
              local_minimum = geometry.determinant;
            }
            if (geometry.determinant > local_maximum) {
              local_maximum = geometry.determinant;
            }
          } else if (geometry.determinant <
                     -mesh->geometry_policy.absolute_determinant_tolerance) {
            has_negative = 1;
          } else {
            has_near_singular = 1;
          }
          if (geometry.condition_estimate > mesh->geometry_policy.maximum_condition) {
            has_near_singular = 1;
          }
        }
        if ((has_nonfinite != 0 || has_near_singular != 0 || has_negative != 0) &&
            bad_quadrature == SIZE_MAX) {
          bad_quadrature = audit_point;
          bad_determinant = geometry.determinant;
          bad_condition = geometry.condition_estimate;
          bad_rule_id = quadrature->id;
        }
        ++audit_point;
      }
    }
    if (scan_status != SS_OK ||
        audit_point != (size_t)mesh->summary.geometry_audit_point_count) {
      scan_status = SS_ERROR_INTERNAL;
      break;
    }
    if (has_nonfinite != 0 || has_near_singular != 0 || has_negative != 0) {
      const int written = snprintf(local.quadrature_rule_id,
                                   sizeof(local.quadrature_rule_id), "%s", bad_rule_id);
      if (written < 0 || (size_t)written >= sizeof(local.quadrature_rule_id)) {
        scan_status = SS_ERROR_LIMIT;
        break;
      }
      local.valid = 1;
      local.status = SS_ERROR_INTEGRITY;
      local.stable_entity_id = cell_id;
      local.quadrature_index = bad_quadrature;
      local.determinant = bad_determinant;
      local.condition_estimate = bad_condition;
      if (has_nonfinite != 0) {
        (void)strcpy(local.code, "MESH_GEOMETRY_NONFINITE");
      } else if (has_positive != 0 && has_negative != 0) {
        (void)strcpy(local.code, "MESH_GEOMETRY_SIGN_CHANGE");
      } else if (has_negative != 0) {
        (void)strcpy(local.code, "MESH_GEOMETRY_INVERTED");
      } else {
        (void)strcpy(local.code, "MESH_GEOMETRY_NEAR_SINGULAR");
      }
      local_bad_id = cell_id;
      break;
    }
  }
  scan_status = ss_status_allreduce_max(mesh->communicator, scan_status);
  if (scan_status != SS_OK) {
    return scan_status;
  }
  if (MPI_Allreduce(&local_bad_id, &global_bad_id, 1, MPI_UINT64_T, MPI_MIN,
                    mesh->communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (global_bad_id != UINT64_MAX) {
    local_owner = local.valid != 0 && local.stable_entity_id == global_bad_id
                      ? mesh->rank
                      : mesh->size;
    if (MPI_Allreduce(&local_owner, &owner_rank, 1, MPI_INT, MPI_MIN,
                      mesh->communicator) != MPI_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    if (mesh->rank == owner_rank) {
      *selected = local;
    }
    if (MPI_Bcast(selected, (int)sizeof(*selected), MPI_BYTE, owner_rank,
                  mesh->communicator) != MPI_SUCCESS) {
      return SS_ERROR_INTERNAL;
    }
    return SS_ERROR_INTEGRITY;
  }
  if (local_minimum == DBL_MAX) {
    local_minimum = DBL_MAX;
  }
  if (local_maximum == -DBL_MAX) {
    local_maximum = -DBL_MAX;
  }
  if (MPI_Allreduce(&local_minimum, &global_minimum, 1, MPI_DOUBLE, MPI_MIN,
                    mesh->communicator) != MPI_SUCCESS ||
      MPI_Allreduce(&local_maximum, &global_maximum, 1, MPI_DOUBLE, MPI_MAX,
                    mesh->communicator) != MPI_SUCCESS ||
      MPI_Allreduce(&local_minimum_condition, &global_minimum_condition, 1, MPI_DOUBLE,
                    MPI_MIN, mesh->communicator) != MPI_SUCCESS ||
      MPI_Allreduce(&local_maximum_condition, &global_maximum_condition, 1, MPI_DOUBLE,
                    MPI_MAX, mesh->communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  mesh->summary.minimum_determinant = global_minimum;
  mesh->summary.maximum_determinant = global_maximum;
  mesh->summary.minimum_condition = global_minimum_condition;
  mesh->summary.maximum_condition = global_maximum_condition;
  raw[0] = global_minimum < DBL_MAX ? 1 : 0;
  raw[1] = global_maximum > -DBL_MAX ? 1 : 0;
  if (raw[0] == 0 || raw[1] == 0 || global_minimum_condition == DBL_MAX ||
      global_maximum_condition == -DBL_MAX) {
    return SS_ERROR_INTEGRITY;
  }
  return SS_OK;
}
