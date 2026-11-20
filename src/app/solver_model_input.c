/* Read compiled problem models, materials, supports and outputs.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/core/util.h"
#include "solver_input_internal.h"

#include "yyjson.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  const char *id;
  uint64_t mask;
} SsSolverOutputRegistryEntry;

static const SsSolverOutputRegistryEntry ss_solver_output_registry[] = {
    {"displacement", SS_SOLVER_QUANTITY_DISPLACEMENT},
    {"strain", SS_SOLVER_QUANTITY_STRAIN},
    {"stress", SS_SOLVER_QUANTITY_STRESS},
    {"material_branch", SS_SOLVER_QUANTITY_MATERIAL_BRANCH},
    {"plastic_indicator", SS_SOLVER_QUANTITY_PLASTIC_INDICATOR},
    {"yield_residual", SS_SOLVER_QUANTITY_YIELD_RESIDUAL},
    {"pore_pressure", SS_SOLVER_QUANTITY_PORE_PRESSURE},
    {"pressure_gradient", SS_SOLVER_QUANTITY_PRESSURE_GRADIENT},
    {"conductivity", SS_SOLVER_QUANTITY_CONDUCTIVITY},
    {"hydraulic_ramp", SS_SOLVER_QUANTITY_HYDRAULIC_RAMP},
    {"hydraulic_epsilon", SS_SOLVER_QUANTITY_HYDRAULIC_EPSILON},
    {"saturation", SS_SOLVER_QUANTITY_SATURATION},
    {"pressure_flux", SS_SOLVER_QUANTITY_PRESSURE_FLUX},
    {"darcy_velocity", SS_SOLVER_QUANTITY_DARCY_VELOCITY},
    {"coupled_body_force", SS_SOLVER_QUANTITY_COUPLED_BODY_FORCE},
    {"global_flux_balance", SS_SOLVER_QUANTITY_GLOBAL_FLUX_BALANCE},
    {"continuation", SS_SOLVER_QUANTITY_CONTINUATION},
    {"nonlinear_history", SS_SOLVER_QUANTITY_NONLINEAR_HISTORY},
    {"safety_factor", SS_SOLVER_QUANTITY_SAFETY_FACTOR},
    {"load_factor", SS_SOLVER_QUANTITY_LOAD_FACTOR},
    {"work_control", SS_SOLVER_QUANTITY_WORK_CONTROL},
    {"maximum_displacement", SS_SOLVER_QUANTITY_MAXIMUM_DISPLACEMENT},
};

SsStatus ss_solver_read_file(const char *path, unsigned char **data, size_t *size) {
  FILE *stream;
  long extent;
  size_t read_size;
  if (path == NULL || data == NULL || size == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *data = NULL;
  *size = 0U;
  stream = fopen(path, "rb");
  if (stream == NULL) {
    return SS_ERROR_IO;
  }
  if (fseek(stream, 0L, SEEK_END) != 0 || (extent = ftell(stream)) < 0L ||
      fseek(stream, 0L, SEEK_SET) != 0) {
    (void)fclose(stream);
    return SS_ERROR_IO;
  }
  if ((uintmax_t)extent > (uintmax_t)SIZE_MAX - 1U) {
    (void)fclose(stream);
    return SS_ERROR_OVERFLOW;
  }
  *size = (size_t)extent;
  *data = (unsigned char *)malloc(*size + 1U);
  if (*data == NULL) {
    (void)fclose(stream);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  read_size = fread(*data, 1U, *size, stream);
  if (read_size != *size || fclose(stream) != 0) {
    free(*data);
    *data = NULL;
    *size = 0U;
    return SS_ERROR_IO;
  }
  (*data)[*size] = '\0';
  return SS_OK;
}

int ss_solver_json_string_equal(yyjson_val *value, const char *expected) {
  const size_t expected_length = expected == NULL ? 0U : strlen(expected);
  return yyjson_is_str(value) && expected != NULL &&
         yyjson_get_len(value) == expected_length &&
         memcmp(yyjson_get_str(value), expected, expected_length) == 0;
}

yyjson_val *ss_solver_object(yyjson_val *parent, const char *name) {
  yyjson_val *value = parent == NULL ? NULL : yyjson_obj_get(parent, name);
  return yyjson_is_obj(value) ? value : NULL;
}

const char *ss_solver_string(yyjson_val *parent, const char *name) {
  yyjson_val *value = parent == NULL ? NULL : yyjson_obj_get(parent, name);
  return yyjson_is_str(value) ? yyjson_get_str(value) : NULL;
}

SsStatus ss_solver_parse_output_quantities(yyjson_val *array, uint64_t *quantity_mask) {
  size_t index;
  size_t maximum;
  yyjson_val *item;
  uint64_t parsed = 0U;
  if (!yyjson_is_arr(array) || yyjson_arr_size(array) == 0U || quantity_mask == NULL) {
    return SS_ERROR_SCHEMA;
  }
  yyjson_arr_foreach(array, index, maximum, item) {
    const char *id = yyjson_is_str(item) ? yyjson_get_str(item) : NULL;
    uint64_t item_mask = 0U;
    size_t registry_index;
    if (id == NULL) {
      return SS_ERROR_SCHEMA;
    }
    for (registry_index = 0U; registry_index < sizeof(ss_solver_output_registry) /
                                                   sizeof(ss_solver_output_registry[0]);
         ++registry_index) {
      if (strcmp(id, ss_solver_output_registry[registry_index].id) == 0) {
        item_mask = ss_solver_output_registry[registry_index].mask;
        break;
      }
    }
    if (item_mask == 0U || (parsed & item_mask) != 0U) {
      return SS_ERROR_SCHEMA;
    }
    parsed |= item_mask;
  }
  *quantity_mask = parsed;
  return SS_OK;
}

uint64_t ss_solver_available_output_quantities(const SsSolverProblemModel *model) {
  uint64_t available = SS_SOLVER_QUANTITY_NONLINEAR_HISTORY;
  if (model->mechanics_enabled != 0) {
    available |= SS_SOLVER_QUANTITY_DISPLACEMENT | SS_SOLVER_QUANTITY_STRAIN |
                 SS_SOLVER_QUANTITY_STRESS | SS_SOLVER_QUANTITY_MATERIAL_BRANCH |
                 SS_SOLVER_QUANTITY_PLASTIC_INDICATOR |
                 SS_SOLVER_QUANTITY_YIELD_RESIDUAL | SS_SOLVER_QUANTITY_CONTINUATION |
                 SS_SOLVER_QUANTITY_WORK_CONTROL |
                 SS_SOLVER_QUANTITY_MAXIMUM_DISPLACEMENT;
    available |= model->analysis_method == SS_ANALYSIS_LIMIT_LOAD
                     ? SS_SOLVER_QUANTITY_LOAD_FACTOR
                     : SS_SOLVER_QUANTITY_SAFETY_FACTOR;
  }
  if (model->hydraulic_enabled != 0) {
    available |= SS_SOLVER_QUANTITY_PORE_PRESSURE |
                 SS_SOLVER_QUANTITY_PRESSURE_GRADIENT |
                 SS_SOLVER_QUANTITY_CONDUCTIVITY | SS_SOLVER_QUANTITY_HYDRAULIC_RAMP |
                 SS_SOLVER_QUANTITY_HYDRAULIC_EPSILON | SS_SOLVER_QUANTITY_SATURATION |
                 SS_SOLVER_QUANTITY_PRESSURE_FLUX | SS_SOLVER_QUANTITY_DARCY_VELOCITY |
                 SS_SOLVER_QUANTITY_GLOBAL_FLUX_BALANCE;
  }
  if (model->coupling_enabled != 0) {
    available |= SS_SOLVER_QUANTITY_COUPLED_BODY_FORCE;
  }
  return available;
}

int ss_solver_boolean(yyjson_val *parent, const char *name, int fallback) {
  yyjson_val *value = parent == NULL ? NULL : yyjson_obj_get(parent, name);
  return yyjson_is_bool(value) ? yyjson_get_bool(value) : fallback;
}

double ss_solver_number(yyjson_val *parent, const char *name, double fallback) {
  yyjson_val *value = parent == NULL ? NULL : yyjson_obj_get(parent, name);
  return yyjson_is_num(value) ? yyjson_get_num(value) : fallback;
}

uint64_t ss_solver_uint(yyjson_val *parent, const char *name, uint64_t fallback) {
  yyjson_val *value = parent == NULL ? NULL : yyjson_obj_get(parent, name);
  return yyjson_is_uint(value) ? yyjson_get_uint(value) : fallback;
}

static double ss_solver_parameter(yyjson_val *parameters, const char *name,
                                  double fallback) {
  yyjson_val *record = parameters == NULL ? NULL : yyjson_obj_get(parameters, name);
  yyjson_val *value;
  const char *unit;
  double number;
  if (yyjson_is_num(record)) {
    return yyjson_get_num(record);
  }
  if (!yyjson_is_obj(record)) {
    return fallback;
  }
  value = yyjson_obj_get(record, "value");
  if (!yyjson_is_num(value)) {
    return fallback;
  }
  number = yyjson_get_num(value);
  unit = ss_solver_string(record, "unit");
  if (unit != NULL && strcmp(unit, "degree") == 0) {
    number *= 0.017453292519943295769236907684886;
  }
  return number;
}

static SsStatus ss_solver_number_vector(yyjson_val *value, size_t expected_count,
                                        double *output) {
  size_t index;
  if (!yyjson_is_arr(value) || output == NULL ||
      yyjson_arr_size(value) != expected_count) {
    return SS_ERROR_SCHEMA;
  }
  for (index = 0U; index < expected_count; ++index) {
    yyjson_val *item = yyjson_arr_get(value, index);
    if (!yyjson_is_num(item) || !isfinite(yyjson_get_num(item))) {
      return SS_ERROR_SCHEMA;
    }
    output[index] = yyjson_get_num(item);
  }
  return SS_OK;
}

static SsStatus ss_solver_displacement_vector(yyjson_val *value, yyjson_val *components,
                                              size_t dimension, double *output) {
  const size_t component_count =
      yyjson_is_arr(components) ? yyjson_arr_size(components) : 0U;
  const size_t value_count = yyjson_is_arr(value) ? yyjson_arr_size(value) : 0U;
  size_t index;
  if (output == NULL || component_count == 0U || value_count == 0U) {
    return SS_ERROR_SCHEMA;
  }
  memset(output, 0, dimension * sizeof(*output));
  if (component_count == 1U) {
    yyjson_val *component = yyjson_arr_get(components, 0U);
    const char *name = yyjson_is_str(component) ? yyjson_get_str(component) : NULL;
    if (name != NULL && strcmp(name, "all") == 0) {
      return ss_solver_number_vector(value, dimension, output);
    }
  }
  if (value_count == component_count) {
    for (index = 0U; index < component_count; ++index) {
      yyjson_val *component = yyjson_arr_get(components, index);
      yyjson_val *item = yyjson_arr_get(value, index);
      const char *name = yyjson_is_str(component) ? yyjson_get_str(component) : NULL;
      size_t axis;
      if (name == NULL || !yyjson_is_num(item) || !isfinite(yyjson_get_num(item))) {
        return SS_ERROR_SCHEMA;
      }
      if (strcmp(name, "x") == 0) {
        axis = 0U;
      } else if (strcmp(name, "y") == 0 && dimension >= 2U) {
        axis = 1U;
      } else if (strcmp(name, "z") == 0 && dimension == 3U) {
        axis = 2U;
      } else {
        return SS_ERROR_SCHEMA;
      }
      output[axis] = yyjson_get_num(item);
    }
    return SS_OK;
  }
  return ss_solver_number_vector(value, dimension, output);
}

static SsStatus ss_solver_number_matrix(yyjson_val *value, size_t expected_extent,
                                        double *output) {
  size_t row;
  if (!yyjson_is_arr(value) || output == NULL ||
      yyjson_arr_size(value) != expected_extent) {
    return SS_ERROR_SCHEMA;
  }
  for (row = 0U; row < expected_extent; ++row) {
    SsStatus status = ss_solver_number_vector(
        yyjson_arr_get(value, row), expected_extent, output + row * expected_extent);
    if (status != SS_OK) {
      return status;
    }
  }
  return SS_OK;
}

static uint32_t ss_solver_material_support_value(const SsSolverProblemModel *model,
                                                 size_t material_index,
                                                 const char *support) {
  size_t index;
  uint32_t maximum = 0U;
  for (index = 0U; index < material_index; ++index) {
    if (strcmp(model->materials[index].support, support) == 0) {
      return model->materials[index].support_value;
    }
    if (model->materials[index].support_value > maximum) {
      maximum = model->materials[index].support_value;
    }
  }
  return maximum + 1U;
}

static SsStatus ss_solver_parse_prescribed_saturation(yyjson_val *root,
                                                      SsSolverProblemModel *model) {
  yyjson_val *record = yyjson_obj_get(root, "prescribed_saturation");
  yyjson_val *coordinates, *levels, *axis;
  const char *unit;
  size_t index;
  if (record == NULL)
    return SS_OK;
  if (!yyjson_is_obj(record) || yyjson_obj_size(record) != 4U ||
      model->mechanics_enabled == 0 || model->hydraulic_enabled != 0 ||
      model->coupling_enabled != 0) {
    return SS_ERROR_SCHEMA;
  }
  axis = yyjson_obj_get(record, "axis");
  coordinates = yyjson_obj_get(record, "coordinates");
  levels = yyjson_obj_get(record, "levels");
  unit = ss_solver_string(record, "length_unit");
  if (!yyjson_is_uint(axis) || yyjson_get_uint(axis) > 2U ||
      !yyjson_is_arr(coordinates) || !yyjson_is_arr(levels) || unit == NULL ||
      strcmp(unit, "m") != 0 ||
      yyjson_arr_size(coordinates) != yyjson_arr_size(levels) ||
      yyjson_arr_size(coordinates) == 0U ||
      yyjson_arr_size(coordinates) > SS_SOLVER_SATURATION_CAPACITY) {
    return SS_ERROR_SCHEMA;
  }
  model->prescribed_saturation.axis = (uint32_t)yyjson_get_uint(axis);
  model->prescribed_saturation.count = yyjson_arr_size(coordinates);
  for (index = 0U; index < model->prescribed_saturation.count; ++index) {
    yyjson_val *coordinate = yyjson_arr_get(coordinates, index);
    yyjson_val *level = yyjson_arr_get(levels, index);
    if (!yyjson_is_num(coordinate) || !yyjson_is_num(level))
      return SS_ERROR_SCHEMA;
    model->prescribed_saturation.coordinates[index] = yyjson_get_num(coordinate);
    model->prescribed_saturation.levels[index] = yyjson_get_num(level);
  }
  return ss_solver_saturation_validate(&model->prescribed_saturation,
                                       model->dimension) == SS_OK
             ? SS_OK
             : SS_ERROR_SCHEMA;
}

static SsStatus ss_solver_parse_materials(yyjson_val *root,
                                          const SsCompiledManifest *manifest, int rank,
                                          int ranks, SsSolverProblemModel *model) {
  yyjson_val *materials = ss_solver_object(root, "materials");
  yyjson_obj_iter iterator;
  yyjson_val *key;
  size_t index = 0U;
  if (materials == NULL) {
    return SS_ERROR_SCHEMA;
  }
  model->material_count = yyjson_obj_size(materials);
  model->materials = (SsSolverMaterial *)calloc(
      model->material_count == 0U ? 1U : model->material_count,
      sizeof(*model->materials));
  if (model->materials == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  iterator = yyjson_obj_iter_with(materials);
  while ((key = yyjson_obj_iter_next(&iterator)) != NULL) {
    yyjson_val *record = yyjson_obj_iter_get_val(key);
    yyjson_val *parameters = ss_solver_object(record, "parameters");
    const char *name = ss_solver_string(record, "name");
    const char *support = ss_solver_string(record, "support");
    const char *material_model = ss_solver_string(record, "model");
    const char *davis;
    SsSolverMaterial *material;
    SsStatus status;
    if (index >= model->material_count || !yyjson_is_obj(record) || name == NULL ||
        support == NULL || material_model == NULL || parameters == NULL) {
      return SS_ERROR_SCHEMA;
    }
    material = &model->materials[index];
    if (!ss_copy_text(material->name, sizeof(material->name), name) ||
        !ss_copy_text(material->support, sizeof(material->support), support)) {
      return SS_ERROR_LIMIT;
    }
    material->support_value =
        ss_solver_material_support_value(model, index, material->support);
    uint64_t global_count = 0U;
    status = ss_io_read_manifest_ids(manifest, SS_MANIFEST_MATERIAL_IDS, index, rank,
                                     ranks, &material->entity_ids,
                                     &material->entity_count, &global_count);
    if (status != SS_OK) {
      return status;
    }
    if (strcmp(material_model, "mohr_coulomb") == 0) {
      material->kind = SS_SOLVER_MATERIAL_MECHANICS;
      material->young_modulus = ss_solver_parameter(parameters, "young_modulus", NAN);
      material->poisson_ratio = ss_solver_parameter(parameters, "poisson_ratio", NAN);
      material->cohesion = ss_solver_parameter(parameters, "cohesion", NAN);
      material->friction_angle = ss_solver_parameter(parameters, "friction_angle", NAN);
      material->dilatancy_angle =
          ss_solver_parameter(parameters, "dilation_angle", NAN);
      material->unsaturated_unit_weight =
          ss_solver_parameter(parameters, "unsaturated_unit_weight",
                              ss_solver_parameter(parameters, "unit_weight", NAN));
      material->saturated_unit_weight = ss_solver_parameter(
          parameters, "saturated_unit_weight", material->unsaturated_unit_weight);
      davis = ss_solver_string(parameters, "davis");
      material->davis = davis != NULL && strcmp(davis, "A") == 0   ? SS_MC_DAVIS_A
                        : davis != NULL && strcmp(davis, "C") == 0 ? SS_MC_DAVIS_C
                                                                   : SS_MC_DAVIS_B;
      if (!isfinite(material->young_modulus) || !isfinite(material->poisson_ratio) ||
          !isfinite(material->cohesion) || !isfinite(material->friction_angle) ||
          !isfinite(material->dilatancy_angle) ||
          !isfinite(material->unsaturated_unit_weight) ||
          (model->prescribed_saturation.count != 0U &&
           (!isfinite(material->saturated_unit_weight) ||
            material->saturated_unit_weight < 0.0))) {
        return SS_ERROR_SCHEMA;
      }
    } else if (strcmp(material_model, "steady_unconfined_darcy") == 0) {
      material->kind = SS_SOLVER_MATERIAL_HYDRAULIC;
      material->conductivity =
          ss_solver_parameter(parameters, "hydraulic_conductivity",
                              ss_solver_parameter(parameters, "conductivity", NAN));
      material->water_unit_weight =
          ss_solver_parameter(parameters, "water_unit_weight", 9.81);
      if (!isfinite(material->conductivity) || material->conductivity <= 0.0 ||
          !isfinite(material->water_unit_weight) ||
          material->water_unit_weight <= 0.0) {
        return SS_ERROR_SCHEMA;
      }
    } else {
      return SS_ERROR_SCHEMA;
    }
    ++index;
  }
  return index == model->material_count ? SS_OK : SS_ERROR_SCHEMA;
}

static SsStatus ss_solver_parse_boundaries(yyjson_val *root,
                                           const SsCompiledManifest *manifest, int rank,
                                           int ranks, SsSolverProblemModel *model) {
  yyjson_val *boundaries = ss_solver_object(root, "boundary_rules");
  yyjson_obj_iter iterator;
  yyjson_val *key;
  size_t index = 0U;
  if (boundaries == NULL || yyjson_obj_size(boundaries) > (size_t)UINT32_MAX - 1U) {
    return SS_ERROR_SCHEMA;
  }
  model->boundary_count = yyjson_obj_size(boundaries);
  model->boundaries = (SsSolverBoundaryRule *)calloc(
      model->boundary_count == 0U ? 1U : model->boundary_count,
      sizeof(*model->boundaries));
  if (model->boundaries == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  iterator = yyjson_obj_iter_with(boundaries);
  while ((key = yyjson_obj_iter_next(&iterator)) != NULL) {
    yyjson_val *record = yyjson_obj_iter_get_val(key);
    yyjson_val *parameters = ss_solver_object(record, "parameters");
    yyjson_val *components = yyjson_obj_get(record, "components");
    const char *name = ss_solver_string(record, "name");
    const char *field = ss_solver_string(record, "field");
    const char *condition = ss_solver_string(record, "condition");
    const char *support = ss_solver_string(record, "support");
    const char *value_model = ss_solver_string(record, "value_model");
    const char *geometry_patch = ss_solver_string(record, "geometry_patch");
    SsSolverBoundaryRule *rule;
    SsStatus status;
    size_t item_index;
    size_t maximum;
    yyjson_val *item;
    int saw_all = 0;
    int saw_normal = 0;
    int saw_scalar = 0;
    int saw_coordinate = 0;
    if (index >= model->boundary_count || !yyjson_is_obj(record) || name == NULL ||
        field == NULL || condition == NULL || support == NULL || value_model == NULL ||
        parameters == NULL) {
      return SS_ERROR_SCHEMA;
    }
    rule = &model->boundaries[index];
    if (!ss_copy_text(rule->name, sizeof(rule->name), name) ||
        !ss_copy_text(rule->field, sizeof(rule->field),
                      strcmp(field, "pore_pressure") == 0 ? "pressure" : field) ||
        !ss_copy_text(rule->support, sizeof(rule->support), support)) {
      return SS_ERROR_LIMIT;
    }
    if (strcmp(condition, "essential") == 0) {
      rule->condition_kind = SS_SOLVER_BOUNDARY_ESSENTIAL;
    } else if (strcmp(condition, "natural") == 0 && geometry_patch != NULL &&
               geometry_patch[0] != '\0') {
      rule->condition_kind = SS_SOLVER_BOUNDARY_NATURAL;
    } else {
      return SS_ERROR_SCHEMA;
    }
    rule->support_value = (uint32_t)index + 2U;
    uint64_t global_count = 0U;
    status =
        ss_io_read_manifest_ids(manifest, SS_MANIFEST_BOUNDARY_IDS, index, rank, ranks,
                                &rule->entity_ids, &rule->entity_count, &global_count);
    if (status != SS_OK) {
      return status;
    }
    if (global_count == 0U) {
      return SS_ERROR_SCHEMA;
    }
    if (strcmp(value_model, "constant_vector") == 0 ||
        strcmp(value_model, "constant_scalar") == 0) {
      rule->value_kind = SS_SOLVER_VALUE_CONSTANT;
    } else if (strcmp(value_model, "hydrostatic") == 0) {
      rule->value_kind = SS_SOLVER_VALUE_HYDROSTATIC;
    } else if (strcmp(value_model, "piecewise_linear_level") == 0) {
      rule->value_kind = SS_SOLVER_VALUE_PIECEWISE_LEVEL;
    } else if (strcmp(value_model, "global_traction") == 0) {
      rule->value_kind = SS_SOLVER_VALUE_GLOBAL_TRACTION;
    } else if (strcmp(value_model, "spatial_global_traction") == 0) {
      rule->value_kind = SS_SOLVER_VALUE_SPATIAL_GLOBAL_TRACTION;
    } else if (strcmp(value_model, "normal_pressure") == 0) {
      rule->value_kind = SS_SOLVER_VALUE_NORMAL_PRESSURE;
    } else if (strcmp(value_model, "prescribed_flux") == 0) {
      rule->value_kind = SS_SOLVER_VALUE_PRESCRIBED_FLUX;
    } else if (strcmp(value_model, "affine_coordinate") == 0) {
      rule->value_kind = SS_SOLVER_VALUE_AFFINE_COORDINATE;
    } else {
      return SS_ERROR_SCHEMA;
    }
    if (!yyjson_is_arr(components) || yyjson_arr_size(components) == 0U) {
      return SS_ERROR_SCHEMA;
    }
    yyjson_arr_foreach(components, item_index, maximum, item) {
      const char *component = yyjson_is_str(item) ? yyjson_get_str(item) : NULL;
      uint32_t bit = 0U;
      if (component == NULL) {
        return SS_ERROR_SCHEMA;
      }
      if (strcmp(component, "x") == 0) {
        bit = 1U;
        saw_coordinate = 1;
      } else if (strcmp(component, "y") == 0 && model->dimension >= 2U) {
        bit = 2U;
        saw_coordinate = 1;
      } else if (strcmp(component, "z") == 0 && model->dimension == 3U) {
        bit = 4U;
        saw_coordinate = 1;
      } else if (strcmp(component, "pressure") == 0 ||
                 strcmp(component, "scalar") == 0) {
        bit = 1U;
        saw_scalar = 1;
      } else if (strcmp(component, "all") == 0) {
        bit = (UINT32_C(1) << model->dimension) - UINT32_C(1);
        saw_all = 1;
      } else if (strcmp(component, "normal") == 0) {
        bit = (UINT32_C(1) << model->dimension) - UINT32_C(1);
        saw_normal = 1;
      } else {
        return SS_ERROR_SCHEMA;
      }
      if ((rule->component_mask & bit) != 0U) {
        return SS_ERROR_SCHEMA;
      }
      rule->component_mask |= bit;
    }
    if ((saw_all != 0 || saw_normal != 0) && yyjson_arr_size(components) != 1U) {
      return SS_ERROR_SCHEMA;
    }

    if (strcmp(rule->field, "displacement") == 0) {
      if (model->mechanics_enabled == 0) {
        return SS_ERROR_SCHEMA;
      }
      if (rule->condition_kind == SS_SOLVER_BOUNDARY_ESSENTIAL) {
        if ((rule->value_kind != SS_SOLVER_VALUE_CONSTANT &&
             rule->value_kind != SS_SOLVER_VALUE_AFFINE_COORDINATE) ||
            saw_normal != 0 || saw_scalar != 0 || rule->component_mask == 0U) {
          return SS_ERROR_SCHEMA;
        }
      } else if ((rule->value_kind == SS_SOLVER_VALUE_GLOBAL_TRACTION ||
                  rule->value_kind == SS_SOLVER_VALUE_SPATIAL_GLOBAL_TRACTION) &&
                 saw_all == 0) {
        return SS_ERROR_SCHEMA;
      } else if ((rule->value_kind == SS_SOLVER_VALUE_NORMAL_PRESSURE ||
                  rule->value_kind == SS_SOLVER_VALUE_HYDROSTATIC) &&
                 saw_normal == 0) {
        return SS_ERROR_SCHEMA;
      } else if (rule->value_kind != SS_SOLVER_VALUE_GLOBAL_TRACTION &&
                 rule->value_kind != SS_SOLVER_VALUE_SPATIAL_GLOBAL_TRACTION &&
                 rule->value_kind != SS_SOLVER_VALUE_NORMAL_PRESSURE &&
                 rule->value_kind != SS_SOLVER_VALUE_HYDROSTATIC) {
        return SS_ERROR_SCHEMA;
      }
    } else if (strcmp(rule->field, "pressure") == 0) {
      if (model->hydraulic_enabled == 0 || saw_coordinate != 0 || saw_all != 0) {
        return SS_ERROR_SCHEMA;
      }
      if (rule->condition_kind == SS_SOLVER_BOUNDARY_ESSENTIAL) {
        if (saw_scalar == 0 || (rule->value_kind != SS_SOLVER_VALUE_CONSTANT &&
                                rule->value_kind != SS_SOLVER_VALUE_AFFINE_COORDINATE &&
                                rule->value_kind != SS_SOLVER_VALUE_HYDROSTATIC &&
                                rule->value_kind != SS_SOLVER_VALUE_PIECEWISE_LEVEL)) {
          return SS_ERROR_SCHEMA;
        }
      } else if (saw_normal == 0 ||
                 rule->value_kind != SS_SOLVER_VALUE_PRESCRIBED_FLUX) {
        return SS_ERROR_SCHEMA;
      }
    } else {
      return SS_ERROR_SCHEMA;
    }

    if (rule->value_kind == SS_SOLVER_VALUE_CONSTANT) {
      yyjson_val *value = yyjson_obj_get(parameters, "value");
      if (strcmp(rule->field, "displacement") == 0) {
        status = ss_solver_displacement_vector(value, components, model->dimension,
                                               rule->values);
        rule->value_count = status == SS_OK ? model->dimension : 0U;
      } else if (!yyjson_is_num(value) || !isfinite(yyjson_get_num(value))) {
        status = SS_ERROR_SCHEMA;
      } else {
        rule->values[0] = yyjson_get_num(value);
        rule->value_count = 1U;
        status = SS_OK;
      }
      if (status != SS_OK) {
        return status;
      }
    } else if (rule->value_kind == SS_SOLVER_VALUE_GLOBAL_TRACTION) {
      status = ss_solver_number_vector(yyjson_obj_get(parameters, "value"),
                                       model->dimension, rule->values);
      if (status != SS_OK) {
        return status;
      }
      rule->value_count = model->dimension;
    } else if (rule->value_kind == SS_SOLVER_VALUE_SPATIAL_GLOBAL_TRACTION) {
      status = ss_solver_number_vector(yyjson_obj_get(parameters, "offset"),
                                       model->dimension, rule->values);
      if (status == SS_OK) {
        status = ss_solver_number_matrix(yyjson_obj_get(parameters, "gradient"),
                                         model->dimension, rule->gradient);
      }
      if (status != SS_OK) {
        return status;
      }
      rule->value_count = model->dimension;
    } else if (rule->value_kind == SS_SOLVER_VALUE_AFFINE_COORDINATE) {
      yyjson_val *offset = yyjson_obj_get(parameters, "offset");
      if (!yyjson_is_num(offset) || !isfinite(yyjson_get_num(offset))) {
        return SS_ERROR_SCHEMA;
      }
      rule->values[0] = yyjson_get_num(offset);
      rule->value_count = 1U;
      status = ss_solver_number_vector(yyjson_obj_get(parameters, "gradient"),
                                       model->dimension, rule->gradient);
      if (status != SS_OK) {
        return status;
      }
    } else if (rule->value_kind == SS_SOLVER_VALUE_NORMAL_PRESSURE ||
               rule->value_kind == SS_SOLVER_VALUE_PRESCRIBED_FLUX) {
      yyjson_val *value = yyjson_obj_get(parameters, "value");
      if (!yyjson_is_num(value) || !isfinite(yyjson_get_num(value))) {
        return SS_ERROR_SCHEMA;
      }
      rule->values[0] = yyjson_get_num(value);
      rule->value_count = 1U;
    } else if (rule->value_kind == SS_SOLVER_VALUE_HYDROSTATIC) {
      rule->reference_level = ss_solver_parameter(parameters, "reference_level", NAN);
      rule->reference_pressure =
          ss_solver_parameter(parameters, "reference_pressure", NAN);
      rule->water_unit_weight =
          ss_solver_parameter(parameters, "water_unit_weight",
                              ss_solver_parameter(parameters, "unit_weight", NAN));
      if (!isfinite(rule->reference_level) || !isfinite(rule->reference_pressure) ||
          !isfinite(rule->water_unit_weight) || rule->water_unit_weight <= 0.0) {
        return SS_ERROR_SCHEMA;
      }
    } else if (rule->value_kind == SS_SOLVER_VALUE_PIECEWISE_LEVEL) {
      yyjson_val *points = yyjson_obj_get(parameters, "points");
      yyjson_val *coordinates = yyjson_obj_get(parameters, "coordinates");
      yyjson_val *levels = yyjson_obj_get(parameters, "values");
      if (yyjson_is_arr(points)) {
        yyjson_arr_foreach(points, item_index, maximum, item) {
          yyjson_val *coordinate;
          yyjson_val *level;
          if (item_index >= SS_HYDRAULIC_HEAD_POINT_CAPACITY || !yyjson_is_obj(item)) {
            return SS_ERROR_LIMIT;
          }
          coordinate = yyjson_obj_get(item, "coordinate");
          level = yyjson_obj_get(item, "level");
          if (!yyjson_is_num(coordinate) || !yyjson_is_num(level)) {
            return SS_ERROR_SCHEMA;
          }
          rule->piecewise_coordinates[item_index] = yyjson_get_num(coordinate);
          rule->piecewise_levels[item_index] = yyjson_get_num(level);
          rule->piecewise_count = item_index + 1U;
        }
      } else if (yyjson_is_arr(coordinates) && yyjson_is_arr(levels) &&
                 yyjson_arr_size(coordinates) == yyjson_arr_size(levels)) {
        const size_t point_count = yyjson_arr_size(coordinates);
        if (point_count > SS_HYDRAULIC_HEAD_POINT_CAPACITY) {
          return SS_ERROR_LIMIT;
        }
        for (item_index = 0U; item_index < point_count; ++item_index) {
          yyjson_val *coordinate = yyjson_arr_get(coordinates, item_index);
          yyjson_val *level = yyjson_arr_get(levels, item_index);
          if (!yyjson_is_num(coordinate) || !yyjson_is_num(level)) {
            return SS_ERROR_SCHEMA;
          }
          rule->piecewise_coordinates[item_index] = yyjson_get_num(coordinate);
          rule->piecewise_levels[item_index] = yyjson_get_num(level);
        }
        rule->piecewise_count = point_count;
      }
      rule->reference_pressure =
          ss_solver_parameter(parameters, "reference_pressure", 0.0);
      rule->water_unit_weight =
          ss_solver_parameter(parameters, "water_unit_weight",
                              ss_solver_parameter(parameters, "unit_weight", 9.81));
      if (rule->piecewise_count < 2U || !isfinite(rule->reference_pressure) ||
          !isfinite(rule->water_unit_weight) || rule->water_unit_weight <= 0.0) {
        return SS_ERROR_SCHEMA;
      }
      for (item_index = 0U; item_index < rule->piecewise_count; ++item_index) {
        if (!isfinite(rule->piecewise_coordinates[item_index]) ||
            !isfinite(rule->piecewise_levels[item_index]) ||
            (item_index != 0U && rule->piecewise_coordinates[item_index - 1U] >=
                                     rule->piecewise_coordinates[item_index])) {
          return SS_ERROR_SCHEMA;
        }
      }
    }
    ++index;
  }
  return index == model->boundary_count ? SS_OK : SS_ERROR_SCHEMA;
}

SsStatus ss_solver_parse_problem(const SsCompiledManifest *manifest, int rank,
                                 int ranks, SsSolverProblemModel *model) {
  yyjson_doc *document = NULL;
  yyjson_val *root;
  yyjson_val *physics;
  yyjson_val *spaces;
  yyjson_val *operators;
  yyjson_val *mesh;
  yyjson_val *outputs;
  const char *analysis;
  SsInputFileIdentity after = {0};
  SsStatus status = SS_OK;
  if (manifest == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  document =
      yyjson_read((const char *)manifest->data, manifest->size, YYJSON_READ_NOFLAG);
  if (document == NULL) {
    return SS_ERROR_JSON;
  }
  root = yyjson_doc_get_root(document);
  physics = ss_solver_object(root, "physics");
  spaces = ss_solver_object(root, "spaces");
  operators = ss_solver_object(root, "operators");
  mesh = ss_solver_object(root, "mesh");
  outputs = ss_solver_object(root, "outputs");
  if (!yyjson_is_obj(root) || physics == NULL || spaces == NULL || operators == NULL ||
      mesh == NULL || outputs == NULL) {
    status = SS_ERROR_SCHEMA;
    goto cleanup;
  }
  model->dimension = (uint32_t)ss_solver_uint(mesh, "dimension", 0U);
  model->coordinate_order =
      (uint32_t)ss_solver_uint(ss_solver_object(spaces, "coordinates"), "order", 0U);
  model->mechanics_enabled = ss_solver_boolean(physics, "mechanics", 0);
  model->hydraulic_enabled = ss_solver_boolean(physics, "seepage", 0);
  {
    const char *coupling = ss_solver_string(physics, "coupling");
    model->coupling_enabled = coupling != NULL && strcmp(coupling, "none") != 0;
    if (coupling != NULL && strcmp(coupling, "pressure_gradient_and_weight") == 0) {
      model->coupling_mode = SS_COUPLING_PRESSURE_GRADIENT_AND_WEIGHT;
    } else if (coupling != NULL && strcmp(coupling, "weight_only") == 0) {
      model->coupling_mode = SS_COUPLING_WEIGHT_ONLY;
    } else if (coupling == NULL || strcmp(coupling, "none") != 0) {
      status = SS_ERROR_SCHEMA;
      goto cleanup;
    }
  }
  model->mechanics_order =
      model->mechanics_enabled != 0
          ? (uint32_t)ss_solver_uint(ss_solver_object(spaces, "displacement"), "order",
                                     0U)
          : 0U;
  model->hydraulic_order =
      model->hydraulic_enabled != 0
          ? (uint32_t)ss_solver_uint(ss_solver_object(spaces, "pore_pressure") != NULL
                                         ? ss_solver_object(spaces, "pore_pressure")
                                         : ss_solver_object(spaces, "pressure"),
                                     "order", 0U)
          : 0U;
  {
    /* Optional problem-level volume rule (PHY-05); absent means the default. */
    yyjson_val *rule = yyjson_obj_get(operators, "volume_quadrature");
    model->volume_quadrature_mode = SS_FE_QUADRATURE_MODE_DEFAULT;
    if (rule != NULL && yyjson_equals_str(rule, "paired_higher")) {
      model->volume_quadrature_mode = SS_FE_QUADRATURE_MODE_PAIRED_HIGHER;
    } else if (rule != NULL && !yyjson_equals_str(rule, "default")) {
      status = SS_ERROR_SCHEMA;
      goto cleanup;
    }
  }
  analysis = ss_solver_string(operators, "analysis");
  if (analysis == NULL ||
      !ss_copy_text(model->analysis_identity, sizeof(model->analysis_identity),
                    analysis) ||
      !ss_copy_text(model->mesh_id, sizeof(model->mesh_id),
                    ss_solver_string(mesh, "logical_digest"))) {
    status = SS_ERROR_SCHEMA;
    goto cleanup;
  }
  if (strncmp(analysis, "limit_load.", 11U) == 0) {
    model->analysis_method = SS_ANALYSIS_LIMIT_LOAD;
    (void)ss_copy_text(model->result_analysis_method,
                       sizeof(model->result_analysis_method), "limit_load");
  } else if (strncmp(analysis, "ssr.direct", 10U) == 0) {
    model->analysis_method = SS_ANALYSIS_SSR_DIRECT;
    (void)ss_copy_text(model->result_analysis_method,
                       sizeof(model->result_analysis_method), "ssr_direct");
  } else if (strncmp(analysis, "ssr.indirect", 12U) == 0) {
    model->analysis_method = SS_ANALYSIS_SSR_INDIRECT;
    (void)ss_copy_text(model->result_analysis_method,
                       sizeof(model->result_analysis_method), "ssr_indirect");
  } else if (strcmp(analysis, "seepage.steady_unconfined") == 0 &&
             model->mechanics_enabled == 0 && model->hydraulic_enabled != 0) {
    /*
     * Pure seepage does not enter the continuation controller.  Keep a valid
     * analysis enum for shared result metadata and use the dedicated method
     * identity at publication time.
     */
    model->analysis_method = SS_ANALYSIS_LIMIT_LOAD;
    (void)ss_copy_text(model->result_analysis_method,
                       sizeof(model->result_analysis_method), "steady_seepage");
  } else {
    status = SS_ERROR_SCHEMA;
    goto cleanup;
  }
  if ((model->dimension != 2U && model->dimension != 3U) ||
      (model->coordinate_order != 1U && model->coordinate_order != 2U &&
       model->coordinate_order != 4U) ||
      (model->mechanics_enabled != 0 && model->mechanics_order != 1U &&
       model->mechanics_order != 2U && model->mechanics_order != 4U &&
       model->mechanics_order != 8U) ||
      (model->hydraulic_enabled != 0 && model->hydraulic_order != 1U &&
       model->hydraulic_order != 2U && model->hydraulic_order != 4U) ||
      (model->coupling_enabled != 0 &&
       (model->mechanics_enabled == 0 || model->hydraulic_enabled == 0)) ||
      (model->coupling_enabled == 0 && model->coupling_mode != 0)) {
    status = SS_ERROR_SCHEMA;
    goto cleanup;
  }
  status = ss_solver_parse_prescribed_saturation(root, model);
  if (status != SS_OK)
    goto cleanup;
  status = ss_solver_parse_output_quantities(yyjson_obj_get(outputs, "required"),
                                             &model->required_output_mask);
  if (status == SS_OK) {
    status = ss_solver_parse_materials(root, manifest, rank, ranks, model);
  }
  if (status == SS_OK) {
    status = ss_solver_parse_boundaries(root, manifest, rank, ranks, model);
  }

  if (status == SS_OK && (ss_io_input_identity(manifest->descriptor, &after) != SS_OK ||
                          memcmp(&after, &manifest->identity, sizeof(after)) != 0)) {
    status = SS_ERROR_INTEGRITY;
  }

cleanup:
  yyjson_doc_free(document);
  return status;
}
