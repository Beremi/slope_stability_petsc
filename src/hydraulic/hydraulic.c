/* Steady unconfined seepage: constitutive ramp, point kernels, head and
 * flux evaluation and the hydraulic operator model.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/hydraulic/hydraulic.h"
#include "slope_internal/core/sort.h"
#include "slope_internal/core/util.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  SsEntityId stable_cell_id;
  double epsilon;
  size_t conductivity_count;
  double *conductivity;
} SsHydraulicOwnedCellCoefficients;

struct SsHydraulicOperatorModel {
  uint32_t dimension;
  uint32_t upward_axis;
  double gamma_water;
  SsHydraulicGravityMode gravity_mode;
  size_t cell_count;
  SsHydraulicOwnedCellCoefficients *cells;
  SsHydraulicOperatorModelSummary summary;
};

static int ss_hydraulic_valid_identity(const char *identity) {
  const size_t length = ss_bounded_length(identity, SS_HYDRAULIC_ID_CAPACITY);
  return length != 0U && length < SS_HYDRAULIC_ID_CAPACITY;
}

static int ss_hydraulic_cell_compare(const void *left, const void *right) {
  const SsHydraulicOwnedCellCoefficients *first =
      (const SsHydraulicOwnedCellCoefficients *)left;
  const SsHydraulicOwnedCellCoefficients *second =
      (const SsHydraulicOwnedCellCoefficients *)right;
  return first->stable_cell_id < second->stable_cell_id
             ? -1
             : (first->stable_cell_id > second->stable_cell_id ? 1 : 0);
}

static const SsHydraulicOwnedCellCoefficients *
ss_hydraulic_operator_find_cell(const SsHydraulicOperatorModel *model,
                                SsEntityId stable_cell_id) {
  size_t left = 0U;
  size_t right = model->cell_count;
  while (left < right) {
    const size_t middle = left + (right - left) / 2U;
    if (model->cells[middle].stable_cell_id < stable_cell_id) {
      left = middle + 1U;
    } else {
      right = middle;
    }
  }
  return left < model->cell_count && model->cells[left].stable_cell_id == stable_cell_id
             ? &model->cells[left]
             : NULL;
}

static void ss_hydraulic_operator_model_release(SsHydraulicOperatorModel *model) {
  size_t cell;
  if (model == NULL) {
    return;
  }
  for (cell = 0U; cell < model->cell_count; ++cell) {
    free(model->cells[cell].conductivity);
  }
  free(model->cells);
  free(model);
}

SsStatus ss_hydraulic_ramp(double pressure, double epsilon, SsHydraulicRamp *ramp) {
  if (ramp == NULL || !ss_is_finite(pressure) || !ss_is_finite(epsilon) ||
      epsilon <= 0.0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(ramp, 0, sizeof(*ramp));
  ramp->epsilon = epsilon;
  if (pressure <= 0.0) {
    ramp->region = SS_HYDRAULIC_RAMP_DRY;
  } else if (pressure < epsilon) {
    ramp->region = SS_HYDRAULIC_RAMP_TRANSITION;
    ramp->value = pressure / epsilon;
    ramp->derivative = 1.0 / epsilon;
  } else {
    ramp->region = SS_HYDRAULIC_RAMP_SATURATED;
    ramp->value = 1.0;
  }
  return SS_OK;
}

SsStatus ss_hydraulic_initialization_select(double boundary_residual_norm,
                                            double saturated_residual_norm,
                                            int saturated_candidate_valid,
                                            SsHydraulicInitializationRecord *record) {
  if (record == NULL || !ss_is_finite(boundary_residual_norm) ||
      boundary_residual_norm < 0.0 ||
      (saturated_candidate_valid != 0 &&
       (!ss_is_finite(saturated_residual_norm) || saturated_residual_norm < 0.0))) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(record, 0, sizeof(*record));
  record->boundary_residual_norm = boundary_residual_norm;
  record->saturated_residual_norm =
      saturated_candidate_valid != 0 ? saturated_residual_norm : 0.0;
  record->saturated_candidate_valid = saturated_candidate_valid != 0;
  /*
   * The frozen MATLAB and petsc4py algorithms always start Newton from the
   * dimension-specific linear seepage solve.  Its residual need not be lower
   * than the boundary envelope's residual: it is valuable because it lies in
   * the reference Newton basin.  Keep the envelope only as a recovery path
   * when the linear initializer cannot be formed or Newton rejects it.
   */
  record->choice = saturated_candidate_valid != 0
                       ? SS_HYDRAULIC_INITIAL_SATURATED_LINEAR
                       : SS_HYDRAULIC_INITIAL_BOUNDARY_ENVELOPE;
  record->primary_choice = record->choice;
  return SS_OK;
}

SsStatus ss_hydraulic_initialization_fallback(SsHydraulicInitializationRecord *record) {
  if (record == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (record->saturated_candidate_valid == 0 || record->fallback_attempted != 0 ||
      (record->choice != SS_HYDRAULIC_INITIAL_BOUNDARY_ENVELOPE &&
       record->choice != SS_HYDRAULIC_INITIAL_SATURATED_LINEAR) ||
      record->primary_choice != record->choice) {
    return SS_ERROR_INVALID_STATE;
  }
  record->choice = record->choice == SS_HYDRAULIC_INITIAL_BOUNDARY_ENVELOPE
                       ? SS_HYDRAULIC_INITIAL_SATURATED_LINEAR
                       : SS_HYDRAULIC_INITIAL_BOUNDARY_ENVELOPE;
  record->fallback_attempted = 1;
  return SS_OK;
}

SsStatus ss_hydraulic_cell_epsilon(uint32_t dimension, const double *vertex_coordinates,
                                   double gamma_water, double *minimum_edge,
                                   double *epsilon) {
  const size_t vertex_count = (size_t)dimension + 1U;
  double minimum = INFINITY;
  size_t first;
  if ((dimension != 2U && dimension != 3U) || vertex_coordinates == NULL ||
      minimum_edge == NULL || epsilon == NULL || !ss_is_finite(gamma_water) ||
      gamma_water <= 0.0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (first = 0U; first < vertex_count; ++first) {
    size_t second;
    for (second = first + 1U; second < vertex_count; ++second) {
      double squared = 0.0;
      uint32_t coordinate;
      for (coordinate = 0U; coordinate < dimension; ++coordinate) {
        const double left = vertex_coordinates[first * dimension + (size_t)coordinate];
        const double right =
            vertex_coordinates[second * dimension + (size_t)coordinate];
        const double difference = left - right;
        if (!ss_is_finite(left) || !ss_is_finite(right)) {
          return SS_ERROR_INVALID_ARGUMENT;
        }
        squared += difference * difference;
      }
      if (!ss_is_finite(squared) || squared <= 0.0) {
        return SS_ERROR_INTEGRITY;
      }
      if (sqrt(squared) < minimum) {
        minimum = sqrt(squared);
      }
    }
  }
  if (!ss_is_finite(minimum) || minimum <= 0.0 ||
      !ss_is_finite(gamma_water * minimum / 2.0)) {
    return SS_ERROR_INTEGRITY;
  }
  *minimum_edge = minimum;
  *epsilon = gamma_water * minimum / 2.0;
  return SS_OK;
}

SsStatus ss_hydraulic_point_evaluate(const SsHydraulicPointModel *model,
                                     double pressure, const double *gradient,
                                     SsHydraulicPointResult *result) {
  uint32_t coordinate;
  if (model == NULL || gradient == NULL || result == NULL ||
      (model->dimension != 2U && model->dimension != 3U) ||
      model->upward_axis >= model->dimension || !ss_is_finite(model->gamma_water) ||
      model->gamma_water <= 0.0 || !ss_is_finite(model->conductivity) ||
      model->conductivity <= 0.0 || !ss_is_finite(model->epsilon) ||
      model->epsilon <= 0.0 || !ss_is_finite(pressure)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  result->pressure = pressure;
  if (ss_hydraulic_ramp(pressure, model->epsilon, &result->ramp) != SS_OK) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (coordinate = 0U; coordinate < model->dimension; ++coordinate) {
    const double gravity = coordinate == model->upward_axis
                               ? model->gamma_water * result->ramp.value
                               : 0.0;
    if (!ss_is_finite(gradient[coordinate])) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
    result->gradient[coordinate] = gradient[coordinate];
    result->pressure_flux[coordinate] =
        model->conductivity * (gradient[coordinate] + gravity);
    if (!ss_is_finite(result->pressure_flux[coordinate])) {
      return SS_ERROR_INTEGRITY;
    }
  }
  return SS_OK;
}

SsStatus ss_hydraulic_point_kernel(const void *context, const SsScalarPointInput *input,
                                   SsScalarPointOutput *output) {
  const SsHydraulicPointModel *model = (const SsHydraulicPointModel *)context;
  SsHydraulicPointResult result;
  SsStatus status;
  uint32_t coordinate;
  if (model == NULL || input == NULL || output == NULL ||
      input->dimension != model->dimension) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status = ss_hydraulic_point_evaluate(model, input->value, input->gradient, &result);
  if (status != SS_OK) {
    return status;
  }
  memset(output, 0, sizeof(*output));
  for (coordinate = 0U; coordinate < model->dimension; ++coordinate) {
    output->flux[coordinate] = result.pressure_flux[coordinate];
    output->flux_derivative_gradient[coordinate * model->dimension + coordinate] =
        model->conductivity;
  }
  output->flux_derivative_value[model->upward_axis] =
      model->conductivity * model->gamma_water * result.ramp.derivative;
  return SS_OK;
}

SsStatus ss_hydraulic_head_evaluate(const SsHydraulicHeadModel *model,
                                    const double *physical, double *pressure) {
  double level = 0.0;
  uint32_t coordinate;
  if (model == NULL || physical == NULL || pressure == NULL ||
      (model->dimension != 2U && model->dimension != 3U) ||
      model->upward_axis >= model->dimension || !ss_is_finite(model->gamma_water) ||
      model->gamma_water <= 0.0 || !ss_hydraulic_valid_identity(model->identity)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (coordinate = 0U; coordinate < model->dimension; ++coordinate) {
    if (!ss_is_finite(physical[coordinate])) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
  }
  switch (model->kind) {
  case SS_HYDRAULIC_HEAD_DRY:
    *pressure = 0.0;
    return SS_OK;
  case SS_HYDRAULIC_HEAD_CONSTANT_LEVEL:
    level = model->level;
    break;
  case SS_HYDRAULIC_HEAD_COMPILED_AFFINE_LEVEL:
    level = model->affine_level[0];
    for (coordinate = 0U; coordinate < model->dimension; ++coordinate) {
      level += model->affine_level[(size_t)coordinate + 1U] * physical[coordinate];
    }
    break;
  case SS_HYDRAULIC_HEAD_PIECEWISE_LINEAR_LEVEL: {
    size_t point;
    const double abscissa = model->piecewise_axis < model->dimension
                                ? physical[model->piecewise_axis]
                                : NAN;
    if (model->piecewise_axis >= model->dimension ||
        model->piecewise_point_count < 2U ||
        model->piecewise_point_count > SS_HYDRAULIC_HEAD_POINT_CAPACITY ||
        !ss_is_finite(abscissa)) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
    for (point = 0U; point < model->piecewise_point_count; ++point) {
      if (!ss_is_finite(model->piecewise_coordinates[point]) ||
          !ss_is_finite(model->piecewise_levels[point]) ||
          (point != 0U && model->piecewise_coordinates[point] <=
                              model->piecewise_coordinates[point - 1U])) {
        return SS_ERROR_INVALID_ARGUMENT;
      }
    }
    if (abscissa <= model->piecewise_coordinates[0]) {
      level = model->piecewise_levels[0];
    } else if (abscissa >=
               model->piecewise_coordinates[model->piecewise_point_count - 1U]) {
      level = model->piecewise_levels[model->piecewise_point_count - 1U];
    } else {
      for (point = 1U; point < model->piecewise_point_count; ++point) {
        if (abscissa <= model->piecewise_coordinates[point]) {
          const double left_coordinate = model->piecewise_coordinates[point - 1U];
          const double right_coordinate = model->piecewise_coordinates[point];
          const double fraction =
              (abscissa - left_coordinate) / (right_coordinate - left_coordinate);
          level = model->piecewise_levels[point - 1U] +
                  fraction * (model->piecewise_levels[point] -
                              model->piecewise_levels[point - 1U]);
          break;
        }
      }
    }
    break;
  }
  case SS_HYDRAULIC_HEAD_NODAL_VALUE:
    if (!ss_is_finite(model->nodal_value)) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
    *pressure = model->nodal_value;
    return SS_OK;
  default:
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (!ss_is_finite(level)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *pressure = model->gamma_water * fmax(level - physical[model->upward_axis], 0.0);
  return ss_is_finite(*pressure) ? SS_OK : SS_ERROR_INTEGRITY;
}

SsStatus ss_hydraulic_flux_evaluate(const SsHydraulicFluxModel *model,
                                    const double *physical, double *normal_flux) {
  return ss_hydraulic_flux_evaluate_normal(model, physical, NULL, normal_flux);
}

SsStatus ss_hydraulic_flux_evaluate_normal(const SsHydraulicFluxModel *model,
                                           const double *physical,
                                           const double *outward_normal,
                                           double *normal_flux) {
  uint32_t coordinate;
  double value;
  if (model == NULL || physical == NULL || normal_flux == NULL ||
      (model->dimension != 2U && model->dimension != 3U) ||
      !ss_hydraulic_valid_identity(model->identity)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (coordinate = 0U; coordinate < model->dimension; ++coordinate) {
    if (!ss_is_finite(physical[coordinate])) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
  }
  switch (model->kind) {
  case SS_HYDRAULIC_FLUX_ZERO:
    value = 0.0;
    break;
  case SS_HYDRAULIC_FLUX_CONSTANT:
    value = model->value;
    break;
  case SS_HYDRAULIC_FLUX_COMPILED_AFFINE:
    value = model->affine[0];
    for (coordinate = 0U; coordinate < model->dimension; ++coordinate) {
      value += model->affine[(size_t)coordinate + 1U] * physical[coordinate];
    }
    break;
  case SS_HYDRAULIC_FLUX_COMPILED_VECTOR_NORMAL:
    if (outward_normal == NULL) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
    value = 0.0;
    for (coordinate = 0U; coordinate < model->dimension; ++coordinate) {
      if (!ss_is_finite(model->vector[coordinate]) ||
          !ss_is_finite(outward_normal[coordinate])) {
        return SS_ERROR_INVALID_ARGUMENT;
      }
      value += model->vector[coordinate] * outward_normal[coordinate];
    }
    break;
  default:
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (!ss_is_finite(value)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *normal_flux = value;
  return SS_OK;
}

SsStatus ss_hydraulic_flux_point_kernel(const void *context,
                                        const SsNaturalPointInput *input,
                                        SsNaturalPointOutput *output) {
  const SsHydraulicFluxModel *model = (const SsHydraulicFluxModel *)context;
  double flux = 0.0;
  if (model == NULL || input == NULL || output == NULL ||
      input->dimension != model->dimension || input->components != 1U ||
      ss_hydraulic_flux_evaluate_normal(model, input->physical, input->outward_normal,
                                        &flux) != SS_OK) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(output, 0, sizeof(*output));
  output->load[0] = flux;
  output->signed_flux = flux;
  return SS_OK;
}

SsStatus ss_hydraulic_operator_model_create(const SsHydraulicOperatorModelSpec *spec,
                                            SsHydraulicOperatorModel **model) {
  SsHydraulicOperatorModel *created = NULL;
  size_t coefficient_count = 0U;
  size_t cell;
  if (model != NULL) {
    *model = NULL;
  }
  if (spec == NULL || model == NULL ||
      (spec->dimension != 2U && spec->dimension != 3U) ||
      spec->upward_axis >= spec->dimension || !ss_is_finite(spec->gamma_water) ||
      spec->gamma_water <= 0.0 || (spec->cell_count != 0U && spec->cells == NULL) ||
      !ss_hydraulic_valid_identity(spec->model_identity) ||
      !ss_hydraulic_valid_identity(spec->pressure_unit) ||
      !ss_hydraulic_valid_identity(spec->coordinate_unit) ||
      !ss_hydraulic_valid_identity(spec->conductivity_unit)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (spec->cell_count > SIZE_MAX / sizeof(SsHydraulicOwnedCellCoefficients)) {
    return SS_ERROR_OVERFLOW;
  }
  created = calloc(1U, sizeof(*created));
  if (created == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  created->cells =
      calloc(spec->cell_count == 0U ? 1U : spec->cell_count, sizeof(*created->cells));
  if (created->cells == NULL) {
    ss_hydraulic_operator_model_release(created);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  created->dimension = spec->dimension;
  created->upward_axis = spec->upward_axis;
  created->gamma_water = spec->gamma_water;
  created->gravity_mode = SS_HYDRAULIC_GRAVITY_NONLINEAR;
  created->cell_count = spec->cell_count;
  created->summary.dimension = spec->dimension;
  created->summary.upward_axis = spec->upward_axis;
  created->summary.gamma_water = spec->gamma_water;
  created->summary.cell_count = spec->cell_count;
  created->summary.conductivity_minimum = INFINITY;
  created->summary.conductivity_maximum = -INFINITY;
  created->summary.epsilon_minimum = INFINITY;
  created->summary.epsilon_maximum = -INFINITY;
  ss_copy_text(created->summary.model_identity, sizeof(created->summary.model_identity),
               spec->model_identity);
  ss_copy_text(created->summary.pressure_unit, sizeof(created->summary.pressure_unit),
               spec->pressure_unit);
  ss_copy_text(created->summary.coordinate_unit,
               sizeof(created->summary.coordinate_unit), spec->coordinate_unit);
  ss_copy_text(created->summary.conductivity_unit,
               sizeof(created->summary.conductivity_unit), spec->conductivity_unit);
  for (cell = 0U; cell < spec->cell_count; ++cell) {
    const SsHydraulicCellCoefficients *source = &spec->cells[cell];
    SsHydraulicOwnedCellCoefficients *target = &created->cells[cell];
    size_t coefficient;
    if (source->stable_cell_id == 0U || !ss_is_finite(source->epsilon) ||
        source->epsilon <= 0.0 || source->conductivity_count == 0U ||
        source->conductivity == NULL) {
      ss_hydraulic_operator_model_release(created);
      return SS_ERROR_INVALID_ARGUMENT;
    }
    if (source->conductivity_count > SIZE_MAX / sizeof(double) ||
        source->conductivity_count > SIZE_MAX - coefficient_count) {
      ss_hydraulic_operator_model_release(created);
      return SS_ERROR_OVERFLOW;
    }
    target->stable_cell_id = source->stable_cell_id;
    target->epsilon = source->epsilon;
    target->conductivity_count = source->conductivity_count;
    target->conductivity =
        malloc(source->conductivity_count * sizeof(*target->conductivity));
    if (target->conductivity == NULL) {
      ss_hydraulic_operator_model_release(created);
      return SS_ERROR_OUT_OF_MEMORY;
    }
    for (coefficient = 0U; coefficient < source->conductivity_count; ++coefficient) {
      const double value = source->conductivity[coefficient];
      if (!ss_is_finite(value) || value <= 0.0) {
        ss_hydraulic_operator_model_release(created);
        return SS_ERROR_INVALID_ARGUMENT;
      }
      target->conductivity[coefficient] = value;
      created->summary.conductivity_minimum =
          fmin(created->summary.conductivity_minimum, value);
      created->summary.conductivity_maximum =
          fmax(created->summary.conductivity_maximum, value);
    }
    created->summary.epsilon_minimum =
        fmin(created->summary.epsilon_minimum, source->epsilon);
    created->summary.epsilon_maximum =
        fmax(created->summary.epsilon_maximum, source->epsilon);
    coefficient_count += source->conductivity_count;
  }
  if (spec->cell_count == 0U) {
    created->summary.conductivity_minimum = 0.0;
    created->summary.conductivity_maximum = 0.0;
    created->summary.epsilon_minimum = 0.0;
    created->summary.epsilon_maximum = 0.0;
  }
  ss_sort(created->cells, created->cell_count, sizeof(*created->cells),
          ss_hydraulic_cell_compare);
  for (cell = 1U; cell < created->cell_count; ++cell) {
    if (created->cells[cell - 1U].stable_cell_id ==
        created->cells[cell].stable_cell_id) {
      ss_hydraulic_operator_model_release(created);
      return SS_ERROR_DUPLICATE;
    }
  }
  created->summary.quadrature_coefficient_count = coefficient_count;
  *model = created;
  return SS_OK;
}

SsStatus ss_hydraulic_operator_model_set_gravity_mode(SsHydraulicOperatorModel *model,
                                                      SsHydraulicGravityMode mode) {
  if (model == NULL || (mode != SS_HYDRAULIC_GRAVITY_NONLINEAR &&
                        mode != SS_HYDRAULIC_GRAVITY_DISABLED &&
                        mode != SS_HYDRAULIC_GRAVITY_SATURATED)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  model->gravity_mode = mode;
  return SS_OK;
}

SsStatus ss_hydraulic_operator_model_summary(const SsHydraulicOperatorModel *model,
                                             SsHydraulicOperatorModelSummary *summary) {
  if (model == NULL || summary == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *summary = model->summary;
  return SS_OK;
}

SsStatus ss_hydraulic_operator_point_kernel(const void *context,
                                            const SsScalarPointInput *input,
                                            SsScalarPointOutput *output) {
  const SsHydraulicOperatorModel *model = (const SsHydraulicOperatorModel *)context;
  const SsHydraulicOwnedCellCoefficients *cell;
  SsHydraulicPointModel point_model;
  size_t coefficient_index;
  if (model == NULL || input == NULL || output == NULL ||
      input->dimension != model->dimension || input->stable_cell_id == 0U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  cell = ss_hydraulic_operator_find_cell(model, input->stable_cell_id);
  if (cell == NULL) {
    return SS_ERROR_NOT_FOUND;
  }
  coefficient_index =
      cell->conductivity_count == 1U ? 0U : (size_t)input->quadrature_index;
  if (coefficient_index >= cell->conductivity_count) {
    return SS_ERROR_NOT_FOUND;
  }
  point_model = (SsHydraulicPointModel){
      model->dimension,   model->upward_axis,
      model->gamma_water, cell->conductivity[coefficient_index],
      cell->epsilon,
  };
  if (model->gravity_mode == SS_HYDRAULIC_GRAVITY_NONLINEAR) {
    return ss_hydraulic_point_kernel(&point_model, input, output);
  }
  if (!ss_is_finite(input->value)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(output, 0, sizeof(*output));
  for (uint32_t coordinate = 0U; coordinate < model->dimension; ++coordinate) {
    const double gravity = coordinate == model->upward_axis &&
                                   model->gravity_mode == SS_HYDRAULIC_GRAVITY_SATURATED
                               ? model->gamma_water
                               : 0.0;
    if (!ss_is_finite(input->gradient[coordinate])) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
    output->flux[coordinate] =
        point_model.conductivity * (input->gradient[coordinate] + gravity);
    output->flux_derivative_gradient[coordinate * model->dimension + coordinate] =
        point_model.conductivity;
    if (!ss_is_finite(output->flux[coordinate])) {
      return SS_ERROR_INTEGRITY;
    }
  }
  return SS_OK;
}

SsStatus ss_hydraulic_operator_model_destroy(SsHydraulicOperatorModel **model) {
  if (model == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  ss_hydraulic_operator_model_release(*model);
  *model = NULL;
  return SS_OK;
}

SsStatus ss_hydraulic_balance(MPI_Comm communicator,
                              const SsHydraulicBalanceLocal *local,
                              double physical_scale, double normalized_tolerance,
                              double free_residual_tolerance,
                              SsHydraulicBalance *balance) {
  double send[5];
  double *received = NULL;
  int initialized = 0;
  int rank = 0;
  int size = 0;
  int local_valid;
  int global_valid = 0;
  if (local == NULL || balance == NULL || !ss_is_finite(physical_scale) ||
      physical_scale <= 0.0 || !ss_is_finite(normalized_tolerance) ||
      normalized_tolerance < 0.0 || !ss_is_finite(free_residual_tolerance) ||
      free_residual_tolerance < 0.0 || MPI_Initialized(&initialized) != MPI_SUCCESS ||
      initialized == 0 || communicator == MPI_COMM_NULL ||
      MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(communicator, &size) != MPI_SUCCESS || size <= 0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  send[0] = local->domain_source;
  send[1] = local->prescribed_boundary_flux;
  send[2] = local->natural_boundary_flux;
  send[3] = local->dirichlet_reaction_flux;
  send[4] = local->free_residual_l2_squared;
  local_valid = ss_is_finite(send[0]) && ss_is_finite(send[1]) &&
                ss_is_finite(send[2]) && ss_is_finite(send[3]) &&
                ss_is_finite(send[4]) && send[4] >= 0.0;
  if (MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, communicator) !=
      MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (global_valid == 0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (rank == 0) {
    received = calloc((size_t)size * 5U, sizeof(*received));
    if (received == NULL) {
      local_valid = 0;
    }
  }
  if (MPI_Bcast(&local_valid, 1, MPI_INT, 0, communicator) != MPI_SUCCESS) {
    free(received);
    return SS_ERROR_INTERNAL;
  }
  if (local_valid == 0) {
    free(received);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  if (MPI_Gather(send, 5, MPI_DOUBLE, received, 5, MPI_DOUBLE, 0, communicator) !=
      MPI_SUCCESS) {
    free(received);
    return SS_ERROR_INTERNAL;
  }
  memset(balance, 0, sizeof(*balance));
  if (rank == 0) {
    long double sums[5] = {0.0L, 0.0L, 0.0L, 0.0L, 0.0L};
    int source_rank;
    for (source_rank = 0; source_rank < size; ++source_rank) {
      size_t field;
      for (field = 0U; field < 5U; ++field) {
        sums[field] += (long double)received[(size_t)source_rank * 5U + field];
      }
    }
    balance->domain_source = (double)sums[0];
    balance->prescribed_boundary_flux = (double)sums[1];
    balance->natural_boundary_flux = (double)sums[2];
    balance->dirichlet_reaction_flux = (double)sums[3];
    balance->signed_imbalance =
        balance->domain_source + balance->prescribed_boundary_flux +
        balance->natural_boundary_flux + balance->dirichlet_reaction_flux;
    balance->physical_scale = physical_scale;
    balance->normalization =
        fabs(balance->domain_source) + fabs(balance->prescribed_boundary_flux) +
        fabs(balance->natural_boundary_flux) + fabs(balance->dirichlet_reaction_flux);
    balance->normalization = fmax(balance->normalization, physical_scale);
    balance->normalized_imbalance =
        fabs(balance->signed_imbalance) / balance->normalization;
    balance->free_residual_l2 = sqrt((double)sums[4]);
    balance->accepted = balance->normalized_imbalance <= normalized_tolerance &&
                        balance->free_residual_l2 <= free_residual_tolerance;
  }
  free(received);
  if (MPI_Bcast(balance, (int)sizeof(*balance), MPI_BYTE, 0, communicator) !=
      MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  return SS_OK;
}

SsStatus ss_hydraulic_operator_traits(SsHydraulicOperatorTraits *traits) {
  if (traits == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(traits, 0, sizeof(*traits));
  traits->scalar = 1;
  traits->real = 1;
  traits->generally_nonsymmetric = 1;
  traits->same_free_row_column_space = 1;
  traits->fixed_structure = 1;
  traits->pressure_dependent_values = 1;
  ss_copy_text(traits->identity, sizeof(traits->identity),
               "hydraulic-scalar-general-v1");
  return SS_OK;
}
