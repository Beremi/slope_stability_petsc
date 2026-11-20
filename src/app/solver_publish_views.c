/* Shared result views, quantity lists and hydraulic outputs of the publishers.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "solver_output_internal.h"

#include <string.h>

static const char *const ss_solver_vector_2d[] = {"x", "y"};
static const char *const ss_solver_vector_3d[] = {"x", "y", "z"};

SsResultFieldView ss_solver_nodal_view(const char *quantity_id, const char *unit,
                                       const char *const *component_ids,
                                       uint32_t component_count,
                                       const SsSolverOutputField *field,
                                       SsStateId state_id) {
  SsResultFieldView view = {
      .quantity_id = quantity_id,
      .quantity_version = 1U,
      .unit = unit,
      .location = "field_dof",
      .basis = "lagrange",
      .component_ids = component_ids,
      .component_count = component_count,
      .dataset_kind = SS_RESULT_DATASET_FIELD,
  };
  ss_solver_output_field_bind(&view, field, state_id);
  return view;
}

SsResultFieldView ss_solver_quadrature_view(const char *quantity_id, const char *unit,
                                            const char *location,
                                            const char *const *component_ids,
                                            uint32_t component_count,
                                            const SsSolverOutputField *field,
                                            SsStateId state_id) {
  SsResultFieldView view = {
      .quantity_id = quantity_id,
      .quantity_version = 1U,
      .unit = unit,
      .location = location,
      .basis = "cell_quadrature",
      .component_ids = component_ids,
      .component_count = component_count,
      .dataset_kind = SS_RESULT_DATASET_QUADRATURE,
  };
  ss_solver_output_field_bind(&view, field, state_id);
  return view;
}

SsResultCategoryView
ss_solver_category_view(const char *quantity_id, const char *unit, const char *location,
                        const char *const *category_ids, uint32_t category_count,
                        const SsSolverOutputCategory *category, SsStateId state_id) {
  SsResultCategoryView view = {
      .quantity_id = quantity_id,
      .quantity_version = 1U,
      .unit = unit,
      .location = location,
      .category_ids = category_ids,
      .category_count = category_count,
  };
  ss_solver_output_category_bind(&view, category, state_id);
  return view;
}

SsStatus ss_solver_publish_declare(SsResultWriter *writer, const SsSolverGraph *graph,
                                   const SsSolverPublishedQuantity *quantities,
                                   size_t count, const char **stage) {
  size_t index;
  SsStatus status = SS_OK;
  for (index = 0U; status == SS_OK && index < count; ++index) {
    const SsSolverPublishedQuantity *entry = &quantities[index];
    if (!ss_solver_output_requested(graph, entry->quantity)) {
      continue;
    }
    *stage = entry->declare_stage;
    if (entry->field != NULL) {
      const SsResultQuantityDeclaration declaration =
          ss_solver_quantity_declaration(entry->field);
      status = ss_result_writer_declare_quantity(writer, &declaration);
    } else {
      const SsResultCategoryDeclaration declaration =
          ss_solver_category_declaration(entry->category);
      status = ss_result_writer_declare_category(writer, &declaration);
    }
  }
  return status;
}

SsStatus ss_solver_publish_write(SsResultWriter *writer, const SsSolverGraph *graph,
                                 const SsSolverPublishedQuantity *quantities,
                                 size_t count, const char **stage) {
  size_t index;
  SsStatus status = SS_OK;
  for (index = 0U; status == SS_OK && index < count; ++index) {
    const SsSolverPublishedQuantity *entry = &quantities[index];
    if (!ss_solver_output_requested(graph, entry->quantity)) {
      continue;
    }
    *stage = entry->write_stage;
    if (entry->field == NULL) {
      status = ss_result_writer_write_category(writer, entry->category);
    } else if (entry->field->dataset_kind == SS_RESULT_DATASET_FIELD) {
      status = ss_result_writer_write_field(writer, entry->field);
    } else {
      status = ss_result_writer_write_quantity(writer, entry->field);
    }
  }
  return status;
}

SsStatus ss_solver_hydraulic_outputs_pack(const SsSolverGraph *graph,
                                          const SsSolverHydraulic *hydraulic,
                                          const PetscScalar *local_values,
                                          SsSolverHydraulicOutputs *outputs,
                                          const char **stage) {
  SsStatus status;
  *stage = "pressure_pack";
  status = ss_solver_nodal_output(graph, graph->hydraulic_space, 1U, local_values,
                                  &outputs->pressure);
  if (status == SS_OK) {
    *stage = "hydraulic_quadrature_pack";
    status = ss_solver_hydraulic_quadrature_output(
        graph, hydraulic, local_values, &outputs->gradient, &outputs->conductivity,
        &outputs->ramp, &outputs->epsilon, &outputs->pressure_flux,
        &outputs->darcy_velocity, &outputs->saturation);
  }
  return status;
}

void ss_solver_hydraulic_outputs_views(const SsSolverGraph *graph, SsStateId state_id,
                                       SsSolverHydraulicOutputs *outputs) {
  static const char *const pressure_components[] = {"pressure"};
  static const char *const conductivity_components[] = {"k"};
  static const char *const ramp_components[] = {"kappa"};
  static const char *const epsilon_components[] = {"epsilon"};
  static const char *const saturation_categories[] = {
      "unsaturated",
      "saturated",
  };
  const char *const *vector =
      graph->model.dimension == 2U ? ss_solver_vector_2d : ss_solver_vector_3d;
  const uint32_t dimension = graph->model.dimension;
  const char *const quadrature = "hydraulic_quadrature";
  outputs->pressure_view = ss_solver_nodal_view(
      "pore_pressure", "kPa", pressure_components, 1U, &outputs->pressure, state_id);
  outputs->gradient_view =
      ss_solver_quadrature_view("pressure_gradient", "kPa/m", quadrature, vector,
                                dimension, &outputs->gradient, state_id);
  outputs->conductivity_view = ss_solver_quadrature_view(
      "conductivity", "m/s", quadrature, conductivity_components, 1U,
      &outputs->conductivity, state_id);
  outputs->ramp_view =
      ss_solver_quadrature_view("hydraulic_ramp", "dimensionless", quadrature,
                                ramp_components, 1U, &outputs->ramp, state_id);
  outputs->epsilon_view =
      ss_solver_quadrature_view("hydraulic_epsilon", "kPa", quadrature,
                                epsilon_components, 1U, &outputs->epsilon, state_id);
  outputs->pressure_flux_view =
      ss_solver_quadrature_view("pressure_flux", "kPa/s", quadrature, vector, dimension,
                                &outputs->pressure_flux, state_id);
  outputs->darcy_velocity_view =
      ss_solver_quadrature_view("darcy_velocity", "m/s", quadrature, vector, dimension,
                                &outputs->darcy_velocity, state_id);
  if (outputs->with_coupled_body_force != 0) {
    outputs->coupled_body_force_view = ss_solver_quadrature_view(
        "coupled_body_force", "kN/m^3", "mechanics_quadrature", vector, dimension,
        &outputs->coupled_body_force, state_id);
  }
  outputs->saturation_view = ss_solver_category_view(
      "saturation", "boolean", quadrature, saturation_categories, 2U,
      &outputs->saturation, state_id);
}

void ss_solver_hydraulic_outputs_bind(SsSolverHydraulicOutputs *outputs,
                                      SsStateId state_id) {
  ss_solver_output_field_bind(&outputs->pressure_view, &outputs->pressure, state_id);
  ss_solver_output_field_bind(&outputs->gradient_view, &outputs->gradient, state_id);
  ss_solver_output_field_bind(&outputs->conductivity_view, &outputs->conductivity,
                              state_id);
  ss_solver_output_field_bind(&outputs->ramp_view, &outputs->ramp, state_id);
  ss_solver_output_field_bind(&outputs->epsilon_view, &outputs->epsilon, state_id);
  ss_solver_output_field_bind(&outputs->pressure_flux_view, &outputs->pressure_flux,
                              state_id);
  ss_solver_output_field_bind(&outputs->darcy_velocity_view, &outputs->darcy_velocity,
                              state_id);
  if (outputs->with_coupled_body_force != 0) {
    ss_solver_output_field_bind(&outputs->coupled_body_force_view,
                                &outputs->coupled_body_force, state_id);
  }
  ss_solver_output_category_bind(&outputs->saturation_view, &outputs->saturation,
                                 state_id);
}

size_t ss_solver_hydraulic_outputs_list(const SsSolverHydraulicOutputs *outputs,
                                        SsSolverPublishedQuantity *quantities) {
  size_t count = 0U;
  quantities[count++] =
      (SsSolverPublishedQuantity){SS_SOLVER_QUANTITY_PORE_PRESSURE, "declare_pressure",
                                  "write_pressure", &outputs->pressure_view, NULL};
  quantities[count++] = (SsSolverPublishedQuantity){
      SS_SOLVER_QUANTITY_PRESSURE_GRADIENT, "declare_pressure_gradient",
      "write_pressure_gradient", &outputs->gradient_view, NULL};
  quantities[count++] = (SsSolverPublishedQuantity){
      SS_SOLVER_QUANTITY_CONDUCTIVITY, "declare_conductivity", "write_conductivity",
      &outputs->conductivity_view, NULL};
  quantities[count++] = (SsSolverPublishedQuantity){
      SS_SOLVER_QUANTITY_HYDRAULIC_RAMP, "declare_hydraulic_ramp",
      "write_hydraulic_ramp", &outputs->ramp_view, NULL};
  quantities[count++] = (SsSolverPublishedQuantity){
      SS_SOLVER_QUANTITY_HYDRAULIC_EPSILON, "declare_hydraulic_epsilon",
      "write_hydraulic_epsilon", &outputs->epsilon_view, NULL};
  quantities[count++] = (SsSolverPublishedQuantity){
      SS_SOLVER_QUANTITY_PRESSURE_FLUX, "declare_pressure_flux", "write_pressure_flux",
      &outputs->pressure_flux_view, NULL};
  quantities[count++] = (SsSolverPublishedQuantity){
      SS_SOLVER_QUANTITY_DARCY_VELOCITY, "declare_darcy_velocity",
      "write_darcy_velocity", &outputs->darcy_velocity_view, NULL};
  if (outputs->with_coupled_body_force != 0) {
    quantities[count++] = (SsSolverPublishedQuantity){
        SS_SOLVER_QUANTITY_COUPLED_BODY_FORCE, "declare_coupled_body_force",
        "write_coupled_body_force", &outputs->coupled_body_force_view, NULL};
  }
  quantities[count++] =
      (SsSolverPublishedQuantity){SS_SOLVER_QUANTITY_SATURATION, "declare_saturation",
                                  "write_saturation", NULL, &outputs->saturation_view};
  return count;
}

void ss_solver_hydraulic_outputs_destroy(SsSolverHydraulicOutputs *outputs) {
  if (outputs == NULL) {
    return;
  }
  ss_solver_output_category_destroy(&outputs->saturation);
  ss_solver_output_field_destroy(&outputs->coupled_body_force);
  ss_solver_output_field_destroy(&outputs->darcy_velocity);
  ss_solver_output_field_destroy(&outputs->pressure_flux);
  ss_solver_output_field_destroy(&outputs->epsilon);
  ss_solver_output_field_destroy(&outputs->ramp);
  ss_solver_output_field_destroy(&outputs->conductivity);
  ss_solver_output_field_destroy(&outputs->gradient);
  ss_solver_output_field_destroy(&outputs->pressure);
  memset(outputs, 0, sizeof(*outputs));
}
