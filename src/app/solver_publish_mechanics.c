/* Publish accepted mechanics fields and continuation diagnostics.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "solver_output_internal.h"

#include "slope_internal/diagnostics/diagnostics.h"

#include <stdio.h>

static SsStatus ss_solver_append_mechanics_equilibrium(
    SsResultWriter *writer, SsStateId state_id, uint32_t dimension,
    const SsSolverMechanicsEquilibrium *equilibrium) {
  static const char *const reaction_ids[] = {
      "mechanics_reaction_x",
      "mechanics_reaction_y",
      "mechanics_reaction_z",
  };
  static const char *const external_ids[] = {
      "mechanics_external_resultant_x",
      "mechanics_external_resultant_y",
      "mechanics_external_resultant_z",
  };
  const char *force_unit;
  uint32_t component;
  SsStatus status = SS_OK;
  if (writer == NULL || equilibrium == NULL || equilibrium->valid == 0 ||
      (dimension != 2U && dimension != 3U)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  force_unit = dimension == 2U ? "kN/m" : "kN";
  for (component = 0U; status == SS_OK && component < dimension; ++component) {
    status =
        ss_solver_append_analysis_scalar(writer, reaction_ids[component], force_unit,
                                         state_id, equilibrium->reaction[component]);
  }
  for (component = 0U; status == SS_OK && component < dimension; ++component) {
    status = ss_solver_append_analysis_scalar(
        writer, external_ids[component], force_unit, state_id,
        equilibrium->external_resultant[component]);
  }
  if (status == SS_OK) {
    status =
        ss_solver_append_analysis_scalar(writer, "mechanics_reaction_norm", force_unit,
                                         state_id, equilibrium->reaction_norm);
  }
  if (status == SS_OK) {
    status = ss_solver_append_analysis_scalar(
        writer, "mechanics_external_resultant_norm", force_unit, state_id,
        equilibrium->external_resultant_norm);
  }
  if (status == SS_OK) {
    status = ss_solver_append_analysis_scalar(writer, "mechanics_equilibrium_imbalance",
                                              force_unit, state_id,
                                              equilibrium->equilibrium_imbalance_norm);
  }
  if (status == SS_OK) {
    status = ss_solver_append_analysis_scalar(
        writer, "mechanics_normalized_equilibrium", "dimensionless", state_id,
        equilibrium->normalized_equilibrium_imbalance);
  }
  if (status == SS_OK) {
    status = ss_solver_append_analysis_scalar(writer, "mechanics_free_residual_l2",
                                              force_unit, state_id,
                                              equilibrium->free_residual_l2);
  }
  return status;
}

SsStatus ss_solver_publish_mechanics(
    const SsSolverRequest *request, const SsSolverGraph *graph, Vec solution,
    const SsAnalysisRecord *analysis_record, SsSolverMechanics *mechanics,
    SsSolverHydraulic *hydraulic, Vec hydraulic_solution,
    const SsSolverHydraulicEvidence *hydraulic_evidence,
    const SsSolverCouplingSnapshot *coupling_snapshot,
    const SsCouplingDiagnostics *coupling_diagnostics, SsSolverMonitor *monitor,
    SsSolverSummary *summary, SsErrorRecord *error) {
  static const char *const displacement_2d[] = {"x", "y"};
  static const char *const displacement_3d[] = {"x", "y", "z"};
  static const char *const stress_2d[] = {"xx", "yy", "zz", "xy"};
  static const char *const stress_3d[] = {"xx", "yy", "zz", "xy", "yz", "xz"};
  static const char *const strain_2d[] = {"xx", "yy", "xy"};
  static const char *const strain_3d[] = {"xx", "yy", "zz", "xy", "yz", "xz"};
  static const char *const yield_components[] = {"yield"};
  static const char *const material_branch_categories[] = {
      "elastic", "smooth", "left_edge", "right_edge", "apex",
  };
  static const char *const plastic_categories[] = {
      "elastic",
      "plastic",
  };
  Vec local_vector = NULL;
  Vec hydraulic_local_vector = NULL;
  Vec accepted_snapshot = NULL;
  const PetscScalar *local_values = NULL;
  const PetscScalar *hydraulic_values = NULL;
  SsSolverOutputField displacement = {0};
  SsSolverOutputField stress = {0};
  SsSolverOutputField strain = {0};
  SsSolverOutputField yield = {0};
  SsSolverOutputCategory material_branch = {0};
  SsSolverOutputCategory plastic_indicator = {0};
  SsSolverHydraulicOutputs hydraulic_outputs = {0};
  SsResultFieldView displacement_view = {0};
  SsResultFieldView stress_view = {0};
  SsResultFieldView strain_view = {0};
  SsResultFieldView yield_view = {0};
  SsResultCategoryView material_branch_view = {0};
  SsResultCategoryView plastic_indicator_view = {0};
  SsSolverPublishedQuantity quantities[6U + SS_SOLVER_HYDRAULIC_OUTPUT_COUNT];
  size_t quantity_count = 0U;
  SsResultWriterSpec writer_spec;
  SsResultWriter *writer = NULL;
  SsResultWriterSummary writer_summary = {0};
  SsSolverMechanicsEquilibrium equilibrium = {0};
  SsStateId state_id;
  double factor;
  size_t output_state_begin;
  size_t output_state_index;
  const int coupled = hydraulic_solution != NULL;
  const int cancelled = analysis_record != NULL &&
                        analysis_record->stop_reason == SS_ANALYSIS_STOP_CANCELLED;
  SsStatus status;
  const char *publication_stage = "output_extract";
  if (request == NULL || graph == NULL || solution == NULL || analysis_record == NULL ||
      mechanics == NULL || summary == NULL ||
      (analysis_record->terminal_success == 0 && cancelled == 0 &&
       ss_solver_checkpoint_boundary_ready(graph, analysis_record) == 0) ||
      (graph->policy.output_frequency == SS_SOLVER_OUTPUT_ACCEPTED_STATE &&
       monitor == NULL) ||
      analysis_record->point_count == 0U ||
      (coupled != 0 && (hydraulic == NULL || hydraulic_evidence == NULL ||
                        hydraulic_evidence->valid == 0 || coupling_snapshot == NULL ||
                        coupling_diagnostics == NULL)) ||
      (coupled == 0 && (hydraulic != NULL || hydraulic_evidence != NULL ||
                        coupling_snapshot != NULL || coupling_diagnostics != NULL))) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  state_id = (SsStateId)analysis_record->point_count;
  factor = (double)analysis_record->final_factor;
  status = ss_field_space_extract_local(graph->mechanics_space, solution, &local_vector,
                                        &local_values);
  if (status == SS_OK) {
    publication_stage = "displacement_pack";
    status =
        ss_solver_nodal_output(graph, graph->mechanics_space, graph->model.dimension,
                               local_values, &displacement);
  }
  if (status == SS_OK) {
    publication_stage = "stress_pack";
    status = ss_solver_stress_output(
        graph, local_values,
        graph->model.analysis_method == SS_ANALYSIS_LIMIT_LOAD ? 1.0 : factor, &stress,
        &strain, &yield, &material_branch, &plastic_indicator);
  }
  if (status == SS_OK) {
    publication_stage = "mechanics_equilibrium";
    status = ss_solver_mechanics_equilibrium(graph, mechanics, local_values, solution,
                                             factor, &equilibrium, error);
  }
  if (local_values != NULL &&
      VecRestoreArrayRead(local_vector, &local_values) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (VecDestroy(&local_vector) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    (void)ss_memory_view(graph->communicator, "publish_after_mechanics_pack");
  }
  if (status == SS_OK && coupled != 0) {
    publication_stage = "hydraulic_output_extract";
    status = ss_field_space_extract_local(graph->hydraulic_space, hydraulic_solution,
                                          &hydraulic_local_vector, &hydraulic_values);
  }
  if (status == SS_OK && coupled != 0) {
    status = ss_solver_hydraulic_outputs_pack(graph, hydraulic, hydraulic_values,
                                              &hydraulic_outputs, &publication_stage);
  }
  if (status == SS_OK && coupled != 0) {
    publication_stage = "coupled_body_force_pack";
    hydraulic_outputs.with_coupled_body_force = 1;
    status = ss_solver_coupled_body_force_output(graph, coupling_snapshot,
                                                 &hydraulic_outputs.coupled_body_force);
  }
  if (hydraulic_values != NULL &&
      VecRestoreArrayRead(hydraulic_local_vector, &hydraulic_values) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (VecDestroy(&hydraulic_local_vector) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (status != SS_OK) {
    ss_solver_hydraulic_outputs_destroy(&hydraulic_outputs);
    ss_solver_output_category_destroy(&plastic_indicator);
    ss_solver_output_category_destroy(&material_branch);
    ss_solver_output_field_destroy(&yield);
    ss_solver_output_field_destroy(&strain);
    ss_solver_output_field_destroy(&stress);
    ss_solver_output_field_destroy(&displacement);
    if (error != NULL && error->valid == 0) {
      char message[SS_ERROR_MESSAGE_CAPACITY];
      (void)snprintf(message, sizeof(message),
                     "scientific result failed during %s (status=%d)",
                     publication_stage, (int)status);
      ss_solver_error(error, status, "RESULT_PUBLICATION", message, 4U);
    }
    return status;
  }
  displacement_view = ss_solver_nodal_view(
      "displacement", "m",
      graph->model.dimension == 2U ? displacement_2d : displacement_3d,
      displacement.component_count, &displacement, state_id);
  stress_view =
      ss_solver_quadrature_view("stress", "kPa", "mechanics_quadrature",
                                graph->model.dimension == 2U ? stress_2d : stress_3d,
                                stress.component_count, &stress, state_id);
  strain_view =
      ss_solver_quadrature_view("strain", "dimensionless", "mechanics_quadrature",
                                graph->model.dimension == 2U ? strain_2d : strain_3d,
                                strain.component_count, &strain, state_id);
  yield_view =
      ss_solver_quadrature_view("yield_residual", "kPa", "mechanics_quadrature",
                                yield_components, 1U, &yield, state_id);
  material_branch_view = ss_solver_category_view(
      "material_branch", "enum", "mechanics_quadrature", material_branch_categories, 5U,
      &material_branch, state_id);
  plastic_indicator_view =
      ss_solver_category_view("plastic_indicator", "boolean", "mechanics_quadrature",
                              plastic_categories, 2U, &plastic_indicator, state_id);
  quantities[quantity_count++] = (SsSolverPublishedQuantity){
      SS_SOLVER_QUANTITY_DISPLACEMENT, "declare_displacement", "write_displacement",
      &displacement_view, NULL};
  quantities[quantity_count++] = (SsSolverPublishedQuantity){
      SS_SOLVER_QUANTITY_STRESS, "declare_stress", "write_stress", &stress_view, NULL};
  quantities[quantity_count++] = (SsSolverPublishedQuantity){
      SS_SOLVER_QUANTITY_STRAIN, "declare_strain", "write_strain", &strain_view, NULL};
  quantities[quantity_count++] = (SsSolverPublishedQuantity){
      SS_SOLVER_QUANTITY_YIELD_RESIDUAL, "declare_yield_residual",
      "write_yield_residual", &yield_view, NULL};
  quantities[quantity_count++] = (SsSolverPublishedQuantity){
      SS_SOLVER_QUANTITY_MATERIAL_BRANCH, "declare_material_branch",
      "write_material_branch", NULL, &material_branch_view};
  quantities[quantity_count++] = (SsSolverPublishedQuantity){
      SS_SOLVER_QUANTITY_PLASTIC_INDICATOR, "declare_plastic_indicator",
      "write_plastic_indicator", NULL, &plastic_indicator_view};
  if (coupled != 0) {
    ss_solver_hydraulic_outputs_views(graph, state_id, &hydraulic_outputs);
    quantity_count += ss_solver_hydraulic_outputs_list(&hydraulic_outputs,
                                                       quantities + quantity_count);
  }
  writer_spec = (SsResultWriterSpec){
      .final_directory = request->result_directory,
      .problem_id = request->config->problem.id,
      .run_id = request->config->run.id,
      .build_id = request->build_id,
      .parent_checkpoint_id =
          graph->restart_provenance_valid != 0 ? graph->parent_checkpoint_id : NULL,
      .parent_run_id =
          graph->restart_provenance_valid != 0 ? graph->parent_run_id : NULL,
      .parent_build_id =
          graph->restart_provenance_valid != 0 ? graph->parent_build_id : NULL,
      .parent_accepted_state_id = graph->parent_accepted_state_id,
      .parent_source_index_bits = graph->parent_source_index_bits,
      .mesh_id = graph->model.mesh_id,
      .geometry_id = graph->geometry_id,
      .analysis_method = graph->model.result_analysis_method,
      .dimension = graph->model.dimension,
      .field_order = graph->model.mechanics_order,
      .geometry_order = graph->model.coordinate_order,
      .mechanics_assembly_minimum_threads = equilibrium.assembly_minimum_threads,
      .mechanics_assembly_maximum_threads = equilibrium.assembly_maximum_threads,
      .dataset_chunk_rows = 0U,
      .communicator = graph->communicator,
  };
  publication_stage = "writer_create";
  status = ss_result_writer_create(&writer_spec, &writer);
  if (status == SS_OK) {
    publication_stage = "writer_stage";
    status = ss_result_writer_stage(writer);
  }
  if (status == SS_OK) {
    (void)ss_memory_view(graph->communicator, "publish_after_writer_stage");
  }
  if (status == SS_OK) {
    status = ss_solver_publish_declare(writer, graph, quantities, quantity_count,
                                       &publication_stage);
  }
  if (status == SS_OK) {
    publication_stage = "begin_stream";
    status = ss_result_writer_begin_stream(writer);
  }
  if (status == SS_OK) {
    publication_stage = "append_analysis_history";
    status = ss_solver_append_analysis_history(
        writer, analysis_record, mechanics->deflation, coupled != 0 ? 1U : 0U,
        hydraulic_evidence, coupling_diagnostics,
        monitor != NULL && monitor->history_open != 0 ? &monitor->history : NULL);
  }
  if (status == SS_OK) {
    publication_stage = "append_mechanics_equilibrium";
    status = ss_solver_append_mechanics_equilibrium(
        writer, state_id, graph->model.dimension, &equilibrium);
  }
  output_state_begin = graph->policy.output_frequency == SS_SOLVER_OUTPUT_ACCEPTED_STATE
                           ? 0U
                           : analysis_record->point_count - 1U;
  if (status == SS_OK &&
      graph->policy.output_frequency == SS_SOLVER_OUTPUT_ACCEPTED_STATE &&
      monitor->states.count != (uint64_t)analysis_record->point_count) {
    publication_stage = "accepted_state_stream";
    status = SS_ERROR_INTEGRITY;
  }
  if (status == SS_OK &&
      graph->policy.output_frequency == SS_SOLVER_OUTPUT_ACCEPTED_STATE &&
      VecDuplicate(solution, &accepted_snapshot) != PETSC_SUCCESS) {
    status = SS_ERROR_OUT_OF_MEMORY;
  }
  for (output_state_index = output_state_begin;
       status == SS_OK && output_state_index < analysis_record->point_count;
       ++output_state_index) {
    const SsStateId output_state_id = (SsStateId)output_state_index + 1U;
    if (graph->policy.output_frequency == SS_SOLVER_OUTPUT_ACCEPTED_STATE) {
      ss_solver_output_category_destroy(&plastic_indicator);
      ss_solver_output_category_destroy(&material_branch);
      ss_solver_output_field_destroy(&yield);
      ss_solver_output_field_destroy(&strain);
      ss_solver_output_field_destroy(&stress);
      ss_solver_output_field_destroy(&displacement);
      publication_stage = "accepted_state_read";
      status = ss_solver_state_spool_read(
          &monitor->states, (uint64_t)output_state_index, accepted_snapshot);
      status = ss_status_allreduce_max(graph->communicator, status);
      if (status == SS_OK) {
        publication_stage = "accepted_state_extract";
        status = ss_field_space_extract_local(graph->mechanics_space, accepted_snapshot,
                                              &local_vector, &local_values);
      }
      if (status == SS_OK) {
        publication_stage = "accepted_displacement_pack";
        status =
            ss_solver_nodal_output(graph, graph->mechanics_space,
                                   graph->model.dimension, local_values, &displacement);
      }
      if (status == SS_OK) {
        publication_stage = "accepted_stress_pack";
        status = ss_solver_stress_output(
            graph, local_values,
            graph->model.analysis_method == SS_ANALYSIS_LIMIT_LOAD
                ? 1.0
                : (double)analysis_record->points[output_state_index].factor,
            &stress, &strain, &yield, &material_branch, &plastic_indicator);
      }
      if (local_values != NULL &&
          VecRestoreArrayRead(local_vector, &local_values) != PETSC_SUCCESS &&
          status == SS_OK) {
        status = SS_ERROR_INTERNAL;
      }
      local_values = NULL;
      if (VecDestroy(&local_vector) != PETSC_SUCCESS && status == SS_OK) {
        status = SS_ERROR_INTERNAL;
      }
    }
    if (status == SS_OK) {
      ss_solver_output_field_bind(&displacement_view, &displacement, output_state_id);
      ss_solver_output_field_bind(&stress_view, &stress, output_state_id);
      ss_solver_output_field_bind(&strain_view, &strain, output_state_id);
      ss_solver_output_field_bind(&yield_view, &yield, output_state_id);
      ss_solver_output_category_bind(&material_branch_view, &material_branch,
                                     output_state_id);
      ss_solver_output_category_bind(&plastic_indicator_view, &plastic_indicator,
                                     output_state_id);
      if (coupled != 0) {
        ss_solver_hydraulic_outputs_bind(&hydraulic_outputs, output_state_id);
      }
      status = ss_solver_publish_write(writer, graph, quantities, quantity_count,
                                       &publication_stage);
    }
  }
  if (status == SS_OK) {
    (void)ss_memory_view(graph->communicator, "publish_after_quantity_writes");
  }
  if (VecDestroy(&accepted_snapshot) != PETSC_SUCCESS && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK && coupled != 0) {
    publication_stage = "append_coupling_diagnostics";
    status = ss_solver_append_coupled_diagnostics(
        writer, graph->communicator, state_id, graph->model.dimension,
        hydraulic_evidence, coupling_diagnostics);
  }
  if (status == SS_OK && cancelled == 0 &&
      graph->policy.checkpoint_mode == SS_SOLVER_CHECKPOINT_WRITE_ACCEPTED) {
    publication_stage = "write_accepted_checkpoint";
    status = ss_solver_checkpoint_write_accepted(
        request, graph, writer, solution, analysis_record, mechanics,
        hydraulic_solution, hydraulic, NULL, hydraulic_evidence, NULL, state_id, error);
  }
  if (status == SS_OK && cancelled != 0 && monitor != NULL &&
      monitor->boundary_valid != 0 &&
      monitor->boundary.point_count == analysis_record->point_count &&
      (monitor->boundary.next_operation == SS_ANALYSIS_NEXT_SECOND_SEED ||
       monitor->boundary.next_operation == SS_ANALYSIS_NEXT_CONTINUATION_ATTEMPT)) {
    /* A cancelled run publishes a restartable checkpoint of its last boundary. */
    publication_stage = "write_cancellation_checkpoint";
    status = ss_solver_checkpoint_write_accepted(
        request, graph, writer, mechanics->checkpoint_current_state, analysis_record,
        mechanics, hydraulic_solution, hydraulic, NULL, hydraulic_evidence,
        &monitor->boundary, state_id, error);
  }
  if (status == SS_OK) {
    publication_stage = "analysis_outcome";
    status = ss_solver_set_result_outcome(
        writer, request,
        ss_solver_analysis_outcome(analysis_record,
                                   hydraulic_evidence != NULL &&
                                       hydraulic_evidence->nonlinear_outcome ==
                                           SS_NONLINEAR_ACCEPTED_RELAXED),
        mechanics->operator_spec.determinism == SS_ASSEMBLY_QUALIFICATION
            ? "qualification"
            : "performance");
  }
  if (status == SS_OK) {
    publication_stage = "terminalize";
    status = ss_result_writer_terminalize(
        writer, cancelled != 0 ? SS_RESULT_CANCELLED : SS_RESULT_SUCCEEDED,
        ss_solver_analysis_stop_reason(analysis_record->stop_reason), NULL);
  }
  if (status == SS_OK) {
    (void)ss_memory_view(graph->communicator, "publish_after_terminalize");
  }
  if (status == SS_OK) {
    publication_stage = "publish";
    status = ss_result_writer_publish(writer);
  }
  if (status == SS_OK) {
    (void)ss_memory_view(graph->communicator, "publish_after_writer_publish");
  }
  if (writer != NULL) {
    (void)ss_result_writer_summary(writer, &writer_summary);
  }
  if (ss_result_writer_destroy(&writer) != SS_OK && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    (void)ss_memory_view(graph->communicator, "publish_after_writer_destroy");
  }
  ss_solver_hydraulic_outputs_destroy(&hydraulic_outputs);
  ss_solver_output_category_destroy(&plastic_indicator);
  ss_solver_output_category_destroy(&material_branch);
  ss_solver_output_field_destroy(&yield);
  ss_solver_output_field_destroy(&strain);
  ss_solver_output_field_destroy(&stress);
  ss_solver_output_field_destroy(&displacement);
  if (status == SS_OK) {
    (void)ss_memory_view(graph->communicator, "publish_after_output_destroy");
  }
  if (status != SS_OK) {
    if (error->valid == 0) {
      char message[SS_ERROR_MESSAGE_CAPACITY];
      int written = snprintf(message, sizeof(message),
                             "scientific result failed during %s (status=%d)",
                             publication_stage, (int)status);
      ss_solver_error(error, status, "RESULT_PUBLICATION",
                      written >= 0 && (size_t)written < sizeof(message)
                          ? message
                          : "scientific result publication failed",
                      4U);
    }
    return status;
  }
  (void)snprintf(summary->result_id, sizeof(summary->result_id), "%s",
                 writer_summary.result_id);
  summary->accepted_states = (uint64_t)analysis_record->point_count;
  summary->terminal_factor = factor;
  summary->result_published = 1;
  return SS_OK;
}
