/* Result and checkpoint writer: state, schema, index, history, digests and
 * atomic publication; result_checkpoint_restart_cases.c holds the restart,
 * repartitioned restore and rejection cases.
 *
 * Launched by tests/native/support/mpi_layout_runner.py --suite result. The
 * Python reader and visualization tests also run it (SLOPE_RESULT_TEST_EXE)
 * to create result fixtures.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "result_checkpoint_internal.h"

int fixture_rank = -1;
static int fixture_observed_threads = 1;

static int parse_positive(const char *text, int *value) {
  char *end = NULL;
  long parsed;
  if (text == NULL || value == NULL || text[0] == '\0') {
    return 0;
  }
  parsed = strtol(text, &end, 10);
  if (end == text || *end != '\0' || parsed <= 0 || parsed > 4096) {
    return 0;
  }
  *value = (int)parsed;
  return 1;
}

static int parse_arguments(int argc, char **argv, Arguments *arguments) {
  int index;
  memset(arguments, 0, sizeof(*arguments));
  arguments->expected_threads = 1;
  for (index = 1; index < argc; ++index) {
    if (strcmp(argv[index], "--test-id") == 0 && index + 1 < argc) {
      arguments->test_id = argv[++index];
    } else if (strcmp(argv[index], "--artifact-root") == 0 && index + 1 < argc) {
      arguments->artifact_root = argv[++index];
    } else if (strcmp(argv[index], "--output") == 0 && index + 1 < argc) {
      arguments->output = argv[++index];
    } else if (strcmp(argv[index], "--expected-ranks") == 0 && index + 1 < argc) {
      if (!parse_positive(argv[++index], &arguments->expected_ranks)) {
        return 0;
      }
    } else if (strcmp(argv[index], "--expected-threads") == 0 && index + 1 < argc) {
      if (!parse_positive(argv[++index], &arguments->expected_threads)) {
        return 0;
      }
    } else if (strcmp(argv[index], "--checkpoint-directory") == 0 && index + 1 < argc) {
      arguments->checkpoint_directory = argv[++index];
    } else if (strcmp(argv[index], "--checkpoint-mode") == 0 && index + 1 < argc) {
      arguments->checkpoint_mode = argv[++index];
    } else {
      return 0;
    }
  }
  return arguments->test_id != NULL && arguments->artifact_root != NULL &&
         arguments->output != NULL && arguments->expected_ranks > 0 &&
         ((arguments->checkpoint_directory == NULL &&
           arguments->checkpoint_mode == NULL) ||
          (arguments->checkpoint_directory != NULL &&
           arguments->checkpoint_directory[0] == '/' &&
           arguments->checkpoint_mode != NULL &&
           (strcmp(arguments->checkpoint_mode, "write") == 0 ||
            strcmp(arguments->checkpoint_mode, "restore") == 0)));
}

uint64_t fixture_dataset_chunk_rows(void) {
  const char *text = getenv("SLOPE_RESULT_DATASET_CHUNK_ROWS");
  char *end = NULL;
  unsigned long long parsed;
  if (text == NULL || text[0] == '\0') {
    return 0U;
  }
  parsed = strtoull(text, &end, 10);
  if (end == text || *end != '\0' || parsed > UINT32_MAX) {
    return UINT64_MAX;
  }
  return (uint64_t)parsed;
}

static SsEntityId fixture_id(uint64_t ordinal, uint64_t count) {
  return UINT64_C(1000) + ((ordinal * UINT64_C(5)) % count) * UINT64_C(11);
}

double fixture_value(SsEntityId stable_id, uint32_t component) {
  return (double)stable_id * 0.01 + (double)component * 0.125;
}

static void distributed_data_init(DistributedData *data, int rank, int size,
                                  uint64_t global_count) {
  uint64_t ordinal;
  memset(data, 0, sizeof(*data));
  data->global_count = global_count;
  for (ordinal = 0U; ordinal < global_count; ++ordinal) {
    size_t item;
    if ((int)((ordinal * UINT64_C(7) + UINT64_C(3)) % (uint64_t)size) != rank) {
      continue;
    }
    item = data->count++;
    data->ids[item] = fixture_id(ordinal, global_count);
  }
  for (ordinal = 0U; ordinal < (uint64_t)data->count; ++ordinal) {
    const size_t item = (size_t)ordinal;
    const SsEntityId stable_id = data->ids[item];
    const size_t slot = data->count - item - 1U;
    data->slots[item] = slot;
    data->primary[slot * 2U] = fixture_value(stable_id, 0U);
    data->primary[slot * 2U + 1U] = fixture_value(stable_id, 1U);
    data->pressure[slot] = fixture_value(stable_id, 2U);
    {
      uint32_t quadrature;
      for (quadrature = 0U; quadrature < FIXTURE_QUADRATURE_POINTS; ++quadrature) {
        uint32_t component;
        data->saturation[slot * FIXTURE_QUADRATURE_POINTS + quadrature] =
            (uint32_t)((stable_id + quadrature) % UINT64_C(2));
        for (component = 0U; component < FIXTURE_STRESS_COMPONENTS; ++component) {
          const size_t value_index = (slot * FIXTURE_QUADRATURE_POINTS + quadrature) *
                                         FIXTURE_STRESS_COMPONENTS +
                                     component;
          data->stress[value_index] =
              fixture_value(stable_id, component) + 0.01 * (double)quadrature;
        }
      }
    }
    data->plastic_fraction[slot] = (double)(stable_id % UINT64_C(17)) / 16.0;
  }
  data->deflation_global_count = global_count + UINT64_C(5);
  for (ordinal = 0U; ordinal < data->deflation_global_count; ++ordinal) {
    size_t item;
    if ((int)((ordinal * UINT64_C(11) + UINT64_C(1)) % (uint64_t)size) != rank) {
      continue;
    }
    item = data->deflation_count++;
    data->deflation_ids[item] = UINT64_C(90001) + ordinal * UINT64_C(13);
  }
  for (ordinal = 0U; ordinal < (uint64_t)data->deflation_count; ++ordinal) {
    const size_t item = (size_t)ordinal;
    const SsEntityId stable_id = data->deflation_ids[item];
    const size_t slot = data->deflation_count - item - 1U;
    uint32_t basis;
    data->deflation_slots[item] = slot;
    for (basis = 0U; basis < 2U; ++basis) {
      data->deflation_left[slot * 2U + basis] =
          fixture_value(stable_id, basis) * (double)(basis + 1U);
      data->deflation_right[slot * 2U + basis] =
          fixture_value(stable_id, basis) * (double)(basis + 3U);
      data->deflation_source[slot * 2U + basis] =
          fixture_value(stable_id, basis) * (double)(basis + 5U);
    }
  }
  data->deflation_source_weights[0] = 1.5;
  data->deflation_source_weights[1] = 2.5;
  data->deflation_source_kinds[0] = 1.0;
  data->deflation_source_kinds[1] = 2.0;
}

static int observe_thread_team(const DistributedData *data, int *observed_threads) {
  double serial = 0.0;
  double parallel = 0.0;
  size_t item;
  int observed = 1;
  for (item = 0U; item < data->count; ++item) {
    serial += data->primary[data->slots[item] * 2U];
  }
#ifdef _OPENMP
#pragma omp parallel reduction(+ : parallel)
  {
#pragma omp single
    observed = omp_get_num_threads();
#pragma omp for schedule(static)
    for (item = 0U; item < data->count; ++item) {
      parallel += data->primary[data->slots[item] * 2U];
    }
  }
#else
  parallel = serial;
#endif
  *observed_threads = observed;
  return fabs(parallel - serial) <= 1.0e-13 * fmax(1.0, fabs(serial));
}

int make_path(char output[SS_RESULT_PATH_CAPACITY], const char *root,
              const char *leaf) {
  int count = snprintf(output, SS_RESULT_PATH_CAPACITY, "%s/%s", root, leaf);
  return count > 0 && (size_t)count < SS_RESULT_PATH_CAPACITY;
}

static SsResultWriterSpec writer_spec(const char *path, MPI_Comm communicator) {
  SsResultWriterSpec spec;
  memset(&spec, 0, sizeof(spec));
  spec.final_directory = path;
  spec.problem_id =
      "sha256:1111111111111111111111111111111111111111111111111111111111111111";
  spec.run_id =
      "sha256:2222222222222222222222222222222222222222222222222222222222222222";
  spec.build_id =
      "sha256:3333333333333333333333333333333333333333333333333333333333333333";
  spec.mesh_id =
      "sha256:4444444444444444444444444444444444444444444444444444444444444444";
  spec.geometry_id =
      "sha256:5555555555555555555555555555555555555555555555555555555555555555";
  spec.analysis_method = "direct_ssr";
  spec.dimension = 2U;
  spec.field_order = 2U;
  spec.geometry_order = 4U;
  spec.dataset_chunk_rows = fixture_dataset_chunk_rows();
  spec.communicator = communicator;
  return spec;
}

static SsResultFieldView displacement_view(const DistributedData *data,
                                           SsStateId state_id) {
  static const char *const components[] = {"x", "y"};
  SsResultFieldView view;
  memset(&view, 0, sizeof(view));
  view.quantity_id = "displacement";
  view.quantity_version = 1U;
  view.unit = "m";
  view.location = "field_dof";
  view.basis = "lagrange";
  view.component_ids = components;
  view.component_count = 2U;
  view.state_id = state_id;
  view.stable_ids = data->ids;
  view.local_slots = data->slots;
  view.values = data->primary;
  view.local_count = data->count;
  view.global_count = data->global_count;
  return view;
}

static SsResultFieldView pressure_view(const DistributedData *data,
                                       SsStateId state_id) {
  static const char *const components[] = {"pressure"};
  SsResultFieldView view;
  memset(&view, 0, sizeof(view));
  view.quantity_id = "pore_pressure";
  view.quantity_version = 1U;
  view.unit = "Pa";
  view.location = "field_dof";
  view.basis = "lagrange";
  view.component_ids = components;
  view.component_count = 1U;
  view.state_id = state_id;
  view.stable_ids = data->ids;
  view.local_slots = data->slots;
  view.values = data->pressure;
  view.local_count = data->count;
  view.global_count = data->global_count;
  return view;
}

static SsResultFieldView stress_view(const DistributedData *data, SsStateId state_id) {
  static const char *const components[] = {"xx", "yy", "zz", "xy"};
  SsResultFieldView view;
  memset(&view, 0, sizeof(view));
  view.quantity_id = "stress";
  view.quantity_version = 1U;
  view.unit = "kPa";
  view.location = "mechanics_quadrature";
  view.basis = "cell_quadrature";
  view.component_ids = components;
  view.component_count = FIXTURE_STRESS_COMPONENTS;
  view.state_id = state_id;
  view.stable_ids = data->ids;
  view.local_slots = data->slots;
  view.values = data->stress;
  view.local_count = data->count;
  view.global_count = data->global_count;
  view.dataset_kind = SS_RESULT_DATASET_QUADRATURE;
  view.quadrature_id = "triangle_degree_4_v1";
  view.sample_count = FIXTURE_QUADRATURE_POINTS;
  return view;
}

static SsResultCategoryView saturation_view(const DistributedData *data,
                                            SsStateId state_id) {
  static const char *const categories[] = {"unsaturated", "saturated"};
  SsResultCategoryView view;
  memset(&view, 0, sizeof(view));
  view.quantity_id = "saturation";
  view.quantity_version = 1U;
  view.unit = "boolean";
  view.location = "hydraulic_quadrature";
  view.quadrature_id = "triangle_degree_4_v1";
  view.category_ids = categories;
  view.category_count = 2U;
  view.state_id = state_id;
  view.stable_ids = data->ids;
  view.local_slots = data->slots;
  view.values = data->saturation;
  view.local_count = data->count;
  view.global_count = data->global_count;
  view.sample_count = FIXTURE_QUADRATURE_POINTS;
  return view;
}

static SsResultFieldView plastic_fraction_view(const DistributedData *data,
                                               SsStateId state_id) {
  static const char *const components[] = {"fraction"};
  SsResultFieldView view;
  memset(&view, 0, sizeof(view));
  view.quantity_id = "plastic_fraction";
  view.quantity_version = 1U;
  view.unit = "dimensionless";
  view.location = "cell";
  view.basis = "cell_reduction";
  view.component_ids = components;
  view.component_count = 1U;
  view.state_id = state_id;
  view.stable_ids = data->ids;
  view.local_slots = data->slots;
  view.values = data->plastic_fraction;
  view.local_count = data->count;
  view.global_count = data->global_count;
  view.dataset_kind = SS_RESULT_DATASET_CELL_REDUCTION;
  view.sample_count = 1U;
  return view;
}

static SsResultQuantityDeclaration quantity_declaration(const SsResultFieldView *view) {
  SsResultQuantityDeclaration declaration;
  memset(&declaration, 0, sizeof(declaration));
  declaration.quantity_id = view->quantity_id;
  declaration.quantity_version = view->quantity_version;
  declaration.unit = view->unit;
  declaration.location = view->location;
  declaration.basis = view->basis;
  declaration.component_ids = view->component_ids;
  declaration.component_count = view->component_count;
  declaration.stable_ids = view->stable_ids;
  declaration.local_slots = view->local_slots;
  declaration.local_count = view->local_count;
  declaration.global_count = view->global_count;
  declaration.dataset_kind = view->dataset_kind;
  declaration.quadrature_id = view->quadrature_id;
  declaration.sample_count = view->sample_count;
  return declaration;
}

static SsResultCategoryDeclaration
category_declaration(const SsResultCategoryView *view) {
  SsResultCategoryDeclaration declaration;
  memset(&declaration, 0, sizeof(declaration));
  declaration.quantity_id = view->quantity_id;
  declaration.quantity_version = view->quantity_version;
  declaration.unit = view->unit;
  declaration.location = view->location;
  declaration.quadrature_id = view->quadrature_id;
  declaration.category_ids = view->category_ids;
  declaration.category_count = view->category_count;
  declaration.stable_ids = view->stable_ids;
  declaration.local_slots = view->local_slots;
  declaration.local_count = view->local_count;
  declaration.global_count = view->global_count;
  declaration.sample_count = view->sample_count;
  return declaration;
}

static SsStatus declare_result_quantities(SsResultWriter *writer,
                                          const DistributedData *data) {
  SsResultFieldView displacement = displacement_view(data, 11U);
  SsResultFieldView pressure = pressure_view(data, 17U);
  SsResultFieldView stress = stress_view(data, 17U);
  SsResultCategoryView saturation = saturation_view(data, 17U);
  SsResultFieldView plastic_fraction = plastic_fraction_view(data, 17U);
  SsResultQuantityDeclaration displacement_declaration =
      quantity_declaration(&displacement);
  SsResultQuantityDeclaration pressure_declaration = quantity_declaration(&pressure);
  SsResultQuantityDeclaration stress_declaration = quantity_declaration(&stress);
  SsResultCategoryDeclaration saturation_declaration =
      category_declaration(&saturation);
  SsResultQuantityDeclaration reduction_declaration =
      quantity_declaration(&plastic_fraction);
  SsStatus status =
      ss_result_writer_declare_quantity(writer, &displacement_declaration);
  if (status == SS_OK) {
    status = ss_result_writer_declare_quantity(writer, &pressure_declaration);
  }
  if (status == SS_OK) {
    status = ss_result_writer_declare_quantity(writer, &stress_declaration);
  }
  if (status == SS_OK) {
    status = ss_result_writer_declare_category(writer, &saturation_declaration);
  }
  if (status == SS_OK) {
    status = ss_result_writer_declare_quantity(writer, &reduction_declaration);
  }
  return status;
}

static SsResultAcceptedState accepted_state_for(SsStateId state_id,
                                                SsStateId parent_state_id,
                                                uint64_t accepted_ordinal,
                                                uint64_t history_begin,
                                                uint64_t history_end) {
  SsResultAcceptedState state;
  memset(&state, 0, sizeof(state));
  state.state_id = state_id;
  state.parent_state_id = parent_state_id;
  state.accepted_ordinal = accepted_ordinal;
  state.hydraulic_generation = 7U;
  state.mechanics_generation = 9U;
  state.control_generation = 12U;
  state.factor = 1.0 + 0.125 * (double)accepted_ordinal;
  state.work = 0.25 + 0.25 * (double)accepted_ordinal;
  state.direct_control = 0.005 + 0.01 * (double)accepted_ordinal;
  state.displacement_maximum = 0.5 + 0.5 * (double)accepted_ordinal;
  state.history_begin = history_begin;
  state.history_end = history_end;
  state.complete = 1;
  state.accepted = 1;
  return state;
}

static SsResultAcceptedState accepted_state(void) {
  return accepted_state_for(17U, 0U, 2U, 0U, 2U);
}

static SsResultHistoryRow history_row_for(uint64_t row_id, int accepted,
                                          SsStateId state_id) {
  SsResultHistoryRow row;
  memset(&row, 0, sizeof(row));
  row.kind = accepted != 0 ? SS_RESULT_HISTORY_ACCEPTED : SS_RESULT_HISTORY_ATTEMPT;
  row.row_id = row_id;
  row.state_id = state_id;
  row.attempt_id = 40U + row_id;
  row.parent_row_id = row_id == 0U ? 0U : row_id - 1U;
  row.ordinal = row_id;
  row.control = 1.0 + 0.1 * (double)row_id;
  row.residual = 0.01 / (double)(row_id + 1U);
  row.complete = 1;
  row.accepted = accepted;
  return row;
}

static SsResultHistoryRow history_row(uint64_t row_id, int accepted) {
  return history_row_for(row_id, accepted, accepted != 0 ? 17U : 0U);
}

static int write_result(const char *path, MPI_Comm communicator,
                        const DistributedData *data,
                        SsResultTerminalStatus terminal_status,
                        SsResultFailurePoint failure_point,
                        SsResultWriterSummary *summary) {
  SsResultWriterSpec spec = writer_spec(path, communicator);
  SsResultFieldView displacement_first = displacement_view(data, 11U);
  SsResultFieldView displacement_terminal = displacement_view(data, 17U);
  SsResultFieldView pressure = pressure_view(data, 17U);
  SsResultFieldView stress = stress_view(data, 17U);
  SsResultCategoryView saturation = saturation_view(data, 17U);
  SsResultFieldView plastic_fraction = plastic_fraction_view(data, 17U);
  SsResultAcceptedState first_state = accepted_state_for(11U, 0U, 1U, 0U, 2U);
  SsResultAcceptedState terminal_state = accepted_state_for(17U, 11U, 2U, 2U, 4U);
  SsResultHistoryRow first_attempt = history_row_for(0U, 0, 11U);
  SsResultHistoryRow first_accepted = history_row_for(1U, 1, 11U);
  SsResultHistoryRow terminal_attempt = history_row_for(2U, 0, 17U);
  SsResultHistoryRow terminal_accepted = history_row_for(3U, 1, 17U);
  SsResultScalarValue factor = {
      .quantity_id = "stability_factor",
      .quantity_version = 1U,
      .unit = "dimensionless",
      .state_id = 17U,
      .value = 1.25,
      .complete = 1,
  };
  SsResultCheckpointReference checkpoint = {
      .state_id = 17U,
      .checkpoint_id =
          "sha256:6666666666666666666666666666666666666666666666666666666666666666",
      .relative_path = "checkpoints/17",
      .complete = 1,
  };
  SsErrorRecord error;
  SsResultWriter *writer = NULL;
  SsStatus status;
  memset(&error, 0, sizeof(error));
  error.valid = terminal_status == SS_RESULT_FAILED;
  error.status = SS_ERROR_INTERNAL;
  strcpy(error.code, "NUMERICAL_FIXTURE_FAILURE");
  strcpy(error.message, "deterministic fixture failure");
  status = ss_result_writer_create(&spec, &writer);
  if (status == SS_OK) {
    status = ss_result_writer_inject_failure(writer, failure_point);
  }
  if (status == SS_OK) {
    status = ss_result_writer_stage(writer);
  }
  if (status == SS_OK) {
    status = declare_result_quantities(writer, data);
  }
  if (status == SS_OK) {
    status = ss_result_writer_begin_stream(writer);
  }
  if (status == SS_OK) {
    status = ss_result_writer_append_history(writer, &first_attempt);
  }
  if (status == SS_OK) {
    status = ss_result_writer_append_history(writer, &first_accepted);
  }
  if (status == SS_OK) {
    status = ss_result_writer_accept_state(writer, &first_state);
  }
  if (status == SS_OK) {
    status = ss_result_writer_write_field(writer, &displacement_first);
  }
  if (status == SS_OK) {
    status = ss_result_writer_append_history(writer, &terminal_attempt);
  }
  if (status == SS_OK) {
    status = ss_result_writer_append_history(writer, &terminal_accepted);
  }
  if (status == SS_OK) {
    status = ss_result_writer_accept_state(writer, &terminal_state);
  }
  if (status == SS_OK) {
    status = ss_result_writer_write_field(writer, &displacement_terminal);
  }
  if (status == SS_OK) {
    status = ss_result_writer_write_field(writer, &pressure);
  }
  if (status == SS_OK) {
    status = ss_result_writer_write_quantity(writer, &stress);
  }
  if (status == SS_OK) {
    status = ss_result_writer_write_category(writer, &saturation);
  }
  if (status == SS_OK) {
    status = ss_result_writer_write_quantity(writer, &plastic_fraction);
  }
  if (status == SS_OK) {
    status = ss_result_writer_append_scalar(writer, &factor);
  }
  if (status == SS_OK) {
    status = ss_result_writer_index_checkpoint(writer, &checkpoint);
  }
  if (status == SS_OK && terminal_status != SS_RESULT_FAILED) {
    /* The fixture declares what its synthetic analysis achieved. */
    const SsResultOutcome fixture_outcome = {
        .outcome = terminal_status == SS_RESULT_SUCCEEDED ? "converged_criterion"
                                                          : "cancelled",
        .validation_policy = "high_accuracy",
    };
    status = ss_result_writer_set_outcome(writer, &fixture_outcome);
  }
  if (status == SS_OK) {
    status = ss_result_writer_terminalize(
        writer, terminal_status,
        terminal_status == SS_RESULT_SUCCEEDED
            ? "converged"
            : (terminal_status == SS_RESULT_FAILED ? "nonlinear_failure"
                                                   : "user_cancelled"),
        terminal_status == SS_RESULT_FAILED ? &error : NULL);
  }
  if (status == SS_OK) {
    status = ss_result_writer_publish(writer);
  }
  if (summary != NULL && writer != NULL) {
    (void)ss_result_writer_summary(writer, summary);
  }
  if (ss_result_writer_destroy(&writer) != SS_OK && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  return (int)status;
}

static int write_empty_terminal(const char *path, MPI_Comm communicator,
                                SsResultTerminalStatus terminal_status,
                                SsResultWriterSummary *summary) {
  SsResultWriterSpec spec = writer_spec(path, communicator);
  SsResultWriter *writer = NULL;
  SsErrorRecord error;
  SsStatus status;
  memset(&error, 0, sizeof(error));
  error.valid = terminal_status == SS_RESULT_FAILED;
  error.status = SS_ERROR_INTERNAL;
  strcpy(error.code, "PREFLIGHT_FIXTURE_FAILURE");
  strcpy(error.message, "fixture failed before a scientific state existed");
  status = ss_result_writer_create(&spec, &writer);
  if (status == SS_OK) {
    status = ss_result_writer_stage(writer);
  }
  if (status == SS_OK) {
    status = ss_result_writer_begin_stream(writer);
  }
  if (status == SS_OK) {
    status = ss_result_writer_terminalize(
        writer, terminal_status,
        terminal_status == SS_RESULT_FAILED ? "preflight_failure" : "user_cancelled",
        terminal_status == SS_RESULT_FAILED ? &error : NULL);
  }
  if (status == SS_OK) {
    status = ss_result_writer_publish(writer);
  }
  if (summary != NULL && writer != NULL) {
    (void)ss_result_writer_summary(writer, summary);
  }
  if (ss_result_writer_destroy(&writer) != SS_OK && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  return (int)status;
}

static int test_state(const Arguments *arguments, MPI_Comm communicator,
                      const DistributedData *data, TestOutcome *outcome) {
  char path[SS_RESULT_PATH_CAPACITY];
  SsResultWriterSpec spec;
  SsResultWriter *writer = NULL;
  SsResultWriterSummary summary;
  SsResultFieldView field = displacement_view(data, 17U);
  SsResultFieldView undeclared = pressure_view(data, 17U);
  SsResultQuantityDeclaration declaration = quantity_declaration(&field);
  SsResultAcceptedState state = accepted_state();
  SsResultHistoryRow attempt = history_row(0U, 0);
  SsResultHistoryRow accepted = history_row(1U, 1);
  CHECK(make_path(path, arguments->artifact_root, "state-result"));
  spec = writer_spec(path, communicator);
  spec.parent_checkpoint_id =
      "sha256:6666666666666666666666666666666666666666666666666666666666666666";
  CHECK(ss_result_writer_create(&spec, &writer) == SS_ERROR_INVALID_ARGUMENT);
  CHECK(writer == NULL);
  spec.parent_run_id = spec.run_id;
  spec.parent_build_id =
      "sha256:8888888888888888888888888888888888888888888888888888888888888888";
  spec.parent_accepted_state_id = 17U;
  spec.parent_source_index_bits = 32U;
  CHECK(ss_result_writer_create(&spec, &writer) == SS_ERROR_INVALID_ARGUMENT);
  CHECK(writer == NULL);
  spec.parent_run_id =
      "sha256:7777777777777777777777777777777777777777777777777777777777777777";
  CHECK(ss_result_writer_create(&spec, &writer) == SS_OK);
  CHECK(ss_result_writer_begin_stream(writer) == SS_ERROR_INVALID_STATE);
  CHECK(ss_result_writer_publish(writer) == SS_ERROR_INVALID_STATE);
  CHECK(ss_result_writer_stage(writer) == SS_OK);
  CHECK(ss_result_writer_stage(writer) == SS_ERROR_INVALID_STATE);
  CHECK(ss_result_writer_write_field(writer, &field) == SS_ERROR_INVALID_STATE);
  CHECK(ss_result_writer_declare_quantity(writer, &declaration) == SS_OK);
  CHECK(ss_result_writer_declare_quantity(writer, &declaration) == SS_ERROR_DUPLICATE);
  CHECK(ss_result_writer_begin_stream(writer) == SS_OK);
  CHECK(ss_result_writer_declare_quantity(writer, &declaration) ==
        SS_ERROR_INVALID_STATE);
  CHECK(ss_result_writer_append_history(writer, &attempt) == SS_OK);
  CHECK(ss_result_writer_append_history(writer, &accepted) == SS_OK);
  CHECK(ss_result_writer_accept_state(writer, &state) == SS_OK);
  CHECK(ss_result_writer_write_field(writer, &undeclared) == SS_ERROR_SCHEMA);
  CHECK(ss_result_writer_write_field(writer, &field) == SS_OK);
  CHECK(ss_result_writer_terminalize(writer, SS_RESULT_SUCCEEDED, "converged", NULL) ==
        SS_OK);
  CHECK(ss_result_writer_write_field(writer, &field) == SS_ERROR_INVALID_STATE);
  CHECK(ss_result_writer_publish(writer) == SS_OK);
  CHECK(ss_result_writer_publish(writer) == SS_ERROR_INVALID_STATE);
  memset(&summary, 0, sizeof(summary));
  CHECK(ss_result_writer_summary(writer, &summary) == SS_OK);
  CHECK(summary.writer_state == SS_RESULT_WRITER_PUBLISHED);
  CHECK(summary.terminal_status == SS_RESULT_SUCCEEDED);
  strcpy(outcome->invariant, summary.logical_digest);
  outcome->logical_bytes = summary.logical_bytes;
  outcome->payload_bytes = summary.payload_bytes;
  CHECK(ss_result_writer_destroy(&writer) == SS_OK);
  CHECK(writer == NULL);
  return 0;
}

static int test_index(MPI_Comm communicator, const DistributedData *data,
                      TestOutcome *outcome) {
  SsResultFileIndexSpec spec = {
      communicator, data->ids, data->slots, data->count, data->global_count,
  };
  SsResultFileIndexPlan *plan = NULL;
  SsResultFileIndexSummary summary;
  size_t local;
  CHECK(ss_result_file_index_plan_create(&spec, &plan) == SS_OK);
  for (local = 0U; local < data->count; ++local) {
    SsResultFileIndexEntry entry;
    uint64_t ordinal;
    uint64_t expected = 0U;
    CHECK(ss_result_file_index_plan_entry(plan, local, &entry) == SS_OK);
    for (ordinal = 0U; ordinal < data->global_count; ++ordinal) {
      if (fixture_id(ordinal, data->global_count) < data->ids[local]) {
        ++expected;
      }
    }
    CHECK(entry.stable_id == data->ids[local]);
    CHECK(entry.local_slot == data->slots[local]);
    CHECK(entry.file_index == expected);
  }
  memset(&summary, 0, sizeof(summary));
  CHECK(ss_result_file_index_plan_summary(plan, &summary) == SS_OK);
  CHECK(summary.global_count == data->global_count);
  CHECK(summary.local_count == data->count);
  CHECK(summary.root_gathered_value_bytes == 0U);
  outcome->checksum = (double)summary.contiguous_runs;
  outcome->logical_bytes = summary.exchanged_bytes;
  CHECK(ss_result_file_index_plan_destroy(&plan) == SS_OK);
  CHECK(plan == NULL);
  if (fixture_rank == 0 && data->global_count > 1U) {
    SsEntityId duplicate_ids[2] = {101U, 101U};
    size_t slots[2] = {0U, 1U};
    SsResultFileIndexSpec duplicate = {
        MPI_COMM_SELF, duplicate_ids, slots, 2U, 2U,
    };
    CHECK(ss_result_file_index_plan_create(&duplicate, &plan) == SS_ERROR_DUPLICATE);
    CHECK(plan == NULL);
    {
      SsEntityId zero_ids[2] = {0U, 2U};
      size_t zero_slots[2] = {0U, 1U};
      SsResultFileIndexSpec zero = {
          MPI_COMM_SELF, zero_ids, zero_slots, 2U, 2U,
      };
      SsResultFileIndexEntry first;
      CHECK(ss_result_file_index_plan_create(&zero, &plan) == SS_OK);
      CHECK(ss_result_file_index_plan_entry(plan, 0U, &first) == SS_OK);
      CHECK(first.stable_id == 0U);
      CHECK(first.file_index == 0U);
      CHECK(ss_result_file_index_plan_destroy(&plan) == SS_OK);
    }
  }
  return 0;
}

static int verify_schema_file(const char *directory, uint64_t dataset_chunk_rows,
                              uint64_t global_count) {
  char hdf5_path[SS_RESULT_PATH_CAPACITY];
  char json_path[SS_RESULT_PATH_CAPACITY];
  char envelope[8192];
  FILE *stream;
  size_t bytes;
  hid_t file;
  hid_t dataset;
  hid_t creation;
  hid_t space;
  hid_t data_type;
  hsize_t dimensions[4] = {0U, 0U, 0U, 0U};
  int rank;
  static const char *const groups[] = {
      "/states",          "/fields",  "/quadrature",  "/quadrature_categories",
      "/cell_reductions", "/history", "/diagnostics",
  };
  size_t index;
  (void)dataset_chunk_rows;
  CHECK(make_path(hdf5_path, directory, "result.h5"));
  CHECK(make_path(json_path, directory, "result.json"));
  file = H5Fopen(hdf5_path, H5F_ACC_RDONLY, H5P_DEFAULT);
  CHECK(file >= 0);
  CHECK(H5Aexists(file, "layout_id") > 0);
  CHECK(H5Aexists(file, "schema_major") > 0);
  CHECK(H5Aexists(file, "collective_plan_digest") > 0);
  CHECK(H5Aexists(file, "collective_plan_frozen") > 0);
  for (index = 0U; index < sizeof(groups) / sizeof(groups[0]); ++index) {
    CHECK(H5Lexists(file, groups[index], H5P_DEFAULT) > 0);
  }
  dataset = H5Dopen2(file, "/fields/displacement/values", H5P_DEFAULT);
  CHECK(dataset >= 0);
  creation = H5Dget_create_plist(dataset);
  CHECK(creation >= 0);
  CHECK(H5Pget_layout(creation) == H5D_CHUNKED);
  CHECK(H5Pclose(creation) >= 0);
  space = H5Dget_space(dataset);
  CHECK(space >= 0);
  rank = H5Sget_simple_extent_ndims(space);
  CHECK(rank == 3);
  CHECK(H5Sget_simple_extent_dims(space, dimensions, NULL) == rank);
  CHECK(dimensions[0] == 2U);
  CHECK(dimensions[1] == global_count);
  CHECK(dimensions[2] == 2U);
  CHECK(H5Sclose(space) >= 0);
  for (index = 0U; index < sizeof((const char *const[]){
                               "quantity_id", "quantity_version", "unit", "mesh_id",
                               "geometry_id", "state_ids", "completeness", "value_type",
                               "digest_method", "digest"}) /
                               sizeof(const char *);
       ++index) {
    static const char *const attributes[] = {
        "quantity_id", "quantity_version", "unit",       "mesh_id",       "geometry_id",
        "state_ids",   "completeness",     "value_type", "digest_method", "digest",
    };
    CHECK(H5Aexists(dataset, attributes[index]) > 0);
  }
  CHECK(H5Dclose(dataset) >= 0);
  dataset = H5Dopen2(file, "/fields/displacement/state_ids", H5P_DEFAULT);
  CHECK(dataset >= 0);
  {
    uint64_t state_ids[2] = {0U, 0U};
    CHECK(H5Dread(dataset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  state_ids) >= 0);
    CHECK(state_ids[0] == 11U);
    CHECK(state_ids[1] == 17U);
  }
  CHECK(H5Dclose(dataset) >= 0);
  dataset = H5Dopen2(file, "/quadrature/stress/values", H5P_DEFAULT);
  CHECK(dataset >= 0);
  space = H5Dget_space(dataset);
  CHECK(space >= 0);
  rank = H5Sget_simple_extent_ndims(space);
  CHECK(rank == 4);
  CHECK(H5Sget_simple_extent_dims(space, dimensions, NULL) == rank);
  CHECK(dimensions[0] == 1U);
  CHECK(dimensions[1] == global_count);
  CHECK(dimensions[2] == FIXTURE_QUADRATURE_POINTS);
  CHECK(dimensions[3] == FIXTURE_STRESS_COMPONENTS);
  CHECK(H5Aexists(dataset, "quadrature_id") > 0);
  CHECK(H5Aexists(dataset, "dataset_kind") > 0);
  CHECK(H5Sclose(space) >= 0);
  CHECK(H5Dclose(dataset) >= 0);
  dataset = H5Dopen2(file, "/quadrature_categories/saturation/values", H5P_DEFAULT);
  CHECK(dataset >= 0);
  space = H5Dget_space(dataset);
  CHECK(space >= 0);
  rank = H5Sget_simple_extent_ndims(space);
  CHECK(rank == 3);
  CHECK(H5Sget_simple_extent_dims(space, dimensions, NULL) == rank);
  CHECK(dimensions[0] == 1U);
  CHECK(dimensions[1] == global_count);
  CHECK(dimensions[2] == FIXTURE_QUADRATURE_POINTS);
  data_type = H5Dget_type(dataset);
  CHECK(data_type >= 0);
  CHECK(H5Tget_class(data_type) == H5T_INTEGER);
  CHECK(H5Tget_size(data_type) == sizeof(uint32_t));
  CHECK(H5Tclose(data_type) >= 0);
  CHECK(H5Aexists(dataset, "category_ids") > 0);
  CHECK(H5Aexists(dataset, "quadrature_id") > 0);
  CHECK(H5Aexists(dataset, "value_type") > 0);
  CHECK(H5Sclose(space) >= 0);
  CHECK(H5Dclose(dataset) >= 0);
  dataset = H5Dopen2(file, "/cell_reductions/plastic_fraction/values", H5P_DEFAULT);
  CHECK(dataset >= 0);
  space = H5Dget_space(dataset);
  CHECK(space >= 0);
  rank = H5Sget_simple_extent_ndims(space);
  CHECK(rank == 3);
  CHECK(H5Sget_simple_extent_dims(space, dimensions, NULL) == rank);
  CHECK(dimensions[0] == 1U);
  CHECK(dimensions[1] == global_count);
  CHECK(dimensions[2] == 1U);
  CHECK(H5Sclose(space) >= 0);
  CHECK(H5Dclose(dataset) >= 0);
  CHECK(H5Fclose(file) >= 0);
  stream = fopen(json_path, "rb");
  CHECK(stream != NULL);
  bytes = fread(envelope, 1U, sizeof(envelope) - 1U, stream);
  CHECK(ferror(stream) == 0);
  CHECK(fclose(stream) == 0);
  envelope[bytes] = '\0';
  CHECK(strstr(envelope, "\"kind\":\"slope_stability.result\"") != NULL);
  CHECK(strstr(envelope, "\"terminal_status\"") != NULL);
  CHECK(strstr(envelope, "\"analysis_outcome\"") != NULL);
  CHECK(strstr(envelope, "\"stability_factor\"") != NULL);
  CHECK(strstr(envelope, "\"collective_plan_digest\"") != NULL);
  CHECK(strstr(envelope, "\"checkpoints/17\"") != NULL);
  CHECK(strstr(envelope, "\"result_pack\"") != NULL);
  CHECK(strstr(envelope, "\"artifacts\"") != NULL);
  CHECK(strstr(envelope, "\"errors\"") != NULL);
  return 0;
}

static int test_schema_or_hdf5(const Arguments *arguments, MPI_Comm communicator,
                               const DistributedData *data, TestOutcome *outcome,
                               int schema_only) {
  char path[SS_RESULT_PATH_CAPACITY];
  SsResultWriterSummary summary;
  SsResultWriterSummary verified_summary;
  int communicator_size = 1;
  CHECK(MPI_Comm_size(communicator, &communicator_size) == MPI_SUCCESS);
  if (communicator_size > 1) {
    SsResultWriter *writer = NULL;
    SsResultWriterSpec disagree_spec;
    SsResultFieldView disagree_field = displacement_view(data, 17U);
    SsResultQuantityDeclaration disagree_declaration;
    CHECK(make_path(path, arguments->artifact_root, "rank-metadata-disagree"));
    disagree_spec = writer_spec(path, communicator);
    if (fixture_rank == 1) {
      disagree_field.unit = "mm";
    }
    disagree_declaration = quantity_declaration(&disagree_field);
    CHECK(ss_result_writer_create(&disagree_spec, &writer) == SS_OK);
    CHECK(ss_result_writer_stage(writer) == SS_OK);
    CHECK(ss_result_writer_declare_quantity(writer, &disagree_declaration) ==
          SS_ERROR_INTEGRITY);
    CHECK(ss_result_writer_destroy(&writer) == SS_OK);
    if (fixture_rank == 0) {
      CHECK(access(path, F_OK) != 0);
      CHECK(errno == ENOENT);
    }
    CHECK(MPI_Barrier(communicator) == MPI_SUCCESS);
  }
  CHECK(make_path(path, arguments->artifact_root,
                  schema_only != 0 ? "schema-result" : "parallel-result"));
  memset(&summary, 0, sizeof(summary));
  memset(&verified_summary, 0, sizeof(verified_summary));
  CHECK(write_result(path, communicator, data, SS_RESULT_SUCCEEDED, SS_RESULT_FAIL_NONE,
                     &summary) == SS_OK);
  CHECK(ss_result_verify_terminal(communicator, path, &verified_summary) == SS_OK);
  CHECK(strcmp(summary.logical_digest, verified_summary.logical_digest) == 0);
  CHECK(strcmp(summary.payload_sha256, verified_summary.payload_sha256) == 0);
  CHECK(summary.payload_bytes == verified_summary.payload_bytes);
  if (fixture_rank == 0) {
    CHECK(verify_schema_file(path, fixture_dataset_chunk_rows(), data->global_count) ==
          0);
  }
  CHECK(MPI_Barrier(communicator) == MPI_SUCCESS);
  CHECK(summary.field_count == 5U);
  CHECK(summary.accepted_state_count == 2U);
  CHECK(summary.history_row_count == 4U);
  strcpy(outcome->invariant, summary.logical_digest);
  strcpy(outcome->payload_sha256, summary.payload_sha256);
  outcome->logical_bytes = summary.logical_bytes;
  outcome->payload_bytes = summary.payload_bytes;
  return 0;
}

static int test_history(const Arguments *arguments, MPI_Comm communicator,
                        const DistributedData *data, TestOutcome *outcome) {
  char path[SS_RESULT_PATH_CAPACITY];
  SsResultWriterSpec spec;
  SsResultWriter *writer = NULL;
  SsResultHistoryRow incomplete = history_row(0U, 0);
  SsResultHistoryRow complete = history_row(0U, 0);
  SsResultHistoryRow accepted = history_row(1U, 1);
  SsResultAcceptedState state = accepted_state();
  SsResultFieldView field = displacement_view(data, 17U);
  SsResultQuantityDeclaration declaration = quantity_declaration(&field);
  SsResultWriterSummary summary;
  CHECK(make_path(path, arguments->artifact_root, "history-result"));
  spec = writer_spec(path, communicator);
  incomplete.complete = 0;
  CHECK(ss_result_writer_create(&spec, &writer) == SS_OK);
  CHECK(ss_result_writer_stage(writer) == SS_OK);
  CHECK(ss_result_writer_declare_quantity(writer, &declaration) == SS_OK);
  CHECK(ss_result_writer_begin_stream(writer) == SS_OK);
  CHECK(ss_result_writer_append_history(writer, &incomplete) == SS_ERROR_INVALID_STATE);
  CHECK(ss_result_writer_append_history(writer, &complete) == SS_OK);
  state.history_end = 2U;
  CHECK(ss_result_writer_accept_state(writer, &state) == SS_ERROR_INVALID_STATE);
  CHECK(ss_result_writer_append_history(writer, &accepted) == SS_OK);
  CHECK(ss_result_writer_accept_state(writer, &state) == SS_OK);
  CHECK(ss_result_writer_append_history(writer, &accepted) == SS_ERROR_DUPLICATE);
  CHECK(ss_result_writer_write_field(writer, &field) == SS_OK);
  CHECK(ss_result_writer_terminalize(writer, SS_RESULT_SUCCEEDED, "converged", NULL) ==
        SS_OK);
  CHECK(ss_result_writer_publish(writer) == SS_OK);
  memset(&summary, 0, sizeof(summary));
  CHECK(ss_result_writer_summary(writer, &summary) == SS_OK);
  CHECK(summary.history_row_count == 2U);
  CHECK(summary.accepted_state_count == 1U);
  strcpy(outcome->invariant, summary.logical_digest);
  CHECK(ss_result_writer_destroy(&writer) == SS_OK);
  return 0;
}

static int staging_absent(const char *root, const char *prefix) {
  DIR *directory = opendir(root);
  struct dirent *entry;
  const size_t prefix_length = strlen(prefix);
  if (directory == NULL) {
    return 0;
  }
  while ((entry = readdir(directory)) != NULL) {
    if (strncmp(entry->d_name, prefix, prefix_length) == 0) {
      (void)closedir(directory);
      return 0;
    }
  }
  return closedir(directory) == 0;
}

static int test_atomic(const Arguments *arguments, MPI_Comm communicator,
                       const DistributedData *data, TestOutcome *outcome) {
  SsResultFailurePoint point;
  for (point = SS_RESULT_FAIL_DIRECTORY_CREATE;
       point <= SS_RESULT_FAIL_DIRECTORY_RENAME;
       point = (SsResultFailurePoint)((int)point + 1)) {
    char leaf[64];
    char path[SS_RESULT_PATH_CAPACITY];
    SsResultWriterSummary summary;
    CHECK(snprintf(leaf, sizeof(leaf), "atomic-%02d", (int)point) > 0);
    CHECK(make_path(path, arguments->artifact_root, leaf));
    memset(&summary, 0, sizeof(summary));
    CHECK(write_result(path, communicator, data, SS_RESULT_SUCCEEDED, point,
                       &summary) == SS_ERROR_INJECTED);
    CHECK(ss_result_verify_terminal(communicator, path, &summary) ==
          SS_ERROR_NOT_FOUND);
    if (fixture_rank == 0) {
      CHECK(staging_absent(arguments->artifact_root, ".slope-result-"));
    }
    if (point != SS_RESULT_FAIL_ENVIRONMENT_WRITE) {
      SsCheckpointSpec spec;
      SsCheckpointRestoreSummary checkpoint_summary;
      CHECK(snprintf(leaf, sizeof(leaf), "checkpoint-atomic-%02d", (int)point) > 0);
      CHECK(make_path(path, arguments->artifact_root, leaf));
      spec = checkpoint_spec(path, communicator, data);
      memset(&checkpoint_summary, 0, sizeof(checkpoint_summary));
      CHECK(ss_checkpoint_write_atomic(&spec, point, &checkpoint_summary) ==
            SS_ERROR_INJECTED);
      if (fixture_rank == 0) {
        CHECK(access(path, F_OK) != 0);
        CHECK(errno == ENOENT);
        CHECK(staging_absent(arguments->artifact_root, ".slope-checkpoint-"));
      }
      CHECK(MPI_Barrier(communicator) == MPI_SUCCESS);
    }
  }
  strcpy(outcome->invariant,
         "sha256:ef2bfaf62e6414de8e14c3fd56f0a647838f4925d1916441bf678b8f4f1db2ca");
  outcome->checksum = (double)SS_RESULT_FAIL_DIRECTORY_RENAME;
  return 0;
}

static int test_terminal(const Arguments *arguments, MPI_Comm communicator,
                         const DistributedData *data, TestOutcome *outcome) {
  static const SsResultTerminalStatus statuses[] = {
      SS_RESULT_SUCCEEDED,
      SS_RESULT_FAILED,
      SS_RESULT_CANCELLED,
  };
  size_t index;
  char combined[3U * (SS_SHA256_HEX_CAPACITY + 8U)];
  size_t offset = 0U;
  int communicator_size = 1;
  memset(combined, 0, sizeof(combined));
  CHECK(MPI_Comm_size(communicator, &communicator_size) == MPI_SUCCESS);
  if (communicator_size > 1) {
    char path[SS_RESULT_PATH_CAPACITY];
    SsResultWriterSpec spec;
    SsResultWriter *writer = NULL;
    CHECK(make_path(path, arguments->artifact_root, "terminal-rank-disagree"));
    spec = writer_spec(path, communicator);
    CHECK(ss_result_writer_create(&spec, &writer) == SS_OK);
    CHECK(ss_result_writer_stage(writer) == SS_OK);
    CHECK(ss_result_writer_begin_stream(writer) == SS_OK);
    CHECK(ss_result_writer_terminalize(writer, SS_RESULT_CANCELLED,
                                       fixture_rank == 1 ? "different_cancel_point"
                                                         : "user_cancelled",
                                       NULL) == SS_ERROR_INTEGRITY);
    CHECK(ss_result_writer_destroy(&writer) == SS_OK);
    if (fixture_rank == 0) {
      CHECK(access(path, F_OK) != 0);
      CHECK(errno == ENOENT);
    }
    CHECK(MPI_Barrier(communicator) == MPI_SUCCESS);
  }
  for (index = 0U; index < sizeof(statuses) / sizeof(statuses[0]); ++index) {
    char leaf[64];
    char path[SS_RESULT_PATH_CAPACITY];
    SsResultWriterSummary summary;
    CHECK(snprintf(leaf, sizeof(leaf), "terminal-%s",
                   ss_result_terminal_status_name(statuses[index])) > 0);
    CHECK(make_path(path, arguments->artifact_root, leaf));
    CHECK((statuses[index] == SS_RESULT_SUCCEEDED
               ? write_result(path, communicator, data, statuses[index],
                              SS_RESULT_FAIL_NONE, &summary)
               : write_empty_terminal(path, communicator, statuses[index], &summary)) ==
          SS_OK);
    CHECK(ss_result_verify_terminal(communicator, path, &summary) == SS_OK);
    CHECK(summary.terminal_status == statuses[index]);
    CHECK(summary.result_id[0] != '\0');
    CHECK(summary.field_count == (statuses[index] == SS_RESULT_SUCCEEDED ? 5U : 0U));
    CHECK(summary.accepted_state_count ==
          (statuses[index] == SS_RESULT_SUCCEEDED ? 2U : 0U));
    memcpy(combined + offset, summary.result_id, strlen(summary.result_id));
    offset += strlen(summary.result_id);
  }
  CHECK(ss_sha256_bytes((const unsigned char *)combined, offset,
                        outcome->invariant + 7U) == SS_OK);
  memcpy(outcome->invariant, "sha256:", 7U);
  outcome->checksum = 3.0;
  return 0;
}

static int test_performance(const Arguments *arguments, MPI_Comm communicator,
                            const DistributedData *data, TestOutcome *outcome) {
  char path[SS_RESULT_PATH_CAPACITY];
  SsResultWriterSummary summary;
  struct rusage usage;
  uint64_t local_peak_rss_bytes;
  uint64_t maximum_peak_rss_bytes = 0U;
  double imbalance;
  CHECK(make_path(path, arguments->artifact_root, "io-performance"));
  CHECK(write_result(path, communicator, data, SS_RESULT_SUCCEEDED, SS_RESULT_FAIL_NONE,
                     &summary) == SS_OK);
  CHECK(summary.logical_bytes >=
        data->global_count *
            (UINT64_C(1) + UINT64_C(5) +
             FIXTURE_QUADRATURE_POINTS * FIXTURE_STRESS_COMPONENTS) *
            sizeof(double));
  CHECK(summary.payload_bytes > summary.logical_bytes);
  CHECK(summary.payload_bytes < summary.logical_bytes + UINT64_C(4) * 1024U * 1024U);
  CHECK(summary.minimum_rank_bytes > 0U);
  CHECK(summary.maximum_rank_bytes >= summary.minimum_rank_bytes);
  CHECK(summary.maximum_rank_bytes <= summary.logical_bytes);
  imbalance = (double)summary.maximum_rank_bytes / (double)summary.minimum_rank_bytes;
  CHECK(isfinite(imbalance));
  CHECK(imbalance <= 1.01);
  CHECK(isfinite(summary.pack_seconds) && summary.pack_seconds >= 0.0 &&
        summary.pack_seconds < 60.0);
  CHECK(isfinite(summary.hdf5_seconds) && summary.hdf5_seconds >= 0.0 &&
        summary.hdf5_seconds < 60.0);
  CHECK(isfinite(summary.digest_seconds) && summary.digest_seconds >= 0.0 &&
        summary.digest_seconds < 60.0);
  CHECK(isfinite(summary.publication_seconds) && summary.publication_seconds >= 0.0 &&
        summary.publication_seconds < 60.0);
  CHECK(getrusage(RUSAGE_SELF, &usage) == 0);
  CHECK(usage.ru_maxrss > 0);
  CHECK((uint64_t)usage.ru_maxrss <= UINT64_MAX / UINT64_C(1024));
  local_peak_rss_bytes = (uint64_t)usage.ru_maxrss * UINT64_C(1024);
  CHECK(MPI_Allreduce(&local_peak_rss_bytes, &maximum_peak_rss_bytes, 1, MPI_UINT64_T,
                      MPI_MAX, communicator) == MPI_SUCCESS);
  CHECK(maximum_peak_rss_bytes < UINT64_C(2) * 1024U * 1024U * 1024U);
  strcpy(outcome->invariant, summary.logical_digest);
  outcome->logical_bytes = summary.logical_bytes;
  outcome->payload_bytes = summary.payload_bytes;
  outcome->minimum_rank_bytes = summary.minimum_rank_bytes;
  outcome->maximum_rank_bytes = summary.maximum_rank_bytes;
  outcome->peak_rss_bytes = maximum_peak_rss_bytes;
  outcome->pack_seconds = summary.pack_seconds;
  outcome->hdf5_seconds = summary.hdf5_seconds;
  outcome->digest_seconds = summary.digest_seconds;
  outcome->publication_seconds = summary.publication_seconds;
  outcome->io_imbalance = imbalance;
  outcome->checksum = (double)summary.maximum_rank_bytes;
  return 0;
}

static int dispatch(const Arguments *arguments, MPI_Comm communicator,
                    const DistributedData *data, TestOutcome *outcome) {
  if (strcmp(arguments->test_id, "OUT-UNIT-STATE-001") == 0) {
    return test_state(arguments, communicator, data, outcome);
  }
  if (strcmp(arguments->test_id, "OUT-UNIT-SCHEMA-001") == 0) {
    return test_schema_or_hdf5(arguments, communicator, data, outcome, 1);
  }
  if (strcmp(arguments->test_id, "OUT-UNIT-INDEX-001") == 0) {
    return test_index(communicator, data, outcome);
  }
  if (strcmp(arguments->test_id, "OUT-CONTRACT-HDF5-001") == 0 ||
      strcmp(arguments->test_id, "OUT-CONTRACT-DIGEST-001") == 0) {
    return test_schema_or_hdf5(arguments, communicator, data, outcome, 0);
  }
  if (strcmp(arguments->test_id, "OUT-CONTRACT-ATOMIC-001") == 0) {
    return test_atomic(arguments, communicator, data, outcome);
  }
  if (strcmp(arguments->test_id, "OUT-CONTRACT-TERMINAL-001") == 0) {
    return test_terminal(arguments, communicator, data, outcome);
  }
  if (strcmp(arguments->test_id, "OUT-UNIT-HISTORY-001") == 0) {
    return test_history(arguments, communicator, data, outcome);
  }
  if (strcmp(arguments->test_id, "OUT-UNIT-CHECKPOINT-001") == 0) {
    return test_checkpoint_or_restart(arguments, communicator, data, outcome, 0);
  }
  if (strcmp(arguments->test_id, "OUT-NEG-COMPAT-001") == 0) {
    return test_compatibility(arguments, communicator, data, outcome);
  }
  if (strcmp(arguments->test_id, "OUT-CONTRACT-RESTART-001") == 0 ||
      strcmp(arguments->test_id, "OUT-CONTRACT-REPART-001") == 0 ||
      strcmp(arguments->test_id, "OUT-CONTRACT-DEFL-001") == 0) {
    return test_checkpoint_or_restart(arguments, communicator, data, outcome, 1);
  }
  if (strcmp(arguments->test_id, "OUT-PERF-IO-001") == 0) {
    return test_performance(arguments, communicator, data, outcome);
  }
  fprintf(stderr, "unknown result or checkpoint test: %s\n", arguments->test_id);
  return 2;
}

static int publish_outcome(const Arguments *arguments, MPI_Comm communicator,
                           const TestOutcome *outcome) {
  int rank = 0;
  FILE *stream;
  CHECK(MPI_Comm_rank(communicator, &rank) == MPI_SUCCESS);
  if (rank != 0) {
    return 0;
  }
  stream = fopen(arguments->output, "wb");
  if (stream == NULL) {
    return 1;
  }
  if (fprintf(stream,
              "{\"test_id\":\"%s\",\"ranks\":%d,\"requested_threads\":%d,"
              "\"observed_threads\":%d,\"invariant\":\"%s\","
              "\"maximum_error\":%.17g,\"checksum\":%.17g,"
              "\"logical_bytes\":%llu,\"payload_bytes\":%llu,"
              "\"minimum_rank_bytes\":%llu,\"maximum_rank_bytes\":%llu,"
              "\"peak_rss_bytes\":%llu,\"pack_seconds\":%.17g,"
              "\"hdf5_seconds\":%.17g,\"digest_seconds\":%.17g,"
              "\"publication_seconds\":%.17g,\"io_imbalance\":%.17g,"
              "\"payload_sha256\":\"%s\",\"physical_layout\":\"%s\"}\n",
              arguments->test_id, arguments->expected_ranks,
              arguments->expected_threads, fixture_observed_threads, outcome->invariant,
              outcome->maximum_error, outcome->checksum,
              (unsigned long long)outcome->logical_bytes,
              (unsigned long long)outcome->payload_bytes,
              (unsigned long long)outcome->minimum_rank_bytes,
              (unsigned long long)outcome->maximum_rank_bytes,
              (unsigned long long)outcome->peak_rss_bytes, outcome->pack_seconds,
              outcome->hdf5_seconds, outcome->digest_seconds,
              outcome->publication_seconds, outcome->io_imbalance,
              outcome->payload_sha256, outcome->physical_layout) < 0 ||
      fclose(stream) != 0) {
    return 1;
  }
  return 0;
}

int main(int argc, char **argv) {
  Arguments arguments;
  DistributedData data;
  TestOutcome outcome;
  MPI_Comm communicator;
  int size = 0;
  int result = 0;
  int global_result = 0;
  int local_observed = 1;
  int minimum_observed = 1;
  int maximum_observed = 1;
  uint64_t global_count = 12U;
  if (!parse_arguments(argc, argv, &arguments)) {
    fprintf(stderr,
            "usage: %s --test-id ID --artifact-root DIR --output FILE "
            "--expected-ranks N --expected-threads N\n",
            argv[0]);
    return 2;
  }
  if (PetscInitialize(&argc, &argv, NULL, NULL) != PETSC_SUCCESS) {
    return 2;
  }
  communicator = PETSC_COMM_WORLD;
  (void)MPI_Comm_rank(communicator, &fixture_rank);
  (void)MPI_Comm_size(communicator, &size);
  if (size != arguments.expected_ranks) {
    result = 2;
  }
  if (strcmp(arguments.test_id, "OUT-CONTRACT-HDF5-001") == 0) {
    global_count = 3U;
  }
  distributed_data_init(&data, fixture_rank, size, global_count);
  if (!observe_thread_team(&data, &local_observed) ||
      MPI_Allreduce(&local_observed, &minimum_observed, 1, MPI_INT, MPI_MIN,
                    communicator) != MPI_SUCCESS ||
      MPI_Allreduce(&local_observed, &maximum_observed, 1, MPI_INT, MPI_MAX,
                    communicator) != MPI_SUCCESS ||
      minimum_observed != maximum_observed ||
      minimum_observed != arguments.expected_threads) {
    result = 2;
  }
  fixture_observed_threads = maximum_observed;
  memset(&outcome, 0, sizeof(outcome));
  strcpy(outcome.invariant,
         "sha256:0000000000000000000000000000000000000000000000000000000000000000");
  strcpy(outcome.physical_layout,
         fixture_dataset_chunk_rows() == 0U ? "chunked_default" : "chunked_requested");
  if (result == 0) {
    result = dispatch(&arguments, communicator, &data, &outcome);
  }
  if (MPI_Allreduce(&result, &global_result, 1, MPI_INT, MPI_MAX, communicator) !=
      MPI_SUCCESS) {
    global_result = 2;
  }
  if (global_result == 0) {
    global_result = publish_outcome(&arguments, communicator, &outcome);
  }
  if (PetscFinalize() != PETSC_SUCCESS && global_result == 0) {
    global_result = 2;
  }
  return global_result;
}
