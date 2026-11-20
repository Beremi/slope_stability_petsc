/* Entry point of slope-engine: command line, MPI and PETSc start-up,
 * install identity, the validate and run operations, terminal publication
 * and the exit status.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/app/lifecycle.h"
#include "slope_internal/app/runtime.h"
#include "slope_internal/app/solver.h"
#include "slope_internal/build/config.h"
#include "slope_internal/build/features.h"
#include "slope_internal/build/parallel_hdf5.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/diagnostics/diagnostics.h"
#include "slope_internal/io/input.h"

#include <mpi.h>
#include <petscsys.h>
#include <sys/utsname.h>

#if SLOPE_ENABLE_OPENMP
#include <omp.h>
#endif

#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef enum { SS_CLI_VALIDATE = 1, SS_CLI_RUN = 2 } SsCliOperation;

typedef struct {
  SsCliOperation operation;
  const char *operation_name;
  const char *problem_directory;
  const char *resolved_run_path;
  const char *result_directory;
  SsSolverRunOptions options;
} SsCli;

typedef struct {
  char machine_id[SS_ID_TEXT_CAPACITY];
  char launcher_id[SS_ID_TEXT_CAPACITY];
  char thread_id[SS_ID_TEXT_CAPACITY];
  char placement_id[SS_ID_TEXT_CAPACITY];
  char identity_graph_id[SS_ID_TEXT_CAPACITY];
  char *rank_ids;
  char *rank_ids_json;
  char *rank_numbers_json;
} SsRuntimeProvenance;

static const char ss_unknown_id[] =
    "sha256:0000000000000000000000000000000000000000000000000000000000000000";

static void print_help(void) {
  puts("usage: slope-engine validate <problem-dir> <resolved-run.json> "
       "<result-dir>");
  puts("       slope-engine run <problem-dir> <resolved-run.json> <result-dir> "
       "[run options]");
  puts("       slope-engine --version | --help");
  puts("run options (operational only; results are unchanged):");
  puts("  --run-directory DIR      journal, spools and checkpoints "
       "(default <result-dir>.run)");
  puts("  --journal PATH           JSON-lines progress journal "
       "(default <run-dir>/journal.jsonl)");
  puts("  --no-journal             do not write the journal");
  puts("  --no-progress            no human-readable progress on stderr");
  puts(
      "  --checkpoint-every N     checkpoint every N accepted states (default 0: off)");
  puts("  --checkpoint-seconds S   checkpoint when S seconds passed (default 0: off)");
  puts("  --checkpoint-keep K      in-run checkpoints retained (default 1)");
  puts("SIGINT/SIGTERM stop at the next safe point and publish a cancelled result "
       "with a restart checkpoint of the last accepted state.");
}

static int ss_parse_unsigned(const char *text, uint64_t *value) {
  char *end = NULL;
  unsigned long long parsed;
  if (text == NULL || text[0] < '0' || text[0] > '9') {
    return 0;
  }
  parsed = strtoull(text, &end, 10);
  if (end == NULL || *end != '\0') {
    return 0;
  }
  *value = (uint64_t)parsed;
  return 1;
}

static int ss_parse_run_options(int argc, char **argv, int first, SsCli *cli) {
  int index;
  cli->options.progress = 1;
  for (index = first; index < argc; ++index) {
    const char *option = argv[index];
    const char *value = index + 1 < argc ? argv[index + 1] : NULL;
    uint64_t number = 0U;
    if (strcmp(option, "--no-progress") == 0) {
      cli->options.progress = 0;
    } else if (strcmp(option, "--no-journal") == 0) {
      cli->options.journal_path = "";
    } else if (value == NULL || value[0] == '\0' || value[0] == '-') {
      return 0;
    } else if (strcmp(option, "--journal") == 0) {
      cli->options.journal_path = value;
      ++index;
    } else if (strcmp(option, "--run-directory") == 0) {
      cli->options.run_directory = value;
      ++index;
    } else if (strcmp(option, "--checkpoint-every") == 0 &&
               ss_parse_unsigned(value, &number)) {
      cli->options.checkpoint_every = number;
      ++index;
    } else if (strcmp(option, "--checkpoint-seconds") == 0 &&
               ss_parse_unsigned(value, &number)) {
      cli->options.checkpoint_seconds = (double)number;
      ++index;
    } else if (strcmp(option, "--checkpoint-keep") == 0 &&
               ss_parse_unsigned(value, &number) && number > 0U && number <= 32U) {
      cli->options.checkpoint_keep = (uint32_t)number;
      ++index;
    } else {
      return 0;
    }
  }
  return 1;
}

static int print_version(const char *argv0) {
  SsInstallIdentity install = {0};
  SsErrorRecord error = {0};
  if (ss_install_identity_load(argv0, &install, &error) == SS_OK) {
    printf("{\"build_id\":\"%s\",\"dependency_id\":\"%s\","
           "\"executable_id\":\"%s\",\"petsc_int_bits\":%d,"
           "\"project\":\"slope-stability\",\"schema_id\":\"%s\","
           "\"source_id\":\"%s\",\"version\":\"%s\"}\n",
           ss_build_id(), ss_dependency_id(), install.executable_id,
           ss_petsc_index_bits(), install.schema_id, ss_source_id(),
           ss_project_version());
  } else {
    printf("{\"build_id\":\"%s\",\"dependency_id\":\"%s\","
           "\"petsc_int_bits\":%d,\"project\":\"slope-stability\","
           "\"source_id\":\"%s\",\"version\":\"%s\"}\n",
           ss_build_id(), ss_dependency_id(), ss_petsc_index_bits(), ss_source_id(),
           ss_project_version());
  }
  return 0;
}

static int ss_parse_cli(int argc, char **argv, SsCli *cli) {
  if (argc < 5 || cli == NULL) {
    return 0;
  }
  memset(cli, 0, sizeof(*cli));
  if (strcmp(argv[1], "validate") == 0) {
    cli->operation = SS_CLI_VALIDATE;
    cli->operation_name = "validate";
  } else if (strcmp(argv[1], "run") == 0) {
    cli->operation = SS_CLI_RUN;
    cli->operation_name = "run";
  } else {
    return 0;
  }
  if (argv[2][0] == '\0' || argv[3][0] == '\0' || argv[4][0] == '\0' ||
      argv[2][0] == '-' || argv[3][0] == '-' || argv[4][0] == '-') {
    return 0;
  }
  cli->problem_directory = argv[2];
  cli->resolved_run_path = argv[3];
  cli->result_directory = argv[4];
  if (cli->operation == SS_CLI_VALIDATE && argc != 5) {
    return 0;
  }
  return ss_parse_run_options(argc, argv, 5, cli);
}

static int ss_actual_openmp_threads(void) {
  int threads = 1;
#if SLOPE_ENABLE_OPENMP
#pragma omp parallel
  {
#pragma omp single
    threads = omp_get_num_threads();
  }
#endif
  return threads;
}

static SsStatus ss_runtime_provenance(MPI_Comm communicator, int rank, int size,
                                      int threads, const SsNativeConfig *config,
                                      const SsInstallIdentity *install,
                                      SsRuntimeProvenance *provenance) {
  struct utsname system_identity;
  char machine_text[512];
  char launcher_text[MPI_MAX_LIBRARY_VERSION_STRING + 64U];
  char rank_text[768];
  char thread_text[128];
  char local_rank_id[SS_ID_TEXT_CAPACITY];
  char *identity_projection = NULL;
  char *rank_cursor;
  char *number_cursor;
  size_t rank_json_capacity;
  size_t number_json_capacity;
  size_t identity_projection_capacity;
  int launcher_size = 0;
  int processor_size = 0;
  char processor[MPI_MAX_PROCESSOR_NAME];
  int cpu = sched_getcpu();
  int index;
  int local_failed = 0;
  int any_failed = 0;
  int rank_written;
  int thread_written;
  char digest[SS_SHA256_HEX_CAPACITY];
  SsStatus local_status = SS_OK;
  SsStatus collective_status;
  SsIdentityInputs graph;
  if (provenance == NULL || config == NULL || install == NULL || size < 1 ||
      size > (int)SS_RESOURCE_RANK_CAPACITY || rank < 0 || rank >= size ||
      threads < 1) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(machine_text, 0, sizeof(machine_text));
  memset(launcher_text, 0, sizeof(launcher_text));
  memset(processor, 0, sizeof(processor));
  if (uname(&system_identity) != 0 ||
      gethostname(machine_text, sizeof(machine_text) - 1U) != 0 ||
      MPI_Get_library_version(launcher_text, &launcher_size) != MPI_SUCCESS ||
      MPI_Get_processor_name(processor, &processor_size) != MPI_SUCCESS) {
    local_failed = 1;
  }
  if (launcher_size < 0 || (size_t)launcher_size >= sizeof(launcher_text) ||
      processor_size < 0 || (size_t)processor_size >= sizeof(processor)) {
    local_failed = 1;
  }
  collective_status = ss_status_allreduce_max(
      communicator, local_failed != 0 ? SS_ERROR_INTERNAL : SS_OK);
  any_failed = collective_status != SS_OK ? 1 : 0;
  if (any_failed != 0) {
    return collective_status;
  }
  if (launcher_size < 0 || (size_t)launcher_size >= sizeof(launcher_text) ||
      processor_size < 0 || (size_t)processor_size >= sizeof(processor)) {
    return SS_ERROR_INTERNAL;
  }
  launcher_text[launcher_size] = '\0';
  processor[processor_size] = '\0';
  {
    size_t used = strlen(machine_text);
    int written =
        snprintf(machine_text + used, sizeof(machine_text) - used, "|%s|%s|%s|%s",
                 system_identity.sysname, system_identity.release,
                 system_identity.version, system_identity.machine);
    if (written < 0 || (size_t)written >= sizeof(machine_text) - used) {
      local_status = SS_ERROR_LIMIT;
    }
  }
  if (local_status == SS_OK && (size_t)launcher_size >= sizeof(launcher_text) - 32U) {
    local_status = SS_ERROR_LIMIT;
  } else if (local_status == SS_OK) {
    int written =
        snprintf(launcher_text + launcher_size,
                 sizeof(launcher_text) - (size_t)launcher_size, "|world=%d", size);
    if (written < 0 ||
        (size_t)written >= sizeof(launcher_text) - (size_t)launcher_size) {
      local_status = SS_ERROR_LIMIT;
    }
  }
  if (local_status == SS_OK) {
    rank_written = snprintf(rank_text, sizeof(rank_text),
                            "processor=%.*s|rank=%d|cpu=%d|threads=%d", processor_size,
                            processor, rank, cpu, threads);
    thread_written = snprintf(
        thread_text, sizeof(thread_text), "requested=%u|actual=%d|funneled=true",
        config->sealed != 0 && config->has_run != 0 ? config->run.openmp_threads
                                                    : (uint32_t)threads,
        threads);
    if (rank_written < 0 || (size_t)rank_written >= sizeof(rank_text) ||
        thread_written < 0 || (size_t)thread_written >= sizeof(thread_text)) {
      local_status = SS_ERROR_LIMIT;
    } else if (ss_identity_from_text("machine", machine_text, provenance->machine_id) !=
                   SS_OK ||
               ss_identity_from_text("launcher", launcher_text,
                                     provenance->launcher_id) != SS_OK ||
               ss_identity_from_text("rank-placement", rank_text, local_rank_id) !=
                   SS_OK ||
               ss_identity_from_text("threads", thread_text, provenance->thread_id) !=
                   SS_OK) {
      local_status = SS_ERROR_INTERNAL;
    }
  }
  collective_status = ss_status_allreduce_max(communicator, local_status);
  if (collective_status != SS_OK) {
    return collective_status;
  }
  provenance->rank_ids = (char *)calloc((size_t)size, SS_ID_TEXT_CAPACITY);
  rank_json_capacity = (size_t)size * (SS_ID_TEXT_CAPACITY + 4U) + 2U;
  number_json_capacity = (size_t)size * 16U + 2U;
  provenance->rank_ids_json = (char *)calloc(rank_json_capacity, 1U);
  provenance->rank_numbers_json = (char *)calloc(number_json_capacity, 1U);
  local_failed = provenance->rank_ids == NULL || provenance->rank_ids_json == NULL ||
                         provenance->rank_numbers_json == NULL
                     ? 1
                     : 0;
  collective_status = ss_status_allreduce_max(
      communicator, local_failed != 0 ? SS_ERROR_OUT_OF_MEMORY : SS_OK);
  if (collective_status != SS_OK) {
    return collective_status;
  }
  if (provenance->rank_ids == NULL || provenance->rank_ids_json == NULL ||
      provenance->rank_numbers_json == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  if (MPI_Allgather(local_rank_id, (int)SS_ID_TEXT_CAPACITY, MPI_CHAR,
                    provenance->rank_ids, (int)SS_ID_TEXT_CAPACITY, MPI_CHAR,
                    communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  rank_cursor = provenance->rank_ids_json;
  number_cursor = provenance->rank_numbers_json;
  *rank_cursor++ = '[';
  *number_cursor++ = '[';
  for (index = 0; index < size; ++index) {
    int number_written;
    size_t rank_remaining =
        rank_json_capacity - (size_t)(rank_cursor - provenance->rank_ids_json);
    size_t number_remaining =
        number_json_capacity - (size_t)(number_cursor - provenance->rank_numbers_json);
    const char *rank_id = provenance->rank_ids + (size_t)index * SS_ID_TEXT_CAPACITY;
    rank_written = snprintf(rank_cursor, rank_remaining, "%s\"%s\"",
                            index == 0 ? "" : ",", rank_id);
    number_written =
        snprintf(number_cursor, number_remaining, "%s%d", index == 0 ? "" : ",", index);
    if (rank_written < 0 || (size_t)rank_written >= rank_remaining ||
        number_written < 0 || (size_t)number_written >= number_remaining) {
      return SS_ERROR_LIMIT;
    }
    rank_cursor += rank_written;
    number_cursor += number_written;
  }
  *rank_cursor++ = ']';
  *rank_cursor = '\0';
  *number_cursor++ = ']';
  *number_cursor = '\0';
  identity_projection_capacity =
      strlen(provenance->rank_ids_json) + 4U * SS_ID_TEXT_CAPACITY + 64U;
  identity_projection = (char *)malloc(identity_projection_capacity);
  collective_status = ss_status_allreduce_max(
      communicator, identity_projection == NULL ? SS_ERROR_OUT_OF_MEMORY : SS_OK);
  if (collective_status != SS_OK) {
    free(identity_projection);
    return collective_status;
  }
  rank_written = snprintf(identity_projection, identity_projection_capacity,
                          "machine=%s\nlauncher=%s\nthreads=%s\nranks=%s\n",
                          provenance->machine_id, provenance->launcher_id,
                          provenance->thread_id, provenance->rank_ids_json);
  if (rank_written < 0 || (size_t)rank_written >= identity_projection_capacity ||
      ss_sha256_bytes((const unsigned char *)identity_projection, (size_t)rank_written,
                      digest) != SS_OK ||
      snprintf(provenance->placement_id, sizeof(provenance->placement_id), "sha256:%s",
               digest) != 71) {
    free(identity_projection);
    return SS_ERROR_LIMIT;
  }
  free(identity_projection);
  graph = (SsIdentityInputs){
      ss_source_id(),
      ss_build_id(),
      install->executable_id,
      config->sealed != 0 ? config->problem.id : ss_unknown_id,
      config->sealed != 0 && config->has_run != 0 ? config->run.id : ss_unknown_id,
      provenance->machine_id,
      provenance->launcher_id,
  };
  local_status = ss_identity_link(&graph, provenance->identity_graph_id);
  return ss_status_allreduce_max(communicator, local_status);
}

static void ss_runtime_provenance_destroy(SsRuntimeProvenance *provenance) {
  if (provenance != NULL) {
    free(provenance->rank_ids);
    free(provenance->rank_ids_json);
    free(provenance->rank_numbers_json);
    memset(provenance, 0, sizeof(*provenance));
  }
}

static int ss_exit_code(SsStatus status) {
  switch (status) {
  case SS_OK:
    return 0;
  case SS_ERROR_JSON:
  case SS_ERROR_SCHEMA:
  case SS_ERROR_LIMIT:
    return 2;
  case SS_ERROR_RESOURCE:
    return 3;
  case SS_ERROR_NUMERICAL:
    return 5;
  case SS_ERROR_INTEGRITY:
  case SS_ERROR_INVALID_PATH:
  case SS_ERROR_NOT_FOUND:
  case SS_ERROR_HDF5:
  case SS_ERROR_IO:
  case SS_ERROR_INVALID_STATE:
    return 4;
  case SS_ERROR_CANCELLED:
    return 6;
  default:
    return 70;
  }
}

static const char *ss_terminal_status(SsStatus status) {
  if (status == SS_OK) {
    return "succeeded";
  }
  if (status == SS_ERROR_CANCELLED) {
    return "cancelled";
  }
  if (status == SS_ERROR_NUMERICAL) {
    return "failed_numerical";
  }
  if (status == SS_ERROR_JSON || status == SS_ERROR_SCHEMA ||
      status == SS_ERROR_LIMIT || status == SS_ERROR_RESOURCE) {
    return "failed_preflight";
  }
  return "failed_setup";
}

static const char *ss_error_category(SsStatus status) {
  if (status == SS_ERROR_JSON || status == SS_ERROR_SCHEMA ||
      status == SS_ERROR_LIMIT) {
    return "schema";
  }
  if (status == SS_ERROR_RESOURCE) {
    return "resource";
  }
  if (status == SS_ERROR_CANCELLED) {
    return "cancellation";
  }
  if (status == SS_ERROR_NUMERICAL) {
    return "numerical";
  }
  return "native_setup";
}

static SsStatus ss_publish_terminal(const SsCli *cli, const SsNativeConfig *config,
                                    const SsInstallIdentity *install,
                                    const SsRuntimeProvenance *provenance,
                                    const SsH5Validation *hdf5, int mpi_ranks,
                                    int openmp_threads, SsStatus terminal_status,
                                    const SsErrorRecord *terminal_error,
                                    int solver_executed, char result_id[72]) {
  char error_body[4096];
  char *events_body = NULL;
  char *result_body = NULL;
  char error_id[72] = "";
  char events_id[72];
  size_t events_capacity;
  size_t result_capacity;
  SsErrorRecord publication_error = {0};
  SsStatus publication_status = SS_OK;
  const char *problem_id = config->sealed != 0 ? config->problem.id : ss_unknown_id;
  const char *run_id =
      config->sealed != 0 && config->has_run != 0 ? config->run.id : ss_unknown_id;
  const char *machine_id =
      provenance->machine_id[0] != '\0' ? provenance->machine_id : ss_unknown_id;
  const char *launcher_id =
      provenance->launcher_id[0] != '\0' ? provenance->launcher_id : ss_unknown_id;
  const char *thread_id =
      provenance->thread_id[0] != '\0' ? provenance->thread_id : ss_unknown_id;
  const char *placement_id =
      provenance->placement_id[0] != '\0' ? provenance->placement_id : ss_unknown_id;
  const char *identity_graph_id = provenance->identity_graph_id[0] != '\0'
                                      ? provenance->identity_graph_id
                                      : ss_unknown_id;
  const char *rank_ids =
      provenance->rank_ids_json != NULL ? provenance->rank_ids_json : "[]";
  const char *rank_numbers =
      provenance->rank_numbers_json != NULL ? provenance->rank_numbers_json : "[]";
  const char *failure_category = ss_error_category(terminal_status);
  const char *failure_phase =
      terminal_status == SS_ERROR_NUMERICAL ? "analysis" : "setup";
  const char *events_name =
      solver_executed != 0 ? "native-terminal-events" : "native-foundation-events";
  const char *result_name =
      solver_executed != 0 ? "native-terminal-result" : "native-foundation-result";
  const char *capability =
      solver_executed != 0 ? "numerical-analysis" : "foundation-validation";
  const char *terminal_name = ss_terminal_status(terminal_status);
  int written;
  if (ss_checked_add_size(strlen(rank_ids), 8192U, &events_capacity) != SS_OK ||
      ss_checked_add_size(strlen(rank_ids), strlen(rank_numbers), &result_capacity) !=
          SS_OK ||
      ss_checked_add_size(result_capacity, 16384U, &result_capacity) != SS_OK) {
    return SS_ERROR_LIMIT;
  }
  events_body = (char *)malloc(events_capacity);
  result_body = (char *)malloc(result_capacity);
  if (events_body == NULL || result_body == NULL) {
    free(events_body);
    free(result_body);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  if (terminal_status != SS_OK) {
    const char *code = terminal_error != NULL && terminal_error->valid != 0
                           ? terminal_error->code
                           : "NATIVE_INTERNAL";
    const char *message = terminal_error != NULL && terminal_error->valid != 0
                              ? terminal_error->message
                              : "native operation failed";
    written =
        snprintf(error_body, sizeof(error_body),
                 "{\"category\":{\"name\":\"%s\"},\"causes\":{\"count\":0},"
                 "\"code\":{\"name\":\"%s\"},\"created_by\":{\"source_id\":\"%s\","
                 "\"tool\":\"slope-engine\",\"tool_version\":\"%s\"},"
                 "\"diagnostic_artifacts\":{\"events\":\"events.json\"},"
                 "\"extensions\":{},\"identity\":{\"name\":\"terminal-error\"},"
                 "\"kind\":\"slope_stability.error\",\"message\":{\"text\":\"%s\"},"
                 "\"operation\":{\"name\":\"%s\"},\"phase\":{\"name\":\"%s\"},"
                 "\"provenance\":{\"build_id\":\"%s\",\"problem_id\":\"%s\","
                 "\"run_id\":\"%s\"},\"schema_version\":{\"major\":1,\"minor\":0,"
                 "\"patch\":0},\"severity\":{\"name\":\"error\"}}",
                 failure_category, code, ss_source_id(), ss_project_version(), message,
                 cli->operation_name, failure_phase, ss_build_id(), problem_id, run_id);
    if (written < 0 || (size_t)written >= sizeof(error_body) ||
        ss_publish_content_json(cli->result_directory, "error.json", error_body,
                                error_id, &publication_error) != SS_OK) {
      publication_status = SS_ERROR_IO;
      goto done;
    }
  }
  written = snprintf(
      events_body, events_capacity,
      "{\"created_by\":{\"source_id\":\"%s\",\"tool\":\"slope-engine\","
      "\"tool_version\":\"%s\"},\"events\":{\"count\":5,\"terminal\":\"%s\"},"
      "\"extensions\":{},\"identity\":{\"name\":\"%s\"},"
      "\"kind\":\"slope_stability.events\",\"provenance\":{\"build_id\":\"%s\","
      "\"identity_graph_id\":\"%s\",\"launcher_id\":\"%s\","
      "\"machine_id\":\"%s\",\"placement_id\":\"%s\","
      "\"runtime\":{\"python_loaded\":false},\"schema_id\":\"%s\"},"
      "\"rank_summary\":{\"placement_ids\":%s,\"ranks\":%d},"
      "\"run\":{\"problem_id\":\"%s\",\"run_id\":\"%s\"},"
      "\"schema_version\":{\"major\":1,\"minor\":0,\"patch\":0},"
      "\"stages\":{\"output\":\"complete\",\"setup\":\"complete\"},"
      "\"thread_summary\":{\"actual\":%d,\"requested\":%u,"
      "\"thread_id\":\"%s\"}}",
      ss_source_id(), ss_project_version(), terminal_name, events_name, ss_build_id(),
      identity_graph_id, launcher_id, machine_id, placement_id, install->schema_id,
      rank_ids, mpi_ranks, problem_id, run_id, openmp_threads,
      config->sealed != 0 && config->has_run != 0 ? config->run.openmp_threads
                                                  : (uint32_t)openmp_threads,
      thread_id);
  if (written < 0 || (size_t)written >= events_capacity ||
      ss_publish_content_json(cli->result_directory, "events.json", events_body,
                              events_id, &publication_error) != SS_OK) {
    publication_status = SS_ERROR_IO;
    goto done;
  }
  written = snprintf(
      result_body, result_capacity,
      "{\"analysis_outcome\":{\"capability\":\"%s\","
      "\"operation\":\"%s\",\"solver_executed\":%s},"
      "\"artifacts\":{\"error_id\":\"%s\",\"events_id\":\"%s\","
      "\"events_path\":\"events.json\",\"hdf5_layout\":\"compiled-problem-v1\"},"
      "\"created_by\":{\"source_id\":\"%s\",\"tool\":\"slope-engine\","
      "\"tool_version\":\"%s\"},\"errors\":{\"count\":%d,\"primary_id\":\"%s\"},"
      "\"extensions\":{},\"identity\":{\"name\":\"%s\"},"
      "\"kind\":\"slope_stability.result\",\"provenance\":{\"build_id\":\"%s\","
      "\"dependency_id\":\"%s\",\"executable_id\":\"%s\","
      "\"identity_graph_id\":\"%s\",\"launcher_id\":\"%s\","
      "\"machine_id\":\"%s\",\"placement\":{\"id\":\"%s\","
      "\"rank_ids\":%s,\"ranks\":%s},\"schema_id\":\"%s\","
      "\"source_id\":\"%s\",\"thread_id\":\"%s\"},"
      "\"run\":{\"problem_id\":\"%s\",\"run_id\":\"%s\"},"
      "\"schema_version\":{\"major\":1,\"minor\":0,\"patch\":0},"
      "\"summary\":{\"hdf5\":{\"cell_rows\":%llu,"
      "\"coordinate_values\":%llu,\"zero_owner_ranks\":%u},"
      "\"resources\":{\"mpi_ranks\":%d,\"openmp_threads\":%d}},"
      "\"terminal_status\":{\"exit_code\":%d,\"status\":\"%s\"}}",
      capability, cli->operation_name, solver_executed != 0 ? "true" : "false",
      terminal_status == SS_OK ? "" : error_id, events_id, ss_source_id(),
      ss_project_version(), terminal_status == SS_OK ? 0 : 1,
      terminal_status == SS_OK ? "" : error_id, result_name, ss_build_id(),
      ss_dependency_id(), install->executable_id, identity_graph_id, launcher_id,
      machine_id, placement_id, rank_ids, rank_numbers, install->schema_id,
      ss_source_id(), thread_id, problem_id, run_id,
      (unsigned long long)(hdf5 == NULL ? 0U : hdf5->cell_rows),
      (unsigned long long)(hdf5 == NULL ? 0U : hdf5->coordinate_values),
      hdf5 == NULL ? 0U : hdf5->zero_owner_ranks, mpi_ranks, openmp_threads,
      ss_exit_code(terminal_status), terminal_name);
  if (written < 0 || (size_t)written >= result_capacity) {
    publication_status = SS_ERROR_LIMIT;
    goto done;
  }
  publication_status = ss_publish_content_json(
      cli->result_directory, "result.json", result_body, result_id, &publication_error);

done:
  free(result_body);
  free(events_body);
  return publication_status;
}

static SsStatus ss_prepare_collective_result_directory(MPI_Comm communicator, int rank,
                                                       const char *path,
                                                       SsErrorRecord *selected_error) {
  SsStatus status = SS_OK;
  SsErrorRecord local_error = {0};
  if (rank == 0) {
    status = ss_prepare_result_directory(path, &local_error);
    local_error.key.rank = 0U;
  }
  if (ss_collective_error_select(communicator, &local_error, selected_error) != SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  if (selected_error->valid != 0) {
    status = selected_error->status;
  }
  if (MPI_Barrier(communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  return status;
}

static SsStatus ss_load_install_collective(MPI_Comm communicator, int rank,
                                           const char *argv0,
                                           SsInstallIdentity *install,
                                           SsErrorRecord *selected_error) {
  SsErrorRecord local_error = {0};
  SsStatus status = ss_install_identity_load(argv0, install, &local_error);
  local_error.key.rank = (uint32_t)rank;
  if (status != SS_OK && local_error.valid == 0) {
    local_error.valid = 1;
    local_error.status = status;
  }
  if (ss_collective_error_select(communicator, &local_error, selected_error) != SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  return selected_error->valid != 0 ? selected_error->status : SS_OK;
}

int main(int argc, char **argv) {
  SsCli cli;
  SsCancellation cancellation;
  SsLifecycle lifecycle;
  SsInstallIdentity install = {0};
  SsNativeConfig config = {0};
  SsRuntimeProvenance provenance = {0};
  SsH5Validation hdf5 = {0};
  SsRuntimeIdentity runtime;
  SsSolverSummary solver_summary = {0};
  SsErrorRecord selected_error = {0};
  SsStatus status = SS_OK;
  int solver_executed = 0;
  int provided = MPI_THREAD_SINGLE;
  int rank = -1;
  int size = 0;
  int local_rank = -1;
  MPI_Comm local_communicator = MPI_COMM_NULL;
  int openmp_threads = 1;
  int petsc_initialized = 0;
  int mpi_initialized = 0;
  int result_directory_ready = 0;
  int exit_code;
  char payload_path[1024];
  char publication_probe[1024];
  char result_id[72] = "";
  if (argc == 2 && strcmp(argv[1], "--help") == 0) {
    print_help();
    return 0;
  }
  if (argc == 2 && strcmp(argv[1], "--version") == 0) {
    return print_version(argv[0]);
  }
  if (!ss_parse_cli(argc, argv, &cli)) {
    print_help();
    return 2;
  }
  ss_cancellation_init(&cancellation);
  ss_lifecycle_init(&lifecycle);
  if (ss_cancellation_install_handlers(&cancellation) != SS_OK ||
      MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
    return 70;
  }
  mpi_initialized = 1;
  if (provided < MPI_THREAD_FUNNELED ||
      MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS) {
    status = SS_ERROR_RESOURCE;
    goto finalize;
  }
  if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                          &local_communicator) != MPI_SUCCESS ||
      MPI_Comm_rank(local_communicator, &local_rank) != MPI_SUCCESS) {
    if (local_communicator != MPI_COMM_NULL) {
      (void)MPI_Comm_free(&local_communicator);
    }
    status = SS_ERROR_RESOURCE;
    goto finalize;
  }
  if (MPI_Comm_free(&local_communicator) != MPI_SUCCESS) {
    status = SS_ERROR_RESOURCE;
    goto finalize;
  }
  if (PetscInitializeNoArguments() != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
    goto finalize;
  }
  petsc_initialized = 1;
  /* Keep the cancellation handlers authoritative after runtime initialization. */
  if (ss_cancellation_install_handlers(&cancellation) != SS_OK) {
    status = SS_ERROR_INTERNAL;
    goto finalize;
  }
  openmp_threads = ss_actual_openmp_threads();
  status = ss_load_install_collective(MPI_COMM_WORLD, rank, argv[0], &install,
                                      &selected_error);
  if (status != SS_OK) {
    goto finalize;
  }
  if (cli.operation == SS_CLI_VALIDATE) {
    status = ss_prepare_collective_result_directory(
        MPI_COMM_WORLD, rank, cli.result_directory, &selected_error);
    if (status != SS_OK) {
      goto finalize;
    }
    result_directory_ready = 1;
  }
  runtime = (SsRuntimeIdentity){
      ss_build_id(),      install.executable_id,
      ss_dependency_id(), (uint32_t)ss_petsc_index_bits(),
      (uint32_t)size,     (uint32_t)openmp_threads,
      (uint32_t)rank,     (uint32_t)local_rank,
  };
  status = ss_io_load_pair_collective(MPI_COMM_WORLD, cli.problem_directory,
                                      cli.resolved_run_path, &runtime, &config,
                                      &selected_error);
  if (status != SS_OK) {
    goto publish;
  }
  status = ss_cancellation_collective_safe_point(MPI_COMM_WORLD, &cancellation,
                                                 SS_SAFEPOINT_SETUP);
  if (status != SS_OK) {
    selected_error = (SsErrorRecord){
        1,
        status,
        {1U, 1U, 0U, 0U, 0U, 0U},
        "CANCELLED_AT_SAFEPOINT",
        "cancellation observed at setup safe point",
    };
    goto publish;
  }
  {
    int payload_written = snprintf(payload_path, sizeof(payload_path), "%s/%s",
                                   cli.problem_directory, config.problem.payload_path);
    int probe_written =
        cli.operation == SS_CLI_VALIDATE
            ? snprintf(publication_probe, sizeof(publication_probe), "%s/%s",
                       cli.result_directory, ".parallel-hdf5-probe.h5")
            : snprintf(publication_probe, sizeof(publication_probe), "%s%s",
                       cli.result_directory, ".parallel-hdf5-probe.h5");
    if (payload_written < 0 || (size_t)payload_written >= sizeof(payload_path) ||
        probe_written < 0 || (size_t)probe_written >= sizeof(publication_probe)) {
      status = SS_ERROR_LIMIT;
      selected_error = (SsErrorRecord){
          1,
          status,
          {1U, 3U, 0U, 0U, 0U, 0U},
          "PACKAGE_PATH_LIMIT",
          "problem or result path exceeds native capacity",
      };
      goto publish;
    }
  }
  status = ss_io_validate_hdf5(MPI_COMM_WORLD, payload_path, &hdf5, &selected_error);
  if (status != SS_OK) {
    goto publish;
  }
  if (ss_parallel_hdf5_probe(MPI_COMM_WORLD, publication_probe, 0) != 0) {
    status = SS_ERROR_HDF5;
    selected_error = (SsErrorRecord){
        1,
        status,
        {4U, 3U, 0U, 0U, 0U, 0U},
        "RESULT_HDF5_PROBE",
        "parallel HDF5 result-location probe failed",
    };
    goto publish;
  }
  status = ss_cancellation_collective_safe_point(MPI_COMM_WORLD, &cancellation,
                                                 SS_SAFEPOINT_HDF5);
  if (status != SS_OK) {
    selected_error = (SsErrorRecord){
        1,
        status,
        {1U, 1U, 0U, 0U, 0U, 0U},
        "CANCELLED_AT_SAFEPOINT",
        "cancellation observed at HDF5 safe point",
    };
    goto publish;
  }
  if (cli.operation == SS_CLI_VALIDATE) {
    const SsSolverRequest solver_request = {
        MPI_COMM_WORLD,
        cli.problem_directory,
        cli.resolved_run_path,
        cli.result_directory,
        &config,
        ss_build_id(),
        openmp_threads,
        &cli.options,
    };
    status = ss_solver_validate(&solver_request, &solver_summary, &selected_error);
    if (status != SS_OK) {
      goto publish;
    }
  }
  status = ss_lifecycle_transition(&lifecycle, SS_ENGINE_VALIDATED);
  if (status == SS_OK && cli.operation == SS_CLI_RUN) {
    const SsSolverRequest solver_request = {
        MPI_COMM_WORLD,
        cli.problem_directory,
        cli.resolved_run_path,
        cli.result_directory,
        &config,
        ss_build_id(),
        openmp_threads,
        &cli.options,
    };
    status = ss_lifecycle_transition(&lifecycle, SS_ENGINE_RUNNING);
    if (status == SS_OK) {
      solver_executed = 1;
      status = ss_solver_run(&solver_request, &solver_summary, &selected_error);
    }
    if (status == SS_OK) {
      status = ss_lifecycle_transition(&lifecycle, SS_ENGINE_SUCCEEDED);
      result_directory_ready = solver_summary.result_published;
      memcpy(result_id, solver_summary.result_id, sizeof(result_id));
      result_id[sizeof(result_id) - 1U] = '\0';
    }
  }

publish:
  {
    SsStatus provenance_status = ss_runtime_provenance(
        MPI_COMM_WORLD, rank, size, openmp_threads, &config, &install, &provenance);
    if (provenance_status != SS_OK && status == SS_OK) {
      status = provenance_status;
      selected_error = (SsErrorRecord){
          1,
          status,
          {4U, 2U, 0U, 0U, 0U, 0U},
          "PROVENANCE_CAPTURE",
          "runtime provenance capture failed",
      };
    }
  }
  if (status == SS_OK) {
    status = ss_cancellation_collective_safe_point(MPI_COMM_WORLD, &cancellation,
                                                   SS_SAFEPOINT_OUTPUT);
    if (status != SS_OK) {
      selected_error = (SsErrorRecord){
          1,
          status,
          {1U, 1U, 0U, 0U, 0U, 0U},
          "CANCELLED_AT_SAFEPOINT",
          "cancellation observed at output safe point",
      };
    }
  }
  if (status == SS_OK && cli.operation == SS_CLI_VALIDATE) {
    status = ss_lifecycle_transition(&lifecycle, SS_ENGINE_SUCCEEDED);
  } else if (status == SS_ERROR_CANCELLED && (lifecycle.state == SS_ENGINE_CREATED ||
                                              lifecycle.state == SS_ENGINE_VALIDATED ||
                                              lifecycle.state == SS_ENGINE_RUNNING)) {
    (void)ss_lifecycle_transition(&lifecycle, SS_ENGINE_CANCELLED);
  }
  if (solver_summary.result_published == 0 && result_directory_ready == 0) {
    SsErrorRecord directory_error = {0};
    SsStatus directory_status = ss_prepare_collective_result_directory(
        MPI_COMM_WORLD, rank, cli.result_directory, &directory_error);
    if (directory_status == SS_OK) {
      result_directory_ready = 1;
    } else {
      status = directory_status;
      selected_error = directory_error;
    }
  }
  if (rank == 0 && result_directory_ready != 0 &&
      solver_summary.result_published == 0) {
    SsStatus publication_status = ss_publish_terminal(
        &cli, &config, &install, &provenance, &hdf5, size, openmp_threads, status,
        status == SS_OK ? NULL : &selected_error, solver_executed, result_id);
    if (publication_status != SS_OK) {
      status = publication_status;
    }
  }
  {
    int status_value = (int)status;
    if (MPI_Bcast(&status_value, 1, MPI_INT, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    } else {
      status = (SsStatus)status_value;
    }
  }
  if (rank == 0 && status == SS_OK) {
    printf("{\"operation\":\"%s\",\"result_id\":\"%s\"}\n", cli.operation_name,
           result_id);
  } else if (rank == 0 && status != SS_OK) {
    fprintf(stderr,
            "{\"category\":\"%s\",\"code\":\"%s\",\"operation\":\"%s\","
            "\"status\":\"%s\"}\n",
            ss_error_category(status),
            selected_error.valid != 0 ? selected_error.code : "NATIVE_INTERNAL",
            cli.operation_name, ss_terminal_status(status));
  }

finalize:
  if (lifecycle.state != SS_ENGINE_DESTROYED) {
    if (lifecycle.state == SS_ENGINE_CREATED ||
        lifecycle.state == SS_ENGINE_VALIDATED ||
        lifecycle.state == SS_ENGINE_RUNNING) {
      (void)ss_lifecycle_transition(&lifecycle, SS_ENGINE_FAILED);
    }
    (void)ss_lifecycle_destroy(&lifecycle);
  }
  ss_runtime_provenance_destroy(&provenance);
  ss_io_config_destroy(&config);
  if (petsc_initialized != 0 && PetscFinalize() != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  if (mpi_initialized != 0 && MPI_Finalize() != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  exit_code = ss_exit_code(status);
  if (result_directory_ready == 0 && status != SS_OK && rank <= 0) {
    fprintf(stderr,
            "{\"category\":\"native_startup\",\"code\":\"%s\","
            "\"operation\":\"%s\",\"status\":\"failed_setup\"}\n",
            selected_error.valid != 0 ? selected_error.code : "NATIVE_STARTUP_FAILED",
            cli.operation_name);
  }
  return exit_code;
}
