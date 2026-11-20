/* Assembly MPI fixture topology and affinity checks.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "assembly_mpi_internal.h"

static int topology_value(int cpu, const char *name, int *value) {
  char path[128];
  FILE *stream;
  int parsed;
  if (cpu < 0 || name == NULL || value == NULL ||
      snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/%s", cpu,
               name) < 0) {
    return 0;
  }
  stream = fopen(path, "r");
  if (stream == NULL) {
    return 0;
  }
  parsed = fscanf(stream, "%d", value);
  (void)fclose(stream);
  return parsed == 1 ? 1 : 0;
}

int validate_global_affinity(MPI_Comm communicator,
                             const SsAssemblyDiagnostics *diagnostics, int size,
                             int requested_threads, uint64_t *global_cpu_count,
                             uint64_t *global_core_count) {
  char *all_lists = NULL;
  int *cpus = NULL;
  size_t cpu_count = 0U;
  int result = 1;
  int source_rank;
  all_lists = (char *)calloc((size_t)size, (size_t)SS_ERROR_MESSAGE_CAPACITY);
  cpus = (int *)calloc((size_t)size * (size_t)requested_threads, sizeof(*cpus));
  if (all_lists == NULL || cpus == NULL ||
      MPI_Allgather(diagnostics->worker_cpus, (int)SS_ERROR_MESSAGE_CAPACITY, MPI_CHAR,
                    all_lists, (int)SS_ERROR_MESSAGE_CAPACITY, MPI_CHAR,
                    communicator) != MPI_SUCCESS) {
    goto cleanup;
  }
  for (source_rank = 0; source_rank < size; ++source_rank) {
    const char *cursor =
        all_lists + (size_t)source_rank * (size_t)SS_ERROR_MESSAGE_CAPACITY;
    size_t parsed_for_rank = 0U;
    while (*cursor != '\0') {
      char *end = NULL;
      long parsed = strtol(cursor, &end, 10);
      size_t prior;
      if (end == cursor || parsed < 0L || parsed > 1048576L ||
          cpu_count >= (size_t)size * (size_t)requested_threads) {
        goto cleanup;
      }
      for (prior = 0U; prior < cpu_count; ++prior) {
        if (cpus[prior] == (int)parsed) {
          goto cleanup;
        }
      }
      cpus[cpu_count++] = (int)parsed;
      ++parsed_for_rank;
      cursor = end;
      if (*cursor == ',') {
        ++cursor;
      } else if (*cursor != '\0') {
        goto cleanup;
      }
    }
    if (parsed_for_rank != (size_t)requested_threads) {
      goto cleanup;
    }
  }
  if (cpu_count != (size_t)size * (size_t)requested_threads) {
    goto cleanup;
  }
  {
    size_t cpu;
    uint64_t core_count = 0U;
    for (cpu = 0U; cpu < cpu_count; ++cpu) {
      int package = 0;
      int core = 0;
      size_t prior;
      int seen = 0;
      if (topology_value(cpus[cpu], "physical_package_id", &package) == 0 ||
          topology_value(cpus[cpu], "core_id", &core) == 0) {
        goto cleanup;
      }
      for (prior = 0U; prior < cpu; ++prior) {
        int prior_package = 0;
        int prior_core = 0;
        if (topology_value(cpus[prior], "physical_package_id", &prior_package) != 0 &&
            topology_value(cpus[prior], "core_id", &prior_core) != 0 &&
            package == prior_package && core == prior_core) {
          seen = 1;
          break;
        }
      }
      if (seen == 0) {
        ++core_count;
      }
    }
    if (core_count != (uint64_t)size * (uint64_t)requested_threads) {
      goto cleanup;
    }
    *global_cpu_count = (uint64_t)cpu_count;
    *global_core_count = core_count;
  }
  result = 0;

cleanup:
  free(all_lists);
  free(cpus);
  return result;
}
