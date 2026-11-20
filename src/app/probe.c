/* Entry point of slope-probe: build identity, MPI/OpenMP binding and
 * PETSc index checks used by the launcher.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "slope_internal/app/runtime.h"
#include "slope_internal/build/config.h"
#include "slope_internal/build/features.h"
#include "slope_internal/build/parallel_hdf5.h"
#include "slope_internal/protocol/json.h"
#include "yyjson.h"

#include <hdf5.h>
#include <mpi.h>
#include <petscsys.h>

#if SLOPE_ENABLE_OPENMP
#include <omp.h>
#endif

#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void print_help(void) {
  puts("usage: slope-probe <probe-request.json> <probe-result-dir>");
  puts("       slope-probe --version | --help");
}

static int print_version(void) {
  printf("{\"build_id\":\"%s\",\"dependency_id\":\"%s\","
         "\"petsc_int_bits\":%d,\"project\":\"slope-stability-probe\","
         "\"source_id\":\"%s\",\"version\":\"%s\"}\n",
         ss_build_id(), ss_dependency_id(), ss_petsc_index_bits(), ss_source_id(),
         ss_project_version());
  return 0;
}

static int identity_probe(const char *request_path, const char *result_directory) {
  struct stat metadata;
  FILE *stream;
  unsigned char *input;
  unsigned char *canonical = NULL;
  size_t canonical_size = 0U;
  SsJsonError json_error = {0};
  yyjson_doc *document;
  yyjson_val *root;
  SsErrorRecord error = {0};
  char body[2048];
  char result_id[72];
  int written;
  int descriptor = open(request_path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0 || fstat(descriptor, &metadata) != 0 ||
      !S_ISREG(metadata.st_mode) || metadata.st_size < 0 || metadata.st_size > 65536) {
    if (descriptor >= 0) {
      (void)close(descriptor);
    }
    return 2;
  }
  input = (unsigned char *)malloc(
      (size_t)metadata.st_size == 0U ? 1U : (size_t)metadata.st_size);
  if (input == NULL) {
    (void)close(descriptor);
    return 70;
  }
  stream = fdopen(descriptor, "rb");
  if (stream == NULL ||
      ((size_t)metadata.st_size != 0U && fread(input, 1U, (size_t)metadata.st_size,
                                               stream) != (size_t)metadata.st_size)) {
    if (stream != NULL) {
      (void)fclose(stream);
    } else {
      (void)close(descriptor);
    }
    free(input);
    return 2;
  }
  if (fclose(stream) != 0 ||
      ss_json_canonicalize(input, (size_t)metadata.st_size, &canonical, &canonical_size,
                           &json_error) != SS_JSON_OK) {
    free(input);
    return 2;
  }
  free(input);
  document = yyjson_read((const char *)canonical, canonical_size, YYJSON_READ_NOFLAG);
  ss_json_free(canonical);
  if (document == NULL || !yyjson_is_obj(yyjson_doc_get_root(document))) {
    yyjson_doc_free(document);
    return 2;
  }
  root = yyjson_doc_get_root(document);
  if (yyjson_obj_size(root) != 2U ||
      !yyjson_equals_str(yyjson_obj_get(root, "kind"),
                         "slope_stability.probe_request") ||
      !yyjson_equals_str(yyjson_obj_get(root, "operation"), "identity")) {
    yyjson_doc_free(document);
    return 2;
  }
  yyjson_doc_free(document);
  if (ss_prepare_result_directory(result_directory, &error) != SS_OK) {
    return 4;
  }
  written = snprintf(
      body, sizeof(body),
      "{\"build_id\":\"%s\",\"dependency_id\":\"%s\","
      "\"kind\":\"slope_stability.probe_result\",\"operation\":\"identity\","
      "\"petsc_int_bits\":%d,\"schema_version\":{\"major\":1,\"minor\":0,"
      "\"patch\":0},\"source_id\":\"%s\"}",
      ss_build_id(), ss_dependency_id(), ss_petsc_index_bits(), ss_source_id());
  if (written < 0 || (size_t)written >= sizeof(body) ||
      ss_publish_content_json(result_directory, "probe.json", body, result_id,
                              &error) != SS_OK) {
    return 4;
  }
  printf("{\"result_id\":\"%s\"}\n", result_id);
  return 0;
}

static int actual_openmp_team(int **cpu_ids) {
  int team = 1;
  int *observed = NULL;
#if SLOPE_ENABLE_OPENMP
  int maximum = omp_get_max_threads();
  if (maximum < 1) {
    maximum = 1;
  }
  observed = (int *)calloc((size_t)maximum, sizeof(*observed));
  if (observed == NULL) {
    *cpu_ids = NULL;
    return 0;
  }
#pragma omp parallel
  {
    const int thread = omp_get_thread_num();
    observed[thread] = sched_getcpu();
#pragma omp single
    team = omp_get_num_threads();
  }
#else
  observed = (int *)calloc(1U, sizeof(*observed));
  if (observed != NULL) {
    observed[0] = sched_getcpu();
  }
#endif
  *cpu_ids = observed;
  return observed == NULL ? 0 : team;
}

static void print_cpu_affinity(const int *teams, const int *cpu_ids, int size,
                               int maximum_team) {
  int process;
  fputs(",\"openmp_cpu_affinity\":[", stdout);
  for (process = 0; process < size; ++process) {
    int thread;
    if (process != 0) {
      fputc(',', stdout);
    }
    fputc('[', stdout);
    for (thread = 0; thread < teams[process]; ++thread) {
      if (thread != 0) {
        fputc(',', stdout);
      }
      printf("%d", cpu_ids[process * maximum_team + thread]);
    }
    fputc(']', stdout);
  }
  fputc(']', stdout);
}

static int check_index(const char *text) {
  char *end = NULL;
  unsigned long long parsed;
  PetscInt converted = 0;
  int accepted;
  errno = 0;
  parsed = strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') {
    fputs("{\"accepted\":false,\"reason\":\"invalid_integer\"}\n", stdout);
    return 2;
  }
  accepted = ss_petsc_index_from_u64((uint64_t)parsed, &converted) == 0 ? 1 : 0;
  printf("{\"accepted\":%s,\"petsc_int_bits\":%d,\"value\":\"%s\"}\n",
         accepted != 0 ? "true" : "false", ss_petsc_index_bits(), text);
  return accepted != 0 ? 0 : 6;
}

int main(int argc, char **argv) {
  int provided = MPI_THREAD_SINGLE;
  int rank = -1;
  int size = 0;
  int team;
  int maximum_team = 0;
  int *local_cpu_ids = NULL;
  int *all_cpu_ids = NULL;
  int *all_teams = NULL;
  int status = 0;
  char mpi_library[MPI_MAX_LIBRARY_VERSION_STRING];
  int mpi_library_size = 0;
  unsigned int hdf5_major = 0U;
  unsigned int hdf5_minor = 0U;
  unsigned int hdf5_release = 0U;
  if (argc == 2 && strcmp(argv[1], "--help") == 0) {
    print_help();
    return 0;
  }
  if (argc == 2 && strcmp(argv[1], "--version") == 0) {
    return print_version();
  }
  if (argc == 3 && argv[1][0] != '-') {
    return identity_probe(argv[1], argv[2]);
  }
  if (argc == 3 && strcmp(argv[1], "--check-index") == 0) {
    return check_index(argv[2]);
  }
  if (argc != 3 || strcmp(argv[1], "--hdf5") != 0) {
    print_help();
    return 2;
  }
  if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS ||
      provided < MPI_THREAD_FUNNELED) {
    return 3;
  }
  if (PetscInitializeNoArguments() != PETSC_SUCCESS) {
    MPI_Abort(MPI_COMM_WORLD, 4);
    return 4;
  }
  if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS ||
      MPI_Get_library_version(mpi_library, &mpi_library_size) != MPI_SUCCESS ||
      H5get_libversion(&hdf5_major, &hdf5_minor, &hdf5_release) < 0) {
    status = 1;
  }
  team = actual_openmp_team(&local_cpu_ids);
  if (team <= 0) {
    status = 1;
  }
  {
    int collective_status = 0;
    if (MPI_Allreduce(&status, &collective_status, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD) != MPI_SUCCESS ||
        MPI_Allreduce(&team, &maximum_team, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD) !=
            MPI_SUCCESS) {
      status = 1;
    } else {
      status = collective_status;
    }
  }
  if (status == 0 && maximum_team > team) {
    int *expanded =
        (int *)realloc(local_cpu_ids, (size_t)maximum_team * sizeof(*local_cpu_ids));
    int thread;
    if (expanded == NULL) {
      status = 1;
    } else {
      local_cpu_ids = expanded;
      for (thread = team; thread < maximum_team; ++thread) {
        local_cpu_ids[thread] = -1;
      }
    }
  }
  {
    int collective_status = 0;
    if (MPI_Allreduce(&status, &collective_status, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD) != MPI_SUCCESS) {
      status = 1;
    } else {
      status = collective_status;
    }
  }
  if (rank == 0 && status == 0) {
    all_teams = (int *)calloc((size_t)size, sizeof(*all_teams));
    all_cpu_ids =
        (int *)calloc((size_t)size * (size_t)maximum_team, sizeof(*all_cpu_ids));
    if (all_teams == NULL || all_cpu_ids == NULL) {
      status = 1;
    }
  }
  if (MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
    status = 1;
  }
  if (status == 0 &&
      (MPI_Gather(&team, 1, MPI_INT, all_teams, 1, MPI_INT, 0, MPI_COMM_WORLD) !=
           MPI_SUCCESS ||
       MPI_Gather(local_cpu_ids, maximum_team, MPI_INT, all_cpu_ids, maximum_team,
                  MPI_INT, 0, MPI_COMM_WORLD) != MPI_SUCCESS)) {
    status = 1;
  }
  if (status == 0) {
    status = ss_parallel_hdf5_probe(MPI_COMM_WORLD, argv[2], 0);
  }
  if (rank == 0) {
    printf("{\"build_id\":\"%s\",\"hdf5\":\"%u.%u.%u\","
           "\"hdf5_collective\":%s,\"mpi_library_size\":%d,"
           "\"mpi_ranks\":%d,\"mpi_thread_provided\":%d,"
           "\"mpi_thread_required\":%d,\"openmp_enabled\":%s,"
           "\"openmp_team\":%d,\"petsc_int_bits\":%d,"
           "\"petsc_hwloc\":%s,\"petsc_kokkos\":%s,\"petsc_mumps\":%s,"
           "\"petsc_openblas_thread_control\":%s,"
           "\"petsc_openmp\":%s,\"petsc_openmp_kernels\":%s,"
           "\"petsc_superlu_dist\":%s,\"superlu_dist_openmp\":%s",
           ss_build_id(), hdf5_major, hdf5_minor, hdf5_release,
           status == 0 ? "true" : "false", mpi_library_size, size, provided,
           MPI_THREAD_FUNNELED, ss_openmp_enabled() != 0 ? "true" : "false", team,
           ss_petsc_index_bits(), ss_petsc_hwloc_enabled() != 0 ? "true" : "false",
           ss_petsc_kokkos_enabled() != 0 ? "true" : "false",
           ss_petsc_mumps_enabled() != 0 ? "true" : "false",
           ss_petsc_openblas_thread_control_enabled() != 0 ? "true" : "false",
           ss_petsc_openmp_enabled() != 0 ? "true" : "false",
           ss_petsc_openmp_kernels_enabled() != 0 ? "true" : "false",
           ss_petsc_superlu_dist_enabled() != 0 ? "true" : "false",
           ss_superlu_dist_openmp_enabled() != 0 ? "true" : "false");
    print_cpu_affinity(all_teams, all_cpu_ids, size, maximum_team);
    fputs("}\n", stdout);
  }
  free(all_teams);
  free(all_cpu_ids);
  free(local_cpu_ids);
  if (PetscFinalize() != PETSC_SUCCESS) {
    status = 1;
  }
  if (MPI_Finalize() != MPI_SUCCESS) {
    status = 1;
  }
  return status == 0 ? 0 : 5;
}
