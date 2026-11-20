/* Compiled-v1 manifest streaming, identity and empty membership slices.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/io/input.h"

#include <fcntl.h>
#include <petscsys.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char document[] =
    "{\"boundary_rules\":{\"a\":{\"entity_ids\":[\"0000000000000002\","
    "\"ffffffffffffffff\"]}},\"id\":\"example\","
    "\"materials\":{\"a\":{\"entity_ids\":[\"0000000000000005\"]}},"
    "\"supports\":{\"a\":{\"entity_ids\":[]}}}";

static const char projection[] =
    "{\"boundary_rules\":{\"a\":{\"entity_ids\":[\"0000000000000002\","
    "\"ffffffffffffffff\"]}},\"materials\":{\"a\":{\"entity_ids\":["
    "\"0000000000000005\"]}},"
    "\"supports\":{\"a\":{\"entity_ids\":[]}}}";

static const char compact[] =
    "{\"boundary_rules\":{\"a\":{\"entity_ids\":[]}},"
    "\"id\":\"example\",\"materials\":{\"a\":{\"entity_ids\":[]}},"
    "\"supports\":{\"a\":{\"entity_ids\":[]}}}";

static int ss_check_ids(const SsCompiledManifest *manifest, int rank, int ranks) {
  SsEntityId *ids = NULL;
  size_t count = 0U;
  uint64_t global = 0U;
  int failed = 0;
  if (ss_io_read_manifest_ids(manifest, SS_MANIFEST_BOUNDARY_IDS, 0U, rank, ranks, &ids,
                              &count, &global) != SS_OK ||
      global != 2U || count != (rank < 2 ? 1U : 0U) ||
      (count != 0U && ids[0] != (rank == 0 ? 2U : UINT64_MAX))) {
    failed = 1;
  }
  free(ids);
  if (ss_io_read_manifest_ids(manifest, SS_MANIFEST_MATERIAL_IDS, 0U, rank, ranks, &ids,
                              &count, &global) != SS_OK ||
      global != 1U || count != (rank == 0 ? 1U : 0U) || (count != 0U && ids[0] != 5U)) {
    failed = 1;
  }
  free(ids);
  return failed;
}

int main(int argc, char **argv) {
  char path[] = "input-manifest-test-XXXXXX";
  char digest[SS_SHA256_HEX_CAPACITY];
  SsCompiledManifest manifest = {.descriptor = -1};
  SsErrorRecord error = {0};
  int rank = 0, ranks = 0, failed = 0, global_failed = 0;
  if (PetscInitialize(&argc, &argv, NULL, NULL) != PETSC_SUCCESS) {
    return 1;
  }
  (void)MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  (void)MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
  if (rank == 0) {
    const int fd = mkstemp(path);
    if (fd < 0) {
      failed = 1;
    } else {
      failed = write(fd, document, sizeof(document) - 1U) !=
               (ssize_t)(sizeof(document) - 1U);
      failed |= close(fd) != 0;
    }
  }
  (void)MPI_Bcast(&failed, 1, MPI_INT, 0, PETSC_COMM_WORLD);
  (void)MPI_Bcast(path, (int)sizeof(path), MPI_CHAR, 0, PETSC_COMM_WORLD);
  if (failed == 0) {
    if (ss_io_read_compiled_manifest(PETSC_COMM_WORLD, path, &manifest, &error) !=
            SS_OK ||
        ss_sha256_bytes((const unsigned char *)projection, sizeof(projection) - 1U,
                        digest) != SS_OK ||
        strcmp(digest, manifest.digest) != 0 || manifest.size != sizeof(compact) - 1U ||
        memcmp(manifest.data, compact, manifest.size) != 0) {
      failed = 1;
    } else {
      failed |= ss_check_ids(&manifest, rank, ranks);
      failed |= ss_io_require_input_identity(path, &manifest.identity) != SS_OK;
    }
    (void)MPI_Barrier(PETSC_COMM_WORLD);
    if (rank == 0) {
      const int fd = open(path, O_WRONLY | O_APPEND);
      if (fd < 0) {
        failed = 1;
      } else {
        failed |= write(fd, " ", 1) != 1;
        failed |= close(fd) != 0;
      }
    }
    (void)MPI_Barrier(PETSC_COMM_WORLD);
    failed |=
        ss_io_require_input_identity(path, &manifest.identity) != SS_ERROR_INTEGRITY;
    ss_io_compiled_manifest_destroy(&manifest);
    failed |= ss_io_read_compiled_manifest(PETSC_COMM_WORLD, path, &manifest, &error) !=
              SS_ERROR_INTEGRITY;
  }
  ss_io_compiled_manifest_destroy(&manifest);
  (void)MPI_Barrier(PETSC_COMM_WORLD);
  if (rank == 0) {
    failed |= unlink(path) != 0;
  }
  (void)MPI_Allreduce(&failed, &global_failed, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD);
  return PetscFinalize() == PETSC_SUCCESS && global_failed == 0 ? 0 : 1;
}
