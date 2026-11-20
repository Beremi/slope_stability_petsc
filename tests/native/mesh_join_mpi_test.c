/* MPI stable-ID routing contracts for sparse source and resident copies.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/mesh/distribution.h"

#include <petscsys.h>

int main(int argc, char **argv) {
  const SsEntityId queries[] = {UINT64_MAX, 6U, 5U, 0U};
  const SsMeshIdValue source[] = {{6U, 2U}, {5U, 3U}, {UINT64_MAX, 17U}};
  uint32_t values[4] = {0U};
  int rank = 0;
  int failed = 0;
  int any_failed = 0;
  if (PetscInitialize(&argc, &argv, NULL, NULL) != PETSC_SUCCESS) {
    return 1;
  }
  (void)MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (ss_mesh_join_ids(PETSC_COMM_WORLD, rank == 0 ? source : NULL, rank == 0 ? 3U : 0U,
                       queries, 4U, values) != SS_OK ||
      values[0] != 17U || values[1] != 2U || values[2] != 3U || values[3] != 0U) {
    failed = 1;
  }
  if (ss_mesh_join_ids(PETSC_COMM_WORLD, source, 1U, queries + 2U, 2U, values) !=
      SS_ERROR_NOT_FOUND) {
    failed = 1;
  }
  (void)MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD);
  return PetscFinalize() == PETSC_SUCCESS && any_failed == 0 ? 0 : 1;
}
