/* Collective parallel-HDF5 write and read-back in which all ranks but the
 * first two own no data (zero-owner selections).
 *
 * Launched by tests/native/support/hdf5_matrix_runner.py.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/build/parallel_hdf5.h"

#include <mpi.h>

#include <stdio.h>

int main(int argc, char **argv) {
  int provided = MPI_THREAD_SINGLE;
  int status;
  if (argc != 2) {
    fprintf(stderr, "usage: %s OUTPUT.h5\n", argv[0]);
    return 2;
  }
  if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS ||
      provided < MPI_THREAD_FUNNELED) {
    return 3;
  }
  status = ss_parallel_hdf5_probe(MPI_COMM_WORLD, argv[1], 0);
  if (MPI_Finalize() != MPI_SUCCESS) {
    return 4;
  }
  return status == 0 ? 0 : 5;
}
