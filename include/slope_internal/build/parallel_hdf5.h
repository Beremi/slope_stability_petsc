/** @file
 * @brief Collective probe of the parallel HDF5 (MPI-IO) runtime.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_BUILD_PARALLEL_HDF5_H
#define SS_BUILD_PARALLEL_HDF5_H

#include <mpi.h>

/**
 * Write and read back a small dataset with collective MPI-IO transfers.
 *
 * Every rank writes its own entry of `/zero_owner` into a new file, the file
 * is reopened read-only and checked, and rank zero removes it unless it is to
 * be preserved. Collective on @p communicator; the outcome is agreed, so all
 * ranks return the same value.
 *
 * @param[in] communicator Communicator of the ranks that open the file.
 * @param[in] path File to create; it is truncated if it exists.
 * @param[in] preserve_file Nonzero keeps the file for inspection.
 * @return 0 when the round trip succeeded on every rank, 1 otherwise.
 */
int ss_parallel_hdf5_probe(MPI_Comm communicator, const char *path, int preserve_file);

#endif
