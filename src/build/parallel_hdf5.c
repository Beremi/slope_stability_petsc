/* Collective probe that writes and reads a small parallel HDF5 file.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/build/parallel_hdf5.h"

#include <hdf5.h>

#include <stdio.h>

static int ss_hdf5_close(hid_t identifier, herr_t (*closer)(hid_t),
                         int current_status) {
  if (identifier >= 0 && closer(identifier) < 0 && current_status == 0) {
    return 1;
  }
  return current_status;
}

int ss_parallel_hdf5_probe(MPI_Comm communicator, const char *path, int preserve_file) {
  int rank = -1;
  int size = 0;
  int local_status = 0;
  int global_status = 0;
  int value;
  int values[2] = {-1, -1};
  hsize_t dimensions[1] = {2U};
  hsize_t one[1] = {1U};
  hsize_t start[1];
  hid_t access = -1;
  hid_t file = -1;
  hid_t space = -1;
  hid_t dataset = -1;
  hid_t transfer = -1;
  hid_t memory = -1;
  if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(communicator, &size) != MPI_SUCCESS || size < 1) {
    return 1;
  }
  access = H5Pcreate(H5P_FILE_ACCESS);
  if (access < 0 || H5Pset_fapl_mpio(access, communicator, MPI_INFO_NULL) < 0) {
    local_status = 1;
    goto close_objects;
  }
  file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, access);
  if (file < 0) {
    local_status = 1;
    goto close_objects;
  }
  space = H5Screate_simple(1, dimensions, NULL);
  if (space < 0) {
    local_status = 1;
    goto close_objects;
  }
  dataset = H5Dcreate2(file, "/zero_owner", H5T_NATIVE_INT, space, H5P_DEFAULT,
                       H5P_DEFAULT, H5P_DEFAULT);
  if (dataset < 0) {
    local_status = 1;
    goto close_objects;
  }
  transfer = H5Pcreate(H5P_DATASET_XFER);
  if (transfer < 0 || H5Pset_dxpl_mpio(transfer, H5FD_MPIO_COLLECTIVE) < 0) {
    local_status = 1;
    goto close_objects;
  }
  memory = H5Screate_simple(1, one, NULL);
  if (memory < 0) {
    local_status = 1;
    goto close_objects;
  }
  if (rank < 2) {
    start[0] = (hsize_t)rank;
    value = 100 + rank;
    if (H5Sselect_hyperslab(space, H5S_SELECT_SET, start, NULL, one, NULL) < 0 ||
        H5Dwrite(dataset, H5T_NATIVE_INT, memory, space, transfer, &value) < 0) {
      local_status = 1;
      goto close_objects;
    }
  } else {
    if (H5Sselect_none(space) < 0 || H5Sselect_none(memory) < 0 ||
        H5Dwrite(dataset, H5T_NATIVE_INT, memory, space, transfer, NULL) < 0) {
      local_status = 1;
      goto close_objects;
    }
  }

close_objects:
  local_status = ss_hdf5_close(memory, H5Sclose, local_status);
  local_status = ss_hdf5_close(transfer, H5Pclose, local_status);
  local_status = ss_hdf5_close(dataset, H5Dclose, local_status);
  local_status = ss_hdf5_close(space, H5Sclose, local_status);
  if (file >= 0 && H5Fflush(file, H5F_SCOPE_GLOBAL) < 0) {
    local_status = 1;
  }
  local_status = ss_hdf5_close(file, H5Fclose, local_status);
  local_status = ss_hdf5_close(access, H5Pclose, local_status);
  if (MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX, communicator) !=
          MPI_SUCCESS ||
      global_status != 0) {
    return 1;
  }
  if (MPI_Barrier(communicator) != MPI_SUCCESS) {
    return 1;
  }
  access = H5Pcreate(H5P_FILE_ACCESS);
  if (access < 0 || H5Pset_fapl_mpio(access, communicator, MPI_INFO_NULL) < 0) {
    local_status = 1;
  } else {
    file = H5Fopen(path, H5F_ACC_RDONLY, access);
    dataset = file < 0 ? -1 : H5Dopen2(file, "/zero_owner", H5P_DEFAULT);
    transfer = H5Pcreate(H5P_DATASET_XFER);
    if (file < 0 || dataset < 0 || transfer < 0 ||
        H5Pset_dxpl_mpio(transfer, H5FD_MPIO_COLLECTIVE) < 0 ||
        H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, transfer, values) < 0 ||
        values[0] != 100 || (size > 1 && values[1] != 101) ||
        (size == 1 && values[1] != 0)) {
      local_status = 1;
    }
  }
  local_status = ss_hdf5_close(transfer, H5Pclose, local_status);
  local_status = ss_hdf5_close(dataset, H5Dclose, local_status);
  local_status = ss_hdf5_close(file, H5Fclose, local_status);
  local_status = ss_hdf5_close(access, H5Pclose, local_status);
  if (MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX, communicator) !=
      MPI_SUCCESS) {
    return 1;
  }
  if (MPI_Barrier(communicator) != MPI_SUCCESS) {
    return 1;
  }
  if (rank == 0 && preserve_file == 0 && remove(path) != 0) {
    global_status = 1;
  }
  if (MPI_Bcast(&global_status, 1, MPI_INT, 0, communicator) != MPI_SUCCESS) {
    return 1;
  }
  return global_status;
}
