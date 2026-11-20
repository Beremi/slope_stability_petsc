/* Accepted-state stream: accepted mechanics states in a staged parallel HDF5 file.
 *
 * With accepted-state output, every accepted state is written collectively
 * to `/states` with shape [accepted, global free DOFs]; each rank writes its
 * contiguous PETSc ownership range.  Publication reads the states back one at
 * a time.  Live memory is independent of the number of accepted states and
 * the stream lives on the result filesystem instead of an anonymous
 * temporary file.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "solver_internal.h"

#include <hdf5.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static SsStatus ss_solver_spool_close_id(hid_t *identifier, herr_t (*closer)(hid_t),
                                         SsStatus status) {
  if (identifier != NULL && *identifier >= 0) {
    if (closer(*identifier) < 0 && status == SS_OK) {
      status = SS_ERROR_HDF5;
    }
    *identifier = -1;
  }
  return status;
}

static SsStatus ss_solver_state_spool_select(const SsSolverStateSpool *spool,
                                             uint64_t state, hid_t file_space,
                                             hid_t *memory_space) {
  hsize_t start[2] = {(hsize_t)state, (hsize_t)spool->local_offset};
  hsize_t count[2] = {1U, (hsize_t)spool->local_size};
  hsize_t memory_dimensions[1] = {spool->local_size == 0 ? 1U
                                                         : (hsize_t)spool->local_size};
  *memory_space = H5Screate_simple(1, memory_dimensions, NULL);
  if (*memory_space < 0) {
    return SS_ERROR_HDF5;
  }
  if (spool->local_size == 0) {
    return H5Sselect_none(file_space) < 0 || H5Sselect_none(*memory_space) < 0
               ? SS_ERROR_HDF5
               : SS_OK;
  }
  return H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL) < 0
             ? SS_ERROR_HDF5
             : SS_OK;
}

SsStatus ss_solver_state_spool_open(SsSolverStateSpool *spool, Vec layout,
                                    const char *path) {
  MPI_Comm communicator = MPI_COMM_NULL;
  hid_t access = -1;
  hid_t space = -1;
  hid_t creation = -1;
  PetscInt begin = 0;
  PetscInt end = 0;
  PetscInt global = 0;
  hsize_t dimensions[2] = {0U, 0U};
  hsize_t maximum[2] = {H5S_UNLIMITED, 0U};
  hsize_t chunk[2] = {1U, 0U};
  SsStatus status = SS_OK;
  if (spool == NULL || layout == NULL || path == NULL ||
      PetscObjectGetComm((PetscObject)layout, &communicator) != PETSC_SUCCESS ||
      VecGetOwnershipRange(layout, &begin, &end) != PETSC_SUCCESS ||
      VecGetSize(layout, &global) != PETSC_SUCCESS || global <= 0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(spool, 0, sizeof(*spool));
  spool->file = -1;
  spool->dataset = -1;
  spool->communicator = communicator;
  spool->local_offset = begin;
  spool->local_size = end - begin;
  spool->global_size = global;
  if ((size_t)snprintf(spool->path, sizeof(spool->path), "%s", path) >=
      sizeof(spool->path)) {
    return SS_ERROR_LIMIT;
  }
  dimensions[1] = (hsize_t)global;
  maximum[1] = (hsize_t)global;
  /* One state per chunk row, capped at 8 MiB per chunk. */
  chunk[1] =
      (hsize_t)global < ((hsize_t)1U << 20U) ? (hsize_t)global : ((hsize_t)1U << 20U);
  access = H5Pcreate(H5P_FILE_ACCESS);
  if (access < 0 || H5Pset_fapl_mpio(access, communicator, MPI_INFO_NULL) < 0) {
    status = SS_ERROR_HDF5;
  } else {
    spool->file = H5Fcreate(spool->path, H5F_ACC_TRUNC, H5P_DEFAULT, access);
    space = H5Screate_simple(2, dimensions, maximum);
    creation = H5Pcreate(H5P_DATASET_CREATE);
    if (spool->file < 0 || space < 0 || creation < 0 ||
        H5Pset_chunk(creation, 2, chunk) < 0) {
      status = SS_ERROR_HDF5;
    } else {
      spool->dataset = H5Dcreate2(spool->file, "/states", H5T_IEEE_F64LE, space,
                                  H5P_DEFAULT, creation, H5P_DEFAULT);
      if (spool->dataset < 0) {
        status = SS_ERROR_HDF5;
      }
    }
  }
  status = ss_solver_spool_close_id(&creation, H5Pclose, status);
  status = ss_solver_spool_close_id(&space, H5Sclose, status);
  status = ss_solver_spool_close_id(&access, H5Pclose, status);
  status = ss_status_allreduce_max(communicator, status);
  if (status != SS_OK) {
    (void)ss_solver_state_spool_close(spool, 1);
    return status;
  }
  spool->open = 1;
  return SS_OK;
}

SsStatus ss_solver_state_spool_append(SsSolverStateSpool *spool, Vec state) {
  const PetscScalar *values = NULL;
  hsize_t extent[2];
  hid_t file_space = -1;
  hid_t memory_space = -1;
  hid_t transfer = -1;
  PetscInt begin = 0;
  PetscInt end = 0;
  SsStatus status = SS_OK;
  if (spool == NULL || spool->open == 0 || state == NULL ||
      VecGetOwnershipRange(state, &begin, &end) != PETSC_SUCCESS ||
      begin != spool->local_offset || end - begin != spool->local_size) {
    return SS_ERROR_INTEGRITY;
  }
  extent[0] = (hsize_t)(spool->count + 1U);
  extent[1] = (hsize_t)spool->global_size;
  if (H5Dset_extent(spool->dataset, extent) < 0) {
    status = SS_ERROR_HDF5;
  }
  file_space = status == SS_OK ? H5Dget_space(spool->dataset) : -1;
  transfer = H5Pcreate(H5P_DATASET_XFER);
  if (file_space < 0 || transfer < 0 ||
      H5Pset_dxpl_mpio(transfer, H5FD_MPIO_COLLECTIVE) < 0) {
    status = SS_ERROR_HDF5;
  }
  if (status == SS_OK) {
    status =
        ss_solver_state_spool_select(spool, spool->count, file_space, &memory_space);
  }
  if (status == SS_OK && VecGetArrayRead(state, &values) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(spool->communicator, status);
  if (status == SS_OK && H5Dwrite(spool->dataset, H5T_NATIVE_DOUBLE, memory_space,
                                  file_space, transfer, values) < 0) {
    status = SS_ERROR_HDF5;
  }
  if (values != NULL && VecRestoreArrayRead(state, &values) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_solver_spool_close_id(&transfer, H5Pclose, status);
  status = ss_solver_spool_close_id(&memory_space, H5Sclose, status);
  status = ss_solver_spool_close_id(&file_space, H5Sclose, status);
  status = ss_status_allreduce_max(spool->communicator, status);
  if (status == SS_OK) {
    ++spool->count;
  }
  return status;
}

SsStatus ss_solver_state_spool_read(SsSolverStateSpool *spool, uint64_t index,
                                    Vec state) {
  PetscScalar *values = NULL;
  hid_t file_space = -1;
  hid_t memory_space = -1;
  hid_t transfer = -1;
  PetscInt begin = 0;
  PetscInt end = 0;
  SsStatus status = SS_OK;
  if (spool == NULL || spool->open == 0 || state == NULL || index >= spool->count ||
      VecGetOwnershipRange(state, &begin, &end) != PETSC_SUCCESS ||
      begin != spool->local_offset || end - begin != spool->local_size) {
    return SS_ERROR_INTEGRITY;
  }
  file_space = H5Dget_space(spool->dataset);
  transfer = H5Pcreate(H5P_DATASET_XFER);
  if (file_space < 0 || transfer < 0 ||
      H5Pset_dxpl_mpio(transfer, H5FD_MPIO_COLLECTIVE) < 0) {
    status = SS_ERROR_HDF5;
  }
  if (status == SS_OK) {
    status = ss_solver_state_spool_select(spool, index, file_space, &memory_space);
  }
  if (status == SS_OK && VecGetArray(state, &values) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(spool->communicator, status);
  if (status == SS_OK && H5Dread(spool->dataset, H5T_NATIVE_DOUBLE, memory_space,
                                 file_space, transfer, values) < 0) {
    status = SS_ERROR_HDF5;
  }
  if (values != NULL && VecRestoreArray(state, &values) != PETSC_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_solver_spool_close_id(&transfer, H5Pclose, status);
  status = ss_solver_spool_close_id(&memory_space, H5Sclose, status);
  status = ss_solver_spool_close_id(&file_space, H5Sclose, status);
  return ss_status_allreduce_max(spool->communicator, status);
}

SsStatus ss_solver_state_spool_close(SsSolverStateSpool *spool, int remove_file) {
  SsStatus status = SS_OK;
  int rank = 0;
  if (spool == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (spool->dataset >= 0) {
    status = ss_solver_spool_close_id(&spool->dataset, H5Dclose, status);
  }
  if (spool->file >= 0) {
    status = ss_solver_spool_close_id(&spool->file, H5Fclose, status);
  }
  if (spool->communicator != MPI_COMM_NULL && remove_file != 0 &&
      spool->path[0] != '\0' &&
      MPI_Comm_rank(spool->communicator, &rank) == MPI_SUCCESS && rank == 0 &&
      unlink(spool->path) != 0 && status == SS_OK) {
    status = SS_ERROR_IO;
  }
  spool->open = 0;
  return status;
}
