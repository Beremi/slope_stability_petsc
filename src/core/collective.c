/* Collective reduction for native status codes.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/core/collective.h"

SsStatus ss_status_allreduce_max(MPI_Comm communicator, SsStatus local_status) {
  int local = (int)local_status;
  int global = (int)SS_OK;
  if (MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, communicator) !=
      MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  /* An opaque MPI call can appear to leave its output untouched to analyzers.
   * A successful maximum cannot discard a valid local error. */
  if (local > (int)SS_OK && global == (int)SS_OK) {
    return local_status;
  }
  return (SsStatus)global;
}

SsStatus ss_vec_duplicate_collective(Vec source, Vec *const *destinations,
                                     size_t count) {
  MPI_Comm communicator = PetscObjectComm((PetscObject)source);
  SsStatus status = SS_OK;
  for (size_t i = 0U; status == SS_OK && i < count; ++i) {
    status = ss_status_allreduce_max(
        communicator, VecDuplicate(source, destinations[i]) == PETSC_SUCCESS
                          ? SS_OK
                          : SS_ERROR_OUT_OF_MEMORY);
  }
  return status;
}
