/* PMPI counts and full-width ordering for collective assembly errors.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "operator_internal.h"

#include <stdio.h>

static int recording;
static int allgathers, reductions, int_max_reductions, broadcasts;

int __wrap_MPI_Allreduce(const void *send, void *receive, int count, MPI_Datatype type,
                         MPI_Op op, MPI_Comm comm);
int __wrap_MPI_Allgather(const void *send, int send_count, MPI_Datatype send_type,
                         void *receive, int receive_count, MPI_Datatype receive_type,
                         MPI_Comm comm);
int __wrap_MPI_Bcast(void *buffer, int count, MPI_Datatype type, int root,
                     MPI_Comm comm);

int __wrap_MPI_Allreduce(const void *send, void *receive, int count, MPI_Datatype type,
                         MPI_Op op, MPI_Comm comm) {
  if (recording) {
    ++reductions;
    int_max_reductions += count == 1 && type == MPI_INT && op == MPI_MAX;
  }
  return PMPI_Allreduce(send, receive, count, type, op, comm);
}

int __wrap_MPI_Allgather(const void *send, int send_count, MPI_Datatype send_type,
                         void *receive, int receive_count, MPI_Datatype receive_type,
                         MPI_Comm comm) {
  allgathers += recording;
  return PMPI_Allgather(send, send_count, send_type, receive, receive_count,
                        receive_type, comm);
}

int __wrap_MPI_Bcast(void *buffer, int count, MPI_Datatype type, int root,
                     MPI_Comm comm) {
  broadcasts += recording;
  return PMPI_Bcast(buffer, count, type, root, comm);
}

static int ss_expect_selected(SsOperator *operation, const SsErrorRecord records[2],
                              int rank, int winner) {
  const SsErrorRecord *expected = &records[winner];
  SsErrorRecord selected = records[rank];
  /* The destination may alias the rank-local input. */
  const SsStatus status =
      ss_operator_collective_validation(operation, &selected, &selected);
  return status == expected->status && selected.status == expected->status &&
         ss_error_compare(&selected, expected) == 0 &&
         strcmp(selected.message, expected->message) == 0;
}

int main(int argc, char **argv) {
  if (PetscInitialize(&argc, &argv, NULL, NULL) != PETSC_SUCCESS) {
    return 2;
  }
  int rank = -1, size = 0, failed = 0, total_failed = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  if (size != 2) {
    (void)PetscFinalize();
    return 2;
  }
  SsAssemblyPlan plan = {.communicator = PETSC_COMM_WORLD};
  SsOperator operation = {.plan = &plan};
  SsErrorRecord selected = {.valid = 1};
  recording = 1;
  const SsStatus success =
      ss_operator_collective_validation(&operation, NULL, &selected);
  recording = 0;
  (void)printf("rank=%d assembly_validation_success allgather=%d allreduce=%d "
               "single_int_max=%d bcast=%d\n",
               rank, allgathers, reductions, int_max_reductions, broadcasts);
  failed |= success != SS_OK || selected.valid != 0 || allgathers != 0 ||
            reductions != 1 || int_max_reductions != 1 || broadcasts != 0;

  const SsErrorKey key = {UINT32_MAX, UINT32_MAX, UINT64_MAX,
                          UINT64_MAX, UINT32_MAX, UINT32_MAX};
  SsErrorRecord records[2] = {{0}, {0}};
  ss_error_set(&records[1], SS_ERROR_NUMERICAL, key, "RANK_1", "only rank 1 failed");
  failed |= !ss_expect_selected(&operation, records, rank, 1);
  ss_error_set(&records[0], SS_ERROR_INTEGRITY, key, "Z_CODE", "rank 0");
  failed |= !ss_expect_selected(&operation, records, rank, 1);
  records[0].key.entity -= 1U;
  failed |= !ss_expect_selected(&operation, records, rank, 0);
  records[0].key = key;
  records[0].key.quadrature -= 1U;
  failed |= !ss_expect_selected(&operation, records, rank, 0);
  records[0].key = key;
  records[0].key.rank -= 1U;
  failed |= !ss_expect_selected(&operation, records, rank, 0);
  ss_error_set(&records[0], SS_ERROR_INTEGRITY, key, "RANK_1", "first rank wins tie");
  failed |= !ss_expect_selected(&operation, records, rank, 0);
  MPI_Allreduce(&failed, &total_failed, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD);
  if (PetscFinalize() != PETSC_SUCCESS) {
    return 2;
  }
  return total_failed;
}
