/* Shared fixtures for ss_test_assembly_mpi.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_TEST_ASSEMBLY_MPI_INTERNAL_H
#define SS_TEST_ASSEMBLY_MPI_INTERNAL_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "slope_internal/assembly/assembly.h"
#include "slope_internal/diagnostics/diagnostics.h"

#include <petscmat.h>
#include <petscsys.h>
#include <petscvec.h>

#include <mpi.h>

#include <limits.h>
#include <math.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GRID_NX 8U
#define GRID_NY 4U
#define GRID_VERTICES ((GRID_NX + 1U) * (GRID_NY + 1U))
#define GRID_CELLS (2U * GRID_NX * GRID_NY)
#define GRID_EDGES                                                                     \
  (GRID_NX * (GRID_NY + 1U) + (GRID_NX + 1U) * GRID_NY + GRID_NX * GRID_NY)
#define GRID_BOUNDARY_FACETS (2U * GRID_NX + 2U * GRID_NY)
#define GRID_P2_ROWS (GRID_VERTICES + GRID_EDGES)

#define CHECK(expression)                                                              \
  do {                                                                                 \
    if (!(expression)) {                                                               \
      fprintf(stderr, "rank %d: %s:%d: assertion failed: %s\n", test_rank, __FILE__,   \
              __LINE__, #expression);                                                  \
      local_status = 1;                                                                \
    }                                                                                  \
  } while (0)

typedef struct {
  SsMeshInput input;
  SsEntityId *vertex_ids;
  double *vertex_coordinates;
  SsEntityId *cell_ids;
  uint64_t *cells;
  SsEntityId *facet_ids;
  uint64_t *facets;
  SsMeshIntermediateEntity *intermediate_entities;
  SsMeshLabelAssignment *labels;
  SsEntityId *coordinate_node_ids;
  double *coordinate_coordinates;
  uint64_t *cell_coordinate_nodes;
} GridInput;

typedef struct {
  double conductivity[4];
  double flux_value[2];
  double source;
  double reaction;
} DistributedScalarContext;

typedef struct {
  double offset;
  double derivative;
} DistributedNaturalContext;

typedef struct {
  uint64_t mpi_calls;
  uint64_t petsc_calls;
  uint64_t hdf5_calls;
  uint64_t parallel_calls;
} ServiceObservation;

typedef struct {
  GridInput input;
  SsMesh *mesh;
  SsFieldSpace *space;
  SsAssemblyPlan *plan;
  SsOperator *operator_handle;
  Vec field;
  Vec residual;
  Mat jacobian;
  DistributedScalarContext context;
  DistributedNaturalContext natural_context;
  SsScalarOperatorSpec scalar_spec;
  SsNaturalOperatorSpec natural_spec;
  SsOperatorSpec operator_spec;
  SsOperatorState state;
  SsMeshSummary mesh_summary;
  SsFieldSummary field_summary;
  SsAssemblyPlanSummary plan_summary;
  ServiceObservation observation;
} OperatorFixture;

extern int test_rank;
/** Check that the worker CPUs of all ranks are disjoint whole cores; 0 on success. */
int validate_global_affinity(MPI_Comm communicator,
                             const SsAssemblyDiagnostics *diagnostics, int size,
                             int requested_threads, uint64_t *global_cpu_count,
                             uint64_t *global_core_count);

#endif
