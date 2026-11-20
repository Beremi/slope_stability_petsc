/* Shared fixtures for ss_test_mesh_field_mpi.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_TEST_MESH_FIELD_MPI_INTERNAL_H
#define SS_TEST_MESH_FIELD_MPI_INTERNAL_H

#include "slope_internal/assembly/assembly.h"
#include "slope_internal/core/core.h"
#include "slope_internal/fe/fe.h"
#include "slope_internal/mesh/mesh.h"

#include <petscmat.h>
#include <petscsys.h>
#include <petscvec.h>

#include <mpi.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LOCAL_CHECK(expression)                                                        \
  do {                                                                                 \
    if (!(expression)) {                                                               \
      fprintf(stderr, "rank %d: %s:%d: assertion failed: %s\n", rank, __FILE__,        \
              __LINE__, #expression);                                                  \
      local_status = 1;                                                                \
    }                                                                                  \
  } while (0)

typedef struct {
  SsMeshInput input;
  SsEntityId vertex_ids[4];
  double vertex_coordinates[12];
  SsEntityId cell_ids[1];
  uint64_t cells[4];
  SsEntityId facet_ids[4];
  uint64_t facets[12];
  SsMeshIntermediateEntity intermediate_entities[10];
  SsMeshLabelAssignment labels[5];
  SsEntityId coordinate_node_ids[SS_FE_MAX_NODES];
  double coordinate_coordinates[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION];
  uint64_t cell_coordinate_nodes[SS_FE_MAX_NODES];
  size_t owned_vertex_start;
  size_t owned_coordinate_start;
} CompactFixture;

typedef struct {
  SsMeshInput input;
  SsEntityId vertex_ids[4];
  double vertex_coordinates[8];
  SsEntityId cell_ids[2];
  uint64_t cells[6];
  SsEntityId facet_ids[4];
  uint64_t facets[8];
  SsMeshIntermediateEntity intermediate_entities[6];
  SsMeshLabelAssignment labels[6];
  SsEntityId coordinate_node_ids[4];
  double coordinate_coordinates[8];
  uint64_t cell_coordinate_nodes[6];
} StripFixture;

typedef struct {
  SsMeshInput input;
  SsEntityId vertex_ids[5];
  double vertex_coordinates[15];
  SsEntityId cell_ids[2];
  uint64_t cells[8];
  SsEntityId facet_ids[6];
  uint64_t facets[18];
  SsMeshIntermediateEntity intermediate_entities[16];
  SsMeshLabelAssignment labels[8];
  SsEntityId coordinate_node_ids[5];
  double coordinate_coordinates[15];
  uint64_t cell_coordinate_nodes[8];
} TetraPairFixture;

/* Shared-point balancing of every fixture mesh; main() sets it from
 * --shared-point-balance. */
extern SsMeshSharedPointBalance fixture_shared_point_balance;

/** Reference coordinate of a canonical element node. */
double descriptor_node_coordinate(const SsFeDescriptor *descriptor, size_t node,
                                  uint32_t coordinate);
/** Stable identity of a fixture geometry node. */
SsEntityId fixture_coordinate_id(uint32_t dimension, uint32_t coordinate_order,
                                 size_t global_node);
/** Stable identity of a strip-fixture vertex. */
SsEntityId strip_vertex_id(size_t ordinal);
/** Stable identity of a strip-fixture cell. */
SsEntityId strip_cell_id(size_t ordinal);
/** Stable identity of a strip-fixture edge. */
SsEntityId strip_intermediate_id(size_t first, size_t second);
/** Physical coordinate of a fixture geometry node. */
void fixture_coordinate(const SsFeDescriptor *descriptor, size_t global_node,
                        double result[SS_FE_MAX_DIMENSION]);
/** Build this rank's slab of the compact curved fixture; 0 on success. */
int build_fixture(uint32_t dimension, uint32_t coordinate_order, int rank, int size,
                  CompactFixture *fixture);
/** Build this rank's slab of the strip fixture; 0 on success. */
int build_strip_fixture(int rank, int size, StripFixture *fixture);
/** Build this rank's slab of the tetrahedron-pair fixture; 0 on success. */
int build_tetra_pair_fixture(int rank, int size, TetraPairFixture *fixture);

#endif
