/* Shared fixtures for ss_test_result_checkpoint.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_TEST_RESULT_CHECKPOINT_INTERNAL_H
#define SS_TEST_RESULT_CHECKPOINT_INTERNAL_H

#include "slope_internal/result/result.h"

#include <hdf5.h>
#include <petscsys.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

#define FIXTURE_MAX_LOCAL 32U
#define FIXTURE_QUADRATURE_POINTS 3U
#define FIXTURE_STRESS_COMPONENTS 4U

#define CHECK(expression)                                                              \
  do {                                                                                 \
    if (!(expression)) {                                                               \
      fprintf(stderr, "rank %d %s:%d: assertion failed: %s\n", fixture_rank, __FILE__, \
              __LINE__, #expression);                                                  \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

#define CHECK_NEAR(actual, expected, tolerance)                                        \
  do {                                                                                 \
    const double checked_actual = (actual);                                            \
    const double checked_expected = (expected);                                        \
    if (!isfinite(checked_actual) ||                                                   \
        fabs(checked_actual - checked_expected) > (tolerance)) {                       \
      fprintf(stderr, "rank %d %s:%d: |%.17g - %.17g| > %.3g\n", fixture_rank,         \
              __FILE__, __LINE__, checked_actual, checked_expected,                    \
              (double)(tolerance));                                                    \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

typedef struct {
  const char *test_id;
  const char *artifact_root;
  const char *output;
  int expected_ranks;
  int expected_threads;
  const char *checkpoint_directory;
  const char *checkpoint_mode;
} Arguments;

typedef struct {
  SsEntityId ids[FIXTURE_MAX_LOCAL];
  size_t slots[FIXTURE_MAX_LOCAL];
  double primary[FIXTURE_MAX_LOCAL * 2U];
  double pressure[FIXTURE_MAX_LOCAL];
  double
      stress[FIXTURE_MAX_LOCAL * FIXTURE_QUADRATURE_POINTS * FIXTURE_STRESS_COMPONENTS];
  uint32_t saturation[FIXTURE_MAX_LOCAL * FIXTURE_QUADRATURE_POINTS];
  double plastic_fraction[FIXTURE_MAX_LOCAL];
  SsEntityId deflation_ids[FIXTURE_MAX_LOCAL];
  size_t deflation_slots[FIXTURE_MAX_LOCAL];
  double deflation_left[FIXTURE_MAX_LOCAL * 2U];
  double deflation_right[FIXTURE_MAX_LOCAL * 2U];
  double deflation_source[FIXTURE_MAX_LOCAL * 2U];
  double deflation_source_weights[2U];
  double deflation_source_kinds[2U];
  size_t count;
  uint64_t global_count;
  size_t deflation_count;
  uint64_t deflation_global_count;
} DistributedData;

typedef struct {
  char invariant[SS_SHA256_HEX_CAPACITY + 8U];
  double checksum;
  double maximum_error;
  uint64_t logical_bytes;
  uint64_t payload_bytes;
  uint64_t minimum_rank_bytes;
  uint64_t maximum_rank_bytes;
  uint64_t peak_rss_bytes;
  double pack_seconds;
  double hdf5_seconds;
  double digest_seconds;
  double publication_seconds;
  double io_imbalance;
  char payload_sha256[SS_SHA256_HEX_CAPACITY];
  char physical_layout[16];
} TestOutcome;

extern int fixture_rank;
/** HDF5 chunk rows of the checkpoint fixture. */
uint64_t fixture_dataset_chunk_rows(void);
/** Deterministic field value of a stable identity and component. */
double fixture_value(SsEntityId stable_id, uint32_t component);
/** Join a root and a leaf into a result path; 0 on success. */
int make_path(char output[SS_RESULT_PATH_CAPACITY], const char *root, const char *leaf);
/** Checkpoint specification of the distributed fixture data. */
SsCheckpointSpec checkpoint_spec(const char *path, MPI_Comm communicator,
                                 const DistributedData *data);
/** Case: write a checkpoint or restart from it, by mode. */
int test_checkpoint_or_restart(const Arguments *arguments, MPI_Comm communicator,
                               const DistributedData *data, TestOutcome *outcome,
                               int mode);
/** Case: restore rejects incompatible identities. */
int test_compatibility(const Arguments *arguments, MPI_Comm communicator,
                       const DistributedData *data, TestOutcome *outcome);

#endif
