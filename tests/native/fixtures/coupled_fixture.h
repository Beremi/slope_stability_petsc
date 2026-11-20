/* Compact hydraulic-to-mechanics coupling fixture of the coupling tests.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_TEST_COUPLED_FIXTURE_H
#define SS_TEST_COUPLED_FIXTURE_H

#include "slope_internal/coupling/coupling.h"

#include <mpi.h>

#include <stdint.h>

typedef struct CoupledFixture CoupledFixture;

/** Build the compact hydraulic-to-mechanics coupling fixture; collective, 0 on success.
 */
int coupled_fixture_create(MPI_Comm communicator, uint32_t dimension,
                           int requested_threads, CoupledFixture **fixture);
/** Coupling snapshot of the fixture. */
const SsCouplingFieldSnapshot *coupled_fixture_snapshot(const CoupledFixture *fixture);
/** Release the coupling fixture; collective, 0 on success. */
int coupled_fixture_destroy(CoupledFixture **fixture);

#endif
