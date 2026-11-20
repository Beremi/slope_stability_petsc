/* Compiled-in build identities and dependency feature queries.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/build/features.h"

#include "slope_internal/build/config.h"

const char *ss_build_id(void) { return SS_BUILD_ID; }

const char *ss_dependency_id(void) { return SS_DEPENDENCY_ID; }

const char *ss_source_id(void) { return SS_SOURCE_ID; }

const char *ss_project_version(void) { return SS_PROJECT_VERSION; }

int ss_petsc_index_bits(void) { return SS_PETSC_INDEX_BITS; }

int ss_openmp_enabled(void) { return SLOPE_ENABLE_OPENMP; }

int ss_petsc_openmp_enabled(void) { return SLOPE_PETSC_HAVE_OPENMP; }

int ss_petsc_openmp_kernels_enabled(void) { return SLOPE_PETSC_USE_OPENMP_KERNELS; }

int ss_petsc_kokkos_enabled(void) { return SLOPE_PETSC_HAVE_KOKKOS; }

int ss_petsc_mumps_enabled(void) { return SLOPE_PETSC_HAVE_MUMPS; }

int ss_petsc_hwloc_enabled(void) { return SLOPE_PETSC_HAVE_HWLOC; }

int ss_petsc_openblas_thread_control_enabled(void) {
  return SLOPE_PETSC_HAVE_OPENBLAS_THREAD_CONTROL;
}

int ss_petsc_superlu_dist_enabled(void) { return SLOPE_PETSC_HAVE_SUPERLU_DIST; }

int ss_superlu_dist_openmp_enabled(void) { return SLOPE_SUPERLU_DIST_HAVE_OPENMP; }

int ss_petsc_index_from_u64(uint64_t value, PetscInt *converted) {
  if (converted == NULL || value > (uint64_t)PETSC_MAX_INT) {
    return 1;
  }
  *converted = (PetscInt)value;
  return 0;
}
