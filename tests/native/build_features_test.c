/* Build identity and required PETSc features of the configured build.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/build/features.h"

#include <petscconf.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
  const char *build = ss_build_id();
  if (strncmp(build, "sha256:", 7U) != 0 || strlen(build) != 71U) {
    fputs("invalid build identity\n", stderr);
    return 1;
  }
#ifndef PETSC_USE_REAL_DOUBLE
  fputs("PETSc is not real double\n", stderr);
  return 2;
#endif
#ifndef PETSC_HAVE_HDF5
  fputs("PETSc lacks HDF5\n", stderr);
  return 3;
#endif
#ifndef PETSC_HAVE_HYPRE
  fputs("PETSc lacks HYPRE\n", stderr);
  return 4;
#endif
#ifdef PETSC_USE_64BIT_INDICES
  if (ss_petsc_index_bits() != 64) {
    return 5;
  }
#else
  if (ss_petsc_index_bits() != 32) {
    return 6;
  }
#endif
  {
    PetscInt converted = 0;
    if (ss_petsc_index_from_u64(0U, &converted) != 0 || converted != 0 ||
        ss_petsc_index_from_u64((uint64_t)PETSC_MAX_INT, &converted) != 0 ||
        ss_petsc_index_from_u64((uint64_t)PETSC_MAX_INT + 1U, &converted) == 0 ||
        ss_petsc_index_from_u64(0U, NULL) == 0) {
      return 7;
    }
  }
#ifdef PETSC_HAVE_OPENMP
  if (ss_petsc_openmp_enabled() == 0) {
    return 8;
  }
#else
  if (ss_petsc_openmp_enabled() != 0) {
    return 9;
  }
#endif
#ifdef PETSC_USE_OPENMP_KERNELS
  if (ss_petsc_openmp_kernels_enabled() == 0) {
    return 10;
  }
#else
  if (ss_petsc_openmp_kernels_enabled() != 0) {
    return 11;
  }
#endif
#ifdef PETSC_HAVE_KOKKOS
  if (ss_petsc_kokkos_enabled() == 0) {
    return 12;
  }
#else
  if (ss_petsc_kokkos_enabled() != 0) {
    return 13;
  }
#endif
#ifdef PETSC_HAVE_MUMPS
  if (ss_petsc_mumps_enabled() == 0) {
    return 14;
  }
#else
  if (ss_petsc_mumps_enabled() != 0) {
    return 15;
  }
#endif
#ifdef PETSC_HAVE_HWLOC
  if (ss_petsc_hwloc_enabled() == 0) {
    return 17;
  }
#else
  if (ss_petsc_hwloc_enabled() != 0) {
    return 18;
  }
#endif
#ifdef PETSC_HAVE_OPENBLAS_SET_NUM_THREADS
  if (ss_petsc_openblas_thread_control_enabled() == 0) {
    return 19;
  }
#else
  if (ss_petsc_openblas_thread_control_enabled() != 0) {
    return 20;
  }
#endif
  if (ss_superlu_dist_openmp_enabled() != 0 && ss_petsc_superlu_dist_enabled() == 0) {
    return 16;
  }
  return 0;
}
