/** @file
 * @brief Build identities and the dependency features compiled into the engine.
 *
 * All queries return constants fixed at configure time and are safe from any
 * thread.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_BUILD_FEATURES_API_H
#define SS_BUILD_FEATURES_API_H

#include <petscsystypes.h>

#include <stdint.h>

/** Build identity.
 * @return Static `sha256:` identity of sources, dependencies and configuration. */
const char *ss_build_id(void);
/** Dependency identity.
 * @return Static `sha256:` identity of the dependency configuration. */
const char *ss_dependency_id(void);
/** Source identity.
 * @return Static identity of the source revision or tree content. */
const char *ss_source_id(void);
/** Project version.
 * @return Static semantic version string, for example `0.1.0`. */
const char *ss_project_version(void);
/** PETSc index width.
 * @return 32 or 64, the bit width of `PetscInt`. */
int ss_petsc_index_bits(void);
/** OpenMP support of the engine.
 * @return Nonzero when the engine kernels are built with OpenMP. */
int ss_openmp_enabled(void);
/** OpenMP support of PETSc.
 * @return Nonzero when PETSc was configured with OpenMP. */
int ss_petsc_openmp_enabled(void);
/** Threaded PETSc kernels.
 * @return Nonzero when PETSc uses OpenMP kernels. */
int ss_petsc_openmp_kernels_enabled(void);
/** Kokkos support of PETSc.
 * @return Nonzero when PETSc provides Kokkos. */
int ss_petsc_kokkos_enabled(void);
/** MUMPS support of PETSc.
 * @return Nonzero when PETSc provides MUMPS. */
int ss_petsc_mumps_enabled(void);
/** hwloc support of PETSc.
 * @return Nonzero when PETSc provides hwloc. */
int ss_petsc_hwloc_enabled(void);
/** OpenBLAS thread control.
 * @return Nonzero when PETSc can set the OpenBLAS thread count. */
int ss_petsc_openblas_thread_control_enabled(void);
/** SuperLU_DIST support of PETSc.
 * @return Nonzero when PETSc provides SuperLU_DIST for distributed coarse LU. */
int ss_petsc_superlu_dist_enabled(void);
/** Threaded SuperLU_DIST.
 * @return Nonzero when SuperLU_DIST was built with OpenMP. */
int ss_superlu_dist_openmp_enabled(void);

/**
 * Convert an unsigned 64-bit count to `PetscInt` without overflow.
 *
 * @param[in] value Count to convert.
 * @param[out] converted Converted value; unchanged on failure.
 * @return 0 on success, 1 when @p converted is null or @p value exceeds
 *   `PETSC_MAX_INT`.
 */
int ss_petsc_index_from_u64(uint64_t value, PetscInt *converted);

#endif
