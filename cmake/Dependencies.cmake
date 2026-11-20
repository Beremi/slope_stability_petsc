# SPDX-License-Identifier: BSD-2-Clause

include(CheckCSourceCompiles)
include(CMakePushCheckState)

if(SLOPE_PROTOCOL_ONLY)
  return()
endif()

find_package(MPI 3.1 REQUIRED COMPONENTS C)

# MPI compiler wrappers may inject a runtime path (-Wl,-rpath) into the link
# flags. An installed engine must not carry an absolute RUNPATH, so the
# wrapper runtime path is filtered according to SLOPE_MPI_WRAPPER_RPATH:
#
# - system (default): drop runtime paths that name a default loader
#   directory (they are redundant) or a directory listed in
#   SLOPE_MPI_RPATH_ALLOWLIST; fail on any other runtime path;
# - strip: drop every wrapper runtime path (the MPI libraries must then be
#   found through the loader configuration or LD_LIBRARY_PATH);
# - reject: fail if the wrapper injects any runtime path.
set(
  SLOPE_MPI_WRAPPER_RPATH
  "system"
  CACHE STRING
  "Handling of MPI wrapper runtime paths: system, strip or reject"
)
set_property(CACHE SLOPE_MPI_WRAPPER_RPATH PROPERTY STRINGS system strip reject)
set(
  SLOPE_MPI_RPATH_ALLOWLIST
  ""
  CACHE STRING
  "Additional MPI wrapper runtime paths that the system policy may drop"
)
if(NOT SLOPE_MPI_WRAPPER_RPATH MATCHES "^(system|strip|reject)$")
  message(FATAL_ERROR "SLOPE_MPI_WRAPPER_RPATH must be system, strip or reject")
endif()

function(slope_filter_mpi_rpath)
  if(NOT MPI_C_LINK_FLAGS MATCHES "rpath")
    return()
  endif()
  set(
    loader_directories
    /lib
    /lib64
    /usr/lib
    /usr/lib64
    /lib/x86_64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /lib/aarch64-linux-gnu
    /usr/lib/aarch64-linux-gnu
    ${SLOPE_MPI_RPATH_ALLOWLIST}
  )
  separate_arguments(tokens UNIX_COMMAND "${MPI_C_LINK_FLAGS}")
  set(kept "")
  set(dropped "")
  set(rejected "")
  set(expect_directory FALSE)
  foreach(token IN LISTS tokens)
    set(directory "")
    if(expect_directory)
      set(expect_directory FALSE)
      if(token MATCHES "^-Wl,(.+)$")
        set(directory "${CMAKE_MATCH_1}")
      else()
        set(directory "${token}")
      endif()
    elseif(token STREQUAL "-Wl,-rpath" OR token STREQUAL "-Wl,--rpath")
      set(expect_directory TRUE)
      continue()
    elseif(token MATCHES "^-Wl,-?-rpath[,=](.+)$")
      set(directory "${CMAKE_MATCH_1}")
    elseif(token STREQUAL "-Wl,--enable-new-dtags" OR token STREQUAL "-Wl,--disable-new-dtags")
      # Only meaningful together with a runtime path.
      continue()
    else()
      list(APPEND kept "${token}")
      continue()
    endif()
    string(REPLACE "," ";" directories "${directory}")
    foreach(path IN LISTS directories)
      string(REGEX REPLACE "/+$" "" normalized "${path}")
      if(SLOPE_MPI_WRAPPER_RPATH STREQUAL "strip")
        list(APPEND dropped "${path}")
      elseif(SLOPE_MPI_WRAPPER_RPATH STREQUAL "system" AND normalized IN_LIST loader_directories)
        list(APPEND dropped "${path}")
      else()
        list(APPEND rejected "${path}")
      endif()
    endforeach()
  endforeach()
  if(rejected)
    message(
      FATAL_ERROR
      "MPI wrapper injects runtime path(s) ${rejected} (link flags: "
      "${MPI_C_LINK_FLAGS}). Use -DSLOPE_MPI_WRAPPER_RPATH=strip to drop "
      "them, or list them in SLOPE_MPI_RPATH_ALLOWLIST."
    )
  endif()
  list(JOIN kept " " remaining)
  message(STATUS "Dropped MPI wrapper runtime path(s): ${dropped}")
  set(MPI_C_LINK_FLAGS "${remaining}" CACHE STRING "MPI C linker flags" FORCE)
  if(remaining STREQUAL "")
    set_property(TARGET MPI::MPI_C PROPERTY INTERFACE_LINK_OPTIONS "")
  else()
    set_property(TARGET MPI::MPI_C PROPERTY INTERFACE_LINK_OPTIONS "SHELL:${remaining}")
  endif()
endfunction()
slope_filter_mpi_rpath()

find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET GLOBAL PETSc)

# PETSc's own OpenMP build must not select project kernel compilation mode.
# The supported GNU flag is supplied by OpenMP::OpenMP_C when the project
# enables OpenMP; retain PETSc's libraries and every unrelated usage option.
get_target_property(_slope_petsc_compile_options PkgConfig::PETSC INTERFACE_COMPILE_OPTIONS)
if(_slope_petsc_compile_options)
  list(REMOVE_ITEM _slope_petsc_compile_options "-fopenmp")
  set_property(TARGET PkgConfig::PETSC PROPERTY INTERFACE_COMPILE_OPTIONS
               "${_slope_petsc_compile_options}")
endif()

if(NOT PETSC_VERSION MATCHES "^3\\.24\\.")
  message(FATAL_ERROR "PETSc 3.24.x is required; discovered ${PETSC_VERSION}")
endif()

set(HDF5_PREFER_PARALLEL TRUE)
find_package(HDF5 REQUIRED COMPONENTS C HL)
if(NOT HDF5_IS_PARALLEL)
  message(FATAL_ERROR "Parallel HDF5 is required")
endif()

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

if(SLOPE_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS C)
endif()

if(SLOPE_ENABLE_FAILURE_INJECTION)
  if(SLOPE_TEST_FORCE_MISSING_HDF5)
    message(FATAL_ERROR "injected: parallel HDF5 is unavailable")
  endif()
  if(SLOPE_TEST_FORCE_MPI_HDF5_MISMATCH)
    message(FATAL_ERROR "injected: MPI/HDF5 implementation mismatch")
  endif()
  if(SLOPE_TEST_FORCE_PETSC_INCOMPATIBLE)
    message(FATAL_ERROR "injected: PETSc family/scalar/index mismatch")
  endif()
endif()

pkg_get_variable(SLOPE_PETSC_CCOMPILER PETSc ccompiler)
pkg_get_variable(SLOPE_PETSC_CFLAGS_EXTRA PETSc cflags_extra)
string(REPLACE ";" " " SLOPE_PETSC_CFLAGS_EXTRA "${SLOPE_PETSC_CFLAGS_EXTRA}")
find_program(SLOPE_PETSC_CCOMPILER_PATH NAMES "${SLOPE_PETSC_CCOMPILER}" REQUIRED)
file(REAL_PATH "${SLOPE_PETSC_CCOMPILER_PATH}" SLOPE_PETSC_CCOMPILER_REAL)
file(REAL_PATH "${MPI_C_COMPILER}" SLOPE_MPI_CCOMPILER_REAL)
if(NOT SLOPE_PETSC_CCOMPILER_REAL STREQUAL SLOPE_MPI_CCOMPILER_REAL)
  message(
    FATAL_ERROR
    "PETSc compiler wrapper ${SLOPE_PETSC_CCOMPILER_REAL} differs from "
    "project MPI wrapper ${SLOPE_MPI_CCOMPILER_REAL}"
  )
endif()

if(HDF5_C_COMPILER_EXECUTABLE)
  execute_process(
    COMMAND "${HDF5_C_COMPILER_EXECUTABLE}" -show
    RESULT_VARIABLE SLOPE_HDF5_SHOW_RESULT
    OUTPUT_VARIABLE SLOPE_HDF5_SHOW
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT SLOPE_HDF5_SHOW_RESULT EQUAL 0)
    message(FATAL_ERROR "Unable to inspect parallel HDF5 compiler wrapper")
  endif()
  separate_arguments(SLOPE_HDF5_SHOW_ARGS UNIX_COMMAND "${SLOPE_HDF5_SHOW}")
  list(GET SLOPE_HDF5_SHOW_ARGS 0 SLOPE_HDF5_MPI_WRAPPER)
  find_program(SLOPE_HDF5_MPI_WRAPPER_PATH NAMES "${SLOPE_HDF5_MPI_WRAPPER}" REQUIRED)
  file(REAL_PATH "${SLOPE_HDF5_MPI_WRAPPER_PATH}" SLOPE_HDF5_MPI_WRAPPER_REAL)
  if(NOT SLOPE_HDF5_MPI_WRAPPER_REAL STREQUAL SLOPE_MPI_CCOMPILER_REAL)
    message(
      FATAL_ERROR
      "HDF5 compiler wrapper ${SLOPE_HDF5_MPI_WRAPPER_REAL} differs from "
      "project MPI wrapper ${SLOPE_MPI_CCOMPILER_REAL}"
    )
  endif()
endif()

cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_LIBRARIES PkgConfig::PETSC MPI::MPI_C HDF5::HDF5)
set(CMAKE_REQUIRED_QUIET FALSE)
if(SLOPE_PETSC_INDEX_BITS STREQUAL "64")
  set(SLOPE_INDEX_PROBE "
#ifndef PETSC_USE_64BIT_INDICES
#error PETSc 64-bit indices required
#endif")
else()
  set(SLOPE_INDEX_PROBE "
#ifdef PETSC_USE_64BIT_INDICES
#error PETSc 32-bit indices required
#endif")
endif()
check_c_source_compiles(
  "
#include <petsc.h>
#include <petscdmplex.h>
#include <petscksp.h>
#include <petscviewerhdf5.h>
#include <petscconf.h>
#include <petscversion.h>
#if PETSC_VERSION_MAJOR != 3 || PETSC_VERSION_MINOR != 24
#error PETSc 3.24.x required
#endif
#ifndef PETSC_USE_REAL_DOUBLE
#error PETSc real double required
#endif
#ifdef PETSC_USE_COMPLEX
#error PETSc complex scalars unsupported
#endif
#ifndef PETSC_HAVE_HDF5
#error PETSc HDF5 viewer support required
#endif
#ifndef PETSC_HAVE_HYPRE
#error PETSc HYPRE support required
#endif
#if !defined(PETSC_HAVE_PARMETIS) && !defined(PETSC_HAVE_PTSCOTCH)
#error ParMETIS or PT-Scotch required
#endif
${SLOPE_INDEX_PROBE}
int main(int argc, char **argv) {
  DM dm = NULL;
  KSP ksp = NULL;
  PetscViewer viewer = NULL;
  if (PetscInitialize(&argc, &argv, NULL, NULL)) return 1;
  if (DMPlexCreate(PETSC_COMM_WORLD, &dm)) return 2;
  if (KSPCreate(PETSC_COMM_WORLD, &ksp)) return 3;
  if (PetscViewerHDF5Open(PETSC_COMM_WORLD, \"probe.h5\", FILE_MODE_WRITE, &viewer)) return 4;
  PetscViewerDestroy(&viewer);
  KSPDestroy(&ksp);
  DMDestroy(&dm);
  return PetscFinalize();
}"
  SLOPE_PETSC_FEATURE_PROBE
)
cmake_pop_check_state()
if(NOT SLOPE_PETSC_FEATURE_PROBE)
  message(
    FATAL_ERROR
    "PETSc feature probe failed: require 3.24.x, real double, requested index "
    "width, HDF5, HYPRE, and ParMETIS or PT-Scotch"
  )
endif()

cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_LIBRARIES PkgConfig::PETSC MPI::MPI_C)
check_c_source_compiles(
  "
#include <petscconf.h>
#ifndef PETSC_HAVE_SUPERLU_DIST
#error PETSc SuperLU_DIST support unavailable
#endif
int main(void) { return 0; }
"
  SLOPE_PETSC_HAVE_SUPERLU_DIST
)
cmake_pop_check_state()

function(slope_check_petsc_define definition output)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_LIBRARIES PkgConfig::PETSC MPI::MPI_C)
  check_c_source_compiles(
    "
#include <petscconf.h>
#ifndef ${definition}
#error ${definition} unavailable
#endif
int main(void) { return 0; }
"
    "${output}"
  )
  cmake_pop_check_state()
  set("${output}" "${${output}}" PARENT_SCOPE)
endfunction()

slope_check_petsc_define(PETSC_HAVE_OPENMP SLOPE_PETSC_HAVE_OPENMP)
slope_check_petsc_define(PETSC_USE_OPENMP_KERNELS SLOPE_PETSC_USE_OPENMP_KERNELS)
slope_check_petsc_define(PETSC_HAVE_KOKKOS SLOPE_PETSC_HAVE_KOKKOS)
slope_check_petsc_define(PETSC_HAVE_MUMPS SLOPE_PETSC_HAVE_MUMPS)
slope_check_petsc_define(PETSC_HAVE_HWLOC SLOPE_PETSC_HAVE_HWLOC)
slope_check_petsc_define(PETSC_HAVE_OPENBLAS_SET_NUM_THREADS
                         SLOPE_PETSC_HAVE_OPENBLAS_THREAD_CONTROL)

if(SLOPE_PETSC_USE_OPENMP_KERNELS AND NOT SLOPE_PETSC_HAVE_OPENMP)
  message(FATAL_ERROR "PETSc reports OpenMP kernels without OpenMP support")
endif()

# SuperLU_DIST has no installed feature macro for its OpenMP compilation.
# Detect direct references to the OpenMP runtime in the exact linked library;
# an indirect libgomp dependency (for example through a Fortran runtime) does
# not qualify the solver itself as threaded.
set(SLOPE_SUPERLU_DIST_HAVE_OPENMP FALSE)
if(SLOPE_PETSC_HAVE_SUPERLU_DIST)
  find_library(
    SLOPE_SUPERLU_DIST_LIBRARY
    NAMES superlu_dist
    PATHS ${PETSC_STATIC_LIBRARY_DIRS}
    NO_DEFAULT_PATH
  )
  if(SLOPE_SUPERLU_DIST_LIBRARY AND CMAKE_NM)
    execute_process(
      COMMAND "${CMAKE_NM}" -D "${SLOPE_SUPERLU_DIST_LIBRARY}"
      RESULT_VARIABLE SLOPE_SUPERLU_DIST_NM_RESULT
      OUTPUT_VARIABLE SLOPE_SUPERLU_DIST_SYMBOLS
      ERROR_QUIET
    )
    if(NOT SLOPE_SUPERLU_DIST_NM_RESULT EQUAL 0)
      execute_process(
        COMMAND "${CMAKE_NM}" "${SLOPE_SUPERLU_DIST_LIBRARY}"
        RESULT_VARIABLE SLOPE_SUPERLU_DIST_NM_RESULT
        OUTPUT_VARIABLE SLOPE_SUPERLU_DIST_SYMBOLS
        ERROR_QUIET
      )
    endif()
    if(
      SLOPE_SUPERLU_DIST_NM_RESULT EQUAL 0
      AND SLOPE_SUPERLU_DIST_SYMBOLS MATCHES "(GOMP_|omp_get_|omp_set_)"
    )
      set(SLOPE_SUPERLU_DIST_HAVE_OPENMP TRUE)
    endif()
  endif()
endif()

cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_C HDF5::HDF5)
check_c_source_compiles(
  "
#include <mpi.h>
#include <hdf5.h>
#ifndef H5_HAVE_PARALLEL
#error parallel HDF5 required
#endif
int main(void) {
  hid_t property = H5Pcreate(H5P_FILE_ACCESS);
  if (property < 0) return 1;
  if (H5Pset_fapl_mpio(property, MPI_COMM_WORLD, MPI_INFO_NULL) < 0) return 2;
  return H5Pclose(property) < 0;
}"
  SLOPE_PARALLEL_HDF5_PROBE
)
cmake_pop_check_state()
if(NOT SLOPE_PARALLEL_HDF5_PROBE)
  message(FATAL_ERROR "Native parallel HDF5 compile/link probe failed")
endif()

message(STATUS "MPI C wrapper: ${SLOPE_MPI_CCOMPILER_REAL}")
message(STATUS "PETSc: ${PETSC_VERSION} (${PETSC_PREFIX})")
message(STATUS "HDF5: ${HDF5_VERSION} parallel=${HDF5_IS_PARALLEL}")
message(STATUS "Required PetscInt width: ${SLOPE_PETSC_INDEX_BITS}")
message(STATUS "Project OpenMP: ${SLOPE_ENABLE_OPENMP}")
message(
  STATUS
  "PETSc OpenMP: ${SLOPE_PETSC_HAVE_OPENMP}; kernels: ${SLOPE_PETSC_USE_OPENMP_KERNELS}"
)
message(STATUS "SuperLU_DIST OpenMP: ${SLOPE_SUPERLU_DIST_HAVE_OPENMP}")
message(
  STATUS
  "PETSc optional backends: Kokkos=${SLOPE_PETSC_HAVE_KOKKOS}; MUMPS=${SLOPE_PETSC_HAVE_MUMPS}; hwloc=${SLOPE_PETSC_HAVE_HWLOC}"
)
message(STATUS "PETSc OpenBLAS thread control: ${SLOPE_PETSC_HAVE_OPENBLAS_THREAD_CONTROL}")
