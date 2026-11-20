# SPDX-License-Identifier: BSD-2-Clause

cmake_minimum_required(VERSION 3.31)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are forbidden; choose an external build directory")
endif()

project(
  slope_stability
  VERSION 0.1.0
  DESCRIPTION "Standalone C/PETSc slope-stability engine"
  LANGUAGES C
)

include(CTest)
include(GNUInstallDirs)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_SKIP_INSTALL_RPATH ON)

option(SLOPE_ENABLE_OPENMP "Enable project OpenMP kernels" ON)
option(SLOPE_ENABLE_FAILURE_INJECTION "Enable configure-test failure injection" OFF)
option(SLOPE_PROTOCOL_ONLY "Build protocol tools without numerical dependencies" OFF)
option(SLOPE_TSAN_KERNELS_ONLY "Build only isolated OpenMP kernels under TSan" OFF)
set(SLOPE_PETSC_INDEX_BITS "32" CACHE STRING "Required PetscInt width: 32 or 64")
set_property(CACHE SLOPE_PETSC_INDEX_BITS PROPERTY STRINGS 32 64)
set(SLOPE_SANITIZER "none" CACHE STRING "Sanitizer: none, address-undefined, or thread")
set_property(CACHE SLOPE_SANITIZER PROPERTY STRINGS none address-undefined thread)
option(SLOPE_ENABLE_COVERAGE "Enable compiler coverage instrumentation" OFF)
set(SLOPE_BUILD_PROFILE "developer" CACHE STRING "Stable build-profile identity")

if(NOT SLOPE_PETSC_INDEX_BITS MATCHES "^(32|64)$")
  message(FATAL_ERROR "SLOPE_PETSC_INDEX_BITS must be 32 or 64")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Dependencies)
include(BuildManifest)

function(slope_project_warnings target)
  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
      "${target}"
      PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
        -Wconversion
        -Wshadow
        -Wstrict-prototypes
        -Wmissing-prototypes
        -Wformat=2
    )
  else()
    message(FATAL_ERROR "Only GCC and Clang-family compilers are supported")
  endif()
endfunction()

function(slope_instrument target)
  if(SLOPE_SANITIZER STREQUAL "address-undefined")
    target_compile_options("${target}" PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options("${target}" PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  elseif(SLOPE_SANITIZER STREQUAL "thread")
    target_compile_options("${target}" PRIVATE -fsanitize=thread -fno-omit-frame-pointer)
    target_link_options("${target}" PRIVATE -fsanitize=thread -fno-omit-frame-pointer)
  elseif(NOT SLOPE_SANITIZER STREQUAL "none")
    message(FATAL_ERROR "Unsupported SLOPE_SANITIZER=${SLOPE_SANITIZER}")
  endif()
  if(SLOPE_ENABLE_COVERAGE)
    target_compile_options("${target}" PRIVATE --coverage -O0)
    target_link_options("${target}" PRIVATE --coverage)
  endif()
endfunction()

add_library(
  slope_protocol
  STATIC
    generated/src/ss_protocol_generated.c
    src/core/sort.c
    src/protocol/json.c
)
target_include_directories(
  slope_protocol
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/generated/include>"
)
slope_project_warnings(slope_protocol)
slope_instrument(slope_protocol)

add_executable(slope-schema-check src/protocol/schema_check.c)
target_link_libraries(slope-schema-check PRIVATE slope_protocol)
slope_project_warnings(slope-schema-check)
slope_instrument(slope-schema-check)

if(NOT SLOPE_PROTOCOL_ONLY)
  slope_write_build_manifest()
  configure_file(
    cmake/ss_features.h.in
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include/slope_internal/build/config.h"
    @ONLY
  )

  if(SLOPE_TSAN_KERNELS_ONLY)
    if(NOT SLOPE_SANITIZER STREQUAL "thread")
      message(FATAL_ERROR "SLOPE_TSAN_KERNELS_ONLY requires thread sanitizer")
    endif()
    if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
      message(FATAL_ERROR "SLOPE_TSAN_KERNELS_ONLY requires Clang/LLVM")
    endif()
    add_executable(slope-tsan-kernel-probe tests/native/tsan_kernel_probe.c)
    target_include_directories(
      slope-tsan-kernel-probe
      PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated/include"
    )
    target_link_libraries(slope-tsan-kernel-probe PRIVATE OpenMP::OpenMP_C)
    slope_project_warnings(slope-tsan-kernel-probe)
    slope_instrument(slope-tsan-kernel-probe)
    add_executable(
      ss_test_assembly_worker_race
      tests/native/assembly_worker_race_test.c
      generated/src/ss_fe_generated.c
      src/assembly/element.c
      src/core/sha256.c
      src/fe/fe.c
      src/material/material.c
    )
    target_include_directories(
      ss_test_assembly_worker_race
      PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/generated/include"
        "${CMAKE_CURRENT_BINARY_DIR}/generated/include"
        ${MPI_C_INCLUDE_DIRS}
        ${PETSC_INCLUDE_DIRS}
        ${HDF5_INCLUDE_DIRS}
    )
    target_compile_options(
      ss_test_assembly_worker_race
      PRIVATE -ffunction-sections -fdata-sections
    )
    target_link_options(ss_test_assembly_worker_race PRIVATE -Wl,--gc-sections)
    target_link_libraries(ss_test_assembly_worker_race PRIVATE OpenMP::OpenMP_C m)
    slope_project_warnings(ss_test_assembly_worker_race)
    slope_instrument(ss_test_assembly_worker_race)
    find_library(SLOPE_ARCHER_LIBRARY NAMES archer REQUIRED)
    message(STATUS "OpenMP race detector: ${SLOPE_ARCHER_LIBRARY}")
  else()
    add_library(slope_hash STATIC src/core/sha256.c)
    target_include_directories(
      slope_hash
      PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    )
    slope_project_warnings(slope_hash)
    slope_instrument(slope_hash)

    add_library(slope_material STATIC src/material/material.c src/material/saturation.c)
    target_include_directories(
      slope_material
      PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    )
    target_link_libraries(slope_material PUBLIC slope_hash m)
    slope_project_warnings(slope_material)
    slope_instrument(slope_material)

    add_library(slope_yyjson STATIC third_party/yyjson/src/yyjson.c)
    target_include_directories(
      slope_yyjson
      PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/yyjson/src>"
    )
    set_target_properties(
      slope_yyjson
      PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS OFF
    )
    slope_instrument(slope_yyjson)

    add_library(
      slope_fe
      STATIC
        generated/src/ss_fe_generated.c
        src/fe/fe.c
    )
    target_include_directories(
      slope_fe
      PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/generated/include>"
    )
    target_link_libraries(slope_fe PUBLIC m)
    slope_project_warnings(slope_fe)
    slope_instrument(slope_fe)

    add_library(
      slope_native
      STATIC
        src/app/lifecycle.c
        src/app/runtime.c
        src/app/solver.c
        src/app/solver_cell_geometry.c
        src/app/solver_checkpoint.c
        src/app/solver_checkpoint_restore.c
        src/app/solver_checkpoint_write.c
        src/app/solver_hydraulic.c
        src/app/solver_hydraulic_output.c
        src/app/solver_mechanics.c
        src/app/solver_mechanics_analysis.c
        src/app/solver_pmg_setup.c
        src/app/solver_postprocess.c
        src/app/solver_publish.c
        src/app/solver_publish_hydraulic.c
        src/app/solver_publish_mechanics.c
        src/app/solver_publish_history.c
        src/app/solver_publish_monitor.c
        src/app/solver_publish_spool.c
        src/app/solver_publish_views.c
        src/app/solver_checkpoint_boundary.c
        src/app/solver_tuning.c
        src/app/stable_id.c
        src/app/solver_coupling.c
        src/app/solver_constraints.c
        src/app/solver_graph.c
        src/app/solver_mesh_input.c
        src/app/solver_model_input.c
        src/app/solver_run_policy.c
        src/analysis/analysis.c
        src/analysis/continuation_load.c
        src/analysis/continuation_work.c
        src/assembly/element.c
        src/assembly/operator.c
        src/assembly/operator_assembled.c
        src/assembly/operator_assembly_coo.c
        src/assembly/operator_coarse.c
        src/assembly/operator_matfree.c
        src/assembly/operator_matfree_action.c
        src/assembly/operator_matfree_exchange.c
        src/assembly/operator_matfree_reference3d.c
        src/assembly/operator_matfree_blocks.c
        src/assembly/operator_setup.c
        src/assembly/operator_workspace.c
        src/assembly/plan.c
        src/build/features.c
        src/build/parallel_hdf5.c
        src/core/collective.c
        src/core/core.c
        src/core/solver_options.c
        src/core/util.c
        src/coupling/coupling.c
        src/coupling/coupling_location.c
        src/coupling/coupling_snapshot.c
        src/coupling/coupling_transfer.c
        src/deflation/deflation.c
        src/deflation/deflation_basis.c
        src/deflation/deflation_coarse.c
        src/deflation/deflation_compress.c
        src/deflation/deflation_legacy.c
        src/deflation/deflation_persist.c
        src/diagnostics/diagnostics.c
        src/diagnostics/journal.c
        src/io/compiled_manifest.c
        src/io/hdf5_input.c
        src/io/json_input.c
        src/io/package.c
        src/hydraulic/hydraulic.c
        src/linear/linear.c
        src/linear/linear_recycle.c
        src/linear/linear_solve.c
        src/mesh/distribution.c
        src/mesh/facet_plan.c
        src/mesh/field_cell_plan.c
        src/mesh/field_constraints.c
        src/mesh/field_output.c
        src/mesh/field_space.c
        src/mesh/field_transfer.c
        src/mesh/mesh.c
        src/mesh/mesh_attach.c
        src/mesh/mesh_audit.c
        src/mesh/mesh_coordinates.c
        src/mesh/mesh_digest.c
        src/mesh/mesh_geometry_audit.c
        src/mesh/mesh_index.c
        src/mesh/mesh_labels.c
        src/mesh/mesh_supports.c
        src/mesh/mesh_point_map.c
        src/mesh/ownership.c
        src/nonlinear/nonlinear.c
        src/nonlinear/nonlinear_controlled.c
        src/nonlinear/nonlinear_fixed.c
        src/nonlinear/nonlinear_iteration.c
        src/nonlinear/nonlinear_policy.c
        src/nonlinear/nonlinear_workspace.c
        src/pmg/pmg.c
        src/pmg/pmg_active.c
        src/pmg/pmg_cycle.c
        src/pmg/pmg_levels.c
        src/pmg/pmg_nullspace.c
        src/pmg/pmg_policy.c
        src/pmg/pmg_setup.c
        src/result/checkpoint_metadata.c
        src/result/checkpoint_records.c
        src/result/checkpoint_restore.c
        src/result/checkpoint_write.c
        src/result/common.c
        src/result/file_io.c
        src/result/index.c
        src/result/quantity.c
        src/result/quantity_digest.c
        src/result/result_verify.c
        src/result/writer.c
        src/result/writer_hdf5.c
        src/result/writer_history.c
        src/result/writer_layout.c
        src/result/writer_publish.c
        src/result/writer_records.c
        src/result/writer_spec.c
    )
    target_include_directories(
      slope_native
      PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/generated/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated/include>"
    )
    target_link_libraries(
      slope_native
      PUBLIC
        slope_protocol
        slope_fe
        slope_hash
        slope_material
        slope_yyjson
        MPI::MPI_C
        PkgConfig::PETSC
        HDF5::HDF5
        BLAS::BLAS
        LAPACK::LAPACK
    )
    if(SLOPE_ENABLE_OPENMP)
      target_link_libraries(slope_native PUBLIC OpenMP::OpenMP_C)
    endif()
    slope_project_warnings(slope_native)
    slope_instrument(slope_native)

    add_executable(slope-engine src/app/engine.c)
    target_link_libraries(slope-engine PRIVATE slope_native)
    slope_project_warnings(slope-engine)
    slope_instrument(slope-engine)

    add_executable(slope-probe src/app/probe.c)
    target_link_libraries(slope-probe PRIVATE slope_native)
    slope_project_warnings(slope-probe)
    slope_instrument(slope-probe)

    # Native test executables; cmake/TestRegistration.cmake registers the tests.
    function(slope_test_executable name)
      cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "SOURCES;LINK")
      if(NOT arg_LINK)
        set(arg_LINK slope_native)
      endif()
      add_executable("${name}" ${arg_SOURCES})
      target_link_libraries("${name}" PRIVATE ${arg_LINK})
      slope_project_warnings("${name}")
      slope_instrument("${name}")
    endfunction()

    add_library(
      ss_test_fixtures
      STATIC
        tests/native/fixtures/coupled_fixture.c
        tests/native/fixtures/hydraulic_fixture.c
        tests/native/fixtures/mechanics_fixture.c
    )
    target_link_libraries(ss_test_fixtures PUBLIC slope_native)
    slope_project_warnings(ss_test_fixtures)
    slope_instrument(ss_test_fixtures)

    slope_test_executable(ss_test_build_features SOURCES tests/native/build_features_test.c)
    slope_test_executable(ss_test_parallel_hdf5 SOURCES tests/native/parallel_hdf5_test.c)
    slope_test_executable(
      ss_test_input_manifest_mpi
      SOURCES tests/native/input_manifest_mpi_test.c
    )
    slope_test_executable(
      ss_test_core_io_lifecycle
      SOURCES tests/native/core_io_lifecycle_test.c
    )
    slope_test_executable(ss_test_app_mpi SOURCES tests/native/app_mpi_test.c)
    slope_test_executable(ss_test_fe SOURCES tests/native/fe_test.c LINK slope_fe)
    slope_test_executable(ss_test_mesh SOURCES tests/native/mesh_test.c)
    slope_test_executable(
      ss_test_mesh_field_mpi
      SOURCES tests/native/mesh_field_mpi_test.c tests/native/mesh_field_mpi_fixtures.c
    )
    slope_test_executable(ss_test_mesh_join_mpi SOURCES tests/native/mesh_join_mpi_test.c)
    slope_test_executable(ss_test_boundary_trace SOURCES tests/native/boundary_trace_test.c)
    target_include_directories(ss_test_boundary_trace PRIVATE src/app)
    slope_test_executable(
      ss_test_material
      SOURCES tests/native/material_test.c
      LINK slope_material
    )
    if(SLOPE_ENABLE_OPENMP)
      target_link_libraries(ss_test_material PRIVATE OpenMP::OpenMP_C)
    endif()
    slope_test_executable(
      ss_test_assembly
      SOURCES tests/native/assembly_test.c tests/native/assembly_matrix_free_cases.c
    )
    slope_test_executable(
      ss_test_assembly_mpi
      SOURCES tests/native/assembly_mpi_test.c tests/native/assembly_mpi_affinity.c
    )
    slope_test_executable(
      ss_test_solver_correctness
      SOURCES tests/native/solver_correctness_test.c
    )
    slope_test_executable(
      ss_test_linear_deflation
      SOURCES
        tests/native/linear_deflation_test.c
        tests/native/linear_deflation_projection_cases.c
    )
    slope_test_executable(
      ss_test_linear_deflation_mpi
      SOURCES tests/native/linear_deflation_mpi_test.c
    )
    slope_test_executable(ss_test_pmg SOURCES tests/native/pmg_test.c)
    slope_test_executable(ss_test_pmg_mpi SOURCES tests/native/pmg_mpi_test.c)
    slope_test_executable(
      ss_test_nonlinear_analysis
      SOURCES
        tests/native/nonlinear_analysis_test.c
        tests/native/nonlinear_analysis_continuation_cases.c
      LINK ss_test_fixtures
    )
    slope_test_executable(
      ss_test_nonlinear_analysis_mpi
      SOURCES tests/native/nonlinear_analysis_mpi_test.c
    )
    slope_test_executable(
      ss_test_hydraulic_coupling
      SOURCES tests/native/hydraulic_coupling_test.c
      LINK ss_test_fixtures
    )
    slope_test_executable(
      ss_test_hydraulic_coupling_mpi
      SOURCES tests/native/hydraulic_coupling_mpi_test.c
      LINK ss_test_fixtures
    )
    slope_test_executable(
      ss_test_result_checkpoint
      SOURCES
        tests/native/result_checkpoint_test.c
        tests/native/result_checkpoint_restart_cases.c
    )
    slope_test_executable(
      ss_test_result_append_io
      SOURCES tests/native/result_append_io_test.c
    )
    target_link_options(ss_test_result_append_io PRIVATE "LINKER:--wrap=H5Dread")
    slope_test_executable(
      ss_test_checkpoint_basis
      SOURCES tests/native/checkpoint_basis_test.c
    )
    slope_test_executable(ss_test_kernel_oracles SOURCES tests/native/kernel_oracle_test.c)
    slope_test_executable(ss_test_natural_energy SOURCES tests/native/natural_energy_test.c)
    slope_test_executable(
      ss_test_curved_operator
      SOURCES tests/native/curved_operator_test.c
    )

    install(TARGETS slope-engine slope-probe RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
    install(
      FILES
        "${SLOPE_BUILD_MANIFEST_PATH}"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/slope-stability"
    )
    install(
      DIRECTORY schemas/generated/json schemas/hdf5-layout
      DESTINATION "${CMAKE_INSTALL_DATADIR}/slope-stability/schemas"
    )
    install(
      FILES LICENSE THIRD_PARTY_NOTICES.md
      DESTINATION "${CMAKE_INSTALL_DATADIR}/slope-stability"
    )
    install(
      DIRECTORY licenses/
      DESTINATION "${CMAKE_INSTALL_DATADIR}/slope-stability/licenses"
    )
    configure_file(
      cmake/InstallManifest.cmake.in
      "${CMAKE_CURRENT_BINARY_DIR}/InstallManifest.cmake"
      @ONLY
    )
    install(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/InstallManifest.cmake")
  endif()
endif()

if(BUILD_TESTING)
  if(SLOPE_TSAN_KERNELS_ONLY)
    add_test(NAME BLD-UNIT-FEATURE-001 COMMAND slope-tsan-kernel-probe)
    add_test(NAME ASM-RACE-ARCHER-001 COMMAND ss_test_assembly_worker_race)
    set_tests_properties(
      BLD-UNIT-FEATURE-001
      ASM-RACE-ARCHER-001
      PROPERTIES
        ENVIRONMENT
          "OMP_NUM_THREADS=8;OMP_DYNAMIC=FALSE;OMP_TOOL_LIBRARIES=${SLOPE_ARCHER_LIBRARY}"
        LABELS "build;native;serial;sanitizer"
        TIMEOUT 30
    )
    set_tests_properties(
      ASM-RACE-ARCHER-001
      PROPERTIES
        LABELS "assembly;native;serial;race;sanitizer"
        TIMEOUT 120
    )
  else()
    find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
    include(TestRegistration)
    slope_register_native_tests()
    include(PythonTests)
    slope_register_python_tests()

    if(TARGET slope_native)
      include(tests/native/RobustnessTests.cmake)
      slope_register_collective_robustness_tests()
    endif()

    # Sanitized builds run several times slower; scale every explicit test
    # timeout so slow-but-correct tests are not reported as hangs.
    if(NOT SLOPE_SANITIZER STREQUAL "none")
      get_property(slope_registered_tests DIRECTORY PROPERTY TESTS)
      foreach(slope_test IN LISTS slope_registered_tests)
        get_test_property("${slope_test}" TIMEOUT slope_test_timeout)
        if(slope_test_timeout)
          math(EXPR slope_test_timeout "${slope_test_timeout} * 4")
          set_tests_properties("${slope_test}" PROPERTIES TIMEOUT "${slope_test_timeout}")
        endif()
      endforeach()
    endif()
  endif()
endif()
