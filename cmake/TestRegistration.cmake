# SPDX-License-Identifier: BSD-2-Clause
#
# Registration of the native (C) tests and the shared test labels.
#
# Every test carries three kinds of labels:
#   - one area label, derived from the identifier prefix by slope_test_area()
#     or given with AREA (mesh, fe, assembly, pmg, nonlinear, output, ...);
#   - a language label, native or python;
#   - execution labels: serial, mpi, hybrid (a layout with more than one
#     OpenMP thread per rank), openmp, integration, e2e, smoke, ...
#
# slope_native_test(<id> <target> [MPI <ranks>] [ARGS <arg>... | NO_ARGS]
#                   [AREA <label>] [LABELS <label>...] [TIMEOUT <s>]
#                   [PROCESSORS <n>] [ENVIRONMENT <var=value>...])
#   runs `<target> <id>` (or `<target> <args>`), under mpiexec with <ranks>
#   processes when MPI is given.
#
# slope_native_layout_test(<id> <target> SUITE <suite> LAYOUTS <layout>...
#                          [RUNNER_ARGS <arg>...] [AREA <label>]
#                          [LABELS <label>...] [TIMEOUT <s>] [PROCESSORS <n>]
#                          [ENVIRONMENT <var=value>...])
#   runs <target> through tests/native/support/mpi_layout_runner.py at each
#   layout: `R` ranks, `RxT` ranks x OpenMP threads, or `RxT/A` with A active
#   PMG coarse ranks. The suite selects the native protocol and the checks.
#
# Pass list-valued arguments (LABELS, ENVIRONMENT, ...) as separate words, not
# as one semicolon-separated string.

set(
  SLOPE_TEST_AREAS
  ANA=analysis
  APP=app
  ASM=assembly
  AUT=authoring
  BLD=build
  CAT=catalog
  CMP=compilation
  COR=core
  CPL=coupling
  DEF=deflation
  DIA=diagnostics
  E2E=e2e
  FE=fe
  HYD=hydraulic
  IO=io
  LCH=launch
  LIN=linear
  MAT=material
  MESH=mesh
  MSH=mesh
  NLS=nonlinear
  OUT=output
  PMG=pmg
  PRO=protocol
  RDR=reader
  RES=resolution
  SOLVER=solver
  VIS=visualization
)

# Area label of <test_id> from its prefix, e.g. PMG-CONTRACT-REUSE-001 -> pmg.
function(slope_test_area out_var test_id)
  string(REGEX MATCH "^[A-Z0-9]+" prefix "${test_id}")
  foreach(entry IN LISTS SLOPE_TEST_AREAS)
    string(REPLACE "=" ";" entry "${entry}")
    list(GET entry 0 entry_prefix)
    if(entry_prefix STREQUAL prefix)
      list(GET entry 1 area)
      set("${out_var}" "${area}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  message(FATAL_ERROR "Test ${test_id} has no area for prefix '${prefix}'; pass AREA")
endfunction()

# Labels "<area>;<language>;<extra>..." of <test_id>; <area> may be empty.
function(slope_test_labels out_var test_id area language)
  if(area STREQUAL "")
    slope_test_area(area "${test_id}")
  endif()
  set(labels "${area}" "${language}" ${ARGN})
  set("${out_var}" "${labels}" PARENT_SCOPE)
endfunction()

function(slope_native_test test_id target)
  cmake_parse_arguments(
    PARSE_ARGV 2 arg "NO_ARGS" "MPI;AREA;TIMEOUT;PROCESSORS" "ARGS;LABELS;ENVIRONMENT"
  )
  if(arg_NO_ARGS)
    set(native_args "")
  elseif(DEFINED arg_ARGS)
    set(native_args ${arg_ARGS})
  else()
    set(native_args "${test_id}")
  endif()
  if(arg_MPI)
    set(launcher "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${arg_MPI}")
    set(execution mpi)
    if(NOT arg_PROCESSORS)
      set(arg_PROCESSORS "${arg_MPI}")
    endif()
  else()
    set(launcher "")
    set(execution serial)
  endif()
  if(NOT arg_TIMEOUT)
    set(arg_TIMEOUT 60)
  endif()
  slope_test_labels(labels "${test_id}" "${arg_AREA}" native ${execution} ${arg_LABELS})
  add_test(NAME "${test_id}" COMMAND ${launcher} "$<TARGET_FILE:${target}>" ${native_args})
  set_tests_properties("${test_id}" PROPERTIES LABELS "${labels}" TIMEOUT "${arg_TIMEOUT}")
  if(arg_PROCESSORS)
    set_tests_properties("${test_id}" PROPERTIES PROCESSORS "${arg_PROCESSORS}")
  endif()
  if(arg_ENVIRONMENT)
    set_tests_properties("${test_id}" PROPERTIES ENVIRONMENT "${arg_ENVIRONMENT}")
  endif()
endfunction()

function(slope_native_layout_test test_id target)
  cmake_parse_arguments(
    PARSE_ARGV 2 arg "" "SUITE;AREA;TIMEOUT;PROCESSORS"
    "LAYOUTS;RUNNER_ARGS;LABELS;ENVIRONMENT"
  )
  if(NOT arg_SUITE OR NOT arg_LAYOUTS)
    message(FATAL_ERROR "slope_native_layout_test(${test_id}) needs SUITE and LAYOUTS")
  endif()
  if(NOT arg_TIMEOUT)
    set(arg_TIMEOUT 180)
  endif()
  if(NOT arg_PROCESSORS)
    set(arg_PROCESSORS 4)
  endif()
  set(layout_args "")
  set(execution mpi)
  foreach(layout IN LISTS arg_LAYOUTS)
    list(APPEND layout_args --layout "${layout}")
    if(layout MATCHES "x([0-9]+)")
      if(CMAKE_MATCH_1 GREATER 1)
        set(execution mpi hybrid)
      endif()
    endif()
  endforeach()
  if("--direct" IN_LIST arg_RUNNER_ARGS)
    set(execution serial)
  endif()
  slope_test_labels(labels "${test_id}" "${arg_AREA}" native ${execution} ${arg_LABELS})
  set(environment PYTHONDONTWRITEBYTECODE=1 ${arg_ENVIRONMENT})
  add_test(
    NAME "${test_id}"
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/support/mpi_layout_runner.py"
      --suite "${arg_SUITE}"
      --mpiexec "${MPIEXEC_EXECUTABLE}"
      "--numproc-flag=${MPIEXEC_NUMPROC_FLAG}"
      --executable "$<TARGET_FILE:${target}>"
      --test-id "${test_id}"
      --artifact-root "${CMAKE_CURRENT_BINARY_DIR}/test-artifacts/${test_id}"
      ${layout_args}
      ${arg_RUNNER_ARGS}
  )
  set_tests_properties(
    "${test_id}"
    PROPERTIES
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      ENVIRONMENT "${environment}"
      LABELS "${labels}"
      PROCESSORS "${arg_PROCESSORS}"
      TIMEOUT "${arg_TIMEOUT}"
  )
endfunction()

function(slope_register_build_native_tests)
  slope_native_test(BLD-UNIT-FEATURE-001 ss_test_build_features NO_ARGS TIMEOUT 30)
endfunction()

function(slope_register_protocol_native_tests)
  slope_test_labels(labels PRO-H5-LAYOUT-001 "" native mpi)
  add_test(
    NAME PRO-H5-LAYOUT-001
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/support/hdf5_matrix_runner.py"
      --mpiexec "${MPIEXEC_EXECUTABLE}"
      "--numproc-flag=${MPIEXEC_NUMPROC_FLAG}"
      --executable "$<TARGET_FILE:ss_test_parallel_hdf5>"
      --artifact-root "${CMAKE_CURRENT_BINARY_DIR}/test-artifacts/PRO-H5-LAYOUT-001"
      --layout-source "${CMAKE_CURRENT_SOURCE_DIR}/schemas/source/hdf5-layouts.json"
      --layout-generated "${CMAKE_CURRENT_SOURCE_DIR}/schemas/hdf5-layout/generated-layouts.json"
      --ranks 1 2 4
  )
  set_tests_properties(
    PRO-H5-LAYOUT-001
    PROPERTIES
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
      LABELS "${labels}"
      PROCESSORS 4
      TIMEOUT 120
  )
  slope_native_test(
    PRO-COMPILED-MANIFEST-001 ss_test_input_manifest_mpi
    MPI 4 NO_ARGS TIMEOUT 60 ENVIRONMENT OMP_NUM_THREADS=1
  )
endfunction()

# Native core, diagnostics, input and application lifecycle.
function(slope_register_core_tests)
  foreach(
    test_id
    COR-UNIT-ARITH-001 COR-UNIT-INDEX-001 COR-UNIT-LIFECYCLE-001
    DIA-UNIT-IDENTITY-001 DIA-UNIT-ERROR-001
    IO-JSON-STRICT-001 IO-JSON-LIMITS-001 IO-PKG-INTEGRITY-001 IO-PKG-PATH-001
    IO-UNIT-CONFIG-001
    APP-LIFE-STATE-001
  )
    slope_native_test("${test_id}" ss_test_core_io_lifecycle TIMEOUT 30)
  endforeach()
  # The collective HDF5 input tests read a problem compiled by the Python
  # package, so the runner needs the in-tree packages on PYTHONPATH.
  foreach(
    test_id
    DIA-CONTRACT-MPI-001
    IO-H5-COLLECTIVE-001 IO-H5-ZEROOWNER-001 IO-H5-CORRUPT-001
    APP-THREAD-FUNNELED-001 APP-CANCEL-SAFEPOINT-001
  )
    slope_native_layout_test(
      "${test_id}" ss_test_app_mpi SUITE app LAYOUTS 1 2 4
      ENVIRONMENT
        "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python/src:${CMAKE_CURRENT_SOURCE_DIR}/generators/src:${CMAKE_CURRENT_SOURCE_DIR}"
    )
  endforeach()
endfunction()

function(slope_register_mesh_fe_tests)
  foreach(
    test_id
    MSH-UNIT-IDMAP-001 MSH-UNIT-LABEL-001 MSH-UNIT-SECTION-001 MSH-UNIT-LIFTING-001
    MSH-UNIT-FACET-001 MSH-UNIT-CURVED-FACET-001
  )
    slope_native_test("${test_id}" ss_test_mesh)
  endforeach()
  foreach(
    test_id
    FE-UNIT-BASIS-001 FE-REG-P4-DERIV-001 FE-UNIT-QUADRATURE-001 FE-NEG-QUADRATURE-001
    FE-UNIT-CURVED-001 FE-UNIT-FACET-001 FE-UNIT-PERMUTE-001 FE-UNIT-TRANSFER-001
  )
    slope_native_test("${test_id}" ss_test_fe)
  endforeach()

  # Distribution and FE fields must not depend on the rank count or on the
  # shared-point balancing policy.
  set(
    all_policies
    --mesh-shared-point-policy petsc
    --mesh-shared-point-policy stable_hash_p2
    --mesh-shared-point-policy stable_hash_high_order
  )
  foreach(test_id MSH-CONTRACT-DISTRIBUTE-001 FE-CONTRACT-MPI-001)
    slope_native_layout_test(
      "${test_id}" ss_test_mesh_field_mpi SUITE mesh LAYOUTS 1 2 4
      RUNNER_ARGS ${all_policies}
    )
  endforeach()
  slope_native_layout_test(
    MSH-NEG-GEOMETRY-001 ss_test_mesh_field_mpi SUITE mesh LAYOUTS 1 2
  )

  slope_native_test(
    MESH-DISTRIBUTED-JOIN-001 ss_test_mesh_join_mpi
    MPI 2 NO_ARGS ENVIRONMENT OMP_NUM_THREADS=1
  )
  slope_native_test(
    QAL-CONTRACT-BOUNDARY-TRACE-001 ss_test_boundary_trace AREA mesh NO_ARGS
  )
endfunction()

function(slope_register_material_tests)
  foreach(
    test_id
    MAT-UNIT-DAVIS-001 MAT-UNIT-BRANCH-001 MAT-UNIT-TANGENT-001 MAT-REG-ZEROPHI-001
    MAT-REG-EIGEN-001 MAT-UNIT-PLANE-001 MAT-UNIT-BATCH-001 MAT-NEG-INPUT-001
  )
    slope_native_test("${test_id}" ss_test_material)
  endforeach()
endfunction()

function(slope_register_assembly_tests)
  foreach(
    test_id
    ASM-UNIT-PATTERN-001 ASM-UNIT-ELEMENT-001 ASM-UNIT-LIFTING-001 ASM-UNIT-MATFREE-001
    ASM-PERF-MEMORY-001
  )
    slope_native_test("${test_id}" ss_test_assembly TIMEOUT 120)
  endforeach()

  if(SLOPE_ENABLE_OPENMP)
    slope_native_test(
      ASM-CONTRACT-MATFREE-OPENMP-001 ss_test_assembly
      LABELS openmp TIMEOUT 120 PROCESSORS 4
      ENVIRONMENT OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
    )
    set(options PROCESSORS 32 TIMEOUT 300)
    set(performance 1x1 2x1 4x1 1x4)
    set(matfree 2x1 4x1 2x4)
    set(row_owner 1x1 1x2 1x4 1x8)
    set(petsc_funnel 1x8)
    set(transaction 2x4 4x8)
    set(layout 1x1 2x1 4x1 1x2 1x4 1x8 2x4 4x8)
  else()
    set(options PROCESSORS 4 TIMEOUT 300)
    set(performance 1x1 2x1 4x1)
    set(matfree 2x1 4x1)
    set(row_owner 1x1)
    set(petsc_funnel 1x1)
    set(transaction 2x1 4x1)
    set(layout 1x1 2x1 4x1)
  endif()
  # Target and runner suite shared by the layout tests below.
  set(assembly_mpi ss_test_assembly_mpi SUITE assembly)
  slope_native_layout_test(ASM-UNIT-OVERLAP-001 ${assembly_mpi} LAYOUTS 1x1 2x1 4x1 ${options})
  slope_native_layout_test(
    ASM-CONTRACT-PERFORMANCE-001 ${assembly_mpi} LAYOUTS ${performance} ${options}
  )
  slope_native_layout_test(ASM-CONTRACT-COO-COLLECTIVE-001 ${assembly_mpi} LAYOUTS 4x1 ${options})
  slope_native_layout_test(ASM-CONTRACT-MATFREE-MPI-001 ${assembly_mpi} LAYOUTS ${matfree} ${options})
  slope_native_layout_test(ASM-CONTRACT-ROWOWNER-001 ${assembly_mpi} LAYOUTS ${row_owner} ${options})
  slope_native_layout_test(
    ASM-CONTRACT-PETSCFUNNEL-001 ${assembly_mpi} LAYOUTS ${petsc_funnel} ${options}
  )
  slope_native_layout_test(
    ASM-CONTRACT-TRANSACTION-001 ${assembly_mpi} LAYOUTS ${transaction} ${options}
  )
  slope_native_layout_test(ASM-CONTRACT-LAYOUT-001 ${assembly_mpi} LAYOUTS ${layout} ${options})
  slope_native_layout_test(ASM-RACE-MUST-001 ${assembly_mpi} LAYOUTS 1x1 2x1 4x1 ${options})
endfunction()

# Solver-stack regressions of one executable that takes a mode argument.
function(slope_register_solver_regression_tests)
  set(single_thread OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1)
  slope_native_test(
    PMG-REG-SPARSE-ROUTE-001 ss_test_solver_correctness
    MPI 4 ARGS sparse-pmg TIMEOUT 30 ENVIRONMENT ${single_thread}
  )
  foreach(ranks 2 4)
    slope_native_test(
      "SOLVER-REG-collective-${ranks}" ss_test_solver_correctness
      MPI ${ranks} ARGS collective TIMEOUT 30 ENVIRONMENT ${single_thread}
    )
  endforeach()
  foreach(capture none full)
    slope_native_test(
      "SOLVER-REG-workspace-${capture}" ss_test_solver_correctness
      MPI 2 TIMEOUT 45 ENVIRONMENT ${single_thread}
      ARGS workspace -malloc_debug -objects_dump all -review_capture "${capture}"
    )
  endforeach()
  foreach(
    mode
    signed minres projected preconditioned natural memory fixed augmented
    sensitivity stationary-fixed stationary-augmented quantized-control load-only
  )
    slope_native_test(
      "SOLVER-REG-${mode}" ss_test_solver_correctness
      ARGS "${mode}" TIMEOUT 30 ENVIRONMENT ${single_thread}
    )
  endforeach()
  slope_native_test(
    SOLVER-REG-signed-mpi ss_test_solver_correctness
    MPI 2 ARGS signed TIMEOUT 30 ENVIRONMENT ${single_thread}
  )
endfunction()

function(slope_register_linear_deflation_tests)
  foreach(
    test_id
    LIN-UNIT-POLICY-001 LIN-UNIT-ACCEPT-001 LIN-REG-HESSENBERG-001 LIN-UNIT-FALLBACK-001
    LIN-CONTRACT-OPTIONS-001 LIN-REG-KRYLOV-RECYCLE-001
    DEF-UNIT-PROJECTOR-001 DEF-REG-SIGN-001 DEF-UNIT-OBLIQUE-001 DEF-UNIT-DEPENDENT-001
    DEF-UNIT-ROLLBACK-001
  )
    slope_native_test("${test_id}" ss_test_linear_deflation)
  endforeach()
  foreach(test_id LIN-CONTRACT-MPI-001 LIN-CANCEL-001 DEF-CONTRACT-MPI-001)
    slope_native_layout_test(
      "${test_id}" ss_test_linear_deflation_mpi SUITE linear LAYOUTS 1 2 4
    )
  endforeach()
endfunction()

function(slope_register_pmg_tests)
  foreach(
    test_id
    PMG-UNIT-HIERARCHY-001 PMG-UNIT-GALERKIN-001 PMG-UNIT-VCYCLE-001
    PMG-UNIT-NULLSPACE-001
  )
    slope_native_test("${test_id}" ss_test_pmg TIMEOUT 90)
  endforeach()

  # Layouts RxT/A: R ranks with T threads, A of them active on coarse levels.
  set(active 5x1/2 7x1/3 9x1/4)
  if(NOT SLOPE_SANITIZER STREQUAL "address-undefined")
    list(APPEND active 72x1/64 96x1/64)
  endif()
  if(SLOPE_ENABLE_OPENMP)
    set(memory 1x1/1 2x1/2 4x8/4)
  else()
    set(memory 1x1/1 2x1/2 4x1/4)
  endif()
  set(pmg_mpi ss_test_pmg_mpi SUITE pmg)
  set(options PROCESSORS 96 TIMEOUT 900)
  slope_native_layout_test(PMG-CONTRACT-ACTIVE-001 ${pmg_mpi} LAYOUTS ${active} ${options})
  slope_native_layout_test(
    PMG-CONTRACT-REUSE-001 ${pmg_mpi} LAYOUTS 1x1/1 2x1/2 4x1/3 ${options}
  )
  slope_native_layout_test(
    PMG-CONTRACT-REDUNDANT-LU-001 ${pmg_mpi} LAYOUTS 2x1/2 4x1/4 8x1/8 ${options}
  )
  if(SLOPE_PETSC_HAVE_SUPERLU_DIST)
    slope_native_layout_test(
      PMG-CONTRACT-PARALLEL-LU-001 ${pmg_mpi} LAYOUTS 2x1/2 4x1/4 8x1/8 ${options}
    )
  endif()
  slope_native_layout_test(
    PMG-CONTRACT-PCMG-001 ${pmg_mpi} LAYOUTS 1x1/1 2x1/2 4x1/4 ${options}
  )
  slope_native_layout_test(
    PMG-NEG-LEVELFAIL-001 ${pmg_mpi} LAYOUTS 1x1/1 2x1/1 4x1/3 ${options}
  )
  slope_native_layout_test(PMG-PERF-MEMORY-001 ${pmg_mpi} LAYOUTS ${memory} ${options})
endfunction()

function(slope_register_nonlinear_analysis_tests)
  foreach(
    test_id
    NLS-UNIT-FIXED-001 NLS-UNIT-ADAPTIVE-GUARD-001 NLS-UNIT-BISECT-001
    NLS-REG-MONOTONE-001 NLS-UNIT-REGULARIZE-001 NLS-UNIT-SINGULAR-001
    NLS-CONTRACT-ROLLBACK-001
    ANA-REG-LL-DAMP-001 ANA-UNIT-LL-001 ANA-REG-DIRECT-COORD-001 ANA-UNIT-DIRECT-001
    ANA-UNIT-INDIRECT-001 ANA-UNIT-INDIRECT-ADAPTIVE-001
    ANA-UNIT-INDIRECT-EFFICIENCY-001
    ASM-REGION-BODY-FORCE-001
  )
    slope_native_test("${test_id}" ss_test_nonlinear_analysis TIMEOUT 120)
  endforeach()

  foreach(test_id NLS-CONTRACT-MPI-001 ANA-CONTRACT-RESTART-001)
    slope_native_layout_test(
      "${test_id}" ss_test_nonlinear_analysis_mpi SUITE nonlinear
      LAYOUTS 1x1 2x1 4x1 TIMEOUT 900
    )
  endforeach()
  set(layouts 1x1 2x1 4x1)
  if(SLOPE_ENABLE_OPENMP AND NOT SLOPE_SANITIZER STREQUAL "address-undefined")
    list(APPEND layouts 4x8)
  endif()
  slope_native_layout_test(
    ANA-CONTRACT-LAYOUT-001 ss_test_nonlinear_analysis_mpi SUITE nonlinear
    LAYOUTS ${layouts} TIMEOUT 900
  )
endfunction()

function(slope_register_hydraulic_coupling_tests)
  foreach(
    test_id
    HYD-UNIT-RAMP-001 HYD-REG-TETEDGE-001 HYD-UNIT-ELEMENT-001 HYD-PROP-JACOBIAN-001
    HYD-UNIT-HEAD-001 HYD-UNIT-FLUX-001 HYD-CONTRACT-NONSYM-001
    CPL-UNIT-SAME-001 CPL-UNIT-MODES-001 CPL-UNIT-IMMUTABLE-001
  )
    slope_native_test("${test_id}" ss_test_hydraulic_coupling TIMEOUT 120)
  endforeach()

  foreach(test_id HYD-CONTRACT-BALANCE-001 CPL-CONTRACT-DIFFERENT-001 CPL-NEG-COVERAGE-001)
    slope_native_layout_test(
      "${test_id}" ss_test_hydraulic_coupling_mpi SUITE hydraulic
      LAYOUTS 1x1 2x1 4x1 TIMEOUT 900
    )
  endforeach()
  if(SLOPE_ENABLE_OPENMP AND NOT SLOPE_SANITIZER STREQUAL "address-undefined")
    set(layouts 1x1 2x2 4x4 4x8)
  else()
    set(layouts 1x1 2x1 4x1)
  endif()
  slope_native_layout_test(
    CPL-CONTRACT-LAYOUT-001 ss_test_hydraulic_coupling_mpi SUITE hydraulic
    LAYOUTS ${layouts} TIMEOUT 900
  )
endfunction()

# Result and checkpoint writer.
function(slope_register_result_tests)
  foreach(
    test_id
    OUT-UNIT-STATE-001 OUT-UNIT-SCHEMA-001 OUT-UNIT-INDEX-001 OUT-UNIT-HISTORY-001
    OUT-UNIT-CHECKPOINT-001 OUT-NEG-COMPAT-001
  )
    slope_native_layout_test(
      "${test_id}" ss_test_result_checkpoint SUITE result LAYOUTS 1x1
      RUNNER_ARGS --direct TIMEOUT 900
    )
  endforeach()
  foreach(test_id OUT-CONTRACT-HDF5-001 OUT-PERF-IO-001)
    slope_native_layout_test(
      "${test_id}" ss_test_result_checkpoint SUITE result LAYOUTS 1x1 2x1 4x1
      RUNNER_ARGS --require-invariant TIMEOUT 900
    )
  endforeach()
  set(repartition 1x1 2x1 4x1)
  if(SLOPE_ENABLE_OPENMP AND NOT SLOPE_SANITIZER STREQUAL "address-undefined")
    set(repartition 1x1 2x2 4x4 4x8)
  endif()
  set(identical RUNNER_ARGS --require-invariant --require-checksum TIMEOUT 900)
  foreach(
    test_id
    OUT-CONTRACT-DIGEST-001 OUT-CONTRACT-ATOMIC-001 OUT-CONTRACT-TERMINAL-001
    OUT-CONTRACT-RESTART-001 OUT-CONTRACT-DEFL-001
  )
    slope_native_layout_test(
      "${test_id}" ss_test_result_checkpoint SUITE result LAYOUTS 1x1 2x1 4x1
      ${identical}
    )
  endforeach()
  slope_native_layout_test(
    OUT-CONTRACT-REPART-001 ss_test_result_checkpoint SUITE result
    LAYOUTS ${repartition} ${identical}
  )

  # Appends must not reread earlier states; publication reads each stored value
  # once to finalize and once to verify. Three ranks give uneven chunk owners.
  set(append_root "${CMAKE_CURRENT_BINARY_DIR}/test-artifacts/OUT-PERF-APPEND-001")
  add_test(
    NAME OUT-PERF-APPEND-001-SETUP
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${append_root}"
  )
  slope_test_labels(labels OUT-PERF-APPEND-001-SETUP "" native serial)
  set_tests_properties(
    OUT-PERF-APPEND-001-SETUP
    PROPERTIES FIXTURES_SETUP slope_output_artifacts LABELS "${labels}"
  )
  slope_native_test(
    OUT-PERF-APPEND-001 ss_test_result_append_io
    MPI 3 ARGS "${append_root}" TIMEOUT 120 ENVIRONMENT OMP_NUM_THREADS=1
  )
  # Checkpoints with more deflation columns than controller values.
  slope_native_test(
    OUT-CHECKPOINT-WIDE-BASIS-001 ss_test_checkpoint_basis
    MPI 2 ARGS "${append_root}" TIMEOUT 60 ENVIRONMENT OMP_NUM_THREADS=1
  )
  set_tests_properties(
    OUT-PERF-APPEND-001
    OUT-CHECKPOINT-WIDE-BASIS-001
    PROPERTIES FIXTURES_REQUIRED slope_output_artifacts
  )
endfunction()

# Comparisons of production kernels with independent references.
function(slope_register_oracle_tests)
  foreach(
    test_id
    FE-ORACLE-PRODUCT-DEGREE-001 HYD-ORACLE-GRAVITY-DEGREE-001
    FE-ORACLE-FACET-NORMAL-001 MAT-ORACLE-TANGENT-SCALE-001
  )
    slope_native_test("${test_id}" ss_test_kernel_oracles LABELS oracle TIMEOUT 120)
  endforeach()
  slope_native_test(
    ASM-ORACLE-NATURAL-ENERGY-001 ss_test_natural_energy LABELS oracle TIMEOUT 120
  )
  slope_native_test(
    ASM-ORACLE-CURVED-MATFREE-001 ss_test_curved_operator
    LABELS oracle TIMEOUT 300 ENVIRONMENT OMP_NUM_THREADS=2
  )
  slope_native_test(
    ASM-ORACLE-CURVED-MATFREE-MPI-001 ss_test_curved_operator
    MPI 2 ARGS ASM-ORACLE-CURVED-MATFREE-001
    LABELS oracle TIMEOUT 300 PROCESSORS 4
    ENVIRONMENT OMP_NUM_THREADS=2 OMP_PROC_BIND=false OPENBLAS_NUM_THREADS=1
  )
endfunction()

function(slope_register_native_tests)
  slope_register_build_native_tests()
  slope_register_protocol_native_tests()
  slope_register_core_tests()
  slope_register_mesh_fe_tests()
  slope_register_material_tests()
  slope_register_assembly_tests()
  slope_register_solver_regression_tests()
  slope_register_linear_deflation_tests()
  slope_register_pmg_tests()
  slope_register_nonlinear_analysis_tests()
  slope_register_hydraulic_coupling_tests()
  slope_register_result_tests()
  slope_register_oracle_tests()
endfunction()
