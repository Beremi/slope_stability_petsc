# SPDX-License-Identifier: BSD-2-Clause
#
# Registration of the Python (unittest) contract tests.
#
# slope_python_unittest(<id> TARGETS <module[.Class[.method]]>...
#                       [AREA <label>] [LABELS <label>...] [TIMEOUT <s>]
#                       [PROCESSORS <n>] [ENVIRONMENT <var=value>...])
# runs `python -m unittest <targets>` from the source tree with the in-tree
# Python packages on PYTHONPATH. The test is labelled with its area (see
# slope_test_area in TestRegistration.cmake), `python` and LABELS.

set(
  SLOPE_ASAN_MPI_ENVIRONMENT
  "OMPI_MCA_memory=set:^patcher;ASAN_OPTIONS=set:detect_leaks=0:halt_on_error=1"
)
set(
  SLOPE_ASAN_RESULT_ENVIRONMENT
  "OMPI_MCA_memory=set:^patcher;ASAN_OPTIONS=set:detect_leaks=0:halt_on_error=1;UBSAN_OPTIONS=set:halt_on_error=1:print_stacktrace=1"
)

function(slope_python_unittest test_id)
  cmake_parse_arguments(
    PARSE_ARGV 1 arg "" "AREA;TIMEOUT;PROCESSORS" "TARGETS;LABELS;ENVIRONMENT"
  )
  if(NOT arg_TARGETS)
    message(FATAL_ERROR "slope_python_unittest(${test_id}) requires TARGETS")
  endif()
  if(NOT arg_TIMEOUT)
    set(arg_TIMEOUT 300)
  endif()
  if(NOT arg_PROCESSORS)
    set(arg_PROCESSORS 1)
  endif()
  slope_test_labels(labels "${test_id}" "${arg_AREA}" python ${arg_LABELS})
  set(
    environment
    "PYTHONDONTWRITEBYTECODE=1"
    "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python/src:${CMAKE_CURRENT_SOURCE_DIR}/generators/src:${CMAKE_CURRENT_SOURCE_DIR}"
    ${arg_ENVIRONMENT}
  )
  add_test(
    NAME "${test_id}"
    COMMAND "${Python3_EXECUTABLE}" -m unittest ${arg_TARGETS} -v
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

function(slope_register_protocol_and_build_tests)
  set(protocol tests.protocol.test_protocol_contract.ProtocolContract)
  foreach(
    row
    "PRO-UNIT-CANON-001:test_canonical_valid_corpus"
    "PRO-UNIT-CANON-002:test_canonical_invalid_corpus"
    "PRO-UNIT-PATH-001:test_paths"
    "PRO-UNIT-ID-001:test_identity_mutation"
    "PRO-CROSS-CONFORMANCE-001:test_cross_language_conformance"
    "PRO-COMPAT-MATRIX-001:test_compatibility"
    "PRO-NEG-LOCALID-001:test_local_and_executable_controls_rejected"
  )
    string(REPLACE ":" ";" row "${row}")
    list(GET row 0 test_id)
    list(GET row 1 method)
    slope_python_unittest(
      "${test_id}" TARGETS "${protocol}.${method}"
      LABELS serial
      TIMEOUT 30
      ENVIRONMENT "SLOPE_SCHEMA_CHECK=$<TARGET_FILE:slope-schema-check>"
    )
  endforeach()

  # BLD-CONFIG-FAILCLOSED-001 re-runs configure; it must discover PETSc
  # through the pkg-config search path of the configure that created this
  # build tree, not through whatever the ctest environment happens to export.
  if(DEFINED ENV{PKG_CONFIG_PATH})
    set(
      SLOPE_CONFIGURE_PKG_CONFIG_PATH
      "$ENV{PKG_CONFIG_PATH}"
      CACHE INTERNAL "pkg-config search path of the configure environment"
    )
  endif()
  set(build_test_pkg_config "")
  if(NOT "${SLOPE_CONFIGURE_PKG_CONFIG_PATH}" STREQUAL "")
    set(build_test_pkg_config "PKG_CONFIG_PATH=${SLOPE_CONFIGURE_PKG_CONFIG_PATH}")
  endif()

  set(build tests.build.test_build_contract.BuildContract)
  foreach(
    row
    "BLD-UNIT-MANIFEST-001:test_build_manifest"
    "BLD-UNIT-INDEX-001:test_index_contract"
    "BLD-GEN-REPRO-001:test_generation_reproducible"
    "SOLVER-OPTIONS-DOC-001:test_solver_options_reference"
    "BLD-CONFIG-FAILCLOSED-001:test_configure_fail_closed"
    "BLD-INSTALL-RELOCATE-001:test_install_relocation"
    "BLD-LOCK-CONSISTENCY-001:test_lock_consistency"
    "BLD-REUSE-LICENSE-001:test_reuse_license"
  )
    string(REPLACE ":" ";" row "${row}")
    list(GET row 0 test_id)
    list(GET row 1 method)
    slope_python_unittest(
      "${test_id}" TARGETS "${build}.${method}"
      LABELS serial
      TIMEOUT 300
      ENVIRONMENT
        "SLOPE_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
        "SLOPE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
        "SLOPE_SCHEMA_CHECK=$<TARGET_FILE:slope-schema-check>"
        "SLOPE_ENGINE=$<TARGET_FILE:slope-engine>"
        "SLOPE_PROBE=$<TARGET_FILE:slope-probe>"
        "SLOPE_MPIEXEC=${MPIEXEC_EXECUTABLE}"
        "SLOPE_MPI_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
        "SLOPE_RUNTIME_LIBRARY_PATH=${PETSC_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
        "SLOPE_PYTHON=${Python3_EXECUTABLE}"
        ${build_test_pkg_config}
    )
  endforeach()
  set(documentation tests.build.test_documentation_contract.DocumentationContract)
  foreach(
    row
    "BLD-DOC-SPDX-001:test_spdx_headers"
    "BLD-DOC-HEADERS-001:test_header_documentation"
    "BLD-DOC-LINKS-001:test_markdown_links"
  )
    string(REPLACE ":" ";" row "${row}")
    list(GET row 0 test_id)
    list(GET row 1 method)
    slope_python_unittest(
      "${test_id}" TARGETS "${documentation}.${method}"
      LABELS serial
      TIMEOUT 60
    )
  endforeach()
  # The complete Doxygen check of include/ runs only where Doxygen exists.
  find_program(SLOPE_DOXYGEN_EXECUTABLE doxygen)
  if(SLOPE_DOXYGEN_EXECUTABLE)
    slope_python_unittest(
      BLD-DOC-DOXYGEN-001 TARGETS "${documentation}.test_doxygen_clean"
      LABELS serial
      TIMEOUT 300
      ENVIRONMENT "SLOPE_DOXYGEN=${SLOPE_DOXYGEN_EXECUTABLE}"
    )
  endif()
  if(SLOPE_SANITIZER STREQUAL "address-undefined")
    set_tests_properties(
      BLD-INSTALL-RELOCATE-001
      PROPERTIES ENVIRONMENT_MODIFICATION "${SLOPE_ASAN_MPI_ENVIRONMENT}"
    )
  endif()
endfunction()

function(slope_register_preparation_python_tests)
  set(authoring tests.authoring.test_authoring_contract.AuthoringContract)
  foreach(
    row
    "AUT-UNIT-TYPES-001:test_strict_types_and_immutability"
    "AUT-UNIT-UNITS-001:test_units"
    "AUT-UNIT-SPANS-001:test_spans_and_accumulated_validation"
    "AUT-UNIT-REGISTRY-001:test_registered_models"
    "AUT-NEG-SEMANTIC-001:test_semantic_failures_precede_compilation"
    "AUT-PROP-ROUNDTRIP-001:test_roundtrip_identity"
  )
    string(REPLACE ":" ";" row "${row}")
    list(GET row 0 test_id)
    list(GET row 1 method)
    slope_python_unittest(
      "${test_id}" TARGETS "${authoring}.${method}"
      LABELS serial
      TIMEOUT 120
    )
  endforeach()

  set(compilation tests.compilation.test_compilation_contract.CompilationContract)
  foreach(
    row
    "CMP-PRESCRIBED-SATURATION-001:test_prescribed_saturation"
    "CMP-IMPORT-GMSH-001:test_gmsh_import"
    "CMP-IMPORT-UNIFORM-REFINE-001:test_uniform_simplex_refinement"
    "CMP-IMPORT-REFERENCE-001:test_reference_import"
    "CMP-UNIT-CANON-001:test_canonical_mesh_reordered_input"
    "CMP-MESH-ID-001:test_stable_ids"
    "CMP-MESH-SUPPORT-001:test_support_and_material_coverage"
    "CMP-GEO-MATRIX-001:test_geometry_matrix"
    "CMP-GEO-INVALID-001:test_invalid_geometry"
    "CMP-BC-SEMANTIC-001:test_boundary_semantics"
    "CMP-PKG-DETERMINISM-001:test_package_determinism"
    "CMP-PKG-SINGLE-WRITE-001:test_single_payload_build"
    "CMP-PKG-ATOMIC-001:test_atomic_publication"
    "CMP-E2E-NEWPROBLEM-001:test_new_problem_data_only"
  )
    string(REPLACE ":" ";" row "${row}")
    list(GET row 0 test_id)
    list(GET row 1 method)
    slope_python_unittest(
      "${test_id}" TARGETS "${compilation}.${method}"
      LABELS serial
      TIMEOUT 120
    )
  endforeach()

  set(resolution tests.resolution.test_resolution_contract.ResolutionContract)
  foreach(
    row
    "RES-UNIT-PRECEDENCE-001:test_precedence"
    "RES-UNIT-SEEPAGE-001:test_pure_seepage_profile_controls_hydraulic_solve"
    "RES-UNIT-SEEPAGE-COUPLED-001:test_coupled_seepage_profile_does_not_replace_mechanics_policy"
    "RES-PMG-DYNAMIC-HIERARCHY-001:test_pmg_profile_family_resolves_from_actual_fine_order"
    "RES-UNIT-INHERIT-001:test_inheritance_is_forbidden"
    "RES-UNIT-SWEEP-001:test_cartesian_sweeps"
    "RES-UNIT-RESOURCE-001:test_resource_resolution"
    "RES-SOLVER-COMPAT-001:test_solver_compatibility"
    "RES-SOLVER-OPTIONS-001:test_solver_options"
    "RES-ACCEPTED-STATE-LIMIT-001:test_accepted_state_limit"
    "RES-UNIT-TRACE-001:test_complete_trace_and_identity"
  )
    string(REPLACE ":" ";" row "${row}")
    list(GET row 0 test_id)
    list(GET row 1 method)
    slope_python_unittest(
      "${test_id}" TARGETS "${resolution}.${method}"
      LABELS serial
      TIMEOUT 120
    )
  endforeach()

  set(launch tests.launch.test_launch_contract)
  slope_python_unittest(
    LCH-UNIT-LAUNCH-001
    TARGETS
      "${launch}.LaunchLayoutContract"
      "${launch}.LaunchEnvironmentContract"
      "${launch}.BoundedProcessContract"
    LABELS serial
    TIMEOUT 120
  )
  slope_python_unittest(
    QAL-CONTRACT-RESOURCES-001 TARGETS "${launch}.ExecutionResourceContract"
    AREA launch
    LABELS serial
    TIMEOUT 120
  )
  slope_python_unittest(
    LCH-CONTRACT-BINDING-001 TARGETS "${launch}.BindingProbeContract"
    LABELS mpi
    PROCESSORS 8
    TIMEOUT 120
    ENVIRONMENT
      "SLOPE_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "SLOPE_MPI_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
  )
endfunction()

function(slope_register_engine_python_tests)
  set(engine tests.app.test_engine_contract.EngineContract)
  set(
    engine_environment
    "SLOPE_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
    "SLOPE_ENGINE=$<TARGET_FILE:slope-engine>"
    "SLOPE_PROBE=$<TARGET_FILE:slope-probe>"
    "SLOPE_MPIEXEC=${MPIEXEC_EXECUTABLE}"
    "SLOPE_MPI_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
    "SLOPE_RUNTIME_LIBRARY_PATH=${PETSC_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
  )
  foreach(
    row
    "APP-UNIT-CLI-001:test_cli"
    "APP-SOLVER-DETAIL-001:test_live_solver_detail"
    "APP-ASSEMBLY-MODE-001:test_mechanics_assembly_mode"
    "APP-ACCEPTED-STATE-LIMIT-001:test_continuation_state_limit"
    "APP-VALIDATE-LAYOUT-001:test_validate_layout"
    "APP-PRESCRIBED-SATURATION-001:test_prescribed_saturation_native"
    "APP-PRESCRIBED-SATURATION-NEG-001:test_prescribed_saturation_native_rejection"
    "APP-CONTRACT-NOPYTHON-001:test_no_python"
    "E2E-FOUNDATION-001:test_foundation"
  )
    string(REPLACE ":" ";" row "${row}")
    list(GET row 0 test_id)
    list(GET row 1 method)
    slope_python_unittest(
      "${test_id}" TARGETS "${engine}.${method}"
      LABELS integration
      PROCESSORS 4
      TIMEOUT 300
      ENVIRONMENT ${engine_environment}
    )
  endforeach()
  set_tests_properties(
    APP-PRESCRIBED-SATURATION-001 PROPERTIES PROCESSORS 32 RUN_SERIAL TRUE
  )
  slope_python_unittest(
    MECHANICS-REG-FCG-001 TARGETS tests.native.test_mechanics_fcg
    AREA linear
    LABELS integration mpi
    PROCESSORS 2
    TIMEOUT 45
    ENVIRONMENT ${engine_environment}
  )
  slope_python_unittest(
    PMG-REG-CURVED-P4-G4-001 TARGETS tests.native.test_pmg_curved
    LABELS integration mpi
    PROCESSORS 2
    TIMEOUT 45
    ENVIRONMENT ${engine_environment}
  )
  slope_python_unittest(
    E2E-RESTART-001 TARGETS "${engine}.test_restart_matrix"
    LABELS restart mpi hybrid
    PROCESSORS 4
    TIMEOUT 900
    ENVIRONMENT ${engine_environment}
  )
  slope_python_unittest(
    APP-VOLUME-RULE-001 TARGETS "${engine}.test_volume_quadrature_rule"
    LABELS integration mpi
    PROCESSORS 2
    TIMEOUT 600
    ENVIRONMENT ${engine_environment}
  )
  slope_python_unittest(
    E2E-CANCEL-001 TARGETS "${engine}.test_cancel_restart"
    LABELS restart mpi
    PROCESSORS 2
    TIMEOUT 600
    ENVIRONMENT ${engine_environment}
  )
  if(SLOPE_SANITIZER STREQUAL "address-undefined")
    set_tests_properties(
      APP-VALIDATE-LAYOUT-001
      APP-CONTRACT-NOPYTHON-001
      E2E-FOUNDATION-001
      PROPERTIES ENVIRONMENT_MODIFICATION "${SLOPE_ASAN_MPI_ENVIRONMENT}"
    )
  endif()

  slope_python_unittest(
    QAL-CONTRACT-RESTART-001 TARGETS tests.restart.test_restart_contract
    AREA restart
    LABELS serial
    TIMEOUT 120
  )
endfunction()

function(slope_register_result_python_tests)
  set(result tests.result.test_result_contract.ResultContract)
  set(visualization tests.visualization.test_visualization_contract.VisualizationContract)
  set(
    rows
    "RDR-UNIT-OPEN-001:${result}.test_open_and_integrity"
    "RDR-UNIT-LAZY-001:${result}.test_lazy_selections"
    "RDR-UNIT-ALIGN-001:${result}.test_stable_identity_alignment"
    "RDR-CMP-SCALAR-001:${result}.test_scalar_comparator"
    "RDR-CMP-FIELD-001:${result}.test_field_comparator"
    "RDR-CMP-CONSERVATION-001:${result}.test_conservation_comparator"
    "RDR-CMP-TRAJECTORY-001:${result}.test_trajectory_comparator"
    "RDR-GOLD-PROMOTE-001:${result}.test_golden_promotion"
    "VIS-UNIT-DERIVED-001:${visualization}.test_derived_artifacts_do_not_mutate_source"
    "VIS-UNIT-PERMUTE-001:${visualization}.test_vtk_permutations"
    "VIS-UNIT-PROJECTION-001:${visualization}.test_named_projection"
    "VIS-ROUNDTRIP-2D-001:${visualization}.test_roundtrip_2d"
    "VIS-ROUNDTRIP-3D-001:${visualization}.test_roundtrip_3d"
    "VIS-UNIT-REPORT-001:${visualization}.test_report_labels_and_evidence_status"
    "VIS-NEG-INCOMPLETE-001:${visualization}.test_incomplete_and_failed_are_unavailable"
  )
  foreach(row IN LISTS rows)
    string(REPLACE ":" ";" row "${row}")
    list(GET row 0 test_id)
    list(GET row 1 target)
    slope_python_unittest(
      "${test_id}" TARGETS "${target}"
      LABELS serial
      TIMEOUT 300
      ENVIRONMENT "SLOPE_RESULT_TEST_EXE=$<TARGET_FILE:ss_test_result_checkpoint>"
    )
    if(SLOPE_SANITIZER STREQUAL "address-undefined")
      # Reader and visualization tests create sanitized native result
      # fixtures as child processes.
      set_tests_properties(
        "${test_id}"
        PROPERTIES ENVIRONMENT_MODIFICATION "${SLOPE_ASAN_RESULT_ENVIRONMENT}"
      )
    endif()
  endforeach()
endfunction()

function(slope_register_catalog_tests)
  set(catalog_module tests.catalog.test_catalog_contract)
  slope_python_unittest(
    CAT-MESH-SHA-001 TARGETS "${catalog_module}.CatalogMeshContract"
    LABELS serial
    TIMEOUT 120
  )
  slope_python_unittest(
    CAT-COMPILE-001 TARGETS "${catalog_module}.CatalogCompileContract"
    LABELS serial
    TIMEOUT 600
  )
  slope_python_unittest(
    QAL-CONTRACT-HYD-UNITS-001 TARGETS tests.catalog.test_hydraulic_catalog
    AREA catalog
    LABELS serial
    TIMEOUT 120
  )
  slope_python_unittest(
    E2E-CATALOG-SMOKE-001 TARGETS tests.catalog.test_catalog_smoke
    LABELS smoke mpi
    PROCESSORS 4
    TIMEOUT 900
    ENVIRONMENT
      "SLOPE_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
      "SLOPE_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "SLOPE_MPI_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      "SLOPE_RUNTIME_LIBRARY_PATH=${PETSC_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
  )
  if(SLOPE_SANITIZER STREQUAL "address-undefined")
    set_tests_properties(
      E2E-CATALOG-SMOKE-001
      PROPERTIES ENVIRONMENT_MODIFICATION "${SLOPE_ASAN_MPI_ENVIRONMENT}"
    )
  endif()
endfunction()

function(slope_register_python_tests)
  slope_register_protocol_and_build_tests()
  slope_register_preparation_python_tests()
  slope_register_engine_python_tests()
  slope_register_result_python_tests()
  slope_register_catalog_tests()
endfunction()
