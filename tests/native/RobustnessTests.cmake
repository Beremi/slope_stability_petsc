# SPDX-License-Identifier: BSD-2-Clause
#
# Registration of the collective robustness tests: failures injected on one
# rank must end every rank with the same status.
function(slope_register_collective_robustness_tests)
  add_executable(ss_test_assembly_error_collective
    tests/native/assembly_error_collective_test.c)
  target_include_directories(ss_test_assembly_error_collective PRIVATE src/assembly)
  target_link_libraries(ss_test_assembly_error_collective PRIVATE slope_native)
  target_link_options(ss_test_assembly_error_collective PRIVATE
    "LINKER:--wrap=MPI_Allgather" "LINKER:--wrap=MPI_Allreduce" "LINKER:--wrap=MPI_Bcast")
  slope_project_warnings(ss_test_assembly_error_collective)
  slope_instrument(ss_test_assembly_error_collective)
  add_test(NAME ASM-COLLECTIVE-ERROR-001
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
      "$<TARGET_FILE:ss_test_assembly_error_collective>")

  add_executable(ss_test_monitor_collective_allocation
    tests/native/monitor_collective_allocation_test.c)
  target_include_directories(ss_test_monitor_collective_allocation PRIVATE src/app)
  target_link_libraries(ss_test_monitor_collective_allocation PRIVATE slope_native)
  target_link_options(ss_test_monitor_collective_allocation PRIVATE "LINKER:--wrap=VecDuplicate")
  slope_project_warnings(ss_test_monitor_collective_allocation)
  slope_instrument(ss_test_monitor_collective_allocation)
  add_test(NAME APP-COLLECTIVE-ALLOCATION-001
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
      "$<TARGET_FILE:ss_test_monitor_collective_allocation>")

  set_tests_properties(ASM-COLLECTIVE-ERROR-001 APP-COLLECTIVE-ALLOCATION-001
    PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1" PROCESSORS 2 TIMEOUT 30)
  set_tests_properties(ASM-COLLECTIVE-ERROR-001
    PROPERTIES LABELS "assembly;native;mpi")
  set_tests_properties(APP-COLLECTIVE-ALLOCATION-001
    PROPERTIES LABELS "app;native;mpi")

  slope_python_unittest(APP-HYDRAULIC-ASSEMBLY-MODE-001
    TARGETS tests.app.test_hydraulic_assembly_mode.HydraulicAssemblyMode
    LABELS integration mpi
    PROCESSORS 2 TIMEOUT 120
    ENVIRONMENT
      "SLOPE_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
      "SLOPE_ENGINE=$<TARGET_FILE:slope-engine>"
      "SLOPE_PROBE=$<TARGET_FILE:slope-probe>"
      "SLOPE_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "SLOPE_MPI_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      "SLOPE_RUNTIME_LIBRARY_PATH=${PETSC_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
endfunction()
