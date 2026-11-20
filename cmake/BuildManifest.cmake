# SPDX-License-Identifier: BSD-2-Clause

function(slope_json_escape output input)
  string(REPLACE "\\" "\\\\" value "${input}")
  string(REPLACE "\"" "\\\"" value "${value}")
  string(REPLACE "\n" "\\n" value "${value}")
  string(REPLACE "\r" "\\r" value "${value}")
  string(REPLACE "\t" "\\t" value "${value}")
  set("${output}" "${value}" PARENT_SCOPE)
endfunction()

set(
  SLOPE_SOURCE_REVISION
  ""
  CACHE STRING
  "Source commit recorded when the source tree is not a Git checkout"
)

# Identity of a source tree without Git metadata (for example `git archive`
# output or a release tarball). The revision comes from SLOPE_SOURCE_REVISION
# or from cmake/source-revision.txt, which `git archive` fills through the
# export-subst attribute. The tree is identified by its content: the patch
# digest hashes every regular file below the source directory (excluding a
# build directory placed inside it), so a modified tarball yields a
# different source identity. Such a tree is never reported as dirty because
# there is no recorded base to compare with.
function(
  slope_archive_source_identity
  revision_output
  dirty_output
  digest_output
  describe_output
)
  set(revision "${SLOPE_SOURCE_REVISION}")
  set(describe "")
  set(stamp_path "${CMAKE_CURRENT_SOURCE_DIR}/cmake/source-revision.txt")
  if(revision STREQUAL "" AND EXISTS "${stamp_path}")
    file(STRINGS "${stamp_path}" stamp_lines)
    foreach(line IN LISTS stamp_lines)
      if(line MATCHES "^revision: *([0-9a-f]+)$")
        set(revision "${CMAKE_MATCH_1}")
      elseif(line MATCHES "^describe: *(.+)$")
        set(describe "${CMAKE_MATCH_1}")
      endif()
    endforeach()
  endif()
  string(LENGTH "${revision}" revision_length)
  if(
    NOT revision MATCHES "^[0-9a-f]+$"
    OR NOT (revision_length EQUAL 40 OR revision_length EQUAL 64)
  )
    message(
      FATAL_ERROR
      "The source tree has no Git metadata and no recorded revision. "
      "Configure from a Git checkout or `git archive` output, or pass "
      "-DSLOPE_SOURCE_REVISION=<full commit id>."
    )
  endif()
  if(describe STREQUAL "" OR describe MATCHES "[$]Format:")
    string(SUBSTRING "${revision}" 0 12 describe)
  endif()
  file(
    GLOB_RECURSE source_files
    LIST_DIRECTORIES false
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/*"
  )
  file(
    RELATIVE_PATH binary_relative
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}"
  )
  list(SORT source_files)
  set(projection "")
  foreach(relative IN LISTS source_files)
    string(FIND "${relative}" "${binary_relative}/" binary_position)
    if(NOT binary_relative MATCHES "^[.][.]" AND binary_position EQUAL 0)
      continue()
    endif()
    file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${relative}" file_digest)
    string(APPEND projection "${relative}:${file_digest}\n")
  endforeach()
  string(SHA256 tree_digest "archive\n${projection}")
  set("${revision_output}" "${revision}" PARENT_SCOPE)
  set("${dirty_output}" false PARENT_SCOPE)
  set("${digest_output}" "${tree_digest}" PARENT_SCOPE)
  set("${describe_output}" "${describe}" PARENT_SCOPE)
endfunction()

# Source identity of a Git checkout: HEAD, whether the worktree is dirty and a
# digest of the working-tree patch (including untracked files). Trees without
# their own Git metadata use slope_archive_source_identity().
function(
  slope_source_identity
  revision_output
  dirty_output
  digest_output
  describe_output
)
  find_package(Git QUIET)
  set(toplevel "")
  if(GIT_EXECUTABLE)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" rev-parse --show-toplevel
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      RESULT_VARIABLE toplevel_result
      OUTPUT_VARIABLE toplevel
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT toplevel_result EQUAL 0)
      set(toplevel "")
    endif()
  endif()
  set(is_checkout FALSE)
  if(NOT toplevel STREQUAL "")
    file(REAL_PATH "${toplevel}" toplevel_real)
    file(REAL_PATH "${CMAKE_CURRENT_SOURCE_DIR}" source_real)
    if(toplevel_real STREQUAL source_real)
      set(is_checkout TRUE)
    endif()
  endif()
  if(NOT is_checkout)
    # Not a checkout of this project (for example an extracted archive,
    # possibly below some other repository).
    slope_archive_source_identity(revision dirty patch_digest describe)
    set("${revision_output}" "${revision}" PARENT_SCOPE)
    set("${dirty_output}" "${dirty}" PARENT_SCOPE)
    set("${digest_output}" "${patch_digest}" PARENT_SCOPE)
    set("${describe_output}" "${describe}" PARENT_SCOPE)
    return()
  endif()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE revision_result
    OUTPUT_VARIABLE revision
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT revision_result EQUAL 0)
    message(FATAL_ERROR "Unable to read the Git source revision")
  endif()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" describe --always --tags HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE describe_result
    OUTPUT_VARIABLE describe
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT describe_result EQUAL 0)
    message(FATAL_ERROR "Unable to describe the source revision")
  endif()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" status --porcelain=v1 --untracked-files=all
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE status_result
    OUTPUT_VARIABLE status
  )
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" diff --binary HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE diff_result
    OUTPUT_VARIABLE patch
  )
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" ls-files --others --exclude-standard
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE untracked_result
    OUTPUT_VARIABLE untracked
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT status_result EQUAL 0 OR NOT diff_result EQUAL 0 OR NOT untracked_result EQUAL 0)
    message(FATAL_ERROR "Unable to compute source identity")
  endif()
  set(untracked_projection "")
  if(NOT untracked STREQUAL "")
    string(REPLACE "\n" ";" untracked_files "${untracked}")
    list(SORT untracked_files)
    foreach(relative IN LISTS untracked_files)
      file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${relative}" file_digest)
      string(APPEND untracked_projection "${relative}:${file_digest}\n")
    endforeach()
  endif()
  string(SHA256 patch_digest "${status}\n${patch}\n${untracked_projection}")
  if(status STREQUAL "")
    set(dirty false)
  else()
    set(dirty true)
  endif()
  set("${revision_output}" "${revision}" PARENT_SCOPE)
  set("${dirty_output}" "${dirty}" PARENT_SCOPE)
  set("${digest_output}" "${patch_digest}" PARENT_SCOPE)
  set("${describe_output}" "${describe}" PARENT_SCOPE)
endfunction()

function(slope_write_build_manifest)
  slope_source_identity(
    source_revision source_dirty source_patch_digest source_describe
  )
  file(
    SHA256
    "${CMAKE_CURRENT_SOURCE_DIR}/generated/manifest.json"
    generated_manifest_sha256
  )
  file(
    SHA256
    "${CMAKE_CURRENT_SOURCE_DIR}/environments/spack.lock"
    native_lock_sha256
  )
  file(
    SHA256
    "${CMAKE_CURRENT_SOURCE_DIR}/uv.lock"
    python_lock_sha256
  )
  file(
    SHA256
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/yyjson/src/yyjson.c"
    yyjson_c_sha256
  )
  file(
    SHA256
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/yyjson/src/yyjson.h"
    yyjson_h_sha256
  )
  string(
    SHA256
    yyjson_source_sha256
    "yyjson.c:${yyjson_c_sha256}\nyyjson.h:${yyjson_h_sha256}\n"
  )
  if(CMAKE_C_COMPILER_TARGET STREQUAL "")
    execute_process(
      COMMAND "${CMAKE_C_COMPILER}" -dumpmachine
      RESULT_VARIABLE target_result
      OUTPUT_VARIABLE compiler_target_raw
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT target_result EQUAL 0)
      message(FATAL_ERROR "Unable to determine compiler target")
    endif()
  else()
    set(compiler_target_raw "${CMAKE_C_COMPILER_TARGET}")
  endif()
  slope_json_escape(compiler_path "${CMAKE_C_COMPILER}")
  slope_json_escape(compiler_id "${CMAKE_C_COMPILER_ID}")
  slope_json_escape(compiler_version "${CMAKE_C_COMPILER_VERSION}")
  slope_json_escape(compiler_target "${compiler_target_raw}")
  slope_json_escape(build_type "${CMAKE_BUILD_TYPE}")
  string(TOUPPER "${CMAKE_BUILD_TYPE}" build_type_upper)
  slope_json_escape(
    c_flags
    "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${build_type_upper}}"
  )
  slope_json_escape(mpi_version "${MPI_C_VERSION}")
  slope_json_escape(petsc_version "${PETSC_VERSION}")
  slope_json_escape(hdf5_version "${HDF5_VERSION}")
  slope_json_escape(generator "${CMAKE_GENERATOR}")
  slope_json_escape(build_profile "${SLOPE_BUILD_PROFILE}")
  slope_json_escape(cmake_version "${CMAKE_VERSION}")
  slope_json_escape(link_flags "${CMAKE_EXE_LINKER_FLAGS}")
  slope_json_escape(source_describe_json "${source_describe}")
  if(SLOPE_ENABLE_OPENMP)
    set(openmp_json true)
    slope_json_escape(openmp_version "${OpenMP_C_VERSION}")
  else()
    set(openmp_json false)
    set(openmp_version "")
  endif()
  if(SLOPE_PETSC_HAVE_OPENMP)
    set(petsc_openmp_json true)
  else()
    set(petsc_openmp_json false)
  endif()
  if(SLOPE_PETSC_USE_OPENMP_KERNELS)
    set(petsc_openmp_kernels_json true)
  else()
    set(petsc_openmp_kernels_json false)
  endif()
  if(SLOPE_PETSC_HAVE_KOKKOS)
    set(petsc_kokkos_json true)
  else()
    set(petsc_kokkos_json false)
  endif()
  if(SLOPE_PETSC_HAVE_MUMPS)
    set(petsc_mumps_json true)
  else()
    set(petsc_mumps_json false)
  endif()
  if(SLOPE_PETSC_HAVE_HWLOC)
    set(petsc_hwloc_json true)
  else()
    set(petsc_hwloc_json false)
  endif()
  if(SLOPE_PETSC_HAVE_OPENBLAS_THREAD_CONTROL)
    set(petsc_openblas_thread_control_json true)
  else()
    set(petsc_openblas_thread_control_json false)
  endif()
  if(SLOPE_SUPERLU_DIST_HAVE_OPENMP)
    set(superlu_dist_openmp_json true)
  else()
    set(superlu_dist_openmp_json false)
  endif()
  slope_json_escape(blas_libraries "${BLAS_LIBRARIES}")
  slope_json_escape(lapack_libraries "${LAPACK_LIBRARIES}")
  slope_json_escape(petsc_cflags_extra "${SLOPE_PETSC_CFLAGS_EXTRA}")
  if(SLOPE_PETSC_HAVE_SUPERLU_DIST)
    set(petsc_superlu_dist_json true)
  else()
    set(petsc_superlu_dist_json false)
  endif()
  set(
    source_projection
    "{\"describe\":\"${source_describe_json}\",\"dirty\":${source_dirty},\"generated_manifest_sha256\":\"${generated_manifest_sha256}\",\"patch_digest\":\"sha256:${source_patch_digest}\",\"revision\":\"${source_revision}\",\"submodules\":[]}"
  )
  if(SLOPE_PROTOCOL_ONLY)
    set(hydraulic_pmg_budget_json false)
  else()
    set(hydraulic_pmg_budget_json true)
  endif()
  set(
    features_projection
    "{\"application_openmp\":${openmp_json},\"hdf5_parallel\":true,\"hydraulic_pmg_budget\":${hydraulic_pmg_budget_json},\"openmp\":${openmp_json},\"petsc_hdf5\":true,\"petsc_hwloc\":${petsc_hwloc_json},\"petsc_hypre\":true,\"petsc_kokkos\":${petsc_kokkos_json},\"petsc_mumps\":${petsc_mumps_json},\"petsc_openblas_thread_control\":${petsc_openblas_thread_control_json},\"petsc_openmp\":${petsc_openmp_json},\"petsc_openmp_kernels\":${petsc_openmp_kernels_json},\"petsc_partitioner\":true,\"petsc_superlu_dist\":${petsc_superlu_dist_json},\"superlu_dist_openmp\":${superlu_dist_openmp_json}}"
  )
  string(SHA256 source_digest "${source_projection}")
  set(
    dependency_projection
    "{\"features\":${features_projection},\"hdf5\":\"${hdf5_version}\",\"linear_algebra\":{\"blas_libraries\":\"${blas_libraries}\",\"lapack_libraries\":\"${lapack_libraries}\"},\"mpi\":\"${mpi_version}\",\"native_lock_sha256\":\"${native_lock_sha256}\",\"openmp\":\"${openmp_version}\",\"petsc\":\"${petsc_version}\",\"petsc_configuration\":{\"cflags_extra\":\"${petsc_cflags_extra}\"},\"petsc_int_bits\":${SLOPE_PETSC_INDEX_BITS},\"python_lock_sha256\":\"${python_lock_sha256}\",\"yyjson\":{\"revision\":\"8b4a38dc994a110abaec8a400615567bd996105f\",\"source_sha256\":\"${yyjson_source_sha256}\",\"version\":\"0.12.0\"}}"
  )
  string(SHA256 dependency_digest "${dependency_projection}")
  set(
    build_projection
    "{\"build_type\":\"${build_type}\",\"cmake\":\"${cmake_version}\",\"compiler\":{\"flags\":\"${c_flags}\",\"id\":\"${compiler_id}\",\"path\":\"${compiler_path}\",\"target\":\"${compiler_target}\",\"version\":\"${compiler_version}\"},\"dependency_id\":\"sha256:${dependency_digest}\",\"generator\":\"${generator}\",\"link_flags\":\"${link_flags}\",\"openmp\":${openmp_json},\"profile\":\"${build_profile}\",\"sanitizer\":\"${SLOPE_SANITIZER}\",\"source_id\":\"sha256:${source_digest}\"}"
  )
  string(SHA256 build_digest "${build_projection}")
  set(SLOPE_SOURCE_ID "sha256:${source_digest}" CACHE INTERNAL "")
  set(SLOPE_DEPENDENCY_ID "sha256:${dependency_digest}" CACHE INTERNAL "")
  set(SLOPE_BUILD_ID "sha256:${build_digest}" CACHE INTERNAL "")
  set(
    manifest
    "{\"build\":${build_projection},\"build_id\":\"${SLOPE_BUILD_ID}\",\"dependency\":${dependency_projection},\"dependency_id\":\"${SLOPE_DEPENDENCY_ID}\",\"features\":${features_projection},\"kind\":\"slope_stability.build\",\"schema_version\":{\"major\":1,\"minor\":0,\"patch\":0},\"source\":${source_projection},\"source_id\":\"${SLOPE_SOURCE_ID}\"}\n"
  )
  set(SLOPE_BUILD_MANIFEST_PATH "${CMAKE_CURRENT_BINARY_DIR}/build-manifest.json" CACHE INTERNAL "")
  file(WRITE "${SLOPE_BUILD_MANIFEST_PATH}" "${manifest}")
  set(SLOPE_SOURCE_ID "${SLOPE_SOURCE_ID}" PARENT_SCOPE)
  set(SLOPE_DEPENDENCY_ID "${SLOPE_DEPENDENCY_ID}" PARENT_SCOPE)
  set(SLOPE_BUILD_ID "${SLOPE_BUILD_ID}" PARENT_SCOPE)
  set(SLOPE_BUILD_MANIFEST_PATH "${SLOPE_BUILD_MANIFEST_PATH}" PARENT_SCOPE)
endfunction()
