/* Single-process tests of native arithmetic, indexing, diagnostics, strict
 * JSON input, package integrity and the application lifecycle.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/app/lifecycle.h"
#include "slope_internal/core/core.h"
#include "slope_internal/core/sort.h"
#include "slope_internal/diagnostics/diagnostics.h"
#include "slope_internal/io/input.h"

#include <hdf5.h>
#include <mpi.h>
#include <petscsystypes.h>

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define CHECK(expression)                                                              \
  do {                                                                                 \
    if (!(expression)) {                                                               \
      fprintf(stderr, "%s:%d: assertion failed: %s\n", __FILE__, __LINE__,             \
              #expression);                                                            \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

static int test_arithmetic(void) {
  size_t value = 0U;
  const size_t dimensions[] = {2U, 3U, 5U};
  CHECK(ss_checked_add_size(2U, 3U, &value) == SS_OK && value == 5U);
  CHECK(ss_checked_add_size(SIZE_MAX, 1U, &value) == SS_ERROR_OVERFLOW);
  CHECK(ss_checked_mul_size(7U, 9U, &value) == SS_OK && value == 63U);
  CHECK(ss_checked_mul_size(SIZE_MAX, 2U, &value) == SS_ERROR_OVERFLOW);
  CHECK(ss_checked_product_size(dimensions, 3U, &value) == SS_OK && value == 30U);
  CHECK(ss_checked_product_size(NULL, 1U, &value) == SS_ERROR_INVALID_ARGUMENT);
  CHECK(ss_checked_align_size(65U, 64U, &value) == SS_OK && value == 128U);
  CHECK(ss_checked_align_size(SIZE_MAX, 64U, &value) == SS_ERROR_OVERFLOW);
  CHECK(ss_checked_align_size(1U, 3U, &value) == SS_ERROR_INVALID_ARGUMENT);
  return 0;
}

static int test_index(void) {
  size_t size_value = 0U;
  hsize_t hdf5_value = 0U;
  PetscInt petsc_value = 0;
  MPI_Count mpi_value = 0;
  CHECK(sizeof(SsEntityId) == sizeof(uint64_t));
  CHECK(ss_checked_u64_to_size(0U, &size_value) == SS_OK && size_value == 0U);
  CHECK(ss_checked_u64_to_hsize(UINT64_MAX, &hdf5_value) == SS_OK);
  CHECK(ss_checked_u64_to_petsc((uint64_t)PETSC_MAX_INT, &petsc_value) == SS_OK);
  CHECK(ss_checked_u64_to_petsc((uint64_t)PETSC_MAX_INT + 1U, &petsc_value) ==
        SS_ERROR_OVERFLOW);
  CHECK(ss_checked_u64_to_mpi_count((uint64_t)INT_MAX, &mpi_value) == SS_OK);
  CHECK(ss_checked_u64_to_size(1U, NULL) == SS_ERROR_INVALID_ARGUMENT);
  return 0;
}

static int test_lifecycle(void) {
  SsLifecycle lifecycle;
  ss_lifecycle_init(&lifecycle);
  CHECK(lifecycle.state == SS_ENGINE_CREATED);
  CHECK(ss_lifecycle_transition(&lifecycle, SS_ENGINE_RUNNING) ==
        SS_ERROR_INVALID_STATE);
  CHECK(ss_lifecycle_transition(&lifecycle, SS_ENGINE_VALIDATED) == SS_OK);
  CHECK(ss_lifecycle_transition(&lifecycle, SS_ENGINE_RUNNING) == SS_OK);
  CHECK(ss_lifecycle_transition(&lifecycle, SS_ENGINE_SUCCEEDED) == SS_OK);
  CHECK(ss_lifecycle_transition(&lifecycle, SS_ENGINE_FAILED) ==
        SS_ERROR_INVALID_STATE);
  CHECK(ss_lifecycle_destroy(&lifecycle) == SS_OK);
  CHECK(ss_lifecycle_destroy(&lifecycle) == SS_OK);
  CHECK(lifecycle.state == SS_ENGINE_DESTROYED && lifecycle.destroy_count == 1U);
  return 0;
}

static int test_identity(void) {
  SsIdentityInputs inputs = {
      "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
      "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
      "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
      "machine-a",
      "launcher-a",
  };
  char first[SS_ID_TEXT_CAPACITY];
  char second[SS_ID_TEXT_CAPACITY];
  CHECK(ss_identity_link(&inputs, first) == SS_OK);
  CHECK(strncmp(first, "sha256:", 7U) == 0 && strlen(first) == 71U);
  inputs.machine_id = "machine-b";
  CHECK(ss_identity_link(&inputs, second) == SS_OK);
  CHECK(strcmp(first, second) != 0);
  inputs.machine_id = "machine-a";
  CHECK(ss_identity_link(&inputs, second) == SS_OK);
  CHECK(strcmp(first, second) == 0);
  return 0;
}

static int test_error_order(void) {
  SsErrorRecord records[4] = {0};
  records[0].valid = 1;
  records[0].key = (SsErrorKey){2U, 3U, 4U, 5U, 1U, 0U};
  records[1].valid = 1;
  records[1].key = (SsErrorKey){2U, 3U, 4U, 5U, 0U, 1U};
  records[2].valid = 1;
  records[2].key = (SsErrorKey){1U, 9U, 9U, 9U, 9U, 2U};
  records[3].valid = 0;
  CHECK(ss_error_compare(&records[2], &records[0]) < 0);
  CHECK(ss_error_compare(&records[2], &records[1]) < 0);
  CHECK(ss_error_compare(&records[2], &records[3]) < 0);
  records[2].valid = 0;
  CHECK(ss_error_compare(&records[1], &records[0]) < 0);
  CHECK(ss_error_compare(&records[1], &records[2]) < 0);
  CHECK(ss_error_compare(&records[1], &records[3]) < 0);
  return 0;
}

static const char valid_problem[] =
    "{\"boundary_rules\":{},\"coupling\":{},\"created_by\":{\"source_id\":\"x\","
    "\"tool\":\"compiler\",\"tool_version\":\"1\"},\"extensions\":{},"
    "\"id\":\"sha256:"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\","
    "\"identity\":{},\"initial_state\":{},\"kind\":\"slope_stability.compiled_"
    "problem\","
    "\"materials\":{},\"mesh\":{\"coordinate_order\":1,\"dimension\":2,"
    "\"uniform_refinement_levels\":1},"
    "\"operators\":{\"analysis\":\"limit_load.work_controlled\","
    "\"operator_class\":\"nonsymmetric\"},\"outputs\":{},"
    "\"payloads\":{\"problem_hdf5\":{"
    "\"byte_count\":3,\"layout_id\":\"compiled-problem-v1\","
    "\"path\":\"problem.h5\",\"sha256\":\""
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\"}},"
    "\"physics\":{},\"provenance\":{},\"schema_version\":{\"major\":1,"
    "\"minor\":0,\"patch\":0},\"spaces\":{\"displacement\":{\"order\":1}},"
    "\"supports\":{},\"units\":{}}";

static int test_json_strict(void) {
  SsJsonLimits limits = ss_json_default_limits();
  SsProblemSpec problem = {0};
  SsErrorRecord error = {0};
  const char duplicate[] = "{\"kind\":\"slope_stability.compiled_problem\","
                           "\"kind\":\"slope_stability.compiled_problem\"}";
  const char unknown[] =
      "{\"kind\":\"slope_stability.compiled_problem\",\"unexpected\":1}";
  const char nonfinite[] = "{\"kind\":\"slope_stability.compiled_problem\",\"x\":NaN}";
  const char coercion[] =
      "{\"kind\":\"slope_stability.compiled_problem\",\"schema_version\":"
      "{\"major\":\"1\"}}";
  char nested_unknown[sizeof(valid_problem) + 32U];
  char wrong_object[sizeof(valid_problem)];
  const char *mesh_end = strstr(valid_problem, "\"uniform_refinement_levels\":1}");
  char *units;
  size_t mesh_prefix;

  /* Empty and singleton ranges need neither storage nor a comparator. */
  ss_sort(NULL, 0U, sizeof(size_t), NULL);
  ss_sort(NULL, 1U, sizeof(size_t), NULL);

  CHECK(mesh_end != NULL);
  mesh_prefix =
      (size_t)(mesh_end - valid_problem) + strlen("\"uniform_refinement_levels\":1");
  CHECK(mesh_prefix <= (size_t)INT_MAX);
  CHECK(snprintf(nested_unknown, sizeof(nested_unknown), "%.*s,\"unexpected\":1%s",
                 (int)mesh_prefix, valid_problem, valid_problem + mesh_prefix) > 0);
  memcpy(wrong_object, valid_problem, sizeof(valid_problem));
  units = strstr(wrong_object, "\"units\":{}");
  CHECK(units != NULL);
  units[strlen("\"units\":")] = '[';
  units[strlen("\"units\":[")] = ']';
  CHECK(ss_io_parse_problem((const unsigned char *)valid_problem, strlen(valid_problem),
                            &limits, &problem, &error) == SS_OK);
  {
    char prescribed[sizeof(valid_problem) + 180U];
    CHECK(snprintf(prescribed, sizeof(prescribed),
                   "{\"prescribed_saturation\":{\"axis\":0,\"coordinates\":[0],"
                   "\"levels\":[1],\"length_unit\":\"m\"},%s",
                   valid_problem + 1) > 0);
    CHECK(ss_io_parse_problem((const unsigned char *)prescribed, strlen(prescribed),
                              &limits, &problem, &error) == SS_OK);
  }
  CHECK(strcmp(problem.payload_path, "problem.h5") == 0);
  CHECK(problem.uniform_refinement_levels == 1U);
  CHECK(ss_io_parse_problem((const unsigned char *)duplicate, strlen(duplicate),
                            &limits, &problem, &error) == SS_ERROR_DUPLICATE);
  CHECK(ss_io_parse_problem((const unsigned char *)unknown, strlen(unknown), &limits,
                            &problem, &error) == SS_ERROR_SCHEMA);
  CHECK(ss_io_parse_problem((const unsigned char *)nonfinite, strlen(nonfinite),
                            &limits, &problem, &error) == SS_ERROR_JSON);
  CHECK(ss_io_parse_problem((const unsigned char *)coercion, strlen(coercion), &limits,
                            &problem, &error) == SS_ERROR_SCHEMA);
  CHECK(ss_io_parse_problem((const unsigned char *)nested_unknown,
                            strlen(nested_unknown), &limits, &problem,
                            &error) == SS_ERROR_SCHEMA);
  CHECK(ss_io_parse_problem((const unsigned char *)wrong_object, strlen(wrong_object),
                            &limits, &problem, &error) == SS_ERROR_SCHEMA);
  CHECK(ss_io_parse_problem((const unsigned char *)"{} trailing", 11U, &limits,
                            &problem, &error) == SS_ERROR_JSON);
  return 0;
}

static int test_json_limits(void) {
  SsJsonLimits limits = ss_json_default_limits();
  SsProblemSpec problem = {0};
  SsErrorRecord error = {0};
  const char deep[] = "{\"a\":{\"b\":{\"c\":{}}}}";
  const char long_string[] = "{\"a\":\"123456789\"}";
  const char many[] = "{\"a\":[1,2,3]}";
  const char large_integer[] = "{\"a\":101}";
  CHECK(limits.maximum_bytes > strlen(valid_problem));
  limits.maximum_bytes = 8U;
  CHECK(ss_io_parse_problem((const unsigned char *)valid_problem, strlen(valid_problem),
                            &limits, &problem, &error) == SS_ERROR_LIMIT);
  limits = ss_json_default_limits();
  limits.maximum_depth = 2U;
  CHECK(ss_io_parse_problem((const unsigned char *)deep, strlen(deep), &limits,
                            &problem, &error) == SS_ERROR_LIMIT);
  limits = ss_json_default_limits();
  limits.maximum_string_bytes = 4U;
  CHECK(ss_io_parse_problem((const unsigned char *)long_string, strlen(long_string),
                            &limits, &problem, &error) == SS_ERROR_LIMIT);
  limits = ss_json_default_limits();
  limits.maximum_array_items = 2U;
  CHECK(ss_io_parse_problem((const unsigned char *)many, strlen(many), &limits,
                            &problem, &error) == SS_ERROR_LIMIT);
  limits = ss_json_default_limits();
  limits.maximum_integer = 100U;
  CHECK(ss_io_parse_problem((const unsigned char *)large_integer, strlen(large_integer),
                            &limits, &problem, &error) == SS_ERROR_LIMIT);
  return 0;
}

static int write_bytes(const char *path, const void *data, size_t size) {
  FILE *stream = fopen(path, "wb");
  if (stream == NULL) {
    return 0;
  }
  if (size != 0U && fwrite(data, 1U, size, stream) != size) {
    (void)fclose(stream);
    return 0;
  }
  return fclose(stream) == 0;
}

/* Create a fresh directory <stem>-XXXXXX under $TMPDIR, or /tmp when TMPDIR is
 * unset or empty, and store its path in path[PATH_MAX]. */
static int make_temporary_directory(const char *stem, char path[PATH_MAX]) {
  const char *base = getenv("TMPDIR");
  int written;
  if (base == NULL || base[0] == '\0') {
    base = "/tmp";
  }
  written = snprintf(path, PATH_MAX, "%s/%s-XXXXXX", base, stem);
  return written > 0 && written < PATH_MAX && mkdtemp(path) != NULL;
}

/* Store <directory>/<leaf> in path[PATH_MAX]; fails on truncation. */
static int join_path(const char *directory, const char *leaf, char path[PATH_MAX]) {
  const int written = snprintf(path, PATH_MAX, "%s/%s", directory, leaf);
  return written > 0 && written < PATH_MAX;
}

static int test_package_integrity(void) {
  char root[PATH_MAX];
  char payload[PATH_MAX];
  const unsigned char bytes[] = {1U, 2U, 3U};
  SsProblemSpec problem = {0};
  SsErrorRecord error = {0};
  CHECK(make_temporary_directory("ss-core-io-integrity", root));
  CHECK(join_path(root, "problem.h5", payload));
  CHECK(write_bytes(payload, bytes, sizeof(bytes)) != 0);
  strcpy(problem.payload_path, "problem.h5");
  problem.payload_bytes = sizeof(bytes);
  CHECK(ss_sha256_bytes(bytes, sizeof(bytes), problem.payload_sha256) == SS_OK);
  CHECK(ss_io_verify_payload(root, &problem, &error) == SS_OK);
  CHECK(write_bytes(payload, "\1\2\4", 3U) != 0);
  CHECK(ss_io_verify_payload(root, &problem, &error) == SS_ERROR_INTEGRITY);
  CHECK(unlink(payload) == 0 && rmdir(root) == 0);
  return 0;
}

static int test_package_path(void) {
  char root[PATH_MAX];
  char outside[PATH_MAX];
  char path[PATH_MAX];
  SsErrorRecord error = {0};
  CHECK(make_temporary_directory("ss-core-io-path", root));
  CHECK(make_temporary_directory("ss-core-io-outside", outside));
  CHECK(ss_io_validate_package_path(root, "problem.h5", &error) == SS_ERROR_NOT_FOUND);
  CHECK(ss_io_validate_package_path(root, "../escape", &error) ==
        SS_ERROR_INVALID_PATH);
  CHECK(join_path(root, "problem.h5", path));
  CHECK(symlink(outside, path) == 0);
  CHECK(ss_io_validate_package_path(root, "problem.h5", &error) ==
        SS_ERROR_INVALID_PATH);
  CHECK(unlink(path) == 0);
  CHECK(mkdir(path, 0700) == 0);
  CHECK(ss_io_validate_package_path(root, "problem.h5", &error) ==
        SS_ERROR_INVALID_PATH);
  CHECK(rmdir(path) == 0 && rmdir(root) == 0 && rmdir(outside) == 0);
  return 0;
}

static int test_config(void) {
  SsJsonLimits limits = ss_json_default_limits();
  SsNativeConfig config = {0};
  SsErrorRecord error = {0};
  char mutable_problem[sizeof(valid_problem)];
  memcpy(mutable_problem, valid_problem, sizeof(valid_problem));
  CHECK(ss_io_configure((const unsigned char *)mutable_problem, strlen(mutable_problem),
                        NULL, 0U, &limits, &config, &error) == SS_OK);
  mutable_problem[2] = 'X';
  CHECK(ss_io_config_is_sealed(&config) != 0);
  CHECK(strcmp(ss_io_problem(&config)->payload_path, "problem.h5") == 0);
  CHECK(ss_io_configure((const unsigned char *)valid_problem, strlen(valid_problem),
                        NULL, 0U, &limits, &config, &error) == SS_ERROR_INVALID_STATE);
  ss_io_config_destroy(&config);
  ss_io_config_destroy(&config);
  return 0;
}

typedef int (*TestFunction)(void);

typedef struct {
  const char *id;
  TestFunction function;
} TestEntry;

int main(int argc, char **argv) {
  static const TestEntry tests[] = {
      {"COR-UNIT-ARITH-001", test_arithmetic},
      {"COR-UNIT-INDEX-001", test_index},
      {"COR-UNIT-LIFECYCLE-001", test_lifecycle},
      {"DIA-UNIT-IDENTITY-001", test_identity},
      {"DIA-UNIT-ERROR-001", test_error_order},
      {"IO-JSON-STRICT-001", test_json_strict},
      {"IO-JSON-LIMITS-001", test_json_limits},
      {"IO-PKG-INTEGRITY-001", test_package_integrity},
      {"IO-PKG-PATH-001", test_package_path},
      {"IO-UNIT-CONFIG-001", test_config},
      {"APP-LIFE-STATE-001", test_lifecycle},
  };
  size_t index;
  if (argc != 2) {
    fputs("usage: ss_test_core_io_lifecycle TEST-ID\n", stderr);
    return 2;
  }
  for (index = 0U; index < sizeof(tests) / sizeof(tests[0]); ++index) {
    if (strcmp(argv[1], tests[index].id) == 0) {
      return tests[index].function();
    }
  }
  fprintf(stderr, "unknown core, I/O or lifecycle test: %s\n", argv[1]);
  return 2;
}
