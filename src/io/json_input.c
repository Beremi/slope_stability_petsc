/* Strict parsing of problem.json and resolved-run.json into the sealed
 * native configuration.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/analysis/analysis.h"
#include "slope_internal/core/sort.h"
#include "slope_internal/core/util.h"
#include "slope_internal/io/input.h"

#include "yyjson.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void ss_io_error(SsErrorRecord *error, SsStatus status, const char *code,
                        const char *message) {
  ss_error_set(error, status, (SsErrorKey){1U, 3U, 0U, 0U, 0U, 0U}, code, message);
}

SsJsonLimits ss_json_default_limits(void) {
  SsJsonLimits limits = {
      64U * 1024U * 1024U, 128U,          1024U * 1024U,
      1024U * 1024U,       1024U * 1024U, UINT64_MAX,
  };
  return limits;
}

static SsStatus ss_json_prescan(const unsigned char *data, size_t size,
                                const SsJsonLimits *limits, SsErrorRecord *error) {
  size_t index;
  size_t depth = 0U;
  size_t string_size = 0U;
  int in_string = 0;
  int escaped = 0;
  if (size > limits->maximum_bytes) {
    ss_io_error(error, SS_ERROR_LIMIT, "JSON_BYTE_LIMIT", "JSON byte limit exceeded");
    return SS_ERROR_LIMIT;
  }
  if (size >= 3U && data[0] == 0xefU && data[1] == 0xbbU && data[2] == 0xbfU) {
    ss_io_error(error, SS_ERROR_JSON, "JSON_BOM", "JSON BOM is forbidden");
    return SS_ERROR_JSON;
  }
  for (index = 0U; index < size; ++index) {
    unsigned char byte = data[index];
    if (in_string != 0) {
      if (escaped != 0) {
        escaped = 0;
        ++string_size;
      } else if (byte == (unsigned char)'\\') {
        escaped = 1;
      } else if (byte == (unsigned char)'"') {
        in_string = 0;
        string_size = 0U;
      } else {
        ++string_size;
      }
      if (string_size > limits->maximum_string_bytes) {
        ss_io_error(error, SS_ERROR_LIMIT, "JSON_STRING_LIMIT",
                    "JSON string limit exceeded");
        return SS_ERROR_LIMIT;
      }
    } else if (byte == (unsigned char)'"') {
      in_string = 1;
      string_size = 0U;
    } else if (byte == (unsigned char)'{' || byte == (unsigned char)'[') {
      ++depth;
      if (depth > limits->maximum_depth) {
        ss_io_error(error, SS_ERROR_LIMIT, "JSON_DEPTH_LIMIT",
                    "JSON nesting limit exceeded");
        return SS_ERROR_LIMIT;
      }
    } else if (byte == (unsigned char)'}' || byte == (unsigned char)']') {
      if (depth != 0U) {
        --depth;
      }
    }
  }
  return SS_OK;
}

static int ss_same_key(const yyjson_val *left, const yyjson_val *right) {
  size_t left_size = yyjson_get_len((yyjson_val *)left);
  size_t right_size = yyjson_get_len((yyjson_val *)right);
  const char *left_text = yyjson_get_str((yyjson_val *)left);
  const char *right_text = yyjson_get_str((yyjson_val *)right);
  return left_size == right_size &&
         (left_size == 0U || memcmp(left_text, right_text, left_size) == 0);
}

static int ss_compare_key_pointer(const void *left_pointer, const void *right_pointer) {
  const yyjson_val *left = *(const yyjson_val *const *)left_pointer;
  const yyjson_val *right = *(const yyjson_val *const *)right_pointer;
  size_t left_size = yyjson_get_len((yyjson_val *)left);
  size_t right_size = yyjson_get_len((yyjson_val *)right);
  size_t common_size = left_size < right_size ? left_size : right_size;
  int comparison = memcmp(yyjson_get_str((yyjson_val *)left),
                          yyjson_get_str((yyjson_val *)right), common_size);
  if (comparison != 0) {
    return comparison;
  }
  return left_size < right_size ? -1 : left_size > right_size ? 1 : 0;
}

static SsStatus ss_validate_tree(yyjson_val *value, size_t depth,
                                 const SsJsonLimits *limits, SsErrorRecord *error) {
  if (depth > limits->maximum_depth) {
    ss_io_error(error, SS_ERROR_LIMIT, "JSON_DEPTH_LIMIT",
                "JSON nesting limit exceeded");
    return SS_ERROR_LIMIT;
  }
  if (yyjson_is_obj(value)) {
    size_t count = yyjson_obj_size(value);
    yyjson_val **keys;
    yyjson_obj_iter iterator;
    yyjson_val *key;
    size_t index = 0U;
    if (count > limits->maximum_object_members) {
      ss_io_error(error, SS_ERROR_LIMIT, "JSON_MEMBER_LIMIT",
                  "JSON object member limit exceeded");
      return SS_ERROR_LIMIT;
    }
    keys = count == 0U ? NULL : (yyjson_val **)calloc(count, sizeof(*keys));
    if (count != 0U && keys == NULL) {
      return SS_ERROR_OUT_OF_MEMORY;
    }
    iterator = yyjson_obj_iter_with(value);
    while ((key = yyjson_obj_iter_next(&iterator)) != NULL) {
      if (index >= count) {
        free(keys);
        return SS_ERROR_INTERNAL;
      }
      keys[index++] = key;
    }
    if (index != count) {
      free(keys);
      return SS_ERROR_INTERNAL;
    }
    if (count > 1U) {
      ss_sort(keys, count, sizeof(*keys), ss_compare_key_pointer);
    }
    for (index = 1U; index < count; ++index) {
      if (ss_same_key(keys[index - 1U], keys[index]) != 0) {
        free(keys);
        ss_io_error(error, SS_ERROR_DUPLICATE, "JSON_DUPLICATE_KEY",
                    "duplicate JSON object key");
        return SS_ERROR_DUPLICATE;
      }
    }
    iterator = yyjson_obj_iter_with(value);
    while ((key = yyjson_obj_iter_next(&iterator)) != NULL) {
      SsStatus status =
          ss_validate_tree(yyjson_obj_iter_get_val(key), depth + 1U, limits, error);
      if (status != SS_OK) {
        free(keys);
        return status;
      }
    }
    free(keys);
  } else if (yyjson_is_arr(value)) {
    size_t index;
    size_t maximum;
    yyjson_val *item;
    if (yyjson_arr_size(value) > limits->maximum_array_items) {
      ss_io_error(error, SS_ERROR_LIMIT, "JSON_ARRAY_LIMIT",
                  "JSON array item limit exceeded");
      return SS_ERROR_LIMIT;
    }
    yyjson_arr_foreach(value, index, maximum, item) {
      SsStatus status = ss_validate_tree(item, depth + 1U, limits, error);
      if (status != SS_OK) {
        return status;
      }
    }
  } else if (yyjson_is_uint(value)) {
    if (yyjson_get_uint(value) > limits->maximum_integer) {
      ss_io_error(error, SS_ERROR_LIMIT, "JSON_INTEGER_LIMIT",
                  "JSON integer exceeds configured magnitude");
      return SS_ERROR_LIMIT;
    }
  } else if (yyjson_is_sint(value)) {
    int64_t signed_value = yyjson_get_sint(value);
    uint64_t magnitude = signed_value < 0
                             ? (uint64_t)(-(signed_value + INT64_C(1))) + UINT64_C(1)
                             : (uint64_t)signed_value;
    if (magnitude > limits->maximum_integer) {
      ss_io_error(error, SS_ERROR_LIMIT, "JSON_INTEGER_LIMIT",
                  "JSON integer exceeds configured magnitude");
      return SS_ERROR_LIMIT;
    }
  }
  return SS_OK;
}

static SsStatus ss_read_document(const unsigned char *data, size_t size,
                                 const SsJsonLimits *limits, yyjson_doc **doc,
                                 SsErrorRecord *error) {
  yyjson_read_err read_error = {0};
  SsStatus status;
  if (data == NULL || limits == NULL || doc == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *doc = NULL;
  status = ss_json_prescan(data, size, limits, error);
  if (status != SS_OK) {
    return status;
  }
  *doc = yyjson_read_opts((char *)(void *)data, size, YYJSON_READ_NOFLAG, NULL,
                          &read_error);
  if (*doc == NULL) {
    char message[SS_ERROR_MESSAGE_CAPACITY];
    (void)snprintf(message, sizeof(message), "strict JSON parse failed at %zu",
                   read_error.pos);
    ss_io_error(error, SS_ERROR_JSON, "JSON_INVALID", message);
    return SS_ERROR_JSON;
  }
  if (!yyjson_is_obj(yyjson_doc_get_root(*doc))) {
    yyjson_doc_free(*doc);
    *doc = NULL;
    ss_io_error(error, SS_ERROR_SCHEMA, "JSON_ROOT_TYPE",
                "manifest root must be an object");
    return SS_ERROR_SCHEMA;
  }
  status = ss_validate_tree(yyjson_doc_get_root(*doc), 1U, limits, error);
  if (status != SS_OK) {
    yyjson_doc_free(*doc);
    *doc = NULL;
  }
  return status;
}

static int ss_name_in(const char *name, const char *const *names, size_t count) {
  size_t index;
  for (index = 0U; index < count; ++index) {
    if (strcmp(name, names[index]) == 0) {
      return 1;
    }
  }
  return 0;
}

static int ss_extension_name_valid(const char *name) {
  const char *cursor;
  if (name == NULL ||
      !(strncmp(name, "org.", 4U) == 0 || strncmp(name, "com.", 4U) == 0 ||
        strncmp(name, "net.", 4U) == 0 || strncmp(name, "io.", 3U) == 0)) {
    return 0;
  }
  cursor = strchr(name, '.') + 1U;
  if (*cursor == '\0') {
    return 0;
  }
  for (; *cursor != '\0'; ++cursor) {
    if (!((*cursor >= 'a' && *cursor <= 'z') || (*cursor >= '0' && *cursor <= '9') ||
          *cursor == '.' || *cursor == '-')) {
      return 0;
    }
  }
  return 1;
}

static SsStatus ss_validate_object_shape(yyjson_val *object,
                                         const char *const *required,
                                         size_t required_count,
                                         const char *const *allowed,
                                         size_t allowed_count, SsErrorRecord *error) {
  yyjson_obj_iter iterator;
  yyjson_val *key;
  size_t index;
  if (!yyjson_is_obj(object)) {
    ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_OBJECT",
                "required JSON object has wrong type");
    return SS_ERROR_SCHEMA;
  }
  iterator = yyjson_obj_iter_with(object);
  while ((key = yyjson_obj_iter_next(&iterator)) != NULL) {
    const char *name = yyjson_get_str(key);
    if (name == NULL || ss_name_in(name, allowed, allowed_count) == 0) {
      ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_UNKNOWN_FIELD",
                  "unknown manifest field");
      return SS_ERROR_SCHEMA;
    }
  }
  for (index = 0U; index < required_count; ++index) {
    if (yyjson_obj_get(object, required[index]) == NULL) {
      ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_REQUIRED_FIELD",
                  "required manifest field is missing");
      return SS_ERROR_SCHEMA;
    }
  }
  return SS_OK;
}

static SsStatus ss_validate_common_objects(yyjson_val *root,
                                           const char *const *object_names,
                                           size_t object_count, SsErrorRecord *error) {
  static const char *const created_by_shape[] = {
      "source_id",
      "tool",
      "tool_version",
  };
  yyjson_val *created_by;
  yyjson_val *extensions;
  yyjson_obj_iter iterator;
  yyjson_val *key;
  size_t index;
  for (index = 0U; index < object_count; ++index) {
    if (!yyjson_is_obj(yyjson_obj_get(root, object_names[index]))) {
      ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_OBJECT",
                  "manifest object section has wrong type");
      return SS_ERROR_SCHEMA;
    }
  }
  created_by = yyjson_obj_get(root, "created_by");
  if (ss_validate_object_shape(created_by, created_by_shape, 3U, created_by_shape, 3U,
                               error) != SS_OK) {
    return SS_ERROR_SCHEMA;
  }
  for (index = 0U; index < 3U; ++index) {
    yyjson_val *value = yyjson_obj_get(created_by, created_by_shape[index]);
    if (!yyjson_is_str(value) || yyjson_get_len(value) == 0U) {
      ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_CREATED_BY",
                  "created_by strings must be non-empty");
      return SS_ERROR_SCHEMA;
    }
  }
  extensions = yyjson_obj_get(root, "extensions");
  iterator = yyjson_obj_iter_with(extensions);
  while ((key = yyjson_obj_iter_next(&iterator)) != NULL) {
    if (!ss_extension_name_valid(yyjson_get_str(key))) {
      ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_EXTENSION_NAME",
                  "extension key is outside the approved namespace");
      return SS_ERROR_SCHEMA;
    }
  }
  return SS_OK;
}

static SsStatus ss_copy_string(yyjson_val *value, char *destination, size_t capacity,
                               SsErrorRecord *error) {
  const char *text;
  size_t size;
  if (!yyjson_is_str(value)) {
    ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_STRING",
                "manifest string has wrong type");
    return SS_ERROR_SCHEMA;
  }
  text = yyjson_get_str(value);
  size = yyjson_get_len(value);
  if (size >= capacity) {
    ss_io_error(error, SS_ERROR_LIMIT, "SCHEMA_STRING_LIMIT",
                "manifest string exceeds native capacity");
    return SS_ERROR_LIMIT;
  }
  memcpy(destination, text, size);
  destination[size] = '\0';
  return SS_OK;
}

static SsStatus ss_schema_version(yyjson_val *object, uint32_t *major, uint32_t *minor,
                                  uint32_t *patch, SsErrorRecord *error) {
  yyjson_val *major_value;
  yyjson_val *minor_value;
  yyjson_val *patch_value;
  static const char *const keys[] = {"major", "minor", "patch"};
  if (ss_validate_object_shape(object, keys, 3U, keys, 3U, error) != SS_OK) {
    return SS_ERROR_SCHEMA;
  }
  major_value = yyjson_obj_get(object, "major");
  minor_value = yyjson_obj_get(object, "minor");
  patch_value = yyjson_obj_get(object, "patch");
  if (!yyjson_is_uint(major_value) || !yyjson_is_uint(minor_value) ||
      !yyjson_is_uint(patch_value) || yyjson_get_uint(major_value) > UINT32_MAX ||
      yyjson_get_uint(minor_value) > UINT32_MAX ||
      yyjson_get_uint(patch_value) > UINT32_MAX) {
    ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_VERSION_TYPE",
                "schema version must contain unsigned integers");
    return SS_ERROR_SCHEMA;
  }
  *major = (uint32_t)yyjson_get_uint(major_value);
  *minor = (uint32_t)yyjson_get_uint(minor_value);
  *patch = (uint32_t)yyjson_get_uint(patch_value);
  if (*major != 1U) {
    ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_MAJOR_INCOMPATIBLE",
                "unsupported schema major version");
    return SS_ERROR_SCHEMA;
  }
  return SS_OK;
}

static SsStatus ss_schema_version_array(yyjson_val *array, uint32_t *major,
                                        SsErrorRecord *error) {
  yyjson_val *value;
  size_t index;
  size_t count;
  if (!yyjson_is_arr(array) || yyjson_arr_size(array) != 3U) {
    ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_VERSION_TYPE",
                "referenced schema version must be a three-integer array");
    return SS_ERROR_SCHEMA;
  }
  yyjson_arr_foreach(array, index, count, value) {
    if (!yyjson_is_uint(value) || yyjson_get_uint(value) > UINT32_MAX) {
      ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_VERSION_TYPE",
                  "referenced schema version must contain unsigned integers");
      return SS_ERROR_SCHEMA;
    }
    if (index == 0U) {
      *major = (uint32_t)yyjson_get_uint(value);
    }
  }
  if (*major != 1U) {
    ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_MAJOR_INCOMPATIBLE",
                "unsupported referenced schema major version");
    return SS_ERROR_SCHEMA;
  }
  return SS_OK;
}

SsStatus ss_io_parse_problem(const unsigned char *data, size_t size,
                             const SsJsonLimits *limits, SsProblemSpec *problem,
                             SsErrorRecord *error) {
  static const char *const required[] = {
      "boundary_rules", "coupling",      "created_by", "extensions", "id",
      "identity",       "initial_state", "kind",       "materials",  "mesh",
      "operators",      "outputs",       "payloads",   "physics",    "provenance",
      "schema_version", "spaces",        "supports",   "units",
  };
  static const char *const allowed[] = {
      "boundary_rules",
      "coupling",
      "created_by",
      "extensions",
      "id",
      "identity",
      "initial_state",
      "kind",
      "materials",
      "mesh",
      "operators",
      "outputs",
      "payloads",
      "physics",
      "prescribed_saturation",
      "provenance",
      "schema_version",
      "spaces",
      "supports",
      "units",
  };
  static const char *const mesh_required[] = {"coordinate_order", "dimension"};
  static const char *const object_names[] = {
      "boundary_rules", "coupling",      "created_by", "extensions",
      "identity",       "initial_state", "materials",  "mesh",
      "operators",      "outputs",       "payloads",   "physics",
      "provenance",     "spaces",        "supports",   "units",
  };
  static const char *const mesh_allowed[] = {
      "cell_count", "cell_type",      "coordinate_node_count",     "coordinate_order",
      "dimension",  "logical_digest", "uniform_refinement_levels",
  };
  static const char *const payloads_shape[] = {"problem_hdf5"};
  static const char *const payload_required[] = {
      "byte_count",
      "layout_id",
      "path",
      "sha256",
  };
  static const char *const payload_allowed[] = {
      "byte_count", "layout_id", "logical_digest", "media_type", "path", "sha256",
  };
  yyjson_doc *doc = NULL;
  yyjson_val *root;
  yyjson_val *kind;
  yyjson_val *mesh;
  yyjson_val *payloads;
  yyjson_val *payload;
  yyjson_val *spaces;
  yyjson_val *displacement;
  yyjson_val *operators;
  yyjson_val *value;
  SsStatus status;
  if (problem == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(problem, 0, sizeof(*problem));
  status = ss_read_document(data, size, limits, &doc, error);
  if (status != SS_OK) {
    return status;
  }
  root = yyjson_doc_get_root(doc);
  status =
      ss_validate_object_shape(root, required, sizeof(required) / sizeof(required[0]),
                               allowed, sizeof(allowed) / sizeof(allowed[0]), error);
  if (status != SS_OK) {
    goto done;
  }
  status = ss_validate_common_objects(
      root, object_names, sizeof(object_names) / sizeof(object_names[0]), error);
  if (status != SS_OK) {
    goto done;
  }
  kind = yyjson_obj_get(root, "kind");
  if (!yyjson_equals_str(kind, "slope_stability.compiled_problem")) {
    ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_KIND",
                "manifest kind is not compiled_problem");
    status = SS_ERROR_SCHEMA;
    goto done;
  }
  status = ss_copy_string(yyjson_obj_get(root, "id"), problem->id, sizeof(problem->id),
                          error);
  if (status != SS_OK || !ss_valid_content_id(problem->id)) {
    if (status == SS_OK) {
      ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_ID",
                  "problem content identity is invalid");
      status = SS_ERROR_SCHEMA;
    }
    goto done;
  }
  status =
      ss_schema_version(yyjson_obj_get(root, "schema_version"), &problem->schema_major,
                        &problem->schema_minor, &problem->schema_patch, error);
  if (status != SS_OK) {
    goto done;
  }
  mesh = yyjson_obj_get(root, "mesh");
  status = ss_validate_object_shape(
      mesh, mesh_required, sizeof(mesh_required) / sizeof(mesh_required[0]),
      mesh_allowed, sizeof(mesh_allowed) / sizeof(mesh_allowed[0]), error);
  if (status != SS_OK) {
    goto done;
  }
  value = yyjson_is_obj(mesh) ? yyjson_obj_get(mesh, "dimension") : NULL;
  if (!yyjson_is_uint(value) || yyjson_get_uint(value) > UINT32_MAX) {
    ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_DIMENSION",
                "mesh dimension must be an unsigned integer");
    status = SS_ERROR_SCHEMA;
    goto done;
  }
  problem->dimension = (uint32_t)yyjson_get_uint(value);
  if (problem->dimension != 2U && problem->dimension != 3U) {
    ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_DIMENSION",
                "mesh dimension must be 2 or 3");
    status = SS_ERROR_SCHEMA;
    goto done;
  }
  value = yyjson_obj_get(mesh, "coordinate_order");
  if (!yyjson_is_uint(value) || yyjson_get_uint(value) > UINT32_MAX) {
    ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_COORDINATE_ORDER",
                "coordinate order must be an unsigned integer");
    status = SS_ERROR_SCHEMA;
    goto done;
  }
  problem->coordinate_order = (uint32_t)yyjson_get_uint(value);
  if (problem->coordinate_order != 1U && problem->coordinate_order != 2U &&
      problem->coordinate_order != 4U) {
    ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_COORDINATE_ORDER",
                "coordinate order must be 1, 2, or 4");
    status = SS_ERROR_SCHEMA;
    goto done;
  }
  value = yyjson_obj_get(mesh, "uniform_refinement_levels");
  if (value != NULL) {
    if (!yyjson_is_uint(value) || yyjson_get_uint(value) > UINT32_MAX) {
      ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_UNIFORM_REFINEMENT",
                  "uniform refinement levels must be an unsigned integer");
      status = SS_ERROR_SCHEMA;
      goto done;
    }
    problem->uniform_refinement_levels = (uint32_t)yyjson_get_uint(value);
  }
  spaces = yyjson_obj_get(root, "spaces");
  displacement = yyjson_is_obj(spaces) ? yyjson_obj_get(spaces, "displacement") : NULL;
  value = yyjson_is_obj(displacement) ? yyjson_obj_get(displacement, "order") : NULL;
  if (!yyjson_is_uint(value) || yyjson_get_uint(value) > UINT32_MAX) {
    ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_FIELD_ORDER",
                "displacement field order must be an unsigned integer");
    status = SS_ERROR_SCHEMA;
    goto done;
  }
  problem->field_order = (uint32_t)yyjson_get_uint(value);
  if (problem->field_order != 1U && problem->field_order != 2U &&
      problem->field_order != 4U && problem->field_order != 8U) {
    ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_FIELD_ORDER",
                "displacement field order must be 1, 2, 4, or 8");
    status = SS_ERROR_SCHEMA;
    goto done;
  }
  operators = yyjson_obj_get(root, "operators");
  status = ss_copy_string(yyjson_obj_get(operators, "analysis"), problem->analysis,
                          sizeof(problem->analysis), error);
  if (status == SS_OK) {
    status =
        ss_copy_string(yyjson_obj_get(operators, "operator_class"),
                       problem->operator_class, sizeof(problem->operator_class), error);
  }
  if (status != SS_OK || problem->analysis[0] == '\0' ||
      problem->operator_class[0] == '\0') {
    if (status == SS_OK) {
      ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_OPERATOR",
                  "operator fields must be non-empty");
      status = SS_ERROR_SCHEMA;
    }
    goto done;
  }
  value = yyjson_obj_get(operators, "volume_quadrature");
  if (value != NULL && !yyjson_equals_str(value, "default") &&
      !yyjson_equals_str(value, "paired_higher")) {
    ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_VOLUME_QUADRATURE",
                "operators.volume_quadrature must be default or paired_higher");
    status = SS_ERROR_SCHEMA;
    goto done;
  }
  payloads = yyjson_obj_get(root, "payloads");
  status =
      ss_validate_object_shape(payloads, payloads_shape, 1U, payloads_shape, 1U, error);
  if (status != SS_OK) {
    goto done;
  }
  payload = yyjson_is_obj(payloads) ? yyjson_obj_get(payloads, "problem_hdf5") : NULL;
  status = ss_validate_object_shape(
      payload, payload_required, sizeof(payload_required) / sizeof(payload_required[0]),
      payload_allowed, sizeof(payload_allowed) / sizeof(payload_allowed[0]), error);
  if (status != SS_OK) {
    goto done;
  }
  status = ss_copy_string(yyjson_obj_get(payload, "path"), problem->payload_path,
                          sizeof(problem->payload_path), error);
  if (status != SS_OK) {
    goto done;
  }
  status = ss_copy_string(yyjson_obj_get(payload, "sha256"), problem->payload_sha256,
                          sizeof(problem->payload_sha256), error);
  if (status != SS_OK || !ss_valid_lower_hex(problem->payload_sha256, 64U)) {
    if (status == SS_OK) {
      ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_PAYLOAD_DIGEST",
                  "payload digest is invalid");
      status = SS_ERROR_SCHEMA;
    }
    goto done;
  }
  value = yyjson_obj_get(payload, "byte_count");
  if (!yyjson_is_uint(value)) {
    ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_PAYLOAD_SIZE",
                "payload byte count must be unsigned");
    status = SS_ERROR_SCHEMA;
    goto done;
  }
  problem->payload_bytes = yyjson_get_uint(value);
  status = ss_copy_string(yyjson_obj_get(payload, "layout_id"), problem->layout_id,
                          sizeof(problem->layout_id), error);
  if (status == SS_OK && strcmp(problem->layout_id, "compiled-problem-v1") != 0) {
    ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_HDF5_LAYOUT",
                "unsupported compiled-problem HDF5 layout");
    status = SS_ERROR_SCHEMA;
  }

done:
  yyjson_doc_free(doc);
  return status;
}

static SsStatus ss_required_bool(yyjson_val *object, const char *name, int *result,
                                 SsErrorRecord *error) {
  yyjson_val *value = yyjson_is_obj(object) ? yyjson_obj_get(object, name) : NULL;
  if (!yyjson_is_bool(value)) {
    ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_BOOLEAN",
                "build feature must be a boolean");
    return SS_ERROR_SCHEMA;
  }
  *result = yyjson_get_bool(value) ? 1 : 0;
  return SS_OK;
}

static SsStatus ss_parse_accepted_state_limit(yyjson_val *root, SsRunSpec *run,
                                              SsErrorRecord *error) {
  yyjson_val *value =
      yyjson_obj_get(yyjson_obj_get(root, "continuation"), "maximum_accepted_states");
  if (value == NULL) {
    return SS_OK;
  }
  const SsAnalysisMethod method =
      strcmp(run->analysis, "limit_load.work_controlled") == 0
          ? SS_ANALYSIS_LIMIT_LOAD
          : SS_ANALYSIS_SSR_INDIRECT;
  const uint64_t limit = (uint64_t)ss_analysis_maximum_accepted_steps(method);
  if (!yyjson_is_uint(value) || yyjson_get_uint(value) == 0U ||
      yyjson_get_uint(value) > limit) {
    char message[SS_ERROR_MESSAGE_CAPACITY];
    (void)snprintf(message, sizeof(message),
                   "continuation.maximum_accepted_states must be an integer from 1 "
                   "to %llu for %s",
                   (unsigned long long)limit, run->analysis);
    ss_io_error(error, SS_ERROR_SCHEMA, "CONTINUATION_STATE_LIMIT", message);
    return SS_ERROR_SCHEMA;
  }
  run->maximum_accepted_states = (uint32_t)yyjson_get_uint(value);
  return SS_OK;
}

/* Record which validation profile the resolver selected; the profile ID is
 * optional so that runs resolved before profile provenance stay readable. */
static SsStatus ss_parse_run_validation(yyjson_val *root, SsRunSpec *run,
                                        SsErrorRecord *error) {
  yyjson_val *profiles =
      yyjson_obj_get(yyjson_obj_get(root, "provenance"), "profile_ids");
  yyjson_val *profile =
      yyjson_is_obj(profiles) ? yyjson_obj_get(profiles, "validation") : NULL;
  (void)snprintf(run->validation_profile, sizeof(run->validation_profile), "%s",
                 "unspecified");
  if (profile != NULL) {
    const SsStatus status = ss_copy_string(profile, run->validation_profile,
                                           sizeof(run->validation_profile), error);
    if (status != SS_OK || run->validation_profile[0] == '\0') {
      ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_VALIDATION_PROFILE",
                  "validation profile ID must be a nonempty string");
      return SS_ERROR_SCHEMA;
    }
  }
  return SS_OK;
}

SsStatus ss_io_parse_run(const unsigned char *data, size_t size,
                         const SsJsonLimits *limits, SsRunSpec *run,
                         SsErrorRecord *error) {
  static const char *const required[] = {
      "analysis",
      "build",
      "continuation",
      "created_by",
      "deflation",
      "diagnostics",
      "extensions",
      "id",
      "identity",
      "kind",
      "linear",
      "nonlinear",
      "outputs",
      "petsc_options",
      "pmg",
      "problem",
      "provenance",
      "resolution_trace",
      "resources",
      "result_policy",
      "schema_version",
      "validation",
  };
  static const char *const analysis_shape[] = {"selector"};
  static const char *const object_names[] = {
      "analysis",         "build",         "continuation",  "created_by", "deflation",
      "diagnostics",      "extensions",    "identity",      "linear",     "nonlinear",
      "outputs",          "petsc_options", "pmg",           "problem",    "provenance",
      "resolution_trace", "resources",     "result_policy", "validation",
  };
  static const char *const build_required[] = {
      "build_id", "dependency_id", "executable_id", "hypre",          "mpi",
      "openmp",   "parallel_hdf5", "parmetis",      "petsc_int_bits", "schema_version",
  };
  static const char *const problem_required[] = {
      "coordinate_order", "dimension",      "field_order", "id",
      "operator_class",   "schema_version",
  };
#define SS_RESOURCE_REQUIRED_FIELDS                                                    \
  "active_ranks", "cores_per_rank", "cpu_binding", "local_rank_map", "memory_binding", \
      "mpi_thread_level", "name", "nodes", "numa_policy", "openmp_environment",        \
      "openmp_threads", "ranks_per_node", "smt", "third_party_threads",                \
      "total_workers", "world_size",
  static const char *const resources_required[] = {SS_RESOURCE_REQUIRED_FIELDS};
  static const char *const resources_allowed[] = {
      SS_RESOURCE_REQUIRED_FIELDS "memory_per_node_mib",
      "address_space_per_rank_mib",
#define SS_RESOURCE_continuation(name)
#define SS_RESOURCE_nonlinear(name)
#define SS_RESOURCE_linear(name)
#define SS_RESOURCE_pmg(name)
#define SS_RESOURCE_outputs(name)
#define SS_RESOURCE_validation(name)
#define SS_RESOURCE_resources(name) #name,
#define SS_STABLE(type, section, name, fallback, minimum, maximum, choices, doc)       \
  SS_RESOURCE_##section(name)
#define SS_EXPERT(type, name, option, fallback, minimum, maximum, choices, doc)
#include "slope_internal/core/solver_options.def"
#undef SS_EXPERT
#undef SS_STABLE
#undef SS_RESOURCE_resources
#undef SS_RESOURCE_validation
#undef SS_RESOURCE_outputs
#undef SS_RESOURCE_pmg
#undef SS_RESOURCE_linear
#undef SS_RESOURCE_nonlinear
#undef SS_RESOURCE_continuation
  };
#undef SS_RESOURCE_REQUIRED_FIELDS
  static const char *const rank_row_shape[] = {"local_rank", "node", "rank"};
  yyjson_doc *doc = NULL;
  yyjson_val *root;
  yyjson_val *build;
  yyjson_val *problem;
  yyjson_val *resources;
  yyjson_val *value;
  uint32_t ignored_minor = 0U;
  uint32_t ignored_patch = 0U;
  uint32_t ignored_build_major = 0U;
  SsStatus status;
  if (run == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(run, 0, sizeof(*run));
  status = ss_read_document(data, size, limits, &doc, error);
  if (status != SS_OK) {
    return status;
  }
  root = yyjson_doc_get_root(doc);
  status =
      ss_validate_object_shape(root, required, sizeof(required) / sizeof(required[0]),
                               required, sizeof(required) / sizeof(required[0]), error);
  if (status != SS_OK) {
    goto done;
  }
  status = ss_validate_common_objects(
      root, object_names, sizeof(object_names) / sizeof(object_names[0]), error);
  if (status != SS_OK) {
    goto done;
  }
  if (!yyjson_equals_str(yyjson_obj_get(root, "kind"),
                         "slope_stability.resolved_run")) {
    ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_KIND",
                "manifest kind is not resolved_run");
    status = SS_ERROR_SCHEMA;
    goto done;
  }
  status = ss_copy_string(yyjson_obj_get(root, "id"), run->id, sizeof(run->id), error);
  if (status != SS_OK || !ss_valid_content_id(run->id)) {
    if (status == SS_OK) {
      ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_ID",
                  "run content identity is invalid");
      status = SS_ERROR_SCHEMA;
    }
    goto done;
  }
  status = ss_schema_version(yyjson_obj_get(root, "schema_version"), &run->schema_major,
                             &ignored_minor, &ignored_patch, error);
  if (status != SS_OK) {
    goto done;
  }
  problem = yyjson_obj_get(root, "problem");
  status = ss_validate_object_shape(
      problem, problem_required, sizeof(problem_required) / sizeof(problem_required[0]),
      problem_required, sizeof(problem_required) / sizeof(problem_required[0]), error);
  if (status != SS_OK) {
    goto done;
  }
  status = ss_copy_string(yyjson_is_obj(problem) ? yyjson_obj_get(problem, "id") : NULL,
                          run->problem_id, sizeof(run->problem_id), error);
  if (status != SS_OK || !ss_valid_content_id(run->problem_id)) {
    status = SS_ERROR_SCHEMA;
    goto done;
  }
  status = ss_schema_version_array(yyjson_obj_get(problem, "schema_version"),
                                   &run->problem_schema_major, error);
  if (status != SS_OK) {
    goto done;
  }
  value = yyjson_obj_get(problem, "dimension");
  if (!yyjson_is_uint(value) || yyjson_get_uint(value) > UINT32_MAX) {
    status = SS_ERROR_SCHEMA;
    ss_io_error(error, status, "SCHEMA_DIMENSION",
                "resolved problem dimension must be unsigned");
    goto done;
  }
  run->dimension = (uint32_t)yyjson_get_uint(value);
  value = yyjson_obj_get(problem, "field_order");
  if (!yyjson_is_uint(value) || yyjson_get_uint(value) > UINT32_MAX) {
    status = SS_ERROR_SCHEMA;
    ss_io_error(error, status, "SCHEMA_FIELD_ORDER",
                "resolved field order must be unsigned");
    goto done;
  }
  run->field_order = (uint32_t)yyjson_get_uint(value);
  value = yyjson_obj_get(problem, "coordinate_order");
  if (!yyjson_is_uint(value) || yyjson_get_uint(value) > UINT32_MAX) {
    status = SS_ERROR_SCHEMA;
    ss_io_error(error, status, "SCHEMA_COORDINATE_ORDER",
                "resolved coordinate order must be unsigned");
    goto done;
  }
  run->coordinate_order = (uint32_t)yyjson_get_uint(value);
  status = ss_copy_string(yyjson_obj_get(problem, "operator_class"),
                          run->operator_class, sizeof(run->operator_class), error);
  if (status != SS_OK || (run->dimension != 2U && run->dimension != 3U) ||
      (run->field_order != 1U && run->field_order != 2U && run->field_order != 4U &&
       run->field_order != 8U) ||
      (run->coordinate_order != 1U && run->coordinate_order != 2U &&
       run->coordinate_order != 4U) ||
      run->operator_class[0] == '\0') {
    if (status == SS_OK) {
      ss_io_error(error, SS_ERROR_SCHEMA, "SCHEMA_PROBLEM_REFERENCE",
                  "resolved problem reference is incompatible");
      status = SS_ERROR_SCHEMA;
    }
    goto done;
  }
  build = yyjson_obj_get(root, "build");
  status = ss_validate_object_shape(
      build, build_required, sizeof(build_required) / sizeof(build_required[0]),
      build_required, sizeof(build_required) / sizeof(build_required[0]), error);
  if (status != SS_OK) {
    goto done;
  }
  status = ss_schema_version_array(yyjson_obj_get(build, "schema_version"),
                                   &ignored_build_major, error);
  if (status != SS_OK) {
    goto done;
  }
  status =
      ss_copy_string(yyjson_is_obj(build) ? yyjson_obj_get(build, "build_id") : NULL,
                     run->build_id, sizeof(run->build_id), error);
  if (status != SS_OK || !ss_valid_content_id(run->build_id)) {
    status = SS_ERROR_SCHEMA;
    goto done;
  }
  status = ss_copy_string(yyjson_obj_get(build, "executable_id"), run->executable_id,
                          sizeof(run->executable_id), error);
  if (status != SS_OK || !ss_valid_content_id(run->executable_id)) {
    status = SS_ERROR_SCHEMA;
    goto done;
  }
  status = ss_copy_string(yyjson_obj_get(build, "dependency_id"), run->dependency_id,
                          sizeof(run->dependency_id), error);
  if (status != SS_OK || !ss_valid_content_id(run->dependency_id)) {
    status = SS_ERROR_SCHEMA;
    goto done;
  }
  value = yyjson_obj_get(build, "petsc_int_bits");
  if (!yyjson_is_uint(value) || yyjson_get_uint(value) > UINT32_MAX) {
    status = SS_ERROR_SCHEMA;
    ss_io_error(error, status, "SCHEMA_INDEX_WIDTH",
                "PETSc index width must be unsigned");
    goto done;
  }
  run->petsc_int_bits = (uint32_t)yyjson_get_uint(value);
  if (run->petsc_int_bits != 32U && run->petsc_int_bits != 64U) {
    status = SS_ERROR_SCHEMA;
    ss_io_error(error, status, "SCHEMA_INDEX_WIDTH",
                "PETSc index width must be 32 or 64");
    goto done;
  }
  if (ss_required_bool(build, "mpi", &run->mpi, error) != SS_OK ||
      ss_required_bool(build, "openmp", &run->openmp, error) != SS_OK ||
      ss_required_bool(build, "parallel_hdf5", &run->parallel_hdf5, error) != SS_OK ||
      ss_required_bool(build, "hypre", &run->hypre, error) != SS_OK ||
      ss_required_bool(build, "parmetis", &run->parmetis, error) != SS_OK) {
    status = SS_ERROR_SCHEMA;
    goto done;
  }
  value = yyjson_obj_get(root, "analysis");
  status =
      ss_validate_object_shape(value, analysis_shape, 1U, analysis_shape, 1U, error);
  if (status != SS_OK) {
    goto done;
  }
  value = yyjson_obj_get(value, "selector");
  status = ss_copy_string(value, run->analysis, sizeof(run->analysis), error);
  if (status != SS_OK) {
    goto done;
  }
  status = ss_parse_accepted_state_limit(root, run, error);
  if (status != SS_OK) {
    goto done;
  }
  status = ss_parse_run_validation(root, run, error);
  if (status != SS_OK) {
    goto done;
  }
  resources = yyjson_obj_get(root, "resources");
  status = ss_validate_object_shape(
      resources, resources_required,
      sizeof(resources_required) / sizeof(resources_required[0]), resources_allowed,
      sizeof(resources_allowed) / sizeof(resources_allowed[0]), error);
  if (status != SS_OK) {
    goto done;
  }
  const char *invalid_setting = NULL;
  status = ss_solver_settings_read(root, &run->settings, &invalid_setting);
  if (status != SS_OK) {
    char message[SS_ERROR_MESSAGE_CAPACITY];
    (void)snprintf(message, sizeof(message), "invalid solver setting: %s",
                   invalid_setting != NULL ? invalid_setting : "unknown");
    ss_io_error(error, status, "SCHEMA_SOLVER_SETTING", message);
    goto done;
  }
  static const char *const memory_metadata[] = {"memory_per_node_mib",
                                                "address_space_per_rank_mib"};
  for (size_t i = 0; i < sizeof(memory_metadata) / sizeof(memory_metadata[0]); ++i) {
    value = yyjson_obj_get(resources, memory_metadata[i]);
    if (value != NULL && (!yyjson_is_uint(value) || yyjson_get_uint(value) == 0U)) {
      status = SS_ERROR_SCHEMA;
      ss_io_error(error, status, "SCHEMA_RESOURCE_MEMORY", memory_metadata[i]);
      goto done;
    }
  }
  value = yyjson_is_obj(resources) ? yyjson_obj_get(resources, "world_size") : NULL;
  if (!yyjson_is_uint(value) || yyjson_get_uint(value) > UINT32_MAX) {
    status = SS_ERROR_SCHEMA;
    ss_io_error(error, status, "SCHEMA_WORLD_SIZE", "world size must be unsigned");
    goto done;
  }
  run->world_size = (uint32_t)yyjson_get_uint(value);
  if (run->world_size == 0U) {
    status = SS_ERROR_SCHEMA;
    goto done;
  }
  value = yyjson_obj_get(resources, "openmp_threads");
  if (!yyjson_is_uint(value) || yyjson_get_uint(value) > UINT32_MAX) {
    status = SS_ERROR_SCHEMA;
    ss_io_error(error, status, "SCHEMA_THREAD_COUNT",
                "OpenMP thread count must be unsigned");
    goto done;
  }
  run->openmp_threads = (uint32_t)yyjson_get_uint(value);
  if (run->openmp_threads == 0U) {
    status = SS_ERROR_SCHEMA;
    goto done;
  }
  status = ss_copy_string(yyjson_obj_get(resources, "mpi_thread_level"),
                          run->mpi_thread_level, sizeof(run->mpi_thread_level), error);
  if (status == SS_OK) {
    yyjson_val *rank_map = yyjson_obj_get(resources, "local_rank_map");
    size_t rank_index;
    size_t rank_count;
    yyjson_val *rank_row;
    if (!yyjson_is_arr(rank_map) ||
        yyjson_arr_size(rank_map) > SS_RESOURCE_RANK_CAPACITY) {
      ss_io_error(error, SS_ERROR_LIMIT, "RESOURCE_RANK_MAP_LIMIT",
                  "local rank map is absent or exceeds native capacity");
      status = SS_ERROR_LIMIT;
      goto done;
    }
    run->local_rank_map_count = (uint32_t)yyjson_arr_size(rank_map);
    yyjson_arr_foreach(rank_map, rank_index, rank_count, rank_row) {
      yyjson_val *rank_value;
      yyjson_val *node_value;
      yyjson_val *local_value;
      if (ss_validate_object_shape(rank_row, rank_row_shape, 3U, rank_row_shape, 3U,
                                   error) != SS_OK) {
        status = SS_ERROR_SCHEMA;
        break;
      }
      rank_value = yyjson_obj_get(rank_row, "rank");
      node_value = yyjson_obj_get(rank_row, "node");
      local_value = yyjson_obj_get(rank_row, "local_rank");
      if (!yyjson_is_uint(rank_value) || !yyjson_is_uint(node_value) ||
          !yyjson_is_uint(local_value) || yyjson_get_uint(rank_value) > UINT32_MAX ||
          yyjson_get_uint(node_value) > UINT32_MAX ||
          yyjson_get_uint(local_value) > UINT32_MAX) {
        status = SS_ERROR_SCHEMA;
        break;
      }
      run->local_rank_map_rank[rank_index] = (uint32_t)yyjson_get_uint(rank_value);
      run->local_rank_map_node[rank_index] = (uint32_t)yyjson_get_uint(node_value);
      run->local_rank_map_local[rank_index] = (uint32_t)yyjson_get_uint(local_value);
    }
    if (status != SS_OK) {
      ss_io_error(error, status, "RESOURCE_RANK_MAP_SCHEMA",
                  "local rank map row is invalid");
    }
  }

done:
  yyjson_doc_free(doc);
  return status;
}

SsStatus ss_io_configure(const unsigned char *problem_data, size_t problem_size,
                         const unsigned char *run_data, size_t run_size,
                         const SsJsonLimits *limits, SsNativeConfig *config,
                         SsErrorRecord *error) {
  SsStatus status;
  if (config == NULL || config->sealed != 0) {
    return config == NULL ? SS_ERROR_INVALID_ARGUMENT : SS_ERROR_INVALID_STATE;
  }
  status =
      ss_io_parse_problem(problem_data, problem_size, limits, &config->problem, error);
  if (status != SS_OK) {
    return status;
  }
  if (run_data != NULL || run_size != 0U) {
    status = ss_io_parse_run(run_data, run_size, limits, &config->run, error);
    if (status != SS_OK) {
      memset(config, 0, sizeof(*config));
      return status;
    }
    if (strcmp(config->problem.id, config->run.problem_id) != 0) {
      ss_io_error(error, SS_ERROR_INTEGRITY, "PROBLEM_RUN_ID_MISMATCH",
                  "resolved run references a different problem");
      memset(config, 0, sizeof(*config));
      return SS_ERROR_INTEGRITY;
    }
    if (config->problem.schema_major != config->run.problem_schema_major ||
        config->problem.dimension != config->run.dimension ||
        config->problem.field_order != config->run.field_order ||
        config->problem.coordinate_order != config->run.coordinate_order ||
        strcmp(config->problem.analysis, config->run.analysis) != 0 ||
        strcmp(config->problem.operator_class, config->run.operator_class) != 0) {
      ss_io_error(error, SS_ERROR_INTEGRITY, "PROBLEM_RUN_CONTRACT_MISMATCH",
                  "resolved run problem contract differs from compiled problem");
      memset(config, 0, sizeof(*config));
      return SS_ERROR_INTEGRITY;
    }
    config->has_run = 1;
  }
  config->sealed = 1;
  return SS_OK;
}

int ss_io_config_is_sealed(const SsNativeConfig *config) {
  return config != NULL ? config->sealed : 0;
}

const SsProblemSpec *ss_io_problem(const SsNativeConfig *config) {
  return config != NULL && config->sealed != 0 ? &config->problem : NULL;
}

void ss_io_config_destroy(SsNativeConfig *config) {
  if (config != NULL) {
    if (config->compiled_manifest != NULL) {
      ss_io_compiled_manifest_destroy(config->compiled_manifest);
      free(config->compiled_manifest);
    }
    memset(config, 0, sizeof(*config));
  }
}
