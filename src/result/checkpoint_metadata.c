/* Validate, digest and serialize checkpoint identities and envelopes.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "checkpoint_internal.h"
#include "result_envelope.h"

#include "yyjson.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

SsStatus ss_checkpoint_path(char *output, size_t capacity, const char *root,
                            const char *leaf) {
  const int count = snprintf(output, capacity, "%s/%s", root, leaf);
  return count < 0 || (size_t)count >= capacity ? SS_ERROR_OVERFLOW : SS_OK;
}

SsStatus ss_checkpoint_components(const char *const *component_ids,
                                  uint32_t component_count, char *output,
                                  size_t capacity) {
  size_t cursor = 0U;
  uint32_t component;
  if (component_ids == NULL || component_count == 0U || output == NULL ||
      capacity == 0U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  output[0] = '\0';
  for (component = 0U; component < component_count; ++component) {
    const char *const component_id = component_ids[component];
    size_t length;
    const size_t separator = component == 0U ? 0U : 1U;
    uint32_t prior;
    if (!ss_result_name_valid(component_id)) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
    length = strlen(component_id);
    if (length > capacity - 1U || separator > capacity - 1U - length ||
        cursor > capacity - 1U - length - separator) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
    for (prior = 0U; prior < component; ++prior) {
      if (strcmp(component_id, component_ids[prior]) == 0) {
        return SS_ERROR_DUPLICATE;
      }
    }
    if (separator != 0U) {
      output[cursor++] = ',';
    }
    memcpy(output + cursor, component_id, length);
    cursor += length;
    output[cursor] = '\0';
  }
  return SS_OK;
}

static SsStatus ss_checkpoint_append_field_id(char *output, size_t capacity,
                                              size_t *cursor, const char *quantity_id,
                                              size_t field_index) {
  size_t length;
  const size_t separator = field_index == 0U ? 0U : 1U;
  if (output == NULL || cursor == NULL || quantity_id == NULL || capacity == 0U ||
      !ss_result_name_valid(quantity_id)) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  length = strlen(quantity_id);
  if (length > capacity - 1U || separator > capacity - 1U - length ||
      *cursor > capacity - 1U - length - separator) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (separator != 0U) {
    output[(*cursor)++] = ',';
  }
  memcpy(output + *cursor, quantity_id, length);
  *cursor += length;
  output[*cursor] = '\0';
  return SS_OK;
}

SsStatus ss_checkpoint_spec_field_ids(const SsCheckpointSpec *spec, char *output,
                                      size_t capacity) {
  size_t cursor = 0U;
  size_t field;
  SsStatus status = SS_OK;
  if (spec == NULL || spec->fields == NULL || spec->field_count == 0U ||
      output == NULL || capacity == 0U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  output[0] = '\0';
  for (field = 0U; status == SS_OK && field < spec->field_count; ++field) {
    status = ss_checkpoint_append_field_id(output, capacity, &cursor,
                                           spec->fields[field].quantity_id, field);
  }
  return status;
}

SsStatus ss_checkpoint_restore_field_ids(const SsCheckpointRestoreView *restore,
                                         char *output, size_t capacity) {
  size_t cursor = 0U;
  size_t field;
  SsStatus status = SS_OK;
  if (restore == NULL || restore->fields == NULL || restore->field_count == 0U ||
      output == NULL || capacity == 0U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  output[0] = '\0';
  for (field = 0U; status == SS_OK && field < restore->field_count; ++field) {
    status = ss_checkpoint_append_field_id(output, capacity, &cursor,
                                           restore->fields[field].quantity_id, field);
  }
  return status;
}

SsStatus ss_checkpoint_spec_validate(const SsCheckpointSpec *spec) {
  size_t field;
  size_t prior;
  if (spec == NULL ||
      !ss_result_spec_identity_valid(spec->final_directory, spec->problem_id,
                                     spec->run_id, spec->build_id, spec->mesh_id,
                                     spec->geometry_id) ||
      !ss_valid_content_id(spec->build_compatibility_id) ||
      !ss_valid_content_id(spec->numerical_policy_id) ||
      !ss_result_text_is_valid(spec->field_space_id, SS_RESULT_TEXT_CAPACITY) ||
      !ss_result_text_is_valid(spec->material_model_id, SS_RESULT_TEXT_CAPACITY) ||
      !ss_result_text_is_valid(spec->hydraulic_model_id, SS_RESULT_TEXT_CAPACITY) ||
      !ss_result_text_is_valid(spec->coupling_model_id, SS_RESULT_TEXT_CAPACITY) ||
      !ss_result_text_is_valid(spec->analysis_method, SS_RESULT_TEXT_CAPACITY) ||
      !ss_result_text_is_valid(spec->controller_id, SS_RESULT_TEXT_CAPACITY) ||
      !ss_result_orders_valid(spec->field_order, spec->geometry_order) ||
      spec->scalar_bits != 64U ||
      (spec->source_index_bits != 32U && spec->source_index_bits != 64U) ||
      spec->accepted_state_id == 0U || spec->fields == NULL ||
      spec->field_count == 0U || spec->field_count > SS_CHECKPOINT_FIELD_CAPACITY ||
      spec->controller_value_count > SS_RESULT_CONTROLLER_CAPACITY ||
      spec->coupling_value_count > SS_RESULT_COUPLING_CAPACITY ||
      (spec->controller_value_count != 0U && spec->controller_values == NULL) ||
      (spec->coupling_value_count != 0U && spec->coupling_values == NULL) ||
      spec->prior_state_count > SS_CHECKPOINT_PRIOR_STATE_CAPACITY ||
      (spec->prior_state_count != 0U && spec->prior_states == NULL) ||
      !isfinite(spec->next_step) || spec->next_step <= 0.0 ||
      spec->dataset_chunk_rows > UINT32_MAX ||
      !ss_result_text_is_valid(spec->deflation.provenance_id,
                               SS_RESULT_TEXT_CAPACITY) ||
      !ss_result_text_is_valid(spec->deflation.operator_identity,
                               SS_RESULT_TEXT_CAPACITY) ||
      spec->deflation.basis_count > SS_RESULT_COMPONENT_CAPACITY ||
      ((spec->deflation.basis_count == 0U) != (spec->deflation.global_count == 0U)) ||
      (spec->deflation.basis_count != 0U && (spec->deflation.source_weights == NULL ||
                                             spec->deflation.source_kinds == NULL)) ||
      (spec->deflation.local_count != 0U &&
       (spec->deflation.stable_free_dof_ids == NULL ||
        spec->deflation.left_values == NULL || spec->deflation.right_values == NULL ||
        spec->deflation.source_values == NULL))) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (field = 0U; field < spec->field_count; ++field) {
    const SsCheckpointFieldView *view = &spec->fields[field];
    char components[SS_RESULT_COMPONENT_CAPACITY * SS_RESULT_TEXT_CAPACITY];
    if (!ss_result_name_valid(view->quantity_id) ||
        !ss_result_text_is_valid(view->unit, SS_RESULT_TEXT_CAPACITY) ||
        !ss_result_name_valid(view->space_id) || !ss_result_name_valid(view->basis) ||
        view->component_count == 0U ||
        view->component_count > SS_RESULT_COMPONENT_CAPACITY ||
        view->global_count == 0U ||
        (view->local_count != 0U &&
         (view->stable_ids == NULL || view->values == NULL)) ||
        ss_checkpoint_components(view->component_ids, view->component_count, components,
                                 sizeof(components)) != SS_OK ||
        (field != 0U &&
         strcmp(spec->fields[field - 1U].quantity_id, view->quantity_id) >= 0)) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
  }
  for (prior = 0U; prior < spec->prior_state_count; ++prior) {
    const SsCheckpointPriorState *state = &spec->prior_states[prior];
    size_t earlier;
    if (state->state_id == 0U || state->state_id == spec->accepted_state_id ||
        !isfinite(state->factor) || !isfinite(state->work) ||
        !isfinite(state->direct_control)) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
    for (earlier = 0U; earlier < prior; ++earlier) {
      if (spec->prior_states[earlier].state_id == state->state_id) {
        return SS_ERROR_DUPLICATE;
      }
    }
  }
  for (field = 0U; field < spec->deflation.basis_count; ++field) {
    const double weight = spec->deflation.source_weights[field];
    const double kind = spec->deflation.source_kinds[field];
    if (!isfinite(weight) || weight <= 0.0 || !isfinite(kind) || floor(kind) != kind ||
        kind < 0.0 || kind > 2.0) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
  }
  return SS_OK;
}

SsStatus ss_checkpoint_spec_agree(const SsCheckpointSpec *spec) {
  unsigned char payload[65536];
  char digest[SS_SHA256_HEX_CAPACITY];
  size_t cursor = 0U;
  size_t field;
  size_t index;
  int size = 0;
  int count;
  SsStatus status = SS_OK;
#define SS_CHECKPOINT_APPEND_FORMAT(...)                                               \
  do {                                                                                 \
    count = snprintf((char *)payload + cursor, sizeof(payload) - cursor, __VA_ARGS__); \
    if (count < 0 || (size_t)count >= sizeof(payload) - cursor) {                      \
      return SS_ERROR_OVERFLOW;                                                        \
    }                                                                                  \
    cursor += (size_t)count;                                                           \
  } while (0)
  SS_CHECKPOINT_APPEND_FORMAT(
      "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%u|%u|%u|%u|%llu|"
      "%llu|%llu|%llu|%.17g|%llu|%zu|%zu|%zu|%u|%llu|%s|%s|",
      spec->final_directory, spec->problem_id, spec->run_id, spec->build_id,
      spec->build_compatibility_id, spec->mesh_id, spec->geometry_id,
      spec->field_space_id, spec->numerical_policy_id, spec->material_model_id,
      spec->hydraulic_model_id, spec->coupling_model_id, spec->analysis_method,
      spec->field_order, spec->geometry_order, spec->scalar_bits,
      spec->source_index_bits, (unsigned long long)spec->accepted_state_id,
      (unsigned long long)spec->history_cursor,
      (unsigned long long)spec->accepted_state_cursor,
      (unsigned long long)spec->retry_count, spec->next_step,
      (unsigned long long)spec->dataset_chunk_rows, spec->field_count,
      spec->controller_value_count, spec->coupling_value_count,
      spec->deflation.basis_count, (unsigned long long)spec->deflation.global_count,
      spec->deflation.provenance_id, spec->deflation.operator_identity);
  SS_CHECKPOINT_APPEND_FORMAT("%s|%zu|", spec->controller_id, spec->prior_state_count);
  for (field = 0U; field < spec->field_count; ++field) {
    char components[SS_RESULT_COMPONENT_CAPACITY * SS_RESULT_TEXT_CAPACITY];
    status = ss_checkpoint_components(spec->fields[field].component_ids,
                                      spec->fields[field].component_count, components,
                                      sizeof(components));
    if (status != SS_OK) {
      return status;
    }
    SS_CHECKPOINT_APPEND_FORMAT(
        "%s|%s|%s|%s|%s|%u|%llu|", spec->fields[field].quantity_id,
        spec->fields[field].unit, spec->fields[field].space_id,
        spec->fields[field].basis, components, spec->fields[field].component_count,
        (unsigned long long)spec->fields[field].global_count);
  }
#undef SS_CHECKPOINT_APPEND_FORMAT
  for (index = 0U; index < spec->controller_value_count; ++index) {
    ss_result_store_double_le(payload + cursor, spec->controller_values[index]);
    cursor += sizeof(double);
  }
  for (index = 0U; index < spec->coupling_value_count; ++index) {
    ss_result_store_double_le(payload + cursor, spec->coupling_values[index]);
    cursor += sizeof(double);
  }
  for (index = 0U; index < spec->prior_state_count; ++index) {
    const SsCheckpointPriorState *prior = &spec->prior_states[index];
    ss_result_store_u64_le(payload + cursor, prior->state_id);
    cursor += 8U;
    ss_result_store_u64_le(payload + cursor, prior->parent_state_id);
    cursor += 8U;
    ss_result_store_u64_le(payload + cursor, prior->accepted_ordinal);
    cursor += 8U;
    ss_result_store_double_le(payload + cursor, prior->factor);
    cursor += 8U;
    ss_result_store_double_le(payload + cursor, prior->work);
    cursor += 8U;
    ss_result_store_double_le(payload + cursor, prior->direct_control);
    cursor += 8U;
  }
  for (index = 0U; index < spec->deflation.basis_count; ++index) {
    ss_result_store_double_le(payload + cursor, spec->deflation.source_weights[index]);
    cursor += 8U;
    ss_result_store_double_le(payload + cursor, spec->deflation.source_kinds[index]);
    cursor += 8U;
  }
  if (MPI_Comm_size(spec->communicator, &size) != MPI_SUCCESS || size <= 0 ||
      ss_sha256_bytes(payload, cursor, digest) != SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  return ss_result_collective_digest_agree(spec->communicator, size, digest, SS_OK);
}

SsStatus ss_checkpoint_logical_digest_values(
    MPI_Comm communicator, const char *problem_id, const char *build_id,
    const char *build_compatibility_id, const char *mesh_id, const char *geometry_id,
    const char *field_space_id, const char *numerical_policy_id,
    const char *material_model_id, const char *hydraulic_model_id,
    const char *coupling_model_id, const char *analysis_method,
    const char *controller_id, const char *deflation_provenance_id,
    const char *operator_identity, uint32_t field_order, uint32_t geometry_order,
    uint32_t scalar_bits, uint32_t source_index_bits, SsStateId accepted_state_id,
    const SsCheckpointFieldDigest *fields, size_t field_count, const char *left_digest,
    const char *right_digest, const char *source_digest,
    const char *source_weights_digest, const char *source_kinds_digest,
    const char *controller_digest, const char *coupling_digest,
    const char *prior_digest, const char *cursor_digest,
    char output[SS_SHA256_HEX_CAPACITY + 8U]) {
  char *semantic = NULL;
  size_t capacity = 65536U;
  size_t cursor = 0U;
  size_t field;
  int rank = -1;
  int count;
  SsStatus status = SS_OK;
  if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (rank == 0) {
    semantic = (char *)malloc(capacity);
    if (semantic == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  if (rank == 0 && status == SS_OK) {
    count = snprintf(semantic, capacity,
                     "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%u|%u|%u|%u|"
                     "%llu|fields=%zu|",
                     problem_id, build_id, build_compatibility_id, mesh_id, geometry_id,
                     field_space_id, numerical_policy_id, material_model_id,
                     hydraulic_model_id, coupling_model_id, analysis_method,
                     controller_id, deflation_provenance_id, operator_identity,
                     field_order, geometry_order, scalar_bits, source_index_bits,
                     (unsigned long long)accepted_state_id, field_count);
    if (count < 0 || (size_t)count >= capacity) {
      status = SS_ERROR_OVERFLOW;
    } else {
      cursor = (size_t)count;
    }
    for (field = 0U; status == SS_OK && field < field_count; ++field) {
      count = snprintf(
          semantic + cursor, capacity - cursor, "%s|%s|%s|%s|%s|%u|%llu|%s|",
          fields[field].quantity_id, fields[field].unit, fields[field].space_id,
          fields[field].basis, fields[field].components, fields[field].component_count,
          (unsigned long long)fields[field].global_count, fields[field].digest);
      if (count < 0 || (size_t)count >= capacity - cursor) {
        status = SS_ERROR_OVERFLOW;
      } else {
        cursor += (size_t)count;
      }
    }
    if (status == SS_OK) {
      count = snprintf(
          semantic + cursor, capacity - cursor,
          "deflation=%s|%s|%s|%s|%s|controller=%s|coupling=%s|prior=%s|cursor=%s",
          left_digest, right_digest, source_digest, source_weights_digest,
          source_kinds_digest, controller_digest, coupling_digest, prior_digest,
          cursor_digest);
      if (count < 0 || (size_t)count >= capacity - cursor) {
        status = SS_ERROR_OVERFLOW;
      } else {
        cursor += (size_t)count;
      }
    }
    if (status == SS_OK) {
      status = ss_result_digest_bytes(semantic, cursor, output);
    }
  }
  free(semantic);
  return ss_result_broadcast_digest(communicator, status, output);
}

SsStatus ss_checkpoint_logical_digest(
    const SsCheckpointSpec *spec, const SsCheckpointFieldDigest *fields,
    const char *left_digest, const char *right_digest, const char *source_digest,
    const char *source_weights_digest, const char *source_kinds_digest,
    const char *controller_digest, const char *coupling_digest,
    const char *prior_digest, const char *cursor_digest,
    char output[SS_SHA256_HEX_CAPACITY + 8U]) {
  return ss_checkpoint_logical_digest_values(
      spec->communicator, spec->problem_id, spec->build_id,
      spec->build_compatibility_id, spec->mesh_id, spec->geometry_id,
      spec->field_space_id, spec->numerical_policy_id, spec->material_model_id,
      spec->hydraulic_model_id, spec->coupling_model_id, spec->analysis_method,
      spec->controller_id, spec->deflation.provenance_id,
      spec->deflation.operator_identity, spec->field_order, spec->geometry_order,
      spec->scalar_bits, spec->source_index_bits, spec->accepted_state_id, fields,
      spec->field_count, left_digest, right_digest, source_digest,
      source_weights_digest, source_kinds_digest, controller_digest, coupling_digest,
      prior_digest, cursor_digest, output);
}

SsStatus ss_checkpoint_id(MPI_Comm communicator, const char *run_id, SsStateId state_id,
                          const char *logical_digest,
                          char output[SS_SHA256_HEX_CAPACITY + 8U]) {
  char semantic[1024];
  int rank = -1;
  int count;
  SsStatus status = SS_OK;
  if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  if (rank == 0) {
    count = snprintf(semantic, sizeof(semantic), "%s|%llu|%s", run_id,
                     (unsigned long long)state_id, logical_digest);
    if (count < 0 || (size_t)count >= sizeof(semantic)) {
      status = SS_ERROR_OVERFLOW;
    } else {
      status = ss_result_digest_bytes(semantic, (size_t)count, output);
    }
  }
  return ss_result_broadcast_digest(communicator, status, output);
}

SsStatus ss_checkpoint_envelope(const SsCheckpointSpec *spec, const char *checkpoint_id,
                                const char *logical_digest, const char *payload_digest,
                                uint64_t payload_bytes,
                                const SsCheckpointFieldDigest *fields,
                                unsigned char **output, size_t *output_size) {
  yyjson_mut_doc *document = yyjson_mut_doc_new(NULL);
  yyjson_mut_val *root = document == NULL ? NULL : yyjson_mut_obj(document);
  yyjson_mut_val *version;
  yyjson_mut_val *identity;
  yyjson_mut_val *run;
  yyjson_mut_val *state;
  yyjson_mut_val *compatibility;
  yyjson_mut_val *field_array;
  yyjson_mut_val *payloads;
  yyjson_mut_val *payload;
  size_t field;
  if (document == NULL || root == NULL) {
    yyjson_mut_doc_free(document);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  yyjson_mut_doc_set_root(document, root);
  version = yyjson_mut_obj_add_obj(document, root, "schema_version");
  identity = yyjson_mut_obj_add_obj(document, root, "identity");
  run = yyjson_mut_obj_add_obj(document, root, "run");
  state = yyjson_mut_obj_add_obj(document, root, "accepted_state");
  compatibility = yyjson_mut_obj_add_obj(document, root, "compatibility");
  field_array = yyjson_mut_obj_add_arr(document, root, "fields");
  payloads = yyjson_mut_obj_add_obj(document, root, "payloads");
  payload = payloads == NULL
                ? NULL
                : yyjson_mut_obj_add_obj(document, payloads, "checkpoint.h5");
  if (version == NULL || identity == NULL || run == NULL || state == NULL ||
      compatibility == NULL || field_array == NULL || payloads == NULL ||
      payload == NULL ||
      !ss_result_envelope_header(document, root, version, "slope_stability.checkpoint",
                                 checkpoint_id, 2U) ||
      !ss_result_envelope_identity(document, identity, "checkpoint_id", checkpoint_id,
                                   logical_digest) ||
      !ss_result_envelope_run(document, run, spec->run_id, spec->problem_id,
                              spec->build_id) ||
      !yyjson_mut_obj_add_uint(document, state, "state_id", spec->accepted_state_id) ||
      !yyjson_mut_obj_add_bool(document, state, "accepted", true) ||
      !yyjson_mut_obj_add_bool(document, state, "complete", true) ||
      !yyjson_mut_obj_add_str(document, compatibility, "problem_id",
                              spec->problem_id) ||
      !yyjson_mut_obj_add_str(document, compatibility, "build_compatibility_id",
                              spec->build_compatibility_id) ||
      !yyjson_mut_obj_add_str(document, compatibility, "mesh_id", spec->mesh_id) ||
      !yyjson_mut_obj_add_str(document, compatibility, "geometry_id",
                              spec->geometry_id) ||
      !yyjson_mut_obj_add_str(document, compatibility, "field_space_id",
                              spec->field_space_id) ||
      !yyjson_mut_obj_add_str(document, compatibility, "numerical_policy_id",
                              spec->numerical_policy_id) ||
      !yyjson_mut_obj_add_str(document, compatibility, "material_model_id",
                              spec->material_model_id) ||
      !yyjson_mut_obj_add_str(document, compatibility, "hydraulic_model_id",
                              spec->hydraulic_model_id) ||
      !yyjson_mut_obj_add_str(document, compatibility, "coupling_model_id",
                              spec->coupling_model_id) ||
      !yyjson_mut_obj_add_str(document, compatibility, "analysis_method",
                              spec->analysis_method) ||
      !yyjson_mut_obj_add_str(document, compatibility, "controller_id",
                              spec->controller_id) ||
      !yyjson_mut_obj_add_str(document, compatibility, "operator_identity",
                              spec->deflation.operator_identity) ||
      !yyjson_mut_obj_add_uint(document, compatibility, "field_order",
                               spec->field_order) ||
      !yyjson_mut_obj_add_uint(document, compatibility, "geometry_order",
                               spec->geometry_order) ||
      !yyjson_mut_obj_add_uint(document, compatibility, "scalar_bits",
                               spec->scalar_bits) ||
      !yyjson_mut_obj_add_uint(document, compatibility, "source_index_bits",
                               spec->source_index_bits) ||
      !ss_result_envelope_payload(document, payload, "checkpoint.h5", payload_bytes,
                                  payload_digest, logical_digest)) {
    yyjson_mut_doc_free(document);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (field = 0U; field < spec->field_count; ++field) {
    yyjson_mut_val *entry = yyjson_mut_arr_add_obj(document, field_array);
    if (entry == NULL ||
        !yyjson_mut_obj_add_str(document, entry, "quantity_id",
                                fields[field].quantity_id) ||
        !yyjson_mut_obj_add_str(document, entry, "logical_digest",
                                fields[field].digest) ||
        !yyjson_mut_obj_add_uint(document, entry, "global_count",
                                 fields[field].global_count) ||
        !yyjson_mut_obj_add_uint(document, entry, "component_count",
                                 fields[field].component_count)) {
      yyjson_mut_doc_free(document);
      return SS_ERROR_OUT_OF_MEMORY;
    }
  }
  return ss_result_envelope_serialize(document, output, output_size);
}

SsStatus ss_checkpoint_envelope_read(const char *directory,
                                     char checkpoint_id[SS_SHA256_HEX_CAPACITY + 8U],
                                     char logical_digest[SS_SHA256_HEX_CAPACITY + 8U],
                                     char payload_digest[SS_SHA256_HEX_CAPACITY],
                                     uint64_t *payload_bytes, SsStateId *state_id) {
  char path[SS_RESULT_PATH_CAPACITY];
  yyjson_doc *document = NULL;
  yyjson_val *root;
  yyjson_val *version;
  yyjson_val *identity;
  yyjson_val *state;
  yyjson_val *payloads;
  yyjson_val *payload;
  char linked_id[SS_SHA256_HEX_CAPACITY + 8U];
  char linked_logical[SS_SHA256_HEX_CAPACITY + 8U];
  char payload_path[SS_RESULT_TEXT_CAPACITY];
  const char *kind;
  SsStatus status = SS_OK;
  if (snprintf(path, sizeof(path), "%s/checkpoint.json", directory) < 0) {
    return SS_ERROR_OVERFLOW;
  }
  document = yyjson_read_file(path, YYJSON_READ_NOFLAG, NULL, NULL);
  if (document == NULL) {
    return errno == ENOENT ? SS_ERROR_NOT_FOUND : SS_ERROR_JSON;
  }
  root = yyjson_doc_get_root(document);
  version = yyjson_obj_get(root, "schema_version");
  identity = yyjson_obj_get(root, "identity");
  state = yyjson_obj_get(root, "accepted_state");
  payloads = yyjson_obj_get(root, "payloads");
  payload = yyjson_obj_get(payloads, "checkpoint.h5");
  kind = yyjson_get_str(yyjson_obj_get(root, "kind"));
  if (!yyjson_is_obj(root) || kind == NULL ||
      strcmp(kind, "slope_stability.checkpoint") != 0 ||
      yyjson_get_uint(yyjson_obj_get(version, "major")) != 1U ||
      !ss_result_json_string(root, "id", checkpoint_id, SS_SHA256_HEX_CAPACITY + 8U) ||
      !ss_result_json_string(identity, "checkpoint_id", linked_id, sizeof(linked_id)) ||
      !ss_result_json_string(identity, "logical_digest", logical_digest,
                             SS_SHA256_HEX_CAPACITY + 8U) ||
      !ss_result_json_string(payload, "logical_digest", linked_logical,
                             sizeof(linked_logical)) ||
      !ss_result_json_string(payload, "sha256", payload_digest,
                             SS_SHA256_HEX_CAPACITY) ||
      !ss_result_json_string(payload, "path", payload_path, sizeof(payload_path)) ||
      strcmp(checkpoint_id, linked_id) != 0 ||
      strcmp(logical_digest, linked_logical) != 0 ||
      strcmp(payload_path, "checkpoint.h5") != 0 ||
      !ss_valid_content_id(checkpoint_id) || !ss_valid_content_id(logical_digest) ||
      !ss_valid_lower_hex(payload_digest, SS_SHA256_HEX_CAPACITY - 1U) ||
      !yyjson_get_bool(yyjson_obj_get(state, "accepted")) ||
      !yyjson_get_bool(yyjson_obj_get(state, "complete")) ||
      yyjson_get_uint(yyjson_obj_get(state, "state_id")) == 0U ||
      yyjson_get_uint(yyjson_obj_get(payload, "bytes")) == 0U) {
    status = SS_ERROR_SCHEMA;
  } else {
    *payload_bytes = yyjson_get_uint(yyjson_obj_get(payload, "bytes"));
    *state_id = yyjson_get_uint(yyjson_obj_get(state, "state_id"));
  }
  yyjson_doc_free(document);
  return status;
}
