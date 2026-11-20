/* Result declarations, specification validation and rank agreement.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "writer_internal.h"

#include "slope_internal/core/sha256.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdio.h>
#include <string.h>

int ss_result_relative_path_valid(const char *value) {
  const char *segment;
  const char *cursor;
  if (!ss_result_text_is_valid(value, SS_RESULT_PATH_CAPACITY) || value[0] == '/' ||
      value[0] == '\0') {
    return 0;
  }
  segment = value;
  for (cursor = value;; ++cursor) {
    if (*cursor == '/' || *cursor == '\0') {
      const size_t length = (size_t)(cursor - segment);
      if (length == 0U || (length == 1U && segment[0] == '.') ||
          (length == 2U && segment[0] == '.' && segment[1] == '.')) {
        return 0;
      }
      if (*cursor == '\0') {
        return 1;
      }
      segment = cursor + 1;
    }
  }
}

const char *ss_result_writer_dataset_kind_name(SsResultDatasetKind kind) {
  switch (kind) {
  case SS_RESULT_DATASET_FIELD:
    return "field";
  case SS_RESULT_DATASET_QUADRATURE:
    return "quadrature";
  case SS_RESULT_DATASET_CELL_REDUCTION:
    return "cell_reduction";
  case SS_RESULT_DATASET_QUADRATURE_CATEGORY:
    return "quadrature_category";
  default:
    return "invalid";
  }
}

SsStatus ss_result_writer_component_text(const char *const *component_ids,
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
    if (!ss_result_text_is_valid(component_id, SS_RESULT_TEXT_CAPACITY)) {
      return SS_ERROR_INVALID_ARGUMENT;
    }
    length = strlen(component_id);
    if (length > capacity - 1U || separator > capacity - 1U - length ||
        cursor > capacity - 1U - length - separator) {
      return SS_ERROR_OVERFLOW;
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

SsStatus ss_result_writer_history_agree(const SsResultWriter *writer,
                                        const SsResultHistoryRow *rows, size_t count) {
  SsSha256 context;
  char digest[SS_SHA256_HEX_CAPACITY];
  size_t index;
  SsStatus status = SS_OK;
  ss_sha256_init(&context);
  for (index = 0U; status == SS_OK && index < count; ++index) {
    const SsResultHistoryRow *row = &rows[index];
    unsigned char payload[18U * sizeof(uint64_t)];
    size_t cursor = 0U;
#define SS_RESULT_APPEND_HISTORY_U64(value)                                            \
  do {                                                                                 \
    ss_result_store_u64_le(payload + cursor, (uint64_t)(value));                       \
    cursor += sizeof(uint64_t);                                                        \
  } while (0)
#define SS_RESULT_APPEND_HISTORY_F64(value)                                            \
  do {                                                                                 \
    ss_result_store_double_le(payload + cursor, (value));                              \
    cursor += sizeof(double);                                                          \
  } while (0)
    SS_RESULT_APPEND_HISTORY_U64(row->kind);
    SS_RESULT_APPEND_HISTORY_U64(row->row_id);
    SS_RESULT_APPEND_HISTORY_U64(row->state_id);
    SS_RESULT_APPEND_HISTORY_U64(row->attempt_id);
    SS_RESULT_APPEND_HISTORY_U64(row->parent_row_id);
    SS_RESULT_APPEND_HISTORY_U64(row->ordinal);
    SS_RESULT_APPEND_HISTORY_F64(row->control);
    SS_RESULT_APPEND_HISTORY_F64(row->residual);
    SS_RESULT_APPEND_HISTORY_U64(row->complete != 0);
    SS_RESULT_APPEND_HISTORY_U64(row->accepted != 0);
    SS_RESULT_APPEND_HISTORY_U64(row->role);
    SS_RESULT_APPEND_HISTORY_U64((uint32_t)row->status);
    SS_RESULT_APPEND_HISTORY_U64((uint32_t)row->reason);
    SS_RESULT_APPEND_HISTORY_U64(row->iterations);
    SS_RESULT_APPEND_HISTORY_F64(row->tolerance);
    SS_RESULT_APPEND_HISTORY_F64(row->reference);
    SS_RESULT_APPEND_HISTORY_F64(row->step);
    SS_RESULT_APPEND_HISTORY_F64(row->auxiliary);
#undef SS_RESULT_APPEND_HISTORY_F64
#undef SS_RESULT_APPEND_HISTORY_U64
    status = ss_sha256_update(&context, payload, cursor);
  }
  ss_sha256_final(&context, digest);
  return ss_result_collective_digest_agree(writer->communicator, writer->size, digest,
                                           status);
}

SsStatus ss_result_writer_state_agree(const SsResultWriter *writer,
                                      const SsResultAcceptedState *state) {
  unsigned char payload[14U * sizeof(uint64_t)];
  size_t cursor = 0U;
#define SS_RESULT_APPEND_STATE_U64(value)                                              \
  do {                                                                                 \
    ss_result_store_u64_le(payload + cursor, (uint64_t)(value));                       \
    cursor += sizeof(uint64_t);                                                        \
  } while (0)
  SS_RESULT_APPEND_STATE_U64(state->state_id);
  SS_RESULT_APPEND_STATE_U64(state->parent_state_id);
  SS_RESULT_APPEND_STATE_U64(state->accepted_ordinal);
  SS_RESULT_APPEND_STATE_U64(state->hydraulic_generation);
  SS_RESULT_APPEND_STATE_U64(state->mechanics_generation);
  SS_RESULT_APPEND_STATE_U64(state->control_generation);
  ss_result_store_double_le(payload + cursor, state->factor);
  cursor += sizeof(double);
  ss_result_store_double_le(payload + cursor, state->work);
  cursor += sizeof(double);
  ss_result_store_double_le(payload + cursor, state->direct_control);
  cursor += sizeof(double);
  ss_result_store_double_le(payload + cursor, state->displacement_maximum);
  cursor += sizeof(double);
  SS_RESULT_APPEND_STATE_U64(state->history_begin);
  SS_RESULT_APPEND_STATE_U64(state->history_end);
  SS_RESULT_APPEND_STATE_U64(state->complete != 0);
  SS_RESULT_APPEND_STATE_U64(state->accepted != 0);
#undef SS_RESULT_APPEND_STATE_U64
  return ss_result_collective_payload_agree(writer->communicator, payload, cursor);
}

SsStatus ss_result_writer_field_agree(const SsResultWriter *writer,
                                      const SsResultFieldView *field,
                                      SsResultDatasetKind dataset_kind,
                                      const char *quadrature_id,
                                      uint32_t sample_count) {
  char payload[16384];
  size_t cursor = 0U;
  size_t index;
  int count = snprintf(
      payload, sizeof(payload),
      "%zu:%s|%u|%zu:%s|%zu:%s|%zu:%s|%u|%zu:%s|%u|%u|%llu|%llu",
      strlen(field->quantity_id), field->quantity_id, field->quantity_version,
      strlen(field->unit), field->unit, strlen(field->location), field->location,
      strlen(field->basis), field->basis, (unsigned int)dataset_kind,
      strlen(quadrature_id), quadrature_id, sample_count, field->component_count,
      (unsigned long long)field->state_id, (unsigned long long)field->global_count);
  if (count < 0 || (size_t)count >= sizeof(payload)) {
    return SS_ERROR_OVERFLOW;
  }
  cursor = (size_t)count;
  for (index = 0U; index < field->component_count; ++index) {
    count = snprintf(payload + cursor, sizeof(payload) - cursor, "|%zu:%s",
                     strlen(field->component_ids[index]), field->component_ids[index]);
    if (count < 0 || (size_t)count >= sizeof(payload) - cursor) {
      return SS_ERROR_OVERFLOW;
    }
    cursor += (size_t)count;
  }
  return ss_result_collective_payload_agree(writer->communicator,
                                            (const unsigned char *)payload, cursor);
}

SsStatus ss_result_writer_category_agree(const SsResultWriter *writer,
                                         const SsResultCategoryView *category) {
  char payload[16384];
  size_t cursor = 0U;
  size_t index;
  int count = snprintf(
      payload, sizeof(payload), "%zu:%s|%u|%zu:%s|%zu:%s|%zu:%s|%u|%u|%llu|%llu",
      strlen(category->quantity_id), category->quantity_id, category->quantity_version,
      strlen(category->unit), category->unit, strlen(category->location),
      category->location, strlen(category->quadrature_id), category->quadrature_id,
      category->sample_count, category->category_count,
      (unsigned long long)category->state_id,
      (unsigned long long)category->global_count);
  if (count < 0 || (size_t)count >= sizeof(payload)) {
    return SS_ERROR_OVERFLOW;
  }
  cursor = (size_t)count;
  for (index = 0U; index < category->category_count; ++index) {
    count =
        snprintf(payload + cursor, sizeof(payload) - cursor, "|%zu:%s",
                 strlen(category->category_ids[index]), category->category_ids[index]);
    if (count < 0 || (size_t)count >= sizeof(payload) - cursor) {
      return SS_ERROR_OVERFLOW;
    }
    cursor += (size_t)count;
  }
  return ss_result_collective_payload_agree(writer->communicator,
                                            (const unsigned char *)payload, cursor);
}

SsStatus ss_result_writer_terminal_agree(const SsResultWriter *writer,
                                         SsResultTerminalStatus status,
                                         const char *stop_reason,
                                         const SsErrorRecord *error) {
  char payload[SS_ERROR_MESSAGE_CAPACITY + SS_ERROR_CODE_CAPACITY + 512U];
  int count = snprintf(
      payload, sizeof(payload), "%d|%zu:%s|%d|%d|%u|%u|%llu|%llu|%u|%u|%zu:%s|%zu:%s",
      (int)status, strlen(stop_reason), stop_reason, error == NULL ? 0 : error->valid,
      error == NULL ? 0 : (int)error->status, error == NULL ? 0U : error->key.phase,
      error == NULL ? 0U : error->key.component,
      (unsigned long long)(error == NULL ? 0U : error->key.entity),
      (unsigned long long)(error == NULL ? 0U : error->key.quadrature),
      error == NULL ? 0U : error->key.thread, error == NULL ? 0U : error->key.rank,
      error == NULL ? 0U : strlen(error->code), error == NULL ? "" : error->code,
      error == NULL ? 0U : strlen(error->message), error == NULL ? "" : error->message);
  if (count < 0 || (size_t)count >= sizeof(payload)) {
    return SS_ERROR_OVERFLOW;
  }
  return ss_result_collective_payload_agree(
      writer->communicator, (const unsigned char *)payload, (size_t)count);
}

SsStatus ss_result_writer_spec_validate(const SsResultWriterSpec *spec) {
  int has_restart;
  if (spec == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  has_restart = spec->parent_checkpoint_id != NULL || spec->parent_run_id != NULL ||
                spec->parent_build_id != NULL;
  if (spec == NULL ||
      !ss_result_spec_identity_valid(spec->final_directory, spec->problem_id,
                                     spec->run_id, spec->build_id, spec->mesh_id,
                                     spec->geometry_id) ||
      !ss_result_name_valid(spec->analysis_method) ||
      (spec->dimension != 2U && spec->dimension != 3U) ||
      !ss_result_orders_valid(spec->field_order, spec->geometry_order) ||
      spec->dataset_chunk_rows > UINT32_MAX ||
      spec->mechanics_assembly_minimum_threads < 0 ||
      spec->mechanics_assembly_maximum_threads <
          spec->mechanics_assembly_minimum_threads ||
      (spec->mechanics_assembly_minimum_threads == 0 &&
       spec->mechanics_assembly_maximum_threads != 0) ||
      (has_restart != 0 &&
       (spec->parent_checkpoint_id == NULL || spec->parent_run_id == NULL ||
        spec->parent_build_id == NULL ||
        !ss_valid_content_id(spec->parent_checkpoint_id) ||
        !ss_valid_content_id(spec->parent_run_id) ||
        !ss_valid_content_id(spec->parent_build_id) ||
        strcmp(spec->parent_run_id, spec->run_id) == 0 ||
        spec->parent_accepted_state_id == 0U ||
        (spec->parent_source_index_bits != 32U &&
         spec->parent_source_index_bits != 64U))) ||
      (has_restart == 0 && (spec->parent_accepted_state_id != 0U ||
                            spec->parent_source_index_bits != 0U))) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return SS_OK;
}

SsStatus ss_result_writer_spec_agree(const SsResultWriterSpec *spec, int size) {
  char local[4096];
  char digest[SS_SHA256_HEX_CAPACITY];
  int count;
  SsStatus status = SS_OK;
  count = snprintf(
      local, sizeof(local),
      "%s|%s|%s|%s|%s|%s|%s|%u|%u|%u|%llu|restart=%s|%s|%s|%llu|%u|assembly=%d|%d",
      spec->final_directory, spec->problem_id, spec->run_id, spec->build_id,
      spec->mesh_id, spec->geometry_id, spec->analysis_method, spec->dimension,
      spec->field_order, spec->geometry_order,
      (unsigned long long)spec->dataset_chunk_rows,
      spec->parent_checkpoint_id == NULL ? "-" : spec->parent_checkpoint_id,
      spec->parent_run_id == NULL ? "-" : spec->parent_run_id,
      spec->parent_build_id == NULL ? "-" : spec->parent_build_id,
      (unsigned long long)spec->parent_accepted_state_id,
      spec->parent_source_index_bits, spec->mechanics_assembly_minimum_threads,
      spec->mechanics_assembly_maximum_threads);
  if (count < 0 || (size_t)count >= sizeof(local) ||
      ss_sha256_bytes((const unsigned char *)local, (size_t)count, digest) != SS_OK) {
    status = SS_ERROR_OVERFLOW;
  }
  status = ss_status_allreduce_max(spec->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  return ss_result_collective_digest_agree(spec->communicator, size, digest, SS_OK);
}

SsStatus ss_result_writer_copy_spec(SsResultWriter *writer,
                                    const SsResultWriterSpec *spec) {
  SsStatus status = ss_result_copy_text(
      writer->final_directory, sizeof(writer->final_directory), spec->final_directory);
  if (status == SS_OK) {
    status = ss_result_copy_text(writer->problem_id, sizeof(writer->problem_id),
                                 spec->problem_id);
  }
  if (status == SS_OK) {
    status = ss_result_copy_text(writer->run_id, sizeof(writer->run_id), spec->run_id);
  }
  if (status == SS_OK) {
    status =
        ss_result_copy_text(writer->build_id, sizeof(writer->build_id), spec->build_id);
  }
  if (status == SS_OK) {
    status =
        ss_result_copy_text(writer->mesh_id, sizeof(writer->mesh_id), spec->mesh_id);
  }
  if (status == SS_OK) {
    status = ss_result_copy_text(writer->geometry_id, sizeof(writer->geometry_id),
                                 spec->geometry_id);
  }
  if (status == SS_OK) {
    status =
        ss_result_copy_text(writer->analysis_method, sizeof(writer->analysis_method),
                            spec->analysis_method);
  }
  if (status == SS_OK && spec->parent_checkpoint_id != NULL) {
    status = ss_result_copy_text(writer->parent_checkpoint_id,
                                 sizeof(writer->parent_checkpoint_id),
                                 spec->parent_checkpoint_id);
  }
  if (status == SS_OK && spec->parent_run_id != NULL) {
    status = ss_result_copy_text(writer->parent_run_id, sizeof(writer->parent_run_id),
                                 spec->parent_run_id);
  }
  if (status == SS_OK && spec->parent_build_id != NULL) {
    status =
        ss_result_copy_text(writer->parent_build_id, sizeof(writer->parent_build_id),
                            spec->parent_build_id);
  }
  if (status == SS_OK && spec->parent_checkpoint_id != NULL) {
    writer->parent_accepted_state_id = spec->parent_accepted_state_id;
    writer->parent_source_index_bits = spec->parent_source_index_bits;
    writer->restart_provenance_valid = 1;
  }
  writer->dimension = spec->dimension;
  writer->field_order = spec->field_order;
  writer->geometry_order = spec->geometry_order;
  writer->mechanics_assembly_minimum_threads = spec->mechanics_assembly_minimum_threads;
  writer->mechanics_assembly_maximum_threads = spec->mechanics_assembly_maximum_threads;
  writer->dataset_chunk_rows = spec->dataset_chunk_rows;
  return status;
}
