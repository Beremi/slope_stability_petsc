/* Declare and append result quantities, history and accepted states.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "writer_internal.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SS_RESULT_INITIAL_CAPACITY 8U

static SsStatus ss_result_reserve(void **items, size_t *capacity, size_t count,
                                  size_t item_size) {
  size_t target = SS_RESULT_INITIAL_CAPACITY;
  size_t bytes;
  void *replacement;
  if (items == NULL || capacity == NULL || item_size == 0U || count == SIZE_MAX) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  if (count + 1U <= *capacity) {
    return SS_OK;
  }
  if (*capacity != 0U && ss_checked_mul_size(*capacity, 2U, &target) != SS_OK) {
    return SS_ERROR_OVERFLOW;
  }
  if (target < count + 1U) {
    target = count + 1U;
  }
  if (ss_checked_mul_size(target, item_size, &bytes) != SS_OK) {
    return SS_ERROR_OVERFLOW;
  }
  replacement = realloc(*items, bytes);
  if (replacement == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  *items = replacement;
  *capacity = target;
  return SS_OK;
}

SsStatus
ss_result_writer_declare_quantity(SsResultWriter *writer,
                                  const SsResultQuantityDeclaration *declaration) {
  SsResultFieldView agreement;
  SsResultFileIndexSpec index_spec;
  const SsResultFileIndexPlan *plan = NULL;
  SsResultFieldRecord *record = NULL;
  char component_text[SS_RESULT_COMPONENT_CAPACITY * SS_RESULT_TEXT_CAPACITY];
  char group_path[SS_RESULT_PATH_CAPACITY];
  size_t layout_index = 0U;
  const char *quadrature_id;
  const char *root_path;
  SsResultDatasetKind dataset_kind;
  uint32_t sample_count;
  size_t index;
  SsStatus status = SS_OK;
  if (writer == NULL || declaration == NULL ||
      writer->state != SS_RESULT_WRITER_STAGED) {
    return SS_ERROR_INVALID_STATE;
  }
  dataset_kind = declaration->dataset_kind == 0 ? SS_RESULT_DATASET_FIELD
                                                : declaration->dataset_kind;
  sample_count = declaration->sample_count == 0U ? 1U : declaration->sample_count;
  quadrature_id =
      declaration->quadrature_id == NULL ? "none" : declaration->quadrature_id;
  if (!ss_result_name_valid(declaration->quantity_id) ||
      !ss_result_text_is_valid(declaration->unit, SS_RESULT_TEXT_CAPACITY) ||
      !ss_result_name_valid(declaration->location) ||
      !ss_result_name_valid(declaration->basis) ||
      declaration->quantity_version == 0U || declaration->component_ids == NULL ||
      declaration->component_count == 0U ||
      declaration->component_count > SS_RESULT_COMPONENT_CAPACITY ||
      declaration->global_count == 0U || dataset_kind < SS_RESULT_DATASET_FIELD ||
      dataset_kind > SS_RESULT_DATASET_CELL_REDUCTION ||
      (dataset_kind != SS_RESULT_DATASET_QUADRATURE && sample_count != 1U) ||
      (dataset_kind == SS_RESULT_DATASET_QUADRATURE &&
       !ss_result_name_valid(quadrature_id)) ||
      (dataset_kind != SS_RESULT_DATASET_QUADRATURE &&
       strcmp(quadrature_id, "none") != 0) ||
      (declaration->local_count != 0U && declaration->stable_ids == NULL)) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  for (index = 0U; status == SS_OK && index < declaration->component_count; ++index) {
    size_t other;
    if (!ss_result_name_valid(declaration->component_ids[index])) {
      status = SS_ERROR_INVALID_ARGUMENT;
      break;
    }
    for (other = 0U; other < index; ++other) {
      if (strcmp(declaration->component_ids[index],
                 declaration->component_ids[other]) == 0) {
        status = SS_ERROR_DUPLICATE;
        break;
      }
    }
  }
  for (index = 0U; status == SS_OK && index < writer->field_count; ++index) {
    if (strcmp(writer->fields[index].quantity_id, declaration->quantity_id) == 0) {
      status = SS_ERROR_DUPLICATE;
    }
  }
  if (status == SS_OK) {
    status = ss_result_writer_component_text(declaration->component_ids,
                                             declaration->component_count,
                                             component_text, sizeof(component_text));
  }
  memset(&agreement, 0, sizeof(agreement));
  agreement.quantity_id = declaration->quantity_id;
  agreement.quantity_version = declaration->quantity_version;
  agreement.unit = declaration->unit;
  agreement.location = declaration->location;
  agreement.basis = declaration->basis;
  agreement.component_ids = declaration->component_ids;
  agreement.component_count = declaration->component_count;
  agreement.global_count = declaration->global_count;
  agreement.dataset_kind = dataset_kind;
  agreement.quadrature_id = quadrature_id;
  agreement.sample_count = sample_count;
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK) {
    status = ss_result_writer_field_agree(writer, &agreement, dataset_kind,
                                          quadrature_id, sample_count);
  }
  if (status != SS_OK) {
    return status;
  }
  memset(&index_spec, 0, sizeof(index_spec));
  index_spec.communicator = writer->communicator;
  index_spec.stable_ids = declaration->stable_ids;
  index_spec.local_slots = declaration->local_slots;
  index_spec.local_count = declaration->local_count;
  index_spec.global_count = declaration->global_count;
  status = ss_result_writer_layout_acquire(writer, &index_spec, &layout_index);
  plan = status == SS_OK ? writer->layouts[layout_index] : NULL;
  root_path = dataset_kind == SS_RESULT_DATASET_FIELD
                  ? "/fields"
                  : (dataset_kind == SS_RESULT_DATASET_QUADRATURE ? "/quadrature"
                                                                  : "/cell_reductions");
  if (status == SS_OK && snprintf(group_path, sizeof(group_path), "%s/%s", root_path,
                                  declaration->quantity_id) < 0) {
    status = SS_ERROR_OVERFLOW;
  }
  if (status == SS_OK) {
    status = ss_result_reserve((void **)&writer->fields, &writer->field_capacity,
                               writer->field_count, sizeof(*writer->fields));
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK) {
    record = &writer->fields[writer->field_count];
    memset(record, 0, sizeof(*record));
    status = ss_result_quantity_declare(
        writer->file, writer->communicator, group_path, dataset_kind, sample_count,
        declaration->component_count, writer->dataset_chunk_rows, plan);
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK && record != NULL) {
    (void)ss_result_copy_text(record->quantity_id, sizeof(record->quantity_id),
                              declaration->quantity_id);
    (void)ss_result_copy_text(record->unit, sizeof(record->unit), declaration->unit);
    (void)ss_result_copy_text(record->location, sizeof(record->location),
                              declaration->location);
    (void)ss_result_copy_text(record->basis, sizeof(record->basis), declaration->basis);
    (void)ss_result_copy_text(record->quadrature_id, sizeof(record->quadrature_id),
                              quadrature_id);
    (void)ss_result_copy_text(record->component_ids, sizeof(record->component_ids),
                              component_text);
    (void)ss_result_copy_text(record->group_path, sizeof(record->group_path),
                              group_path);
    record->quantity_version = declaration->quantity_version;
    record->component_count = declaration->component_count;
    record->sample_count = sample_count;
    record->dataset_kind = dataset_kind;
    record->global_count = declaration->global_count;
    record->layout_index = layout_index;
    ++writer->field_count;
    writer->summary.field_count = (uint64_t)writer->field_count;
    writer->local_io_bytes += (uint64_t)declaration->local_count * sizeof(SsEntityId);
  } else if (status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status != SS_OK && record != NULL) {
    memset(record, 0, sizeof(*record));
  }
  return status;
}

SsStatus
ss_result_writer_declare_category(SsResultWriter *writer,
                                  const SsResultCategoryDeclaration *declaration) {
  SsResultCategoryView agreement;
  SsResultFileIndexSpec index_spec;
  const SsResultFileIndexPlan *plan = NULL;
  SsResultFieldRecord *record = NULL;
  char category_text[SS_RESULT_COMPONENT_CAPACITY * SS_RESULT_TEXT_CAPACITY];
  char group_path[SS_RESULT_PATH_CAPACITY];
  size_t layout_index = 0U;
  size_t index;
  SsStatus status = SS_OK;
  if (writer == NULL || declaration == NULL ||
      writer->state != SS_RESULT_WRITER_STAGED) {
    return SS_ERROR_INVALID_STATE;
  }
  if (!ss_result_name_valid(declaration->quantity_id) ||
      !ss_result_text_is_valid(declaration->unit, SS_RESULT_TEXT_CAPACITY) ||
      !ss_result_name_valid(declaration->location) ||
      !ss_result_name_valid(declaration->quadrature_id) ||
      declaration->quantity_version == 0U || declaration->category_ids == NULL ||
      declaration->category_count == 0U ||
      declaration->category_count > SS_RESULT_COMPONENT_CAPACITY ||
      declaration->global_count == 0U || declaration->sample_count == 0U ||
      (declaration->local_count != 0U && declaration->stable_ids == NULL)) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  for (index = 0U; status == SS_OK && index < declaration->category_count; ++index) {
    size_t other;
    if (!ss_result_name_valid(declaration->category_ids[index])) {
      status = SS_ERROR_INVALID_ARGUMENT;
      break;
    }
    for (other = 0U; other < index; ++other) {
      if (strcmp(declaration->category_ids[index], declaration->category_ids[other]) ==
          0) {
        status = SS_ERROR_DUPLICATE;
        break;
      }
    }
  }
  for (index = 0U; status == SS_OK && index < writer->field_count; ++index) {
    if (strcmp(writer->fields[index].quantity_id, declaration->quantity_id) == 0) {
      status = SS_ERROR_DUPLICATE;
    }
  }
  if (status == SS_OK) {
    status = ss_result_writer_component_text(declaration->category_ids,
                                             declaration->category_count, category_text,
                                             sizeof(category_text));
  }
  memset(&agreement, 0, sizeof(agreement));
  agreement.quantity_id = declaration->quantity_id;
  agreement.quantity_version = declaration->quantity_version;
  agreement.unit = declaration->unit;
  agreement.location = declaration->location;
  agreement.quadrature_id = declaration->quadrature_id;
  agreement.category_ids = declaration->category_ids;
  agreement.category_count = declaration->category_count;
  agreement.global_count = declaration->global_count;
  agreement.sample_count = declaration->sample_count;
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK) {
    status = ss_result_writer_category_agree(writer, &agreement);
  }
  if (status != SS_OK) {
    return status;
  }
  memset(&index_spec, 0, sizeof(index_spec));
  index_spec.communicator = writer->communicator;
  index_spec.stable_ids = declaration->stable_ids;
  index_spec.local_slots = declaration->local_slots;
  index_spec.local_count = declaration->local_count;
  index_spec.global_count = declaration->global_count;
  status = ss_result_writer_layout_acquire(writer, &index_spec, &layout_index);
  plan = status == SS_OK ? writer->layouts[layout_index] : NULL;
  if (status == SS_OK &&
      snprintf(group_path, sizeof(group_path), "/quadrature_categories/%s",
               declaration->quantity_id) < 0) {
    status = SS_ERROR_OVERFLOW;
  }
  if (status == SS_OK) {
    status = ss_result_reserve((void **)&writer->fields, &writer->field_capacity,
                               writer->field_count, sizeof(*writer->fields));
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK) {
    record = &writer->fields[writer->field_count];
    memset(record, 0, sizeof(*record));
    status = ss_result_quantity_declare(writer->file, writer->communicator, group_path,
                                        SS_RESULT_DATASET_QUADRATURE_CATEGORY,
                                        declaration->sample_count, 1U,
                                        writer->dataset_chunk_rows, plan);
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK && record != NULL) {
    (void)ss_result_copy_text(record->quantity_id, sizeof(record->quantity_id),
                              declaration->quantity_id);
    (void)ss_result_copy_text(record->unit, sizeof(record->unit), declaration->unit);
    (void)ss_result_copy_text(record->location, sizeof(record->location),
                              declaration->location);
    (void)ss_result_copy_text(record->basis, sizeof(record->basis), "categorical");
    (void)ss_result_copy_text(record->quadrature_id, sizeof(record->quadrature_id),
                              declaration->quadrature_id);
    (void)ss_result_copy_text(record->category_ids, sizeof(record->category_ids),
                              category_text);
    (void)ss_result_copy_text(record->group_path, sizeof(record->group_path),
                              group_path);
    record->quantity_version = declaration->quantity_version;
    record->category_count = declaration->category_count;
    record->sample_count = declaration->sample_count;
    record->dataset_kind = SS_RESULT_DATASET_QUADRATURE_CATEGORY;
    record->global_count = declaration->global_count;
    record->layout_index = layout_index;
    ++writer->field_count;
    writer->summary.field_count = (uint64_t)writer->field_count;
    writer->local_io_bytes += (uint64_t)declaration->local_count * sizeof(SsEntityId);
  } else if (status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status != SS_OK && record != NULL) {
    memset(record, 0, sizeof(*record));
  }
  return status;
}

/* Buffered rows that trigger a flush when an accepted state closes a window. */
#define SS_RESULT_HISTORY_FLUSH_ROWS 512U

SsStatus ss_result_writer_append_history_rows(SsResultWriter *writer,
                                              const SsResultHistoryRow *rows,
                                              size_t count) {
  size_t buffered;
  size_t index;
  SsStatus status = SS_OK;
  if (writer == NULL || (rows == NULL && count != 0U) ||
      writer->state != SS_RESULT_WRITER_STREAMING) {
    return SS_ERROR_INVALID_STATE;
  }
  for (index = 0U; status == SS_OK && index < count; ++index) {
    const SsResultHistoryRow *row = &rows[index];
    const uint64_t expected = (uint64_t)writer->history_count + (uint64_t)index;
    if (row->kind < SS_RESULT_HISTORY_ACCEPTED ||
        row->kind > SS_RESULT_HISTORY_PMG_APPLY || row->complete == 0 ||
        !isfinite(row->control) || !isfinite(row->residual) ||
        !isfinite(row->tolerance) || !isfinite(row->reference) ||
        !isfinite(row->step) || !isfinite(row->auxiliary) ||
        row->role > SS_RESULT_ROLE_SEEPAGE || row->row_id != expected ||
        (row->row_id == 0U && row->parent_row_id != 0U) ||
        (row->row_id != 0U && row->parent_row_id >= row->row_id) ||
        (row->accepted != 0 && row->state_id == 0U) ||
        ((row->kind == SS_RESULT_HISTORY_ACCEPTED) != (row->accepted != 0))) {
      status = row->row_id < expected ? SS_ERROR_DUPLICATE : SS_ERROR_INVALID_STATE;
    }
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  status = ss_result_writer_history_agree(writer, rows, count);
  if (status != SS_OK) {
    return status;
  }
  buffered = writer->history_count - (size_t)writer->history_buffer_begin;
  if (count > SIZE_MAX - buffered) {
    status = SS_ERROR_OVERFLOW;
  }
  while (status == SS_OK && buffered + count > writer->history_capacity) {
    status = ss_result_reserve((void **)&writer->history, &writer->history_capacity,
                               writer->history_capacity, sizeof(*writer->history));
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  if (count != 0U) {
    memcpy(writer->history + buffered, rows, count * sizeof(*rows));
  }
  writer->history_count += count;
  writer->summary.history_row_count = (uint64_t)writer->history_count;
  return SS_OK;
}

SsStatus ss_result_writer_append_history(SsResultWriter *writer,
                                         const SsResultHistoryRow *row) {
  if (row == NULL) {
    return SS_ERROR_INVALID_STATE;
  }
  return ss_result_writer_append_history_rows(writer, row, 1U);
}

SsStatus ss_result_writer_accept_state(SsResultWriter *writer,
                                       const SsResultAcceptedState *state) {
  size_t index;
  size_t accepted_rows = 0U;
  SsStatus status = SS_OK;
  if (writer == NULL || state == NULL || writer->state != SS_RESULT_WRITER_STREAMING) {
    return SS_ERROR_INVALID_STATE;
  }
  /* A window may not reach back into rows already closed by an earlier state. */
  if (state->state_id == 0U || state->complete == 0 || state->accepted == 0 ||
      !isfinite(state->factor) || !isfinite(state->work) ||
      !isfinite(state->direct_control) || !isfinite(state->displacement_maximum) ||
      state->displacement_maximum < 0.0 || state->history_begin >= state->history_end ||
      state->history_end > (uint64_t)writer->history_count ||
      state->history_begin < writer->history_buffer_begin) {
    status = SS_ERROR_INVALID_STATE;
  }
  if (status == SS_OK && state->parent_state_id != 0U) {
    int found = 0;
    for (index = 0U; index < writer->state_count; ++index) {
      if (writer->states[index].state_id == state->parent_state_id) {
        found = 1;
      }
    }
    if (found == 0) {
      status = SS_ERROR_INVALID_STATE;
    }
  }
  for (index = 0U; status == SS_OK && index < writer->state_count; ++index) {
    if (writer->states[index].state_id == state->state_id) {
      status = SS_ERROR_DUPLICATE;
    }
  }
  for (index = (size_t)(state->history_begin - writer->history_buffer_begin);
       status == SS_OK &&
       index < (size_t)(state->history_end - writer->history_buffer_begin);
       ++index) {
    const SsResultHistoryRow *row = &writer->history[index];
    if (row->complete == 0 ||
        (row->state_id != 0U && row->state_id != state->state_id)) {
      status = SS_ERROR_INVALID_STATE;
    } else if (row->kind == SS_RESULT_HISTORY_ACCEPTED) {
      if (row->accepted == 0 || row->state_id != state->state_id) {
        status = SS_ERROR_INVALID_STATE;
      } else {
        ++accepted_rows;
      }
    }
  }
  if (status == SS_OK && accepted_rows != 1U) {
    status = SS_ERROR_INVALID_STATE;
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  status = ss_result_writer_state_agree(writer, state);
  if (status != SS_OK) {
    return status;
  }
  status = ss_result_reserve((void **)&writer->states, &writer->state_capacity,
                             writer->state_count, sizeof(*writer->states));
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  writer->states[writer->state_count++] = *state;
  writer->summary.accepted_state_count = (uint64_t)writer->state_count;
  if (writer->history_count - (size_t)writer->history_buffer_begin >=
      SS_RESULT_HISTORY_FLUSH_ROWS) {
    status = ss_result_writer_flush_history(writer, state->history_end);
  }
  return status;
}

SsStatus ss_result_writer_write_quantity(SsResultWriter *writer,
                                         const SsResultFieldView *field) {
  SsResultFileIndexSpec index_spec;
  SsResultFileIndexPlan *temporary = NULL;
  const SsResultFileIndexPlan *plan = NULL;
  SsResultFieldRecord *record = NULL;
  char group_path[SS_RESULT_PATH_CAPACITY];
  char component_text[SS_RESULT_COMPONENT_CAPACITY * SS_RESULT_TEXT_CAPACITY];
  const char *quadrature_id;
  const char *root_path;
  SsResultDatasetKind dataset_kind;
  uint32_t sample_count;
  uint64_t logical_bytes = 0U;
  uint64_t local_bytes;
  size_t index;
  size_t state_position = 0U;
  int existing = 0;
  int state_found = 0;
  double started;
  SsStatus status = SS_OK;
  if (writer == NULL || field == NULL || writer->state != SS_RESULT_WRITER_STREAMING) {
    return SS_ERROR_INVALID_STATE;
  }
  dataset_kind =
      field->dataset_kind == 0 ? SS_RESULT_DATASET_FIELD : field->dataset_kind;
  sample_count = field->sample_count == 0U ? 1U : field->sample_count;
  quadrature_id = field->quadrature_id == NULL ? "none" : field->quadrature_id;
  if (!ss_result_name_valid(field->quantity_id) ||
      !ss_result_text_is_valid(field->unit, SS_RESULT_TEXT_CAPACITY) ||
      !ss_result_name_valid(field->location) || !ss_result_name_valid(field->basis) ||
      field->quantity_version == 0U || field->component_ids == NULL ||
      field->component_count == 0U ||
      field->component_count > SS_RESULT_COMPONENT_CAPACITY || field->state_id == 0U ||
      field->global_count == 0U || sample_count == 0U ||
      dataset_kind < SS_RESULT_DATASET_FIELD ||
      dataset_kind > SS_RESULT_DATASET_CELL_REDUCTION ||
      (dataset_kind != SS_RESULT_DATASET_QUADRATURE && sample_count != 1U) ||
      (dataset_kind == SS_RESULT_DATASET_QUADRATURE &&
       !ss_result_name_valid(quadrature_id)) ||
      (dataset_kind != SS_RESULT_DATASET_QUADRATURE &&
       strcmp(quadrature_id, "none") != 0) ||
      (field->local_count != 0U &&
       (field->stable_ids == NULL || field->values == NULL))) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  for (index = 0U; status == SS_OK && index < field->component_count; ++index) {
    size_t other;
    if (!ss_result_name_valid(field->component_ids[index])) {
      status = SS_ERROR_INVALID_ARGUMENT;
      break;
    }
    for (other = 0U; other < index; ++other) {
      if (strcmp(field->component_ids[index], field->component_ids[other]) == 0) {
        status = SS_ERROR_DUPLICATE;
        break;
      }
    }
  }
  for (index = 0U; status == SS_OK && index < writer->state_count; ++index) {
    if (writer->states[index].state_id == field->state_id) {
      state_found = 1;
      state_position = index;
      break;
    }
  }
  if (status == SS_OK && state_found == 0) {
    status = SS_ERROR_INVALID_STATE;
  }
  for (index = 0U; status == SS_OK && index < writer->field_count; ++index) {
    if (strcmp(writer->fields[index].quantity_id, field->quantity_id) == 0) {
      record = &writer->fields[index];
      existing = 1;
      break;
    }
  }
  if (status == SS_OK && existing == 0) {
    status = SS_ERROR_SCHEMA;
  }
  if (status == SS_OK && existing != 0) {
    if (record->dataset_kind != dataset_kind ||
        record->quantity_version != field->quantity_version ||
        record->component_count != field->component_count ||
        record->sample_count != sample_count ||
        record->global_count != field->global_count ||
        strcmp(record->unit, field->unit) != 0 ||
        strcmp(record->location, field->location) != 0 ||
        strcmp(record->basis, field->basis) != 0 ||
        strcmp(record->quadrature_id, quadrature_id) != 0) {
      status = SS_ERROR_SCHEMA;
    }
    for (index = 0U; status == SS_OK && index < record->state_count; ++index) {
      if (record->state_ids[index] == field->state_id) {
        status = SS_ERROR_DUPLICATE;
      }
    }
    if (status == SS_OK && record->state_count != 0U) {
      size_t previous_position;
      int previous_found = 0;
      for (previous_position = 0U; previous_position < writer->state_count;
           ++previous_position) {
        if (writer->states[previous_position].state_id ==
            record->state_ids[record->state_count - 1U]) {
          previous_found = 1;
          break;
        }
      }
      if (previous_found == 0 || state_position <= previous_position) {
        status = SS_ERROR_INVALID_STATE;
      }
    }
  }
  if (status == SS_OK) {
    status =
        ss_result_writer_component_text(field->component_ids, field->component_count,
                                        component_text, sizeof(component_text));
  }
  if (status == SS_OK && existing != 0 &&
      strcmp(record->component_ids, component_text) != 0) {
    status = SS_ERROR_SCHEMA;
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  status = ss_result_writer_field_agree(writer, field, dataset_kind, quadrature_id,
                                        sample_count);
  if (status != SS_OK) {
    return status;
  }
  status = record == NULL ? SS_ERROR_INTERNAL : SS_OK;
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  if (record == NULL) {
    return SS_ERROR_INTERNAL;
  }
  status = ss_result_writer_fail(writer, SS_RESULT_FAIL_DATA_WRITE);
  if (status != SS_OK) {
    return status;
  }
  memset(&index_spec, 0, sizeof(index_spec));
  index_spec.communicator = writer->communicator;
  index_spec.stable_ids = field->stable_ids;
  index_spec.local_slots = field->local_slots;
  index_spec.local_count = field->local_count;
  index_spec.global_count = field->global_count;
  started = MPI_Wtime();
  status = ss_result_writer_layout_bind(writer, record->layout_index, &index_spec,
                                        &temporary, &plan);
  if (status != SS_OK) {
    return status;
  }
  root_path = dataset_kind == SS_RESULT_DATASET_FIELD
                  ? "/fields"
                  : (dataset_kind == SS_RESULT_DATASET_QUADRATURE ? "/quadrature"
                                                                  : "/cell_reductions");
  if (snprintf(group_path, sizeof(group_path), "%s/%s", root_path, field->quantity_id) <
      0) {
    status = SS_ERROR_OVERFLOW;
  }
  if (status == SS_OK) {
    status = ss_result_reserve((void **)&record->state_ids, &record->state_capacity,
                               record->state_count, sizeof(*record->state_ids));
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK) {
    status = ss_result_quantity_append(
        writer->file, writer->communicator, group_path, dataset_kind, field->state_id,
        field->component_count, sample_count, plan, field->values, record->state_count,
        &logical_bytes);
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK && writer->summary.logical_bytes < record->logical_bytes) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK) {
    writer->summary.logical_bytes -= record->logical_bytes;
    record->state_ids[record->state_count++] = field->state_id;
    record->logical_bytes = logical_bytes;
    writer->summary.field_count = (uint64_t)writer->field_count;
    writer->summary.logical_bytes += logical_bytes;
    local_bytes = (uint64_t)field->local_count * (uint64_t)sample_count *
                  (uint64_t)field->component_count * sizeof(double);
    writer->local_io_bytes += local_bytes;
    writer->summary.pack_seconds += MPI_Wtime() - started;
  }
  if (ss_result_file_index_plan_destroy(&temporary) != SS_OK && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  return ss_status_allreduce_max(writer->communicator, status);
}

SsStatus ss_result_writer_write_field(SsResultWriter *writer,
                                      const SsResultFieldView *field) {
  SsResultFieldView normalized;
  if (writer == NULL || field == NULL || writer->state != SS_RESULT_WRITER_STREAMING) {
    return SS_ERROR_INVALID_STATE;
  }
  if ((field->dataset_kind != 0 && field->dataset_kind != SS_RESULT_DATASET_FIELD) ||
      field->sample_count > 1U || field->quadrature_id != NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  normalized = *field;
  normalized.dataset_kind = SS_RESULT_DATASET_FIELD;
  normalized.sample_count = 1U;
  normalized.quadrature_id = NULL;
  return ss_result_writer_write_quantity(writer, &normalized);
}

SsStatus ss_result_writer_write_category(SsResultWriter *writer,
                                         const SsResultCategoryView *category) {
  SsResultFileIndexSpec index_spec;
  SsResultFileIndexPlan *temporary = NULL;
  const SsResultFileIndexPlan *plan = NULL;
  SsResultFieldRecord *record = NULL;
  char group_path[SS_RESULT_PATH_CAPACITY];
  char category_text[SS_RESULT_COMPONENT_CAPACITY * SS_RESULT_TEXT_CAPACITY];
  uint64_t logical_bytes = 0U;
  uint64_t local_bytes;
  size_t index;
  size_t state_position = 0U;
  int existing = 0;
  int state_found = 0;
  double started;
  SsStatus status = SS_OK;
  if (writer == NULL || category == NULL ||
      writer->state != SS_RESULT_WRITER_STREAMING) {
    return SS_ERROR_INVALID_STATE;
  }
  if (!ss_result_name_valid(category->quantity_id) ||
      !ss_result_text_is_valid(category->unit, SS_RESULT_TEXT_CAPACITY) ||
      !ss_result_name_valid(category->location) ||
      !ss_result_name_valid(category->quadrature_id) ||
      category->quantity_version == 0U || category->category_ids == NULL ||
      category->category_count == 0U ||
      category->category_count > SS_RESULT_COMPONENT_CAPACITY ||
      category->state_id == 0U || category->global_count == 0U ||
      category->sample_count == 0U ||
      (category->local_count != 0U &&
       (category->stable_ids == NULL || category->values == NULL))) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  for (index = 0U; status == SS_OK && index < category->category_count; ++index) {
    size_t other;
    if (!ss_result_name_valid(category->category_ids[index])) {
      status = SS_ERROR_INVALID_ARGUMENT;
      break;
    }
    for (other = 0U; other < index; ++other) {
      if (strcmp(category->category_ids[index], category->category_ids[other]) == 0) {
        status = SS_ERROR_DUPLICATE;
        break;
      }
    }
  }
  for (index = 0U; status == SS_OK && index < writer->state_count; ++index) {
    if (writer->states[index].state_id == category->state_id) {
      state_found = 1;
      state_position = index;
      break;
    }
  }
  if (status == SS_OK && state_found == 0) {
    status = SS_ERROR_INVALID_STATE;
  }
  for (index = 0U; status == SS_OK && index < writer->field_count; ++index) {
    if (strcmp(writer->fields[index].quantity_id, category->quantity_id) == 0) {
      record = &writer->fields[index];
      existing = 1;
      break;
    }
  }
  if (status == SS_OK && existing == 0) {
    status = SS_ERROR_SCHEMA;
  }
  if (status == SS_OK && existing != 0) {
    if (record->dataset_kind != SS_RESULT_DATASET_QUADRATURE_CATEGORY ||
        record->quantity_version != category->quantity_version ||
        record->category_count != category->category_count ||
        record->sample_count != category->sample_count ||
        record->global_count != category->global_count ||
        strcmp(record->unit, category->unit) != 0 ||
        strcmp(record->location, category->location) != 0 ||
        strcmp(record->quadrature_id, category->quadrature_id) != 0) {
      status = SS_ERROR_SCHEMA;
    }
    for (index = 0U; status == SS_OK && index < record->state_count; ++index) {
      if (record->state_ids[index] == category->state_id) {
        status = SS_ERROR_DUPLICATE;
      }
    }
    if (status == SS_OK && record->state_count != 0U) {
      size_t previous_position;
      int previous_found = 0;
      for (previous_position = 0U; previous_position < writer->state_count;
           ++previous_position) {
        if (writer->states[previous_position].state_id ==
            record->state_ids[record->state_count - 1U]) {
          previous_found = 1;
          break;
        }
      }
      if (previous_found == 0 || state_position <= previous_position) {
        status = SS_ERROR_INVALID_STATE;
      }
    }
  }
  if (status == SS_OK) {
    status = ss_result_writer_component_text(category->category_ids,
                                             category->category_count, category_text,
                                             sizeof(category_text));
  }
  if (status == SS_OK && existing != 0 &&
      strcmp(record->category_ids, category_text) != 0) {
    status = SS_ERROR_SCHEMA;
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  status = ss_result_writer_category_agree(writer, category);
  if (status != SS_OK) {
    return status;
  }
  status = record == NULL ? SS_ERROR_INTERNAL : SS_OK;
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  if (record == NULL) {
    return SS_ERROR_INTERNAL;
  }
  status = ss_result_writer_fail(writer, SS_RESULT_FAIL_DATA_WRITE);
  if (status != SS_OK) {
    return status;
  }
  memset(&index_spec, 0, sizeof(index_spec));
  index_spec.communicator = writer->communicator;
  index_spec.stable_ids = category->stable_ids;
  index_spec.local_slots = category->local_slots;
  index_spec.local_count = category->local_count;
  index_spec.global_count = category->global_count;
  started = MPI_Wtime();
  status = ss_result_writer_layout_bind(writer, record->layout_index, &index_spec,
                                        &temporary, &plan);
  if (status != SS_OK) {
    return status;
  }
  if (snprintf(group_path, sizeof(group_path), "/quadrature_categories/%s",
               category->quantity_id) < 0) {
    status = SS_ERROR_OVERFLOW;
  }
  if (status == SS_OK) {
    status = ss_result_reserve((void **)&record->state_ids, &record->state_capacity,
                               record->state_count, sizeof(*record->state_ids));
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK) {
    status = ss_result_category_append(writer->file, writer->communicator, group_path,
                                       category->state_id, category->category_count,
                                       category->sample_count, plan, category->values,
                                       record->state_count, &logical_bytes);
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK && writer->summary.logical_bytes < record->logical_bytes) {
    status = SS_ERROR_INTERNAL;
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK) {
    writer->summary.logical_bytes -= record->logical_bytes;
    record->state_ids[record->state_count++] = category->state_id;
    record->logical_bytes = logical_bytes;
    writer->summary.field_count = (uint64_t)writer->field_count;
    writer->summary.logical_bytes += logical_bytes;
    local_bytes = (uint64_t)category->local_count * (uint64_t)category->sample_count *
                  sizeof(uint32_t);
    writer->local_io_bytes += local_bytes;
    writer->summary.pack_seconds += MPI_Wtime() - started;
  }
  if (ss_result_file_index_plan_destroy(&temporary) != SS_OK && status == SS_OK) {
    status = SS_ERROR_INTERNAL;
  }
  return ss_status_allreduce_max(writer->communicator, status);
}

SsStatus ss_result_writer_append_scalar(SsResultWriter *writer,
                                        const SsResultScalarValue *scalar) {
  char payload[1024];
  unsigned char agreement[2048];
  size_t payload_size;
  size_t index;
  int count;
  int state_found = 0;
  SsStatus status = SS_OK;
  if (writer == NULL || scalar == NULL || writer->state != SS_RESULT_WRITER_STREAMING) {
    return SS_ERROR_INVALID_STATE;
  }
  if (!ss_result_name_valid(scalar->quantity_id) ||
      !ss_result_text_is_valid(scalar->unit, SS_RESULT_TEXT_CAPACITY) ||
      scalar->quantity_version == 0U || scalar->state_id == 0U ||
      !isfinite(scalar->value) || scalar->complete == 0) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  for (index = 0U; status == SS_OK && index < writer->state_count; ++index) {
    if (writer->states[index].state_id == scalar->state_id) {
      state_found = 1;
      break;
    }
  }
  if (status == SS_OK && state_found == 0) {
    status = SS_ERROR_INVALID_STATE;
  }
  for (index = 0U; status == SS_OK && index < writer->field_count; ++index) {
    if (strcmp(writer->fields[index].quantity_id, scalar->quantity_id) == 0) {
      status = SS_ERROR_DUPLICATE;
    }
  }
  for (index = 0U; status == SS_OK && index < writer->scalar_count; ++index) {
    if (strcmp(writer->scalars[index].quantity_id, scalar->quantity_id) == 0) {
      status = SS_ERROR_DUPLICATE;
    }
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  count = snprintf(payload, sizeof(payload), "%s|%u|%s|%llu|", scalar->quantity_id,
                   scalar->quantity_version, scalar->unit,
                   (unsigned long long)scalar->state_id);
  if (count < 0 || (size_t)count >= sizeof(payload) ||
      (size_t)count > sizeof(agreement) - 16U) {
    status = SS_ERROR_OVERFLOW;
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  payload_size = (size_t)count;
  memcpy(agreement, payload, payload_size);
  ss_result_store_double_le(agreement + payload_size, scalar->value);
  payload_size += sizeof(double);
  ss_result_store_u64_le(agreement + payload_size, (uint64_t)(scalar->complete != 0));
  payload_size += sizeof(uint64_t);
  status =
      ss_result_collective_payload_agree(writer->communicator, agreement, payload_size);
  if (status == SS_OK) {
    status = ss_result_reserve((void **)&writer->scalars, &writer->scalar_capacity,
                               writer->scalar_count, sizeof(*writer->scalars));
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK) {
    SsResultScalarRecord *record = &writer->scalars[writer->scalar_count++];
    memset(record, 0, sizeof(*record));
    (void)ss_result_copy_text(record->quantity_id, sizeof(record->quantity_id),
                              scalar->quantity_id);
    (void)ss_result_copy_text(record->unit, sizeof(record->unit), scalar->unit);
    record->quantity_version = scalar->quantity_version;
    record->state_id = scalar->state_id;
    record->value = scalar->value;
    record->complete = 1;
  }
  return status;
}

SsStatus
ss_result_writer_index_checkpoint(SsResultWriter *writer,
                                  const SsResultCheckpointReference *checkpoint) {
  char payload[2U * SS_RESULT_PATH_CAPACITY];
  size_t index;
  int count;
  int state_found = 0;
  SsStatus status = SS_OK;
  if (writer == NULL || checkpoint == NULL ||
      writer->state != SS_RESULT_WRITER_STREAMING) {
    return SS_ERROR_INVALID_STATE;
  }
  if (checkpoint->state_id == 0U || !ss_valid_content_id(checkpoint->checkpoint_id) ||
      !ss_result_relative_path_valid(checkpoint->relative_path) ||
      checkpoint->complete == 0) {
    status = SS_ERROR_INVALID_ARGUMENT;
  }
  for (index = 0U; status == SS_OK && index < writer->state_count; ++index) {
    if (writer->states[index].state_id == checkpoint->state_id) {
      state_found = 1;
      break;
    }
  }
  if (status == SS_OK && state_found == 0) {
    status = SS_ERROR_INVALID_STATE;
  }
  for (index = 0U; status == SS_OK && index < writer->checkpoint_count; ++index) {
    if (writer->checkpoints[index].state_id == checkpoint->state_id ||
        strcmp(writer->checkpoints[index].checkpoint_id, checkpoint->checkpoint_id) ==
            0 ||
        strcmp(writer->checkpoints[index].relative_path, checkpoint->relative_path) ==
            0) {
      status = SS_ERROR_DUPLICATE;
    }
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  count = snprintf(payload, sizeof(payload), "%llu|%s|%s|%d",
                   (unsigned long long)checkpoint->state_id, checkpoint->checkpoint_id,
                   checkpoint->relative_path, checkpoint->complete != 0);
  if (count < 0 || (size_t)count >= sizeof(payload)) {
    status = SS_ERROR_OVERFLOW;
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status != SS_OK) {
    return status;
  }
  status = ss_result_collective_payload_agree(
      writer->communicator, (const unsigned char *)payload, (size_t)count);
  if (status == SS_OK) {
    status =
        ss_result_reserve((void **)&writer->checkpoints, &writer->checkpoint_capacity,
                          writer->checkpoint_count, sizeof(*writer->checkpoints));
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK) {
    SsResultCheckpointRecord *record = &writer->checkpoints[writer->checkpoint_count++];
    memset(record, 0, sizeof(*record));
    record->state_id = checkpoint->state_id;
    (void)ss_result_copy_text(record->checkpoint_id, sizeof(record->checkpoint_id),
                              checkpoint->checkpoint_id);
    (void)ss_result_copy_text(record->relative_path, sizeof(record->relative_path),
                              checkpoint->relative_path);
    record->complete = 1;
  }
  return status;
}
