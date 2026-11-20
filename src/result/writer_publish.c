/* Digest and publish terminal result artifacts.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "result_envelope.h"
#include "slope_internal/core/sort.h"
#include "writer_internal.h"

#include "yyjson.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int ss_result_field_record_compare(const void *left_raw, const void *right_raw) {
  const SsResultFieldRecord *left = (const SsResultFieldRecord *)left_raw;
  const SsResultFieldRecord *right = (const SsResultFieldRecord *)right_raw;
  return strcmp(left->quantity_id, right->quantity_id);
}

static int ss_result_scalar_record_compare(const void *left_raw,
                                           const void *right_raw) {
  const SsResultScalarRecord *left = (const SsResultScalarRecord *)left_raw;
  const SsResultScalarRecord *right = (const SsResultScalarRecord *)right_raw;
  const int identity = strcmp(left->quantity_id, right->quantity_id);
  return identity != 0 ? identity
                       : (left->state_id < right->state_id
                              ? -1
                              : (left->state_id > right->state_id ? 1 : 0));
}

static int ss_result_checkpoint_record_compare(const void *left_raw,
                                               const void *right_raw) {
  const SsResultCheckpointRecord *left = (const SsResultCheckpointRecord *)left_raw;
  const SsResultCheckpointRecord *right = (const SsResultCheckpointRecord *)right_raw;
  if (left->state_id != right->state_id) {
    return left->state_id < right->state_id ? -1 : 1;
  }
  return strcmp(left->checkpoint_id, right->checkpoint_id);
}

static SsStatus ss_result_writer_logical_digest(SsResultWriter *writer) {
  SsResultFieldRecord *ordered = NULL;
  SsResultScalarRecord *ordered_scalars = NULL;
  SsResultCheckpointRecord *ordered_checkpoints = NULL;
  unsigned char *payload = NULL;
  size_t capacity = 0U;
  size_t cursor = 0U;
  size_t index;
  size_t field_count;
  size_t scalar_count;
  size_t checkpoint_count;
  size_t state_count;
  size_t history_count;
  SsStatus status = SS_OK;
  if (writer == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  field_count = writer->field_count;
  scalar_count = writer->scalar_count;
  checkpoint_count = writer->checkpoint_count;
  state_count = writer->state_count;
  history_count = writer->history_count;
  if (field_count > writer->field_capacity || scalar_count > writer->scalar_capacity ||
      checkpoint_count > writer->checkpoint_capacity ||
      state_count > writer->state_capacity ||
      (field_count != 0U && writer->fields == NULL) ||
      (scalar_count != 0U && writer->scalars == NULL) ||
      (checkpoint_count != 0U && writer->checkpoints == NULL) ||
      (state_count != 0U && writer->states == NULL)) {
    status = SS_ERROR_INTERNAL;
  }
  if (writer->rank == 0) {
    capacity = 1024U;
#define SS_RESULT_DIGEST_CAPACITY_ADD(count, bytes)                                    \
  do {                                                                                 \
    if ((count) > (SIZE_MAX - capacity) / (bytes)) {                                   \
      status = SS_ERROR_OVERFLOW;                                                      \
    } else {                                                                           \
      capacity += (count) * (bytes);                                                   \
    }                                                                                  \
  } while (0)
    SS_RESULT_DIGEST_CAPACITY_ADD(field_count, 4096U);
    SS_RESULT_DIGEST_CAPACITY_ADD(scalar_count, 1024U);
    SS_RESULT_DIGEST_CAPACITY_ADD(checkpoint_count, SS_RESULT_PATH_CAPACITY + 512U);
    SS_RESULT_DIGEST_CAPACITY_ADD(state_count, 14U * sizeof(uint64_t));
#undef SS_RESULT_DIGEST_CAPACITY_ADD
    if (status == SS_OK) {
      ordered = (SsResultFieldRecord *)malloc((field_count == 0U ? 1U : field_count) *
                                              sizeof(*ordered));
      ordered_scalars = (SsResultScalarRecord *)malloc(
          (scalar_count == 0U ? 1U : scalar_count) * sizeof(*ordered_scalars));
      ordered_checkpoints = (SsResultCheckpointRecord *)malloc(
          (checkpoint_count == 0U ? 1U : checkpoint_count) *
          sizeof(*ordered_checkpoints));
      payload = (unsigned char *)malloc(capacity);
    }
    if (status == SS_OK && (ordered == NULL || ordered_scalars == NULL ||
                            ordered_checkpoints == NULL || payload == NULL)) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
    if (status == SS_OK) {
      int count = snprintf(
          (char *)payload, capacity,
          "run=%s|status=%s|analysis=%s|states=%llu|history=%llu|"
          "scalars=%llu|checkpoints=%llu|restart=%s|%s|%s|%llu|%u|",
          writer->run_id, ss_result_terminal_status_name(writer->terminal_status),
          writer->analysis_method, (unsigned long long)state_count,
          (unsigned long long)history_count, (unsigned long long)scalar_count,
          (unsigned long long)checkpoint_count,
          writer->restart_provenance_valid != 0 ? writer->parent_checkpoint_id : "-",
          writer->restart_provenance_valid != 0 ? writer->parent_run_id : "-",
          writer->restart_provenance_valid != 0 ? writer->parent_build_id : "-",
          (unsigned long long)writer->parent_accepted_state_id,
          writer->parent_source_index_bits);
      if (count < 0 || (size_t)count >= capacity) {
        status = SS_ERROR_OVERFLOW;
      } else {
        cursor = (size_t)count;
      }
    }
    if (status == SS_OK) {
      if (field_count != 0U) {
        memcpy(ordered, writer->fields, field_count * sizeof(*ordered));
        ss_sort(ordered, field_count, sizeof(*ordered), ss_result_field_record_compare);
      }
      for (index = 0U; index < field_count; ++index) {
        size_t state_index;
        int count = snprintf((char *)payload + cursor, capacity - cursor,
                             "%s|%u|%s|%u|%u|%u|%u|%llu|", ordered[index].quantity_id,
                             ordered[index].quantity_version, ordered[index].unit,
                             (unsigned int)ordered[index].dataset_kind,
                             ordered[index].component_count,
                             ordered[index].category_count, ordered[index].sample_count,
                             (unsigned long long)ordered[index].global_count);
        if (count < 0 || (size_t)count >= capacity - cursor) {
          status = SS_ERROR_OVERFLOW;
          break;
        }
        cursor += (size_t)count;
        for (state_index = 0U; state_index < ordered[index].state_count;
             ++state_index) {
          count = snprintf((char *)payload + cursor, capacity - cursor, "%s%llu",
                           state_index == 0U ? "" : ",",
                           (unsigned long long)ordered[index].state_ids[state_index]);
          if (count < 0 || (size_t)count >= capacity - cursor) {
            status = SS_ERROR_OVERFLOW;
            break;
          }
          cursor += (size_t)count;
        }
        if (status != SS_OK) {
          break;
        }
        count = snprintf((char *)payload + cursor, capacity - cursor, "|%s|",
                         ordered[index].logical_digest);
        if (count < 0 || (size_t)count >= capacity - cursor) {
          status = SS_ERROR_OVERFLOW;
          break;
        }
        cursor += (size_t)count;
      }
    }
    if (status == SS_OK) {
      if (scalar_count != 0U) {
        memcpy(ordered_scalars, writer->scalars,
               scalar_count * sizeof(*ordered_scalars));
        ss_sort(ordered_scalars, scalar_count, sizeof(*ordered_scalars),
                ss_result_scalar_record_compare);
      }
      for (index = 0U; index < scalar_count; ++index) {
        const SsResultScalarRecord *scalar = &ordered_scalars[index];
        unsigned char encoded[2U * sizeof(uint64_t)];
        int count = snprintf((char *)payload + cursor, capacity - cursor,
                             "scalar=%s|%u|%s|%llu|", scalar->quantity_id,
                             scalar->quantity_version, scalar->unit,
                             (unsigned long long)scalar->state_id);
        if (count < 0 || (size_t)count >= capacity - cursor) {
          status = SS_ERROR_OVERFLOW;
          break;
        }
        cursor += (size_t)count;
        if (cursor > capacity - sizeof(encoded)) {
          status = SS_ERROR_OVERFLOW;
          break;
        }
        ss_result_store_double_le(encoded, scalar->value);
        ss_result_store_u64_le(encoded + sizeof(uint64_t),
                               (uint64_t)(scalar->complete != 0));
        memcpy(payload + cursor, encoded, sizeof(encoded));
        cursor += sizeof(encoded);
      }
    }
    if (status == SS_OK) {
      if (checkpoint_count != 0U) {
        memcpy(ordered_checkpoints, writer->checkpoints,
               checkpoint_count * sizeof(*ordered_checkpoints));
        ss_sort(ordered_checkpoints, checkpoint_count, sizeof(*ordered_checkpoints),
                ss_result_checkpoint_record_compare);
      }
      for (index = 0U; index < checkpoint_count; ++index) {
        const SsResultCheckpointRecord *checkpoint = &ordered_checkpoints[index];
        int count = snprintf(
            (char *)payload + cursor, capacity - cursor, "checkpoint=%llu|%s|%s|%d|",
            (unsigned long long)checkpoint->state_id, checkpoint->checkpoint_id,
            checkpoint->relative_path, checkpoint->complete != 0);
        if (count < 0 || (size_t)count >= capacity - cursor) {
          status = SS_ERROR_OVERFLOW;
          break;
        }
        cursor += (size_t)count;
      }
    }
    if (status == SS_OK) {
      for (index = 0U; index < state_count; ++index) {
        const SsResultAcceptedState *state = &writer->states[index];
        unsigned char encoded[14U * sizeof(uint64_t)];
        size_t offset = 0U;
        ss_result_store_u64_le(encoded + offset, state->state_id);
        offset += 8U;
        ss_result_store_u64_le(encoded + offset, state->parent_state_id);
        offset += 8U;
        ss_result_store_u64_le(encoded + offset, state->accepted_ordinal);
        offset += 8U;
        ss_result_store_u64_le(encoded + offset, state->hydraulic_generation);
        offset += 8U;
        ss_result_store_u64_le(encoded + offset, state->mechanics_generation);
        offset += 8U;
        ss_result_store_u64_le(encoded + offset, state->control_generation);
        offset += 8U;
        ss_result_store_double_le(encoded + offset, state->factor);
        offset += 8U;
        ss_result_store_double_le(encoded + offset, state->work);
        offset += 8U;
        ss_result_store_double_le(encoded + offset, state->direct_control);
        offset += 8U;
        ss_result_store_double_le(encoded + offset, state->displacement_maximum);
        offset += 8U;
        ss_result_store_u64_le(encoded + offset, state->history_begin);
        offset += 8U;
        ss_result_store_u64_le(encoded + offset, state->history_end);
        offset += 8U;
        ss_result_store_u64_le(encoded + offset, (uint64_t)(state->complete != 0));
        offset += 8U;
        ss_result_store_u64_le(encoded + offset, (uint64_t)(state->accepted != 0));
        offset += 8U;
        if (cursor + offset > capacity) {
          status = SS_ERROR_OVERFLOW;
          break;
        }
        memcpy(payload + cursor, encoded, offset);
        cursor += offset;
      }
    }
    if (status == SS_OK) {
      /* History rows are streamed back from the closed file in row-ID order. */
      SsSha256 context;
      ss_sha256_init(&context);
      status = ss_sha256_update(&context, payload, cursor);
      if (status == SS_OK) {
        status = ss_result_writer_history_digest(writer, writer->hdf5_temporary_path,
                                                 &context);
      }
      if (status == SS_OK) {
        memcpy(writer->summary.logical_digest, "sha256:", 7U);
        ss_sha256_final(&context, writer->summary.logical_digest + 7U);
      }
    }
  }
  status = ss_result_broadcast_digest(writer->communicator, status,
                                      writer->summary.logical_digest);
  free(payload);
  free(ordered_checkpoints);
  free(ordered_scalars);
  free(ordered);
  return status;
}

static SsStatus ss_result_writer_result_id(SsResultWriter *writer) {
  char identity[1024];
  int count;
  int status_value;
  SsStatus status = SS_OK;
  if (writer->rank == 0) {
    count = snprintf(identity, sizeof(identity), "%s|%s|%s|%s", writer->run_id,
                     ss_result_terminal_status_name(writer->terminal_status),
                     writer->stop_reason, writer->summary.logical_digest);
    if (count < 0 || (size_t)count >= sizeof(identity)) {
      status = SS_ERROR_OVERFLOW;
    } else {
      memcpy(writer->summary.result_id, "sha256:", 7U);
      if (ss_sha256_bytes((const unsigned char *)identity, (size_t)count,
                          writer->summary.result_id + 7U) != SS_OK) {
        status = SS_ERROR_INTERNAL;
      }
    }
  }
  status_value = (int)status;
  if (MPI_Bcast(&status_value, 1, MPI_INT, 0, writer->communicator) != MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  status = (SsStatus)status_value;
  if (status != SS_OK) {
    return status;
  }
  return MPI_Bcast(writer->summary.result_id, SS_SHA256_HEX_CAPACITY + 7, MPI_CHAR, 0,
                   writer->communicator) == MPI_SUCCESS
             ? SS_OK
             : SS_ERROR_INTERNAL;
}

static SsStatus ss_result_writer_envelope(SsResultWriter *writer,
                                          unsigned char **output, size_t *output_size) {
  yyjson_mut_doc *document = NULL;
  yyjson_mut_val *root;
  yyjson_mut_val *version;
  yyjson_mut_val *created;
  yyjson_mut_val *provenance;
  yyjson_mut_val *restart = NULL;
  yyjson_mut_val *identity;
  yyjson_mut_val *run;
  yyjson_mut_val *terminal;
  yyjson_mut_val *analysis;
  yyjson_mut_val *summary;
  yyjson_mut_val *quantities;
  yyjson_mut_val *scalars;
  yyjson_mut_val *checkpoints;
  yyjson_mut_val *events;
  yyjson_mut_val *artifacts;
  yyjson_mut_val *payload;
  yyjson_mut_val *environment_payload;
  yyjson_mut_val *errors;
  yyjson_mut_val *extensions;
  size_t index;
  if (writer == NULL || output == NULL || output_size == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *output = NULL;
  *output_size = 0U;
  document = yyjson_mut_doc_new(NULL);
  root = document == NULL ? NULL : yyjson_mut_obj(document);
  if (document == NULL || root == NULL) {
    yyjson_mut_doc_free(document);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  yyjson_mut_doc_set_root(document, root);
  version = yyjson_mut_obj_add_obj(document, root, "schema_version");
  created = yyjson_mut_obj_add_obj(document, root, "created_by");
  provenance = yyjson_mut_obj_add_obj(document, root, "provenance");
  identity = yyjson_mut_obj_add_obj(document, root, "identity");
  run = yyjson_mut_obj_add_obj(document, root, "run");
  terminal = yyjson_mut_obj_add_obj(document, root, "terminal_status");
  analysis = yyjson_mut_obj_add_obj(document, root, "analysis_outcome");
  summary = yyjson_mut_obj_add_obj(document, root, "summary");
  artifacts = yyjson_mut_obj_add_obj(document, root, "artifacts");
  errors = yyjson_mut_obj_add_obj(document, root, "errors");
  extensions = yyjson_mut_obj_add_obj(document, root, "extensions");
  if (version == NULL || created == NULL || provenance == NULL || identity == NULL ||
      run == NULL || terminal == NULL || analysis == NULL || summary == NULL ||
      artifacts == NULL || errors == NULL || extensions == NULL ||
      !ss_result_envelope_header(document, root, version, "slope_stability.result",
                                 writer->summary.result_id, 0U) ||
      !yyjson_mut_obj_add_str(document, created, "tool", "slope-engine") ||
      !yyjson_mut_obj_add_str(document, created, "tool_version", "0.1.0") ||
      !yyjson_mut_obj_add_str(document, created, "source_id", writer->build_id) ||
      !yyjson_mut_obj_add_str(document, provenance, "build_id", writer->build_id) ||
      !yyjson_mut_obj_add_str(document, provenance, "environment",
                              "environment.json") ||
      !ss_result_envelope_identity(document, identity, "result_id",
                                   writer->summary.result_id,
                                   writer->summary.logical_digest) ||
      !ss_result_envelope_run(document, run, writer->run_id, writer->problem_id,
                              writer->build_id) ||
      !yyjson_mut_obj_add_str(document, run, "mesh_id", writer->mesh_id) ||
      !yyjson_mut_obj_add_str(document, run, "geometry_id", writer->geometry_id) ||
      !yyjson_mut_obj_add_str(
          document, terminal, "status",
          ss_result_terminal_status_name(writer->terminal_status)) ||
      !yyjson_mut_obj_add_bool(document, terminal, "complete", true) ||
      !yyjson_mut_obj_add_str(document, terminal, "stop_reason", writer->stop_reason) ||
      !yyjson_mut_obj_add_str(document, analysis, "method", writer->analysis_method) ||
      !yyjson_mut_obj_add_str(
          document, analysis, "outcome",
          writer->analysis_outcome[0] != '\0'
              ? writer->analysis_outcome
              : (writer->terminal_status == SS_RESULT_SUCCEEDED
                     ? "unspecified"
                     : (writer->terminal_status == SS_RESULT_FAILED ? "failed"
                                                                    : "cancelled"))) ||
      !yyjson_mut_obj_add_uint(document, summary, "accepted_state_count",
                               writer->state_count) ||
      !yyjson_mut_obj_add_uint(document, summary, "history_row_count",
                               writer->history_count) ||
      !yyjson_mut_obj_add_str(document, summary, "validation_policy",
                              writer->validation_policy[0] != '\0'
                                  ? writer->validation_policy
                                  : "unspecified") ||
      !yyjson_mut_obj_add_uint(document, summary, "dimension", writer->dimension) ||
      !yyjson_mut_obj_add_uint(document, summary, "field_order", writer->field_order) ||
      !yyjson_mut_obj_add_uint(document, summary, "geometry_order",
                               writer->geometry_order) ||
      !yyjson_mut_obj_add_str(document, summary, "collective_plan_digest",
                              writer->collective_plan_digest)) {
    yyjson_mut_doc_free(document);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  if (writer->mechanics_assembly_mode[0] != '\0' &&
      !yyjson_mut_obj_add_str(document, summary, "mechanics_assembly_mode",
                              writer->mechanics_assembly_mode)) {
    yyjson_mut_doc_free(document);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  if (writer->restart_provenance_valid != 0) {
    restart = yyjson_mut_obj_add_obj(document, provenance, "restart");
    if (restart == NULL ||
        !yyjson_mut_obj_add_str(document, restart, "mode", "accepted_boundary") ||
        !yyjson_mut_obj_add_str(document, restart, "parent_checkpoint_id",
                                writer->parent_checkpoint_id) ||
        !yyjson_mut_obj_add_str(document, restart, "parent_run_id",
                                writer->parent_run_id) ||
        !yyjson_mut_obj_add_str(document, restart, "parent_build_id",
                                writer->parent_build_id) ||
        !yyjson_mut_obj_add_uint(document, restart, "parent_accepted_state_id",
                                 writer->parent_accepted_state_id) ||
        !yyjson_mut_obj_add_uint(document, restart, "source_index_bits",
                                 writer->parent_source_index_bits)) {
      yyjson_mut_doc_free(document);
      return SS_ERROR_OUT_OF_MEMORY;
    }
  }
  quantities = yyjson_mut_obj_add_arr(document, summary, "quantities");
  if (quantities == NULL) {
    yyjson_mut_doc_free(document);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (index = 0U; index < writer->field_count; ++index) {
    const SsResultFieldRecord *field = &writer->fields[index];
    yyjson_mut_val *quantity = yyjson_mut_arr_add_obj(document, quantities);
    yyjson_mut_val *state_ids =
        quantity == NULL ? NULL
                         : yyjson_mut_obj_add_arr(document, quantity, "state_ids");
    size_t state_index;
    if (quantity == NULL || state_ids == NULL ||
        !yyjson_mut_obj_add_str(document, quantity, "id", field->quantity_id) ||
        !yyjson_mut_obj_add_uint(document, quantity, "version",
                                 field->quantity_version) ||
        !yyjson_mut_obj_add_str(document, quantity, "unit", field->unit) ||
        !yyjson_mut_obj_add_str(
            document, quantity, "kind",
            ss_result_writer_dataset_kind_name(field->dataset_kind)) ||
        !yyjson_mut_obj_add_str(document, quantity, "location", field->location) ||
        !yyjson_mut_obj_add_str(document, quantity, "basis", field->basis) ||
        !yyjson_mut_obj_add_str(document, quantity, "quadrature_id",
                                field->quadrature_id) ||
        !yyjson_mut_obj_add_str(document, quantity, "group", field->group_path) ||
        !yyjson_mut_obj_add_uint(document, quantity, "state_count",
                                 field->state_count) ||
        !yyjson_mut_obj_add_uint(document, quantity, "entity_count",
                                 field->global_count) ||
        !yyjson_mut_obj_add_uint(document, quantity, "component_count",
                                 field->component_count) ||
        !yyjson_mut_obj_add_uint(document, quantity, "category_count",
                                 field->category_count) ||
        !yyjson_mut_obj_add_uint(document, quantity, "sample_count",
                                 field->sample_count) ||
        !yyjson_mut_obj_add_str(
            document, quantity, "value_type",
            field->dataset_kind == SS_RESULT_DATASET_QUADRATURE_CATEGORY ? "uint32"
                                                                         : "float64") ||
        !yyjson_mut_obj_add_str(document, quantity, "logical_digest",
                                field->logical_digest) ||
        !yyjson_mut_obj_add_bool(document, quantity, "complete", true)) {
      yyjson_mut_doc_free(document);
      return SS_ERROR_OUT_OF_MEMORY;
    }
    for (state_index = 0U; state_index < field->state_count; ++state_index) {
      if (!yyjson_mut_arr_add_uint(document, state_ids,
                                   field->state_ids[state_index])) {
        yyjson_mut_doc_free(document);
        return SS_ERROR_OUT_OF_MEMORY;
      }
    }
  }
  scalars = yyjson_mut_obj_add_arr(document, summary, "scalars");
  checkpoints = yyjson_mut_obj_add_arr(document, summary, "checkpoints");
  events = yyjson_mut_obj_add_arr(document, summary, "events");
  if (scalars == NULL || checkpoints == NULL || events == NULL) {
    yyjson_mut_doc_free(document);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  for (index = 0U; index < writer->scalar_count; ++index) {
    const SsResultScalarRecord *scalar = &writer->scalars[index];
    yyjson_mut_val *value = yyjson_mut_arr_add_obj(document, scalars);
    if (value == NULL ||
        !yyjson_mut_obj_add_str(document, value, "id", scalar->quantity_id) ||
        !yyjson_mut_obj_add_uint(document, value, "version",
                                 scalar->quantity_version) ||
        !yyjson_mut_obj_add_str(document, value, "unit", scalar->unit) ||
        !yyjson_mut_obj_add_uint(document, value, "state_id", scalar->state_id) ||
        !yyjson_mut_obj_add_real(document, value, "value", scalar->value) ||
        !yyjson_mut_obj_add_bool(document, value, "complete", scalar->complete != 0)) {
      yyjson_mut_doc_free(document);
      return SS_ERROR_OUT_OF_MEMORY;
    }
  }
  for (index = 0U; index < writer->checkpoint_count; ++index) {
    const SsResultCheckpointRecord *checkpoint = &writer->checkpoints[index];
    yyjson_mut_val *value = yyjson_mut_arr_add_obj(document, checkpoints);
    if (value == NULL ||
        !yyjson_mut_obj_add_uint(document, value, "state_id", checkpoint->state_id) ||
        !yyjson_mut_obj_add_str(document, value, "checkpoint_id",
                                checkpoint->checkpoint_id) ||
        !yyjson_mut_obj_add_str(document, value, "path", checkpoint->relative_path) ||
        !yyjson_mut_obj_add_bool(document, value, "complete",
                                 checkpoint->complete != 0)) {
      yyjson_mut_doc_free(document);
      return SS_ERROR_OUT_OF_MEMORY;
    }
  }
  {
    static const char *const names[] = {
        "result_pack",
        "result_hdf5",
        "result_digest",
    };
    const double seconds[] = {
        writer->summary.pack_seconds,
        writer->summary.hdf5_seconds,
        writer->summary.digest_seconds,
    };
    for (index = 0U; index < sizeof(names) / sizeof(names[0]); ++index) {
      yyjson_mut_val *event = yyjson_mut_arr_add_obj(document, events);
      if (event == NULL ||
          !yyjson_mut_obj_add_str(document, event, "name", names[index]) ||
          !yyjson_mut_obj_add_real(document, event, "seconds", seconds[index]) ||
          !yyjson_mut_obj_add_str(document, event, "unit", "s") ||
          !yyjson_mut_obj_add_bool(document, event, "complete", true)) {
        yyjson_mut_doc_free(document);
        return SS_ERROR_OUT_OF_MEMORY;
      }
    }
  }
  payload = yyjson_mut_obj_add_obj(document, artifacts, "result.h5");
  environment_payload = yyjson_mut_obj_add_obj(document, artifacts, "environment.json");
  if (payload == NULL || !ss_result_envelope_payload(document, payload, "result.h5",
                                                     writer->summary.payload_bytes,
                                                     writer->summary.payload_sha256,
                                                     writer->summary.logical_digest)) {
    yyjson_mut_doc_free(document);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  if (environment_payload == NULL ||
      !yyjson_mut_obj_add_str(document, environment_payload, "path",
                              "environment.json") ||
      !yyjson_mut_obj_add_uint(document, environment_payload, "bytes",
                               writer->environment_bytes) ||
      !yyjson_mut_obj_add_str(document, environment_payload, "sha256",
                              writer->environment_sha256) ||
      !yyjson_mut_obj_add_bool(document, environment_payload, "authoritative", false)) {
    yyjson_mut_doc_free(document);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  if (writer->terminal_error.valid != 0 &&
      (!yyjson_mut_obj_add_str(document, errors, "code", writer->terminal_error.code) ||
       !yyjson_mut_obj_add_str(document, errors, "message",
                               writer->terminal_error.message) ||
       !yyjson_mut_obj_add_uint(document, errors, "status",
                                (uint64_t)writer->terminal_error.status))) {
    yyjson_mut_doc_free(document);
    return SS_ERROR_OUT_OF_MEMORY;
  }
  return ss_result_envelope_serialize(document, output, output_size);
}

SsStatus ss_result_writer_publish(SsResultWriter *writer) {
  unsigned char *envelope = NULL;
  size_t envelope_size = 0U;
  SsResultWriterSummary verified;
  uint64_t minimum_bytes = 0U;
  uint64_t maximum_bytes = 0U;
  char parent[SS_RESULT_PATH_CAPACITY];
  char name[SS_RESULT_TEXT_CAPACITY];
  double started;
  SsStatus status;
  if (writer == NULL || writer->state != SS_RESULT_WRITER_CLOSING) {
    return SS_ERROR_INVALID_STATE;
  }
  started = MPI_Wtime();
  status = ss_result_writer_finalize_quantities(writer);
  writer->summary.digest_seconds += MPI_Wtime() - started;
  started = MPI_Wtime();
  if (status == SS_OK) {
    status = ss_result_writer_write_relations(writer);
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK) {
    status = ss_result_writer_fail(writer, SS_RESULT_FAIL_FLUSH);
  }
  if (status == SS_OK && H5Fflush(writer->file, H5F_SCOPE_GLOBAL) < 0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK) {
    status = ss_result_writer_fail(writer, SS_RESULT_FAIL_HDF5_CLOSE);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_close(&writer->file, H5Fclose, status);
    status = ss_result_hdf5_close(&writer->access, H5Pclose, status);
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  writer->summary.hdf5_seconds += MPI_Wtime() - started;
  if (status != SS_OK) {
    writer->state = SS_RESULT_WRITER_PUBLICATION_FAILED;
    writer->summary.writer_state = writer->state;
    return status;
  }
  started = MPI_Wtime();
  status = ss_result_writer_fail(writer, SS_RESULT_FAIL_LOGICAL_DIGEST);
  if (status == SS_OK) {
    status = ss_result_writer_logical_digest(writer);
  }
  if (status == SS_OK) {
    status = ss_result_writer_fail(writer, SS_RESULT_FAIL_PAYLOAD_DIGEST);
  }
  if (status == SS_OK && writer->rank == 0) {
    status = ss_result_payload_digest(writer->hdf5_temporary_path,
                                      writer->summary.payload_sha256,
                                      &writer->summary.payload_bytes);
  }
  {
    int status_value = (int)status;
    if (MPI_Bcast(&status_value, 1, MPI_INT, 0, writer->communicator) != MPI_SUCCESS ||
        MPI_Bcast(writer->summary.payload_sha256, SS_SHA256_HEX_CAPACITY, MPI_CHAR, 0,
                  writer->communicator) != MPI_SUCCESS ||
        MPI_Bcast(&writer->summary.payload_bytes, 1, MPI_UINT64_T, 0,
                  writer->communicator) != MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    } else {
      status = (SsStatus)status_value;
    }
  }
  if (status == SS_OK) {
    status = ss_result_writer_result_id(writer);
  }
  writer->summary.digest_seconds += MPI_Wtime() - started;
  if (status != SS_OK) {
    writer->state = SS_RESULT_WRITER_PUBLICATION_FAILED;
    writer->summary.writer_state = writer->state;
    return status;
  }
  started = MPI_Wtime();
  status = ss_result_writer_fail(writer, SS_RESULT_FAIL_PAYLOAD_RENAME);
  if (status == SS_OK) {
    status = ss_result_rank_zero_rename(writer->communicator,
                                        writer->hdf5_temporary_path, writer->hdf5_path);
  }
  if (status == SS_OK) {
    status = ss_result_writer_fail(writer, SS_RESULT_FAIL_ENVELOPE_WRITE);
  }
  if (status == SS_OK && writer->rank == 0) {
    status = ss_result_writer_envelope(writer, &envelope, &envelope_size);
    if (status == SS_OK) {
      status = ss_result_write_staged_file(writer->result_temporary_path, envelope,
                                           envelope_size);
    }
  }
  {
    int status_value = (int)status;
    if (MPI_Bcast(&status_value, 1, MPI_INT, 0, writer->communicator) != MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    } else {
      status = (SsStatus)status_value;
    }
  }
  free(envelope);
  if (status == SS_OK) {
    status = ss_result_writer_fail(writer, SS_RESULT_FAIL_ENVELOPE_RENAME);
  }
  if (status == SS_OK) {
    status = ss_result_rank_zero_rename(
        writer->communicator, writer->result_temporary_path, writer->result_path);
  }
  if (status == SS_OK) {
    status = ss_result_writer_verify_hdf5(writer);
  }
  if (status == SS_OK) {
    memset(&verified, 0, sizeof(verified));
    status = ss_result_verify_terminal(writer->communicator, writer->staging_directory,
                                       &verified);
    if (status == SS_OK &&
        (verified.writer_state != SS_RESULT_WRITER_PUBLISHED ||
         verified.terminal_status != writer->terminal_status ||
         verified.field_count != writer->field_count ||
         verified.accepted_state_count != writer->state_count ||
         verified.history_row_count != writer->history_count ||
         verified.payload_bytes != writer->summary.payload_bytes ||
         strcmp(verified.payload_sha256, writer->summary.payload_sha256) != 0 ||
         strcmp(verified.logical_digest, writer->summary.logical_digest) != 0 ||
         strcmp(verified.result_id, writer->summary.result_id) != 0)) {
      status = SS_ERROR_INTEGRITY;
    }
  }
  if (status == SS_OK && writer->rank == 0) {
    status = ss_result_fsync_directory(writer->staging_directory);
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK) {
    status = ss_result_writer_fail(writer, SS_RESULT_FAIL_DIRECTORY_RENAME);
  }
  if (status == SS_OK) {
    status = ss_result_rank_zero_rename(writer->communicator, writer->staging_directory,
                                        writer->final_directory);
  }
  if (status == SS_OK &&
      ss_result_path_split(writer->final_directory, parent, sizeof(parent), name,
                           sizeof(name)) == SS_OK &&
      writer->rank == 0) {
    status = ss_result_fsync_directory(parent);
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  writer->summary.publication_seconds += MPI_Wtime() - started;
  if (status != SS_OK) {
    writer->state = SS_RESULT_WRITER_PUBLICATION_FAILED;
    writer->summary.writer_state = writer->state;
    return status;
  }
  if (MPI_Allreduce(&writer->local_io_bytes, &minimum_bytes, 1, MPI_UINT64_T, MPI_MIN,
                    writer->communicator) != MPI_SUCCESS ||
      MPI_Allreduce(&writer->local_io_bytes, &maximum_bytes, 1, MPI_UINT64_T, MPI_MAX,
                    writer->communicator) != MPI_SUCCESS) {
    writer->state = SS_RESULT_WRITER_PUBLICATION_FAILED;
    writer->summary.writer_state = writer->state;
    return SS_ERROR_INTERNAL;
  }
  writer->summary.minimum_rank_bytes = minimum_bytes;
  writer->summary.maximum_rank_bytes = maximum_bytes;
  writer->state = SS_RESULT_WRITER_PUBLISHED;
  writer->summary.writer_state = writer->state;
  return SS_OK;
}
