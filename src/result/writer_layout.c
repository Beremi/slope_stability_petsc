/* Entity layouts shared by result quantities: one file-index plan per layout.
 *
 * A layout is the distributed set of (stable ID, local slot) pairs that maps a
 * rank's local values to canonical file rows.  Quantities declared with the
 * same layout share one plan, and every append reuses the plan of its
 * quantity, so no route exchange or sort is repeated per state.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "writer_internal.h"

#include <stdlib.h>
#include <string.h>

/* 1 when the local spec arrays equal the plan's local entries. */
static int ss_result_layout_matches_local(const SsResultFileIndexPlan *plan,
                                          const SsResultFileIndexSpec *spec) {
  size_t index;
  if (plan->summary.global_count != spec->global_count ||
      plan->local_count != spec->local_count) {
    return 0;
  }
  for (index = 0U; index < spec->local_count; ++index) {
    const size_t slot = spec->local_slots == NULL ? index : spec->local_slots[index];
    if (plan->entries[index].stable_id != spec->stable_ids[index] ||
        plan->entries[index].local_slot != slot) {
      return 0;
    }
  }
  return 1;
}

/* Collective: 1 on every rank when every rank's spec matches the plan. */
static SsStatus ss_result_layout_matches(MPI_Comm communicator,
                                         const SsResultFileIndexPlan *plan,
                                         const SsResultFileIndexSpec *spec,
                                         int *matches) {
  int local = ss_result_layout_matches_local(plan, spec);
  if (MPI_Allreduce(&local, matches, 1, MPI_INT, MPI_MIN, communicator) !=
      MPI_SUCCESS) {
    return SS_ERROR_INTERNAL;
  }
  return SS_OK;
}

SsStatus ss_result_writer_layout_acquire(SsResultWriter *writer,
                                         const SsResultFileIndexSpec *spec,
                                         size_t *layout_index) {
  SsResultFileIndexPlan *plan = NULL;
  size_t index;
  SsStatus status = SS_OK;
  if (writer == NULL || spec == NULL || layout_index == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (index = 0U; index < writer->layout_count; ++index) {
    int matches = 0;
    status = ss_result_layout_matches(writer->communicator, writer->layouts[index],
                                      spec, &matches);
    if (status != SS_OK) {
      return status;
    }
    if (matches != 0) {
      *layout_index = index;
      return SS_OK;
    }
  }
  if (writer->layout_count == writer->layout_capacity) {
    const size_t capacity =
        writer->layout_capacity == 0U ? 4U : 2U * writer->layout_capacity;
    SsResultFileIndexPlan **layouts = (SsResultFileIndexPlan **)realloc(
        writer->layouts, capacity * sizeof(*writer->layouts));
    if (layouts == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    } else {
      writer->layouts = layouts;
      writer->layout_capacity = capacity;
    }
  }
  status = ss_status_allreduce_max(writer->communicator, status);
  if (status == SS_OK) {
    status = ss_result_file_index_plan_create(spec, &plan);
  }
  if (status != SS_OK) {
    return status;
  }
  writer->layouts[writer->layout_count] = plan;
  *layout_index = writer->layout_count++;
  return SS_OK;
}

SsStatus ss_result_writer_layout_bind(SsResultWriter *writer, size_t layout_index,
                                      const SsResultFileIndexSpec *spec,
                                      SsResultFileIndexPlan **temporary,
                                      const SsResultFileIndexPlan **plan) {
  const SsResultFileIndexPlan *layout;
  int matches = 0;
  int local_equal;
  SsStatus status;
  if (writer == NULL || spec == NULL || temporary == NULL || plan == NULL ||
      layout_index >= writer->layout_count) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *temporary = NULL;
  *plan = NULL;
  layout = writer->layouts[layout_index];
  status = ss_result_layout_matches(writer->communicator, layout, spec, &matches);
  if (status != SS_OK) {
    return status;
  }
  if (matches != 0) {
    *plan = layout;
    return SS_OK;
  }
  /*
   * The caller redistributed the same entities differently.  Canonical file
   * rows are owned by stable-ID range, so equal ID sets give equal sorted
   * segments; anything else is a different entity set and an integrity error.
   */
  status = ss_result_file_index_plan_create(spec, temporary);
  if (status != SS_OK) {
    return status;
  }
  local_equal = (*temporary)->sorted_offset == layout->sorted_offset &&
                (*temporary)->sorted_count == layout->sorted_count &&
                (layout->sorted_count == 0U ||
                 memcmp((*temporary)->sorted_ids, layout->sorted_ids,
                        layout->sorted_count * sizeof(*layout->sorted_ids)) == 0);
  if (MPI_Allreduce(&local_equal, &matches, 1, MPI_INT, MPI_MIN,
                    writer->communicator) != MPI_SUCCESS) {
    status = SS_ERROR_INTERNAL;
  } else if (matches == 0) {
    status = SS_ERROR_INTEGRITY;
  }
  if (status != SS_OK) {
    (void)ss_result_file_index_plan_destroy(temporary);
    return status;
  }
  *plan = *temporary;
  return SS_OK;
}

SsStatus ss_result_writer_layouts_destroy(SsResultWriter *writer) {
  size_t index;
  SsStatus status = SS_OK;
  if (writer == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (index = 0U; index < writer->layout_count; ++index) {
    if (ss_result_file_index_plan_destroy(&writer->layouts[index]) != SS_OK) {
      status = SS_ERROR_INTERNAL;
    }
  }
  free(writer->layouts);
  writer->layouts = NULL;
  writer->layout_count = 0U;
  writer->layout_capacity = 0U;
  return status;
}

SsStatus ss_result_writer_finalize_quantities(SsResultWriter *writer) {
  size_t index;
  SsStatus status = SS_OK;
  if (writer == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (index = 0U; status == SS_OK && index < writer->field_count; ++index) {
    SsResultFieldRecord *record = &writer->fields[index];
    const int category = record->dataset_kind == SS_RESULT_DATASET_QUADRATURE_CATEGORY;
    const SsResultQuantityFinal final = {
        .kind = record->dataset_kind,
        .quantity_id = record->quantity_id,
        .quantity_version = record->quantity_version,
        .unit = record->unit,
        .location = record->location,
        .basis = record->basis,
        .quadrature_id = record->quadrature_id,
        .mesh_id = writer->mesh_id,
        .geometry_id = writer->geometry_id,
        .field_order = writer->field_order,
        .geometry_order = writer->geometry_order,
        .semantic_ids = category ? record->category_ids : record->component_ids,
        .state_ids = record->state_ids,
        .state_count = (uint64_t)record->state_count,
    };
    if (record->state_count == 0U) {
      continue;
    }
    status =
        ss_result_quantity_finalize(writer->file, writer->communicator,
                                    record->group_path, &final, record->logical_digest);
  }
  return ss_status_allreduce_max(writer->communicator, status);
}
