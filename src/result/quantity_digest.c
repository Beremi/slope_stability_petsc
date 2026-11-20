/* Canonical logical digests of result quantities with bounded scratch.
 *
 * The digest is the stable `-chunks-v1/v2` quantity digest of the result-v1
 * layout: entity rows are grouped into chunks of 1024 canonical file rows, each
 * chunk hashes `stable_id || values(state, sample, component)` in little
 * endian, and rank zero hashes the semantic text followed by the ordered chunk
 * digests.  This implementation streams every chunk through an incremental
 * SHA-256 context with a fixed value scratch and gathers chunk digests to rank
 * zero in fixed-size batches, so neither the scratch nor rank-zero staging
 * grows with the number of states or entities.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "result_internal.h"

#include "slope_internal/core/sha256.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Maximum decoded value bytes held per read (all states of an entity block). */
#define SS_RESULT_DIGEST_VALUE_SCRATCH_BYTES ((size_t)8U << 20U)
/* Encoding buffer for little-endian canonical bytes fed to SHA-256. */
#define SS_RESULT_DIGEST_ENCODE_BYTES ((size_t)64U << 10U)
/* Maximum chunk digests staged on rank zero per gather round. */
#define SS_RESULT_DIGEST_GATHER_RECORDS 4096U
#define SS_RESULT_DIGEST_HEX (SS_SHA256_HEX_CAPACITY - 1U)

static void ss_result_quantity_store_u32_le(unsigned char output[4], uint32_t value) {
  output[0] = (unsigned char)(value & UINT32_C(0xff));
  output[1] = (unsigned char)((value >> 8U) & UINT32_C(0xff));
  output[2] = (unsigned char)((value >> 16U) & UINT32_C(0xff));
  output[3] = (unsigned char)((value >> 24U) & UINT32_C(0xff));
}

const char *ss_result_quantity_kind_name(SsResultDatasetKind kind) {
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
    return NULL;
  }
}

SsStatus ss_result_quantity_shape(hid_t ids_dataset, hid_t states_dataset,
                                  hid_t values_dataset, SsResultQuantityShape *shape) {
  hid_t ids_space = -1;
  hid_t states_space = -1;
  hid_t values_space = -1;
  hsize_t ids_dimensions[1] = {0U};
  hsize_t state_dimensions[1] = {0U};
  hsize_t value_dimensions[4] = {0U, 0U, 0U, 0U};
  char kind_name[SS_RESULT_TEXT_CAPACITY];
  int rank;
  SsStatus status = SS_OK;
  if (ids_dataset < 0 || states_dataset < 0 || values_dataset < 0 || shape == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(shape, 0, sizeof(*shape));
  ids_space = H5Dget_space(ids_dataset);
  states_space = H5Dget_space(states_dataset);
  values_space = H5Dget_space(values_dataset);
  rank = values_space < 0 ? -1 : H5Sget_simple_extent_ndims(values_space);
  if (ids_space < 0 || states_space < 0 || values_space < 0 ||
      H5Sget_simple_extent_ndims(ids_space) != 1 ||
      H5Sget_simple_extent_ndims(states_space) != 1 || (rank != 3 && rank != 4) ||
      H5Sget_simple_extent_dims(ids_space, ids_dimensions, NULL) != 1 ||
      H5Sget_simple_extent_dims(states_space, state_dimensions, NULL) != 1 ||
      H5Sget_simple_extent_dims(values_space, value_dimensions, NULL) != rank ||
      ids_dimensions[0] == 0U || state_dimensions[0] == 0U ||
      value_dimensions[0] != state_dimensions[0] ||
      value_dimensions[1] != ids_dimensions[0] || value_dimensions[rank - 1] == 0U ||
      value_dimensions[rank - 1] > UINT32_MAX ||
      (rank == 4 && (value_dimensions[2] == 0U || value_dimensions[2] > UINT32_MAX))) {
    status = SS_ERROR_SCHEMA;
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_string_attribute(values_dataset, "dataset_kind",
                                                  kind_name, sizeof(kind_name));
  }
  if (status == SS_OK) {
    if (strcmp(kind_name, "field") == 0) {
      shape->kind = SS_RESULT_DATASET_FIELD;
    } else if (strcmp(kind_name, "quadrature") == 0) {
      shape->kind = SS_RESULT_DATASET_QUADRATURE;
    } else if (strcmp(kind_name, "cell_reduction") == 0) {
      shape->kind = SS_RESULT_DATASET_CELL_REDUCTION;
    } else if (strcmp(kind_name, "quadrature_category") == 0) {
      shape->kind = SS_RESULT_DATASET_QUADRATURE_CATEGORY;
    } else {
      status = SS_ERROR_SCHEMA;
    }
  }
  if (status == SS_OK && ((shape->kind == SS_RESULT_DATASET_QUADRATURE && rank != 4) ||
                          (shape->kind != SS_RESULT_DATASET_QUADRATURE && rank != 3))) {
    status = SS_ERROR_SCHEMA;
  }
  if (status == SS_OK) {
    shape->states = (uint64_t)state_dimensions[0];
    shape->entities = (uint64_t)ids_dimensions[0];
    shape->samples = shape->kind == SS_RESULT_DATASET_QUADRATURE_CATEGORY
                         ? (uint32_t)value_dimensions[2]
                         : (rank == 4 ? (uint32_t)value_dimensions[2] : 1U);
    shape->components = shape->kind == SS_RESULT_DATASET_QUADRATURE_CATEGORY
                            ? 1U
                            : (uint32_t)value_dimensions[rank - 1];
    shape->value_rank = rank;
  }
  status = ss_result_hdf5_close(&values_space, H5Sclose, status);
  status = ss_result_hdf5_close(&states_space, H5Sclose, status);
  status = ss_result_hdf5_close(&ids_space, H5Sclose, status);
  return status;
}

SsStatus ss_result_quantity_state_text(const SsStateId *state_ids, uint64_t state_count,
                                       char *output, size_t capacity) {
  size_t cursor = 0U;
  uint64_t index;
  if (state_ids == NULL || state_count == 0U || output == NULL || capacity == 0U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  output[0] = '\0';
  for (index = 0U; index < state_count; ++index) {
    int count;
    uint64_t prior;
    if (state_ids[index] == 0U) {
      return SS_ERROR_INTEGRITY;
    }
    for (prior = 0U; prior < index; ++prior) {
      if (state_ids[index] == state_ids[prior]) {
        return SS_ERROR_INTEGRITY;
      }
    }
    count = snprintf(output + cursor, capacity - cursor, "%s%llu",
                     index == 0U ? "" : ",", (unsigned long long)state_ids[index]);
    if (count < 0 || (size_t)count >= capacity - cursor) {
      return SS_ERROR_OVERFLOW;
    }
    cursor += (size_t)count;
  }
  return SS_OK;
}

static SsStatus ss_result_quantity_read_state_text(hid_t states_dataset,
                                                   uint64_t state_count, char *output,
                                                   size_t capacity) {
  SsStateId *state_ids;
  SsStatus status = SS_OK;
  if (states_dataset < 0 || state_count == 0U || state_count > SIZE_MAX) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  state_ids = (SsStateId *)malloc((size_t)state_count * sizeof(*state_ids));
  if (state_ids == NULL) {
    return SS_ERROR_OUT_OF_MEMORY;
  }
  if (H5Dread(states_dataset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              state_ids) < 0) {
    status = SS_ERROR_HDF5;
  }
  if (status == SS_OK) {
    status = ss_result_quantity_state_text(state_ids, state_count, output, capacity);
  }
  free(state_ids);
  return status;
}

static SsStatus ss_result_quantity_semantic(hid_t values_dataset, hid_t states_dataset,
                                            const SsResultQuantityShape *shape,
                                            char *semantic, size_t semantic_capacity) {
  char quantity_id[SS_RESULT_TEXT_CAPACITY];
  char unit[SS_RESULT_TEXT_CAPACITY];
  char location[SS_RESULT_TEXT_CAPACITY];
  char basis[SS_RESULT_TEXT_CAPACITY];
  char quadrature_id[SS_RESULT_TEXT_CAPACITY];
  char mesh_id[SS_RESULT_TEXT_CAPACITY];
  char geometry_id[SS_RESULT_TEXT_CAPACITY];
  char semantic_ids[SS_RESULT_COMPONENT_CAPACITY * SS_RESULT_TEXT_CAPACITY];
  char state_text[SS_RESULT_STATE_TEXT_CAPACITY];
  uint64_t quantity_version = 0U;
  uint64_t field_order = 0U;
  uint64_t geometry_order = 0U;
  const char *kind_name;
  const int category = shape->kind == SS_RESULT_DATASET_QUADRATURE_CATEGORY;
  int count;
  SsStatus status;
  status = ss_result_hdf5_read_string_attribute(values_dataset, "quantity_id",
                                                quantity_id, sizeof(quantity_id));
  if (status == SS_OK) {
    status = ss_result_hdf5_read_u64_attribute(values_dataset, "quantity_version",
                                               &quantity_version);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_string_attribute(values_dataset, "unit", unit,
                                                  sizeof(unit));
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_string_attribute(values_dataset, "location", location,
                                                  sizeof(location));
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_string_attribute(values_dataset, "basis", basis,
                                                  sizeof(basis));
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_string_attribute(values_dataset, "quadrature_id",
                                                  quadrature_id, sizeof(quadrature_id));
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_string_attribute(values_dataset, "mesh_id", mesh_id,
                                                  sizeof(mesh_id));
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_string_attribute(values_dataset, "geometry_id",
                                                  geometry_id, sizeof(geometry_id));
  }
  if (status == SS_OK) {
    status =
        ss_result_hdf5_read_u64_attribute(values_dataset, "field_order", &field_order);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_u64_attribute(values_dataset, "geometry_order",
                                               &geometry_order);
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_string_attribute(
        values_dataset, category ? "category_ids" : "component_ids", semantic_ids,
        sizeof(semantic_ids));
  }
  if (status == SS_OK) {
    status = ss_result_quantity_read_state_text(states_dataset, shape->states,
                                                state_text, sizeof(state_text));
  }
  kind_name = ss_result_quantity_kind_name(shape->kind);
  if (status == SS_OK && kind_name == NULL) {
    status = SS_ERROR_SCHEMA;
  }
  if (status != SS_OK) {
    return status;
  }
  count =
      snprintf(semantic, semantic_capacity,
               "quantity=%s|version=%llu|unit=%s|kind=%s|location=%s|basis=%s|"
               "quadrature=%s|mesh=%s|geometry=%s|field_order=%llu|"
               "geometry_order=%llu|states=%s|%s=%s|shape=%llu,%llu,%u",
               quantity_id, (unsigned long long)quantity_version, unit, kind_name,
               location, basis, quadrature_id, mesh_id, geometry_id,
               (unsigned long long)field_order, (unsigned long long)geometry_order,
               state_text, category ? "categories" : "components", semantic_ids,
               (unsigned long long)shape->states, (unsigned long long)shape->entities,
               shape->kind == SS_RESULT_DATASET_FIELD ||
                       shape->kind == SS_RESULT_DATASET_CELL_REDUCTION
                   ? shape->components
                   : shape->samples);
  if (count >= 0 && (size_t)count < semantic_capacity &&
      shape->kind == SS_RESULT_DATASET_QUADRATURE) {
    const int extra = snprintf(semantic + count, semantic_capacity - (size_t)count,
                               ",%u", shape->components);
    count = extra < 0 ? -1 : count + extra;
  }
  if (count < 0 || (size_t)count >= semantic_capacity) {
    return SS_ERROR_OVERFLOW;
  }
  return SS_OK;
}

/* Bounded byte encoder feeding one SHA-256 context. */
typedef struct {
  SsSha256 context;
  unsigned char buffer[SS_RESULT_DIGEST_ENCODE_BYTES];
  size_t size;
} SsResultDigestEncoder;

static SsStatus ss_result_encoder_reserve(SsResultDigestEncoder *encoder,
                                          size_t bytes) {
  if (encoder->size + bytes > sizeof(encoder->buffer)) {
    const SsStatus status =
        ss_sha256_update(&encoder->context, encoder->buffer, encoder->size);
    encoder->size = 0U;
    return status;
  }
  return SS_OK;
}

typedef struct {
  hid_t ids_dataset;
  hid_t values_dataset;
  hid_t transfer;
  const SsResultQuantityShape *shape;
  void *scratch;
  SsEntityId *ids;
  SsResultDigestEncoder *encoder;
} SsResultDigestReader;

/* Read values[state_begin:state_begin+states, entity_begin:+entities, ...]. */
static SsStatus ss_result_digest_read_values(const SsResultDigestReader *reader,
                                             uint64_t state_begin, uint64_t states,
                                             uint64_t entity_begin, uint64_t entities) {
  const SsResultQuantityShape *shape = reader->shape;
  const int category = shape->kind == SS_RESULT_DATASET_QUADRATURE_CATEGORY;
  hsize_t start[4] = {(hsize_t)state_begin, (hsize_t)entity_begin, 0U, 0U};
  hsize_t count[4] = {(hsize_t)states, (hsize_t)entities, 0U, 0U};
  hid_t file_space = -1;
  hid_t memory_space = -1;
  SsStatus status = SS_OK;
  if (shape->value_rank == 4) {
    count[2] = (hsize_t)shape->samples;
    count[3] = (hsize_t)shape->components;
  } else {
    count[2] = (hsize_t)(category ? shape->samples : shape->components);
  }
  file_space = H5Dget_space(reader->values_dataset);
  memory_space = H5Screate_simple(shape->value_rank, count, NULL);
  if (file_space < 0 || memory_space < 0 ||
      H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL) < 0 ||
      H5Dread(reader->values_dataset, category ? H5T_NATIVE_UINT32 : H5T_NATIVE_DOUBLE,
              memory_space, file_space, reader->transfer, reader->scratch) < 0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_result_hdf5_close(&memory_space, H5Sclose, status);
  status = ss_result_hdf5_close(&file_space, H5Sclose, status);
  return status;
}

static SsStatus ss_result_digest_read_ids(const SsResultDigestReader *reader,
                                          uint64_t entity_begin, uint64_t entities) {
  hsize_t start[1] = {(hsize_t)entity_begin};
  hsize_t count[1] = {(hsize_t)entities};
  hid_t file_space = H5Dget_space(reader->ids_dataset);
  hid_t memory_space = H5Screate_simple(1, count, NULL);
  SsStatus status = SS_OK;
  if (file_space < 0 || memory_space < 0 ||
      H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL) < 0 ||
      H5Dread(reader->ids_dataset, H5T_NATIVE_UINT64, memory_space, file_space,
              reader->transfer, reader->ids) < 0) {
    status = SS_ERROR_HDF5;
  }
  status = ss_result_hdf5_close(&memory_space, H5Sclose, status);
  status = ss_result_hdf5_close(&file_space, H5Sclose, status);
  return status;
}

/* Encode `states` states of entity `entity` from a [states, entities, ...] block. */
static SsStatus ss_result_digest_emit(const SsResultDigestReader *reader,
                                      uint64_t states, uint64_t entities,
                                      uint64_t entity) {
  const SsResultQuantityShape *shape = reader->shape;
  const size_t per_entity = (size_t)shape->samples * (size_t)shape->components;
  SsResultDigestEncoder *encoder = reader->encoder;
  uint64_t state;
  for (state = 0U; state < states; ++state) {
    const size_t base =
        ((size_t)state * (size_t)entities + (size_t)entity) * per_entity;
    size_t index;
    for (index = 0U; index < per_entity; ++index) {
      SsStatus status = ss_result_encoder_reserve(encoder, sizeof(double));
      if (status != SS_OK) {
        return status;
      }
      if (shape->kind == SS_RESULT_DATASET_QUADRATURE_CATEGORY) {
        ss_result_quantity_store_u32_le(
            encoder->buffer + encoder->size,
            ((const uint32_t *)reader->scratch)[base + index]);
        encoder->size += sizeof(uint32_t);
      } else {
        const double value = ((const double *)reader->scratch)[base + index];
        if (!isfinite(value)) {
          return SS_ERROR_INTEGRITY;
        }
        ss_result_store_double_le(encoder->buffer + encoder->size, value);
        encoder->size += sizeof(double);
      }
    }
  }
  return SS_OK;
}

static SsStatus ss_result_digest_chunk(const SsResultDigestReader *reader,
                                       uint64_t chunk_start, uint64_t chunk_rows,
                                       uint64_t entity_block, uint64_t state_block,
                                       char digest[SS_SHA256_HEX_CAPACITY]) {
  const SsResultQuantityShape *shape = reader->shape;
  SsResultDigestEncoder *encoder = reader->encoder;
  uint64_t block_start;
  SsStatus status = SS_OK;
  ss_sha256_init(&encoder->context);
  encoder->size = 0U;
  for (block_start = 0U; status == SS_OK && block_start < chunk_rows;
       block_start += entity_block) {
    const uint64_t entities = chunk_rows - block_start < entity_block
                                  ? chunk_rows - block_start
                                  : entity_block;
    uint64_t entity;
    status = ss_result_digest_read_ids(reader, chunk_start + block_start, entities);
    if (status == SS_OK && state_block == shape->states) {
      status = ss_result_digest_read_values(reader, 0U, shape->states,
                                            chunk_start + block_start, entities);
    }
    for (entity = 0U; status == SS_OK && entity < entities; ++entity) {
      status = ss_result_encoder_reserve(encoder, sizeof(uint64_t));
      if (status != SS_OK) {
        break;
      }
      ss_result_store_u64_le(encoder->buffer + encoder->size, reader->ids[entity]);
      encoder->size += sizeof(uint64_t);
      if (state_block == shape->states) {
        status = ss_result_digest_emit(reader, shape->states, entities, entity);
      } else {
        uint64_t state_start;
        for (state_start = 0U; status == SS_OK && state_start < shape->states;
             state_start += state_block) {
          const uint64_t states = shape->states - state_start < state_block
                                      ? shape->states - state_start
                                      : state_block;
          status = ss_result_digest_read_values(reader, state_start, states,
                                                chunk_start + block_start + entity, 1U);
          if (status == SS_OK) {
            status = ss_result_digest_emit(reader, states, 1U, 0U);
          }
        }
      }
    }
  }
  if (status == SS_OK) {
    status = ss_sha256_update(&encoder->context, encoder->buffer, encoder->size);
  }
  if (status == SS_OK) {
    ss_sha256_final(&encoder->context, digest);
  }
  return status;
}

/* Hash the locally owned chunks (chunk % size == rank) into `local_digests`. */
static SsStatus ss_result_digest_local_chunks(hid_t ids_dataset, hid_t values_dataset,
                                              const SsResultQuantityShape *shape,
                                              uint64_t chunk_count, int rank, int size,
                                              char *local_digests) {
  SsResultDigestReader reader;
  SsResultDigestEncoder *encoder = NULL;
  const size_t value_size = shape->kind == SS_RESULT_DATASET_QUADRATURE_CATEGORY
                                ? sizeof(uint32_t)
                                : sizeof(double);
  const uint64_t per_state_entity =
      (uint64_t)shape->samples * (uint64_t)shape->components * (uint64_t)value_size;
  uint64_t entity_block;
  uint64_t state_block;
  uint64_t chunk;
  size_t local_index = 0U;
  size_t scratch_bytes;
  SsStatus status = SS_OK;
  if (per_state_entity == 0U ||
      per_state_entity > SS_RESULT_DIGEST_VALUE_SCRATCH_BYTES) {
    return SS_ERROR_LIMIT;
  }
  if (shape->states <= SS_RESULT_DIGEST_VALUE_SCRATCH_BYTES / per_state_entity) {
    state_block = shape->states;
    entity_block =
        SS_RESULT_DIGEST_VALUE_SCRATCH_BYTES / (shape->states * per_state_entity);
    if (entity_block > SS_RESULT_QUANTITY_DIGEST_CHUNK_ROWS) {
      entity_block = SS_RESULT_QUANTITY_DIGEST_CHUNK_ROWS;
    }
    if (entity_block == 0U) {
      entity_block = 1U;
    }
  } else {
    entity_block = 1U;
    state_block = SS_RESULT_DIGEST_VALUE_SCRATCH_BYTES / per_state_entity;
  }
  scratch_bytes = (size_t)(state_block * entity_block * per_state_entity);
  memset(&reader, 0, sizeof(reader));
  reader.ids_dataset = ids_dataset;
  reader.values_dataset = values_dataset;
  reader.shape = shape;
  reader.transfer = H5Pcreate(H5P_DATASET_XFER);
  reader.scratch = malloc(scratch_bytes == 0U ? 1U : scratch_bytes);
  reader.ids = (SsEntityId *)malloc((size_t)entity_block * sizeof(*reader.ids));
  encoder = (SsResultDigestEncoder *)malloc(sizeof(*encoder));
  reader.encoder = encoder;
  if (reader.transfer < 0 || reader.scratch == NULL || reader.ids == NULL ||
      encoder == NULL) {
    status = reader.transfer < 0 ? SS_ERROR_HDF5 : SS_ERROR_OUT_OF_MEMORY;
  } else if (H5Pset_dxpl_mpio(reader.transfer, H5FD_MPIO_INDEPENDENT) < 0) {
    status = SS_ERROR_HDF5;
  }
  for (chunk = (uint64_t)(uint32_t)rank; status == SS_OK && chunk < chunk_count;
       chunk += (uint64_t)(uint32_t)size) {
    const uint64_t start = chunk * SS_RESULT_QUANTITY_DIGEST_CHUNK_ROWS;
    const uint64_t rows = shape->entities - start < SS_RESULT_QUANTITY_DIGEST_CHUNK_ROWS
                              ? shape->entities - start
                              : SS_RESULT_QUANTITY_DIGEST_CHUNK_ROWS;
    char digest[SS_SHA256_HEX_CAPACITY];
    status =
        ss_result_digest_chunk(&reader, start, rows, entity_block, state_block, digest);
    if (status == SS_OK) {
      memcpy(local_digests + local_index * SS_RESULT_DIGEST_HEX, digest,
             SS_RESULT_DIGEST_HEX);
      ++local_index;
    }
  }
  if (reader.transfer >= 0) {
    status = ss_result_hdf5_close(&reader.transfer, H5Pclose, status);
  }
  free(encoder);
  free(reader.ids);
  free(reader.scratch);
  return status;
}

/*
 * Rank zero hashes `semantic || NUL || (chunk_le64 || chunk_hex)*` in chunk
 * order.  Chunks are owned round-robin, so gather round r holds chunks
 * r*size .. r*size+size-1 in rank order; a batch of rounds is gathered at once
 * with a fixed staging cap.
 */
static SsStatus ss_result_digest_combine(MPI_Comm communicator, int rank, int size,
                                         const char *semantic, uint64_t chunk_count,
                                         const char *local_digests,
                                         char output[SS_SHA256_HEX_CAPACITY + 8U]) {
  const uint64_t rounds = (chunk_count + (uint64_t)size - 1U) / (uint64_t)size;
  uint64_t batch_rounds = SS_RESULT_DIGEST_GATHER_RECORDS / (uint64_t)(uint32_t)size;
  char *staging = NULL;
  int *counts = NULL;
  int *offsets = NULL;
  SsSha256 context;
  uint64_t round_begin;
  SsStatus status = SS_OK;
  if (batch_rounds == 0U) {
    batch_rounds = 1U;
  }
  if (batch_rounds * SS_RESULT_DIGEST_HEX > (uint64_t)INT_MAX ||
      batch_rounds * (uint64_t)(uint32_t)size * SS_RESULT_DIGEST_HEX >
          (uint64_t)INT_MAX) {
    return SS_ERROR_OVERFLOW;
  }
  if (rank == 0) {
    staging = (char *)malloc((size_t)(batch_rounds * (uint64_t)(uint32_t)size) *
                             SS_RESULT_DIGEST_HEX);
    counts = (int *)calloc((size_t)size, sizeof(*counts));
    offsets = (int *)calloc((size_t)size, sizeof(*offsets));
    if (staging == NULL || counts == NULL || offsets == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
    ss_sha256_init(&context);
    if (status == SS_OK) {
      status = ss_sha256_update(&context, semantic, strlen(semantic) + 1U);
    }
  }
  status = ss_status_allreduce_max(communicator, status);
  for (round_begin = 0U; status == SS_OK && round_begin < rounds;
       round_begin += batch_rounds) {
    const uint64_t round_end =
        rounds - round_begin < batch_rounds ? rounds : round_begin + batch_rounds;
    uint64_t local_first = round_begin;
    uint64_t local_count = 0U;
    uint64_t round;
    int process;
    for (round = round_begin; round < round_end; ++round) {
      if (round * (uint64_t)(uint32_t)size + (uint64_t)(uint32_t)rank < chunk_count) {
        ++local_count;
      }
    }
    if (rank == 0) {
      int offset = 0;
      for (process = 0; process < size; ++process) {
        uint64_t owned = 0U;
        for (round = round_begin; round < round_end; ++round) {
          if (round * (uint64_t)(uint32_t)size + (uint64_t)(uint32_t)process <
              chunk_count) {
            ++owned;
          }
        }
        counts[process] = (int)(owned * SS_RESULT_DIGEST_HEX);
        offsets[process] = offset;
        offset += (int)(batch_rounds * SS_RESULT_DIGEST_HEX);
      }
    }
    if (MPI_Gatherv(local_digests + local_first * SS_RESULT_DIGEST_HEX,
                    (int)(local_count * SS_RESULT_DIGEST_HEX), MPI_CHAR, staging,
                    counts, offsets, MPI_CHAR, 0, communicator) != MPI_SUCCESS) {
      status = SS_ERROR_INTERNAL;
    }
    for (round = round_begin; status == SS_OK && rank == 0 && round < round_end;
         ++round) {
      for (process = 0; status == SS_OK && process < size; ++process) {
        const uint64_t chunk =
            round * (uint64_t)(uint32_t)size + (uint64_t)(uint32_t)process;
        unsigned char index_bytes[8];
        if (chunk >= chunk_count) {
          break;
        }
        ss_result_store_u64_le(index_bytes, chunk);
        status = ss_sha256_update(&context, index_bytes, sizeof(index_bytes));
        if (status == SS_OK) {
          status =
              ss_sha256_update(&context,
                               staging + (size_t)offsets[process] +
                                   (size_t)(round - round_begin) * SS_RESULT_DIGEST_HEX,
                               SS_RESULT_DIGEST_HEX);
        }
      }
    }
    status = ss_status_allreduce_max(communicator, status);
  }
  if (status == SS_OK && rank == 0) {
    memcpy(output, "sha256:", 7U);
    ss_sha256_final(&context, output + 7U);
  }
  free(offsets);
  free(counts);
  free(staging);
  return ss_result_broadcast_digest(communicator, status, output);
}

SsStatus ss_result_quantity_digest(hid_t ids_dataset, hid_t states_dataset,
                                   hid_t values_dataset, MPI_Comm communicator,
                                   char output[SS_SHA256_HEX_CAPACITY + 8U],
                                   SsResultQuantityShape *shape_out) {
  SsResultQuantityShape shape;
  char *semantic = NULL;
  char *local_digests = NULL;
  uint64_t chunk_count = 0U;
  uint64_t local_count = 0U;
  int rank = -1;
  int size = 0;
  SsStatus status;
  if (output == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  status =
      ss_result_quantity_shape(ids_dataset, states_dataset, values_dataset, &shape);
  if (status == SS_OK && (shape.states == 0U || shape.entities == 0U ||
                          shape.samples == 0U || shape.components == 0U)) {
    status = SS_ERROR_SCHEMA;
  }
  if (status == SS_OK &&
      (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS ||
       MPI_Comm_size(communicator, &size) != MPI_SUCCESS || size <= 0)) {
    status = SS_ERROR_INTERNAL;
  }
  if (status == SS_OK) {
    chunk_count = (shape.entities + SS_RESULT_QUANTITY_DIGEST_CHUNK_ROWS - 1U) /
                  SS_RESULT_QUANTITY_DIGEST_CHUNK_ROWS;
    local_count =
        chunk_count > (uint64_t)(uint32_t)rank
            ? (chunk_count - (uint64_t)(uint32_t)rank + (uint64_t)(uint32_t)size - 1U) /
                  (uint64_t)(uint32_t)size
            : 0U;
    local_digests = (char *)malloc((size_t)(local_count == 0U ? 1U : local_count) *
                                   SS_RESULT_DIGEST_HEX);
    if (local_digests == NULL) {
      status = SS_ERROR_OUT_OF_MEMORY;
    }
  }
  /* Every rank reads the attributes: collective metadata reads may be enabled. */
  if (status == SS_OK) {
    semantic = (char *)malloc(SS_RESULT_SEMANTIC_CAPACITY);
    status = semantic == NULL
                 ? SS_ERROR_OUT_OF_MEMORY
                 : ss_result_quantity_semantic(values_dataset, states_dataset, &shape,
                                               semantic, SS_RESULT_SEMANTIC_CAPACITY);
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    status = ss_result_digest_local_chunks(ids_dataset, values_dataset, &shape,
                                           chunk_count, rank, size, local_digests);
    status = ss_status_allreduce_max(communicator, status);
  }
  if (status == SS_OK) {
    status = ss_result_digest_combine(communicator, rank, size, semantic, chunk_count,
                                      local_digests, output);
  }
  if (status == SS_OK && shape_out != NULL) {
    *shape_out = shape;
  }
  free(local_digests);
  free(semantic);
  return ss_status_allreduce_max(communicator, status);
}

SsStatus
ss_result_verify_quantity_digest(hid_t file, MPI_Comm communicator,
                                 const char *group_path,
                                 char logical_digest[SS_SHA256_HEX_CAPACITY + 8U]) {
  char ids_path[SS_RESULT_PATH_CAPACITY];
  char states_path[SS_RESULT_PATH_CAPACITY];
  char values_path[SS_RESULT_PATH_CAPACITY];
  char expected[SS_SHA256_HEX_CAPACITY + 1U];
  hid_t ids_dataset = -1;
  hid_t states_dataset = -1;
  hid_t values_dataset = -1;
  SsStatus status = SS_OK;
  if (file < 0 || group_path == NULL || logical_digest == NULL ||
      snprintf(ids_path, sizeof(ids_path), "%s/stable_ids", group_path) < 0 ||
      snprintf(states_path, sizeof(states_path), "%s/state_ids", group_path) < 0 ||
      snprintf(values_path, sizeof(values_path), "%s/values", group_path) < 0) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  ids_dataset = H5Dopen2(file, ids_path, H5P_DEFAULT);
  states_dataset = H5Dopen2(file, states_path, H5P_DEFAULT);
  values_dataset = H5Dopen2(file, values_path, H5P_DEFAULT);
  if (ids_dataset < 0 || states_dataset < 0 || values_dataset < 0) {
    status = SS_ERROR_HDF5;
  }
  if (status == SS_OK) {
    status = ss_result_hdf5_read_string_attribute(values_dataset, "digest", expected,
                                                  sizeof(expected));
  }
  status = ss_status_allreduce_max(communicator, status);
  if (status == SS_OK) {
    status = ss_result_quantity_digest(ids_dataset, states_dataset, values_dataset,
                                       communicator, logical_digest, NULL);
  }
  if (status == SS_OK && (strlen(expected) != SS_RESULT_DIGEST_HEX ||
                          strcmp(expected, logical_digest + 7U) != 0)) {
    status = SS_ERROR_INTEGRITY;
  }
  status = ss_result_hdf5_close(&values_dataset, H5Dclose, status);
  status = ss_result_hdf5_close(&states_dataset, H5Dclose, status);
  status = ss_result_hdf5_close(&ids_dataset, H5Dclose, status);
  return ss_status_allreduce_max(communicator, status);
}
