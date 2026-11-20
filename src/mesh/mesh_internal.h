/* Private DMPlex mesh and field state with module interfaces.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_MESH_MESH_INTERNAL_H
#define SS_MESH_MESH_INTERNAL_H

#include "slope_internal/fe/fe.h"
#include "slope_internal/mesh/mesh.h"

#include <petscdm.h>
#include <petsclog.h>

#define SS_ID_CHUNK_COUNT 8U
#define SS_COORDINATE_SLOT_COUNT 3U

typedef struct {
  SsEntityId stable_id;
  SsMeshEntityKind kind;
  PetscInt point;
} SsPointEntry;

typedef struct {
  SsEntityId stable_id;
  PetscInt point;
} SsPointIdEntry;

typedef struct {
  uint32_t dimension;
  SsEntityId vertex_ids[4];
  PetscInt point;
} SsEntityVertexEntry;

typedef struct {
  uint32_t dimension;
  SsEntityId vertex_ids[3];
  SsEntityId stable_id;
  size_t record_index;
} SsIntermediateIndexEntry;

typedef struct {
  SsEntityId vertex_ids[3];
  SsEntityId stable_id;
} SsBoundaryIndexEntry;

typedef struct {
  DMLabel chunks[SS_ID_CHUNK_COUNT];
} SsChunkedLabels;

struct SsMesh {
  MPI_Comm communicator;
  int rank;
  int size;
  DM dm;
  uint32_t dimension;
  uint32_t coordinate_order;
  const SsFeDescriptor *coordinate_descriptor;
  SsFeGeometryPolicy geometry_policy;
  SsPointEntry *points;
  size_t point_count;
  SsEntityVertexEntry *entities;
  size_t entity_count;
  PetscInt cache_chart_start;
  PetscInt cache_chart_end;
  SsEntityId *cached_stable_ids;
  SsMeshEntityKind *cached_kinds;
  unsigned char *cached_point_valid;
  unsigned char *cached_point_ghost;
  SsEntityId *cached_cell_vertex_ids;
  unsigned char *cached_cell_vertex_counts;
  SsEntityId *cached_coordinate_node_ids;
  unsigned char *cached_coordinate_node_counts;
  uint32_t *facet_plan_support_values;
  SsFacetPack *facet_plans;
  size_t facet_plan_count;
  size_t active_field_spaces;
  SsMeshSummary summary;
};

struct SsFieldSpace {
  SsMesh *mesh;
  PetscSection local_section;
  PetscSection global_section;
  SsStrongConstraint *constraints;
  size_t constraint_count;
  PetscSF constraint_sf;
  const SsFeDescriptor *descriptor;
  SsCellPlan *owned_cell_plans;
  size_t owned_cell_plan_count;
  SsCellPlan *incident_cell_plans;
  size_t incident_cell_plan_count;
  void *incident_cell_plan_storage;
  size_t incident_cell_plan_storage_bytes;
  VecScatter extraction_scatter;
  Vec extraction_values;
  SsFieldOwnedNode *owned_nodes;
  size_t owned_node_count;
  int registered_with_mesh;
  SsFieldSummary summary;
};

/** Offset of a chart point in the metadata cache. */
static inline SsStatus ss_mesh_cached_point_offset(const SsMesh *mesh, PetscInt point,
                                                   size_t *offset) {
  if (mesh == NULL || offset == NULL || point < mesh->cache_chart_start ||
      point >= mesh->cache_chart_end || mesh->cached_point_valid == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *offset = (size_t)(point - mesh->cache_chart_start);
  if (mesh->cached_point_valid[*offset] == 0U) {
    return SS_ERROR_NOT_FOUND;
  }
  return SS_OK;
}

/** Cached stable identity of a chart point. */
static inline SsStatus ss_mesh_cached_stable_id(const SsMesh *mesh, PetscInt point,
                                                SsEntityId *stable_id) {
  size_t offset = 0U;
  if (stable_id == NULL || ss_mesh_cached_point_offset(mesh, point, &offset) != SS_OK ||
      mesh->cached_stable_ids == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *stable_id = mesh->cached_stable_ids[offset];
  return SS_OK;
}

/** Cached stable vertex identities of a cell. */
static inline SsStatus ss_mesh_cached_cell_vertices(const SsMesh *mesh, PetscInt point,
                                                    uint32_t dimension,
                                                    SsEntityId vertex_ids[4]) {
  size_t offset = 0U;
  size_t vertex;
  if (vertex_ids == NULL || dimension < 2U || dimension > SS_FE_MAX_DIMENSION ||
      ss_mesh_cached_point_offset(mesh, point, &offset) != SS_OK ||
      mesh->cached_cell_vertex_ids == NULL || mesh->cached_cell_vertex_counts == NULL ||
      mesh->cached_cell_vertex_counts[offset] != dimension + 1U) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (vertex = 0U; vertex <= (size_t)dimension; ++vertex) {
    vertex_ids[vertex] = mesh->cached_cell_vertex_ids[offset * 4U + vertex];
  }
  return SS_OK;
}

/** Cached stable identity of a cell's geometry node at a slot. */
static inline SsStatus ss_mesh_cached_coordinate_node(const SsMesh *mesh,
                                                      PetscInt point, uint32_t slot,
                                                      SsEntityId *stable_id) {
  size_t offset = 0U;
  if (stable_id == NULL || ss_mesh_cached_point_offset(mesh, point, &offset) != SS_OK ||
      mesh->cached_coordinate_node_ids == NULL ||
      mesh->cached_coordinate_node_counts == NULL ||
      slot >= mesh->cached_coordinate_node_counts[offset]) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *stable_id =
      mesh->cached_coordinate_node_ids[offset * SS_COORDINATE_SLOT_COUNT + slot];
  return SS_OK;
}

typedef struct {
  uint64_t route;
  uint64_t stable_id;
  uint64_t dimension;
  uint64_t vertex_ids[3];
} SsIntermediateAuditRecord;

typedef struct {
  uint64_t count;
  uint64_t sum_first;
  uint64_t sum_second;
  uint64_t xor_first;
  uint64_t xor_second;
} SsMeshDigestAccumulator;

/* PETSc events */
extern const char *const ss_mesh_id_chunk_labels[SS_ID_CHUNK_COUNT];

/* Semantic labels and chunked stable IDs on DMPlex points */
/** DMLabel name of a semantic label kind. */
const char *ss_mesh_semantic_label_name(SsMeshLabelKind kind);
/** DMLabel name of one label kind and value. */
SsStatus ss_mesh_semantic_membership_name(SsMeshLabelKind kind, uint32_t value,
                                          char name[64]);
/** Parse a membership label name into its kind and value. */
int ss_mesh_semantic_membership_parse(const char *name, SsMeshLabelKind *kind,
                                      uint32_t *value);
/** Create a DMLabel if it does not exist; local. */
SsStatus ss_mesh_create_label(DM dm, const char *name);
/** Whether a point carries any value of a label kind; local. */
SsStatus ss_mesh_point_has_any_membership(DM dm, SsMeshLabelKind kind, PetscInt point,
                                          int *contains);
/** Read a 64-bit stable identity stored in chunked int labels. */
SsStatus ss_mesh_get_chunked_id(DM dm, PetscInt point,
                                const char *const names[SS_ID_CHUNK_COUNT],
                                SsEntityId *stable_id);
/** Look up the chunk labels of a stable-identity label set once. */
SsStatus ss_mesh_chunked_labels_resolve(DM dm,
                                        const char *const names[SS_ID_CHUNK_COUNT],
                                        SsChunkedLabels *labels);
/** Read a stable identity through resolved chunk labels. */
SsStatus ss_mesh_chunked_labels_get(const SsChunkedLabels *labels, PetscInt point,
                                    SsEntityId *stable_id);
/** Store the stable identity and kind of a point in labels. */
SsStatus ss_mesh_set_point_id(DM dm, PetscInt point, SsEntityId stable_id,
                              SsMeshEntityKind kind);
/** Chunk label names of a per-slot identity label. */
void ss_mesh_slot_label_names(const char *prefix, uint32_t slot,
                              char storage[SS_ID_CHUNK_COUNT][48],
                              const char *names[SS_ID_CHUNK_COUNT]);
/** Store a per-slot stable identity of a point. */
SsStatus ss_mesh_set_slot_id(DM dm, PetscInt point, const char *prefix, uint32_t slot,
                             SsEntityId stable_id);
/** Read a per-slot stable identity of a point. */
SsStatus ss_mesh_get_slot_id(DM dm, PetscInt point, const char *prefix, uint32_t slot,
                             SsEntityId *stable_id);

/* Point, vertex tuple and boundary indexes for mesh entities */
/** Compare entities by stable identity for qsort(). */
int ss_mesh_entity_id_compare(const void *left_raw, const void *right_raw);
/** Compare point-index entries by stable identity for qsort(). */
int ss_point_entry_compare(const void *left_raw, const void *right_raw);
/** Sorted stable vertex identities of a point's closure. */
SsStatus ss_mesh_point_vertex_ids(DM dm, const SsMesh *mesh,
                                  const SsChunkedLabels *stable_labels, PetscInt point,
                                  SsEntityId vertex_ids[4], size_t *vertex_count);
/** Build a sorted stable-identity-to-point index of the chart; local. */
SsStatus ss_mesh_build_point_id_index(DM dm, SsPointIdEntry **index, size_t *count);
/** Find a point in a stable-identity index. */
SsStatus ss_mesh_point_id_index_find(const SsPointIdEntry *index, size_t count,
                                     SsEntityId stable_id, PetscInt *point);
/** Build a vertex-tuple-to-point index of edges and faces; local. */
SsStatus ss_mesh_build_entity_vertex_index(DM dm, const SsMesh *mesh,
                                           SsEntityVertexEntry **index, size_t *count);
/** Find an entity in a vertex-tuple index. */
SsStatus ss_mesh_entity_vertex_index_find(const SsEntityVertexEntry *index,
                                          size_t count, uint32_t dimension,
                                          const SsEntityId *vertex_ids,
                                          size_t vertex_count, PetscInt *point);
/** Whether two sorted vertex tuples are equal. */
int ss_mesh_same_vertex_set(const SsEntityId *left, const SsEntityId *right,
                            size_t count);
/** Index the input edges and faces by vertex tuple; local. */
SsStatus ss_mesh_build_intermediate_index(const SsMeshInput *input,
                                          SsIntermediateIndexEntry **index,
                                          size_t *count);
/** Stable identity of an input edge or face by vertex tuple. */
SsStatus ss_mesh_intermediate_index_find(const SsIntermediateIndexEntry *index,
                                         size_t count, uint32_t entity_dimension,
                                         const SsEntityId *vertex_ids,
                                         size_t vertex_count, SsEntityId *stable_id,
                                         size_t *record_index);
/** Stable identity of a vertex from its global input ordinal. */
SsStatus ss_mesh_adjacent_vertex_id(PetscInt adjacent_count,
                                    const PetscInt *vertices_adjacent,
                                    const SsEntityId *adjacent_ids,
                                    uint64_t global_ordinal, SsEntityId *stable_id);
/** Index the input boundary facets by vertex tuple; local. */
SsStatus ss_mesh_build_boundary_index(const SsMeshInput *input, PetscInt adjacent_count,
                                      const PetscInt *vertices_adjacent,
                                      const SsEntityId *adjacent_ids,
                                      SsBoundaryIndexEntry **index, size_t *count);
/** Stable identity of an input boundary facet by vertex tuple. */
SsStatus ss_mesh_boundary_index_find(const SsBoundaryIndexEntry *index, size_t count,
                                     const SsEntityId *vertex_ids, size_t vertex_count,
                                     SsEntityId *stable_id);

/* Attach stable IDs and semantic supports to topology */
/** Attach stable vertex identities after distribution; collective. */
SsStatus ss_mesh_attach_vertex_ids(DM dm, PetscSF vertex_sf, const SsMeshInput *input,
                                   SsEntityId **adjacent_ids, PetscInt *adjacent_count);
/** Attach stable cell identities; collective. */
SsStatus ss_mesh_attach_cells(DM dm, const SsMeshInput *input, PetscInt adjacent_count,
                              const PetscInt *vertices_adjacent,
                              const SsEntityId *adjacent_ids);
/** Attach stable edge and face identities; collective. */
SsStatus ss_mesh_attach_intermediate_ids(DM dm, const SsMeshInput *input,
                                         PetscInt adjacent_count,
                                         const PetscInt *vertices_adjacent,
                                         const SsEntityId *adjacent_ids);
/** Attach material, support and interface labels; collective. */
SsStatus ss_mesh_attach_semantic_labels(DM dm, const SsMeshInput *input,
                                        const SsPointIdEntry *point_index,
                                        size_t point_count);

/* Collective input validation, ownership, and migration audits */
/** Check that every cell has a material label; collective. */
SsStatus ss_mesh_audit_semantic_coverage(DM dm, uint32_t dimension);
/** Validate the input slabs and select the geometry element; collective. */
SsStatus ss_mesh_validate_collective(MPI_Comm communicator, const SsMeshInput *input,
                                     const SsFeDescriptor **descriptor);
/** Check that every point has exactly one owner; collective. */
SsStatus ss_mesh_audit_point_ownership(SsMesh *mesh, uint64_t expected_global_points);
/** Fill the global counts of the mesh summary; collective. */
SsStatus ss_mesh_finish_summary(SsMesh *mesh);
/** Check that distribution preserved identities and labels; collective. */
SsStatus ss_mesh_audit_migration(SsMesh *mesh);

/* Construct and query oriented coordinate closures */
/** Node slots of an entity depth at a Lagrange order. */
uint32_t ss_mesh_slots_for_depth(uint32_t dimension, uint32_t order, uint32_t depth);
/** Stable vertex identities of a cell in reference order. */
SsStatus ss_mesh_cell_vertex_ids(DM dm, const SsMesh *mesh, PetscInt cell,
                                 uint32_t dimension, SsEntityId vertex_ids[4]);
/** Entity point and slot of a cell's canonical node. */
SsStatus ss_mesh_node_entity_point(const SsEntityVertexEntry *entity_index,
                                   size_t entity_count,
                                   const SsFeDescriptor *descriptor, PetscInt cell,
                                   const SsEntityId *cell_vertex_ids, size_t node,
                                   PetscInt *entity_point, uint32_t *entity_slot);
/** Create the high-order coordinate section and values; collective. */
SsStatus ss_mesh_build_coordinate_space(DM dm, const SsMeshInput *input,
                                        const SsFeDescriptor *descriptor,
                                        const SsPointIdEntry *point_index,
                                        size_t point_count,
                                        const SsEntityVertexEntry *entity_index,
                                        size_t entity_count);

/* Rank-independent logical mesh and topology digests */
/** Mix a 64-bit value into a running hash. */
uint64_t ss_mesh_hash_u64(uint64_t hash, uint64_t value);
/** Store a 64-bit value in little-endian order. */
void ss_mesh_store_u64(unsigned char output[8], uint64_t value);
/** Store a 32-bit value in little-endian order. */
void ss_mesh_store_u32(unsigned char output[4], uint32_t value);
/** Add one record hash to an order-independent digest. */
void ss_mesh_digest_add(SsMeshDigestAccumulator *accumulator, uint64_t first,
                        uint64_t second);
/** Start a digest record with a tag. */
void ss_mesh_digest_record_begin(uint64_t tag, uint64_t *first, uint64_t *second);
/** Add a 64-bit value to a digest record. */
void ss_mesh_digest_record_u64(uint64_t value, uint64_t *first, uint64_t *second);
/** Add a 32-bit value to a digest record. */
void ss_mesh_digest_record_u32(uint32_t value, uint64_t *first, uint64_t *second);
/** Add the bits of a double to a digest record. */
void ss_mesh_digest_record_f64(double value, uint64_t *first, uint64_t *second);
/** Combine the local accumulators into a global digest; collective. */
SsStatus ss_mesh_digest_finalize(MPI_Comm communicator,
                                 const SsMeshDigestAccumulator *local,
                                 char output[SS_SHA256_HEX_CAPACITY]);
/** Add one tagged stable identity to a digest. */
void ss_mesh_digest_id(SsMeshDigestAccumulator *accumulator, uint64_t tag,
                       SsEntityId id);
/** Partition-independent digests of the input mesh; collective. */
SsStatus ss_mesh_compute_digests(MPI_Comm communicator, const SsMeshInput *input,
                                 const SsFeDescriptor *descriptor,
                                 SsMeshSummary *summary);
/** Partition-independent digest of the distributed topology; collective. */
SsStatus ss_mesh_compute_topology_digest(DM dm, const SsMesh *mesh,
                                         MPI_Comm communicator,
                                         char output[SS_SHA256_HEX_CAPACITY]);

/* Cache and query stable point identities and ghost metadata */
/** Release the point metadata cache. */
void ss_mesh_clear_metadata_cache(SsMesh *mesh);
/** Mark the ghost points of a chart range; local. */
SsStatus ss_mesh_ghost_flags_fill(DM dm, PetscInt chart_start, PetscInt chart_end,
                                  unsigned char *ghost_flags);
/** Rebuild the stable-identity point map after renumbering; local. */
SsStatus ss_mesh_rebuild_point_map(SsMesh *mesh);
/** Rebuild the entity vertex map after renumbering; local. */
SsStatus ss_mesh_rebuild_entity_map(SsMesh *mesh);
/** Whether a point is a ghost on this rank. */
int ss_mesh_point_is_ghost(const SsMesh *mesh, DM dm, PetscInt point);
/** Point of a resident entity of any kind. */
SsStatus ss_mesh_point_from_stable_id_any(const SsMesh *mesh, SsEntityId stable_id,
                                          PetscInt *point);
/** Point of a resident cell. */
SsStatus ss_mesh_cell_point(const SsMesh *mesh, SsEntityId stable_cell_id,
                            PetscInt *point);

/* Audit element geometry through the quadrature registry */
/** Audit every curved cell and select the first failure; collective. */
SsStatus ss_mesh_collective_geometry_audit(SsMesh *mesh, SsMeshError *selected);

/* Construct and validate discretized fields and essential constraints */
/** Route, sort and validate the constraints of a field space; collective. */
SsStatus ss_field_constraints_prepare(SsFieldSpace *space, const SsFieldSpec *spec,
                                      int distributed, uint64_t *global_count,
                                      char digest[SS_SHA256_HEX_CAPACITY]);
/** Compare constraints by entity, slot and component for qsort(). */
int ss_field_constraint_compare(const void *left_raw, const void *right_raw);

/* Build and query packed owned and incident field-cell plans */
/** Build the owned cell plans of a field space; local. */
SsStatus ss_field_owned_cell_plans_build(SsFieldSpace *space);
/** Build the owned and ghost cell plans of a field space; local. */
SsStatus ss_field_incident_cell_plans_build(SsFieldSpace *space);
/** Owned cell plan of a stable cell, or NULL. */
const SsCellPlan *ss_field_owned_cell_plan_find(const SsFieldSpace *space,
                                                SsEntityId stable_cell_id);
/** Owned or ghost cell plan of a stable cell, or NULL. */
const SsCellPlan *ss_field_incident_cell_plan_find(const SsFieldSpace *space,
                                                   SsEntityId stable_cell_id);

/* Build mesh and field boundary-facet plans */
/** Build the owned boundary-facet plans of a mesh; local. */
SsStatus ss_mesh_owned_facet_plans_build(SsMesh *mesh);

#endif
