/** @file
 * @brief Distributed DMPlex mesh with stable entity identities, labels,
 * essential supports and curved coordinates, and the field spaces built on it.
 *
 * Every entity carries a stable identity that does not depend on the number
 * of ranks or the partition. Functions that create, destroy or exchange data
 * are collective on the mesh communicator; lookups and views are local. All
 * calls are made from the master thread; views may then be read concurrently.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_MESH_MESH_H
#define SS_MESH_MESH_H

#include "slope_internal/core/core.h"
#include "slope_internal/diagnostics/diagnostics.h"
#include "slope_internal/fe/layout.h"

#include <mpi.h>
#include <petscmat.h>
#include <petscsystypes.h>
#include <petscvec.h>

#include <stddef.h>
#include <stdint.h>

/** Opaque distributed mesh; see ss_mesh_create(). */
typedef struct SsMesh SsMesh;
/** Opaque field space on a mesh; see ss_mesh_field_space_create(). */
typedef struct SsFieldSpace SsFieldSpace;

/** Largest number of unknowns in a cell closure (P8 tetrahedron, 3D). */
#define SS_FIELD_MAX_CLOSURE (SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION)
/** Largest number of nodes on a facet (P8 triangle). */
#define SS_FIELD_MAX_FACET_NODES 45U
/** Largest number of unknowns in a facet closure. */
#define SS_FIELD_MAX_FACET_CLOSURE (SS_FIELD_MAX_FACET_NODES * SS_FE_MAX_DIMENSION)

/** Topological kind of a mesh entity. */
typedef enum {
  SS_MESH_ENTITY_VERTEX = 1, /**< Vertex. */
  SS_MESH_ENTITY_EDGE = 2,   /**< Edge. */
  SS_MESH_ENTITY_FACET = 3,  /**< Facet. */
  SS_MESH_ENTITY_CELL = 4    /**< Cell. */
} SsMeshEntityKind;

/** Kind of an entity label. */
typedef enum {
  SS_MESH_LABEL_MATERIAL = 1, /**< Material region of a cell. */
  SS_MESH_LABEL_SUPPORT = 2,  /**< Boundary support of an entity. */
  SS_MESH_LABEL_INTERFACE = 3 /**< Interface between regions. */
} SsMeshLabelKind;

/** One label value assigned to a stable entity. */
typedef struct {
  SsMeshLabelKind kind;        /**< Label kind. */
  SsEntityId stable_entity_id; /**< Labelled entity. */
  uint32_t value;              /**< Label value. */
} SsMeshLabelAssignment;

/** Stable identity of an edge or face by its sorted vertex identities. */
typedef struct {
  uint32_t dimension;          /**< 1 for an edge, 2 for a face. */
  SsEntityId stable_entity_id; /**< Stable identity of the entity. */
  SsEntityId vertex_ids[3];    /**< Increasing vertex identities. */
} SsMeshIntermediateEntity;

/** Ownership of points shared by several ranks after distribution. */
typedef enum {
  SS_MESH_BALANCE_PETSC = 0,            /**< PETSc's default ownership. */
  SS_MESH_BALANCE_STABLE_P2 = 1,        /**< Stable-hash balance of P2 points. */
  SS_MESH_BALANCE_STABLE_HIGH_ORDER = 2 /**< Stable-hash balance of P4 points. */
} SsMeshSharedPointBalance;

/** Distributed mesh input: each rank passes its own slab of the global arrays;
 * slabs may be empty. All pointers are borrowed for the call. */
typedef struct {
  uint32_t dimension;                        /**< Spatial dimension, 2 or 3. */
  uint64_t global_vertex_count;              /**< Vertices on all ranks. */
  uint64_t global_cell_count;                /**< Cells on all ranks. */
  uint64_t global_boundary_facet_count;      /**< Boundary facets on all ranks. */
  uint64_t global_intermediate_entity_count; /**< Edges and faces on all ranks. */
  uint64_t global_coordinate_node_count;     /**< Geometry nodes on all ranks. */
  size_t vertex_count;                       /**< Vertices in this slab. */
  size_t cell_count;                         /**< Cells in this slab. */
  size_t boundary_facet_count;               /**< Boundary facets in this slab. */
  const SsEntityId *vertex_ids;              /**< Stable vertex identities. */
  const double *vertex_coordinates;          /**< Vertex-major coordinates in m. */
  const SsEntityId *cell_ids;                /**< Stable cell identities. */
  /** Global vertex indices of each cell, `dimension + 1` per cell. */
  const uint64_t *cells;
  const SsEntityId *boundary_facet_ids; /**< Stable facet identities. */
  /** Global vertex indices of each boundary facet, `dimension` per facet. */
  const uint64_t *boundary_facets;
  size_t intermediate_entity_count; /**< Edges and faces in this slab. */
  const SsMeshIntermediateEntity *intermediate_entities; /**< Edges and faces. */
  size_t label_count;                    /**< Label assignments in this slab. */
  const SsMeshLabelAssignment *labels;   /**< Label assignments. */
  uint32_t coordinate_order;             /**< Geometry order, 1, 2 or 4. */
  size_t coordinate_node_count;          /**< Geometry nodes in this slab. */
  const SsEntityId *coordinate_node_ids; /**< Stable geometry-node identities. */
  const double *coordinate_coordinates;  /**< Node-major geometry coordinates. */
  /** Global geometry-node indices of each cell in canonical order. */
  const uint64_t *cell_coordinate_nodes;
  double absolute_determinant_tolerance;         /**< Smallest admissible `det J`. */
  double relative_determinant_tolerance;         /**< Relative `det J` tolerance. */
  SsMeshSharedPointBalance shared_point_balance; /**< Shared-point ownership. */
} SsMeshInput;

/** First mesh or geometry error, selected deterministically across ranks. */
typedef struct {
  int valid;                   /**< Nonzero when an error is recorded. */
  SsStatus status;             /**< Status of the error. */
  SsEntityId stable_entity_id; /**< Entity that failed. */
  size_t quadrature_index;     /**< Audit point that failed. */
  double determinant;          /**< `det J` at the point. */
  double condition_estimate;   /**< Condition estimate at the point. */
  char quadrature_rule_id[SS_ID_TEXT_CAPACITY]; /**< Audit rule. */
  char code[SS_ERROR_CODE_CAPACITY];            /**< Error code. */
} SsMeshError;

/** Partition-independent counts, generations and digests of a mesh, plus the
 * owned and ghost counts of this rank. */
typedef struct {
  uint32_t dimension;                    /**< Spatial dimension. */
  uint32_t coordinate_order;             /**< Geometry order. */
  uint64_t global_vertices;              /**< Vertices on all ranks. */
  uint64_t global_edges;                 /**< Edges on all ranks. */
  uint64_t global_facets;                /**< Facets on all ranks. */
  uint64_t global_intermediate_entities; /**< Edges and faces on all ranks. */
  uint64_t global_cells;                 /**< Cells on all ranks. */
  uint64_t global_coordinate_nodes;      /**< Geometry nodes on all ranks. */
  uint64_t owned_vertices;               /**< Owned vertices. */
  uint64_t ghost_vertices;               /**< Ghost vertices. */
  uint64_t owned_edges;                  /**< Owned edges. */
  uint64_t ghost_edges;                  /**< Ghost edges. */
  uint64_t owned_facets;                 /**< Owned facets. */
  uint64_t ghost_facets;                 /**< Ghost facets. */
  uint64_t owned_intermediate_entities;  /**< Owned edges and faces. */
  uint64_t ghost_intermediate_entities;  /**< Ghost edges and faces. */
  uint64_t owned_cells;                  /**< Owned cells. */
  uint64_t ghost_cells;                  /**< Ghost cells. */
  uint64_t owned_coordinate_nodes;       /**< Owned geometry nodes. */
  uint64_t ghost_coordinate_nodes;       /**< Ghost geometry nodes. */
  uint64_t minimum_owned_cells;          /**< Fewest owned cells of a rank. */
  uint64_t maximum_owned_cells;          /**< Most owned cells of a rank. */
  uint64_t topology_generation;          /**< Generation of the topology. */
  uint64_t coordinate_generation;        /**< Generation of the coordinates. */
  uint64_t label_generation;             /**< Generation of the labels. */
  uint64_t partition_generation;         /**< Generation of the partition. */
  uint64_t cache_memory_bytes;           /**< Memory of the rank-local caches. */
  uint64_t owned_facet_plan_count;       /**< Owned boundary-facet plans. */
  uint64_t facet_plan_bytes;             /**< Memory of the facet plans. */
  uint64_t geometry_audit_rule_count;    /**< Rules used by the geometry audit. */
  uint64_t geometry_audit_point_count;   /**< Points checked by the audit. */
  double minimum_determinant;            /**< Smallest audited `det J`. */
  double maximum_determinant;            /**< Largest audited `det J`. */
  double minimum_condition;              /**< Smallest audited condition estimate. */
  double maximum_condition;              /**< Largest audited condition estimate. */
  double partition_imbalance;            /**< Largest over mean owned cells. */
  char coordinate_basis_id[SS_ID_TEXT_CAPACITY];    /**< Geometry element identity. */
  char geometry_quadrature_id[SS_ID_TEXT_CAPACITY]; /**< Audit rule identity. */
  char geometry_audit_rule_digest[SS_SHA256_HEX_CAPACITY]; /**< Digest of the audit
                                                              rules. */
  char
      stable_id_digest[SS_SHA256_HEX_CAPACITY]; /**< Digest of the stable identities. */
  char topology_digest[SS_SHA256_HEX_CAPACITY]; /**< Digest of the topology. */
  char label_digest[SS_SHA256_HEX_CAPACITY];    /**< Digest of the labels. */
  char coordinate_digest[SS_SHA256_HEX_CAPACITY]; /**< Digest of the coordinates. */
  char closure_digest[SS_SHA256_HEX_CAPACITY];    /**< Digest of the cell closures. */
} SsMeshSummary;

/** Prescribed value of one unknown. */
typedef struct {
  SsEntityId stable_entity_id; /**< Entity carrying the unknown. */
  uint32_t entity_slot;        /**< Node slot on the entity. */
  uint32_t component;          /**< Field component. */
  double value;                /**< Prescribed value. */
} SsStrongConstraint;

/** Field space request. */
typedef struct {
  uint32_t order;                        /**< Lagrange order. */
  uint32_t components;                   /**< Components per node. */
  size_t constraint_count;               /**< Number of constraints. */
  const SsStrongConstraint *constraints; /**< Constraints; borrowed. */
} SsFieldSpec;

/** Size and identity of a field space. */
typedef struct {
  uint32_t order;                            /**< Lagrange order. */
  uint32_t components;                       /**< Components per node. */
  uint32_t vertex_slots;                     /**< Nodes per vertex. */
  uint32_t edge_interior_slots;              /**< Interior nodes per edge. */
  uint32_t face_interior_slots;              /**< Interior nodes per face. */
  uint32_t cell_interior_slots;              /**< Interior nodes per cell. */
  uint64_t global_dof_count;                 /**< Unknowns before constraints. */
  uint64_t global_free_dof_count;            /**< Free unknowns. */
  uint64_t owned_free_dof_count;             /**< Free unknowns owned here. */
  uint64_t constrained_dof_count;            /**< Constrained unknowns. */
  uint64_t owned_cell_plan_count;            /**< Owned cell plans. */
  uint64_t incident_cell_plan_count;         /**< Owned and ghost cell plans. */
  uint64_t closure_plan_bytes;               /**< Memory of the owned plans. */
  uint64_t incident_closure_plan_bytes;      /**< Memory of all plans. */
  uint32_t overlap_depth;                    /**< Ghost layers. */
  uint64_t section_generation;               /**< Generation of the section. */
  uint64_t constraint_generation;            /**< Generation of the constraints. */
  char basis_id[SS_ID_TEXT_CAPACITY];        /**< Element identity. */
  char identity_digest[SS_ID_TEXT_CAPACITY]; /**< Field-space identity. */
} SsFieldSummary;

/** Copy of one cell's geometry and field closure, sized for the largest
 * element. */
typedef struct {
  uint32_t dimension;                              /**< Spatial dimension. */
  SsEntityId stable_cell_id;                       /**< Stable cell. */
  uint32_t material_value;                         /**< Material label value. */
  uint32_t coordinate_order;                       /**< Geometry order. */
  size_t coordinate_node_count;                    /**< Geometry nodes. */
  SsEntityId coordinate_node_ids[SS_FE_MAX_NODES]; /**< Stable geometry nodes. */
  double coordinate_coordinates[SS_FE_MAX_NODES *
                                SS_FE_MAX_DIMENSION]; /**< Geometry-node coordinates. */
  uint32_t order;                                     /**< Field order. */
  uint32_t components;                                /**< Components per node. */
  size_t field_node_count;                            /**< Field nodes. */
  size_t closure_count;                               /**< Unknowns in the closure. */
  SsEntityId field_entity_ids[SS_FIELD_MAX_CLOSURE];  /**< Entity of each unknown. */
  uint32_t field_entity_slots[SS_FIELD_MAX_CLOSURE];  /**< Node slot of each unknown. */
  uint32_t field_components[SS_FIELD_MAX_CLOSURE];    /**< Component of each unknown. */
  PetscInt local_indices[SS_FIELD_MAX_CLOSURE];       /**< Local section offset. */
  PetscInt global_indices[SS_FIELD_MAX_CLOSURE]; /**< Global free row, or negative. */
  uint8_t constrained[SS_FIELD_MAX_CLOSURE]; /**< Nonzero for a constrained unknown. */
  uint8_t owned[SS_FIELD_MAX_CLOSURE];       /**< Nonzero for an owned unknown. */
  uint8_t cell_owned;                        /**< Nonzero for an owned cell. */
  double prescribed_values[SS_FIELD_MAX_CLOSURE]; /**< Prescribed values. */
  double field_node_coordinates[SS_FE_MAX_NODES *
                                SS_FE_MAX_DIMENSION]; /**< Field-node coordinates. */
} SsCellPack;

/**
 * Read-only cell plan used by field spaces and assembly operators.
 *
 * ::SsCellPack remains the fixed-capacity transfer type for callers that request
 * copied cell data.  Keeping that ABI-shaped type out of persistent solver
 * storage is important: its arrays are sized for the largest supported 3-D
 * element (P8), which otherwise multiplies memory for every P1/P2/P4 cell and
 * every PMG level. ::SsCellPlan points into one field-space-owned, contiguous
 * allocation whose slices are sized to the active coordinate and field orders.
 */
typedef struct {
  uint32_t dimension;              /**< Spatial dimension. */
  SsEntityId stable_cell_id;       /**< Stable cell. */
  uint32_t material_value;         /**< Material label value. */
  uint32_t coordinate_order;       /**< Geometry order. */
  size_t coordinate_node_count;    /**< Geometry nodes. */
  SsEntityId *coordinate_node_ids; /**< Stable geometry nodes. */
  double *coordinate_coordinates;  /**< Geometry-node coordinates. */
  uint32_t order;                  /**< Field order. */
  uint32_t components;             /**< Components per node. */
  size_t field_node_count;         /**< Field nodes. */
  size_t closure_count;            /**< Unknowns in the closure. */
  SsEntityId *field_entity_ids;    /**< Entity of each unknown. */
  uint32_t *field_entity_slots;    /**< Node slot of each unknown. */
  uint32_t *field_components;      /**< Component of each unknown. */
  PetscInt *local_indices;         /**< Local section offset. */
  PetscInt *global_indices;        /**< Global free row, or negative. */
  uint8_t *constrained;            /**< Nonzero for a constrained unknown. */
  uint8_t *owned;                  /**< Nonzero for an owned unknown. */
  uint8_t cell_owned;              /**< Nonzero for an owned cell. */
  double *prescribed_values;       /**< Prescribed values. */
  double *field_node_coordinates;  /**< Field-node coordinates. */
} SsCellPlan;

/** Geometry of one boundary facet and its parent cell. */
typedef struct {
  uint32_t dimension;               /**< Spatial dimension of the parent. */
  SsEntityId stable_facet_id;       /**< Stable facet. */
  SsEntityId stable_parent_cell_id; /**< Stable parent cell. */
  uint32_t support_value;           /**< Support label value. */
  size_t vertex_count;              /**< Facet vertices. */
  SsEntityId vertex_ids[3];         /**< Stable facet vertices. */
  uint32_t coordinate_order;        /**< Geometry order. */
  size_t coordinate_node_count;     /**< Parent geometry nodes. */
  SsEntityId coordinate_node_ids[SS_FE_MAX_NODES]; /**< Stable parent geometry nodes. */
  double
      coordinate_coordinates[SS_FE_MAX_NODES *
                             SS_FE_MAX_DIMENSION]; /**< Parent geometry coordinates. */
  uint32_t facet_index;                        /**< Local facet index in the parent. */
  double parent_centroid[SS_FE_MAX_DIMENSION]; /**< Parent centroid. */
  double centroid[3];                          /**< Facet centroid. */
  double outward_normal[3];                    /**< Outward normal at the centroid. */
  /** Diagnostic normal component of (facet centroid - parent centroid); it may
   * be nonpositive on curved cells. */
  double outward_dot;
  /** +1: the normal is the topologically oriented outward normal. */
  int orientation;
  uint8_t facet_owned; /**< Nonzero for an owned facet. */
} SsFacetPack;

/** Facet geometry with its field closure. */
typedef struct {
  SsFacetPack geometry;    /**< Facet geometry. */
  uint32_t order;          /**< Field order. */
  uint32_t components;     /**< Components per node. */
  size_t field_node_count; /**< Field nodes. */
  size_t closure_count;    /**< Unknowns in the closure. */
  size_t parent_closure_slots[SS_FIELD_MAX_FACET_CLOSURE]; /**< Slot of each unknown in
                                                              the parent closure. */
  SsEntityId
      field_entity_ids[SS_FIELD_MAX_FACET_CLOSURE]; /**< Entity of each unknown. */
  uint32_t
      field_entity_slots[SS_FIELD_MAX_FACET_CLOSURE]; /**< Node slot of each unknown. */
  uint32_t
      field_components[SS_FIELD_MAX_FACET_CLOSURE];   /**< Component of each unknown. */
  PetscInt local_indices[SS_FIELD_MAX_FACET_CLOSURE]; /**< Local section offset. */
  PetscInt
      global_indices[SS_FIELD_MAX_FACET_CLOSURE];  /**< Global free row, or negative. */
  uint8_t constrained[SS_FIELD_MAX_FACET_CLOSURE]; /**< Nonzero for a constrained
                                                      unknown. */
  uint8_t owned[SS_FIELD_MAX_FACET_CLOSURE];       /**< Nonzero for an owned unknown. */
  double prescribed_values[SS_FIELD_MAX_FACET_CLOSURE]; /**< Prescribed values. */
  double
      field_node_coordinates[SS_FIELD_MAX_FACET_NODES *
                             SS_FE_MAX_DIMENSION]; /**< Facet field-node coordinates. */
} SsFieldFacetPack;

/** Geometry of a facet at one reference point. */
typedef struct {
  uint32_t dimension;                   /**< Spatial dimension. */
  double physical[SS_FE_MAX_DIMENSION]; /**< Physical point. */
  /** Facet tangents. */
  double tangents[SS_FE_MAX_DIMENSION * (SS_FE_MAX_DIMENSION - 1U)];
  double outward_normal[SS_FE_MAX_DIMENSION]; /**< Outward unit normal. */
  double measure;                             /**< Measure density. */
} SsFacetSample;

/** Distribute input slabs into a DMPlex mesh with stable identities; collective.
 *
 * Partitions with ParMETIS (or PETSc's partitioner), adds one layer of ghost
 * cells, attaches labels and the curved geometry, and audits every cell. No
 * rank assembles the global mesh. The input buffers may be freed afterwards.
 *
 * @param[in] communicator Communicator of the mesh.
 * @param[in] input This rank's slab of the global mesh.
 * @param[out] mesh New mesh, released with ss_mesh_destroy().
 * @param[out] error First mesh or geometry error, identical on all ranks.
 * @return ::SS_OK or the agreed failure status (::SS_ERROR_INTEGRITY for an
 *   invalid or inverted mesh, ::SS_ERROR_LIMIT, ::SS_ERROR_OVERFLOW, ...). */
SsStatus ss_mesh_create(MPI_Comm communicator, const SsMeshInput *input, SsMesh **mesh,
                        SsMeshError *error);
/** Destroy a mesh; collective. Field spaces on it must be destroyed first.
 * @param[in,out] mesh Mesh pointer, set to NULL; `*mesh` may be NULL.
 * @return ::SS_OK, ::SS_ERROR_INVALID_STATE while field spaces remain,
 *   ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_INTERNAL. */
SsStatus ss_mesh_destroy(SsMesh **mesh);
/** Counts, generations and digests of a mesh; local.
 * @param[in] mesh Mesh.
 * @param[out] summary Summary.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_mesh_summary(const SsMesh *mesh, SsMeshSummary *summary);

/** DMPlex point of a resident entity; local.
 * @param[in] mesh Mesh.
 * @param[in] kind Entity kind.
 * @param[in] stable_id Stable identity.
 * @param[out] point DMPlex point.
 * @return ::SS_OK, ::SS_ERROR_NOT_FOUND or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_mesh_point_from_stable_id(const SsMesh *mesh, SsMeshEntityKind kind,
                                      SsEntityId stable_id, PetscInt *point);
/** Stable identity of a DMPlex point; local.
 * @param[in] mesh Mesh.
 * @param[in] point DMPlex point.
 * @param[out] stable_id Stable identity.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_mesh_stable_id_from_point(const SsMesh *mesh, PetscInt point,
                                      SsEntityId *stable_id);
/** Permute the local DMPlex numbering with a seed (test hook for
 * numbering independence); collective.
 * @param[in,out] mesh Mesh without field spaces.
 * @param[in] seed Permutation seed.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_mesh_renumber(SsMesh *mesh, uint64_t seed);

/** Label value of a resident entity; local.
 * @param[in] mesh Mesh.
 * @param[in] kind Label kind.
 * @param[in] stable_id Stable identity.
 * @param[out] value Label value.
 * @return ::SS_OK, ::SS_ERROR_NOT_FOUND, ::SS_ERROR_INVALID_ARGUMENT or
 *   ::SS_ERROR_INTERNAL. */
SsStatus ss_mesh_label_value(const SsMesh *mesh, SsMeshLabelKind kind,
                             SsEntityId stable_id, uint32_t *value);
/** Whether a resident entity carries a label value; local.
 * @param[in] mesh Mesh.
 * @param[in] kind Label kind.
 * @param[in] stable_id Stable identity.
 * @param[in] value Label value.
 * @param[out] contains Nonzero when the entity carries @p value.
 * @return ::SS_OK, ::SS_ERROR_NOT_FOUND, ::SS_ERROR_INVALID_ARGUMENT or
 *   ::SS_ERROR_INTERNAL. */
SsStatus ss_mesh_label_contains(const SsMesh *mesh, SsMeshLabelKind kind,
                                SsEntityId stable_id, uint32_t value, int *contains);
/** Number of owned entities with a label value; local.
 * @param[in] mesh Mesh.
 * @param[in] kind Label kind.
 * @param[in] value Label value.
 * @return Owned entities with the value; 0 for invalid arguments. */
size_t ss_mesh_label_cardinality(const SsMesh *mesh, SsMeshLabelKind kind,
                                 uint32_t value);
/** Number of entities with a label value on all ranks; collective.
 * @param[in] mesh Mesh.
 * @param[in] kind Label kind.
 * @param[in] value Label value.
 * @param[out] count Global count.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_mesh_label_cardinality_global(const SsMesh *mesh, SsMeshLabelKind kind,
                                          uint32_t value, uint64_t *count);

/** Stable identity and dimension of an entity. */
typedef struct {
  SsEntityId stable_id; /**< Stable identity. */
  uint32_t dimension;   /**< Topological dimension. */
} SsMeshEntityRef;

/** Topological dimension of a resident (owned or ghost) entity; local.
 * @param[in] mesh Mesh.
 * @param[in] id Stable identity.
 * @param[out] dimension Topological dimension.
 * @return ::SS_OK, ::SS_ERROR_NOT_FOUND, ::SS_ERROR_INTEGRITY or
 *   ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_mesh_entity_dimension(const SsMesh *mesh, SsEntityId id,
                                  uint32_t *dimension);

/** Create a resident essential-rule closure from distributed stable
 * identities; collective.
 *
 * Vertex-only sets also select intermediate entities on the exterior whose
 * vertices all belong to the set.
 *
 * @param[in,out] mesh Mesh.
 * @param[in] rule Rule number of the closure.
 * @param[in] ids This rank's share of the stable identities.
 * @param[in] count Number of identities.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_mesh_essential_support_set(SsMesh *mesh, uint32_t rule,
                                       const SsEntityId *ids, size_t count);

/** This rank's owned and ghost entities of an essential-rule closure; local.
 * @param[in] mesh Mesh.
 * @param[in] rule Rule number.
 * @param[out] members Entities, released by the caller with free().
 * @param[out] count Number of entities.
 * @return ::SS_OK, ::SS_ERROR_OUT_OF_MEMORY or ::SS_ERROR_INTERNAL. */
SsStatus ss_mesh_essential_members(const SsMesh *mesh, uint32_t rule,
                                   SsMeshEntityRef **members, size_t *count);

/** Whether a resident entity belongs to an essential-rule closure; local.
 * @param[in] mesh Mesh.
 * @param[in] rule Rule number.
 * @param[in] id Stable identity.
 * @return Nonzero when the entity belongs to the closure. */
int ss_mesh_essential_contains(const SsMesh *mesh, uint32_t rule, SsEntityId id);

/** Create a field space with distributed constraints; collective.
 *
 * Constraints may be given on any rank and are routed to the owners.
 *
 * @param[in,out] mesh Mesh; it must outlive the space.
 * @param[in] spec Order, components and constraints.
 * @param[out] space New space, released with ss_field_space_destroy().
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_mesh_field_space_create(SsMesh *mesh, const SsFieldSpec *spec,
                                    SsFieldSpace **space);
/** Create a field space from constraints on resident points; collective.
 *
 * Every copy of a shared point must supply the same tuples.
 *
 * @param[in,out] mesh Mesh; it must outlive the space.
 * @param[in] spec Order, components and resident constraints.
 * @param[out] space New space, released with ss_field_space_destroy().
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_mesh_field_space_create_local(SsMesh *mesh, const SsFieldSpec *spec,
                                          SsFieldSpace **space);

/** Candidate value of a shared constraint from one incident cell. */
typedef struct {
  SsEntityId cell_id; /**< Incident owned cell that proposes the value. */
  double value;       /**< Proposed value. */
  int present;        /**< Nonzero when this rank has a candidate. */
} SsConstraintCandidate;

/** Replace the sorted resident constraint values with the exact bits from the
 * lowest stable incident owned cell; collective, for a local field space.
 * @param[in] space Field space.
 * @param[in] candidates One candidate per constraint; may be absent on a rank.
 * @param[in,out] constraints Sorted constraints whose values are resolved.
 * @param[in] count Number of constraints.
 * @return ::SS_OK, ::SS_ERROR_INTEGRITY, ::SS_ERROR_INVALID_ARGUMENT or
 *   ::SS_ERROR_OUT_OF_MEMORY, agreed on all ranks. */
SsStatus
ss_field_space_resolve_constraint_values(const SsFieldSpace *space,
                                         const SsConstraintCandidate *candidates,
                                         SsStrongConstraint *constraints, size_t count);

/** Size and identity of a field space; local.
 * @param[in] space Field space.
 * @param[out] summary Summary.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_field_space_summary(const SsFieldSpace *space, SsFieldSummary *summary);
/** Stable entity and local slot of every owned free unknown, in global row
 * order; collective.
 * @param[in] space Field space.
 * @param[out] stable_ids Entity of each unknown, or NULL to query the count.
 * @param[out] local_slots Local section slot of each unknown, or NULL.
 * @param[in] capacity Capacity of the arrays.
 * @param[out] count Owned free unknowns.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_field_space_owned_free_dof_map(const SsFieldSpace *space,
                                           SsEntityId *stable_ids, size_t *local_slots,
                                           size_t capacity, size_t *count);
/** Lift the prescribed values of an owned cell out of its element system;
 * pure and safe inside OpenMP regions.
 * @param[in] space Field space.
 * @param[in] stable_cell_id Owned cell.
 * @param[in] element_matrix Row-major element matrix.
 * @param[in] element_rhs Element right-hand side.
 * @param[in] trial_values Closure values before lifting.
 * @param[in] closure_count Unknowns in the closure.
 * @param[out] complete_values Closure values with prescribed entries.
 * @param[out] free_slots Closure slot of each free unknown.
 * @param[out] free_rhs Lifted right-hand side of each free unknown.
 * @param[out] free_count Number of free unknowns.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_field_space_apply_cell_lifting(
    const SsFieldSpace *space, SsEntityId stable_cell_id, const double *element_matrix,
    const double *element_rhs, const double *trial_values, size_t closure_count,
    double *complete_values, size_t *free_slots, double *free_rhs, size_t *free_count);
/** Copy the owned cell plans into packs; local.
 * @param[in] space Field space.
 * @param[out] packs Packs, or NULL to query the count.
 * @param[in] capacity Capacity of @p packs.
 * @param[out] count Owned cells.
 * @return ::SS_OK, ::SS_ERROR_LIMIT or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_field_space_pack_owned_cells(const SsFieldSpace *space, SsCellPack *packs,
                                         size_t capacity, size_t *count);
/** Copy the owned and ghost cell plans into packs; local.
 * @param[in] space Field space.
 * @param[out] packs Packs, or NULL to query the count.
 * @param[in] capacity Capacity of @p packs.
 * @param[out] count Incident cells.
 * @return ::SS_OK, ::SS_ERROR_LIMIT or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_field_space_pack_incident_cells(const SsFieldSpace *space,
                                            SsCellPack *packs, size_t capacity,
                                            size_t *count);
/** Borrow the owned cell plans, sorted by stable cell; local.
 * @param[in] space Field space.
 * @param[out] plans Plans valid until the space is destroyed.
 * @param[out] count Owned cells.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_field_space_owned_cell_plan_view(const SsFieldSpace *space,
                                             const SsCellPlan **plans, size_t *count);
/** Borrow the owned and ghost cell plans, sorted by stable cell; local.
 * @param[in] space Field space.
 * @param[out] plans Plans valid until the space is destroyed.
 * @param[out] count Incident cells.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_field_space_incident_cell_plan_view(const SsFieldSpace *space,
                                                const SsCellPlan **plans,
                                                size_t *count);
/** Copy per-cell values from cell owners to every rank holding the cell as a
 * ghost; collective (PetscSF broadcast).
 * @param[in] space Field space.
 * @param[in] values_per_cell Values per cell.
 * @param[in] owned_cell_ids Owned cells, sorted.
 * @param[in] owned_values Values of the owned cells, cell-major.
 * @param[in] owned_cell_count Owned cells.
 * @param[out] incident_cell_ids Owned and ghost cells, sorted.
 * @param[out] incident_values Values of the incident cells.
 * @param[in] incident_cell_count Incident cells.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_field_space_broadcast_owned_cell_values(
    const SsFieldSpace *space, size_t values_per_cell, const SsEntityId *owned_cell_ids,
    const double *owned_values, size_t owned_cell_count, SsEntityId *incident_cell_ids,
    double *incident_values, size_t incident_cell_count);
/** Copy the owned boundary facets of a support into packs; local.
 * @param[in] space Field space.
 * @param[in] support_value Support label value.
 * @param[out] packs Packs, or NULL to query the count.
 * @param[in] capacity Capacity of @p packs.
 * @param[out] count Owned facets of the support.
 * @return ::SS_OK, ::SS_ERROR_LIMIT or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_field_space_pack_owned_facets(const SsFieldSpace *space,
                                          uint32_t support_value,
                                          SsFieldFacetPack *packs, size_t capacity,
                                          size_t *count);
/** Copy the owned and ghost boundary facets of a support into packs; local.
 * @param[in] space Field space.
 * @param[in] support_value Support label value.
 * @param[out] packs Packs, or NULL to query the count.
 * @param[in] capacity Capacity of @p packs.
 * @param[out] count Incident facets of the support.
 * @return ::SS_OK, ::SS_ERROR_LIMIT or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_field_space_pack_incident_facets(const SsFieldSpace *space,
                                             uint32_t support_value,
                                             SsFieldFacetPack *packs, size_t capacity,
                                             size_t *count);
/** Create a vector over the free unknowns; collective.
 * @param[in] space Field space.
 * @param[out] vector New vector, owned by the caller.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_field_space_create_global_vector(const SsFieldSpace *space, Vec *vector);
/** One owned nodal functional and its component mappings into field vectors. */
typedef struct {
  SsEntityId entity_id;  /**< Entity carrying the node. */
  uint32_t entity_slot;  /**< Node slot on the entity. */
  PetscInt local_offset; /**< Local section offset of the node. */
  /** Global free row of each component, or negative when constrained. */
  PetscInt global_indices[SS_FE_MAX_DIMENSION];
  double prescribed[SS_FE_MAX_DIMENSION]; /**< Prescribed component values. */
} SsFieldOwnedNode;

/** Scatter free values into resident section offsets; collective.
 *
 * Restore the returned array and destroy its Vec reference before extracting
 * again on the same field space. Constrained offsets are unspecified; use the
 * plans' prescribed values.
 *
 * @param[in,out] space Field space (owns the scatter).
 * @param[in] source Global free-space vector.
 * @param[out] local Reference to the local work vector; destroy it after use.
 * @param[out] values Read-only local array; restore it with
 *   `VecRestoreArrayRead(*local, values)`.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_field_space_extract_local(SsFieldSpace *space, Vec source, Vec *local,
                                      const PetscScalar **values);

/** Borrow the cached table of owned nodes, including fully constrained ones;
 * local. The first call builds the table.
 * @param[in,out] space Field space.
 * @param[out] nodes Nodes valid until the space is destroyed.
 * @param[out] count Owned nodes.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_field_space_owned_nodes(SsFieldSpace *space, const SsFieldOwnedNode **nodes,
                                    size_t *count);

/** Largest Euclidean nodal norm of a field, including prescribed components;
 * collective.
 * @param[in,out] space Field space.
 * @param[in] state Free-space vector.
 * @param[out] maximum Global maximum.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_field_space_nodal_maximum(SsFieldSpace *space, Vec state,
                                      PetscReal *maximum);

/** Select the nested coarse nodal values from a fine free-space vector;
 * collective.
 * @param[in] coarse Coarse field space on the same mesh.
 * @param[in] fine Fine field space.
 * @param[out] injection New coarse-by-fine matrix, owned by the caller.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_field_space_injection_create(const SsFieldSpace *coarse,
                                         const SsFieldSpace *fine, Mat *injection);
/** Build the nodal interpolation from a coarse to a fine order on the same
 * mesh, zero on constrained fine rows; collective.
 * @param[in] coarse Coarse field space.
 * @param[in] fine Fine field space.
 * @param[out] prolongation New fine-by-coarse matrix, owned by the caller.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_field_space_transfer_create(const SsFieldSpace *coarse,
                                        const SsFieldSpace *fine, Mat *prolongation);
/** Destroy a field space; collective.
 * @param[in,out] space Space pointer, set to NULL; `*space` may be NULL.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_INTERNAL. */
SsStatus ss_field_space_destroy(SsFieldSpace **space);

/** Copy the owned boundary facets of a support with their geometry; local.
 * @param[in] mesh Mesh.
 * @param[in] support_value Support label value.
 * @param[out] packs Packs, or NULL to query the count.
 * @param[in] capacity Capacity of @p packs.
 * @param[out] count Owned facets of the support.
 * @return ::SS_OK, ::SS_ERROR_LIMIT or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_mesh_pack_facets(const SsMesh *mesh, uint32_t support_value,
                             SsFacetPack *packs, size_t capacity, size_t *count);
/** Geometry of a packed facet at a facet reference point; pure.
 * @param[in] pack Facet pack.
 * @param[in] facet_reference_point Facet reference coordinates.
 * @param[out] sample Geometry at the point.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT or a geometry failure. */
SsStatus ss_mesh_facet_geometry_at(const SsFacetPack *pack,
                                   const double *facet_reference_point,
                                   SsFacetSample *sample);
/** Whether this rank owns a resident cell; local.
 * @param[in] mesh Mesh.
 * @param[in] stable_cell_id Stable cell.
 * @param[out] owned Nonzero for an owned cell.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_mesh_cell_is_owned(const SsMesh *mesh, SsEntityId stable_cell_id,
                               int *owned);
/** Geometry nodes of a resident cell in canonical order; local.
 * @param[in] mesh Mesh.
 * @param[in] stable_cell_id Stable cell.
 * @param[out] node_ids Stable geometry nodes.
 * @param[out] coordinates Node-major coordinates.
 * @param[in] capacity Capacity of @p node_ids in nodes.
 * @param[out] node_count Geometry nodes of the cell.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_INTERNAL. */
SsStatus ss_mesh_cell_coordinate_closure(const SsMesh *mesh, SsEntityId stable_cell_id,
                                         SsEntityId *node_ids, double *coordinates,
                                         size_t capacity, size_t *node_count);

#endif
