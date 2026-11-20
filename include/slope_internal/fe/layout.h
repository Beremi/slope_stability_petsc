/** @file
 * @brief Data layout of the generated finite-element tables: element
 * descriptors, quadrature rules and node permutations.
 *
 * The tables in `generated/` are immutable and shared by all threads.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_FE_LAYOUT_H
#define SS_FE_LAYOUT_H

#include <stddef.h>
#include <stdint.h>

/** Largest spatial dimension. */
#define SS_FE_MAX_DIMENSION 3U
/** Largest number of nodes of one element (P8 tetrahedron). */
#define SS_FE_MAX_NODES 165U

/** Node ordering of a simplex element. */
typedef enum {
  SS_FE_ORDER_CANONICAL = 1, /**< Vertices, then edge, face and interior nodes. */
  SS_FE_ORDER_PETSC = 2,     /**< PETSc order; identical to the canonical one. */
  SS_FE_ORDER_GMSH = 3,      /**< Gmsh MSH node order. */
  SS_FE_ORDER_VTK = 4,       /**< VTK Lagrange node order. */
  SS_FE_ORDER_REFERENCE = 5  /**< Order of the earlier reference implementation. */
} SsFeOrdering;

/** Domain of a quadrature rule. */
typedef enum {
  SS_FE_QUADRATURE_VOLUME = 1, /**< Cell interior. */
  SS_FE_QUADRATURE_FACET = 2   /**< Cell facet (edge in 2D, face in 3D). */
} SsFeQuadratureDomain;

/**
 * @brief Integral families used to derive a minimum exact quadrature degree.
 *
 * With field order p, coordinate order g, coefficient order c and parent
 * dimension d, polynomial terms use the degree of their complete product,
 * including det J of degree d (g - 1).  Curved stiffness and diffusion contain
 * a rational metric and cannot be integrated exactly by any finite rule.
 */
typedef enum {
  /** Sampling for curved-geometry validity audits. */
  SS_FE_OPERATION_GEOMETRY_QUALITY = 1,
  /** The fused mechanics volume operator: internal force and tangent, body
   * load and its work share one rule, so the minimum covers p + c + d (g - 1).
   */
  SS_FE_OPERATION_MECHANICS_STIFFNESS = 2,
  /** Body load moments N_a f det J: p + c + d (g - 1). */
  SS_FE_OPERATION_MECHANICS_BODY = 3,
  /** Facet traction/pressure moments: p + c + (d - 1)(g - 1). */
  SS_FE_OPERATION_MECHANICS_NATURAL = 4,
  /** The fused seepage volume operator: diffusion and the nonlinear gravity
   * transport, whose ramp-transition residual and Jacobian have degree
   * 2p - 1 + c + (d - 1)(g - 1). */
  SS_FE_OPERATION_HYDRAULIC_DIFFUSION = 5,
  /** Facet flux moments: p + c + (d - 1)(g - 1). */
  SS_FE_OPERATION_HYDRAULIC_FLUX = 6,
  /** Mass moments N_a N_b det J: 2p + c + d (g - 1). */
  SS_FE_OPERATION_MASS = 7,
  /** L2 projection of material data: 2p + c + d (g - 1). */
  SS_FE_OPERATION_MATERIAL_PROJECTION = 8,
  /** Source-to-target transfer averages: p + c + d (g - 1). */
  SS_FE_OPERATION_COUPLING_TRANSFER = 9
} SsFeOperationKind;

/** How a quadrature plan selects its rule. */
typedef enum {
  /** The registered default rule of the element order. */
  SS_FE_QUADRATURE_MODE_DEFAULT = 1,
  /** The cheapest positive-weight rule of degree above the default rule's
   * degree that also meets the operation's minimum exact degree. */
  SS_FE_QUADRATURE_MODE_PAIRED_HIGHER = 2
} SsFeQuadratureMode;

/** Evaluate all basis functions at one reference point.
 * @param reference_point Reference coordinates, one per dimension.
 * @param values Basis values, one per node.
 * @param gradients Reference gradients, node-major with one entry per
 *   dimension. */
typedef void (*SsFeBasisScalarEvaluator)(const double *reference_point, double *values,
                                         double *gradients);
/** Evaluate all basis functions at several reference points.
 * @param point_count Number of points.
 * @param reference_points Point-major reference coordinates.
 * @param values Point-major basis values.
 * @param gradients Point-major, then node-major reference gradients. */
typedef void (*SsFeBasisBatchEvaluator)(size_t point_count,
                                        const double *reference_points, double *values,
                                        double *gradients);

/** Lagrange simplex element of one dimension and order. */
typedef struct {
  const char *id;          /**< Stable identifier, for example `triangle-p2-v1`. */
  uint32_t dimension;      /**< Reference dimension, 2 or 3. */
  uint32_t order;          /**< Polynomial order. */
  size_t node_count;       /**< Number of nodes. */
  size_t facet_node_count; /**< Nodes on one facet. */
  size_t edge_count;       /**< Number of cell edges. */
  size_t face_count;       /**< Number of cell faces (3D) or edges (2D). */
  /** Integer barycentric coordinates of each node, `dimension + 1` per node. */
  const uint8_t *barycentric_indices;
  /** Dimension of the topological entity that owns each node. */
  const uint8_t *entity_dimensions;
  /** Local index of the owning entity within its dimension. */
  const uint8_t *entity_indices;
  /** Slot of the node among the nodes of its owning entity. */
  const uint8_t *entity_slots;
  SsFeBasisScalarEvaluator basis_scalar_eval; /**< Single-point evaluator. */
  SsFeBasisBatchEvaluator basis_batch_eval;   /**< Batched evaluator. */
  const char *basis_digest;                   /**< Digest of the basis definition. */
  const char *generator_digest; /**< Digest of the generator source data. */
} SsFeDescriptor;

/** Quadrature rule on a reference simplex. */
typedef struct {
  const char *id;              /**< Stable identifier. */
  SsFeQuadratureDomain domain; /**< Volume or facet rule. */
  uint32_t parent_dimension;   /**< Dimension of the cell. */
  uint32_t dimension;          /**< Dimension of the integration domain. */
  uint32_t exact_degree;       /**< Polynomial degree integrated exactly. */
  size_t point_count;          /**< Number of points. */
  int has_negative_weight;     /**< Nonzero when any weight is negative. */
  const double *points;        /**< Point-major reference coordinates. */
  const double *weights;       /**< Weights on the reference simplex. */
  const char *source;          /**< Origin of the rule. */
} SsFeQuadrature;

/** Node permutation between the canonical and an external ordering. */
typedef struct {
  const char *id;                       /**< Stable identifier. */
  SsFeOrdering ordering;                /**< External ordering. */
  size_t node_count;                    /**< Number of nodes. */
  const uint8_t *canonical_to_external; /**< External index of each canonical node. */
  const uint8_t *external_to_canonical; /**< Canonical index of each external node. */
  const char *source_version;           /**< Version of the external format. */
  const char *node_coordinate_digest;   /**< Digest of the node coordinates. */
} SsFePermutation;

#endif
