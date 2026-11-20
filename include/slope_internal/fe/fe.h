/** @file
 * @brief Simplex bases, quadrature planning, isoparametric geometry, node
 * permutations and order-to-order transfer.
 *
 * Every function is a pure function of its arguments and of the immutable
 * generated tables, so all of them are safe inside OpenMP regions. Arrays
 * of reference points and of nodal values are point-major (node-major).
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_FE_FE_H
#define SS_FE_FE_H

#include "slope_internal/core/status.h"
#include "slope_internal/fe/layout.h"

#include <stddef.h>
#include <stdint.h>

/** Admissibility limits for an isoparametric map at one reference point. */
typedef struct {
  double absolute_determinant_tolerance; /**< Smallest admissible `det J`. */
  double maximum_condition; /**< Largest admissible condition estimate of `J`. */
} SsFeGeometryPolicy;

/** Physical point, Jacobian, inverse transpose and determinant of a map. */
typedef struct {
  uint32_t dimension;                   /**< Spatial dimension. */
  double physical[SS_FE_MAX_DIMENSION]; /**< Physical point. */
  /** Jacobian `J`, row-major. */
  double jacobian[SS_FE_MAX_DIMENSION * SS_FE_MAX_DIMENSION];
  /** `J^{-T}`, row-major. */
  double inverse_transpose[SS_FE_MAX_DIMENSION * SS_FE_MAX_DIMENSION];
  double determinant;        /**< `det J`. */
  double condition_estimate; /**< Frobenius-norm condition estimate of `J`. */
} SsFeGeometry;

/** Physical point, tangents, outward unit normal and measure density of a
 * facet sample. */
typedef struct {
  uint32_t dimension;                   /**< Spatial dimension of the parent. */
  double physical[SS_FE_MAX_DIMENSION]; /**< Physical point. */
  /** Facet tangents, one vector per facet direction. */
  double tangents[SS_FE_MAX_DIMENSION * (SS_FE_MAX_DIMENSION - 1U)];
  double outward_normal[SS_FE_MAX_DIMENSION]; /**< Outward unit normal. */
  double measure; /**< Facet measure density (length or area per reference unit). */
} SsFeFacetGeometry;

/** Integral family, discretization orders and selection mode of a quadrature
 * request; a non-NULL override names a registered rule. */
typedef struct {
  uint32_t parent_dimension;    /**< Dimension of the cell. */
  SsFeQuadratureDomain domain;  /**< Volume or facet rule. */
  SsFeOperationKind operation;  /**< Integral family. */
  uint32_t field_order;         /**< Field order `p`. */
  uint32_t coordinate_order;    /**< Geometry order `g`. */
  uint32_t coefficient_order;   /**< Coefficient order `c`. */
  SsFeQuadratureMode mode;      /**< Rule selection mode. */
  const char *override_rule_id; /**< Registered rule to use instead, or NULL. */
} SsFeQuadratureRequest;

/** Resolved rules: the selected rule, the default rule, the cheapest paired
 * control of strictly higher degree (NULL when none exists and the mode is the
 * default) and the minimum exact degree the request requires. */
typedef struct {
  const SsFeQuadrature *selected_rule; /**< Rule to integrate with. */
  const SsFeQuadrature *default_rule;  /**< Default rule of the request. */
  const SsFeQuadrature *paired_rule;   /**< Paired higher rule, or NULL. */
  uint32_t minimum_exact_degree;       /**< Degree the request requires. */
  int used_override;                   /**< Nonzero when the override was used. */
} SsFeQuadraturePlan;

/** Find the generated simplex descriptor of a dimension and order.
 * @param[in] dimension Reference dimension, 2 or 3.
 * @param[in] order Polynomial order.
 * @param[out] descriptor Borrowed static descriptor.
 * @return ::SS_OK, ::SS_ERROR_NOT_FOUND or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_fe_descriptor_find(uint32_t dimension, uint32_t order,
                               const SsFeDescriptor **descriptor);
/** Evaluate all basis values and, optionally, reference gradients at one
 * reference point.
 * @param[in] descriptor Element descriptor.
 * @param[in] reference_point Reference coordinates.
 * @param[out] values One value per node.
 * @param[out] gradients Node-major reference gradients, or NULL.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_fe_basis_eval_scalar(const SsFeDescriptor *descriptor,
                                 const double *reference_point, double *values,
                                 double *gradients);
/** Evaluate basis values and, optionally, reference gradients at a batch of
 * reference points.
 * @param[in] descriptor Element descriptor.
 * @param[in] point_count Number of points.
 * @param[in] reference_points Point-major reference coordinates.
 * @param[out] values Point-major basis values.
 * @param[out] gradients Point-major, node-major reference gradients, or NULL.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_fe_basis_eval(const SsFeDescriptor *descriptor, size_t point_count,
                          const double *reference_points, double *values,
                          double *gradients);

/** Legacy order-indexed volume rule (affine stiffness degree). New code plans
 * rules with ss_fe_quadrature_plan().
 * @param[in] dimension Reference dimension.
 * @param[in] order Field order.
 * @param[out] quadrature Borrowed static rule.
 * @return ::SS_OK, ::SS_ERROR_NOT_FOUND or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_fe_quadrature_default(uint32_t dimension, uint32_t order,
                                  const SsFeQuadrature **quadrature);
/** Order-indexed facet rule used by natural-boundary operators.
 * @param[in] parent_dimension Dimension of the parent cell.
 * @param[in] order Field order.
 * @param[out] quadrature Borrowed static rule.
 * @return ::SS_OK, ::SS_ERROR_NOT_FOUND or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_fe_facet_quadrature_default(uint32_t parent_dimension, uint32_t order,
                                        const SsFeQuadrature **quadrature);
/** Find a registered rule by its stable identifier.
 * @param[in] id Rule identifier.
 * @param[out] quadrature Borrowed static rule.
 * @return ::SS_OK, ::SS_ERROR_NOT_FOUND or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_fe_quadrature_find(const char *id, const SsFeQuadrature **quadrature);
/** Number of registered quadrature rules.
 * @return Size of the rule registry. */
size_t ss_fe_quadrature_registry_count(void);
/** Registered rule at a registry index.
 * @param[in] index Index below ss_fe_quadrature_registry_count().
 * @param[out] quadrature Borrowed static rule.
 * @return ::SS_OK, ::SS_ERROR_NOT_FOUND or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_fe_quadrature_registry_at(size_t index, const SsFeQuadrature **quadrature);
/** Verify point admissibility, weight signs and monomial exactness up to the
 * declared degree.
 * @param[in] quadrature Rule to verify.
 * @param[out] bad_index First failing point; may be NULL.
 * @return ::SS_OK, ::SS_ERROR_INTEGRITY or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_fe_quadrature_validate(const SsFeQuadrature *quadrature, size_t *bad_index);
/**
 * Resolve the rule for an integral family.
 *
 * The minimum exact degree follows ::SsFeOperationKind. The default rule is
 * the positive rule with the fewest points among those exact to that degree
 * (ties: lower degree, then identifier); an override must be registered,
 * positive and exact to that degree.
 *
 * @param[in] request Integral family and orders.
 * @param[out] plan Resolved rules.
 * @return ::SS_OK; ::SS_ERROR_INTEGRITY for an under-degree or signed override;
 *   ::SS_ERROR_NOT_FOUND when no sufficient (or no paired) rule exists;
 *   ::SS_ERROR_INVALID_ARGUMENT.
 */
SsStatus ss_fe_quadrature_plan(const SsFeQuadratureRequest *request,
                               SsFeQuadraturePlan *plan);

/** Evaluate the isoparametric map at one reference point.
 * @param[in] descriptor Geometry descriptor.
 * @param[in] coordinates Node-major coordinate nodes of the cell.
 * @param[in] reference_point Reference coordinates.
 * @param[in] policy Admissibility limits.
 * @param[out] geometry Map data at the point.
 * @return ::SS_OK; ::SS_ERROR_INTEGRITY for a nonfinite, inverted or
 *   ill-conditioned map; ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_fe_geometry_eval(const SsFeDescriptor *descriptor,
                             const double *coordinates, const double *reference_point,
                             const SsFeGeometryPolicy *policy, SsFeGeometry *geometry);
/** Evaluate the map at several reference points.
 * @param[in] descriptor Geometry descriptor.
 * @param[in] coordinates Node-major coordinate nodes of the cell.
 * @param[in] point_count Number of points.
 * @param[in] reference_points Point-major reference coordinates.
 * @param[in] policy Admissibility limits.
 * @param[out] bad_index First failing point.
 * @return ::SS_OK, the status of the first failing point, or
 *   ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_fe_geometry_audit(const SsFeDescriptor *descriptor,
                              const double *coordinates, size_t point_count,
                              const double *reference_points,
                              const SsFeGeometryPolicy *policy, size_t *bad_index);
/**
 * Map node-major reference gradients to physical gradients by `J^{-T}`
 * without argument checks.
 *
 * This is the single implementation shared by cached and on-the-fly geometry,
 * so both produce identical values.
 *
 * @param[in] dimension Spatial dimension.
 * @param[in] node_count Number of nodes.
 * @param[in] inverse_transpose Row-major `J^{-T}`.
 * @param[in] reference_gradients Node-major reference gradients.
 * @param[out] physical_gradients Node-major physical gradients.
 */
void ss_fe_physical_gradients_apply(uint32_t dimension, size_t node_count,
                                    const double *inverse_transpose,
                                    const double *reference_gradients,
                                    double *physical_gradients);
/** Map reference basis gradients to physical gradients by `J^{-T}`.
 * @param[in] descriptor Element descriptor (node count).
 * @param[in] geometry Map data at the point.
 * @param[in] reference_gradients Node-major reference gradients.
 * @param[out] physical_gradients Node-major physical gradients.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_fe_physical_gradients(const SsFeDescriptor *descriptor,
                                  const SsFeGeometry *geometry,
                                  const double *reference_gradients,
                                  double *physical_gradients);
/** Map facet reference coordinates to parent reference coordinates.
 * @param[in] dimension Dimension of the parent cell.
 * @param[in] facet_index Local facet index in reference ordering.
 * @param[in] facet_reference_point `dimension - 1` facet coordinates.
 * @param[out] parent_reference_point `dimension` parent coordinates.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_fe_facet_parent_reference(uint32_t dimension, uint32_t facet_index,
                                      const double *facet_reference_point,
                                      double *parent_reference_point);
/**
 * Evaluate the physical point, tangents, outward unit normal and measure
 * density of a parent-cell facet.
 *
 * The outward sign is topological: the reference facet normal is mapped by the
 * cofactor of the parent Jacobian, whose determinant must be positive. It is
 * therefore correct on every admissible curved cell, including cells that are
 * not star-shaped with respect to their centroid.
 *
 * @param[in] descriptor Coordinate (geometry) descriptor of the parent.
 * @param[in] facet_index Local facet index in reference ordering.
 * @param[in] coordinates Parent coordinate nodes, node-major.
 * @param[in] facet_reference_point Facet reference coordinates (d - 1 values).
 * @param[in] parent_centroid Unused; retained for interface stability and may
 *   be NULL. It is never an orientation authority.
 * @param[out] facet Output geometry.
 * @return ::SS_OK, ::SS_ERROR_NOT_FOUND for a bad facet index,
 *   ::SS_ERROR_INTEGRITY for a degenerate or inverted parent map.
 */
SsStatus ss_fe_facet_geometry_eval(const SsFeDescriptor *descriptor,
                                   uint32_t facet_index, const double *coordinates,
                                   const double *facet_reference_point,
                                   const double *parent_centroid,
                                   SsFeFacetGeometry *facet);

/** Find the node permutation between the canonical and an external ordering.
 * @param[in] descriptor Element descriptor.
 * @param[in] ordering External ordering.
 * @param[out] permutation Borrowed static permutation.
 * @return ::SS_OK, ::SS_ERROR_NOT_FOUND or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_fe_permutation_find(const SsFeDescriptor *descriptor, SsFeOrdering ordering,
                                const SsFePermutation **permutation);
/** Reorder node-major values from the canonical to the external ordering.
 * @param[in] permutation Node permutation.
 * @param[in] components Values per node.
 * @param[in] canonical Values in canonical order.
 * @param[out] external Values in external order; must not alias @p canonical.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_fe_permute_to_external(const SsFePermutation *permutation,
                                   size_t components, const double *canonical,
                                   double *external);
/** Reorder node-major values from the external to the canonical ordering.
 * @param[in] permutation Node permutation.
 * @param[in] components Values per node.
 * @param[in] external Values in external order.
 * @param[out] canonical Values in canonical order; must not alias @p external.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_fe_permute_to_canonical(const SsFePermutation *permutation,
                                    size_t components, const double *external,
                                    double *canonical);

/** Build the fine-by-coarse nodal interpolation matrix of nested spaces.
 * @param[in] coarse Coarse element of lower order.
 * @param[in] fine Fine element of the same dimension.
 * @param[out] matrix Row-major matrix, one row per fine node.
 * @param[in] capacity Capacity of @p matrix in entries.
 * @param[out] rows Number of fine nodes.
 * @param[out] columns Number of coarse nodes.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_INTERNAL. */
SsStatus ss_fe_transfer_build(const SsFeDescriptor *coarse, const SsFeDescriptor *fine,
                              double *matrix, size_t capacity, size_t *rows,
                              size_t *columns);
/** Apply an interpolation matrix, writing zero to constrained fine rows.
 * @param[in] rows Number of fine nodes.
 * @param[in] columns Number of coarse nodes.
 * @param[in] matrix Row-major interpolation matrix.
 * @param[in] components Values per node.
 * @param[in] coarse_values Node-major coarse values.
 * @param[in] constrained_rows Nonzero for each constrained fine node.
 * @param[out] fine_values Node-major fine values.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_fe_transfer_apply_constrained(size_t rows, size_t columns,
                                          const double *matrix, size_t components,
                                          const double *coarse_values,
                                          const uint8_t *constrained_rows,
                                          double *fine_values);

#endif
