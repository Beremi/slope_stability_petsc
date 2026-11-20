/** @file
 * @brief Elastic-perfectly plastic Mohr-Coulomb material with Davis A, B and C
 * strength reduction.
 *
 * Strains use engineering Voigt notation: `xx, yy, xy` in 2D (plane strain)
 * and `xx, yy, zz, xy, yz, xz` in 3D. Stresses use the same order, with the
 * out-of-plane `zz` appended in 2D. Angles are in radians; moduli, cohesion
 * and stresses are in the internal stress unit (kPa). Parameter objects are immutable
 * after creation; all evaluation functions are reentrant and safe inside OpenMP
 * regions.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_MATERIAL_MATERIAL_H
#define SS_MATERIAL_MATERIAL_H

#include "slope_internal/core/status.h"

#include <stddef.h>
#include <stdint.h>

/** Largest number of strain components (3D). */
#define SS_MC_MAX_STRAIN_COMPONENTS 6U
/** Largest number of stress components (3D). */
#define SS_MC_MAX_STRESS_COMPONENTS 6U
/** Largest number of tangent entries (6 x 6 in 3D). */
#define SS_MC_MAX_TANGENT_COMPONENTS 36U
/** Capacity of an ::SsMcError code. */
#define SS_MC_ERROR_CODE_CAPACITY 64U
/** Capacity of a `sha256:` identity. */
#define SS_MC_ID_CAPACITY 72U

/** Opaque, immutable material parameters; see ss_mc_parameters_create(). */
typedef struct SsMcParameters SsMcParameters;

/** Spatial dimension of an evaluation. */
typedef enum {
  SS_MC_DIMENSION_2 = 2, /**< Plane strain. */
  SS_MC_DIMENSION_3 = 3  /**< Three dimensions. */
} SsMcDimension;

/** Davis variant of strength reduction for non-associated flow. */
typedef enum {
  SS_MC_DAVIS_A = 1, /**< Davis A. */
  SS_MC_DAVIS_B = 2, /**< Davis B. */
  SS_MC_DAVIS_C = 3  /**< Davis C. */
} SsMcDavis;

/** Branch of the return mapping that produced a response. */
typedef enum {
  SS_MC_BRANCH_ELASTIC = 1,    /**< Elastic trial state. */
  SS_MC_BRANCH_SMOOTH = 2,     /**< Return to the smooth part of the yield surface. */
  SS_MC_BRANCH_LEFT_EDGE = 3,  /**< Return to the left edge. */
  SS_MC_BRANCH_RIGHT_EDGE = 4, /**< Return to the right edge. */
  SS_MC_BRANCH_APEX = 5        /**< Return to the apex. */
} SsMcBranch;

/** Robustness flags of an ::SsMcResponse. */
enum {
  SS_MC_ROBUST_ZERO_FRICTION = 1U << 0U,     /**< Reduced friction is near zero. */
  SS_MC_ROBUST_REPEATED_SPECTRUM = 1U << 1U, /**< Principal strains coincide. */
  SS_MC_ROBUST_NONSMOOTH = 1U << 2U          /**< State on an edge or the apex. */
};

/** Request flags of ss_mc_evaluate_prepared(). */
enum {
  SS_MC_EVALUATE_TANGENT = 1U << 0U /**< Compute the consistent tangent. */
};

/** Material parameters as authored. */
typedef struct {
  double young_modulus;          /**< Young's modulus, positive. */
  double poisson_ratio;          /**< Poisson's ratio in (-1, 0.5). */
  double cohesion;               /**< Cohesion, nonnegative. */
  double friction_angle;         /**< Friction angle in [0, pi/2) radians. */
  double dilatancy_angle;        /**< Dilatancy angle in [0, friction angle]. */
  SsMcDavis davis;               /**< Strength-reduction variant. */
  uint64_t material_region_id;   /**< Stable identity of the material region. */
  const char *provenance_digest; /**< `sha256:` identity of the source document. */
} SsMcParameterInput;

/** Strength parameters after reduction by a factor. */
typedef struct {
  double cohesion;         /**< Reduced cohesion. */
  double friction_angle;   /**< Reduced friction angle in radians. */
  double c_bar;            /**< `2 c cos(phi)` of the reduced parameters. */
  double sin_friction;     /**< `sin(phi)` of the reduced friction angle. */
  double beta;             /**< Davis factor applied to cohesion and friction. */
  int davis_c_high_branch; /**< Davis C: nonzero when the reduced friction
                                exceeds the dilatancy angle. */
} SsMcReduced;

/** Validated parameters with derived elastic constants and identity. */
typedef struct {
  double young_modulus;                      /**< Young's modulus. */
  double poisson_ratio;                      /**< Poisson's ratio. */
  double cohesion;                           /**< Cohesion. */
  double friction_angle;                     /**< Friction angle in radians. */
  double dilatancy_angle;                    /**< Dilatancy angle in radians. */
  double shear_modulus;                      /**< Shear modulus. */
  double bulk_modulus;                       /**< Bulk modulus. */
  double lame_first;                         /**< First Lame parameter. */
  SsMcDavis davis;                           /**< Strength-reduction variant. */
  uint64_t material_region_id;               /**< Stable region identity. */
  char provenance_digest[SS_MC_ID_CAPACITY]; /**< Identity of the source. */
  char identity[SS_MC_ID_CAPACITY];          /**< Identity of the parameters. */
} SsMcParameterSummary;

/** First material error of an evaluation, ordered deterministically. */
typedef struct {
  int valid;                            /**< Nonzero when an error is recorded. */
  SsStatus status;                      /**< Status of the error. */
  size_t local_index;                   /**< Batch index of the failing point. */
  uint64_t stable_entity_id;            /**< Stable cell of the failing point. */
  uint32_t quadrature_index;            /**< Quadrature point within the cell. */
  char code[SS_MC_ERROR_CODE_CAPACITY]; /**< Error code, for example `MC_COHESION`. */
} SsMcError;

/** Response at one material point. */
typedef struct {
  /** Stress in Voigt order (4 components in 2D, 6 in 3D). */
  double stress[SS_MC_MAX_STRESS_COMPONENTS];
  /** Consistent tangent, column-major (3 x 3 in 2D, 6 x 6 in 3D); zero unless
   * requested. */
  double tangent[SS_MC_MAX_TANGENT_COMPONENTS];
  double potential;          /**< Potential whose strain gradient is the stress. */
  double trial_yield;        /**< Yield function of the elastic trial stress. */
  double yield_residual;     /**< Yield function of the returned stress. */
  SsMcReduced reduced;       /**< Reduced strength parameters used. */
  SsMcBranch branch;         /**< Return-mapping branch. */
  uint32_t robustness_flags; /**< Bitwise OR of `SS_MC_ROBUST_*` flags. */
} SsMcResponse;

/** Strains of a batch of points, stored component-major. */
typedef struct {
  size_t count;  /**< Number of points. */
  size_t stride; /**< Distance between components, at least @ref count. */
  /** Strain component `k` of point `i` at `strain[k * stride + i]`. */
  const double *strain;
  const uint64_t *stable_entity_ids;  /**< Stable cell of each point. */
  const uint32_t *quadrature_indices; /**< Quadrature index of each point. */
} SsMcBatchInput;

/** Responses of a batch of points, stored component-major like the input. */
typedef struct {
  size_t stride;              /**< Distance between components. */
  double *stress;             /**< Stress components. */
  double *tangent;            /**< Tangent entries. */
  double *potential;          /**< Potential per point. */
  double *trial_yield;        /**< Trial yield function per point. */
  double *yield_residual;     /**< Returned yield function per point. */
  SsMcBranch *branch;         /**< Branch per point. */
  uint32_t *robustness_flags; /**< Robustness flags per point. */
} SsMcBatchOutput;

/** Validate authored parameters and create an immutable parameter object.
 * @param[in] input Authored parameters.
 * @param[out] parameters New object, released with ss_mc_parameters_destroy().
 * @param[out] error Error code of an invalid parameter; may be NULL.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT for an invalid parameter, or
 *   ::SS_ERROR_OUT_OF_MEMORY. */
SsStatus ss_mc_parameters_create(const SsMcParameterInput *input,
                                 SsMcParameters **parameters, SsMcError *error);
/** Copy the validated parameters and derived constants.
 * @param[in] parameters Parameter object.
 * @param[out] summary Copy of the parameters.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_mc_parameters_summary(const SsMcParameters *parameters,
                                  SsMcParameterSummary *summary);
/** Release a parameter object.
 * @param[in,out] parameters Object pointer, set to NULL; `*parameters` may be NULL.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_mc_parameters_destroy(SsMcParameters **parameters);

/** Reduce the strength parameters by a factor with the Davis variant.
 * @param[in] parameters Parameter object.
 * @param[in] reduction_factor Positive strength-reduction factor.
 * @param[out] reduced Reduced parameters.
 * @param[out] error Error code on failure; may be NULL.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT for an invalid factor or a
 *   degenerate Davis denominator. */
SsStatus ss_mc_reduce(const SsMcParameters *parameters, double reduction_factor,
                      SsMcReduced *reduced, SsMcError *error);
/** Evaluate one point with previously reduced parameters.
 * @param[in] parameters Parameter object.
 * @param[in] dimension Spatial dimension.
 * @param[in] reduced Result of ss_mc_reduce().
 * @param[in] request Bitwise OR of request flags such as ::SS_MC_EVALUATE_TANGENT.
 * @param[in] engineering_strain Strain in Voigt order (3 or 6 components).
 * @param[out] response Stress, tangent and diagnostics.
 * @param[out] error Error code on failure; may be NULL.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT for invalid or nonfinite input,
 *   or ::SS_ERROR_INTERNAL when the spectral decomposition fails. */
SsStatus ss_mc_evaluate_prepared(const SsMcParameters *parameters,
                                 SsMcDimension dimension, const SsMcReduced *reduced,
                                 uint32_t request, const double *engineering_strain,
                                 SsMcResponse *response, SsMcError *error);
/** Reduce the parameters and evaluate one point with its tangent.
 * @param[in] parameters Parameter object.
 * @param[in] dimension Spatial dimension.
 * @param[in] reduction_factor Positive strength-reduction factor.
 * @param[in] engineering_strain Strain in Voigt order (3 or 6 components).
 * @param[out] response Stress, tangent and diagnostics.
 * @param[out] error Error code on failure; may be NULL.
 * @return As ss_mc_reduce() and ss_mc_evaluate_prepared(). */
SsStatus ss_mc_evaluate(const SsMcParameters *parameters, SsMcDimension dimension,
                        double reduction_factor, const double *engineering_strain,
                        SsMcResponse *response, SsMcError *error);
/** Evaluate a batch of points with their tangents.
 *
 * On failure, @p error describes the point with the smallest stable cell and
 * quadrature index, independent of the batch order.
 *
 * @param[in] parameters Parameter object.
 * @param[in] dimension Spatial dimension.
 * @param[in] reduction_factor Positive strength-reduction factor.
 * @param[in] input Component-major strains and point identities.
 * @param[out] output Component-major responses.
 * @param[out] error First error in stable order; may be NULL.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_INTERNAL. */
SsStatus ss_mc_evaluate_batch(const SsMcParameters *parameters, SsMcDimension dimension,
                              double reduction_factor, const SsMcBatchInput *input,
                              SsMcBatchOutput *output, SsMcError *error);

#endif
