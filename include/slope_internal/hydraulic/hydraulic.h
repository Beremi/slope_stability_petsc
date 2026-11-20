/** @file
 * @brief Steady unconfined seepage: the saturation ramp, Darcy flux kernels,
 * boundary pressure and flux data, the per-cell operator model and the
 * global flux balance.
 *
 * The flux is `q = k (grad p + gamma_w ramp(p) e_up)` with a linear ramp of
 * width `epsilon` between dry and saturated states.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_HYDRAULIC_HYDRAULIC_H
#define SS_HYDRAULIC_HYDRAULIC_H

#include "slope_internal/assembly/assembly.h"
#include "slope_internal/core/status.h"

#include <mpi.h>

#include <stddef.h>
#include <stdint.h>

/** Capacity of a hydraulic identity or unit string. */
#define SS_HYDRAULIC_ID_CAPACITY 72U
/** Largest number of breakpoints of a piecewise-linear head level. */
#define SS_HYDRAULIC_HEAD_POINT_CAPACITY 32U

/** Region of the saturation ramp. */
typedef enum {
  SS_HYDRAULIC_RAMP_DRY = 1,        /**< `p <= 0`: ramp value 0. */
  SS_HYDRAULIC_RAMP_TRANSITION = 2, /**< `0 < p < epsilon`: value `p / epsilon`. */
  SS_HYDRAULIC_RAMP_SATURATED = 3   /**< `p >= epsilon`: value 1. */
} SsHydraulicRampRegion;

/** Saturation ramp at one pressure. */
typedef struct {
  SsHydraulicRampRegion region; /**< Region of the pressure. */
  double epsilon;               /**< Ramp width (pressure units). */
  double value;                 /**< Ramp value in [0, 1]. */
  double derivative;            /**< Derivative with respect to pressure. */
} SsHydraulicRamp;

/** Treatment of the gravity term `gamma_w * ramp(p)` in the Darcy flux. */
typedef enum {
  SS_HYDRAULIC_GRAVITY_NONLINEAR = 1, /**< Ramp-weighted gravity (the model). */
  SS_HYDRAULIC_GRAVITY_DISABLED = 2,  /**< No gravity term (linear initializer). */
  SS_HYDRAULIC_GRAVITY_SATURATED = 3  /**< Full gravity everywhere (linear). */
} SsHydraulicGravityMode;

/** Initial pressure of the seepage Newton iteration. */
typedef enum {
  /** Envelope of the boundary heads. */
  SS_HYDRAULIC_INITIAL_BOUNDARY_ENVELOPE = 1,
  /** Solution of the linear, fully saturated problem. */
  SS_HYDRAULIC_INITIAL_SATURATED_LINEAR = 2
} SsHydraulicInitializationChoice;

/** Choice of the initial pressure and its recovery path. */
typedef struct {
  SsHydraulicInitializationChoice primary_choice; /**< First choice. */
  SsHydraulicInitializationChoice choice;         /**< Current choice. */
  double boundary_residual_norm;  /**< Residual norm of the boundary envelope. */
  double saturated_residual_norm; /**< Residual norm of the linear solution. */
  int saturated_candidate_valid;  /**< Nonzero when the linear solution exists. */
  int fallback_attempted;         /**< Nonzero after the one allowed fallback. */
} SsHydraulicInitializationRecord;

/** Material data of one seepage point. */
typedef struct {
  uint32_t dimension;   /**< Spatial dimension. */
  uint32_t upward_axis; /**< Index of the vertical axis. */
  double gamma_water;   /**< Unit weight of water. */
  double conductivity;  /**< Positive hydraulic conductivity. */
  double epsilon;       /**< Positive ramp width. */
} SsHydraulicPointModel;

/** Constitutive response at one seepage point. */
typedef struct {
  double pressure;                      /**< Pore pressure. */
  double gradient[SS_FE_MAX_DIMENSION]; /**< Pressure gradient. */
  /** Flux `k (grad p + gamma_w ramp(p) e_up)`. */
  double pressure_flux[SS_FE_MAX_DIMENSION];
  SsHydraulicRamp ramp; /**< Saturation ramp. */
} SsHydraulicPointResult;

/** Kind of prescribed pressure (head) boundary data. */
typedef enum {
  SS_HYDRAULIC_HEAD_DRY = 1,                    /**< Zero pressure. */
  SS_HYDRAULIC_HEAD_CONSTANT_LEVEL = 2,         /**< Hydrostatic below a level. */
  SS_HYDRAULIC_HEAD_PIECEWISE_LINEAR_LEVEL = 3, /**< Level varying along an axis. */
  SS_HYDRAULIC_HEAD_COMPILED_AFFINE_LEVEL =
      4,                            /**< Affine level `a0 + sum_i a_i x_i`. */
  SS_HYDRAULIC_HEAD_NODAL_VALUE = 5 /**< Fixed pressure value. */
} SsHydraulicHeadKind;

/** Prescribed pressure on a boundary: `gamma_w (level - x_up)`, or zero above
 * the level. */
typedef struct {
  SsHydraulicHeadKind kind; /**< Kind of data. */
  uint32_t dimension;       /**< Spatial dimension. */
  uint32_t upward_axis;     /**< Index of the vertical axis. */
  uint32_t piecewise_axis;  /**< Axis of a piecewise-linear level. */
  double gamma_water;       /**< Unit weight of water. */
  double level;             /**< Constant water level. */
  /** Affine level coefficients: constant, then one per coordinate. */
  double affine_level[SS_FE_MAX_DIMENSION + 1U];
  size_t piecewise_point_count; /**< Breakpoints of a piecewise level (2 or more). */
  /** Strictly increasing breakpoint coordinates. */
  double piecewise_coordinates[SS_HYDRAULIC_HEAD_POINT_CAPACITY];
  /** Water level at each breakpoint. */
  double piecewise_levels[SS_HYDRAULIC_HEAD_POINT_CAPACITY];
  double nodal_value;                      /**< Fixed pressure value. */
  char identity[SS_HYDRAULIC_ID_CAPACITY]; /**< Identity of the data. */
} SsHydraulicHeadModel;

/** Kind of prescribed normal flux. */
typedef enum {
  SS_HYDRAULIC_FLUX_ZERO = 1,     /**< No flux. */
  SS_HYDRAULIC_FLUX_CONSTANT = 2, /**< Constant normal flux. */
  SS_HYDRAULIC_FLUX_COMPILED_AFFINE =
      3, /**< Affine normal flux `a0 + sum_i a_i x_i`. */
  SS_HYDRAULIC_FLUX_COMPILED_VECTOR_NORMAL =
      4 /**< Normal component of a constant vector. */
} SsHydraulicFluxKind;

/** Prescribed normal flux on a boundary. */
typedef struct {
  SsHydraulicFluxKind kind; /**< Kind of data. */
  uint32_t dimension;       /**< Spatial dimension. */
  double value;             /**< Constant normal flux. */
  /** Affine coefficients: constant, then one per coordinate. */
  double affine[SS_FE_MAX_DIMENSION + 1U];
  double vector[SS_FE_MAX_DIMENSION];      /**< Flux vector `q`. */
  char identity[SS_HYDRAULIC_ID_CAPACITY]; /**< Identity of the data. */
} SsHydraulicFluxModel;

/** Rank-local contributions to the global flux balance. */
typedef struct {
  double domain_source;            /**< Volume source. */
  double prescribed_boundary_flux; /**< Prescribed boundary flux. */
  double natural_boundary_flux;    /**< Natural (outflow) boundary flux. */
  double dirichlet_reaction_flux;  /**< Reaction flux on prescribed pressures. */
  double free_residual_l2_squared; /**< Squared residual norm on free rows. */
} SsHydraulicBalanceLocal;

/** Global flux balance of a seepage solution, identical on all ranks. */
typedef struct {
  double domain_source;            /**< Volume source. */
  double prescribed_boundary_flux; /**< Prescribed boundary flux. */
  double natural_boundary_flux;    /**< Natural boundary flux. */
  double dirichlet_reaction_flux;  /**< Reaction flux. */
  double signed_imbalance;         /**< Sum of the four fluxes. */
  double physical_scale;           /**< Lower bound of the normalization. */
  /** Maximum of the sum of absolute fluxes and @ref physical_scale. */
  double normalization;
  double normalized_imbalance; /**< `|signed_imbalance| / normalization`. */
  double free_residual_l2;     /**< Residual norm on free rows. */
  int accepted;                /**< Nonzero when both tolerances hold. */
} SsHydraulicBalance;

/** Algebraic traits of the hydraulic operator. */
typedef struct {
  int scalar;                     /**< One unknown per node. */
  int real;                       /**< Real-valued. */
  int generally_nonsymmetric;     /**< The Jacobian is not symmetric. */
  int same_free_row_column_space; /**< Rows and columns share the free space. */
  int fixed_structure;            /**< The sparsity pattern is fixed. */
  int pressure_dependent_values;  /**< Values depend on the pressure. */
  char identity[SS_HYDRAULIC_ID_CAPACITY]; /**< Identity of the traits. */
} SsHydraulicOperatorTraits;

/** Seepage coefficients of one cell. */
typedef struct {
  SsEntityId stable_cell_id;  /**< Stable cell identity. */
  double epsilon;             /**< Ramp width of the cell. */
  size_t conductivity_count;  /**< 1 (constant) or one per quadrature point. */
  const double *conductivity; /**< Conductivity values. */
} SsHydraulicCellCoefficients;

/** Inputs of ss_hydraulic_operator_model_create(); the arrays are copied. */
typedef struct {
  uint32_t dimension;                               /**< Spatial dimension. */
  uint32_t upward_axis;                             /**< Index of the vertical axis. */
  double gamma_water;                               /**< Unit weight of water. */
  size_t cell_count;                                /**< Number of local cells. */
  const SsHydraulicCellCoefficients *cells;         /**< Coefficients of each cell. */
  char model_identity[SS_HYDRAULIC_ID_CAPACITY];    /**< Identity of the model. */
  char pressure_unit[SS_HYDRAULIC_ID_CAPACITY];     /**< Pressure unit. */
  char coordinate_unit[SS_HYDRAULIC_ID_CAPACITY];   /**< Coordinate unit. */
  char conductivity_unit[SS_HYDRAULIC_ID_CAPACITY]; /**< Conductivity unit. */
} SsHydraulicOperatorModelSpec;

/** Ranges and identity of an operator model on this rank. */
typedef struct {
  uint32_t dimension;                  /**< Spatial dimension. */
  uint32_t upward_axis;                /**< Index of the vertical axis. */
  double gamma_water;                  /**< Unit weight of water. */
  size_t cell_count;                   /**< Number of local cells. */
  size_t quadrature_coefficient_count; /**< Number of stored conductivities. */
  double conductivity_minimum;         /**< Smallest conductivity. */
  double conductivity_maximum;         /**< Largest conductivity. */
  double epsilon_minimum;              /**< Smallest ramp width. */
  double epsilon_maximum;              /**< Largest ramp width. */
  char model_identity[SS_HYDRAULIC_ID_CAPACITY];    /**< Identity of the model. */
  char pressure_unit[SS_HYDRAULIC_ID_CAPACITY];     /**< Pressure unit. */
  char coordinate_unit[SS_HYDRAULIC_ID_CAPACITY];   /**< Coordinate unit. */
  char conductivity_unit[SS_HYDRAULIC_ID_CAPACITY]; /**< Conductivity unit. */
} SsHydraulicOperatorModelSummary;

/** Opaque per-cell operator model; see ss_hydraulic_operator_model_create(). */
typedef struct SsHydraulicOperatorModel SsHydraulicOperatorModel;

/** Evaluate the saturation ramp; pure.
 * @param[in] pressure Pore pressure.
 * @param[in] epsilon Positive ramp width.
 * @param[out] ramp Ramp region, value and derivative.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_hydraulic_ramp(double pressure, double epsilon, SsHydraulicRamp *ramp);
/** Choose the initial pressure: the saturated linear solution when it exists,
 * else the boundary envelope.
 * @param[in] boundary_residual_norm Residual norm of the boundary envelope.
 * @param[in] saturated_residual_norm Residual norm of the linear solution.
 * @param[in] saturated_candidate_valid Nonzero when the linear solution exists.
 * @param[out] record Choice and its inputs.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_hydraulic_initialization_select(double boundary_residual_norm,
                                            double saturated_residual_norm,
                                            int saturated_candidate_valid,
                                            SsHydraulicInitializationRecord *record);
/** Switch once to the other initial pressure after Newton rejected the first.
 * @param[in,out] record Choice record.
 * @return ::SS_OK, ::SS_ERROR_INVALID_STATE when no fallback is available, or
 *   ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_hydraulic_initialization_fallback(SsHydraulicInitializationRecord *record);
/** Ramp width of a cell: `gamma_w h_min / 2` with the shortest edge `h_min`.
 * @param[in] dimension Spatial dimension.
 * @param[in] vertex_coordinates Vertex-major coordinates of the simplex.
 * @param[in] gamma_water Unit weight of water.
 * @param[out] minimum_edge Shortest edge length.
 * @param[out] epsilon Ramp width.
 * @return ::SS_OK, ::SS_ERROR_INTEGRITY for a degenerate cell, or
 *   ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_hydraulic_cell_epsilon(uint32_t dimension, const double *vertex_coordinates,
                                   double gamma_water, double *minimum_edge,
                                   double *epsilon);
/** Evaluate the Darcy flux of the unconfined model at one point; pure.
 * @param[in] model Point material data.
 * @param[in] pressure Pore pressure.
 * @param[in] gradient Pressure gradient.
 * @param[out] result Flux and ramp.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_INTEGRITY for a
 *   nonfinite flux. */
SsStatus ss_hydraulic_point_evaluate(const SsHydraulicPointModel *model,
                                     double pressure, const double *gradient,
                                     SsHydraulicPointResult *result);
/** Scalar assembly kernel for one constant material; safe inside OpenMP
 * regions.
 * @param[in] context ::SsHydraulicPointModel.
 * @param[in] input Point value and gradient.
 * @param[out] output Flux and its derivatives.
 * @return As ss_hydraulic_point_evaluate(). */
SsStatus ss_hydraulic_point_kernel(const void *context, const SsScalarPointInput *input,
                                   SsScalarPointOutput *output);
/** Evaluate prescribed boundary pressure at a point; pure.
 * @param[in] model Head data.
 * @param[in] physical Point coordinates.
 * @param[out] pressure Prescribed pressure.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_hydraulic_head_evaluate(const SsHydraulicHeadModel *model,
                                    const double *physical, double *pressure);
/** Evaluate a prescribed normal flux that does not need the normal; pure.
 * @param[in] model Flux data.
 * @param[in] physical Point coordinates.
 * @param[out] normal_flux Normal flux.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT (also for
 *   ::SS_HYDRAULIC_FLUX_COMPILED_VECTOR_NORMAL). */
SsStatus ss_hydraulic_flux_evaluate(const SsHydraulicFluxModel *model,
                                    const double *physical, double *normal_flux);
/** Evaluate a prescribed normal flux with the outward normal; pure.
 * @param[in] model Flux data.
 * @param[in] physical Point coordinates.
 * @param[in] outward_normal Outward unit normal; needed for vector data.
 * @param[out] normal_flux Normal flux.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_hydraulic_flux_evaluate_normal(const SsHydraulicFluxModel *model,
                                           const double *physical,
                                           const double *outward_normal,
                                           double *normal_flux);
/** Natural-boundary kernel of a prescribed flux; safe inside OpenMP regions.
 * @param[in] context ::SsHydraulicFluxModel.
 * @param[in] input Facet point.
 * @param[out] output Load at the point.
 * @return As ss_hydraulic_flux_evaluate_normal(). */
SsStatus ss_hydraulic_flux_point_kernel(const void *context,
                                        const SsNaturalPointInput *input,
                                        SsNaturalPointOutput *output);
/** Copy per-cell coefficients into an operator model sorted by stable cell.
 * @param[in] spec Model inputs.
 * @param[out] model New model, released with ss_hydraulic_operator_model_destroy().
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT, ::SS_ERROR_OVERFLOW or
 *   ::SS_ERROR_OUT_OF_MEMORY. */
SsStatus ss_hydraulic_operator_model_create(const SsHydraulicOperatorModelSpec *spec,
                                            SsHydraulicOperatorModel **model);
/** Select the gravity treatment; not thread-safe with concurrent kernels.
 * @param[in,out] model Operator model.
 * @param[in] mode Gravity treatment.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_hydraulic_operator_model_set_gravity_mode(SsHydraulicOperatorModel *model,
                                                      SsHydraulicGravityMode mode);
/** Ranges and identity of an operator model.
 * @param[in] model Operator model.
 * @param[out] summary Summary.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_hydraulic_operator_model_summary(const SsHydraulicOperatorModel *model,
                                             SsHydraulicOperatorModelSummary *summary);
/** Scalar assembly kernel with per-cell coefficients and the selected gravity
 * treatment; read-only and safe inside OpenMP regions.
 * @param[in] context ::SsHydraulicOperatorModel.
 * @param[in] input Point value, gradient, stable cell and quadrature index.
 * @param[out] output Flux and its derivatives.
 * @return ::SS_OK, ::SS_ERROR_NOT_FOUND for an unknown cell or point,
 *   ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_INTEGRITY. */
SsStatus ss_hydraulic_operator_point_kernel(const void *context,
                                            const SsScalarPointInput *input,
                                            SsScalarPointOutput *output);
/** Release an operator model.
 * @param[in,out] model Model pointer, set to NULL; `*model` may be NULL.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_hydraulic_operator_model_destroy(SsHydraulicOperatorModel **model);
/** Reduce the local flux contributions into the global balance.
 *
 * Collective on @p communicator: rank zero gathers five values per rank, sums
 * them in rank order and broadcasts the balance.
 *
 * @param[in] communicator Communicator of the ranks.
 * @param[in] local Rank-local contributions.
 * @param[in] physical_scale Positive lower bound of the normalization.
 * @param[in] normalized_tolerance Bound of the normalized imbalance.
 * @param[in] free_residual_tolerance Bound of the free residual norm.
 * @param[out] balance Global balance, identical on all ranks.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT, ::SS_ERROR_OUT_OF_MEMORY or
 *   ::SS_ERROR_INTERNAL. */
SsStatus ss_hydraulic_balance(MPI_Comm communicator,
                              const SsHydraulicBalanceLocal *local,
                              double physical_scale, double normalized_tolerance,
                              double free_residual_tolerance,
                              SsHydraulicBalance *balance);
/** Algebraic traits of the hydraulic operator.
 * @param[out] traits Traits.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_hydraulic_operator_traits(SsHydraulicOperatorTraits *traits);

#endif
