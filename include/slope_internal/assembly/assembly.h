/** @file
 * @brief Assembly plans, element kernels, and assembled and matrix-free
 * nonlinear operators for mechanics and scalar flux problems.
 *
 * Element work runs in a persistent OpenMP team; every MPI, PETSc and HDF5
 * call stays on the master thread outside the parallel regions. Each owned
 * row is written by one thread in a fixed order.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_ASSEMBLY_ASSEMBLY_H
#define SS_ASSEMBLY_ASSEMBLY_H

#include "slope_internal/core/core.h"
#include "slope_internal/fe/fe.h"
#include "slope_internal/material/material.h"
#include "slope_internal/mesh/mesh.h"

#include <mpi.h>
#include <petscmat.h>
#include <petscvec.h>

#include <stddef.h>
#include <stdint.h>

/** Capacity of an assembly identity string. */
#define SS_ASSEMBLY_ID_CAPACITY 72U

/** Opaque owned-row assembly plan; see ss_assembly_plan_create(). */
typedef struct SsAssemblyPlan SsAssemblyPlan;
/** Opaque nonlinear operator; see ss_operator_create(). */
typedef struct SsOperator SsOperator;

/** Physics of an operator. */
typedef enum {
  SS_ASSEMBLY_MECHANICS = 1,  /**< Vector mechanics with Mohr-Coulomb. */
  SS_ASSEMBLY_SCALAR_FLUX = 2 /**< Scalar flux problem (seepage). */
} SsAssemblyPhysics;

/** Quantities requested from an evaluation (bitwise OR). */
typedef enum {
  SS_ASSEMBLY_RESIDUAL = 1U << 0U,  /**< Residual vector. */
  SS_ASSEMBLY_JACOBIAN = 1U << 1U,  /**< Jacobian matrix. */
  SS_ASSEMBLY_POTENTIAL = 1U << 2U, /**< Energy functional. */
  SS_ASSEMBLY_FUSED = SS_ASSEMBLY_RESIDUAL | SS_ASSEMBLY_JACOBIAN /**< Both. */
} SsAssemblyRequest;

/** Summation order of element contributions. */
typedef enum {
  /** Fixed order of contributions within each owned row, independent of the
   * OpenMP team size (`validation.qualification_mode`). */
  SS_ASSEMBLY_QUALIFICATION = 1,
  /** Staged writes for speed; the order may depend on the team. */
  SS_ASSEMBLY_PERFORMANCE = 2
} SsAssemblyDeterminism;

/** Stage at which a test injects an assembly failure. */
typedef enum {
  SS_ASSEMBLY_FAIL_NONE = 0,                 /**< No failure. */
  SS_ASSEMBLY_FAIL_PACK = 1,                 /**< While packing cell data. */
  SS_ASSEMBLY_FAIL_WORKER = 2,               /**< In an OpenMP worker. */
  SS_ASSEMBLY_FAIL_NONFINITE = 3,            /**< A nonfinite contribution. */
  SS_ASSEMBLY_FAIL_CANCEL_AFTER_PACK = 4,    /**< Cancellation after packing. */
  SS_ASSEMBLY_FAIL_CANCEL_AFTER_COMPUTE = 5, /**< Cancellation after compute. */
  SS_ASSEMBLY_FAIL_PRECOMMIT = 6,            /**< Before committing values. */
  SS_ASSEMBLY_FAIL_ACTION = 7                /**< In a matrix-free action. */
} SsAssemblyFailurePoint;

/** Library service called by the assembly, reported to the service observer. */
typedef enum {
  SS_ASSEMBLY_SERVICE_MPI = 1,   /**< MPI call. */
  SS_ASSEMBLY_SERVICE_PETSC = 2, /**< PETSc call. */
  SS_ASSEMBLY_SERVICE_HDF5 = 3   /**< HDF5 call. */
} SsAssemblyService;

/** Size and identity of an assembly plan on this rank. */
typedef struct {
  uint64_t local_owned_rows;            /**< Owned rows. */
  uint64_t global_rows;                 /**< Global rows. */
  uint64_t local_incident_cells;        /**< Cells touching owned rows. */
  uint64_t local_owned_cells;           /**< Owned cells. */
  uint64_t local_halo_cells;            /**< Ghost cells. */
  uint64_t local_nonzeros;              /**< Nonzeros of the owned rows. */
  uint64_t local_diagonal_nonzeros;     /**< Nonzeros in the diagonal block. */
  uint64_t local_off_diagonal_nonzeros; /**< Nonzeros in the off-diagonal block. */
  uint64_t contribution_count;          /**< Cell-row contributions. */
  uint64_t plan_bytes;                  /**< Memory of the plan. */
  uint32_t overlap_depth;               /**< Ghost layers used. */
  size_t maximum_closure_count;         /**< Largest cell closure. */
  uint64_t pattern_generation;          /**< Generation of the pattern. */
  char pattern_digest[SS_ASSEMBLY_ID_CAPACITY]; /**< Global pattern digest. */
} SsAssemblyPlanSummary;

/** Description of one owned row. */
typedef struct {
  PetscInt global_row;        /**< Global row. */
  uint32_t component;         /**< Field component. */
  SsEntityId field_entity_id; /**< Stable entity carrying the unknown. */
  uint32_t field_entity_slot; /**< Slot of the unknown on the entity. */
  size_t column_count;        /**< Nonzero columns. */
  size_t incident_cell_count; /**< Cells contributing to the row. */
} SsAssemblyRowSummary;

/** Coupling body force at one mechanics quadrature point; called inside
 * OpenMP regions, so it must be read-only and thread-safe.
 * @param context Coupling context.
 * @param dimension Spatial dimension.
 * @param stable_cell_id Stable cell.
 * @param quadrature_index Quadrature index within the cell.
 * @param physical Physical coordinates.
 * @param body_force Body force.
 * @return ::SS_OK or a failure status. */
typedef SsStatus (*SsMechanicsCouplingPointKernel)(
    const void *context, uint32_t dimension, SsEntityId stable_cell_id,
    uint32_t quadrature_index, const double *physical, double *body_force);

/** Scaling of the point (coupling) body force. */
typedef enum {
  SS_MECHANICS_POINT_FORCE_FIXED = 0, /**< Applied unscaled. */
  SS_MECHANICS_POINT_FORCE_LOAD = 1   /**< Scaled by the external load factor. */
} SsMechanicsPointForceMode;

/** Inputs of one mechanics element evaluation; all pointers are borrowed. */
typedef struct {
  uint32_t dimension;                          /**< Spatial dimension. */
  const SsFeDescriptor *field_descriptor;      /**< Displacement element. */
  const SsFeDescriptor *coordinate_descriptor; /**< Geometry element. */
  const SsFeQuadrature *quadrature;            /**< Volume rule. */
  const double *coordinate_values;             /**< Node-major coordinates. */
  const double *field_values;                  /**< Node-major displacements. */
  const double *body_force;                    /**< Constant body force. */
  /** Constant coupling force added at every point, or NULL. */
  const double *coupling_body_force;
  const SsMcParameters *material; /**< Material of the cell. */
  double reduction_factor;        /**< Strength-reduction factor. */
  double external_load_factor;    /**< Load factor. */
  SsEntityId stable_cell_id;      /**< Stable cell. */
  /** Point coupling force, or NULL. */
  SsMechanicsCouplingPointKernel coupling_point_kernel;
  const void *coupling_point_context;         /**< Context of the point kernel. */
  SsMechanicsPointForceMode point_force_mode; /**< Scaling of the point force. */
} SsMechanicsElementInput;

/** State at one scalar quadrature point. */
typedef struct {
  uint32_t dimension;                   /**< Spatial dimension. */
  double physical[SS_FE_MAX_DIMENSION]; /**< Physical coordinates. */
  double value;                         /**< Field value. */
  double gradient[SS_FE_MAX_DIMENSION]; /**< Field gradient. */
  SsEntityId stable_cell_id;            /**< Stable cell. */
  uint32_t quadrature_index;            /**< Quadrature index within the cell. */
} SsScalarPointInput;

/** Constitutive response at one scalar quadrature point. */
typedef struct {
  double flux[SS_FE_MAX_DIMENSION];                  /**< Flux `q`. */
  double flux_derivative_value[SS_FE_MAX_DIMENSION]; /**< `dq / du`. */
  /** `dq / d grad u`, row-major. */
  double flux_derivative_gradient[SS_FE_MAX_DIMENSION * SS_FE_MAX_DIMENSION];
  double source;                  /**< Volume source. */
  double source_derivative_value; /**< Its derivative with respect to `u`. */
  double potential_density;       /**< Energy density, if any. */
} SsScalarPointOutput;

/** Scalar constitutive kernel; called inside OpenMP regions, so it must be
 * read-only and thread-safe.
 * @param context Kernel context.
 * @param input Point state.
 * @param output Point response.
 * @return ::SS_OK or a failure status. */
typedef SsStatus (*SsScalarPointKernel)(const void *context,
                                        const SsScalarPointInput *input,
                                        SsScalarPointOutput *output);

/** Inputs of one scalar element evaluation; all pointers are borrowed. */
typedef struct {
  uint32_t dimension;                          /**< Spatial dimension. */
  const SsFeDescriptor *field_descriptor;      /**< Field element. */
  const SsFeDescriptor *coordinate_descriptor; /**< Geometry element. */
  const SsFeQuadrature *quadrature;            /**< Volume rule. */
  const double *coordinate_values;             /**< Node-major coordinates. */
  const double *field_values;                  /**< Nodal field values. */
  SsScalarPointKernel point_kernel;            /**< Constitutive kernel. */
  const void *point_context;                   /**< Context of the kernel. */
  SsEntityId stable_cell_id;                   /**< Stable cell. */
} SsScalarElementInput;

/** Caller-owned element output buffers. */
typedef struct {
  size_t closure_capacity; /**< Capacity of @ref residual. */
  size_t matrix_capacity;  /**< Capacity of @ref jacobian. */
  double *residual;        /**< Element residual. */
  double *jacobian;        /**< Row-major element Jacobian. */
  double potential;        /**< Element energy. */
} SsElementOutput;

/** State at one natural-boundary quadrature point. */
typedef struct {
  uint32_t dimension;                         /**< Spatial dimension. */
  uint32_t components;                        /**< Field components. */
  double physical[SS_FE_MAX_DIMENSION];       /**< Physical coordinates. */
  double outward_normal[SS_FE_MAX_DIMENSION]; /**< Outward unit normal. */
  double value[SS_FE_MAX_DIMENSION];          /**< Field value. */
  SsEntityId stable_facet_id;                 /**< Stable facet. */
  uint32_t quadrature_index;                  /**< Quadrature index on the facet. */
} SsNaturalPointInput;

/** Boundary load at one natural-boundary quadrature point. */
typedef struct {
  double load[SS_FE_MAX_DIMENSION]; /**< Load density. */
  /** Derivative of the load with respect to the field, row-major. */
  double load_derivative[SS_FE_MAX_DIMENSION * SS_FE_MAX_DIMENSION];
  double signed_flux; /**< Signed normal flux density for balances. */
} SsNaturalPointOutput;

/** Natural-boundary kernel; called inside OpenMP regions, so it must be
 * read-only and thread-safe.
 * @param context Kernel context.
 * @param input Point state.
 * @param output Point load.
 * @return ::SS_OK or a failure status. */
typedef SsStatus (*SsNaturalPointKernel)(const void *context,
                                         const SsNaturalPointInput *input,
                                         SsNaturalPointOutput *output);

/** Inputs of one natural-boundary facet evaluation. */
typedef struct {
  const SsFieldFacetPack *facet;          /**< Facet data. */
  const SsFeDescriptor *field_descriptor; /**< Field element. */
  const SsFeQuadrature *quadrature;       /**< Facet rule. */
  const double *field_values;             /**< Nodal field values. */
  SsNaturalPointKernel point_kernel;      /**< Load kernel. */
  const void *point_context;              /**< Context of the kernel. */
} SsNaturalElementInput;

/** Material and body force of one material region. */
typedef struct {
  uint32_t material_value;                /**< Material label value of the region. */
  const SsMcParameters *parameters;       /**< Material parameters. */
  int has_body_force;                     /**< Nonzero when @ref body_force applies. */
  double body_force[SS_FE_MAX_DIMENSION]; /**< Region body force. */
} SsMechanicsMaterialRegion;

/** Mechanics part of an operator specification; borrowed. */
typedef struct {
  const SsMechanicsMaterialRegion *regions; /**< Material regions. */
  size_t region_count;                      /**< Number of regions. */
  double body_force[SS_FE_MAX_DIMENSION];   /**< Default body force. */
  /** Constant coupling force added everywhere. */
  double fixed_coupling_force[SS_FE_MAX_DIMENSION];
  /** Point coupling force, or NULL. */
  SsMechanicsCouplingPointKernel coupling_point_kernel;
  const void *coupling_point_context;         /**< Context of the point kernel. */
  SsMechanicsPointForceMode point_force_mode; /**< Scaling of the point force. */
} SsMechanicsOperatorSpec;

/** Scalar part of an operator specification; borrowed. */
typedef struct {
  SsScalarPointKernel point_kernel; /**< Constitutive kernel. */
  const void *point_context;        /**< Context of the kernel. */
} SsScalarOperatorSpec;

/** One natural-boundary term of an operator; borrowed. */
typedef struct {
  uint32_t support_value;            /**< Support label of the facets. */
  SsNaturalPointKernel point_kernel; /**< Load kernel. */
  const void *point_context;         /**< Context of the kernel. */
  int scaled_by_external_load;       /**< 1 scales the load by the load factor. */
} SsNaturalOperatorSpec;

/** Operator specification; borrowed until ss_operator_destroy(). */
typedef struct {
  SsAssemblyPhysics physics;         /**< Physics. */
  SsAssemblyDeterminism determinism; /**< Summation order. */
  size_t memory_cap_bytes;           /**< Admission cap of the operator workspace. */
  uint64_t openmp_work_threshold;    /**< Smallest cell count that uses OpenMP. */
  int requested_threads;             /**< OpenMP threads. */
  const SsMechanicsOperatorSpec *mechanics;        /**< Mechanics data, or NULL. */
  const SsScalarOperatorSpec *scalar;              /**< Scalar data, or NULL. */
  const SsNaturalOperatorSpec *natural_operations; /**< Boundary terms. */
  size_t natural_operation_count;                  /**< Number of boundary terms. */
  int matrix_free; /**< Nonzero for a matrix-free tangent. */
  /** Fine-level matrix-free block size override; zero uses the default. */
  size_t matrix_free_block_nodes;
  /** Fine-level matrix-free batch override; zero uses the default. */
  size_t matrix_free_batch_cells;
  /** Registered volume-rule policy of the fused operator: 0 or
   * ::SS_FE_QUADRATURE_MODE_DEFAULT selects the planned default rule,
   * ::SS_FE_QUADRATURE_MODE_PAIRED_HIGHER its cheapest higher-degree control.
   * Coupling targets and outputs must request the same mode. */
  SsFeQuadratureMode volume_quadrature_mode;
} SsOperatorSpec;

/** State at which an operator is evaluated. */
typedef struct {
  Vec field;                   /**< Global field vector. */
  double reduction_factor;     /**< Strength-reduction factor. */
  double external_load_factor; /**< Load factor. */
  unsigned int cancelled;      /**< Nonzero requests cancellation. */
} SsOperatorState;

/** Algebraic properties and generations of an operator. */
typedef struct {
  SsAssemblyPhysics physics;         /**< Physics. */
  SsAssemblyDeterminism determinism; /**< Summation order. */
  uint32_t dimension;                /**< Spatial dimension. */
  uint32_t components;               /**< Field components. */
  uint64_t local_rows;               /**< Owned rows. */
  uint64_t global_rows;              /**< Global rows. */
  int linear;                        /**< Nonzero for a linear operator. */
  int state_dependent;               /**< Nonzero when values depend on the state. */
  int symmetric;                     /**< Nonzero for a symmetric Jacobian. */
  int positive_definite;             /**< Nonzero for a positive-definite Jacobian. */
  uint32_t supported_requests;       /**< Supported ::SsAssemblyRequest bits. */
  uint64_t pattern_generation;       /**< Generation of the pattern. */
  uint64_t value_generation;         /**< Generation of the last values. */
  uint64_t coordinate_generation;    /**< Generation of the coordinates. */
  uint64_t constraint_generation;    /**< Generation of the constraints. */
  char pattern_digest[SS_ASSEMBLY_ID_CAPACITY]; /**< Pattern digest. */
} SsOperatorProperties;

/** Counters and memory of an operator on this rank. */
typedef struct {
  uint64_t assembly_calls;     /**< Evaluations. */
  uint64_t successful_commits; /**< Evaluations that committed values. */
  uint64_t value_generation;   /**< Generation of the last values. */
  uint64_t packed_bytes;       /**< Packed cell data. */
  uint64_t shadow_bytes;       /**< Ordered shadow buffers (qualification). */
  uint64_t scratch_peak_bytes; /**< Peak per-thread scratch. */
  uint64_t matrix_free_bytes;  /**< Matrix-free quadrature data. */
  /** Per-point geometry and reference tables built at setup (rank-local). */
  uint64_t geometry_cache_bytes;
  /** Volume quadrature points over the rank-local cells. */
  uint64_t quadrature_points;
  /** Identifier of the volume rule shared by every local cell ("" if none). */
  char volume_quadrature_id[SS_ASSEMBLY_ID_CAPACITY];
  uint64_t admitted_peak_bytes;                 /**< Admitted peak memory. */
  uint64_t tile_capacity;                       /**< Cells per work tile. */
  uint64_t tile_count;                          /**< Work tiles. */
  uint64_t cell_evaluations;                    /**< Cell evaluations. */
  uint64_t duplicate_cell_evaluations;          /**< Ghost-cell evaluations. */
  uint64_t natural_facet_evaluations;           /**< Facet evaluations. */
  uint64_t duplicate_natural_facet_evaluations; /**< Ghost-facet evaluations. */
  uint64_t completed_rows;                      /**< Rows completed (qualification). */
  uint64_t row_writer_violations;         /**< Rows written by the wrong thread. */
  uint64_t worker_service_calls;          /**< MPI/PETSc/HDF5 calls from workers (0). */
  uint64_t parallel_pack_calls;           /**< Parallel packing passes. */
  uint64_t parallel_stage_calls;          /**< Parallel staging passes. */
  uint64_t parallel_stage_entities;       /**< Entities staged in parallel. */
  uint64_t petsc_new_nonzero_allocations; /**< New nonzeros PETSc allocated (0). */
  uint64_t branch_elastic;                /**< Points on the elastic branch. */
  uint64_t branch_smooth;                 /**< Points on the smooth branch. */
  uint64_t branch_left_edge;              /**< Points on the left edge. */
  uint64_t branch_right_edge;             /**< Points on the right edge. */
  uint64_t branch_apex;                   /**< Points at the apex. */
  double natural_signed_flux;             /**< Sum of the signed boundary flux. */
  int requested_threads;                  /**< Requested OpenMP threads. */
  int observed_threads;                   /**< Observed OpenMP team size. */
  int used_openmp;                        /**< Nonzero when OpenMP was used. */
  uint64_t unique_worker_cpus;            /**< Distinct CPUs of the workers. */
  uint64_t unique_physical_cores; /**< Distinct physical cores of the workers. */
  char worker_cpus[SS_ERROR_MESSAGE_CAPACITY];  /**< CPU list of the workers. */
  char pattern_digest[SS_ASSEMBLY_ID_CAPACITY]; /**< Pattern digest. */
} SsAssemblyDiagnostics;

/** Observer of library calls made during assembly (test hook).
 * @param service Library service.
 * @param in_parallel Nonzero when called inside an OpenMP region.
 * @param thread OpenMP thread number.
 * @param context Observer context. */
typedef void (*SsAssemblyServiceObserver)(SsAssemblyService service, int in_parallel,
                                          int thread, void *context);

/** Build the owned-row plan and assembled sparsity of a field space;
 * collective.
 * @param[in] communicator Communicator of the field space.
 * @param[in] field_space Field space; borrowed for the plan's lifetime.
 * @param[out] plan New plan, released with ss_assembly_plan_destroy().
 * @param[out] error Collectively selected error record.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_assembly_plan_create(MPI_Comm communicator, const SsFieldSpace *field_space,
                                 SsAssemblyPlan **plan, SsErrorRecord *error);
/** Build a plan for matrix-free operators, without the assembled sparsity;
 * collective.
 * @param[in] communicator Communicator of the field space.
 * @param[in] field_space Field space; borrowed for the plan's lifetime.
 * @param[out] plan New plan, released with ss_assembly_plan_destroy().
 * @param[out] error Collectively selected error record.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_assembly_plan_create_matrix_free(MPI_Comm communicator,
                                             const SsFieldSpace *field_space,
                                             SsAssemblyPlan **plan,
                                             SsErrorRecord *error);
/** Size and identity of a plan; local.
 * @param[in] plan Plan.
 * @param[out] summary Summary.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_assembly_plan_summary(const SsAssemblyPlan *plan,
                                  SsAssemblyPlanSummary *summary);
/** Columns and incident cells of one owned row; local.
 * @param[in] plan Plan.
 * @param[in] local_row Owned row index.
 * @param[out] summary Row description.
 * @param[out] columns Global columns, or NULL.
 * @param[in] column_capacity Capacity of @p columns.
 * @param[out] column_count Number of columns.
 * @param[out] incident_cells Stable incident cells, or NULL.
 * @param[in] cell_capacity Capacity of @p incident_cells.
 * @param[out] cell_count Number of incident cells.
 * @return ::SS_OK, ::SS_ERROR_LIMIT when a capacity is too small, or
 *   ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_assembly_plan_row(const SsAssemblyPlan *plan, size_t local_row,
                              SsAssemblyRowSummary *summary, PetscInt *columns,
                              size_t column_capacity, size_t *column_count,
                              SsEntityId *incident_cells, size_t cell_capacity,
                              size_t *cell_count);
/** Create a vector with the plan's row layout; collective.
 * @param[in] plan Plan.
 * @param[out] vector New vector, owned by the caller.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_INTERNAL. */
SsStatus ss_assembly_plan_create_vector(const SsAssemblyPlan *plan, Vec *vector);
/** Create a preallocated MPIAIJ matrix with the plan's pattern; collective.
 * New nonzero allocations are errors.
 * @param[in] plan Assembled plan.
 * @param[out] matrix New matrix, owned by the caller.
 * @return ::SS_OK, ::SS_ERROR_INVALID_STATE for a matrix-free plan, or a
 *   failure status. */
SsStatus ss_assembly_plan_create_matrix(const SsAssemblyPlan *plan, Mat *matrix);
/** Release a plan; local.
 * @param[in,out] plan Plan pointer, set to NULL; `*plan` may be NULL.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_assembly_plan_destroy(SsAssemblyPlan **plan);

/** Evaluate one mechanics element; pure and safe inside OpenMP regions.
 * @param[in] input Element inputs.
 * @param[in] request Bitwise OR of ::SsAssemblyRequest values.
 * @param[in,out] output Caller-owned output buffers.
 * @param[out] error Error record of a failing point.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT, ::SS_ERROR_INTEGRITY for a
 *   nonfinite or invalid point, or a material status. */
SsStatus ss_assembly_mechanics_element(const SsMechanicsElementInput *input,
                                       uint32_t request, SsElementOutput *output,
                                       SsErrorRecord *error);
/** Evaluate one scalar element; pure and safe inside OpenMP regions.
 * @param[in] input Element inputs.
 * @param[in] request Bitwise OR of ::SsAssemblyRequest values.
 * @param[in,out] output Caller-owned output buffers.
 * @param[out] error Error record of a failing point.
 * @return ::SS_OK or a failure status. */
SsStatus ss_assembly_scalar_element(const SsScalarElementInput *input, uint32_t request,
                                    SsElementOutput *output, SsErrorRecord *error);
/** Evaluate one natural-boundary facet; pure and safe inside OpenMP regions.
 * @param[in] input Facet inputs.
 * @param[in] request Bitwise OR of ::SsAssemblyRequest values.
 * @param[in,out] output Caller-owned output buffers.
 * @param[out] signed_flux Integrated signed flux of the facet.
 * @param[out] error Error record of a failing point.
 * @return ::SS_OK or a failure status. */
SsStatus ss_assembly_natural_element(const SsNaturalElementInput *input,
                                     uint32_t request, SsElementOutput *output,
                                     double *signed_flux, SsErrorRecord *error);

/** Create an operator over a plan; collective.
 *
 * Packs the cell data, admits the workspace against the memory cap and
 * creates the ghost scatter.
 *
 * @param[in] plan Plan; borrowed for the operator's lifetime.
 * @param[in] spec Specification; borrowed for the operator's lifetime.
 * @param[out] operator_handle New operator, released with ss_operator_destroy().
 * @param[out] error Collectively selected error record.
 * @return ::SS_OK or the agreed failure status (including
 *   ::SS_ERROR_RESOURCE above the memory cap). */
SsStatus ss_operator_create(const SsAssemblyPlan *plan, const SsOperatorSpec *spec,
                            SsOperator **operator_handle, SsErrorRecord *error);
/** Properties and generations of an operator; local.
 * @param[in] operator_handle Operator.
 * @param[out] properties Properties.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_operator_properties(const SsOperator *operator_handle,
                                SsOperatorProperties *properties);
/** Counters and memory of an operator; local.
 * @param[in] operator_handle Operator.
 * @param[out] diagnostics Diagnostics.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_operator_diagnostics(const SsOperator *operator_handle,
                                 SsAssemblyDiagnostics *diagnostics);
/** Assemble the residual; collective. The element work runs in the OpenMP
 * team; all MPI and PETSc calls stay on the master thread.
 * @param[in,out] operator_handle Operator.
 * @param[in] state Evaluation state.
 * @param[out] residual Residual vector with the plan's layout.
 * @param[out] error Collectively selected error record.
 * @return ::SS_OK, ::SS_ERROR_CANCELLED or the agreed failure status. */
SsStatus ss_operator_residual(SsOperator *operator_handle, const SsOperatorState *state,
                              Vec residual, SsErrorRecord *error);
/** Assemble the Jacobian; collective.
 * @param[in,out] operator_handle Assembled operator.
 * @param[in] state Evaluation state.
 * @param[in,out] jacobian Matrix from ss_assembly_plan_create_matrix().
 * @param[out] error Collectively selected error record.
 * @return As ss_operator_residual(). */
SsStatus ss_operator_jacobian(SsOperator *operator_handle, const SsOperatorState *state,
                              Mat jacobian, SsErrorRecord *error);
/** Assemble the residual and the Jacobian in one pass; collective.
 * @param[in,out] operator_handle Assembled operator.
 * @param[in] state Evaluation state.
 * @param[out] residual Residual vector.
 * @param[in,out] jacobian Jacobian matrix.
 * @param[out] error Collectively selected error record.
 * @return As ss_operator_residual(). */
SsStatus ss_operator_residual_jacobian(SsOperator *operator_handle,
                                       const SsOperatorState *state, Vec residual,
                                       Mat jacobian, SsErrorRecord *error);
/** Assemble the residual and the global energy functional; collective.
 * @param[in,out] operator_handle Operator.
 * @param[in] state Evaluation state.
 * @param[out] residual Residual vector.
 * @param[out] functional Global functional, identical on all ranks.
 * @param[out] error Collectively selected error record.
 * @return As ss_operator_residual(). */
SsStatus ss_operator_residual_functional(SsOperator *operator_handle,
                                         const SsOperatorState *state, Vec residual,
                                         double *functional, SsErrorRecord *error);
/** Assemble the residual, the Jacobian and the functional; collective.
 * @param[in,out] operator_handle Assembled operator.
 * @param[in] state Evaluation state.
 * @param[out] residual Residual vector.
 * @param[in,out] jacobian Jacobian matrix.
 * @param[out] functional Global functional.
 * @param[out] error Collectively selected error record.
 * @return As ss_operator_residual(). */
SsStatus ss_operator_residual_jacobian_functional(SsOperator *operator_handle,
                                                  const SsOperatorState *state,
                                                  Vec residual, Mat jacobian,
                                                  double *functional,
                                                  SsErrorRecord *error);
/** Evaluate the global energy functional; collective.
 * @param[in,out] operator_handle Operator.
 * @param[in] state Evaluation state.
 * @param[out] functional Global functional.
 * @param[out] error Collectively selected error record.
 * @return As ss_operator_residual(). */
SsStatus ss_operator_functional(SsOperator *operator_handle,
                                const SsOperatorState *state, double *functional,
                                SsErrorRecord *error);
/**
 * Create the mechanics tangent shell owned by an operator; collective.
 *
 * The shell applies the quadrature-point tangent directly and has no
 * assembled fine-grid storage. ss_operator_matrix_free_linearize() must
 * succeed before the first MatMult, MatMultTranspose or MatGetDiagonal.
 *
 * @param[in,out] operator_handle Matrix-free mechanics operator.
 * @param[out] tangent New shell; the caller destroys it before the operator.
 * @param[out] error Collectively selected error record.
 * @return ::SS_OK or the agreed failure status.
 */
SsStatus ss_operator_matrix_free_create(SsOperator *operator_handle, Mat *tangent,
                                        SsErrorRecord *error);
/** Store the quadrature-point tangents at a state, blended with the elastic
 * tangent; collective, with the point work in the OpenMP team.
 * @param[in,out] operator_handle Matrix-free mechanics operator.
 * @param[in] state Linearization state.
 * @param[in] elastic_regularization Blend `r` of `(1 - r) K + r K_e`.
 * @param[out] error Collectively selected error record.
 * @return ::SS_OK, ::SS_ERROR_CANCELLED or the agreed failure status. */
SsStatus ss_operator_matrix_free_linearize(SsOperator *operator_handle,
                                           const SsOperatorState *state,
                                           double elastic_regularization,
                                           SsErrorRecord *error);
/** Consume the rank-local failure flag of a native matrix-free shell.
 *
 * The caller agrees on the status at the next solver boundary, including
 * ranks with zero owned entries.
 *
 * @param[in] matrix Matrix-free shell or another PETSc matrix.
 * @return Nonzero when an action failed on this rank since the last call;
 *   zero for other matrix types. */
int ss_operator_matrix_free_take_action_failure(Mat matrix);
/** Assemble the algebraic P1 projection of a linearized high-order shell;
 * collective, with the cell work in the OpenMP team.
 * @param[in] fine_operator Linearized high-order matrix-free operator.
 * @param[in] p1_plan Assembled P1 plan on the same mesh.
 * @param[in,out] p1_jacobian Matrix of @p p1_plan.
 * @param[out] error Collectively selected error record.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_operator_matrix_free_project_p1_jacobian(const SsOperator *fine_operator,
                                                     const SsAssemblyPlan *p1_plan,
                                                     Mat p1_jacobian,
                                                     SsErrorRecord *error);
/** Refresh a coarse shell's Jacobi data from the exact Galerkin projection;
 * collective.
 * @param[in] fine_operator Linearized fine matrix-free operator.
 * @param[in,out] coarse_operator Coarse matrix-free operator.
 * @param[out] error Collectively selected error record.
 * @return ::SS_OK or the agreed failure status. */
SsStatus ss_operator_matrix_free_project_coarse(const SsOperator *fine_operator,
                                                SsOperator *coarse_operator,
                                                SsErrorRecord *error);
/** Inject a failure at a stage of the next evaluation (test hook); local.
 * @param[in,out] operator_handle Operator.
 * @param[in] point Stage of the failure.
 * @param[in] rank Rank that fails.
 * @param[in] stable_entity_id Entity at which it fails, or 0.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_operator_inject_failure(SsOperator *operator_handle,
                                    SsAssemblyFailurePoint point, int rank,
                                    SsEntityId stable_entity_id);
/** Report every MPI, PETSc and HDF5 call of the assembly to an observer
 * (test hook); local.
 * @param[in,out] operator_handle Operator.
 * @param[in] observer Observer, or NULL to clear it.
 * @param[in] context Observer context.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_operator_set_service_observer(SsOperator *operator_handle,
                                          SsAssemblyServiceObserver observer,
                                          void *context);
/** Destroy an operator and its PETSc objects; collective.
 * @param[in,out] operator_handle Operator pointer, set to NULL; may point to
 *   NULL.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_INTERNAL. */
SsStatus ss_operator_destroy(SsOperator **operator_handle);

#endif
