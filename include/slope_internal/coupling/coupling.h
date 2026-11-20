/** @file
 * @brief One-way coupling of steady seepage into mechanics.
 *
 * A transfer plan evaluates the seepage pressure and its gradient at the
 * mechanics quadrature points, and an immutable snapshot freezes the
 * resulting body forces for the whole mechanics analysis.
 *
 * The engine always uses ::SS_COUPLING_TRANSFER_SAME_MESH. The
 * ::SS_COUPLING_TRANSFER_DIFFERENT_MESH geometric search is a reference
 * implementation exercised only by the tests: it performs world collectives
 * for every target point and does not scale.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_COUPLING_COUPLING_H
#define SS_COUPLING_COUPLING_H

#include "slope_internal/core/core.h"
#include "slope_internal/core/status.h"

#include <mpi.h>

#include <stddef.h>
#include <stdint.h>

/** Capacity of a coupling identity string. */
#define SS_COUPLING_ID_CAPACITY 72U

/** Relation between the seepage and the mechanics mesh. */
typedef enum {
  /** Both fields live on the same mesh; each target cell is its source cell. */
  SS_COUPLING_TRANSFER_SAME_MESH = 1,
  /** Geometric search on an unrelated mesh; test and reference use only, with
   * O(targets) world collectives. */
  SS_COUPLING_TRANSFER_DIFFERENT_MESH = 2
} SsCouplingTransferType;

/** Loads that seepage applies to mechanics. */
typedef enum {
  /** Seepage force `-grad p` plus the saturation-dependent unit weight. */
  SS_COUPLING_PRESSURE_GRADIENT_AND_WEIGHT = 1,
  /** Saturation-dependent unit weight only. */
  SS_COUPLING_WEIGHT_ONLY = 2
} SsCouplingMode;

/** Resolve a known same-mesh cell to its pressure owner and reference point.
 *
 * This local callback uses the target's stable cell identity and must not
 * call MPI. Owners are communicator ranks; only the owner evaluates the
 * pressure.
 *
 * @param context Caller context of the geometry callbacks.
 * @param target Target point.
 * @param source_rank Rank that owns the source cell.
 * @param reference Reference coordinates of the point in the source cell.
 * @return ::SS_OK or a failure status. */
typedef SsStatus (*SsCouplingSameMeshMap)(void *context,
                                          const SsTransferTargetPoint *target,
                                          int *source_rank,
                                          double reference[SS_TRANSFER_MAX_DIMENSION]);

/** Configuration of a transfer plan. */
typedef struct {
  uint32_t dimension;                   /**< Spatial dimension, 2 or 3. */
  uint32_t upward_axis;                 /**< Index of the vertical axis. */
  SsCouplingTransferType transfer_type; /**< Mesh relation. */
  int requested_threads;                /**< OpenMP threads for evaluation. */
  /** Minimum local point count for which evaluation uses OpenMP. */
  uint64_t openmp_work_threshold;
  /** Identity of the mechanics mesh. */
  char target_mesh_identity[SS_COUPLING_ID_CAPACITY];
  /** Identity of the mechanics field space. */
  char target_space_identity[SS_COUPLING_ID_CAPACITY];
  /** Identity of the mechanics quadrature. */
  char target_quadrature_identity[SS_COUPLING_ID_CAPACITY];
  SsTransferGeometryOps geometry; /**< Target and source callbacks. */
  /** Optional ownership map for same-mesh transfer. Without a map, both
   * spaces must share cell ownership and locate() runs on the target rank. */
  SsCouplingSameMeshMap same_mesh_map;
} SsCouplingPlanSpec;

/** Size, provenance and memory of a transfer plan. */
typedef struct {
  uint32_t dimension;                    /**< Spatial dimension. */
  SsCouplingTransferType transfer_type;  /**< Mesh relation. */
  uint64_t local_target_points;          /**< Target points on this rank. */
  uint64_t global_target_points;         /**< Target points on all ranks. */
  uint64_t local_target_cells;           /**< Target cells on this rank. */
  uint64_t global_target_cells;          /**< Target cells on all ranks. */
  uint64_t resident_target_points;       /**< Points stored by this rank. */
  uint64_t unresolved_points;            /**< Points without a source cell. */
  uint64_t ambiguous_points;             /**< Points with several source cells. */
  uint64_t source_mesh_generation;       /**< Generation of the source mesh. */
  uint64_t source_coordinate_generation; /**< Generation of the coordinates. */
  size_t resident_bytes;                 /**< Bytes held by the plan on this rank. */
  int requested_threads;                 /**< OpenMP threads for evaluation. */
  char transfer_identity[SS_COUPLING_ID_CAPACITY]; /**< Identity of the plan. */
} SsCouplingPlanSummary;

/** Read-only view of a snapshot, borrowed from it. Point arrays are sorted by
 * stable cell and quadrature index; cell arrays by stable cell. */
typedef struct {
  size_t point_count;                  /**< Local target points. */
  size_t cell_count;                   /**< Local target cells. */
  uint32_t dimension;                  /**< Spatial dimension. */
  SsCouplingMode mode;                 /**< Loads applied to mechanics. */
  const SsEntityId *point_ids;         /**< Stable identity of each point. */
  const SsEntityId *cell_ids_by_point; /**< Stable cell of each point. */
  const uint32_t *quadrature_indices;  /**< Quadrature index of each point. */
  const double *pressure;              /**< Pore pressure at each point. */
  const double *gradient;              /**< Pressure gradient, point-major. */
  const double *body_force;            /**< Mechanics body force, point-major. */
  const SsEntityId *cell_ids;          /**< Stable identity of each cell. */
  const double *cell_average_pressure; /**< Weighted mean pressure of each cell. */
  const double *cell_epsilon;          /**< Hydraulic regularization width. */
  const double *selected_unit_weight;  /**< Unit weight chosen for each cell. */
  /** Nonzero for cells whose mean pressure reaches a tenth of the cell's
   * regularization width. */
  const uint8_t *saturated;
  uint64_t source_generation; /**< Accepted generation of the source field. */
  char source_identity[SS_COUPLING_ID_CAPACITY];       /**< Source field. */
  char source_mesh_identity[SS_COUPLING_ID_CAPACITY];  /**< Source mesh. */
  char source_space_identity[SS_COUPLING_ID_CAPACITY]; /**< Source field space. */
  char source_model_identity[SS_COUPLING_ID_CAPACITY]; /**< Source model. */
  char pressure_unit[SS_COUPLING_ID_CAPACITY];         /**< Pressure unit. */
  char target_mesh_identity[SS_COUPLING_ID_CAPACITY];  /**< Mechanics mesh. */
  char target_space_identity[SS_COUPLING_ID_CAPACITY]; /**< Mechanics space. */
  /** Identity of the mechanics quadrature. */
  char target_quadrature_identity[SS_COUPLING_ID_CAPACITY];
  char transfer_identity[SS_COUPLING_ID_CAPACITY]; /**< Transfer plan. */
  char snapshot_identity[SS_COUPLING_ID_CAPACITY]; /**< This snapshot. */
} SsCouplingSnapshotView;

/** Global diagnostics of a snapshot, identical on all ranks except the
 * `local_*` members. */
typedef struct {
  uint64_t local_target_points;  /**< Target points on this rank. */
  uint64_t global_target_points; /**< Target points on all ranks. */
  uint64_t local_target_cells;   /**< Target cells on this rank. */
  uint64_t global_target_cells;  /**< Target cells on all ranks. */
  uint64_t source_evaluations;   /**< Source evaluations performed. */
  uint64_t saturated_cells;      /**< Cells with the saturated weight. */
  uint64_t unsaturated_cells;    /**< Cells with the unsaturated weight. */
  uint64_t unresolved_points;    /**< Points without a source cell. */
  uint64_t ambiguous_points;     /**< Points with several source cells. */
  size_t snapshot_bytes;         /**< Bytes held by the snapshot on this rank. */
  double pressure_minimum;       /**< Smallest pressure. */
  double pressure_maximum;       /**< Largest pressure. */
  double gradient_norm_maximum;  /**< Largest gradient norm. */
  int requested_threads;         /**< Requested OpenMP threads. */
  int observed_threads;          /**< Largest observed OpenMP team. */
  int used_openmp;               /**< Nonzero when a rank used OpenMP. */
  char content_digest[SS_COUPLING_ID_CAPACITY]; /**< Digest of the content. */
} SsCouplingDiagnostics;

/** Opaque transfer plan; see ss_coupling_plan_create(). */
typedef struct SsCouplingTransferPlan SsCouplingTransferPlan;
/** Opaque immutable snapshot; see ss_coupling_field_snapshot_build(). */
typedef struct SsCouplingFieldSnapshot SsCouplingFieldSnapshot;

/** Body force of the effective unit weight and, optionally, seepage.
 *
 * The force is `-grad p` (for ::SS_COUPLING_PRESSURE_GRADIENT_AND_WEIGHT)
 * minus the unit weight along the upward axis. Pure; safe inside OpenMP
 * regions.
 *
 * @param[in] dimension Spatial dimension, 2 or 3.
 * @param[in] upward_axis Index of the vertical axis.
 * @param[in] mode Coupled loads.
 * @param[in] pressure_gradient Pressure gradient, @p dimension values.
 * @param[in] selected_unit_weight Positive unit weight.
 * @param[out] body_force Body force, @p dimension values.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_INTEGRITY for a
 *   nonfinite result. */
SsStatus ss_coupling_body_force(uint32_t dimension, uint32_t upward_axis,
                                SsCouplingMode mode, const double *pressure_gradient,
                                double selected_unit_weight, double *body_force);
/** Resolve the source cell and reference point of every target point.
 *
 * Collective on @p communicator, which the plan duplicates. Same-mesh plans
 * use one batched exchange; different-mesh plans use per-target collectives
 * (tests only).
 *
 * @param[in] communicator Communicator of the ranks.
 * @param[in] spec Configuration; the callbacks are borrowed for the plan's
 *   lifetime.
 * @param[out] plan New plan, released with ss_coupling_plan_destroy().
 * @return ::SS_OK or a failure agreed on all ranks: ::SS_ERROR_INVALID_ARGUMENT,
 *   ::SS_ERROR_RESOURCE, ::SS_ERROR_NOT_FOUND for an unresolved point,
 *   ::SS_ERROR_DUPLICATE for an ambiguous one, or another callback status. */
SsStatus ss_coupling_plan_create(MPI_Comm communicator, const SsCouplingPlanSpec *spec,
                                 SsCouplingTransferPlan **plan);
/** Size, provenance and memory of a transfer plan; local.
 * @param[in] plan Plan.
 * @param[out] summary Plan summary.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT or ::SS_ERROR_OVERFLOW. */
SsStatus ss_coupling_plan_summary(const SsCouplingTransferPlan *plan,
                                  SsCouplingPlanSummary *summary);
/** Destroy a transfer plan and free its communicator; collective.
 * @param[in,out] plan Plan pointer, set to NULL; `*plan` may be NULL.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_coupling_plan_destroy(SsCouplingTransferPlan **plan);
/** Evaluate the source pressure at all targets and freeze the mechanics loads.
 *
 * Collective on the plan's communicator. Local evaluation uses an OpenMP team
 * above the plan's work threshold; remote values are exchanged in one batch.
 *
 * @param[in] plan Transfer plan; it may be destroyed after this call.
 * @param[in] mode Coupled loads.
 * @param[out] snapshot New immutable snapshot, released with
 *   ss_coupling_field_snapshot_destroy().
 * @return ::SS_OK or a failure agreed on all ranks. */
SsStatus ss_coupling_field_snapshot_build(const SsCouplingTransferPlan *plan,
                                          SsCouplingMode mode,
                                          SsCouplingFieldSnapshot **snapshot);
/** Borrow the point and cell data of a snapshot; local.
 * @param[in] snapshot Snapshot.
 * @param[out] view View valid until the snapshot is destroyed.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_coupling_field_snapshot_view(const SsCouplingFieldSnapshot *snapshot,
                                         SsCouplingSnapshotView *view);
/** Copy the snapshot diagnostics and content digest; local.
 * @param[in] snapshot Snapshot.
 * @param[out] diagnostics Diagnostics.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_coupling_field_snapshot_diagnostics(const SsCouplingFieldSnapshot *snapshot,
                                                SsCouplingDiagnostics *diagnostics);
/** Look up a frozen body force by stable cell and quadrature index.
 *
 * The signature matches the mechanics body-force callback. Read-only binary
 * search; safe inside OpenMP regions.
 *
 * @param[in] snapshot ::SsCouplingFieldSnapshot passed as callback context.
 * @param[in] dimension Spatial dimension of the snapshot.
 * @param[in] stable_cell_id Stable cell of the point.
 * @param[in] quadrature_index Quadrature index within the cell.
 * @param[in] physical Physical coordinates; checked for finiteness only.
 * @param[out] body_force Frozen body force, @p dimension values.
 * @return ::SS_OK, ::SS_ERROR_NOT_FOUND or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_coupling_field_snapshot_mechanics_force(
    const void *snapshot, uint32_t dimension, SsEntityId stable_cell_id,
    uint32_t quadrature_index, const double *physical, double *body_force);
/** Release the local storage of a snapshot; local.
 * @param[in,out] snapshot Snapshot pointer, set to NULL; `*snapshot` may be NULL.
 * @return ::SS_OK or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_coupling_field_snapshot_destroy(SsCouplingFieldSnapshot **snapshot);

#endif
