/** @file
 * @brief Core identifiers, checked size arithmetic, deterministic error
 * records and the callback interface of field transfer.
 *
 * All functions here are local to the calling rank and reentrant.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_CORE_CORE_H
#define SS_CORE_CORE_H

#include "slope_internal/core/sha256.h"
#include "slope_internal/core/status.h"

#include <hdf5.h>
#include <mpi.h>
#include <petscsystypes.h>

#include <stddef.h>
#include <stdint.h>

/** Capacity of an ::SsErrorRecord code. */
#define SS_ERROR_CODE_CAPACITY 64U
/** Capacity of an ::SsErrorRecord message. */
#define SS_ERROR_MESSAGE_CAPACITY 256U

/** Stable identity of a mesh entity, independent of partition and rank count. */
typedef uint64_t SsEntityId;
/** Identity of an accepted state. */
typedef uint64_t SsStateId;
/** Monotonic generation counter of mutable data. */
typedef uint64_t SsGeneration;
/** Identity of a registered model. */
typedef uint32_t SsModelId;
/** Identity of a registered rule. */
typedef uint32_t SsRuleId;
/** Identity of an output quantity. */
typedef uint32_t SsQuantityId;

/** Largest spatial dimension of a transfer point. */
#define SS_TRANSFER_MAX_DIMENSION 3U
/** Capacity of a transfer identity string. */
#define SS_TRANSFER_ID_CAPACITY 72U
/** Capacity of a transfer unit string. */
#define SS_TRANSFER_UNIT_CAPACITY 32U
/** Largest number of source candidates reported for one target point. */
#define SS_TRANSFER_MAX_CANDIDATES 8U

/** Mechanics quadrature point that receives a transferred field value. */
typedef struct {
  SsEntityId stable_point_id;       /**< Stable identity of the point. */
  SsEntityId stable_target_cell_id; /**< Stable identity of its target cell. */
  uint32_t quadrature_index;        /**< Quadrature index within the cell. */
  uint32_t dimension;               /**< Spatial dimension. */
  double physical[SS_TRANSFER_MAX_DIMENSION]; /**< Physical coordinates in m. */
  double physical_weight;         /**< Quadrature weight times the Jacobian. */
  double cell_epsilon;            /**< Hydraulic regularization width of the cell. */
  double saturated_unit_weight;   /**< Unit weight below the phreatic surface. */
  double unsaturated_unit_weight; /**< Unit weight above the phreatic surface. */
} SsTransferTargetPoint;

/** Source cell that may contain a target point. */
typedef struct {
  SsEntityId stable_source_cell_id;            /**< Stable identity of the cell. */
  double reference[SS_TRANSFER_MAX_DIMENSION]; /**< Reference coordinates. */
  double interior_margin; /**< Distance inside the cell; negative outside. */
  int owned;              /**< Nonzero when the calling rank owns the cell. */
  int ambiguous;          /**< Nonzero when the location is not unique. */
} SsTransferSourceCandidate;

/** Provenance of the source field of a transfer. */
typedef struct {
  uint64_t accepted_generation;   /**< Generation of the accepted source state. */
  uint64_t mesh_generation;       /**< Generation of the source mesh. */
  uint64_t coordinate_generation; /**< Generation of the source coordinates. */
  char mesh_identity[SS_TRANSFER_ID_CAPACITY];        /**< Source mesh identity. */
  char field_space_identity[SS_TRANSFER_ID_CAPACITY]; /**< Source space identity. */
  char model_identity[SS_TRANSFER_ID_CAPACITY];       /**< Source model identity. */
  char pressure_unit[SS_TRANSFER_UNIT_CAPACITY];      /**< Unit of the pressure. */
} SsTransferSourceState;

/** Number of local target points.
 * @param context Caller context of ::SsTransferGeometryOps.
 * @param count Number of local target points.
 * @return ::SS_OK or a failure status. */
typedef SsStatus (*SsTransferTargetCount)(void *context, size_t *count);
/** One local target point.
 * @param context Caller context of ::SsTransferGeometryOps.
 * @param index Index in `[0, count)`.
 * @param target Target point.
 * @return ::SS_OK or a failure status. */
typedef SsStatus (*SsTransferTargetAt)(void *context, size_t index,
                                       SsTransferTargetPoint *target);
/** Source cells of the calling rank that may contain a target point.
 * @param context Caller context of ::SsTransferGeometryOps.
 * @param target Target point.
 * @param candidates Up to ::SS_TRANSFER_MAX_CANDIDATES candidates.
 * @param candidate_count Number of candidates written.
 * @return ::SS_OK or a failure status. */
typedef SsStatus (*SsTransferLocate)(
    void *context, const SsTransferTargetPoint *target,
    SsTransferSourceCandidate candidates[SS_TRANSFER_MAX_CANDIDATES],
    size_t *candidate_count);
/** Evaluate the source pressure and its gradient in an owned source cell.
 * @param context Caller context of ::SsTransferGeometryOps.
 * @param stable_source_cell_id Owned source cell.
 * @param reference Reference coordinates in the cell.
 * @param pressure Pressure value.
 * @param gradient Physical pressure gradient.
 * @return ::SS_OK or a failure status. */
typedef SsStatus (*SsTransferSourceEvaluate)(
    void *context, SsEntityId stable_source_cell_id,
    const double reference[SS_TRANSFER_MAX_DIMENSION], double *pressure,
    double gradient[SS_TRANSFER_MAX_DIMENSION]);
/** Provenance of the source field.
 * @param context Caller context of ::SsTransferGeometryOps.
 * @param state Source provenance.
 * @return ::SS_OK or a failure status. */
typedef SsStatus (*SsTransferSourceStateRead)(void *context,
                                              SsTransferSourceState *state);

/** Callbacks through which a transfer plan reads target points and the
 * source field. None of the callbacks may call MPI. */
typedef struct {
  void *context;                            /**< Context passed to each callback. */
  SsTransferTargetCount target_count;       /**< Number of target points. */
  SsTransferTargetAt target_at;             /**< Target point access. */
  SsTransferLocate locate;                  /**< Source cell location. */
  SsTransferSourceEvaluate evaluate_source; /**< Source field evaluation. */
  SsTransferSourceStateRead source_state;   /**< Source provenance. */
} SsTransferGeometryOps;

/** Deterministic selection key of an error: smaller keys win. */
typedef struct {
  uint32_t phase;      /**< Phase of the operation, compared first. */
  uint32_t component;  /**< Component within the phase. */
  uint64_t entity;     /**< Stable entity, or zero. */
  uint64_t quadrature; /**< Quadrature point, or zero. */
  uint32_t thread;     /**< OpenMP thread that observed the error. */
  uint32_t rank;       /**< MPI rank that observed the error. */
} SsErrorKey;

/** One error with its selection key. */
typedef struct {
  int valid;                               /**< Nonzero when an error is recorded. */
  SsStatus status;                         /**< Status of the error. */
  SsErrorKey key;                          /**< Selection key. */
  char code[SS_ERROR_CODE_CAPACITY];       /**< Stable error code. */
  char message[SS_ERROR_MESSAGE_CAPACITY]; /**< Human-readable message. */
} SsErrorRecord;

/** Add two sizes without overflow.
 * @param[in] left First term.
 * @param[in] right Second term.
 * @param[out] result Sum.
 * @return ::SS_OK, ::SS_ERROR_OVERFLOW or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_checked_add_size(size_t left, size_t right, size_t *result);
/** Multiply two sizes without overflow.
 * @param[in] left First factor.
 * @param[in] right Second factor.
 * @param[out] result Product.
 * @return ::SS_OK, ::SS_ERROR_OVERFLOW or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_checked_mul_size(size_t left, size_t right, size_t *result);
/** Multiply several sizes without overflow.
 * @param[in] values @p count factors.
 * @param[in] count Number of factors; the empty product is one.
 * @param[out] result Product.
 * @return ::SS_OK, ::SS_ERROR_OVERFLOW or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_checked_product_size(const size_t *values, size_t count, size_t *result);
/** Round a size up to a multiple of a power of two.
 * @param[in] value Size to round.
 * @param[in] alignment Power-of-two alignment.
 * @param[out] result Rounded size.
 * @return ::SS_OK, ::SS_ERROR_OVERFLOW or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_checked_align_size(size_t value, size_t alignment, size_t *result);
/** Convert a 64-bit count to `size_t` without loss.
 * @param[in] value Count.
 * @param[out] result Converted count.
 * @return ::SS_OK, ::SS_ERROR_OVERFLOW or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_checked_u64_to_size(uint64_t value, size_t *result);
/** Convert a 64-bit count to an HDF5 size without loss.
 * @param[in] value Count.
 * @param[out] result Converted count.
 * @return ::SS_OK, ::SS_ERROR_OVERFLOW or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_checked_u64_to_hsize(uint64_t value, hsize_t *result);
/** Convert a 64-bit count to `PetscInt` without loss.
 * @param[in] value Count.
 * @param[out] result Converted count.
 * @return ::SS_OK, ::SS_ERROR_OVERFLOW or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_checked_u64_to_petsc(uint64_t value, PetscInt *result);
/** Convert a 64-bit count to `MPI_Count` without loss.
 * @param[in] value Count.
 * @param[out] result Converted count.
 * @return ::SS_OK, ::SS_ERROR_OVERFLOW or ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_checked_u64_to_mpi_count(uint64_t value, MPI_Count *result);

/** Initialize an error record with a status and a deterministic selection key.
 * @param[out] record Record; NULL is ignored.
 * @param[in] status Status of the error.
 * @param[in] key Selection key.
 * @param[in] code Error code, truncated to fit; NULL leaves it empty.
 * @param[in] message Message, truncated to fit; NULL leaves it empty. */
void ss_error_set(SsErrorRecord *record, SsStatus status, SsErrorKey key,
                  const char *code, const char *message);
/** Compare keys by phase, component, entity, quadrature, thread and rank.
 * @param[in] left First key.
 * @param[in] right Second key.
 * @return Negative, zero or positive as @p left orders before, equal to or
 *   after @p right. */
int ss_error_key_compare(const SsErrorKey *left, const SsErrorKey *right);
/** Order error records: valid before invalid, then by key and code.
 * @param[in] left First record or NULL.
 * @param[in] right Second record or NULL.
 * @return Negative, zero or positive as @p left orders before, equal to or
 *   after @p right. */
int ss_error_compare(const SsErrorRecord *left, const SsErrorRecord *right);
/** Stable name of a status.
 * @param[in] status Status to name.
 * @return Static lowercase name such as `invalid_argument`. */
const char *ss_status_name(SsStatus status);

#endif
