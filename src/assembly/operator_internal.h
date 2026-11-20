/* Private state and interfaces shared by assembly operator modules.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_ASSEMBLY_OPERATOR_INTERNAL_H
#define SS_ASSEMBLY_OPERATOR_INTERNAL_H

#include "assembly_impl.h"
#include "slope_internal/build/config.h"

#include <petsclog.h>
#include <petscsf.h>

#include <string.h>

#define SS_ASSEMBLY_DEFAULT_MEMORY_CAP (UINT64_C(256) * UINT64_C(1024) * UINT64_C(1024))
#define SS_OPERATOR_REFERENCE_ACTION_MAX_QUADRATURE 512U
#define SS_OPERATOR_MATRIX_FREE_MAX_BLOCK_NODES 35U
#define SS_OPERATOR_MATRIX_FREE_MAX_ACTION_BATCH 32U
#define SS_OPERATOR_MATRIX_FREE_MAX_PROJECTION_PHYSICAL_PATTERNS 256U
#define SS_OPERATOR_MATRIX_FREE_MAX_BLOCK_DOFS                                         \
  (SS_FE_MAX_DIMENSION * SS_OPERATOR_MATRIX_FREE_MAX_BLOCK_NODES)

typedef struct {
  const SsFeDescriptor *field_descriptor;
  const SsFeDescriptor *coordinate_descriptor;
  const SsFeQuadrature *quadrature;
  const SsMcParameters *material;
  const double *body_force;
  size_t material_region_index;
  double inverse_transpose[SS_FE_MAX_DIMENSION * SS_FE_MAX_DIMENSION];
  int affine_geometry;
  int action_needed;
  SsMechanicsPreparedElement prepared;
} SsOperatorCell;

typedef struct {
  SsFieldFacetPack pack;
  const SsFeDescriptor *field_descriptor;
  const SsFeQuadrature *quadrature;
  size_t parent_cell_index;
  size_t operation_index;
} SsOperatorNaturalFacet;

/** Persistent input and contribution exchanges for the matrix-free shell. */
typedef struct {
  PetscSF input;
  PetscSF action;
  size_t owned_offset;
  size_t interior_cells;
  size_t active_cells;
  size_t local_rows;
  const size_t *rows;
  size_t cells[];
} SsMatrixFreeExchange;

struct SsOperator {
  const SsAssemblyPlan *plan;
  SsOperatorSpec spec;
  SsMechanicsOperatorSpec mechanics;
  SsScalarOperatorSpec scalar;
  SsMechanicsMaterialRegion *material_regions;
  SsNaturalOperatorSpec *natural_operations;
  SsOperatorNaturalFacet *natural_facets;
  size_t natural_facet_count;
  SsOperatorCell *cells;
  double *cell_basis_cache;
  PetscScalar *basis_blas_cache;
  double *cell_inverse_transpose_cache;
  double *reference_gradient_cache;
  PetscScalar *reference_gradient_blas_cache;
  size_t reference_gradient_count;
  double *cell_physical_cache;
  double *cell_weight_cache;
  SsMcReduced *reduced_materials;
  double cached_reduction_factor;
  uint64_t cell_cache_bytes;
  int reduced_materials_valid;
  PetscInt *gather_indices;
  size_t gather_count;
  unsigned char *pack_required;
  size_t *packed_slots;
  double *packed_values;
  IS gather_is;
  Vec gathered_values;
  VecScatter scatter;
  size_t maximum_closure;
  size_t matrix_stride;
  size_t tile_capacity;
  double *element_residual;
  double *element_jacobian;
  double *element_potential;
  double *element_natural_work;
  SsErrorRecord *element_errors;
  size_t *row_offsets;
  double *shadow_residual;
  double *shadow_jacobian;
  unsigned char *row_complete;
  unsigned char *row_writer_violation;
  int *row_writer;
  int *worker_cpus;
  Vec staging_residual;
  Mat staging_jacobian;
  PetscScalar *stage_matrix_scratch;
  PetscScalar *coo_residual_values;
  PetscScalar *coo_jacobian_values;
  size_t *coo_residual_offsets;
  size_t *coo_jacobian_offsets;
  size_t coo_residual_count;
  size_t coo_jacobian_count;
  size_t coo_entity_count;
  int performance_coo;
  Mat matrix_free_operator;
  size_t *matrix_free_tangent_offsets;
  double *matrix_free_tangents;
  double *matrix_free_element_action;
  double *matrix_free_diagonal;
  PetscInt matrix_free_block_count;
  PetscInt *matrix_free_block_sizes;
  size_t *matrix_free_block_row_offsets;
  size_t *matrix_free_block_offsets;
  PetscScalar *matrix_free_inverse_blocks;
  PetscScalar *matrix_free_block_workspace;
  size_t matrix_free_block_workspace_stride;
  PetscScalar *matrix_free_action_workspace;
  size_t matrix_free_action_workspace_stride;
  size_t matrix_free_action_batch_cells;
  SsMatrixFreeExchange *matrix_free_exchange;
  size_t matrix_free_block_node_cap;
  int matrix_free_threaded_block_apply;
  size_t matrix_free_tangent_count;
  size_t *matrix_free_p1_constraint_pattern_indices;
  size_t matrix_free_p1_constraint_pattern_count;
  PetscScalar *matrix_free_p1_reference_gradient_blas_cache;
  const SsOperator *matrix_free_projection_source;
  size_t *matrix_free_projection_gradient_offsets;
  double *matrix_free_projection_gradients;
  size_t matrix_free_projection_gradient_count;
  double *matrix_free_projection_transfer;
  size_t matrix_free_projection_transfer_rows;
  size_t matrix_free_projection_transfer_columns;
  unsigned char *matrix_free_projection_source_constrained;
  size_t *matrix_free_projection_constraint_pattern_indices;
  size_t *matrix_free_projection_constraint_pattern_representatives;
  size_t matrix_free_projection_constraint_pattern_count;
  PetscScalar *matrix_free_projection_constraint_reference_blas_cache;
  size_t *matrix_free_projection_physical_pattern_indices;
  size_t *matrix_free_projection_physical_pattern_representatives;
  size_t matrix_free_projection_physical_pattern_count;
  double *matrix_free_projection_pattern_physical_gradients;
  double *matrix_free_projection_reference_gradients;
  PetscScalar *matrix_free_projection_reference_gradient_blas_cache;
  size_t matrix_free_projection_reference_gradient_count;
  int matrix_free_linearized;
  int matrix_free_action_failed;
  uint64_t matrix_free_generation;
  SsOperatorProperties properties;
  int *affinity_cpus;
  int affinity_count;
  SsAssemblyDiagnostics diagnostics;
  SsAssemblyFailurePoint failure_point;
  int failure_rank;
  SsEntityId failure_entity;
  SsAssemblyServiceObserver service_observer;
  void *service_context;
};

/** J^{-T} at one quadrature point: the per-cell map of an affine cell or the
 * per-point map of a curved cell. */
static inline const double *
ss_operator_cell_inverse_transpose(const SsOperatorCell *metadata,
                                   size_t quadrature_index) {
  return metadata->prepared.inverse_transposes +
         quadrature_index * metadata->prepared.inverse_transpose_stride;
}

/** Return one physical basis gradient without a per-point Q-by-P-by-d cache.
 * Every cell keeps the shared reference table and its J^{-T} map(s); the
 * arithmetic matches ss_fe_physical_gradients_apply term by term. */
static inline int ss_operator_mechanics_physical_gradient(
    const SsOperator *operator_handle, size_t cell_index, size_t quadrature_index,
    size_t node, double physical[SS_FE_MAX_DIMENSION]) {
  const SsCellPlan *cell;
  const SsOperatorCell *metadata;
  const size_t dimension =
      operator_handle == NULL ? 0U : operator_handle->properties.dimension;
  const double *reference;
  const double *inverse_transpose;
  size_t row;
  if (operator_handle == NULL || physical == NULL ||
      cell_index >= operator_handle->plan->cell_count || dimension == 0U ||
      dimension > SS_FE_MAX_DIMENSION) {
    return 0;
  }
  cell = &operator_handle->plan->cells[cell_index];
  metadata = &operator_handle->cells[cell_index];
  if (quadrature_index >= metadata->quadrature->point_count ||
      node >= metadata->field_descriptor->node_count || cell->dimension != dimension) {
    return 0;
  }
  if (metadata->prepared.physical_gradients != NULL) {
    const double *cached =
        metadata->prepared.physical_gradients +
        (quadrature_index * metadata->field_descriptor->node_count + node) * dimension;
    memcpy(physical, cached, dimension * sizeof(*physical));
    return 1;
  }
  if (metadata->prepared.inverse_transposes == NULL ||
      operator_handle->reference_gradient_cache == NULL ||
      operator_handle->reference_gradient_count !=
          metadata->quadrature->point_count * metadata->field_descriptor->node_count *
              dimension) {
    return 0;
  }
  reference =
      operator_handle->reference_gradient_cache +
      (quadrature_index * metadata->field_descriptor->node_count + node) * dimension;
  inverse_transpose = ss_operator_cell_inverse_transpose(metadata, quadrature_index);
  for (row = 0U; row < dimension; ++row) {
    size_t column;
    physical[row] = 0.0;
    for (column = 0U; column < dimension; ++column) {
      physical[row] += inverse_transpose[row * dimension + column] * reference[column];
    }
  }
  return 1;
}

/** All physical basis gradients of one quadrature point (node-major). */
static inline void
ss_operator_mechanics_point_gradients(const SsOperator *operator_handle,
                                      const SsOperatorCell *metadata,
                                      size_t quadrature_index, double *physical) {
  const size_t node_count = metadata->field_descriptor->node_count;
  const size_t dimension = operator_handle->properties.dimension;
  if (metadata->prepared.physical_gradients != NULL) {
    memcpy(physical,
           metadata->prepared.physical_gradients +
               quadrature_index * node_count * dimension,
           node_count * dimension * sizeof(*physical));
    return;
  }
  ss_fe_physical_gradients_apply(
      (uint32_t)dimension, node_count,
      ss_operator_cell_inverse_transpose(metadata, quadrature_index),
      operator_handle->reference_gradient_cache +
          quadrature_index * node_count * dimension,
      physical);
}

/* PETSc events */
extern PetscLogEvent ss_operator_assembly_event;
extern PetscLogEvent ss_operator_scatter_event;
extern PetscLogEvent ss_operator_prepare_event;
extern PetscLogEvent ss_operator_compute_event;
extern PetscLogEvent ss_operator_stage_event;
extern PetscLogEvent ss_operator_commit_event;
extern PetscLogEvent ss_operator_validate_event;
extern PetscLogEvent ss_operator_p1_project_event;
extern PetscLogEvent ss_operator_matrix_free_fine_event;
extern PetscLogEvent ss_operator_matrix_free_projected_event;
extern PetscLogEvent ss_operator_matrix_free_block_apply_event;
extern PetscLogEvent ss_operator_matrix_free_block_setup_event;
extern PetscLogEvent ss_operator_matrix_free_tangent_event;

/* Operator lifecycle, diagnostics and assembled-operation entry points */
/** Register the PETSc log events of the operators once. */
PetscErrorCode ss_operator_register_events(void);
/** Whether a request is a valid combination of ::SsAssemblyRequest bits. */
int ss_operator_request_valid(uint32_t request);

/* Cell metadata, material lookup, natural facets, and input gather setup */
/** Compare material regions by label value for qsort(). */
int ss_operator_material_compare(const void *left, const void *right);
/** Build the per-cell geometry, material and quadrature metadata; local. */
SsStatus ss_operator_build_cell_metadata(SsOperator *operator_handle);
/** Reduce every region's strength parameters by a factor; local. */
SsStatus ss_operator_prepare_reduced_materials(SsOperator *operator_handle,
                                               double reduction_factor);
/** Pack the natural-boundary facets of the operator; local. */
SsStatus ss_operator_build_natural_facets(SsOperator *operator_handle);
/** Create the ghost scatter of the field coefficients; collective. */
SsStatus ss_operator_build_gather(SsOperator *operator_handle);

/* Pack, evaluate and publish assembled residuals, tangents and potentials */
/** OpenMP thread number of the caller, or 0 outside OpenMP. */
int ss_operator_thread(void);
/** Record a library call; fails with ::SS_ERROR_THREAD inside a parallel region. */
SsStatus ss_operator_service(SsOperator *operator_handle, SsAssemblyService service);
/** Gather resident coefficients and borrow their read-only array. */
SsStatus ss_operator_gather(SsOperator *operator_handle, const SsOperatorState *state,
                            const PetscScalar **gathered);
/** Pack coefficients in the active assembly team; all workers must enter. */
void ss_operator_pack_cells(SsOperator *operator_handle, const PetscScalar *gathered);
/** Merge one element failure using the fixed assembly diagnostic order. */
static inline void ss_operator_keep_first_error(SsErrorRecord *first,
                                                const SsErrorRecord *candidate) {
  if (candidate->valid && ss_error_compare(candidate, first) < 0) {
    *first = *candidate;
  }
}
#if SLOPE_ENABLE_OPENMP
#pragma omp declare reduction(                                                         \
        ss_first_error:SsErrorRecord : ss_operator_keep_first_error(                   \
                &omp_out, &omp_in)) initializer(omp_priv = {0})
#endif
/** Whether the local work is large enough to use the OpenMP team. */
int ss_operator_use_openmp(const SsOperator *operator_handle);
/** Agree on the local error records of an evaluation; collective. */
SsStatus ss_operator_collective_validation(SsOperator *operator_handle,
                                           const SsErrorRecord *local_error,
                                           SsErrorRecord *selected);
/** Assemble the requested quantities inside the PETSc log events; collective. */
SsStatus ss_operator_assemble_profiled(SsOperator *operator_handle,
                                       const SsOperatorState *state, uint32_t request,
                                       Vec residual, Mat jacobian, double *functional,
                                       SsErrorRecord *error);

/* Prepare and accumulate the production COO matrix staging */
/** Position of a global column in an owned row, or SIZE_MAX. */
size_t ss_operator_find_row_column(const SsOwnedRowPlan *row, PetscInt global_column);
/** Prepare the COO staging of performance-mode Jacobians; collective. */
SsStatus ss_operator_prepare_performance_coo(SsOperator *operator_handle);
/** Stage one evaluated cell. COO slots are disjoint across workers. */
void ss_operator_stage_cell(SsOperator *operator_handle, uint32_t request,
                            size_t cell_index, size_t tile_slot);
/** Stage one evaluated natural facet. COO slots are disjoint across workers. */
void ss_operator_stage_natural(SsOperator *operator_handle, uint32_t request,
                               size_t facet_index, size_t tile_slot);
/** Clear PETSc staging objects before a new local worker pass. */
SsStatus ss_operator_prepare_performance_staging(SsOperator *operator_handle,
                                                 uint32_t request);
/** Commit staged values into PETSc after a performance-mode pass; collective. */
SsStatus ss_operator_finish_performance_staging(SsOperator *operator_handle,
                                                uint32_t request);

/* Size and allocate workspaces for assembled and matrix-free operators */
/** Byte count of an array without overflow. */
SsStatus ss_operator_checked_bytes(size_t count, size_t item_size, uint64_t *bytes);
/** Unknowns in the variable point block of an owned row. */
SsStatus ss_operator_mechanics_block_size(const SsOperator *operator_handle, size_t row,
                                          size_t *block_size);
/** Number and total size of the mechanics point blocks. */
SsStatus ss_operator_mechanics_block_requirements(const SsOperator *operator_handle,
                                                  size_t *block_count,
                                                  size_t *entry_count);
/** Per-thread workspace of the matrix-free block inverses. */
SsStatus ss_operator_matrix_free_block_workspace_requirements(
    const SsOperator *operator_handle, size_t *stride, size_t *entry_count);
/** Grow the block workspace to a point and strain count. */
SsStatus ss_operator_matrix_free_reserve_block_workspace(SsOperator *operator_handle,
                                                         size_t point_count,
                                                         size_t strain_count);
/** Per-thread workspace of the matrix-free action. */
SsStatus ss_operator_matrix_free_action_workspace_requirements(
    const SsOperator *operator_handle, size_t *stride, size_t *entry_count);
/** Stored tangents and action data of the matrix-free operator. */
SsStatus ss_operator_matrix_free_storage_requirements(const SsOperator *operator_handle,
                                                      size_t *tangent_count,
                                                      size_t *action_count,
                                                      uint64_t *storage_bytes);
/** Admit and allocate the operator workspaces against the memory cap. */
SsStatus ss_operator_allocate_work(SsOperator *operator_handle);

/* Reference-table tetrahedral residual and matrix-free action kernels */
/** Residual of one tetrahedron with the shared reference tables; pure. */
SsStatus ss_operator_matrix_free_reference_residual_cell_3d(
    const SsOperator *operator_handle, size_t cell_index, const SsOperatorState *state,
    uint32_t request, SsElementOutput *output, SsErrorRecord *error);
/** Tangent action of one tetrahedron with reference tables; pure. */
void ss_operator_matrix_free_reference_cell_action_3d(
    const SsOperator *action_operator, const SsOperator *tangent_operator,
    size_t cell_index, const double *reference_gradients, size_t node_count,
    const PetscScalar *gathered, int transpose);
/** Tangent action of one tetrahedron through dense BLAS kernels. */
void ss_operator_matrix_free_blas_cell_action_3d(
    const SsOperator *action_operator, const SsOperator *tangent_operator,
    size_t cell_index, const PetscScalar *reference_gradient_matrices,
    size_t node_count, const PetscScalar *gathered, int transpose);
/** Tangent action of a batch of tetrahedra through dense BLAS kernels. */
int ss_operator_matrix_free_blas_cell_batch_3d(const SsOperator *operator_handle,
                                               const size_t *cell_indices,
                                               size_t batch_count,
                                               const PetscScalar *gathered,
                                               int transpose, PetscScalar *workspace,
                                               size_t workspace_stride);

/* Apply matrix-free tangents and their transposes */
/** Fine-level gradient of a coarse basis function at a fine quadrature point. */
int ss_operator_matrix_free_projected_physical_gradient(
    const SsOperator *fine_operator, const SsOperator *coarse_operator,
    size_t cell_index, size_t quadrature_index, size_t coarse_slot,
    double physical[SS_FE_MAX_DIMENSION]);
/** MatMult of the matrix-free shell; flags local failures on the output. */
PetscErrorCode ss_operator_matrix_free_mult(Mat matrix, Vec input, Vec output);
/** MatMultTranspose of the matrix-free shell. */
PetscErrorCode ss_operator_matrix_free_mult_transpose(Mat matrix, Vec input,
                                                      Vec output);
/** Build the interior/halo split and exchange schedule; collective. */
SsStatus ss_operator_matrix_free_prepare_exchange(SsOperator *operator_handle);
/** MatGetDiagonal of the matrix-free shell. */
PetscErrorCode ss_operator_matrix_free_get_diagonal(Mat matrix, Vec diagonal);
/** MatCreateVecs of the matrix-free shell. */
PetscErrorCode ss_operator_matrix_free_create_vecs(Mat matrix, Vec *right, Vec *left);

/* Build and apply matrix-free variable-block inverses */
/** Entry of a cell's linearized tangent between two closure slots. */
double ss_operator_matrix_free_cell_entry(const SsOperator *operator_handle,
                                          size_t cell_index, size_t row_slot,
                                          size_t column_slot);
/** Closure slot of an owned row in a cell, or SIZE_MAX. */
size_t ss_operator_matrix_free_cell_slot(const SsCellPlan *cell,
                                         const SsOwnedRowPlan *row);
/** Assemble and invert one entity block of the tangent diagonal. */
int ss_operator_matrix_free_entity_block(const SsOperator *operator_handle,
                                         size_t block_row_start, PetscInt block_size,
                                         PetscScalar *inverse, PetscScalar *workspace);
/** Invert a block in place. The caller validates the packed inverse before use. */
int ss_operator_matrix_free_invert_block(PetscInt size, PetscScalar *values);
/** Whether every stored block inverse is finite. */
int ss_operator_matrix_free_inverses_finite(const SsOperator *operator_handle);
/** MATOP_INVERT_VBLOCK_DIAGONAL of the matrix-free shell. */
PetscErrorCode ss_operator_matrix_free_invert_variable_block_diagonal(
    Mat matrix, PetscInt block_count, const PetscInt *block_sizes, PetscScalar *values);
/** Apply the stored variable-block inverses (point-block Jacobi). */
PetscErrorCode ss_operator_matrix_free_apply_inverse_variable_blocks(Mat matrix,
                                                                     Vec input,
                                                                     Vec output);

#endif
