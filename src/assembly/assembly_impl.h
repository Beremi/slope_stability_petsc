/* Assembly plan state shared by the plan and element sources.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_ASSEMBLY_IMPL_H
#define SS_ASSEMBLY_IMPL_H

#include "slope_internal/assembly/assembly.h"

typedef struct {
  size_t cell_index;
  size_t row_slot;
} SsRowCellSlot;

typedef struct {
  PetscInt global_row;
  uint32_t component;
  SsEntityId field_entity_id;
  uint32_t field_entity_slot;
  PetscInt *columns;
  size_t column_count;
  size_t diagonal_count;
  size_t off_diagonal_count;
  const SsRowCellSlot *cell_slots;
  size_t cell_count;
} SsOwnedRowPlan;

struct SsAssemblyPlan {
  MPI_Comm communicator;
  int rank;
  int size;
  PetscInt row_start;
  PetscInt row_end;
  const SsFieldSpace *field_space;
  SsFieldSummary field_summary;
  const SsCellPlan *cells;
  size_t cell_count;
  SsOwnedRowPlan *rows;
  SsRowCellSlot *row_cell_slots;
  size_t row_count;
  int matrix_pattern;
  SsAssemblyPlanSummary summary;
};

/*
 * Geometry and material data that are invariant during Newton iterations.
 * Arrays are point-major and use the descriptor sizes carried by the public
 * element input.  Keeping this private preserves the standalone element
 * contract while allowing an assembled operator to amortize geometry work.
 *
 * Physical gradients are not stored per point: they are reconstructed from
 * the shared reference table and J^{-T}, which is stored once per affine cell
 * (stride 0) or once per point of a curved cell (stride d * d).  The optional
 * physical_gradients table is used verbatim when a caller provides one.
 */
typedef struct {
  const double *basis;
  const double *physical_gradients;
  const double *reference_gradients;
  const double *inverse_transposes;
  size_t inverse_transpose_stride;
  const double *physical_points;
  const double *weights;
  const SsMcReduced *reduced_material;
} SsMechanicsPreparedElement;

/** Mechanics element with precomputed geometry; pure, safe in OpenMP regions. */
SsStatus ss_assembly_mechanics_element_prepared(
    const SsMechanicsElementInput *input, const SsMechanicsPreparedElement *prepared,
    uint32_t request, SsElementOutput *output, SsErrorRecord *error);

/** Agree on an assembly status across ranks; collective. */
SsStatus ss_assembly_collective_status_private(MPI_Comm communicator,
                                               SsStatus local_status);
/** Fill an assembly error record with its code, entity and rank. */
void ss_assembly_error_private(SsErrorRecord *error, SsStatus status, const char *code,
                               SsEntityId entity, int rank);

#endif
