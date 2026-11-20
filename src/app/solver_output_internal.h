/* Private interfaces shared by the result publishers and the output post-processing.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_APP_SOLVER_OUTPUT_INTERNAL_H
#define SS_APP_SOLVER_OUTPUT_INTERNAL_H

#include "solver_internal.h"

/**
 * One requested quantity in publication order.
 *
 * Exactly one of `field` and `category` is nonnull. The views are owned by the
 * publisher; the list only borrows them while declaring and writing.
 */
typedef struct {
  /** `SS_SOLVER_QUANTITY_*` bit that requests the quantity. */
  uint64_t quantity;
  /** Publication stage reported if the declaration fails. */
  const char *declare_stage;
  /** Publication stage reported if the state write fails. */
  const char *write_stage;
  /** Nodal or quadrature field view, or NULL for a category. */
  const SsResultFieldView *field;
  /** Categorical view, or NULL for a field. */
  const SsResultCategoryView *category;
} SsSolverPublishedQuantity;

/** Largest number of entries `ss_solver_hydraulic_outputs_list()` produces. */
#define SS_SOLVER_HYDRAULIC_OUTPUT_COUNT 9U

/**
 * Hydraulic output quantities of one accepted state and their result views.
 *
 * The hydraulic publisher and the coupled mechanics publisher both publish this
 * set. The coupled body force belongs to the coupled mechanics result only; it is
 * listed between the Darcy velocity and the saturation when
 * `with_coupled_body_force` is nonzero. Built views point into the packed
 * outputs, so the structure must not be copied after
 * `ss_solver_hydraulic_outputs_views()`.
 */
typedef struct {
  SsSolverOutputField pressure;
  SsSolverOutputField gradient;
  SsSolverOutputField conductivity;
  SsSolverOutputField ramp;
  SsSolverOutputField epsilon;
  SsSolverOutputField pressure_flux;
  SsSolverOutputField darcy_velocity;
  /** Packed by the caller with `ss_solver_coupled_body_force_output()`. */
  SsSolverOutputField coupled_body_force;
  SsSolverOutputCategory saturation;
  SsResultFieldView pressure_view;
  SsResultFieldView gradient_view;
  SsResultFieldView conductivity_view;
  SsResultFieldView ramp_view;
  SsResultFieldView epsilon_view;
  SsResultFieldView pressure_flux_view;
  SsResultFieldView darcy_velocity_view;
  SsResultFieldView coupled_body_force_view;
  SsResultCategoryView saturation_view;
  /** Nonzero when the coupled body force is part of the published set. */
  int with_coupled_body_force;
} SsSolverHydraulicOutputs;

/**
 * Build a nodal (`field_dof`, Lagrange) view of `field` for `state_id`.
 *
 * The strings and component identifiers are borrowed and must outlive the view.
 */
SsResultFieldView ss_solver_nodal_view(const char *quantity_id, const char *unit,
                                       const char *const *component_ids,
                                       uint32_t component_count,
                                       const SsSolverOutputField *field,
                                       SsStateId state_id);

/**
 * Build a cell-quadrature view of `field` at `location` for `state_id`.
 *
 * The strings and component identifiers are borrowed and must outlive the view.
 */
SsResultFieldView ss_solver_quadrature_view(const char *quantity_id, const char *unit,
                                            const char *location,
                                            const char *const *component_ids,
                                            uint32_t component_count,
                                            const SsSolverOutputField *field,
                                            SsStateId state_id);

/**
 * Build a categorical quadrature view of `category` at `location` for `state_id`.
 *
 * The strings and category identifiers are borrowed and must outlive the view.
 */
SsResultCategoryView
ss_solver_category_view(const char *quantity_id, const char *unit, const char *location,
                        const char *const *category_ids, uint32_t category_count,
                        const SsSolverOutputCategory *category, SsStateId state_id);

/**
 * Declare every requested entry of `quantities` in list order.
 *
 * Collective over the writer communicator. `stage` is updated to the declaration
 * stage of the entry being processed so that a failure can be reported.
 */
SsStatus ss_solver_publish_declare(SsResultWriter *writer, const SsSolverGraph *graph,
                                   const SsSolverPublishedQuantity *quantities,
                                   size_t count, const char **stage);

/**
 * Write the current state of every requested entry of `quantities` in list order.
 *
 * Nodal views are written as fields, quadrature views as quantities and
 * categorical views as categories. Collective over the writer communicator.
 * `stage` is updated to the write stage of the entry being processed.
 */
SsStatus ss_solver_publish_write(SsResultWriter *writer, const SsSolverGraph *graph,
                                 const SsSolverPublishedQuantity *quantities,
                                 size_t count, const char **stage);

/**
 * Pack the pore pressure and the hydraulic quadrature quantities of one state.
 *
 * `local_values` is the ghosted local hydraulic solution. Collective over the
 * graph communicator. `stage` names the failing packing step. The coupled body
 * force is not packed here. On failure the caller still owns `outputs` and
 * releases it with `ss_solver_hydraulic_outputs_destroy()`.
 */
SsStatus ss_solver_hydraulic_outputs_pack(const SsSolverGraph *graph,
                                          const SsSolverHydraulic *hydraulic,
                                          const PetscScalar *local_values,
                                          SsSolverHydraulicOutputs *outputs,
                                          const char **stage);

/** Build all hydraulic views of `outputs` for `state_id`; local, no communication. */
void ss_solver_hydraulic_outputs_views(const SsSolverGraph *graph, SsStateId state_id,
                                       SsSolverHydraulicOutputs *outputs);

/** Rebind the built hydraulic views of `outputs` to their packed data at `state_id`. */
void ss_solver_hydraulic_outputs_bind(SsSolverHydraulicOutputs *outputs,
                                      SsStateId state_id);

/**
 * Append the hydraulic publication entries of `outputs` to `quantities`.
 *
 * `quantities` must have room for `SS_SOLVER_HYDRAULIC_OUTPUT_COUNT` entries.
 * Returns the number of entries written. The entries borrow the views of
 * `outputs`.
 */
size_t ss_solver_hydraulic_outputs_list(const SsSolverHydraulicOutputs *outputs,
                                        SsSolverPublishedQuantity *quantities);

/** Release the packed hydraulic outputs and reset `outputs`; NULL is accepted. */
void ss_solver_hydraulic_outputs_destroy(SsSolverHydraulicOutputs *outputs);

/**
 * Evaluate the global hydraulic flux balance and free residual of `solution`.
 *
 * `local_balance` holds this rank's boundary contributions from
 * `ss_solver_hydraulic_boundary_balance()`. Rank 0 adds the prescribed natural
 * flux and the squared free residual norm before the global reduction. The
 * acceptance verdict is `balance->accepted`; reporting a rejection is left to
 * the caller. Collective over the graph communicator.
 */
SsStatus ss_solver_hydraulic_balance_evaluate(
    const SsSolverGraph *graph, SsSolverHydraulic *hydraulic, Vec solution,
    const SsNonlinearSolveRecord *solve_record, SsHydraulicBalanceLocal *local_balance,
    SsHydraulicBalance *balance, SsErrorRecord *error);

/**
 * Receive one owned natural-boundary facet and its element residual.
 *
 * `residual` has `facet->closure_count` entries in closure order. A nonzero
 * status stops the traversal on this rank; the traversal agrees on it after the
 * current operation.
 */
typedef SsStatus (*SsSolverNaturalFacetVisit)(void *context,
                                              const SsFieldFacetPack *facet,
                                              const double *residual);

/**
 * Evaluate the natural-boundary residual on every owned facet of `operations`.
 *
 * For each operation in order, the owned support facets of `space` are packed and
 * their closure values are taken from the prescribed values or from the ghosted
 * `local_values`. The facet residual of the operation's point kernel is then
 * passed to `visit` in pack order. Every facet must carry `components` field
 * components. Collective over the graph communicator once per operation.
 */
SsStatus ss_solver_natural_facet_residuals(
    const SsSolverGraph *graph, SsFieldSpace *space, uint32_t field_order,
    uint32_t components, const SsNaturalOperatorSpec *operations,
    size_t operation_count, const PetscScalar *local_values,
    SsSolverNaturalFacetVisit visit, void *context);

#endif
