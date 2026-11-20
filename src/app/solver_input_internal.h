/* Private interfaces for reading a compiled problem and building its solver graph.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_APP_SOLVER_INPUT_INTERNAL_H
#define SS_APP_SOLVER_INPUT_INTERNAL_H

#include "solver_internal.h"

#include "yyjson.h"

/* Read, label and partition the compiled mesh payload */
/** Release a raw mesh read from the payload. */
void ss_solver_raw_destroy(SsSolverRawMesh *raw);
/** Binary search of a stable identity in a sorted array. */
int ss_solver_entity_id_index(const SsEntityId *ids, size_t count, SsEntityId target,
                              size_t *index);
/** Read and route balanced input slabs into the initial local cell closures. */
SsStatus ss_solver_load_mesh(MPI_Comm communicator, const char *path,
                             const SsSolverProblemModel *model, SsSolverRawMesh *raw);

/* Read compiled problem models, materials, supports and outputs */
/** Read a whole file; the caller frees the data. */
SsStatus ss_solver_read_file(const char *path, unsigned char **data, size_t *size);
/** Whether a JSON value is the given string. */
int ss_solver_json_string_equal(yyjson_val *value, const char *expected);
/** Member of a JSON object, or NULL. */
yyjson_val *ss_solver_object(yyjson_val *parent, const char *name);
/** String member of a JSON object, or NULL. */
const char *ss_solver_string(yyjson_val *parent, const char *name);
/** Parse requested output quantities into a bit mask. */
SsStatus ss_solver_parse_output_quantities(yyjson_val *array, uint64_t *quantity_mask);
/** Output quantities available for the problem's physics. */
uint64_t ss_solver_available_output_quantities(const SsSolverProblemModel *model);
/** Boolean member of a JSON object, or the fallback. */
int ss_solver_boolean(yyjson_val *parent, const char *name, int fallback);
/** Number member of a JSON object, or the fallback. */
double ss_solver_number(yyjson_val *parent, const char *name, double fallback);
/** Unsigned integer member of a JSON object, or the fallback. */
uint64_t ss_solver_uint(yyjson_val *parent, const char *name, uint64_t fallback);
/** Decode model metadata and this rank's portions of its membership lists. */
SsStatus ss_solver_parse_problem(const SsCompiledManifest *manifest, int rank,
                                 int ranks, SsSolverProblemModel *model);

/* Read resolved solver policies, controls and result options */
/** Parse the resolved solver policies, controls and result options of a run; local. */
SsStatus ss_solver_parse_run(const char *path, SsSolverRunPolicy *policy);

typedef SsStatus (*SsSolverConstraintValueEvaluator)(const SsSolverGraph *graph,
                                                     SsEntityId stable_entity_id,
                                                     const double *physical,
                                                     uint32_t component, double *value);

/* Build and evaluate essential mechanics and hydraulic constraints */
/** Prescribed displacement component at a boundary entity. */
SsStatus ss_solver_displacement_boundary_value(const SsSolverGraph *graph,
                                               SsEntityId stable_entity_id,
                                               const double *physical,
                                               uint32_t component, double *value);
/** Prescribed pore pressure at a boundary entity. */
SsStatus ss_solver_pressure_constraint_value(const SsSolverGraph *graph,
                                             SsEntityId stable_entity_id,
                                             const double *physical, uint32_t component,
                                             double *value);
/** Evaluate constraint values at resident nodes and resolve shared values; collective.
 */
SsStatus ss_solver_refine_constraints(SsSolverGraph *graph, SsFieldSpace *space,
                                      SsStrongConstraint *constraints,
                                      size_t constraint_count, uint32_t components,
                                      SsSolverConstraintValueEvaluator evaluator);

#endif
