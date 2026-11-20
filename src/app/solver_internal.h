/* Private graph, solver state and interfaces shared by native application modules.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_APP_SOLVER_INTERNAL_H
#define SS_APP_SOLVER_INTERNAL_H

#include "slope_internal/analysis/analysis.h"
#include "slope_internal/app/solver.h"
#include "slope_internal/assembly/assembly.h"
#include "slope_internal/core/collective.h"
#include "slope_internal/core/solver_options.h"
#include "slope_internal/coupling/coupling.h"
#include "slope_internal/diagnostics/journal.h"
#include "slope_internal/hydraulic/hydraulic.h"
#include "slope_internal/material/material.h"
#include "slope_internal/mesh/mesh.h"
#include "slope_internal/nonlinear/nonlinear.h"
#include "slope_internal/result/result.h"

#include <hdf5.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define SS_SOLVER_NAME_CAPACITY 128U

typedef enum {
  SS_SOLVER_MATERIAL_MECHANICS = 1,
  SS_SOLVER_MATERIAL_HYDRAULIC = 2
} SsSolverMaterialKind;

typedef enum {
  SS_SOLVER_VALUE_CONSTANT = 1,
  SS_SOLVER_VALUE_HYDROSTATIC = 2,
  SS_SOLVER_VALUE_PIECEWISE_LEVEL = 3,
  SS_SOLVER_VALUE_GLOBAL_TRACTION = 4,
  SS_SOLVER_VALUE_SPATIAL_GLOBAL_TRACTION = 5,
  SS_SOLVER_VALUE_NORMAL_PRESSURE = 6,
  SS_SOLVER_VALUE_PRESCRIBED_FLUX = 7,
  SS_SOLVER_VALUE_AFFINE_COORDINATE = 8
} SsSolverBoundaryValueKind;

typedef enum {
  SS_SOLVER_BOUNDARY_ESSENTIAL = 1,
  SS_SOLVER_BOUNDARY_NATURAL = 2
} SsSolverBoundaryConditionKind;

typedef enum {
  SS_SOLVER_OUTPUT_TERMINAL = 1,
  SS_SOLVER_OUTPUT_ACCEPTED_STATE = 2
} SsSolverOutputFrequency;

typedef enum {
  SS_SOLVER_CHECKPOINT_NONE = 0,
  SS_SOLVER_CHECKPOINT_WRITE_ACCEPTED = 1,
  SS_SOLVER_CHECKPOINT_RESTORE_ACCEPTED = 2
} SsSolverCheckpointMode;

#define SS_SOLVER_QUANTITY_DISPLACEMENT (UINT64_C(1) << 0U)
#define SS_SOLVER_QUANTITY_STRAIN (UINT64_C(1) << 1U)
#define SS_SOLVER_QUANTITY_STRESS (UINT64_C(1) << 2U)
#define SS_SOLVER_QUANTITY_MATERIAL_BRANCH (UINT64_C(1) << 3U)
#define SS_SOLVER_QUANTITY_PLASTIC_INDICATOR (UINT64_C(1) << 4U)
#define SS_SOLVER_QUANTITY_YIELD_RESIDUAL (UINT64_C(1) << 5U)
#define SS_SOLVER_QUANTITY_PORE_PRESSURE (UINT64_C(1) << 6U)
#define SS_SOLVER_QUANTITY_PRESSURE_GRADIENT (UINT64_C(1) << 7U)
#define SS_SOLVER_QUANTITY_CONDUCTIVITY (UINT64_C(1) << 8U)
#define SS_SOLVER_QUANTITY_HYDRAULIC_RAMP (UINT64_C(1) << 9U)
#define SS_SOLVER_QUANTITY_HYDRAULIC_EPSILON (UINT64_C(1) << 10U)
#define SS_SOLVER_QUANTITY_SATURATION (UINT64_C(1) << 11U)
#define SS_SOLVER_QUANTITY_PRESSURE_FLUX (UINT64_C(1) << 12U)
#define SS_SOLVER_QUANTITY_DARCY_VELOCITY (UINT64_C(1) << 13U)
#define SS_SOLVER_QUANTITY_COUPLED_BODY_FORCE (UINT64_C(1) << 14U)
#define SS_SOLVER_QUANTITY_GLOBAL_FLUX_BALANCE (UINT64_C(1) << 15U)
#define SS_SOLVER_QUANTITY_CONTINUATION (UINT64_C(1) << 16U)
#define SS_SOLVER_QUANTITY_NONLINEAR_HISTORY (UINT64_C(1) << 17U)
#define SS_SOLVER_QUANTITY_SAFETY_FACTOR (UINT64_C(1) << 18U)
#define SS_SOLVER_QUANTITY_LOAD_FACTOR (UINT64_C(1) << 19U)
#define SS_SOLVER_QUANTITY_WORK_CONTROL (UINT64_C(1) << 20U)
#define SS_SOLVER_QUANTITY_MAXIMUM_DISPLACEMENT (UINT64_C(1) << 21U)

typedef struct {
  char name[SS_SOLVER_NAME_CAPACITY];
  char support[SS_SOLVER_NAME_CAPACITY];
  SsSolverMaterialKind kind;
  uint32_t support_value;
  SsEntityId *entity_ids;
  size_t entity_count;
  double young_modulus;
  double poisson_ratio;
  double cohesion;
  double friction_angle;
  double dilatancy_angle;
  SsMcDavis davis;
  double unsaturated_unit_weight;
  double saturated_unit_weight;
  double conductivity;
  double water_unit_weight;
} SsSolverMaterial;

typedef struct {
  char name[SS_SOLVER_NAME_CAPACITY];
  char field[SS_SOLVER_NAME_CAPACITY];
  char support[SS_SOLVER_NAME_CAPACITY];
  SsSolverBoundaryConditionKind condition_kind;
  SsSolverBoundaryValueKind value_kind;
  uint32_t support_value;
  uint32_t component_mask;
  SsEntityId *entity_ids;
  size_t entity_count;
  double values[SS_FE_MAX_DIMENSION];
  size_t value_count;
  double gradient[SS_FE_MAX_DIMENSION * SS_FE_MAX_DIMENSION];
  double reference_level;
  double reference_pressure;
  double water_unit_weight;
  double piecewise_coordinates[SS_HYDRAULIC_HEAD_POINT_CAPACITY];
  double piecewise_levels[SS_HYDRAULIC_HEAD_POINT_CAPACITY];
  size_t piecewise_count;
} SsSolverBoundaryRule;

typedef struct {
  uint32_t dimension;
  uint32_t coordinate_order;
  uint32_t mechanics_order;
  uint32_t hydraulic_order;
  int mechanics_enabled;
  int hydraulic_enabled;
  int coupling_enabled;
  SsCouplingMode coupling_mode;
  SsSolverPrescribedSaturation prescribed_saturation;
  SsAnalysisMethod analysis_method;
  /** Volume rule of every fused volume operator, output and coupling target. */
  SsFeQuadratureMode volume_quadrature_mode;
  char analysis_identity[SS_SOLVER_NAME_CAPACITY];
  char result_analysis_method[SS_SOLVER_NAME_CAPACITY];
  char mesh_id[SS_SHA256_HEX_CAPACITY + 8U];
  SsSolverMaterial *materials;
  size_t material_count;
  SsSolverBoundaryRule *boundaries;
  size_t boundary_count;
  uint64_t required_output_mask;
} SsSolverProblemModel;

typedef struct {
  size_t count;
  uint32_t width;
  SsEntityId *ids;
  SsEntityId *vertex_ids;
  unsigned char *exterior;
} SsSolverEntityTable;

typedef struct {
  SsMeshInput input;
  SsEntityId *vertex_ids;
  double *vertex_coordinates;
  SsEntityId *cell_ids;
  uint64_t *cells;
  SsEntityId *boundary_facet_ids;
  uint64_t *boundary_facets;
  SsMeshIntermediateEntity *intermediate_entities;
  SsMeshLabelAssignment *labels;
  size_t label_count;
  SsEntityId *coordinate_node_ids;
  double *coordinate_coordinates;
  uint64_t *cell_coordinate_nodes;
  SsSolverEntityTable entities[SS_FE_MAX_DIMENSION + 1U];
} SsSolverRawMesh;

typedef struct {
  SsSolverSettings settings;
  double initial_factor;
  double initial_step;
  double minimum_step;
  double maximum_step;
  double maximum_work;
  double initial_work_scale;
  double scaled_slope_tolerance;
  double factor_increment_stop;
  PetscInt maximum_accepted_states;
  SsNonlinearStoppingCriterion initial_nonlinear_stopping_criterion;
  double initial_nonlinear_stopping_tolerance;
  double nonlinear_rtol;
  double nonlinear_atol;
  SsNonlinearStoppingCriterion nonlinear_stopping_criterion;
  double nonlinear_stopping_tolerance;
  double nonlinear_regularization_minimum;
  PetscInt nonlinear_maximum_iterations;
  double hydraulic_nonlinear_rtol;
  double hydraulic_nonlinear_atol;
  PetscInt hydraulic_nonlinear_maximum_iterations;
  SsLinearMethod linear_method;
  double linear_rtol;
  double linear_atol;
  SsLinearStoppingCondition linear_stopping_condition;
  PetscInt linear_maximum_iterations;
  double hydraulic_linear_rtol;
  double hydraulic_linear_atol;
  PetscInt hydraulic_linear_maximum_iterations;
  PetscInt linear_restart;
  PetscInt linear_recycle_dimension;
  int pmg_enabled;
  SsPmgLevelMethod pmg_smoother_method;
  PetscInt pmg_base_iterations;
  PetscInt pmg_hydraulic_base_iterations;
  PetscInt pmg_pre_smoothing_steps;
  PetscInt pmg_post_smoothing_steps;
  SsPmgLevelMethod pmg_coarse_method;
  uint64_t pmg_coarse_direct_maximum_dofs;
  size_t pmg_coarse_redundant_group_size;
  size_t pmg_level_count;
  uint32_t pmg_level_orders[SS_PMG_MAX_LEVELS];
  size_t pmg_active_rank_counts[SS_PMG_MAX_LEVELS];
  PetscMPIInt *pmg_active_ranks[SS_PMG_MAX_LEVELS];
  int deflation_enabled;
  size_t deflation_maximum_dimension;
  double deflation_admission_tolerance;
  double deflation_pivot_tolerance;
  double deflation_condition_limit;
  double deflation_invariant_tolerance;
  size_t deflation_reorthogonalization_passes;
  SsDeflationProjector deflation_projector;
  SsDeflationCandidatePolicy deflation_candidate_policy;
  SsDeflationCompression deflation_compression;
  size_t deflation_compression_dimension;
  double deflation_compression_history_decay;
  SsSolverOutputFrequency output_frequency;
  int output_history;
  uint64_t requested_output_mask;
  SsSolverCheckpointMode checkpoint_mode;
  char checkpoint_path[SS_RESULT_PATH_CAPACITY];
  char numerical_policy_id[SS_SHA256_HEX_CAPACITY + 8U];
  double conservation_physical_scale;
  double conservation_normalized_tolerance;
} SsSolverRunPolicy;

typedef struct {
  MPI_Comm communicator;
  int rank;
  int size;
  SsSolverProblemModel model;
  SsSolverRunPolicy policy;
  SsSolverRawMesh raw;
  SsMesh *mesh;
  SsStrongConstraint *mechanics_constraints;
  size_t mechanics_constraint_count;
  SsStrongConstraint *hydraulic_constraints;
  size_t hydraulic_constraint_count;
  SsFieldSpace *mechanics_space;
  SsFieldSpace *hydraulic_space;
  SsAssemblyPlan *mechanics_plan;
  SsAssemblyPlan *hydraulic_plan;
  SsMcParameters **material_parameters;
  SsMechanicsMaterialRegion *mechanics_regions;
  size_t mechanics_region_count;
  char geometry_id[SS_SHA256_HEX_CAPACITY + 8U];
  int restart_provenance_valid;
  char parent_checkpoint_id[SS_SHA256_HEX_CAPACITY + 8U];
  char parent_run_id[SS_SHA256_HEX_CAPACITY + 8U];
  char parent_build_id[SS_SHA256_HEX_CAPACITY + 8U];
  SsStateId parent_accepted_state_id;
  uint32_t parent_source_index_bits;
} SsSolverGraph;

typedef struct SsSolverCouplingSnapshot SsSolverCouplingSnapshot;

/* Internal topology/constraint contract, also exercised by compact fixtures. */
/** Flag the raw-mesh entities that lie on the exterior boundary; local. */
SsStatus ss_solver_mark_exterior_entities(SsSolverRawMesh *raw, uint32_t dimension);
/** Whether a resident entity belongs to a boundary rule's support; local. */
int ss_solver_rule_contains_entity(const SsSolverGraph *graph,
                                   const SsSolverBoundaryRule *rule,
                                   SsEntityId stable_entity_id);
/** Build the essential constraints of a field from the boundary rules; collective. The
 * caller frees the constraints. */
SsStatus ss_solver_build_constraints_for_field(const SsSolverGraph *graph,
                                               const char *field, uint32_t order,
                                               uint32_t components,
                                               SsStrongConstraint **constraints,
                                               size_t *constraint_count);
/** Read the inputs and build the distributed solver graph (mesh, spaces, plans);
 * collective. */
SsStatus ss_solver_graph_create(const SsSolverRequest *request, SsSolverGraph *graph,
                                SsErrorRecord *error);
/** Freeze the seepage pressure at the mechanics quadrature points into a coupling
 * snapshot; collective. */
SsStatus ss_solver_coupling_snapshot_build(SsSolverGraph *graph, Vec hydraulic_solution,
                                           int requested_threads,
                                           SsSolverCouplingSnapshot **snapshot,
                                           SsCouplingDiagnostics *diagnostics,
                                           SsErrorRecord *error);
/** Point body force of prescribed saturation (uncoupled water level); pure, safe in
 * OpenMP regions. */
SsStatus ss_solver_prescribed_mechanics_force(const void *context, uint32_t dimension,
                                              SsEntityId stable_cell_id,
                                              uint32_t quadrature_index,
                                              const double *physical,
                                              double *body_force);
/** Frozen coupled body force at a mechanics quadrature point; read-only, safe in OpenMP
 * regions. */
SsStatus ss_solver_coupling_snapshot_mechanics_force(
    const void *snapshot, uint32_t dimension, SsEntityId stable_cell_id,
    uint32_t quadrature_index, const double *physical, double *body_force);
/** Release a coupling snapshot; local. */
SsStatus ss_solver_coupling_snapshot_destroy(SsSolverCouplingSnapshot **snapshot);
/** Release the solver graph and everything it owns; collective. */
void ss_solver_graph_destroy(SsSolverGraph *graph);
/** Free the raw mesh payload buffers once the distributed mesh exists; local. */
void ss_solver_raw_release_payload(SsSolverRawMesh *raw);
/** Fill an error record with a solver code, message and component key. */
void ss_solver_error(SsErrorRecord *error, SsStatus status, const char *code,
                     const char *message, uint32_t component);

#define SS_SOLVER_DEFAULT_OPENMP_WORK_THRESHOLD UINT64_C(4096)
#define SS_SOLVER_ADAPTIVE_GUARD_SAMPLE_COUNT_INDEX 8U
#define SS_SOLVER_ADAPTIVE_GUARD_MAXIMUM_INDEX 9U
#define SS_SOLVER_ADAPTIVE_GUARD_VERSION_INDEX 10U
#define SS_SOLVER_ADAPTIVE_GUARD_CHECKPOINT_INDEX 11U
#define SS_SOLVER_ADAPTIVE_GUARD_MINIMUM_RATIO_INDEX 12U
#define SS_SOLVER_ADAPTIVE_GUARD_MAXIMUM_RATIO_INDEX 13U
#define SS_SOLVER_WORK_EFFICIENCY_ITERATIONS_INDEX 14U
#define SS_SOLVER_WORK_EFFICIENCY_ELASTICITY_INDEX 15U
#define SS_SOLVER_ADAPTIVE_GUARD_CHECKPOINT_VERSION 3.0

typedef struct {
  SsSolverGraph *graph;
  SsMechanicsOperatorSpec mechanics_spec;
  SsOperatorSpec operator_spec;
  SsNaturalOperatorSpec *natural_operations;
  size_t natural_operation_count;
  SsOperator *operator_handle;
  Vec reference_load;
  Mat elastic_reference;
  int matrix_free;
  int matrix_free_galerkin;
  size_t matrix_free_level_count;
  SsFieldSpace *matrix_free_spaces[SS_PMG_MAX_LEVELS - 1U];
  SsAssemblyPlan *matrix_free_plans[SS_PMG_MAX_LEVELS - 1U];
  SsOperator *matrix_free_operators[SS_PMG_MAX_LEVELS - 1U];
  Vec matrix_free_states[SS_PMG_MAX_LEVELS - 1U];
  Mat matrix_free_level_operators[SS_PMG_MAX_LEVELS];
  Mat matrix_free_diagonal_operators[SS_PMG_MAX_LEVELS - 1U];
  Mat matrix_free_injections[SS_PMG_MAX_LEVELS - 1U];
  SsPmg *pmg;
  SsDeflation *deflation;
  SsLinearRecycle *krylov_recycle;
  SsNonlinearProblem nonlinear;
  SsNonlinearPolicy nonlinear_policy;
  SsNonlinearPolicy initialization_nonlinear_policy;
  SsNonlinearProgressGuardMode progress_guard_mode;
  SsNonlinearAdaptiveProgressGuard adaptive_progress_guard;
  SsNonlinearAdaptiveProgressGuard adaptive_progress_guard_snapshot;
  uint64_t next_value_generation;
  uint64_t transaction_generation;
  uint64_t accepted_generation;
  int transaction_open;
  Vec checkpoint_previous_state;
  Vec checkpoint_current_state;
  size_t checkpoint_state_count;
  size_t checkpoint_first_ordinal;
  SsDeflationDiagnostics last_deflation_diagnostics;
  SsStatus last_transaction_status;
} SsSolverMechanics;

typedef struct {
  SsEntityId stable_cell_id;
  double conductivity;
  double epsilon;
} SsSolverHydraulicCell;

typedef struct {
  SsSolverGraph *graph;
  SsSolverHydraulicCell *cells;
  SsHydraulicCellCoefficients *coefficients;
  size_t cell_count;
  double gamma_water;
  SsHydraulicOperatorModel *model;
  SsScalarOperatorSpec scalar_spec;
  SsOperatorSpec operator_spec;
  SsNaturalOperatorSpec *natural_operations;
  SsHydraulicFluxModel *natural_models;
  size_t natural_operation_count;
  SsOperator *operator_handle;
  Vec reference_load;
  Mat elastic_reference;
  SsPmg *pmg;
  SsDeflation *deflation;
  SsNonlinearProblem nonlinear;
  SsNonlinearPolicy nonlinear_policy;
  SsHydraulicInitializationRecord initialization;
  SsNonlinearOutcome primary_outcome;
  int primary_outcome_valid;
} SsSolverHydraulic;

typedef struct {
  SsHydraulicBalance balance;
  double reconstructed_natural_flux;
  double reconstructed_dirichlet_flux;
  PetscReal initial_residual_norm;
  PetscReal final_residual_norm;
  PetscReal load_norm;
  SsNonlinearOutcome nonlinear_outcome;
  SsHydraulicInitializationRecord initialization;
  int valid;
} SsSolverHydraulicEvidence;

typedef struct {
  double reaction[SS_FE_MAX_DIMENSION];
  double external_resultant[SS_FE_MAX_DIMENSION];
  double reaction_norm;
  double external_resultant_norm;
  double equilibrium_imbalance_norm;
  double normalized_equilibrium_imbalance;
  double free_residual_l2;
  int assembly_minimum_threads;
  int assembly_maximum_threads;
  int valid;
} SsSolverMechanicsEquilibrium;

typedef struct {
  SsEntityId *ids;
  size_t *slots;
  double *values;
  size_t local_count;
  uint64_t global_count;
  uint32_t component_count;
  uint32_t sample_count;
  char quadrature_id[SS_RESULT_TEXT_CAPACITY];
} SsSolverOutputField;

typedef struct {
  SsEntityId *ids;
  size_t *slots;
  uint32_t *values;
  size_t local_count;
  uint64_t global_count;
  uint32_t category_count;
  uint32_t sample_count;
  char quadrature_id[SS_RESULT_TEXT_CAPACITY];
} SsSolverOutputCategory;

typedef struct {
  SsEntityId *stable_ids;
  size_t *local_slots;
  double *values;
  size_t local_count;
  uint64_t global_count;
  char field_space_id[SS_RESULT_TEXT_CAPACITY];
} SsSolverCheckpointFieldBuffer;

typedef struct {
  double *left_values;
  double *right_values;
  double *source_values;
  double *source_weights;
  double *source_kinds;
  size_t local_count;
  uint64_t global_count;
  uint32_t basis_count;
  uint32_t basis_capacity;
} SsSolverCheckpointDeflationBuffer;

typedef struct {
  SsSolverCheckpointFieldBuffer previous_displacement;
  SsSolverCheckpointFieldBuffer displacement;
  SsSolverCheckpointFieldBuffer pressure;
  SsSolverCheckpointDeflationBuffer deflation;
  double controller_values[SS_RESULT_CONTROLLER_CAPACITY];
  double coupling_values[SS_RESULT_COUPLING_CAPACITY];
  SsCheckpointPriorState prior_states[SS_CHECKPOINT_PRIOR_STATE_CAPACITY];
  SsCheckpointRestoreSummary summary;
} SsSolverCheckpointBuffer;

typedef struct {
  PetscBool nonlinear_allow_relaxed;
  PetscBool indirect_line_search_legacy_alg5;
  PetscBool indirect_adaptive_linear_tolerance;
  PetscInt nonlinear_maximum_line_search_trials;
  PetscInt indirect_adaptive_linear_iteration_threshold;
  PetscInt indirect_progress_guard_iteration;
  PetscReal indirect_progress_guard_maximum_convergence;
  PetscReal indirect_adaptive_linear_alpha_threshold;
  PetscReal indirect_adaptive_linear_tightening_factor;
  PetscReal indirect_adaptive_linear_minimum_relative_tolerance;
  SsNonlinearProgressGuardMode indirect_progress_guard_mode;
  PetscReal control_sensitivity_relative_step;
  SsNonlinearControlSensitivityMethod control_sensitivity_method;
} SsSolverTuning;

/* PETSc option readers and continuation tuning for the native solver */
/** Return the operator assembly mode requested by the resolved validation policy. */
SsAssemblyDeterminism ss_solver_assembly_determinism(const SsSolverRunPolicy *run);
/** Resolve the mechanics PMG, matrix-free and workspace tuning from settings and
 * options; local. */
SsStatus ss_solver_mechanics_tuning(const SsSolverGraph *graph,
                                    SsSolverMechanics *mechanics,
                                    SsSolverTuning *tuning);
/** Apply the resolved PETSc backend options of the run; master thread only. */
SsStatus ss_solver_backend_settings(const SsSolverRunPolicy *run);
/** Set the SuperLU_DIST reuse options of the distributed coarse LU in PETSc's options
 * database. */
SsStatus ss_solver_superlu_dist_settings(void);
/** Indirect-SSR sensitivity method and relative step from the typed settings. */
void ss_solver_indirect_sensitivity_options(const SsSolverSettings *settings,
                                            SsNonlinearControlSensitivityMethod *method,
                                            PetscReal *relative_step);
/** Whether a progress-guard mode learns from committed solves. */
int ss_solver_progress_guard_learns(SsNonlinearProgressGuardMode mode);
/** Resolve the indirect progress-guard mode, checkpoint and bound of a run. */
SsStatus ss_solver_indirect_progress_guard_options(const SsSolverRunPolicy *run,
                                                   SsNonlinearProgressGuardMode *mode,
                                                   PetscInt *checkpoint_iteration,
                                                   PetscReal *maximum_convergence);
/** Resolve the work-step controller options of indirect SSR. */
void ss_solver_indirect_work_step_options(const SsSolverRunPolicy *run,
                                          SsAnalysisWorkStepController *controller,
                                          PetscInt *growth_iteration_limit,
                                          PetscInt *efficiency_iteration_maximum,
                                          PetscReal *predictor_error_target,
                                          PetscReal *scale_minimum,
                                          PetscReal *scale_maximum);

/* Mechanics nodal fields, constitutive output, and equilibrium recovery */
/** Release the buffers of an output field. */
void ss_solver_output_field_destroy(SsSolverOutputField *field);
/** Release the buffers of an output category. */
void ss_solver_output_category_destroy(SsSolverOutputCategory *category);
/** Build a nodal output field from resident section values; local. */
SsStatus ss_solver_nodal_output(const SsSolverGraph *graph, SsFieldSpace *space,
                                uint32_t component_count,
                                const PetscScalar *local_values,
                                SsSolverOutputField *field);
/** Volume rule of an operation at a field order under the problem's quadrature mode. */
SsStatus ss_solver_volume_quadrature(const SsSolverGraph *graph,
                                     SsFeOperationKind operation, uint32_t field_order,
                                     const SsFeQuadrature **quadrature);
/** Evaluate stress, strain, yield and branch outputs at the quadrature points; local,
 * OpenMP inside. */
SsStatus ss_solver_stress_output(
    const SsSolverGraph *graph, const PetscScalar *local_values,
    double reduction_factor, SsSolverOutputField *stress_field,
    SsSolverOutputField *strain_field, SsSolverOutputField *yield_field,
    SsSolverOutputCategory *branch_category, SsSolverOutputCategory *plastic_category);
/** Recover reactions and the equilibrium residual of a mechanics state; collective. */
SsStatus ss_solver_mechanics_equilibrium(const SsSolverGraph *graph,
                                         SsSolverMechanics *mechanics,
                                         const PetscScalar *local_values, Vec solution,
                                         double factor,
                                         SsSolverMechanicsEquilibrium *equilibrium,
                                         SsErrorRecord *error);

/* Checkpoint buffers, controller state and deflation persistence adapters */
/** Release a checkpoint buffer. */
void ss_solver_checkpoint_buffer_destroy(SsSolverCheckpointBuffer *buffer);
/** Create a stable-identity buffer of a field vector, optionally copying its values;
 * collective. */
SsStatus ss_solver_checkpoint_field_buffer_create(const SsFieldSpace *space, Vec vector,
                                                  int capture,
                                                  SsSolverCheckpointFieldBuffer *field);
/** Copy a restored field buffer into a vector; local. */
SsStatus
ss_solver_checkpoint_field_copy_to_vector(const SsSolverCheckpointFieldBuffer *field,
                                          Vec vector);
/** Identity of the material model recorded in checkpoints. */
const char *ss_solver_checkpoint_material_model(const SsSolverGraph *graph);
/** Identity of the seepage model recorded in checkpoints. */
const char *ss_solver_checkpoint_hydraulic_model(const SsSolverGraph *graph);
/** Identity of the coupling model recorded in checkpoints. */
const char *ss_solver_checkpoint_coupling_model(const SsSolverGraph *graph);
/** Identity of the continuation controller recorded in checkpoints. */
const char *ss_solver_checkpoint_controller(const SsSolverGraph *graph);
/** Identity of the mechanics operator recorded in checkpoints. */
const char *ss_solver_checkpoint_operator(const SsSolverGraph *graph);
/** Identity of the checkpointed field space; collective. */
SsStatus ss_solver_checkpoint_field_space_id(const SsSolverGraph *graph,
                                             const SsSolverCheckpointBuffer *buffer,
                                             char output[SS_SHA256_HEX_CAPACITY + 8U]);
/** Copy the persistent deflation basis into a checkpoint buffer; collective. */
SsStatus ss_solver_checkpoint_deflation_capture(
    const SsSolverMechanics *mechanics,
    const SsSolverCheckpointFieldBuffer *displacement,
    SsSolverCheckpointDeflationBuffer *buffer);
/** Allocate the buffer that receives a restored deflation basis. */
SsStatus ss_solver_checkpoint_deflation_restore_buffer(
    const SsSolverGraph *graph, const SsSolverCheckpointFieldBuffer *displacement,
    SsSolverCheckpointDeflationBuffer *buffer);
/** Load a restored basis into the mechanics deflation store and refresh it; collective.
 */
SsStatus ss_solver_checkpoint_deflation_apply(SsSolverMechanics *mechanics,
                                              Vec template_vector,
                                              SsSolverCheckpointDeflationBuffer *buffer,
                                              uint32_t basis_count, PetscReal factor,
                                              uint64_t accepted_generation);

/* Capture and publish accepted solver checkpoint states */
/** Whether the analysis is at a boundary where a checkpoint may be written. */
int ss_solver_checkpoint_boundary_ready(const SsSolverGraph *graph,
                                        const SsAnalysisRecord *analysis_record);
typedef struct SsSolverBoundary SsSolverBoundary;
/** Destination and controller source of one accepted-boundary checkpoint. */
typedef struct SsSolverCheckpointTarget SsSolverCheckpointTarget;
/**
 * Collectively write the accepted-boundary checkpoint `state_id` into
 * `target->directory` (atomically).  With `target->boundary` the controller
 * state is that boundary snapshot (an in-run or cancellation checkpoint);
 * otherwise the analysis must have stopped at a resumable step limit.
 */
SsStatus ss_solver_checkpoint_write_state(
    const SsSolverRequest *request, const SsSolverGraph *graph,
    const SsSolverCheckpointTarget *target, Vec mechanics_solution,
    const SsAnalysisRecord *analysis_record, const SsSolverMechanics *mechanics,
    Vec hydraulic_solution, const SsSolverHydraulic *hydraulic,
    const SsNonlinearSolveRecord *hydraulic_solve,
    const SsSolverHydraulicEvidence *hydraulic_evidence, SsStateId state_id,
    SsCheckpointRestoreSummary *checkpoint_summary, SsErrorRecord *error);
/**
 * Write the accepted checkpoint into the result staging directory and index
 * it as `checkpoint-accepted`; `boundary` selects a boundary snapshot (used
 * for a cancelled run) instead of the final step-limit state.
 */
SsStatus ss_solver_checkpoint_write_accepted(
    const SsSolverRequest *request, const SsSolverGraph *graph, SsResultWriter *writer,
    Vec mechanics_solution, const SsAnalysisRecord *analysis_record,
    const SsSolverMechanics *mechanics, Vec hydraulic_solution,
    const SsSolverHydraulic *hydraulic, const SsNonlinearSolveRecord *hydraulic_solve,
    const SsSolverHydraulicEvidence *hydraulic_evidence,
    const SsSolverBoundary *boundary, SsStateId state_id, SsErrorRecord *error);

/* ---- Complete solver histories (solver_publish_history.c) ---- */

/** `parent_index` of an entry whose parent is its group's attempt row. */
#define SS_SOLVER_HISTORY_PARENT_GROUP UINT32_MAX

/** One history row with a parent relative to its attempt group. */
typedef struct {
  /** Attempt ordinal + 1; zero for solves outside continuation attempts. */
  uint64_t attempt_id;
  /** Index of the parent entry within the group, or the group marker. */
  uint32_t parent_index;
  uint32_t reserved;
  /** Row content; `row_id` and `parent_row_id` are assigned on replay. */
  SsResultHistoryRow row;
} SsSolverHistoryEntry;

typedef SsStatus (*SsSolverHistoryEmit)(void *context,
                                        const SsSolverHistoryEntry *entry);

/** Entry sink; `entry_count` indexes entries within the current group. */
typedef struct {
  SsSolverHistoryEmit emit;
  void *context;
  uint32_t entry_count;
} SsSolverHistoryEmitter;

/**
 * Rank-zero spool file of history entries in chronological order.
 * `written` counts entries on every rank; only rank zero owns `file`.
 */
typedef struct {
  MPI_Comm communicator;
  int rank;
  FILE *file;
  char path[SS_RESULT_PATH_CAPACITY];
  uint64_t written;
  uint64_t read;
  SsSolverHistoryEntry lookahead;
  int has_lookahead;
} SsSolverHistorySpool;

/**
 * Emit the complete per-iteration history of one nonlinear solve: one NEWTON
 * row per iteration with its LINEAR (main, control, residual), LINE_SEARCH
 * and refreshed PMG-setup children, plus an optional PMG apply-counter row.
 */
SsStatus ss_solver_history_solve(SsSolverHistoryEmitter *emitter,
                                 const SsNonlinearSolveRecord *solve,
                                 uint64_t attempt_id, uint32_t role,
                                 const SsPmgDiagnostics *pmg_delta);
/** Emit an already populated optional history row below the current attempt. */
SsStatus ss_solver_history_detail(SsSolverHistoryEmitter *emitter,
                                  const SsResultHistoryRow *row);
/** Emit one LINEAR row for a standalone Krylov solve (e.g. elastic predictor). */
SsStatus ss_solver_history_linear_solve(SsSolverHistoryEmitter *emitter,
                                        const SsLinearSolveRecord *solve,
                                        uint64_t attempt_id, uint32_t role);
/** Collectively open a rank-zero spool file at `path`. */
SsStatus ss_solver_history_spool_open(SsSolverHistorySpool *spool,
                                      MPI_Comm communicator, const char *path);
/** Emitter appending to the spool (other ranks only count entries). */
SsSolverHistoryEmitter ss_solver_history_spool_emitter(SsSolverHistorySpool *spool);
/** Collectively flush the spool so completed solves survive a forced stop. */
SsStatus ss_solver_history_spool_commit(SsSolverHistorySpool *spool);
/** Collectively rewind the spool for replay. */
SsStatus ss_solver_history_spool_rewind(SsSolverHistorySpool *spool);
/**
 * Collectively append the next spooled group with `attempt_id` to the writer
 * in bounded batches; group-level parents become `group_row`.
 */
SsStatus ss_solver_history_spool_replay(SsSolverHistorySpool *spool,
                                        SsResultWriter *writer, uint64_t attempt_id,
                                        uint64_t group_row, uint64_t *row_id);
/** Close the spool and optionally remove its file. */
SsStatus ss_solver_history_spool_close(SsSolverHistorySpool *spool, int remove_file);
/** Collectively append a solve's complete history directly to a writer. */
SsStatus ss_solver_history_append_solve(SsResultWriter *writer,
                                        const SsNonlinearSolveRecord *solve,
                                        uint64_t attempt_id, uint32_t role,
                                        uint64_t group_row, uint64_t *row_id);

/* ---- Accepted-state stream (solver_publish_spool.c) ---- */

/** Accepted states in a staged parallel HDF5 file, [state, global DOF]. */
typedef struct {
  MPI_Comm communicator;
  hid_t file;
  hid_t dataset;
  PetscInt local_offset;
  PetscInt local_size;
  PetscInt global_size;
  uint64_t count;
  int open;
  char path[SS_RESULT_PATH_CAPACITY];
} SsSolverStateSpool;

/** Collectively create the stream for vectors with `layout`'s distribution. */
SsStatus ss_solver_state_spool_open(SsSolverStateSpool *spool, Vec layout,
                                    const char *path);
/** Collectively append one accepted state. */
SsStatus ss_solver_state_spool_append(SsSolverStateSpool *spool, Vec state);
/** Collectively read accepted state `index` into `state`. */
SsStatus ss_solver_state_spool_read(SsSolverStateSpool *spool, uint64_t index,
                                    Vec state);
/** Collectively close the stream and optionally remove its file. */
SsStatus ss_solver_state_spool_close(SsSolverStateSpool *spool, int remove_file);

/* ---- Run monitor: journal, histories, cancellation, checkpoints ---- */

/** Controller state of the newest accepted boundary (resume point). */
struct SsSolverBoundary {
  size_t point_count;
  PetscReal next_step;
  PetscReal first_control;
  SsAnalysisNextOperation next_operation;
  PetscInt work_efficiency_previous_iterations;
  PetscReal work_efficiency_elasticity;
  int work_efficiency_elasticity_ready;
};

struct SsSolverCheckpointTarget {
  /** Final checkpoint directory. */
  const char *directory;
  uint64_t history_cursor;
  /** Accepted-state cursor of a seepage-only checkpoint. */
  uint64_t accepted_state_count;
  /** Boundary snapshot, or NULL for the final step-limit state. */
  const SsSolverBoundary *boundary;
};

/** Observation and recovery state of one native run. */
typedef struct {
  const SsSolverRequest *request;
  const SsSolverGraph *graph;
  MPI_Comm communicator;
  int rank;
  SsSolverRunOptions options;
  char run_directory[SS_RESULT_PATH_CAPACITY];
  SsProgressJournal *journal;
  SsSolverHistorySpool history;
  int history_open;
  SsSolverHistoryEmitter emitter;
  uint64_t emitter_attempt;
  uint64_t detail_attempt;
  uint64_t krylov_details;
  uint64_t pmg_details;
  uint64_t linear_sequence;
  uint64_t newton_ordinal;
  uint32_t newton_role;
  uint32_t linear_role;
  SsStatus detail_status;
  SsSolverStateSpool states;
  /* Mechanics analysis interposition. */
  SsSolverMechanics *mechanics;
  SsAnalysisOps base_operations;
  void *base_context;
  SsPmgDiagnostics pmg_last;
  SsSolverHydraulic *hydraulic;
  Vec hydraulic_solution;
  const SsSolverHydraulicEvidence *hydraulic_evidence;
  /* Accepted boundaries and in-run checkpoints. */
  SsSolverBoundary boundary;
  int boundary_valid;
  size_t checkpointed_points;
  uint64_t checkpoint_count;
  double last_checkpoint_time;
  uint64_t accepted_states;
  uint64_t attempts;
  int cancel_observed;
  /* Test hook (-native_test_signal_after_accepted N): raise SIGTERM on rank 0. */
  PetscInt signal_after_accepted;
} SsSolverMonitor;

/** Collectively prepare the run directory and open the journal. */
SsStatus ss_solver_monitor_open(SsSolverMonitor *monitor,
                                const SsSolverRequest *request,
                                const SsSolverGraph *graph);
/** Journal a phase transition (rank zero). */
void ss_solver_monitor_phase(SsSolverMonitor *monitor, const char *phase);
/** Install live observations and the cancellation poll on this physics stage. */
void ss_solver_monitor_watch_nonlinear(SsSolverMonitor *monitor,
                                       SsNonlinearProblem *problem, uint32_t role);
/** Attach the run monitor to the seepage solve observers. */
void ss_solver_monitor_watch_hydraulic(SsSolverMonitor *monitor,
                                       SsSolverHydraulic *hydraulic);
/** Nonzero when SIGINT/SIGTERM was received by this process (not collective). */
/** Cancellation operation for analysis and nonlinear problems (process-local). */
int ss_solver_cancellation_poll(void *context);
/**
 * Collectively wrap a mechanics analysis problem: interpose its operations
 * (accepted-state stream, checkpoint vectors, cancellation, attempt journal),
 * install the solve and boundary observers and open the history spool.
 */
SsStatus ss_solver_monitor_attach(SsSolverMonitor *monitor,
                                  SsSolverMechanics *mechanics,
                                  SsSolverHydraulic *hydraulic, Vec hydraulic_solution,
                                  const SsSolverHydraulicEvidence *hydraulic_evidence,
                                  SsAnalysisProblem *problem);
/** Collectively record a solve outside continuation attempts (attempt 0). */
SsStatus ss_solver_monitor_record_solve(SsSolverMonitor *monitor,
                                        const SsNonlinearSolveRecord *solve,
                                        uint32_t role);
/** Collectively record a standalone Krylov solve (attempt 0). */
SsStatus ss_solver_monitor_record_linear(SsSolverMonitor *monitor,
                                         const SsLinearSolveRecord *solve,
                                         uint32_t role);
/** Journal the terminal status and release spools and the journal. */
SsStatus ss_solver_monitor_close(SsSolverMonitor *monitor, SsStatus status,
                                 const char *outcome);

/* ---- In-run checkpoints (solver_checkpoint_boundary.c) ---- */

/** Collectively write a checkpoint of the newest boundary into the run directory. */
SsStatus ss_solver_checkpoint_boundary_write(SsSolverMonitor *monitor,
                                             const SsAnalysisRecord *record);

/* Shared native result declarations, scalars and history publication */
/** Point a result field view at an output field for one state. */
void ss_solver_output_field_bind(SsResultFieldView *view,
                                 const SsSolverOutputField *field, SsStateId state_id);
/** Point a result category view at an output category for one state. */
void ss_solver_output_category_bind(SsResultCategoryView *view,
                                    const SsSolverOutputCategory *category,
                                    SsStateId state_id);
/** Whether the output profile requests a quantity. */
int ss_solver_output_requested(const SsSolverGraph *graph, uint64_t quantity);
/** Result declaration matching a field view. */
SsResultQuantityDeclaration
ss_solver_quantity_declaration(const SsResultFieldView *view);
/** Result declaration matching a category view. */
SsResultCategoryDeclaration
ss_solver_category_declaration(const SsResultCategoryView *view);
/** Protocol name of an analysis stop reason. */
const char *ss_solver_analysis_stop_reason(SsAnalysisStopReason reason);
/**
 * Result-contract outcome family of a finished mechanics analysis: the stop
 * reason (criterion, control horizon, requested step horizon, failure or
 * cancellation) with "accepted_relaxed" taking precedence over a completed
 * outcome when any committed attempt or the coupled seepage stage was
 * accepted under the relaxed criterion.
 */
const char *ss_solver_analysis_outcome(const SsAnalysisRecord *record,
                                       int hydraulic_relaxed);
/** Result-contract outcome family of a single nonlinear (seepage) solve. */
const char *ss_solver_nonlinear_outcome(SsNonlinearOutcome outcome);
/** Collectively record the outcome and effective mechanics assembly mode. */
SsStatus ss_solver_set_result_outcome(SsResultWriter *writer,
                                      const SsSolverRequest *request,
                                      const char *outcome,
                                      const char *mechanics_assembly_mode);
/** Append one analysis scalar to the result; collective. */
SsStatus ss_solver_append_analysis_scalar(SsResultWriter *writer,
                                          const char *quantity_id, const char *unit,
                                          SsStateId state_id, double value);
/** Append the seepage and coupling diagnostics of a coupled run; collective. */
SsStatus
ss_solver_append_coupled_diagnostics(SsResultWriter *writer, MPI_Comm communicator,
                                     SsStateId state_id, uint32_t dimension,
                                     const SsSolverHydraulicEvidence *hydraulic,
                                     const SsCouplingDiagnostics *coupling);
/**
 * Collectively append the relational history of an analysis.  With a spool
 * the complete per-iteration rows replace the per-attempt summaries.
 */
SsStatus ss_solver_append_analysis_history(
    SsResultWriter *writer, const SsAnalysisRecord *record,
    const SsDeflation *deflation, uint64_t hydraulic_generation,
    const SsSolverHydraulicEvidence *hydraulic_evidence,
    const SsCouplingDiagnostics *coupling_diagnostics, SsSolverHistorySpool *spool);

/* Mechanics operators, callbacks, and solver lifecycle */
/** Assemble the mechanics residual, tangent and functional at factors; collective. */
SsStatus ss_solver_mechanics_evaluate_common(SsSolverMechanics *mechanics, Vec state,
                                             PetscReal reduction_factor,
                                             PetscReal external_load_factor,
                                             Vec residual, Mat tangent,
                                             PetscReal *functional);
/** Check the Krylov and deflation vector payload against the memory cap; collective. */
SsStatus ss_solver_admit_linear_vectors(const SsSolverGraph *graph, int hydraulic,
                                        SsErrorRecord *error);
/** Linear policy chain of the mechanics or seepage solve. */
SsStatus ss_solver_linear_policy(const SsSolverGraph *graph, int hydraulic,
                                 SsLinearPolicyChain *chain);
/** Create the mechanics deflation store from the run policy; collective. */
SsStatus ss_solver_deflation_create(const SsSolverGraph *graph,
                                    SsDeflation **deflation);
/** Linearize the matrix-free tangent and refresh its PMG coarse data; collective. */
SsStatus ss_solver_mechanics_matrix_free_linearize_common(
    SsSolverMechanics *mechanics, Vec fine_state, PetscReal reduction_factor,
    PetscReal external_load_factor, PetscReal elastic_regularization);
/** Nonlinear linearize callback of fixed-factor matrix-free mechanics. */
SsStatus ss_solver_mechanics_matrix_free_linearize(void *opaque, Vec state,
                                                   PetscReal factor,
                                                   PetscReal elastic_regularization);
/** Nonlinear linearize callback of work-controlled matrix-free mechanics. */
SsStatus ss_solver_mechanics_matrix_free_controlled_linearize(
    void *opaque, SsNonlinearControlMode mode, Vec state, PetscReal control,
    PetscReal elastic_regularization);
/** Create the mechanics operators, PMG, deflation and Newton workspace; collective. */
SsStatus ss_solver_mechanics_create(SsSolverGraph *graph, int requested_threads,
                                    const SsSolverCouplingSnapshot *coupling_snapshot,
                                    SsSolverMechanics *mechanics, SsErrorRecord *error);
/** Release the mechanics solver context; collective. */
SsStatus ss_solver_mechanics_destroy(SsSolverMechanics *mechanics);

/* Mechanics continuation policy and accepted-attempt callbacks */
/** Fill the continuation problem callbacks of the mechanics context. */
void ss_solver_mechanics_analysis_problem(SsSolverMechanics *mechanics,
                                          SsAnalysisProblem *problem);
/** Continuation policy of the run scaled by the elastic reference work. */
SsStatus ss_solver_mechanics_analysis_policy(const SsSolverGraph *graph,
                                             PetscReal reference_work,
                                             SsAnalysisPolicy *policy);

/* Field constraints, near-nullspace and PMG hierarchy setup */
/** Create a PMG hierarchy (spaces, constraints, transfers, near-nullspaces) for a
 * field; collective. */
SsStatus ss_solver_pmg_create(const SsSolverGraph *graph, SsFieldSpace *fine_space,
                              uint32_t fine_order, uint32_t components, int hydraulic,
                              const SsStrongConstraint *fine_constraints,
                              size_t fine_constraint_count, const char *options_stem,
                              SsFieldSpace **retained_coarse_spaces, Mat *injections,
                              SsPmg **pmg, SsErrorRecord *error);
/** Create the matrix-free PMG level operators of mechanics; collective. */
SsStatus ss_solver_mechanics_matrix_free_hierarchy_create(SsSolverMechanics *mechanics,
                                                          SsErrorRecord *error);

/* Hydraulic boundary data, operator lifecycle, initialization and solve */
/** Seepage coefficients of a resident cell, or NULL. */
const SsSolverHydraulicCell *
ss_solver_hydraulic_cell_find(const SsSolverHydraulic *hydraulic,
                              SsEntityId stable_cell_id);
/** Create the seepage operator, boundary data and solver context; collective. */
SsStatus ss_solver_hydraulic_create(SsSolverGraph *graph, int requested_threads,
                                    SsSolverHydraulic *hydraulic, SsErrorRecord *error);
/** Release the seepage solver context; collective. */
SsStatus ss_solver_hydraulic_destroy(SsSolverHydraulic *hydraulic);
/** Solve the steady unconfined seepage problem with initialization fallback;
 * collective. */
SsStatus ss_solver_hydraulic_solve(const SsSolverGraph *graph,
                                   SsSolverHydraulic *hydraulic, Vec initial,
                                   Vec solution, SsNonlinearSolveRecord *record,
                                   SsErrorRecord *error);

/* Vertex coordinates in reference cell order for solver and coupling */
/** Vertex coordinates of a packed cell in reference order. */
SsStatus ss_solver_cell_vertex_coordinates(const SsCellPack *pack,
                                           const SsFeDescriptor *coordinate_descriptor,
                                           double *vertex_coordinates);

/* Stable ID projections for field nodes and coupling quadrature points */
/** Stable identity of a coupling quadrature point. */
SsStatus ss_solver_coupling_point_id(uint32_t dimension, SsEntityId stable_cell_id,
                                     uint32_t quadrature_index,
                                     SsEntityId *stable_point_id);
/** Stable identity of a field node from its entity and slot. */
SsStatus ss_solver_field_node_id(uint32_t dimension, uint32_t order,
                                 SsEntityId entity_id, uint32_t entity_slot,
                                 SsEntityId *stable_id);

/* Hydraulic balance, quadrature data and result acceptance */
/** Rank-local flux balance terms and reconstructed boundary fluxes of a seepage
 * solution. */
SsStatus ss_solver_hydraulic_boundary_balance(const SsSolverGraph *graph,
                                              const SsSolverHydraulic *hydraulic,
                                              const PetscScalar *local_values,
                                              SsHydraulicBalanceLocal *local_balance,
                                              double *reconstructed_natural_flux,
                                              double *reconstructed_dirichlet_flux);
/** Seepage gradient, conductivity and saturation outputs at quadrature points; local.
 */
SsStatus ss_solver_hydraulic_quadrature_output(
    const SsSolverGraph *graph, const SsSolverHydraulic *hydraulic,
    const PetscScalar *local_values, SsSolverOutputField *gradient,
    SsSolverOutputField *conductivity, SsSolverOutputField *ramp,
    SsSolverOutputField *epsilon, SsSolverOutputField *pressure_flux,
    SsSolverOutputField *darcy_velocity, SsSolverOutputCategory *saturation);
/** Coupled body-force output field from a snapshot; local. */
SsStatus ss_solver_coupled_body_force_output(const SsSolverGraph *graph,
                                             const SsSolverCouplingSnapshot *snapshot,
                                             SsSolverOutputField *field);
/** Check convergence and flux balance of a seepage solution; collective. */
SsStatus ss_solver_hydraulic_acceptance(const SsSolverGraph *graph,
                                        SsSolverHydraulic *hydraulic, Vec solution,
                                        const SsNonlinearSolveRecord *solve_record,
                                        SsSolverHydraulicEvidence *evidence,
                                        SsErrorRecord *error);

/* Publish accepted mechanics fields and continuation diagnostics */
/** Publish the mechanics result directory; collective. */
SsStatus ss_solver_publish_mechanics(
    const SsSolverRequest *request, const SsSolverGraph *graph, Vec solution,
    const SsAnalysisRecord *analysis_record, SsSolverMechanics *mechanics,
    SsSolverHydraulic *hydraulic, Vec hydraulic_solution,
    const SsSolverHydraulicEvidence *hydraulic_evidence,
    const SsSolverCouplingSnapshot *coupling_snapshot,
    const SsCouplingDiagnostics *coupling_diagnostics, SsSolverMonitor *monitor,
    SsSolverSummary *summary, SsErrorRecord *error);

/* Publish accepted hydraulic fields and diagnostics */
/** Publish the seepage-only result directory; collective. */
SsStatus ss_solver_publish_hydraulic(const SsSolverRequest *request,
                                     const SsSolverGraph *graph,
                                     SsSolverHydraulic *hydraulic, Vec solution,
                                     const SsNonlinearSolveRecord *solve_record,
                                     SsSolverMonitor *monitor, SsSolverSummary *summary,
                                     SsErrorRecord *error);

/* Restore accepted state into hydraulics, mechanics and continuation */
/** Restore an accepted-state checkpoint into the solver graph and monitor; collective.
 */
SsStatus ss_solver_checkpoint_restore_accepted(const SsSolverRequest *request,
                                               SsSolverGraph *graph,
                                               SsSolverMonitor *monitor,
                                               SsSolverSummary *summary,
                                               SsErrorRecord *error);

#endif
