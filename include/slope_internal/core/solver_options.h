/** @file
 * @brief Typed solver settings and registered expert options.
 *
 * `solver_options.def` is the single registry: each `SS_STABLE` row is a
 * typed field of the resolved run, and each `SS_EXPERT` row is a PETSc
 * option read by the engine. The generator `slope_codegen.solver_options`
 * renders the same registry into `docs/solver-options.md` and the Python
 * models.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_CORE_SOLVER_OPTIONS_H
#define SS_CORE_SOLVER_OPTIONS_H

#include "slope_internal/core/status.h"

#include <petscsys.h>

/** Value of one registered option; the member matches the option type. */
typedef union {
  PetscBool boolean; /**< `BOOL` option. */
  PetscInt integer;  /**< `INT` or `CHOICE` (index into the choices) option. */
  PetscReal real;    /**< `REAL` option. */
} SsSolverOptionValue;

/** @cond SS_XMACRO */
#define SS_OPT_C_BOOL PetscBool
#define SS_OPT_C_INT PetscInt
#define SS_OPT_C_CHOICE PetscInt
#define SS_OPT_C_REAL PetscReal
/** @endcond */

/** Stable settings of a resolved run: one member `<section>_<name>` per
 * `SS_STABLE` row of `solver_options.def`, with the row's C type. */
typedef struct {
/** @cond SS_XMACRO */
#define SS_STABLE(type, section, name, fallback, minimum, maximum, choices, doc)       \
  SS_OPT_C_##type section##_##name;
#define SS_EXPERT(type, name, option, fallback, minimum, maximum, choices, doc)
#include "solver_options.def"
#undef SS_EXPERT
#undef SS_STABLE
  /** @endcond */
} SsSolverSettings;
/** @cond SS_XMACRO */
#undef SS_OPT_C_REAL
#undef SS_OPT_C_CHOICE
#undef SS_OPT_C_INT
#undef SS_OPT_C_BOOL
/** @endcond */

/** Expert options: one enumerator `SS_SOLVER_EXPERT_<name>` per `SS_EXPERT`
 * row of `solver_options.def`, followed by the count. */
typedef enum {
/** @cond SS_XMACRO */
#define SS_STABLE(type, section, name, fallback, minimum, maximum, choices, doc)
#define SS_EXPERT(type, name, option, fallback, minimum, maximum, choices, doc)        \
  SS_SOLVER_EXPERT_##name,
#include "solver_options.def"
#undef SS_EXPERT
#undef SS_STABLE
  /** @endcond */
  SS_SOLVER_EXPERT_COUNT /**< Number of expert options. */
} SsSolverExpertOption;

struct yyjson_val;

/** Read the typed settings of a verified resolved run.
 *
 * Omitted fields take their registry defaults; present fields must have the
 * registered type and lie within the inclusive bounds.
 *
 * @param[in] root Root object of `resolved-run.json`.
 * @param[out] settings Settings with defaults applied.
 * @param[out] name Name of the first invalid field on failure.
 * @return ::SS_OK, ::SS_ERROR_SCHEMA for an invalid field, or
 *   ::SS_ERROR_INVALID_ARGUMENT. */
SsStatus ss_solver_settings_read(struct yyjson_val *root, SsSolverSettings *settings,
                                 const char **name);

/** Read a registered expert option from the PETSc options database.
 *
 * Uses the registered default when the option is absent. Reads the global
 * PETSc options database; call it from the master thread.
 *
 * @param[in] option Option to read.
 * @param[out] value Typed value.
 * @return ::SS_OK, or ::SS_ERROR_INVALID_ARGUMENT for an unknown option, a
 *   malformed value or a value outside the registered bounds. */
SsStatus ss_solver_expert_option(SsSolverExpertOption option,
                                 SsSolverOptionValue *value);

/** Validate every expert option before the solver graph is built.
 * @param[out] name Name of the first invalid option on failure.
 * @return ::SS_OK or the status of the first invalid option. */
SsStatus ss_solver_expert_options_validate(const char **name);

/** Convert a cap in MiB into bytes.
 * @param[in] mib Positive cap in MiB.
 * @param[out] bytes Cap in bytes.
 * @return ::SS_OK, ::SS_ERROR_INVALID_ARGUMENT for a nonpositive cap, or
 *   ::SS_ERROR_OVERFLOW. */
SsStatus ss_solver_cap_bytes(PetscInt mib, size_t *bytes);

#endif
