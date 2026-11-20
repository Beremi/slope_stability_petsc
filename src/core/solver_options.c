/* Registry and shared parsing of solver-owned run settings.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/core/solver_options.h"
#include "slope_internal/core/core.h"

#include "yyjson.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

typedef enum { SS_OPT_BOOL, SS_OPT_INT, SS_OPT_REAL, SS_OPT_CHOICE } SsOptionType;

typedef struct {
  const char *section;
  const char *key;
  const char *name;
  SsOptionType type;
  size_t offset;
  double fallback;
  double minimum;
  double maximum;
  const char *choices;
  const char *description;
  const char *stability;
} SsSolverOption;

/* Expert rows are indexed by SsSolverExpertOption. Stable rows follow. */
static const SsSolverOption ss_solver_options[] = {
#define SS_STABLE(type, section, name, fallback, minimum, maximum, choices, doc)
#define SS_EXPERT(type, name, option, fallback, minimum, maximum, choices, doc)        \
  [SS_SOLVER_EXPERT_##name] = {NULL,    option,   option,  SS_OPT_##type,              \
                               0U,      fallback, minimum, maximum,                    \
                               choices, doc,      "expert"},
#include "slope_internal/core/solver_options.def"
#undef SS_EXPERT
#undef SS_STABLE
#define SS_STABLE(type, section, name, fallback, minimum, maximum, choices, doc)       \
  {#section,                                                                           \
   #name,                                                                              \
   #section "." #name,                                                                 \
   SS_OPT_##type,                                                                      \
   offsetof(SsSolverSettings, section##_##name),                                       \
   fallback,                                                                           \
   minimum,                                                                            \
   maximum,                                                                            \
   choices,                                                                            \
   doc,                                                                                \
   "stable"},
#define SS_EXPERT(type, name, option, fallback, minimum, maximum, choices, doc)
#include "slope_internal/core/solver_options.def"
#undef SS_EXPERT
#undef SS_STABLE
};

static int ss_option_choice(const char *choices, const char *value, size_t size,
                            double *selected) {
  *selected = 0.0;
  if (value == NULL) {
    return 0;
  }
  for (const char *choice = choices;;) {
    const char *end = strchr(choice, '|');
    const size_t length = end == NULL ? strlen(choice) : (size_t)(end - choice);
    if (size == length && memcmp(choice, value, length) == 0) {
      return 1;
    }
    if (end == NULL) {
      return 0;
    }
    choice = end + 1;
    *selected += 1.0;
  }
}

static int ss_option_in_range(const SsSolverOption *option, double value) {
  return isfinite(value) && value >= option->minimum && value <= option->maximum;
}

static int ss_option_json_value(const SsSolverOption *option, yyjson_val *value,
                                double *selected) {
  *selected = option->fallback;
  if (value == NULL) {
    return 1;
  }
  switch (option->type) {
  case SS_OPT_BOOL:
    if (!yyjson_is_bool(value)) {
      return 0;
    }
    *selected = yyjson_get_bool(value) ? 1.0 : 0.0;
    break;
  case SS_OPT_INT:
    if (!yyjson_is_int(value)) {
      return 0;
    }
    *selected = yyjson_get_num(value);
    break;
  case SS_OPT_REAL:
    if (!yyjson_is_num(value)) {
      return 0;
    }
    *selected = yyjson_get_num(value);
    break;
  case SS_OPT_CHOICE:
    return ss_option_choice(option->choices, yyjson_get_str(value),
                            yyjson_get_len(value), selected);
  }
  return ss_option_in_range(option, *selected);
}

SsStatus ss_solver_settings_read(yyjson_val *root, SsSolverSettings *settings,
                                 const char **name) {
  if (root == NULL || settings == NULL || name == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (size_t index = SS_SOLVER_EXPERT_COUNT;
       index < sizeof(ss_solver_options) / sizeof(ss_solver_options[0]); ++index) {
    const SsSolverOption *option = &ss_solver_options[index];
    yyjson_val *section = yyjson_obj_get(root, option->section);
    yyjson_val *value = yyjson_obj_get(section, option->key);
    unsigned char *field = (unsigned char *)settings + option->offset;
    double selected = 0.0;
    if (!ss_option_json_value(option, value, &selected)) {
      *name = option->name;
      return SS_ERROR_SCHEMA;
    }
    switch (option->type) {
    case SS_OPT_BOOL:
      *(PetscBool *)(void *)field = selected != 0.0 ? PETSC_TRUE : PETSC_FALSE;
      break;
    case SS_OPT_INT:
    case SS_OPT_CHOICE:
      *(PetscInt *)(void *)field = (PetscInt)selected;
      break;
    case SS_OPT_REAL:
      *(PetscReal *)(void *)field = (PetscReal)selected;
      break;
    }
  }
  return SS_OK;
}

SsStatus ss_solver_expert_option(SsSolverExpertOption id, SsSolverOptionValue *value) {
  if (id < 0 || id >= SS_SOLVER_EXPERT_COUNT || value == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  const SsSolverOption *option = &ss_solver_options[id];
  double selected;
  PetscErrorCode ierr;
  if (option->type == SS_OPT_BOOL) {
    value->boolean = option->fallback != 0.0 ? PETSC_TRUE : PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL, NULL, option->key, &value->boolean, NULL);
    selected = (double)value->boolean;
  } else {
    value->integer = (PetscInt)option->fallback;
    ierr = PetscOptionsGetInt(NULL, NULL, option->key, &value->integer, NULL);
    selected = (double)value->integer;
  }
  return ierr == PETSC_SUCCESS && ss_option_in_range(option, selected)
             ? SS_OK
             : SS_ERROR_INVALID_ARGUMENT;
}

SsStatus ss_solver_expert_options_validate(const char **name) {
  for (SsSolverExpertOption id = 0; id < SS_SOLVER_EXPERT_COUNT; ++id) {
    SsSolverOptionValue unused;
    const SsStatus status = ss_solver_expert_option(id, &unused);
    if (status != SS_OK) {
      *name = ss_solver_options[id].name;
      return status;
    }
  }
  return SS_OK;
}

SsStatus ss_solver_cap_bytes(PetscInt mib, size_t *bytes) {
  if (mib < 1) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  return ss_checked_mul_size((size_t)mib, 1024U * 1024U, bytes);
}
