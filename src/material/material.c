/* Mohr-Coulomb parameters with Davis A/B/C strength reduction, return
 * mapping, material branches and the consistent tangent.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/material/material.h"

#include "slope_internal/core/sha256.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SS_MC_PI_OVER_TWO 1.57079632679489661923132169163975144

struct SsMcParameters {
  SsMcParameterSummary summary;
};

static void ss_mc_error_clear(SsMcError *error) {
  if (error != NULL) {
    memset(error, 0, sizeof(*error));
  }
}

static SsStatus ss_mc_error_set(SsMcError *error, SsStatus status, const char *code) {
  if (error != NULL) {
    memset(error, 0, sizeof(*error));
    error->valid = 1;
    error->status = status;
    (void)snprintf(error->code, sizeof(error->code), "%s", code);
  }
  return status;
}

static int ss_mc_valid_digest(const char *value) {
  size_t index;
  if (value == NULL || strlen(value) != SS_MC_ID_CAPACITY - 1U ||
      strncmp(value, "sha256:", 7U) != 0) {
    return 0;
  }
  for (index = 7U; index < SS_MC_ID_CAPACITY - 1U; ++index) {
    const char character = value[index];
    if (!((character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f'))) {
      return 0;
    }
  }
  return 1;
}

static SsStatus ss_mc_parameter_identity(const SsMcParameterSummary *summary,
                                         char identity[SS_MC_ID_CAPACITY]) {
  char canonical[512];
  char digest[SS_SHA256_HEX_CAPACITY];
  int count;
  count = snprintf(
      canonical, sizeof(canonical),
      "ss-mc-v1|E=%a|nu=%a|c=%a|phi=%a|psi=%a|davis=%u|"
      "region=%llu|provenance=%s",
      summary->young_modulus, summary->poisson_ratio, summary->cohesion,
      summary->friction_angle, summary->dilatancy_angle, (unsigned int)summary->davis,
      (unsigned long long)summary->material_region_id, summary->provenance_digest);
  if (count < 0 || (size_t)count >= sizeof(canonical)) {
    return SS_ERROR_OVERFLOW;
  }
  if (ss_sha256_bytes((const unsigned char *)canonical, (size_t)count, digest) !=
      SS_OK) {
    return SS_ERROR_INTERNAL;
  }
  count = snprintf(identity, SS_MC_ID_CAPACITY, "sha256:%s", digest);
  return count == (int)(SS_MC_ID_CAPACITY - 1U) ? SS_OK : SS_ERROR_INTERNAL;
}

SsStatus ss_mc_parameters_create(const SsMcParameterInput *input,
                                 SsMcParameters **parameters, SsMcError *error) {
  SsMcParameters *candidate;
  SsMcParameterSummary summary;
  double davis_denominator;
  if (parameters == NULL) {
    return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_PARAMETER_OUTPUT");
  }
  *parameters = NULL;
  ss_mc_error_clear(error);
  if (input == NULL) {
    return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_PARAMETER_INPUT");
  }
  if (!isfinite(input->young_modulus) || !isfinite(input->poisson_ratio) ||
      !isfinite(input->cohesion) || !isfinite(input->friction_angle) ||
      !isfinite(input->dilatancy_angle)) {
    return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_NONFINITE_PARAMETER");
  }
  if (!(input->young_modulus > 0.0)) {
    return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_YOUNG_MODULUS");
  }
  if (!(input->poisson_ratio > -1.0 && input->poisson_ratio < 0.5)) {
    return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_POISSON_RATIO");
  }
  if (input->cohesion < 0.0) {
    return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_COHESION");
  }
  if (!(input->friction_angle >= 0.0 && input->friction_angle < SS_MC_PI_OVER_TWO)) {
    return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_FRICTION_ANGLE");
  }
  if (!(input->dilatancy_angle >= 0.0 &&
        input->dilatancy_angle <= input->friction_angle)) {
    return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_DILATANCY_ANGLE");
  }
  if (input->davis != SS_MC_DAVIS_A && input->davis != SS_MC_DAVIS_B &&
      input->davis != SS_MC_DAVIS_C) {
    return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_DAVIS_METHOD");
  }
  if (!ss_mc_valid_digest(input->provenance_digest)) {
    return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_PROVENANCE_ID");
  }
  davis_denominator = 1.0 - sin(input->friction_angle) * sin(input->dilatancy_angle);
  if (!(davis_denominator > 64.0 * DBL_EPSILON *
                                fmax(1.0, fabs(sin(input->friction_angle) *
                                               sin(input->dilatancy_angle))))) {
    return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_DAVIS_DENOMINATOR");
  }

  memset(&summary, 0, sizeof(summary));
  summary.young_modulus = input->young_modulus;
  summary.poisson_ratio = input->poisson_ratio;
  summary.cohesion = input->cohesion;
  summary.friction_angle = input->friction_angle;
  summary.dilatancy_angle = input->dilatancy_angle;
  summary.shear_modulus = input->young_modulus / (2.0 * (1.0 + input->poisson_ratio));
  summary.bulk_modulus =
      input->young_modulus / (3.0 * (1.0 - 2.0 * input->poisson_ratio));
  summary.lame_first = summary.bulk_modulus - (2.0 / 3.0) * summary.shear_modulus;
  summary.davis = input->davis;
  summary.material_region_id = input->material_region_id;
  (void)snprintf(summary.provenance_digest, sizeof(summary.provenance_digest), "%s",
                 input->provenance_digest);
  if (!isfinite(summary.shear_modulus) || !isfinite(summary.bulk_modulus) ||
      !isfinite(summary.lame_first) ||
      ss_mc_parameter_identity(&summary, summary.identity) != SS_OK) {
    return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_ELASTIC_CONSTANT");
  }
  candidate = (SsMcParameters *)calloc(1U, sizeof(*candidate));
  if (candidate == NULL) {
    return ss_mc_error_set(error, SS_ERROR_OUT_OF_MEMORY, "MC_PARAMETER_ALLOCATION");
  }
  candidate->summary = summary;
  *parameters = candidate;
  return SS_OK;
}

SsStatus ss_mc_parameters_summary(const SsMcParameters *parameters,
                                  SsMcParameterSummary *summary) {
  if (parameters == NULL || summary == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  *summary = parameters->summary;
  return SS_OK;
}

SsStatus ss_mc_parameters_destroy(SsMcParameters **parameters) {
  if (parameters == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  free(*parameters);
  *parameters = NULL;
  return SS_OK;
}

SsStatus ss_mc_reduce(const SsMcParameters *parameters, double reduction_factor,
                      SsMcReduced *reduced, SsMcError *error) {
  SsMcReduced candidate;
  const SsMcParameterSummary *summary;
  double beta;
  double denominator;
  double reduced_cohesion;
  double reduced_friction;
  double phi_one;
  double psi_one;
  if (parameters == NULL || reduced == NULL) {
    return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_REDUCTION_ARGUMENT");
  }
  ss_mc_error_clear(error);
  if (!isfinite(reduction_factor) || !(reduction_factor > 0.0)) {
    return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_REDUCTION_FACTOR");
  }
  summary = &parameters->summary;
  memset(&candidate, 0, sizeof(candidate));
  beta = 1.0;
  if (summary->davis == SS_MC_DAVIS_A) {
    denominator = 1.0 - sin(summary->friction_angle) * sin(summary->dilatancy_angle);
    if (!(denominator > 64.0 * DBL_EPSILON *
                            fmax(1.0, fabs(sin(summary->friction_angle) *
                                           sin(summary->dilatancy_angle))))) {
      return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_DAVIS_DENOMINATOR");
    }
    beta = cos(summary->friction_angle) * cos(summary->dilatancy_angle) / denominator;
    reduced_cohesion = beta * summary->cohesion / reduction_factor;
    reduced_friction = atan2(beta * sin(summary->friction_angle),
                             reduction_factor * cos(summary->friction_angle));
  } else {
    phi_one = atan2(sin(summary->friction_angle),
                    reduction_factor * cos(summary->friction_angle));
    reduced_cohesion = summary->cohesion / reduction_factor;
    if (summary->davis == SS_MC_DAVIS_B) {
      psi_one = atan2(sin(summary->dilatancy_angle),
                      reduction_factor * cos(summary->dilatancy_angle));
      denominator = 1.0 - sin(phi_one) * sin(psi_one);
      if (!(denominator >
            64.0 * DBL_EPSILON * fmax(1.0, fabs(sin(phi_one) * sin(psi_one))))) {
        return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT,
                               "MC_DAVIS_DENOMINATOR");
      }
      beta = cos(phi_one) * cos(psi_one) / denominator;
    } else if (phi_one > summary->dilatancy_angle) {
      candidate.davis_c_high_branch = 1;
      denominator = 1.0 - sin(phi_one) * sin(summary->dilatancy_angle);
      if (!(denominator >
            64.0 * DBL_EPSILON *
                fmax(1.0, fabs(sin(phi_one) * sin(summary->dilatancy_angle))))) {
        return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT,
                               "MC_DAVIS_DENOMINATOR");
      }
      beta = cos(phi_one) * cos(summary->dilatancy_angle) / denominator;
    }
    reduced_cohesion *= beta;
    reduced_friction = atan2(beta * sin(phi_one), cos(phi_one));
  }
  candidate.cohesion = reduced_cohesion;
  candidate.friction_angle = reduced_friction;
  candidate.c_bar = 2.0 * reduced_cohesion * cos(reduced_friction);
  candidate.sin_friction = sin(reduced_friction);
  candidate.beta = beta;
  if (!isfinite(candidate.cohesion) || !isfinite(candidate.friction_angle) ||
      !isfinite(candidate.c_bar) || !isfinite(candidate.sin_friction) ||
      !isfinite(candidate.beta) || candidate.beta <= 0.0) {
    return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_REDUCTION_NONFINITE");
  }
  *reduced = candidate;
  return SS_OK;
}

typedef struct {
  double values[3];
  double vectors[9];
  uint32_t flags;
} SsMcSpectrum;

static double ss_mc_max_abs(const double *values, size_t count) {
  double maximum = 0.0;
  size_t index;
  for (index = 0U; index < count; ++index) {
    maximum = fmax(maximum, fabs(values[index]));
  }
  return maximum;
}

static int ss_mc_all_finite(const double *values, size_t count) {
  size_t index;
  for (index = 0U; index < count; ++index) {
    if (!isfinite(values[index])) {
      return 0;
    }
  }
  return 1;
}

static void ss_mc_jacobi_rotation(double matrix[9], double vectors[9], size_t first,
                                  size_t second) {
  const double off_diagonal = matrix[first * 3U + second];
  const double diagonal_first = matrix[first * 3U + first];
  const double diagonal_second = matrix[second * 3U + second];
  const double theta = 0.5 * (diagonal_second - diagonal_first) / off_diagonal;
  const double tangent = copysign(1.0, theta) / (fabs(theta) + hypot(1.0, theta));
  const double cosine = 1.0 / hypot(1.0, tangent);
  const double sine = tangent * cosine;
  size_t other;

  matrix[first * 3U + first] = diagonal_first - tangent * off_diagonal;
  matrix[second * 3U + second] = diagonal_second + tangent * off_diagonal;
  matrix[first * 3U + second] = 0.0;
  matrix[second * 3U + first] = 0.0;
  for (other = 0U; other < 3U; ++other) {
    double old_first;
    double old_second;
    if (other == first || other == second) {
      continue;
    }
    old_first = matrix[other * 3U + first];
    old_second = matrix[other * 3U + second];
    matrix[other * 3U + first] = cosine * old_first - sine * old_second;
    matrix[first * 3U + other] = matrix[other * 3U + first];
    matrix[other * 3U + second] = sine * old_first + cosine * old_second;
    matrix[second * 3U + other] = matrix[other * 3U + second];
  }
  for (other = 0U; other < 3U; ++other) {
    const double old_first = vectors[other * 3U + first];
    const double old_second = vectors[other * 3U + second];
    vectors[other * 3U + first] = cosine * old_first - sine * old_second;
    vectors[other * 3U + second] = sine * old_first + cosine * old_second;
  }
}

static void ss_mc_swap_eigenpairs(SsMcSpectrum *spectrum, size_t first, size_t second) {
  double temporary = spectrum->values[first];
  size_t row;
  spectrum->values[first] = spectrum->values[second];
  spectrum->values[second] = temporary;
  for (row = 0U; row < 3U; ++row) {
    temporary = spectrum->vectors[row * 3U + first];
    spectrum->vectors[row * 3U + first] = spectrum->vectors[row * 3U + second];
    spectrum->vectors[row * 3U + second] = temporary;
  }
}

/* Relative gap below which two principal strains are treated as repeated.
 *
 * The divided difference (g_i - g_j) / (e_i - e_j) carries a cancellation error
 * of about eps |g| / |e_i - e_j|, while the repeated-root limit is exact for the
 * branch responses that are admissible at e_i = e_j (isotropy forces g_i = g_j
 * there).  A threshold of sqrt(eps) times the spectral scale bounds the former
 * near 1e-8 relative and, unlike an absolute floor, is positively homogeneous
 * of degree one, so the tangent is invariant under strain scaling. */
static double ss_mc_spectral_gap_tolerance(const double values[3]) {
  return sqrt(DBL_EPSILON) * ss_mc_max_abs(values, 3U);
}

static SsStatus ss_mc_spectrum(const double tensor[9], SsMcSpectrum *spectrum) {
  double matrix[9];
  double scale;
  double residual = 0.0;
  size_t sweep;
  size_t row;
  size_t column;
  if (tensor == NULL || spectrum == NULL) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  memset(spectrum, 0, sizeof(*spectrum));
  spectrum->vectors[0] = 1.0;
  spectrum->vectors[4] = 1.0;
  spectrum->vectors[8] = 1.0;
  scale = ss_mc_max_abs(tensor, 9U);
  if (scale == 0.0) {
    spectrum->flags |= SS_MC_ROBUST_REPEATED_SPECTRUM;
    return SS_OK;
  }
  for (row = 0U; row < 9U; ++row) {
    matrix[row] = tensor[row] / scale;
  }
  for (sweep = 0U; sweep < 24U; ++sweep) {
    const size_t pairs[3][2] = {{0U, 1U}, {0U, 2U}, {1U, 2U}};
    int rotated = 0;
    size_t pair;
    for (pair = 0U; pair < 3U; ++pair) {
      const size_t first = pairs[pair][0];
      const size_t second = pairs[pair][1];
      const double threshold = 8.0 * DBL_EPSILON *
                               fmax(1.0, fmax(fabs(matrix[first * 3U + first]),
                                              fabs(matrix[second * 3U + second])));
      if (fabs(matrix[first * 3U + second]) > threshold) {
        ss_mc_jacobi_rotation(matrix, spectrum->vectors, first, second);
        rotated = 1;
      }
    }
    /* A sweep with no rotations leaves the matrix unchanged, so all later
     * sweeps would be exact no-ops under the same thresholds. */
    if (rotated == 0) {
      break;
    }
  }
  for (row = 0U; row < 3U; ++row) {
    spectrum->values[row] = matrix[row * 3U + row] * scale;
  }
  if (spectrum->values[1] > spectrum->values[0]) {
    ss_mc_swap_eigenpairs(spectrum, 0U, 1U);
  }
  if (spectrum->values[2] > spectrum->values[1]) {
    ss_mc_swap_eigenpairs(spectrum, 1U, 2U);
  }
  if (spectrum->values[1] > spectrum->values[0]) {
    ss_mc_swap_eigenpairs(spectrum, 0U, 1U);
  }
  for (column = 0U; column < 3U; ++column) {
    size_t pivot = 0U;
    double pivot_magnitude = fabs(spectrum->vectors[column]);
    for (row = 1U; row < 3U; ++row) {
      const double magnitude = fabs(spectrum->vectors[row * 3U + column]);
      if (magnitude > pivot_magnitude) {
        pivot = row;
        pivot_magnitude = magnitude;
      }
    }
    if (spectrum->vectors[pivot * 3U + column] < 0.0) {
      for (row = 0U; row < 3U; ++row) {
        spectrum->vectors[row * 3U + column] = -spectrum->vectors[row * 3U + column];
      }
    }
  }
  for (row = 0U; row < 3U; ++row) {
    for (column = 0U; column < 3U; ++column) {
      double reconstructed = 0.0;
      size_t eigen;
      for (eigen = 0U; eigen < 3U; ++eigen) {
        reconstructed += spectrum->vectors[row * 3U + eigen] * spectrum->values[eigen] *
                         spectrum->vectors[column * 3U + eigen];
      }
      residual = fmax(residual, fabs(reconstructed - tensor[row * 3U + column]));
    }
  }
  if (residual > 2.0e-12 * fmax(1.0, scale)) {
    return SS_ERROR_INTERNAL;
  }
  {
    const double gap_tolerance = ss_mc_spectral_gap_tolerance(spectrum->values);
    if (fabs(spectrum->values[0] - spectrum->values[1]) <= gap_tolerance ||
        fabs(spectrum->values[1] - spectrum->values[2]) <= gap_tolerance) {
      spectrum->flags |= SS_MC_ROBUST_REPEATED_SPECTRUM;
    }
  }
  return SS_OK;
}

static void ss_mc_base_elastic(const SsMcParameterSummary *parameters,
                               const double eigenvalues[3], double gradient[3],
                               double hessian[3][3], double *potential) {
  const double trace = eigenvalues[0] + eigenvalues[1] + eigenvalues[2];
  size_t row;
  size_t column;
  *potential = 0.5 * parameters->lame_first * trace * trace +
               parameters->shear_modulus *
                   (eigenvalues[0] * eigenvalues[0] + eigenvalues[1] * eigenvalues[1] +
                    eigenvalues[2] * eigenvalues[2]);
  for (row = 0U; row < 3U; ++row) {
    gradient[row] = parameters->lame_first * trace +
                    2.0 * parameters->shear_modulus * eigenvalues[row];
    for (column = 0U; column < 3U; ++column) {
      hessian[row][column] = parameters->lame_first +
                             (row == column ? 2.0 * parameters->shear_modulus : 0.0);
    }
  }
}

static double ss_mc_linear_form(const double coefficients[3], const double values[3],
                                double constant) {
  return coefficients[0] * values[0] + coefficients[1] * values[1] +
         coefficients[2] * values[2] + constant;
}

static void ss_mc_quadratic_correction(const double coefficients[3], double numerator,
                                       double denominator, double gradient[3],
                                       double hessian[3][3], double *potential) {
  size_t row;
  size_t column;
  *potential -= 0.5 * numerator * numerator / denominator;
  for (row = 0U; row < 3U; ++row) {
    gradient[row] -= numerator * coefficients[row] / denominator;
    for (column = 0U; column < 3U; ++column) {
      hessian[row][column] -= coefficients[row] * coefficients[column] / denominator;
    }
  }
}

static void ss_mc_base_edge(const SsMcParameterSummary *parameters,
                            const double eigenvalues[3], int left, double gradient[3],
                            double hessian[3][3], double *potential) {
  const double trace = eigenvalues[0] + eigenvalues[1] + eigenvalues[2];
  const double pair =
      left ? eigenvalues[0] + eigenvalues[1] : eigenvalues[1] + eigenvalues[2];
  const size_t single = left ? 2U : 0U;
  size_t row;
  size_t column;
  *potential = 0.5 * parameters->lame_first * trace * trace +
               parameters->shear_modulus *
                   (eigenvalues[single] * eigenvalues[single] + 0.5 * pair * pair);
  for (row = 0U; row < 3U; ++row) {
    const int in_pair = left ? (row == 0U || row == 1U) : (row == 1U || row == 2U);
    gradient[row] =
        parameters->lame_first * trace +
        parameters->shear_modulus * (in_pair ? pair : 2.0 * eigenvalues[single]);
    for (column = 0U; column < 3U; ++column) {
      const int column_in_pair =
          left ? (column == 0U || column == 1U) : (column == 1U || column == 2U);
      hessian[row][column] = parameters->lame_first;
      if (in_pair && column_in_pair) {
        hessian[row][column] += parameters->shear_modulus;
      } else if (!in_pair && !column_in_pair) {
        hessian[row][column] += 2.0 * parameters->shear_modulus;
      }
    }
  }
}

static void ss_mc_mark_boundary(double left, double right, double tolerance,
                                uint32_t *flags) {
  if (fabs(left - right) <= tolerance) {
    *flags |= SS_MC_ROBUST_NONSMOOTH;
  }
}

static SsStatus ss_mc_principal_response(const SsMcParameterSummary *parameters,
                                         const SsMcReduced *reduced,
                                         const double eigenvalues[3],
                                         double gradient[3], double hessian[3][3],
                                         double *potential, double *trial_yield,
                                         SsMcBranch *branch, uint32_t *flags) {
  const double shear = parameters->shear_modulus;
  const double bulk = parameters->bulk_modulus;
  const double lame = parameters->lame_first;
  const double s = reduced->sin_friction;
  const double trace = eigenvalues[0] + eigenvalues[1] + eigenvalues[2];
  /* Stress scale of the point without a unit-dependent absolute floor, so the
   * branch decision is invariant under strain scaling when c_bar = 0. */
  const double scale = fmax(reduced->c_bar, shear * ss_mc_max_abs(eigenvalues, 3U));
  const double tolerance = 256.0 * DBL_EPSILON * scale;
  double gamma_smooth_left;
  double gamma_smooth_right;
  double gamma_left_apex;
  double gamma_right_apex;
  double denominator_smooth;
  double denominator_left;
  double denominator_right;
  double multiplier_smooth;
  double multiplier_left;
  double multiplier_right;
  double coefficients[3];
  double numerator;

  *trial_yield =
      2.0 * shear * ((1.0 + s) * eigenvalues[0] - (1.0 - s) * eigenvalues[2]) +
      2.0 * lame * s * trace - reduced->c_bar;
  ss_mc_mark_boundary(*trial_yield, 0.0, tolerance, flags);
  if (*trial_yield <= tolerance) {
    *branch = SS_MC_BRANCH_ELASTIC;
    ss_mc_base_elastic(parameters, eigenvalues, gradient, hessian, potential);
    return SS_OK;
  }

  gamma_smooth_left = (eigenvalues[0] - eigenvalues[1]) / (1.0 + s);
  gamma_smooth_right = (eigenvalues[1] - eigenvalues[2]) / (1.0 - s);
  gamma_left_apex =
      (eigenvalues[0] + eigenvalues[1] - 2.0 * eigenvalues[2]) / (3.0 - s);
  gamma_right_apex =
      (2.0 * eigenvalues[0] - eigenvalues[1] - eigenvalues[2]) / (3.0 + s);
  denominator_smooth = 4.0 * lame * s * s + 2.0 * shear * (1.0 + s) * (1.0 + s) +
                       2.0 * shear * (1.0 - s) * (1.0 - s);
  denominator_left = 4.0 * lame * s * s + shear * (1.0 + s) * (1.0 + s) +
                     2.0 * shear * (1.0 - s) * (1.0 - s);
  denominator_right = 4.0 * lame * s * s + 2.0 * shear * (1.0 + s) * (1.0 + s) +
                      shear * (1.0 - s) * (1.0 - s);
  multiplier_smooth = *trial_yield / denominator_smooth;
  coefficients[0] = 2.0 * shear * (1.0 + s) + 2.0 * lame * s;
  coefficients[1] = 2.0 * lame * s;
  coefficients[2] = -2.0 * shear * (1.0 - s) + 2.0 * lame * s;
  ss_mc_mark_boundary(multiplier_smooth, gamma_smooth_left, tolerance / shear, flags);
  ss_mc_mark_boundary(multiplier_smooth, gamma_smooth_right, tolerance / shear, flags);
  if (multiplier_smooth <=
      fmin(gamma_smooth_left, gamma_smooth_right) + tolerance / shear) {
    *branch = SS_MC_BRANCH_SMOOTH;
    ss_mc_base_elastic(parameters, eigenvalues, gradient, hessian, potential);
    ss_mc_quadratic_correction(coefficients, *trial_yield, denominator_smooth, gradient,
                               hessian, potential);
    return SS_OK;
  }

  coefficients[0] = shear * (1.0 + s) + 2.0 * lame * s;
  coefficients[1] = coefficients[0];
  coefficients[2] = -2.0 * shear * (1.0 - s) + 2.0 * lame * s;
  numerator = ss_mc_linear_form(coefficients, eigenvalues, -reduced->c_bar);
  multiplier_left = numerator / denominator_left;
  ss_mc_mark_boundary(gamma_smooth_left, gamma_smooth_right, tolerance / shear, flags);
  ss_mc_mark_boundary(multiplier_left, gamma_smooth_left, tolerance / shear, flags);
  ss_mc_mark_boundary(multiplier_left, gamma_left_apex, tolerance / shear, flags);
  if (gamma_smooth_left < gamma_smooth_right - tolerance / shear &&
      multiplier_left >= gamma_smooth_left - tolerance / shear &&
      multiplier_left <= gamma_left_apex + tolerance / shear) {
    *branch = SS_MC_BRANCH_LEFT_EDGE;
    ss_mc_base_edge(parameters, eigenvalues, 1, gradient, hessian, potential);
    ss_mc_quadratic_correction(coefficients, numerator, denominator_left, gradient,
                               hessian, potential);
    return SS_OK;
  }

  coefficients[0] = 2.0 * shear * (1.0 + s) + 2.0 * lame * s;
  coefficients[1] = -shear * (1.0 - s) + 2.0 * lame * s;
  coefficients[2] = coefficients[1];
  numerator = ss_mc_linear_form(coefficients, eigenvalues, -reduced->c_bar);
  multiplier_right = numerator / denominator_right;
  ss_mc_mark_boundary(multiplier_right, gamma_smooth_right, tolerance / shear, flags);
  ss_mc_mark_boundary(multiplier_right, gamma_right_apex, tolerance / shear, flags);
  if (gamma_smooth_left > gamma_smooth_right + tolerance / shear &&
      multiplier_right >= gamma_smooth_right - tolerance / shear &&
      multiplier_right <= gamma_right_apex + tolerance / shear) {
    *branch = SS_MC_BRANCH_RIGHT_EDGE;
    ss_mc_base_edge(parameters, eigenvalues, 0, gradient, hessian, potential);
    ss_mc_quadratic_correction(coefficients, numerator, denominator_right, gradient,
                               hessian, potential);
    return SS_OK;
  }

  if (s <= 64.0 * DBL_EPSILON) {
    *flags |= SS_MC_ROBUST_ZERO_FRICTION | SS_MC_ROBUST_NONSMOOTH;
    if (gamma_smooth_left <= gamma_smooth_right) {
      coefficients[0] = shear;
      coefficients[1] = shear;
      coefficients[2] = -2.0 * shear;
      numerator = ss_mc_linear_form(coefficients, eigenvalues, -reduced->c_bar);
      *branch = SS_MC_BRANCH_LEFT_EDGE;
      ss_mc_base_edge(parameters, eigenvalues, 1, gradient, hessian, potential);
      ss_mc_quadratic_correction(coefficients, numerator, denominator_left, gradient,
                                 hessian, potential);
    } else {
      coefficients[0] = 2.0 * shear;
      coefficients[1] = -shear;
      coefficients[2] = -shear;
      numerator = ss_mc_linear_form(coefficients, eigenvalues, -reduced->c_bar);
      *branch = SS_MC_BRANCH_RIGHT_EDGE;
      ss_mc_base_edge(parameters, eigenvalues, 0, gradient, hessian, potential);
      ss_mc_quadratic_correction(coefficients, numerator, denominator_right, gradient,
                                 hessian, potential);
    }
    return SS_OK;
  }

  *branch = SS_MC_BRANCH_APEX;
  *flags |= SS_MC_ROBUST_NONSMOOTH;
  {
    const double denominator_apex = 4.0 * bulk * s * s;
    const double numerator_apex = 2.0 * bulk * s * trace - reduced->c_bar;
    const double apex_stress = reduced->c_bar / (2.0 * s);
    size_t row;
    size_t column;
    *potential = 0.5 * bulk * trace * trace -
                 0.5 * numerator_apex * numerator_apex / denominator_apex;
    for (row = 0U; row < 3U; ++row) {
      gradient[row] = apex_stress;
      for (column = 0U; column < 3U; ++column) {
        hessian[row][column] = 0.0;
      }
    }
  }
  return SS_OK;
}

static void ss_mc_tensor_from_engineering(const double strain[6], double tensor[9]) {
  memset(tensor, 0, 9U * sizeof(double));
  tensor[0] = strain[0];
  tensor[4] = strain[1];
  tensor[8] = strain[2];
  tensor[1] = tensor[3] = 0.5 * strain[3];
  tensor[5] = tensor[7] = 0.5 * strain[4];
  tensor[2] = tensor[6] = 0.5 * strain[5];
}

static void ss_mc_stress_from_tensor(const double tensor[9], double stress[6]) {
  stress[0] = tensor[0];
  stress[1] = tensor[4];
  stress[2] = tensor[8];
  stress[3] = 0.5 * (tensor[1] + tensor[3]);
  stress[4] = 0.5 * (tensor[5] + tensor[7]);
  stress[5] = 0.5 * (tensor[2] + tensor[6]);
}

static void ss_mc_rotate_from_principal(const double vectors[9],
                                        const double principal[9], double tensor[9]) {
  double work[9] = {0.0};
  size_t row;
  size_t column;
  size_t inner;
  for (row = 0U; row < 3U; ++row) {
    for (column = 0U; column < 3U; ++column) {
      for (inner = 0U; inner < 3U; ++inner) {
        work[row * 3U + column] +=
            vectors[row * 3U + inner] * principal[inner * 3U + column];
      }
    }
  }
  memset(tensor, 0, 9U * sizeof(double));
  for (row = 0U; row < 3U; ++row) {
    for (column = 0U; column < 3U; ++column) {
      for (inner = 0U; inner < 3U; ++inner) {
        tensor[row * 3U + column] +=
            work[row * 3U + inner] * vectors[column * 3U + inner];
      }
    }
  }
}

static void ss_mc_rotate_to_principal(const double vectors[9], const double tensor[9],
                                      double principal[9]) {
  double work[9] = {0.0};
  size_t row;
  size_t column;
  size_t inner;
  for (row = 0U; row < 3U; ++row) {
    for (column = 0U; column < 3U; ++column) {
      for (inner = 0U; inner < 3U; ++inner) {
        work[row * 3U + column] +=
            vectors[inner * 3U + row] * tensor[inner * 3U + column];
      }
    }
  }
  memset(principal, 0, 9U * sizeof(double));
  for (row = 0U; row < 3U; ++row) {
    for (column = 0U; column < 3U; ++column) {
      for (inner = 0U; inner < 3U; ++inner) {
        principal[row * 3U + column] +=
            work[row * 3U + inner] * vectors[inner * 3U + column];
      }
    }
  }
}

static void ss_mc_tangent(const SsMcSpectrum *spectrum, const double gradient[3],
                          double hessian[3][3], double tangent[36]) {
  const double gap_tolerance = ss_mc_spectral_gap_tolerance(spectrum->values);
  size_t column;
  memset(tangent, 0, 36U * sizeof(double));
  for (column = 0U; column < 6U; ++column) {
    double engineering_basis[6] = {0.0};
    double tensor_basis[9];
    double principal_basis[9];
    double principal_response[9] = {0.0};
    double tensor_response[9];
    double stress_response[6];
    size_t row;
    size_t first;
    size_t second;
    engineering_basis[column] = 1.0;
    ss_mc_tensor_from_engineering(engineering_basis, tensor_basis);
    ss_mc_rotate_to_principal(spectrum->vectors, tensor_basis, principal_basis);
    for (first = 0U; first < 3U; ++first) {
      for (second = 0U; second < 3U; ++second) {
        principal_response[first * 3U + first] +=
            hessian[first][second] * principal_basis[second * 3U + second];
      }
    }
    for (first = 0U; first < 3U; ++first) {
      for (second = first + 1U; second < 3U; ++second) {
        const double gap = spectrum->values[first] - spectrum->values[second];
        double coefficient;
        if (fabs(gap) <= gap_tolerance) {
          coefficient = 0.5 * ((hessian[first][first] - hessian[first][second]) +
                               (hessian[second][second] - hessian[second][first]));
        } else {
          coefficient = (gradient[first] - gradient[second]) / gap;
        }
        principal_response[first * 3U + second] =
            coefficient * principal_basis[first * 3U + second];
        principal_response[second * 3U + first] =
            coefficient * principal_basis[second * 3U + first];
      }
    }
    ss_mc_rotate_from_principal(spectrum->vectors, principal_response, tensor_response);
    ss_mc_stress_from_tensor(tensor_response, stress_response);
    for (row = 0U; row < 6U; ++row) {
      tangent[row + 6U * column] = stress_response[row];
    }
  }
}

static SsStatus ss_mc_evaluate_3d(const SsMcParameterSummary *parameters,
                                  const SsMcReduced *reduced, const double strain[6],
                                  uint32_t request, SsMcResponse *response) {
  double tensor[9];
  SsMcSpectrum spectrum;
  double gradient[3] = {0.0, 0.0, 0.0};
  double hessian[3][3] = {{0.0}};
  double principal_stress[9] = {0.0};
  double stress_tensor[9];
  SsStatus status;
  size_t index;
  if ((request & SS_MC_EVALUATE_TANGENT) == 0U) {
    memset(response->tangent, 0, sizeof(response->tangent));
  }
  response->reduced = *reduced;
  ss_mc_tensor_from_engineering(strain, tensor);
  status = ss_mc_spectrum(tensor, &spectrum);
  if (status != SS_OK) {
    return status;
  }
  response->robustness_flags = spectrum.flags;
  if (reduced->sin_friction <= 64.0 * DBL_EPSILON) {
    response->robustness_flags |= SS_MC_ROBUST_ZERO_FRICTION;
  }
  status = ss_mc_principal_response(
      parameters, reduced, spectrum.values, gradient, hessian, &response->potential,
      &response->trial_yield, &response->branch, &response->robustness_flags);
  if (status != SS_OK) {
    return status;
  }
  {
    const double maximum_stress = fmax(gradient[0], fmax(gradient[1], gradient[2]));
    const double minimum_stress = fmin(gradient[0], fmin(gradient[1], gradient[2]));
    response->yield_residual = (1.0 + reduced->sin_friction) * maximum_stress -
                               (1.0 - reduced->sin_friction) * minimum_stress -
                               reduced->c_bar;
  }
  for (index = 0U; index < 3U; ++index) {
    principal_stress[index * 3U + index] = gradient[index];
  }
  ss_mc_rotate_from_principal(spectrum.vectors, principal_stress, stress_tensor);
  ss_mc_stress_from_tensor(stress_tensor, response->stress);
  if ((request & SS_MC_EVALUATE_TANGENT) != 0U) {
    ss_mc_tangent(&spectrum, gradient, hessian, response->tangent);
  }
  if (!isfinite(response->potential) || !isfinite(response->trial_yield) ||
      !isfinite(response->yield_residual) || !ss_mc_all_finite(response->stress, 6U) ||
      ((request & SS_MC_EVALUATE_TANGENT) != 0U &&
       !ss_mc_all_finite(response->tangent, 36U))) {
    return SS_ERROR_INTERNAL;
  }
  return SS_OK;
}

static SsStatus ss_mc_evaluate_reduced(const SsMcParameterSummary *parameters,
                                       const SsMcReduced *reduced,
                                       SsMcDimension dimension,
                                       const double *engineering_strain,
                                       uint32_t request, SsMcResponse *response) {
  if (dimension == SS_MC_DIMENSION_3) {
    return ss_mc_evaluate_3d(parameters, reduced, engineering_strain, request,
                             response);
  }
  {
    const double embedded[6] = {
        engineering_strain[0],
        engineering_strain[1],
        0.0,
        engineering_strain[2],
        0.0,
        0.0,
    };
    static const size_t map[3] = {0U, 1U, 3U};
    SsMcResponse full;
    size_t row;
    size_t column;
    SsStatus status = ss_mc_evaluate_3d(parameters, reduced, embedded, request, &full);
    if (status != SS_OK) {
      return status;
    }
    memset(response, 0, sizeof(*response));
    response->stress[0] = full.stress[0];
    response->stress[1] = full.stress[1];
    response->stress[2] = full.stress[3];
    response->stress[3] = full.stress[2];
    if ((request & SS_MC_EVALUATE_TANGENT) != 0U) {
      for (column = 0U; column < 3U; ++column) {
        for (row = 0U; row < 3U; ++row) {
          response->tangent[row + 3U * column] =
              full.tangent[map[row] + 6U * map[column]];
        }
      }
    }
    response->potential = full.potential;
    response->trial_yield = full.trial_yield;
    response->yield_residual = full.yield_residual;
    response->reduced = full.reduced;
    response->branch = full.branch;
    response->robustness_flags = full.robustness_flags;
  }
  return SS_OK;
}

SsStatus ss_mc_evaluate_prepared(const SsMcParameters *parameters,
                                 SsMcDimension dimension, const SsMcReduced *reduced,
                                 uint32_t request, const double *engineering_strain,
                                 SsMcResponse *response, SsMcError *error) {
  SsMcResponse candidate;
  size_t component;
  size_t component_count;
  SsStatus status;
  if (parameters == NULL || reduced == NULL || engineering_strain == NULL ||
      response == NULL) {
    return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_EVALUATE_ARGUMENT");
  }
  ss_mc_error_clear(error);
  if (dimension != SS_MC_DIMENSION_2 && dimension != SS_MC_DIMENSION_3) {
    return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_DIMENSION");
  }
  if (request != 0U && request != (uint32_t)SS_MC_EVALUATE_TANGENT) {
    return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_EVALUATE_REQUEST");
  }
  if (!isfinite(reduced->cohesion) || !isfinite(reduced->friction_angle) ||
      !isfinite(reduced->c_bar) || !isfinite(reduced->sin_friction) ||
      !isfinite(reduced->beta) || reduced->cohesion < 0.0 ||
      reduced->friction_angle < 0.0 || reduced->friction_angle >= SS_MC_PI_OVER_TWO ||
      reduced->c_bar < 0.0 || reduced->sin_friction < 0.0 ||
      reduced->sin_friction >= 1.0 || reduced->beta <= 0.0 ||
      (reduced->davis_c_high_branch != 0 && reduced->davis_c_high_branch != 1)) {
    return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_REDUCTION_INVALID");
  }
  component_count = dimension == SS_MC_DIMENSION_2 ? 3U : 6U;
  for (component = 0U; component < component_count; ++component) {
    if (!isfinite(engineering_strain[component])) {
      return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_NONFINITE_STRAIN");
    }
  }
  status = ss_mc_evaluate_reduced(&parameters->summary, reduced, dimension,
                                  engineering_strain, request, &candidate);
  if (status != SS_OK) {
    return ss_mc_error_set(error, status, "MC_SPECTRAL_EVALUATION");
  }
  *response = candidate;
  return SS_OK;
}

SsStatus ss_mc_evaluate(const SsMcParameters *parameters, SsMcDimension dimension,
                        double reduction_factor, const double *engineering_strain,
                        SsMcResponse *response, SsMcError *error) {
  SsMcReduced reduced;
  SsMcResponse candidate;
  size_t component;
  size_t component_count;
  SsStatus status;
  if (parameters == NULL || engineering_strain == NULL || response == NULL) {
    return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_EVALUATE_ARGUMENT");
  }
  ss_mc_error_clear(error);
  if (dimension != SS_MC_DIMENSION_2 && dimension != SS_MC_DIMENSION_3) {
    return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_DIMENSION");
  }
  component_count = dimension == SS_MC_DIMENSION_2 ? 3U : 6U;
  for (component = 0U; component < component_count; ++component) {
    if (!isfinite(engineering_strain[component])) {
      return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_NONFINITE_STRAIN");
    }
  }
  status = ss_mc_reduce(parameters, reduction_factor, &reduced, error);
  if (status != SS_OK) {
    return status;
  }
  status =
      ss_mc_evaluate_reduced(&parameters->summary, &reduced, dimension,
                             engineering_strain, SS_MC_EVALUATE_TANGENT, &candidate);
  if (status != SS_OK) {
    return ss_mc_error_set(error, status, "MC_SPECTRAL_EVALUATION");
  }
  *response = candidate;
  return SS_OK;
}

static int ss_mc_error_key_precedes(uint64_t entity, uint32_t quadrature, size_t index,
                                    const SsMcError *current) {
  return current->valid == 0 || entity < current->stable_entity_id ||
         (entity == current->stable_entity_id &&
          (quadrature < current->quadrature_index ||
           (quadrature == current->quadrature_index && index < current->local_index)));
}

SsStatus ss_mc_evaluate_batch(const SsMcParameters *parameters, SsMcDimension dimension,
                              double reduction_factor, const SsMcBatchInput *input,
                              SsMcBatchOutput *output, SsMcError *error) {
  SsMcReduced reduced;
  SsMcError first = {0};
  size_t strain_count;
  size_t stress_count;
  size_t tangent_count;
  size_t index;
  size_t component;
  SsStatus status;
  ss_mc_error_clear(error);
  if (parameters == NULL || input == NULL || output == NULL) {
    return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_BATCH_ARGUMENT");
  }
  if (dimension != SS_MC_DIMENSION_2 && dimension != SS_MC_DIMENSION_3) {
    return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_DIMENSION");
  }
  if (input->count == 0U) {
    return SS_OK;
  }
  if (input->stride < input->count || output->stride < input->count ||
      input->strain == NULL || input->stable_entity_ids == NULL ||
      input->quadrature_indices == NULL || output->stress == NULL ||
      output->tangent == NULL || output->potential == NULL ||
      output->trial_yield == NULL || output->yield_residual == NULL ||
      output->branch == NULL || output->robustness_flags == NULL) {
    return ss_mc_error_set(error, SS_ERROR_INVALID_ARGUMENT, "MC_BATCH_LAYOUT");
  }
  status = ss_mc_reduce(parameters, reduction_factor, &reduced, error);
  if (status != SS_OK) {
    return status;
  }
  strain_count = dimension == SS_MC_DIMENSION_2 ? 3U : 6U;
  stress_count = dimension == SS_MC_DIMENSION_2 ? 4U : 6U;
  tangent_count = dimension == SS_MC_DIMENSION_2 ? 9U : 36U;
  for (index = 0U; index < input->count; ++index) {
    for (component = 0U; component < strain_count; ++component) {
      if (!isfinite(input->strain[component * input->stride + index]) &&
          ss_mc_error_key_precedes(input->stable_entity_ids[index],
                                   input->quadrature_indices[index], index, &first)) {
        (void)ss_mc_error_set(&first, SS_ERROR_INVALID_ARGUMENT, "MC_NONFINITE_STRAIN");
        first.local_index = index;
        first.stable_entity_id = input->stable_entity_ids[index];
        first.quadrature_index = input->quadrature_indices[index];
      }
    }
  }
  if (first.valid != 0) {
    if (error != NULL) {
      *error = first;
    }
    return first.status;
  }
  for (index = 0U; index < input->count; ++index) {
    double strain[6] = {0.0};
    SsMcResponse response;
    for (component = 0U; component < strain_count; ++component) {
      strain[component] = input->strain[component * input->stride + index];
    }
    status = ss_mc_evaluate_reduced(&parameters->summary, &reduced, dimension, strain,
                                    SS_MC_EVALUATE_TANGENT, &response);
    if (status != SS_OK) {
      if (error != NULL) {
        (void)ss_mc_error_set(error, status, "MC_SPECTRAL_EVALUATION");
        error->local_index = index;
        error->stable_entity_id = input->stable_entity_ids[index];
        error->quadrature_index = input->quadrature_indices[index];
      }
      return status;
    }
    for (component = 0U; component < stress_count; ++component) {
      output->stress[component * output->stride + index] = response.stress[component];
    }
    for (component = 0U; component < tangent_count; ++component) {
      output->tangent[component * output->stride + index] = response.tangent[component];
    }
    output->potential[index] = response.potential;
    output->trial_yield[index] = response.trial_yield;
    output->yield_residual[index] = response.yield_residual;
    output->branch[index] = response.branch;
    output->robustness_flags[index] = response.robustness_flags;
  }
  return SS_OK;
}
