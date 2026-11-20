/* Inline Voigt strain-displacement and tangent kernels for simplex mechanics.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#ifndef SS_ASSEMBLY_VOIGT_H
#define SS_ASSEMBLY_VOIGT_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/** Add one basis gradient's contribution to an engineering strain. */
static inline void ss_mechanics_accumulate_strain(uint32_t dimension,
                                                  const double *gradient,
                                                  uint32_t component, double value,
                                                  double strain[6]) {
  if (dimension == 2U) {
    if (component == 0U) {
      strain[0] += gradient[0] * value;
      strain[2] += gradient[1] * value;
    } else {
      strain[1] += gradient[1] * value;
      strain[2] += gradient[0] * value;
    }
  } else if (component == 0U) {
    strain[0] += gradient[0] * value;
    strain[3] += gradient[1] * value;
    strain[5] += gradient[2] * value;
  } else if (component == 1U) {
    strain[1] += gradient[1] * value;
    strain[3] += gradient[0] * value;
    strain[4] += gradient[2] * value;
  } else {
    strain[2] += gradient[2] * value;
    strain[4] += gradient[1] * value;
    strain[5] += gradient[0] * value;
  }
}

/** Strain-displacement column of one basis function and component. */
static inline void ss_mechanics_b_column(uint32_t dimension, const double *gradient,
                                         uint32_t component, double column[6]) {
  memset(column, 0, 6U * sizeof(*column));
  if (dimension == 2U) {
    if (component == 0U) {
      column[0] = gradient[0];
      column[2] = gradient[1];
    } else {
      column[1] = gradient[1];
      column[2] = gradient[0];
    }
  } else if (component == 0U) {
    column[0] = gradient[0];
    column[3] = gradient[1];
    column[5] = gradient[2];
  } else if (component == 1U) {
    column[1] = gradient[1];
    column[3] = gradient[0];
    column[4] = gradient[2];
  } else {
    column[2] = gradient[2];
    column[4] = gradient[1];
    column[5] = gradient[0];
  }
}

/** First term of a dot product, optionally summed from positive zero. */
static inline double ss_mechanics_first_dot_term(double term, int sum_from_zero) {
  return sum_from_zero != 0 ? 0.0 + term : term;
}

/** Element assembly starts the dot product at positive zero. The operator
 * projection starts at the first product, which can preserve a signed zero. */
static inline double ss_mechanics_b_dot(uint32_t dimension, const double *gradient,
                                        uint32_t component, const double *vector,
                                        int sum_from_zero) {
  if (dimension == 2U) {
    return component == 0U
               ? ss_mechanics_first_dot_term(gradient[0] * vector[0], sum_from_zero) +
                     gradient[1] * vector[2]
               : ss_mechanics_first_dot_term(gradient[1] * vector[1], sum_from_zero) +
                     gradient[0] * vector[2];
  }
  if (component == 0U) {
    return ss_mechanics_first_dot_term(gradient[0] * vector[0], sum_from_zero) +
           gradient[1] * vector[3] + gradient[2] * vector[5];
  }
  if (component == 1U) {
    return ss_mechanics_first_dot_term(gradient[1] * vector[1], sum_from_zero) +
           gradient[0] * vector[3] + gradient[2] * vector[4];
  }
  return ss_mechanics_first_dot_term(gradient[2] * vector[2], sum_from_zero) +
         gradient[1] * vector[4] + gradient[0] * vector[5];
}

/** Tangent times one strain-displacement column. */
static inline void ss_mechanics_tangent_times_b(uint32_t dimension,
                                                const double *tangent,
                                                const double *gradient,
                                                uint32_t component, double product[6]) {
  const size_t strain_count = dimension == 2U ? 3U : 6U;
  size_t row;
  for (row = 0U; row < strain_count; ++row) {
    double value = 0.0;
    if (dimension == 2U) {
      if (component == 0U) {
        value += tangent[row] * gradient[0];
        value += tangent[row + 2U * strain_count] * gradient[1];
      } else {
        value += tangent[row + strain_count] * gradient[1];
        value += tangent[row + 2U * strain_count] * gradient[0];
      }
    } else if (component == 0U) {
      value += tangent[row] * gradient[0];
      value += tangent[row + 3U * strain_count] * gradient[1];
      value += tangent[row + 5U * strain_count] * gradient[2];
    } else if (component == 1U) {
      value += tangent[row + strain_count] * gradient[1];
      value += tangent[row + 3U * strain_count] * gradient[0];
      value += tangent[row + 4U * strain_count] * gradient[2];
    } else {
      value += tangent[row + 2U * strain_count] * gradient[2];
      value += tangent[row + 4U * strain_count] * gradient[1];
      value += tangent[row + 5U * strain_count] * gradient[0];
    }
    product[row] = value;
  }
}

/** Tangent (or its transpose) times a strain vector. */
static inline void ss_mechanics_tangent_product(const double *tangent,
                                                size_t strain_count,
                                                const double input[6], int transpose,
                                                double output[6]) {
  size_t row;
  for (row = 0U; row < strain_count; ++row) {
    size_t column;
    double value = 0.0;
    for (column = 0U; column < strain_count; ++column) {
      value += (transpose == 0 ? tangent[row + column * strain_count]
                               : tangent[column + row * strain_count]) *
               input[column];
    }
    output[row] = value;
  }
}

#endif
