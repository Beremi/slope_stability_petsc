/* Mohr-Coulomb material: Davis variants, return branches, tangents, plane
 * strain and batched evaluation.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/material/material.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expression)                                                              \
  do {                                                                                 \
    if (!(expression)) {                                                               \
      fprintf(stderr, "%s:%d: assertion failed: %s\n", __FILE__, __LINE__,             \
              #expression);                                                            \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

#define CHECK_NEAR(actual, expected, absolute, relative)                               \
  do {                                                                                 \
    const double check_actual = (actual);                                              \
    const double check_expected = (expected);                                          \
    const double check_scale = fmax(fabs(check_actual), fabs(check_expected));         \
    const double check_tolerance = (absolute) + (relative) * check_scale;              \
    if (!isfinite(check_actual) ||                                                     \
        fabs(check_actual - check_expected) > check_tolerance) {                       \
      fprintf(stderr, "%s:%d: |%.17g - %.17g| > %.3g\n", __FILE__, __LINE__,           \
              check_actual, check_expected, check_tolerance);                          \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

typedef struct {
  SsMcDavis davis;
  double cohesion;
  double friction_degrees;
  double dilatancy_degrees;
  double factor;
  double c_bar;
  double sin_friction;
} ReductionOracle;

static const ReductionOracle REDUCTION_ORACLES[] = {
    {SS_MC_DAVIS_A, 6.0, 45.0, 0.0, 1.0, 6.92820323027551, 0.5773502691896257},
    {SS_MC_DAVIS_A, 6.0, 45.0, 0.0, 1.2, 6.092076990801715, 0.5076730825668095},
    {SS_MC_DAVIS_A, 6.0, 45.0, 0.0, 2.0, 4.0, 0.3333333333333333},
    {SS_MC_DAVIS_B, 6.0, 45.0, 0.0, 1.0, 6.92820323027551, 0.5773502691896257},
    {SS_MC_DAVIS_B, 6.0, 45.0, 0.0, 1.2, 6.469966392206305, 0.539163866017192},
    {SS_MC_DAVIS_B, 6.0, 45.0, 0.0, 2.0, 4.898979485566357, 0.40824829046386296},
    {SS_MC_DAVIS_C, 6.0, 45.0, 0.0, 1.0, 6.92820323027551, 0.5773502691896257},
    {SS_MC_DAVIS_C, 6.0, 45.0, 0.0, 1.2, 6.469966392206305, 0.539163866017192},
    {SS_MC_DAVIS_C, 6.0, 45.0, 0.0, 2.0, 4.898979485566357, 0.40824829046386296},
    {SS_MC_DAVIS_A, 12.0, 30.0, 10.0, 1.0, 19.729505814267398, 0.47461814553523207},
    {SS_MC_DAVIS_A, 12.0, 30.0, 10.0, 2.0, 10.821109589343626, 0.26031543893074927},
    {SS_MC_DAVIS_A, 12.0, 30.0, 10.0, 3.274316085206515, 6.754724643554309,
     0.16249342046907828},
    {SS_MC_DAVIS_A, 12.0, 30.0, 10.0, 4.0, 5.553518443841454, 0.1335968903542256},
    {SS_MC_DAVIS_B, 12.0, 30.0, 10.0, 1.0, 19.729505814267398, 0.47461814553523207},
    {SS_MC_DAVIS_B, 12.0, 30.0, 10.0, 2.0, 11.32604919545829, 0.2724623979938662},
    {SS_MC_DAVIS_B, 12.0, 30.0, 10.0, 3.274316085206515, 7.1669603204182595,
     0.17241026959436867},
    {SS_MC_DAVIS_B, 12.0, 30.0, 10.0, 4.0, 5.909620403116747, 0.14216337210616495},
    {SS_MC_DAVIS_C, 12.0, 30.0, 10.0, 1.0, 19.729505814267398, 0.47461814553523207},
    {SS_MC_DAVIS_C, 12.0, 30.0, 10.0, 2.0, 11.465833189135042, 0.27582507825960256},
    {SS_MC_DAVIS_C, 12.0, 30.0, 10.0, 3.274316085206515, 7.218419192660895,
     0.17364817766693033},
    {SS_MC_DAVIS_C, 12.0, 30.0, 10.0, 4.0, 5.938459911664722, 0.14285714285714285},
};

typedef struct {
  const char *id;
  uint32_t dimension;
  SsMcBranch branch;
  double strain[SS_MC_MAX_STRAIN_COMPONENTS];
  double stress[SS_MC_MAX_STRESS_COMPONENTS];
  double tangent[SS_MC_MAX_TANGENT_COMPONENTS];
  double finite_difference_step;
  double finite_difference_error;
} PointOracle;

static const PointOracle POINT_ORACLES[] = {
    {
        "MAT-P2-ELASTIC",
        2U,
        SS_MC_BRANCH_ELASTIC,
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0},
        {53846.153846153844, 23076.92307692307, 0.0, 23076.92307692307,
         53846.153846153844, 0.0, 0.0, 0.0, 15384.615384615385},
        1e-08,
        0.0,
    },
    {
        "MAT-P2-SMOOTH",
        2U,
        SS_MC_BRANCH_SMOOTH,
        {-0.006, 0.0005, -0.006},
        {-315.59468692717616, -124.00745917134921, -88.4248743488432,
         -131.88064382955758},
        {49335.216335475474, 10423.061981607505, 4355.629032732421, 10423.061981607505,
         13734.743617393433, 12075.358375613137, 4355.629032732421, 12075.358375613137,
         11174.52705373405},
        1e-08,
        1.9783838069997728e-06,
    },
    {
        "MAT-P2-RIGHT",
        2U,
        SS_MC_BRANCH_RIGHT_EDGE,
        {-0.0015, 0.006, -0.0015},
        {3.14293186723512, 5.208517124510806, -0.2065585257275685, 3.1224785424891337},
        {34287.51184045574, 9491.777276432276, 2452.032319638671, 9491.777276432276,
         2636.2350869191214, 713.0953557149896, 2452.032319638671, 713.0953557149896,
         311.59938021074714},
        1e-08,
        2.5929610956154647e-06,
    },
    {
        "MAT-P2-APEX",
        2U,
        SS_MC_BRANCH_APEX,
        {-0.001, 0.0045, -0.001},
        {6.000000000000001, 6.000000000000001, 0.0, 6.000000000000001},
        {0.0},
        1e-08,
        0.0,
    },
    {
        "MAT-P3-ELASTIC",
        3U,
        SS_MC_BRANCH_ELASTIC,
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {53846.153846153844,
         23076.92307692307,
         23076.92307692307,
         0.0,
         0.0,
         0.0,
         23076.92307692307,
         53846.153846153844,
         23076.92307692307,
         0.0,
         0.0,
         0.0,
         23076.92307692307,
         23076.92307692307,
         53846.153846153844,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         15384.615384615385,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         15384.615384615385,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         15384.615384615385},
        1e-08,
        0.0,
    },
    {
        "MAT-P3-SMOOTH",
        3U,
        SS_MC_BRANCH_SMOOTH,
        {-0.004, -0.0035, 0.0025, 0.0, 0.0, 0.0},
        {-242.47491666032468, -230.91607097565472, -60.57865325852459, 0.0, 0.0, 0.0},
        {52196.51479117316,
         19854.75863315612,
         13986.013986014003,
         0.0,
         0.0,
         0.0,
         19854.75863315612,
         47552.447552447535,
         5320.066541669054,
         0.0,
         0.0,
         0.0,
         13986.013986014003,
         5320.066541669054,
         3747.5411528828263,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         11558.845684669795,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         14194.784809760851,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         13992.020261676924},
        1e-08,
        1.2708493159152567e-05,
    },
    {
        "MAT-P3-LEFT",
        3U,
        SS_MC_BRANCH_LEFT_EDGE,
        {-0.004, 0.0005, 0.0035, 0.0, 0.0, 0.0},
        {-149.0936306828959, -35.557213092692756, -35.55721309269278, 0.0, 0.0, 0.0},
        {51349.434465714934,
         13759.039496883177,
         13759.039496883179,
         0.0,
         0.0,
         0.0,
         13759.039496883177,
         3686.7235218177666,
         3686.723521817781,
         0.0,
         0.0,
         0.0,
         13759.039496883179,
         3686.723521817781,
         3686.723521817774,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         12615.157510022573,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         -2.9178413181041716e-12,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         7569.094506013544},
        1e-08,
        1.2299096852075309e-06,
    },
    {
        "MAT-P3-RIGHT",
        3U,
        SS_MC_BRANCH_RIGHT_EDGE,
        {-0.004, -0.0035, 0.004, 0.0, 0.0, 0.0},
        {-222.27162853531445, -222.27162853531445, -55.16519852097468, 0.0, 0.0, 0.0},
        {34774.81493972069,
         34774.81493972069,
         9317.883580039903,
         0.0,
         0.0,
         0.0,
         34774.81493972069,
         34774.81493972069,
         9317.88358003989,
         0.0,
         0.0,
         0.0,
         9317.883580039903,
         9317.88358003989,
         2496.719380438866,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         1.207847696385627e-12,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         11140.428667622651,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0,
         10444.151875896236},
        1e-08,
        1.6884368960745633e-06,
    },
    {
        "MAT-P3-APEX",
        3U,
        SS_MC_BRANCH_APEX,
        {-0.0015, 0.002, 0.004, 0.0, 0.0, 0.0},
        {6.000000000000001, 6.000000000000001, 6.000000000000001, 0.0, 0.0, 0.0},
        {0.0},
        1e-08,
        0.0,
    },
    {
        "MAT-P3-ROTATED",
        3U,
        /*
         * MAT-001 reports "smooth", but its frozen inequalities select the
         * left edge and its stored stress/tangent are the left-edge values.
         */
        SS_MC_BRANCH_LEFT_EDGE,
        {0.001, -0.002, 0.0005, 0.0012, -0.0007, 0.0009},
        {-21.991753210397206, -83.1200481755701, -20.880878381776473,
         13.555470171446302, -10.613925740509705, 2.248094422840573},
        {6255.836579043851,   12769.646341497137,  4522.315998633843,
         -5987.354224027643,  1967.0657145710934,  -1105.3512239780991,
         12769.646341497137,  49596.29714280565,   12605.458211465357,
         -3136.1964161257756, 1599.385204256514,   423.8445632088216,
         4522.315998633843,   12605.458211465357,  5485.511715476583,
         -2279.8610523134134, 5362.414208053433,   -1209.6816125220528,
         -5987.354224027643,  -3136.1964161257756, -2279.8610523134134,
         9559.187660215635,   -970.7622045718956,  1750.665169302229,
         1967.0657145710934,  1599.385204256514,   5362.414208053433,
         -970.7622045718956,  11474.186888177876,  -2589.8341953009285,
         -1105.3512239780991, 423.8445632088216,   -1209.6816125220528,
         1750.665169302229,   -2589.8341953009285, 865.3579369183757},
        1e-08,
        2.8634676709771156e-06,
    },
};

static double radians(double degrees) {
  return degrees * 0.017453292519943295769236907684886;
}

static SsMcParameterInput oracle_input(const ReductionOracle *oracle) {
  const SsMcParameterInput input = {
      40000.0,
      0.3,
      oracle->cohesion,
      radians(oracle->friction_degrees),
      radians(oracle->dilatancy_degrees),
      oracle->davis,
      17U,
      "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  };
  return input;
}

static SsStatus create_base_parameters(double friction, SsMcParameters **parameters) {
  const SsMcParameterInput input = {
      40000.0,
      0.3,
      6.0,
      friction,
      0.0,
      SS_MC_DAVIS_B,
      17U,
      "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  };
  return ss_mc_parameters_create(&input, parameters, NULL);
}

static size_t strain_components(uint32_t dimension) {
  return dimension == 2U ? 3U : 6U;
}

static size_t stress_components(uint32_t dimension) {
  return dimension == 2U ? 4U : 6U;
}

static size_t tangent_components(uint32_t dimension) {
  return dimension == 2U ? 9U : 36U;
}

static int test_davis(void) {
  size_t index;
  for (index = 0U; index < sizeof(REDUCTION_ORACLES) / sizeof(REDUCTION_ORACLES[0]);
       ++index) {
    const ReductionOracle *oracle = &REDUCTION_ORACLES[index];
    const SsMcParameterInput input = oracle_input(oracle);
    SsMcParameters *parameters = NULL;
    SsMcParameterSummary summary = {0};
    SsMcReduced reduced = {0};
    SsMcError error = {0};
    CHECK(ss_mc_parameters_create(&input, &parameters, &error) == SS_OK);
    CHECK(parameters != NULL && error.valid == 0);
    CHECK(ss_mc_parameters_summary(parameters, &summary) == SS_OK);
    CHECK(summary.davis == oracle->davis);
    CHECK(summary.material_region_id == 17U);
    CHECK(summary.shear_modulus == 15384.615384615385);
    CHECK(summary.bulk_modulus == 33333.33333333333);
    CHECK(summary.lame_first == 23076.92307692307);
    CHECK(strncmp(summary.identity, "sha256:", 7U) == 0);
    CHECK(ss_mc_reduce(parameters, oracle->factor, &reduced, &error) == SS_OK);
    CHECK_NEAR(reduced.c_bar, oracle->c_bar, 3.0e-13, 3.0e-14);
    CHECK_NEAR(reduced.sin_friction, oracle->sin_friction, 3.0e-14, 3.0e-14);
    CHECK(isfinite(reduced.cohesion) && reduced.cohesion >= 0.0);
    CHECK(isfinite(reduced.friction_angle));
    CHECK(isfinite(reduced.beta) && reduced.beta > 0.0);
    CHECK(ss_mc_parameters_destroy(&parameters) == SS_OK && parameters == NULL);
  }
  {
    const double switch_factor = 3.274316085206515;
    ReductionOracle oracle = {
        SS_MC_DAVIS_C, 12.0, 30.0, 10.0, switch_factor, 0.0, 0.0,
    };
    const SsMcParameterInput input = oracle_input(&oracle);
    SsMcParameters *parameters = NULL;
    SsMcReduced below = {0};
    SsMcReduced exact = {0};
    SsMcReduced above = {0};
    CHECK(ss_mc_parameters_create(&input, &parameters, NULL) == SS_OK);
    CHECK(ss_mc_reduce(parameters, nextafter(switch_factor, 0.0), &below, NULL) ==
          SS_OK);
    CHECK(ss_mc_reduce(parameters, switch_factor, &exact, NULL) == SS_OK);
    CHECK(ss_mc_reduce(parameters, nextafter(switch_factor, INFINITY), &above, NULL) ==
          SS_OK);
    CHECK(below.davis_c_high_branch == 1);
    CHECK(exact.davis_c_high_branch == 0);
    CHECK(above.davis_c_high_branch == 0);
    CHECK(ss_mc_parameters_destroy(&parameters) == SS_OK);
  }
  return 0;
}

static int expect_parameter_error(SsMcParameterInput input, const char *code) {
  SsMcParameters *parameters = (SsMcParameters *)(uintptr_t)1U;
  SsMcError error = {0};
  CHECK(ss_mc_parameters_create(&input, &parameters, &error) != SS_OK);
  CHECK(parameters == NULL);
  CHECK(error.valid != 0 && strcmp(error.code, code) == 0);
  return 0;
}

static int test_invalid(void) {
  ReductionOracle oracle = {SS_MC_DAVIS_B, 6.0, 45.0, 0.0, 1.0, 0.0, 0.0};
  SsMcParameterInput input = oracle_input(&oracle);
  SsMcParameters *parameters = NULL;
  SsMcReduced reduced = {7.0, 7.0, 7.0, 7.0, 7.0, 7};
  SsMcError error = {0};

  input.young_modulus = 0.0;
  CHECK(expect_parameter_error(input, "MC_YOUNG_MODULUS") == 0);
  input = oracle_input(&oracle);
  input.poisson_ratio = 0.5;
  CHECK(expect_parameter_error(input, "MC_POISSON_RATIO") == 0);
  input = oracle_input(&oracle);
  input.cohesion = -DBL_MIN;
  CHECK(expect_parameter_error(input, "MC_COHESION") == 0);
  input = oracle_input(&oracle);
  input.friction_angle = INFINITY;
  CHECK(expect_parameter_error(input, "MC_NONFINITE_PARAMETER") == 0);
  input = oracle_input(&oracle);
  input.friction_angle = 1.5707963267948966;
  CHECK(expect_parameter_error(input, "MC_FRICTION_ANGLE") == 0);
  input = oracle_input(&oracle);
  input.dilatancy_angle = -DBL_MIN;
  CHECK(expect_parameter_error(input, "MC_DILATANCY_ANGLE") == 0);
  input = oracle_input(&oracle);
  input.davis = (SsMcDavis)99;
  CHECK(expect_parameter_error(input, "MC_DAVIS_METHOD") == 0);
  input = oracle_input(&oracle);
  input.provenance_digest = "not-a-digest";
  CHECK(expect_parameter_error(input, "MC_PROVENANCE_ID") == 0);
  input = oracle_input(&oracle);
  input.friction_angle = nextafter(1.5707963267948966, 0.0);
  input.dilatancy_angle = input.friction_angle;
  CHECK(expect_parameter_error(input, "MC_DAVIS_DENOMINATOR") == 0);

  input = oracle_input(&oracle);
  CHECK(ss_mc_parameters_create(&input, &parameters, NULL) == SS_OK);
  CHECK(ss_mc_reduce(parameters, 0.0, &reduced, &error) != SS_OK);
  CHECK(error.valid != 0 && strcmp(error.code, "MC_REDUCTION_FACTOR") == 0);
  CHECK(reduced.cohesion == 7.0 && reduced.c_bar == 7.0);
  CHECK(ss_mc_reduce(parameters, NAN, &reduced, &error) != SS_OK);
  CHECK(strcmp(error.code, "MC_REDUCTION_FACTOR") == 0);
  {
    const double invalid_strain[6] = {0.0, 0.0, 0.0, 0.0, NAN, 0.0};
    SsMcResponse response;
    memset(&response, 0x5a, sizeof(response));
    CHECK(ss_mc_evaluate(parameters, (SsMcDimension)4, 1.0, invalid_strain, &response,
                         &error) != SS_OK);
    CHECK(error.valid != 0 && strcmp(error.code, "MC_DIMENSION") == 0);
    CHECK(response.stress[0] != 0.0);
    CHECK(ss_mc_evaluate(parameters, SS_MC_DIMENSION_3, 1.0, invalid_strain, &response,
                         &error) != SS_OK);
    CHECK(strcmp(error.code, "MC_NONFINITE_STRAIN") == 0);
    CHECK(response.stress[0] != 0.0);
  }
  CHECK(ss_mc_parameters_destroy(&parameters) == SS_OK);
  return 0;
}

static int test_branch_oracles(void) {
  SsMcParameters *parameters = NULL;
  SsMcReduced reduced = {0};
  size_t oracle_index;
  CHECK(create_base_parameters(radians(45.0), &parameters) == SS_OK);
  CHECK(ss_mc_reduce(parameters, 1.0, &reduced, NULL) == SS_OK);
  for (oracle_index = 0U;
       oracle_index < sizeof(POINT_ORACLES) / sizeof(POINT_ORACLES[0]);
       ++oracle_index) {
    const PointOracle *oracle = &POINT_ORACLES[oracle_index];
    SsMcResponse response = {0};
    SsMcResponse prepared = {0};
    SsMcError error = {0};
    size_t component;
    CHECK(ss_mc_evaluate(parameters, (SsMcDimension)oracle->dimension, 1.0,
                         oracle->strain, &response, &error) == SS_OK);
    if (error.valid != 0 || response.branch != oracle->branch) {
      fprintf(stderr, "%s: expected branch %d, got %d (error=%s)\n", oracle->id,
              (int)oracle->branch, (int)response.branch,
              error.valid != 0 ? error.code : "none");
    }
    CHECK(error.valid == 0 && response.branch == oracle->branch);
    CHECK(ss_mc_evaluate_prepared(parameters, (SsMcDimension)oracle->dimension,
                                  &reduced, 0U, oracle->strain, &prepared,
                                  &error) == SS_OK);
    CHECK(error.valid == 0 && prepared.branch == response.branch &&
          prepared.robustness_flags == response.robustness_flags);
    CHECK(prepared.potential == response.potential &&
          prepared.trial_yield == response.trial_yield &&
          prepared.yield_residual == response.yield_residual);
    CHECK(isfinite(response.potential) && isfinite(response.trial_yield) &&
          isfinite(response.yield_residual));
    if (response.branch == SS_MC_BRANCH_ELASTIC) {
      CHECK_NEAR(response.yield_residual, response.trial_yield, 2.0e-10, 2.0e-13);
    } else {
      CHECK(response.trial_yield > 0.0);
      CHECK_NEAR(response.yield_residual, 0.0, 2.0e-9, 0.0);
    }
    for (component = 0U; component < stress_components(oracle->dimension);
         ++component) {
      CHECK_NEAR(response.stress[component], oracle->stress[component], 8.0e-10,
                 3.0e-12);
      CHECK(prepared.stress[component] == response.stress[component]);
    }
    for (component = 0U; component < tangent_components(oracle->dimension);
         ++component) {
      CHECK_NEAR(response.tangent[component], oracle->tangent[component], 8.0e-8,
                 3.0e-12);
      CHECK(prepared.tangent[component] == 0.0);
    }
  }
  CHECK(ss_mc_parameters_destroy(&parameters) == SS_OK);
  return 0;
}

static int test_tangent(void) {
  SsMcParameters *parameters = NULL;
  size_t oracle_index;
  CHECK(create_base_parameters(radians(45.0), &parameters) == SS_OK);
  for (oracle_index = 0U;
       oracle_index < sizeof(POINT_ORACLES) / sizeof(POINT_ORACLES[0]);
       ++oracle_index) {
    const PointOracle *oracle = &POINT_ORACLES[oracle_index];
    SsMcResponse center = {0};
    const size_t components = strain_components(oracle->dimension);
    size_t column;
    size_t row;
    CHECK(ss_mc_evaluate(parameters, (SsMcDimension)oracle->dimension, 1.0,
                         oracle->strain, &center, NULL) == SS_OK);
    for (column = 0U; column < components; ++column) {
      double plus_strain[6] = {0.0};
      double minus_strain[6] = {0.0};
      SsMcResponse plus = {0};
      SsMcResponse minus = {0};
      memcpy(plus_strain, oracle->strain, components * sizeof(double));
      memcpy(minus_strain, oracle->strain, components * sizeof(double));
      plus_strain[column] += oracle->finite_difference_step;
      minus_strain[column] -= oracle->finite_difference_step;
      CHECK(ss_mc_evaluate(parameters, (SsMcDimension)oracle->dimension, 1.0,
                           plus_strain, &plus, NULL) == SS_OK);
      CHECK(ss_mc_evaluate(parameters, (SsMcDimension)oracle->dimension, 1.0,
                           minus_strain, &minus, NULL) == SS_OK);
      for (row = 0U; row < components; ++row) {
        const double finite_difference = (plus.stress[row] - minus.stress[row]) /
                                         (2.0 * oracle->finite_difference_step);
        CHECK_NEAR(center.tangent[row + components * column], finite_difference,
                   oracle->finite_difference_error + 3.0e-5, 2.0e-9);
      }
    }
    for (column = 0U; column < components; ++column) {
      for (row = 0U; row < components; ++row) {
        CHECK_NEAR(center.tangent[row + components * column],
                   center.tangent[column + components * row], 5.0e-10, 2.0e-13);
      }
    }
  }
  CHECK(ss_mc_parameters_destroy(&parameters) == SS_OK);
  return 0;
}

static int test_plane(void) {
  static const size_t map[3] = {0U, 1U, 3U};
  SsMcParameters *parameters = NULL;
  size_t oracle_index;
  CHECK(create_base_parameters(radians(45.0), &parameters) == SS_OK);
  for (oracle_index = 0U; oracle_index < 4U; ++oracle_index) {
    const PointOracle *oracle = &POINT_ORACLES[oracle_index];
    const double embedded[6] = {
        oracle->strain[0], oracle->strain[1], 0.0, oracle->strain[2], 0.0, 0.0,
    };
    SsMcResponse response_2d = {0};
    SsMcResponse response_3d = {0};
    size_t column;
    size_t row;
    CHECK(ss_mc_evaluate(parameters, SS_MC_DIMENSION_2, 1.0, oracle->strain,
                         &response_2d, NULL) == SS_OK);
    CHECK(ss_mc_evaluate(parameters, SS_MC_DIMENSION_3, 1.0, embedded, &response_3d,
                         NULL) == SS_OK);
    CHECK(response_2d.branch == response_3d.branch);
    CHECK_NEAR(response_2d.stress[0], response_3d.stress[0], 2.0e-11, 2.0e-13);
    CHECK_NEAR(response_2d.stress[1], response_3d.stress[1], 2.0e-11, 2.0e-13);
    CHECK_NEAR(response_2d.stress[2], response_3d.stress[3], 2.0e-11, 2.0e-13);
    CHECK_NEAR(response_2d.stress[3], response_3d.stress[2], 2.0e-11, 2.0e-13);
    for (column = 0U; column < 3U; ++column) {
      for (row = 0U; row < 3U; ++row) {
        CHECK_NEAR(response_2d.tangent[row + 3U * column],
                   response_3d.tangent[map[row] + 6U * map[column]], 2.0e-9, 2.0e-13);
      }
    }
  }
  CHECK(ss_mc_parameters_destroy(&parameters) == SS_OK);
  return 0;
}

static void strain_to_tensor(const double strain[6], double tensor[9]) {
  memset(tensor, 0, 9U * sizeof(double));
  tensor[0] = strain[0];
  tensor[4] = strain[1];
  tensor[8] = strain[2];
  tensor[1] = tensor[3] = 0.5 * strain[3];
  tensor[5] = tensor[7] = 0.5 * strain[4];
  tensor[2] = tensor[6] = 0.5 * strain[5];
}

static void stress_to_tensor(const double stress[6], double tensor[9]) {
  memset(tensor, 0, 9U * sizeof(double));
  tensor[0] = stress[0];
  tensor[4] = stress[1];
  tensor[8] = stress[2];
  tensor[1] = tensor[3] = stress[3];
  tensor[5] = tensor[7] = stress[4];
  tensor[2] = tensor[6] = stress[5];
}

static void tensor_to_strain(const double tensor[9], double strain[6]) {
  strain[0] = tensor[0];
  strain[1] = tensor[4];
  strain[2] = tensor[8];
  strain[3] = tensor[1] + tensor[3];
  strain[4] = tensor[5] + tensor[7];
  strain[5] = tensor[2] + tensor[6];
}

static void tensor_to_stress(const double tensor[9], double stress[6]) {
  stress[0] = tensor[0];
  stress[1] = tensor[4];
  stress[2] = tensor[8];
  stress[3] = 0.5 * (tensor[1] + tensor[3]);
  stress[4] = 0.5 * (tensor[5] + tensor[7]);
  stress[5] = 0.5 * (tensor[2] + tensor[6]);
}

static void rotate_tensor(const double rotation[9], const double input[9],
                          double output[9]) {
  double work[9] = {0.0};
  size_t row;
  size_t column;
  size_t inner;
  for (row = 0U; row < 3U; ++row) {
    for (column = 0U; column < 3U; ++column) {
      for (inner = 0U; inner < 3U; ++inner) {
        work[row * 3U + column] +=
            rotation[row * 3U + inner] * input[inner * 3U + column];
      }
    }
  }
  memset(output, 0, 9U * sizeof(double));
  for (row = 0U; row < 3U; ++row) {
    for (column = 0U; column < 3U; ++column) {
      for (inner = 0U; inner < 3U; ++inner) {
        output[row * 3U + column] +=
            work[row * 3U + inner] * rotation[column * 3U + inner];
      }
    }
  }
}

static void tangent_action(const double tangent[36], const double strain[6],
                           double stress[6]) {
  size_t row;
  size_t column;
  for (row = 0U; row < 6U; ++row) {
    stress[row] = 0.0;
    for (column = 0U; column < 6U; ++column) {
      stress[row] += tangent[row + 6U * column] * strain[column];
    }
  }
}

static int test_repeated_eigenvalues(void) {
  static const double rotation[9] = {
      0.8, 0.0, 0.6, 0.0, 1.0, 0.0, -0.6, 0.0, 0.8,
  };
  const double base_strain[6] = {-0.004, -0.004, 0.003, 0.0, 0.0, 0.0};
  const double perturbation[6] = {0.3, -0.2, 0.1, 0.4, -0.1, 0.25};
  double base_tensor[9];
  double rotated_tensor[9];
  double rotated_strain[6];
  double base_stress_tensor[9];
  double expected_stress_tensor[9];
  double expected_stress[6];
  double perturbation_tensor[9];
  double rotated_perturbation_tensor[9];
  double rotated_perturbation[6];
  double base_action[6];
  double base_action_tensor[9];
  double expected_action_tensor[9];
  double expected_action[6];
  double rotated_action[6];
  SsMcParameters *parameters = NULL;
  SsMcResponse base = {0};
  SsMcResponse rotated = {0};
  SsMcResponse near = {0};
  double near_strain[6] = {-0.004 + 1.0e-14, -0.004 - 1.0e-14, 0.003, 0.0, 0.0, 0.0};
  size_t component;
  CHECK(create_base_parameters(radians(45.0), &parameters) == SS_OK);
  strain_to_tensor(base_strain, base_tensor);
  rotate_tensor(rotation, base_tensor, rotated_tensor);
  tensor_to_strain(rotated_tensor, rotated_strain);
  CHECK(ss_mc_evaluate(parameters, SS_MC_DIMENSION_3, 1.0, base_strain, &base, NULL) ==
        SS_OK);
  CHECK(ss_mc_evaluate(parameters, SS_MC_DIMENSION_3, 1.0, rotated_strain, &rotated,
                       NULL) == SS_OK);
  CHECK((base.robustness_flags & SS_MC_ROBUST_REPEATED_SPECTRUM) != 0U);
  CHECK((rotated.robustness_flags & SS_MC_ROBUST_REPEATED_SPECTRUM) != 0U);
  stress_to_tensor(base.stress, base_stress_tensor);
  rotate_tensor(rotation, base_stress_tensor, expected_stress_tensor);
  tensor_to_stress(expected_stress_tensor, expected_stress);
  for (component = 0U; component < 6U; ++component) {
    CHECK_NEAR(rotated.stress[component], expected_stress[component], 3.0e-9, 3.0e-12);
  }
  tangent_action(base.tangent, perturbation, base_action);
  stress_to_tensor(base_action, base_action_tensor);
  rotate_tensor(rotation, base_action_tensor, expected_action_tensor);
  tensor_to_stress(expected_action_tensor, expected_action);
  strain_to_tensor(perturbation, perturbation_tensor);
  rotate_tensor(rotation, perturbation_tensor, rotated_perturbation_tensor);
  tensor_to_strain(rotated_perturbation_tensor, rotated_perturbation);
  tangent_action(rotated.tangent, rotated_perturbation, rotated_action);
  for (component = 0U; component < 6U; ++component) {
    CHECK_NEAR(rotated_action[component], expected_action[component], 2.0e-7, 3.0e-11);
  }
  CHECK(ss_mc_evaluate(parameters, SS_MC_DIMENSION_3, 1.0, near_strain, &near, NULL) ==
        SS_OK);
  CHECK((near.robustness_flags & SS_MC_ROBUST_REPEATED_SPECTRUM) != 0U);
  for (component = 0U; component < 36U; ++component) {
    CHECK(isfinite(near.tangent[component]));
  }
  CHECK(ss_mc_parameters_destroy(&parameters) == SS_OK);
  return 0;
}

static int test_zero_friction(void) {
  static const double strains[][6] = {
      {-0.004, 0.0005, 0.0035, 0.0, 0.0, 0.0},
      {-0.002, -0.002, -0.002, 0.0, 0.0, 0.0},
      {0.001, -0.003, 0.0004, 0.001, -0.0003, 0.0007},
  };
  SsMcParameters *zero = NULL;
  SsMcParameters *small = NULL;
  size_t strain_index;
  CHECK(create_base_parameters(0.0, &zero) == SS_OK);
  CHECK(create_base_parameters(1.0e-10, &small) == SS_OK);
  for (strain_index = 0U; strain_index < sizeof(strains) / sizeof(strains[0]);
       ++strain_index) {
    SsMcResponse exact = {0};
    SsMcResponse limit = {0};
    size_t component;
    CHECK(ss_mc_evaluate(zero, SS_MC_DIMENSION_3, 1.0, strains[strain_index], &exact,
                         NULL) == SS_OK);
    CHECK(ss_mc_evaluate(small, SS_MC_DIMENSION_3, 1.0, strains[strain_index], &limit,
                         NULL) == SS_OK);
    CHECK((exact.robustness_flags & SS_MC_ROBUST_ZERO_FRICTION) != 0U);
    CHECK(exact.branch != SS_MC_BRANCH_APEX);
    for (component = 0U; component < 6U; ++component) {
      CHECK_NEAR(exact.stress[component], limit.stress[component], 2.0e-5, 2.0e-10);
    }
    for (component = 0U; component < 36U; ++component) {
      CHECK(isfinite(exact.tangent[component]));
      CHECK_NEAR(exact.tangent[component], limit.tangent[component], 3.0e-3, 3.0e-9);
    }
  }
  CHECK(ss_mc_parameters_destroy(&zero) == SS_OK);
  CHECK(ss_mc_parameters_destroy(&small) == SS_OK);
  return 0;
}

static int check_batch_dimension(SsMcParameters *parameters, size_t first, size_t count,
                                 uint32_t dimension) {
  double strain[SS_MC_MAX_STRAIN_COMPONENTS * 6U] = {0.0};
  double stress[SS_MC_MAX_STRESS_COMPONENTS * 6U] = {0.0};
  double tangent[SS_MC_MAX_TANGENT_COMPONENTS * 6U] = {0.0};
  double potential[6] = {0.0};
  double trial_yield[6] = {0.0};
  double yield_residual[6] = {0.0};
  SsMcBranch branch[6] = {0};
  uint32_t flags[6] = {0};
  uint64_t entities[6] = {0};
  uint32_t quadrature[6] = {0};
  SsMcBatchInput input = {count, count, strain, entities, quadrature};
  SsMcBatchOutput output = {
      count, stress, tangent, potential, trial_yield, yield_residual, branch, flags,
  };
  size_t index;
  size_t component;
  for (index = 0U; index < count; ++index) {
    const PointOracle *oracle = &POINT_ORACLES[first + index];
    entities[index] = 100U + index;
    quadrature[index] = (uint32_t)index;
    for (component = 0U; component < strain_components(dimension); ++component) {
      strain[component * count + index] = oracle->strain[component];
    }
  }
  CHECK(ss_mc_evaluate_batch(parameters, (SsMcDimension)dimension, 1.0, &input, &output,
                             NULL) == SS_OK);
  for (index = 0U; index < count; ++index) {
    const PointOracle *oracle = &POINT_ORACLES[first + index];
    SsMcResponse scalar = {0};
    CHECK(ss_mc_evaluate(parameters, (SsMcDimension)dimension, 1.0, oracle->strain,
                         &scalar, NULL) == SS_OK);
    CHECK(branch[index] == scalar.branch && flags[index] == scalar.robustness_flags);
    CHECK(potential[index] == scalar.potential);
    CHECK(trial_yield[index] == scalar.trial_yield);
    CHECK(yield_residual[index] == scalar.yield_residual);
    for (component = 0U; component < stress_components(dimension); ++component) {
      CHECK(stress[component * count + index] == scalar.stress[component]);
    }
    for (component = 0U; component < tangent_components(dimension); ++component) {
      CHECK(tangent[component * count + index] == scalar.tangent[component]);
    }
  }
  return 0;
}

static int test_batch(void) {
  SsMcParameters *parameters = NULL;
  double strain[6U * 4U];
  double stress[6U * 4U];
  double tangent[36U * 4U];
  double potential[4];
  double trial_yield[4];
  double yield_residual[4];
  SsMcBranch branch[4];
  uint32_t flags[4];
  uint64_t entities[4] = {100U, 50U, 75U, 20U};
  uint32_t quadrature[4] = {3U, 2U, 1U, 9U};
  SsMcBatchInput input = {4U, 4U, strain, entities, quadrature};
  SsMcBatchOutput output = {
      4U, stress, tangent, potential, trial_yield, yield_residual, branch, flags,
  };
  SsMcError error = {0};
  size_t index;
  CHECK(create_base_parameters(radians(45.0), &parameters) == SS_OK);
  CHECK(check_batch_dimension(parameters, 0U, 4U, 2U) == 0);
  CHECK(check_batch_dimension(parameters, 4U, 6U, 3U) == 0);
  {
    int parallel_status[64] = {0};
    int batch;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (batch = 0; batch < 64; ++batch) {
      parallel_status[batch] = check_batch_dimension(parameters, 4U, 6U, 3U);
    }
    for (batch = 0; batch < 64; ++batch) {
      CHECK(parallel_status[batch] == 0);
    }
  }
  for (index = 0U; index < sizeof(strain) / sizeof(strain[0]); ++index) {
    strain[index] = 0.0;
  }
  strain[0U * 4U + 1U] = NAN;
  strain[2U * 4U + 3U] = NAN;
  for (index = 0U; index < sizeof(stress) / sizeof(stress[0]); ++index) {
    stress[index] = -777.0;
  }
  for (index = 0U; index < sizeof(tangent) / sizeof(tangent[0]); ++index) {
    tangent[index] = -777.0;
  }
  CHECK(ss_mc_evaluate_batch(parameters, SS_MC_DIMENSION_3, 1.0, &input, &output,
                             &error) != SS_OK);
  CHECK(error.valid != 0 && error.local_index == 3U);
  CHECK(error.stable_entity_id == 20U && error.quadrature_index == 9U);
  for (index = 0U; index < sizeof(stress) / sizeof(stress[0]); ++index) {
    CHECK(stress[index] == -777.0);
  }
  for (index = 0U; index < sizeof(tangent) / sizeof(tangent[0]); ++index) {
    CHECK(tangent[index] == -777.0);
  }
  CHECK(ss_mc_parameters_destroy(&parameters) == SS_OK);
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s TEST-ID\n", argv[0]);
    return 2;
  }
  if (strcmp(argv[1], "MAT-UNIT-DAVIS-001") == 0) {
    return test_davis();
  }
  if (strcmp(argv[1], "MAT-NEG-INPUT-001") == 0) {
    return test_invalid();
  }
  if (strcmp(argv[1], "MAT-UNIT-BRANCH-001") == 0) {
    return test_branch_oracles();
  }
  if (strcmp(argv[1], "MAT-UNIT-TANGENT-001") == 0) {
    return test_tangent();
  }
  if (strcmp(argv[1], "MAT-REG-ZEROPHI-001") == 0) {
    return test_zero_friction();
  }
  if (strcmp(argv[1], "MAT-REG-EIGEN-001") == 0) {
    return test_repeated_eigenvalues();
  }
  if (strcmp(argv[1], "MAT-UNIT-PLANE-001") == 0) {
    return test_plane();
  }
  if (strcmp(argv[1], "MAT-UNIT-BATCH-001") == 0) {
    return test_batch();
  }
  fprintf(stderr, "unsupported test id: %s\n", argv[1]);
  return 2;
}
