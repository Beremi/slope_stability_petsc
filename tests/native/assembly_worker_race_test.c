/* Assembly OpenMP worker kernels without PETSc or MPI, for the
 * ThreadSanitizer and Archer race check of the tsan-kernels preset.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "slope_internal/assembly/assembly.h"

#include <omp.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

enum {
  WORK_ITEMS = 512,
  EXPECTED_THREADS = 8,
};

static double reference_coordinate(const SsFeDescriptor *descriptor, size_t node,
                                   uint32_t coordinate) {
  const size_t offset =
      node * (size_t)(descriptor->dimension + 1U) + (size_t)coordinate + 1U;
  return (double)descriptor->barycentric_indices[offset] / (double)descriptor->order;
}

int main(void) {
  const SsMcParameterInput material_input = {
      8.0e4,
      0.27,
      1.0e6,
      0.45,
      0.2,
      SS_MC_DAVIS_B,
      UINT64_C(1),
      "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  };
  const SsFeDescriptor *field_descriptor = NULL;
  const SsFeDescriptor *coordinate_descriptor = NULL;
  SsFeQuadratureRequest quadrature_request = {
      2U,
      SS_FE_QUADRATURE_VOLUME,
      SS_FE_OPERATION_MECHANICS_STIFFNESS,
      4U,
      4U,
      0U,
      SS_FE_QUADRATURE_MODE_DEFAULT,
      NULL,
  };
  SsFeQuadraturePlan quadrature = {0};
  SsMcParameters *material = NULL;
  SsMcError material_error = {0};
  double coordinates[SS_FE_MAX_NODES * SS_FE_MAX_DIMENSION] = {0.0};
  double first[WORK_ITEMS] = {0.0};
  double second[WORK_ITEMS] = {0.0};
  int observed_threads = 0;
  int repeat;
  size_t node;
  if (ss_fe_descriptor_find(2U, 4U, &field_descriptor) != SS_OK ||
      ss_fe_descriptor_find(2U, 4U, &coordinate_descriptor) != SS_OK ||
      ss_fe_quadrature_plan(&quadrature_request, &quadrature) != SS_OK ||
      quadrature.selected_rule == NULL ||
      ss_mc_parameters_create(&material_input, &material, &material_error) != SS_OK) {
    fputs("cannot create the P4 project worker fixture\n", stderr);
    return 2;
  }
  for (node = 0U; node < coordinate_descriptor->node_count; ++node) {
    const double r = reference_coordinate(coordinate_descriptor, node, 0U);
    const double s = reference_coordinate(coordinate_descriptor, node, 1U);
    coordinates[2U * node] = 2.0 * r + 0.1 * r * s;
    coordinates[2U * node + 1U] = 1.5 * s + 0.05 * r * s;
  }
  omp_set_dynamic(0);
  for (repeat = 0; repeat < 2; ++repeat) {
    double *checksums = repeat == 0 ? first : second;
    int item;
#pragma omp parallel num_threads(EXPECTED_THREADS)
    {
#pragma omp single
      observed_threads = omp_get_num_threads();
#pragma omp for schedule(static)
      for (item = 0; item < WORK_ITEMS; ++item) {
        double field_values[2U * SS_FE_MAX_NODES] = {0.0};
        double residual[2U * SS_FE_MAX_NODES] = {0.0};
        double jacobian[4U * SS_FE_MAX_NODES * SS_FE_MAX_NODES] = {0.0};
        const double body_force[3] = {0.3, -1.7, 0.0};
        const double coupling_force[3] = {-0.04, 0.08, 0.0};
        SsElementOutput output = {
            2U * SS_FE_MAX_NODES,
            4U * SS_FE_MAX_NODES * SS_FE_MAX_NODES,
            residual,
            jacobian,
            0.0,
        };
        SsMechanicsElementInput input = {
            .dimension = 2U,
            .field_descriptor = field_descriptor,
            .coordinate_descriptor = coordinate_descriptor,
            .quadrature = quadrature.selected_rule,
            .coordinate_values = coordinates,
            .field_values = field_values,
            .body_force = body_force,
            .coupling_body_force = coupling_force,
            .material = material,
            .reduction_factor = 1.0,
            .external_load_factor = 0.75,
            .stable_cell_id = UINT64_C(0x510000000000) + (SsEntityId)item,
            .coupling_point_kernel = NULL,
            .coupling_point_context = NULL,
        };
        SsErrorRecord error = {0};
        double checksum = 0.0;
        size_t field_node;
        size_t row;
        for (field_node = 0U; field_node < field_descriptor->node_count; ++field_node) {
          const double r = reference_coordinate(field_descriptor, field_node, 0U);
          const double s = reference_coordinate(field_descriptor, field_node, 1U);
          const double perturbation = 1.0e-8 * (double)(item + 1);
          field_values[2U * field_node] = 1.0e-4 * (0.5 * r - 0.2 * s) + perturbation;
          field_values[2U * field_node + 1U] =
              1.0e-4 * (0.1 * r + 0.4 * s) - perturbation;
        }
        if (ss_assembly_mechanics_element(&input,
                                          SS_ASSEMBLY_RESIDUAL | SS_ASSEMBLY_JACOBIAN |
                                              SS_ASSEMBLY_POTENTIAL,
                                          &output, &error) != SS_OK ||
            error.valid != 0) {
          checksums[item] = NAN;
          continue;
        }
        for (row = 0U; row < 2U * field_descriptor->node_count; ++row) {
          checksum += residual[row] * (double)(row + 1U);
          checksum +=
              jacobian[row * (2U * field_descriptor->node_count) + row] * 1.0e-6;
        }
        checksums[item] = checksum + output.potential;
      }
    }
  }
  if (ss_mc_parameters_destroy(&material) != SS_OK) {
    return 3;
  }
  if (observed_threads != EXPECTED_THREADS) {
    fprintf(stderr, "expected %d OpenMP workers, observed %d\n", EXPECTED_THREADS,
            observed_threads);
    return 4;
  }
  for (node = 0U; node < WORK_ITEMS; ++node) {
    if (!isfinite(first[node]) || first[node] != second[node]) {
      fprintf(stderr, "non-deterministic project element at item %zu\n", node);
      return 5;
    }
  }
  printf("{\"kernel\":\"mechanics-P4-G4\",\"openmp_team\":%d,"
         "\"race_free\":true,\"work_items\":%d}\n",
         observed_threads, WORK_ITEMS);
  return 0;
}
