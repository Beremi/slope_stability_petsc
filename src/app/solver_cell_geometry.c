/* Vertex coordinates in reference cell order for solver and coupling.
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include "solver_internal.h"

SsStatus ss_solver_cell_vertex_coordinates(const SsCellPack *pack,
                                           const SsFeDescriptor *coordinate_descriptor,
                                           double *vertex_coordinates) {
  size_t node;
  size_t found = 0U;
  if (pack == NULL || coordinate_descriptor == NULL || vertex_coordinates == NULL ||
      pack->coordinate_node_count != coordinate_descriptor->node_count ||
      pack->dimension != coordinate_descriptor->dimension) {
    return SS_ERROR_INVALID_ARGUMENT;
  }
  for (node = 0U; node < coordinate_descriptor->node_count; ++node) {
    const uint8_t *alpha = coordinate_descriptor->barycentric_indices +
                           node * ((size_t)pack->dimension + 1U);
    uint32_t barycentric;
    for (barycentric = 0U; barycentric <= pack->dimension; ++barycentric) {
      uint32_t other;
      int vertex = alpha[barycentric] == coordinate_descriptor->order;
      for (other = 0U; other <= pack->dimension && vertex != 0; ++other) {
        if (other != barycentric && alpha[other] != 0U) {
          vertex = 0;
        }
      }
      if (vertex != 0) {
        uint32_t coordinate;
        for (coordinate = 0U; coordinate < pack->dimension; ++coordinate) {
          vertex_coordinates[(size_t)barycentric * pack->dimension + coordinate] =
              pack->coordinate_coordinates[node * (size_t)pack->dimension + coordinate];
        }
        ++found;
        break;
      }
    }
  }
  return found == (size_t)pack->dimension + 1U ? SS_OK : SS_ERROR_INTEGRITY;
}
