# Native engine sources

The C11 sources of `slope-engine` and `slope-probe`. Each directory is one
module with a single responsibility; its private interface is declared in
`include/slope_internal/<module>/` and module-internal headers stay next to
the sources (`*_internal.h`).

| Directory | Responsibility |
|---|---|
| `core/` | Status codes, collective status agreement, SHA-256, sorting, helpers |
| `protocol/` | Strict JSON and the `slope-schema-check` tool |
| `io/` | Package verification and parallel input |
| `mesh/` | Distributed DMPlex mesh, supports, geometry, field spaces and constraints |
| `fe/` | Lagrange bases, quadrature and geometry maps |
| `material/` | Mohr–Coulomb with Davis A/B/C, self-weight and prescribed saturation |
| `assembly/` | Assembly plans and kernels, assembled and matrix-free operators |
| `hydraulic/` | Steady unconfined seepage |
| `coupling/` | Seepage-to-mechanics transfer |
| `linear/` | Krylov solves and stopping policies |
| `pmg/` | Polynomial multigrid |
| `deflation/` | Deflation spaces and projectors |
| `nonlinear/` | Newton solves |
| `analysis/` | Limit-load and SSR continuation |
| `result/` | Result and checkpoint I/O |
| `diagnostics/` | Error records, events and the progress journal |
| `build/` | Compiled-in feature checks |
| `app/` | Engine entry point, solver setup, publication and run monitoring |

[docs/architecture.md](../docs/architecture.md) describes the modules, the
flow of a run and the files of `app/`. Each header documents its functions
(Doxygen; `doxygen docs/Doxyfile` renders `include/`). Conventions for new code are in
[CONTRIBUTING.md](../CONTRIBUTING.md).
