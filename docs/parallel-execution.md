# Parallel execution

A run uses `R` MPI ranks with `T` OpenMP threads each, written `RxT`. This
page explains what each level parallelizes, how the launcher places ranks and
threads, and what measurements on one workstation suggest for choosing a
layout.

## What runs where

- **MPI ranks.** PETSc partitions the mesh with ParMETIS (or PT-Scotch).
  Each rank reads its share of the compiled package with parallel HDF5, owns
  the cells and unknowns of its partition plus one layer of ghost cells, and
  takes part in every PETSc vector, matrix, Krylov, PMG and deflation
  operation.
- **OpenMP threads.** Inside a rank, a persistent thread team evaluates the
  element-level work: residual and tangent assembly, the matrix-free operator
  apply and its block inverses, the Mohr–Coulomb return mapping and the
  coupling body forces. Each thread writes disjoint rows in a fixed order.
  The distributed matrix-free apply evaluates cells on their owners when the
  exchange schedule fits the workspace budget. It overlaps the coefficient
  exchange with the interior cells, and the return of off-rank contributions
  with the sum of local-only rows; each row owner adds contributions in its
  existing cell order.
- **Master thread.** MPI is initialized with `MPI_THREAD_FUNNELED`. All MPI,
  PETSc and HDF5 calls are made by the master thread outside the OpenMP
  regions of the project. PETSc can use the same OpenMP team for supported
  sparse kernels when built with `--with-openmp-kernels=1`. HYPRE and BLAS
  use one thread per rank in the documented builds.

A layout must fit the physical cores: `R × T` may not exceed the number of
cores, and hardware threads (SMT siblings) are not counted. The launcher
refuses an oversubscribed layout before starting anything.

## Placement and binding

`slope-stability run --layout RxT` launches

```text
mpiexec -n R --map-by ppr:N:package:pe=T --bind-to core slope-engine run ...
```

where `N = ceil(R / packages)`. Ranks are spread evenly over the CPU packages
and each rank is bound to `T` whole cores of one package, so rank masks are
disjoint. `--verify-binding` launches a probe with the same arguments first and
checks the observed masks; the catalog run report records them under
`observed_binding`. For `4x8` on a 32-core single-package machine:

```json
{"0": [0, 1, 2, 3, 4, 5, 6, 7], "1": [8, 9, 10, 11, 12, 13, 14, 15],
 "2": [16, 17, 18, 19, 20, 21, 22, 23], "3": [24, 25, 26, 27, 28, 29, 30, 31]}
```

The launcher uses this OpenMP and library thread environment for the engine.

```text
OMP_NUM_THREADS=T  OMP_PROC_BIND=close  OMP_PLACES=cores  OMP_DYNAMIC=FALSE
OMP_MAX_ACTIVE_LEVELS=1  OMP_WAIT_POLICY=PASSIVE  GOMP_SPINCOUNT=20000
OPENBLAS_NUM_THREADS=1  MKL_NUM_THREADS=1  BLIS_NUM_THREADS=1  NUMEXPR_NUM_THREADS=1
```

With GCC's libgomp, workers spin briefly before sleeping. The bounded spin
reduces wake-up time for repeated matrix-free applies and lets workers sleep
during longer PETSc operations. Set `OMP_WAIT_POLICY` or `GOMP_SPINCOUNT`
before `slope-stability run` to override the defaults. Setting only
`OMP_WAIT_POLICY` leaves the spin count to the runtime. Use
`OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0` for immediate sleep.

Reproduce this environment and the binding when you launch `slope-engine`
yourself, for example under a batch scheduler. `--binding none` disables
binding, which is useful only for debugging. Placement across several NUMA
domains or nodes has not been measured.

`--coarse-active-ranks N` limits the coarsest PMG level to `N` ranks; the
measurements below used two.

## The 4x8 layout

`4x8` is the reference hybrid layout. The hybrid assembly tests run it, and
the documentation examples were verified with it on a 32-core machine. Use it
to check a hybrid installation even when production runs use another
layout.

## Measured guidance

The numbers below were measured during development, before this release, on
one workstation: AMD Ryzen Threadripper PRO 7975WX (32 cores, 64 hardware
threads, one NUMA node), GCC 16.2.1, Open MPI 5.0.11, parallel HDF5 2.2.0 and
PETSc 3.24.2 with 32-bit indices and SuperLU_DIST, with verified core binding.
These measurements used the PETSc build with serial sparse kernels.
The workstation ran other jobs concurrently, so the timings vary between
waves. They show tendencies on one machine, not performance guarantees.

### Complete strength-reduction runs

`CASE-021` (3D SSR, 15,356 tetrahedra, P2 elements with straight geometry)
with its catalog policies, four accepted states, two coarse active ranks and
terminal output. Each run made 47 outer linear solves. Times are medians of
three paired executions.

| Layout | Outer Krylov iterations | Median engine wall time |
|---|---:|---:|
| `4x8` | 448 | 14.1 s |
| `32x1` | 444 | 8.1 s |

With these policies the mechanics operators are assembled, and the pure-MPI
layout was about 1.7 times faster. PETSc's sparse-matrix, vector and Krylov
operations run on one thread per rank, so a layout with fewer ranks leaves
that part of the solver on fewer cores. An earlier measurement wave gave a
`4x8` median of 13.1 s, which indicates the timing spread on the shared
machine.

The same case also shows the memory effect of hybrid layouts. In a
twelve-state run, the PETSc-allocated memory summed over all ranks stayed
constant from the fifth to the twelfth continuation transaction, at
715.5 MiB for `4x8` and 1085.7 MiB for `32x1`.

Complete runs of the same case to the continuation criterion, with bounded
waiting (the default of this release), two coarse active ranks and terminal
output, took 85.2 s at `4x8` and 60.9 s at `32x1` (one profiled launch per
layout).
At `4x8`, the maximum-rank PETSc event totals were 30.1 s in `MatMult`,
30.6 s in `PCApply` (including smoothing), 6.2 s in numeric Galerkin
products, 3.8 s in `VecMDot` and 0.04 s in `VecAXPY`. The events overlap.
Sparse matrix-vector products and smoothing dominate, and they run on one
thread per rank; dot products still require MPI reductions.

### Matrix-free operator apply

A curved P4 operator with order-4 geometry on 3,072 tetrahedra and a
degree-14 quadrature, applied after linearization. The table gives the median
of three repeats of the slowest rank's mean time per apply.

| Layout | Apply time |
|---|---:|
| `1x32` | 1.45 ms |
| `4x8` | 1.82 ms |
| `32x1` | 3.94 ms |

The matrix-free apply was fastest with threads and slowest with pure MPI.
A later measurement of the same operator with the overlapped exchange and
bounded waiting of this release gave medians of 1.71 ms at `1x32`, 1.57 ms
at `4x8` and 1.83 ms at `32x1` over three interleaved launches; the two
waves ran under different background load.
In an earlier measurement on the same operator, assembling the tangent matrix
took 223 ms at `1x32`, 191 ms at `4x8` and 168 ms at `32x1`, so assembly
favored ranks instead.

### Memory per rank

`CASE-010` (3D, 55,895 tetrahedra, P2 mechanics and seepage) in
`slope-engine validate`, which builds the distributed mesh, fields,
constraints and assembly plans without solving, with one thread per rank.
Peak resident set sizes per process, median over ranks, in MiB:

| Ranks | Owned + ghost cells per rank | Peak RSS after input | Peak RSS after setup | Mesh-input heap per rank |
|---:|---:|---:|---:|---:|
| 1 | 55,895 + 0 | 109.1 | 466.3 | 14.05 |
| 2 | 27,948 + 1,843 | 108.6 | 303.4 | 7.11 |
| 4 | 14,070 + 2,149 | 89.5 | 223.4 | 3.74 |
| 8 | 6,968 + 2,003 | 78.6 | 183.5 | 2.03 |
| 16 | 3,517 + 1,499 | 74.0 | 171.7 | 1.19 |
| 32 | 1,732 + 1,237 | 74.9 | 162.8 | 0.75 |

The mesh input divides between ranks, and rank 0 needs about as much input
memory as the others. Each process, however, carries a fixed cost of about
75 MiB for the MPI, PETSc and HDF5 runtimes, and the ghost layer grows
relative to the owned cells. Summed over ranks, the setup peak therefore grows from 466 MiB on one
rank to about 5,200 MiB on 32. When memory per node is the constraint, prefer
fewer ranks with more threads.

### Choosing a layout

- No layout is best for every case. Compare two or three layouts on a short
  run of your own case (`--maximum-accepted-states 4`) before a long run.
- For assembled operators with low polynomial order, include pure MPI
  (`Rx1`); on the workstation above it was the fastest. Start with `32x1`
  for an assembled case like `CASE-021` when memory permits.
- For matrix-free high-order operators, include a hybrid layout. With
  bounded waiting and the overlapped exchange, the curved P4/G4 apply
  medians over three interleaved launches were 1.71 ms at `1x32`, 1.57 ms
  at `4x8` and 1.83 ms at `32x1`.
- When memory is tight, reduce the number of ranks and increase the threads.
- Keep `R × T` at or below the number of physical cores.

## Reproducibility

Results do not depend on the OpenMP team size when
`validation.qualification_mode` is set: it fixes the order of element
contributions within each assembled row for mechanics and seepage. For
assembled PMG with MPIAIJ, PETSc's Galerkin backend gathers remote contributions
into rank-ordered PetscSF slots and sums them on the row owner. Each numeric
refresh uses the local product layouts and the COO insertion plan from
symbolic setup.

In performance mode, the default, PETSc's Galerkin product can sum off-rank
contributions in message completion order, so repeated multi-rank runs at
the same layout can differ slightly. Different layouts and solver backends
change partitions and reductions, and hence floating-point results, in both
modes. Compare layouts or backends with numerical tolerances
([outputs.md](outputs.md#reading-results-from-python)). The result records
the effective mode as `summary.mechanics_assembly_mode`. Debug builds can
print rank-local matrix and vector digests on standard error with
`SLOPE_DEBUG_STAGE_DIGEST=1`.

## Stopping parallel runs

Send `SIGTERM` or `SIGINT` to the engine ranks of the run, not to `mpiexec`,
for example by matching the result directory in their command lines:

```sh
pkill -TERM -f '^[^ ]*slope-engine run .*run-4x8'
```

The ranks agree on a safe point, publish a `cancelled` result with a restart
checkpoint of the last accepted state (if one was accepted), and exit with
status 6. Open MPI aborts
the whole job when its launcher receives `SIGTERM`, and nothing is published.
`slope-stability run --timeout-seconds` and `Ctrl-C` signal the ranks in this
way ([running.md](running.md#stopping-a-run)).
