# Running

A run passes through four artifacts. The Python command `slope-stability`
produces the first three and launches the engine for the last:

1. **Authoring documents** (JSON): an asset, a case and its solver profiles
   ([input-format.md](input-format.md)).
2. **Compiled problem package**: `problem.json` plus `problem.h5`, identified
   by the SHA-256 of its content.
3. **Resolved run**: `resolved-run.json`, which fixes every solver policy, the
   MPI × OpenMP layout and the installed build.
4. **Result directory** written by `slope-engine run`
   ([outputs.md](outputs.md)).

[architecture.md](architecture.md) describes this pipeline in more detail.

## The `slope-stability` command

Install the Python package and an engine build first
([building.md](building.md)). The commands below assume these variables:

```sh
export PREFIX=$SLOPE_INSTALL_ROOT/release-hybrid-i32    # installed engine
export PETSC_LIB=$PETSC_DIR/$PETSC_ARCH/lib             # PETSc runtime libraries
```

Every subcommand prints JSON lines on standard output. The exit status is 0
on success, 1 when a verification or a run failed, and 2 on invalid input.

The catalog commands locate `cases/catalog` and `meshes/` by searching
`$SLOPE_STABILITY_ROOT`, then the current directory and its parents. Run them
inside the source tree, set `SLOPE_STABILITY_ROOT`, or pass `--catalog DIR`
and `--meshes DIR` before the subcommand.

### `verify-meshes`

```sh
slope-stability verify-meshes
```

Checks every file in `meshes/` against `meshes/SHA256SUMS`, the catalog index
and the `mesh_source.sha256` of each asset document (`--no-assets` skips the
asset check). One line is printed per mesh and a final summary with
`"status":"pass"` or `"fail"`.

### `compile`

```sh
slope-stability compile --case-id CASE-003 --destination $WORK/compiled
```

Compiles catalog variants into problem packages. `--case-id` and
`--variant-id` may be repeated; without them all 24 variants are compiled.
`--uniform-refinement-levels N` splits every straight simplex `N` times
before compilation, which gives larger problems from the same catalog case.
The destination holds:

```text
compiled/
├── compile-catalog.json                 # catalog identity and one row per variant
└── packages/sha256-<problem id>/
    ├── problem.json                     # strict manifest of the problem
    ├── problem.h5                       # mesh, supports and coordinates
    └── compile-report.json              # unit conversions and geometry audit
```

The package name is the problem identity, so compiling the same inputs again
yields the same package. Compilation refuses a mesh whose SHA-256 differs from
the asset document.

### `run`

```sh
slope-stability run --prefix $PREFIX --runtime-library-path $PETSC_LIB \
    --compiled $WORK/compiled --case-id CASE-003 --layout 1x1 \
    --destination $WORK/run-1x1
slope-stability run --prefix $PREFIX --runtime-library-path $PETSC_LIB \
    --compiled $WORK/compiled --case-id CASE-003 --layout 4x8 \
    --verify-binding --destination $WORK/run-4x8
```

For each selected variant, `run` resolves the case against the installed
build, launches `mpiexec` with the requested layout, waits for the engine and
verifies the result against its run, problem and build identities. A variant
passes when the engine exits with 0 and the verified result has terminal
status `succeeded`. The destination must not exist. Output is written to a
hidden staging directory next to it and renamed when all variants have
finished:

```text
run-4x8/
├── run-catalog.json                     # layout, binding, identities, one row per variant
├── resolved/case-003-canonical/sha256-<run id>/resolved-run.json
├── results/case-003-canonical/          # result directory (outputs.md)
├── results/case-003-canonical.run/      # journal and in-run checkpoints
└── logs/case-003-canonical.{stdout,stderr}
```

Launch options:

| Option | Meaning |
|---|---|
| `--layout RxT` | `R` MPI ranks with `T` OpenMP threads each ([parallel-execution.md](parallel-execution.md)) |
| `--binding core` / `none` | Bind each rank to `T` whole cores (default) or do not bind |
| `--verify-binding` | Launch a probe first and check that the rank CPU masks are disjoint |
| `--runtime-library-path DIR` | Prepend `DIR` to `LD_LIBRARY_PATH` of the engine |
| `--mpiexec`, `--numproc-flag` | Launcher and its rank-count flag (default `mpiexec`, `-n`) |
| `--coarse-active-ranks N` | Ranks active on the coarsest PMG level (default: all) |
| `--petsc-threads N` | Override the recorded PETSc kernel team size. By default it is `T` with OpenMP-enabled PETSc kernels and 1 with serial kernels. |
| `--timeout-seconds S` | Stop a variant after `S` seconds (default 86400) |
| `--checkpoint-every N`, `--checkpoint-seconds S`, `--checkpoint-keep K` | In-run checkpoints ([outputs.md](outputs.md#checkpoints-and-restart)) |
| `--quiet` | No engine progress lines on standard error |
| `--operator-workspace-mib M` | Memory cap, in MiB, of the mechanics operator workspace |
| `--address-space-limit-bytes B` | `RLIMIT_AS` applied to the launched processes |

The launcher sets `OMP_NUM_THREADS=T`, `OMP_PROC_BIND=close`,
`OMP_PLACES=cores`, `OMP_DYNAMIC=FALSE`, a bounded spin before sleeping
(`OMP_WAIT_POLICY=PASSIVE`, `GOMP_SPINCOUNT=20000`) and one thread for
OpenBLAS, MKL, BLIS and NumExpr ([parallel-execution.md](parallel-execution.md#placement-and-binding)).
PETSc's threaded sparse kernels use the launcher's `OMP_NUM_THREADS`.
In catalog mode it rejects an inherited non-empty `PETSC_OPTIONS`, so that
hidden PETSc options cannot change a recorded run.

### Policy overrides

`resolve` and `run` accept typed overrides of the selected solver profiles,
applied after the case's own overrides. Examples:

```sh
slope-stability run --prefix $PREFIX --runtime-library-path $PETSC_LIB \
    --compiled $WORK/compiled --case-id CASE-003 --layout 2x2 \
    --maximum-accepted-states 4 --linear-relative-tolerance 1e-3 \
    --destination $WORK/run-short
```

Typed flags exist for the nonlinear and linear tolerances and limits, the
continuation horizon (`--maximum-accepted-states`, `--maximum-work`, the
stopping criteria), the PMG hierarchy and smoothers (`--pmg-levels 1 2 4`),
deflation and the output frequency. `slope-stability run --help` lists them,
and `--set PATH=JSON` sets any other resolvable policy path, for example
`--set 'pmg.levels=[1,2]'`. The overrides are part of the resolved run and
therefore of its identity. [solver-options.md](solver-options.md) describes
the typed solver settings.

### `resolve` and direct runs

For a case outside the catalog, or to keep a resolved run for later, resolve
first and launch the resolved run directly:

```sh
slope-stability resolve --case-id CASE-003 \
    --problem-package $WORK/compiled/packages/sha256-<problem id> \
    --prefix $PREFIX --layout 2x2 --destination $WORK/resolved
slope-stability run --prefix $PREFIX --runtime-library-path $PETSC_LIB \
    --package $WORK/compiled/packages/sha256-<problem id> \
    --resolved-run $WORK/resolved/sha256-<run id>/resolved-run.json \
    --destination $WORK/run-direct
```

`resolve --case FILE --profiles-dir DIR` reads a case document and its
profiles from your own files instead of the catalog. `resolve` takes the same
layout, binding, memory-cap and override options as `run`. The layout is part of
the run identity, so a direct run launches exactly the ranks and threads
recorded in `resolved-run.json`, and only with the build it was resolved for.
In direct mode the result directory is the destination itself.

### `inspect`

```sh
slope-stability inspect $WORK/run-4x8/results/case-003-canonical
```

Prints a JSON summary of a result directory, problem package, resolved run,
compilation, catalog run, catalog or mesh file. For a result it reports the
identities, terminal status, analysis outcome, quantities and restart parent.
[outputs.md](outputs.md) shows how to read the result data from Python.

## Progress and the journal

While a run is in progress, rank 0 prints one line per phase and accepted
state on standard error (stored in `logs/<variant>.stderr` by the catalog
runner):

```text
slope-engine [      0.2s] accepted state 1: factor=0.9 work=425.171 u_max=0.07671 attempts=1
```

The same events are appended, one JSON object per line and flushed after each
line, to `journal.jsonl` in the run directory
(`results/<variant>.run/` by default). Event types are `start`, `phase`,
`solve`, `attempt`, `attempt_rolled_back`, `accepted`, `checkpoint`,
`checkpoint_failed` and `terminal`:

```json
{"seq":7,"elapsed":0.226204,"time":1795097889,"event":"accepted","state":1,"accepted_ordinal":0,"factor":0.90000000000000002,"work":425.17086029653859,"control":425.17086029653859,"displacement_maximum":0.076706960351560702,"attempts":1,"next_step":0.10000000000000001}
```

The journal is rotated to `journal.jsonl.1` after 8 MiB. A journal write
failure disables the journal but does not stop the run. The journal is a
monitoring aid; the result directory is the authoritative output.

## Stopping a run

The engine stops at its next collective safe point when its ranks receive
`SIGINT` or `SIGTERM`. It then publishes a result with status `cancelled`
and, when at least one state was accepted, a restart checkpoint of the last
accepted state ([outputs.md](outputs.md)).

- Signal the engine ranks of the run, not `mpiexec`. The ranks' command
  lines start with the engine path and contain the result directory, so a
  pattern anchored at the start selects them and not the launcher:

  ```sh
  pkill -TERM -f '^[^ ]*slope-engine run .*run-4x8'
  ```

  Open MPI aborts the whole job when the launcher itself receives `SIGTERM`,
  and nothing is published. `pkill -x slope-engine` would also stop every
  other engine run of the same user.
- `--timeout-seconds` and `Ctrl-C` on `slope-stability run` signal the ranks
  in the same way and fall back to `SIGKILL` after a grace period. After
  `Ctrl-C` the partial output stays in the hidden staging directory, whose
  path is printed.

## The engine command line

`slope-stability run` is the supported launcher. The engine can also be
started directly, for example under a batch system:

```text
slope-engine validate <problem-dir> <resolved-run.json> <result-dir>
slope-engine run      <problem-dir> <resolved-run.json> <result-dir> [options]
slope-engine --version | --help
```

`validate` verifies the inputs, builds the distributed mesh, fields,
constraints and assembly plans, and stops without solving.
The `run` options affect only monitoring and checkpoint files, never the
numerical results or the run identity:

| Option | Default |
|---|---|
| `--run-directory DIR` | `<result-dir>.run` |
| `--journal PATH` / `--no-journal` | `<run-dir>/journal.jsonl` |
| `--no-progress` | progress lines on standard error |
| `--checkpoint-every N` | 0 (off) |
| `--checkpoint-seconds S` | 0 (off) |
| `--checkpoint-keep K` | 1 |

Launch it with the layout recorded in `resolved-run.json` and the OpenMP
environment of [parallel-execution.md](parallel-execution.md). The engine
exit status is:

| Exit | Terminal status | Meaning |
|---:|---|---|
| 0 | `succeeded` | The analysis finished and the result was published |
| 2 | `failed_preflight` | Invalid JSON, schema or limit |
| 3 | `failed_preflight` | Resource failure, for example a memory admission limit |
| 4 | `failed_setup` | Integrity, path, HDF5 or I/O failure |
| 5 | `failed_numerical` | The analysis failed numerically |
| 6 | `cancelled` | Stopped by `SIGINT` or `SIGTERM` |
| 70 | `failed_setup` | Internal error |
