# Outputs

A run publishes a **result directory** and keeps a **run directory** next to
it. Both are written by the engine; the catalog runner of `slope-stability
run` places them under `results/` of its destination
([running.md](running.md#run)).

## Result directory

| File | Content |
|---|---|
| `result.json` | Terminal envelope: identities, terminal status, analysis outcome, summary, artifact hashes |
| `result.h5` | Fields, quadrature data, accepted states and histories (parallel HDF5), written at publication |
| `environment.json` | Observed MPI size and OpenMP threads of the run |
| `checkpoint-accepted/` | Restart checkpoint (`checkpoint.json`, `checkpoint.h5`), when requested or after cancellation |
| `error.json`, `events.json` | Error record and diagnostic events, present when the run failed or was cancelled before publishing data |

The engine writes into staging files and publishes the directory atomically,
so a reader never sees a partially written result. `result.json` records the
SHA-256 and byte count of every artifact and a logical digest of the HDF5 data
that does not depend on the file layout. `slope_stability.results.open_result`
and `slope-stability inspect` verify all of them before returning data. A run
that ended before any data was published has only an error envelope;
`inspect` then reports that the result is not complete, and `error.json`
names the failure.

### Terminal status and analysis outcome

`terminal_status.status` says whether the run produced a valid result:

| Status | Meaning |
|---|---|
| `succeeded` | The analysis ran to one of its stopping rules and the result is complete |
| `cancelled` | The run was stopped by `SIGINT` or `SIGTERM` |
| `failed_preflight` | The inputs or resources were rejected before setup |
| `failed_setup` | Setup or I/O failed |
| `failed_numerical` | The analysis failed numerically |

`analysis_outcome.outcome` says how the analysis ended:

| Outcome | Meaning |
|---|---|
| `converged_criterion` | The continuation met its stopping criterion: the scaled-slope tolerance, or the minimum step after a successful final state |
| `completed_control_horizon` | The work horizon (or, for direct continuation, the control horizon) was reached |
| `completed_requested_step_horizon` | The requested number of accepted states was reached |
| `accepted_relaxed` | Finished, but at least one accepted step used relaxed nonlinear acceptance |
| `failed_retry_limit` | The step could not be reduced further, or too many consecutive attempts failed |
| `failed_iteration_limit`, `failed_line_search`, `failed_regularization`, `failed_linear`, `failed_singular_constraint`, `failed_nonfinite` | The nonlinear or linear solve failed in the named way |
| `cancelled` | The analysis was cancelled |

A run with `--maximum-accepted-states 4`, for example, succeeds with
`completed_requested_step_horizon`, which is not a converged factor of safety.
`terminal_status.stop_reason` gives the precise reason, for example
`minimum_step_reached`. The `summary` section lists the quantities, the final
scalar values (`load_factor`, `external_work`, `displacement_maximum`,
reactions and others, each with its unit), the accepted-state count and timing
events.

### `result.h5`

All arrays are ordered by stable entity identity, so their layout does not
depend on the number of ranks. Each quantity group stores `stable_ids`,
`state_ids` and `values`:

| Group | Shape of `values` | Examples |
|---|---|---|
| `fields/<quantity>` | state × node × component | `displacement`, `pore_pressure` |
| `quadrature/<quantity>` | state × cell × point × component | `stress`, `strain`, `yield_residual` |
| `quadrature_categories/<quantity>` | state × cell × point | `material_branch`, `plastic_indicator` |
| `cell_reductions/<quantity>` | state × cell × … | Per-cell reductions, when requested |
| `states/index/rows` | one row per accepted state | Factor, work, control, maximum displacement, history range |
| `history/<table>/rows` | one row per record | See below |

The output profile selects the quantities and whether they are stored at
every accepted state (`frequency: accepted_state`) or only at the end
(`terminal`).

### Histories

The `history` tables record every solve of the run, including rejected
attempts, so a run can be analyzed without the progress journal. All tables
share one row type (`row_id`, `state_id`, `attempt_id`, `parent_row_id`,
`ordinal`, `control`, `residual`, `iterations`, `tolerance`, `status`,
`reason`, `accepted` and related columns):

| Table | One row per |
|---|---|
| `accepted` | Accepted continuation state |
| `attempt` | Continuation attempt, accepted or rolled back |
| `newton` | Newton iteration |
| `line_search` | Line-search evaluation |
| `linear` | Linear solve |
| `pmg` | PMG setup or refresh |
| `deflation` | Deflation basis update |
| `seepage` | Seepage iteration |
| `io` | Result or checkpoint I/O record |

### Reading results from Python

```python
import sys

from slope_stability.results import open_result

result = open_result(sys.argv[1])
try:
    print(result.status, result.analysis.outcome)
    factor = result.scalar("load_factor")
    print(f"final load factor: {factor.value:.6f}")
    displacement = result.field("displacement", state="terminal")
    print(displacement.unit, displacement.values().shape)
    accepted = result.history("accepted").rows()
    print(len(accepted), "accepted states")
finally:
    result.close()
```

`field`, `quadrature` and `category` accept a state identifier or
`"terminal"`, and their `read` methods select entities and components without
loading the whole array. `slope_stability.results` also provides comparators
that align two results by stable identity and compare fields, scalars and
trajectories with explicit tolerances and unit conversion.
`slope_stability.visualization` exports high-order results to VTU and XDMF.

## Run directory

The run directory is `<result-dir>.run` unless `--run-directory` is given. It
holds monitoring data that is not part of the result:

| Path | Content |
|---|---|
| `journal.jsonl` | Live progress journal ([running.md](running.md#progress-and-the-journal)) |
| `history.spool` | History rows streamed by rank 0 during the run and replayed into `result.h5` at publication |
| `accepted-states.h5` | Accepted states (parallel HDF5), written when the output profile stores every accepted state, and copied into `result.h5` at publication |
| `checkpoints/state-NNNNNN/` | In-run checkpoints of accepted states |

## Checkpoints and restart

A checkpoint stores one accepted state: the current and previous
displacement, the continuation controller with its recent states, the
deflation basis, the coupling data and the history position, together with
compatibility identities of the problem, mesh, field space, numerical policy
and build. Three mechanisms write checkpoints:

- `--checkpoint` on `slope-stability run` publishes the final accepted state
  as `checkpoint-accepted/` in the result directory.
- `--checkpoint-every N` and `--checkpoint-seconds S` write in-run
  checkpoints to `<run-dir>/checkpoints/` at accepted states. Only the newest
  `--checkpoint-keep K` (default 1) are kept, so disk use does not grow with
  the run length.
- After `SIGINT` or `SIGTERM`, a cancelled result includes
  `checkpoint-accepted/` with the last accepted state, if there is one. A run
  cancelled before its first accepted state publishes only `result.json`,
  `error.json` and `events.json`.

Restart from any of them with `--restart-from`:

```sh
slope-stability run --prefix $PREFIX --runtime-library-path $PETSC_LIB \
    --compiled $WORK/compiled --case-id CASE-003 --layout 2x2 \
    --maximum-accepted-states 6 --checkpoint --destination $WORK/run-part1
slope-stability run --prefix $PREFIX --runtime-library-path $PETSC_LIB \
    --compiled $WORK/compiled --case-id CASE-003 --layout 4x8 \
    --restart-from $WORK/run-part1/results/case-003-canonical/checkpoint-accepted \
    --destination $WORK/run-part2
```

The restarted run must use the same problem and the same build, and the
engine checks the compatibility identities before it continues. The layout may
differ: the checkpoint is redistributed by stable identity. The continuation
horizon may also change, as in the example, where the second run continues to
convergence. The restarted result names its parent run and checkpoint under
`restart`; `slope-stability inspect` shows them.
