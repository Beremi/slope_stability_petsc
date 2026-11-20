# Python package

The `slope-stability` distribution prepares, launches and inspects runs of
the native `slope-engine`. It is serial, contains no PETSc, MPI or C
extension, and is never imported on the MPI ranks. It requires CPython 3.12
or newer. Install it into a virtual environment outside the source tree
(`$WORK` is any work directory):

```sh
python3.12 -m venv $WORK/venv
$WORK/venv/bin/python -m pip install ./python
```

This installs the `slope-stability` command
([docs/running.md](../docs/running.md)) and two packages:

| Module | Purpose |
|---|---|
| `slope_stability.authoring` | Strict asset, case, profile and suite documents; units; semantic validation |
| `slope_stability.compile` | Gmsh import, canonical stable identities, geometry audit, content-addressed problem packages |
| `slope_stability.catalog` | The in-repository case catalog, mesh verification and catalog compilation |
| `slope_stability.resolution` | Profile composition and run resolution against an installed build |
| `slope_stability.launch` | MPI × OpenMP layouts, per-rank CPU binding, OpenMP environment, bounded execution |
| `slope_stability.results` | Verified result reading and unit-aware comparators |
| `slope_stability.restart` | Comparison of runs restarted from checkpoints |
| `slope_stability.visualization` | VTU/XDMF export of high-order results |
| `slope_stability.cli` | The `slope-stability` command |
| `slope_protocol` | Strict JSON, canonicalization and content identities |

[docs/input-format.md](../docs/input-format.md) shows how to compile a new
case with the API, and [docs/outputs.md](../docs/outputs.md) how to read
results.
