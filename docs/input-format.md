# Input format

A problem is described by strict JSON documents: an **asset** (geometry and
its named supports), a **case** (materials, physics, boundary conditions,
analysis and outputs) and the solver **profiles** the case selects. The
catalog in `cases/catalog/` contains the assets, cases and profiles of the 23
reference cases; they are the best starting point for a new problem. The
JSON Schemas generated from the protocol are in `schemas/generated/json/`.

## Common envelope

Every document has the same top-level envelope:

```json
{
  "kind": "slope_stability.case",
  "schema_version": {"major": 1, "minor": 0, "patch": 0},
  "created_by": {"tool": "...", "tool_version": "1", "source_id": "sha256:..."},
  "provenance": {"origin": "...", "license": "BSD-2-Clause"},
  "extensions": {},
  "identity": {"name": "case.my-slope", "title": "My slope"},
  "id": "sha256:..."
}
```

- `kind` is `slope_stability.asset`, `slope_stability.case`,
  `slope_stability.profile` or `slope_stability.suite`.
- `id` is the SHA-256 of the canonical JSON of the document without its `id`
  field. Loading fails when the recorded `id` does not match the content, so
  recompute it after every edit.
- Parsing is strict: unknown fields, duplicate keys, non-finite numbers,
  negative zero and type coercions (for example `"1"` for `1`) are rejected.
- `extensions` holds tool-specific data that does not change the meaning of
  the document unless a registered extension says so (see
  [Assets](#assets)).

This script recomputes the identity of a document in place:

```python
# set_id.py
import json
import sys

from slope_protocol import content_id

path = sys.argv[1]
with open(path, encoding="utf-8") as stream:
    document = json.load(stream)
document.pop("id", None)
document["id"] = content_id(document)
with open(path, "w", encoding="utf-8") as stream:
    json.dump(document, stream, indent=2)
    stream.write("\n")
print(document["id"])
```

## Units

Physical values are written as quantities with an explicit unit, for example
`{"value": 6.0, "unit": "kPa"}`. Any unit that Pint can convert to the
expected dimension is accepted. Compilation converts every value to the
internal unit system and records each conversion in `compile-report.json`.

| Dimension | Internal unit |
|---|---|
| Length | m |
| Force | kN |
| Stress, pore pressure | kPa |
| Unit weight | kN/m³ |
| Hydraulic conductivity | m/s |
| Angle | rad |
| Time | s |

The vertical coordinate is always `y`, in 2D and in 3D. Gravity acts in the
negative `y` direction.

## Assets

An asset names a source mesh and declares the supports that cases may use.

| Field | Meaning |
|---|---|
| `mesh_source` | `importer` (`gmsh.msh4`), `sha256` of the mesh file, and `path` = `meshes/<sha256>.msh`, the location inside a compiled package |
| `dimension` | 2 or 3 |
| `cell_type` | `triangle` in 2D, `tetrahedron` in 3D |
| `coordinate_order` | Geometry order 1 (straight), 2 or 4 (curved) |
| `variants` | Named variants, for example a refinement level |
| `supports` | Named supports with `kind` (`region`, `boundary`, `node_set`) and topological `dimension` |

**Meshes.** The importer reads Gmsh MSH 4.x files. The domain cells must be
Lagrange simplices of the asset's coordinate order: 3-, 6- or 15-node
triangles and 4-, 10- or 35-node tetrahedra. Each support is a Gmsh physical
group whose name is the support name:

| Prefix | Kind | Entities |
|---|---|---|
| `region:` | `region` | Domain cells; carries the material assignment |
| `boundary:` | `boundary` | Boundary facets (edges in 2D, triangles in 3D); essential or natural conditions |
| `nodeset:` | `node_set` | Nodes; essential conditions only |

The mesh file must hash to `mesh_source.sha256`, and compilation never
modifies it. A registered extension can derive a support that the mesh does
not tag: the catalog asset of `meshes/3d_homo_slope/adaptive_family_a_l1.msh`
declares `hydraulic_support_adaptation`, which adds
`boundary:homogeneous_hydraulic_reference` to that exact mesh
([meshes/README.md](../meshes/README.md)).

## Cases

| Field | Meaning |
|---|---|
| `asset` | `name` (the asset's `identity.name`) and `variant` |
| `discretization` | `field_order` (1, 2, 4; 8 is experimental), `coordinate_order` (1, 2, 4), optional `volume_quadrature` (`default` or `paired_higher`) |
| `materials` | One entry per material and region; see below |
| `physics` | `mechanics` and `seepage` (booleans) and `coupling` |
| `prescribed_saturation` | Optional water level for uncoupled mechanics; see below |
| `boundary_rules` | Essential and natural boundary conditions |
| `analysis` | `selector`, one of the analyses below |
| `profiles` | The profile name selected for each policy family |
| `case_overrides` | Case-specific data; `resolution_overrides` maps policy paths to values |
| `required_outputs` | Quantities that the run must publish |

`volume_quadrature: "paired_higher"` selects the next registered volume rule
above the default one for every operator, output and coupling quadrature.

### Materials

Each entry has a `name`, a region `support`, a registered `model` and its
`parameters`.

`mohr_coulomb` (mechanics):

| Parameter | Kind |
|---|---|
| `cohesion` | Stress |
| `friction_angle`, `dilation_angle` | Angle |
| `young_modulus` | Stress |
| `poisson_ratio` | Number or dimensionless quantity |
| `saturated_unit_weight` and `unsaturated_unit_weight`, or `unit_weight` | Unit weight |
| `davis` | `"A"`, `"B"` or `"C"` |

`steady_unconfined_darcy` (seepage): `conductivity` (hydraulic conductivity)
and `water_unit_weight` (unit weight).

### Physics and coupling

| `coupling` | Meaning |
|---|---|
| `none` | Mechanics and seepage are independent (or only one is enabled) |
| `pressure_gradient_and_weight` | The seepage solution enters mechanics through the pore-pressure gradient and the saturation-dependent unit weight |
| `weight_only` | Only the saturation-dependent unit weight enters mechanics |

Coupling is one way: the seepage problem is solved first and its solution is
frozen for the whole mechanics analysis.

### Analyses

| `selector` | Analysis |
|---|---|
| `limit_load.work_controlled` | Limit load: the load factor is increased by work-controlled continuation |
| `ssr.direct_potential` | Strength reduction with direct continuation in the reduction factor |
| `ssr.indirect_work_controlled` | Strength reduction with indirect (work-controlled) continuation |
| `seepage.steady_unconfined` | Steady unconfined seepage only |

### Boundary rules

Each rule has a `name`, a `field` (`displacement` or `pore_pressure`), a
`condition` (`essential` or `natural`), a `support`, the constrained
`components`, a `value_model` with its `parameters`, and for natural
conditions a `geometry_patch`.

| Field and condition | Value models | Components |
|---|---|---|
| Displacement, essential | `constant_vector`, `affine_coordinate` | Subset of `x`, `y`, `z`, or `all` |
| Displacement, natural | `global_traction`, `spatial_global_traction` | `all` |
| Displacement, natural | `normal_pressure`, `hydrostatic` | `normal` |
| Pore pressure, essential | `constant_scalar`, `affine_coordinate`, `hydrostatic`, `piecewise_linear_level` | `scalar` or `pressure` |
| Pore pressure, natural | `prescribed_flux` | `normal` |

`hydrostatic` takes a `reference_level`, a `reference_pressure` and a
`unit_weight`; below the reference level the value grows linearly with depth.
`piecewise_linear_level` takes a water level at strictly increasing `x`
coordinates and prescribes the hydrostatic pressure below that level, zero
above it. Every value model states its units explicitly.

### Outputs

`required_outputs` and the output profile select quantities from:
`displacement`, `strain`, `stress`, `material_branch`, `plastic_indicator`,
`yield_residual`, `pore_pressure`, `pressure_gradient`, `conductivity`,
`hydraulic_ramp`, `hydraulic_epsilon`, `saturation`, `pressure_flux`,
`darcy_velocity`, `coupled_body_force`, `global_flux_balance`,
`continuation`, `nonlinear_history`, `safety_factor`, `load_factor`,
`work_control` and `maximum_displacement`. [outputs.md](outputs.md) describes
where they are stored.

## Prescribed saturation

An uncoupled mechanics case (`seepage: false`, `coupling: "none"`) may
declare a phreatic surface that switches each point between the saturated
and the unsaturated unit weight of its material:

```json
"prescribed_saturation": {
  "axis": "x",
  "coordinates": [0.0, 40.0, 75.0],
  "levels": [12.0, 10.5, 8.0],
  "length_unit": "m"
}
```

- `axis` is the horizontal coordinate of the profile: `x`, or `z` for a 3D
  asset.
- `coordinates` are 1 to 32 finite, strictly increasing knots and `levels`
  the same number of finite water levels. One knot gives a constant level.
  Between knots the level is interpolated linearly; outside the knot interval
  it is extended as a constant.
- A point is saturated exactly when its `y` coordinate is at or below the
  level.

The engine evaluates the profile at every owned quadrature point, on curved
geometry through the physical coordinates of the element map. The correction
from the unsaturated to the saturated weight is added to the region's
gravity load, consistently in the residual, the potential, the load vector and
the matrix-free operator. In a limit-load analysis both the self-weight and
the correction are multiplied by the load factor; in SSR the load factor is
one. The profile is part of the compiled problem and of its identity; a case
without it keeps the identity it had before the field existed. It cannot be
combined with a seepage solve or with coupling.

## Profiles

A profile is one named policy for one concern:

```json
{
  "kind": "slope_stability.profile",
  "family": "pmg",
  "identity": {"name": "pmg.migrated.pmg-deflated-baseline", "title": "..."},
  "policy": {"enabled": true, "smoother": "chebyshev", "...": "..."},
  "...": "envelope fields"
}
```

| Family | Controls |
|---|---|
| `continuation` | Initial factor and steps, work budget, stopping rules, number of accepted states |
| `nonlinear` | Newton tolerances, stopping criterion, regularization, iteration limit |
| `linear` | Krylov method, operator class, stopping condition, tolerances, restart, recycling |
| `pmg` | Polynomial levels, smoother and steps, coarse solver, active ranks on the coarse level |
| `deflation` | Basis source and size, projector, admission tolerances, compression |
| `seepage` | Linear and nonlinear limits of the seepage solve (seepage cases only) |
| `output` | Published quantities, histories and capture frequency |
| `validation` | Strictness of option checking and conservation tolerances |

A case selects one profile per family by name. The profile is read from
`<profiles-dir>/<name>.json`, and its `family` and `identity.name` must match
the selection. Profiles cannot inherit from each other. The policy fields and
their resolution are described in [solver-options.md](solver-options.md).

`slope_stability.suite` documents combine cases, profiles, sweep axes and
resource layouts for measurement studies; the `slope-stability` command does
not use them.

## Compiled package and resolved run

`slope-stability compile` binds a case to its asset and mesh and writes a
problem package:

- `problem.json`: the compiled problem with all values in internal units:
  spaces, materials, supports, boundary data, outputs and payload hashes;
- `problem.h5`: the canonical mesh, stable entity identities, coordinates
  and support memberships;
- `compile-report.json`: unit conversions, counts and the geometry audit
  (signed Jacobians of every cell).

`slope-stability resolve` (also run inside `slope-stability run`) combines
the compiled problem, the case's profiles and overrides, the layout and the
installed build into `resolved-run.json`. Its `resolution_trace` records where
each resolved value came from. Neither file is meant to be edited by hand.

## A new case

Copy a catalog case and asset as a template, edit them and recompute their
identities. This example changes the cohesion of `CASE-003`; the commands run
in a scratch directory, with `S` pointing to the source tree:

```sh
cp $S/cases/catalog/cases/CASE-003-canonical.json my-case.json
cp $S/cases/catalog/assets/asset.2d_homo_slope.h1.0.g1.json my-asset.json
cp $S/meshes/2d_homo_slope/h1.0.msh my-mesh.msh
cp -r $S/cases/catalog/profiles profiles
# edit my-case.json: identity.name, materials[0].parameters.cohesion.value, ...
python set_id.py my-case.json
```

For a new mesh, set `mesh_source.sha256` and `mesh_source.path` in the asset
to the file's SHA-256 (`sha256sum my-mesh.msh`), list its supports, and
recompute the asset identity as well. Compile with the Python API:

```python
# compile_case.py
from pathlib import Path

from slope_stability.authoring import load_document, validate_documents
from slope_stability.catalog import load_case_profiles
from slope_stability.compile import GmshImporter, ImportRequest, compile_problem

asset = load_document(Path("my-asset.json"))
case = load_document(Path("my-case.json"))
profiles = load_case_profiles(case, Path("profiles"))
validate_documents(asset=asset, case=case, profiles=profiles).require_valid()
mesh = GmshImporter().import_mesh(
    Path("my-mesh.msh"),
    ImportRequest(
        dimension=asset.dimension,
        coordinate_order=asset.coordinate_order,
        required_supports=tuple(support.name for support in asset.supports),
    ),
)
package = compile_problem(asset=asset, case=case, mesh=mesh, destination=Path("compiled"))
print(package.problem_id, package.path)
```

Then resolve and run it as described in
[running.md](running.md#resolve-and-direct-runs):

```sh
python compile_case.py
slope-stability resolve --case my-case.json --profiles-dir profiles \
    --problem-package compiled/sha256-<problem id> --prefix $PREFIX \
    --layout 2x2 --destination resolved
slope-stability run --prefix $PREFIX --runtime-library-path $PETSC_LIB \
    --package compiled/sha256-<problem id> \
    --resolved-run resolved/sha256-<run id>/resolved-run.json \
    --destination run-direct
```

`validate_documents` reports every semantic problem at once, with a JSON
pointer into the offending document and a suggested remedy.
