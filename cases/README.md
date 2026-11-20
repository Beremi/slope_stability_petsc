# Case catalog

`catalog/` holds the strict authoring inputs of the 23 reference cases
(CASE-001 to CASE-023) in 24 variants (CASE-005 has P2 and P4 variants):

- `catalog/assets/`: geometry assets; each names its source mesh by SHA-256
  (`mesh_source.path` is the content-addressed path inside a compiled
  package, not a source path);
- `catalog/cases/`: case documents (materials, physics, boundary rules,
  analysis, profile selection and required outputs);
- `catalog/profiles/`: solver profiles (continuation, nonlinear, linear, PMG,
  deflation, seepage, output and validation policies);
- `catalog/index.json`: one record per variant with its case and asset
  documents and their content identities, the field order, and the source
  mesh relative to `mesh_root` (`../../meshes`) with its SHA-256. The
  `catalog_id` is the content identity of the rest of the index.

The documents keep the bytes, and therefore the content identities, with
which they were generated from the earlier implementations; their
`created_by`, `provenance` and `extensions` fields record that lineage.

```sh
slope-stability verify-meshes                      # check meshes/ bindings
slope-stability compile --destination work/compiled --case-id CASE-003
slope-stability inspect cases/catalog              # list the variants
```

Compiling a variant from `meshes/` reproduces the compiled problem identity
recorded when the catalog was admitted; `CAT-COMPILE-001` checks this for all
2D variants. The catalog cases are reference inputs; scientific validation
of their results is tracked outside this repository.
