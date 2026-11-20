# SPDX-License-Identifier: BSD-2-Clause
"""The in-repository case catalog, its meshes and catalog compilation.

The catalog (``cases/catalog``) indexes case variants.  Each record names a
case document, its asset document and the source mesh under the mesh root
(``meshes/`` by default) together with the mesh SHA-256.  Every mesh is also
listed in ``meshes/SHA256SUMS``; :func:`verify_meshes` checks both bindings
and the asset's ``mesh_source.sha256`` before anything is compiled.

:func:`compile_catalog` compiles selected variants into a destination
directory holding content-addressed ``packages/`` and a
``compile-catalog.json`` report.  The destination is published atomically.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping

from slope_protocol import content_id

from ._io import dumps_json, file_sha256, fsync_path, read_json_object
from .authoring import Asset, Case, Profile, load_document, validate_documents
from .compile import CanonicalMesh, GmshImporter, ImportRequest, compile_problem
from .compile.adaptations import adapt_mesh, derived_support_names

CATALOG_KIND = "slope_stability.case_catalog"
COMPILATION_KIND = "slope_stability.case_catalog_compilation"
COMPILATION_REPORT = "compile-catalog.json"
SHA256SUMS = "SHA256SUMS"
_ENVIRONMENT_ROOT = "SLOPE_STABILITY_ROOT"
_RECORD_FIELDS = {
    "case_id",
    "variant_id",
    "field_order",
    "case_document",
    "case_document_id",
    "asset_document",
    "asset_document_id",
    "mesh",
    "mesh_sha256",
}


class CatalogError(ValueError):
    """The catalog, a mesh or a compilation request is invalid."""


def find_repository_root(start: str | os.PathLike[str] | None = None) -> Path:
    """Locate the source tree that holds ``cases/catalog`` and ``meshes``.

    The search order is ``$SLOPE_STABILITY_ROOT``, the parents of ``start``
    (default: the current directory) and the parents of this package.

    Raises:
        CatalogError: No candidate contains ``cases/catalog/index.json``.
    """
    candidates: list[Path] = []
    configured = os.environ.get(_ENVIRONMENT_ROOT)
    if configured:
        candidates.append(Path(configured))
    origin = Path.cwd() if start is None else Path(start)
    candidates.extend((origin.resolve(), *origin.resolve().parents))
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / "cases" / "catalog" / "index.json").is_file():
            return candidate
    raise CatalogError(
        f"cannot locate the catalog; pass --catalog/--meshes or set {_ENVIRONMENT_ROOT}"
    )


@dataclass(frozen=True)
class CatalogVariant:
    """One catalog record: a case variant, its documents and its mesh.

    Document paths are relative to the catalog directory; ``mesh`` is
    relative to the catalog's mesh root.
    """

    case_id: str
    variant_id: str
    field_order: int
    case_document: str
    case_document_id: str
    asset_document: str
    asset_document_id: str
    mesh: str
    mesh_sha256: str

    @property
    def key(self) -> str:
        """``CASE-NNN/variant`` label of the record."""
        return f"{self.case_id}/{self.variant_id}"

    @property
    def slug(self) -> str:
        """File-name-safe lowercase label, for example ``case-003-canonical``."""
        return f"{self.case_id}-{self.variant_id}".lower().replace("_", "-")


@dataclass(frozen=True)
class Catalog:
    """A loaded, identity-checked catalog index."""

    directory: Path
    catalog_id: str
    mesh_root: Path
    variants: tuple[CatalogVariant, ...]

    @property
    def profiles_directory(self) -> Path:
        """Directory of the profile documents referenced by the cases."""
        return self.directory / "profiles"

    def select(
        self,
        case_ids: Iterable[str] | None = None,
        variant_ids: Iterable[str] | None = None,
    ) -> tuple[CatalogVariant, ...]:
        """Return the records matching the given case and variant identifiers.

        ``None`` selects every value.  Unknown identifiers are rejected.
        """
        cases = None if case_ids is None else set(case_ids)
        variants = None if variant_ids is None else set(variant_ids)
        known_cases = {record.case_id for record in self.variants}
        known_variants = {record.variant_id for record in self.variants}
        if cases is not None and not cases <= known_cases:
            raise CatalogError(f"unknown case ids: {sorted(cases - known_cases)}")
        if variants is not None and not variants <= known_variants:
            raise CatalogError(
                f"unknown variant ids: {sorted(variants - known_variants)}"
            )
        selected = tuple(
            record
            for record in self.variants
            if (cases is None or record.case_id in cases)
            and (variants is None or record.variant_id in variants)
        )
        if not selected:
            raise CatalogError("the case selection is empty")
        return selected

    def variant(self, case_id: str, variant_id: str = "canonical") -> CatalogVariant:
        """Return exactly one record."""
        for record in self.variants:
            if record.case_id == case_id and record.variant_id == variant_id:
                return record
        raise CatalogError(f"unknown catalog variant {case_id}/{variant_id}")

    def mesh_path(self, record: CatalogVariant) -> Path:
        """Absolute path of a record's source mesh."""
        return self.mesh_root / record.mesh

    def load_case(self, record: CatalogVariant) -> Case:
        """Load a record's case document and check its identity."""
        case = load_document(self.directory / record.case_document)
        if not isinstance(case, Case) or case.id != record.case_document_id:
            raise CatalogError(f"{record.key}: case document identity mismatch")
        return case

    def load_asset(self, record: CatalogVariant) -> Asset:
        """Load a record's asset document and check its identity and mesh."""
        asset = load_document(self.directory / record.asset_document)
        if not isinstance(asset, Asset) or asset.id != record.asset_document_id:
            raise CatalogError(f"{record.key}: asset document identity mismatch")
        if asset.mesh_source.sha256 != record.mesh_sha256:
            raise CatalogError(f"{record.key}: asset mesh digest mismatch")
        return asset


def _record(value: Any, index: int) -> CatalogVariant:
    if not isinstance(value, dict) or set(value) != _RECORD_FIELDS:
        raise CatalogError(f"catalog record {index} has invalid fields")
    mesh = value["mesh"]
    if (
        not isinstance(mesh, str)
        or Path(mesh).is_absolute()
        or ".." in Path(mesh).parts
        or not mesh.endswith(".msh")
    ):
        raise CatalogError(f"catalog record {index} has an invalid mesh path")
    return CatalogVariant(**value)


def load_catalog(
    directory: str | os.PathLike[str] | None = None,
    *,
    mesh_root: str | os.PathLike[str] | None = None,
) -> Catalog:
    """Load and identity-check ``index.json`` of a catalog directory.

    Args:
        directory: Catalog directory; defaults to ``cases/catalog`` of the
            repository found by :func:`find_repository_root`.
        mesh_root: Overrides the index's ``mesh_root`` (relative to the
            catalog directory).

    Raises:
        CatalogError: The index is malformed or its ``catalog_id`` does not
            match its content.
    """
    root = (
        find_repository_root() / "cases" / "catalog"
        if directory is None
        else Path(directory)
    ).resolve()
    index = read_json_object(root / "index.json", error=CatalogError)
    if index.get("kind") != CATALOG_KIND or set(index) != {
        "kind",
        "schema_version",
        "catalog_id",
        "mesh_root",
        "cases",
    }:
        raise CatalogError("index.json is not a case catalog")
    identity_input = dict(index)
    catalog_id = identity_input.pop("catalog_id")
    if catalog_id != content_id(identity_input):
        raise CatalogError("catalog_id does not match the index content")
    cases = index["cases"]
    if not isinstance(cases, list) or not cases:
        raise CatalogError("catalog has no cases")
    records = tuple(_record(value, index) for index, value in enumerate(cases))
    if len({(r.case_id, r.variant_id) for r in records}) != len(records):
        raise CatalogError("catalog records are not unique")
    meshes = (
        (root / str(index["mesh_root"])).resolve()
        if mesh_root is None
        else Path(mesh_root).resolve()
    )
    return Catalog(
        directory=root, catalog_id=catalog_id, mesh_root=meshes, variants=records
    )


def read_sha256sums(mesh_root: str | os.PathLike[str]) -> dict[str, str]:
    """Parse ``SHA256SUMS`` (``sha256sum`` text format) of a mesh root."""
    path = Path(mesh_root) / SHA256SUMS
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise CatalogError(f"cannot read {path}: {error}") from error
    digests: dict[str, str] = {}
    for number, line in enumerate(lines, start=1):
        digest, separator, name = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not name
            or name in digests
        ):
            raise CatalogError(f"{path}:{number}: malformed checksum line")
        digests[name] = digest
    return digests


@dataclass(frozen=True)
class MeshCheck:
    """Verification of one mesh file against all of its recorded digests."""

    mesh: str
    sha256: str | None
    byte_count: int | None
    expected: Mapping[str, str]

    @property
    def passed(self) -> bool:
        """Whether the file exists and every recorded digest matches."""
        return self.sha256 is not None and all(
            value == self.sha256 for value in self.expected.values()
        )


def verify_meshes(
    catalog: Catalog,
    *,
    check_assets: bool = True,
) -> tuple[MeshCheck, ...]:
    """Hash every mesh and compare it with all recorded digests.

    Each file listed in ``SHA256SUMS`` is checked against that list, the
    catalog records that use it and (with ``check_assets``) the
    ``mesh_source.sha256`` of their asset documents.  Unlisted ``.msh`` files
    and catalog meshes missing from ``SHA256SUMS`` are reported as failures.
    """
    listed = read_sha256sums(catalog.mesh_root)
    expected: dict[str, dict[str, str]] = {
        name: {SHA256SUMS: digest} for name, digest in listed.items()
    }
    for record in catalog.variants:
        entry = expected.setdefault(record.mesh, {})
        entry[f"catalog:{record.key}"] = record.mesh_sha256
        if check_assets:
            asset = catalog.load_asset(record)
            entry[f"asset:{record.asset_document}"] = asset.mesh_source.sha256
    for path in sorted(catalog.mesh_root.rglob("*.msh")):
        expected.setdefault(path.relative_to(catalog.mesh_root).as_posix(), {})
    checks = []
    for name, digests in sorted(expected.items()):
        path = catalog.mesh_root / name
        if path.is_file():
            sha256, size = file_sha256(path), path.stat().st_size
        else:
            sha256, size = None, None
        if SHA256SUMS not in digests:
            digests = {**digests, SHA256SUMS: "<unlisted>"}
        checks.append(MeshCheck(name, sha256, size, digests))
    return tuple(checks)


def load_case_profiles(
    case: Case, profiles_directory: str | os.PathLike[str]
) -> tuple[Profile, ...]:
    """Load the profile document selected by a case for every concern.

    Raises:
        CatalogError: A profile is missing or does not match its selection.
    """
    selections = case.profiles.model_dump(mode="json", exclude_none=True)
    profiles: list[Profile] = []
    for family, name in sorted(selections.items()):
        path = Path(profiles_directory) / f"{name}.json"
        profile = load_document(path)
        if (
            not isinstance(profile, Profile)
            or profile.family != family
            or profile.identity.name != name
        ):
            raise CatalogError(
                f"{path}: selection {family}={name!r} does not match the document"
            )
        profiles.append(profile)
    return tuple(profiles)


def import_catalog_mesh(
    catalog: Catalog,
    record: CatalogVariant,
    asset: Asset,
    *,
    uniform_refinement_levels: int = 0,
) -> CanonicalMesh:
    """Import, verify and adapt the source mesh of one record."""
    path = catalog.mesh_path(record)
    if not path.is_file():
        raise CatalogError(f"{record.key}: mesh {record.mesh} is missing")
    if file_sha256(path) != record.mesh_sha256:
        raise CatalogError(f"{record.key}: mesh {record.mesh} digest mismatch")
    derived = derived_support_names(asset)
    mesh = GmshImporter().import_mesh(
        path,
        ImportRequest(
            dimension=asset.dimension,
            coordinate_order=asset.coordinate_order,
            required_supports=tuple(
                item.name for item in asset.supports if item.name not in derived
            ),
            uniform_refinement_levels=uniform_refinement_levels,
        ),
    )
    if mesh.provenance.source_sha256 != asset.mesh_source.sha256:
        raise CatalogError(f"{record.key}: imported mesh digest mismatch")
    return adapt_mesh(asset, mesh)


def compile_catalog(
    catalog: Catalog,
    destination: str | os.PathLike[str],
    *,
    case_ids: Iterable[str] | None = None,
    variant_ids: Iterable[str] | None = None,
    uniform_refinement_levels: int = 0,
) -> dict[str, Any]:
    """Compile selected catalog variants into a new destination directory.

    Every selected case is validated together with its asset and profiles,
    its mesh is verified against ``SHA256SUMS`` and the catalog, and each
    distinct mesh is imported once.  The destination must not exist; it is
    published atomically with ``packages/`` and ``compile-catalog.json``.

    Returns:
        The compilation report that was written.
    """
    if type(uniform_refinement_levels) is not int or uniform_refinement_levels < 0:
        raise CatalogError("uniform refinement levels must be a nonnegative integer")
    target = Path(destination)
    if target.exists():
        raise CatalogError(f"destination already exists: {target}")
    records = catalog.select(case_ids, variant_ids)
    listed = read_sha256sums(catalog.mesh_root)
    for record in records:
        if listed.get(record.mesh) != record.mesh_sha256:
            raise CatalogError(f"{record.key}: mesh is not listed in {SHA256SUMS}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        meshes: dict[str, CanonicalMesh] = {}
        rows = []
        for record in records:
            asset = catalog.load_asset(record)
            case = catalog.load_case(record)
            validate_documents(
                asset=asset,
                case=case,
                profiles=load_case_profiles(case, catalog.profiles_directory),
            ).require_valid()
            mesh_key = f"{asset.id}:uniform-l{uniform_refinement_levels}"
            mesh = meshes.get(mesh_key)
            if mesh is None:
                mesh = import_catalog_mesh(
                    catalog,
                    record,
                    asset,
                    uniform_refinement_levels=uniform_refinement_levels,
                )
                meshes[mesh_key] = mesh
            package = compile_problem(
                asset=asset,
                case=case,
                mesh=mesh,
                destination=staging / "packages",
            )
            compile_report = read_json_object(package.compile_report)
            hdf5 = package.path / "problem.h5"
            rows.append(
                {
                    "case_id": record.case_id,
                    "variant_id": record.variant_id,
                    "case_document_id": case.id,
                    "asset_document_id": asset.id,
                    "mesh": record.mesh,
                    "mesh_sha256": record.mesh_sha256,
                    "mesh_logical_digest": compile_report["mesh_logical_digest"],
                    "problem_id": package.problem_id,
                    "package": package.path.relative_to(staging).as_posix(),
                    "hdf5_logical_digest": package.hdf5_logical_digest,
                    "hdf5_sha256": file_sha256(hdf5),
                    "hdf5_byte_count": hdf5.stat().st_size,
                    "counts": compile_report["counts"],
                    "geometry_audit": compile_report["geometry_audit"],
                    "uniform_refinement_levels": uniform_refinement_levels,
                    "status": "pass",
                }
            )
        report = {
            "kind": COMPILATION_KIND,
            "schema_version": {"major": 2, "minor": 0, "patch": 0},
            "catalog_id": catalog.catalog_id,
            "uniform_refinement_levels": uniform_refinement_levels,
            "case_variant_count": len(rows),
            "unique_mesh_count": len(meshes),
            "cases": rows,
            "status": "pass",
        }
        report_path = staging / COMPILATION_REPORT
        report_path.write_text(dumps_json(report), encoding="utf-8")
        fsync_path(report_path)
        fsync_path(staging)
        staging.rename(target)
        fsync_path(target.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def load_compilation(
    directory: str | os.PathLike[str], catalog: Catalog | None = None
) -> dict[str, Any]:
    """Read ``compile-catalog.json`` of a compilation directory.

    With ``catalog``, the report must belong to that catalog.
    """
    report = read_json_object(Path(directory) / COMPILATION_REPORT, error=CatalogError)
    if report.get("kind") != COMPILATION_KIND or report.get("status") != "pass":
        raise CatalogError(f"{directory}: not a passing catalog compilation")
    if catalog is not None and report.get("catalog_id") != catalog.catalog_id:
        raise CatalogError(f"{directory}: compiled from a different catalog")
    return report


__all__ = [
    "Catalog",
    "CatalogError",
    "CatalogVariant",
    "MeshCheck",
    "compile_catalog",
    "find_repository_root",
    "import_catalog_mesh",
    "load_case_profiles",
    "load_catalog",
    "load_compilation",
    "read_sha256sums",
    "verify_meshes",
]
