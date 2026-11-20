# SPDX-License-Identifier: BSD-2-Clause
"""Source-bound mesh adaptations declared by asset extensions.

Some catalog assets need a support that the source mesh does not tag.  The
asset declares the adaptation in ``extensions`` and pins the exact source-mesh
SHA-256; compilation then derives the support from the imported mesh without
modifying the source file.  An unknown declaration, or a declaration bound to
a different mesh, is rejected.

Registered adaptations
----------------------

``hydraulic_support_adaptation`` = ``homogeneous_hydraulic_exterior_planes_v1``
    Adds ``boundary:homogeneous_hydraulic_reference`` to the homogeneous 3D
    slope mesh ``meshes/3d_homo_slope/adaptive_family_a_l1.msh`` (CASE-015).
    The support is the set of exterior triangles that lie wholly on one of the
    pressure planes of the MATLAB reference driver
    ``sysala/slope_stability@8401cc5:slope_stability/slope_stability_3D_homo_seepage_SSR.m``
    (lines 99-116), mapped into the native frame through ``x_M = 10 - x``
    (reuse record REUSE-0010).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import replace
from itertools import combinations

from ..authoring import Asset
from .canonical import CanonicalMesh, EntitySet

EXTENSION_KEY = "hydraulic_support_adaptation"
SUPPORT_NAME = "boundary:homogeneous_hydraulic_reference"
SOURCE_SHA256 = "290811aa1d1ced24bc2851e8b7d34f4d8d29fc27318e623bb63ab69db61ddead"
ADAPTATION = {
    "id": "homogeneous_hydraulic_exterior_planes_v1",
    "source_sha256": SOURCE_SHA256,
    "support_name": SUPPORT_NAME,
    "source_revision": "8401cc52d9594b60a2b45df35f15c509ccdc9884",
    "source_path": "slope_stability/slope_stability_3D_homo_seepage_SSR.m",
    "coordinate_tolerance_m": 0.001,
}
_TOLERANCE_M = 0.001


def select_homogeneous_hydraulic_faces(
    vertices: Mapping[int, tuple[float, float, float]],
    faces: Mapping[int, tuple[int, int, int]],
    cells: Iterable[tuple[int, int, int, int]],
) -> tuple[int, ...]:
    """Select exterior triangles lying wholly on one hydraulic plane.

    Args:
        vertices: Vertex coordinates [m] by stable vertex identity.
        faces: Triangle vertex identities by stable face identity.
        cells: Tetrahedra as vertex-identity tuples.

    Returns:
        Sorted stable identities of the selected faces.  A face is exterior
        when exactly one cell uses it.  Testing each plane separately (rather
        than the union of nodal predicates) excludes side-wall triangles whose
        vertices happen to touch different planes.
    """
    incidence = Counter(
        tuple(sorted(face)) for cell in cells for face in combinations(cell, 3)
    )
    tolerance = _TOLERANCE_M
    selected = []
    for face_id, ids in faces.items():
        if incidence[tuple(sorted(ids))] != 1:
            continue
        points = [vertices[i] for i in ids]
        on_plane = (
            all(abs(x + 30) <= tolerance for x, y, z in points)
            or all(abs(x - 10) <= tolerance for x, y, z in points)
            or all(abs(y - 20) <= tolerance for x, y, z in points)
            or all(
                abs(y - 10) <= tolerance and x <= -15 + tolerance for x, y, z in points
            )
            or all(
                abs(y - x - 25) <= tolerance and -15 - tolerance <= x <= -5 + tolerance
                for x, y, z in points
            )
        )
        if on_plane:
            selected.append(face_id)
    return tuple(sorted(selected))


def derived_support_names(asset: Asset) -> tuple[str, ...]:
    """Return the supports that an asset's declared adaptation derives.

    The importer must not require these supports from the source mesh.

    Raises:
        ValueError: The declaration or its source binding is not registered,
            or the asset lacks the declared derived support.
    """
    declared = asset.extensions.get(EXTENSION_KEY)
    if declared is None:
        return ()
    if declared != ADAPTATION or asset.mesh_source.sha256 != SOURCE_SHA256:
        raise ValueError("unrecognized hydraulic support adaptation or source")
    if asset.dimension != 3 or asset.coordinate_order != 1:
        raise ValueError("homogeneous hydraulic adaptation requires 3D G1 geometry")
    support = next((s for s in asset.supports if s.name == SUPPORT_NAME), None)
    if support is None or support.kind != "boundary" or support.dimension != 2:
        raise ValueError("hydraulic adaptation lacks its declared boundary support")
    return (SUPPORT_NAME,)


def adapt_mesh(asset: Asset, mesh: CanonicalMesh) -> CanonicalMesh:
    """Apply the asset's declared adaptation to an imported canonical mesh.

    Imported supports, geometry and provenance are preserved; only the derived
    support is added.  Meshes of assets without a declaration are returned
    unchanged.

    Raises:
        ValueError: The mesh is not the declared source, the derived support
            already exists, or no face is selected.
    """
    if not derived_support_names(asset):
        return mesh
    if mesh.provenance.source_sha256 != SOURCE_SHA256:
        raise ValueError("hydraulic adaptation imported-source mismatch")
    if mesh.dimension != 3 or mesh.coordinate_order != 1:
        raise ValueError("hydraulic adaptation imported-geometry mismatch")
    vertices = {
        int(i): tuple(map(float, xyz))
        for i, xyz in zip(
            mesh.topology_vertex_ids, mesh.topology_vertex_coordinates, strict=True
        )
    }
    faces = {
        int(i): tuple(map(int, face))
        for i, face in zip(
            mesh.stable_entity_ids[2], mesh.stable_entity_vertex_ids[2], strict=True
        )
    }
    cells = [
        tuple(int(mesh.topology_vertex_ids[int(i)]) for i in cell)
        for cell in mesh.topology_cells
    ]
    selected = select_homogeneous_hydraulic_faces(vertices, faces, cells)
    if not selected:
        raise ValueError("hydraulic adaptation selected no exterior faces")
    supports = dict(mesh.supports)
    if SUPPORT_NAME in supports:
        raise ValueError("hydraulic derived support already exists")
    supports[SUPPORT_NAME] = EntitySet(SUPPORT_NAME, "boundary", 2, selected)
    return replace(mesh, supports=supports)


__all__ = [
    "ADAPTATION",
    "EXTENSION_KEY",
    "SOURCE_SHA256",
    "SUPPORT_NAME",
    "adapt_mesh",
    "derived_support_names",
    "select_homogeneous_hydraulic_faces",
]
