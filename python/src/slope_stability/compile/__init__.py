# SPDX-License-Identifier: BSD-2-Clause
"""Mesh import, canonicalization and problem-package compilation.

The compilation pipeline is serial and PETSc-free:

1. :mod:`.mesh_import` reads Gmsh MSH 4.x or frozen HDF5 reference meshes
   into :class:`RawMesh` objects (optionally refined by :mod:`.refine`);
2. :mod:`.canonical` assigns stable entity identities and canonical
   simplex orientation;
3. :mod:`.geometry` audits signed Jacobians of the curved geometry;
4. :mod:`.model` binds validated authoring documents to the mesh in SI units;
5. :mod:`.package` writes, verifies and atomically publishes the
   content-addressed ``problem.json``/``problem.h5`` package.
"""

from __future__ import annotations

from ._common import CompileError
from .canonical import (
    CanonicalMesh,
    EntitySet,
    ImportMaps,
    MeshProvenance,
    canonicalize_mesh,
    compact_simplex_mesh,
)
from .geometry import GeometryAudit, audit_geometry
from .mesh_import import GmshImporter, Hdf5ReferenceImporter
from .model import CompiledProblemModel, compile_problem_model
from .package import CompiledPackage, compile_problem
from .raw import ImportRequest, RawMesh, RawSupport
from .refine import uniformly_refine_raw_mesh

__all__ = [
    "CanonicalMesh",
    "CompileError",
    "CompiledPackage",
    "CompiledProblemModel",
    "EntitySet",
    "GeometryAudit",
    "GmshImporter",
    "Hdf5ReferenceImporter",
    "ImportMaps",
    "ImportRequest",
    "MeshProvenance",
    "RawMesh",
    "RawSupport",
    "audit_geometry",
    "canonicalize_mesh",
    "compact_simplex_mesh",
    "compile_problem",
    "compile_problem_model",
    "uniformly_refine_raw_mesh",
]
