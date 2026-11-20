# SPDX-License-Identifier: BSD-2-Clause
"""Strict, serial authoring models and semantic validation.

This module deliberately has no PETSc, MPI, or native-library dependency.
Serialized documents are validated through Pydantic's strict JSON boundary so
JSON arrays can become immutable tuples without permitting scalar coercion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib
from typing import Any, Literal, Mapping

import pint
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from slope_protocol import (
    CanonicalizationError,
    canonicalize,
    content_id,
    loads_strict,
    normalize_package_path,
    reject_forbidden_public_keys,
)
from ._solver_settings import (
    MAXIMUM_ACCEPTED_STATES,
    ContinuationSettings,
    LinearSettings,
    NonlinearSettings,
    OutputsSettings,
    PmgSettings,
    ResourcesSettings,
    ValidationSettings,
)


class AuthoringError(ValueError):
    """Stable error raised before compilation or launch."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        phase: str = "authoring",
        issues: tuple["ValidationIssue", ...] = (),
    ) -> None:
        self.code = code
        self.phase = phase
        self.issues = issues
        super().__init__(f"{code}: {detail}")


class UnitError(AuthoringError):
    """Stable unit conversion failure."""


class RegistryError(AuthoringError):
    """Stable registered-model failure."""


def _immutable(*_args: Any, **_kwargs: Any) -> None:
    raise TypeError("authoring mappings are immutable")


class FrozenDict(dict[Any, Any]):
    """A JSON-serializable mapping that rejects every mutation path."""

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __deepcopy__(self, _memo: dict[int, Any]) -> "FrozenDict":
        return self


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, dict):
        return FrozenDict((key, _deep_freeze(item)) for key, item in value.items())
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_deep_freeze(item) for item in value)
    return value


class FrozenModel(BaseModel):
    """Base for immutable strict authoring values."""

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        validate_default=True,
    )

    def model_post_init(self, _context: Any) -> None:
        for name in type(self).model_fields:
            object.__setattr__(self, name, _deep_freeze(getattr(self, name)))


class SourceSpan(FrozenModel):
    """One-based line/column range of a JSON pointer in its source document."""

    path: str
    line: int
    column: int
    end_line: int
    end_column: int


class Version(FrozenModel):
    """Semantic schema version ``major.minor.patch``."""

    major: int = Field(ge=0)
    minor: int = Field(ge=0)
    patch: int = Field(ge=0)


class CreatedBy(FrozenModel):
    """Tool, tool version and source identity that produced a document."""

    tool: str
    tool_version: str
    source_id: str


class Provenance(FrozenModel):
    """Origin description and SPDX licence of a document's content."""

    origin: str
    license: str


class Identity(FrozenModel):
    """Stable short name and human-readable title of a document."""

    name: str = Field(min_length=1)
    title: str = Field(min_length=1)


class Quantity(FrozenModel):
    """Scalar value with an explicit Pint-parsable unit string."""

    value: float
    unit: str

    @model_validator(mode="after")
    def finite_value(self) -> "Quantity":
        if not math.isfinite(self.value):
            raise ValueError("quantity value must be finite")
        return self


class UnitConversion(FrozenModel):
    """Record of one conversion into the internal SI unit system.

    The internal value is ``original_value * scale + offset`` in ``internal_unit``;
    ``dimension`` names the checked physical dimension.
    """

    original_value: float
    original_unit: str
    converted_value: float
    internal_unit: str
    scale: float
    offset: float
    dimension: str


class Document(FrozenModel):
    """Common envelope of every authoring document.

    ``id`` is the ``sha256:`` content identity of the canonical document without its
    ``id`` field; loading rejects documents whose recorded identity does not match
    their content.
    """

    kind: str
    schema_version: Version
    created_by: CreatedBy
    provenance: Provenance
    extensions: dict[str, Any]
    identity: Identity
    id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    _source_path: str = PrivateAttr(default="<memory>")
    _spans: Mapping[str, SourceSpan] = PrivateAttr(default_factory=dict)

    def source_span(self, pointer: str) -> SourceSpan:
        return self._spans.get(
            pointer,
            SourceSpan(
                path=self._source_path,
                line=1,
                column=1,
                end_line=1,
                end_column=1,
            ),
        )


class MeshSource(FrozenModel):
    """Content-addressed mesh input of an asset.

    ``path`` is the package-internal path (``meshes/<sha256>.msh``) and ``sha256``
    the digest of the source mesh bytes; ``importer`` selects the reader.
    """

    importer: str
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def canonical_package_path(self) -> "MeshSource":
        normalize_package_path(self.path)
        return self


class AssetVariant(FrozenModel):
    """Named variant of an asset (for example a refinement level)."""

    name: str


class SemanticSupport(FrozenModel):
    """Named region, boundary or node set and its topological dimension."""

    name: str
    kind: Literal["region", "boundary", "node_set"]
    dimension: int = Field(ge=0, le=3)


class Asset(Document):
    """Geometry asset: mesh source, dimension, cell type, geometry order and the
    semantic supports that cases may reference.
    """

    kind: Literal["slope_stability.asset"]
    mesh_source: MeshSource
    dimension: Literal[2, 3]
    cell_type: Literal["triangle", "tetrahedron"]
    coordinate_order: Literal[1, 2, 4]
    variants: tuple[AssetVariant, ...]
    supports: tuple[SemanticSupport, ...]

    @model_validator(mode="after")
    def topology_matches_dimension(self) -> "Asset":
        expected = "triangle" if self.dimension == 2 else "tetrahedron"
        if self.cell_type != expected:
            raise ValueError(f"dimension {self.dimension} requires {expected}")
        if len({item.name for item in self.variants}) != len(self.variants):
            raise ValueError("asset variant names must be unique")
        if len({item.name for item in self.supports}) != len(self.supports):
            raise ValueError("semantic support names must be unique")
        return self


class AssetReference(FrozenModel):
    """Reference from a case to an asset name and variant."""

    name: str
    variant: str


class Discretization(FrozenModel):
    """Field polynomial order, coordinate (geometry) order and volume rule.

    ``volume_quadrature`` selects the registered volume rule used by every
    operator, output and coupling target: ``"default"`` (also when absent) or
    ``"paired_higher"``, the next registered rule above the operator degree.
    """

    field_order: Literal[1, 2, 4, 8]
    coordinate_order: Literal[1, 2, 4]
    volume_quadrature: Literal["default", "paired_higher"] | None = None


class MaterialAssignment(FrozenModel):
    """Material model and parameters assigned to one region support."""

    name: str
    support: str
    model: str
    parameters: dict[str, Any]


class Physics(FrozenModel):
    """Enabled physics (mechanics, steady seepage) and their coupling mode."""

    mechanics: bool
    seepage: bool
    coupling: Literal["none", "pressure_gradient_and_weight", "weight_only"]


class BoundaryRule(FrozenModel):
    """Essential or natural boundary condition on one support.

    ``value_model`` names a registered value model whose ``parameters`` are validated
    by the model registry.
    """

    name: str
    field: Literal["displacement", "pore_pressure"]
    condition: Literal["essential", "natural"]
    support: str
    components: tuple[str, ...]
    value_model: str
    parameters: dict[str, Any]
    geometry_patch: str | None = None


class AnalysisRequest(FrozenModel):
    """Requested analysis: limit load, SSR direct or indirect continuation, or steady
    seepage.
    """

    selector: Literal[
        "limit_load.work_controlled",
        "ssr.direct_potential",
        "ssr.indirect_work_controlled",
        "seepage.steady_unconfined",
    ]


class ProfileSelection(FrozenModel):
    """Names of the solver profile selected for each concern."""

    continuation: str
    nonlinear: str
    linear: str
    pmg: str
    deflation: str
    output: str
    validation: str
    seepage: str | None = None


OutputQuantity = Literal[
    "displacement",
    "strain",
    "stress",
    "material_branch",
    "plastic_indicator",
    "yield_residual",
    "pore_pressure",
    "pressure_gradient",
    "conductivity",
    "hydraulic_ramp",
    "hydraulic_epsilon",
    "saturation",
    "pressure_flux",
    "darcy_velocity",
    "coupled_body_force",
    "global_flux_balance",
    "continuation",
    "nonlinear_history",
    "safety_factor",
    "load_factor",
    "work_control",
    "maximum_displacement",
]


class PrescribedSaturation(FrozenModel):
    """Clamped piecewise-linear water level; saturation means y <= level."""

    axis: Literal["x", "z"]
    coordinates: tuple[float, ...] = Field(min_length=1, max_length=32)
    levels: tuple[float, ...] = Field(min_length=1, max_length=32)
    length_unit: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_profile(self) -> "PrescribedSaturation":
        if len(self.coordinates) != len(self.levels):
            raise ValueError(
                "saturation coordinates and levels must have equal lengths"
            )
        if not all(math.isfinite(value) for value in (*self.coordinates, *self.levels)):
            raise ValueError("saturation coordinates and levels must be finite")
        if any(
            right <= left for left, right in zip(self.coordinates, self.coordinates[1:])
        ):
            raise ValueError("saturation coordinates must be strictly increasing")
        return self


class Case(Document):
    """Complete problem definition: asset, discretization, materials, physics, boundary
    rules, analysis, profile selection and required outputs.
    """

    kind: Literal["slope_stability.case"]
    asset: AssetReference
    discretization: Discretization
    materials: tuple[MaterialAssignment, ...]
    physics: Physics
    prescribed_saturation: PrescribedSaturation | None = None
    boundary_rules: tuple[BoundaryRule, ...]
    analysis: AnalysisRequest
    profiles: ProfileSelection
    case_overrides: dict[str, Any]
    required_outputs: tuple[OutputQuantity, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_required_outputs(self) -> "Case":
        if self.prescribed_saturation is not None and (
            not self.physics.mechanics
            or self.physics.seepage
            or self.physics.coupling != "none"
        ):
            raise ValueError(
                "prescribed saturation requires uncoupled mechanics "
                "without a seepage solve"
            )
        if len(set(self.required_outputs)) != len(self.required_outputs):
            raise ValueError("required output IDs must be unique")
        return self


class Profile(Document):
    """Named solver policy for one concern (``family``).

    ``policy`` is validated against the typed policy model of that family.
    """

    kind: Literal["slope_stability.profile"]
    family: Literal[
        "continuation",
        "nonlinear",
        "linear",
        "pmg",
        "deflation",
        "seepage",
        "output",
        "validation",
    ]
    policy: dict[str, Any]


class ContinuationPolicy(FrozenModel, ContinuationSettings):
    """Load-factor continuation controls (steps, work budget and stopping rules)."""

    method: str
    initial_factor: float = Field(gt=0.0)
    initial_step: float = Field(gt=0.0)
    minimum_step: float = Field(gt=0.0)
    maximum_step: float = Field(gt=0.0)
    maximum_work: float = Field(gt=0.0)
    initial_work_scale: float = Field(gt=0.0)
    scaled_slope_tolerance: float = Field(ge=0.0)
    factor_increment_stop: float = Field(ge=0.0)
    maximum_accepted_states: int = Field(ge=1, le=MAXIMUM_ACCEPTED_STATES)
    initial_nonlinear_stopping_criterion: Literal[
        "relative_residual",
        "relative_correction",
    ]
    initial_nonlinear_stopping_tolerance: float = Field(gt=0.0)

    @model_validator(mode="after")
    def ordered_steps(self) -> "ContinuationPolicy":
        if not (self.minimum_step <= self.initial_step <= self.maximum_step):
            raise ValueError("minimum_step <= initial_step <= maximum_step is required")
        return self


class NonlinearPolicy(FrozenModel, NonlinearSettings):
    """Newton solver tolerances, stopping criterion and regularization."""

    method: str
    relative_tolerance: float = Field(gt=0.0)
    absolute_tolerance: float = Field(gt=0.0)
    stopping_criterion: Literal[
        "relative_residual",
        "relative_correction",
        "absolute_delta_lambda",
    ]
    stopping_tolerance: float = Field(gt=0.0)
    regularization_minimum: float = Field(gt=0.0, le=1.0)
    maximum_iterations: int = Field(ge=1)


class LinearPolicy(FrozenModel, LinearSettings):
    """Krylov method, stopping test and tolerances of the mechanics linear solve."""

    ksp_type: str
    operator_class: str
    stopping_condition: Literal[
        "original_rhs_true_residual",
        "projected_rhs_petsc",
        "preconditioned_residual_original_rhs",
        "natural_residual_initial",
    ] = "original_rhs_true_residual"
    relative_tolerance: float = Field(gt=0.0)
    absolute_tolerance: float = Field(gt=0.0)
    maximum_iterations: int = Field(ge=1)
    restart: int = Field(ge=1)
    recycle_dimension: int = Field(default=0, ge=0)
    expert_options: dict[str, str | None]


class PMGPolicy(FrozenModel, PmgSettings):
    """P-multigrid hierarchy, smoother and coarse-solver policy."""

    enabled: bool
    smoother: Literal[
        "none",
        "chebyshev",
        "chebyshev_point",
        "fgmres",
        "fgmres_block",
        "richardson",
        "richardson_block",
    ]
    base_iterations: int = Field(default=2, ge=1)
    hydraulic_base_iterations: int | None = Field(default=None, strict=True, ge=1)
    pre_smoothing_steps: int = Field(default=2, ge=1)
    post_smoothing_steps: int = Field(default=2, ge=1)
    coarse_solver: Literal["gamg", "preonly_lu"]
    coarse_direct_maximum_dofs: int = Field(ge=0)
    coarse_redundant_group_size: int = Field(default=0, ge=0, le=1)
    active_rank_policy: str
    levels: tuple[Literal[1, 2, 4, 8], ...] | None = None

    @model_validator(mode="after")
    def enabled_smoother(self) -> "PMGPolicy":
        if self.enabled and self.smoother == "none":
            raise ValueError("enabled PMG requires a smoother")
        if self.coarse_redundant_group_size and self.coarse_solver != "preonly_lu":
            raise ValueError("redundant L1 solves require coarse_solver=preonly_lu")
        return self


class DeflationPolicy(FrozenModel):
    """Deflation basis source, projector, admission and compression policy."""

    enabled: bool
    projector: str
    maximum_dimension: int = Field(ge=0)
    candidate_source: Literal[
        "accepted_newton_and_control_directions",
        "accepted_state_vectors",
        "accepted_states_and_full_krylov",
        "accepted_states_and_system_solutions",
    ] = "accepted_newton_and_control_directions"
    recycle: Literal[
        "accepted_steps",
        "attempt_snapshot_restore",
    ] = "accepted_steps"
    admission_tolerance: float = Field(default=1.0e-3, gt=0.0)
    pivot_tolerance: float = Field(default=1.0e-13, gt=0.0)
    condition_limit: float = Field(default=1.0e12, gt=1.0)
    invariant_tolerance: float = Field(default=1.0e-9, gt=0.0)
    reorthogonalization_passes: int = Field(default=2, ge=1, le=4)
    dense_factor: Literal["partial_pivot_lu"] = "partial_pivot_lu"
    compression: Literal[
        "none",
        "snapshot_pod",
        "operator_svd",
        "snapshot_pod_total",
        "system_solution_window",
    ] = "none"
    compression_dimension: int = Field(default=0, ge=0)
    compression_history_decay: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def compatible_lifecycle(self) -> "DeflationPolicy":
        supported = {
            (
                "accepted_newton_and_control_directions",
                "accepted_steps",
            ),
            (
                "accepted_state_vectors",
                "attempt_snapshot_restore",
            ),
            (
                "accepted_states_and_full_krylov",
                "attempt_snapshot_restore",
            ),
            (
                "accepted_states_and_system_solutions",
                "attempt_snapshot_restore",
            ),
        }
        if (self.candidate_source, self.recycle) not in supported:
            raise ValueError("candidate_source and recycle must select one lifecycle")
        if (self.compression == "none") != (self.compression_dimension == 0):
            raise ValueError(
                "compression_dimension must be zero only when compression is none"
            )
        full_krylov_compression = self.compression in {
            "snapshot_pod",
            "operator_svd",
            "snapshot_pod_total",
        }
        rolling_compression = self.compression == "system_solution_window"
        if self.compression != "none" and (
            self.projector != "legacy_a_orthogonal"
            or self.compression_dimension > self.maximum_dimension
            or (
                full_krylov_compression
                and self.candidate_source != "accepted_states_and_full_krylov"
            )
            or (
                rolling_compression
                and self.candidate_source != "accepted_states_and_system_solutions"
            )
        ):
            raise ValueError(
                "compression requires its matching full-Krylov or system-solution "
                "lifecycle, the legacy projector, and a dimension within "
                "maximum_dimension"
            )
        return self


class SeepagePolicy(FrozenModel):
    """Linear and nonlinear iteration limits of the steady seepage solve."""

    linear_tolerance: float = Field(gt=0.0)
    linear_max_iter: int = Field(ge=1)
    nonlinear_max_iter: int = Field(ge=1)


class OutputPolicy(FrozenModel, OutputsSettings):
    """Published quantities, history publication and capture frequency."""

    quantities: tuple[OutputQuantity, ...] = Field(min_length=1)
    history: bool
    frequency: Literal["terminal", "accepted_state"]

    @model_validator(mode="after")
    def unique_quantities(self) -> "OutputPolicy":
        if len(set(self.quantities)) != len(self.quantities):
            raise ValueError("output quantity IDs must be unique")
        return self


class ValidationPolicy(FrozenModel, ValidationSettings):
    """Qualification mode and conservation-balance tolerances."""

    reject_unused_options: bool
    conservation_physical_scale: float = Field(gt=0.0)
    conservation_normalized_tolerance: float = Field(ge=0.0)


_PROFILE_POLICY_MODELS: dict[str, type[FrozenModel]] = {
    "continuation": ContinuationPolicy,
    "nonlinear": NonlinearPolicy,
    "linear": LinearPolicy,
    "pmg": PMGPolicy,
    "deflation": DeflationPolicy,
    "seepage": SeepagePolicy,
    "output": OutputPolicy,
    "validation": ValidationPolicy,
}


class SweepDefinition(FrozenModel):
    """One Cartesian sweep axis over a dotted policy path."""

    name: str
    path: str
    values: tuple[Any, ...] = Field(min_length=1)


class OpenMPPolicy(FrozenModel):
    """OpenMP runtime controls (dynamic teams, nesting, binding, places)."""

    dynamic: bool
    max_active_levels: int = Field(ge=1)
    proc_bind: str
    places: str


class ThreadPools(FrozenModel):
    """Thread counts of third-party libraries (BLAS, HYPRE, PETSc)."""

    blas: int = Field(ge=1)
    hypre: int = Field(ge=1)
    petsc: int = Field(ge=1)


class ActiveRanks(FrozenModel):
    """World ranks active on the fine and on the coarsest PMG level."""

    fine: tuple[int, ...] = Field(min_length=1)
    coarse: tuple[int, ...] = Field(min_length=1)


class ResourceRow(FrozenModel, ResourcesSettings):
    """Exact MPI x OpenMP resource layout of one run.

    ``world_size`` ranks with ``openmp_threads`` threads each, bound to
    ``cores_per_rank`` cores; MPI is initialized with ``MPI_THREAD_FUNNELED``.
    """

    name: str
    nodes: int = Field(ge=1)
    world_size: int = Field(ge=1)
    ranks_per_node: int = Field(ge=1)
    openmp_threads: int = Field(ge=1)
    cores_per_rank: int = Field(ge=1)
    smt: Literal["off", "on"]
    cpu_binding: str
    memory_binding: str
    numa_policy: str
    mpi_thread_level: Literal["funneled"]
    openmp: OpenMPPolicy
    third_party_threads: ThreadPools
    active_ranks: ActiveRanks
    memory_per_node_mib: int | None = Field(default=None, strict=True, ge=1)
    address_space_per_rank_mib: int | None = Field(default=None, strict=True, ge=1)


class CollectionPolicy(FrozenModel):
    """Repetition and warm-up counts of a measurement suite."""

    repetitions: int = Field(ge=1)
    warmups: int = Field(ge=0)


class Suite(Document):
    """Collection of cases, profiles, sweep axes and resource rows."""

    kind: Literal["slope_stability.suite"]
    cases: tuple[str, ...] = Field(min_length=1)
    profiles: tuple[str, ...]
    axes: tuple[SweepDefinition, ...]
    resources: tuple[ResourceRow, ...] = Field(min_length=1)
    collection: CollectionPolicy


AuthoringDocument = Asset | Case | Profile | Suite


class UnitSystem:
    """Project-owned wrapper around Pint with fixed internal units."""

    _INTERNAL_UNITS = {
        "length": "m",
        "time": "s",
        "force": "kN",
        "stress": "kPa",
        "pore_pressure": "kPa",
        "unit_weight": "kN/m^3",
        "conductivity": "m/s",
        "pressure_gradient": "kPa/m",
        "dimensionless": "1",
        "angle": "radian",
    }

    def __init__(self) -> None:
        self._registry = pint.UnitRegistry(
            autoconvert_offset_to_baseunit=False,
        )

    def to_internal(
        self,
        quantity: Quantity,
        *,
        expected_dimension: str,
        compilation_boundary: bool,
    ) -> UnitConversion:
        if not compilation_boundary:
            raise UnitError(
                "conversion_outside_compilation",
                "unit conversion is only permitted at the compilation boundary",
            )
        if expected_dimension not in self._INTERNAL_UNITS:
            raise UnitError(
                "unknown_dimension",
                f"unknown physical dimension {expected_dimension!r}",
            )
        if not quantity.unit:
            raise UnitError("missing_unit", "an explicit unit is required")
        try:
            parsed = self._registry.Quantity(quantity.value, quantity.unit)
            target = self._registry.Unit(self._INTERNAL_UNITS[expected_dimension])
            converted = parsed.to(target)
            unit_scale = self._registry.Quantity(1.0, quantity.unit).to(target)
        except pint.errors.UndefinedUnitError as error:
            raise UnitError("unknown_unit", str(error)) from error
        except (
            pint.errors.DimensionalityError,
            pint.errors.OffsetUnitCalculusError,
        ) as error:
            raise UnitError("incompatible_unit", str(error)) from error
        converted_value = float(converted.magnitude)
        scale = float(unit_scale.magnitude)
        if not math.isfinite(converted_value) or not math.isfinite(scale):
            raise UnitError("nonfinite_conversion", quantity.unit)
        return UnitConversion(
            original_value=quantity.value,
            original_unit=quantity.unit,
            converted_value=converted_value,
            internal_unit=self._INTERNAL_UNITS[expected_dimension],
            scale=scale,
            offset=0.0,
            dimension=expected_dimension,
        )


class _EmptyParameters(FrozenModel):
    pass


class _ConstantScalar(FrozenModel):
    value: float
    unit: str


class _ConstantVector(FrozenModel):
    value: tuple[float, ...] = Field(min_length=1)
    unit: str


class _AffineCoordinate(FrozenModel):
    offset: float
    gradient: tuple[float, ...] = Field(min_length=1)
    unit: str


class _Hydrostatic(FrozenModel):
    reference_level: float
    reference_pressure: float
    unit_weight: float = Field(gt=0.0)
    length_unit: str
    pressure_unit: str


class _PiecewiseLinear(FrozenModel):
    coordinates: tuple[float, ...] = Field(min_length=2)
    values: tuple[float, ...] = Field(min_length=2)
    coordinate_unit: str
    value_unit: str

    @model_validator(mode="after")
    def matching_lengths(self) -> "_PiecewiseLinear":
        if len(self.coordinates) != len(self.values):
            raise ValueError("coordinates and values must have equal length")
        if any(
            left >= right for left, right in zip(self.coordinates, self.coordinates[1:])
        ):
            raise ValueError("coordinates must be strictly increasing")
        return self


class _SpatialTraction(FrozenModel):
    offset: tuple[float, ...] = Field(min_length=1)
    gradient: tuple[tuple[float, ...], ...] = Field(min_length=1)
    unit: str


class _MohrCoulomb(FrozenModel):
    cohesion: Quantity
    friction_angle: Quantity
    dilation_angle: Quantity
    young_modulus: Quantity
    poisson_ratio: float | Quantity
    unit_weight: Quantity | None = None
    saturated_unit_weight: Quantity | None = None
    unsaturated_unit_weight: Quantity | None = None
    davis: Literal["A", "B", "C"]

    @model_validator(mode="after")
    def complete_unit_weight(self) -> "_MohrCoulomb":
        legacy = self.unit_weight is not None
        pair = (
            self.saturated_unit_weight is not None
            and self.unsaturated_unit_weight is not None
        )
        partial_pair = (self.saturated_unit_weight is None) != (
            self.unsaturated_unit_weight is None
        )
        if partial_pair or legacy == pair:
            raise ValueError("provide unit_weight or the saturated/unsaturated pair")
        return self


class _SteadyDarcy(FrozenModel):
    conductivity: Quantity
    water_unit_weight: Quantity


@dataclass(frozen=True)
class AdaptedModel:
    """Registered model identifier with its validated parameter model."""

    model_id: str
    parameters: FrozenModel


@dataclass(frozen=True)
class ModelDescriptor:
    """Registry entry: model id, major version, parameter validator, allowed fields and
    the support dimension it requires.
    """

    id: str
    major: int
    parameter_adapter: TypeAdapter[Any]
    allowed_fields: frozenset[str]
    required_support_dimension: int | None


class ModelRegistry:
    """Lookup of registered material, value and analysis models."""

    def __init__(self, descriptors: tuple[ModelDescriptor, ...]) -> None:
        self._descriptors = {item.id: item for item in descriptors}

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._descriptors))

    def adapt(self, model_id: str, parameters: Mapping[str, Any]) -> AdaptedModel:
        descriptor = self._descriptors.get(model_id)
        if descriptor is None:
            raise RegistryError("unknown_model", model_id)
        try:
            adapted = descriptor.parameter_adapter.validate_json(
                canonicalize(dict(parameters))
            )
        except (ValidationError, CanonicalizationError) as error:
            raise RegistryError(
                "invalid_model_parameters", f"{model_id}: {error}"
            ) from error
        return AdaptedModel(model_id=model_id, parameters=adapted)


def _descriptor(
    model_id: str,
    model: type[FrozenModel],
    *,
    support_dimension: int | None = None,
) -> ModelDescriptor:
    return ModelDescriptor(
        id=model_id,
        major=1,
        parameter_adapter=TypeAdapter(model),
        allowed_fields=frozenset(model.model_fields),
        required_support_dimension=support_dimension,
    )


def default_model_registry() -> ModelRegistry:
    """Return the registry of all built-in material, boundary-value, coupling and
    analysis models.
    """

    descriptors = (
        _descriptor("constant_scalar", _ConstantScalar),
        _descriptor("constant_vector", _ConstantVector),
        _descriptor("affine_coordinate", _AffineCoordinate),
        _descriptor("hydrostatic", _Hydrostatic),
        _descriptor("piecewise_linear_level", _PiecewiseLinear),
        _descriptor("global_traction", _ConstantVector),
        _descriptor("spatial_global_traction", _SpatialTraction),
        _descriptor("normal_pressure", _ConstantScalar),
        _descriptor("prescribed_flux", _ConstantScalar),
        _descriptor("mohr_coulomb", _MohrCoulomb),
        _descriptor("steady_unconfined_darcy", _SteadyDarcy),
        _descriptor("coupling_none", _EmptyParameters),
        _descriptor("coupling_pressure_gradient_and_weight", _EmptyParameters),
        _descriptor("coupling_weight_only", _EmptyParameters),
        _descriptor("limit_load.work_controlled", _EmptyParameters),
        _descriptor("ssr.direct_potential", _EmptyParameters),
        _descriptor("ssr.indirect_work_controlled", _EmptyParameters),
        _descriptor("seepage.steady_unconfined", _EmptyParameters),
    )
    return ModelRegistry(descriptors)


@dataclass(frozen=True)
class _BoundarySemanticIssue:
    code: str
    pointer: str
    message: str
    remediation: str


def _boundary_rule_semantic_issues(
    rule: BoundaryRule,
    *,
    dimension: int,
    physics: Physics,
) -> tuple[_BoundarySemanticIssue, ...]:
    """Return native-compatible boundary-rule semantic failures."""
    issues: list[_BoundarySemanticIssue] = []
    axes = tuple(("x", "y", "z")[:dimension])
    components = tuple(rule.components)

    def add(
        code: str,
        pointer: str,
        message: str,
        remediation: str,
    ) -> None:
        issues.append(
            _BoundarySemanticIssue(
                code=code,
                pointer=pointer,
                message=message,
                remediation=remediation,
            )
        )

    if rule.field == "displacement" and not physics.mechanics:
        add(
            "boundary_physics_incompatible",
            "/field",
            "displacement boundary data requires enabled mechanics",
            "enable mechanics or remove the displacement rule",
        )
    elif rule.field == "pore_pressure" and not physics.seepage:
        add(
            "boundary_physics_incompatible",
            "/field",
            "pore-pressure boundary data requires enabled seepage",
            "enable seepage or remove the pore-pressure rule",
        )

    if rule.condition == "natural" and not rule.geometry_patch:
        add(
            "missing_geometry_patch",
            "/geometry_patch",
            "natural boundary data requires a geometry patch",
            "select the coordinate-field geometry patch for the support",
        )

    allowed_models: frozenset[str]
    if rule.field == "displacement" and rule.condition == "essential":
        allowed_models = frozenset({"constant_vector", "affine_coordinate"})
    elif rule.field == "displacement":
        allowed_models = frozenset(
            {
                "global_traction",
                "spatial_global_traction",
                "normal_pressure",
                "hydrostatic",
            }
        )
    elif rule.condition == "essential":
        allowed_models = frozenset(
            {
                "constant_scalar",
                "affine_coordinate",
                "hydrostatic",
                "piecewise_linear_level",
            }
        )
    else:
        allowed_models = frozenset({"prescribed_flux"})
    if rule.value_model not in allowed_models:
        add(
            "boundary_value_model_incompatible",
            "/value_model",
            (
                f"{rule.value_model!r} is incompatible with "
                f"{rule.field} {rule.condition} data"
            ),
            "select a registered value model for this field and condition",
        )

    components_valid = True
    component_remediation = ""
    if rule.field == "displacement" and rule.condition == "essential":
        components_valid = components == ("all",) or (
            bool(components)
            and len(set(components)) == len(components)
            and all(component in axes for component in components)
        )
        component_remediation = (
            f"select unique components from {axes!r}, or select 'all' alone"
        )
    elif (
        rule.field == "displacement"
        and rule.condition == "natural"
        and rule.value_model in {"global_traction", "spatial_global_traction"}
    ):
        components_valid = components == ("all",)
        component_remediation = "select the single component 'all'"
    elif (
        rule.field == "displacement"
        and rule.condition == "natural"
        and rule.value_model in {"normal_pressure", "hydrostatic"}
    ):
        components_valid = components == ("normal",)
        component_remediation = "select the single component 'normal'"
    elif rule.field == "pore_pressure" and rule.condition == "essential":
        components_valid = components in {("scalar",), ("pressure",)}
        component_remediation = "select the single scalar pressure component"
    elif rule.field == "pore_pressure" and rule.condition == "natural":
        components_valid = components == ("normal",)
        component_remediation = "select the single component 'normal'"
    if not components_valid:
        add(
            "boundary_components_incompatible",
            "/components",
            (
                f"components {components!r} are incompatible with "
                f"{rule.field} {rule.condition} {rule.value_model!r}"
            ),
            component_remediation,
        )

    parameters = rule.parameters

    def exact_vector(name: str) -> None:
        value = parameters.get(name)
        if not isinstance(value, (list, tuple)) or len(value) != dimension:
            add(
                "boundary_parameter_dimension",
                f"/parameters/{name}",
                f"{name!r} must contain exactly {dimension} values",
                f"supply one {name!r} value per physical coordinate",
            )

    if (
        rule.value_model == "constant_vector"
        and rule.field == "displacement"
        and rule.condition == "essential"
    ):
        value = parameters.get("value")
        allowed_lengths = {dimension}
        if components != ("all",):
            allowed_lengths.add(len(components))
        if not isinstance(value, (list, tuple)) or len(value) not in allowed_lengths:
            add(
                "boundary_parameter_dimension",
                "/parameters/value",
                (
                    "'value' must contain either one entry per selected "
                    f"component or exactly {dimension} coordinate entries"
                ),
                "supply a compact selected-component vector or a full physical vector",
            )
    elif rule.value_model == "global_traction":
        exact_vector("value")
    elif rule.value_model == "affine_coordinate":
        exact_vector("gradient")
    elif rule.value_model == "spatial_global_traction":
        exact_vector("offset")
        gradient = parameters.get("gradient")
        if (
            not isinstance(gradient, (list, tuple))
            or len(gradient) != dimension
            or any(
                not isinstance(row, (list, tuple)) or len(row) != dimension
                for row in gradient
            )
        ):
            add(
                "boundary_parameter_dimension",
                "/parameters/gradient",
                f"'gradient' must be an exact {dimension} by {dimension} matrix",
                "supply one physical-coordinate gradient row per traction component",
            )

    return tuple(issues)


class ValidationIssue(FrozenModel):
    """One semantic validation finding with its source span and remedy."""

    code: str
    severity: Literal["error", "warning"]
    span: SourceSpan
    pointer: str
    object_id: str
    message: str
    remediation: str


class ValidationReport(FrozenModel):
    """Accumulated semantic validation findings of a document set."""

    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def require_valid(self) -> None:
        if not self.valid:
            raise AuthoringError(
                "authoring_invalid",
                f"{len(self.issues)} authoring issue(s)",
                issues=self.issues,
            )


_DOCUMENT_TYPES: dict[str, type[Document]] = {
    "slope_stability.asset": Asset,
    "slope_stability.case": Case,
    "slope_stability.profile": Profile,
    "slope_stability.suite": Suite,
}


def _raw_document(
    data: Mapping[str, Any] | Path | str | bytes,
    *,
    source_path: str,
) -> dict[str, Any]:
    try:
        if isinstance(data, Mapping):
            decoded = loads_strict(canonicalize(dict(data)))
        elif isinstance(data, Path):
            payload = data.read_bytes()
            if data.suffix.lower() == ".toml":
                decoded = tomllib.loads(payload.decode("utf-8", "strict"))
            else:
                decoded = loads_strict(payload)
        elif isinstance(data, (str, bytes)):
            if source_path.lower().endswith(".toml"):
                text = (
                    data.decode("utf-8", "strict") if isinstance(data, bytes) else data
                )
                decoded = tomllib.loads(text)
            else:
                decoded = loads_strict(data)
        else:
            raise TypeError(type(data).__name__)
    except (
        CanonicalizationError,
        tomllib.TOMLDecodeError,
        UnicodeDecodeError,
        TypeError,
    ) as error:
        raise AuthoringError("invalid_document_encoding", str(error)) from error
    if not isinstance(decoded, dict):
        raise AuthoringError("document_not_object", type(decoded).__name__)
    return decoded


def load_document(
    data: Mapping[str, Any] | Path | str | bytes,
    *,
    source_path: str = "<memory>",
    spans: Mapping[str, SourceSpan] | None = None,
) -> AuthoringDocument:
    """Load one strict, immutable authoring document."""
    if isinstance(data, Path) and source_path == "<memory>":
        source_path = str(data)
    raw = _raw_document(data, source_path=source_path)
    kind = raw.get("kind")
    if kind == "slope_stability.profile":
        forbidden = sorted({"parent", "extends", "inherit"}.intersection(raw))
        if forbidden:
            raise AuthoringError(
                "profile_inheritance_forbidden",
                f"profile key {forbidden[0]!r} is forbidden",
            )
    model_type = _DOCUMENT_TYPES.get(kind)
    if model_type is None:
        raise AuthoringError("unknown_document_kind", repr(kind))
    reject_forbidden_public_keys(raw)
    supplied_id = raw.get("id")
    identity_input = dict(raw)
    identity_input.pop("id", None)
    expected_id = content_id(identity_input)
    if supplied_id != expected_id:
        raise AuthoringError(
            "document_identity_mismatch",
            f"expected {expected_id}, received {supplied_id!r}",
        )
    model = model_type.model_validate_json(canonicalize(raw))
    if isinstance(model, Profile) and model.family in _PROFILE_POLICY_MODELS:
        TypeAdapter(_PROFILE_POLICY_MODELS[model.family]).validate_json(
            canonicalize(model.policy)
        )
    object.__setattr__(model, "_source_path", source_path)
    object.__setattr__(model, "_spans", FrozenDict(spans or {}))
    return model


def render_document(document: AuthoringDocument) -> bytes:
    """Render canonical JSON bytes suitable for hashing and round trips."""
    return canonicalize(document.model_dump(mode="json", exclude_none=True))


def authoring_identity(document: AuthoringDocument) -> str:
    """Return the content identity (``sha256:...``) of an authoring document."""

    return document.id


def _issue(
    case: Case,
    code: str,
    pointer: str,
    message: str,
    remediation: str,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        severity="error",
        span=case.source_span(pointer),
        pointer=pointer,
        object_id=case.identity.name,
        message=message,
        remediation=remediation,
    )


def validate_documents(
    *,
    asset: Asset,
    case: Case,
    profiles: tuple[Profile, ...] | list[Profile],
) -> ValidationReport:
    """Accumulate independent semantic authoring errors deterministically."""
    issues: list[ValidationIssue] = []
    if case.analysis.selector == "seepage.steady_unconfined":
        compatible = case.physics.seepage
    else:
        compatible = case.physics.mechanics
    if not compatible:
        issues.append(
            _issue(
                case,
                "analysis_physics_incompatible",
                "/analysis/selector",
                "the selected analysis is not enabled by the physics policy",
                "enable the required physics or select a compatible analysis",
            )
        )
    if case.physics.coupling != "none" and not (
        case.physics.mechanics and case.physics.seepage
    ):
        issues.append(
            _issue(
                case,
                "coupling_without_mechanics_and_seepage",
                "/physics/coupling",
                "one-way coupling requires mechanics and seepage",
                "enable both physics or select coupling 'none'",
            )
        )

    if case.asset.name != asset.identity.name:
        issues.append(
            _issue(
                case,
                "unresolved_asset",
                "/asset/name",
                f"asset {case.asset.name!r} does not match {asset.identity.name!r}",
                "select the supplied asset by its stable name",
            )
        )
    variants = {variant.name for variant in asset.variants}
    if case.asset.variant not in variants:
        issues.append(
            _issue(
                case,
                "unresolved_asset_variant",
                "/asset/variant",
                f"asset variant {case.asset.variant!r} is absent",
                "select a declared asset variant",
            )
        )
    if case.discretization.coordinate_order != asset.coordinate_order:
        issues.append(
            _issue(
                case,
                "coordinate_order_mismatch",
                "/discretization/coordinate_order",
                "case and asset coordinate orders differ",
                "select the asset's coordinate order",
            )
        )

    supports_by_name = {support.name: support for support in asset.supports}
    support_names = set(supports_by_name)
    for index, material in enumerate(case.materials):
        if material.support not in support_names:
            pointer = f"/materials/{index}/support"
            issues.append(
                _issue(
                    case,
                    "unresolved_material_support",
                    pointer,
                    f"material support {material.support!r} is absent",
                    "declare the region support in the asset",
                )
            )
        elif (
            supports_by_name[material.support].kind != "region"
            or supports_by_name[material.support].dimension != asset.dimension
        ):
            pointer = f"/materials/{index}/support"
            issues.append(
                _issue(
                    case,
                    "material_support_incompatible",
                    pointer,
                    f"{material.support!r} is not a full-dimensional region",
                    "assign the material to a region support",
                )
            )
    for index, rule in enumerate(case.boundary_rules):
        if rule.support not in support_names:
            pointer = f"/boundary_rules/{index}/support"
            issues.append(
                _issue(
                    case,
                    "unresolved_boundary_support",
                    pointer,
                    f"boundary support {rule.support!r} is absent",
                    "declare the boundary support in the asset",
                )
            )
        else:
            support = supports_by_name[rule.support]
            compatible = (
                support.kind == "boundary" and support.dimension == asset.dimension - 1
            ) or (
                rule.condition == "essential"
                and support.kind == "node_set"
                and support.dimension < asset.dimension
            )
            if compatible:
                continue
            pointer = f"/boundary_rules/{index}/support"
            issues.append(
                _issue(
                    case,
                    "boundary_support_incompatible",
                    pointer,
                    f"{rule.support!r} is incompatible with {rule.condition}",
                    "use a node set for essential data or a boundary facet",
                )
            )

    registry = default_model_registry()
    registered = set(registry.ids())
    for index, material in enumerate(case.materials):
        if material.model not in registered:
            issues.append(
                _issue(
                    case,
                    "unknown_material_model",
                    f"/materials/{index}/model",
                    f"material model {material.model!r} is not registered",
                    "select a registered native material model",
                )
            )
        else:
            try:
                registry.adapt(material.model, material.parameters)
            except RegistryError as error:
                issues.append(
                    _issue(
                        case,
                        "invalid_material_parameters",
                        f"/materials/{index}/parameters",
                        str(error),
                        "supply the complete registered material schema",
                    )
                )
    for index, rule in enumerate(case.boundary_rules):
        if rule.value_model not in registered:
            issues.append(
                _issue(
                    case,
                    "unknown_value_model",
                    f"/boundary_rules/{index}/value_model",
                    f"value model {rule.value_model!r} is not registered",
                    "select a registered data-only value model",
                )
            )
        else:
            try:
                registry.adapt(rule.value_model, rule.parameters)
            except RegistryError as error:
                issues.append(
                    _issue(
                        case,
                        "invalid_value_parameters",
                        f"/boundary_rules/{index}/parameters",
                        str(error),
                        "supply the complete registered value-model schema",
                    )
                )
            else:
                for semantic in _boundary_rule_semantic_issues(
                    rule,
                    dimension=asset.dimension,
                    physics=case.physics,
                ):
                    issues.append(
                        _issue(
                            case,
                            semantic.code,
                            f"/boundary_rules/{index}{semantic.pointer}",
                            semantic.message,
                            semantic.remediation,
                        )
                    )

    selected = case.profiles.model_dump(exclude_none=True)
    profile_by_name = {profile.identity.name: profile for profile in profiles}
    for family, name in sorted(selected.items()):
        if name not in profile_by_name:
            issues.append(
                _issue(
                    case,
                    "unresolved_profile",
                    f"/profiles/{family}",
                    f"profile {name!r} is absent",
                    "provide the selected concern profile",
                )
            )
        elif profile_by_name[name].family != family:
            issues.append(
                _issue(
                    case,
                    "profile_family_mismatch",
                    f"/profiles/{family}",
                    f"profile {name!r} belongs to {profile_by_name[name].family!r}",
                    f"select a {family!r} profile",
                )
            )
    return ValidationReport(issues=tuple(issues))


__all__ = [
    "ActiveRanks",
    "Asset",
    "AuthoringError",
    "BoundaryRule",
    "Case",
    "Document",
    "ModelDescriptor",
    "ModelRegistry",
    "Profile",
    "Quantity",
    "RegistryError",
    "ResourceRow",
    "SourceSpan",
    "Suite",
    "UnitConversion",
    "UnitError",
    "UnitSystem",
    "ValidationIssue",
    "ValidationReport",
    "authoring_identity",
    "default_model_registry",
    "load_document",
    "render_document",
    "validate_documents",
]
