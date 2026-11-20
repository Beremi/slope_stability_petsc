# SPDX-License-Identifier: BSD-2-Clause
"""Render the C solver option registry for authoring, schemas and users."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = Path("include/slope_internal/core/solver_options.def")
REFERENCE = Path("docs/solver-options.md")
PYTHON_SETTINGS = Path("python/src/slope_stability/_solver_settings.py")
MAXIMUM_ACCEPTED_STATES = 128


@dataclass(frozen=True)
class Option:
    """One typed option from the native registry."""

    stability: str
    section: str
    name: str
    kind: str
    default: bool | int | float | str
    minimum: int | float
    maximum: int | float
    choices: tuple[str, ...]
    description: str

    @property
    def path(self) -> str:
        """Return the resolved path or exact expert flag."""
        return f"{self.section}.{self.name}" if self.section else self.name

    def schema(self) -> dict[str, Any]:
        """Return the JSON Schema declaration for this setting."""
        common = {"default": self.default, "description": self.description}
        if self.choices:
            return {**common, "type": "string", "enum": list(self.choices)}
        if self.kind == "BOOL":
            return {**common, "type": "boolean"}
        return {
            **common,
            "type": "integer" if self.kind == "INT" else "number",
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


def read_registry(root: Path) -> tuple[Option, ...]:
    """Read and check the X-macro source consumed by the native parser."""
    options = []
    for line in (root / REGISTRY).read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"SS_(STABLE|EXPERT)\((.*)\)", line)
        if match is None:
            if line.startswith("SS_"):
                raise ValueError(f"invalid registry row {line!r}")
            continue
        stability = match[1].lower()
        fields = next(csv.reader([match[2]], skipinitialspace=True, strict=True))
        kind, section, name, fallback, minimum, maximum, labels, description = fields
        low, high = map(float if kind == "REAL" else int, (minimum, maximum))
        if not low <= float(fallback) <= high:
            raise ValueError(f"registry default outside bounds for {name}")
        choices = tuple(labels.split("|")) if labels else ()
        if kind == "CHOICE" and (low != 0 or high != len(choices) - 1):
            raise ValueError(f"registry choice bounds mismatch for {name}")
        default: bool | int | float | str
        if kind == "CHOICE":
            default = choices[int(fallback)]
        elif kind == "BOOL":
            default = bool(int(fallback))
        else:
            default = float(fallback) if kind == "REAL" else int(fallback)
        options.append(
            Option(
                stability,
                section if stability == "stable" else "",
                name,
                kind,
                default,
                low,
                high,
                choices,
                description,
            )
        )
    if not options or len({option.path for option in options}) != len(options):
        raise ValueError("solver registry is empty or has duplicate paths")
    return tuple(options)


def render_reference(options: tuple[Option, ...]) -> str:
    """Render user documentation from the exact native defaults and bounds."""
    lines = [
        "# Solver settings",
        "",
        "<!-- Generated from the native solver option registry. -->",
        "",
        "Stable settings are typed fields in a resolved run. Profile settings can",
        "also be overridden during resolution with `--set SECTION.FIELD=JSON`.",
        "The `outputs` section is authored in the `output` profile. Expert flags",
        "are read from PETSc's options database by the native engine.",
        "",
        "| Setting | Status | Type | Default | Allowed | Effect |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for option in options:
        allowed = (
            ", ".join(f"`{label}`" for label in option.choices)
            if option.choices
            else "`false`, `true`"
            if option.kind == "BOOL"
            else f"{option.minimum} to {option.maximum}, inclusive"
        )
        default = (
            str(option.default).lower()
            if isinstance(option.default, bool)
            else str(option.default)
        )
        kind = {
            "INT": "integer",
            "REAL": "number",
            "BOOL": "boolean",
            "CHOICE": "string",
        }[option.kind]
        lines.append(
            f"| `{option.path}` | {option.stability} | {kind} | `{default}` | "
            f"{allowed} | {option.description} |"
        )
    lines.extend(
        [
            "",
            "## Memory admission",
            "",
            "The local launcher resolves the aggregate linear-vector cap to one",
            "quarter of the lower of available physical memory and the effective",
            "cgroup v2 memory headroom. With a per-rank address-space limit, it also",
            "bounds the cap by one quarter of that limit times the number of ranks.",
            "For an explicit multi-node resource row, `memory_per_node_mib` describes",
            "the smallest admitted node allocation and is multiplied by the node",
            "count. Resolution records these inputs and the selected cap in",
            "`resources`. A caller may explicitly set",
            "`resources.linear_vector_memory_cap_mib` from a known allocation.",
            "If memory information is unavailable the fallback is 1,024 MiB.",
            "",
            "The engine estimates the actual configured Krylov basis, optional",
            "correction capture and deflation store from the global number of free",
            "rows before creating assembly plans or running a continuation attempt.",
            "The estimate covers vector payload. PMG storage, operator workspace and",
            "per-rank dense scratch have separate costs.",
            "",
            "## Qualification and continuation limits",
            "",
            "`validation.qualification_mode` selects fixed-order mechanics and PMG",
            "assembly. Its per-row summation order is independent of OpenMP thread",
            "count at a fixed MPI layout. MPI communication and reductions can",
            "produce different floating-point results. The result records the",
            "effective mode as `summary.mechanics_assembly_mode`.",
            "",
            f"SSR supports up to {MAXIMUM_ACCEPTED_STATES} accepted states. Limit load",
            f"supports {MAXIMUM_ACCEPTED_STATES - 1} continuation steps and reserves",
            "one state for its elastic reference. The resolver and engine reject",
            "values outside these bounds.",
            "",
            "## PETSc backends and iteration detail",
            "",
            "`superlu_dist_reuse` selects SamePattern_SameRowPerm, NOROWPERM,",
            "METIS_AT_PLUS_A and equilibration for the distributed coarse LU.",
            "It disables tiny-pivot replacement, refinement, parallel symbolic",
            "factorization, the 3D grid, GPU offload and solver statistics.",
            "Single-rank coarse levels use the PETSc sequential LU. PETSc backend",
            "options such as KSP monitors remain available under the outer and",
            "level prefixes.",
            "",
            "Krylov and PMG detail limits count rows across the run. These optional",
            "tables use the same relational history columns as the solve summaries.",
            "`reference` is the Newton ordinal and `auxiliary` joins rows from the",
            "same linear solve. The journal's `linear_start` event gives the same",
            "sequence. `linear_end` reports the original-system verification and",
            "the number of additional cancellation reductions. `krylov` records",
            "PETSc's reported norm at each iteration.",
            "`pmg_apply` records the cycle ordinal, level solve count in `control`,",
            "rank-zero inner iterations, final local residual and reason, bridge",
            "bytes in `tolerance` and operator applications in `step`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_python(options: tuple[Option, ...]) -> str:
    """Generate typed authoring model fields and resolver validators."""
    groups: dict[str, list[Option]] = {}
    for option in options:
        if option.stability == "stable":
            groups.setdefault(option.section, []).append(option)
    lines = [
        '"""Typed fields generated from the native solver option registry."""',
        "",
        "# SPDX-License-Identifier: BSD-2-Clause",
        "# Generated by slope_codegen.solver_options.",
        "",
        "from typing import Literal",
        "",
        "from pydantic import BaseModel, Field, StrictBool, TypeAdapter",
        "",
        f"MAXIMUM_ACCEPTED_STATES = {MAXIMUM_ACCEPTED_STATES}",
        "",
        "",
    ]
    for section, members in groups.items():
        name = section.capitalize() + "Settings"
        lines += [
            f"class {name}(BaseModel):",
            f'    """Typed {section} settings."""',
            "",
        ]
        for option in members:
            field_type = (
                "Literal[" + ", ".join(repr(label) for label in option.choices) + "]"
                if option.choices
                else {"INT": "int", "REAL": "float", "BOOL": "StrictBool"}[option.kind]
            )
            if section == "resources":
                field_type += " | None"
            default = repr(None if section == "resources" else option.default)
            if option.kind in {"INT", "REAL"}:
                lines += [
                    f"    {option.name}: {field_type} = Field(",
                    f"        default={default}, strict=True,",
                    f"        ge={option.minimum!r}, le={option.maximum!r},",
                    "    )",
                ]
            else:
                lines += [f"    {option.name}: {field_type} = {default}"]
        lines += ["", ""]
    lines += ["SETTINGS_MODELS = {"]
    lines += [f'    "{key}": {key.capitalize()}Settings,' for key in groups]
    lines += [
        "}",
        "",
        "",
        "def setting_adapter(section: str, name: str) -> TypeAdapter | None:",
        '    """Return a strict validator for one registered resolved-run setting."""',
        "    model = SETTINGS_MODELS.get(section)",
        "    field = model.model_fields.get(name) if model is not None else None",
        "    if field is None:",
        "        return None",
        "    return TypeAdapter(field.rebuild_annotation())",
        "",
        "",
        "__all__ = [",
        '    "MAXIMUM_ACCEPTED_STATES",',
        *[f'    "{section.capitalize()}Settings",' for section in groups],
        '    "SETTINGS_MODELS",',
        '    "setting_adapter",',
        "]",
        "",
    ]
    # Literal option lists can be long, so format the annotation across lines.
    for index, line in enumerate(lines):
        if len(line) > 88 and ": Literal[" in line:
            prefix, rest = line.split("Literal[", 1)
            labels, assignment = rest.split("]", 1)
            lines[index] = (
                prefix
                + "Literal[\n"
                + "".join("        " + label + ",\n" for label in labels.split(", "))
                + "    ]"
                + assignment
            )
    return "\n".join(lines)


def apply_schemas(records: list[dict[str, Any]], options: tuple[Option, ...]) -> None:
    """Enrich profile and resolved-run schemas from the registry."""
    groups: dict[str, dict[str, Any]] = {}
    for option in options:
        if option.stability == "stable":
            groups.setdefault(option.section, {})[option.name] = option.schema()
    groups["continuation"]["maximum_accepted_states"] = {
        "type": "integer",
        "minimum": 1,
        "maximum": MAXIMUM_ACCEPTED_STATES,
    }
    resources = dict(groups["resources"])
    for name in ("memory_per_node_mib", "address_space_per_rank_mib"):
        resources[name] = {"type": "integer", "minimum": 1}
    for record in records:
        kind = record["kind"]
        if kind == "slope_stability.resolved_run":
            record["section_schemas"] = {
                section: {"type": "object", "properties": fields}
                for section, fields in groups.items()
            }
            record["section_schemas"]["resources"]["properties"] = resources
            record["all_of"] = [
                {
                    "if": {
                        "properties": {
                            "analysis": {
                                "required": ["selector"],
                                "properties": {
                                    "selector": {"const": "limit_load.work_controlled"}
                                },
                            }
                        }
                    },
                    "then": {
                        "properties": {
                            "continuation": {
                                "properties": {
                                    "maximum_accepted_states": {
                                        "maximum": MAXIMUM_ACCEPTED_STATES - 1
                                    },
                                }
                            }
                        }
                    },
                }
            ]
        elif kind == "slope_stability.profile":
            record["all_of"] = [
                {
                    "if": {"properties": {"family": {"const": family}}},
                    "then": {"properties": {"policy": {"properties": groups[section]}}},
                }
                for family, section in (
                    ("continuation", "continuation"),
                    ("nonlinear", "nonlinear"),
                    ("linear", "linear"),
                    ("pmg", "pmg"),
                    ("output", "outputs"),
                    ("validation", "validation"),
                )
            ]
        elif kind == "slope_stability.suite":
            record["section_schemas"]["resources"]["items"] = {
                "type": "object",
                "properties": resources,
            }


def main() -> int:
    """Write the generated option reference, or check it for drift."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render_reference(read_registry(ROOT))
    path = ROOT / REFERENCE
    if args.check:
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            print(f"{REFERENCE} is stale. Run python -m slope_codegen.solver_options")
            return 1
    else:
        path.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
