# SPDX-License-Identifier: BSD-2-Clause
"""Private structural validators shared by the result readers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .._io import read_json_object
from .errors import ResultValidationError


_DIGEST_PREFIX = "sha256:"
_DIGEST_CHUNK_ROWS = 1024


_HISTORY_TABLES = (
    "accepted",
    "attempt",
    "newton",
    "line_search",
    "linear",
    "pmg",
    "deflation",
    "seepage",
    "io",
    "krylov",
    "pmg_apply",
)

_OPTIONAL_HISTORY_TABLES = frozenset({"krylov", "pmg_apply"})


def _sha256_bytes(payload: bytes) -> str:
    return _DIGEST_PREFIX + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith(_DIGEST_PREFIX)
        or any(byte not in "0123456789abcdef" for byte in value[7:])
    ):
        raise ResultValidationError(f"{label} is not a canonical sha256 identity")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return read_json_object(path, error=ResultValidationError)


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ResultValidationError(f"{label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(ord(byte) < 0x20 for byte in value)
    ):
        raise ResultValidationError(f"{label} must be a nonempty string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ResultValidationError(f"{label} must be an integer >= {minimum}")
    return value


def _sortable_integer(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _attribute_text(value: object, label: str) -> str:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ResultValidationError(f"{label} is not UTF-8") from error
    return _string(value, label)


def _contained_artifact(root: Path, relative: object) -> Path:
    relative_text = _string(relative, "artifact path")
    relative_path = Path(relative_text)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ResultValidationError("artifact path escapes result directory")
    target = (root / relative_path).resolve()
    if target.parent != root and root not in target.parents:
        raise ResultValidationError("artifact path escapes result directory")
    return target


def _relative_reference(value: object, label: str) -> Path:
    text = _string(value, label)
    path = Path(text)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ResultValidationError(f"{label} is not a contained relative path")
    return path
