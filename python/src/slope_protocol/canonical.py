# SPDX-License-Identifier: BSD-2-Clause
"""Strict project-owned canonical JSON, paths, and content identities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
import struct
from typing import Any
from urllib.parse import urlsplit

from .generated import FORBIDDEN_PUBLIC_KEYS


class CanonicalizationError(ValueError):
    """Stable strict-protocol rejection."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _reject_constant(value: str) -> None:
    raise CanonicalizationError("invalid_number", value)


def _strict_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise CanonicalizationError("duplicate_key", key)
        result[key] = value
    return result


def _validate_string(value: str) -> None:
    for character in value:
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise CanonicalizationError("invalid_unicode", "surrogate code point")


def _validate_value(value: Any) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if not -(2**63) <= value <= 2**63 - 1:
            raise CanonicalizationError("integer_range", str(value))
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError("invalid_number", repr(value))
        if value == 0.0 and math.copysign(1.0, value) < 0.0:
            raise CanonicalizationError("ambiguous_number", "negative zero")
        return
    if isinstance(value, str):
        _validate_string(value)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError("object_key_type", type(key).__name__)
            _validate_string(key)
            _validate_value(item)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _validate_value(item)
        return
    raise CanonicalizationError("unsupported_type", type(value).__name__)


def loads_strict(data: str | bytes) -> Any:
    """Decode exactly one strict JSON value."""
    try:
        text = data.decode("utf-8", "strict") if isinstance(data, bytes) else data
    except UnicodeDecodeError as error:
        raise CanonicalizationError("invalid_unicode", str(error)) from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except CanonicalizationError:
        raise
    except json.JSONDecodeError as error:
        code = "trailing_data" if error.msg == "Extra data" else "invalid_json"
        raise CanonicalizationError(code, error.msg) from error
    _validate_value(value)
    return value


def _same_float(left: float, right: float) -> bool:
    return struct.pack(">d", left) == struct.pack(">d", right)


_EXPONENT = re.compile(r"^(.*)[eE]([+-]?)(0*)([0-9]+)$")


def _normalize_exponent(text: str) -> str:
    match = _EXPONENT.match(text)
    if match is None:
        return text
    mantissa, sign, _zeros, digits = match.groups()
    exponent_sign = "-" if sign == "-" else ""
    return f"{mantissa}e{exponent_sign}{int(digits)}"


def _format_float(value: float) -> str:
    for precision in range(1, 18):
        candidate = _normalize_exponent(format(value, f".{precision}g"))
        if _same_float(float(candidate), value):
            return candidate
    raise AssertionError("binary64 did not round-trip at precision 17")


def _format_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _encode(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _format_float(value)
    if isinstance(value, str):
        return _format_string(value)
    if isinstance(value, Mapping):
        return (
            "{"
            + ",".join(
                f"{_format_string(key)}:{_encode(value[key])}" for key in sorted(value)
            )
            + "}"
        )
    return "[" + ",".join(_encode(item) for item in value) + "]"


def canonicalize(value: Any) -> bytes:
    """Return canonical UTF-8 bytes without a trailing newline."""
    _validate_value(value)
    return _encode(value).encode("utf-8")


def canonicalize_json(data: str | bytes) -> bytes:
    return canonicalize(loads_strict(data))


def content_id(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonicalize(value)).hexdigest()


def normalize_package_path(path: str) -> str:
    """Normalize and validate a protocol package-relative path."""
    _validate_string(path)
    if not path or "\x00" in path or "\\" in path:
        raise CanonicalizationError("invalid_path", path)
    if urlsplit(path).scheme or path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        raise CanonicalizationError("absolute_or_url_path", path)
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise CanonicalizationError("path_traversal", path)
    normalized = str(PurePosixPath(*parts))
    if normalized != path:
        raise CanonicalizationError("noncanonical_path", path)
    return normalized


def package_id(
    manifest_without_id: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
) -> str:
    """Digest semantic manifest content and an ordered artifact index."""
    manifest = dict(manifest_without_id)
    manifest.pop("id", None)
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in artifacts:
        item = dict(record)
        item["path"] = normalize_package_path(str(item["path"]))
        if item["path"] in seen:
            raise CanonicalizationError("duplicate_artifact_path", item["path"])
        seen.add(item["path"])
        normalized.append(item)
    normalized.sort(key=lambda item: item["path"])
    return content_id({"manifest": manifest, "artifacts": normalized})


def compatibility(
    reader: tuple[int, int, int],
    artifact: tuple[int, int, int],
    *,
    maximum_minor: int = 0,
) -> bool:
    """Apply v1 major/minor/patch compatibility."""
    reader_major, reader_minor, _reader_patch = reader
    artifact_major, artifact_minor, _artifact_patch = artifact
    return (
        reader_major == artifact_major == 1
        and 0 <= artifact_minor <= maximum_minor
        and reader_minor >= artifact_minor
    )


def reject_forbidden_public_keys(value: Any, pointer: str = "") -> None:
    """Reject process-local or executable controls anywhere in public data."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{pointer}/{key}"
            if key in FORBIDDEN_PUBLIC_KEYS:
                raise CanonicalizationError("forbidden_public_key", child)
            reject_forbidden_public_keys(item, child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            reject_forbidden_public_keys(item, f"{pointer}/{index}")
