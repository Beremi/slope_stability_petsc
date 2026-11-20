# SPDX-License-Identifier: BSD-2-Clause
"""Private file-system helpers shared by the serial product layer.

Every JSON document read here goes through :func:`slope_protocol.loads_strict`,
so duplicate keys, non-finite numbers, negative zero, invalid Unicode and
trailing data are rejected exactly as the native strict reader rejects them.
Durable publication uses ``fsync`` on files and their parent directories.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from slope_protocol import CanonicalizationError, loads_strict

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_HASH_BLOCK_BYTES = 1024 * 1024


def read_json_object(
    path: str | os.PathLike[str],
    *,
    error: type[Exception] = ValueError,
) -> dict[str, Any]:
    """Read one strict JSON object from ``path``.

    Args:
        path: File to read.
        error: Exception type raised for unreadable, malformed or non-object
            documents. The original exception is chained.

    Returns:
        The decoded object.
    """
    source = Path(path)
    try:
        value = loads_strict(source.read_bytes())
    except (OSError, CanonicalizationError) as failure:
        raise error(f"cannot read {source.name}: {failure}") from failure
    if not isinstance(value, dict):
        raise error(f"{source.name} must contain a JSON object")
    return value


def dumps_json(value: Any) -> str:
    """Render a JSON document with stable key order and a trailing newline."""
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    )


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Return the lowercase hexadecimal SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(_HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def fsync_path(path: str | os.PathLike[str]) -> None:
    """Flush a file or directory to stable storage."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_text_atomic(path: str | os.PathLike[str], text: str) -> Path:
    """Durably replace ``path`` with ``text`` through a same-directory rename."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        fsync_path(target.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def rename_noreplace(
    source: str | os.PathLike[str], destination: str | os.PathLike[str]
) -> None:
    """Atomically rename ``source`` to ``destination`` unless it exists.

    Raises:
        FileExistsError: ``destination`` already exists.
        OSError: The kernel or C library cannot perform the rename.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "atomic no-replace rename is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    status = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if status != 0:
        number = ctypes.get_errno()
        if number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(number, os.strerror(number), os.fspath(destination))
        raise OSError(number, os.strerror(number), os.fspath(destination))


__all__ = [
    "dumps_json",
    "file_sha256",
    "fsync_path",
    "read_json_object",
    "rename_noreplace",
    "write_text_atomic",
]
