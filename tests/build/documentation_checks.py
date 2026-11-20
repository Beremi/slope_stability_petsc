# SPDX-License-Identifier: BSD-2-Clause
"""Check source headers, header documentation and Markdown links.

The checks need only the Python standard library and Git, so they run in a
plain checkout and in continuous integration without Doxygen:

``spdx``
    Every C, header, Python and CMake source owned by the project carries an
    ``SPDX-License-Identifier`` line. Vendored code under ``third_party/`` is
    excluded.
``prototypes``
    Every function declared in a project header (``include/``, ``src/``,
    ``generated/include/`` and the native test headers) is preceded by a
    Doxygen comment. Under ``include/`` the comment must also name every
    parameter with ``@param`` and describe a non-void result with
    ``@return``.
``links``
    Every relative link in a tracked Markdown file resolves to a tracked path.

``docs/Doxyfile`` runs the complete Doxygen check of ``include/`` with
``WARN_IF_UNDOCUMENTED`` when Doxygen is installed; this script is the
dependency-free subset that the test suite runs.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Prototype",
    "check_links",
    "check_prototypes",
    "check_spdx",
    "header_prototypes",
    "main",
]

ROOT = Path(__file__).resolve().parents[2]
SPDX = "SPDX-License-Identifier: BSD-2-Clause"
SOURCE_PATTERNS = (
    "*.c",
    "*.h",
    "*.h.in",
    "*.def",
    "*.py",
    "*.cmake",
    "*.cmake.in",
    "CMakeLists.txt",
    "*.toml",
)
EXCLUDED_PREFIXES = ("third_party/",)
HEADER_PATTERNS = ("*.h",)
STRICT_PREFIX = "include/"
_LINK = re.compile(r"\]\(([^)\s]+)\)")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class Prototype:
    """One function declaration found in a header."""

    path: str
    line: int
    name: str
    parameters: tuple[str, ...]
    returns_value: bool
    documentation: str | None


def _tracked(root: Path, patterns: tuple[str, ...]) -> list[str]:
    """Tracked files matching the patterns, or all files outside a checkout."""
    try:
        output = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "-z", "--", *patterns],
            stderr=subprocess.DEVNULL,
        )
        names = [name for name in output.decode("utf-8").split("\0") if name]
    except (OSError, subprocess.CalledProcessError):
        names = sorted(
            {
                path.relative_to(root).as_posix()
                for pattern in patterns
                for path in root.rglob(pattern)
                if ".git" not in path.relative_to(root).parts
            }
        )
    return sorted(
        name
        for name in names
        if not name.startswith(EXCLUDED_PREFIXES) and (root / name).is_file()
    )


def check_spdx(root: Path = ROOT) -> list[str]:
    """Return the owned source files that lack the SPDX line."""
    return [
        name
        for name in _tracked(root, SOURCE_PATTERNS)
        if SPDX not in (root / name).read_text(encoding="utf-8", errors="replace")
    ]


def _mask(text: str) -> tuple[str, list[tuple[int, int, str]]]:
    """Blank comments, strings and preprocessor lines; keep doc comments."""
    masked = list(text)
    comments: list[tuple[int, int, str]] = []
    index = 0
    length = len(text)
    line_start = True
    while index < length:
        char = text[index]
        if line_start and char == "#":
            end = index
            while end < length:
                newline = text.find("\n", end)
                if newline == -1:
                    end = length
                    break
                if text[newline - 1] == "\\":
                    end = newline + 1
                    continue
                end = newline
                break
            for position in range(index, end):
                if masked[position] != "\n":
                    masked[position] = " "
            index = end
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            end = length if end == -1 else end + 2
            comments.append((index, end, text[index:end]))
            for position in range(index, end):
                if masked[position] != "\n":
                    masked[position] = " "
            index = end
            continue
        if text.startswith("//", index):
            end = text.find("\n", index)
            end = length if end == -1 else end
            comments.append((index, end, text[index:end]))
            for position in range(index, end):
                masked[position] = " "
            index = end
            continue
        if char in "\"'":
            end = index + 1
            while end < length and text[end] != char:
                end += 2 if text[end] == "\\" else 1
            for position in range(index + 1, min(end, length)):
                if masked[position] != "\n":
                    masked[position] = " "
            index = end + 1
            continue
        if char == "\n":
            line_start = True
        elif not char.isspace():
            line_start = False
        index += 1
    return "".join(masked), comments


def _is_doc(comment: str) -> bool:
    return comment.startswith(("/**", "/*!", "///", "//!")) and not comment.startswith(
        "/**/"
    )


def _parameter_names(parameters: str) -> tuple[str, ...]:
    """Names of the declared parameters of one prototype."""
    parameters = parameters.strip()
    if parameters in {"", "void"}:
        return ()
    depth = 0
    parts: list[str] = []
    current: list[str] = []
    for char in parameters:
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    names: list[str] = []
    for part in parts:
        part = part.strip()
        if part == "...":
            continue
        pointer = re.search(r"\(\s*\*\s*(?:const\s+)?([A-Za-z_]\w*)\s*\)", part)
        if pointer:
            names.append(pointer.group(1))
            continue
        part = re.sub(r"\[[^\]]*\]", "", part)
        identifiers = _IDENTIFIER.findall(part)
        if len(identifiers) >= 2:
            names.append(identifiers[-1])
    return tuple(names)


def header_prototypes(root: Path, name: str) -> list[Prototype]:
    """Function declarations of one header with their preceding doc comment."""
    text = (root / name).read_text(encoding="utf-8")
    masked, comments = _mask(text)
    prototypes: list[Prototype] = []
    depth = 0
    start = 0
    index = 0
    while index < len(masked):
        char = masked[index]
        if char == "{":
            if depth == 0:
                _record(prototypes, name, text, comments, masked[start:index], start)
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                start = index + 1
        elif char == ";" and depth == 0:
            _record(prototypes, name, text, comments, masked[start:index], start)
            start = index + 1
        index += 1
    return prototypes


def _record(
    prototypes: list[Prototype],
    name: str,
    text: str,
    comments: list[tuple[int, int, str]],
    statement: str,
    offset: int,
) -> None:
    stripped = statement.strip()
    if not stripped or stripped.startswith(("typedef", "struct", "enum", "union")):
        return
    match = re.search(r"([A-Za-z_]\w*)\s*\(((?:[^()]|\([^()]*\))*)\)", stripped)
    if match is None or "=" in stripped[: match.start()]:
        return
    head = stripped[: match.start()].split()
    if not head:
        return
    function = match.group(1)
    begin = offset + (len(statement) - len(statement.lstrip()))
    previous_end = len(text[:begin].rstrip())
    documentation = next(
        (
            comment
            for _, comment_end, comment in comments
            if comment_end == previous_end and _is_doc(comment)
        ),
        None,
    )
    returns_value = (
        re.fullmatch(r"(?:static\s+|inline\s+|extern\s+)*void", " ".join(head)) is None
    )
    prototypes.append(
        Prototype(
            name,
            text.count("\n", 0, begin) + 1,
            function,
            _parameter_names(match.group(2)),
            returns_value,
            documentation,
        )
    )


def check_prototypes(root: Path = ROOT) -> list[str]:
    """Return one message per undocumented or incompletely documented prototype."""
    problems: list[str] = []
    for name in _tracked(root, HEADER_PATTERNS):
        for prototype in header_prototypes(root, name):
            where = f"{prototype.path}:{prototype.line}: {prototype.name}"
            if prototype.documentation is None:
                problems.append(f"{where}: no doc comment")
                continue
            if not name.startswith(STRICT_PREFIX):
                continue
            documented = set(
                re.findall(r"@param(?:\[[a-z, ]+\])?\s+(\w+)", prototype.documentation)
            )
            missing = [item for item in prototype.parameters if item not in documented]
            if missing:
                problems.append(f"{where}: no @param for {', '.join(missing)}")
            if prototype.returns_value and "@return" not in prototype.documentation:
                problems.append(f"{where}: no @return")
    return problems


def check_links(root: Path = ROOT) -> list[str]:
    """Return the relative Markdown links that do not resolve."""
    problems: list[str] = []
    for name in _tracked(root, ("*.md",)):
        text = (root / name).read_text(encoding="utf-8")
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        for target in _LINK.findall(text):
            target = target.split("#", 1)[0]
            if not target or re.match(r"^[a-z][a-z0-9+.-]*:", target):
                continue
            resolved = os.path.normpath(os.path.join(os.path.dirname(name), target))
            if resolved.startswith("..") or not (root / resolved).exists():
                problems.append(f"{name}: {target}")
    return problems


CHECKS = {
    "spdx": check_spdx,
    "prototypes": check_prototypes,
    "links": check_links,
}


def main(argv: list[str] | None = None) -> int:
    """Run the selected checks and print one line per problem."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checks", nargs="*", help=f"any of {', '.join(CHECKS)}")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    unknown = sorted(set(args.checks) - set(CHECKS))
    if unknown:
        parser.error(f"unknown check: {', '.join(unknown)}")
    failed = False
    for check in args.checks or list(CHECKS):
        problems = CHECKS[check](args.root)
        for problem in problems:
            print(f"{check}: {problem}")
        print(f"{check}: {len(problems)} problem(s)", file=sys.stderr)
        failed = failed or bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
