# SPDX-License-Identifier: BSD-2-Clause
"""Write-once promotion of verified results into reference stores."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .._io import fsync_path, rename_noreplace
from ._validate import _object, _require_digest, _sha256_bytes
from .errors import PromotionError
from .reader import open_result
from .views import IncompleteResult


def _rename_noreplace(source: Path, destination: Path) -> None:
    try:
        rename_noreplace(source, destination)
    except FileExistsError as error:
        raise PromotionError(
            "write-once promotion destination already exists"
        ) from error
    except OSError as error:
        raise PromotionError(
            f"atomic promotion publication failed: {os.strerror(error.errno or 0)}"
        ) from error


@dataclass(frozen=True)
class PromotionAuthority:
    """Reviewed authority that approves a reference promotion."""

    authority_id: str
    evidence_digest: str
    reviewer: str
    approved: bool
    known_defect_disposition: str
    #: Resolved validation-profile IDs the authority accepts as high accuracy.
    #: Results record the content ID of their validation profile; the literal
    #: "high_accuracy" label remains accepted for synthetic fixtures.
    high_accuracy_profile_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromotionRecord:
    """Identity and location of a promoted reference dataset."""

    dataset_id: str
    destination: Path
    source_result_id: str
    authority_id: str


def promote_reference(
    candidate: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    authority: PromotionAuthority,
) -> PromotionRecord:
    """Publish a verified, succeeded result into a write-once reference store.

    The destination must be independent of the candidate and must not exist; the copy
    is fsynced and published with an atomic no-replace rename.

    Raises: PromotionError: The candidate, authority or destination is not
    admissible.
    """

    source_path = Path(candidate).resolve()
    destination_path = Path(destination).resolve()
    if source_path == destination_path or source_path in destination_path.parents:
        raise PromotionError("promotion destination must be independent")
    result = open_result(source_path)
    if isinstance(result, IncompleteResult) or result.status != "succeeded":
        raise PromotionError("only a complete succeeded candidate can be promoted")
    summary = _object(result.envelope["summary"], "summary")
    policy = summary.get("validation_policy")
    if policy != "high_accuracy" and policy not in authority.high_accuracy_profile_ids:
        raise PromotionError("candidate did not use the high-accuracy policy")
    analysis = _object(result.envelope["analysis_outcome"], "analysis_outcome")
    if analysis.get("outcome") != "converged_criterion":
        raise PromotionError("candidate did not converge by its stopping criterion")
    if (
        not authority.approved
        or not authority.reviewer
        or not authority.known_defect_disposition
    ):
        raise PromotionError("promotion lacks reviewed independent authority")
    _require_digest(authority.evidence_digest, "authority evidence digest")
    if (
        authority.authority_id == result.result_id
        or authority.evidence_digest == result.result_id
    ):
        raise PromotionError("promotion authority is not independent")
    if destination_path.exists():
        raise PromotionError("write-once promotion destination already exists")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_id = _sha256_bytes(
        (
            f"{result.result_id}|{authority.authority_id}|"
            f"{authority.evidence_digest}|{authority.reviewer}|"
            f"{authority.known_defect_disposition}"
        ).encode("utf-8")
    )
    result.close()
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_path.name}.promotion-",
            dir=destination_path.parent,
        )
    )
    try:
        for item in source_path.iterdir():
            target = staging / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            elif item.is_file():
                shutil.copy2(item, target)
            else:
                raise PromotionError(f"unsupported candidate artifact: {item.name}")
        record = {
            "kind": "slope_stability.reference_promotion",
            "schema_version": {"major": 1, "minor": 0, "patch": 0},
            "dataset_id": dataset_id,
            "source_result_id": result.result_id,
            "authority": {
                "id": authority.authority_id,
                "evidence_digest": authority.evidence_digest,
                "reviewer": authority.reviewer,
                "approved": authority.approved,
                "known_defect_disposition": authority.known_defect_disposition,
            },
            "immutable": True,
        }
        (staging / "promotion.json").write_text(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        copied = open_result(staging)
        if isinstance(copied, IncompleteResult):
            raise PromotionError("copied promotion candidate is not terminal")
        copied.close()
        for path in staging.rglob("*"):
            if path.is_file():
                fsync_path(path)
                path.chmod(0o444)
        for path in sorted(
            (item for item in staging.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            fsync_path(path)
            path.chmod(0o555)
        fsync_path(staging)
        staging.chmod(0o555)
        _rename_noreplace(staging, destination_path)
        fsync_path(destination_path.parent)
    except Exception:
        if staging.exists():
            for path in staging.rglob("*"):
                if path.is_dir():
                    path.chmod(0o755)
                elif path.is_file():
                    path.chmod(0o644)
            staging.chmod(0o755)
            shutil.rmtree(staging)
        raise
    return PromotionRecord(
        dataset_id,
        destination_path,
        result.result_id,
        authority.authority_id,
    )


__all__ = [
    "PromotionAuthority",
    "PromotionRecord",
    "promote_reference",
]
