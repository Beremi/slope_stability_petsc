# SPDX-License-Identifier: BSD-2-Clause
"""Exceptions raised by result readers, comparators and promotion."""

from __future__ import annotations


class ResultValidationError(ValueError):
    """A terminal artifact failed structural, identity, or digest validation."""


class QuantityUnavailableError(LookupError):
    """A requested quantity is absent or scientifically unavailable."""


class PromotionError(RuntimeError):
    """A reference promotion violates the write-once authority workflow."""


__all__ = [
    "PromotionError",
    "QuantityUnavailableError",
    "ResultValidationError",
]
