"""Governance — PII detection + redaction (mandatory)."""
from __future__ import annotations

import re
from ..contracts import *  # noqa


# Common PII patterns.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "email",
        re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        ),
    ),
    (
        "phone",
        re.compile(
            r"(?<!\d)"
            r"(?:\+?880[\s.-]?)?"
            r"01[3-9][\s.-]?\d{3}[\s.-]?\d{4}"
            r"(?!\d)"
        ),
    ),
    (
        "credit_card",
        re.compile(
            r"(?<!\d)"
            r"(?:\d{4}[-\s]?){3}\d{4}"
            r"(?!\d)"
        ),
    ),
    (
        "nid",
        re.compile(
            r"(?<![\d+])"
            r"\d{10,17}"
            r"(?!\d)"
        ),
    ),
]


def detect(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, type) PII spans."""
    if not isinstance(text, str) or not text:
        return []

    spans: list[tuple[int, int, str]] = []

    for pii_type, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            spans.append(
                (match.start(), match.end(), pii_type)
            )

    # Sort by position in the original text.
    spans.sort(key=lambda span: (span[0], span[1]))

    # Remove overlapping matches.
    # Earlier/more-specific matches take precedence.
    result: list[tuple[int, int, str]] = []
    last_end = -1

    for start, end, pii_type in spans:
        if start >= last_end:
            result.append((start, end, pii_type))
            last_end = end

    return result


def redact(text: str) -> str:
    """Replace detected PII spans with typed redaction markers."""
    if not isinstance(text, str) or not text:
        return text

    spans = detect(text)

    if not spans:
        return text

    parts: list[str] = []
    cursor = 0

    for start, end, pii_type in spans:
        parts.append(text[cursor:start])
        parts.append(f"[REDACTED_{pii_type.upper()}]")
        cursor = end

    parts.append(text[cursor:])

    return "".join(parts)


def _redact_value(value):
    """Recursively redact strings contained in pipeline data."""
    if isinstance(value, str):
        return redact(value)

    if isinstance(value, list):
        return [_redact_value(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)

    if isinstance(value, dict):
        return {
            key: _redact_value(item)
            for key, item in value.items()
        }

    # Handle Pydantic-style objects without changing their type.
    if hasattr(value, "model_copy"):
        updates = {}

        for field_name in getattr(value, "model_fields", {}):
            try:
                field_value = getattr(value, field_name)
            except Exception:
                continue

            redacted_value = _redact_value(field_value)

            if redacted_value != field_value:
                updates[field_name] = redacted_value

        if updates:
            return value.model_copy(update=updates)

    return value


def register(hooks) -> None:
    """Wire PII redaction into the pipeline."""

    def _scrub(ctx: dict) -> dict:
        if not isinstance(ctx, dict):
            return ctx

        return {
            key: _redact_value(value)
            for key, value in ctx.items()
        }

    hooks.register(hooks.AFTER_OCR, _scrub)
    hooks.register(hooks.BEFORE_ANSWER, _scrub)
    hooks.register(hooks.ON_LOG, _scrub)