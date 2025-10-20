"""Utilities for cleaning conversation text and detecting echoes."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable

_PRINTABLE_RE = re.compile(r"[^\x09\x0a\x0d\x20-\x7E]")
_WHITESPACE_RE = re.compile(r"\s+")


def sanitize_text(text: str, *, placeholder: str = "(no response yet)") -> str:
    """Remove non-printable characters and collapse whitespace."""

    cleaned = _PRINTABLE_RE.sub("", text)
    cleaned = cleaned.replace("\r", " ")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned if cleaned else placeholder


def normalize_for_echo(text: str) -> str:
    cleaned = sanitize_text(text)
    return re.sub(r"[^a-z0-9]+", "", cleaned.lower())


def looks_like_echo(
    candidate: str,
    references: Iterable[str],
    *,
    min_length: int = 3,
    similarity_threshold: float = 0.85,
    prefix_threshold: float = 0.7,
    max_suffix_chars: int = 3,
) -> bool:
    target = normalize_for_echo(candidate)
    if len(target) < min_length:
        return False
    candidate_clean = sanitize_text(candidate).lower()
    for ref in references:
        ref_norm = normalize_for_echo(ref)
        if len(ref_norm) < min_length:
            continue
        if target == ref_norm:
            return True
        ref_clean = sanitize_text(ref).lower()
        # Prefix-based echo detection allows small deviations/noise at the end.
        prefix_len = 0
        for c_char, r_char in zip(candidate_clean, ref_clean):
            if c_char != r_char:
                break
            prefix_len += 1
        if prefix_len >= prefix_threshold * min(len(candidate_clean), len(ref_clean)):
            extra = candidate_clean[prefix_len:]
            if len(extra) <= max_suffix_chars:
                return True
            continue
        # Treat short additive noise as an echo (e.g., repeats with small gibberish suffix).
        if target.startswith(ref_norm):
            extra = target[len(ref_norm) :]
            if len(extra) <= max_suffix_chars:
                return True
        if ref_norm.startswith(target):
            extra = ref_norm[len(target) :]
            if len(extra) <= max_suffix_chars:
                return True
        if SequenceMatcher(a=target, b=ref_norm).ratio() >= similarity_threshold:
            return True
    return False
