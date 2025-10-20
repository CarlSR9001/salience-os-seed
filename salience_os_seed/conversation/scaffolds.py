"""Behavior scaffold loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping

_DEFAULT_SCAFFOLDS: Mapping[str, List[str]] = {
    "greet": ["Hello! How can I assist you today?"],
    "clarify": ["I'm not sure I follow yet—could you share a bit more detail?"],
    "ask_back": ["What outcome would you like to reach so I can help more precisely?"],
    "say_no": ["I can't safely handle that request, but I'm happy to discuss alternatives."],
    "help": ["Let's tackle this together. What's the first thing you'd like to check?"],
}


def load_scaffolds(path: str | Path | None) -> Dict[str, List[str]]:
    if path is None:
        return {key: list(values) for key, values in _DEFAULT_SCAFFOLDS.items()}
    target = Path(path)
    if not target.exists():
        return {key: list(values) for key, values in _DEFAULT_SCAFFOLDS.items()}
    scaffolds: Dict[str, List[str]] = {key: list(values) for key, values in _DEFAULT_SCAFFOLDS.items()}
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        label = str(payload.get("label", "")).strip()
        text = str(payload.get("text", "")).strip()
        if not label or not text:
            continue
        scaffolds.setdefault(label, []).append(text)
    return scaffolds


def pick_scaffold(scaffolds: Mapping[str, List[str]], label: str, fallback: str) -> str:
    options = scaffolds.get(label, [])
    return options[0] if options else fallback
