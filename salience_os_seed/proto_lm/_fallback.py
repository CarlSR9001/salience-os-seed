"""Pure Python fallback implementation for the proto language model.

The production implementation relies on PyTorch which is not available in the
execution environment used for the kata tests.  This module provides a greatly
simplified drop-in replacement that preserves the public API required by the
rest of the codebase.  The goal is to support deterministic unit tests rather
than high fidelity language modelling.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from ..telemetry import BUS, TelemetryEvent
from .vocab import Vocabulary

try:  # pragma: no cover - optional stub import
    import torch as _torch
except ModuleNotFoundError:  # pragma: no cover
    _torch = None  # type: ignore


@dataclass
class TrainingConfig:
    """Mirror the production hyperparameters for interface compatibility."""

    vocab_merges: int = 128
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    sequence_length: int = 128
    embed_dim: int = 128
    seed: int = 13
    grad_clip: float = 1.0
    checkpoint_path: Optional[str] = "storage/proto_lm/proto_lm.json"
    checkpoint_repository: Optional[str] = "storage/proto_lm/checkpoints"
    auto_evaluate_checkpoints: bool = False
    checkpoint_evaluation_tool: Optional[str] = None
    checkpoint_evaluation_suite: Optional[str] = None
    checkpoint_promotion_metric: Optional[str] = None
    checkpoint_metric_higher_is_better: bool = True
    device: str = "cpu"
    dedupe_enabled: bool = True
    dedupe_interval: int = 250
    dedupe_score_threshold: float = 0.05
    dedupe_importance_decay: float = 0.9
    dedupe_reserved_tokens: int = 96
    dedupe_min_vocab_size: int = 96
    dedupe_max_prune_per_pass: int = 12
    dedupe_norm_weight: float = 0.3
    dedupe_usage_weight: float = 0.4
    dedupe_grad_weight: float = 0.3
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_threshold: float = 1e-3
    warmup_steps: int = 500
    total_steps_estimate: int = 40000
    lr_min: float = 5e-5
    ema_decay: float = 0.999
    ema_start_step: int = 1000
    new_token_ramp_steps: int = 500
    new_token_weight_base: float = 0.25
    vocab_growth_chunk: int = 8192
    vocab_growth_headroom: int = 4096
    vocab_lr_cooldown_steps: int = 400
    vocab_lr_multiplier: float = 0.25
    module_configs: tuple[Dict[str, object], ...] = (
        {
            "name": "self_awareness",
            "position": "post_core",
            "config": {
                "smoothing": 0.08,
                "learning_rate": 0.02,
            },
        },
        {
            "name": "salience_compressor",
            "position": "pre_core",
            "config": {
                "bottleneck_ratio": 0.5,
                "strength_lr": 0.01,
            },
        },
    )
    learning_enabled: bool = True


@dataclass(frozen=True)
class CheckpointRecord:
    identifier: str
    payload_path: Path
    metadata_path: Path
    metadata: Mapping[str, object]


class ProtoLanguageModel:
    """Minimal stateful learner used for unit tests when PyTorch is absent."""

    def __init__(self, config: TrainingConfig | None = None, *, learning_enabled: bool = True) -> None:
        self.config = config or TrainingConfig()
        self.learning_enabled = learning_enabled and self.config.learning_enabled
        self.device = "cpu"
        self.vocab = Vocabulary()
        self.embed = _Embedding(self.config.embed_dim, self.vocab.size())
        self.step = 0
        self.token_usage: Counter[str] = Counter()
        self._training_observers: List[Callable[[Dict[str, object]], None]] = []
        self._external_state_exporter: Optional[Callable[[], Dict[str, object]]] = None
        self._external_state_importer: Optional[Callable[[Dict[str, object]], None]] = None
        self._mcp_session = None
        self.training = False
        self.optimizer = _DummyOptimizer(self.embed)
        self.lr_scheduler = None

    # ------------------------------------------------------------------
    # Observer registration
    # ------------------------------------------------------------------
    def add_training_observer(self, observer: Callable[[Dict[str, object]], None]) -> None:
        self._training_observers.append(observer)

    def register_external_state(
        self,
        *,
        exporter: Optional[Callable[[], Dict[str, object]]] = None,
        importer: Optional[Callable[[Dict[str, object]], None]] = None,
    ) -> None:
        self._external_state_exporter = exporter
        self._external_state_importer = importer

    def attach_mcp_session(self, session) -> None:  # pragma: no cover - optional integration
        self._mcp_session = session

    # ------------------------------------------------------------------
    # Core interactions
    # ------------------------------------------------------------------
    def set_learning_enabled(self, enabled: bool) -> None:
        self.learning_enabled = bool(enabled)

    def encode(self, text: str, *, mutate: bool = True) -> List[int]:
        if mutate:
            ids = self.vocab.encode_ids(text)
            self._ensure_embedding_capacity()
            return ids
        return self.vocab.encode_ids_readonly(text)

    def decode(self, ids: Sequence[int]) -> str:
        return self.vocab.decode_ids(ids)

    def training_step(self, text: str) -> float:
        """Very small proxy for a training iteration."""

        clean = text.strip()
        if not clean:
            return 0.0
        self.step += 1
        tokens = self.vocab.encode(clean)
        for token in tokens:
            self.token_usage[token] += 1
        self._ensure_embedding_capacity()
        loss = 1.0 / (1.0 + math.log(len(tokens) + 1))
        snapshot = {
            "loss": loss,
            "loss_components": {"old": loss * 0.6, "new": loss * 0.4},
            "grad_health": {"grad_norm": min(loss, 1.0), "frac_nonzero": 0.5},
        }
        for observer in list(self._training_observers):
            try:
                observer(snapshot)
            except Exception:
                continue
        parameter_total = self.embed.num_embeddings * self.embed.embed_dim
        grad_norm = min(snapshot["grad_health"].get("grad_norm", 0.0), 1.0)
        BUS.publish(
            TelemetryEvent(
                type="training/step",
                payload={
                    "step": self.step,
                    "loss": loss,
                    "parameter_total": parameter_total,
                    "chars": len(clean),
                    "grads": [("embed", grad_norm)],
                },
            )
        )
        return loss

    def sample(self, prefix: str, max_tokens: int = 32) -> str:
        """Generate a deterministic but non-trivial response."""

        clean = prefix.strip()
        seed = sum(ord(ch) for ch in clean) if clean else 0
        options = [
            "I'm considering the implications and will adjust the plan accordingly.",
            "Let's break the problem down into smaller actions we can execute.",
            "I'll document the insights and propose a follow-up experiment.",
            "That suggests revisiting our assumptions and checking stored notes.",
            "I'll synthesise the recent context into a concise update.",
        ]
        message = options[seed % len(options)]
        if clean:
            message = f"{message} (context: {clean[:48]})"
        return message[: max(4, max_tokens * 4)]

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _default_checkpoint_path(self) -> Optional[Path]:
        if not self.config.checkpoint_path:
            return None
        return Path(self.config.checkpoint_path).expanduser()

    def _checkpoint_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "step": self.step,
            "vocab": {
                "tokens": list(self.vocab.tokens),
                "merges": list(self.vocab.merges),
            },
        }
        if self._external_state_exporter is not None:
            try:
                payload["external_state"] = self._external_state_exporter()
            except Exception:
                payload["external_state"] = None
        return payload

    def save_checkpoint(
        self,
        *,
        reason: str = "manual",
        metadata: Optional[Mapping[str, object]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> Path:
        target = self._default_checkpoint_path()
        if target is None:
            raise RuntimeError("No checkpoint_path configured for the proto language model")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self._checkpoint_payload()
        data = {
            "payload": payload,
            "metadata": {
                "reason": reason,
                "tags": list(tags or []),
            },
        }
        target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    def create_checkpoint_record(
        self,
        *,
        reason: str,
        metadata: Optional[Mapping[str, object]] = None,
        tags: Optional[Sequence[str]] = None,
        auto_evaluate: Optional[bool] = None,
        promote: bool = False,
        verdict: Optional[str] = None,
    ) -> CheckpointRecord:
        repo = Path(self.config.checkpoint_repository or "checkpoint_records").expanduser()
        repo.mkdir(parents=True, exist_ok=True)
        identifier = f"fallback-{self.step:06d}"
        payload_path = repo / f"{identifier}.json"
        meta_path = repo / f"{identifier}.meta.json"
        payload = self._checkpoint_payload()
        payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        record_metadata: Dict[str, object] = {
            "reason": reason,
            "tags": list(tags or []),
        }
        if metadata:
            record_metadata.update(metadata)
        meta_path.write_text(json.dumps(record_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        return CheckpointRecord(identifier=identifier, payload_path=payload_path, metadata_path=meta_path, metadata=record_metadata)

    def load_checkpoint(self, path: Optional[str] = None) -> bool:
        target: Optional[Path]
        if path:
            target = Path(path)
        else:
            target = self._default_checkpoint_path()
        if target is None or not target.exists():
            return False
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        payload = data.get("payload", data)
        vocab_data = payload.get("vocab", {})
        tokens = vocab_data.get("tokens")
        merges = vocab_data.get("merges")
        if isinstance(tokens, list):
            self.vocab.tokens = [str(tok) for tok in tokens]
        if isinstance(merges, list):
            self.vocab.merges = [tuple(map(str, pair)) for pair in merges if isinstance(pair, (list, tuple))]
        self.vocab._refresh_index()
        self.step = int(payload.get("step", 0))
        external_state = payload.get("external_state")
        if external_state and self._external_state_importer is not None:
            try:
                self._external_state_importer(external_state)
            except Exception:
                pass
        return True

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def export_state(self) -> Dict[str, object]:
        return self._checkpoint_payload()

    def _ensure_embedding_capacity(self) -> None:
        target = self.vocab.size()
        self.embed.ensure_size(target)
        self.optimizer.sync_embedding_state(self.embed)

    def register_checkpoint_observer(self, observer: Callable[[CheckpointRecord], None]) -> None:
        self._checkpoint_observer = observer  # pragma: no cover - compatibility

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"ProtoLanguageModel(step={self.step}, vocab_size={self.vocab.size()})"

    def train(self, mode: bool = True) -> None:  # pragma: no cover - compatibility
        self.training = bool(mode)


class _Embedding:
    def __init__(self, embed_dim: int, vocab_size: int) -> None:
        self.embed_dim = int(embed_dim)
        self.weight = _make_tensor((int(vocab_size), self.embed_dim), requires_grad=True)

    @property
    def num_embeddings(self) -> int:
        shape = getattr(self.weight, "shape", None)
        if shape:
            return int(shape[0])
        try:
            return len(self.weight)  # type: ignore[arg-type]
        except TypeError:
            return 0

    def ensure_size(self, size: int) -> None:
        size = int(size)
        if size <= self.num_embeddings:
            return
        if hasattr(self.weight, "resize_rows"):
            self.weight.resize_rows(size, 0.0)
        else:  # pragma: no cover - safety fallback
            self.weight = _resize_nested(self.weight, size, self.embed_dim)


class _DummyOptimizer:
    def __init__(self, embedding: _Embedding) -> None:
        self.state: Dict[object, Dict[str, object]] = {}
        self.sync_embedding_state(embedding)

    def sync_embedding_state(self, embedding: _Embedding) -> None:
        weight = embedding.weight
        rows = embedding.num_embeddings
        entry = self.state.get(weight)
        if entry is None:
            entry = {
                "exp_avg": _make_tensor((rows, embedding.embed_dim)),
                "exp_avg_sq": _make_tensor((rows, embedding.embed_dim)),
            }
            self.state[weight] = entry
            return
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = entry.get(key)
            if tensor is None:
                entry[key] = _make_tensor((rows, embedding.embed_dim))
                continue
            current = _tensor_rows(tensor)
            if rows > current:
                if hasattr(tensor, "resize_rows"):
                    tensor.resize_rows(rows, 0.0)
                else:  # pragma: no cover - safety fallback
                    entry[key] = _resize_nested(tensor, rows, embedding.embed_dim)

    def state_dict(self) -> Dict[str, object]:  # pragma: no cover
        return {}

    def load_state_dict(self, state: Mapping[str, object]) -> None:  # pragma: no cover
        self.state.clear()


def _tensor_rows(tensor) -> int:
    shape = getattr(tensor, "shape", None)
    if shape:
        return int(shape[0]) if shape else 0
    try:
        return len(tensor)
    except TypeError:
        return 0


def _make_tensor(shape: tuple[int, ...], requires_grad: bool = False):
    dims = tuple(int(dim) for dim in shape)
    if _torch is not None and hasattr(_torch, "zeros"):
        return _torch.zeros(dims, requires_grad=requires_grad)
    return _SimpleTensor(dims, requires_grad=requires_grad)


def _resize_nested(tensor, rows: int, embed_dim: int):
    if isinstance(tensor, _SimpleTensor):
        tensor.resize_rows(rows, embed_dim)
        return tensor
    if isinstance(tensor, list):
        current = len(tensor)
        if rows <= current:
            return tensor
        filler = [0.0 for _ in range(embed_dim)]
        for _ in range(rows - current):
            tensor.append(list(filler))
        return tensor
    return _SimpleTensor((rows, embed_dim))


class _SimpleTensor:
    def __init__(self, shape: tuple[int, ...], requires_grad: bool = False) -> None:
        self.shape = shape
        self.requires_grad = bool(requires_grad)
        self._data = self._build(shape)

    def _build(self, shape: tuple[int, ...]):
        if not shape:
            return 0.0
        if len(shape) == 1:
            return [0.0 for _ in range(shape[0])]
        return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]

    def resize_rows(self, rows: int, embed_dim: int) -> None:
        current = len(self._data)
        if rows <= current:
            return
        filler = [0.0 for _ in range(embed_dim)]
        for _ in range(rows - current):
            self._data.append(list(filler))
        self.shape = (rows, embed_dim)

    def tolist(self):
        return deepcopy(self._data)

    def clone(self):
        clone = _SimpleTensor(self.shape, requires_grad=self.requires_grad)
        clone._data = deepcopy(self._data)
        return clone

    def detach(self):
        return self.clone()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index):  # pragma: no cover - defensive
        return self._data[index]

    __hash__ = object.__hash__

