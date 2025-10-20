"""Conversational loop using SalienceRuntime and ProtoLanguageModel."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Deque, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from datetime import datetime, timezone

from ..core.memory import StructuredMemory
from ..core.operators import MemoryOperator
from ..runtime.orchestrator import RuntimeConfig, RuntimeMetrics, SalienceRuntime
from ..ingestion.reader import CorpusReader
from ..ingestion.dedupe import IngestIndex, content_digest
from ..proto_lm.trainer import ProtoLanguageModel, TrainingConfig
from ..telemetry import BUS, TelemetryEvent
from .sanitize import looks_like_echo, sanitize_text
from .scaffolds import load_scaffolds, pick_scaffold
from .filters import IngestionThresholds, SalienceFilter
from ..adaptive.manager import AdaptiveCoordinator, GatingSummary


@dataclass
class AdaptiveFilteringConfig:
    enabled: bool = False
    warmup_chunks: int = 32
    evaluation_window: int = 32
    enable_ratio: float = 0.3
    disable_ratio: float = 0.6
    cooldown_chunks: int = 32


@dataclass
class IngestionConfig:
    thresholds: IngestionThresholds = field(default_factory=IngestionThresholds)
    chunk_size: int = 2048
    checkpoint_interval: int = 50
    max_chunks: Optional[int] = None
    progress_enabled: bool = True
    adaptive: AdaptiveFilteringConfig = field(default_factory=AdaptiveFilteringConfig)
    chunk_overlap: int = 0
    batch_size: int = 1
    dedupe_enabled: bool = True
    dedupe_db_path: str = "storage/ingestion/index.db"
    allow_reingest_duplicates: bool = False


@dataclass
class ConversationConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    lm: TrainingConfig = field(default_factory=TrainingConfig)
    max_history: int = 16
    response_tokens: int = 256
    auto_save_path: Optional[str] = None
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    learning_enabled: bool = True
    behavior_scaffolds_path: Optional[str] = None
    archive_checkpoint_on_start: bool = True


@dataclass
class ConversationSnapshot:
    metrics: RuntimeMetrics
    response: str
    meta_report: str
    todos: Sequence[str]
    generator_description: str


class ConversationSession:
    """Maintains salience-aware conversation with emergent language learning."""

    def __init__(
        self,
        config: ConversationConfig | None = None,
        corpus_root: Optional[str] = None,
    ) -> None:
        self.config = config or ConversationConfig()
        self.runtime = SalienceRuntime(self.config.runtime)
        self.proto_lm = ProtoLanguageModel(self.config.lm, learning_enabled=self.config.learning_enabled)
        loaded_checkpoint = self.proto_lm.load_checkpoint()
        if self.config.archive_checkpoint_on_start and loaded_checkpoint:
            self._archive_existing_checkpoint()
        self.memory_operator = MemoryOperator(self.runtime.memory)
        self.history: Deque[Tuple[str, str]] = deque(maxlen=self.config.max_history)
        self.reader: Optional[CorpusReader] = None
        self.scaffolds = load_scaffolds(self.config.behavior_scaffolds_path)
        self._goal_embedding = self._compute_embedding("maintain truthful helpful dialogue with the user")
        self.runtime.update_sensor_context("goal", {"embedding": self._goal_embedding})
        self._auto_save_path: Optional[Path] = (
            Path(self.config.auto_save_path).expanduser().resolve()
            if self.config.auto_save_path
            else None
        )
        self._filter = SalienceFilter(self.config.ingestion.thresholds)
        self._ingest_index: Optional[IngestIndex] = None
        if self.config.ingestion.dedupe_enabled and not self.config.ingestion.allow_reingest_duplicates:
            self._ingest_index = IngestIndex(self.config.ingestion.dedupe_db_path)
        if corpus_root:
            self.reader = CorpusReader(corpus_root)
            self.bootstrap_from_corpus(self.reader)
        if self._auto_save_path and self._auto_save_path.exists():
            self.load_state(self._auto_save_path)
        self._learning_buffer: List[str] = []
        self._learning_accumulated_chars: int = 0
        self._learning_last_flush_step: int = self.proto_lm.step
        self._adaptive = AdaptiveCoordinator(runtime=self.runtime, proto_lm=self.proto_lm)
        self._last_gating_summary: Optional[GatingSummary] = None
        self.proto_lm.register_external_state(
            exporter=self._adaptive.export_state,
            importer=self._adaptive.import_state,
        )

    def _maybe_train_on_buffer(self) -> None:
        if not self.config.learning_enabled:
            return
        threshold_chars = max(64, self.config.response_tokens * 2)
        threshold_steps = max(4, self.config.response_tokens // 8)
        if self._learning_accumulated_chars < threshold_chars and (
            self.proto_lm.step - self._learning_last_flush_step
        ) < threshold_steps:
            return
        if not self._learning_buffer:
            return
        payload = "\n".join(self._learning_buffer)
        if payload.strip():
            self.proto_lm.training_step(payload)
        self._learning_buffer.clear()
        self._learning_accumulated_chars = 0
        self._learning_last_flush_step = self.proto_lm.step

    def _schedule_learning(self, text: str) -> None:
        if not self.config.learning_enabled:
            return
        clean = text.strip()
        if not clean:
            return
        self._learning_buffer.append(clean)
        self._learning_accumulated_chars += len(clean)
        self._maybe_train_on_buffer()

    def _set_filter_enabled(self, enabled: bool) -> None:
        self._filter.thresholds = IngestionThresholds(
            enabled=enabled,
            min_uncertainty=self.config.ingestion.thresholds.min_uncertainty,
            min_novelty=self.config.ingestion.thresholds.min_novelty,
            max_drag=self.config.ingestion.thresholds.max_drag,
        )

    def is_filter_enabled(self) -> bool:
        return bool(self._filter.thresholds.enabled)

    def _archive_existing_checkpoint(self) -> None:
        checkpoint_path = self.config.lm.checkpoint_path
        if not checkpoint_path:
            return
        path = Path(checkpoint_path)
        if not path.exists():
            return
        if path.parent.name.startswith("checkpoints_"):
            return
        archive_dir = path.parent / "checkpoints_old"
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        target = archive_dir / f"{path.stem}-{timestamp}{path.suffix or '.pt'}"
        try:
            shutil.move(str(path), str(target))
        except OSError:
            # If move fails, leave original in place rather than crash startup.
            return

    def bootstrap_from_corpus(
        self,
        reader: CorpusReader,
        max_chunks: Optional[int] = None,
        progress_cb: Optional[Callable[[Mapping[str, object]], None]] = None,
    ) -> "IngestionStats":
        effective_max = max_chunks if max_chunks is not None else self.config.ingestion.max_chunks
        loader = IngestionLoop(
            session=self,
            reader=reader,
            chunk_size=self.config.ingestion.chunk_size,
            checkpoint_interval=self.config.ingestion.checkpoint_interval,
            max_chunks=effective_max,
            progress_cb=progress_cb,
            chunk_overlap=self.config.ingestion.chunk_overlap,
            batch_size=self.config.ingestion.batch_size,
        )
        return loader.run()

    def ingest_directory(
        self,
        root: str,
        max_chunks: Optional[int] = None,
        progress_cb: Optional[Callable[[Mapping[str, object]], None]] = None,
    ) -> "IngestionStats":
        reader = CorpusReader(root)
        effective_progress_cb = progress_cb if self.config.ingestion.progress_enabled else None
        return self.bootstrap_from_corpus(reader, max_chunks=max_chunks, progress_cb=effective_progress_cb)

    def ingest_text(
        self,
        text: str,
        *,
        source: str = "upload",
        max_chars: int = 2048,
        allow_duplicates: Optional[bool] = None,
    ) -> tuple[int, Optional[RuntimeMetrics]]:
        """Feed raw text into the runtime for lightweight training."""

        cleaned = text.strip()
        if not cleaned:
            return 0, None

        # Skip if the full payload has already been ingested.
        doc_digest: Optional[str] = None
        index = self._ingest_index
        allow_dup = (
            self.config.ingestion.allow_reingest_duplicates
            if allow_duplicates is None
            else bool(allow_duplicates)
        )
        if index is not None:
            doc_digest = content_digest(cleaned)
            if index.seen(doc_digest) and not allow_dup:
                summary = f"Skipped duplicate ingestion for {source}"
                self.history.append((source, summary))
                return 0, None

        segments: List[str] = []
        buffer: List[str] = []
        size = 0
        for line in cleaned.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            buffer.append(stripped)
            size += len(stripped) + 1
            if size >= max_chars:
                segments.append("\n".join(buffer))
                buffer.clear()
                size = 0
        if buffer:
            segments.append("\n".join(buffer))

        if len(segments) > 1:
            rng = np.random.default_rng()
            rng.shuffle(segments)

        processed = 0
        last_metrics: Optional[RuntimeMetrics] = None
        for order_idx, segment in enumerate(segments):
            snippet = segment.strip()
            if not snippet:
                continue
            seg_digest: Optional[str] = None
            if index is not None:
                seg_digest = content_digest(snippet)
                if index.seen(seg_digest) and not allow_dup:
                    continue
            if self.config.learning_enabled:
                self.proto_lm.training_step(snippet)
            self._record_memory(source, snippet)
            last_metrics = self.runtime.run_step(self._build_state(snippet, speaker="assistant"))
            processed += 1
            if seg_digest is None:
                seg_digest = content_digest(snippet)
            BUS.publish(
                TelemetryEvent(
                    type="training/sample",
                    payload={
                        "step": self.proto_lm.step,
                        "digest": seg_digest[:12],
                        "source": source,
                        "order": order_idx,
                        "length": len(snippet),
                    },
                )
            )
            if index is not None and seg_digest is not None and not allow_dup:
                index.mark(
                    seg_digest,
                    doc_name=source,
                    length=len(snippet),
                    metadata={
                        "kind": "segment",
                        "source": source,
                    },
                )

        if processed:
            summary = f"Ingested {processed} segments from {source}"
            self.history.append((source, summary))
            if index is not None and doc_digest is not None and not allow_dup:
                index.mark(
                    doc_digest,
                    doc_name=source,
                    length=len(cleaned),
                    metadata={
                        "kind": "document",
                        "segments": processed,
                        "source": source,
                    },
                )

        return processed, last_metrics

    def process_user_input(self, text: str) -> RuntimeMetrics:
        clean_text = text.strip()
        if not clean_text:
            clean_text = "<silence>"
        clean_text = sanitize_text(clean_text)
        self.history.append(("user", clean_text))
        self._schedule_learning(clean_text)
        self._record_memory("user", clean_text)
        state = self._build_state(clean_text, speaker="user")
        metrics = self.runtime.run_step(state)
        self._adaptive.track_runtime(metrics)
        self._maybe_train_on_buffer()
        if (
            self.config.learning_enabled
            and self._learning_last_flush_step != self.proto_lm.step
        ):
            self._maybe_autosave()
        return metrics

    def generate_response(self, prompt: Optional[str] = None) -> ConversationSnapshot:
        prefix = prompt if prompt is not None else self.history[-1][1] if self.history else ""
        response = self.proto_lm.sample(prefix, max_tokens=self.config.response_tokens)
        response = self._postprocess_response(response)
        if looks_like_echo(response, (text for speaker, text in self.history if speaker == "user")):
            response = pick_scaffold(
                self.scaffolds,
                "clarify",
                "I may need a clearer direction. What would you like me to focus on?",
            )
        self.history.append(("assistant", response))
        self._schedule_learning(response)
        self._record_memory("assistant", response)
        metrics = self.runtime.run_step(self._build_state(response, speaker="assistant"))
        self._adaptive.track_runtime(metrics)
        response, gating = self._adaptive.assess_response(response, metrics)
        self._last_gating_summary = gating
        todos = [record.text for record in self.runtime.memory.todos.iter()]
        self._maybe_train_on_buffer()
        if (
            self.config.learning_enabled
            and self._learning_last_flush_step != self.proto_lm.step
        ):
            self._maybe_autosave()
        return ConversationSnapshot(
            metrics=metrics,
            response=response,
            meta_report=metrics.meta_report,
            todos=todos,
            generator_description=f"vocab_size={self.proto_lm.vocab.size()} step={self.proto_lm.step}",
        )

    def save_state(self, path: Optional[str] = None) -> Path:
        target = Path(path or self._auto_save_path or "conversation_state.json").expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path: Optional[str] = None
        if self.proto_lm.config.checkpoint_path:
            checkpoint_path = str(self.proto_lm.save_checkpoint())
        adaptive_state = self._adaptive.export_state()
        payload = {
            "history": list(self.history),
            "lm": asdict(self.config.lm),
            "proto_lm": {
                "step": self.proto_lm.step,
                "vocab": self.proto_lm.vocab.tokens,
                "merges": self.proto_lm.vocab.merges,
                "checkpoint": checkpoint_path,
            },
            "memory": self.runtime.memory.serialize(),
            "adaptive": adaptive_state,
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def load_state(self, path: str | Path) -> None:
        target = Path(path)
        if not target.exists():
            return
        data = json.loads(target.read_text(encoding="utf-8"))
        history = data.get("history", [])
        self.history.clear()
        for speaker, text in history:
            self.history.append((speaker, text))
        lm_config = TrainingConfig(**data.get("lm", {}))
        self.config.lm = lm_config
        self.proto_lm = ProtoLanguageModel(self.config.lm, learning_enabled=self.config.learning_enabled)
        proto_data = data.get("proto_lm", {})
        if proto_data:
            checkpoint = proto_data.get("checkpoint")
            loaded = False
            if checkpoint:
                loaded = self.proto_lm.load_checkpoint(checkpoint)
            if not loaded:
                self.proto_lm.vocab.tokens = proto_data.get("vocab", self.proto_lm.vocab.tokens)
                self.proto_lm.vocab.merges = proto_data.get("merges", self.proto_lm.vocab.merges)
                self.proto_lm.vocab._refresh_index()
                self.proto_lm.step = proto_data.get("step", 0)
        # Recreate adaptive coordinator bindings for the new proto LM instance.
        self._adaptive = AdaptiveCoordinator(runtime=self.runtime, proto_lm=self.proto_lm)
        self.proto_lm.register_external_state(
            exporter=self._adaptive.export_state,
            importer=self._adaptive.import_state,
        )
        memory_snapshot = data.get("memory")
        if memory_snapshot:
            self._restore_memory(memory_snapshot)
        adaptive_payload = data.get("adaptive")
        if isinstance(adaptive_payload, dict):
            self._adaptive.import_state(adaptive_payload)

    def _maybe_autosave(self) -> None:
        if self._auto_save_path:
            self.save_state(self._auto_save_path)

    def _restore_memory(self, snapshot: Mapping[str, object]) -> None:
        self.runtime.memory.reset()
        for entry in snapshot.get("facts", []):
            self.runtime.memory.facts.add(entry.get("text", ""), score=entry.get("score", 0.0))
        for entry in snapshot.get("hypotheses", []):
            self.runtime.memory.hypotheses.add(entry.get("text", ""), score=entry.get("score", 0.0))
        for entry in snapshot.get("todos", []):
            self.runtime.memory.todos.add(entry.get("text", ""), score=entry.get("score", 0.0))

    def _record_memory(self, speaker: str, text: str) -> None:
        verb = {
            "op": "schedule_todo" if speaker == "user" else "add_fact",
            "text": f"{speaker}:{text[:160]}",
            "score": min(len(text) / 80.0, 2.0),
        }
        self.memory_operator.execute(verb)

    def _build_state(self, text: str, speaker: str) -> Mapping[str, object]:
        ids = self.proto_lm.encode(text)
        logits = self._logits_from_ids(ids)
        token_cost = max(1.0, float(len(ids)))
        prompt_embedding = self._compute_embedding(text)
        return {
            "sequence_id": 900 if speaker == "assistant" else 800,
            "prediction": {
                "token_logits": logits,
                "steps_remaining": max(0.0, 10.0 - token_cost / 4.0),
                "entropy_estimate": float(np.std(logits)),
            },
            "context": {
                "tokens": text.split(),
                "text": text,
            },
            "embeddings": {
                "prompt": prompt_embedding,
            },
            "decision_proposal": {
                "operator": "SASS_WITH_JUMP" if speaker == "assistant" else "MEMORY_OP",
                "cot_depth": 3 if speaker == "assistant" else 1,
            },
            "teleport_trigger": speaker == "assistant",
            "reasoner_trigger": len(text.split()) > 6,
            "token_cost": token_cost,
            "memory_verb": {
                "op": "schedule_todo",
                "text": f"trace::{speaker}:{text[:120]}",
            },
            "contradictions": 0.0,
        }

    def _logits_from_ids(self, ids: Sequence[int]) -> np.ndarray:
        vocab_size = self.proto_lm.vocab.size()
        logits = np.full(vocab_size, -4.0, dtype=np.float64)
        if not ids:
            return logits[np.newaxis, :]
        counts = Counter(ids)
        for token_id, count in counts.items():
            if 0 <= token_id < vocab_size:
                logits[token_id] = np.log(count + 1.0)
        logits -= logits.max()
        return logits[np.newaxis, :]

    def _postprocess_response(self, text: str) -> str:
        cleaned = sanitize_text(text)
        return cleaned if cleaned else "(no response yet)"

    def _compute_embedding(self, text: str) -> np.ndarray:
        tokens = text.lower().split()
        dim = 128
        vec = np.zeros(dim, dtype=np.float32)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            for i in range(0, len(digest), 2):
                idx = int.from_bytes(digest[i : i + 2], "little") % dim
                vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0.0:
            vec /= norm
        return vec


@dataclass
class IngestionStats:
    files_processed: int = 0
    chunks_processed: int = 0
    chunks_accepted: int = 0
    chunks_rejected: int = 0

    def as_mapping(self) -> Mapping[str, int]:
        return {
            "files_processed": self.files_processed,
            "chunks_processed": self.chunks_processed,
            "chunks_accepted": self.chunks_accepted,
            "chunks_rejected": self.chunks_rejected,
        }


class IngestionLoop:
    """Coordinate salience-filtered ingestion with progress tracking."""

    def __init__(
        self,
        session: ConversationSession,
        reader: CorpusReader,
        chunk_size: int,
        checkpoint_interval: int,
        max_chunks: Optional[int],
        progress_cb: Optional[Callable[[Mapping[str, object]], None]],
        chunk_overlap: int,
        batch_size: int,
    ) -> None:
        self.session = session
        self.reader = reader
        self.chunk_size = chunk_size
        self.checkpoint_interval = checkpoint_interval
        self.max_chunks = max_chunks
        self.progress_cb = progress_cb
        self.stats = IngestionStats()
        self.chunk_overlap = max(0, chunk_overlap)
        self.batch_size = max(1, batch_size)
        self.work_estimate = reader.estimate_work(chunk_size=chunk_size, chunk_overlap=self.chunk_overlap)
        self.start_time: Optional[float] = None
        self.adaptive_cfg = session.config.ingestion.adaptive
        self._adaptive_history: Deque[bool] = deque(maxlen=max(1, self.adaptive_cfg.evaluation_window))
        self._chunks_seen = 0
        self._cooldown = 0
        self._warmup_done = self.adaptive_cfg.warmup_chunks <= 0
        self._filter_enabled = session.is_filter_enabled()
        if self.adaptive_cfg.enabled and not self._warmup_done:
            session._set_filter_enabled(False)
            self._filter_enabled = False

    def run(self) -> IngestionStats:
        self.start_time = time.time()
        meta_snapshot = self.session.runtime.meta_state.snapshot()
        memory_snapshot = self.session.runtime.memory.as_runtime_mapping()
        exhausted = False
        for batch in self.reader.stream(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            batch_size=self.batch_size,
        ):
            for metadata, chunk in batch:
                if self.max_chunks is not None and self.stats.chunks_processed >= self.max_chunks:
                    exhausted = True
                    break
                self.stats.chunks_processed += 1
                accepts, readings = self._evaluate_chunk(chunk, meta_snapshot, memory_snapshot)
                if accepts:
                    if self.session.config.learning_enabled:
                        self.session.proto_lm.training_step(chunk)
                    self.stats.chunks_accepted += 1
                    if self.stats.chunks_accepted % self.checkpoint_interval == 0:
                        self.session._maybe_autosave()
                else:
                    self.stats.chunks_rejected += 1
                if self.progress_cb:
                    payload = self._progress_payload(metadata, accepts, readings)
                    self.progress_cb(payload)
                if self.session._auto_save_path and self.stats.chunks_accepted % self.checkpoint_interval == 0:
                    self.session._maybe_autosave()
                self._adaptive_update(accepts, readings)
                self.stats.files_processed = max(self.stats.files_processed, metadata["file_index"])
                self._emit_ingestion_event(metadata, accepts, readings)
            if exhausted:
                break
        self.session._maybe_autosave()
        return self.stats

    def _progress_payload(
        self,
        metadata: Mapping[str, object],
        accepts: bool,
        readings: Mapping[str, float],
    ) -> Mapping[str, object]:
        return {
            "metadata": metadata,
            "stats": self.stats.as_mapping(),
            "estimate": {
                "files": self.work_estimate.total_files,
                "chunks": self.work_estimate.total_chunks,
                "eta": self._eta_safe_estimate(
                    self.stats.chunks_processed,
                    self.work_estimate.total_chunks,
                ),
            },
            "accepted": accepts,
            "readings": readings,
            "filter_enabled": self._filter_enabled,
        }

    def _emit_ingestion_event(
        self,
        metadata: Mapping[str, object],
        accepts: bool,
        readings: Mapping[str, float],
    ) -> None:
        BUS.publish(
            TelemetryEvent(
                type="ingestion/chunk",
                payload={
                    "path": metadata.get("path"),
                    "file_index": metadata.get("file_index"),
                    "chunk_index": metadata.get("chunk_index"),
                    "chunk_total": metadata.get("chunk_total"),
                    "accepted": accepts,
                    "stats": self.stats.as_mapping(),
                    "filter_enabled": self._filter_enabled,
                    "readings": readings,
                },
            )
        )

    def _evaluate_chunk(
        self,
        chunk: str,
        meta_snapshot: Mapping[str, float],
        memory_snapshot: Mapping[str, object],
    ) -> Tuple[bool, Mapping[str, float]]:
        state = {
            "prediction": {
                "token_logits": self._naive_logits(chunk),
                "entropy_estimate": float(len(chunk)) / max(1, self.chunk_size),
                "steps_remaining": 1.0,
            },
            "context": {
                "tokens": chunk.split(),
                "text": chunk,
            },
            "decision_proposal": {"operator": "SASS", "cot_depth": 1},
        }
        return self.session._filter.evaluate(state, memory_snapshot, meta_snapshot)

    def _naive_logits(self, chunk: str) -> List[List[float]]:
        length = max(4, len(chunk) % 16 + 4)
        return [[0.1] * length]

    def _adaptive_update(self, accepts: bool, readings: Mapping[str, float]) -> None:
        if not self.adaptive_cfg.enabled:
            return
        self._chunks_seen += 1
        if self._filter_enabled:
            self._adaptive_history.append(accepts)
        else:
            self._adaptive_history.append(self._preview_accept(readings))
        if not self._warmup_done:
            if self._chunks_seen >= self.adaptive_cfg.warmup_chunks:
                self._warmup_done = True
                self._maybe_enable_filter()
            return
        if self._cooldown > 0:
            self._cooldown -= 1
            return
        if self._filter_enabled:
            ratio = self._history_ratio()
            if ratio >= self.adaptive_cfg.disable_ratio:
                self._set_filter(False)
        else:
            ratio = self._history_ratio()
            if ratio <= self.adaptive_cfg.enable_ratio:
                self._set_filter(True)

    def _maybe_enable_filter(self) -> None:
        ratio = self._history_ratio(default=1.0)
        if ratio <= self.adaptive_cfg.enable_ratio:
            self._set_filter(True)

    def _history_ratio(self, default: float = 1.0) -> float:
        if not self._adaptive_history:
            return default
        return sum(1 for flag in self._adaptive_history if flag) / len(self._adaptive_history)

    def _set_filter(self, enabled: bool) -> None:
        if enabled == self._filter_enabled:
            return
        self.session._set_filter_enabled(enabled)
        self._filter_enabled = enabled
        self._adaptive_history.clear()
        self._cooldown = self.adaptive_cfg.cooldown_chunks

    def _preview_accept(self, readings: Mapping[str, float]) -> bool:
        t = self.session.config.ingestion.thresholds
        if not t.enabled:
            uncertainty = float(readings.get("uncertainty", 0.0))
            novelty = float(readings.get("novelty", 0.0))
            drag = float(readings.get("drag", 0.0))
            return (
                uncertainty >= t.min_uncertainty
                and novelty >= t.min_novelty
                and drag <= t.max_drag
            )
        return True

    def _eta_safe_estimate(self, processed: int, total: int) -> str:
        if total == 0 or processed == 0 or self.start_time is None:
            return "unknown"
        elapsed = time.time() - self.start_time
        eta = elapsed * (total - processed) / processed
        if eta < 0:
            eta = 0
        return time.strftime("%H:%M:%S", time.gmtime(int(eta)))
