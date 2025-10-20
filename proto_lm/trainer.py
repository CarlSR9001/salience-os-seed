"""Proto language model trainer leveraging SalienceRuntime."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

from ..core.operators.sass import SASSConfig, SASSCore
from ..runtime.health.grads import grad_health
from ..telemetry import BUS, ParameterEvent, TelemetryEvent
from .vocab import Vocabulary
from .modules import ModuleBlueprint, ModuleManager


@dataclass
class TrainingConfig:
    """Hyperparameters governing the proto language model."""

    vocab_merges: int = 128
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    sequence_length: int = 128
    embed_dim: int = 128
    seed: int = 13
    grad_clip: float = 1.0
    checkpoint_path: Optional[str] = "storage/proto_lm/proto_lm.pt"
    device: str = "cpu"  # "cpu", "cuda", or "auto"
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
    module_configs: Tuple[Dict[str, object], ...] = (
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


class ProtoLanguageModel(nn.Module):
    """Lightweight autoregressive learner updated online."""

    def __init__(self, config: TrainingConfig | None = None, *, learning_enabled: bool = True) -> None:
        super().__init__()
        self.config = config or TrainingConfig()
        torch.manual_seed(self.config.seed)
        self.learning_enabled = learning_enabled
        self.device = self._resolve_device(self.config.device)

        self.vocab = Vocabulary()
        sass_cfg = SASSConfig(
            d_model=self.config.embed_dim,
            state_channels=self.config.embed_dim * 2,
            num_layers=4,
            dropout=0.05,
        )
        self.core = SASSCore(sass_cfg)
        self.embed = nn.Embedding(self.vocab.size(), self.config.embed_dim)
        self.output = nn.Linear(self.config.embed_dim, self.vocab.size())
        self.module_manager = ModuleManager(
            embed_dim=self.config.embed_dim,
            blueprints=self._build_module_blueprints(self.config.module_configs),
            device=self.device,
        )
        self._tie_weights_if_possible()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer: torch.optim.Optimizer
        self.lr_scheduler: Optional[ReduceLROnPlateau]
        self._reset_optimizer()
        self.step: int = 0
        self._parameter_total: int = self._count_parameters()
        self.to(self.device)
        self._embed_grad_importance: torch.Tensor = torch.zeros(self.vocab.size(), device=self.device)
        self.token_usage: Counter[str] = Counter()
        self._dedupe_events: int = 0
        self._latest_grad_health: Dict[str, float] = {"frac_nonzero": 0.0, "grad_norm": 0.0}
        self._latest_loss: float = 0.0
        self._last_growth_step: int = -1
        self._growth_events: List[Dict[str, object]] = []
        self._pending_lr_restore: Dict[int, Dict[str, float]] = {}
        self._ema_state: Dict[str, torch.Tensor] = {}
        self._token_birth_step: Dict[int, int] = {idx: 0 for idx in range(self.vocab.size())}
        self._latest_new_token_fraction: float = 0.0
        self._latest_loss_components: Dict[str, float] = {"old": 0.0, "new": 0.0}
        self._last_loss_was_nan: bool = False
        self.eos_token: str = "\n\n"
        self.eos_token_id: Optional[int] = self.vocab.token_to_id.get(self.eos_token)
        self.stop_sequences: tuple[str, ...] = ("\n\n",)
        self._training_observers: List[Callable[[Dict[str, object]], None]] = []
        self._external_state_exporter: Optional[Callable[[], Dict[str, object]]] = None
        self._external_state_importer: Optional[Callable[[Dict[str, object]], None]] = None

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _resolve_device(self, requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def set_learning_enabled(self, enabled: bool) -> None:
        self.learning_enabled = enabled

    def _build_module_blueprints(
        self,
        configs: Sequence[Mapping[str, object]],
    ) -> List[ModuleBlueprint]:
        blueprints: List[ModuleBlueprint] = []
        for raw in configs:
            if not isinstance(raw, Mapping):
                continue
            name = raw.get("name")
            if not isinstance(name, str) or not name:
                continue
            position = raw.get("position", "pre_core")
            if not isinstance(position, str):
                position = "pre_core"
            cfg = raw.get("config", {})
            cfg_dict = dict(cfg) if isinstance(cfg, Mapping) else {}
            blueprints.append(ModuleBlueprint(name=name, position=position, config=cfg_dict))
        return blueprints

    # ------------------------------------------------------------------
    # Vocabulary management
    # ------------------------------------------------------------------
    def build_vocab(self, stats: Mapping[str, object]) -> None:
        if isinstance(stats, Vocabulary):
            self.vocab = stats
        elif hasattr(stats, "top_merges"):
            self.vocab.build_from_statistics(stats, merges=self.config.vocab_merges)
        self._ensure_capacity(self.vocab.size())

    def encode(self, text: str, *, mutate: bool = True) -> List[int]:
        if not mutate:
            try:
                ids = self.vocab.encode_ids_readonly(text)
            except KeyError:
                fallback_id = self.vocab.token_to_id.get("?")
                if fallback_id is None:
                    raise

                ids = []
                for token in self.vocab.encode(text):
                    token_id = self.vocab.token_to_id.get(token)
                    if token_id is not None:
                        ids.append(token_id)
                        continue

                    if len(token) > 1:
                        for symbol in token:
                            symbol_id = self.vocab.token_to_id.get(symbol)
                            if symbol_id is not None:
                                ids.append(symbol_id)
                            else:
                                ids.append(fallback_id)
                        continue

                    ids.append(fallback_id)
            if self.eos_token_id is None and self.eos_token in self.vocab.token_to_id:
                self.eos_token_id = self.vocab.token_to_id[self.eos_token]
            return ids

        prev_size = self.vocab.size()
        ids = self.vocab.encode_ids(text)
        new_size = self.vocab.size()
        if new_size > prev_size:
            for token_id in range(prev_size, new_size):
                self._token_birth_step[token_id] = self.step
            self._ensure_capacity(new_size)
        self._ensure_capacity(self.vocab.size())
        self._register_token_usage(ids)
        if self.eos_token_id is None and self.eos_token in self.vocab.token_to_id:
            self.eos_token_id = self.vocab.token_to_id[self.eos_token]
        return ids

    def _ensure_capacity(self, size: int) -> None:
        current = self.embed.num_embeddings
        if size <= current:
            return
        chunk = max(1, self.config.vocab_growth_chunk)
        reserve = max(0, self.config.vocab_growth_headroom)
        target = size + reserve
        if target % chunk != 0:
            target = ((target // chunk) + 1) * chunk
        new_embed = nn.Embedding(target, self.config.embed_dim).to(self.device)
        new_embed.weight.data[:current] = self.embed.weight.data
        tail = new_embed.weight.data[current:]
        nn.init.normal_(tail, mean=0.0, std=0.02)

        new_output = nn.Linear(self.config.embed_dim, target).to(self.device)
        new_output.weight.data[: self.output.out_features] = self.output.weight.data
        new_output.bias.data[: self.output.out_features] = self.output.bias.data
        nn.init.zeros_(new_output.weight.data[self.output.out_features :])
        nn.init.zeros_(new_output.bias.data[self.output.out_features :])

        self.embed = new_embed
        self.output = new_output
        self._tie_weights_if_possible()
        self.loss_fn = nn.CrossEntropyLoss()
        self._reset_optimizer()
        self._resize_importance_tracker(target)
        self.module_manager.on_vocab_expand(new_size=target, delta=target - current)
        init_std = float(tail.std().detach().cpu().item()) if tail.numel() else 0.0
        growth = {
            "step": self.step,
            "old_capacity": current,
            "new_capacity": target,
            "delta_capacity": target - current,
            "active_vocab": size,
            "init_std": init_std,
        }
        self._growth_events.append(growth)
        self._growth_events = self._growth_events[-64:]
        self._last_growth_step = self.step
        self._emit_parameter_update()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        prefix: str,
        *,
        max_tokens: int = 32,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stop_sequences: Optional[Sequence[str]] = None,
    ) -> str:
        was_training = self.training
        self.eval()

        prefix_ids = self.encode(prefix, mutate=False)
        if not prefix_ids:
            prefix_ids = [0]
        generated = torch.tensor(prefix_ids, device=self.device, dtype=torch.long)

        stop_sequences = tuple(stop_sequences) if stop_sequences is not None else self.stop_sequences

        for _ in range(max_tokens):
            context = generated[-self.config.sequence_length :]
            logits = self._forward_logits(context.unsqueeze(0))
            next_id = self._sample_next_token(
                logits[0, -1, :],
                generated,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            next_tensor = torch.tensor([next_id], device=self.device, dtype=torch.long)
            generated = torch.cat([generated, next_tensor], dim=0)
            if self._should_stop(generated, next_id, stop_sequences):
                break

        text = self.vocab.decode_ids(generated.tolist())
        if was_training:
            self.train()
        return text

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> int:
        logits = logits.clone().to(dtype=torch.float32)
        temperature = max(float(temperature), 1e-6)
        logits = logits / temperature

        if repetition_penalty and repetition_penalty != 1.0 and generated.numel() > 0:
            unique_ids = torch.unique(generated)
            original = logits[unique_ids]
            adjusted = torch.where(
                original < 0,
                original * repetition_penalty,
                original / repetition_penalty,
            )
            logits[unique_ids] = adjusted

        vocab_size = logits.size(0)
        if top_k and top_k > 0 and top_k < vocab_size:
            values, indices = torch.topk(logits, min(top_k, vocab_size))
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(0, indices, values)
            logits = filtered

        if top_p and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative > top_p
            if torch.any(cutoff):
                first_idx = torch.nonzero(cutoff, as_tuple=False)[0].item()
                if first_idx + 1 < sorted_logits.size(0):
                    sorted_logits[first_idx + 1 :] = float("-inf")
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(0, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        if torch.isnan(probs).any() or torch.sum(probs).item() <= 0:
            return int(torch.argmax(logits).item())
        return int(torch.multinomial(probs, num_samples=1).item())

    def _should_stop(
        self,
        generated: torch.Tensor,
        next_id: int,
        stop_sequences: Sequence[str] | None,
    ) -> bool:
        if self.eos_token_id is not None and next_id == self.eos_token_id:
            return True
        if not stop_sequences:
            return False
        text = self.vocab.decode_ids(generated.tolist())
        return any(text.endswith(seq) for seq in stop_sequences)

    # ------------------------------------------------------------------
    # Forward utilities
    # ------------------------------------------------------------------
    def _forward_with_state(
        self,
        token_ids: torch.Tensor,
        *,
        layer_states: Optional[Sequence[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        elif token_ids.dim() != 2:
            raise ValueError("token_ids must have shape (batch, seq)")
        embedded = self.embed(token_ids)
        embedded = self.module_manager.apply_pre(embedded)
        hidden, new_states = self.core(embedded, layer_states=layer_states)
        hidden = self.module_manager.apply_post(hidden)
        trimmed_states = [state[:, -1:, :].detach().contiguous() for state in new_states]
        logits = self.output(hidden)
        return logits, trimmed_states

    def _forward_logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self._forward_with_state(token_ids)
        return logits

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, text: str) -> float:
        if not self.learning_enabled:
            return 0.0
        ids = self.encode(text)
        if len(ids) < 2:
            return 0.0

        token_tensor = torch.tensor(ids, dtype=torch.long, device=self.device)
        inputs = token_tensor[:-1].unsqueeze(0)
        targets = token_tensor[1:].unsqueeze(0)

        self.train()
        self._update_learning_rates()
        logits = self._forward_logits(inputs)
        if not torch.isfinite(logits).all():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        loss, loss_old, loss_new = self._compute_age_weighted_loss(logits, targets)
        if loss is None:
            return 0.0
        loss_components = {"old": float(loss_old), "new": float(loss_new)}
        if not torch.isfinite(loss):
            fallback_loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss = fallback_loss
            loss_components["fallback"] = float(fallback_loss.detach().cpu().item())
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        healthy, grad_stats = grad_health(
            self,
            min_frac_nonzero=0.2,
            min_norm=1e-6,
        )
        self._latest_grad_health = grad_stats
        loss_value = float(loss.detach().cpu().item())
        self._latest_loss = loss_value
        if not healthy:
            grad_norm = grad_stats.get("grad_norm")
            if math.isnan(grad_norm):
                # Treat NaN gradients as recoverable by skipping the update but not crashing.
                self.optimizer.zero_grad(set_to_none=True)
                self._last_loss_was_nan = True
                self._latest_loss_components = loss_components
                return loss_value
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(10.0)
            self.optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(f"Dead/vanishing grads detected: {grad_stats}")
        self._update_gradient_importance()
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self._apply_lr_cooldown()
        self._update_learning_rates()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(loss_value)
        self._update_ema()
        self._maybe_dedupe_tokens()
        self.step += 1
        self._latest_loss_components = loss_components
        self._last_loss_was_nan = False
        snapshot = self._collect_training_snapshot(loss_value)
        self._emit_training_step(loss=loss_value, snapshot=snapshot)
        self.module_manager.on_training_step(loss=loss_value, snapshot=snapshot)
        self._notify_training_observers(snapshot)
        return loss_value

    def _reset_optimizer(self) -> None:
        lr = self.config.learning_rate
        embed_param = self.embed.weight
        seen: set[int] = set()
        other_params: List[nn.Parameter] = []
        for _, param in self.named_parameters():
            if param is embed_param:
                continue
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            other_params.append(param)

        param_groups = []
        if other_params:
            param_groups.append({"params": other_params, "lr": lr})
        param_groups.append({"params": [embed_param], "lr": lr})

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=self.config.weight_decay,
        )
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            threshold=self.config.scheduler_threshold,
        )
        if not hasattr(self, "_pending_lr_restore"):
            self._pending_lr_restore = {}
        else:
            self._pending_lr_restore.clear()
        if self.config.vocab_lr_multiplier < 1.0 and self.optimizer.param_groups:
            embed_idx = len(self.optimizer.param_groups) - 1
            embed_group = self.optimizer.param_groups[embed_idx]
            embed_group.setdefault("_lr_multiplier", 1.0)
            embed_group.setdefault("_cooldown_steps", 0)
            multiplier = float(self.config.vocab_lr_multiplier)
            cooldown = max(0, self.config.vocab_lr_cooldown_steps)
            if multiplier < 1.0 and cooldown > 0:
                embed_group["_lr_multiplier"] = multiplier
                embed_group["_cooldown_steps"] = cooldown
                self._pending_lr_restore[embed_idx] = {"steps": cooldown}
            else:
                embed_group["_lr_multiplier"] = 1.0
        for group in self.optimizer.param_groups:
            group.setdefault("_lr_multiplier", 1.0)
            group.setdefault("_cooldown_steps", 0)

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def save_checkpoint(self, path: Optional[str] = None) -> Optional[Path]:
        target = Path(path or self.config.checkpoint_path) if self.config.checkpoint_path or path else None
        if not target:
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "vocab": {
                "tokens": self.vocab.tokens,
                "merges": self.vocab.merges,
            },
        }
        if self.lr_scheduler is not None:
            payload["scheduler"] = self.lr_scheduler.state_dict()
        if self._external_state_exporter is not None:
            try:
                external_state = self._external_state_exporter()
            except Exception:
                external_state = None
            if external_state is not None:
                payload["external_state"] = external_state
        torch.save(payload, target)
        return target

    def load_checkpoint(self, path: Optional[str] = None) -> bool:
        target = Path(path or self.config.checkpoint_path) if self.config.checkpoint_path or path else None
        if not target or not target.exists():
            return False
        payload = torch.load(target, map_location=self.device)
        self.vocab.tokens = payload["vocab"]["tokens"]
        self.vocab.merges = payload["vocab"]["merges"]
        self.vocab._refresh_index()
        model_state = dict(payload["model"])
        saved_embed = model_state.get("embed.weight")
        saved_vocab = saved_embed.shape[0] if saved_embed is not None else self.vocab.size()
        self._ensure_capacity(max(self.vocab.size(), saved_vocab))
        current_state = self.state_dict()

        def _resize_tensor(saved: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            if saved.shape == target.shape:
                return saved.to(dtype=target.dtype, device=target.device)
            if saved.dim() == 0 and target.dim() == 0:
                return saved.to(dtype=target.dtype, device=target.device)
            resized = target.clone()
            slices = tuple(slice(0, min(saved.size(dim), target.size(dim))) for dim in range(saved.dim()))
            resized[slices] = saved[slices].to(dtype=target.dtype, device=target.device)
            return resized

        for key, saved_tensor in list(model_state.items()):
            target_tensor = current_state.get(key)
            if target_tensor is None:
                continue
            model_state[key] = _resize_tensor(saved_tensor, target_tensor)

        self.load_state_dict(model_state, strict=False)
        self._tie_weights_if_possible()
        self._reset_optimizer()
        optimizer_state = payload["optimizer"]
        try:
            self.optimizer.load_state_dict(optimizer_state)
        except (ValueError, RuntimeError):
            # Optimizer shape drift (e.g., new modules) – fall back to fresh state.
            self._reset_optimizer()
        for param, state in self.optimizer.state.items():
            if not isinstance(state, dict):
                continue
            for key, value in list(state.items()):
                if not isinstance(value, torch.Tensor):
                    continue
                if value.dim() == 0:
                    continue
                if value.shape != param.data.shape:
                    state[key] = _resize_tensor(value, param.data)
        scheduler_state = payload.get("scheduler")
        if scheduler_state and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(scheduler_state)
        self.step = int(payload.get("step", 0))
        self._parameter_total = self._count_parameters()
        self.to(self.device)
        self._resize_importance_tracker(self.vocab.size())
        external_state = payload.get("external_state")
        if self._external_state_importer is not None and isinstance(external_state, dict):
            try:
                self._external_state_importer(external_state)
            except Exception:
                pass
        return True

    # ------------------------------------------------------------------
    # Adaptive observation hooks
    # ------------------------------------------------------------------
    def add_training_observer(self, callback: Callable[[Dict[str, object]], None]) -> None:
        if callback not in self._training_observers:
            self._training_observers.append(callback)

    def remove_training_observer(self, callback: Callable[[Dict[str, object]], None]) -> None:
        if callback in self._training_observers:
            self._training_observers.remove(callback)

    def _notify_training_observers(self, snapshot: Dict[str, object]) -> None:
        if not self._training_observers:
            return
        for observer in list(self._training_observers):
            try:
                observer(snapshot)
            except Exception:
                continue

    def observe_salience(
        self,
        salience: Mapping[str, float],
        *,
        gating_decision: Optional[str] = None,
        metrics: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.module_manager.on_salience_update(
            salience=salience,
            gating_decision=gating_decision,
            metrics=metrics or {},
        )

    def _collect_training_snapshot(self, loss_value: float) -> Dict[str, object]:
        total_norm = 0.0
        max_norm = 0.0
        parameter_samples: Dict[str, float] = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            norm = float(param.data.detach().norm().cpu().item())
            total_norm += norm
            max_norm = max(max_norm, norm)
            if len(parameter_samples) < 12:
                parameter_samples[name] = norm
        gradient_health = dict(self._latest_grad_health)
        loss_components = dict(self._latest_loss_components)
        snapshot: Dict[str, object] = {
            "step": self.step,
            "loss": loss_value,
            "loss_components": loss_components,
            "grad_health": gradient_health,
            "parameter_norm_total": total_norm,
            "parameter_norm_max": max_norm,
            "parameter_samples": parameter_samples,
        }
        return snapshot

    # ------------------------------------------------------------------
    # External state persistence
    # ------------------------------------------------------------------
    def register_external_state(
        self,
        *,
        exporter: Callable[[], Dict[str, object]],
        importer: Callable[[Dict[str, object]], None],
    ) -> None:
        self._external_state_exporter = exporter
        self._external_state_importer = importer

    # ------------------------------------------------------------------
    # Parameter exposure for optimizer recreation
    # ------------------------------------------------------------------
    def parameters(self, recurse: bool = True):  # type: ignore[override]
        return super().parameters(recurse=recurse)

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------
    def _count_parameters(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def _emit_parameter_update(self) -> None:
        current = self._count_parameters()
        delta = current - self._parameter_total
        self._parameter_total = current
        BUS.publish(
            ParameterEvent(
                payload={
                    "step": self.step,
                    "total": current,
                    "delta": delta,
                    "model": "proto_lm",
                }
            )
        )

    def _emit_training_step(self, *, loss: float, snapshot: Optional[Dict[str, object]] = None) -> None:
        grads: List[Tuple[str, float]] = []
        for name, param in self.named_parameters():
            if param.grad is not None:
                grads.append((name, float(param.grad.detach().norm().cpu().item())))
        payload = {
            "loss": loss,
            "parameter_total": self._parameter_total,
            "grad_norms": grads,
            "step": self.step,
        }
        BUS.publish(TelemetryEvent(type="training/step", payload=payload))
        ParameterEventPayload = {
            "delta": grads[0][1] if grads else 0.0,
            "total": self._parameter_total,
            "step": self.step,
        }
        BUS.publish(ParameterEvent(payload=ParameterEventPayload))
        if snapshot is not None:
            BUS.publish(
                TelemetryEvent(
                    type="training/snapshot",
                    payload={
                        "step": self.step,
                        "snapshot": snapshot,
                        "growth_events": list(self._growth_events[-4:]),
                        "loss_components": dict(self._latest_loss_components),
                        "new_token_fraction": self._latest_new_token_fraction,
                    },
                )
            )

    # ------------------------------------------------------------------
    # Salience-aware dedupe helpers
    # ------------------------------------------------------------------
    def _resize_importance_tracker(self, size: int) -> None:
        if self._embed_grad_importance.numel() == size:
            return
        device = self._embed_grad_importance.device
        if self._embed_grad_importance.numel() == 0:
            self._embed_grad_importance = torch.zeros(size, device=device)
            return
        old = self._embed_grad_importance
        if size > old.numel():
            pad = torch.zeros(size - old.numel(), device=device)
            self._embed_grad_importance = torch.cat([old, pad], dim=0)
        else:
            self._embed_grad_importance = old[:size]

    def _register_token_usage(self, ids: Iterable[int]) -> None:
        tokens = self.vocab.tokens
        for idx in ids:
            if 0 <= idx < len(tokens):
                self.token_usage[tokens[idx]] += 1

    def _update_gradient_importance(self) -> None:
        if not self.config.dedupe_enabled:
            return
        if self.embed.weight.grad is None:
            return
        decay = float(self.config.dedupe_importance_decay)
        grad_norms = self.embed.weight.grad.detach().norm(dim=1)
        self._resize_importance_tracker(grad_norms.numel())
        self._embed_grad_importance.mul_(decay).add_((1.0 - decay) * grad_norms)

    def _maybe_dedupe_tokens(self) -> None:
        if not self.config.dedupe_enabled:
            return
        if self.step == 0:
            return
        if self.config.dedupe_interval <= 0:
            return
        if self.step % self.config.dedupe_interval != 0:
            return
        vocab_size = self.vocab.size()
        if vocab_size <= max(self.config.dedupe_min_vocab_size, self.config.dedupe_reserved_tokens):
            return
        grad = self._embed_grad_importance.detach().cpu()
        embed_norm = self.embed.weight.detach().norm(dim=1).cpu()
        if grad.numel() < vocab_size or embed_norm.numel() < vocab_size:
            vocab_size = min(vocab_size, grad.numel(), embed_norm.numel())
        max_usage = max(self.token_usage.values(), default=1)
        usage = torch.tensor([self.token_usage.get(tok, 0) for tok in self.vocab.tokens[:vocab_size]], dtype=torch.float32)
        grad = grad[:vocab_size]
        embed_norm = embed_norm[:vocab_size]
        usage_norm = usage / max(float(max_usage), 1.0)
        grad_norm = grad / max(float(grad.max().item() if grad.numel() else 1.0), 1.0)
        embed_norm = embed_norm / max(float(embed_norm.max().item() if embed_norm.numel() else 1.0), 1.0)

        score = (
            self.config.dedupe_grad_weight * grad_norm
            + self.config.dedupe_usage_weight * usage_norm
            + self.config.dedupe_norm_weight * embed_norm
        )
        threshold = float(self.config.dedupe_score_threshold)
        reserved = int(self.config.dedupe_reserved_tokens)
        candidates: List[int] = []
        for idx, value in enumerate(score.tolist()):
            if idx < reserved:
                continue
            token = self.vocab.tokens[idx]
            if len(token) <= 1:
                continue
            if value >= threshold:
                continue
            candidates.append(idx)
        if not candidates:
            return
        candidates.sort(key=lambda idx: score[idx])
        limit = max(1, min(len(candidates), int(self.config.dedupe_max_prune_per_pass)))
        prune_ids = candidates[:limit]
        self._prune_tokens(prune_ids)

    def _prune_tokens(self, prune_ids: Sequence[int]) -> None:
        if not prune_ids:
            return
        current = self.vocab.size()
        keep_mask = torch.ones(current, dtype=torch.bool, device=self.device)
        keep_mask[list(prune_ids)] = False
        new_embed = self.embed.weight.data[keep_mask, :].clone()
        new_output_weight = self.output.weight.data[keep_mask, :].clone()
        new_output_bias = self.output.bias.data[keep_mask].clone()

        self.embed = nn.Embedding(new_embed.size(0), new_embed.size(1)).to(self.device)
        self.embed.weight.data.copy_(new_embed)
        self.output = nn.Linear(self.config.embed_dim, new_embed.size(0)).to(self.device)
        self.output.weight.data.copy_(new_output_weight)
        self.output.bias.data.copy_(new_output_bias)

        removed_tokens = {self.vocab.tokens[idx] for idx in prune_ids if idx < len(self.vocab.tokens)}
        self.vocab.drop_ids(prune_ids)
        for token in removed_tokens:
            self.token_usage.pop(token, None)

        self._resize_importance_tracker(self.vocab.size())
        self._embed_grad_importance = self._embed_grad_importance.to(self.device)
        self._embed_grad_importance = self._embed_grad_importance[keep_mask.cpu()].to(self.device)
        self._reset_optimizer()
        self._parameter_total = self._count_parameters()
        self._dedupe_events += len(prune_ids)
        self._emit_parameter_update()

    def _tie_weights_if_possible(self) -> None:
        try:
            if self.output.weight.data_ptr() != self.embed.weight.data_ptr():
                self.output.weight = self.embed.weight
        except Exception:
            return

    def vocab_is_weight_tied(self) -> bool:
        return self.output.weight.data_ptr() == self.embed.weight.data_ptr()

    def _apply_lr_cooldown(self) -> None:
        if not self._pending_lr_restore:
            return
        to_remove: List[int] = []
        for idx, meta in list(self._pending_lr_restore.items()):
            steps_left = int(meta.get("steps", 0)) - 1
            group = self.optimizer.param_groups[idx]
            group["_cooldown_steps"] = max(0, steps_left)
            meta["steps"] = steps_left
            if steps_left <= 0:
                group["_lr_multiplier"] = 1.0
                to_remove.append(idx)
        for idx in to_remove:
            self._pending_lr_restore.pop(idx, None)
        if to_remove:
            self._update_learning_rates()

    def _update_learning_rates(self) -> None:
        base_lr = self._scheduled_base_lr(self.step)
        for idx, group in enumerate(self.optimizer.param_groups):
            multiplier = float(group.get("_lr_multiplier", 1.0))
            group["lr"] = base_lr * multiplier
            group["base_lr"] = base_lr

    def _scheduled_base_lr(self, step: int) -> float:
        warmup = max(1, self.config.warmup_steps)
        base = self.config.learning_rate
        if step < warmup:
            return base * float(step + 1) / warmup
        total = max(warmup + 1, self.config.total_steps_estimate)
        progress = min(1.0, float(step - warmup) / max(1, total - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.lr_min + (base - self.config.lr_min) * cosine

    def _compute_age_weighted_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[Optional[torch.Tensor], float, float]:
        vocab_size = logits.size(-1)
        flat_logits = logits.reshape(-1, vocab_size)
        flat_targets = targets.reshape(-1)
        if flat_targets.numel() == 0:
            return None, 0.0, 0.0
        log_probs = F.log_softmax(flat_logits, dim=-1)
        ages = []
        for token_id in flat_targets.cpu().tolist():
            birth = self._token_birth_step.get(int(token_id), 0)
            ages.append(max(0, self.step - birth))
        age_tensor = torch.tensor(ages, dtype=torch.float32, device=self.device)
        ramp = max(1, self.config.new_token_ramp_steps)
        mask_new = age_tensor < ramp
        self._latest_new_token_fraction = float(mask_new.float().mean().item()) if age_tensor.numel() else 0.0

        loss_old = torch.tensor(0.0, device=self.device)
        loss_new = torch.tensor(0.0, device=self.device)

        if (~mask_new).any():
            idx_old = torch.nonzero(~mask_new, as_tuple=False).squeeze(1)
            if idx_old.numel() > 0:
                loss_old = -log_probs[idx_old, flat_targets[idx_old]].mean()

        if mask_new.any():
            idx_new = torch.nonzero(mask_new, as_tuple=False).squeeze(1)
            alpha_vals = torch.clamp(age_tensor[idx_new] / ramp, 0.0, 1.0)
            base_weight = float(self.config.new_token_weight_base)
            weighted_loss = torch.tensor(0.0, device=self.device)
            weight_sum = torch.tensor(0.0, device=self.device)
            for offset, token_id, alpha in zip(idx_new.tolist(), flat_targets[idx_new].tolist(), alpha_vals.tolist()):
                lp = log_probs[offset]
                parents = self.vocab.parent_ids(int(token_id))
                if parents:
                    parent_lp = sum(lp[parent] for parent in parents) / len(parents)
                else:
                    parent_lp = lp[token_id]
                token_lp = lp[token_id]
                sample_loss = -((1.0 - alpha) * parent_lp + alpha * token_lp)
                weight = base_weight + (1.0 - base_weight) * alpha
                weighted_loss = weighted_loss + weight * sample_loss
                weight_sum = weight_sum + weight
            if weight_sum.item() > 0:
                loss_new = weighted_loss / weight_sum

        total_loss = torch.tensor(0.0, device=self.device)
        components = []
        if loss_old.requires_grad or loss_old.item() > 0:
            components.append(loss_old)
        if loss_new.requires_grad or loss_new.item() > 0:
            components.append(loss_new)
        if components:
            total_loss = sum(components)
        return total_loss, float(loss_old.detach().cpu().item()), float(loss_new.detach().cpu().item())

    def _update_ema(self) -> None:
        if self.step < self.config.ema_start_step:
            return
        decay = self.config.ema_decay
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            buf = self._ema_state.get(name)
            if buf is None:
                self._ema_state[name] = param.detach().clone()
            else:
                buf.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
