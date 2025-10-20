"""Configuration objects for the Salience runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..core.controller import BanditConfig
from ..core.controller.policy import ControllerConfig
from ..core.ideas import IdeaFactoryConfig
from ..core.memory import MaintenanceThresholds
from ..core.operators import GraphReasonerConfig, SASSConfig
from ..core.scheduler import SchedulerConfig
from ..core.meta import MetaStateConfig


@dataclass(slots=True)
class ReflectionConfig:
    """Settings controlling runtime reflection facilities."""

    workspace_root: Optional[str] = None
    scratchpad_tokens: int = 512
    history_capacity: int = 128


@dataclass(slots=True)
class MaintenanceConfig:
    """Settings for background memory maintenance routines."""

    enabled: bool = True
    thresholds: MaintenanceThresholds = field(default_factory=MaintenanceThresholds)


@dataclass(slots=True)
class ExperimentConfig:
    """Configuration for self-experiments during runtime."""

    enabled: bool = True
    parameters: Sequence[str] = field(
        default_factory=lambda: ("controller.lambda_cost", "scheduler.min_budget_ratio")
    )
    duration: int = 16
    max_concurrent: int = 1


@dataclass(slots=True)
class EpisodicConfig:
    """Configuration for episodic memory logging."""

    enabled: bool = True
    store_path: Optional[str] = None


@dataclass(slots=True)
class RuntimeConfig:
    """Top-level runtime configuration (seed release defaults)."""

    budget_tokens: int = 1024
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    controller_bandit: BanditConfig = field(default_factory=BanditConfig)
    meta: MetaStateConfig = field(default_factory=MetaStateConfig)
    graph_reasoner: GraphReasonerConfig = field(default_factory=GraphReasonerConfig)
    sass: SASSConfig = field(default_factory=SASSConfig)
    idea_factory: IdeaFactoryConfig = field(default_factory=IdeaFactoryConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    maintenance: MaintenanceConfig = field(default_factory=MaintenanceConfig)
    experiments: ExperimentConfig = field(default_factory=ExperimentConfig)
    episodic: EpisodicConfig = field(default_factory=EpisodicConfig)


__all__ = [
    "RuntimeConfig",
    "ReflectionConfig",
    "MaintenanceConfig",
    "ExperimentConfig",
    "EpisodicConfig",
]
