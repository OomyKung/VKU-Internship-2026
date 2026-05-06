"""Retired legacy experiment-runner compatibility module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .removed_swarm import REMOVED_SWARM_FEATURE_MESSAGE


@dataclass(slots=True)
class ExperimentSettings:
    """Minimal compatibility shell for legacy imports."""

    protected_attribute: str = ""
    budget: int = 0
    output_dir: Path | None = None


def run_experiment(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(REMOVED_SWARM_FEATURE_MESSAGE)


def run_loaded_experiment(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(REMOVED_SWARM_FEATURE_MESSAGE)
