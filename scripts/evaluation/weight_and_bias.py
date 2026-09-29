from __future__ import annotations

from typing import Any

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    wandb = None  # type: ignore


def _init(project: str = "adaptive-laya", entity: str | None = None, **kw: Any) -> Any:
    if not _WANDB_AVAILABLE:
        return None
    return wandb.init(project=project, entity=entity, **kw)


def log(metrics: dict[str, float], step: int | None = None, **kw: Any) -> None:
    if not _WANDB_AVAILABLE or wandb.run is None:
        return
    wandb.log(metrics, step=step, **kw)


def finish() -> None:
    if not _WANDB_AVAILABLE:
        return
    wandb.finish()


def config(**kw: Any) -> None:
    if not _WANDB_AVAILABLE or wandb.run is None:
        return
    wandb.config.update(kw)
