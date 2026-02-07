from __future__ import annotations

from typing import Any, Mapping

import wandb


def init_wandb(
    *,
    project: str | None,
    entity: str | None,
    group: str | None,
    config: Mapping[str, Any] | None,
) -> bool:
    if wandb is None:
        return False
    wandb.init(project=project, entity=entity, group=group, config=config)
    return wandb.run is not None


def log_wandb(metrics: Mapping[str, Any], step: int | None = None) -> None:
    if wandb is None or wandb.run is None:
        return
    wandb.log(dict(metrics), step=step)


def finish_wandb() -> None:
    if wandb is None or wandb.run is None:
        return
    wandb.finish()
