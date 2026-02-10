from __future__ import annotations

from typing import Any, Mapping

import wandb


def init_wandb(
    *,
    project: str | None,
    entity: str | None,
    group: str | None,
    name: str | None = None,
    config: Mapping[str, Any] | None,
) -> bool:
    if wandb is None:
        return False
    wandb.init(project=project, entity=entity, group=group, config=config, name=name)
    return wandb.run is not None


def log_wandb(metrics: Mapping[str, Any], step: int | None = None, silent: bool = False) -> None:
    if wandb is None or wandb.run is None:
        return
    wandb.log(dict(metrics), step=step)
    if not silent:
        print(f"step {step}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))


def finish_wandb() -> None:
    if wandb is None or wandb.run is None:
        return
    wandb.finish()
