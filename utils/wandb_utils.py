from __future__ import annotations

import os
import re
from typing import Any, Mapping

import wandb


def init_wandb(
    *,
    project: str | None,
    entity: str | None,
    group: str | None,
    name: str | None = None,
    config: Mapping[str, Any] | None,
    monitor_gym: bool = False,
    save_code: bool = False,
) -> bool:
    if wandb is None:
        return False
    init_kwargs: dict[str, Any] = {
        "project": project,
        "entity": entity,
        "group": group,
        "config": config,
        "name": name,
        "save_code": save_code,
        "monitor_gym": monitor_gym,
    }
    wandb.init(**init_kwargs)
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


