from __future__ import annotations

from typing import Any, Mapping

import wandb



def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, (float, int, bool, str)):
        return value
    if value is None:
        return value
    # NumPy scalar / zero-dim array compatibility.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            return value
    return value



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
    payload: dict[str, Any] = {
        str(key): _coerce_scalar(value) for key, value in dict(metrics).items()
    }
    wandb.log(payload, step=step)
    if not silent:
        print(f"step {step}: " + ", ".join(f"{k}={v:.3f}" for k, v in payload.items()))


def finish_wandb() -> None:
    if wandb is None or wandb.run is None:
        return
    wandb.finish()
