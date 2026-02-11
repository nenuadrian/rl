from __future__ import annotations

from typing import Any

from hyperparameters.vmpo import get as _get_vmpo_preset


def get(env_id: str) -> dict[str, Any]:
    preset = _get_vmpo_preset(env_id)
    preset["optimizer_type"] = "sgd"
    return preset
