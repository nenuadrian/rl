from __future__ import annotations

from typing import Any

from hyperparameters._common import get_preset


SHARED_VMPO_PARAMS: dict[str, Any] = {
    "num_envs": 1,
    "rollout_steps": 4096,
    "m_steps": 16,
    "policy_layer_sizes": (512, 256),
    "value_layer_sizes": (512, 256),
    "gamma": 0.98,
    "policy_lr": 1e-4,
    "value_lr": 1e-4,
    "topk_fraction": 0.6,
    "temperature_init": 1.0,
    "temperature_lr": 1e-4,
    "alpha_lr": 1e-4,
    "epsilon_eta": 0.15,
    "epsilon_mu": 0.05,
    "epsilon_sigma": 0.05,
    "max_grad_norm": 2.0,
    "normalize_advantages": True,
    "optimizer_type": "adam",
    "shared_encoder": False,
}


PRESETS: dict[str, dict[str, Any]] = {
    "dm_control/cheetah/run": {
        "total_steps": 3_000_000,
        **SHARED_VMPO_PARAMS,
    },
    "dm_control/humanoid/run": {
        "total_steps": 10_000_000,
        **SHARED_VMPO_PARAMS,
    },
    "dm_control/humanoid/run_pure_state": {
        "total_steps": 10_000_000,
        **SHARED_VMPO_PARAMS,
    },
    "dm_control/humanoid/walk": {
        "total_steps": 5_000_000,
        **SHARED_VMPO_PARAMS,
    },
    "dm_control/walker/walk": {
        "total_steps": 3_000_000,
        **SHARED_VMPO_PARAMS,
    },
    "dm_control/walker/run": {
        "total_steps": 3_000_000,
        **SHARED_VMPO_PARAMS,
    },
    "Humanoid-v5": {
        "total_steps": 5_000_000,
        **SHARED_VMPO_PARAMS,
    },
    "Ant-v5": {
        "total_steps": 5_000_000,
        **SHARED_VMPO_PARAMS,
    },
    "HalfCheetah-v5": {
        "total_steps": 2_000_000,
        **SHARED_VMPO_PARAMS,
    },
    "Walker2d-v5": {
        "total_steps": 3_000_000,
        **SHARED_VMPO_PARAMS,
    },
}


def get(env_id: str) -> dict[str, Any]:
    return get_preset(
        env_id=env_id,
        presets=PRESETS,
        algorithm_name="VMPO",
        defaults={
            "optimizer_type": "adam",
            "sgd_momentum": 0.9,
            "advantage_estimator": "gae",
            "gae_lambda": 0.95,
        },
    )
