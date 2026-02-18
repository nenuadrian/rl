from __future__ import annotations

from typing import Any

from hyperparameters._common import get_preset


SHARED_VMPO_PARAMS: dict[str, Any] = {
    "num_envs": 1,
    "rollout_steps": 4096,
    "m_steps": 5,
    "policy_layer_sizes": (512, 256, 256),
    "value_layer_sizes": (512, 256, 256),
    "gamma": 0.99,
    "policy_lr": 1e-4,
    "value_lr": 1e-4,
    "topk_fraction": 0.4,
    "temperature_init": 1.0,
    "temperature_lr": 1e-4,
    "alpha_lr": 1e-4,
    "epsilon_eta": 0.1,
    "epsilon_mu": 0.02,
    "epsilon_sigma":  0.005,
    "max_grad_norm": 10.0,
    "normalize_advantages": True,
    "optimizer_type": "adam",
    "shared_encoder": False,
}

HUMANOID_V5_VMPO_PARAMS: dict[str, Any] = {
    **SHARED_VMPO_PARAMS,
    "num_envs": 1,
    "rollout_steps": 2048,
    "m_steps": 1,
    "policy_layer_sizes": (256, 256, 256),
    "value_layer_sizes": (512, 512, 256),
    "gamma": 0.995,
    "policy_lr": 2e-4,
    "value_lr": 3e-4,
    "topk_fraction": 0.5,
    "temperature_lr": 5e-4,
    "alpha_lr": 3e-4,
    "epsilon_eta": 0.05,
    "epsilon_mu": 0.05,
    "epsilon_sigma": 0.005,
    "max_grad_norm": 1.0,
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
        "total_steps": 10_000_000,
        **HUMANOID_V5_VMPO_PARAMS,
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
            "gae_lambda": 0.85,
        },
    )
