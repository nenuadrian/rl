from __future__ import annotations

from typing import Any

from hyperparameters._common import get_preset


SHARED_PPO_PARAMS: dict[str, Any] = {
    "num_envs": 1,
    "policy_layer_sizes": (256, 256, 256),
    "critic_layer_sizes": (512, 512, 256),
    "rollout_steps": 2048,
    "update_epochs": 10,
    "minibatch_size": 32,
    "policy_lr": 2e-4,
    "clip_ratio": 0.2,
    "ent_coef": 0,
    "gamma": 0.98,
    "gae_lambda": 0.92,
    "vf_coef": 0.5,
    "max_grad_norm": 0.8,
    "target_kl": 0,
    "norm_adv": True,
    "clip_vloss": True,
    "anneal_lr": True,
    "normalize_obs": True,
}


PRESETS: dict[str, dict[str, Any]] = {
    "dm_control/cheetah/run": {
        "total_steps": 500_000,
        **SHARED_PPO_PARAMS,
    },
    "dm_control/humanoid/run": {
        "total_steps": 500_000,
        **SHARED_PPO_PARAMS,
    },
    "dm_control/humanoid/walk": {
        "total_steps": 500_000,
        **SHARED_PPO_PARAMS,
    },
    "dm_control/walker/walk": {
        "total_steps": 1_000_000,
        **SHARED_PPO_PARAMS,
    },
    "dm_control/walker/run": {
        "total_steps": 1_000_000,
        **SHARED_PPO_PARAMS,
    },
    "dm_control/cartpole/swingup": {
        "total_steps": 500_000,
        **SHARED_PPO_PARAMS,
    },
    "Humanoid-v5": {
        "total_steps": 500_000,
        **SHARED_PPO_PARAMS,
    },
    "HalfCheetah-v5": {
        "total_steps": 500_000,
        **SHARED_PPO_PARAMS,
    },
    "Ant-v5": {
        "total_steps": 500_000,
        **SHARED_PPO_PARAMS,
    },
    "Walker2d-v5": {
        "total_steps": 500_000,
        **SHARED_PPO_PARAMS,
    },
}


def get(env_id: str) -> dict[str, Any]:
    return get_preset(
        env_id=env_id,
        presets=PRESETS,
        algorithm_name="PPO",
        defaults={"optimizer_type": "adam", "sgd_momentum": 0.9},
    )
