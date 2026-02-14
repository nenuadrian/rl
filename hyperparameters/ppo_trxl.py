from __future__ import annotations

from typing import Any

from hyperparameters._common import get_preset


_CLEANRL_MEMORY_GYM_PPO_TRXL_DEFAULTS: dict[str, Any] = {
    # Train/eval/checkpoint control
    "num_envs": 1,
    "total_steps": 200_000_000,
    # PPO rollout/update
    "num_steps": 512,
    "num_minibatches": 8,
    "update_epochs": 3,
    "init_lr": 2.75e-4,
    "final_lr": 1e-5,
    "anneal_steps": 163_840_000,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "norm_adv": False,
    "clip_coef": 0.1,
    "clip_vloss": True,
    "init_ent_coef": 1e-4,
    "final_ent_coef": 1e-6,
    "vf_coef": 0.5,
    "max_grad_norm": 0.25,
    "target_kl": None,
    # TRxL
    "trxl_num_layers": 3,
    "trxl_num_heads": 4,
    "trxl_dim": 384,
    "trxl_memory_length": 119,
    "trxl_positional_encoding": "absolute",
    "reconstruction_coef": 0.0,
}


PRESETS: dict[str, dict[str, Any]] = {
    "ProofofMemory-v0": {
        # Train/eval/checkpoint control
        "num_envs": 1,
        "total_steps": 25_000,
        # PPO rollout/update
        "num_steps": 128,
        "num_minibatches": 8,
        "update_epochs": 4,
        "init_lr": 3e-4,
        "final_lr": 1e-5,
        "anneal_steps": 163_840_000,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "norm_adv": False,
        "clip_coef": 0.2,
        "clip_vloss": True,
        "init_ent_coef": 1e-3,
        "final_ent_coef": 1e-6,
        "vf_coef": 0.1,
        "max_grad_norm": 0.5,
        "target_kl": None,
        # TRxL
        "trxl_num_layers": 4,
        "trxl_num_heads": 1,
        "trxl_dim": 64,
        "trxl_memory_length": 16,
        "trxl_positional_encoding": "none",
        "reconstruction_coef": 0.0,
    },
    "MortarMayhem-Grid-v0": {
        **_CLEANRL_MEMORY_GYM_PPO_TRXL_DEFAULTS,
        "total_steps": 100_000_000,
        "norm_adv": True,
        "trxl_memory_length": 119,
    },
    "MysteryPath-Grid-v0": {
        **_CLEANRL_MEMORY_GYM_PPO_TRXL_DEFAULTS,
        "total_steps": 100_000_000,
        "trxl_memory_length": 96,
    },
    "MysteryPath-v0": {
        **_CLEANRL_MEMORY_GYM_PPO_TRXL_DEFAULTS,
        "trxl_memory_length": 256,
    },
}


def get(env_id: str) -> dict[str, Any]:
    return get_preset(
        env_id=env_id,
        presets=PRESETS,
        algorithm_name="PPO-TRxL",
    )
