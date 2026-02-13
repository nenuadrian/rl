from __future__ import annotations

from typing import Any

from hyperparameters._common import get_preset


PRESETS: dict[str, dict[str, Any]] = {
    "ProofofMemory-v0": {
        # Train/eval/checkpoint control
        "num_envs": 16,
        "total_steps": 25_000,
        "eval_interval": 2_048,
        "save_interval": 8_192,
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
    }
}


def get(env_id: str) -> dict[str, Any]:
    return get_preset(
        env_id=env_id,
        presets=PRESETS,
        algorithm_name="PPO-TRxL",
    )
