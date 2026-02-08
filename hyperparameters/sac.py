from __future__ import annotations

from typing import Any


PRESETS: dict[tuple[str, str], dict[str, Any]] = {
    ("cheetah", "run"): {
        "total_steps": 3_000_000,
        "start_steps": 10_000,
        "update_after": 1_000,
        "batch_size": 256,
        "updates_per_step": 1,
        "replay_size": 1_000_000,
        "eval_interval": 10_000,
        "save_interval": 50_000,
        "hidden_sizes": (256, 256),
        "gamma": 0.99,
        "tau": 0.005,
        "policy_lr": 3e-4,
        "q_lr": 3e-4,
        "alpha_lr": 3e-4,
        "automatic_entropy_tuning": True,
    },
    ("humanoid", "run"): {
        "total_steps": 10_000_000,
        "start_steps": 20_000,
        "update_after": 5_000,
        "batch_size": 512,
        "updates_per_step": 1,
        "replay_size": 2_000_000,
        "eval_interval": 25_000,
        "save_interval": 100_000,
        "hidden_sizes": (512, 256),
        "gamma": 0.99,
        "tau": 0.005,
        "policy_lr": 3e-4,
        "q_lr": 3e-4,
        "alpha_lr": 3e-4,
        "automatic_entropy_tuning": True,
    },
    ("humanoid", "walk"): {
        "total_steps": 5_000_000,
        "start_steps": 20_000,
        "update_after": 2_000,
        "batch_size": 512,
        "updates_per_step": 1,
        "replay_size": 1_000_000,
        "eval_interval": 20_000,
        "save_interval": 100_000,
        "hidden_sizes": (512, 256),
        "gamma": 0.99,
        "tau": 0.005,
        "policy_lr": 3e-4,
        "q_lr": 3e-4,
        "alpha_lr": 3e-4,
        "automatic_entropy_tuning": True,
    },
    ("hopper", "stand"): {
        "total_steps": 2_000_000,
        "start_steps": 10_000,
        "update_after": 1_000,
        "batch_size": 256,
        "updates_per_step": 1,
        "replay_size": 1_000_000,
        "eval_interval": 10_000,
        "save_interval": 50_000,
        "hidden_sizes": (256, 256),
        "gamma": 0.99,
        "tau": 0.005,
        "policy_lr": 3e-4,
        "q_lr": 3e-4,
        "alpha_lr": 3e-4,
        "automatic_entropy_tuning": True,
    },
    ("cartpole", "swingup"): {
        "total_steps": 500_000,
        "start_steps": 5_000,
        "update_after": 1_000,
        "batch_size": 256,
        "updates_per_step": 1,
        "replay_size": 200_000,
        "eval_interval": 5_000,
        "save_interval": 25_000,
        "hidden_sizes": (128, 128),
        "gamma": 0.99,
        "tau": 0.005,
        "policy_lr": 3e-4,
        "q_lr": 3e-4,
        "alpha_lr": 3e-4,
        "automatic_entropy_tuning": True,
    },
}


def get(domain: str, task: str) -> dict[str, Any]:
    key = (domain, task)
    if key not in PRESETS:
        available = ", ".join([f"{d}/{t}" for (d, t) in sorted(PRESETS.keys())])
        raise KeyError(f"No SAC preset for {domain}/{task}. Available: {available}")
    return dict(PRESETS[key])
