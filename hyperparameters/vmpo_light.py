from __future__ import annotations

from typing import Any


PRESETS: dict[tuple[str, str], dict[str, Any]] = {
    ("cheetah", "run"): {
        "total_steps": 10_000_000,
        "eval_interval": 25_000,
        "save_interval": 50_000,
        "policy_layer_sizes": (256, 256),
        "rollout_steps": 4096,
        "gamma": 0.99,
        "policy_lr": 1e-4,
        "value_lr": 1e-4,
        "eta": 5.0,
        "eta_lr": 1e-3,
        "epsilon_eta": 0.1,
    },
    ("humanoid", "run"): {
        "total_steps": 30_000_000,
        "eval_interval": 50_000,
        "save_interval": 100_000,
        "policy_layer_sizes": (512, 256),
        "rollout_steps": 16384,
        "gamma": 0.995,
        "policy_lr": 5e-5,
        "value_lr": 1e-4,
        "eta": 10.0,
        "eta_lr": 5e-4,
        "epsilon_eta": 0.3,
    },
    ("humanoid", "walk"): {
        "total_steps": 30_000_000,
        "eval_interval": 50_000,
        "save_interval": 100_000,
        "policy_layer_sizes": (512, 256),
        "rollout_steps": 16384,
        "gamma": 0.995,
        "policy_lr": 5e-5,
        "value_lr": 1e-4,
        "eta": 10.0,
        "eta_lr": 5e-4,
        "epsilon_eta": 0.3,
    },
    ("hopper", "stand"): {
        "total_steps": 5_000_000,
        "eval_interval": 25_000,
        "save_interval": 50_000,
        "policy_layer_sizes": (256, 256),
        "rollout_steps": 2048,
        "gamma": 0.99,
        "policy_lr": 1e-4,
        "value_lr": 1e-4,
        "eta": 5.0,
        "eta_lr": 1e-3,
        "epsilon_eta": 0.1,
    },
    ("cartpole", "swingup"): {
        "total_steps": 2_000_000,
        "eval_interval": 10_000,
        "save_interval": 25_000,
        "policy_layer_sizes": (128, 128),
        "rollout_steps": 2048,
        "gamma": 0.99,
        "policy_lr": 1e-4,
        "value_lr": 1e-4,
        "eta": 5.0,
        "eta_lr": 1e-3,
        "epsilon_eta": 0.1,
    },
    ("walker", "walk"): {
        "num_envs": 1,
        "total_steps": 3_000_000,
        "eval_interval": 25_000,
        "save_interval": 50_000,
        "policy_layer_sizes": (512, 256),
        "rollout_steps": 16_384,
        "gamma": 0.99,
        "policy_lr": 3e-4,
        "value_lr": 1e-4,
        "eta": 5.0,
        "eta_lr": 5e-4,
        "epsilon_eta": 0.1,
    },
    ("walker", "run"): {
        "num_envs": 1,
        "total_steps": 5_000_000,
        "eval_interval": 25_000,
        "save_interval": 50_000,
        "policy_layer_sizes": (512, 256),
        "rollout_steps": 24_576,
        "gamma": 0.995,
        "policy_lr": 2e-4,
        "value_lr": 1e-4,
        "eta": 8.0,
        "eta_lr": 3e-4,
        "epsilon_eta": 0.2,
    },
}


def get(domain: str, task: str) -> dict[str, Any]:
    key = (domain, task)
    if key not in PRESETS:
        available = ", ".join([f"{d}/{t}" for (d, t) in sorted(PRESETS.keys())])
        raise KeyError(f"No VMPO preset for {domain}/{task}. Available: {available}")
    return dict(PRESETS[key])
