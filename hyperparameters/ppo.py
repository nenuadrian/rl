from __future__ import annotations

from typing import Any

from hyperparameters._common import get_preset, merge_preset

_PPO_DEFAULTS: dict[str, Any] = {
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.02,
    "normalize_obs": True,
}


def _ppo_preset(**overrides: Any) -> dict[str, Any]:
    return merge_preset(_PPO_DEFAULTS, **overrides)


_HUMANOID_RUN_BASE = _ppo_preset(
    num_envs=8,
    total_steps=10_000_000,
    save_interval=1_000_000,
    policy_layer_sizes=(256, 256, 256),
    critic_layer_sizes=(512, 256, 256),
    rollout_steps=8192,
    update_epochs=12,
    minibatch_size=512,
    policy_lr=1e-4,
    value_lr=2e-5,
    clip_ratio=0.15,
    ent_coef=3e-4,
)

_WALKER_BASE = _ppo_preset(
    num_envs=4,
    total_steps=20_000_000,
    eval_interval=15_000,
    save_interval=50_000,
    policy_layer_sizes=(256, 256, 256),
    critic_layer_sizes=(256, 256, 256),
    rollout_steps=2048,
    update_epochs=4,
    policy_lr=2e-4,
    value_lr=1e-4,
    clip_ratio=0.2,
    ent_coef=1e-4,
)

_MUJOCO_MEDIUM_BASE = _ppo_preset(
    num_envs=16,
    total_steps=30_000_000,
    eval_interval=10_000,
    save_interval=1_000_000,
    policy_layer_sizes=(256, 256),
    critic_layer_sizes=(256, 256),
    rollout_steps=8192,
    update_epochs=12,
    minibatch_size=512,
    policy_lr=1e-4,
    value_lr=2e-5,
    clip_ratio=0.15,
    ent_coef=3e-4,
)


PRESETS: dict[str, dict[str, Any]] = {
    "dm_control/cheetah/run": _ppo_preset(
        num_envs=1,
        total_steps=3_000_000,
        eval_interval=10_000,
        save_interval=1_000_000,
        policy_layer_sizes=(256, 256, 256),
        critic_layer_sizes=(256, 256, 256),
        rollout_steps=2048,
        update_epochs=4,
        minibatch_size=256,
        policy_lr=2e-4,
        value_lr=3e-4,
        clip_ratio=0.25,
        ent_coef=1e-4,
        vf_coef=1.0,
    ),
    "dm_control/humanoid/run": merge_preset(_HUMANOID_RUN_BASE, eval_interval=25_000),
    "dm_control/humanoid/walk": _ppo_preset(
        num_envs=16,
        total_steps=5_000_000,
        eval_interval=50_000,
        save_interval=1_000_000,
        policy_layer_sizes=(256, 256),
        critic_layer_sizes=(512, 256),
        rollout_steps=2048,
        update_epochs=2,
        minibatch_size=256,
        policy_lr=2e-4,
        value_lr=5e-5,
        clip_ratio=0.2,
        ent_coef=5e-4,
    ),
    "dm_control/walker/walk": merge_preset(_WALKER_BASE, minibatch_size=128),
    "dm_control/walker/run": merge_preset(_WALKER_BASE, minibatch_size=256),
    "dm_control/cartpole/swingup": _ppo_preset(
        num_envs=2,
        total_steps=5_000_000,
        eval_interval=50_000,
        save_interval=1_000_000,
        policy_layer_sizes=(256, 256, 256),
        critic_layer_sizes=(256, 256, 256),
        rollout_steps=512,
        update_epochs=3,
        minibatch_size=128,
        policy_lr=3e-4,
        value_lr=1e-4,
        clip_ratio=0.2,
        ent_coef=1e-3,
        normalize_obs=False,
    ),
    "Humanoid-v5": merge_preset(_HUMANOID_RUN_BASE, eval_interval=10_000),
    "HalfCheetah-v5": merge_preset(_MUJOCO_MEDIUM_BASE),
    "Ant-v5": merge_preset(_MUJOCO_MEDIUM_BASE),
    "Walker2d-v5": merge_preset(_MUJOCO_MEDIUM_BASE),
}


def get(env_id: str) -> dict[str, Any]:
    return get_preset(env_id=env_id, presets=PRESETS, algorithm_name="PPO")
