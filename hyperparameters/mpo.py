from __future__ import annotations

from typing import Any

from hyperparameters._common import get_preset, merge_preset

_MPO_DEFAULTS: dict[str, Any] = {
    "replay_size": 1_000_000,
    "save_interval": 50_000,
    "gamma": 0.995,
    "tau": 0.005,
    "policy_lr": 3e-4,
    "q_lr": 3e-4,
    "temperature_lr": 3e-4,
    "kl_epsilon": 0.1,
    "mstep_kl_epsilon": 0.1,
    "per_dim_constraining": False,
    "lambda_init": 1.0,
    "lambda_lr": 3e-4,
    "action_penalization": False,
    "epsilon_penalty": 0.001,
    "max_grad_norm": 1.0,
    "action_samples": 256,
    "temperature_init": 1.0,
    "use_retrace": True,
    "retrace_steps": 2,
    "retrace_mc_actions": 8,
    "retrace_lambda": 0.95,
}


def _mpo_preset(**overrides: Any) -> dict[str, Any]:
    return merge_preset(_MPO_DEFAULTS, **overrides)


_DM_HUMANOID_BASE = _mpo_preset(
    total_steps=50_000_000,
    update_after=500_000,
    batch_size=512,
    eval_interval=2_000,
    updates_per_step=1,
    policy_layer_sizes=(256, 256, 256),
    critic_layer_sizes=(256, 256, 256),
)

_DM_WALKER_BASE = _mpo_preset(
    updates_per_step=2,
    total_steps=40_000_000,
    update_after=500_000,
    batch_size=256,
    eval_interval=3_000,
    policy_layer_sizes=(256, 256, 256),
    critic_layer_sizes=(256, 256, 256),
)

_GYM_BASE = _mpo_preset(
    updates_per_step=1,
    total_steps=50_000_000,
    update_after=10_000,
    batch_size=512,
    eval_interval=7_000,
    policy_layer_sizes=(256, 256, 256),
    critic_layer_sizes=(256, 256, 256),
    temperature_init=3.0,
)


PRESETS: dict[str, dict[str, Any]] = {
    "dm_control/cheetah/run": _mpo_preset(
        updates_per_step=1,
        total_steps=20_000_000,
        update_after=1_000,
        batch_size=256,
        eval_interval=5_000,
        policy_layer_sizes=(256, 256, 256),
        critic_layer_sizes=(256, 256, 256),
        kl_epsilon=0.2,
        per_dim_constraining=True,
        action_samples=128,
    ),
    "dm_control/humanoid/run": merge_preset(_DM_HUMANOID_BASE),
    "dm_control/humanoid/walk": merge_preset(
        _DM_HUMANOID_BASE,
        updates_per_step=2,
        eval_interval=20_000,
        critic_layer_sizes=(512, 256, 256),
    ),
    "dm_control/walker/walk": merge_preset(_DM_WALKER_BASE),
    "dm_control/walker/run": merge_preset(_DM_WALKER_BASE, temperature_init=3.0),
    "dm_control/cartpole/swingup": _mpo_preset(
        updates_per_step=1,
        total_steps=500_000,
        update_after=1_000,
        batch_size=256,
        replay_size=200_000,
        eval_interval=5_000,
        policy_layer_sizes=(256, 256, 256),
        critic_layer_sizes=(512, 512, 256),
        per_dim_constraining=True,
        lambda_lr=1e-3,
        action_samples=64,
        use_retrace=False,
    ),
    "Humanoid-v5": _mpo_preset(
        updates_per_step=1,
        total_steps=50_000_000,
        update_after=50_000,
        batch_size=512,
        eval_interval=2_000,
        policy_layer_sizes=(256, 256, 256),
        critic_layer_sizes=(256, 256, 256),
        temperature_init=3.0,
    ),
    "HalfCheetah-v5": merge_preset(_GYM_BASE),
    "Walker2d-v5": merge_preset(_GYM_BASE, critic_layer_sizes=(512, 256, 256)),
    "Ant-v5": merge_preset(_GYM_BASE, critic_layer_sizes=(512, 512, 256)),
}


def get(env_id: str) -> dict[str, Any]:
    return get_preset(env_id=env_id, presets=PRESETS, algorithm_name="MPO")
