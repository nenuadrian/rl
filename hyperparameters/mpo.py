from __future__ import annotations

from typing import Any

from hyperparameters._common import get_preset
from hyperparameters.vmpo import SHARED_VMPO_PARAMS


SHARED_MPO_PARAMS: dict[str, Any] = {
    "replay_size": 1_000_000,
    "total_steps": 500_000,
    "update_after": 10_000,
    "batch_size": 256,
    "updates_per_step": 1,
    "policy_layer_sizes": (256, 256, 256),
    "critic_layer_sizes": (512, 512, 256),
    "gamma": 0.99,
    "target_networks_update_period": 100,
    "policy_lr": 1e-4,
    "q_lr": 1e-4,
    "temperature_lr": 1e-3,
    "kl_epsilon": 0.1,
    "mstep_kl_epsilon": 2.5e-3,
    "lambda_init": 1.0,
    "lambda_lr": 1e-2,
    "epsilon_penalty": 0.001,
    "max_grad_norm": 1.0,
    "action_samples": 64,
    "temperature_init": 10.0,
    "use_retrace": False,
    "retrace_steps": 2,
    "retrace_mc_actions": 8,
    "retrace_lambda": 0.95,
    "init_log_alpha_mean": 10.0,
    "init_log_alpha_stddev": 1000.0,
}


PRESETS: dict[str, dict[str, Any]] = {
    "dm_control/cheetah/run": {
        **SHARED_MPO_PARAMS,
    },
    "dm_control/humanoid/run": {
        **SHARED_MPO_PARAMS,
    },
    "dm_control/humanoid/run_pure_state": {
        **SHARED_VMPO_PARAMS,
    },
    "dm_control/humanoid/walk": {
        **SHARED_MPO_PARAMS,
    },
    "dm_control/walker/walk": {
        **SHARED_MPO_PARAMS,
    },
    "dm_control/walker/run": {
        **SHARED_MPO_PARAMS,
    },
    "Humanoid-v5": {
        **SHARED_MPO_PARAMS,
    },
    "HalfCheetah-v5": {
        **SHARED_MPO_PARAMS,
    },
    "Walker2d-v5": {
        **SHARED_MPO_PARAMS,
    },
    "Ant-v5": {
        **SHARED_MPO_PARAMS,
    },
}


def get(env_id: str) -> dict[str, Any]:
    return get_preset(
        env_id=env_id,
        presets=PRESETS,
        algorithm_name="MPO",
        defaults={"optimizer_type": "adam", "sgd_momentum": 0.9},
    )
