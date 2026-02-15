from __future__ import annotations

from typing import Any

from hyperparameters._common import get_preset


PRESETS: dict[str, dict[str, Any]] = {
    "math/addition": {
        "model_name": "Qwen/Qwen1.5-0.5B",
        "num_iterations": 500,
        "batch_size": 8,
        "mini_batch_size": 4,
        "ppo_epochs": 1,
        "learning_rate": 5e-6,
        "clip_range": 0.2,
        "clip_range_value": 0.2,
        "vf_coef": 0.2,
        "ent_coef": 0.0,
        "gamma": 1.0,
        "lam": 0.95,
        "kl_coef": 0.5,
        "normalize_advantages": True,
        "max_prompt_length": 128,
        "max_response_length": 8,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 50,
        "target_kl": 0.03,
        "max_grad_norm": 1.0,
        "force_fp32_policy": True,
        "detach_value_head_input": True,
        "reward_type": "shaped",
        "reward_alpha": 1.0,
        "enable_token_level_reward": True,
        "token_level_reward_coef": 1.0,
        "length_penalty_beta": 0.05,
        "stop_on_newline": True,
        "response_parse_mode": "strict_line",
        "correct_reward": 1.0,
        "incorrect_reward": -1.0,
        "invalid_reward": -1.0,
        "reward_error_scale": 100.0,
        "sft_epochs": 0,
        "sft_batch_size": 32,
        "sft_learning_rate": 1e-5,
        "log_num_examples": 10,
        "eval_compare_modes": True,
        "dataset_size": 10_000,
        "dataset_min_value": 100,
        "dataset_max_value": 9999,
        "dataset_seed": 123,
        "prompt_template": (
            "Solve this addition problem and reply with only the final integer.\n"
            "Question: {a} + {b} = "
        ),
        "save_every": 100,
        "log_every": 1,
    },
}


def get(env_id: str) -> dict[str, Any]:
    return get_preset(
        env_id=env_id,
        presets=PRESETS,
        algorithm_name="GPTPPO",
        defaults={
            "optimizer_type": "adam",
            "sgd_momentum": 0.9,
        },
    )
