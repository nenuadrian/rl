from __future__ import annotations

from typing import Any

from hyperparameters._common import get_preset


PRESETS: dict[str, dict[str, Any]] = {
    "math/addition": {
        "model_name": "distilgpt2",
        "num_iterations": 500,
        "batch_size": 8,
        "mini_batch_size": 4,
        "ppo_epochs": 4,
        "learning_rate": 1e-5,
        "clip_range": 0.2,
        "clip_range_value": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.0,
        "gamma": 1.0,
        "lam": 0.95,
        "kl_coef": 0.05,
        "normalize_advantages": True,
        "max_prompt_length": 96,
        "max_response_length": 24,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "target_kl": 0.0,
        "max_grad_norm": 1.0,
        "reward_type": "exact",
        "correct_reward": 1.0,
        "incorrect_reward": 0.0,
        "invalid_reward": 0.0,
        "reward_error_scale": 100.0,
        "sft_epochs": 2,
        "sft_batch_size": 32,
        "sft_learning_rate": 5e-5,
        "log_num_examples": 10,
        "eval_compare_modes": True,
        "dataset_size": 10_000,
        "dataset_min_value": 0,
        "dataset_max_value": 9,
        "dataset_seed": 123,
        "prompt_template": (
            "Solve this addition problem and reply with only the final integer.\\n"
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
