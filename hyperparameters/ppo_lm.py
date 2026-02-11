from __future__ import annotations

from typing import Any


def _base_gsm8k_preset(model_name: str) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "dtype": "bfloat16",
        "learning_rate": 2e-6,
        "weight_decay": 0.0,
        "adam_eps": 1e-8,
        "num_steps": 3000,
        "prompts_per_step": 16,
        "ppo_epochs": 4,
        "minibatch_size": 8,
        "clip_epsilon": 0.2,
        "cliprange_value": 0.2,
        "vf_coef": 0.1,
        "ent_coef": 5e-4,
        "max_grad_norm": 1.0,
        "gamma": 1.0,
        "lam": 0.95,
        "whiten_advantages": True,
        "whiten_rewards": False,
        "kl_coef": 0.02,
        "adaptive_kl": True,
        "target_ref_kl": 0.08,
        "kl_horizon": 10_000,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 50,
        "max_prompt_length": 384,
        "max_new_tokens": 160,
        "eval_max_new_tokens": 192,
        "eval_every": 25,
        "eval_examples": 128,
        "save_every": 250,
        "dataset_name": "openai/gsm8k",
        "dataset_config": "main",
        "train_split": "train",
        "eval_split": "test",
        "train_subset_size": None,
        "eval_subset_size": 500,
        "prompt_template": (
            "Solve the following grade-school math problem. "
            "Show your work, then end with `#### <number>`.\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
        "reward_correct": 1.0,
        "reward_has_answer_tag": 0.1,
        "reward_parseable": 0.1,
        "reward_wrong": 0.0,
        "reference_on_cpu": False,
    }


PRESETS: dict[str, dict[str, Any]] = {
    "smollm-135m": {
        **_base_gsm8k_preset("HuggingFaceTB/SmolLM-135M"),
    },
    "smollm-360m": {
        **_base_gsm8k_preset("HuggingFaceTB/SmolLM-360M"),
    },
    "smollm-135m-gsm8k": {
        **_base_gsm8k_preset("HuggingFaceTB/SmolLM-135M"),
    },
    "smollm-360m-gsm8k": {
        **_base_gsm8k_preset("HuggingFaceTB/SmolLM-360M"),
    },
    "smollm-135m-gsm8k-socratic": {
        **_base_gsm8k_preset("HuggingFaceTB/SmolLM-135M"),
        "dataset_config": "socratic",
        "max_new_tokens": 192,
        "eval_max_new_tokens": 224,
    },
    "smollm-360m-gsm8k-socratic": {
        **_base_gsm8k_preset("HuggingFaceTB/SmolLM-360M"),
        "dataset_config": "socratic",
        "max_new_tokens": 192,
        "eval_max_new_tokens": 224,
    },
}


def get(env_id: str) -> dict[str, Any]:
    if env_id not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise KeyError(f"No PPO-LM preset for {env_id}. Available: {available}")
    return dict(PRESETS[env_id])
