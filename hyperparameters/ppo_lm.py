from __future__ import annotations

from typing import Any

from hyperparameters._common import get_preset


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


def _gsm8k_preset(model_name: str, **overrides: Any) -> dict[str, Any]:
    return {
        **_base_gsm8k_preset(model_name),
        **overrides,
    }


_VINE_MATH_PPO_INSPIRED_OVERRIDES: dict[str, Any] = {
    "learning_rate": 1e-6,
    "ppo_epochs": 2,
    "gamma": 1.0,
    "lam": 1.0,
    "adaptive_kl": False,
    "kl_coef": 1e-4,
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 0,
    "reference_on_cpu": True,
}


_DEEPSEEK_MATH_7B_BASE_GSM8K = _gsm8k_preset(
    "deepseek-ai/deepseek-math-7b-base",
    **_VINE_MATH_PPO_INSPIRED_OVERRIDES,
    num_steps=2000,
    prompts_per_step=4,
    minibatch_size=1,
    max_new_tokens=160,
    eval_max_new_tokens=192,
    eval_every=50,
    eval_examples=64,
    save_every=200,
)


_RHO_MATH_1B_V0_1_GSM8K = _gsm8k_preset(
    "microsoft/rho-math-1b-v0.1",
    **_VINE_MATH_PPO_INSPIRED_OVERRIDES,
    num_steps=2500,
    prompts_per_step=8,
    minibatch_size=2,
    max_new_tokens=192,
    eval_max_new_tokens=224,
    eval_every=40,
    eval_examples=96,
    save_every=200,
)


PRESETS: dict[str, dict[str, Any]] = {
    "smollm-135m-gsm8k": _gsm8k_preset("HuggingFaceTB/SmolLM-135M-Instruct"),
    "smollm-360m-gsm8k": _gsm8k_preset("HuggingFaceTB/SmolLM-360M-Instruct"),
    "deepseek-math-7b-base-gsm8k": _DEEPSEEK_MATH_7B_BASE_GSM8K,
    "rho-math-1b-v0.1-gsm8k": _RHO_MATH_1B_V0_1_GSM8K,
}


def get(env_id: str) -> dict[str, Any]:
    return get_preset(env_id=env_id, presets=PRESETS, algorithm_name="PPO-LM")
