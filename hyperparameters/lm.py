from __future__ import annotations

from typing import Any


def _base_lm_ppo_preset(model_name: str) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "dtype": "bfloat16",
        "learning_rate": 5e-6,
        "weight_decay": 0.0,
        "num_steps": 10000,
        "prompts_per_step": 32,
        "ppo_epochs": 4,
        "minibatch_size": 32,
        "clip_epsilon": 0.2,
        "max_grad_norm": 1.0,
        "ent_coef": 1e-3,
        "head_only_steps": 0,
        "eval_every": 25,
        "eval_examples": 512,
        "save_every": 500,
        "reward_std_eps": 1e-6,
        "advantage_mode": "ema_baseline",
        "normalize_advantages": True,
        "baseline_momentum": 0.9,
        "kl_coef": 0.02,
        "kl_coef_min": 1e-3,
        "target_ref_kl": 0.08,
        "kl_adaptation_factor": 1.5,
        "kl_coef_up_mult": 1.02,
        "kl_coef_down_div": 1.02,
        "prompt_template": "Sentence: {text}\nSentiment:",
        "max_prompt_length": 256,
        "train_subset_size": None,
        "eval_subset_size": 4096,
        "train_transformer": True,
    }


PRESETS: dict[str, dict[str, Any]] = {
    "smollm-135m": {
        **_base_lm_ppo_preset("HuggingFaceTB/SmolLM-135M"),
        "dataset_name": "glue",
        "dataset_config": "sst2",
        "train_split": "train",
        "eval_split": "validation",
        "text_key": "sentence",
        "label_key": "label",
        "negative_label_ids": (0,),
        "positive_label_ids": (1,),
    },
    "smollm-360m": {
        **_base_lm_ppo_preset("HuggingFaceTB/SmolLM-360M"),
        "dataset_name": "glue",
        "dataset_config": "sst2",
        "train_split": "train",
        "eval_split": "validation",
        "text_key": "sentence",
        "label_key": "label",
        "negative_label_ids": (0,),
        "positive_label_ids": (1,),
    },
    "smollm-135m-agnews-binary": {
        **_base_lm_ppo_preset("HuggingFaceTB/SmolLM-135M"),
        "dataset_name": "ag_news",
        "dataset_config": None,
        "train_split": "train",
        "eval_split": "test",
        "text_key": "text",
        "label_key": "label",
        # Binary collapse: {World, Sports}=negative vs {Business, Sci/Tech}=positive
        "negative_label_ids": (0, 1),
        "positive_label_ids": (2, 3),
    },
    "smollm-360m-agnews-binary": {
        **_base_lm_ppo_preset("HuggingFaceTB/SmolLM-360M"),
        "dataset_name": "ag_news",
        "dataset_config": None,
        "train_split": "train",
        "eval_split": "test",
        "text_key": "text",
        "label_key": "label",
        "negative_label_ids": (0, 1),
        "positive_label_ids": (2, 3),
    },
}


def get(env_id: str) -> dict[str, Any]:
    if env_id not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise KeyError(f"No LM preset for {env_id}. Available: {available}")
    return dict(PRESETS[env_id])
