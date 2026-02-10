from __future__ import annotations

from typing import Any



PRESETS: dict[str, dict[str, Any]] = {
    "test": {
        "num_epochs": 1,
        "device_batch_size": 8,
        "examples_per_step": 16,
        "num_samples": 16,
        "max_new_tokens": 256,
        "temperature": 1.0,
        "top_k": 50,
        "embedding_lr": 0.2,
        "unembedding_lr": 0.004,
        "matrix_lr": 0.02,
        "weight_decay": 0.0,
        "init_lr_frac": 0.05,
        "eval_every": 60,
        "eval_examples": 400,
        "save_every": 60,
        "model_tag": None,
        "model_step": None,
        "dtype": "bfloat16",
    },
}


def get(env_id: str) -> dict[str, Any]:
    if env_id not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise KeyError(f"No nanochat_vmpo preset for {env_id}. Available: {available}")
    return dict(PRESETS[env_id])
