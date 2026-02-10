from __future__ import annotations

from typing import Any



PRESETS: dict[tuple[str, str], dict[str, Any]] = {
    ("test", "test"): {
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


def get(domain: str, task: str) -> dict[str, Any]:
    key = (domain, task)
    if key not in PRESETS:
        available = ", ".join([f"{d}/{t}" for (d, t) in sorted(PRESETS.keys())])
        raise KeyError(f"No PPO preset for {domain}/{task}. Available: {available}")
    return dict(PRESETS[key])
