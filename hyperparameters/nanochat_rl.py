from __future__ import annotations

from typing import Any

from hyperparameters._common import get_preset, merge_preset

_NANOCHAT_DEFAULTS: dict[str, Any] = {
    "num_epochs": 1,
    "device_batch_size": 8,
    "examples_per_step": 16,
    "train_examples": 20000,
    "num_samples": 16,
    "max_new_tokens": 256,
    "temperature": 1.0,
    "top_k": 50,
    "embedding_lr": 0.2,
    "unembedding_lr": 0.004,
    "matrix_lr": 0.02,
    "weight_decay": 0.0,
    "init_lr_frac": 0.05,
    "eval_examples": 10,
    "save_every": 100,
    "model_step": None,
    "dtype": "bfloat16",
}


def _nanochat_preset(*, checkpoint_dir: str, eval_every: int) -> dict[str, Any]:
    return merge_preset(
        _NANOCHAT_DEFAULTS,
        checkpoint_dir=checkpoint_dir,
        eval_every=eval_every,
    )


PRESETS: dict[str, dict[str, Any]] = {
    "csf": _nanochat_preset(
        checkpoint_dir="/mnt/iusers01/fatpou01/compsci01/mbax2an2/.cache/nanochat/base_checkpoints/d26",
        eval_every=60,
    ),
    "isambard": _nanochat_preset(
        checkpoint_dir="/scratch/u6g/nenuadrian.u6g/.cache/nanochat/base_checkpoints/d26/",
        eval_every=100,
    ),
}


def get(env_id: str) -> dict[str, Any]:
    return get_preset(env_id=env_id, presets=PRESETS, algorithm_name="nanochat_rl")
