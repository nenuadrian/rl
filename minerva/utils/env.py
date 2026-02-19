import random

import gymnasium as gym
import numpy as np
import shimmy
import torch


def infer_obs_dim(obs_space: gym.Space) -> int:
    if isinstance(obs_space, gym.spaces.Dict):
        dims = []
        for v in obs_space.spaces.values():
            if v.shape is None:
                raise ValueError("Observation space has no shape.")
            dims.append(int(np.prod(v.shape)))
        return int(sum(dims))
    if obs_space.shape is None:
        raise ValueError("Observation space has no shape.")
    return int(np.prod(obs_space.shape))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def wrap_record_video(env: gym.Env, video_dir: str) -> gym.Env:
    """Wrap env with RecordVideo and patch compatibility attrs for wandb monitor."""
    wrapped_env = gym.wrappers.RecordVideo(env, video_dir)

    # wandb monitor_gym still expects pre-1.0 gymnasium fields on RecordVideo.
    # gymnasium>=1.0 removed `enabled`; setting it avoids teardown crashes.
    if not hasattr(wrapped_env, "enabled"):
        setattr(wrapped_env, "enabled", False)

    return wrapped_env
