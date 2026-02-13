import random

import gymnasium as gym
import numpy as np
import shimmy
import torch


def _transform_observation(env: gym.Env, fn):
    """Gymnasium compatibility shim across wrapper signatures."""
    try:
        # Newer Gymnasium versions.
        return gym.wrappers.TransformObservation(env, fn)
    except TypeError:
        # Older versions require the transformed observation space explicitly.
        return gym.wrappers.TransformObservation(env, fn, env.observation_space)


def _transform_reward(env: gym.Env, fn):
    """Gymnasium compatibility shim across wrapper signatures."""
    try:
        return gym.wrappers.TransformReward(env, fn)
    except TypeError:
        # Some versions may require reward_range to be passed explicitly.
        return gym.wrappers.TransformReward(env, fn, env.reward_range)


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


def flatten_obs(obs):
    """
    Flatten dm_control / Gymnasium observations.

    Returns:
      - (obs_dim,)            for single-env observations
      - (num_envs, obs_dim)   for vectorised observations
    """
    if isinstance(obs, dict):
        if not obs:
            return np.asarray([], dtype=np.float32)

        parts = []
        for key in sorted(obs.keys()):
            p = np.asarray(obs[key], dtype=np.float32)
            if p.ndim == 0:
                p = p.reshape(1)
            parts.append(p)

        # Infer whether this is vectorised by looking for a shared leading dim
        leading_dims = [p.shape[0] for p in parts if p.ndim >= 2]
        is_vectorised = len(leading_dims) > 0

        if is_vectorised:
            n = leading_dims[0]
            for key, p in zip(sorted(obs.keys()), parts):
                if p.ndim == 1:
                    # Per-env scalar → (N, 1)
                    if p.shape[0] != n:
                        raise ValueError(
                            f"Mismatched batch size for key '{key}': "
                            f"expected {n}, got {p.shape[0]}"
                        )
                else:
                    if p.shape[0] != n:
                        raise ValueError(
                            f"Mismatched batch size for key '{key}': "
                            f"expected {n}, got {p.shape[0]}"
                        )

            flat_parts = [
                p.reshape(n, 1) if p.ndim == 1 else p.reshape(n, -1) for p in parts
            ]
            return np.concatenate(flat_parts, axis=1)

        # Single-env dict
        flat_parts = []
        for key, p in zip(sorted(obs.keys()), parts):
            if p.ndim != 1:
                raise ValueError(
                    f"Unexpected shape for key '{key}' in single-env obs: {p.shape}"
                )
            flat_parts.append(p.reshape(-1))

        return np.concatenate(flat_parts, axis=0)

    arr = np.asarray(obs, dtype=np.float32)

    if arr.ndim == 0:
        # Single scalar
        return arr.reshape(1)

    if arr.ndim == 1:
        # Single env, already flat
        return arr

    # Vectorised or higher-rank: keep batch dim, flatten the rest
    return arr.reshape(arr.shape[0], -1)


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


def make_env(
    env_id: str,
    seed=None,
    render_mode=None,
    *,
    ppo_wrappers: bool = False,
    gamma: float = 0.99,
    normalize_observation: bool = True,
    clip_observation: float | None = 10.0,
    normalize_reward: bool = True,
    clip_reward: float | None = 10.0,
    capture_video: bool = False,
    run_name: str | None = None,
    idx: int = 0,
):
    if env_id.startswith("dm_control/"):
        # Parse dm_control/domain/task
        _, domain, task = env_id.split("/")
        env = gym.make(
            f"dm_control/{domain}-{task}-v0",
            render_mode=render_mode,
        )
    else:
        env = gym.make(env_id, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    if capture_video and run_name is not None and idx == 0:
        env = wrap_record_video(env, f"videos/{run_name}")
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if ppo_wrappers:
        # CleanRL-style wrapper stack for continuous-control PPO.
        if isinstance(env.observation_space, gym.spaces.Dict):
            env = gym.wrappers.FlattenObservation(env)
        if isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.ClipAction(env)
        if normalize_observation:
            env = gym.wrappers.NormalizeObservation(env)
            if clip_observation is not None:
                env = _transform_observation(
                    env,
                    lambda obs: np.clip(obs, -clip_observation, clip_observation),
                )
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            if clip_reward is not None:
                env = _transform_reward(
                    env,
                    lambda reward: np.clip(reward, -clip_reward, clip_reward),
                )

    return env
