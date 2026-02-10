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


def make_dm_control_env(domain, task, seed=None, render_mode=None):
    """Create a dm_control environment via shimmy's Gymnasium wrapper."""
    env = gym.make(
        f"dm_control/{domain}-{task}-v0",
        render_mode=render_mode,
    )
    if seed is not None:
        env.reset(seed=seed)
        try:
            env.action_space.seed(seed)
        except Exception:
            pass
    return env


def evaluate(
    device,
    policy,
    domain: str,
    task: str,
    n_episodes: int = 10,
    max_steps: int = 1000,
    obs_normalizer=None,
    eval_seed: int = 0,
):
    env = make_dm_control_env(domain, task, seed=eval_seed)
    assert isinstance(env.action_space, gym.spaces.Box)

    policy.eval()
    returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=eval_seed + ep)
        obs = flatten_obs(obs)
        if obs_normalizer is not None:
            obs = obs_normalizer.normalize(obs)

        ep_return = 0.0

        for _ in range(max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                mean, log_std = policy(obs_t)
                action = policy.sample_action(
                    obs=obs_t,
                    mean=mean,
                    log_std=log_std,
                    deterministic=True,
                )
                action = action.cpu().numpy().squeeze(0)

            action = np.clip(
                action,
                env.action_space.low,
                env.action_space.high,
            )

            obs, reward, terminated, truncated, _ = env.step(action)
            obs = flatten_obs(obs)
            if obs_normalizer is not None:
                obs = obs_normalizer.normalize(obs)

            ep_return += reward
            if terminated or truncated:
                break

        returns.append(ep_return)

    policy.train()
    env.close()

    return {
        "eval/return_mean": float(np.mean(returns)),
        "eval/return_std": float(np.std(returns)),
        "eval/return_min": float(np.min(returns)),
        "eval/return_max": float(np.max(returns)),
    }
