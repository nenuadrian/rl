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
    """Flatten a dm_control observation dict/OrderedDict into a 1-D numpy array."""
    if isinstance(obs, dict):
        parts = []
        for key in sorted(obs.keys()):
            parts.append(np.asarray(obs[key], dtype=np.float32).flatten())
        return np.concatenate(parts)
    return np.asarray(obs, dtype=np.float32).flatten()


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


def evaluate(device, policy, domain: str, task: str, n_episodes=10, max_steps=1000):
    env = make_dm_control_env(domain, task)
    returns = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        obs = flatten_obs(obs)
        ep_return = 0.0

        for _ in range(max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                mean, log_std = policy(obs_t)
                action = policy.sample_action(mean, log_std, deterministic=True)
                action = action.cpu().numpy().squeeze(0)

            action = np.clip(
                action,
                env.action_space.low,
                env.action_space.high,
            )

            obs, reward, terminated, truncated, _ = env.step(action)
            obs = flatten_obs(obs)

            ep_return += reward
            if terminated or truncated:
                break

        returns.append(ep_return)

    env.close()

    return {
        "eval/return_mean": float(np.mean(returns)),
        "eval/return_std": float(np.std(returns)),
        "eval/return_min": float(np.min(returns)),
        "eval/return_max": float(np.max(returns)),
    }
