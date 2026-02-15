from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable

import gymnasium as gym
import imageio
import numpy as np
import torch

from utils.env import infer_obs_dim, set_seed


@dataclass(frozen=True)
class VideoRenderConfig:
    max_steps: int = 1000
    fps: int = 30


def find_latest_checkpoint(out_dir: str, algo: str) -> str:
    """Return the most recent checkpoint for an algo in out_dir.

    Prefers the highest step number from filenames like `{algo}_step_123.pt`.
    Falls back to modification time if step parsing fails.
    """
    pattern = re.compile(rf"^{re.escape(algo)}_step_(\\d+)\\.pt$")

    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"out_dir does not exist: {out_dir}")

    candidates: list[tuple[int | None, float, str]] = []
    for name in os.listdir(out_dir):
        if not name.startswith(f"{algo}") or not name.endswith(".pt"):
            continue
        match = pattern.match(name)
        step = int(match.group(1)) if match else None
        path = os.path.join(out_dir, name)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        candidates.append((step, mtime, path))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found for algo='{algo}' in out_dir='{out_dir}'. "
            f"Expected files like: {algo}_step_50000.pt"
        )

    def sort_key(item: tuple[int | None, float, str]) -> tuple[int, float]:
        step, mtime, _ = item
        # Put parsed steps first; then sort descending by step/mtime.
        return (-(step if step is not None else -1), -mtime)

    candidates.sort(key=sort_key)
    return candidates[0][2]


def _as_tuple_ints(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def _make_env(
    env_id: str,
    *,
    seed: int | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    if env_id.startswith("dm_control/"):
        _, domain, task = env_id.split("/")
        env = gym.make(f"dm_control/{domain}-{task}-v0", render_mode=render_mode)
    else:
        env = gym.make(env_id, render_mode=render_mode)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def build_policy_for_algo(
    *,
    algo: str,
    obs_dim: int,
    act_dim: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    policy_layer_sizes: Iterable[int],
    value_layer_sizes: Iterable[int] | None,
    device: torch.device,
) -> torch.nn.Module:
    policy_sizes = _as_tuple_ints(policy_layer_sizes)
    value_sizes = (
        _as_tuple_ints(value_layer_sizes)
        if value_layer_sizes is not None
        else policy_sizes
    )

    if algo in {"ppo"}:
        from trainers.ppo.trainer import Agent

        policy = Agent(
            obs_dim,
            act_dim,
            policy_layer_sizes=policy_sizes,
            value_layer_sizes=value_sizes,
        )
    elif algo == "mpo":
        from trainers.mpo.agent import DiagonalGaussianPolicy

        policy = DiagonalGaussianPolicy(
            obs_dim,
            act_dim,
            layer_sizes=policy_sizes,
            action_low=action_low,
            action_high=action_high,
        )
    elif algo == "vmpo":
        from trainers.vmpo.gaussian_mlp_policy import SquashedGaussianPolicy

        policy = SquashedGaussianPolicy(
            obs_dim,
            act_dim,
            policy_layer_sizes=policy_sizes,
            value_layer_sizes=value_sizes,
            action_low=action_low,
            action_high=action_high,
        )
    else:
        raise ValueError(f"Unsupported algo for video rendering: {algo}")

    return policy.to(device)


def _extract_action_tensor(action_out: object) -> torch.Tensor:
    if isinstance(action_out, torch.Tensor):
        return action_out
    if isinstance(action_out, (tuple, list)) and len(action_out) > 0:
        first = action_out[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(
        f"Expected action tensor or (action, ...) tuple, got {type(action_out)}"
    )


def _select_deterministic_action(
    policy: torch.nn.Module, obs_t: torch.Tensor
) -> torch.Tensor:
    """Best-effort deterministic action across policy implementations."""
    if hasattr(policy, "act_deterministic"):
        action_out = policy.act_deterministic(obs_t)  # type: ignore[attr-defined]
        return _extract_action_tensor(action_out)

    # PPO style: sample_action(obs, deterministic=True)
    if hasattr(policy, "sample_action"):
        try:
            action_out = policy.sample_action(obs_t, deterministic=True)  # type: ignore[attr-defined]
            return _extract_action_tensor(action_out)
        except TypeError:
            pass

    # VMPO/MPO style: forward -> sample_action(mean, log_std, deterministic=True)
    mean_logstd = policy(obs_t)
    if not (isinstance(mean_logstd, (tuple, list)) and len(mean_logstd) >= 2):
        raise TypeError(
            "Policy forward must return (mean, log_std) when act_deterministic/sample_action(obs, ...) are unavailable."
        )
    mean, log_std = mean_logstd[0], mean_logstd[1]
    if not hasattr(policy, "sample_action"):
        raise AttributeError("Policy has no sample_action method")

    action_out = policy.sample_action(mean, log_std, deterministic=True)  # type: ignore[attr-defined]
    return _extract_action_tensor(action_out)


def render_policy_video(
    *,
    checkpoint_path: str,
    algo: str,
    env_id: str,
    out_path: str,
    seed: int,
    config: VideoRenderConfig = VideoRenderConfig(),
    policy_layer_sizes: Iterable[int] = (256, 256, 256),
    value_layer_sizes: Iterable[int] | None = None,
    device: torch.device | None = None,
    num_attempts: int = 10,
) -> tuple[str, int]:
    """Render a trained checkpoint into an mp4 and return (out_path, num_frames).
    Runs the environment multiple times and picks the attempt with the highest reward.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(int(seed))

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "policy" not in ckpt:
        raise KeyError(
            f"Checkpoint missing 'policy' key: {checkpoint_path}. Keys: {list(ckpt.keys())}"
        )

    def _make_rgb_env(env_seed: int):
        try:
            return _make_env(env_id, seed=env_seed, render_mode="rgb_array")
        except ImportError as e:
            # Common on macOS when MUJOCO_GL is set to egl.
            if "Unable to load EGL library" in str(e):
                os.environ["MUJOCO_GL"] = "glfw"
                return _make_env(env_id, seed=env_seed, render_mode="rgb_array")
            raise

    # Build policy once
    dummy_env = _make_rgb_env(seed)

    if not isinstance(dummy_env.action_space, gym.spaces.Box):
        raise TypeError(
            f"Expected a continuous (Box) action space for video rendering, got {type(dummy_env.action_space)}"
        )
    obs_dim = infer_obs_dim(dummy_env.observation_space)
    if dummy_env.action_space.shape is None:
        raise ValueError("Action space has no shape.")
    act_dim = int(np.prod(dummy_env.action_space.shape))
    action_low = np.asarray(dummy_env.action_space.low, dtype=np.float32)
    action_high = np.asarray(dummy_env.action_space.high, dtype=np.float32)
    policy = build_policy_for_algo(
        algo=algo,
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low=action_low,
        action_high=action_high,
        policy_layer_sizes=policy_layer_sizes,
        value_layer_sizes=value_layer_sizes,
        device=device,
    )
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    dummy_env.close()

    best_reward = float("-inf")
    best_frames = []

    for attempt in range(num_attempts):
        env = _make_rgb_env(seed + attempt)
        frames: list[np.ndarray] = []
        total_reward = 0.0
        obs, _ = env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        for _ in range(int(config.max_steps)):
            frame = env.render()
            if frame is None:
                raise RuntimeError(
                    "env.render() returned None. Ensure render_mode='rgb_array' is supported."
                )
            frames.append(np.asarray(frame))
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_t = _select_deterministic_action(policy, obs_t)
            action = action_t.detach().cpu().numpy().squeeze(0)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, _ = env.step(action)
            obs = np.asarray(obs, dtype=np.float32)
            total_reward += reward
            if terminated or truncated:
                break
        env.close()
        if total_reward > best_reward:
            best_reward = total_reward
            best_frames = frames

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    imageio.mimsave(out_path, list(best_frames), fps=int(config.fps))
    print(f"Best attempt reward: {best_reward:.2f} over {len(best_frames)} frames")
    return out_path, len(best_frames)
