from __future__ import annotations

import os
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


def find_latest_checkpoint(out_dir: str, algo: str, env_id: str) -> str:
    """Return the newest checkpoint for an env under current run directories.

    Expected layout:
      out_dir/{algo}_{env_slug}-{optimizer}-{adv_type}_{timestamp}/{algo}_best.pt
    Falls back to `{algo}_last.pt` if `{algo}_best.pt` is missing.
    """
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"out_dir does not exist: {out_dir}")

    env_slug = str(env_id).replace("/", "-")
    run_prefix = f"{algo}_{env_slug}-"
    candidates: list[tuple[float, str]] = []

    for name in os.listdir(out_dir):
        run_dir = os.path.join(out_dir, name)
        if not os.path.isdir(run_dir):
            continue
        if not name.startswith(run_prefix):
            continue

        best_path = os.path.join(run_dir, f"{algo}_best.pt")
        last_path = os.path.join(run_dir, f"{algo}_last.pt")
        path = best_path if os.path.isfile(best_path) else last_path
        if not os.path.isfile(path):
            continue

        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        candidates.append((mtime, path))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found for algo='{algo}', env='{env_id}' in out_dir='{out_dir}'. "
            f"Expected run directories with prefix '{run_prefix}' containing "
            f"'{algo}_best.pt' or '{algo}_last.pt'."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


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
    shared_encoder: bool = False,
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
            shared_encoder=bool(shared_encoder),
        )
    else:
        raise ValueError(f"Unsupported algo for video rendering: {algo}")

    return policy.to(device)


def _load_policy_state_from_checkpoint(
    policy: torch.nn.Module, ckpt: object, *, algo: str
) -> None:
    if not isinstance(ckpt, dict):
        raise TypeError(
            f"Checkpoint at render time must be a dict, got {type(ckpt).__name__}"
        )

    if algo == "ppo":
        required_keys = {"actor_mean", "actor_logstd", "critic"}
        missing = sorted(required_keys.difference(ckpt.keys()))
        if missing:
            raise KeyError(
                f"PPO checkpoint missing required keys: {missing}. "
                f"Found keys: {sorted(ckpt.keys())}"
            )

        for attr in ("actor_mean", "actor_logstd", "critic"):
            if not hasattr(policy, attr):
                raise AttributeError(
                    f"Loaded PPO policy is missing expected attribute '{attr}'"
                )

        actor_mean_state = ckpt["actor_mean"]
        critic_state = ckpt["critic"]
        actor_logstd = ckpt["actor_logstd"]
        if not isinstance(actor_mean_state, dict) or not isinstance(critic_state, dict):
            raise TypeError("PPO checkpoint 'actor_mean' and 'critic' must be state dicts")
        if not isinstance(actor_logstd, torch.Tensor):
            raise TypeError(
                "PPO checkpoint 'actor_logstd' must be a torch.Tensor, "
                f"got {type(actor_logstd).__name__}"
            )

        policy.actor_mean.load_state_dict(actor_mean_state)
        policy.critic.load_state_dict(critic_state)
        with torch.no_grad():
            policy.actor_logstd.copy_(actor_logstd.to(policy.actor_logstd.device))
        return

    if algo in {"mpo", "vmpo"}:
        if "policy" not in ckpt:
            raise KeyError(
                f"{algo.upper()} checkpoint missing required key 'policy'. "
                f"Found keys: {sorted(ckpt.keys())}"
            )
        policy_state = ckpt["policy"]
        if not isinstance(policy_state, dict):
            raise TypeError(
                f"{algo.upper()} checkpoint 'policy' must be a state dict, "
                f"got {type(policy_state).__name__}"
            )
        policy.load_state_dict(policy_state)
        return

    raise ValueError(f"Unsupported algo for checkpoint loading: {algo}")


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
    # PPO Agent exposes actor_mean directly and has no nn.Module.forward.
    if hasattr(policy, "actor_mean"):
        action_out = policy.actor_mean(obs_t)  # type: ignore[attr-defined]
        return _extract_action_tensor(action_out)

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
    gif_out_path: str | None = None,
    seed: int,
    config: VideoRenderConfig = VideoRenderConfig(),
    policy_layer_sizes: Iterable[int] = (256, 256, 256),
    value_layer_sizes: Iterable[int] | None = None,
    shared_encoder: bool = False,
    device: torch.device | None = None,
    num_attempts: int = 10,
) -> tuple[str, str, int]:
    """Render a trained checkpoint into mp4+gif and return paths + frame count.
    Runs the environment multiple times and picks the attempt with the highest reward.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(int(seed))

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

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
        shared_encoder=shared_encoder,
        device=device,
    )
    _load_policy_state_from_checkpoint(policy, ckpt, algo=algo)
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
    if gif_out_path is None:
        gif_out_path = os.path.splitext(out_path)[0] + ".gif"
    gif_dir = os.path.dirname(gif_out_path)
    if gif_dir:
        os.makedirs(gif_dir, exist_ok=True)

    imageio.mimsave(out_path, list(best_frames), fps=int(config.fps))

    gif_fps = 5
    gif_max_seconds = 5
    gif_max_frames = max(1, gif_fps * gif_max_seconds)
    gif_frames = list(best_frames[:gif_max_frames])
    imageio.mimsave(gif_out_path, gif_frames, fps=gif_fps)
    print(f"Best attempt reward: {best_reward:.2f} over {len(best_frames)} frames")
    return out_path, gif_out_path, len(best_frames)
