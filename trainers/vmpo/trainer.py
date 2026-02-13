from __future__ import annotations

import os
from typing import Dict, Mapping, Tuple

import gymnasium as gym
import numpy as np
import torch

from trainers.vmpo.agent import VMPOAgent, VMPOConfig
from utils.env import flatten_obs, make_env, infer_obs_dim
from utils.wandb_utils import log_wandb


def _format_metrics(metrics: Mapping[str, float]) -> str:
    return ", ".join(
        f"{key}={float(value):.4f}" for key, value in sorted(metrics.items())
    )


class VMPOTrainer:
    def __init__(
        self,
        env_id: str,
        seed: int,
        device: torch.device,
        policy_layer_sizes: Tuple[int, ...],
        value_layer_sizes: Tuple[int, ...],
        rollout_steps: int,
        config: VMPOConfig,
        num_envs: int = 1,
        capture_video: bool = False,
        run_name: str | None = None,
    ):
        self.run_name = run_name
        self.num_envs = int(num_envs)
        self.env_id = env_id
        self.seed = seed
        self.capture_video = bool(capture_video)
        self.video_dir = f"videos/{run_name}"

        env_fns = [
            (
                lambda i=i: self._make_train_env(
                    env_id=env_id,
                    seed=seed + i,
                    env_index=i,
                )
            )
            for i in range(self.num_envs)
        ]
        # Always use vectorized env, even when num_envs == 1.
        self.env = gym.vector.SyncVectorEnv(
            env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
        )
        obs_space = self.env.single_observation_space
        act_space = self.env.single_action_space

        obs_dim = infer_obs_dim(obs_space)
        if not isinstance(self.env.action_space, gym.spaces.Box):
            # for vector envs, act_space above already set
            raise ValueError("VMPO only supports continuous action spaces.")
        if act_space.shape is None:
            raise ValueError("Action space has no shape.")
        act_shape = act_space.shape
        act_dim = int(np.prod(act_shape))
        action_low = getattr(act_space, "low", None)
        action_high = getattr(act_space, "high", None)
        self.act_shape = act_shape

        self.agent = VMPOAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            action_low=action_low,
            action_high=action_high,
            device=device,
            policy_layer_sizes=policy_layer_sizes,
            value_layer_sizes=value_layer_sizes,
            config=config,
        )

        self.rollout_steps = rollout_steps

        self.obs_buf: list[np.ndarray] = []
        self.actions_buf: list[np.ndarray] = []
        self.rewards_buf: list[np.ndarray] = []
        self.dones_buf: list[np.ndarray] = []
        self.values_buf: list[np.ndarray] = []
        self.means_buf: list[np.ndarray] = []
        self.log_stds_buf: list[np.ndarray] = []

        # episode returns per environment
        self.episode_return = np.zeros(self.num_envs, dtype=np.float32)

    def _make_train_env(
        self,
        *,
        env_id: str,
        seed: int,
        env_index: int,
    ):
        env = make_env(
            env_id,
            seed=seed,
        )
        return env

    def _reset_rollout(self) -> None:
        self.obs_buf.clear()
        self.actions_buf.clear()
        self.rewards_buf.clear()
        self.dones_buf.clear()
        self.values_buf.clear()
        self.means_buf.clear()
        self.log_stds_buf.clear()

    def _rollout_full(self) -> bool:
        return len(self.obs_buf) >= self.rollout_steps

    def train(
        self,
        total_steps: int,
        eval_interval: int,
        save_interval: int,
        out_dir: str,
        updates_per_step: int = 1,
    ):
        console_log_interval = max(
            1, min(1_000, eval_interval if eval_interval > 0 else 1_000)
        )
        print(
            "[VMPO] training started: "
            f"total_steps={total_steps}, "
            f"rollout_steps={self.rollout_steps}, "
            f"num_envs={self.num_envs}, "
            f"updates_per_step={int(updates_per_step)}, "
            f"console_log_interval={console_log_interval}"
        )
        interval_metric_sums: Dict[str, float] = {}
        interval_update_count = 0
        interval_episode_count = 0
        interval_episode_sum = 0.0
        interval_episode_min = float("inf")
        interval_episode_max = float("-inf")
        total_update_count = 0

        obs, _ = self.env.reset()
        obs = flatten_obs(obs)

        for step in range(1, total_steps + 1):
            action, value, mean, log_std = self.agent.act(obs, deterministic=False)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_obs = flatten_obs(next_obs)
            done = np.asarray(terminated) | np.asarray(truncated)
            reward = np.asarray(reward, dtype=np.float32)

            # Store step data (works for single- and multi-env)
            self.obs_buf.append(obs)
            self.actions_buf.append(action)
            self.rewards_buf.append(reward)
            self.dones_buf.append(done)
            self.values_buf.append(value)
            self.means_buf.append(mean)
            self.log_stds_buf.append(log_std)

            obs = next_obs

            self.episode_return += reward

            finished_mask = done.astype(bool)
            if np.any(finished_mask):
                finished_indices = np.flatnonzero(finished_mask)
                finished_returns = self.episode_return[finished_mask]
                self.episode_return[finished_mask] = 0.0
                for env_i, episode_return in zip(finished_indices, finished_returns):
                    episode_return_f = float(episode_return)
                    log_wandb(
                        {"train/episode_return": episode_return_f},
                        step=step,
                        silent=True,
                    )
                    print(
                        f"[VMPO][episode] step={step}/{total_steps}, "
                        f"env={int(env_i)}, return={episode_return_f:.3f}"
                    )
                    interval_episode_count += 1
                    interval_episode_sum += episode_return_f
                    interval_episode_min = min(interval_episode_min, episode_return_f)
                    interval_episode_max = max(interval_episode_max, episode_return_f)

            if self._rollout_full():
                # Stack collected arrays and reshape for batch processing
                obs_arr = np.stack(self.obs_buf)
                actions_arr = np.stack(self.actions_buf)
                rewards_arr = np.asarray(self.rewards_buf, dtype=np.float32)
                dones_arr = np.asarray(self.dones_buf, dtype=np.float32)
                values_arr = np.asarray(self.values_buf, dtype=np.float32)
                means_arr = np.stack(self.means_buf)
                log_stds_arr = np.stack(self.log_stds_buf)

                last_value = self.agent.value(obs)
                last_value = last_value * (1.0 - dones_arr[-1])

                # obs_arr shape (T, N, obs_dim) -> flatten to (T*N, obs_dim)
                T, N, _ = obs_arr.shape
                obs_flat = obs_arr.reshape(T * N, -1)
                actions_flat = actions_arr.reshape(T * N, -1)
                rewards_flat = rewards_arr.reshape(T, N)
                dones_flat = dones_arr.reshape(T, N)
                values_flat = values_arr.reshape(T, N)
                means_flat = means_arr.reshape(T * N, -1)
                log_stds_flat = log_stds_arr.reshape(T * N, -1)

                returns = _compute_returns(
                    rewards_flat, dones_flat, last_value, self.agent.config.gamma
                )
                returns_flat = returns.reshape(T * N, 1)
                values_flat2 = values_flat.reshape(T * N, 1)

                advantages = returns_flat - values_flat2
                # Zero advantages where done == 1
                advantages[dones_flat.reshape(-1) == 1.0] = 0.0

                batch = {
                    "obs": torch.tensor(
                        obs_flat, dtype=torch.float32, device=self.agent.device
                    ),
                    "actions": torch.tensor(
                        actions_flat, dtype=torch.float32, device=self.agent.device
                    ),
                    "returns": torch.tensor(
                        returns_flat, dtype=torch.float32, device=self.agent.device
                    ),
                    "advantages": torch.tensor(
                        advantages, dtype=torch.float32, device=self.agent.device
                    ),
                    "old_means": torch.tensor(
                        means_flat, dtype=torch.float32, device=self.agent.device
                    ),
                    "old_log_stds": torch.tensor(
                        log_stds_flat, dtype=torch.float32, device=self.agent.device
                    ),
                }

                metrics = {}
                for _ in range(updates_per_step):
                    metrics = self.agent.update(batch)
                    log_wandb(metrics, step=step)
                    for key, value in metrics.items():
                        interval_metric_sums[key] = interval_metric_sums.get(
                            key, 0.0
                        ) + float(value)
                    interval_update_count += 1
                    total_update_count += 1

                self._reset_rollout()

            if eval_interval > 0 and step % eval_interval == 0:
                metrics = _evaluate_vectorized(
                    agent=self.agent,
                    env_id=self.env_id,
                    seed=self.seed + 1000,
                    capture_video=self.capture_video,
                    run_name=self.run_name,
                )
                log_wandb(metrics, step=step)
                print(
                    f"[VMPO][eval] step={step}/{total_steps}: {_format_metrics(metrics)}"
                )

            if save_interval > 0 and step % save_interval == 0:
                ckpt_path = os.path.join(out_dir, f"vmpo.pt")
                os.makedirs(out_dir, exist_ok=True)
                torch.save({"policy": self.agent.policy.state_dict()}, ckpt_path)
                print(f"[VMPO][checkpoint] step={step}: saved {ckpt_path}")

            should_print_progress = (
                step == total_steps or step % console_log_interval == 0
            )
            if should_print_progress:
                progress = 100.0 * float(step) / float(total_steps)
                print(
                    "[VMPO][progress] "
                    f"step={step}/{total_steps} ({progress:.2f}%), "
                    f"updates={total_update_count}"
                )
                if interval_episode_count > 0:
                    print(
                        "[VMPO][episode-stats] "
                        f"count={interval_episode_count}, "
                        f"return_mean={interval_episode_sum / interval_episode_count:.3f}, "
                        f"return_min={interval_episode_min:.3f}, "
                        f"return_max={interval_episode_max:.3f}"
                    )
                    interval_episode_count = 0
                    interval_episode_sum = 0.0
                    interval_episode_min = float("inf")
                    interval_episode_max = float("-inf")
                if interval_update_count > 0:
                    mean_metrics = {
                        key: value / float(interval_update_count)
                        for key, value in interval_metric_sums.items()
                    }
                    print(
                        "[VMPO][train-metrics] "
                        f"updates={interval_update_count}, {_format_metrics(mean_metrics)}"
                    )
                    interval_metric_sums.clear()
                    interval_update_count = 0

        self.env.close()


def _compute_returns(
    rewards: np.ndarray, dones: np.ndarray, last_value: np.ndarray, gamma: float
) -> np.ndarray:
    """
    Vectorised return computation.

    rewards: shape (T, N) or (T,) for single-env
    dones: shape (T, N) or (T,)
    last_value: shape (N,) or scalar
    Returns: shape (T, N) or (T,)
    """
    rewards_np = np.asarray(rewards)
    dones_np = np.asarray(dones)

    # Ensure 2D (T, N)
    if rewards_np.ndim == 1:
        rewards_np = rewards_np.reshape(-1, 1)
    if dones_np.ndim == 1:
        dones_np = dones_np.reshape(-1, 1)

    T, N = rewards_np.shape

    returns = np.zeros_like(rewards_np, dtype=np.float32)

    # Last value per environment
    R = np.zeros(N, dtype=np.float32)
    last_val_arr = np.asarray(last_value)
    if last_val_arr.ndim == 0:
        R[:] = float(last_val_arr)
    else:
        R[:] = last_val_arr

    for t in reversed(range(T)):
        R = rewards_np[t] + gamma * (1.0 - dones_np[t]) * R
        returns[t] = R

    # If original input was 1D, return 1D
    if returns.shape[1] == 1:
        return returns.reshape(-1)
    return returns


@torch.no_grad()
def _evaluate_vectorized(
    agent: VMPOAgent,
    env_id: str,
    n_episodes: int = 10,
    seed: int = 42,
    obs_normalizer=None,
    capture_video: bool = False,
    run_name: str | None = None,
) -> Dict[str, float]:
    """
    High-performance vectorized evaluation.
    Runs all n_episodes in parallel using a SyncVectorEnv.
    """
    eval_envs = gym.vector.SyncVectorEnv(
        [
            lambda i=i: make_env(
                env_id,
                seed=seed + i,
                capture_video=capture_video,
                render_mode="rgb_array" if capture_video else None,
                run_name=run_name,
                idx=i,
            )
            for i in range(n_episodes)
        ]
    )

    agent.policy.eval()
    obs, _ = eval_envs.reset(seed=seed)

    episode_returns = np.zeros(n_episodes)
    final_returns = []

    # Track which envs have finished their first episode
    dones = np.zeros(n_episodes, dtype=bool)

    while len(final_returns) < n_episodes:
        # Pre-process observations
        obs = flatten_obs(obs)
        if obs_normalizer is not None:
            obs = obs_normalizer.normalize(obs)

        # Deterministic Action Selection
        # We call act() with deterministic=True to use the mean of the Gaussian
        action, _, _, _ = agent.act(obs, deterministic=True)

        # Clip actions to valid range
        action = np.clip(
            action,
            eval_envs.single_action_space.low,
            eval_envs.single_action_space.high,
        )

        # Step environment
        next_obs, reward, terminated, truncated, infos = eval_envs.step(action)

        # Accumulate rewards for envs that haven't finished yet
        episode_returns += reward

        # Check for completions
        # Gymnasium VectorEnv resets automatically; we catch the return in infos
        for i in range(n_episodes):
            if not dones[i] and (terminated[i] or truncated[i]):
                final_returns.append(episode_returns[i])
                dones[i] = True

        obs = next_obs
    eval_envs.close()
    agent.policy.train()

    return {
        "eval/return_mean": float(np.mean(final_returns)),
        "eval/return_std": float(np.std(final_returns)),
        "eval/return_min": float(np.min(final_returns)),
        "eval/return_max": float(np.max(final_returns)),
    }
