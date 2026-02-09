from __future__ import annotations

import os
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch

from trainers.vmpo_parallel.agent import VMPOParallelAgent, VMPOParallelConfig
from utils.env import evaluate, flatten_obs, make_dm_control_env, infer_obs_dim
from utils.wandb_utils import log_wandb


def _compute_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Vectorized discounted returns.

    Args:
        rewards: (T, N)
        dones: (T, N) float/bool, 1 when episode ended at step t.
        last_values: (N,) bootstrap values for obs_{T}.
    """
    if rewards.ndim != 2 or dones.ndim != 2:
        raise ValueError(
            f"rewards/dones must be (T, N); got {rewards.shape=} {dones.shape=}"
        )
    if last_values.ndim != 1:
        raise ValueError(f"last_values must be (N,); got {last_values.shape=}")

    T, N = rewards.shape
    if dones.shape != (T, N):
        raise ValueError("rewards and dones must have the same shape")
    if last_values.shape[0] != N:
        raise ValueError("last_values must match num envs")

    returns = np.zeros_like(rewards, dtype=np.float32)
    R = last_values.astype(np.float32).copy()
    for t in reversed(range(T)):
        R = rewards[t] + gamma * (1.0 - dones[t]) * R
        returns[t] = R
    return returns


def _make_env_fn(domain: str, task: str, seed: int, idx: int):
    def _thunk():
        return make_dm_control_env(domain, task, seed=seed + idx)

    return _thunk


def _flatten_time_env(x: np.ndarray) -> np.ndarray:
    """(T, N, ...) -> (T*N, ...)"""
    return x.reshape(-1, *x.shape[2:])


class Trainer:
    def __init__(
        self,
        domain: str,
        task: str,
        seed: int,
        device: torch.device,
        policy_layer_sizes: Tuple[int, ...],
        num_envs: int,
        rollout_steps: int,
        config: VMPOParallelConfig,
    ):
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = int(num_envs)

        self.env = gym.vector.AsyncVectorEnv(
            [_make_env_fn(domain, task, seed, i) for i in range(self.num_envs)]
        )

        obs_dim = infer_obs_dim(self.env.single_observation_space)
        if not isinstance(self.env.single_action_space, gym.spaces.Box):
            raise ValueError("VMPO only supports continuous action spaces.")
        if self.env.single_action_space.shape is None:
            raise ValueError("Action space has no shape.")
        act_shape = self.env.single_action_space.shape
        act_dim = int(np.prod(act_shape))
        action_low = self.env.single_action_space.low
        action_high = self.env.single_action_space.high
        self.act_shape = act_shape

        self.domain = domain
        self.task = task

        self.agent = VMPOParallelAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            action_low=action_low,
            action_high=action_high,
            device=device,
            policy_layer_sizes=policy_layer_sizes,
            config=config,
        )

        self.rollout_steps = rollout_steps

        # Each buffer entry is shape (N, ...), so stacking gives (T, N, ...)
        self.obs_buf: list[np.ndarray] = []
        self.actions_buf: list[np.ndarray] = []
        self.rewards_buf: list[np.ndarray] = []
        self.dones_buf: list[np.ndarray] = []
        self.values_buf: list[np.ndarray] = []
        self.means_buf: list[np.ndarray] = []
        self.log_stds_buf: list[np.ndarray] = []

        self.episode_return = np.zeros(self.num_envs, dtype=np.float32)

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
        updates_per_step: int,
        eval_interval: int,
        save_interval: int,
        out_dir: str,
    ):
        save_interval *= self.num_envs
        total_steps *= self.num_envs

        obs, _ = self.env.reset()
        obs = flatten_obs(obs)  # (N, obs_dim)

        global_step = 0
        next_eval_at = int(eval_interval) if eval_interval > 0 else None
        next_save_at = int(save_interval) if save_interval > 0 else None

        while global_step < total_steps:
            action, value, mean, log_std = self.agent.act_batch(
                obs, deterministic=False
            )

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_obs = flatten_obs(next_obs)
            done = np.logical_or(terminated, truncated).astype(np.float32)

            self.obs_buf.append(obs)
            self.actions_buf.append(action)
            self.rewards_buf.append(np.asarray(reward, dtype=np.float32))
            self.dones_buf.append(done)
            self.values_buf.append(np.asarray(value, dtype=np.float32).squeeze(-1))
            self.means_buf.append(mean)
            self.log_stds_buf.append(log_std)

            obs = next_obs

            self.episode_return += np.asarray(reward, dtype=np.float32)
            finished = done.astype(bool)
            if np.any(finished):
                finished_returns = self.episode_return[finished]
                print(
                    f"step={global_step + self.num_envs} episodes_finished={int(finished.sum())} "
                    f"episode_return_mean={float(finished_returns.mean()):.2f}"
                )
                log_wandb(
                    {
                        "train/episode_return_mean": float(finished_returns.mean()),
                        "train/episodes_finished": int(finished.sum()),
                    },
                    step=global_step + self.num_envs,
                )
                self.episode_return[finished] = 0.0

            global_step += self.num_envs

            if self._rollout_full():
                last_values = self.agent.value_batch(obs)  # (N,)
                obs_arr = np.stack(self.obs_buf)  # (T, N, obs_dim)
                actions_arr = np.stack(self.actions_buf)  # (T, N, act_dim)
                rewards_arr = np.stack(self.rewards_buf).astype(np.float32)  # (T, N)
                dones_arr = np.stack(self.dones_buf).astype(np.float32)  # (T, N)
                values_arr = np.stack(self.values_buf).astype(np.float32)  # (T, N)
                means_arr = np.stack(self.means_buf)  # (T, N, act_dim)
                log_stds_arr = np.stack(self.log_stds_buf)  # (T, N, act_dim)

                returns_arr = _compute_returns(
                    rewards_arr, dones_arr, last_values, self.agent.config.gamma
                )  # (T, N)
                advantages_arr = returns_arr - values_arr

                T = obs_arr.shape[0]
                group_ids = np.tile(
                    np.arange(self.num_envs, dtype=np.int64),
                    T,
                )

                batch = {
                    "obs": torch.tensor(
                        _flatten_time_env(obs_arr),
                        dtype=torch.float32,
                        device=self.agent.device,
                    ),
                    "actions": torch.tensor(
                        _flatten_time_env(actions_arr),
                        dtype=torch.float32,
                        device=self.agent.device,
                    ),
                    "group_ids": torch.tensor(
                        group_ids,
                        dtype=torch.long,
                        device=self.agent.device,
                    ),
                    "returns": torch.tensor(
                        returns_arr.reshape(-1, 1),
                        dtype=torch.float32,
                        device=self.agent.device,
                    ),
                    "advantages": torch.tensor(
                        advantages_arr.reshape(-1, 1),
                        dtype=torch.float32,
                        device=self.agent.device,
                    ),
                    "old_means": torch.tensor(
                        _flatten_time_env(means_arr),
                        dtype=torch.float32,
                        device=self.agent.device,
                    ),
                    "old_log_stds": torch.tensor(
                        _flatten_time_env(log_stds_arr),
                        dtype=torch.float32,
                        device=self.agent.device,
                    ),
                }

                metrics = {
                    "train/step_without_envs": int(global_step / self.num_envs),
                }
                for _ in range(updates_per_step):
                    metrics = self.agent.update(batch)
                    log_wandb(metrics, step=global_step)
                if metrics:
                    print(metrics)

                self._reset_rollout()

            if next_eval_at is not None and global_step >= next_eval_at:
                metrics = evaluate(
                    self.agent.device, self.agent.policy, self.domain, self.task
                )
                metrics_str = " ".join(f"{k}={v:.3f}" for k, v in metrics.items())
                print(f"step={global_step} {metrics_str}")
                log_wandb(metrics, step=global_step)
                next_eval_at += int(eval_interval)

            if next_save_at is not None and global_step >= next_save_at:
                ckpt_path = os.path.join(out_dir, f"vmpo_step_{global_step}.pt")
                torch.save(
                    {
                        "policy": self.agent.policy.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"saved checkpoint: {ckpt_path}")
                next_save_at += int(save_interval)

        self.env.close()
