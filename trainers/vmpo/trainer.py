from __future__ import annotations

import os
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch

from trainers.vmpo.agent import VMPOAgent, VMPOConfig
from utils.env import evaluate, flatten_obs, make_dm_control_env, infer_obs_dim
from utils.wandb_utils import log_wandb


def _compute_returns(
    rewards: np.ndarray, dones: np.ndarray, last_value: float, gamma: float
) -> np.ndarray:
    returns = np.zeros_like(rewards)
    R = last_value
    for t in reversed(range(rewards.shape[0])):
        R = rewards[t] + gamma * (1.0 - dones[t]) * R
        returns[t] = R
    return returns


class Trainer:
    def __init__(
        self,
        domain: str,
        task: str,
        seed: int,
        device: torch.device,
        policy_layer_sizes: Tuple[int, ...],
        rollout_steps: int,
        config: VMPOConfig,
    ):
        self.env = make_dm_control_env(domain, task, seed=seed)

        obs_dim = infer_obs_dim(self.env.observation_space)
        if not isinstance(self.env.action_space, gym.spaces.Box):
            raise ValueError("VMPO only supports continuous action spaces.")
        if self.env.action_space.shape is None:
            raise ValueError("Action space has no shape.")
        act_shape = self.env.action_space.shape
        act_dim = int(np.prod(act_shape))
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high
        self.act_shape = act_shape

        self.domain = domain
        self.task = task

        self.agent = VMPOAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            action_low=action_low,
            action_high=action_high,
            device=device,
            policy_layer_sizes=policy_layer_sizes,
            config=config,
        )

        self.rollout_steps = rollout_steps

        self.obs_buf: list[np.ndarray] = []
        self.actions_buf: list[np.ndarray] = []
        self.rewards_buf: list[float] = []
        self.dones_buf: list[float] = []
        self.values_buf: list[float] = []
        self.means_buf: list[np.ndarray] = []
        self.log_stds_buf: list[np.ndarray] = []

        self.episode_return = 0.0

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
        update_epochs: int,
        eval_interval: int,
        save_interval: int,
        out_dir: str,
    ):
        obs, _ = self.env.reset()
        obs = flatten_obs(obs)

        for step in range(1, total_steps + 1):
            action, value, mean, log_std = self.agent.act(obs, deterministic=False)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_obs = flatten_obs(next_obs)
            done = float(terminated)

            self.obs_buf.append(obs)
            self.actions_buf.append(action)
            self.rewards_buf.append(float(reward))
            self.dones_buf.append(float(done))
            self.values_buf.append(float(value))
            self.means_buf.append(mean)
            self.log_stds_buf.append(log_std)

            obs = next_obs

            self.episode_return += float(reward)

            if terminated or truncated:
                print(f"step={step} episode_return={self.episode_return:.2f}")
                log_wandb(
                    {
                        "train/episode_return": float(self.episode_return),
                    },
                    step=step,
                )
                obs, _ = self.env.reset()
                obs = flatten_obs(obs)
                self.episode_return = 0.0

            if self._rollout_full():
                last_value = self.agent.value(obs)
                obs_arr = np.stack(self.obs_buf)
                actions_arr = np.stack(self.actions_buf)
                rewards_arr = np.asarray(self.rewards_buf, dtype=np.float32).reshape(
                    -1, 1
                )
                dones_arr = np.asarray(self.dones_buf, dtype=np.float32).reshape(-1, 1)
                values_arr = np.asarray(self.values_buf, dtype=np.float32).reshape(
                    -1, 1
                )
                means_arr = np.stack(self.means_buf)
                log_stds_arr = np.stack(self.log_stds_buf)

                returns = _compute_returns(
                    rewards_arr, dones_arr, last_value, self.agent.config.gamma
                )
                advantages = returns - values_arr

                batch = {
                    "obs": torch.tensor(
                        obs_arr, dtype=torch.float32, device=self.agent.device
                    ),
                    "actions": torch.tensor(
                        actions_arr,
                        dtype=torch.float32,
                        device=self.agent.device,
                    ),
                    "returns": torch.tensor(
                        returns, dtype=torch.float32, device=self.agent.device
                    ),
                    "advantages": torch.tensor(
                        advantages, dtype=torch.float32, device=self.agent.device
                    ),
                    "old_means": torch.tensor(
                        means_arr, dtype=torch.float32, device=self.agent.device
                    ),
                    "old_log_stds": torch.tensor(
                        log_stds_arr,
                        dtype=torch.float32,
                        device=self.agent.device,
                    ),
                }

                metrics = {}
                for _ in range(update_epochs):
                    metrics = self.agent.update(batch)
                    log_wandb(metrics, step=step)
                if metrics:
                    print(metrics)

                self._reset_rollout()

            if eval_interval > 0 and step % eval_interval == 0:
                metrics = evaluate(
                    self.agent.device, self.agent.policy, self.domain, self.task
                )
                metrics_str = " ".join(f"{k}={v:.3f}" for k, v in metrics.items())
                print(f"step={step} {metrics_str}")
                log_wandb(metrics, step=step)

            if save_interval > 0 and step % save_interval == 0:
                ckpt_path = os.path.join(out_dir, f"vmpo_step_{step}.pt")
                torch.save(
                    {
                        "policy": self.agent.policy.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"saved checkpoint: {ckpt_path}")

        self.env.close()
