from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch

from trainers.vmpo.agent import VMPOAgent, VMPOConfig
from utils.env import evaluate, flatten_obs, make_dm_control_env
from utils.wandb_utils import log_wandb


def _infer_obs_dim(obs_space: gym.Space) -> int:
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


@dataclass
class RolloutBuffer:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    means: np.ndarray
    log_stds: np.ndarray
    ptr: int
    max_size: int

    @classmethod
    def create(cls, obs_dim: int, act_dim: int, size: int) -> "RolloutBuffer":
        return cls(
            obs=np.zeros((size, obs_dim), dtype=np.float32),
            actions=np.zeros((size, act_dim), dtype=np.float32),
            rewards=np.zeros((size, 1), dtype=np.float32),
            dones=np.zeros((size, 1), dtype=np.float32),
            values=np.zeros((size, 1), dtype=np.float32),
            means=np.zeros((size, act_dim), dtype=np.float32),
            log_stds=np.zeros((size, act_dim), dtype=np.float32),
            ptr=0,
            max_size=size,
        )

    def reset(self) -> None:
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: float,
        value: float,
        mean: np.ndarray,
        log_std: np.ndarray,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.means[self.ptr] = mean
        self.log_stds[self.ptr] = log_std
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.max_size

    def compute_returns(self, last_value: float, gamma: float) -> np.ndarray:
        returns = np.zeros_like(self.rewards)
        R = last_value
        for t in reversed(range(self.max_size)):
            R = self.rewards[t] + gamma * (1.0 - self.dones[t]) * R
            returns[t] = R
        return returns


class Trainer:
    def __init__(
        self,
        domain: str,
        task: str,
        seed: int,
        device: torch.device,
        hidden_sizes: Tuple[int, int],
        rollout_steps: int,
    ):
        self.env = make_dm_control_env(domain, task, seed=seed)

        obs_dim = _infer_obs_dim(self.env.observation_space)
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
            hidden_sizes=hidden_sizes,
            config=VMPOConfig(),
        )

        self.rollout_steps = rollout_steps
        self.buffer = RolloutBuffer.create(obs_dim, act_dim, rollout_steps)

        self.episode_return = 0.0
        self.episode_len = 0

    @staticmethod
    def _build_prev_inputs(
        actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        prev_actions = np.zeros_like(actions)
        prev_rewards = np.zeros_like(rewards)
        for t in range(1, actions.shape[0]):
            if dones[t - 1] > 0.5:
                continue
            prev_actions[t] = actions[t - 1]
            prev_rewards[t] = rewards[t - 1]
        return prev_actions, prev_rewards

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

        prev_action = np.zeros(self.act_shape, dtype=np.float32)
        prev_reward = 0.0
        hidden = self.agent.init_hidden()

        for step in range(1, total_steps + 1):
            action, value, mean, log_std, hidden = self.agent.act(
                obs, prev_action, prev_reward, hidden, deterministic=False
            )

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_obs = flatten_obs(next_obs)
            done = float(terminated or truncated)

            self.buffer.add(
                obs,
                action,
                float(reward),
                float(done),
                float(value),
                mean,
                log_std,
            )

            obs = next_obs
            prev_action = action
            prev_reward = float(reward)

            self.episode_return += float(reward)
            self.episode_len += 1

            if terminated or truncated:
                print(
                    f"step={step} episode_return={self.episode_return:.2f} episode_len={self.episode_len}"
                )
                log_wandb(
                    {
                        "train/episode_return": float(self.episode_return),
                        "train/episode_len": int(self.episode_len),
                    },
                    step=step,
                )
                obs, _ = self.env.reset()
                obs = flatten_obs(obs)
                prev_action = np.zeros(self.act_shape, dtype=np.float32)
                prev_reward = 0.0
                hidden = self.agent.init_hidden()
                self.episode_return = 0.0
                self.episode_len = 0

            if self.buffer.is_full():
                last_value = self.agent.value(
                    obs, prev_action, prev_reward, hidden
                )
                returns = self.buffer.compute_returns(
                    last_value, self.agent.config.gamma
                )
                advantages = returns - self.buffer.values

                prev_actions, prev_rewards = self._build_prev_inputs(
                    self.buffer.actions, self.buffer.rewards, self.buffer.dones
                )

                batch = {
                    "obs": torch.tensor(
                        self.buffer.obs, dtype=torch.float32, device=self.agent.device
                    ),
                    "actions": torch.tensor(
                        self.buffer.actions,
                        dtype=torch.float32,
                        device=self.agent.device,
                    ),
                    "prev_actions": torch.tensor(
                        prev_actions, dtype=torch.float32, device=self.agent.device
                    ),
                    "prev_rewards": torch.tensor(
                        prev_rewards, dtype=torch.float32, device=self.agent.device
                    ),
                    "dones": torch.tensor(
                        self.buffer.dones, dtype=torch.float32, device=self.agent.device
                    ),
                    "returns": torch.tensor(
                        returns, dtype=torch.float32, device=self.agent.device
                    ),
                    "advantages": torch.tensor(
                        advantages, dtype=torch.float32, device=self.agent.device
                    ),
                    "old_means": torch.tensor(
                        self.buffer.means, dtype=torch.float32, device=self.agent.device
                    ),
                    "old_log_stds": torch.tensor(
                        self.buffer.log_stds,
                        dtype=torch.float32,
                        device=self.agent.device,
                    ),
                }

                for _ in range(update_epochs):
                    metrics = self.agent.update(batch)
                    log_wandb(metrics, step=step)

                self.buffer.reset()

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
                        "popart_mean": self.agent.popart.popart_mean.detach().cpu(),
                        "popart_std": self.agent.popart.popart_std.detach().cpu(),
                    },
                    ckpt_path,
                )
                print(f"saved checkpoint: {ckpt_path}")

        self.env.close()
