from __future__ import annotations

import os
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch

from trainers.mpo.agent import MPOAgent, MPOConfig
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


class MPOReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idxs],
            "actions": self.actions[idxs],
            "rewards": self.rewards[idxs],
            "next_obs": self.next_obs[idxs],
            "dones": self.dones[idxs],
        }


class Trainer:
    def __init__(
        self,
        domain: str,
        task: str,
        seed: int,
        device: torch.device,
        hidden_sizes: Tuple[int, int],
        replay_size: int,
    ):
        self.env = make_dm_control_env(domain, task, seed=seed)

        obs_dim = _infer_obs_dim(self.env.observation_space)
        if not isinstance(self.env.action_space, gym.spaces.Box):
            raise ValueError("MPO only supports continuous action spaces.")
        if self.env.action_space.shape is None:
            raise ValueError("Action space has no shape.")
        act_dim = int(np.prod(self.env.action_space.shape))
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high

        self.domain = domain
        self.task = task

        self.agent = MPOAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            action_low=action_low,
            action_high=action_high,
            device=device,
            hidden_sizes=hidden_sizes,
            config=MPOConfig(),
        )

        self.replay = MPOReplayBuffer(obs_dim, act_dim, capacity=replay_size)

        self.episode_return = 0.0
        self.episode_len = 0

    def train(
        self,
        total_steps: int,
        start_steps: int,
        update_after: int,
        batch_size: int,
        updates_per_step: int,
        eval_interval: int,
        save_interval: int,
        out_dir: str,
    ):
        obs, _ = self.env.reset()
        obs = flatten_obs(obs)
        for step in range(1, total_steps + 1):
            if step < start_steps:
                action = self.env.action_space.sample()
            else:
                action = self.agent.act(obs, deterministic=False)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_obs = flatten_obs(next_obs)
            done = float(terminated or truncated)

            self.replay.add(obs, action, reward, next_obs, done)
            obs = next_obs

            self.episode_return += reward
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
                self.episode_return = 0.0
                self.episode_len = 0

            if step >= update_after and self.replay.size >= batch_size:
                for _ in range(updates_per_step):
                    batch = self.replay.sample(batch_size)
                    losses = self.agent.update(batch)
                    log_wandb(losses, step=step)

            if eval_interval > 0 and step % eval_interval == 0:
                metrics = evaluate(
                    self.agent.device, self.agent.policy, self.domain, self.task
                )
                metrics_str = " ".join(f"{k}={v:.3f}" for k, v in metrics.items())
                print(f"step={step} {metrics_str}")
                log_wandb(metrics, step=step)

            if save_interval > 0 and step % save_interval == 0:
                ckpt_path = os.path.join(out_dir, f"mpo_step_{step}.pt")
                torch.save(
                    {
                        "policy": self.agent.policy.state_dict(),
                        "q1": self.agent.q1.state_dict(),
                        "q2": self.agent.q2.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"saved checkpoint: {ckpt_path}")

        self.env.close()
