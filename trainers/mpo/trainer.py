from __future__ import annotations

import os
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch

from trainers.mpo.agent import MPOAgent, MPOConfig
from utils.env import evaluate, flatten_obs, make_dm_control_env, infer_obs_dim
from utils.wandb_utils import log_wandb


class MPOReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self._global_step = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions_exec = np.zeros((capacity, act_dim), dtype=np.float32)
        self.actions_raw = np.zeros((capacity, act_dim), dtype=np.float32)
        self.behaviour_logp = np.zeros((capacity, 1), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.step_ids = np.zeros((capacity,), dtype=np.int64)

    def add(
        self,
        obs: np.ndarray,
        action_exec: np.ndarray,
        action_raw: np.ndarray,
        behaviour_logp: float,
        reward: float,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions_exec[self.ptr] = action_exec
        self.actions_raw[self.ptr] = action_raw
        self.behaviour_logp[self.ptr] = behaviour_logp
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.step_ids[self.ptr] = self._global_step
        self._global_step += 1
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idxs],
            "actions": self.actions_exec[idxs],
            "rewards": self.rewards[idxs],
            "next_obs": self.next_obs[idxs],
            "dones": self.dones[idxs],
        }

    def sample_sequences(self, batch_size: int, seq_len: int) -> dict:
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if self.size < seq_len:
            raise ValueError("Not enough data in replay buffer for sequence sampling")

        obs_dim = self.obs.shape[-1]
        act_dim = self.actions_exec.shape[-1]

        obs_b = np.zeros((batch_size, seq_len, obs_dim), dtype=np.float32)
        next_obs_b = np.zeros((batch_size, seq_len, obs_dim), dtype=np.float32)
        actions_exec_b = np.zeros((batch_size, seq_len, act_dim), dtype=np.float32)
        actions_raw_b = np.zeros((batch_size, seq_len, act_dim), dtype=np.float32)
        rewards_b = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
        dones_b = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
        beh_logp_b = np.zeros((batch_size, seq_len, 1), dtype=np.float32)

        filled = 0
        max_tries = batch_size * 200
        tries = 0

        if self.size < self.capacity:
            starts = np.random.randint(0, self.size - seq_len + 1, size=batch_size)
            for i, start in enumerate(starts):
                idxs = np.arange(start, start + seq_len)
                obs_b[i] = self.obs[idxs]
                next_obs_b[i] = self.next_obs[idxs]
                actions_exec_b[i] = self.actions_exec[idxs]
                actions_raw_b[i] = self.actions_raw[idxs]
                rewards_b[i] = self.rewards[idxs]
                dones_b[i] = self.dones[idxs]
                beh_logp_b[i] = self.behaviour_logp[idxs]
            return {
                "obs": obs_b,
                "actions_exec": actions_exec_b,
                "actions_raw": actions_raw_b,
                "behaviour_logp": beh_logp_b,
                "rewards": rewards_b,
                "next_obs": next_obs_b,
                "dones": dones_b,
            }

        while filled < batch_size and tries < max_tries:
            tries += 1
            start = np.random.randint(0, self.capacity)
            idxs = (start + np.arange(seq_len)) % self.capacity
            step_ids = self.step_ids[idxs]
            if not np.all(step_ids[1:] == step_ids[:-1] + 1):
                continue

            obs_b[filled] = self.obs[idxs]
            next_obs_b[filled] = self.next_obs[idxs]
            actions_exec_b[filled] = self.actions_exec[idxs]
            actions_raw_b[filled] = self.actions_raw[idxs]
            rewards_b[filled] = self.rewards[idxs]
            dones_b[filled] = self.dones[idxs]
            beh_logp_b[filled] = self.behaviour_logp[idxs]
            filled += 1

        if filled < batch_size:
            raise RuntimeError(
                "Failed to sample enough contiguous sequences; consider increasing replay size or reducing seq_len."
            )

        return {
            "obs": obs_b,
            "actions_exec": actions_exec_b,
            "actions_raw": actions_raw_b,
            "behaviour_logp": beh_logp_b,
            "rewards": rewards_b,
            "next_obs": next_obs_b,
            "dones": dones_b,
        }


class Trainer:
    def __init__(
        self,
        domain: str,
        task: str,
        seed: int,
        device: torch.device,
        policy_layer_sizes: Tuple[int, ...],
        critic_layer_sizes: Tuple[int, ...],
        replay_size: int,
        config: MPOConfig,
    ):
        self.env = make_dm_control_env(domain, task, seed=seed)

        obs_dim = infer_obs_dim(self.env.observation_space)
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
            policy_layer_sizes=policy_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            config=config,
        )

        self.replay = MPOReplayBuffer(obs_dim, act_dim, capacity=replay_size)

        self.episode_return = 0.0

    def train(
        self,
        total_steps: int,
        update_after: int,
        batch_size: int,
        eval_interval: int,
        save_interval: int,
        out_dir: str,
        updates_per_step: int = 1,
    ):
        obs, _ = self.env.reset()
        obs = flatten_obs(obs)
        for step in range(1, total_steps + 1):
            action_exec, action_raw, behaviour_logp = self.agent.act_with_logp(
                obs, deterministic=False
            )

            next_obs, reward, terminated, truncated, _ = self.env.step(action_exec)
            next_obs = flatten_obs(next_obs)
            reward_f = float(reward)
            done = float(terminated or truncated)

            self.replay.add(
                obs,
                action_exec,
                action_raw,
                behaviour_logp,
                reward_f,
                next_obs,
                done,
            )
            obs = next_obs

            self.episode_return += reward_f

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

            if step >= update_after and self.replay.size >= batch_size:
                for _ in range(int(updates_per_step)):
                    seq_len = self.agent.config.retrace_steps
                    if self.agent.config.use_retrace and seq_len > 1:
                        if self.replay.size >= batch_size + seq_len:
                            batch = self.replay.sample_sequences(
                                batch_size, seq_len=seq_len
                            )
                        else:
                            continue
                    else:
                        batch = self.replay.sample(batch_size)

                    metrics = self.agent.update(batch)
                    log_wandb(metrics, step=step)

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
