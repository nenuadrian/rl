from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Tuple

import gymnasium as gym
import numpy as np
import torch

from trainers.ppo.agent import PPOAgent, PPOConfig
from utils.env import evaluate, flatten_obs, make_dm_control_env, infer_obs_dim
from utils.obs_normalizer import ObsNormalizer
from utils.wandb_utils import log_wandb


@dataclass
class RolloutBuffer:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    log_probs: np.ndarray
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
            log_probs=np.zeros((size, 1), dtype=np.float32),
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
        log_prob: float,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.max_size

    def compute_returns_advantages(
        self, last_value: float, gamma: float, gae_lambda: float
    ) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(self.rewards)
        last_gae = 0.0
        for t in reversed(range(self.max_size)):
            if t == self.max_size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        return returns, advantages

    def minibatches(self, batch_size: int) -> Iterator[tuple[np.ndarray, ...]]:
        indices = np.arange(self.max_size)
        np.random.shuffle(indices)
        for start in range(0, self.max_size, batch_size):
            batch_idx = indices[start : start + batch_size]
            yield (
                self.obs[batch_idx],
                self.actions[batch_idx],
                self.log_probs[batch_idx],
                self.values[batch_idx],
                batch_idx,
            )


class Trainer:
    def __init__(
        self,
        domain: str,
        task: str,
        seed: int,
        device: torch.device,
        policy_layer_sizes: Tuple[int, ...],
        critic_layer_sizes: Tuple[int, ...],
        rollout_steps: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        update_epochs: int = 10,
        minibatch_size: int = 64,
        policy_lr: float = 3e-4,
        value_lr: float = 1e-4,
        clip_ratio: float = 0.2,
        ent_coef: float = 1e-3,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.02,
        normalize_obs: bool = False,
    ):
        self.env = make_dm_control_env(domain, task, seed=seed)

        obs_dim = infer_obs_dim(self.env.observation_space)
        if not isinstance(self.env.action_space, gym.spaces.Box):
            raise ValueError("PPO only supports continuous action spaces.")
        if self.env.action_space.shape is None:
            raise ValueError("Action space has no shape.")
        act_dim = int(np.prod(self.env.action_space.shape))
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high

        self.domain = domain
        self.task = task

        self.agent = PPOAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            action_low=action_low,
            action_high=action_high,
            device=device,
            policy_layer_sizes=policy_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            config=PPOConfig(
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_ratio=clip_ratio,
                policy_lr=policy_lr,
                value_lr=value_lr,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                target_kl=target_kl,
            ),
        )

        self.rollout_steps = int(rollout_steps)
        self.update_epochs = int(update_epochs)
        self.minibatch_size = int(minibatch_size)

        self.normalize_obs = bool(normalize_obs)
        self.obs_normalizer = ObsNormalizer(obs_dim) if self.normalize_obs else None

        self.buffer = RolloutBuffer.create(obs_dim, act_dim, self.rollout_steps)

        self.episode_return = 0.0

    def train(
        self,
        total_steps: int,
        eval_interval: int,
        save_interval: int,
        out_dir: str,
        batch_size: int | None = None,
        update_epochs: int | None = None,
        minibatch_size: int | None = None,
        policy_lr: float | None = None,
        value_lr: float | None = None,
        clip_ratio: float | None = None,
        ent_coef: float | None = None,
        vf_coef: float | None = None,
        max_grad_norm: float | None = None,
        target_kl: float | None = None,
    ):
        obs, _ = self.env.reset()
        obs = flatten_obs(obs)
        if self.obs_normalizer is not None:
            self.obs_normalizer.update(obs)
            obs = self.obs_normalizer.normalize(obs)

        if policy_lr is not None or value_lr is not None or clip_ratio is not None:
            self.agent.set_hparams(
                clip_ratio=clip_ratio,
                policy_lr=policy_lr,
                value_lr=value_lr,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                target_kl=target_kl,
            )

        effective_update_epochs = (
            int(update_epochs) if update_epochs is not None else self.update_epochs
        )
        if minibatch_size is not None:
            effective_minibatch_size = int(minibatch_size)
        elif batch_size is not None:
            effective_minibatch_size = int(batch_size)
        else:
            effective_minibatch_size = self.minibatch_size

        for step in range(1, total_steps + 1):
            obs_t = (
                torch.tensor(obs, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.agent.device)
            )
            action_t, log_prob_t, value_t = self.agent.act(obs_t, deterministic=False)
            action = action_t.cpu().numpy().squeeze(0)
            log_prob = float(log_prob_t.item())
            value = float(value_t.item())

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_obs = flatten_obs(next_obs)
            if self.obs_normalizer is not None:
                self.obs_normalizer.update(next_obs)
                next_obs = self.obs_normalizer.normalize(next_obs)
            done = float(terminated)

            self.buffer.add(
                obs,
                action,
                float(reward),
                float(done),
                float(value),
                float(log_prob),
            )
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
                if self.obs_normalizer is not None:
                    self.obs_normalizer.update(obs)
                    obs = self.obs_normalizer.normalize(obs)
                self.episode_return = 0.0

            if self.buffer.is_full():
                with torch.no_grad():
                    obs_t = (
                        torch.tensor(obs, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(self.agent.device)
                    )
                    last_value = self.agent.value(obs_t).cpu().numpy().squeeze(0)

                returns, advantages = self.buffer.compute_returns_advantages(
                    last_value, self.agent.config.gamma, self.agent.config.gae_lambda
                )
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                early_stop = False
                for _ in range(effective_update_epochs):
                    for (
                        obs_b,
                        actions_b,
                        log_probs_b,
                        values_old_b,
                        idx_b,
                    ) in self.buffer.minibatches(effective_minibatch_size):
                        batch = {
                            "obs": torch.tensor(
                                obs_b, dtype=torch.float32, device=self.agent.device
                            ),
                            "actions": torch.tensor(
                                actions_b,
                                dtype=torch.float32,
                                device=self.agent.device,
                            ),
                            "log_probs": torch.tensor(
                                log_probs_b,
                                dtype=torch.float32,
                                device=self.agent.device,
                            ),
                            "values_old": torch.tensor(
                                values_old_b,
                                dtype=torch.float32,
                                device=self.agent.device,
                            ),
                            "returns": torch.tensor(
                                returns[idx_b],
                                dtype=torch.float32,
                                device=self.agent.device,
                            ),
                            "advantages": torch.tensor(
                                advantages[idx_b],
                                dtype=torch.float32,
                                device=self.agent.device,
                            ),
                        }
                        metrics = self.agent.update(batch)
                        log_wandb(metrics, step=step)

                        if metrics.get("approx_kl", 0.0) > self.agent.config.target_kl:
                            early_stop = True
                            break

                    if early_stop:
                        break

                self.buffer.reset()

            if eval_interval > 0 and step % eval_interval == 0:
                metrics = evaluate(
                    self.agent.device,
                    self.agent.policy,
                    self.domain,
                    self.task,
                    obs_normalizer=self.obs_normalizer,
                )
                metrics_str = " ".join(f"{k}={v:.3f}" for k, v in metrics.items())
                print(f"step={step} {metrics_str}")
                log_wandb(metrics, step=step)

            if save_interval > 0 and step % save_interval == 0:
                ckpt_path = os.path.join(out_dir, f"ppo_step_{step}.pt")
                torch.save(
                    {
                        "policy": self.agent.policy.state_dict(),
                        "value": self.agent.value.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"saved checkpoint: {ckpt_path}")

        self.env.close()
