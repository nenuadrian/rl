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


# =========================
# Rollout buffer (time-major, vectorised-safe)
# =========================


@dataclass
class RolloutBuffer:
    obs: np.ndarray  # (T, N, obs_dim)
    actions: np.ndarray  # (T, N, act_dim)
    rewards: np.ndarray  # (T, N)
    dones: np.ndarray  # (T, N)
    values: np.ndarray  # (T, N)
    log_probs: np.ndarray  # (T, N)
    T: int
    N: int
    ptr: int = 0

    @classmethod
    def create(
        cls, obs_dim: int, act_dim: int, rollout_steps: int, num_envs: int
    ) -> "RolloutBuffer":
        return cls(
            obs=np.zeros((rollout_steps, num_envs, obs_dim), dtype=np.float32),
            actions=np.zeros((rollout_steps, num_envs, act_dim), dtype=np.float32),
            rewards=np.zeros((rollout_steps, num_envs), dtype=np.float32),
            dones=np.zeros((rollout_steps, num_envs), dtype=np.float32),
            values=np.zeros((rollout_steps, num_envs), dtype=np.float32),
            log_probs=np.zeros((rollout_steps, num_envs), dtype=np.float32),
            T=rollout_steps,
            N=num_envs,
            ptr=0,
        )

    def reset(self) -> None:
        self.ptr = 0

    def add(
        self,
        t: int,
        env_i: int,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: float,
        value: float,
        log_prob: float,
    ) -> None:
        self.obs[t, env_i] = obs
        self.actions[t, env_i] = action
        self.rewards[t, env_i] = reward
        self.dones[t, env_i] = done
        self.values[t, env_i] = value
        self.log_probs[t, env_i] = log_prob

    def compute_returns_advantages(
        self,
        last_values: np.ndarray,  # shape (N,)
        gamma: float,
        gae_lambda: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T, N = self.T, self.N
        advantages = np.zeros((T, N), dtype=np.float32)
        last_gae = np.zeros(N, dtype=np.float32)

        for t in reversed(range(T)):
            if t == T - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]

            non_terminal = 1.0 - self.dones[t]
            delta = (
                self.rewards[t] + gamma * next_values * non_terminal - self.values[t]
            )
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values

        # flatten (T, N) -> (T*N)
        return (
            self.obs.reshape(T * N, -1),
            self.actions.reshape(T * N, -1),
            self.log_probs.reshape(T * N),
            self.values.reshape(T * N),
            returns.reshape(T * N),
            advantages.reshape(T * N),
        )

    def minibatches(self, batch_size: int) -> Iterator[np.ndarray]:
        idx = np.arange(self.T * self.N)
        np.random.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            yield idx[start : start + batch_size]


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
        num_envs: int = 1,
    ):
        self.num_envs = int(num_envs)

        if self.num_envs > 1:
            env_fns = [
                (lambda i=i: make_dm_control_env(domain, task, seed=seed + i))
                for i in range(self.num_envs)
            ]
            self.env = gym.vector.AsyncVectorEnv(env_fns)
            obs_space = self.env.single_observation_space
            act_space = self.env.single_action_space
        else:
            self.env = make_dm_control_env(domain, task, seed=seed)
            obs_space = self.env.observation_space
            act_space = self.env.action_space

        obs_dim = infer_obs_dim(obs_space)
        act_dim = int(np.prod(act_space.shape))

        self.agent = PPOAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            action_low=act_space.low,
            action_high=act_space.high,
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

        self.rollout_steps = rollout_steps
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

        self.obs_normalizer = ObsNormalizer(obs_dim) if normalize_obs else None
        self.buffer = RolloutBuffer.create(
            obs_dim, act_dim, rollout_steps, self.num_envs
        )
        self.episode_return = np.zeros(self.num_envs, dtype=np.float32)

        self.domain = domain
        self.task = task
        
        self.last_eval = 0
        self.last_checkpoint = 0

    def train(
        self,
        total_steps: int,
        eval_interval: int,
        save_interval: int,
        out_dir: str,
    ):
        if self.num_envs > 1:
            obs, _ = self.env.reset()
            # obs is a dict mapping -> per-key arrays with leading dim N
            obs = flatten_obs(obs)  
            eval_interval *= self.num_envs  
            save_interval *= self.num_envs  
        else:
            obs, _ = self.env.reset()
            obs = flatten_obs(obs)

        if self.obs_normalizer:
            self.obs_normalizer.update(obs)
            obs = self.obs_normalizer.normalize(obs)

        step = 0
        while step < total_steps:
            for t in range(self.rollout_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
                if self.num_envs == 1:
                    obs_t = obs_t.unsqueeze(0)

                action_t, logp_t, value_t = self.agent.act(obs_t, deterministic=False)
                action = action_t.cpu().numpy()
                logp = logp_t.cpu().numpy().squeeze(-1)
                value = value_t.cpu().numpy().squeeze(-1)

                if self.num_envs > 1:
                    next_obs, reward, terminated, truncated, _ = self.env.step(action)
                    # next_obs is a dict from vectorised env; flatten the whole dict
                    next_obs = flatten_obs(next_obs)  # <-- fixed
                    done = (terminated | truncated).astype(np.float32)
                else:
                    next_obs, reward, terminated, truncated, _ = self.env.step(
                        action[0]
                    )
                    next_obs = flatten_obs(next_obs)
                    reward = np.asarray([reward])
                    done = np.asarray([float(terminated or truncated)])

                # Defensive check
                if not (isinstance(next_obs, np.ndarray) and next_obs.ndim in (1, 2)):
                    raise ValueError(
                        f"flatten_obs returned unexpected array at step: shape={getattr(next_obs,'shape',None)}"
                    )

                if self.obs_normalizer:
                    self.obs_normalizer.update(next_obs)
                    next_obs = self.obs_normalizer.normalize(next_obs)

                for i in range(self.num_envs):
                    self.buffer.add(
                        t,
                        i,
                        obs[i] if self.num_envs > 1 else obs,
                        action[i],
                        float(reward[i]),
                        float(done[i]),
                        float(value[i]),
                        float(logp[i]),
                    )
                    self.episode_return[i] += reward[i]
                    # Log episode return for each env when done
                    if done[i]:
                        print(
                            f"step={step+1} env={i} episode_return={self.episode_return[i]:.2f}"
                        )
                        log_wandb(
                            {
                                "train/episode_return": float(self.episode_return[i]),
                                "env_id": i,
                            },
                            step=step + 1,
                        )
                        self.episode_return[i] = 0.0

                obs = next_obs
                step += self.num_envs
                if step >= total_steps:
                    break

            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
                if self.num_envs == 1:
                    obs_t = obs_t.unsqueeze(0)
                last_values = self.agent.value(obs_t).cpu().numpy().squeeze(-1)

            data = self.buffer.compute_returns_advantages(
                last_values, self.agent.config.gamma, self.agent.config.gae_lambda
            )
            obs_f, act_f, logp_f, val_f, ret_f, adv_f = data

            adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

            obs_f = torch.tensor(obs_f, dtype=torch.float32, device=self.agent.device)
            act_f = torch.tensor(act_f, dtype=torch.float32, device=self.agent.device)
            logp_f = torch.tensor(logp_f, dtype=torch.float32, device=self.agent.device)
            val_f = torch.tensor(val_f, dtype=torch.float32, device=self.agent.device)
            ret_f = torch.tensor(ret_f, dtype=torch.float32, device=self.agent.device)
            adv_f = torch.tensor(adv_f, dtype=torch.float32, device=self.agent.device)

            for _ in range(self.update_epochs):
                for idx in self.buffer.minibatches(self.minibatch_size):
                    batch = {
                        "obs": obs_f[idx],
                        "actions": act_f[idx],
                        "log_probs": logp_f[idx].unsqueeze(-1),
                        "values_old": val_f[idx].unsqueeze(-1),
                        "returns": ret_f[idx].unsqueeze(-1),
                        "advantages": adv_f[idx].unsqueeze(-1),
                    }
                    metrics = self.agent.update(batch)
                    log_wandb(metrics, step=step)
                    approx_kl = metrics.get("approx_kl", None)
                    if approx_kl is not None:
                        if approx_kl > 1.5 * self.agent.config.target_kl:
                            # Stop PPO updates early to preserve trust region
                            break

            self.buffer.reset()

            if eval_interval > 0 and self.last_eval < step // eval_interval:
                self.last_eval = step // eval_interval
                metrics = evaluate(
                    self.agent.device,
                    self.agent.policy,
                    self.domain,
                    self.task,
                    obs_normalizer=self.obs_normalizer,
                )
                print(f"step={step} " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
                log_wandb(metrics, step=step)

            if save_interval > 0 and self.last_checkpoint < step // save_interval:
                self.last_checkpoint = step // save_interval
                print(f"Saving checkpoint at step {step} to {out_dir}")
                os.makedirs(out_dir, exist_ok=True)
                torch.save(
                    {
                        "policy": self.agent.policy.state_dict(),
                        "value": self.agent.value.state_dict(),
                    },
                    os.path.join(out_dir, "ppo.pt"),
                )

        self.env.close()
