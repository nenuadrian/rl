from __future__ import annotations

import os
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch

from trainers.mpo.agent import MPOAgent, MPOConfig
from utils.env import flatten_obs, make_env, infer_obs_dim
from utils.wandb_utils import log_wandb
from trainers.mpo.replay_buffer import MPOReplayBuffer


class MPOTrainer:
    def __init__(
        self,
        env_id: str,
        seed: int,
        device: torch.device,
        policy_layer_sizes: Tuple[int, ...],
        critic_layer_sizes: Tuple[int, ...],
        replay_size: int,
        config: MPOConfig,
    ):
        self.seed = seed
        self.env = make_env(env_id, seed=seed)
        self.env_id = env_id

        obs_dim = infer_obs_dim(self.env.observation_space)
        if not isinstance(self.env.action_space, gym.spaces.Box):
            raise ValueError("MPO only supports continuous action spaces.")
        if self.env.action_space.shape is None:
            raise ValueError("Action space has no shape.")
        act_dim = int(np.prod(self.env.action_space.shape))
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high

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
                if step >= update_after:
                    log_wandb(
                        {"train/episode_return": float(self.episode_return)},
                        step=step,
                        silent=True,
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
                    log_wandb(metrics, step=step, silent=True)

                if eval_interval > 0 and step % eval_interval == 0:
                    metrics = _evaluate_vectorized(
                        agent=self.agent,
                        env_id=self.env_id,
                        seed=self.seed + 1000,
                    )
                    log_wandb(metrics, step=step)

                if save_interval > 0 and step % save_interval == 0:
                    ckpt_path = os.path.join(out_dir, f"mpo.pt")
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


@torch.no_grad()
def _evaluate_vectorized(
    agent: MPOAgent,
    env_id: str,
    n_episodes: int = 10,
    seed: int = 42,
) -> Dict[str, float]:
    """
    High-performance vectorized evaluation.
    Runs all n_episodes in parallel using a SyncVectorEnv.
    """
    eval_envs = gym.vector.SyncVectorEnv(
        [lambda i=i: make_env(env_id, seed=seed + i) for i in range(n_episodes)]
    )

    agent.policy.eval()
    obs, _ = eval_envs.reset(seed=seed)

    episode_returns = np.zeros(n_episodes, dtype=np.float32)
    final_returns = []
    dones = np.zeros(n_episodes, dtype=bool)

    while len(final_returns) < n_episodes:
        obs = flatten_obs(obs)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device)

        mean, log_std = agent.policy(obs_t)
        action = (
            agent.policy.sample_action(mean=mean, log_std=log_std, deterministic=True)
            .cpu()
            .numpy()
        )
        action = np.clip(
            action,
            eval_envs.single_action_space.low,
            eval_envs.single_action_space.high,
        )

        next_obs, reward, terminated, truncated, _ = eval_envs.step(action)
        episode_returns += np.asarray(reward, dtype=np.float32)

        done = np.asarray(terminated) | np.asarray(truncated)
        for i in range(n_episodes):
            if not dones[i] and done[i]:
                final_returns.append(float(episode_returns[i]))
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
