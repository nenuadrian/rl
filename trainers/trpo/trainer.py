from __future__ import annotations

import os
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch

from trainers.ppo.rollout_buffer import RolloutBuffer
from trainers.trpo.agent import TRPOAgent, TRPOConfig
from utils.env import flatten_obs, infer_obs_dim, make_env
from utils.wandb_utils import log_wandb


class TRPOTrainer:
    def __init__(
        self,
        env_id: str,
        seed: int,
        device: torch.device,
        policy_layer_sizes: Tuple[int, ...],
        critic_layer_sizes: Tuple[int, ...],
        rollout_steps: int,
        config: TRPOConfig,
        normalize_obs: bool = False,
        num_envs: int = 1,
    ):
        self.seed = seed
        self.num_envs = int(num_envs)
        self.env_id = env_id
        self.normalize_obs = bool(normalize_obs)

        env_fns = [
            (
                lambda i=i: make_env(
                    env_id,
                    seed=seed + i,
                    ppo_wrappers=True,
                    gamma=config.gamma,
                    normalize_observation=self.normalize_obs,
                )
            )
            for i in range(self.num_envs)
        ]

        # Avoid NEXT_STEP synthetic transitions after done=True.
        self.env = gym.vector.SyncVectorEnv(
            env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
        )

        obs_space = self.env.single_observation_space
        act_space = self.env.single_action_space

        obs_dim = infer_obs_dim(obs_space)
        if act_space.shape is None:
            raise ValueError("Action space has no shape.")
        act_dim = int(np.prod(np.array(act_space.shape)))
        action_low = getattr(act_space, "low", None)
        action_high = getattr(act_space, "high", None)

        # ClipAction exposes an unbounded action space on the wrapper.
        # TRPO's Gaussian policy needs finite env bounds for scaling.
        if action_low is None or action_high is None or not (
            np.all(np.isfinite(action_low)) and np.all(np.isfinite(action_high))
        ):
            probe_env = make_env(env_id, seed=seed, ppo_wrappers=False)
            try:
                base_action_space = probe_env.action_space
                if not isinstance(base_action_space, gym.spaces.Box):
                    raise ValueError("TRPO only supports continuous action spaces.")
                action_low = np.asarray(base_action_space.low, dtype=np.float32)
                action_high = np.asarray(base_action_space.high, dtype=np.float32)
            finally:
                probe_env.close()

        self.agent = TRPOAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            action_low=action_low,
            action_high=action_high,
            device=device,
            policy_layer_sizes=policy_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            config=config,
        )

        self.rollout_steps = int(rollout_steps)
        self.buffer = RolloutBuffer.create(
            obs_dim=obs_dim,
            act_dim=act_dim,
            rollout_steps=self.rollout_steps,
            num_envs=self.num_envs,
        )

        self.episode_return = np.zeros(self.num_envs, dtype=np.float32)
        self.last_eval = 0
        self.last_checkpoint = 0

    def train(
        self,
        total_steps: int,
        eval_interval: int,
        save_interval: int,
        out_dir: str,
    ):
        obs, _ = self.env.reset()
        obs = flatten_obs(obs)

        step = 0
        while step < total_steps:
            for t in range(self.rollout_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)

                action_t, logp_t, value_t = self.agent.act(obs_t, deterministic=False)
                action = action_t.cpu().numpy()
                logp = logp_t.cpu().numpy().squeeze(-1)
                value = value_t.cpu().numpy().squeeze(-1)

                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                next_obs = flatten_obs(next_obs)
                done = np.asarray(terminated) | np.asarray(truncated)
                reward = np.asarray(reward)

                if not (isinstance(next_obs, np.ndarray) and next_obs.ndim in (1, 2)):
                    raise ValueError(
                        f"flatten_obs returned unexpected array at step: shape={getattr(next_obs, 'shape', None)}"
                    )

                for i in range(self.num_envs):
                    self.buffer.add(
                        t=t,
                        env_i=i,
                        obs=obs[i],
                        action=action[i],
                        reward=float(reward[i]),
                        done=float(done[i]),
                        value=float(value[i]),
                        log_prob=float(logp[i]),
                    )
                    self.episode_return[i] += float(reward[i])
                    if done[i]:
                        log_wandb(
                            {"train/episode_return": float(self.episode_return[i])},
                            step=step + 1,
                            silent=True,
                        )
                        self.episode_return[i] = 0.0

                obs = next_obs
                step += self.num_envs
                if step >= total_steps:
                    break

            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
                last_values = self.agent.value(obs_t).cpu().numpy().squeeze(-1)

            obs_f, act_f, logp_f, _, ret_f, adv_f = self.buffer.compute_returns_advantages(
                last_values,
                self.agent.config.gamma,
                self.agent.config.gae_lambda,
            )

            if self.agent.config.normalize_advantages:
                adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

            batch = {
                "obs": torch.tensor(obs_f, dtype=torch.float32, device=self.agent.device),
                "actions": torch.tensor(
                    act_f,
                    dtype=torch.float32,
                    device=self.agent.device,
                ),
                "log_probs": torch.tensor(
                    logp_f,
                    dtype=torch.float32,
                    device=self.agent.device,
                ).unsqueeze(-1),
                "returns": torch.tensor(
                    ret_f,
                    dtype=torch.float32,
                    device=self.agent.device,
                ).unsqueeze(-1),
                "advantages": torch.tensor(
                    adv_f,
                    dtype=torch.float32,
                    device=self.agent.device,
                ).unsqueeze(-1),
            }

            metrics = self.agent.update(batch)
            log_wandb(metrics, step=step, silent=True)

            self.buffer.reset()

            if eval_interval > 0 and self.last_eval < step // eval_interval:
                self.last_eval = step // eval_interval
                metrics = _evaluate_vectorized(
                    agent=self.agent,
                    env_id=self.env_id,
                    seed=self.seed + 1000,
                    gamma=self.agent.config.gamma,
                    normalize_observation=self.normalize_obs,
                )
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
                    os.path.join(out_dir, "trpo.pt"),
                )

        self.env.close()


@torch.no_grad()
def _evaluate_vectorized(
    agent: TRPOAgent,
    env_id: str,
    n_episodes: int = 10,
    seed: int = 42,
    gamma: float = 0.99,
    normalize_observation: bool = True,
) -> Dict[str, float]:
    eval_envs = gym.vector.SyncVectorEnv(
        [
            (
                lambda i=i: make_env(
                    env_id,
                    seed=seed + i,
                    ppo_wrappers=True,
                    gamma=gamma,
                    normalize_observation=normalize_observation,
                )
            )
            for i in range(n_episodes)
        ]
    )

    agent.policy.eval()
    obs, _ = eval_envs.reset(seed=seed)

    episode_returns = np.zeros(n_episodes, dtype=np.float32)
    final_returns = []
    dones = np.zeros(n_episodes, dtype=bool)

    while len(final_returns) < n_episodes:
        obs = flatten_obs(obs)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device)
        action = agent.policy.sample_action(obs_t, deterministic=True).cpu().numpy()
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
