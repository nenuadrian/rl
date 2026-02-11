from __future__ import annotations

import os
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch

from trainers.ppo.agent import PPOAgent, PPOConfig
from utils.env import flatten_obs, make_env, infer_obs_dim
from utils.obs_normalizer import ObsNormalizer
from utils.wandb_utils import log_wandb
from trainers.ppo.rollout_buffer import RolloutBuffer


class Trainer:
    def __init__(
        self,
        env_id: str,
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
        self.seed = seed
        self.num_envs = int(num_envs)
        self.env_id = env_id
        env_fns = [
            (lambda i=i: make_env(env_id, seed=seed + i)) for i in range(self.num_envs)
        ]
        # Avoid NEXT_STEP synthetic transitions after done=True.
        self.env = gym.vector.AsyncVectorEnv(
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
        self.rollout_steps = rollout_steps
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.obs_normalizer = ObsNormalizer(obs_dim) if normalize_obs else None
        self.buffer = RolloutBuffer.create(
            obs_dim, act_dim, rollout_steps, self.num_envs
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
        # obs is a dict mapping -> per-key arrays with leading dim N
        obs = flatten_obs(obs)

        if self.obs_normalizer:
            self.obs_normalizer.update(obs)
            obs = self.obs_normalizer.normalize(obs)

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
                        obs[i],
                        action[i],
                        float(reward[i]),
                        float(done[i]),
                        float(value[i]),
                        float(logp[i]),
                    )
                    self.episode_return[i] += float(reward[i])
                    # Log episode return for each env when done
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
                    log_wandb(metrics, step=step, silent=True)
                    approx_kl = metrics.get("approx_kl", None)
                    if approx_kl is not None:
                        if approx_kl > 1.5 * self.agent.config.target_kl:
                            # Stop PPO updates early to preserve trust region
                            break

            self.buffer.reset()

            if eval_interval > 0 and self.last_eval < step // eval_interval:
                self.last_eval = step // eval_interval
                metrics = _evaluate_vectorized(
                    agent=self.agent,
                    env_id=self.env_id,
                    seed=self.seed + 1000,
                    obs_normalizer=self.obs_normalizer,
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
                    os.path.join(out_dir, "ppo.pt"),
                )

        self.env.close()


@torch.no_grad()
def _evaluate_vectorized(
    agent: PPOAgent,
    env_id: str,
    n_episodes: int = 10,
    seed: int = 42,
    obs_normalizer: ObsNormalizer | None = None,
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
        if obs_normalizer is not None:
            obs = obs_normalizer.normalize(obs)

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
