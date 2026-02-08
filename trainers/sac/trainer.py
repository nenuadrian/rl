import os
from typing import Tuple

import numpy as np
import torch

from trainers.sac.agent import SACAgent, SACConfig
from utils.env import evaluate, flatten_obs, make_dm_control_env, infer_obs_dim
from utils.wandb_utils import log_wandb
from utils.replay_buffer import ReplayBuffer


class Trainer:
    def __init__(
        self,
        domain: str,
        task: str,
        seed: int,
        device: torch.device,
        hidden_sizes: Tuple[int, int],
        replay_size: int,
        config: SACConfig | None = None,
    ):
        self.env = make_dm_control_env(domain, task, seed=seed)

        obs_dim = infer_obs_dim(self.env.observation_space)
        act_dim = int(np.prod(self.env.action_space.shape))
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high

        self.domain = domain
        self.task = task

        self.agent = SACAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            action_low=action_low,
            action_high=action_high,
            device=device,
            hidden_sizes=hidden_sizes,
            config=config or SACConfig(),
        )

        self.replay = ReplayBuffer(obs_dim, act_dim, capacity=replay_size)

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
                ckpt_path = os.path.join(out_dir, f"sac_step_{step}.pt")
                torch.save(
                    {
                        "policy": self.agent.policy.state_dict(),
                        "q1": self.agent.q1.state_dict(),
                        "q2": self.agent.q2.state_dict(),
                        "log_alpha": self.agent.log_alpha.detach().cpu(),
                    },
                    ckpt_path,
                )
                print(f"saved checkpoint: {ckpt_path}")

        self.env.close()
