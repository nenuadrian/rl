from __future__ import annotations

import os
from typing import Dict, Mapping, Tuple

import gymnasium as gym
import numpy as np
import torch

from trainers.mpo.agent import MPOAgent, MPOConfig
from utils.env import flatten_obs, make_env, infer_obs_dim, wrap_record_video
from utils.wandb_utils import log_wandb
from trainers.mpo.replay_buffer import MPOReplayBuffer


def _format_metrics(metrics: Mapping[str, float]) -> str:
    return ", ".join(
        f"{key}={float(value):.4f}" for key, value in sorted(metrics.items())
    )


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
        capture_video: bool = False,
        run_name: str | None = None,
    ):
        self.seed = seed
        self.capture_video = bool(capture_video)
        safe_run_name = (
            run_name if run_name is not None else f"mpo-{env_id}-seed{seed}"
        ).replace("/", "-")
        self.video_dir = os.path.join("videos", safe_run_name)
        self.env = make_env(
            env_id,
            seed=seed,
            render_mode="rgb_array" if self.capture_video else None,
        )
        if self.capture_video:
            self.env = wrap_record_video(self.env, self.video_dir)
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
        console_log_interval = max(
            1, min(1_000, eval_interval if eval_interval > 0 else 1_000)
        )
        print(
            "[MPO] training started: "
            f"total_steps={total_steps}, "
            f"update_after={update_after}, "
            f"batch_size={batch_size}, "
            f"updates_per_step={int(updates_per_step)}, "
            f"console_log_interval={console_log_interval}"
        )

        interval_metric_sums: Dict[str, float] = {}
        interval_update_count = 0
        interval_episode_count = 0
        interval_episode_sum = 0.0
        interval_episode_min = float("inf")
        interval_episode_max = float("-inf")

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
                episode_return = float(self.episode_return)
                if step >= update_after:
                    log_wandb(
                        {"train/episode_return": episode_return},
                        step=step,
                        silent=True,
                    )
                print(
                    f"[MPO][episode] step={step}/{total_steps}, return={episode_return:.3f}"
                )
                interval_episode_count += 1
                interval_episode_sum += episode_return
                interval_episode_min = min(interval_episode_min, episode_return)
                interval_episode_max = max(interval_episode_max, episode_return)
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
                    for key, value in metrics.items():
                        interval_metric_sums[key] = (
                            interval_metric_sums.get(key, 0.0) + float(value)
                        )
                    interval_update_count += 1

                if eval_interval > 0 and step % eval_interval == 0:
                    metrics = _evaluate_vectorized(
                        agent=self.agent,
                        env_id=self.env_id,
                        seed=self.seed + 1000,
                    )
                    log_wandb(metrics, step=step)
                    print(
                        f"[MPO][eval] step={step}/{total_steps}: {_format_metrics(metrics)}"
                    )

                if save_interval > 0 and step % save_interval == 0:
                    ckpt_path = os.path.join(out_dir, f"mpo.pt")
                    torch.save(
                        {
                            "policy": self.agent.policy.state_dict(),
                            "q": self.agent.q.state_dict(),
                        },
                        ckpt_path,
                    )
                    print(f"[MPO][checkpoint] step={step}: saved {ckpt_path}")

            should_print_progress = (
                step == total_steps
                or step == update_after
                or step % console_log_interval == 0
            )
            if should_print_progress:
                progress = 100.0 * float(step) / float(total_steps)
                print(
                    "[MPO][progress] "
                    f"step={step}/{total_steps} ({progress:.2f}%), "
                    f"replay={self.replay.size}/{self.replay.capacity}"
                )

                if interval_episode_count > 0:
                    print(
                        "[MPO][episode-stats] "
                        f"count={interval_episode_count}, "
                        f"return_mean={interval_episode_sum / interval_episode_count:.3f}, "
                        f"return_min={interval_episode_min:.3f}, "
                        f"return_max={interval_episode_max:.3f}"
                    )
                    interval_episode_count = 0
                    interval_episode_sum = 0.0
                    interval_episode_min = float("inf")
                    interval_episode_max = float("-inf")

                if interval_update_count > 0:
                    mean_metrics = {
                        key: value / float(interval_update_count)
                        for key, value in interval_metric_sums.items()
                    }
                    print(
                        "[MPO][train-metrics] "
                        f"updates={interval_update_count}, {_format_metrics(mean_metrics)}"
                    )
                    interval_metric_sums.clear()
                    interval_update_count = 0
                elif step < update_after:
                    print(
                        "[MPO][train-metrics] "
                        f"waiting for replay warmup ({step}/{update_after} steps)"
                    )

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
