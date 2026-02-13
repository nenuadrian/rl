from __future__ import annotations

import os
from typing import Dict, Mapping, Tuple

import gymnasium as gym
import numpy as np
import torch

from trainers.vmpo.agent import VMPOAgent
from trainers.vmpo.targets import compute_rollout_targets
from utils.env import flatten_obs, infer_obs_dim, wrap_record_video
from utils.wandb_utils import log_wandb


def _format_metrics(metrics: Mapping[str, float]) -> str:
    return ", ".join(
        f"{key}={float(value):.4f}" for key, value in sorted(metrics.items())
    )


def _transform_observation(env: gym.Env, fn):
    """Gymnasium compatibility shim across wrapper signatures."""
    try:
        return gym.wrappers.TransformObservation(env, fn)
    except TypeError:
        return gym.wrappers.TransformObservation(env, fn, env.observation_space)


def _transform_reward(env: gym.Env, fn):
    """Gymnasium compatibility shim across wrapper signatures."""
    try:
        return gym.wrappers.TransformReward(env, fn)
    except TypeError:
        return gym.wrappers.TransformReward(env, fn, env.reward_range)


def _make_env(
    env_id: str,
    *,
    seed: int | None = None,
    render_mode: str | None = None,
    gamma: float = 0.99,
    normalize_observation: bool = True,
    clip_observation: float | None = 10.0,
    normalize_reward: bool = True,
    clip_reward: float | None = 10.0,
    capture_video: bool = False,
    run_name: str | None = None,
    idx: int = 0,
) -> gym.Env:
    if env_id.startswith("dm_control/"):
        _, domain, task = env_id.split("/")
        env = gym.make(f"dm_control/{domain}-{task}-v0", render_mode=render_mode)
    else:
        env = gym.make(env_id, render_mode=render_mode)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    if capture_video and run_name is not None and idx == 0:
        env = wrap_record_video(env, f"videos/{run_name}")

    env = gym.wrappers.RecordEpisodeStatistics(env)

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)
    if normalize_observation:
        env = gym.wrappers.NormalizeObservation(env)
        if clip_observation is not None:
            env = _transform_observation(
                env,
                lambda obs: np.clip(obs, -clip_observation, clip_observation),
            )
    if normalize_reward:
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        if clip_reward is not None:
            env = _transform_reward(
                env,
                lambda reward: np.clip(reward, -clip_reward, clip_reward),
            )

    return env


class VMPOTrainer:
    def __init__(
        self,
        env_id: str,
        seed: int,
        device: torch.device,
        policy_layer_sizes: Tuple[int, ...],
        value_layer_sizes: Tuple[int, ...],
        rollout_steps: int,
        normalize_advantages: bool = True,
        gamma: float = 0.99,
        advantage_estimator: str = "returns",
        gae_lambda: float = 0.95,
        policy_lr: float = 5e-4,
        value_lr: float = 1e-3,
        topk_fraction: float = 0.5,
        temperature_init: float = 1.0,
        temperature_lr: float = 1e-4,
        epsilon_eta: float = 0.1,
        epsilon_mu: float = 0.01,
        epsilon_sigma: float = 0.01,
        alpha_lr: float = 1e-4,
        max_grad_norm: float = 10.0,
        optimizer_type: str = "adam",
        sgd_momentum: float = 0.9,
        num_envs: int = 1,
        capture_video: bool = False,
        run_name: str | None = None,
    ):
        self.run_name = run_name
        self.num_envs = int(num_envs)
        self.env_id = env_id
        self.seed = seed
        self.capture_video = bool(capture_video)
        self.video_dir = f"videos/{run_name}"
        self.gamma = float(gamma)
        self.advantage_estimator = str(advantage_estimator)
        self.gae_lambda = float(gae_lambda)

        env_fns = [
            (
                lambda i=i: self._make_train_env(
                    env_id=env_id,
                    seed=seed + i,
                    env_index=i,
                    gamma=self.gamma,
                )
            )
            for i in range(self.num_envs)
        ]
        # Always use vectorized env, even when num_envs == 1.
        self.env = gym.vector.SyncVectorEnv(
            env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
        )
        obs_space = self.env.single_observation_space
        act_space = self.env.single_action_space

        obs_dim = infer_obs_dim(obs_space)
        if not isinstance(self.env.action_space, gym.spaces.Box):
            # for vector envs, act_space above already set
            raise ValueError("VMPO only supports continuous action spaces.")
        if act_space.shape is None:
            raise ValueError("Action space has no shape.")
        act_shape = act_space.shape
        act_dim = int(np.prod(act_shape))
        action_low = getattr(act_space, "low", None)
        action_high = getattr(act_space, "high", None)
        self.act_shape = act_shape

        self.agent = VMPOAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            action_low=action_low,
            action_high=action_high,
            device=device,
            policy_layer_sizes=policy_layer_sizes,
            value_layer_sizes=value_layer_sizes,
            normalize_advantages=normalize_advantages,
            gamma=gamma,
            advantage_estimator=advantage_estimator,
            gae_lambda=gae_lambda,
            policy_lr=policy_lr,
            value_lr=value_lr,
            topk_fraction=topk_fraction,
            temperature_init=temperature_init,
            temperature_lr=temperature_lr,
            epsilon_eta=epsilon_eta,
            epsilon_mu=epsilon_mu,
            epsilon_sigma=epsilon_sigma,
            alpha_lr=alpha_lr,
            max_grad_norm=max_grad_norm,
            optimizer_type=optimizer_type,
            sgd_momentum=sgd_momentum,
        )

        self.rollout_steps = rollout_steps

        self.obs_buf: list[np.ndarray] = []
        self.actions_buf: list[np.ndarray] = []
        self.rewards_buf: list[np.ndarray] = []
        self.dones_buf: list[np.ndarray] = []
        self.values_buf: list[np.ndarray] = []
        self.means_buf: list[np.ndarray] = []
        self.log_stds_buf: list[np.ndarray] = []

        # episode returns per environment
        self.episode_return = np.zeros(self.num_envs, dtype=np.float32)

    def _make_train_env(
        self,
        *,
        env_id: str,
        seed: int,
        env_index: int,
        gamma: float,
    ):
        env = _make_env(
            env_id,
            seed=seed,
            gamma=gamma,
        )
        return env

    def _reset_rollout(self) -> None:
        self.obs_buf.clear()
        self.actions_buf.clear()
        self.rewards_buf.clear()
        self.dones_buf.clear()
        self.values_buf.clear()
        self.means_buf.clear()
        self.log_stds_buf.clear()

    def _rollout_full(self) -> bool:
        return len(self.obs_buf) >= self.rollout_steps

    def train(
        self,
        total_steps: int,
        eval_interval: int,
        save_interval: int,
        out_dir: str,
        updates_per_step: int = 1,
    ):
        console_log_interval = max(
            1, min(1_000, eval_interval if eval_interval > 0 else 1_000)
        )
        print(
            "[VMPO] training started: "
            f"total_steps={total_steps}, "
            f"rollout_steps={self.rollout_steps}, "
            f"num_envs={self.num_envs}, "
            f"updates_per_step={int(updates_per_step)}, "
            f"console_log_interval={console_log_interval}"
        )
        interval_metric_sums: Dict[str, float] = {}
        interval_update_count = 0
        interval_episode_count = 0
        interval_episode_sum = 0.0
        interval_episode_min = float("inf")
        interval_episode_max = float("-inf")
        total_update_count = 0

        obs, _ = self.env.reset()
        obs = flatten_obs(obs)

        for step in range(1, total_steps + 1):
            action, value, mean, log_std = self.agent.act(obs, deterministic=False)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_obs = flatten_obs(next_obs)
            done = np.asarray(terminated) | np.asarray(truncated)
            reward = np.asarray(reward, dtype=np.float32)

            # Store step data (works for single- and multi-env)
            self.obs_buf.append(obs)
            self.actions_buf.append(action)
            self.rewards_buf.append(reward)
            self.dones_buf.append(done)
            self.values_buf.append(value)
            self.means_buf.append(mean)
            self.log_stds_buf.append(log_std)

            obs = next_obs

            self.episode_return += reward

            finished_mask = done.astype(bool)
            if np.any(finished_mask):
                finished_indices = np.flatnonzero(finished_mask)
                finished_returns = self.episode_return[finished_mask]
                self.episode_return[finished_mask] = 0.0
                for env_i, episode_return in zip(finished_indices, finished_returns):
                    episode_return_f = float(episode_return)
                    log_wandb(
                        {"train/episode_return": episode_return_f},
                        step=step,
                        silent=True,
                    )
                    print(
                        f"[VMPO][episode] step={step}/{total_steps}, "
                        f"env={int(env_i)}, return={episode_return_f:.3f}"
                    )
                    interval_episode_count += 1
                    interval_episode_sum += episode_return_f
                    interval_episode_min = min(interval_episode_min, episode_return_f)
                    interval_episode_max = max(interval_episode_max, episode_return_f)

            if self._rollout_full():
                # Stack collected arrays and reshape for batch processing
                obs_arr = np.stack(self.obs_buf)
                actions_arr = np.stack(self.actions_buf)
                rewards_arr = np.asarray(self.rewards_buf, dtype=np.float32)
                dones_arr = np.asarray(self.dones_buf, dtype=np.float32)
                values_arr = np.asarray(self.values_buf, dtype=np.float32)
                means_arr = np.stack(self.means_buf)
                log_stds_arr = np.stack(self.log_stds_buf)

                last_value = self.agent.value(obs)
                last_value = last_value * (1.0 - dones_arr[-1])

                # obs_arr shape (T, N, obs_dim) -> flatten to (T*N, obs_dim)
                T, N, _ = obs_arr.shape
                obs_flat = obs_arr.reshape(T * N, -1)
                actions_flat = actions_arr.reshape(T * N, -1)
                rewards_flat = rewards_arr.reshape(T, N)
                dones_flat = dones_arr.reshape(T, N)
                values_flat = values_arr.reshape(T, N)
                means_flat = means_arr.reshape(T * N, -1)
                log_stds_flat = log_stds_arr.reshape(T * N, -1)

                returns, advantages = compute_rollout_targets(
                    rewards=rewards_flat,
                    dones=dones_flat,
                    values=values_flat,
                    last_value=last_value,
                    gamma=self.gamma,
                    estimator=self.advantage_estimator,
                    gae_lambda=self.gae_lambda,
                )
                returns_flat = returns.reshape(T * N, 1)
                advantages_flat = advantages.reshape(T * N, 1)

                batch = {
                    "obs": torch.tensor(
                        obs_flat, dtype=torch.float32, device=self.agent.device
                    ),
                    "actions": torch.tensor(
                        actions_flat, dtype=torch.float32, device=self.agent.device
                    ),
                    "returns": torch.tensor(
                        returns_flat, dtype=torch.float32, device=self.agent.device
                    ),
                    "advantages": torch.tensor(
                        advantages_flat, dtype=torch.float32, device=self.agent.device
                    ),
                    "old_means": torch.tensor(
                        means_flat, dtype=torch.float32, device=self.agent.device
                    ),
                    "old_log_stds": torch.tensor(
                        log_stds_flat, dtype=torch.float32, device=self.agent.device
                    ),
                }

                metrics = {}
                for _ in range(updates_per_step):
                    metrics = self.agent.update(batch)
                    log_wandb(metrics, step=step)
                    for key, value in metrics.items():
                        interval_metric_sums[key] = interval_metric_sums.get(
                            key, 0.0
                        ) + float(value)
                    interval_update_count += 1
                    total_update_count += 1

                self._reset_rollout()

            if eval_interval > 0 and step % eval_interval == 0:
                metrics = _evaluate_vectorized(
                    agent=self.agent,
                    env_id=self.env_id,
                    seed=self.seed + 1000,
                    gamma=self.gamma,
                    capture_video=self.capture_video,
                    run_name=self.run_name,
                )
                log_wandb(metrics, step=step)
                print(
                    f"[VMPO][eval] step={step}/{total_steps}: {_format_metrics(metrics)}"
                )

            if save_interval > 0 and step % save_interval == 0:
                ckpt_path = os.path.join(out_dir, f"vmpo.pt")
                os.makedirs(out_dir, exist_ok=True)
                torch.save({"policy": self.agent.policy.state_dict()}, ckpt_path)
                print(f"[VMPO][checkpoint] step={step}: saved {ckpt_path}")

            should_print_progress = (
                step == total_steps or step % console_log_interval == 0
            )
            if should_print_progress:
                progress = 100.0 * float(step) / float(total_steps)
                print(
                    "[VMPO][progress] "
                    f"step={step}/{total_steps} ({progress:.2f}%), "
                    f"updates={total_update_count}"
                )
                if interval_episode_count > 0:
                    print(
                        "[VMPO][episode-stats] "
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
                        "[VMPO][train-metrics] "
                        f"updates={interval_update_count}, {_format_metrics(mean_metrics)}"
                    )
                    interval_metric_sums.clear()
                    interval_update_count = 0

        self.env.close()


@torch.no_grad()
def _evaluate_vectorized(
    agent: VMPOAgent,
    env_id: str,
    n_episodes: int = 10,
    seed: int = 42,
    gamma: float = 0.99,
    obs_normalizer=None,
    capture_video: bool = False,
    run_name: str | None = None,
) -> Dict[str, float]:
    """
    High-performance vectorized evaluation.
    Runs all n_episodes in parallel using a SyncVectorEnv.
    """
    eval_envs = gym.vector.SyncVectorEnv(
        [
            lambda i=i: _make_env(
                env_id,
                seed=seed + i,
                gamma=gamma,
                capture_video=capture_video,
                render_mode="rgb_array" if capture_video else None,
                run_name=run_name,
                idx=i,
            )
            for i in range(n_episodes)
        ]
    )

    agent.policy.eval()
    obs, _ = eval_envs.reset(seed=seed)

    episode_returns = np.zeros(n_episodes)
    final_returns = []

    # Track which envs have finished their first episode
    dones = np.zeros(n_episodes, dtype=bool)

    while len(final_returns) < n_episodes:
        # Pre-process observations
        obs = flatten_obs(obs)
        if obs_normalizer is not None:
            obs = obs_normalizer.normalize(obs)

        # Deterministic Action Selection
        # We call act() with deterministic=True to use the mean of the Gaussian
        action, _, _, _ = agent.act(obs, deterministic=True)

        # Clip actions to valid range
        action = np.clip(
            action,
            eval_envs.single_action_space.low,
            eval_envs.single_action_space.high,
        )

        # Step environment
        next_obs, reward, terminated, truncated, infos = eval_envs.step(action)

        # Accumulate rewards for envs that haven't finished yet
        episode_returns += reward

        # Check for completions
        # Gymnasium VectorEnv resets automatically; we catch the return in infos
        for i in range(n_episodes):
            if not dones[i] and (terminated[i] or truncated[i]):
                final_returns.append(episode_returns[i])
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
