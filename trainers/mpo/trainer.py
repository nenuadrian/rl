from __future__ import annotations

import os
import time
from typing import Dict, Mapping, Tuple

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage
from torchrl.data.replay_buffers import TensorDictReplayBuffer

from trainers.mpo.agent import MPOAgent
from utils.env import infer_obs_dim
from utils.wandb_utils import log_wandb


def _format_metrics(metrics: Mapping[str, float]) -> str:
    return ", ".join(
        f"{key}={float(value):.4f}" for key, value in sorted(metrics.items())
    )


def _make_env(
    env_id: str,
    *,
    seed: int | None = None,
) -> gym.Env:
    if env_id.startswith("dm_control/"):
        _, domain, task = env_id.split("/")
        env = gym.make(f"dm_control/{domain}-{task}-v0")
    else:
        env = gym.make(env_id)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def _to_device_tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = np.asarray(value, dtype=np.float32)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return torch.from_numpy(arr).to(device=device, non_blocking=True)


class MPOTrainer:
    def __init__(
        self,
        env_id: str,
        seed: int,
        device: torch.device,
        policy_layer_sizes: Tuple[int, ...],
        critic_layer_sizes: Tuple[int, ...],
        replay_size: int,
        gamma: float = 0.99,
        target_networks_update_period: int = 100,
        policy_lr: float = 3e-4,
        q_lr: float = 3e-4,
        kl_epsilon: float = 0.1,
        mstep_kl_epsilon: float = 0.1,
        temperature_init: float = 1.0,
        temperature_lr: float = 3e-4,
        lambda_init: float = 1.0,
        lambda_lr: float = 3e-4,
        epsilon_penalty: float = 0.001,
        max_grad_norm: float = 1.0,
        action_samples: int = 20,
        use_retrace: bool = False,
        retrace_steps: int = 2,
        retrace_mc_actions: int = 8,
        retrace_lambda: float = 0.95,
        optimizer_type: str = "adam",
        sgd_momentum: float = 0.9,
        init_log_alpha_mean: float = 10.0,
        init_log_alpha_stddev: float = 1000.0,
    ):
        self.seed = seed
        self.use_retrace = bool(use_retrace)
        self.retrace_steps = int(retrace_steps)
        self.env = _make_env(
            env_id,
            seed=seed,
        )
        self.eval_episodes = 50
        self.eval_seed = self.seed + 1000
        self.eval_envs = gym.vector.SyncVectorEnv(
            [
                (lambda i=i: _make_env(env_id, seed=self.eval_seed + i))
                for i in range(self.eval_episodes)
            ]
        )
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
            gamma=gamma,
            target_networks_update_period=target_networks_update_period,
            policy_lr=policy_lr,
            q_lr=q_lr,
            kl_epsilon=kl_epsilon,
            mstep_kl_epsilon=mstep_kl_epsilon,
            temperature_init=temperature_init,
            temperature_lr=temperature_lr,
            lambda_init=lambda_init,
            lambda_lr=lambda_lr,
            epsilon_penalty=epsilon_penalty,
            max_grad_norm=max_grad_norm,
            action_samples=action_samples,
            use_retrace=use_retrace,
            retrace_steps=retrace_steps,
            retrace_mc_actions=retrace_mc_actions,
            retrace_lambda=retrace_lambda,
            optimizer_type=optimizer_type,
            sgd_momentum=sgd_momentum,
            init_log_alpha_mean=init_log_alpha_mean,
            init_log_alpha_stddev=init_log_alpha_stddev,
        )

        self.replay_capacity = int(replay_size)
        self.replay = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=self.replay_capacity),
        )
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._sequence_offsets: dict[int, torch.Tensor] = {}
        self._dropped_nonfinite_transitions = 0
        self._skipped_nonfinite_batches = 0

        self.episode_return = 0.0

    @property
    def replay_size(self) -> int:
        return int(len(self.replay))

    @property
    def replay_ptr(self) -> int:
        return int(self.replay.write_count % self.replay_capacity)

    @staticmethod
    def _as_cpu_float_tensor(
        value: np.ndarray | float,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device="cpu").reshape(shape)

    @staticmethod
    def _is_finite_value(value: np.ndarray | float) -> bool:
        arr = np.asarray(value, dtype=np.float32)
        return bool(np.isfinite(arr).all())

    def _add_transition(
        self,
        obs: np.ndarray,
        action_exec: np.ndarray,
        action_raw: np.ndarray,
        behaviour_logp: float,
        reward: float,
        next_obs: np.ndarray,
        done: float,
    ) -> bool:
        fields = {
            "obs": obs,
            "action_exec": action_exec,
            "action_raw": action_raw,
            "behaviour_logp": behaviour_logp,
            "reward": reward,
            "next_obs": next_obs,
            "done": done,
        }
        bad_fields = [
            name for name, value in fields.items() if not self._is_finite_value(value)
        ]
        if bad_fields:
            self._dropped_nonfinite_transitions += 1
            if (
                self._dropped_nonfinite_transitions <= 5
                or self._dropped_nonfinite_transitions % 100 == 0
            ):
                print(
                    "[MPO][warn] "
                    f"dropped non-finite transition fields={bad_fields} "
                    f"count={self._dropped_nonfinite_transitions}"
                )
            return False

        transition = TensorDict(
            {
                "obs": self._as_cpu_float_tensor(obs, (self._obs_dim,)),
                "next_obs": self._as_cpu_float_tensor(next_obs, (self._obs_dim,)),
                "actions_exec": self._as_cpu_float_tensor(
                    action_exec, (self._act_dim,)
                ),
                "actions_raw": self._as_cpu_float_tensor(action_raw, (self._act_dim,)),
                "behaviour_logp": self._as_cpu_float_tensor(behaviour_logp, (1,)),
                "rewards": self._as_cpu_float_tensor(reward, (1,)),
                "dones": self._as_cpu_float_tensor(done, (1,)),
            },
            batch_size=[],
        )
        self.replay.add(transition)
        return True

    def _get_sequence_offsets(self, seq_len: int) -> torch.Tensor:
        offsets = self._sequence_offsets.get(seq_len)
        if offsets is None:
            offsets = torch.arange(seq_len, dtype=torch.long)
            self._sequence_offsets[seq_len] = offsets
        return offsets

    def _sample_sequences(self, batch_size: int, seq_len: int) -> TensorDict:
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if self.replay_size < seq_len:
            raise ValueError("Not enough data in replay buffer for sequence sampling")

        batch_size = int(batch_size)
        offsets = self._get_sequence_offsets(seq_len)
        if self.replay_size < self.replay_capacity:
            starts = torch.randint(
                0,
                self.replay_size - seq_len + 1,
                (batch_size,),
                dtype=torch.long,
            )
            idxs = starts.unsqueeze(1) + offsets.unsqueeze(0)
        else:
            valid_start_count = self.replay_capacity - (seq_len - 1)
            if valid_start_count <= 0:
                raise RuntimeError(
                    "Failed to sample enough contiguous sequences; consider reducing seq_len."
                )
            starts = (
                self.replay_ptr
                + torch.randint(0, valid_start_count, (batch_size,), dtype=torch.long)
            ) % self.replay_capacity
            idxs = (starts.unsqueeze(1) + offsets.unsqueeze(0)) % self.replay_capacity

        return self.replay.storage[idxs]

    def train(
        self,
        total_steps: int,
        update_after: int,
        batch_size: int,
        out_dir: str,
        updates_per_step: int = 1,
    ):
        total_steps = int(total_steps)
        update_after = int(update_after)
        batch_size = int(batch_size)
        eval_interval = max(1, total_steps // 150)
        console_log_interval = max(1, min(1_000, eval_interval))
        print(
            "[MPO] training started: "
            f"total_steps={total_steps}, "
            f"update_after={update_after}, "
            f"batch_size={batch_size}, "
            f"updates_per_step={int(updates_per_step)}, "
            f"eval_interval={eval_interval}, "
            f"console_log_interval={console_log_interval}"
        )

        interval_metric_sums: Dict[str, float] = {}
        interval_update_count = 0
        total_update_count = 0
        interval_episode_count = 0
        interval_episode_sum = 0.0
        interval_episode_min = float("inf")
        interval_episode_max = float("-inf")
        start_time = time.perf_counter()
        last_progress_time = start_time
        last_progress_step = 0
        last_progress_updates = 0
        best_eval_score = float("-inf")
        os.makedirs(out_dir, exist_ok=True)

        obs, _ = self.env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        try:
            for step in range(1, total_steps + 1):
                action_exec, action_raw, behaviour_logp = self.agent.act_with_logp(
                    obs, deterministic=False
                )

                next_obs, reward, terminated, truncated, _ = self.env.step(action_exec)
                next_obs = np.asarray(next_obs, dtype=np.float32)
                reward_f = float(reward)
                done = float(terminated or truncated)

                self._add_transition(
                    obs,
                    action_exec,
                    action_raw,
                    behaviour_logp,
                    reward_f,
                    next_obs,
                    done,
                )
                obs = next_obs

                if np.isfinite(reward_f):
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
                    obs = np.asarray(obs, dtype=np.float32)
                    self.episode_return = 0.0

                if step >= update_after and self.replay_size >= batch_size:
                    step_metric_sums: Dict[str, float] = {}
                    step_update_count = 0
                    for _ in range(int(updates_per_step)):
                        seq_len = self.retrace_steps
                        if self.use_retrace and seq_len > 1:
                            if self.replay_size >= batch_size + seq_len:
                                batch = self._sample_sequences(
                                    batch_size=batch_size, seq_len=seq_len
                                )
                            else:
                                continue
                        else:
                            batch = self.replay.sample(batch_size=int(batch_size))

                        metrics = self.agent.update(batch)
                        if metrics is None:
                            self._skipped_nonfinite_batches += 1
                            continue

                        for key, value in metrics.items():
                            step_metric_sums[key] = step_metric_sums.get(
                                key, 0.0
                            ) + float(value)
                        step_update_count += 1
                        for key, value in metrics.items():
                            interval_metric_sums[key] = interval_metric_sums.get(
                                key, 0.0
                            ) + float(value)
                        interval_update_count += 1
                        total_update_count += 1

                    if step_update_count > 0:
                        step_mean_metrics = {
                            key: value / float(step_update_count)
                            for key, value in step_metric_sums.items()
                        }
                        log_wandb(step_mean_metrics, step=step, silent=True)

                if step % eval_interval == 0 and step >= update_after:
                    metrics = _evaluate_vectorized(
                        agent=self.agent,
                        eval_envs=self.eval_envs,
                        seed=self.eval_seed,
                    )
                    log_wandb(metrics, step=step)
                    print(
                        f"[MPO][eval] step={step}/{total_steps}: {_format_metrics(metrics)}"
                    )
                    ckpt_payload = {
                        "policy": self.agent.policy.state_dict(),
                        "q": self.agent.q.state_dict(),
                    }
                    ckpt_last_path = os.path.join(out_dir, "mpo_last.pt")
                    torch.save(ckpt_payload, ckpt_last_path)
                    print(f"[MPO][checkpoint] step={step}: saved {ckpt_last_path}")

                    eval_score = float(metrics["eval/return_mean"])
                    if eval_score > best_eval_score:
                        best_eval_score = eval_score
                        ckpt_best_path = os.path.join(out_dir, "mpo_best.pt")
                        torch.save(ckpt_payload, ckpt_best_path)
                        print(
                            f"[MPO][checkpoint-best] step={step}: "
                            f"score={eval_score:.6f}, saved {ckpt_best_path}"
                        )

                should_print_progress = (
                    step == total_steps
                    or step == update_after
                    or step % console_log_interval == 0
                )
                if should_print_progress:
                    now = time.perf_counter()
                    elapsed_total = max(now - start_time, 1e-9)
                    elapsed_window = max(now - last_progress_time, 1e-9)
                    steps_window = step - last_progress_step
                    updates_window = total_update_count - last_progress_updates

                    progress = 100.0 * float(step) / float(total_steps)
                    print(
                        "[MPO][progress] "
                        f"step={step}/{total_steps} ({progress:.2f}%), "
                        f"replay={self.replay_size}/{self.replay_capacity}, "
                        f"updates={total_update_count}, "
                        f"dropped={self._dropped_nonfinite_transitions}, "
                        f"skipped_batches={self._skipped_nonfinite_batches}, "
                        f"sps={steps_window / elapsed_window:.2f}, "
                        f"ups={updates_window / elapsed_window:.2f}, "
                        f"sps_total={step / elapsed_total:.2f}"
                    )
                    last_progress_time = now
                    last_progress_step = step
                    last_progress_updates = total_update_count

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
        finally:
            self.env.close()
            self.eval_envs.close()


@torch.inference_mode()
def _evaluate_vectorized(
    agent: MPOAgent,
    eval_envs: gym.vector.VectorEnv,
    seed: int = 42,
) -> Dict[str, float]:
    """
    High-performance vectorized evaluation.
    Runs all n_episodes in parallel using a SyncVectorEnv.
    """
    n_episodes = int(eval_envs.num_envs)
    was_training = agent.policy.training
    agent.policy.eval()
    obs, _ = eval_envs.reset(seed=seed)

    episode_returns = np.zeros(n_episodes, dtype=np.float32)
    final_returns = []
    dones = np.zeros(n_episodes, dtype=bool)

    while len(final_returns) < n_episodes:
        obs_t = _to_device_tensor(obs, agent.device)

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

    if was_training:
        agent.policy.train()

    return {
        "eval/return_mean": float(np.mean(final_returns)),
        "eval/length_mean": float(np.mean([len(final_returns)])),
        "eval/return_std": float(np.std(final_returns)),
        "eval/return_min": float(np.min(final_returns)),
        "eval/return_max": float(np.max(final_returns)),
    }
