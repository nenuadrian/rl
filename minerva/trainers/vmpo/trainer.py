from __future__ import annotations

import os
from typing import Dict, Mapping, Tuple

import gymnasium as gym
import numpy as np
import torch

from minerva.trainers.vmpo.agent import VMPOAgent
from minerva.trainers.vmpo.targets import compute_rollout_targets
from minerva.utils.env import infer_obs_dim
from minerva.utils.wandb_utils import log_wandb


def _format_metrics(metrics: Mapping[str, float]) -> str:
    return ", ".join(
        f"{key}={float(value):.4f}" for key, value in sorted(metrics.items())
    )


def _iter_final_info_entries(infos) -> list[tuple[int, dict]]:
    """Yield (env_index, final_info_dict) entries across possible final_info layouts."""
    if not isinstance(infos, dict):
        return []

    final_infos = infos.get("final_info")
    if final_infos is None:
        return []

    entries: list[tuple[int, dict]] = []

    if isinstance(final_infos, (list, tuple)):
        seq = list(final_infos)
        mask = infos.get("_final_info")
        mask_arr = np.asarray(mask).reshape(-1) if mask is not None else None
        for i, item in enumerate(seq):
            if mask_arr is not None and i < mask_arr.size and not bool(mask_arr[i]):
                continue
            if isinstance(item, dict):
                entries.append((int(i), item))
        return entries

    if isinstance(final_infos, np.ndarray):
        seq = final_infos.tolist()
        mask = infos.get("_final_info")
        mask_arr = np.asarray(mask).reshape(-1) if mask is not None else None
        for i, item in enumerate(seq):
            if mask_arr is not None and i < mask_arr.size and not bool(mask_arr[i]):
                continue
            if isinstance(item, dict):
                entries.append((int(i), item))
        return entries

    if isinstance(final_infos, dict):
        # Single final_info payload.
        if (
            "episode" in final_infos
            or "terminal_observation" in final_infos
            or "final_observation" in final_infos
            or "final_obs" in final_infos
        ):
            return [(0, final_infos)]

        # Mapping keyed by env indices.
        for key, item in final_infos.items():
            if not isinstance(item, dict):
                continue
            try:
                idx = int(key)
            except Exception:
                idx = 0
            entries.append((idx, item))
        return entries

    return []


def _extract_final_observations(infos, num_envs: int) -> list[np.ndarray | None]:
    """Extract per-env terminal observations from vector-env info dict."""
    finals: list[np.ndarray | None] = [None] * int(num_envs)
    if not isinstance(infos, dict):
        return finals

    # Gymnasium vector envs expose one of these key pairs.
    key_pairs = (
        ("final_obs", "_final_obs"),
        ("final_observation", "_final_observation"),
    )
    for obs_key, mask_key in key_pairs:
        if obs_key not in infos:
            continue
        obs_values = infos.get(obs_key)
        mask_values = infos.get(mask_key)
        for i in range(int(num_envs)):
            has_final = True
            if mask_values is not None:
                has_final = bool(mask_values[i])
            if not has_final:
                continue
            obs_i = obs_values[i]
            if obs_i is None:
                continue
            finals[i] = np.asarray(obs_i, dtype=np.float32)
        break

    # Fallback for wrappers that only expose terminal observation in final_info.
    for i, final_info in _iter_final_info_entries(infos):
        if i < 0 or i >= int(num_envs):
            continue
        if finals[i] is not None:
            continue
        if not final_info:
            continue
        terminal_obs = final_info.get("terminal_observation")
        if terminal_obs is None:
            terminal_obs = final_info.get("final_observation")
        if terminal_obs is None:
            terminal_obs = final_info.get("final_obs")
        if terminal_obs is None:
            continue
        finals[i] = np.asarray(terminal_obs, dtype=np.float32)

    return finals


def _extract_episode_returns(infos) -> list[tuple[int, float]]:
    """Extract per-env episode returns from vector-env info dict."""
    returns: list[tuple[int, float]] = []
    if not isinstance(infos, dict):
        return returns

    # Vector envs commonly expose episode stats as infos["episode"] with mask infos["_episode"].
    if "episode" in infos:
        episode = infos["episode"]
        ep_returns = np.asarray(episode["r"]).reshape(-1)
        ep_mask = np.asarray(
            infos.get("_episode", np.ones_like(ep_returns, dtype=bool))
        ).reshape(-1)
        for idx in np.where(ep_mask)[0]:
            returns.append((int(idx), float(ep_returns[idx])))
        return returns

    # Some wrappers/setups expose terminal episode stats via final_info.
    for idx, item in _iter_final_info_entries(infos):
        if not item:
            continue
        episode = item.get("episode")
        if not isinstance(episode, dict):
            continue
        if "r" not in episode:
            continue
        episodic_return = float(np.asarray(episode["r"]).reshape(-1)[0])
        returns.append((int(idx), episodic_return))

    return returns


def _format_scalar(value: float) -> str:
    value_f = float(value)
    if np.isnan(value_f):
        return "nan"
    if np.isposinf(value_f):
        return "+inf"
    if np.isneginf(value_f):
        return "-inf"
    return f"{value_f:.6g}"


def _array_preview(values: np.ndarray, max_items: int = 12) -> str:
    flat = np.asarray(values).reshape(-1)
    if flat.size == 0:
        return "[]"
    preview = ", ".join(_format_scalar(v) for v in flat[:max_items])
    if flat.size > max_items:
        preview = f"{preview}, ..."
    return f"[{preview}]"


def _array_stats(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.float64)
    total = int(arr.size)
    finite_mask = np.isfinite(arr)
    finite_count = int(finite_mask.sum())
    if finite_count > 0:
        finite_vals = arr[finite_mask]
        min_value = _format_scalar(float(np.min(finite_vals)))
        max_value = _format_scalar(float(np.max(finite_vals)))
        range_text = f"{min_value}..{max_value}"
    else:
        range_text = "n/a"
    return (
        f"shape={arr.shape}, finite={finite_count}/{total}, "
        f"range={range_text}, sample={_array_preview(arr)}"
    )


def _space_summary(name: str, space: gym.Space) -> list[str]:
    lines = [f"{name}: type={type(space).__name__}, repr={space!r}"]
    shape = getattr(space, "shape", None)
    dtype = getattr(space, "dtype", None)
    if shape is not None or dtype is not None:
        lines.append(f"{name}: shape={shape}, dtype={dtype}")

    if isinstance(space, gym.spaces.Box):
        lines.append(f"{name}: low({_array_stats(space.low)})")
        lines.append(f"{name}: high({_array_stats(space.high)})")
    elif isinstance(space, gym.spaces.Discrete):
        lines.append(f"{name}: n={int(space.n)}")
    elif isinstance(space, gym.spaces.MultiDiscrete):
        lines.append(f"{name}: nvec={_array_preview(space.nvec)}")
    elif isinstance(space, gym.spaces.MultiBinary):
        lines.append(f"{name}: n={space.n}")

    return lines


def _wrapper_chain(env: gym.Env, max_depth: int = 64) -> list[str]:
    chain: list[str] = []
    current = env
    for _ in range(max_depth):
        if current is None:
            break
        chain.append(type(current).__name__)
        current = getattr(current, "env", None)
    return chain


def _resolve_env_id(env_id: str) -> str:
    if env_id.startswith("dm_control/"):
        parts = env_id.split("/")
        _, domain, task = parts
        return f"dm_control/{domain}-{task}-v0"
    return env_id


def _make_env(
    gym_id: str,
    seed: int,
):
    def thunk():
        resolved_env_id = _resolve_env_id(gym_id)
        env = gym.make(resolved_env_id)

        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        # Observation normalization is handled by RunningNorm inside the network.
        # dm_control rewards are bounded [0, 1]; normalization is unnecessary and harmful.
        if not gym_id.startswith("dm_control/"):
            env = gym.wrappers.NormalizeReward(env)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def find_wrapper(env, wrapper_type):
    current = env
    while current is not None:
        if isinstance(current, wrapper_type):
            return current
        current = getattr(current, "env", None)
    return None


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
        shared_encoder: bool = False,
        m_steps: int = 1,
    ):
        self.num_envs = int(num_envs)
        self.env_id = env_id
        self.resolved_env_id = _resolve_env_id(env_id)
        self.seed = seed
        self.gamma = float(gamma)
        self.advantage_estimator = str(advantage_estimator)
        self.gae_lambda = float(gae_lambda)
        self.m_steps = int(m_steps)

        self.envs = gym.vector.SyncVectorEnv(
            [
                _make_env(
                    gym_id=env_id,
                    seed=seed + i,
                )
                for i in range(self.num_envs)
            ],
        )
        obs_space = self.envs.single_observation_space
        act_space = self.envs.single_action_space

        obs_dim = infer_obs_dim(obs_space)
        if not isinstance(self.envs.action_space, gym.spaces.Box):
            # for vector envs, act_space above already set
            raise ValueError("VMPO only supports continuous action spaces.")
        act_shape = act_space.shape
        act_dim = int(np.prod(act_shape))
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.act_shape = act_shape

        self.agent = VMPOAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
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
            shared_encoder=shared_encoder,
            m_steps=m_steps,
        )

        self.rollout_steps = rollout_steps

        self.obs_buf: list[np.ndarray] = []
        self.actions_buf: list[np.ndarray] = []
        self.rewards_buf: list[np.ndarray] = []
        self.dones_buf: list[np.ndarray] = []
        self.timeout_bootstrap_buf: list[np.ndarray] = []
        self.values_buf: list[np.ndarray] = []
        self.means_buf: list[np.ndarray] = []
        self.log_stds_buf: list[np.ndarray] = []

        # episode returns per environment
        self.episode_return = np.zeros(self.num_envs, dtype=np.float32)
        self.last_eval = 0
        self.eval_episodes = 15
        self.eval_seed = self.seed + 1000

    def _print_env_summary(self) -> None:
        print(
            "[VMPO][env] "
            f"requested_id={self.env_id}, resolved_id={self.resolved_env_id}, "
            f"num_envs={self.num_envs}, seed_range=[{self.seed}, {self.seed + self.num_envs - 1}]"
        )
        print(
            "[VMPO][env] "
            f"vector_env_type={type(self.envs).__name__}, "
            f"obs_dim={self.obs_dim}, act_dim={self.act_dim}, act_shape={self.act_shape}"
        )

        first_env = None
        envs_list = getattr(self.envs, "envs", None)
        if isinstance(envs_list, list) and len(envs_list) > 0:
            first_env = envs_list[0]

        if first_env is not None:
            spec = getattr(first_env, "spec", None)
            if spec is not None:
                print(
                    "[VMPO][env] "
                    f"spec.id={getattr(spec, 'id', None)!r}, "
                    f"max_episode_steps={getattr(spec, 'max_episode_steps', None)}, "
                    f"reward_threshold={getattr(spec, 'reward_threshold', None)}"
                )
            reward_range = getattr(first_env, "reward_range", None)
            if reward_range is not None:
                print(f"[VMPO][env] reward_range={reward_range}")
            metadata = getattr(first_env, "metadata", None)
            if isinstance(metadata, dict):
                print(
                    "[VMPO][env] "
                    f"metadata.render_modes={metadata.get('render_modes')}, "
                    f"metadata.render_fps={metadata.get('render_fps')}"
                )
            wrappers = _wrapper_chain(first_env)
            if wrappers:
                print(f"[VMPO][env] wrappers[env0]={' -> '.join(wrappers)}")

        for line in _space_summary(
            "observation_space", self.envs.single_observation_space
        ):
            print(f"[VMPO][env] {line}")
        for line in _space_summary("action_space", self.envs.single_action_space):
            print(f"[VMPO][env] {line}")

    def _reset_rollout(self) -> None:
        self.obs_buf.clear()
        self.actions_buf.clear()
        self.rewards_buf.clear()
        self.dones_buf.clear()
        self.timeout_bootstrap_buf.clear()
        self.values_buf.clear()
        self.means_buf.clear()
        self.log_stds_buf.clear()

    def _rollout_full(self) -> bool:
        return len(self.obs_buf) >= self.rollout_steps

    def train(
        self,
        total_steps: int,
        out_dir: str,
    ):
        total_steps = int(total_steps)
        eval_interval = min(5000, max(1, total_steps // 150))
        console_log_interval = max(1, min(1_000, eval_interval))
        self._print_env_summary()
        print(
            "[VMPO] training started: "
            f"total_steps={total_steps}, "
            f"rollout_steps={self.rollout_steps}, "
            f"num_envs={self.num_envs}, "
            f"m_steps={self.m_steps}, "
            f"eval_interval={eval_interval}, "
            f"console_log_interval={console_log_interval}"
        )
        interval_metric_sums: Dict[str, float] = {}
        interval_update_count = 0
        interval_episode_count = 0
        interval_episode_sum = 0.0
        interval_episode_min = float("inf")
        interval_episode_max = float("-inf")
        total_update_count = 0
        best_eval_score = float("-inf")
        os.makedirs(out_dir, exist_ok=True)

        obs, _ = self.envs.reset()
        obs = np.asarray(obs, dtype=np.float32)
        global_step = 0
        env_steps = 0

        while global_step < total_steps:
            env_steps += 1
            global_step += self.num_envs
            action, value, mean, log_std = self.agent.act(obs, deterministic=False)

            next_obs, reward, terminated, truncated, infos = self.envs.step(action)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            terminated = np.asarray(terminated, dtype=bool)
            truncated = np.asarray(truncated, dtype=bool)
            done = terminated | truncated
            reward = np.asarray(reward, dtype=np.float32)

            # Time-limit fix:
            # Keep truncation as an episode boundary in target recursion, but for
            # truncated/non-terminated transitions bootstrap from V(final_obs).
            timeout_bootstrap = np.zeros(self.num_envs, dtype=np.float32)
            timeout_mask = np.logical_and(truncated, np.logical_not(terminated))
            if np.any(timeout_mask):
                final_obs = _extract_final_observations(infos, self.num_envs)
                timeout_indices = np.flatnonzero(timeout_mask)
                timeout_obs_batch = []
                timeout_obs_env_idx = []
                for env_i in timeout_indices:
                    obs_i = final_obs[int(env_i)]
                    if obs_i is None:
                        # Fallback for older wrappers without final_obs metadata.
                        obs_i = np.asarray(next_obs[int(env_i)], dtype=np.float32)
                    timeout_obs_batch.append(obs_i)
                    timeout_obs_env_idx.append(int(env_i))

                if timeout_obs_batch:
                    timeout_values = np.asarray(
                        self.agent.value(np.stack(timeout_obs_batch, axis=0)),
                        dtype=np.float32,
                    ).reshape(-1)
                    for env_i, v_i in zip(timeout_obs_env_idx, timeout_values):
                        timeout_bootstrap[env_i] = float(v_i)

            # Store step data (works for single- and multi-env)
            self.obs_buf.append(obs)
            self.actions_buf.append(action)
            self.rewards_buf.append(reward)
            self.dones_buf.append(done)
            self.timeout_bootstrap_buf.append(timeout_bootstrap)
            self.values_buf.append(value)
            self.means_buf.append(mean)
            self.log_stds_buf.append(log_std)

            obs = next_obs

            self.episode_return += reward

            episode_stats = _extract_episode_returns(infos)
            logged_envs: set[int] = set()
            for env_i, episode_return_f in episode_stats:
                logged_envs.add(int(env_i))
                log_wandb(
                    {"train/episode_return": episode_return_f},
                    step=global_step,
                    silent=True,
                )
                print(
                    f"[VMPO][episode] step={global_step}/{total_steps}, "
                    f"env={int(env_i)}, return={episode_return_f:.3f}"
                )
                interval_episode_count += 1
                interval_episode_sum += episode_return_f
                interval_episode_min = min(interval_episode_min, episode_return_f)
                interval_episode_max = max(interval_episode_max, episode_return_f)

            # Fallback when wrappers don't expose episode info payload.
            finished_mask = done.astype(bool)
            if np.any(finished_mask):
                finished_indices = np.flatnonzero(finished_mask)
                for env_i in finished_indices:
                    if int(env_i) in logged_envs:
                        self.episode_return[int(env_i)] = 0.0
                        continue
                    episode_return_f = float(self.episode_return[int(env_i)])
                    log_wandb(
                        {"train/episode_return": episode_return_f},
                        step=global_step,
                        silent=True,
                    )
                    print(
                        f"[VMPO][episode] step={global_step}/{total_steps}, "
                        f"env={int(env_i)}, return={episode_return_f:.3f}"
                    )
                    interval_episode_count += 1
                    interval_episode_sum += episode_return_f
                    interval_episode_min = min(interval_episode_min, episode_return_f)
                    interval_episode_max = max(interval_episode_max, episode_return_f)
                    self.episode_return[int(env_i)] = 0.0

            if self._rollout_full():
                # Stack collected arrays and reshape for batch processing
                obs_arr = np.stack(self.obs_buf)
                actions_arr = np.stack(self.actions_buf)
                rewards_arr = np.asarray(self.rewards_buf, dtype=np.float32)
                dones_arr = np.asarray(self.dones_buf, dtype=np.float32)
                timeout_bootstrap_arr = np.asarray(
                    self.timeout_bootstrap_buf, dtype=np.float32
                )
                values_arr = np.asarray(self.values_buf, dtype=np.float32)
                means_arr = np.stack(self.means_buf)
                log_stds_arr = np.stack(self.log_stds_buf)

                # Apply time-limit bootstrap correction per truncated transition.
                rewards_arr = rewards_arr + (self.gamma * timeout_bootstrap_arr)

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
                    "obs": torch.as_tensor(
                        obs_flat, dtype=torch.float32, device=self.agent.device
                    ),
                    "actions": torch.as_tensor(
                        actions_flat, dtype=torch.float32, device=self.agent.device
                    ),
                    "returns": torch.as_tensor(
                        returns_flat, dtype=torch.float32, device=self.agent.device
                    ),
                    "advantages": torch.as_tensor(
                        advantages_flat, dtype=torch.float32, device=self.agent.device
                    ),
                    "old_means": torch.as_tensor(
                        means_flat, dtype=torch.float32, device=self.agent.device
                    ),
                    "old_log_stds": torch.as_tensor(
                        log_stds_flat, dtype=torch.float32, device=self.agent.device
                    ),
                }

                metrics = self.agent.update(batch)
                log_wandb(metrics, step=global_step, silent=True)
                for key, value in metrics.items():
                    interval_metric_sums[key] = interval_metric_sums.get(
                        key, 0.0
                    ) + float(value)
                interval_update_count += 1
                total_update_count += 1

                self._reset_rollout()

            if self.last_eval < global_step // eval_interval:
                self.last_eval = global_step // eval_interval
                eval_envs = gym.vector.SyncVectorEnv(
                    [
                        _make_env(
                            self.env_id,
                            seed=self.eval_seed + i,
                        )
                        for i in range(self.eval_episodes)
                    ]
                )
                metrics = _evaluate_vectorized(
                    agent=self.agent,
                    eval_envs=eval_envs,
                    seed=self.eval_seed,
                )
                log_wandb(metrics, step=global_step, silent=True)
                print(
                    f"[VMPO][eval] step={global_step}/{total_steps}: {_format_metrics(metrics)}"
                )
                ckpt_payload = {"policy": self.agent.policy.state_dict()}
                ckpt_last_path = os.path.join(out_dir, "vmpo_last.pt")
                torch.save(ckpt_payload, ckpt_last_path)
                print(f"[VMPO][checkpoint] step={global_step}: saved {ckpt_last_path}")

                eval_score = float(metrics["eval/return_mean"])
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    ckpt_best_path = os.path.join(out_dir, "vmpo_best.pt")
                    torch.save(ckpt_payload, ckpt_best_path)
                    print(
                        f"[VMPO][checkpoint-best] step={global_step}: "
                        f"score={eval_score:.6f}, saved {ckpt_best_path}"
                    )

            should_print_progress = (
                global_step >= total_steps or global_step % console_log_interval == 0
            )
            if should_print_progress:
                progress = (
                    100.0 * float(min(global_step, total_steps)) / float(total_steps)
                )
                print(
                    "[VMPO][progress] "
                    f"step={global_step}/{total_steps} ({progress:.2f}%), "
                    f"env_steps={env_steps}, updates={total_update_count}"
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

        self.envs.close()


@torch.no_grad()
def _evaluate_vectorized(
    agent: VMPOAgent,
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

    episode_returns = np.zeros(n_episodes, dtype=np.float64)
    episode_lengths = np.zeros(n_episodes, dtype=np.int64)
    final_returns = []
    final_lengths = []

    # Track which envs have finished their first episode
    dones = np.zeros(n_episodes, dtype=bool)

    while len(final_returns) < n_episodes:
        # Pre-process observations
        obs = np.asarray(obs, dtype=np.float32)

        # Deterministic Action Selection
        # We call act() with deterministic=True to use the mean of the Gaussian
        action, _, _, _ = agent.act(obs, deterministic=True)


        # Step environment
        next_obs, reward, terminated, truncated, infos = eval_envs.step(action)
        reward_arr = np.asarray(reward, dtype=np.float64)
        active_mask = ~dones

        # Accumulate rewards for envs that haven't finished yet
        episode_returns[active_mask] += reward_arr[active_mask]
        episode_lengths[active_mask] += 1

        episode_stats = _extract_episode_returns(infos)
        episode_return_by_env = {env_i: ep_ret for env_i, ep_ret in episode_stats}

        # Check for completions
        # Gymnasium VectorEnv resets automatically; we catch the return in infos
        for i in range(n_episodes):
            if not dones[i] and (terminated[i] or truncated[i]):
                if i in episode_return_by_env:
                    final_returns.append(float(episode_return_by_env[i]))
                else:
                    final_returns.append(float(episode_returns[i]))
                final_lengths.append(int(episode_lengths[i]))
                dones[i] = True

        obs = next_obs

    if was_training:
        agent.policy.train()

    return {
        "eval/return_median": float(np.median(final_returns)),
        "eval/return_mean": float(np.mean(final_returns)),
        "eval/length_mean": float(np.mean(final_lengths)),
        "eval/return_std": float(np.std(final_returns)),
        "eval/return_min": float(np.min(final_returns)),
        "eval/return_max": float(np.max(final_returns)),
    }
