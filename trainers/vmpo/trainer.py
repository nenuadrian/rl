from __future__ import annotations

import os
from typing import Dict, Mapping, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch

from trainers.vmpo.agent import VMPOAgent
from trainers.vmpo.targets import compute_rollout_targets
from utils.env import infer_obs_dim
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


def _find_wrapper(env: gym.Env, wrapper_type: type[gym.Wrapper]):
    """Return the first wrapper of type `wrapper_type` in an env wrapper chain."""
    current = env
    while current is not None:
        if isinstance(current, wrapper_type):
            return current
        current = getattr(current, "env", None)
    return None


def _merge_obs_rms_stats(
    stats_seq: Sequence[tuple[np.ndarray, np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Merge per-env running mean/var stats into a single aggregate RMS state."""
    if len(stats_seq) == 0:
        return None

    mean_acc = np.array(stats_seq[0][0], dtype=np.float64, copy=True)
    var_acc = np.array(stats_seq[0][1], dtype=np.float64, copy=True)
    count_acc = float(stats_seq[0][2])

    for mean_i, var_i, count_i in stats_seq[1:]:
        count_i = float(count_i)
        if count_i <= 0.0:
            continue
        mean_i = np.array(mean_i, dtype=np.float64, copy=False)
        var_i = np.array(var_i, dtype=np.float64, copy=False)
        if count_acc <= 0.0:
            mean_acc = mean_i.copy()
            var_acc = var_i.copy()
            count_acc = count_i
            continue

        total = count_acc + count_i  # LaTeX: n = n_{acc} + n_i
        delta = mean_i - mean_acc  # LaTeX: \delta = \mu_i - \mu_{acc}
        mean_new = mean_acc + delta * (count_i / total)  # LaTeX: \mu' = \mu_{acc} + \delta\frac{n_i}{n}
        m2_acc = var_acc * count_acc  # LaTeX: M_{2,acc} = \sigma_{acc}^2 n_{acc}
        m2_i = var_i * count_i  # LaTeX: M_{2,i} = \sigma_i^2 n_i
        m2_total = m2_acc + m2_i + (delta**2) * (count_acc * count_i / total)  # LaTeX: M_2 = M_{2,acc} + M_{2,i} + \delta^2\frac{n_{acc}n_i}{n}

        mean_acc = mean_new
        var_acc = m2_total / total  # LaTeX: \sigma^2 = \frac{M_2}{n}
        count_acc = total

    return mean_acc, var_acc, count_acc


def _collect_vector_obs_rms_stats(
    vec_env: gym.vector.VectorEnv,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Collect merged NormalizeObservation running stats from a vector env."""
    envs = getattr(vec_env, "envs", None)
    if envs is None:
        return None

    stats_seq: list[tuple[np.ndarray, np.ndarray, float]] = []
    for env in envs:
        obs_norm = _find_wrapper(env, gym.wrappers.NormalizeObservation)
        if obs_norm is None or not hasattr(obs_norm, "obs_rms"):
            continue
        obs_rms = obs_norm.obs_rms
        stats_seq.append(
            (
                np.array(obs_rms.mean, dtype=np.float64, copy=True),
                np.array(obs_rms.var, dtype=np.float64, copy=True),
                float(obs_rms.count),
            )
        )
    return _merge_obs_rms_stats(stats_seq)


def _apply_obs_rms_stats(
    vec_env: gym.vector.VectorEnv,
    obs_rms_stats: tuple[np.ndarray, np.ndarray, float] | None,
) -> None:
    """Copy provided observation running stats into all eval envs."""
    if obs_rms_stats is None:
        return

    envs = getattr(vec_env, "envs", None)
    if envs is None:
        return

    mean, var, count = obs_rms_stats
    for env in envs:
        obs_norm = _find_wrapper(env, gym.wrappers.NormalizeObservation)
        if obs_norm is None or not hasattr(obs_norm, "obs_rms"):
            continue
        obs_norm.obs_rms.mean = np.array(mean, dtype=np.float64, copy=True)
        obs_norm.obs_rms.var = np.array(var, dtype=np.float64, copy=True)
        obs_norm.obs_rms.count = float(count)
        if hasattr(obs_norm, "update_running_mean"):
            try:
                obs_norm.update_running_mean = False
            except Exception:
                pass


def _make_env(
    env_id: str,
    *,
    seed: int | None = None,
    gamma: float = 0.99,
    normalize_observation: bool = True,
    clip_observation: float | None = 10.0,
    normalize_reward: bool = True,
    clip_reward: float | None = 10.0,
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
        shared_encoder: bool = False,
    ):
        self.num_envs = int(num_envs)
        self.env_id = env_id
        self.seed = seed
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
            shared_encoder=shared_encoder,
        )

        self.rollout_steps = rollout_steps

        self.obs_buf: list[np.ndarray] = []
        self.actions_buf: list[np.ndarray] = []
        self.rewards_buf: list[np.ndarray] = []
        self.dones_buf: list[np.ndarray] = []
        self.restarting_weights_buf: list[np.ndarray] = []
        self.importance_weights_buf: list[np.ndarray] = []
        self.timeout_bootstrap_buf: list[np.ndarray] = []
        self.values_buf: list[np.ndarray] = []
        self.means_buf: list[np.ndarray] = []
        self.log_stds_buf: list[np.ndarray] = []

        # episode returns per environment
        self.episode_return = np.zeros(self.num_envs, dtype=np.float32)
        # Current observation is episode start until the first action is taken.
        self.episode_start_flags = np.ones(self.num_envs, dtype=bool)
        self.last_eval = 0
        self.eval_episodes = 50
        self.eval_seed = self.seed + 1000
        self.eval_env = gym.vector.SyncVectorEnv(
            [
                (
                    lambda i=i: _make_env(
                        self.env_id,
                        seed=self.eval_seed + i,
                        gamma=self.gamma,
                        # Evaluate on raw environment reward so returns are comparable
                        # across checkpoints/runs. Training can still use reward norm.
                        normalize_reward=False,
                    )
                )
                for i in range(self.eval_episodes)
            ]
        )

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
        self.restarting_weights_buf.clear()
        self.importance_weights_buf.clear()
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
        updates_per_step: int = 1,
    ):
        total_steps = int(total_steps)
        eval_interval = max(1, total_steps // 50)  # LaTeX: \Delta t_{eval} = \max\left(1, \left\lfloor \frac{T_{total}}{50} \right\rfloor\right)
        console_log_interval = max(1, min(1_000, eval_interval))
        print(
            "[VMPO] training started: "
            f"total_steps={total_steps}, "
            f"rollout_steps={self.rollout_steps}, "
            f"num_envs={self.num_envs}, "
            f"updates_per_step={int(updates_per_step)}, "
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

        obs, _ = self.env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        self.episode_start_flags = np.ones(self.num_envs, dtype=bool)
        global_step = 0
        env_steps = 0

        while global_step < total_steps:
            env_steps += 1
            global_step += self.num_envs  # LaTeX: t \leftarrow t + N
            restarting_weights = 1.0 - self.episode_start_flags.astype(np.float32)  # LaTeX: w_t^{restart} = 1 - \mathbf{1}_{episode\_start}
            importance_weights = np.ones(self.num_envs, dtype=np.float32)
            action, value, mean, log_std = self.agent.act(obs, deterministic=False)

            next_obs, reward, terminated, truncated, infos = self.env.step(action)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            terminated = np.asarray(terminated, dtype=bool)
            truncated = np.asarray(truncated, dtype=bool)
            done = terminated | truncated  # LaTeX: d_t = d_t^{term} \lor d_t^{trunc}
            reward = np.asarray(reward, dtype=np.float32)

            # Time-limit fix:
            # Keep truncation as an episode boundary in target recursion, but for
            # truncated/non-terminated transitions bootstrap from V(final_obs).
            timeout_bootstrap = np.zeros(self.num_envs, dtype=np.float32)
            timeout_mask = np.logical_and(truncated, np.logical_not(terminated))  # LaTeX: m_t^{timeout} = d_t^{trunc} \land \neg d_t^{term}
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
            self.restarting_weights_buf.append(restarting_weights)
            self.importance_weights_buf.append(importance_weights)
            self.timeout_bootstrap_buf.append(timeout_bootstrap)
            self.values_buf.append(value)
            self.means_buf.append(mean)
            self.log_stds_buf.append(log_std)

            obs = next_obs
            self.episode_start_flags = done.astype(bool)

            self.episode_return += reward  # LaTeX: G_i \leftarrow G_i + r_{t,i}

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
                restarting_weights_arr = np.asarray(
                    self.restarting_weights_buf, dtype=np.float32
                )
                importance_weights_arr = np.asarray(
                    self.importance_weights_buf, dtype=np.float32
                )
                timeout_bootstrap_arr = np.asarray(
                    self.timeout_bootstrap_buf, dtype=np.float32
                )
                values_arr = np.asarray(self.values_buf, dtype=np.float32)
                means_arr = np.stack(self.means_buf)
                log_stds_arr = np.stack(self.log_stds_buf)

                # Apply time-limit bootstrap correction per truncated transition.
                rewards_arr = rewards_arr + (self.gamma * timeout_bootstrap_arr)  # LaTeX: r_t' = r_t + \gamma b_t^{timeout}

                last_value = self.agent.value(obs)
                last_value = last_value * (1.0 - dones_arr[-1])  # LaTeX: V_T = V(s_T)(1 - d_T)

                # obs_arr shape (T, N, obs_dim) -> flatten to (T*N, obs_dim)
                T, N, _ = obs_arr.shape
                obs_flat = obs_arr.reshape(T * N, -1)  # LaTeX: \mathbf{S} \in \mathbb{R}^{(TN)\times d_s}
                actions_flat = actions_arr.reshape(T * N, -1)  # LaTeX: \mathbf{A} \in \mathbb{R}^{(TN)\times d_a}
                rewards_flat = rewards_arr.reshape(T, N)
                dones_flat = dones_arr.reshape(T, N)
                restarting_weights_flat = restarting_weights_arr.reshape(T * N, 1)
                importance_weights_flat = importance_weights_arr.reshape(T * N, 1)
                values_flat = values_arr.reshape(T, N)
                means_flat = means_arr.reshape(T * N, -1)  # LaTeX: \mu_{old} \in \mathbb{R}^{(TN)\times d_a}
                log_stds_flat = log_stds_arr.reshape(T * N, -1)  # LaTeX: \log \sigma_{old} \in \mathbb{R}^{(TN)\times d_a}

                returns, advantages = compute_rollout_targets(  # LaTeX: (R_t, A_t) \leftarrow \text{Targets}(r_t, d_t, V_t, V_T, \gamma, \lambda)
                    rewards=rewards_flat,
                    dones=dones_flat,
                    values=values_flat,
                    last_value=last_value,
                    gamma=self.gamma,
                    estimator=self.advantage_estimator,
                    gae_lambda=self.gae_lambda,
                )
                returns_flat = returns.reshape(T * N, 1)  # LaTeX: \mathbf{R} \in \mathbb{R}^{(TN)\times 1}
                advantages_flat = advantages.reshape(T * N, 1)  # LaTeX: \mathbf{A} \in \mathbb{R}^{(TN)\times 1}

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
                    "restarting_weights": torch.as_tensor(
                        restarting_weights_flat,
                        dtype=torch.float32,
                        device=self.agent.device,
                    ),
                    "importance_weights": torch.as_tensor(
                        importance_weights_flat,
                        dtype=torch.float32,
                        device=self.agent.device,
                    ),
                    "old_means": torch.as_tensor(
                        means_flat, dtype=torch.float32, device=self.agent.device
                    ),
                    "old_log_stds": torch.as_tensor(
                        log_stds_flat, dtype=torch.float32, device=self.agent.device
                    ),
                }

                metrics = {}
                for _ in range(updates_per_step):
                    metrics = self.agent.update(batch)
                    log_wandb(metrics, step=global_step, silent=True)
                    for key, value in metrics.items():
                        interval_metric_sums[key] = interval_metric_sums.get(key, 0.0) + float(value)  # LaTeX: M_k \leftarrow M_k + m_k
                    interval_update_count += 1
                    total_update_count += 1

                self._reset_rollout()

            if self.last_eval < global_step // eval_interval:
                self.last_eval = global_step // eval_interval
                metrics = _evaluate_vectorized(
                    agent=self.agent,
                    eval_envs=self.eval_env,
                    seed=self.eval_seed,
                    obs_rms_stats=_collect_vector_obs_rms_stats(self.env),
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
                # LaTeX: p = 100 \cdot \frac{\min(t, T_{total})}{T_{total}}
                progress = 100.0 * float(min(global_step, total_steps)) / float(total_steps)
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
                    # LaTeX: \bar{m}_k = \frac{1}{U}\sum_{u=1}^{U} m_{k,u}
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
        self.eval_env.close()


@torch.no_grad()
def _evaluate_vectorized(
    agent: VMPOAgent,
    eval_envs: gym.vector.VectorEnv,
    seed: int = 42,
    obs_rms_stats: tuple[np.ndarray, np.ndarray, float] | None = None,
) -> Dict[str, float]:
    """
    High-performance vectorized evaluation.
    Runs all n_episodes in parallel using a SyncVectorEnv.
    """
    n_episodes = int(eval_envs.num_envs)
    _apply_obs_rms_stats(eval_envs, obs_rms_stats)

    was_training = agent.policy.training
    agent.policy.eval()
    obs, _ = eval_envs.reset(seed=seed)

    episode_returns = np.zeros(n_episodes)
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

        # Clip actions to valid range
        action = np.clip(
            action,
            eval_envs.single_action_space.low,
            eval_envs.single_action_space.high,
        )

        # Step environment
        next_obs, reward, terminated, truncated, infos = eval_envs.step(action)
        reward_arr = np.asarray(reward, dtype=np.float32)
        active_mask = ~dones

        # Accumulate rewards for envs that haven't finished yet
        episode_returns[active_mask] += reward_arr[active_mask]  # LaTeX: G_i \leftarrow G_i + r_{t,i}
        episode_lengths[active_mask] += 1

        # Check for completions
        # Gymnasium VectorEnv resets automatically; we catch the return in infos
        for i in range(n_episodes):
            if not dones[i] and (terminated[i] or truncated[i]):
                final_returns.append(episode_returns[i])  # LaTeX: \mathcal{G} \leftarrow \mathcal{G} \cup \{G_i\}
                final_lengths.append(int(episode_lengths[i]))
                dones[i] = True

        obs = next_obs
    if was_training:
        agent.policy.train()

    return {
        "eval/return_median": float(np.median(final_returns)),  # LaTeX: \tilde{G} = \operatorname{median}(\mathcal{G})
        "eval/return_mean": float(np.mean(final_returns)),  # LaTeX: \bar{G} = \frac{1}{|\mathcal{G}|}\sum_{g\in\mathcal{G}} g
        "eval/length_mean": float(np.mean(final_lengths)),
        "eval/return_std": float(np.std(final_returns)),  # LaTeX: \sigma_G = \sqrt{\frac{1}{|\mathcal{G}|}\sum_{g\in\mathcal{G}}(g-\bar{G})^2}
        "eval/return_min": float(np.min(final_returns)),  # LaTeX: G_{\min} = \min(\mathcal{G})
        "eval/return_max": float(np.max(final_returns)),  # LaTeX: G_{\max} = \max(\mathcal{G})
    }
