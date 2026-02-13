from __future__ import annotations

import os
from typing import Dict, Mapping, Tuple

import gymnasium as gym
import numpy as np
import torch

from trainers.ppo.agent import PPOAgent
from utils.env import infer_obs_dim, wrap_record_video
from utils.wandb_utils import log_wandb
from trainers.ppo.rollout_buffer import RolloutBuffer


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
) -> gym.Env:
    if env_id.startswith("dm_control/"):
        _, domain, task = env_id.split("/")
        env = gym.make(f"dm_control/{domain}-{task}-v0", render_mode=render_mode)
    else:
        env = gym.make(env_id, render_mode=render_mode)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)

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


class PPOTrainer:
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
        norm_adv: bool = True,
        clip_vloss: bool = True,
        anneal_lr: bool = True,
        normalize_obs: bool = False,
        num_envs: int = 1,
        optimizer_type: str = "adam",
        sgd_momentum: float = 0.9,
        capture_video: bool = False,
        run_name: str | None = None,
    ):
        self.seed = seed
        self.num_envs = int(num_envs)
        self.env_id = env_id
        self.capture_video = bool(capture_video)
        safe_run_name = (
            run_name if run_name is not None else f"ppo-{env_id}-seed{seed}"
        ).replace("/", "-")
        self.video_dir = os.path.join("videos", safe_run_name)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.target_kl = float(target_kl)
        self.anneal_lr = bool(anneal_lr)
        env_fns = [
            (
                lambda i=i: self._make_train_env(
                    env_id=env_id,
                    seed=seed + i,
                    env_index=i,
                    gamma=self.gamma,
                    normalize_obs=normalize_obs,
                )
            )
            for i in range(self.num_envs)
        ]
        self.env = gym.vector.SyncVectorEnv(env_fns)
        obs_space = self.env.single_observation_space
        act_space = self.env.single_action_space
        obs_dim = infer_obs_dim(obs_space)
        if not isinstance(act_space, gym.spaces.Box):
            raise ValueError("PPO only supports continuous action spaces.")
        if act_space.shape is None:
            raise ValueError("Action space has no shape.")
        act_dim = int(np.prod(np.array(act_space.shape)))
        action_low = getattr(act_space, "low", None)
        action_high = getattr(act_space, "high", None)
        if action_low is None or action_high is None:
            action_low = -np.ones(act_dim, dtype=np.float32)
            action_high = np.ones(act_dim, dtype=np.float32)
        self.agent = PPOAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            action_low=action_low,
            action_high=action_high,
            device=device,
            policy_layer_sizes=policy_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            policy_lr=policy_lr,
            value_lr=value_lr,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            norm_adv=norm_adv,
            clip_vloss=clip_vloss,
            anneal_lr=anneal_lr,
            optimizer_type=optimizer_type,
            sgd_momentum=sgd_momentum,
        )
        self.rollout_steps = rollout_steps
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.normalize_obs = bool(normalize_obs)
        self.buffer = RolloutBuffer.create(
            obs_dim, act_dim, rollout_steps, self.num_envs
        )
        self._base_policy_lr = float(policy_lr)
        self._base_value_lr = float(value_lr)
        self.episode_return = np.zeros(self.num_envs, dtype=np.float32)
        self.last_eval = 0
        self.last_checkpoint = 0

    def _make_train_env(
        self,
        *,
        env_id: str,
        seed: int,
        env_index: int,
        gamma: float,
        normalize_obs: bool,
    ):
        should_record = self.capture_video and env_index == 0
        env = _make_env(
            env_id,
            seed=seed,
            render_mode="rgb_array" if should_record else None,
            gamma=gamma,
            normalize_observation=normalize_obs,
        )
        if should_record:
            env = wrap_record_video(env, self.video_dir)
        return env

    def train(
        self,
        total_steps: int,
        eval_interval: int,
        save_interval: int,
        out_dir: str,
    ):
        console_log_interval = max(
            1, min(1_000, eval_interval if eval_interval > 0 else 1_000)
        )
        batch_size = self.rollout_steps * self.num_envs
        num_updates = total_steps // batch_size
        scheduled_steps = num_updates * batch_size
        print(
            "[PPO] training started: "
            f"requested_total_steps={total_steps}, "
            f"scheduled_total_steps={scheduled_steps}, "
            f"rollout_steps={self.rollout_steps}, "
            f"num_envs={self.num_envs}, "
            f"batch_size={batch_size}, "
            f"update_epochs={self.update_epochs}, "
            f"minibatch_size={self.minibatch_size}, "
            f"console_log_interval={console_log_interval}"
        )
        if num_updates <= 0:
            print(
                "[PPO] no updates scheduled because requested_total_steps < batch_size "
                f"({total_steps} < {batch_size})."
            )
            self.env.close()
            return
        interval_metric_sums: Dict[str, float] = {}
        interval_update_count = 0
        interval_episode_count = 0
        interval_episode_sum = 0.0
        interval_episode_min = float("inf")
        interval_episode_max = float("-inf")

        obs, _ = self.env.reset()
        obs = np.asarray(obs, dtype=np.float32)

        step = 0
        for update_idx in range(num_updates):
            if self.anneal_lr:
                frac = 1.0 - (update_idx / float(num_updates))
                frac = max(frac, 0.0)
                for group in self.agent.policy_opt.param_groups:
                    group["lr"] = self._base_policy_lr * frac
                for group in self.agent.value_opt.param_groups:
                    group["lr"] = self._base_value_lr * frac
                log_wandb(
                    {
                        "train/lr_policy": self.agent.policy_opt.param_groups[0]["lr"],
                        "train/lr_value": self.agent.value_opt.param_groups[0]["lr"],
                    },
                    step=step,
                    silent=True,
                )

            for t in range(self.rollout_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)

                action_t, logp_t, value_t = self.agent.act(obs_t, deterministic=False)
                action = action_t.cpu().numpy()
                action_for_buffer = action.copy()
                logp = logp_t.cpu().numpy().squeeze(-1)
                value = value_t.cpu().numpy().squeeze(-1)
                action = np.clip(
                    action,
                    self.env.single_action_space.low,
                    self.env.single_action_space.high,
                )

                next_obs, reward, terminated, truncated, infos = self.env.step(action)
                next_obs = np.asarray(next_obs, dtype=np.float32)
                done = np.asarray(terminated) | np.asarray(truncated)
                reward = np.asarray(reward)
                final_infos = infos.get("final_info", None) if isinstance(infos, dict) else None

                # Defensive check
                if not (isinstance(next_obs, np.ndarray) and next_obs.ndim in (1, 2)):
                    raise ValueError(
                        f"unexpected observation array at step: shape={getattr(next_obs,'shape',None)}"
                    )

                for i in range(self.num_envs):
                    self.buffer.add(
                        t,
                        i,
                        obs[i],
                        action_for_buffer[i],
                        float(reward[i]),
                        float(done[i]),
                        float(value[i]),
                        float(logp[i]),
                    )
                    self.episode_return[i] += float(reward[i])
                    # Log episode return for each env when done
                    if done[i]:
                        episode_return = None
                        if final_infos is not None and i < len(final_infos):
                            final_info = final_infos[i]
                            if final_info and "episode" in final_info:
                                episode_return = float(final_info["episode"]["r"])
                        if episode_return is None:
                            episode_return = float(self.episode_return[i])
                        log_wandb(
                            {"train/episode_return": episode_return},
                            step=step + 1,
                            silent=True,
                        )
                        print(
                            f"[PPO][episode] step={step + 1}/{total_steps}, "
                            f"env={i}, return={episode_return:.3f}"
                        )
                        interval_episode_count += 1
                        interval_episode_sum += episode_return
                        interval_episode_min = min(interval_episode_min, episode_return)
                        interval_episode_max = max(interval_episode_max, episode_return)
                        self.episode_return[i] = 0.0

                obs = next_obs
                step += self.num_envs

            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.agent.device)
                last_values = self.agent.value(obs_t).cpu().numpy().squeeze(-1)

            data = self.buffer.compute_returns_advantages(
                last_values, self.gamma, self.gae_lambda
            )
            obs_f, act_f, logp_f, val_f, ret_f, adv_f = data
            var_y = np.var(ret_f)
            explained_var = np.nan if var_y == 0 else 1.0 - np.var(ret_f - val_f) / var_y
            log_wandb({"train/explained_variance": float(explained_var)}, step=step, silent=True)

            obs_f = torch.tensor(obs_f, dtype=torch.float32, device=self.agent.device)
            act_f = torch.tensor(act_f, dtype=torch.float32, device=self.agent.device)
            logp_f = torch.tensor(logp_f, dtype=torch.float32, device=self.agent.device)
            val_f = torch.tensor(val_f, dtype=torch.float32, device=self.agent.device)
            ret_f = torch.tensor(ret_f, dtype=torch.float32, device=self.agent.device)
            adv_f = torch.tensor(adv_f, dtype=torch.float32, device=self.agent.device)

            early_stop = False
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
                    for key, value in metrics.items():
                        interval_metric_sums[key] = (
                            interval_metric_sums.get(key, 0.0) + float(value)
                        )
                    interval_update_count += 1
                    approx_kl = metrics.get("approx_kl", None)
                    if (
                        approx_kl is not None
                        and self.target_kl > 0.0
                        and approx_kl > self.target_kl
                    ):
                        # Stop PPO updates early to preserve trust region.
                        early_stop = True
                        break
                if early_stop:
                    break

            self.buffer.reset()

            if eval_interval > 0 and self.last_eval < step // eval_interval:
                self.last_eval = step // eval_interval
                metrics = _evaluate_vectorized(
                    agent=self.agent,
                    env_id=self.env_id,
                    seed=self.seed + 1000,
                    gamma=self.gamma,
                    normalize_observation=self.normalize_obs,
                )
                log_wandb(metrics, step=step)
                print(
                    f"[PPO][eval] step={step}/{scheduled_steps}: "
                    f"{_format_metrics(metrics)}"
                )

            if save_interval > 0 and self.last_checkpoint < step // save_interval:
                self.last_checkpoint = step // save_interval
                os.makedirs(out_dir, exist_ok=True)
                ckpt_path = os.path.join(out_dir, "ppo.pt")
                torch.save(
                    {
                        "policy": self.agent.policy.state_dict(),
                        "value": self.agent.value.state_dict(),
                    },
                    ckpt_path,
                )
                print(
                    f"[PPO][checkpoint] step={step}/{scheduled_steps}: saved {ckpt_path}"
                )

            update_display = update_idx + 1
            step_display = step
            should_print_progress = (
                update_display >= num_updates or step_display % console_log_interval == 0
            )
            if should_print_progress:
                progress = 100.0 * float(step_display) / float(scheduled_steps)
                print(
                    "[PPO][progress] "
                    f"step={step_display}/{scheduled_steps} ({progress:.2f}%), "
                    f"update={update_display}/{num_updates}"
                )
                if interval_episode_count > 0:
                    print(
                        "[PPO][episode-stats] "
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
                        "[PPO][train-metrics] "
                        f"updates={interval_update_count}, {_format_metrics(mean_metrics)}"
                    )
                    interval_metric_sums.clear()
                    interval_update_count = 0

        self.env.close()

@torch.no_grad()
def _evaluate_vectorized(
    agent: PPOAgent,
    env_id: str,
    n_episodes: int = 10,
    seed: int = 42,
    gamma: float = 0.99,
    normalize_observation: bool = True,
) -> Dict[str, float]:
    """
    High-performance vectorized evaluation.
    Runs all n_episodes in parallel using a SyncVectorEnv.
    """
    eval_envs = gym.vector.SyncVectorEnv(
        [
            (
                lambda i=i: _make_env(
                    env_id,
                    seed=seed + i,
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
        obs = np.asarray(obs, dtype=np.float32)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device)
        action = agent.policy.sample_action(obs_t, deterministic=True).cpu().numpy()
        action = np.clip(
            action,
            eval_envs.single_action_space.low,
            eval_envs.single_action_space.high,
        )

        next_obs, reward, terminated, truncated, infos = eval_envs.step(action)
        episode_returns += np.asarray(reward, dtype=np.float32)
        done = np.asarray(terminated) | np.asarray(truncated)

        if isinstance(infos, dict) and "final_info" in infos:
            for i, final_info in enumerate(infos["final_info"]):
                if not dones[i] and final_info and "episode" in final_info:
                    final_returns.append(float(final_info["episode"]["r"]))
                    dones[i] = True
        else:
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
