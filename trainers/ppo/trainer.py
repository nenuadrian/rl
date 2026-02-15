from __future__ import annotations

import os
import random
import time
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from utils.wandb_utils import log_wandb


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


def _resolve_env_id(env_id: str) -> str:
    if env_id.startswith("dm_control/"):
        parts = env_id.split("/")
        if len(parts) != 3:
            raise ValueError(
                "Expected dm_control env id format 'dm_control/<domain>/<task>', "
                f"got '{env_id}'"
            )
        _, domain, task = parts
        return f"dm_control/{domain}-{task}-v0"
    return env_id


def make_env(
    gym_id: str,
    seed: int,
    normalize_observation: bool = True,
):
    def thunk():
        resolved_env_id = _resolve_env_id(gym_id)
        env = gym.make(resolved_env_id)

        # Keep dm_control compatibility while preserving implementation-details PPO logic.
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        if normalize_observation:
            env = gym.wrappers.NormalizeObservation(env)
            env = _transform_observation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = _transform_reward(env, lambda reward: np.clip(reward, -10, 10))

        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_eval_env(gym_id: str, seed: int, normalize_observation: bool = True):
    resolved_env_id = _resolve_env_id(gym_id)
    env = gym.make(resolved_env_id)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    if normalize_observation:
        env = gym.wrappers.NormalizeObservation(env)
        env = _transform_observation(env, lambda obs: np.clip(obs, -10, 10))

    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def find_wrapper(env, wrapper_type):
    current = env
    while current is not None:
        if isinstance(current, wrapper_type):
            return current
        current = getattr(current, "env", None)
    return None


def sync_obs_rms(train_env, eval_env):
    train_obs_norm = find_wrapper(train_env, gym.wrappers.NormalizeObservation)
    eval_obs_norm = find_wrapper(eval_env, gym.wrappers.NormalizeObservation)
    if train_obs_norm is None or eval_obs_norm is None:
        return
    eval_obs_norm.obs_rms.mean = np.copy(train_obs_norm.obs_rms.mean)
    eval_obs_norm.obs_rms.var = np.copy(train_obs_norm.obs_rms.var)
    eval_obs_norm.obs_rms.count = train_obs_norm.obs_rms.count


def evaluate(
    agent: "Agent",
    eval_env: gym.Env,
    device: torch.device,
    num_episodes: int,
    deterministic: bool = True,
):
    was_training = agent.training
    agent.eval()
    episode_returns = []
    episode_lengths = []

    eval_obs_norm = find_wrapper(eval_env, gym.wrappers.NormalizeObservation)
    old_update_running_mean = None
    if eval_obs_norm is not None and hasattr(eval_obs_norm, "update_running_mean"):
        old_update_running_mean = eval_obs_norm.update_running_mean
        eval_obs_norm.update_running_mean = False

    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _ = eval_env.reset()
            done = False
            episodic_return = 0.0
            episodic_length = 0
            while not done:
                obs_tensor = torch.as_tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                action_mean = agent.actor_mean(obs_tensor)
                if deterministic:
                    action = action_mean
                else:
                    action_std = torch.exp(agent.actor_logstd.expand_as(action_mean))
                    action = Normal(action_mean, action_std).sample()
                obs, reward, terminated, truncated, _ = eval_env.step(
                    action.squeeze(0).cpu().numpy()
                )
                done = terminated or truncated
                episodic_return += float(reward)
                episodic_length += 1
            episode_returns.append(episodic_return)
            episode_lengths.append(episodic_length)

    if old_update_running_mean is not None:
        eval_obs_norm.update_running_mean = old_update_running_mean
    if was_training:
        agent.train()

    return np.array(episode_returns), np.array(episode_lengths)


def log_episode_stats(infos, global_step: int):
    if not isinstance(infos, dict):
        return

    # Vector envs commonly expose episode stats as infos["episode"] with infos["_episode"] mask.
    if "episode" in infos:
        episode = infos["episode"]
        ep_returns = np.asarray(episode["r"]).reshape(-1)
        ep_lengths = np.asarray(episode["l"]).reshape(-1)
        ep_mask = np.asarray(
            infos.get("_episode", np.ones_like(ep_returns, dtype=bool))
        ).reshape(-1)
        for idx in np.where(ep_mask)[0]:
            episodic_return = float(ep_returns[idx])
            episodic_length = float(ep_lengths[idx])
            print(f"global_step={global_step}, episodic_return={episodic_return}")
            log_wandb(
                {
                    "charts/episodic_return": episodic_return,
                    "charts/episodic_length": episodic_length,
                },
                step=global_step,
                silent=True,
            )

    # Some wrappers/setups expose terminal episode stats via final_info.
    elif "final_info" in infos:
        for item in infos["final_info"]:
            if item and "episode" in item:
                episodic_return = float(np.asarray(item["episode"]["r"]).reshape(-1)[0])
                episodic_length = float(np.asarray(item["episode"]["l"]).reshape(-1)[0])
                print(f"global_step={global_step}, episodic_return={episodic_return}")
                log_wandb(
                    {
                        "charts/episodic_return": episodic_return,
                        "charts/episodic_length": episodic_length,
                    },
                    step=global_step,
                    silent=True,
                )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        policy_layer_sizes: Tuple[int, ...],
        value_layer_sizes: Tuple[int, ...],
    ):
        super().__init__()
        if len(policy_layer_sizes) == 0:
            raise ValueError("policy_layer_sizes must contain at least one layer size")
        if len(value_layer_sizes) == 0:
            raise ValueError("critic_layer_sizes must contain at least one layer size")

        self.critic = self._build_mlp(
            input_dim=obs_dim,
            hidden_layer_sizes=value_layer_sizes,
            output_dim=1,
            output_std=1.0,
        )
        self.actor_mean = self._build_mlp(
            input_dim=obs_dim,
            hidden_layer_sizes=policy_layer_sizes,
            output_dim=act_dim,
            output_std=0.01,
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    @staticmethod
    def _build_mlp(
        input_dim: int,
        hidden_layer_sizes: Tuple[int, ...],
        output_dim: int,
        output_std: float,
    ) -> nn.Sequential:
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            layers.extend([layer_init(nn.Linear(last_dim, hidden_dim)), nn.Tanh()])
            last_dim = hidden_dim
        layers.append(layer_init(nn.Linear(last_dim, output_dim), std=output_std))
        return nn.Sequential(*layers)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


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
        clip_ratio: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.02,
        norm_adv: bool = True,
        clip_vloss: bool = True,
        anneal_lr: bool = True,
        normalize_obs: bool = True,
        num_envs: int = 1,
        optimizer_type: str = "adam",
        sgd_momentum: float = 0.9,
    ):
        self.env_id = str(env_id)
        self.seed = int(seed)
        self.device = device

        self.num_envs = int(num_envs)
        self.num_steps = int(rollout_steps)
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(minibatch_size)

        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.gae = True

        self.update_epochs = int(update_epochs)
        self.clip_coef = float(clip_ratio)
        self.clip_vloss = bool(clip_vloss)
        self.norm_adv = bool(norm_adv)
        self.ent_coef = float(ent_coef)
        self.vf_coef = float(vf_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.target_kl = (
            None if target_kl is None or float(target_kl) <= 0.0 else float(target_kl)
        )
        self.anneal_lr = bool(anneal_lr)

        self.learning_rate = float(policy_lr)
        self.optimizer_type = str(optimizer_type).strip().lower()
        self.sgd_momentum = float(sgd_momentum)

        self.normalize_obs = bool(normalize_obs)

        self.eval_episodes = 50
        self.eval_deterministic = True

        # Keep ppo-implementation-details behavior deterministic.
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.envs = gym.vector.SyncVectorEnv(
            [
                make_env(
                    self.env_id,
                    self.seed + i,
                    normalize_observation=self.normalize_obs,
                )
                for i in range(self.num_envs)
            ]
        )
        self.eval_env = make_eval_env(
            self.env_id,
            self.seed + 10_000,
            normalize_observation=self.normalize_obs,
        )
        assert isinstance(
            self.envs.single_action_space, gym.spaces.Box
        ), "only continuous action space is supported"

        obs_shape = self.envs.single_observation_space.shape
        act_shape = self.envs.single_action_space.shape
        if obs_shape is None:
            raise ValueError("observation space has no shape")
        if act_shape is None:
            raise ValueError("action space has no shape")

        obs_dim = int(np.array(obs_shape).prod())
        act_dim = int(np.prod(act_shape))

        self.agent = Agent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            policy_layer_sizes=tuple(policy_layer_sizes),
            value_layer_sizes=tuple(critic_layer_sizes),
        ).to(self.device)

        self.optimizer = self._build_optimizer()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer_type == "adam":
            return optim.Adam(
                self.agent.parameters(),
                lr=self.learning_rate,
                eps=1e-5,
            )
        if self.optimizer_type == "sgd":
            return optim.SGD(
                self.agent.parameters(),
                lr=self.learning_rate,
                momentum=self.sgd_momentum,
            )
        raise ValueError(
            f"Unsupported PPO optimizer_type '{self.optimizer_type}'. "
            "Expected one of: adam, sgd."
        )

    def train(
        self,
        total_steps: int,
        out_dir: str,
    ):
        total_steps = int(total_steps)
        eval_interval = max(1, total_steps // 150)

        num_updates = total_steps // self.batch_size
        if num_updates <= 0:
            print(
                "[PPO] no updates scheduled because requested_total_steps < batch_size "
                f"({total_steps} < {self.batch_size})."
            )
            self.envs.close()
            self.eval_env.close()
            return

        obs = torch.zeros(
            (self.num_steps, self.num_envs) + self.envs.single_observation_space.shape,
            device=self.device,
        )
        actions = torch.zeros(
            (self.num_steps, self.num_envs) + self.envs.single_action_space.shape,
            device=self.device,
        )
        logprobs = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        dones = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        values = torch.zeros((self.num_steps, self.num_envs), device=self.device)

        global_step = 0
        start_time = time.time()

        next_obs, _ = self.envs.reset()
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        next_done = torch.zeros(self.num_envs, device=self.device)

        last_eval = 0
        best_eval_score = float("-inf")
        os.makedirs(out_dir, exist_ok=True)

        for update in range(1, num_updates + 1):
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                global_step += self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(
                        next_obs
                    )
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs_np, reward, terminated, truncated, infos = self.envs.step(
                    action.cpu().numpy()
                )
                done = np.logical_or(terminated, truncated)

                rewards[step] = torch.as_tensor(
                    reward, dtype=torch.float32, device=self.device
                ).view(-1)
                next_obs = torch.as_tensor(
                    next_obs_np, dtype=torch.float32, device=self.device
                )
                next_done = torch.as_tensor(
                    done, dtype=torch.float32, device=self.device
                )

                log_episode_stats(infos, global_step)

                if global_step // eval_interval > last_eval:
                    last_eval = global_step // eval_interval
                    sync_obs_rms(self.envs.envs[0], self.eval_env)
                    eval_returns, eval_lengths = evaluate(
                        self.agent,
                        self.eval_env,
                        self.device,
                        self.eval_episodes,
                        deterministic=self.eval_deterministic,
                    )
                    metrics = {
                        "eval/return_max": float(np.max(eval_returns)),
                        "eval/return_std": float(np.std(eval_returns)),
                        "eval/return_mean": float(np.mean(eval_returns)),
                        "eval/length_mean": float(np.mean(eval_lengths)),
                        "eval/return_min": float(np.min(eval_returns)),
                    }
                    print(f"eval global_step={global_step}, " f"{metrics}")
                    log_wandb(
                        metrics,
                        step=global_step,
                        silent=True,
                    )

                    ckpt_payload = {
                        "actor_mean": self.agent.actor_mean.state_dict(),
                        "actor_logstd": self.agent.actor_logstd.detach().cpu(),
                        "critic": self.agent.critic.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    }
                    ckpt_last_path = os.path.join(out_dir, "ppo_last.pt")
                    torch.save(ckpt_payload, ckpt_last_path)
                    print(
                        f"[PPO][checkpoint] step={global_step}/{total_steps}: "
                        f"saved {ckpt_last_path}"
                    )

                    eval_score = float(metrics["eval/return_mean"])
                    if eval_score > best_eval_score:
                        best_eval_score = eval_score
                        ckpt_best_path = os.path.join(out_dir, "ppo_best.pt")
                        torch.save(ckpt_payload, ckpt_best_path)
                        print(
                            f"[PPO][checkpoint-best] step={global_step}/{total_steps}: "
                            f"score={eval_score:.6f}, saved {ckpt_best_path}"
                        )

            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                if self.gae:
                    advantages = torch.zeros_like(rewards, device=self.device)
                    lastgaelam = 0
                    for t in reversed(range(self.num_steps)):
                        if t == self.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = (
                            rewards[t]
                            + self.gamma * nextvalues * nextnonterminal
                            - values[t]
                        )
                        advantages[t] = lastgaelam = (
                            delta
                            + self.gamma
                            * self.gae_lambda
                            * nextnonterminal
                            * lastgaelam
                        )
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards, device=self.device)
                    for t in reversed(range(self.num_steps)):
                        if t == self.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = (
                            rewards[t] + self.gamma * nextnonterminal * next_return
                        )
                    advantages = returns - values

            b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            b_inds = np.arange(self.batch_size)
            clipfracs = []

            # Track last minibatch values for logging parity with reference implementation.
            pg_loss = torch.tensor(0.0, device=self.device)
            v_loss = torch.tensor(0.0, device=self.device)
            entropy_loss = torch.tensor(0.0, device=self.device)
            old_approx_kl = torch.tensor(0.0, device=self.device)
            approx_kl = torch.tensor(0.0, device=self.device)

            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                if self.target_kl is not None:
                    if approx_kl > self.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            sps = int(global_step / max(time.time() - start_time, 1e-8))
            log_wandb(
                {
                    "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/old_approx_kl": old_approx_kl.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
                    "losses/explained_variance": float(explained_var),
                    "charts/SPS": float(sps),
                },
                step=global_step,
                silent=True,
            )
            print("SPS:", sps)

        self.envs.close()
        self.eval_env.close()
