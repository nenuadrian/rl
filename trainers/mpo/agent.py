from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def _mlp(in_dim: int, hidden_sizes: Tuple[int, ...]) -> nn.Sequential:
    layers = []
    last_dim = in_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(last_dim, size))
        layers.append(nn.ReLU())
        last_dim = size
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        self.net = nn.Sequential(
            _mlp(obs_dim + act_dim, hidden_sizes),
            nn.Linear(hidden_sizes[-1], 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class MPONetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
    ):
        super().__init__()
        if len(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must have at least one layer.")

        self.encoder = _mlp(obs_dim, hidden_sizes)
        self.policy_mean = nn.Linear(hidden_sizes[-1], act_dim)
        self.policy_logstd = nn.Parameter(torch.zeros(act_dim))

        if action_low is None or action_high is None:
            action_low = -np.ones(act_dim, dtype=np.float32)
            action_high = np.ones(act_dim, dtype=np.float32)

        action_low_t = torch.tensor(action_low, dtype=torch.float32)
        action_high_t = torch.tensor(action_high, dtype=torch.float32)
        self.action_scale: torch.Tensor
        self.action_bias: torch.Tensor
        self.register_buffer("action_scale", (action_high_t - action_low_t) / 2.0)
        self.register_buffer("action_bias", (action_high_t + action_low_t) / 2.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(obs)
        mean = self.policy_mean(h)
        log_std = self.policy_logstd.expand_as(mean)
        return mean, log_std

    def log_prob(
        self, mean: torch.Tensor, log_std: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        std = log_std.exp()
        normal = Normal(mean, std)
        y_t = (actions - self.action_bias) / self.action_scale
        y_t = torch.clamp(y_t, -0.999999, 0.999999)
        x_t = 0.5 * torch.log((1.0 + y_t) / (1.0 - y_t))
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob - torch.log(1.0 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob = log_prob - torch.log(self.action_scale).sum()
        return log_prob

    def sample_action(
        self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool
    ) -> torch.Tensor:
        if deterministic:
            action = torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            action = torch.tanh(normal.rsample())
        return action * self.action_scale + self.action_bias

    def sample_actions(self, obs: torch.Tensor, num_actions: int) -> torch.Tensor:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        eps = normal.rsample(sample_shape=(num_actions,))
        eps = eps.permute(1, 0, 2)
        actions = torch.tanh(eps)
        return actions * self.action_scale + self.action_bias


@dataclass
class MPOConfig:
    gamma: float = 0.99
    tau: float = 0.005
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    eta: float = 1.0
    kl_mean_coef: float = 1.0
    kl_std_coef: float = 1.0
    max_grad_norm: float = 1.0
    action_samples: int = 16


class MPOAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        config: MPOConfig | None = None,
    ):
        self.device = device
        self.config = config or MPOConfig()

        self.policy = MPONetwork(
            obs_dim,
            act_dim,
            hidden_sizes=hidden_sizes,
            action_low=action_low,
            action_high=action_high,
        ).to(device)
        self.q1 = QNetwork(obs_dim, act_dim, hidden_sizes).to(device)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_sizes).to(device)

        self.q1_target = copy.deepcopy(self.q1).to(device)
        self.q2_target = copy.deepcopy(self.q2).to(device)
        self.q1_target.eval()
        self.q2_target.eval()

        self.policy_opt = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.policy_lr
        )
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=self.config.q_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=self.config.q_lr)

    def act(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.policy(obs_t)
            action = self.policy.sample_action(mean, log_std, deterministic)
        return (
            action.detach().cpu().numpy().squeeze(0),
            mean.detach().cpu().numpy().squeeze(0),
            log_std.detach().cpu().numpy().squeeze(0),
        )

    def update(self, batch: dict) -> dict:
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(
            batch["actions"], dtype=torch.float32, device=self.device
        )
        rewards = torch.tensor(
            batch["rewards"], dtype=torch.float32, device=self.device
        )
        next_obs = torch.tensor(
            batch["next_obs"], dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device)
        old_means = torch.tensor(
            batch["old_means"], dtype=torch.float32, device=self.device
        )
        old_log_stds = torch.tensor(
            batch["old_log_stds"], dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            next_mean, next_log_std = self.policy(next_obs)
            next_actions = self.policy.sample_action(
                next_mean, next_log_std, deterministic=False
            )
            q1_target = self.q1_target(next_obs, next_actions)
            q2_target = self.q2_target(next_obs, next_actions)
            q_target = torch.min(q1_target, q2_target)
            target = rewards + (1.0 - dones) * self.config.gamma * q_target

        q1 = self.q1(obs, actions)
        q2 = self.q2(obs, actions)

        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), self.config.max_grad_norm)
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), self.config.max_grad_norm)
        self.q2_opt.step()

        mean, log_std = self.policy(obs)
        with torch.no_grad():
            sampled_actions = self.policy.sample_actions(
                obs, num_actions=self.config.action_samples
            )
            batch_size = obs.shape[0]
            obs_rep = obs.unsqueeze(1).expand(
                batch_size, self.config.action_samples, obs.shape[-1]
            )
            obs_flat = obs_rep.reshape(-1, obs.shape[-1])
            act_flat = sampled_actions.reshape(-1, sampled_actions.shape[-1])
            q1_vals = self.q1(obs_flat, act_flat)
            q2_vals = self.q2(obs_flat, act_flat)
            q_vals = torch.min(q1_vals, q2_vals)
            q_vals = q_vals.reshape(batch_size, self.config.action_samples)
            weights = torch.softmax(q_vals / self.config.eta, dim=1)

        mean_exp = mean.unsqueeze(1).expand(
            batch_size, self.config.action_samples, mean.shape[-1]
        )
        log_std_exp = log_std.unsqueeze(1).expand_as(mean_exp)
        log_prob = self.policy.log_prob(
            mean_exp.reshape(-1, mean.shape[-1]),
            log_std_exp.reshape(-1, log_std.shape[-1]),
            sampled_actions.reshape(-1, sampled_actions.shape[-1]),
        )
        log_prob = log_prob.reshape(batch_size, self.config.action_samples)

        old_std = old_log_stds.exp()
        new_std = log_std.exp()
        kl_mean = ((mean - old_means) ** 2 / (2.0 * (old_std**2))).sum(dim=-1)
        kl_std = 0.5 * (
            (new_std / old_std) ** 2
            - 1.0
            - 2.0 * (log_std - old_log_stds)
        ).sum(dim=-1)

        policy_loss = -(weights.detach() * log_prob).sum(dim=1).mean()
        policy_loss = policy_loss + self.config.kl_mean_coef * kl_mean.mean()
        policy_loss = policy_loss + self.config.kl_std_coef * kl_std.mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.policy_opt.step()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return {
            "loss/q1": float(q1_loss.item()),
            "loss/q2": float(q2_loss.item()),
            "loss/policy": float(policy_loss.item()),
            "kl/mean": float(kl_mean.mean().item()),
            "kl/std": float(kl_std.mean().item()),
        }

    def _soft_update(self, net: nn.Module, target: nn.Module) -> None:
        tau = self.config.tau
        for param, target_param in zip(net.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * param.data)
