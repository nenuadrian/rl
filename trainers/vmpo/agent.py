from __future__ import annotations

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


class PopArt(nn.Module):
    def __init__(self, beta: float = 0.99999, eps: float = 1e-5):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.popart_mean: torch.Tensor
        self.popart_std: torch.Tensor
        self.register_buffer("popart_mean", torch.zeros(1))
        self.register_buffer("popart_std", torch.ones(1))

    def normalize(self, targets: torch.Tensor) -> torch.Tensor:
        return (targets - self.popart_mean) / (self.popart_std + self.eps)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.popart_std + self.popart_mean

    @torch.no_grad()
    def update(self, targets: torch.Tensor, value_head: nn.Linear) -> None:
        batch_mean = targets.mean()
        batch_var = targets.var(unbiased=False)

        old_mean = self.popart_mean.clone()
        old_std = self.popart_std.clone()

        new_mean = self.beta * self.popart_mean + (1.0 - self.beta) * batch_mean
        new_var = self.beta * (self.popart_std**2) + (1.0 - self.beta) * batch_var
        new_std = torch.sqrt(new_var + self.eps)

        weight = value_head.weight.data
        bias = value_head.bias.data

        weight.mul_(old_std / new_std)
        bias.mul_(old_std)
        bias.add_(old_mean - new_mean)
        bias.div_(new_std)

        self.popart_mean.copy_(new_mean)
        self.popart_std.copy_(new_std)


class SquashedGaussianPolicy(nn.Module):
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
        lstm_input_dim = hidden_sizes[-1] + act_dim + 1
        self.lstm = nn.LSTM(lstm_input_dim, hidden_sizes[-1], batch_first=True)

        self.policy_mean = nn.Linear(hidden_sizes[-1], act_dim)
        self.policy_logstd = nn.Linear(hidden_sizes[-1], act_dim)
        self.value_head = nn.Linear(hidden_sizes[-1], 1)

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
        batch = obs.shape[0]
        prev_action = torch.zeros(
            (batch, self.action_scale.shape[0]),
            dtype=obs.dtype,
            device=obs.device,
        )
        prev_reward = torch.zeros((batch, 1), dtype=obs.dtype, device=obs.device)
        hidden = self.init_hidden(batch, obs.device)
        mean, log_std, _, _ = self.forward_step(obs, prev_action, prev_reward, hidden)
        return mean, log_std

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        return h, c

    def forward_step(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]
    ]:
        encoded = self.encoder(obs)
        lstm_input = torch.cat([encoded, prev_action, prev_reward], dim=-1).unsqueeze(1)
        lstm_out, next_hidden = self.lstm(lstm_input, hidden)
        lstm_out = lstm_out.squeeze(1)
        mean = self.policy_mean(lstm_out)
        log_std = self.policy_logstd(lstm_out)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        value_norm = self.value_head(lstm_out)
        return mean, log_std, value_norm, next_hidden

    def forward_sequence(
        self,
        obs: torch.Tensor,
        prev_actions: torch.Tensor,
        prev_rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = 1
        hidden = self.init_hidden(batch_size, obs.device)
        means = []
        log_stds = []
        values = []
        for t in range(obs.shape[0]):
            if t > 0 and dones[t - 1].item() > 0.5:
                hidden = self.init_hidden(batch_size, obs.device)
            mean, log_std, value_norm, hidden = self.forward_step(
                obs[t : t + 1],
                prev_actions[t : t + 1],
                prev_rewards[t : t + 1],
                hidden,
            )
            means.append(mean)
            log_stds.append(log_std)
            values.append(value_norm)
        return (
            torch.cat(means, dim=0),
            torch.cat(log_stds, dim=0),
            torch.cat(values, dim=0),
        )

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


@dataclass
class VMPOConfig:
    gamma: float = 0.99
    policy_lr: float = 1e-4
    value_lr: float = 1e-4
    topk_fraction: float = 0.5
    eta: float = 5.0
    kl_mean_coef: float = 1e-3
    kl_std_coef: float = 1e-3
    popart_beta: float = 0.99999
    max_grad_norm: float = 0.5


class VMPOAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        config: VMPOConfig | None = None,
    ):
        self.device = device
        self.config = config or VMPOConfig()

        self.policy = SquashedGaussianPolicy(
            obs_dim,
            act_dim,
            hidden_sizes=hidden_sizes,
            action_low=action_low,
            action_high=action_high,
        ).to(device)
        self.popart = PopArt(beta=self.config.popart_beta).to(device)

        self.policy_opt = torch.optim.Adam(
            list(self.policy.encoder.parameters())
            + list(self.policy.lstm.parameters())
            + list(self.policy.policy_mean.parameters())
            + list(self.policy.policy_logstd.parameters()),
            lr=self.config.policy_lr,
        )

        self.value_opt = torch.optim.Adam(
            list(self.policy.encoder.parameters())
            + list(self.policy.lstm.parameters())
            + list(self.policy.value_head.parameters()),
            lr=self.config.value_lr,
        )

    def init_hidden(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.policy.init_hidden(batch_size, self.device)

    def act(
        self,
        obs: np.ndarray,
        prev_action: np.ndarray,
        prev_reward: float,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[
        np.ndarray, float, np.ndarray, np.ndarray, Tuple[torch.Tensor, torch.Tensor]
    ]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        prev_action_t = torch.tensor(
            prev_action, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        prev_reward_t = torch.tensor(
            [prev_reward], dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            mean, log_std, value_norm, next_hidden = self.policy.forward_step(
                obs_t, prev_action_t, prev_reward_t, hidden
            )
            action_t = self.policy.sample_action(mean, log_std, deterministic)
            value = self.popart.denormalize(value_norm)

        return (
            action_t.detach().cpu().numpy().squeeze(0),
            float(value.item()),
            mean.detach().cpu().numpy().squeeze(0),
            log_std.detach().cpu().numpy().squeeze(0),
            next_hidden,
        )

    def value(
        self,
        obs: np.ndarray,
        prev_action: np.ndarray,
        prev_reward: float,
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ) -> float:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        prev_action_t = torch.tensor(
            prev_action, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        prev_reward_t = torch.tensor(
            [prev_reward], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            _, _, value_norm, _ = self.policy.forward_step(
                obs_t, prev_action_t, prev_reward_t, hidden
            )
            value = self.popart.denormalize(value_norm)
        return float(value.item())

    def update(self, batch: dict) -> dict:
        obs = batch["obs"]
        actions = batch["actions"]
        prev_actions = batch["prev_actions"]
        prev_rewards = batch["prev_rewards"]
        dones = batch["dones"]
        returns = batch["returns"]
        advantages = batch["advantages"].squeeze(-1)
        old_means = batch["old_means"]
        old_log_stds = batch["old_log_stds"]

        k = max(1, int(self.config.topk_fraction * advantages.numel()))
        topk_vals, _ = torch.topk(advantages, k)
        threshold = topk_vals.min()
        mask = (advantages >= threshold).float()
        weights = torch.exp(advantages / self.config.eta) * mask
        weights = weights / (weights.sum() + 1e-10)

        mean, log_std, _ = self.policy.forward_sequence(
            obs, prev_actions, prev_rewards, dones
        )
        log_prob = self.policy.log_prob(mean, log_std, actions).squeeze(-1)

        old_std = old_log_stds.exp()
        new_std = log_std.exp()
        kl_mean = ((mean - old_means) ** 2 / (2.0 * (old_std**2))).sum(dim=-1)
        kl_std = 0.5 * (
            (new_std / old_std) ** 2 - 1.0 - 2.0 * (log_std - old_log_stds)
        ).sum(dim=-1)

        policy_loss = -(weights.detach() * log_prob).mean()
        policy_loss = policy_loss + self.config.kl_mean_coef * kl_mean.mean()
        policy_loss = policy_loss + self.config.kl_std_coef * kl_std.mean()

        # entropy bonus
        entropy = (log_std + 0.5 * np.log(2 * np.pi * np.e)).sum(dim=-1)
        entropy_coef = 1e-2
        policy_loss -= entropy_coef * entropy.mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.policy_opt.step()

        with torch.no_grad():
            self.popart.update(returns, self.policy.value_head)
        returns_norm = self.popart.normalize(returns)

        _, _, value_norm = self.policy.forward_sequence(
            obs, prev_actions, prev_rewards, dones
        )
        value_loss = F.mse_loss(value_norm, returns_norm)

        self.value_opt.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.value_opt.step()

        return {
            "loss/policy": float(policy_loss.item()),
            "loss/value": float(value_loss.item()),
            "kl/mean": float(kl_mean.mean().item()),
            "kl/std": float(kl_std.mean().item()),
        }
