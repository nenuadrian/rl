from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def _mlp(in_dim: int, hidden_sizes: Tuple[int, ...]) -> nn.Sequential:
    layers = []
    last_dim = in_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(last_dim, size))
        layers.append(nn.ReLU())
        last_dim = size
    return nn.Sequential(*layers)


def orthogonal_init(module: nn.Module, gain: float = 1.0):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        nn.init.constant_(module.bias, 0.0)


class GaussianMLPPolicy(nn.Module):
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

        self.encoder.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2)))
        orthogonal_init(self.policy_mean, gain=0.01)
        orthogonal_init(self.policy_logstd, gain=0.01)
        self.policy_logstd.bias.data.fill_(0.0)
        orthogonal_init(self.value_head, gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(obs)
        mean = self.policy_mean(encoded)
        log_std = self.policy_logstd(encoded)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        return mean, log_std

    def value_norm(self, obs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(obs)
        return self.value_head(encoded)

    def forward_all(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        value_norm = self.value_norm(obs)
        return mean, log_std, value_norm

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
        self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool, **kwargs
    ) -> torch.Tensor:
        if deterministic:
            action = torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            action = torch.tanh(normal.rsample())
        return action * self.action_scale + self.action_bias
