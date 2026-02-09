from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class LayerNormMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        layer_sizes: Tuple[int, ...],
        activate_final: bool = False,
    ):
        super().__init__()

        layers = []

        # First layer: Linear -> LayerNorm -> tanh
        layers.append(nn.Linear(in_dim, layer_sizes[0]))
        layers.append(nn.LayerNorm(layer_sizes[0]))
        layers.append(nn.Tanh())

        # Remaining layers: ELU
        for i in range(1, len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            if activate_final or i < len(layer_sizes) - 1:
                layers.append(nn.ELU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallInitLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, std: float = 0.01):
        super().__init__(in_features, out_features)
        nn.init.trunc_normal_(self.weight, std=std)
        nn.init.zeros_(self.bias)


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

        # Match MPO's network buildout (LayerNorm torso + tanh/ELU activations).
        self.encoder = LayerNormMLP(obs_dim, hidden_sizes, activate_final=True)
        self.policy_mean = nn.Linear(hidden_sizes[-1], act_dim)
        self.policy_logstd = nn.Linear(hidden_sizes[-1], act_dim)
        self.value_head = SmallInitLinear(hidden_sizes[-1], 1, std=0.01)

        if action_low is None or action_high is None:
            action_low = -np.ones(act_dim, dtype=np.float32)
            action_high = np.ones(act_dim, dtype=np.float32)

        action_low_t = torch.tensor(action_low, dtype=torch.float32)
        action_high_t = torch.tensor(action_high, dtype=torch.float32)
        self.action_scale: torch.Tensor
        self.action_bias: torch.Tensor
        self.register_buffer("action_scale", (action_high_t - action_low_t) / 2.0)
        self.register_buffer("action_bias", (action_high_t + action_low_t) / 2.0)

        nn.init.kaiming_normal_(
            self.policy_mean.weight, a=0.0, mode="fan_in", nonlinearity="linear"
        )
        nn.init.zeros_(self.policy_mean.bias)
        nn.init.zeros_(self.policy_logstd.weight)
        nn.init.zeros_(self.policy_logstd.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(obs)
        mean = self.policy_mean(encoded)
        log_std = self.policy_logstd(encoded)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        log_std = torch.nan_to_num(log_std, nan=0.0, posinf=0.0, neginf=0.0)
        log_std = torch.clamp(log_std, -20.0, 2.0)
        return mean, log_std

    def value_norm(self, obs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(obs)
        return self.value_head(encoded)

    def forward_all(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        # "Freeze" encoder gradients for the value head
        value_norm = self.value_norm(obs.detach())
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
