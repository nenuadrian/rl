from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MPOEncoder(nn.Module):
    """
    Standard encoder for MPO/V-MPO:
    Linear -> LayerNorm -> Tanh (First Layer)
    Linear -> ELU (Subsequent Layers)
    """

    def __init__(
        self,
        in_dim: int,
        layer_sizes: Tuple[int, ...],
        activate_final: bool = True,
        ppo_like: bool = False,
    ):
        super().__init__()
        if len(layer_sizes) == 0:
            raise ValueError("layer_sizes must contain at least one hidden size")
        layers = []

        if ppo_like:
            # PPO-like baseline stack: orthogonal Linear + Tanh at each hidden layer.
            last_dim = in_dim
            for hidden_dim in layer_sizes:
                layers.append(_layer_init(nn.Linear(last_dim, hidden_dim)))
                layers.append(nn.Tanh())
                last_dim = hidden_dim
            self.net = nn.Sequential(*layers)
            return

        # MPO Architecture typically uses LayerNorm after the first projection
        layers.append(nn.Linear(in_dim, layer_sizes[0]))
        layers.append(nn.LayerNorm(layer_sizes[0]))
        layers.append(nn.Tanh())

        for i in range(1, len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            if activate_final or i < len(layer_sizes) - 1:
                layers.append(nn.ELU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SquashedGaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        policy_layer_sizes: Tuple[int, ...] = (256, 256),
        value_layer_sizes: Tuple[int, ...] = (256, 256),
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        shared_encoder: bool = False,
        ppo_like_backbone: bool = False,
    ):
        super().__init__()

        self.shared_encoder = shared_encoder
        self.ppo_like_backbone = bool(ppo_like_backbone)
        if shared_encoder:
            self.policy_encoder = MPOEncoder(
                obs_dim,
                policy_layer_sizes,
                ppo_like=self.ppo_like_backbone,
            )
            self.value_encoder = self.policy_encoder
        else:
            self.policy_encoder = MPOEncoder(
                obs_dim,
                policy_layer_sizes,
                ppo_like=self.ppo_like_backbone,
            )
            self.value_encoder = MPOEncoder(
                obs_dim,
                value_layer_sizes,
                ppo_like=self.ppo_like_backbone,
            )

        self.policy_mean = nn.Linear(policy_layer_sizes[-1], act_dim)
        if self.ppo_like_backbone:
            self.policy_logstd = nn.Parameter(torch.zeros(1, act_dim))
        else:
            self.policy_logstd = nn.Linear(policy_layer_sizes[-1], act_dim)

        value_head_in_dim = (
            policy_layer_sizes[-1] if shared_encoder else value_layer_sizes[-1]
        )
        self.value_head = nn.Linear(value_head_in_dim, 1)

        # 3. Action Scaling
        if action_low is None or action_high is None:
            # Assume normalized environment [-1, 1] if not provided
            action_low = -np.ones(act_dim, dtype=np.float32)
            action_high = np.ones(act_dim, dtype=np.float32)

        action_low_t = torch.tensor(action_low, dtype=torch.float32)
        action_high_t = torch.tensor(action_high, dtype=torch.float32)

        self.register_buffer("action_scale", (action_high_t - action_low_t) / 2.0)
        self.register_buffer("action_bias", (action_high_t + action_low_t) / 2.0)

        if self.ppo_like_backbone:
            _layer_init(self.policy_mean, std=0.01)
            _layer_init(self.value_head, std=1.0)
        else:
            # Keep prior VMPO init path as default behavior.
            nn.init.xavier_uniform_(self.policy_mean.weight)
            nn.init.zeros_(self.policy_mean.bias)
            nn.init.xavier_uniform_(self.policy_logstd.weight)
            # Initialize log_std to match roughly std=0.5 to 1.0 initially
            nn.init.constant_(self.policy_logstd.bias, -0.5)
            nn.init.xavier_uniform_(self.value_head.weight)
            nn.init.zeros_(self.value_head.bias)

    def policy_logstd_parameters(self) -> list[nn.Parameter]:
        if isinstance(self.policy_logstd, nn.Parameter):
            return [self.policy_logstd]
        return list(self.policy_logstd.parameters())

    def _mean_and_log_std(self, encoded_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.policy_mean(encoded_obs)
        if isinstance(self.policy_logstd, nn.Parameter):
            log_std = self.policy_logstd.expand_as(mean)
        else:
            log_std = self.policy_logstd(encoded_obs)
        return mean, torch.clamp(log_std, -20.0, 2.0)

    def get_policy_dist_params(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Internal helper to get raw distribution parameters."""
        h = self.policy_encoder(obs)
        return self._mean_and_log_std(h)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_policy_dist_params(obs)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        if self.shared_encoder:
            h = self.policy_encoder(obs)
        else:
            h = self.value_encoder(obs)
        return self.value_head(h)

    def forward_all(self, obs):
        h = self.policy_encoder(obs)
        mean, log_std = self._mean_and_log_std(h)
        h_val = h if self.shared_encoder else self.value_encoder(obs)
        v = self.value_head(h_val)
        return mean, log_std, v

    def log_prob(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates log prob of actions.
        NOTE: 'actions' are the final environment actions (scaled).
        """
        std = log_std.exp()
        normal = Normal(mean, std)

        # 1. Undo Scale
        # y_t \in [-1, 1] (roughly)
        y_t = (actions - self.action_bias) / self.action_scale

        # 2. Stability Clamp
        # Prevents NaNs in arctanh when y_t is exactly -1 or 1 due to fp errors
        y_t = torch.clamp(y_t, -0.999999, 0.999999)

        # 3. Undo Tanh (Arctanh)
        x_t = 0.5 * torch.log((1.0 + y_t) / (1.0 - y_t))

        # 4. Log Prob Calculation
        # log p(a) = log p(x) - log det(dy/dx) - log det(da/dy)
        log_prob = normal.log_prob(x_t)

        # Jacobian of Tanh: 1 - tanh^2(x) = 1 - y^2
        jacobian_tanh = torch.log(1.0 - y_t.pow(2) + 1e-6)

        # Jacobian of Scale: action_scale
        # We sum over the action dimension (dim=-1)
        log_prob = log_prob - jacobian_tanh
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob = log_prob - torch.log(self.action_scale).sum()

        return log_prob

    def sample_action(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action: Scaled action for environment
            log_prob: Log probability of that action
        """
        if deterministic:
            y_t = torch.tanh(mean)
            log_prob = torch.zeros((mean.shape[0], 1), device=mean.device)  # Dummy
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()  # Reparameterized sample
            y_t = torch.tanh(x_t)

            # Calculate log_prob using x_t directly (More stable than inversion)
            log_prob = normal.log_prob(x_t)
            log_prob = log_prob - torch.log(1.0 - y_t.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            log_prob = log_prob - torch.log(self.action_scale).sum()

        action = y_t * self.action_scale + self.action_bias
        return action, log_prob
