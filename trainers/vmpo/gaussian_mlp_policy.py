from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


# =========================
# PopArt value head
# =========================
class PopArt(nn.Module):
    """
    PopArt-normalised scalar value head.

    The network outputs a normalised value v_hat.
    The unnormalised value is: v = sigma * v_hat + mu.

    Statistics (mu, sigma) are updated externally via update_stats().
    """

    def __init__(self, in_dim: int, beta: float = 1e-4, eps: float = 1e-8):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

        # Running statistics of returns
        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("nu", torch.ones(1))  # second moment
        self.register_buffer("sigma", torch.ones(1))

        self.beta = beta
        self.eps = eps

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Normalised value v_hat
        return self.linear(h)

    def denormalize(self, v_hat: torch.Tensor) -> torch.Tensor:
        # Unnormalised value
        return self.sigma * v_hat + self.mu

    @torch.no_grad()
    def update_stats(self, returns: torch.Tensor) -> None:
        """
        Update running statistics using raw (unnormalised) returns
        and apply the PopArt invariance correction to the linear layer.
        """
        batch_mu = returns.mean()
        batch_nu = (returns**2).mean()

        mu_old = self.mu.clone()
        sigma_old = self.sigma.clone()

        # Update running moments
        self.mu.mul_(1.0 - self.beta).add_(self.beta * batch_mu)
        self.nu.mul_(1.0 - self.beta).add_(self.beta * batch_nu)

        self.sigma = torch.sqrt(torch.clamp(self.nu - self.mu**2, min=self.eps))

        # Invariance-preserving weight update
        w, b = self.linear.weight.data, self.linear.bias.data
        w.mul_(sigma_old / self.sigma)
        b.copy_((sigma_old * b + mu_old - self.mu) / self.sigma)


# =========================
# Encoder
# =========================
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

        # Remaining layers: Linear -> ELU (optionally final)
        for i in range(1, len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            if activate_final or i < len(layer_sizes) - 1:
                layers.append(nn.ELU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================
# Policy
# =========================
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

        # Shared encoder (LayerNorm MLP as in MPO)
        self.encoder = LayerNormMLP(obs_dim, hidden_sizes, activate_final=True)

        # Policy heads
        self.policy_mean = nn.Linear(hidden_sizes[-1], act_dim)
        self.policy_logstd = nn.Linear(hidden_sizes[-1], act_dim)

        # PopArt value head (normalised output)
        self.value_head = PopArt(hidden_sizes[-1])

        # Action scaling
        if action_low is None or action_high is None:
            action_low = -np.ones(act_dim, dtype=np.float32)
            action_high = np.ones(act_dim, dtype=np.float32)

        action_low_t = torch.tensor(action_low, dtype=torch.float32)
        action_high_t = torch.tensor(action_high, dtype=torch.float32)

        self.register_buffer("action_scale", (action_high_t - action_low_t) / 2.0)
        self.register_buffer("action_bias", (action_high_t + action_low_t) / 2.0)

        # Initialisation (matches typical MPO practice)
        nn.init.kaiming_normal_(
            self.policy_mean.weight,
            a=0.0,
            mode="fan_in",
            nonlinearity="linear",
        )
        nn.init.zeros_(self.policy_mean.bias)
        nn.init.zeros_(self.policy_logstd.weight)
        nn.init.zeros_(self.policy_logstd.bias)

    # -------- Policy --------
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(obs)
        mean = self.policy_mean(encoded)
        log_std = self.policy_logstd(encoded)

        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        log_std = torch.nan_to_num(log_std, nan=0.0, posinf=0.0, neginf=0.0)
        log_std = torch.clamp(log_std, -20.0, 2.0)

        return mean, log_std

    # -------- Value (PopArt) --------
    def value_norm(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Normalised value v_hat.
        Used ONLY for critic loss.
        """
        encoded = self.encoder(obs)
        return self.value_head(encoded)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Unnormalised value v = sigma * v_hat + mu.
        Used for advantages, logging, diagnostics.
        """
        encoded = self.encoder(obs)
        v_hat = self.value_head(encoded)
        return self.value_head.denormalize(v_hat)

    def forward_all(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          mean, log_std, UNNORMALISED value
        """
        mean, log_std = self.forward(obs)

        # Freeze encoder gradients for critic (MPO-style)
        with torch.no_grad():
            encoded = self.encoder(obs)
            v_hat = self.value_head(encoded)
            v = self.value_head.denormalize(v_hat)

        return mean, log_std, v

    # -------- Distribution utilities --------
    def log_prob(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        actions: torch.Tensor,
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
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        deterministic: bool,
        **kwargs,
    ) -> torch.Tensor:
        if deterministic:
            action = torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            action = torch.tanh(normal.rsample())

        return action * self.action_scale + self.action_bias
