from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class PopArt(nn.Module):
    """
    PopArt-normalised scalar value head.
    Preserves outputs precisely while adaptively rescaling targets.
    """

    def __init__(
        self,
        in_dim: int,
        beta: float,
        eps: float,
        min_sigma: float,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

        # Initialize linear layer to be close to 0 to prevent initial shock
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("nu", torch.ones(1))
        self.register_buffer("sigma", torch.ones(1))

        self.beta = beta
        self.eps = eps
        self.min_sigma = min_sigma

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns the Normalized value (v_hat)."""
        return self.linear(h)

    def denormalize(self, v_hat: torch.Tensor) -> torch.Tensor:
        """Converts v_hat -> v."""
        return self.sigma * v_hat + self.mu

    @torch.no_grad()
    def update_stats(self, returns: torch.Tensor) -> None:
        batch_mu = returns.mean()
        batch_nu = (returns**2).mean()

        mu_old = self.mu.clone()
        sigma_old = self.sigma.clone()

        # Update EMA moments
        self.mu.mul_(1.0 - self.beta).add_(self.beta * batch_mu)
        self.nu.mul_(1.0 - self.beta).add_(self.beta * batch_nu)

        # Variance = E[x^2] - (E[x])^2
        var = torch.clamp(self.nu - self.mu**2, min=self.min_sigma**2)
        self.sigma.copy_(torch.sqrt(var))

        # Update weights to preserve output: v_old(x) == v_new(x)
        # w_new = (sigma_old / sigma_new) * w_old
        self.linear.weight.mul_(sigma_old / self.sigma)
        # b_new = (sigma_old * b_old + mu_old - mu_new) / sigma_new
        self.linear.bias.copy_(
            (sigma_old * self.linear.bias + mu_old - self.mu) / self.sigma
        )


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
    ):
        super().__init__()
        layers = []

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
        popart_beta: float = 1e-4,  # Standard VMPO beta
        popart_eps: float = 1e-4,
        popart_min_sigma: float = 1e-4,
        policy_layer_sizes: Tuple[int, ...] = (256, 256),
        value_layer_sizes: Tuple[int, ...] = (256, 256),
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        shared_encoder: bool = False,  # Default to False for research quality
    ):
        super().__init__()

        self.shared_encoder = shared_encoder
        if shared_encoder:
            self.policy_encoder = MPOEncoder(obs_dim, policy_layer_sizes)
            self.value_encoder = self.policy_encoder
        else:
            self.policy_encoder = MPOEncoder(obs_dim, policy_layer_sizes)
            self.value_encoder = MPOEncoder(obs_dim, value_layer_sizes)

        self.policy_mean = nn.Linear(policy_layer_sizes[-1], act_dim)
        self.policy_logstd = nn.Linear(policy_layer_sizes[-1], act_dim)

        self.value_head = PopArt(
            value_layer_sizes[-1],
            beta=popart_beta,
            eps=popart_eps,
            min_sigma=popart_min_sigma,
        )

        # 3. Action Scaling
        if action_low is None or action_high is None:
            # Assume normalized environment [-1, 1] if not provided
            action_low = -np.ones(act_dim, dtype=np.float32)
            action_high = np.ones(act_dim, dtype=np.float32)

        action_low_t = torch.tensor(action_low, dtype=torch.float32)
        action_high_t = torch.tensor(action_high, dtype=torch.float32)

        self.register_buffer("action_scale", (action_high_t - action_low_t) / 2.0)
        self.register_buffer("action_bias", (action_high_t + action_low_t) / 2.0)

        # 4. Initialization (Kaiming for policy heads, closer to 0 for logstd)
        nn.init.xavier_uniform_(self.policy_mean.weight)
        nn.init.zeros_(self.policy_mean.bias)
        nn.init.xavier_uniform_(self.policy_logstd.weight)
        # Initialize log_std to match roughly std=0.5 to 1.0 initially
        nn.init.constant_(self.policy_logstd.bias, -0.5)

    def get_policy_dist_params(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Internal helper to get raw distribution parameters."""
        h = self.policy_encoder(obs)
        mean = self.policy_mean(h)
        log_std = self.policy_logstd(h)
        log_std = torch.clamp(log_std, -20.0, 2.0)
        return mean, log_std

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_policy_dist_params(obs)

    def get_value(self, obs: torch.Tensor, normalized: bool = False) -> torch.Tensor:
        """Flexible value getter."""
        h = self.value_encoder(obs)
        v_hat = self.value_head(h)
        if normalized:
            return v_hat
        return self.value_head.denormalize(v_hat)

    def forward_all(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Efficient forward pass returning: mean, log_std, UNNORMALIZED value.
        Avoids re-running encoder if shared.
        """
        # Policy Path
        mean, log_std = self.get_policy_dist_params(obs)

        # Value Path
        if self.shared_encoder:
            # We must re-encode if we didn't save the hidden state,
            # or refactor to return hidden state.
            # In PyTorch, re-running the linear layers of the head
            # on the same cached 'h' is tricky without modifying signature.
            # Assuming standard use case, we re-run encoder for clean code
            # UNLESS we manually optimize.
            # For strict correctness with shared encoder:
            h_val = self.policy_encoder(
                obs
            )  # This is technically redundant in compute but cleaner in code
        else:
            h_val = self.value_encoder(obs)

        v_hat = self.value_head(h_val)
        v = self.value_head.denormalize(v_hat)

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
