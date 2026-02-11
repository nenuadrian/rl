from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class LayerNormMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        layer_sizes: Tuple[int, ...],
        activate_final: bool = False,
    ):
        super().__init__()
        if len(layer_sizes) < 1:
            raise ValueError("layer_sizes must have at least one layer.")

        layers: list[nn.Module] = []

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


class SmallInitLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, std: float = 0.01):
        super().__init__(in_features, out_features)
        nn.init.trunc_normal_(self.weight, std=std)
        nn.init.zeros_(self.bias)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        log_std_bounds: Tuple[float, float] = (-20.0, 2.0),
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
    ):
        super().__init__()

        self.net = LayerNormMLP(obs_dim, hidden_sizes, activate_final=True)
        self.mean_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_min, self.log_std_max = log_std_bounds

        if action_low is None or action_high is None:
            action_low = -np.ones(act_dim, dtype=np.float32)
            action_high = np.ones(act_dim, dtype=np.float32)
        if not (
            np.all(np.isfinite(action_low)) and np.all(np.isfinite(action_high))
        ):
            raise ValueError(
                "GaussianPolicy requires finite action bounds. "
                "Received non-finite action_low/action_high."
            )

        action_low_t = torch.tensor(action_low, dtype=torch.float32)
        action_high_t = torch.tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high_t - action_low_t) / 2.0)
        self.register_buffer("action_bias", (action_high_t + action_low_t) / 2.0)

        nn.init.kaiming_normal_(self.mean_layer.weight, a=0.0, mode="fan_in")
        nn.init.zeros_(self.mean_layer.bias)
        nn.init.zeros_(self.log_std_layer.weight)
        nn.init.zeros_(self.log_std_layer.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)

        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        log_std = torch.nan_to_num(log_std, nan=0.0, posinf=0.0, neginf=0.0)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def _distribution(self, obs: torch.Tensor) -> Normal:
        mean, log_std = self.forward(obs)
        return Normal(mean, log_std.exp())

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        normal = self._distribution(obs)
        x = normal.rsample()
        y = torch.tanh(x)

        action = y * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x)
        log_prob = log_prob - torch.log(1.0 - y.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob = log_prob - torch.log(self.action_scale).sum()

        return action, log_prob

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        normal = self._distribution(obs)
        y = (actions - self.action_bias) / self.action_scale
        y = torch.clamp(y, -0.999999, 0.999999)
        x = torch.atanh(y)

        log_prob = normal.log_prob(x)
        log_prob = log_prob - torch.log(1.0 - y.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob = log_prob - torch.log(self.action_scale).sum()

        return log_prob

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        normal = self._distribution(obs)
        return normal.entropy().sum(dim=-1, keepdim=True)

    def sample_action(
        self, obs: torch.Tensor, deterministic: bool = False, **kwargs
    ) -> torch.Tensor:
        mean, log_std = self.forward(obs)
        if deterministic:
            y = torch.tanh(mean)
        else:
            y = torch.tanh(Normal(mean, log_std.exp()).rsample())
        return y * self.action_scale + self.action_bias


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        self.encoder = LayerNormMLP(obs_dim, hidden_sizes, activate_final=True)
        self.value_head = SmallInitLinear(hidden_sizes[-1], 1, std=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.encoder(obs))


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device,
        policy_layer_sizes: Tuple[int, ...],
        critic_layer_sizes: Tuple[int, ...],
        config: PPOConfig | None = None,
    ):
        self.device = device
        self.config = config or PPOConfig()

        self.policy = GaussianPolicy(
            obs_dim,
            act_dim,
            hidden_sizes=policy_layer_sizes,
            action_low=action_low,
            action_high=action_high,
        ).to(device)

        self.value = ValueNetwork(obs_dim, critic_layer_sizes).to(device)

        self.policy_opt = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.policy_lr, eps=1e-5
        )
        self.value_opt = torch.optim.Adam(
            self.value.parameters(), lr=self.config.value_lr, eps=1e-5
        )

    def set_hparams(
        self,
        *,
        clip_ratio: float | None = None,
        policy_lr: float | None = None,
        value_lr: float | None = None,
        ent_coef: float | None = None,
        vf_coef: float | None = None,
        max_grad_norm: float | None = None,
        target_kl: float | None = None,
    ) -> None:
        if clip_ratio is not None:
            self.config.clip_ratio = float(clip_ratio)
        if ent_coef is not None:
            self.config.ent_coef = float(ent_coef)
        if vf_coef is not None:
            self.config.vf_coef = float(vf_coef)
        if max_grad_norm is not None:
            self.config.max_grad_norm = float(max_grad_norm)
        if target_kl is not None:
            self.config.target_kl = float(target_kl)

        if policy_lr is not None:
            self.config.policy_lr = float(policy_lr)
            for g in self.policy_opt.param_groups:
                g["lr"] = self.config.policy_lr

        if value_lr is not None:
            self.config.value_lr = float(value_lr)
            for g in self.value_opt.param_groups:
                g["lr"] = self.config.value_lr

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if deterministic:
                action = self.policy.sample_action(obs, deterministic=True)
                log_prob = self.policy.log_prob(obs, action)
            else:
                action, log_prob = self.policy.sample(obs)
            value = self.value(obs)
        return action, log_prob, value

    def update(self, batch: dict) -> dict:
        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        returns = batch["returns"]
        advantages = batch["advantages"]
        values_old = batch.get("values_old", None)

        log_probs = self.policy.log_prob(obs, actions)
        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)

        clipped_ratio = torch.clamp(
            ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
        )
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        entropy = self.policy.entropy(obs).mean()
        policy_loss = policy_loss - self.config.ent_coef * entropy

        values = self.value(obs)
        if values_old is None:
            value_loss = F.mse_loss(values, returns)
        else:
            values_clipped = values_old + torch.clamp(
                values - values_old,
                -self.config.clip_ratio,
                self.config.clip_ratio,
            )
            value_loss = torch.max(
                (values - returns).pow(2),
                (values_clipped - returns).pow(2),
            ).mean()

        value_loss = value_loss * self.config.vf_coef

        with torch.no_grad():
            _, log_std = self.policy.forward(obs)
            log_std_mean = log_std.mean()
            log_std_std = log_std.std(unbiased=False)
            ratio_mean = ratio.mean()
            ratio_std = ratio.std(unbiased=False)
            value_mae = (values - returns).abs().mean()
            old_approx_kl = (-log_ratio).mean()
            # k3 estimator from http://joschu.net/blog/kl-approx.html
            approx_kl = ((ratio - 1.0) - log_ratio).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.policy_opt.step()

        self.value_opt.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(), self.config.max_grad_norm)
        self.value_opt.step()

        return {
            "loss/policy": float(policy_loss.item()),
            "loss/value": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "old_approx_kl": float(old_approx_kl.item()),
            "approx_kl": float(approx_kl.item()),
            "policy/log_std_mean": float(log_std_mean.item()),
            "policy/log_std_std": float(log_std_std.item()),
            "policy/ratio_mean": float(ratio_mean.item()),
            "policy/ratio_std": float(ratio_std.item()),
            "value/mae": float(value_mae.item()),
        }
