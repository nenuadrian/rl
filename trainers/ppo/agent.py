from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def _layer_init(
    layer: nn.Module, std: float = math.sqrt(2.0), bias_const: float = 0.0
) -> nn.Module:
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
    return layer


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        layer_sizes: Tuple[int, ...],
    ):
        super().__init__()
        if len(layer_sizes) < 1:
            raise ValueError("layer_sizes must have at least one layer.")

        layers: list[nn.Module] = []
        prev_dim = in_dim
        for hidden_dim in layer_sizes:
            layers.append(_layer_init(nn.Linear(prev_dim, hidden_dim)))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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

        self.net = MLP(obs_dim, hidden_sizes)
        self.mean_layer = _layer_init(nn.Linear(hidden_sizes[-1], act_dim), std=0.01)
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))
        self.log_std_min, self.log_std_max = log_std_bounds

        if action_low is None or action_high is None:
            action_low = -np.ones(act_dim, dtype=np.float32)
            action_high = np.ones(act_dim, dtype=np.float32)
        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer(
            "action_high", torch.tensor(action_high, dtype=torch.float32)
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mean = self.mean_layer(h)
        log_std = self.log_std.expand_as(mean)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        log_std = torch.nan_to_num(log_std, nan=0.0, posinf=0.0, neginf=0.0)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def _distribution(self, obs: torch.Tensor) -> Normal:
        mean, log_std = self.forward(obs)
        return Normal(mean, log_std.exp())

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        normal = self._distribution(obs)
        action = normal.sample()
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        normal = self._distribution(obs)
        return normal.log_prob(actions).sum(dim=-1, keepdim=True)

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        normal = self._distribution(obs)
        return normal.entropy().sum(dim=-1, keepdim=True)

    def sample_action(
        self, obs: torch.Tensor, deterministic: bool = False, **kwargs
    ) -> torch.Tensor:
        mean, log_std = self.forward(obs)
        if deterministic:
            return mean
        else:
            return Normal(mean, log_std.exp()).sample()


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        self.encoder = MLP(obs_dim, hidden_sizes)
        self.value_head = _layer_init(nn.Linear(hidden_sizes[-1], 1), std=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.encoder(obs))


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
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        policy_lr: float = 3e-4,
        value_lr: float = 1e-3,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.02,
        norm_adv: bool = True,
        clip_vloss: bool = True,
        anneal_lr: bool = True,
        optimizer_type: str = "adam",
        sgd_momentum: float = 0.9,
    ):
        self.device = device
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.policy_lr = float(policy_lr)
        self.value_lr = float(value_lr)
        self.ent_coef = float(ent_coef)
        self.vf_coef = float(vf_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.target_kl = float(target_kl)
        self.norm_adv = bool(norm_adv)
        self.clip_vloss = bool(clip_vloss)
        self.anneal_lr = bool(anneal_lr)
        self.optimizer_type = str(optimizer_type)
        self.sgd_momentum = float(sgd_momentum)

        self.policy = GaussianPolicy(
            obs_dim,
            act_dim,
            hidden_sizes=policy_layer_sizes,
            action_low=action_low,
            action_high=action_high,
        ).to(device)

        self.value = ValueNetwork(obs_dim, critic_layer_sizes).to(device)

        self.policy_opt = self._build_optimizer(
            self.policy.parameters(), lr=self.policy_lr
        )
        self.value_opt = self._build_optimizer(
            self.value.parameters(), lr=self.value_lr
        )

    def _build_optimizer(
        self, params, *, lr: float
    ) -> torch.optim.Optimizer:
        optimizer_type = self.optimizer_type.strip().lower()
        if optimizer_type == "adam":
            return torch.optim.Adam(params, lr=float(lr), eps=1e-5)
        if optimizer_type == "sgd":
            return torch.optim.SGD(
                params,
                lr=float(lr),
                momentum=float(self.sgd_momentum),
            )
        raise ValueError(
            f"Unsupported PPO optimizer_type '{self.optimizer_type}'. "
            "Expected one of: adam, sgd."
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
        norm_adv: bool | None = None,
        clip_vloss: bool | None = None,
        anneal_lr: bool | None = None,
    ) -> None:
        if clip_ratio is not None:
            self.clip_ratio = float(clip_ratio)
        if ent_coef is not None:
            self.ent_coef = float(ent_coef)
        if vf_coef is not None:
            self.vf_coef = float(vf_coef)
        if max_grad_norm is not None:
            self.max_grad_norm = float(max_grad_norm)
        if target_kl is not None:
            self.target_kl = float(target_kl)
        if norm_adv is not None:
            self.norm_adv = bool(norm_adv)
        if clip_vloss is not None:
            self.clip_vloss = bool(clip_vloss)
        if anneal_lr is not None:
            self.anneal_lr = bool(anneal_lr)

        if policy_lr is not None:
            self.policy_lr = float(policy_lr)
            for g in self.policy_opt.param_groups:
                g["lr"] = self.policy_lr

        if value_lr is not None:
            self.value_lr = float(value_lr)
            for g in self.value_opt.param_groups:
                g["lr"] = self.value_lr

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

        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

        log_probs = self.policy.log_prob(obs, actions)
        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
        )
        policy_loss = torch.max(pg_loss1, pg_loss2).mean()

        entropy = self.policy.entropy(obs).mean()

        new_values = self.value(obs)
        if self.clip_vloss and values_old is not None:
            value_loss_unclipped = (new_values - returns).pow(2)
            values_clipped = values_old + torch.clamp(
                new_values - values_old,
                -self.clip_ratio,
                self.clip_ratio,
            )
            value_loss_clipped = (values_clipped - returns).pow(2)
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
        else:
            value_loss = 0.5 * F.mse_loss(new_values, returns)

        total_loss = (
            policy_loss
            - self.ent_coef * entropy
            + self.vf_coef * value_loss
        )

        with torch.no_grad():
            _, log_std = self.policy.forward(obs)
            log_std_mean = log_std.mean()
            log_std_std = log_std.std(unbiased=False)
            ratio_mean = ratio.mean()
            ratio_std = ratio.std(unbiased=False)
            value_mae = (new_values - returns).abs().mean()
            old_approx_kl = (-log_ratio).mean()
            # k3 estimator from http://joschu.net/blog/kl-approx.html
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_ratio).float().mean()

        self.policy_opt.zero_grad()
        self.value_opt.zero_grad()
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.value.parameters()),
            self.max_grad_norm,
        )
        self.policy_opt.step()
        self.value_opt.step()

        return {
            "loss/policy": float(policy_loss.item()),
            "loss/value": float(value_loss.item()),
            "loss/total": float(total_loss.item()),
            "entropy": float(entropy.item()),
            "old_approx_kl": float(old_approx_kl.item()),
            "approx_kl": float(approx_kl.item()),
            "clipfrac": float(clipfrac.item()),
            "grad_norm": float(grad_norm.item()),
            "policy/log_std_mean": float(log_std_mean.item()),
            "policy/log_std_std": float(log_std_std.item()),
            "policy/ratio_mean": float(ratio_mean.item()),
            "policy/ratio_std": float(ratio_std.item()),
            "value/mae": float(value_mae.item()),
        }
