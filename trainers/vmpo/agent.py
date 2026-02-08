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


@dataclass
class VMPOConfig:
    gamma: float = 0.99
    policy_lr: float = 1e-4
    value_lr: float = 1e-4
    topk_fraction: float = 0.8
    eta: float = 5.0
    eta_lr: float = 1e-3
    epsilon_eta: float = 0.1
    epsilon_mu: float = 0.01
    epsilon_sigma: float = 1e-5
    alpha_lr: float = 1e-3
    kl_mean_coef: float = 1e-3
    kl_std_coef: float = 1e-3
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

        self.policy = GaussianMLPPolicy(
            obs_dim,
            act_dim,
            hidden_sizes=hidden_sizes,
            action_low=action_low,
            action_high=action_high,
        ).to(device)

        shared_params = list(self.policy.encoder.parameters())
        policy_params = list(self.policy.policy_mean.parameters()) + list(
            self.policy.policy_logstd.parameters()
        )
        value_params = list(self.policy.value_head.parameters())

        self.opt = torch.optim.Adam(
            [
                {"params": shared_params, "lr": self.config.policy_lr},
                {"params": policy_params, "lr": self.config.policy_lr},
                {"params": value_params, "lr": self.config.value_lr},
            ]
        )

        # Learnable temperature (dual variable), kept positive via exp(log_eta).
        self.log_eta = torch.nn.Parameter(
            torch.log(torch.tensor(float(self.config.eta), device=device))
        )
        self.eta_opt = torch.optim.Adam([self.log_eta], lr=self.config.eta_lr)

        # Learnable KL multipliers (dual variables), kept positive via exp(log_alpha_*).
        self.log_alpha_mu = torch.nn.Parameter(torch.zeros(1, device=device))
        self.log_alpha_sigma = torch.nn.Parameter(torch.zeros(1, device=device))
        self.alpha_opt = torch.optim.Adam(
            [self.log_alpha_mu, self.log_alpha_sigma], lr=self.config.alpha_lr
        )

    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            mean, log_std, value_norm = self.policy.forward_all(obs_t)
            action_t = self.policy.sample_action(mean, log_std, deterministic)
            value = value_norm

        return (
            action_t.detach().cpu().numpy().squeeze(0),
            float(value.item()),
            mean.detach().cpu().numpy().squeeze(0),
            log_std.detach().cpu().numpy().squeeze(0),
        )

    def value(self, obs: np.ndarray) -> float:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            value_norm = self.policy.value_norm(obs_t)
        return float(value_norm.item())

    def update(self, batch: dict) -> dict:
        obs = batch["obs"]
        actions = batch["actions"]
        returns_raw = batch["returns"]
        advantages = batch["advantages"].squeeze(-1)
        old_means = batch["old_means"]
        old_log_stds = batch["old_log_stds"]

        # E-step: top-k selection
        k = max(1, int(self.config.topk_fraction * advantages.numel()))
        topk_vals, _ = torch.topk(advantages, k)
        threshold = topk_vals.min()
        mask_bool = advantages >= threshold

        A = advantages[mask_bool].detach()
        K = A.numel()

        # Dual descent on eta (optimize log_eta)
        eta = self.log_eta.exp()
        logK = torch.log(torch.tensor(float(K), device=A.device))
        dual_loss = eta * self.config.epsilon_eta + eta * (
            torch.logsumexp(A / eta, dim=0) - logK
        )

        self.eta_opt.zero_grad(set_to_none=True)
        dual_loss.backward()
        self.eta_opt.step()

        # Recompute weights with updated eta, only on top-k set
        with torch.no_grad():
            eta = self.log_eta.exp()
            weights = torch.softmax(A / eta, dim=0)

        mean, log_std = self.policy(obs)
        log_prob = self.policy.log_prob(mean, log_std, actions).squeeze(-1)

        # KL diagnostics (not used in loss once eta is learned)
        old_std = old_log_stds.exp()
        new_std = log_std.exp()
        kl_mean = ((mean - old_means) ** 2 / (2.0 * (old_std**2))).sum(dim=-1)
        kl_std = 0.5 * (
            (new_std / old_std) ** 2 - 1.0 - 2.0 * (log_std - old_log_stds)
        ).sum(dim=-1)

        # --- Trust-region (decoupled) KL constraints on selected samples ---
        mean_sel = mean[mask_bool]
        log_std_sel = log_std[mask_bool]
        old_mean_sel = old_means[mask_bool]
        old_log_std_sel = old_log_stds[mask_bool]

        old_std_sel = old_log_std_sel.exp()
        new_std_sel = log_std_sel.exp()

        # KL_mu: KL(N(old_mu, old_std) || N(new_mu, old_std))  (std fixed to old)
        kl_mu_sel = (
            (0.5 * ((mean_sel - old_mean_sel).pow(2) / (old_std_sel.pow(2) + 1e-8)))
            .sum(dim=-1)
            .mean()
        )

        # KL_sigma: KL(N(old_mu, old_std) || N(old_mu, new_std)) (mean fixed to old)
        kl_sigma_sel = (
            (
                (log_std_sel - old_log_std_sel)
                + (old_std_sel.pow(2) / (2.0 * (new_std_sel.pow(2) + 1e-8)))
                - 0.5
            )
            .sum(dim=-1)
            .mean()
        )

        alpha_mu = self.log_alpha_mu.exp()
        alpha_sigma = self.log_alpha_sigma.exp()

        # Dual update for alphas: minimize alpha * (epsilon - KL_detached)
        alpha_loss = alpha_mu * (
            self.config.epsilon_mu - kl_mu_sel.detach()
        ) + alpha_sigma * (self.config.epsilon_sigma - kl_sigma_sel.detach())
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        # M-step: weighted negative log-likelihood over selected samples only
        log_prob_sel = log_prob[mask_bool]
        weighted_nll = -(weights.detach() * log_prob_sel).sum()

        # Policy loss with trust-region penalties (stop-grad on alphas)
        with torch.no_grad():
            alpha_mu_det = self.log_alpha_mu.exp()
            alpha_sigma_det = self.log_alpha_sigma.exp()
        policy_loss = (
            weighted_nll + alpha_mu_det * kl_mu_sel + alpha_sigma_det * kl_sigma_sel
        )

        value_norm = self.policy.value_norm(obs)
        value_loss = F.mse_loss(value_norm, returns_raw)

        total_loss = policy_loss + value_loss

        self.opt.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_grad_norm
        )
        self.opt.step()

        with torch.no_grad():
            # weight diagnostics
            w = weights.detach()
            ess = 1.0 / (w.pow(2).sum() + 1e-12)
            selected_frac = torch.tensor(
                float(K) / float(advantages.numel()), device=advantages.device
            )

            v_pred_raw = value_norm
            y = returns_raw.squeeze(-1)
            yhat = v_pred_raw.squeeze(-1)
            var_y = y.var(unbiased=False)
            explained_var = 1.0 - (y - yhat).var(unbiased=False) / (var_y + 1e-8)

        return {
            "loss/total": float(total_loss.item()),
            "loss/policy": float(policy_loss.item()),
            "loss/policy_weighted_nll": float(weighted_nll.item()),
            "loss/policy_kl_mean_pen": float((alpha_mu_det * kl_mu_sel).item()),
            "loss/policy_kl_std_pen": float((alpha_sigma_det * kl_sigma_sel).item()),
            "loss/alpha": float(alpha_loss.item()),
            "kl/mean": float(kl_mean.mean().item()),
            "kl/std": float(kl_std.mean().item()),
            "kl/mean_sel": float(kl_mu_sel.item()),
            "kl/std_sel": float(kl_sigma_sel.item()),
            "vmpo/alpha_mu": float(alpha_mu_det.item()),
            "vmpo/alpha_sigma": float(alpha_sigma_det.item()),
            "vmpo/dual_loss": float(dual_loss.item()),
            "vmpo/epsilon_eta": float(self.config.epsilon_eta),
            "vmpo/eta": float(eta.item()),
            "vmpo/topk_k": int(k),
            "vmpo/selected_frac": float(selected_frac.item()),
            "vmpo/threshold": float(threshold.item()),
            "vmpo/ess": float(ess.item()),
            "adv/raw_mean": float(advantages.mean().item()),
            "adv/raw_std": float((advantages.std(unbiased=False) + 1e-8).item()),
            "returns/raw_mean": float(returns_raw.mean().item()),
            "returns/raw_std": float((returns_raw.std(unbiased=False) + 1e-8).item()),
            "value/raw_mean": float(v_pred_raw.mean().item()),
            "value/raw_std": float((v_pred_raw.std(unbiased=False) + 1e-8).item()),
            "value/explained_var": float(explained_var.item()),
            "grad/norm": float(
                grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            ),
        }
