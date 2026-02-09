from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.vmpo.gaussian_mlp_policy import GaussianMLPPolicy


@dataclass
class VMPOLightConfig:
    gamma: float
    policy_lr: float
    value_lr: float
    temperature_init: float
    temperature_lr: float
    epsilon_eta: float


class VMPOLightAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: VMPOLightConfig,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device,
        policy_layer_sizes: Tuple[int, ...],
    ):
        self.device = device
        self.config = config

        self.policy = GaussianMLPPolicy(
            obs_dim,
            act_dim,
            hidden_sizes=policy_layer_sizes,
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
            torch.log(torch.tensor(float(self.config.temperature_init), device=device))
        )
        self.eta_opt = torch.optim.Adam([self.log_eta], lr=self.config.temperature_lr)

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

        # >>> normalize advantages (stabilises E-step)
        adv = advantages.squeeze(-1)  # shape (N,)
        adv_mean = adv.mean()
        adv_std = adv.std(unbiased=False) + 1e-8
        adv_norm = (adv - adv_mean) / adv_std

        # E-step (no top-k): use full batch
        A = adv_norm.detach()
        K = A.numel()

        # Dual descent on eta (optimize log_eta)
        eta = self.log_eta.exp()
        eta_clamped = torch.clamp(eta, min=1e-6, max=1e3)
        logK = torch.log(torch.tensor(float(K), device=A.device))
        dual_loss = eta_clamped * self.config.epsilon_eta + eta_clamped * (
            torch.logsumexp(A / eta_clamped, dim=0) - logK
        )

        self.eta_opt.zero_grad(set_to_none=True)
        dual_loss.backward()
        self.eta_opt.step()

        # Recompute weights with updated eta, full batch
        with torch.no_grad():
            eta = self.log_eta.exp()
            eta_clamped = torch.clamp(eta, min=1e-6, max=1e3)
            weights = torch.softmax(A / eta_clamped, dim=0)

        mean, log_std = self.policy(obs)
        log_prob = self.policy.log_prob(mean, log_std, actions).squeeze(-1)

        # KL diagnostics (per-sample) retained for monitoring only
        old_std = old_log_stds.exp()
        new_std = log_std.exp()
        kl_mean = ((mean - old_means) ** 2 / (2.0 * (old_std**2))).sum(dim=-1)
        kl_std = 0.5 * (
            (new_std / old_std) ** 2 - 1.0 - 2.0 * (log_std - old_log_stds)
        ).sum(dim=-1)

        # M-step: weighted negative log-likelihood over full batch
        weighted_nll = -(weights.detach() * log_prob).sum()

        policy_loss = weighted_nll

        value_norm = self.policy.value_norm(obs)
        value_loss = F.mse_loss(value_norm, returns_raw)

        total_loss = policy_loss + value_loss

        with torch.no_grad():
            params_before = nn.utils.parameters_to_vector(
                self.policy.parameters()
            ).detach()

        self.opt.zero_grad(set_to_none=True)
        total_loss.backward()
        self.opt.step()

        with torch.no_grad():
            params_after = nn.utils.parameters_to_vector(
                self.policy.parameters()
            ).detach()
            param_delta = torch.norm(params_after - params_before).item()

        with torch.no_grad():
            w = weights.detach()
            ess = 1.0 / (w.pow(2).sum() + 1e-12)
            selected_frac = torch.tensor(1.0, device=advantages.device)

            adv_std_over_eta = float(
                (advantages.std(unbiased=False) / (eta_clamped + 1e-12)).item()
            )

            mean_eval, log_std_eval = self.policy(obs)
            action_eval = self.policy.sample_action(
                mean_eval, log_std_eval, deterministic=True
            )
            mean_abs_action = float(action_eval.abs().mean().item())

            v_pred_raw = value_norm
            y = returns_raw.squeeze(-1)
            yhat = v_pred_raw.squeeze(-1)
            var_y = y.var(unbiased=False)
            explained_var = 1.0 - (y - yhat).var(unbiased=False) / (var_y + 1e-8)

        return {
            "loss/total": float(total_loss.item()),
            "loss/policy": float(policy_loss.item()),
            "loss/policy_weighted_nll": float(weighted_nll.item()),
            "kl/mean": float(kl_mean.mean().item()),
            "kl/std": float(kl_std.mean().item()),
            "vmpo/dual_loss": float(dual_loss.item()),
            "vmpo/epsilon_eta": float(self.config.epsilon_eta),
            "vmpo/eta": float(eta_clamped.item()),
            "vmpo/eta_raw": float(eta.item()),
            "vmpo/adv_std_over_eta": adv_std_over_eta,
            "vmpo/selected_frac": float(selected_frac.item()),
            "vmpo/ess": float(ess.item()),
            "train/param_delta": float(param_delta),
            "train/mean_abs_action": float(mean_abs_action),
            "adv/raw_mean": float(advantages.mean().item()),
            "adv/raw_std": float((advantages.std(unbiased=False) + 1e-8).item()),
            "returns/raw_mean": float(returns_raw.mean().item()),
            "returns/raw_std": float((returns_raw.std(unbiased=False) + 1e-8).item()),
            "value/raw_mean": float(v_pred_raw.mean().item()),
            "value/raw_std": float((v_pred_raw.std(unbiased=False) + 1e-8).item()),
            "value/explained_var": float(explained_var.item()),
        }
