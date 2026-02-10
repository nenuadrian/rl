from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.vmpo.gaussian_mlp_policy import GaussianMLPPolicy


# =========================
# Configuration
# =========================
@dataclass
class VMPOConfig:
    gamma: float
    policy_lr: float
    value_lr: float
    topk_fraction: float
    temperature_init: float
    temperature_lr: float
    epsilon_eta: float
    epsilon_mu: float
    epsilon_sigma: float
    alpha_lr: float
    max_grad_norm: float


# =========================
# VMPO Agent (PopArt-correct)
# =========================
class VMPOAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: VMPOConfig,
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

        # Parameter groups
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
            ],
            eps=1e-5
        )

        # Temperature dual (eta)
        temperature_init_t = torch.tensor(self.config.temperature_init, device=device)
        temperature_init_t = torch.clamp(temperature_init_t, min=1e-8)
        self.log_temperature = nn.Parameter(torch.log(temperature_init_t))
        self.eta_opt = torch.optim.Adam([self.log_temperature], lr=self.config.temperature_lr, eps=1e-5)

        # KL duals (mean / std)
        self.log_alpha_mu = nn.Parameter(torch.zeros(1, device=device))
        self.log_alpha_sigma = nn.Parameter(torch.zeros(1, device=device))
        self.alpha_opt = torch.optim.Adam(
            [self.log_alpha_mu, self.log_alpha_sigma], lr=self.config.alpha_lr, eps=1e-5
        )

    # -------- Acting --------
    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        obs_np = np.asarray(obs)

        # Support single observation (1D) or batched observations (2D)
        if obs_np.ndim == 1:
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            squeeze_out = True
        else:
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
            squeeze_out = False

        with torch.no_grad():
            mean, log_std, value = self.policy.forward_all(obs_t)
            action_t = self.policy.sample_action(mean, log_std, deterministic)

        action_np = action_t.cpu().numpy()
        mean_np = mean.cpu().numpy()
        log_std_np = log_std.cpu().numpy()
        value_np = value.cpu().numpy().squeeze(-1)

        if squeeze_out:
            return (
                action_np.squeeze(0),
                float(value_np.item()),
                mean_np.squeeze(0),
                log_std_np.squeeze(0),
            )
        else:
            return action_np, value_np, mean_np, log_std_np

    def value(self, obs: np.ndarray) -> float:
        obs_np = np.asarray(obs)
        if obs_np.ndim == 1:
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            squeeze_out = True
        else:
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
            squeeze_out = False

        with torch.no_grad():
            v = self.policy.value(obs_t).squeeze(-1)

        v_np = v.cpu().numpy()
        if squeeze_out:
            return float(v_np.item())
        else:
            return v_np

    # -------- Learning --------
    def update(self, batch: dict) -> dict:
        obs = batch["obs"]
        actions = batch["actions"]
        returns_raw = batch["returns"].squeeze(-1)
        advantages = batch["advantages"].squeeze(-1)
        old_means = batch["old_means"]
        old_log_stds = batch["old_log_stds"]

        # --- PopArt statistics update (must happen before critic loss) ---
        with torch.no_grad():
            self.policy.value_head.update_stats(returns_raw)

        # ================================================================
        # E-step (raw advantages, NO normalisation)
        # ================================================================
        adv = advantages.detach()

        k = max(1, int(self.config.topk_fraction * adv.numel()))
        topk_vals, _ = torch.topk(adv, k)
        threshold = topk_vals.min()
        mask_bool = adv >= threshold

        A = adv[mask_bool]
        A = A - A.mean()  # centre only
        K = A.numel()

        # Dual update for temperature eta
        temperature = torch.exp(self.log_temperature) + 1e-8
        logK = torch.log(torch.tensor(float(K), device=A.device))
        dual_loss = temperature * self.config.epsilon_eta + temperature * (
            torch.logsumexp(A / temperature, dim=0) - logK
        )

        self.eta_opt.zero_grad(set_to_none=True)
        dual_loss.backward()
        self.eta_opt.step()

        # Importance weights
        with torch.no_grad():
            temperature = torch.exp(self.log_temperature) + 1e-8
            weights = torch.softmax(A / temperature, dim=0)

        # ================================================================
        # M-step
        # ================================================================
        mean, log_std = self.policy(obs)
        log_prob = self.policy.log_prob(mean, log_std, actions).squeeze(-1)

        # --- KL diagnostics ---
        old_std = old_log_stds.exp()
        new_std = log_std.exp()

        kl_mean = ((mean - old_means) ** 2 / (2.0 * old_std**2)).sum(dim=-1)
        kl_std = 0.5 * (
            (new_std / old_std) ** 2 - 1.0 - 2.0 * (log_std - old_log_stds)
        ).sum(dim=-1)

        # Selected samples
        mean_sel = mean[mask_bool]
        log_std_sel = log_std[mask_bool]
        old_mean_sel = old_means[mask_bool]
        old_log_std_sel = old_log_stds[mask_bool]

        old_std_sel = old_log_std_sel.exp()
        new_std_sel = log_std_sel.exp()

        # Decoupled KLs
        kl_mu_sel = (
            (0.5 * ((mean_sel - old_mean_sel) ** 2 / (old_std_sel**2 + 1e-8)))
            .sum(dim=-1)
            .mean()
        )

        kl_sigma_sel = (
            (
                (log_std_sel - old_log_std_sel)
                + (old_std_sel**2) / (2.0 * (new_std_sel**2 + 1e-8))
                - 0.5
            )
            .sum(dim=-1)
            .mean()
        )

        # Dual updates for alphas
        alpha_mu = self.log_alpha_mu.exp()
        alpha_sigma = self.log_alpha_sigma.exp()

        alpha_loss = alpha_mu * (
            self.config.epsilon_mu - kl_mu_sel.detach()
        ) + alpha_sigma * (self.config.epsilon_sigma - kl_sigma_sel.detach())

        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        # Policy loss
        log_prob_sel = log_prob[mask_bool]
        weighted_nll = -(weights.detach() * log_prob_sel).sum()

        with torch.no_grad():
            alpha_mu_det = self.log_alpha_mu.exp()
            alpha_sigma_det = self.log_alpha_sigma.exp()

        policy_loss = (
            weighted_nll + alpha_mu_det * kl_mu_sel + alpha_sigma_det * kl_sigma_sel
        )

        # ================================================================
        # Critic loss (PopArt-normalised)
        # ================================================================
        v_hat = self.policy.value_norm(obs).squeeze(-1)
        target_hat = (
            returns_raw - self.policy.value_head.mu
        ) / self.policy.value_head.sigma

        value_loss = F.mse_loss(v_hat, target_hat.detach())

        total_loss = policy_loss + value_loss

        # Optimisation step
        with torch.no_grad():
            params_before = nn.utils.parameters_to_vector(
                self.policy.parameters()
            ).detach()
        self.opt.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_grad_norm
        )
        self.opt.step()

        # ================================================================
        # Diagnostics
        # ================================================================
        with torch.no_grad():
            ess = 1.0 / (weights.pow(2).sum() + 1e-12)
            selected_frac = float(K) / float(advantages.numel())

            mean_eval, log_std_eval = self.policy(obs)
            action_eval = self.policy.sample_action(
                mean_eval, log_std_eval, deterministic=True
            )
            mean_abs_action = float(action_eval.abs().mean().item())

            v_pred = self.policy.value(obs).squeeze(-1)
            y = returns_raw
            var_y = y.var(unbiased=False)
            explained_var = 1.0 - (y - v_pred).var(unbiased=False) / (var_y + 1e-8)

            params_after = nn.utils.parameters_to_vector(
                self.policy.parameters()
            ).detach()
            param_delta = torch.norm(params_after - params_before).item()

            # weight diagnostics
            w = weights.detach()
            ess = 1.0 / (w.pow(2).sum() + 1e-12)
            selected_frac = torch.tensor(
                float(K) / float(advantages.numel()), device=advantages.device
            )

            adv_std_over_temperature = float(
                (advantages.std(unbiased=False) / (temperature + 1e-12)).item()
            )

            mean_eval, log_std_eval = self.policy(obs)
            action_eval = self.policy.sample_action(
                mean_eval, log_std_eval, deterministic=True
            )
            mean_abs_action = float(action_eval.abs().mean().item())
            entropy = 0.5 * (1.0 + torch.log(2 * torch.pi * torch.exp(log_std)**2)).sum(dim=-1).mean()

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
            "vmpo/temperature_raw": float(temperature.item()),
            "vmpo/adv_std_over_temperature": adv_std_over_temperature,
            "vmpo/selected_frac": float(selected_frac.item()),
            "vmpo/threshold": float(threshold.item()),
            "vmpo/ess": float(ess.item()),
            "train/entropy": float(entropy.item()),
            "train/param_delta": float(param_delta),
            "train/mean_abs_action": float(mean_abs_action),
            "adv/raw_mean": float(advantages.mean().item()),
            "adv/raw_std": float((advantages.std(unbiased=False) + 1e-8).item()),
            "returns/raw_mean": float(returns_raw.mean().item()),
            "returns/raw_std": float((returns_raw.std(unbiased=False) + 1e-8).item()),
            "value/explained_var": float(explained_var.item()),
            "grad/norm": float(
                grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            ),
            "popart/value_head_mu": float(self.policy.value_head.mu.item()),
            "popart/value_head_sigma": float(self.policy.value_head.sigma.item()),
            "popart/target_hat_mean": float(target_hat.mean().item()),
            "popart/target_hat_std": float(target_hat.std(unbiased=False).item()),
            "popart/v_hat_mean": float(v_hat.mean().item()),
            "popart/v_hat_std": float(v_hat.std(unbiased=False).item()),
        }
