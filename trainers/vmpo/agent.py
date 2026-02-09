from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.vmpo.gaussian_mlp_policy import GaussianMLPPolicy


@dataclass
class VMPOConfig:
    gamma: float
    policy_lr: float
    value_lr: float
    topk_fraction: float
    eta_init: float
    eta_lr: float
    epsilon_eta: float
    epsilon_mu: float
    epsilon_sigma: float
    alpha_lr: float
    max_grad_norm: float


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
        print(self.policy)

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

        # Learnable temperature (dual variable), kept positive via exp(log_temperature).
        eta_init_t = torch.tensor(self.config.eta_init, device=device)
        eta_init_t = torch.clamp(eta_init_t, min=1e-8)
        self.log_temperature = torch.nn.Parameter(torch.log(torch.expm1(eta_init_t)))

        self.eta_opt = torch.optim.Adam([self.log_temperature], lr=self.config.eta_lr)

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
            value_norm = self.policy.value_norm(obs_t.detach())
        return float(value_norm.item())

    def update(self, batch: dict) -> dict:
        obs = batch["obs"]
        actions = batch["actions"]
        returns_raw = batch["returns"]
        advantages = batch["advantages"].squeeze(-1)
        old_means = batch["old_means"]
        old_log_stds = batch["old_log_stds"]

        # >>> normalize advantages (stabilises E-step)
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False) + 1e-8
        adv_norm = (advantages - adv_mean) / adv_std

        # E-step: top-k selection
        # use adv_norm for selection and weighting
        k = max(1, int(self.config.topk_fraction * adv_norm.numel()))
        topk_vals, _ = torch.topk(adv_norm, k)
        threshold = topk_vals.min()
        mask_bool = adv_norm >= threshold

        A_norm = adv_norm[mask_bool].detach()
        K = A_norm.numel()

        # Dual descent on eta (optimize log_temperature)
        # compute eta and clamp to reasonable bounds
        temperature = F.softplus(self.log_temperature) + 1e-8
        logK = torch.log(torch.tensor(float(K), device=A_norm.device))
        dual_loss = temperature * self.config.epsilon_eta + temperature * (
            torch.logsumexp(A_norm / temperature, dim=0) - logK
        )

        self.eta_opt.zero_grad(set_to_none=True)
        dual_loss.backward()
        self.eta_opt.step()

        # Recompute weights with updated eta, only on top-k set
        with torch.no_grad():
            temperature = F.softplus(self.log_temperature) + 1e-8
            weights = torch.softmax(A_norm / temperature, dim=0)

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
        weighted_nll = -(weights.detach() * log_prob_sel).sum() / K

        # Policy loss with trust-region penalties (stop-grad on alphas)
        with torch.no_grad():
            alpha_mu_det = self.log_alpha_mu.exp()
            alpha_sigma_det = self.log_alpha_sigma.exp()
        policy_loss = (
            weighted_nll + alpha_mu_det * kl_mu_sel + alpha_sigma_det * kl_sigma_sel
        )

        value_norm = self.policy.value_norm(obs.detach())
        value_loss = F.mse_loss(value_norm, returns_raw)

        total_loss = policy_loss + value_loss

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

        with torch.no_grad():
            params_after = nn.utils.parameters_to_vector(
                self.policy.parameters()
            ).detach()
            param_delta = torch.norm(params_after - params_before).item()

        with torch.no_grad():
            # weight diagnostics
            w = weights.detach()
            ess = 1.0 / (w.pow(2).sum() + 1e-12)
            selected_frac = torch.tensor(
                float(K) / float(adv_norm.numel()), device=adv_norm.device
            )

            adv_std_over_temperature = float(
                (adv_norm.std(unbiased=False) / (temperature + 1e-12)).item()
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
            "train/param_delta": float(param_delta),
            "train/mean_abs_action": float(mean_abs_action),
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
