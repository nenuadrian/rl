from __future__ import annotations

import math
from typing import Tuple, Dict, Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from minerva.trainers.vmpo.gaussian_mlp_policy import SquashedGaussianPolicy


class VMPOAgent:

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device,
        policy_layer_sizes: Tuple[int, ...] = (256, 256),
        value_layer_sizes: Tuple[int, ...] = (256, 256),
        normalize_advantages: bool = True,
        gamma: float = 0.99,
        advantage_estimator: Literal["returns", "dae", "gae"] = "returns",
        gae_lambda: float = 0.95,
        policy_lr: float = 5e-4,
        value_lr: float = 1e-3,
        topk_fraction: float = 0.5,
        temperature_init: float = 1.0,
        temperature_lr: float = 1e-4,
        epsilon_eta: float = 0.1,
        epsilon_mu: float = 0.01,
        epsilon_sigma: float = 0.01,
        alpha_lr: float = 1e-4,
        max_grad_norm: float = 10.0,
        optimizer_type: str = "adam",
        sgd_momentum: float = 0.9,
        shared_encoder: bool = False,
        ppo_like_backbone: bool = False,
        m_steps: int = 1,
    ):
        self.device = device
        self.normalize_advantages = bool(normalize_advantages)
        self.gamma = float(gamma)
        self.advantage_estimator = advantage_estimator
        self.gae_lambda = float(gae_lambda)
        self.policy_lr = float(policy_lr)
        self.value_lr = float(value_lr)
        self.topk_fraction = float(topk_fraction)
        self.temperature_init = float(temperature_init)
        self.temperature_lr = float(temperature_lr)
        self.epsilon_eta = float(epsilon_eta)
        self.epsilon_mu = float(epsilon_mu)
        self.epsilon_sigma = float(epsilon_sigma)
        self.alpha_lr = float(alpha_lr)
        self.max_grad_norm = float(max_grad_norm)
        self.optimizer_type = str(optimizer_type)
        self.sgd_momentum = float(sgd_momentum)
        self.m_steps = int(m_steps)

        self.policy = SquashedGaussianPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            policy_layer_sizes=policy_layer_sizes,
            value_layer_sizes=value_layer_sizes,
            action_low=action_low,
            action_high=action_high,
            shared_encoder=shared_encoder,
            ppo_like_backbone=ppo_like_backbone,
        ).to(device)

        value_params = list(self.policy.value_head.parameters())
        if not self.policy.shared_encoder:
            value_params = list(self.policy.value_encoder.parameters()) + value_params
        # LaTeX: \tilde{\lambda}_{\pi} = \frac{\lambda_{\pi}}{\sqrt{M}}
        policy_lr_eff = self.policy_lr
        # LaTeX: \tilde{\lambda}_{V} = \frac{\lambda_{V}}{\sqrt{M}}
        value_lr_eff = self.value_lr

        self.combined_opt = self._build_optimizer(
            [
                {
                    "params": self.policy.policy_encoder.parameters(),
                    "lr": policy_lr_eff,
                },
                {"params": self.policy.policy_mean.parameters(), "lr": policy_lr_eff},
                {"params": self.policy.policy_logstd_parameters(), "lr": policy_lr_eff},
                {"params": value_params, "lr": value_lr_eff},
            ]
        )
        self.opt = self.combined_opt

        # Lagrange Multipliers (Dual Variables)

        # Temperature (eta) for advantage weighting
        temperature_init_t = torch.tensor(
            self.temperature_init, dtype=torch.float32, device=device
        )
        temperature_init_t = torch.clamp(temperature_init_t, min=1e-8)
        self.log_temperature = nn.Parameter(torch.log(torch.expm1(temperature_init_t)))
        self.eta_opt = self._build_optimizer(
            [self.log_temperature], lr=self.temperature_lr
        )

        # KL Penalties (alpha) for trust region
        def inv_softplus(x):
            return np.log(np.expm1(x))

        self.log_alpha_mu = nn.Parameter(
            torch.tensor(inv_softplus(1.0), dtype=torch.float32, device=device)
        )
        self.log_alpha_sigma = nn.Parameter(
            torch.tensor(inv_softplus(1.0), dtype=torch.float32, device=device)
        )
        # LaTeX: \tilde{\lambda}_{\alpha} = \frac{\lambda_{\alpha}}{\sqrt{M}}
        effective_alpha_lr = self.alpha_lr / math.sqrt(self.m_steps)
        self.alpha_opt = self._build_optimizer(
            [self.log_alpha_mu, self.log_alpha_sigma], lr=effective_alpha_lr
        )

    def _build_optimizer(
        self, params, lr: float | None = None
    ) -> torch.optim.Optimizer:
        optimizer_type = self.optimizer_type.strip().lower()
        kwargs: dict[str, float] = {}
        if lr is not None:
            kwargs["lr"] = float(lr)

        if optimizer_type == "adam":
            kwargs["eps"] = 1e-5
            return torch.optim.Adam(params, **kwargs)
        if optimizer_type == "sgd":
            kwargs["momentum"] = float(self.sgd_momentum)
            return torch.optim.SGD(params, **kwargs)

    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        obs_np = np.asarray(obs)
        is_batch = obs_np.ndim > 1

        if not is_batch:
            obs_np = obs_np[None, ...]  # Add batch dim

        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            mean, log_std, value = self.policy.forward_all(obs_t)
            action_t, _ = self.policy.sample_action(mean, log_std, deterministic)

        action_np = action_t.cpu().numpy()
        value_np = value.cpu().numpy().squeeze(-1)
        mean_np = mean.cpu().numpy()
        log_std_np = log_std.cpu().numpy()

        if not is_batch:
            return (
                action_np[0],
                float(value_np.item()),
                mean_np[0],
                log_std_np[0],
            )
        return action_np, value_np, mean_np, log_std_np

    def value(self, obs: np.ndarray) -> float | np.ndarray:
        obs_np = np.asarray(obs)
        is_batch = obs_np.ndim > 1
        if not is_batch:
            obs_np = obs_np[None, ...]

        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            v = self.policy.get_value(obs_t).squeeze(-1)

        v_np = v.cpu().numpy()
        if not is_batch:
            return float(v_np.item())
        return v_np

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        old_means = batch["old_means"]
        old_log_stds = batch["old_log_stds"]

        returns_raw = batch["returns"].squeeze(-1)
        advantages = batch["advantages"].squeeze(-1)
        restarting_weights = batch.get("restarting_weights", None)
        importance_weights = batch.get("importance_weights", None)
        if restarting_weights is None:
            restarting_weights = torch.ones_like(advantages)
        else:
            restarting_weights = restarting_weights.squeeze(-1)
        if importance_weights is None:
            importance_weights = torch.ones_like(advantages)
        else:
            importance_weights = importance_weights.squeeze(-1)

        # Advantage normalization
        if self.normalize_advantages:
            # LaTeX: \hat{A}_t = \frac{A_t - \mu_A}{\sigma_A + \epsilon}
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        # Capture params before update for 'param_delta'
        with torch.no_grad():
            params_before = nn.utils.parameters_to_vector(
                self.policy.parameters()
            ).detach()

        # ================================================================
        # E-Step: Re-weighting advantages (DeepMind-style semantics)
        # ================================================================
        # LaTeX: \eta = \operatorname{softplus}(\tilde{\eta}) + \epsilon
        eta = F.softplus(self.log_temperature) + 1e-8
        # LaTeX: s_t = \frac{w_t^{restart}\hat{A}_t}{\eta}
        scaled_advantages = (restarting_weights * advantages) / eta

        # Numerical stability: subtract global max before exponentiation.
        max_scaled_advantage = scaled_advantages.max().detach()

        with torch.no_grad():
            if not 0.0 < self.topk_fraction <= 1.0:
                raise ValueError(
                    f"`topk_fraction` must be in (0, 1], got {self.topk_fraction}"
                )

            if self.topk_fraction < 1.0:
                # Exclude restarting states from top-k selection.
                valid_scaled_advantages = scaled_advantages.detach().clone()
                valid_scaled_advantages[restarting_weights <= 0.0] = -torch.inf
                # LaTeX: k = \lfloor \rho N \rfloor
                k = int(self.topk_fraction * valid_scaled_advantages.numel())
                if k <= 0:
                    raise ValueError(
                        "topk_fraction too low to select any scaled advantages."
                    )
                topk_vals, _ = torch.topk(valid_scaled_advantages, k)
                # LaTeX: \tau = \min(\operatorname{TopK}(s, k))
                threshold = topk_vals.min()
                # LaTeX: m_t = \mathbf{1}[s_t \ge \tau]
                topk_weights = (valid_scaled_advantages >= threshold).to(
                    restarting_weights.dtype
                )
                # LaTeX: \bar{w}_t = w_t^{restart} \cdot m_t
                topk_restarting_weights = restarting_weights * topk_weights
            else:
                # LaTeX: \tau = \min_t s_t
                threshold = scaled_advantages.detach().min()
                # LaTeX: \bar{w}_t = w_t^{restart}
                topk_restarting_weights = restarting_weights

            # Mask for selected samples (used for KL terms and diagnostics).
            mask_bool = topk_restarting_weights > 0.0
            if not bool(mask_bool.any()):
                # Fallback to avoid empty reductions on tiny rollouts.
                mask_bool = torch.ones_like(mask_bool, dtype=torch.bool)
                topk_restarting_weights = torch.ones_like(topk_restarting_weights)

        # Use stop-gradient semantics for importance weights.
        importance_weights_sg = importance_weights.detach()
        # LaTeX: \tilde{\psi}_t = \bar{w}_t \, w_t^{imp} \exp(s_t - s_{\max})
        unnormalized_weights = (
            topk_restarting_weights
            * importance_weights_sg
            * torch.exp(scaled_advantages - max_scaled_advantage)
        )
        # LaTeX: Z = \sum_t \tilde{\psi}_t
        sum_weights = unnormalized_weights.sum() + 1e-8
        # LaTeX: N_{sel} = \sum_t \bar{w}_t
        num_samples = topk_restarting_weights.sum() + 1e-8
        # LaTeX: \psi_t = \frac{\tilde{\psi}_t}{Z}
        weights = unnormalized_weights / sum_weights
        weights_detached = weights.detach()

        # LaTeX: \log \bar{\psi} = \log Z + s_{\max} - \log N_{sel}

        log_mean_weights = (
            torch.log(sum_weights) + max_scaled_advantage - torch.log(num_samples)
        )
        # LaTeX: \mathcal{L}_{\eta} = \eta \left(\epsilon_{\eta} + \log \bar{\psi}\right)
        dual_loss = eta * (self.epsilon_eta + log_mean_weights)

        self.eta_opt.zero_grad()
        dual_loss.backward()
        self.eta_opt.step()

        # Compute Final Weights for Policy
        with torch.no_grad():
            # LaTeX: \eta' = \operatorname{softplus}(\tilde{\eta}) + \epsilon
            eta_final = F.softplus(self.log_temperature) + 1e-8
            # Logging: effective sample size and selection stats.
            ess = 1.0 / (weights_detached.pow(2).sum() + 1e-12)
            selected_frac = float(mask_bool.float().mean().item())
            adv_std_over_temperature = (
                advantages.std(unbiased=False) / (eta_final + 1e-12)
            ).item()

        # ================================================================
        # M-Step: Policy & Value Update
        # ================================================================

        for _ in range(self.m_steps):  # Multiple epochs of optimization per batch
            # Run forward pass on ALL observations
            current_mean, current_log_std, v_pred_fw = self.policy.forward_all(obs)
            v_pred = v_pred_fw.squeeze(-1)

            # -- Policy Loss --
            log_prob = self.policy.log_prob(
                current_mean, current_log_std, actions
            ).squeeze(-1)
            # LaTeX: \mathcal{L}_{\pi}^{NLL} = -\sum_t \psi_t \log \pi_{\theta}(a_t|s_t)
            weighted_nll = -(weights_detached * log_prob).sum()

            # -- KL Divergence (Full Batch Diagnostics) --
            # We compute this for logging purposes to see global drift
            with torch.no_grad():
                old_std = old_log_stds.exp()
                new_std = current_log_std.exp()
                # LaTeX: D_{\mu}^{all} = \mathbb{E}\left[\sum_j \frac{(\mu_j-\mu_j^{old})^2}{2(\sigma_j^{old})^2}\right]
                kl_mean_all = (
                    ((current_mean - old_means) ** 2 / (2.0 * old_std**2 + 1e-8))
                    .sum(dim=-1)
                    .mean()
                )
                # LaTeX: D_{\sigma}^{all} = \mathbb{E}\left[\frac{1}{2}\sum_j\left(\frac{(\sigma_j^{old})^2}{\sigma_j^2} - 1 + 2(\log \sigma_j - \log \sigma_j^{old})\right)\right]
                kl_std_all = (
                    0.5
                    * (
                        (old_std**2) / (new_std**2 + 1e-8)
                        - 1.0
                        + 2.0 * (current_log_std - old_log_stds)
                    )
                    .sum(dim=-1)
                    .mean()
                )

            # -- KL Divergence (Selected Samples for Optimization) --
            mean_sel = current_mean[mask_bool]
            log_std_sel = current_log_std[mask_bool]
            old_mean_sel = old_means[mask_bool]
            old_log_std_sel = old_log_stds[mask_bool]

            old_std_sel = old_log_std_sel.exp()
            new_std_sel = log_std_sel.exp()

            # Decoupled KL
            # LaTeX: D_{\mu} = \mathbb{E}_{t \in \mathcal{S}}\left[\sum_j \frac{(\mu_j-\mu_j^{old})^2}{2(\sigma_j^{old})^2}\right]
            kl_mu_sel = (
                (0.5 * ((mean_sel - old_mean_sel) ** 2 / (old_std_sel**2 + 1e-8)))
                .sum(dim=-1)
                .mean()
            )
            # LaTeX: D_{\sigma} = \mathbb{E}_{t \in \mathcal{S}}\left[\frac{1}{2}\sum_j\left(\frac{(\sigma_j^{old})^2}{\sigma_j^2} - 1 + 2(\log \sigma_j - \log \sigma_j^{old})\right)\right]
            kl_sigma_sel = (
                (
                    0.5
                    * (
                        (old_std_sel**2) / (new_std_sel**2 + 1e-8)
                        - 1.0
                        + 2.0 * (log_std_sel - old_log_std_sel)
                    )
                )
                .sum(dim=-1)
                .mean()
            )

            # -- Alpha Optimization --
            # LaTeX: \alpha_{\mu} = \operatorname{softplus}(\tilde{\alpha}_{\mu}) + \epsilon
            alpha_mu = F.softplus(self.log_alpha_mu) + 1e-8
            # LaTeX: \alpha_{\sigma} = \operatorname{softplus}(\tilde{\alpha}_{\sigma}) + \epsilon
            alpha_sigma = F.softplus(self.log_alpha_sigma) + 1e-8

            # We minimize: alpha * (epsilon - KL)
            # LaTeX: \mathcal{L}_{\alpha} = \alpha_{\mu}(\epsilon_{\mu} - D_{\mu}) + \alpha_{\sigma}(\epsilon_{\sigma} - D_{\sigma})
            alpha_loss = alpha_mu * (
                self.epsilon_mu - kl_mu_sel.detach()
            ) + alpha_sigma * (self.epsilon_sigma - kl_sigma_sel.detach())

            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            # -- Final Policy Loss --
            with torch.no_grad():
                alpha_mu_det = F.softplus(self.log_alpha_mu).detach() + 1e-8
                alpha_sigma_det = F.softplus(self.log_alpha_sigma).detach() + 1e-8

            # LaTeX: \mathcal{L}_{\pi} = \mathcal{L}_{\pi}^{NLL} + \alpha_{\mu}D_{\mu} + \alpha_{\sigma}D_{\sigma}

            policy_loss = (
                weighted_nll
                + (alpha_mu_det * kl_mu_sel)
                + (alpha_sigma_det * kl_sigma_sel)
            )

            # -- Critic Loss --
            # LaTeX: \mathcal{L}_{V} = \frac{1}{2}\mathbb{E}\left[(V_{\phi}(s_t) - R_t)^2\right]
            value_loss = 0.5 * F.mse_loss(v_pred, returns_raw.detach())

            # LaTeX: \mathcal{L} = \mathcal{L}_{\pi} + \mathcal{L}_{V}

            total_loss = policy_loss + value_loss
            self.combined_opt.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy.policy_encoder.parameters())
                + list(self.policy.policy_mean.parameters())
                + self.policy.policy_logstd_parameters()
                + (
                    []
                    if self.policy.shared_encoder
                    else list(self.policy.value_encoder.parameters())
                )
                + list(self.policy.value_head.parameters()),
                self.max_grad_norm,
            )
            self.combined_opt.step()

        # ================================================================
        # Post-Update Diagnostics
        # ================================================================
        with torch.no_grad():
            # 1. Parameter Delta
            params_after = nn.utils.parameters_to_vector(
                self.policy.parameters()
            ).detach()
            param_delta = torch.norm(params_after - params_before).item()

            # 2. Explained Variance (Unnormalized)
            v_pred = self.policy.get_value(obs).squeeze(-1)
            y = returns_raw
            var_y = y.var(unbiased=False)
            # LaTeX: \operatorname{EV} = 1 - \frac{\operatorname{Var}[R - V]}{\operatorname{Var}[R] + \epsilon}
            explained_var = 1.0 - (y - v_pred).var(unbiased=False) / (var_y + 1e-8)

            # 3. Action Saturation
            mean_eval, log_std_eval = self.policy(obs)
            action_eval, _ = self.policy.sample_action(
                mean_eval, log_std_eval, deterministic=True
            )
            mean_abs_action = float(action_eval.abs().mean().item())

            # LaTeX: \mathcal{H} = \mathbb{E}\left[\sum_j \frac{1}{2}\left(1 + \log(2\pi\sigma_j^2)\right)\right]

            entropy = (
                (0.5 * (1 + torch.log(2 * torch.pi * new_std_sel**2))).sum(-1).mean()
            )

        return {
            "loss/total": float(total_loss.item()),
            "loss/value": float(value_loss.item()),
            "loss/policy": float(policy_loss.item()),
            "loss/policy_weighted_nll": float(weighted_nll.item()),
            "loss/policy_kl_mean_pen": float((alpha_mu_det * kl_mu_sel).item()),
            "loss/policy_kl_std_pen": float((alpha_sigma_det * kl_sigma_sel).item()),
            "loss/alpha": float(alpha_loss.item()),
            # KL Diagnostics
            "kl/mean": float(kl_mean_all.item()),
            "kl/std": float(kl_std_all.item()),
            "kl/mean_sel": float(kl_mu_sel.item()),
            "kl/std_sel": float(kl_sigma_sel.item()),
            # Dual Variables
            "vmpo/alpha_mu": float(alpha_mu_det.item()),
            "vmpo/alpha_sigma": float(alpha_sigma_det.item()),
            "vmpo/dual_loss": float(dual_loss.item()),
            "vmpo/epsilon_eta": float(self.epsilon_eta),
            "vmpo/temperature_raw": float(eta_final.item()),
            "vmpo/adv_std_over_temperature": float(adv_std_over_temperature),
            # Selection Stats
            "vmpo/selected_frac": float(selected_frac),
            "vmpo/threshold": float(threshold.item()),
            "vmpo/ess": float(ess.item()),
            "vmpo/restarting_frac": float(
                (restarting_weights <= 0.0).float().mean().item()
            ),
            "vmpo/importance_mean": float(importance_weights_sg.mean().item()),
            # Training Dynamics
            "train/entropy": float(entropy.item()),
            "train/param_delta": float(param_delta),
            "train/mean_abs_action": float(mean_abs_action),
            # Data Stats
            "adv/raw_mean": float(advantages.mean().item()),
            "adv/raw_std": float((advantages.std(unbiased=False) + 1e-8).item()),
            "returns/raw_mean": float(returns_raw.mean().item()),
            "returns/raw_std": float((returns_raw.std(unbiased=False) + 1e-8).item()),
            "value/explained_var": float(explained_var.item()),
            "value/pred_mean": float(v_pred.mean().item()),
            "value/pred_std": float(v_pred.std(unbiased=False).item()),
        }
