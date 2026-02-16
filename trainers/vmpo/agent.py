from __future__ import annotations

from typing import Tuple, Dict, Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.vmpo.gaussian_mlp_policy import SquashedGaussianPolicy


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

        self.policy = SquashedGaussianPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            policy_layer_sizes=policy_layer_sizes,
            value_layer_sizes=value_layer_sizes,
            action_low=action_low,
            action_high=action_high,
            shared_encoder=False,  # Explicitly using separate encoders
        ).to(device)

        # Policy optimizer (actor only).
        self.policy_opt = self._build_optimizer(
            [
                {
                    "params": self.policy.policy_encoder.parameters(),
                    "lr": self.policy_lr,
                },
                {
                    "params": self.policy.policy_mean.parameters(),
                    "lr": self.policy_lr,
                },
                {
                    "params": self.policy.policy_logstd.parameters(),
                    "lr": self.policy_lr,
                },
            ],
        )

        # Value optimizer (critic only).
        self.value_opt = self._build_optimizer(
            [
                {
                    "params": self.policy.value_encoder.parameters(),
                    "lr": self.value_lr,
                },
                {
                    "params": self.policy.value_head.parameters(),
                    "lr": self.value_lr,
                },
            ],
        )
        # Cache explicit parameter lists for per-network gradient clipping.
        self._policy_params = [
            *self.policy.policy_encoder.parameters(),
            *self.policy.policy_mean.parameters(),
            *self.policy.policy_logstd.parameters(),
        ]
        self._value_params = [
            *self.policy.value_encoder.parameters(),
            *self.policy.value_head.parameters(),
        ]

        # Lagrange Multipliers (Dual Variables)

        # Temperature (eta) for advantage weighting
        temperature_init_t = torch.tensor(
            self.temperature_init, dtype=torch.float32, device=device
        )
        temperature_init_t = torch.clamp(temperature_init_t, min=1e-8)
        self.log_temperature = nn.Parameter(
            torch.log(torch.expm1(temperature_init_t))
        )
        self.eta_opt = self._build_optimizer(
            [self.log_temperature], lr=self.temperature_lr
        )

        # KL Penalties (alpha) for trust region
        # Initializing to 0.1 (log space) for a stronger initial penalty than 0
        self.log_alpha_mu = nn.Parameter(
            torch.tensor(np.log(1.0), dtype=torch.float32, device=device)
        )
        self.log_alpha_sigma = nn.Parameter(
            torch.tensor(np.log(1.0), dtype=torch.float32, device=device)
        )

        self.alpha_opt = self._build_optimizer(
            [self.log_alpha_mu, self.log_alpha_sigma], lr=self.alpha_lr
        )

    def _build_optimizer(self, params, lr: float | None = None) -> torch.optim.Optimizer:
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
            # Efficient single pass
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
        eta = F.softplus(self.log_temperature) + 1e-8
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
                k = int(self.topk_fraction * valid_scaled_advantages.numel())
                if k <= 0:
                    raise ValueError(
                        "topk_fraction too low to select any scaled advantages."
                    )
                topk_vals, _ = torch.topk(valid_scaled_advantages, k)
                threshold = topk_vals.min()
                topk_weights = (valid_scaled_advantages >= threshold).to(
                    restarting_weights.dtype
                )
                topk_restarting_weights = restarting_weights * topk_weights
            else:
                threshold = scaled_advantages.detach().min()
                topk_restarting_weights = restarting_weights

            # Mask for selected samples (used for KL terms and diagnostics).
            mask_bool = topk_restarting_weights > 0.0
            if not bool(mask_bool.any()):
                # Fallback to avoid empty reductions on tiny rollouts.
                mask_bool = torch.ones_like(mask_bool, dtype=torch.bool)
                topk_restarting_weights = torch.ones_like(topk_restarting_weights)

        # Use stop-gradient semantics for importance weights.
        importance_weights_sg = importance_weights.detach()
        unnormalized_weights = (
            topk_restarting_weights
            * importance_weights_sg
            * torch.exp(scaled_advantages - max_scaled_advantage)
        )
        sum_weights = unnormalized_weights.sum() + 1e-8
        num_samples = topk_restarting_weights.sum() + 1e-8
        weights = unnormalized_weights / sum_weights
        weights_detached = weights.detach()

        log_mean_weights = (
            torch.log(sum_weights)
            + max_scaled_advantage
            - torch.log(num_samples)
        )
        dual_loss = eta * (self.epsilon_eta + log_mean_weights)

        self.eta_opt.zero_grad()
        dual_loss.backward()
        self.eta_opt.step()

        # Compute Final Weights for Policy
        with torch.no_grad():
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

        # Run forward pass on ALL observations
        current_mean, current_log_std = self.policy(obs)

        # -- Policy Loss --
        log_prob = self.policy.log_prob(current_mean, current_log_std, actions).squeeze(
            -1
        )
        weighted_nll = -(weights_detached * log_prob).sum()

        # -- KL Divergence (Full Batch Diagnostics) --
        # We compute this for logging purposes to see global drift
        with torch.no_grad():
            old_std = old_log_stds.exp()
            new_std = current_log_std.exp()
            kl_mean_all = (
                ((current_mean - old_means) ** 2 / (2.0 * old_std**2 + 1e-8))
                .sum(dim=-1)
                .mean()
            )
            kl_std_all = (
                (
                    (current_log_std - old_log_stds)
                    + (old_std**2) / (2.0 * (new_std**2 + 1e-8))
                    - 0.5
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

        # -- Alpha Optimization --
        alpha_mu = F.softplus(self.log_alpha_mu) + 1e-8
        alpha_sigma = F.softplus(self.log_alpha_sigma) + 1e-8

        # We minimize: alpha * (epsilon - KL)
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

        policy_loss = (
            weighted_nll + (alpha_mu_det * kl_mu_sel) + (alpha_sigma_det * kl_sigma_sel)
        )

        # -- Critic Loss --
        v_pred = self.policy.get_value(obs).squeeze(-1)
        value_loss = 0.5 * F.mse_loss(v_pred, returns_raw.detach())

        total_loss = policy_loss + value_loss

        # -- Update Actor Weights --
        self.policy_opt.zero_grad()
        policy_loss.backward()
        policy_grad_norm = nn.utils.clip_grad_norm_(
            self._policy_params, self.max_grad_norm
        )
        self.policy_opt.step()

        # -- Update Critic Weights --
        self.value_opt.zero_grad()
        value_loss.backward()
        value_grad_norm = nn.utils.clip_grad_norm_(
            self._value_params, self.max_grad_norm
        )
        self.value_opt.step()

        policy_grad_norm_f = float(
            policy_grad_norm.item()
            if torch.is_tensor(policy_grad_norm)
            else policy_grad_norm
        )
        value_grad_norm_f = float(
            value_grad_norm.item()
            if torch.is_tensor(value_grad_norm)
            else value_grad_norm
        )

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
            explained_var = 1.0 - (y - v_pred).var(unbiased=False) / (var_y + 1e-8)

            # 3. Action Saturation
            mean_eval, log_std_eval = self.policy(obs)
            action_eval, _ = self.policy.sample_action(
                mean_eval, log_std_eval, deterministic=True
            )
            mean_abs_action = float(action_eval.abs().mean().item())

            entropy = (
                (0.5 * (1 + torch.log(2 * torch.pi * new_std_sel**2))).sum(-1).mean()
            )

        return {
            "loss/total": float(total_loss.item()),
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
            "grad/norm": policy_grad_norm_f,
            "grad/norm_value": value_grad_norm_f,
            # Data Stats
            "adv/raw_mean": float(advantages.mean().item()),
            "adv/raw_std": float((advantages.std(unbiased=False) + 1e-8).item()),
            "returns/raw_mean": float(returns_raw.mean().item()),
            "returns/raw_std": float((returns_raw.std(unbiased=False) + 1e-8).item()),
            "value/explained_var": float(explained_var.item()),
            "value/pred_mean": float(v_pred.mean().item()),
            "value/pred_std": float(v_pred.std(unbiased=False).item()),
        }
