from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.vmpo.gaussian_mlp_policy import SquashedGaussianPolicy


@dataclass
class VMPOConfig:
    normalize_advantages: bool = True
    gamma: float = 0.99
    policy_lr: float = 5e-4
    value_lr: float = 1e-3
    topk_fraction: float = 0.5
    temperature_init: float = 1.0
    temperature_lr: float = 1e-4
    epsilon_eta: float = 0.1 
    epsilon_mu: float = 0.01 
    epsilon_sigma: float = 0.01  
    alpha_lr: float = 1e-4
    max_grad_norm: float = 10.0
    popart_beta: float = 3e-4
    popart_eps: float = 1e-4
    popart_min_sigma: float = 1e-4
    optimizer_type: str = "adam"
    sgd_momentum: float = 0.0


class VMPOAgent:

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: VMPOConfig,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device,
        policy_layer_sizes: Tuple[int, ...] = (256, 256),
        value_layer_sizes: Tuple[int, ...] = (256, 256),
    ):
        self.device = device
        self.config = config

        self.policy = SquashedGaussianPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            policy_layer_sizes=policy_layer_sizes,
            value_layer_sizes=value_layer_sizes,
            action_low=action_low,
            action_high=action_high,
            popart_beta=config.popart_beta,
            popart_eps=config.popart_eps,
            popart_min_sigma=config.popart_min_sigma,
            shared_encoder=False,  # Explicitly using separate encoders
        ).to(device)

        # We group parameters by network to apply specific LRs
        self.opt = self._build_optimizer(
            [
                {
                    "params": self.policy.policy_encoder.parameters(),
                    "lr": self.config.policy_lr,
                },
                {
                    "params": self.policy.policy_mean.parameters(),
                    "lr": self.config.policy_lr,
                },
                {
                    "params": self.policy.policy_logstd.parameters(),
                    "lr": self.config.policy_lr,
                },
                # Value function (Encoder + Head)
                {
                    "params": self.policy.value_encoder.parameters(),
                    "lr": self.config.value_lr,
                },
                {
                    "params": self.policy.value_head.parameters(),
                    "lr": self.config.value_lr,
                },
            ],
        )

        # Lagrange Multipliers (Dual Variables)

        # Temperature (eta) for advantage weighting
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(self.config.temperature_init, device=device))
        )
        self.eta_opt = self._build_optimizer(
            [self.log_temperature], lr=self.config.temperature_lr
        )

        # KL Penalties (alpha) for trust region
        # Initializing to 0.1 (log space) for a stronger initial penalty than 0
        self.log_alpha_mu = nn.Parameter(torch.tensor(np.log(1.0), device=device))
        self.log_alpha_sigma = nn.Parameter(torch.tensor(np.log(1.0), device=device))

        self.alpha_opt = self._build_optimizer(
            [self.log_alpha_mu, self.log_alpha_sigma], lr=self.config.alpha_lr
        )

    def _build_optimizer(self, params, lr: float | None = None) -> torch.optim.Optimizer:
        optimizer_type = self.config.optimizer_type.strip().lower()
        kwargs: dict[str, float] = {}
        if lr is not None:
            kwargs["lr"] = float(lr)

        if optimizer_type == "adam":
            kwargs["eps"] = 1e-5
            return torch.optim.Adam(params, **kwargs)
        if optimizer_type == "sgd":
            kwargs["momentum"] = float(self.config.sgd_momentum)
            return torch.optim.SGD(params, **kwargs)

        raise ValueError(
            f"Unsupported VMPO optimizer_type '{self.config.optimizer_type}'. "
            "Expected one of: adam, sgd."
        )

    def compute_gae(self, rewards, values, dones, gamma, lam):
        """
        Compute Generalized Advantage Estimation (GAE) and returns.
        """
        T = rewards.shape[0]
        advantages = torch.zeros(T, dtype=values.dtype, device=values.device)
        lastgaelam = 0
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * next_nonterminal - values[t]
            lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values[:-1]
        return advantages, returns

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
            v = self.policy.get_value(obs_t, normalized=False).squeeze(-1)

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

        # Advantage normalization
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        # Capture params before update for 'param_delta'
        with torch.no_grad():
            params_before = nn.utils.parameters_to_vector(
                self.policy.parameters()
            ).detach()

        # 1. PopArt Update (Critical: Before computing losses)
        with torch.no_grad():
            self.policy.value_head.update_stats(returns_raw)

        # ================================================================
        # E-Step: Re-weighting Advantages
        # ================================================================
        with torch.no_grad():
            # Select Top-K Advantages
            k = max(1, int(self.config.topk_fraction * advantages.numel()))
            topk_vals, _ = torch.topk(advantages, k)
            threshold = topk_vals.min()

            # Mask for top-k samples
            mask_bool = advantages >= threshold
            A_sel = advantages[mask_bool]
            K_scalar = float(A_sel.numel())

        # -- Temperature (Eta) Optimization --
        eta = F.softplus(self.log_temperature) + 1e-8

        # Numerical stability: subtract max(A) before exp
        A_max = A_sel.max().detach()
        log_mean_exp = (
            torch.logsumexp((A_sel - A_max) / eta, dim=0)
            - np.log(K_scalar)
            + (A_max / eta)
        )

        dual_loss = eta * self.config.epsilon_eta + eta * log_mean_exp

        self.eta_opt.zero_grad()
        dual_loss.backward()
        self.eta_opt.step()

        # Compute Final Weights for Policy
        with torch.no_grad():
            eta_final = F.softplus(self.log_temperature) + 1e-8
            log_weights = A_sel / eta_final
            weights = torch.softmax(log_weights, dim=0)

            # Logging: Effective Sample Size
            ess = 1.0 / (weights.pow(2).sum() + 1e-12)
            selected_frac = float(K_scalar) / float(advantages.numel())
            adv_std_over_temperature = (advantages.std() / (eta_final + 1e-12)).item()

        # ================================================================
        # M-Step: Policy & Value Update
        # ================================================================

        # Run forward pass on ALL observations
        current_mean, current_log_std = self.policy(obs)

        # -- Policy Loss --
        log_prob = self.policy.log_prob(current_mean, current_log_std, actions).squeeze(
            -1
        )
        log_prob_sel = log_prob[mask_bool]
        weighted_nll = -(weights.detach() * log_prob_sel).sum()

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
            self.config.epsilon_mu - kl_mu_sel.detach()
        ) + alpha_sigma * (self.config.epsilon_sigma - kl_sigma_sel.detach())

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

        # -- Critic Loss (PopArt) --
        v_hat = self.policy.get_value(obs, normalized=True).squeeze(-1)
        target_hat = (
            returns_raw - self.policy.value_head.mu
        ) / self.policy.value_head.sigma
        value_loss = 0.5 * F.mse_loss(v_hat, target_hat.detach())

        total_loss = policy_loss + value_loss

        # -- Update Weights --
        self.opt.zero_grad()
        total_loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_grad_norm
        )
        self.opt.step()

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
            v_pred = self.policy.get_value(obs, normalized=False).squeeze(-1)
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
            "vmpo/epsilon_eta": float(self.config.epsilon_eta),
            "vmpo/temperature_raw": float(eta_final.item()),
            "vmpo/adv_std_over_temperature": float(adv_std_over_temperature),
            # Selection Stats
            "vmpo/selected_frac": float(selected_frac),
            "vmpo/threshold": float(threshold.item()),
            "vmpo/ess": float(ess.item()),
            # Training Dynamics
            "train/entropy": float(entropy.item()),
            "train/param_delta": float(param_delta),
            "train/mean_abs_action": float(mean_abs_action),
            "grad/norm": float(
                grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            ),
            # Data Stats
            "adv/raw_mean": float(advantages.mean().item()),
            "adv/raw_std": float((advantages.std(unbiased=False) + 1e-8).item()),
            "returns/raw_mean": float(returns_raw.mean().item()),
            "returns/raw_std": float((returns_raw.std(unbiased=False) + 1e-8).item()),
            "value/explained_var": float(explained_var.item()),
            # PopArt Internals
            "popart/value_head_mu": float(self.policy.value_head.mu.item()),
            "popart/value_head_sigma": float(self.policy.value_head.sigma.item()),
            "popart/target_hat_mean": float(target_hat.mean().item()),
            "popart/target_hat_std": float(target_hat.std(unbiased=False).item()),
            "popart/v_hat_mean": float(v_hat.mean().item()),
            "popart/v_hat_std": float(v_hat.std(unbiased=False).item()),
        }
