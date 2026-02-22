from __future__ import annotations

import math
from typing import Tuple, Dict, Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from minerva.utils.running_norm import RunningNorm


def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class DiagonalGaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        policy_layer_sizes: Tuple[int, ...] = (256, 256),
        value_layer_sizes: Tuple[int, ...] = (256, 256),
        shared_encoder: bool = False,
    ):
        super().__init__()
        self.shared_encoder = shared_encoder
        self.obs_normalizer = RunningNorm(obs_dim)

        self.policy_encoder = self._build_mlp(obs_dim, policy_layer_sizes)
        self.value_encoder = (
            self.policy_encoder
            if shared_encoder
            else self._build_mlp(obs_dim, value_layer_sizes)
        )

        self.policy_mean = layer_init(
            nn.Linear(policy_layer_sizes[-1], act_dim), std=0.01
        )
        self.policy_logstd = nn.Parameter(torch.zeros(1, act_dim))

        value_head_in_dim = (
            policy_layer_sizes[-1] if shared_encoder else value_layer_sizes[-1]
        )
        self.value_head = layer_init(nn.Linear(value_head_in_dim, 1), std=1.0)

    @staticmethod
    def _build_mlp(
        input_dim: int,
        hidden_layer_sizes: Tuple[int, ...],
    ) -> nn.Sequential:
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            layers.extend([layer_init(nn.Linear(last_dim, hidden_dim)), nn.Tanh()])
            last_dim = hidden_dim
        return nn.Sequential(*layers)

    def policy_logstd_parameters(self) -> list[nn.Parameter]:
        return [self.policy_logstd]

    def _mean_and_log_std(self, encoded_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Output raw (unbounded) mean, will be clipped by environment
        mean = self.policy_mean(encoded_obs)
        log_std = self.policy_logstd.expand_as(mean)
        return mean, log_std

    def get_policy_dist_params(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Internal helper to get raw distribution parameters."""
        obs = self.obs_normalizer(obs)
        h = self.policy_encoder(obs)
        return self._mean_and_log_std(h)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_policy_dist_params(obs)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.obs_normalizer(obs)
        if self.shared_encoder:
            h = self.policy_encoder(obs)
        else:
            h = self.value_encoder(obs)
        return self.value_head(h)

    def forward_all(self, obs):
        obs = self.obs_normalizer(obs)
        h = self.policy_encoder(obs)
        mean, log_std = self._mean_and_log_std(h)
        h_val = h if self.shared_encoder else self.value_encoder(obs)
        v = self.value_head(h_val)
        return mean, log_std, v

    def log_prob(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probability of actions under the policy."""
        std = log_std.exp()
        normal = Normal(mean, std)
        return normal.log_prob(actions).sum(dim=-1, keepdim=True)

    def sample_action(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if deterministic:
            return mean, torch.zeros((mean.shape[0], 1), device=mean.device)
        std = log_std.exp()
        normal = Normal(mean, std)
        action = normal.sample()
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob


class VMPOAgent:

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
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

        self.policy = DiagonalGaussianPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            policy_layer_sizes=policy_layer_sizes,
            value_layer_sizes=value_layer_sizes,
            shared_encoder=shared_encoder,
        ).to(device)

        value_params = list(self.policy.value_head.parameters())
        if not self.policy.shared_encoder:
            value_params = list(self.policy.value_encoder.parameters()) + value_params
        policy_lr_eff = self.policy_lr / math.sqrt(self.m_steps)
        value_lr_eff = self.value_lr / math.sqrt(self.m_steps)

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

        # Temperature (eta) for advantage weighting: eta = exp(log_temperature)
        temperature_init_t = torch.tensor(
            self.temperature_init, dtype=torch.float32, device=device
        )
        temperature_init_t = torch.clamp(temperature_init_t, min=1e-8)
        self.log_temperature = nn.Parameter(torch.log(temperature_init_t))
        self.eta_opt = self._build_optimizer(
            [self.log_temperature], lr=self.temperature_lr
        )

        # KL Penalties (alpha) for trust region: alpha = exp(log_alpha)
        # exp(0.0) = 1.0
        self.log_alpha_mu = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32, device=device)
        )
        self.log_alpha_sigma = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32, device=device)
        )
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
        log_std_np = log_std.detach().cpu().numpy()

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

        # Update obs stats once per rollout batch, then freeze during repeated
        # m-step passes so normalization does not depend on optimizer epochs.
        obs_normalizer = self.policy.obs_normalizer
        was_obs_norm_training = bool(obs_normalizer.training)
        if obs.numel() > 0:
            with torch.no_grad():
                obs_normalizer.update_stats(obs.detach())
        obs_normalizer.eval()

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
        # LaTeX: \eta = \exp(\log \eta)
        eta = torch.exp(self.log_temperature)
        # LaTeX: s_t = \frac{\hat{A}_t}{\eta}
        scaled_advantages = advantages / eta

        # Numerical stability: subtract global max before exponentiation.
        max_scaled_advantage = scaled_advantages.max().detach()

        with torch.no_grad():
            if self.topk_fraction < 1.0:
                # LaTeX: k = \lfloor \rho N \rfloor
                k = int(self.topk_fraction * scaled_advantages.numel())
                topk_vals, _ = torch.topk(scaled_advantages.detach(), k)
                # LaTeX: \tau = \min(\operatorname{TopK}(s, k))
                threshold = topk_vals.min()
                # LaTeX: m_t = \mathbf{1}[s_t \ge \tau]
                topk_weights = (scaled_advantages.detach() >= threshold).to(
                    scaled_advantages.dtype
                )
            else:
                # LaTeX: \tau = \min_t s_t
                threshold = scaled_advantages.detach().min()
                topk_weights = torch.ones_like(scaled_advantages)

            # Mask for selected samples (used for KL terms and diagnostics).
            mask_bool = topk_weights > 0.0
            if not bool(mask_bool.any()):
                # Fallback to avoid empty reductions on tiny rollouts.
                mask_bool = torch.ones_like(mask_bool, dtype=torch.bool)
                topk_weights = torch.ones_like(topk_weights)

        # LaTeX: \tilde{\psi}_t = m_t \exp(s_t - s_{\max})
        unnormalized_weights = (
            topk_weights
            * torch.exp(scaled_advantages - max_scaled_advantage)
        )
        # LaTeX: Z = \sum_t \tilde{\psi}_t
        sum_weights = unnormalized_weights.sum() + 1e-8
        # LaTeX: N_{sel} = \sum_t m_t
        num_samples = topk_weights.sum() + 1e-8
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
            # LaTeX: \eta' = \exp(\log \eta)
            eta_final = torch.exp(self.log_temperature)
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

            # -- Decomposed Policy Loss (MPO-style) --
            # Separate mean and std optimization to ensure both get
            # independent gradient signals.  Using the joint NLL allows
            # the mean to "absorb" all the learning, leaving logstd
            # with near-zero gradients and kl/std stuck at 0.

            # Mean loss: cross-entropy with current mean but FIXED (old) std
            # LaTeX: \mathcal{L}_{\mu} = -\sum_t \psi_t \log \mathcal{N}(a_t; \mu_\theta(s_t), \sigma_{old})
            log_prob_fixed_std = self.policy.log_prob(
                current_mean, old_log_stds, actions
            ).squeeze(-1)
            weighted_nll_mean = -(weights_detached * log_prob_fixed_std).sum()

            # Std loss: cross-entropy with FIXED (old) mean but current std
            # LaTeX: \mathcal{L}_{\sigma} = -\sum_t \psi_t \log \mathcal{N}(a_t; \mu_{old}, \sigma_\theta(s_t))
            log_prob_fixed_mean = self.policy.log_prob(
                old_means, current_log_std, actions
            ).squeeze(-1)
            weighted_nll_std = -(weights_detached * log_prob_fixed_mean).sum()

            # Combined NLL with decomposed gradients
            weighted_nll = weighted_nll_mean + weighted_nll_std

            # -- KL Divergence (Full Batch D, per paper Eq. 5) --
            # Paper: "Note that here we use the full batch D, not ˜D."
            # The KL constraint must be over ALL states, not just selected ones.
            old_std = old_log_stds.exp()
            new_std = current_log_std.exp()

            # Decoupled KL over full batch (Appendix C, Eqs. 25-26)
            # LaTeX: D_{\mu} = \frac{1}{|D|}\sum_{s \in D} \frac{1}{2}(\mu_\theta - \mu_{old})^T \Sigma_{old}^{-1} (\mu_\theta - \mu_{old})
            kl_mu = (
                (0.5 * ((current_mean - old_means) ** 2 / (old_std**2 + 1e-8)))
                .sum(dim=-1)
                .mean()
            )
            # LaTeX: D_{\Sigma} = \frac{1}{|D|}\sum_{s \in D} \frac{1}{2}\left(\text{Tr}(\Sigma_\theta^{-1}\Sigma_{old}) - d + \log\frac{|\Sigma_\theta|}{|\Sigma_{old}|}\right)
            kl_sigma = (
                (
                    0.5
                    * (
                        (old_std**2) / (new_std**2 + 1e-8)
                        - 1.0
                        + 2.0 * (current_log_std - old_log_stds)
                    )
                )
                .sum(dim=-1)
                .mean()
            )

            # Diagnostic: KL on selected samples only (for logging)
            with torch.no_grad():
                mean_sel = current_mean[mask_bool]
                old_mean_sel = old_means[mask_bool]
                old_std_sel = old_log_stds[mask_bool].exp()
                log_std_sel = current_log_std[mask_bool]
                new_std_sel = log_std_sel.exp()
                kl_mu_sel = (
                    (0.5 * ((mean_sel - old_mean_sel) ** 2 / (old_std_sel**2 + 1e-8)))
                    .sum(dim=-1)
                    .mean()
                )
                kl_sigma_sel = (
                    (
                        0.5
                        * (
                            (old_std_sel**2) / (new_std_sel**2 + 1e-8)
                            - 1.0
                            + 2.0 * (log_std_sel - old_log_stds[mask_bool])
                        )
                    )
                    .sum(dim=-1)
                    .mean()
                )

            # -- Alpha Optimization (Eq. 5, first term with sg on KL) --
            # LaTeX: \alpha_{\mu} = \exp(\log \alpha_{\mu})
            alpha_mu = torch.exp(self.log_alpha_mu)
            # LaTeX: \alpha_{\sigma} = \exp(\log \alpha_{\sigma})
            alpha_sigma = torch.exp(self.log_alpha_sigma)

            # LaTeX: \mathcal{L}_{\alpha} = \alpha_{\mu}(\epsilon_{\mu} - \text{sg}[D_{\mu}]) + \alpha_{\sigma}(\epsilon_{\sigma} - \text{sg}[D_{\sigma}])
            alpha_loss = alpha_mu * (
                self.epsilon_mu - kl_mu.detach()
            ) + alpha_sigma * (self.epsilon_sigma - kl_sigma.detach())

            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            # -- Final Policy Loss (Eq. 5, second term with sg on alpha) --
            with torch.no_grad():
                alpha_mu_det = torch.exp(self.log_alpha_mu).detach()
                alpha_sigma_det = torch.exp(self.log_alpha_sigma).detach()

            # LaTeX: \mathcal{L}_{\pi} = \mathcal{L}_{\pi}^{NLL} + \text{sg}[\alpha_{\mu}]D_{\mu} + \text{sg}[\alpha_{\sigma}]D_{\sigma}
            policy_loss = (
                weighted_nll
                + (alpha_mu_det * kl_mu)
                + (alpha_sigma_det * kl_sigma)
            )

            # -- Critic Loss --
            # LaTeX: \mathcal{L}_{V} = \frac{1}{2}\mathbb{E}\left[(V_{\phi}(s_t) - R_t)^2\right]
            value_loss = 0.5 * F.mse_loss(v_pred, returns_raw.detach())

            # LaTeX: \mathcal{L} = \mathcal{L}_{\pi} + \mathcal{L}_{V}

            total_loss = policy_loss + value_loss
            self.combined_opt.zero_grad()
            total_loss.backward()

            # Capture gradient norms before clipping (last m-step only).
            with torch.no_grad():
                policy_params = (
                    list(self.policy.policy_encoder.parameters())
                    + list(self.policy.policy_mean.parameters())
                )
                logstd_params = self.policy.policy_logstd_parameters()
                value_params = (
                    (
                        []
                        if self.policy.shared_encoder
                        else list(self.policy.value_encoder.parameters())
                    )
                    + list(self.policy.value_head.parameters())
                )
                grad_policy_norm = torch.nn.utils.clip_grad_norm_(
                    policy_params, float("inf")
                ).item()
                grad_logstd_norm = torch.nn.utils.clip_grad_norm_(
                    logstd_params, float("inf")
                ).item()
                grad_value_norm = torch.nn.utils.clip_grad_norm_(
                    value_params, float("inf")
                ).item()

            nn.utils.clip_grad_norm_(
                policy_params + logstd_params + value_params,
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
                (0.5 * (1 + torch.log(2 * torch.pi * new_std**2))).sum(-1).mean()
            )

            # 4. Policy std diagnostics
            std_eval = log_std_eval.exp()
            policy_std_mean = float(std_eval.mean().item())
            policy_std_min = float(std_eval.min().item())
            policy_std_max = float(std_eval.max().item())
            policy_logstd_mean = float(log_std_eval.mean().item())

            # 5. Obs normalizer diagnostics
            obs_norm_mean_norm = float(obs_normalizer.mean.norm().item())
            obs_norm_std_mean = float(
                torch.sqrt(obs_normalizer.var + obs_normalizer.eps).mean().item()
            )
            obs_norm_count = float(obs_normalizer.count.item())

        obs_normalizer.train(was_obs_norm_training)

        return {
            "loss/total": float(total_loss.item()),
            "loss/value": float(value_loss.item()),
            "loss/policy": float(policy_loss.item()),
            "loss/policy_weighted_nll": float(weighted_nll.item()),
            "loss/nll_mean": float(weighted_nll_mean.item()),
            "loss/nll_std": float(weighted_nll_std.item()),
            "loss/policy_kl_mean_pen": float((alpha_mu_det * kl_mu).item()),
            "loss/policy_kl_std_pen": float((alpha_sigma_det * kl_sigma).item()),
            "loss/alpha": float(alpha_loss.item()),
            # KL Diagnostics (full batch D, used in optimization)
            "kl/mean": float(kl_mu.detach().item()),
            "kl/std": float(kl_sigma.detach().item()),
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
            # Training Dynamics
            "train/entropy": float(entropy.item()),
            "train/param_delta": float(param_delta),
            "train/mean_abs_action": float(mean_abs_action),
            # Gradient norms (pre-clip, last m-step)
            "grad/policy_norm": grad_policy_norm,
            "grad/logstd_norm": grad_logstd_norm,
            "grad/value_norm": grad_value_norm,
            # Policy std diagnostics
            "policy/std_mean": policy_std_mean,
            "policy/std_min": policy_std_min,
            "policy/std_max": policy_std_max,
            "policy/logstd_mean": policy_logstd_mean,
            # Obs normalizer diagnostics
            "obs_norm/mean_norm": obs_norm_mean_norm,
            "obs_norm/std_mean": obs_norm_std_mean,
            "obs_norm/count": obs_norm_count,
            # Data Stats
            "adv/raw_mean": float(advantages.mean().item()),
            "adv/raw_std": float((advantages.std(unbiased=False) + 1e-8).item()),
            "returns/raw_mean": float(returns_raw.mean().item()),
            "returns/raw_std": float((returns_raw.std(unbiased=False) + 1e-8).item()),
            "value/explained_var": float(explained_var.item()),
            "value/pred_mean": float(v_pred.mean().item()),
            "value/pred_std": float(v_pred.std(unbiased=False).item()),
        }
