from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.ppo.agent import GaussianPolicy, ValueNetwork


def _flat_params(module: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()])


def _set_flat_params(module: nn.Module, flat_params: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for param in module.parameters():
            numel = param.numel()
            param.copy_(flat_params[offset : offset + numel].view_as(param))
            offset += numel


def _flat_grads(
    grads: tuple[torch.Tensor | None, ...], params: tuple[nn.Parameter, ...]
) -> torch.Tensor:
    out: list[torch.Tensor] = []
    for grad, param in zip(grads, params):
        if grad is None:
            out.append(torch.zeros_like(param).reshape(-1))
        else:
            out.append(grad.reshape(-1))
    return torch.cat(out)


def _conjugate_gradient(
    Avp,
    b: torch.Tensor,
    nsteps: int,
    residual_tol: float = 1e-10,
) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)

    for _ in range(nsteps):
        Avp_p = Avp(p)
        denom = torch.dot(p, Avp_p)
        if torch.abs(denom) < 1e-12:
            break
        alpha = rdotr / (denom + 1e-8)
        x = x + alpha * p
        r = r - alpha * Avp_p
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr

    return x


def _kl_old_to_new(
    old_mean: torch.Tensor,
    old_log_std: torch.Tensor,
    new_mean: torch.Tensor,
    new_log_std: torch.Tensor,
) -> torch.Tensor:
    old_std = old_log_std.exp()
    new_std = new_log_std.exp()
    mean_diff_sq = (old_mean - new_mean).pow(2)

    kl = (
        new_log_std
        - old_log_std
        + (old_std.pow(2) + mean_diff_sq) / (2.0 * new_std.pow(2) + 1e-8)
        - 0.5
    )
    return kl.sum(dim=-1, keepdim=True)


class TRPOAgent:
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
        target_kl: float = 0.01,
        cg_iters: int = 10,
        cg_damping: float = 0.1,
        backtrack_coeff: float = 0.8,
        backtrack_iters: int = 10,
        value_lr: float = 3e-4,
        value_epochs: int = 10,
        value_minibatch_size: int = 256,
        max_grad_norm: float = 0.5,
        normalize_advantages: bool = True,
        optimizer_type: str = "adam",
        sgd_momentum: float = 0.9,
    ):
        self.device = device
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.target_kl = float(target_kl)
        self.cg_iters = int(cg_iters)
        self.cg_damping = float(cg_damping)
        self.backtrack_coeff = float(backtrack_coeff)
        self.backtrack_iters = int(backtrack_iters)
        self.value_lr = float(value_lr)
        self.value_epochs = int(value_epochs)
        self.value_minibatch_size = int(value_minibatch_size)
        self.max_grad_norm = float(max_grad_norm)
        self.normalize_advantages = bool(normalize_advantages)
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
        self.value_opt = self._build_value_optimizer(self.value.parameters())

    def _build_value_optimizer(self, params) -> torch.optim.Optimizer:
        optimizer_type = self.optimizer_type.strip().lower()
        if optimizer_type == "adam":
            return torch.optim.Adam(params, lr=self.value_lr, eps=1e-5)
        if optimizer_type == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.value_lr,
                momentum=float(self.sgd_momentum),
            )
        raise ValueError(
            f"Unsupported TRPO optimizer_type '{self.optimizer_type}'. "
            "Expected one of: adam, sgd."
        )

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if deterministic:
                action = self.policy.sample_action(obs, deterministic=True)
                log_prob = self.policy.log_prob(obs, action)
            else:
                action, log_prob = self.policy.sample(obs)
            value = self.value(obs)
        return action, log_prob, value

    def _policy_objective(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs = self.policy.log_prob(obs, actions)
        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        surrogate = (ratio * advantages).mean()
        return surrogate, ratio, log_ratio

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        returns = batch["returns"]
        advantages = batch["advantages"]

        if old_log_probs.ndim == 1:
            old_log_probs = old_log_probs.unsqueeze(-1)
        if returns.ndim == 1:
            returns = returns.unsqueeze(-1)
        if advantages.ndim == 1:
            advantages = advantages.unsqueeze(-1)

        with torch.no_grad():
            old_mean, old_log_std = self.policy.forward(obs)

        policy_params = tuple(self.policy.parameters())

        surrogate_before_t, _, _ = self._policy_objective(
            obs=obs,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
        )
        surrogate_before = float(surrogate_before_t.item())

        grads = torch.autograd.grad(surrogate_before_t, policy_params, create_graph=False)
        g = _flat_grads(grads, policy_params).detach()

        step_norm = 0.0
        expected_improve = 0.0
        line_search_frac = 0.0
        line_search_accepted = False

        old_params = _flat_params(self.policy)

        if torch.isfinite(g).all() and torch.norm(g) > 1e-12:

            def hvp(v: torch.Tensor) -> torch.Tensor:
                mean, log_std = self.policy.forward(obs)
                kl = _kl_old_to_new(old_mean, old_log_std, mean, log_std).mean()
                kl_grads = torch.autograd.grad(
                    kl, policy_params, create_graph=True, retain_graph=True
                )
                flat_kl_grads = _flat_grads(kl_grads, policy_params)
                kl_v = (flat_kl_grads * v).sum()
                hvp_parts = torch.autograd.grad(
                    kl_v, policy_params, create_graph=False, retain_graph=False
                )
                fisher_v = _flat_grads(hvp_parts, policy_params).detach()
                return fisher_v + self.cg_damping * v

            step_dir = _conjugate_gradient(
                Avp=hvp,
                b=g,
                nsteps=int(self.cg_iters),
            )

            fisher_step_dir = hvp(step_dir)
            step_den = 0.5 * torch.dot(step_dir, fisher_step_dir)

            if torch.isfinite(step_den) and step_den > 0:
                scale = torch.sqrt(
                    torch.tensor(
                        self.target_kl,
                        dtype=step_den.dtype,
                        device=step_den.device,
                    )
                    / (step_den + 1e-8)
                )
                full_step = step_dir * scale
                step_norm = float(torch.norm(full_step).item())
                expected_improve = float(torch.dot(g, full_step).item())

                if np.isfinite(expected_improve) and expected_improve > 0:
                    for j in range(int(self.backtrack_iters)):
                        frac = self.backtrack_coeff**j
                        candidate_params = old_params + frac * full_step
                        _set_flat_params(self.policy, candidate_params)

                        with torch.no_grad():
                            surrogate_after_t, _, _ = self._policy_objective(
                                obs=obs,
                                actions=actions,
                                old_log_probs=old_log_probs,
                                advantages=advantages,
                            )
                            new_mean, new_log_std = self.policy.forward(obs)
                            kl_after_t = _kl_old_to_new(
                                old_mean,
                                old_log_std,
                                new_mean,
                                new_log_std,
                            ).mean()

                        improve = float(surrogate_after_t.item()) - surrogate_before
                        kl_after = float(kl_after_t.item())

                        if (
                            np.isfinite(improve)
                            and np.isfinite(kl_after)
                            and improve > 0.0
                            and kl_after <= self.target_kl
                        ):
                            line_search_frac = float(frac)
                            line_search_accepted = True
                            break

                    if not line_search_accepted:
                        _set_flat_params(self.policy, old_params)
                else:
                    _set_flat_params(self.policy, old_params)
            else:
                _set_flat_params(self.policy, old_params)

        value_losses: list[float] = []
        n_samples = obs.shape[0]
        mb_size = min(int(self.value_minibatch_size), n_samples)

        for _ in range(int(self.value_epochs)):
            perm = torch.randperm(n_samples, device=obs.device)
            for start in range(0, n_samples, mb_size):
                idx = perm[start : start + mb_size]
                value_loss = F.mse_loss(self.value(obs[idx]), returns[idx])
                self.value_opt.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                self.value_opt.step()
                value_losses.append(float(value_loss.item()))

        with torch.no_grad():
            surrogate_after_t, ratio_t, log_ratio_t = self._policy_objective(
                obs=obs,
                actions=actions,
                old_log_probs=old_log_probs,
                advantages=advantages,
            )
            new_mean, new_log_std = self.policy.forward(obs)
            kl_after_t = _kl_old_to_new(old_mean, old_log_std, new_mean, new_log_std).mean()
            entropy_t = self.policy.entropy(obs).mean()

            values_t = self.value(obs)
            value_mae_t = (values_t - returns).abs().mean()
            returns_var = torch.var(returns)
            explained_var_t = 1.0 - torch.var(returns - values_t) / (returns_var + 1e-8)

        value_loss_mean = float(np.mean(value_losses)) if value_losses else 0.0

        return {
            "policy/surrogate_before": surrogate_before,
            "policy/surrogate_after": float(surrogate_after_t.item()),
            "policy/improvement": float(surrogate_after_t.item()) - surrogate_before,
            "policy/kl": float(kl_after_t.item()),
            "policy/entropy": float(entropy_t.item()),
            "policy/step_norm": step_norm,
            "policy/expected_improve": expected_improve,
            "policy/line_search_frac": line_search_frac,
            "policy/line_search_accepted": float(line_search_accepted),
            "policy/ratio_mean": float(ratio_t.mean().item()),
            "policy/ratio_std": float(ratio_t.std(unbiased=False).item()),
            "policy/old_approx_kl": float((-log_ratio_t).mean().item()),
            "loss/value": value_loss_mean,
            "value/mae": float(value_mae_t.item()),
            "value/explained_variance": float(explained_var_t.item()),
        }
