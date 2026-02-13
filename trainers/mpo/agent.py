from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class LayerNormMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        layer_sizes: Tuple[int, ...],
        activate_final: bool = False,
    ):
        super().__init__()

        layers = []

        # First layer: Linear → LayerNorm → tanh
        layers.append(nn.Linear(in_dim, layer_sizes[0]))
        layers.append(nn.LayerNorm(layer_sizes[0]))
        layers.append(nn.Tanh())

        # Remaining layers: ELU
        for i in range(1, len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            if activate_final or i < len(layer_sizes) - 1:
                layers.append(nn.ELU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallInitLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, std: float = 0.01):
        super().__init__(in_features, out_features)
        nn.init.trunc_normal_(self.weight, std=std)
        nn.init.zeros_(self.bias)


class Critic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        layer_sizes: Tuple[int, ...],
        action_low: np.ndarray,
        action_high: np.ndarray,
    ):
        super().__init__()

        self.action_low: torch.Tensor
        self.action_high: torch.Tensor
        self.register_buffer(
            "action_low", torch.tensor(action_low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.tensor(action_high, dtype=torch.float32)
        )

        self.encoder = LayerNormMLP(
            obs_dim + act_dim,
            layer_sizes,
            activate_final=True,
        )

        # Acme uses a small-init linear head for nondistributional critics.
        self.head = SmallInitLinear(layer_sizes[-1], 1, std=0.01)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        act = torch.maximum(torch.minimum(act, self.action_high), self.action_low)
        x = torch.cat([obs, act], dim=-1)
        return self.head(self.encoder(x))


class DiagonalGaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        layer_sizes: Tuple[int, ...],
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
    ):
        super().__init__()

        self.encoder = LayerNormMLP(
            obs_dim,
            layer_sizes,
            activate_final=True,
        )
        self.policy_mean = nn.Linear(layer_sizes[-1], act_dim)
        self.policy_logstd = nn.Linear(layer_sizes[-1], act_dim)

        nn.init.kaiming_normal_(
            self.policy_mean.weight, a=0.0, mode="fan_in", nonlinearity="linear"
        )
        nn.init.zeros_(self.policy_mean.bias)

        if action_low is None or action_high is None:
            action_low = -np.ones(act_dim, dtype=np.float32)
            action_high = np.ones(act_dim, dtype=np.float32)

        action_low_t = torch.tensor(action_low, dtype=torch.float32)
        action_high_t = torch.tensor(action_high, dtype=torch.float32)
        self.action_low: torch.Tensor
        self.action_high: torch.Tensor
        self.register_buffer("action_low", action_low_t)
        self.register_buffer("action_high", action_high_t)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(obs)
        return self.forward_with_features(h)

    def forward_with_features(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = features
        mean = self.policy_mean(h)
        log_std = self.policy_logstd(h)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        log_std = torch.nan_to_num(log_std, nan=0.0, posinf=0.0, neginf=0.0)
        log_std = torch.clamp(log_std, -20.0, 2.0)
        return mean, log_std

    def log_prob(
        self, mean: torch.Tensor, log_std: torch.Tensor, actions_raw: torch.Tensor
    ) -> torch.Tensor:
        log_std = torch.clamp(log_std, -20.0, 2.0)
        std = log_std.exp()
        normal = Normal(mean, std)
        log_prob = normal.log_prob(actions_raw)
        return log_prob.sum(dim=-1, keepdim=True)

    def _clip_to_env_bounds(self, actions_raw: torch.Tensor) -> torch.Tensor:
        return torch.maximum(
            torch.minimum(actions_raw, self.action_high), self.action_low
        )

    def sample_action_raw_and_exec(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        deterministic: bool,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, -20.0, 2.0)
        if deterministic:
            actions_raw = mean
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            actions_raw = normal.rsample()
        actions_exec = self._clip_to_env_bounds(actions_raw)
        return actions_raw, actions_exec

    def sample_action(
        self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool, **kwargs
    ) -> torch.Tensor:
        _, actions_exec = self.sample_action_raw_and_exec(
            mean, log_std, deterministic, **kwargs
        )
        return actions_exec

    def sample_actions_raw_and_exec(
        self, obs: torch.Tensor, num_actions: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        log_std = torch.clamp(log_std, -20.0, 2.0)
        std = log_std.exp()
        normal = Normal(mean, std)
        actions_raw = normal.rsample(sample_shape=(num_actions,))
        actions_raw = actions_raw.permute(1, 0, 2)
        actions_exec = self._clip_to_env_bounds(actions_raw)

        return actions_raw, actions_exec

    def sample_actions(self, obs: torch.Tensor, num_actions: int) -> torch.Tensor:
        _, actions_exec = self.sample_actions_raw_and_exec(obs, num_actions)
        return actions_exec

    def head_parameters(self):
        return list(self.policy_mean.parameters()) + list(
            self.policy_logstd.parameters()
        )


@dataclass
class MPOConfig:
    gamma: float = 0.99
    # Acme-style hard target sync periods (in learner updates).
    target_policy_update_period: int = 100
    target_critic_update_period: int = 100
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    # MPO loss hyperparameters (aligned with Acme JAX MPO loss).
    # E-step (non-parametric) KL constraint.
    kl_epsilon: float = 0.1
    # M-step (parametric) mean/stddev KL constraints.
    mstep_kl_epsilon: float = 0.1
    per_dim_constraining: bool = True
    # Dual variables (temperature + alphas). We keep legacy names `eta_*` and
    # `lambda_*` to match existing CLI/hparams, but they now correspond to
    # Acme's `temperature` and `alpha_{mean,stddev}`.
    temperature_init: float = 1.0
    temperature_lr: float = 3e-4
    lambda_init: float = 1.0
    lambda_lr: float = 3e-4
    # Optional action penalization (MO-MPO style). By default disabled because
    # this implementation already clips executed actions to env bounds.
    action_penalization: bool = False
    epsilon_penalty: float = 0.001
    max_grad_norm: float = 1.0
    action_samples: int = 20
    use_retrace: bool = False
    retrace_steps: int = 2
    retrace_mc_actions: int = 8
    retrace_lambda: float = 0.95
    optimizer_type: str = "adam"
    sgd_momentum: float = 0.9


class MPOAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device,
        policy_layer_sizes: Tuple[int, ...],
        critic_layer_sizes: Tuple[int, ...],
        config: MPOConfig,
    ):
        self.device = device
        self.config = config
        if self.config.target_policy_update_period <= 0:
            raise ValueError("target_policy_update_period must be >= 1")
        if self.config.target_critic_update_period <= 0:
            raise ValueError("target_critic_update_period must be >= 1")

        # Learner step counter for periodic target synchronization.
        self._num_steps = 0

        self.policy = DiagonalGaussianPolicy(
            obs_dim,
            act_dim,
            layer_sizes=policy_layer_sizes,
            action_low=action_low,
            action_high=action_high,
        ).to(device)

        self.policy_target = copy.deepcopy(self.policy).to(device)
        self.policy_target.eval()
        # Single critic using the Acme control-network critic torso.
        self.q = Critic(
            obs_dim,
            act_dim,
            layer_sizes=critic_layer_sizes,
            action_low=action_low,
            action_high=action_high,
        ).to(device)

        print(self.policy)
        print(self.q)

        self.q_target = copy.deepcopy(self.q).to(device)
        self.q_target.eval()

        # Train policy encoder + head together; critics share a separate encoder.
        self.policy_opt = self._build_optimizer(
            self.policy.parameters(), lr=self.config.policy_lr
        )

        # Critic optimizer.
        self.q_opt = self._build_optimizer(self.q.parameters(), lr=self.config.q_lr)

        # Dual variables (temperature + KL multipliers) in log-space.
        temperature_init_t = torch.tensor(self.config.temperature_init, device=device)
        temperature_init_t = torch.clamp(temperature_init_t, min=1e-8)
        self.log_temperature = nn.Parameter(torch.log(torch.expm1(temperature_init_t)))

        lambda_init_t = torch.tensor(self.config.lambda_init, device=device)
        lambda_init_t = torch.clamp(lambda_init_t, min=1e-8)
        dual_shape = (act_dim,) if self.config.per_dim_constraining else (1,)
        self.log_alpha_mean = nn.Parameter(
            torch.full(
                dual_shape, torch.log(torch.expm1(lambda_init_t)).item(), device=device
            )
        )
        self.log_alpha_stddev = nn.Parameter(
            torch.full(
                dual_shape, torch.log(torch.expm1(lambda_init_t)).item(), device=device
            )
        )

        if self.config.action_penalization:
            self.log_penalty_temperature = nn.Parameter(
                torch.log(torch.expm1(temperature_init_t))
                .clone()
                .detach()
                .requires_grad_(True)
            )
        else:
            self.log_penalty_temperature = None

        # Dual optimizer uses separate LRs for temperature vs alphas.
        temperature_params = [self.log_temperature]
        if self.log_penalty_temperature is not None:
            temperature_params.append(self.log_penalty_temperature)
        alpha_params = [self.log_alpha_mean, self.log_alpha_stddev]
        self.dual_opt = self._build_optimizer(
            [
                {"params": temperature_params, "lr": self.config.temperature_lr},
                {"params": alpha_params, "lr": self.config.lambda_lr},
            ],
        )

    def _build_optimizer(
        self,
        params,
        lr: float | None = None,
    ) -> torch.optim.Optimizer:
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
            f"Unsupported MPO optimizer_type '{self.config.optimizer_type}'. "
            "Expected one of: adam, sgd."
        )

    def _forward_kl_diag_gaussians(
        self,
        mean0: torch.Tensor,
        log_std0: torch.Tensor,
        mean1: torch.Tensor,
        log_std1: torch.Tensor,
    ) -> torch.Tensor:
        var0 = torch.exp(2.0 * log_std0)
        var1 = torch.exp(2.0 * log_std1)
        kl_per_dim = 0.5 * (
            var0 / var1
            + (mean1 - mean0).pow(2) / var1
            - 1.0
            + 2.0 * (log_std1 - log_std0)
        )
        return kl_per_dim.sum(dim=-1, keepdim=True)

    def _kl_diag_gaussian_per_dim(
        self,
        mean_p: torch.Tensor,
        log_std_p: torch.Tensor,
        mean_q: torch.Tensor,
        log_std_q: torch.Tensor,
    ) -> torch.Tensor:
        """KL( p || q ) for diagonal Gaussians, returned per-dimension.

        Shapes: mean/log_std are (B,D). Returns (B,D).
        """
        var_p = torch.exp(2.0 * log_std_p)
        var_q = torch.exp(2.0 * log_std_q)
        return (
            (log_std_q - log_std_p)
            + 0.5 * (var_p + (mean_p - mean_q).pow(2)) / var_q
            - 0.5
        )

    def _compute_weights_and_temperature_loss(
        self,
        q_values: torch.Tensor,
        epsilon: float,
        temperature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Acme-style E-step weights and dual temperature loss.

        q_values shape (B,N). Returns weights (B,N) detached and loss scalar.
        """
        q_detached = q_values.detach() / temperature
        weights = torch.softmax(q_detached, dim=1).detach()
        q_logsumexp = torch.logsumexp(q_detached, dim=1)
        log_num_actions = math.log(q_values.shape[1])
        loss_temperature = temperature * (
            float(epsilon) + q_logsumexp.mean() - log_num_actions
        )
        return weights, loss_temperature

    def _compute_nonparametric_kl_from_weights(
        self, weights: torch.Tensor
    ) -> torch.Tensor:
        """Estimates KL(nonparametric || target) like Acme's diagnostics.

        weights shape (B,N). Returns (B,) KL.
        """
        n = float(weights.shape[1])
        integrand = torch.log(n * weights + 1e-8)
        return (weights * integrand).sum(dim=1)

    def act_with_logp(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, np.ndarray, float]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.policy(obs_t)
            action_raw, action_exec = self.policy.sample_action_raw_and_exec(
                mean, log_std, deterministic
            )
            logp = self.policy.log_prob(mean, log_std, action_raw)
        return (
            action_exec.detach().cpu().numpy().squeeze(0),
            action_raw.detach().cpu().numpy().squeeze(0),
            float(logp.item()),
        )

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        action_exec, _, _ = self.act_with_logp(obs, deterministic=deterministic)
        return action_exec

    def _expected_q_current(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            actions = self.policy_target.sample_actions(
                obs, num_actions=self.config.retrace_mc_actions
            )
            batch_size = obs.shape[0]
            obs_rep = obs.unsqueeze(1).expand(
                batch_size, self.config.retrace_mc_actions, obs.shape[-1]
            )
            obs_flat = obs_rep.reshape(-1, obs.shape[-1])
            act_flat = actions.reshape(-1, actions.shape[-1])
            q = self.q_target(obs_flat, act_flat)
            return q.reshape(batch_size, self.config.retrace_mc_actions).mean(
                dim=1, keepdim=True
            )

    def _retrace_q_target(self, batch: dict) -> torch.Tensor:
        obs_seq = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions_exec_seq = torch.tensor(
            batch["actions_exec"], dtype=torch.float32, device=self.device
        )
        actions_raw_seq = torch.tensor(
            batch["actions_raw"], dtype=torch.float32, device=self.device
        )
        rewards_seq = torch.tensor(
            batch["rewards"], dtype=torch.float32, device=self.device
        )
        next_obs_seq = torch.tensor(
            batch["next_obs"], dtype=torch.float32, device=self.device
        )
        dones_seq = torch.tensor(
            batch["dones"], dtype=torch.float32, device=self.device
        )
        behaviour_logp_seq = torch.tensor(
            batch["behaviour_logp"], dtype=torch.float32, device=self.device
        )

        batch_size, seq_len, obs_dim = obs_seq.shape
        act_dim = actions_exec_seq.shape[-1]

        with torch.no_grad():
            obs_flat = obs_seq.reshape(batch_size * seq_len, obs_dim)
            act_exec_flat = actions_exec_seq.reshape(batch_size * seq_len, act_dim)
            q_t = self.q_target(obs_flat, act_exec_flat).reshape(batch_size, seq_len, 1)

            next_obs_flat = next_obs_seq.reshape(batch_size * seq_len, obs_dim)
            v_next = self._expected_q_current(next_obs_flat).reshape(
                batch_size, seq_len, 1
            )

            delta = rewards_seq + (1.0 - dones_seq) * self.config.gamma * v_next - q_t

            mean, log_std = self.policy_target(obs_flat)
            actions_raw_flat = actions_raw_seq.reshape(batch_size * seq_len, act_dim)
            log_pi = self.policy_target.log_prob(
                mean, log_std, actions_raw_flat
            ).reshape(batch_size, seq_len, 1)
            log_b = behaviour_logp_seq
            log_ratio = log_pi - log_b
            rho = torch.exp(log_ratio).squeeze(-1)
            c = (
                self.config.retrace_lambda * torch.minimum(torch.ones_like(rho), rho)
            ).detach()

            # Correct Retrace recursion:
            # Qret(s0,a0) = Q(s0,a0) + sum_{t=0}^{T-1} gamma^t (prod_{i=1}^t c_i) delta_t
            q_ret = q_t[:, 0, :].clone()  # (B,1)
            cont = torch.ones((batch_size, 1), device=self.device)
            c_prod = torch.ones((batch_size, 1), device=self.device)
            discount = torch.ones((batch_size, 1), device=self.device)

            dones_flat = dones_seq.squeeze(-1)  # (B,T)

            for t in range(seq_len):
                if t > 0:
                    cont = cont * (1.0 - dones_flat[:, t - 1 : t])
                    c_prod = c_prod * c[:, t : t + 1]
                    discount = discount * self.config.gamma

                q_ret = q_ret + cont * discount * c_prod * delta[:, t, :]

        return q_ret

    def update(self, batch: dict) -> dict:
        # Acme-style periodic hard sync of online -> target networks.
        if self._num_steps % self.config.target_policy_update_period == 0:
            print("[MPOAgent] Syncing policy_target networks...")
            self._sync_module(self.policy, self.policy_target)
        if self._num_steps % self.config.target_critic_update_period == 0:
            print("[MPOAgent] Syncing target networks...")
            self._sync_module(self.q, self.q_target)

        is_sequence_batch = (
            isinstance(batch.get("obs"), np.ndarray) and batch["obs"].ndim == 3
        )
        use_retrace = bool(getattr(self.config, "use_retrace", True))

        if use_retrace and is_sequence_batch and self.config.retrace_steps > 1:
            target = self._retrace_q_target(batch)
            obs = torch.tensor(
                batch["obs"][:, 0, :], dtype=torch.float32, device=self.device
            )
            actions = torch.tensor(
                batch["actions_exec"][:, 0, :], dtype=torch.float32, device=self.device
            )
        else:
            obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
            actions = torch.tensor(
                batch["actions"], dtype=torch.float32, device=self.device
            )
            rewards = torch.tensor(
                batch["rewards"], dtype=torch.float32, device=self.device
            )
            next_obs = torch.tensor(
                batch["next_obs"], dtype=torch.float32, device=self.device
            )
            dones = torch.tensor(
                batch["dones"], dtype=torch.float32, device=self.device
            )

            with torch.no_grad():
                next_actions = self.policy_target.sample_actions(
                    next_obs, num_actions=self.config.action_samples
                )
                batch_size = next_obs.shape[0]
                next_obs_rep = next_obs.unsqueeze(1).expand(
                    batch_size, self.config.action_samples, next_obs.shape[-1]
                )
                next_obs_flat = next_obs_rep.reshape(-1, next_obs.shape[-1])
                next_act_flat = next_actions.reshape(-1, next_actions.shape[-1])
                q_target = self.q_target(next_obs_flat, next_act_flat)
                q_target = q_target.reshape(
                    batch_size, self.config.action_samples
                ).mean(dim=1, keepdim=True)
                target = rewards + (1.0 - dones) * self.config.gamma * q_target

        q = self.q(obs, actions)
        q_loss = F.mse_loss(q, target)

        # Phase A: critic update
        self.q_opt.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(
            self.q.parameters(),
            self.config.max_grad_norm,
        )
        self.q_opt.step()

        # Phase B (Acme-style): update dual vars then do policy M-step.
        batch_size = obs.shape[0]
        num_samples = self.config.action_samples

        mean_online, log_std_online = self.policy(obs)
        with torch.no_grad():
            mean_target, log_std_target = self.policy_target(obs)

            sampled_actions_raw, sampled_actions_exec = (
                self.policy_target.sample_actions_raw_and_exec(
                    obs, num_actions=num_samples
                )
            )  # (B,N,D)

            obs_rep = obs.unsqueeze(1).expand(batch_size, num_samples, obs.shape[-1])
            obs_flat = obs_rep.reshape(-1, obs.shape[-1])
            act_exec_flat = sampled_actions_exec.reshape(
                -1, sampled_actions_exec.shape[-1]
            )

            q_vals = self.q_target(obs_flat, act_exec_flat).reshape(
                batch_size, num_samples
            )

        temperature = F.softplus(self.log_temperature) + 1e-8
        weights, loss_temperature = self._compute_weights_and_temperature_loss(
            q_vals, self.config.kl_epsilon, temperature
        )

        penalty_kl_rel = torch.tensor(0.0, device=self.device)
        if self.config.action_penalization and self.log_penalty_temperature is not None:
            penalty_temperature = F.softplus(self.log_penalty_temperature) + 1e-8
            diff = sampled_actions_raw.detach() - torch.clamp(
                sampled_actions_raw.detach(),
                self.policy.action_low,
                self.policy.action_high,
            )
            cost = -torch.linalg.norm(diff, dim=-1)  # (B,N)
            penalty_weights, loss_penalty_temperature = (
                self._compute_weights_and_temperature_loss(
                    cost, self.config.epsilon_penalty, penalty_temperature
                )
            )
            weights = weights + penalty_weights
            loss_temperature = loss_temperature + loss_penalty_temperature
            penalty_kl = self._compute_nonparametric_kl_from_weights(penalty_weights)
            penalty_kl_rel = penalty_kl.mean() / float(self.config.epsilon_penalty)

        # KL(nonparametric || target) diagnostic (relative).
        kl_nonparametric = self._compute_nonparametric_kl_from_weights(weights)
        kl_q_rel = kl_nonparametric.mean() / float(self.config.kl_epsilon)

        # Compute Acme-style decomposed losses.
        std_online = torch.exp(log_std_online)
        std_target = torch.exp(log_std_target)

        # Fixed distributions for decomposition.
        actions = sampled_actions_raw.detach()  # (B,N,D), stop-gradient wrt sampling.
        mean_online_exp = mean_online.unsqueeze(1)
        std_online_exp = std_online.unsqueeze(1)
        mean_target_exp = mean_target.unsqueeze(1)
        std_target_exp = std_target.unsqueeze(1)

        # fixed_stddev: mean=online_mean, std=target_std
        log_prob_fixed_stddev = (
            Normal(mean_online_exp, std_target_exp).log_prob(actions).sum(dim=-1)
        )
        # fixed_mean: mean=target_mean, std=online_std
        log_prob_fixed_mean = (
            Normal(mean_target_exp, std_online_exp).log_prob(actions).sum(dim=-1)
        )

        # Cross entropy / weighted log-prob.
        loss_policy_mean = -(weights * log_prob_fixed_stddev).sum(dim=1).mean()
        loss_policy_std = -(weights * log_prob_fixed_mean).sum(dim=1).mean()
        loss_policy = loss_policy_mean + loss_policy_std

        # Decomposed KL constraints (target || online-decomposed).
        if self.config.per_dim_constraining:
            kl_mean = self._kl_diag_gaussian_per_dim(
                mean_target.detach(),
                log_std_target.detach(),
                mean_online,
                log_std_target.detach(),
            )  # (B,D)
            kl_std = self._kl_diag_gaussian_per_dim(
                mean_target.detach(),
                log_std_target.detach(),
                mean_target.detach(),
                log_std_online,
            )  # (B,D)
        else:
            kl_mean = self._forward_kl_diag_gaussians(
                mean_target.detach(),
                log_std_target.detach(),
                mean_online,
                log_std_target.detach(),
            )  # (B,1)
            kl_std = self._forward_kl_diag_gaussians(
                mean_target.detach(),
                log_std_target.detach(),
                mean_target.detach(),
                log_std_online,
            )  # (B,1)

        mean_kl_mean = kl_mean.mean(dim=0)
        mean_kl_std = kl_std.mean(dim=0)

        alpha_mean = F.softplus(self.log_alpha_mean) + 1e-8
        alpha_std = F.softplus(self.log_alpha_stddev) + 1e-8

        loss_kl_mean = (alpha_mean.detach() * mean_kl_mean).sum()
        loss_kl_std = (alpha_std.detach() * mean_kl_std).sum()
        loss_kl_penalty = loss_kl_mean + loss_kl_std

        loss_alpha_mean = (
            alpha_mean * (self.config.mstep_kl_epsilon - mean_kl_mean.detach())
        ).sum()
        loss_alpha_std = (
            alpha_std * (self.config.mstep_kl_epsilon - mean_kl_std.detach())
        ).sum()

        # Update dual variables (temperature + alphas).
        dual_loss = loss_temperature + loss_alpha_mean + loss_alpha_std
        self.dual_opt.zero_grad()
        dual_loss.backward()
        nn.utils.clip_grad_norm_(
            [
                p
                for p in [
                    self.log_temperature,
                    self.log_alpha_mean,
                    self.log_alpha_stddev,
                    self.log_penalty_temperature,
                ]
                if p is not None
            ],
            self.config.max_grad_norm,
        )
        self.dual_opt.step()

        # Policy update (M-step).
        policy_total_loss = loss_policy + loss_kl_penalty
        self.policy_opt.zero_grad()
        policy_total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.policy_opt.step()

        # Increment step counter for target-sync cadence bookkeeping.
        self._num_steps += 1

        # Diagnostics for training monitoring.
        temperature_val = float(
            (F.softplus(self.log_temperature) + 1e-8).detach().item()
        )
        lambda_val = float(
            (F.softplus(self.log_alpha_mean).mean() + 1e-8).detach().item()
        )

        return {
            "loss/q": float(q_loss.item()),
            "loss/policy": float(loss_policy.item()),
            "loss/dual_eta": float(loss_temperature.detach().item()),
            "loss/dual": float(dual_loss.detach().item()),
            "kl/q_pi": float(kl_q_rel.detach().item()),
            "kl/mean": float(mean_kl_mean.mean().detach().item()),
            "kl/std": float(mean_kl_std.mean().detach().item()),
            "eta": temperature_val,
            "lambda": lambda_val,
            "alpha_mean": float(
                (F.softplus(self.log_alpha_mean) + 1e-8).mean().detach().item()
            ),
            "alpha_std": float(
                (F.softplus(self.log_alpha_stddev) + 1e-8).mean().detach().item()
            ),
            "q/min": float(q_vals.min().detach().item()),
            "q/max": float(q_vals.max().detach().item()),
            "pi/std_min": float(std_online.min().detach().item()),
            "pi/std_max": float(std_online.max().detach().item()),
            "penalty_kl/q_pi": float(penalty_kl_rel.detach().item()),
        }

    def _sync_module(self, net: nn.Module, target: nn.Module) -> None:
        target.load_state_dict(net.state_dict())
