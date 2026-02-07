from __future__ import annotations

import copy
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


class QNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Tuple[int, ...],
        obs_encoder: nn.Module | None = None,
        encoder_out_dim: int | None = None,
    ):
        super().__init__()
        self.obs_encoder = obs_encoder
        if self.obs_encoder is None:
            in_dim = obs_dim + act_dim
        else:
            if encoder_out_dim is None:
                raise ValueError(
                    "encoder_out_dim must be provided when using obs_encoder."
                )
            in_dim = encoder_out_dim + act_dim
        self.net = nn.Sequential(
            _mlp(in_dim, hidden_sizes),
            nn.Linear(hidden_sizes[-1], 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.obs_encoder is None:
            x = torch.cat([obs, action], dim=-1)
        else:
            x = torch.cat([self.obs_encoder(obs), action], dim=-1)
        return self.net(x)

    def head_parameters(self):
        return self.net.parameters()


class SquashedGaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        encoder: nn.Module | None = None,
    ):
        super().__init__()
        if len(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must have at least one layer.")

        self.encoder = encoder if encoder is not None else _mlp(obs_dim, hidden_sizes)
        self.policy_mean = nn.Linear(hidden_sizes[-1], act_dim)
        self.policy_logstd = nn.Linear(hidden_sizes[-1], act_dim)

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
        h = self.encoder(obs)
        mean = self.policy_mean(h)
        log_std = self.policy_logstd(h)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        log_std = torch.nan_to_num(log_std, nan=0.0, posinf=0.0, neginf=0.0)
        log_std = torch.clamp(log_std, -20.0, 2.0)
        return mean, log_std

    def log_prob(
        self, mean: torch.Tensor, log_std: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        log_std = torch.clamp(log_std, -20.0, 2.0)
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
        log_std = torch.clamp(log_std, -20.0, 2.0)
        if deterministic:
            action = torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            action = torch.tanh(normal.rsample())
        return action * self.action_scale + self.action_bias

    def sample_actions(self, obs: torch.Tensor, num_actions: int) -> torch.Tensor:
        mean, log_std = self.forward(obs)
        log_std = torch.clamp(log_std, -20.0, 2.0)
        std = log_std.exp()
        normal = Normal(mean, std)
        eps = normal.rsample(sample_shape=(num_actions,))
        eps = eps.permute(1, 0, 2)
        actions = torch.tanh(eps)
        return actions * self.action_scale + self.action_bias

    def head_parameters(self):
        return list(self.policy_mean.parameters()) + list(
            self.policy_logstd.parameters()
        )


class DiagonalGaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        encoder: nn.Module | None = None,
    ):
        super().__init__()
        if len(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must have at least one layer.")

        self.encoder = encoder if encoder is not None else _mlp(obs_dim, hidden_sizes)
        self.policy_mean = nn.Linear(hidden_sizes[-1], act_dim)
        self.policy_logstd = nn.Linear(hidden_sizes[-1], act_dim)

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
        return torch.maximum(torch.minimum(actions_raw, self.action_high), self.action_low)

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
    tau: float = 0.005
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    eta_init: float = 1.0
    eta_lr: float = 1e-3
    kl_epsilon: float = 0.1
    max_grad_norm: float = 1.0
    action_samples: int = 16
    retrace_steps: int = 5
    retrace_mc_actions: int = 16


class MPOAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        config: MPOConfig | None = None,
    ):
        self.device = device
        self.config = config or MPOConfig()

        self.shared_encoder = _mlp(obs_dim, hidden_sizes)

        self.policy = DiagonalGaussianPolicy(
            obs_dim,
            act_dim,
            hidden_sizes=hidden_sizes,
            action_low=action_low,
            action_high=action_high,
            encoder=self.shared_encoder,
        ).to(device)
        self.policy_target = copy.deepcopy(self.policy).to(device)
        self.policy_target.eval()
        self.q1 = QNetwork(
            obs_dim,
            act_dim,
            hidden_sizes,
            obs_encoder=self.shared_encoder,
            encoder_out_dim=hidden_sizes[-1],
        ).to(device)
        self.q2 = QNetwork(
            obs_dim,
            act_dim,
            hidden_sizes,
            obs_encoder=self.shared_encoder,
            encoder_out_dim=hidden_sizes[-1],
        ).to(device)

        self.q1_target = copy.deepcopy(self.q1).to(device)
        self.q2_target = copy.deepcopy(self.q2).to(device)
        self.q1_target.eval()
        self.q2_target.eval()

        self.policy_opt = torch.optim.Adam(
            self.policy.head_parameters(), lr=self.config.policy_lr
        )
        self.q1_opt = torch.optim.Adam(self.q1.head_parameters(), lr=self.config.q_lr)
        self.q2_opt = torch.optim.Adam(self.q2.head_parameters(), lr=self.config.q_lr)
        self.encoder_opt = torch.optim.Adam(
            self.shared_encoder.parameters(), lr=self.config.policy_lr
        )

        self.log_eta = nn.Parameter(
            torch.log(torch.tensor(self.config.eta_init, device=device))
        )
        self.eta_opt = torch.optim.Adam([self.log_eta], lr=self.config.eta_lr)

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

    def _expected_q_target(self, obs: torch.Tensor) -> torch.Tensor:
        actions = self.policy_target.sample_actions(
            obs, num_actions=self.config.retrace_mc_actions
        )
        batch_size = obs.shape[0]
        obs_rep = obs.unsqueeze(1).expand(
            batch_size, self.config.retrace_mc_actions, obs.shape[-1]
        )
        obs_flat = obs_rep.reshape(-1, obs.shape[-1])
        act_flat = actions.reshape(-1, actions.shape[-1])
        q1 = self.q1_target(obs_flat, act_flat)
        q2 = self.q2_target(obs_flat, act_flat)
        q = torch.min(q1, q2)
        return q.reshape(batch_size, self.config.retrace_mc_actions).mean(dim=1, keepdim=True)

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
        dones_seq = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device)
        behaviour_logp_seq = torch.tensor(
            batch["behaviour_logp"], dtype=torch.float32, device=self.device
        )

        batch_size, seq_len, obs_dim = obs_seq.shape
        act_dim = actions_exec_seq.shape[-1]

        with torch.no_grad():
            obs_flat = obs_seq.reshape(batch_size * seq_len, obs_dim)
            act_exec_flat = actions_exec_seq.reshape(batch_size * seq_len, act_dim)
            q1_t = self.q1_target(obs_flat, act_exec_flat)
            q2_t = self.q2_target(obs_flat, act_exec_flat)
            q_t = torch.min(q1_t, q2_t).reshape(batch_size, seq_len, 1)

            next_obs_flat = next_obs_seq.reshape(batch_size * seq_len, obs_dim)
            v_next = self._expected_q_target(next_obs_flat).reshape(batch_size, seq_len, 1)

            delta = rewards_seq + (1.0 - dones_seq) * self.config.gamma * v_next - q_t

            mean, log_std = self.policy(obs_flat)
            actions_raw_flat = actions_raw_seq.reshape(batch_size * seq_len, act_dim)
            log_pi = self.policy.log_prob(mean, log_std, actions_raw_flat).reshape(
                batch_size, seq_len, 1
            )
            log_b = behaviour_logp_seq
            log_ratio = log_pi - log_b
            c = torch.exp(torch.minimum(torch.zeros_like(log_ratio), log_ratio)).squeeze(-1)

            ones = torch.ones(batch_size, 1, device=self.device)
            c_shift = torch.cat([ones, c[:, 1:]], dim=1)
            prod_c = torch.cumprod(c_shift, dim=1)

            not_done = (1.0 - dones_seq.squeeze(-1))
            alive = torch.cumprod(torch.cat([ones, not_done[:, :-1]], dim=1), dim=1)

            gammas = (self.config.gamma ** torch.arange(seq_len, device=self.device)).view(1, seq_len)

            q0 = q_t[:, 0, 0]
            retrace_sum = torch.sum(
                gammas * prod_c * alive * delta.squeeze(-1), dim=1
            )
            q_ret = (q0 + retrace_sum).unsqueeze(-1)

        return q_ret

    def update(self, batch: dict) -> dict:
        is_sequence_batch = isinstance(batch.get("obs"), np.ndarray) and batch["obs"].ndim == 3

        if is_sequence_batch and self.config.retrace_steps > 1:
            target = self._retrace_q_target(batch)
            obs = torch.tensor(batch["obs"][:, 0, :], dtype=torch.float32, device=self.device)
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
            dones = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device)

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
                q1_target = self.q1_target(next_obs_flat, next_act_flat)
                q2_target = self.q2_target(next_obs_flat, next_act_flat)
                q_target = torch.min(q1_target, q2_target)
                q_target = q_target.reshape(batch_size, self.config.action_samples).mean(
                    dim=1, keepdim=True
                )
                target = rewards + (1.0 - dones) * self.config.gamma * q_target

        q1 = self.q1(obs, actions)
        q2 = self.q2(obs, actions)

        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)

        mean, log_std = self.policy(obs)
        with torch.no_grad():
            sampled_actions_raw, sampled_actions_exec = (
                self.policy_target.sample_actions_raw_and_exec(
                    obs, num_actions=self.config.action_samples
                )
            )
            batch_size = obs.shape[0]
            obs_rep = obs.unsqueeze(1).expand(
                batch_size, self.config.action_samples, obs.shape[-1]
            )
            obs_flat = obs_rep.reshape(-1, obs.shape[-1])
            act_flat = sampled_actions_exec.reshape(-1, sampled_actions_exec.shape[-1])
            q1_vals = self.q1(obs_flat, act_flat)
            q2_vals = self.q2(obs_flat, act_flat)
            q_vals = torch.min(q1_vals, q2_vals)
            q_vals = q_vals.reshape(batch_size, self.config.action_samples)

        q_vals_detach = q_vals.detach()
        eta = F.softplus(self.log_eta) + 1e-8
        dual_loss = (
            eta * self.config.kl_epsilon
            + eta * torch.log(torch.mean(torch.exp(q_vals_detach / eta), dim=1) + 1e-8)
        ).mean()
        self.eta_opt.zero_grad()
        dual_loss.backward()
        self.eta_opt.step()
        eta = F.softplus(self.log_eta).detach() + 1e-8

        weights = torch.softmax(q_vals_detach / eta, dim=1)

        mean_exp = mean.unsqueeze(1).expand(
            batch_size, self.config.action_samples, mean.shape[-1]
        )
        log_std_exp = log_std.unsqueeze(1).expand_as(mean_exp)
        log_prob = self.policy.log_prob(
            mean_exp.reshape(-1, mean.shape[-1]),
            log_std_exp.reshape(-1, log_std.shape[-1]),
            sampled_actions_raw.reshape(-1, sampled_actions_raw.shape[-1]),
        )
        log_prob = log_prob.reshape(batch_size, self.config.action_samples)

        policy_loss = -(weights.detach() * log_prob).sum(dim=1).mean()
        kl_q_pi = (weights * (torch.log(weights + 1e-8) - log_prob)).sum(dim=1).mean()

        total_loss = q1_loss + q2_loss + policy_loss

        self.q1_opt.zero_grad()
        self.q2_opt.zero_grad()
        self.policy_opt.zero_grad()
        self.encoder_opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q1.head_parameters()), self.config.max_grad_norm
        )
        nn.utils.clip_grad_norm_(
            list(self.q2.head_parameters()), self.config.max_grad_norm
        )
        nn.utils.clip_grad_norm_(
            self.policy.head_parameters(), self.config.max_grad_norm
        )
        nn.utils.clip_grad_norm_(
            self.shared_encoder.parameters(), self.config.max_grad_norm
        )
        self.q1_opt.step()
        self.q2_opt.step()
        self.policy_opt.step()
        self.encoder_opt.step()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        self._soft_update(self.policy, self.policy_target)

        return {
            "loss/q1": float(q1_loss.item()),
            "loss/q2": float(q2_loss.item()),
            "loss/policy": float(policy_loss.item()),
            "loss/dual_eta": float(dual_loss.item()),
            "kl/q_pi": float(kl_q_pi.item()),
            "eta": float(eta.item()),
        }

    def _soft_update(self, net: nn.Module, target: nn.Module) -> None:
        tau = self.config.tau
        for param, target_param in zip(net.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * param.data)
