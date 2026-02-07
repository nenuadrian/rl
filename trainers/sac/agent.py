import copy
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal



def _mlp(in_dim: int, hidden_sizes: Tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers = []
    last_dim = in_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(last_dim, size))
        layers.append(nn.ReLU())
        last_dim = size
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        self.net = _mlp(obs_dim + act_dim, hidden_sizes, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        log_std_bounds: Tuple[float, float] = (-20.0, 2.0),
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
    ):
        super().__init__()
        self.net = _mlp(obs_dim, hidden_sizes, hidden_sizes[-1])
        self.mean_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_min, self.log_std_max = log_std_bounds

        if action_low is None or action_high is None:
            action_low = -np.ones(act_dim, dtype=np.float32)
            action_high = np.ones(act_dim, dtype=np.float32)

        action_low_t = torch.tensor(action_low, dtype=torch.float32)
        action_high_t = torch.tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high_t - action_low_t) / 2.0)
        self.register_buffer("action_bias", (action_high_t + action_low_t) / 2.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob = log_prob - torch.log(1.0 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob = log_prob - torch.log(self.action_scale).sum()

        return action, log_prob

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self.forward(obs)
        if deterministic:
            action = torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            action = torch.tanh(normal.rsample())
        return action * self.action_scale + self.action_bias

    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.act(obs, deterministic=True)



@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    alpha_lr: float = 3e-4
    automatic_entropy_tuning: bool = True


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        config: SACConfig | None = None,
    ):
        self.device = device
        self.config = config or SACConfig()

        self.policy = GaussianPolicy(
            obs_dim,
            act_dim,
            hidden_sizes=hidden_sizes,
            action_low=action_low,
            action_high=action_high,
        ).to(device)
        self.q1 = QNetwork(obs_dim, act_dim, hidden_sizes).to(device)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_sizes).to(device)

        self.q1_target = copy.deepcopy(self.q1).to(device)
        self.q2_target = copy.deepcopy(self.q2).to(device)
        self.q1_target.eval()
        self.q2_target.eval()

        self.policy_opt = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.policy_lr
        )
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=self.config.q_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=self.config.q_lr)

        self.automatic_entropy_tuning = self.config.automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -float(act_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
        else:
            self.log_alpha = torch.tensor(0.0, device=device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.policy.act(obs_t, deterministic=deterministic)
        return action.cpu().numpy().squeeze(0)

    def update(self, batch: dict) -> dict:
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
            next_actions, next_log_prob = self.policy.sample(next_obs)
            q1_target = self.q1_target(next_obs, next_actions)
            q2_target = self.q2_target(next_obs, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_prob
            target = rewards + (1.0 - dones) * self.config.gamma * q_target

        q1 = self.q1(obs, actions)
        q2 = self.q2(obs, actions)

        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        new_actions, log_prob = self.policy.sample(obs)
        q1_pi = self.q1(obs, new_actions)
        q2_pi = self.q2(obs, new_actions)
        q_pi = torch.min(q1_pi, q2_pi)

        policy_loss = (self.alpha * log_prob - q_pi).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_prob + self.target_entropy).detach()
            ).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return {
            "loss/q1": float(q1_loss.item()),
            "loss/q2": float(q2_loss.item()),
            "loss/policy": float(policy_loss.item()),
            "loss/alpha": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
        }

    def _soft_update(self, net: nn.Module, target: nn.Module) -> None:
        tau = self.config.tau
        for param, target_param in zip(net.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * param.data)
