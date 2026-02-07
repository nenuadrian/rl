from __future__ import annotations

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
		self.action_scale: torch.Tensor
		self.action_bias: torch.Tensor
		self.register_buffer("action_scale", (action_high_t - action_low_t) / 2.0)
		self.register_buffer("action_bias", (action_high_t + action_low_t) / 2.0)

	def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		h = self.net(obs)
		mean = self.mean_layer(h)
		log_std = self.log_std_layer(h)
		log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
		return mean, log_std

	def _distribution(self, obs: torch.Tensor) -> Normal:
		mean, log_std = self.forward(obs)
		std = log_std.exp()
		return Normal(mean, std)

	def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		normal = self._distribution(obs)
		x_t = normal.rsample()
		y_t = torch.tanh(x_t)
		action = y_t * self.action_scale + self.action_bias

		log_prob = normal.log_prob(x_t)
		log_prob = log_prob - torch.log(1.0 - y_t.pow(2) + 1e-6)
		log_prob = log_prob.sum(dim=-1, keepdim=True)
		log_prob = log_prob - torch.log(self.action_scale).sum()

		return action, log_prob

	def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
		normal = self._distribution(obs)
		y_t = (actions - self.action_bias) / self.action_scale
		y_t = torch.clamp(y_t, -0.999999, 0.999999)
		x_t = 0.5 * torch.log((1.0 + y_t) / (1.0 - y_t))
		log_prob = normal.log_prob(x_t)
		log_prob = log_prob - torch.log(1.0 - y_t.pow(2) + 1e-6)
		log_prob = log_prob.sum(dim=-1, keepdim=True)
		log_prob = log_prob - torch.log(self.action_scale).sum()
		return log_prob

	def entropy(self, obs: torch.Tensor) -> torch.Tensor:
		normal = self._distribution(obs)
		return normal.entropy().sum(dim=-1, keepdim=True)

	def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
		mean, log_std = self.forward(obs)
		if deterministic:
			action = torch.tanh(mean)
		else:
			std = log_std.exp()
			normal = Normal(mean, std)
			action = torch.tanh(normal.rsample())
		return action * self.action_scale + self.action_bias


class ValueNetwork(nn.Module):
	def __init__(self, obs_dim: int, hidden_sizes: Tuple[int, ...]):
		super().__init__()
		self.net = _mlp(obs_dim, hidden_sizes, 1)

	def forward(self, obs: torch.Tensor) -> torch.Tensor:
		return self.net(obs)


@dataclass
class PPOConfig:
	gamma: float = 0.99
	gae_lambda: float = 0.95
	clip_ratio: float = 0.2
	policy_lr: float = 3e-4
	value_lr: float = 1e-3
	ent_coef: float = 0.0
	vf_coef: float = 0.5
	max_grad_norm: float = 0.5
	target_kl: float = 0.02


class PPOAgent:
	def __init__(
		self,
		obs_dim: int,
		act_dim: int,
		action_low: np.ndarray,
		action_high: np.ndarray,
		device: torch.device,
		hidden_sizes: Tuple[int, ...] = (256, 256),
		config: PPOConfig | None = None,
	):
		self.device = device
		self.config = config or PPOConfig()

		self.policy = GaussianPolicy(
			obs_dim,
			act_dim,
			hidden_sizes=hidden_sizes,
			action_low=action_low,
			action_high=action_high,
		).to(device)
		self.value = ValueNetwork(obs_dim, hidden_sizes).to(device)

		self.policy_opt = torch.optim.Adam(
			self.policy.parameters(), lr=self.config.policy_lr
		)
		self.value_opt = torch.optim.Adam(
			self.value.parameters(), lr=self.config.value_lr
		)

	def act(
		self, obs: torch.Tensor, deterministic: bool = False
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		with torch.no_grad():
			if deterministic:
				action = self.policy.act(obs, deterministic=True)
				log_prob = self.policy.log_prob(obs, action)
			else:
				action, log_prob = self.policy.sample(obs)
			value = self.value(obs)
		return action, log_prob, value

	def update(self, batch: dict) -> dict:
		obs = batch["obs"]
		actions = batch["actions"]
		old_log_probs = batch["log_probs"]
		returns = batch["returns"]
		advantages = batch["advantages"]

		log_probs = self.policy.log_prob(obs, actions)
		ratio = torch.exp(log_probs - old_log_probs)
		clipped_ratio = torch.clamp(
			ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
		)
		policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

		entropy = self.policy.entropy(obs).mean()
		policy_loss = policy_loss - self.config.ent_coef * entropy

		values = self.value(obs)
		value_loss = F.mse_loss(values, returns) * self.config.vf_coef

		self.policy_opt.zero_grad()
		policy_loss.backward()
		nn.utils.clip_grad_norm_(
			self.policy.parameters(), self.config.max_grad_norm
		)
		self.policy_opt.step()

		self.value_opt.zero_grad()
		value_loss.backward()
		nn.utils.clip_grad_norm_(self.value.parameters(), self.config.max_grad_norm)
		self.value_opt.step()

		approx_kl = (old_log_probs - log_probs).mean().abs()

		return {
			"loss/policy": float(policy_loss.item()),
			"loss/value": float(value_loss.item()),
			"entropy": float(entropy.item()),
			"approx_kl": float(approx_kl.item()),
		}
