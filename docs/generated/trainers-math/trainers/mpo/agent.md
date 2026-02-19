# `trainers.mpo.agent` Math-Annotated Source

_Source: `minerva/trainers/mpo/agent.py`_

Each `# LaTeX:` annotation is rendered below next to its source line.

## Rendered Math Annotations

### Line 281

```python
        policy_lr_effective = float(self.policy_lr) / max(1, int(self.m_steps))
```

$$
\tilde{\lambda}_{\pi} = \frac{\lambda_{\pi}}{\max(1, M)}
$$

### Line 356

```python
        q_detached = q_values.detach() / temperature
```

$$
\bar{Q}_{b,i} = \frac{Q_{b,i}}{\eta}
$$

### Line 357

```python
        weights = torch.softmax(q_detached, dim=1).detach()
```

$$
q_{b,i} = \frac{\exp(\bar{Q}_{b,i})}{\sum_j \exp(\bar{Q}_{b,j})}
$$

### Line 358

```python
        q_logsumexp = torch.logsumexp(q_detached, dim=1)
```

$$
\log Z_b = \log \sum_i \exp(\bar{Q}_{b,i})
$$

### Line 360

```python
        loss_temperature = temperature * (
```

$$
\mathcal{L}_{\eta} = \eta\left(\epsilon + \frac{1}{B}\sum_b \log Z_b - \log N\right)
$$

### Line 373

```python
        integrand = torch.log(n * weights + 1e-8)
```

$$
\ell_{b,i} = \log\!\left(N q_{b,i}\right)
$$

### Line 374

```python
        return (weights * integrand).sum(dim=1)
```

$$
D_{KL}(q_b\|u) = \sum_i q_{b,i}\log\!\left(N q_{b,i}\right)
$$

### Line 467

```python
            delta = rewards_seq + (1.0 - dones_seq) * self.gamma * v_next - q_t
```

$$
\delta_t = r_t + \gamma(1-d_t)V(s_{t+1}) - Q(s_t,a_t)
$$

### Line 475

```python
            log_ratio = log_pi - log_b
```

$$
\log \rho_t = \log \pi(a_t|s_t) - \log b(a_t|s_t)
$$

### Line 476

```python
            rho = torch.exp(log_ratio).squeeze(-1)
```

$$
\rho_t = \exp(\log \rho_t)
$$

### Line 477

```python
            c = (self.retrace_lambda * torch.minimum(torch.ones_like(rho), rho)).detach()
```

$$
c_t = \lambda \min(1, \rho_t)
$$

### Line 481

```python
            q_ret = q_t[:, 0, :].clone()
```

$$
Q^{ret} \leftarrow Q(s_0, a_0)
$$

### Line 490

```python
                    cont = cont * (1.0 - dones_flat[:, t - 1 : t])
```

$$
m_t \leftarrow m_{t-1}(1-d_{t-1})
$$

### Line 491

```python
                    c_prod = c_prod * c[:, t : t + 1]
```

$$
C_t \leftarrow C_{t-1} c_t
$$

### Line 492

```python
                    discount = discount * self.gamma
```

$$
\Gamma_t \leftarrow \Gamma_{t-1}\gamma
$$

### Line 494

```python
                q_ret = q_ret + cont * discount * c_prod * delta[:, t, :]
```

$$
Q^{ret} \leftarrow Q^{ret} + m_t \Gamma_t C_t \delta_t
$$

### Line 579

```python
                target = rewards + (1.0 - dones) * self.gamma * q_target
```

$$
y_t = r_t + \gamma(1-d_t)\bar{Q}(s_{t+1})
$$

### Line 587

```python
        q_loss = F.mse_loss(q, target)
```

$$
\mathcal{L}_Q = \mathbb{E}\!\left[(Q(s_t,a_t) - y_t)^2\right]
$$

### Line 629

```python
        temperature = F.softplus(log_temp) + 1e-8
```

$$
\eta = \operatorname{softplus}(\tilde{\eta}) + \epsilon
$$

### Line 636

```python
        kl_q_rel = kl_nonparametric.mean() / float(self.kl_epsilon)
```

$$
\mathrm{KL}_{rel} = \frac{\mathbb{E}[D_{KL}(q\|u)]}{\epsilon}
$$

### Line 659

```python
        loss_policy_mean = -(weights * log_prob_fixed_stddev).sum(dim=1).mean()
```

$$
\mathcal{L}_{\pi,\mu} = -\mathbb{E}_b\sum_i q_{b,i}\log \pi_{\mu,\sigma'}(a_{b,i}|s_b)
$$

### Line 660

```python
        loss_policy_std = -(weights * log_prob_fixed_mean).sum(dim=1).mean()
```

$$
\mathcal{L}_{\pi,\sigma} = -\mathbb{E}_b\sum_i q_{b,i}\log \pi_{\mu',\sigma}(a_{b,i}|s_b)
$$

### Line 661

```python
        loss_policy = loss_policy_mean + loss_policy_std
```

$$
\mathcal{L}_{\pi} = \mathcal{L}_{\pi,\mu} + \mathcal{L}_{\pi,\sigma}
$$

### Line 676

```python
        mean_kl_mean = kl_mean.mean(dim=0)
```

$$
\bar{D}_{\mu,j} = \frac{1}{B}\sum_b D_{\mu,b,j}
$$

### Line 677

```python
        mean_kl_std = kl_std.mean(dim=0)
```

$$
\bar{D}_{\sigma,j} = \frac{1}{B}\sum_b D_{\sigma,b,j}
$$

### Line 682

```python
        alpha_mean = F.softplus(log_alpha_mean) + 1e-8
```

$$
\alpha_{\mu,j} = \operatorname{softplus}(\tilde{\alpha}_{\mu,j}) + \epsilon
$$

### Line 683

```python
        alpha_std = F.softplus(log_alpha_stddev) + 1e-8
```

$$
\alpha_{\sigma,j} = \operatorname{softplus}(\tilde{\alpha}_{\sigma,j}) + \epsilon
$$

### Line 685

```python
        loss_kl_mean = (alpha_mean.detach() * mean_kl_mean).sum()
```

$$
\mathcal{L}_{KL,\mu} = \sum_j \alpha_{\mu,j}\bar{D}_{\mu,j}
$$

### Line 686

```python
        loss_kl_std = (alpha_std.detach() * mean_kl_std).sum()
```

$$
\mathcal{L}_{KL,\sigma} = \sum_j \alpha_{\sigma,j}\bar{D}_{\sigma,j}
$$

### Line 687

```python
        loss_kl_penalty = loss_kl_mean + loss_kl_std
```

$$
\mathcal{L}_{KL} = \mathcal{L}_{KL,\mu} + \mathcal{L}_{KL,\sigma}
$$

### Line 691

```python
        ).sum()
```

$$
\mathcal{L}_{\alpha_{\mu}} = \sum_j \alpha_{\mu,j}(\epsilon_{KL} - \bar{D}_{\mu,j})
$$

### Line 694

```python
        ).sum()
```

$$
\mathcal{L}_{\alpha_{\sigma}} = \sum_j \alpha_{\sigma,j}(\epsilon_{KL} - \bar{D}_{\sigma,j})
$$

### Line 697

```python
        dual_loss = loss_temperature + loss_alpha_mean + loss_alpha_std
```

$$
\mathcal{L}_{dual} = \mathcal{L}_{\eta} + \mathcal{L}_{\alpha_{\mu}} + \mathcal{L}_{\alpha_{\sigma}}
$$

### Line 724

```python
        alpha_mean_det = (F.softplus(self.log_alpha_mean) + 1e-8).detach()
```

$$
\alpha_{\mu,j}' = \operatorname{softplus}(\tilde{\alpha}_{\mu,j}) + \epsilon
$$

### Line 725

```python
        alpha_std_det = (F.softplus(self.log_alpha_stddev) + 1e-8).detach()
```

$$
\alpha_{\sigma,j}' = \operatorname{softplus}(\tilde{\alpha}_{\sigma,j}) + \epsilon
$$

### Line 774

```python
            policy_total_loss = loss_policy + loss_kl_penalty
```

$$
\mathcal{L}_{\pi}^{total} = \mathcal{L}_{\pi} + \mathcal{L}_{KL}
$$

### Line 796

```python
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
```

$$
\mathcal{H}(q) = -\mathbb{E}_b\sum_i q_{b,i}\log q_{b,i}
$$

## Full Source

```python
from __future__ import annotations

import copy
import math
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
        gamma: float = 0.99,
        target_networks_update_period: int = 100,
        policy_lr: float = 3e-4,
        q_lr: float = 3e-4,
        kl_epsilon: float = 0.1,
        mstep_kl_epsilon: float = 0.1,
        temperature_init: float = 1.0,
        temperature_lr: float = 3e-4,
        lambda_init: float = 1.0,
        lambda_lr: float = 3e-4,
        epsilon_penalty: float = 0.001,
        max_grad_norm: float = 1.0,
        action_samples: int = 20,
        use_retrace: bool = False,
        retrace_steps: int = 2,
        retrace_mc_actions: int = 8,
        retrace_lambda: float = 0.95,
        optimizer_type: str = "adam",
        sgd_momentum: float = 0.9,
        init_log_alpha_mean: float = 10.0,
        init_log_alpha_stddev: float = 1000.0,
        m_steps: int = 1,
    ):
        self.device = device
        self.gamma = float(gamma)
        self.target_networks_update_period = int(target_networks_update_period)
        self.policy_lr = float(policy_lr)
        self.q_lr = float(q_lr)
        self.kl_epsilon = float(kl_epsilon)
        self.mstep_kl_epsilon = float(mstep_kl_epsilon)
        self.temperature_init = float(temperature_init)
        self.temperature_lr = float(temperature_lr)
        self.lambda_init = float(lambda_init)
        self.lambda_lr = float(lambda_lr)
        self.epsilon_penalty = float(epsilon_penalty)
        self.max_grad_norm = float(max_grad_norm)
        self.action_samples = int(action_samples)
        self.use_retrace = bool(use_retrace)
        self.retrace_steps = int(retrace_steps)
        self.retrace_mc_actions = int(retrace_mc_actions)
        self.retrace_lambda = float(retrace_lambda)
        self.optimizer_type = str(optimizer_type)
        self.sgd_momentum = float(sgd_momentum)
        self.m_steps = int(m_steps)

        # Learner step counter for periodic target synchronization.
        self._num_steps = 0
        self._skipped_nonfinite_batches = 0

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
        policy_lr_effective = float(self.policy_lr) / max(1, int(self.m_steps))  # LaTeX: \tilde{\lambda}_{\pi} = \frac{\lambda_{\pi}}{\max(1, M)}
        self.policy_opt = self._build_optimizer(
            self.policy.parameters(), lr=policy_lr_effective
        )

        # Critic optimizer.
        self.q_opt = self._build_optimizer(self.q.parameters(), lr=self.q_lr)

        # Dual variables (temperature + KL multipliers) in log-space.
        self.log_temperature = nn.Parameter(
            torch.tensor(self.temperature_init, device=device)
        )

        lambda_init_t = torch.tensor(self.lambda_init, device=device)
        lambda_init_t = torch.clamp(lambda_init_t, min=1e-8)
        dual_shape = (act_dim,)
        self.log_alpha_mean = nn.Parameter(
            torch.full(dual_shape, init_log_alpha_mean, device=device)
        )

        self.log_alpha_stddev = nn.Parameter(
            torch.full(dual_shape, init_log_alpha_stddev, device=device)
        )

        # Dual optimizer uses separate LRs for temperature vs alphas.
        temperature_params = [self.log_temperature]
        alpha_params = [self.log_alpha_mean, self.log_alpha_stddev]
        self.dual_opt = self._build_optimizer(
            [
                {"params": temperature_params, "lr": self.temperature_lr},
                {"params": alpha_params, "lr": self.lambda_lr},
            ],
        )

    def _build_optimizer(
        self,
        params,
        lr: float | None = None,
    ) -> torch.optim.Optimizer:
        optimizer_type = self.optimizer_type.strip().lower()
        kwargs: dict[str, float] = {}
        if lr is not None:
            kwargs["lr"] = float(lr)

        if optimizer_type == "sgd":
            kwargs["momentum"] = float(self.sgd_momentum)
            return torch.optim.SGD(params, **kwargs)
        else:
            kwargs["eps"] = 1e-5
            return torch.optim.Adam(params, **kwargs)

    def _kl_diag_gaussian_per_dim(
        self,
        mean_p,
        log_std_p,
        mean_q,
        log_std_q,
    ):
        inv_var_q = torch.exp(-2.0 * log_std_q)
        var_ratio = torch.exp(2.0 * (log_std_p - log_std_q))

        return 0.5 * (var_ratio + (mean_p - mean_q).pow(2) * inv_var_q - 1.0) + (
            log_std_q - log_std_p
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
        q_detached = q_values.detach() / temperature  # LaTeX: \bar{Q}_{b,i} = \frac{Q_{b,i}}{\eta}
        weights = torch.softmax(q_detached, dim=1).detach()  # LaTeX: q_{b,i} = \frac{\exp(\bar{Q}_{b,i})}{\sum_j \exp(\bar{Q}_{b,j})}
        q_logsumexp = torch.logsumexp(q_detached, dim=1)  # LaTeX: \log Z_b = \log \sum_i \exp(\bar{Q}_{b,i})
        log_num_actions = math.log(q_values.shape[1])
        loss_temperature = temperature * (  # LaTeX: \mathcal{L}_{\eta} = \eta\left(\epsilon + \frac{1}{B}\sum_b \log Z_b - \log N\right)
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
        integrand = torch.log(n * weights + 1e-8)  # LaTeX: \ell_{b,i} = \log\!\left(N q_{b,i}\right)
        return (weights * integrand).sum(dim=1)  # LaTeX: D_{KL}(q_b\|u) = \sum_i q_{b,i}\log\!\left(N q_{b,i}\right)

    def _to_device_tensor(self, value: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Fast path for float32 host arrays -> device tensors."""
        if isinstance(value, torch.Tensor):
            return value.to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=True,
            )

        arr = np.asarray(value, dtype=np.float32)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        return torch.from_numpy(arr).to(device=self.device, non_blocking=True)

    def _assert_finite_tensors(self, tensors: dict[str, torch.Tensor]) -> bool:
        try:
            for name, tensor in tensors.items():
                assert bool(
                    torch.isfinite(tensor).all()
                ), f"non-finite values in '{name}'"
        except AssertionError as exc:
            self._skipped_nonfinite_batches += 1
            if (
                self._skipped_nonfinite_batches <= 5
                or self._skipped_nonfinite_batches % 100 == 0
            ):
                print(
                    "[MPO][warn] "
                    f"{exc}; skipped batch #{self._skipped_nonfinite_batches}"
                )
            return False
        return True

    def act_with_logp(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, np.ndarray, float]:
        obs_t = self._to_device_tensor(obs).unsqueeze(0)
        with torch.inference_mode():
            mean, log_std = self.policy(obs_t)
            action_raw, action_exec = self.policy.sample_action_raw_and_exec(
                mean, log_std, deterministic
            )
            logp = self.policy.log_prob(mean, log_std, action_raw)
        return (
            action_exec.cpu().numpy().squeeze(0),
            action_raw.cpu().numpy().squeeze(0),
            float(logp.item()),
        )

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        action_exec, _, _ = self.act_with_logp(obs, deterministic=deterministic)
        return action_exec

    def _expected_q_current(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            actions = self.policy_target.sample_actions(
                obs, num_actions=self.retrace_mc_actions
            )
            batch_size = obs.shape[0]
            obs_rep = obs.unsqueeze(1).expand(
                batch_size, self.retrace_mc_actions, obs.shape[-1]
            )
            obs_flat = obs_rep.reshape(-1, obs.shape[-1])
            act_flat = actions.reshape(-1, actions.shape[-1])
            q = self.q_target(obs_flat, act_flat)
            return q.reshape(batch_size, self.retrace_mc_actions).mean(
                dim=1, keepdim=True
            )

    def _retrace_q_target(self, batch: dict) -> torch.Tensor:
        obs_seq = self._to_device_tensor(batch["obs"])
        actions_exec_seq = self._to_device_tensor(batch["actions_exec"])
        actions_raw_seq = self._to_device_tensor(batch["actions_raw"])
        rewards_seq = self._to_device_tensor(batch["rewards"])
        next_obs_seq = self._to_device_tensor(batch["next_obs"])
        dones_seq = self._to_device_tensor(batch["dones"])
        behaviour_logp_seq = self._to_device_tensor(batch["behaviour_logp"])

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

            delta = rewards_seq + (1.0 - dones_seq) * self.gamma * v_next - q_t  # LaTeX: \delta_t = r_t + \gamma(1-d_t)V(s_{t+1}) - Q(s_t,a_t)

            mean, log_std = self.policy_target(obs_flat)
            actions_raw_flat = actions_raw_seq.reshape(batch_size * seq_len, act_dim)
            log_pi = self.policy_target.log_prob(
                mean, log_std, actions_raw_flat
            ).reshape(batch_size, seq_len, 1)
            log_b = behaviour_logp_seq
            log_ratio = log_pi - log_b  # LaTeX: \log \rho_t = \log \pi(a_t|s_t) - \log b(a_t|s_t)
            rho = torch.exp(log_ratio).squeeze(-1)  # LaTeX: \rho_t = \exp(\log \rho_t)
            c = (self.retrace_lambda * torch.minimum(torch.ones_like(rho), rho)).detach()  # LaTeX: c_t = \lambda \min(1, \rho_t)

            # Correct Retrace recursion:
            # Qret(s0,a0) = Q(s0,a0) + sum_{t=0}^{T-1} gamma^t (prod_{i=1}^t c_i) delta_t
            q_ret = q_t[:, 0, :].clone()  # LaTeX: Q^{ret} \leftarrow Q(s_0, a_0)
            cont = torch.ones((batch_size, 1), device=self.device)
            c_prod = torch.ones((batch_size, 1), device=self.device)
            discount = torch.ones((batch_size, 1), device=self.device)

            dones_flat = dones_seq.squeeze(-1)  # (B,T)

            for t in range(seq_len):
                if t > 0:
                    cont = cont * (1.0 - dones_flat[:, t - 1 : t])  # LaTeX: m_t \leftarrow m_{t-1}(1-d_{t-1})
                    c_prod = c_prod * c[:, t : t + 1]  # LaTeX: C_t \leftarrow C_{t-1} c_t
                    discount = discount * self.gamma  # LaTeX: \Gamma_t \leftarrow \Gamma_{t-1}\gamma

                q_ret = q_ret + cont * discount * c_prod * delta[:, t, :]  # LaTeX: Q^{ret} \leftarrow Q^{ret} + m_t \Gamma_t C_t \delta_t

        return q_ret

    def update(self, batch: dict) -> dict | None:
        if self._num_steps % self.target_networks_update_period == 0:
            self.policy_target.load_state_dict(self.policy.state_dict())
            self.q_target.load_state_dict(self.q.state_dict())

        obs_batch = batch.get("obs")
        is_sequence_batch = isinstance(obs_batch, (np.ndarray, torch.Tensor)) and (
            obs_batch.ndim == 3
        )
        use_retrace = self.use_retrace

        if use_retrace and is_sequence_batch and self.retrace_steps > 1:
            obs_seq = self._to_device_tensor(batch["obs"])
            actions_exec_seq = self._to_device_tensor(batch["actions_exec"])
            actions_raw_seq = self._to_device_tensor(batch["actions_raw"])
            rewards_seq = self._to_device_tensor(batch["rewards"])
            next_obs_seq = self._to_device_tensor(batch["next_obs"])
            dones_seq = self._to_device_tensor(batch["dones"])
            behaviour_logp_seq = self._to_device_tensor(batch["behaviour_logp"])

            if not self._assert_finite_tensors(
                {
                    "obs": obs_seq,
                    "actions_exec": actions_exec_seq,
                    "actions_raw": actions_raw_seq,
                    "rewards": rewards_seq,
                    "next_obs": next_obs_seq,
                    "dones": dones_seq,
                    "behaviour_logp": behaviour_logp_seq,
                }
            ):
                return None

            target = self._retrace_q_target(
                {
                    "obs": obs_seq,
                    "actions_exec": actions_exec_seq,
                    "actions_raw": actions_raw_seq,
                    "rewards": rewards_seq,
                    "next_obs": next_obs_seq,
                    "dones": dones_seq,
                    "behaviour_logp": behaviour_logp_seq,
                }
            )
            obs = obs_seq[:, 0, :]
            actions = actions_exec_seq[:, 0, :]
        else:
            obs = self._to_device_tensor(batch["obs"])
            actions_key = (
                "actions_exec" if "actions_exec" in batch.keys() else "actions"
            )
            actions = self._to_device_tensor(batch[actions_key])
            rewards = self._to_device_tensor(batch["rewards"])
            next_obs = self._to_device_tensor(batch["next_obs"])
            dones = self._to_device_tensor(batch["dones"])

            if not self._assert_finite_tensors(
                {
                    "obs": obs,
                    "actions": actions,
                    "rewards": rewards,
                    "next_obs": next_obs,
                    "dones": dones,
                }
            ):
                return None

            with torch.no_grad():
                next_actions = self.policy_target.sample_actions(
                    next_obs, num_actions=self.action_samples
                )
                batch_size = next_obs.shape[0]
                next_obs_rep = next_obs.unsqueeze(1).expand(
                    batch_size, self.action_samples, next_obs.shape[-1]
                )
                next_obs_flat = next_obs_rep.reshape(-1, next_obs.shape[-1])
                next_act_flat = next_actions.reshape(-1, next_actions.shape[-1])
                q_target = self.q_target(next_obs_flat, next_act_flat)
                q_target = q_target.reshape(batch_size, self.action_samples).mean(
                    dim=1, keepdim=True
                )
                target = rewards + (1.0 - dones) * self.gamma * q_target  # LaTeX: y_t = r_t + \gamma(1-d_t)\bar{Q}(s_{t+1})

        if not self._assert_finite_tensors(
            {"obs": obs, "actions": actions, "target": target}
        ):
            return None

        q = self.q(obs, actions)
        q_loss = F.mse_loss(q, target)  # LaTeX: \mathcal{L}_Q = \mathbb{E}\!\left[(Q(s_t,a_t) - y_t)^2\right]
        if not self._assert_finite_tensors({"q": q, "q_loss": q_loss}):
            return None

        # Phase A: critic update
        self.q_opt.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(
            self.q.parameters(),
            self.max_grad_norm,
        )
        self.q_opt.step()

        # Phase B (Acme-style): update dual vars then do policy M-step.
        batch_size = obs.shape[0]
        num_samples = self.action_samples

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

        if not self._assert_finite_tensors({"q_vals": q_vals}):
            return None

        min_log = torch.tensor(-18.0, device=self.device)
        log_temp = torch.maximum(self.log_temperature, min_log)
        temperature = F.softplus(log_temp) + 1e-8  # LaTeX: \eta = \operatorname{softplus}(\tilde{\eta}) + \epsilon
        weights, loss_temperature = self._compute_weights_and_temperature_loss(
            q_vals, self.kl_epsilon, temperature
        )

        # KL(nonparametric || target) diagnostic (relative).
        kl_nonparametric = self._compute_nonparametric_kl_from_weights(weights)
        kl_q_rel = kl_nonparametric.mean() / float(self.kl_epsilon)  # LaTeX: \mathrm{KL}_{rel} = \frac{\mathbb{E}[D_{KL}(q\|u)]}{\epsilon}

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
        loss_policy_mean = -(weights * log_prob_fixed_stddev).sum(dim=1).mean()  # LaTeX: \mathcal{L}_{\pi,\mu} = -\mathbb{E}_b\sum_i q_{b,i}\log \pi_{\mu,\sigma'}(a_{b,i}|s_b)
        loss_policy_std = -(weights * log_prob_fixed_mean).sum(dim=1).mean()  # LaTeX: \mathcal{L}_{\pi,\sigma} = -\mathbb{E}_b\sum_i q_{b,i}\log \pi_{\mu',\sigma}(a_{b,i}|s_b)
        loss_policy = loss_policy_mean + loss_policy_std  # LaTeX: \mathcal{L}_{\pi} = \mathcal{L}_{\pi,\mu} + \mathcal{L}_{\pi,\sigma}

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

        mean_kl_mean = kl_mean.mean(dim=0)  # LaTeX: \bar{D}_{\mu,j} = \frac{1}{B}\sum_b D_{\mu,b,j}
        mean_kl_std = kl_std.mean(dim=0)  # LaTeX: \bar{D}_{\sigma,j} = \frac{1}{B}\sum_b D_{\sigma,b,j}

        log_alpha_mean = torch.maximum(self.log_alpha_mean, min_log)
        log_alpha_stddev = torch.maximum(self.log_alpha_stddev, min_log)

        alpha_mean = F.softplus(log_alpha_mean) + 1e-8  # LaTeX: \alpha_{\mu,j} = \operatorname{softplus}(\tilde{\alpha}_{\mu,j}) + \epsilon
        alpha_std = F.softplus(log_alpha_stddev) + 1e-8  # LaTeX: \alpha_{\sigma,j} = \operatorname{softplus}(\tilde{\alpha}_{\sigma,j}) + \epsilon

        loss_kl_mean = (alpha_mean.detach() * mean_kl_mean).sum()  # LaTeX: \mathcal{L}_{KL,\mu} = \sum_j \alpha_{\mu,j}\bar{D}_{\mu,j}
        loss_kl_std = (alpha_std.detach() * mean_kl_std).sum()  # LaTeX: \mathcal{L}_{KL,\sigma} = \sum_j \alpha_{\sigma,j}\bar{D}_{\sigma,j}
        loss_kl_penalty = loss_kl_mean + loss_kl_std  # LaTeX: \mathcal{L}_{KL} = \mathcal{L}_{KL,\mu} + \mathcal{L}_{KL,\sigma}

        loss_alpha_mean = (
            alpha_mean * (self.mstep_kl_epsilon - mean_kl_mean.detach())
        ).sum()  # LaTeX: \mathcal{L}_{\alpha_{\mu}} = \sum_j \alpha_{\mu,j}(\epsilon_{KL} - \bar{D}_{\mu,j})
        loss_alpha_std = (
            alpha_std * (self.mstep_kl_epsilon - mean_kl_std.detach())
        ).sum()  # LaTeX: \mathcal{L}_{\alpha_{\sigma}} = \sum_j \alpha_{\sigma,j}(\epsilon_{KL} - \bar{D}_{\sigma,j})

        # Update dual variables (temperature + alphas).
        dual_loss = loss_temperature + loss_alpha_mean + loss_alpha_std  # LaTeX: \mathcal{L}_{dual} = \mathcal{L}_{\eta} + \mathcal{L}_{\alpha_{\mu}} + \mathcal{L}_{\alpha_{\sigma}}
        self.dual_opt.zero_grad()
        dual_loss.backward()
        nn.utils.clip_grad_norm_(
            [
                p
                for p in [
                    self.log_temperature,
                    self.log_alpha_mean,
                    self.log_alpha_stddev,
                ]
                if p is not None
            ],
            self.max_grad_norm,
        )
        self.dual_opt.step()

        # ---------- after dual_opt.step() ----------
        # Recompute the post-update dual multipliers and freeze fixed E-step tensors.
        with torch.no_grad():
            mean_target_det = mean_target.detach()
            log_std_target_det = log_std_target.detach()
            std_target_det = std_target.detach()
            weights_det = weights.detach()  # (B,N)
            actions_det = actions.detach()  # (B,N,D)

        # Recompute dual multipliers AFTER dual_opt.step()
        alpha_mean_det = (F.softplus(self.log_alpha_mean) + 1e-8).detach()  # LaTeX: \alpha_{\mu,j}' = \operatorname{softplus}(\tilde{\alpha}_{\mu,j}) + \epsilon
        alpha_std_det = (F.softplus(self.log_alpha_stddev) + 1e-8).detach()  # LaTeX: \alpha_{\sigma,j}' = \operatorname{softplus}(\tilde{\alpha}_{\sigma,j}) + \epsilon

        # Parameter delta diagnostic: snapshot BEFORE M-step
        with torch.no_grad():
            params_before = (
                nn.utils.parameters_to_vector(self.policy.parameters()).detach().clone()
            )

        # Inner M-step (recompute online outputs each iteration)
        for _ in range(int(self.m_steps)):
            mean_online, log_std_online = self.policy(obs)  # (B,D), (B,D)
            std_online = torch.exp(log_std_online)

            # expand shapes for (B,N,D)
            mean_online_exp = mean_online.unsqueeze(1)
            std_online_exp = std_online.unsqueeze(1)
            mean_target_exp = mean_target_det.unsqueeze(1)
            std_target_exp = std_target_det.unsqueeze(1)

            # cross-entropy pieces
            log_prob_fixed_stddev = (
                Normal(mean_online_exp, std_target_exp)
                .log_prob(actions_det)
                .sum(dim=-1)
            )
            log_prob_fixed_mean = (
                Normal(mean_target_exp, std_online_exp)
                .log_prob(actions_det)
                .sum(dim=-1)
            )

            loss_policy_mean = -(weights_det * log_prob_fixed_stddev).sum(dim=1).mean()
            loss_policy_std = -(weights_det * log_prob_fixed_mean).sum(dim=1).mean()
            loss_policy = loss_policy_mean + loss_policy_std

            # KL penalties (compare online -> frozen target)
            kl_mean = self._kl_diag_gaussian_per_dim(
                mean_target_det, log_std_target_det, mean_online, log_std_target_det
            )  # (B,D)
            kl_std = self._kl_diag_gaussian_per_dim(
                mean_target_det, log_std_target_det, mean_target_det, log_std_online
            )  # (B,D)
            mean_kl_mean = kl_mean.mean(dim=0)  # (D,)
            mean_kl_std = kl_std.mean(dim=0)  # (D,)

            loss_kl_mean = (alpha_mean_det * mean_kl_mean).sum()
            loss_kl_std = (alpha_std_det * mean_kl_std).sum()
            loss_kl_penalty = loss_kl_mean + loss_kl_std

            policy_total_loss = loss_policy + loss_kl_penalty  # LaTeX: \mathcal{L}_{\pi}^{total} = \mathcal{L}_{\pi} + \mathcal{L}_{KL}

            self.policy_opt.zero_grad()
            policy_total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy_opt.step()

        # snapshot after M-step
        with torch.no_grad():
            params_after = (
                nn.utils.parameters_to_vector(self.policy.parameters()).detach().clone()
            )
            param_delta = torch.norm(params_after - params_before)

        # Increment step counter for target-sync cadence bookkeeping.
        self._num_steps += 1

        # Diagnostics for training monitoring.
        temperature_val = float(
            (F.softplus(self.log_temperature) + 1e-8).detach().item()
        )

        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()  # LaTeX: \mathcal{H}(q) = -\mathbb{E}_b\sum_i q_{b,i}\log q_{b,i}

        return {
            "train/param_delta": float(param_delta.item()),
            "loss/q": float(q_loss.item()),
            "loss/policy": float(loss_policy.item()),
            "loss/dual_eta": float(loss_temperature.detach().item()),
            "loss/dual": float(dual_loss.detach().item()),
            "kl/q_pi": float(kl_q_rel.detach().item()),
            "kl/mean": float(mean_kl_mean.mean().detach().item()),
            "kl/std": float(mean_kl_std.mean().detach().item()),
            "eta": temperature_val,
            "lambda": float(
                (F.softplus(self.log_alpha_mean) + 1e-8).mean().detach().item()
            ),
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
            "entropy": float(entropy.detach().item()),
        }
```
