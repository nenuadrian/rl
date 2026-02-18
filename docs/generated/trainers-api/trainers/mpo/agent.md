# `trainers.mpo.agent`

_Source: `trainers/mpo/agent.py`_

## Classes

### `LayerNormMLP`

Base classes: `nn.Module`

_No docstring provided._

#### Methods in `LayerNormMLP`

##### `__init__`

```python
def __init__(self, in_dim: int, layer_sizes: Tuple[int, ...], activate_final: bool = False)
```

_No docstring provided._

##### `forward`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

_No docstring provided._

### `SmallInitLinear`

Base classes: `nn.Linear`

_No docstring provided._

#### Methods in `SmallInitLinear`

##### `__init__`

```python
def __init__(self, in_features: int, out_features: int, std: float = 0.01)
```

_No docstring provided._

### `Critic`

Base classes: `nn.Module`

_No docstring provided._

#### Methods in `Critic`

##### `__init__`

```python
def __init__(self, obs_dim: int, act_dim: int, layer_sizes: Tuple[int, ...], action_low: np.ndarray, action_high: np.ndarray)
```

_No docstring provided._

##### `forward`

```python
def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor
```

_No docstring provided._

### `DiagonalGaussianPolicy`

Base classes: `nn.Module`

_No docstring provided._

#### Methods in `DiagonalGaussianPolicy`

##### `__init__`

```python
def __init__(self, obs_dim: int, act_dim: int, layer_sizes: Tuple[int, ...], action_low: np.ndarray | None = None, action_high: np.ndarray | None = None)
```

_No docstring provided._

##### `forward`

```python
def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

_No docstring provided._

##### `forward_with_features`

```python
def forward_with_features(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

_No docstring provided._

##### `log_prob`

```python
def log_prob(self, mean: torch.Tensor, log_std: torch.Tensor, actions_raw: torch.Tensor) -> torch.Tensor
```

_No docstring provided._

##### `_clip_to_env_bounds`

```python
def _clip_to_env_bounds(self, actions_raw: torch.Tensor) -> torch.Tensor
```

_No docstring provided._

##### `sample_action_raw_and_exec`

```python
def sample_action_raw_and_exec(self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool, **kwargs) -> tuple[torch.Tensor, torch.Tensor]
```

_No docstring provided._

##### `sample_action`

```python
def sample_action(self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool, **kwargs) -> torch.Tensor
```

_No docstring provided._

##### `sample_actions_raw_and_exec`

```python
def sample_actions_raw_and_exec(self, obs: torch.Tensor, num_actions: int) -> tuple[torch.Tensor, torch.Tensor]
```

_No docstring provided._

##### `sample_actions`

```python
def sample_actions(self, obs: torch.Tensor, num_actions: int) -> torch.Tensor
```

_No docstring provided._

##### `head_parameters`

```python
def head_parameters(self)
```

_No docstring provided._

### `MPOAgent`

_No docstring provided._

#### Methods in `MPOAgent`

##### `__init__`

```python
def __init__(self, obs_dim: int, act_dim: int, action_low: np.ndarray, action_high: np.ndarray, device: torch.device, policy_layer_sizes: Tuple[int, ...], critic_layer_sizes: Tuple[int, ...], gamma: float = 0.99, target_networks_update_period: int = 100, policy_lr: float = 0.0003, q_lr: float = 0.0003, kl_epsilon: float = 0.1, mstep_kl_epsilon: float = 0.1, temperature_init: float = 1.0, temperature_lr: float = 0.0003, lambda_init: float = 1.0, lambda_lr: float = 0.0003, epsilon_penalty: float = 0.001, max_grad_norm: float = 1.0, action_samples: int = 20, use_retrace: bool = False, retrace_steps: int = 2, retrace_mc_actions: int = 8, retrace_lambda: float = 0.95, optimizer_type: str = 'adam', sgd_momentum: float = 0.9, init_log_alpha_mean: float = 10.0, init_log_alpha_stddev: float = 1000.0, m_steps: int = 1)
```

_No docstring provided._

##### `_build_optimizer`

```python
def _build_optimizer(self, params, lr: float | None = None) -> torch.optim.Optimizer
```

_No docstring provided._

##### `_kl_diag_gaussian_per_dim`

```python
def _kl_diag_gaussian_per_dim(self, mean_p, log_std_p, mean_q, log_std_q)
```

_No docstring provided._

##### `_compute_weights_and_temperature_loss`

```python
def _compute_weights_and_temperature_loss(self, q_values: torch.Tensor, epsilon: float, temperature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
```

Acme-style E-step weights and dual temperature loss.

q_values shape (B,N). Returns weights (B,N) detached and loss scalar.

##### `_compute_nonparametric_kl_from_weights`

```python
def _compute_nonparametric_kl_from_weights(self, weights: torch.Tensor) -> torch.Tensor
```

Estimates KL(nonparametric || target) like Acme's diagnostics.

weights shape (B,N). Returns (B,) KL.

##### `_to_device_tensor`

```python
def _to_device_tensor(self, value: np.ndarray | torch.Tensor) -> torch.Tensor
```

Fast path for float32 host arrays -> device tensors.

##### `_assert_finite_tensors`

```python
def _assert_finite_tensors(self, tensors: dict[str, torch.Tensor]) -> bool
```

_No docstring provided._

##### `act_with_logp`

```python
def act_with_logp(self, obs: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, np.ndarray, float]
```

_No docstring provided._

##### `act`

```python
def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray
```

_No docstring provided._

##### `_expected_q_current`

```python
def _expected_q_current(self, obs: torch.Tensor) -> torch.Tensor
```

_No docstring provided._

##### `_retrace_q_target`

```python
def _retrace_q_target(self, batch: dict) -> torch.Tensor
```

_No docstring provided._

##### `update`

```python
def update(self, batch: dict) -> dict | None
```

_No docstring provided._
