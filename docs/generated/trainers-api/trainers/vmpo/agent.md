# `trainers.vmpo.agent`

_Source: `minerva/trainers/vmpo/agent.py`_

## Functions

### `layer_init`

```python
def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias_const: float = 0.0)
```

_No docstring provided._

## Classes

### `DiagonalGaussianPolicy`

Base classes: `nn.Module`

_No docstring provided._

#### Methods in `DiagonalGaussianPolicy`

##### `__init__`

```python
def __init__(self, obs_dim: int, act_dim: int, policy_layer_sizes: Tuple[int, ...] = (256, 256), value_layer_sizes: Tuple[int, ...] = (256, 256), shared_encoder: bool = False)
```

_No docstring provided._

##### `_build_mlp`

```python
def _build_mlp(input_dim: int, hidden_layer_sizes: Tuple[int, ...]) -> nn.Sequential
```

_No docstring provided._

##### `policy_logstd_parameters`

```python
def policy_logstd_parameters(self) -> list[nn.Parameter]
```

_No docstring provided._

##### `_mean_and_log_std`

```python
def _mean_and_log_std(self, encoded_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

_No docstring provided._

##### `get_policy_dist_params`

```python
def get_policy_dist_params(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

Internal helper to get raw distribution parameters.

##### `forward`

```python
def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

_No docstring provided._

##### `get_value`

```python
def get_value(self, obs: torch.Tensor) -> torch.Tensor
```

_No docstring provided._

##### `forward_all`

```python
def forward_all(self, obs)
```

_No docstring provided._

##### `log_prob`

```python
def log_prob(self, mean: torch.Tensor, log_std: torch.Tensor, actions: torch.Tensor) -> torch.Tensor
```

_No docstring provided._

##### `sample_action`

```python
def sample_action(self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]
```

_No docstring provided._

### `VMPOAgent`

_No docstring provided._

#### Methods in `VMPOAgent`

##### `__init__`

```python
def __init__(self, obs_dim: int, act_dim: int, device: torch.device, policy_layer_sizes: Tuple[int, ...] = (256, 256), value_layer_sizes: Tuple[int, ...] = (256, 256), normalize_advantages: bool = True, gamma: float = 0.99, advantage_estimator: Literal['returns', 'dae', 'gae'] = 'returns', gae_lambda: float = 0.95, policy_lr: float = 0.0005, value_lr: float = 0.001, topk_fraction: float = 0.5, temperature_init: float = 1.0, temperature_lr: float = 0.0001, epsilon_eta: float = 0.1, epsilon_mu: float = 0.01, epsilon_sigma: float = 0.01, alpha_lr: float = 0.0001, max_grad_norm: float = 10.0, optimizer_type: str = 'adam', sgd_momentum: float = 0.9, shared_encoder: bool = False, m_steps: int = 1)
```

_No docstring provided._

##### `_build_optimizer`

```python
def _build_optimizer(self, params, lr: float | None = None) -> torch.optim.Optimizer
```

_No docstring provided._

##### `act`

```python
def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]
```

_No docstring provided._

##### `value`

```python
def value(self, obs: np.ndarray) -> float | np.ndarray
```

_No docstring provided._

##### `update`

```python
def update(self, batch: Dict[str, Any]) -> Dict[str, float]
```

_No docstring provided._
