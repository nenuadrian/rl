# `trainers.vmpo.agent`

_Source: `minerva/trainers/vmpo/agent.py`_

## Classes

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
