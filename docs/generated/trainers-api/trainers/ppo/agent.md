# `trainers.ppo.agent`

_Source: `trainers/ppo/agent.py`_

## Functions

### `_layer_init`

```python
def _layer_init(layer: nn.Module, std: float = math.sqrt(2.0), bias_const: float = 0.0) -> nn.Module
```

_No docstring provided._

## Classes

### `MLP`

Base classes: `nn.Module`

_No docstring provided._

#### Methods in `MLP`

##### `__init__`

```python
def __init__(self, in_dim: int, layer_sizes: Tuple[int, ...])
```

_No docstring provided._

##### `forward`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

_No docstring provided._

### `GaussianPolicy`

Base classes: `nn.Module`

_No docstring provided._

#### Methods in `GaussianPolicy`

##### `__init__`

```python
def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Tuple[int, ...] = (256, 256), log_std_bounds: Tuple[float, float] = (-20.0, 2.0), action_low: np.ndarray | None = None, action_high: np.ndarray | None = None)
```

_No docstring provided._

##### `forward`

```python
def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

_No docstring provided._

##### `_distribution`

```python
def _distribution(self, obs: torch.Tensor) -> Normal
```

_No docstring provided._

##### `sample`

```python
def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

_No docstring provided._

##### `log_prob`

```python
def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor
```

_No docstring provided._

##### `entropy`

```python
def entropy(self, obs: torch.Tensor) -> torch.Tensor
```

_No docstring provided._

##### `sample_action`

```python
def sample_action(self, obs: torch.Tensor, deterministic: bool = False, **kwargs) -> torch.Tensor
```

_No docstring provided._

### `ValueNetwork`

Base classes: `nn.Module`

_No docstring provided._

#### Methods in `ValueNetwork`

##### `__init__`

```python
def __init__(self, obs_dim: int, hidden_sizes: Tuple[int, ...])
```

_No docstring provided._

##### `forward`

```python
def forward(self, obs: torch.Tensor) -> torch.Tensor
```

_No docstring provided._

### `PPOAgent`

_No docstring provided._

#### Methods in `PPOAgent`

##### `__init__`

```python
def __init__(self, obs_dim: int, act_dim: int, action_low: np.ndarray, action_high: np.ndarray, device: torch.device, policy_layer_sizes: Tuple[int, ...], critic_layer_sizes: Tuple[int, ...], gamma: float = 0.99, gae_lambda: float = 0.95, clip_ratio: float = 0.2, policy_lr: float = 0.0003, value_lr: float = 0.001, ent_coef: float = 0.0, vf_coef: float = 0.5, max_grad_norm: float = 0.5, target_kl: float = 0.02, norm_adv: bool = True, clip_vloss: bool = True, anneal_lr: bool = True, optimizer_type: str = 'adam', sgd_momentum: float = 0.9)
```

_No docstring provided._

##### `_build_optimizer`

```python
def _build_optimizer(self, params, *, lr: float) -> torch.optim.Optimizer
```

_No docstring provided._

##### `set_hparams`

```python
def set_hparams(self, *, clip_ratio: float | None = None, policy_lr: float | None = None, value_lr: float | None = None, ent_coef: float | None = None, vf_coef: float | None = None, max_grad_norm: float | None = None, target_kl: float | None = None, norm_adv: bool | None = None, clip_vloss: bool | None = None, anneal_lr: bool | None = None) -> None
```

_No docstring provided._

##### `act`

```python
def act(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

_No docstring provided._

##### `update`

```python
def update(self, batch: dict) -> dict
```

_No docstring provided._
