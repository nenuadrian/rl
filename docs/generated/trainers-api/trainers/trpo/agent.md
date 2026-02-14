# `trainers.trpo.agent`

_Source: `trainers/trpo/agent.py`_

## Functions

### `_flat_params`

```python
def _flat_params(module: nn.Module) -> torch.Tensor
```

_No docstring provided._

### `_set_flat_params`

```python
def _set_flat_params(module: nn.Module, flat_params: torch.Tensor) -> None
```

_No docstring provided._

### `_flat_grads`

```python
def _flat_grads(grads: tuple[torch.Tensor | None, ...], params: tuple[nn.Parameter, ...]) -> torch.Tensor
```

_No docstring provided._

### `_conjugate_gradient`

```python
def _conjugate_gradient(Avp, b: torch.Tensor, nsteps: int, residual_tol: float = 1e-10) -> torch.Tensor
```

_No docstring provided._

### `_kl_old_to_new`

```python
def _kl_old_to_new(old_mean: torch.Tensor, old_log_std: torch.Tensor, new_mean: torch.Tensor, new_log_std: torch.Tensor) -> torch.Tensor
```

_No docstring provided._

## Classes

### `TRPOAgent`

_No docstring provided._

#### Methods in `TRPOAgent`

##### `__init__`

```python
def __init__(self, obs_dim: int, act_dim: int, action_low: np.ndarray, action_high: np.ndarray, device: torch.device, policy_layer_sizes: Tuple[int, ...], critic_layer_sizes: Tuple[int, ...], gamma: float = 0.99, gae_lambda: float = 0.95, target_kl: float = 0.01, cg_iters: int = 10, cg_damping: float = 0.1, backtrack_coeff: float = 0.8, backtrack_iters: int = 10, value_lr: float = 0.0003, value_epochs: int = 10, value_minibatch_size: int = 256, max_grad_norm: float = 0.5, normalize_advantages: bool = True, optimizer_type: str = 'adam', sgd_momentum: float = 0.9)
```

_No docstring provided._

##### `_build_value_optimizer`

```python
def _build_value_optimizer(self, params) -> torch.optim.Optimizer
```

_No docstring provided._

##### `act`

```python
def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

_No docstring provided._

##### `_policy_objective`

```python
def _policy_objective(self, obs: torch.Tensor, actions: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

_No docstring provided._

##### `update`

```python
def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]
```

_No docstring provided._
