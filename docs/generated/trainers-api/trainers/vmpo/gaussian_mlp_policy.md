# `trainers.vmpo.gaussian_mlp_policy`

_Source: `minerva/trainers/vmpo/gaussian_mlp_policy.py`_

## Functions

### `_layer_init`

```python
def _layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias_const: float = 0.0)
```

_No docstring provided._

## Classes

### `MPOEncoder`

Base classes: `nn.Module`

Encoder for V-MPO: orthogonal Linear + Tanh at each hidden layer.

#### Methods in `MPOEncoder`

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

### `SquashedGaussianPolicy`

Base classes: `nn.Module`

_No docstring provided._

#### Methods in `SquashedGaussianPolicy`

##### `__init__`

```python
def __init__(self, obs_dim: int, act_dim: int, policy_layer_sizes: Tuple[int, ...] = (256, 256), value_layer_sizes: Tuple[int, ...] = (256, 256), shared_encoder: bool = False)
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
