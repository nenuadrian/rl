# `trainers.ppo_trxl.trainer`

_Source: `trainers/ppo_trxl/trainer.py`_

## Functions

### `_format_metrics`

```python
def _format_metrics(metrics: Mapping[str, float]) -> str
```

_No docstring provided._

### `layer_init`

```python
def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0)
```

_No docstring provided._

### `batched_index_select`

```python
def batched_index_select(input_tensor: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor
```

_No docstring provided._

## Classes

### `PositionalEncoding`

Base classes: `nn.Module`

_No docstring provided._

#### Methods in `PositionalEncoding`

##### `__init__`

```python
def __init__(self, dim: int, min_timescale: float = 2.0, max_timescale: float = 10000.0)
```

_No docstring provided._

##### `forward`

```python
def forward(self, seq_len: int) -> torch.Tensor
```

_No docstring provided._

### `MultiHeadAttention`

Base classes: `nn.Module`

_No docstring provided._

#### Methods in `MultiHeadAttention`

##### `__init__`

```python
def __init__(self, embed_dim: int, num_heads: int)
```

_No docstring provided._

##### `forward`

```python
def forward(self, values: torch.Tensor, keys: torch.Tensor, query: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]
```

_No docstring provided._

### `TransformerLayer`

Base classes: `nn.Module`

_No docstring provided._

#### Methods in `TransformerLayer`

##### `__init__`

```python
def __init__(self, dim: int, num_heads: int)
```

_No docstring provided._

##### `forward`

```python
def forward(self, value: torch.Tensor, key: torch.Tensor, query: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
```

_No docstring provided._

### `Transformer`

Base classes: `nn.Module`

_No docstring provided._

#### Methods in `Transformer`

##### `__init__`

```python
def __init__(self, num_layers: int, dim: int, num_heads: int, max_episode_steps: int, positional_encoding: str)
```

_No docstring provided._

##### `forward`

```python
def forward(self, x: torch.Tensor, memories: torch.Tensor, mask: torch.Tensor, memory_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
```

_No docstring provided._

### `TrXLAgent`

Base classes: `nn.Module`

_No docstring provided._

#### Methods in `TrXLAgent`

##### `__init__`

```python
def __init__(self, observation_space: gym.Space, action_space_shape: Sequence[int], *, trxl_num_layers: int, trxl_num_heads: int, trxl_dim: int, trxl_positional_encoding: str, max_episode_steps: int, reconstruction_coef: float)
```

_No docstring provided._

##### `_encode`

```python
def _encode(self, x: torch.Tensor) -> torch.Tensor
```

_No docstring provided._

##### `get_value`

```python
def get_value(self, x: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor, memory_indices: torch.Tensor) -> torch.Tensor
```

_No docstring provided._

##### `get_action_and_value`

```python
def get_action_and_value(self, x: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor, memory_indices: torch.Tensor, action: torch.Tensor | None = None, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

_No docstring provided._

##### `reconstruct_observation`

```python
def reconstruct_observation(self) -> torch.Tensor
```

_No docstring provided._

### `PPOTRxLTrainer`

_No docstring provided._

#### Methods in `PPOTRxLTrainer`

##### `__init__`

```python
def __init__(self, *, env_id: str, seed: int, device: torch.device, num_envs: int, num_steps: int, num_minibatches: int, update_epochs: int, init_lr: float, final_lr: float, anneal_steps: int, gamma: float, gae_lambda: float, norm_adv: bool, clip_coef: float, clip_vloss: bool, init_ent_coef: float, final_ent_coef: float, vf_coef: float, max_grad_norm: float, target_kl: float | None, trxl_num_layers: int, trxl_num_heads: int, trxl_dim: int, trxl_memory_length: int, trxl_positional_encoding: str, reconstruction_coef: float, capture_video: bool = False, run_name: str | None = None)
```

_No docstring provided._

##### `_resolve_max_episode_steps`

```python
def _resolve_max_episode_steps(self, env: gym.Env) -> int
```

_No docstring provided._

##### `_make_env`

```python
def _make_env(self, *, seed: int, env_index: int, record_video: bool) -> gym.Env
```

_No docstring provided._

##### `_build_memory_helpers`

```python
def _build_memory_helpers(*, memory_length: int, max_episode_steps: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]
```

_No docstring provided._

##### `train`

```python
def train(self, *, total_steps: int, eval_interval: int, save_interval: int, out_dir: str)
```

_No docstring provided._

##### `_close_env`

```python
def _close_env(env: gym.Env | None) -> None
```

_No docstring provided._

##### `_get_eval_env`

```python
def _get_eval_env(self, seed: int) -> gym.Env
```

_No docstring provided._

##### `evaluate`

```python
def evaluate(self, n_episodes: int = 10, seed: int = 42) -> Dict[str, float]
```

_No docstring provided._
