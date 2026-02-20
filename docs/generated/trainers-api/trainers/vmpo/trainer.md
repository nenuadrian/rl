# `trainers.vmpo.trainer`

_Source: `minerva/trainers/vmpo/trainer.py`_

## Functions

### `_format_metrics`

```python
def _format_metrics(metrics: Mapping[str, float]) -> str
```

_No docstring provided._

### `_iter_final_info_entries`

```python
def _iter_final_info_entries(infos) -> list[tuple[int, dict]]
```

Yield (env_index, final_info_dict) entries across possible final_info layouts.

### `_extract_final_observations`

```python
def _extract_final_observations(infos, num_envs: int) -> list[np.ndarray | None]
```

Extract per-env terminal observations from vector-env info dict.

### `_extract_episode_returns`

```python
def _extract_episode_returns(infos) -> list[tuple[int, float]]
```

Extract per-env episode returns from vector-env info dict.

### `_resolve_env_id`

```python
def _resolve_env_id(env_id: str) -> str
```

_No docstring provided._

### `_make_env`

```python
def _make_env(gym_id: str, seed: int)
```

_No docstring provided._

### `find_wrapper`

```python
def find_wrapper(env, wrapper_type)
```

_No docstring provided._

### `_evaluate_vectorized`

```python
def _evaluate_vectorized(agent: VMPOAgent, eval_envs: gym.vector.VectorEnv, seed: int = 42) -> Dict[str, float]
```

High-performance vectorized evaluation.
Runs all n_episodes in parallel using a SyncVectorEnv.

## Classes

### `VMPOTrainer`

_No docstring provided._

#### Methods in `VMPOTrainer`

##### `__init__`

```python
def __init__(self, env_id: str, seed: int, device: torch.device, policy_layer_sizes: Tuple[int, ...], value_layer_sizes: Tuple[int, ...], rollout_steps: int, normalize_advantages: bool = True, gamma: float = 0.99, advantage_estimator: str = 'returns', gae_lambda: float = 0.95, policy_lr: float = 0.0005, value_lr: float = 0.001, topk_fraction: float = 0.5, temperature_init: float = 1.0, temperature_lr: float = 0.0001, epsilon_eta: float = 0.1, epsilon_mu: float = 0.01, epsilon_sigma: float = 0.01, alpha_lr: float = 0.0001, max_grad_norm: float = 10.0, optimizer_type: str = 'adam', sgd_momentum: float = 0.9, num_envs: int = 1, shared_encoder: bool = False, m_steps: int = 1)
```

_No docstring provided._

##### `_reset_rollout`

```python
def _reset_rollout(self) -> None
```

_No docstring provided._

##### `_rollout_full`

```python
def _rollout_full(self) -> bool
```

_No docstring provided._

##### `train`

```python
def train(self, total_steps: int, out_dir: str)
```

_No docstring provided._
