# `trainers.vmpo.trainer`

_Source: `trainers/vmpo/trainer.py`_

## Functions

### `_format_metrics`

```python
def _format_metrics(metrics: Mapping[str, float]) -> str
```

_No docstring provided._

### `_transform_observation`

```python
def _transform_observation(env: gym.Env, fn)
```

Gymnasium compatibility shim across wrapper signatures.

### `_transform_reward`

```python
def _transform_reward(env: gym.Env, fn)
```

Gymnasium compatibility shim across wrapper signatures.

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

### `_find_wrapper`

```python
def _find_wrapper(env: gym.Env, wrapper_type: type[gym.Wrapper])
```

Return the first wrapper of type `wrapper_type` in an env wrapper chain.

### `_merge_obs_rms_stats`

```python
def _merge_obs_rms_stats(stats_seq: Sequence[tuple[np.ndarray, np.ndarray, float]]) -> tuple[np.ndarray, np.ndarray, float] | None
```

Merge per-env running mean/var stats into a single aggregate RMS state.

### `_collect_vector_obs_rms_stats`

```python
def _collect_vector_obs_rms_stats(vec_env: gym.vector.VectorEnv) -> tuple[np.ndarray, np.ndarray, float] | None
```

Collect merged NormalizeObservation running stats from a vector env.

### `_apply_obs_rms_stats`

```python
def _apply_obs_rms_stats(vec_env: gym.vector.VectorEnv, obs_rms_stats: tuple[np.ndarray, np.ndarray, float] | None) -> None
```

Copy provided observation running stats into all eval envs.

### `_make_env`

```python
def _make_env(env_id: str, *, seed: int | None = None, gamma: float = 0.99, normalize_observation: bool = True, clip_observation: float | None = 10.0, normalize_reward: bool = True, clip_reward: float | None = 10.0) -> gym.Env
```

_No docstring provided._

### `_evaluate_vectorized`

```python
def _evaluate_vectorized(agent: VMPOAgent, eval_envs: gym.vector.VectorEnv, seed: int = 42, obs_rms_stats: tuple[np.ndarray, np.ndarray, float] | None = None) -> Dict[str, float]
```

High-performance vectorized evaluation.
Runs all n_episodes in parallel using a SyncVectorEnv.

## Classes

### `VMPOTrainer`

_No docstring provided._

#### Methods in `VMPOTrainer`

##### `__init__`

```python
def __init__(self, env_id: str, seed: int, device: torch.device, policy_layer_sizes: Tuple[int, ...], value_layer_sizes: Tuple[int, ...], rollout_steps: int, normalize_advantages: bool = True, gamma: float = 0.99, advantage_estimator: str = 'returns', gae_lambda: float = 0.95, policy_lr: float = 0.0005, value_lr: float = 0.001, topk_fraction: float = 0.5, temperature_init: float = 1.0, temperature_lr: float = 0.0001, epsilon_eta: float = 0.1, epsilon_mu: float = 0.01, epsilon_sigma: float = 0.01, alpha_lr: float = 0.0001, max_grad_norm: float = 10.0, optimizer_type: str = 'adam', sgd_momentum: float = 0.9, num_envs: int = 1)
```

_No docstring provided._

##### `_make_train_env`

```python
def _make_train_env(self, *, env_id: str, seed: int, env_index: int, gamma: float)
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
def train(self, total_steps: int, out_dir: str, updates_per_step: int = 1)
```

_No docstring provided._
