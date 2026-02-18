# `trainers.mpo.trainer`

_Source: `trainers/mpo/trainer.py`_

## Functions

### `_format_metrics`

```python
def _format_metrics(metrics: Mapping[str, float]) -> str
```

_No docstring provided._

### `_make_env`

```python
def _make_env(env_id: str, *, seed: int | None = None) -> gym.Env
```

_No docstring provided._

### `_to_device_tensor`

```python
def _to_device_tensor(value: np.ndarray, device: torch.device) -> torch.Tensor
```

_No docstring provided._

### `_evaluate_vectorized`

```python
def _evaluate_vectorized(agent: MPOAgent, eval_envs: gym.vector.VectorEnv, seed: int = 42) -> Dict[str, float]
```

High-performance vectorized evaluation.
Runs all n_episodes in parallel using a SyncVectorEnv.

## Classes

### `MPOTrainer`

_No docstring provided._

#### Methods in `MPOTrainer`

##### `__init__`

```python
def __init__(self, env_id: str, seed: int, device: torch.device, policy_layer_sizes: Tuple[int, ...], critic_layer_sizes: Tuple[int, ...], replay_size: int, gamma: float = 0.99, target_networks_update_period: int = 100, policy_lr: float = 0.0003, q_lr: float = 0.0003, kl_epsilon: float = 0.1, mstep_kl_epsilon: float = 0.1, temperature_init: float = 1.0, temperature_lr: float = 0.0003, lambda_init: float = 1.0, lambda_lr: float = 0.0003, epsilon_penalty: float = 0.001, max_grad_norm: float = 1.0, action_samples: int = 20, use_retrace: bool = False, retrace_steps: int = 2, retrace_mc_actions: int = 8, retrace_lambda: float = 0.95, optimizer_type: str = 'adam', sgd_momentum: float = 0.9, init_log_alpha_mean: float = 10.0, init_log_alpha_stddev: float = 1000.0)
```

_No docstring provided._

##### `replay_size`

```python
def replay_size(self) -> int
```

_No docstring provided._

##### `replay_ptr`

```python
def replay_ptr(self) -> int
```

_No docstring provided._

##### `_as_cpu_float_tensor`

```python
def _as_cpu_float_tensor(value: np.ndarray | float, shape: tuple[int, ...]) -> torch.Tensor
```

_No docstring provided._

##### `_is_finite_value`

```python
def _is_finite_value(value: np.ndarray | float) -> bool
```

_No docstring provided._

##### `_add_transition`

```python
def _add_transition(self, obs: np.ndarray, action_exec: np.ndarray, action_raw: np.ndarray, behaviour_logp: float, reward: float, next_obs: np.ndarray, done: float) -> bool
```

_No docstring provided._

##### `_get_sequence_offsets`

```python
def _get_sequence_offsets(self, seq_len: int) -> torch.Tensor
```

_No docstring provided._

##### `_sample_sequences`

```python
def _sample_sequences(self, batch_size: int, seq_len: int) -> TensorDict
```

_No docstring provided._

##### `train`

```python
def train(self, total_steps: int, update_after: int, batch_size: int, out_dir: str, m_steps: int = 1)
```

_No docstring provided._
