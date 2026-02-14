# `trainers.trpo.trainer`

_Source: `trainers/trpo/trainer.py`_

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

### `_make_env`

```python
def _make_env(env_id: str, *, seed: int | None = None, render_mode: str | None = None, gamma: float = 0.99, normalize_observation: bool = True, clip_observation: float | None = 10.0, normalize_reward: bool = True, clip_reward: float | None = 10.0) -> gym.Env
```

_No docstring provided._

### `_evaluate_vectorized`

```python
def _evaluate_vectorized(agent: TRPOAgent, env_id: str, n_episodes: int = 10, seed: int = 42, gamma: float = 0.99, normalize_observation: bool = True) -> Dict[str, float]
```

_No docstring provided._

## Classes

### `TRPOTrainer`

_No docstring provided._

#### Methods in `TRPOTrainer`

##### `__init__`

```python
def __init__(self, env_id: str, seed: int, device: torch.device, policy_layer_sizes: Tuple[int, ...], critic_layer_sizes: Tuple[int, ...], rollout_steps: int, gamma: float = 0.99, gae_lambda: float = 0.95, target_kl: float = 0.01, cg_iters: int = 10, cg_damping: float = 0.1, backtrack_coeff: float = 0.8, backtrack_iters: int = 10, value_lr: float = 0.0003, value_epochs: int = 10, value_minibatch_size: int = 256, max_grad_norm: float = 0.5, normalize_advantages: bool = True, optimizer_type: str = 'adam', sgd_momentum: float = 0.9, normalize_obs: bool = False, num_envs: int = 1, capture_video: bool = False, run_name: str | None = None)
```

_No docstring provided._

##### `_make_train_env`

```python
def _make_train_env(self, *, env_id: str, seed: int, env_index: int, gamma: float)
```

_No docstring provided._

##### `train`

```python
def train(self, total_steps: int, eval_interval: int, save_interval: int, out_dir: str)
```

_No docstring provided._
