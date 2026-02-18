# `trainers.ppo.trainer`

_Source: `trainers/ppo/trainer.py`_

## Functions

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

### `_resolve_env_id`

```python
def _resolve_env_id(env_id: str) -> str
```

_No docstring provided._

### `make_env`

```python
def make_env(gym_id: str, seed: int, normalize_observation: bool = True)
```

_No docstring provided._

### `make_eval_env`

```python
def make_eval_env(gym_id: str, seed: int, normalize_observation: bool = True)
```

_No docstring provided._

### `find_wrapper`

```python
def find_wrapper(env, wrapper_type)
```

_No docstring provided._

### `_sync_obs_rms_to_eval_envs`

```python
def _sync_obs_rms_to_eval_envs(train_envs: gym.vector.VectorEnv, eval_envs: gym.vector.VectorEnv)
```

Copy obs RMS stats from the first training env to all eval envs.

### `_evaluate_vectorized`

```python
def _evaluate_vectorized(agent: 'Agent', eval_envs: gym.vector.VectorEnv, device: torch.device, seed: int = 42) -> tuple[np.ndarray, np.ndarray]
```

Vectorized evaluation: runs all episodes in parallel across eval_envs.

### `log_episode_stats`

```python
def log_episode_stats(infos, global_step: int)
```

_No docstring provided._

### `layer_init`

```python
def layer_init(layer, std = np.sqrt(2), bias_const = 0.0)
```

_No docstring provided._

## Classes

### `Agent`

Base classes: `nn.Module`

_No docstring provided._

#### Methods in `Agent`

##### `__init__`

```python
def __init__(self, obs_dim: int, act_dim: int, policy_layer_sizes: Tuple[int, ...], value_layer_sizes: Tuple[int, ...])
```

_No docstring provided._

##### `_build_mlp`

```python
def _build_mlp(input_dim: int, hidden_layer_sizes: Tuple[int, ...], output_dim: int, output_std: float) -> nn.Sequential
```

_No docstring provided._

##### `get_value`

```python
def get_value(self, x)
```

_No docstring provided._

##### `get_action_and_value`

```python
def get_action_and_value(self, x, action = None)
```

_No docstring provided._

### `PPOTrainer`

_No docstring provided._

#### Methods in `PPOTrainer`

##### `__init__`

```python
def __init__(self, env_id: str, seed: int, device: torch.device, policy_layer_sizes: Tuple[int, ...], critic_layer_sizes: Tuple[int, ...], rollout_steps: int, gamma: float = 0.99, gae_lambda: float = 0.95, update_epochs: int = 10, minibatch_size: int = 64, policy_lr: float = 0.0003, clip_ratio: float = 0.2, ent_coef: float = 0.0, vf_coef: float = 0.5, max_grad_norm: float = 0.5, target_kl: float = 0.02, norm_adv: bool = True, clip_vloss: bool = True, anneal_lr: bool = True, normalize_obs: bool = True, num_envs: int = 1, optimizer_type: str = 'adam', sgd_momentum: float = 0.9)
```

_No docstring provided._

##### `_build_optimizer`

```python
def _build_optimizer(self) -> torch.optim.Optimizer
```

_No docstring provided._

##### `train`

```python
def train(self, total_steps: int, out_dir: str)
```

_No docstring provided._
