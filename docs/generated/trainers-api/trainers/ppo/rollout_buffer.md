# `trainers.ppo.rollout_buffer`

_Source: `trainers/ppo/rollout_buffer.py`_

## Classes

### `RolloutBuffer`

_No docstring provided._

#### Methods in `RolloutBuffer`

##### `create`

```python
def create(cls, obs_dim: int, act_dim: int, rollout_steps: int, num_envs: int) -> 'RolloutBuffer'
```

_No docstring provided._

##### `reset`

```python
def reset(self) -> None
```

_No docstring provided._

##### `add`

```python
def add(self, t: int, env_i: int, obs: np.ndarray, action: np.ndarray, reward: float, done: float, value: float, log_prob: float) -> None
```

_No docstring provided._

##### `compute_returns_advantages`

```python
def compute_returns_advantages(self, last_values: np.ndarray, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

_No docstring provided._

##### `minibatches`

```python
def minibatches(self, batch_size: int) -> Iterator[np.ndarray]
```

_No docstring provided._
