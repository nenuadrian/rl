# `trainers.mpo.replay_buffer`

_Source: `trainers/mpo/replay_buffer.py`_

## Classes

### `MPOReplayBuffer`

_No docstring provided._

#### Methods in `MPOReplayBuffer`

##### `__init__`

```python
def __init__(self, obs_dim: int, act_dim: int, capacity: int)
```

_No docstring provided._

##### `add`

```python
def add(self, obs: np.ndarray, action_exec: np.ndarray, action_raw: np.ndarray, behaviour_logp: float, reward: float, next_obs: np.ndarray, done: float) -> None
```

_No docstring provided._

##### `sample`

```python
def sample(self, batch_size: int) -> dict
```

_No docstring provided._

##### `sample_sequences`

```python
def sample_sequences(self, batch_size: int, seq_len: int) -> dict
```

_No docstring provided._
