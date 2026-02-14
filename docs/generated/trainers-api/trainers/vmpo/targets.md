# `trainers.vmpo.targets`

_Source: `trainers/vmpo/targets.py`_

## Functions

### `compute_returns_targets`

```python
def compute_returns_targets(rewards: np.ndarray, dones: np.ndarray, last_value: np.ndarray, gamma: float) -> np.ndarray
```

_No docstring provided._

### `compute_dae_targets`

```python
def compute_dae_targets(rewards: np.ndarray, dones: np.ndarray, values: np.ndarray, last_value: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]
```

_No docstring provided._

### `compute_gae_targets`

```python
def compute_gae_targets(rewards: np.ndarray, dones: np.ndarray, values: np.ndarray, last_value: np.ndarray, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]
```

_No docstring provided._

### `compute_rollout_targets`

```python
def compute_rollout_targets(rewards: np.ndarray, dones: np.ndarray, values: np.ndarray, last_value: np.ndarray, gamma: float, estimator: Literal['returns', 'dae', 'gae'], gae_lambda: float) -> tuple[np.ndarray, np.ndarray]
```

_No docstring provided._
