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
def _make_env(env_id: str, *, seed: int | None = None, render_mode: str | None = None) -> gym.Env
```

_No docstring provided._

### `_evaluate_vectorized`

```python
def _evaluate_vectorized(agent: MPOAgent, env_id: str, n_episodes: int = 10, seed: int = 42) -> Dict[str, float]
```

High-performance vectorized evaluation.
Runs all n_episodes in parallel using a SyncVectorEnv.

## Classes

### `MPOTrainer`

_No docstring provided._

#### Methods in `MPOTrainer`

##### `__init__`

```python
def __init__(self, env_id: str, seed: int, device: torch.device, policy_layer_sizes: Tuple[int, ...], critic_layer_sizes: Tuple[int, ...], replay_size: int, gamma: float = 0.99, target_policy_update_period: int = 100, target_critic_update_period: int = 100, policy_lr: float = 0.0003, q_lr: float = 0.0003, kl_epsilon: float = 0.1, mstep_kl_epsilon: float = 0.1, per_dim_constraining: bool = True, temperature_init: float = 1.0, temperature_lr: float = 0.0003, lambda_init: float = 1.0, lambda_lr: float = 0.0003, action_penalization: bool = False, epsilon_penalty: float = 0.001, max_grad_norm: float = 1.0, action_samples: int = 20, use_retrace: bool = False, retrace_steps: int = 2, retrace_mc_actions: int = 8, retrace_lambda: float = 0.95, optimizer_type: str = 'adam', sgd_momentum: float = 0.9, capture_video: bool = False, run_name: str | None = None)
```

_No docstring provided._

##### `train`

```python
def train(self, total_steps: int, update_after: int, batch_size: int, eval_interval: int, save_interval: int, out_dir: str, updates_per_step: int = 1)
```

_No docstring provided._
