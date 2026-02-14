# `trainers.ppo_trxl.pom_env`

_Source: `trainers/ppo_trxl/pom_env.py`_

## Functions

### `register_pom_env`

```python
def register_pom_env() -> None
```

_No docstring provided._

## Classes

### `PoMEnv`

Base classes: `gym.Env`

Proof-of-Concept memory environment used in CleanRL's TRxL example.

#### Methods in `PoMEnv`

##### `__init__`

```python
def __init__(self, render_mode: str | None = None)
```

_No docstring provided._

##### `step`

```python
def step(self, action: int)
```

_No docstring provided._

##### `reset`

```python
def reset(self, *, seed: int | None = None, options = None)
```

_No docstring provided._

##### `render`

```python
def render(self)
```

_No docstring provided._

##### `close`

```python
def close(self)
```

_No docstring provided._
