# `hyperparameters/__init__.py`

```python
"""Domain/task-specific hyperparameter presets.

Each algorithm module exposes `get(domain, task)` returning a dict of preset values.
Presets use *unprefixed* keys (e.g., `rollout_steps`, `policy_lr`). `main.py` is
responsible for mapping these keys to the underlying CLI args.
"""

from __future__ import annotations
```
