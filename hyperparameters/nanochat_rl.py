from __future__ import annotations

from typing import Any


PRESETS: dict[tuple[str, str], dict[str, Any]] = {
    ("test", "test"): {
    },
}


def get(domain: str, task: str) -> dict[str, Any]:
    key = (domain, task)
    if key not in PRESETS:
        available = ", ".join([f"{d}/{t}" for (d, t) in sorted(PRESETS.keys())])
        raise KeyError(f"No PPO preset for {domain}/{task}. Available: {available}")
    return dict(PRESETS[key])
