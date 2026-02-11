from __future__ import annotations

from typing import Any, Mapping


def merge_preset(
    *sources: Mapping[str, Any],
    **overrides: Any,
) -> dict[str, Any]:
    preset: dict[str, Any] = {}
    for source in sources:
        preset.update(source)
    preset.update(overrides)
    return preset


def get_preset(
    *,
    env_id: str,
    presets: Mapping[str, Mapping[str, Any]],
    algorithm_name: str,
    defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if env_id not in presets:
        available = ", ".join(sorted(presets.keys()))
        raise KeyError(f"No {algorithm_name} preset for {env_id}. Available: {available}")

    preset = dict(presets[env_id])
    if defaults:
        for key, value in defaults.items():
            preset.setdefault(key, value)
    return preset
