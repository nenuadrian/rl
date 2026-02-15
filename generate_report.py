from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import time
import torch
from utils.video import VideoRenderConfig, render_policy_video


if "MPLCONFIGDIR" not in os.environ:
    _mpl_cache_dir = Path(".mplconfig")
    _mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_cache_dir.resolve())
if "XDG_CACHE_HOME" not in os.environ:
    _xdg_cache_dir = Path(".cache")
    _xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(_xdg_cache_dir.resolve())

import matplotlib
import numpy as np
import wandb
from matplotlib.ticker import AutoMinorLocator

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_PROJECT = "minerva-rl-benchmark-6"
DEFAULT_ENTITY = "adrian-research"
DEFAULT_MIN_STEPS = 1
DEFAULT_STEP_KEY = "_step"
DEFAULT_METRIC_KEY = "eval/return_mean"
DEFAULT_POINTS = 200
DEFAULT_OUTPUT_DIR = "reports"
DEFAULT_HISTORY_SAMPLES = 600
DEFAULT_WORKERS = 8
DEFAULT_CACHE_DIR = ".report_cache"
DEFAULT_CHECKPOINTS_DIR = "checkpoints"
DEFAULT_VIDEOS_DIR = "videos"
DEFAULT_VIDEO_MAX_STEPS = 1000
DEFAULT_VIDEO_FPS = 30
DEFAULT_VIDEO_ATTEMPTS = 10
CACHE_VERSION = "v1"
_KNOWN_ALGOS_FOR_NAME_PARSE = ("ppo", "vmpo", "mpo", "vmpo-gtrxl", "r2d2-gtrxl", "ppo-gtrxl")
_STEP_KEY_ALIASES = (
    "_step",
    "evaluator_step",
)
_METRIC_KEY_ALIAS_GROUPS = (
    (
        "eval/return_mean",
        "eval/reward_mean",
        "evaluator_step/reward_mean",
        "evaluator_step/return_mean",
        "eval/episodic_return_mean",
        "eval_return_mean",
    ),
    (
        "eval/return_max",
        "eval/reward_max",
        "evaluator_step/reward_max",
        "evaluator_step/return_max",
        "eval/episodic_return_max",
        "eval_return_max",
    ),
)


@dataclass
class RunRecord:
    run: Any
    run_id: str
    name: str
    url: str
    step: float
    algorithm: str
    env: str
    env_id: str
    optimizer: str
    adv_type: str


@dataclass
class BestRunVideoResult:
    env: str
    algorithm: str
    run_name: str | None
    run_url: str | None
    metric_value: float | None
    checkpoint_path: str | None
    video_path: str | None
    gif_path: str | None
    status: str
    detail: str


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _as_int(value: Any) -> int | None:
    out = _as_float(value)
    if out is None:
        return None
    return int(out)


def _ordered_unique(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return tuple(out)


def _step_key_candidates(step_key: str) -> tuple[str, ...]:
    requested = str(step_key).strip()
    values = [requested]
    for alias in _STEP_KEY_ALIASES:
        values.append(alias)
    return _ordered_unique(values)


def _metric_key_candidates(metric_key: str) -> tuple[str, ...]:
    requested = str(metric_key).strip()
    values = [requested]
    for group in _METRIC_KEY_ALIAS_GROUPS:
        if requested in group:
            values.extend(group)
            break
    return _ordered_unique(values)


def _first_float(mapping: dict[str, Any], *, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _as_float(mapping.get(key))
        if value is not None:
            return value
    return None


def _extract_first_config_value(
    config: dict[str, Any], keys: tuple[str, ...]
) -> str | None:
    for key in keys:
        value = config.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _normalize_env_fallback(env_text: str) -> str:
    text = (env_text or "").strip("-").strip()
    if not text:
        return text
    if text.startswith("dm_control-"):
        parts = text.split("-")
        if len(parts) >= 3:
            domain = parts[1]
            task = "-".join(parts[2:])
            return f"dm_control/{domain}/{task}"
    return text


def _default_optimizer_for_algorithm(algorithm: str | None) -> str:
    algo = (algorithm or "").strip().lower()
    if algo in {"ppo", "vmpo", "mpo"}:
        return "adam"
    return "unknown"


def _default_adv_type_for_algorithm(algorithm: str | None) -> str:
    algo = (algorithm or "").strip().lower()
    if algo == "vmpo":
        return "returns"
    if algo in {"ppo"}:
        return "gae"
    if algo == "mpo":
        return "none"
    return "unknown"


def _normalize_meta_value(value: str | None, *, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip().lower()
    if not text or text == "none":
        return fallback
    return text


def _make_env_key(env_id: str, optimizer: str, adv_type: str) -> str:
    return f"{env_id} || opt={optimizer} || adv={adv_type}"


def _split_env_key(env_key: str) -> tuple[str, str, str]:
    env = (env_key or "").strip()
    marker_opt = " || opt="
    marker_adv = " || adv="
    if marker_opt in env and marker_adv in env:
        env_id, rest = env.split(marker_opt, 1)
        optimizer, adv_type = rest.split(marker_adv, 1)
        return env_id.strip(), optimizer.strip(), adv_type.strip()
    return env, "unknown", "unknown"


def _parse_run_name(
    run_name: str,
) -> tuple[str | None, str | None, str | None, str | None]:
    name = (run_name or "").strip()
    if not name:
        return None, None, None, None

    prefix, sep, suffix = name.rpartition("_")
    if sep and re.fullmatch(r"\d{8}-\d{6}", suffix):
        algorithm = None
        body = None
        for algo in _KNOWN_ALGOS_FOR_NAME_PARSE:
            algo_prefix = f"{algo}_"
            if prefix.startswith(algo_prefix):
                algorithm = algo
                body = prefix[len(algo_prefix) :]
                break
        if algorithm is None and "_" in prefix:
            algorithm, body = prefix.split("_", 1)

        if algorithm is not None and body is not None:
            algorithm = algorithm.strip()
            body = body.strip("-")
            if not algorithm:
                return None, None, None, None
            parts = body.rsplit("-", 2)
            if len(parts) == 3:
                env_part, optimizer, adv_type = parts
                env = _normalize_env_fallback(env_part)
                return (
                    algorithm,
                    env if env else None,
                    optimizer.strip() or None,
                    adv_type.strip() or None,
                )
            env = _normalize_env_fallback(body)
            return algorithm, env if env else None, None, None

    if "-" not in name:
        return None, None, None, None
    algorithm, env_part = name.split("-", 1)
    if not algorithm:
        return None, None, None, None
    env = re.sub(r"-seed\d+$", "", env_part).strip("-")
    env = _normalize_env_fallback(env)
    return algorithm, env if env else None, None, None


def _extract_run_axes(
    run_name: str, config: dict[str, Any]
) -> tuple[str | None, str | None, str, str]:
    algorithm = _extract_first_config_value(
        config, ("command", "algo", "algorithm", "algorithm_name")
    )
    env = _extract_first_config_value(config, ("env", "env_id", "environment"))
    optimizer = _extract_first_config_value(config, ("optimizer_type", "optimizer"))
    adv_type = _extract_first_config_value(
        config, ("advantage_estimator", "adv_type", "advantage_type")
    )

    if algorithm is None or env is None or optimizer is None or adv_type is None:
        fallback_algorithm, fallback_env, fallback_optimizer, fallback_adv_type = (
            _parse_run_name(run_name)
        )
        if algorithm is None:
            algorithm = fallback_algorithm
        if env is None:
            env = fallback_env
        if optimizer is None:
            optimizer = fallback_optimizer
        if adv_type is None:
            adv_type = fallback_adv_type

    if algorithm is not None:
        algorithm = algorithm.lower().strip()
    if env is not None:
        env = env.strip()
    optimizer = _normalize_meta_value(
        optimizer,
        fallback=_default_optimizer_for_algorithm(algorithm),
    )
    adv_type = _normalize_meta_value(
        adv_type,
        fallback=_default_adv_type_for_algorithm(algorithm),
    )
    return algorithm, env, optimizer, adv_type


def collect_runs(
    entity: str, project: str, min_steps: int, *, step_key: str
) -> list[RunRecord]:
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    selected: list[RunRecord] = []
    skipped_missing_step = 0
    skipped_short = 0
    skipped_missing_config = 0
    step_candidates = _step_key_candidates(step_key)

    for run in runs:
        summary = run.summary or {}
        step = _first_float(summary, keys=step_candidates)
        if step is None:
            skipped_missing_step += 1
            continue
        if step <= min_steps:
            skipped_short += 1
            continue

        config = dict(run.config or {})
        algorithm, env_id, optimizer, adv_type = _extract_run_axes(run.name, config)
        env_key = (
            _make_env_key(env_id, optimizer, adv_type)
            if env_id is not None
            else None
        )
        if not algorithm or not env_key or not env_id:
            skipped_missing_config += 1
            continue

        selected.append(
            RunRecord(
                run=run,
                run_id=run.id,
                name=run.name,
                url=run.url,
                step=step,
                algorithm=algorithm,
                env=env_key,
                env_id=env_id,
                optimizer=optimizer,
                adv_type=adv_type,
            )
        )

    print(
        "Collected runs:",
        f"selected={len(selected)},",
        f"skipped_missing_step={skipped_missing_step},",
        f"skipped_below_threshold={skipped_short},",
        f"skipped_missing_algorithm_or_env={skipped_missing_config}",
    )
    return selected


def group_runs_by_env_and_algorithm(
    runs: list[RunRecord],
) -> dict[str, dict[str, list[RunRecord]]]:
    grouped: dict[str, dict[str, list[RunRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in runs:
        grouped[record.env][record.algorithm].append(record)
    return grouped


def _rows_to_step_value_arrays(
    rows: Any, *, step_key: str, metric_key: str
) -> tuple[np.ndarray, np.ndarray]:
    steps: list[float] = []
    values: list[float] = []

    if rows is None:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    if isinstance(rows, dict):
        row_iterable = [rows]
    elif hasattr(rows, "to_dict"):
        try:
            row_iterable = rows.to_dict(orient="records")
        except Exception:
            row_iterable = rows
    else:
        row_iterable = rows

    step_candidates = _step_key_candidates(step_key)
    metric_candidates = _metric_key_candidates(metric_key)

    for row in row_iterable:
        if not isinstance(row, dict):
            continue
        step = _first_float(row, keys=step_candidates)
        value = _first_float(row, keys=metric_candidates)
        if step is None or value is None:
            continue
        steps.append(step)
        values.append(value)

    if not steps:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    paired = sorted(zip(steps, values), key=lambda item: item[0])
    dedup_steps: list[float] = []
    dedup_values: list[float] = []
    for step, value in paired:
        if dedup_steps and step == dedup_steps[-1]:
            dedup_values[-1] = value
        else:
            dedup_steps.append(step)
            dedup_values.append(value)

    return np.asarray(dedup_steps, dtype=float), np.asarray(dedup_values, dtype=float)


def _load_full_history_series(
    run: Any, *, step_key: str, metric_key: str
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[dict[str, Any]] = []
    keys = list(
        _ordered_unique(
            [
                *_step_key_candidates(step_key),
                *_metric_key_candidates(metric_key),
            ]
        )
    )
    try:
        for row in run.scan_history(keys=keys, page_size=1000):
            if isinstance(row, dict):
                rows.append(row)
    except Exception as exc:
        print(
            f"Warning: failed to scan full history for run '{run.name}' ({run.id}): {exc}"
        )
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return _rows_to_step_value_arrays(rows, step_key=step_key, metric_key=metric_key)


def load_run_series(
    run: Any,
    *,
    step_key: str,
    metric_key: str,
    history_samples: int,
    full_history: bool,
    fallback_scan_history: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if full_history:
        return _load_full_history_series(run, step_key=step_key, metric_key=metric_key)

    keys = list(
        _ordered_unique(
            [
                *_step_key_candidates(step_key),
                *_metric_key_candidates(metric_key),
            ]
        )
    )
    try:
        sampled_rows = run.history(
            keys=keys,
            samples=history_samples,
            pandas=False,
        )
    except Exception as exc:
        print(f"Warning: failed to sample history for run '{run.name}' ({run.id}): {exc}")
        if fallback_scan_history:
            return _load_full_history_series(
                run, step_key=step_key, metric_key=metric_key
            )
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    sampled_steps, sampled_values = _rows_to_step_value_arrays(
        sampled_rows, step_key=step_key, metric_key=metric_key
    )
    if sampled_steps.size > 0:
        return sampled_steps, sampled_values

    if fallback_scan_history:
        return _load_full_history_series(run, step_key=step_key, metric_key=metric_key)
    return np.asarray([], dtype=float), np.asarray([], dtype=float)


def normalize_and_resample(
    steps: np.ndarray, values: np.ndarray, x_grid: np.ndarray
) -> np.ndarray | None:
    if steps.size == 0 or values.size == 0:
        return None
    if steps.size == 1:
        return np.full_like(x_grid, float(values[0]), dtype=float)

    span = float(steps[-1] - steps[0])
    if span <= 0:
        return np.full_like(x_grid, float(values[-1]), dtype=float)

    x = (steps - steps[0]) / span
    x_unique, unique_indices = np.unique(x, return_index=True)
    y_unique = values[unique_indices]
    if x_unique.size == 1:
        return np.full_like(x_grid, float(y_unique[0]), dtype=float)

    return np.interp(x_grid, x_unique, y_unique, left=y_unique[0], right=y_unique[-1])


def _curve_cache_path(
    cache_dir: Path,
    record: RunRecord,
    *,
    step_key: str,
    metric_key: str,
    points: int,
    history_samples: int,
    full_history: bool,
) -> Path:
    payload = (
        f"{CACHE_VERSION}|{record.run_id}|{int(record.step)}|{step_key}|{metric_key}|"
        f"{points}|{history_samples}|{int(full_history)}"
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.npy"


def _load_or_compute_curve(
    record: RunRecord,
    *,
    x_grid: np.ndarray,
    step_key: str,
    metric_key: str,
    history_samples: int,
    full_history: bool,
    fallback_scan_history: bool,
    cache_dir: Path | None,
) -> tuple[np.ndarray | None, bool]:
    if cache_dir is not None:
        cache_path = _curve_cache_path(
            cache_dir,
            record,
            step_key=step_key,
            metric_key=metric_key,
            points=int(x_grid.size),
            history_samples=history_samples,
            full_history=full_history,
        )
        if cache_path.exists():
            try:
                cached_curve = np.load(cache_path, allow_pickle=False)
                if cached_curve.shape == x_grid.shape:
                    return cached_curve, True
            except Exception:
                pass
    else:
        cache_path = None

    steps, values = load_run_series(
        record.run,
        step_key=step_key,
        metric_key=metric_key,
        history_samples=history_samples,
        full_history=full_history,
        fallback_scan_history=fallback_scan_history,
    )
    curve = normalize_and_resample(steps, values, x_grid)
    if curve is not None and cache_path is not None:
        try:
            np.save(cache_path, curve, allow_pickle=False)
        except Exception:
            pass
    return curve, False


def compute_weighted_curves(
    grouped_runs: dict[str, dict[str, list[RunRecord]]],
    *,
    step_key: str,
    metric_key: str,
    points: int,
    history_samples: int,
    full_history: bool,
    fallback_scan_history: bool,
    workers: int,
    cache_dir: Path | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    x_grid = np.linspace(0.0, 1.0, points)
    run_records: list[RunRecord] = []
    for by_algorithm in grouped_runs.values():
        for records in by_algorithm.values():
            run_records.extend(records)

    if not run_records:
        return {}

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    grouped_curves: dict[tuple[str, str], list[tuple[np.ndarray, float]]] = defaultdict(
        list
    )
    cached_hits = 0
    computed = 0
    processed = 0
    total = len(run_records)

    def process_record(record: RunRecord) -> tuple[str, str, np.ndarray, float, bool] | None:
        curve, from_cache = _load_or_compute_curve(
            record,
            x_grid=x_grid,
            step_key=step_key,
            metric_key=metric_key,
            history_samples=history_samples,
            full_history=full_history,
            fallback_scan_history=fallback_scan_history,
            cache_dir=cache_dir,
        )
        if curve is None:
            return None
        weight = max(float(record.step), 1.0)
        return record.env, record.algorithm, curve, weight, from_cache

    if workers <= 1:
        results = map(process_record, run_records)
        for result in results:
            processed += 1
            if result is None:
                continue
            env, algorithm, curve, weight, from_cache = result
            grouped_curves[(env, algorithm)].append((curve, weight))
            if from_cache:
                cached_hits += 1
            else:
                computed += 1
            if processed % 20 == 0 or processed == total:
                print(
                    f"Loaded curves: {processed}/{total} "
                    f"(cache_hits={cached_hits}, fetched={computed})"
                )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_record, record) for record in run_records]
            for future in concurrent.futures.as_completed(futures):
                processed += 1
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"Warning: curve worker failed: {exc}")
                    continue
                if result is None:
                    continue
                env, algorithm, curve, weight, from_cache = result
                grouped_curves[(env, algorithm)].append((curve, weight))
                if from_cache:
                    cached_hits += 1
                else:
                    computed += 1
                if processed % 20 == 0 or processed == total:
                    print(
                        f"Loaded curves: {processed}/{total} "
                        f"(cache_hits={cached_hits}, fetched={computed})"
                    )

    aggregated: dict[str, dict[str, dict[str, Any]]] = {}
    for (env, algorithm), items in grouped_curves.items():
        curves = [curve for curve, _ in items]
        if not curves:
            continue
        weights = np.asarray([weight for _, weight in items], dtype=float)
        curve_stack = np.vstack(curves)
        weighted_mean = np.average(curve_stack, axis=0, weights=weights)
        aggregated.setdefault(env, {})[algorithm] = {
            "x": x_grid,
            "y": weighted_mean,
            "num_runs": len(curves),
            "total_weight_steps": float(np.sum(weights)),
        }

    elapsed = time.perf_counter() - started_at
    print(
        f"Curve loading finished in {elapsed:.1f}s "
        f"(runs={total}, cache_hits={cached_hits}, fetched={computed}, workers={workers})"
    )
    return aggregated


def save_overview_plot(
    aggregated: dict[str, dict[str, dict[str, Any]]],
    *,
    metric_key: str,
    output_path: Path,
) -> str | None:
    envs = sorted(aggregated.keys())
    if not envs:
        return None

    algorithms = sorted(
        {
            algorithm
            for env in envs
            for algorithm in aggregated.get(env, {}).keys()
        }
    )
    palette = [
        "#d39a37",  # warm gold
        "#5f3b7a",  # deep purple
        "#d8bf72",  # light gold
        "#8c6ea6",  # soft purple
        "#b8722f",  # amber
        "#4f6d8f",  # muted blue
    ]
    colors = {algorithm: palette[i % len(palette)] for i, algorithm in enumerate(algorithms)}

    n_envs = len(envs)
    n_cols = 2 if n_envs > 1 else 1
    n_rows = math.ceil(n_envs / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.2 * n_cols, 4.2 * n_rows),
        squeeze=False,
        sharex=True,
        facecolor="#d9d9d9",
    )
    fig.patch.set_facecolor("#d9d9d9")

    for idx, env in enumerate(envs):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        ax.set_facecolor("#d9d9d9")
        env_curves = aggregated[env]
        final_points: list[tuple[float, str]] = []
        for algorithm in sorted(env_curves.keys()):
            curve = env_curves[algorithm]
            x_vals = curve["x"]
            y_vals = curve["y"]
            n_runs = int(curve["num_runs"])
            color = colors[algorithm]
            ax.plot(
                x_vals,
                y_vals,
                label=f"{algorithm} (n={n_runs})",
                color=color,
                linewidth=2.8,
            )
            if len(x_vals) > 0 and len(y_vals) > 0:
                final_points.append((float(y_vals[-1]), color))
                ax.scatter(
                    [float(x_vals[-1])],
                    [float(y_vals[-1])],
                    color=color,
                    s=26,
                    edgecolors="#1e1e1e",
                    linewidths=0.6,
                    zorder=3,
                )

        if len(final_points) <= 4:
            for final_y, color in final_points:
                ax.text(
                    0.985,
                    final_y,
                    f"{final_y:.1f}",
                    ha="right",
                    va="bottom",
                    fontsize=9.5,
                    color="#111111",
                    fontweight="semibold",
                    bbox={
                        "facecolor": "#d9d9d9",
                        "alpha": 0.9,
                        "edgecolor": "none",
                        "pad": 0.1,
                    },
                    zorder=4,
                )

        for spine in ax.spines.values():
            spine.set_color("#222222")
            spine.set_linewidth(1.2)

        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(axis="y", which="major", color="#9e9e9e", linewidth=1.2, alpha=1.0)
        ax.grid(axis="y", which="minor", color="#9e9e9e", linewidth=0.8, alpha=1.0)
        ax.grid(axis="x", which="both", linewidth=0.0)
        ax.tick_params(axis="both", labelsize=10, colors="#1c1c1c")

        y_min, y_max = ax.get_ylim()
        y_span = y_max - y_min
        if y_span > 0:
            ax.set_ylim(y_min - y_span * 0.03, y_max + y_span * 0.08)

        ax.set_title(_display_env_name(env), fontsize=16, fontweight="bold", pad=8)
        ax.set_xlim(0.0, 1.0)
        if row == n_rows - 1:
            ax.set_xlabel("Training Progress →", fontsize=11, fontweight="medium")

        legend = ax.legend(
            fontsize=9,
            frameon=True,
            fancybox=False,
            loc="upper left",
        )
        legend.get_frame().set_facecolor("#d9d9d9")
        legend.get_frame().set_edgecolor("#444444")
        legend.get_frame().set_linewidth(0.9)

        if col != 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel(metric_key, fontsize=11, fontweight="medium")

    for idx in range(n_envs, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].axis("off")

    fig.suptitle(
        "Environment Curves",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.96])
    fig.savefig(output_path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path.name


def _format_step(step: float) -> str:
    if float(step).is_integer():
        return str(int(step))
    return f"{step:.1f}"


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _display_env_name(env: str) -> str:
    env_id, optimizer, adv_type = _split_env_key(env)
    name = env_id.strip()
    if name.startswith("dm_control/"):
        name = name[len("dm_control/") :]
    if name.endswith("-v5"):
        name = name[: -len("-v5")]
    return f"{name} [opt={optimizer}, adv={adv_type}]"


_CONFIG_EXCLUDED_KEYS = {
    "command",
    "env",
    "wandb_entity",
    "wandb_project",
    "wandb_group",
    "out_dir",
}


def _stringify_config_value(value: Any, *, max_len: int = 80) -> str:
    if isinstance(value, (dict, list, tuple)):
        try:
            text = json.dumps(value, sort_keys=True, separators=(",", ":"))
        except Exception:
            text = str(value)
    elif isinstance(value, float):
        text = f"{value:.6g}"
    else:
        text = str(value)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _escape_md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def _build_algorithm_hparam_tables(
    grouped_runs: dict[str, dict[str, list[RunRecord]]],
) -> dict[str, dict[str, dict[str, set[str]]]]:
    tables: dict[str, dict[str, dict[str, set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    for env, by_algorithm in grouped_runs.items():
        for algorithm, records in by_algorithm.items():
            for record in records:
                config = dict(record.run.config or {})
                for key, value in config.items():
                    if key in _CONFIG_EXCLUDED_KEYS or key.startswith("_"):
                        continue
                    tables[algorithm][env][key].add(
                        _stringify_config_value(value)
                    )
    return tables


def _write_algorithm_hparam_tables(
    handle: Any,
    grouped_runs: dict[str, dict[str, list[RunRecord]]],
) -> None:
    tables = _build_algorithm_hparam_tables(grouped_runs)
    if not tables:
        return

    handle.write("## Hyperparameters by Algorithm\n\n")
    handle.write(
        "Rows are hyperparameters and columns are environments. "
        "If multiple runs differ for a cell, values are listed together.\n\n"
    )

    for algorithm in sorted(tables.keys()):
        env_map = tables[algorithm]
        envs = sorted(env_map.keys())
        all_hparams = sorted({key for env in envs for key in env_map[env].keys()})
        if not envs or not all_hparams:
            continue

        handle.write(f"### `{algorithm}`\n\n")
        handle.write(
            "| Hyperparameter | "
            + " | ".join(
                f"`{_escape_md_cell(_display_env_name(env))}`" for env in envs
            )
            + " |\n"
        )
        handle.write("|" + "---|" * (1 + len(envs)) + "\n")

        for hparam in all_hparams:
            row: list[str] = [f"`{_escape_md_cell(hparam)}`"]
            for env in envs:
                values = sorted(env_map[env].get(hparam, set()))
                if not values:
                    row.append("-")
                elif len(values) == 1:
                    row.append(_escape_md_cell(values[0]))
                else:
                    joined = " / ".join(_escape_md_cell(v) for v in values[:4])
                    if len(values) > 4:
                        joined += " / ..."
                    row.append(joined)
            handle.write("| " + " | ".join(row) + " |\n")
        handle.write("\n")


def _build_max_achieved_table(
    grouped_runs: dict[str, dict[str, list[RunRecord]]],
    *,
    metric_key: str,
) -> list[str]:
    envs = sorted(grouped_runs.keys())
    algorithms = sorted(
        {algorithm for env in envs for algorithm in grouped_runs[env].keys()}
    )
    if not envs or not algorithms:
        return []

    lines: list[str] = []
    header = "| Environment | " + " | ".join(f"`{algo}`" for algo in algorithms) + " |"
    separator = "|" + "---|" * (1 + len(algorithms))
    lines.append(header)
    lines.append(separator)

    for env in envs:
        row: list[str] = [f"`{_display_env_name(env)}`"]
        for algorithm in algorithms:
            records = grouped_runs[env].get(algorithm, [])
            run_max_values: list[float] = []
            for record in records:
                summary = record.run.summary or {}
                value = _first_float(
                    summary,
                    keys=_metric_key_candidates("eval/return_max"),
                )
                if value is None:
                    value = _first_float(summary, keys=_metric_key_candidates(metric_key))
                if value is not None:
                    run_max_values.append(value)

            if not run_max_values:
                row.append("-")
            else:
                best = float(np.max(run_max_values))
                spread = float(np.std(run_max_values))
                row.append(f"{best:.1f} +/- {spread:.1f}")

        lines.append("| " + " | ".join(row) + " |")
    return lines


def _default_mujoco_gl_backend() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "glfw"
    if system == "linux":
        return "egl"
    return "glfw"


def _run_score(record: RunRecord, *, metric_key: str) -> float | None:
    summary = record.run.summary or {}
    score = _first_float(
        summary,
        keys=_metric_key_candidates("eval/return_max"),
    )
    if score is None:
        score = _first_float(summary, keys=_metric_key_candidates(metric_key))
    return score


def _select_best_run_for_video(
    records: list[RunRecord],
    *,
    metric_key: str,
) -> tuple[RunRecord | None, float | None]:
    best_record: RunRecord | None = None
    best_score: float | None = None
    for record in records:
        score = _run_score(record, metric_key=metric_key)
        if score is None:
            continue
        if best_record is None:
            best_record = record
            best_score = score
            continue
        if score > (best_score if best_score is not None else float("-inf")):
            best_record = record
            best_score = score
            continue
        if score == best_score and record.step > best_record.step:
            best_record = record
            best_score = score
    return best_record, best_score


def _coerce_int_tuple(value: Any) -> tuple[int, ...] | None:
    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            return None
    if not isinstance(parsed, (list, tuple)):
        return None
    out: list[int] = []
    for item in parsed:
        maybe_int = _as_int(item)
        if maybe_int is None or maybe_int <= 0:
            return None
        out.append(maybe_int)
    if not out:
        return None
    return tuple(out)


def _value_layers_for_algorithm(
    config: dict[str, Any],
    *,
    algorithm: str,
) -> tuple[int, ...] | None:
    if algorithm == "vmpo":
        return _coerce_int_tuple(config.get("value_layer_sizes")) or _coerce_int_tuple(
            config.get("critic_layer_sizes")
        )
    if algorithm in {"ppo", "mpo"}:
        return _coerce_int_tuple(config.get("critic_layer_sizes")) or _coerce_int_tuple(
            config.get("value_layer_sizes")
        )
    return _coerce_int_tuple(config.get("value_layer_sizes"))


def generate_best_run_videos(
    grouped_runs: dict[str, dict[str, list[RunRecord]]],
    *,
    metric_key: str,
    checkpoints_dir: str,
    videos_dir: str,
    gifs_dir: str,
    video_max_steps: int,
    video_fps: int,
    video_attempts: int,
) -> list[BestRunVideoResult]:
    results: list[BestRunVideoResult] = []
    pending: list[tuple[RunRecord, float, Path, Path, Path]] = []
    checkpoints_root = Path(checkpoints_dir)
    videos_root = Path(videos_dir)
    gifs_root = Path(gifs_dir)

    for env in sorted(grouped_runs.keys()):
        for algorithm in sorted(grouped_runs[env].keys()):
            records = grouped_runs[env].get(algorithm, [])
            best_record, best_score = _select_best_run_for_video(
                records,
                metric_key=metric_key,
            )
            if best_record is None or best_score is None:
                results.append(
                    BestRunVideoResult(
                        env=env,
                        algorithm=algorithm,
                        run_name=None,
                        run_url=None,
                        metric_value=None,
                        checkpoint_path=None,
                        video_path=None,
                        gif_path=None,
                        status="skipped",
                        detail=f"No eval/return_max or {metric_key} value in run summaries.",
                    )
                )
                continue

            checkpoint_path = (
                checkpoints_root / algorithm / best_record.name / f"{algorithm}_best.pt"
            )
            video_out_path = videos_root / f"{best_record.name}.mp4"
            gif_out_path = gifs_root / f"{best_record.name}.gif"
            if not checkpoint_path.exists():
                results.append(
                    BestRunVideoResult(
                        env=env,
                        algorithm=algorithm,
                        run_name=best_record.name,
                        run_url=best_record.url,
                        metric_value=best_score,
                        checkpoint_path=str(checkpoint_path),
                        video_path=str(video_out_path),
                        gif_path=str(gif_out_path),
                        status="skipped",
                        detail="Checkpoint file not found.",
                    )
                )
                continue

            pending.append(
                (best_record, best_score, checkpoint_path, video_out_path, gif_out_path)
            )

    if not pending:
        return results

    os.environ.setdefault("MUJOCO_GL", _default_mujoco_gl_backend())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for best_record, best_score, checkpoint_path, video_out_path, gif_out_path in pending:
        config = dict(best_record.run.config or {})
        policy_layer_sizes = _coerce_int_tuple(config.get("policy_layer_sizes"))
        if policy_layer_sizes is None:
            policy_layer_sizes = (256, 256, 256)
        value_layer_sizes = _value_layers_for_algorithm(
            config,
            algorithm=best_record.algorithm,
        )
        seed = _as_int(config.get("seed"))
        if seed is None:
            seed = 42

        try:
            saved_path, saved_gif_path, n_frames = render_policy_video(
                checkpoint_path=str(checkpoint_path),
                algo=best_record.algorithm,
                env_id=best_record.env_id,
                out_path=str(video_out_path),
                gif_out_path=str(gif_out_path),
                seed=seed,
                config=VideoRenderConfig(
                    max_steps=int(video_max_steps),
                    fps=int(video_fps),
                ),
                policy_layer_sizes=policy_layer_sizes,
                value_layer_sizes=value_layer_sizes,
                device=device,
                num_attempts=int(video_attempts),
            )
            results.append(
                BestRunVideoResult(
                    env=best_record.env,
                    algorithm=best_record.algorithm,
                    run_name=best_record.name,
                    run_url=best_record.url,
                    metric_value=best_score,
                    checkpoint_path=str(checkpoint_path),
                    video_path=str(saved_path),
                    gif_path=str(saved_gif_path),
                    status="generated",
                    detail=f"Frames: {n_frames}",
                )
            )
            print(
                f"[video] generated env={best_record.env_id} algo={best_record.algorithm} "
                f"run={best_record.name} video={saved_path} gif={saved_gif_path}"
            )
        except Exception as exc:
            results.append(
                BestRunVideoResult(
                    env=best_record.env,
                    algorithm=best_record.algorithm,
                    run_name=best_record.name,
                    run_url=best_record.url,
                    metric_value=best_score,
                    checkpoint_path=str(checkpoint_path),
                    video_path=str(video_out_path),
                    gif_path=str(gif_out_path),
                    status="failed",
                    detail=str(exc),
                )
            )
            print(
                f"Warning: failed to generate video for run "
                f"'{best_record.name}' ({best_record.algorithm}, {best_record.env_id}): {exc}"
            )

    return results


def write_readme(
    readme_path: Path,
    *,
    entity: str,
    project: str,
    min_steps: int,
    step_key: str,
    metric_key: str,
    runs: list[RunRecord],
    grouped_runs: dict[str, dict[str, list[RunRecord]]],
    aggregated: dict[str, dict[str, dict[str, Any]]],
    overview_image_name: str | None,
    video_results: list[BestRunVideoResult],
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _md_rel_path(path_text: str | None) -> str | None:
        if not path_text:
            return None
        path_obj = Path(path_text)
        try:
            rel = path_obj.relative_to(readme_path.parent)
        except ValueError:
            rel = Path(os.path.relpath(path_obj, readme_path.parent))
        return rel.as_posix()

    with readme_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Report: `{entity}/{project}`\n\n")
        handle.write(f"- Generated: {timestamp}\n")
        handle.write(f"- Included runs: {len(runs)} (`{step_key}` > {min_steps})\n")
        handle.write("- Algorithm key source: run config `command`\n")
        handle.write(
            "- Environment key source: run config `env` + `optimizer_type` + `advantage_estimator`\n"
        )
        handle.write(f"- Metric: `{metric_key}`\n\n")

        if overview_image_name:
            handle.write(f"![overview]({overview_image_name})\n\n")
            handle.write(
                "Each line is a time-weighted average across runs for a single "
                "environment/optimizer/advantage-type and algorithm. "
                "Every run timeline is normalized to `[0, 1]`.\n\n"
            )
            top_table_lines = _build_max_achieved_table(
                grouped_runs,
                metric_key=metric_key,
            )
            if top_table_lines:
                handle.write(
                    "Max achieved table (`eval/return_max`, fallback to selected metric), "
                    "reported as `max +/- std` across runs.\n\n"
                )
                for line in top_table_lines:
                    handle.write(line + "\n")
                handle.write("\n")
                _write_algorithm_hparam_tables(handle, grouped_runs)
        else:
            handle.write(
                "No plottable metric history was found for the selected runs.\n\n"
            )

        handle.write("## Summary\n\n")
        handle.write("| Environment / Optimizer / Adv Type | Algorithms | Runs |\n")
        handle.write("|---|---|---:|\n")
        for env in sorted(grouped_runs.keys()):
            algorithms = sorted(grouped_runs[env].keys())
            run_count = sum(len(grouped_runs[env][algo]) for algo in algorithms)
            algo_display = ", ".join(f"`{algo}`" for algo in algorithms)
            handle.write(
                f"| `{_display_env_name(env)}` | {algo_display} | {run_count} |\n"
            )
        handle.write("\n")

        for env in sorted(grouped_runs.keys()):
            handle.write(f"## {_display_env_name(env)}\n\n")
            env_aggregated = aggregated.get(env, {})
            if env_aggregated:
                handle.write("| Algorithm | Averaged Runs | Total Weight (_step) |\n")
                handle.write("|---|---:|---:|\n")
                for algorithm in sorted(env_aggregated.keys()):
                    curve = env_aggregated[algorithm]
                    handle.write(
                        f"| `{algorithm}` | {int(curve['num_runs'])} | {int(curve['total_weight_steps'])} |\n"
                    )
                handle.write("\n")

            handle.write(f"| Run | Algorithm | {step_key} | {metric_key} |\n")
            handle.write("|---|---|---:|---:|\n")
            env_records: list[RunRecord] = []
            for algorithm in sorted(grouped_runs[env].keys()):
                env_records.extend(grouped_runs[env][algorithm])
            env_records.sort(key=lambda rec: (rec.algorithm, rec.name))

            for record in env_records:
                metric_value = _first_float(
                    (record.run.summary or {}),
                    keys=_metric_key_candidates(metric_key),
                )
                handle.write(
                    f"| [{record.name}]({record.url}) | `{record.algorithm}` | "
                    f"{_format_step(record.step)} | {_format_metric(metric_value)} |\n"
                )
            handle.write("\n")

            env_video_results = sorted(
                (result for result in video_results if result.env == env),
                key=lambda item: (
                    item.algorithm,
                    item.run_name if item.run_name is not None else "",
                ),
            )
            env_gif_results: list[tuple[BestRunVideoResult, str]] = []
            for result in env_video_results:
                if result.status != "generated" or not result.gif_path:
                    continue
                gif_path_obj = Path(result.gif_path)
                if not gif_path_obj.is_file():
                    continue
                gif_rel_path = _md_rel_path(result.gif_path)
                if not gif_rel_path:
                    continue
                env_gif_results.append((result, gif_rel_path))

            if env_gif_results:
                handle.write("### Best-Run GIFs\n\n")
                handle.write(
                    "Best run per algorithm by `eval/return_max` "
                    f"(fallback `{metric_key}`).\n\n"
                )
                handle.write("| Algorithm | Run | Best Metric | Preview |\n")
                handle.write("|---|---|---:|---|\n")
                for result, gif_rel_path in env_gif_results:
                    run_cell = "-"
                    if result.run_name and result.run_url:
                        run_cell = (
                            f"[{_escape_md_cell(result.run_name)}]"
                            f"({result.run_url})"
                        )
                    elif result.run_name:
                        run_cell = f"`{_escape_md_cell(result.run_name)}`"

                    preview_cell = f"![preview]({gif_rel_path})"
                    handle.write(
                        "| "
                        + " | ".join(
                            [
                                f"`{_escape_md_cell(result.algorithm)}`",
                                run_cell,
                                _format_metric(result.metric_value),
                                preview_cell,
                            ]
                        )
                        + " |\n"
                    )
                handle.write("\n")


def create_report_folder(output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_dir / f"report_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def copy_latest_and_build_pdf(output_dir: Path, report_dir: Path) -> None:
    latest_dir = output_dir / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(report_dir, latest_dir)
    print(f"Copied report to latest: {latest_dir}")

    try:
        pandoc_cmd = [
            "pandoc",
            "--from=gfm",
            "--toc",
            f"--resource-path={latest_dir}",
            "-V",
            "fontsize=10pt",
            "-V",
            "geometry:margin=1.5cm",
            "--pdf-engine=xelatex",
            "-o",
            "report.pdf",
            "README.md",
        ]
        subprocess.run(pandoc_cmd, check=True, cwd=latest_dir)
        print(f"Generated PDF report: {latest_dir / 'report.pdf'}")
    except Exception as exc:
        print(f"Warning: failed to build PDF report: {exc}")


def generate_report(
    *,
    project: str,
    entity: str,
    min_steps: int,
    metric_key: str,
    step_key: str,
    output_dir: str,
    points: int,
    history_samples: int,
    full_history: bool,
    fallback_scan_history: bool,
    workers: int,
    cache_dir: str | None,
    generate_videos: bool,
    checkpoints_dir: str,
    videos_dir: str,
    video_max_steps: int,
    video_fps: int,
    video_attempts: int,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_dir) if cache_dir else None

    print(
        "History loading mode:",
        "full_history" if full_history else f"sampled(history_samples={history_samples})",
        f"workers={workers}",
        f"cache_dir={cache_path if cache_path is not None else 'disabled'}",
        f"fallback_scan_history={fallback_scan_history}",
    )

    runs = collect_runs(
        entity=entity,
        project=project,
        min_steps=min_steps,
        step_key=step_key,
    )
    grouped_runs = group_runs_by_env_and_algorithm(runs)
    aggregated = compute_weighted_curves(
        grouped_runs,
        step_key=step_key,
        metric_key=metric_key,
        points=points,
        history_samples=history_samples,
        full_history=full_history,
        fallback_scan_history=fallback_scan_history,
        workers=workers,
        cache_dir=cache_path,
    )

    report_dir = create_report_folder(output_path)
    overview_image_name = save_overview_plot(
        aggregated,
        metric_key=metric_key,
        output_path=report_dir / "overview.png",
    )

    video_results: list[BestRunVideoResult] = []
    if generate_videos:
        video_results = generate_best_run_videos(
            grouped_runs,
            metric_key=metric_key,
            checkpoints_dir=checkpoints_dir,
            videos_dir=videos_dir,
            gifs_dir=str(report_dir),
            video_max_steps=video_max_steps,
            video_fps=video_fps,
            video_attempts=video_attempts,
        )

    write_readme(
        report_dir / "README.md",
        entity=entity,
        project=project,
        min_steps=min_steps,
        step_key=step_key,
        metric_key=metric_key,
        runs=runs,
        grouped_runs=grouped_runs,
        aggregated=aggregated,
        overview_image_name=overview_image_name,
        video_results=video_results,
    )
    print(f"Report generated: {report_dir / 'README.md'}")

    copy_latest_and_build_pdf(output_path, report_dir)
    return report_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a W&B report by averaging runs per "
            "environment-optimizer-adv-type combination and algorithm."
        )
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help="W&B project name (default: %(default)s)",
    )
    parser.add_argument(
        "--entity",
        default=DEFAULT_ENTITY,
        help="W&B entity name (default: %(default)s)",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=DEFAULT_MIN_STEPS,
        help="Minimum `_step` to include a run (strictly greater than this value).",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC_KEY,
        help="Metric key used for plotting and report tables.",
    )
    parser.add_argument(
        "--step-key",
        default=DEFAULT_STEP_KEY,
        help="Step key used for reading run history.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=DEFAULT_POINTS,
        help="Number of points used for normalized interpolation.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where report folders are created.",
    )
    parser.add_argument(
        "--history-samples",
        type=int,
        default=DEFAULT_HISTORY_SAMPLES,
        help="Number of sampled history points per run (ignored with --full-history).",
    )
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="Use full run history via scan_history (slower, highest fidelity).",
    )
    parser.add_argument(
        "--fallback-scan-history",
        action="store_true",
        help="If sampled history is empty/failed, fallback to full scan_history for that run.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of parallel workers for loading run histories.",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="Directory for cached resampled curves. Set to empty string to disable.",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help=(
            "Skip best-run video generation. By default, videos are generated for "
            "the best run per environment and algorithm when checkpoints are present."
        ),
    )
    parser.add_argument(
        "--checkpoints-dir",
        default=DEFAULT_CHECKPOINTS_DIR,
        help=(
            "Checkpoint root directory. Expected layout: "
            "checkpoints/<algo>/<run_name>/<algo>_best.pt"
        ),
    )
    parser.add_argument(
        "--videos-dir",
        default=DEFAULT_VIDEOS_DIR,
        help="Directory where generated videos are written.",
    )
    parser.add_argument(
        "--video-max-steps",
        type=int,
        default=DEFAULT_VIDEO_MAX_STEPS,
        help="Maximum steps per rollout while rendering videos.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=DEFAULT_VIDEO_FPS,
        help="Frames-per-second for generated videos.",
    )
    parser.add_argument(
        "--video-attempts",
        type=int,
        default=DEFAULT_VIDEO_ATTEMPTS,
        help="Number of rollout attempts; the highest-return attempt is saved.",
    )
    args = parser.parse_args()

    if args.points < 2:
        parser.error("--points must be >= 2")
    if args.min_steps < 0:
        parser.error("--min-steps must be >= 0")
    if args.history_samples < 2:
        parser.error("--history-samples must be >= 2")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.video_max_steps < 1:
        parser.error("--video-max-steps must be >= 1")
    if args.video_fps < 1:
        parser.error("--video-fps must be >= 1")
    if args.video_attempts < 1:
        parser.error("--video-attempts must be >= 1")

    generate_report(
        project=args.project,
        entity=args.entity,
        min_steps=args.min_steps,
        metric_key=args.metric,
        step_key=args.step_key,
        output_dir=args.output_dir,
        points=args.points,
        history_samples=args.history_samples,
        full_history=bool(args.full_history),
        fallback_scan_history=bool(args.fallback_scan_history),
        workers=args.workers,
        cache_dir=(args.cache_dir.strip() if args.cache_dir else None),
        generate_videos=not bool(args.skip_videos),
        checkpoints_dir=args.checkpoints_dir,
        videos_dir=args.videos_dir,
        video_max_steps=int(args.video_max_steps),
        video_fps=int(args.video_fps),
        video_attempts=int(args.video_attempts),
    )


if __name__ == "__main__":
    main()
