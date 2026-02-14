from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import math
import os
import re
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import time

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_PROJECT = "minerva-rl-benchmark-1"
DEFAULT_ENTITY = "adrian-research"
DEFAULT_MIN_STEPS = 10_000
DEFAULT_STEP_KEY = "_step"
DEFAULT_METRIC_KEY = "eval/return_mean"
DEFAULT_POINTS = 200
DEFAULT_OUTPUT_DIR = "reports"
DEFAULT_HISTORY_SAMPLES = 600
DEFAULT_WORKERS = 8
DEFAULT_CACHE_DIR = ".report_cache"
CACHE_VERSION = "v1"


@dataclass
class RunRecord:
    run: Any
    run_id: str
    name: str
    url: str
    step: float
    algorithm: str
    env: str


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


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


def _parse_run_name(run_name: str) -> tuple[str | None, str | None]:
    name = (run_name or "").strip()
    if "-" not in name:
        return None, None
    algorithm, env_part = name.split("-", 1)
    if not algorithm:
        return None, None
    env = re.sub(r"-seed\d+$", "", env_part).strip("-")
    return algorithm, env if env else None


def _extract_algorithm_and_env(
    run_name: str, config: dict[str, Any]
) -> tuple[str | None, str | None]:
    algorithm = _extract_first_config_value(
        config, ("command", "algo", "algorithm", "algorithm_name")
    )
    env = _extract_first_config_value(config, ("env", "env_id", "environment"))

    if algorithm is None or env is None:
        fallback_algorithm, fallback_env = _parse_run_name(run_name)
        if algorithm is None:
            algorithm = fallback_algorithm
        if env is None:
            env = fallback_env

    if algorithm is not None:
        algorithm = algorithm.lower().strip()
    if env is not None:
        env = env.strip()
    return algorithm, env


def collect_runs(entity: str, project: str, min_steps: int) -> list[RunRecord]:
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    selected: list[RunRecord] = []
    skipped_missing_step = 0
    skipped_short = 0
    skipped_missing_config = 0

    for run in runs:
        summary = run.summary or {}
        step = _as_float(summary.get("_step"))
        if step is None:
            skipped_missing_step += 1
            continue
        if step <= min_steps:
            skipped_short += 1
            continue

        config = dict(run.config or {})
        algorithm, env = _extract_algorithm_and_env(run.name, config)
        if not algorithm or not env:
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
                env=env,
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

    for row in row_iterable:
        if not isinstance(row, dict):
            continue
        step = _as_float(row.get(step_key))
        value = _as_float(row.get(metric_key))
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
    try:
        for row in run.scan_history(keys=[step_key, metric_key], page_size=1000):
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

    try:
        sampled_rows = run.history(
            keys=[step_key, metric_key],
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
    color_map = plt.get_cmap("tab10")
    colors = {algorithm: color_map(i % 10) for i, algorithm in enumerate(algorithms)}

    n_envs = len(envs)
    n_cols = min(3, max(1, math.ceil(math.sqrt(n_envs))))
    n_rows = math.ceil(n_envs / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.0 * n_cols, 3.8 * n_rows),
        squeeze=False,
        sharex=True,
    )

    for idx, env in enumerate(envs):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        env_curves = aggregated[env]
        for algorithm in sorted(env_curves.keys()):
            curve = env_curves[algorithm]
            x_vals = curve["x"]
            y_vals = curve["y"]
            n_runs = int(curve["num_runs"])
            ax.plot(
                x_vals,
                y_vals,
                label=f"{algorithm} (n={n_runs})",
                color=colors[algorithm],
                linewidth=2.0,
            )

        ax.set_title(env)
        ax.set_xlim(0.0, 1.0)
        ax.grid(alpha=0.25)
        if col == 0:
            ax.set_ylabel(metric_key)
        if row == n_rows - 1:
            ax.set_xlabel("normalized training progress")
        ax.legend(fontsize=8, frameon=False)

    for idx in range(n_envs, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].axis("off")

    fig.suptitle("Environment Curves (time-weighted averages across runs)")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(output_path, dpi=180)
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


def write_readme(
    readme_path: Path,
    *,
    entity: str,
    project: str,
    min_steps: int,
    metric_key: str,
    runs: list[RunRecord],
    grouped_runs: dict[str, dict[str, list[RunRecord]]],
    aggregated: dict[str, dict[str, dict[str, Any]]],
    overview_image_name: str | None,
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with readme_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Report: `{entity}/{project}`\n\n")
        handle.write(f"- Generated: {timestamp}\n")
        handle.write(f"- Included runs: {len(runs)} (`_step` > {min_steps})\n")
        handle.write("- Algorithm key source: run config `command`\n")
        handle.write("- Environment key source: run config `env`\n")
        handle.write(f"- Metric: `{metric_key}`\n\n")

        if overview_image_name:
            handle.write(f"![overview]({overview_image_name})\n\n")
            handle.write(
                "Each line is a time-weighted average across runs for a single "
                "environment and algorithm. Every run timeline is normalized to `[0, 1]`.\n\n"
            )
        else:
            handle.write(
                "No plottable metric history was found for the selected runs.\n\n"
            )

        handle.write("## Summary\n\n")
        handle.write("| Environment | Algorithms | Runs |\n")
        handle.write("|---|---|---:|\n")
        for env in sorted(grouped_runs.keys()):
            algorithms = sorted(grouped_runs[env].keys())
            run_count = sum(len(grouped_runs[env][algo]) for algo in algorithms)
            algo_display = ", ".join(f"`{algo}`" for algo in algorithms)
            handle.write(f"| `{env}` | {algo_display} | {run_count} |\n")
        handle.write("\n")

        for env in sorted(grouped_runs.keys()):
            handle.write(f"## {env}\n\n")
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

            handle.write(f"| Run | Algorithm | _step | {metric_key} |\n")
            handle.write("|---|---|---:|---:|\n")
            env_records: list[RunRecord] = []
            for algorithm in sorted(grouped_runs[env].keys()):
                env_records.extend(grouped_runs[env][algorithm])
            env_records.sort(key=lambda rec: (rec.algorithm, rec.name))

            for record in env_records:
                metric_value = _as_float((record.run.summary or {}).get(metric_key))
                handle.write(
                    f"| [{record.name}]({record.url}) | `{record.algorithm}` | "
                    f"{_format_step(record.step)} | {_format_metric(metric_value)} |\n"
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

    runs = collect_runs(entity=entity, project=project, min_steps=min_steps)
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

    write_readme(
        report_dir / "README.md",
        entity=entity,
        project=project,
        min_steps=min_steps,
        metric_key=metric_key,
        runs=runs,
        grouped_runs=grouped_runs,
        aggregated=aggregated,
        overview_image_name=overview_image_name,
    )
    print(f"Report generated: {report_dir / 'README.md'}")

    copy_latest_and_build_pdf(output_path, report_dir)
    return report_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a W&B report by averaging runs per environment and algorithm."
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
    args = parser.parse_args()

    if args.points < 2:
        parser.error("--points must be >= 2")
    if args.min_steps < 0:
        parser.error("--min-steps must be >= 0")
    if args.history_samples < 2:
        parser.error("--history-samples must be >= 2")
    if args.workers < 1:
        parser.error("--workers must be >= 1")

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
    )


if __name__ == "__main__":
    main()
