import wandb
import os
import shutil
import subprocess
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

PREFIX = "env-"
ALGORITHMS = ["ppo", "vmpo", "mpo", "vmpo_sgd", "ppo_lm", "nanochat_rl"]
ENTITY = "adrian-research"
MIN_STEPS = 50_000


def get_runs_for_algorithm(
    algorithm, prefix=PREFIX, entity=ENTITY, min_steps=MIN_STEPS
):
    project = f"{prefix}{algorithm}"
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    filtered = []
    try:
        for run in runs:
            steps = run.summary.get("_step", 0)
            if steps >= min_steps:
                filtered.append(run)
    except Exception as e:
        print(f"Error fetching runs for {project}: {e}")
    return filtered


def parse_run_name(run_name):
    # Expected: prefix-env-... (e.g., ppo-Humanoid-v5-...)
    parts = run_name.split("-")
    if len(parts) < 2:
        return None
    environment = parts[1]
    return environment


def collect_results(
    algorithms=ALGORITHMS, prefix=PREFIX, entity=ENTITY, min_steps=MIN_STEPS
):
    results = {}
    environments = set()
    runs_by_algo = {algo: [] for algo in algorithms}
    for algo in algorithms:
        runs = get_runs_for_algorithm(
            algo, prefix=prefix, entity=entity, min_steps=min_steps
        )
        print(f"Found {len(runs)} runs for algorithm '{algo}' with prefix '{prefix}'")
        for run in runs:
            environment = parse_run_name(run.name)
            if environment:
                key = environment
                environments.add(key)

                steps = run.summary.get("_step", 0)
                val = run.summary.get("eval/return_max", None)

                # collect all runs for per-algorithm tables
                runs_by_algo[algo].append(
                    {
                        "name": run.name,
                        "url": run.url,
                        "environment": environment,
                        "steps": steps,
                        "val": val,
                        "run_obj": run,
                    }
                )

                # keep best value per environment for the main table
                if val is not None:
                    if key not in results:
                        results[key] = {}
                    existing = results[key].get(algo)
                    if existing is None:
                        results[key][algo] = {"val": val, "url": run.url, "run": run}
                    else:
                        if val > existing["val"]:
                            results[key][algo] = {
                                "val": val,
                                "url": run.url,
                                "run": run,
                            }
    return results, environments, runs_by_algo


def generate_report(
    algorithms=ALGORITHMS,
    prefix=PREFIX,
    entity=ENTITY,
    min_steps=MIN_STEPS,
    output_dir="reports",
):
    results, environments, runs_by_algo = collect_results(
        algorithms=algorithms, prefix=prefix, entity=entity, min_steps=min_steps
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    report_dir = os.path.join(output_dir, f"report_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "README.md")
    header = "| Environment | " + " | ".join(algorithms) + " |\n"
    # separator: one `---` per column (Environment and each algorithm)
    sep_cols = 1 + len(algorithms)
    header += "|" + "---|" * sep_cols + "\n"
    rows = []
    # Keep ordered list of environments that we will plot
    plotted_envs = []
    for environment in sorted(environments):
        # only include rows where at least two algorithms have eval/return_max > 100
        entry_map = results.get(environment, {})
        row = [environment]
        for algo in algorithms:
            entry = entry_map.get(algo)
            if entry:
                display = f"[{int(entry['val'])}]({entry['url']})"
            else:
                display = "-"
            row.append(display)
        rows.append("| " + " | ".join(row) + " |\n")
        plotted_envs.append(environment)

    # Helper: fetch series from a wandb run for given keys
    def _get_series_from_run(run_obj, step_key="_step", val_key="eval/return_max"):
        # Try pandas=True first (if pandas is available), otherwise fallback
        steps = []
        vals = []
        df = run_obj.history(keys=[step_key, val_key], pandas=True)
        if df is not None and len(df) > 0:
            if step_key in df.columns and val_key in df.columns:
                steps = df[step_key].tolist()
                vals = df[val_key].tolist()
            else:
                # fallback to scanning
                for row in df.to_dict(orient="records"):
                    s = row.get(step_key)
                    v = row.get(val_key)
                    if s is not None and v is not None:
                        steps.append(s)
                        vals.append(v)
        # Ensure lists are sorted by step
        if len(steps) != len(vals):
            # align by index up to min length
            n = min(len(steps), len(vals))
            steps = steps[:n]
            vals = vals[:n]
        try:
            paired = sorted(zip(steps, vals), key=lambda x: x[0])
            steps, vals = zip(*paired) if paired else ([], [])
            return list(steps), list(vals)
        except Exception:
            return steps, vals

    # Helper: create time-series plot for an environment using the best run per algorithm
    def save_time_series_plot(environment, results_map, algs, out_dir):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = {algs[i]: c for i, c in enumerate(["#4C72B0", "#55A868", "#C44E52"])}
        any_plotted = False
        for a in algs:
            entry = results_map.get(environment, {}).get(a)
            if not entry or entry.get("run") is None:
                continue
            run_obj = entry.get("run")
            steps, vals = _get_series_from_run(run_obj)
            if not steps or not vals:
                continue
            # Normalize x so each algorithm spans the full chart width, regardless of step count.
            n = len(vals)
            if n == 1:
                x = [0.0, 1.0]
                y = [vals[0], vals[0]]
            else:
                x = [i / (n - 1) for i in range(n)]
                y = vals
            ax.plot(x, y, label=a, color=colors.get(a))
            any_plotted = True
        if not any_plotted:
            plt.close(fig)
            return None
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("training progress")
        ax.set_ylabel("eval/return_max")
        ax.set_title(f"{environment}")
        ax.legend()
        fname = f"{environment.lower()}.png".replace("/", "_")
        path = os.path.join(out_dir, fname)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return fname

    # Generate plots for each included environment
    images = {}
    for environment in plotted_envs:
        img_name = save_time_series_plot(environment, results, algorithms, report_dir)
        if img_name:
            images[environment] = img_name

    with open(report_path, "w") as f:
        f.write(f"# Report\n\n")
        f.write(header)
        for r in rows:
            f.write(r)
        f.write("\n")

        # Insert per-environment image sections and a table of runs for that environment
        for environment in plotted_envs:
            img = images.get(environment)
            f.write(f"## {environment}\n\n")
            if img:
                f.write(f"![{environment}]({img})\n\n")
            else:
                f.write("No data available for plot.\n\n")

            # Insert wandb config JSON markdown for each best run per algorithm
            entry_map = results.get(environment, {})
            for algo in algorithms:
                entry = entry_map.get(algo)
                if entry and entry.get("run") is not None:
                    config = dict(entry["run"].config)
                    import json

                    config_json = json.dumps(config, indent=2, sort_keys=True)
                    f.write(f"**{algo} config:**\n\n")
                    f.write("```json\n" + config_json + "\n```\n\n")

            # Gather runs for this environment across all algorithms
            env_runs = []
            for algo in algorithms:
                for run_entry in runs_by_algo.get(algo, []):
                    if run_entry.get("environment") == environment:
                        # include algorithm label for clarity
                        r = run_entry.copy()
                        r["algorithm"] = algo
                        env_runs.append(r)

            if env_runs:
                # sort by value (descending), missing values go last
                env_runs.sort(
                    key=lambda r: (
                        r["algorithm"],
                        -(r["val"] if r.get("val") is not None else float("-inf")),
                    )
                )
                f.write("| Run | Algorithm | _step | eval/return_max |\n")
                f.write("|---|---|---:|---:|\n")
                for run_entry in env_runs:
                    name_link = f"[{run_entry['name']}]({run_entry['url']})"
                    algo = run_entry.get("algorithm")
                    steps = run_entry.get("steps", 0)
                    val = run_entry.get("val")
                    val_display = str(int(val)) if val is not None else "-"
                    f.write(f"| {name_link} | {algo} | {steps} | {val_display} |\n")
                f.write("\n")
            else:
                f.write("No runs available for this environment.\n\n")
    print(f"Report generated: {report_path}")
    # Create or replace a 'latest' copy of the report folder
    latest_dir = os.path.join(output_dir, "latest")
    try:
        if os.path.exists(latest_dir):
            shutil.rmtree(latest_dir)
        shutil.copytree(report_dir, latest_dir)
        print(f"Copied report to latest: {latest_dir}")

        # Also generate a PDF inside reports/latest from the copied README.
        latest_readme = "README.md"
        latest_pdf = "report.pdf"
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
            latest_pdf,
            latest_readme,
        ]
        subprocess.run(pandoc_cmd, check=True, cwd=latest_dir)
        print(f"Generated PDF report: {os.path.join(latest_dir, latest_pdf)}")
    except Exception as e:
        print(f"Warning: failed to create latest copy/pdf: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate WandB evaluation report")
    parser.add_argument(
        "--prefix", default=PREFIX, help="Project prefix (default: %(default)s)"
    )
    parser.add_argument(
        "--entity", default=ENTITY, help="WandB entity (default: %(default)s)"
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=MIN_STEPS,
        help="Minimum global steps to include run",
    )
    parser.add_argument(
        "--output-dir", default="reports", help="Directory for generated report"
    )
    args = parser.parse_args()

    generate_report(
        algorithms=ALGORITHMS,
        prefix=args.prefix,
        entity=args.entity,
        min_steps=args.min_steps,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
