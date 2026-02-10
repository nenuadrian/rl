import wandb
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

PREFIX = "dm_control-"
ALGORITHMS = ["ppo", "vmpo", "mpo"]
ENTITY = "adrian-research"
MIN_STEPS = 50_000


def get_runs_for_algorithm(
    algorithm, prefix=PREFIX, entity=ENTITY, min_steps=MIN_STEPS
):
    project = f"{prefix}{algorithm}"
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    filtered = []
    for run in runs:
        steps = run.summary.get("_step", 0)
        if steps >= min_steps:
            filtered.append(run)
    return filtered


def parse_run_name(run_name):
    # Expected: algo-domain-task-other-data
    parts = run_name.split("-")
    if len(parts) < 3:
        return None, None
    domain = parts[1]
    task = parts[2]
    return domain, task


def collect_results(
    algorithms=ALGORITHMS, prefix=PREFIX, entity=ENTITY, min_steps=MIN_STEPS
):
    results = {}
    domains_tasks = set()
    runs_by_algo = {algo: [] for algo in algorithms}
    for algo in algorithms:
        runs = get_runs_for_algorithm(
            algo, prefix=prefix, entity=entity, min_steps=min_steps
        )
        print(f"Found {len(runs)} runs for algorithm '{algo}' with prefix '{prefix}'")
        for run in runs:
            domain, task = parse_run_name(run.name)
            if domain and task:
                key = (domain, task)
                domains_tasks.add(key)

                steps = run.summary.get("_step", 0)
                val = run.summary.get("eval/return_max", None)

                # collect all runs for per-algorithm tables
                runs_by_algo[algo].append(
                    {
                        "name": run.name,
                        "url": run.url,
                        "domain": domain,
                        "task": task,
                        "steps": steps,
                        "val": val,
                        "run_obj": run,
                    }
                )

                # keep best value per domain/task for the main table
                if val is not None:
                    if key not in results:
                        results[key] = {}
                    existing = results[key].get(algo)
                    if existing is None:
                        results[key][algo] = {"val": val, "url": run.url, "run": run}
                    else:
                        if val > existing["val"]:
                            results[key][algo] = {"val": val, "url": run.url, "run": run}
    return results, domains_tasks, runs_by_algo


def generate_report(
    algorithms=ALGORITHMS,
    prefix=PREFIX,
    entity=ENTITY,
    min_steps=MIN_STEPS,
    output_dir="reports",
):
    results, domains_tasks, runs_by_algo = collect_results(
        algorithms=algorithms, prefix=prefix, entity=entity, min_steps=min_steps
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    report_dir = os.path.join(output_dir, f"report_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "report.md")
    header = "| Domain | Task | " + " | ".join(algorithms) + " |\n"
    # separator: one `---` per column (Domain, Task, and each algorithm)
    sep_cols = 2 + len(algorithms)
    header += "|" + "---|" * sep_cols + "\n"
    rows = []
    # Keep ordered list of domain/task rows that we will plot
    plotted_rows = []
    for domain, task in sorted(domains_tasks):
        # only include rows where at least two algorithms have eval/return_max > 100
        entry_map = results.get((domain, task), {})
        present_count = sum(
            1
            for algo in algorithms
            if (entry_map.get(algo) is not None and entry_map.get(algo).get("val") is not None and entry_map.get(algo).get("val") > 100)
        )
        if present_count < 2:
            continue

        row = [domain, task]
        for algo in algorithms:
            entry = entry_map.get(algo)
            if entry:
                display = f"[{int(entry['val'])}]({entry['url']})"
            else:
                display = "-"
            row.append(display)
        rows.append("| " + " | ".join(row) + " |\n")
        plotted_rows.append((domain, task))
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

    # Helper: create time-series plot for a domain/task using the best run per algorithm
    def save_time_series_plot(domain, task, results_map, algs, out_dir):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = {algs[i]: c for i, c in enumerate(["#4C72B0", "#55A868", "#C44E52"]) }
        any_plotted = False
        for a in algs:
            entry = results_map.get((domain, task), {}).get(a)
            if not entry or entry.get("run") is None:
                continue
            run_obj = entry.get("run")
            steps, vals = _get_series_from_run(run_obj)
            if not steps or not vals:
                continue
            ax.plot(steps, vals, label=a, color=colors.get(a))
            any_plotted = True
        if not any_plotted:
            plt.close(fig)
            return None
        ax.set_xlabel("_step")
        ax.set_ylabel("eval/return_max")
        ax.set_title(f"{domain} - {task}")
        ax.legend()
        fname = f"{domain}_{task}.png".replace("/", "_")
        path = os.path.join(out_dir, fname)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return fname

    # Generate plots for each included domain/task
    images = {}
    for domain, task in plotted_rows:
        img_name = save_time_series_plot(domain, task, results, algorithms, report_dir)
        if img_name:
            images[(domain, task)] = img_name

    with open(report_path, "w") as f:
        f.write(f"# Report\n\n")
        f.write(header)
        for r in rows:
            f.write(r)
        f.write("\n")

        # Insert per-domain/task image sections and a table of runs for that domain/task
        for domain, task in plotted_rows:
            img = images.get((domain, task))
            f.write(f"## {domain} - {task}\n\n")
            if img:
                f.write(f"![{domain} {task}]({img})\n\n")
            else:
                f.write("No data available for plot.\n\n")

            # Insert wandb config JSON markdown for each best run per algorithm
            entry_map = results.get((domain, task), {})
            for algo in algorithms:
                entry = entry_map.get(algo)
                if entry and entry.get("run") is not None:
                    config = dict(entry["run"].config)
                    import json
                    config_json = json.dumps(config, indent=2, sort_keys=True)
                    f.write(f"**{algo} config:**\n\n")
                    f.write("```json\n" + config_json + "\n```\n\n")

            # Gather runs for this domain/task across all algorithms
            domain_runs = []
            for algo in algorithms:
                for run_entry in runs_by_algo.get(algo, []):
                    if run_entry.get("domain") == domain and run_entry.get("task") == task:
                        # include algorithm label for clarity
                        r = run_entry.copy()
                        r["algorithm"] = algo
                        domain_runs.append(r)

            if domain_runs:
                # sort by value (descending), missing values go last
                domain_runs.sort(key=lambda r: (r["algorithm"], -(r["val"] if r.get("val") is not None else float("-inf"))))
                f.write("| Run | Algorithm | _step | eval/return_max |\n")
                f.write("|---|---|---:|---:|\n")
                for run_entry in domain_runs:
                    name_link = f"[{run_entry['name']}]({run_entry['url']})"
                    algo = run_entry.get("algorithm")
                    steps = run_entry.get("steps", 0)
                    val = run_entry.get("val")
                    val_display = str(int(val)) if val is not None else "-"
                    f.write(f"| {name_link} | {algo} | {steps} | {val_display} |\n")
                f.write("\n")
            else:
                f.write("No runs available for this domain/task.\n\n")
    print(f"Report generated: {report_path}")


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
