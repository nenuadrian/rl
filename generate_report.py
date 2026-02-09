import wandb
import os
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
                    }
                )

                # keep best value per domain/task for the main table
                if val is not None:
                    if key not in results:
                        results[key] = {}
                    existing = results[key].get(algo)
                    if existing is None:
                        results[key][algo] = {"val": val, "url": run.url}
                    else:
                        if val > existing["val"]:
                            results[key][algo] = {"val": val, "url": run.url}
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
    report_path = os.path.join(output_dir, f"report_{timestamp}.md")
    header = "| Domain | Task | " + " | ".join(algorithms) + " |\n"
    # separator: one `---` per column (Domain, Task, and each algorithm)
    sep_cols = 2 + len(algorithms)
    header += "|" + "---|" * sep_cols + "\n"
    rows = []
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
    with open(report_path, "w") as f:
        f.write(f"# Report\n\n")
        f.write(header)
        for r in rows:
            f.write(r)
        f.write("\n")
        # Per-algorithm sections: list all runs sorted by domain, task, val(desc)
        for algo in algorithms:
            f.write(f"## {algo}\n\n")
            f.write("| Run | Domain | Task | _step | eval/return_max |\n")
            f.write("|---|---|---|---:|---:|\n")
            runs = runs_by_algo.get(algo, [])
            def sort_key(r):
                val_sort = r["val"] if r["val"] is not None else float("-inf")
                return (r["domain"], r["task"], -val_sort)

            for run_entry in sorted(runs, key=sort_key):
                name_link = f"[{run_entry['name']}]({run_entry['url']})"
                steps = run_entry.get("steps", 0)
                val = run_entry.get("val")
                val_display = str(int(val)) if val is not None else "-"
                f.write(f"| {name_link} | {run_entry['domain']} | {run_entry['task']} | {steps} | {val_display} |\n")
            f.write("\n")
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
