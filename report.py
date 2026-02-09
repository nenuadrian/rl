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
                val = run.summary.get("eval/return_max", None)
                if val is not None:
                    if key not in results:
                        results[key] = {}
                    existing = results[key].get(algo)
                    if existing is None:
                        results[key][algo] = {"val": val, "url": run.url}
                    else:
                        if val > existing["val"]:
                            results[key][algo] = {"val": val, "url": run.url}
    return results, domains_tasks


def generate_report(
    algorithms=ALGORITHMS,
    prefix=PREFIX,
    entity=ENTITY,
    min_steps=MIN_STEPS,
    output_dir="reports",
):
    results, domains_tasks = collect_results(
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
        row = [domain, task]
        for algo in algorithms:
            entry = results.get((domain, task), {}).get(algo)
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
