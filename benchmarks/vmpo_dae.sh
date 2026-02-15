#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

environments=(
    "dm_control/cheetah/run"
    "dm_control/humanoid/walk"
    "dm_control/humanoid/run"
    "dm_control/walker/walk"
    "dm_control/walker/run"
    "HalfCheetah-v5"
    "Walker2d-v5"
    "Humanoid-v5"
    "Ant-v5"
)

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <benchmark-suffix> [extra main.py args...]" >&2
    exit 1
fi
BENCHMARK_SUFFIX="$1"
shift
WANDB_PROJECT_NAME="minerva-rl-benchmark-${BENCHMARK_SUFFIX}"
SEEDS="${SEEDS:-3}"
SEED_START="${SEED_START:-42}"

for seed in $(seq "$SEED_START" "$((SEED_START + SEEDS - 1))"); do
    for env in "${environments[@]}"; do
        echo "[vmpo_dae] env=${env} seed=${seed}"
        python main.py vmpo \
        --env "$env" \
        --wandb_project "$WANDB_PROJECT_NAME" \
        --seed "$seed" --advantage_estimator dae 

    done
done

