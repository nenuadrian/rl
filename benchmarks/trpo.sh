#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

environments=(
    "HalfCheetah-v5"
    "Walker2d-v5"
    "Humanoid-v5"
    "Ant-v5"
    "dm_control/cheetah/run"
    "dm_control/humanoid/walk"
    "dm_control/humanoid/run"
    "dm_control/walker/walk"
    "dm_control/walker/run"
)

WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-minerva-rl-benchmark-3}"
SEEDS="${SEEDS:-1}"
SEED_START="${SEED_START:-42}"
EXTRA_ARGS=("$@")

for env in "${environments[@]}"; do
    for seed in $(seq "$SEED_START" "$((SEED_START + SEEDS - 1))"); do
        echo "[trpo] env=${env} seed=${seed}"
        python main.py trpo \
        --env "$env" \
        --wandb_project "$WANDB_PROJECT_NAME" \
        --seed "$seed" \
        "${EXTRA_ARGS[@]}"
    done
done
