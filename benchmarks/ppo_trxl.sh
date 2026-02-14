#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

environments=(
    #"ProofofMemory-v0"
    "MortarMayhem-Grid-v0"
    "MysteryPath-Grid-v0"
    "MysteryPath-v0"
)

WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-minerva-rl-benchmark-2}"
SEEDS="${SEEDS:-1}"
SEED_START="${SEED_START:-42}"
EXTRA_ARGS=("$@")

for env in "${environments[@]}"; do
    for seed in $(seq "$SEED_START" "$((SEED_START + SEEDS - 1))"); do
        echo "[ppo_trxl] env=${env} seed=${seed}"
        python main.py ppo_trxl \
        --env "$env" \
        --wandb_project "$WANDB_PROJECT_NAME" \
        --seed "$seed" \
        "${EXTRA_ARGS[@]}"
    done
done
