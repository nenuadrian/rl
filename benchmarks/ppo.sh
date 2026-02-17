#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
JOBS_DIR="$ROOT_DIR/jobs"
mkdir -p "$JOBS_DIR"

environments=(
    "dm_control/cheetah/run"
    "dm_control/humanoid/run"
    "dm_control/humanoid/run_pure_state"
    "dm_control/walker/run"
    "HalfCheetah-v5"
    "Walker2d-v5"
    "Humanoid-v5"
    "Ant-v5"
)

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <benchmark-suffix>" >&2
    exit 1
fi

BENCHMARK_SUFFIX="$1"
WANDB_PROJECT_NAME="minerva-rl-benchmark-${BENCHMARK_SUFFIX}"
SEEDS="${SEEDS:-1}"
SEED_START="${SEED_START:-42}"
SBATCH_PARTITION="${SBATCH_PARTITION:-multicore}"
SBATCH_NTASKS="${SBATCH_NTASKS:-12}"
SBATCH_TIME="${SBATCH_TIME:-0-6}"

if ! [[ "$SEEDS" =~ ^[0-9]+$ ]] || [[ "$SEEDS" -lt 1 ]]; then
    echo "SEEDS must be a positive integer (got: $SEEDS)" >&2
    exit 1
fi

benchmark_slug="${BENCHMARK_SUFFIX//\//-}"
benchmark_slug="${benchmark_slug// /-}"

for env in "${environments[@]}"; do
    env_slug="${env//\//-}"
    env_slug="${env_slug// /-}"
    job_file="$JOBS_DIR/ppo_${benchmark_slug}_${env_slug}.sbatch.sh"

    cat > "$job_file" <<EOF
#!/bin/bash --login
#SBATCH -p ${SBATCH_PARTITION}          
#SBATCH -n ${SBATCH_NTASKS}
#SBATCH -t ${SBATCH_TIME}              
#SBATCH -a 1-${SEEDS}

WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME}"
env="${env}"
SEED_START="${SEED_START}"
seed=\$((SEED_START + \${SLURM_ARRAY_TASK_ID:-0}))
WORKDIR="\${WORKDIR:-\$HOME/scratch/rl}"
CONDA_ENV="\${CONDA_ENV:-minerva}"

cd "\$WORKDIR"
conda activate "\$CONDA_ENV"
python main.py ppo \\
        --env "\$env" \\
        --wandb_project "\$WANDB_PROJECT_NAME" \\
        --seed "\$seed"
EOF

    chmod +x "$job_file"
    echo "Created $job_file"
done

echo "Done. Submit jobs with: sbatch jobs/<job-file>.sbatch.sh"
