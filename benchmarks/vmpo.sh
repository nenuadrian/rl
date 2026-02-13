
environments = [
    "HalfCheetah-v5",
    "Walker2d-v5",
    "Hopper-v5",
    "Humanoid-v5",

    "dm_control/cheetah/run",
    "dm_control/humanoid/walk",
    "dm_control/humanoid/run",
    "dm_control/walker/walk",
    "dm_control/walker/run",
]

WANDB_PROJECT_NAME="minerva-rl-benchmark-1"
SEEDS=1
SEED_START=42

# loop through in bash
for env in "${environments[@]}"; do
    for seed in $(seq $SEED_START $(($SEED_START + $SEEDS - 1))); do
        python main.py vmpo --env "$env"  --wandb_project_name "$WANDB_PROJECT_NAME" --seed "$seed"
    done
done