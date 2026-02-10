# Reinforcement Learning Algorithms in PyTorch

[![Python tests](https://github.com/nenuadrian/rl/actions/workflows/python-app.yml/badge.svg)](https://github.com/nenuadrian/rl/actions/workflows/python-app.yml)

[See the latest report here.](reports/latest/README.md)


- [Reinforcement Learning Algorithms in PyTorch](#reinforcement-learning-algorithms-in-pytorch)
	- [usage](#usage)
	- [video](#video)
	- [hyperparameters](#hyperparameters)
	- [nanochat](#nanochat)
	- [reports](#reports)

## usage

```bash
python main.py mpo --env dm_control/cheetah/run
python main.py ppo --env HalfCheetah-v5
python main.py vmpo --env dm_control/cheetah/run
python main.py nanochat_rl --env isambard
```

## video

Generate a rollout video from the latest saved checkpoint:

```bash
python generate_video.py ppo --env dm_control/cheetah/run
```

Optionally specify a checkpoint path and output file:

```bash
python generate_video.py ppo --env dm_control/cheetah/run \
	--checkpoint checkpoints/ppo/dm_control-cheetah-run/ppo_step_50000.pt \
	--video_out videos/ppo-dm_control-cheetah-run.mp4
```

## hyperparameters

Hyperparameters are defined in the `hyperparameters/*.py` files for each algorithm.


## nanochat

```bash
export WANDB_RUN="nanochat_test_run"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

python -m nanochat.dataset -n 370 

# train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data
python -m nanochat.scripts.tok_train
# evaluate the tokenizer (report compression ratio etc.)
python -m nanochat.scripts.tok_eval


python -m nanochat.scripts.base_train --depth=26 --target-param-data-ratio=8.25 --device-batch-size=16 --fp8 --run=$WANDB_RUN --save-every 100
# OR train with torchrun for better performance
export TORCH_COMPILE_DISABLE=1
export OMP_NUM_THREADS=1

torchrun --standalone --nproc_per_node=4 -m nanochat.scripts.base_train -- --depth=26 --target-param-data-ratio=8.25 --device-batch-size=16 --fp8 --run=$WANDB_RUN --save-every 100

python -m nanochat.scripts.base_eval --device-batch-size=16

python -m nanochat.scripts.chat_sft --device-batch-size=16 --run=$WANDB_RUN
python -m nanochat.scripts.chat_eval -i sft
```

## reports

```bash
python generate_report.py
cd reports/latest
pandoc --from=gfm --toc -V fontsize=9pt -V geometry:margin=1.5cm --pdf-engine=xelatex -o report.pdf README.md
cd ../..
```
