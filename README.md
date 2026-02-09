# Reinforcement Learning Algorithms in PyTorch

[![Python tests](https://github.com/nenuadrian/rl/actions/workflows/python-app.yml/badge.svg)](https://github.com/nenuadrian/rl/actions/workflows/python-app.yml)

## usage

```bash
python main.py mpo --domain cheetah --task run
python main.py ppo --domain cheetah --task run
python main.py vmpo --domain cheetah --task run
python main.py vmpo_light --domain cheetah --task run
python main.py vmpo_parallel --domain cheetah --task run
```

## video

Generate a rollout video from the latest saved checkpoint:

```bash
python generate_video.py ppo --domain cheetah --task run
```

Optionally specify a checkpoint path and output file:

```bash
python generate_video.py ppo --domain cheetah --task run \
	--checkpoint checkpoints/ppo/cheetah/run/ppo_step_50000.pt \
	--video_out videos/ppo-cheetah-run.mp4
```

## hyperparameters

Hyperparameters are defined in the `hyperparameters/*.py` files for each algorithm.
