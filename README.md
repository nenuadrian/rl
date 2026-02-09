# Reinforcement Learning Algorithms in PyTorch

[![Python tests](https://github.com/nenuadrian/rl/actions/workflows/python-app.yml/badge.svg)](https://github.com/nenuadrian/rl/actions/workflows/python-app.yml)

## usage

```bash
python main.py --algo mpo --domain cheetah --task run
python main.py --algo ppo --domain cheetah --task run
python main.py --algo sac --domain cheetah --task run
python main.py --algo vmpo --domain cheetah --task run
python main.py --algo vmpo_light --domain cheetah --task run
python main.py --algo vmpo_parallel --domain cheetah --task run
```

## hyperparameters

Hyperparameters are defined in the `hyperparameters/*.py` files for each algorithm.
