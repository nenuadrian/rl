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

## hyperparameters

Hyperparameters are defined in the `hyperparameters/*.py` files for each algorithm.
They are intentionally not overridable from the CLI.
