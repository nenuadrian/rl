# Report: `adrian-research/minerva-rl-benchmark-7`

- Generated: 2026-02-15 15:36:10
- Included runs: 17 (`_step` > 1000)
- Algorithm key source: run config `command`
- Environment key source: run config `env` + `optimizer_type` + `advantage_estimator`
- Metric: `eval/return_mean`

![overview](overview.png)

Each line is a time-weighted average across runs for a single environment/optimizer/advantage-type and algorithm. Every run timeline is normalized to `[0, 1]`.

Max achieved table (`eval/return_max`, fallback to selected metric), reported as `max +/- std` across runs.

| Environment | `mpo` | `ppo` | `r2d2-gtrxl` | `vmpo` | `vmpo-gtrxl` |
|---|---|---|---|---|---|
| `HalfCheetah[adam,dae]` | - | - | - | -164.5 +/- 0.0 | - |
| `HalfCheetah[adam,gae]` | - | 1819.0 +/- 0.0 | - | -211.0 +/- 0.0 | - |
| `HalfCheetah[adam,returns]` | - | - | - | 1746.4 +/- 0.0 | - |
| `Humanoid[adam,gae]` | - | 1413.6 +/- 0.0 | - | - | - |
| `Humanoid[adam,returns]` | - | - | - | 621.4 +/- 0.0 | - |
| `PongNoFrameskip-v4` | - | - | 1.0 +/- 0.0 | - | 18.0 +/- 0.0 |
| `SpaceInvadersNoFrameskip-v4` | - | - | - | - | 50.0 +/- 0.0 |
| `Walker2d[adam,gae]` | - | 1013.6 +/- 0.0 | - | - | - |
| `Walker2d[adam,returns]` | - | - | - | 1165.7 +/- 0.0 | - |
| `cheetah/run[adam,gae]` | - | 438.4 +/- 0.0 | - | - | - |
| `cheetah/run[adam,none]` | 530.9 +/- 0.0 | - | - | - | - |
| `humanoid/walk[adam,gae]` | - | 7.2 +/- 0.0 | - | - | - |
| `humanoid/walk[adam,none]` | 2.1 +/- 0.0 | - | - | - | - |
| `pendulum` | - | - | -3.1 +/- 0.0 | - | -3.0 +/- 0.0 |

## Hyperparameters by Algorithm

Rows are hyperparameters and columns are environments. If multiple runs differ for a cell, values are listed together.

### `mpo`

| Hyperparameter | `cheetah/run[adam,none]` | `humanoid/walk[adam,none]` |
|---|---|---|
| `action_penalization` | False | False |
| `action_samples` | 32 | 256 |
| `batch_size` | 128 | 512 |
| `critic_layer_sizes` | [512,512,256] | [512,512,256] |
| `device` | None | None |
| `epsilon_penalty` | 0.001 | 0.001 |
| `gamma` | 0.995 | 0.995 |
| `kl_epsilon` | 0.2 | 0.1 |
| `lambda_init` | 1 | 1 |
| `lambda_lr` | 0.0003 | 0.0003 |
| `max_grad_norm` | 1 | 1 |
| `mstep_kl_epsilon` | 0.3 | 0.1 |
| `optimizer_type` | adam | adam |
| `per_dim_constraining` | False | True |
| `policy_layer_sizes` | [256,256,256] | [256,256,256] |
| `policy_lr` | 0.0003 | 0.0003 |
| `q_lr` | 0.0003 | 0.0003 |
| `replay_size` | 1000000 | 1000000 |
| `retrace_lambda` | 0.95 | 0.95 |
| `retrace_mc_actions` | 8 | 8 |
| `retrace_steps` | 2 | 2 |
| `seed` | 42 | 42 |
| `sgd_momentum` | 0.9 | 0.9 |
| `target_critic_update_period` | 100 | 100 |
| `target_policy_update_period` | 50 | 100 |
| `temperature_init` | 1 | 1 |
| `temperature_lr` | 0.0003 | 0.0003 |
| `total_steps` | 400000 | 400000 |
| `update_after` | 1000 | 1000 |
| `updates_per_step` | 1 | 2 |
| `use_retrace` | False | True |

### `ppo`

Environments 1-4 of 5.

| Hyperparameter | `HalfCheetah[adam,gae]` | `Humanoid[adam,gae]` | `Walker2d[adam,gae]` | `cheetah/run[adam,gae]` |
|---|---|---|---|---|
| `advantage_estimator` | - | - | - | - |
| `anneal_lr` | True | True | True | True |
| `clip_ratio` | 0.2 | 0.2 | 0.2 | 0.2 |
| `clip_vloss` | True | True | True | True |
| `critic_layer_sizes` | [512,512,256] | [512,512,256] | [512,512,256] | [512,512,256] |
| `device` | None | None | None | None |
| `ent_coef` | 0 | 0 | 0 | 0 |
| `gae_lambda` | 0.92 | 0.92 | 0.92 | 0.92 |
| `gamma` | 0.98 | 0.98 | 0.98 | 0.98 |
| `max_grad_norm` | 0.8 | 0.8 | 0.8 | 0.8 |
| `minibatch_size` | 32 | 32 | 32 | 32 |
| `norm_adv` | True | True | True | True |
| `normalize_obs` | True | True | True | True |
| `num_envs` | 1 | 1 | 1 | 1 |
| `optimizer_type` | adam | adam | adam | adam |
| `policy_layer_sizes` | [256,256,256] | [256,256,256] | [256,256,256] | [256,256,256] |
| `policy_lr` | 0.0002 | 0.0002 | 0.0002 | 0.0002 |
| `rollout_steps` | 2048 | 2048 | 2048 | 2048 |
| `seed` | 42 | 42 | 42 | 42 |
| `sgd_momentum` | 0.9 | 0.9 | 0.9 | 0.9 |
| `target_kl` | 0 | 0 | 0 | 0 |
| `total_steps` | 500000 | 500000 | 500000 | 500000 |
| `update_epochs` | 10 | 10 | 10 | 10 |
| `vf_coef` | 0.5 | 0.5 | 0.5 | 0.5 |

Environments 5-5 of 5.

| Hyperparameter | `humanoid/walk[adam,gae]` |
|---|---|
| `advantage_estimator` | None |
| `anneal_lr` | True |
| `clip_ratio` | 0.2 |
| `clip_vloss` | True |
| `critic_layer_sizes` | [512,512,256] |
| `device` | None |
| `ent_coef` | 0 |
| `gae_lambda` | 0.92 |
| `gamma` | 0.98 |
| `max_grad_norm` | 0.8 |
| `minibatch_size` | 32 |
| `norm_adv` | True |
| `normalize_obs` | True |
| `num_envs` | 1 |
| `optimizer_type` | adam |
| `policy_layer_sizes` | [256,256,256] |
| `policy_lr` | 0.0002 |
| `rollout_steps` | 2048 |
| `seed` | 42 |
| `sgd_momentum` | 0.9 |
| `target_kl` | 0 |
| `total_steps` | 500000 |
| `update_epochs` | 10 |
| `vf_coef` | 0.5 |

### `vmpo`

Environments 1-4 of 5.

| Hyperparameter | `HalfCheetah[adam,dae]` | `HalfCheetah[adam,gae]` | `HalfCheetah[adam,returns]` | `Humanoid[adam,returns]` |
|---|---|---|---|---|
| `advantage_estimator` | dae | gae | returns | returns |
| `alpha_lr` | 0.0003 | 0.0003 | 0.0003 | 0.0003 |
| `device` | None | None | None | None |
| `epsilon_eta` | 0.25 | 0.25 | 0.25 | 0.05 |
| `epsilon_mu` | 0.05 | 0.05 | 0.05 | 0.05 |
| `epsilon_sigma` | 0.001 | 0.001 | 0.001 | 0.0003 |
| `gae_lambda` | 0.95 | 0.95 | 0.95 | 0.95 |
| `gamma` | 0.99 | 0.99 | 0.99 | 0.995 |
| `max_grad_norm` | 1 | 1 | 1 | 1 |
| `normalize_advantages` | True | True | True | True |
| `num_envs` | 1 | 1 | 1 | 1 |
| `optimizer_type` | adam | adam | adam | adam |
| `policy_layer_sizes` | [256,256,256] | [256,256,256] | [256,256,256] | [256,256,256] |
| `policy_lr` | 0.0002 | 0.0002 | 0.0002 | 0.0001 |
| `rollout_steps` | 1024 | 1024 | 1024 | 4096 |
| `seed` | 42 | 42 | 42 | 42 |
| `sgd_momentum` | 0.9 | 0.9 | 0.9 | 0.9 |
| `temperature_init` | 2 | 2 | 2 | 1 |
| `temperature_lr` | 0.001 | 0.001 | 0.001 | 0.0002 |
| `topk_fraction` | 0.45 | 0.45 | 0.45 | 0.4 |
| `total_steps` | 3000000 | 3000000 | 3000000 | 3000000 |
| `updates_per_step` | 1 | 1 | 1 | 1 |
| `value_layer_sizes` | [512,512,256] | [512,512,256] | [512,512,256] | [512,512,256] |
| `value_lr` | 0.0003 | 0.0003 | 0.0003 | 0.0001 |

Environments 5-5 of 5.

| Hyperparameter | `Walker2d[adam,returns]` |
|---|---|
| `advantage_estimator` | returns |
| `alpha_lr` | 0.0001 |
| `device` | None |
| `epsilon_eta` | 0.25 |
| `epsilon_mu` | 0.05 |
| `epsilon_sigma` | 0.001 |
| `gae_lambda` | 0.95 |
| `gamma` | 0.99 |
| `max_grad_norm` | 1 |
| `normalize_advantages` | True |
| `num_envs` | 1 |
| `optimizer_type` | adam |
| `policy_layer_sizes` | [256,256,256] |
| `policy_lr` | 0.0001 |
| `rollout_steps` | 4096 |
| `seed` | 42 |
| `sgd_momentum` | 0.9 |
| `temperature_init` | 1 |
| `temperature_lr` | 0.001 |
| `topk_fraction` | 0.4 |
| `total_steps` | 3000000 |
| `updates_per_step` | 1 |
| `value_layer_sizes` | [512,512,256] |
| `value_lr` | 0.0003 |


## HalfCheetah[adam,dae]

| Algorithm | Averaged Runs | Total Weight (_step) |
|---|---:|---:|
| `vmpo` | 1 | 1197000 |

| Run | Algorithm | _step | eval/return_mean |
|---|---|---:|---:|
| [vmpo_HalfCheetah-v5-adam-dae_20260215-150629](https://wandb.ai/adrian-research/minerva-rl-benchmark-7/runs/alf61rnt) | `vmpo` | 1197000 | -225.574 |

## HalfCheetah[adam,gae]

| Algorithm | Averaged Runs | Total Weight (_step) |
|---|---:|---:|
| `ppo` | 1 | 499712 |
| `vmpo` | 1 | 1048576 |

| Run | Algorithm | _step | eval/return_mean |
|---|---|---:|---:|
| [ppo_HalfCheetah-v5-adam-gae_20260215-131433](https://wandb.ai/adrian-research/minerva-rl-benchmark-7/runs/0qvkzb4h) | `ppo` | 499712 | 1705.962 |
| [vmpo_HalfCheetah-v5-adam-gae_20260215-150608](https://wandb.ai/adrian-research/minerva-rl-benchmark-7/runs/x94d6tp7) | `vmpo` | 1048576 | -317.135 |

## HalfCheetah[adam,returns]

| Algorithm | Averaged Runs | Total Weight (_step) |
|---|---:|---:|
| `vmpo` | 1 | 3000000 |

| Run | Algorithm | _step | eval/return_mean |
|---|---|---:|---:|
| [vmpo_HalfCheetah-v5-adam-returns_20260215-131443](https://wandb.ai/adrian-research/minerva-rl-benchmark-7/runs/qondxpk4) | `vmpo` | 3000000 | 840.225 |

## Humanoid[adam,gae]

| Algorithm | Averaged Runs | Total Weight (_step) |
|---|---:|---:|
| `ppo` | 1 | 499712 |

| Run | Algorithm | _step | eval/return_mean |
|---|---|---:|---:|
| [ppo_Humanoid-v5-adam-gae_20260215-141431](https://wandb.ai/adrian-research/minerva-rl-benchmark-7/runs/pt6u357t) | `ppo` | 499712 | 763.375 |

## Humanoid[adam,returns]

| Algorithm | Averaged Runs | Total Weight (_step) |
|---|---:|---:|
| `vmpo` | 1 | 939893 |

| Run | Algorithm | _step | eval/return_mean |
|---|---|---:|---:|
| [vmpo_Humanoid-v5-adam-returns_20260215-150405](https://wandb.ai/adrian-research/minerva-rl-benchmark-7/runs/hm29njqx) | `vmpo` | 939893 | 353.521 |

## PongNoFrameskip-v4

| Algorithm | Averaged Runs | Total Weight (_step) |
|---|---:|---:|
| `r2d2-gtrxl` | 1 | 6397 |
| `vmpo-gtrxl` | 1 | 27555 |

| Run | Algorithm | _step | eval/return_mean |
|---|---|---:|---:|
| [r2d2-gtrxl_PongNoFrameskip-v4_20260215-132841](https://wandb.ai/adrian-research/minerva-rl-benchmark-7/runs/jkb7kfo7) | `r2d2-gtrxl` | 6397 | -8.600 |
| [vmpo-gtrxl_PongNoFrameskip-v4_20260215-132238](https://wandb.ai/adrian-research/minerva-rl-benchmark-7/runs/8uu488es) | `vmpo-gtrxl` | 27555 | 17.875 |

## SpaceInvadersNoFrameskip-v4

| Algorithm | Averaged Runs | Total Weight (_step) |
|---|---:|---:|
| `vmpo-gtrxl` | 1 | 3394 |

| Run | Algorithm | _step | eval/return_mean |
|---|---|---:|---:|
| [vmpo-gtrxl_SpaceInvadersNoFrameskip-v4_20260215-151030](https://wandb.ai/adrian-research/minerva-rl-benchmark-7/runs/e5259rm1) | `vmpo-gtrxl` | 3394 | 21.875 |

## Walker2d[adam,gae]

| Algorithm | Averaged Runs | Total Weight (_step) |
|---|---:|---:|
| `ppo` | 1 | 499712 |

| Run | Algorithm | _step | eval/return_mean |
|---|---|---:|---:|
| [ppo_Walker2d-v5-adam-gae_20260215-135225](https://wandb.ai/adrian-research/minerva-rl-benchmark-7/runs/wpksf09x) | `ppo` | 499712 | 735.812 |

## Walker2d[adam,returns]

| Algorithm | Averaged Runs | Total Weight (_step) |
|---|---:|---:|
| `vmpo` | 1 | 3000000 |

| Run | Algorithm | _step | eval/return_mean |
|---|---|---:|---:|
| [vmpo_Walker2d-v5-adam-returns_20260215-140732](https://wandb.ai/adrian-research/minerva-rl-benchmark-7/runs/xet0xx4v) | `vmpo` | 3000000 | 631.832 |

## cheetah/run[adam,gae]

| Algorithm | Averaged Runs | Total Weight (_step) |
|---|---:|---:|
| `ppo` | 1 | 499712 |

| Run | Algorithm | _step | eval/return_mean |
|---|---|---:|---:|
| [ppo_dm_control-cheetah-run-adam-gae_20260215-144004](https://wandb.ai/adrian-research/minerva-rl-benchmark-7/runs/5zwkctpq) | `ppo` | 499712 | 330.048 |

## cheetah/run[adam,none]

| Algorithm | Averaged Runs | Total Weight (_step) |
|---|---:|---:|
| `mpo` | 1 | 400000 |

| Run | Algorithm | _step | eval/return_mean |
|---|---|---:|---:|
| [mpo_dm_control-cheetah-run-adam-none_20260215-131449](https://wandb.ai/adrian-research/minerva-rl-benchmark-7/runs/fl1b85vs) | `mpo` | 400000 | 525.603 |

## humanoid/walk[adam,gae]

| Algorithm | Averaged Runs | Total Weight (_step) |
|---|---:|---:|
| `ppo` | 1 | 43008 |

| Run | Algorithm | _step | eval/return_mean |
|---|---|---:|---:|
| [ppo_dm_control-humanoid-walk-adam-gae_20260215-152619](https://wandb.ai/adrian-research/minerva-rl-benchmark-7/runs/n3yabvty) | `ppo` | 43008 | 1.144 |

## humanoid/walk[adam,none]

| Algorithm | Averaged Runs | Total Weight (_step) |
|---|---:|---:|
| `mpo` | 1 | 107431 |

| Run | Algorithm | _step | eval/return_mean |
|---|---|---:|---:|
| [mpo_dm_control-humanoid-walk-adam-none_20260215-141539](https://wandb.ai/adrian-research/minerva-rl-benchmark-7/runs/bpopkqvp) | `mpo` | 107431 | 0.899 |

## pendulum

| Algorithm | Averaged Runs | Total Weight (_step) |
|---|---:|---:|
| `r2d2-gtrxl` | 1 | 5112 |
| `vmpo-gtrxl` | 1 | 3375 |

| Run | Algorithm | _step | eval/return_mean |
|---|---|---:|---:|
| [r2d2-gtrxl_pendulum_20260215-131600](https://wandb.ai/adrian-research/minerva-rl-benchmark-7/runs/751z6p1t) | `r2d2-gtrxl` | 5112 | -436.572 |
| [vmpo-gtrxl_pendulum_20260215-131600](https://wandb.ai/adrian-research/minerva-rl-benchmark-7/runs/ztkn8e17) | `vmpo-gtrxl` | 3375 | -401.144 |

