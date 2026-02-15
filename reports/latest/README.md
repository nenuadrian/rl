# Report: `adrian-research/minerva-rl-benchmark-6`

- Generated: 2026-02-15 13:05:15
- Included runs: 12 (`_step` > 1)
- Algorithm key source: run config `command`
- Environment key source: run config `env` + `optimizer_type` + `advantage_estimator`
- Metric: `eval/return_mean`

No plottable metric history was found for the selected runs.

## Summary

| Environment / Optimizer / Adv Type | Algorithms | Runs |
|---|---|---:|
| `Ant [opt=adam, adv=gae]` | `ppo` | 1 |
| `HalfCheetah [opt=adam, adv=gae]` | `ppo` | 1 |
| `HalfCheetah [opt=adam, adv=returns]` | `vmpo` | 1 |
| `Humanoid [opt=adam, adv=gae]` | `ppo` | 1 |
| `Humanoid [opt=adam, adv=returns]` | `vmpo` | 1 |
| `Walker2d [opt=adam, adv=gae]` | `ppo` | 1 |
| `Walker2d [opt=adam, adv=returns]` | `vmpo` | 1 |
| `cheetah/run [opt=adam, adv=none]` | `mpo` | 1 |
| `humanoid/walk [opt=adam, adv=none]` | `mpo` | 1 |
| `pendulum [opt=unknown, adv=unknown]` | `pendulum-vmpo-gtrxl-seed0`, `r2d2-gtrxl`, `vmpo-gtrxl` | 3 |

## Ant [opt=adam, adv=gae]

| Run | Algorithm | _step | eval/return_mean |
|---|---|---:|---:|
| [ppo_Ant-v5-adam-gae_20260215-122440](https://wandb.ai/adrian-research/minerva-rl-benchmark-6/runs/aae8x37p) | `ppo` | 499712 | 287.203 |

### Best-Run GIFs

Best run per algorithm by `eval/return_max` (fallback `eval/return_mean`).

| Algorithm | Run | Best Metric | Preview | Status |
|---|---|---:|---|---|
| `ppo` | [ppo_Ant-v5-adam-gae_20260215-122440](https://wandb.ai/adrian-research/minerva-rl-benchmark-6/runs/aae8x37p) | 964.610 | - | skipped: Checkpoint file not found. |

## HalfCheetah [opt=adam, adv=gae]

| Run | Algorithm | _step | eval/return_mean |
|---|---|---:|---:|
| [ppo_HalfCheetah-v5-adam-gae_20260215-101843](https://wandb.ai/adrian-research/minerva-rl-benchmark-6/runs/nqawi5yj) | `ppo` | 499712 | 1735.166 |

### Best-Run GIFs

Best run per algorithm by `eval/return_max` (fallback `eval/return_mean`).

| Algorithm | Run | Best Metric | Preview | Status |
|---|---|---:|---|---|
| `ppo` | [ppo_HalfCheetah-v5-adam-gae_20260215-101843](https://wandb.ai/adrian-research/minerva-rl-benchmark-6/runs/nqawi5yj) | 1816.913 | - | skipped: Checkpoint file not found. |

## HalfCheetah [opt=adam, adv=returns]

| Run | Algorithm | _step | eval/return_mean |
|---|---|---:|---:|
| [vmpo_HalfCheetah-v5-adam-returns_20260215-101844](https://wandb.ai/adrian-research/minerva-rl-benchmark-6/runs/dshbp31d) | `vmpo` | 3000000 | 840.225 |

### Best-Run GIFs

Best run per algorithm by `eval/return_max` (fallback `eval/return_mean`).

| Algorithm | Run | Best Metric | Preview | Status |
|---|---|---:|---|---|
| `vmpo` | [vmpo_HalfCheetah-v5-adam-returns_20260215-101844](https://wandb.ai/adrian-research/minerva-rl-benchmark-6/runs/dshbp31d) | 1746.357 | - | skipped: Checkpoint file not found. |

## Humanoid [opt=adam, adv=gae]

| Run | Algorithm | _step | eval/return_mean |
|---|---|---:|---:|
| [ppo_Humanoid-v5-adam-gae_20260215-115235](https://wandb.ai/adrian-research/minerva-rl-benchmark-6/runs/kqtj0l05) | `ppo` | 499712 | 564.861 |

### Best-Run GIFs

Best run per algorithm by `eval/return_max` (fallback `eval/return_mean`).

| Algorithm | Run | Best Metric | Preview | Status |
|---|---|---:|---|---|
| `ppo` | [ppo_Humanoid-v5-adam-gae_20260215-115235](https://wandb.ai/adrian-research/minerva-rl-benchmark-6/runs/kqtj0l05) | 905.211 | - | skipped: Checkpoint file not found. |

## Humanoid [opt=adam, adv=returns]

| Run | Algorithm | _step | eval/return_mean |
|---|---|---:|---:|
| [vmpo_Humanoid-v5-adam-returns_20260215-120644](https://wandb.ai/adrian-research/minerva-rl-benchmark-6/runs/ees7elbq) | `vmpo` | 2392171 | 569.381 |

### Best-Run GIFs

Best run per algorithm by `eval/return_max` (fallback `eval/return_mean`).

| Algorithm | Run | Best Metric | Preview | Status |
|---|---|---:|---|---|
| `vmpo` | [vmpo_Humanoid-v5-adam-returns_20260215-120644](https://wandb.ai/adrian-research/minerva-rl-benchmark-6/runs/ees7elbq) | 989.586 | - | skipped: Checkpoint file not found. |

## Walker2d [opt=adam, adv=gae]

| Run | Algorithm | _step | eval/return_mean |
|---|---|---:|---:|
| [ppo_Walker2d-v5-adam-gae_20260215-112104](https://wandb.ai/adrian-research/minerva-rl-benchmark-6/runs/n57unbo6) | `ppo` | 499712 | 626.997 |

### Best-Run GIFs

Best run per algorithm by `eval/return_max` (fallback `eval/return_mean`).

| Algorithm | Run | Best Metric | Preview | Status |
|---|---|---:|---|---|
| `ppo` | [ppo_Walker2d-v5-adam-gae_20260215-112104](https://wandb.ai/adrian-research/minerva-rl-benchmark-6/runs/n57unbo6) | 734.302 | - | skipped: Checkpoint file not found. |

## Walker2d [opt=adam, adv=returns]

| Run | Algorithm | _step | eval/return_mean |
|---|---|---:|---:|
| [vmpo_Walker2d-v5-adam-returns_20260215-111111](https://wandb.ai/adrian-research/minerva-rl-benchmark-6/runs/xv2kj8kj) | `vmpo` | 3000000 | 631.832 |

### Best-Run GIFs

Best run per algorithm by `eval/return_max` (fallback `eval/return_mean`).

| Algorithm | Run | Best Metric | Preview | Status |
|---|---|---:|---|---|
| `vmpo` | [vmpo_Walker2d-v5-adam-returns_20260215-111111](https://wandb.ai/adrian-research/minerva-rl-benchmark-6/runs/xv2kj8kj) | 1165.683 | - | skipped: Checkpoint file not found. |

## cheetah/run [opt=adam, adv=none]

| Run | Algorithm | _step | eval/return_mean |
|---|---|---:|---:|
| [mpo_dm_control-cheetah-run-adam-none_20260215-101849](https://wandb.ai/adrian-research/minerva-rl-benchmark-6/runs/jd6870xr) | `mpo` | 400000 | 525.603 |

### Best-Run GIFs

Best run per algorithm by `eval/return_max` (fallback `eval/return_mean`).

| Algorithm | Run | Best Metric | Preview | Status |
|---|---|---:|---|---|
| `mpo` | [mpo_dm_control-cheetah-run-adam-none_20260215-101849](https://wandb.ai/adrian-research/minerva-rl-benchmark-6/runs/jd6870xr) | 530.872 | - | skipped: Checkpoint file not found. |

## humanoid/walk [opt=adam, adv=none]

| Run | Algorithm | _step | eval/return_mean |
|---|---|---:|---:|
| [mpo_dm_control-humanoid-walk-adam-none_20260215-112009](https://wandb.ai/adrian-research/minerva-rl-benchmark-6/runs/dfmjdyaq) | `mpo` | 140242 | 1.312 |

### Best-Run GIFs

Best run per algorithm by `eval/return_max` (fallback `eval/return_mean`).

| Algorithm | Run | Best Metric | Preview | Status |
|---|---|---:|---|---|
| `mpo` | [mpo_dm_control-humanoid-walk-adam-none_20260215-112009](https://wandb.ai/adrian-research/minerva-rl-benchmark-6/runs/dfmjdyaq) | 11.224 | - | skipped: Checkpoint file not found. |

## pendulum [opt=unknown, adv=unknown]

| Run | Algorithm | _step | eval/return_mean |
|---|---|---:|---:|
| [pendulum-vmpo-gtrxl-seed0_pendulum_20260215-130137](https://wandb.ai/adrian-research/minerva-rl-benchmark-6/runs/2lyx6mui) | `pendulum-vmpo-gtrxl-seed0` | 2009 | -1390.144 |
| [r2d2-gtrxl_pendulum_20260215-130409](https://wandb.ai/adrian-research/minerva-rl-benchmark-6/runs/yldp920h) | `r2d2-gtrxl` | 21 | -1713.304 |
| [vmpo-gtrxl_pendulum_20260215-130240](https://wandb.ai/adrian-research/minerva-rl-benchmark-6/runs/zonujpqx) | `vmpo-gtrxl` | 331 | -1307.331 |

### Best-Run GIFs

Best run per algorithm by `eval/return_max` (fallback `eval/return_mean`).

| Algorithm | Run | Best Metric | Preview | Status |
|---|---|---:|---|---|
| `pendulum-vmpo-gtrxl-seed0` | [pendulum-vmpo-gtrxl-seed0_pendulum_20260215-130137](https://wandb.ai/adrian-research/minerva-rl-benchmark-6/runs/2lyx6mui) | -1029.172 | - | skipped: Checkpoint file not found. |
| `r2d2-gtrxl` | [r2d2-gtrxl_pendulum_20260215-130409](https://wandb.ai/adrian-research/minerva-rl-benchmark-6/runs/yldp920h) | -1575.469 | - | skipped: Checkpoint file not found. |
| `vmpo-gtrxl` | [vmpo-gtrxl_pendulum_20260215-130240](https://wandb.ai/adrian-research/minerva-rl-benchmark-6/runs/zonujpqx) | -1014.560 | - | skipped: Checkpoint file not found. |

