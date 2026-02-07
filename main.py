import argparse
import os
from typing import Tuple

import torch

from utils.env import set_seed
from utils.wandb_utils import finish_wandb, init_wandb


def train(
    algo: str,
    domain: str,
    task: str,
    seed: int,
    total_steps: int,
    start_steps: int,
    update_after: int,
    batch_size: int,
    replay_size: int,
    updates_per_step: int,
    eval_interval: int,
    save_interval: int,
    hidden_sizes: Tuple[int, int],
    device: torch.device,
    out_dir: str,
):
    if algo == "sac":
        from trainers.sac.trainer import Trainer

        trainer = Trainer(
            domain=domain,
            task=task,
            seed=seed,
            device=device,
            hidden_sizes=hidden_sizes,
            replay_size=replay_size,
        )
        trainer.train(
            total_steps=total_steps,
            start_steps=start_steps,
            update_after=update_after,
            batch_size=batch_size,
            updates_per_step=updates_per_step,
            eval_interval=eval_interval,
            save_interval=save_interval,
            out_dir=out_dir,
        )
    elif algo == "ppo":
        from trainers.ppo.trainer import Trainer

        trainer = Trainer(
            domain=domain,
            task=task,
            seed=seed,
            device=device,
            hidden_sizes=hidden_sizes,
            rollout_steps=update_after,
        )
        trainer.train(
            total_steps=total_steps,
            batch_size=batch_size,
            update_epochs=updates_per_step,
            eval_interval=eval_interval,
            save_interval=save_interval,
            out_dir=out_dir,
        )
    elif algo == "vmpo":
        from trainers.vmpo.trainer import Trainer

        trainer = Trainer(
            domain=domain,
            task=task,
            seed=seed,
            device=device,
            hidden_sizes=hidden_sizes,
            rollout_steps=update_after,
        )
        trainer.train(
            total_steps=total_steps,
            update_epochs=updates_per_step,
            eval_interval=eval_interval,
            save_interval=save_interval,
            out_dir=out_dir,
        )
    elif algo == "mpo":
        from trainers.mpo.trainer import Trainer

        trainer = Trainer(
            domain=domain,
            task=task,
            seed=seed,
            device=device,
            hidden_sizes=hidden_sizes,
            replay_size=replay_size,
        )
        trainer.train(
            total_steps=total_steps,
            start_steps=start_steps,
            update_after=update_after,
            batch_size=batch_size,
            updates_per_step=updates_per_step,
            eval_interval=eval_interval,
            save_interval=save_interval,
            out_dir=out_dir,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=500_000)
    parser.add_argument("--start_steps", type=int, default=10_000)
    parser.add_argument("--update_after", type=int, default=1_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--replay_size", type=int, default=1_000_000)
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=10_000)
    parser.add_argument("--save_interval", type=int, default=50_000)
    parser.add_argument("--hidden_sizes", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--wandb_project", type=str, default="dm_control")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    group = args.wandb_group or f"{args.algo}-{args.domain}-{args.task}"
    run_name = f"{args.algo}-{args.domain}-{args.task}"
    init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=group,
        name=run_name,
        config=vars(args),
    )

    train(
        algo=args.algo,
        domain=args.domain,
        task=args.task,
        seed=args.seed,
        total_steps=args.total_steps,
        start_steps=args.start_steps,
        update_after=args.update_after,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        updates_per_step=args.updates_per_step,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        hidden_sizes=tuple(args.hidden_sizes),
        device=device,
        out_dir=args.out_dir,
    )

    finish_wandb()
