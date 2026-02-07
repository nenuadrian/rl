import argparse
import os
from typing import Tuple

import torch

from utils.env import set_seed
from utils.wandb_utils import finish_wandb, init_wandb


_CLI_ARGS: argparse.Namespace | None = None


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

        if _CLI_ARGS is None:
            raise RuntimeError("CLI args not initialized")

        ppo_rollout_steps = int(_CLI_ARGS.ppo_rollout_steps)
        ppo_update_epochs = int(_CLI_ARGS.ppo_update_epochs)
        ppo_minibatch_size = int(_CLI_ARGS.ppo_minibatch_size)

        trainer = Trainer(
            domain=domain,
            task=task,
            seed=seed,
            device=device,
            hidden_sizes=hidden_sizes,
            rollout_steps=ppo_rollout_steps,
            update_epochs=ppo_update_epochs,
            minibatch_size=  # `ppo_minibatch_size` is a command-line argument used specifically for the
            # Proximal Policy Optimization (PPO) algorithm. It defines the size of the
            # mini-batches used during training when applying the PPO algorithm.
            # Mini-batches are subsets of the training data that are used to update the
            # policy and value functions in PPO. The `ppo_minibatch_size` argument
            # allows you to specify the number of samples in each mini-batch during the
            # training process.
            ppo_minibatch_size,
            policy_lr=float(_CLI_ARGS.ppo_policy_lr),
            value_lr=float(_CLI_ARGS.ppo_value_lr),
            clip_ratio=float(_CLI_ARGS.ppo_clip_ratio),
            ent_coef=float(_CLI_ARGS.ppo_ent_coef),
            vf_coef=float(_CLI_ARGS.ppo_vf_coef),
            max_grad_norm=float(_CLI_ARGS.ppo_max_grad_norm),
            target_kl=float(_CLI_ARGS.ppo_target_kl),
            normalize_obs=bool(_CLI_ARGS.ppo_normalize_obs),
        )
        trainer.train(
            total_steps=total_steps,
            update_epochs=ppo_update_epochs,
            minibatch_size=ppo_minibatch_size,
            policy_lr=float(_CLI_ARGS.ppo_policy_lr),
            value_lr=float(_CLI_ARGS.ppo_value_lr),
            clip_ratio=float(_CLI_ARGS.ppo_clip_ratio),
            ent_coef=float(_CLI_ARGS.ppo_ent_coef),
            vf_coef=float(_CLI_ARGS.ppo_vf_coef),
            max_grad_norm=float(_CLI_ARGS.ppo_max_grad_norm),
            target_kl=float(_CLI_ARGS.ppo_target_kl),
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
            eval_interval=eval_interval,
            save_interval=save_interval,
            out_dir=out_dir,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        nargs="?",
        choices=["sac", "ppo", "vmpo", "mpo"],
        help="Optional subcommand; e.g. `python main.py ppo --domain ... --task ...`",
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["sac", "ppo", "vmpo", "mpo"],
        default=None,
        help="Algorithm (alternative to subcommand)",
    )
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=500_000)
    parser.add_argument("--start_steps", type=int, default=10_000)
    parser.add_argument("--update_after", type=int, default=1_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--replay_size", type=int, default=1_000_000)
    parser.add_argument("--updates_per_step", type=int, default=4)
    parser.add_argument("--eval_interval", type=int, default=10_000)
    parser.add_argument("--save_interval", type=int, default=50_000)
    parser.add_argument("--hidden_sizes", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)

    # PPO-specific args (only used when algo == 'ppo')
    parser.add_argument("--ppo_rollout_steps", type=int, default=4096)
    parser.add_argument("--ppo_update_epochs", type=int, default=6)
    parser.add_argument("--ppo_minibatch_size", type=int, default=128)
    parser.add_argument("--ppo_clip_ratio", type=float, default=0.2)
    parser.add_argument("--ppo_policy_lr", type=float, default=2e-4)
    parser.add_argument("--ppo_value_lr", type=float, default=5e-5)
    parser.add_argument("--ppo_ent_coef", type=float, default=1e-3)
    parser.add_argument("--ppo_vf_coef", type=float, default=0.5)
    parser.add_argument("--ppo_max_grad_norm", type=float, default=0.5)
    parser.add_argument("--ppo_target_kl", type=float, default=0.02)
    parser.add_argument("--ppo_normalize_obs", action="store_true", default=True)

    args = parser.parse_args()

    _CLI_ARGS = args

    algo = args.command or args.algo
    if algo is None:
        parser.error("Must provide either a subcommand (ppo/sac/vmpo/mpo) or --algo")
    if args.command is not None and args.algo is not None and args.command != args.algo:
        parser.error("If both command and --algo are provided, they must match")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    group = args.wandb_group or f"{algo}-{args.domain}-{args.task}"
    run_name = f"{algo}-{args.domain}-{args.task}"
    project = args.wandb_project or f"dm_control-{algo}"
    init_wandb(
        project=project,
        entity=args.wandb_entity,
        group=group,
        name=run_name,
        config=vars(args),
    )

    train(
        algo=algo,
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
