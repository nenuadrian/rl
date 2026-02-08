import argparse
import os
import sys
import importlib

import torch

from utils.env import set_seed
from utils.wandb_utils import finish_wandb, init_wandb


_CLI_ARGS: argparse.Namespace | None = None


def _cli_option_names(argv: list[str]) -> set[str]:
    names: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        name = token[2:]
        if "=" in name:
            name = name.split("=", 1)[0]
        if name:
            names.add(name)
    return names


def _load_preset(algo: str, domain: str, task: str) -> dict:
    module = importlib.import_module(f"hyperparameters.{algo}")
    get_fn = getattr(module, "get", None)
    if get_fn is None:
        raise RuntimeError(f"hyperparameters.{algo} must define get(domain, task)")
    preset = get_fn(domain, task)
    if not isinstance(preset, dict):
        raise TypeError(f"hyperparameters.{algo}.get must return a dict, got {type(preset)}")
    return preset


def _apply_preset(
    args: argparse.Namespace, preset: dict, overridden: set[str]
) -> None:
    for key, value in preset.items():
        dest = key

        possible_flags = {dest, key}
        if possible_flags & overridden:
            continue

        if dest == "hidden_sizes" and isinstance(value, tuple):
            setattr(args, dest, list(value))
        else:
            setattr(args, dest, value)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        nargs="?",
        choices=["sac", "ppo", "vmpo", "mpo"],
        help="Optional subcommand; e.g. `python main.py ppo --domain ... --task ...`",
    )
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
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

    # On-policy (PPO/VMPO)
    parser.add_argument("--rollout_steps", type=int, default=4096)
    parser.add_argument("--update_epochs", type=int, default=10)

    # Shared discounting
    parser.add_argument("--gamma", type=float, default=0.99)

    # PPO-only (safe to ignore for other algos)
    parser.add_argument("--gae_lambda", type=float, default=0.95)

    # VMPO-only (safe to ignore for other algos)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--topk_fraction", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=5.0)
    parser.add_argument("--eta_lr", type=float, default=1e-3)
    parser.add_argument("--epsilon_eta", type=float, default=0.1)
    parser.add_argument("--epsilon_mu", type=float, default=0.01)
    parser.add_argument("--epsilon_sigma", type=float, default=1e-4)
    parser.add_argument("--alpha_lr", type=float, default=1e-3)
    parser.add_argument("--kl_mean_coef", type=float, default=1e-3)
    parser.add_argument("--kl_std_coef", type=float, default=1e-3)

    # Shared by PPO/VMPO/MPO
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    # Off-policy shared (SAC/MPO)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--q_lr", type=float, default=3e-4)

    # SAC-only (safe to ignore for other algos)
    parser.add_argument(
        "--automatic_entropy_tuning",
        dest="automatic_entropy_tuning",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_automatic_entropy_tuning",
        dest="automatic_entropy_tuning",
        action="store_false",
    )

    # MPO-only (safe to ignore for other algos)
    parser.add_argument("--eta_init", type=float, default=1.0)
    parser.add_argument("--kl_epsilon", type=float, default=0.1)
    parser.add_argument("--mstep_kl_epsilon", type=float, default=0.1)
    parser.add_argument("--lambda_init", type=float, default=1.0)
    parser.add_argument("--lambda_lr", type=float, default=1e-3)
    parser.add_argument("--action_samples", type=int, default=16)
    parser.add_argument("--retrace_steps", type=int, default=5)
    parser.add_argument("--retrace_mc_actions", type=int, default=16)

    # PPO-only (safe to ignore for other algos)
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--value_lr", type=float, default=1e-4)
    parser.add_argument("--ent_coef", type=float, default=1e-3)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=0.02)
    parser.add_argument("--normalize_obs", dest="normalize_obs", action="store_true", default=False)
    parser.add_argument("--no_normalize_obs", dest="normalize_obs", action="store_false")

    overridden = _cli_option_names(sys.argv[1:])
    args = parser.parse_args()

    algo = args.command
    if algo is None:
        raise ValueError("Missing algorithm subcommand: choose one of sac/ppo/vmpo/mpo")

    preset = _load_preset(algo, args.domain, args.task)
    _apply_preset(args, preset, overridden)

    _CLI_ARGS = args

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

    if _CLI_ARGS is None:
        raise RuntimeError("CLI args not initialized")
    if algo == "sac":
        from trainers.sac.trainer import Trainer as SACTrainer
        from trainers.sac.agent import SACConfig

        sac_config = SACConfig(
            gamma=float(args.gamma),
            tau=float(args.tau),
            policy_lr=float(args.policy_lr),
            q_lr=float(args.q_lr),
            alpha_lr=float(args.alpha_lr),
            automatic_entropy_tuning=bool(args.automatic_entropy_tuning),
        )

        trainer = SACTrainer(
            domain=args.domain,
            task=args.task,
            seed=args.seed,
            device=device,
            hidden_sizes=tuple(args.hidden_sizes),
            replay_size=args.replay_size,
            config=sac_config,
        )
        trainer.train(
            total_steps=args.total_steps,
            start_steps=args.start_steps,
            update_after=args.update_after,
            batch_size=args.batch_size,
            updates_per_step=args.updates_per_step,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            out_dir=args.out_dir,
        )
    elif algo == "ppo":
        from trainers.ppo.trainer import Trainer as PPOTrainer

        trainer = PPOTrainer(
            domain=args.domain,
            task=args.task,
            seed=args.seed,
            device=device,
            hidden_sizes=tuple(args.hidden_sizes),
            rollout_steps=int(args.rollout_steps),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            update_epochs=int(args.update_epochs),
            minibatch_size=int(args.minibatch_size),
            policy_lr=float(args.policy_lr),
            value_lr=float(args.value_lr),
            clip_ratio=float(args.clip_ratio),
            ent_coef=float(args.ent_coef),
            vf_coef=float(args.vf_coef),
            max_grad_norm=float(args.max_grad_norm),
            target_kl=float(args.target_kl),
            normalize_obs=bool(args.normalize_obs),
        )
        trainer.train(
            total_steps=args.total_steps,
            update_epochs=int(args.update_epochs),
            minibatch_size=int(args.minibatch_size),
            policy_lr=float(args.policy_lr),
            value_lr=float(args.value_lr),
            clip_ratio=float(args.clip_ratio),
            ent_coef=float(args.ent_coef),
            vf_coef=float(args.vf_coef),
            max_grad_norm=float(args.max_grad_norm),
            target_kl=float(args.target_kl),
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            out_dir=args.out_dir,
        )
    elif algo == "vmpo":
        from trainers.vmpo.trainer import Trainer as VMPOTrainer
        from trainers.vmpo.agent import VMPOConfig

        vmpo_config = VMPOConfig(
            gamma=float(args.gamma),
            policy_lr=float(args.policy_lr),
            value_lr=float(args.value_lr),
            topk_fraction=float(args.topk_fraction),
            eta=float(args.eta),
            eta_lr=float(args.eta_lr),
            epsilon_eta=float(args.epsilon_eta),
            epsilon_mu=float(args.epsilon_mu),
            epsilon_sigma=float(args.epsilon_sigma),
            alpha_lr=float(args.alpha_lr),
            kl_mean_coef=float(args.kl_mean_coef),
            kl_std_coef=float(args.kl_std_coef),
            max_grad_norm=float(args.max_grad_norm),
        )

        trainer = VMPOTrainer(
            domain=args.domain,
            task=args.task,
            seed=args.seed,
            device=device,
            hidden_sizes=tuple(args.hidden_sizes),
            num_envs=int(args.num_envs),
            rollout_steps=int(args.rollout_steps),
            config=vmpo_config,
        )
        trainer.train(
            total_steps=args.total_steps,
            update_epochs=int(args.update_epochs),
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            out_dir=args.out_dir,
        )
    elif algo == "mpo":
        from trainers.mpo.trainer import Trainer as MPOTrainer
        from trainers.mpo.agent import MPOConfig

        mpo_config = MPOConfig(
            gamma=float(args.gamma),
            tau=float(args.tau),
            policy_lr=float(args.policy_lr),
            q_lr=float(args.q_lr),
            eta_init=float(args.eta_init),
            eta_lr=float(args.eta_lr),
            kl_epsilon=float(args.kl_epsilon),
            mstep_kl_epsilon=float(args.mstep_kl_epsilon),
            lambda_init=float(args.lambda_init),
            lambda_lr=float(args.lambda_lr),
            max_grad_norm=float(args.max_grad_norm),
            action_samples=int(args.action_samples),
            retrace_steps=int(args.retrace_steps),
            retrace_mc_actions=int(args.retrace_mc_actions),
        )

        trainer = MPOTrainer(
            domain=args.domain,
            task=args.task,
            seed=args.seed,
            device=device,
            hidden_sizes=tuple(args.hidden_sizes),
            replay_size=args.replay_size,
            config=mpo_config,
        )
        trainer.train(
            total_steps=args.total_steps,
            start_steps=args.start_steps,
            update_after=args.update_after,
            batch_size=args.batch_size,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            out_dir=args.out_dir,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    finish_wandb()
