import argparse
import os
import sys
import importlib
from dataclasses import asdict, is_dataclass
from pprint import pformat

import torch

from utils.env import set_seed
from utils.wandb_utils import finish_wandb, init_wandb
from utils.video import VideoRenderConfig, find_latest_checkpoint, render_policy_video


_CLI_ARGS: argparse.Namespace | None = None


def _load_preset(algo: str, domain: str, task: str) -> dict:
    module = importlib.import_module(f"hyperparameters.{algo}")
    get_fn = getattr(module, "get", None)
    if get_fn is None:
        raise RuntimeError(f"hyperparameters.{algo} must define get(domain, task)")
    preset = get_fn(domain, task)
    if not isinstance(preset, dict):
        raise TypeError(
            f"hyperparameters.{algo}.get must return a dict, got {type(preset)}"
        )
    return preset


def _apply_preset(args: argparse.Namespace, preset: dict) -> None:
    for key, value in preset.items():
        setattr(args, key, value)


def _print_resolved_args(args: argparse.Namespace) -> None:
    print("Resolved args (after hyperparameter preset):")
    print(pformat(dict(sorted(vars(args).items())), sort_dicts=True))


def _print_config(name: str, config: object) -> None:
    print(f"{name}:")
    if is_dataclass(config) and not isinstance(config, type):
        print(pformat(asdict(config), sort_dicts=True))
    else:
        print(repr(config))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        nargs="?",
        choices=["ppo", "vmpo", "mpo", "vmpo_parallel", "vmpo_light"],
    )
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)

    parser.add_argument(
        "--generate_video",
        action="store_true",
        default=False,
        help="Generate a rollout video from the latest checkpoint and exit.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional video checkpoint path (.pt). Defaults to latest in --out_dir for the selected algo.",
    )
    parser.add_argument(
        "--video_out",
        type=str,
        default=None,
        help="Output video path. Defaults to videos/{algo}-{domain}-{task}.mp4",
    )
    parser.add_argument("--video_max_steps", type=int, default=1000)
    args = parser.parse_args()

    algo = args.command
    if algo is None:
        raise ValueError("Missing algorithm subcommand")

    preset = _load_preset(algo, args.domain, args.task)
    _apply_preset(args, preset)

    _print_resolved_args(args)

    _CLI_ARGS = args

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.out_dir = os.path.join(args.out_dir, algo, args.domain, args.task)

    os.makedirs(args.out_dir, exist_ok=True)

    if bool(args.generate_video):
        os.environ.setdefault("MUJOCO_GL", "egl")
        checkpoint_path = (
            str(args.checkpoint)
            if args.checkpoint is not None
            else find_latest_checkpoint(args.out_dir, algo)
        )
        out_path = (
            str(args.video_out)
            if args.video_out is not None
            else os.path.join("videos", f"{algo}-{args.domain}-{args.task}.mp4")
        )
        saved_path, n_frames = render_policy_video(
            checkpoint_path=checkpoint_path,
            algo=algo,
            domain=args.domain,
            task=args.task,
            out_path=out_path,
            seed=int(args.seed),
            config=VideoRenderConfig(
                max_steps=int(args.video_max_steps),
                fps=int(30),
            ),
            policy_layer_sizes=tuple(args.policy_layer_sizes),
            device=device,
        )
        print(f"Saved video to: {saved_path}")
        print(f"Frames: {n_frames}")
        sys.exit(0)

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
    if algo == "ppo":
        from trainers.ppo.trainer import Trainer as PPOTrainer

        _print_config(
            "PPO config",
            {
                "rollout_steps": int(args.rollout_steps),
                "gamma": float(args.gamma),
                "gae_lambda": float(args.gae_lambda),
                "update_epochs": int(args.update_epochs),
                "minibatch_size": int(args.minibatch_size),
                "policy_lr": float(args.policy_lr),
                "value_lr": float(args.value_lr),
                "clip_ratio": float(args.clip_ratio),
                "ent_coef": float(args.ent_coef),
                "vf_coef": float(args.vf_coef),
                "max_grad_norm": float(args.max_grad_norm),
                "target_kl": float(args.target_kl),
                "normalize_obs": bool(args.normalize_obs),
            },
        )

        trainer = PPOTrainer(
            domain=args.domain,
            task=args.task,
            seed=args.seed,
            device=device,
            policy_layer_sizes=tuple(args.policy_layer_sizes),
            critic_layer_sizes=tuple(args.critic_layer_sizes),
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
    elif algo == "vmpo_parallel":
        from trainers.vmpo_parallel.trainer import Trainer as VMPOParallelTrainer
        from trainers.vmpo_parallel.agent import VMPOParallelConfig

        vmpo_config = VMPOParallelConfig(
            gamma=float(args.gamma),
            policy_lr=float(args.policy_lr),
            value_lr=float(args.value_lr),
            topk_fraction=float(args.topk_fraction),
            eta_init=float(args.eta_init),
            eta_lr=float(args.eta_lr),
            epsilon_eta=float(args.epsilon_eta),
            epsilon_mu=float(args.epsilon_mu),
            epsilon_sigma=float(args.epsilon_sigma),
            alpha_lr=float(args.alpha_lr),
            max_grad_norm=float(args.max_grad_norm),
        )
        _print_config("VMPOParallelConfig", vmpo_config)

        trainer = VMPOParallelTrainer(
            domain=args.domain,
            task=args.task,
            seed=args.seed,
            device=device,
            policy_layer_sizes=tuple(args.policy_layer_sizes),
            num_envs=int(args.num_envs),
            rollout_steps=int(args.rollout_steps),
            config=vmpo_config,
        )
        trainer.train(
            total_steps=args.total_steps,
            updates_per_step=int(args.updates_per_step),
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
            eta_init=float(args.eta_init),
            eta_lr=float(args.eta_lr),
            epsilon_eta=float(args.epsilon_eta),
            epsilon_mu=float(args.epsilon_mu),
            epsilon_sigma=float(args.epsilon_sigma),
            alpha_lr=float(args.alpha_lr),
            max_grad_norm=float(args.max_grad_norm),
        )
        _print_config("VMPOConfig", vmpo_config)

        trainer = VMPOTrainer(
            domain=args.domain,
            task=args.task,
            seed=args.seed,
            device=device,
            policy_layer_sizes=tuple(args.policy_layer_sizes),
            rollout_steps=int(args.rollout_steps),
            config=vmpo_config,
        )
        trainer.train(
            total_steps=args.total_steps,
            updates_per_step=int(args.updates_per_step),
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            out_dir=args.out_dir,
        )
    elif algo == "vmpo_light":
        from trainers.vmpo_light.trainer import Trainer as VMPOLightTrainer
        from trainers.vmpo_light.agent import VMPOLightConfig

        vmpo_config = VMPOLightConfig(
            gamma=float(args.gamma),
            policy_lr=float(args.policy_lr),
            value_lr=float(args.value_lr),
            eta_init=float(args.eta_init),
            eta_lr=float(args.eta_lr),
            epsilon_eta=float(args.epsilon_eta),
        )
        _print_config("VMPOLightConfig", vmpo_config)

        trainer = VMPOLightTrainer(
            domain=args.domain,
            task=args.task,
            seed=args.seed,
            device=device,
            policy_layer_sizes=tuple(args.policy_layer_sizes),
            rollout_steps=int(args.rollout_steps),
            config=vmpo_config,
        )
        trainer.train(
            total_steps=args.total_steps,
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
            epsilon_mean=(
                None if args.epsilon_mean is None else float(args.epsilon_mean)
            ),
            epsilon_stddev=(
                None if args.epsilon_stddev is None else float(args.epsilon_stddev)
            ),
            per_dim_constraining=bool(args.per_dim_constraining),
            lambda_init=float(args.lambda_init),
            lambda_lr=float(args.lambda_lr),
            action_penalization=bool(args.action_penalization),
            epsilon_penalty=float(args.epsilon_penalty),
            max_grad_norm=float(args.max_grad_norm),
            action_samples=int(args.action_samples),
            retrace_steps=int(args.retrace_steps),
            retrace_mc_actions=int(args.retrace_mc_actions),
            retrace_lambda=float(args.retrace_lambda),
            use_retrace=bool(args.use_retrace),
        )
        _print_config("MPOConfig", mpo_config)

        trainer = MPOTrainer(
            domain=args.domain,
            task=args.task,
            seed=args.seed,
            device=device,
            policy_layer_sizes=tuple(args.policy_layer_sizes),
            critic_layer_sizes=tuple(args.critic_layer_sizes),
            replay_size=args.replay_size,
            config=mpo_config,
        )
        trainer.train(
            total_steps=args.total_steps,
            update_after=args.update_after,
            batch_size=args.batch_size,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            out_dir=args.out_dir,
            updates_per_step=int(args.updates_per_step),
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    finish_wandb()
