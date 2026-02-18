import argparse
import os
import importlib
import re
from datetime import datetime
from pprint import pformat

import torch

from utils.env import set_seed
from utils.wandb_utils import finish_wandb, init_wandb

from trainers.vmpo.trainer import VMPOTrainer
from trainers.mpo.trainer import MPOTrainer
from trainers.ppo.trainer import PPOTrainer


def _print_banner() -> None:
    print(
        "\n".join(
            [
                "=" * 79,
                "M   M  IIIII  N   N  EEEEE  RRRR   V   V   AAA   L          RRRR   L",
                "MM MM    I    NN  N  E      R   R  V   V  A   A  L          R   R  L",
                "M M M    I    N N N  EEE    RRRR   V   V  AAAAA  L          RRRR   L",
                "M   M    I    N  NN  E      R R     V V   A   A  L          R R    L",
                "M   M  IIIII  N   N  EEEEE  R  RR    V    A   A  LLLLL      R  RR  LLLLL",
                "=" * 79,
            ]
        )
    )


def _load_preset(algo: str, env_id: str) -> dict:
    module = importlib.import_module(f"hyperparameters.{algo}")
    get_fn = getattr(module, "get", None)
    if get_fn is None:
        raise RuntimeError(f"hyperparameters.{algo} must define get(env_id)")
    preset = get_fn(env_id)
    if not isinstance(preset, dict):
        raise TypeError(
            f"hyperparameters.{algo}.get must return a dict, got {type(preset)}"
        )
    return preset


def _apply_preset(args: argparse.Namespace, preset: dict) -> None:
    for key, value in preset.items():
        if (
            key == "advantage_estimator"
            and hasattr(args, "advantage_estimator")
            and getattr(args, "advantage_estimator") is not None
        ):
            continue  # Don't override if already set by CLI
        setattr(args, key, value)


def _print_resolved_args(args: argparse.Namespace) -> None:
    print("Resolved args (after hyperparameter preset):")
    print(pformat(dict(sorted(vars(args).items())), sort_dicts=True))


def _print_config(name: str, config: object) -> None:
    print(f"{name}:")
    print(pformat(config, sort_dicts=True))


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    raw = device_arg.strip().lower()
    if raw in {"mp", "metal"}:
        raw = "mps"

    try:
        device = torch.device(raw)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid --device value '{device_arg}'") from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                f"Requested --device '{device_arg}' but CUDA is not available"
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested --device '{device_arg}' but only {torch.cuda.device_count()} CUDA device(s) are visible"
            )
    elif device.type == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError(
                f"Requested --device '{device_arg}' but MPS is not available"
            )

    return device


def _slug_component(value: object) -> str:
    text = str(value).strip().replace("/", "-")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "unknown"


def _resolve_run_optimizer(algo: str, args: argparse.Namespace) -> str:
    optimizer = getattr(args, "optimizer_type", None)
    if optimizer is not None:
        text = str(optimizer).strip().lower()
        if text and text != "none":
            return text
    return "unknown"


def _resolve_run_adv_type(algo: str, args: argparse.Namespace) -> str:
    if algo == "vmpo":
        estimator = getattr(args, "advantage_estimator", None)
        if estimator is not None:
            text = str(estimator).strip().lower()
            if text and text != "none":
                return text
        return "returns"
    if algo in {"ppo", "trpo"}:
        return "gae"
    if algo == "mpo":
        return "none"
    return "unknown"


if __name__ == "__main__":
    _print_banner()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        nargs="?",
        choices=[
            "ppo",
            "vmpo",
            "mpo",
        ],
    )

    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Environment ID, e.g. dm_control/walker/walk or Humanoid-v5",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--wandb_project", type=str, default="minerva-rl")
    parser.add_argument("--wandb_entity", type=str, default="adrian-research")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument(
        "--advantage_estimator",
        type=str,
        default=None,
        help="Override advantage estimator (returns, gae, dae)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: cpu, cuda, cuda:0, cuda:1, mps (or mp alias)",
    )
    parser.add_argument(
        "--optimizer_type",
        "--optimizer",
        dest="optimizer_type",
        type=str.lower,
        choices=["adam", "sgd"],
        default=None,
        help="Optimizer override for PPO/VMPO/MPO (adam or sgd)",
    )
    parser.add_argument(
        "--sgd_momentum",
        type=float,
        default=None,
        help="SGD momentum override when --optimizer_type=sgd",
    )
    args = parser.parse_args()

    cli_device_override = args.device
    cli_optimizer_override = args.optimizer_type
    cli_sgd_momentum_override = args.sgd_momentum

    algo = args.command
    if algo is None:
        raise ValueError("Missing algorithm subcommand")

    preset = _load_preset(algo, args.env)
    _apply_preset(args, preset)

    if cli_device_override is not None:
        args.device = cli_device_override
    if cli_optimizer_override is not None:
        args.optimizer_type = cli_optimizer_override
    if cli_sgd_momentum_override is not None:
        args.sgd_momentum = cli_sgd_momentum_override

    _print_resolved_args(args)

    set_seed(args.seed)
    device = _resolve_device(args.device)
    print(f"Using device: {device}")

    env_slug = _slug_component(args.env)
    run_optimizer = _slug_component(_resolve_run_optimizer(algo, args))
    run_adv_type = _slug_component(_resolve_run_adv_type(algo, args))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = (
        f"{algo}_{env_slug}-{run_optimizer}-{run_adv_type}_seed{args.seed}_{timestamp}"
    )
    args.out_dir = os.path.join(args.out_dir, algo, run_name)

    os.makedirs(args.out_dir, exist_ok=True)

    group = args.wandb_group or f"{algo}-{args.env}"
    init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=group,
        name=run_name,
        config=vars(args),
        monitor_gym=False,
        save_code=True,
    )

    if algo == "ppo":

        _print_config(
            "PPO config",
            {
                "rollout_steps": int(args.rollout_steps),
                "gamma": float(args.gamma),
                "gae_lambda": float(args.gae_lambda),
                "update_epochs": int(args.update_epochs),
                "minibatch_size": int(args.minibatch_size),
                "policy_lr": float(args.policy_lr),
                "clip_ratio": float(args.clip_ratio),
                "ent_coef": float(args.ent_coef),
                "vf_coef": float(args.vf_coef),
                "max_grad_norm": float(args.max_grad_norm),
                "target_kl": float(args.target_kl),
                "norm_adv": bool(args.norm_adv),
                "clip_vloss": bool(args.clip_vloss),
                "anneal_lr": bool(args.anneal_lr),
                "normalize_obs": bool(args.normalize_obs),
                "optimizer_type": str(args.optimizer_type),
                "sgd_momentum": float(args.sgd_momentum),
            },
        )

        trainer = PPOTrainer(
            env_id=args.env,
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
            clip_ratio=float(args.clip_ratio),
            ent_coef=float(args.ent_coef),
            vf_coef=float(args.vf_coef),
            max_grad_norm=float(args.max_grad_norm),
            target_kl=float(args.target_kl),
            norm_adv=bool(args.norm_adv),
            clip_vloss=bool(args.clip_vloss),
            anneal_lr=bool(args.anneal_lr),
            normalize_obs=bool(args.normalize_obs),
            num_envs=int(args.num_envs),
            optimizer_type=str(args.optimizer_type),
            sgd_momentum=float(args.sgd_momentum),
        )
        trainer.train(
            total_steps=args.total_steps,
            out_dir=args.out_dir,
        )
    elif algo == "vmpo":

        vmpo_params = {
            "gamma": float(args.gamma),
            "advantage_estimator": str(getattr(args, "advantage_estimator", "returns")),
            "gae_lambda": float(getattr(args, "gae_lambda", 0.95)),
            "policy_lr": float(args.policy_lr),
            "value_lr": float(args.value_lr),
            "topk_fraction": float(args.topk_fraction),
            "temperature_init": float(args.temperature_init),
            "temperature_lr": float(args.temperature_lr),
            "epsilon_eta": float(args.epsilon_eta),
            "epsilon_mu": float(args.epsilon_mu),
            "epsilon_sigma": float(args.epsilon_sigma),
            "alpha_lr": float(args.alpha_lr),
            "max_grad_norm": float(args.max_grad_norm),
            "normalize_advantages": bool(args.normalize_advantages),
            "optimizer_type": str(args.optimizer_type),
            "sgd_momentum": float(args.sgd_momentum),
            "shared_encoder": bool(args.shared_encoder),
            "m_steps": int(args.m_steps),
        }
        _print_config("VMPO config", vmpo_params)

        trainer = VMPOTrainer(
            env_id=args.env,
            seed=args.seed,
            device=device,
            policy_layer_sizes=tuple(args.policy_layer_sizes),
            value_layer_sizes=tuple(args.value_layer_sizes),
            rollout_steps=int(args.rollout_steps),
            gamma=float(args.gamma),
            advantage_estimator=str(getattr(args, "advantage_estimator", "returns")),
            gae_lambda=float(getattr(args, "gae_lambda", 0.95)),
            policy_lr=float(args.policy_lr),
            value_lr=float(args.value_lr),
            topk_fraction=float(args.topk_fraction),
            temperature_init=float(args.temperature_init),
            temperature_lr=float(args.temperature_lr),
            epsilon_eta=float(args.epsilon_eta),
            epsilon_mu=float(args.epsilon_mu),
            epsilon_sigma=float(args.epsilon_sigma),
            alpha_lr=float(args.alpha_lr),
            max_grad_norm=float(args.max_grad_norm),
            normalize_advantages=bool(args.normalize_advantages),
            optimizer_type=str(args.optimizer_type),
            sgd_momentum=float(args.sgd_momentum),
            shared_encoder=bool(args.shared_encoder),
            num_envs=int(args.num_envs),
            m_steps=int(args.m_steps),
        )
        trainer.train(
            total_steps=args.total_steps,
            out_dir=args.out_dir,
        )
    elif algo == "mpo":

        mpo_params = {
            "gamma": float(args.gamma),
            "target_networks_update_period": int(args.target_networks_update_period),
            "policy_lr": float(args.policy_lr),
            "q_lr": float(args.q_lr),
            "temperature_init": float(args.temperature_init),
            "temperature_lr": float(args.temperature_lr),
            "kl_epsilon": float(args.kl_epsilon),
            "mstep_kl_epsilon": float(args.mstep_kl_epsilon),
            "lambda_init": float(args.lambda_init),
            "lambda_lr": float(args.lambda_lr),
            "epsilon_penalty": float(args.epsilon_penalty),
            "max_grad_norm": float(args.max_grad_norm),
            "action_samples": int(args.action_samples),
            "retrace_steps": int(args.retrace_steps),
            "retrace_mc_actions": int(args.retrace_mc_actions),
            "retrace_lambda": float(args.retrace_lambda),
            "use_retrace": bool(args.use_retrace),
            "optimizer_type": str(args.optimizer_type),
            "sgd_momentum": float(args.sgd_momentum),
            "m_steps": int(args.m_steps),
        }
        _print_config("MPO config", mpo_params)

        trainer = MPOTrainer(
            env_id=args.env,
            seed=args.seed,
            device=device,
            policy_layer_sizes=tuple(args.policy_layer_sizes),
            critic_layer_sizes=tuple(args.critic_layer_sizes),
            replay_size=args.replay_size,
            gamma=float(args.gamma),
            target_networks_update_period=int(args.target_networks_update_period),
            policy_lr=float(args.policy_lr),
            q_lr=float(args.q_lr),
            temperature_init=float(args.temperature_init),
            temperature_lr=float(args.temperature_lr),
            kl_epsilon=float(args.kl_epsilon),
            mstep_kl_epsilon=float(args.mstep_kl_epsilon),
            lambda_init=float(args.lambda_init),
            lambda_lr=float(args.lambda_lr),
            epsilon_penalty=float(args.epsilon_penalty),
            max_grad_norm=float(args.max_grad_norm),
            action_samples=int(args.action_samples),
            retrace_steps=int(args.retrace_steps),
            retrace_mc_actions=int(args.retrace_mc_actions),
            retrace_lambda=float(args.retrace_lambda),
            use_retrace=bool(args.use_retrace),
            optimizer_type=str(args.optimizer_type),
            sgd_momentum=float(args.sgd_momentum),
            m_steps=int(args.m_steps),
        )
        trainer.train(
            total_steps=args.total_steps,
            update_after=args.update_after,
            batch_size=args.batch_size,
            out_dir=args.out_dir,
        )

    finish_wandb()
