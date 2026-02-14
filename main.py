import argparse
import os
import importlib
from pprint import pformat

import torch

from utils.env import set_seed
from utils.wandb_utils import finish_wandb, init_wandb

from trainers.vmpo.trainer import VMPOTrainer
from trainers.mpo.trainer import MPOTrainer
from trainers.ppo.trainer import PPOTrainer
from trainers.ppo_trxl.trainer import PPOTRxLTrainer
from trainers.trpo.trainer import TRPOTrainer


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


if __name__ == "__main__":
    _print_banner()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        nargs="?",
        choices=[
            "ppo",
            "ppo_trxl",
            "trpo",
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
        help="Optimizer override for PPO/TRPO/VMPO/MPO (adam or sgd)",
    )
    parser.add_argument(
        "--sgd_momentum",
        type=float,
        default=None,
        help="SGD momentum override when --optimizer_type=sgd",
    )
    parser.add_argument(
        "--generate_video",
        action="store_true",
        help="Capture training videos with RecordVideo on env 0 and upload via wandb.",
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

    args.out_dir = os.path.join(args.out_dir, algo, args.env.replace("/", "-"))

    os.makedirs(args.out_dir, exist_ok=True)

    group = args.wandb_group or f"{algo}-{args.env}"
    run_name = f"{algo}-{args.env}-seed{args.seed}"
    init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=group,
        name=run_name,
        config=vars(args),
        monitor_gym=bool(args.generate_video),
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
                "value_lr": float(args.value_lr),
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
            value_lr=float(args.value_lr),
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
            capture_video=bool(args.generate_video),
            run_name=run_name,
        )
        trainer.train(
            total_steps=args.total_steps,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            out_dir=args.out_dir,
        )
    elif algo == "ppo_trxl":
        target_kl = getattr(args, "target_kl", None)
        trxl_params = {
            "num_envs": int(args.num_envs),
            "num_steps": int(args.num_steps),
            "num_minibatches": int(args.num_minibatches),
            "update_epochs": int(args.update_epochs),
            "init_lr": float(args.init_lr),
            "final_lr": float(args.final_lr),
            "anneal_steps": int(args.anneal_steps),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "norm_adv": bool(args.norm_adv),
            "clip_coef": float(args.clip_coef),
            "clip_vloss": bool(args.clip_vloss),
            "init_ent_coef": float(args.init_ent_coef),
            "final_ent_coef": float(args.final_ent_coef),
            "vf_coef": float(args.vf_coef),
            "max_grad_norm": float(args.max_grad_norm),
            "target_kl": None if target_kl is None else float(target_kl),
            "trxl_num_layers": int(args.trxl_num_layers),
            "trxl_num_heads": int(args.trxl_num_heads),
            "trxl_dim": int(args.trxl_dim),
            "trxl_memory_length": int(args.trxl_memory_length),
            "trxl_positional_encoding": str(args.trxl_positional_encoding),
            "reconstruction_coef": float(args.reconstruction_coef),
        }
        _print_config("PPO-TRxL config", trxl_params)

        trainer = PPOTRxLTrainer(
            env_id=args.env,
            seed=args.seed,
            device=device,
            num_envs=int(args.num_envs),
            num_steps=int(args.num_steps),
            num_minibatches=int(args.num_minibatches),
            update_epochs=int(args.update_epochs),
            init_lr=float(args.init_lr),
            final_lr=float(args.final_lr),
            anneal_steps=int(args.anneal_steps),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            norm_adv=bool(args.norm_adv),
            clip_coef=float(args.clip_coef),
            clip_vloss=bool(args.clip_vloss),
            init_ent_coef=float(args.init_ent_coef),
            final_ent_coef=float(args.final_ent_coef),
            vf_coef=float(args.vf_coef),
            max_grad_norm=float(args.max_grad_norm),
            target_kl=None if target_kl is None else float(target_kl),
            trxl_num_layers=int(args.trxl_num_layers),
            trxl_num_heads=int(args.trxl_num_heads),
            trxl_dim=int(args.trxl_dim),
            trxl_memory_length=int(args.trxl_memory_length),
            trxl_positional_encoding=str(args.trxl_positional_encoding),
            reconstruction_coef=float(args.reconstruction_coef),
            capture_video=bool(args.generate_video),
            run_name=run_name,
        )
        trainer.train(
            total_steps=int(args.total_steps),
            eval_interval=int(args.eval_interval),
            save_interval=int(args.save_interval),
            out_dir=args.out_dir,
        )
    elif algo == "trpo":
        trpo_params = {
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "target_kl": float(args.target_kl),
            "cg_iters": int(args.cg_iters),
            "cg_damping": float(args.cg_damping),
            "backtrack_coeff": float(args.backtrack_coeff),
            "backtrack_iters": int(args.backtrack_iters),
            "value_lr": float(args.value_lr),
            "value_epochs": int(args.value_epochs),
            "value_minibatch_size": int(args.value_minibatch_size),
            "max_grad_norm": float(args.max_grad_norm),
            "normalize_advantages": bool(args.normalize_advantages),
            "optimizer_type": str(args.optimizer_type),
            "sgd_momentum": float(args.sgd_momentum),
        }
        _print_config("TRPO config", trpo_params)

        trainer = TRPOTrainer(
            env_id=args.env,
            seed=args.seed,
            device=device,
            policy_layer_sizes=tuple(args.policy_layer_sizes),
            critic_layer_sizes=tuple(args.critic_layer_sizes),
            rollout_steps=int(args.rollout_steps),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            target_kl=float(args.target_kl),
            cg_iters=int(args.cg_iters),
            cg_damping=float(args.cg_damping),
            backtrack_coeff=float(args.backtrack_coeff),
            backtrack_iters=int(args.backtrack_iters),
            value_lr=float(args.value_lr),
            value_epochs=int(args.value_epochs),
            value_minibatch_size=int(args.value_minibatch_size),
            max_grad_norm=float(args.max_grad_norm),
            normalize_advantages=bool(args.normalize_advantages),
            optimizer_type=str(args.optimizer_type),
            sgd_momentum=float(args.sgd_momentum),
            normalize_obs=bool(args.normalize_obs),
            num_envs=int(args.num_envs),
            capture_video=bool(args.generate_video),
            run_name=run_name,
        )
        trainer.train(
            total_steps=int(args.total_steps),
            eval_interval=int(args.eval_interval),
            save_interval=int(args.save_interval),
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
            num_envs=int(args.num_envs),
            capture_video=bool(args.generate_video),
            run_name=run_name,
        )
        trainer.train(
            total_steps=args.total_steps,
            updates_per_step=int(args.updates_per_step),
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            out_dir=args.out_dir,
        )
    elif algo == "mpo":

        mpo_params = {
            "gamma": float(args.gamma),
            "target_policy_update_period": int(args.target_policy_update_period),
            "target_critic_update_period": int(args.target_critic_update_period),
            "policy_lr": float(args.policy_lr),
            "q_lr": float(args.q_lr),
            "temperature_init": float(args.temperature_init),
            "temperature_lr": float(args.temperature_lr),
            "kl_epsilon": float(args.kl_epsilon),
            "mstep_kl_epsilon": float(args.mstep_kl_epsilon),
            "per_dim_constraining": bool(args.per_dim_constraining),
            "lambda_init": float(args.lambda_init),
            "lambda_lr": float(args.lambda_lr),
            "action_penalization": bool(args.action_penalization),
            "epsilon_penalty": float(args.epsilon_penalty),
            "max_grad_norm": float(args.max_grad_norm),
            "action_samples": int(args.action_samples),
            "retrace_steps": int(args.retrace_steps),
            "retrace_mc_actions": int(args.retrace_mc_actions),
            "retrace_lambda": float(args.retrace_lambda),
            "use_retrace": bool(args.use_retrace),
            "optimizer_type": str(args.optimizer_type),
            "sgd_momentum": float(args.sgd_momentum),
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
            target_policy_update_period=int(args.target_policy_update_period),
            target_critic_update_period=int(args.target_critic_update_period),
            policy_lr=float(args.policy_lr),
            q_lr=float(args.q_lr),
            temperature_init=float(args.temperature_init),
            temperature_lr=float(args.temperature_lr),
            kl_epsilon=float(args.kl_epsilon),
            mstep_kl_epsilon=float(args.mstep_kl_epsilon),
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
            optimizer_type=str(args.optimizer_type),
            sgd_momentum=float(args.sgd_momentum),
            capture_video=bool(args.generate_video),
            run_name=run_name,
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
