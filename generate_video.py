import argparse
import importlib
import os
from pprint import pformat

import torch

from utils.video import VideoRenderConfig, find_latest_checkpoint, render_policy_video


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


def _print_resolved_args(args: argparse.Namespace) -> None:
    print("Resolved args (after hyperparameter preset):")
    print(pformat(dict(sorted(vars(args).items())), sort_dicts=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a rollout video from a trained checkpoint."
    )
    parser.add_argument(
        "algo",
        choices=["ppo", "vmpo", "mpo", "vmpo_parallel", "vmpo_light"],
        help="Which algorithm's checkpoint to load.",
    )
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--out_dir",
        type=str,
        default="checkpoints",
        help="Base checkpoints directory (defaults to checkpoints/).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Optional checkpoint path (.pt). If omitted, uses the latest checkpoint "
            "found under --out_dir/{algo}/{domain}/{task}."
        ),
    )
    parser.add_argument(
        "--video_out",
        type=str,
        default=None,
        help="Output video path. Defaults to videos/{algo}-{domain}-{task}.mp4",
    )
    parser.add_argument("--video_max_steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    preset = _load_preset(args.algo, args.domain, args.task)
    for key, value in preset.items():
        # Only set preset values for fields that don't already exist.
        # This keeps the script flexible even if we later add CLI overrides.
        if not hasattr(args, key):
            setattr(args, key, value)

    _print_resolved_args(args)

    os.environ.setdefault("MUJOCO_GL", "egl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_out_dir = os.path.join(args.out_dir, args.algo, args.domain, args.task)

    checkpoint_path = (
        str(args.checkpoint)
        if args.checkpoint is not None
        else find_latest_checkpoint(run_out_dir, args.algo)
    )
    out_path = (
        str(args.video_out)
        if args.video_out is not None
        else os.path.join("videos", f"{args.algo}-{args.domain}-{args.task}.mp4")
    )

    saved_path, n_frames = render_policy_video(
        checkpoint_path=checkpoint_path,
        algo=args.algo,
        domain=args.domain,
        task=args.task,
        out_path=out_path,
        seed=int(args.seed),
        config=VideoRenderConfig(
            max_steps=int(args.video_max_steps),
            fps=int(args.fps),
        ),
        policy_layer_sizes=tuple(args.policy_layer_sizes),
        device=device,
    )
    print(f"Saved video to: {saved_path}")
    print(f"Frames: {n_frames}")


if __name__ == "__main__":
    main()
