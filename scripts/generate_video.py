import argparse
import importlib
import os
import platform
from pprint import pformat

import torch

from minerva.utils.video import VideoRenderConfig, find_latest_checkpoint, render_policy_video


def _default_mujoco_gl_backend() -> str:
    """Choose a sane default backend by platform.

    - macOS: `glfw` (EGL is typically unavailable)
    - Linux: `egl` (common for headless rendering)
    - Other: `glfw`
    """
    system = platform.system().lower()
    if system == "darwin":
        return "glfw"
    if system == "linux":
        return "egl"
    return "glfw"


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


def _print_resolved_args(args: argparse.Namespace) -> None:
    print("Resolved args (after hyperparameter preset):")
    print(pformat(dict(sorted(vars(args).items())), sort_dicts=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a rollout video from a trained checkpoint."
    )
    parser.add_argument(
        "algo",
        choices=["ppo", "vmpo", "mpo"],
        help="Which algorithm's checkpoint to load.",
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
    )
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
            "for the requested env under --out_dir/{algo}/<run_name>/."
        ),
    )
    parser.add_argument(
        "--video_out",
        type=str,
        default=None,
        help="Output video path. Defaults to videos/{algo}-{env}.mp4",
    )
    parser.add_argument(
        "--gif_out",
        type=str,
        default=None,
        help="Output gif path. Defaults to same basename as --video_out with .gif",
    )
    parser.add_argument("--video_max_steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    preset = _load_preset(args.algo, args.env)
    for key, value in preset.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    _print_resolved_args(args)

    os.environ.setdefault("MUJOCO_GL", _default_mujoco_gl_backend())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_dir = args.env.replace("/", "-")
    algo_out_dir = os.path.join(args.out_dir, args.algo)

    checkpoint_path = (
        str(args.checkpoint)
        if args.checkpoint is not None
        else find_latest_checkpoint(algo_out_dir, args.algo, args.env)
    )
    out_path = (
        str(args.video_out)
        if args.video_out is not None
        else os.path.join("videos", f"{args.algo}-{env_dir}.mp4")
    )
    gif_out_path = (
        str(args.gif_out)
        if args.gif_out is not None
        else os.path.splitext(out_path)[0] + ".gif"
    )

    saved_path, saved_gif_path, n_frames = render_policy_video(
        checkpoint_path=checkpoint_path,
        algo=args.algo,
        env_id=args.env,
        out_path=out_path,
        gif_out_path=gif_out_path,
        seed=int(args.seed),
        config=VideoRenderConfig(
            max_steps=int(args.video_max_steps),
            fps=int(args.fps),
        ),
        policy_layer_sizes=tuple(args.policy_layer_sizes),
        value_layer_sizes=(
            tuple(args.value_layer_sizes)
            if hasattr(args, "value_layer_sizes")
            else None
        ),
        shared_encoder=bool(getattr(args, "shared_encoder", False)),
        device=device,
    )
    print(f"Saved video to: {saved_path}")
    print(f"Saved gif to: {saved_gif_path}")
    print(f"Frames: {n_frames}")


if __name__ == "__main__":
    main()
