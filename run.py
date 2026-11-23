import argparse
import os

import torch
from omegaconf import OmegaConf

from invert import inversion_path
from principled_reframing.pipeline import build_pipeline, enable_gradient_checkpointing
from principled_reframing.reframing import TRANSFORMATIONS
from principled_reframing.utils.util import save_video, set_all_seed


def output_name(args):
    parts = [os.path.splitext(os.path.basename(args.config))[0], args.transform or "none"]
    parts += [str(x) for x in (args.direction, args.factor) if x is not None]
    return "_".join(parts) + ".mp4"


def main(args):
    config = OmegaConf.load(args.config)
    pipeline = build_pipeline(config, use_xformers=not args.no_xformers)
    if args.grad_ckpt:
        enable_gradient_checkpointing(pipeline, config.temp_guidance.blocks)

    seed = config.get("seed", args.seed)
    set_all_seed(seed)
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    video = pipeline(
        config,
        inversion_path(args.config, args.inversion_dir),
        generator=generator,
        transformation=args.transform,
        factor=args.factor,
        direction=args.direction,
    )

    out = args.out or os.path.join("results", output_name(args))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    save_video(video[0], out)
    print(f"saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--transform", choices=TRANSFORMATIONS, default=None)
    parser.add_argument("--direction", choices=["up", "down", "left", "right"], default=None)
    parser.add_argument("--factor", type=float, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--inversion-dir", default="inversion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-ckpt", action="store_true", help="use less GPU memory (slower)")
    parser.add_argument("--no-xformers", action="store_true")
    main(parser.parse_args())
