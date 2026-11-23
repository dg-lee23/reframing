"""Reframing + CameraCtrl. Combines a camera move from CameraCtrl with a time-varying
reframing of the source video, so the camera motion and the zoom/pan go together.

    method=ours     warp the source latent over time (latent warping)
    method=feature  time-varying feature transform, last frame only
    method=camera   CameraCtrl + motion transfer, no reframing
"""
import argparse
import os

import torch
from omegaconf import OmegaConf

from invert import inversion_path
from principled_reframing.camera import load_cameractrl, pose_features
from principled_reframing.pipeline import build_pipeline, enable_gradient_checkpointing
from principled_reframing.utils.util import save_video, set_all_seed


MOVES = {
    "pan_right": dict(
        camera=dict(mode="pan", direction="right", max_translation=6.0),
        ours=dict(mode="pan", direction="right", shift_range=(0.0, 0.33)),
        feature=dict(transformation="time", direction="right", factor=3),
    ),
    "zoom_in": dict(
        camera=dict(mode="dolly", direction="in", max_translation=8.0),
        ours=dict(mode="zoom", crop_range=(1.0, 0.65)),
        feature=dict(transformation="time", factor=2.366),
    ),
    "zoom_out": dict(
        camera=dict(mode="dolly", direction="out", max_translation=8.0),
        ours=dict(mode="zoom", crop_range=(1.0, 1.53846)),
        feature=dict(transformation="time", factor=0.4225),
    ),
}


def main(args):
    config = OmegaConf.load(args.config)
    pipeline = build_pipeline(config, use_xformers=not args.no_xformers)
    if args.grad_ckpt:
        enable_gradient_checkpointing(pipeline, config.temp_guidance.blocks)
    pose_encoder = load_cameractrl(pipeline, args.cameractrl_ckpt)

    moves = list(MOVES) if args.move == "all" else [args.move]
    for move in moves:
        preset = MOVES[move]
        seed = config.get("seed", args.seed)
        set_all_seed(seed)
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)

        features = pose_features(pose_encoder, config.L, config.H, config.W, **preset["camera"])
        kwargs = {}
        if args.method == "ours":
            kwargs["latent_warp"] = preset["ours"]
        elif args.method == "feature":
            kwargs.update(preset["feature"], endpoints=True)

        video = pipeline(config, inversion_path(args.config, args.inversion_dir), generator=generator,
                         pose_features=features, **kwargs)

        name = os.path.splitext(os.path.basename(args.config))[0]
        out = os.path.join(args.out_dir, f"{name}_{move}_{args.method}.gif")
        os.makedirs(args.out_dir, exist_ok=True)
        save_video(video[0], out)
        print(f"saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--move", choices=list(MOVES) + ["all"], default="all")
    parser.add_argument("--method", choices=["ours", "feature", "camera"], default="ours")
    parser.add_argument("--out-dir", default="results/cameractrl")
    parser.add_argument("--cameractrl-ckpt", default="models/CameraCtrl/CameraCtrl.ckpt")
    parser.add_argument("--inversion-dir", default="inversion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-ckpt", action="store_true", help="use less GPU memory (slower)")
    parser.add_argument("--no-xformers", action="store_true")
    main(parser.parse_args())
