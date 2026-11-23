import argparse
import os

import torch
from omegaconf import OmegaConf

from principled_reframing.pipeline import build_pipeline
from principled_reframing.utils.util import load_video, set_all_seed


def inversion_path(config_path, inversion_dir="inversion"):
    return os.path.join(inversion_dir, os.path.splitext(os.path.basename(config_path))[0] + ".pkl")


@torch.no_grad()
def main(args):
    config = OmegaConf.load(args.config)
    pipeline = build_pipeline(config, use_xformers=not args.no_xformers)
    set_all_seed(args.seed)

    video = load_video(config.video_path, config.H, config.W, config.L)
    save_path = inversion_path(args.config, args.inversion_dir)
    os.makedirs(args.inversion_dir, exist_ok=True)
    pipeline.invert(video, config.inversion_prompt, config.num_inference_step, save_path)
    print(f"saved {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--inversion-dir", default="inversion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-xformers", action="store_true")
    main(parser.parse_args())
