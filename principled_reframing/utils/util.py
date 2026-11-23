import random

import decord
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open

from .convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from .convert_lora_safetensor_to_diffusers import load_diffusers_lora

decord.bridge.set_bridge('torch')


def set_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    random.seed(seed + 4)


def load_weights(pipeline, motion_module_path="", dreambooth_model_path="",
                 adapter_lora_path="", adapter_lora_scale=1.0):
    if motion_module_path:
        print(f"load motion module from {motion_module_path}")
        state_dict = torch.load(motion_module_path, map_location="cpu")
        state_dict = state_dict.get("state_dict", state_dict)
        state_dict = {k: v for k, v in state_dict.items() if "motion_modules." in k}
        _, unexpected = pipeline.unet.load_state_dict(state_dict, strict=False)
        assert len(unexpected) == 0

    if dreambooth_model_path:
        print(f"load dreambooth model from {dreambooth_model_path}")
        if dreambooth_model_path.endswith(".safetensors"):
            state_dict = {}
            with safe_open(dreambooth_model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        else:
            state_dict = torch.load(dreambooth_model_path, map_location="cpu")

        pipeline.vae.load_state_dict(convert_ldm_vae_checkpoint(state_dict, pipeline.vae.config))
        pipeline.unet.load_state_dict(convert_ldm_unet_checkpoint(state_dict, pipeline.unet.config), strict=False)
        pipeline.text_encoder = convert_ldm_clip_checkpoint(
            state_dict, config=pipeline.text_encoder.config
        ).to(dtype=pipeline.unet.dtype)
        del state_dict

    if adapter_lora_path:
        print(f"load domain adapter from {adapter_lora_path}")
        state_dict = torch.load(adapter_lora_path, map_location="cpu")
        state_dict = state_dict.get("state_dict", state_dict)
        pipeline = load_diffusers_lora(pipeline, state_dict, alpha=adapter_lora_scale)

    return pipeline


def load_video(path, height, width, num_frames):
    """Read up to `num_frames` frames (every 1-3 frames), resize, and scale to [-1, 1]."""
    vr = decord.VideoReader(path)
    interval = min(len(vr) // num_frames, 3)
    index = list(range(0, len(vr), interval))[:num_frames]

    video = rearrange(vr.get_batch(index), "f h w c -> f c h w")
    video = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=True)
    return video / 127.5 - 1.0


def save_video(video, path, fps=8):
    """video: (c, f, h, w) in [0, 1]. Saves .mp4 or .gif depending on the extension."""
    frames = (rearrange(video, "c f h w -> f h w c") * 255).astype(np.uint8)
    if path.endswith(".gif"):
        imageio.mimwrite(path, frames, fps=fps)
    else:
        imageio.mimwrite(path, frames, fps=fps, quality=9)
