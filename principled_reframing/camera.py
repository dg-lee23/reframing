import math

import torch
from einops import rearrange

from .models.pose_adaptor import CameraPoseEncoder


POSE_ENCODER_KWARGS = dict(
    downscale_factor=8, channels=[320, 640, 1280, 1280], nums_rb=2, cin=384,
    ksize=1, sk=True, use_conv=False, compression_factor=1,
    temporal_attention_nhead=8, attention_block_types=["Temporal_Self"],
    temporal_position_encoding=True, temporal_position_encoding_max_len=16,
)


def ray_condition(K, c2w, H, W, device):
    # K: (B, V, 4) fx, fy, cx, cy in pixels; c2w: (B, V, 4, 4) -> plucker (B, V, H, W, 6)
    B, V = K.shape[:2]
    j, i = torch.meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
        indexing='ij',
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5

    fx, fy, cx, cy = K.chunk(4, dim=-1)
    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)
    rays_o = c2w[..., :3, 3]
    rays_o = rays_o[:, :, None].expand_as(rays_d)
    rays_dxo = torch.cross(rays_o, rays_d, dim=-1)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    return plucker.reshape(B, V, H, W, 6)


def camera_trajectory(num_frames, mode, direction, max_translation, fov_deg=55.0):
    """Linear camera translation. mode='pan' moves along x (left/right),
    mode='dolly' moves along z (in/out)."""
    if mode == 'pan':
        axis, sign = 0, (1.0 if direction == 'right' else -1.0)
    elif mode == 'dolly':
        axis, sign = 2, (-1.0 if direction == 'out' else 1.0)
    else:
        raise ValueError(f"unknown camera mode {mode!r}")

    c2w = torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1)
    for t in range(num_frames):
        progress = t / (num_frames - 1) if num_frames > 1 else 0.0
        c2w[t, axis, 3] = sign * max_translation * progress

    focal = 0.5 / math.tan(math.radians(fov_deg) / 2)
    K = torch.tensor([focal, focal, 0.5, 0.5]).unsqueeze(0).repeat(num_frames, 1)
    return K.unsqueeze(0), c2w.unsqueeze(0)


def plucker_embedding(num_frames, H, W, mode, direction, max_translation, fov_deg=55.0):
    K, c2w = camera_trajectory(num_frames, mode, direction, max_translation, fov_deg)
    K[..., 0] *= W
    K[..., 1] *= H
    K[..., 2] *= W
    K[..., 3] *= H
    plucker = ray_condition(K, c2w, H, W, device='cpu')
    return plucker.permute(0, 4, 1, 2, 3).contiguous()  # (1, 6, F, H, W)


def load_cameractrl(pipeline, ckpt_path, device="cuda", dtype=torch.float16):
    """Load CameraCtrl's pose encoder and copy its qkv_merge weights into the UNet."""
    ckpt = torch.load(ckpt_path, map_location="cpu")

    pose_encoder = CameraPoseEncoder(**POSE_ENCODER_KWARGS)
    pose_encoder.load_state_dict(ckpt["pose_encoder_state_dict"], strict=True)
    pose_encoder = pose_encoder.to(device=device, dtype=dtype).eval()

    weights = ckpt["attention_processor_state_dict"]
    for kind in ("down_blocks", "up_blocks"):
        for block_id, block in enumerate(getattr(pipeline.unet, kind)):
            for i, motion_module in enumerate(getattr(block, "motion_modules", [])):
                if motion_module is None:
                    continue
                attn = motion_module.temporal_transformer.transformer_blocks[0].attention_blocks[0]
                if attn.qkv_merge is None:
                    continue
                prefix = (f"{kind}.{block_id}.motion_modules.{i}.temporal_transformer."
                          f"transformer_blocks.0.attention_blocks.0.processor.qkv_merge.")
                with torch.no_grad():
                    attn.qkv_merge.weight.copy_(weights[prefix + "weight"].to(dtype=dtype))
                    attn.qkv_merge.bias.copy_(weights[prefix + "bias"].to(dtype=dtype))
    return pose_encoder


@torch.no_grad()
def pose_features(pose_encoder, num_frames, H, W, mode, direction, max_translation, fov_deg=55.0,
                  device="cuda", dtype=torch.float16):
    plucker = plucker_embedding(num_frames, H, W, mode, direction, max_translation, fov_deg)
    plucker = plucker.to(device=device, dtype=dtype)
    features = pose_encoder(plucker)
    return [rearrange(x, "(b f) c h w -> b c f h w", b=plucker.shape[0]) for x in features]
