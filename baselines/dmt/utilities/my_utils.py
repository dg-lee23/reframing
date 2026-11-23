import torch
import torch.nn.functional as F
import kornia
from einops import rearrange, parse_shape

def _standardize_shape(tensor, pattern):
    """
    Internal helper to convert any input pattern to (h, w, f, dim).
    Returns the standardized tensor and a function to restore the original shape.
    """
    known_dims = parse_shape(tensor, pattern)

    if 'hw' in known_dims and 'h' not in known_dims:
        res = int(known_dims['hw']**0.5)
        h = w = res
    else:
        h, w = known_dims['h'], known_dims['w']

    standard_pattern = 'h w f dim'

    standardized = rearrange(tensor, f'{pattern} -> {standard_pattern}', h=h, w=w)

    def restore(out_tensor):
        return rearrange(out_tensor, f'{standard_pattern} -> {pattern}', **known_dims)

    return standardized, restore, (h, w)

def shift_tensor(tensor, pattern, shift=None, fill=0):
    standard, restore, (h, w) = _standardize_shape(tensor, pattern)

    if shift is None: shift = (-h // 4, 0)
    sx, sy = shift

    out = torch.full_like(standard, fill)

    src_h_slice = slice(max(0, -sx), min(h, h - sx))
    src_w_slice = slice(max(0, -sy), min(w, w - sy))
    dst_h_slice = slice(max(0, sx), min(h, h + sx))
    dst_w_slice = slice(max(0, sy), min(w, w + sy))

    out[dst_h_slice, dst_w_slice] = standard[src_h_slice, src_w_slice]

    return restore(out)

def resize_tensor(tensor, pattern, scale_factor, mode='center'):
    """
    Interpolates by scale_factor, then pads/crops to maintain the ORIGINAL resolution.
    If scale_factor > 1: Interpolates up -> Crops.
    If scale_factor < 1: Interpolates down -> Pads.
    """
    standard, restore, (h, w) = _standardize_shape(tensor, pattern)
    f, dim = standard.shape[2], standard.shape[3]

    # 1. Interpolate to new temporary size
    scale_factor = (scale_factor**0.5)
    target_h, target_w = int(h * scale_factor), int(w * scale_factor)
    working = rearrange(standard, 'h w f d -> 1 (f d) h w').float()
    working = F.interpolate(working, size=(target_h, target_w), mode='bicubic', align_corners=True)

    # 2. Alignment: Get back to original (h, w)
    if scale_factor > 1.0:
        # CROP logic: We are bigger than original, need to cut down
        start_h = (target_h - h) // 2 if 'center' in mode else 0
        start_w = (target_w - w) // 2 if 'center' in mode else 0
        working = working[:, :, start_h:start_h+h, start_w:start_w+w]

    elif scale_factor < 1.0:
        # PAD logic: We are smaller than original, need to fill space
        diff_h, diff_w = h - target_h, w - target_w

        ph0, ph1 = diff_h // 2, diff_h - (diff_h // 2)
        pw0, pw1 = diff_w // 2, diff_w - (diff_w // 2)

        working = F.pad(working, (pw0, pw1, ph0, ph1), value=0)

    # 3. Restore shape
    out = rearrange(working, '1 (f d) h w -> h w f d', f=f, d=dim).to(tensor.dtype)
    return restore(out)

def warp_tensor(tensor, pattern, warp_dir='right', warp_scale=1.0):
    standard, restore, (h, w) = _standardize_shape(tensor, pattern)
    f, dim = standard.shape[2], standard.shape[3]

    d = int(h * 0.2)
    dst_pts = torch.tensor([[[0, 0], [h-1, 0], [h-1, h-1], [0, h-1]]], dtype=torch.float, device=tensor.device)

    if warp_dir == 'right':
        src_pts = torch.tensor([[[0, 0], [h-1, d], [h-1, h-1-d], [0, h-1]]], dtype=torch.float, device=tensor.device)
    elif warp_dir == 'left':
        src_pts = torch.tensor([[[0, d], [h-1, 0], [h-1, h-1], [0, h-1-d]]], dtype=torch.float, device=tensor.device)
    else: # down
        src_pts = torch.tensor([[[0, 0], [h-1, 0], [h-1-d, h-1-d], [d, h-1-d]]], dtype=torch.float, device=tensor.device)

    # Perspective Transform
    M = kornia.geometry.get_perspective_transform(src_pts, dst_pts)

    # Prepare for Kornia: (B, C, H, W) -> where B*C is f*dim
    working = rearrange(standard, 'h w f d -> (f d) 1 h w').float()

    # Apply warp
    for _ in range(int(warp_scale)):
        working = kornia.geometry.warp_perspective(working, M.repeat(working.shape[0], 1, 1), dsize=(h, w), mode='bicubic', align_corners=True)

    out = rearrange(working, '(f d) 1 h w -> h w f d', f=f, d=dim).to(tensor.dtype)
    return restore(out)

def rotate_tensor(tensor, pattern, angle=15.0):
    standard, restore, (h, w) = _standardize_shape(tensor, pattern)
    f, dim = standard.shape[2], standard.shape[3]

    working = rearrange(standard, 'h w f d -> (f d) 1 h w').float()
    angle_t = torch.full((working.shape[0],), angle, dtype=torch.float32, device=tensor.device)

    # Rotate around the center; vacated corners are zero-filled.
    working = kornia.geometry.rotate(working, angle_t, mode='bicubic', padding_mode='zeros', align_corners=True)

    out = rearrange(working, '(f d) 1 h w -> h w f d', f=f, d=dim).to(tensor.dtype)
    return restore(out)