"""Reframing transforms T applied to the source (example) features, and the guidance loss.

Feature maps here are temporal-attention features of shape (hw, f, dim).
"""
import kornia
import numpy as np
import torch
import torch.nn.functional as F


TRANSFORMATIONS = ("shift", "resize", "rotate", "warp", "time")


def pad_center(x, h, w):
    _, _, H, W = x.shape
    top, left = (h - H) // 2, (w - W) // 2
    return F.pad(x, (left, w - W - left, top, h - H - top), mode='constant', value=0)


def crop_center(x, h, w):
    _, _, H, W = x.shape
    top, left = (H - h) // 2, (W - w) // 2
    return x[:, :, top:top + h, left:left + w]


def resize_and_restore(frames, scale, dim, h, w):
    """Resize (n, dim, h, w) by `scale` (in area), then pad/crop back to (h, w)."""
    s = scale ** 0.5
    resized = F.interpolate(frames, size=(int(h * s), int(w * s)), mode='bicubic', align_corners=True)
    if scale <= 1:
        return pad_center(resized, h, w)
    return crop_center(resized, h, w)


def shift_tensor(tensor, shift, aspect_ratio=1.0):
    """Shift a (hw, a, b, c) tensor spatially without wrap-around, filling with zeros.
    Positive shift moves content up / left."""
    hw, a, b, c = tensor.shape
    h = int(round((hw / aspect_ratio) ** 0.5))
    w = hw // h
    sx, sy = shift

    tensor = tensor.view(h, w, a, b, c)
    if sx > 0:
        tensor = torch.cat((tensor[sx:], torch.zeros_like(tensor[:sx])), dim=0)
    elif sx < 0:
        tensor = torch.cat((torch.zeros_like(tensor[sx:]), tensor[:sx]), dim=0)
    if sy > 0:
        tensor = torch.cat((tensor[:, sy:], torch.zeros_like(tensor[:, :sy])), dim=1)
    elif sy < 0:
        tensor = torch.cat((torch.zeros_like(tensor[:, sy:]), tensor[:, :sy]), dim=1)
    return tensor


def _shift_amount(direction, h, w, n, progress=1.0):
    if direction == 'up':
        return (int(int(h // n) * progress), 0)
    if direction == 'down':
        return (-int(int(h // n) * progress), 0)
    if direction == 'left':
        return (0, int(int(w // n) * progress))
    if direction == 'right':
        return (0, -int(int(w // n) * progress))
    raise ValueError(f"unknown direction {direction!r}")


def _perspective(f_example, h, w, f, dim, direction):
    d = int(h * 0.2)
    dst = [(0, 0), (h - 1, 0), (h - 1, h - 1), (0, h - 1)]
    if direction == 'right':
        src = [(0, 0), (h - 1, d), (h - 1, h - 1 - d), (0, h - 1)]
    elif direction == 'left':
        src = [(0, d), (h - 1, 0), (h - 1, h - 1), (0, h - 1 - d)]
    elif direction == 'down':
        src = [(0, 0), (h - 1, 0), (h - 1 - d, h - 1 - d), (d, h - 1 - d)]
    else:
        raise ValueError(f"unknown warp direction {direction!r}")
    dst = torch.tensor(dst, dtype=torch.float, device=f_example.device).unsqueeze(0)
    src = torch.tensor(src, dtype=torch.float, device=f_example.device).unsqueeze(0)
    H = kornia.geometry.get_perspective_transform(src, dst).float()

    x = f_example.view(h, w, f, dim).permute(2, 3, 0, 1).view(f * dim, 1, h, w)
    x = kornia.geometry.warp_perspective(x.float(), H.repeat(f * dim, 1, 1), dsize=(h, w),
                                         mode='bilinear', align_corners=True)
    return x.view(f, dim, h, w).permute(2, 3, 0, 1).view(h * w, f, dim).half()


def transform_features(f_example, h, w, transformation, factor=None, direction=None,
                       aspect_ratio=1.0, endpoints=False):
    """Apply reframing T to source features (hw, f, dim).

    shift:  move by 1/factor of the frame (default 4) towards `direction`
    resize: zoom by `factor` (area ratio; >1 zoom in, <1 zoom out)
    rotate: rotate by `factor` degrees (counter-clockwise)
    warp:   perspective change towards `direction`
    time:   time-varying shift (if `direction` is set) or zoom 1 -> `factor` (default 4)
            over the clip. With endpoints=True only the last frame is transformed.
    """
    if transformation is None:
        return f_example
    hw, f, dim = f_example.shape

    if transformation == 'resize':
        x = f_example.view(h, w, f, dim).permute(2, 3, 0, 1)
        x = resize_and_restore(x, factor, dim, h, w).permute(2, 3, 0, 1)
        return x.reshape(hw, f, dim)

    if transformation == 'shift':
        n = 4 if factor is None else factor
        shift = _shift_amount(direction, h, w, n)
        return shift_tensor(f_example.unsqueeze(-1), shift, aspect_ratio).view(hw, f, dim)

    if transformation == 'warp':
        return _perspective(f_example, h, w, f, dim, direction)

    if transformation == 'rotate':
        x = f_example.view(h, w, f, dim).permute(2, 3, 0, 1).view(f * dim, 1, h, w)
        angle = torch.full((f * dim,), factor, dtype=torch.float, device=x.device)
        x = kornia.geometry.transform.rotate(x.float(), angle, align_corners=True)
        return x.view(f, dim, h, w).permute(2, 3, 0, 1).view(hw, f, dim).half()

    if transformation == 'time':
        def progress(t):
            if endpoints:
                return 1.0 if t == f - 1 else 0.0
            return t / (f - 1) if f > 1 else 0.0

        if direction is not None:
            n = 4 if factor is None else factor
            frames = []
            for t in range(f):
                shift = _shift_amount(direction, h, w, n, progress(t))
                frame = shift_tensor(f_example[:, t:t + 1].unsqueeze(-1), shift, aspect_ratio)
                frames.append(frame.view(hw, 1, dim))
            return torch.cat(frames, dim=1)

        end_scale = 4.0 if factor is None else factor

        def scale(t):
            if endpoints:
                return end_scale if t == f - 1 else 1.0
            return 1.0 + (end_scale - 1.0) * (t / (f - 1))

        x = f_example.view(h, w, f, dim).permute(2, 3, 0, 1)
        frames = [resize_and_restore(x[t:t + 1], scale(t), dim, h, w) for t in range(f)]
        return torch.cat(frames, dim=0).permute(2, 3, 0, 1).reshape(hw, f, dim)

    raise ValueError(f"unknown transformation {transformation!r}")


def feature_loss(f_example_dict, f_control_dict, transformation=None, factor=None, direction=None,
                 aspect_ratio=1.0, endpoints=False, loss_mode='masked_mse', valid_mask=None):
    """MSE between transformed source features and generated features, averaged over blocks.

    loss_mode='masked_mse': ignore positions where the transformed source is ~0 (padding)
    loss_mode='full_mse':   plain MSE, optionally restricted to `valid_mask` (F, h, w)
    """
    losses = []
    for name, f_example in f_example_dict.items():
        f_example = f_example.detach()
        f_control = f_control_dict[name]

        hw, f, dim = f_example.shape
        h = int(round((hw / aspect_ratio) ** 0.5))
        w = hw // h

        f_example = transform_features(f_example, h, w, transformation, factor, direction,
                                       aspect_ratio, endpoints)

        if loss_mode == 'masked_mse':
            mask = f_example > 1e-3
            loss = F.mse_loss(f_control[mask], f_example[mask])
        elif loss_mode == 'full_mse':
            if valid_mask is not None:
                m = F.interpolate(valid_mask.unsqueeze(1), size=(h, w), mode='nearest')
                m = m.squeeze(1).permute(1, 2, 0).reshape(hw, f).bool()
                loss = F.mse_loss(f_control[m], f_example[m])
            else:
                loss = F.mse_loss(f_control, f_example)
        else:
            raise ValueError(f"unknown loss_mode {loss_mode!r}")
        losses.append(loss)

    return 2 * torch.stack(losses).mean()


# Latent warping (used with CameraCtrl): warp the clean source latent x0 directly,
# frame by frame, instead of transforming features. x0: (F, C, h, w).
# Returns the warped latent and a validity mask (F, h, w), 0 where content was filled in.

def zoom_warp_latent(x0, crop_range):
    """Per-frame zoom ramping from crop_range[0] to crop_range[1] (field-of-view ratio).
    <1 crops the center and upsamples (zoom in), >1 shrinks the frame and fills the
    border with the frame mean (zoom out)."""
    num_frames, c, h, w = x0.shape
    fracs = np.linspace(crop_range[0], crop_range[1], num_frames)
    frames, masks = [], []
    for i, frac in enumerate(fracs):
        if frac <= 1.0:
            ch, cw = max(1, int(round(h * frac))), max(1, int(round(w * frac)))
            top, left = (h - ch) // 2, (w - cw) // 2
            crop = x0[i:i + 1, :, top:top + ch, left:left + cw]
            frames.append(F.interpolate(crop, size=(h, w), mode='bicubic', align_corners=True)[0])
            masks.append(torch.ones(h, w, device=x0.device, dtype=x0.dtype))
        else:
            sh, sw = max(1, int(round(h / frac))), max(1, int(round(w / frac)))
            small = F.interpolate(x0[i:i + 1], size=(sh, sw), mode='bicubic', align_corners=True)
            top, left = (h - sh) // 2, (w - sw) // 2
            padded = F.pad(small, (left, w - sw - left, top, h - sh - top), mode='constant', value=0)
            border = torch.ones_like(padded)
            border[:, :, top:top + sh, left:left + sw] = 0
            frames.append((padded + border * x0[i:i + 1].mean(dim=(2, 3), keepdim=True))[0])
            mask = torch.zeros(h, w, device=x0.device, dtype=x0.dtype)
            mask[top:top + sh, left:left + sw] = 1.0
            masks.append(mask)
    return torch.stack(frames), torch.stack(masks)


def pan_warp_latent(x0, direction, shift_range):
    """Per-frame shift ramping from shift_range[0] to shift_range[1] (fraction of the frame),
    filling the revealed border with the frame mean."""
    num_frames, c, h, w = x0.shape
    fracs = np.linspace(shift_range[0], shift_range[1], num_frames)
    size = h if direction in ('up', 'down') else w
    frames, masks = [], []
    for i, frac in enumerate(fracs):
        frame = x0[i:i + 1]
        mask = torch.ones(h, w, device=x0.device, dtype=x0.dtype)
        s = int(round(size * frac))
        if s > 0:
            fill = frame.mean(dim=(2, 3), keepdim=True)
            if direction == 'up':
                frame = torch.cat([frame[:, :, s:], fill.expand(-1, -1, s, w)], dim=2)
                mask = torch.cat([mask[s:], torch.zeros_like(mask[:s])], dim=0)
            elif direction == 'down':
                frame = torch.cat([fill.expand(-1, -1, s, w), frame[:, :, :-s]], dim=2)
                mask = torch.cat([torch.zeros_like(mask[:s]), mask[:-s]], dim=0)
            elif direction == 'left':
                frame = torch.cat([frame[:, :, :, s:], fill.expand(-1, -1, h, s)], dim=3)
                mask = torch.cat([mask[:, s:], torch.zeros_like(mask[:, :s])], dim=1)
            elif direction == 'right':
                frame = torch.cat([fill.expand(-1, -1, h, s), frame[:, :, :, :-s]], dim=3)
                mask = torch.cat([torch.zeros_like(mask[:, :s]), mask[:, :-s]], dim=1)
            else:
                raise ValueError(f"unknown direction {direction!r}")
        frames.append(frame[0])
        masks.append(mask)
    return torch.stack(frames), torch.stack(masks)
