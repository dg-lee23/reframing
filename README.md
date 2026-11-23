# Principled Static Reframing in Motion Transfer with Video Diffusion Models

[Project page](https://dg-lee23.github.io/reframing-page/)

Motion transfer copies the motion of a source video into a new video. Reframing lets you also change
how the shot is framed: move the subject, zoom in or out, rotate, or change the viewpoint. We do this
by transforming the source features that guide generation, so no training is needed.

The main code is built on [MotionClone](https://github.com/LPengYang/MotionClone) with AnimateDiff v3.
We also include reframing for [Ctrl-X](https://github.com/genforce/ctrl-x) and
[DMT](https://github.com/diffusion-motion-transfer/diffusion-motion-transfer) in `baselines/`, and
a combination with [CameraCtrl](https://github.com/hehao13/CameraCtrl).

## Setup

```bash
conda create -n reframing python=3.10 -y
conda activate reframing
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
bash download_models.sh
```

`download_models.sh` puts everything in `models/`: Stable Diffusion 1.5, AnimateDiff v3,
Realistic Vision V6.0 and CameraCtrl.

## Usage

Each config describes one source video and the prompts. First invert the video (once per config),
then generate:

```bash
python invert.py --config configs/examples/fox_snow.yaml
python run.py --config configs/examples/fox_snow.yaml --transform shift --direction right
```

The result is saved to `results/`. Reframing options:

| `--transform` | what it does | `--factor` | `--direction` |
|---|---|---|---|
| `shift` | moves the subject | frame / factor (default 4) | up, down, left, right |
| `resize` | zoom in (>1) or out (<1) | area ratio, e.g. 2.0 or 0.5 | - |
| `rotate` | rotates the frame | degrees, counter-clockwise | - |
| `warp` | changes the viewpoint | - | left, right, down |
| `time` | zoom or pan that changes over time | end zoom (default 4) or frame / factor | set for a pan |

Leave out `--transform` for plain motion transfer.

If you run out of GPU memory, add `--grad-ckpt`. It is slower but uses much less memory.

### With CameraCtrl

```bash
python invert.py --config configs/cameractrl/retriever.yaml
python run_cameractrl.py --config configs/cameractrl/retriever.yaml --move zoom_out
```

`--move` is `zoom_in`, `zoom_out`, `pan_right` or `all`. `--method` picks how reframing is done:
`ours` (warp the source latent over time, default), `feature` (time-varying feature transform)
or `camera` (CameraCtrl only, no reframing). The supplementary results use `W: 512, H: 512`;
change them in the config and invert again.

## Baselines

`baselines/ctrl-x` and `baselines/dmt` have their own environments. See the README in each folder.

## Acknowledgements

This code is based on [MotionClone](https://github.com/LPengYang/MotionClone),
[AnimateDiff](https://github.com/guoyww/AnimateDiff), [CameraCtrl](https://github.com/hehao13/CameraCtrl),
[Ctrl-X](https://github.com/genforce/ctrl-x) and
[DMT](https://github.com/diffusion-motion-transfer/diffusion-motion-transfer). Thanks to the authors
for sharing their code.
