# Principled Static Reframing in Motion Transfer with Video Diffusion Models

[Project page](https://dg-lee23.github.io/principled-reframing-project-page/)

Motion transfer copies the motion of a *reference* video into a new video. Reframing is a training-free method that lets you also change
the video *framing*: move the subject, zoom in or out, rotate, or change the perspective. 

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

`download_models.sh` puts the necessary checkpoints in `models/`: Stable Diffusion 1.5, AnimateDiff v3,Realistic Vision V6.0, and CameraCtrl.

## Usage

Each config describes one source video and the prompts. First invert the video (once per config),
then generate with the desired reframing:

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
| `warp` | changes the perspective | - | left, right, down |

Leave out `--transform` for plain (default) motion transfer.

For a 320x320x16 video, about 22GB of VRAM is required. If you run out of GPU memory, add `--grad-ckpt`. It is slower but uses much less memory.


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

See each README files in `baselines/ctrl-x` and `baselines/dmt`, as they have their own environments.
The DMT baseline provides the best motion fidelity, and Ctrl-X baseline the best video quality with much less memory.

## Acknowledgements

This code is based on [MotionClone](https://github.com/LPengYang/MotionClone),
[AnimateDiff](https://github.com/guoyww/AnimateDiff), [CameraCtrl](https://github.com/hehao13/CameraCtrl),
[Ctrl-X](https://github.com/genforce/ctrl-x) and
[DMT](https://github.com/diffusion-motion-transfer/diffusion-motion-transfer). Thanks to the authors
for sharing their code.
