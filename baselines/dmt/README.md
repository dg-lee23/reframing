# DMT + reframing

[DMT](https://github.com/diffusion-motion-transfer/diffusion-motion-transfer) (ZeroScope) with reframing
applied to the source features in the guidance loss (`utilities/guidance_utils.py`).

```bash
conda create -n dmt python=3.9 -y
conda activate dmt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

mkdir -p data/retriever
ffmpeg -i ../../resources/retriever.mp4 data/retriever/%05d.png
python preprocess_video_ddim.py --config_path configs/preprocess_config.yaml
python run.py --config_path configs/guidance_config.yaml
```

Set the reframing in the guidance config:

```yaml
transformation: shift        # None, shift, resize, warp or rotate
shift_frac: [0, 0.25]        # shift: (down, right) as a fraction of the frame
resize_factor: 2.0           # resize: area ratio
warp_dir: right              # warp
rotate_angle: -10.0          # rotate: degrees, negative is clockwise
```
