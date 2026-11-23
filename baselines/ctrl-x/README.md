# Ctrl-X + reframing

[Ctrl-X](https://github.com/genforce/ctrl-x) adapted to text-to-video with AnimateDiff, with reframing
added to its feature injection (`ctrl_x/utils/feature.py`).

```bash
conda env create -f environment.yaml
conda activate ctrlx_with_reframing
python run.py --config assets/configs/retriever.yaml --transformation shift
```

`--transformation` is `shift`, `resize`, `warp` or `rotate`. For `resize`, `--resize_factor` sets the
area ratio (default 0.5). Models are downloaded from Hugging Face on the first run.
