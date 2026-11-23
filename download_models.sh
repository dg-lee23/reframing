#!/bin/bash
# Downloads all checkpoints into ./models
set -e
M=models

huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --include "model_index.json" "scheduler/*" "tokenizer/*" "feature_extractor/*" \
            "text_encoder/config.json" "text_encoder/model.safetensors" \
            "vae/config.json" "vae/diffusion_pytorch_model.safetensors" \
            "unet/config.json" "unet/diffusion_pytorch_model.bin" \
  --local-dir $M/StableDiffusion/stable-diffusion-v1-5

huggingface-cli download guoyww/animatediff v3_sd15_mm.ckpt v3_sd15_adapter.ckpt --local-dir $M/Motion_Module

huggingface-cli download visible-tactics/realisticVisionV60B1_v51VAE realisticVisionV60B1_v51VAE.safetensors \
  --local-dir $M/DreamBooth_LoRA

huggingface-cli download hehao13/CameraCtrl CameraCtrl.ckpt --local-dir $M/CameraCtrl

# text encoder config, used when loading the DreamBooth checkpoint
huggingface-cli download openai/clip-vit-large-patch14 --include "*.json" "*.txt" "model.safetensors"
