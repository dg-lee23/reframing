# Adapted from MotionClone (https://github.com/LPengYang/MotionClone).
import pickle
from typing import List, Optional

import einops
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from .models.unet import UNet3DConditionModel
from .reframing import feature_loss, zoom_warp_latent, pan_warp_latent
from .scheduler import patch_scheduler
from .utils.util import load_weights


class FeatureRecorder:
    """Attention 'processor' that only stores the value features of the last forward."""

    def record_qkv(self, attn, hidden_states, query, key, value, attention_mask):
        self.value = value

    def record_attn_mask(self, attn, hidden_states, query, key, value, attention_mask):
        pass


class ReframingPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer,
                 unet: UNet3DConditionModel, scheduler: DDIMScheduler):
        super().__init__()
        self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def _text_embeddings(self, prompt):
        tokens = self.tokenizer([prompt], padding="max_length", max_length=self.tokenizer.model_max_length,
                                truncation=True, return_tensors="pt")
        return self.text_encoder(tokens.input_ids.to(self.device))[0]

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = einops.rearrange(1 / 0.18215 * latents, "b c f h w -> (b f) c h w")
        video = torch.cat([self.vae.decode(latents[i:i + 1]).sample for i in tqdm(range(latents.shape[0]))])
        video = einops.rearrange(video, "(b f) c h w -> b c f h w", b=1)
        return (video / 2 + 0.5).clamp(0, 1).cpu().float().numpy()

    @torch.no_grad()
    def invert(self, video, prompt, num_steps, save_path):
        """DDIM inversion of a (f, c, h, w) video in [-1, 1]; saves the latent at every timestep."""
        self.scheduler.set_timesteps(num_steps)
        latent = self.vae.encode(video.to(self.vae.dtype).to(self.vae.device)).latent_dist.sample(None)
        latent = self.vae.config.scaling_factor * latent
        latent = einops.rearrange(latent.unsqueeze(0), "b f c h w -> b c f h w")

        cond = self._text_embeddings(prompt)
        scheduler = self.scheduler
        latents = {}
        for i in tqdm(range(num_steps)):
            t = scheduler.timesteps[len(scheduler.timesteps) - i - 1]
            eps = self.unet(latent, t, encoder_hidden_states=cond)["sample"]
            cur_t = min(t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps, 999)
            alpha_t = scheduler.alphas_cumprod[cur_t] if cur_t >= 0 else scheduler.final_alpha_cumprod
            alpha_next = scheduler.alphas_cumprod[t]
            x0 = (latent - (1 - alpha_t) ** 0.5 * eps) / alpha_t ** 0.5
            latent = alpha_next ** 0.5 * x0 + (1 - alpha_next) ** 0.5 * eps
            latents[t.detach().cpu().item()] = latent

        with open(save_path, "wb") as f:
            pickle.dump({"inversion_prompt": prompt, "all_latents_inversion": latents,
                         "inversion_prompt_embeds": cond}, f)

    def _value_features(self, blocks, second_half=False):
        features = {}
        for name, module in self.unet.named_modules():
            if "VersatileAttention" in type(module).__name__ and any(b in name for b in blocks):
                v = module.processor.value
                features[name] = v[v.shape[0] // 2:] if second_half else v
        return features

    def __call__(
        self,
        config,
        inversion_path: str,
        generator: Optional[torch.Generator] = None,
        transformation: Optional[str] = None,
        factor: Optional[float] = None,
        direction: Optional[str] = None,
        endpoints: bool = False,
        pose_features: Optional[List[torch.Tensor]] = None,
        latent_warp: Optional[dict] = None,
    ):
        """Generate a video that follows the source motion, reframed by T.

        transformation/factor/direction: feature-space reframing (see reframing.transform_features).
        pose_features: CameraCtrl pose features; None runs plain AnimateDiff.
        latent_warp: {'mode': 'zoom', 'crop_range': (a, b)} or
                     {'mode': 'pan', 'direction': d, 'shift_range': (a, b)} to warp the source
                     latent instead of its features.
        """
        for name, module in self.unet.named_modules():
            if "Attention" in type(module).__name__:
                module.set_processor(FeatureRecorder())

        device = self.device
        self.scheduler.set_timesteps(config.num_inference_step, device=device)
        blocks = config.temp_guidance.blocks
        guidance_step = config.guidance_step
        cfg_scale = config.cfg_scale
        loss_mode = config.get("loss_mode", "masked_mse")
        aspect_ratio = config.W / config.H

        uncond = self._text_embeddings(config.get("negative_prompt") or "")
        cond = self._text_embeddings(config.new_prompt)
        text_embeddings = torch.cat([uncond, cond])

        shape = (1, self.unet.config.in_channels, config.L,
                 config.H // self.vae_scale_factor, config.W // self.vae_scale_factor)
        latents = torch.randn(shape, generator=generator, device=device, dtype=text_embeddings.dtype)
        latents = latents * self.scheduler.init_noise_sigma

        with open(inversion_path, "rb") as f:
            inverted = pickle.load(f)
        source_latents = inverted["all_latents_inversion"]
        source_embeds = inverted["inversion_prompt_embeds"].to(device)

        warped_x0, valid_mask, warp_noise = None, None, None
        if latent_warp is not None:
            x0 = source_latents[min(source_latents.keys())].to(device=device, dtype=torch.float32)
            x0 = einops.rearrange(x0, "b c f h w -> (b f) c h w")
            if latent_warp["mode"] == "zoom":
                warped_x0, valid_mask = zoom_warp_latent(x0, latent_warp["crop_range"])
            elif latent_warp["mode"] == "pan":
                warped_x0, valid_mask = pan_warp_latent(x0, latent_warp["direction"], latent_warp["shift_range"])
            else:
                raise ValueError(f"unknown latent_warp mode {latent_warp['mode']!r}")
            warp_noise = torch.randn(warped_x0.shape, generator=torch.Generator(device=device).manual_seed(42),
                                     device=device, dtype=torch.float32)

        for step, t in enumerate(tqdm(self.scheduler.timesteps)):
            if step >= guidance_step:
                latents = self._denoise_step(latents, t, text_embeddings, cfg_scale, pose_features, generator)
                continue

            if warped_x0 is not None:
                abar = self.scheduler.alphas_cumprod.to(device)[t.long()]
                x_t = abar ** 0.5 * warped_x0 + (1 - abar) ** 0.5 * warp_noise
                source = einops.rearrange(x_t.unsqueeze(0), "b f c h w -> b c f h w")
            else:
                source = source_latents[min(source_latents.keys(), key=lambda k: abs(k - t.item()))]
            source = source.to(device=device, dtype=text_embeddings.dtype)

            latents = self._guided_step(
                latents, step, t, source, text_embeddings, source_embeds, cfg_scale, generator,
                blocks, guidance_step, config.grad_guidance_scale, loss_mode, aspect_ratio, pose_features,
                valid_mask, transformation, factor, direction, endpoints,
            )

        return self.decode_latents(latents)

    @torch.no_grad()
    def _denoise_step(self, latents, t, text_embeddings, cfg_scale, pose_features, generator):
        x = torch.cat([latents, latents])
        if pose_features is None:
            eps = self.unet(x, t, encoder_hidden_states=text_embeddings).sample.to(dtype=latents.dtype)
        else:
            eps_u = self.unet(x[[0]], t, encoder_hidden_states=text_embeddings[[0]], pose_embedding_features=pose_features).sample
            eps_c = self.unet(x[[1]], t, encoder_hidden_states=text_embeddings[[1]], pose_embedding_features=pose_features).sample
            eps = torch.cat([eps_u, eps_c]).to(dtype=latents.dtype)
        eps = eps[[1]] + cfg_scale * (eps[[1]] - eps[[0]])
        return self.scheduler.guided_step(eps, t, latents, generator=generator)

    def _guided_step(self, latents, step, t, source, text_embeddings, source_embeds, cfg_scale, generator,
                     blocks, guidance_step, grad_scale, loss_mode, aspect_ratio, pose_features,
                     valid_mask, transformation, factor, direction, endpoints):
        x = torch.cat([latents, source.detach(), latents]).detach()
        x.requires_grad = True
        embeds = torch.cat([text_embeddings[[0]], source_embeds, text_embeddings[[1]]])

        # 1. unconditional + source branch (no grad), record source features
        with torch.no_grad():
            if pose_features is None:
                eps_uncond = self.unet(x[[0, 1]], t, encoder_hidden_states=embeds[[0, 1]]).sample.to(dtype=latents.dtype)
                f_source = self._value_features(blocks, second_half=True)
            else:
                eps_u = self.unet(x[[0]], t, encoder_hidden_states=embeds[[0]],
                                  pose_embedding_features=pose_features).sample.to(dtype=latents.dtype)
                eps_s = self.unet(x[[1]], t, encoder_hidden_states=embeds[[1]]).sample.to(dtype=latents.dtype)
                eps_uncond = torch.cat([eps_u, eps_s])
                f_source = self._value_features(blocks)

        # 2. generated branch (with grad)
        torch.cuda.empty_cache()
        eps = self.unet(x[[2]], t, encoder_hidden_states=embeds[[2]],
                        pose_embedding_features=pose_features).sample.to(dtype=latents.dtype)
        f_control = self._value_features(blocks)

        # 3. guidance: gradient of the feature loss w.r.t. the latent
        loss = 100.0 * feature_loss(f_source, f_control, transformation, factor, direction,
                                    aspect_ratio, endpoints, loss_mode, valid_mask)
        loss = loss * (1 - step / guidance_step)

        torch.cuda.empty_cache()
        grad = torch.autograd.grad(loss, x, allow_unused=True)[0][[2]]
        norm = grad.norm()
        if norm > 0:
            grad = grad * (0.5 / norm)

        eps = eps + cfg_scale * (eps - eps_uncond[[0]])
        return self.scheduler.guided_step(eps, t, latents, generator=generator, score=grad.detach(),
                                          guidance_scale=grad_scale).detach()


def build_pipeline(config, pretrained_model_path="models/StableDiffusion/stable-diffusion-v1-5",
                   device="cuda", dtype=torch.float16, use_xformers=True):
    model_config = OmegaConf.load(config.model_config)

    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder",
                                                 torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae",
                                        torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path, subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(model_config.unet_additional_kwargs),
    ).to(device).to(dtype=dtype)

    if use_xformers and is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    scheduler = patch_scheduler(DDIMScheduler(**OmegaConf.to_container(model_config.noise_scheduler_kwargs)))
    pipeline = ReframingPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler)
    pipeline = load_weights(
        pipeline,
        motion_module_path=config.get("motion_module", ""),
        dreambooth_model_path=config.get("dreambooth_path", ""),
        adapter_lora_path=config.get("adapter_lora_path", ""),
        adapter_lora_scale=config.get("adapter_lora_scale", 1.0),
    ).to(device)
    pipeline.vae.enable_slicing()
    return pipeline


def enable_gradient_checkpointing(pipeline, blocks):
    """Saves memory on small GPUs. The guided blocks are left out so their features keep a graph."""
    pipeline.unet.enable_gradient_checkpointing()
    pipeline.unet.train()
    for name in blocks:
        kind, idx = name.split(".")
        getattr(pipeline.unet, kind)[int(idx)].gradient_checkpointing = False
