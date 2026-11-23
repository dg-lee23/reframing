import numpy as np
import torch
from diffusers.utils import randn_tensor


# DDIM step with gradient guidance, adapted from MotionClone.
@torch.no_grad()
def guided_step(self, model_output, timestep, sample, generator=None, score=None, guidance_scale=0.0):
    prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    pred_epsilon = model_output

    variance = self._get_variance(timestep, prev_timestep)
    eta = 0.025
    std_dev_t = eta * variance ** (0.5)

    if score is not None and guidance_scale > 0.0:
        score = std_dev_t * score / (torch.std(score) + 1e-7)
        pred_epsilon[[0]] = pred_epsilon[[0]] - guidance_scale * (1 - alpha_prod_t) ** (0.5) * score

    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5) * pred_epsilon
    prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

    noise = randn_tensor(model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype)
    return prev_sample + std_dev_t * noise


def set_timesteps(self, num_inference_steps, device=None):
    self.num_inference_steps = num_inference_steps
    timesteps = (
        np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
        .round()[::-1]
        .copy()
        .astype(np.int64)
    )
    self.timesteps = torch.from_numpy(timesteps).to(device)


def patch_scheduler(scheduler):
    scheduler.guided_step = guided_step.__get__(scheduler)
    scheduler.set_timesteps = set_timesteps.__get__(scheduler)
    return scheduler
