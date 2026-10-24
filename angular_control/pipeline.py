"""Inference pipeline: Stable Diffusion 2.1 + LoRA + geometric tokens from the angle encoder.

Two angular quantities are supported, each rendered from two geometric tokens:
  clock     : hour-hand angle and minute-hand angle
  fraction  : start-boundary angle (fixed at 0) and end-boundary angle (2*pi*f)

The geometric tokens replace the input embeddings of two placeholder words in the prompt
before the frozen CLIP text encoder; no bounding boxes or attention constraints are used.
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from .encoder import PeriodicAngleEncoder
from .lora import inject_lora

# Rare CLIP words whose input embeddings are replaced by the geometric tokens.
TOKEN_A = "bnha"  # hour hand / start boundary
TOKEN_B = "bk"    # minute hand / end boundary

PHRASES = {
    "clock": f"a {TOKEN_A} hour hand and a {TOKEN_B} minute hand",
    "fraction": f"a {TOKEN_A} start boundary and a {TOKEN_B} end boundary",
}
PLACEHOLDER = "PLACEHOLDER"

DEFAULT_BASE_MODEL = "sd2-community/stable-diffusion-2-1"


def clock_angles(time_str: str):
    """'HH:MM' or 'HH:MM:SS' -> normalized (hour, minute) hand angles in [0, 1)."""
    parts = [float(p) for p in time_str.split(":")]
    hours, minutes = parts[0], parts[1] + (parts[2] / 60.0 if len(parts) > 2 else 0.0)
    return ((hours % 12.0) + minutes / 60.0) / 12.0, minutes / 60.0


def fraction_angles(fraction: float):
    """fraction f in [0, 1] -> normalized (start, end) boundary angles (start fixed at 0)."""
    return 0.0, float(fraction)


def load_checkpoint(path: str):
    """Accepts a full training state (as written by the training script) or an exported
    file from scripts/export_weights.py. Returns (encoder_state, unet_lora_state, config)."""
    state = torch.load(path, map_location="cpu")
    if "encoder" in state and "unet_lora" in state:
        return state["encoder"], state["unet_lora"], state.get("config", {})
    merger = state["merger"]
    encoder_state = merger["model"] if isinstance(merger, dict) and "model" in merger else merger
    unet = state.get("unet", {})
    lora_state = unet.get("lora") if isinstance(unet, dict) and "lora" in unet else state["unet_lora"]
    return encoder_state, lora_state, {}


@dataclass
class GenerationConfig:
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    negative_prompt: str = ""


class AngularControlPipeline:
    def __init__(self, tokenizer, text_encoder, vae, unet, scheduler, encoder, task: str, device):
        if task not in PHRASES:
            raise ValueError(f"task must be one of {sorted(PHRASES)}")
        self.tokenizer, self.text_encoder, self.vae, self.unet = tokenizer, text_encoder, vae, unet
        self.scheduler, self.encoder, self.task, self.device = scheduler, encoder, task, device
        self.base_embeddings = text_encoder.get_input_embeddings().weight.detach().clone()
        self.token_ids = [self._single_token_id(TOKEN_A), self._single_token_id(TOKEN_B)]

    def _single_token_id(self, word: str) -> int:
        ids = self.tokenizer(word, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"placeholder word {word!r} is not a single CLIP token: {ids}")
        return ids[0]

    @classmethod
    def from_pretrained(cls, checkpoint: str, task: str, base_model: str = DEFAULT_BASE_MODEL,
                        device: Optional[str] = None, lora_rank: Optional[int] = None,
                        local_files_only: bool = False):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = dict(local_files_only=local_files_only)
        tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer", **kw)
        text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", **kw).to(device)
        vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", **kw).to(device)
        unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", **kw).to(device)
        scheduler = DDIMScheduler.from_pretrained(base_model, subfolder="scheduler", **kw)

        encoder_state, lora_state, config = load_checkpoint(checkpoint)
        rank = lora_rank or config.get("lora_rank") or next(
            v.shape[0] for k, v in lora_state.items() if k.endswith("lora_down.weight"))
        inject_lora(unet, r=rank)
        missing = [k for k in lora_state if k not in unet.state_dict()]
        if missing:
            raise ValueError(f"{len(missing)} LoRA weights do not match the UNet, e.g. {missing[:3]}")
        unet.load_state_dict(lora_state, strict=False)

        emb_dim = text_encoder.get_input_embeddings().weight.shape[1]
        encoder = PeriodicAngleEncoder(emb_dim).to(device)
        encoder.load_state_dict({k: v for k, v in encoder_state.items() if k.startswith("linear")})

        for module in (text_encoder, vae, unet, encoder):
            module.requires_grad_(False).eval()
        return cls(tokenizer, text_encoder, vae, unet, scheduler, encoder, task, device)

    # ------------------------------------------------------------------ conditioning
    def _encode(self, prompt: str, angles: Optional[Sequence[float]] = None) -> torch.Tensor:
        ids = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").input_ids.to(self.device)
        embeddings = self.base_embeddings.clone()
        if angles is not None:
            tokens = self.encoder(torch.tensor([list(angles)], dtype=torch.float32, device=self.device))[0]
            for token_id, token in zip(self.token_ids, tokens):
                embeddings[token_id] = token.to(embeddings.dtype)
        self.text_encoder.get_input_embeddings().weight = torch.nn.Parameter(embeddings, requires_grad=False)
        return self.text_encoder(ids)[0]

    def build_prompt(self, template: str) -> str:
        if PLACEHOLDER not in template:
            raise ValueError(f"prompt template must contain {PLACEHOLDER!r}, e.g. 'PLACEHOLDER, a black analog clock'")
        return template.replace(PLACEHOLDER, PHRASES[self.task])

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def __call__(self, template: str, target: Union[str, float], seed: int = 0,
                 config: GenerationConfig = GenerationConfig()) -> Image.Image:
        angles = clock_angles(target) if self.task == "clock" else fraction_angles(float(target))
        cond = self._encode(self.build_prompt(template), angles)
        uncond = self._encode(config.negative_prompt)

        self.scheduler.set_timesteps(config.num_inference_steps)
        latents = torch.randn((1, 4, 64, 64), generator=torch.Generator(device=self.device).manual_seed(seed),
                              device=self.device, dtype=self.vae.dtype)
        context = torch.cat([uncond, cond], dim=0)
        for t in self.scheduler.timesteps:
            model_in = self.scheduler.scale_model_input(torch.cat([latents, latents], dim=0), timestep=t)
            noise_uncond, noise_cond = self.unet(model_in, t, encoder_hidden_states=context).sample.chunk(2)
            noise = noise_uncond + config.guidance_scale * (noise_cond - noise_uncond)
            latents = self.scheduler.step(noise, t, latents).prev_sample

        image = self.vae.decode(latents / 0.18215).sample
        image = (image / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).cpu().float().numpy()
        return Image.fromarray((image * 255).astype("uint8"))  # same truncation as torchvision ToPILImage
