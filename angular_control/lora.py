# Minimal LoRA injection used at inference time.
#
# Adapted from cloneofsimo/lora (https://github.com/cloneofsimo/lora),
# Copyright (c) 2022 Simo Ryu, licensed under the Apache License, Version 2.0.
# Only the parts needed to load trained LoRA weights are kept; the module traversal
# and parameter names are unchanged so that checkpoints load without remapping.
from typing import Iterator, Optional, Set, Tuple

import torch.nn as nn

UNET_TARGET_MODULES = {"CrossAttention", "Attention", "GEGLU"}


class LoraInjectedLinear(nn.Module):
    """y = W x + scale * up(down(x)); parameter names match cloneofsimo/lora."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, r: int = 4, scale: float = 1.0):
        super().__init__()
        if r > min(in_features, out_features):
            raise ValueError(f"LoRA rank {r} must be <= {min(in_features, out_features)}")
        self.r = r
        self.scale = scale
        self.linear = nn.Linear(in_features, out_features, bias)
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.linear(x) + self.lora_up(self.lora_down(x)) * self.scale


def _find_linears(model: nn.Module, ancestor_class: Set[str]) -> Iterator[Tuple[nn.Module, str, nn.Linear]]:
    """Yield (parent, attribute name, linear) for every nn.Linear under a module whose class
    name is in `ancestor_class`, skipping layers that are already LoRA-wrapped."""
    ancestors = [m for m in model.modules() if m.__class__.__name__ in ancestor_class]
    for ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            *path, name = fullname.split(".")
            parent = ancestor
            while path:
                parent = parent.get_submodule(path.pop(0))
            if isinstance(parent, LoraInjectedLinear):
                continue
            yield parent, name, module


def inject_lora(model: nn.Module, r: int = 4, scale: float = 1.0,
                target_modules: Optional[Set[str]] = None) -> int:
    """Wrap the target linear layers of `model` in LoraInjectedLinear (in place).
    Returns the number of wrapped layers."""
    count = 0
    for parent, name, linear in list(_find_linears(model, target_modules or UNET_TARGET_MODULES)):
        wrapped = LoraInjectedLinear(linear.in_features, linear.out_features, linear.bias is not None, r=r, scale=scale)
        wrapped.linear.weight = linear.weight
        if linear.bias is not None:
            wrapped.linear.bias = linear.bias
        wrapped.to(linear.weight.device, linear.weight.dtype)
        parent._modules[name] = wrapped
        count += 1
    return count
