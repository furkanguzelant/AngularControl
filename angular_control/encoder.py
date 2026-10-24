"""Periodic angle encoder: maps an angle to a text-token embedding.

Each angle is given as a normalized value x in [0, 1] (x = theta / 2*pi), encoded as
[sin 2*pi*x, cos 2*pi*x] and passed through a 3-layer MLP. The same network is applied to
every angle of a target (hour/minute hand, or start/end boundary); the semantic role of each
resulting token is given by its position in the prompt.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PeriodicAngleEncoder(nn.Module):
    def __init__(self, output_dim: int = 1024):
        super().__init__()
        self.output_dim = output_dim
        self.linear1 = nn.Linear(2, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.linear3 = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., N) normalized angles in [0, 1] -> (..., N, output_dim)."""
        x = torch.stack([torch.sin(2 * torch.pi * x), torch.cos(2 * torch.pi * x)], dim=-1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)
