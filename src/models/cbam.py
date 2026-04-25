"""Convolutional Block Attention Module (CBAM).

Reference: Woo et al., "CBAM: Convolutional Block Attention Module" (ECCV 2018).
Two sequential gates: channel attention, then spatial attention.
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Reweights channels: 'which feature maps matter for this image?'"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        # Shared MLP applied to both pooled descriptors.
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = x.mean(dim=(2, 3))           # (B, C) global average pool
        mx = x.amax(dim=(2, 3))            # (B, C) global max pool
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))  # (B, C)
        return x * attn.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """Reweights spatial locations: 'where in the image should we focus?'"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        mx = x.amax(dim=1, keepdim=True)   # (B, 1, H, W)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))  # (B, 1, H, W)
        return x * attn


class CBAM(nn.Module):
    """Channel attention -> Spatial attention, applied sequentially."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x
