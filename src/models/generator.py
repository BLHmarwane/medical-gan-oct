"""ResNet-based generator with CBAM attention.

Architecture (for 256x256 input):
  stem (7x7 conv) -> 2x downsample -> N residual+CBAM blocks -> 2x upsample -> 7x7 conv -> tanh
"""
import torch
import torch.nn as nn

from src.models.cbam import CBAM


class ResnetBlock(nn.Module):
    """Residual block: x -> conv-IN-ReLU-conv-IN -> CBAM -> add x."""

    def __init__(self, channels: int, use_cbam: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
        )
        self.attn = CBAM(channels) if use_cbam else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.block(x))


class ResnetGenerator(nn.Module):
    """CycleGAN generator with optional CBAM in every residual block."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        ngf: int = 64,
        n_blocks: int = 9,
        use_cbam: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []

        # --- Stem: 7x7 conv expanding to ngf channels ---
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        # --- Downsample x2 (each step: spatial /2, channels x2) ---
        c = ngf
        for _ in range(2):
            layers += [
                nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(c * 2),
                nn.ReLU(inplace=True),
            ]
            c *= 2

        # --- Bottleneck: residual + CBAM blocks ---
        for _ in range(n_blocks):
            layers.append(ResnetBlock(c, use_cbam=use_cbam))

        # --- Upsample x2 (each step: spatial x2, channels /2) ---
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(
                    c, c // 2, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(c // 2),
                nn.ReLU(inplace=True),
            ]
            c //= 2

        # --- Output: 7x7 conv -> tanh, range [-1, 1] ---
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(c, out_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == "__main__":
    # Sanity: shapes, value range, parameter count.
    net = ResnetGenerator(in_channels=1, out_channels=1, n_blocks=9, use_cbam=True)
    x = torch.randn(2, 1, 256, 256)
    y = net(x)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"input  : {tuple(x.shape)}")
    print(f"output : {tuple(y.shape)}  range=[{y.min():.2f}, {y.max():.2f}]")
    print(f"params : {n_params / 1e6:.2f} M")
