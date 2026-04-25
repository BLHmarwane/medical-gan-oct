"""PatchGAN discriminator (70x70 receptive field).

Outputs a grid of logits — each cell classifies one local patch as real/fake.
Standard CycleGAN / pix2pix architecture.
"""
import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 1, ndf: int = 64, n_layers: int = 3):
        """
        Args:
            in_channels: 1 for grayscale OCT.
            ndf: base feature width (doubles each layer up to 8x).
            n_layers: number of stride-2 conv blocks after the first.
                      Default 3 -> 70x70 receptive field for 256x256 input.
        """
        super().__init__()
        layers: list[nn.Module] = []

        # --- First layer: no norm, plain conv + LeakyReLU ---
        layers += [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # --- Stride-2 blocks: double channels each step (cap at 8x) ---
        c = ndf
        for i in range(1, n_layers):
            c_next = min(ndf * (2 ** i), ndf * 8)
            layers += [
                nn.Conv2d(c, c_next, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(c_next),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            c = c_next

        # --- Penultimate: stride-1 block, channels keep doubling once more ---
        c_next = min(ndf * (2 ** n_layers), ndf * 8)
        layers += [
            nn.Conv2d(c, c_next, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(c_next),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        c = c_next

        # --- Output: 1 logit per patch, no activation ---
        layers.append(nn.Conv2d(c, 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == "__main__":
    net = PatchDiscriminator(in_channels=1)
    x = torch.randn(2, 1, 256, 256)
    y = net(x)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"input  : {tuple(x.shape)}")
    print(f"output : {tuple(y.shape)}  range=[{y.min():.2f}, {y.max():.2f}]")
    print(f"params : {n_params / 1e6:.2f} M")
