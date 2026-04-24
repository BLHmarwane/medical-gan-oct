"""Quick sanity-check viewer for the unpaired OCT dataset.

Run:  python -m src.visualize
Shows one batch: top row = degraded (A), bottom row = clean (B).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.dataset import UnpairedOCTDataset


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Invert Normalize(mean=0.5, std=0.5): [-1, 1] -> [0, 1]."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def show_batch(batch: dict, n: int = 4) -> None:
    """Display n samples from A on top and n from B on bottom."""
    a = denormalize(batch["A"][:n]).cpu()
    b = denormalize(batch["B"][:n]).cpu()

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for i in range(n):
        # squeeze(0) drops the channel dim: (1, H, W) -> (H, W)
        axes[0, i].imshow(a[i].squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"A (degraded) #{i}")
        axes[0, i].axis("off")

        axes[1, i].imshow(b[i].squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"B (clean) #{i}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(0)  # reproducible random pairings

    base = Path(__file__).resolve().parent.parent
    dataset = UnpairedOCTDataset(
        root_a=base / "data" / "processed" / "domain_A",
        root_b=base / "data" / "processed" / "domain_B",
        image_size=256,
    )

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    print(f"Batch A: {tuple(batch['A'].shape)}  range=[{batch['A'].min():.2f}, {batch['A'].max():.2f}]")
    print(f"Batch B: {tuple(batch['B'].shape)}  range=[{batch['B'].min():.2f}, {batch['B'].max():.2f}]")

    show_batch(batch, n=4)
