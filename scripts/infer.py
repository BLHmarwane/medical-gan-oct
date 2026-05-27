"""Run G_AB on one degraded OCT image and save a visual comparison panel."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image

from src.data.dataset import build_transform
from src.training.trainer import CycleGANTrainer
from src.utils.visualize import denormalize


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_trainer(cfg: dict, checkpoint: Path, device: torch.device) -> CycleGANTrainer:
    trainer = CycleGANTrainer(
        device=device,
        ngf=cfg["model"]["ngf"],
        ndf=cfg["model"]["ndf"],
        n_resnet=cfg["model"]["n_resnet"],
        use_cbam=cfg["model"]["use_cbam"],
        lr=cfg["training"]["lr"],
        beta1=cfg["training"]["beta1"],
        lambda_cyc=cfg["training"]["lambda_cyc"],
        lambda_id=cfg["training"]["lambda_id"],
        pool_size=cfg["training"]["pool_size"],
        n_epochs=cfg["training"]["n_epochs"],
        decay_start=cfg["training"]["decay_start"],
    )
    trainer.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    trainer.G_AB.eval()
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--clean", default=None, help="Optional clean reference image.")
    parser.add_argument("--output", default="logs/inference/panel.png")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = pick_device(cfg["training"]["device"])
    transform = build_transform(cfg["data"]["image_size"])
    trainer = load_trainer(cfg, Path(args.checkpoint), device)

    degraded_img = Image.open(args.image).convert("L")
    tensor = transform(degraded_img).unsqueeze(0).to(device)
    with torch.no_grad():
        enhanced = denormalize(trainer.G_AB(tensor))[0, 0].cpu().numpy()

    degraded = denormalize(tensor)[0, 0].cpu().numpy()
    panels = [("Degraded input", degraded), ("G_AB enhanced", enhanced)]
    if args.clean:
        clean = denormalize(transform(Image.open(args.clean).convert("L")).unsqueeze(0))[0, 0].numpy()
        panels.append(("Clean target", clean))

    output = project_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
