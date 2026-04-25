"""Train CycleGAN+CBAM on OCT images.

Usage:
    python scripts/train.py --config configs/cyclegan_cbam.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.data.dataset import UnpairedOCTDataset
from src.training.trainer import CycleGANTrainer
from src.utils.visualize import denormalize


# ----------------------------------------------------------------------

def pick_device(name: str) -> torch.device:
    """Resolve 'auto' to the best available device."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_samples(
    trainer: CycleGANTrainer, batch: dict, path: Path, n: int = 4
) -> None:
    """Save a 6-row grid: A, G_AB(A), G_BA(G_AB(A)), B, G_BA(B), G_AB(G_BA(B))."""
    trainer.G_AB.eval()
    trainer.G_BA.eval()
    with torch.no_grad():
        a = batch["A"][:n].to(trainer.device)
        b = batch["B"][:n].to(trainer.device)
        fake_b = trainer.G_AB(a)
        fake_a = trainer.G_BA(b)
        rec_a = trainer.G_BA(fake_b)
        rec_b = trainer.G_AB(fake_a)
    grid = torch.cat([a, fake_b, rec_a, b, fake_a, rec_b], dim=0)
    save_image(denormalize(grid).cpu(), path, nrow=n)
    trainer.G_AB.train()
    trainer.G_BA.train()


# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    project_root = Path(__file__).resolve().parents[1]

    # --- Device ---
    device = pick_device(cfg["training"]["device"])
    print(f"device: {device}")

    # --- Data ---
    dataset = UnpairedOCTDataset(
        root_a=project_root / cfg["data"]["domain_a"],
        root_b=project_root / cfg["data"]["domain_b"],
        image_size=cfg["data"]["image_size"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    print(f"dataset size: {len(dataset)}  (batches/epoch: {len(loader)})")

    # --- Trainer ---
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

    # --- Output dirs ---
    ckpt_dir = project_root / cfg["logging"]["ckpt_dir"]
    sample_dir = project_root / cfg["logging"]["sample_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Fixed batch reused for every sample grid (so we can see progression).
    sample_batch = next(iter(loader))

    # --- Train loop ---
    n_epochs = cfg["training"]["n_epochs"]
    log_every = cfg["logging"]["log_every"]
    for epoch in range(n_epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{n_epochs}")
        for step, batch in enumerate(pbar):
            losses = trainer.train_step(batch["A"], batch["B"])
            if step % log_every == 0:
                pbar.set_postfix({k: f"{v:.2f}" for k, v in losses.items()})

        trainer.step_schedulers()

        if (epoch + 1) % cfg["logging"]["sample_every"] == 0:
            save_samples(
                trainer, sample_batch, sample_dir / f"epoch_{epoch + 1:03d}.png"
            )

        if (epoch + 1) % cfg["logging"]["ckpt_every"] == 0:
            torch.save(trainer.state_dict(), ckpt_dir / f"epoch_{epoch + 1:03d}.pt")

    torch.save(trainer.state_dict(), ckpt_dir / "final.pt")
    print("done.")


if __name__ == "__main__":
    main()
