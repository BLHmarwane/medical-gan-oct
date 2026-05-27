"""Evaluate OCT enhancement with PSNR and SSIM.

The synthetic degradation pipeline keeps an implicit pair A_i -> B_i, so this
script can compare the degraded baseline and the enhanced output against the
clean target. These numbers are useful for exploration, not clinical validation.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from src.data.dataset import build_transform
from src.training.trainer import CycleGANTrainer
from src.utils.visualize import denormalize

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def numeric_id(path: Path) -> str | None:
    match = re.search(r"(\d+)$", path.stem)
    return match.group(1) if match else None


def paired_files(domain_a: Path, domain_b: Path, limit: int | None) -> list[tuple[Path, Path]]:
    files_a = {numeric_id(p): p for p in domain_a.iterdir() if p.suffix.lower() in IMG_EXTS}
    files_b = {numeric_id(p): p for p in domain_b.iterdir() if p.suffix.lower() in IMG_EXTS}
    keys = sorted((k for k in files_a.keys() & files_b.keys() if k is not None), key=int)
    pairs = [(files_a[k], files_b[k]) for k in keys]
    return pairs[:limit] if limit else pairs


def image01(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    x = pred.astype(np.float64)
    y = target.astype(np.float64)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mux = x.mean()
    muy = y.mean()
    varx = x.var()
    vary = y.var()
    cov = ((x - mux) * (y - muy)).mean()
    return float(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux**2 + muy**2 + c1) * (varx + vary + c2)))


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
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    trainer.load_state_dict(state)
    trainer.G_AB.eval()
    return trainer


def finite_mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return float(np.mean(finite)) if finite else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="logs/evaluation/metrics.csv")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(Path(args.config).read_text())
    image_size = cfg["data"]["image_size"]
    pairs = paired_files(
        project_root / cfg["data"]["domain_a"],
        project_root / cfg["data"]["domain_b"],
        args.limit,
    )
    if not pairs:
        raise RuntimeError("No A_i/B_i pairs found. Run the degradation step first.")

    device = pick_device(cfg["training"]["device"])
    trainer = load_trainer(cfg, Path(args.checkpoint), device) if args.checkpoint else None
    transform = build_transform(image_size)

    rows: list[dict[str, float | str]] = []
    for path_a, path_b in pairs:
        degraded = image01(path_a, image_size)
        clean = image01(path_b, image_size)
        row: dict[str, float | str] = {
            "image": path_a.name,
            "baseline_psnr": psnr(degraded, clean),
            "baseline_ssim": ssim(degraded, clean),
        }
        if trainer is not None:
            tensor = transform(Image.open(path_a).convert("L")).unsqueeze(0).to(device)
            with torch.no_grad():
                enhanced = denormalize(trainer.G_AB(tensor))[0, 0].cpu().numpy()
            row["enhanced_psnr"] = psnr(enhanced, clean)
            row["enhanced_ssim"] = ssim(enhanced, clean)
        rows.append(row)

    output = project_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"evaluated pairs: {len(rows)}")
    print(f"baseline PSNR: {finite_mean([float(r['baseline_psnr']) for r in rows]):.3f}")
    print(f"baseline SSIM: {finite_mean([float(r['baseline_ssim']) for r in rows]):.3f}")
    if trainer is not None:
        print(f"enhanced PSNR: {finite_mean([float(r['enhanced_psnr']) for r in rows]):.3f}")
        print(f"enhanced SSIM: {finite_mean([float(r['enhanced_ssim']) for r in rows]):.3f}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
