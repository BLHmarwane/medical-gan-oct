import os
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def build_transform(image_size: int = 256) -> transforms.Compose:
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),                  # [0, 1], shape (1, H, W)
        transforms.Normalize(mean=[0.5], std=[0.5]),  # -> [-1, 1] for tanh
    ])


class UnpairedOCTDataset(Dataset):
    """Unpaired dataset for CycleGAN-style training.

    Returns a dict {'A': tensor, 'B': tensor} where A is indexed in order
    and B is sampled randomly each call — so pairings differ every epoch.
    """

    def __init__(
        self,
        root_a: str | Path,
        root_b: str | Path,
        image_size: int = 256,
        transform: transforms.Compose | None = None,
    ):
        self.files_a = _list_images(Path(root_a))
        self.files_b = _list_images(Path(root_b))

        if not self.files_a or not self.files_b:
            raise RuntimeError(f"Empty domain: |A|={len(self.files_a)}, |B|={len(self.files_b)}")

        self.transform = transform or build_transform(image_size)

    def __len__(self) -> int:
        # Length = larger domain, so every image in the bigger set is seen per epoch.
        return max(len(self.files_a), len(self.files_b))

    def __getitem__(self, index: int) -> dict:
        path_a = self.files_a[index % len(self.files_a)]
        path_b = self.files_b[random.randint(0, len(self.files_b) - 1)]

        img_a = Image.open(path_a).convert("L")  # force 1-channel
        img_b = Image.open(path_b).convert("L")

        return {
            "A": self.transform(img_a),
            "B": self.transform(img_b),
        }


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    ds = UnpairedOCTDataset(
        root_a=base / "data" / "processed" / "domain_A",
        root_b=base / "data" / "processed" / "domain_B",
        image_size=256,
    )
    print(f"len(dataset) = {len(ds)}")
    sample = ds[0]
    print(f"A: shape={tuple(sample['A'].shape)}, min={sample['A'].min():.3f}, max={sample['A'].max():.3f}")
    print(f"B: shape={tuple(sample['B'].shape)}, min={sample['B'].min():.3f}, max={sample['B'].max():.3f}")
