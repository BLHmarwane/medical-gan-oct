"""Image replay buffer for stable discriminator training.

When training D, we sample from a pool of recent generator outputs instead
of always using the freshest fakes. This prevents D from overfitting to
the current G's quirks and reduces training oscillation.

Reference: Shrivastava et al., "Learning from Simulated and Unsupervised
Images" (2017); also used in CycleGAN (Zhu et al. 2017).
"""
import random

import torch


class ImageBuffer:
    """Fixed-size pool of generated images.

    On `query(batch)`:
      - If pool has room: append each image and return it as-is.
      - If pool is full: with p=0.5, swap with a random slot (return the
        old image, store the new); else return the new image, leave pool
        untouched.
    """

    def __init__(self, pool_size: int = 50):
        self.pool_size = pool_size
        self.images: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        """Args: images shape (B, C, H, W). Returns same shape."""
        if self.pool_size == 0:
            return images

        out = []
        for image in images:
            image = image.unsqueeze(0).detach()  # (1, C, H, W), no grad

            if len(self.images) < self.pool_size:
                self.images.append(image)
                out.append(image)
            elif random.random() < 0.5:
                idx = random.randint(0, self.pool_size - 1)
                old = self.images[idx].clone()
                self.images[idx] = image
                out.append(old)
            else:
                out.append(image)

        return torch.cat(out, dim=0)


if __name__ == "__main__":
    # Sanity: pool fills up, then mixes old and new.
    buf = ImageBuffer(pool_size=4)
    for step in range(6):
        fake = torch.full((2, 1, 8, 8), float(step))
        out = buf.query(fake)
        # mean of the returned tensors tells us which "step" each came from
        means = out.mean(dim=(1, 2, 3)).tolist()
        print(f"step {step}: pool_size={len(buf.images)}, returned means={means}")
