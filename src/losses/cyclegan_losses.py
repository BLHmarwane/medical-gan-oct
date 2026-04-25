"""CycleGAN loss functions.

- GANLoss: LSGAN-style adversarial loss (MSE on raw logits).
- Cycle and identity losses are plain L1 (no wrapper needed).
"""
import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """LSGAN adversarial loss.

    Builds a target tensor matching the prediction's shape on every call,
    so it works with any PatchGAN output size and any device.
    """

    def __init__(self, real_label: float = 1.0, fake_label: float = 0.0):
        super().__init__()
        self.real_label = real_label
        self.fake_label = fake_label
        self.loss = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        value = self.real_label if is_real else self.fake_label
        target = torch.full_like(prediction, value)
        return self.loss(prediction, target)


def cycle_consistency_loss(
    real: torch.Tensor, reconstructed: torch.Tensor
) -> torch.Tensor:
    """L1 between original image and its A->B->A (or B->A->B) reconstruction."""
    return nn.functional.l1_loss(reconstructed, real)


def identity_loss(real: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
    """L1 encouraging G to act as identity when fed its target-domain input."""
    return nn.functional.l1_loss(identity, real)


if __name__ == "__main__":
    # Sanity: shapes and finite values.
    gan = GANLoss()
    pred = torch.randn(2, 1, 30, 30)
    loss_real = gan(pred, is_real=True)
    loss_fake = gan(pred, is_real=False)
    print(f"GANLoss real target: {loss_real.item():.4f}")
    print(f"GANLoss fake target: {loss_fake.item():.4f}")

    a = torch.randn(2, 1, 256, 256)
    a_rec = a + 0.01 * torch.randn_like(a)
    print(f"cycle loss (small noise): {cycle_consistency_loss(a, a_rec).item():.4f}")
    print(f"identity loss (same): {identity_loss(a, a).item():.4f}")
