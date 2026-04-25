"""CycleGAN trainer: holds the 4 networks, 2 optimizers, 2 buffers,
and runs one full G + D update per `train_step`.
"""
from __future__ import annotations

import itertools

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR

from src.losses.cyclegan_losses import (
    GANLoss,
    cycle_consistency_loss,
    identity_loss,
)
from src.models.discriminator import PatchDiscriminator
from src.models.generator import ResnetGenerator
from src.training.buffer import ImageBuffer


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def init_weights(model: nn.Module, std: float = 0.02) -> None:
    """Apply N(0, 0.02) init to Conv/Linear, N(1, 0.02) to InstanceNorm.

    Standard CycleGAN initialization — much smaller than PyTorch defaults,
    helps avoid early-training instability.
    """
    def fn(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
            nn.init.normal_(m.weight, 1.0, std)
            nn.init.zeros_(m.bias)

    model.apply(fn)


def set_requires_grad(nets, flag: bool) -> None:
    """Toggle requires_grad on every parameter of one or more modules."""
    if not isinstance(nets, (list, tuple)):
        nets = [nets]
    for net in nets:
        for p in net.parameters():
            p.requires_grad = flag


def linear_decay_lambda(n_epochs: int, decay_start: int):
    """LR multiplier: 1.0 for [0, decay_start), then linear to 0 at n_epochs."""
    def f(epoch: int) -> float:
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / max(1, n_epochs - decay_start))
    return f


# ----------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------

class CycleGANTrainer:
    def __init__(
        self,
        device: torch.device,
        ngf: int = 64,
        ndf: int = 64,
        n_resnet: int = 9,
        use_cbam: bool = True,
        lr: float = 2e-4,
        beta1: float = 0.5,
        lambda_cyc: float = 10.0,
        lambda_id: float = 5.0,
        pool_size: int = 50,
        n_epochs: int = 200,
        decay_start: int = 100,
    ):
        self.device = device
        self.lambda_cyc = lambda_cyc
        self.lambda_id = lambda_id

        # --- Networks ---
        self.G_AB = ResnetGenerator(1, 1, ngf, n_resnet, use_cbam).to(device)
        self.G_BA = ResnetGenerator(1, 1, ngf, n_resnet, use_cbam).to(device)
        self.D_A = PatchDiscriminator(1, ndf).to(device)
        self.D_B = PatchDiscriminator(1, ndf).to(device)
        for net in (self.G_AB, self.G_BA, self.D_A, self.D_B):
            init_weights(net)

        # --- Loss (reused for all GAN terms) ---
        self.gan_loss = GANLoss().to(device)

        # --- Optimizers (one per pair of networks) ---
        self.opt_G = optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
            lr=lr, betas=(beta1, 0.999),
        )
        self.opt_D = optim.Adam(
            itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
            lr=lr, betas=(beta1, 0.999),
        )

        # --- Schedulers (linear decay) ---
        lr_fn = linear_decay_lambda(n_epochs, decay_start)
        self.sched_G = LambdaLR(self.opt_G, lr_lambda=lr_fn)
        self.sched_D = LambdaLR(self.opt_D, lr_lambda=lr_fn)

        # --- Image replay buffers ---
        self.buf_A = ImageBuffer(pool_size)
        self.buf_B = ImageBuffer(pool_size)

    # ------------------------------------------------------------------

    def train_step(self, real_a: torch.Tensor, real_b: torch.Tensor) -> dict:
        real_a = real_a.to(self.device)
        real_b = real_b.to(self.device)

        # ============== Generator update ==============
        set_requires_grad([self.D_A, self.D_B], False)
        self.opt_G.zero_grad()

        fake_b = self.G_AB(real_a)
        fake_a = self.G_BA(real_b)
        rec_a = self.G_BA(fake_b)
        rec_b = self.G_AB(fake_a)

        loss_gan_AB = self.gan_loss(self.D_B(fake_b), is_real=True)
        loss_gan_BA = self.gan_loss(self.D_A(fake_a), is_real=True)
        loss_cyc = cycle_consistency_loss(real_a, rec_a) + cycle_consistency_loss(real_b, rec_b)

        if self.lambda_id > 0:
            id_a = self.G_BA(real_a)
            id_b = self.G_AB(real_b)
            loss_id = identity_loss(real_a, id_a) + identity_loss(real_b, id_b)
        else:
            loss_id = torch.zeros((), device=self.device)

        loss_G = (
            loss_gan_AB + loss_gan_BA
            + self.lambda_cyc * loss_cyc
            + self.lambda_id * loss_id
        )
        loss_G.backward()
        self.opt_G.step()

        # ============== Discriminator update ==============
        set_requires_grad([self.D_A, self.D_B], True)
        self.opt_D.zero_grad()

        # D_A: real_a vs replayed fake_a
        fake_a_buf = self.buf_A.query(fake_a)
        loss_D_A = 0.5 * (
            self.gan_loss(self.D_A(real_a), is_real=True)
            + self.gan_loss(self.D_A(fake_a_buf.detach()), is_real=False)
        )

        # D_B: real_b vs replayed fake_b
        fake_b_buf = self.buf_B.query(fake_b)
        loss_D_B = 0.5 * (
            self.gan_loss(self.D_B(real_b), is_real=True)
            + self.gan_loss(self.D_B(fake_b_buf.detach()), is_real=False)
        )

        (loss_D_A + loss_D_B).backward()
        self.opt_D.step()

        return {
            "G":        loss_G.item(),
            "G_gan_AB": loss_gan_AB.item(),
            "G_gan_BA": loss_gan_BA.item(),
            "G_cyc":    loss_cyc.item(),
            "G_id":     loss_id.item(),
            "D_A":      loss_D_A.item(),
            "D_B":      loss_D_B.item(),
        }

    def step_schedulers(self) -> None:
        self.sched_G.step()
        self.sched_D.step()

    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "G_AB": self.G_AB.state_dict(),
            "G_BA": self.G_BA.state_dict(),
            "D_A":  self.D_A.state_dict(),
            "D_B":  self.D_B.state_dict(),
            "opt_G": self.opt_G.state_dict(),
            "opt_D": self.opt_D.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.G_AB.load_state_dict(state["G_AB"])
        self.G_BA.load_state_dict(state["G_BA"])
        self.D_A.load_state_dict(state["D_A"])
        self.D_B.load_state_dict(state["D_B"])
        self.opt_G.load_state_dict(state["opt_G"])
        self.opt_D.load_state_dict(state["opt_D"])


# ----------------------------------------------------------------------
# Sanity check
# ----------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cpu")  # MPS/CUDA: change here later
    trainer = CycleGANTrainer(device=device, n_resnet=3, pool_size=4)  # tiny for speed

    # Two random batches, one step each — losses should be finite.
    for step in range(2):
        a = torch.randn(2, 1, 64, 64)
        b = torch.randn(2, 1, 64, 64)
        losses = trainer.train_step(a, b)
        line = " | ".join(f"{k}={v:.3f}" for k, v in losses.items())
        print(f"step {step}: {line}")
