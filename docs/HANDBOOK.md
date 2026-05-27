# CycleGAN + CBAM for OCT — Project Handbook

> A living document. Read this when you're confused about where the project
> stands, why a piece exists, or what to do next. Update it at the end of
> every major step (see the bottom of this file for the procedure).

---

## TL;DR — Where we are right now

- **Last updated**: 2026-05-27
- **Phase**: ✅ Full prototype implemented · ⏳ longer GPU validation pending
- **What works**:
  - Synthetic degradation pipeline (clean OCT → degraded OCT)
  - Unpaired Dataset + DataLoader
  - ResNet generator with CBAM attention (~11 M params)
  - PatchGAN discriminator (~2.7 M params)
  - All four CycleGAN losses (adversarial, cycle, identity, ×2 directions)
  - Image replay buffer for D stability
  - Trainer that orchestrates one full G + D update
  - YAML configs + `scripts/train.py` entry point
  - Colab notebook with public GitHub clone and Drive integration
  - `losses.csv`, `latest.pt`, checkpoints, fixed sample grids
  - Exploratory PSNR/SSIM evaluation (`scripts/evaluate.py`)
  - Single-image visual inference panels (`scripts/infer.py`)
- **What's still pending**: long GPU training, robust clinical validation, and a completed CBAM ablation.
- **Next concrete step**: run the demo config on Colab GPU, inspect sample grids, then evaluate `latest.pt`.

---

## 1. The goal in one sentence

Train an unpaired image-to-image translation network (**CycleGAN with CBAM attention**) to convert **degraded OCT scans into clean ones**, by learning the mapping from noise + blur + low contrast back to crisp retinal layers — without ever seeing paired before/after examples.

## 2. Why this is interesting

In real medical imaging, **paired data almost never exists**: you can't acquire the same retina under both perfect and degraded conditions. CycleGAN solves this by using *cycle consistency* — translate A→B→A and demand you get the original A back. This forces the network to learn a meaningful mapping rather than fabricating images.

Adding **CBAM** (Channel + Spatial attention) on top tells the network "focus on the important channels and the important regions." For OCT, retinal layers occupy specific spatial bands, and certain feature channels encode noise vs. structure — CBAM lets the model learn to amplify both.

---

## 3. The pieces we built (in build order)

| # | Phase | Files | What it does |
|---|-------|-------|--------------|
| 1 | Data preparation | `src/data/degrade.py` | Generates degraded copies (speckle + Gaussian noise + blur + low contrast) |
| 2 | Dataset | `src/data/dataset.py` | Loads unpaired (A, B) batches as tensors in `[-1, 1]` |
| 3 | Visualization | `src/utils/visualize.py` | Renders a batch grid for sanity checks |
| 4 | Project structure | top-level dirs | Modular layout (`src/`, `scripts/`, `configs/`, …) |
| 5 | Generator | `src/models/generator.py`, `src/models/cbam.py` | ResNet generator with CBAM in every residual block |
| 6 | Discriminator | `src/models/discriminator.py` | 70×70 PatchGAN |
| 7 | Losses | `src/losses/cyclegan_losses.py` | LSGAN adversarial + L1 cycle + L1 identity |
| 8 | Replay buffer | `src/training/buffer.py` | Stabilizes D by mixing fresh and historical fakes |
| 9 | Trainer | `src/training/trainer.py` | One full G + D update per call |
| 10 | Entry point | `scripts/train.py`, `configs/cyclegan_cbam.yaml` | One-command training |
| 11 | Colab integration | `notebooks/train_colab.ipynb`, `configs/cyclegan_cbam_demo.yaml` | Public clone, GPU demo training, Drive persistence |
| 12 | Evaluation + inference | `scripts/evaluate.py`, `scripts/infer.py` | PSNR/SSIM and visual panels for experimental analysis |

---

## 4. Architecture at a glance

### Data flow during training

```
  domain_A (degraded)              domain_B (clean)
        │                                 │
        ▼                                 ▼
   ┌──────┐                          ┌──────┐
   │ G_AB │──fake_B──┐         ┌────│ G_BA │──fake_A──┐
   └──────┘          │         │    └──────┘          │
        ▲            │         │         ▲            │
        │     ┌──────┴──┐ ┌────┴──────┐  │     ┌──────┴──┐
        │     │ D_B     │ │ D_A       │  │     │ recover │
        │     │ real?   │ │ real?     │  │     │ via G_AB│
        │     └─────────┘ └───────────┘  │     └─────────┘
        │                                 │
   ┌────┴────────────────┐    ┌──────────┴────────────┐
   │ G_BA(fake_B) ≈ A    │    │ G_AB(fake_A) ≈ B      │  ← cycle consistency
   └─────────────────────┘    └───────────────────────┘
```

### The four losses

| Loss | Form | Pushes G to… | Pushes D to… |
|------|------|--------------|--------------|
| `L_GAN` (adversarial) | MSE on D's logits | produce fakes D thinks are real | tell real apart from fake |
| `L_cyc` (cycle) | L1 on round-trip reconstruction | preserve content | — |
| `L_id` (identity) | L1 on G(target-domain image) | leave already-clean images alone | — |

### Generator (ResNet + CBAM)

```
input (1×256×256)
  → ReflectionPad(3) → Conv7×7 → IN → ReLU                      (64×256×256)
  → Conv3×3 stride 2 → IN → ReLU                                (128×128×128)
  → Conv3×3 stride 2 → IN → ReLU                                (256×64×64)
  → 9 × [ResnetBlock with CBAM]                                 (256×64×64)
  → ConvT3×3 stride 2 → IN → ReLU                               (128×128×128)
  → ConvT3×3 stride 2 → IN → ReLU                               (64×256×256)
  → ReflectionPad(3) → Conv7×7 → Tanh                           (1×256×256, [-1,1])
```

### Discriminator (PatchGAN)

```
input (1×256×256)
  → Conv4×4 s2 → LeakyReLU                                      (64×128×128)
  → Conv4×4 s2 → IN → LeakyReLU                                 (128×64×64)
  → Conv4×4 s2 → IN → LeakyReLU                                 (256×32×32)
  → Conv4×4 s1 → IN → LeakyReLU                                 (512×31×31)
  → Conv4×4 s1                                                  (1×30×30 logits)
```

Each cell of the 30×30 output classifies one 70×70 patch as real or fake.

---

## 5. Key design choices (and why)

| Choice | Why |
|--------|-----|
| **Reflection padding** instead of zero padding | Zero padding produces visible border artifacts; reflection mirrors content. Critical for image-to-image. |
| **InstanceNorm**, not BatchNorm | Translation tasks want per-image statistics; BatchNorm entangles different images in a batch. |
| **LSGAN (MSE)** instead of vanilla GAN (BCE) | Far more stable, less mode collapse, no sigmoid needed. |
| **9 ResNet blocks** at the bottleneck | Standard CycleGAN choice for 256×256 inputs. Skip connections make optimization easy. |
| **CBAM inside each residual block** | Lets the network learn to focus on relevant feature channels and spatial regions. Tiny parameter overhead. |
| **PatchGAN** instead of whole-image D | Forces G to make every local region realistic — important for OCT texture. |
| **Cycle weight λ_cyc = 10** | The original CycleGAN setting. Cycle loss dominates; without it, G could fabricate any plausible image. |
| **Identity weight λ_id = 5** | Encourages G to leave target-domain images alone. Stabilizes early training. |
| **Replay buffer of 50** | D learns from a *history* of fakes, not just the freshest one. Reduces oscillation. |
| **Adam β₁ = 0.5** | Lower than the default 0.9. Standard for GANs — reduces momentum in noisy gradient direction. |
| **N(0, 0.02) weight init** | Smaller than PyTorch defaults; helps avoid initial instability. |
| **Tanh at G output, no sigmoid at D** | Tanh gives `[-1, 1]` matching our normalization; LSGAN expects raw logits at D output. |
| **batch_size = 1** | CycleGAN default; works because InstanceNorm doesn't need a real batch. |

---

## 6. Project layout

```
medical-gan-oct/
├── configs/                       YAML hyperparameters
│   ├── cyclegan_cbam.yaml         (local: device=mps/cpu)
│   └── cyclegan_cbam_colab.yaml   (Colab: device=cuda)
├── data/
│   ├── raw/                       full Kermany dataset (download separately)
│   └── processed/
│       ├── domain_A/              degraded
│       └── domain_B/              clean
├── docs/
│   └── HANDBOOK.md                this file
├── notebooks/
│   └── train_colab.ipynb
├── scripts/
│   └── train.py                   THE entry point: `python scripts/train.py --config ...`
├── src/
│   ├── data/                      dataset, degradation
│   ├── models/                    generator, discriminator, CBAM
│   ├── losses/                    cyclegan_losses
│   ├── training/                  trainer, buffer
│   └── utils/                     visualize
├── checkpoints/                   gitignored — saved weights stream here
├── logs/samples/                  gitignored — sample PNGs per epoch
├── pyproject.toml                 makes src a proper package (`pip install -e .`)
├── requirements.txt
└── README.md
```

---

## 7. How to run

### Local (Mac, MPS or CPU)

```bash
cd /Users/mar1/HOME/1 Projects/medical-gan-oct
pip install -r requirements.txt
pip install -e .
python scripts/train.py --config configs/cyclegan_cbam.yaml
```

Time budget on M-series Mac with 100 images: ~5–10 s/epoch → **30–60 min for 200 epochs**.

### Colab (recommended for full runs)

See **[Colab setup](#colab-setup-private-repo)** below.

Time budget on Colab T4: ~30 s/epoch → **~100 min for 200 epochs**.

---

## 8. Reading the sample grid

`scripts/train.py` saves a PNG to `logs/samples/epoch_NNN.png` every epoch. The grid has 6 rows × `n` columns. Top-to-bottom:

1. **real A** (degraded input)
2. **G_AB(A)** — should look clean (the goal)
3. **G_BA(G_AB(A))** — should reconstruct row 1 (cycle works)
4. **real B** (clean input)
5. **G_BA(B)** — should look degraded
6. **G_AB(G_BA(B))** — should reconstruct row 4

Early epochs (1–30): rows 2 and 5 look like noise. By epoch 50–80 they should resemble the target domain. Watch the cycle reconstruction rows (3 and 6) — they're the strongest indicator that training hasn't collapsed.

---

## 9. Healthy vs. unhealthy training signals

| Signal | Healthy | Unhealthy |
|--------|---------|-----------|
| `D_A`, `D_B` losses | hover around **0.2–0.5** | `< 0.1` (D dominates → G stops learning) or `> 1.0` (D too weak) |
| `G_gan_AB`, `G_gan_BA` | trend down, oscillate around 0.3–1.0 | `> 5` and stuck = G can't fool D |
| `G_cyc` | drops fast in first 20 epochs, stabilizes | stuck high = G can't preserve content |
| Sample grid row 3 vs row 1 | nearly identical | very different = cycle broken |
| Sample grid row 2 | gradually denoises | stays noisy or collapses to a single image (mode collapse) |

If D dominates: lower D's learning rate, increase G's, or add noise to D's inputs.
If G dominates: usually fine — let it run; D will catch up.

---

## 10. What's next

1. **Run the public Colab demo** — use `configs/cyclegan_cbam_demo.yaml`, inspect samples, evaluate `checkpoints/demo/latest.pt`.
2. **Longer training** — train 200 epochs or more on GPU, ideally with more than the current 100-image subset.
3. **Ablation: CBAM on vs. off** — run `configs/cyclegan_no_cbam_demo.yaml` and compare PSNR/SSIM only after both runs are complete.
4. **Scale to full Kermany dataset** — current demo data is intentionally small; full data should improve generalization.
5. **Optional: perceptual loss** (LPIPS or VGG features) — potentially improves perceived quality even when PSNR is similar.

---

## 11. Colab setup (public repo)

The repository is designed to be public, so Colab can clone it directly without extra authentication.

1. Open `notebooks/train_colab.ipynb` from the README Colab badge.
2. Upload `processed.zip` to `MyDrive/processed.zip`. The zip should contain `data/processed/domain_A` and `data/processed/domain_B`.
3. Select **Runtime → Change runtime type → GPU**.
4. Run all cells.

The notebook installs the repo, links checkpoints/logs to Drive, runs the short demo config, previews the latest sample grid, and computes exploratory PSNR/SSIM.

---

## 12. Glossary (revision-ready definitions)

- **CycleGAN** — A GAN architecture with two generators (A→B and B→A) and a *cycle consistency* loss that demands `G_BA(G_AB(a)) ≈ a`. Enables learning from unpaired data.
- **CBAM** — Convolutional Block Attention Module. Sequentially applies channel attention (which feature maps matter?) then spatial attention (where to focus?).
- **InstanceNorm** — Normalizes each (image, channel) slice independently. Per-image statistics, unlike BatchNorm.
- **PatchGAN** — A discriminator that classifies overlapping patches of the input rather than the whole image. Output is a grid of logits.
- **LSGAN** — Least-Squares GAN. Uses MSE loss against {0, 1} targets instead of BCE. More stable than vanilla GAN.
- **Cycle consistency loss** — `‖G_BA(G_AB(a)) − a‖₁`. The constraint that makes CycleGAN's unpaired learning meaningful.
- **Identity loss** — `‖G_AB(b) − b‖₁`. Encourages G to act as identity when fed its target-domain input. Stabilizes early training.
- **Image replay buffer** — A pool of previously-generated fakes. When training D, mix fresh and historical fakes (50/50) to prevent overfitting to current G.
- **Reflection padding** — Padding that mirrors the image content. Avoids the artifacts you'd get from zero-padding at borders.
- **Receptive field** — The region of the input that influences a given output cell. PatchGAN's last layer has a 70×70 receptive field.

---

## 13. How to update this handbook

After each major step (e.g. evaluation script added, ablation run, full dataset switch), edit:

1. **TL;DR** — bump `Last updated`, change `Phase`, update `What works` / `What's blocked` / `Next concrete step`.
2. **Section 3** (pieces) — add a row to the table.
3. **Section 5** (design choices) — add new rows if you adopted a new decision.
4. **Section 10** (what's next) — strike out completed items, add new ones.

Commit with a message like `Update HANDBOOK after <step name>`.

The point of this file is to be the **single place** you read when you've been away from the project for a while. Keep it honest. If something's broken or experimental, say so.
