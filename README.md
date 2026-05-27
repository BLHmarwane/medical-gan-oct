# Medical GAN OCT - CycleGAN + CBAM

Prototype for unsupervised retinal OCT image enhancement with a CycleGAN architecture augmented with CBAM attention.

The goal is to learn a mapping from synthetically degraded OCT scans to cleaner retinal OCT images while keeping the workflow reproducible enough for portfolio review and interview discussion.

[Open in Colab](https://colab.research.google.com/github/BLHmarwane/medical-gan-oct/blob/main/notebooks/train_colab.ipynb)

## Current Status

Implemented:

- synthetic OCT degradation pipeline: speckle noise, Gaussian noise, blur, contrast reduction;
- unpaired CycleGAN dataset with tensors normalized to `[-1, 1]`;
- ResNet generator with optional CBAM attention in residual blocks;
- PatchGAN discriminator, LSGAN loss, cycle loss, identity loss, replay buffer;
- training entry point with YAML configs, checkpoints, `latest.pt`, sample grids, and `losses.csv`;
- Colab notebook for a short GPU demo run;
- exploratory PSNR/SSIM evaluation on synthetic A_i/B_i pairs;
- inference script for visual panels.

Pending:

- full long training run on a larger dataset;
- clinically robust validation;
- completed CBAM vs no-CBAM ablation;
- comparison with supervised denoising/restoration baselines.

This repository should be read as a working research prototype, not a validated clinical model.

## Dataset

The project uses the Kermany2018 retinal OCT dataset:

https://www.kaggle.com/datasets/paultimothymooney/kermany2018

The dataset is not committed to GitHub. Download it separately and place it under:

```text
data/raw/OCT2017/
```

The training domains are:

```text
data/processed/domain_A/   degraded OCT images
data/processed/domain_B/   clean OCT images
```

`domain_A` and `domain_B` are created from the same source images, so evaluation can report exploratory PSNR/SSIM against the clean synthetic target. Training still uses CycleGAN-style unpaired sampling.

## Interview Demo

Setup:

```bash
pip install -r requirements.txt
pip install -e .
```

Train the short demo run:

```bash
python scripts/train.py --config configs/cyclegan_cbam_demo.yaml
```

Evaluate PSNR/SSIM:

```bash
python scripts/evaluate.py \
  --config configs/cyclegan_cbam_demo.yaml \
  --checkpoint checkpoints/demo/latest.pt \
  --limit 100 \
  --output logs/evaluation/demo_metrics.csv
```

Generate a visual panel:

```bash
python scripts/infer.py \
  --config configs/cyclegan_cbam_demo.yaml \
  --checkpoint checkpoints/demo/latest.pt \
  --image data/processed/domain_A/A_0.png \
  --clean data/processed/domain_B/B_0.png \
  --output logs/inference/demo_panel.png
```

For Colab, open the notebook from the badge above, upload `processed.zip` to `MyDrive/processed.zip`, select a GPU runtime, and run all cells.

## Architecture

Generator:

```text
1x256x256 OCT
  -> 7x7 conv stem
  -> 2 downsampling blocks
  -> 9 ResNet blocks with CBAM
  -> 2 upsampling blocks
  -> tanh output in [-1, 1]
```

Discriminator:

```text
PatchGAN discriminator with raw logits for LSGAN training
```

Training losses:

- adversarial LSGAN loss for both directions;
- cycle consistency L1 loss;
- identity L1 loss;
- image replay buffer for discriminator stability.

## Project Layout

```text
configs/          Training configs, including demo and no-CBAM ablation configs
docs/             Project handbook and interview notes
notebooks/        Public Colab training notebook
scripts/          train.py, evaluate.py, infer.py
src/              Data, models, losses, training, visualization
checkpoints/      Ignored training checkpoints
logs/             Ignored samples, metrics, inference panels
data/             Ignored raw and processed datasets
```

## Honest Pitch

This is a prototype of unsupervised retinal OCT restoration. I built the full CycleGAN+CBAM pipeline, the synthetic degradation strategy, the Colab training workflow, and exploratory PSNR/SSIM evaluation. The remaining work is to run longer GPU experiments, complete the CBAM ablation, and validate the method on a broader dataset before making any clinical claim.

## Author

Marwane BELHAMRA

Biomedical engineer - computer vision and medical image processing
