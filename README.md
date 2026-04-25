# Medical Image Enhancement using GANs (CycleGAN with CBAM)

## Overview

This project focuses on understanding and applying Generative Adversarial Networks (GANs) in the context of medical imaging, specifically Optical Coherence Tomography (OCT).

The goal is to develop a CycleGAN-based model capable of enhancing OCT images by learning a mapping between degraded and high-quality image domains.

This project is both a technical implementation and a learning process to deeply understand generative models.

---

## Dataset

We use the Kermany2018 OCT dataset (retinal OCT images).

Even though it is a classification dataset, it is repurposed for image enhancement using domain adaptation.

NB : The dataset is not included in this repository due to size constraints.

You can download it here:
https://www.kaggle.com/datasets/paultimothymooney/kermany2018

Then place it in:
data/raw/

---

## Problem Statement

The Kermany2018 OCT dataset is originally designed for classification tasks and does not provide:

- Paired low-quality and high-quality images  
- A clear separation between degraded and enhanced images  

However, CycleGAN requires two domains:

- Domain A: degraded images  
- Domain B: clean images  

This makes direct application impossible.

---

## Proposed Solution

We solve this problem using a synthetic degradation strategy.

- Domain B (target): original OCT images  
- Domain A (input): artificially degraded versions of the same images  

Degradations applied:

- Speckle noise (relevant for OCT imaging)  
- Gaussian noise  
- Blur (optical degradation)  
- Contrast reduction  

---

## Why This Approach Works

This approach allows:

- Creation of realistic degraded data  
- Full control over experiments  
- Reproducibility  
- Quantitative evaluation (PSNR, SSIM)  
- Training in an unpaired setting (CycleGAN)  

It reflects real-world medical constraints where paired data is rare.


---

## Data Preparation

The script `degrade_images.py` generates the dataset for training:

- Domain A → degraded images  
- Domain B → original images  

This creates the two domains required for CycleGAN training.

---

## Project Structure

```
medical-gan-oct/
├── configs/                  YAML hyperparameter files
│   ├── cyclegan_cbam.yaml          (local / MPS)
│   └── cyclegan_cbam_colab.yaml    (Colab / CUDA)
├── data/
│   ├── raw/                  OCT2017 (download separately)
│   └── processed/
│       ├── domain_A/         degraded images
│       └── domain_B/         clean images
├── notebooks/
│   └── train_colab.ipynb     Colab training notebook
├── scripts/
│   └── train.py              training entry point
├── src/
│   ├── data/                 dataset, degradation pipeline
│   ├── models/               generator, discriminator, CBAM
│   ├── losses/               adversarial, cycle, identity
│   ├── training/             trainer, image replay buffer
│   └── utils/                visualization helpers
├── checkpoints/              saved model weights (gitignored)
├── logs/samples/             sample translations per epoch (gitignored)
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Training

### Local (Mac MPS / CPU)

```bash
pip install -r requirements.txt
pip install -e .
python scripts/train.py --config configs/cyclegan_cbam.yaml
```

### Colab GPU (recommended for full runs)

1. Upload `data/processed/` to Drive at `My Drive/medical-gan-oct/data/processed/`.
2. Open `notebooks/train_colab.ipynb` in Colab.
3. Runtime → Change runtime type → GPU.
4. Run all cells.

Checkpoints and sample PNGs stream directly to Drive.

---

## Technical Stack

- Python 3.10+
- PyTorch / torchvision
- OpenCV, Pillow, NumPy
- Matplotlib (visualization)
- PyYAML, tqdm (training)

---


## Objective

The objective of this project is to:

- Understand GANs deeply  
- Apply them to medical imaging  
- Build a clean and reproducible pipeline  
 

---

## Author

Marwane BEL HAMRA  
Engineer in Computer Vision and Medical Image Processing  
Université Clermont Auvergne
