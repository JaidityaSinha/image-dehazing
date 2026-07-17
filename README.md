![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Image%20Dehazing-green)

<img width="642" height="318" alt="image" src="https://github.com/user-attachments/assets/3b4fcd37-636b-4a67-bdb6-88179d5134c5" />


*Real World Sample — Hazy | Dehazed*

# Image Dehazing using Deep Learning

A PyTorch implementation and comparative study of **U-Net-based image dehazing architectures**, including lightweight depthwise separable variants, channel attention mechanisms, FFT-enhanced skip connections, and SSIM-based optimization.

This project was developed as part of a **Summer Internship in Deep Learning and Computer Vision**.

---

# Overview

Atmospheric haze reduces image visibility by scattering light, resulting in lower contrast and color distortion. This project investigates multiple U-Net architectures for restoring haze-free images from a single hazy image while exploring the trade-offs between reconstruction quality, parameter efficiency, and computational complexity.

Six architectures were implemented and evaluated on the **ITS (Indoor Training Set)** and **RESIDE-6K** datasets.

Evaluation metrics include:

- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)

---

# Implemented Architectures

- Baseline U-Net
- Channel Attention U-Net
- Depthwise Separable U-Net
- Depthwise + Skip Channel Attention U-Net
- Depthwise + Full Channel Attention U-Net
- Depthwise + Skip Channel Attention + FFT U-Net

---

# Highlights

- Implemented six U-Net-based image dehazing architectures in PyTorch.
- Compared standard convolutions with lightweight depthwise separable convolutions.
- Investigated different placements of Channel Attention modules.
- Implemented frequency-domain refinement using FFT on skip connections.
- Compared **MSE Loss** and **SSIM Loss** for all major architectures.
- Performed ablation studies on attention placement.
- Evaluated parameter efficiency alongside PSNR and SSIM.
- Benchmarked results against published image dehazing methods (2019–2024).

---

# Features

- Custom PyTorch dataset pipeline
- Modular U-Net implementations
- Channel Attention (SE) blocks
- Depthwise Separable Convolutions
- FFT-based frequency enhancement module
- Automatic checkpoint saving
- Early stopping
- Training and validation loss visualization
- PSNR and SSIM evaluation
- Visual comparison generation

---

# Project Structure

```text
image-dehazing/
│
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   ├── unet.py
│   ├── channel_attention_unet.py
│   ├── depthwise_unet.py
│   ├── depthwise_skip_ca.py
│   ├── depthwise_full_ca.py
│   └── depthwise_skip_ca_fft.py
│
├── outputs/
│
├── config.py
├── dataset.py
├── train.py
├── eval.py
├── test.py
└── README.md
```

---

# Datasets

> **Note:** The datasets are not included in this repository due to their size.

Experiments were conducted on the following benchmark datasets.

## ITS (Indoor Training Set)

- Indoor synthetic haze dataset
- SOTS-Indoor used for evaluation
- Paired hazy and clear images

## RESIDE-6K

- Outdoor synthetic haze dataset
- Greater scene diversity
- Separate held-out test split used for evaluation

---

# Training Configuration

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 8 |
| Image Size | 256 × 256 |
| Scheduler | ReduceLROnPlateau |
| Output Activation | Sigmoid |
| Loss Functions | Mean Squared Error (MSE), SSIM Loss |

---

# Model Complexity

| Model | Trainable Parameters |
|--------|--------------------:|
| Baseline U-Net | 31,031,875 |
| Channel Attention U-Net | 31,249,987 |
| Depthwise Separable U-Net | 5,988,382 |
| Depthwise + Skip Channel Attention | 6,031,902 |
| Depthwise + Full Channel Attention | 6,206,494 |
| Depthwise + Skip Channel Attention + FFT | 8,821,022 |

---

# Results (SSIM Loss)

## ITS

| Model | PSNR (dB) | SSIM |
|--------|----------:|------:|
| Baseline U-Net | 26.42 | 0.9532 |
| **Channel Attention U-Net** | **27.17** | **0.9572** |
| Depthwise Separable U-Net | 26.04 | 0.9468 |
| Depthwise + Skip CA | 26.56 | 0.9525 |
| Depthwise + Full CA | 26.93 | 0.9559 |
| Depthwise + Skip CA + FFT | 27.08 | 0.9570 |

---

## RESIDE-6K

| Model | PSNR (dB) | SSIM |
|--------|----------:|------:|
| Baseline U-Net | 28.07 | 0.9621 |
| **Channel Attention U-Net** | **28.54** | **0.9651** |
| Depthwise Separable U-Net | 26.67 | 0.9506 |
| Depthwise + Skip CA | 27.53 | 0.9604 |
| Depthwise + Full CA | 28.18 | 0.9636 |
| Depthwise + Skip CA + FFT | 28.40 | 0.9645 |

---

# Key Findings

- Channel Attention consistently improves U-Net performance on both ITS and RESIDE-6K.
- Training with SSIM Loss consistently outperformed MSE Loss across all evaluated architectures.
- Depthwise Separable Convolutions reduced the parameter count by approximately **80%** while maintaining competitive image quality.
- Adding Channel Attention to lightweight architectures significantly recovered performance lost due to parameter reduction.
- FFT-enhanced skip connections produced the strongest lightweight architecture, achieving **28.40 dB PSNR** and **0.9645 SSIM** on RESIDE-6K.

---

# Comparison with Published Methods

The best-performing models were compared against several published image dehazing methods, including:

- GridDehazeNet
- FFA-Net
- MSBDN
- DehazeFormer
- HAA-Net

The proposed **Channel Attention U-Net** achieved **28.54 dB PSNR** and **0.9651 SSIM** on RESIDE-6K, outperforming several earlier CNN-based methods in SSIM while remaining competitive in reconstruction quality.

---

# Qualitative Results

The repository includes scripts for generating side-by-side visual comparisons:

```
Hazy Image | Dehazed Image | Ground Truth
```

Example outputs are available inside the `outputs/` directory.

<img width="697" height="231" alt="Sample Result" src="https://github.com/user-attachments/assets/d36cf4dd-b1fa-4c6f-b8e9-e34936f741d7" />

**Left:** Hazy Input

**Center:** Model Output

**Right:** Ground Truth

---

# Installation

Clone the repository:

```bash
git clone https://github.com/JaidityaSinha/image-dehazing.git
cd image-dehazing
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

Train a model using:

```bash
python train.py
```

Dataset paths, model selection, and hyperparameters can be configured in `config.py`.

---

# Evaluation

Evaluate a trained model using:

```bash
python eval.py
```

Metrics reported:

- Average PSNR
- Average SSIM

---

# Technologies Used

- Python
- PyTorch
- TorchVision
- OpenCV
- NumPy
- Pillow
- scikit-image

---

# Future Work

- Transformer-based image dehazing architectures
- Real-world haze datasets
- Model quantization and pruning
- Knowledge distillation for lightweight models
- Real-time deployment on edge devices

---

# Acknowledgements

This project was completed during a Summer Internship under the guidance of:

**Dr. Madhuchhanda Dasgupta**  
*Principal Investigator (DST WOS-A)*

---

# License

This project is intended for academic and educational purposes.
