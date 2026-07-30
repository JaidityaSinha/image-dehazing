![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Image%20Dehazing-green)

<img width="642" height="318" alt="image" src="https://github.com/user-attachments/assets/3b4fcd37-636b-4a67-bdb6-88179d5134c5" />

*Real-world sample: Hazy Input → Dehazed Output*

# Deep Learning-based Single Image Dehazing
### A Comparative Study of U-Net Architectures

This repository presents a comparative study of **six U-Net-based deep learning architectures** for **single image dehazing**, implemented using **PyTorch**.

The project investigates how architectural enhancements including **Channel Attention**, **Depthwise Separable Convolutions**, **FFT-based Frequency Enhancement**, and **SSIM-based optimization** influence restoration quality, computational efficiency, and model complexity.

The work was completed as part of a **Summer Internship in Deep Learning and Computer Vision** under the guidance of **Dr. Madhuchhanda Dasgupta**.

---

# Overview

Atmospheric haze significantly degrades image quality by scattering light, reducing visibility, lowering contrast, and distorting scene colors. These degradations adversely affect both human perception and numerous computer vision applications, including autonomous driving, surveillance, robotics, and remote sensing.

This project performs a comprehensive comparison of multiple U-Net variants ranging from the baseline architecture to lightweight attention-enhanced models. Each model is evaluated on the **ITS** and **RESIDE-6K** datasets using both **Mean Squared Error (MSE)** and **Structural Similarity (SSIM)** loss functions.

---

# Features

- Six U-Net architectures implemented entirely in PyTorch
- Comparative evaluation on indoor and outdoor dehazing benchmarks
- Channel Attention using Squeeze-and-Excitation (SE) blocks
- Lightweight architectures using Depthwise Separable Convolutions
- FFT-enhanced skip connections for frequency-domain refinement
- Performance comparison using MSE Loss and SSIM Loss
- Parameter efficiency and complexity analysis
- Ablation study of architectural improvements
- Comparison with published image dehazing methods

---

# Implemented Models

| Model | Description |
|--------|-------------|
| Baseline U-Net | Standard encoder-decoder U-Net |
| Channel Attention U-Net | U-Net with SE-based Channel Attention |
| Depthwise Separable U-Net | Lightweight U-Net using depthwise separable convolutions |
| Depthwise + Skip Channel Attention | Lightweight model with attention applied to skip connections |
| Depthwise + Full Channel Attention | Lightweight model with attention throughout the network |
| Depthwise + Skip Channel Attention + FFT | Lightweight architecture with frequency-domain enhancement |

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
├── weights/
│   └── README.md
│
├── config.py
├── dataset.py
├── train.py
├── eval.py
├── test.py
├── requirements.txt
└── README.md
```

---

# Datasets

The datasets are not included in this repository due to their size.

Training and evaluation were conducted using the following publicly available datasets.

## ITS (Indoor Training Set)

- Synthetic indoor hazy image dataset
- Paired hazy and ground-truth images
- Evaluated using the SOTS-Indoor benchmark

## RESIDE-6K

- Large-scale outdoor image dehazing dataset
- Paired hazy and clear image dataset
- Separate testing split for evaluation

---

# Training Configuration

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Optimizer | Adam |
| Learning Rate | 1 × 10⁻⁴ |
| Batch Size | 8 |
| Image Resolution | 256 × 256 |
| Scheduler | ReduceLROnPlateau |
| Output Activation | Sigmoid |
| Loss Functions | MSE Loss, SSIM Loss |

---

# Model Complexity

| Model | Parameters |
|--------|-----------:|
| Baseline U-Net | 31.03 M |
| Channel Attention U-Net | 31.25 M |
| Depthwise Separable U-Net | 5.99 M |
| Depthwise + Skip Channel Attention | 6.03 M |
| Depthwise + Full Channel Attention | 6.21 M |
| Depthwise + Skip Channel Attention + FFT | 8.82 M |

---

# Quantitative Results (SSIM Loss)

## ITS Dataset

| Model | PSNR (dB) | SSIM |
|--------|----------:|------:|
| Baseline U-Net | 26.42 | 0.9532 |
| **Channel Attention U-Net** | **27.17** | **0.9572** |
| Depthwise Separable U-Net | 26.04 | 0.9468 |
| Depthwise + Skip Channel Attention | 26.56 | 0.9525 |
| Depthwise + Full Channel Attention | 26.93 | 0.9559 |
| Depthwise + Skip Channel Attention + FFT | 27.08 | 0.9570 |

## RESIDE-6K Dataset

| Model | PSNR (dB) | SSIM |
|--------|----------:|------:|
| Baseline U-Net | 28.07 | 0.9621 |
| **Channel Attention U-Net** | **28.54** | **0.9651** |
| Depthwise Separable U-Net | 26.67 | 0.9506 |
| Depthwise + Skip Channel Attention | 27.53 | 0.9604 |
| Depthwise + Full Channel Attention | 28.18 | 0.9636 |
| Depthwise + Skip Channel Attention + FFT | 28.40 | 0.9645 |

---

# Key Findings

- Channel Attention consistently improved reconstruction quality over the baseline architecture.
- SSIM Loss produced better perceptual image quality than MSE Loss across all evaluated models.
- Depthwise Separable Convolutions reduced the number of trainable parameters by approximately **81%** while maintaining competitive performance.
- Applying Channel Attention effectively compensated for the performance loss introduced by lightweight convolutions.
- The **Depthwise + Skip Channel Attention + FFT** architecture achieved the strongest lightweight performance with **28.40 dB PSNR** and **0.9645 SSIM** on RESIDE-6K.

---

# Pre-trained Weights

The repository includes the best-performing **Channel Attention U-Net** checkpoints trained using **SSIM Loss**.

These checkpoints are available from the **GitHub Releases** section of this repository.

| Checkpoint | Dataset | Loss |
|------------|---------|------|
| `channel_attention_unet_its_ssim.pth` | ITS | SSIM |
| `channel_attention_unet_reside6k_ssim.pth` | RESIDE-6K | SSIM |

After downloading, place the files inside the `weights/` directory.

```text
weights/
├── channel_attention_unet_its_ssim.pth
└── channel_attention_unet_reside6k_ssim.pth
```

These pretrained models can be evaluated directly using `eval.py` or used for inference with `test.py`.

---

# Qualitative Results

The repository includes scripts for generating qualitative comparisons between hazy inputs, dehazed outputs, and corresponding ground-truth images.

```
Hazy Input | Dehazed Output | Ground Truth
```

<img width="697" height="231" alt="Sample Result" src="https://github.com/user-attachments/assets/d36cf4dd-b1fa-4c6f-b8e9-e34936f741d7" />

---

# Installation

Clone the repository.

```bash
git clone https://github.com/JaidityaSinha/image-dehazing.git
cd image-dehazing
```

Install the required dependencies.

```bash
pip install -r requirements.txt
```

---

# Training

Configure the dataset paths and model settings in `config.py`, then train using

```bash
python train.py
```

---

# Evaluation

Evaluate a pretrained model using

```bash
python eval.py
```

The evaluation reports:

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

- Vision Transformer-based image dehazing
- Physics-informed restoration models
- Real-world haze benchmark evaluation
- Model pruning and quantization
- Knowledge distillation
- ONNX and TensorRT deployment
- Real-time edge-device inference

---

# Acknowledgements

This project was completed during the **Summer Internship in Deep Learning and Computer Vision**.

**Supervisor**

**Dr. Madhuchhanda Dasgupta**

Principal Investigator (DST WOS-A)

IDEAS – Institute of Data Engineering, Analytics and Science Foundation

---

# License

This project is released for **academic and research purposes**.

If you find this repository useful in your research or projects, please consider citing the accompanying internship report.
