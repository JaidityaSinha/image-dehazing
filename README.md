![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Image%20Dehazing-green)

<img width="642" height="318" alt="image" src="https://github.com/user-attachments/assets/3b4fcd37-636b-4a67-bdb6-88179d5134c5" />

*Real-world sample: Hazy Input → Dehazed Output*

# Deep Learning-based Single Image Dehazing
### A Comparative Study of U-Net Architectures

This repository presents a comparative study of **six U-Net-based deep learning architectures** for **single image dehazing**, developed using **PyTorch**.

The project investigates how architectural modifications such as **Channel Attention**, **Depthwise Separable Convolutions**, **FFT-based Frequency Enhancement**, and **SSIM-based optimization** affect restoration quality, parameter efficiency, and computational complexity.

Developed during the **Summer Internship in Deep Learning and Computer Vision** under the guidance of **Dr. Madhuchhanda Dasgupta**.

---

# Overview

Atmospheric haze degrades image quality by scattering light, reducing contrast, washing out colors, and obscuring fine details. These degradations negatively impact both human perception and downstream computer vision applications such as autonomous driving, surveillance, robotics, and remote sensing.

This project explores multiple U-Net variants that progressively improve reconstruction quality while reducing computational cost.

The models were trained and evaluated on the **ITS** and **RESIDE-6K** benchmark datasets using both **Mean Squared Error (MSE)** and **Structural Similarity (SSIM)** loss functions.

---

# Highlights

-  Six U-Net architectures implemented from scratch in PyTorch
-  Comparative evaluation across indoor and outdoor datasets
-  Channel Attention using Squeeze-and-Excitation blocks
-  Lightweight Depthwise Separable U-Net variants
-  FFT-enhanced skip connections for frequency-domain refinement
-  Comparison of MSE Loss and SSIM Loss
-  Parameter efficiency analysis
-  Ablation study on attention placement
-  Benchmark comparison with published image dehazing methods

---

# Implemented Architectures

| Model | Description |
|------|-------------|
| Baseline U-Net | Standard encoder-decoder architecture |
| Channel Attention U-Net | U-Net with SE-based Channel Attention |
| Depthwise Separable U-Net | Lightweight architecture replacing standard convolutions |
| Depthwise + Skip Channel Attention | Channel Attention applied on skip features |
| Depthwise + Full Channel Attention | Attention applied throughout the network |
| Depthwise + Skip Channel Attention + FFT | Frequency-enhanced lightweight architecture |

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

The datasets are **not included** due to their size.

Training and evaluation were performed on the following public datasets.

## ITS (Indoor Training Set)

- Synthetic indoor hazy images
- Paired hazy/clear image dataset
- Evaluated on SOTS-Indoor

## RESIDE-6K

- Outdoor synthetic haze benchmark
- Large-scale paired dataset
- Separate testing split for evaluation

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
| Loss Functions | MSE Loss, SSIM Loss |

---

# Model Complexity

| Model | Parameters |
|--------|-----------:|
| Baseline U-Net | 31,031,875 |
| Channel Attention U-Net | 31,249,987 |
| Depthwise Separable U-Net | 5,988,382 |
| Depthwise + Skip Channel Attention | 6,031,902 |
| Depthwise + Full Channel Attention | 6,206,494 |
| Depthwise + Skip CA + FFT | 8,821,022 |

---

# Quantitative Results (SSIM Loss)

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

- **Channel Attention** consistently improved image restoration across both datasets.
- **SSIM Loss** produced superior perceptual quality compared to MSE Loss.
- **Depthwise Separable Convolutions** reduced trainable parameters by approximately **81%** while maintaining competitive performance.
- Incorporating **Channel Attention** significantly recovered the performance loss introduced by lightweight convolutions.
- The **Depthwise + Skip Channel Attention + FFT** architecture achieved the strongest lightweight performance with **28.40 dB PSNR** and **0.9645 SSIM** on RESIDE-6K.

---

# Qualitative Results

The repository includes scripts for generating side-by-side visual comparisons.

```
Hazy Input | Model Output | Ground Truth
```

<img width="697" height="231" alt="Sample Result" src="https://github.com/user-attachments/assets/d36cf4dd-b1fa-4c6f-b8e9-e34936f741d7" />

**Left:** Hazy Image

**Center:** Dehazed Output

**Right:** Ground Truth

---

# Installation

Clone the repository.

```bash
git clone https://github.com/JaidityaSinha/image-dehazing.git

cd image-dehazing
```

Install the required packages.

```bash
pip install -r requirements.txt
```

---

# Training

Train a selected architecture using

```bash
python train.py
```

Dataset paths, model selection, and training parameters can be configured through `config.py`.

---

# Evaluation

Evaluate a trained checkpoint using

```bash
python eval.py
```

The evaluation reports:

- Average PSNR
- Average SSIM

---

# Technologies

- Python
- PyTorch
- TorchVision
- OpenCV
- NumPy
- Pillow
- scikit-image

---

# Future Work

- Vision Transformer-based dehazing
- Physics-informed image restoration
- Real-world haze benchmarks
- Model pruning and quantization
- Knowledge distillation
- ONNX/TensorRT deployment
- Real-time edge inference

---

# Acknowledgements

This work was completed during the **Summer Internship in Deep Learning and Computer Vision**.

**Supervisor**

**Dr. Madhuchhanda Dasgupta**

Principal Investigator (DST WOS-A)

IDEAS – Institute of Data Engineering, Analytics and Science Foundation

---

# License

This repository is released for **academic and research purposes**.

If you use this work in your research, please consider citing the accompanying internship report.
