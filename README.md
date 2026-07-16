![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Image%20Dehazing-green)

<img width="696" height="233" alt="Sample — Hazy | Dehazed | Ground Truth" src="https://github.com/user-attachments/assets/a80d6730-db74-424d-8b6f-238efeb1dc03" />

Sample — Hazy | Dehazed | Ground Truth

# Image Dehazing using Deep Learning

A PyTorch implementation and comparative study of **UNet-based architectures** for single image dehazing on the **ITS** and **RESIDE-6K** datasets.

This project was developed as part of a Summer Internship in Deep Learning and Computer Vision.

---

## Overview

Atmospheric haze significantly degrades image quality by reducing contrast and obscuring scene details. This project investigates UNet-based architectures for restoring haze-free images from a single hazy input image.

In addition to the Baseline UNet and Channel Attention UNet, this work includes a lightweight UNet variant that incorporates **Depthwise Separable Convolutions**  to improve feature representation while reducing computational complexity.

Performance is evaluated using:

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)

---

## Highlights

- Implemented Baseline UNet, Channel Attention UNet and Depthwise Separable UNet with Channel Attention in PyTorch.
- Trained and evaluated all models on the ITS and RESIDE-6K datasets.
- Channel Attention Unet achieved **28.54 dB PSNR** and **0.9651 SSIM** on RESIDE-6K, improving baseline performance by **+1.43 dB PSNR** and **+0.0162 SSIM**.
- Depthwise Separable UNet with channel attention achieved **28.18 dB PSNR** and **0.9636 SSIM** on RESIDE-6K, improving baseline performance by **+1.07 dB PSNR**   and **+0.0147 SSIM**.

---

## Features

- Custom PyTorch dataset pipeline
- Baseline UNet implementation
- Attention Gate UNet
- Channel Attention UNet
- Training and evaluation scripts
- Automatic checkpoint saving
- PSNR and SSIM evaluation
- Visual comparison generation

---

## Project Structure

```
image-dehazing/
│
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   ├── unet.py
│   ├── unet_attention.py
│   └── unet_channel_attention.py
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

## Datasets

> **Note:** The datasets are not included in this repository due to their large size.

Download the RESIDE datasets from the official source and organize them as:
```
data/
├── train/
├── val/
└── test/
```

### ITS (Indoor Training Set)

- Indoor synthetic haze images
- Paired hazy and clear images
- Used for indoor experiments

### RESIDE-6K

- Outdoor synthetic haze images
- Greater scene diversity
- Used for outdoor experiments

---

## Pre-trained Models

Pre-trained model weights are not included in this repository.

Train the models using:

python train.py

or place the downloaded checkpoints inside the `outputs/` directory.

## Training Configuration

| Parameter | Value |
|----------|------|
| Framework | PyTorch |
| Loss Function | Mean Squared Error (MSE) |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 8 |
| Image Size | 256 × 256 |
| Scheduler | ReduceLROnPlateau |
| Output Activation | Sigmoid |

---

## Results

### ITS Dataset

| Model | Epochs | PSNR (dB) | SSIM |
|------|------:|------:|------:|
| Baseline UNet | 30 | 26.42 | 0.9495 |
| Channel Attention UNet | 35 | **26.57** | 0.9543 |

---

### RESIDE-6K Dataset

| Model | Epochs | PSNR (dB) | SSIM |
|------|------:|------:|------:|
| Baseline UNet | 50 | 27.11 | 0.9489 |
| Channel Attention UNet | 60 | **28.23** | **0.9563** |

---

## Qualitative Results

The repository also contains scripts for generating side-by-side comparisons:

```
Hazy Image | Dehazed Image | Ground Truth
```

These comparisons are saved under the `outputs/` directory.

---

## Best Performance

| Dataset | Best Model | PSNR | SSIM |
|----------|------------|------|------|
| ITS | Channel Attention UNet | 26.57 dB | 0.9473 |
| RESIDE-6K | Channel Attention UNet | 28.23 dB | 0.9563 |

---

## Sample Result

Below is a qualitative comparison from the test set.

<img width="697" height="231" alt="image" src="https://github.com/user-attachments/assets/d36cf4dd-b1fa-4c6f-b8e9-e34936f741d7" />


**Left:** Hazy Input

**Center:** Model Output

**Right:** Ground Truth

---

## Installation

Clone the repository:

```bash
git clone https://github.com/JaidityaSinha/image-dehazing.git
cd image-dehazing
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Train the desired model:

```bash
python train.py
```

Model paths and dataset directories can be configured in `config.py`.

---

## Evaluation

Evaluate the trained model:

```bash
python eval.py
```

Evaluation metrics:

- Average PSNR
- Average SSIM

---

## Technologies Used

- Python
- PyTorch
- TorchVision
- NumPy
- Pillow
- OpenCV
- scikit-image

---

## Future Work

- Validation-based checkpointing
- Frequency-aware attention mechanisms
- Real-world haze datasets
- Lightweight models for real-time inference

---

## Acknowledgements

This project was completed during the Summer Internship under the guidance of:

**Dr. Madhuchhanda Dasgupta**  
Principal Investigator (DST WOS-A)

---

## License

This project is intended for academic and educational purposes.
