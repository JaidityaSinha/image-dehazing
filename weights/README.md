# Pre-trained Weights

This directory stores the pre-trained model checkpoints used for evaluation and inference.

The best-performing models are distributed through the **GitHub Releases** page of this repository.

## Available Checkpoints

| File | Dataset | Loss Function |
|------|---------|---------------|
| `channel_attention_unet_its_ssim_best.pth` | ITS | SSIM Loss |
| `channel_attention_unet_reside6k_ssim_best.pth` | RESIDE-6K | SSIM Loss |

## Usage

1. Download the desired checkpoint(s) from the latest GitHub Release.
2. Place the downloaded `.pth` files inside this directory.

Expected structure:

```text
weights/
├── channel_attention_unet_its_ssim_best.pth
├── channel_attention_unet_reside6k_ssim_best.pth
└── README.md
```