# DCG-Net: Dual-Domain Confidence-Gated Network for Skin Lesion Segmentation

## Overview

DCG-Net is a lightweight encoder-decoder network for skin lesion segmentation that introduces a dual-domain boundary refinement strategy — fusing spatial and frequency-domain edge cues within a confidence-gated framework. The model achieves **1.31M Parameters | 25.16 GFLOPs | 0.0017s Inference Time** while setting state-of-the-art results on PH2 and ISIC-2018 among sub-1.5M-parameter models.

## Architecture

![DCG-Net Architecture](framework.png)

DCG-Net follows the encoder-decoder paradigm with three targeted modules addressing specific failure modes of thin segmentation networks:

```
Input (224×224×3)
    ↓  Stem: Conv3×3 → BN → GELU  [24 ch]
    ↓
    E1: ConvBlock → STAM → DCGEM → MaxPool  [48 ch]
    E2: ConvBlock → STAM → DCGEM → MaxPool  [96 ch]
    E3: ConvBlock → STAM → DCGEM → MaxPool  [144 ch]
    ↓
   AFFM: Learned importance-weighted fusion of [E1, E2, E3] → [48 ch]
    ↓
    D1: ConvBlock → Upsample  [deep supervision head]
    D2: ConvBlock → Full resolution  [main segmentation head]
```

## Key Modules

**STAM — Serial Three-Axis Attention Module**
Refines features along three axes in strict sequence: Channel → Spatial → Cross-Scale → Residual. Ablation confirms this coarse-to-fine order outperforms both parallel and reversed orderings. Addresses feature poverty in narrow channel widths.

**DCGEM — Dual-Domain Confidence-Gated Edge Module**
Combines a spatial edge branch (Conv3×3 → BN → GELU) with a frequency branch (Laplacian-initialised depthwise conv → |·| → BN) to produce a per-pixel confidence map *E* ∈ [0,1]:

$$X_\text{out} = X \odot (1 - E) + X_\text{enhanced} \odot E$$

Flat regions (*E* ≈ 0) pass unchanged; boundary regions (*E* ≈ 1) receive enhanced features. The Laplacian-initialised frequency branch detects edges based on the rate of intensity change rather than absolute colour contrast — directly addressing low-contrast lesion boundaries (amelanotic melanomas, early-stage lesions, darker Fitzpatrick skin types).

**AFFM — Adaptive Feature Fusion Module**
Replaces standard skip connections with softmax-normalised importance weights learned across all encoder levels:

$$X_\text{fused} = \sum_{i=1}^{3} w_i \cdot X'_i, \quad w = \text{Softmax}(\text{MLP}(\text{GAP}(X_{E_1}) \| \text{GAP}(X_{E_2}) \| \text{GAP}(X_{E_3})))$$

Scalar weights (one per level) force competitive allocation: raising one level's importance necessarily lowers the others.

## Loss Function

Training uses a composite loss with three components:

| Component | Formula | Weight |
|-----------|---------|--------|
| Main segmentation | BCE + 0.8·(Dice + IoU) | 1.0 |
| Deep supervision (D1) | BCE + 0.8·Dice | 0.4 |
| Auxiliary edge supervision | Mean BCE across 3 stages | 0.5 |

Edge ground truth is derived via morphological dilation − erosion (kernel size 3) of the binary mask.

## Results

### ISIC-2017 and ISIC-2018

| Model | ISIC-2017 F1 | ISIC-2017 mIoU | ISIC-2018 F1 | ISIC-2018 mIoU | Params |
|-------|-------------|----------------|-------------|----------------|--------|
| U-Net | 0.7802 | 0.7767 | 0.6097 | 0.6251 | 34.53M |
| VM-UNet | **0.8500** | **0.8194** | 0.8659 | 0.8386 | 27.43M |
| EMCADNet-b0 | 0.8168 | 0.8081 | 0.8647 | 0.8314 | 3.92M |
| CMUNeXt | 0.8162 | 0.8034 | 0.8375 | 0.8134 | 3.15M |
| UNeXt | 0.7932 | 0.7860 | 0.8563 | 0.8292 | 1.47M |
| ShuffleNetV2 | 0.8002 | 0.7899 | 0.8504 | 0.8256 | 1.38M |
| LightM-UNet | 0.7415 | 0.7490 | 0.7878 | 0.7628 | 1.15M |
| **DCG-Net (ours)** | 0.8036 | 0.7912 | **0.8688** | **0.8394** | **1.31M** |

### PH2

| Model | RC | PR | F1 | mIoU | Params |
|-------|----|----|----|----|--------|
| VM-UNet | **0.9335** | 0.9233 | 0.9235 | 0.8712 | 27.43M |
| VM-UNet-V2 | 0.9105 | 0.9357 | 0.9182 | **0.8788** | 23.16M |
| UNeXt | 0.8773 | 0.9536 | 0.9079 | 0.8514 | 1.47M |
| ShuffleNetV2 | 0.9020 | 0.9330 | 0.9080 | 0.8543 | 1.38M |
| LightM-UNet | 0.9274 | 0.8825 | 0.9017 | 0.8382 | 1.15M |
| **DCG-Net (ours)** | 0.8582 | **0.9950** | **0.9420** | **0.9123** | **1.31M** |

DCG-Net achieves the best F1 and mIoU on both PH2 and ISIC-2018, outperforming VM-UNet (27.43M) by +1.85 pp F1 and +4.11 pp mIoU on PH2 at 21× fewer parameters.

### Model Complexity

| Model | Params (M) | FLOPs (G) | Time/Img (s) |
|-------|-----------|-----------|-------------|
| U-Net | 34.53 | 50.21 | 0.0080 |
| VM-UNet | 27.43 | 6.26 | 0.0225 |
| EMCADNet-b0 | 3.92 | 1.28 | 0.0158 |
| UNeXt | 1.47 | 0.87 | 0.0053 |
| LightM-UNet | 1.15 | 4.28 | 0.0302 |
| **DCG-Net (ours)** | **1.31** | 25.16 | **0.0017** |

DCG-Net achieves the fastest wall-clock inference time (0.0017 s/image) across all evaluated models — 2.8× faster than HED and 3.1× faster than UNeXt.

## Ablation Study (PH2)

| Variant | STAM | DCGEM | Freq. Branch | AFFM | F1 | mIoU | ΔmIoU |
|---------|------|-------|-------------|------|----|----|-------|
| No AFFM | ✓ | ✓ | ✓ | ✗ | 0.9328 | 0.8377 | +6.23 |
| Spatial-only (no freq.) | ✓ | ✓ | ✗ | ✓ | 0.9032 | 0.8437 | +5.63 |
| No DCGEM | ✓ | ✗ | — | ✓ | 0.8905 | 0.8839 | +1.61 |
| No STAM | ✗ | ✓ | ✓ | ✓ | 0.8928 | 0.8852 | +1.48 |
| **DCG-Net (full)** | ✓ | ✓ | ✓ | ✓ | **0.9182** | **0.9000** | — |

AFFM and the frequency branch of DCGEM together account for +11.86 pp mIoU degradation when removed — the two primary drivers of DCG-Net's accuracy gains.

## Requirements

- Python ≥ 3.7.5
- PyTorch ≥ 2.2.0
- CUDA-compatible GPU (8 GB VRAM recommended)
- OpenCV, NumPy, SciPy

Install dependencies:

```bash
pip install torch torchvision opencv-python numpy scipy matplotlib
```

## Training

```bash
python main.py \
  --mode train \
  --dataset ISIC2018 \
  --models DCGNet \
  --image_size 224 \
  --batch_size 2 \
  --num_epochs 100 \
  --lr 1e-4 \
  --weight_decay 2e-4
```

## Testing

```bash
python main.py \
  --mode test \
  --dataset ISIC2018 \
  --models DCGNet \
  --image_size 224
```

## Datasets

Three publicly available dermoscopic benchmarks are used:

| Dataset | Training | Validation | Test |
|---------|----------|-----------|------|
| ISIC-2017 | 1,250 | 150 | 600 |
| ISIC-2018 | 2,076 | 259 | 259 |
| PH2 | 142 | 18 | 40 |

All images are resized to 224×224. Ground-truth masks are binarised at intensity 128.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{alharith2025dcgnet,
  title   = {{DCG-Net}: Resource-Constrained Skin Lesion Segmentation via
             Dual-Domain Confidence-Gated Boundary Refinement},
  author  = {Alharith, Razan},
  year    = {2025},
  url     = {https://github.com/razanharith/DCG-Net}
}
```

## Contact

For questions about this research, contact Razan Alharith at razanalharith@my.swjtu.edu.cn.
