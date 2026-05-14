"""
DCG-Net: Resource-Constrained Skin Lesion Segmentation via
Dual-Domain Confidence-Gated Boundary Refinement

Architecture (1.31M parameters):
  - Stem: 3 → 24 channels (3×3 Conv → BN → GELU)
  - Encoder: 3 stages (48 → 96 → 144)
      Each stage: ConvBlock → STAM → DCGEM → MaxPool
  - AFFM: learned softmax-normalised importance weights for multi-level fusion
  - Decoder: 2 stages with bilinear upsampling + deep supervision at D1
  - Auxiliary edge supervision at each encoder stage

Key modules:
  STAM  — Serial Three-Axis Attention (Channel → Spatial → Cross-Scale → Residual)
  DCGEM — Dual-Domain Confidence-Gated Edge Module
           (spatial branch + Laplacian-initialised frequency branch → confidence map E)
           X_out = X ⊙ (1 − E) + X_enhanced ⊙ E
  AFFM  — Adaptive Feature Fusion Module
           (GAP → concat → MLP → softmax weights → weighted sum of unified features)

Reference: Chapter 5, "Resource-Constrained Skin Lesion Segmentation via
Dual-Domain Confidence-Gated Boundary Refinement"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════
# BUILDING BLOCKS
# ═══════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Double 3×3 conv with BN, GELU, and learnable residual shortcut."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.shortcut = (
            nn.Identity() if in_ch == out_ch
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


# ═══════════════════════════════════════════════════════
# STAM — Serial Three-Axis Attention Module
# Processing order: CA → SA → CSA → + residual
# ═══════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    """GAP + GMP → shared MLP → sigmoid weights."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        mx  = self.mlp(F.adaptive_max_pool2d(x, 1))
        return self.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    """Channel mean+max → 7×7 conv → sigmoid spatial map."""
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True)[0]
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CrossScaleAttention(nn.Module):
    """Multi-resolution max pooling (3×3, 5×5) → concat → 1×1 fusion.

    Captures multi-scale context that narrow channel widths cannot represent
    with standard convolutions alone.  Operates on features already refined
    by CA and SA (strict serial dependency).
    """
    def __init__(self, channels):
        super().__init__()
        self.pool3 = nn.MaxPool2d(3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(5, stride=1, padding=2)
        self.fuse  = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.fuse(torch.cat([self.pool3(x), self.pool5(x), x], dim=1))


class STAM(nn.Module):
    """Serial Three-Axis Attention Module.

    Processing flow (Eq. 1 in thesis):
        X → CA(X) → SA(X_ca) → CSA(X_sa) → X_enhanced + X

    The strict serial order is deliberate:
      - SA operates on channel-refined features
      - CSA operates on features refined by both CA and SA
    This coarse-to-fine ordering outperforms parallel and reversed orderings
    (confirmed by ablation study, Table 5.4).
    """
    def __init__(self, channels):
        super().__init__()
        self.ca  = ChannelAttention(channels)
        self.sa  = SpatialAttention()
        self.csa = CrossScaleAttention(channels)

    def forward(self, x):
        identity = x
        x = x * self.ca(x)       # Channel refinement
        x = x * self.sa(x)       # Spatial refinement (on channel-refined)
        x = self.csa(x)          # Cross-scale refinement (on CA+SA refined)
        return x + identity       # Residual connection


# ═══════════════════════════════════════════════════════
# DCGEM — Dual-Domain Confidence-Gated Edge Module
# ═══════════════════════════════════════════════════════

class FrequencyEdgeBranch(nn.Module):
    """Learnable frequency-domain edge extractor.

    Initialised with the discrete Laplacian kernel:
        [[0, -1,  0],
         [-1,  4, -1],
         [0, -1,  0]]

    The Laplacian initialisation biases the branch toward high-frequency
    content; weights are updated by backprop so the network learns a
    task-specific frequency response.  Absolute value extracts edge
    magnitude regardless of gradient direction.

    This directly addresses the low-contrast failure mode: the frequency
    branch detects boundaries based on the rate of intensity change,
    not the absolute colour difference — providing signal exactly where
    the spatial branch fails.
    """
    def __init__(self, channels):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            channels, channels, 3, padding=1, bias=False, groups=channels
        )
        nn.init.zeros_(self.dw_conv.weight)
        lap = torch.tensor(
            [[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]], dtype=torch.float32
        )
        for i in range(channels):
            self.dw_conv.weight.data[i, 0] = lap
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(torch.abs(self.dw_conv(x)))


class DCGEM(nn.Module):
    """Dual-Domain Confidence-Gated Edge Module.

    Two parallel branches operate on input X ∈ ℝ^{H×W×C}:

    Edge Detector (spatial + frequency → confidence map E ∈ [0,1]):
      - Spatial branch:  Conv3×3 → BN → GELU
      - Frequency branch: Laplacian-init depthwise conv → |·| → BN
      - Fusion: Concat → 1×1 conv → BN → GELU → 3×3 conv → sigmoid

    Feature Enhancer:
      - Conv3×3 → BN → GELU  (controls WHAT the refined features contain)

    Confidence-gated adaptive fusion (Eq. 10):
      X_out = X ⊙ (1 − E) + X_enhanced ⊙ E

    Flat regions (E ≈ 0): original features pass through unchanged.
    Boundary regions (E ≈ 1): enhanced features dominate.

    The edge map E is returned as an auxiliary output for direct
    BCE supervision against morphological boundary ground truth.
    """
    def __init__(self, channels):
        super().__init__()
        # Spatial edge branch
        self.spatial_edge = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        # Frequency edge branch (Laplacian-initialised depthwise conv)
        self.freq_edge = FrequencyEdgeBranch(channels)
        # Fuse spatial + frequency → single-channel confidence map
        self.edge_fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, 1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.GELU(),
            nn.Conv2d(channels // 2, 1, 3, padding=1, bias=False),
            nn.Sigmoid(),
        )
        # Feature enhancer
        self.feature_enhancer = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        # Dual-domain edge detection
        e_spatial = self.spatial_edge(x)
        e_freq    = self.freq_edge(x)
        E         = self.edge_fuse(torch.cat([e_spatial, e_freq], dim=1))  # [B,1,H,W]

        # Feature enhancement
        x_enhanced = self.feature_enhancer(x)

        # Confidence-gated adaptive fusion
        x_out = x * (1 - E) + x_enhanced * E

        return x_out, E  # E returned for auxiliary edge supervision


# ═══════════════════════════════════════════════════════
# AFFM — Adaptive Feature Fusion Module
# ═══════════════════════════════════════════════════════

class AFFM(nn.Module):
    """Adaptive Feature Fusion Module.

    Replaces standard skip connections with learned importance-weighted
    aggregation across all encoder levels.  Three stages:

    Step 1 — Unification: resize + 1×1 project all encoder features
              to a common (H', W') and channel dimension C'.
    Step 2 — Weight generation: GAP on original features → concat →
              two-layer MLP → softmax (Σ w_i = 1).
    Step 3 — Adaptive fusion: X_fused = Σ w_i · X'_i.

    Softmax normalisation creates a competitive regime: raising one
    level's importance necessarily lowers the others.  Scalar weights
    (one per level, not per-channel) are a deliberate efficiency choice
    that also reduces overfitting risk on small datasets like PH2.
    """
    def __init__(self, channels_list, out_channels, target_hw):
        """
        Args:
            channels_list: [C1, C2, C3] channel dims of encoder stages
            out_channels:  common channel dimension after projection
            target_hw:     (H', W') common spatial resolution (E1 size)
        """
        super().__init__()
        self.target_hw = target_hw
        self.n_levels  = len(channels_list)
        total_ch       = sum(channels_list)  # 48+96+144 = 288

        # 1×1 projections to common dimension
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            for c in channels_list
        ])

        # Importance weighting MLP (288 → 72 → 3)
        r      = 4
        hidden = max(total_ch // r, 16)
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.weight_mlp = nn.Sequential(
            nn.Linear(total_ch, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.n_levels),
        )

    def forward(self, features):
        """
        Args:
            features: [X_E1, X_E2, X_E3] encoder feature maps
        Returns:
            X_fused: [B, out_channels, H', W']
        """
        # Step 1: Unify resolution and channel dimension
        unified = []
        for feat, proj in zip(features, self.projections):
            feat_up = F.interpolate(
                feat, size=self.target_hw, mode='bilinear', align_corners=True
            )
            unified.append(proj(feat_up))

        # Step 2: Compute softmax importance weights from original features
        gaps    = [self.gap(f).flatten(1) for f in features]
        v_cat   = torch.cat(gaps, dim=1)                         # [B, 288]
        weights = torch.softmax(self.weight_mlp(v_cat), dim=1)   # [B, 3]

        # Step 3: Weighted sum
        fused = torch.zeros_like(unified[0])
        for i in range(self.n_levels):
            fused = fused + weights[:, i].view(-1, 1, 1, 1) * unified[i]

        return fused


# ═══════════════════════════════════════════════════════
# DCGNet — Full Architecture
# ═══════════════════════════════════════════════════════

class DCGNet(nn.Module):
    """
    DCG-Net: Dual-domain Confidence-Gated Network for skin lesion segmentation.

    Encoder-decoder with targeted efficiency design:
      - STAM addresses feature poverty in narrow channel widths
      - DCGEM addresses boundary erosion through successive downsampling
      - AFFM addresses undifferentiated multi-level feature fusion
      - Auxiliary edge supervision addresses weak boundary gradients during training

    Architecture:
      Stem:    3 → C0=24   (Conv3×3 → BN → GELU)
      E1:      C0 → C1=48  (ConvBlock → STAM → DCGEM → MaxPool)
      E2:      C1 → C2=96  (ConvBlock → STAM → DCGEM → MaxPool)
      E3:      C2 → C3=144 (ConvBlock → STAM → DCGEM → MaxPool)
      AFFM:   [E1, E2, E3] → C1=48 at E1 spatial resolution
      D1:      C1 → C1     (ConvBlock → bilinear upsample)  [deep supervision head]
      D2:      C1 → C0     (ConvBlock → full-resolution output)
      seg_head: C0 → 1     (1×1 conv, final prediction)

    Training outputs: (final, deep_out, edge1, edge2, edge3)
    Inference output: final
    """

    def __init__(self, in_channels=3, num_classes=1, input_size=224):
        super().__init__()

        C0, C1, C2, C3 = 24, 48, 96, 144
        self.input_size = input_size

        # ── Stem ──
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C0, 3, padding=1, bias=False),
            nn.BatchNorm2d(C0),
            nn.GELU(),
        )

        # ── Encoder Stage 1 (C0 → C1=48) ──
        self.enc1_conv  = ConvBlock(C0, C1)
        self.enc1_stam  = STAM(C1)
        self.enc1_dcgem = DCGEM(C1)
        self.pool1      = nn.MaxPool2d(2)

        # ── Encoder Stage 2 (C1 → C2=96) ──
        self.enc2_conv  = ConvBlock(C1, C2)
        self.enc2_stam  = STAM(C2)
        self.enc2_dcgem = DCGEM(C2)
        self.pool2      = nn.MaxPool2d(2)

        # ── Encoder Stage 3 (C2 → C3=144, deliberate 1.5× expansion) ──
        self.enc3_conv  = ConvBlock(C2, C3)
        self.enc3_stam  = STAM(C3)
        self.enc3_dcgem = DCGEM(C3)
        self.pool3      = nn.MaxPool2d(2)

        # ── AFFM: fuse E1(48) + E2(96) + E3(144) → 48 at E1 resolution ──
        target_hw  = (input_size // 2, input_size // 2)
        self.affm  = AFFM(
            channels_list=[C1, C2, C3],
            out_channels=C1,
            target_hw=target_hw,
        )

        # ── Decoder Stage 1 (D1): AFFM output → C1 ──
        self.dec1_conv = ConvBlock(C1, C1)
        self.dec1_up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # ── Decoder Stage 2 (D2): C1 → C0 ──
        self.dec2_conv = ConvBlock(C1, C0)

        # ── Output heads ──
        self.seg_head  = nn.Conv2d(C0, num_classes, 1)   # main prediction
        self.deep_head = nn.Conv2d(C1, num_classes, 1)   # deep supervision (D1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Training returns: (final, deep_out, edge1, edge2, edge3)
        Inference returns: final
        """
        input_size = x.shape[2:]

        # Stem
        x0 = self.stem(x)                         # [B, 24, H, W]

        # Encoder Stage 1
        e1             = self.enc1_conv(x0)        # [B, 48, H, W]
        e1             = self.enc1_stam(e1)
        e1, edge1      = self.enc1_dcgem(e1)       # edge1: [B,1,H,W]
        e1_pool        = self.pool1(e1)            # [B, 48, H/2, W/2]

        # Encoder Stage 2
        e2             = self.enc2_conv(e1_pool)   # [B, 96, H/2, W/2]
        e2             = self.enc2_stam(e2)
        e2, edge2      = self.enc2_dcgem(e2)
        e2_pool        = self.pool2(e2)            # [B, 96, H/4, W/4]

        # Encoder Stage 3
        e3             = self.enc3_conv(e2_pool)   # [B,144, H/4, W/4]
        e3             = self.enc3_stam(e3)
        e3, edge3      = self.enc3_dcgem(e3)
        # e3_pool not used — AFFM fuses pre-pool encoder features directly

        # AFFM: adaptive importance-weighted fusion
        fused = self.affm([e1, e2, e3])            # [B, 48, H/2, W/2]

        # Decoder Stage 1 (D1)
        d1       = self.dec1_conv(fused)           # [B, 48, H/2, W/2]
        deep_out = self.deep_head(d1)              # deep supervision logits
        d1_up    = self.dec1_up(d1)               # [B, 48, H, W]

        # Decoder Stage 2 (D2) — resize to input spatial dims
        d1_up = F.interpolate(d1_up, size=input_size, mode='bilinear', align_corners=True)
        d2    = self.dec2_conv(d1_up)             # [B, 24, H_in, W_in]

        # Final segmentation head
        final    = self.seg_head(d2)              # [B, 1, H_in, W_in]
        deep_out = F.interpolate(deep_out, size=input_size, mode='bilinear', align_corners=True)

        if self.training:
            return final, deep_out, edge1, edge2, edge3
        else:
            return final


# ═══════════════════════════════════════════════════════
# DCGNetLoss — Composite Training Loss
# ═══════════════════════════════════════════════════════

class DCGNetLoss(nn.Module):
    """Composite loss for DCG-Net (Eq. 12 in thesis).

    Components:
      L_seg  = BCE + α(Dice + IoU)           on final output, α=0.8
      L_deep = BCE + α·Dice                  on D1 auxiliary prediction
      L_edge = mean BCE across 3 encoder stages vs. morphological edge GT

    Total: L_total = L_seg + λ_deep·L_deep + λ_edge·L_edge
           λ_deep=0.4, λ_edge=0.5

    Edge GT derived via dilation−erosion (morphological ring, k=3).
    """

    def __init__(self, alpha=0.8, lambda_deep=0.4, lambda_edge=0.5):
        super().__init__()
        self.alpha       = alpha
        self.lambda_deep = lambda_deep
        self.lambda_edge = lambda_edge

    @staticmethod
    def _bce(pred, target):
        return F.binary_cross_entropy_with_logits(pred, target)

    @staticmethod
    def _dice(pred, target, eps=1e-6):
        p = torch.sigmoid(pred)
        inter = (p * target).sum(dim=(2, 3))
        union = (p * p).sum(dim=(2, 3)) + (target * target).sum(dim=(2, 3))
        return 1.0 - ((2 * inter + eps) / (union + eps)).mean()

    @staticmethod
    def _iou(pred, target, eps=1e-6):
        p     = torch.sigmoid(pred)
        inter = (p * target).sum(dim=(2, 3))
        union = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter
        return 1.0 - ((inter + eps) / (union + eps)).mean()

    @staticmethod
    def _edge_gt(mask, k=3):
        """Morphological boundary ring: dilate(mask) − erode(mask)."""
        pad     = k // 2
        dilated = F.max_pool2d(mask, k, stride=1, padding=pad)
        eroded  = -F.max_pool2d(-mask, k, stride=1, padding=pad)
        return (dilated - eroded).clamp(0, 1)

    def forward(self, outputs, target):
        """
        Args:
            outputs: (final, deep_out, edge1, edge2, edge3) from DCGNet
            target:  [B, 1, H, W] binary ground truth mask
        Returns:
            total loss (scalar), dict of component losses
        """
        final, deep_out, edge1, edge2, edge3 = outputs

        # Main segmentation loss
        loss_seg = (
            self._bce(final, target)
            + self.alpha * (self._dice(final, target) + self._iou(final, target))
        )

        # Deep supervision loss (D1)
        loss_deep = (
            self._bce(deep_out, target)
            + self.alpha * self._dice(deep_out, target)
        )

        # Auxiliary edge supervision (mean over 3 encoder stages)
        loss_edge = 0.0
        for E in [edge1, edge2, edge3]:
            tgt_r    = F.interpolate(target, size=E.shape[2:], mode='nearest')
            edge_gt  = self._edge_gt(tgt_r)
            loss_edge = loss_edge + F.binary_cross_entropy(E, edge_gt)
        loss_edge = loss_edge / 3

        total = (
            loss_seg
            + self.lambda_deep * loss_deep
            + self.lambda_edge * loss_edge
        )

        return total, {
            'seg':   loss_seg.item(),
            'deep':  loss_deep.item(),
            'edge':  loss_edge.item(),
            'total': total.item(),
        }


# ═══════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════

def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:>10,}  ({total/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable:>10,}  ({trainable/1e6:.2f}M)")
    print(f"Model size (FP32):    {total * 4 / 1024**2:>10.2f} MB")
    return total


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model   = DCGNet(in_channels=3, num_classes=1, input_size=224).to(device)
    loss_fn = DCGNetLoss()

    count_parameters(model)

    # Training forward pass
    model.train()
    x    = torch.randn(2, 3, 224, 224, device=device)
    mask = torch.randint(0, 2, (2, 1, 224, 224), dtype=torch.float32, device=device)

    outputs = model(x)
    final, deep_out, edge1, edge2, edge3 = outputs
    print(f"Final output:  {final.shape}")
    print(f"Deep sup:      {deep_out.shape}")
    print(f"Edge E1:       {edge1.shape}")
    print(f"Edge E2:       {edge2.shape}")
    print(f"Edge E3:       {edge3.shape}")

    total_loss, loss_dict = loss_fn(outputs, mask)
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    # Eval forward pass
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Eval output: {out.shape}")
