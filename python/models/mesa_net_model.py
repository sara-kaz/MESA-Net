"""
MESA-Net — Multimodal Edge Sensing and Adaptation Network

Architecture: CNN + Squeeze-and-Excitation + 2-Layer Transformer
Deployed on ESP32-S3 for simultaneous real-time classification of
Physical Activity, Psychological Stress, and Cardiac Arrhythmia
from fused ECG, PPG, and IMU signals.

This file is the public release version of the thesis implementation
and may be modified freely without affecting the frozen thesis code.

Reference:
    Aly, S., & Mirzaei, S. (2026). MESA-Net: Multimodal Edge Sensing
    and Adaptation for Real-Time Biomedical Monitoring on
    Resource-Constrained Devices. IEEE Journal of Biomedical and
    Health Informatics (JBHI).
"""

# ─────────────────────────────────────────────────────────────────────────────
# SETUP INSTRUCTIONS
# Run setup_from_thesis.sh first to copy the base implementation here, then
# replace this stub by editing the copied file.
#
# Or import the thesis model directly during development (read-only):
#   import sys; sys.path.insert(0, '../../src')
#   from models.cnn_transformer_improved import CNNTransformerV3 as MESANet
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Architecture constants (must match firmware NNConfig in nn_inference.h) ──

INPUT_CHANNELS    = 5       # ECG, PPG, AccX, AccY, AccZ
INPUT_SAMPLES     = 1000    # 10 seconds @ 100 Hz
CONV1_OUT_CH      = 32
CONV2_OUT_CH      = 64
CONV3_OUT_CH      = 64
D_MODEL           = 64
NHEAD             = 4
D_FF              = 128
TF_LAYERS         = 2
SE_REDUCTION      = 16

ACTIVITY_CLASSES    = 4   # Sedentary, Walking, Cycling, HighIntensity
STRESS_CLASSES      = 2   # Baseline, Stress
ARRHYTHMIA_CLASSES  = 2   # Normal, Abnormal


# ── Activity label mapping ────────────────────────────────────────────────────

ACTIVITY_LABELS   = ["Sedentary", "Walking", "Cycling", "HighIntensity"]
STRESS_LABELS     = ["Baseline", "Stress"]
ARRHYTHMIA_LABELS = ["Normal", "Abnormal"]

# ── Decision thresholds (match firmware defaults) ─────────────────────────────
# NOTE: These are for the alert gating logic, not for argmax classification.
STRESS_THRESHOLD      = 0.35   # P(stress=1) above this → alert
ARRHYTHMIA_THRESHOLD  = 0.70   # P(arrhythmia=1) above this → alert


def sinusoidal_positional_encoding(seq_len: int, d_model: int,
                                    device: torch.device) -> torch.Tensor:
    """Non-trainable sinusoidal positional encoding. Shape: [1, seq_len, d_model]."""
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1, T, D]


class SqueezeExcitation(nn.Module):
    """Channel attention: globally pools → FC → sigmoid → rescale."""

    def __init__(self, channels: int, reduction: int = SE_REDUCTION):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        w = x.mean(dim=2)              # [B, C]
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w.unsqueeze(2)


class TaskHead(nn.Module):
    """
    3-layer classification head with intermediate LayerNorm.
    Architecture: FC1 → ReLU → LN → FC2 → ReLU → FC3 → logits
    """

    def __init__(self, in_dim: int, hidden1: int, hidden2: int, out_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden1)
        self.ln  = nn.LayerNorm(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, out_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.ln(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MESANet(nn.Module):
    """
    MESA-Net backbone: CNN + SE-Attention + Transformer + 3 task heads.

    Input shape:  [B, 5, 1000]   (batch, channels, time)
    Output:       dict with keys 'activity', 'stress', 'arrhythmia' — each [B, C]

    This is the same architecture as CNNTransformerV3 in the thesis,
    renamed for the public release. All weights are interchangeable.

    Parameters:
        112,552 total
        ├── Backbone (frozen for on-device adaptation): 107,568
        └── Task heads (trainable on-device):             4,984
    """

    def __init__(self):
        super().__init__()

        # ── CNN Backbone ─────────────────────────────────────────────────────
        self.conv1 = nn.Conv1d(INPUT_CHANNELS,  CONV1_OUT_CH, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(CONV1_OUT_CH)
        self.conv2 = nn.Conv1d(CONV1_OUT_CH, CONV2_OUT_CH, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(CONV2_OUT_CH)
        self.conv3 = nn.Conv1d(CONV2_OUT_CH, CONV3_OUT_CH, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(CONV3_OUT_CH)
        self.pool  = nn.MaxPool1d(kernel_size=2)

        # ── Squeeze-and-Excitation ───────────────────────────────────────────
        self.se = SqueezeExcitation(CONV3_OUT_CH, reduction=SE_REDUCTION)

        # ── Transformer ──────────────────────────────────────────────────────
        self.projection = nn.Linear(CONV3_OUT_CH, D_MODEL)
        self.proj_ln    = nn.LayerNorm(D_MODEL)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=D_FF,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=TF_LAYERS)

        # ── Task Heads ───────────────────────────────────────────────────────
        self.activity_head   = TaskHead(D_MODEL, 64, 32, ACTIVITY_CLASSES)
        self.stress_head     = TaskHead(D_MODEL, 48, 24, STRESS_CLASSES)
        self.arrhythmia_head = TaskHead(D_MODEL, 48, 24, ARRHYTHMIA_CLASSES)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 5, 1000] — 5-channel, 1000-sample input window

        Returns:
            dict:
                'activity':   [B, 4] logits
                'stress':     [B, 2] logits
                'arrhythmia': [B, 2] logits
        """
        # CNN feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # [B, 32, 500]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # [B, 64, 250]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # [B, 64, 125]

        # Squeeze-and-Excitation channel reweighting
        x = self.se(x)                                    # [B, 64, 125]

        # Reshape for transformer: [B, T, C]
        x = x.permute(0, 2, 1)                           # [B, 125, 64]

        # Project + LayerNorm + positional encoding
        x = self.proj_ln(self.projection(x))
        x = x + sinusoidal_positional_encoding(x.size(1), D_MODEL, x.device)

        # Transformer
        x = self.transformer(x)                           # [B, 125, 64]

        # Global average pooling over time
        embedding = x.mean(dim=1)                         # [B, 64]

        return {
            'activity':   self.activity_head(embedding),
            'stress':     self.stress_head(embedding),
            'arrhythmia': self.arrhythmia_head(embedding),
        }

    def get_backbone_params(self):
        """Returns backbone parameters (frozen for on-device adaptation)."""
        head_params = set(
            list(self.activity_head.parameters()) +
            list(self.stress_head.parameters()) +
            list(self.arrhythmia_head.parameters())
        )
        return [p for p in self.parameters() if p not in head_params]

    def get_head_params(self):
        """Returns head parameters (trainable for on-device adaptation, 4,984 params)."""
        return (list(self.activity_head.parameters()) +
                list(self.stress_head.parameters()) +
                list(self.arrhythmia_head.parameters()))

    def freeze_backbone(self):
        """Freeze CNN + SE + Transformer for on-device adaptation mode."""
        for p in self.get_backbone_params():
            p.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters for full fine-tuning."""
        for p in self.parameters():
            p.requires_grad = True


# Alias: use CNNTransformerV3 to load thesis checkpoints interchangeably
CNNTransformerV3 = MESANet


if __name__ == '__main__':
    model = MESANet()
    total   = sum(p.numel() for p in model.parameters())
    heads   = sum(p.numel() for p in model.get_head_params())
    backbone = total - heads
    print(f"MESANet | Total params: {total:,} | Backbone: {backbone:,} | Heads: {heads:,}")

    x = torch.randn(2, INPUT_CHANNELS, INPUT_SAMPLES)
    out = model(x)
    for task, logits in out.items():
        print(f"  {task}: {logits.shape}")
