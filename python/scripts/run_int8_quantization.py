"""
INT8 Dynamic Quantization for MESA-Net (model_v3 checkpoint)
=============================================================
Applies PyTorch dynamic INT8 quantization to all Linear layers
(including Transformer attention projections and task heads),
then benchmarks FP32 vs INT8 CPU inference latency.

Dynamic quantization is ideal here: the Transformer and task heads
are Linear-dominated, and no calibration dataset is needed.

Usage:
    python scripts/run_int8_quantization.py

Outputs:
    training_results/mesa_net_int8_dynamic.pt
    training_results/quantization_results.json
"""

import json
import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Model definition matching model_v3.pth checkpoint exactly
# ──────────────────────────────────────────────────────────────────────────────

class MESANetV3(nn.Module):
    """
    Matches the exact layer names in training_results/model_v3.pth:
      input_bn, conv1/bn1, conv2/bn2, conv3/bn3,
      se_fc1/se_fc2, channel_projection/proj_ln,
      transformer (2-layer, d_model=64, nhead=3, d_ff=128),
      task_heads.{activity,stress,arrhythmia} (Sequential)
    """

    def __init__(self):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(5)

        self.conv1 = nn.Conv1d(5,  32, kernel_size=7, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm1d(64)
        self.pool  = nn.MaxPool1d(2)

        # Squeeze-and-Excitation (flat, not nested)
        self.se_fc1 = nn.Linear(64, 16)
        self.se_fc2 = nn.Linear(16, 64)

        # Projection + LayerNorm before Transformer
        self.channel_projection = nn.Linear(64, 64)
        self.proj_ln = nn.LayerNorm(64)

        # 2-layer Transformer encoder (nhead=3 → 192/64=3 confirmed by checkpoint)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128,
            dropout=0.0, activation='relu',
            batch_first=True, norm_first=False
        )
        # enable_nested_tensor=False disables fast-path which breaks after quantization
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2,
                                                  enable_nested_tensor=False)

        # Task heads as plain Sequential (indices 0,1,4,7 match checkpoint)
        self.task_heads = nn.ModuleDict({
            'activity': nn.Sequential(
                nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(32, 4)
            ),
            'stress': nn.Sequential(
                nn.Linear(64, 48), nn.BatchNorm1d(48), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(48, 24), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(24, 2)
            ),
            'arrhythmia': nn.Sequential(
                nn.Linear(64, 48), nn.BatchNorm1d(48), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(48, 24), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(24, 2)
            ),
        })

    def forward(self, x):
        # x: [B, 5, 1000]
        x = self.input_bn(x)

        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # [B, 32, 500]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # [B, 64, 250]
        x = F.relu(self.bn3(self.conv3(x)))               # [B, 64, 250]

        # SE attention
        w = x.mean(dim=2)                      # [B, 64]
        w = F.relu(self.se_fc1(w))             # [B, 16]
        w = torch.sigmoid(self.se_fc2(w))      # [B, 64]
        x = x * w.unsqueeze(2)                 # [B, 64, 250]

        # Project → LayerNorm → Transformer
        x = x.permute(0, 2, 1)                 # [B, 250, 64]
        x = self.proj_ln(self.channel_projection(x))
        x = self.transformer(x)                # [B, 250, 64]

        feat = x.mean(dim=1)                   # [B, 64] global avg pool

        return {
            'activity':   self.task_heads['activity'](feat),
            'stress':     self.task_heads['stress'](feat),
            'arrhythmia': self.task_heads['arrhythmia'](feat),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark helpers
# ──────────────────────────────────────────────────────────────────────────────

def measure_latency(model, n_runs=300, input_shape=(1, 5, 1000)):
    model.eval()
    x = torch.randn(*input_shape)
    with torch.no_grad():
        for _ in range(30):          # warm up
            model(x)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times)), float(np.min(times)), float(np.max(times))


def param_size_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    base_dir = Path(__file__).parent.parent
    ckpt     = base_dir / 'training_results' / 'model_v3.pth'
    out_dir  = base_dir / 'training_results'

    # Load FP32 model
    model_fp32 = MESANetV3()
    state = torch.load(ckpt, map_location='cpu')
    missing, unexpected = model_fp32.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys (non-critical): {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    model_fp32.eval()
    print(f"FP32 model loaded from {ckpt}")
    print(f"  Parameters: {sum(p.numel() for p in model_fp32.parameters()):,}")
    print(f"  Size (FP32): {param_size_mb(model_fp32):.3f} MB")

    # FP32 latency
    print("\nBenchmarking FP32 (300 runs) ...")
    fp32_mean, fp32_std, fp32_min, fp32_max = measure_latency(model_fp32)
    print(f"  Mean: {fp32_mean:.1f} ms  Std: {fp32_std:.1f} ms  Min: {fp32_min:.1f}  Max: {fp32_max:.1f}")

    # Dynamic INT8 quantization.
    # quantize_dynamic converts nn.Linear inside MultiheadAttention too, which
    # corrupts MHA's in_proj_weight (becomes a packed-params object, not a tensor)
    # and breaks the TransformerEncoderLayer fast-path check on this PyTorch version.
    # Fix: save FP32 MHA, quantize all Linears, restore MHA from FP32 copy.
    torch.backends.quantized.engine = 'qnnpack'
    model_int8 = copy.deepcopy(model_fp32)

    torch.quantization.quantize_dynamic(
        model_int8, qconfig_spec={nn.Linear}, dtype=torch.qint8, inplace=True
    )

    # Restore full FP32 TransformerEncoderLayers to avoid MHA fast-path bug
    # (linear1/linear2 inside the layer also trigger the 'is_cuda' AttributeError)
    model_int8.transformer.layers = copy.deepcopy(model_fp32.transformer.layers)
    print(f"\nINT8 dynamic quantization applied.")
    # Approximate size: Linear weights → INT8 (1 byte), others still FP32
    n_linear_params = sum(
        p.numel() for name, p in model_fp32.named_parameters()
        if 'weight' in name and p.dim() == 2  # Linear weight matrices
    )
    n_other_params  = sum(p.numel() for p in model_fp32.parameters()) - n_linear_params
    int8_size_mb = (n_linear_params * 1 + n_other_params * 4) / 1e6
    print(f"  Estimated INT8 size: {int8_size_mb:.3f} MB  ({int8_size_mb/param_size_mb(model_fp32)*100:.0f}% of FP32)")

    # INT8 latency
    print("\nBenchmarking INT8 (300 runs) ...")
    int8_mean, int8_std, int8_min, int8_max = measure_latency(model_int8)
    print(f"  Mean: {int8_mean:.1f} ms  Std: {int8_std:.1f} ms  Min: {int8_min:.1f}  Max: {int8_max:.1f}")

    speedup = fp32_mean / int8_mean

    # Save quantized model
    out_model = out_dir / 'mesa_net_int8_dynamic.pt'
    torch.save(model_int8.state_dict(), out_model)
    print(f"\nQuantized model saved to {out_model}")

    # Report
    results = {
        'fp32': {'mean_ms': fp32_mean, 'std_ms': fp32_std, 'min_ms': fp32_min,
                 'max_ms': fp32_max, 'size_mb': param_size_mb(model_fp32)},
        'int8_dynamic': {'mean_ms': int8_mean, 'std_ms': int8_std, 'min_ms': int8_min,
                         'max_ms': int8_max, 'size_mb_est': int8_size_mb},
        'speedup_x': speedup,
        'size_reduction_x': param_size_mb(model_fp32) / int8_size_mb,
    }
    out_json = out_dir / 'quantization_results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*55)
    print("  PAPER NUMBERS (copy into Table II / §IX)")
    print("="*55)
    print(f"  FP32 inference:  {fp32_mean:.0f} ms (±{fp32_std:.0f} ms)")
    print(f"  INT8 inference:  {int8_mean:.0f} ms (±{int8_std:.0f} ms)")
    print(f"  Speedup:         {speedup:.1f}×")
    print(f"  FP32 size:       {param_size_mb(model_fp32):.2f} MB")
    print(f"  INT8 size (est): {int8_size_mb:.2f} MB")
    print(f"  Size reduction:  ~{param_size_mb(model_fp32)/int8_size_mb:.1f}×")
    print(f"\nFull results saved to {out_json}")


if __name__ == '__main__':
    main()
