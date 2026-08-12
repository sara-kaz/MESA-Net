"""
INT8 Post-Training Static Quantization for MESA-Net
====================================================
Converts the FP32 trained model to INT8 and measures:
  - Model size reduction
  - CPU inference latency (as ESP32-S3 proxy)
  - Accuracy on the held-out test set

Run after training:
    python quantize_int8.py --checkpoint path/to/best_model.pt \
                            --data_dir  path/to/test_data/

Outputs:
    mesa_net_int8.pt          -- quantized TorchScript model
    quantization_results.json -- size, latency, accuracy comparison
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.quantization import (
    QuantStub,
    DeQuantStub,
    get_default_qconfig,
    prepare,
    convert,
)
from torch.utils.data import DataLoader, TensorDataset

from models.mesa_net_model import MESANet

# ──────────────────────────────────────────────────────────────────────────────
# Quantization wrapper
# ──────────────────────────────────────────────────────────────────────────────

class QuantizedMESANet(nn.Module):
    """Wraps MESANet with QuantStub/DeQuantStub for static INT8 quantization."""

    def __init__(self, model: MESANet):
        super().__init__()
        self.quant   = QuantStub()
        self.model   = model
        self.dequant_act  = DeQuantStub()
        self.dequant_str  = DeQuantStub()
        self.dequant_arr  = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        out = self.model(x)
        return {
            'activity':   self.dequant_act(out['activity']),
            'stress':     self.dequant_str(out['stress']),
            'arrhythmia': self.dequant_arr(out['arrhythmia']),
        }

    def fuse_model(self):
        """Fuse Conv+BN+ReLU chains for quantization efficiency."""
        m = self.model
        torch.quantization.fuse_modules(m, [['conv1', 'bn1']], inplace=True)
        torch.quantization.fuse_modules(m, [['conv2', 'bn2']], inplace=True)
        torch.quantization.fuse_modules(m, [['conv3', 'bn3']], inplace=True)


# ──────────────────────────────────────────────────────────────────────────────
# Calibration & conversion
# ──────────────────────────────────────────────────────────────────────────────

def calibrate(model, loader, n_batches=50):
    """Run calibration batches to collect activation statistics."""
    model.eval()
    with torch.no_grad():
        for i, (x, *_) in enumerate(loader):
            model(x)
            if i >= n_batches:
                break


def quantize_model(fp32_model: MESANet, calib_loader: DataLoader) -> QuantizedMESANet:
    qmodel = QuantizedMESANet(fp32_model)
    qmodel.eval()
    qmodel.fuse_model()

    # Use x86 qconfig for simulation; swap to 'fbgemm' or 'qnnpack' as needed
    qmodel.qconfig = get_default_qconfig('fbgemm')
    prepare(qmodel, inplace=True)

    print("Calibrating on representative data...")
    calibrate(qmodel, calib_loader)

    convert(qmodel, inplace=True)
    print("INT8 conversion complete.")
    return qmodel


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def measure_latency(model, input_shape=(1, 5, 1000), n_runs=200):
    """Measure mean CPU inference latency over n_runs (single-sample)."""
    model.eval()
    x = torch.randn(*input_shape)
    # Warm up
    with torch.no_grad():
        for _ in range(20):
            model(x)
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)  # ms
    return float(np.mean(times)), float(np.std(times))


def evaluate(model, loader, device='cpu'):
    """Returns per-task accuracy dict."""
    model.eval()
    correct = {'activity': 0, 'stress': 0, 'arrhythmia': 0}
    total   = {'activity': 0, 'stress': 0, 'arrhythmia': 0}

    with torch.no_grad():
        for batch in loader:
            x, act_lbl, str_lbl, arr_lbl, act_mask, str_mask, arr_mask = batch
            x = x.to(device)
            out = model(x)

            for task, lbl, mask in [
                ('activity',   act_lbl, act_mask),
                ('stress',     str_lbl, str_mask),
                ('arrhythmia', arr_lbl, arr_mask),
            ]:
                idx = mask.bool()
                if idx.sum() == 0:
                    continue
                pred = out[task][idx].argmax(dim=1)
                gt   = lbl[idx]
                correct[task] += (pred == gt).sum().item()
                total[task]   += idx.sum().item()

    return {
        t: (correct[t] / total[t] * 100 if total[t] > 0 else float('nan'))
        for t in correct
    }


def model_size_mb(model) -> float:
    """Estimate model size by saving to buffer."""
    buf = torch.jit.script(model)
    import io
    b = io.BytesIO()
    torch.jit.save(buf, b)
    return len(b.getvalue()) / 1e6


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                        help='Path to FP32 .pt checkpoint (state_dict)')
    parser.add_argument('--data_dir',   required=True,
                        help='Directory containing X_test.npy, y_act_test.npy, etc.')
    parser.add_argument('--out_dir',    default='.',
                        help='Output directory for quantized model + results')
    parser.add_argument('--calib_frac', type=float, default=0.2,
                        help='Fraction of test set used for calibration')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    X        = torch.tensor(np.load(data_dir / 'X_test.npy'),         dtype=torch.float32)
    y_act    = torch.tensor(np.load(data_dir / 'y_act_test.npy'),     dtype=torch.long)
    y_str    = torch.tensor(np.load(data_dir / 'y_str_test.npy'),     dtype=torch.long)
    y_arr    = torch.tensor(np.load(data_dir / 'y_arr_test.npy'),     dtype=torch.long)
    m_act    = torch.tensor(np.load(data_dir / 'mask_act_test.npy'),  dtype=torch.float32)
    m_str    = torch.tensor(np.load(data_dir / 'mask_str_test.npy'),  dtype=torch.float32)
    m_arr    = torch.tensor(np.load(data_dir / 'mask_arr_test.npy'),  dtype=torch.float32)

    dataset = TensorDataset(X, y_act, y_str, y_arr, m_act, m_str, m_arr)
    n_calib = int(len(dataset) * args.calib_frac)
    calib_ds, eval_ds = torch.utils.data.random_split(
        dataset, [n_calib, len(dataset) - n_calib],
        generator=torch.Generator().manual_seed(42)
    )
    calib_loader = DataLoader(calib_ds, batch_size=32, shuffle=False)
    eval_loader  = DataLoader(eval_ds,  batch_size=64, shuffle=False)

    # ── Load FP32 model ──────────────────────────────────────────────────────
    fp32_model = MESANet()
    state = torch.load(args.checkpoint, map_location='cpu')
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    fp32_model.load_state_dict(state)
    fp32_model.eval()

    # ── Baseline FP32 metrics ────────────────────────────────────────────────
    print("\n── FP32 Baseline ──")
    fp32_lat_mean, fp32_lat_std = measure_latency(fp32_model)
    fp32_acc = evaluate(fp32_model, eval_loader)
    fp32_size = sum(p.numel() * 4 for p in fp32_model.parameters()) / 1e6  # MB

    print(f"  Latency: {fp32_lat_mean:.1f} ± {fp32_lat_std:.1f} ms")
    print(f"  Size:    {fp32_size:.2f} MB")
    print(f"  Accuracy: {fp32_acc}")

    # ── INT8 quantization ────────────────────────────────────────────────────
    print("\n── INT8 Quantization ──")
    int8_model = quantize_model(MESANet(), calib_loader)
    # Reload FP32 weights into a fresh model before quantizing
    fp32_fresh = MESANet()
    fp32_fresh.load_state_dict(state)
    fp32_fresh.eval()
    int8_model = quantize_model(fp32_fresh, calib_loader)

    print("\n── INT8 Metrics ──")
    int8_lat_mean, int8_lat_std = measure_latency(int8_model)
    int8_acc = evaluate(int8_model, eval_loader)
    int8_size = fp32_size / 4  # Approximate; exact size from saved model

    print(f"  Latency: {int8_lat_mean:.1f} ± {int8_lat_std:.1f} ms")
    print(f"  Size:    ~{int8_size:.2f} MB (approx)")
    print(f"  Accuracy: {int8_acc}")

    speedup   = fp32_lat_mean / int8_lat_mean
    acc_delta = {t: int8_acc[t] - fp32_acc[t] for t in fp32_acc}

    print(f"\n  Speedup:      {speedup:.1f}x")
    print(f"  Accuracy delta: {acc_delta}")

    # ── Save results ─────────────────────────────────────────────────────────
    results = {
        'fp32': {
            'latency_ms_mean': fp32_lat_mean,
            'latency_ms_std':  fp32_lat_std,
            'size_mb':         fp32_size,
            'accuracy':        fp32_acc,
        },
        'int8': {
            'latency_ms_mean': int8_lat_mean,
            'latency_ms_std':  int8_lat_std,
            'size_mb_approx':  int8_size,
            'accuracy':        int8_acc,
        },
        'speedup_x':     speedup,
        'accuracy_delta': acc_delta,
    }

    out_path = out_dir / 'quantization_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("\n── Copy these numbers into the paper ──")
    print(f"  FP32 inference: {fp32_lat_mean:.0f} ms")
    print(f"  INT8 inference: {int8_lat_mean:.0f} ms  ({speedup:.1f}x speedup)")
    print(f"  Activity acc drop:   {acc_delta['activity']:+.1f} pp")
    print(f"  Stress acc drop:     {acc_delta['stress']:+.1f} pp")
    print(f"  Arrhythmia acc drop: {acc_delta['arrhythmia']:+.1f} pp")


if __name__ == '__main__':
    main()
