# MESA-Net: Multimodal Edge Sensing and Adaptation Network

**IEEE JBHI Submission — Sara Aly & Shahnam Mirzaei, CSUN ECE**

End-to-end wearable health monitoring system deployed on a Seeed XIAO ESP32-S3.
Simultaneously classifies physical activity (4-class), psychological stress (binary),
and cardiac arrhythmia (binary) from ECG, PPG, and accelerometer signals using a
112,552-parameter CNN–SE–Transformer model with on-device adaptive training.

---

## Directory Structure

```
MESA-Net/
├── checkpoints/
│   ├── mesa_net_v3.pth          # FP32 trained model (main checkpoint)
│   ├── mesa_net_int8_dynamic.pt # INT8 dynamic-quantized model
│   └── quantization_results.json
├── python/
│   ├── models/
│   │   └── mesa_net_model.py    # MESANet class definition
│   └── scripts/
│       ├── train_v3.py          # Training pipeline
│       ├── eval_loso.py         # Leave-one-subject-out evaluation
│       ├── ablation_study.py    # Ablation experiments
│       ├── adaptation_experiment.py  # Multi-session adaptation protocol
│       ├── run_int8_quantization.py  # INT8 PTQ benchmark
│       ├── data_loader.py       # Dataset loading utilities
│       └── dataset_integration.py   # PPG-DaLiA / WESAD / MIT-BIH integration
├── firmware/
│   └── esp32/                   # PlatformIO ESP32-S3 firmware (C++)
│       ├── src/main.cpp
│       ├── include/
│       └── platformio.ini
├── figures/                     # Wiring diagrams, architecture figures
├── docs/
│   └── loso_eval_results.json   # Full LOSO evaluation results
└── README.md
```

---

## Model Performance (LOSO, 13 unseen subjects)

| Task | Accuracy | AUC | Dataset |
|------|----------|-----|---------|
| Activity (4-class) | 94.1% | — | PPG-DaLiA |
| Stress (binary) | 87.1% | 0.975 | WESAD |
| Arrhythmia (binary) | 90.8% | 0.879 | MIT-BIH |

**Parameters:** 112,552 (FP32, no compression)  
**Inference:** 2,338 ms / 10-second window on ESP32-S3 (23.4% duty cycle)  
**Adaptation:** 70 ms / episode, 4,984 head parameters, 9 safety mechanisms

---

## Datasets

All datasets are publicly available via [PhysioNet](https://physionet.org):
- **PPG-DaLiA** — activity labels (15 subjects)
- **WESAD** — stress labels (15 subjects)
- **MIT-BIH Arrhythmia Database** — arrhythmia labels (48 subjects)

Use `python/scripts/dataset_integration.py` to preprocess and unify.

---

## Quick Start

```bash
pip install -r python/requirements.txt

# Load model
import torch, sys
sys.path.insert(0, 'python')
from models.mesa_net_model import MESANet

model = MESANet()
model.load_state_dict(torch.load('checkpoints/mesa_net_v3.pth', map_location='cpu'))
model.eval()

# Inference: input shape [B, 5, 1000] (5 channels × 1000 samples @ 100 Hz)
x = torch.randn(1, 5, 1000)
out = model(x)  # {'activity': [B,4], 'stress': [B,2], 'arrhythmia': [B,2]}
```

---

## Firmware

Built with PlatformIO targeting `seeed_xiao_esp32s3`.  
Flash with:
```bash
cd firmware/esp32
pio run --target upload
```

---

## Citation

```bibtex
@article{aly2026mesanet,
  author  = {Aly, Sara and Mirzaei, Shahnam},
  title   = {{MESA-Net}: Multimodal Edge Sensing and Adaptation for Simultaneous
             Arrhythmia, Stress, and Activity Monitoring},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026}
}
```
