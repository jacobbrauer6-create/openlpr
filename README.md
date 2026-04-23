# 🚗 OpenLPR — Open-Source Licence Plate Recognition Pipeline

A comprehensive, production-ready MLOps pipeline for training licence plate recognition models across **70+ countries**, with built-in accuracy benchmarking, energy/CO₂ tracking, and continuous dataset improvement.

---

## Features

| Module | Description |
|--------|-------------|
| `data/` | Dataset collection, labeling, augmentation for domestic + international plates |
| `training/` | Multi-backbone training (ResNet-18/34/50/101), configurable hyperparams |
| `evaluation/` | Accuracy vs. latency benchmarking on target GPU hardware |
| `mlops/` | Dataset versioning, drift detection, iterative improvement loop |
| `scripts/` | CLI tools for every pipeline stage |
| `energy/` | CO₂ and kWh tracking per training run via CodeCarbon |

---

## Quick Start

```bash
git clone https://github.com/jacobbrauer6-create/openlpr.git
cd openlpr

# Set up your virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# 4. Generate Synthetic Training Data
# Since raw datasets are excluded to keep the repo lean, 
# start by generating a synthetic set for testing.
python scripts/generate_synthetic.py --count 1000 --region us

# 1. Download + prepare dataset
python scripts/prepare_dataset.py --regions us eu asia --split 80/10/10

# 2. Train with ResNet-18, could also train with another model here if desired
python scripts/train.py --backbone resnet18 --config configs/base.yaml --track-energy

# 3. Evaluate accuracy vs. latency
python scripts/evaluate.py --checkpoint checkpoints/latest.pt --gpu-profile

# 4. Run full MLOps iteration
python scripts/mlops_iteration.py --version v2 --compare-to v1
```

## Hardware Acceleration

OpenLPR automatically detects the best available device:
- **NVIDIA GPU**: Uses CUDA. Ensure [NVIDIA Drivers](https://www.nvidia.com/drivers) are installed.
- **Apple Silicon (M1/M2/M3)**: Uses Metal (MPS) for high-performance training.
- **Fallback**: Defaults to CPU if no compatible GPU is found.

### GPU Installation Tip
If you have an NVIDIA GPU and `torch.cuda.is_available()` returns `False`, reinstall Torch with the specific CUDA index:
```bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```
---



## Dataset Strategy

### Domestic (USA)
- **Sources**: UFPR-ALPR, OpenALPR dataset, Kaggle LPR datasets, synthetic generation
- **Formats**: Standard, Specialty (veteran, handicap), State variants (50 states)
- **Augmentation**: Weather simulation, motion blur, night/day, partial occlusion

### International
- **Europe**: EU standard blue stripe, country-specific variants (DE, FR, UK, IT, ES, NL, PL...)
- **Asia-Pacific**: China, Japan, South Korea, India, Australia
- **Americas**: Canada, Mexico, Brazil, Argentina
- **Middle East/Africa**: UAE, Saudi Arabia, South Africa

### Data Sources Pipeline
```
Raw Sources → Scraping/Download → Deduplication → Quality Filter
    → Annotation (YOLO format) → Augmentation → Train/Val/Test Split
    → DVC Version Control → MLflow Registry
```

---

## Model Architectures

| Backbone | Params | Typical mAP | Latency (V100) |
|----------|--------|-------------|----------------|
| ResNet-18 | 11M | 91.2% | 4.2ms |
| ResNet-34 | 21M | 93.8% | 6.1ms |
| ResNet-50 | 25M | 95.4% | 9.3ms |
| ResNet-101 | 44M | 96.1% | 16.8ms |

Training uses a two-stage approach:
1. **Detector**: Locates plate bounding box (YOLO-style head on ResNet backbone)
2. **OCR**: Reads characters from cropped plate region (CTC loss + LSTM)

---

## MLOps Loop

```
Deploy → Monitor → Collect Hard Cases → Re-annotate
    → Retrain → Evaluate → A/B Test → Deploy
```

Key tools:
- **DVC** — dataset versioning
- **MLflow** — experiment tracking
- **Label Studio** — active learning annotation
- **Evidently** — data drift detection
- **CodeCarbon** — energy/CO₂ tracking

---

## Energy & CO₂ Tracking

Every training run automatically logs:
- **kWh consumed** (wall-clock power × time)
- **CO₂ equivalent** (kWh × regional grid intensity)
- **Emissions per accuracy point** (efficiency metric)

Regional grid intensities (gCO₂/kWh): US avg ~386, EU avg ~275, France ~85, Norway ~26

See `energy/README.md` for details.

---

## Repo Structure

```
openlpr/
├── configs/          # YAML configs for training runs
├── data/
│   ├── raw/          # Downloaded raw datasets
│   ├── processed/    # Cleaned, labeled data (DVC tracked)
│   └── augmented/    # Augmented training splits
├── models/
│   ├── backbones/    # ResNet implementations
│   └── checkpoints/  # Saved model weights
├── training/
│   ├── trainer.py
│   ├── losses.py
│   └── augmentations.py
├── evaluation/
│   ├── benchmark.py
│   └── metrics.py
├── mlops/
│   ├── drift_detector.py
│   ├── active_learning.py
│   └── versioning.py
├── energy/
│   ├── tracker.py
│   └── report.py
├── scripts/          # CLI entry points
├── dashboard/        # MLOps monitoring dashboard
└── docs/             # Dataset collection guides
```

---

## Contributing

1. Fork the repo
2. Add new country/region dataset in `docs/datasets/`
3. Submit PR with sample images and annotation format

## License

Apache 2.0 — free to use commercially and in research.
