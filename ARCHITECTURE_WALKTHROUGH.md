# Architecture Walkthrough: OpenLPR

This document provides a deep-dive into the engineering decisions and neuroscientific foundations of the **OpenLPR** pipeline. It bridges the gap between the Python implementation and the peer-reviewed research that inspired it.

---

## 1. MLOps Quality Gates
Before training begins, data passes through three rigorous quality gates in `scripts/prepare_dataset.py`.

| Gate | Technique | Purpose |
| :--- | :--- | :--- |
| **Deduplication** | 64-bit Perceptual Hashing | Prevents data leakage between train/test sets by detecting near-identical images. |
| **Blur Filter** | Laplacian Variance Scoring | Discards images below a threshold (e.g., `< 100.0`) to ensure clear character recognition. |
| **Bootstrap** | Procedural Generation | Overcomes the "Cold Start" problem by bootstrapping with 5,000+ synthetic samples. |

---

## 2. Biologically-Inspired Advancements
The core innovation of **OpenLPR v2** is the transition from flat image processing to an ordinal, foveal-peripheral hierarchy.

### A. Space Bigram Ordinal Position Head (SBOPH)
* **Implementation**: `class SBOPH` in `scripts/train_bio_v2.py`.
* **Foundation**: Based on **Visual Word Form Area (VWFA)** research (PNAS 2020), suggesting the brain encodes letters relative to boundaries.
* **Impact**: Replaces absolute pixel coordinates with parallel **left-anchored** and **right-anchored** decoders fused with position-dependent weights.

### B. Foveal-Parafoveal Dual Resolution Module (FPDRM)
* **Implementation**: `class FPDRM` in `scripts/train_bio_v2.py`.
* **Foundation**: Emulates the human retina's dual-stream: high-resolution **fovea** for detail and low-resolution **parafovea** for context (Nature 2021).
* **Result**: This mechanism provided an additive **+0.400pp IoU** improvement in ablation studies.

### C. Sequential Saccadic Character Attention (SSCA)
* **Implementation**: `class SSCA` in `scripts/train_bio_v2.py`.
* **Foundation**: Models discrete **saccadic eye movements** (fixate -> decode -> predict next saccade).
* **Impact**: Utilizes a **Location GRU** to predict coordinates and a **Recognition GRU** to decode the specific glimpse.

---

## 3. Performance & Sustainability
We utilize **CodeCarbon** to quantify the environmental cost of accuracy gains.

### Efficiency Metrics
* **ResNet-18 Baseline**: Achieved **0.9120 Mean IoU** with a p50 latency of **44.8ms** on CPU.
* **v5_LRC Variant**: Pushed the frontier to **0.9740 IoU**.
* **Carbon Footprint**: The standard ResNet-18 run consumed **5.3g CO₂**.
* **Effect Size**: The bio-variants achieved a **Cohen’s d of 4.0**, indicating a "Large" statistical impact on model performance.

---

## 4. Foundational Bibliography
> This project stands on the "shoulders of giants," implementing architectural changes from the following works:

1.  **Xu et al. (2018)**: *Towards End-to-End License Plate Detection and Recognition* (CCPD).
2.  **Rao & Ballard (1999)**: *Predictive coding in the visual cortex*.
3.  **Tan & Le (2019)**: *EfficientNet: Rethinking Model Scaling for CNNs*.
4.  **Dehaene et al. (2005)**: *The neural code for written words* (Ordinal encoding).

---
