# Are You Sure You Are in the Right Place?
### Adaptive Re-ranking for Efficient and Reliable Visual Place Recognition

**Davide Maietta, Valentina Romeo, Christian Rossolino, Davide Sisto**  
Politecnico di Torino — Machine Learning Project (Extension 6.1)

--- 

## Data

Dataset images, raw `.torch` feature files and Colab logs (too large for GitHub) are available on Google Drive:

📁 [Progetto_Machine_Learning — Google Drive](https://drive.google.com/drive/folders/TUO_LINK_QUI)

---

## Overview

Visual Place Recognition (VPR) is commonly formulated as an image retrieval problem, where an optional re-ranking step verifies the top retrieved candidates using local image matching and RANSAC. Re-ranking can improve accuracy, but it is **one to two orders of magnitude more expensive** than retrieval — and, as we show, it can even hurt a strong retriever.

This project has two parts:

1. **Benchmark** of four global descriptors combined with three local matchers on four test sets, reporting Recall@N and per-query timing.
2. **Adaptive re-ranking**: a logistic regression gate that activates full geometric verification only when the top-1 retrieval is likely wrong, saving matching cost without sacrificing accuracy.

---

## Methods

| Component | Options |
|-----------|---------|
| **Retrievers** | NetVLAD, CosPlace, MixVPR, MegaLoc |
| **Matchers** | SuperGlue, SuperPoint+LightGlue, LoFTR |
| **Datasets** | SVOX Sun, SVOX Night, SF-XS, Tokyo-XS |
| **Metric** | Recall@N, τ = 25 m, K = 20 |

---

## Key Results

### Adaptive gate (FULL model)
- **NetVLAD**: recovers +13 to +28 R@1 gain from full re-ranking while skipping 8–46% of matching calls.
- **MegaLoc**: avoids accuracy losses of up to 5.5 R@1 caused by blind re-ranking, skipping 44–100% of full top-20 re-ranking operations.
- Correctness estimator reaches up to **0.99 AUROC**.

### Fixed validation threshold
Using a threshold F1-optimised on SF-XS validation:
- MegaLoc skips **94–100%** of re-ranking with no accuracy cost.
- NetVLAD+SuperPoint+LightGlue tracks full re-ranking within 1 R@1 point while skipping 8–49% of cost.

---

## Setup

```python
# In Colab — set your base directory
BASE_DIR    = "/content/drive/MyDrive/Progetto_Machine_Learning"
PROJECT_DIR = f"{BASE_DIR}/Visual-Place-Recognition-Project"
FEAT_DIR    = f"{BASE_DIR}/feature_csv_6_1"
RESULTS_DIR = f"{BASE_DIR}/results_6_1_alltests"
```

Datasets can be downloaded with:
```bash
python download_datasets.py
```

Dependencies are installed automatically by the notebook setup cell.

> **Note:** Dataset images (~9.7 GB) and raw `.torch` log/feature files (~25 GB) are stored on Google Drive and are not included in this repository.

---

## Authors

| Name | Email |
|------|-------|
| Davide Maietta | s354172@studenti.polito.it |
| Valentina Romeo | s353850@studenti.polito.it |
| Christian Rossolino | s363405@studenti.polito.it |
| Davide Sisto | s360589@studenti.polito.it |
