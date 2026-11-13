# RSNA Knee Abnormality Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Leaderboard](https://img.shields.io/badge/Public%20AUC-0.9576%20%2F%200.956-gold.svg)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)

A complete, production-grade machine learning solution developed for the **[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)** Kaggle competition.

This repository tracks the complete evolutionary journey across **180+ iterative models**—progressing from a **0.532** metadata baseline to a top-tier **0.956 – 0.9576 Public AUC** (placing in the top ~1–2% / Gold-Silver tier among 6,000+ teams).

---

## Table of Contents

- [RSNA Knee Abnormality Detection](#rsna-knee-abnormality-detection)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Performance Milestones](#performance-milestones)
  - [Key Innovations \& Technical Approach](#key-innovations--technical-approach)
    - [1. LLM-Assisted Radiology Report Re-Labeling](#1-llm-assisted-radiology-report-re-labeling)
    - [2. Multi-View 2.5D / 3D Vision Architectures](#2-multi-view-25d--3d-vision-architectures)
    - [3. Frozen-Backbone Bag-Level Head Retraining](#3-frozen-backbone-bag-level-head-retraining)
    - [4. Multi-Stage Ensembling \& Dynamic Time Budgeting](#4-multi-stage-ensembling--dynamic-time-budgeting)
  - [What Worked vs. What Didn't](#what-worked-vs-what-didnt)
  - [Repository Structure](#repository-structure)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Environment Setup](#environment-setup)
    - [Data Layout](#data-layout)
  - [Workflow \& Reproduction Guide](#workflow--reproduction-guide)
    - [Baseline Model](#baseline-model)
    - [Training Custom Models](#training-custom-models)
    - [Generating Submissions](#generating-submissions)
    - [Security \& Secret Scanning](#security--secret-scanning)
  - [Compute \& Infrastructure](#compute--infrastructure)
  - [Acknowledgements \& References](#acknowledgements--references)

---

## Project Overview

The objective of the **RSNA Knee Abnormality Detection** challenge is to accurately identify 12 distinct knee abnormalities from multi-plane MRI exams, including:
- **Ligamentous tears**: Anterior Cruciate Ligament (ACL) tear, Posterior Cruciate Ligament (PCL) tear
- **Meniscal pathology**: Medial meniscus tear, Lateral meniscus tear
- **Cartilage & joint pathology**: Osteochondral lesions, Cartilage defects, Joint effusion, Baker's cyst, Patellofemoral osteoarthritis, and related knee conditions.

Each patient study contains multiple MRI series covering three anatomical planes:
- **Sagittal view** (ideal for ACL, PCL, and meniscal morphology)
- **Coronal view** (collateral ligaments and meniscal bodies)
- **Axial view** (patellofemoral compartment and cartilage surfaces)

A core challenge of this competition is the label distribution: only **58 studies** contain gold-standard expert radiologist annotations, while **4,349 studies** contain unannotated MRI volumes paired with noisy, free-text clinical radiology reports.

---

## Performance Milestones

```
0.960 +----------------------------------------------------------- Model 177-181 (0.9576)
      |                                                           * * * * *
0.950 |                                                 Model 103 (0.947)
      |                                                 *
0.940 |                             Model 61 (0.943)   *
      |                             *
0.930 |           Model 19 (0.935)  *
      |           *
0.530 |  Model 1 (0.532)
      +----------------------------------------------------------------------------------> Iterations
```

| Iteration | Model / Recipe | Validation / Public AUC | Key Technique |
|:---|:---|:---:|:---|
| **Model 1** | Acquisition metadata logistic regression | 0.532 | Diagnostic baseline using slice counts, plane metadata, and sequence flags |
| **Model 10–19** | DINOv2 visual feature classifiers | 0.935 | Frozen DINOv2 embeddings over 6 scan slots with neighboring-slice stacking |
| **Model 23–43** | Multi-view CoAtNet arms + Gemini relabeling | 0.942 | 4-view CoAtNet backbones retrained on Gemini 3.8 Flash structured report extractions |
| **Model 61** | Frozen-backbone bag-level head retrain | 0.943 | Multi-window study bag pooling retrained with plain soft-label BCE |
| **Model 76** | Speedy Raptors pipeline + Model 61 weights | 0.945 | Swapping project-trained Raptor weights into the high-efficiency Speedy inference pipeline |
| **Model 103** | Model 76 + CoAtNet-2 384px Rank-Blend Leg | **0.947** | Rank-blending Model 76 with custom CoAtNet-2 384px models on dense frame caches |
| **Model 110–113** | TPU 5-Fold CoAtNet-2 384 + Pseudo-label student | 0.950+ | Full 5-fold cross-validation on TPU v3-8 (PyTorch XLA) generating OOF teacher predictions |
| **Model 150–176** | Multi-family fleet ensembles | 0.955 | Blending CoAtNet-2 + ConvNeXt + round-2 student models |
| **Model 177–181** | Multi-Stage Raptor (384/224) + Compact R7 Bundles | **0.956 – 0.9576** | High-capacity rank blend of multi-resolution Raptors (384 + 224) with compact ConvNeXt bundles |

---

## Key Innovations & Technical Approach

### 1. LLM-Assisted Radiology Report Re-Labeling
- **The Problem**: Raw training labels extracted by simple heuristics agreed with the 58 expert gold studies at only **0.893 AUC**, injecting severe noise into image backbones.
- **The Solution**: All 4,407 free-text radiology reports were re-parsed using **Gemini 3.8 Flash** with medically guided prompt templates (`Other/scripts/llm_relabel.py`).
- **Result**: Label-to-gold agreement improved from **0.893 to 0.9035 AUC**, lifting the cross-validation score of downstream vision models from 0.9273 to 0.9301.

### 2. Multi-View 2.5D / 3D Vision Architectures
- **Backbone Families**:
  - **CoAtNet-2** (384x384 and 224x224): Combines convolutional locality for fine bone/cartilage edges with self-attention for long-range slice dependencies.
  - **ConvNeXt-Small**: Modernized pure convolutional networks providing diversity against attention-based models.
  - **DINOv2**: Self-supervised vision transformers providing strong foundational representations without task supervision.
- **Volumetric Sampling**:
  - Rather than treating slices independently, slices are packaged into contiguous multi-slice windows (12 to 24 windows per series) across Sagittal, Coronal, and Axial series.

### 3. Frozen-Backbone Bag-Level Head Retraining
- Retrained only the classification heads (`Other/knee/train_heads.py`) on full study bags of extracted backbone features using plain soft-label Binary Cross-Entropy (BCE).
- Boosted every single model arm by **+0.002 to +0.005 AUC** (consistent across 12/12 seeds) with negligible compute overhead (executed in minutes on Kaggle T4 GPUs).

### 4. Multi-Stage Ensembling & Dynamic Time Budgeting
- **Rank Blending**: Ensembles combined predictions using rank-averaging (`scipy.stats.rankdata`) rather than raw probability averaging, eliminating calibration discrepancies across heterogeneous architectures.
- **Dynamic Inference Time-Budgeting (`fleet_leg_v3` – `v6`)**:
  - Kaggle hidden test sets enforce strict notebook timeout limits (e.g., 9 hours total run time).
  - Heavy ensembles run the risk of catastrophic timeouts.
  - Our inference pipeline (`fleet_leg_v6.py`) reads each DICOM file **once** for all geometries and monitors notebook elapsed runtime. If runtime thresholds are approached, it dynamically drops lower-priority leg models and safely produces the high-scoring base blend.

---

## What Worked vs. What Didn't

| Technique | Result | Impact / Reason |
|:---|:---:|:---|
| **Gemini Report Re-labeling** | **Worked** | Raised label quality from 0.893 to 0.9035 AUC; cleaner training signal for all models. |
| **Head Retraining on Full Bags** | **Worked** | Free +0.002 to +0.005 gain across all model arms using soft BCE on frozen features. |
| **Full CoAtNet-2 384px Training** | **Worked** | Best standalone own model family (0.939 single-model AUC). |
| **Rank Blending (Speedy + Fleet)** | **Worked** | Blending complementary architectures pushed LB from 0.945 to 0.947+. |
| **Single-Pass DICOM Reading** | **Worked** | Cut leg inference time by ~40%, allowing deeper ensembles within Kaggle's hidden limit. |
| **Direct Fine-tuning on 58 Gold Studies** | *Harmful* | 5-fold CV collapsed from 0.9254 to 0.9014; severe overfitting to the tiny gold set. |
| **Horizontal Flip TTA** | *Harmful* | MRI knee anatomy is laterally asymmetric; flipping left/right knees inverts medial/lateral compartments. |
| **Raw External MRI Data Transfer** | *Neutral* | External knee datasets (MRNet, fastMRI+) had domain/protocol shift that diluted the competition signal without target-domain adaptation. |
| **Pseudo-Labels for Frozen Heads** | *Neutral* | Only effective when fine-tuning full deep backbones; frozen heads saturated quickly. |

---

## Repository Structure

```text
RSNA Knee Abnormality Detection/
├── README.md                      # Main public documentation (this file)
├── .env.example                   # Template for environment variables & API keys
├── .gitignore                     # Git ignore rules (protects credentials & large data)
├── AGENTS.md                      # System execution guidelines & workflow rules
│
├── Model/                         # 180+ Model Experiment Directories
│   ├── model1/                    # Baseline logistic regression on metadata
│   ├── model10/ ... model19/      # DINOv2 feature classifiers & early ensembles
│   ├── model23/ ... model61/      # CoAtNet arms, Gemini relabels, and retrained heads
│   ├── model75/ ... model103/     # CoAtNet-2 384px training, Speedy Raptors, & fleet legs
│   ├── model110/ ... model113/    # TPU 5-fold models & pseudo-label students
│   └── model150/ ... model181/    # Final ensembles & high-scoring R6/R7 ConvNeXt bundles
│       ├── readme.txt             # Experiment log, parameters, and findings
│       ├── score.txt              # Verified leaderboard / validation score
│       ├── notebook.ipynb         # Kaggle submission notebook
│       └── kernel-metadata.json   # Kaggle API push configuration
│
├── Other/
│   ├── knee/                      # Core Python Library Package
│   │   ├── data.py                # Dataset loaders and PyTorch Dataset definitions
│   │   ├── models.py              # Vision backbones & classification head architectures
│   │   ├── preprocess.py          # DICOM windowing, normalizations, and plane slicing
│   │   ├── dense.py               # Dense slice volume sampling
│   │   └── train_heads.py         # Bag-level head retraining implementation
│   │
│   ├── scripts/                   # Training & Experiment Pipelines
│   │   ├── llm_relabel.py         # Gemini 3.8 Flash radiology report extraction
│   │   ├── train_model16.py       # Full training recipe for CoAtNet architectures
│   │   ├── train_xla.py           # PyTorch XLA training script for Google Cloud TPUs
│   │   ├── run_head_retrain.py    # Head retrain execution across bags
│   │   ├── make_fleet_leg_v6.py   # High-efficiency multi-model leg inference builder
│   │   └── layout_data_links.py   # Idempotent data directory symlinker / layout tool
│   │
│   ├── tools/                     # Automation & CI/CD Tooling
│   │   ├── secret_scan.py         # Pre-commit secret scanning engine
│   │   ├── tick_light.py          # Submission polling & execution watchdog
│   │   └── score_watch.py         # Leaderboard rank and score tracker
│   │
│   ├── cloud/                     # Cloud Infrastructure Automation
│   │   ├── vast.py                # Vast.ai on-demand GPU instance manager
│   │   └── deploy_run.py          # Remote cloud job deployment & artifact recovery
│   │
│   ├── docs/                      # Reference Documentation
│   │   └── official_label_criteria.md  # Annotation definitions for the 12 knee labels
│   │
│   ├── EXPERIMENTS.md             # Detailed chronological experiment log (180+ models)
│   ├── REPRODUCE.md               # End-to-end reproduction guide
│   ├── SOURCES.md                 # Attribution for open-source code & external models
│   └── requirements.txt           # Python dependencies
│
└── Data/                          # Dataset Directory (Ignored from git; see Data Setup)
    ├── competition/               # Raw DICOMs and competition CSVs
    ├── labels/                    # Gemini relabels & weak label sets
    └── external/                  # External reference datasets
```

---

## Getting Started

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA 11.8+ (for local training/inference) or Kaggle/Vast.ai compute
- Git (configured with Git LFS if storing local weights)

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yodharin-monplub/RNSA-KNEE.git
   cd RNSA-KNEE
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r Other/requirements.txt
   # For GPU training:
   pip install -r Other/requirements-gpu.txt
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example Other/.env
   # Edit Other/.env with your Kaggle API credentials, cloud tokens, etc.
   ```

### Data Layout

The codebase uses a clean separation between competition data and repository code. You can automatically configure directory junctions and hardlinks:

```bash
python Other/scripts/layout_data_links.py
```

Expected `Data/` layout:
```text
Data/
├── competition/
│   ├── train.csv
│   ├── test.csv
│   ├── train_series.csv
│   ├── test_series.csv
│   └── sample_submission.csv
└── labels/
    ├── llm_relabel/
    └── weak_labels_v4/
```

---

## Workflow & Reproduction Guide

### Baseline Model
To run the metadata-based baseline evaluation locally:
```bash
python Other/scripts/baseline.py
```
Outputs are generated in `Other/artifacts/baseline/` including `metrics.json`, `folds.csv`, and `oof.csv`.

### Training Custom Models
1. **Re-extract Report Labels via LLM:**
   ```bash
   python Other/scripts/llm_relabel.py --prompt v2 --output Data/labels/llm_relabel/
   ```

2. **Train CoAtNet-2 Backbones (Kaggle TPU / PyTorch XLA):**
   ```bash
   python Other/scripts/train_xla.py --config Model/model110/config.json --cache s96 --fold 0
   ```

3. **Retrain Prediction Heads on Frozen Features:**
   ```bash
   python Other/scripts/run_head_retrain.py --model model61 --loss bce
   ```

### Generating Submissions
Each model folder in `Model/` contains a self-contained submission notebook (`notebook.ipynb`) and metadata (`kernel-metadata.json`). To push a model directly to Kaggle:
```bash
kaggle kernels push -p Model/model103/
```

### Security & Secret Scanning
A pre-commit secret scanning tool is included to prevent any accidental leakage of API keys, `.env` tokens, or sensitive credentials:
```bash
python Other/tools/secret_scan.py
```
This utility scans staged diffs against patterns for Kaggle tokens, Discord tokens, OpenRouter keys, and private keys.

---

## Compute & Infrastructure

This solution was engineered across a cost-effective, hybrid cloud topology:
- **Local PC**: Diagnostic baselines, metadata modeling, and script orchestration.
- **Kaggle T4 / P100 GPUs**: Fast inference validation, dummy submission runs, and frozen-feature head retraining.
- **Kaggle TPU v3-8 (PyTorch XLA)**: High-throughput 5-fold cross-validation and CoAtNet-2 384px pre-training.
- **Vast.ai Cloud**: On-demand spot GPU instances for dense slice cache generation and model fine-tuning (managed via `Other/cloud/vast.py` within an authorized budget).

---

## Acknowledgements & References

- **RSNA (Radiological Society of North America)** for hosting the Knee Abnormality Detection challenge.
- **CoAtNet**: Dai et al., *CoAtNet: Marrying Convolution and Attention for All Data Sizes* (NeurIPS 2021).
- **ConvNeXt**: Liu et al., *A ConvNet for the 2020s* (CVPR 2022).
- **DINOv2**: Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision* (2023).
- **Gemini**: Google DeepMind for Gemini 3.8 Flash utilized for automated clinical report annotation.
- The Kaggle community for inspiring public discussions, benchmark notebooks, and shared baseline architectures.
