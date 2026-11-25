# MS-VMUNet: Multi-Scale Visual Mamba U-Net with High-Frequency Difference Edge Residual Skip Connections for Accurate and Real-Time Colorectal Polyp Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.1-76B900.svg?style=flat&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Mamba](https://img.shields.io/badge/Mamba--SSM-1.2.0-blue.svg)](https://github.com/state-spaces/mamba)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Official PyTorch implementation and reproducible evaluation suite for **MS-VMUNet**, an ultra-lightweight (**1.85M** parameters), real-time (**65.2 FPS**) Visual State Space Model tailored for colorectal polyp segmentation under cold-start (un-pre-trained) clinical constraints.

---

## 🌟 Key Highlights

- **Ultra-Lightweight & Pareto-Efficient**: Only **1.85M parameters** (reducing standard U-Net by **76.2%** and PraNet by **92.5%**) with **11.02 GFLOPs**.
- **Dramatic Boundary Refinement (HD95)**: Reduces 95th percentile Hausdorff Distance from 34.63 px down to **26.41 ± 1.77 px** across 5 independent seeds (**-8.21 px**, paired $t$-test $p = 0.0003 < 0.001^{***}$).
- **Video-Rate Clinical Deployment**: Single-frame forward latency of **15.34 ms (65.2 FPS)** on an NVIDIA RTX 3090, exceeding the 30 FPS clinical video standard by **2.2×** with a peak VRAM allocation of only **207.8 MB**.
- **Robust Multi-Center Zero-Shot Generalization**: Evaluated on **1,188 unseen endoscopic frames** across three international medical centers (CVC-ClinicDB: **67.69% vs. 61.39%**, CVC-ColonDB: **61.32% vs. 52.07%**, ETIS-Larib: **41.51% vs. 38.71%**).
- **100% Empirically Verifiable**: Fully deterministic data splits, pre-trained weights for Seed 42 and Seed 200, and comprehensive 5-seed ablation matrices.

---

## 🏛️ Method Overview

MS-VMUNet introduces three complementary inductive biases to resolve the spatial anisotropy and boundary blurring inherent in naive 2D Mamba networks:
1. **Multi-Scale Depthwise Convolution (MS-Conv)**: Parallel multi-receptive-field branches ($3\times3$, dilated $r=2, 3$, and $1\times1$) inject local spatial priors while optimizing FLOPs relative to dense convolutions (11.02G vs. 12.58G).
2. **4-Way 2D Cross-Scanning (4-Way CS-SSM)**: Traverses 2D feature maps along four complementary trajectories (horizontal forward/backward, vertical forward/backward) to eliminate spatial directional anisotropy.
3. **High-Frequency Difference Edge Residual Skip (HFD-Skip)**: Isolates high-frequency boundary contours via spatial difference ($F_{high} = F_{skip} - \text{AvgPool}(F_{skip})$), preventing low-pass blurring caused by unconstrained wide-kernel convolutions and avoiding multiplicative gradient attenuation.
4. **Sobel Boundary Gradient Loss ($\mathcal{L}_{grad}$)**: Non-parametric spatial 2D gradient supervision forcing steep probability transitions along true anatomical margins.

---

## 📊 Benchmark Results

### 1. 5-Seed Head-to-Head Rigorous Benchmark on Kvasir-SEG (Table 3)
Evaluated across 5 independent random seeds (42, 100, 200, 300, 400) under the un-pre-trained from-scratch protocol:

| Architecture | Params (M) | FLOPs (G) | Test Dice (%) | Test IoU (%) | HD95 (px) $\downarrow$ | Sensitivity (%) | Specificity (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard U-Net** | 7.76 | 13.75 | 84.78 ± 1.19% | 76.76 ± 1.20% | 34.63 ± 1.81 | 87.81 ± 0.84% | 97.46 ± 0.18% |
| **MS-VMUNet-Lite [Ours]** | **1.85** | **11.02** | **87.29 ± 1.42%** | **80.42 ± 1.41%** | **26.41 ± 1.77** | **89.29 ± 1.53%** | **97.85 ± 0.25%** |
| **Paired $t$-test ($\Delta$ / $p$-value)** | **-76.2%** | **-19.9%** | **+2.51 pp ($p=\mathbf{0.0238}^*$)** | **+3.66 pp ($p=\mathbf{0.0027}^{**}$)** | **-8.21 px ($p=\mathbf{0.0003}^{***}$)** | +1.48 pp ($p=0.109$) | +0.39 pp ($p=0.073$) |

*Best single-seed checkpoint (**Seed 200**): **Dice 88.96%**, **IoU 82.20%**, **HD95 24.48 px**.*

---

### 2. Multi-Center Zero-Shot Generalization (Table 5)
Direct blind testing on 1,188 unseen colonoscopy images without fine-tuning:

| Target Dataset | Origin / Clinical Center | Images | Standard U-Net Dice | **MS-VMUNet-Lite Dice [Ours]** | Net Gain | HD95 (px) $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CVC-ClinicDB** | Hospital Clínic de Barcelona, Spain | 612 | 61.39% | **67.69%** | **+6.30 pp** | **46.12 vs. 65.60** |
| **CVC-ColonDB** | Hospital Clínic de Barcelona, Spain | 380 | 52.07% | **61.32%** | **+9.25 pp** | **49.32 vs. 54.74** |
| **ETIS-Larib** | Lariboisière Hospital, France | 196 | 38.71% | **41.51%** | **+2.80 pp** | 65.04 vs. 59.11 |

---

### 3. Hardware Latency, Throughput & Memory Consumption (Table 6)
Profiled on an NVIDIA RTX 3090 GPU (200 test iterations with 50 warmup runs and CUDA synchronization):

| Model | Batch Size ($B$) | Forward Latency (ms) | Throughput (FPS) | Peak VRAM (MB) | Clinical Compliance ($>30$ FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard U-Net** | 1 | 2.00 ms | 501.1 FPS | 228.1 MB | Compliant (16.7×) |
| | 4 | 6.36 ms | 628.6 FPS | 283.2 MB | Compliant (21.0×) |
| | 16 | 23.60 ms | 677.9 FPS | 952.2 MB | Compliant (22.6×) |
| **MS-VMUNet-Lite [Ours]** | 1 | **15.34 ms** | **65.2 FPS** | **207.8 MB** | **Compliant (2.2×)** |
| | 4 | **32.68 ms** | **122.4 FPS** | **551.3 MB** | **Compliant (4.1×)** |
| | 16 | **120.19 ms** | **133.1 FPS** | **2,100.8 MB** | **Compliant (4.4×)** |

---

### 4. Systematic 5-Seed Component Ablation Matrix (Table 7)
Fully executed across Seeds 42, 100, 200, 300, and 400 ($N=5$, mean $\pm$ sample std dev, $ddof=1$):

| Ablation Configuration | Params (M) | FLOPs (G) | Test Dice (%) | Test IoU (%) | HD95 (px) $\downarrow$ | Paired $p$-val (HD95) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full MS-VMUNet-Lite [Proposed]** | **1.85** | **11.02** | **87.29 ± 1.42%** | **80.42 ± 1.41%** | **26.41 ± 1.77** | — (Reference) |
| **w/o 4-Way CS-SSM (2-Way)** | 1.51 | 10.20 | 83.98 ± 0.95% | 76.04 ± 1.40% | 30.07 ± 1.58 | $\mathbf{0.0016}^{**}$ (Collapse) |
| **w/o MS-Conv (Single 3$\times$3 Conv)** | 1.68 | 12.58 | 85.83 ± 0.80% | 78.51 ± 1.01% | 25.96 ± 1.85 | 0.7165 (-1.46 pp Dice) |
| **Plain Skip (Vanilla Concat)** | 1.82 | 10.75 | 85.86 ± 1.37% | 78.91 ± 1.78% | 26.43 ± 3.13 | 0.9945 (Variance ×1.8) |
| **Multiplicative Gated Skip** | 1.85 | 11.01 | 85.80 ± 1.07% | 78.81 ± 1.18% | 25.72 ± 1.65 | 0.3761 (-1.49 pp Dice) |
| **w/o Sobel Boundary Loss** | 1.85 | 11.02 | 86.14 ± 0.65% | 79.09 ± 0.98% | 26.80 ± 1.74 | 0.6572 (-1.15 pp Dice) |
| **VM-UNet Slim [Controlled]** | 1.66 | 12.31 | 86.59 ± 1.10% | 79.51 ± 1.22% | 25.93 ± 2.71 | 0.7020 (Higher Variance) |

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/WangAnChuan/MS-VMUNet.git
cd MS-VMUNet

# Create conda environment
conda create -n msvmunet python=3.10 -y
conda activate msvmunet

# Install PyTorch with CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Install official Mamba CUDA kernels
pip install causal-conv1d==1.2.0 mamba-ssm==1.2.0

# Install other requirements
pip install -r requirements.txt
```

### 2. Download Datasets
Run the one-click dataset downloader to retrieve Kvasir-SEG and PolyDataset (ClinicDB, ColonDB, ETIS):
```bash
bash data/download_datasets.sh
```

---

## ⚡ Reproduce Experiments

### One-Click Reproduction Pipeline
To run the full verification pipeline (hardware profiling, zero-shot evaluation, and figure generation):
```bash
bash reproduce.sh
```

### Evaluate Pre-Trained Weights
We provide pre-trained model checkpoints in `checkpoints/`:
```bash
# Evaluate Seed 200 (Highest Performance Checkpoint: 88.96% Dice / 24.48 px HD95)
python evaluate.py --model ms_vmunet --checkpoint checkpoints/nextgen_full_best_seed200.pth

# Evaluate Seed 42 (Canonical Baseline)
python evaluate.py --model ms_vmunet --checkpoint checkpoints/nextgen_full_best_seed42.pth
```

### Train MS-VMUNet from Scratch
```bash
python train.py --model ms_vmunet --seed 42 --epochs 60 --batch_size 8 --grad_accum 2 --lr 2e-4
```

### Run Hardware Benchmark
```bash
python benchmark_hardware_unified.py
```

### Reproduce the Full 5-Seed Systematic Ablation Matrix
```bash
python run_rigorous_5seed_ablations.py
```

---

## 📂 Repository Structure

```text
MS-VMUNet/
├── models/
│   ├── __init__.py
│   ├── ms_vmunet.py             # Core MS-VMUNet architecture (MS-Conv, 4-Way CS-SSM, HFD-Skip)
│   └── unet.py                  # Standard U-Net baseline
├── checkpoints/
│   ├── nextgen_full_best_seed200.pth  # Top checkpoint (88.96% Dice)
│   ├── nextgen_full_best_seed42.pth   # Canonical baseline checkpoint
│   └── README.md
├── data/
│   ├── download_datasets.sh     # One-click downloader for all 4 benchmark datasets
│   └── README.md
├── results/                     # Empirical benchmark CSVs (Ablations, Zero-shot, Latency)
│   ├── README.md
│   ├── master_nextgen_experiments.csv
│   ├── rigorous_5seed_ablation_summary.csv
│   ├── cross_dataset_unified_results.csv
│   └── hardware_latency_benchmark.csv
├── dataset.py                   # PyTorch dataset loader with deterministic 800:100:100 split
├── losses.py                    # BCE, Dice, TV, and Sobel Boundary Gradient loss
├── metrics.py                   # Quantitative clinical metrics (Dice, IoU, HD95, Sens, Spec)
├── train.py                     # Single-model training script
├── evaluate.py                  # Single-checkpoint multi-dataset evaluation script
├── benchmark_hardware_unified.py# Standardized hardware latency and VRAM profiling
├── run_rigorous_5seed_ablations.py # Systematic 5-seed multi-module ablation runner
├── reproduce.sh                 # Master one-click reproduction script
├── environment.yml              # Conda environment definition
├── requirements.txt             # Pip environment dependencies
├── LICENSE                      # MIT License
└── README.md
```

---

## 📜 Citation

If you find this work or code useful in your research, please cite our paper:

```bibtex
@article{ms_vmunet_2026,
  title={MS-VMUNet: Multi-Scale Visual Mamba U-Net with High-Frequency Difference Edge Residual Skip Connections for Accurate and Real-Time Colorectal Polyp Segmentation},
  author={[Author Names]},
  journal={Computers in Biology and Medicine},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
