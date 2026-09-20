# Federated Learning Under Data Heterogeneity

**Evaluating Federated Learning Under Data Heterogeneity with Analysis of Failure Modes and Practical Mitigations**

*Jyotiprakash Panda, Om Patil — Accepted for Submission by discover Artificial Intellignece (Springer)*

---

## Overview

This repository contains the full experimental code for a systematic empirical study of federated learning under controlled non-IID data heterogeneity. The central finding is that **worst-client accuracy degrades by up to 60% under extreme heterogeneity while global accuracy stays flat** — a silent client-level failure mode undetected by standard benchmarks.

Key contributions:
- Systematic comparison of FedAvg, FedProx, and q-FedAvg under Dirichlet-based label skew
- Empirical demonstration that server-side client re-weighting collapses under strong non-IID conditions
- Validation that lightweight client-side local data balancing consistently outperforms all server-side alternatives
- Generalization of findings to CIFAR-100 (100-class classification)

---

## Repository Structure

```
.
├── data/
│   ├── dataset_loader.py        # MNIST loader (torchvision, auto-download)
│   ├── cifar10_loader.py        # CIFAR-10 loader with Dirichlet partitioning
│   └── cifar100_loader.py       # CIFAR-100 loader with Dirichlet partitioning
│
├── models/
│   ├── mlp.py                   # MLP for MNIST (784 → 128 → 10)
│   └── cnn.py                   # Shallow CNN for CIFAR-10/100
│
├── federated/
│   └── fedavg.py                # FedAvg, FedProx, q-FedAvg, re-weighting, local balancing
│
├── partition/
│   └── heterogneity.py          # Dirichlet-based label skew partitioning
│
├── metrics/
│   └── evaluation.py            # Per-client accuracy evaluation
│
├── utils/
│   └── seed.py                  # Reproducibility seed setter
│
├── experiments/
│   ├── run_experiment.py        # MNIST baseline (FedAvg across 4 alpha values)
│   ├── run_cifar10.py           # CIFAR-10 baseline: FedAvg / FedProx / q-FedAvg
│   ├── run_cifar10_mitigations.py  # CIFAR-10 mitigation comparison (4 methods × 4 alpha × 5 seeds)
│   ├── run_cifar100.py          # CIFAR-100 generalization (baseline + local balancing)
│   └── confidence_intervals.py  # 95% CI computation from results JSON
│
├── plots/
│   ├── plot.py                  # MNIST accuracy plots (Figures 1–2)
│   ├── plot_comparision.py      # Re-weighting vs balancing comparison plots
│   ├── plot_mitigations.py      # CIFAR-10 mitigation bar chart (Figure 5)
│   ├── cifar10_mitigations_comparison.png  # Figure 5
│   └── cifar100_baseline.png    # Figure 6
│
├── results/
│   ├── cifar10_mitigations.json # CIFAR-10 mitigation results (5 seeds)
│   └── cifar100_baseline.json   # CIFAR-100 generalization results (5 seeds)
│
├── requirements.txt
└── README.md
```

---

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Clients | 10 |
| Communication rounds | 10 |
| Local epochs | 1 |
| Optimizer | SGD, lr = 0.01 |
| Batch size | 16 |
| Seeds | 5 (0–4) |
| Heterogeneity grid (α) | {10.0, 1.0, 0.5, 0.1} |

**Heterogeneity interpretation:**

| α | Regime |
|---|--------|
| 10.0 | Near-IID |
| 1.0 | Moderate heterogeneity |
| 0.5 | Strong heterogeneity |
| 0.1 | Extreme heterogeneity |

Data heterogeneity is introduced via Dirichlet-based label skew following [Hsu et al. (2019)](https://arxiv.org/abs/1909.06335).

---

## Installation

```bash
git clone https://github.com/<your-username>/federated-learning-heterogeneity.git
cd federated-learning-heterogeneity
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Datasets (MNIST, CIFAR-10, CIFAR-100) are downloaded automatically via `torchvision` on first run.

---

## Running Experiments

All scripts are run from the project root.

### 1. MNIST Baseline (FedAvg across α values)
```bash
python experiments/run_experiment.py
```
Prints mean and worst-client accuracy for α ∈ {10.0, 1.0, 0.5, 0.1}.

### 2. CIFAR-10 Baseline (FedAvg / FedProx / q-FedAvg)
```bash
python experiments/run_cifar10.py
```
Runs 3 methods × 3 α values × 5 seeds. Reports mean ± std for mean and worst-client accuracy.

### 3. CIFAR-10 Mitigation Comparison
```bash
python experiments/run_cifar10_mitigations.py
```
Runs 4 methods (FedAvg, Re-weighting, Local Balancing, Combined) × 4 α values × 5 seeds.
Saves results to `results/cifar10_mitigations.json`.

### 4. CIFAR-100 Generalization
```bash
python experiments/run_cifar100.py
```
Runs FedAvg baseline and Local Balancing on CIFAR-100 × 4 α values × 5 seeds.
Saves results to `results/cifar100_baseline.json`.

### 5. Generate Plots
```bash
python plots/plot.py                  # MNIST figures (Figures 1–2)
python plots/plot_comparision.py      # Re-weighting vs balancing (Figures 3–4)
python plots/plot_mitigations.py      # CIFAR-10 mitigation comparison (Figure 5)
```

---

## Methods Implemented

### Aggregation Strategies
- **FedAvg** — uniform averaging of client weights
- **FedProx** — proximal regularization term (μ = 0.01) to reduce client drift
- **q-FedAvg** — loss-based re-weighting to emphasize underperforming clients (q = 5.0)
- **Re-weighted FedAvg** — inverse-proportion aggregation by local dataset size

### Client-Side Mitigation
- **Local Data Balancing** — class-level random oversampling (no SMOTE) before local training; privacy-preserving, no inter-client data required

---

## Key Results

### CIFAR-10 Worst-Client Accuracy (mean ± std, 5 seeds)

| α | FedAvg | Re-weighting | Local Balancing | Combined |
|---|--------|--------------|-----------------|----------|
| 10.0 | 0.563 ± 0.004 | 0.563 ± 0.004 | 0.628 ± 0.003 | 0.625 ± 0.004 |
| 1.0  | 0.506 ± 0.013 | 0.471 ± 0.030 | 0.668 ± 0.009 | 0.638 ± 0.010 |
| 0.5  | 0.438 ± 0.030 | 0.383 ± 0.019 | 0.611 ± 0.017 | 0.558 ± 0.023 |
| 0.1  | 0.226 ± 0.095 | 0.096 ± 0.061 | 0.386 ± 0.063 | 0.276 ± 0.070 |

Local balancing dominates across all heterogeneity levels. Re-weighting collapses to near-random at α = 0.1.

### CIFAR-100 Generalization (mean ± std, 5 seeds)

| α | Baseline Worst | Local Balancing Worst |
|---|----------------|-----------------------|
| 10.0 | 0.199 ± 0.008 | 0.302 ± 0.007 |
| 1.0  | 0.196 ± 0.009 | 0.428 ± 0.009 |
| 0.5  | 0.175 ± 0.011 | 0.415 ± 0.008 |
| 0.1  | 0.121 ± 0.007 | 0.271 ± 0.013 |

---

## Evaluation Metrics

- **Mean Client Accuracy** — average accuracy across all 10 clients
- **Worst-Client Accuracy** — minimum accuracy across all 10 clients (primary diagnostic metric)
- **95% Confidence Interval** — mean ± 1.96 × std across 5 seeds

---

## Citation

If you use this code, please cite:

```
Panda, J., Patil, O. (2026). Evaluating Federated Learning Under Data Heterogeneity
with Analysis of Failure Modes and Practical Mitigations.
Discover Artificial Intelligence, Springer.
```

---

## License

This code is released for research reproducibility. Datasets (MNIST, CIFAR-10, CIFAR-100) are publicly available via PyTorch torchvision.
