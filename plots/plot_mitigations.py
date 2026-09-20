"""
Generate all paper figures from saved JSON results.

Usage:
    python plots/plot_mitigations.py

Figures produced:
    plots/cifar10_mitigations_comparison.png  — Table 4 equivalent for CIFAR-10
    plots/cifar100_baseline.png               — CIFAR-100 avg vs worst
"""

import sys
import os
import json
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))

ALPHAS = [0.1, 0.5, 1.0, 10.0]  # ascending for x-axis


def load(filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path) as f:
        return json.load(f)


def plot_cifar10_mitigations():
    data = load("cifar10_mitigations.json")

    methods = {
        "baseline":       ("FedAvg (baseline)", "o", "-"),
        "reweighting":    ("Client Re-weighting", "s", "--"),
        "local_balancing":("Local Data Balancing", "^", "-."),
        "combined":       ("Combined", "D", ":"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for method, (label, marker, ls) in methods.items():
        means  = [data[str(a)][method]["mean_mean"]  for a in ALPHAS]
        worsts = [data[str(a)][method]["worst_mean"] for a in ALPHAS]

        axes[0].plot(ALPHAS, means,  marker=marker, linestyle=ls, label=label)
        axes[1].plot(ALPHAS, worsts, marker=marker, linestyle=ls, label=label)

    for ax, title in zip(axes, ["Mean Client Accuracy", "Worst-Client Accuracy"]):
        ax.set_xscale("log")
        ax.set_xlabel("Dirichlet Alpha (log scale)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"CIFAR-10: {title}")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "cifar10_mitigations_comparison.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


def plot_cifar100_baseline():
    data = load("cifar100_baseline.json")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    methods = {
        "baseline":        ("FedAvg (baseline)", "o", "-"),
        "local_balancing": ("Local Data Balancing", "^", "-."),
    }

    for method, (label, marker, ls) in methods.items():
        means  = [data[str(a)][method]["mean_mean"]  for a in ALPHAS]
        worsts = [data[str(a)][method]["worst_mean"] for a in ALPHAS]

        axes[0].plot(ALPHAS, means,  marker=marker, linestyle=ls, label=label)
        axes[1].plot(ALPHAS, worsts, marker=marker, linestyle=ls, label=label)

    for ax, title in zip(axes, ["Mean Client Accuracy", "Worst-Client Accuracy"]):
        ax.set_xscale("log")
        ax.set_xlabel("Dirichlet Alpha (log scale)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"CIFAR-100: {title}")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "cifar100_baseline.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    files = os.listdir(RESULTS_DIR) if os.path.isdir(RESULTS_DIR) else []

    if "cifar10_mitigations.json" in files:
        plot_cifar10_mitigations()
    else:
        print("cifar10_mitigations.json not found — run run_cifar10_mitigations.py first")

    if "cifar100_baseline.json" in files:
        plot_cifar100_baseline()
    else:
        print("cifar100_baseline.json not found — run run_cifar100.py first")
