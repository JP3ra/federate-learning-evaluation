"""
CIFAR-10 mitigation experiments:
  - baseline (FedAvg)
  - client re-weighting (inverse-proportion aggregation)
  - local data balancing (class-level oversampling)
  - combined (re-weighting + local balancing)

Runs 5 seeds x 4 alpha values x 4 methods.
"""

import sys
import os
import json
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from data.cifar10_loader import load_cifar10_clients
from federated.fedavg import federated_training
from models.cnn import SimpleCNN
from metrics.evaluation import evaluate_clients
from utils.seed import set_seed


SEEDS = [0, 1, 2, 3, 4]
ALPHAS = [10.0, 1.0, 0.5, 0.1]
ROUNDS = 10
NUM_CLIENTS = 10


def run_method(method, alpha):
    mean_accs, worst_accs = [], []

    for seed in SEEDS:
        print(f"  seed={seed} | α={alpha} | method={method}")
        set_seed(seed)

        clients = load_cifar10_clients(num_clients=NUM_CLIENTS, alpha=alpha, seed=seed)
        model = SimpleCNN(num_classes=10)

        kwargs = dict(rounds=ROUNDS, local_epochs=1)

        if method == "baseline":
            pass
        elif method == "reweighting":
            kwargs["use_reweighting"] = True
        elif method == "local_balancing":
            kwargs["local_balancing"] = True
        elif method == "combined":
            kwargs["use_reweighting"] = True
            kwargs["local_balancing"] = True

        trained = federated_training(model, clients, **kwargs)
        mean_acc, worst_acc = evaluate_clients(trained, clients)

        mean_accs.append(mean_acc)
        worst_accs.append(worst_acc)

    return {
        "mean_mean": float(np.mean(mean_accs)),
        "mean_std":  float(np.std(mean_accs)),
        "worst_mean": float(np.mean(worst_accs)),
        "worst_std":  float(np.std(worst_accs)),
        "worst_ci95_lo": float(np.mean(worst_accs) - 1.96 * np.std(worst_accs)),
        "worst_ci95_hi": float(np.mean(worst_accs) + 1.96 * np.std(worst_accs)),
    }


if __name__ == "__main__":
    results = {}

    for alpha in ALPHAS:
        results[alpha] = {}
        print(f"\n=== α = {alpha} ===")

        for method in ["baseline", "reweighting", "local_balancing", "combined"]:
            r = run_method(method, alpha)
            results[alpha][method] = r
            print(
                f"  {method:20s} | "
                f"Mean: {r['mean_mean']:.3f}±{r['mean_std']:.3f} | "
                f"Worst: {r['worst_mean']:.3f}±{r['worst_std']:.3f} "
                f"[95% CI: {r['worst_ci95_lo']:.3f}, {r['worst_ci95_hi']:.3f}]"
            )

    # Save results
    out_path = os.path.join(PROJECT_ROOT, "results", "cifar10_mitigations.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")
