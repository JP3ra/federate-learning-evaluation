# import sys
# import os

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, PROJECT_ROOT)

# from data.cifar10_loader import load_cifar10_clients
# from federated.fedavg import federated_training
# from models.cnn import SimpleCNN
# from metrics.evaluation import evaluate_clients


# def run_cifar10(alpha=1.0):
#     print(f"\n### Running CIFAR-10 Federated Learning (alpha = {alpha}) ###")

#     # Load CIFAR-10 clients
#     clients = load_cifar10_clients(
#         num_clients=20,
#         alpha=alpha
#     )

#     # Initialize CNN model
#     model = SimpleCNN(num_classes=10)

#     # Federated training (FedAvg baseline)
#     print("\n--- q-FedAvg Baseline ---")
#     trained_model = federated_training(
#         model,
#         clients,
#         rounds=10,
#         local_epochs=1,
#         use_qfedavg=True,
#         q=5.0
#     )

#     mean_acc, worst_acc = evaluate_clients(trained_model, clients)
#     print(f"q-FedAvg | Mean: {mean_acc:.3f}, Worst: {worst_acc:.3f}")


#     return mean_acc, worst_acc


# if __name__ == "__main__":
#     for alpha in [10, 1.0, 0.5]:
#         run_cifar10(alpha)


import sys
import os
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from data.cifar10_loader import load_cifar10_clients
from federated.fedavg import federated_training
from models.cnn import SimpleCNN
from metrics.evaluation import evaluate_clients
from utils.seed import set_seed


SEEDS = [0, 1, 2, 3, 4]
ALPHAS = [10.0, 1.0, 0.5]


def run_method(method, alpha):
    mean_accs = []
    worst_accs = []

    for seed in SEEDS:
        print(f"\nSeed {seed} | α = {alpha} | Method = {method}")
        set_seed(seed)

        clients = load_cifar10_clients(
            num_clients=10,
            alpha=alpha
        )

        model = SimpleCNN(num_classes=10)

        kwargs = dict(
            rounds=10,
            local_epochs=1
        )

        if method == "fedavg":
            pass

        elif method == "fedprox":
            kwargs.update(dict(
                use_fedprox=True,
                mu=0.01
            ))

        elif method == "qfedavg":
            kwargs.update(dict(
                use_qfedavg=True,
                q=5.0
            ))

        trained_model = federated_training(
            model,
            clients,
            **kwargs
        )

        mean_acc, worst_acc = evaluate_clients(trained_model, clients)

        mean_accs.append(mean_acc)
        worst_accs.append(worst_acc)

    return (
        np.mean(mean_accs), np.std(mean_accs),
        np.mean(worst_accs), np.std(worst_accs)
    )


if __name__ == "__main__":
    for alpha in ALPHAS:
        print(f"\n=== α = {alpha} ===")
        for method in ["fedavg", "fedprox", "qfedavg"]:
            mean_m, mean_s, worst_m, worst_s = run_method(method, alpha)

            print(
                f"{method.upper()} | "
                f"Mean: {mean_m:.3f} ± {mean_s:.3f} | "
                f"Worst: {worst_m:.3f} ± {worst_s:.3f}"
            )
