import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_loader import load_mnist_subset
from models.model import SimpleNet
from partition.heterogneity import dirichlet_partition
from federated.fedavg import federated_training
from metrics.evaluation import evaluate_model

# Load MNIST (small subset)
dataset = load_mnist_subset(num_samples=3000)

num_clients = 10
num_classes = 10
# alpha = 0.5   # start with one value
alphas = [10.0, 1.0, 0.5, 0.1]
for alpha in alphas:
    print("### Starting the federated learning model for alpha : ", alpha)
    # Create clients
    clients = dirichlet_partition(
        dataset,
        num_clients=num_clients,
        alpha=alpha,
        num_classes=num_classes
    )

    # Initialize model
    model = SimpleNet(input_dim=28*28, num_classes=10)

    # Train with FedAvg
    model = federated_training(
        model,
        clients,
        rounds=10,
        local_epochs=1
    )

    # Evaluate per-client
    print("\nClient accuracies:")
    client_accuracies = []

    for i, client in enumerate(clients):
        acc = evaluate_model(model, client)
        client_accuracies.append(acc)
        print(f"Client {i}: {acc:.3f}")

    print("\nSummary for alpha:", alpha)
    print(f"Mean client accuracy: {sum(client_accuracies)/len(client_accuracies):.3f}")
    print(f"Worst client accuracy: {min(client_accuracies):.3f}")
