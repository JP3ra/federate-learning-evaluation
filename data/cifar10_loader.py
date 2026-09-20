import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import datasets, transforms
from collections import defaultdict
import numpy as np

def load_cifar10_clients(
    root="./data",
    num_clients=20,
    alpha=1.0,
    seed=42
):
    """
    Load CIFAR-10 and partition it across clients using Dirichlet label skew.
    Returns:
        clients: dict {client_id: [(x, y), ...]}
    """

    np.random.seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    dataset = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transform
    )

    # Group samples by label
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label_to_indices[label].append(idx)

    clients = defaultdict(list)

    # Dirichlet partitioning
    for label, indices in label_to_indices.items():
        np.random.shuffle(indices)
        proportions = np.random.dirichlet(
            alpha * np.ones(num_clients)
        )

        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split_indices = np.split(indices, proportions)

        for client_id, idxs in enumerate(split_indices):
            for idx in idxs:
                clients[client_id].append(dataset[idx])

    return clients
