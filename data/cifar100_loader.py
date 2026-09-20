import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import datasets, transforms
from collections import defaultdict
import numpy as np


def load_cifar100_clients(
    root="./data",
    num_clients=10,
    alpha=1.0,
    seed=42
):
    """
    Load CIFAR-100 and partition across clients using Dirichlet label skew.
    Returns:
        clients: dict {client_id: [(x, y), ...]}
    """
    np.random.seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761)
        )
    ])

    dataset = datasets.CIFAR100(
        root=root,
        train=True,
        download=True,
        transform=transform
    )

    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label_to_indices[label].append(idx)

    clients = defaultdict(list)

    for label, indices in label_to_indices.items():
        np.random.shuffle(indices)
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split_indices = np.split(indices, proportions)

        for client_id, idxs in enumerate(split_indices):
            for idx in idxs:
                clients[client_id].append(dataset[idx])

    return clients
