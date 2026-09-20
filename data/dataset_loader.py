# data/dataset_loader.py

import torch
from torchvision import datasets, transforms

def load_mnist_subset(num_samples=3000):
    """
    Loads a small subset of MNIST and returns it
    as a list of (image, label) pairs.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),   # converts to [0,1]
    ])

    mnist_train = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    dataset = []

    for i in range(num_samples):
        image, label = mnist_train[i]
        image = image.view(-1)  # flatten 28x28 → 784
        dataset.append((image, label))

    return dataset
