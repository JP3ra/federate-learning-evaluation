import numpy as np 
import random 
from collections import defaultdict


def group_by_label(dataset):
    """
    Groups are sampled based on their label 

    Input:
        dataset: list of data (data, label)

    Output:
        dict[label] - list of samples
    """

    label_groups = defaultdict(list)
    for data, label in dataset:
        label_groups[label].append([data, label])

    return label_groups


# Step1 -> Do an IID distribution 

def iid_partition(dataset, num_clients):
    """
    Splits the dataset equally across all clients

    Every clients gets:
        - same number of data points
        - equal distribution of labels 
    """

    random.shuffle(dataset)
    client_data = []

    samples_per_client = len(dataset) // num_clients

    for i in range(num_clients):
        start = int(i * samples_per_client)
        end = int(start + samples_per_client)
        client_data.append(dataset[start:end])

    return client_data


# Creating a non IID Dataset

def dirichlet_partition(dataset, num_clients, alpha, num_classes):

    # Step1: group data by label
    label_groups = group_by_label(dataset)

    # Step2: initialize empty clients
    client_data = [[] for _ in range(num_clients)]

    # Step3: for each label, split it acorss every client 
    for label in range(num_classes):
        samples = label_groups[label]
        random.shuffle(samples)

         # Draw proportions for each client
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))

        # Convert proportions into sample counts
        proportions = (np.cumsum(proportions) * len(samples)).astype(int)[:-1]

        # Split samples according to proportions
        split_samples = np.split(np.array(samples, dtype=object), proportions)

        for client_id, client_samples in enumerate(split_samples):
            client_data[client_id].extend(client_samples.tolist())

    return client_data


def print_client_summary(client_data):
    """
    Prints number of samples and label distribution per client.
    """
    for i, client in enumerate(client_data):
        labels = [label for _, label in client]
        unique, counts = np.unique(labels, return_counts=True)

        print(f"\nClient {i}:")
        print(f"  Total samples: {len(client)}")
        print(f"  Label distribution:")
        for u, c in zip(unique, counts):
            print(f"    Label {u}: {c}")


if __name__ == "__main__":
    dataset = []
    for i in range(100):
        label = i % 10
        dataset.append((i, label))

    alphas = [10.0, 1.0, 0.5, 0.1]

    for alpha in alphas:
        print(f"\n=== Dirichlet Non-IID (alpha = {alpha}) ===")
        clients = dirichlet_partition(
            dataset,
            num_clients=5,
            alpha=alpha,
            num_classes=10
        )
        print_client_summary(clients)
