import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
import random


# ---------- Utility: Local data balancing ----------

def balance_client_data(client_data):
    label_groups = defaultdict(list)
    for x, y in client_data:
        label_groups[y].append((x, y))

    max_count = max(len(v) for v in label_groups.values())
    balanced = []

    for samples in label_groups.values():
        if len(samples) < max_count:
            samples = samples + random.choices(samples, k=max_count - len(samples))
        balanced.extend(samples)

    random.shuffle(balanced)
    return balanced


# ---------- Local training (FedAvg / FedProx) ----------

def train_local_model(
    global_model,
    client_data,
    epochs=1,
    batch_size=16,
    lr=0.01,
    local_balancing=False,
    use_fedprox=False,
    mu=0.01
):
    device = torch.device("cpu")

    # Save global parameters for FedProx
    global_params = None
    if use_fedprox:
        global_params = {
            name: p.detach().clone()
            for name, p in global_model.named_parameters()
        }

    model = copy.deepcopy(global_model)
    model.to(device)
    model.train()

    if local_balancing:
        client_data = balance_client_data(client_data)

    X = torch.stack([x for x, _ in client_data])
    y = torch.tensor([y for _, y in client_data])

    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    total_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)

            if use_fedprox:
                prox = 0.0
                for name, param in model.named_parameters():
                    prox += torch.norm(param - global_params[name]) ** 2
                loss = loss + (mu / 2.0) * prox

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    return model.state_dict(), avg_loss


# ---------- Aggregation ----------

def fedavg_aggregate(client_weights):
    new_weights = copy.deepcopy(client_weights[0])
    for k in new_weights:
        new_weights[k] = sum(w[k] for w in client_weights) / len(client_weights)
    return new_weights


def reweighted_fedavg_aggregate(client_weights, client_sizes):
    # Weight each client inversely proportional to its local dataset size
    inv = [1.0 / max(s, 1) for s in client_sizes]
    total = sum(inv)
    w = [v / total for v in inv]

    new_weights = copy.deepcopy(client_weights[0])
    for k in new_weights:
        new_weights[k] = sum(w[i] * client_weights[i][k] for i in range(len(client_weights)))
    return new_weights


def qfedavg_aggregate(client_weights, client_losses, q=5.0, eps=1e-8):
    weights = [(loss + eps) ** q for loss in client_losses]
    total = sum(weights)
    weights = [w / total for w in weights]

    new_weights = copy.deepcopy(client_weights[0])
    for k in new_weights:
        new_weights[k] = sum(
            weights[i] * client_weights[i][k] for i in range(len(client_weights))
        )
    return new_weights


# ---------- Federated training loop ----------

def federated_training(
    global_model,
    clients,
    rounds=10,
    local_epochs=1,
    local_balancing=False,
    use_fedprox=False,
    mu=0.01,
    use_qfedavg=False,
    q=5.0,
    use_reweighting=False
):
    for r in range(rounds):
        print(f"\n--- Round {r+1} ---")

        client_weights = []
        client_losses = []
        client_sizes = []

        for _, client_data in clients.items():
            if len(client_data) == 0:
                continue

            weights, loss = train_local_model(
                global_model,
                client_data,
                epochs=local_epochs,
                local_balancing=local_balancing,
                use_fedprox=use_fedprox,
                mu=mu
            )

            client_weights.append(weights)
            client_losses.append(loss)
            client_sizes.append(len(client_data))

        if use_qfedavg:
            new_weights = qfedavg_aggregate(client_weights, client_losses, q=q)
        elif use_reweighting:
            new_weights = reweighted_fedavg_aggregate(client_weights, client_sizes)
        else:
            new_weights = fedavg_aggregate(client_weights)

        global_model.load_state_dict(new_weights)

    return global_model
