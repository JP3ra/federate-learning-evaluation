# # metrics/evaluation.py

# import torch

# def evaluate_model(model, client_data):
#     """
#     Evaluates model accuracy on one client's data.
#     """

#     model.eval()
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for x, y in client_data:
#             x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
#             y = torch.tensor(y).unsqueeze(0)

#             pred = model(x).argmax(dim=1)
#             correct += (pred == y).sum().item()
#             total += 1

#     return correct / total if total > 0 else 0.0


import torch
import torch.nn.functional as F

def evaluate_clients(model, clients):
    """
    Evaluates model on each client separately.
    Returns:
        mean_client_accuracy
        worst_client_accuracy
    """

    model.eval()
    client_accuracies = []

    with torch.no_grad():
        for client_id, data in clients.items():
            correct = 0
            total = 0

            for x, y in data:
                x = x.unsqueeze(0)  # [1, C, H, W]
                logits = model(x)
                pred = logits.argmax(dim=1).item()

                correct += int(pred == y)
                total += 1

            acc = correct / total if total > 0 else 0.0
            client_accuracies.append(acc)

    mean_acc = sum(client_accuracies) / len(client_accuracies)
    worst_acc = min(client_accuracies)

    return mean_acc, worst_acc
