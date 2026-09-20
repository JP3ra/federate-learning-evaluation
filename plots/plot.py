# plots/plot_results.py

import matplotlib.pyplot as plt

def plot_accuracy_vs_alpha(results):
    """
    results: list of dicts with keys:
        'alpha', 'mean_acc', 'worst_acc'
    """

    alphas = [r["alpha"] for r in results]
    mean_acc = [r["mean_acc"] for r in results]
    worst_acc = [r["worst_acc"] for r in results]

    # Plot Mean Accuracy
    plt.figure()
    plt.plot(alphas, mean_acc, marker='o')
    plt.xscale("log")
    plt.xlabel("Dirichlet Alpha (log scale)")
    plt.ylabel("Mean Client Accuracy")
    plt.title("Mean Client Accuracy vs Data Heterogeneity")
    plt.grid(True)
    plt.show()

    # Plot Worst-Client Accuracy
    plt.figure()
    plt.plot(alphas, worst_acc, marker='o')
    plt.xscale("log")
    plt.xlabel("Dirichlet Alpha (log scale)")
    plt.ylabel("Worst Client Accuracy")
    plt.title("Worst-Client Accuracy vs Data Heterogeneity")
    plt.grid(True)
    plt.show()


results = [
    {"alpha": 10.0, "mean_acc": 0.690, "worst_acc": 0.643},
    {"alpha": 1.0,  "mean_acc": 0.682, "worst_acc": 0.474},
    {"alpha": 0.5,  "mean_acc": 0.711, "worst_acc": 0.485},
    {"alpha": 0.1,  "mean_acc": 0.706, "worst_acc": 0.278},
]

plot_accuracy_vs_alpha(results)