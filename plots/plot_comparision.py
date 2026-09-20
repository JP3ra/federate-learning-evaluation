import matplotlib.pyplot as plt

def plot_local_data_balancing():
    alphas = [10.0, 1.0, 0.5, 0.1]

    mean_acc = [0.775, 0.808, 0.800, 0.720]
    worst_acc = [0.743, 0.756, 0.742, 0.495]

    plt.figure(figsize=(7, 5))

    plt.plot(
        alphas,
        mean_acc,
        marker='o',
        linestyle='-',
        label="Mean Client Accuracy"
    )

    plt.plot(
        alphas,
        worst_acc,
        marker='s',
        linestyle='--',
        label="Worst-Client Accuracy"
    )

    plt.xscale("log")
    plt.xlabel("Dirichlet Alpha (log scale)")
    plt.ylabel("Accuracy")
    plt.title("Local Data Balancing under Increasing Data Heterogeneity")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_local_data_balancing()
