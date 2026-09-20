import math

# n = number of seeds
N = 5
Z = 1.96  # 95% confidence

results = {
    "alpha_10.0": {
        "fedavg":  (0.482, 0.009),
        "fedprox": (0.482, 0.009),
        "qfedavg": (0.481, 0.010),
    },
    "alpha_1.0": {
        "fedavg":  (0.422, 0.015),
        "fedprox": (0.420, 0.015),
        "qfedavg": (0.438, 0.016),
    },
    "alpha_0.5": {
        "fedavg":  (0.392, 0.017),
        "fedprox": (0.392, 0.016),
        "qfedavg": (0.355, 0.027),
    }
}

def ci(mean, std):
    margin = Z * std / math.sqrt(N)
    return mean - margin, mean + margin

for alpha, methods in results.items():
    print(f"\n=== {alpha} ===")
    for method, (mean, std) in methods.items():
        low, high = ci(mean, std)
        print(
            f"{method.upper():8s} | "
            f"Mean: {mean:.3f}, Std: {std:.3f}, "
            f"95% CI: [{low:.3f}, {high:.3f}]"
        )
