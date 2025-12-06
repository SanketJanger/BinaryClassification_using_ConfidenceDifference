import matplotlib.pyplot as plt

# Small-data supervised vs ConfDiff-ABS 

# Measured supervised accuracies from supervised_small_mnist.py
num_labels = [1000, 3000, 5000]
acc_supervised = [0.9151, 0.9453, 0.9587]

# ConfDiff-ABS MNIST accuracy (from replicated ConfDiff experiment)
confdiff_abs_acc = 0.965  

def plot_h3():
    plt.figure(figsize=(8, 5))
    plt.plot(num_labels, acc_supervised, marker="o", label="Supervised MLP")
    plt.axhline(
        y=confdiff_abs_acc,
        linestyle="--",
        label=f"ConfDiff-ABS (0 labels, ≈{confdiff_abs_acc:.3f})"
    )

    plt.xlabel("Number of labeled training samples")
    plt.ylabel("Test accuracy")
    plt.title("Supervised accuracy vs number of labels (MNIST even vs odd)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("small_data_vs_confdiff.png", dpi=300)
    print("Saved: mall_data_vs_confdiff.png")


# Supervised label noise vs ConfDiff-ABS

noise_levels = [0.0, 0.2, 0.4]
acc_noise = [0.9817, 0.9773, 0.9439]  # from supervised_label_noise_mnist.py


def plot_h4():
    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, acc_noise, marker="o", label="Supervised MLP")
    plt.axhline(
        y=confdiff_abs_acc,
        linestyle="--",
        label=f"ConfDiff-ABS (no labels, ≈{confdiff_abs_acc:.3f})"
    )

    plt.xlabel("Label noise rate")
    plt.ylabel("Test accuracy")
    plt.title("Supervised accuracy vs label noise (MNIST even vs odd)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("noise_vs_supervised.png", dpi=300)
    print("Saved: noise_vs_supervised.png")


if __name__ == "__main__":
    plot_h3()
    plot_h4()
