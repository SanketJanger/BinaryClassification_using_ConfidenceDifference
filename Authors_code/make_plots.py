import numpy as np
import matplotlib.pyplot as plt


# All priors used on MNIST
priors = np.array([0.2, 0.5, 0.8])

# ConfDiff-Unbiased 
unbiased_mean = np.array([
    0.8063355684280396,  # prior 0.2
    0.9365949751281738,  # prior 0.5
    0.7710280418395996   # prior 0.8
])
unbiased_std = np.array([
    0.05686914548277855,
    0.017518091946840286,
    0.02150632998373985
])

# ConfDiff-ReLU 
relu_mean = np.array([
    0.9666603207588196,  # prior 0.2
    0.9446548223495483,  # prior 0.5
    0.9727692604064941   # prior 0.8
])
relu_std = np.array([
    0.004748090170323849,
    0.00416369317099452,
    0.003395653096958995
])

# ConfDiff-ABS 
abs_mean = np.array([
    0.9778777360916138,  # prior 0.2
    0.9651283222579956,  # prior 0.5
    0.9834172129631042   # prior 0.8
])
abs_std = np.array([
    0.001995522995480824,
    0.001071015722118318,
    0.0024721685331314802
])

# Pendigits for ConfDiff-ABS, prior = 0.5
pendigits_abs_mean = 0.9899908304214478
pendigits_abs_std = 0.002212896477431059

plt.figure()
x = priors

plt.errorbar(x, unbiased_mean, yerr=unbiased_std, marker='o', label='ConfDiff-Unbiased')
plt.errorbar(x, relu_mean,     yerr=relu_std,     marker='o', label='ConfDiff-ReLU')
plt.errorbar(x, abs_mean,      yerr=abs_std,      marker='o', label='ConfDiff-ABS')

plt.xlabel("Class prior π⁺")
plt.ylabel("Test accuracy")
plt.title("MNIST – ConfDiff Accuracy vs Class Prior")
plt.xticks([0.2, 0.5, 0.8], ["0.2", "0.5", "0.8"])
plt.ylim(0.7, 1.0)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("mnist_confdiff_accuracy.png", dpi=300)
plt.close()


labels = ["MNIST π=0.2", "MNIST π=0.5", "MNIST π=0.8", "Pendigits π=0.5"]
means = [abs_mean[0], abs_mean[1], abs_mean[2], pendigits_abs_mean]
stds  = [abs_std[0],  abs_std[1],  abs_std[2],  pendigits_abs_std]

plt.figure()
x_pos = np.arange(len(labels))

plt.bar(x_pos, means, yerr=stds)
plt.xticks(x_pos, labels, rotation=20)
plt.ylabel("Test accuracy")
plt.title("ConfDiff-ABS – MNIST vs Pendigits")
plt.ylim(0.9, 1.0)
plt.tight_layout()
plt.savefig("confdiff_abs_mnist_pendigits.png", dpi=300)
plt.close()

print("Saved plots:")
print(" - mnist_confdiff_accuracy.png")
print(" - confdiff_abs_mnist_pendigits.png")