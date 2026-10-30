import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

attention_weights = np.array([
    [0.24448753, 0.21731914, 0.34565167, 0.19254166],
    [0.24416949, 0.21713576, 0.34339612, 0.19529862],
    [0.24123678, 0.20855089, 0.37420330, 0.17600903],
    [0.24667263, 0.22385183, 0.33008148, 0.19939406]
])
tokens = ["Token 1", "Token 2", "Token 3", "Token 4"]

print("Attention Weight Matrix:")
print(attention_weights)
print("\nRow Sums:")
print(attention_weights.sum(axis=1))

plt.figure(figsize=(7, 5))
sns.heatmap(
    attention_weights,
    annot=True,
    cmap="Blues",
    xticklabels=tokens,
    yticklabels=tokens,
    fmt=".3f",
    linewidths=0.5
)
plt.title("Self-Attention Weight Heatmap")
plt.xlabel("Key Tokens")
plt.ylabel("Query Tokens")
plt.tight_layout()
plt.show()
