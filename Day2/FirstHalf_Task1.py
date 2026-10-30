import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(0)
tokens = np.random.rand(4, 3)
print("Input Tokens:")
print(tokens)
Wq = np.random.rand(3, 3)
Wk = np.random.rand(3, 3)
Wv = np.random.rand(3, 3)
Q = tokens @ Wq
K = tokens @ Wk
V = tokens @ Wv
print("\nQuery (Q):")
print(Q)
print("\nKey (K):")
print(K)
print("\nValue (V):")
print(V)
d = 3
scores = Q @ K.T / np.sqrt(d)
print("\nAttention Scores:")
print(scores)
def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

attention_weights = softmax(scores)
print("\nAttention Weight Matrix (4x4):")
print(attention_weights)
print("\nRow Sums:")
print(attention_weights.sum(axis=1))
output = attention_weights @ V
print("\nFinal Self Attention Output:")
print(output)
labels = ["Token 1", "Token 2", "Token 3", "Token 4"]

plt.figure(figsize=(6, 5))
sns.heatmap(
    attention_weights,
    annot=True,
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    fmt=".3f"
)
plt.title("Self Attention Heatmap")
plt.xlabel("Key Tokens")
plt.ylabel("Query Tokens")
plt.show()
