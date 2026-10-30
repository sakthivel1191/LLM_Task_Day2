import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel

print("\nMULTI-HEAD ATTENTION")
np.random.seed(0)
tokens = np.random.rand(4, 4)
print("\nInput Tokens:")
print(tokens)
Wq = np.random.rand(4, 4)
Wk = np.random.rand(4, 4)
Wv = np.random.rand(4, 4)
Wo = np.random.rand(4, 4)

Q = tokens @ Wq
K = tokens @ Wk
V = tokens @ Wv

print("\nQuery (Q):")
print(Q)
print("\nKey (K):")
print(K)
print("\nValue (V):")
print(V)

num_heads = 2
head_dim = 2
Q_heads = Q.reshape(4, num_heads, head_dim)
K_heads = K.reshape(4, num_heads, head_dim)
V_heads = V.reshape(4, num_heads, head_dim)

def softmax(x):
    exp_values = np.exp(
        x - np.max(x, axis=1, keepdims=True)
    )
    return exp_values / np.sum(
        exp_values, axis=1, keepdims=True
    )
head_outputs = []

for i in range(num_heads):
    q = Q_heads[:, i, :]
    k = K_heads[:, i, :]
    v = V_heads[:, i, :]
    scores = q @ k.T / np.sqrt(head_dim)
    weights = softmax(scores)
    head_output = weights @ v
    head_outputs.append(head_output)
    print("\nAttention Weights - Head", i + 1)
    print(weights)

concatenated = np.concatenate(head_outputs, axis=1)
print("\nConcatenated Heads:")
print(concatenated)

final_output = concatenated @ Wo
print("\nFinal Multi-Head Attention Output:")
print(final_output)

print("\nHUGGING FACE ATTENTION INSPECTION")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(
    model_name,
    attn_implementation="eager"
)
sentence = "The bank by the river was steep"
inputs = tokenizer(sentence, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
)
print("\nSentence Tokens:")
print(tokens)

model.eval()
with torch.no_grad():
    outputs = model(
        **inputs,
        output_attentions=True
    )
attention = outputs.attentions[0]
head = attention[0, 0]
bank_index = tokens.index("bank")
bank_attention = head[bank_index]
print("\nAttention Weights for the word 'bank':")
for token, weight in zip(tokens, bank_attention):
    print(token, ":", round(weight.item(), 4))

print("\nATTENTION HEATMAP")
plt.figure(figsize=(9, 6))
sns.heatmap(
    head.numpy(),
    annot=True,
    cmap="Blues",
    xticklabels=tokens,
    yticklabels=tokens,
    fmt=".2f"
)
plt.title("BERT Attention Heatmap - First Head")
plt.xlabel("Key Tokens")
plt.ylabel("Query Tokens")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
